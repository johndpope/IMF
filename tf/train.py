import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import wandb
from omegaconf import OmegaConf
import numpy as np
from tqdm.auto import tqdm
import os

# Assuming these are converted to TensorFlow equivalents
# Make sure to adjust the imports according to your project structure
from model import IMFModel, IMFPatchDiscriminator
from VideoDataset import VideoDataset

# Import LPIPS TensorFlow 2.x implementation
from lpips import lpips

def load_config(config_path):
    return OmegaConf.load(config_path)

def get_video_repeat(epoch, max_epochs, initial_repeat, final_repeat):
    return max(final_repeat, initial_repeat - (initial_repeat - final_repeat) * (epoch / max_epochs))

class IMFTrainer:
    def __init__(self, config, model, discriminator, train_dataset, dataset_size):
        self.config = config
        self.model = model
        self.discriminator = discriminator
        self.train_dataset = train_dataset
        self.dataset_size = dataset_size

        self.pixel_loss_fn = tf.keras.losses.MeanAbsoluteError()

        self.style_mixing_prob = config.training.style_mixing_prob
        self.noise_magnitude = config.training.noise_magnitude
        self.r1_gamma = config.training.r1_gamma

        self.optimizer_g = Adam(learning_rate=config.training.learning_rate_g, beta_1=0.5, beta_2=0.999)
        self.optimizer_d = Adam(learning_rate=config.training.initial_learning_rate_d, beta_1=0.5, beta_2=0.999)

        # Initialize checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            model=self.model,
            discriminator=self.discriminator,
            epoch=tf.Variable(0)
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.config.checkpoints.dir,
            max_to_keep=5
        )

        self.batch_size = config.training.batch_size
        self.total_batches_per_epoch = int(np.ceil(self.dataset_size / self.batch_size))

    @tf.function
    def compute_perceptual_loss(self, x, y):
        # x and y are tensors of shape [batch_size, height, width, channels]
        # LPIPS expects images in the range [-1, 1]
        x_normalized = (x / 127.5) - 1.0
        y_normalized = (y / 127.5) - 1.0

        # Compute LPIPS perceptual loss
        perceptual_loss = lpips(x_normalized, y_normalized)
        # Reduce over spatial dimensions and batch
        perceptual_loss = tf.reduce_mean(perceptual_loss)
        return perceptual_loss

    @tf.function
    def train_step(self, x_current, x_reference):
        with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
            # Pass x_current and x_reference as separate arguments
            x_reconstructed = self.model(x_current, x_reference, training=True)

            # Discriminator loss
            real_output = self.discriminator(x_current, training=True)
            fake_output = self.discriminator(x_reconstructed, training=True)

            d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
            d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
            d_loss = d_loss_real + d_loss_fake

            # Generator loss
            g_loss_gan = -tf.reduce_mean(fake_output)
            l_p = self.pixel_loss_fn(x_current, x_reconstructed)
            l_v = self.compute_perceptual_loss(x_current, x_reconstructed)

            g_loss = (self.config.training.lambda_pixel * l_p +
                      self.config.training.lambda_perceptual * l_v +
                      self.config.training.lambda_adv * g_loss_gan)

        # Compute gradients
        grad_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        grad_g = tape_g.gradient(g_loss, self.model.trainable_variables)

        # Apply gradients
        self.optimizer_d.apply_gradients(zip(grad_d, self.discriminator.trainable_variables))
        self.optimizer_g.apply_gradients(zip(grad_g, self.model.trainable_variables))

        return d_loss, g_loss, l_p, l_v, g_loss_gan, x_reconstructed


    def save_images(self, x_reconstructed, x_current, x_reference, global_step):
        # Assuming images are in the range [0, 1]
        save_dir = self.config.logging.sample_dir
        os.makedirs(save_dir, exist_ok=True)

        # Convert tensors to NumPy arrays
        x_reconstructed_np = x_reconstructed.numpy()
        x_current_np = x_current.numpy()
        x_reference_np = x_reference.numpy()

        # Function to save a single image
        def save_image(img_array, filename):
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
            tf.keras.preprocessing.image.save_img(os.path.join(save_dir, filename), img_array)

        # Save images locally
        save_image(x_reconstructed_np[0], f'x_reconstructed_{global_step}.png')
        save_image(x_current_np[0], f'x_current_{global_step}.png')
        save_image(x_reference_np[0], f'x_reference_{global_step}.png')

        # Log images to wandb
        wandb_images = []
        for i in range(min(x_reconstructed_np.shape[0], 4)):  # Log up to 4 samples
            wandb_images.extend([
                wandb.Image(x_reconstructed_np[i], caption=f"x_reconstructed {i}"),
                wandb.Image(x_current_np[i], caption=f"x_current {i}"),
                wandb.Image(x_reference_np[i], caption=f"x_reference {i}")
            ])

        wandb.log({
            "Sample Reconstructions": wandb_images,
            "global_step": global_step
        })

        print(f"Saved and logged sample reconstructions for global step {global_step}")
        

    def train(self, start_epoch=0):
        global_step = 0
        epoch = start_epoch - 1  # Initialize epoch


        for epoch in range(start_epoch, self.config.training.num_epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")

            # Compute video_repeat
            video_repeat = get_video_repeat(
                epoch,
                self.config.training.num_epochs,
                self.config.training.initial_video_repeat,
                self.config.training.final_video_repeat
            )

            progress_bar = tqdm(total=self.total_batches_per_epoch, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}")

            for batch in self.train_dataset:
                source_frames = batch['frames']
                print("source_frames:",len(source_frames))
                batch_size = tf.shape(source_frames)[0]
                num_frames = tf.shape(source_frames)[1]

                for _ in range(int(video_repeat)):
                    if self.config.training.use_many_xrefs:
                        ref_indices = tf.range(0, num_frames, self.config.training.every_xref_frames)
                    else:
                        ref_indices = [0]

                    for ref_idx in ref_indices:
                        x_reference = source_frames[:, ref_idx]

                        for current_idx in range(num_frames):
                            if current_idx == ref_idx:
                                continue

                            x_current = source_frames[:, current_idx]

                            d_loss, g_loss, l_p, l_v, g_loss_gan, x_reconstructed = self.train_step(x_current, x_reference)

                            # Logging
                            wandb.log({
                                "g_loss": g_loss.numpy(),
                                "d_loss": d_loss.numpy(),
                                "pixel_loss": l_p.numpy(),
                                "perceptual_loss": l_v.numpy(),
                                "gan_loss": g_loss_gan.numpy(),
                                "lr_g": self.optimizer_g._decayed_lr(tf.float32).numpy(),
                                "lr_d": self.optimizer_d._decayed_lr(tf.float32).numpy()
                            })

                            # Sample step for saving images
                            if global_step % self.config.logging.sample_every == 0:
                                self.save_images(x_reconstructed, x_current, x_reference, global_step)

                            # Checkpoint saving
                            if global_step % self.config.training.save_steps == 0:
                                self.save_checkpoint(epoch)

                            global_step += 1

                progress_bar.update(1)
                progress_bar.set_postfix({"G Loss": f"{g_loss.numpy():.4f}", "D Loss": f"{d_loss.numpy():.4f}"})

            progress_bar.close()
            epoch += 1
        # Final model saving
        self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        self.checkpoint.epoch.assign(epoch)
        save_path = self.checkpoint_manager.save()
        print(f"Saved checkpoint for epoch {epoch} at {save_path}")

    def load_checkpoint(self):
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"Restored from {latest_checkpoint}")
            start_epoch = int(self.checkpoint.epoch.numpy()) + 1
            return start_epoch
        else:
            print("No checkpoint found, starting from scratch.")
            return 0

def main():
    config = load_config('../config.yaml')
    wandb.init(project='IMF', config=OmegaConf.to_container(config, resolve=True))

    model = IMFModel(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
    )

    discriminator = IMFPatchDiscriminator()

    # Assuming VideoDataset is adapted for TensorFlow
    dataset = VideoDataset(config.dataset.root_dir,
                           frame_skip=0,
                           num_frames=300)
    
    dataset_size = len(dataset)
    print(f"Dataset length: {dataset_size}")
    
    if dataset_size == 0:
        print("Error: Dataset is empty. Please check the root_dir path and ensure it contains video folders with frames.")
        return

    # Create a data generator function
    def data_generator():
        for i in range(dataset_size):
            data = dataset[i]
            print(f"Yielding data for video {i+1}/{dataset_size}: {data['video_name']}")
            print(f"Frame shape: {data['frames'].shape}")
            yield {'frames': data['frames']}

    train_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types={'frames': tf.float32},
        output_shapes={'frames': tf.TensorShape([None, 3, 256, 256])}  # Shape: (frames, channels, height, width)
    ).batch(config.training.batch_size).prefetch(tf.data.AUTOTUNE)

    # Check if the dataset is empty
    for batch in train_dataset.take(1):
        if 'frames' not in batch or batch['frames'].shape[0] == 0:
            print("Error: No data in the first batch. Please check the dataset and data_generator.")
            return
        print(f"First batch shape: {batch['frames'].shape}")
    
    print("Dataset seems to be properly loaded and contains data.")

    trainer = IMFTrainer(config, model, discriminator, train_dataset, dataset_size)

    if config.training.load_checkpoint:
        start_epoch = trainer.load_checkpoint()
    else:
        start_epoch = 0

    trainer.train(0)

if __name__ == "__main__":
    main()


