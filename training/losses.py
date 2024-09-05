import torch
import torch.nn.functional as F

from torch import nn
from criteria import id_loss, moco_loss, id_vit_loss
from criteria.lpips.lpips import LPIPS
from utils.class_registry import ClassRegistry
from configs.paths import DefaultPaths


losses = ClassRegistry()
adv_losses = ClassRegistry()
disc_losses = ClassRegistry()
other_losses = ClassRegistry()


class LossBuilder:
    def __init__(self, enc_losses_dict, disc_losses_dict, device):
        self.coefs_dict = enc_losses_dict
        self.losses_names = [k for k, v in enc_losses_dict.items() if v > 0]
        self.losses = {}
        self.adv_losses = {}
        self.other_losses = {}
        self.device = device

        for loss in self.losses_names:
            if loss in losses.classes.keys():
                self.losses[loss] = losses[loss]().to(self.device).eval()
            elif loss in adv_losses.classes.keys():
                self.adv_losses[loss] = adv_losses[loss]()
            elif loss in other_losses.classes.keys():
                self.other_losses[loss] = other_losses[loss]()
            else:
                raise ValueError(f'Unexepted loss: {loss}')

        self.disc_losses = []
        for loss_name, loss_args in disc_losses_dict.items():
            if loss_args.coef > 0:
                self.disc_losses.append(disc_losses[loss_name](**loss_args))


    def encoder_loss(self, batch_data):
        loss_dict = {}
        global_loss = 0.0

        for loss_name, loss in self.losses.items():
            loss_val = loss(batch_data["y_hat"], batch_data["x"])
            global_loss += self.coefs_dict[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        for loss_name, loss in self.other_losses.items():
            loss_val = loss(batch_data)
            assert torch.isfinite(loss_val)
            global_loss += self.coefs_dict[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        if batch_data["use_adv_loss"]:
            for loss_name, loss in self.adv_losses.items():
                loss_val = loss(batch_data["fake_preds"])
                global_loss += self.coefs_dict[loss_name] * loss_val
                loss_dict[loss_name] = float(loss_val)

        return global_loss, loss_dict

    def disc_loss(self, D, batch_data):
        disc_losses = {}
        total_disc_loss = torch.tensor([0.], device=self.device)

        for loss in self.disc_losses:
            disc_loss, disc_loss_dict = loss(D, batch_data)

            total_disc_loss += disc_loss
            disc_losses.update(disc_loss_dict)

        return total_disc_loss, disc_losses



@losses.add_to_registry(name="l2")
class L2Loss(nn.MSELoss):
    pass


@losses.add_to_registry(name="lpips")
class LPIPSLoss(LPIPS):
    pass


@losses.add_to_registry(name="lpips_scale")
class LPIPSScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = LPIPSLoss()

    def forward(self, x, y):
        out = 0
        for res in [256, 128, 64]:
            x_scale = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            y_scale = F.interpolate(y, size=(res, res), mode="bilinear", align_corners=False)
            out += self.loss_fn.forward(x_scale, y_scale).mean()
        return out


@other_losses.add_to_registry(name="feat_rec")
class FeatReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.loss_fn(batch["feat_recon"], batch["feat_real"]).mean()


@other_losses.add_to_registry(name="feat_rec_l1")
class FeatReconL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, batch):
        return self.loss_fn(batch["feat_recon"], batch["feat_real"]).mean()



@other_losses.add_to_registry(name="l2_latent")
class LatentMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.loss_fn(batch["latent"], batch["latent_rec"]).mean()
        


@losses.add_to_registry(name="id")
class IDLoss(id_loss.IDLoss):
    pass


@losses.add_to_registry(name="id_vit")
class IDVitLoss(id_vit_loss.IDVitLoss):
    pass


@losses.add_to_registry(name="moco")
class MocoLoss(moco_loss.MocoLoss):
    pass


@adv_losses.add_to_registry(name="adv")
class EncoderAdvLoss:
    def __call__(self, fake_preds):
        loss_G_adv = F.softplus(-fake_preds).mean()
        return loss_G_adv


@disc_losses.add_to_registry(name="main")
class AdvLoss:
    def __init__(self, coef=0.0):
        self.coef = coef

    def __call__(self, disc, loss_input):
        real_images = loss_input["x"].detach()
        generated_images = loss_input["y_hat"].detach()
        loss_dict = {}

        fake_preds = disc(generated_images, None)
        real_preds = disc(real_images, None)
        loss = self.d_logistic_loss(real_preds, fake_preds)
        loss_dict["disc/main_loss"] = float(loss)

        return loss, loss_dict

    def d_logistic_loss(self, real_preds, fake_preds):
        real_loss = F.softplus(-real_preds)
        fake_loss = F.softplus(fake_preds)

        return (real_loss.mean() + fake_loss.mean()) / 2


@disc_losses.add_to_registry(name="r1")
class R1Loss:
    def __init__(self, coef=0.0, hyper_d_reg_every=16):
        self.coef = coef
        self.hyper_d_reg_every = hyper_d_reg_every

    def __call__(self, disc, loss_input):
        real_images = loss_input["x"]
        step = loss_input["step"]
        if step % self.hyper_d_reg_every != 0:  # use r1 only once per 'hyper_d_reg_every' steps
            return torch.tensor([0.], requires_grad=True, device='cuda'), {}

        real_images.requires_grad = True
        loss_dict = {}

        real_preds = disc(real_images, None)
        real_preds = real_preds.view(real_images.size(0), -1)
        real_preds = real_preds.mean(dim=1).unsqueeze(1)
        r1_loss = self.d_r1_loss(real_preds, real_images)

        loss_D_R1 = self.coef / 2 * r1_loss * self.hyper_d_reg_every + 0 * real_preds[0]
        loss_dict["disc/r1_reg"] = float(loss_D_R1)
        return loss_D_R1, loss_dict

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty
