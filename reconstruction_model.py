import torch
import torch.nn as nn
import torch.nn.functional as F
from model import IMFModel
from modules.model import Vgg19, ImagePyramide, Transform

class IMFReconstructionModel(nn.Module):
    def __init__(self, imf_model, train_params):
        super(IMFReconstructionModel, self).__init__()
        self.imf_model = imf_model
        self.train_params = train_params
        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, imf_model.num_channels)
        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()

    def forward(self, x):
        source_frame = x['source']
        driving_frame = x['driving']

        generated = self.imf_model(source_frame, driving_frame)
        
        loss_values = {}

        pyramide_real = self.pyramid(driving_frame)
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        if self.loss_weights['l1'] != 0:
            l1_loss = torch.abs(driving_frame - generated['prediction']).mean()
            loss_values['l1'] = self.loss_weights['l1'] * l1_loss

        if self.loss_weights['gan'] != 0:
            # Implement GAN loss if needed
            pass

        return loss_values, generated