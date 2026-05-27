import torch
import torch.nn as nn

from .resnet import *
from .unet import *


class IDM(nn.Module):

    def __init__(self, model_name, *args, **kwargs):
        super(IDM, self).__init__()
        match model_name:
            case "mask":
                self.model = Mask(*args, **kwargs)
            case "unet":
                self.model = Unet(*args, **kwargs)
            case "resnet":
                self.model = ResNet(*args, **kwargs)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")
        train_mean = torch.tensor([-0.26866713, 0.83559588, 0.69520934, -0.29099351, 0.18849116, -0.01014598, 1.41953145, 0.35073715, 1.05651613, 0.8930193, -0.37493264, -0.18510782, -0.0272574, 1.35274259])
        train_std = torch.tensor([0.25945241, 0.65903812, 0.52147207, 0.42150272, 0.32029947, 0.28452226, 1.78270006, 0.29091741, 0.67675932, 0.58250554, 0.42399049, 0.28697442, 0.31100304, 1.67651926])
        self.register_buffer("train_mean", train_mean)
        self.register_buffer("train_std", train_std)

    def normalize(self, x):
        x = (x - self.train_mean) / self.train_std
        return x

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return output[0] * self.train_std + self.train_mean, *output[1:]
        else:
            return output * self.train_std + self.train_mean


class Unet(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        self.mask_model = UNet(3, 3)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, *args, **kwargs):
        outputs = self.resnet_model(torch.tanh(self.mask_model(images)))
        return outputs


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        else:
            return outputs
