import torch
import torch.optim as optim
from torchvision import models


class Painter(object):
    def __init__(self):
        self.model = models.vgg19(pretrained=True).features
        self.layers = None
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def get_features(self, image):
        if self.layers is None:
            self.layers = {'0': 'conv1_1',
                           '5': 'conv2_1',
                           '10': 'conv3_1',
                           '19': 'conv4_1',
                           '21': 'conv4_2',
                           '28': 'conv5_1'}
        features = []
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

    @staticmethod
    def gram_matrix(tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h*w)
        return torch.mm(tensor, tensor.t())
