import torch
import torch.optim as optim
from torchvision import models


class Painter(object):
    def __init__(self):
        """ Instantiate trained VGG19 network and its attributes """
        self.style_weights = None
        self.optimizer = None
        self.target = None
        self.style_grams = None
        self.content_features = None
        self.style_features = None
        self.model = models.vgg19(pretrained=True).features
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if torch.cuda.is_available():
            print('GPU detected, using GPU.')
            self.model.cuda()
        else:
            print('No GPU detected, using CPU.')

    @staticmethod
    def get_features(image, model):
        """ Using specific convolution layers of VGG19 network to get content and style features """
        # conv4_2 layer represents content
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
        features = []
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    @staticmethod
    def gram_matrix(tensor):
        """ Computes gram matrix for given feature """
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h*w)
        return torch.mm(tensor, tensor.t())

    def define_features(self, content_img, style_img):
        """ Gets features for given images as content and style features for target image """
        self.content_features = Painter.get_features(content_img, self.model)
        self.style_features = Painter.get_features(style_img, self.model)
        # Computing gram matrices for style image features
        self.style_grams = {layer: Painter.gram_matrix(self.style_features[layer]) for layer in self.style_features}
        if torch.cuda.is_available():
            self.target = content_img.clone().requires_grad_(True).cuda()
        else:
            self.target = content_img.clone().requires_grad_(True)
        """ Setting style weights of style for more style representations from specific layers"""
        self.style_weights = {'conv1_1': 1.,
                              'conv2_1': 0.75,
                              'conv3_1': 0.4,
                              'conv4_1': 0.2,
                              'conv5_1': 0.2}

    def paint(self,  steps, style_weight=1e7):
        """ Paints target content image with style form style image """
        self.optimizer = optim.Adam([self.target], lr=0.003)
        print('STARTED PAINTING...')
        for step in range(1, steps + 1):
            target_features = Painter.get_features(self.target, self.model)
            """Computing content loss between target and content features"""
            content_loss = torch.mean((target_features['conv4_2'] - self.content_features['conv4_2']) ** 2)
            style_loss = 0  # Set style loss to 0 for every step
            for layer in self.style_weights:
                """Feature and gram matrix for the specific layer in style layers"""
                target_feature = target_features[layer]
                target_gram_matrix = Painter.gram_matrix(target_feature)
                batch_size, depth, height, width = target_feature.shape  # We can ignore batch_size
                style_gram_matrix = self.style_grams[layer]
                """ Style loss for given layer """
                layer_style_loss = self.style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
                style_loss += layer_style_loss / (depth * height * width)
            """ Compute total loss and gradients """
            total_loss = 1 * content_loss + style_weight * style_loss  # 1 represents content weight
            self.optimizer.zero_grad()  # Set gradients to zero
            total_loss.backward()  # Compute gradients with backprop
            self.optimizer.step()  # Perform gradient descent
            print(f"\nCompleted epoch: {step}/{steps}")

