import numpy as np
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms


class Images:
    @staticmethod
    def load_image(path, max_size=400, shape=None):
        """
        Used for loading content and style images and
        converting them to Tensor
        """

        if 'http' in path:
            response = requests.get(path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(path).convert('RGB')
        """ Check image size """
        if max(image.size) > max_size:
            size = max_size
        else:
            size = image.size

        if shape is not None:
            size = shape
        """ Transform image """
        input_transform = transforms.Compose([transforms.Resize(size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406),
                                                                   (0.229, 0.224, 0.225))])
        image = input_transform(image)[:3, :, :].unsqueeze(0)
        return image

    @staticmethod
    def convert_image(tensor):
        """ Converting image from Tensor to numpy array for plotting """
        image = tensor.to('cpu').clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        """ Un-normalize """
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        return image.clip(0, 1)


