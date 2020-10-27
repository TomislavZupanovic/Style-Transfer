from src.model import Painter
from src.processing import Images
from src.utils import show_images
import imageio
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, help='Name of content image, needs type too')
parser.add_argument('--style', type=str, help='Name of style image, needs type too')
args = parser.parse_args()

content_path = f'src/images/content/{args.content}'
style_path = f'src/images/style/{args.style}'
target_path = f'src/images/transformed/transformed-{args.content}'


def main():
    if torch.cuda.is_available():
        content_image = Images().load_image(content_path).cuda()
        style_image = Images().load_image(style_path, shape=content_image.shape[-2:]).cuda()
    else:
        content_image = Images().load_image(content_path)
        style_image = Images().load_image(style_path, shape=content_image.shape[-2:])
    painter = Painter()
    painter.define_features(content_image, style_image)
    painter.paint(steps=2000)
    show_images(Images.convert_image(content_image), Images.convert_image(painter.target))
    imageio.imwrite(target_path, Images.convert_image(painter.target))


if __name__ == '__main__':
    main()