import matplotlib.pyplot as plt


def show_images(one_image, two_image=None):
    if two_image is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(one_image)
        ax2.imshow(two_image)
    else:
        plt.imshow(one_image)
    plt.axis('off')
    plt.show()
