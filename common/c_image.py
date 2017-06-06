import matplotlib
import matplotlib.pyplot as plt

MNIST_IMAGE_WIDTH_AND_HEIGHT = 28


def show_mnist_image(flattened_mnist_image):

    show_flattened_image(flattened_mnist_image,
                         MNIST_IMAGE_WIDTH_AND_HEIGHT,
                         MNIST_IMAGE_WIDTH_AND_HEIGHT)


def show_flattened_image(flattened_image,
                         width,
                         height):

    image = flattened_image.reshape(width, height)
    show_image(image)


def show_image(image):

    plt.imshow(image,
               cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()
