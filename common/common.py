import matplotlib
import matplotlib.pyplot as plt


def show_mnist_image(flattened_mnist_image):
    show_flattened_image(flattened_mnist_image, 28, 28)


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
