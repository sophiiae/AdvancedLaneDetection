import matplotlib.pyplot as plt

# plot single image
def plot1(im, title, colormap):
    plt.imshow(im, cmap=colormap)
    plt.title(title)
    plt.show()

# plot 2 images with defined colormap
def plot2(im_array=[]):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 16))

    ax1.imshow(im_array[0][0], cmap=im_array[0][2])
    ax1.set_title(im_array[0][1])

    ax2.imshow(im_array[1][0], cmap=im_array[1][2])
    ax2.set_title(im_array[1][1])
    plt.show()

# plot four gray images
def plot4(im_array=[]):
    plt.subplots(figsize=(14, 8))

    plt.subplot(221)
    plt.imshow(im_array[0][0], cmap='gray')
    plt.title(im_array[0][1])

    plt.subplot(222)
    plt.imshow(im_array[1][0], cmap='gray')
    plt.title(im_array[1][1])

    plt.subplot(223)
    plt.imshow(im_array[2][0], cmap='gray')
    plt.title(im_array[2][1])

    plt.subplot(224)
    plt.imshow(im_array[3][0], cmap='gray')
    plt.title(im_array[3][1])

    plt.show()

# plot graph
def plot(g):
    plt.plot(g)
    plt.show() 