import matplotlib.pyplot as plt
import numpy as np

__all__ = ['dynamic_plt']

def dynamic_plt(imgs: list,figsize=(16,12)):
    """
    Create a dynamic plots based on the number of images
    Args:
        imgs_array: Array of images stacked or Path of Images 
        figsize: size of the figures
    Return:
        None
    """
    if isinstance(imgs[0], str):
        imgs = [plt.imread(i) for i in imgs]

    num_images = len(imgs)

    num_cols = 2 
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, img in enumerate(imgs):
        ax = axes[i // num_cols, i % num_cols]  # Calculate the correct index for subplots
        ax.imshow(img)
        ax.axis('off')

    for j in range(i + 1, len(axes.flatten())):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()