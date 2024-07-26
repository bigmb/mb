import matplotlib.pyplot as plt
import numpy as np

__all__ = ['dynamic_plt']

def dynamic_plt(imgs: list,labels: list =None, bboxes: list =None ,num_cols: int = 2, figsize=(16, 12), return_fig: bool = False, show: bool = True):
    """
    Create dynamic plots based on the number of images and desired columns
    Args:
        imgs: List of images or paths to images
        labels: List of labels corresponding to the images (default: None)
        bboxes: List of bounding boxes corresponding to the images (default: None)
        num_cols: Number of columns for the subplot grid (default: 2)
        figsize: Size of the figure (default: (16, 12))
        return_fig: Return the figure object (default: False)
        show: Show the plot (default: True)
    Return:
        None
    """
    if isinstance(imgs[0], str):
        imgs = [plt.imread(i) for i in imgs]
    
    num_images = len(imgs)
    num_rows = int(np.ceil(num_images / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Ensure axes is always 2D
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, img in enumerate(imgs):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        if img.shape[0]==3:
            img = np.moveaxis(img, 0, -1)
        ax.imshow(img)
        ax.axis('off')
        if labels:
            ax.set_title(str(labels[i]))

        if bboxes:
            img_bboxes = bboxes[i]
            for bbox in img_bboxes:
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                     fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

    # Remove any unused subplots
    for j in range(num_images, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    if show:    
        plt.show()
    if return_fig:
        return fig