# mb
meta package which has basic package and extra functions


# install

    pip install mb_base
    or 
    uv pip install mb_base

# Extra function

### dynamic_plt

Plot n images in a grid with optional labels and bounding boxes.

```python
from mb.plt.utils import dynamic_plt

# basic usage
dynamic_plt(imgs)

# with labels and 3 columns
dynamic_plt(imgs, labels=labels, num_cols=3)

# with bounding boxes and labels
dynamic_plt(imgs, labels=labels, bboxes=bboxes, bboxes_label=bboxes_label)

# save to file
dynamic_plt(imgs, save_path="output.png", show=False)
```

**Parameters:**
- `imgs` — list of images (numpy arrays or file paths)
- `labels` — list of title labels per image
- `bboxes` — list of bounding boxes per image `[[x, y, w, h], ...]`
- `bboxes_label` — list of labels per bounding box
- `num_cols` — number of columns (default: 2)
- `figsize` — figure size (default: (16, 12))
- `return_fig` — return the figure object
- `save_path` — path to save the plot
- `max_workers` — threads for loading images (default: 4)
