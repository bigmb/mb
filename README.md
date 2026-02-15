# mb

Small collection of utility modules under the `mb.*` namespace.

- Base package (this repo): `mb_base`
- Extra functionality is split into separate “linked” packages (install only what you need).

## Install

```bash
pip install mb_base
# or
uv pip install mb_base
```

Upgrade:

```bash
pip install -U mb_base
```

## Linked packages

Base modules provided by `mb_base` live under `mb.*`.

Other packages extend the same namespace:

| Python module | PyPI package |
| --- | --- |
| `mb.pandas` | `mb_pandas` |
| `mb.rag` | `mb_rag` |
| `mb.llm` | `mb_llm` |
| `mb.sql` | `mb_sql` |
| `mb.utils` | `mb_utils` |
| `mb.plt` | `mb_base` |
| `mb.ffmpeg` | `mb_ffmpeg` |
| `mb.pytorch` | `mb_pytorch` |
| `mb.yolo` | `mb_yolo` |

## Extra functions

### `dynamic_plt`

Plot $n$ images in a grid with optional titles and bounding boxes.

```python
from mb.plt.utils import dynamic_plt

# imgs can be numpy arrays (H, W, C) or file paths

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
- `imgs`: list of images (numpy arrays or file paths)
- `labels`: list of title labels per image
- `bboxes`: list of bounding boxes per image `[[x, y, w, h], ...]`
- `bboxes_label`: list of labels per bounding box
- `num_cols`: number of columns (default: `2`)
- `figsize`: figure size (default: `(16, 12)`)
- `return_fig`: return the `matplotlib` figure object
- `save_path`: path to save the plot
- `max_workers`: threads for loading images (default: `4`)
