# wsimap
Tools for doing analysis on whole slide images (WSIs) converted to chunked zarr arrays and read with dask.
- Uses Shapely library to manage spatial operations
- Prepares the data for tiled analysis: segmentation or other functions

This library has some quality issues and is still a work-in-progress.

# Installation

Clone repo 
Create conda environment
Activate environment

Install pytorch for system

Install detecton2: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

pip installs...

```bash
pip install -r requirements.txt
```

Then you can install the package if in the root directory:
```bash
pip install -e .
```
