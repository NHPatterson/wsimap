import numpy as np
import zarr
from pathlib import Path


class TestWSIzarr:
    def __init__(self, t_dir):
        self.zarr_fp = str(Path(t_dir) / "test.zarr")
        root = zarr.open_group(self.zarr_fp, mode="a")

        pyr_layer = root.create_dataset(
            "0",
            shape=(3, 2 ** 16, 2 ** 16),
            chunks=(3, 1024, 1024),
            dtype=np.uint8,
            overwrite=True,
        )

    def get_zarr_fp(self):
        return self.zarr_fp
