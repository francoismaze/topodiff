import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False
):
    """
    Create the generator used for training the regressor predicting the compliance.
    The dataset should contain:
    - the .npy files of physical fields in the form cons_pf_array_X.npy,
    - the .npy files of loads in the form cons_load_array_X.npy,
    - the .npy files of boundary conditions in the form cons_bc_array_X.npy,
    - the .png images of the topologies in the form gt_topo_X.png.

    :param data_dir: the dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size of the images.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_images, all_constraints_pf, all_loads, all_bcs, all_deflections = _list_image_files_recursively(data_dir)
    deflections = np.load(all_deflections)
    
    dataset = ImageDataset(
        image_size,
        all_images,
        all_constraints_pf,
        all_loads,
        all_bcs,
        deflections=deflections,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    images = []
    constraints_pf = []
    loads = []
    bcs = []
    deflections = ""
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            images.append(full_path)
        elif "." in entry and ext.lower() in ["npy"]:
            if entry == "deflections_scaled_diff.npy":
                deflections = deflections + full_path
            elif "load" in entry:
                loads.append(full_path)
            elif "bc" in entry:
                bcs.append(full_path)
            else:
                constraints_pf.append(full_path)
        elif bf.isdir(full_path):
            images.extend(_list_image_files_recursively(full_path))
            loads.extend(_list_image_files_recursively(full_path))
            constraints_pf.extend(_list_image_files_recursively(full_path))
    return images, constraints_pf, loads, bcs, deflections


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        constraint_pf_paths,
        loads_paths,
        bcs_paths,
        deflections,
        shard=0,
        num_shards=1
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_constraints_pf = constraint_pf_paths[shard:][::num_shards]
        self.local_loads = loads_paths[shard:][::num_shards]
        self.local_bcs = bcs_paths[shard:][::num_shards]
        self.local_deflections = deflections[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        image_path = self.local_images[idx]
        constraint_pf_path = self.local_constraints_pf[idx]
        load_path = self.local_loads[idx]
        bc_path = self.local_bcs[idx]
        num_im = int((image_path.split("_")[-1]).split(".")[0])
        num_cons_pf = int((constraint_pf_path.split("_")[-1]).split(".")[0])
        num_load = int((load_path.split("_")[-1]).split(".")[0])
        num_bc = int((bc_path.split("_")[-1]).split(".")[0])
        assert num_im == num_cons_pf == num_load == num_bc, "Problem while loading the images and constraints"

        with bf.BlobFile(image_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        arr = center_crop_arr(pil_image, self.resolution)

        arr = np.mean(arr, axis = 2)
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = arr.reshape(self.resolution, self.resolution, 1)

        constraints_pf = np.load(constraint_pf_path)
        assert constraints_pf.shape[0:2] == arr.shape[0:2], "The constraints do not fit the dimension of the image"

        loads = np.load(load_path)
        assert loads.shape[0:2] == arr.shape[0:2], "The constraints do not fit the dimension of the image"

        bcs = np.load(bc_path)
        assert bcs.shape[0:2] == arr.shape[0:2], "The constraints do not fit the dimension of the image"

        constraints = np.concatenate([constraints_pf, loads, bcs], axis = 2)

        out_dict = {}
        out_dict["d"] = np.array(self.local_deflections[num_im], dtype=np.float32)
        return np.transpose(arr, [2, 0, 1]).astype(np.float32), np.transpose(constraints, [2, 0, 1]).astype(np.float32), out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]