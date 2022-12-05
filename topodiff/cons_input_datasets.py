import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, deterministic=True
):
    """
    Create the generator used for sampling.
    The dataset should contain:
    - the .npy files of physical fields in the form cons_pf_array_X.npy,
    - the .npy files of loads in the form cons_load_array_X.npy,
    - the .npy files of boundary conditions in the form cons_bc_array_X.npy.

    :param data_dir: the dataset directory.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_input_constraints, all_input_raw_loads, all_input_raw_BCs = _list_input_files_recursively(data_dir)
    dataset = InputConstraintsDataset(
        all_input_constraints,
        all_input_raw_loads,
        all_input_raw_BCs,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_input_files_recursively(data_dir):
    input_constraints = []
    input_raw_loads = []
    input_raw_BCs = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            if "load" in entry:
                input_raw_loads.append(full_path) # Load file
            elif "bc" in entry:
                input_raw_BCs.append(full_path) # BC file
            else:
                input_constraints.append(full_path) # Physical fields file
        elif bf.isdir(full_path):
            input_constraints.extend(_list_input_files_recursively(full_path))
            input_raw_loads.extend(_list_input_files_recursively(full_path))
            input_raw_BCs.extend(_list_input_files_recursively(full_path))
    return input_constraints, input_raw_loads, input_raw_BCs


class InputConstraintsDataset(Dataset):
    def __init__(self, input_constraints_paths, input_raw_loads_paths, input_raw_BCs_paths, shard=0, num_shards=1):
        super().__init__()
        self.local_input_constraints = input_constraints_paths[shard:][::num_shards]
        self.local_input_raw_loads = input_raw_loads_paths[shard:][::num_shards]
        self.local_input_raw_BCs = input_raw_BCs_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.local_input_constraints)

    def __getitem__(self, idx):
        input_constraints_path = self.local_input_constraints[idx]
        input_raw_loads_path = self.local_input_raw_loads[idx]
        input_raw_BCs_path = self.local_input_raw_BCs[idx]
      
        input_constraints = np.load(input_constraints_path)
        input_raw_loads = np.load(input_raw_loads_path)
        input_raw_BCs = np.load(input_raw_BCs_path)
        return np.transpose(input_constraints, [2, 0, 1]).astype(np.float32), np.transpose(input_raw_loads, [2, 0, 1]).astype(np.float32), np.transpose(input_raw_BCs, [2, 0, 1]).astype(np.float32)