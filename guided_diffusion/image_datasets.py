import math
import random
import os
from collections import Counter
from sklearn.model_selection import train_test_split

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import numpy.random as npr
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST, CIFAR10
from torchvision import datasets, transforms
from torch import Tensor


def load_data(
    *,
    dataset,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False, 
    ndata=None,
    shift=False,
    targets_to_shift=[1,2,7],
    shrink_to_proportion=0.01,
    seed=101
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not dataset:
        dataset = 'ImageNet'
    if not data_dir:
        raise ValueError("unspecified data directory")
    if dataset == 'ImageNet':
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
        )
    else:

        if dataset == "CIFAR10":
            # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
            transform_action = []
            if random_crop:
                transform_action.append(transforms.RandomCrop(32, padding=4))
            if random_flip:
                transform_action.append(transforms.RandomHorizontalFlip())
            transform_action.append(transforms.ToTensor())
            transform_action.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    
        
            pytdataset = CIFAR10(data_dir, download=True, train=True, transform=transforms.Compose(transform_action))
        elif dataset == "MNIST":
            transform_action = []
            if random_crop:
                transform_action.append( 
                    transforms.RandomResizedCrop(
                        28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                    )
                )
            if random_flip:
                transform_action.append(transforms.RandomHorizontalFlip())
            transform_action.append(transforms.ToTensor())
            transform_action.append(transforms.Normalize((0.1307,), (0.3081,)))
    
            pytdataset = MNIST(data_dir, download=True, train=True, transform=transforms.Compose(transform_action))
        else:
            raise ValueError(f"Received dataset {dataset}. Only supported 'ImageNet', 'CIFAR10' or 'MNIST'.")

        if ndata is None:
            ndata = len(pytdataset)
        
        ndata = int(ndata)

        indices_saved_filename = '{}_n{}_shift{}_target{}_prop{}_seed{}.npy'.format(
            dataset,
            ndata, 
            str(shift)[0],
            ''.join(map(str, targets_to_shift)),
            shrink_to_proportion,
            seed
        )
        indices_saved_path = os.path.join(data_dir, indices_saved_filename)

        kept_indices = get_data_indices(
            pytdataset,
            ndata, 
            shift,
            targets_to_shift,
            shrink_to_proportion,
            indices_saved_path=indices_saved_path,
            seed=seed
        ) 
        dataset = PytorchDatset(Subset(pytdataset, kept_indices), class_cond) 
        
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
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class PytorchDatset(Dataset):
    def __init__(self, dataset:Dataset, class_cond:bool=False):
        super().__init__() 
        self.dataset    = dataset
        self.class_cond = class_cond

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Retrieve PIL image and label from the original dataset
        pil_image, label = self.dataset[idx]
        # Convert PIL Image to a NumPy array
        np_image = np.array(pil_image)

        out_dict = {}
        if isinstance(label, Tensor):
            label = label.item()
        if self.class_cond:
            out_dict["y"] = np.array(label, dtype=np.int64)
        return np_image, out_dict  # or (np_image, np_label) if converting label to NumPy array


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


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


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def get_data_indices( 
    dataset              : datasets,  
    ndata                : int=None,
    shift                : bool=False, 
    targets_to_shift     : list=[1,2,7],
    shrink_to_proportion : float = 0.2,
    indices_saved_path   : str = None,
    seed                 : int = 101) -> np.ndarray:
    
    if ndata is None:
        ndata = len(dataset)

    
    if shift:
        if not os.path.isfile(indices_saved_path): 
            print(f"Could not find {indices_saved_path} for covariates shift.")
            print(f"Creating {indices_saved_path} ...")

            generator = npr.default_rng(seed)
            ## only keep shrink_to_proportion of data points if their labels are in targets_to_shift 
            ind_indices = np.full(len(dataset), True)
            
            for target_curr in targets_to_shift:
                indices_curr = [i for i, (_, target) in enumerate(dataset) if target == target_curr]
                # number of indices in the current class
                n_curr = len(indices_curr)
                # number of indices to be leftout
                n_to_leftout = int(n_curr*(1-shrink_to_proportion))
                # choice the indices to be leftout
                sub_lefout = generator.choice(n_curr, n_to_leftout, replace=False)
                # note down the indices to be leftout
                ind_indices[np.array(indices_curr)[sub_lefout]] = False

            subindices_to_keep = generator.choice(sum(ind_indices), 
                                                  min(ndata, sum(ind_indices)), 
                                                  replace=False)
            kept_indices = np.where(ind_indices)[0][subindices_to_keep]
            
             
            if isinstance(dataset.targets[0], Tensor):
                label_counts = Counter([dataset.targets[i].item() for i in kept_indices]) 
            else:
                label_counts = Counter([dataset.targets[i] for i in kept_indices]) 
            
            save_dict = {
                'label_counts': label_counts,
                'indices'     : kept_indices, 
            }
            np.save(indices_saved_path, save_dict)
            print(f"Saved the indices for covariates shift in '{indices_saved_path}'.")
        else:
            print(f"Found {indices_saved_path} for covariates shift.")
            print(f"Retrieving results from {indices_saved_path} ...")
            loaded_dict = np.load(indices_saved_path, allow_pickle=True).item()
            kept_indices = loaded_dict['indices']
            label_counts = loaded_dict['label_counts'] 
    else:
        kept_indices = np.arange(len(dataset))
        if ndata < len(dataset): 
            if not os.path.isfile(indices_saved_path): 
                print(f"Could not find {indices_saved_path} for dataset of size {ndata} (out of {len(dataset)}).")
                print(f"Creating {indices_saved_path} ...")
                kept_indices, _ = train_test_split(kept_indices, 
                                                   train_size=ndata,  
                                                   random_state=seed)
            
                if isinstance(dataset.targets[0], Tensor):
                    label_counts = Counter([dataset.targets[i].item() for i in kept_indices]) 
                else:
                    label_counts = Counter([dataset.targets[i] for i in kept_indices]) 

                save_dict = {
                    'label_counts': label_counts,
                    'indices'     : kept_indices, 
                }
                np.save(indices_saved_path, save_dict)
                print(f"Saved the indices for dataset of size {ndata} (out of {len(dataset)}) in '{indices_saved_path}'.")
            else:
                print(f"Found {indices_saved_path} for dataset of size {ndata} (out of {len(dataset)}).")
                print(f"Retrieving results from {indices_saved_path} ...")
                loaded_dict = np.load(indices_saved_path, allow_pickle=True).item()
                kept_indices = loaded_dict['indices']
                label_counts = loaded_dict['label_counts'] 

        else:
            if isinstance(dataset.targets[0], Tensor):
                label_counts = Counter([dataset.targets[i].item() for i in kept_indices]) 
            else:
                label_counts = Counter([dataset.targets[i] for i in kept_indices]) 

    print("label_counts:", dict(label_counts)) 
    return kept_indices 
