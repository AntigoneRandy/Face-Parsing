import os
import glob as gb
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch
class InvalidDatasetException(Exception):
    def __init__(self, len_of_paths, len_of_labels):
        super().__init__(
            f"Number of paths ({len_of_paths}) is not compatible with number of labels ({len_of_labels})"
        )

class CustomData(Dataset):
    def __init__(self, images_dir, masks_dir, transform_method):
        self.image_path = sorted(gb.glob(os.path.join(images_dir, '*')))
        self.mask_path = sorted(gb.glob(os.path.join(masks_dir, '*')))
        self.transform = transform_method
        if len(self.image_path) != len(self.mask_path):
            raise InvalidDatasetException(len(self.image_path), len(self.mask_path))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        tensor_image = self.transform(image)

        mask = Image.open(self.mask_path[index])
        tensor_mask = (transforms.ToTensor()(mask) * 255).long().squeeze(0)

        return tensor_image, tensor_mask

def walk_through_data(dir_path):
    """Helper function to print directory structure."""
    for dirpath, dirnames, filenames in tqdm(os.walk(dir_path)):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}")

def get_file_extensions(directory):
    """Helper function to get file extensions in a directory."""
    extensions = []
    for path in tqdm(os.listdir(directory)):
        if os.path.isfile(os.path.join(directory, path)):
            extensions.append(os.path.splitext(path)[1])
    return extensions

def create_transform(img_size):
    """Helper function to create transform pipeline."""
    return transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor()
    ])

def create_dynamic_transform(epoch, image_size):
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    
    # if epoch >= 15:
    #     augmentation_transforms = [
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     ]
    #     return transforms.Compose(augmentation_transforms + base_transforms)
    
    return transforms.Compose(base_transforms)