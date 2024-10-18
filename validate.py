import torch
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import glob as gb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from models.shufflenet import MobileNetV2Segmentation
from models import network

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = MobileNetV2Segmentation(n_class=19).to(device)
model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16,pretrained_backbone=False).to(device)

model.load_state_dict(torch.load('Deeplab_checkpoint_epoch_400.pt', map_location=device))
model.eval()

# Custom dataset class
class CustomData(Dataset):
    def __init__(self, images_dir, masks_dir, transform_method):
        self.image_path = sorted(gb.glob(os.path.join(images_dir, '*')))
        self.mask_path = sorted(gb.glob(os.path.join(masks_dir, '*')))
        self.transform = transform_method

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        tensor_image = self.transform(image)
        mask = Image.open(self.mask_path[index])
        tensor_mask = (transforms.ToTensor()(mask) * 255).long().squeeze(0)
        # Return the image name to be used for saving later
        img_name = os.path.basename(self.image_path[index])
        return tensor_image, tensor_mask, img_name

# Define the dataset and dataloader
val_images_dir = 'dataset/test_image'
# val_images_dir = 'dataset/val/val_image'
val_masks_dir = './dataset/val/val_mask'
img_size = 512
# mean = torch.tensor([0.5192, 0.4182, 0.3640])
# std = torch.tensor([0.2684, 0.2415, 0.2340])
basic_transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor()
    # transforms.Normalize(mean=mean, std=std)
])

val_set = CustomData(val_images_dir, val_masks_dir, basic_transform)
val_dataloader = DataLoader(dataset=val_set, batch_size=32, shuffle=False)

# Function to save predicted masks as PNGs
def save_pred_as_png(img_name, images, predictions, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(images)):
        # Convert image tensor to numpy array for saving
        img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
        pred_mask = predictions[i].argmax(0).cpu().numpy()

        # Saving the predicted mask as a PNG with a unique filename
        mask_img = Image.fromarray(pred_mask.astype(np.uint8))
        
        # Save with the original image name but change the extension to .png
        file_name = os.path.splitext(img_name[i])[0] + '.png'
        mask_img.save(os.path.join(save_dir, file_name))

# Inference on the validation set
save_dir = './predictions'

with torch.no_grad():
    for batch_idx, (image, mask, img_name) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        image = image.to(device)
        mask_pred = model(image)
        save_pred_as_png(img_name, image.cpu(), mask_pred, save_dir)

print(f"All predictions saved to {save_dir}")
