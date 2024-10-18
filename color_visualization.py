import torch
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import glob as gb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from dataset_utils import InvalidDatasetException
from models.shufflenet import MobileNetV2Segmentation
from models import network  

# Load the trained model
device = "cuda:9" if torch.cuda.is_available() else "cpu"
model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16,pretrained_backbone=False).to(device)

model.load_state_dict(torch.load('Deeplab_checkpoint_epoch_490.pt', map_location=device))
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
        tensor_mask = (self.transform(mask) * 255).long().squeeze(0)
        img_name = os.path.basename(self.image_path[index])
        return tensor_image, tensor_mask, img_name

# Define the dataset and dataloader
# val_images_dir = './dataset/train/train_image'
val_images_dir = './dataset/val/val_image'
val_masks_dir = './dataset/val/val_mask'
img_size = 512

basic_transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor()
])

val_set = CustomData(val_images_dir, val_masks_dir, basic_transform)
val_dataloader = DataLoader(dataset=val_set, batch_size=32, shuffle=False)

# Color mapping function for visualization
def labelcolormap(N):
    cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0), (204, 204, 0),
                     (51, 51, 255), (204, 0, 204),  (51, 255, 255), (247, 206, 205),
                     (102, 51, 0), (255, 0, 0), (102, 204, 0), (255, 255, 0), 
                     (0, 0, 153), (0, 0, 204), (255, 51, 153), (0, 204, 204),
                     (0, 51, 0), (255, 153, 51), (0, 204, 0)], dtype=np.uint8)
    return cmap
def apply_colormap(pred_mask, cmap):
    # Apply color map based on predicted class for each pixel
    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for i in range(19):  # Assuming 19 classes
        color_mask[pred_mask == i] = cmap[i]
    return color_mask

# Function to save predicted masks as PNGs with color
def save_pred_as_png(img_name, images, masks, predictions, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cmap = labelcolormap(19)  # Generate color map for 19 classes
    
    for i in range(len(images)):
        pred_mask = predictions[i].argmax(0).cpu().numpy()

        # Apply colormap to the predicted mask
        colored_pred_mask = apply_colormap(pred_mask, cmap)

        # Convert to PIL image and save with the original image name, but change extension to .png
        mask_img = Image.fromarray(colored_pred_mask)
        
        file_name = os.path.splitext(img_name[i])[0] + '.png'
        
        # Save the mask image as .png
        mask_img.save(os.path.join(save_dir, file_name))



# Inference on the validation set
save_dir = './colored_predictions'

with torch.no_grad():
    for batch_idx, (image, mask, img_name) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        image = image.to(device)
        mask_pred = model(image)
        save_pred_as_png(img_name, image.cpu(), mask.cpu(), mask_pred, save_dir)

print(f"All predictions saved to {save_dir}")
