import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from dataset_utils import CustomData, create_transform
from models import network 
from torch.utils.data import DataLoader

device = 'cuda:0'
model_path = 'Deeplab_checkpoint_epoch_30.pt'
model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16, pretrained_backbone=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() 
dataset_dir = 'dataset'
train_images_dir = os.path.join(dataset_dir, 'train/train_image')
train_masks_dir = os.path.join(dataset_dir, 'train/train_mask')
merged_labels_dir = os.path.join(dataset_dir, 'train/merged_labels')
merged_labels_colored_dir = os.path.join(dataset_dir, 'train/merged_labels_colored')

os.makedirs(merged_labels_dir, exist_ok=True)
os.makedirs(merged_labels_colored_dir, exist_ok=True)

data_transform = create_transform(512)
train_dataset = CustomData(train_images_dir, train_masks_dir, data_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)

confidence_threshold = 0.95

def labelcolormap(N):
    cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0), (204, 204, 0),
                     (51, 51, 255), (204, 0, 204),  (51, 255, 255), (247, 206, 205),
                     (102, 51, 0), (255, 0, 0), (102, 204, 0), (255, 255, 0), 
                     (0, 0, 153), (0, 0, 204), (255, 51, 153), (0, 204, 204),
                     (0, 51, 0), (255, 153, 51), (0, 204, 0)], dtype=np.uint8)
    return cmap

def apply_colormap(pred_mask, cmap):
    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for i in range(19): 
        color_mask[pred_mask == i] = cmap[i]
    return color_mask

def generate_pseudo_labels(model, data_loader, device, threshold, save_dir, save_colored_dir, cmap):
    model.eval()
    
    for i, (images, masks) in enumerate(tqdm(data_loader)):
        images = images.to(device)
        masks = masks.numpy() 

        with torch.no_grad():
            predictions = model(images)
            softmax_preds = F.softmax(predictions, dim=1)  
            conf, pred_labels = torch.max(softmax_preds, dim=1)  

        conf = conf.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

        for j in range(images.shape[0]):  
            original_mask = masks[j]
            new_mask = original_mask.copy()
            confident_pixels = conf[j] > threshold
            new_mask[confident_pixels] = pred_labels[j][confident_pixels]

            image_filename = train_dataset.image_path[i * 64 + j]
            base_name = os.path.basename(image_filename).replace('.jpg', '.png') 
            new_mask_path = os.path.join(save_dir, base_name)

            new_mask_image = Image.fromarray(new_mask.astype(np.uint8))
            new_mask_image.save(new_mask_path)

            color_mask = apply_colormap(new_mask, cmap)
            color_mask_image = Image.fromarray(color_mask)
            color_mask_path = os.path.join(save_colored_dir, base_name)
            color_mask_image.save(color_mask_path)

cmap = labelcolormap(19)

generate_pseudo_labels(model, train_loader, device, confidence_threshold, merged_labels_dir, merged_labels_colored_dir, cmap)

print(f"merged labels saved at: {merged_labels_dir} and {merged_labels_colored_dir}")
