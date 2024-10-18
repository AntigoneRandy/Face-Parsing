# Official Implementation of CelebAMask Face Parsing Challenge Solution

## Introduction
This repository contains the official implementation of our solution for the CelebAMask Face Parsing Challenge. Our method is based on a lightweight DeepLab v3+ architecture with a MobileNet v2 backbone, enhanced by several data augmentation techniques, focal loss, dice loss, and self-distillation-based data regularization. The model achieves good performance on both large and small facial regions, while maintaining a parameter count below 2M.

## File Explanation
The files and dirs in this repo is placed as follows:

```
.
├── Deeplab_checkpoint_epoch_400.pt # our trained checkpoint
├── README.md # this file
├── class_imbalance_fig.py # code for drawing fig.2 in our tech report
├── color_visualization.py # to generate colored predictions for visualization
├── colored_predictions # path to save generated colored predictions
├── dataset # path to place the dataset
├── dataset_utils.py # dataset utils, including custom dataset, augmentation, etc.
├── evaluate.py # used to generate mIoU
├── f1score.py # used to generate f1 scores, precision, and recall
├── generate_labels.py # the self distillation method to re-generate the labels
├── iou_scores.txt # iou score results
├── iouclass.py # used to generate class-wise iou
├── loss.py # loss functions
├── main_deeplab.py # main file, the complete training and validation procedure
├── models # model architecture files
├── options.py # model hyperparameters and args
├── predictions # path to save prediction results
├── scores_with_f1_precision_recall.txt # f1 scores, precision, recall
└── validate.py # generate predictions given checkpoints and save path
```


## How to Run
### 0. Installing Necessary Packages
We recommend to create a new conda environment named `face` using Python 3.9:

```
conda create --name face python=3.9
conda activate face 
```

Then, navigate to this repo and install all the dependencies:

```
pip install -r requirements.txt
```

### 1. Train the Model
To train the model, first you need to place the dataset into `/dataset` as follows:

```
.
├── test_image
├── train
│   ├── train_image
│   └── train_mask
└── val
    ├── val_image
    └── val_mask
```

Then run the following command:

```bash
python main_deeplab.py --batch_size 32 --epochs 400 --learning_rate 0.001 --gpu_id 2
```

This will start training our model with a batch_size of 32 and a learning rate of 0.001. The model will be trained for 400 epochs using gpu 2 (if have multiple gpus). More options please refer to `main_deeplab.py` and `options.py`

### 2. Generate Predictions
To generate predictions, run the following command:

```bash
python validate.py
```
Make sure to modify the checkpoint path and dataset path inside the `validate.py` file. The predictions will be saved in the `/predictions` folder.

### 3. Evaluate Performance
To calculate the F-measure, precision, and recall, use the following command:
```bash
python f1score.py
```

To compute the IoU score for each class, use:
```bash
python iouclass.py
```
The results will be saved as `scores_with_f1_precision_recall.txt` and `iou_scores.txt`, respectively. Ensure that you filled the right dataset path inside these files before running them.

### 4. Default Checkpoint
The default checkpoint is ``Deeplab_checkpoint_epoch_400.pt``, which is the model trained for 400 epochs.

### Additional Notes & Acknowledgement

Note that our pseudo-label generation method is placed in `generate_labels.py`. Currently, we did not merge it into `main_deeplab.py`, so you may need to first train the model for 30 epochs and then run `python generate_labels.py` to merge the labels. Then, you need to change the label path in `main_deeplab.py` and load the checkpoints to continue training.

We used some code from https://github.com/VainF/DeepLabV3Plus-Pytorch, which is really helpful for us.