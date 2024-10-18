import argparse

def parse_common_args(parser):
    parser.add_argument('--model_type', type=str, default='unet', help='Model name (used in model loading)')
    parser.add_argument('--data_type', type=str, default='segmentation', help='Dataset type (used in data loading)')
    parser.add_argument('--save_prefix', type=str, default='exp', help='Prefix for saving models and results')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to a pretrained model')
    parser.add_argument('--load_not_strict', action='store_true', help='Allow partial loading of model parameters')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='Root directory for the dataset')
    parser.add_argument('--gpus', nargs='+', type=int, help='List of GPU IDs to use')
    return parser

def parse_train_args():
    parser = argparse.ArgumentParser(description='Training options')
    parser = parse_common_args(parser)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size')
    parser.add_argument('--image_size', type=int, default=512, help='Input image size for model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels (RGB = 3)')
    parser.add_argument('--out_channels', type=int, default=19, help='Number of output channels (classes)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for training')
    return parser.parse_args()

def parse_test_args():
    parser = argparse.ArgumentParser(description='Testing options')
    parser = parse_common_args(parser)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--save_visualizations', action='store_true', help='Save test visualizations')
    return parser.parse_args()
