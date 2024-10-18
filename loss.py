import torch.nn as nn
import torch
import torch.nn.functional as F 

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, num_classes=19):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)  
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dice_loss = 0.0
        for c in range(self.num_classes):
            pred_c = pred[:, c, :, :].contiguous().view(-1)
            target_c = target_one_hot[:, c, :, :].contiguous().view(-1)
            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_loss += (1 - dice)
        return dice_loss / self.num_classes

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
    
def SoftDistillationLoss(predictions, soft_targets, temperature=1.0):
    # Apply temperature scaling to soften the logits
    predictions = F.log_softmax(predictions / temperature, dim=1)
    soft_targets = F.softmax(soft_targets / temperature, dim=1)
    
    # Calculate KL divergence loss
    kl_loss = F.kl_div(predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
    return kl_loss

def flatten_probas(input, target, labels):
    """
    Flattens predictions in the batch.
    """
    B, C, H, W = input.size()
    input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    target = target.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    valid = labels
    vinput = input[torch.nonzero(valid, as_tuple=False).squeeze()]
    vtarget = target[torch.nonzero(valid, as_tuple=False).squeeze()]
    return vinput, vtarget


class KLDivergenceLoss(nn.Module):
    def __init__(self, T=1):
        super(KLDivergenceLoss, self).__init__()
        self.T = T

    def forward(self, input, target, label):
        log_input_prob = F.log_softmax(input / self.T, dim=1)
        target_porb = F.softmax(target / self.T, dim=1)
        loss = F.kl_div(*flatten_probas(log_input_prob, target_porb, label), reduction='batchmean')
        return self.T*self.T*loss # balanced