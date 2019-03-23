import torch
import torch.nn as nn
import torch.nn.functional as F


class DICELoss(nn.Module):

    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def forward(pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        return loss.mean()



class BCEDicedLoss(nn.Module):
    def __init__(self, smooth=1.0):
        dice_loss_fn = DICELoss(smooth=1.0)


    def forward(pred, target, weight=0.5):
        bce = F.binary_cross_entropy_with_logits(pred,
                                                 target,
                                                 reduction='mean')
        pred = F.sigmoid(pred)
        dice = self.dice_loss_fn(pred,
                                 target)

        loss = bce * bce_weight + dice * (1 - bce_weight)
        return loss
