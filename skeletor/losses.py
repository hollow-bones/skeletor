import torch
import torch.nn as nn
import torch.nn.functional as F


class DICELoss(nn.Module):

    def __init__(self, smooth=1.0):
        super(DICELoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        pred.requires_grad = True
        target.requires_grad = True
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) /
                     (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) +
                      self.smooth)))
        return loss.mean()



class BCEDicedLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super(BCEDicedLoss, self).__init__()
        self.dice_loss_fn = DICELoss(smooth=1.0)


    def forward(self, pred, target, weight=0.5):
        bce = F.binary_cross_entropy(pred,
                                     target)
        dice = self.dice_loss_fn(pred,
                                 target)

        loss = bce * weight + dice * (1 - weight)
        return loss
