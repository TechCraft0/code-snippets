import torch
import torch.nn as nn

from .dice_loss import DiceLoss

class DetailLoss(nn.Module):
    '''Implement detail loss used in paper
       `Rethinking BiSeNet For Real-time Semantic Segmentation`'''
    def __init__(self, dice_loss_coef=1., bce_loss_coef=1., smooth=1):
        super().__init__()
        self.dice_loss_coef = dice_loss_coef
        self.bce_loss_coef = bce_loss_coef
        self.dice_loss_fn = DiceLoss(smooth)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        loss = self.dice_loss_coef * self.dice_loss_fn(logits, labels) + \
               self.bce_loss_coef * self.bce_loss_fn(logits, labels)

        return loss
    


