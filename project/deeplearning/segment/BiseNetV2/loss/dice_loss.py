import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        logits = torch.flatten(logits, 1)
        labels = torch.flatten(labels, 1)

        intersection = torch.sum(logits * labels, dim=1)
        loss = 1 - ((2 * intersection + self.smooth) / (logits.sum(1) + labels.sum(1) + self.smooth))

        return torch.mean(loss)
    

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()