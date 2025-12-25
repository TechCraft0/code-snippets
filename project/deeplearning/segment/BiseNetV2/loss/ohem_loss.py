import torch
import torch.nn as nn


class OhemCELoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_lb=255):
        super().__init__()
        # 禁用阈值梯度 + 数值稳定
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        valid_mask = labels != self.ignore_lb
        n_min = max(1, valid_mask.sum() // 16)  # 确保n_min≥1
        loss = self.criteria(logits, labels).view(-1)

        # 难例挖掘：先选高损失样本，不足则补足n_min个最难的
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)
