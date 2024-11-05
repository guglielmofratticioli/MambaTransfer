#import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class MAELoss(_Loss):
    """
    Mean Absolute Error Loss (L1 Loss), equivalent to Keras's mean_absolute_error.
    """

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, ests, targets):
        if ests.size() != targets.size():
            raise TypeError(f'Inputs must be of the same shape, got {ests.size()} and {targets.size()}')
        return self.l1_loss(ests, targets)