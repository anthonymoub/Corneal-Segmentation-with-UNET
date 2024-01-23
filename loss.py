import torch.nn as nn

# Loss Fucntion option 1: DiceLoss class
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0):
        # inputs = (inputs >= 0.5).float()
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice
    
# Loss Function option 2: BCEwithLogits (this was the one used)    
    

class BCEwLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEwLogitsLoss, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average)

    def forward(self, inputs, targets):
        return self.bce_with_logits(inputs, targets)

    
