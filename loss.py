from torch import nn, Tensor, flatten

class EuclidLoss(nn.Module):
    """"""

    def __init__(self) -> None:
        super(EuclidLoss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # Input: torch.Size([16, 3, 96, 96])
        
        # torch.Size([16*3*96*96])
        sr_tensor = flatten(sr_tensor)
        gt_tensor = flatten(gt_tensor)
        
        # Input: torch.Size([16, 3, 96, 96])
        loss = self.pdist(sr_tensor, gt_tensor)

        return loss

