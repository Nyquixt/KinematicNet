# https://github.com/airalcorn2/pytorch-geodesic-loss/blob/master/geodesic_loss.py
import torch
from torch import nn
from torch import Tensor

class GeodesicLoss(nn.Module):
    r"""Creates a criterion that measures the distance between rotation matrices, which is
    useful for pose estimation problems.
    The distance ranges from 0 to :math:`pi`.
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices and:
    "Metrics for 3D Rotations: Comparison and Analysis" (https://link.springer.com/article/10.1007/s10851-009-0161-2).
    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.
    The loss can be described as:
    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)
    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
    Shape:
        - Input: :math:`(N, J, 3, 3)`.
        - Target: :math:`(N, J, 3, 3)`.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size = input.shape[0]
        device = input.device
        total_loss = torch.zeros(input.shape[1]).to(device)
        for b in range(batch_size):
            R_diffs = input[b] @ target[b].permute(0, 2, 1)
            # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
            traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
            dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps)) # J
            total_loss += dists

        return total_loss.mean() / batch_size

class RotationMatrixL2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size = input.shape[0]
        pred = input.view(batch_size, -1, 9)
        gt = target.view(batch_size, -1, 9)
        loss = torch.mean(torch.sqrt(torch.sum((gt - pred) ** 2, dim=2)))
        loss = loss / batch_size
        return loss

class QuaternionL2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size = input.shape[0]
        loss = torch.mean(torch.sqrt(torch.sum((target - input) ** 2, dim=2)))
        loss = loss / batch_size
        return loss

class AngleL2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size = input.shape[0]
        loss = torch.mean(torch.sqrt(torch.sum((target - input) ** 2, dim=2)))
        loss = loss / batch_size
        return loss

class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss

class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss

class KeypointsMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss

def test():
    m1 = torch.rand(8, 17, 3, 3)
    m2 = torch.rand(8, 17, 3, 3)
    criterion = GeodesicLoss()
    print(criterion(m1, m2))
    criterion = RotationMatrixL2Loss()
    print(criterion(m1, m2))

# test()