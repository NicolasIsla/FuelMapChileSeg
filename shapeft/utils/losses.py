import torch
import torch.nn.functional as F

class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float]) -> None:
        super(WeightedCrossEntropy, self).__init__()
        # Initialize the weights based on the given distribution
        self.weights = [1 / w if w!=0 else 0 for w in distribution]

        # Convert weights to a tensor and move to CUDA
        loss_weights = torch.Tensor(self.weights).to("cuda")
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=loss_weights
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the weighted cross-entropy loss
        return self.loss(logits, target)
    


def _prep_mask(mask: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Returns broadcastable mask of shape (B, 1, H, W) on same device as pred.
    Accepts (B,H,W) or (B,1,H,W). Values expected 0/1 or bool.
    """
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)
    elif mask.ndim != 4:
        raise ValueError(f"mask must be (B,H,W) or (B,1,H,W), got {mask.shape}")
    mask = mask.to(device=pred.device)
    if mask.dtype != torch.bool:
        mask = mask > 0
    return mask


class MaskedHuberLoss(torch.nn.Module):
    """
    SmoothL1/Huber is usually a good default for regression.
    """
    def __init__(self, beta: float = 1.0, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        loss = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")  # (B,C,H,W)

        if mask is None:
            return loss.mean() if self.reduction == "mean" else loss.sum()

        mask = _prep_mask(mask, pred)
        loss = loss * mask

        denom = mask.sum() * pred.shape[1] + self.eps
        if self.reduction == "mean":
            return loss.sum() / denom
        return loss.sum()
