import torch
def count_trainable_parameters(model):
    """
    Returns the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    """
    Returns the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters())


def mean_relative_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes Mean Relative Error (MRE)

    y_pred, y_true: same shape
    returns: scalar tensor
    """
    return torch.mean(
        torch.abs(y_pred - y_true) / (torch.abs(y_true) + eps)
    )

def r2_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes R² score

    returns: scalar tensor
    """
    y_true_mean = torch.mean(y_true)

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)

    return 1.0 - ss_res / (ss_tot + eps)


def evaluate_metrics(y_pred, y_true):
    """
    Computes global MRE and R²
    """
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    mre = mean_relative_error(y_pred, y_true)
    r2  = r2_score(y_pred, y_true)

    return {
        "MRE": mre.item(),
        "R2": r2.item()
    }

def masked_mre(pred, target, mask, eps=1e-8):
    """
    pred, target: [B, 96, 200, 24]
    mask:          [B, 96, 200, 24] (bool)
    """
    B = pred.shape[0]
    mre = 0.0

    for i in range(B):
        p = pred[i][mask[i]]
        t = target[i][mask[i]]

        mre += torch.mean(torch.abs(p - t) / (torch.abs(t) + eps))

    return mre / B

def masked_r2(pred, target, mask):
    B = pred.shape[0]
    r2 = 0.0

    for i in range(B):
        p = pred[i][mask[i]]
        t = target[i][mask[i]]

        t_mean = torch.mean(t)
        ss_res = torch.sum((t - p) ** 2)
        ss_tot = torch.sum((t - t_mean) ** 2)

        r2 += 1.0 - ss_res / (ss_tot + 1e-8)

    return r2 / B


class NormalizedMRELoss(torch.nn.Module):
    def __init__(self, p=2, eps=1e-8):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, pred, target):
        num = torch.norm(pred - target, p=self.p)
        den = torch.norm(target, p=self.p) + self.eps
        return num / den