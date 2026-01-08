import torch

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