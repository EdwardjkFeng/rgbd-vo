""" Guass-Newton or Levenberg-Marquardt """

import torch
import logging

logger = logging.getLogger(__name__)


def optimizer_step(g, H, lambda_=0, mute=False, mask=None, eps=1e-6):
    """One optimization step with Guass-Newton or Levenberg-Marquardt.

    Args:
        g: batched gradient tensor of shape (..., N).
        H: batched Hessian tensor of shape (..., N, N).
        lambda_: damping factor for LM (use GN if lambda_ = 0). Defaults to 0.
        mute: _description_. Defaults to False.
        mask: denotes valid elements of the batch. Defaults to None.
        eps: _description_. Defaults to 1e-6.
    """
    if lambda_ is 0: # use GN
        diag = torch.zeros_like(g)
    else:
        diag = H.diagonal(dim1=-2, dim2=-1) * lambda_
    H = H + diag.clamp(min=eps).diag_embed()

    if mask is not None:
        # make sure that masked elements are not singular
        H = torch.where(mask[..., None, None], H, torch.eye(H.shape[-1]).to(H))
        # set g to 0 for masked elements
        g = g.masked_fill(~mask[..., None], 0.)

    H_, g_ = H.cpu(), g.cpu()
    try:
        U = torch.linalg.cholesky(H_)
    except RuntimeError as e:
        if 'singular U' in str(e):
            if not mute:
                logger.debug(
                    'Cholesky decomposition failed, fallback to LU.'
                )
            delta = -torch.lu_solve(g_[..., None], H_)[0][..., 0]
        else:
            raise
    else:
        delta = -torch.cholesky_solve(g_[..., None], U)[..., 0]

    return delta.to(H.device)

