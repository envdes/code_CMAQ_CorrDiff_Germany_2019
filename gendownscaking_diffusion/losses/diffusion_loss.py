"""Diffusion-only (conditional-EDM) loss — direct full-field prediction without a regression network.

Equivalent to the baseline 'conditional EDM / SR3-style direct conditional diffusion'.
Matches physicsnemo ResidualLoss calling convention (net(noisy_latent, condition, sigma, …)).
"""

import torch
from torch import Tensor
from typing import Callable, Optional, Tuple


class DiffusionLoss:
    """Conditional-EDM denoising loss that predicts the full HR target directly.

    Unlike ResidualLoss this does NOT subtract a regression mean and does NOT
    concatenate a mean-HR channel to the conditioning.  It is a drop-in
    replacement for ``ResidualLoss`` when ``regression_net is None``.

    Parameters
    ----------
    P_mean : float
        Log-mean of the noise-level distribution.  Default ``-1.2`` (the
        standard EDM value) is suitable for direct full-field prediction.
    P_std : float
        Log-std of the noise-level distribution.
    sigma_data : float
        Expected std of the target data (used in the EDM preconditioning weight).
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        **kwargs,
    ) -> Tensor:
        """Compute the per-pixel conditional-EDM loss (no reduction).

        The calling convention mirrors ``ResidualLoss.__call__``:
        ``net(noisy_latent, y_lr, sigma, …)``, with sigma passed as a tensor
        of shape ``(B, 1, 1, 1)``.
        """
        # --- data augmentation (same pattern as ResidualLoss) ---
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]       # HR target
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]    # conditioning

        # --- noise level (log-normal, same as ResidualLoss) ---
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # --- forward pass ---
        # net signature: net(latent, y_lr, sigma, augment_labels=…)  (matches EDMPrecondSuperResolution)
        latent = y + torch.randn_like(y) * sigma
        D_yn = net(latent, y_lr, sigma, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss
