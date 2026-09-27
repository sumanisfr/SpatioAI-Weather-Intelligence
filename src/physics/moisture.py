"""Optional moisture-related proxy terms.

These utilities are disabled by default and are not moisture-conservation laws.
They require supplied humidity, temperature, and pressure fields.
"""

from typing import Optional

import torch
from torch.nn import functional as F


def moisture_proxy_loss(
    precipitation: torch.Tensor,
    specific_humidity: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    surface_pressure: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Match normalized precipitation and humidity spatial patterns.

    This is a diagnostic correlation-style proxy only; it does not enforce
    moisture continuity or a column water budget.
    """
    if specific_humidity.shape != precipitation.shape:
        raise ValueError("specific_humidity must match precipitation shape")
    precip = precipitation - precipitation.mean(dim=(-2, -1), keepdim=True)
    humidity = specific_humidity - specific_humidity.mean(dim=(-2, -1), keepdim=True)
    precip = precip / precip.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    humidity = humidity / humidity.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return F.smooth_l1_loss(precip, humidity.detach())
