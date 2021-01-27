from flax import linen as nn
import jax.numpy as np
from typing import Callable

def affineCoupling(shift_and_log_scale_fn, _reverse_mask=False):
  class AffineCoupling(nn.Module):
    shift_and_log_scale_fn: Callable
    _reverse_mask: bool

    def setup(self):
      self.shift_and_log_scale = self.shift_and_log_scale_fn()

    def __call__(self, x, reverse=False):
      mask_size = x.shape[-1] // 2
      if reverse:
        z = x

        z0 = z[..., :mask_size]
        z1 = z[..., mask_size:]
        
        if self._reverse_mask:
          z0, z1 = z1, z0

        translation, log_scale = self.shift_and_log_scale(z0)
        z1 -= translation
        z1 *= np.exp(-log_scale)

        if self._reverse_mask:
          z1, z0 = z0, z1

        x = np.concatenate([z0, z1], axis=-1)

        return x
      else:
        x0 = x[..., :mask_size]
        x1 = x[..., mask_size:]

        if self._reverse_mask:
          x0, x1 = x1, x0

        translation, log_scale = self.shift_and_log_scale(x0)
        x1 *= np.exp(log_scale)
        x1 += translation

        if self._reverse_mask:
          x1, x0 = x0, x1

        z = np.concatenate([x0, x1], axis=-1)

        return z, np.sum(log_scale, axis=1)

  def create_layer():
    return AffineCoupling(shift_and_log_scale_fn, _reverse_mask)

  return create_layer
