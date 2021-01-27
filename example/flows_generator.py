from flax import linen as nn
import jax.numpy as np
from jax import random
from typing import Callable

from distributions.normal import standard_normal

def create_flows(bijections, dummy_input, prior=standard_normal, seed=0):
  input_dim = dummy_input.shape[-1]
  sample_key, flow_key = random.split(random.PRNGKey(seed=seed), 2)
  prior_log_pdf, prior_sample = standard_normal(input_dim=input_dim)

  class Flows(nn.Module):
    def setup(self):
        self.layers = [b() for b in bijections]

    def __call__(self, inputs):
      return self.log_prob(inputs)

    def g(self, z):
      x = z
      for layer in reversed(self.layers):
        x = layer(x, reverse=True)
      # TODO add log_det_J_layer
      return x

    def f(self, x):
      log_det_J, z =  np.zeros(x.shape[0]), x
      for layer in self.layers:
        z, log_det_J_layer = layer(z)
        log_det_J += log_det_J_layer

      return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return prior_log_pdf(z) + logp
        
    def sample(self, num_samples): 
        z = prior_sample(sample_key, num_samples)
        x = self.g(z)
        return x

  flows = Flows()
  params = flows.init(flow_key, dummy_input)

  def sample(params, num_samples):
    return flows.apply(params, num_samples, method=flows.sample)

  def log_pdf(params, input):
    return flows.apply(params, input, method=flows.log_prob)

  return params, log_pdf, sample