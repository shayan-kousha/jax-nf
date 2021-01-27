import sys

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as np
from jax import random
from typing import Callable
import matplotlib.pyplot as plt

from distributions.normal import standard_normal
from dataloader import MNIST, CIFAR10
from dataloader.utils import disp_imdata, logistic

from bijectors.affine_coupling import affineCoupling
from example.flows_generator import create_flows


def init(key, shape, dtype=np.float32):
  return random.uniform(key, shape, dtype, minval=-np.sqrt(1/shape[0]), maxval=np.sqrt(1/shape[0])) 

def translate_log_scale(hidden_layer, output_layer, kernel_init=init, bias_init=init):
  class Transform(nn.Module):
    kernel_init: Callable
    bias_init: Callable 
    
    @nn.compact
    def __call__(self, x):
      x = nn.Dense(hidden_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
      x = nn.leaky_relu(x)
      x = nn.Dense(hidden_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
      x = nn.leaky_relu(x)
      x = nn.Dense(output_layer, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
      x, log_scale = np.split(x, 2, axis=-1)

      return x, nn.tanh(log_scale)

  def create_layer():
    return Transform(kernel_init, bias_init)

  return create_layer

def train_real_nvp(dataset, monitor_every=10):
  data = dataset(logit=True, dequantize=True)
  hidden_nodes = 1024
  output_nodes = data.trn.x[0].shape[0]
  learning_rate = 1e-4
  epoch = 500
  batch_size = 100
  num_layers = 10
  num_training_data = data.trn.x.shape[0]

  bijections = [affineCoupling(translate_log_scale(hidden_nodes, output_nodes), _reverse_mask=layer % 2 != 0) for layer in range(num_layers)]
  params, log_pdf, sample = create_flows(bijections, data.trn.x[:2], prior=standard_normal, seed=0)

  optimizer_def = optim.Adam(learning_rate=learning_rate)
  optimizer = optimizer_def.create(params)

  @jax.jit
  def loss_fn(params, batch):
    return -np.mean(log_pdf(params=params, input=batch))

  @jax.jit
  def train_step(optimizer, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grad = grad_fn(optimizer.target, batch)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss_val


  for e in range(epoch):
    for batch in range(int(num_training_data/batch_size)):
      train_data = data.trn.x[batch*batch_size:(batch+1)*batch_size]
      optimizer, loss_val = train_step(optimizer, np.round(train_data, 4))
      
    if e % 5 == 0:
      validation_loss = loss_fn(optimizer.target, data.val.x)
      
      x = sample(optimizer.target, 25)
      x = (logistic(x) - dataset.alpha) / (1 - 2*dataset.alpha)
      disp_imdata(x, data.image_size, [5, 5])
      plt.savefig('./sample_data/{}/epoch-{}.png'.format(dataset.name, e))

      print('epoch %s/%s batch %s/%s:' % (e+1, epoch, batch, int(num_training_data/batch_size)), 'loss = %.3f' % loss_val, 'val_loss = %0.3f' % validation_loss)

datasets = {
  'mnist': MNIST,
  'cifar10': CIFAR10
}
if __name__ == "__main__":
  train_real_nvp(datasets[sys.argv[-1]])