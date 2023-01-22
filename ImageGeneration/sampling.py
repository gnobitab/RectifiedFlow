# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

import matplotlib.pyplot as plt

import torchvision
from tqdm import tqdm


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'rectified_flow':
    sampling_fn = get_rectified_flow_sampler(sde=sde, shape=shape, inverse_scaler=inverse_scaler, device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


def get_rectified_flow_sampler(sde, shape, inverse_scaler, device='cuda'):
  """
  Get rectified flow sampler

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  def euler_sampler(model, z=None):
    """The probability flow ODE sampler with simple Euler discretization.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      model_fn = mutils.get_model_fn(model, train=False) 
      
      ### Uniform
      dt = 1./sde.sample_N
      eps = 1e-3 # default: 1e-3
      for i in range(sde.sample_N):
        
        num_t = i /sde.sample_N * (sde.T - eps) + eps
        t = torch.ones(shape[0], device=device) * num_t
        pred = model_fn(x, t*999) ### Copy from models/utils.py 

        # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability 
        sigma_t = sde.sigma_t(num_t)
        pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.detach().clone())

        x = x.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
      
      x = inverse_scaler(x)
      nfe = sde.sample_N
      return x, nfe
  
  def rk45_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      rtol=atol=sde.ode_tol
      method='RK45'
      eps=1e-3

      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      model_fn = mutils.get_model_fn(model, train=False)

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model_fn(x, vec_t*999)

        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      x = inverse_scaler(x)
      
      return x, nfe
  

  print('Type of Sampler:', sde.use_ode_sampler)
  if sde.use_ode_sampler=='rk45':
      return rk45_sampler
  elif sde.use_ode_sampler=='euler':
      return euler_sampler
  else:
      assert False, 'Not Implemented!'
