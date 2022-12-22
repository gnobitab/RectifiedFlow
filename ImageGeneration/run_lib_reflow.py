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
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

FLAGS = flags.FLAGS


def finetune_reflow(config, workdir):
  """Runs the rematching finetune pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)

  ### NOTE: load pre-trained checkpoint
  
  ckpt_dir = config.reflow.last_flow_ckpt 
  loaded_state = torch.load(ckpt_dir, map_location=config.device)
  score_model.load_state_dict(loaded_state['model'], strict=False)
  loaded_ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  loaded_ema.load_state_dict(loaded_state['ema'])
  ema_score_model = mutils.create_model(config)
  loaded_ema.copy_to(ema_score_model.parameters())
  print('Loaded:', ckpt_dir, 'Step:', loaded_state['step'])

  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  elif config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, reflow_flag=True, reflow_t_schedule=config.reflow.reflow_t_schedule, reflow_loss=config.reflow.reflow_loss, use_ode_sampler=config.sampling.use_ode_sampler)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  data_root = config.reflow.data_root 
  print('DATA PATH:', data_root)
  print('T SCHEDULE:', config.reflow.reflow_t_schedule, 'LOSS:', config.reflow.reflow_loss)
  if config.reflow.reflow_type == 'generate_data_from_z0':
      # NOTE: Prepare reflow dataset with ODE
      print('Start generating data with ODE from z0', ', SEED:', config.seed)
      
      loaded_ema.copy_to(score_model.parameters())
      data_cllt = []
      z0_cllt = []
      for data_step in range(config.reflow.total_number_of_samples // config.training.batch_size):
        print(data_step)
        z0 = sde.get_z0(torch.zeros((config.training.batch_size, 3, config.data.image_size, config.data.image_size), device=config.device), train=False).to(config.device)
        batch = sde.ode(z0, score_model)
        
        print(batch.shape, batch.max(), batch.min(), z0.mean(), z0.std())

        z0_cllt.append(z0.cpu())
        data_cllt.append(batch.cpu())

      z0_cllt = torch.cat(z0_cllt)
      data_cllt = torch.cat(data_cllt)
      print(data_cllt.shape, z0_cllt.shape)
      print(z0_cllt.mean(), z0_cllt.std())
      if not os.path.exists(os.path.join(data_root, str(config.seed))):
        os.mkdir(os.path.join(data_root, str(config.seed)))
      np.save(os.path.join(data_root, str(config.seed), 'z1.npy'), data_cllt.numpy())
      np.save(os.path.join(data_root, str(config.seed), 'z0.npy'), z0_cllt.numpy())

      import sys 
      print('Successfully generated z1 from random z0 with random seed:', config.seed, 'Total number of pairs:', (data_step+1)*config.training.batch_size)
      sys.exit(0)

  elif config.reflow.reflow_type == 'train_reflow':
      # NOTE: load existing dataset
      print('START training with (Z0, Z1) pair')
      
      z0_cllt = []
      data_cllt = []
      folder_list = os.listdir(data_root)
      for folder in folder_list:
          print('FOLDER:', folder)
          z0 = np.load(os.path.join(data_root, folder, 'z0.npy'))
          print('Loaded z0')
          data = np.load(os.path.join(data_root, folder, 'z1.npy'))
          print('Loaded z1')
          z0 = torch.from_numpy(z0).cpu()
          data = torch.from_numpy(data).cpu()

          z0_cllt.append(z0)
          data_cllt.append(data)
          print('z0 shape:', z0.shape, 'z0 min:', z0.min(), 'z0 max:', z0.max())
          print('z1 shape:', data.shape, 'z1 min:', data.min(), 'z1 max:', data.max())
      
      print('Successfully Loaded (z0, z1) pairs!!!')
      z0_cllt = torch.cat(z0_cllt)
      data_cllt = torch.cat(data_cllt)
      print('Shape of z0:', z0_cllt.shape, 'Shape of z1:', data_cllt.shape)
  
  elif config.reflow.reflow_type == 'train_online_reflow':      
      pass
  else:
      assert False, 'Not implemented'

  print('Initial step of the model:', initial_step)
  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting reflow training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    if config.reflow.reflow_type == 'train_reflow':
        indices = torch.randperm(len(data_cllt))[:config.training.batch_size]
        data = data_cllt[indices].to(config.device).float()
        z0 = z0_cllt[indices].to(config.device).float()
    elif config.reflow.reflow_type == 'train_online_reflow':
        z0 = sde.get_z0(torch.zeros((config.training.batch_size, 3, config.data.image_size, config.data.image_size), device=config.device), train=False).to(config.device)
        data = sde.euler_ode(z0, ema_score_model, N=20)
        z0 = z0.to(config.device).float()
        data = data.to(config.device).float()

    batch = [z0, data]
    
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      if config.reflow.reflow_type == 'train_reflow':
          indices = torch.randperm(len(data_cllt))[:config.training.batch_size]
          data = data_cllt[indices].to(config.device).float()
          z0 = z0_cllt[indices].to(config.device).float()
      elif config.reflow.reflow_type == 'train_online_reflow':
          z0 = sde.get_z0(torch.zeros((config.training.batch_size, 3, config.data.image_size, config.data.image_size), device=config.device), train=False).to(config.device)
          data = sde.euler_ode(z0, ema_score_model, N=20)
          z0 = z0.to(config.device).float()
          data = data.to(config.device).float()

      eval_batch = [z0, data]
 
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)

