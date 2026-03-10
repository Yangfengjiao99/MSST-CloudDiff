import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import copy




class HierarchicalVarianceScheduler:
    def __init__(self, total_steps=100, gran_levels=3):
        self.total_steps = total_steps
        self.gran_levels = gran_levels
        self.alpha_self = {}
        self.alpha = {}
        self.beta = {}
        self.gamma = {}
        self.alpha_cumprod = {}


    def init_schedules(self, device):

        self.alpha_self[1] = 1.0 - torch.linspace(0.00015, 0.025, self.total_steps).to(device)
        self.alpha_self[2] = 1.0 - torch.linspace(0.0001, 0.02, self.total_steps).to(device)
        #self.alpha_self[1] = (1.0 - torch.linspace(1e-4, 0.1**0.5, self.total_steps)**2).to(device)
        self.alpha_self[3] = 1.0 - torch.linspace(0.00005, 0.015, self.total_steps).to(device)
        self.alpha_self[4] = 1.0 - torch.linspace(0.00001, 0.01, self.total_steps).to(device)

        for g in range(1, self.gran_levels + 1):
            if g == self.gran_levels:
                self.gamma[g] = torch.zeros(self.total_steps, device=device)
            else:
                #self.gamma[g] = torch.sigmoid(torch.linspace(-5, 5, self.total_steps)).to(device)
                self.gamma[g] = torch.sigmoid(torch.linspace(-5, 5, self.total_steps)).to(device)
        if self.gran_levels >= 4:
            self.alpha[4] = self.alpha_self[4]
        self.alpha[3] = self.gamma[3] * self.alpha[4] + (1 - self.gamma[3]) * self.alpha_self[3]
        self.alpha[2] = self.gamma[2] * self.alpha[3] + (1 - self.gamma[2]) * self.alpha_self[2]
        self.alpha[1] = self.gamma[1] * self.alpha[2] + (1 - self.gamma[1]) * self.alpha_self[1]
        for g in range(1, self.gran_levels + 1):
            self.beta[g] = 1.0 - self.alpha[g]

        for g in range(1, self.gran_levels + 1):
            self.alpha_cumprod[g] = torch.cumprod(self.alpha[g], dim=0)
    def forward_diffusion(self, x0, g):

        batch_size = x0.shape[0]

        seq_len = x0.shape[1]
        features = x0.shape[2]


        n = torch.randint(0, self.total_steps, (batch_size,), device=x0.device)
        eps = torch.randn_like(x0)
        a_n = self.alpha_cumprod[g][n]  # \bar{\alpha}_t
        a_n = a_n.view(-1, 1, 1)
        x_n = torch.sqrt(a_n) * x0 + torch.sqrt(1 - a_n) * eps
        return x_n, eps, n

    def reverse_diffusion(self, xt, t, predicted, g):
        """

        Args:
            current_sample: (batch_size, seq_len, features)
            t:
            predicted:
            g:
        Returns:
            x_{t-1}:
        """
        current_sample=xt

        if t > 1:
            eps = torch.randn_like(current_sample)
        else:
            eps = 0.0

        alpha_hat_t = self.alpha_self[g][t]
        alpha_hat_t = alpha_hat_t.view(-1, 1, 1)
        current_sample = 1 / (alpha_hat_t ** 0.5) * (
                current_sample - (1 - alpha_hat_t) / (1 - alpha_hat_t) ** 0.5 * predicted
        )+(1 - alpha_hat_t)**0.5*eps

        return current_sample
class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            input_size,
            share_ratio_list,
            beta_end=0.1,
            diff_steps=100,
            loss_type="l2",
            betas=None,
            beta_schedule="linear",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.__scale = None
        self.share_ratio_list = share_ratio_list  # ratio of betas are shared
        self.loss_type = loss_type

        self.variance_scheduler = HierarchicalVarianceScheduler(
            total_steps=diff_steps,
            gran_levels=4
        )


    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def p_losses(self, x_start, observed_mask,cond_mask,g):
        """
        :param x_start: (batch_size, features,seq_len)
        :param observed_mask: (batch_size, features,seq_len)
        :param cond_mask: (batch_size, features,seq_len)
        :param adj:  (features, features)
        :param g:
        """

        if not hasattr(self.variance_scheduler, 'alpha') or self.variance_scheduler.alpha.get(1) is None:
            self.variance_scheduler.init_schedules(device=x_start.device)

        pre_mask =observed_mask-cond_mask
        for_noisy, noise, n = self.variance_scheduler.forward_diffusion(x_start, g)
        x_recon = self.denoise_fn(for_noisy,x_start, cond_mask, n)

        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)* pre_mask
        elif self.loss_type == "l2":
            residual = (noise - x_recon) * pre_mask
            num_eval = pre_mask.sum()
            loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)* pre_mask
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        return loss

    def  noise_forward (self,x,g):
        if not hasattr(self.variance_scheduler, 'alpha') or self.variance_scheduler.alpha.get(1) is None:
            self.variance_scheduler.init_schedules(device=x.device)
        for_noisy, noise, n = self.variance_scheduler.forward_diffusion(x, g)
        return  for_noisy




    def p_sample_loop(self,x_start,data,conda_mask,n,g):
        if not hasattr(self.variance_scheduler, 'alpha') or self.variance_scheduler.alpha.get(1) is None:
            self.variance_scheduler.init_schedules(device=x_start.device)
        x_recon = self.denoise_fn(x_start,data,conda_mask, n)
        x0 = self.variance_scheduler.reverse_diffusion(x_start, n, x_recon, g)

        return x0

    def log_prob(self, x, observed_mask,cond_mask,g,*args, **kwargs):
        loss = self.p_losses(
            x, observed_mask,cond_mask,g,
            *args, **kwargs
        )

        return loss
