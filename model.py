import torch.nn as nn
import torch
import numpy as np
from network import diff_CSDI

from diff import GaussianDiffusion
class WorkloadDiff(nn.Module):
    def __init__(self,config,share_ratio_list, device):
        super().__init__()
        self.target_dim = config['others']['feature_num']
        self.is_unconditional = False
        self.device = device
        self.target_strategy = config["model"]["target_strategy"]
        config_diff = config["diffusion"]
        self.conditioning_length =24
        self.share_ratio_list = share_ratio_list
        self.diffmodel = diff_CSDI(config)  # dinosing network
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        self.weights = [0.6, 0.3,0.1]

        self.diffusion = GaussianDiffusion(
            self.diffmodel,
            input_size=self.target_strategy,
            diff_steps=self.num_steps,
            loss_type='l2',
            beta_end=0.1,
            # share ratio, new argument to control diffusion and sampling
            share_ratio_list=self.share_ratio_list,
            beta_schedule="linear",
        )  # diffusion network

    def process_data(self, batch):
        """

        :param batch:observed_data (B,F,L)

        :return: (B,L,F)      GT_MASK :(0,1)
        """



        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        observed_data_2=batch["observed_data_2"].to(self.device).float()
        observed_data_6 = batch["observed_data_4"].to(self.device).float()
        observed_data_7 = batch["observed_data_6"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        observed_data_2 = observed_data_2.permute(0, 2, 1)
        observed_data_6 =observed_data_6.permute(0, 2, 1)
        observed_data_7 = observed_data_7.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)

        for_pattern_mask = observed_mask
        return (
            observed_data,
            observed_mask,
            observed_data_2,
            observed_data_6,
            observed_data_7,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
    def Diff_noise(self,observed_data, cond_mask, observed_mask, observed_data_2,
            observed_data_6, observed_data_7,is_train, set_t=-1):

        observed_data_list = [observed_data, observed_data_2, observed_data_6]

        g =[1,2,3]
        likelihoods = []
        for ratio_index, (data,g) in enumerate(zip(observed_data_list, g)):

            cur_likelihood = self.diffusion.log_prob(
                x=data,
                observed_mask=observed_mask,
                cond_mask=cond_mask,
                g=g,
                )
            likelihoods.append(cur_likelihood)

        w1, w2,w3 = self.weights

     #   likelihood1, likelihood2,likelihood3 = likelihoods[0], likelihoods[1],likelihoods[2]
        #likelihood1, likelihood2= likelihoods[0], likelihoods[1]
        likelihood1, likelihood2,likelihood3= likelihoods[0],likelihoods[1],likelihoods[2]
        total_loss =w1*likelihood1+w2*likelihood2+w3*likelihood3
        return total_loss,likelihoods



    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_data_2,
            observed_data_6,
            observed_data_7,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        cond_mask = gt_mask
        loss_func = self.Diff_noise if is_train == 1 else self.Diff_noise_valid
        ans = loss_func(observed_data, cond_mask, observed_mask,observed_data_2,
            observed_data_6,observed_data_7,is_train)
        return ans


    def impute(self, observed_data, cond_mask, observed_mask, observed_data_2, observed_data_6,observed_data_7):
            B, K, L = observed_data.shape
            observed_data_list = [observed_data, observed_data_2,observed_data_6]
            #observed_data_list = [observed_data,observed_data_2,observed_data_6,observed_data_7]
            gran_levels = len(observed_data_list)
            gran_samples = {g: [] for g in range(1, gran_levels + 1)}
            nsample = 10
            for i in range(nsample):
               for g, data in enumerate(observed_data_list, start=1):
                 x = torch.randn_like(data)
                 for t in range(self.num_steps-1,-1, -1):
                    x = self.diffusion.p_sample_loop(x, data,cond_mask, t,g)
                 gran_samples[g].append(x)

            for g in gran_samples:
                 gran_samples[g] = torch.stack(gran_samples[g], dim=0)

            return gran_samples



    def evaluate(self, batch):
        (   observed_data,
            observed_mask,
            observed_data_2,
            observed_data_6,
            observed_data_7,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        ) = self.process_data(batch)
        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            samples = self.impute(observed_data, cond_mask, observed_mask,observed_data_2,
            observed_data_6,observed_data_7)
            for i in range(len(cut_length)):
                target_mask[i, ..., 0: cut_length[i].item()] = 0
            return samples, observed_data, observed_data_2,observed_data_6, target_mask, observed_mask, observed_tp