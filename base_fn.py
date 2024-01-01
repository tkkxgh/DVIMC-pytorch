import torch
import numpy as np


def log_gaussian(x, mu, var):
    return -0.5 * (torch.log(torch.tensor(2.0 * np.pi)) + torch.log(var) + torch.pow(x - mu, 2) / var)


def gaussian_kl(q_mu, q_var, p_mu, p_var):  # KL(Q|P)
    return - 0.5 * (torch.log(q_var / p_var) - q_var / p_var - torch.pow(q_mu - p_mu, 2) / p_var + 1)


def vade_trick(mc_sample, mog_pi, mog_mu, mog_var):
    log_pz_c = torch.sum(log_gaussian(mc_sample.unsqueeze(1), mog_mu.unsqueeze(0), mog_var.unsqueeze(0)), dim=-1)
    log_pc = torch.log(mog_pi.unsqueeze(0))
    log_pc_z = log_pc + log_pz_c
    pc_z = torch.exp(log_pc_z) + 1e-10
    normalized_pc_z = pc_z / torch.sum(pc_z, dim=1, keepdim=True)
    return normalized_pc_z


def kl_term(z_mu, z_var, qc_x, mog_weight, mog_mu, mog_var):
    z_kl_div = torch.sum(qc_x * torch.sum(gaussian_kl(z_mu.unsqueeze(1), z_var.unsqueeze(1), mog_mu.unsqueeze(0), mog_var.unsqueeze(0)), dim=-1),
                         dim=1)
    z_kl_div_mean = torch.mean(z_kl_div)

    c_kl_div = torch.sum(qc_x * torch.log(qc_x / mog_weight.unsqueeze(0)), dim=1)
    c_kl_div_mean = torch.mean(c_kl_div)
    return z_kl_div_mean, c_kl_div_mean


def coherence_function(vs_mus, vs_vars, aggregated_mu, aggregated_var, mask=None):
    coherence_loss_list = []
    mask_stack = torch.cat(mask, dim=1)  # Batch size * V
    norm = torch.sum(mask_stack, dim=1)  # |Via|
    for v in range(len(vs_mus)):
        uniview_coherence_loss = torch.sum(gaussian_kl(aggregated_mu, aggregated_var, vs_mus[v], vs_vars[v]), dim=1)
        exist_loss = uniview_coherence_loss * mask[v].squeeze()
        coherence_loss_list.append(exist_loss)
    coherence_loss = torch.mean(sum(coherence_loss_list) / norm)
    return coherence_loss
