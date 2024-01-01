import torch
import torch.nn as nn


class Gaussian_sampling(nn.Module):
    def forward(self, mu, var):
        std = torch.sqrt(var)
        epi = std.data.new(std.size()).normal_()
        return epi * std + mu


class Gaussian_poe(nn.Module):
    def forward(self, mu, var, mask=None):
        mask_matrix = torch.stack(mask, dim=0)
        exist_mu = mu * mask_matrix
        T = 1. / var
        exist_T = T * mask_matrix
        aggregate_T = torch.sum(exist_T, dim=0)
        aggregate_var = 1. / aggregate_T
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / aggregate_T
        return aggregate_mu, aggregate_var


class view_specific_encoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super(view_specific_encoder, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(self.x_dim, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 2000),
                                     nn.ReLU()
                                     )
        self.z_mu = nn.Linear(2000, self.z_dim)
        self.z_var = nn.Sequential(nn.Linear(2000, self.z_dim), nn.Softplus())

    def forward(self, x):
        hidden_feature = self.encoder(x)
        vs_mu = self.z_mu(hidden_feature)
        vs_var = self.z_var(hidden_feature)
        return vs_mu, vs_var


class view_specific_decoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super(view_specific_decoder, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.decoder = nn.Sequential(nn.Linear(self.z_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, self.x_dim),
                                     )

    def forward(self, z):
        xr = self.decoder(z)
        return xr


class DVIMC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_dim_list = args.multiview_dims
        self.k = args.class_num
        self.z_dim = args.z_dim
        self.num_views = args.num_views

        self.prior_weight = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        self.prior_mu = nn.Parameter(torch.full((self.k, self.z_dim), 0.0), requires_grad=True)
        self.prior_var = nn.Parameter(torch.full((self.k, self.z_dim), 1.0), requires_grad=True)

        self.encoders = nn.ModuleDict({f'view_{v}': view_specific_encoder(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})
        self.decoders = nn.ModuleDict({f'view_{v}': view_specific_decoder(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})

        self.aggregated_fn = Gaussian_poe()
        self.sampling_fn = Gaussian_sampling()

    def mv_encode(self, x_list):
        latent_representation_list = []
        for v in range(self.num_views):
            latent_representation, _ = self.encoders[f'view_{v}'](x_list[v])
            latent_representation_list.append(latent_representation)
        return latent_representation_list

    def sv_encode(self, x, view_idx):
        latent_representation, _ = self.encoders[f'view_{view_idx}'](x)
        xr = self.decoders[f'view_{view_idx}'](latent_representation)
        return latent_representation, xr

    def inference_z(self, x_list, mask):
        vs_mus, vs_vars = [], []
        for v in range(self.num_views):
            vs_mu, vs_var = self.encoders[f'view_{v}'](x_list[v])
            vs_mus.append(vs_mu)
            vs_vars.append(vs_var)
        mu = torch.stack(vs_mus)
        var = torch.stack(vs_vars)
        aggregated_mu, aggregated_var = self.aggregated_fn(mu, var, mask)
        return vs_mus, vs_vars, aggregated_mu, aggregated_var

    def generation_x(self, z):
        xr_list = [vs_decoder(z) for vs_decoder in self.decoders.values()]
        return xr_list

    def forward(self, x_list, mask=None):
        vs_mus, vs_vars, aggregated_mu, aggregated_var = self.inference_z(x_list, mask)
        z_sample = self.sampling_fn(aggregated_mu, aggregated_var)
        xr_list = self.generation_x(z_sample)
        vade_z_sample = self.sampling_fn(aggregated_mu, aggregated_var)
        return z_sample, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample
