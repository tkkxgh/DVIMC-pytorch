import copy
import torch
from torch import optim
from torch.utils.data import DataLoader
from datasets import build_dataset
import torch.nn as nn
from base_model import DVIMC
from evaluate import evaluate
from base_fn import kl_term, vade_trick, coherence_function
import numpy as np
import random
import argparse
from sklearn.cluster import KMeans


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def initialization(model, sv_loaders, cmv_data, args):
    print('Initializing......')
    criterion = nn.MSELoss()
    for v in range(args.num_views):
        optimizer = optim.Adam([{"params": model.encoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                {"params": model.decoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                ])
        for e in range(1, args.initial_epochs + 1):
            for batch_idx, xv in enumerate(sv_loaders[v]):
                optimizer.zero_grad()
                batch_size = xv.shape[0]
                xv = xv.reshape(batch_size, -1).to(args.device)
                _, xvr = model.sv_encode(xv, v)
                view_rec_loss = criterion(xvr, xv)
                view_rec_loss.backward()
                optimizer.step()
    with torch.no_grad():
        initial_data = [torch.tensor(csv_data, dtype=torch.float32).to(args.device) for csv_data in cmv_data]
        latent_representation_list = model.mv_encode(initial_data)
        assert len(latent_representation_list) == args.num_views
        fused_latent_representations = sum(latent_representation_list) / len(latent_representation_list)
        fused_latent_representations = fused_latent_representations.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=args.class_num, n_init=10)
        kmeans.fit(fused_latent_representations)
        model.prior_mu.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(args.device)


def train(model, optimizer, scheduler, imv_loader, args):
    print('Training......')
    eval_data = copy.deepcopy(imv_loader.dataset.data_list)
    eval_mask = copy.deepcopy(imv_loader.dataset.mask_list)
    for v in range(args.num_views):
        eval_data[v] = torch.tensor(eval_data[v], dtype=torch.float32).to(args.device)
        eval_mask[v] = torch.tensor(eval_mask[v], dtype=torch.float32).to(args.device)
    eval_labels = imv_loader.dataset.labels

    if args.likelihood == 'Bernoulli':
        likelihood_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        likelihood_fn = nn.MSELoss(reduction='none')

    for epoch in range(1, args.epochs + 1):
        epoch_loss = []
        for batch_idx, (batch_data, batch_mask) in enumerate(imv_loader):
            optimizer.zero_grad()
            batch_data = [sv_d.to(args.device) for sv_d in batch_data]
            batch_mask = [sv_m.to(args.device) for sv_m in batch_mask]
            z_sample, vs_mus, vs_vars, aggregated_mu, aggregated_var, xr_list, vade_z_sample = model(batch_data, batch_mask)

            qc_x = vade_trick(vade_z_sample, model.prior_weight, model.prior_mu, model.prior_var)
            z_loss, c_loss = kl_term(aggregated_mu, aggregated_var, qc_x, model.prior_weight, model.prior_mu, model.prior_var)
            kl_loss = z_loss + c_loss

            rec_term = []
            for v in range(args.num_views):
                sv_rec = torch.sum(likelihood_fn(xr_list[v], batch_data[v]), dim=1)  # ( Batch size * Dv )
                exist_rec = sv_rec * batch_mask[v].squeeze()
                view_rec_loss = torch.mean(exist_rec)
                rec_term.append(view_rec_loss)
            rec_loss = sum(rec_term)

            coherence_loss = coherence_function(vs_mus, vs_vars, aggregated_mu, aggregated_var, batch_mask)
            batch_loss = rec_loss + kl_loss + args.alpha * coherence_loss
            epoch_loss.append(batch_loss.item())
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            model.prior_weight.data = model.prior_weight.data / model.prior_weight.data.sum()
        scheduler.step()
        overall_loss = sum(epoch_loss) / len(epoch_loss)

        if epoch % args.interval == 0 or epoch == args.epochs:
            with torch.no_grad():
                _, _, _, aggregated_mu, _, _, _ = model(eval_data, eval_mask)

                mog_weight = model.prior_weight.data.detach().cpu()
                mog_mu = model.prior_mu.data.detach().cpu()
                mog_var = model.prior_var.data.detach().cpu()
                aggregated_mu = aggregated_mu.detach().cpu()

                c_assignment = vade_trick(aggregated_mu, mog_weight, mog_mu, mog_var)
                predict = torch.argmax(c_assignment, dim=1).numpy()
                acc, nmi, ari, pur = evaluate(eval_labels, predict)

                print(f'Epoch {epoch:>3}/{args.epochs}  Loss:{overall_loss:.2f}  ACC:{acc * 100:.2f}  '
                      f'NMI:{nmi * 100:.2f}  ARI:{ari * 100:.2f}  PUR:{pur * 100:.2f}')
    print('Finish training')
    return acc, nmi, ari, pur


def main(args):
    for t in range(1, args.test_times + 1):
        print(f'Test {t}')
        np.random.seed(t)
        random.seed(t)
        cmv_data, imv_dataset, sv_datasets = build_dataset(args)
        setup_seed(args.seed)
        imv_loader = DataLoader(imv_dataset, batch_size=args.batch_size, shuffle=True)
        sv_loaders = [DataLoader(sv_dataset, batch_size=args.batch_size, shuffle=True) for sv_dataset in sv_datasets]
        model = DVIMC(args).to(args.device)

        optimizer = optim.Adam(
            [{"params": model.encoders.parameters(), 'lr': args.learning_rate},
             {"params": model.decoders.parameters(), 'lr': args.learning_rate},
             {"params": model.prior_weight, 'lr': args.prior_learning_rate},
             {"params": model.prior_mu, 'lr': args.prior_learning_rate},
             {"params": model.prior_var, 'lr': args.prior_learning_rate},
             ])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
        initialization(model, sv_loaders, cmv_data, args)
        acc, nmi, ari, pur = train(model, optimizer, scheduler, imv_loader, args)
        test_record["ACC"].append(acc)
        test_record["NMI"].append(nmi)
        test_record["ARI"].append(ari)
        test_record["PUR"].append(pur)
    print('Average ACC {:.2f} Average NMI {:.2f} Average ARI {:.2f} Average PUR {:.2f}'.format(np.mean(test_record["ACC"]) * 100,
                                                                                               np.mean(test_record["NMI"]) * 100,
                                                                                               np.mean(test_record["ARI"]) * 100,
                                                                                               np.mean(test_record["PUR"]) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--initial_epochs', type=int, default=200, help='initialization epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--prior_learning_rate', type=float, default=0.05, help='initial mixture-of-gaussian learning rate')
    parser.add_argument('--z_dim', type=int, default=10, help='latent dimensions')
    parser.add_argument('--lr_decay_step', type=float, default=10, help='StepLr_Step_size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9, help='StepLr_Gamma')

    parser.add_argument('--dataset', type=int, default=0, choices=range(4), help='0:Caltech7-5v, 1:Scene-15, 2:Multi-Fashion, 3:NoisyMNIST')
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--test_times', type=int, default=10)
    parser.add_argument('--missing_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=5)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataset_dir_base = "./npz_data/"

    if args.dataset == 0:
        args.dataset_name = 'Caltech7-5V'
        args.alpha = 5
        args.seed = 5
        args.likelihood = 'Gaussian'
    elif args.dataset == 1:
        args.dataset_name = 'Scene-15'
        args.alpha = 20
        args.seed = 19
        args.likelihood = 'Gaussian'
    elif args.dataset == 2:
        args.dataset_name = 'Multi-Fashion'
        args.alpha = 10
        args.seed = 15
        args.likelihood = 'Bernoulli'
    else:
        args.dataset_name = 'NoisyMNIST'
        args.alpha = 10
        args.seed = 10
        args.likelihood = 'Bernoulli'
        args.batch_size = 512

    for missing_rate in [0.7, 0.5, 0.3, 0.1]:
        args.missing_rate = missing_rate
        print(f"Dataset : {args.dataset_name:<15} Missing rate : {args.missing_rate}")
        test_record = {"ACC": [], "NMI": [], "PUR": [], "ARI": []}
        main(args)
