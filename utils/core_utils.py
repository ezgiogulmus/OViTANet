from argparse import Namespace
from collections import OrderedDict
import os
import pandas as pd
import pickle 
import h5py

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score

import torch
from torch.utils.data import DataLoader, RandomSampler

from models.vit2d import ViT
from models.mlp_model import MLP
from utils.utils import *
from utils.loss_func import CoxSurvLoss, NLLSurvLoss
import torch.nn as nn
device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Device: ", device)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, mode="max", warmup=5, patience=15, stop_epoch=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, epoch, score, model, ckpt_name = 'checkpoint.pt'):

        if self.mode == "min":
            score = -score

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
        elif score <= self.best_score or score == np.inf or np.isnan(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, score, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Saving the best model ...')
        torch.save(model.state_dict(), ckpt_name)

def init_model(args, ckpt_path=None):
    # Model, optimizer and loss

    model_dict = {
        "target_features": args.target_dim,
        "n_classes": args.n_classes if args.surv_model == "discrete" else 1,
        "drop_out": args.drop_out,
        "mlp_depth": args.mlp_depth,
        "mlp_type": args.mlp_type,
        "activation": args.activation,
        "skip": args.mlp_skip,
        "batch_norm": True if args.batch_size > 1 else False
        }
    
    if "path" not in args.mode:
        model = MLP(input_dim=args.nb_tabular_data, **model_dict)
    else:
        model_dict.update({
            "nb_tabular_data": args.nb_tabular_data,
            "mm_fusion": args.mm_fusion,
            "mm_fusion_type": args.mm_fusion_type,
            "path_input_dim": args.path_input_dim,
            "depth": args.depth, 
            "mha_heads": args.mha_heads, 
            "model_dim": args.model_dim if args.model_dim not in [None, "None", "none"] else None, 
            "mlp_dim": args.mlp_dim,
            "dim_head": args.dim_head,
            "pool": args.pool
        })
        model = ViT(**model_dict)
    
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-7)

    if args.surv_model == "cont":
        loss_fn = CoxSurvLoss(device)
    else:
        loss_fn = NLLSurvLoss()
    return model, optimizer, loss_fn, scheduler

def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    train_survival = np.array(list(zip(train_split.slide_data["event"].values, train_split.slide_data["survival_months"].values)), dtype=[('event', bool), ('time', np.float64)])
    max_surv_limit = int(np.min([train_split.slide_data["survival_months"].max(), val_split.slide_data["survival_months"].max(), test_split.slide_data["survival_months"].max()]))
    if args.surv_model == "discrete":
        time_intervals = list(train_split.time_breaks[1:-1])
        time_intervals.append(max_surv_limit)

    else:
        
        time_intervals = np.array(range(0, max_surv_limit, max_surv_limit//10))[1:]
    save_splits(datasets, ['train', 'val', "test"], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    train_loader = get_split_loader(train_split, training=True, 
        weighted = args.weighted_sample, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, batch_size=args.batch_size)
    test_loader = get_split_loader(test_split, batch_size=args.batch_size)

    model, optimizer, loss_fn, scheduler = init_model(args)
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(mode="min", warmup=2, patience=args.early_stopping, stop_epoch=20, verbose = True)
    else:
        early_stopping = None

    # window_size=1000 if args.model_type=="vit" else None
    window_size = None

    for epoch in range(args.max_epochs):
        print('Epoch: {}/{}'.format(epoch, args.max_epochs))
        loop_survival(cur, model, train_loader, epoch, optimizer=optimizer, writer=writer, loss_fn=loss_fn, gc=args.gc, window_size=window_size, training=True, discrete_time=True if args.surv_model == "discrete" else False, training_frac=args.train_fraction, mode=args.mode, time_intervals=time_intervals)
        stop = loop_survival(cur, model, val_loader, epoch, scheduler=scheduler, early_stopping=early_stopping, writer=writer, loss_fn=loss_fn, window_size=window_size, results_dir=args.results_dir, training=False, discrete_time=True if args.surv_model == "discrete" else False, mode=args.mode, train_survival=train_survival, time_intervals=time_intervals)
        if stop:
            break

    if os.path.isfile(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))):
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    results_val_dict, val_cindex, val_auc, val_ibs, val_loss = loop_survival(cur, model, val_loader, epoch, loss_fn=loss_fn, window_size=window_size, return_summary=True, discrete_time=True if args.surv_model == "discrete" else False, mode=args.mode, train_survival=train_survival, time_intervals=time_intervals)
    print('Val loss: {:4f} | c-Index: {:.4f} | mean AUC: {:.4f} | mean IBS: {:.4f}'.format(val_loss, val_cindex, val_auc, val_ibs))
    log = {
        "val_loss": val_loss,
        "val_cindex": val_cindex,
        "val_auc": val_auc,
        "val_ibs": val_ibs,
    }
    
    if not args.bootstrapping:
        results_test_dict, test_cindex, test_auc, test_ibs, test_loss = loop_survival(cur, model, test_loader, epoch, loss_fn=loss_fn, window_size=window_size, return_summary=True, discrete_time=True if args.surv_model == "discrete" else False, mode=args.mode, train_survival=train_survival, time_intervals=time_intervals)
        print("Test loss: {:4f} | c-Index: {:4f} | mean AUC: {:.4f} | mean IBS: {:.4f}".format(test_loss, test_cindex, test_auc, test_ibs))
        log.update({
            "test_loss": test_loss,
            "test_cindex": test_cindex,
            "test_auc": test_auc,
            "test_ibs": test_ibs,
        })
    else:
        print("Starting bootstraps", end=" ")
        if args.early_stopping > 0:
            print("on test set")
            target_split = test_split
        else: 
            print("on val set")
            target_split = val_split
        results_test_dict = None
        
        n_bootstrap=1000

        bs_cindex, bs_loss = [], []
        for i in range(n_bootstrap):
            if i % 100 == 0 and i !=0:
                print("\t", i, "/", n_bootstrap)
            bs_loader = DataLoader(target_split, batch_size=args.batch_size, sampler = RandomSampler(target_split, replacement=True), collate_fn = collate_MIL_survival)
            _, test_cindex, _, _, test_loss = loop_survival(cur=None, model=model, loader=bs_loader, loss_fn=loss_fn, window_size=window_size, return_summary=True, discrete_time=True if args.surv_model == "discrete" else False, mode=args.mode, train_survival=train_survival, time_intervals=time_intervals, cidx_only=True)
            
            bs_cindex.append(test_cindex)
            bs_loss.append(test_loss)
        bs_cindex = np.array(bs_cindex)
        bs_loss = np.array(bs_loss)

        mean_cindex = np.mean(bs_cindex)
        mean_loss = np.mean(bs_loss)

        std_cindex = np.std(bs_cindex)
        std_loss = np.std(bs_loss)
        conf_interval = np.percentile(bs_cindex, [2.5, 97.5])

        print(f"Loss: {round(mean_loss, 2)} +/- {round(std_loss, 2)} | C-index: {round(mean_cindex, 2)} +/- {round(std_cindex, 2)}, 95% CI: {conf_interval.round(2)}")
        
        log.update({
            "test_loss": mean_loss,
            "test_loss_std": std_loss,
            "test_cindex": mean_cindex,
            "test_cindex_std": std_cindex,
            "test_cindex_cilow": conf_interval[0],
            "test_cindex_ciup": conf_interval[1]
        })

    if writer:
        writer.add_scalar('bestval_c_index', log["val_cindex"])
        writer.add_scalar('test_c_index', log["test_cindex"])
        writer.add_scalar('bestval_loss', log["val_loss"])
        writer.add_scalar('test_loss', log["test_loss"])
    writer.close()
    
    
    return log, results_val_dict, results_test_dict


def loop_survival(
    cur, model, loader, epoch=None, scheduler=None, optimizer=None,
    early_stopping=None, writer=None, loss_fn=None, gc=16,
    window_size=None, results_dir=None, training=False, 
    return_summary=False, return_feats=False, dataname=None,
    discrete_time=True, training_frac=1., mode="path+tab", 
    train_survival=None, time_intervals=None, cidx_only=False
):
    model.train() if training else model.eval()
    loss_surv = 0.

    if return_summary or return_feats:
        patient_results = {}
        all_features, all_attentions = [], []
    # num_intervals = loader.dataset.num_classes // 2
    
    all_ids, all_events, all_risk_scores, all_event_times = [], [], [], []
    all_surv_probs = []
    for batch_idx, (data_WSI, y_disc, event_time, event, data_tab, case_id) in enumerate(loader):
        
        if training and training_frac < 1.:
            np.random.seed(7)
            random_ids = np.random.permutation(np.array(range(data_WSI.shape[0])))[:int(data_WSI.shape[0]*training_frac)]
            # print(data_WSI.shape)
            data_WSI = data_WSI[random_ids]
            # print(data_WSI.shape)
        data_WSI = data_WSI.to(device) if mode != "tab" else None
        data_tab = data_tab.to(device) if "tab" in mode else None
        # if torch.isnan(data_tab).any():
        #     print("NaN values in tabular features")
        #     print(data_tab)
        #     exit(1)
        # if torch.isinf(data_tab).any():
        #     print("Inf values in tabular features")
        #     print(data_tab)
        #     exit(1)
        
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        event = event.to(device)
        
        with torch.set_grad_enabled(training):
            if mode == "tab":
                logits = model(data_tab)
            else:
                out = model(x_path=data_WSI, x_tabular=data_tab, return_feats=return_feats or return_summary)

                if return_feats or return_summary:
                    logits, feats, atts = out
                    all_features.append(feats.detach().cpu().numpy())
                else:
                    logits = out

        if discrete_time:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            
            all_surv_probs.append(S.detach().cpu().numpy())
            risk = -torch.mean(S, dim=1).detach().cpu().numpy()
        else:
            
            y_pred = logits
            risk = y_pred.detach().cpu().numpy()
        
        all_ids.append(case_id)
        all_events.append(event.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())
        all_risk_scores.append(risk)
        
        if return_summary and discrete_time:
            # print(np.squeeze(hazards.detach().cpu().numpy()).shape)
            pt_dict = {
                'case_id': case_id, 
                'risk': risk, 
                'time': event_time.detach().cpu().numpy(), 
                'event': event.detach().cpu().numpy(),
                "hazards": np.squeeze(hazards.detach().cpu().numpy()),
            }
            
            patient_results.update({case_id.item(): pt_dict})

        if discrete_time:
            loss = loss_fn(h=hazards, y=y_disc, e=event)
        else:
            loss = loss_fn(y_pred, torch.unsqueeze(event_time, -1), torch.unsqueeze(event, -1))
        loss_value = loss.item()
        
        loss_surv += loss_value
        if mode != "tab" and batch_idx % 100 == 0:
            print('batch {}, loss: {:.2f}, event: {}, event_time: {:.2f}, risk: {:.2f}, bag_size: {}'.format(batch_idx, loss_value, event.detach().cpu().item(), float(event_time.detach().cpu().item()), float(risk), data_WSI.size(0)))
        
        if training:
            loss = loss / gc
            loss.backward()
            if (batch_idx + 1) % gc == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()
                optimizer.zero_grad()

    loss_surv /= len(loader)
    all_ids = np.concatenate(all_ids)
    
    all_risk_scores = np.concatenate(all_risk_scores)
    all_events = np.concatenate(all_events)
    all_event_times = np.concatenate(all_event_times)

    # c_index_intervals = [concordance_index(all_event_times, all_surv_probs[:, i], all_events) for i in range(num_intervals)]
    
    c_index = concordance_index_censored(all_events.astype(bool), all_event_times, np.squeeze(all_risk_scores), tied_tol=1e-08)[0]
    if training or cidx_only:       
        mean_auc, ibs = 0., 100
    else:
        survival = np.array(list(zip(all_events, all_event_times)), dtype=[('event', bool), ('time', np.float64)])
        try:
            auc_values, mean_auc = cumulative_dynamic_auc(train_survival, survival, np.squeeze(all_risk_scores), time_intervals)
        except ValueError:
            mean_auc = 0.
            
        ibs = 100
        if discrete_time:
            all_surv_probs = np.concatenate(all_surv_probs)
            try:
                ibs = integrated_brier_score(train_survival, survival, all_surv_probs, time_intervals)
            except ValueError:
                ibs = 100
    
    if return_feats:
        assert os.path.isdir(results_dir)
        df = pd.DataFrame()
        if len(all_features) > 0:
            df = pd.DataFrame(np.concatenate(all_features), columns=[f"feat{i}" for i in range(np.concatenate(all_features).shape[1])])
        df["time"] = all_event_times
        df["event"] = all_events
        df["risk"] = all_risk_scores
        df["ids"] = all_ids
        
        if discrete_time:
            hazards = np.squeeze(hazards.detach().cpu().numpy())
            for c in range(len(hazards)):
                df['hazard_{}'.format(c)] = hazards[c]
        
        file_path = os.path.join(results_dir, f'{dataname}_cv{cur}_features.csv')
        df.to_csv(file_path, index=False)
                
        return c_index, mean_auc, ibs, loss_surv

    if return_summary:
        return patient_results, c_index, mean_auc, ibs, loss_surv
        
    split = "Train" if training else "Validation"
    # print('\n{}, loss: {:.4f}, c_index: {:.4f},  mean_auc: {:.4f}'.format(split, loss_surv, c_index, mean_auc))
    print('{}, loss: {:.4f}, c_index: {:.4f},  mean_auc: {:.4f}, ibs: {:.4f}\n'.format(split, loss_surv, c_index, mean_auc, ibs))
    if writer:
        writer.add_scalar(f'{split}/loss', loss_surv, epoch)
        writer.add_scalar(f'{split}/c_index', c_index, epoch)
        writer.add_scalar(f'{split}/mean_auc', mean_auc, epoch)
        # writer.add_scalar(f'{split}/mean_c_index', np.mean(c_index_intervals), epoch)
        
    if early_stopping:
        assert results_dir
        if early_stopping.mode == "max":
            score = c_index
        else:
            score = round(loss_surv, 6)
        early_stopping(epoch, score, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True
        scheduler.step(loss_surv)

    return False


def eval_model(dataset, results_dir, args: Namespace, return_feats=True):
    """   
        eval for a single fold
    """
    print("Testing on {} samples".format(len(dataset)))

    model, _, loss_fn, _ = init_model(args)
    model.load_state_dict(torch.load(args.load_from))
    
    print('\nInit Data Loader...', end=' ')
    test_loader = get_split_loader(dataset, batch_size=args.batch_size)
    print('Done!')
    
    # window_size=1000 if args.model_type=="vit" else None
    window_size=None
    cindex, _, _, loss = loop_survival(args.cv, model, test_loader, loss_fn=loss_fn, window_size=window_size, results_dir=results_dir, return_feats=return_feats, dataname=args.dataname, discrete_time=True if args.surv_model == "discrete" else False, mode=args.mode,cidx_only=True)
    
    print("Test c-Index: {:4f} | loss: {:4f}".format(cindex, loss))
    return {
            'fold': int(args.cv), 
            'cindex': cindex, 
            "loss": loss
        }
