"""
HE2RNA: definition of the algorithm to generate a model for gene expression prediction
Copyright (C) 2020  Owkin Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import torch
import time
import os
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scipy import stats
import torch.nn.functional as F

from classic_algos.nn.modules.loss import CoxLoss, SmoothCindexLoss
from lifelines.utils import concordance_index


class HE2RNA(nn.Module):
    """Model that generates one score per tile and per predicted gene.

    Args
        output_dim (int): Output dimension, must match the number of genes to
            predict.
        layers (list): List of the layers' dimensions
        nonlin (torch.nn.modules.activation)
        ks (list): list of numbers of highest-scored tiles to keep in each
            channel.
        dropout (float)
        device (str): 'cpu' or 'cuda'
        mode (str): 'binary' or 'regression'
    """
    def __init__(self, input_dim, output_dim,
                 layers=[1], nonlin=nn.ReLU(), ks=[10],
                 dropout=0.5, device='cpu',
                 bias_init=None, **kwargs):
        super(HE2RNA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [input_dim] + layers + [output_dim]
        self.layers = []
        for i in range(len(layers) - 1):
            layer = nn.Conv1d(in_channels=layers[i],
                              out_channels=layers[i+1],
                              kernel_size=1,
                              stride=1,
                              bias=True)
            setattr(self, 'conv' + str(i), layer)
            self.layers.append(layer)
        if bias_init is not None:
            self.layers[-1].bias = bias_init
        self.ks = np.array(ks)

        self.nonlin = nonlin
        self.do = nn.Dropout(dropout)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        if self.training:
            k = int(np.random.choice(self.ks))
            return self.forward_fixed_k(x, k)
        else:
            pred = 0
            for k in self.ks:
                pred += self.forward_fixed_k(x, int(k)) / len(self.ks)
            return pred

    def forward_fixed_k(self, x, k):
        mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = (mask > 0).float()
        x = self.conv(x) * mask
        t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)
        x = torch.sum(t * mask[:, :, :k], dim=2) / torch.sum(mask[:, :, :k], dim=2)
        return x
    
    def forward_fixed_k_random(self, x, k):
        mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = (mask > 0).float()
        x = self.conv(x) * mask
        t = x[:, :, np.random.choice(range(x.shape[2]), k)] 
        x = torch.sum(t * mask[:, :, :k], dim=2) / torch.sum(mask[:, :, :k], dim=2)
        return x
    
    def conv(self, x):
        x = x[:, x.shape[1] - self.input_dim:]
        for i in range(len(self.layers) - 1):
            x = self.do(self.nonlin(self.layers[i](x)))
        x = self.layers[-1](x)
        return x

def training_epoch(model, dataloader, optimizer, class_weights, device):
    """Train model for one epoch.
    """
    model.train()
    loss_fn = CoxLoss()
    train_loss = []
    for x, time, event in tqdm(dataloader):
        pred = model(x.float().to(device)) # compute mean prediction per slide
        time = time.float().to(device=device, dtype=torch.float32)
        event = event.float().to(device=device, dtype=torch.float32)
        loss = loss_fn(pred, (time[~torch.isnan(pred[:,0])] * (2 * event[~torch.isnan(pred[:,0])] - 1)))
        train_loss += [loss.detach().cpu().numpy()]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(train_loss)
    return train_loss


def evaluate(model, dataloader, projects, device):
    """Evaluate the model on the validation set and return loss.
    """
    model.eval()
    loss_fn = CoxLoss()
    valid_loss = []
    preds = []
    times = []
    events = []
    for x, time, event in dataloader:
        pred = model(x.float().to(device)) # compute mean prediction per slide
        time = time.float().to(device=device, dtype=torch.float32)
        event = event.float().to(device=device, dtype=torch.float32)
        loss = loss_fn(pred, (time[~torch.isnan(pred[:,0])] * (2 * event[~torch.isnan(pred[:,0])] - 1)))
        valid_loss += [loss.detach().cpu().numpy()]
        
        times += [time.detach().cpu().numpy()]
        events += [event.detach().cpu().numpy()]
        preds += [pred.detach().cpu().numpy()]
        
    valid_loss = np.mean(valid_loss)
    preds = np.concatenate(preds)
    times = np.concatenate(times)
    events = np.concatenate(events)
    score = compute_cindex(preds, times, events, projects)
    return valid_loss, score

def compute_cindex(preds, times, events, projects):
    metrics = []
    for project in np.unique(projects):
        times_true = times[projects == project]
        events_true = events[projects == project]
        if len(np.unique(events_true)) > 1:
            y_pred = preds[projects == project, ]
            #print(y_pred)
            cindex = concordance_index(event_times=times_true,
                                        predicted_scores=-preds,
                                        event_observed=events_true,
                                        )
            metrics.append(cindex)
    metrics = np.asarray(metrics)
    return np.mean(metrics)


def predict(model, dataloader, device):
    """Perform prediction on the test set.
    """
    model.eval()
    times = []
    events = []
    preds = []
    for x, time, event in dataloader:
        pred = model(x.float().to(device))
        times += [time.detach().cpu().numpy()]
        events += [event.detach().cpu().numpy()]
        preds += [pred.detach().cpu().numpy()]
    preds = np.concatenate(preds)
    times = np.concatenate(times)
    events = np.concatenate(events)
    return preds, times, events


def fit(model,
        train_set,
        train_set_labels,
        valid_set,
        valid_projects,
        params={},
        optimizer=None,
        scheduler=None,
        test_set=None,
        test_projects=None,
        path=None,
        logdir='./exp'):
    """Fit the model and make prediction on evaluation set.

    Args:
        model (nn.Module)
        train_set (torch.utils.data.Dataset)
        valid_set (torch.utils.data.Dataset)
        valid_projects (np.array): list of integers encoding the projects
            validation samples belong to.
        params (dict): Dictionary for specifying training parameters.
            keys are 'max_epochs' (int, default=200), 'patience' (int,
            default=20) and 'batch_size' (int, default=16).
        optimizer (torch.optim.Optimizer): Optimizer for training the model
        test_set (None or torch.utils.data.Dataset): If None, return
            predictions on the validation set.
        path (str): Path to the folder where th model will be saved.
        logdir (str): Path for TensoboardX.
    """

    if path is not None and not os.path.exists(path):
        os.mkdir(path)

    default_params = {
        'max_epochs': 200,
        'patience': 20,
        'batch_size': 16,
        'num_workers': 0}
    default_params.update(params)
    batch_size = default_params['batch_size']
    patience = default_params['patience']
    max_epochs = default_params['max_epochs']
    num_workers = default_params['num_workers']
    class_weights = default_params['class_weights']
    
    writer = SummaryWriter(log_dir=logdir)

    # SET num_workers TO 0 WHEN WORKING WITH hdf5 FILES
    if 'class_weights_loader' in default_params:
        print("use balanced batches")
        class_weights_loader = default_params['class_weights_loader']
        sample_weights = [class_weights_loader[l] for l in train_set_labels]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, 
                                                                batch_size, replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers, 
                                  sampler = sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, 
                                  shuffle=True, num_workers=num_workers)

    if valid_set is not None:
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if test_set is not None:
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if optimizer is None:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3,
                                     weight_decay=0.)

    epoch_since_best = 0
    start_time = time.time()
    metrics = "cindex"
    if valid_set is not None:
        valid_loss, best = evaluate(
            model, valid_loader, valid_projects, device=params['device'])
        if np.isnan(best):
            best = 0
        print('{}: {:.3f}'.format(metrics, best))
        if test_set is not None:
            preds, times, events = predict(model, test_loader, device=params['device'])
        else:
            preds, times, events = predict(model, valid_loader, device=params['device'])

    try:

        for e in range(max_epochs):

            epoch_since_best += 1
            
            train_loss = training_epoch(model, train_loader, optimizer, class_weights, 
                                        device=params['device'])
            scheduler.step()
            
            dic_loss = {'train_loss': train_loss}

            print('Epoch {}/{} - {:.2f}s'.format(
                e + 1,
                max_epochs,
                time.time() - start_time))
            start_time = time.time()

            if valid_set is not None:
                valid_loss, score = evaluate(
                    model, valid_loader, valid_projects, device=params['device'])
                dic_loss['valid_loss'] = valid_loss
                writer.add_scalars('data/losses',
                                   dic_loss,
                                   e)
                writer.add_scalar('data/metrics', score, e)
                print('loss: {:.4f}, val loss: {:.4f}'.format(
                    train_loss,
                    valid_loss))
                print('{}: {:.3f}'.format(metrics, score))
            else:
                writer.add_scalars('data/losses',
                                   dic_loss,
                                   e)
                print('loss: {:.4f}'.format(train_loss))
            
            
            '''if (e % 5) == 0: # save each 2 echos
                torch.save(model.state_dict(), os.path.join(path, 'weights.') + str(e)+'.pt')
            '''
            
            if valid_set is not None:
                criterion = (score > best)

                if criterion:
                    epoch_since_best = 0
                    best = valid_loss
                    if path is not None:
                        #torch.save(model, os.path.join(path, 'model.pt'))
                        torch.save(model.state_dict(), os.path.join(path, 'weights.pt'))
                    elif test_set is not None:
                        preds, times, events = predict(model, test_loader, device=params['device'])
                    else:
                        preds, times,events = predict(model, valid_loader, device=params['device'])

                if epoch_since_best == patience:
                    print('Early stopping at epoch {}'.format(e + 1))
                    break

    except KeyboardInterrupt:
        pass

    if path is not None and os.path.exists(os.path.join(path, 'model.pt')):
        model = torch.load(os.path.join(path, 'model.pt'))

    elif path is not None:
        torch.save(model.state_dict(), os.path.join(path, 'weights.pt'))
        #torch.save(model, os.path.join(path, 'model.pt'))

    if test_set is not None:
        preds, times, events = predict(model, test_loader, device=params['device'])
    elif valid_set is not None:
        preds, times, events = predict(model, valid_loader, device=params['device'])
    else:
        preds = None
        times = None
        events = None

    writer.close()

    return preds, times, events
