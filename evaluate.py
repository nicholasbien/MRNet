import argparse
import os
import torch

from sklearn import metrics

from loader import load_data
from model import SeriesModel

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser

def run_model(model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0
    
    for batch in loader:
        if train:
            optimizer.zero_grad()
            
        vol, label = batch
        logit = model.forward(vol)

        loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)
    
    return avg_loss, auc

def evaluate(split, model_path, use_gpu):
    train_loader, valid_loader, test_loader = load_data('data', use_gpu)

    model = SeriesModel()
    if use_gpu:
        model.cuda()
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    else:
        loader = test_loader

    loss, auc = run_model(model, loader)
    print(f'Average {split} loss: {loss:0.4f}')
    print(f'Average {split} AUC: {auc:0.4f}')

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.gpu)
