import argparse
import os
import numpy as np
import torch

from sklearn import metrics

from loader import load_data
from model import SeriesModel
from train import run_model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    return parser

def evaluate_ensemble(split, model_dir, diagnosis, use_gpu):
    all_preds = []
    for model_path in os.listdir(model_dir):
        preds, labels = evaluate(split, os.path.join(model_dir, model_path), diagnosis, use_gpu)
        all_preds.append(preds)
    preds = np.array(all_preds)
    labels = np.array(labels)

    preds = np.mean(preds, axis=0)
    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)
    print(f'ensemble {split} AUC: {auc:0.4f}')
    

def evaluate(split, model_path, diagnosis, use_gpu):
    train_loader, valid_loader, test_loader = load_data('data', diagnosis, use_gpu)

    model = SeriesModel()
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    else:
        loader = test_loader

    loss, auc, preds, labels = run_model(model, loader)
    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    if args.ensemble:
        evaluate_ensemble(args.split, args.model_path, args.diagnosis, args.gpu)
    else:
        evaluate(args.split, args.model_path, args.diagnosis, args.gpu)

