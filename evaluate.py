import argparse
import os
import torch

from sklearn import metrics

from loader import load_data
from model import SeriesModel
from train import run_model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser

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
