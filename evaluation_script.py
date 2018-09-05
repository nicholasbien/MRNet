import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from sklearn import metrics
from torch.autograd import Variable

from loader import load_data
from model import SeriesModel
from train import run_model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    return parser

def evaluate(split, model_path, diagnosis):
    train_loader, valid_loader, test_loader = load_data(diagnosis)

    model = SeriesModel()
    state_dict = torch.load(model_path, map_location=(None if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    loss, auc, preds, labels = run_model(model, loader)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    if torch.cuda.is_available():
        torch.device('cuda')
    evaluate(args.split, args.model_path, args.diagnosis)
