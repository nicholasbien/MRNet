# knee-mr

# Setup

`bash download.sh`

`conda env create -f environment.yml`

`source activate knee-mr`

## Train

`python train.py --rundir [experiment name] --diagnosis 0 --gpu`

- diagnosis is highest stajduhar diagnosis allowed for negative label (0 = injury task, 1 = tear task)
- arguments saved at `[experiment-name]/args.json`
- prints training & validation metrics (loss & AUC) after each epoch
- models saved at `[experiment-name]/[val_loss]_[train_loss]_epoch[epoch_num]`

## Evaluate

`python evaluate.py --split [train/valid/test] --diagnosis 0 --model_path [experiment-name]/[val_loss]_[train_loss]_epoch[epoch_num] --gpu`

- prints loss & AUC
