# knee-mr

# Setup

- extract all volumes from http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/ and place in data/volumes
- save pretrained models at models/[view]-[task]

## Train

`python train.py --rundir [experiment-name] --model_path models/sagittal-acl`

OR

`export CUDA_VISIBLE_DEVICES=0; python train.py --rundir [experiment-name] --model_path models/sagittal-acl --gpu`

- arguments saved at [experiment-name]/args.json
- models saved at [experiment-name]/[val_loss]_[train_loss]_epoch[epoch_num]
- prints training & validation metrics (loss & AUC) after each epoch

## Evaluate

`python evaluate.py --split [train/valid/test] --model_path models/sagittal-acl`

OR

`export CUDA_VISIBLE_DEVICES=0; python evaluate.py --split [train/valid/test] __model_path [experiment_name]/[model_path]`

- prints loss & AUC
