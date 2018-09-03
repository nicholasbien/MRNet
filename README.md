# knee-mr

# Setup

- extract all volumes from http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/ and in folders 'vol01'-'vol10' in root directory

## Train

`python train.py --rundir [experiment name] --diagnosis 0 --gpu`

- arguments saved at [experiment-name]/args.json
- models saved at [experiment-name]/[val_loss]_[train_loss]_epoch[epoch_num]
- prints training & validation metrics (loss & AUC) after each epoch

## Evaluate

`python evaluate.py --split [train/valid/test] --model_path my-experiment/val0.1609_train0.0595_epoch15 --diagnosis 0 --gpu`

- prints loss & AUC
