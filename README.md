# knee-mr

# Setup

- extract all `.7z` volumes from http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/volumetric_data/ and place enclosed .pck files in in folders named 'vol01'-'vol10' in root directory
- download http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/data/metadata.csv and place in root directory

## Train

`python train.py --rundir [experiment name] --diagnosis 0 --gpu`

- diagnosis is highest stajduhar diagnosis allowed for negative label (0 = injury task, 1 = tear task)
- arguments saved at `[experiment-name]/args.json`
- prints training & validation metrics (loss & AUC) after each epoch
- models saved at `[experiment-name]/[val_loss]_[train_loss]_epoch[epoch_num]`

## Evaluate

`python evaluate.py --split [train/valid/test] --diagnosis 0 --model_path [experiment-name]/[val_loss]_[train_loss]_epoch[epoch_num] --gpu`

- prints loss & AUC
