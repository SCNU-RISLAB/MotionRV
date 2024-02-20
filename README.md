# MotionRV
An Accurate Moving Object Segmentation Network for LiDAR Range-View

# How to use
[MotionRV_1stage](https://github.com/SCNU-RISLAB/MotionRV/MotionRV_1stage) is for one-stage training. After one-stage training, please put the one-stage pretrained weight into the [MotionRV_2stage](https://github.com/SCNU-RISLAB/MotionRV/MotionRV_2stage) for two-stage training.

## Dataset
Download SemanticKITTI dataset from [SemanticKITTI.](http://www.semantic-kitti.org/dataset.html#download)

## pretrained weight
Our pretrained weight (training in one-stage for the best in the validation seq08, with the IoU of 73.88%) can be downloaded from [OneDrive.](https://1drv.ms/f/s!Ak9z9MBScOueg_Mu-_8SAE_QGzjLbQ?e=Khpkes)
Our pretrained weight (training in two-stage for the best in the validation seq08, with the IoU of 76.67%) can be downloaded from [OneDrive.](https://1drv.ms/f/s!Ak9z9MBScOueg_Myu-jDR0pe5x2JFA?e=anHAzo)

## Pretreatment
Run [auto_gen_residual_images.py](MotionRV_1stage/utils/auto_gen_residual_images.py) to bulid residual images(num_last_n=8), and check that the path is correct before running.

## Environment
`Linux:`
Ubuntu 18.04, CUDA 11.1+Pytorch 1.7.1
Use conda to create the conda environment and activate it:
```shell
cd MotionRV_1stage
conda env create -f environment.yml
conda activate motionrv
```
`TorchSparse:`
```shell
sudo apt install libsparsehash-dev 
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## Training
### one-stage training
Check the path correctly in [ddp_train.sh](MotionRV_1stage/script/ddp_train.sh), and run it to train with 4 GPUs(Change according to actual situation):
```shell
cd MotionRV_1stage
bash script/ddp_train.sh
```
### two-stage training
Check the path correctly in [train_2stage.sh](MotionRV_2stage/script/train_2stage.sh), and run it to train with single GPU.
```shell
cd MotionRV_2stage
bash script/train_2stage.sh
```

## Inferring
Check the path correctly in [infer.sh](MotionRV_1stage/script/infer.sh), and run it to infer the predictive labels.
```shell
cd MotionRV_1stage / cd MotionRV_2stage
bash script/infer.sh
```

## Evaluation
Check the path correctly in [eval.sh](MotionRV_1stage/script/eval.sh), and run it to evaluate and get IoU which can copy in the paper.
```shell
cd MotionRV_1stage / cd MotionRV_2stage
bash script/eval.sh
```
You can also use our pretrained weight to validate its MOS performance.
