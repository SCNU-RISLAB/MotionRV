python -m torch.distributed.launch --nproc_per_node=4 train.py \
                                                        --dataset /path/to/dataset   \
                                                        --arch_cfg ./train_yaml/mos_coarse_stage.yml    \
                                                        --data_cfg ./config/labels/semantic-kitti-mos.raw.yaml  \
                                                        --log log/train_v1  \
                                                        --name your_projectname \
                                                        --gpus '0,1,2,3'    \
                                                        # -p /path/to/pretrainmodel





