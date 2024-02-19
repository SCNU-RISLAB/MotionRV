#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 && python3 ./train_2stage.py \
                                        -d /path/to/dataset \
                                        -ac ./train_yaml/mos_pointrefine_stage.yml \
                                        -dc ./config/labels/semantic-kitti-mos.raw.yaml \
                                        -l  log/train_v2 \
                                        -n your_projectname \
                                        -p /path/to/one-satge-modelpath
