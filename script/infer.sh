#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 && python3 ./infer.py \
                                        -d /path/to/dataset \
                                        -m /path/to/model \
                                        -l log/infer_v1 \
                                        -s valid  \
