#!/bin/bash
python3 ./utils/evaluate_mos.py \
        -d /path/to/dataset \
        -p /path/to/label \
        -dc config/labels/semantic-kitti-mos.raw.yaml \
        -s valid
