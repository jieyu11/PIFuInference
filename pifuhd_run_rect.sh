#!/bin/bash

# in the docker container
cd pifuhd/lightweight-human-pose-estimation.pytorch/
python pifuhd_get_rect.py -i /data/images/stereo_rectified -o /data/rect/stereo_rectified
