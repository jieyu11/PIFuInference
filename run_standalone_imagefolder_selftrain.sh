#!/bin/bash

#
# need to update the parameters before running!
#

dockertag="3af2dc_jie"
dockerimage=bgsegmentation_selftrain:${dockertag}
imgext="png"
inputdir=${HOME}/Workarea/analysis/PIFu/images/source/EricFootball
outputdir=${HOME}/Workarea/analysis/PIFu/images/mask/EricFootball
INPUT_PAT="/work/data/inputs/*.${imgext}"
OUTPUT_DIR="/work/data/outputs"

nvidia-docker run --rm \
    -v $inputdir:/work/data/inputs \
    -v $outputdir:/work/data/outputs \
    $dockerimage \
    /bin/bash -c "export PYTHONPATH=$PYTHONPATH:/work/tpu/models:/work/opt;\
    echo running docker; \
    python tpu/models/official/detection/seg.py \
        --image_file_pattern="${INPUT_PAT?}" \
        --output_folder="${OUTPUT_DIR?}" \
    "
