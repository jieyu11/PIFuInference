#!/bin/bash

srcdir=stereo_rectified
images=$PWD/images/source/$srcdir
rectdir=$PWD/images/rect/$srcdir
output=$PWD/outputs/$srcdir
mkdir -p $output
docker run -it --rm --gpus all \
	-v $images:/data/images \
	-v $rectdir:/data/rect \
	-v $output:/output \
	pifu \
	/bin/bash -c "cd pifuhd; mkdir -p $srcdir; \
	cp /data/images/* /data/rect/* $srcdir; \
	python -m apps.simple_test -r 512 --use_rect -i $srcdir -o /output; \
	"