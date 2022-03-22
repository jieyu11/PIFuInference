#
# generating dockerfile according to: https://github.com/shunsukesaito/PIFu
# as well as: https://github.com/facebookresearch/pifuhd
#
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
ENV WORK_ROOT /work
RUN mkdir $WORK_ROOT
WORKDIR $WORK_ROOT

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         sudo

# needed by detectron pytroch implementation
RUN apt-get update && apt-get install -y \
    libsm-dev \
    libxrender-dev \
    libxext-dev \
    libgl1-mesa-glx \
	libopenexr-dev \
	freeglut3-dev \
	libgl1-mesa-dri \
	libegl1-mesa \
	libgbm1 \
	wget

ENV PATH=$PATH:/opt/conda/bin:/opt/conda/envs/pytorch/bin
RUN conda install -c conda-forge pyembree
RUN conda install pillow
RUN conda install scikit-image
RUN conda install -c menpo opencv
RUN pip install tqdm trimesh pyopengl

RUN git clone https://github.com/shunsukesaito/PIFu
RUN cd PIFu && ./scripts/download_trained_model.sh
RUN git clone https://github.com/facebookresearch/pifuhd.git
COPY pifuhd_download_trained_model.sh $WORK_ROOT/pifuhd/scripts/pifuhd_download_trained_model.sh
RUN cd pifuhd && ./scripts/pifuhd_download_trained_model.sh
# for getting rect file for pifuhd
RUN cd pifuhd && git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
RUN cd pifuhd/lightweight-human-pose-estimation.pytorch && \
    wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
COPY pifuhd_*.py $WORK_ROOT/pifuhd/scripts/
COPY pifuhd_get_rect.py $WORK_ROOT/pifuhd/lightweight-human-pose-estimation.pytorch/
# coco api needed for rect calculation
RUN conda install -c conda-forge pycocotools 
# pip install git+https://github.com/SShajmoha/cocoapi.git#subdirectory=PythonAPI

CMD ["/bin/bash"]