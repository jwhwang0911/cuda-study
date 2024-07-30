FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    g++\
    gcc\
    git\
 && rm -rf /var/lib/apt/lists/*

RUN pip install \
    numpy \
    scipy \
    #sets \
    scikit-image \
    ninja   \
    matplotlib  \
    opencv-python
    
VOLUME /Data
VOLUME /Code
VOLUME /Result

WORKDIR /Code

# TODO: pip install  this repository
# -e git+https://github.com/jwhwang0911/py_exr.git@155a62d96fe5d07a2a47206984374a2d5cf119e8#egg=exr\
