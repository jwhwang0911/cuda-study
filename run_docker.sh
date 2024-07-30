docker build . -t cuda-container
DIR=`pwd`
nvidia-docker run \
    --rm \
    -v ${DIR}/Code:/Code \
    --shm-size=8G \
    -it cuda-container /bin/bash;


    # -v ${DIR}/Data:/Data \
    # -v ${DIR}/Result:/Result \
