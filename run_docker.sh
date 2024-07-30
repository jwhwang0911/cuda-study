docker build . -t cuda-container
DIR=`pwd`
nvidia-docker run \
    --rm \
    --shm-size=8G \
    -it cuda-container /bin/bash;


    # -v ${DIR}/Data:/Data \
    # -v ${DIR}/Code:/Code \
    # -v ${DIR}/Result:/Result \
