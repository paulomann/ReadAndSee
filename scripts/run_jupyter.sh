#!/bin/bash
#NV_GPU=$1 nvidia-docker run -tid --rm --shm-size=20g --ulimit memlock=-1 -v /home/paulomann/workspace:/workspace -p $2:$2 pmann:py3.6
#containerId=$(docker ps | grep 'pmann:py3.6' | awk '{ print $1 }')
#docker exec -ti $containerId bash -c 'cd /workspace/ReadOrSee;source env/bin/activate;cd notebooks;jupyter notebook --ip=0.0.0.0 --no-browser --port '"$1"' --allow-root'
jupyter notebook --ip=0.0.0.0 --no-browser --port '"$1"' --allow
