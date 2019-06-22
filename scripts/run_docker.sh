#!/bin/bash
NV_GPU=$1 nvidia-docker run -tid --rm --shm-size=20g --ulimit memlock=-1 -v /home/paulomann/workspace:/workspace pmann:py3.6
