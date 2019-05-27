FROM nvcr.io/nvidia/tensorflow:18.04-py3


# Install python3.6 and make an alias to run it
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
# RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3.6-dev
