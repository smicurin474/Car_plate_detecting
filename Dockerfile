FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

WORKDIR /yolo9

COPY requirements_yolo9_for_docker.txt config_dataset.yaml *.py /yolo9/ 

RUN pip install -r requirements_yolo9_for_docker.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
