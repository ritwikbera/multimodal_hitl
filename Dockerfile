# Filename: Dockerfile 
# FROM ritwikbera/multimodal:latest
FROM pytorch/pytorch:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    hdf5-tools \
    postgresql \
    postgresql-contrib \
    postgresql-server-dev-all \
    libsm6 \
    libxext6 \ 
    libxrender-dev \ 
    ffmpeg \
    gcc \
    unzip \
    nano \
    wget
COPY requirements.txt /workspace/ 
RUN conda install -y pip
RUN /opt/conda/bin/pip install -r requirements.txt
RUN wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
RUN unzip *.zip

