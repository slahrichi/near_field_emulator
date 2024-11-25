FROM kovaleskilab/ml_basic:v4

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install kubernetes

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code