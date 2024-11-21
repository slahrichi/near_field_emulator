FROM kovaleskilab/ml_basic:v3

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get upgrade -y

RUN pip install kubernetes

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code