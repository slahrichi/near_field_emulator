FROM kovaleskilab/ml_basic:v1

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get upgrade -y

RUN pip install scikit-learn

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code