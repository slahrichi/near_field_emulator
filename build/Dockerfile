FROM kovaleskilab/ml_basic:v4

SHELL ["/bin/bash", "-c"]

# Install dependencies and download kubectl
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && mv kubectl /usr/local/bin/kubectl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install complexPyTorch

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code