# Use the NVIDIA CUDA base image (with the required CUDA version)
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install required packages
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    vim \
    htop \
    wget \
    bzip2 \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH="/opt/conda/bin:$PATH"

# Create a new environment with Python 3.10 and install Conda packages
RUN conda create -n meep-env python=3.10 -y && \
    conda install -n meep-env -c conda-forge \
    pymeep pymeep-extras \
    matplotlib pandas ipython scipy tqdm pyyaml -y

# Install PyTorch with GPU support (compatible with CUDA 11.7)
RUN conda install -n meep-env -c pytorch \
    pytorch torchvision torchaudio cudatoolkit=11.7 -y

# Install PyTorch Lightning and segmentation-models-pytorch
RUN conda install -n meep-env -c conda-forge \
    pytorch-lightning segmentation-models-pytorch -y

# Install GeomLoss using pip inside the Conda environment
RUN /opt/conda/bin/conda run -n meep-env pip install geomloss

# Ensure conda environment is activated when the container starts
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate meep-env" >> /root/.bashrc

# Ensure conda environment is activated and geomloss is available when starting the container
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate meep-env && bash"]

WORKDIR /develop/code
WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/resims