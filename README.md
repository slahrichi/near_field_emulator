# Near Field Wave Emulation

Exploration of deep learning models for emulating wavefront propagation and response from metasurfaces. Supports two distinct problems: Time-series networks for modeling propagation directly, and MLP-based methods for mapping metasurface parameters to downstream fields. Currently supports the following architectures: MLP, CVNN, LSTM, Autoencoder-LSTM, and Autoencoder-ConvLSTM.

## Prerequisites

- Docker
- Python 3.8+
- CUDA-capable GPU (for training)
- Access to specific Kubernetes namespaces and PVCs
- deploy key associated with the remote repository

## Description of Contents

- `kube/` : contains files for loading, configuring, and launcing Kubernetes jobs. This includes scripts for training, evaluation, and analyzing results.
  - **launch_training.py** : This file generates a useable YAML file for a training instance; similar files exist for evaluation and data manipulation
  - **save_eval_plots.py** : necessary helper for saving evaluation plots from cloud storage to disk for viewing
- `core/` : contains various key files such as the models and dataloader.
  - **WaveMLP.py** : MLP implementation
  - **WaveModel.py** : Various model implementations utilizing LSTM in some capacity
  - **datamodule.py** : Master file handling dataloading for both problems
  - **autoencoder.py** : autoencoder implementation
  - **ConvLSTM.py** : Convolutional LSTM implementation
  - **CVNN.py** : implementation of Complex-Valued NN, activations
- `sim/` : contains files required for dataset generation/meep simulations
  - (incomplete)
- `evaluation/` : contains files used for eval pipeline
  - **eval_model.py** : Primary file for evaluation process
  - **evaluation.py** : File containing all plotting, measuring, etc. methods
  - **inference.py** : Contains additional evaluation functions for real-time analysis
- `utils/` : Helper files
  - **compile_data.py** : Data after preprocessing is saved as .pkl files, This file contains a function that compiles it into a single pytorch file for use with the model.
  - **mapping.py** : Primarily contains the functions `cartesian_to_polar` and its vice versa counterpart for converting between polar and cartesian before displaying plots
  - **model_loader.py** : This wrapper dynamically determines which model we're using
  - **parameter_manager.py** : Formats and organizes `config.yaml` contents in a manner consistent with the needs of files which reference them such as the model, data module, etc.
  - **visualize.py** : contains implementation of plotting function that shows animation of field progression - can be seen in the presentation
- `build/` : Contains the Dockerfile for building the Docker container.
- `config.yaml` : Specifies all aspects of training, evaluation, setup, everything.
- `main.py` : The driver file, refers to `config.yaml` to determine what to do when a process is started.
- `train.py` : The training process

## Configuration

The `config.yaml` file controls all aspects of training and evaluation. Key parameters include:

- `experiment`: 
  - `0`: Train network
  - `1`: Run evaluation
  - `2`: Data compilation
  - `3`: Load results

- `arch`:
  - `0`: Dual MLPs (separate real/imaginary)
  - `1`: CVNN (complex-valued)
  - `2`: LSTM
  - `3`: ConvLSTM
  - `4`: Autoencoder-Wrapped LSTM
  - `5`: Autoencoder-Wrapped ConvLSTM
  - `6`: LSTM with specific modes encoded first
  - `7`: Autoencoder

- `model_id`: Unique identifier for the model
- `batch_size`: Training batch size
- `num_epochs`: Maximum training epochs
- `learning_rate`: Initial learning rate
- `cross_validation`: whether to do cross validation or not
- `kube`: specifies parameters used to create kubernetes YAMLs
- additional configurations for architecture, training, etc.
- these addl parameters are explained where necessary within the file itself

## Running the Code

### Docker

A docker container must be created to do any kind of running with this code. To build, execute the following when starting from the root directory:

```
cd build
docker build -t kovaleskilab/ml_basic:v4 .
```

From there, we want to run the container but its critical that we mount it utilizing the following scheme to ensure the code exists within the container:

```
sudo docker run \
-v /path/to/data:/develop/data \
-v /path/to/results:/develop/results \
-v {parent directory containing near_field_emulator}:/develop/code \
  kovaleskilab/ml_basic:v4'
```


### Basic Usage

From within the Docker container after launching (should be at **/develop/code**):

```
cd near_field_emulator
python3 main.py --config config.yaml
```

### Training

1. Set `experiment: 0` in config.yaml
2. Choose architecture with `arch` parameter
3. Set desired `model_id` and hyperparameters
4. Run `python3 main.py --config config.yaml`

### Evaluation

1. Set `experiment: 4` in config.yaml
2. Set other desired parameters
3. Run `python3 main.py --config config.yaml`

Evaluation outputs (saved to results directory):
- Predicted vs actual field distributions - magnitude and phase
- Error metrics - resub and validation
- Flipbooks of wave propagation

## Data Generation

### Running Meep Simulations

(Incomplete)

To generate training data using Meep:

```
python3 sim/run_sim.py -neighbor_index <index>
```

This simulates a metasurface with specified radii configurations and saves:
- Near-field distributions
- Far-field patterns
- Epsilon data
- Field slices

### Data Preprocessing

(Incomplete)

After generating raw simulation data:

```
python3 sim/preprocess_data_dip.py
```

This creates preprocessed datasets containing:
- Normalized field components
- Phase information
- Derivative calculations

## Kubernetes Jobs

The `kube/` directory contains job templates for distributed training. Creating templates is as simple as setting parameters in `config.yaml` and running the following:

```
python3 -m kube.launch_training # (or launch_evaluation)
```

After the above is ran a complete YAML configuration file is generated and placed in `kube/kube_jobs`. Starting a training job, or evaluation job, or whatever process we're doing then requires running the following from a terminal (example is for convlstm training):

```
kubectl apply -f kube/kube_jobs/convlstm-training.yaml
```

**Note:** Running these jobs requires access to specific Kubernetes namespaces and PVCs, as well as the private github repo hosting this code. External users will need to modify configurations for their environment and create a ssh key to allow pulling the code within a kube pod.