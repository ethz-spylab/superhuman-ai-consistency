# rl-testing-experiments

## Table of contents
1. [Setup](#setup)
    - [Creating virtual environment](#creating-virtual-environment)
    - [Installing the package](#installing-the-package)
    - [Setting up a Leela Chess Zero instance](#setting-up-a-leela-chess-zero-instance)
    - [Downloading a Leela Chess Zero weight file](#downloading-a-leela-chess-zero-weight-file)
    - [Configuration file for Leela Chess Zero instance](#configuration-file-for-leela-chess-zero-instance)
    - [Configuration file for data](#configuration-file-for-data)
2. [Reproducing the experiments](#reproducing-the-experiments)
    - [Prerequisites](#prerequisites)
    - [Running the experiments](#running-the-experiments)

## Setup
### Creating virtual environment
This project was developed using Python 3.8. It is recommended to install this repository in a virtual environment.
Make sure you have [Python 3.8](https://www.python.org/downloads/release/python-380/) installed on your machine. Then, initialize your virtual environment in this folder, for example via the command 
```bash
python3.8 -m venv .venv
```
You can activate the virtual environment via the command
```bash
source .venv/bin/activate
```

### Installing the package
The package can be installed via the command
```bash
pip install -e .
```

### Setting up a Leela Chess Zero instance
In order to run the experiments you need access to an instance of [Leela Chess Zero](https://github.com/LeelaChessZero/lc0). You can either install it on the same machine you want to run the experiments on, or on a remote machine to which you have SSH access. Our experiments use the `release/0.29` version, compiled from source and with GPU support enabled.

### Downloading a Leela Chess Zero weight file
All weight files can be found on [this website](https://training.lczero.org/networks/?show_all=1). For our experiments we used the network with ID `807785`.

### Configuration file for Leela Chess Zero instance
Configurations for the Leela Chess Zero instance must be stored in a configuration file in the `experiments/configs/engine_configs` folder. Each config file has to contain information about where to find the installed Leela Chess Zero instance and which configuration parameters should be set. See as example the following config:
```python
[General]
# 'engine_type' Must be either 'local_engine' or 'remote_engine'
engine_type = remote_engine 
engine_path = /path/to/lc0/on/the/machine/where/it/has/been/installed
network_base_path = /path/to/folder/where/weightfiles/are/stored

# Leela Chess Zero configs used for experiments
# See https://github.com/LeelaChessZero/lc0/wiki/Lc0-options
# for a list of all options
[EngineConfig]
Backend = cuda-fp16
VerboseMoveStats = true
SmartPruningFactor = 0
Threads = 1
TaskWorkers = 0
MinibatchSize = 1
MaxPrefetch = 0
NNCacheSize = 200000
TwoFoldDraws = false

# For how long Leela Chess Zero should evaluate a position
# See https://python-chess.readthedocs.io/en/latest/engine.html#chess.engine.Limit
# for a list of options.
[SearchLimits]
nodes = 400


# The following parameters are only required if you installed
# Leela Chess Zero on a different machine than the one you're using
# to run the experiments
[Remote]
remote_host = uri.of.server.com
remote_user = username
password_required = True
```

### Configuration file for data
In addition to the engine config, our experiments also require a config file containing information where to find the input data (usually chess positions). This configuration file must be stored in the `experiments/configs/data_generator_configs` folder. We support either a simple `.txt` file containing a list of [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)s, or a `.pgn` database containing games in [PGN](https://en.wikipedia.org/wiki/Portable_Game_Notation). All data files should be stored in the  `data` folder. Alternatively, you can also set the `DATASET_PATH` environment variable in which case the data-files are expected to be stored in `DATASET_PATH/chess-data`. See as example the following config:
```python
[General]
# 'data_generator_type' must be either 'fen_database_board_generator' 
# (for a simple text file containing one fen per row) or 
# 'database_board_generator' (for a database file in .pgn format)
data_generator_type = fen_database_board_generator

[DataGeneratorConfig]
database_name = name_of_data_file.txt
open_now = True
```

## Reproducing the experiments
### Prerequisites
- Leela Chess Zero instance installed and configured as described above
- Data file containing chess positions stored in `data` folder. The specific chess positions used in our experiments can be extracted from the result files in the `experiments/results/final_data` folder.

### Running the experiments
All experiments can be run in a two-step process. First, the main experiment file is run. This file handles everything from loading the data, writing results, and coordinating the distributed queues. In a second steps, one or several workers are started. Each worker runs a Leela Chess Zero instance and evaluates positions provided by the main experiment file.

For the forced-move and the recommended-move experiments, the main experiment file can be run via the command
```bash
python experiments/recommended_move_invariance_testing.py --engine_config_name your_engine_config.ini --data_config_name --your_data_config.ini --num_positions number_of_positions_to_evaluate
```

For the board-mirroring and board-transformation experiments, the main experiment file can be run via the command
```bash
# '--transformations' must be a subset of [rot90, rot180, rot270, flip_diag, flip_anti_diag, flip_hor, flip_vert, mirror]
python experiments/transformation_invariance_testing.py --engine_config_name your_engine_config.ini --data_config_name --your_data_config.ini --num_positions number_of_positions_to_evaluate --transformations a list of transformations to apply to the board
```

For the evolutionary algorithm experiments, the main experiment file can be run via the command
```bash
python experiments/evolutionary_algorithms/evolutionary_algorithm_distributed_oracle_queries_async.py 
```

For all experiments, a worker can be started via the command
```bash
python rl_testing/engine_generators/worker.py --engine_config_name your_engine_config.ini --network_name name_of_weight_file
```