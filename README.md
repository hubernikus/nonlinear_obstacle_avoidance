# Nonlinear Obstacle Avoidance
---
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
---

Obstacle avoidance around linear and nonlinear dynamics.

# Installation
To setup got to your install/code directory, and type:
```sh
git clone https://github.com/hubernikus/nonlinear_obstacle_avoidance.git
```
(Make sure submodules are there if `various_tools` library is not installed. To initialize submodules after cloning use `git submodule update --init --recursive`.
To update all submodules `git submodule update --recursive`

## Custom Environment
Choose your favorite python-environment. I recommend to use [virtual environment venv](https://docs.python.org/3/library/venv.html).
Setup virtual environment (use whatever compatible environment manager that you have with Python >=3.9).

``` bash
python3.10 -m venv .venv
```
with python -V >= 3.9

Activate your environment
``` sh
source .venv/bin/activate
```

### Setup Dependencies
Install all requirements:
``` bash
pip install -r requirements.txt && pip install -e .
```

Make sure to install submodules:
``` bash
mkdir libraries_source && cd libraries_source
git clone https://github.com/epfl-lasa/dynamic_obstacle_avoidance.git
cd dynamic_obstacle_avoidance && pip install -r requirements.txt && pip install -e . && cd ..
git clone https://github.com/hubernikus/various_tools
cd various_tool && pip install -r requirements.txt && pip install -e . && cd ..
```




