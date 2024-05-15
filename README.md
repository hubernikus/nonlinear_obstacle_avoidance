# Nonlinear Obstacle Avoidance
---
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)
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

Make sure to install the required submodules:
``` bash
pip install "git+https://github.com/hubernikus/various_tools.git"
pip install "git+https://github.com/hubernikus/dynamic_obstacle_avoidance.git"
```

If you need to modify certain elements of the software, its advised to clones these repositories to better browser them. 


## Citing Repository
If you use this repository in a scientific publication, we would appreciate citations to the following paper:

Huber, Lukas. _Exact Obstacle Avoidance for Robots in Complex and Dynamic Environments Using Local Modulation._ No. 10373., EPFL, 2024.

Bibtex entry:
``` bibtex
@phdthesis{huber2024exact,
  title={Exact Obstacle Avoidance for Robots in Complex and Dynamic Environments Using Local Modulation},
  author={Huber, Lukas},
  year={2024},
  month={April},
  address={Lausanne, Switzerland},
  school={EPFL},
  type={PhD thesis}
}
```
