# causal-safe-wm-on-sgf

## Setup

### Prerequisites
* Python >=3.8 <3.11 (tested on 3.10.18)
* [uv](https://github.com/astral-sh/uv) (v0.7.20)

Install `graphviz` for visualization graphs.

To run experiments locally, give the following a try:

```bash
uv venv --python 3.10.16
source .venv/bin/activate

wget https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip
unzip main.zip
nano safety-gymnasium-main/pyproject.toml
```

Modify the dependencies of package `gymnasium-robotics` from `(==1.2.2)` to `(>=1.2.2,<=1.2.3)` under `safety-gymnasium-main/pyproject.toml`. This will allow the use of `numpy (>=1.24.1)`. (Note: the dependencies are not recommended by official repo though this modification has passed CI/tests with `gymnasium-robotics==1.2.3`)

```toml
dependencies = [
    "gymnasium == 0.28.1",
    "gymnasium-robotics (>= 1.2.2, <= 1.2.3)", # this line
    ...
]
```

Install `safety_gymnasium` v1.2.0 together with ```uv```.

```bash
uv lock && uv sync --group safetygym
```


## Training

To start a training run, execute the following command:

```bash
$ python src/main.py --device cuda:0 --game Breakout --project sgf --config configs/default.yaml --amp --compile --seed 1
```

The training script will log all relevant information to Weights & Biases.

To change the hyperparameters, create a copy of the `default.yaml` file and adjust the values as needed.

You can speed up training by setting `--agent_eval final`, which will only evaluate the agent at the end of training. To train an additional decoder for debugging, set `--wm_eval decoder`.

## Acknowledgements

### Simple, Good, Fast (SGF)

Official code of the paper [Simple, Good, Fast: Self-Supervised World Models Free of Baggage](https://openreview.net/forum?id=yFGR36PLDJ).  
Published as a conference paper at ICLR 2025.

If you find this code or paper helpful, please reference it using the following citation:

```
@inproceedings{
  robine2025simple,
  title={Simple, Good, Fast: Self-Supervised World Models Free of Baggage},
  author={Jan Robine and Marc H{\"o}ftmann and Stefan Harmeling},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=yFGR36PLDJ}
}
```

