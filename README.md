# deep-learning-with-pure-numpy

This repository contains python code using numpy only (not pytorch) to implement various neural net based models.

Any PRs are warmly welcomed!

## Setting up environment

We use [poetry](https://github.com/python-poetry/poetry) to manage dependencies of repository.

Use poetry with version `2.1.1`.

```shell
$ poetry --version
Poetry (version 2.1.1)
```

Python version should be `3.11.x`.

```shell
$ python --version
Python 3.11.11
```

If python version is lower than `3.11`, try installing required version using `pyenv`.

Create virtual environment.

```shell
$ poetry env activate
```

If your global python version is not 3.11, run following command.

```shell
$ poetry env use python3.11
```

You can check virtual environment path info and its executable python path using following command.

```shell
$ poetry env info
```

After setting up python version, just run following command which will install all the required packages from `poetry.lock`.

```shell
$ poetry install
```

### Note

If you want to add package to `pyproject.toml`, please use following command.

```shell
$ poetry add "package==1.0.0"
```

Then, update `poetry.lock` to ensure that repository members share same environment setting.

```shell
$ poetry lock
```

## Setting up git hook

Set up automatic linting using the following commands:
```shell
# This command will ensure linting runs automatically every time you commit code.
pre-commit install
```

## Why made this repository?
- To start learning how to train neural network quickly, machine learning frameworks such as pytorch or tensor flow are best options to do this.
- However, it is difficult to understand what is happening inside the neural networks (forward / backward propagation, etc..)
- This repository will help you to fully understand how neural network updates its parameters.

## Rules
If you want to contribute to this repository, please follow below rules.
- Implementation should contain both forward and backward method.
- Use base class when creating loss / module / neural network
  - base loss class: `src/loss/base.py`
  - base module class: `src/modules/base.py`
  - base neural network class: `src/nn/base.py`
- Add test code comparing your implementation with pytorch results. Refer to `test/` directory for more details.

## Current implementation
|Category|File path|
|------|---|
|loss|`src/loss/classification.py`|
|loss|`src/loss/regression.py`|
|modules|`src/modules/linear.py`|
|modules|`src/modules/convolution.py`|
|neural network|`src/nn/mlp.py`|
|neural network|`src/nn/single_cnn.py`|

## Examples

You can construct multi-layer perceptron or convolution neural network. The examples are in `examples/` directory.

## How to test
```bash
$ pytest
```
