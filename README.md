# deep-learning-with-pure-numpy

This repository contains python code using numpy only (not pytorch) to implement various neural net based models.

Any PRs are warmly welcomed!

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
