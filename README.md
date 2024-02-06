# Code to reproduce the ICLR 2024 Paper : "_Implicit regularization of deep residual networks towards neural ODEs_"

`Paper <https://arxiv.org/abs/2309.01213>_` 

## Compat

This package has been developed and tested with `python3.11`. It is therefore not guaranteed to work with earlier versions of python.

## Install the repository on your machine


This package can easily be installed using `pip`, with the following command:

```bash
pip install -e .
```

This will install the package and all its dependencies, listed in `requirements.txt`. To test that the installation has been successful, you can install `pytest` and run the test suite using

```
pip install pytest
pytest
```

## Reproducing the experiments/figures of the paper

### Experiment in Figure 1 - Large-depth limit 

We train the model as described in the paragraph _Large-depth limit_, with reduced number of training epochs. Feel free to adjust it. 

To reproduce the left-hand figure: 

```bash
python finite_training_time_lipshitz.py
```

The plot is saved in the folder figures.

To reproduce the right-hand figure: 

```bash
python finite_training_time_convergence.py
```

The plot is saved in the folder figures.


### Experiment in Figure 2 - Infinite training time

We train the model as described in the paragraph _Long-time limit_, with reduced number of training epochs. Feel free to adjust it. 

To reproduce the figure:

```bash
python infinite_training_time.py
```

### Experiment in Figure 3 - Weights after training on CIFAR 

You can directly visualize the results using the notebsook learned_weights.ipynb using the pretrained models available in the folder checkpoints. 

Alternatively, you can train models from scratch using the following.

```bash
python one_expe_cifar.py --lr 4e-2 --depth 256  --seed 1 --smooth_init SMOOTH_INIT --non_lin NON_LIN
```

Where SMOOTH_INIT is in [True, False]. When True, the weights are initialized smoothly. NON_LIN is the non-linearity used and can be relu, gelu, or linear. 
