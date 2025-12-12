# Sucrose

A Python package for managing PyTorch experiment projects.

## Overview

Sucrose provides utilities for managing machine learning experiments including configuration management, checkpoint handling, logging, and scenario-based organization of experiments.

## Key Features

### Scenario Management
The core concept in Sucrose is the `Scenario`, which manages file paths, configurations, checkpoints, and logs for your experiments.

### Configuration Management
Sucrose allows users to define and manage experiment configurations using YAML
files, like
```yaml
base:
  model.arg1: 100
  model.arg2: 120
  optim.lr: 0.01
```
Or inherit from a parent domain
```yaml
base/case1:
  model.arg1: 200
```
This allows you to easily create, track and replicate different scenarios
based on a common base configuration.

### Checkpoint and Logging
Automatically save and load model and optimizer states, ensuring that you can
resume training from any point.
It also integrates with TensorBoard for real-time monitoring of training progress
and other metrics.

## Quick start

### Start a scenario

Let the config.yaml file be directly under the project's workspace directory `<workspace>/config.yaml`,
and it has the following content, for example,
```yaml
base:
  model.input_dim: 100
  model.hidden_dim: 100
  model.output_dim: 100

base/case1:
  model.input_dim: 200
```

then run
```python
import sucrose

ssc = sucrose.scenario('path/to/workspace', 'base/case1')
```
to handle a scenario named `base/case1`, which is corresponding to the domain with the same name in the config file.
Every operations below requires a started scenario.

### Access configurations

1. Configurations can be accessed by `__getitem__` like a dict
```python
print(ssc['model.input_dim']) # 200
```

2. Pass configuration key-value pairs into callables like what `functools.partial` does
```python
model = ssc.partial(ExampleModule, "model")()
# Here "model" is the prefix to filter the fields.
# So this is equivalent to:
from functools import partial

model = partial(
  ExampleModule,
  inputdim=200,
  hidden_dim=100,
  output_dim=100
)()
```

### Load state dict

If the NN module, oprimizer, or any other object supporting `load_state_dict` method have been initialized, run

```python
ssc.load_state_dict(model=model, optim=optim, loader_kwds={'weights_only': True})
```

to load the last state dict automatically from
`<workspace>/ckpts/<scenario>/`.
Skipped if not exists.

In this example, the checkpoint file constains a dict, where the state dict for the `model` object is the value of key `"model"`, and the state dict for the `optim` object is the value of key `"optim"`.

### Tensorboard

If tensorboard is needed, run

```python
writer = ssc.start_pytorch_tensorboard()
```

to start a writter on log_dir `<workspace>/logs/<scenario>/`.

### The training loop

Number of epoches finished is marked on the file name of the checkpoint, therefore `sucrose` can do another `N` epoches by running

```python
for epoch in sucrose.epoch_range(N):
    ... # training scripts
```

### Save state dict

Like the `load_state_dict` function, for any object supporting a `state_dict` method, the returned dict can be save to the scenario's checkpoint file by running

```python
ssc.save_state_dict(10, model=model, optim=optim)
```

where the first positional argument `10` is the saving interval which can be any positive integer.
The epoch number increases by 1 every time this function is called, but only 1 file are saved in every 10 times of calling.

In this function, the `save_step` keyword defaults to `True`, means that the number global steps will be recorded in the checkpoint files by default.
Therefore, don't forget to call `sucrose.step()` after `optimizer.step()` to let `sucrose` know the step increasement.

To get the current global step, use
`ssc.num_stes`.
