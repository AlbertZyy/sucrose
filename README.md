# Sucrose

Project file management tool for PyTorch experiments.

## Quick start

### Start a project

Let the relative path to the project's config.json file be

`<relative_path>/logs/<project_name>/config.json`,

then run
```python
import sucrose

sucrose.start_project('relative_path', 'project_name')
```
to start a project named `project_name`.
Every operations below requires a started project.

### Load state dict

If the NN module, oprimizer, and other object supporting `load_state_dict` method have been initialized, run

```python
sucrose.load_state_dict(model=model, optim=optim, loader_kwds={'weights_only': True})
```

to load the last state dict automatically from
`<relative_path>/ckpts/<project_name>/`.
Skipped if not exists.

In this example, the checkpoint file constains a dict, where the state dict for the `model` object is the value of key `"model"`, and the state dict for the `optim` object is the value of key `"optim"`.

### Tensorboard

If tensorboard is needed, run

```python
writer = sucrose.start_pytorch_tensorboard()
```

to start a writter on log_dir `<relative_path>/logs/<project_name>/`.

### The training loop

Number of epoches finished is marked on the file name of the checkpoint, therefore `sucrose` can do another `N` epoches by running

```python
for epoch in sucrose.epoch_range(N):
    ... # training scripts
```

### Save state dict

Like the `load_state_dict` function, for any object supporting a `state_dict` method, the returned dict can be save to the project's checkpoint file by running

```python
sucrose.save_state_dict(10, model=model, optim=optim)
```

where the first positional argument `10` is the saving interval which can be any positive integer.
The epoch number increases by 1 every time this function is called, but only 1 file are saved in every 10 times of calling.

In this function, the `save_step` keyword defaults to `True`, means that the number global steps will be recorded in the checkpoint files by default.
Therefore, don't forget to call `sucrose.step()` after `optimizer.step()` to let `sucrose` know the step increasement.

To get the current global step, use
`sucrose.get_current_step()`.

### More

These operations called from the sucrose "name space" act in the context of the latest-started project.
Users can handle the project explicitly after fetching the Project object, which is returned by `sucrose.start_project`.
```python
proj = sucrose.start_project(...)
```
Methods like `proj.load_state_dict`, `proj.save_state_dict` and `proj.step` are equivalent to the version in `sucrose` if `proj` is the latest-started project.