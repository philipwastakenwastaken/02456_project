# 02456 Deep Learning - Project

## Set train or evaluate
```python src/main.py session=train```
```python src/main.py session=evaluate```

## Examples of how to modify other hyperparameters
Set session mode to train. Overwrite learning rate, batch size and number of episodes set in config files.
```python src/main.py session=train model.learning_rate=0.001 train.batch_size=32 train.num_episodes=100```

Change environment
```python src/main.py session=train environment=mspacman```

```python src/main.py session=train environment=asterix```

## Changing wrapper (and model used)
```python src/main.py session=train model=crop_model```

```python src/main.py session=train model=resize_model```

```python src/main.py session=train model=stretch_mode```

**NOTE** that this changes the input size of the model. This means that models trained with one setting are not
compatible with each other.

## Weights and Biases
Currently, API key from s174274 is used (set in config file).
You can see runs here ```https://wandb.ai/philipwastaken/02456_project```.
You can disable WandB if you so chose.

## Notes on `model_path`
`src/hparams/model/exp1.yaml` contains the field `model_path`. This is the name of the model you wish to either train
or run. ONLY input the name of the model file, NO preceding directories (e.g. `models/<model_name_here>`.
If this is set to an empty string (i.e., `''`), a new model is generated and trained from scratch.
If you set this parameter to a previously trained model, you can continue training the model (and even use different
hyperparameters).

It is important that you set this parameter if you wish to run the model. Otherwise it simply chooses the most
recently modified model for convinience.

## Changing `device`
Acceptable device settings are `auto`, `cpu` or `gpu`. `auto` will prefer `gpu` if possible and fallback to `cpu`. If `gpu` is chosen but not available, the program exits.