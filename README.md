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

