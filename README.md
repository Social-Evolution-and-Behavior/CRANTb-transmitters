# CRANTb-transmitters

## Pre-requisites

Install [pixi](https://pixi.sh/latest/).

The `main.py` function holds all the commands that we will use (aside from data caching, see below).
If running a command with `pixi`, there will be an accompanying task. If using `conda` or some other environment manager, you will have to replace: 
```
pixi run <task> <arguments>
```
with 
```
python main.py <task> <arguments>
```


## Getting data
To download and cache a set of data points, use the `getting_data.py` script. You can see what the arguments are by running: 

```
pixi run python getting_data.py --help
```

Note that if you don't specify the `start_index` and `end_index` it will download the whole dataset in a single go! You will want to run several jobs, each with a chunk of the data.

For example, to download just the first 10 samples, run: 
```
pixi run python getting_data.py --locations_file </path/to/your/file> --cache_path </path/to/cache> --start_index 0 --end_index 10
```

## Splitting the data into train and validation
The `split` task is used to split data between train and validation. It reads details from the configuration file, specifically under `gt`. 
Make sure to set the following feather file paths: 
- `gt.base`: the current location of your ground truth data.
- `gt.train`: the location you want for your training data.
- `gt.val`: the location you want for your validation data.
The `split` task will read from `base` and split into `train` and `val`. 
- `gt.neurotransmitters`: the list 

To run with `pixi`:
```
pixi split --cfg config.yaml
```

## Training a model
The `train` task is used to train (and validate) the model.
It reads details from the configuration file, especially under `data`, `train`, and `validate`.

Make sure to set: 
- `data.container`: the URL to the precomputed dataset
- `data.cache`: the location of your local cache of the data.
- `train.experiment_dir`: the place where we should store all results and checkpoints. 
- `train.input_shape`: the x,y,z shape of each block around a synapse.
- `train.batch_size`: the batch size for training. Can scale with the size of your GPU.
- `validate.batch_size`: the batch size for validation. Can be larger than for training.

You can train for longer than what your configuration file says by using the `--epochs` option. This will override what is in the configuration file. This will not change your configuration file, however, so make sure to keep track of that some other way!
```
pixi run train --cfg config.yaml --epochs 10
```

Additionally, the training script will resume runs by default: it will load the latest checkpoint and train from there until the `epochs` value that you have defined. 
If you would prefer to start training from scratch for this configuration, set `--restart`.