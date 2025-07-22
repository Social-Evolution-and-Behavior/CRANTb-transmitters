# CRANTb-transmitters

## Pre-requisites

Install [pixi](https://pixi.sh/latest/).

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