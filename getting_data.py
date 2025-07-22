# %%
# Notebook to figure out how to access the raw data
from cloudvolume import CloudVolume
import pandas as pd
from tqdm import tqdm
import time
from time import perf_counter
from typing import Optional
import typer


def main(
    locations_file: str = "/ru-auth/local/home/jlee11/scratch/nt_predictions/250721_pseudoGT_neurotransmitter_noduplicates.feather",
    cache_path: str = "/ru-auth/local/home/jlee11/scratch/nt_predictions/caches",
    start_index: int = 0,
    end_index: Optional[int] = None,
    dx: int = 80,
    dy: int = 80,
    dz: int = 16,
    cloud_volume_path: str = "precomputed://gs://dkronauer-ant-001-alignment-final/aligned",
):
    """
    Download data from a CloudVolume and cache it locally.

    Args:
        locations_file (str): Path to the feather file containing the locations.
        cache_path (str): Path to the local cache directory -- this is where the data will be cached.
        start_index (int): Start index for the locations to download.
        end_index (Optional[int]): End index for the locations to download. If None, all locations from start_index to the end will be downloaded.
        dx (int): Width of the data to download.
        dy (int): Height of the data to download.
        dz (int): Depth of the data to download.
        cloud_volume_path (str): Path to the CloudVolume to download data from.
    """
    df = pd.read_feather(locations_file)
    #subsetting
    df = df.sample(n=10000, random_state=42)
    df.to_feather(f"imported_synapses_{time.strftime("%Y%m%d")}.feather")
    
    
    locations = df[["x", "y", "z"]].values
    # Only get the locations between start_index and end_index
    if end_index is None:
        end_index = len(locations)
    locations = locations[start_index:end_index]

    cv = CloudVolume(
        cloudpath=cloud_volume_path,
        use_https=True,
        cache=cache_path,
        progress=False,
        parallel=True,
    )

    # tqdm option

    # for i, (x, y, z) in tqdm(
    #     enumerate(locations),
    #     total=len(locations),
    #     desc="Downloading data",
    # ):
    #     # Get the data from the cloud volume
    #     data = cv[x : x + dx, y : y + dy, z : z + dz]
    #     # This should cache the data locally
    
    #every 50 synapses for slurm 
    total = len(locations)
    start_loop_time = perf_counter()
    for i, (x, y, z) in enumerate(locations):
        # Get the data from the cloud volume
        data = cv[x : x + dx, y : y + dy, z : z + dz]
        # This should cache the data locally
        if i == 1 or i % 50 == 0 and i > 0:
            elapsed = perf_counter() - start_loop_time
            rate = elapsed / i
            remaining = rate * (total - i)
            print(f"Processed {i}/{total}. Elapsed: {elapsed:.2f}s, Estimated remaining: {remaining:.2f}s")

    # Check that the data is cached by comparing the time it takes
    # Get the first location
    start_time = perf_counter()
    data = cv[
        locations[0][0] : locations[0][0] + dx,
        locations[0][1] : locations[0][1] + dy,
        locations[0][2] : locations[0][2] + dz,
    ]
    end_time = perf_counter()
    print(f"Time taken to get the first location: {end_time - start_time:.2f} seconds")

    # Compare to the time it takes if I turn caching off
    cv_no_cache = CloudVolume(
        cloudpath=cloud_volume_path,
        use_https=True,
        cache=False,
        progress=False,
        parallel=True,
    )
    start_time = perf_counter()
    data_no_cache = cv_no_cache[
        locations[0][0] : locations[0][0] + dx,
        locations[0][1] : locations[0][1] + dy,
        locations[0][2] : locations[0][2] + dz,
    ]
    end_time = perf_counter()
    print(
        f"Time taken to get the first location without cache: {end_time - start_time:.2f} seconds"
    )
    # The data should be the same
    assert (data == data_no_cache).all(), "Data does not match with and without cache"


if __name__ == "__main__":
    typer.run(main)
