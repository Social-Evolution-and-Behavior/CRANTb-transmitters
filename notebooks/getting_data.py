# %%
# Notebook to figure out how to access the raw data
from cloudvolume import CloudVolume
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm

df = pd.read_feather("test_experiment/train.feather")
locations = df[["x", "y", "z"]].values
classes = df["neurotransmitter"]
dx, dy, dz = 80, 80, 16


# %%
cv = CloudVolume(
    "precomputed://gs://dkronauer-ant-001-alignment-final/aligned",
    use_https=True,
    cache=False,
    progress=False,
    parallel=True,
)

# %%
# KDTree implementation for finding optimal bounding box sizes
from sklearn.neighbors import KDTree
import numpy as np


def find_bounding_box_for_n_samples(
    kdtree, query_point, target_n_samples, max_radius=1000
):
    """
    Find the minimum bounding box size needed to get target_n_samples from a KDTree.

    Parameters:
    - kdtree: sklearn KDTree object
    - query_point: point to query around (shape: (3,) for 3D)
    - target_n_samples: desired number of samples
    - max_radius: maximum search radius

    Returns:
    - radius: minimum radius needed
    - bounding_box_size: (width, height, depth) of the bounding box
    """

    # Binary search for the minimum radius
    low, high = 0, max_radius
    tolerance = 1e-6

    while high - low > tolerance:
        mid_radius = (low + high) / 2

        # Query neighbors within this radius
        indices = kdtree.query_radius([query_point], r=mid_radius)[0]
        n_found = len(indices)

        if n_found >= target_n_samples:
            high = mid_radius
        else:
            low = mid_radius

    radius = high
    # For a cubic bounding box, the side length is 2 * radius
    bounding_box_size = (2 * radius, 2 * radius, 2 * radius)

    return radius, bounding_box_size


def find_exact_bounding_box_for_n_samples(kdtree, query_point, target_n_samples):
    """
    Find the exact bounding box needed for exactly target_n_samples.
    """
    # Get all distances and indices
    distances, indices = kdtree.query(
        [query_point], k=min(target_n_samples, kdtree.data.shape[0])
    )

    if len(distances[0]) < target_n_samples:
        # Not enough samples in the tree
        return None, None

    # The radius needed is the distance to the k-th nearest neighbor
    radius = distances[0][target_n_samples - 1]
    bounding_box_size = (2 * radius, 2 * radius, 2 * radius)

    return radius, bounding_box_size


# %%
# Build KDTree from your location data
kdtree = KDTree(locations)

# Example: Find bounding box for 100 samples around the first location
query_point = locations[0]
target_samples = 16

radius, bbox_size = find_exact_bounding_box_for_n_samples(
    kdtree, query_point, target_samples
)

print(f"Query point: {query_point}")
print(f"Radius needed for {target_samples} samples: {radius:.2f}")
print(f"Bounding box size (width, height, depth): {bbox_size}")

# Compare with your current fixed box size
current_box = (dx, dy, dz)
print(f"Current fixed box size: {current_box}")

# %%
# Analyze actual neighbor distribution (non-cubic bounding box)
indices = kdtree.query([query_point], k=target_samples)[1][0]
neighbor_coords = locations[indices]

# Calculate actual bounding box dimensions needed
min_coords = neighbor_coords.min(axis=0)
max_coords = neighbor_coords.max(axis=0)
actual_bbox = max_coords - min_coords

print(f"Actual bounding box dimensions needed: {actual_bbox}")
print(f"Bounding box ranges (includes query point):")
print(f"  X range: {min_coords[0]:.1f} to {max_coords[0]:.1f}")
print(f"  Y range: {min_coords[1]:.1f} to {max_coords[1]:.1f}")
print(f"  Z range: {min_coords[2]:.1f} to {max_coords[2]:.1f}")
print(f"Query point: {query_point} (should be within these ranges)")
print(f"Actual number of neighbors found: {len(neighbor_coords)}")

# %%
# Benchmark timing for finding bounding boxes at different locations
import time
from statistics import mean, stdev
from tqdm import tqdm


def benchmark_bbox_finding(kdtree, locations, n_samples=32, n_test_points=50):
    """
    Benchmark the time it takes to find bounding boxes for different query points.

    Parameters:
    - kdtree: KDTree object
    - locations: array of all locations
    - n_samples: number of samples to find in each bounding box
    - n_test_points: number of different query points to test

    Returns:
    - timing results dictionary
    """

    # Select random test points from the locations
    test_indices = np.random.choice(
        len(locations), min(n_test_points, len(locations)), replace=False
    )
    test_points = locations[test_indices]

    # Benchmark the exact method
    exact_times = []
    binary_times = []

    print(
        f"Benchmarking bbox finding for {n_samples} samples across {len(test_points)} locations..."
    )

    for i, query_point in enumerate(tqdm(test_points)):
        # Time the exact method
        start_time = time.perf_counter()
        radius_exact, bbox_exact = find_exact_bounding_box_for_n_samples(
            kdtree, query_point, n_samples
        )
        exact_time = time.perf_counter() - start_time
        exact_times.append(exact_time)

        # Time the binary search method
        start_time = time.perf_counter()
        radius_binary, bbox_binary = find_bounding_box_for_n_samples(
            kdtree, query_point, n_samples
        )
        binary_time = time.perf_counter() - start_time
        binary_times.append(binary_time)

    return {
        "exact_times": exact_times,
        "binary_times": binary_times,
        "n_samples": n_samples,
        "n_test_points": len(test_points),
    }


# Run the benchmark
benchmark_results = benchmark_bbox_finding(
    kdtree, locations, n_samples=100, n_test_points=100
)

# %%
# Analyze benchmark results
exact_times = benchmark_results["exact_times"]
binary_times = benchmark_results["binary_times"]

print("=== Timing Results ===")
print(f"Exact method (query k-nearest):")
print(f"  Mean: {mean(exact_times)*1000:.3f} ms")
print(f"  Std:  {stdev(exact_times)*1000:.3f} ms")
print(f"  Min:  {min(exact_times)*1000:.3f} ms")
print(f"  Max:  {max(exact_times)*1000:.3f} ms")

print(f"\nBinary search method (query_radius):")
print(f"  Mean: {mean(binary_times)*1000:.3f} ms")
print(f"  Std:  {stdev(binary_times)*1000:.3f} ms")
print(f"  Min:  {min(binary_times)*1000:.3f} ms")
print(f"  Max:  {max(binary_times)*1000:.3f} ms")

print(f"\nSpeedup factor: {mean(binary_times)/mean(exact_times):.2f}x")
print(
    f"Exact method is {'faster' if mean(exact_times) < mean(binary_times) else 'slower'}"
)

# %%  Get the data from the bounding box for the n-th location
n_samples = 16
query_point = locations[10]
indices = kdtree.query([query_point], k=n_samples)[1][0]
neighbor_coords = locations[indices]
neighbor_classes = classes.iloc[indices]  # Get corresponding classes
min_coords = neighbor_coords.min(axis=0).astype(int)
max_coords = neighbor_coords.max(axis=0).astype(int)
# Scale the coordinates up so that
# We can get dx, dy, dz around each point in the box
min_coords = np.maximum(min_coords - np.array([dx, dy, dz]) // 2, 0)
max_coords = np.minimum(max_coords + np.array([dx, dy, dz]) // 2, cv.shape[:-1])
# Get the data from the bounding box

t0 = time.perf_counter()
bbox_data = cv[
    min_coords[0] : max_coords[0],
    min_coords[1] : max_coords[1],
    min_coords[2] : max_coords[2],
]
t1 = time.perf_counter()
print(f"Time taken to load bounding box: {t1 - t0:.2f} seconds")

# Get a crop of size (dx, dy, dz) around each neighbor point
crops = []
crop_classes = []
t0 = time.perf_counter()
for coord, class_label in zip(neighbor_coords, neighbor_classes):
    coord = coord.astype(int)
    # Compute crop bounds relative to the bbox_data (not global coordinates)
    min_x = max(coord[0] - dx // 2, min_coords[0]) - min_coords[0]
    max_x = min(coord[0] + dx // 2, max_coords[0]) - min_coords[0]
    min_y = max(coord[1] - dy // 2, min_coords[1]) - min_coords[1]
    max_y = min(coord[1] + dy // 2, max_coords[1]) - min_coords[1]
    min_z = max(coord[2] - dz // 2, min_coords[2]) - min_coords[2]
    max_z = min(coord[2] + dz // 2, max_coords[2]) - min_coords[2]

    crop = bbox_data[min_x:max_x, min_y:max_y, min_z:max_z]
    crops.append(crop)
    crop_classes.append(class_label)
print(f"Time taken to crop {time.perf_counter() - t0 }")

# Print class information
print(f"Classes of the {n_samples} neighbors:")
for i, class_label in enumerate(crop_classes):
    print(f"  Crop {i}: {class_label}")

# Count class distribution
from collections import Counter

class_counts = Counter(crop_classes)
print(f"\nClass distribution:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count}")

# plot the center slice of each crop with class labels
fig, axes = plt.subplots(4, len(crops) // 4, figsize=(12, 8))
for i, (crop, class_label, ax) in enumerate(zip(crops, crop_classes, axes.ravel())):
    center_slice = crop[:, :, crop.shape[2] // 2]
    ax.imshow(center_slice, cmap="gray")
    ax.set_title(f"{class_label}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()


# %%
def get_neighbor_crops_and_classes(
    cv, kdtree, locations, classes, query_point, n_samples, dx, dy, dz
):
    """
    Get crops and corresponding classes for neighbors around a query point.

    Parameters:
    - cv: CloudVolume object
    - kdtree: KDTree object
    - locations: array of all locations
    - classes: pandas Series of class labels
    - query_point: point to query around (shape: (3,) for 3D)
    - n_samples: number of samples to find
    - dx, dy, dz: crop dimensions

    Returns:
    - crops: list of numpy arrays (image crops)
    - crop_classes: list of class labels
    - neighbor_coords: coordinates of the neighbors
    """
    # Find neighbors
    indices = kdtree.query([query_point], k=n_samples)[1][0]
    neighbor_coords = locations[indices]
    neighbor_classes = classes.iloc[indices].values  # Convert to numpy array

    # Calculate bounding box
    min_coords = neighbor_coords.min(axis=0).astype(int)
    max_coords = neighbor_coords.max(axis=0).astype(int)
    min_coords = np.maximum(min_coords - np.array([dx, dy, dz]) // 2, 0)
    max_coords = np.minimum(max_coords + np.array([dx, dy, dz]) // 2, cv.shape[:-1])

    # Load bounding box data
    bbox_data = cv[
        min_coords[0] : max_coords[0],
        min_coords[1] : max_coords[1],
        min_coords[2] : max_coords[2],
    ]

    # Extract crops
    crops = []
    for coord in neighbor_coords:
        coord = coord.astype(int)
        min_x = max(coord[0] - dx // 2, min_coords[0]) - min_coords[0]
        max_x = min(coord[0] + dx // 2, max_coords[0]) - min_coords[0]
        min_y = max(coord[1] - dy // 2, min_coords[1]) - min_coords[1]
        max_y = min(coord[1] + dy // 2, max_coords[1]) - min_coords[1]
        min_z = max(coord[2] - dz // 2, min_coords[2]) - min_coords[2]
        max_z = min(coord[2] + dz // 2, max_coords[2]) - min_coords[2]

        crop = bbox_data[min_x:max_x, min_y:max_y, min_z:max_z]
        crops.append(crop)

    return crops, neighbor_classes.tolist(), neighbor_coords


def benchmark_cloudvolume_loading(
    cv, kdtree, locations, n_samples=32, n_test_points=20
):
    """
    Benchmark the time it takes to load bounding box data from CloudVolume for different query points.

    Parameters:
    - cv: CloudVolume object
    - kdtree: KDTree object
    - locations: array of all locations
    - n_samples: number of samples to find in each bounding box
    - n_test_points: number of different query points to test

    Returns:
    - list of timing results (seconds)
    """
    test_indices = np.random.choice(
        len(locations), min(n_test_points, len(locations)), replace=False
    )
    test_points = locations[test_indices]
    times = []

    for query_point in tqdm(test_points, desc="CloudVolume loading benchmark"):
        # Find actual bounding box for n_samples neighbors
        indices = kdtree.query([query_point], k=n_samples)[1][0]
        neighbor_coords = locations[indices]
        min_coords = neighbor_coords.min(axis=0).astype(int)
        max_coords = neighbor_coords.max(axis=0).astype(int)

        t0 = time.perf_counter()
        _ = cv[
            min_coords[0] : max_coords[0],
            min_coords[1] : max_coords[1],
            min_coords[2] : max_coords[2],
        ]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(
        f"CloudVolume loading: mean={np.mean(times):.3f}s, std={np.std(times):.3f}s, min={np.min(times):.3f}s, max={np.max(times):.3f}s"
    )
    return times


# Example usage:
times = cloudvolume_times = benchmark_cloudvolume_loading(
    cv, kdtree, locations, n_samples=16, n_test_points=50
)
