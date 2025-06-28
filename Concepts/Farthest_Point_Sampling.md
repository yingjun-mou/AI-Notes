# Farthest Point Sampling (FPS)

## Use case
Sample point cloud to reduce the number points to a specific dimension. FPS and Uniform Sampling are two common ways of point cloud sampling.

## Definition
Instead of random/uniform sampling, FPS **ensures better spatial coverage of the entire shape** by selecting points iteratively:
* Start with a random point.
* At each iteration, choose the point farthest from all previously selected points.
* Repeat until reaching the desired number of samples.

## Pros and Cons
### pros
* Ensure better spatial/geometric coverage of the entire shape.
* For batch FPS(many point clouds together) or GPU acceleration, PyTorch has specialize libraries.

### cons
* Slightly larger time complexity `O(n_samples * N)`, only feasible for small/medium point clouds.


## Heuristics
* FPS has been the standard practice in PointNet++ and other 3D learning models.

## Minimal Implementation
* PyTorch
```
import torch

def fps(xyz, npoints):
    """
    xyz: point cloud data, [N, 3] tensor on GPU
    npoint: number of samples

    Returns:
    centroids: sampled point indices, [npoint]
    """

    N, _ = xyz.shape  # Number of points

    # Tensor to store sampled point indices
    centroids = torch.zeros(npoint, dtype=torch.long).cuda()

    # Initialize distances for all points as very large (so any first point will reduce them)
    distance = torch.ones(N).cuda() * 1e10

    # Randomly choose the first centroid index
    farthest = torch.randint(0, N, (1,), dtype=torch.long).cuda()

    for i in range(npoint):
        centroids[i] = farthest  # Record this point's index

        # Get coordinates of the farthest point
        centroid = xyz[farthest, :].view(1, 3)

        # Calculate squared Euclidean distance to all other points
        dist = torch.sum((xyz - centroid) ** 2, -1)

        # Update distances: for each point, keep minimum distance to any selected centroid so far
        distance = torch.min(distance, dist)

        # Next farthest point is the one with the maximum of these minimum distances
        farthest = torch.max(distance, -1)[1]

    return centroids

# Example usage
if __name__ == "__main__":
    xyz = torch.rand(10000, 3).cuda()
    sampled_idx = fps(xyz, 1024)
    sampled_points = xyz[sampled_idx, :]

```

* TensorFlow
TBD.

* Raw Python
```
import numpy as np

def fps(points, n_samples):
	"""
	points: numpy array of shape (N, 3)
	n_samples: number of points to sample
	"""
	N, _ = points.shape
	sampled_pts = np.zeros((n_samples,))  # Placeholders of sampled points' indices
	distance = np.ones((N,)) * 1e10  # Initialize distances to a large number

	# Initialize with a random point
	farthest = np.random.randint(0,N)
	for i in range(n_samples):
		sampled_pts[i] = farthest
		centroid = points[farthest, :]
		dist = np.sum((points - centroid) ** 2, axis=1)
		distances = np.minimum(distances, dist)
		farthest =np.argmax(distances)

	sampled_pts = sampled_pts.astype(np.int32)
	return points[sampled_pts, :]

# Example usage
if __name__ == "__main__":
    pcd_np = np.random.rand(10000, 3)  # random point cloud for demo
    sampled = farthest_point_sampling(pcd_np, 1024)
    print("Sampled shape:", sampled.shape)

```

## Reference
* TBD
