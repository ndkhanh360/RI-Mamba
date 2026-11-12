import torch

def sort_vim_orders(center, space_size=1024, x_offset=0, y_offset=1, z_offset=2):
    """
    Sort point centers in a batch of point clouds based on a depth-first, row-major scanning pattern.
    
    Args:
        points (torch.Tensor): Tensor of shape (B, G, 3), where B is the batch size,
                               G is the number of point centers, and 3 are the spatial coordinates.
        voxel_size (int): The size of the voxel grid (e.g., 1024 for dividing the space into 1024^3 voxels).
        offsets: specify order of x, y, z in center coordinate. 
    Returns:
        torch.Tensor: Sorted points of shape (B, G, 3).
    """
    B, G, _ = center.shape
    
    # Normalize points to [0, 1] range (assuming point cloud bounds are known, adjust as needed)
    min_coords = center.min(dim=1, keepdim=True).values  # Shape: (B, 1, 3)
    max_coords = center.max(dim=1, keepdim=True).values  # Shape: (B, 1, 3)
    normalized_points = (center - min_coords) / (max_coords - min_coords + 1e-8)  # Avoid division by zero
    
    # Compute voxel indices
    voxel_grid_size = space_size ** (1 / 3)  # Assuming cubic voxel grid
    voxel_grid_size = int(voxel_grid_size)
    voxel_indices = (normalized_points * voxel_grid_size).long()  # Shape: (B, G, 3)
    
    # Convert 3D voxel indices to 1D depth-first, row-major indices
    voxel_1d_indices = (
        voxel_indices[..., z_offset] * voxel_grid_size**2 +  # Depth (z-axis)
        voxel_indices[..., y_offset] * voxel_grid_size +    # Row (y-axis)
        voxel_indices[..., x_offset]                       # Column (x-axis)
    )  # Shape: (B, G)
    
    # Sort points based on 1D voxel indices
    sorted_indices = voxel_1d_indices.argsort(dim=1)  # Shape: (B, G)
    
    return sorted_indices

