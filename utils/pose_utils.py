import torch

def old_content_orientation_disentanglement(data):
    '''
    :param data: the input point cloud, [B, G, M, 3]
    '''
       
    B, G, M, _ = data.shape
    data = data.reshape(B*G, M, 3)
    data_copy = data                       
    data = data.transpose(2, 1)     
    data = data - data.mean(axis=-1,keepdims=True)  
    data_T = data.transpose(2, 1) 
    try:
        H = torch.matmul(data, data_T)
    except:
        import pdb; pdb.set_trace()
    eigenvectors, eigenvalues, _ = torch.linalg.svd(H, full_matrices=True)
        
    data_copy = torch.matmul(data_copy, eigenvectors)
    
    center = torch.mean(data_copy, dim=1, keepdim=True)
    sign = torch.sign(center)
    sign[sign==0] = 1
    data_copy = data_copy * sign
    data_copy = data_copy.reshape(B, G, M, 3)

    eigenvectors = eigenvectors * sign
    eigenvectors = eigenvectors.transpose(2, 1) # why transpose ??? 
    eigenvectors = eigenvectors.reshape(B, G, 3, 3)
    
    return data_copy, eigenvectors # eigenvectors are ROWS

def content_orientation_disentanglement(data):
    '''
    :param data: the input point cloud, [B, G, M, 3]
    '''
       
    B, G, M, _ = data.shape
    data = data.reshape(B*G, M, 3)
    _, eigenvectors = compute_pca(data, disambiguate=True) # (B*G, 3, 3), y is up-axis 
    
    data = torch.matmul(data, eigenvectors)
    
    return data.reshape(B, G, M, 3), eigenvectors.reshape(B, G, 3, 3) # eigenvectors are COLS

def compute_pca(pc_batch, y_up=True, disambiguate=False, return_eigenvals=False):
    """
    Computes the principal component axes for a batch of point clouds.

    Args:
        pc_batch (torch.Tensor): Batch of point clouds with shape (B, N, 3).

    Returns:
        mean (torch.Tensor): Mean of each point cloud (B, 1, 3).
        eigenvectors (torch.Tensor): Principal component axes for each batch (B, 3, 3).
    """
    B, N, _ = pc_batch.shape  # (B, N, 3)

    # Compute mean for each point cloud
    mean = pc_batch.mean(dim=1, keepdim=True)  # (B, 1, 3)

    # Center the point clouds
    centered_pc = pc_batch - mean  # (B, N, 3)

    # Compute covariance matrix: (B, 3, 3)
    cov_matrix = torch.einsum('bni,bnj->bij', centered_pc, centered_pc) / N  # (B, 3, 3)

    # Compute eigenvalues and eigenvectors using torch.linalg.eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # (B, 3), (B, 3, 3)
    # eigenvalues, eigenvectors = symeig3x3(cov_matrix)  # (B, 3), (B, 3, 3)

    # Sort by decreasing eigenvalues
    sorted_indices = eigenvalues.argsort(dim=-1, descending=True)
    eigenvectors = torch.gather(eigenvectors, 2, sorted_indices.unsqueeze(1).expand(-1, 3, -1))
    eigenvalues_sorted = torch.gather(eigenvalues, 1, sorted_indices)

    if disambiguate:
        eigenvectors = disambiguate_pca_directions(centered_pc, eigenvectors, y_up=y_up)

    if return_eigenvals:
        return mean, eigenvectors, eigenvalues_sorted
    else:
        return mean, eigenvectors

def disambiguate_pca_directions(pcl: torch.Tensor, eig_vecs: torch.Tensor, y_up=True) -> torch.Tensor:
    """
    Disambiguates the directions of PCA eigenvectors for each batch of point clouds.
    
    Args:
        pcl: (B, N, 3) Batch of N points in B point clouds.
        eig_vecs: (B, 3, 3) Batch of eigenvectors for each point cloud (each column is an eigenvector).
        y_up: whether align normal (e3) to y-axis 

    Returns:
        eig_vecs: (B, 3, 3) Disambiguated eigenvectors.
    """
    B, N, _ = pcl.shape
    flip_axis_idx = 1 # default flip y axis to ensure valid rotation matrix (2nd most sig eigenvec)
    if y_up:
        eig_vecs = eig_vecs[:, :, [0, 2, 1]] # x-e1, y-e3, z-e2
        flip_axis_idx = 2
    
    # Compute mean point for each point cloud (B, 1, 3)
    mean_pcl = pcl.mean(dim=1, keepdim=True)

    # Compute the difference vectors from mean
    df = pcl - mean_pcl  # Shape: (B, N, 3)

    # Project differences onto each eigenvector (B, N, 3)
    proj = torch.einsum('bnd,bdc->bnc', df, eig_vecs)

    # Count positive projections for each eigenvector (B, 1, 3)
    n_pos = (proj > 0).float().sum(dim=1, keepdim=True)

    # Flip each eigenvector if more than half of projections are negative
    flip_mask = (n_pos < (0.5 * N)).float()  # Shape: (B, 1, 3)
    eig_vecs = (1.0 - 2.0 * flip_mask) * eig_vecs  # Flip signs where needed

    # Compute the determinant of the eigenvector matrix
    det = torch.det(eig_vecs) # (B,)

    # If determinant is negative, flip the sign of the vectors in the specified axis
    flip_sign_mask = (det < 0).float()  # Determine if the determinant is negative
    eig_vecs[:, :, flip_axis_idx] = (1.0 - 2.0 * flip_sign_mask.unsqueeze(-1)) * eig_vecs[:, :, flip_axis_idx]  # Flip the specified axis    

    return eig_vecs

def transform_to_pca_frame(pc_batch, normal_batch, mean, eigenvectors):
    """
    Transforms point clouds and normals into their PCA-aligned coordinate system.

    Args:
        pc_batch (torch.Tensor): Batch of point clouds (B, N, 3).
        normal_batch (torch.Tensor): Batch of normals (B, N, 3).
        mean (torch.Tensor): Mean of each point cloud (B, 1, 3).
        eigenvectors (torch.Tensor): Principal component axes (B, 3, 3).

    Returns:
        aligned_pc (torch.Tensor): PCA-aligned point clouds (B, N, 3).
        aligned_normals (torch.Tensor): Transformed normals (B, N, 3).
    """
    # Center the point clouds
    centered_pc = pc_batch - mean  # (B, N, 3)

    # Rotate points using eigenvectors
    aligned_pc = torch.einsum('bij,bnj->bni', eigenvectors.transpose(1, 2), centered_pc)  # (B, N, 3)

    # Rotate normals using eigenvectors (without translation)
    if normal_batch is not None:
        aligned_normals = torch.einsum('bij,bnj->bni', eigenvectors.transpose(1, 2), normal_batch)  # (B, N, 3)
    else:
        aligned_normals = None 
    
    return aligned_pc, aligned_normals

def matrix_to_6d(R):
    # stack two columns of the rotation
    return R[:, :2].T.reshape(6)

def rotation_6d_to_matrix(R6D: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of 6D rotation representations into a batch of 3x3 rotation matrices.

    Args:
        R6D (torch.Tensor): Tensor of shape (N, 6), where each row contains a 6D rotation representation.

    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) containing valid 3x3 rotation matrices.
    """
    eps = 1e-8
    # Split into x and y columns
    x_col = R6D[:, :3]  # First 3 elements for each row as X column
    y_col = R6D[:, 3:]  # Last 3 elements for each row as Y column

    # Normalize X column (N, 3)
    x_col = x_col / (torch.norm(x_col, dim=1, keepdim=True) + eps)

    # Make Y orthogonal to X
    y_col = y_col - torch.sum(y_col * x_col, dim=1, keepdim=True) * x_col

    # Normalize Y column (N, 3)
    y_col = y_col / (torch.norm(y_col, dim=1, keepdim=True) + eps)

    # Compute Z column using cross product (N, 3)
    z_col = torch.cross(x_col, y_col, dim=1)

    # Reconstruct rotation matrix (N, 3, 3)
    R_rec = torch.stack([x_col, y_col, z_col], dim=2)

    return R_rec