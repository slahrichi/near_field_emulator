import torch
import math
from tqdm import tqdm
from scipy.special import genlaguerre
import os
import matplotlib.pyplot as plt
    
'''def svd(x, config):
    """
    Computes the Singular Value Decomposition on the entire sequence for each sample.
    
    Args:
        x (torch.Tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        config: Configuration parameters
    
    Returns:
        torch.Tensor: SVD-transformed tensor of size [samples, r_i, 1, k, slices]
    """
    samples, r_i, xdim, ydim, slices = x.size()
    full_svd_params = []
    
    # Iterate over each sample
    for i in tqdm(range(samples), desc='Processing Samples'):
        # Initialize a list to store real and imaginary channels
        channels = []
        
        # Iterate over each channel (Real and Imaginary)
        for c in range(r_i):
            # Extract the data for the current channel and reshape it
            # Original shape: [xdim, ydim, slices]
            # Reshaped to: [xdim * ydim, slices]
            channel_data = x[i, c].reshape(xdim * ydim, slices)  # Shape: [166*166, 63]
            channels.append(channel_data)
        
        # Concatenate both channels along the first dimension
        # Resulting shape: [2 * 166 * 166, 63]
        aggregated_matrix = torch.cat(channels, dim=0)  # Shape: [2*166*166, 63]
        
        # Perform SVD on the aggregated matrix
        # U: [2*166*166, 63], S: [63], Vh: [63, 63]
        try:
            u, s, vh = torch.linalg.svd(aggregated_matrix, full_matrices=False)
        except RuntimeError as e:
            print(f"SVD did not converge for sample {i}. Error: {e}")
            # Handle SVD convergence issues, e.g., assign zeros or skip
            continue
        
        # Store full decomposition for potential full reconstruction later
        full_svd_params.append({
            'u': u,      # [2*xdim*ydim, slices]
            's': s,      # [slices]
            'vh': vh     # [slices, slices]
        })
        
        if i == 0:
            # Find the optimal k that captures 95% of the energy
            k = find_optimal_k_svd(s, threshold=0.95)
            print(f"Optimal k was found to be: {k}")
            # Initialize the output tensor with the desired shape
            # [samples, r_i, 1, k, slices]
            x_svd = torch.zeros(
                samples, r_i, 1, k, slices, 
                device=x.device, dtype=x.dtype
            )
            
        # Extract the top k right singular vectors
        # vh has shape [63, 63], so vh[:k, :] has shape [k, 63]
        topk_v = vh[:k, :]  # Shape: [k, 63]
        
        # Assign the top k singular vectors to each channel and slice
        # This implies that each slice j has a k-dimensional feature vector: topk_v[:, j]
        # We'll broadcast this across both channels
        
        for c in range(r_i):
            # Assign the topk_v for all slices to the output tensor
            # x_svd shape: [samples, r_i, 1, k, slices]
            # topk_v has shape [k, 63]
            # We need to assign [k, slices] to [1, k, slices] for each channel
            x_svd[i, c, 0, :, :] = topk_v  # Broadcasting the same topk_v across channels
    
    return x_svd, full_svd_params'''
    
def svd(x, config):
    """
    GLOBAL SVD approach:
      - x has shape [samples, r_i, xdim, ydim, slices].
      - Flatten all samples & slices into one big matrix => do SVD => pick top k.
      - Return (x_svd, global_svd_params) so you can project/invert later.

    Returns:
      x_svd: shape [samples, r_i, 1, k, slices]
      global_svd_params: dict with 'U_k' and 'S_k' 
                         (and possibly the full 'U','S','Vh' if you want).
    """
    device = x.device
    dtype = x.dtype

    samples, r_i, xdim, ydim, num_slices = x.shape
    spatial_dim = r_i * xdim * ydim  # e.g. 2 * 166 * 166 if r_i=2

    # 1) Build the "mega-matrix"
    # We'll stack all slices from all samples along columns.
    # shape => [spatial_dim, (samples * num_slices)]
    # Each column is ONE slice from ONE sample, flattened in space.
    mega_list = []
    for i in tqdm(range(samples), desc='Collecting slices'):
        # shape for sample i => [r_i, xdim, ydim, num_slices]
        # Flatten each slice => [spatial_dim], then stack across slices => [spatial_dim, num_slices]
        sample_data = x[i].reshape(r_i * xdim * ydim, num_slices)
        mega_list.append(sample_data)
    # Now we have 'samples' arrays of shape [spatial_dim, num_slices].
    # Concatenate along dimension=1 => total columns = samples * num_slices
    mega_matrix = torch.cat(mega_list, dim=1).to(device)  # shape [spatial_dim, samples*num_slices]

    # 2) SVD on mega_matrix
    # Let's do full_matrices=False for efficiency.
    #   U: [spatial_dim, rank], S: [rank], Vh: [rank, samples*num_slices]
    print("Performing global SVD on mega_matrix of shape:", mega_matrix.shape)
    U, S, Vh = torch.linalg.svd(mega_matrix, full_matrices=False)

    # 3) Choose k
    k = find_optimal_k_svd(S, threshold=0.95)
    print(f"Optimal k for {0.95*100:.0f}% energy is: {k}")

    # Let's define top-k versions
    U_k = U[:, :k]  # shape [spatial_dim, k]
    S_k = S[:k]     # shape [k]
    # If you want, you can store Vh_k = Vh[:k, :] but for *projection of slices*, you only need U_k, S_k.
    # Because to get "slice in k-dim" = (U_k^T * diag(S_k))^-1 * ...
    # Actually, simpler is to define a direct transform for "slice -> k-dim" below.

    # 4) "Project" each slice in x to get x_svd 
    #    For each slice => flatten => multiply by U_k^T (and S_k if you want rank-k approx).
    #    We often do: z = U_k^T * x_slice => shape [k], ignoring S for a pure subspace basis (like PCA).
    #    Or we do z = diag(S_k) * U_k^T * x_slice if we want an SVD-based scaling. 
    # Typically for dimensionality reduction, we do "z = U_k^T * x_slice" (like PCA).
    # We'll keep it simpler: "z = U_k^T * x_slice".
    # Then reconstruct: x_slice_approx = U_k * z.

    # We'll store x_svd in shape [samples, r_i, 1, k, slices],
    # to match existing usage. Each slice is a length-k vector.
    x_svd = torch.zeros(samples, r_i, 1, k, num_slices, device=device, dtype=dtype)

    # We want a function for "project slice -> k-dim"
    # flatten: shape [spatial_dim], z = U_k^T @ slice_vec => shape [k]
    def project_slice(slice_data_flat):
        # slice_data_flat: shape [spatial_dim]
        # z = U_k^T @ slice_data_flat
        return U_k.T @ slice_data_flat

    idx_sample = 0
    for i in tqdm(range(samples), desc='Projecting slices to k-dim'):
        # shape => [r_i, xdim, ydim, num_slices] => flatten in space => [spatial_dim, num_slices]
        sample_data_2d = x[i].reshape(spatial_dim, num_slices)  # [spatial_dim, num_slices]
        
        # For each slice, project
        # result => [k, num_slices]
        z = project_slice(sample_data_2d)  # but we must do it slice by slice
        # Actually, we can do matrix multiply: z = U_k^T @ sample_data_2d
        # shape => [k, num_slices]
        z = U_k.T @ sample_data_2d
        
        # Now we have z: shape [k, num_slices]
        # We want to store it in x_svd[i, r_i, 1, k, slices], but r_i=2 is along dimension=1.
        # However, we already merged r_i into "spatial_dim". 
        # So typically you'd have 1 "channel" dimension if r_i=2 is folded into the flattening.
        # Let's just store it all in x_svd[i, 0, 0, :, :] for now if r_i=2 was collapsed.
        
        # If you want to keep 'r_i' dimension purely for consistency, we can do:
        # x_svd[i, :, 0, :, :] => shape [r_i, k, slices].
        # But we actually only have 1 "flattened" channel. 
        # We'll do:
        for c in range(r_i):
            # c-th channel slice_data => we need to pick out the c-th portion from the flatten
            # Actually, it's simpler to say "We won't separate c inside x_svd," but let's do it for clarity:
            # The portion of 'z' that corresponds to channel c is z_c = z for c's subrange in spatial_dim.
            # This is complicated if we don't do it at the big SVD step differently.
            # 
            # ***In the simpler approach***: We treat the entire (r_i * xdim * ydim) as one channel. 
            # We'll just store the same z into x_svd[i, c], but that would be duplicating. 
            # Typically you'd keep a single k-dim representation for the entire slice (both channels).
            # 
            # Let's do a simpler approach: x_svd[i, 0, 0, :, :] = z. 
            # i.e. no separate real/imag channel dimension. 
            pass

        # For demonstration, let's just store z in the first channel dimension
        # shape => [1, k, num_slices], but we have x_svd shape => [samples, r_i, 1, k, num_slices]
        # We'll store the same z in each channel to fill it up, 
        # or you can decide you want r_i=1 from now on, because we merged them anyway.
        # 
        # We'll do the "lowest friction" approach: replicate z across c in r_i.
        for c in range(r_i):
            x_svd[i, c, 0, :, :] = z
    
    # 5) Save the relevant params to reconstruct later
    global_svd_params = {
        'U_k': U_k,     # shape [spatial_dim, k]
        'S': S,         # full singular values if you want
        'k': k,
        # possibly store 'Vh': Vh if you want full reconstruction. Usually for "project" we only need U_k.
    }

    return x_svd, global_svd_params

def find_optimal_k_svd(s, threshold=0.95):
    """
    Find the smallest k such that the top k singular values
    capture at least `threshold` fraction of the total energy.
    """    
    # Compute the total energy
    total_energy = (s ** 2).sum()  # sum of squares of singular values
    
    # Compute the cumulative energy ratio
    cumulative_energy = torch.cumsum(s**2, dim=0)
    energy_ratio = cumulative_energy / total_energy
    
    # Find the smallest k for which energy_ratio[k-1] >= threshold
    # (k-1 because PyTorch indexing is 0-based)
    ks = torch.where(energy_ratio >= threshold)[0]
    if len(ks) == 0:
        # Means even if we take all singular values, we don't reach threshold
        k_opt = len(s)
    else:
        k_opt = ks[0].item() + 1  # +1 to turn index into count
    
    return k_opt

def reconstruct_svd(z, full_svd_params):
    """
    Reconstructs the original data from the SVD decomposition.
    """
    U_k = full_svd_params['U_k']
    output = torch.zeros(2, 166, 166, z.shape[-1])
    
    for slice in range(z.shape[-1]):
        z_kdim = z[:, slice] # shape [k]
        slice_flat = U_k @ z_kdim
        output[:, :, :, slice] = slice_flat.reshape(2, 166, 166)
    
    return output

def random_proj(x, config):
    """
    Computes a random projection/Johnson Lindenstrauss for the data.
    For simplicity, we create a random projection matrix to reduce ydim to k
    
    Args:
        x (tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        config: configuration parameters
    """
    # i_dims is our latent dimensionality
    k = config.model.modelstm.i_dims
    samples, r_i, xdim, ydim, slices = x.size()
    d = xdim * ydim
    
    # calculate single dim for output
    k_dim = int(math.sqrt(k))
    # Verify k is a perfect square
    if k_dim * k_dim != k:
        raise ValueError(f"i_dims ({k}) must be a perfect square")
    # init output [samples, r_i, sqrt(k), sqrt(k), slices]
    x_rp = torch.zeros(samples, r_i, k_dim, k_dim, slices, 
                       device=x.device, dtype=x.dtype)
    
    # reproducibility
    torch.manual_seed(config.model.modelstm.seed)
    
    # create a random projection matrix [d, k]
    w = torch.randn(d, k, device=x.device, dtype=x.dtype) / math.sqrt(k)
    
    for i in tqdm(range(samples), desc='Processing Samples'):
        for j in range(slices):
            # extract channels
            real = x[i, 0, :, :, j].reshape(-1) # [d]
            imag = x[i, 1, :, :, j].reshape(-1) # [d]
            
            # apply projection: [d] * [d, k]
            real_emb = real @ w # [k]
            imag_emb = imag @ w # [k]
            
            # reshape embeddings to square matrices
            real_emb = real_emb.reshape(k_dim, k_dim)
            imag_emb = imag_emb.reshape(k_dim, k_dim)
            
            # Store results
            x_rp[i, 0, :, :, j] = real_emb
            x_rp[i, 1, :, :, j] = imag_emb

    return x_rp
    

def generate_gl_modes(xdim, ydim, k, w0, p_max, l_max, device, dtype):
    """
    Generate at least k Gaussian-Laguerre modes on a 2D grid.
    We'll enumerate (p,l) pairs from p=0..p_max, l=-l_max..l_max
    until we have at least k modes.

    Returns:
        modes: complex tensor of shape [k, xdim, ydim], complex64 or complex128
        (Depending on dtype support. We'll use torch.complex64 if dtype is float32.)
    """
    # create coord grid
    # center grid from -1 to 1 in both x and y #TODO real physical scale?
    x_lin = torch.linspace(-1.0, 1.0, xdim, device=device, dtype=dtype)
    y_lin = torch.linspace(-1.0, 1.0, ydim, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='ij') # [xdim, ydim]
    
    R = torch.sqrt(X**2 + Y**2) # radius
    Theta = torch.atan2(Y, X) # angle
    
    modes_list = []
    count = 0
    
    # enumerate (p, l) pairs
    for p in tqdm(range(p_max+1), desc='Generating GL Modes'):
        for l in range(-l_max, l_max+1):
            if count >= k:
                break
            # Compute LG_{p}^{l}(r, theta)
            # Laguerre polynomial: L_p^{|l|}(x)
            # set x = 2r^2/w0^2
            r_scaled = 2.0 * R.pow(2) / (w0**2)
            L_p_l = torch.from_numpy(genlaguerre(p, abs(l))(r_scaled.cpu().numpy())).to(device=device, dtype=dtype)
            
            # LG mode formula (up to normalization constant)
            # common norm in lieu of exact constants #TODO okay?
            # LG_{p}^{l}(r, theta) ~ (sqrt(2)*r/w0)^{|l|} * exp(-r^2/w0^2) * L_p^{|l|}(2r^2/w0^2) * exp(i l θ)
            # TODO add normalization constant
            radial_part = (math.sqrt(2.0) * R / w0).pow(abs(l)) * torch.exp(-R.pow(2)/ (w0**2)) * L_p_l
            phase_part = torch.exp(1j * l * Theta)  # complex
            mode = radial_part * phase_part  # complex-valued
            
            modes_list.append(mode)
            count += 1
        if count >= k:
            break
        
    # [k, xdim, ydim]
    modes = torch.stack(modes_list, dim=0)
    return modes

def gauss_laguerre_proj(x, config):
    """
    Project the input field onto Gaussian-Laguerre modes.

    Args:
        x (tensor): [samples, r_i, xdim, ydim, slices]
        config: configuration parameters
    """
    k = config.model.modelstm.i_dims
    w0 = config.model.modelstm.w0
    p_max = config.model.modelstm.p_max
    l_max = config.model.modelstm.l_max
    
    samples, r_i, xdim, ydim, slices = x.size()
    
    # generate k modes [k, xdim, ydim] (complex)
    modes = generate_gl_modes(xdim, ydim, k, w0, p_max, l_max, x.device, x.dtype)
    
    # calculate single dim for output
    k_dim = int(math.sqrt(k))
    # Verify k is a perfect square
    if k_dim * k_dim != k:
        raise ValueError(f"i_dims ({k}) must be a perfect square")
    x_lg = torch.zeros(samples, r_i, k_dim, k_dim, slices, device=x.device, dtype=x.dtype)
    
    # Projection: coeff: sum over x,y of E(x,y)*conjugate(mode(x,y))
    # E(x,y) = E_r + i E_i. mode is complex
    for i in tqdm(range(samples), desc='Processing Samples'):
        for j in range(slices):
            # construct complex field for this slice
            E_r = x[i, 0, :, :, j]
            E_i = x[i, 1, :, :, j]
            E = torch.complex(E_r, E_i)
            
            # compute inner products
            modes_conj = torch.conj(modes) # [k, xdim, ydim]
            E_expanded = E.unsqueeze(0) # Expanded: [1, xdim, ydim], broadcast mul
            coeffs = (E_expanded * modes_conj).sum(dim=(-2, -1)) # [k]
            
            #print(f'real coeffs shape: {coeffs.real.shape}')
            
            # reshape embeddings to square matrices - friendly for datamodule 
            # but we flatten back for actual processing
            real = coeffs.real.reshape(k_dim, k_dim)
            imag = coeffs.imag.reshape(k_dim, k_dim)
            
            # update output tensor
            x_lg[i, 0, :, :, j] = real
            x_lg[i, 1, :, :, j] = imag
            
    return x_lg
    
def fourier_modes(x, config):
    """
    Encoding fourier modes on the input data
    
    Args:
        x (tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        config: configuration parameters
    """
    pass

def encode_modes(data, config):
    """Takes the input formatted dataset and applies a specified modal decomposition

    Args:
        data (tensor): the dataset
        config: mode encoding parameters
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset with encoded data
    """
    near_fields = data['near_fields'].clone()
    
    method = config.model.modelstm.method
    
    if method == 'svd': # encoding singular value decomposition
        encoded_fields, full_svd_params = svd(near_fields, config)
        data['full_svd_params'] = full_svd_params
    elif method == 'random': # random projection / Johnson-Lindenstrauss
        encoded_fields = random_proj(near_fields, config)
    elif method == 'gauss': # gauss-laguerre modes
        encoded_fields = gauss_laguerre_proj(near_fields, config)
    elif method == 'fourier': # a fourier encoding
        encoded_fields = fourier_modes(near_fields, config)
    else:
        raise NotImplementedError(f"Mode encoding method '{method}' not recognized.")
    
    # update the real data
    data['near_fields'] = encoded_fields
    
    return data

def run(config):
    datasets_path = os.path.join(config.paths.data, 'preprocessed_data')
    # grab the original preprocessed data
    full_data = torch.load(os.path.join(datasets_path, f'dataset_155.pt'))
    # encode accordingly
    encoded_data = encode_modes(full_data, config)
    
    # construct appropriate save path
    save_path = os.path.join(datasets_path, f"dataset_{config.model.modelstm.method}.pt")
    if os.path.exists(save_path):
        raise FileExistsError(f"Output file {save_path} already exists!")
    
    # save the new data to disk
    torch.save(encoded_data, save_path)

    
    
    