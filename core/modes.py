import torch
import math
from tqdm import tqdm
from scipy.special import genlaguerre
import os

def svd(x, params):
    """
    Computes the Singular Value Decomposition on the input data
    
    Args:
        x (tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        params (dict): configuration parameters
    """
    k = params['top_k'] # top k singular values
    # construct the output tensor
    samples, r_i, xdim, ydim, slices = x.size()
    x_svd = torch.zeros(samples, r_i, xdim, k, slices)
    
    # Compute SVD
    for i in tqdm(range(samples), desc='Processing Samples'):
        for j in range(slices):
            # selecting the matrix of a single slice of a single sample in r and i
            real = x[i, 0, :, :, j] # [166, 166]
            imag = x[i, 1, :, :, j] # [166, 166]
            
            # perform SVD on each channel separately 
            u_real, s_real, vh_real = torch.linalg.svd(real, full_matrices=False)
            u_r_k = u_real[:, :k]
            s_r_k = s_real[:k]
            # embed
            real_emb = u_r_k * s_r_k # [166, k]
            
            u_imag, s_imag, vh_imag = torch.linalg.svd(imag, full_matrices=False)
            u_i_k = u_imag[:, :k]
            s_i_k = s_imag[:k]
            # embed
            imag_emb = u_i_k * s_i_k # [166, k]
            
            # store results accordingly in output tensor
            x_svd[i, 0, :, :, j] = real_emb
            x_svd[i, 1, :, :, j] = imag_emb
    
    return x_svd

def random_proj(x, params):
    """
    Computes a random projection/Johnson Lindenstrauss for the data.
    For simplicity, we create a random projection matrix to reduce ydim to k
    
    Args:
        x (tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        params (dict): configuration parameters
    """
    # i_dims is our latent dimensionality
    k = params['i_dims']
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
    torch.manual_seed(params['seed'])
    
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

def gauss_laguerre_proj(x, params):
    """
    Project the input field onto Gaussian-Laguerre modes.

    Args:
        x (tensor): [samples, r_i, xdim, ydim, slices]
        params (dict): includes 'i_dims', 'top_k', 'w0', 'p_max', 'l_max'
    """
    k = params['i_dims']
    w0 = params['w0']
    p_max = params['p_max']
    l_max = params['l_max']
    
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
    
def fourier_modes(x, params):
    """
    Encoding fourier modes on the input data
    
    Args:
        x (tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        params (dict): configuration parameters
    """
    pass

def encode_modes(data, params):
    """Takes the input formatted dataset and applies a specified modal decomposition

    Args:
        data (tensor): the dataset
        params (dict): mode encoding parameters
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset with encoded data
    """
    near_fields = data['near_fields'].clone()
    
    method = params['method']
    print(f"Performing '{method}' encoding...")
    
    if method == 'svd': # encoding singular value decomposition
        encoded_fields = svd(near_fields, params)
    elif method == 'random': # random projection / Johnson-Lindenstrauss
        encoded_fields = random_proj(near_fields, params)
    elif method == 'gauss': # gauss-laguerre modes
        encoded_fields = gauss_laguerre_proj(near_fields, params)
    elif method == 'fourier': # a fourier encoding
        encoded_fields = fourier_modes(near_fields, params)
    else:
        raise NotImplementedError(f"Mode encoding method '{method}' not recognized.")
    
    # update the real data
    data['near_fields'] = encoded_fields
    
    return data

def run(params):
    datasets_path = os.path.join(params['path_root'], params['path_data'])
    # grab the original preprocessed data
    full_data = torch.load(os.path.join(datasets_path, 'dataset.pt'))
    # encode accordingly
    encoded_data = encode_modes(full_data, params['modelstm'])
    
    # construct appropriate save path
    save_path = os.path.join(datasets_path, f"dataset_{params['modelstm']['method']}.pt")
    if os.path.exists(save_path):
        raise FileExistsError(f"Output file {save_path} already exists!")
    
    # save the new data to disk
    torch.save(encoded_data, save_path)

    
    
    