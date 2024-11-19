import torch
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import utils.mapping as mapping

def predict_sequence(model, initial_field, num_steps=5):
    """Performs inference on an LSTM model given an initial field  
       to propagate for num_steps iterations.

    Args:
        model (WaveLSTM): The trained model
        initial_field (tensor): (batch=1, seq_len=1, r_i*xdim*ydim)
        num_steps (int, optional): How far to propagate. Defaults to 5.
    """
    model.eval()
    
    with torch.no_grad():
        current_input = initial_field
        predictions = []
        
        for _ in range(num_steps):
            pred, _ = model(current_input)
            predictions.append(pred)
            current_input = pred
    
    # performing some serious reshaping to ensure we return in (r_i, 166, 166, num_steps)       
    predictions = [pred.squeeze(0).squeeze(0).view(2, 166, 166, 1) for pred in predictions]    
    return torch.cat(predictions, dim=-1)

def predict_next_field(model, initial_delta):
    """Standard inference - give the model seq_len slices to recieve  
       its seq_len+1 prediction

    Args:
        model (WaveLSTM): The trained model
        initial_delta (tensor): (batch=1, seq_len, r_i*xdim*ydim)
    """
    model.eval()
    
    with torch.no_grad():
        pred, _ = model(initial_delta)
        print(pred.shape)
        
    pred = pred.squeeze(0).view(5, 2, 166, 166).permute(1, 2, 3, 0)
    return pred

def ae_reconstruction(model, sample):
    model.eval()
    
    with torch.no_grad():
        pred = model(sample)
        
    # reshape for plotting
    pred = pred.squeeze(0)
    sample = sample.squeeze(0)
    
    truth_component_1, truth_component_2 = mapping.cartesian_to_polar(sample[0, :, :], sample[1, :, :])
    pred_component_1, pred_component_2 = mapping.cartesian_to_polar(pred[0, :, :], pred[1, :, :])
      
    # plot original next to reconstructed for mag and phase
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(truth_component_1.detach().cpu().numpy(), cmap="viridis")
    axs[0, 1].imshow(pred_component_1.detach().cpu().numpy(), cmap="viridis")
    axs[1, 0].imshow(truth_component_2.detach().cpu().numpy(), cmap="twilight_shifted")
    axs[1, 1].imshow(pred_component_2.detach().cpu().numpy(), cmap="twilight_shifted")
    plt.show()
    
    return pred