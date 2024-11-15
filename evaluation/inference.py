import torch

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