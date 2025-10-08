import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_noise(tensor, mean=0., std=1., noise_weight=0.5):
    noise = torch.randn(tensor.size()).to(device) * std + mean
    return torch.clip(tensor + noise_weight * noise, 0., 1.)

def get_data_variance(dataset):
    # Handle both tensor and dict formats
    if isinstance(dataset[0], dict):
        # If dataset returns dict, extract the image tensor
        return np.var(np.array([dataset[i]['image_tensor'].numpy() for i in range(0, min(100, len(dataset)))]))
    else:
        # If dataset returns tensor directly
        return np.var(np.array([dataset[i].numpy() for i in range(0, min(100, len(dataset)))]))


