import torch.nn as nn
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MetaInfer(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(MetaInfer, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=7, stride=3, padding =1),
            nn.BatchNorm2d(1)
        )
        self.fc = nn.Linear(1 * 8 * 8, out_dim)  # Adjusted input size to match the flattened size

    def forward(self, x):
        x = self.conv(x)

        bs, *_ = x.shape
        x = self.fc(x.view(bs, -1))
        return x

class MARL(nn.Module):
    def __init__(self, vqae, add_downstream=False, year_label_num=None, category_num=None, num_second_use=None):
        super(MARL, self).__init__()
        self.vqae = vqae
        self.use_downstream = add_downstream

        if add_downstream:
            # Existing tasks
            self.height_infer = MetaInfer(32, 1).to(device)
            self.age_infer = MetaInfer(32, year_label_num).to(device)
            self.category_infer = MetaInfer(32, category_num).to(device)

            # NEW: Always create these (no if checks)
            self.orientation_infer = MetaInfer(32, 1).to(device)           # MainOrientation
            self.street_width_infer = MetaInfer(32, 1).to(device)          # MainStreetWidth
            self.second_use_infer = MetaInfer(32, num_second_use).to(device)  # SecondUseDescription

            # Save number of classes for reference
            self.num_second_use = num_second_use
    def forward(self, x):
        if not self.use_downstream:
            return {'vqae': self.vqae(x)}
        
        # recon
        latent = self.vqae.encode(x)
        latent_vector = self.vqae.vector_quantization(latent)
        vq_loss, data_recon, perplexity = self.vqae.decode(latent)
        
        # downstream
        height_pred = self.height_infer(latent)
        age_pred = F.softmax(self.age_infer(latent), dim=1)
        category_pred = torch.sigmoid(self.category_infer(latent))
        orientation_pred = self.orientation_infer(latent)           # regression
        street_width_pred = self.street_width_infer(latent)         # regression
        second_use_pred = F.softmax(self.second_use_infer(latent), dim=1)  # classification
        return {
            'latent': latent,
            'vqae': [vq_loss, data_recon, perplexity],
            'height': height_pred,
            'age': age_pred,
            'category': category_pred,
            'orientation': orientation_pred,
            'street_width': street_width_pred,
            'second_use': second_use_pred
        }