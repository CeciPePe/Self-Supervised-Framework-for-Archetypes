from model.VQAE import VQAE
from model.MARL import MARL
from utils import device, add_noise
from tqdm import tqdm
import torch
from data.FloorPlanLoader import *
import torch.nn.functional as F
import random
import json
import os
import shutil



USE_MULTISCALE = True
USE_MULTITASK = True

#Reproducability Checks:
random.seed(0) #Python
torch.manual_seed(0) #Torch
np.random.seed(0) #NumPy
#Hyperparameter
batch_size = 128
n_hiddens = 32
n_residual_hiddens = 32
n_residual_layers = 1
embedding_dim = 64
n_embeddings = 218
beta = .25
lr = 1e-3
epochs = 50
noise=False
noise_weight=0.05
img_channel=3 if USE_MULTISCALE else 1


def train_marl(train_loader, validation_loader, data_variance, val_len, floor, get_pretrain=False, use_multi_task=USE_MULTITASK):
    """
    Train the MARL model with VQ-AE and downstream tasks.
    """
    print("Initializing VQ-AE and MARL model...")

    # Initialize VQ-AE
    vqae = VQAE(n_hiddens, n_residual_hiddens, n_residual_layers,
                n_embeddings, embedding_dim, 
                beta, img_channel).to(device)

    if get_pretrain:
        print("Loading pretrained VQ-AE weights...")
        vqae.load_state_dict(
            torch.load("./best_checkpoint/55-vqae-0.04753296934928414.pt", map_location='cpu')
        )

    # Initialize MARL with downstream heads
    marl = MARL(
        vqae=vqae,
        add_downstream=use_multi_task,
        year_label_num=floor.age_label_num,
        category_num=floor.category_num,
        num_second_use=floor.num_second_use
    ).to(device)

    optimizer = torch.optim.Adam(marl.parameters(), lr=lr, weight_decay=1e-5, amsgrad=False)

    # Loss functions
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # Reused across batches

    # Tracking losses
    train_recon_error = []
    train_height_error = []
    train_age_error = []
    train_usage_error = []
    train_orientation_error = []
    train_street_width_error = []
    train_second_use_error = []

    test_recon_error = []
    test_height_error = []
    test_age_error = []
    test_usage_error = []
    test_orientation_error = []
    test_street_width_error = []
    test_second_use_error = []

    # Early stopping setup
    patience = 8
    counter = 0
    best_smoothed_loss = float('inf')
    min_delta = 1e-4
    smoothing_window = 5
    from collections import deque
    val_loss_history = deque(maxlen=smoothing_window)

    print("Starting training...")

    for epoch in range(epochs):
        marl.train()
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
            for data_dict in tepoch:
                data = data_dict['image_tensor'].to(device)
                bs = data.shape[0]
                optimizer.zero_grad()

                # Add noise (if enabled)
                if noise:
                    data = add_noise(data, noise_weight=noise_weight)

                # Forward pass
                pred = marl(data)
                vq_loss, data_recon, perplexity = pred['vqae']

                # Reconstruction loss (do NOT divide by variance)
                recon_error = F.mse_loss(data_recon, data) / data_variance
                loss = recon_error + vq_loss

                height_error = age_error = category_error = orientation_error = street_width_error = second_use_error = 0.0

                if use_multi_task:
                    # Height (regression)
                    height_pred = pred['height']
                    height_target = data_dict['height'].to(device).view(bs, -1)
                    height_error = F.mse_loss(height_pred, height_target)
                    loss += height_error

                    # Age (classification)
                    age_pred = pred['age']
                    age_labels = data_dict['age_label'].to(device).long()
                    age_error = F.cross_entropy(age_pred, age_labels) * 0.3
                    loss += age_error

                    # Category (multi-label)
                    category_pred = pred['category']
                    cate_labels = data_dict['cate_onehot'].to(device)
                    category_error = bce_criterion(category_pred, cate_labels) * 0.7
                    loss += category_error

                    # Orientation (regression)
                    orientation_pred = pred['orientation']
                    orientation_target = data_dict['orientation'].to(device).view(bs, -1)
                    orientation_error = F.mse_loss(orientation_pred, orientation_target)
                    loss += orientation_error

                    # Street width (regression)
                    street_width_pred = pred['street_width']
                    street_width_target = data_dict['street_width'].to(device).view(bs, -1)
                    street_width_error = F.mse_loss(street_width_pred, street_width_target)
                    loss += street_width_error

                    # Second use (classification)
                    second_use_pred = pred['second_use']
                    second_use_labels = data_dict['second_use_label'].to(device).long()
                    second_use_error = F.cross_entropy(second_use_pred, second_use_labels)
                    loss += second_use_error

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(marl.parameters(), max_norm=1.0)  # Prevent exploding gradients
                optimizer.step()

                # Log training losses
                train_recon_error.append(recon_error.item())
                if use_multi_task:
                    train_height_error.append(height_error.item())
                    train_age_error.append(age_error.item())
                    train_usage_error.append(category_error.item())
                    train_orientation_error.append(orientation_error.item())
                    train_street_width_error.append(street_width_error.item())
                    train_second_use_error.append(second_use_error.item())

                tepoch.set_postfix(
                    recon=recon_error.item(),
                    height=height_error if isinstance(height_error, float) else height_error.item(),
                    age=age_error if isinstance(age_error, float) else age_error.item(),
                    usage=category_error if isinstance(category_error, float) else category_error.item(),
                    orient=orientation_error if isinstance(orientation_error, float) else orientation_error.item(),
                    width=street_width_error if isinstance(street_width_error, float) else street_width_error.item(),
                    second=second_use_error if isinstance(second_use_error, float) else second_use_error.item()
                )

        # Validation
        marl.eval()
        avg_val_loss = 0.0
        with torch.no_grad():
            for data_dict in validation_loader:
                data = data_dict['image_tensor'].to(device)
                pred = marl(data)
                vq_loss, data_recon, _ = pred['vqae']
                recon_error = F.mse_loss(data_recon, data)/ data_variance
                loss = recon_error + vq_loss

                if use_multi_task:
                    height_pred = pred['height']
                    height_error = F.mse_loss(height_pred, data_dict['height'].to(device).view(-1, 1))

                    age_pred = pred['age']
                    age_labels = data_dict['age_label'].to(device).long()
                    age_error = F.cross_entropy(age_pred, age_labels)

                    category_pred = pred['category']
                    cate_labels = data_dict['cate_onehot'].to(device)
                    category_error = bce_criterion(category_pred, cate_labels)

                    orientation_pred = pred['orientation']
                    orientation_error = F.mse_loss(orientation_pred, data_dict['orientation'].to(device).view(-1, 1))

                    street_width_pred = pred['street_width']
                    street_width_error = F.mse_loss(street_width_pred, data_dict['street_width'].to(device).view(-1, 1))

                    second_use_pred = pred['second_use']
                    second_use_labels = data_dict['second_use_label'].to(device).long()
                    second_use_error = F.cross_entropy(second_use_pred, second_use_labels)

                    loss += (height_error + age_error + category_error +
                             orientation_error + street_width_error + second_use_error)

                avg_val_loss += loss.item() * batch_size

            avg_val_loss /= val_len
            val_loss_history.append(avg_val_loss)
            smoothed_loss = sum(val_loss_history) / len(val_loss_history)

        # Logging
        print(f"Epoch {epoch}: Smoothed Val Loss = {smoothed_loss:.6f} | Best = {best_smoothed_loss:.6f}")

        # Save best model
        if smoothed_loss < best_smoothed_loss - min_delta:
            best_smoothed_loss = smoothed_loss
            best_epoch = epoch
            best_model_path = f"./checkpoint/{epoch}-marl-{smoothed_loss:.6f}.pt"
            torch.save(marl.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")

            # Save to category folder
            category_folder = os.path.join("./checkpoint", "Residential")
            os.makedirs(category_folder, exist_ok=True)
            shutil.copy(best_model_path, os.path.join(category_folder, "best.pt"))

            # Save optimizer and errors
            torch.save(optimizer.state_dict(), f"./checkpoint/{best_epoch}-adam.pt")
            error_dict = {
                'train_recon_error': train_recon_error,
                'test_recon_error': test_recon_error,
                'train_height_error': train_height_error,
                'test_height_error': test_height_error,
                'train_age_error': train_age_error,
                'test_age_error': test_age_error,
                'train_usage_error': train_usage_error,
                'test_usage_error': test_usage_error,
                'train_orientation_error': train_orientation_error,
                'test_orientation_error': test_orientation_error,
                'train_street_width_error': train_street_width_error,
                'test_street_width_error': test_street_width_error,
                'train_second_use_error': train_second_use_error,
                'test_second_use_error': test_second_use_error,
            }
            with open(f"./checkpoint/{best_epoch}-error.json", 'w') as f:
                json.dump({k: [float(x) for x in v] for k, v in error_dict.items()}, f)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best smoothed loss: {best_smoothed_loss:.6f}")
                break

    # Save final model
    torch.save(marl.state_dict(), "./checkpoint/last-marl.pt")
    print("Training complete. Final model saved.")


# Main
if __name__ == "__main__":
    print("Loading dataset...")
    floor = FloorPlanDataset(
        multi_scale=True,
        root='data/data_br/residential/',
        data_config='data/data_config_1/residential/Residential/',
        preprocess=True
    )

    print(f"Dataset loaded. Size: {len(floor)}, Data shape: {floor[0]['image_tensor'].shape}, Variance: {floor.var:.6f}")
    print(f"Classes: age={floor.age_label_num}, category={floor.category_num}, second_use={floor.num_second_use}")

    val_len = int(len(floor) * 0.1)
    train_set, val_set = torch.utils.data.random_split(floor, [len(floor) - val_len, val_len])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    train_marl(
        train_loader=train_loader,
        validation_loader=validation_loader,
        data_variance=floor.var,
        val_len=val_len,
        floor=floor,
        get_pretrain=False,
        use_multi_task=USE_MULTITASK
    )