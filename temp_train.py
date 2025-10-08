from model.VQAE import VQAE
from model.MARL import MARL
from utils import device, add_noise
from tqdm import tqdm
import torch
from data.FloorPlanLoader import *
import torch.nn.functional as F
import random
import json
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


# Accuracy tracking functions
def calculate_accuracy(predictions, targets, task_type='classification'):
    """Calculate accuracy for different task types"""
    with torch.no_grad():
        if task_type == 'classification':
            # For classification tasks (age, second_use)
            pred_classes = torch.argmax(predictions, dim=1)
            return accuracy_score(targets.cpu().numpy(), pred_classes.cpu().numpy())
        elif task_type == 'multilabel':
            # For multilabel classification (category)
            pred_probs = torch.sigmoid(predictions)
            pred_labels = (pred_probs > 0.5).float()
            # Calculate accuracy as percentage of correct predictions
            correct = (pred_labels == targets).all(dim=1).float()
            return correct.mean().item()
        elif task_type == 'regression':
            # For regression tasks (height, orientation, street_width)
            mae = mean_absolute_error(targets.cpu().numpy(), predictions.cpu().numpy())
            r2 = r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
            return mae, r2
    return 0.0


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
n_embeddings = 256
beta = .25
lr = 1e-3
epochs = 100
noise=False
noise_weight=0.05
img_channel=3 if USE_MULTISCALE else 1

dataset_name = 'warehouse_parking'
dataset_name_m ='Warehouse_parking'

def train_marl(train_loader=None, validation_loader=None, 
               data_variance=None, val_len=None, year_label_num=None, category_num=None, num_second_use=None,
               get_pretrain=True, use_multi_task=USE_MULTITASK):
    
    vqae = VQAE(n_hiddens, n_residual_hiddens, n_residual_layers,
                n_embeddings, embedding_dim, 
                beta, img_channel).to(device)
    if get_pretrain:
        checkpoint = torch.load("./best_checkpoint/best.pt")
        # extract model_state_dict (VQAE is part of MARL)
        vqae.load_state_dict(checkpoint['model_state_dict'], strict=False)

    marl = MARL(vqae, USE_MULTITASK, year_label_num, category_num, num_second_use)
    optimizer = torch.optim.Adam(marl.parameters(), lr=lr, amsgrad=False)
    train_recon_error = []
    train_height_error = []
    train_age_error = []
    train_usage_error = []
    train_orientation_error = []
    train_streetwidth_error = []
    train_seconduse_error = []
    
    # Accuracy tracking lists
    train_age_accuracy = []
    train_category_accuracy = []
    train_second_use_accuracy = []
    train_height_mae = []
    train_height_r2 = []
    train_orientation_mae = []
    train_orientation_r2 = []
    train_streetwidth_mae = []
    train_streetwidth_r2 = []
    test_recon_error = []
    test_height_error = []
    test_age_error = []
    test_usage_error = []
    test_orientation_error = []
    test_streetwidth_error = []
    test_seconduse_error = []
    
    # Test accuracy tracking lists
    test_age_accuracy = []
    test_category_accuracy = []
    test_second_use_accuracy = []
    test_height_mae = []
    test_height_r2 = []
    test_orientation_mae = []
    test_orientation_r2 = []
    test_streetwidth_mae = []
    test_streetwidth_r2 = []

    patience = 10
    best_loss = 1e10
    for epoch in range(0, epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            marl.train()
            for data_dict in tepoch:
                data = data_dict['image_tensor']
                bs = data.shape[0]
                data_no_noise = data.to(device)
                optimizer.zero_grad()

                if noise:
                    data = add_noise(data_no_noise, noise_weight=noise_weight)
                else:
                    data = data_no_noise
                pred = marl(data)

                # recon loss
                vq_loss, data_recon, perplexity = pred['vqae']
                recon_error = F.mse_loss(data_recon, data) / data_variance
                train_recon_error.append(recon_error.item())
                

                if USE_MULTITASK:
                    # height infer
                    height_pred = pred['height']
                    height_error = F.mse_loss(height_pred, data_dict['height'].to(device).view(bs,-1))
                    train_height_error.append(height_error.item())
                    # Calculate height accuracy (MAE and R²)
                    height_mae, height_r2 = calculate_accuracy(height_pred, data_dict['height'].to(device).view(bs,-1), 'regression')
                    train_height_mae.append(height_mae)
                    train_height_r2.append(height_r2)
                    
                    # age infer
                    age_pred = pred['age']
                    labels = data_dict['age_label'].to(device).long()
                    age_error = F.cross_entropy(age_pred, labels)*0.3
                    train_age_error.append(age_error.item())
                    # Calculate age accuracy
                    age_acc = calculate_accuracy(age_pred, labels, 'classification')
                    train_age_accuracy.append(age_acc)
                    
                    # category infer
                    category_pred = pred['category']
                    labels = data_dict['cate_onehot'].to(device)
                    criterion = torch.nn.BCEWithLogitsLoss()
                    category_error = criterion(category_pred, labels)*0.7
                    train_usage_error.append(category_error.item())
                    # Calculate category accuracy
                    category_acc = calculate_accuracy(category_pred, labels, 'multilabel')
                    train_category_accuracy.append(category_acc)

                    # orientation infer
                    orientation_pred = pred['orientation']
                    orientation_error = F.mse_loss(orientation_pred, data_dict['orientation'].to(device).view(bs,-1))
                    train_orientation_error.append(orientation_error.item())
                    # Calculate orientation accuracy (MAE and R²)
                    orientation_mae, orientation_r2 = calculate_accuracy(orientation_pred, data_dict['orientation'].to(device).view(bs,-1), 'regression')
                    train_orientation_mae.append(orientation_mae)
                    train_orientation_r2.append(orientation_r2)

                    # street_width infer
                    streetwidth_pred = pred['street_width']
                    streetwidth_error = F.mse_loss(streetwidth_pred, data_dict['street_width'].to(device).view(bs,-1))
                    train_streetwidth_error.append(streetwidth_error.item())
                    # Calculate street width accuracy (MAE and R²)
                    streetwidth_mae, streetwidth_r2 = calculate_accuracy(streetwidth_pred, data_dict['street_width'].to(device).view(bs,-1), 'regression')
                    train_streetwidth_mae.append(streetwidth_mae)
                    train_streetwidth_r2.append(streetwidth_r2)

                    # Second use (classification)
                    second_use_pred = pred['second_use']
                    second_use_labels = data_dict['second_use_label'].to(device).long()
                    second_use_error = F.cross_entropy(second_use_pred, second_use_labels)
                    train_seconduse_error.append(second_use_error.item())
                    # Calculate second use accuracy
                    second_use_acc = calculate_accuracy(second_use_pred, second_use_labels, 'classification')
                    train_second_use_accuracy.append(second_use_acc)

                loss = (recon_error + vq_loss) + height_error + age_error + category_error + orientation_error + streetwidth_error + second_use_error   
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(recon_error=float((recon_error+ vq_loss).detach().cpu()),
                                   height_error=float(height_error.detach().cpu()),
                                   age_error=float(age_error.detach().cpu()),
                                   category_error=float(category_error.detach().cpu()),
                                   orientation_error=float(orientation_error.detach().cpu()),
                                   streetwidth_error=float(streetwidth_error.detach().cpu()),
                                   seconduse_error=float(second_use_error.detach().cpu()))   
                
        avg_loss = 0
        marl.eval()
        with torch.no_grad():
            for data_dict in validation_loader:
                data = data_dict['image_tensor']
                bs = data.shape[0]
                data = data.to(device)

                pred = marl(data)
                # recon loss
                vq_loss, data_recon, perplexity = pred['vqae']
                recon_error = F.mse_loss(data_recon, data) / data_variance
                test_recon_error.append(recon_error.item())

                if USE_MULTITASK:
                    # height infer
                    height_pred = pred['height']
                    height_error = F.mse_loss(height_pred, data_dict['height'].to(device).view(bs,-1))
                    test_height_error.append(height_error.item())
                    # Calculate height accuracy (MAE and R²)
                    height_mae, height_r2 = calculate_accuracy(height_pred, data_dict['height'].to(device).view(bs,-1), 'regression')
                    test_height_mae.append(height_mae)
                    test_height_r2.append(height_r2)
                    
                    # age infer
                    age_pred = pred['age']
                    labels = data_dict['age_label'].to(device).long()
                    age_error = F.cross_entropy(age_pred, labels)*0.3
                    test_age_error.append(age_error.item())
                    # Calculate age accuracy
                    age_acc = calculate_accuracy(age_pred, labels, 'classification')
                    test_age_accuracy.append(age_acc)
                    
                    # category infer
                    category_pred = pred['category']
                    labels = data_dict['cate_onehot'].to(device)
                    criterion = torch.nn.BCEWithLogitsLoss()
                    category_error = criterion(category_pred, labels)*0.7
                    test_usage_error.append(category_error.item())
                    # Calculate category accuracy
                    category_acc = calculate_accuracy(category_pred, labels, 'multilabel')
                    test_category_accuracy.append(category_acc)

                    # orientation infer
                    orientation_pred = pred['orientation']
                    orientation_error = F.mse_loss(orientation_pred, data_dict['orientation'].to(device).view(bs,-1))
                    test_orientation_error.append(orientation_error.item())
                    # Calculate orientation accuracy (MAE and R²)
                    orientation_mae, orientation_r2 = calculate_accuracy(orientation_pred, data_dict['orientation'].to(device).view(bs,-1), 'regression')
                    test_orientation_mae.append(orientation_mae)
                    test_orientation_r2.append(orientation_r2)

                    # street_width infer
                    streetwidth_pred = pred['street_width']
                    streetwidth_error = F.mse_loss(streetwidth_pred, data_dict['street_width'].to(device).view(bs,-1))
                    test_streetwidth_error.append(streetwidth_error.item())
                    # Calculate street width accuracy (MAE and R²)
                    streetwidth_mae, streetwidth_r2 = calculate_accuracy(streetwidth_pred, data_dict['street_width'].to(device).view(bs,-1), 'regression')
                    test_streetwidth_mae.append(streetwidth_mae)
                    test_streetwidth_r2.append(streetwidth_r2)

                    # Second use (classification)
                    second_use_pred = pred['second_use']
                    second_use_labels = data_dict['second_use_label'].to(device).long()
                    second_use_error = F.cross_entropy(second_use_pred, second_use_labels)
                    test_seconduse_error.append(second_use_error.item())
                    # Calculate second use accuracy
                    second_use_acc = calculate_accuracy(second_use_pred, second_use_labels, 'classification')
                    test_second_use_accuracy.append(second_use_acc)

                    loss = (recon_error.item() \
                            + height_error.item()\
                            + age_error.item()\
                            + category_error.item()\
                            + orientation_error.item()\
                            + streetwidth_error.item()\
                            + second_use_error.item()\
                            )
                    avg_loss += loss 
                
        avg_loss /= len(validation_loader)       
        if avg_loss<best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': marl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"./checkpoint/best_{dataset_name}.pt")
            print(f"✅ Best model updated at epoch {epoch} with val_loss={best_loss:.4f}")
            
            
            epochs_no_improve = 0
            torch.save(marl.state_dict(), f"./checkpoint/{best_epoch}-marl-{best_loss}.pt")
            torch.save(optimizer.state_dict(), f"./checkpoint/{best_epoch}-adam-{best_loss}.pt")
            if USE_MULTITASK:
                error = {
                    'train_recon_error': train_recon_error,
                    'train_height_error': train_height_error,
                    'train_age_error': train_age_error,
                    'train_usage_error': train_usage_error,
                    'train_orientation_error': train_orientation_error,
                    'train_streetwidth_error': train_streetwidth_error,
                    'train_seconduse_error': train_seconduse_error,
                    'test_recon_error': test_recon_error,
                    'test_height_error': test_height_error,
                    'test_age_error': test_age_error,
                    'test_usage_error': test_usage_error,
                    'test_orientation_error': test_orientation_error,
                    'test_streetwidth_error': test_streetwidth_error,
                    'test_seconduse_error': test_seconduse_error,
                    # Add accuracy metrics
                    'train_age_accuracy': train_age_accuracy,
                    'train_category_accuracy': train_category_accuracy,
                    'train_second_use_accuracy': train_second_use_accuracy,
                    'train_height_mae': train_height_mae,
                    'train_height_r2': train_height_r2,
                    'train_orientation_mae': train_orientation_mae,
                    'train_orientation_r2': train_orientation_r2,
                    'train_streetwidth_mae': train_streetwidth_mae,
                    'train_streetwidth_r2': train_streetwidth_r2,
                    'test_age_accuracy': test_age_accuracy,
                    'test_category_accuracy': test_category_accuracy,
                    'test_second_use_accuracy': test_second_use_accuracy,
                    'test_height_mae': test_height_mae,
                    'test_height_r2': test_height_r2,
                    'test_orientation_mae': test_orientation_mae,
                    'test_orientation_r2': test_orientation_r2,
                    'test_streetwidth_mae': test_streetwidth_mae,
                    'test_streetwidth_r2': test_streetwidth_r2,
                }
            else:
                error = {
                    'train_recon_error': train_recon_error,
                    'test_recon_error': test_recon_error
                }
            with open(f"./checkpoint/{best_epoch}-error-{best_loss}.json", 'w', encoding ='utf8') as json_file:
                json.dump(error, json_file, ensure_ascii = False)

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early sttoping. Best epoch:{best_epoch}")
                break
        
        print(f'Validation Loss: {avg_loss}')
        
        # Print accuracy metrics
        if USE_MULTITASK and test_age_accuracy:
            print(f'📊 Accuracy Metrics:')
            print(f'   Age Accuracy: {np.mean(test_age_accuracy[-len(validation_loader):]):.4f}')
            print(f'   Category Accuracy: {np.mean(test_category_accuracy[-len(validation_loader):]):.4f}')
            print(f'   Second Use Accuracy: {np.mean(test_second_use_accuracy[-len(validation_loader):]):.4f}')
            print(f'   Height MAE: {np.mean(test_height_mae[-len(validation_loader):]):.4f} (R²: {np.mean(test_height_r2[-len(validation_loader):]):.4f})')
            print(f'   Orientation MAE: {np.mean(test_orientation_mae[-len(validation_loader):]):.4f} (R²: {np.mean(test_orientation_r2[-len(validation_loader):]):.4f})')
            print(f'   Street Width MAE: {np.mean(test_streetwidth_mae[-len(validation_loader):]):.4f} (R²: {np.mean(test_streetwidth_r2[-len(validation_loader):]):.4f})')

    # Save final best epoch results to comprehensive file (only at the end)
    if USE_MULTITASK and test_age_accuracy and best_epoch is not None:
        results_file = "./model_accuracy_results.txt"
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(results_file)
        
        with open(results_file, 'a') as f:  # 'a' for append mode
            if not file_exists:
                f.write(f"MARL MODEL ACCURACY RESULTS - ALL DATASETS\n")
                f.write(f"{'='*80}\n")
                f.write(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
            
            f.write(f"DATASET: {dataset_name.upper()} ({dataset_name_m}) - Final Best Epoch {best_epoch}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Validation Loss: {best_loss:.6f}\n")
            f.write(f"Dataset Size: {len(floor)} samples\n")
            f.write(f"Data Variance: {data_variance:.6f}\n")
            f.write(f"Total Epochs Trained: {epoch + 1}\n")
            f.write(f"{'='*60}\n\n")
            f.write("CLASSIFICATION TASKS:\n")
            f.write(f"   Age Accuracy: {np.mean(test_age_accuracy[-len(validation_loader):]):.4f} ({np.mean(test_age_accuracy[-len(validation_loader):])*100:.2f}%)\n")
            f.write(f"   Category Accuracy: {np.mean(test_category_accuracy[-len(validation_loader):]):.4f} ({np.mean(test_category_accuracy[-len(validation_loader):])*100:.2f}%)\n")
            f.write(f"   Second Use Accuracy: {np.mean(test_second_use_accuracy[-len(validation_loader):]):.4f} ({np.mean(test_second_use_accuracy[-len(validation_loader):])*100:.2f}%)\n\n")
            f.write("REGRESSION TASKS:\n")
            f.write(f"   Height MAE: {np.mean(test_height_mae[-len(validation_loader):]):.6f} (R²: {np.mean(test_height_r2[-len(validation_loader):]):.4f})\n")
            f.write(f"   Orientation MAE: {np.mean(test_orientation_mae[-len(validation_loader):]):.6f} (R²: {np.mean(test_orientation_r2[-len(validation_loader):]):.4f})\n")
            f.write(f"   Street Width MAE: {np.mean(test_streetwidth_mae[-len(validation_loader):]):.6f} (R²: {np.mean(test_streetwidth_r2[-len(validation_loader):]):.4f})\n\n")
            f.write("LOSS METRICS:\n")
            f.write(f"   Reconstruction Error: {np.mean(test_recon_error[-len(validation_loader):]):.6f}\n")
            f.write(f"   Height Error: {np.mean(test_height_error[-len(validation_loader):]):.6f}\n")
            f.write(f"   Age Error: {np.mean(test_age_error[-len(validation_loader):]):.6f}\n")
            f.write(f"   Category Error: {np.mean(test_usage_error[-len(validation_loader):]):.6f}\n")
            f.write(f"   Orientation Error: {np.mean(test_orientation_error[-len(validation_loader):]):.6f}\n")
            f.write(f"   Street Width Error: {np.mean(test_streetwidth_error[-len(validation_loader):]):.6f}\n")
            f.write(f"   Second Use Error: {np.mean(test_seconduse_error[-len(validation_loader):]):.6f}\n\n")
            f.write(f"Model saved as: ./checkpoint/{best_epoch}-marl-{best_loss}.pt\n")
            f.write(f"{'='*80}\n\n")
        print(f"💾 Final best epoch results saved to: {results_file}")


if __name__ == "__main__":
    #Load Dataset
    floor = FloorPlanDataset(multi_scale=True, root=f'./data/data_br/{dataset_name}/', data_config='./data/data_config_1/tertiary/Warehouse_parking/', preprocess=True)
    data_variance = floor.var
    val_len = int(len(floor)/10)
    train_set, val_set = torch.utils.data.random_split(floor, [len(floor)-val_len, val_len])

    print(f"data shape: {floor[0]['image_tensor'].shape}, dataset size: {len(floor)}, data variance: {data_variance}")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False)

    train_marl(train_loader, validation_loader, \
               floor.var, int(len(floor)/10), floor.age_label_num, floor.category_num, floor.num_second_use)

