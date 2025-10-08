#!/usr/bin/env python
# coding: utf-8

# In[ ]:


dataset_name = 'warehouse_parking'
dataset_name_m ='Warehouse_parking'  # Current dataset name
curr_data = "BR_Warehouse_parking"  # Current data folder name


# In[ ]:


import sys
sys.path.append('../')
from model.VQAE import VQAE
from model.MARL import MARL
from utils import device, add_noise
from tqdm import tqdm
import torch
from data.FloorPlanLoader import *
import torch.nn.functional as F
import random
import json
import torchvision
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

USE_MULTISCALE = True
USE_MULTITASK = True

#Reproducability Checks:
random.seed(0) #Python
torch.manual_seed(0) #Torch
np.random.seed(0) #NumPy


import torchvision
import torchvision.transforms.functional as TF
import imageio
from torchvision.utils import save_image
# %load_ext autoreload
# %autoreload 2
import csv
import numpy
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


# In[ ]:


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


# ### prepare latent.pt file

# In[ ]:


#Load Dataset

floor = FloorPlanDataset(multi_scale=True, root=f'./data/data_br/{dataset_name}/',\
                         data_config=f'./data/data_config_1/tertiary/{dataset_name_m}/', preprocess=True)
data_loader = torch.utils.data.DataLoader(floor, batch_size=batch_size, shuffle=False)
vqae = VQAE(n_hiddens, n_residual_hiddens, n_residual_layers,
            n_embeddings, embedding_dim, 
            beta, img_channel).to(device)

marl = MARL(vqae, True, floor.age_label_num, floor.category_num, floor.num_second_use).to(device)
checkpoint = torch.load(f"./checkpoint/best_{dataset_name}.pt", map_location=device)
marl.load_state_dict(checkpoint["model_state_dict"])
# print(f"data shape: {floor[0]['image_tensor'].shape}, dataset size: {len(floor)}")
for param in vqae.parameters():
    param.to('cpu')
for param in marl.parameters():
    param.to('cpu')

marl.eval()
latents = None
with torch.no_grad():
    for data in tqdm(data_loader):
        data = data['image_tensor'].to(device)
        valid_recon = marl(data)
        quantized = valid_recon['latent']
        if latents is not None:
            latents = torch.cat([latents, quantized], dim=0)
        else: 
            latents = quantized
latent_file_path = f'./data/data_root/marl_latent_{curr_data}.pt'
print(f"finished latents configuration, now saving to {latent_file_path} ...")
torch.save(latents, latent_file_path)
print(f"latent saved to {latent_file_path} ...")


# # Embedding Space Visualization
# ## Visualize the latent with UMAP and T-SNE

# In[ ]:


latent_file_path = f'./data/data_root/marl_latent_{curr_data}.pt'
latents = torch.load(latent_file_path)
floor = FloorPlanDataset(multi_scale=True, root=f'./data/data_br/{dataset_name}/',\
                         data_config=f'./data/data_config_1/tertiary/{dataset_name_m}/', preprocess=True)
data_loader = torch.utils.data.DataLoader(floor, batch_size=batch_size, shuffle=False)


# In[ ]:


#latent_file_path = f'./data/data_br/{dataset_name}/'
#pt_files = [f for f in os.listdir(latent_file_path) if f.endswith('.pt')]

# Load each file into a list
#latents = [torch.load(os.path.join(latent_file_path, f)) for f in pt_files]
#floor = FloorPlanDataset(multi_scale=True, root=f'./data/data_br/{dataset_name}/',\
#                         data_config=f'./data/data_config_1/tertiary/Warehouse_parking/', preprocess=True)
#data_loader = torch.utils.data.DataLoader(floor, batch_size=batch_size, shuffle=False)


# In[ ]:


#latents = torch.stack(latents)  # assuming latents is a list of tensors

neighbors_list = [30,50]

for n_neighbors in neighbors_list:
    umap = UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
    data_2d_umap = umap.fit_transform(torch.flatten(latents, start_dim=1).cpu())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_2d_umap[:, 0], data_2d_umap[:, 1], s=0.5)
    ax.set_title(f'UMAP with n_neighbors={n_neighbors}')
    # plt# .show()  # Disabled for pipeline  # Disabled for pipeline





# In[ ]:


umap = UMAP(n_neighbors =50, n_components=2, random_state=42)
data_2d_umap = umap.fit_transform(torch.flatten(latents, start_dim=1).cpu())
# Create a figure and a 3D axes object
fig = plt.figure()
ax = fig.add_subplot(111)

# Create the initial scatter plot
ax.scatter(data_2d_umap[:, 0], data_2d_umap[:, 1], s=0.5)

# Show the plot
# plt# .show()  # Disabled for pipeline  # Disabled for pipeline


# In[ ]:


perplexities = [5,10,15,20, 30]

for perplexity in perplexities:
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    data_2d_tsne = tsne.fit_transform(torch.flatten(latents, start_dim=1).cpu())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], s=0.5)
    ax.set_title(f't-SNE with perplexity={perplexity}')
    # plt# .show()  # Disabled for pipeline  # Disabled for pipeline


# In[ ]:


tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
data_2d_tsne = tsne.fit_transform(torch.flatten(latents, start_dim=1).cpu())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], s=0.5)
ax.set_title(f't-SNE with perplexity=30')
save_path = f"./notebooks/{dataset_name}/tsne_perp30.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
# plt# .show()  # Disabled for pipeline  # Disabled for pipeline


# ### deciding the number of clusters

# In[ ]:


def calculate_wcss(latents):
    wcss = []
    for n in tqdm(range(1, 8)):  # change the range according to your needs
        kmeans = KMeans(n_clusters=n)
        data = torch.flatten(latents, start_dim=1).cpu()
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # inertia_ is the WCSS for the model
    return wcss
# calculate WCSS for different numbers of clusters
wcss = calculate_wcss(latents)


# In[ ]:


# Convert list to DataFrame
wcss_df = pd.DataFrame(wcss)
wcss_df.to_csv(f'wcss_{curr_data}.csv',index=False)
wcss_df


# In[ ]:


def plot_elbow(wcss, filename=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(wcss) + 1), wcss, 'bo-')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid()
    filename = f'./notebooks/{dataset_name}/elbow_plot_kmeans.png' 
    plt.savefig(filename)


# In[ ]:


# plot the elbow graph
plot_elbow(wcss)


# the number of clusters are set to 5, 10, 15

# In[ ]:


def calculate_slope(points):
    slopes = []
    for i in range(len(points) - 1):
        slope = (points[i + 1] - points[i]) / (i + 2 - i)
        slopes.append(slope)
    return slopes

def find_optimal_point(slopes, drop_threshold=0.1):
    optimal_point = 1
    for i in range(len(slopes)):
        if slopes[i] / slopes[i - 1] <= drop_threshold:
            optimal_point = i + 1
            break
    return optimal_point


# In[ ]:


# Calculate slopes
slopes = calculate_slope(wcss)
print(slopes)
# Find the optimal point based on the slope
optimal_point = find_optimal_point(slopes)

print("Optimal point:", optimal_point)


# In[ ]:


# Plot the slopes
plt.plot(range(1, len(wcss)), slopes, 'bo-')
plt.title('Slope between Points')
plt.xlabel('Point Index')
plt.ylabel('Slope')
plt.grid()
save_path = f"./notebooks/{dataset_name}/slope_plot.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Slope plot saved to {save_path}")
# plt# .show()  # Disabled for pipeline  # Disabled for pipeline


# In[ ]:


from kneed import KneeLocator
log_file = "silhouette_scores.txt"
K = range(1, len(wcss) + 1)
kneedle = KneeLocator(K, wcss, curve='convex', direction='decreasing')
print("Optimal number of clusters:", kneedle.knee)
print(wcss)
with open(log_file, "a") as f:  # "a" = append mode
    f.write(f"\n--- {dataset_name_m} ---\n")
    msg_w = f"elbow k={kneedle.knee}\n"
    print(msg_w.strip())  # still show in terminal
    f.write(msg_w)


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch

# Flatten your latent vectors for clustering
data = torch.flatten(latents, start_dim=1).cpu().numpy()



best_score = -1
optimal_k = 2 

with open(log_file, "a") as f:
    f.write(f"\n--- {dataset_name_m} ---\n")

    for k in range(2, 10):  # try k=2 to 6
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        msg = f"k={k}, silhouette score={score:.4f}\n"
        print(msg.strip())
        f.write(msg)
        if score > best_score:
            best_score = score
            optimal_k = k

    f.write(f"Optimal number of clusters (silhouette): {optimal_k}\n")


# ### Clustering

# In[ ]:


n_clusters = optimal_k 
data_latent = latents # this is the data for clustering
print(n_clusters)


# In[ ]:


#Initialize the class object
kmeans = KMeans(n_clusters=n_clusters)
data = torch.flatten(data_latent, start_dim=1).cpu()
# #predict the labels of clusters.
label = kmeans.fit_predict(data)
# get cluster centers
cluster_centers = kmeans.cluster_centers_
closest, _ = pairwise_distances_argmin_min(cluster_centers, data)
print(cluster_centers.shape, data.shape)



center_list = closest
print(center_list)
df = np.stack((label, floor.all_data_dirs)).transpose()
df = pd.DataFrame(df,columns=['label','data'])


# In[ ]:


def calculate_index_png(index):
    image_path = floor.all_data_dirs[index]  # Replace with appropriate image paths list
    file_name, file_extension = os.path.splitext(image_path)
    num_index = os.path.basename(file_name)
    return num_index


# In[ ]:


# Save closest points and cluster centers to CSV
df_center = pd.DataFrame({'Cluster Center': cluster_centers.tolist(), 'Closest Point Index': closest})
df_center['index_png'] = df_center['Closest Point Index'].apply(lambda x: calculate_index_png(x))
df_center.to_csv(f'./results/recon_img/{dataset_name}/MARL_{curr_data}_{n_clusters}cluster_centers.csv', index=False)

df_center_index = np.stack((label, floor.all_data_dirs)).transpose()
df_center_index = pd.DataFrame(df_center_index,columns=['label','data'])
df_center_index['index'] = df_center_index['data'].str.extract(r'(\d+).pt')
df_center_index = df_center_index.drop(['data'], axis=1)
df_center_index = df_center_index.dropna(subset=['index'])
df_center_index['index'] = df_center_index['index'].astype(int)
df_center_index.to_csv(f'./results/recon_img/{dataset_name}/MARL_{curr_data}_{n_clusters}clusters_clusterindex.csv', index=False)
print(df_center_index.shape)
df_center_index.head(10)


# In[ ]:


center_list = closest
center_list_annotation = df_center['index_png'].tolist()


# In[ ]:


data_2d = data_2d_umap # this is the data for visualization
dim_name = 'umap'

#Getting unique labels
plt.figure(figsize=(8,6))
u_labels = np.unique(label)
num_labels = len(u_labels)
color_palette = sns.color_palette('husl', num_labels)
for i, each in enumerate(center_list):
    plt.scatter(data_2d[each][0],data_2d[each][1], s=30, c=[color_palette[i]])
    plt.annotate(center_list_annotation[i], (data_2d[each][0],data_2d[each][1]), ha="center", va="center", xytext=(0,10), textcoords='offset points')
for i in u_labels:
    plt.scatter(data_2d[label == i , 0] , data_2d[label == i , 1] , label=f"Cluster {i}", s=0.5, c=[color_palette[i]],alpha=0.4)
legend = plt.legend(loc='lower left', ncol=1, frameon=False)
for legend_handle in legend.legend_handles:
    legend_handle.set_sizes([30])  # Increase the size of legend dots    
    legend_handle.set_alpha(1)
plt.setp(legend.texts, fontsize='9')
plt.axis('off')
plt.savefig(f'./results/recon_img/{dataset_name}/MARL_{curr_data}_{dim_name}_{n_clusters}_kmeans.png')


# In[ ]:


from umap import UMAP
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


umap = UMAP(n_neighbors=50, n_components=3, random_state=42)
data_3d_umap = umap.fit_transform(torch.flatten(latents, start_dim=1).cpu())

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data_3d_umap[:, 0], data_3d_umap[:, 1], data_3d_umap[:, 2], s=1, alpha=0.6)

ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_zlabel("UMAP-3")

plt.tight_layout()
# plt# .show()  # Disabled for pipeline  # Disabled for pipeline


# In[ ]:


import plotly.express as px

fig = px.scatter_3d(
    x=data_3d_umap[:,0],
    y=data_3d_umap[:,1],
    z=data_3d_umap[:,2],
    opacity=0.6,
    size_max=2
)
fig# .show()  # Disabled for pipeline


# In[ ]:


import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

data_3d = data_3d_umap
u_labels = np.unique(label)
num_labels = len(u_labels)
color_palette = sns.color_palette('husl', num_labels)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Plot cluster points
for i in u_labels:
    ax.scatter(
        data_3d[label == i, 0],
        data_3d[label == i, 1],
        data_3d[label == i, 2],
        s=2,
        c=[color_palette[i]],
        alpha=0.4,
        label=f"Cluster {i}"
    )

# Plot cluster centers with annotations
for i, each in enumerate(center_list):
    ax.scatter(
        data_3d[each, 0], data_3d[each, 1], data_3d[each, 2],
        s=30, c=[color_palette[i]], depthshade=True
    )
    ax.text(
        data_3d[each, 0], data_3d[each, 1], data_3d[each, 2],
        center_list_annotation[i],
        fontsize=9, ha='center', va='center'
    )

# Legend and axis off
legend = ax.legend(loc='lower left', ncol=1, frameon=False)
ax.set_axis_off()

# plt# .show()  # Disabled for pipeline  # Disabled for pipeline


# In[ ]:


import plotly.express as px
import pandas as pd

# Create a dataframe for easier plotting
df = pd.DataFrame({
    'UMAP-1': data_3d[:,0],
    'UMAP-2': data_3d[:,1],
    'UMAP-3': data_3d[:,2],
    'Cluster': label
})

# Plot points colored by cluster
fig = px.scatter_3d(
    df,
    x='UMAP-1', y='UMAP-2', z='UMAP-3',
    color='Cluster',
    opacity=0.6,
    size_max=3
)

# Add cluster center annotations
for i, idx in enumerate(center_list):
    fig.add_scatter3d(
        x=[data_3d[idx,0]],
        y=[data_3d[idx,1]],
        z=[data_3d[idx,2]],
        mode='markers+text',
        marker=dict(size=5, color='black'),
        text=[center_list_annotation[i]],
        textposition='top center',
        showlegend=False
    )

fig# .show()  # Disabled for pipeline


# In[ ]:


data_2d = data_2d_tsne # this is the data for visualization
dim_name = 'tsne'

#Getting unique labels
plt.figure(figsize=(8,6))
u_labels = np.unique(label)
num_labels = len(u_labels)
color_palette = sns.color_palette('husl', num_labels)
for i, each in enumerate(center_list):
    plt.scatter(data_2d[each][0],data_2d[each][1], s=30, c=[color_palette[i]])
    plt.annotate(center_list_annotation[i], (data_2d[each][0],data_2d[each][1]), ha="center", va="center", xytext=(0,10), textcoords='offset points')
for i in u_labels:
    plt.scatter(data_2d[label == i , 0] , data_2d[label == i , 1] , label=f"Cluster {i}", s=0.5, c=[color_palette[i]],alpha=0.4)
legend = plt.legend(loc='lower left', ncol=1, frameon=False)
for legend_handle in legend.legend_handles:
    legend_handle.set_sizes([30])  # Increase the size of legend dots 
    legend_handle.set_alpha(1)
plt.setp(legend.texts, fontsize='9')
plt.axis('off')
plt.savefig(f'./results/recon_img/{dataset_name}/MARL_{curr_data}_{dim_name}_{n_clusters}_kmeans.png')


# ## Visualize the cluster center

# In[ ]:


def scale_crop(img): #B,C,H,W
    rescale = transforms.Compose([transforms.Resize(112),
                                  transforms.CenterCrop(56)])
    return rescale(img)


# In[ ]:


indexes = center_list
cluster_count = n_clusters
print(f"Number of clusters: {cluster_count}, Number of indexes: {len(indexes)}")


# In[ ]:


# Create a figure and axes with `cluster_count` subplots
fig, axes = plt.subplots(1, cluster_count, figsize=(8, 3))
image_size = 112
crop_size = 56

# Iterate over the `cluster_count` subplots
for i, index in enumerate(indexes):
    image_path = floor.all_data_dirs[index]
    # Read the image using torch's load
    image = torch.load(image_path)
    scaled = scale_crop(image)
    # Display the image in the corresponding subplot
    image_array = scaled.permute(1, 2, 0).numpy()  # Convert to (56, 56, 3) shape
    axes[i].imshow(image_array, cmap='gray')
    axes[i].set_title(os.path.basename(image_path).split(".")[0])
    axes[i].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Save the plot
plt.savefig(f"./results/recon_img/{dataset_name}/MARL_{curr_data}_{cluster_count}cluster_sample_multiscale.png")


# In[ ]:


def get_zoomed_img(image_path, half_pixel):
    image = mpimg.imread(image_path)
    # Get the center coordinates
    height, width, _ = image.shape
    center_x = width / 2
    center_y = height / 2
    
    # Calculate the coordinates for the center 150x150 pixels
    x1 = int(center_x - half_pixel)
    x2 = int(center_x + half_pixel)
    y1 = int(center_y - half_pixel)
    y2 = int(center_y + half_pixel)
    
    # Extract the center 150x150 pixels from each image
    center_img = image[y1:y2, x1:x2]
    return center_img


# In[ ]:


import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cluster_count = n_clusters
# Create a figure and axes with `cluster_count` subplots
fig, axes = plt.subplots(1, cluster_count, figsize=(8, 3))
image_size = 112
crop_size = 56
new_dir = 'warehouse_parking'
num_indexes = []

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for i, index in enumerate(indexes):
    image_path = floor.all_data_dirs[index]
    # Split the path into the name and the extension
    file_name, file_extension = os.path.splitext(image_path)
    # Replace the '.pt' extension with '.png'
    png_file_path = file_name + '.png'
    # Read the image data
    img = get_zoomed_img(png_file_path, 75)
    # Display the image in the corresponding subplot
    axes[i].imshow(img, cmap='gray')
    # Extract numerical part from the file name
    num_index = int(os.path.basename(file_name))
    num_indexes.append(num_index)
    axes[i].set_title(f"{num_index}")
    axes[i].axis('off')

    # Add the prefix 'VQVAE_' to the original filename
    basename = os.path.basename(file_name)
    # Add the prefix 'VQVAE_' to the basename
    new_basename = f'MARL_{curr_data}_{basename}.png'
    # Copy the PNG file to the new directory with the modified filename
    shutil.copy(png_file_path, os.path.join(new_dir, new_basename))

# Adjust the spacing between subplots
plt.tight_layout()

# Save the plot
plt.savefig(f"./results/recon_img/{dataset_name}/MARL_{curr_data}_{cluster_count}cluster_sample.png")
num_indexes


# In[ ]:


import numpy as np
from collections import Counter
import pandas as pd

# `label` is your kmeans labels (shape [N])
# get true second_use labels from dataset
true_labels = np.array([floor[i]['second_use_label'] for i in range(len(floor))]).astype(int)

# contingency
clusters = np.unique(label)
rows = []
for c in clusters:
    idx = np.where(label == c)[0]
    counts = Counter(true_labels[idx])
    total = len(idx)
    top_label, top_count = counts.most_common(1)[0]
    purity = top_count / total
    rows.append({'cluster': int(c), 'size': total, 'top_label': int(top_label), 'top_count': int(top_count), 'purity': purity, 'counts': dict(counts)})

pd.DataFrame(rows).sort_values('cluster')


# In[ ]:


import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cluster_count = n_clusters
image_size = 112
crop_size = 56


num_indexes = []

# Create the output directory if it doesn't exist
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Loop over each image and save it separately
for i, index in enumerate(indexes):
    image_path = floor.all_data_dirs[index]
    file_name, _ = os.path.splitext(image_path)
    png_file_path = file_name + '.png'

    # Load and process the image
    img = get_zoomed_img(png_file_path, 75)

    # Extract a short name or ID
    num_index = os.path.basename(file_name)
    num_indexes.append(num_index)

    # Compose new filename with prefix
    new_basename = f'MARL_{curr_data}_{index}.png'
    output_path = os.path.join(new_dir, new_basename)

    # Save the image (cmap='gray' is optional for grayscale)
    plt.imsave(output_path, img, cmap='gray')

print(f"Saved {len(indexes)} individual images to '{new_dir}/'")

num_indexes



# In[ ]:


indexes = num_indexes
indexes


# #Distribution of clusters

# In[ ]:


unique_labels, counts = np.unique(label, return_counts=True)
cluster_labels = [center_list_annotation[i] for i in unique_labels]
bar_colors = [color_palette[i] for i in unique_labels]

plt.figure(figsize=(9, 6))
bars = plt.bar(cluster_labels, counts, color=bar_colors, edgecolor="black", alpha=0.85)
plt.title(f"Distribution of Buildings per Cluster ({dataset_name_m})", fontsize=14, weight="bold")
plt.xlabel("Clusters", fontsize=12)
plt.ylabel("Number of Buildings", fontsize=12)
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
             str(count), ha='center', va='bottom', fontsize=10, weight="bold")
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"./results/recon_img/{dataset_name}/MARL_{curr_data}_{n_clusters}_warehouse_parking_cluster_distribution.png", dpi=300)
# plt# .show()  # Disabled for pipeline  # Disabled for pipeline


# In[ ]:


metadata_df = pd.read_csv(f"./data/data_config_1/tertiary/{dataset_name_m}/meta_trsa_{dataset_name_m}.csv")


# In[ ]:


# Get cluster center indices (from your clustering code)
center_indices = [int(idx) for idx in num_indexes]  # num_indexes from your code

# Filter metadata for cluster centers
selected = metadata_df[metadata_df['OBJECTID'].isin(center_indices)].copy()
selected['orientation'] = selected['orientation'] * 355.0
selected['street_width'] = selected['street_width'] * 149.58

# Select columns and print in LaTeX format
cols = ['OBJECTID', 'YearBuilt1', 'HEIGHT', 'UseDescription', 'orientation', 'street_width', 'second_type']
output_file = "latex_table.txt"
with open(output_file, "a") as f:  # "a" = append mode
    f.write(f"\n--- {dataset_name_m} ---\n") 
    for _, row in selected[cols].iterrows():
        line = (
            f"{int(row['OBJECTID'])} & "
            f"{row['YearBuilt1']} & "
            f"{row['HEIGHT']} & "
            f"{int(row['UseDescription'])} & "
            f"{row['orientation']:.2f} & "
            f"{row['street_width']:.2f} & "
            f"{int(row['second_type'])} \\\\\n"
        )
        print(line.strip())  # still print to terminal
        f.write(line)


# ## calculate aggregated area

# In[ ]:


merged_df = pd.merge(df_center_index, metadata_df, left_on='index', right_on = 'OBJECTID', how='inner')
print(df_center_index.shape, merged_df.shape)
merged_df
merged_df.to_csv(f'./results/recon_img/{dataset_name}/MARL_{curr_data}_{n_clusters}clusters_full_metadata.csv', index=False)


# In[ ]:


# --- Multimodal Clustering: Image + Metadata ---
alpha = 0.7  # weight for image embeddings
beta = 0.3   # weight for metadata

# Select metadata columns to combine
meta_features = merged_df[['HEIGHT', 'street_width', 'orientation', 'YearBuilt1']].values
meta_scaled = StandardScaler().fit_transform(meta_features)

# Flatten latent features
img_features = torch.flatten(latents, start_dim=1).cpu().numpy()

# Combine features with weighting
combined_features = np.hstack([alpha * img_features, beta * meta_scaled])

# --- Determine optimal k via silhouette ---
best_score = -1
optimal_k = 2
for k in range(2, 8):
    kmeans_tmp = KMeans(n_clusters=k, random_state=42)
    labels_tmp = kmeans_tmp.fit_predict(combined_features)
    score = silhouette_score(combined_features, labels_tmp)
    print(f"k={k}, silhouette score={score:.4f}")
    if score > best_score:
        best_score = score
        optimal_k = k
print("Optimal k (multimodal):", optimal_k)

# --- Final KMeans ---
kmeans_multi = KMeans(n_clusters=optimal_k, random_state=42)
labels_multi = kmeans_multi.fit_predict(combined_features)
cluster_centers_multi = kmeans_multi.cluster_centers_

# --- Closest points to cluster centers ---
closest_multi, _ = pairwise_distances_argmin_min(cluster_centers_multi, combined_features)

# --- Save cluster centers ---
df_centers_multi = pd.DataFrame({
    'Cluster': np.arange(optimal_k),
    'Closest_Point_Index': closest_multi
})
df_centers_multi.to_csv(f'./results/recon_img/{dataset_name}/MARL_{curr_data}_multimodal_centers.csv', index=False)

# --- UMAP visualization ---
umap_multi = UMAP(n_neighbors=50, n_components=2, random_state=42)
data_2d_umap_multi = umap_multi.fit_transform(combined_features)
plt.figure(figsize=(8,6))
u_labels = np.unique(labels_multi)
color_palette = sns.color_palette('husl', len(u_labels))
for i, center_idx in enumerate(closest_multi):
    plt.scatter(data_2d_umap_multi[center_idx,0], data_2d_umap_multi[center_idx,1], s=30, c=[color_palette[i]])
for i in u_labels:
    plt.scatter(data_2d_umap_multi[labels_multi==i,0], data_2d_umap_multi[labels_multi==i,1], s=0.5,
                c=[color_palette[i]], alpha=0.4, label=f"Cluster {i}")
plt.axis('off')
plt.title("UMAP Multimodal Clustering")
plt.savefig(f'./results/recon_img/{dataset_name}/MARL_{curr_data}_multimodal_umap.png')
# plt# .show()  # Disabled for pipeline  # Disabled for pipeline
# Get the file paths (or IDs) of the closest points to the multimodal cluster centers
center_ids_multi = [floor.all_data_dirs[idx] for idx in closest_multi]

# Add filenames to the dataframe
df_centers_multi['Center_File'] = center_ids_multi
df_centers_multi.to_csv(f'./results/recon_img/{dataset_name}/MARL_{curr_data}_multimodal_centers_with_ids.csv', index=False)

# Print results
print("Multimodal cluster centers (indices):", closest_multi)
print("Corresponding file paths / IDs:")
for idx, file in zip(closest_multi, center_ids_multi):
    print(f"Index {idx} -> {file}")



# In[ ]:


grouped_df_lab = merged_df.groupby('YearBuilt1').size().reset_index(name='count')
grouped_df_lab.to_csv(f"{new_dir}/MARL_{curr_data}_{cluster_count}cluster_count.csv", index=False)
grouped_df_lab


# In[ ]:


grouped_df = merged_df.groupby('UseDescription').size().reset_index(name='count')
grouped_df['index_png'] = df_center['index_png']
grouped_df.to_csv(f"{new_dir}/MARL_{curr_data}_{cluster_count}cluster_area_aggregation.csv")
grouped_df


# In[ ]:


#json_index = curr_data.split('_')[0].split('data')[1]
#print(json_index)
json_df = gpd.read_file(f'./data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.geojson', driver='GeoJSON')
json_df = json_df[['id','reference', 'beginning', 'numberOf_1', 'currentUse', 'value', 'geometry']]
json_df


# In[ ]:


csv_df = pd.read_csv('./data_pipeline/trusted_zone/preprocessed_data/08279_br_results_exploded.csv')
csv_df = csv_df[['idx','building_reference', 'br__mean_building_space_effective_year', 'br_floors_per_type', 'use_type', 'br__above_ground_built_area_by_floor']]
csv_df.head()


# In[ ]:


df_center['Closest Point Index'] = num_indexes
df_center


# In[ ]:


df_center['Closest Point Index'] = df_center['Closest Point Index'].astype(int)
csv_df['idx'] = csv_df['idx'].astype(int)

center_metadata = pd.merge(df_center, csv_df, left_on='Closest Point Index', right_on='idx', how='inner')


# In[ ]:


center_metadata.to_csv(f"{new_dir}/MARL_{curr_data}_{cluster_count}cluster_metadata.csv")


# ## recon the cluster center and export metadata

# In[ ]:


kmeans.cluster_centers_.shape


# (82, 3, 112, 112) -> 

# In[ ]:


truncated_centers = kmeans.cluster_centers_[:, :25088]
latent_centers = torch.from_numpy(truncated_centers.reshape(n_clusters, 32, 28, 28).astype('float32')).to(device)

# def show(img, title):
#     npimg = img.numpy()
#     fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     fig.axes.set_title(title)
# plt.figure()  # Create a new plot
# show(torchvision.utils.make_grid(valid_recon.cpu().data) + 0.5, "VQ-VAE Reconstructed")
# plt.savefig(f"{n_clusters}_VQrecon.png", bbox_inches='tight')


# In[ ]:


latent = vqae.pre_quantization_conv(latent_centers)
embedding_loss, latent, perplexity, _ = vqae.vector_quantization(latent)
x_hat = vqae.decoder(latent)

def show(img, title):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.set_title(title)
plt.figure()  # Create a new plot
show(torchvision.utils.make_grid(x_hat.cpu().data) + 0.5, "VQ-VAE Reconstructed")
plt.savefig(f"./results/recon_img/MARL_{curr_data}_{cluster_count}cluster_center_recon.png", bbox_inches='tight')



# In[ ]:


latent.shape


# In[ ]:


valid_recon = vqae.pre_quantization_conv(latent_centers)
valid_recon.shape


# In[ ]:


valid_recon[0].shape

