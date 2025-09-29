#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import numpy as np
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
from torchvision import transforms
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib.pyplot as plt
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
dataset_name = 'commercial'
dataset_name_m ='Commercial'
curr_data = 'BR_Commercial'


# In[2]:


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

# In[3]:


#Load Dataset

floor = FloorPlanDataset(multi_scale=True, root=f'../data/data_br/{dataset_name}/',\
                         data_config=f'../data/data_config_1/tertiary/{dataset_name_m}/', preprocess=True)
data_loader = torch.utils.data.DataLoader(floor, batch_size=batch_size, shuffle=False)
vqae = VQAE(n_hiddens, n_residual_hiddens, n_residual_layers,
            n_embeddings, embedding_dim, 
            beta, img_channel).to(device)

marl = MARL(vqae, True, floor.age_label_num, floor.category_num, floor.num_second_use).to(device)
checkpoint = torch.load(f"../checkpoint/best_{dataset_name}.pt", map_location=device)
marl.load_state_dict(checkpoint["model_state_dict"])
# print(f"data shape: {floor[0]['image_tensor'].shape}, dataset size: {len(floor)}")
for param in vqae.parameters():
    param.to('cpu')
for param in marl.parameters():
    param.to('cpu')

# Compute VQAE latents in-order with dataset (no shuffle)
latent_file_path = f'../data/data_root/marl_latent_{curr_data}.pt'
os.makedirs(os.path.dirname(latent_file_path), exist_ok=True)

# Try to load existing latents first
if os.path.exists(latent_file_path):
    print(f"Loading existing latents from {latent_file_path}")
    latents = torch.load(latent_file_path, map_location=device)
    sample_paths = floor.all_data_dirs.copy()
else:
    print("Computing latents...")
    marl.eval()
    latents = None
    sample_paths = floor.all_data_dirs.copy()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting latents'):
            data = batch['image_tensor'].to(device)
            out = marl(data)
            quantized = out['latent']
            latents = quantized if latents is None else torch.cat([latents, quantized], dim=0)
    
    print(f"Saving latents to {latent_file_path}")
    torch.save(latents, latent_file_path)
    print(f"Latents saved successfully")

# Ensure results directory exists
results_dir = f"../results/recon_img/{dataset_name}"
os.makedirs(results_dir, exist_ok=True)


# # Embedding Space Visualization
# ## Visualize the latent with UMAP and T-SNE

# In[4]:


# latents: [N, C, H, W], sample_paths aligned to dataset order
floor = FloorPlanDataset(multi_scale=True, root=f'../data/data_br/{dataset_name}/',\
                         data_config=f'../data/data_config_1/tertiary/{dataset_name_m}/', preprocess=True)
data_loader = torch.utils.data.DataLoader(floor, batch_size=batch_size, shuffle=False)


# In[5]:


# Flatten latents for dimensionality reduction

neighbors_list = [30,50]

for n_neighbors in neighbors_list:
    umap = UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
    data_2d_umap = umap.fit_transform(torch.flatten(latents, start_dim=1).cpu())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_2d_umap[:, 0], data_2d_umap[:, 1], s=0.5)
    ax.set_title(f'UMAP with n_neighbors={n_neighbors}')
    plt.show()





# In[6]:


umap = UMAP(n_neighbors =50, n_components=2, random_state=42)
data_2d_umap = umap.fit_transform(torch.flatten(latents, start_dim=1).cpu())
# Create a figure and a 3D axes object
fig = plt.figure()
ax = fig.add_subplot(111)

# Create the initial scatter plot
ax.scatter(data_2d_umap[:, 0], data_2d_umap[:, 1], s=0.5)

# Show the plot
plt.show()


# In[7]:


# Reuse computed latents for TSNE

perplexities = [5,10,15,20, 30]

for perplexity in perplexities:
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    data_2d_tsne = tsne.fit_transform(torch.flatten(latents, start_dim=1).cpu())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], s=0.5)
    ax.set_title(f't-SNE with perplexity={perplexity}')
    plt.show()


# In[8]:


tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
data_2d_tsne = tsne.fit_transform(torch.flatten(latents, start_dim=1).cpu())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], s=0.5)
ax.set_title(f't-SNE with perplexity=30')
save_path = f"../notebooks/{dataset_name}/tsne_perp30.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()


# ### deciding the number of clusters

# In[9]:


def calculate_wcss(latents):
    wcss = []
    for n in tqdm(range(1, 7)):  # change the range according to your needs
        kmeans = KMeans(n_clusters=n)
        data = torch.flatten(latents, start_dim=1).cpu()
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # inertia_ is the WCSS for the model
    return wcss
# calculate WCSS for different numbers of clusters
wcss = calculate_wcss(latents)


# In[10]:


# Convert list to DataFrame
wcss_df = pd.DataFrame(wcss)
wcss_df.to_csv(f'wcss_{curr_data}.csv',index=False)
wcss_df


# In[11]:


def plot_elbow(wcss, filename=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(wcss) + 1), wcss, 'bo-')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid()
    filename = f'../notebooks/{dataset_name}/elbow_plot_kmeans.png' 
    plt.savefig(filename)


# In[12]:


# plot the elbow graph
plot_elbow(wcss)


# the number of clusters are set to 5, 10, 15

# In[13]:


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


# In[14]:


# Calculate slopes
slopes = calculate_slope(wcss)
print(slopes)
# Find the optimal point based on the slope
optimal_point = find_optimal_point(slopes)

print("Optimal point:", optimal_point)


# In[15]:


# Plot the slopes
plt.plot(range(1, len(wcss)), slopes, 'bo-')
plt.title('Slope between Points')
plt.xlabel('Point Index')
plt.ylabel('Slope')
plt.grid()
save_path = f"../notebooks/{dataset_name}/slope_plot.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Slope plot saved to {save_path}")
plt.show()


# In[16]:


from kneed import KneeLocator

K = range(1, len(wcss) + 1)
kneedle = KneeLocator(K, wcss, curve='convex', direction='decreasing')
print("Optimal number of clusters:", kneedle.knee)
print(wcss)



# In[17]:


if kneedle.knee is None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import torch

    # Flatten your latent vectors for clustering
    data = torch.flatten(latents, start_dim=1).cpu().numpy()

    best_score = -1
    optimal_k = 2  # silhouette requires at least 2 clusters

    for k in range(2, 7):  # try k=2 to 6
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        print(f"k={k}, silhouette score={score:.4f}")
        if score > best_score:
            best_score = score
            optimal_k = k

    print("Optimal number of clusters (silhouette):", optimal_k)


# ### Clustering

# In[18]:


n_clusters= kneedle.knee
if kneedle.knee is None:
    n_clusters = optimal_k
data_latent = latents # this is the data for clustering
print(n_clusters)


# In[19]:


#Initialize the class object
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data = torch.flatten(data_latent, start_dim=1).cpu()
# #predict the labels of clusters.
label = kmeans.fit_predict(data)
# get cluster centers
cluster_centers = kmeans.cluster_centers_
closest, _ = pairwise_distances_argmin_min(cluster_centers, data)
print(cluster_centers.shape, data.shape)



center_list = closest
print(center_list)
df = pd.DataFrame({'label': label, 'data': sample_paths})


# In[20]:


def calculate_index_png(index):
    image_path = sample_paths[index]
    file_name, file_extension = os.path.splitext(image_path)
    num_index = os.path.basename(file_name)
    return num_index


# In[21]:


# Save closest points and cluster centers to CSV
df_center = pd.DataFrame({'Cluster Center': cluster_centers.tolist(), 'Closest Point Index': closest})
df_center['index_png'] = df_center['Closest Point Index'].apply(lambda x: calculate_index_png(x))
df_center.to_csv(f'../results/recon_img/{dataset_name}/MARL_{curr_data}_{n_clusters}cluster_centers.csv', index=False)

df_center_index = pd.DataFrame({'label': label, 'data': sample_paths})
df_center_index['index'] = df_center_index['data'].str.extract(r'(\d+)\.pt')
df_center_index = df_center_index.drop(['data'], axis=1)
df_center_index = df_center_index.dropna(subset=['index'])
df_center_index['index'] = df_center_index['index'].astype(int)
df_center_index.to_csv(f'../results/recon_img/{dataset_name}/MARL_{curr_data}_{n_clusters}clusters_clusterindex.csv', index=False)
print(df_center_index.shape)
df_center_index.head(10)


# In[22]:


center_list = closest
center_list_annotation = df_center['index_png'].tolist()


# In[23]:


data_2d = data_2d_umap # this is the data for visualization
dim_name = 'umap'

#Getting unique labels
plt.figure(figsize=(8,6))
u_labels = np.unique(label)
label_to_idx = {lab: idx for idx, lab in enumerate(sorted(u_labels))}
color_palette = sns.color_palette('husl', len(u_labels))
for i, each in enumerate(center_list):
    plt.scatter(data_2d[each][0],data_2d[each][1], s=30, c=[color_palette[i]])
    plt.annotate(center_list_annotation[i], (data_2d[each][0],data_2d[each][1]), ha="center", va="center", xytext=(0,10), textcoords='offset points')
for lab in u_labels:
    mask = (label == lab)
    plt.scatter(data_2d[mask, 0], data_2d[mask, 1], label=f"Cluster {lab}", s=0.5, c=[color_palette[label_to_idx[lab]]], alpha=0.4)
legend = plt.legend(loc='lower left', ncol=1, frameon=False)
for legend_handle in legend.legend_handles:
    legend_handle.set_sizes([30])  # Increase the size of legend dots    
    legend_handle.set_alpha(1)
plt.setp(legend.texts, fontsize='9')
plt.axis('off')
plt.savefig(f'../results/recon_img/{dataset_name}/MARL_{curr_data}_{dim_name}_{n_clusters}_kmeans.png')


# In[24]:


data_2d = data_2d_tsne # this is the data for visualization
dim_name = 'tsne'

#Getting unique labels
plt.figure(figsize=(8,6))
u_labels = np.unique(label)
label_to_idx = {lab: idx for idx, lab in enumerate(sorted(u_labels))}
color_palette = sns.color_palette('husl', len(u_labels))
for i, each in enumerate(center_list):
    plt.scatter(data_2d[each][0],data_2d[each][1], s=30, c=[color_palette[i]])
    plt.annotate(center_list_annotation[i], (data_2d[each][0],data_2d[each][1]), ha="center", va="center", xytext=(0,10), textcoords='offset points')
for lab in u_labels:
    mask = (label == lab)
    plt.scatter(data_2d[mask, 0], data_2d[mask, 1], label=f"Cluster {lab}", s=0.5, c=[color_palette[label_to_idx[lab]]], alpha=0.4)
legend = plt.legend(loc='lower left', ncol=1, frameon=False)
for legend_handle in legend.legend_handles:
    legend_handle.set_sizes([30])  # Increase the size of legend dots 
    legend_handle.set_alpha(1)
plt.setp(legend.texts, fontsize='9')
plt.axis('off')
plt.savefig(f'../results/recon_img/{dataset_name}/MARL_{curr_data}_{dim_name}_{n_clusters}_kmeans.png')


# ## Visualize the cluster center

# In[25]:


def scale_crop(img): #B,C,H,W
    rescale = transforms.Compose([transforms.Resize(112),
                                  transforms.CenterCrop(56)])
    return rescale(img)


# In[26]:


indexes = center_list
cluster_count = n_clusters
print(f"Number of clusters: {cluster_count}, Number of indexes: {len(indexes)}")


# In[27]:


# Create a figure and axes with `cluster_count` subplots
fig, axes = plt.subplots(1, cluster_count, figsize=(8, 3))
image_size = 112
crop_size = 56

# Iterate over the `cluster_count` subplots
for i, index in enumerate(indexes):
    image_path = sample_paths[index]
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
plt.savefig(f"../results/recon_img/{dataset_name}/MARL_{curr_data}_{cluster_count}cluster_sample_multiscale.png")


# In[28]:


def get_zoomed_img(image_path, half_pixel):
    image = mpimg.imread(image_path)
    # Get the center coordinates
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    
    # Calculate the coordinates for the center 150x150 pixels
    x1 = center_x - half_pixel
    x2 = center_x + half_pixel
    y1 = center_y - half_pixel
    y2 = center_y + half_pixel
    
    # Extract the center 150x150 pixels from each image
    center_img = image[y1:y2, x1:x2]
    return center_img


# In[29]:


import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cluster_count = n_clusters
# Create a figure and axes with `cluster_count` subplots
fig, axes = plt.subplots(1, cluster_count, figsize=(8, 3))
image_size = 112
crop_size = 56
new_dir = dataset_name
num_indexes = []

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for i, index in enumerate(indexes):
    image_path = sample_paths[index]
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
plt.savefig(f"../results/recon_img/{dataset_name}/MARL_{curr_data}_{cluster_count}cluster_sample.png")
num_indexes


# In[30]:


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


# In[31]:


import os
import shutil
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



# In[32]:


indexes = num_indexes
indexes


# #Distribution of clusters

# In[33]:


unique_labels, counts = np.unique(label, return_counts=True)
cluster_labels = [center_list_annotation[label_to_idx[i]] for i in unique_labels]
bar_colors = [sns.color_palette('husl', len(np.unique(label)))[label_to_idx[i]] for i in unique_labels]

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
plt.savefig(f"../results/recon_img/{dataset_name}/MARL_{curr_data}_{n_clusters}_commercial_cluster_distribution.png", dpi=300)
plt.show()


# ## calculate aggregated area

# In[34]:


metadata_df = pd.read_csv(f"../data/data_config_1/tertiary/{dataset_name_m}/meta_trsa_{dataset_name_m}.csv")
merged_df = pd.merge(df_center_index, metadata_df, left_on='index', right_on = 'OBJECTID', how='inner')
print(df_center_index.shape, merged_df.shape)
merged_df


# In[35]:


# --- Multimodal Clustering: Image + Metadata ---
alpha = 0.7  # weight for image embeddings
beta = 0.3   # weight for metadata

# Select metadata columns to combine
meta_features = merged_df[['HEIGHT', 'street_width', 'orientation', 'YearBuilt1']].values
meta_scaled = StandardScaler().fit_transform(meta_features)

# Align multimodal features to rows present in merged_df
img_features_all = torch.flatten(latents, start_dim=1).cpu().numpy()
# Map sample_paths -> index (OBJECTID) extracted above
samples_df = df_center_index[['index']].copy()
samples_df['row_id'] = np.arange(len(samples_df))
merged_for_features = pd.merge(samples_df, merged_df[['OBJECTID']], left_on='index', right_on='OBJECTID', how='inner')
row_ids = merged_for_features['row_id'].to_numpy()
img_features = img_features_all[row_ids]
meta_features = merged_df[['HEIGHT', 'street_width', 'orientation', 'YearBuilt1']].to_numpy()
meta_scaled = StandardScaler().fit_transform(meta_features)
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
df_centers_multi.to_csv(f'../results/recon_img/{dataset_name}/MARL_{curr_data}_multimodal_centers.csv', index=False)

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
plt.savefig(f'../results/recon_img/{dataset_name}/MARL_{curr_data}_multimodal_umap.png')
plt.show()
# Get the file paths (or IDs) of the closest points to the multimodal cluster centers
center_ids_multi = [sample_paths[idx] for idx in closest_multi]

# Add filenames to the dataframe
df_centers_multi['Center_File'] = center_ids_multi
df_centers_multi.to_csv(f'../results/recon_img/{dataset_name}/MARL_{curr_data}_multimodal_centers_with_ids.csv', index=False)

# Print results
print("Multimodal cluster centers (indices):", closest_multi)
print("Corresponding file paths / IDs:")
for idx, file in zip(closest_multi, center_ids_multi):
    print(f"Index {idx} -> {file}")



# In[36]:


grouped_df_lab = merged_df.groupby('YearBuilt1').size().reset_index(name='count')
grouped_df_lab.to_csv(f"{new_dir}/MARL_{curr_data}_{cluster_count}cluster_count.csv", index=False)
grouped_df_lab


# In[37]:


grouped_df = merged_df.groupby('UseDescription').size().reset_index(name='count')
grouped_df['index_png'] = df_center['index_png']
grouped_df.to_csv(f"{new_dir}/MARL_{curr_data}_{cluster_count}cluster_area_aggregation.csv")
grouped_df


# In[38]:


#json_index = curr_data.split('_')[0].split('data')[1]
#print(json_index)
json_df = gpd.read_file(f'../data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.geojson', driver='GeoJSON')
json_df = json_df[['id','reference', 'beginning', 'numberOf_1', 'currentUse', 'value', 'geometry']]
json_df


# In[39]:


csv_df = pd.read_csv('../data_pipeline/trusted_zone/preprocessed_data/08279_br_results_exploded.csv')
csv_df = csv_df[['idx','building_reference', 'br__mean_building_space_effective_year', 'br_floors_per_type', 'use_type', 'br__above_ground_built_area_by_floor']]
csv_df.head()


# In[40]:


df_center['Closest Point Index'] = num_indexes
df_center


# In[41]:


df_center['Closest Point Index'] = df_center['Closest Point Index'].astype(int)
csv_df['idx'] = csv_df['idx'].astype(int)

center_metadata = pd.merge(df_center, csv_df, left_on='Closest Point Index', right_on='idx', how='inner')


# In[42]:


center_metadata.to_csv(f"{new_dir}/MARL_{curr_data}_{cluster_count}cluster_metadata.csv")


# ## recon the cluster center and export metadata

# In[43]:


kmeans.cluster_centers_.shape


# (82, 3, 112, 112) -> 

# In[44]:


# Reconstruct cluster centers from latent space centers
latent_dim = latents.shape[1:]  # (C, H, W)
flat_dim = int(np.prod(latent_dim))
truncated_centers = kmeans.cluster_centers_[:, :flat_dim]
latent_centers = torch.from_numpy(truncated_centers.reshape(n_clusters, *latent_dim).astype('float32')).to(device)

# def show(img, title):
#     npimg = img.numpy()
#     fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     fig.axes.set_title(title)
# plt.figure()  # Create a new plot
# show(torchvision.utils.make_grid(valid_recon.cpu().data) + 0.5, "VQ-VAE Reconstructed")
# plt.savefig(f"{n_clusters}_VQrecon.png", bbox_inches='tight')


# In[45]:


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
plt.savefig(f"../results/recon_img/MARL_{curr_data}_{cluster_count}cluster_center_recon.png", bbox_inches='tight')



# In[46]:


latent.shape


# In[47]:


valid_recon = vqae.pre_quantization_conv(latent_centers)
valid_recon.shape


# In[48]:


valid_recon[0].shape


# # Visualize and Save Reconstruction Image

# In[ ]:


floor.all_data_dirs[8834]


# In[ ]:


kmeans.cluster_centers_.shape


# In[ ]:


64*14*14



# In[ ]:


latent_centers = torch.from_numpy(kmeans.cluster_centers_.reshape(n_clusters, 64, 28, 21).astype('float32')).to(device)
vqae.eval()
valid_recon = vqae.decoder(latent_centers)
def show(img, title):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.set_title(title)
plt.figure()  # Create a new plot
show(torchvision.utils.make_grid(valid_recon.cpu().data) + 0.5, "VQ-VAE Reconstructed")
plt.savefig(f"{n_clusters}_VQrecon.png", bbox_inches='tight')


# In[ ]:


def rescale_img(img): #B,C,H,W
    rescale = transforms.Compose([transforms.Resize(112),
                                  transforms.Pad(56, fill=1.0)])
    return rescale(img)
rescaled_recon = rescale_img(valid_recon+0.5)

for i in range(rescaled_recon.shape[0]):
    save_image(rescaled_recon[i], f'../results/recon_img/{n_clusters}cluster-img{i}.jpg')
    
print('save to results/recon_img/*.jpg')


# ### For rhino mapping-back 

# In[ ]:


import glob
import cv2
import os
import numpy as np
import torchvision.transforms.functional as TF
import imageio
import matplotlib.pyplot as plt


# In[ ]:


folder_path = "../results/recon_img/"  
img_files = []
for filename in os.listdir(folder_path):
    if "5cluster" in filename:
        print(filename)
        img_files.append(filename)


# In[ ]:


def calc_height(image_path):
    print(image_path)
    fullpath = f"../results/recon_img/"+str(image_path)
    image = cv2.imread(fullpath)
    height, width, channels = image.shape
    print(height, width, channels)
    
    # Calculate the starting and ending indices for the central region
    start = (height - central_region_size) // 2
    end = start + central_region_size
    
    # Get the average pixel value in the central region for each channel
    central_region = image[start:end, start:end, :]
    average_pixel_value = np.mean(central_region, axis=(0,1))
    
    # Draw a red rectangle around the central region
    img_with_rectangle = np.copy(image)
    cv2.rectangle(img_with_rectangle, (start, start), (end, end), (0, 0, 255), 2)  # Draw red rectangle
    
    # Save the image with the red rectangle
    image_with_rectangle_path = fullpath.replace(".jpg", "_rectangle.jpg")  # Update the path as per your requirement
    imageio.imwrite(image_with_rectangle_path, img_with_rectangle)
    print(image_with_rectangle_path)
    return average_pixel_value

central_region_size = 10
height_info = []
for image_path in img_files:
    average_pixel_value = calc_height(image_path)
    height_info.append(average_pixel_value[0])


# In[ ]:


height_info


# In[ ]:


# Loop through each image
image_paths = glob.glob('recon_img/*.png')
lower_thres = 0.3
higher_thres = 0.7
for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray)
    edges = cv2.Canny(gray, lower_thres, higher_thres)  # Adjust the threshold values as needed

    # Save the edge image
    filename = os.path.splitext(os.path.basename(image_path))[0]  # Extract the filename without extension
    edge_image_path = f'edge_img/{filename}_edge.jpg'
    cv2.imwrite(edge_image_path, edges)


"""
Optional area aggregation utilities below depend on external inputs (objectid_ref_df etc.).
Guarded by ENABLE_AREA_AGG to prevent NameError when those inputs are not present.
"""
ENABLE_AREA_AGG = False

if ENABLE_AREA_AGG:
    # ### get aggregated area

    # In[ ]:


    def get_csv_df(index):
        desired_number = str(index)
        for filename in os.listdir(directory_path):
            if desired_number in filename:
                file_path = os.path.join(directory_path, filename)
                csv_df = pd.read_csv(file_path,index_col=None)
                csv_df['index'] = csv_df['data'].str.extract(r'(\d+)\.pt')
                csv_df['index']=csv_df['index'].astype(int)
        return csv_df


# In[ ]:


    def calculate_aggregated_areas(csv_df):
        desired_features = []
        for feature in jsonfile['features']:
            if feature['properties']['OBJECTID'] in csv_df['index'].values:
                desired_features.append(feature)

        # Extract information for objectID, areas, and heights from desired features
        objectID_list = []
        footprint_list = []
        area_list = []
        height_list = []

        for feature in desired_features:
            objectID = feature['properties']['OBJECTID']
            footprint = feature['properties']['footprint']
            area = feature['properties']['Shape_Area_1']
            height = feature['properties']['HEIGHT']

            objectID_list.append(objectID)
            footprint_list.append(footprint)
            area_list.append(area)
            height_list.append(height)

            # Create a new DataFrame from the extracted information
            json_df = pd.DataFrame({
                'objectID': objectID_list,
                'footprint': footprint_list,
                'area': area_list,
                'height': height_list
            })
        merged_df = json_df.merge(csv_df, left_on='objectID', right_on='index')
        sum_of_area = merged_df['area'].sum()
        return sum_of_area, merged_df


# In[ ]:


    def get_sum_of_area():
        merged_df = csv_df.merge(objectid_ref_df, left_on='index', right_on='OBJECTID')
        print(f"csv_df shape = {csv_df.shape}, objectid_ref_df = {objectid_ref_df.shape}")
        print(f"merged_df shape = {merged_df.shape}")
        sum_of_area = merged_df['building.area.m2'].sum()
        return sum_of_area


# In[ ]:


    sum_of_area_list = []
    json_file_path = r"..\data\data_root\geojson_0.json"
    directory_path = r'../results/area_aggregation'
    objectID_df = pd.DataFrame(columns=[0])
    with open(json_file_path) as file:
        jsonfile = json.load(file)

    for index in tqdm(num_indexes):
        csv_df = get_csv_df(index)
        sum_of_area = get_sum_of_area()
        sum_of_area_list.append(sum_of_area)


# In[ ]:


    sum_of_area_list


# In[ ]:


    objectid_ref_df


# In[ ]:


    objectid_ref_df.sum()


# In[ ]:


    sum_of_area_list


# In[ ]:


    new_df = pd.DataFrame({'index': num_indexes, 'area': sum_of_area_list})
    new_df.to_csv('area_aggregation_kmeans_5.csv')

