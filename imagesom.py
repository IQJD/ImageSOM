# -*- coding: utf-8 -*-
"""IMAGESOM_GITHUB"""

#pip install MiniSom

#pip install ramanspy

from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.gridspec import GridSpec
from matplotlib.patches import RegularPolygon
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import ramanspy
from minisom import MiniSom
from ramanspy import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')

img = np.asarray(Image.open('/route/image.jpg'))
print(img.shape)
#imgplot = plt.imshow(img)

plt.imshow(img, interpolation='nearest', extent=[1,30,30,1])
plt.xticks(range(10, 31, 10))
plt.yticks(range(10, 31, 10))
plt.gca().invert_yaxis()
plt.show()

x = '/route/file.txt'
df= pd.read_csv(x, delimiter='\t',header=None)
print(df.shape)
df

spectral_data = df.iloc[1:,].to_numpy(dtype=float)
spectral_axis = df.iloc[0,].tolist()
raman_spectra = ramanspy.Spectrum(spectral_data, spectral_axis)

#preprocesing
pipe = ramanspy.preprocessing.Pipeline([
    ramanspy.preprocessing.despike.WhitakerHayes(kernel_size=4),
    ramanspy.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
    ramanspy.preprocessing.baseline.IARPLS(lam=1e2, tol=1e-3, max_iter=50),
    ramanspy.preprocessing.normalise.MinMax()
    ])
preprocessed_spectra = pipe.apply(raman_spectra)
preprocessed_spectra.plot(title='Preprocessed spectra',ylabel='Normalized Intensity')

image_data = preprocessed_spectra.spectral_data

#Training SOM
max_iter = 2000
som = MiniSom(12,12, image_data.shape[1], sigma=1.7, learning_rate=0.3,activation_distance='cosine',topology= 'hexagonal',neighborhood_function='gaussian',random_seed=123)
som.random_weights_init(image_data)
som.train_batch(image_data, max_iter)

quantization_error = som.quantization_error(image_data)
topographic_error_value = som.topographic_error(image_data)

print(f'SOM Quantization Error: {quantization_error}')
print(f'SOM Topographic Error: {topographic_error_value}')

xx, yy      = som.get_euclidean_coordinates()
umatrix     = som.distance_map()
weights     = som.get_weights()
norm = plt.Normalize(vmin=vmin_u, vmax=vmax_u)
sm = cm.ScalarMappable(norm=norm, cmap='viridis')
m, n, dim = weights.shape
protos = weights.reshape(-1, dim)

#Davies–Bouldin (k = 2 - 6)
k_values = range(2, 7)
db_scores = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=0).fit(protos)
    db = davies_bouldin_score(protos, km.labels_)
    db_scores.append(db)

#optimal k
optimal_k = k_values[np.argmin(db_scores)]
print(f"Optimal k by Davies–Bouldin: {optimal_k}")

km          = KMeans(n_clusters=optimal_k, random_state=0).fit(protos)
cluster_lbl = km.labels_.reshape(umatrix.shape)

# CLuster colors
cluster_colors = {0:'red',
                  1: 'blue',
                  2: 'green',
                  3: 'orange'}

#U-MATRIX
fig = plt.figure(figsize=(14, 8), constrained_layout=True)

outer = fig.add_gridspec(2, 2, width_ratios=[2.2, 2], wspace=0.12, hspace=0.22)
left  = outer[:, 0].subgridspec(2, 2,width_ratios=[30, 0.6], wspace=0.000005, hspace=0.05)

ax_a = fig.add_subplot(left[0, 0])
cax  = fig.add_subplot(left[0, 1])
ax_c = fig.add_subplot(left[1, 0])
fig.add_subplot(left[1, 1]).axis("off")
ax_b = fig.add_subplot(outer[:, 1])
hex_size = 0.55

def tight_hex(ax, xx, yy, pad):
    ax.set_aspect('equal')
    ax.set_xlim(xx.min() - pad, xx.max() + pad)
    ax.set_ylim(yy.min() - pad, yy.max() + pad)
    ax.margins(0)
    ax.axis('off')

for x, y, u in zip(xx.flatten(), yy.flatten(), umatrix.flatten()):
    ax_a.add_patch(RegularPolygon((x, y), 6, radius=hex_size,
                                  edgecolor='k', facecolor=sm.to_rgba(u)))

tight_hex(ax_a, xx, yy, hex_size)
fig.colorbar(sm, cax=cax, pad=0)

ax_b.plot(list(k_values), db_scores, marker='o', linestyle='-')
ax_b.set_xticks(list(k_values))
ax_b.set_xlabel('Number of clusters', fontweight='bold')
ax_b.set_ylabel('Davies–Bouldin Index', fontweight='bold')

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x, y = xx[i, j], yy[i, j]
        lbl = cluster_lbl[i, j]
        ax_c.add_patch(RegularPolygon((x, y), 6, radius=hex_size,
                                      edgecolor='k', facecolor=cluster_colors[lbl]))

tight_hex(ax_c, xx, yy, hex_size)
plt.show()

km = KMeans(n_clusters=optimal_k, random_state=0).fit(protos)
proto_labels = km.labels_

bmus = np.array([som.winner(x) for x in image_data])
rows, cols = bmus[:,0], bmus[:,1]

bmu_idx = rows * n + cols
pixel_labels = proto_labels[bmu_idx]

n_pc = np.sum(pixel_labels == 0)
n_psu  = np.sum(pixel_labels == 1)
print(f"Píxeles PSU (cluster 0):       {n_psu}")
print(f"Píxeles Policarbonato (cluster 1): {n_pc}")

n_pix = image_data.shape[0]
alt = anc = int(np.sqrt(n_pix))
label_img = pixel_labels.reshape(alt, anc)

fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1.0,1.0,1.0]},constrained_layout=True)

ax[0].imshow(img, interpolation='nearest', extent=[0, 30, 30, 0])
ax[0].invert_yaxis()
ticks = [0, 10, 20, 30]
ax[0].set_xticks(ticks)
ax[0].set_yticks(ticks)
ax[0].set_ylabel('Pixel', fontweight='bold')
ax[0].set_xlabel('Pixel', fontweight='bold')

ax[1].imshow(label_img, cmap='jet', interpolation='nearest', extent=[0, 30, 30, 0])
ax[1].invert_yaxis()
ax[1].set_xticks(ticks)
ax[1].set_yticks(ticks)
ax[1].set_ylabel('Pixel', fontweight='bold')
ax[1].set_xlabel('Pixel', fontweight='bold')

bars = ax[2].bar(['polysulfone','polycarbonate'], [n_psu, n_pc], color=['red','blue'], edgecolor='black', linewidth=2, )
for bar in bars:
    h = bar.get_height()
    ax[2].text(bar.get_x()+bar.get_width()/2, h+10, f'{h}',ha='center', va='bottom')
ax[2].set_ylabel('Number of pixels', fontweight='bold')
for a in ax:
    a.set_box_aspect(1)
plt.show()

#Cosine similarity

bmus = np.array([som.winner(x) for x in image_data])
rows, cols = bmus[:, 0], bmus[:, 1]
m, n = cluster_lbl.shape
cluster_flat = cluster_lbl.flatten()
bmu_idx      = rows * n + cols
pixel_clusters = cluster_flat[bmu_idx]

spectra_cluster_1 = image_data[pixel_clusters == 1]

print(f"Numero de espectros en el cluster (blue): {spectra_cluster_1.shape[0]} spectra")

mean1 = np.mean(spectra_cluster_1, axis=0)
std1  = np.std(spectra_cluster_1, axis=0)


sim_1 = cosine_similarity(mean1.reshape(1, -1),preprocessed_spectra.spectral_data.reshape(1, -1))[0, 0]
print(f"Cosine similarity (mean polysulfone vs. Polysulfone reference):   {sim_1:.3f}")

W = som.get_weights()
m, n, dim = W.shape
W_flat = W.reshape(m * n, dim)

cluster_flat = cluster_lbl.flatten()
prototypes_blue = W_flat[cluster_flat == 1]

mean_blue = np.mean(prototypes_blue, axis=0)
std_blue  = np.std(prototypes_blue,  axis=0)

x_axis        = spectral_axis
patron_spec   = preprocessed_spectra.spectral_data

#Similarity cosine between standard and samples
sim_blue = cosine_similarity(mean_blue.reshape(1, -1),patron_spec.reshape(1, -1))[0, 0]

#Top 10 Most Variable Wavelengths in SOM Prototype Weights
weights = som.get_weights()
W = weights.reshape(-1, weights.shape[2])
var_w = W.std(axis=0)
topN = np.argsort(var_w)[-10:][::-1]
selected_indices = topN
spectral_axis_np = np.asarray(spectral_axis)
selected_lambdas = spectral_axis_np[selected_indices]
print(selected_lambdas)

h = 188 #index of the raman shift selected
xx, yy = som.get_euclidean_coordinates()
fig, ax = plt.subplots(figsize=(6, 6))
plane = weights[:, :, i]
vmin, vmax = plane.min(), plane.max()
for x, y, val in zip(xx.flatten(), yy.flatten(), plane.flatten()):
    norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    color = plt.cm.viridis(norm_val)
    hex_patch = RegularPolygon((x, y),
        numVertices=6,
        radius=hex_size,
        facecolor=color,
        edgecolor='black',
        lw=0.8)
    ax.add_patch(hex_patch)
ax.set_xlim(xx.min() - hex_size, xx.max() + hex_size)
ax.set_ylim(yy.min() - hex_size, yy.max() + hex_size)
ax.set_aspect('equal')
ax.axis('off')

# Colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()