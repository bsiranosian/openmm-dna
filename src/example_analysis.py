import numpy as np
import matplotlib.pyplot as plt
from src.useful_3D import xyz_to_distance_many, distance_to_contact_many

# load structure ensemble
many_struc = np.load('data/example_structures.npy')
print('50 structures of 65 bins each, 3D coordinates')
print('many_struc.shape = ' + str(many_struc.shape))
# convert to pairwise distances
many_dist = xyz_to_distance_many(many_struc)
# contact proabability at contact_threshold radii
contact_threshold = 12
many_contact = distance_to_contact_many(many_dist, threshold=contact_threshold)

# average across this small ensemble
many_dist_average = np.average(many_dist, axis=0)
many_contact_average = np.average(many_contact, axis=0)

# create a figure showing both averages
fig, (ax0, ax1) = plt.subplots(ncols=2)
im0 = ax0.matshow(many_dist_average, cmap='magma_r')
ax0.set_title('Mean pairwise distance', y=1.15)
fig.colorbar(im0, ax=ax0, shrink=0.5)

im1 = ax1.matshow(many_contact_average, cmap='magma')
ax1.set_title('Mean pairwise contact at r=' + str(contact_threshold), y=1.15)
fig.colorbar(im1, ax=ax1, shrink=0.5)
fig.tight_layout()
plt.show(block=False)
