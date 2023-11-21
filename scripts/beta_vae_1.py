# %%
#https://youtu.be/YV9D3TWY5Zo
#https://youtu.be/8wrLjnQ7EWQ

"""
VAEs can be used for generative purposes. 
This code demonstrates VAE using MNIST dataset.
Just like regular autoencoder VAE returns an array (image) of same domensions
as input but we can introduce variation by tweaking the latent vector.
"""
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Input, Flatten, Dense, Lambda, Reshape
#from keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import normalize
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage import io

from skimage.transform import resize

import sys 
import os 

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

##################
# VAE parameters #
##################

img_depth = 32
img_width = 32
img_height = 32
num_channels = 2


beta = float(sys.argv[1])
latent_dim = int(sys.argv[2]) # Number of latent dim parameters

print(beta, latent_dim)
print('image dimensions: ', img_depth * img_height * img_width)

from datetime import datetime

# datetime object containing current date and time
now = datetime.strftime(datetime.now(), '%d%m%y_%H%M')

model_name = f'vae/32_px_patch_attempt_1_beta_{beta}_ndim_{latent_dim}_time_{now}'

def import_files_in_folder(file_names, folder_path):
    # Get a list of all the files in the folder
    # file_names = os.listdir(folder_path)
    # Create an empty list to store the arrays
    arrays = []
    # Loop through each file and import it as a numpy array
    for file_name in file_names:
        # Get the full path of the file
        file_path = os.path.join(folder_path, file_name)

        # Import the file as a numpy array
        array = np.load(file_path)

        # Append the array to the list of arrays
        arrays.append(array)

    # Stack the arrays into a single numpy array
    stacked_array = np.stack(arrays)

    return stacked_array

folder_path = 'output/cell_images/'

cell_list = os.listdir(folder_path)

x = import_files_in_folder(cell_list, folder_path)
x = np.swapaxes(x, 1, -1)
print(x.shape)

x = normalize(x)

# detect any cells with NaN values by flattening them, then filter out bad cells 
mask = np.isnan(x.reshape(x.shape[0], -1)).any(axis=1)
x = x[~mask]
masked_cell_list = np.array(cell_list)[~mask]

# choose permutation 
p = np.random.permutation(len(x))
x = x[p]
masked_cell_list = masked_cell_list[p]

print(x.shape)

data_copy  = x.copy() 

# x = np.load('../output/preprocessed_vae_data/img_2_16px_resized.npy', allow_pickle = True)
split_at = int(x.shape[0] * 0.9)

print(split_at, len(x))
# split_at = 3000

# x_train = [np.random.choice(x.shape[0], 3000, replace=False)]
# x_train = np.random.choice(x, size=3000, replace=False, axis = 0)

x_train = x[0:split_at]
x_test = x[split_at:]

# y_train = tissues[0:split_at]
# y_test = tissues[split_at:]

print('finished pre processing data')

# x_train = x_train.reshape(x_train.shape[0], img_depth, img_height, img_width, num_channels)
# x_test = x_test.reshape(x_test.shape[0], img_depth, img_height, img_width, num_channels)


input_shape = (img_depth, img_height, img_width, num_channels)
# input_shape = (32, 32, 32, 2)


# BUILD THE MODEL

# # ================= #############
# # Encoder
#Let us define 4 conv2D, flatten and then dense
# # ================= ############

input_img = Input(shape=input_shape, name='encoder_input')
x = Conv3D(32, 3, padding='same', activation='relu')(input_img)
x = Conv3D(64, 3, padding='same', activation='relu',strides=(2, 2, 2))(x)
x = Conv3D(64, 3, padding='same', activation='relu')(x)
x = Conv3D(64, 3, padding='same', activation='relu')(x)


# %%
conv_shape = K.int_shape(x) #Shape of conv to be provided to decoder
#Flatten
x = Flatten()(x)
x = Dense(32, activation='relu')(x)


# Two outputs, for latent mean and log variance (std. dev.)
#Use these to sample random variables in latent space to which inputs are mapped. 
z_mu = Dense(latent_dim, name='latent_mu')(x)   #Mean values of encoded input
z_sigma = Dense(latent_dim, name='latent_sigma')(x)  #Std dev. (variance) of encoded input


#REPARAMETERIZATION TRICK
# Define sampling function to sample from the distribution
# Reparameterize sample based on the process defined by Gunderson and Huang
# into the shape of: mu + sigma squared x eps
#This is to allow gradient descent to allow for gradient estimation accurately. 
def sample_z(args):
  z_mu, z_sigma = args
  eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
  return z_mu + K.exp(z_sigma / 2) * eps

# sample vector from the latent distribution
# z is the labda custom layer we are adding for gradient descent calculations
  # using mu and variance (sigma)
z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mu, z_sigma])

#Z (lambda layer) will be the last layer in the encoder.
# Define and summarize encoder model.
encoder = Model(input_img, [z_mu, z_sigma, z], name='encoder')
print(encoder.summary())


# decoder takes the latent vector as input
decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

# Need to start with a shape that can be remapped to original image shape as
#we want our final utput to be same shape original input.
#So, add dense layer with dimensions that can be reshaped to desired output shape
x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3]*conv_shape[4], activation='relu')(decoder_input)
# reshape to the shape of last conv. layer in the encoder, so we can 
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3], conv_shape[4]))(x)
# upscale (conv2D transpose) back to original shape
# use Conv2DTranspose to reverse the conv layers defined in the encoder
x = Conv3DTranspose(32, 3, padding='same', activation='relu',strides=(2, 2, 2))(x)
#Can add more conv2DTranspose layers, if desired. 
#Using sigmoid activation
x = Conv3DTranspose(num_channels, 3, padding='same', activation='sigmoid', name='decoder_output')(x)

# Define and summarize decoder model
decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

# apply the decoder to the latent sample 
z_decoded = decoder(z)




# =========================
#Define custom loss
#VAE is trained using two loss functions reconstruction loss and KL divergence
#Let us add a class to define a custom layer with loss
class CustomLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        
        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)

        # KL divergence
        kl_loss = -5e-4 * beta * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return K.mean(recon_loss + kl_loss)

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomLayer()([input_img, z_decoded])
# y is basically the original image after encoding input img to mu, sigma, z
# and decoding sampled z values.
#This will be used as output for vae



# =================
# VAE 
# =================
vae = Model(input_img, y, name='vae')

# Compile VAE
vae.compile(optimizer='adam', loss=None)
vae.summary()

# Train autoencoder
print(x_train.shape)
print(x_train.dtype)
vae.fit(x_train, None, epochs = 70, batch_size = 32, validation_split = 0.2)


# Save the entire model
vae.save('vae_model.h5')

plt.plot(vae.history.history['loss'][5:])
plt.plot(vae.history.history['val_loss'][5:])
plt.title('model accuracy')
# plt.xlim(80, 100)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.savefig(f'models/{model_name}_loss.png')
plt.close()

np.save(
    f'models/{model_name}_loss.npy', 
    np.array([vae.history.history['val_loss'], vae.history.history['loss']])
)


print(data_copy.shape)

mu, _, _ = encoder.predict(data_copy)

mu_df = pd.DataFrame(mu)
mu_df['cellname'] = masked_cell_list
mu_df.to_csv(f'models/{model_name}_latent_space_of_cells.csv')

import umap 
reducer = umap.UMAP(n_neighbors = 200)

mu = reducer.fit_transform(mu)
#Plot dim1 and dim2 for mu
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

ax.scatter(mu[:, 0], mu[:, 1])#, c=y_test_2, cmap='brg')
# plt.colorbar()
plt.show()

plt.savefig(f'models/{model_name}_umap.png')
plt.close()

# # %%


# # %%
# fig, ax = plt.subplots(1, 4, figsize = (15, 4), sharex = True, sharey = True, tight_layout = True)

# tissuelabels = ['PSM proper', 'tailbud', 'somite', 'spinal']
# tissuetypes = [3.0, 5, 4.0, 1.0]


# for i in range(4): 
#     # tissuetype = i+1
#     truths = [k == tissuetypes[i] for k in y_test_2]
#     axs = fig.add_subplot(1, 4, i+1)
#     ax[i].set_title(tissuelabels[i])
#     # axs.scatter3D(mu[:, 0], mu[:, 1], mu[:, 2], s = 1, c = 'darkgrey')
#     axs.scatter(
#         mu[truths, 0], 
#         mu[truths, 1], 
#         # mu[truths, 2], 
#         c=xpos[truths], cmap='plasma', s = 3)
#     ax[i].axis('off')



# # fig.suptitle('Cells in the early tailbud occupy different regions of latent space to those in the PSM proper, and spinal cord\nColor represents X position: dark- posterior; light- anterior')

# plt.savefig(f'models/{model_name}_tissues.png')
# plt.close()


# # fig, ax = plt.subplots(1, 4, figsize = (15, 4), sharex = True, sharey = True, tight_layout = True)

# # tissuelabels = ['PSM proper', 'tailbud', 'somite', 'spinal']
# # tissuetypes = [3.0, 5, 4.0, 1.0]

# # mu, _, _ = encoder.predict(x_train)

# # y_test_2 = centroids_subset.reset_index(drop = True).loc[:split_at - 1, 'tissue_b']
# # xpos = np.array(centroids_subset.reset_index(drop = True).loc[:split_at - 1, 'X'])

# # for i in range(4): 
# #     # tissuetype = i+1
# #     truths = [k == tissuetypes[i] for k in y_test_2]
# #     ax[i].set_title(tissuelabels[i])
# #     ax[i].scatter(mu[:, 0], mu[:, 1], s = 1, c = 'lightgrey')
# #     ax[i].scatter(
# #         mu[truths, 0], 
# #         mu[truths, 1], 
# #         c=xpos[truths], cmap='plasma', s = 1)

# # # plt.colorbar()

# # fig.suptitle('Cells in the early tailbud occupy different regions of latent space to those in the PSM proper, and spinal cord\nColor represents X position: dark- posterior; light- anterior')

# # # %%
# # plt.scatter(mu[truths, 0], xpos[truths], s = 5)


# # import seaborn as sns

# # # when we graph 

# # df = pd.DataFrame(mu)
# # df['tissue'] = list(y_test_2)
# # df['tissue'] = df['tissue'].astype(str)

# # sns.kdeplot(data = df, x = 0, y = 1, hue = 'tissue')


# n = 10  

# figure = np.zeros((img_width * n, img_height * n))

# #Create a Grid of latent variables, to be provided as inputs to decoder.predict
# #Creating vectors within range -5 to 5 as that seems to be the range in latent space
# grid_x = np.linspace(-3, 3, n)
# grid_y = np.linspace(-3, 3, n)[::-1]

# # decoder for each square in the grid
# for i, yi in enumerate(grid_y):
#     for j, xi in enumerate(grid_x):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = decoder.predict(z_sample)
#         digit = x_decoded[0].reshape(img_depth, img_width, img_height, num_channels)[:, 16, :, 1]
#         figure[i * img_width: (i + 1) * img_width,
#                j * img_height: (j + 1) * img_height] = digit

# plt.figure(figsize=(10, 10))
# #Reshape for visualization
# fig_shape = np.shape(figure)
# figure = figure.reshape((fig_shape[0], fig_shape[1]))

# plt.imshow(figure, cmap='viridis')
# plt.show()  

# plt.savefig(f'../models/{model_name}_{beta}_20-10-22_reconstructed_nuclei.png')
# plt.close()

# # %%
# figure = np.zeros((img_width * n, img_height * n))

# #Create a Grid of latent variables, to be provided as inputs to decoder.predict
# #Creating vectors within range -5 to 5 as that seems to be the range in latent space


# # decoder for each square in the grid
# for i, yi in enumerate(grid_y):
#     for j, xi in enumerate(grid_x):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = decoder.predict(z_sample)
#         digit = x_decoded[0].reshape(img_depth, img_width, img_height, num_channels)[16, :, :, 0]
#         figure[i * img_width: (i + 1) * img_width,
#                j * img_height: (j + 1) * img_height] = digit

# plt.figure(figsize=(10, 10))
# #Reshape for visualization
# fig_shape = np.shape(figure)
# figure = figure.reshape((fig_shape[0], fig_shape[1]))

# plt.imshow(figure, cmap='viridis')
# plt.show()  
# plt.savefig(f'../models/{model_name}_{beta}_20-10-22_reconstructed_membranes.png')
# plt.close()

