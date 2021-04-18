"""____________________Config for Dual-VAE/GAN training___________________________"""

pretrained_gan = 'gan_20210413-102934'  # 'gan_cogenc_20210131-183016'
load_epoch = 335
evaluate = False

image_crop = 375
image_size = 100
latent_dim = 512

device = 'cuda:0'  # cuda or cpu
device2 = 'cuda:3'
device3 = 'cuda:5'

patience = 0   # for early stopping, 0 = deactivate early stopping
data_split = 0.2
batch_size = 100
learning_rate = 0.0001
weight_decay = 1e-7
n_epochs = 400
num_workers = 4
step_size = 30  # for scheduler
gamma = 0.1     # for scheduler
recon_level = 3
lambda_mse = 1e-6
decay_lr = 0.98
decay_margin = 1
decay_mse = 1
decay_equilibrium = 1
margin = 0.35
equilibrium = 0.68
beta = 1.0

kernel_size = 4
stride = 2
padding_mode = [1, 1, 1, 1, 1, 0]
dropout = 0.7

save_images = 5
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# [model, epoch]
# Trained model for stage II (model from stage I)
decoder_weights = ['gan_20210127-012348', 90]       # latent dim = 128
# decoder_weights = ['wae_gan_20210222-164403', 16]
# decoder_weights = ['wae_gan_20210218-013356', 80] # latent dim = 512
# decoder_weights = ['gan_20210222-152350', 355]    # latent dim = 128
# decoder_weights = ['gan_20210310-171435', 90]     # vae
# decoder_weights = ['gan_20210311-105943', 395]    # latent dim = 128
# decoder_weights = ['gan_20210311-105943', 75]     # latent dim = 128
# decoder_weights = ['gan_20210318-230948', 40]     # latent dim = 512
decoder_weights = ['gan_20210320-233307', 60]     # latent dim = 512, resolution 100
# decoder_weights = ['gan_20210325-001446', 25]     # wae + vae/gan, latent dim = 512, resolution 100
# decoder_weights = ['gan_20210414-001926', 10]       # mnist

# Trained model for stage III (model from stage II)
cog_encoder_weights = ['gan_cog_2st_20210223-224153', 395]
# cog_encoder_weights = ['gan_cog_2st_20210224-231103', 395]  # subject 3
cog_encoder_weights = ['gan_cog_2st_20210302-234434', 375]    # latent dim = 128
# cog_encoder_weights = ['gan_cog_2st_20210223-224153', 260]
# cog_encoder_weights = ['gan_cog_2st_20210307-125311', 60]   # latent dim = 128
cog_encoder_weights = ['gan_cog_2st_20210308-201047', 390]    # subject 3
# cog_encoder_weights = ['gan_cog_2st_20210311-123228', 390]  # vae
# cog_encoder_weights = ['gan_cog_2st_20210313-002005', 395]
# cog_encoder_weights = ['gan_cog_2st_20210315-153149', 330]  # latent dim = 128
cog_encoder_weights = ['gan_cog_2st_20210321-092029', 395]  # latent dim = 512, resolution 100
# cog_encoder_weights = ['gan_cog_2st_20210325-110326', 395]  # latent dim = 512, resolution 100 wae + vae/gan
# cog_encoder_weights = ['gan_cog_2st_20210414-101744', 335]    # mnist
