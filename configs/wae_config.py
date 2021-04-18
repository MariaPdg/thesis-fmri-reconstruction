"""____________________Config for WAE/GAN training___________________________"""

pretrained_gan = None  # 'gan_20210203-173210' # 'gan_20210127-012348'
load_epoch = 90

image_crop = 375
image_size = 64
latent_dim = 128

device = 'cuda:2'
device2 = 'cuda:3'
device3 = 'cuda:5'

patience = 0   # for early stopping, 0 = deactivate early stopping
data_split = 0.2
batch_size = 64
learning_rate = 0.0001
weight_decay = 1e-7
n_epochs = 200
num_workers = 4
step_size = 30  # for scheduler
decay_lr = 0.5    # for scheduler
recon_level = 3
lambda_mse = 1e-6

decay_margin = 1
decay_mse = 1
decay_equilibrium = 1
margin = 0.35
equilibrium = 0.68

save_images = 5
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# [model, epoch]
# decoder_weights = ['wae_gan_20210218-013356', 80]   # latent dim = 512
decoder_weights = ['wae_gan_20210222-164403', 398]    # latent dim = 128
# decoder_weights = ['wae_gan_20210217-012437', 150]  # latent dim = 1024
# decoder_weights = ['wae_gan_20210302-211428', 365]  # latent dim = 512
# decoder_weights = ['wae_gan_20210302-211537', 395]  # latent dim = 1024

cog_encoder_weights = ['waegan_cog_20210223-225621', 395]    # latent dim = 128
# cog_encoder_weights = ['waegan_cog_20210225-111206', 395]  # latent dim = 1024
# cog_encoder_weights = ['waegan_cog_20210304-140820', 150]  # latent dim = 512
# cog_encoder_weights = ['waegan_cog_20210304-223112', 360]  # latent dim = 1024
