
pretrained_gan = None  # 'gan_20210203-173210' # 'gan_20210127-012348'
load_epoch = 395
evaluate = False

patience = 0   # for early stopping, 0 = deactivate early stopping

image_crop = 375
image_size = 100
latent_dim = 512

data_split = 0.2
batch_size = 64
learning_rate = 0.0001
weight_decay = 1e-7
n_epochs = 400
num_workers = 4
step_size = 20  # for scheduler
gamma = 0.1     # for scheduler
recon_level = 3
lambda_mse = 1e-6
decay_lr = 0.75
decay_margin = 1
decay_mse = 1
decay_equilibrium = 1
margin = 0.35
equilibrium = 0.68
beta = 1.0

device = 'cuda:2'
device2 = 'cuda:3'
device3 = 'cuda:5'

save_images = 5
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

decoder_weights = ['wae_gan_20210222-164403', 398]  # model, epoch