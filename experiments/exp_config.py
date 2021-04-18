
pretrained_gan = 'exp_dcgan_20210314-225719'  # 'gan_cogenc_20210131-183016'
load_epoch = 190
evaluate = False

image_crop = 375
image_size = 64

patience = 0   # for early stopping, 0 = deactivate early stopping

data_split = 0.2
batch_size = 100
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

device = 'cuda:6'
device2 = 'cuda:3'
device3 = 'cuda:5'

kernel_size = 4
stride = 2
padding_mode = [1, 1, 1, 1, 1, 0]
dropout = 0.7

latent_dim = 128

save_images = 5
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Trained model for stage II (model from stage I)
# decoder_weights = 'gan_20210126-100450'
decoder_weights = ['exp_dcgan_20210222-001759', 390]  # model, epoch

# Trained model for stage III (model from stage II)
cog_encoder_weights = ['dcgan_2st_20210227-132740', 390]
