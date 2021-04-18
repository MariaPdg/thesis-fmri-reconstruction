"""___________________Config for inference_________________________________"""

train_data = 'BOLD5000/bold_train/bold_CSI4_pad.pickle'
valid_data = 'BOLD5000/bold_train/bold_CSI4_pad.pickle'

"""_______________Dual-VAE/GAN  Stage III resolution 64____________________"""

dataset = 'bold'       # 'bold' or 'coco'
mode = 'vae-gan'       # vae, wae-gan, vae-gan

folder_name = 'gan_cog_3st'
pretrained_gan = 'gan_cog_3st_20210310-214859'  # all
# pretrained_gan = 'gan_cog_3st_20210310-215029'  # subject 1
# pretrained_gan = 'gan_cog_3st_20210310-215143'  # subject 2
# pretrained_gan = 'gan_cog_3st_20210310-215244'  # subject 3
# pretrained_gan = 'gan_cog_3st_20210310-215345'  # subject 4
load_epoch = 195

"""________________Dual-VAE/GAN  Stage III resolution 100___________________"""

# dataset = 'bold'       # 'bold' or 'coco'
# mode = 'vae-gan'       # vae, wae-gan, vae-gan
#
# folder_name = 'gan_cog_3st'
# # pretrained_gan = 'gan_cog_3st_20210323-002831'  # latent dim 512, resolution 100, subject 4
# # pretrained_gan = 'gan_cog_3st_20210322-222648'  # latent dim 512, resolution 100, subject 3
# load_epoch = 195


"""________________________Dual-VAE Stage III ______________________________"""
#
# dataset = 'bold'       # 'bold' or 'coco'
# mode = 'vae'           # vae, wae-gan, vae-gan
#
# folder_name = 'gan_cog_3st'
# pretrained_gan = 'gan_cog_3st_20210323-213512'  # vae, all
# # pretrained_gan = 'gan_cog_3st_20210312-082122'  # vae subject 3
# load_epoch = 195


"""__________________________WAE/GAN Stage III _______________________________"""
#
# dataset = 'bold'       # 'bold' or 'coco'
# mode = 'wae-gan'       # vae, wae-gan, vae-gan
#
# folder_name = 'wae_3st'
# pretrained_gan = 'wae_gan_20210222-164403'
# load_epoch = 190
# pretrained_gan = 'wae_3st_20210312-222544'
# # pretrained_gan = 'wae_3st_20210323-182653'  # model all
#

"""________________________WAE/Dual-GAN Stage I______________________________"""
#
dataset = 'coco'       # 'bold' or 'coco'
mode = 'vae-gan'       # vae, wae-gan, vae-gan

folder_name = 'gan'
# pretrained_gan = 'gan_20210325-001446'  # wae + vae/gan, latent dim = 512, resolution 100
# load_epoch = 50
pretrained_gan = 'gan_20210413-102934'  # higher performance
load_epoch = 40

"""__________________________VAE/GAN Stage I__________________________________"""

# dataset = 'coco'       # 'bold' or 'coco'
# mode = 'vae-gan'       # vae, wae-gan, vae-gan, latent dim = 512, resolution 100
#
# folder_name = 'gan'
# pretrained_gan = 'gan_20210320-233307'
# load_epoch = 15


"""___________________________Inference parameters_____________________________"""

evaluate = True       # True if you want to evaluate
save = True         # True to save images
save_to_folder = None  # specify folder name if you want to save in specific directory
file_to_save = None    # save .csv file with results

image_crop = 375
image_size = 100
latent_dim = 512

batch_size = 64
num_workers = 4
recon_level = 3

device = 'cuda:4'
device2 = 'cuda:3'
device3 = 'cuda:5'

save_images = 5
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
