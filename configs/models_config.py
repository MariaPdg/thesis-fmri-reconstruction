"""_______________________Config with model parameters__________________"""

kernel_size = 5
stride = 2
padding = 2
dropout = 0.7

encoder_channels = [64, 128, 256]
decoder_channels = [256, 128, 32, 3]
discrim_channels = [32, 128, 256, 256, 512]

# paper settings
image_size = 100
fc_input = 13  # 8/13/14/16/28 image_size = 64/100/112/128/224
fc_output = 1024
fc_input_gan = 7
fc_output_gan = 256
stride_gan = 2
latent_dim = 512
output_pad_dec = [False, True, True]
decoder_channels = [256, 128, 64, 3]

# settings for resolution 64
# image_size = 64
# fc_input = 8
# fc_output = 1024
# fc_input_gan = 8
# fc_output_gan = 512
# stride_gan = 1
# latent_dim = 128
# output_pad_dec = [True, True, True]
