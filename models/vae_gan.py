import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

import configs.models_config as config


class EncoderBlock(nn.Module):
    """
    Encoder block used in encoder and discriminator
    """
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=config.kernel_size,
                              padding=config.padding, stride=config.stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = F.relu(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
            return ten


class DecoderBlock(nn.Module):
    """
    Decoder block used in decoder
    """
    def __init__(self, channel_in, channel_out, out=False):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        if out:
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=config.kernel_size, padding=config.padding,
                                           stride=config.stride, output_padding=1,
                                           bias=False)
        else:
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=config.kernel_size,
                                           padding=config.padding,
                                           stride=config.stride,
                                           bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten


class Encoder(nn.Module):
    """
    Visual encoder with parameters from config.
    Used to transform images to visual latent features in training in Stage I
    """
    def __init__(self, channel_in=3, z_size=128):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(3):
            layers_list.append(EncoderBlock(channel_in=self.size, channel_out=config.encoder_channels[i]))
            self.size = config.encoder_channels[i]

        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=config.fc_input * config.fc_input * self.size,
                                          out_features=config.fc_output, bias=False),
                                nn.BatchNorm1d(num_features=config.fc_output, momentum=0.9),
                                nn.ReLU(True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=config.fc_output, out_features=z_size)
        self.l_var = nn.Linear(in_features=config.fc_output, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    """
    Decoder with parameters from config.
    Used to transform latent features (visual or cognitive) to images
    """
    def __init__(self, z_size, size):
        super(Decoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=config.fc_input * config.fc_input * size, bias=False),
                                nn.BatchNorm1d(num_features=config.fc_input * config.fc_input * size, momentum=0.9),
                                nn.ReLU(True))
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size, out=config.output_pad_dec[0]))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=config.decoder_channels[1], out=config.output_pad_dec[1]))
        self.size = config.decoder_channels[1]
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=config.decoder_channels[2], out=config.output_pad_dec[2]))
        self.size = config.decoder_channels[2]
        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=config.decoder_channels[3], kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, config.fc_input, config.fc_input)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class Discriminator(nn.Module):
    """
    Discriminator with parameters from config. Used in VAE/GAN
    """
    def __init__(self, channel_in=3, recon_level=3):
        super(Discriminator, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=config.discrim_channels[0], kernel_size=5, stride=config.stride_gan, padding=2),
            nn.ReLU(inplace=True)))
        self.size = config.discrim_channels[0]
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=config.discrim_channels[1]))
        self.size = config.discrim_channels[1]
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=config.discrim_channels[2]))
        self.size = config.discrim_channels[2]
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=config.discrim_channels[3]))
        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=config.fc_input_gan * config.fc_input_gan * self.size,
                      out_features=config.fc_output_gan, bias=False),
            nn.BatchNorm1d(num_features=config.fc_output_gan, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=config.fc_output_gan, out_features=1),
        )

    def forward(self, ten_orig, ten_predicted, ten_sampled, mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            return F.sigmoid(ten)
            # return ten

    def __call__(self, *args, **kwargs):
        return super(Discriminator, self).__call__(*args, **kwargs)


class CognitiveEncoder(nn.Module):
    """
    Cognitive encoder to transform fMRI to cognitive latent representations.
    Used in training on Stage II and III
    """
    def __init__(self, input_size, z_size=128, channel_in=3):
        super(CognitiveEncoder, self).__init__()
        self.size = channel_in

        self.fc1 = nn.Sequential(nn.Linear(in_features=input_size, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                nn.ReLU(True))
        # self.fc2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512, bias=False),
        #                         nn.BatchNorm1d(num_features=512, momentum=0.9),
        #                         nn.LeakyReLU(True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=z_size)
        self.l_var = nn.Linear(in_features=1024, out_features=z_size)
    #     self.init_parameters()
    #
    # def init_parameters(self):
    #     # just explore the network, find every weight and bias matrix and fill it
    #     for m in self.modules():
    #         if isinstance(m, (nn.Linear)):
    #             if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
    #                 # init as original implementation
    #                 # scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
    #                 # scale /= numpy.sqrt(3)
    #                 nn.init.xavier_normal_(m.weight, 1)
    #                 # nn.init.constant(m.weight, 0.005)
    #                 # nn.init.uniform_(m.weight, -scale, scale)
    #             if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
    #                 nn.init.constant_(m.bias, 0.0)

    def forward(self, ten):
        ten = self.fc1(ten)
        # ten = self.fc2(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(CognitiveEncoder, self).__call__(*args, **kwargs)


class VaeGan(nn.Module):
    """
    VAE/GAN model, which is used in training on Stage I: image-to-image translation.
    Modified from https://github.com/lucabergamini/VAEGAN-PYTORCH
    """
    def __init__(self, device, z_size=128, recon_level=3):
        super(VaeGan, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size).to(device)
        # self.encoder = ResNetEncoder(z_size=self.z_size).to(device)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size).to(device)
        self.discriminator = Discriminator(channel_in=3, recon_level=recon_level).to(device)
        # self-defined function to init the parameters
        self.init_parameters()
        self.device = device

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x, gen_size=10):

        if x is not None:
            x = Variable(x).to(self.device)

        if self.training:
            mus, log_variances = self.encoder(x)
            z = self.reparameterize(mus, log_variances)
            x_tilde = self.decoder(z)

            z_p = Variable(torch.randn(len(x), self.z_size).to(self.device), requires_grad=True)
            x_p = self.decoder(z_p)

            disc_layer = self.discriminator(x, x_tilde, x_p, "REC")  # discriminator for reconstruction
            disc_class = self.discriminator(x, x_tilde, x_p, "GAN")

            return x_tilde, disc_class, disc_layer, mus, log_variances
        else:
            if x is None:
                z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)  # just sample and decode
                x_p = self.decoder(z_p)
                return x_p
            else:
                mus, log_variances = self.encoder(x)
                z = self.reparameterize(mus, log_variances)
                x_tilde = self.decoder(z)
                return x_tilde

    def __call__(self, *args, **kwargs):
        return super(VaeGan, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(x, x_tilde, disc_layer_original, disc_layer_predicted, disc_layer_sampled, disc_class_original,
             disc_class_predicted, disc_class_sampled, mus, variances):

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (x.view(len(x), -1) - x_tilde.view(len(x_tilde), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)

        # mse between intermediate layers
        mse = torch.sum(0.5 * (disc_layer_original - disc_layer_predicted) ** 2, 1)

        # bce for decoder and discriminator for original and reconstructed
        bce_dis_original = -torch.log(disc_class_original + 1e-3)
        bce_dis_predicted = -torch.log(1 - disc_class_predicted + 1e-3)
        bce_dis_sampled = -torch.log(1 - disc_class_sampled + 1e-3)

        return nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled


class VaeGanCognitive(nn.Module):

    """
    Dual-VAE/GAN model, which is trained with 3-stage training procedure.
    This model used in training on Stage II and III.
    On Stage II we do knowledge distillation from teacher network (trained on Stage I):
    fix decoder weights and use images generated on Stage I as real for discriminator
    Stage III is fine-tuning with the fixed cognitive encoder.

    """

    def __init__(self, device, encoder, decoder, discriminator, z_size=128, recon_level=3, teacher_net=None, stage=1,
                 mode='vae'):
        super(VaeGanCognitive, self).__init__()
        # latent space size
        self.device = device
        self.z_size = z_size
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.teacher_net = teacher_net
        self.stage = stage
        self.mode = mode

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, sample, gen_size=10):

        if sample is not None:

            x = Variable(sample['fmri'], requires_grad=False).to(self.device)
            gt_x = Variable(sample['image'], requires_grad=False).to(self.device)

            if self.training:

                    if self.mode == 'vae':

                        mus, log_variances = self.encoder(x)
                        z = self.reparameterize(mus, log_variances)
                        x_tilde = self.decoder(z)

                        if self.teacher_net is not None and self.stage == 2:

                            for param in self.teacher_net.encoder.parameters():
                                param.requires_grad = False

                            # Inter-modality knowledge distillation
                            mu_teacher, logvar_teacher = self.teacher_net.encoder(gt_x)
                            # Re-parametrization trick
                            z_teacher = self.reparameterize(mu_teacher, logvar_teacher)
                            # Reconstruct gt by the teacher net
                            gt_x = self.decoder(z_teacher)

                    elif self.mode == 'wae':

                        mus, log_variances = self.encoder(x)
                        x_tilde = self.decoder(mus)

                        # Inter-modality knowledge distillation
                        mu_teacher, logvar_teacher = self.teacher_net.encoder(gt_x)
                        # Reconstruct gt by the teacher net
                        gt_x = self.decoder(mu_teacher)

                    z_p = Variable(torch.randn(len(x), self.z_size).to(self.device), requires_grad=True)
                    x_p = self.decoder(z_p)

                    disc_layer = self.discriminator(gt_x, x_tilde, x_p, "REC")  # discriminator for reconstruction
                    disc_class = self.discriminator(gt_x, x_tilde, x_p, "GAN")

                    return gt_x, x_tilde, disc_class, disc_layer, mus, log_variances

            else:
                # Use fmri only for evaluation
                mus, log_variances = self.encoder(x)
                z = self.reparameterize(mus, log_variances)
                x_tilde = self.decoder(z)
                return x_tilde
        else:
                z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)  # just sample and decode
                x_p = self.decoder(z_p)
                return x_p

    def __call__(self, *args, **kwargs):
        return super(VaeGanCognitive, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(gt_x, x_tilde, disc_layer_original, disc_layer_predicted, disc_layer_sampled, disc_class_original,
             disc_class_predicted, disc_class_sampled, mus, variances):

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (gt_x.view(len(gt_x), -1) - x_tilde.view(len(x_tilde), -1)) ** 2
        # mse_loss = nn.MSELoss()
        # nle = mse_loss(gt_x, x_tilde)

        # kl-divergence
        kld_weight = 1
        kld = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1) * kld_weight

        # mse between intermediate layers
        mse = torch.sum(0.5 * (disc_layer_original - disc_layer_predicted) ** 2, 1)

        # bce for decoder and discriminator for original and reconstructed
        bce_dis_original = -torch.log(disc_class_original + 1e-3)
        bce_dis_predicted = -torch.log(1 - disc_class_predicted + 1e-3)
        bce_dis_sampled = -torch.log(1 - disc_class_sampled + 1e-3)

        return nle, kld, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled


class WaeGan(nn.Module):
    """
    WAE with GAN-based penalty. WAE discriminator distinguishes real and fake latent distributions
    rather than images such as for standard GANs.
    """
    def __init__(self, device, z_size=128):
        super(WaeGan, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size).to(device)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size).to(device)
        # self.decoder = WaeDecoder(z_size=self.z_size, size=self.encoder.size).to(device)
        self.discriminator = WaeDiscriminator(z_size=self.z_size).to(device)
        # self-defined function to init the parameters
        self.init_parameters()
        self.device = device

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, x, gen_size=10):

        if x is not None:
            x = Variable(x).to(self.device)

        if self.training:

            # TODO: currently not used

            mus, log_variances = self.encoder(x)
            x_tilde = self.decoder(mus)

            z_p = Variable(torch.randn(len(mus), self.z_size).to(self.device), requires_grad=True)
            x_p = self.decoder(z_p)

            disc_class = self.discriminator(mus, x_p, "GAN")  # encoder distribution

            return x_tilde, disc_class, mus, log_variances
        else:
            if x is None:
                # z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)  # just sample and decode
                z_p = Variable(torch.randn_like(x).to(self.device), requires_grad=False)
                x_p = self.decoder(z_p)
                return x_p
            else:
                mus, log_variances = self.encoder(x)
                x_tilde = self.decoder(mus)
                return x_tilde

    def __call__(self, *args, **kwargs):
        return super(WaeGan, self).__call__(*args, **kwargs)


class WaeDiscriminator(nn.Module):

    """
    Discriminator for the latent space to distinguish real and fake latent representations
    """

    def __init__(self, z_size=128, dim_h=512):
        super(WaeDiscriminator, self).__init__()
        self.n_z = z_size
        self.dim_h = dim_h

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, 1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.main(x)
        return x


class WaeGanCognitive(nn.Module):

    """
    WAE/GAN model for training in Stage II and III
    """
    def __init__(self, device, encoder, decoder, z_size=128, recon_level=3):
        super(WaeGanCognitive, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = encoder
        self.discriminator = WaeDiscriminator(z_size=self.z_size).to(device)
        self.device = device
        self.decoder = decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x, gen_size=10):

        if x is not None:
            x = Variable(x).to(self.device)

        if self.training:
            mus, log_variances = self.encoder(x)
            z = self.reparameterize(mus, log_variances)
            x_tilde = self.decoder(mus)

            z_p = Variable(torch.randn(len(mus), self.z_size).to(self.device), requires_grad=True)
            x_p = self.decoder(z_p)

            disc_class = self.discriminator(mus, x_p, "GAN")  # encoder distribution

            return x_tilde, disc_class, mus, log_variances
        else:
            if x is None:
                # z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)  # just sample and decode
                z_p = Variable(torch.randn_like(x).to(self.device), requires_grad=False)
                x_p = self.decoder(z_p)
                return x_p
            else:
                mus, log_variances = self.encoder(x)
                x_tilde = self.decoder(mus)
                return x_tilde


class DCGan(nn.Module):
    """
    Simple DCGAN implementation
    """

    def __init__(self, device, decoder, discriminator, z_size=128, recon_level=3):
        super(DCGan, self).__init__()
        # latent space size
        self.device = device
        self.z_size = z_size
        self.decoder = decoder
        self.discriminator = discriminator

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, sample, gen_size=10):

        if sample is not None:

            gt_x = Variable(sample).to(self.device)

            if self.training:

                z_p = Variable(torch.randn(len(gt_x), self.z_size).to(self.device), requires_grad=True)
                x_tilde = self.decoder(z_p)

                disc_layer = self.discriminator(gt_x, x_tilde, x_tilde, "REC")  # discriminator for reconstruction
                disc_class = self.discriminator(gt_x, x_tilde, x_tilde, "GAN")

                return gt_x, x_tilde, disc_class, disc_layer

            else:
                z_p = Variable(torch.randn(gt_x.shape[0], self.z_size).to(self.device), requires_grad=False)  # just sample and decode
                x_p = self.decoder(z_p)
                return x_p
        else:
                z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)  # just sample and decode
                x_p = self.decoder(z_p)
                return x_p


class WaeDecoder(nn.Module):

    def __init__(self, z_size, size):
        super(WaeDecoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=16 * 16 * 1024, bias=False),
                                nn.BatchNorm1d(num_features=16 * 16 * 1024, momentum=0.9),
                                nn.ReLU(True))
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=1024, channel_out=512, out=config.output_pad_dec[0]))
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, out=config.output_pad_dec[1]))
        self.size = config.output_pad_dec[1]
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, out=config.output_pad_dec[2]))
        self.size = 128
        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, config.fc_input, config.fc_input)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(WaeDecoder, self).__call__(*args, **kwargs)


class ResNetEncoder(nn.Module):

    def __init__(self, z_size=128, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3):
        super(ResNetEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, z_size

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.size = self.ch4

        # encoding components
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar
