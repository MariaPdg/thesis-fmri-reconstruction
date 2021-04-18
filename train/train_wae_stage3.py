import os
import time
import numpy
import json
import torch
import pickle
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from sklearn.model_selection import train_test_split

import configs.data_config as data_cfg
import configs.wae_config as wae_cfg
from models.vae_gan import WaeGan, CognitiveEncoder, WaeGanCognitive, Decoder
from train.train_utils import evaluate, PearsonCorrelation, StructuralSimilarity
from data_preprocessing.data_loader import BoldRoiDataloader, CenterCrop, Rescale, RandomShift, SampleToTensor, \
    Normalization, split_subject_data

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

DEBUG = False


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="user path where the datasets are located", type=str)
    parser.add_argument('--output', '-o', help='user path where to save', type=str)
    parser.add_argument('--logs', '-l', help='path where to save logs', type=str)
    parser.add_argument('--batch_size', '-b', default=wae_cfg.batch_size, help='batch size for dataloader', type=int)
    parser.add_argument('--learning_rate', '-lr', default=wae_cfg.learning_rate, help='learning rate', type=float)
    parser.add_argument('--epochs', '-e', default=wae_cfg.n_epochs, help='number of epochs', type=int)
    parser.add_argument('--image_crop', '-im_crop', default=wae_cfg.image_crop, help='size to which image should '
                                                                                     'be cropped', type=int)
    parser.add_argument('--image_size', '-im_size', default=wae_cfg.image_size, help='size to which image should '
                                                                                     'be scaled', type=int)
    parser.add_argument('--device', '-d', default=wae_cfg.device, help='what device to use', type=str)
    parser.add_argument('--num_workers', '-nw', default=wae_cfg.num_workers, help='number of workers for dataloader',
                        type=int)
    parser.add_argument('--step_size', '-step', default=wae_cfg.step_size, help='number of epochs after which '
                                                                                'to decrease learning rate', type=int)
    parser.add_argument('--patience', '-p', default=wae_cfg.patience, help='number of epochs with unchanged lr '
                                                                           'for early stopping', type=int)
    parser.add_argument('--weight_decay', '--wd', default=wae_cfg.weight_decay, help='weight decay used by optimizer',
                        type=float)
    parser.add_argument('--latent_dim', '-lat_dim', default=wae_cfg.latent_dim, help='dimension of the latent space',
                        type=int)
    parser.add_argument('--message', '-m', default='default message', help='experiment description', type=str)
    parser.add_argument('--pretrained_gan', '-pretrain', default=wae_cfg.pretrained_gan, help='pretrained gan',
                        type=str)
    parser.add_argument('-load_epoch', '-pretrain_epoch', default=wae_cfg.load_epoch,
                        help='epoch of the pretrained model',
                        type=int)
    parser.add_argument('--recon_level', default=wae_cfg.recon_level, help='reconstruction level in the descriminator',
                        type=int)
    parser.add_argument('--lambda_mse', default=wae_cfg.lambda_mse, type=float, help='weight for style error')
    parser.add_argument('--decay_mse', default=wae_cfg.decay_mse, type=float, help='mse weight decrease')
    parser.add_argument('--decay_lr', default=wae_cfg.decay_lr, type=float, help='learning rate decay for lr scheduler')
    parser.add_argument('--margin', default=wae_cfg.margin, type=float, help='margin for generator/discriminator game')
    parser.add_argument('--equilibrium', default=wae_cfg.equilibrium, type=float,
                        help='equilibrium for the generator/discriminator game')
    parser.add_argument('--decay_margin', default=wae_cfg.decay_margin, type=float,
                        help='margin decay for the generator/discriminator game')
    parser.add_argument('--decay_equilibrium', default=wae_cfg.decay_equilibrium, type=float,
                        help='equilibrium decay for the generator/discriminator game')
    parser.add_argument('--decoder', '-dec', default=wae_cfg.decoder_weights,
                        help='pretrained wae-gan model to fix decoder weights, training stage II', type=str)
    parser.add_argument('--cog_encoder', '-enc', default=wae_cfg.cog_encoder_weights,
                        help='pretrained vae-gan-cog model to fix cognitive encoder weights , training stage III', type=str)

    args = parser.parse_args()

    # Path to pickle file with bold5000 data
    USER_ROOT = args.output
    DATA_PATH = os.path.join(args.input, data_cfg.data_root)
    SAVE_PATH = os.path.join(USER_ROOT, data_cfg.save_training_results)
    TRAIN_DATA_PATH = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.train_data)
    VALID_DATA_PATH = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.valid_data)

    COCO_TEST_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_test_data)
    COCO_TRAIN_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_train_data)
    COCO_VALID_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_valid_data)

    # Split with fixed stimuli IDs
    TRAIN_STIMULI = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.train_stimuli_split)
    VALID_STIMULI = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.valid_stimuli_split)

    ENCODER_WEIGHTS = os.path.join(USER_ROOT, data_cfg.save_training_results, 'waegan_cog', args.cog_encoder[0],
                                   args.cog_encoder[0] + '_' + str(args.cog_encoder[1]) + '.pth')
    DECODER_WEIGHTS = os.path.join(USER_ROOT, data_cfg.save_training_results, 'wae_gan', args.decoder[0], args.decoder[0] +
                                   '_' + str(args.decoder[1]) + '.pth')

    # Create directory to save weights
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Info logging
    timestep = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.logs, 'train_wae_3st' + timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Check available gpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device2 = wae_cfg.device2
    device3 = wae_cfg.device3
    logger.info("Used device: %s" % device)

    logging.info('set up random seeds')
    torch.manual_seed(12345)

    # Create directory for results
    if DEBUG:
        saving_dir = os.path.join(SAVE_PATH, 'debug', 'debug_wae_3st_{}'.format(timestep))
    else:
        saving_dir = os.path.join(SAVE_PATH, 'wae_3st', 'wae_3st_{}'.format(timestep))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    if args.pretrained_gan is not None:
        pretrained_model_dir = os.path.join(SAVE_PATH, 'wae_3st', args.pretrained_gan, args.pretrained_gan + '.pth')
    saving_name = os.path.join(saving_dir, 'wae_3st_{}.pth'.format(timestep))

    # Save arguments
    with open(os.path.join(saving_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load data which were concatenated for 4 subjects and split into training and validation sets
    with open(TRAIN_DATA_PATH, "rb") as input_file:
        train_data = pickle.load(input_file)
    with open(VALID_DATA_PATH, "rb") as input_file:
        valid_data = pickle.load(input_file)

    # Split and save data if train only for 1 subject
    # train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=12345)
    # print(len(train_data))
    # print(len(valid_data))
    # with open(os.path.join(saving_dir, 'train_data'), 'wb') as f:
    #     pickle.dump(train_data, f)
    # with open(os.path.join(saving_dir, 'valid_data'), 'wb') as f:
    #     pickle.dump(valid_data, f)

    # Split with fixed stimuli IDs, use for one subject
    train_data = split_subject_data(train_data, TRAIN_STIMULI)
    valid_data = split_subject_data(valid_data, VALID_STIMULI)

    # Load data
    training_data = BoldRoiDataloader(dataset=train_data,
                                      root_path=DATA_PATH,
                                      transform=transforms.Compose([
                                                                CenterCrop(output_size=args.image_crop),
                                                                Rescale(output_size=args.image_size),
                                                                RandomShift(),
                                                                SampleToTensor(),
                                                                Normalization(mean=wae_cfg.mean,
                                                                              std=wae_cfg.std)]))

    validation_data = BoldRoiDataloader(dataset=valid_data,
                                        root_path=DATA_PATH,
                                        transform=transforms.Compose([
                                                                CenterCrop(output_size=args.image_crop),
                                                                Rescale(output_size=args.image_size),
                                                                SampleToTensor(),
                                                                Normalization(mean=wae_cfg.mean,
                                                                              std=wae_cfg.std)]))

    dataloader_train = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    decay_mse = args.decay_mse
    lambda_mse = args.lambda_mse
    lr = args.learning_rate
    decay_lr = args.decay_lr

    writer = SummaryWriter(saving_dir + '/runs_' + timestep)
    writer_encoder = SummaryWriter(saving_dir + '/runs_' + timestep + '/reconstruction')
    writer_decoder = SummaryWriter(saving_dir + '/runs_' + timestep + '/penalty')
    writer_discriminator_real = SummaryWriter(saving_dir + '/runs_' + timestep + '/discriminator_real')
    writer_discriminator_fake = SummaryWriter(saving_dir + '/runs_' + timestep + '/discriminator_fake')

    NUM_VOXELS = len(train_data[0]['fmri'])
    logging.info('Number of voxels:', NUM_VOXELS)

    teacher_model = WaeGan(device=device, z_size=args.latent_dim).to(device)
    teacher_model.load_state_dict(torch.load(DECODER_WEIGHTS, map_location=device))
    for param in teacher_model.encoder.parameters():
        param.requires_grad = False

    decoder = Decoder(z_size=args.latent_dim, size=256).to(device)
    cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=args.latent_dim).to(device)
    trained_model = WaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=decoder,
                            z_size=args.latent_dim, recon_level=args.recon_level).to(device)

    trained_model.load_state_dict(torch.load(ENCODER_WEIGHTS, map_location=device))
    model = WaeGanCognitive(device=device, encoder=trained_model.encoder, decoder=trained_model.decoder,
                            z_size=args.latent_dim, recon_level=args.recon_level).to(device)
    # Fix encoder weights
    for param in model.encoder.parameters():
        param.requires_grad = False

    if args.pretrained_gan is not None and os.path.exists(pretrained_model_dir.replace(".pth", ".csv")):
        logging.info('Load pretrained model')
        model_dir = pretrained_model_dir.replace(".pth", '_{}.pth'.format(args.load_epoch))
        model.load_state_dict(torch.load(model_dir))
        model.eval()
        results = pd.read_csv(pretrained_model_dir.replace(".pth", ".csv"))
        results = {col_name: list(results[col_name].values) for col_name in results.columns}
        stp = 1 + len(results['epochs'])
        images_dir = os.path.join(saving_dir, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        pcc, ssim, mse, is_mean = evaluate(model, dataloader_valid, norm=True, mean=wae_cfg.mean, std=wae_cfg.std, path=images_dir)
        print("Mean PCC:", pcc)
        print("Mean SSIM:", ssim)
        print("Mean MSE:", mse)
        print("IS mean", is_mean)
    else:
        logging.info('Initialize')
        stp = 1

    results = dict(
        epochs=[],
        loss_reconstruction=[],
        loss_penalty=[],
        loss_discriminator_fake=[],
        loss_discriminator_real=[]
    )

    margin = 0.35
    equilibrium = 0.68
    # mse_lambda = 1.0

    # Optimizers
    optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_decoder = torch.optim.Adam(model.decoder.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    lr_encoder = StepLR(optimizer_encoder, step_size=30, gamma=0.5)
    lr_decoder = StepLR(optimizer_decoder, step_size=30, gamma=0.5)
    lr_discriminator = StepLR(optimizer_discriminator, step_size=30, gamma=0.5)

    # Metrics
    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity(mean=wae_cfg.mean, std=wae_cfg.std)
    mse_loss = nn.MSELoss()

    result_metrics_train = {}
    result_metrics_valid = {}
    metrics_train = {'train_PCC': pearson_correlation, 'train_SSIM': structural_similarity, 'train_MSE': mse_loss}
    metrics_valid = {'valid_PCC': pearson_correlation, 'valid_SSIM': structural_similarity, 'valid_MSE': mse_loss}

    if metrics_valid is not None:
        for key in metrics_valid.keys():
            results.update({key: []})
        for key, value in metrics_valid.items():
            result_metrics_valid.update({key: 0.0})

    if metrics_train is not None:
        for key in metrics_train.keys():
            results.update({key: []})
        for key, value in metrics_train.items():
            result_metrics_train.update({key: 0.0})

    batch_number = len(dataloader_train)
    step_index = 0

    for idx_epoch in range(args.epochs):

        try:
            # for each batch
            for batch_idx, data_batch in enumerate(dataloader_train):

                model.train()
                frozen_params(model.encoder)
                batch_size = len(data_batch)
                model.decoder.zero_grad()
                model.discriminator.zero_grad()

                x_fmri = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
                x_image = Variable(data_batch['image'], requires_grad=False).float().to(device)

                # ----------Train discriminator-------------

                frozen_params(model.decoder)
                free_params(model.discriminator)

                z_fake, var = model.encoder(x_fmri)
                z_real, var = teacher_model.encoder(x_image)
                # z_fake, var = model.encoder(x_fmri)
                # z_fake = Variable(torch.randn_like(z_real) * 0.5).to(device)

                d_real = model.discriminator(z_real)
                d_fake = model.discriminator(z_fake)

                loss_discriminator_fake = - 10 * torch.sum(torch.log(d_fake + 1e-3))
                loss_discriminator_real = - 10 * torch.sum(torch.log(1 - d_real + 1e-3))
                loss_discriminator_fake.backward(retain_graph=True)
                loss_discriminator_real.backward(retain_graph=True)

                # loss_discriminator.backward(retain_graph=True)
                # [p.grad.data.clamp_(-1, 1) for p in model.discriminator.parameters()]
                optimizer_discriminator.step()

                # ----------Train generator----------------

                free_params(model.decoder)
                frozen_params(model.discriminator)

                z_real, var = model.encoder(x_fmri)
                x_recon = model.decoder(z_real)
                d_real = model.discriminator(z_real)

                mse_loss = nn.MSELoss()
                # loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x_image) ** 2, 1))
                loss_reconstruction = mse_loss(x_recon, x_image)
                loss_penalty = - 10 * torch.mean(torch.log(d_real + 1e-3))
                # loss_wae = (loss_reconstruction + loss_penalty) / batch_size

                loss_reconstruction.backward(retain_graph=True)
                # loss_penalty.backward()
                # [p.grad.data.clamp_(-1, 1) for p in model.encoder.parameters()]
                # loss_wae.backward()
                optimizer_decoder.step()

                # register mean values of the losses for logging
                loss_reconstruction_mean = loss_reconstruction.data.cpu().numpy() / batch_size
                loss_penalty_mean = loss_penalty.data.cpu().numpy() / batch_size
                loss_discriminator_fake_mean = loss_discriminator_fake.data.cpu().numpy() / batch_size
                loss_discriminator_real_mean = loss_discriminator_real.data.cpu().numpy() / batch_size

                logging.info(
                    f'Epoch  {idx_epoch} {batch_idx + 1:3.0f} / {100 * (batch_idx + 1) / len(dataloader_train):2.3f}%, '
                    f'---- recon loss: {loss_reconstruction_mean:.5f} ---- | '
                    f'---- penalty loss: {loss_penalty_mean:.5f} ---- | '
                    f'---- discrim fake loss: {loss_discriminator_fake:.5f} ---- | '
                    f'---- discrim real loss: {loss_discriminator_real:.5f}')

                writer_encoder.add_scalar('loss_reconstruction_batch', loss_reconstruction_mean, step_index)
                writer_decoder.add_scalar('loss_penalty_batch', loss_penalty_mean, step_index)
                writer_discriminator_fake.add_scalar('loss_discriminator_batch', loss_discriminator_fake_mean, step_index)
                writer_discriminator_real.add_scalar('loss_discriminator_batch', loss_discriminator_real_mean, step_index)

                step_index += 1

            # EPOCH END
            lr_encoder.step()
            lr_decoder.step()
            lr_discriminator.step()

            writer_encoder.add_scalar('loss_reconstruction', loss_reconstruction_mean, idx_epoch)
            writer_decoder.add_scalar('loss_penalty', loss_penalty_mean, idx_epoch)
            writer_discriminator_fake.add_scalar('loss_discriminator', loss_discriminator_fake_mean, idx_epoch)
            writer_discriminator_real.add_scalar('loss_discriminator', loss_discriminator_real_mean, idx_epoch)

            if not idx_epoch % 2:

                images_dir = os.path.join(saving_dir, 'images', 'train')
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(x_image[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                plt.savefig(gt_dir)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(x_recon[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                plt.savefig(output_dir)

            logging.info('Evaluation')

            for batch_idx, data_batch in enumerate(dataloader_valid):

                model.eval()
                data_in = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
                data_target = Variable(data_batch['image'], requires_grad=False).float().to(device)
                out = model(data_in)

                # Validation metrics for the first validation batch
                if metrics_valid is not None:
                    for key, metric in metrics_valid.items():
                        if key == 'cosine_similarity':
                            result_metrics_valid[key] = metric(out, data_target).mean()
                        else:
                            result_metrics_valid[key] = metric(out, data_target)

                # Training metrics for the last training batch
                if metrics_train is not None:
                    for key, metric in metrics_train.items():
                        if key == 'cosine_similarity':
                            result_metrics_train[key] = metric(x_recon, x_image).mean()
                        else:
                            result_metrics_train[key] = metric(x_recon, x_image)

                images_dir = os.path.join(saving_dir, 'images', 'valid')
                image_name = os.path.join(images_dir, '{}.jpg'.format(idx_epoch))
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)

                out = out.data.cpu()

                if idx_epoch == 0:
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(make_grid(data_target[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                    gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + str(batch_idx))
                    plt.savefig(gt_dir)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + str(batch_idx))
                plt.savefig(output_dir)

                # out = (out + 1) / 2
                out = make_grid(out, nrow=8, normalize=True)
                writer.add_image("reconstructed", out, step_index)

                z = model.encoder(data_in)
                z_p = Variable(torch.randn_like(z[0]) * 0.5).to(device)
                out = model.decoder(z_p)
                out = make_grid(out, nrow=8, normalize=True)
                writer.add_image("generated", out, step_index)

                out = data_target.data.cpu()
                # out = (out + 1) / 2
                out = make_grid(out, nrow=8, normalize=True)
                writer.add_image("original", out, step_index)

                if metrics_valid is not None:
                    for key, values in result_metrics_valid.items():
                        result_metrics_valid[key] = torch.mean(values)
                    # Logging metrics
                    if writer is not None:
                        for key, values in result_metrics_valid.items():
                            writer.add_scalar(key, values, stp + idx_epoch)

                if metrics_train is not None:
                    for key, values in result_metrics_train.items():
                        result_metrics_train[key] = torch.mean(values)
                    # Logging metrics
                    if writer is not None:
                        for key, values in result_metrics_train.items():
                            writer.add_scalar(key, values, stp + idx_epoch)

                logging.info(
                    f'Epoch  {idx_epoch} ---- train PCC:  {result_metrics_train["train_PCC"].item():.5f} ---- | '
                    f'---- train SSIM: {result_metrics_train["train_SSIM"].item():.5f} ---- '
                    f'---- train MSE: {result_metrics_train["train_MSE"].item():.5f} ---- ')

                logging.info(
                    f'Epoch  {idx_epoch} ---- valid PCC:  {result_metrics_valid["valid_PCC"].item():.5f} ---- | '
                    f'---- valid SSIM: {result_metrics_valid["valid_SSIM"].item():.5f} ---- '
                    f'---- valid MSE: {result_metrics_valid["valid_MSE"].item():.5f} ---- ')

                break

            if not idx_epoch % 10 and not DEBUG:
                torch.save(model.state_dict(), saving_name.replace('.pth', '_' + str(idx_epoch) + '.pth'))
                logging.info('Saving model')

                # Record losses & scores
            results['epochs'].append(idx_epoch + stp)
            results['loss_reconstruction'].append(loss_reconstruction_mean)
            results['loss_penalty'].append(loss_penalty_mean)
            results['loss_discriminator_fake'].append(loss_discriminator_fake_mean)
            results['loss_discriminator_real'].append(loss_discriminator_real_mean)

            if metrics_valid is not None:
                for key, value in result_metrics_valid.items():
                    metric_value = torch.tensor(value, dtype=torch.float64).item()
                    results[key].append(metric_value)

            if metrics_train is not None:
                for key, value in result_metrics_train.items():
                    metric_value = torch.tensor(value, dtype=torch.float64).item()
                    results[key].append(metric_value)

            results_to_save = pd.DataFrame(results)
            results_to_save.to_csv(saving_name.replace(".pth", ".csv"), index=False)

        except KeyboardInterrupt as e:
             logging.info(e, 'Saving plots')

        finally:

            plt.figure(figsize=(10, 5))
            plt.title("Fake/Real Discriminator Loss During Training")
            plt.plot(results['loss_discriminator_fake'], label="F")
            plt.plot(results['loss_discriminator_real'], label="R")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plots_dir = os.path.join(saving_dir, 'plots')
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            plot_dir = os.path.join(plots_dir, 'GD_loss')
            plt.savefig(plot_dir)

            plt.figure(figsize=(10, 5))
            plt.title("Penalty and Reconstruction Loss During Training")
            plt.plot(results['loss_penalty'], label="Penalty")
            plt.plot(results['loss_reconstruction'], label="Reconstruction")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plots_dir = os.path.join(saving_dir, 'plots')
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            plot_dir = os.path.join(plots_dir, 'ER_loss')
            plt.savefig(plot_dir)
            logging.info("Plots are saved")

    exit(0)
