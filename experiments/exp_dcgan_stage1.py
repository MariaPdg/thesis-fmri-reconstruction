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

from torch import nn, no_grad
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

import experiments.exp_config as exp_cfg
import configs.data_config as data_cfg
from models.vae_gan import CognitiveEncoder, Encoder, Decoder, VaeGanCognitive, VaeGan, Discriminator, DCGan
from train.train_utils import evaluate, objective_assessment, PearsonCorrelation, StructuralSimilarity
from data_preprocessing.data_loader import CocoDataloader, GreyToColor

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

DEBUG = False


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', help='user path where to save', type=str)
    parser.add_argument('--logs', '-l', help='path where to save logs', type=str)
    parser.add_argument('--batch_size', '-b', default=exp_cfg.batch_size, help='batch size for dataloader', type=int)
    parser.add_argument('--learning_rate', '-lr', default=exp_cfg.learning_rate, help='learning rate', type=float)
    parser.add_argument('--epochs', '-e', default=exp_cfg.n_epochs, help='number of epochs', type=int)
    parser.add_argument('--image_crop', '-im_crop', default=exp_cfg.image_crop, help='size to which image should '
                                                                                        'be cropped', type=int)
    parser.add_argument('--image_size', '-im_size', default=exp_cfg.image_size, help='size to which image should '
                                                                                        'be scaled', type=int)
    parser.add_argument('--device', '-d', default=exp_cfg.device, help='what device to use', type=str)
    parser.add_argument('--num_workers', '-nw', default=exp_cfg.num_workers, help='number of workers for dataloader',
                        type=int)
    parser.add_argument('--step_size', '-step', default=exp_cfg.step_size, help='number of epochs after which '
                                                                                   'to decrease learning rate',
                        type=int)
    parser.add_argument('--patience', '-p', default=exp_cfg.patience, help='number of epochs with unchanged lr '
                                                                              'for early stopping', type=int)
    parser.add_argument('--weight_decay', '--wd', default=exp_cfg.weight_decay,
                        help='weight decay used by optimizer', type=float)
    parser.add_argument('--latent_dim', '-lat_dim', default=exp_cfg.latent_dim, help='dimension of the latent space',
                        type=int)
    parser.add_argument('--message', '-m', default='default message', help='experiment description', type=str)
    parser.add_argument('--pretrained_gan', '-pretrain', default=exp_cfg.pretrained_gan, help='pretrained gan',
                        type=str)
    parser.add_argument('-load_epoch', '-pretrain_epoch', default=exp_cfg.load_epoch,
                        help='epoch of the pretrained model', type=int)
    parser.add_argument('--recon_level', default=exp_cfg.recon_level, type=int,
                        help='reconstruction level in the descriminator')
    parser.add_argument('--lambda_mse', default=exp_cfg.lambda_mse, type=float, help='weight for style error')
    parser.add_argument('--decay_mse', default=exp_cfg.decay_mse, type=float, help='mse weight decrease')
    parser.add_argument('--decay_lr', default=exp_cfg.decay_lr, type=float,
                        help='learning rate decay for lr scheduler')
    parser.add_argument('--margin', default=exp_cfg.margin, type=float, help='margin for generator/discriminator game')
    parser.add_argument('--equilibrium', default=exp_cfg.equilibrium, type=float,
                        help='equilibrium for the generator/discriminator game')
    parser.add_argument('--decay_margin', default=exp_cfg.decay_margin, type=float,
                        help='margin decay for the generator/discriminator game')
    parser.add_argument('--decay_equilibrium', default=exp_cfg.decay_equilibrium, type=float,
                        help='equilibrium decay for the generator/discriminator game')
    parser.add_argument('--decoder', '-dec', default=exp_cfg.decoder_weights,
                        help='pretrained vae-gan model to fix decoder weights, training stage II', type=str)
    parser.add_argument('--cog_encoder', '-enc', default=exp_cfg.cog_encoder_weights,
                        help='pretrained vae-gan-cog model to fix cognitive encoder weights , training stage III', type=str)
    args = parser.parse_args()

    # Path to pickle file with bold5000 data
    USER_ROOT = args.output
    DATA_PATH = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.bold_roi_data)
    SAVE_PATH = os.path.join(USER_ROOT, data_cfg.save_training_results)
    TRAIN_DATA_PATH = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.train_data)
    VALID_DATA_PATH = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.valid_data)
    DECODER_WEIGHTS = os.path.join(USER_ROOT, data_cfg.save_training_results, 'gan', args.decoder[0], args.decoder[0] +
                                   '_' + str(args.decoder[1]) + '.pth')
    COCO_TEST_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_test_data)
    COCO_TRAIN_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_train_data)
    COCO_VALID_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_valid_data)
    # Create directory to save weights
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Info logging
    timestep = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.logs, 'exp_dcgan_stage1' + timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Check available gpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device2 = exp_cfg.device2
    device3 = exp_cfg.device3
    logger.info("Used device: %s" % device)

    logging.info('set up random seeds')
    torch.manual_seed(12345)

    # Create directory for results
    if DEBUG:
        saving_dir = os.path.join(SAVE_PATH, 'debug', 'debug_exp_dcgan_{}'.format(timestep))
    else:
        saving_dir = os.path.join(SAVE_PATH, 'experiments', 'exp_dcgan_{}'.format(timestep))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    if args.pretrained_gan is not None:
        pretrained_model_dir = os.path.join(SAVE_PATH, 'experiments', args.pretrained_gan, args.pretrained_gan + '.pth')
    saving_name = os.path.join(saving_dir, 'exp_dcgan_{}.pth'.format(timestep))

    # Save arguments
    with open(os.path.join(saving_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load data which were concatenated for 4 subjects and split into training and validation sets
    with open(TRAIN_DATA_PATH, "rb") as input_file:
        train_data = pickle.load(input_file)
    with open(VALID_DATA_PATH, "rb") as input_file:
        valid_data = pickle.load(input_file)

    # Load data
    # Load data
    training_data = CocoDataloader(COCO_TRAIN_DATA, pickle=False,
                                   transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                        args.image_crop)),
                                                                 transforms.Resize((args.image_size,
                                                                                    args.image_size)),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor(),
                                                                 GreyToColor(args.image_size),
                                                                 transforms.Normalize(exp_cfg.mean,
                                                                                      exp_cfg.std)
                                                                 ]))
    validation_data = CocoDataloader(COCO_VALID_DATA, pickle=False,
                                     transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                          args.image_crop)),
                                                                   transforms.Resize((args.image_size,
                                                                                      args.image_size)),
                                                                   transforms.ToTensor(),
                                                                   GreyToColor(args.image_size),
                                                                   transforms.Normalize(exp_cfg.mean,
                                                                                        exp_cfg.std)
                                                                   ]))
    test_data = CocoDataloader(COCO_TEST_DATA, pickle=False,
                               transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                    args.image_crop)),
                                                             transforms.Resize((args.image_size,
                                                                                args.image_size)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             GreyToColor(args.image_size),
                                                             transforms.Normalize(exp_cfg.mean,
                                                                                  exp_cfg.std)
                                                             ]))
    train_test_data = ConcatDataset([training_data, test_data])

    dataloader_train = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    writer = SummaryWriter(saving_dir + '/runs_' + timestep)
    writer_encoder = SummaryWriter(saving_dir + '/runs_' + timestep + '/encoder')
    writer_decoder = SummaryWriter(saving_dir + '/runs_' + timestep + '/decoder')
    writer_discriminator = SummaryWriter(saving_dir + '/runs_' + timestep + '/discriminator')

    NUM_VOXELS = len(train_data[0]['fmri'])
    logging.info('Number of voxels:', NUM_VOXELS)
    decoder = Decoder(z_size=args.latent_dim, size=256).to(device)
    discriminator = Discriminator().to(device)
    model = DCGan(device=device, decoder=decoder, discriminator=discriminator,  z_size=args.latent_dim).to(device)

    if args.pretrained_gan is not None and os.path.exists(pretrained_model_dir.replace(".pth", ".csv")):
        logging.info('Load pretrained model')
        model_dir = pretrained_model_dir.replace(".pth", '_{}.pth'.format(args.load_epoch))
        model.load_state_dict(torch.load(model_dir))
        model.eval()
        results = pd.read_csv(pretrained_model_dir.replace(".pth", ".csv"))
        results = {col_name: list(results[col_name].values) for col_name in results.columns}
        stp = 1 + len(results['epochs'])
        if exp_cfg.evaluate:
            images_dir = os.path.join(saving_dir, 'images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            pcc, ssim, mse, _ = evaluate(model, dataloader_train, norm=True, mean=exp_cfg.mean, std=exp_cfg.std, path=images_dir)
            print("Mean PCC:", pcc)
            print("Mean SSIM:", ssim)
            print("Mean MSE:", mse)
            obj_score = objective_assessment(model, dataloader_valid, dataset='bold')
            print("Objective score PCC", obj_score[0])
            print("Objective score SSIM", obj_score[1])
            exit(0)
    else:
        logging.info('Initialize')
        stp = 1

    results = dict(
        epochs=[],
        loss_decoder=[],
        loss_discriminator=[]
    )

    margin = args.margin
    equilibrium = args.equilibrium
    lambda_mse = args.lambda_mse

    # Optimizers
    optimizer_decoder = torch.optim.RMSprop(params=model.decoder.parameters(), lr=args.learning_rate, alpha=0.9,
                                            eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)
    optimizer_discriminator = torch.optim.RMSprop(params=model.discriminator.parameters(), lr=args.learning_rate,
                                                  alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=args.decay_lr)

    #
    # optimizer_decoder = torch.optim.Adam(params=model.decoder.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    # lr_decoder = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)
    # optimizer_discriminator = torch.optim.Adam(params=model.discriminator.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    # lr_discriminator = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)

    # Metrics
    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity(mean=exp_cfg.mean, std=exp_cfg.std)

    # Metrics
    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity(mean=exp_cfg.mean, std=exp_cfg.std)
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

    mse_loss = nn.MSELoss()

    for idx_epoch in range(args.epochs):

        try:

            # for each batch
            for batch_idx, data_batch in enumerate(dataloader_train):

                model.train()
                batch_size = args.batch_size

                x_gt, x_tilde, disc_class, disc_layer = model(data_batch)

                # split so we can get the different parts
                disc_layer_original = disc_layer[:batch_size]
                disc_layer_predicted = disc_layer[batch_size:-batch_size]
                disc_layer_sampled = disc_layer[-batch_size:]

                disc_class_original = disc_class[:batch_size]
                disc_class_predicted = disc_class[batch_size:-batch_size]
                disc_class_sampled = disc_class[-batch_size:]

                bce_dis_original = -torch.log(disc_class_original + 1e-3)
                bce_dis_predicted = -torch.log(disc_class_predicted + 1e-3)
                bce_dis_sampled = -torch.log(1 - disc_class_sampled + 1e-3)

                loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_sampled)
                loss_decoder = torch.sum(bce_dis_predicted)

                # Register mean values
                loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / batch_size
                loss_decoder_mean = loss_decoder.data.cpu().numpy() / batch_size

                # Selectively disable the decoder of the discriminator if they are unbalanced
                train_dis = True
                train_dec = True

                if torch.mean(bce_dis_original).item() < equilibrium - margin or \
                        torch.mean(bce_dis_predicted).item() < equilibrium - margin:
                    train_dis = False
                if torch.mean(bce_dis_original).item() > equilibrium + margin or \
                        torch.mean(bce_dis_predicted).item() > equilibrium + margin:
                    train_dec = False
                if train_dec is False and train_dis is False:
                    train_dis = True
                    train_dec = True

                # BACKPROP
                # clean grads
                model.zero_grad()

                # Discriminator
                if train_dis:
                    loss_discriminator.backward(retain_graph=True)
                    [p.grad.data.clamp_(-1, 1) for p in model.discriminator.parameters()]
                    optimizer_discriminator.step()
                    model.discriminator.zero_grad()

                # Decoder
                if train_dec:
                    loss_decoder.backward()
                    [p.grad.data.clamp_(-1, 1) for p in model.decoder.parameters()]
                    optimizer_decoder.step()
                    model.decoder.zero_grad()

                logging.info(
                    f'Epoch  {idx_epoch} {batch_idx + 1:3.0f} / {100 * (batch_idx + 1) / len(dataloader_train):2.3f}%, '
                    f'---- decoder loss: {loss_decoder_mean:.5f} ---- | '
                    f'---- discriminator loss: {loss_discriminator_mean:.5f}')

                writer_decoder.add_scalar('loss_decoder_discriminator_batch', loss_decoder_mean, step_index)
                writer_discriminator.add_scalar('loss_decoder_discriminator_batch', loss_discriminator_mean, step_index)

                step_index += 1

            # EPOCH END
            lr_decoder.step()
            lr_discriminator.step()
            margin *= args.decay_margin
            equilibrium *= args.decay_equilibrium

            if margin > equilibrium:
                equilibrium = margin
            lambda_mse *= args.decay_mse
            if lambda_mse > 1:
                lambda_mse = 1

            writer_decoder.add_scalar('loss_decoder_discriminator', loss_decoder_mean, idx_epoch)
            writer_discriminator.add_scalar('loss_decoder_discriminator', loss_discriminator_mean, idx_epoch)

            if not idx_epoch % 2:

                images_dir = os.path.join(saving_dir, 'images', 'train')
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(data_batch[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                plt.savefig(gt_dir)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(x_tilde[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                plt.savefig(output_dir)

            logging.info('Evaluation')

            for batch_idx, data_batch in enumerate(dataloader_valid):
                model.eval()

                with no_grad():

                    data_target = Variable(data_batch, requires_grad=False).float().to(device)
                    out = model(data_batch)

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
                                result_metrics_train[key] = metric(x_tilde, x_gt).mean()
                            else:
                                result_metrics_train[key] = metric(x_tilde, x_gt)

                    out = out.data.cpu()

                    images_dir = os.path.join(saving_dir, 'images')
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)

                    if idx_epoch == 0:
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.imshow(make_grid(data_target[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                        plt.savefig(gt_dir)

                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                    output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                    plt.savefig(output_dir)

                    out = (out + 1) / 2
                    out = make_grid(out, nrow=8)
                    writer.add_image("reconstructed", out, step_index)

                    out = model(None, 100)
                    out = out.data.cpu()
                    out = (out + 1) / 2
                    out = make_grid(out, nrow=8)
                    writer.add_image("generated", out, step_index)

                    out = data_target.data.cpu()
                    out = (out + 1) / 2
                    out = make_grid(out, nrow=8)
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

            results['epochs'].append(idx_epoch + stp)
            results['loss_decoder'].append(loss_decoder_mean)
            results['loss_discriminator'].append(loss_discriminator_mean)

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
             logging.info('Saving plots')

        finally:

            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(results['loss_decoder'], label="G")
            plt.plot(results['loss_discriminator'], label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plots_dir = os.path.join(saving_dir, 'plots')
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            plot_dir = os.path.join(plots_dir, 'GD_loss')
            plt.savefig(plot_dir)
            logging.info("Plots are saved")
            plt.show()

    exit(0)
