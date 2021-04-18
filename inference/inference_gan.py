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

from torch import no_grad, nn
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import configs.inference_config as inf_cfg
import configs.data_config as data_cfg
from models.vae_gan import CognitiveEncoder, Encoder, Decoder, VaeGanCognitive, VaeGan, Discriminator, WaeGan, \
    WaeGanCognitive
from train.train_utils import evaluate, objective_assessment, PearsonCorrelation, StructuralSimilarity
from data_preprocessing.data_loader import BoldRoiDataloader, Rescale, CenterCrop, SampleToTensor, RandomShift, \
    Normalization, split_subject_data, CocoDataloader, GreyToColor

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

DEBUG = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="user path where the datasets are located", type=str)
    parser.add_argument('--output', '-o', help='user path where to save', type=str)
    parser.add_argument('--logs', '-l', help='path where to save logs', type=str)
    parser.add_argument('--batch_size', '-b', default=inf_cfg.batch_size, help='batch size for dataloader', type=int)
    parser.add_argument('--image_crop', '-im_crop', default=inf_cfg.image_crop, help='size to which image should '
                                                                                        'be cropped', type=int)
    parser.add_argument('--image_size', '-im_size', default=inf_cfg.image_size, help='size to which image should '
                                                                                        'be scaled', type=int)
    parser.add_argument('--device', '-d', default=inf_cfg.device, help='what device to use', type=str)
    parser.add_argument('--num_workers', '-nw', default=inf_cfg.num_workers, help='number of workers for dataloader',
                        type=int)
    parser.add_argument('--latent_dim', '-lat_dim', default=inf_cfg.latent_dim, help='dimension of the latent space',
                        type=int)
    parser.add_argument('--message', '-m', default='default message', help='experiment description', type=str)
    parser.add_argument('--pretrained_gan', '-pretrain', default=inf_cfg.pretrained_gan, help='pretrained gan',
                        type=str)
    parser.add_argument('-load_epoch', '-pretrain_epoch', default=inf_cfg.load_epoch,
                        help='epoch of the pretrained model', type=int)
    parser.add_argument('--recon_level', default=inf_cfg.recon_level, type=int,
                        help='reconstruction level in the descriminator')
    parser.add_argument('--dataset', default=inf_cfg.dataset, help='dataset type, "coco", "bold" ', type=str)
    parser.add_argument('--mode', default=inf_cfg.mode, help='model type', type=str)

    args = parser.parse_args()

    # Path to pickle file with bold5000 data
    USER_ROOT = args.output
    DATA_PATH = os.path.join(args.input, data_cfg.data_root)
    SAVE_PATH = os.path.join(USER_ROOT, data_cfg.save_training_results)
    TRAIN_DATA_PATH = os.path.join(USER_ROOT, data_cfg.data_root, inf_cfg.train_data)
    VALID_DATA_PATH = os.path.join(USER_ROOT, data_cfg.data_root, inf_cfg.valid_data)

    COCO_TEST_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_test_data)
    COCO_TRAIN_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_train_data)
    COCO_VALID_DATA = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.coco_valid_data)

    # Split with fixed stimuli IDs
    TRAIN_STIMULI = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.train_stimuli_split)
    VALID_STIMULI = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.valid_stimuli_split)

    # Create directory to save weights
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Info logging
    timestep = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.logs, 'inference' + timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Check available gpu
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device2 = inf_cfg.device2
    device3 = inf_cfg.device3
    logger.info("Used device: %s" % device)

    logging.info('set up random seeds')
    torch.manual_seed(12345)

    # Create directory for results
    if DEBUG:
        saving_dir = os.path.join(SAVE_PATH, 'debug', 'inference_{}_{}'.format(args.pretrained_gan, timestep))
    else:
        if inf_cfg.save_to_folder is not None:
            saving_dir = os.path.join(SAVE_PATH, 'inference', inf_cfg.save_to_folder,
                                      'inference_{}_{}'.format(args.pretrained_gan, timestep))
        else:
            saving_dir = os.path.join(SAVE_PATH, 'inference', 'inference_{}_{}'.format(args.pretrained_gan, timestep))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    if args.pretrained_gan is not None:
        pretrained_model_dir = os.path.join(SAVE_PATH, inf_cfg.folder_name, args.pretrained_gan, args.pretrained_gan + '.pth')
    saving_name = os.path.join(saving_dir, 'inference.pth'.format(args.pretrained_gan))

    if inf_cfg.file_to_save is not None:
        saving_file = os.path.join(SAVE_PATH, 'inference', inf_cfg.file_to_save)

    # Save arguments
    with open(os.path.join(saving_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load data which were concatenated for 4 subjects and split into training and validation sets
    with open(TRAIN_DATA_PATH, "rb") as input_file:
        train_data = pickle.load(input_file)
    with open(VALID_DATA_PATH, "rb") as input_file:
        valid_data = pickle.load(input_file)

    # Split and save data if train only for one subject
    # train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=12345)
    # print(len(train_data))
    # print(len(valid_data))
    # with open(os.path.join(saving_dir, 'train_data'), 'wb') as f:
    #     pickle.dump(train_data, f)
    # with open(os.path.join(saving_dir, 'valid_data'), 'wb') as f:
    #     pickle.dump(valid_data, f)

    if args.dataset == 'bold':

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
                                                        Normalization(mean=inf_cfg.mean,
                                                                      std=inf_cfg.std)]))

        validation_data = BoldRoiDataloader(dataset=valid_data,
                                            root_path=DATA_PATH,
                                            transform=transforms.Compose([
                                                        CenterCrop(output_size=args.image_crop),
                                                        Rescale(output_size=args.image_size),
                                                        SampleToTensor(),
                                                        Normalization(mean=inf_cfg.mean,
                                                                      std=inf_cfg.std)
                                                        ]))

    if args.dataset == 'coco':
        # Load data
        training_data = CocoDataloader(COCO_TRAIN_DATA, pickle=False,
                                       transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                            args.image_crop)),
                                                                    transforms.Resize((args.image_size,
                                                                                       args.image_size)),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    GreyToColor(args.image_size),
                                                                    transforms.Normalize(inf_cfg.mean,
                                                                    inf_cfg.std)]))

        validation_data = CocoDataloader(COCO_VALID_DATA, pickle=False,
                                         transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                              args.image_crop)),
                                                                       transforms.Resize((args.image_size,
                                                                                          args.image_size)),
                                                                       transforms.ToTensor(),
                                                                       GreyToColor(args.image_size),
                                                                       transforms.Normalize(inf_cfg.mean,
                                                                                            inf_cfg.std)]))

        test_data = CocoDataloader(COCO_TEST_DATA, pickle=False,
                                   transform=transforms.Compose([transforms.CenterCrop((args.image_crop,
                                                                                        args.image_crop)),
                                                                transforms.Resize((args.image_size,
                                                                                   args.image_size)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                GreyToColor(args.image_size),
                                                                transforms.Normalize(inf_cfg.mean,
                                                                                     inf_cfg.std)]))

        training_data = ConcatDataset([training_data, test_data])

    dataloader_train = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    writer = SummaryWriter(saving_dir + '/runs_' + timestep)
    writer_encoder = SummaryWriter(saving_dir + '/runs_' + timestep + '/encoder')
    writer_decoder = SummaryWriter(saving_dir + '/runs_' + timestep + '/decoder')
    writer_discriminator = SummaryWriter(saving_dir + '/runs_' + timestep + '/discriminator')

    NUM_VOXELS = len(train_data[0]['fmri'])
    logging.info('Number of voxels:', NUM_VOXELS)
    logging.info('Training data length:', len(train_data))
    logging.info('Validation data length:', len(valid_data))

    # Define model
    if args.dataset == 'coco':

        if args.mode == 'vae-gan':
            model = VaeGan(device=device, z_size=args.latent_dim, recon_level=args.recon_level).to(device)
        elif args.mode == 'wae-gan':
            model = WaeGan(device=device, z_size=args.latent_dim).to(device)

    elif args.dataset == 'bold':

        if args.mode == 'vae-gan':

            # Teacher model from stage I
            encoder = Encoder(z_size=args.latent_dim).to(device)
            decoder = Decoder(z_size=args.latent_dim, size=encoder.size).to(device)
            teacher_model = VaeGan(device=device, z_size=args.latent_dim).to(device)
            for param in teacher_model.parameters():
                param.requires_grad = False

            # Define model for stage II
            cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=args.latent_dim).to(device)
            decoder = Decoder(z_size=args.latent_dim, size=256).to(device)
            discriminator = Discriminator().to(device)
            model = VaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=decoder,
                                    discriminator=discriminator, teacher_net=teacher_model,
                                    z_size=args.latent_dim, stage=3).to(device)

        elif args.mode == 'vae':

            # VAE
            cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=args.latent_dim).to(device)
            decoder = Decoder(z_size=args.latent_dim, size=256).to(device)
            discriminator = Discriminator().to(device)
            model = VaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=decoder,
                                    discriminator=discriminator, teacher_net=None,
                                    z_size=args.latent_dim, stage=3).to(device)

        elif args.mode == 'wae-gan':

            # WAE
            decoder = Decoder(z_size=args.latent_dim, size=256).to(device)
            cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=args.latent_dim).to(device)
            trained_model = WaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=decoder,
                                    z_size=args.latent_dim, recon_level=args.recon_level).to(device)

            model = WaeGanCognitive(device=device, encoder=trained_model.encoder, decoder=trained_model.decoder,
                                    z_size=args.latent_dim, recon_level=args.recon_level).to(device)

    if args.pretrained_gan is not None and os.path.exists(pretrained_model_dir.replace(".pth", ".csv")):
        logging.info('Load pretrained model')

        model_dir = pretrained_model_dir.replace(".pth", '_{}.pth'.format(args.load_epoch))
        model.load_state_dict(torch.load(model_dir))
        model.eval()
        results = pd.read_csv(pretrained_model_dir.replace(".pth", ".csv"))
        results = {col_name: list(results[col_name].values) for col_name in results.columns}
        stp = 1 + len(results['epochs'])
        if inf_cfg.evaluate:
            images_dir = os.path.join(saving_dir, 'images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            pcc, ssim, mse, is_score = evaluate(model, dataloader_valid, norm=False, mean=inf_cfg.mean, std=inf_cfg.std,
                                                path=images_dir, dataset=args.dataset, mode=args.mode, save=inf_cfg.save,
                                                resize=200)
            print("Mean PCC:", pcc)
            print("Mean SSIM:", ssim)
            print("Mean MSE:", mse)
            print("Mean IS:", is_score)

            # Plot histogram for objective assessment
            obj_score = dict(pcc=[], ssim=[])
            for top in [2, 5, 10]:
                obj_pcc, obj_ssim = objective_assessment(model, dataloader_valid, dataset=args.dataset, mode=args.mode, top=top)
                obj_score['pcc'].append(obj_pcc)
                obj_score['ssim'].append(obj_ssim)

            obj_results_to_save = pd.DataFrame(obj_score)
            results_to_save = pd.DataFrame(obj_results_to_save)
            if inf_cfg.file_to_save is not None:
                results_to_save.to_csv(saving_file, index=False)

            x_axis = ['2-way', '5-way', '10-way']
            y_axis = [obj_score['pcc'][0], obj_score['pcc'][1], obj_score['pcc'][2]]
            bars = plt.bar(x_axis, y_axis, width=0.5)
            plt.axhline(y=0.5, xmin=0, xmax=0.33, linewidth=1, color='k')
            plt.axhline(y=0.2, xmin=0.33, xmax=0.66, linewidth=1, color='k')
            plt.axhline(y=0.1, xmin=0.66, xmax=1.0, linewidth=1, color='k')
            for i, bar in enumerate(bars):
                yval = bar.get_height()
                plt.text(bar.get_x() + 0.10, yval + .005, f'{y_axis[i].item() * 100:.2f}')
            plt.ylabel('Pixel correlation (%)')
            plt.title('Objective assessment')
            plt.show()
            print("Objective score PCC", obj_score['pcc'][0])
            print("Objective score SSIM", obj_score['ssim'][0])
            exit(0)
    else:
        logging.info('Initialize')
        stp = 1

    results = dict()

    # Metrics
    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity(mean=inf_cfg.mean, std=inf_cfg.std)
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

    # for each batch
    for batch_idx, data_batch in enumerate(dataloader_train):

        model.eval()

        try:
            x_tilde = model(data_batch)
        except TypeError as e:
            if args.mode == 'wae-gan':
                x_tilde = model(data_batch['fmri'])
            else:
                logging.info('Wrong data type')

        if args.dataset == 'bold':
            batch_size = len(data_batch['image'])
            x_gt = data_batch['image'].to(device)
        elif args.dataset == 'coco':
            batch_size = len(data_batch)
            x_gt = data_batch.to(device)

        break

    logging.info('Evaluation')

    for batch_idx, data_batch in enumerate(dataloader_valid):

        model.eval()

        with no_grad():

            try:
                out = model(data_batch)
            except TypeError as e:
                if args.mode == 'wae-gan':
                    out = model(data_batch['fmri'])
                else:
                    logging.info('Wrong data type')

            if args.dataset == 'bold':
                data_target = Variable(data_batch['image'], requires_grad=False).float().to(device)

            elif args.dataset == 'coco':
                data_target = data_batch.to(device)

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

            if inf_cfg.save:

                images_dir = os.path.join(saving_dir, 'images', 'valid')
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(data_target[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                gt_dir = os.path.join(images_dir, 'batch_' + str(batch_idx) + '_ground_truth_' + 'grid')
                plt.savefig(gt_dir)

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                output_dir = os.path.join(images_dir, 'batch_' + str(batch_idx) + '_output_' + 'grid')
                plt.savefig(output_dir)

            logging.info(
                f'---- train PCC:  {result_metrics_train["train_PCC"].item():.5f} ---- | '
                f'---- train SSIM: {result_metrics_train["train_SSIM"].item():.5f} ---- '
                f'---- train MSE: {result_metrics_train["train_MSE"].item():.5f} ---- ')

            logging.info(
                f'---- valid PCC:  {result_metrics_valid["valid_PCC"].item():.5f} ---- | '
                f'---- valid SSIM: {result_metrics_valid["valid_SSIM"].item():.5f} ---- '
                f'---- valid MSE: {result_metrics_valid["valid_MSE"].item():.5f} ---- ')

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


