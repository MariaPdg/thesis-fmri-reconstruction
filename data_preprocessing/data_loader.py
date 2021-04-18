import os
import time
import glob
import torch
import pickle
import logging
import argparse
import numpy as np
import nibabel as nib
import matplotlib.image as mpimg
import torch.multiprocessing as mp

import configs.data_config as data_cfg

from PIL import Image
from os import listdir
from skimage import transform
from scipy.ndimage import shift
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# multiprocessing mode
mp.set_start_method('spawn', force=True)


class Bold5000Dataloader(object):

    def __init__(self, data_ref, transform=None):
        """
        The constructor to initialized paths to images and fmri data
        :param data_ref: aggregated data with all data relevant to fMRI study
        :param transform:
        """
        # paths to unprocessed fmri data
        self.fmri_path = data_ref['fmri_path']
        # paths to corresponding stimuli
        self.stimuli_path = data_ref['stimuli_path']
        # the trial number for each stimuli, to extract relevant voxel information (in getitem)
        self.trials = [(trial - 1) * 5 for trial in data_ref['trial']]
        self.transform = transform

    def __len__(self):
        return len(self.fmri_path)

    def __getitem__(self, idx):

        """
        This function takes the mean value of relevant peaked frames for each fMRI image (4-8 sec)

        For each run, each session and each participant:
        - At the beginning and end of each run, centered
        on a blank black screen, a fixation cross was shown for 6 sec and 12 sec, respectively.
        - Following the initial fixation cross, all 37 stimuli were shown sequentially.
        - Each image was presented for 1 sec followed by a 9 sec fixation cross.
        Total:
        37 *10 = 370 sec of stimulus presentation
        370 + 6 + 12 = 388 sec (6 min 28 sec) of data acquired in each run.
        TR = 2 s slice thickness
        TE = 30 ms

        see paper N. Chang et al. 2019 - BOLD5000, a public fMRI dataset while viewing 5000 visual images
        https://www.nature.com/articles/s41597-019-0052-3

        :param idx: sample index
        :return: sample: dictionary {'fmri', 'image'}
            'fmri': torch.tensor [batch_size, depth, hight, width],
                depth = 69, hight = width = 106
            'image': torch.tensor [batch_size, depth, hight, width ],
                depth = 3, hight = wigth = 375
        """

        fmri = nib.load(self.fmri_path[idx]).get_data()
        stimulus = mpimg.imread(self.stimuli_path[idx])
        # For each image: 10/2 = 5 frames, take the average of frames 2-3
        voxels = np.mean(fmri[..., self.trials[idx] + 2: self.trials[idx] + 4], axis=3)
        voxels = voxels.transpose((2, 0, 1))
        pixels = stimulus.transpose((2, 0, 1))

        sample = {'fmri': torch.FloatTensor(voxels),
                  'image': torch.FloatTensor(pixels)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalization(object):
    """
    Stimuli normalization (fmri are normalized during concatenation) with mean and std per channel,
    apply to tensors

    :return: transformed_sample {'fmri', 'image'}
        Sample with normalized fmri and images
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.mean = mean
        self.std = std

    def __call__(self, sample):

        image, fmri = sample['image'], sample['fmri']
        for i in range(3):
            image[i] = (image[i] - self.mean[i]) / self.std[i]
        norm_img = image

        transformed_sample = {'fmri': fmri, 'image': norm_img}

        return transformed_sample


class Rescale(object):
    """
    Rescale images for pretrained model (375x375 -> 224x224)
    """
    def __init__(self, output_size=224):
        """
        @param output_size: image output size
        """
        self.output_size = output_size

    def __call__(self, sample):
        """
        @return: transformed_sample: {'fmri', 'image'}
        Sample with rescaled images
        """

        image, fmri = sample['image'], sample['fmri']
        scaled_img = transform.resize(image, (self.output_size, self.output_size))
        transformed_sample = {'fmri': fmri, 'image': scaled_img}

        return transformed_sample


class CenterCrop(object):
    """
    Center crop to the output size
    """

    def __init__(self, output_size=374):
        """
        @param output_size: image output size
        """
        if type(output_size) == 'int':
            self.cropx = output_size[0]
            self.cropy = output_size[1]
        else:
            self.cropx = self.cropy = output_size

    def __call__(self, sample):
        """
        @return: transformed_sample: {'fmri', 'image'}
        Sample with rescaled images
        """

        image, fmri = sample['image'], sample['fmri']

        _, y, x = image.shape
        startx = x // 2 - (self.cropx // 2)
        starty = y // 2 - (self.cropy // 2)
        cropped_img = image[:, starty:starty + self.cropy, startx:startx + self.cropx]

        transformed_sample = {'fmri': fmri, 'image': cropped_img}

        return transformed_sample


class SampleToTensor(object):

    """
    Transforms numpy or PIl image to tensor
    """

    def __call__(self, sample):

        image, fmri = sample['image'], sample['fmri']
        image_array = image.transpose(2, 0, 1)
        image_tensor = torch.FloatTensor(image_array)
        fmri_tensor = torch.FloatTensor(sample['fmri'])

        sample_tensor = {'fmri': fmri_tensor, 'image': image_tensor}

        return sample_tensor


class RandomShift(object):
    """
    Randomly shifts image with max_shift
    """

    def __init__(self, max_shift=5):

        self.max_shift = max_shift

    def __call__(self, sample):

        image, fmri = sample['image'], sample['fmri']
        image_shifted = rand_shift(image, self.max_shift)

        transformed_sample = {'fmri': fmri, 'image': image_shifted}

        return transformed_sample


def rand_shift(img, max_shift=0):
    """
    Shifts images randomly
    @param img: np.array [size x size x channels]
        Image to be shifted
    @param max_shift: shift value

    @return: image shifted along x and y directions
    """
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    img_shifted = shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode='nearest')
    return img_shifted


class BoldRoiDataloader(object):
    """
    Dataloader for ROI information in BOLD5000 dataset
    https://ndownloader.figshare.com/files/12965447
    """

    def __init__(self, dataset, root_path=None, transform=None):
        """
        The constructor to initialized paths to images and fmri data
        :param data_dir: directories to fmri and image data
        :param transform: list of transformations to be applied
        """
        self.dataset = dataset
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        voxels = self.dataset[idx]['fmri']

        # TODO: here we replace stimuli paths with your own with root_path
        # It's supposed that fMRI data are in the folder "/BOLD5000/"
        if self.root_path is not None and self.root_path not in self.dataset[idx]['image']:
            name = self.dataset[idx]['image'].split('BOLD5000')[0]
            self.dataset[idx]['image'] = self.dataset[idx]['image'].replace(name, self.root_path)

        stimulus = mpimg.imread(self.dataset[idx]['image'])

        sample = {'fmri': voxels, 'image': stimulus}

        if self.transform:
            sample = self.transform(sample)

        return sample


def concatenate_bold_data(data_dir):
    """
    Concatenates subject data in the whole dataset and normalizes them on the fly
    (stored with roi_extraction.py script)

    :param data_dir: path to data saved per subjects, directory must look like:
        data_dir/CSI1/CSI1_roi_pad.pickle
        data_dir/CSI1/CSI1_stimuli_paths.pickle
        data_dir/CSI2/CSI2_roi_pad.pickle
        data_dir/CSI2/CSI2_stimuli_paths.pickle
        data_dir/CSI3/...
        data_dir/CSI4/...

    :return fmri_image_dataset: list [{'fmri: , 'image'}]
        Concatenated fmri data

    """

    subjects = data_cfg.subjects

    all_fmri = []
    all_stimuli = []
    activation_len = []
    for sub in subjects:
        with open(data_dir + sub + '/' + sub +'_roi_pad.pickle', "rb") as input_file:
            fmri = pickle.load(input_file)
            # norm_fmri = preprocessing.scale(np.abs(fmri)) * np.sign(fmri)
            norm_fmri = preprocessing.scale(fmri)
            # norm_fmri = softmax_normalization(fmri)
            all_fmri.append(norm_fmri)
            # all_fmri.append(fmri / np.max(np.abs(fmri)))  # normalization
            # all_fmri.append(linear_normalization(fmri))
            # all_fmri.append(fmri)  # without norm
            activation_len.append(fmri.shape[1])
        with open(data_dir + sub + '/' + sub + '_stimuli_paths.pickle',
                  "rb") as input_file:
            stim_path = pickle.load(input_file)
            all_stimuli.extend(stim_path)

    fmri_dataset = np.concatenate(all_fmri, axis=0)

    # create dataset: fmri + image paths
    fmri_image_dataset = []
    for idx, (vox, pix) in enumerate(zip(fmri_dataset, all_stimuli)):
        fmri_image_dataset.append({'fmri': vox, 'image': pix})

    return fmri_image_dataset


def softmax_normalization(x, lam=2):

    return 1 / (1 + np.exp((x - np.mean(x)) / (lam * np.std(x) / 2 * np.pi)))


def linear_normalization(x):

    # return (x - np.min(x)) / (np.max(x) - np.min(x))
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1


def prepare_external_data(data_dir, pickle_name, save=False):
    """
    Helpful function to prepare coco data as an external dataset
    Need to filter out grey scale images
    Saves the list of paths to rgb images

    @param data_dir: path to coco test dataset
    @param save: path where to save
    """
    image_names = glob.glob(data_dir + '*.jpg')

    name_list = []
    for idx, name in enumerate(image_names):
        img = mpimg.imread(name)
        # shapes of grey scale images without channel
        if len(img.shape) > 2:
            name_list.append(name)
        else:
            print(img.shape)
        if idx % 1000 == 0:
            print(len(name_list))

    if save:
        with open(os.path.join(SAVE_PATH, pickle_name), 'wb') as f:
            pickle.dump(name_list, f)


class CocoDataloader(object):

    def __init__(self, data_dir, transform=None, pickle=True):
        """
        The constructor to initialized paths to coco images
        :param data_dir: directory to coco images
        :param transform: image transformations
        :param pickle: True if names are stored in pickle file (deprecated)
        """
        self.transform = transform
        if not pickle:
            self.image_names = [os.path.join(data_dir, img) for img in listdir(data_dir) if os.path.join(data_dir, img)]
        else:
            self.image_names = data_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image = Image.open(self.image_names[idx])

        if self.transform:
            image = self.transform(image)

        return image


class GreyToColor(object):
    """
    Converts grey tensor images to tensor with 3 channels
    """

    def __init__(self, size):
        """
        @param size: image size
        """
        self.image = torch.zeros([3, size, size])

    def __call__(self, image):
        """
        @param image: image as a torch.tensor
        @return: transformed image if it is grey scale, otherwise original image
        """

        out_image = self.image

        if image.shape[0] == 3:
            out_image = image
        else:
            out_image[0, :, :] = torch.unsqueeze(image, 0)
            out_image[1, :, :] = torch.unsqueeze(image, 0)
            out_image[2, :, :] = torch.unsqueeze(image, 0)

        return out_image


def split_subject_data(dataset, reference):
    """
    Returns only the data with stimuli names specified in reference file, stored with train_test_stimuli_split function

    @param dataset:  data from BoldRoiDataloader
    @param reference: path to pickle file list of stimuli stored with pickle file
    @return: data with stimuli specified in reference file
    """

    stimuli_list = []
    with open(reference, "rb") as input_file:
        file = pickle.load(input_file)
    for item in dataset:
        if item['image'].split('/')[-1] in file:
           stimuli_list.append(item)
           # file.remove(item['image'].split('/')[-1])  # find unique stimuli
    return stimuli_list


class MnistDataloader(object):
    """
    Dataloader for MNIST69 dataset
    Resizes images to np.array of size 28x28x3
    Rotates and horizontally flips images
    """

    def __init__(self, dataset, transform=None):

        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        voxels = self.dataset[idx][28*28:]

        stimulus = self.dataset[idx][:28*28].reshape(28, 28, 1)/255.
        stimulus = np.rot90(stimulus)
        stimulus = np.flip(stimulus, 1)
        tranformed_img = np.zeros([28, 28, 3])
        tranformed_img[:, :, 0] = stimulus[..., 0]
        tranformed_img[:, :, 1] = stimulus[..., 0]
        tranformed_img[:, :, 2] = stimulus[..., 0]

        sample = {'fmri': voxels, 'image': tranformed_img}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="user path where the datasets are located", type=str)
    parser.add_argument('--output', '-o', help="user path where to save", type=str)
    parser.add_argument('--logs', '-l', help="path where to save logs", type=str)
    args = parser.parse_args()

    # Path to pickle file with bold5000 data
    # USER_PATH = sys.argv
    USER_ROOT = args.output
    DATA_ROOT = os.path.join(USER_ROOT, data_cfg.data_root)
    BOLD_PICKLE_PATH = os.path.join(args.output, data_cfg.data_root, data_cfg.bold_pickle_file)
    EXT_DATA_PATH = os.path.join(args.input, data_cfg.data_root, data_cfg.coco_valid_data)
    SAVE_PATH = os.path.join(DATA_ROOT, data_cfg.save_path)
    COCO_PICKLE_PATH = os.path.join(args.output, data_cfg.data_root, data_cfg.external_data_pickle)

    # info logging
    timestep = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.logs, 'bold_loader_' + timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # check available gpu
    if torch.cuda.is_available():
      dev = 'cuda:4'
    else:
      dev = 'cpu'
    device = torch.device(dev)
    logger.info("Used device: %s" % device)

    data_path = os.path.join(DATA_ROOT, 'BOLD5000/bold_roi/')
    # Concatenate data for all subjects
    bold_dataset = concatenate_bold_data(data_path)
    # Split into training and validation sets
    train_data, valid_data = train_test_split(bold_dataset, test_size=0.2, random_state=12345)
    print(len(train_data))
    print(len(valid_data))

    # Save fixed split for all fmri
    # TRAIN_STIMULI = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.train_stimuli_split)
    # VALID_STIMULI = os.path.join(USER_ROOT, data_cfg.data_root, data_cfg.valid_stimuli_split)
    # fixed_train_data = split_subject_data(bold_dataset, TRAIN_STIMULI)
    # with open(os.path.join(SAVE_PATH, 'bold_train', 'bold_train_all_fixed.pickle'), 'wb') as f:
    #     pickle.dump(fixed_train_data, f)

    SAVE = False  # save train and valid data paths
    if SAVE:
        # save data per subject
        with open(os.path.join(SAVE_PATH, 'bold_train', 'bold_' + data_cfg.subjects[0] + '_pad.pickle'), 'wb') as f:
            pickle.dump(bold_dataset, f)

        # save data for all subjects
        # with open(os.path.join(SAVE_PATH, 'bold_train', 'bold_train_norm.pickle'), 'wb') as f:
        #     pickle.dump(train_data, f)
        # with open(os.path.join(SAVE_PATH, 'bold_validation', 'bold_valid_norm.pickle'), 'wb') as f:
        #     pickle.dump(valid_data, f)

        print("Shape: {} x {}".format(len(bold_dataset), len(bold_dataset[0]['fmri'])))
        print('Data are saved to {}'.format(SAVE_PATH))
