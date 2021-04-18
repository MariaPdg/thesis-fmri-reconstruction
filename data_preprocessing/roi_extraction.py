import os
import glob
import time
import h5py
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split

import configs.data_config as data_cfg


def extract_roi_by_mask(preproc_dir, events_dir, mask_dir, data_dir, save=False):

    """
    Extract relevant preprocessed fmri data and applied masks and stores the concatenated data

    Modified from "Categorization of seen images from brain activity using sequence models"
        https://github.com/arashjamalian/fmriNet

    @param preproc_dir: path to the preprocessed fmri data: dataset/fmriprep folder
    @param events_dir:  path to the events files: dataset folder, should contain subject folders
    @param mask_dir:    path to the masks: dataset/spm folder
    @param data_dir:    path where the data should be saved

    """

    # Dicts holding training set and labels for each mask
    X = []
    Y = []

    # Holds subject, session, run information for each sample
    sample2image2session = pd.DataFrame()

    # Get masks in mask directory
    mask_files = glob.glob(mask_dir + '/*/*.nii.gz')
    prev_sub = ''

    # Walk through ROI mask directory
    for mask_file in mask_files:
        sub = mask_file.split('/')[-2]
        print(sub)
        preproc_files = glob.glob(preproc_dir + sub + '/*/*/*-5000scenes_*_preproc.nii.gz')
        logging.info("Using mask file: %s" % mask_file)
        if prev_sub != sub:
            X = []
            Y = []
            sample2image2session = pd.DataFrame()
        for pnum, preproc in enumerate(preproc_files):
            logging.info('Preprocessed file %d out of %d' % ((pnum + 1), len(preproc_files)))
            items = preproc.split('_')
            ses = items[1]
            run = items[3]
            subname = items[0].split('/')[-1]
            event_file = glob.glob(os.path.join(events_dir, subname, ses, 'func', '*' + run + '_events.tsv'))[0]
            # Load events and image
            events = pd.read_csv(event_file, sep = '\t')
            img = nib.load(preproc).get_fdata()
            mask = nib.load(mask_file).get_fdata().astype(bool)
            # Apply mask
            roi = img[mask]  # Shape: voxels x TRs
            # Get relevant time intervals and labels from events file
            for index, row in events.iterrows():
                # Beginning TR of trial
                start = int(round(row['onset']) / 2)
                # Ending TR of trial, start + 10 sec, or 5 TRs
                end = start + 5
                x = roi[:, start:end].T
                X.append(x)  # Big X should be of shape (samples, timepoints, features)
                Y.append(row['ImgName'])
                sample2image2session = sample2image2session.append({'SampleIndex': len(X) - 1,
                                                                    'ImgName': row['ImgName'],
                                                                    'Subject': subname,
                                                                    'Session': int(ses[-2:]),
                                                                    'Run': int(run[-2:])}, ignore_index = True)
            # Save last ten TRs as no stimulus, if enough data is left
            if roi.shape[1] - end >= 5:
                x = roi[:, end:end+5].T
                X.append(x)
                Y.append('none')
                sample2image2session = sample2image2session.append({'SampleIndex': len(X) - 1,
                                                                    'ImgName': 'none',
                                                                    'Subject': subname,
                                                                    'Session': int(ses[-2:]),
                                                                    'Run': int(run[-2:])}, ignore_index = True)

            if save:

                path = data_dir + sub

                if not os.path.exists(path):
                    os.makedirs(path)

                with open(path + '/' + sub + '_fmri.pickle', 'wb') as f:
                    pickle.dump(X, f)

                with open(path + sub + '/' + sub + '_image_names.pickle', 'wb') as f:
                    pickle.dump(Y, f)

                with open(path + sub + '/' + sub + '_df.pickle', 'wb') as f:
                    pickle.dump(sample2image2session, f)


def extract_roi(data_dir, roi_dir, save=False):

    """
    Extracts ROIs corresponding to the 4-8 sec (max of brain activity) for each subject, concatenates them
    with zero-padding to max number of voxels among 4 subjects and stores in specified directory
    Can be downloaded here: https://ndownloader.figshare.com/files/12965447

    @param data_dir: path where the data should be saved
    @param roi_dir: path to ROIs data (from BOLD5000 dataset)
    @param save: True to save results, otherwise False
    """
    max_roi = max_roi_length(roi_dir)

    subject_paths = glob.glob(roi_dir + 'CSI*')
    for path in subject_paths:
        sub = path.split('/')[-1]
        roi_TR34_path = os.path.join(roi_dir, sub, 'h5', sub + '_ROIs_TR34.h5')
        roi = h5py.File(roi_TR34_path, 'r')
        roi_list = []
        for region in roi.keys():
            item = np.array(roi[region])
            item = np.pad(item, ((0, 0), (0, max_roi[region] - item.shape[1])))
            # print('Sub', sub, item.shape)
            roi_list.append(np.array(item))
        all_sub_roi = np.concatenate(roi_list, axis=1)
        print(all_sub_roi.shape)

        if save:
            # save results
            path = data_dir + sub

            if not os.path.exists(path):
                os.makedirs(path)

            with open(path + '/' + sub + '_roi_pad.pickle', 'wb') as f:
                pickle.dump(all_sub_roi, f)


def max_roi_length(roi_dir):
    """
    Calculates the maximum lengths for each ROI for all subjects

    @param roi_dir: directory to roi data
    @return: max_length_dict:
        dictionary: {'ROI_name': max_len, ...}
    """

    subjects = data_cfg.subjects

    max_length_dict = {}
    for sub in subjects:
        roi_TR34_path = os.path.join(roi_dir, sub, 'h5', sub + '_ROIs_TR34.h5')
        roi = h5py.File(roi_TR34_path, 'r')
        for region in roi.keys():
            max_len = roi[region].shape[1]
            for sub in subjects:
                roi_TR34_path_sub = os.path.join(roi_dir, sub, 'h5', sub + '_ROIs_TR34.h5')
                roi_sub = h5py.File(roi_TR34_path_sub, 'r')
                if roi_sub[region].shape[1] > max_len:
                    max_len = roi_sub[region].shape[1]
                    max_length_dict.update({region: max_len})

    return max_length_dict


def find_stimuli_path(data_dir, roi_dir, save=False):
    """
    Finds the stimuli paths corresponding to extracted ROIs for each subject, concatenates them
    and stores in specified directory
    Uses BOLD_PICKLE_PATH to pickle file from bold_parser.py

    @param data_dir: path where the data should be saved
    @param roi_dir:  path to ROIs data (from BOLD5000 dataset)
    @param save: True to save results, otherwise False

    """

    stimuli_names = os.path.join(roi_dir, 'stim_lists')
    stimuli_lists = glob.glob(stimuli_names + '/*.txt')
    with open(BOLD_PICKLE_PATH, "rb") as input_file:
        aggregated_data = pickle.load(input_file)
    unique_pairs = set(zip(aggregated_data['img_name'], aggregated_data['stimuli_path']))

    for sub_stim_list in stimuli_lists:
        full_stimuli_paths = []
        file = open(sub_stim_list, 'r')
        sub_stimuli = file.readlines()
        for stim_name in sub_stimuli:
            # remove 'rep' at the beginning of names for repeated stimuli
            if stim_name[:3] =='rep':
                stim_name = stim_name[4:]
            for (name, path) in unique_pairs:
                # remove '\n' at the end of each name
                if stim_name.split('\n')[0] == name:
                    full_stimuli_paths.append(path)
        logger.info(len(full_stimuli_paths))

        # take only subject names
        name = sub_stim_list.split('/')[-1]
        sub = name[:3] + name[4]
        path = os.path.join(data_dir, sub)

    if save:
        # save results
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/' + sub +'_stimuli_paths.pickle', 'wb') as f:
            pickle.dump(full_stimuli_paths, f)


def train_test_stimuli_split(data_dir, roi_dir, ratio=0.1, save=False):
    """
    Define train and validation split for stimuli
    @param data_dir: path where to save
    @param roi_dir: path where rois are stored
    @param ratio: test set size
    @param save: True or False
    @return: train and validation lists of stimuli IDs, save if True
    """
    stimuli_names = os.path.join(roi_dir, 'stim_lists/CSI01_stim_lists.txt')

    full_stimuli_paths = []
    file = open(stimuli_names, 'r')
    sub_stimuli = file.readlines()
    for stim_name in sub_stimuli:
        # remove 'rep' at the beginning of names for repeated stimuli
        if stim_name[:3] =='rep':
            stim_name = stim_name[4:]
        full_stimuli_paths.append(stim_name.split('\n')[0])
    # logger.info(len(full_stimuli_paths))
    unique_set = list(set(full_stimuli_paths))
    train, test = train_test_split(unique_set, test_size=ratio, random_state=12345)

    if save:
        with open(data_dir + '/stimuli_train.pickle', 'wb') as f:
            pickle.dump(train, f)
        with open(data_dir + '/stimuli_valid.pickle', 'wb') as f:
            pickle.dump(test, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="user path where the datasets are located", type=str)
    parser.add_argument('--output', '-o', help="user path where to save", type=str)
    parser.add_argument('--logs', '-l', help="path where to save logs", type=str)
    args = parser.parse_args()

    # USER_PATH = sys.argv
    USER_ROOT = args.input
    DATA_ROOT = os.path.join(USER_ROOT, data_cfg.data_root)
    BOLD_PICKLE_PATH = os.path.join(args.output, data_cfg.data_root, data_cfg.bold_pickle_file)
    ROI_PATH = os.path.join(args.output, data_cfg.data_root, data_cfg.rois_path)
    SAVE_PATH = os.path.join(args.output, data_cfg.data_root, data_cfg.save_path, 'bold_roi')

    # info logging
    timestep = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(args.logs, 'roi_extraction_' + timestep))
    logger = logging.getLogger()
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    max_len = max_roi_length(ROI_PATH)
    print(max_len)

    save = False  # True if you want to save data

    # We save rois and corresponding stimuli paths here, uncomment if you want to run
    # extract_roi(SAVE_PATH, ROI_PATH, save=save)
    # find_stimuli_path(SAVE_PATH, ROI_PATH, save=save) # finds stimuli paths using BOLD_PICKLE_PATH

    # Function to save data with the fixed split
    # train_test_stimuli_split(SAVE_PATH, ROI_PATH, save=True)

