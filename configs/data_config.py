
""" _________________________________bold_parser.py_____________________________________"""

# path to the dataset user_path/[data_root], user_path is specified as arguments
data_root = 'datasets/'
# path to presented stimuli
bold_stimuli_path = 'BOLD5000/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/'
# path to stimuli labels
bold_labels_path = 'BOLD5000/BOLD5000_Stimuli/Image_Labels/'
# path to unprocessed bold5000 data
bold_session_path = 'BOLD5000/ds001499-download/'
# path to COCO annotations
coco_annotation_file = 'coco_2017/annotations/annotations_image_info_test2017/image_info_test2017.json'
# path where to save preprocessed data user_path/[save_path]
save_path = 'BOLD5000/'


"""__________________________________roi_extraction.py____________________________________"""

# path to pickle file created by bold_parser.py
bold_pickle_file = 'BOLD5000/bold5000.pickle'
# path to ROIs
rois_path = 'bold5000_figshare/ROIs/'


"""_________________________________data_loader.py_________________________________________"""

# path to pickle file with list of paths to coco test data, created with the function 'prepare_external_data'
# (no longer used)
external_data_pickle = 'BOLD5000/coco_external_data.pickle'  # coco test
coco_train_pickle = 'BOLD5000/coco_data_train.pickle'
coco_valid_pickle = 'BOLD5000/coco_data_valid.pickle'

# paths to coco data
coco_train_data = 'coco/coco_train2017/train2017'
coco_valid_data = 'coco/coco_valid2017/val2017'
coco_test_data = 'coco/coco_test2017/test2017'

# data split with/without fixed stimuli IDs
train_data = 'BOLD5000/bold_train/bold_train_all_fixed.pickle'
valid_data = 'BOLD5000/bold_valid/bold_valid_all_fixed.pickle'

# stimuli split to fix train and validation sets
train_stimuli_split = 'BOLD5000/bold_roi/stimuli_train.pickle'
valid_stimuli_split = 'BOLD5000/bold_roi/stimuli_valid.pickle'

"""__________________________________training________________________________________________"""

# roi data created with the function 'extract_roi'
bold_roi_data = 'BOLD5000/bold_roi/'
save_training_results = 'results/'

# path to mnist69 dataset (optional)
mnist_path = 'mnist/69dataset.mat'

"""_________________________other data parameters for BOLD5000_________________________________"""

image_size = 64

subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']

rois_max = {'LHEarlyVis': 522,
            'LHLOC': 455,
            'LHOPA': 279,
            'LHRSC': 86,
            'LHPPA': 172,
            'RHEarlyVis': 696,
            'RHLOC': 597,
            'RHOPA': 335,
            'RHRSC': 278,
            'RHPPA': 200}

num_voxels = 3620