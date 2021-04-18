# Image reconstruction from human brain activity 

## Data setup:

* The whole dataset: 
  - [BOLD5000 openneuro.org](https://openneuro.org/datasets/ds001499/versions/1.3.0)
  - [BOLD5000 figshare.com](https://figshare.com/articles/dataset/BOLD5000/6459449)
* You can download only ROI data: [BOLD5000 ROIs](https://ndownloader.figshare.com/files/12965447) (in that case you have to match ROIs and Stimuli and skip *Data setup* )
* Stimuli: [BOLD5000 stimuli](https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1)

### Preliminary steps:

1. Set python path
```
export PYTHONPATH=$PYTHONPATH:[absolute path to the folder]/fmri_reconstruction/
```
2. For data setup you should modify ```config/data_configs.py``` (see details below).

3. Preprocessed data are available [here](https://drive.google.com/drive/folders/1yftoRrlOOb1INTxs2Jcq5bMKICI5VwQ5?usp=sharing). 
   If you use downloaded files you have to locate them in the folder with the name *BOLD5000*. 
   
    We modify stimuli paths in `BoldRoiDataloader`:
   ```python
    # TODO: here we replace stimuli paths with your own with root_path
    # It's supposed that fMRI data are in the folder "/BOLD5000/"
    if self.root_path is not None and self.root_path not in self.dataset[idx]['image']:
        name = self.dataset[idx]['image'].split('BOLD5000')[0]
        self.dataset[idx]['image'] = self.dataset[idx]['image'].replace(name, self.root_path)
    ```

4. We recommend preparing the data directory with the similar structure following the steps described below. 
  
### Data preprocessing

### BOLD Parsing

1. Specify data root in ```data_configs:```
```python
# path to the dataset user_path/[data_root], user_path is specified as arguments
data_root = 'datasets/'
```
1. Specified paths to fMRI and stimuli paths in ```data_config:```
   
```python
    # path to presented stimuli
    bold_stimuli_path = 'BOLD5000/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/'
    # path to stimuli labels
    bold_labels_path = 'BOLD5000/BOLD5000_Stimuli/Image_Labels/'
    # path to unprocessed bold5000 data
    bold_session_path = 'BOLD5000/ds001499-download/'
    # path to COCO annotations
    coco_annotation_file = 'coco_2017/annotations/annotations_image_info_test2017/image_info_test2017.json'

```
2. Path to save pickle file, which contains paths to bold data and corresponding stimuli paths

```python
    # path where to save preprocessed data user_path/[save_path]
    save_path = 'BOLD5000/'
```
3. Run  ```bold_parser.py``` with flags:

```
   python3 data_preprocessing/bold_parser.py -i [user path 1] -o [user path 2] -l [path to logs]
```
  Flags:
  * ``` -i [user path 1]``` user path where datasets are located 
  * ``` -o [user path 2]``` user path where the processed data should be saved 
  * ``` -l [path to logs]``` path to save logs
   
NOTE: ```user_path1``` and ```user_path2``` can be the same

4. Pickle file will be saved in ```user_path/[save_path]```

5. See an example file: [bold500.pickle](https://drive.google.com/file/d/1L_a4pjeUUh8NlD2bgiSCpNAj5MviKzEJ/view?usp=sharing)


### Extraction of ROIs 

Here we use the script ```data_preprocessing/roi_extraction.py```

1. Specified path to the saved pickle file in *data_config*

```python
bold_pickle_file = 'BOLD5000/bold5000.pickle'
```

2. Run ```data_preprocessing/roi_extraction.py``` with the same flags and use (uncomment) the following functions:

```python
    
    from  data_preprocessing.roi_extraction import  extract_roi, find_stimuli_path

    extract_roi(SAVE_PATH, ROI_PATH, save=save)
    find_stimuli_path(SAVE_PATH, ROI_PATH, save=save)
```

3. Pickle files with ROIs data and corresponding stimuli paths will be saved. The directory will look like:
   
    ```python
            |-- BOLD5000
                |-- bold_roi
                    |-- CSI1
                        |--CSI1_roi_pad.pickle
                        |--CSI1_stimuli_paths.pickle 
                    |-- CSI2
                        |--CSI2_roi_pad.pickle
                        |--CSI2_stimuli_paths.pickle
                    |-- CSI3
                       |-- ...
                    |-- CSI4
                       |-- ...
    ```
   See [example](https://drive.google.com/drive/folders/1NPXDvj12O0pD2YCxHraprLtCrSPxnAhT?usp=sharing). 
   You can download these data, but you have to change absolute paths to stimuli.

### Dataloader

Here we use the script ```data_preprocessing/data_loader.py```

1. The function ``` concatenate_bold_data(data_dir)``` allows to concatenate all fmri data. 
   We also split these data in train and validation sets.

```python

    import pickle
    from data_preprocessing.data_loader import concatenate_bold_data
    from sklearn.model_selection import train_test_split
    
    data_path = ... # the directory BOLD5000/bold_roi/... must be specified
    # Concatenate data for all subjects
    bold_dataset = concatenate_bold_data(data_path)
    # Split into training and validation sets
    train_data, valid_data = train_test_split(bold_dataset, test_size=0.2, random_state=12345)
    with open(os.path.join(SAVE_PATH, 'bold_train', 'bold_train_norm.pickle'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(SAVE_PATH, 'bold_validation', 'bold_valid_norm.pickle'), 'wb') as f:
        pickle.dump(valid_data, f)
```
2.  Data are saved to directories:
   
   ```python
   train_data = 'BOLD5000/bold_train/bold_train_norm.pickle'
   valid_data = 'BOLD5000/bold_valid/bold_valid_norm.pickle'
   ```
   * Example [bold train](https://drive.google.com/file/d/1FXth92p9gI1dGI8-c-P032psIB_VU6ZL/view?usp=sharing)
   * Example [bold valid](https://drive.google.com/file/d/1p7_i8M9tWc8B3wc6YzBtWhkzh2yjzJdk/view?usp=sharing)
   
3. You also can use fixed data split, see below.


### Fixed data split

Here we prepare the data with fixed stimuli split. 

You can download files with the list of fixed stimuli IDs:

* [Stimuli train](https://drive.google.com/file/d/1COGYwtJvZnQlA23bKULsrmh_nosr6a8C/view?usp=sharing)
* [Stimuli valid](https://drive.google.com/file/d/1hBb79RQ64RnnQSiqy9Bb_6TTXnKgozLa/view?usp=sharing)

They include stimuli IDs for the training set (90%) abd the validation set (10%).

If you want to prepare you own data split, follow the steps below:

1. Run the function ```train_test_stimuli_split```

```python
from  data_preprocessing.roi_extraction import train_test_stimuli_split

ratio = 0.1 # size of the validation set
train_test_stimuli_split(SAVE_PATH, ROI_PATH, ratio=ratio, save=save)
```
   
* Training stimuli IDs will be saved in file: ```stimuli_train.pickle``` in path SAVE_PATH
* Validation stimuli IDs will be saved in file: ```stimuli_valid.pickle``` in path SAVE_PATH

The current paths:
```python
# stimuli split to fix train and validation sets
train_stimuli_split = 'BOLD5000/bold_roi/stimuli_train.pickle'
valid_stimuli_split = 'BOLD5000/bold_roi/stimuli_valid.pickle'
```

2. All concatenated fMRI with fixed stimuli IDs:

* [Training data](https://drive.google.com/file/d/1Zohf2I-ZHsY8f-NdLJl9oSmxfzylHVpS/view?usp=sharing)
* [Validation data](https://drive.google.com/file/d/1nkJ9OwJ3kR1wDS2BBBQg1mnzbsG_S-Rn/view?usp=sharing)

In ```data_configs.py:```
```
# data split with/without fixed stimuli IDs
train: 'BOLD5000/bold_train/bold_train_all_fixed.pickle'
valid: 'BOLD5000/bold_valid/bold_valid_all_fixed.pickle'
```

3. In order to use data for one subject:

In ```data_configs.py:```
```
# data split with/without fixed stimuli IDs
train: 'BOLD5000/bold_train/bold_CSI3_pad.pickle'
valid: 'BOLD5000/bold_train/bold_CSI3_pad.pickle'
```

For training/inference do the following:

```python
from data_preprocessing.bold_loader import split_subject_data

train_data = split_subject_data(train_data, TRAIN_STIMULI)
valid_data = split_subject_data(train_data, VALID_STIMULI)
```

Congrats, you prepare all data for training!

