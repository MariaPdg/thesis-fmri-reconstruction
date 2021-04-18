"""
Adapted from " BOLD5000 Data Aggregation"
https://git.hcics.simtech.uni-stuttgart.de/collaboration-projects/bold5000
"""

import os
import re


def get_identifier(filename, source_dataset):
    '''
    extracts identifier from filename for different dataset sources
    '''
    dataset = source_dataset.lower()
    if dataset in ['coco', 'rep_coco']:
        return filename[-16:-4].lstrip('0')
    if dataset in ['imagenet', 'rep_imagenet']:
        return filename[:-5]
    if dataset in ['scenes', 'rep_scenes', 'scene', 'sun']:
        return filename[:-4]
    else:
        raise Exception('unknown source dataset: {}'.format(source_dataset))


def get_stimuli_path(filename, source_dataset, bold_stimuli_path):
    '''
    generates the path according to the original BOLD project files structure and the stimuli path constant
    '''
    dataset = source_dataset.lower()
    if dataset in ['coco', 'rep_coco']:
        return os.path.join(bold_stimuli_path, 'COCO', filename)
    if dataset in ['imagenet', 'rep_imagenet']:
        return os.path.join(bold_stimuli_path, 'ImageNet', filename)
    if dataset in ['scenes', 'rep_scenes']:
        return os.path.join(bold_stimuli_path, 'Scene', filename)
    else:
        raise Exception('unknown source dataset: {}'.format(source_dataset))


def get_fmri_path(subj, sess, run, bold_session_path):
    return os.path.join(bold_session_path,
                        'sub-CSI{subj}/ses-{sess:02d}/func/sub-CSI{subj}_ses-{sess:02d}_task-5000scenes_run-{run:02d}_bold.nii.gz'.format(
                            subj=subj, sess=sess, run=run))


def snake_case(name):
    '''
    converts input to snakecase
    '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()