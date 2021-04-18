"""
Adapted from " BOLD5000 Data Aggregation"
https://git.hcics.simtech.uni-stuttgart.de/collaboration-projects/bold5000
"""
import os
import re
import pickle
import json
import argparse
import pandas as pd
import configs.data_config as data_cfg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nilearn import plotting, image
from pyparsing import Word, Suppress, Group, OneOrMore, LineEnd, alphanums, nums, ZeroOrMore
from data_preprocessing.data_utils import get_identifier, get_fmri_path, get_stimuli_path, snake_case


class Bold5000Parser(object):
    """
    Bold5000 parser to define bold signals (fmri data) with corresponding parameters and stimuli matching
    Dataset description: https://bold5000.github.io/
    """

    def __init__(self, root_dir, bold_stimuli_dir, bold_labels_dir, bold_session_dir, coco_annotation_dir):
        """
        :param root_dir: the root dir where Bold5000 dataset is saved
        :param bold_stimuli_dir: the root to stimuli
        :param bold_labels_dir: the root to stimuli labels
        :param bold_session_dir: the root to fmri session data
        :param coco_annotation_dir: the root to COCO annotation file
        """
        super(Bold5000Parser, self).__init__()
        self.root_dir = root_dir
        self.bold_stimuli_dir = bold_stimuli_dir
        self.bold_labels_dir = bold_labels_dir
        self.bold_session_dir = bold_session_dir
        self.coco_annotation_dir = coco_annotation_dir

    def coco_mapping(self):
        """
        This function performs bold-coco mapping

        :return: bold_coco_annotations: dataframe ['filename', 'identifier', 'source', 'coco_annotations']
            Shows path to the COCO image with identifier as well as its category
        :return: bold_coco_mapping: dataframe ['identifier', 'category_id']
            Shows COCO image identifier for each BOLD sample as well as COCO category
        """

        dataframe = {'filename': [], 'identifier': [], 'source': []}

        for (dirpath, dirs, files) in os.walk(os.path.join(self.root_dir, self.bold_stimuli_dir, 'COCO')):
            for filename in files:
                dataframe['filename'].append(filename)
                dataframe['identifier'].append(get_identifier(filename, 'coco'))
                dataframe['source'].append('coco')

        with open(os.path.join(self.root_dir, self.bold_labels_dir, 'coco_final_annotations.pkl'), 'rb') as file:
            bold_annotations = pickle.load(file)

        with open(self.coco_annotation_dir) as file:
            coco = json.load(file)

        coco_annotations = pd.DataFrame(coco['categories'])
        coco_annotations = coco_annotations.set_index('id')

        bold_coco_mapping = pd.DataFrame(
            [{'identifier': i, 'category_id': a.get('category_id')} for i in dataframe['identifier'] for a in
             bold_annotations.get(int(i))])
        dataframe['coco_annotations'] = [
            list({coco_annotations.loc[a.get('category_id')]['name'] for a in bold_annotations.get(int(i))}) for i in
            dataframe['identifier']]
        bold_coco_annotations = pd.DataFrame(dataframe, index=pd.Index(dataframe['identifier'], name='identifier'))

        return bold_coco_annotations, bold_coco_mapping

    def imagenet_mapping(self):

        """
        :return: imagenet_annotations: dataframe ['synset', 'name']
            Image met identifier (eg. n01440764) and the category name
        :return: bold_imagenet_annotations: dataframe ['filename', 'identifier', 'source', 'synset', 'imagenet_annotations']
            Shows the path to the image net images for each bold sample as well as the identifier and the categories
        """

        dataframe = {'filename': [], 'identifier': [], 'source': [], 'synset': []}

        for (dirpath, dirs, files) in os.walk(os.path.join(self.root_dir, self.bold_stimuli_dir, 'ImageNet')):
            for filename in files:
                dataframe['filename'].append(filename)
                dataframe['identifier'].append(get_identifier(filename, 'imagenet'))
                dataframe['synset'].append(get_identifier(filename, 'imagenet').split('_')[0])
                dataframe['source'].append('imagenet')

        with open(os.path.join(self.root_dir, self.bold_labels_dir, 'imagenet_final_labels.txt'), 'rb') as file:
            annotations = file.read().decode('utf-8')

            # parse imagenet annotation file using pyparsing
        identifier = Word('n', nums)
        annotation_list = Group(Word(alphanums + ' -\'.') + ZeroOrMore(Suppress(',') + Word(alphanums + ' -\'.')))
        annotation_pattern = OneOrMore(identifier + annotation_list + ZeroOrMore(Suppress(LineEnd())))
        parsed = annotation_pattern.parseString(annotations).asList()

        # unravel list of lists to clean dataframe
        imagenet_annotations = pd.DataFrame({'synset': parsed[::2], 'raw_annotations': parsed[1::2]})
        imagenet_annotations = imagenet_annotations.raw_annotations \
            .apply(pd.Series) \
            .merge(imagenet_annotations, left_index=True, right_index=True) \
            .drop('raw_annotations', axis=1) \
            .melt(id_vars=['synset'], value_name="name") \
            .drop('variable', axis=1) \
            .dropna()

        dataframe['imagenet_annotations'] = [imagenet_annotations.loc[imagenet_annotations.synset == s].name.tolist()
                                             for s in dataframe['synset']]
        bold_imagenet_annotations = pd.DataFrame(dataframe, index=pd.Index(dataframe['identifier'], name='identifier'))

        return imagenet_annotations, bold_imagenet_annotations

    def scene_mapping(self):

        """

        :return: scene_annotations: dataframe [index, category]
            Shows the scene name for each sample
        :return: bold_scene_annotations: ['filename', 'identifier', 'source', 'scene_label']
            Shown the scene image corresponding to each bold image
        """

        dataframe = {'filename': [], 'identifier': [], 'source': [], 'scene_label': []}
        label_finder = re.compile('^[^\d]*')

        for (dirpath, dirs, files) in os.walk(os.path.join(self.root_dir, self.bold_stimuli_dir, 'Scene')):
            for filename in files:
                dataframe['filename'].append(filename)
                dataframe['identifier'].append(get_identifier(filename, 'scene'))
                dataframe['source'].append('scene')
                dataframe['scene_label'].append(label_finder.match(filename).group())

        with open(os.path.join(self.root_dir, self.bold_labels_dir, 'scene_final_labels.txt'), 'rb') as file:
            annotations = file.read().decode('utf-8')

        scene_annotations = pd.DataFrame(annotations.split())
        bold_scene_annotations = pd.DataFrame(dataframe, index=pd.Index(dataframe['identifier'], name='identifier'))

        return scene_annotations, bold_scene_annotations

    def session_data(self):
        """
        Shows fmri data from BOLD5000 dataset

        :return: bold_session_data: dataframe ['onset', 'duration', 'subj', 'sess', 'run', 'trial', 'img_name',
            'img_type', 'stim_on(s)', 'stim_off(s)', 'response', 'rt', 'stim_file',
            'stimuli_path', 'fmri_path']
        """

        session_file_list = []

        event_file_pattern = re.compile(r'sub-CSI\d_ses-\d\d_task-5000scenes_run-\d\d_events\.tsv')
        for (dirpath, dirs, files) in os.walk(os.path.join(self.root_dir, self.bold_session_dir)):
            for filename in files:
                if event_file_pattern.match(filename):
                    session_file_list.append(dirpath + '/' + filename)

        bold_session_data = pd.concat([pd.read_csv(file, sep='\t') for file in session_file_list])
        bold_session_data.columns = [snake_case(c) for c in bold_session_data.columns]
        bold_session_data['identifier'] = bold_session_data.apply(
            lambda row: get_identifier(row['img_name'], row['img_type']), axis=1)
        bold_session_data['stimuli_path'] = bold_session_data.apply(
            lambda row: get_stimuli_path(row['img_name'], row['img_type'], self.bold_stimuli_dir), axis=1)
        bold_session_data['fmri_path'] = bold_session_data.apply(
            lambda row: get_fmri_path(row['subj'], row['sess'], row['run'], self.bold_session_dir), axis=1)
        bold_session_data.set_index('identifier', inplace=True)

        return bold_session_data

    @staticmethod
    def aggregation(bold_session_data, bold_coco_annotations, bold_imagenet_annotations, bold_scene_annotations):

        """
        Maps fmri data from BOLD5000 and images from COCO, ImageNet, Scene datasets

        :param bold_session_data: dataframe after session_data() method
        :param bold_coco_annotations: dataframe after coco_mapping() method
        :param bold_imagenet_annotations: dataframe after imagenaet_mapping() method
        :param bold_scene_annotations: dataframe after scene_mapping() method

        :return: aggregation: aggregated data from BOLD500, COCO, ImageNet, Scene datasets
            dataframe ['onset', 'duration', 'subj', 'sess', 'run', 'trial', 'img_name',
            'img_type', 'stim_on(s)', 'stim_off(s)', 'response', 'rt', 'stim_file',
            'stimuli_path', 'fmri_path', 'coco_annotations', 'imagenet_annotations',
            'scene_label']

        """
        aggregation = bold_session_data.join(bold_coco_annotations['coco_annotations']).join(
            bold_imagenet_annotations['imagenet_annotations']).join(bold_scene_annotations['scene_label'])

        aggregation.index.name = 'identifier'
        # aggregation.reset_index(inplace=True)
        # aggregation.set_index(['Subj', 'Sess', 'Run', 'Trial'], drop=False, inplace=True)
        aggregation.sort_index(inplace=True)

        return aggregation


if __name__ == "__main__":

    # USER_PATH = sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="user path where the datasets are located", type=str)
    parser.add_argument('--output', '-o', help="user path where to save", type=str)
    parser.add_argument('--logs', '-l', help="path where to save logs", type=str)
    args = parser.parse_args()

    # dataset is not in a home directory (specified as a parameter)
    USER_ROOT = args.input
    DATA_ROOT = os.path.join(USER_ROOT, data_cfg.data_root)
    BOLD_STIMULI_PATH = os.path.join(DATA_ROOT, data_cfg.bold_stimuli_path)
    BOLD_LABELS_PATH = os.path.join(DATA_ROOT, data_cfg.bold_labels_path)
    BOLD_SESSIONS_PATH = os.path.join(DATA_ROOT, data_cfg.bold_session_path)
    COCO_ANNOTATION_FILE = os.path.join(DATA_ROOT, data_cfg.coco_annotation_file)
    SAVE_PATH = os.path.join(args.output, data_cfg.data_root, data_cfg.save_path)

    if not os.path.exists(BOLD_STIMULI_PATH):
        raise Exception('Stimuli path does not exist!')

    bold_object = Bold5000Parser(root_dir=DATA_ROOT,
                                 bold_stimuli_dir=BOLD_STIMULI_PATH,
                                 bold_labels_dir=BOLD_LABELS_PATH,
                                 bold_session_dir=BOLD_SESSIONS_PATH,
                                 coco_annotation_dir=COCO_ANNOTATION_FILE)

    bold_coco_annotation, bold_coco_map = bold_object.coco_mapping()
    imagenet_annotation, bold_imagenet_annotation = bold_object.imagenet_mapping()
    scene_annotation, bold_scene_annotation = bold_object.scene_mapping()
    bold_sess_data = bold_object.session_data()
    aggregated_data = bold_object.aggregation(bold_session_data=bold_sess_data,
                                              bold_coco_annotations=bold_coco_annotation,
                                              bold_imagenet_annotations=bold_imagenet_annotation,
                                              bold_scene_annotations=bold_scene_annotation)

    with open(os.path.join(SAVE_PATH, 'bold5000.pickle'), 'wb') as f:
        pickle.dump(aggregated_data, f, protocol=4)

    print(aggregated_data['stimuli_path'][0])
    print(aggregated_data['fmri_path'][0])

    image_mp = mpimg.imread(aggregated_data['stimuli_path'][0])
    imgplot = plt.imshow(image_mp)
    plt.plot()
    plt.savefig('image.png')

    # visualize only zero frame: 4d => 3d
    img4d = image.index_img(aggregated_data['fmri_path'][0], 0)
    plotting.plot_img(img4d, output_file='fmri.png')
