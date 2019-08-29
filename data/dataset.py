import numpy as np
import os.path as osp
ospj = osp.join
ospeu = osp.expanduser

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader

from data.common import loadPickle

class GeneralDataLoader(dataset.Dataset):

    def __init__(self, args, transform, dname, dtype):
        self.transform = transform
        self.loader = default_loader
        dataset_rt_dir = args.datadir

        assert dname in ['market1501', 'duke', 'cuhk03', 'rap2'], \
            "Unsupported Dataset {}".format(dname)
        assert dtype in ['train', 'test', 'query'], \
            "Unsupported Dataset part {}".format(dtype)

        # Get filenames
        im_dir, partition_file = self._getDatasetFile(dataset_rt_dir, dname)
        # Get sample paths
        self.imgs = self._loadImgNames(im_dir, partition_file, dtype)
        # Labels
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def _getDatasetFile(self, dataset_rt_dir, dname):
        if dname == 'cuhk03':
            im_type = ['detected', 'labeled'][0]
            dataset_rt_dir = ospj(dataset_rt_dir, dname)
            im_dir = ospeu(ospj(dataset_rt_dir, im_type, 'images'))
            partition_file = ospeu(ospj(dataset_rt_dir, im_type, \
                'partitions.pkl'))
        else:
            im_dir = ospeu(ospj(dataset_rt_dir, dname, 'images'))
            partition_file = ospeu(ospj(dataset_rt_dir, dname, \
                'partitions.pkl'))

        return im_dir, partition_file

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def _loadImgNames(self, im_dir, partition_file, dtype):
        partitions = loadPickle(partition_file)
        if dtype == 'train':
            pname = 'trainval'
        elif dtype == 'test' or dtype == 'query':
            pname = 'test'
        else:
            print("Error: Bad data type!")
            exit(-1)
        # Get marks of testset
        marks = np.asarray(partitions['test_marks'])
        # Get image fnames.
        im_fnames = partitions['{}_im_names'.format(pname)]
        im_fnames = np.asarray(im_fnames)
        if dtype == 'train':
            im_names = [ospj(im_dir, iname) for iname in im_fnames]
        elif dtype == 'test':
            ginds = marks == 1
            im_fnames_temp = im_fnames[ginds]
            im_names = [ospj(im_dir, iname) for iname in im_fnames_temp]
        else: # Query
            qinds = marks == 0
            im_fnames_temp = im_fnames[qinds]
            im_names = [ospj(im_dir, iname) for iname in im_fnames_temp]
        return im_names

    @staticmethod
    def id(file_path):
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        return int(file_path.split('/')[-1].split('_')[1])

    @property
    def ids(self):
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        return sorted(set(self.ids))

    @property
    def cameras(self):
        return [self.camera(path) for path in self.imgs]
