import os
import yaml
import numpy as np
from torchvision import datasets, transforms

def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])
    return np.array(images), np.array(labels)

def build_transform(is_train, args):
    input_size = 224
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform
    t = []
    size = int((256 / 224) * input_size)
    t.append(transforms.Resize(size, interpolation=3))
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    return t

def replace_data_path(paths, new_base_path):
    """
    Replace the 'data' part of each path with the new_base_path.
    Args:
    paths (list of str): List of paths to be modified.
    new_base_path (str): The new base path to replace 'data'.
    Returns:
    list of str: List of modified paths.
    """
    modified_paths = []
    for path in paths:
        # Split the path into parts
        parts = path.split(os.sep)
        # Find the index of 'data' and replace it with the new_base_path
        if 'data' in parts:
            index = parts.index('data')
            parts[index] = new_base_path
            # Join the parts back into a path
            new_path = os.sep.join(parts)
            modified_paths.append(new_path)
        else:
            # If 'data' is not found, append the original path
            modified_paths.append(path)
    return modified_paths

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNetA(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class OxfordPets(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(37).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/OxfordPets/train/"
        test_dir = "./data/OxfordPets/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class EuroSAT(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(10).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/EuroSAT/train/"
        test_dir = "./data/EuroSAT/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class PlantVillage(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(15).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/PlantVillage/train/"
        test_dir = "./data/PlantVillage/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
class VTAB(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(50).tolist()

    def download_data(self):
        train_dir = "./data/VTAB/train/"
        test_dir = "./data/VTAB/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class Kvasir(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        self.class_order = np.arange(8).tolist()

    def download_data(self):
        train_dir = "./data/Kvasir/train/"
        test_dir = "./data/Kvasir/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class DomainNet(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, self.args)
        self.test_trsf = build_transform(False, self.args)
        self.common_trsf = []
        # self.train_trsf = [
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        # ]
        # self.test_trsf = [
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        # ]
        # self.common_trsf = [
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        # ]
        self.class_order = np.arange(345).tolist()

    def download_data(self):
        # load splits from config file
        train_config = yaml.load(open('/media/david/HDD2/Datasets/DomainNet/domainnet_train.yaml', 'r'), Loader=yaml.Loader)
        test_config = yaml.load(open('/media/david/HDD2/Datasets/DomainNet/domainnet_test.yaml', 'r'), Loader=yaml.Loader)
        train_data = replace_data_path(train_config['data'], '/media/david/HDD2/Datasets')
        test_data = replace_data_path(test_config['data'], '/media/david/HDD2/Datasets')
        self.train_data, self.train_targets = np.array(train_data), np.array(train_config['targets'])
        self.test_data, self.test_targets = np.array(test_data), np.array(test_config['targets'])