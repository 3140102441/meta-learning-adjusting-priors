import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random

root = "/data5/dataset/cifar100"
img_root = "/data5/dataset/cifar100/imgs"

def load_ae_cifar100():
    train_d = loadCSV(os.path.join(root, "train.csv"))
    val_d = loadCSV(os.path.join(root, "val.csv"))
    test_d = loadCSV(os.path.join(root, "test.csv"))
    X = []
    y = []
    cls = 0
    for d in [train_d, val_d, test_d]:
        for k,v in d.items():
            num = len(v)
            X.extend(v)
            y.extend([cls] * num)
            cls += 1
        print(cls)
        assert(len(X) == len(y))

    print(cls)
    assert cls == 100
    return CIFAR100_dataset([X, y])


def load_pretrain_dataset():
    split_portion = 0.9

    d = loadCSV(os.path.join(root, "train.csv"))

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for k,v in d.items():
        train_num = split_portion * len(v)
        train_num = int(train_num)
        test_num = len(v) - train_num
        X_train.extend(v[:train_num])
        X_test.extend(v[train_num:])

        y_train.extend([int(k)] * train_num)
        y_test.extend([int(k)] * test_num)

        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    train_dataset = CIFAR100_dataset([X_train, y_train])
    test_dataset = CIFAR100_dataset([X_test, y_test])

    info = {}
    info["class_num"] = len(d)


    return train_dataset, test_dataset, info

"""
Interface with data_gen.py
return a dict:
    {
        "meta_train" : {cls1:[file1, file2 ....], ...}
        "meta_test" : {cls1:[file1, file2 ....], ...}
    }
"""
def split_classes():
    re = {}
    d = loadCSV(os.path.join(root, "train.csv"))
    re['meta_train'] = d
    d = loadCSV(os.path.join(root, "test.csv"))
    re['meta_test'] = d

    return re

"""
Interface with data_gen.py

"""
def get_task(d, n_way, k_shot):
    min_shot = 10000
    for k,v in d.items():
        if len(v) < min_shot:
            min_shot = len(v)

    if k_shot >= min_shot:
        raise Exception("K is too large for current dataset")

    class_inds = np.random.choice(len(d), n_way, replace = False)
    all_classes = list(d.items())
    classes = [all_classes[i] for i in class_inds]

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    i = 0
    for k,v in classes:
        random.shuffle(v)

        X_train.extend(v[:k_shot])
        X_test.extend(v[k_shot:])

        y_train.extend([int(i)] * k_shot)
        y_test.extend([int(i)] * (len(v) - k_shot))

        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        i += 1

    train_dataset = CIFAR100_dataset([X_train, y_train])
    test_dataset = CIFAR100_dataset([X_test, y_test])



    return train_dataset, test_dataset

"""
Interface with data_gen.py

"""
def get_split_task(d, n_way, k_shot, i_task):
    task_num = 12
    assert n_way == 5
    assert i_task >= 0 and i_task < task_num

    min_shot = 10000
    for k,v in d.items():
        if len(v) < min_shot:
            min_shot = len(v)

    if k_shot >= min_shot:
        raise Exception("K is too large for current dataset")

    #class_inds = np.random.choice(len(d), n_way, replace = False)
    all_classes = list(d.items())
    classes = all_classes[i_task * n_way : (i_task + 1) * n_way]

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    i = 0
    for k,v in classes:
        random.shuffle(v)

        X_train.extend(v[:k_shot])
        X_test.extend(v[k_shot:])

        y_train.extend([int(i)] * k_shot)
        y_test.extend([int(i)] * (len(v) - k_shot))

        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        i += 1

    train_dataset = CIFAR100_dataset([X_train, y_train])
    test_dataset = CIFAR100_dataset([X_test, y_test])



    return train_dataset, test_dataset



def loadCSV(csvf):
    """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
    """
    d = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            #this label is not the final label
            #because it is not continuous
            label = row[1]
            if label not in d:
                d[label] = []
            d[label].append(filename)
            # append filename to current label

    """
    将每个类映射到顺序编号上，作为这个类的label
    """
    new_d = {}
    startidx = 0
    for i, (k, v) in enumerate(d.items()):
        new_d[startidx + i] = v

    return new_d




class CIFAR100_dataset(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, data, resize = 32):
        """
        :param data: [X, y]
        :param resize: resize to
        """
        super(CIFAR100_dataset, self).__init__()
        self.resize = resize

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                         transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                                 ])
        self.path = img_root
        self.X = data[0]
        self.y = data[1]
        assert len(self.X) == len(self.y)
        self.len = len(self.y)

        print("Generate CIFAR dataset, path : {}, imgs_num : {}".format(root, self.len))

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        img_path = self.X[index]
        label = self.y[index]
        # img = self.loader(img_path)
        img = os.path.join(self.path, img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.len

