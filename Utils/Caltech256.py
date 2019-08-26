import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random


class Caltech256(Dataset):
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

    def __init__(self, root, mode, resize = 96, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        super(Caltech256, self).__init__()
        self.startidx = startidx  # index label not from 0, but from startidx
        self.resize = resize

        #if mode == 'train':
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                         transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        #else:
        #    self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
        #                                         transforms.Resize((self.resize, self.resize)),
        #                                         transforms.ToTensor(),
        #                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #                                         ])

        self.path = os.path.join(root, '256_ObjectCategories')  # image path
        if mode == "all":
            X_train, y_train_ = self.loadCSV(os.path.join(root, "train" + '.csv'))  # csv path
            X_val, y_val_ = self.loadCSV(os.path.join(root, "val" + '.csv'))  # csv path
            X_test, y_test_ = self.loadCSV(os.path.join(root, "test" + '.csv'))  # csv path
            X = X_train + X_val + X_test
            y_ = y_train_ + y_val_ + y_test_
        elif mode == "pretrain":
            X, y_ = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path

        label2label = {}
        uni_y = list(set(y_))
        """
        将每个类映射到顺序编号上，作为这个类的label
        """
        for i, label_ in enumerate(uni_y):
            label2label[label_] = startidx + i

        self.cls_num = len(uni_y)

        y = []
        for e in y_:
            y.append(label2label[e])
        self.X = X
        self.y = y
        print(len(self.X))
        print(len(self.y))
        assert len(self.X) == len(self.y)
        self.len = len(self.y)

        print("Generate Caltech 256 dataset, split : {}, path : {}, imgs_num : {}".format(mode, root, self.len))


    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """

        X = []
        y = []
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                #this label is not the final label
                #because it is not continuous
                label = row[1][:3]
                # append filename to current label
                X.append(filename)
                y.append(label)
        return X,y

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

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

def load_pretrain_dataset():
    split_portion = 0.9

    root = "/data5/dataset/Caltech-256"
    path = os.path.join(root, '256_ObjectCategories')  # image path
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

    train_dataset = Caltech256_dataset([X_train, y_train])
    test_dataset = Caltech256_dataset([X_test, y_test])

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
    root = "/data5/dataset/Caltech-256"
    path = os.path.join(root, '256_ObjectCategories')  # image path
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

    train_dataset = Caltech256_dataset([X_train, y_train])
    test_dataset = Caltech256_dataset([X_test, y_test])



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
            label = row[1][:3]
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




class Caltech256_dataset(Dataset):
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

    def __init__(self, data, resize = 84):
        """
        :param data: [X, y]
        :param resize: resize to
        """
        super(Caltech256_dataset, self).__init__()
        self.resize = resize

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                         transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        root = "/data5/dataset/Caltech-256"
        self.path = os.path.join(root, '256_ObjectCategories')  # image path
        self.X = data[0]
        self.y = data[1]
        assert len(self.X) == len(self.y)
        self.len = len(self.y)

        print("Generate Caltech 256 dataset, path : {}, imgs_num : {}".format(root, self.len))

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

