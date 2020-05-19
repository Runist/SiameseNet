# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/5/18 14:37
# @Software: PyCharm
# @Brief:
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import cv2 as cv
import pickle
import numpy as np
# from load_data import *


class SiameseLoader:
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, path, data_subsets=None):
        if data_subsets is None:
            data_subsets = ["train", "valid"]
        self.data = {}
        self.categories = {}
        self.info = {}

        for name in data_subsets:
            # 从pickle文件中加载数据集
            file_path = os.path.join(path, name + ".pickle")
            print("loading data from {}".format(file_path))
            with open(file_path, "rb") as f:
                (X, c) = pickle.load(f)
                # 将data和categories 都分为 train、val然后根据写入图片、分类
                self.data[name] = X
                self.categories[name] = c

        self.train_num = self.data['train'].shape[0]
        self.valid_num = self.data['valid'].shape[0]

    def get_batch(self, batch_size, mode="train"):
        """
        创建n对的批次，一半相同的类，一半不同的类
        :param batch_size:
        :param mode:
        :return:
        """
        X = self.data[mode]
        n_classes, n_examples, w, h = X.shape

        # 随机采样几个类以用于批处理
        categories = np.random.choice(n_classes, size=(batch_size,), replace=False)
        # 为输入图像批处理初始化2个空数组，分别存网络输入的两张图片
        image_group = [np.zeros((batch_size, h, w, 1)) for _ in range(2)]
        # 初始化目标的向量，并将后一半数据设为“1”，代表批处理的后一半是相同的类输入到网络里
        labels = np.zeros((batch_size,))
        labels[batch_size // 2:] = 1

        # 遍历batch_size次，前半段给image_group[0]、image_group[1]存上不同类别的图片，后半段存上相同类别的图片
        for i in range(batch_size):
            category_1 = categories[i]

            # 随机生成一个 0-20 id
            idx_1 = np.random.randint(0, n_examples)
            # 取 该语言下 的 0-20中任意一张图片
            image_group[0][i, :, :, :] = X[category_1, idx_1].reshape(w, h, 1)
            # 再随机取一个id，作为第二张图片
            idx_2 = np.random.randint(0, n_examples)

            if i >= batch_size // 2:
                # 如果i > batch_size // 2，我们就给网络输入相同类别、但不同手写的图片
                category_2 = category_1
            else:
                # 如果是< batch_size // 2，就给他随便加上一个数字，对20求余，使它与category_1不等
                category_2 = (category_1 + np.random.randint(1, n_classes)) % n_classes
            image_group[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

        return image_group, labels

    def generate(self, batch_size, mode="train"):
        """
        写一个用生成器，方便后面model.fit_generator读取。
        TODO:tf.data来写
        :param batch_size:
        :param mode:
        :return:
        """
        while True:
            image_group, labels = self.get_batch(batch_size, mode)
            yield image_group, labels

    def make_oneshot_task(self, N, mode="valid", language=None):
        """
        创建两张的测试图像（从验证集中随机选取），用于充当support set 进行N-way one-shot的测试
        :param N: 分类个数
        :param mode: train or valid
        :param language: 具体哪个语言
        :return:
        """
        X = self.data[mode]
        n_classes, n_examples, w, h = X.shape
        # 随机取N个 0-20的数字，作为索引
        indices = np.random.randint(0, n_examples, size=(N,))

        if language is not None:
            # 读取这个语言的 始末索引
            low, high = self.categories[mode][language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = np.random.choice(range(low, high), size=(N,), replace=False)

        else:
            # 如果未指定语言，就完全随机选，有可能完全是不同语言的，有可能是相同语言不同字母
            categories = np.random.choice(range(n_classes), size=(N,), replace=False)

        # 设置第一个是对的
        true_category = categories[0]
        ex1, ex2 = np.random.choice(n_examples, replace=False, size=(2,))
        # 生成N 张一样的照片放进test_image中
        test_image = np.asarray([X[true_category, ex1, :, :]] * N, dtype=np.float32).reshape(N, w, h, 1)

        # 制作support set
        support_set = X[categories, indices, :, :]
        # 第1个元素设置成ex2的图片
        support_set[0] = X[true_category, ex2]
        support_set = support_set.reshape(N, w, h, 1)
        support_set = support_set.astype(np.float32)

        # 设置标签，第一个是对的，其他是错的
        labels = np.zeros((N,))
        labels[0] = 1

        # 第一个值作为稀疏矩阵，第二个值作为压缩矩阵传入。
        # sklearn的shuffle函数：第三个参数 相当于 第一个参数的索引，打乱的时候索引和数据一起改变
        labels, test_image, support_set = shuffle(labels, test_image, support_set)
        image_group = [test_image, support_set]

        return image_group, labels

    def test_oneshot(self, model, N, num, verbose=True):
        """
        测试孪生网络在 num N个类别一张图片的下的准确性
        :param model: 模型
        :param N: N类别
        :param num: num次
        :param verbose: 打印提示语
        :return:
        """
        n_correct = 0
        if verbose:
            print("Evaluating logs on {} unique {} way one-shot learning tasks ...".format(num, N))
        for i in range(num):
            inputs, labels = self.make_oneshot_task(N)
            probs = model.predict(inputs)
            # 相同的图片索引预测出来的索引最大值也 等于 标签的最大值索引
            if np.argmax(probs) == np.argmax(labels):
                n_correct += 1

        percent_correct = (100.0 * n_correct / num)
        if verbose:
            print("Got an average of {:.2f}% {} way one-shot learning accuracy".format(percent_correct, N))

        return percent_correct



