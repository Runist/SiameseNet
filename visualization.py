# -*- coding: utf-8 -*-
# @File : visualization.py
# @Author: Runist
# @Time : 2020/5/19 14:50
# @Software: PyCharm
# @Brief: 样本数据、训练过程可视化
import os
import re
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv

from dataReader import SiameseLoader
from main import siamese_network
import config as cfg


def gen_class_names(base_class_name):
    """
    生成分类名字的列表
    :param base_class_name: 分类名的前缀
    :return:
    """
    classes = []
    for i in range(1, 21):
        if i < 10:
            classes.append("{}0{}".format(base_class_name, i))
        else:
            classes.append("{}{}".format(base_class_name, i))
    return classes


def generate_one_hot_encoding(classes):
    """
    利用sklearn中的preprocessing包快速生成onehot编码
    :param classes: 分类的名字
    :return: 生成的onehot编码列表
    """
    encoder = LabelBinarizer()
    transfomed_labels = encoder.fit_transform(classes)

    return transfomed_labels


def plot_images(path):
    """
    绘制一种语言的某个字符的所有20个样本
    """
    f, axarr = plt.subplots(5, 4, figsize=(10, 10))
    images_list = []
    for image in os.listdir(path):
        image_path = os.path.join(path, image)
        img = cv.imread(image_path)
        images_list.append(img)

    for i in range(5):
        for j in range(4):
            axarr[i, j].imshow(images_list.pop())

    plt.show()


def nearest_neighbour_correct(image_group, labels):
    """
    计算L2距离，以欧式距离作为衡量标准
    :param image_group: 测试集
    :param labels:
    :return:
    """
    L2_distances = np.zeros_like(labels)
    for i in range(len(labels)):
        L2_distances[i] = np.sum(np.sqrt(image_group[0][i]**2 - image_group[1][i]**2))

    if np.argmin(L2_distances) == np.argmax(labels):
        return 1
    return 0


def test_nn_accuracy(N_ways, num, loader):
    """
    测试one shot的准确率
    :param N_ways: 测试的种类
    :param num: 测试的数量
    :param loader: 数据加载器
    :return: 准确率
    """
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(num, N_ways))

    correct = 0

    for i in range(num):
        image_group, labels = loader.make_oneshot_task(N_ways, "valid")
        correct += nearest_neighbour_correct(image_group, labels)

    return 100.0 * correct / num


def test_nn_and_siamese(weight_path, curves_path):
    """
    测试knn和孪生网络在测试集上的表现
    :param weight_path: 模型路径
    :param curves_path: 曲线数据存储路径
    :return:
    """
    ways = np.arange(1, 30, 2)
    valid_accs, train_accs, nn_accs = [], [], []
    test_num = 2

    another_strategy = tf.distribute.MirroredStrategy()
    with another_strategy.scope():
        model = siamese_network()
        model.load_weights(weight_path)

        for N in ways:
            train_accs.append(loader.test_oneshot(model, N, test_num, "train"))
            valid_accs.append(loader.test_oneshot(model, N, test_num, "valid"))
            nn_accs.append(test_nn_accuracy(N, test_num, loader))

    # 把数据保存下来，服务器上不好显示，在本机显示图像
    with open(curves_path, 'w') as f:
        f.write("[train_acc]\n")
        for acc in train_accs:
            f.write("{:.2f},".format(acc))

        f.write("\n[valid_accs]\n")
        for acc in valid_accs:
            f.write("{:.2f},".format(acc))

        f.write("\n[nn_accs]\n")
        for acc in nn_accs:
            f.write("{:.2f},".format(acc))


def randomcolor():
    """
    随机生成颜色
    :return:
    """
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = "#"
    for i in range(6):
        color += colorArr[np.random.randint(0, 14)]
    return color


def plot_curves(file):
    """
    根据txt文件来生成图标，可视化显示结果
    :param file: txt文件
    :return: None
    """
    with open(file, 'r') as f:
        content = f.read()
        name = re.findall(r"([a-z_a-z]+)", content)
        data = re.findall(r"(.*)[\d]", content)

    history = {}
    for i in range(len(data)):
        # 按照,分割数字后，映射成浮点数
        history[name[i]] = list(map(float, data[i].split(",")))

    for key, values in history.items():
        plt.plot(range(len(values)), values, color=randomcolor())

    plt.legend(history.keys(), loc=0)
    plt.show()


if __name__ == '__main__':
    loader = SiameseLoader(cfg.data_path)

    # plot_images(os.path.join(cfg.data_path, 'images_background/Arcadian/character03/'))

    test_curvesh = cfg.summary_path + "test_curves.txt"
    test_nn_and_siamese(cfg.model_path, test_curvesh)

    # train_curves = cfg.summary_path + "test_curves.txt"
    # plot_curves(train_curves)
