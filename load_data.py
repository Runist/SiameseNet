# -*- coding: utf-8 -*-
# @File : load_data.py
# @Author: Runist
# @Time : 2020/5/18 14:59
# @Software: PyCharm
# @Brief: 用脚本对omniglot数据集进行预处理，并将其他们制作成一个pickle文件，方便读取

import sys
import numpy as np
import cv2 as cv
import pickle
import os
import matplotlib.pyplot as plt


data_path = './dataset/'
train_folder = os.path.join(data_path, 'images_background')
valpath = os.path.join(data_path, 'images_evaluation')

save_path = './dataset/'


def loadimgs(path, n=0):
    # 数据没有解压，就先解压
    if not os.path.exists(path):
        print("You must unzip {}.zip.".format(path))
        exit(-1)

    X = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    # 加载每个字母，方便后面分离他们
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)

        # 建立对应的映射关系，各个语言所包含的字母从 多少到多少
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)

        # 每个不同语言又有不同的字母，分开加载
        for letter in os.listdir(alphabet_path):
            # 记录 语言、字母
            cat_dict[curr_y] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)

            # 记录每个字母的不同手写状态
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                category_images.append(image)
                y.append(curr_y)

            # 把一个类别统一的加载到X列表中
            X.append(np.stack(category_images))

            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1

    # 变成(19280, 1)的数据
    y = np.vstack(y)
    # 变成(964, 20, 105, 105, 3)的数据
    X = np.stack(X)

    return X, y, lang_dict


X, _, c = loadimgs(train_folder)
with open(os.path.join(save_path, "train.pickle"), "wb") as f:
    # dumps 将数据通过特殊的形式转换为只有python语言认识的字符串
    pickle.dump((X, c), f)


X, _, c = loadimgs(valpath)
with open(os.path.join(save_path, "valid.pickle"), "wb") as f:
    pickle.dump((X, c), f)
