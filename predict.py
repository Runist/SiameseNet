# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2020/5/20 10:23
# @Software: PyCharm
# @Brief: 模型预测

import tensorflow as tf
from main import siamese_network
import config as cfg
import cv2 as cv
import numpy as np
import os


def main():
    left_path = "./dataset/images_background/Arcadian/character07/0007_11.png"
    right_path = "./dataset/images_background/Arcadian/character07/0007_03.png"
    font = cv.FONT_HERSHEY_SIMPLEX

    left_img = cv.imread(left_path, cv.IMREAD_GRAYSCALE)
    right_img = cv.imread(right_path, cv.IMREAD_GRAYSCALE)
    image = np.concatenate((left_img, right_img), axis=1)

    model = siamese_network()
    model.load_weights(cfg.model_path)
    left_img = tf.expand_dims(left_img, axis=0)
    left_img = tf.expand_dims(left_img, axis=-1)
    right_img = tf.expand_dims(right_img, axis=0)
    right_img = tf.expand_dims(right_img, axis=-1)

    left_img = tf.cast(left_img, tf.float32)
    right_img = tf.cast(right_img, tf.float32)

    result = model.predict([left_img, right_img])

    if result > cfg.similar_threshold:
        cv.putText(image, "True", (8, 18), font, 0.7, (0, 0, 0), 1)
    else:
        cv.putText(image, "False", (8, 18), font, 0.7, (0, 0, 0), 1)

    print("similarity: {:.2f}%".format(result[0][0] * 100))
    cv.imshow("result", image)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
