# -*- coding: utf-8 -*-
# @File : main.py
# @Author: Runist
# @Time : 2020/5/18 10:25
# @Software: PyCharm
# @Brief: 孪生网络


from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import initializers

import tensorflow as tf
import numpy as np
import os
import time
from dataReader import SiameseLoader


def siamese_network(input_shape=(105, 105, 1), lr=0.00006):
    """
    孪生网络
    :param input_shape: 输入的shape
    :param lr: 学习率
    :return:
    """
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # 均值为0，标准差为0.01的正态分布
    w_init = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    b_init = initializers.RandomNormal(mean=0.5, stddev=0.01, seed=None)

    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,   # shape(n, 64, 94, 94)
                     kernel_initializer=w_init, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',  # shape(n, 128, 88, 88)
                     kernel_initializer=w_init, kernel_regularizer=l2(2e-4), bias_initializer=b_init))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu',  # shape(n, 128, 85, 85)
                     kernel_initializer=w_init, kernel_regularizer=l2(2e-4), bias_initializer=b_init))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu',  # shape(n, 256, 82, 82)
                     kernel_initializer=w_init, kernel_regularizer=l2(2e-4), bias_initializer=b_init))
    model.add(Flatten())  # shape(n, 1721344)
    model.add(Dense(4096, activation="sigmoid",
              kernel_initializer=w_init, kernel_regularizer=l2(1e-3), bias_initializer=b_init))

    # 输出两个编码向量
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # 合并两个编码的L1距离
    l1_layer = Lambda(lambda x: tf.abs(x[0] - x[1]))
    l1_distance = l1_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(l1_distance)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    optimizer = Adam(lr=lr)

    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)
    # siamese_net.summary()

    return siamese_net


def train_on_low_level(model, loader, batch_size, weights_path, n_way, num):
    """
    以底层的方式训练 且 验证
    :param model: 模型架构
    :param loader: 数据读取器
    :param batch_size:
    :param weights_path: 模型保存位置
    :param n_way: 测试的类别
    :param num: 测试的数量
    :return:
    """
    best_acc = 0
    evaluate_every = 10
    loss_every = 20
    n_iter = 2000

    start_time = time.time()
    for i in range(1, n_iter):
        inputs, labels = loader.get_batch(batch_size)
        # 在需要对每个batch都进行处理的时候，就用train_on_batch
        loss = model.train_on_batch(inputs, labels)

        print("Loss: {:.4f}".format(loss))
        if i % evaluate_every == 0:
            print("Time for {} iterations: {:.2f}s".format(i, time.time() - start_time))

            valid_acc = loader.test_oneshot(model, n_way, num, verbose=True)
            if valid_acc >= best_acc:
                print("Current best: {0}, previous best: {1}".format(valid_acc, best_acc))
                print("Saving weights to: {0} \n".format(weights_path))
                model.save_weights(weights_path)
                best_acc = valid_acc

        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i, loss))


def train_by_generator(model, loader, batch_size, epochs, weight_path):
    """
    用fit_generator来训练
    :param model: 模型架构
    :param loader: 数据读取器
    :param batch_size:
    :param epochs:
    :param weight_path: 模型保存路径
    :return:
    """
    model.fit_generator(loader.generate(batch_size, 'train'),
                        steps_per_epoch=max(1, loader.train_num // batch_size),
                        validation_data=loader.generate(batch_size, 'valid'),
                        validation_steps=max(1, loader.valid_num // batch_size),
                        epochs=epochs
                        )

    model.save_weights(weight_path)


def main():
    INPUT_SHAPE = (105, 105, 1)
    LEARNING_RATE = 0.00006
    BATCH_SIZE = 32
    EPOCHS = 50
    data_path = './dataset'
    weight_path = "./logs/model/few_shot_learning.h5"
    train_mode = "on_batch"

    logs_path = os.path.split(weight_path)[0]
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    model = siamese_network(INPUT_SHAPE, LEARNING_RATE)
    loader = SiameseLoader(data_path)

    if train_mode == "generator":
        train_by_generatn_itersor(model, loader, BATCH_SIZE, EPOCHS, weight_path)
    else:
        train_on_low_level(model, loader, BATCH_SIZE, weight_path, 20, 250)


if __name__ == '__main__':
    main()
