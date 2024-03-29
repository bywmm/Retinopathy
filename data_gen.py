import tensorflow as tf
import numpy as np
import random
from random import shuffle
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy, PIL
import cv2
from PIL import Image


class DataGen(object):

    def __init__(self, inres, num_classes, is_train, mini=True):
        self.inres = inres
        self.num_classes = num_classes
        self.is_train = is_train
        self.base_image_dir = "/home/wangjunfu/dataset/retinopathy/"
        # self.base_image_dir = 'D:/dataset/tangniaobing/'
        if self.is_train:
            self.anno = pd.read_csv(self.base_image_dir+('mini_train.csv' if mini else 'train.csv'))
        else:
            self.anno = pd.read_csv(self.base_image_dir+('mini_valid.csv' if mini else 'valid.csv'))

    def generator(self, batch_size, is_shuffle=False, flip_flag=False,
                  scale_flag=False, rot_flag=False):

        if not self.is_train:
            assert (is_shuffle == False), 'shuffle must be off in val model'
            # assert (rot_flag == False), 'rot_flag must be off in val model'

        X_batch = np.zeros(shape=(batch_size, self.inres[0], self.inres[1], 3), dtype=np.float)
        y_batch = np.zeros(shape=(batch_size, self.num_classes), dtype=np.float)

        while True:
            if is_shuffle:
                self.anno = self.anno.sample(frac=1, replace=False)
            i = 0
            for _, row in self.anno.iterrows():
                _imageaug, _label = self.process_image(row, flip_flag, scale_flag, rot_flag)
                _index = i % batch_size
                X_batch[_index, :, :, :] = _imageaug
                y_batch[_index, :] = to_categorical(_label, self.num_classes)

                if i % batch_size == (batch_size-1):

                    yield X_batch, y_batch
                i = i+1

    def process_image(self, anno, flip_flag, scale_flag, rot_flag):
        image_dir = anno['path']
        label = anno['level']
        image = scipy.misc.imread(image_dir)
        image = np.array(Image.fromarray(image).resize(self.inres))
        if flip_flag:
            image = cv2.flip(image, flipCode=1)
        if rot_flag:  # and random.choice([0, 1]):
            rot = np.random.randint(-1 * 30, 30)
            M = cv2.getRotationMatrix2D((512 / 2, 512 / 2), rot, 1)
            image = cv2.warpAffine(image, M, (512, 512))

        # img = tf.placeholder(shape=(512, 512, 3), dtype=float)
        # img_bri = tf.image.random_brightness(img, max_delta=0.1)
        # img_sat = tf.image.random_saturation(img_bri, lower=0.75, upper=1.5)
        # img_hue = tf.image.random_hue(img_sat, max_delta=0.15)
        # img_con = tf.image.random_contrast(img_hue, lower=0.75, upper=1.5)
        # with tf.Session() as sess:
        #     image = sess.run(img_con, feed_dict={img: image})

        return image, int(label)

    def get_dataset_size(self):
        return len(self.anno)


if __name__ == "__main__":
    print("ok")
    IMG_SIZE = (512, 512)
    train_dataset = DataGen(IMG_SIZE, 5, True)
    train_gen = train_dataset.generator(48, True)
    t_x, t_y = next(train_gen)
    fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
    for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
        c_ax.imshow(np.array(c_x, dtype=int))
        c_ax.set_title('Severity {}'.format(np.argmax(c_y, -1)))
        c_ax.axis('off')
    plt.show()
