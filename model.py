from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, model_from_json
from keras import layers as KL
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import sys
import matplotlib.pyplot as plt
import numpy as np
import datetime
from data_gen import DataGen
from eval_callback import EvalCallBack
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import os

sys.path.insert(0, "../")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Inceptionv3WithAttention(object):

    def __init__(self, num_classes):
        "nothing"
        self.num_classes = num_classes
        self.IMG_SIZE = (512, 512)

    def build_model(self, show=False):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        base_features = base_model.output
        for layer in base_model.layers:
            layer.trainable = False
        depth = base_model.get_output_shape_at(0)[-1]
        # attention mechanism instead of global average pooling: 各个位置对预测的影响不同
        # x = GlobalAveragePooling2D()(x)
        attention_features = KL.Conv2D(64, kernel_size=(1, 1), activation='relu', name='att1')(base_features)
        attention_features = KL.Conv2D(16, kernel_size=(1, 1), activation='relu', name='att2')(attention_features)
        attention_features = KL.Conv2D(4, kernel_size=(1, 1), activation='relu', name='att3')(attention_features)
        attention_features = KL.Conv2D(1, kernel_size=(1, 1), activation='sigmoid', name='att4')(attention_features)

        up_c2_w = np.ones((1, 1, 1, depth))
        up_c2 = KL.Conv2D(depth, kernel_size=(1, 1), padding='same',
                       activation='linear', use_bias=False, weights=[up_c2_w])
        up_c2.trainable = False
        attention_features = up_c2(attention_features)

        mask_features = KL.multiply([attention_features, base_features])
        gap_features = KL.GlobalAveragePooling2D()(mask_features)
        gap_mask = KL.GlobalAveragePooling2D()(attention_features)

        # 这里就奇怪了,为什么要再除回来呢
        gap = KL.Lambda(lambda x: x[0]/x[1], name='RescaleGAP')([gap_features, gap_mask])
        X = KL.Dropout(0.25)(gap)
        X = KL.Dense(128, activation='relu')(X)
        X = KL.Dropout(0.25)(X)
        out = KL.Dense(self.num_classes, activation='softmax')(X)

        self.model = Model(inputs=base_model.input, output=out)

        if show:
            self.model.summary()

    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)

    def train(self, batch_size, epoches):
        train_dataset = DataGen(self.IMG_SIZE, 5, True)
        train_gen = train_dataset.generator(batch_size, True)

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())

        callbacks_list = [EvalCallBack(), EarlyStopping(monitor='val_loss', mode='min', patience=6),
                          TensorBoard(log_dir='logs/'+TIMESTAMP, batch_size=batch_size, update_freq='epoch')]

        self.model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                            epochs=epoches, callbacks=callbacks_list)

    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):

        self.load_model(model_json, model_weights)
        self.model.compile(optimizer=Adam(lr=5e-4), loss='categorical_crossentropy', metrics=["categorical_accuracy"])

        train_dataset = DataGen(self.IMG_SIZE, 5, True)
        train_gen = train_dataset.generator(batch_size, True)

        model_dir = os.path.dirname(os.path.abspath(model_json))
        print(model_dir, model_json)

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
        callbacks_list = [EvalCallBack(), EarlyStopping(monitor='val_loss', mode='min', patience=6),
                          TensorBoard(log_dir='logs/' + TIMESTAMP, batch_size=batch_size, update_freq='epoch')]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                 initial_epoch=init_epoch, epochs=epochs, callbacks=callbacks_list)

    def eval(self, batch_size):
        val_dataset = DataGen(self.IMG_SIZE, 5, False)
        val_dataset.generator(batch_size)

        print("val data size: ", val_dataset.get_dataset_size())

        Y_pred = []
        X = []
        Y_gt = []
        count = 0
        for X_batch, y_batch in val_dataset.generator(batch_size):

            count += batch_size
            print("count:", count)

            if count > val_dataset.get_dataset_size():
                break

            y_pred = self.model.predict(X_batch)

            y_pred = np.argmax(y_pred, -1)
            y_gt = np.argmax(y_batch, -1)
            Y_pred = np.concatenate([Y_pred, y_pred])
            Y_gt = np.concatenate([Y_gt, y_gt])
        acc = accuracy_score(Y_gt, Y_pred)
        print('Eval Accuracy: %2.2f%%' % acc)
        # sns.heatmap(confusion_matrix(Y_gt, Y_pred),
        #             annot=True, fmt="d", cbar=False, cmap=plt.cm.Blues, vmax=Y_pred.shape[0] // 16)
        # plt.show()
        np_confusion = confusion_matrix(Y_gt, Y_pred)
        np.save('confusion.npy', np_confusion)


if __name__ == '__main__':

    model = Inceptionv3WithAttention(5)

    # # train
    # model.build_model(True)
    # model.train(4, 20)

    model_json = 'checkpoints/net_arch.json'
    model_weights = 'checkpoints/weights_epoch19.h5'

    model.resume_train(48, model_json, model_weights, 20, 25)
    # eval  ok??
    # model.load_model(model_json, model_weights)
    # model.eval(16)