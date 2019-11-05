import keras
import os
import numpy as np
from data_gen import DataGen
from sklearn.metrics import accuracy_score, classification_report


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, out_name):
        self.IMG_SIZE = (512, 512)
        self.out_name = out_name

    def run_eval(self, epoch, batch_size=16):
        val_dataset = DataGen(self.IMG_SIZE, 5, False)
        val_gen = val_dataset.generator(batch_size)

        count = 0
        Y_pred = []
        Y_gt = []
        for X_batch, y_batch in val_gen:

            count += batch_size
            if count > val_dataset.get_dataset_size():
                break

            y_pred = self.model.predict(X_batch)

            y_pred = np.argmax(y_pred, -1)
            y_gt = np.argmax(y_batch, -1)
            Y_pred = np.concatenate([Y_pred, y_pred])
            Y_gt = np.concatenate([Y_gt, y_gt])
        acc = accuracy_score(Y_gt, Y_pred)
        print('Eval Accuracy: %2.2f%%' % acc, '@ Epoch ', epoch)
        if (epoch+1) % 10 == 0:
            print(classification_report(Y_gt, Y_pred))

        with open('checkpoints/'+self.out_name+'/val.txt', 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

    def on_epoch_end(self, epoch, logs=None):
        # This is a walkaround to sovle model.save() issue
        # in which large network can't be saved due to size.

        # save model to json
        if epoch == 0:
            jsonfile = "checkpoints/"+self.out_name+"/net_arch.json"
            with open(jsonfile, 'w') as f:
                f.write(self.model.to_json())

        # save weights
        modelName = "checkpoints/"+self.out_name+"/weights_epoch" + str(epoch) + ".h5"
        self.model.save_weights(modelName)

        print("Saving model to ", modelName)
        self.run_eval(epoch)