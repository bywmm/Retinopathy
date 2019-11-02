from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import sys
from data_gen import DataGen
sys.path.insert(0, "../")

IMG_SIZE = (512, 512)
BATCH_SIZE = 2

train_dataset = DataGen(IMG_SIZE, 5, True)
train_gen = train_dataset.generator(48, True)
val_dataset = DataGen(IMG_SIZE, 5, True)
val_gen = val_dataset.generator(BATCH_SIZE, True)


base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

callbacks_list = [EarlyStopping(monitor='acc', patience=1),
                  ModelCheckpoint(filepath='out/my_model.h5', monitor='val_loss', save_best_only=True),
                  TensorBoard(log_dir='logs', batch_size=BATCH_SIZE, update_freq='epoch')]

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // BATCH_SIZE,
                    epochs=20, callbacks=callbacks_list, validation_data=val_gen,
                    validation_steps=val_dataset.get_dataset_size() // BATCH_SIZE)

## step two

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // BATCH_SIZE,
                    epochs=20, callbacks=callbacks_list, validation_data=val_gen,
                    validation_steps=val_dataset.get_dataset_size() // BATCH_SIZE)
