# import tensorflow as tf

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import json

from i2c import does_dir_exist, does_file_exist,  make_classes_info, prepare_nn_dataset_raw, process_raw_dataset


# def main_nn():
#
#     model = Sequential([
#         Dense(32, input_shape=(784,)),
#         Activation('relu'),
#         Dense(10),
#         Activation('softmax'),
#     ])
#
#     # For a multi-class classification problem
#     model.compile(optimizer='rmsprop',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])


def main_test_nn():
    dataset_id = '003'
    results_dir_path = 'imgs/results/results_' + dataset_id + '/'

    num_instances_per_class = 10000
    train_validate_ratio = 0.8

    dataset_path = results_dir_path + 'dataset/'
    classes_path = dataset_path + 'classes/'
    class_names_path = dataset_path + 'classes.txt'
    data_path = dataset_path + 'data/'
    correct_classes_path = dataset_path + 'roots.txt'

    if not does_dir_exist(classes_path):
        imgs_path = dataset_path + 'imgs/'
        img_filenames_path = dataset_path + 'imgs.txt'
        prepare_nn_dataset_raw(imgs_path, classes_path, img_filenames_path, correct_classes_path)

    if not does_file_exist(class_names_path):
        make_classes_info(classes_path, class_names_path, correct_classes_path)

    if not does_dir_exist(data_path):
        with open(class_names_path, 'r') as f:
            classes_info = json.loads(f.read())
            process_raw_dataset(data_path, classes_info, num_instances_per_class, train_validate_ratio)

    print("Train & validation data are ready.")

    train_datagen = ImageDataGenerator()
    valid_datagen = ImageDataGenerator()

    img_width, img_height = 32, 32  # TODO

    nb_train_samples = 40000  # TODO
    nb_validation_samples = 10000  # TODO
    epochs = 50
    batch_size = 16

    num_classes = 5  # TODO !!! count automatically

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    train_generator = train_datagen.flow_from_directory(
        data_path + 'train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        data_path + 'validation',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights(results_dir_path + 'model.h5')
