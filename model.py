import os
import csv
import cv2
import numpy as np
import sklearn
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math as m

def import_date(directory = "Data_Collection Challenge 2"):
    """
    This function imports all csv file in directory and extracts the path for the images as well as the steering
    wheel angels
    :param directory: directory with the generated data
    :return: list of training and validation samples
    """
    files = os.listdir(directory)
    data_files = [file for file in files if file[-3:] == 'csv']
    samples = []
    take_every_xth_image = 1
    correction = 0.04
    for data_file in data_files:
        lines = []
        cnt = 0
        with open(directory + "/" + data_file) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                lines.append(line)
        for line in lines:
            cnt += 1
            if cnt % take_every_xth_image == 0 or data_file[-8:-4] == 'rare':
                steering_center = float(line[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                source_path_center = line[0]
                source_path_left = line[1]
                source_path_right = line[2]
                filename_center = source_path_center.split('\\')[-1].split('/')[-1]
                filename_left = source_path_left.split('\\')[-1].split('/')[-1]
                filename_right = source_path_right.split('\\')[-1].split('/')[-1]
                current_path_center = directory + '/' + data_file[:-4] + '/' + filename_center
                current_path_left = directory + '/' + data_file[:-4] + '/' + filename_left
                current_path_right = directory + '/' + data_file[:-4] + '/' + filename_right
                sample = [current_path_center, current_path_left, current_path_right, steering_center, steering_left,
                          steering_right]
                samples.append(sample)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def generator(samples, batch_size=32):
    """
    This function generates batch data to avoid memory issues while training
    :param samples: list of samples, each sample includes image paths and steering values
    :param batch_size: number of samples in batch
    :yield: batch of data
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

                left_image = cv2.imread(batch_sample[1])
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

                right_image = cv2.imread(batch_sample[2])
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                center_angle = batch_sample[3]
                left_angle = batch_sample[4]
                right_angle = batch_sample[5]
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def train_MyNet(train_samples, validation_samples, train_generator, validation_generator):
    """
    This function includes the architecture of the network as well as the training method
    :param train_samples: list of samples, each sample includes image paths and steering values, for training
    :param validation_samples: list of samples, each sample includes image paths and steering values, for validation
    :param train_generator: generator object to give training batches
    :param validation_generator: generator object to give validation batches
    :return: saves trained model and plots training and validation loss over number of epochs
    """
    drop_rate = 0.05
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(12, 5, 5, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(rate=drop_rate))
    model.add(Convolution2D(12, 5, 5, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(rate=drop_rate))
    model.add(Convolution2D(12, 3, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(rate=drop_rate))
    model.add(Convolution2D(12, 3, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(rate=drop_rate))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(25))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=drop_rate))
    model.add(Dense(1))
    # NO ACTIVATION FUNCTION BECAUSE REGRESSION NETWORK


    model.compile(loss='mse', optimizer='adam')
    #history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)
    history_object = model.fit_generator(train_generator, steps_per_epoch=m.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=m.ceil(len(validation_samples)/batch_size), epochs=12, verbose=1)
    model.save('model.h5')
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set'], loc='upper right')
    plt.show()
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['validation set'], loc='upper right')
    plt.show()



if __name__ == '__main__':
    # Getting training and validation data from directory
    train_samples, validation_samples = import_date(directory = "Data_Collection Challenge 2")

    # Set our batch size
    batch_size=32

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    train_MyNet(train_samples, validation_samples, train_generator, validation_generator)


