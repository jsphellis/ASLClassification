'''
Convolution
Creates an instance of a CNN model to analyze an inputted training and testing set, saving model and history for further analysis.
Includes prepocessing functions for data such that it can be analyzed in other locations.
'''

import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import pickle

class Convolution:

    '''
    Convolution constructor creates the datasets, and calls the corresponding functions to define the model and fit it
    '''
    def __init__(self, train, test, batch=32):

        self.model=None

        self.train_dataset, self.validation_dataset, self.test_dataset, self.train_df, self.validation_df, self.test_df = Convolution.datasetCreation(train, test)
        print('Datasets created and processed...\n')
        self.ModelDefine()
        print('Model defined and compiled...\n')
        self.ModelFit()
        print('Model trained and evaluated...\n')

        pass

    '''
    dfCreation
    Creates panda dataframes for the train and test images based on the folders in the dataset.
    These dataframes include the file paths and labels for each image.
    '''
    @staticmethod
    def dfCreation(train, test):
        def imgPaths(filepath):
            labels = [os.path.split(os.path.split(str(p))[0])[1] for p in filepath]

            filepath = pd.Series(filepath, name='Filepath').astype(str)
            labels = pd.Series(labels, name='Label')

            df = pd.concat([filepath, labels], axis=1)
            df = df.sample(frac=1).reset_index(drop=True)
            return df

        train_image_dir = Path(train)
        train_filepaths = list(train_image_dir.glob(r'**/*.jpg'))

        test_image_dir = Path(test)
        test_filepaths = list(test_image_dir.glob(r'**/*.jpg'))

        train_df = imgPaths(train_filepaths)
        test_df = imgPaths(test_filepaths)

        return train_df, test_df
    
    '''
    datasetCreation
    Performs a split of 70-15-15 to create sufficiently large training, validation, and testing datasets.
    These datasets are created using a train and test generator, of which the train generator performs different augmentations for variation.
    '''
    @staticmethod
    def datasetCreation(train, test):
        train_df, test_df = Convolution.dfCreation(train, test)
        train_df, temp_df = train_test_split(train_df, test_size=0.3, random_state=2077, stratify=train_df['Label'])
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=2077, stratify=temp_df['Label'])
        
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,  
            horizontal_flip=True,
            brightness_range=(0.75, 1.3),
            rotation_range=20,
        )

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input, 
        )

        train_dataset = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=True,
            seed=2077,
        )
        validation_dataset = train_generator.flow_from_dataframe(
            dataframe=validation_df,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=True,
            seed=2077,
        )
        test_dataset = test_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col='Filepath',
            y_col='Label',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            seed=2077,
            shuffle=False
        )

        return train_dataset, validation_dataset, test_dataset, train_df, validation_df, test_df

    '''
    ModelDefine
    Defines the base model (ResNet50V2) which is given extra layers.
    These layers include dense, normalization, and dropout techniques to avoid overfitting.
    '''
    def ModelDefine(self):
        baseModel = tf.keras.applications.ResNet50V2(
            input_shape=(224, 224, 3),
            include_top=False, 
            weights='imagenet', 
            pooling='avg'
        )

        baseModel.trainable = False

        inputs = baseModel.input

        x = tf.keras.layers.Dense(128, activation='relu')(baseModel.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x) 
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x) 
        outputs = tf.keras.layers.Dense(29, activation='softmax')(x)

        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

        adam = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        )

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    '''
    ModelFit
    Creates a model checkpoint and early stopping function for the model fitting process.
    This model fitting saves the model with the lowest validation loss.
    The history is saved upon completion, for later use.
    '''
    def ModelFit(self, num_epochs=5):
        model_checkpoint = ModelCheckpoint(
            filepath='models/best_model.h5',  
            monitor='val_loss',
            save_best_only=True, 
            verbose=1,
            mode='min'
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.validation_dataset,
            epochs=num_epochs,
            callbacks=[
                early_stopping,
                model_checkpoint
            ]
        )

        with open('trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        loss0, accuracy0 = self.model.evaluate(self.test_dataset)
        print(f'Test loss: {loss0}')
        print(f'Test accuracy: {accuracy0}')

        pass
