'''
DataVisualization
Includes a series of functions to assess the dataset and model ability.
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from CNN import Convolution as CN

class DataVisualization:

    '''
    The constructor for the DataVisualization class includes setting the parent folder, along with the datasets and their corresponding pandas dataframe.
    The model and history locations are also initialized
    '''
    def __init__(self, parent, train, test, modelLoc, historyLoc):
        
        self.parent = parent
        self.train_dataset, self.validation_dataset, self.test_dataset, self.train_df, self.validation_df, self.test_df = CN.datasetCreation(train, test)
        self.model= tf.keras.models.load_model(modelLoc)
        self.historyLoc = historyLoc

        pass

    '''
    ImageExamples
    Shows examples of random images from the dataset to help explain the data
    '''
    def ImageExamples(self, num):
        samples = self.train_df.sample(n=num, random_state=1)

        for _, row in samples.iterrows():
            img = Image.open(row['Filepath'])
            plt.imshow(img)
            plt.title(f"Class: {row['Label']}")
            plt.show()

        pass

    '''
    modelScores
    Displays the loss and accuracy of the model on the testing dataset
    '''
    def modelScores(self):
        loss0, accuracy0 = self.model.evaluate(self.test_dataset)
        print(f'Test loss: {loss0}')
        print(f'Test accuracy: {accuracy0}')

        pass

    '''
    AccLoss
    Loads in the history object then creates a plot depicting the accuracy, validation accuracy, loss, and validation loss over the epochs
    '''
    def AccLoss(self):
        with open(self.historyLoc, 'rb') as file:
            loaded_history = pickle.load(file)

        acc = loaded_history['accuracy']
        val_acc = loaded_history['val_accuracy']

        loss = loaded_history['loss']
        val_loss = loaded_history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.ylim([min(plt.ylim()),1])
        plt.savefig('accuracy_loss.png')
        plt.show()

        pass

    '''
    basicSummary
    Displays the layers of the model
    '''
    def basicSummary(self):
        self.model.summary()

        pass

    '''
    predFolder
    Takes in a folder of images, predicts the labels corresponding with images, and produces a string of predicted labels
    '''
    def predFolder(self, folder_path):

        predictions = []
        file_names = sorted(os.listdir(folder_path))
        images = []

        for file_name in file_names:
            img_path = os.path.join(folder_path, file_name)
            
            img = load_img(img_path, target_size=(224, 224))
            images.append(img)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            pred = self.model.predict(img_array, verbose=0)
            pred_class = np.argmax(pred, axis=1)

            index_to_label_map = {v: k for k, v in self.train_dataset.class_indices.items()}
            predicted_label = index_to_label_map[pred_class[0]]

            predictions.append((predicted_label, np.max(pred)))

        fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
        for i, (image, (label, conf)) in enumerate(zip(images, predictions)):
            axs[i].imshow(image)
            axs[i].set_title(f"{label}\nConf: {conf:.2f}")
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()
    
    '''
    iterOcclusion
    Pads and preprocesses images for inclusion in heatmap generation
    '''
    def iterOcclusion(self, image, size=8):

        occlusion = np.full((size, size, 3), 0.5, np.float32)
        occlusion_padding = size * 2

        image_padded = np.pad(image, (
            (occlusion_padding, occlusion_padding),
            (occlusion_padding, occlusion_padding),
            (0, 0)), 'constant', constant_values=0.0)

        for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):
            for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
                tmp = image_padded.copy()
                tmp[y:y + size, x:x + size] = occlusion
                yield x - occlusion_padding, y - occlusion_padding, tmp[occlusion_padding:-occlusion_padding, occlusion_padding:-occlusion_padding]

    '''
    genOcclusionHeatmap
    Utilizes the iterOcclusion function and model to generate a heatmap of the image
    '''
    def genOcclusionHeatmap(self, image, model, correct_class, size=8):
        heatmap = np.zeros((image.shape[0], image.shape[1]), np.float32)
        for x, y, occluded_image in self.iterOcclusion(image, size=size):

            preds = model.predict(np.expand_dims(occluded_image, axis=0), verbose=0)
           
            heatmap[y:y+size, x:x+size] = preds[0][correct_class]
        plt.figure(figsize=(8, 8)) 
        plt.imshow(heatmap, cmap='hot', interpolation='nearest') 
        plt.colorbar() 
        plt.savefig('heatmap.png')
        plt.show()         

    '''
    occlusionMap
    Selects an image and its label to be utilized in the heatmap generation
    '''
    def occlusionMap(self):

        batch_images, batch_labels = next(self.test_dataset) 

        image = batch_images[0]
        label = batch_labels[0]
        correct_class_index = np.argmax(label)

        heat = self.genOcclusionHeatmap(image, self.model, correct_class_index)


