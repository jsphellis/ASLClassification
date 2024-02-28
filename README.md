# CNN Analysis - ASL Alphabet

## Code

* CNN.py included functions such that creation of a new object automatically creates, fits, and saves a CNN model
* DataVis.py included a variety of functions to analyze the dataset and model, utilizes dataset processing functions from CNN.py
* dataExploration.ipynb shows the calling of different functions to get different visualizations used in the report 
* CNN.py and DataVis.py code would require the dataset to be downloaded and put into structure as described in following 'Data' section

## Data

* I utilized the ASL Alphabet dataset from Kaggle:
    * https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data
    * I extracted it as a folder titled 'ASL'
    * The subdirectories inside are titled 'asl_alphabet_test' and 'asl_alphabet_train' and are referred to as such in any code
        * The subdirectories initially included code within another subdirectory, this was changed such that the folder structure was:
        ASL -> asl_alphabet_test, asl_alphabet_train

