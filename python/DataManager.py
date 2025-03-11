import os
from PIL import Image
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

    #from timer.Timer import Timer


class DataFrameImage:
    def __init__(self, image: np.ndarray = None):
        self.image = image
        self.grayscale = None
        pass

class DataManager:
    def __init__(self):
        self.TrainingData = pd.DataFrame()
        pass

    ############################# data loading methods ###############################################
    def LoadTrainingData(self, folderName: str = None, csvFileName: str = None, numFiles: int = None):
        # read image csv
        if numFiles is None:
            self.TrainingData = pd.read_csv(folderName + csvFileName)
        else:
            self.TrainingData = pd.read_csv(folderName + csvFileName, nrows=numFiles)

        # load files into dataframe with a new column for path
        loadedFiles = 0
        missingFiles = 0
        for i in range(len(self.TrainingData)):
            filepath = folderName + "\\train\\" + self.TrainingData.loc[i, "filename"]
            self.TrainingData.loc[i, "path"] = filepath
            try:
                image = DataFrameImage(np.asarray( Image.open(filepath) ))
                loadedFiles += 1
            except FileNotFoundError:
                image = DataFrameImage()
                missingFiles += 1
            self.TrainingData.loc[i, "image"] = image

        print("Loaded ", loadedFiles, " files, ", missingFiles, " files were missing.")



        pass

    def LoadTestData(self, filepath: str = None):
        pass

    def RemoveMissingData(self):
        # remove missing dataframe rows based on the self.TrainingData["image"].image being None type?
        pass


    #################################### Preprocessing Methods #############################################
    def ConvertToGrayScale(self, image: DataFrameImage=None):
        # convert each image to grayscale
        # should this run on individual images or the whole dataset? should the color image be stored inside the
        # dataframeimage class?
        if image is None:
            # perform grayscale on all images in the datasets
            for image in self.TrainingData["image"]:
                image.grayscale = np.mean(image.image[:,:,:], 2)

        else:
            image.grayscale =  np.mean(image.image[:,:,:], 2)

        pass

    ################################### Data visualization Methods #########################################
    def PrintStats(self):
        # print stats of dataset

        # potentially plot some stats

        pass

    def ShowRandomImages(self, numImages: int = 10, showGrayscale: bool = False):
        # show a grid of images, selected at random, with the titles the labels of the image
        # random list of values
        imagesToShow = np.random.randint(0, len(self.TrainingData), numImages)
        nrows = int(np.sqrt(numImages))
        ncols = nrows+1
        print("nrows: ", nrows, "ncols: ", ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
        for i in range(numImages):
            image = self.TrainingData.loc[imagesToShow[i], "image"]
            pix = image.image
            if showGrayscale:
                pix = image.grayscale
            axs[i//ncols, i%ncols].imshow(pix, cmap='gray' if showGrayscale else None)
            axs[i//ncols, i%ncols].set_title(self.TrainingData.loc[imagesToShow[i], "label"])
            axs[i//ncols, i%ncols].axis('off')
        plt.show()


        pass



############################################# debugger ###################################################
# run the DataManager.py file to run below
if __name__ == "__main__":

    sys.path.insert(0, os.getcwd() + '/../timer/')
    from time.Timer import Timer

    t = Timer()

    print("Running DataManager tests")
    print("Test 1: load training data and store in dataframe...")
    dm = DataManager()
    currentFolder = os.getcwd()
    print("Current folder: ", currentFolder)
    t.start()
    dm.LoadTrainingData(folderName=currentFolder+ "\\..\\..\\data\\", csvFileName="Training_set.csv", numFiles=100)
    t.stop()

    print(dm.TrainingData.head())
    print(dm.TrainingData.describe())
    print(dm.TrainingData.info())

    print("Test 2: convert images to grayscale...")
    t.start()
    dm.ConvertToGrayScale()
    t.stop()

    print("Test 3: show random images...")
    t.start()
    dm.ShowRandomImages(numImages=5, showGrayscale=False)
    t.stop()
    
    

