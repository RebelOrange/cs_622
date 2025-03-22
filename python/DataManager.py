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
        self.TestData = pd.DataFrame()
        pass

    ############################# data loading methods ###############################################
    def load_data(self, folder_name, csv_filename, subfolder, num_files=None):
        # Read image csv
        if num_files is None:
            data = pd.read_csv(folder_name + csv_filename)
        else:
            data = pd.read_csv(folder_name + csv_filename, nrows=num_files)

        # Load files into dataframe with a new column for path
        loaded_files = 0
        missing_files = 0
        for i in range(len(data)):
            filepath = folder_name + "//" + subfolder + "//" + data.loc[i, "filename"]
            data.loc[i, "path"] = filepath
            try:
                image = DataFrameImage(np.asarray(Image.open(filepath)))
                loaded_files += 1
            except FileNotFoundError:
                image = DataFrameImage()
                missing_files += 1
            data.loc[i, "image"] = image

        print(f"Loaded {loaded_files} files, {missing_files} files were missing.")
    
        return data
    
    def LoadTrainingData(self, folderName: str = None, csvFileName: str = None, numFiles: int = None):
        self.TrainingData = self.load_data(folderName, csvFileName, "train", numFiles)
        pass
    
    ## maybe better to merge this and above together to avoid code duplication
    def LoadTestData(self, folderName: str = None, csvFileName: str = None, numFiles: int = None):
        self.TestData = self.load_data(folderName, csvFileName, "test", numFiles)
        pass

    def RemoveMissingData(self):
        # remove missing dataframe rows based on the self.TrainingData["image"].image being None type? and maybe when 
        # img is null to cover all cases
        initial_count = len(self.TrainingData)
        
        # 1st case: null 
        self.TrainingData = self.TrainingData[self.TrainingData["image"].notnull()]

        #2nd case: None
        valid_rows = [img.image is not None for img in self.TrainingData["image"]]
        self.TrainingData = self.TrainingData[valid_rows]

        removed_count = initial_count - len(self.TrainingData)
        print(f"Removed {removed_count} rows with missing images. Remaining: {len(self.TrainingData)} rows.")

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
        print("Training Data Stats:")
        print("Describe Data:")
        print(self.TrainingData.describe())
        print("\nInfo:")
        print(self.TrainingData.info())

        # Print label distribution
        label_counts = self.TrainingData['label'].value_counts()
        print("\nLabel Distribution:")
        print(label_counts)

        # potentially plot some stats

        # 1st: Visualize label distribution
        plt.figure(figsize=(12, 6))
        label_counts.plot(kind='bar')
        plt.title("Distribution of Labels in Dataset")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

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

    from Timer import Timer

    t = Timer()

    print("Running DataManager tests")
    print("Test 1: load training data and store in dataframe...")
    dm = DataManager()
    currentFolder = os.getcwd()
    print("Current folder: ", currentFolder)
    t.start()
    dm.LoadTrainingData(folderName=currentFolder+ "//..//data//", csvFileName="Training_set.csv", numFiles=1000)
    t.stop()

    #print(dm.TrainingData.head())
    #print(dm.TrainingData.describe())
    #print(dm.TrainingData.info())

    print("Test 1.5: Print stats...")
    t.start()
    dm.PrintStats()
    t.stop()

    print("Test 2: convert images to grayscale...")
    t.start()
    dm.ConvertToGrayScale()
    t.stop()

    print("Test 3: show random images...")
    t.start()
    dm.ShowRandomImages(numImages=5, showGrayscale=True)
    t.stop()

    print("Test 4: remove missing data...")
    t.start()
    dm.RemoveMissingData()
    t.stop()


