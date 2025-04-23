import os
from PIL import Image
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

    #from timer.Timer import Timer


class DataFrameImage:
    def __init__(self, image: np.ndarray = None):
        self.image = image
        self.grayscale = None
        #self.segmented = None
        pass

class DataManager:
    def __init__(self):
        self.TrainingData = pd.DataFrame()
        self.TestData = pd.DataFrame()

        #individual data frames for the actions.
        self.CallingSet = pd.DataFrame
        self.ClappingSet = pd.DataFrame
        self.CyclingSet  = pd.DataFrame
        self.DancingSet = pd.DataFrame
        self.DrinkingSet = pd.DataFrame
        self.EatingSet = pd.DataFrame
        self.FightingSet = pd.DataFrame
        self.HuggingSet = pd.DataFrame
        self.LaughingSet = pd.DataFrame
        self.Listening_to_musicSet = pd.DataFrame
        self.RunningSet = pd.DataFrame
        self.SittingSet  = pd.DataFrame
        self.SleepingSet = pd.DataFrame
        self.TextingSet = pd.DataFrame
        self.Using_laptopSet = pd.DataFrame
        self.LabelSet = []
        pass

    ############################# data loading methods ###############################################
    def LoadData(self, folder_name, csv_filename, subfolder, num_files=None):
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

        #'calling' 'clapping' 'cycling' 'dancing' 'drinking' 'eating' 'fighting'
        #'hugging' 'laughing' 'listening_to_music' 'running' 'sitting' 'sleeping'
        #'texting' 'using_laptop'

        # Add in a class filter into the datamanager so we can select a list of classes
        # (["sitting","standing",...]). Plan on taking all 840 images of the class
        # so its easier to filter
        #       Data is filtered upon calling LoadData, each set is in it's own self.LabelSet

        # Filter data into individual sets, based on label, size are correct
        self.CallingSet = data[data["label"] == 'calling']
        self.ClappingSet = data[data["label"] == 'clapping']
        self.CyclingSet = data[data["label"] == 'cycling']
        self.DancingSet = data[data["label"] == 'dancing']
        self.DrinkingSet = data[data["label"] == 'drinking']
        self.EatingSet = data[data["label"] == 'eating']
        self.FightingSet = data[data["label"] == 'fighting']
        self.HuggingSet = data[data["label"] == 'hugging']
        self.LaughingSet = data[data["label"] == 'laughing']
        self.Listening_to_musicSet = data[data["label"] == 'listening_to_music']
        self.RunningSet = data[data["label"] == 'running']
        self.SittingSet = data[data["label"] == 'sitting']
        self.SleepingSet = data[data["label"] == 'sleeping']
        self.TextingSet = data[data["label"] == 'texting']
        self.Using_laptopSet = data[data["label"] == 'using_laptop']

        self.LabelSet = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting',
        'hugging', 'laughing', 'listening_to_music', 'running', 'sitting', 'sleeping',
        'texting', 'using_laptop']

        self.LabelDistroSet = [len(self.CallingSet), len(self.ClappingSet), len(self.CyclingSet),
                               len(self.DancingSet),len(self.DrinkingSet), len(self.EatingSet),
                               len(self.FightingSet), len(self.HuggingSet), len(self.LaughingSet),
                               len(self.Listening_to_musicSet), len(self.RunningSet),
                               len(self.SittingSet), len(self.SleepingSet), len(self.TextingSet),
                               len(self.Using_laptopSet)]

        return data

    def ReduceSetTo(self, reduceto: int = None):
        if reduceto is None:
            return

        AmountWeWantToHave = math.floor(reduceto / 15)
        ReduceAmount = len(self.CallingSet) - AmountWeWantToHave

        print(self.LabelDistroSet)

        for elements in self.LabelDistroSet:
            if elements < ReduceAmount:
                print ("Reducing this amount will delete all data points in a certain label,"
                       " try again with a higher value")
                return

        if ReduceAmount < 0:
            print ("Can not reduce by negative amounts")
            return

        # method to reduce the size of the individual sets.
        # N can be changed to increase the number of removed data points
        # 1
        n = ReduceAmount
        drop_indices = self.CallingSet.sample(n).index
        temp_df = self.CallingSet.drop(drop_indices)
        self.CallingSet = temp_df
        print(len(self.CallingSet))

        # 2
        drop_indices = self.ClappingSet.sample(n).index
        temp_df = self.ClappingSet.drop(drop_indices)
        self.ClappingSet = temp_df

        # 3
        drop_indices = self.CyclingSet.sample(n).index
        temp_df = self.CyclingSet.drop(drop_indices)
        self.CyclingSet = temp_df

        # 4
        drop_indices = self.DancingSet.sample(n).index
        temp_df = self.DancingSet.drop(drop_indices)
        self.DancingSet = temp_df

        # 5
        drop_indices = self.DrinkingSet.sample(n).index
        temp_df = self.DrinkingSet.drop(drop_indices)
        self.DrinkingSet = temp_df

        # 6
        drop_indices = self.EatingSet.sample(n).index
        temp_df = self.EatingSet.drop(drop_indices)
        self.EatingSet = temp_df

        # 7
        drop_indices = self.FightingSet.sample(n).index
        temp_df = self.FightingSet.drop(drop_indices)
        self.FightingSet = temp_df

        # 8
        drop_indices = self.HuggingSet.sample(n).index
        temp_df = self.HuggingSet.drop(drop_indices)
        self.HuggingSet = temp_df

        # 9
        drop_indices = self.LaughingSet.sample(n).index
        temp_df = self.LaughingSet.drop(drop_indices)
        self.LaughingSet = temp_df

        # 10
        drop_indices = self.Listening_to_musicSet.sample(n).index
        temp_df = self.Listening_to_musicSet.drop(drop_indices)
        self.Listening_to_musicSet = temp_df

        # 11
        drop_indices = self.RunningSet.sample(n).index
        temp_df = self.RunningSet.drop(drop_indices)
        self.RunningSet = temp_df

        # 12
        drop_indices = self.SittingSet.sample(n).index
        temp_df = self.SittingSet.drop(drop_indices)
        self.SittingSet = temp_df

        # 13
        drop_indices = self.SleepingSet.sample(n).index
        temp_df = self.SleepingSet.drop(drop_indices)
        self.SleepingSet = temp_df

        # 14
        drop_indices = self.TextingSet.sample(n).index
        temp_df = self.TextingSet.drop(drop_indices)
        self.TextingSet = temp_df

        # 15
        drop_indices = self.Using_laptopSet.sample(n).index
        temp_df = self.Using_laptopSet.drop(drop_indices)
        self.Using_laptopSet = temp_df

        self.LabelDistroSet = [len(self.CallingSet), len(self.ClappingSet), len(self.CyclingSet),
                               len(self.DancingSet),len(self.DrinkingSet), len(self.EatingSet),
                               len(self.FightingSet), len(self.HuggingSet), len(self.LaughingSet),
                               len(self.Listening_to_musicSet), len(self.RunningSet),
                               len(self.SittingSet), len(self.SleepingSet), len(self.TextingSet),
                               len(self.Using_laptopSet)]

        print(self.LabelDistroSet)

        newdf = pd.concat([self.CallingSet, self.ClappingSet, self.CyclingSet,
                               self.DancingSet, self.DrinkingSet, self.EatingSet,
                               self.FightingSet, self.HuggingSet, self.LaughingSet,
                               self.Listening_to_musicSet, self.RunningSet,
                               self.SittingSet, self.SleepingSet, self.TextingSet,
                               self.Using_laptopSet], ignore_index=True)
        print(newdf.shape)

        self.TrainingData = newdf

        pass
    
    def LoadTrainingData(self, folderName: str = None, csvFileName: str = None, numFiles: int = None):
        self.TrainingData = self.LoadData(folderName, csvFileName, "train", numFiles)
        pass
    
    ## maybe better to merge this and above together to avoid code duplication
    def LoadTestData(self, folderName: str = None, csvFileName: str = None, numFiles: int = None):
        self.TestData = self.LoadData(folderName, csvFileName, "test", numFiles)
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

    def ResizeImages(self, TargetSize=(160,160)):
        # resize all images to the target size
        for image in self.TrainingData["image"]:
            image.image = np.array(Image.fromarray(image.image).resize(TargetSize))

        pass

    def NormalizeImages(self):
        # need to det which normalization method to use will be best 

        # 1st: /255
        for image in self.TrainingData["image"]:
            image.image = image.image / 255.0

        # 2nd: Z-score Normalization? 
        #for image in self.TrainingData["image"]:
        #    image.image = (image.image - np.mean(image.image)) / np.std(image.image) 

        # 3rd: Min-Max?
        #for image in self.TrainingData["image"]:
        #    image.image = (image.image - np.min(image.image)) / (np.max(image.image) - np.min(image.image))
        pass

    def SegmentImages(self):
        # should we do this here or in the model?
        pass

    ################################### Data visualization Methods #########################################
    def PrintStats(self):
        # print stats of dataset
        print("Training Data Stats:")
        print("Describe Data:")
        #print(self.TrainingData.describe())
        print("\nInfo:")
        #print(self.TrainingData.info())

        # Print label distribution
        label_counts = self.TrainingData['label'].value_counts()
        print("\nLabel Distribution:")
        print(label_counts)

        # potentially plot some stats
        #pie chart
        plt.pie(self.LabelDistroSet, labels=self.LabelSet, autopct='%1.1f%%')
        plt.title("Distribution of Labels in Dataset")

        # 1st: Visualize label distribution
        plt.figure(figsize=(12, 6))
        label_counts.plot(kind='bar')
        plt.title("Distribution of Labels in Dataset")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        #printing distributions

        pass

    def ShowRandomImages(self, numImages: int = 10, showGrayscale: bool = False, showSegmented: bool = False):
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
    dm.LoadTrainingData(folderName=currentFolder+ "//..//data//", csvFileName="Training_set.csv", numFiles=None)
    t.stop()

    #print(dm.TrainingData.head())
    #print(dm.TrainingData.describe())
    #print(dm.TrainingData.info())

    #print("\nTest 2: Print stats...")
    #t.start()
    #dm.PrintStats()
    #t.stop()

    #print("\nTest 3: remove missing data...")
    #t.start()
    #dm.RemoveMissingData()
    #t.stop()

    #print("\nTest 4: Resize images...")
    #t.start()
    #dm.ResizeImages()
    #t.stop()

    #print("\nTest 5: convert images to grayscale...")
    #t.start()
    #dm.ConvertToGrayScale()
    #t.stop()

    #print("\nTest 7: Norm imgs...")
    #t.start()
    #dm.NormalizeImages()
    #t.stop()

    #print("\nTest 8: show random images...")
    #t.start()
    #dm.ShowRandomImages(numImages=5, showGrayscale=False, showSegmented=True)
    #t.stop()

    print("Test 8: reduce list")
    t.start()
    dm.ReduceSetTo(300)
    t.stop()
