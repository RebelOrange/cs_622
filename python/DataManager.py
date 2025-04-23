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
        #self.segmented = None
        pass

class DataManager:
    def __init__(self):
        self.TrainingData = pd.DataFrame()
        self.TestData = pd.DataFrame()
        self.KTrainingData = pd.DataFrame()
        self.KTestData = pd.DataFrame()
        pass

    ############################# data loading methods ###############################################
    def LoadData(self, folder_name, csv_filename, subfolder, num_files=None, classFilter: list[str] = None):
        # Read image csv
        data = pd.read_csv(folder_name + csv_filename)

        if classFilter is not None:
            data = data[data["label"].isin(classFilter)]
            print(f"Filtered data to only include classes: {classFilter}")
            print(f"Unique classes: {data['label'].unique()}")

        # select num_files of each class
        if num_files is None:
            pass
        else:
            filtered_df = pd.DataFrame()
            for label in data["label"].unique():
               filtered_df = pd.concat([filtered_df, data[data["label"] == label].head(num_files)])

            del data
            data = filtered_df
            del filtered_df

        # Load files into dataframe with a new column for path
        loaded_files = 0
        missing_files = 0
        for n in range(len(data)):
            i = data.index[n]
            filepath = folder_name + "//" + subfolder + "//" + data.loc[i, "filename"]
            data.loc[i, "path"] = filepath
            try:
                image = DataFrameImage(np.asarray(Image.open(filepath)))
                loaded_files += 1
            except FileNotFoundError:
                image = DataFrameImage()
                missing_files += 1
            data.loc[i, "image"] = image
            if n % 1000 == 0:
                print(f"Loaded {i} files...")

        print(f"Loaded {loaded_files} files, {missing_files} files were missing.")
    
        return data
    
    def LoadTrainAndTestData(self, folderName: str = None, csvFileName: str = None, numFiles: int = None, classFilter: list[str] = None, split: float = 0.8):
        data = self.LoadData(folderName, csvFileName, "train", numFiles, classFilter=classFilter)
        # get length of data
        dummyLabel = data["label"].iloc[0]
        dataLen = len(data[data["label"] == dummyLabel])
        print("There are ", dataLen, " images per class.")

        splitIndex = int(dataLen * split)
        print(f"Splitting data at index: {splitIndex} ({split})")

        for label in data["label"].unique():
            self.TrainingData = pd.concat([self.TrainingData, data[data["label"] == label].head(splitIndex)])
            self.TestData = pd.concat([self.TestData, data[data["label"]==label].tail(dataLen-splitIndex)])

        self.data = data
        self.TestData.reset_index(drop=True, inplace=True)
        self.TrainingData.reset_index(drop=True, inplace=True)
        pass

    def SplitKFold(self, k=5, foldIndex=0):
        dummyLabel = self.TrainingData["label"].iloc[0]
        dataLen = len(self.TrainingData[self.TrainingData["label"] == dummyLabel])
        print("There are ", dataLen, " images per class.")

        splitSize = 1/k
        splitSizeIndex = int(dataLen * splitSize)
        print(f"Splitting data into {k} folds, each with {splitSizeIndex} images per class.")
        del self.KTrainingData
        del self.KTestData
        self.KTrainingData = pd.DataFrame()
        self.KTestData = pd.DataFrame()
        for i in range(k):
            mask = range(i*splitSizeIndex, (i+1)*splitSizeIndex)
            if i == foldIndex:
                for label in self.data["label"].unique():
                    self.KTestData = pd.concat([self.KTestData, self.TrainingData[self.TrainingData["label"] == label].iloc[mask]])
            else:
                for label in self.data["label"].unique():
                    self.KTrainingData = pd.concat([self.KTrainingData, self.TrainingData[self.TrainingData["label"] == label].iloc[mask]])



        self.KTestData.reset_index(drop=True, inplace=True)
        self.KTrainingData.reset_index(drop=True, inplace=True)
        print("Final k-fold size of training data: ", len(self.KTrainingData))
        print("Final k-fold size of validation data: ", len(self.KTestData))


    def RemoveMissingData(self):
        # remove missing dataframe rows based on the self.TrainingData["image"].image being None type? and maybe when 
        # img is null to cover all cases
        initial_count = len(self.TrainingData) + len(self.TestData)
        
        # 1st case: null 
        self.TrainingData = self.TrainingData[self.TrainingData["image"].notnull()]
        self.TestData = self.TestData[self.TestData["image"].notnull()]

        #2nd case: None
        valid_rows = [img.image is not None for img in self.TrainingData["image"]]
        self.TrainingData = self.TrainingData[valid_rows]
        valid_rows = [img.image is not None for img in self.TestData["image"]]
        self.TestData = self.TestData[valid_rows]

        removed_count = initial_count - len(self.TrainingData) -  len(self.TestData)
        print(f"Removed {removed_count} rows with missing images. Remaining: {len(self.TrainingData)} rows.")

        pass

    def ResetData(self):
        del self.TrainingData
        del self.TestData
        self.TrainingData = pd.DataFrame()
        self.TestData = pd.DataFrame()
        del self.KTrainingData
        del self.KTestData
        self.KTrainingData = pd.DataFrame()
        self.KTestData = pd.DataFrame()
        pass


    #################################### Preprocessing Methods #############################################
    def ConvertToGrayScale(self, image: DataFrameImage=None):
        # convert each image to grayscale
        # should this run on individual images or the whole dataset? should the color image be stored inside the
        # dataframeimage class?
        if image is None:
            # perform grayscale on all images in the datasets
            for image in self.TrainingData["image"]:
                image = np.mean(image.image[:,:,:], 2)

        else:
            image =  np.mean(image.image[:,:,:], 2)

        pass

    def ResizeImages(self, TargetSize=(256,256)):
        # resize all images to the target size
        for image in self.TrainingData["image"]:
            image.image = np.array(Image.fromarray(image.image).resize(TargetSize))

        for image in self.TestData["image"]:
            image.image = np.array(Image.fromarray(image.image).resize(TargetSize))

        self.ImageSize = TargetSize

        pass

    def NormalizeImages(self):
        # need to det which normalization method to use will be best 

        # 1st: /255
        for image in self.TrainingData["image"]:
            image.image = (image.image / 255.0)
        for image in self.TestData["image"]:
            image.image = (image.image / 255.0)

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

    def ShowRandomImages(self, numImages: int = 10, showGrayscale: bool = False, showSegmented: bool = False):
        # show a grid of images, selected at random, with the titles the labels of the image
        # random list of values
        imagesToShow = np.random.randint(0, len(self.TrainingData), numImages)
        print(f"Number of images: {len(self.TrainingData['image'])}")

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

    def GetTestImages(self):
        images = []
        for image in self.TestData["image"]:
            images.append(image.image)
        return images

    def GetTestLabels(self):
        labels = []
        for label in self.TestData["label"]:
            labels.append(label)
        return labels

    def GetKTestImages(self):
        images = []
        for image in self.KTestData["image"]:
            images.append(image.image)
        return images

    def GetKTestLabels(self):
        labels = []
        for label in self.KTestData["label"]:
            labels.append(label)
        return labels
    
    def PlotDataDistrobution(self, train: pd.DataFrame = None, test: pd.DataFrame = None):
        # Get class distributions
        train_counts = train['label'].value_counts(normalize=True) * 100
        test_counts = test['label'].value_counts(normalize=True) * 100
    
        # Plot side-by-side pie charts
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].pie(train_counts, labels=train_counts.index, autopct='%1.1f%%', startangle=140)
        axs[0].set_title(f"Training Data Distribution (# Files: {len(train)})")
    
        axs[1].pie(test_counts, labels=test_counts.index, autopct='%1.1f%%', startangle=140)
        axs[1].set_title(f"Test Data Distribution (# Files: {len(test)})")
    
        plt.tight_layout()
        plt.show()
        



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

    classes = ["sitting", "running", "drinking","eating"]
    t.start()
    dm.LoadTrainAndTestData(folderName=currentFolder+ "//..//data//", csvFileName="Training_set.csv", numFiles=100, classFilter=classes)
    print("Training data loaded successfully.")
    print("Number of training images: ", len(dm.TrainingData))
    print("Number of classes: ", len(dm.TrainingData["label"].unique()))
    print("Classes: ", dm.TrainingData["label"].unique())
    print("Number of missing images: ", dm.TrainingData["image"].isnull().sum())
    print("Number of valid images: ", dm.TrainingData["image"].notnull().sum())
    print("Image size: ", dm.TrainingData["image"].iloc[0].image.shape)

    print("Test Data length: ", len(dm.TestData[dm.TestData["label"]=="drinking"]))
    print("Data split percentage: ", len(dm.TestData)/(len(dm.TrainingData)+len(dm.TestData)))

    t.stop()

    print("K-Fold Test")
    k = 5
    for i in range(k):
        dm.SplitKFold(k=k, foldIndex=i)


    sys.exit()
    #print(dm.TrainingData.head())
    #print(dm.TrainingData.describe())
    #print(dm.TrainingData.info())

    print("\nTest 2: Print stats...")
    t.start()
    dm.PrintStats()
    t.stop()

    print("\nTest 3: remove missing data...")
    t.start()
    dm.RemoveMissingData()
    t.stop()

    print("\nTest 4: Resize images...")
    t.start()
    dm.ResizeImages()
    t.stop()

    print("\nTest 5: convert images to grayscale...")
    t.start()
    dm.ConvertToGrayScale()
    t.stop()

    print("\nTest 7: Norm imgs...")
    t.start()
    dm.NormalizeImages()
    t.stop()

    print("\nTest 8: show random images...")
    t.start()
    dm.ShowRandomImages(numImages=5, showGrayscale=False, showSegmented=True)
    t.stop()


