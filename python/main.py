from DataManager import DataManager

# need to split data

if __name__ == "__main__":
    dm = DataManager()
    dm.LoadTrainingData(folderName="../data/", csvFileName="Training_set.csv", numFiles=100)
    pass