import pandas as pd
from DataManager import DataFrameImage

class Trainer:
    def __init__(self):
        pass

    def Train(self, image: DataFrameImage, label: str, model = None):
        pass

    def TrainAll(self, df: pd.DataFrame(), model = None):

        for i in range(len(df)):
            self.Train(df.iloc[i, "image"].image, df.iloc[i, "label"], model)

