import os 

class  Model:
    def __init__(self):
        self.model = None
        self.model_dir = "models"

    def Train(self, df, **kwargs):
        raise NotImplementedError("Train method not implemented. Implement in child class")

    def Predict(self, df = None):
        if df is None:
            raise ValueError("Input data is required for prediction")
        else:
            return None

    def Preprocess(self, df):
        return df


