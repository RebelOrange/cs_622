import os
from DataManager import DataFrameImage

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
        if df is None:
            raise ValueError("Error: Dataframe is None")

        if len(df) == 0:
            raise ValueError("Error: Empty dataframe provided")

        try:
            sample_img = df["image"].iloc[0]
            if isinstance(sample_img, DataFrameImage):
                img_array = sample_img.image
            else:
                img_array = sample_img

            if img_array.ndim < 2:
                raise ValueError("Error: Images must be 2D or 3D arrays")

            ## add something more?

        except Exception as e:
            raise ValueError(f"Error examining images: {str(e)}")

        return df


