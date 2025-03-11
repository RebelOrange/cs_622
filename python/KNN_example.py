from Model import Model

class KNN(Model):
    def __init__(self):
        super(KNN).__init__()
        pass

    ###################################### Base Class Model Overrides ##########################################
    def Train(self, image, label): # overrides Model.Train
        pass



    ###################################### KNN Methods ########################################################3
    #def Distance


if __name__ == "__main__":
    knn = KNN()

    knn.Predict()
