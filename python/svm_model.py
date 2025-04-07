import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from DataManager import DataManager

# define svm class
class svmModel:
    # initialize svm model with specified kernel type
    def __init__(self, kernel_type='linear', C=1.0, degree=3, gamma='scale'):
        self.kernel_type = kernel_type
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.model = None

    # train the svm model using specified kernel and hyperparameters
    def train(self, X_trn, y_trn):
        self.model = SVC(kernel=self.kernel_type, C=self.C, gamma=self.gamma)
        self.model.fit(X_trn, y_trn)
        print(f"Model trained with kernel: ", self.kernel_type)

    # predict using the trained svm model
    def predict(self, X_tst):
        return self.model.predict(X_tst)

    # evaluate svm model's accuracy
    def evaluate(self, X_tst, y_tst):
        # get predicted labels
        y_pred = self.predict(X_tst)
        correct_preds = np.sum(y_pred == y_tst)
        total_preds = len(y_tst)

        acc = correct_preds / total_preds
        return acc


if __name__ == "__main__":
    # Load data
    dm = DataManager()
    dm.LoadTrainingData(folderName='data/', csvFileName="Training_set.csv")
    dm.LoadTestData(folderName='data/', csvFileName="Testing_set.csv")

    # convert to grayscale
    dm.ConvertToGrayScale()

    # flatten images for training data
    X_trn = []
    y_trn = []
    for index, row in dm.TrainingData.iterrows():
        img = row['image'].image
        label = row['label']
        X_trn.append(img.flatten())
        y_trn.append(label)

    X_trn = np.array(X_trn)
    y_trn = np.array(y_trn)

    # z-score normalization
    mean_x = np.mean(X_trn, axis=0)
    std_x = np.std(X_trn, axis=0)
    X_train_normalized = (X_trn - mean_x) / std_x

    # apply pca
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_normalized)

    # flatten images for testing data
    X_tst = []
    y_tst = []
    for index, row in dm.TestData.iterrows():
        img = row['image'].image
        label = row['label']
        X_tst.append(img.flatten())
        y_tst.append(label)

    X_tst = np.array(X_tst)
    y_tst = np.array(y_tst)

    # z-score normalization
    X_tst_normalized = (X_tst - mean_x) / std_x

    # apply pca
    X_tst_pca = pca.transform(X_tst_normalized)

    ######      Test with different kernels     ######

    # initialize and train the svm model with a linear kernel
    svm_model = svmModel(kernel_type='linear', C=1.0)
    svm_model.train(X_trn, y_trn)

    # initialize and train the svm model with a poly kernel
    # svm_model = svmModel(kernel_type='poly', C=1.0, degree=3, gamma='scale')
    # svm_model.train(X_trn, y_trn)

    # initialize and train the svm model with a rbf kernel
    # svm_model = svmModel(kernel_type='rbf', C=1.0, gamma='scale')
    # svm_model.train(X_trn, y_trn)

    ######      Evaluate the model     ######

    model_acc = svm_model.evaluate(X_tst_pca, y_tst)
    print(f"Accuracy: {model_acc}")


