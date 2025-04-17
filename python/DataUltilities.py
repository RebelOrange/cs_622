import pandas as pd
from sklearn.model_selection import train_test_split

def splitDataset(dataset, trainRatio=0.7, valRatio=0.15, testRatio=0.15, seed=42):
    if abs(trainRatio + valRatio + testRatio - 1.0) > 0.001:
        total = trainRatio + valRatio + testRatio
        trainRatio /= total
        valRatio /= total
        testRatio /= total

    if testRatio > 0:
        restData, testData = train_test_split(
            dataset, test_size=testRatio, random_state=seed,
            stratify=dataset["label"] if "label" in dataset.columns else None
        )
        testData = testData.reset_index(drop=True)
    else:
        restData = dataset
        testData = pd.DataFrame()

    if valRatio > 0:
        valRatioAdj = valRatio / (trainRatio + valRatio)
        trainData, valData = train_test_split(
            restData, test_size=valRatioAdj, random_state=seed,
            stratify=restData["label"] if "label" in restData.columns else None
        )
        trainData = trainData.reset_index(drop=True)
        valData = valData.reset_index(drop=True)
    else:
        trainData = restData.reset_index(drop=True)
        valData = pd.DataFrame()

    print(f"\nDataset split: Train: {len(trainData)}, Val: {len(valData)}, Test: {len(testData)}")
    return trainData, valData, testData
