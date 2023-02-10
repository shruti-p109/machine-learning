from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance 

def euc(a,b):
    return distance.euclidean(a, b)

class KNN_Classifier():
    def fit(self, trainingData, trainingTarget):
        self.TrainingData = trainingData
        self.TrainingTarget = trainingTarget

    def closet(self, row):
        minimumDistance = euc(row, self.TrainingData[0])
        minimumIndex = 0
        for i in range(1, len(self.TrainingData)):
            Distance = euc(row, self.TrainingData[i])
            if Distance < minimumDistance:
                minimumDistance = Distance
                minimumIndex = i
        return self.TrainingTarget[minimumIndex]

    def predict(self, TestData):
        predictions = []
        for value in TestData:
            result = self.closet(value)
            predictions.append(result)
        return predictions

def MLClassifier():
    # 1 Load the data
    Dataset = load_iris()

    # Features / Attributes
    Data = Dataset.data
    # Labels / Target
    Target = Dataset.target

    # 2 - Manipulate Data
    Data_Train, Data_Test, Target_Train, Target_Test = train_test_split(Data, Target, test_size = 0.5)

    Classifier = KNN_Classifier()
    # 3 Train and Build the model
    Classifier.fit(Data_Train, Target_Train)
    # 4 Test the model
    Predictions = Classifier.predict(Data_Test)

    Accuracy = accuracy_score(Target_Test, Predictions) # actual, predicted

    # Improve - model - missing
    return Accuracy

def main():
    Ret = MLClassifier()
    print("Accuracy of Iris dataset is", Ret*100)

if __name__ == "__main__":
    main()