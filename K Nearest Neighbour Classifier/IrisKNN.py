from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def KNN_Classifier():
	# 1 Load the data
	Dataset = load_iris()

	# Features / Attributes
	Data = Dataset.data
	# Labels / Target
	Target = Dataset.target

	# 2 - Manipulate Data
	Data_Train, Data_Test, Target_Train, Target_Test = train_test_split(Data, Target, test_size = 0.5)

	Classifier = KNeighborsClassifier()
	# 3 Train and Build the model
	Classifier.fit(Data_Train, Target_Train)
	# 4 Test the model
	Predictions = Classifier.predict(Data_Test)

	Accuracy = accuracy_score(Target_Test, Predictions) # actual, predicted

	# Improve - model - missing
	return Accuracy

def main():
	Ret = KNN_Classifier()
	print("Accuracy of Iris dataset is", Ret*100)

if __name__ == "__main__":
	main()