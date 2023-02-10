import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def HeadBrainPredictor():
	data = pd.read_csv("HeadBrain.csv")

	print("Data Shape:")
	print(data.shape)

	print("Data info:")
	print(data.info())

	print("Checking for null values:")
	print(data.isnull().any())

	print("Checking unique values:")
	print(data.nunique())

	print("Plotting scatter plot of X & Y, X being Head Size & Y being Brain Weight")
	plt.figure(figsize=(6, 6))
	sns.scatterplot(y="Brain Weight(grams)", x="Head Size(cm^3)", data=data)
	plt.show()

	X = data["Head Size(cm^3)"].values
	Y = data["Brain Weight(grams)"].values
	# reshaping X into 2D array
	X = X.reshape((-1, 1))
	print("X & Y shapes:")
	print(X.shape)
	print(Y.shape)

	# train - test dataset splitting
	X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.3)

	# training
	LinearRegressionObj = LinearRegression()
	LinearRegressionObj.fit(X_train, y_train)

	# testing - prediction for y in test dataset
	y_predicted = LinearRegressionObj.predict(X_test)

	# r2 score
	print("R-squared:", r2_score(y_test, y_predicted))

	# plotting regression line
	plt.plot(X_test, y_predicted, color='red', label='Linear Regression')
	plt.scatter(X_train, y_train, c='b', label='Scatter Plot')
	plt.xlabel("Head Size")
	plt.ylabel("Brain Weight")
	plt.legend()
	plt.show()

def main():
	print("Linear Regression on Head Brain Dataset")
	HeadBrainPredictor()

if __name__ == "__main__":
	main()