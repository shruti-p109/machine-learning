import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def UserBuiltHeadBrainPredictor():
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
	print("X & Y shapes:")
	print(X.shape)
	print(Y.shape)

	# train - test dataset splitting
	X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.3)

	# Regression line : y = mx+ c
	# calculation of m & c by formulae
	# calulating means
	X_bar = np.mean(X_train)
	Y_bar = np.mean(y_train)

	# calculating numerator sum & denominator sum
	numerator = 0
	denominator = 0
	for i in range(len(X_train)):
		numerator += ((X_train[i] - X_bar) * (y_train[i] - Y_bar))
		denominator += (X_train[i] - X_bar)**2

	# cal slope
	m = numerator / denominator
	print("Slope (m):", m)

	# cal c from m & equation
	c = Y_bar - (m * X_bar)
	print("Y intercept (c):", c)

	# using above calculated m & c do the predictions on test dataset - testing
	y_predicted = []
	for i in range(len(X_test)):
		y_predicted.append(((m * X_test[i]) + c))

	# r2 score formulae calculation - y_test (obs) & y_predicted
	SumOfResidualSquares = 0
	TotalSumOfSquares = 0
	y_test_bar = np.mean(y_test)
	for i in range(len(y_test)):
		SumOfResidualSquares += (y_test[i] - y_predicted[i])**2
		TotalSumOfSquares += (y_test[i] - y_test_bar)**2
	r2score = (1 - (SumOfResidualSquares / TotalSumOfSquares))
	print("R-squared:", r2score)

	# plotting regression line
	plt.plot(X_test, y_predicted, color='red', label='Linear Regression')
	plt.scatter(X_train, y_train, c='b', label='Scatter Plot')
	plt.xlabel("Head Size")
	plt.ylabel("Brain Weight")
	plt.legend()
	plt.show()

def main():
	print("User Defined Linear Regression on Head Brain Dataset")
	UserBuiltHeadBrainPredictor()

if __name__ == "__main__":
	main()