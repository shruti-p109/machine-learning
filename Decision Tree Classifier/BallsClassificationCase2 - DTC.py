from sklearn import tree
from sys import *

def PredictBallsType(Features, Labels, TestInput):
	obj = tree.DecisionTreeClassifier()

	obj = obj.fit(Features, Labels)

	Result = [2, 1]
	print(Result)
	print(Result[0])
	# decode
	# if there is only one element in list, below statement will get that value and then compare it instead of throwing error
	# this will not work if Result has multiple elements, no error but logic wont work
	if Result == 1:
		print("Your object seems like Tennis Ball")
	elif Result == 2:
		print("Your object seems like Cricket Ball")

def main():
	print("Ball Type Predictor Case Study")
	Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[101,0],[35,1],[95,0]]
	Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

	print("Please enter weight of ball:")
	Weight = int(input())

	print("Please enter type of surface of  ball:")
	Surface = input().lower()

	if Surface == "rough":
		Surface = 1
	elif Surface == "smooth":
		Surface = 0
	else:
		print("Invalid Surface type.")
		exit()

	TestInput = [[Weight,Surface]]
	PredictBallsType(Features, Labels, TestInput)

if __name__ == "__main__":
	main()