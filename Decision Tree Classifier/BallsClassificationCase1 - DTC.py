# import required library
from sklearn import tree

# load dataset
# Features = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],[48,"Rough"],[90,"Smooth"],[35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"],[96,"Smooth"],[43,"Rough"],[101,"Smooth"],[35,"Rough"],[95,"Smooth"]]

# Labels = ["Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket","Tennis","Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket"]

# encoding for string
# Rough - 1
# Smooth - 0
# Cricket - 2
# Tennis - 1
Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[101,0],[35,1],[95,0]]
Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

# Decide machine learning algorithm
obj = tree.DecisionTreeClassifier()

# Perform training of model
obj = obj.fit(Features, Labels)

# perform testing
# encode string value
# print(obj.predict([[97,"Smooth"]]))
print(obj.predict([[97,0],[35,1]]))