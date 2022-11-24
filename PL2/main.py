## Data modification and visualization imports 
import pandas as pd
import matplotlib.pyplot as plt

## General purpose imports 
import time
import warnings
warnings.filterwarnings("ignore")

## scikit-learn imports 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay

start = time.time()

# importing the data base. 
data = pd.read_csv("./nba2021.csv")

# Data modification, seperating class and data values.
data_values = data.loc[:, 'FG':'PTS']
class_values = data.loc[:, 'Pos':'Pos']
class_name = list(set(class_values.values.ravel()))
data_features = data.columns.values.ravel()

feature_names = [e for e in data_features if e not in ("Player","Pos","Age","Tm","G","GS","MP",)]


# Spliting data into traning and testing data with the ratio od 75:25 respectivly
train_feature, test_feature, train_class, test_class = train_test_split(data_values, class_values.values.ravel(), random_state=0,test_size=0.25)


#Using LinearSVC to classify the data into diffrent classes
linearsvm = LinearSVC(dual= False,random_state=0,max_iter=-1,multi_class= "crammer_singer",tol=0.1,fit_intercept=True,intercept_scaling = 2).fit(train_feature, train_class)
print("Test set score for SVM: {:.3f}".format(linearsvm.score(test_feature, test_class)))

# Creating confusion matrix for the LinearSVC model
disp = ConfusionMatrixDisplay.from_estimator(linearsvm,test_feature,test_class,display_labels=class_name,cmap=plt.cm.Blues,)

# 10-fold stratified cross-validation procedeue using identical model settings
linearsvm = LinearSVC(dual= False,random_state=0,max_iter=-1,multi_class= "crammer_singer",tol=0.1,fit_intercept=True,intercept_scaling = 2)
scores = cross_val_score(linearsvm, data_values, class_values.values.ravel(), cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
end = time.time()

print("Total Run time of the program: ",end-start,"secs")

# Display confusion matrix 
plt.show()

