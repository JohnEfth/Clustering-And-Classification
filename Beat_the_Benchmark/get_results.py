from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import VotingClassifier
import csv

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


#----------------------Main ----------------------------------
#Read Data

df=pd.read_csv("../train_set.csv",sep="\t")
df_test=pd.read_csv("../test_set.csv",sep="\t")

X_test=df_test['Content']

# Labeling the categories


le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train=le.transform(df["Category"])

# vectorizing X_train

InputT=[]

length=len(df["Content"])

for i in range(length):
	InputT.append(df["Content"][i])
	for j in range(5):	
		InputT[i]=InputT[i]+df["Title"][i]	

vectorizer=CountVectorizer(stop_words='english')
transformer=TfidfTransformer(norm="l2")
svd=TruncatedSVD(n_components=50, random_state=42)

clf=KNeighborsClassifier()
clf1=RandomForestClassifier()
clf2=LinearSVC()

eclf = VotingClassifier(estimators=[('knn',clf),('rf', clf1),('svm',clf2)],voting='hard')


pipelineV = Pipeline([
('vect', vectorizer),
('tfidf', transformer),
('svd',svd),
('clf', eclf),
])

InputTest=[]

length=len(df_test["Content"])

for i in range(length):
	InputTest.append(df["Content"][i])
	for j in range(5):	
		InputTest[i]=InputTest[i]+df["Title"][i]

print "After seting my Input Data Test"

#------------Trainning and Prediction--------------#

pipelineV.fit(InputT,Y_train)

predicted=pipelineV.predict(InputTest)
         
#---------Creating the csv file with the results-------#


my_test_csv=csv.writer(open("testSet_categories.csv","wb"))
my_test_csv.writerow(["ID","Predicted_Category"])


new_predicted=le.inverse_transform(predicted)

c=0

for i in new_predicted:
	my_test_csv.writerow([df_test["Id"][c],i])		
	c=c+1

#--------------------------------------------------------#


  
