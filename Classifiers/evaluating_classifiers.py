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
import csv

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def Print_ROC_Plots(clf,Politics,Film,Football,Business,Technology): 

	roc_means=[] # roc means returned

	c=0
	roc_sum=0

	for i in Politics:
		roc_auc=auc(i[0],i[1])
		roc_sum=roc_sum+roc_auc
		plt.plot(i[0],i[1],lw=1,label="ROC fold %d (area = %0.2f)" % (c,roc_auc) )
		c=c+1

	m=float(roc_sum)/len(Politics)
	m=float("{0:.6f}".format(m))

	roc_means.append(["Politics",m])	

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	if clf=="svm":
		plt.title('Politics - LinearSVC')
	if clf=="bern":
		plt.title('Politics - Bernoulli')
	if clf=="knn":
		plt.title('Politics - K_Nearest_Neighbors')
	if clf=="rf":
		plt.title('Politics - Random_Forests')
	if clf=="multi":
		plt.title('Politics - Multinomial_NB')
						
	plt.legend(loc="lower right",fontsize="x-small")
	plt.show()

	c=0
	roc_sum=0

	for i in Film:
		roc_auc=auc(i[0],i[1])
		roc_sum=roc_sum+roc_auc
		plt.plot(i[0],i[1],lw=1,label="ROC fold %d (area = %0.2f)" % (c,roc_auc) )
		c=c+1       

	m=float(roc_sum)/len(Film)
	m=float("{0:.6f}".format(m))

	roc_means.append(["Film",m])	

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	if clf=="svm":
		plt.title('Film - LinearSVC')
	if clf=="bern":
		plt.title('Film - Bernoulli')
	if clf=="knn":
		plt.title('Film - K_Nearest_Neighbors')
	if clf=="rf":
		plt.title('Film - Random_Forests')
	if clf=="multi":
		plt.title('Film - Multinomial_NB')

	plt.legend(loc="lower right",fontsize="x-small")					
	plt.show()

	c=0
	roc_sum=0

	for i in Football:
		roc_auc=auc(i[0],i[1])
		roc_sum=roc_sum+roc_auc
		plt.plot(i[0],i[1],lw=1,label="ROC fold %d (area = %0.2f)" % (c,roc_auc) )
		c=c+1   


	m=float(roc_sum)/len(Football)
	m=float("{0:.6f}".format(m))

	roc_means.append(["Football",m])

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	if clf=="svm":
		plt.title('Football - LinearSVC')
	if clf=="bern":
		plt.title('Football - Bernoulli')
	if clf=="knn":
		plt.title('Football - K_Nearest_Neighbors')
	if clf=="rf":
		plt.title('Football - Random_Forests')
	if clf=="multi":
		plt.title('Football - Multinomial_NB')
	
	plt.legend(loc="lower right",fontsize="x-small")
	plt.show()


	c=0
	roc_sum=0

	for i in Business:
		roc_auc=auc(i[0],i[1])
		roc_sum=roc_sum+roc_auc
		plt.plot(i[0],i[1],lw=1,label="ROC fold %d (area = %0.2f)" % (c,roc_auc) )
		c=c+1   

	m=float(roc_sum)/len(Business)
	m=float("{0:.6f}".format(m))

	roc_means.append(["Business",m])

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	if clf=="svm":
		plt.title('Business - LinearSVC')
	if clf=="bern":
		plt.title('Business - Bernoulli')
	if clf=="knn":
		plt.title('Business - K_Nearest_Neighbors')
	if clf=="rf":
		plt.title('Business - Random_Forests')
	if clf=="multi":
		plt.title('Business - Multinomial_NB')
	
	plt.legend(loc="lower right",fontsize="x-small")
	plt.show()

	c=0
	roc_sum=0

	for i in Technology:
		roc_auc=auc(i[0],i[1])
		roc_sum=roc_sum+roc_auc
		plt.plot(i[0],i[1],lw=1,label="ROC fold %d (area = %0.2f)" % (c,roc_auc) )
		c=c+1   
	

	m=float(roc_sum)/len(Technology)
	m=float("{0:.6f}".format(m))
	
	roc_means.append(["Technology",m])	

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	if clf=="svm":
		plt.title('Technology - LinearSVC')
	if clf=="bern":
		plt.title('Technology - Bernoulli')
	if clf=="knn":
		plt.title('Technology - K_Nearest_Neighbors')
	if clf=="rf":
		plt.title('Technology - Random_Forests')
	if clf=="multi":
		plt.title('Technology - Multinomial_NB')
	
	plt.legend(loc="lower right",fontsize="x-small")
	plt.show()

	return roc_means

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
	for j in range(1):	
		InputT[i]=InputT[i]+df["Title"][i]	


print "After seting my Input Data Train"


print "After transforming my Input Data Train"

# Vectorizing the test.csv

InputTest=[]
length=len(df_test["Content"])

for i in range(length):
	InputTest.append(df_test["Content"][i])
	for j in range(1):  #Adding title in our input data
		InputTest[i]=InputTest[i]+df_test["Title"][i]

print "After seting my Input Data Test"

print "After Transforming  my Input Data Test"
# Initializing the classifiers

clf=LinearSVC()
clf1=BernoulliNB()
clf2=KNeighborsClassifier()
clf3=RandomForestClassifier()
clf4=MultinomialNB()

vectorizer=CountVectorizer(stop_words='english')
transformer=TfidfTransformer(norm="l2")
svd=TruncatedSVD(n_components=2, random_state=42)
clf=LinearSVC()

pipeline = Pipeline([
('vect', vectorizer),
('tfidf', transformer),
('svd',svd),
('clf', clf),
])

pipeline1 = Pipeline([
('vect', vectorizer),
('tfidf', transformer),
('svd',svd),
('clf', clf1),
])

pipeline2 = Pipeline([
('vect', vectorizer),
('tfidf', transformer),
('svd',svd),
('clf', clf2),
])

pipeline3 = Pipeline([
('vect', vectorizer),
('tfidf', transformer),
('svd',svd),
('clf', clf3),
])

#------ For Multinomial Bayes only---------

X_MultiNB=df['Content']
vectorizerNB=CountVectorizer(stop_words='english')
transformerNB=TfidfTransformer()

pipeline4 = Pipeline([
    ('vect', vectorizerNB),
    ('tfidf', transformerNB),
    ('clf', clf4)
])

# ----- Until Here ----------

# Lists for the accuracies of each classifier

ac_clf=[]
ac_clf1=[]
ac_clf2=[]
ac_clf3=[]
ac_clf4=[]


kf = KFold(len(InputT), n_folds=10) # we get indexes

# List that hold the Roc curves fro each alogorithm#

Politics_csv=[]
Film_csv=[]
Football_csv=[]
Business_csv=[]
Technology_csv=[]

Politics_bern=[]
Film_bern=[]
Football_bern=[]
Business_bern=[]
Technology_bern=[]

Politics_knn=[]
Film_knn=[]
Football_knn=[]
Business_knn=[]
Technology_knn=[]

Politics_rf=[]
Film_rf=[]
Football_rf=[]
Business_rf=[]
Technology_rf=[]

Politics_multi=[]
Film_multi=[]
Football_multi=[]
Business_multi=[]
Technology_multi=[]
c=1
for train, test in kf:

        print "Calculating",c,"/ 10 fold cross validation for 4 Classifiers ....."
        c=c+1

	X_new_train=[]
	Y_new_train=[]

	for j in train:

		X_new_train.append(InputT[j])
		Y_new_train.append(Y_train[j]) # new category set for each new train data
	
	X_test=[]
	Y_test=[]

	for t in test:
		X_test.append(InputT[t])
		Y_test.append(Y_train[t])		

#------------Support Vector Machines--------------#

	pipeline.fit(X_new_train,Y_new_train)

	predicted=pipeline.predict(X_test)

	correct_predictions=0

	for i in range(len(predicted)):
		if predicted[i]==Y_test[i]:
			correct_predictions=correct_predictions+1

	total_predictions=len(X_test)

	Accuracy=float(correct_predictions)/total_predictions
	Accuracy=float("{0:.6f}".format(Accuracy))
	ac_clf.append(Accuracy)


	Politics_csv.append(roc_curve(Y_test,predicted,pos_label=3)) # We set labels according to the fundtion label
	Film_csv.append(roc_curve(Y_test,predicted,pos_label=1))
	Football_csv.append(roc_curve(Y_test,predicted,pos_label=2))
	Business_csv.append(roc_curve(Y_test,predicted,pos_label=0))
	Technology_csv.append(roc_curve(Y_test,predicted,pos_label=4))         





########## Bernoulli #######################

	pipeline1.fit(X_new_train,Y_new_train)
	predicted=pipeline1.predict(X_test)

	correct_predictions=0

	for i in range(len(predicted)):
		if predicted[i]==Y_test[i]:
			correct_predictions=correct_predictions+1

 	total_predictions=len(X_test)
	Accuracy=float(correct_predictions)/total_predictions
	Accuracy=float("{0:.6f}".format(Accuracy))
	ac_clf1.append(Accuracy)

	Politics_bern.append(roc_curve(Y_test,predicted,pos_label=3))
	Film_bern.append(roc_curve(Y_test,predicted,pos_label=1))
	Football_bern.append(roc_curve(Y_test,predicted,pos_label=2))
	Business_bern.append(roc_curve(Y_test,predicted,pos_label=0))
	Technology_bern.append(roc_curve(Y_test,predicted,pos_label=4))

# ########K-Nearest neighbors ###################

	pipeline2.fit(X_new_train,Y_new_train)
	predicted=pipeline2.predict(X_test)

	correct_predictions=0

	for i in range(len(predicted)):
	   	if predicted[i]==Y_test[i]:
			correct_predictions=correct_predictions+1

	total_predictions=len(X_test)

	Accuracy=float(correct_predictions)/total_predictions
	Accuracy=float("{0:.6f}".format(Accuracy))
	ac_clf2.append(Accuracy)

	Politics_knn.append(roc_curve(Y_test,predicted,pos_label=3))
	Film_knn.append(roc_curve(Y_test,predicted,pos_label=1))
	Football_knn.append(roc_curve(Y_test,predicted,pos_label=2))
	Business_knn.append(roc_curve(Y_test,predicted,pos_label=0))
	Technology_knn.append(roc_curve(Y_test,predicted,pos_label=4))

# ############## Random_Forets#################

	pipeline3.fit(X_new_train,Y_new_train)
	predicted=pipeline3.predict(X_test)

	correct_predictions=0

	for i in range(len(predicted)):
		if predicted[i]==Y_test[i]:
			correct_predictions=correct_predictions+1

	total_predictions=len(X_test)

	Accuracy=float(correct_predictions)/total_predictions
	Accuracy=float("{0:.6f}".format(Accuracy))
	ac_clf3.append(Accuracy)

	Politics_rf.append(roc_curve(Y_test,predicted,pos_label=3))
	Film_rf.append(roc_curve(Y_test,predicted,pos_label=1))
	Football_rf.append(roc_curve(Y_test,predicted,pos_label=2))
	Business_rf.append(roc_curve(Y_test,predicted,pos_label=0))
	Technology_rf.append(roc_curve(Y_test,predicted,pos_label=4))

# ############### MultinomialNB ###################

	pipeline4.fit(X_new_train,Y_new_train)
	predicted=pipeline4.predict(X_test)

	correct_predictions=0

	for i in range(len(predicted)):
	   	if predicted[i]==Y_test[i]:
			correct_predictions=correct_predictions+1

	total_predictions=len(X_test)

	Accuracy=float(correct_predictions)/total_predictions
	Accuracy=float("{0:.6f}".format(Accuracy))
	ac_clf4.append(Accuracy)

	Politics_multi.append(roc_curve(Y_test,predicted,pos_label=3))
	Film_multi.append(roc_curve(Y_test,predicted,pos_label=1))
	Football_multi.append(roc_curve(Y_test,predicted,pos_label=2))
	Business_multi.append(roc_curve(Y_test,predicted,pos_label=0))
	Technology_multi.append(roc_curve(Y_test,predicted,pos_label=4))

#----------------Making the ROC Plots for SVM.....----------------------#

svm_roc=Print_ROC_Plots("svm",Politics_csv,Film_csv,Football_csv,Business_csv,Technology_csv)

#-------....for Bernoulli..............#

bern_roc=Print_ROC_Plots("bern",Politics_bern,Film_bern,Football_bern,Business_bern,Technology_bern)

#............for K_Nearest_Neighbors............#

knn_roc=Print_ROC_Plots("knn",Politics_knn,Film_knn,Football_knn,Business_knn,Technology_knn)

#............for Random_Forests.................#

rf_roc=Print_ROC_Plots("rf",Politics_rf,Film_rf,Football_rf,Business_rf,Technology_rf)

#..........and Multinomial_NB...................#

multi_roc=Print_ROC_Plots("multi",Politics_multi,Film_multi,Football_multi,Business_multi,Technology_multi)

######Creating the CSV with the accuracies #################

my_csv=csv.writer(open("EvaluationMetric_10fold.csv","wb"))
my_csv.writerow(["Statistic Measure","SVM","Bernoulli","KNN","Random Forests","Naive Bayes Multinomial","My Method"])

for i in range(10):
	my_csv.writerow(["Accuracy",ac_clf[i],ac_clf1[i],ac_clf2[i],ac_clf3[i],ac_clf4[i]])

my_csv.writerow([" "])
my_csv.writerow(["Average Accuracy",reduce(lambda x,y: x+y,ac_clf)/len(ac_clf),reduce(lambda x,y: x+y,ac_clf1)/len(ac_clf1),reduce(lambda x,y: x+y,ac_clf2)/len(ac_clf2),reduce(lambda x,y: x+y,ac_clf3)/len(ac_clf3),reduce(lambda x,y: x+y,ac_clf4)/len(ac_clf4)])
my_csv.writerow([" "])
my_csv.writerow(["ROC"])

my_csv.writerow(["Business",svm_roc[0][1],bern_roc[0][1],knn_roc[0][1],rf_roc[0][1],multi_roc[0][1]])
my_csv.writerow(["Film",svm_roc[1][1],bern_roc[1][1],knn_roc[1][1],rf_roc[1][1],multi_roc[1][1]])
my_csv.writerow(["Football",svm_roc[2][1],bern_roc[2][1],knn_roc[2][1],rf_roc[2][1],multi_roc[2][1]])
my_csv.writerow(["Politics",svm_roc[3][1],bern_roc[3][1],knn_roc[3][1],rf_roc[3][1],multi_roc[3][1]])
my_csv.writerow(["Technology",svm_roc[4][1],bern_roc[4][1],knn_roc[4][1],rf_roc[4][1],multi_roc[4][1]])



  