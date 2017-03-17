from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from random import randint
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import csv

#Calculating new Gravity Center for every Cluster

def Gravity_Center(cluster,vector):

	sum=numpy.array([[0]]) # creating a zero vector

	for i in cluster[1:]:
		sum=numpy.add(sum,vector[i:i+1])

	l=len(cluster)-1

	c=0

	sum_list=sum.tolist()

	for i in sum_list:
		sum_list[0][c]=float(sum[0][c])/l
		c=c+1

	new_center=numpy.array(sum_list)

	return new_center

def Find_Percent(Clusters,Categories):
	
	total_percents=[]
	category_count=[0,0,0,0,0] # list of sums : politics = category_count[0],Film .........

	for cluster in Clusters:
		for doc_num in cluster[1:]:
			category=Categories[doc_num]

			if category == "Politics":
				category_count[0]=category_count[0]+1
			if category == "Film":
				category_count[1]=category_count[1]+1
			if category == "Football":
				category_count[2]=category_count[2]+1
			if category == "Business":
				category_count[3]=category_count[3]+1
			if category == "Technology":
				category_count[4]=category_count[4]+1


		for i in range(5):
			l=len(cluster)-1

			if l > 0:
				category_count[i]=float(category_count[i])/l
				category_count[i]=float("{0:.2f}".format(category_count[i]))
		
		total_percents.append(category_count)
		category_count=[0,0,0,0,0]

	return total_percents

def Find_Distant_Centers(vector,center_list):

	min_index=0
	min_sum=0

	for center in center_list: # initializing the sum
		cs=cosine_similarity(vector[0:1],vector[center:center+1])
		min_sum=min_sum+cs
	

        for doc_num in range(vector.shape[0]):
		sum=0
		for center in center_list:
                	cs=cosine_similarity(vector[doc_num:doc_num+1],vector[center:center+1])
			sum=sum+cs
                if sum < min_sum: # if there is more distance between them
                        min_index=doc_num
                        min_sum=sum

	return min_index


def My_Kmeans(num_of_clusters,dimentions,vector,categories):
	print "I will create ",num_of_clusters," clusters !!!"

	# Creating the Clusters as a list of lists
	
	Clusters=[]


	for i in range(num_of_clusters):
		Clusters.append([]) # the first number of each list is the cluster's number
		

	# finding the first cluster centers
	
	cluster_centers=[] # the rand numers
	
	count=0

	rand=randint(1,dimentions-1) # only one random number that won't be used as center

	min_cs=cosine_similarity(vector[rand:rand+1],vector[0:1])
	min_index=0	

	for doc_num in range(vector.shape[0]):
		cs=cosine_similarity(vector[doc_num:doc_num+1],vector[rand:rand+1])

		if cs < min_cs: # if there is more distance between them
			min_index=doc_num
			min_cs=cs

	cluster_centers.append(min_index) # first center

	for i in range(4):
		cluster_centers.append(Find_Distant_Centers(vector,cluster_centers))	
	
	count=0
	for i in cluster_centers: # putting the dianysma as first in each cluster
		Clusters[count].append(vector[i:i+1])
		count=count+1		

	# Finding the distance between each center and each document
	

	for repeat in range(7):

		print "Fining centre number : ",repeat+1
		
		for doc_num in range(vector.shape[0]): # for the number of documents
			document=vector[doc_num:doc_num+1]
			
			temp=[] # tuple of tuples to insert to cos_sims

			for center in Clusters:
				cs=cosine_similarity(document,center[0])
				temp.append(cs) # ta 5 similarities

			#print "temp : ",temp
			
		
			# Finding the closest center for each document in order to create the clusters

			max=temp[0]
			index=0

			for i in range(5):
				if temp[i] > max:
					max=temp[i]
					index=i
							

			Clusters[index].append(doc_num) 
		
		# Calculating again the center for each cluster


		temp_gravity=[]
		
		Clusters_fp=Clusters

		for i in Clusters:
			new_center=Gravity_Center(i,vector)
			temp_gravity.append([new_center])

		Clusters=temp_gravity
			
	total_percents=Find_Percent(Clusters_fp,categories)

	return total_percents		

# Vectorizing the text data
	
new_dimentions=200 # for dimentionallity reduction

count_vectorizer = CountVectorizer(stop_words='english')

#vectorizer=TfidfVectorizer(stop_words='english') # na vkalw kai tis alles parametrous
df=pd.read_csv("../train_set.csv",sep="\t")

count_vectorizer.fit_transform(df["Content"])
X_freq=count_vectorizer.transform(df["Content"])
docs_ids=df["Id"]

tfidf=TfidfTransformer(norm="l2")
tfidf.fit(X_freq)

X_train = tfidf.transform(X_freq)

# Dimentionality reduction

svd=TruncatedSVD(n_components=new_dimentions, random_state=42)
X_lsi=svd.fit_transform(X_train)


total_percents=My_Kmeans(5,X_lsi.shape[0],X_lsi,df["Category"])

my_csv=csv.writer(open("clustering_KMeans.csv","wb"))

my_csv.writerow([" ","Politics","Film","Football","Business","Technology"])
my_csv.writerow(["Cluster 1 ",total_percents[0][0],total_percents[0][1],total_percents[0][2],total_percents[0][3],total_percents[0][4]])
my_csv.writerow(["Cluster 2 ",total_percents[1][0],total_percents[1][1],total_percents[1][2],total_percents[1][3],total_percents[1][4]])
my_csv.writerow(["Cluster 3 ",total_percents[2][0],total_percents[2][1],total_percents[2][2],total_percents[2][3],total_percents[2][4]])
my_csv.writerow(["Cluster 4 ",total_percents[3][0],total_percents[3][1],total_percents[3][2],total_percents[3][3],total_percents[3][4]])
my_csv.writerow(["Cluster 5 ",total_percents[4][0],total_percents[4][1],total_percents[4][2],total_percents[4][3],total_percents[4][4]])

