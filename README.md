# Clustering-And-Classification
Applying Clustering and Classification algorithms on text datasets.

In order to run the python applications you will need Pandas library and Scikit-Learn.

The texts have 5 categories:

Technology

Business

Film

Football

Politics


Directory Beat_the_Benchmark:

With get_results.py I use the train csv file in order to train the classifiers K-Nearest-Neightboor, Random-Forests, LinearSVC and through a voting system they decide the categories that the texts from the test csv file belong. Output is given in a csv file.

Directory Classifiers:

With evaluating_classifiers.py I evaluate the five classifiers : LinearSVC, Bernoulli,  K-Nearest-Neighbors, Random-Forests, Multinomial-NB. For the evaluation 25 plots are created and a csv file as output.

Directory Clustering:

With clustering.py given the train set without the categories it decides and disrtibutes all the texts in 5 clusters. Results are given as output in a csv file. I use the K-Means algorithm.

Directory WordCloud:

With worldcloud.py you will aslo need WorldCloud library, given the train csv file it creates five word clouds one for every category.
