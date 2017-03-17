import sys
from os import path
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

d=path.dirname(__file__)

def create(filename):
    print "creating new  file with name : ",filename
    """name=raw_input ("enter the name of file:")
    extension=raw_input ("enter extension of file:")"""
    try:
        #name=name+"."+extension
        file=open(filename,'a')

        file.close()
    except:
            print("error occured")
            sys.exit(0)


######creating a file ###############

create("Politics_articles.txt")
create("Film_articles.txt")
create("Football_articles.txt")
create("Business_articles.txt")
create("Technology_articles.txt")

######reading from CSV's and wriring the contents each file#####

df=pd.read_csv("../train_set.csv",sep='\t')
#contents=str(df["Content"])

#name=raw_input ("enter the name of file you want to write the contsnts od the CSV's :")
#file=open(name,'a')

for index, row in df.iterrows():

    category=str(row["Category"])

    if category=="Politics":
        file=open("Politics_articles.txt","a")
        content=str(row["Content"])
        file.write(content)
        file.close()
    if category=="Film":
        file=open("Film_articles.txt","a")
        content=str(row["Content"])
        file.write(content)
        file.close()
    if category=="Football":
        file=open("Football_articles.txt","a")
        content=str(row["Content"])
        file.write(content)
        file.close()           
    if category=="Business":
        file=open("Business_articles.txt","a")
        content=str(row["Content"])
        file.write(content)
        file.close()
    if category=="Technology":
        file=open("Technology_articles.txt","a")
        content=str(row["Content"])
        file.write(content)
        file.close()        
#file.close()

######Creating the wordcloud for each category ##############

text_Po = open(path.join(d,"Politics_articles.txt")).read()
text_Fi = open(path.join(d,"Film_articles.txt")).read()
text_Fo = open(path.join(d,"Football_articles.txt")).read()
text_Bu = open(path.join(d,"Business_articles.txt")).read()
text_Te = open(path.join(d,"Technology_articles.txt")).read()

#----------For Politics--------------------#

wordcloud=WordCloud().generate(text_Po)

plt.title("Wordcloud of Politics 1")
plt.imshow(wordcloud)
plt.axis("off") 

wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text_Po)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Wordcloud of Politics 2")
plt.show()

#---------For Film------------------------------#

wordcloud=WordCloud().generate(text_Fi)

plt.title("Wordcloud of Film 1")
plt.imshow(wordcloud)
plt.axis("off") 

wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text_Fi)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Wordcloud of Film 2")
plt.show()

#---------------For Football------------------------------------#

wordcloud=WordCloud().generate(text_Fo)

plt.title("Wordcloud of Football 1")
plt.imshow(wordcloud)
plt.axis("off") 

wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text_Fo)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Wordcloud of Football 2")
plt.show()

#------------------For Business------------------------------------------#

wordcloud=WordCloud().generate(text_Bu)

plt.title("Wordcloud of Business 1")
plt.imshow(wordcloud)
plt.axis("off") 

wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text_Bu)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Wordcloud of Business 2")
plt.show()

#---------------- For Technology----------------------------------------#

wordcloud=WordCloud().generate(text_Te)

plt.title("Wordcloud of Technology 1")
plt.imshow(wordcloud)
plt.axis("off") 

wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text_Te)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Wordcloud of Technology 2")
plt.show()
