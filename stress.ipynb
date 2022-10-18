import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


#first step is getting the data
data = pd.read_csv ("stress.csv")
#print(data.head())


"""Checking and removing nulls"""
#print(data.isnull().sum())

"""Since the dataset lacks any nulls, we clean and remove any links, language error and special symbols"""
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["text"] = data["text"].apply(clean)

"""After the clean up we start looking for most used words
Here the try to visualize the most used words using WORD CLOUD
 """

text = " ".join(i for i in data.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#plt.show()
"""Now from the data we can get use machine learning to develop a stress detection model from the words used"""

data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]
#print(data.head())


"""The next part is about spliting the data into training and testing dataset"""
#Python used the sklearn library for spliting the data into training and testing


x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size=0.33, 
                                                random_state=4)


"""For the purpose of this model we will use binary classification, hence, Bernoulli Naive Bayes Algorithm 
is the best choice for stress detection"""
model = BernoulliNB()
model.fit(xtrain,ytrain)

""""At this point we have a model that has some trained and tested data to see the level of stress someone has 
in a sentence"""

"""In the next section we will test the perfomance of the model on random sentences to see mental state of users"""
user = input("Enter a Text: ")
print(user)
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)

