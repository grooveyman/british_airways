from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import textblob
from textblob import TextBlob

#function to analyze the reviews
def getSubjectivity(review):
    return TextBlob(review).sentiment.subjectivity
#function to calculate polarity
def getPolarity(review):
    return TextBlob(review).sentiment.polarity
    
#function to analyze reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


data = pd.read_csv("sentiment.csv")

data['subjectivity'] = data['lemma'].apply(getSubjectivity)
data['polarity'] = data['lemma'].apply(getPolarity)
data['analysis'] = data['polarity'].apply(analysis)
print(data.head(5))

data.to_csv("polarized.csv")