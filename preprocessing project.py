import nltk, csv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas
df = pandas.read_csv(r'C:\Users\Akhil Sunny\Desktop\Akhil17595_tweets.csv')
text=(df.iloc[:,-1])

tokenized_text=sent_tokenize(str(text))
#print(tokenized_text)

tokenized_word=word_tokenize(str(text))
#print(tokenized_word)

stop_words=set(stopwords.words("english"))
#print(stop_words)

filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
#print("Tokenized Sentence:",tokenized_word)
#print("Filterd Sentence:",filtered_sent)

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

#print("Filtered Sentence:",filtered_sent)
#print("Stemmed Sentence:",stemmed_words)

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
#print("Lemmatized Word:",lem.lemmatize(word,"v"))
#print("Stemmed Word:",stem.stem(word))
#print(nltk.pos_tag(tokenized_word))
f=open(r"F:\NRC.txt",'a')
f.writelines(tokenized_word)
i=0
#with open(r"G:\NRC.txt", 'w') as f:
    #i+=1
   # f.write(tokenized_word[i])
#df.to_csv(r'F:\NRC.txt', header=None, index=None, sep=' ', mode='a')
