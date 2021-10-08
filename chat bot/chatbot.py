#This is a 'smart' chat bott program
import numpy as np
import nltk
import string
import random

#Importing and reading the corpus
f = open('chatbot.txt', 'r', errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower() #Convert Text to lower-case
nltk.download('punkt') #use  the punkt tokenizer
nltk.download('wordnet') #using the word net dictionary
sent_tokens = nltk.sent_tokenize(raw_doc) #convert doc to a list of sentences
word_tokens = nltk.word_tokenize(raw_doc) #convert doc to a list of words
#Basically from doc to sentence then from sentence to word...

#Text Processing
lemmer = nltk.stem.WordNetLemmatizer()
#word net is a semantically-oriented dictionary of english included Nltk.
def lemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def lemNormalize(text):
  return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Defining The greeting function
GREET_INPUTS = ('hello', 'hi', 'hey', 'greetings', 'sup', 'what\'s up')
GREET_RESPONSES = ['hi', 'hey', '*nods*', 'hi there', 'hello', 'I\'m glad you\'re talking to me']
def greet(sentence):
  for word in sentence.split():
    if word.lower() in GREET_INPUTS:
      return random.choice(GREET_RESPONSES)

#Response Generation
from sklearn.feature_extraction.text import TfidfVectorizer #tf ->term frequency; idf->inverse document frequency
#term frequency is the frequency of occurance of terms in a corpus
#inverse doc freq(idf) shows how rare the occurance of a word is in a corpus
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
  robo1_response = ''
  TfidfVec = TfidfVectorizer(tokenizer=lemNormalize, stop_words='english')
  #print('TfidfVec',TfidfVec) 
  tfidf = TfidfVec.fit_transform(sent_tokens)
  #print('tfidf',tfidf)
  vals = cosine_similarity(tfidf[-1], tfidf)
  #print('vals',vals)
  idx = vals.argsort()[0][-2]
  #print('idx', idx)
  flat = vals.flatten()
  flat.sort()
  #print('flat',flat)
  req_tfidf = flat[-2]
  #print('req_',req_tfidf)
  if(req_tfidf == 0):
    robo1_response = robo1_response + 'Sorry I don\'t understand you'
    return robo1_response
  else:
    robo1_response = robo1_response + sent_tokens[idx]
    return robo1_response

#Defining conversation start and end protocols
flag = True
print('Bot: My name is Jarvis, Let\'s have aconversation ! Also if you want to exit anytime, just type Bye')
while(flag == True):
  user_response = input()
  user_response = user_response.lower()
  if(user_response != 'bye'):
    if(user_response == 'thanks' or user_response == 'thank you'):
      flag = False
      print('Bot: My pleasure')
    else:
      if(greet(user_response) != None):
        print('Bot: ' + greet(user_response))
      else:
        sent_tokens.append(user_response)
        word_tokens = word_tokens + nltk.word_tokenize(user_response)
        final_words = list(set(word_tokens))
        print('Bot: ', end='')
        print(response(user_response))
        sent_tokens.remove(user_response)
  else:
    flag = False
    print('Bot: Goodbye! Take care <3')
