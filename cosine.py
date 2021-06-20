
import PyPDF2
 
#create file object variable
#opening method will be rb
pdffileobj=open('paper1.pdf','rb')
pdffileobj2=open('paper 2.pdf','rb')
pdffileobj3=open('paper 3.pdf','rb')
pdffileobj4=open('paper 4.pdf','rb')
 
#create reader variable that will read the pdffileobj
pdfreader=PyPDF2.PdfFileReader(pdffileobj)
pdfreader2=PyPDF2.PdfFileReader(pdffileobj2)
pdfreader3=PyPDF2.PdfFileReader(pdffileobj3)
pdfreader4=PyPDF2.PdfFileReader(pdffileobj4)

num_pages = pdfreader.numPages
num_pages2 = pdfreader2.numPages
num_pages3 = pdfreader3.numPages
num_pages4 = pdfreader4.numPages

count=0
cnt2=0
cnt3=0
cnt4 = 0

d1 = ""
d2 = ""
d3 = ""
d4 = ""

while count < num_pages:
    pageObj = pdfreader.getPage(count)
    count +=1
    d1 += pageObj.extractText()

while cnt2 < num_pages2:
    pageObj = pdfreader2.getPage(cnt2)
    cnt2 +=1
    d2 += pageObj.extractText()

while cnt3 < num_pages3:
    pageObj = pdfreader3.getPage(cnt3)
    cnt3 +=1
    d3 += pageObj.extractText()

while cnt4 < num_pages4:
    pageObj = pdfreader4.getPage(cnt4)
    cnt4 +=1
    d4 += pageObj.extractText()



documents = [d1,d2,d3,d4]


import nltk
import string
import numpy

stemmer = nltk.stem.porter.PorterStemmer()
def StemTokens(tokens):
     return [stemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def StemNormalize(text):
     return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

from sklearn.feature_extraction.text import CountVectorizer
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(documents)

print(LemVectorizer.vocabulary_) 

 
tf_matrix = LemVectorizer.transform(documents).toarray()
#print (tf_matrix)
tf_matrix.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)
#print (tfidfTran.idf_)


tfidf_matrix = tfidfTran.transform(tf_matrix)
#print (tfidf_matrix.toarray())

print("\n")
cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print (cos_similarity_matrix)
