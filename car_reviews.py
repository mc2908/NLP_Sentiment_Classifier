from csvreader import CsvReader, read_csv
from table import Table
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split   # does this shuffle the data? YES
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter


# IMPORT CSV
# creating an instance of CsvReader which analyzes the file
#csv_reader = CsvReader('car-reviews.csv', verbose=1)
df = pd.read_csv('car-reviews.csv')
# If no header is detected automatically. force the first row in the file to be the header
#if csv_reader.header is None:
#    csv_reader.header = csv_reader.raw_data[0]
#parse the file and return a Table object
#tbl = csv_reader.parse()
#set the maximum row to be printed on screen
#tbl.max_row_print = 10
#print(tbl)


def preprocess(X, Y, stemming=True, ngram=1):
    punctuation = {".", ",", ";", "?", "!"}
    ps = PorterStemmer()
    # this function takes as input a vector of reviews (X) and their sentiment (Y)
    # it output a list of dictionaries (one for each review) of words-frequency after having removed punctuation,
    # stopwords and any word which contains a number. it also outputs the review sentiment as ndarray where 1 = Pos and
    # 0 = Neg
    # MAKE REVIEWS ALL LOWER CASE
    print("Making all reviews lower case....")
    X = [review.lower() for review in X]

    # REMOVE PUNCTUATION, STOP WORDS AND ANY WORD WHICH CONTAINS A NUMBER
    print("Removing punctuation and stop words....")
    stopwords_set = set(stopwords.words('english'))
    if stemming:
        X_processed = [[ps.stem(word) for word in word_tokenize(review) if
                    word not in punctuation and word not in stopwords_set and word.isalpha()] for review in X]
    else:
        X_processed = [[word for word in word_tokenize(review) if
                        word not in punctuation and word not in stopwords_set and word.isalpha()] for review in X]

    temp = [zip(*[review[i:] for i in range(0, ngram)]) for review in X_processed]
    X_processed_n_gram = [[' '.join(ngram) for ngram in zip_review] for zip_review in temp]

    # CREATE A WORD DICTIONARY FOR EACH REVIEW
    print("Creating a review dictionary....")
    X_feat_dict = [dict(Counter(review)) for review in X_processed_n_gram]
    Y_int = (Y == 'Pos').array.astype(int)

    return X_feat_dict, Y_int

# SHUFFLE THE RECORDS and SPLIT INTO TRAINING (80%) AND TEST DATA (20%)
train_data, test_data = train_test_split(df, test_size=0.2)
Y_train = train_data.Sentiment
X_train = train_data.Review
Y_test = test_data.Sentiment
X_test = test_data.Review

# CREATE THE MULTINOMIAL NAIVE BAYES MODEL
MN_NaiveBayes = MultinomialNB()

# PERFORM K-FOLD VALIDATED GRID SEARCH FOR MODEL SELECTION
param = {"alpha": np.logspace(-3, 1, num=20)}
gsCV = GridSearchCV(MN_NaiveBayes, param, scoring= 'accuracy', cv=5, verbose=3)
X_train_dict, Y_train_int = preprocess(X_train, Y_train)
dv = DictVectorizer()
X_train_int = dv.fit_transform(X_train_dict)
gsCV.fit(X_train_int, Y_train_int)

# GET THE BEST MODEL
best_param = gsCV.best_params_
best_score = gsCV.best_score_
best_model = gsCV.best_estimator_

# EVALUATE THE BEST MODEL PERFORMANCES ON THE TEST DATA
print("Evaluation Model performances om Test Data...")
X_test_dict, Y_test_int = preprocess(X_test, Y_test)
X_test_dict_int = dv.transform(X_test_dict)
Y_test_pred = best_model.predict(X_test_dict_int)
cnf_matrix = confusion_matrix(Y_test_int, Y_test_pred)
avg_acc = accuracy_score(Y_test_int, Y_test_pred)
f1 = f1_score(Y_test_int, Y_test_pred)
print(f"Average accuracy = {avg_acc}")
print(f"F1 score = {f1}")
print(cnf_matrix)





