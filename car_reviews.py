from csvreader import CsvReader, read_csv
from table import Table
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split   # does this shuffle the data? YES
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
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


def preprocess(X, Y):
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
    X_processed = [[ps.stem(word) for word in word_tokenize(review) if
                    word not in punctuation and word not in stopwords_set and word.isalpha()] for review in X]

    # CREATE A WORD DICTIONARY FOR EACH REVIEW
    print("Creating a review dictionary....")
    X_feat_dict = [dict(Counter(review)) for review in X_processed]
    Y_int = (Y == 'Pos').array.astype(int)

    return X_feat_dict, Y_int

# SHUFFLE THE RECORDS and SPLIT INTO TRAINING (80%) AND TEST DATA (20%)
train_data, test_data = train_test_split(df, test_size=0.2)
Y_train = train_data.Sentiment
X_train = train_data.Review
Y_test = train_data.Sentiment
X_test = train_data.Review

# PERFORM K-FOLD VALIDATION FOR MODEL SELECTION
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X_train):
    X_train_kf, X_eval_kf = X_train.iloc[train_index], X_train.iloc[test_index]
    Y_train_kf, Y_eval_kf = Y_train.iloc[train_index], Y_train.iloc[test_index]

    X_train_kf_dict, Y_train_kf_int = preprocess(X_train_kf, Y_train_kf)
    dv = DictVectorizer()
    X_train_kf_int = dv.fit_transform(X_train_kf_dict)

    # TRAIN THE MODEL
    print("Training a Multinomial Naive Bayes Model")
    MN_NaiveBayes = MultinomialNB()
    MN_NaiveBayes.fit(X_train_kf_int, Y_train_kf_int)

    # EVALUATE THE MODEL
    X_eval_kf_dict, Y_eval_kf_int = preprocess(X_eval_kf, Y_eval_kf)
    X_eval_kf_int = dv.transform(X_eval_kf_dict)
    Y_eval_pred = MN_NaiveBayes.predict(X_eval_kf_int)
    cnf_matrix = confusion_matrix(Y_eval_kf_int, Y_eval_pred)
    avg_acc = sum(Y_eval_pred == Y_eval_kf_int) / len(Y_eval_pred)
    print(f"Average accuracy = {avg_acc}")
    print(cnf_matrix)


# SELECT BEST MODEL
print("Selecting the best model....")
X_train_dict, Y_train_int = preprocess(X_train, Y_train)
dv = DictVectorizer()
X_train_int = dv.fit_transform(X_train_dict)

# TRAIN THE MODEL ON ALL TRAINING DATA
print("Training The final model on all training data....")
MN_NaiveBayes = MultinomialNB()
MN_NaiveBayes.fit(X_train_int, Y_train_int)

# EVALUATE THE MODEL PERFORMANCES ON THE TEST DATA
print("Evaluation Model performances")
X_test_dict, Y_test_int = preprocess(X_test, Y_test)
X_test_dict_int = dv.transform(X_test_dict)
Y_test_pred = MN_NaiveBayes.predict(X_test_dict_int)
cnf_matrix = confusion_matrix(Y_test_int, Y_test_pred)
avg_acc = sum(Y_test_pred == Y_test_int) / len(Y_test_pred)
print(f"Average accuracy = {avg_acc}")
print(cnf_matrix)



