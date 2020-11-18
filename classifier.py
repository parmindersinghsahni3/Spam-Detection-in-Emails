
import numpy as np
import pandas as pd
import csv
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn import feature_extraction, model_selection, metrics, svm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split


def read_dataset():
    csv1_path = "dataset/icse_id.csv"
    csv2_path = "dataset/vldb_id.csv"

    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    df1.columns = ['Raw Text', 'Clean Text', 'Year', 'Date', 'Series',
                   'Conference Type', 'Publication Type', 'Serial No1', 'Serial No2', 'Serial No3']
    df2.columns = ['Index No', 'Raw Text', 'Clean Text', 'Year', 'Date', 'Series',
                   'Conference Type', 'Publication Type', 'Serial No1', 'Serial No2', 'Serial No3']

    return df1['Clean Text'].values, df2['Clean Text'].values


def print_dataset(data1, data2):
    print(data1)
    print("\n")
    print(data2)



def preprocess_data(data1, data2):
    data1 = clean_dataset(data1)
    data2 = clean_dataset(data2)
    labels1 = ['icse'] * data1.shape[0]
    labels2 = ['vldb'] * data2.shape[0]
    data = list(data1) + list(data2)
    labels = list(labels1) + list(labels2)

    training_set = np.column_stack((data, labels))
    np.random.shuffle(training_set)
    data = training_set[:, :1]
    labels = training_set[:, 1:]
    data = [row[0] for row in data]
    labels = [row[0] for row in labels]
    labels = [1 if y == 'icse' else 0 for y in labels]

    return data, labels


def clean_dataset(X):

    cachedStopWords = stopwords.words("english")

    for idx in range(X.shape[0]):
        X[idx] = ' '.join([word for word in X[idx].split()
                           if word not in cachedStopWords])
    return X


def corpus_dictionary(corpus):
    words = []
    for line in corpus:
        words.extend(line.split())
    dic = Counter(words)
    dic = dict(dic.most_common(3000))
    return dic


def feature_extraction(X):

    dic = corpus_dictionary(X)
    words = list(dic.keys())
    feature_matrix = np.zeros((len(X), len(words)))

    for row in range(len(X)):
        sentence = X[row]
        words_in_sentence = sentence.split()

        for col in range(len(words)):
            word = words[col]

            if word in words_in_sentence:
                feature_matrix[row, col] = 1

    return feature_matrix


def get_results(y_test, y_predict):

    conf_matrix = confusion_matrix(y_test, y_predict)
    print("\nConfuction Rate: \n", conf_matrix)

    accuracy = accuracy_score(y_test, y_predict)
    print("\nAccuracy: ", str(round(accuracy, 4)))

    f_score = f1_score(y_test, y_predict, average="macro")
    print("\nF Score: ", str(round(f_score, 4)))

    precision = precision_score(y_test, y_predict, average="macro")
    print("\nPrecision: ", str(round(precision, 4)))

    recall = recall_score(y_test, y_predict, average="macro")
    print("\nRecall: ", str(round(recall, 4)))

    ham_misclassification_rate = float(
        conf_matrix[0][1]/(conf_matrix[0][0] + conf_matrix[0][1]))
    print("\nIcse Miscalssification Rate: ", str(
        round(ham_misclassification_rate, 4)))

    spam_misclassification_rate = float(
        conf_matrix[1][0]/(conf_matrix[1][0] + conf_matrix[1][1]))
    print("\nVldb Miscalssification Rate: ", str(
        round(spam_misclassification_rate, 4)))


def linear_model():
    return LinearSVC()


def polynomial_model():
    return SVC(kernel='poly', degree=4)


if __name__ == "__main__":

    data1, data2 = read_dataset()

    print("Data Before Cleaning")
    print_dataset(data1, data2)
    print("\n\n")
    
    print("Data After Cleaning")
    print_dataset(clean_dataset(data1), clean_dataset(data2))
    print("\n\n")

    data, labels = preprocess_data(data1, data2)

    print("Common WORDS:", data)

    feature_matrix = feature_extraction(data)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix, labels, test_size=0.40)

    model = linear_model()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    get_results(y_test, y_predict)
