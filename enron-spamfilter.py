import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC, SVC

COMMON_WORD_LIMIT = 3000

def make_dictonary(dataset_path):

    dirs = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path)]
    all_words = [] 

    for dir in dirs:
        emails = os.listdir(dir)
        
        for email in emails:
            email_path = os.path.join(dir,email)
            with open(email_path, 'r', errors='ignore') as mail:
                for line in mail:
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    list_to_remove = [k for k in dictionary]
    
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(COMMON_WORD_LIMIT)
    
    np.save('dict_enron.npy',dictionary) 

    return dictionary
    

def extract_feature(dataset_path):
    dirs = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path)]
    docID = 0
    features_matrix = np.zeros((33716, COMMON_WORD_LIMIT))
    train_labels = np.zeros(33716)
    dictionary = make_dictonary(dataset_path)
    for dir in dirs:
        emails = os.listdir(dir)
        for email in emails:
            email_path = os.path.join(dir,email)
            with open(email_path, 'r', errors='ignore') as mail:

                words = []
                for line in mail:
                    words = line.split()
                    # all_words += words

                for word in words:
                      wordID = 0
                      for i,d in enumerate(dictionary):
                        if d[0] == word:
                            wordID = i
                            features_matrix[docID,wordID] = words.count(word)
            
            train_labels[docID] = int(email.split(".")[-2] == 'spam')
            docID = docID + 1

    return features_matrix,train_labels

def get_results(y_test, y_predict): 
    conf_matrix = confusion_matrix(y_test, y_predict)

    f_score = f1_score(y_test, y_predict, average="macro")
    
    precision = precision_score(y_test, y_predict, average="macro")
    
    recall = recall_score(y_test, y_predict, average="macro")

    ham_misclassification_rate = float(conf_matrix[0][1]/(conf_matrix[0][0] + conf_matrix[0][1]))

    spam_misclassification_rate = float(conf_matrix[1][0]/(conf_matrix[1][0] + conf_matrix[1][1]))

    print("Confuction Rate: ", conf_matrix)

    print("F Score: ", f_score)

    print("Precision: ", precision)

    print("Recall: ", recall)

    print("Ham Miscalssification Rate: ", str(round(ham_misclassification_rate, 4)))

    print("Spam Miscalssification Rate: ", str(round(spam_misclassification_rate, 4)))


def linear_model():
    # print("Linear Model")
    return LinearSVC()


def polynomial_model():
    # print("Polynomial Model")
    return SVC(kernel='poly', degree=4)



def main():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dir_path, "enron1/")
    make_dictonary(dataset_path)
    dictionary = make_dictonary(dataset_path)

    features_matrix,labels = extract_feature(dataset_path)
    np.save('enron_features_matrix.npy',features_matrix)
    np.save('enron_labels.npy',labels)

    X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

    # model = linear_model()
    model = polynomial_model()

    model.fit(X_train,y_train)

    y_predict = model.predict(X_test)
    get_results(y_test, y_predict)
    

if __name__ == "__main__":
    main()


