import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

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
    