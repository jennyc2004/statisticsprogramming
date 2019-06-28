import numpy as np
import math
import heapq
from TextMining_tm_wk_2 import *

# pip install scikit-learn
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.externals.six import StringIO
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

# Extract bow feature
def extract_bow_feature(topics,bow,category):
        """
        Extracts Bag-of-Words feature for topic classifier
        Input:
            category for positive class
        Output: 
            feature(KEY=class_label,VALUE=vector representation of document (list))
        """
        print('TMPY> Extracting bow features')
        pos         = []
        neg         = []
        feature     = {}
        for cat,doc_ids in topics.items():
            if cat == category:
                for doc_id in doc_ids:
                    pos.append(bow[doc_id])
            else:
                for doc_id in doc_ids:
                    neg.append(bow[doc_id])
        feature[0]   = pos
        feature[1]   = neg
        return feature
                
 


# Extract tfidf feature
def extract_tfidf_feature(bow):
        """
        Extracts tf-idf feature for topic classifier
        Input:
            bow feature
        Output: 
            tfidf feature
        """
        print('TMPY> Extracting tfidf features')
        bow_arr     = np.array(bow)
        num_docs    = bow_arr.shape[0]
        num_words   = bow_arr.shape[1]
        idf         = np.zeros(num_words)
        for i in range(0,num_words):
            denom  = sum(bow_arr[:,i] > 0)
            idf[i] = math.log10(num_docs/denom)
        for i in range(0,num_docs):
             for j in range(0,num_words):
                 bow_arr[i,j] = bow_arr[i,j] * idf[j]
        return bow_arr





# Extract frequent-k bow feature
def extract_frequent_k_bow_feature(bow,k):
        """
        Extracts frequent-k bow feature for topic classifier
        Input:
            bow feature, k
        Output: 
            frequent-k bow feature
        """
        print('TMPY> Extracting frequent-k features')
        bow_arr     = np.array(bow)
        num_docs    = bow_arr.shape[0]
        num_words   = bow_arr.shape[1]
        count       = np.zeros(num_words)
        
        for i in range(0,num_words):
            count[i]  = sum(bow_arr[:,i])
        
        k_freq = [t[0] for t in heapq.nlargest(k, enumerate(count), lambda t: t[1])]
        
        out_bow = []
        for i in range(0,num_docs):
            temp_bow = []
            for j in range(0,num_words):
                if j in k_freq:
                    temp_bow.append(bow_arr[i,j])
            out_bow.append(temp_bow)
    
        return np.array(out_bow)

       


# Create Training Data
def create_training_data(category,word_count_threshold,training_param_file):
        """
        Creating training data
        Input:
            category, word count threshold, training param file
        Output: 
            training feature, training vocabulary
        """
        print('TMPY> Creating training data')
        tm_train = TextMiner(training_param_file)
        tm_train.tokenize()
        tm_train.stopwords()
        tm_train.stem()
        tm_train.mkdict(category,word_count_threshold)
        tm_train.vectorize()
        train_feature = extract_bow_feature(tm_train.topics, tm_train.bow, category)
        return train_feature,tm_train.vocab




# Create Testing Data
def create_testing_data(category,vocab,testing_param_file):
    """
        Creating testing data
        Input:
            category, vocabulary file, testing param file
        Output: 
            testing feature
        """
    print('TMPY> Creating testing data')
    tm_test = TextMiner(testing_param_file)
    tm_test.tokenize()
    tm_test.stopwords()
    tm_test.stem()
    tm_test.vocab = vocab
    tm_test.vectorize()
    test_feature = extract_bow_feature(tm_test.topics, tm_test.bow, category)
    return test_feature
    



# Train decision tree
def train_decision_tree(feature,label):
    """
        Training decision tree
        Input:
            features, labels
        Output: 
            trained decision tree
        """
    print('TMPY> Training decision tree')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(feature, label)
    return clf



# Plot decision tree
def print_decision_tree(clf,rule_file):
    """
        Print decision tree rules
        Input:
            trained decision tree, output file
        Output: 
            None
        """
    print('TMPY> Saving decision tree')
    out = StringIO()
    tree.export_graphviz(clf, out_file=out)
    f   = open(rule_file,'w')
    f.write(out.getvalue())
    f.close()




# Evaluate classifier
def evaluate_classifier(clf,feature):
    """
        Evaluating classifier
        Input:
            trained classifer, test feature
        Output: 
            None
        """
    print('TMPY> Evaluating classifier')
    pred = clf.predict(feature)
    return pred





# Train naive bayes
def train_naive_bayes(feature,label):
    """
        Training naive bayes
        Input:
            features, labels
        Output: 
            trained naive bayes model
        """
    print('TMPY> Training naive bayes')
    clf = GaussianNB()
    clf = clf.fit(feature, label)
    return clf



# Train linear model
def train_linear_model(feature,label):
    """
        Training linear model
        Input:
            features, labels
        Output: 
            trained linear model
        """
    print('TMPY> Training linear model')
    clf = linear_model.SGDClassifier(max_iter=100)
    clf = clf.fit(feature, label)
    return clf



# Compute evaluation metrics
def compute_evaluation_metrics(label,pred):
    """
        Computing evaluation metrics 
        Input:
            labels, predictions
        Output: 
            None
        """
    print('TMPY> Computing evaluation metrics')
    y_true = np.array(label)
    y_scores = np.array(pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_scores)
    print('Precision:',precision)
    print('Recall:',recall)
    print('Accuracy:',accuracy)

