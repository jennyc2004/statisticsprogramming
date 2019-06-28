import sys
import re
import numpy as np

def create_training_data(train_file):
    """
        Creating training data
        Input:
            train_file
        Output: 
            train document vector    
    """
    labels,t     = tokenize(train_file)
    vocab       = mkdict(t)
    tr_vec      = vectorize(t,vocab)
    return tr_vec,labels,vocab
    




def create_testing_data(test_file,vocab):
    """
        Creating testing data,vocab
        Input:
            test_file
        Output: 
            test document vector
    """
    labels,t     = tokenize(test_file)
    ts_vec      = vectorize(t,vocab)
    return ts_vec,labels
    
    
    
    
def tokenize(data_file):
    """
        Tokenizing data file
        Input:
            data_file
        Output: 
            labels, tokens
    """
    labels = []
    tokens = []
    for line in open(data_file,'r'):
        words  = line.strip().split()
        label  = words.pop(0)
        labels.append(label)
        text   = ' '.join(words)
        text = re.sub('[!@#$-.,;/\_]', '', text)
        text = text.lower()
        ws   = text.split()
        toks = []
        for w in ws:
            if w == 'rt':
                continue
            toks.append(w)
        tokens.append(toks)
    return np.array(labels),tokens

 


def mkdict(tokens):
    """
        Creating vocabulary file
        Input:
            tokens
        Output: 
            vocabulary
    """
    vocab = dict()
    idx   = 0
    for ts in tokens:
        for t in ts:
            if t in vocab:
                idx      = idx + 1
                vocab[t] = idx
            else:
                vocab[t] = 0
    return vocab

      
 

def vectorize(tokens,vocab):
    """
        Vectorizing documents
        Input:
            tokens, vocabulary
        Output: 
            doc_vectors
    """
    doc = []
    for ts in tokens:
        d = np.zeros(len(vocab.keys()))
        for t in ts:
            if t in vocab:
                d[vocab[t]] = d[vocab[t]] + 1
        doc.append(d)
    return np.array(doc)
     

def compute_accuracy(prediction,truth):
    """
        Computing prediction accuracy
        Input:
            prediction, truth
        Output: 
            None
    """
    len_p = len(prediction)
    len_t = len(truth)
    if len_p != len_t:
        print('Something went wrong, mismatch in sizes')
        sys.exit()
    correct     = 0
    incorrect   = 0
    for i in range(0,len_p):
        if prediction[i] == truth[i]:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    print('correct = ',correct,' incorrect = ',incorrect)