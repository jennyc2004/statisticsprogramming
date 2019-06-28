import numpy as np
import heapq
from TextMining_cls_wk_3 import *
from TextMining_tm_wk_3 import *
from  sklearn.cluster import KMeans
from sklearn import metrics


# Create document vectors
def create_document_vectors(training_param_file, count_threshold):
        """
        Creating document vectors
        Input:
            word count threshold, training param file
        Output: 
            training bow feature, training vocabulary
        """
        print('TMPY> Creating document vectors')
        tm_train = TextMiner(training_param_file)
        tm_train.tokenize()
        tm_train.stopwords()
        tm_train.stem()
        tm_train.mkdict(count_threshold)
        tm_train.vectorize()
        bow = []
        for doc_id,words in tm_train.bow.items():
            bow.append(words)
        return np.array(bow),tm_train.vocab




# Create query vectors
def create_query_vectors(vocab,testing_param_file):
        """
        Creating query vectors
        Input:
            vocabulary, testing param file
        Output: 
            testing bow feature
        """
        print('TMPY> Creating query vector')
        tm_test = TextMiner(testing_param_file)
        tm_test.tokenize()
        tm_test.stopwords()
        tm_test.stem()
        tm_test.vocab = vocab
        tm_test.vectorize()
        bow = []
        for doc_id,words in tm_test.bow.items():
            bow.append(words)
        return np.array(bow)
        
        
     
     

# Find similar k documents
def find_similar_k_docs(query, docs,k):
        """
        Finds k similar documents to input query
        Input:
            query, doc, k
        Output: 
            k similar document ids
        """
        print('TMPY> Compute query-doc similarity')
        num_docs        = docs.shape[0]
        num_features    = docs.shape[1]
        similarity      = np.zeros(num_docs)
        
        for i in range(0,num_docs):
             sim = 0.0
             for j in range(0,num_features):
                 sim = sim + docs[i,j] * query[0,j]
             similarity[i] = sim
        
        k_similar = [t[0] for t in heapq.nlargest(k, enumerate(similarity), lambda t: t[1])]
        return k_similar




# Cluster k-means
def cluster_k_means(docs,k):
        """
        Cluster document vectors into k-clusters
        Input:
            docs, k
        Output: 
            k-means model, Silhouette Coefficient score for model (higher the better)
        """
        print('TMPY> Clustering document vectors')
        km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1,verbose=True)
        km.fit(docs)
        labels = km.labels_
        scores = metrics.silhouette_score(docs, labels, metric='euclidean')
        return km,scores