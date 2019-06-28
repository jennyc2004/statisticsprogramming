import sys
import os
import re
#from string import maketrans


class TextMiner:
    
    
    
    def __init__(self,property_file):
        """
        Initializes the class with properties file
        """
        # Define file names and objects
        self.property_file=property_file
        properties    = {}    
        # Check to see if properties file exists
        # If not, create a file with default parameters
        if not os.path.isfile(property_file):
            print('TMPY: Creating property file with default parameters')
            self.create_default_property_file(property_file)
        # Parse the properties file for parameters
        f = open(property_file, 'r')
        for line in f:
            cleanedLine = line.strip()
            if cleanedLine:
                if cleanedLine.startswith('#'):
                    continue
                if len(cleanedLine.split('=')) == 2:
                    params = cleanedLine.split('=')
                    properties[params[0]]=params[1]
                else:
                    print('TMPY: Incorrect format in property file at:',cleanedLine)
                    sys.exit(1)
        f.close()
        self.properties = properties
    
    
    
    
    
    def create_default_property_file(self):
        """
        Creates the default properties file with default arguments
        """
        f = open(self.property_file,'w')
        f.write('# this tag identifies individual documents. case sensitive.\n')
        f.write('doctag=REUTERS\n\n')
        f.write('# these are the tags for the text to be used. case sensitive.\n')
        f.write('bodytags=TITLE BODY\n\n')
        f.write('#labeltag is the tag for the categories/labels\n')
        f.write('labeltag=TOPICS\n\n')
        f.write('# input can be a zip file (with extn .zip) or an xml file\n')
        f.write('infile=trn\n\n')
        f.write('# input dictionary file\n')
        f.write('dictionary=new.dx\n\n')
        f.close()

    
    
    
    
    def replace_text(self, text, wordDic):
        """
        take a text and replace words that match a key in a dictionary with
        the associated value, return the changed text
        """
        rc = re.compile('|'.join(map(re.escape, wordDic)))
        def translate(match):
            return wordDic[match.group(0)]
        return rc.sub(translate, text)




    def clean_text(self, text):
        """
        Cleans the text by replacing unwanted characters with space
        """
        text = re.sub('[!@#$-.,;/\_0123456789<>]', ' ', text)
        return text
    
    
    
    
    
    
    def tokenize(self):
        """
        Tokenize the input document
        Output: 
            tokens(KEY=doc_id,VALUE=list of words)
            topics(KEY=topic,VALUE=list of doc_ids)
        """
        print('TMPY> Tokenizing documents')        
        properties = self.properties      
        tokens = dict()
        topics = dict()
        in_file    = open(properties['infile'],'r')
        lines      = in_file.readlines()
        in_file.close()
        
        sentences_raw = " ".join(lines)
    
        proc_dic = {
        '<D>':' ',  
        '</D>':' '}    
        
        sentences = self.replace_text(sentences_raw,proc_dic).split("<")
        
        doc_ids = [i for i, x in enumerate(sentences) if x.startswith(properties['doctag']) ]
        num_docs = len(doc_ids)
        for doc_id in range(num_docs):
            if doc_id == num_docs-1:
                current_start = doc_ids[doc_id]
                current_end   = len(sentences)
            else:
                current_start = doc_ids[doc_id]
                current_end   = doc_ids[doc_id+1] - 1
            current_doc   = sentences[current_start:current_end:1]
            topic_ids = [i for i, x in enumerate(current_doc) if x.startswith(properties['labeltag']) ]
            current_topic  = self.clean_text(current_doc[topic_ids[0]].strip().lower())
            current_topics = current_topic.split()        
            current_topics.pop(0)
            # Add topics
            if len(current_topics) > 0:
                for t in current_topics:
                    if t in topics:
                        topics[t].append(doc_id)
                    else:
                        topics[t] = []
                        topics[t].append(doc_id)
              
            # Add body contents          
            body_tags = properties['bodytags'].strip().split()
            tokens[doc_id] = []
            for body_tag in body_tags:
                body_ids = [i for i, x in enumerate(current_doc) if x.startswith(body_tag) ]
                if len(body_ids) > 0:            
                    current_body_text  = self.clean_text(current_doc[body_ids[0]].strip().lower())
                    body_text          = current_body_text.split()        
                    body_text.pop(0)  
                    words = " ".join(body_text).strip().lower().split()
                    for word in words:
                        tokens[doc_id].append(word)
        self.tokens = tokens
        self.topics = topics   
            





    def stopwords(self):
        """
        Removes stopwords from tokens based on input stopwords file
        Output: 
            tokens(KEY=doc_id,VALUE=list of words)
        """
        print('TMPY> Removing stopwords')     
        # Load stopwords from file
        stopwords = [line.strip() for line in open(self.properties['stopwords'])]
        # Remove stopwords token
        tokens = self.tokens
        for doc_id,words in tokens.items():
            for stopword in stopwords:
                words = [w for w in words if w != stopword]
            tokens[doc_id] = words    
        self.tokens = tokens
        
        
        
        
        
        
    def stem(self):
        """
        Performs stemming based on input stemmer dictionary
        Output: 
            tokens(KEY=doc_id,VALUE=list of words)
        """
        print('TMPY> Stemming tokens')
        # Load stem dictionary from file
        stem_dict = dict([line.split() for line in open(self.properties['stems'])])
        # Perform stemming
        tokens = self.tokens
        for doc_id,words in tokens.items():
            words = [stem_dict[word] if word in stem_dict else word for word in words]
            tokens[doc_id] = words
        self.tokens = tokens
    




   
    def mkdict(self,*args):
        """
        Generates dictionary from tokens
        Either uses 0, 1 or 2 arguments
            mkdict() - generates dictionary on entire dataset
            mkdict(count_threshold) - generates dictionary with words having frequency greater than threshold
            mkdict(topic,count_threshold), topic = string, count_threshold = int
        Output: 
            vocab - list of dictionary words
        """
        print('TMPY> Generating dictionary')
        vocab_dict  = {} 
        tokens = self.tokens
        if len(args) == 0:
            for doc_id,words in tokens.items():
                for word in words:
                    if word in vocab_dict:
                        vocab_dict[word] += 1
                    else:
                        vocab_dict[word] = 1
        elif len(args) == 1:
            temp_dict = {}
            size      = args[0]
            tokens    = self.tokens
            for doc_id,words in tokens.items():
                for word in words:
                    if word in temp_dict:
                        temp_dict[word] += 1
                    else:
                        temp_dict[word] = 1
            vocab_dict = {key: value for key, value in temp_dict.items() if value >= size}
        elif len(args) == 2:
            temp_dict = {}
            topic     = args[0]
            size      = args[1]
            if topic in self.topics:
                for doc in self.topics[topic]:
                    for word in self.tokens[doc]:
                        if word in temp_dict:
                            temp_dict[word] += 1
                        else:
                            temp_dict[word] = 1
            vocab_dict = {key: value for key, value in temp_dict.items() if value >= size}
        else:
            print('TMPY> ERROR: mkdict takes 0 or exactly 2 arguments, returning empty dictionary')
        self.vocab = list(vocab_dict.keys())
        
    
    
          
          
          
    
    def vectorize(self):
        """
        Removes stopwords from tokens
        Output: 
            bow(KEY=doc_id,VALUE=vector representation of document (list))
        """
        print('TMPY> Vectorizing documents')
        num_docs    = len(self.tokens)
        bow         = {}
        for doc_id,words in self.tokens.items():
            doc_vector = [0] * len(self.vocab)
            for word in words:
                if word in self.vocab:
                    idx = self.vocab.index(word)
                    doc_vector[idx] = doc_vector[idx] + 1
                else:
                    pass
            bow[doc_id] = doc_vector
        self.bow = bow
        

        