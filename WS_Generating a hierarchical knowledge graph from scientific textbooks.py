# WS_progect 2019
# Generating a hierarchical knowledge graph from scientific textbooks


# PRE-PROCESSING

import sys
import re
import os
import treetaggerwrapper
from tqdm import tqdm


def get_data(data_folder):
    ''' 
    get data and return list of docs
    each doc is a list of paragraphs
    which is a list of sentences
    '''
    docs = []
    for file in tqdm(os.listdir(data_folder), desc="Retrieving documents"):
        with open(data_folder + "/" + file, "r") as txt_file:
            doc = " ".join([line.strip("\n") if line != "\n" else line for line in txt_file])
            paragraphs = [paragraph for paragraph in doc.split("\n") if paragraph != " "]
            sentences = [[sentence for sentence in re.split("[?.!]", paragraph) if sentence != " "]
                                         for paragraph in paragraphs]
            docs.append(sentences)
    print("%2d files found and retrieved" %len(docs))
    return docs

def tag_docs(docs):
    '''
    tag docs
    '''
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="en",
                             TAGDIR="/home/mathijs/Documents/Master/Thesis/Master-Thesis/tagger/")

    tagged_docs = []
    for doc in tqdm(docs, desc="Tagging documents   "):
        paragraphs = []
        for paragraph in doc:
            sentences = []
            for sentence in paragraph:
                tagged_sentence = [tuple(tag.split("\t")) for tag in tagger.tag_text(sentence)]
                sentences.append(tagged_sentence)
            paragraphs.append(sentences)
        tagged_docs.append(paragraphs)
    return tagged_docs


# 2. FILTERING
from nltk.corpus import stopwords as stopwords_nltk
stopwords = set(stopwords_nltk.words("english"))

def filterword(word):
    '''
    input:
        word - tuple in the form of (word, part-of-speech, stem)
    output:
        boolean - True if word passes the filter, False otherwise
    '''
    if word[2].lower() in stopwords:
        return False
    if re.search("[^a-zA-Z-\']", word[0]):
        return False
    if len(word[0]) <= 2:
        return False
    if word[1][:2] not in ["JJ", "NN"]:
        return False
    return True

def filter_docs(docs):
    '''
    input:
        docs - list of lists of lists containing documents, paragraphs and sentences respectively
               each sentence contains tuples of (word, part-of-speech, stem)
    output:
        docs in the same format, but with its words filtered
    '''
    return [[[[word for word in sentence if filterword(word)] 
                            for sentence in paragraph] 
                                        for paragraph in doc] 
                                               for doc in tqdm(docs, desc="Filtering words     ")]

    
# 3. MAIN METHOD
# Functions that are used by all 3 methods
def get_bigrams(words, docs):
    '''
    get all bigrams for every word with nouns and adjectives
    '''
    bigrams_per_word = {}
    for doc in tqdm(docs, desc="Constructing bigrams"):
        for paragraph in doc:
            for sentence in paragraph:
                
                sentence_filter = {x:filterword(x) for x in sentence}
                # don't actually filter out the words yet, because we need the
                # location in the sentence for creating bigrams
                sentence_clean = [x[0] for x in sentence]
    
                for word in words:
                    if word[0] in sentence_clean:
                        # get location of term
                        i = sentence_clean.index(word[0]) 
            
                        try:
                            if sentence_filter[sentence[i-1]]:
                                # if previous word matches filter,
                                # add bigram to the dictionary
                                if word in bigrams_per_word.keys():
                                    bigrams_per_word[word].append((sentence[i-1], word[0]))
                                else:
                                    bigrams_per_word[word] = [(sentence[i-1], word[0])]

                        except IndexError:
                            pass  # when previous word doesn't exist, proceed

                        try:
                            if sentence_filter[sentence[i+1]]:
                                # if next word matches filter,
                                # add bigram to the dictionary
                                if word in bigrams_per_word.keys():
                                    bigrams_per_word[word].append((word[0], sentence[i+1]))
                                else:
                                    bigrams_per_word[word] = [(word[0], sentence[i+1])]

                        except IndexError:
                            pass  # when previous word doesn't exist, proceed

    return bigrams_per_word

def get_top_bigrams(bigram_dict):
    '''
    get the most common bigrams for each word
    '''
    bigrams = []
    for word in sorted(list(bigram_dict), key=lambda x:x[1], reverse=True):
        bigrams.append(Counter(bigram_dict[word]).most_common(1))
    return bigrams

def get_top_bigrams_clean(bigram_dict):
    '''
    get the most common bigrams for each word (only word, not tuple)
    '''
    bigrams = []
    for word in sorted(list(bigram_dict), key=lambda x:x[1], reverse=True):
        word1, word2 = (Counter(bigram_dict[word]).most_common(1))[0][0]
        # the original word is stored as (word, pos, stem) tuple,
        # we need to find that and extract the word to make a clean
        # list of bigrams
        if isinstance(word1, tuple):
            word1 = word1[0]
        else:
            word2 = word2[0]
        bigrams.append((word1, word2))
    return bigrams


# Hierarchy Construction
def get_sentence_set(word, docs):
    '''
    retrieve the set of sentences that contain word
    '''
    sentences = set()
    for doc in docs:
        for paragraph in doc:
            for sentence in paragraph:
                clean_sentence = [x[0] for x in sentence]
                if word in clean_sentence:
                    sentences.add(sentence)
    return sentences

def get_sentence_sets(words, docs):
    '''
    retrieve the set of sentences for each word
    '''
    sentence_sets = {}
    for word in words:
        sentence_sets[word] = get_sentence_set(word, docs)
    return sentence_sets 



# 3a. TextRank
import itertools
from collections import Counter
import networkx as nx

def get_edges(docs, words=[]):
    '''
    get a list of co-occurrence edges based on tokens in doc
    optionally give a list of words to use
    '''
    edges = []
    for doc in tqdm(docs, desc="Constructing edges  "):
        for paragraph in doc:
            for sentence in paragraph:
                if len(words) > 0:
                    clean_sentence = [x[0] for x in sentence]
                    filtered_sentence = [word for word in clean_sentence if word in words]
                    edges.extend(list(itertools.combinations((word for word
                                                                 in filtered_sentence), 2)))
                else:
                    edges.extend(list(itertools.combinations((word[2] for word in sentence), 2)))
    return edges

def weigh_edges(edges):
    '''
    weigh edges based on count
    '''
    edges_counts = Counter(edges)
    return [(edge[0], edge[1], edges_counts[edge]) for edge in edges_counts.keys()]

def get_nodes(edges):
    '''
    get nodes based on edges
    '''
    a,b = zip(*edges)
    return list(set(a)|set(b))

def perform_textrank_method(docs):
    '''
    performs the textrank algorithm on the input.
    input:
        docs - list of lists of lists containing documents, paragraphs and sentences respectively
               each sentence contains tuples of (word, part-of-speech, stem)
    output:
        
    '''
    # construct edges and nodes
    edges = get_edges(docs)
    weighted_edges = weigh_edges(edges)
    nodes = get_nodes(edges)

    # create graph
    wtrg = nx.Graph() # Weighted TextRank Graph (wtrg)
    wtrg.add_nodes_from(nodes)
    wtrg.add_weighted_edges_from(weighted_edges)
    print("Graph constructed with %2d nodes and %2d edges" %(len(nodes), len(weighted_edges)))    

    # compute textrank
    print("Computing TextRank...")
    textranks = nx.pagerank(wtrg, weight="weight")
    textrank_unigram_results = sorted([(word, textranks[word]) for word in nodes],
                                            key=lambda x:x[1], reverse=True)[:100]    

    # get bigrams
    textrank_bigrams = get_top_bigrams_clean(get_bigrams(textrank_unigram_results, docs))
    textrank_bigrams_clean = [a+" "+b for a,b in textrank_bigrams]

    return textrank_bigrams_clean

# 3b. LDA
import gensim.corpora as corpora
import gensim
import math

def flatten_docs(docs):
    '''
    input:
        docs - list of lists of lists containing documents, paragraphs and sentences respectively
    output:
        flat_docs - list of documents, each element is a list of tokenised words 
    ''' 
    flat_docs = []
    for doc in tqdm(docs, desc="Preparing documents "):
        flat = []
        for paragraph in doc:
            for sentence in paragraph:
                try:
                    flat.extend([word[2] for word in sentence])
                except:
                    pass
        flat_docs.append(flat)
    return flat_docs

def get_topic_doc_clusters(lda_model, corpus):
    '''
    cluster documents based what topics they contain
    ''' 
    topic_doc_clusters = {}
    for i, doc in tqdm(enumerate(corpus), desc="test"):
        for topic, prob in lda_model.get_document_topics(doc):
            if prob >= 0.05:
    
                if topic not in topic_doc_clusters.keys():
                    topic_doc_clusters[topic] = [(i, doc)]
                else:
                    topic_doc_clusters[topic].append((i, doc))
    return topic_doc_clusters

def get_tfitfs(topic_doc_clusters, topic_words, w2id, corpus_dict):
    '''
    compute term frequency, inverse topic frequency

    insert formula here
    '''
    tfitfs = {}

    for ti in tqdm(sorted(list(topic_doc_clusters.keys())), desc="Computing TFITFs    "):
        topic_tfitf = {} 
        
        for term in topic_words[ti]:
            
            # tf
            # term frequency
            term_id = w2id[term]
            term_count = 0
            docs_w_no_term = 0
    
            for doc in topic_doc_clusters[ti]:
                try: # if the term exists, add the count to the total
                    term_count += corpus_dict[doc[0]][term_id]
                except KeyError:
                    docs_w_no_term += 1
            
            # amount of terms
            word_ids = [[y[0] for y in x[1]] for x in topic_doc_clusters[ti]]
            total_term_count = len(set([item for sublist in word_ids for item in sublist]))

            TF = term_count / total_term_count
            
            # itf
            doc_count = len(topic_doc_clusters[ti])
            
            try:
                ITF = math.log(doc_count/docs_w_no_term)
            except ZeroDivisionError:
                ITF = 0

            TFITF = TF*ITF

            topic_tfitf[term] = TFITF

        tfitfs[ti] = topic_tfitf
   
    return tfitfs
    
def get_top_tfitfs(word_scores, n=100):
    '''
    get n highest rated words based on tfitf, across all documents
    '''
    all_tfitfs = [list(scores.items()) for ti, scores in list(word_scores.items())]
    tfitfs_flat = [item for sublist in all_tfitfs for item in sublist]
    top_tfitfs = sorted(tfitfs_flat, key=lambda x:x[1], reverse=True)

    top_tfitfs_unique = []
    for term in top_tfitfs:
        if term[0] not in [x[0] for x in top_tfitfs_unique]:
            top_tfitfs_unique.append(term)
    
    return top_tfitfs_unique[:n]

def perform_lda_method(docs, n_topics):
    '''
    performs the lda algorithm on the input.
    input:
        docs - list of lists of lists containing documents, paragraphs and sentences respectively
               each sentence contains tuples of (word, part-of-speech, stem)
    output:
        
    '''
    # transform the documents to the right format for LDA
    flat_docs = flatten_docs(docs)

    id2w = corpora.Dictionary(flat_docs) # each word gets an id
    w2id = {y:x for x,y in list(id2w.items())} # convertion utility

    corpus = [id2w.doc2bow(doc) for doc in flat_docs] # tuples of id and occurrence count
    corpus_dict = {i:dict(x) for i,x in enumerate(corpus)} # idem but as dictionary

    # perform LDA
    print("Computing LDA...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                                id2word = id2w,
                                                num_topics = n_topics,
                                                random_state = 420,
                                                update_every = 1,
                                                chunksize = 100, # num of documents in each chunk
                                                passes = 10, # num of passes through corpus
                                                alpha = "auto", 
                                                per_word_topics = True)

    # group documents based on their topics
    topic_doc_clusters = get_topic_doc_clusters(lda_model, corpus)

    # get top words from each topic
    n_top_words = 100
    topic_words = {ti:[x[0] for x in lda_model.show_topic(ti, n_top_words)] for ti in 
                                                sorted(list(topic_doc_clusters.keys()))}

    # compute tfitf for each word
    word_scores = get_tfitfs(topic_doc_clusters, topic_words, w2id, corpus_dict)

    # get top 100 unique words based tfitf
    lda_unigram_results = get_top_tfitfs(word_scores)

    # get bigrams
    lda_bigrams = get_top_bigrams_clean(get_bigrams(lda_unigram_results, docs))
    lda_bigrams_clean = [a+" "+b for a,b in lda_bigrams]

    return lda_bigrams_clean


# 3b. Combination (LDA + TextRank)
   
def perform_combination_method(docs, n_topics):
    '''
    description
    ''' 
    # transform the documents to the right format for LDA
    flat_docs = flatten_docs(docs)

    id2w = corpora.Dictionary(flat_docs) # each word gets an id
    w2id = {y:x for x,y in list(id2w.items())} # convertion utility

    corpus = [id2w.doc2bow(doc) for doc in flat_docs] # tuples of id and occurrence count
    corpus_dict = {i:dict(x) for i,x in enumerate(corpus)} # idem but as dictionary

    # perform LDA
    print("Computing LDA...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                                id2word = id2w,
                                                num_topics = n_topics,
                                                random_state = 420,
                                                update_every = 1,
                                                chunksize = 100, # num of documents in each chunk
                                                passes = 10, # num of passes through corpus
                                                alpha = "auto", 
                                                per_word_topics = True)

    # group documents based on their topics
    topic_doc_clusters = get_topic_doc_clusters(lda_model, corpus)

    # get top words from each topic
    n_top_words = 100
    topic_words = {ti:[x[0] for x in lda_model.show_topic(ti, n_top_words)] for ti in 
                                                sorted(list(topic_doc_clusters.keys()))}


    # get words to be used for textrank
    textrank_words = [item for sublist in [topic_words[key] for key in topic_words] 
                                                                for item in sublist]

    # get edges
    edges = get_edges(docs, textrank_words)
    weighted_edges = weigh_edges(edges) 
    nodes = get_nodes(edges)

    # create graph
    wtrg = nx.Graph() # Weighted TextRank Graph (wtrg)
    wtrg.add_nodes_from(nodes)
    wtrg.add_weighted_edges_from(weighted_edges)
    print("Graph constructed with %2d nodes and %2d edges" %(len(nodes), len(weighted_edges)))    

    # compute textrank
    print("Computing TextRank...")
    textranks = nx.pagerank(wtrg, weight="weight")
    textrank_lda_unigram_results = sorted([(word, textranks[word]) for word in nodes],
                                            key=lambda x:x[1], reverse=True)[:100]    

    # get bigrams
    textrank_lda_bigrams = get_top_bigrams_clean(get_bigrams(textrank_lda_unigram_results, docs))
    textrank_lda_bigrams_clean = [a+" "+b for a,b in textrank_lda_bigrams]


    return textrank_lda_bigrams_clean


# ---------------------- # 
import datetime
import argparse

def main(args):
    # 1. PRE-PROCESSING
    # retrieve documents
    docs = get_data(args.data_folder)

    # tag and tokenise sentences
    tagged_docs = tag_docs(docs)

    # flattening list of documents
    flat_docs = flatten_docs(tagged_docs)


    # 2. FILTERING
    filtered_docs = filter_docs(tagged_docs) 

    
    # 3. MAIN METHOD
    if args.method == "textrank":
        results = perform_textrank_method(filtered_docs)

    elif args.method == "lda":
        results = perform_lda_method(filtered_docs, args.n_topics)

    else:
        results = perform_combination_method(filtered_docs, args.n_topics)

    
    # 4. RESULTS
    if args.results == "print":
        print("Top %2d results from the %s method:" %(args.n_results, args.method))
        for res in results[:args.n_results]:
            print(res)

    elif args.results == "save":
        
        location = "results/" + args.method + "_" + str(datetime.datetime.now().isoformat())  + ".txt"
        with open(location, "w") as file:
            for res in results[:args.n_results]:
                file.write(res + "\n")
        print("Results saved successfully under %s" %(location))




def check_results_value(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 100:
        raise argparse.ArgumentTypeError("%s is an invalid value, choose between 1 and 100" %value)
    return ivalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python program for extracting keyphrases from a folder of text files")
    parser.add_argument("data_folder", 
                        type=str, 
                        help="the location of the text files")
    parser.add_argument("method", 
                        type=str, 
                        choices=["textrank", "lda", "combination"], 
                        default="combination",
                        help="the method used for extracting keyphrases")
    parser.add_argument("n_topics",
                        type=int,
                        default=15,
                        help="the amount of topics to use for the LDA")
    parser.add_argument("results",
                        type=str,
                        choices=["print", "save"],
                        default="print",
                        help="what is done with the results")
    parser.add_argument("n_results",
                        type=check_results_value,
                        default=10,
                        help="the amount of results to be printed or saved")
    args = parser.parse_args()

    main(args)


