from tkinter import X
from flask import Flask, request, jsonify
import re
import nltk
import collections
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
from pathlib import Path
import pickle
from inverted_index_gcp import InvertedIndex
from inverted_index_gcp import MultiFileReader
from inverted_index_gcp import MultiFileWriter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from BM25 import BM25_from_index

# regular indexs
inverted_title = InvertedIndex.read_index("title_index", "title_index")
inverted_anchor= InvertedIndex.read_index("anchor_index", "anchor_index")
inverted_body = InvertedIndex.read_index("body_index", "body_index")
# stemming indexs
inverted_stem_title = InvertedIndex.read_index("stem_title_index", "stem_title_index")
inverted_stem_body = InvertedIndex.read_index("stem_body_index", "stem_body_index")


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# reading pickel files for later use---------------------------------------------------------------------------------------

with open(Path("general") / f'doc_title_pairs_bykey.pickle', 'rb') as f: #returns the title of doc_id
    doc_title_pairs_bykey = pickle.load(f)

with open(Path("general") / f'DL_dict_body.pickle', 'rb') as f: #returns len of docs in corpus
    DL_body = pickle.load(f)

with open(Path("general") / f'DL_dict_title.pickle', 'rb') as f: #returns len of docs in corpus
    DL_title = pickle.load(f)
"""
with open(Path("general") / f'doc_nf.pickle', 'rb') as f: #returns norm of docs in corpus
    NF = pickle.load(f)
"""
with open(Path("general") / f'docs_tfidf.pickle', 'rb') as f: #returns norm of docs in corpus
    NF = pickle.load(f)
with open(Path("general") / f'page_view_stats.pickle', 'rb') as f: #returns page rank results
    pv = pickle.load(f)

with open(Path("general") / f'page_rank_dict.pickle', 'rb') as f:#returns page view results
    pr = pickle.load(f)

# help functions ----------------------------------------------------------------------------------------------------------

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD_OLD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?[\w,]?[\w.]?(?:['\-]?[\w,]?[\w])){0,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)

# original from Ass3
def tokenize(q):
    res = []
    list_of_tokens = [token.group() for token in RE_WORD_OLD.finditer(q.lower()) if token.group() not in all_stopwords]
    return list_of_tokens

# out tokenizer, diffrent RE expression & stemming
def stem_tokenize(q):
    res = []
    list_of_tokens = [token.group() for token in RE_WORD.finditer(q.lower()) if token.group() not in all_stopwords]
    stemmer = PorterStemmer()
    for token in list_of_tokens:
        res.append(stemmer.stem(token))
    return res


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing


def read_posting_list(inverted, w, index_name):  # finding the exact posting list for w from bin files
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, index_name)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def generate_query_tfidf_vector(query_to_search, index):
    epsilon = .0000001
    res = []
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
        df = index.df[token]
        idf = math.log((len(DL_body)) / (df + epsilon), 10)  # smoothing
        res.append(tf * idf)
    return res


def cosine_similarity_upgraded(index,
                               query):  # returns a dictionary of cosine similarity between docs in index by the terms of the query
    res = {}
    norm_q = 0
    tok_query = tokenize(query)
    query_vector = generate_query_tfidf_vector(tok_query, index)
    
    for term in tok_query:  # adding to res the overall weight between query and doc
        posting_list = read_posting_list(index, term, "body_index")
        for post in posting_list:
            if DL_body[post[0]]==0:
                continue           
            normlized_tfidf = (post[1] / DL_body[post[0]]) * math.log(len(DL_body) / index.df[term], 10)  # tfidf of doc_id
            if post[0] in res:
                res[post[0]] += query_vector[tok_query.index(term)] * normlized_tfidf
            else:
                res[post[0]] = query_vector[tok_query.index(term)] * normlized_tfidf

    # normalizing query
    norm_q = math.sqrt(sum([val ** 2 for val in query_vector]))

    # normalizing all sim with norm_q and norm_d
    for doc in res.keys():
        norm_d = NF[doc]
        res[doc] = res[doc]/( norm_q * norm_d)

    return res


def get_top_n(sim_dict, N=3):  # returns a ranked list of pairs (doc_id, score) in the length of N
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def merge_results(title_scores, body_scores, title_weight, text_weight, N=3): # function that merge and sort
    # documents retrieved by its weighte score
    for doc_id, score in title_scores:
        score = score*title_weight
    for doc_id, score in body_scores:
        score = score*text_weight
    merged = title_scores + body_scores
    res = sorted([(doc_id, score) for doc_id, score in merged], key=lambda x: x[1], reverse=True)[:N]

    return res

# normalization for search function, this is how we use Page Rank & Page View
def min_max_normalization(numbers, new_min, new_max):
    old_min = min(numbers)
    old_max = max(numbers)
    old_range = old_max - old_min
    new_range = new_max - new_min
    normalized = [(n - old_min) * new_range / old_range + new_min for n in numbers]
    return normalized

# start of search functions -----------------------------------------------------------------------------------------------------------

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    title_scores= search_title_BM25(query)
    body_scores= search_body_BM25(query)
    res_temp= merge_results(title_scores, body_scores, 0.45, 0.55, 100)
    ids = []
    # create list of IDs
    for i in range(len(res_temp)):
        ids.append(res_temp[i][0])
    # find PR & Pv with IDs list
    pr = search_pr(ids)
    pv = search_pv(ids)
    # normilazing PV PR resualts with title and body scores
    norm_pr = min_max_normalization(pr, res_temp[-1][1], res_temp[0][1])
    norm_pv = min_max_normalization(pv, res_temp[-1][1], res_temp[0][1])
    # creating final score for each doc with title, body, PR & PV scores
    for i in range(len(res_temp)):
        res_temp[i] = (res_temp[i][0], res_temp[i][1]*0.5+norm_pr[i]*0.3+norm_pv[i]*0.2)
    res = sorted(res_temp,key=lambda x: x[1], reverse=True)[:100]
    final_res=[]
    #get title of each ID
    for tup in res:
        title = doc_title_pairs_bykey[tup[0]]  # getting the doc title
        final_res.append((tup[0], title))  # adding the tuple(doc_id,title)
    # END SOLUTION
    return jsonify(final_res)

def search_pr(wiki_ids):
  res=[]
  for i in wiki_ids:
      res.append(pr[i])
  return res

def search_pv(wiki_ids):
  res =[]
  for i in wiki_ids:
     res.append(pv[i])
  return res

  
@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
       return jsonify(res)
    # BEGIN SOLUTION
    # find cosine similarity and get 100 best
    cosim = cosine_similarity_upgraded(inverted_body, query)
    temp_res = get_top_n(cosim, 100)
    #find titles of pages
    for tup in temp_res:
        title = doc_title_pairs_bykey[tup[0]]  # getting the doc title
        res.append((tup[0], title))  # adding the tuple(doc_id,title)
    print(res)
    # END SOLUTION
    return jsonify(res)


def search_body_BM25(query):
    res = []
    tok_query = stem_tokenize(query)
    bm25_title = BM25_from_index(inverted_stem_body, "stem_body_index", DL_body)
    res = bm25_title.search(tok_query)
    return res



@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    terms_in_docs = Counter()
    tok_query = tokenize(query)

    for token in tok_query:
        try:
            posting_token = read_posting_list(inverted_title, token,
                                              'title_index')  # getting the posting list for each term
            print(posting_token)
            for doc in posting_token:  # counting the number of accourences
                terms_in_docs[doc[0]] += 1
        except:  # term isnt in index_title
            continue
    # sorting by most common results
    temp = terms_in_docs.most_common()

    for tup in temp:
        title = doc_title_pairs_bykey[tup[0]]  # getting the doc title
        res.append((tup[0], title))  # adding the tuple(doc_id,title)
    # END SOLUTION

    return jsonify(res)




def search_title_BM25(query):
    res = []
    tok_query = stem_tokenize(query)
    bm25_title = BM25_from_index(inverted_stem_title, "stem_title_index", DL_title)
    res = bm25_title.search(tok_query)

    print(res)
    return res


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tok_query = tokenize(query)
    terms_linking_to_docs = Counter()

    for token in tok_query:
        print(token)
        try:
            posting_token = list(
                set(read_posting_list(inverted_anchor, token, "anchor_index")))  # getting the posting list for each term
            for doc in posting_token:  # counting the number of links to any document
                terms_linking_to_docs[doc[0]] += 1
        except:  # token isnt in anchor_index
            continue

    temp = terms_linking_to_docs.most_common()

    # get doc title for output
    for tup in temp:
        try:
            title = doc_title_pairs_bykey[tup[0]]  # getting the doc title
            res.append((tup[0], title))  # adding the tuple(doc_id,title)
        except:
            pass
    # END SOLUTION
    return jsonify(res)



@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    res =[]
    for i in wiki_ids:
      res.append(pr[i])
    
    # END SOLUTION
    return jsonify(res)



@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res =[]
    for i in wiki_ids:
      res.append(pv[i])
  
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
