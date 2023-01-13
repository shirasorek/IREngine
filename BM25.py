import math
from itertools import chain
import time
import numpy as np
from pathlib import Path
import pickle


#with open(Path("general") / f'doc_title_to_all_term_fq.pickle', 'rb') as f:
#    term_frequencies = pickle.load(f)

class BM25_from_index:
    def __init__(self, index, index_name, DL, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        # self.words, self.pls = zip(*self.index.posting_lists_iter(index_name))
        self.idf = {}
        self.index_name = index_name
        self.DL = DL
        #self.term_frequencies=term_frequencies

    def calc_idf(self,
                 list_of_tokens):  # calculate the idf values according to the BM25 idf formula for each term in the
        # query
        idf = {}
        for term in list_of_tokens:
            n_ti = self.index.df[term]
            idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
        return idf

    def _score(self, freq, doc_id, term):  # calculate the bm25 score for given query and document returns float
        score = 0.0
        doc_len = self.DL[doc_id]
        numerator = self.idf[term] * freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
        score = (numerator / denominator)
        return score

    def search(self, query, N=3):  # This function calculate the bm25 score for given query and document (doc_id, score)
        between_calc={}
        res = []
        self.idf = self.calc_idf(query)
        words, pls = zip(*self.index.posting_lists_iter(self.index_name, query))
        for term in words:
            pos_list= pls[words.index(term)]
            for doc_id, freq in pos_list:
                if doc_id in between_calc:
                    between_calc[doc_id] += self._score(freq, doc_id, term)
                else:
                    between_calc[doc_id] = self._score(freq, doc_id, term)
        res = sorted([(doc_id, score) for doc_id, score in between_calc.items()], key=lambda x: x[1], reverse=True)

        return res

"""
    def _score(self, query, doc_id):  # calculate the bm25 score for given query and document returns float
        score = 0.0
        doc_len = self.DL[doc_id]
        self.idf = self.calc_idf(query)
        for term in query:
            freq= self.term_frequencies[doc_id][term]
            numerator = self.idf[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
            score += (numerator / denominator)
        return score

    def search(self, query, N=3):  # This function calculate the bm25 score for given query and document (doc_id, score)
        res = []
        words, pls = zip(*self.index.posting_lists_iter(self.index_name, query))
        for post in pls:
            for doc_id, freq in post:
                res.append((doc_id, self._score(query, doc_id)))
        res = sorted(list(dict.fromkeys(res)), key=lambda x: x[1], reverse=True)
"""