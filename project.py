# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    time.sleep(5)
    book_req = requests.get(url)
    book_text = book_req.text
    
    book_text = re.sub(r"(\r\n)", r"\n", book_text)
    book_text = re.findall(r"\*\*\*(\n+[\S\s]*)\*\*\* END", book_text)
    
    return book_text[0]
# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    paras = re.split('\n{2,}', book_string)

    paras = [paragraph for paragraph in paras if paragraph.strip()]

    token = ['\x02']

    for i, paragraph in enumerate(paras):
        paragraph = paragraph.strip()

        para_tokens = re.findall(r'\w+|[^\w\s]', paragraph)

        token.extend(token for token in para_tokens if token.strip())

        if i < len(paras) - 1:
            token.append('\x03')

        if i < len(paras) - 1:
            token.append('\x02')

    token.append('\x03')

    return token
    
    # # Replace two or more newlines with '\x03\x02' to show paragraph breaks
    # book_string = re.sub(r'\n{2,}', '\x03\x02', book_string)
    
    # # Replace single newlines with a space
    # book_string = book_string.replace('\n', ' ')
    
    # # Tokenize words, numbers, and punctuation
    # tokens = re.findall(r'\w+|[^\w\s]', book_string)
    
    # # Insert '\x02' at the beginning and '\x03' at the end of the list
    # tokens.insert(0, '\x02')
    # tokens.append('\x03')
    
    # return tokens




# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        uniq_token = pd.Series(tokens).unique()
        probs = pd.Series(1/len(uniq_token), index = uniq_token)
        return probs
    
    def probability(self, words):
        prob = 1
        for word in words:
            prob *= self.mdl.get(word, 0)
        return prob
        
    def sample(self, M):
        words = self.mdl.sample(M, replace = True).index
        result_string = ''
        
        for word in words:
            result_string += word + " "
        
        result_string = result_string[:-1]
        return result_string


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        dict = {}
        
        for token in tokens:
            if token not in dict:
                dict[token] = 1
            else:
                dict[token] += 1 
        return pd.Series(dict) / len(tokens)
    
    def probability(self, words):
        if len(words) == 0:
            return None
        
        prob = 1
        
        for i in range(len(words)):
            if words[i] not in self.mdl.index:
                return 0 
            prob *= self.mdl.loc[words[i]]
        
        return prob
        
    def sample(self, M):
        return ' '.join(np.random.choice(self.mdl.index, p = self.mdl.values, size = M))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        data = []
        
        for i in range(len(tokens) - (self.N - 1)):
            ngram = tokens[i:i+self.N - 1]
            data.append(tuple(ngram))
        
        return data
            
                    
    def train(self, ngrams):
        # N-Gram counts C(w_1, ..., w_n)
        ...
        
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        ...

        # Create the conditional probabilities
        ...
        
        # Put it all together
        
        if len(ngrams) == 0:
            return pd.DataFrame()
        
        count_ngrams = pd.Series(ngrams).value_counts()
        n_grams = pd.Series([ngram[:-1] for ngram in ngrams]).value_counts()
    
        df_ngrams = pd.DataFrame({
            'ngram': ngrams,
            'n1gram': [ngram[:-1] for ngram in ngrams],
        })
        
        df_ngrams['prob'] = df_ngrams.apply(lambda row: count_ngrams[row['ngram']] / n_grams[row['n1gram']], axis=1)
        
        return df_ngrams.drop_duplicates()
    
    def probability(self, words):
        probs = 1
        
        for i in range(self.N - 1, len(words)):
            n_gram = tuple(words[i - (self.N - 1): i + 1])
            n1_gram = tuple(words[i - (self.N-1):i])
            
            conds = self.mdl[(self.mdl['ngram'] == n_gram) & (self.mdl['n1_gram'] == n1_gram)]
            
            if conds.shape[0] == 0:
                return 0 
            probs *= conds['prob'].values[0]
            
        
        if self.N == 2:
            mdl_prev = self.prev_mdl.mdl
            probs *= mdl_prev[words[0]]
        else:
            probs *= self.mdl_prev.probability(words[:self.N - 1])
        
        return probs
    
    def gen_tokens(self, master):
        if self.N == 2:
            mdl_pv = self.mdl
            length = mdl_pv[mdl_pv['n1gram'] == tuple(['\x02'])].set_index('ngram')['prob']
            choice = np.random.choice(length.index, p=length.values)
            master += choice
        else:
            self.prev_mdl.calc_next(master)
            n1gram = master[-self.N:]
            pv_mdl = self.mdl
            length = pv_mdl[pv_mdl['n1gram'] == tuple(n1gram)].set_index('ngram')['prob']
            choice = list(np.random.choice(length.index, p=length.values))
            master += [choice[-1]]

    def sample(self, M):
        
        options = []
        self.gen_tokens(options)
        for i in range(self.N, M):
            n1_gram = options[-(self.N-1):]
            mdl_pv = self.mdl
            length = mdl_pv[mdl_pv['n1gram'] == tuple(n1_gram)].set_index('ngram')['prob']
            choice = list(np.random.choice(length.index, p = length.values))
            options += [choice[-1]]
        options += ['\x03']
        
        s = ''
        for val in options:
            s += val + " "
        return s[:-1]
