import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
import gensim.downloader as api

from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

class ExtractKeywords:
    def __init__(self):
        self.result = {}
        self.model = api.load("glove-twitter-100")
        self.pre_computed_embeddings = self.compute_embeddings()

    def compute_embeddings(self):
        ## initializing a list of keywords that are used as features
        features_dict = {}
        features_dict['display_size'] = [10.1, 11.6, 12.4, 13.0, 13.3, 13.4, 13.5, 13.6, 14.0, 14.1, 14.2, 14.5, 15.0, 15.3, 15.6, 16.0, 16.1, 16.2, 17.3, 18.0]
        features_dict['brand'] = ['acer', 'apple', 'asus', 'avita', 'axl', 'chuwi', 'dell', 'fujitsu', 'gigabyte', 'honor', 'hp', 'iball', 'infinix', 'jio', 'lenovo', 'lg', 'microsoft', 'msi', 'primebook', 'realme', 'samsung', 'tecno', 'ultimus', 'walker', 'wings', 'zebronics']
        features_dict['ram_memory'] = ['2gb', '4gb', '8gb', '12gb', '16gb', '32gb', '36gb']
        features_dict['processor_tier'] = ['celeron', 'core i3', 'core i5', 'core i7', 'core i9', 'core ultra 7', 'm1', 'm2', 'm3', 'other', 'pentium', 'ryzen 3', 'ryzen 5', 'ryzen 7', 'ryzen 9']

        for category, items in features_dict.items():
            self.precomputed_embeddings[category] = {
                item: self.get_embedding(str(item)) for item in items
        }
            
    def get_embedding(self,word):
        """
        This method returns the embeddings
        """
        try:
            return self.model[word]
        except KeyError:
            return np.zeros(self.model.vector_size)
        
    def preprocess_input(self, input):
        tokens = input.split(" ")
        stop_words = set(stopwords.words('english'))

        return [token for token in tokens if token not in stop_words]
    
    def get_results(self):
        return self.result

    def classify_tokens(self, tokens): # model does not need to be self. we can try passing different models and calculate accuracy
        results = []
        
        for token in tokens:
            embedding = self.get_embedding(token, self.model)
            
            if not np.any(embedding): # if there is no embedding returned, then we return 0
                results.append((token, "unknown", 0))
                continue
            
            # Calculate similarity for each category
            token_scores = {}
            for category, embeddings in self.precomputed_embeddings.items():
                similarities = self.calculate_cosine_similarity(embedding, embeddings)
                if similarities:
                    best_match = max(similarities, key=similarities.get)
                    token_scores[category] = (best_match, similarities[best_match])
            
            # Find the highest similarity category
            print(f"token scores are {token_scores}")
            if token_scores:
                final_category = max(token_scores, key=lambda x: token_scores[x][1])
                best_label, best_score = token_scores[final_category]

                if best_score > 0.7: ## our score is 0.5 for it to be classified as a label
                    results.append((token, final_category, best_score))
            else:
                results.append((token, "unknown", 0))
        
        return results
