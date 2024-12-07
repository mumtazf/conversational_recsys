import numpy as np
import pandas as pd
import os

from gensim.models import KeyedVectors
import gensim.downloader as api

from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


import nltk
from nltk.corpus import stopwords

class ExtractKeywords:
    def __init__(self):
        self.result = {}
        self.model = self.initialize_model()
        self.precomputed_embeddings = self.compute_embeddings()
    
    def initialize_model(self):
        self.model_path = "glove-twitter-100.model"
        if not os.path.exists(self.model_path):
            self.model = api.load("glove-twitter-100")
            self.model.save(self.model_path)
        else:
            self.model = KeyedVectors.load(self.model_path)  
        return self.model  

    def get_phrase_embeddings(self, phrase):
        """
        This method is used to get phrase-level embeddings by taking the average of individual
        word embedding
        """
        words = phrase.split()

        vectors = [self.model[word] for word in words if word in self.model]
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.model.vector_size)

    def compute_embeddings(self):
        """
        We initialize a dictionary for relevant keywords. We then precompute embeddings for all the keywords
        """
        ## initializing a list of keywords that are used as features
        features_dict = {}
        precomputed_embeddings = {}

        features_dict['display_size'] = [10.1, 11.6, 12.4, 13.0, 13.3, 13.4, 13.5, 13.6, 14.0, 14.1, 14.2, 14.5, 15.0, 15.3, 15.6, 16.0, 16.1, 16.2, 17.3, 18.0]
        features_dict['brand'] = ['acer', 'apple', 'asus', 'avita', 'axl', 'chuwi', 'dell', 'fujitsu', 'gigabyte', 'honor', 'hp', 'iball', 'infinix', 'jio', 'lenovo', 'lg', 'microsoft', 'msi', 'primebook', 'realme', 'samsung', 'tecno', 'ultimus', 'walker', 'wings', 'zebronics']
        features_dict['ram_memory'] = ['2gb', '4gb', '8gb', '12gb', '16gb', '32gb', '36gb', 'fast']

        for category, items in features_dict.items():
            precomputed_embeddings[category] = {
                item: self.get_embedding(str(item)) for item in items
        }

        ## Special pre-processing for processor_tier:     
        # we calculate phrase embedding for processors because they aren't 1 word, instead they are a phrase    
        features_dict['processor_tier'] = ['celeron', 'core i3', 'core i5', 'core i7', 'core i9', 'core ultra 7', 'm1', 'm2', 'm3', 'other', 'pentium', 'ryzen 3', 'ryzen 5', 'ryzen 7', 'ryzen 9']
        precomputed_embeddings['processor_tier'] = {}

        for item in features_dict['processor_tier']:
            precomputed_embeddings['processor_tier'][item] = self.get_phrase_embeddings(str(item))
            
        return precomputed_embeddings
            
    def get_embedding(self,word):
        """
        This method returns the embeddings for each word
        """
        try:
            return self.model[word]
        except KeyError:
            return np.zeros(self.model.vector_size)
        
    def preprocess_input(self, input):
        """
        This method splits the input on the basis of white-space anre removes stopwords like 'the, and, a'
        """
        tokens = input.split(" ")
        stop_words = set(stopwords.words('english'))

        return [token for token in tokens if token not in stop_words]
    
    def get_results(self):
        return self.result
    
    def calculate_cosine_similarity(self, input_vector, category_embeddings):
        similarities = {}
        for label, vector in category_embeddings.items():
            if np.any(vector):
                similarity = cosine_similarity([input_vector], [vector])[0][0]
                similarities[label] = similarity
        return similarities

    def classify_tokens(self, user_input): # model does not need to be self. we can try passing different models and calculate accuracy
        results = []

        tokens = self.preprocess_input(user_input)
        
        for token in tokens:
            token_embedding = self.get_embedding(token)
            
            if not np.any(token_embedding): # if there is no embedding returned, then we return 0
                results.append((token, "unknown", 0))
                continue
            
            # Calculate similarity for each category
            token_scores = {}
            for category, embeddings in self.precomputed_embeddings.items():
                similarities = self.calculate_cosine_similarity(token_embedding, embeddings)
                if similarities:
                    best_match = max(similarities, key=similarities.get)
                    token_scores[category] = (best_match, similarities[best_match])
            
            # Find the highest similarity category
            #print(f"token scores are {token_scores}")
            if token_scores:
                final_category = max(token_scores, key=lambda x: token_scores[x][1])
                best_label, best_score = token_scores[final_category]

                if best_score >= 0.85: # setting the threshold for when a token can be labeled as something
                    self.result[final_category] = token
                    results.append((token, final_category, best_score))
            else:
                results.append((token, "unknown", 0))
        
        return results
    
    def get_yes_no_label(self, input):
        model = SentenceTransformer('all-MiniLM-L6-v2')

        input_embedding = model.encode([input])

        yes_phrases = [
            'yes', 
            'it matters', 
            'definitely', 
            'absolutely', 
            'of course', 
            'indeed', 
            'certainly', 
            'sure', 
            'affirmative'
        ]
        
        no_phrases = [
            'no', 
            'it does not matter', 
            'doesn\'t matter', 
            'not important', 
            'negative', 
            'nope', 
            'never mind', 
            'not really', 
            'not at all'
        ]
        
        yes_embeddings = model.encode(yes_phrases)
        no_embeddings = model.encode(no_phrases)

        yes_similarities = cosine_similarity(input_embedding, yes_embeddings)[0]
        no_similarities = cosine_similarity(input_embedding, no_embeddings)[0]

        max_yes_sim = np.max(yes_similarities)
        max_no_sim = np.max(no_similarities)
        
        # Determine classification
        if max_yes_sim > max_no_sim:
            return 'yes'
        else:
            return 'no'
       


if __name__ == "__main__":
    model = ExtractKeywords()
    user_input = input("Does it matter to you?")

    print(model.get_yes_no_label(user_input))