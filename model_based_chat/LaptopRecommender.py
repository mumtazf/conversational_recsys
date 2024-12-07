import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from user import User

class LaptopRecommender:
    def __init__(self, dataset, user):
        self.df = pd.read_csv("data/laptops.csv")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.user = user
        
    def compute_similarity(self, user_preferences):        
        # Similarity scores for categorical features
        similarity_scores = []

        if user_preferences['brand'] != "":
            brand_match = (self.df['brand'].str.lower() == user_preferences['brand'].lower()).astype(int)
            similarity_scores.append(brand_match)

        if user_preferences['processor_tier'] != "":
            proc_embeddings = self.embedding_model.encode(self.df['processor_tier'].fillna('').tolist())
            user_proc_embedding = self.embedding_model.encode([user_preferences['processor_tier']])
            proc_sim = cosine_similarity(user_proc_embedding, proc_embeddings)[0]
            similarity_scores.append(proc_sim)

        if user_preferences['budget'] != "":
            budget_diff = np.abs(self.df['price (usd)'] - user_preferences['budget'])
            budget_sim = 1 / (1 + budget_diff / user_preferences['budget'])
            similarity_scores.append(budget_sim)
        


