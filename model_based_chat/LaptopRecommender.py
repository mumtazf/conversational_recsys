import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)))))

from user import User

class LaptopRecommender:
    def __init__(self, user):
        self.df = pd.read_csv("/data/laptops.csv")
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
        
        if user_preferences['ram_memory']:
            # If RAM is specified, create a similarity score
            ram_sim = (self.df['ram'].astype(str) == str(user_preferences['ram_memory'])).astype(int)
            similarity_scores.append(ram_sim)
        
        # Display size matching
        if user_preferences.get['display_size']:
            # Compute similarity based on closeness of display size
            display_diff = np.abs(self.df['display_size'] - float(user_preferences['display_size']))
            display_sim = 1 / (1 + display_diff)
            similarity_scores.append(display_sim)
        
    def recommend(self, user_preferences, top_k=5):
        """
        This method calls the compute_similarity and then sorts the dataset to get the most
        similar recommendations
        """
        similarities = self.compute_similarity(user_preferences)
        self.df['similarity_score'] = similarities
        
        top_recommendations = self.df.sort_values('similarity_score', ascending=False).head(top_k)
        
        return top_recommendations

if __name__ == "__main__":
    recommender = LaptopRecommender(User())
    recommender.recommend({})
