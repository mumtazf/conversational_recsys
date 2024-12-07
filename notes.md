Embedding and cosine similarity approach
--------------------

For cosine similarity, I used 0.8 as the threshold because the model would classify items like 'cause' 'brand' etc with 0.7 score. So we want something higher than that number

For example, with a 0.7 threshold, we have -- 
```
mumtaz> hmmm i want a dell laptop or a macbook. the budget can be around 1000. it should be fast cause I want to do some ML work on it
[('want', 'processor_tier', 0.73186064), ('dell', 'brand', 0.99999994), ('laptop', 'brand', 0.76949346), ('macbook.', 'unknown', 0), ('around', 'processor_tier', 0.8056171), ('1000.', 'unknown', 0), ('cause', 'processor_tier', 0.75946116), ('I', 'unknown', 0), ('want', 'processor_tier', 0.73186064), ('ML', 'unknown', 0), ('work', 'processor_tier', 0.7347539)]
```

With a 0.8 threshold, we have 
mumtaz> hmmm i want a dell laptop or a macbook. the budget can be around 1000. it should be fast cause I want to do some ML work on it
[('dell', 'brand', 0.99999994), ('macbook.', 'unknown', 0), ('around', 'processor_tier', 0.8056171), ('1000.', 'unknown', 0), ('I', 'unknown', 0), ('ML', 'unknown', 0)]

With a 0.9 threshold we have
mu> hmmm i want a dell laptop or a macbook. the budget can be around 1000. it should be fast cause I want to do some ML work on it
[('dell', 'brand', 0.99999994), ('macbook.', 'unknown', 0), ('1000.', 'unknown', 0), ('I', 'unknown', 0), ('ML', 'unknown', 0)]


### Heuristics
1. Based on the parsing, I see that the model is able to predict budget as an "unknown" entity. So in our `refine_results()`, I parse through that and set it as our budget. I observed it through these results:

user_query = im looking for a dell laptop in the budget range of 1000-2000. 
named_entities = [('dell', 'brand', 0.99999994), ('1000-2000.', 'unknown', 0), ('', 'unknown', 0)]


2. For processor_tier, it is hard to compare word embeddings because as the name suggests it is "word" and not necessarily "phrase". We need phrase embeddings for processors because their names are usually like core i9, core 

3. Through our initial human evaluation, we found that users didn't really know or understand or care much about processor_tiers. so i do additional pre-processing there to ask whether the user cares about it or not. 
   1. If a user doesn't care about processor types then we choose the 
most frequently occuring processors within our dataset as a default.
   2. For user intention about prcoessors -- Uses binary classification by implementing sentence-level transformer to detect whether the user's response said yes or no