import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

data = [
    (["The", "laptop", "has", "8gb", "RAM", "from", "Lenovo", "."], ["O", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O", "BRAND", "O"]),
    (["This", "Acer", "laptop", "comes", "with", "16gb", "RAM", "."], ["O", "BRAND", "O", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O"]),
    (["Apple", "MacBook", "has", "a", "13.3", "inch", "display", "."], ["BRAND", "BRAND", "O", "O", "DISPLAY_SIZE", "O", "O", "O"]),
    (["Lenovo", "offers", "an", "affordable", "laptop", "with", "4gb", "RAM", "and", "a", "14", "inch", "screen", "."], ["BRAND", "O", "O", "O", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O", "O", "DISPLAY_SIZE", "O", "O", "O"]),
    (["Samsung", "Galaxy", "Book", "has", "12gb", "RAM", "and", "a", "15.6", "inch", "display", "."], ["BRAND", "BRAND", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O", "O", "DISPLAY_SIZE", "O", "O", "O"]),
    (["I", "bought", "a", "dell", "laptop", "with", "32gb", "RAM", "."], ["O", "O", "O", "BRAND", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O"]),
    (["This", "HP", "laptop", "has", "a", "17.3", "inch", "screen", "and", "fast", "RAM", "."], ["O", "BRAND", "O", "O", "O", "DISPLAY_SIZE", "O", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O"]),
    (["Gigabyte", "laptops", "are", "available", "with", "36gb", "RAM", "."], ["BRAND", "O", "O", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O"]),
    (["Microsoft", "Surface", "Pro", "offers", "8gb", "RAM", "with", "a", "13.5", "inch", "display", "."], ["BRAND", "BRAND", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O", "O", "DISPLAY_SIZE", "O", "O", "O"]),
    (["Honor", "laptops", "come", "with", "a", "14.1", "inch", "screen", "and", "16gb", "RAM", "."], ["BRAND", "O", "O", "O", "O", "DISPLAY_SIZE", "O", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O"]),
    (["Asus", "ZenBook", "is", "equipped", "with", "a", "15.0", "inch", "display", "and", "12gb", "RAM", "."], ["BRAND", "BRAND", "O", "O", "O", "O", "DISPLAY_SIZE", "O", "O", "O", "RAM_MEMORY", "RAM_MEMORY", "O"])
]


from transformers import BertTokenizer

# Use the non-fast tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_data(data):
    sentences, labels = zip(*data)
    encodings = tokenizer(list(sentences), truncation=True, padding=True, is_split_into_words=True, return_tensors="pt")
    
    label_list = []
    for i, sentence in enumerate(sentences):
        word_labels = labels[i]
        # Manually align the labels with tokens
        tokenized_sentence = tokenizer.tokenize(" ".join(sentence))
        word_to_token_map = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        
        aligned_labels = []
        word_idx = 0
        print(word_to_token_map)
        for token in word_to_token_map:
            if token is not int:
                if token.startswith("##"):  # If token is a subword
                    aligned_labels.append(word_labels[word_idx])
                else:
                    aligned_labels.append(word_labels[word_idx])
                    word_idx += 1
        
        label_list.append(aligned_labels)
    
    return encodings, label_list


encodings, labels = encode_data(data)

class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = NERDataset(encodings, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load BERT pre-trained model for token classification
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(set([label for sublist in labels for label in sublist])))
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define training loop
model.train()
for epoch in range(2): #2 epochs is fine  
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()

        print(f"Epoch {epoch + 1} Loss: {loss.item()}")

