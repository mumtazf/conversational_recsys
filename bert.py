from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

from transformers import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

class LaptopDataset(Dataset):
    def __init__(self, queries, descriptions, labels, tokenizer, max_length):
        self.queries = queries
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        query = self.queries[idx]
        description = self.descriptions[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            query,
            description,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


    def train():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)  
        df = pd.read_csv("enhanced_laptop_query_dataset.csv")
        dataset = LaptopDataset(df['query'].tolist(), df['description'].tolist(), df['label'].tolist(), tokenizer, max_length=128)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        ## training
        optimizer = AdamW(model.parameters(), lr=0.001)
        loss_fn = CrossEntropyLoss()

        model.train()
        for epoch in range(2):  # Number of epochs
            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # Update progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        from sklearn.metrics import classification_report

        # Example predictions
        preds = model(input_ids, attention_mask).logits.argmax(dim=1)
        print(classification_report(labels, preds))

        model_save_path = "bert_sequence_classification_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    def load_trained_model():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6).to(device)  # Adjust num_labels as needed

        model_load_path = "bert_sequence_classification_model.pth"
        model.load_state_dict(torch.load(model_load_path))

        # Set the model to evaluation mode
        model.eval()
        print(f"Model loaded from {model_load_path}")
        return model


    def use_model_for_prediction(model, tokenizer, input_text):
    # Tokenize the input text
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Make predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)

        return preds.item()

    # Example usage
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_text = "I am a CS student and need a heavy-duty laptop for engineering"
    model = load_trained_model()
    prediction = use_model_for_prediction(model, tokenizer, input_text)
    print(f"Prediction: {prediction}")