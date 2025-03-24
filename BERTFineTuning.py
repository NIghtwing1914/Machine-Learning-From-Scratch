import torch 
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Fine-tune BERT function
def fine_tune_bert(texts, labels, num_classes=2, epochs=3, batch_size=8, lr=5e-5):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    dataset = TextDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids, attention_mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
    
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    texts = ["I love deep learning!", "Transformers are amazing.", "AI is the future.", "BERT is great for NLP tasks."]
    labels = [1, 1, 0, 1]  # Binary classification example
    
    model, tokenizer = fine_tune_bert(texts, labels)
    
    # Save the model
    model.save_pretrained("fine_tuned_bert")
    tokenizer.save_pretrained("fine_tuned_bert")
    print("Model fine-tuned and saved successfully!")

def embed_sentence(sentence, model, tokenizer):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        output = model.bert(input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state.mean(dim=1)  # Mean pooling
        
    return embeddings

# Example usage of embedding a sentence
sentence = "BERT embeddings are useful for many NLP tasks."
embedding = embed_sentence(sentence, model, tokenizer)
print("Sentence embedding:", embedding)