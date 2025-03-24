import torch
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings using CLS token
def get_bert_embeddings(sentences, tokenizer, model, device='cpu'):
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for sentence in sentences:
            encoded_input = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
            output = model(**encoded_input)
            cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
    return embeddings

# Example usage
sentences = ["I love programming.", "The movie was great!", "I dislike the weather."]
embeddings = get_bert_embeddings(sentences, tokenizer, model)

print("Embeddings shape:", len(embeddings), embeddings[0].shape)
