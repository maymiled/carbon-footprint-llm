import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class CarbonDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer_name, max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        # On garde l'instruct, car Qwen en a besoin pour savoir quoi faire du texte
        self.instruction = "Instruct: Estime l'intensité carbone de cette ligne de base de données.\nQuery: "

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        # On envoie la ligne brute du CSV directement
        full_text = f"{self.instruction}{self.texts[idx]}<|endoftext|>"
        enc = self.tokenizer(full_text, padding='max_length', truncation=True, 
                             max_length=self.max_length, return_tensors='pt')
        
        target = torch.log1p(torch.tensor(self.labels[idx], dtype=torch.float))
        return {'ids': enc['input_ids'].flatten(), 'mask': enc['attention_mask'].flatten(), 'y': target}

    
    
class CarbonAttentionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # 1. Encodeur gelé
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for param in self.encoder.parameters(): 
            param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        
        # 2. Normalisation de sortie de l'encodeur
        self.post_encoder_norm = nn.LayerNorm(hidden_size)
        
        # 3. Attention avec normalisation dédiée
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # 4. Régresseur avec normalisation d'entrée
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256), # Normalisation intermédiaire
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, ids, mask):
        # Extraction des caractéristiques (Frozen)
        with torch.no_grad():
            outputs = self.encoder(input_ids=ids, attention_mask=mask)
            lhs = outputs.last_hidden_state # [Batch, Seq, Hidden]
        
        # Normalisation après transformer
        lhs = self.post_encoder_norm(lhs)
        
        # Attention avec connexion résiduelle
        # On ajoute l'original (lhs) au résultat de l'attention pour garder l'info stable
        attn_out, _ = self.attention(lhs, lhs, lhs)
        lhs = self.attn_norm(lhs + attn_out) 
        
        # Pooling (Moyenne sur la séquence)
        pooled = lhs.mean(dim=1)
        
        return self.regressor(pooled)