import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 256 # taille max du block d'input / prédiction
batch_size = 64  # BS du training
batch_test = 200 # BS du test
max_steps = 5000 # Nombre de step entrainant le modèle sur BS
test_steps = 200 # Nombre de step avant évaluation
lr = 1e-3 # learning rate
emb_dim = 128 # taille de l'embedding de chaque char
heads = 4 # Nombre de tête du MHSA
dropout = 0.2
n_layers = 4 # Nombre de couches Transformer


# Lecture du texte dataset
with open('./Data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Char unique dans le texte
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encodeur et décodeur
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(char):
    l = []
    for i in char:
        l.append(stoi[i])
    return l

def decode(list):
    l = ''
    for i in list:
        l = l + itos[i]
    return l

# Données d'entrainement et de validation torch (tensor)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

# Définition du DataLoader avec le batchsize
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Définition du Decodeur 

# Définition du mask pour le decoder
def generate_causal_mask():
    mask = torch.triu(torch.ones(block_size, block_size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(device)

class TransformerBlock_decoder(nn.Module):
    def __init__(self, emb_dim, heads, dropout=0.1, mlp_ratio=4):
        super().__init__()

        self.mhsa_masked = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.mask = generate_causal_mask()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_ratio * emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Masked attention
        _, S, _ = x.shape
        y, _ = self.mhsa_masked(x, x, x, attn_mask = self.mask[:S,:S])
        x = x + y
        x= self.norm1(x)

        # MLP
        y = self.mlp(x)
        x = x + y
        x = self.norm2(x)

        return x

class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,emb_dim)
        self.position_embedding_table = nn.Embedding(block_size,emb_dim)
        self.attn_blocks = nn.Sequential(*[TransformerBlock_decoder(emb_dim, heads, dropout = dropout) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(emb_dim)
        self.Linear_head = nn.Linear(emb_dim, vocab_size)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, S = x.shape # B = batch size, S = sequence lenght

        # Token embedding + position embedding
        tok_emb = self.token_embedding_table(x) # B, S, C with C = emb_dim
        pos_emb = self.position_embedding_table(torch.arange(S, device = device))
        x = tok_emb + pos_emb # Size B, S, C 

        # Transformer
        x = self.attn_blocks(x) # B,S,C
        x = self.norm_f(x) # B,S,C
        x = self.Linear_head(x) # B,S,vocab_size
        # Nous donne pour chaque numéro de la séquence S, la répartition du proba 
        # pour la prochaine lettre
        return x
    
    def generate(self, x, max_new_token):
        # On a un input (B, S) et on souhaite générer les S + max_new_token pour chaque B
        for _ in range(max_new_token):
            x_cond = x[:, -block_size:] # (B, S, C) On prend les block_size derniers éléments
            x_new = self(x_cond) # (B, S, vocab_size) On génère le prochain élément
            x_new = x_new[:,-1,:] # (B, vocab_size) On récupère le dernier élément (size emb_dim)
            prob = F.softmax(x_new, dim=-1) # (B, vocab_size) On calcul les probas pour déterminer quelle lettre est la plus probable
            x_next = torch.multinomial(prob, num_samples=1) # (B, 1) On récupère le numéro de colonne où la proba est la plus élevée
            x = torch.cat((x, x_next),dim=1) # (B, S + 1)
        return x

model = NanoGPT()
model = model.to(device)

# Print le nombre de paramètres dans le modèle
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Création de l'optim et de la fonction perte
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for steps in range(max_steps):
    epoch_loss = 0
    model.train()
    
    inputs, targets = get_batch('train')
    optimizer.zero_grad()
    outputs = model(inputs)

    outputs = outputs.view(batch_size*block_size, vocab_size)
    targets = targets.view(batch_size*block_size)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Affichage de la perte pour le dataset de train et de validation toutes les test_steps et avant la fin de l'entrainement
    if steps % test_steps == 0 or steps == max_steps - 1:
        torch.no_grad()
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(batch_test)
            for k in range(batch_test):
                inputs, targets = get_batch(split)
                outputs = model(inputs)
                outputs = outputs.view(batch_size*block_size, vocab_size)
                targets = targets.view(batch_size*block_size)
                loss = criterion(outputs, targets)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        print(f"step {steps}: train loss {out['train']:.4f}, val loss {out['val']:.4f}")
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(decode(model.generate(context, max_new_token=200)[0].tolist()))

# génération après entrainement
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_token=1000)[0].tolist()))
