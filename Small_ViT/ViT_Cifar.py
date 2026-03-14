import torch
import numpy as np
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
from random import random
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from torch import Tensor
import torch.optim as optim
from torchinfo import summary
from pathlib import Path

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128 # BS de l'entrainement
batch_test = 128 # BS de la validation
epoch = 20 # Nombre d'epoch d'entrainement
lr = 3e-4 # learning rate
weight_decay = 0.05 
emb_dim = 128 # Dimension d'embedding
heads = 4 # Nombre de tête du MHSA
patch_size = 4 # Taille du patch séparant l'image à l'entrée du ViT
dropout = 0.1
n_layers = 4 # Nombre de Transformer
img_size = 32 # Taille de l'inuput (img_size x img_size x channels) avec channels = 3
out_dim = 100 # Nombre de catégories en sortie (CIFAR100 = 100 sorties)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Fonction pour observer le dataset
def show_images(images, num_samples=40, cols=8):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    idx = int(len(images) / num_samples)
    for i, img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples/cols) + 1, cols, int(i/idx) + 1)
            plt.imshow(to_pil_image(img[0]))
            plt.axis("off")

    plt.tight_layout()
    plt.show()

# 60000 images de 100 classes : 50 000 train - 10 000 test
train_data = CIFAR100(root="./Data", download=True, transform=transform, train=True)
test_data = CIFAR100(root="./Data", download=True, transform=transform, train=False)

#show_images(train_data)

#Loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_test, shuffle=True, num_workers=0)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, dropout=0.1, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_ratio * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self MHSA
        y = self.norm1(x)
        y, _ = self.mhsa(y, y, y)
        x = x + y

        # MLP
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + y

        return x

# Defining the model

class ViT(nn.Module):
    def __init__(self, ch=3):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = nn.Conv2d(ch, emb_dim, kernel_size=patch_size, stride=patch_size)
        
        # Pos encoding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        
        # Token pour classification
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer block
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, heads, dropout)
            for _ in range(n_layers)
        ])

        # Classification
        self.classification = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))

    def forward(self, img):
        # Patch embedding
        x = self.patch_embedding(img)
        x = x.flatten(2)              # (B, emb_dim, H*W)
        x = x.transpose(1, 2)
        b, n, _ = x.shape

        # Ajout du cls token à l'inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)

        # Position encoding
        x += self.pos_embedding[:, :(n + 1)] 

        # Transformers
        for block in self.blocks:
            x = block(x)
        
        x = x[:,0]
        #print(x.shape)
        # Output classification à partir du cls token
        return self.classification(x)

def main() -> None:
    # Training
    model = ViT().to(device)
    summary(model, input_size=(1, 3, img_size, img_size))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_model = model
    best_accuracy = 0

    for epoch in range(epoch):
        epoch_loss = 0
        model.train()
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            #print(f"Step numéro {step} sur {len(train_data)//128}") # Vérification du fonctionnement
        print(f">>> Epoch {epoch} train loss: ", epoch_loss)

        # Validation
        model.eval()
        epoch_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_loss+=loss.item()

                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total * 100
            print(f">>> Epoch {epoch} test loss: ", epoch_loss)
            print(f"Pourcentage de succès sur test : {accuracy}")

            # Enregistrement du meilleur modèle sur la validation
            if accuracy > best_accuracy:
                print(f"Model upgraded from {best_accuracy} to {accuracy}")
                best_accuracy = accuracy
                best_model = model

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents = True, exist_ok = True)
    MODEL_NAME = "Small_ViT_Test.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    print(f"Saving best model to: {MODEL_SAVE_PATH}")
    torch.save(best_model.state_dict(), f=MODEL_SAVE_PATH)

if __name__ == '__main__':
    main()
