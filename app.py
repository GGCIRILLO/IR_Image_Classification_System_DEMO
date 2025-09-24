#!/usr/bin/env python3
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json, os
import chromadb
import re

import torch.nn as nn
import torchvision.models as models


# --- CONFIG ---
WEIGHTS_PATH = "best_weights.pth"
TEST_PATHS_JSON = "test_paths.json"
TEST_LABELS_JSON = "test_labels.json"
CLASSES_JSON = "classes.json"
DATABASE_PATH = "vector_db"   
IMG_SIZE = 224

# --- MODEL DEF ---
class IRVehicleNet(nn.Module):
    """ResNet18 modificato per estrazione embeddings da immagini IR"""

    def __init__(self, num_classes, embedding_dim=512, dropout_rate=0.5, freeze_backbone=True):
        super(IRVehicleNet, self).__init__()

        # Carica ResNet18 pre-trained
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        # Congela i layer della backbone (opzionale)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Sblocca sempre l'ultimo block + layer norm/fc (fine-tuning mirato)
        for param in list(self.backbone.layer4.parameters()):
            param.requires_grad = True

        # Numero di feature in uscita dalla backbone
        num_features = self.backbone.fc.in_features
        # Rimuove il classificatore
        self.backbone.fc = nn.Identity()  # type: ignore 

        # Head per embeddings
        self.embedding_head = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Classificatore finale
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Salva la dimensione degli embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x, return_embeddings=False):
        # 1) Feature extraction
        features = self.backbone(x)  # [batch, 512]

        # 2) Embedding projection
        embeddings = self.embedding_head(features)  # [batch, embedding_dim]

        if return_embeddings:
            return embeddings  # solo embeddings

        # 3) Classificazione
        logits = self.classifier(embeddings)
        return logits, embeddings

    def get_embeddings(self, x):
        """Estrai solo gli embeddings (senza gradienti)"""
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            embeddings = self.embedding_head(features)
        return embeddings

# --- LOAD DATA ---
with open(TEST_PATHS_JSON) as f: TEST_PATHS = json.load(f)
with open(TEST_LABELS_JSON) as f: TEST_LABELS = json.load(f)
with open(CLASSES_JSON) as f: CLASSES = json.load(f)

def fix_path(p: str) -> str:
    # sostituisci "data/Chunk<number>/" con "data/processed/"
    return re.sub(r"data/Chunk\d+", "data/processed", p).replace(".png", "_orig.png")

TEST_PATHS = [fix_path(p) for p in TEST_PATHS]

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
model = IRVehicleNet(num_classes=len(CLASSES), embedding_dim=512).to(device)
state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- DB ---
client = chromadb.PersistentClient(path=DATABASE_PATH)
collection = client.get_or_create_collection(name="ir_vehicles")

# --- FUNZIONI ---
def get_embedding_from_path(path: str):
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device) # type: ignore
    with torch.inference_mode():
        emb = model.get_embeddings(img_t)
        emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()[0], img

# --- STREAMLIT UI ---
st.set_page_config(page_title="IR Retrieval Demo", layout="wide")
st.title("IR Image Classification System")

# Scelta immagine
options = [f"{i} - {CLASSES[label]}" for i, label in enumerate(TEST_LABELS)]
idx = st.selectbox("Seleziona un'immagine di test:", options, index=0)

# Recupera indice reale dalla stringa selezionata
idx = int(idx.split(" - ")[0])
sel_path = TEST_PATHS[idx]
sel_label = TEST_LABELS[idx]

# Mostra immagine query
st.subheader(f"Query (Classe: {CLASSES[sel_label]})")
q_img = Image.open(sel_path).convert("RGB")
st.image(q_img, caption=os.path.basename(sel_path), width=300)

# Bottone query                
if st.button("Esegui Query"):
    q_emb, _ = get_embedding_from_path(sel_path)
    results = collection.query(query_embeddings=[q_emb], n_results=5)

    st.subheader("Risultati più simili:")
    if not results["ids"]:
        st.warning("⚠️ Nessun risultato trovato. Hai popolato ChromaDB?")
    else:
        for i, (md, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):  # type: ignore
            label_idx = md.get('label', '?')
            if isinstance(label_idx, int) and 0 <= label_idx < len(CLASSES):
                title = CLASSES[label_idx]
            else:
                title = str(label_idx)

            # Rimappa il path salvato nel DB
            raw_path = str(md.get("path")) if md.get("path") else None
            img_path = fix_path(raw_path) if raw_path else None

            sim = 1 - dist

            # Layout a 2 colonne: immagine a sinistra, info a destra
            col_img, col_meta = st.columns([1, 2])
            with col_img:
                st.image(
                    img_path if img_path and os.path.exists(img_path) else "https://via.placeholder.com/224",
                    width=200,
                )
            with col_meta:
                st.markdown(f"### Top {i+1}")
                st.write(f"**Classe:** {title}")
                st.write(f"**Similarità:** {sim:.2f}")
                st.write(f"**File:** {os.path.basename(img_path) if img_path else 'N/A'}")

            st.markdown("---")  # separatore tra i risultati