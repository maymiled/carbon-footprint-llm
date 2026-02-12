import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model_utils import CarbonDataset, CarbonAttentionModel

# --- CONFIG ---
FILE_PATH = "train_model/data/training_set_finish.csv"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 15
PATIENCE = 3 # Arrêt après 3 époques sans amélioration
MAX_LR = 2e-4

# --- 1. DATA PREP ---
df = pd.read_csv(FILE_PATH, sep=";")
df['FE.VAL'] = pd.to_numeric(df['FE.VAL'].astype(str).str.replace(',', '.'), errors='coerce')
df = df.dropna(subset=['FE.VAL']).query('`FE.VAL` > 0')

# Split 80/10/10
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_loader = DataLoader(CarbonDataset(train_df['DETAILS'].tolist(), train_df['FE.VAL'].tolist(), MODEL_NAME), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CarbonDataset(val_df['DETAILS'].tolist(), val_df['FE.VAL'].tolist(), MODEL_NAME), batch_size=BATCH_SIZE)
test_loader = DataLoader(CarbonDataset(test_df['DETAILS'].tolist(), test_df['FE.VAL'].tolist(), MODEL_NAME), batch_size=BATCH_SIZE)

# --- 2. WANDB INIT ---
wandb.init(project="carbon-qwen-final", config={"warmup": 0.1, "patience": PATIENCE, "lr": MAX_LR})

# --- 3. INIT MODEL ---
model = CarbonAttentionModel(MODEL_NAME).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=len(train_loader)*EPOCHS, pct_start=0.1)

best_val_loss = float('inf')
patience_counter = 0

# --- 4. TRAIN ---
for epoch in range(EPOCHS):
    model.train()
    t_loss = 0
    for b in train_loader:
        optimizer.zero_grad()
        out = model(b['ids'].to(DEVICE), b['mask'].to(DEVICE))
        loss = criterion(out.squeeze(), b['y'].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        t_loss += loss.item()
        wandb.log({"batch_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for b in val_loader:
            out = model(b['ids'].to(DEVICE), b['mask'].to(DEVICE))
            v_loss += criterion(out.squeeze(), b['y'].to(DEVICE)).item()
    
    avg_t, avg_v = t_loss/len(train_loader), v_loss/len(val_loader)
    wandb.log({"epoch": epoch, "train_loss": avg_t, "val_loss": avg_v})
    print(f"Epoch {epoch+1} | Train: {avg_t:.4f} | Val: {avg_v:.4f}")

    if avg_v < best_val_loss:
        best_val_loss = avg_v
        torch.save(model.state_dict(), "best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early Stopping! Pas d'amélioration depuis {PATIENCE} époques.")
            break

# --- 5. TEST FINAL ---
print("\nÉvaluation finale sur le Test Set...")
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
test_table = wandb.Table(columns=["Texte", "Réel", "Prédit"])

with torch.no_grad():
    for b in test_loader:
        out = model(b['ids'].to(DEVICE), b['mask'].to(DEVICE))
        real = torch.expm1(b['y']).cpu().numpy()
        pred = torch.expm1(out).cpu().squeeze().numpy()
        # On ajoute quelques exemples à WandB
        for i in range(min(3, len(real))):
            test_table.add_data("Extrait brut", real[i], pred[i])

wandb.log({"Test_Results_Table": test_table})
wandb.finish()