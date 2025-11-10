import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_transformer(model, train_loader, val_loader, delta_scaler,
                      player_side_embedding, player_position_embedding, player_role_embedding,
                      n_epochs=10, lr=1e-3, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model.to(device)
    player_side_embedding.to(device)
    player_position_embedding.to(device)
    player_role_embedding.to(device)

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            X_batch, Y_batch, side_idx, pos_idx, role_idx = batch
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            side_idx = side_idx.to(device)
            pos_idx = pos_idx.to(device)
            role_idx = role_idx.to(device)

            optimizer.zero_grad()

            # Get embeddings
            side_emb = player_side_embedding(side_idx)
            pos_emb = player_position_embedding(pos_idx)
            role_emb = player_role_embedding(role_idx)

            # Concatenate embeddings to features
            X_input = torch.cat([X_batch, side_emb, pos_emb, role_emb], dim=-1)
            output = model(X_input)

            # Separate loss per feature
            loss_x = criterion(output[:,:,0], Y_batch[:,:,0]).mean()
            loss_y = criterion(output[:,:,1], Y_batch[:,:,1]).mean()
            loss_huber = 0.5 * (loss_x + loss_y)
            loss_huber.backward()
            optimizer.step()

            train_loss += loss_huber.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss_huber = 0.0
        val_mse_real = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                X_batch, Y_batch, side_idx, pos_idx, role_idx = batch
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                side_idx = side_idx.to(device)
                pos_idx = pos_idx.to(device)
                role_idx = role_idx.to(device)

                side_emb = player_side_embedding(side_idx)
                pos_emb = player_position_embedding(pos_idx)
                role_emb = player_role_embedding(role_idx)
                X_input = torch.cat([X_batch, side_emb, pos_emb, role_emb], dim=-1)

                output = model(X_input)

                loss_x = criterion(output[:,:,0], Y_batch[:,:,0]).mean()
                loss_y = criterion(output[:,:,1], Y_batch[:,:,1]).mean()
                val_loss_huber += 0.5 * (loss_x.item() + loss_y.item()) * X_batch.size(0)

                # Real-world RMSE in yards
                y_pred = delta_scaler.inverse_transform(output.cpu().numpy().reshape(-1,2))
                y_true = delta_scaler.inverse_transform(Y_batch.cpu().numpy().reshape(-1,2))
                val_mse_real += np.sum((y_pred - y_true)**2)
                total_samples += y_pred.shape[0]

        val_loss_huber /= len(val_loader.dataset)
        val_rmse_real = np.sqrt(val_mse_real / total_samples)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss_huber:.4f} | Val RMSE (yards): {val_rmse_real:.4f}")
