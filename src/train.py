import torch
import torch.nn as nn


def train_model(model, train_loader, val_loader, device,
                num_epochs=50, lr=0.001, patience=10):
    """
    Train the LSTM with early stopping and LR scheduling.

    Key design decisions:
      - MSELoss: standard for regression tasks.
      - Adam optimiser: adaptive LR, works well for RNNs.
      - ReduceLROnPlateau: halves LR when val loss plateaus.
      - Gradient clipping (max_norm=1.0): prevents exploding gradients
        that are common in RNNs/LSTMs due to backprop through many
        timesteps.
      - Early stopping: stock data is noisy — overfitting shows up fast
        as val loss diverges from train loss.

    Returns:
        history dict with 'train_loss' and 'val_loss' lists.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    model = model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_count = 0
    best_weights = None

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()

            # Gradient clipping — prevents exploding gradients in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        tl = train_loss / len(train_loader)
        vl = val_loss / len(val_loader)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        scheduler.step(vl)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: {tl:.6f} | Val: {vl:.6f}")

        # --- Early stopping ---
        if vl < best_val_loss:
            best_val_loss = vl
            best_weights = model.state_dict().copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    return history
