import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, ConfusionMatrixDisplay)
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 8: TRAINING MODELS")
print("=" * 70)

def train_model(mode, epochs=EPOCHS, patience=PATIENCE, lr=LR):
    print(f"\n  Training: {mode.upper()}")
    model     = DualCNNGRUFusion(n_feat, mode=mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1  = -1
    best_weights = None
    no_improve   = 0
    history      = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xs, xm, yb, rb in train_dl:
            xs, xm, yb, rb = xs.to(device), xm.to(device), yb.to(device), rb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xs, xm, rb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_ds)

        # ── Validation (early stopping only — never touches test set) ────────
        model.eval()
        val_loss, all_preds, all_true = 0, [], []
        with torch.no_grad():
            for xs, xm, yb, rb in val_dl:
                xs, xm, yb, rb = xs.to(device), xm.to(device), yb.to(device), rb.to(device)
                logits   = model(xs, xm, rb)
                val_loss += criterion(logits, yb).item() * len(yb)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(yb.cpu().numpy())
        val_loss /= len(val_ds)
        val_acc   = accuracy_score(all_true, all_preds)
        val_f1    = f1_score(all_true, all_preds, average="macro", zero_division=0)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | TrLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
                  f"ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f}")

        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch}  (best ValF1={best_val_f1:.4f})")
            break

    # ── Final evaluation on held-out TEST set (best weights, touched once) ───
    model.load_state_dict(best_weights)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xs, xm, yb, rb in test_dl:
            xs, xm = xs.to(device), xm.to(device)
            rb = rb.to(device)
            all_preds.extend(model(xs, xm, rb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
    print(f"\n  ── Final TEST: {mode.upper()} ──")
    print(f"  Accuracy : {acc:.4f}   F1 Macro : {f1:.4f}")
    print(classification_report(all_true, all_preds,
                                target_names=[STATE_NAMES[i] for i in range(4)],
                                zero_division=0))
    return model, acc, f1, np.array(all_preds), np.array(all_true), history

results_nn = {}
for mode in ["cnn_only", "gru_only", "dual_cnn", "fusion"]:
    model, acc, f1, preds, trues, history = train_model(mode)
    results_nn[mode] = {
        "model": model, "acc": acc, "f1": f1,
        "preds": preds, "trues": trues, "history": history
    }