import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SEQUENCE DATASET — DUAL-SCALE  [CHANGE 4]
#   Each sample contains TWO sequence windows:
#     X_short  : (batch, SEQ_LEN_S=20, n_features)  — inflection detection
#     X_medium : (batch, SEQ_LEN_M=60, n_features)  — trend structure
#   Plus a scalar regime code per sample.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 6: DUAL-SCALE SEQUENCE DATASET  [v2: 20-day + 60-day]")
print("=" * 70)

feat_scaled_all   = scaler.transform(feat_clean.values)
label_arr         = labels_clean.values
regime_arr        = regime_clean.values.astype(np.int64)

def build_dual_sequences(X, y, regime, seq_short, seq_long):
    """
    Returns X_short (N, seq_short, F), X_medium (N, seq_long, F),
            y (N,), regime (N,)
    Indexed from seq_long onward so both windows are always valid.
    """
    Xs_s, Xs_m, ys, rs = [], [], [], []
    for i in range(seq_long, len(X)):
        Xs_s.append(X[i - seq_short: i])
        Xs_m.append(X[i - seq_long:  i])
        ys.append(y[i])
        rs.append(regime[i])
    return (np.array(Xs_s, dtype=np.float32),
            np.array(Xs_m, dtype=np.float32),
            np.array(ys,   dtype=np.int64),
            np.array(rs,   dtype=np.int64))

Xs_s, Xs_m, y_seq, r_seq = build_dual_sequences(
    feat_scaled_all, label_arr, regime_arr, SEQ_LEN_S, SEQ_LEN_M)

# Chronological boundaries in sequence space (offset by SEQ_LEN_M warmup)
seq_split_tr  = split_tr  - SEQ_LEN_M
seq_split_val = split_val - SEQ_LEN_M

Xs_s_tr,  Xs_s_val,  Xs_s_te  = Xs_s[:seq_split_tr],  Xs_s[seq_split_tr:seq_split_val],  Xs_s[seq_split_val:]
Xs_m_tr,  Xs_m_val,  Xs_m_te  = Xs_m[:seq_split_tr],  Xs_m[seq_split_tr:seq_split_val],  Xs_m[seq_split_val:]
y_seq_tr, y_seq_val, y_seq_te  = y_seq[:seq_split_tr], y_seq[seq_split_tr:seq_split_val], y_seq[seq_split_val:]
r_seq_tr, r_seq_val, r_seq_te  = r_seq[:seq_split_tr], r_seq[seq_split_tr:seq_split_val], r_seq[seq_split_val:]

print(f"  Short sequences  : {Xs_s.shape}  (samples × {SEQ_LEN_S} × features)")
print(f"  Medium sequences : {Xs_m.shape}  (samples × {SEQ_LEN_M} × features)")
print(f"  Train / Val / Test : {len(y_seq_tr)} / {len(y_seq_val)} / {len(y_seq_te)}")

class DualSeqDataset(Dataset):
    def __init__(self, Xs, Xm, y, r):
        self.Xs = torch.tensor(Xs, dtype=torch.float32)
        self.Xm = torch.tensor(Xm, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.long)
        self.r  = torch.tensor(r,  dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.Xs[i], self.Xm[i], self.y[i], self.r[i]

train_ds = DualSeqDataset(Xs_s_tr,  Xs_m_tr,  y_seq_tr,  r_seq_tr)
val_ds   = DualSeqDataset(Xs_s_val, Xs_m_val, y_seq_val, r_seq_val)
test_ds  = DualSeqDataset(Xs_s_te,  Xs_m_te,  y_seq_te,  r_seq_te)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

class_counts  = np.bincount(y_seq_tr, minlength=4)
class_weights = torch.tensor(
    1.0 / (class_counts + 1e-6), dtype=torch.float32
).to(device)
class_weights = class_weights / class_weights.sum() * 4