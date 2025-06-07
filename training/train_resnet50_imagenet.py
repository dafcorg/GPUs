import time, datetime, os, math, torch, torchvision
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast

# ---------- Configuración ----------
DATA_DIR   = "/data"
BATCH_SIZE = 256                # H100 80 GB
EPOCHS     = 90                 # Imagenet recipe
LR_BASE    = 0.1 * BATCH_SIZE / 256   # lineal con batch
NUM_WORKERS= 8
OUTPUT_LOG = "resnet50_timings.log"
DEVICE     = "cuda"

# ---------- Dataset ----------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize(mean, std),
])

val_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize(mean, std),
])

train_ds = torchvision.datasets.ImageFolder(f"{DATA_DIR}/train", train_tf)
val_ds   = torchvision.datasets.ImageFolder(f"{DATA_DIR}/val",   val_tf)
train_dl = DataLoader(train_ds, BATCH_SIZE, True,  num_workers=NUM_WORKERS, pin_memory=True)
val_dl   = DataLoader(val_ds,   BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)

# ---------- Modelo ----------
model = torchvision.models.resnet50(weights=None).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler()

# ---------- utilidades de timing ----------
def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def fmt(sec): return str(datetime.timedelta(seconds=int(sec)))

log_lines = []
header = f"# Run start {now()} | batch={BATCH_SIZE} | epochs={EPOCHS}"
print(header); log_lines.append(header)

total_start = time.perf_counter()

for epoch in range(EPOCHS):
    epoch_start = time.perf_counter()
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # ---- validación rápida (top-1) ----
    model.eval(); correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE)
            with autocast():
                preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
    acc = correct / len(val_ds)

    scheduler.step()
    epoch_time = time.perf_counter() - epoch_start
    msg = (f"Epoch {epoch+1:02}/{EPOCHS} "
           f"| lr={scheduler.get_last_lr()[0]:.4f} "
           f"| val_acc={acc:.4f} "
           f"| time={fmt(epoch_time)}")
    print(msg); log_lines.append(msg)

total_time = time.perf_counter() - total_start
summary = f"Total time {fmt(total_time)}"
print(summary); log_lines.append(summary)

# ---------- guarda log ----------
with open(OUTPUT_LOG, "w") as f:
    f.write("\n".join(log_lines))
print(f"Timings guardados en {OUTPUT_LOG}")
