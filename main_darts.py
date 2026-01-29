import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from data_loader_darts import get_dataloaders_8020  # <- dùng 80/20 chia dữ liệu
from darts_search_bdp import train_darts_search_bdp  # <- BDP NAS search
from model_build import FinalNetwork
from cell_plot import plot_cell

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_final_model(model, train_loader, val_loader, device, epochs=25):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.squeeze(-1)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = x.squeeze(-1)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_final_model.pt")

        scheduler.step()
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.legend(); plt.title("Loss Curve")
    plt.savefig("loss_curve.png"); plt.close()

    plt.figure()
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.legend(); plt.title("Accuracy Curve")
    plt.savefig("accuracy_curve.png"); plt.close()

def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).squeeze(-1), y.to(device)
            logits = model(x)
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
            y_true.extend(y.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)
    print("[✓] Saved confusion matrix to confusion_matrix.csv")

if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n[INFO] Loading 80/20 split data...")
    train_loader, val_loader, num_classes = get_dataloaders_8020(batch_size=16)

    print("[INFO] Running DARTS search with BDP...")
    searched_genotype, pruned_train_loader, pruned_val_loader = train_darts_search_bdp(
        train_loader, val_loader, num_classes,
        epochs=20, device=device,
        prune_every=5, pt=0.15, pv=0.05
    )

    print("[INFO] Visualizing searched cells...")
    plot_cell(searched_genotype, 'normal')
    plot_cell(searched_genotype, 'reduce')

    print("[INFO] Training final model from scratch on pruned data...")
    model = FinalNetwork(C=16, num_classes=num_classes, layers=6, genotype=searched_genotype).to(device)
    train_final_model(model, pruned_train_loader, pruned_val_loader, device=device, epochs=25)

    print("[INFO] Evaluating best model on validation set...")
    model.load_state_dict(torch.load("best_final_model.pt"))
    evaluate_model(model, pruned_val_loader, device)
