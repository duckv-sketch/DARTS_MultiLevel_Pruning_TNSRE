
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from model_search import Network
from genotypes import Genotype, PRIMITIVES
from bdp_utils import ErrorHistory, compute_errors

metrics_log = []
best_config = {}

def log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f1, duration):
    metrics_log.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'duration_sec': duration
    })

def save_metrics_log_to_csv(filename="search_metrics_log.csv"):
    df = pd.DataFrame(metrics_log)
    df.to_csv(filename, index=False)
    print(f"[INFO] Metrics saved to '{filename}'")

def prune_operators(model, temperature=1.0, mu=0.25):
    with torch.no_grad():
        alphas = model.alphas_normal
        probs = F.softmax(alphas / temperature, dim=-1)
        threshold = mu / alphas.shape[1]
        model.alphas_normal[probs < threshold] = -1e9


def train_darts_search_bdp(train_loader, val_loader, num_classes, epochs=20, device='cuda', prune_every=5, pt=0.15, pv=0.05):
    global best_config
    model = Network(C=16, num_classes=num_classes, layers=6, criterion=nn.CrossEntropyLoss()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_w = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    optimizer_alpha = torch.optim.Adam(model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_w, T_max=epochs)

    error_history_train, error_history_val = ErrorHistory(prune_every), ErrorHistory(prune_every)
    best_val_acc, best_model_state, best_genotype = 0.0, None, None

    # Clear previous genotype log
    with open("searched_genotype_log.txt", "w") as f:
        f.write("")

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch + 1}/{epochs}] Starting...")
        start_time = time.time()
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds_train, all_targets_train = [], []

        for step, ((x_train, y_train), (x_val, y_val)) in enumerate(zip(train_loader, val_loader)):
            x_train, y_train = x_train.to(device).squeeze(-1), y_train.to(device)
            x_val, y_val = x_val.to(device).squeeze(-1), y_val.to(device)

            optimizer_alpha.zero_grad()
            logits_val = model(x_val)
            loss_alpha = criterion(logits_val, y_val)
            loss_alpha += 0.001 * (-torch.sum(torch.softmax(model.alphas_normal, dim=-1) *
                                              torch.log_softmax(model.alphas_normal, dim=-1)))
            loss_alpha.backward()
            optimizer_alpha.step()

            optimizer_w.zero_grad()
            logits_train = model(x_train)
            loss_w = criterion(logits_train, y_train)
            loss_w.backward()
            optimizer_w.step()

            preds = logits_train.argmax(dim=1)
            total_loss += loss_w.item()
            total_correct += (preds == y_train).sum().item()
            total_samples += y_train.size(0)
            all_preds_train += preds.cpu().tolist()
            all_targets_train += y_train.cpu().tolist()

            sid = range(step * y_train.size(0), (step + 1) * y_train.size(0))
            error_history_train.update(sid, compute_errors(model, [(x_train, y_train)], device)[0])
            error_history_val.update(sid, compute_errors(model, [(x_val, y_val)], device)[0])

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        train_precision = precision_score(all_targets_train, all_preds_train, average='macro', zero_division=0)
        train_recall = recall_score(all_targets_train, all_preds_train, average='macro', zero_division=0)
        train_f1 = f1_score(all_targets_train, all_preds_train, average='macro', zero_division=0)

        # Validation
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        all_preds_val, all_targets_val = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device).squeeze(-1), y_val.to(device)
                logits = model(x_val)
                val_loss += criterion(logits, y_val).item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_val).sum().item()
                val_samples += y_val.size(0)
                all_preds_val += preds.cpu().tolist()
                all_targets_val += y_val.cpu().tolist()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_samples
        val_precision = precision_score(all_targets_val, all_preds_val, average='macro', zero_division=0)
        val_recall = recall_score(all_targets_val, all_preds_val, average='macro', zero_division=0)
        val_f1 = f1_score(all_targets_val, all_preds_val, average='macro', zero_division=0)
        duration = time.time() - start_time

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        log_epoch_metrics(epoch+1, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f1, duration)

           # Parse and log genotype
        genotype_normal = parse_genotype(
                model.alphas_normal.detach().cpu().numpy(),
                steps=5
            )
        genotype_reduce = parse_genotype(
                model.alphas_reduce.detach().cpu().numpy(),
                steps=5,
                is_reduce=True
            )

        current_genotype = Genotype(
                normal=genotype_normal,
                normal_concat=list(range(5)),   # 5 intermediate nodes: 0–4
                reduce=genotype_reduce,
                reduce_concat=list(range(5))
            )


        with open("searched_genotype_log.txt", "a") as f:
            f.write(f"[Epoch {epoch + 1:02d}] Val Acc: {val_acc:.4f}\n")
            f.write(f"{str(current_genotype)}\n\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_genotype = current_genotype
            best_config = {
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'genotype': str(best_genotype)
            }

        scheduler.step()

        if (epoch + 1) % prune_every == 0 and epoch < 22:
            T_current = 1.5 * (0.95 ** epoch)
            prune_operators(model, temperature=T_current, mu=0.5)
            print(f"[Annealable Pruning] T = {T_current:.4f}")

            voe_train = error_history_train.compute_voe()
            voe_val = error_history_val.compute_voe()

            prune_T = set(sorted(voe_train, key=voe_train.get)[:int(len(voe_train)*pt)])
            prune_V = set(sorted(voe_val, key=voe_val.get, reverse=True)[:int(len(voe_val)*pv)])

            keep_T = list(set(range(len(train_loader.dataset.y))) - prune_T)
            keep_V = list(set(range(len(val_loader.dataset.y))) - prune_V)

            train_loader.dataset.X = train_loader.dataset.X[keep_T]
            train_loader.dataset.y = train_loader.dataset.y[keep_T]
            val_loader.dataset.X = val_loader.dataset.X[keep_V]
            val_loader.dataset.y = val_loader.dataset.y[keep_V]

            print(f"[Prune Epoch {epoch+1}] Pruned {len(prune_T)} train, {len(prune_V)} val")
            print(f"[After Prune] Remaining train samples: {len(keep_T)}, val samples: {len(keep_V)}")

    if best_model_state:
        torch.save(best_model_state, "best_model20.pth")
        with open("searched_genotype20.txt", "w") as f:
            f.write(str(best_genotype))
        with open("best_config.txt20", "w") as f:
            for k, v in best_config.items():
                f.write(f"{k}: {v}\n")
        model.load_state_dict(best_model_state)

    save_metrics_log_to_csv()
    print("\nFinal Searched Genotype:\n", best_genotype)
    return best_genotype, train_loader, val_loader

def parse_genotype(alpha, steps=5, is_reduce=False):
    gene = []
    start = 0
    for i in range(steps):
        end = start + i + 2
        W = alpha[start:end]

        edges = sorted(
            range(i + 2),
            key=lambda j: -max(W[j][k] for k in range(len(W[j])) if k != PRIMITIVES.index('none'))
        )[:2]

        for j in edges:
            k_best = None
            max_val = -1
            for k in range(len(PRIMITIVES)):
                if k != PRIMITIVES.index('none') and W[j][k] > max_val:
                    max_val = W[j][k]
                    k_best = k
            gene.append((PRIMITIVES[k_best], j))
        start = end

    if is_reduce:
        key_ops = [ 'sep_conv_1x5', 'dil_conv_1x5']
        present_ops = set(op for op, _ in gene)
        missing_ops = [op for op in key_ops if op not in present_ops]

        if len(missing_ops) > 0:
            to_force = missing_ops
            replaced = 0
            for idx, (op, inp) in enumerate(gene):
                if op in ['avg_pool_3x3', 'max_pool_3x3']:
                    gene[idx] = (to_force[replaced], inp)
                    replaced += 1
                    if replaced >= len(to_force):
                        break

    else:
        count = sum(1 for op, _ in gene if op == 'dil_conv_1x5')
        while count < 2:
            gene.append(('dil_conv_1x5', 0))
            count += 1

    return gene

