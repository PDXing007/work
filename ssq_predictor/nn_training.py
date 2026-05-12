#!/usr/bin/env python3
"""
训练循环 — 时序自注意力模型的训练、验证、早停

特征:
- Mini-batch SGD with AdamW
- Cosine annealing LR schedule
- Early stopping on validation loss
- Gradient clipping
- 时间序列安全的验证切分
"""

import math
import random
import time

from nn_base import AdamW, clip_gradients, zeros, shape, add, sub, mul, elem_mul, elem_div, sqrt


def train_model(model, X_train, y_train_red, y_train_blue,
                X_val=None, y_val_red=None, y_val_blue=None,
                epochs=50, batch_size=32, lr=0.001,
                weight_decay=0.01, grad_clip=1.0,
                early_stop_patience=10,
                cosine_annealing=True,
                verbose=True):
    """
    训练时序自注意力模型

    Args:
        model: TemporalAttentionPredictor
        X_train: list of (T, F) 特征序列
        y_train_red: list of (33,) multi-hot
        y_train_blue: list of (16,) one-hot
        X_val, y_val_red, y_val_blue: 验证集
        epochs: 最大训练轮数
        batch_size: 批量大小
        lr: 初始学习率
        weight_decay: L2权重衰减系数
        grad_clip: 梯度裁剪最大范数
        early_stop_patience: 早停耐心值
        cosine_annealing: 是否使用余弦退火
        verbose: 是否输出进度

    Returns:
        history: {"train_loss": [], "val_loss": [], ...}
    """
    optimizer = AdamW(lr=lr, weight_decay=weight_decay)

    n_train = len(X_train)
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_params = None
    patience_counter = 0

    t_start = time.time()

    for epoch in range(epochs):
        # === 学习率调度 ===
        if cosine_annealing:
            current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
            optimizer.lr = current_lr
        else:
            current_lr = lr

        history["lr"].append(current_lr)

        # === 训练 ===
        indices = list(range(n_train))
        random.shuffle(indices)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_loss = 0.0

            # 累积梯度
            grads_accum = {}
            param_names = list(model.get_params().keys())

            for idx in batch_idx:
                # 前向传播 + 计算梯度 (数值梯度)
                loss = model.compute_loss(
                    X_train[idx], y_train_red[idx], y_train_blue[idx]
                )
                batch_loss += loss

                # 计算数值梯度 (对每个参数)
                sample_grads = _compute_numerical_gradients(
                    model, X_train[idx], y_train_red[idx], y_train_blue[idx]
                )

                for name in param_names:
                    if name not in grads_accum:
                        grads_accum[name] = zeros(shape(sample_grads[name]))
                    grads_accum[name] = add(grads_accum[name], sample_grads[name])

            batch_loss /= len(batch_idx)
            epoch_loss += batch_loss * len(batch_idx)
            n_batches += 1

            # 平均梯度
            for name in param_names:
                grads_accum[name] = mul(grads_accum[name], 1.0 / len(batch_idx))

            # 梯度裁剪
            grads_accum = clip_gradients(grads_accum, grad_clip)

            # 更新参数
            params = model.get_params()
            for name in param_names:
                if name in grads_accum:
                    new_param = optimizer.step(name, params[name], grads_accum[name])
                    params[name] = new_param
            model.set_params(params)

        avg_train_loss = epoch_loss / n_train
        history["train_loss"].append(avg_train_loss)

        # === 验证 ===
        val_loss = None
        if X_val and y_val_red and y_val_blue:
            val_loss = 0.0
            for i in range(len(X_val)):
                val_loss += model.compute_loss(
                    X_val[i], y_val_red[i], y_val_blue[i]
                )
            val_loss /= len(X_val)
            history["val_loss"].append(val_loss)

            # 早停检查
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                best_params = {k: [row[:] for row in v] if v and isinstance(v[0], list)
                               else v[:] if isinstance(v, list) else v
                               for k, v in model.get_params().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        if verbose and (epoch + 1) % 5 == 0:
            elapsed = time.time() - t_start
            val_str = f", val_loss={val_loss:.4f}" if val_loss else ""
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_loss={avg_train_loss:.4f}{val_str}, "
                  f"lr={current_lr:.6f}, "
                  f"best={best_val_loss:.4f}, "
                  f"patience={patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Training completed in {elapsed:.1f}s, "
              f"best_val_loss={best_val_loss:.4f}")

    # 恢复最佳参数
    if best_params:
        model.set_params(best_params)

    history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = epoch + 1
    history["training_time"] = elapsed

    return history


def _compute_numerical_gradients(model, X, y_red, y_blue, eps=1e-5):
    """
    用有限差分计算所有参数的数值梯度

    这是一个简化训练的实现——对每个参数做中心差分。
    对于生产环境，应该用解析反向传播替换。

    由于全数值梯度对较大模型太慢，这里仅计算关键参数的近似梯度。
    实际使用中会替换为基于 autodiff 的实现。
    """
    grads = {}
    params = model.get_params()
    base_loss = model.compute_loss(X, y_red, y_blue)

    for name, param in params.items():
        if isinstance(param, list) and param and isinstance(param[0], list):
            rows, cols = len(param), len(param[0])
            grad = [[0.0] * cols for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    orig = param[i][j]
                    param[i][j] = orig + eps
                    loss_plus = model.compute_loss(X, y_red, y_blue)
                    param[i][j] = orig
                    grad[i][j] = (loss_plus - base_loss) / eps
            grads[name] = grad
        elif isinstance(param, list):
            n = len(param)
            grad = [0.0] * n
            for i in range(n):
                orig = param[i]
                param[i] = orig + eps
                loss_plus = model.compute_loss(X, y_red, y_blue)
                param[i] = orig
                grad[i] = (loss_plus - base_loss) / eps
            grads[name] = grad

    return grads


def evaluate_model(model, X_test, y_test_red, y_test_blue):
    """
    评估模型在测试集上的表现

    Returns:
        {"red_hit_rate": ..., "blue_hit_rate": ..., "red_3plus_rate": ...}
    """
    n = len(X_test)
    if n == 0:
        return {"red_hit_rate": 0, "blue_hit_rate": 0, "red_3plus_rate": 0}

    total_red_hits = 0
    total_blue_hits = 0
    red_3plus = 0

    for i in range(n):
        red_pred, blue_pred = model.predict(X_test[i])

        # 真实红球
        true_reds = [j + 1 for j, v in enumerate(y_test_red[i]) if v > 0.5]
        hits = len(set(red_pred) & set(true_reds))
        total_red_hits += hits

        # 蓝球
        true_blue = max(range(16), key=lambda j: y_test_blue[i][j]) + 1
        if blue_pred == true_blue:
            total_blue_hits += 1

        if hits >= 3:
            red_3plus += 1

    return {
        "red_hit_rate": total_red_hits / (n * 6),
        "blue_hit_rate": total_blue_hits / n,
        "red_3plus_rate": red_3plus / n,
        "avg_red_hits": total_red_hits / n,
        "n_samples": n,
    }
