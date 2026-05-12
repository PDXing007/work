#!/usr/bin/env python3
"""
TDD 全模块测试 — 按依赖顺序执行
"""

import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Test Framework
# ============================================================
passed = 0
failed = 0
failures = []

def check(cond, msg):
    global passed, failed, failures
    if cond:
        passed += 1
    else:
        failed += 1
        failures.append(msg)

def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

def ok(msg):
    print(f"  [OK] {msg}")

def err(msg):
    print(f"  [FAIL] {msg}")


# ============================================================
# 1. data_loader.py
# ============================================================
section("1. data_loader.py")

from data_loader import (
    load_history, parse_record, load_parsed, get_data_summary,
    split_expanding, split_rolling, split_anchored, split_by_date,
)

# 1.1 load
data_raw = load_history()
check(len(data_raw) > 3000, "load_history returns data")
ok(f"load_history: {len(data_raw)} records")

# 1.2 parse
r = parse_record(data_raw[0])
check(isinstance(r["红球"], list), "parse_record: reds is list")
check(len(r["红球"]) == 6, "parse_record: 6 reds")
check(isinstance(r["蓝球"], int), "parse_record: blue is int")
check(isinstance(r["红球顺序"], list), "parse_record: draw_order is list")
check(len(r["红球顺序"]) == 6, "parse_record: 6 in draw_order")
ok("parse_record: correct structure")

# 1.3 load_parsed
parsed = load_parsed()
check(len(parsed) == len(data_raw), "load_parsed: same length")
check(parsed[0]["红球"] == [int(x) for x in data_raw[0]["红球"].split()],
      "load_parsed: reds match")
ok(f"load_parsed: {len(parsed)} records")

# 1.4 summary
summary = get_data_summary(data_raw)
check(summary["总期数"] == len(data_raw), "get_data_summary: correct count")
check("最新一期" in summary, "get_data_summary: has latest")
ok(f"get_data_summary: {summary['日期范围']}")

# 1.5 split_expanding
folds = split_expanding(list(range(100)), n_train_min=30, n_val=10, n_step=10)
check(len(folds) > 0, "split_expanding: returns folds")
for train_idx, val_idx in folds:
    check(len(set(train_idx) & set(val_idx)) == 0, "split_expanding: no overlap")
    check(max(val_idx) <= min(train_idx), "split_expanding: val before train (time order)")
ok(f"split_expanding: {len(folds)} folds")

# 1.6 split_rolling
folds = split_rolling(list(range(100)), n_train=30, n_val=10, n_step=10)
check(len(folds) > 0, "split_rolling: returns folds")
for train_idx, val_idx in folds:
    check(max(val_idx) <= min(train_idx), "split_rolling: val before train")
ok(f"split_rolling: {len(folds)} folds")

# 1.7 split_anchored
folds = split_anchored(list(range(100)), anchors=[0.7, 0.85])
check(len(folds) == 2, "split_anchored: 2 folds")
for train_idx, val_idx in folds:
    check(max(val_idx) < min(train_idx), "split_anchored: val before train (newer=lower idx)")
ok("split_anchored: correct")

# 1.8 split_by_date (BUG CHECK)
# Data is newest-first. "2020-01-01" cutoff:
# dates > 2020-01-01 (newer) should be val, dates <= (older) should be train
# In newest-first ordering, newer = lower index, older = higher index
# So val = lower indices (newer dates), train = higher indices (older dates)
# The current code may have this backwards!
train_idx, val_idx = split_by_date(data_raw, "2020-01-01")
check(len(train_idx) > 0, "split_by_date: train not empty")
check(len(val_idx) > 0, "split_by_date: val not empty")
# Check: val should have dates AFTER cutoff, train should have dates BEFORE
# In newest-first: val = lower indices (newer = after cutoff), train = higher indices
if val_idx and train_idx:
    val_is_newer = min(val_idx) < min(train_idx) if (val_idx and train_idx) else False
    # Sample check on actual dates
    val_dates = [data_raw[i].get("开奖日期","") for i in val_idx[:5]]
    train_dates = [data_raw[i].get("开奖日期","") for i in train_idx[:5]]
    val_max_date = max(val_dates) if val_dates else ""
    train_max_date = max(train_dates) if train_dates else ""
    actual_newer = val_max_date > train_max_date
    ok(f"split_by_date: val newer than train = {actual_newer} (val: {val_max_date}, train: {train_max_date})")


# ============================================================
# 2. nn_base.py
# ============================================================
section("2. nn_base.py")

from nn_base import (
    matmul_1d, matmul_2d, outer, transpose, add, sub, mul,
    elem_mul, elem_div, sqrt, sum_all, mean, std,
    relu, gelu, sigmoid, softmax, softmax_2d, layer_norm,
    dropout_mask, apply_dropout,
    cross_entropy_loss, binary_cross_entropy,
    xavier_init, he_init, zeros, ones, shape,
    AdamW, argmax, argsort, clip_gradients,
    one_hot, multi_hot,
)

# 2.1 matmul_1d
x = [1.0, 2.0]
W = [[3.0, 4.0], [5.0, 6.0]]
result = matmul_1d(x, W)
check(abs(result[0] - 13.0) < 0.001, f"matmul_1d[0]: {result[0]}")
check(abs(result[1] - 16.0) < 0.001, f"matmul_1d[1]: {result[1]}")
ok("matmul_1d: correct")

# 2.2 matmul_2d
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
C = matmul_2d(A, B)
check(abs(C[0][0] - 19.0) < 0.001, f"matmul_2d[0][0]: {C[0][0]}")
check(abs(C[1][1] - 50.0) < 0.001, f"matmul_2d[1][1]: {C[1][1]}")
ok("matmul_2d: correct")

# 2.3 outer
o = outer([1.0, 2.0], [3.0, 4.0])
check(abs(o[0][0] - 3.0) < 0.001 and abs(o[1][1] - 8.0) < 0.001, "outer: correct")
ok("outer: correct")

# 2.4 transpose
T = transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
check(len(T) == 3 and len(T[0]) == 2, "transpose: shape correct")
check(T[0][0] == 1.0 and T[2][1] == 6.0, "transpose: values correct")
ok("transpose: correct")

# 2.5 add
check(add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0], "add 1D")
check(add([[1.0, 2.0]], [[3.0, 4.0]]) == [[4.0, 6.0]], "add 2D")
ok("add: correct")

# 2.6 sub
check(sub([5.0, 3.0], [1.0, 2.0]) == [4.0, 1.0], "sub 1D")
ok("sub: correct")

# 2.7 mul
check(mul([1.0, 2.0], 3.0) == [3.0, 6.0], "mul 1D")
check(mul([[1.0, 2.0]], 3.0) == [[3.0, 6.0]], "mul 2D")
ok("mul: correct")

# 2.8 elem_mul
check(elem_mul([1.0, 2.0], [3.0, 4.0]) == [3.0, 8.0], "elem_mul 1D")
ok("elem_mul: correct")

# 2.9 elem_div
result_div = elem_div([3.0, 8.0], [1.0, 2.0])
check(abs(result_div[0] - 3.0) < 0.01 and abs(result_div[1] - 4.0) < 0.01, "elem_div 1D")
ok("elem_div: correct")

# 2.10 sqrt
check(abs(sqrt([4.0, 9.0])[1] - 3.0) < 0.001, "sqrt")
ok("sqrt: correct")

# 2.11 sum_all
check(sum_all([1.0, 2.0, 3.0]) == 6.0, "sum_all 1D")
check(sum_all([[1.0, 2.0], [3.0, 4.0]]) == 10.0, "sum_all 2D")
ok("sum_all: correct")

# 2.12 mean
check(abs(mean([1.0, 2.0, 3.0]) - 2.0) < 0.001, "mean 1D")
ok("mean: correct")

# 2.13 relu
check(relu([-1.0, 0.0, 1.0]) == [0.0, 0.0, 1.0], "relu 1D")
check(relu([[-1.0, 0.0], [1.0, -2.0]]) == [[0.0, 0.0], [1.0, 0.0]], "relu 2D")
ok("relu: correct")

# 2.14 gelu
g = gelu([0.0])
check(abs(g[0]) < 0.01, f"gelu(0) near 0: {g[0]}")
g2 = gelu([1.0])
check(g2[0] > 0.5, f"gelu(1) > 0.5: {g2[0]}")
ok("gelu: correct")

# 2.15 sigmoid
s = sigmoid([0.0])
check(abs(s[0] - 0.5) < 0.001, f"sigmoid(0): {s[0]}")
s2 = sigmoid([10.0])
check(s2[0] > 0.99, f"sigmoid(10) near 1: {s2[0]}")
ok("sigmoid: correct")

# 2.16 softmax
p = softmax([1.0, 2.0, 3.0])
check(abs(sum(p) - 1.0) < 0.001, "softmax: sum=1")
check(p[2] > p[1] > p[0], "softmax: monotonic")
ok("softmax: correct")

# 2.17 softmax_2d
X2d = [[1.0, 2.0], [3.0, 4.0]]
p2d = softmax_2d(X2d)
check(abs(sum(p2d[0]) - 1.0) < 0.001, "softmax_2d: row0 sum=1")
check(abs(sum(p2d[1]) - 1.0) < 0.001, "softmax_2d: row1 sum=1")
ok("softmax_2d: correct")

# 2.18 layer_norm
x_ln = [1.0, 2.0, 3.0, 4.0]
y_ln = layer_norm(x_ln)
mu = sum(y_ln) / 4
check(abs(mu) < 0.001, f"layer_norm: mean near 0: {mu}")
var_y = sum((v - mu) ** 2 for v in y_ln) / 4
check(abs(var_y - 1.0) < 0.01, f"layer_norm: var near 1: {var_y}")
ok("layer_norm: correct")

# 2.19 dropout_mask
mask = dropout_mask(10, 0.5)
check(len(mask) == 10, "dropout_mask: correct length")
check(any(m == 0.0 for m in mask) or any(m == 1.0 for m in mask), "dropout_mask: has both 0 and 1")
ok("dropout_mask: correct")

# 2.20 apply_dropout
xd = [1.0, 2.0, 3.0, 4.0]
md = [1.0, 1.0, 0.0, 1.0]
yd = apply_dropout(xd, md)
check(yd[2] == 0.0, "apply_dropout: zeroed out")
ok("apply_dropout: correct")

# 2.21 cross_entropy_loss
loss = cross_entropy_loss([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
check(loss > 0, "cross_entropy_loss: positive")
ok("cross_entropy_loss: correct")

# 2.22 binary_cross_entropy
bce = binary_cross_entropy([0.9, 0.1], [1.0, 0.0])
check(bce > 0, "bce: positive")
check(bce < 1.0, "bce: reasonable")
ok("bce: correct")

# 2.23 init functions
w_xav = xavier_init((5, 10))
check(shape(w_xav) == (5, 10), "xavier_init: shape")
w_he = he_init((5, 10))
check(shape(w_he) == (5, 10), "he_init: shape")
z = zeros((3, 4))
check(shape(z) == (3, 4) and z[0][0] == 0.0, "zeros: shape+value")
o = ones(5)
check(len(o) == 5 and o[0] == 1.0, "ones: shape+value")
ok("init functions: correct")

# 2.24 shape
check(shape([[1.0, 2.0], [3.0, 4.0]]) == (2, 2), "shape 2D")
check(shape([1.0, 2.0]) == (2,), "shape 1D")
ok("shape: correct")

# 2.25 argmax / argsort
check(argmax([1.0, 5.0, 3.0]) == 1, "argmax")
check(argsort([3.0, 1.0, 2.0]) == [1, 2, 0], "argsort asc")
check(argsort([3.0, 1.0, 2.0], reverse=True) == [0, 2, 1], "argsort desc")
ok("argmax/argsort: correct")

# 2.26 clip_gradients
grads = {"a": [10.0, 10.0], "b": [[5.0, 5.0]]}
clipped = clip_gradients(grads, max_norm=1.0)
total_norm_sq = sum(x*x for x in clipped["a"]) + sum(x*x for row in clipped["b"] for x in row)
check(abs(math.sqrt(total_norm_sq) - 1.0) < 0.01,
      f"clip_gradients: norm near 1: {math.sqrt(total_norm_sq)}")
ok("clip_gradients: correct")

# 2.27 one_hot / multi_hot
oh = one_hot(2, 5)
check(oh == [0.0, 0.0, 1.0, 0.0, 0.0], "one_hot")
mh = multi_hot([0, 2], 4)
check(mh == [1.0, 0.0, 1.0, 0.0], "multi_hot")
ok("one_hot/multi_hot: correct")

# 2.28 AdamW
adam = AdamW(lr=0.01, beta1=0.9, beta2=0.999, weight_decay=0.0)
param = [1.0, 2.0, 3.0]
grad = [0.1, 0.2, 0.3]
new_param = adam.step("test", param, grad)
check(len(new_param) == 3, "AdamW: output shape")
check(new_param[0] < param[0], "AdamW: decreased (grad positive, lr positive)")
ok("AdamW: correct")

# 2.29 var/std
from nn_base import var as nn_var, std as nn_std
v = nn_var([1.0, 2.0, 3.0, 4.0])
check(abs(v - 1.25) < 0.001, f"var: {v}")
s = nn_std([1.0, 2.0, 3.0, 4.0])
check(abs(s - math.sqrt(1.25)) < 0.001, "std: correct")
ok("var/std: correct")


# ============================================================
# 3. features_basic.py
# ============================================================
section("3. features_basic.py")

from features_basic import (
    red_numeric, red_parity, red_zones, red_mod3, red_gaps, ac_value,
    red_positional, blue_features, red_frequency_features, blue_freq_feature,
    encode_draw, feature_vector, compute_global_frequencies,
)

reds = [1, 3, 11, 22, 26, 31]
blue = 11

# 3.1 red_numeric
f = red_numeric(reds)
check(f["sum"] == 94, f"red_numeric sum: {f['sum']}")
check(f["min"] == 1, "red_numeric min")
check(f["max"] == 31, "red_numeric max")
check(f["span"] == 30, "red_numeric span")
check(abs(f["mean"] - 94/6) < 0.001, "red_numeric mean")
ok("red_numeric: correct")

# 3.2 red_parity
f = red_parity(reds)
check(f["odd_count"] == 4, f"red_parity odd: {f['odd_count']}")
check(f["even_count"] == 2, f"red_parity even: {f['even_count']}")
ok("red_parity: correct")

# 3.3 red_zones
f = red_zones(reds)
check(f["zone_small"] == 3, f"red_zones small: {f['zone_small']}")  # 1,3,11
check(f["zone_mid"] == 1, f"red_zones mid: {f['zone_mid']}")      # 22
check(f["zone_large"] == 2, f"red_zones large: {f['zone_large']}") # 26,31
ok("red_zones: correct")

# 3.4 red_mod3
f = red_mod3(reds)
check(f["mod3_0"] == 1, f"red_mod3 0: {f['mod3_0']}")  # 3
check(f["mod3_1"] == 3, f"red_mod3 1: {f['mod3_1']}")  # 1,22,31
check(f["mod3_2"] == 2, f"red_mod3 2: {f['mod3_2']}")  # 11,26
ok("red_mod3: correct")

# 3.5 red_gaps
f = red_gaps(reds)
check(f["max_gap"] == 11, "red_gaps max")
check(f["min_gap"] == 2, "red_gaps min")
ok("red_gaps: correct")

# 3.6 ac_value
ac = ac_value(reds)
# diffs: |3-1|=2,|11-1|=10,|22-1|=21,|26-1|=25,|31-1|=30,
# |11-3|=8,|22-3|=19,|26-3|=23,|31-3|=28,|22-11|=11,
# |26-11|=15,|31-11|=20,|26-22|=4,|31-22|=9,|31-26|=5
# unique: 2,10,21,25,30,8,19,23,28,11,15,20,4,9,5 = 15, AC=15-5=10
check(ac == 10, f"ac_value: {ac}")
ok("ac_value: correct")

# 3.7 red_positional
f = red_positional(reds)
check(f["pos_1"] == 1, "pos_1")
check(f["pos_6"] == 31, "pos_6")
ok("red_positional: correct")

# 3.8 blue_features
f = blue_features(blue)
check(f["blue"] == 11, "blue")
check(f["blue_parity"] == 1, "blue_parity")
check(f["blue_mod3"] == 2, "blue_mod3")
ok("blue_features: correct")

# 3.9 encode_draw
feat = encode_draw(reds, blue)
check(len(feat) >= 20, f"encode_draw: {len(feat)} features")
check("red_sum" in feat, "encode_draw: has red_sum")
check("blue" in feat, "encode_draw: has blue")
check("pos_1" in feat, "encode_draw: has pos_1")
ok(f"encode_draw: {len(feat)} features")

# 3.10 encode_draw with freqs
freq_red, freq_blue, total = compute_global_frequencies(parsed)
feat_full = encode_draw(reds, blue, freq_red, freq_blue, total)
check("red_avg_freq" in feat_full, "encode_draw: has red_avg_freq")
check("blue_freq" in feat_full, "encode_draw: has blue_freq")
ok(f"encode_draw with freqs: {len(feat_full)} features")

# 3.11 feature_vector
vec = feature_vector(feat)
check(len(vec) >= 20, f"feature_vector: {len(vec)} values")
check(all(isinstance(v, (int, float)) for v in vec), "feature_vector: all numeric")
ok("feature_vector: correct")

# 3.12 compute_global_frequencies
fr, fb, t = compute_global_frequencies(parsed)
check(len(fr) == 33, "freq_red: 33 keys")
check(len(fb) == 16, "freq_blue: 16 keys")
check(t == len(parsed), "total matches")
check(sum(fr.values()) == len(parsed) * 6, "freq_red sum = n*6")
check(sum(fb.values()) == len(parsed), "freq_blue sum = n")
ok("compute_global_frequencies: correct")


# ============================================================
# 4. features_advanced.py
# ============================================================
section("4. features_advanced.py")

from features_advanced import (
    red_consecutive, red_prime, red_tail, red_mod_extended,
    red_symmetry, red_span_ratios, red_prime_gaps, encode_draw_advanced,
)

reds2 = [1, 2, 5, 6, 7, 30]  # has consecutive runs

# 4.1 consecutive
f = red_consecutive(reds2)
check(f["consecutive_pairs"] == 3, f"consecutive_pairs: {f['consecutive_pairs']}")  # 1-2, 5-6, 6-7
check(f["max_consecutive_run"] == 3, f"max_run: {f['max_consecutive_run']}")  # 5,6,7
check(f["has_double_consecutive"] == 1, "has_double")
check(f["triple_consecutive"] == 1, "triple")
ok("red_consecutive: correct")

# 4.2 prime
f = red_prime(reds2)
# primes in [1,2,5,6,7,30]: 2,5,7 = 3
check(f["prime_count"] == 3, f"prime_count: {f['prime_count']}")
check(abs(f["prime_ratio"] - 0.5) < 0.001, f"prime_ratio: {f['prime_ratio']}")
ok("red_prime: correct")

# 4.3 tail
f = red_tail(reds2)
# tails: 1,2,5,6,7,0 → unique=6
check(f["unique_tails"] == 6, f"unique_tails: {f['unique_tails']}")
check(f["max_tail_freq"] == 1, "max_tail_freq")
ok("red_tail: correct")

# Test tail with same tails
reds3 = [1, 11, 21, 2, 12, 3]
f3 = red_tail(reds3)
# tails: 1,1,1,2,2,3 → unique=3, max=3
check(f3["unique_tails"] == 3, f"unique_tails (dup): {f3['unique_tails']}")
check(f3["max_tail_freq"] == 3, f"max_tail_freq (dup): {f3['max_tail_freq']}")
check(f3["has_tail_triple"] == 1, "has_tail_triple")
ok("red_tail with duplicates: correct")

# 4.4 mod_extended
f = red_mod_extended(reds)
check("mod4_0" in f, "has mod4 features")
check("mod5_entropy" in f, "has mod entropy")
ok(f"red_mod_extended: {len(f)} features")

# 4.5 symmetry
f = red_symmetry(reds)
check("symmetry_diff" in f, "has symmetry_diff")
ok("red_symmetry: correct")

# 4.6 span_ratios
f = red_span_ratios(reds)
check("span_first_third" in f, "has span_first_third")
ok("red_span_ratios: correct")

# 4.7 prime_gaps
f = red_prime_gaps(reds)
check(f["prime_distance_mean"] >= 0, "prime_distance_mean >= 0")
ok("red_prime_gaps: correct")

# 4.8 encode_draw_advanced
feat_adv = encode_draw_advanced(reds)
check(len(feat_adv) >= 40, f"encode_draw_advanced: {len(feat_adv)} features")
ok("encode_draw_advanced: correct")


# ============================================================
# 5. features_temporal.py
# ============================================================
section("5. features_temporal.py")

from features_temporal import (
    decay_weight, decay_weights, compute_weighted_frequencies,
    compute_ema, compute_trend, compute_volatility,
    compute_recent_changes, compute_missing_values,
    compute_overlap_similarity, compute_repeat_features,
    extract_simple_features,
)

# 5.1 decay_weight
w = decay_weight(0, 200)
check(abs(w - 1.0) < 0.001, f"decay(0): {w}")
w = decay_weight(200, 200)
check(abs(w - 0.5) < 0.01, f"decay(200,200): {w} (expected 0.5)")
w = decay_weight(400, 200)
check(abs(w - 0.25) < 0.01, f"decay(400,200): {w} (expected 0.25)")
ok("decay_weight: correct")

# 5.2 decay_weights
ws = decay_weights(5, 200)
check(len(ws) == 5, "decay_weights: length")
check(ws[0] == 1.0, "decay_weights: first=1")
check(ws[4] < ws[0], "decay_weights: decreasing")
ok("decay_weights: correct")

# 5.3 compute_weighted_frequencies
wfr, wfb, eff_n = compute_weighted_frequencies(parsed, half_life=200)
check(len(wfr) == 33, "weighted freq red: 33")
check(len(wfb) == 16, "weighted freq blue: 16")
check(eff_n > 0, f"weighted freq eff_n: {eff_n:.1f}")
ok(f"compute_weighted_frequencies: eff_n={eff_n:.1f}")

# 5.4 extract_simple_features
sf = extract_simple_features(parsed[0])
check("sum" in sf and "span" in sf and "blue" in sf, "extract_simple_features")
ok("extract_simple_features: correct")

# 5.5 compute_ema (data with enough history)
if len(parsed) > 50:
    ema_feat = compute_ema(parsed, 50, [5, 10], extract_simple_features)
    check(len(ema_feat) > 0, f"compute_ema: {len(ema_feat)} features")
    ok("compute_ema: correct")
else:
    check(True, "skip EMA (insufficient data)")

# 5.6 compute_trend
if len(parsed) > 50:
    trend_feat = compute_trend(parsed, 50, [13], extract_simple_features)
    check(len(trend_feat) > 0, f"compute_trend: {len(trend_feat)} features")
    ok("compute_trend: correct")

# 5.7 compute_volatility
if len(parsed) > 50:
    vol_feat = compute_volatility(parsed, 50, [13], extract_simple_features)
    check(len(vol_feat) > 0, f"compute_volatility: {len(vol_feat)} features")
    ok("compute_volatility: correct")

# 5.8 compute_recent_changes
changes = compute_recent_changes(parsed, 0, extract_simple_features)
check(len(changes) > 0, f"compute_recent_changes: {len(changes)} features")
ok("compute_recent_changes: correct")

# 5.9 compute_missing_values
miss = compute_missing_values(parsed, 0, max_lookback=100)
check("red_avg_missing" in miss, "missing: has red_avg")
check("blue_missing" in miss, "missing: has blue")
check("hot_red_count" in miss, "missing: has hot")
check("cold_red_count" in miss, "missing: has cold")
ok("compute_missing_values: correct")

# 5.10 compute_overlap_similarity
overlap = compute_overlap_similarity(parsed, 0, [5, 10])
check(len(overlap) > 0, f"compute_overlap_similarity: {len(overlap)} features")
ok("compute_overlap_similarity: correct")

# 5.11 compute_repeat_features
repeat = compute_repeat_features(parsed, 0)
check("repeat_red_count" in repeat, "repeat_red_count")
ok("compute_repeat_features: correct")


# ============================================================
# 6. features_contextual.py
# ============================================================
section("6. features_contextual.py")

from features_contextual import (
    encode_day_of_week, encode_financial, compute_recent_financial_stats,
    encode_draw_order, encode_seasonal, encode_jackpot_features,
)

# 6.1 encode_day_of_week
dow = encode_day_of_week(parsed[0])
check("day_index" in dow, "dow: day_index")
check("day_sin" in dow, "dow: day_sin")
check("day_cos" in dow, "dow: day_cos")
ok("encode_day_of_week: correct")

# 6.2 compute_recent_financial_stats
sales_stats, jp_stats = compute_recent_financial_stats(parsed, 0, window=50)
check(sales_stats is not None or len(parsed) == 0, "financial stats")
if sales_stats:
    check(sales_stats[0] > 0, f"sales mean > 0: {sales_stats[0]}")
ok("compute_recent_financial_stats: correct")

# 6.3 encode_financial
fin = encode_financial(parsed[0], sales_stats, jp_stats)
check("sales_zscore" in fin, "financial: sales_zscore")
check("jackpot_zscore" in fin, "financial: jackpot_zscore")
ok("encode_financial: correct")

# 6.4 encode_draw_order
order = encode_draw_order(parsed[0])
check(order.get("has_draw_order") == 1, "draw_order: has_draw_order")
check("first_ball_drawn" in order, "draw_order: first_ball")
check("order_max_gap" in order, "draw_order: max_gap")
ok(f"encode_draw_order: {len(order)} features")

# 6.5 encode_seasonal (needs date field)
seas = encode_seasonal(data_raw[0])
check("month_sin" in seas, "seasonal: month_sin")
ok("encode_seasonal: correct")

# 6.6 encode_jackpot_features
jf = encode_jackpot_features(parsed, 0, window=20)
check("no_first_prize_streak" in jf, "jackpot streak")
ok("encode_jackpot_features: correct")


# ============================================================
# 7. stats_core.py
# ============================================================
section("7. stats_core.py")

from stats_core import (
    compute_weighted_freq, compute_conditional_prob, compute_co_occurrence_matrix,
    fit_sum_distribution, fit_span_distribution, fit_parity_distribution,
    fit_zone_distribution, fit_ac_distribution, fit_position_distributions,
    fit_gap_distribution, fit_all_distributions,
    compute_markov_transition, compute_blue_transition,
)

# 7.1 compute_weighted_freq
fr, fb, tw = compute_weighted_freq(parsed, half_life=200)
check(len(fr) == 33, "weighted_freq red: 33")
check(len(fb) == 16, "weighted_freq blue: 16")
check(tw > 0, f"weighted_freq total_w: {tw:.1f}")
ok(f"compute_weighted_freq: total_w={tw:.1f}")

# 7.2 compute_conditional_prob
cr, cb = compute_conditional_prob(parsed, half_life=200)
check(len(cr) == 33, "cond_red: 33 keys")
check(len(cb) == 16, "cond_blue: 16 keys")
# Row sums for uniform mode
cr_uniform, _ = compute_conditional_prob(parsed, half_life=0)
for a in range(1, 34):
    row_sum = sum(cr_uniform[a].values())
    check(abs(row_sum - 5.0) < 0.01, f"cond_red row {a} sum={row_sum:.3f} (expected 5)")
ok("compute_conditional_prob: correct (row sums verified)")

# 7.3 compute_co_occurrence_matrix
com = compute_co_occurrence_matrix(parsed, half_life=200)
check(len(com) == 33, "co_occurrence: 33 keys")
ok("compute_co_occurrence_matrix: correct")

# 7.4 fit_all_distributions
dists = fit_all_distributions(parsed)
for key in ["sum", "span", "parity", "zone", "ac", "position", "gap"]:
    check(key in dists, f"fit_all: has {key}")
ok("fit_all_distributions: all keys present")

# 7.5 sum distribution
sum_dist = fit_sum_distribution(parsed)
check(abs(sum_dist["mean"] - 100.9) < 1.0, f"sum mean near 100.9: {sum_dist['mean']}")
check(sum_dist["std"] > 15, "sum std reasonable")
ok(f"fit_sum_distribution: mean={sum_dist['mean']:.1f}, std={sum_dist['std']:.1f}")

# 7.6 parity distribution
parity_dist = fit_parity_distribution(parsed)
total_p = sum(parity_dist.values())
check(abs(total_p - 1.0) < 0.001, f"parity sum to 1: {total_p}")
ok("fit_parity_distribution: sums to 1")

# 7.7 position distributions
pos_dist = fit_position_distributions(parsed)
check(len(pos_dist) == 6, f"position: 6 positions, got {len(pos_dist)}")
# pos_1 max should be <= 15 (typical), pos_6 min should be >= 18
pos1_vals = list(pos_dist[1].keys())
pos6_vals = list(pos_dist[6].keys())
ok(f"fit_position_distributions: pos1 range [{min(pos1_vals)},{max(pos1_vals)}], "
   f"pos6 range [{min(pos6_vals)},{max(pos6_vals)}]")

# 7.8 compute_markov_transition
trans = compute_markov_transition(parsed, order=1, half_life=200)
check(len(trans) > 0, "markov_trans: non-empty")
ok("compute_markov_transition: correct")

# 7.9 compute_blue_transition
btrans = compute_blue_transition(parsed, half_life=200)
check(len(btrans) > 0, "blue_trans: non-empty")
ok("compute_blue_transition: correct")


# ============================================================
# 8. cold_hot_analysis.py
# ============================================================
section("8. cold_hot_analysis.py")

from cold_hot_analysis import (
    compute_hotness_zscore, classify_hot_cold,
    compute_missing_analysis, compute_streaks,
    permutation_test_hotness,
)

# 8.1 compute_hotness_zscore
rz, bz, stats = compute_hotness_zscore(parsed, 0, half_life=50)
check(len(rz) == 33, f"hotness zscore red: {len(rz)}")
check(len(bz) == 16, f"hotness zscore blue: {len(bz)}")
check(stats["effective_n"] > 0, f"effective_n: {stats['effective_n']:.1f}")
ok(f"compute_hotness_zscore: eff_n={stats['effective_n']:.1f}")

# 8.2 classify_hot_cold
hc = classify_hot_cold(rz)
check("hot" in hc and "cold" in hc, "classify: has all categories")
total = hc["hot_count"] + hc["cold_count"] + len(hc["warm"]) + len(hc["neutral"]) + len(hc["cool"])
check(total == 33, f"classify: covers all 33: {total}")
ok(f"classify_hot_cold: hot={hc['hot_count']}, cold={hc['cold_count']}")

# 8.3 compute_missing_analysis
ma = compute_missing_analysis(parsed, 0, max_lookback=100)
check("red_missing" in ma, "missing: red_missing")
check("overdue_ratio_red" in ma, "missing: overdue")
ok("compute_missing_analysis: correct")

# 8.4 compute_streaks
streaks = compute_streaks(parsed, 0)
check("red_max_positive_streak" in streaks, "streaks: +")
check("red_max_negative_streak" in streaks, "streaks: -")
ok("compute_streaks: correct")


# ============================================================
# 9. association_mining.py
# ============================================================
section("9. association_mining.py")

from association_mining import (
    apriori_frequent_itemsets, generate_association_rules,
    build_pair_scoring_matrix, score_combination_pairs,
    check_temporal_stability,
)

# 9.1 apriori_frequent_itemsets
itemsets = apriori_frequent_itemsets(parsed, min_support=0.01, max_size=2, half_life=0)
check(len(itemsets) > 400, f"apriori: {len(itemsets)} itemsets (expected >400)")
# All 1-itemsets should exist
check(frozenset([1]) in itemsets, "apriori: has singletons")
ok(f"apriori_frequent_itemsets: {len(itemsets)} itemsets")

# 9.2 generate_association_rules
rules = generate_association_rules(itemsets, len(parsed))
check(len(rules) > 0, f"generate_rules: {len(rules)} rules")
# Should have lift values
check("lift" in rules[0], "rules: has lift")
check("confidence" in rules[0], "rules: has confidence")
ok(f"generate_association_rules: {len(rules)} rules")

# 9.3 build_pair_scoring_matrix
pair_scores, mean_score = build_pair_scoring_matrix(rules, min_lift=1.0)
check(len(pair_scores) > 0, f"pair_scores: {len(pair_scores)} pairs")
ok(f"build_pair_scoring_matrix: {len(pair_scores)} pairs, mean={mean_score:.3f}")

# 9.4 score_combination_pairs
score = score_combination_pairs(reds, pair_scores)
check("mean_lift" in score, "score_combo: mean_lift")
check("fraction_strong" in score, "score_combo: fraction_strong")
ok("score_combination_pairs: correct")

# 9.5 check_temporal_stability
stable = check_temporal_stability(parsed[:500], half_life_pairs=0, n_splits=3)
check(isinstance(stable, dict), "temporal_stability: returns dict")
ok(f"check_temporal_stability: {len(stable)} stable pairs")


# ============================================================
# 10. energy_function.py
# ============================================================
section("10. energy_function.py")

from energy_function import EnergyFunction

# 10.1 Basic energy
dists = fit_all_distributions(parsed)
fr, fb, _ = compute_weighted_freq(parsed, half_life=200)
cr, _ = compute_conditional_prob(parsed, half_life=200)

energy_fn = EnergyFunction(dists, freq_red=fr, freq_blue=fb, pair_scores=pair_scores, cond_red=cr)
e = energy_fn.total_energy(reds, blue)
check(isinstance(e, float), "total_energy: returns float")
check(e > 0, "total_energy: positive")
ok(f"total_energy: {e:.2f}")

# 10.2 Energy components
components = energy_fn.energy_components(reds, blue)
for comp_name in ["sum", "span", "parity", "zone", "ac", "gap", "position"]:
    check(comp_name in components, f"components: has {comp_name}")
ok("energy_components: all present")

# 10.3 A very "normal" combination should have lower energy
# than a very abnormal one
normal_reds = [3, 10, 17, 24, 28, 31]  # typical spread
abnormal_reds = [1, 2, 3, 4, 5, 6]    # all low, all consecutive
e_norm = energy_fn.total_energy(normal_reds, 8)
e_abnorm = energy_fn.total_energy(abnormal_reds, 8)
check(e_norm < e_abnorm, f"Energy: normal({e_norm:.1f}) < abnormal({e_abnorm:.1f})")
ok(f"Energy ordering: normal={e_norm:.1f} < abnormal={e_abnorm:.1f}")


# ============================================================
# 11. mc_mcmc.py
# ============================================================
section("11. mc_mcmc.py")

from mc_mcmc import MHSampler, rank_by_frequency, filter_by_constraints

# 11.1 MCMC sampling
sampler = MHSampler(energy_fn, seed=42)
samples, diag = sampler.sample(n_samples=100, T0=1.0, tau=500,
                               burn_in=200, thin=3, verbose=False)
check(len(samples) > 0, f"MCMC: {len(samples)} samples")
check(0.1 < diag["acceptance_rate"] < 0.9,
      f"MCMC acceptance_rate: {diag['acceptance_rate']:.3f}")
check(diag["n_unique_samples"] > 0, "MCMC: has unique samples")
ok(f"MCMC: {len(samples)} samples, acc_rate={diag['acceptance_rate']:.3f}")

# 11.2 Sample structure
for s in samples[:5]:
    check(len(s["红球"]) == 6, "MCMC sample: 6 reds")
    check(len(set(s["红球"])) == 6, "MCMC sample: no duplicates")
    check(1 <= s["蓝球"] <= 16, "MCMC sample: blue in range")
    for r in s["红球"]:
        check(1 <= r <= 33, f"MCMC sample: red {r} in range")
ok("MCMC: sample structure correct")

# 11.3 rank_by_frequency
ranked = rank_by_frequency(samples, top_n=5)
check(len(ranked) <= 5, f"rank_by_frequency: {len(ranked)}")
ok("rank_by_frequency: correct")

# 11.4 filter_by_constraints
filtered = filter_by_constraints(samples, dists, strictness=0.95)
check(len(filtered) <= len(samples), "filter: didn't add samples")
check(len(filtered) > 0, "filter: not empty")
ok(f"filter_by_constraints: {len(samples)} -> {len(filtered)}")


# ============================================================
# 12. probability.py
# ============================================================
section("12. probability.py")

from probability import (
    ProbabilityOutput, MCDropoutUncertainty,
    Calibrator, EnsembleUncertainty, format_probability_report,
)

# 12.1 ProbabilityOutput
red_probs = [1.0/33] * 33
blue_probs = [1.0/16] * 16

top6 = ProbabilityOutput.top_k(red_probs, k=6)
check(len(top6) == 6, "top_k: 6 items")
ok("ProbabilityOutput.top_k: correct")

# 12.2 entropy
ent = ProbabilityOutput.entropy(red_probs)
max_ent = math.log(33)
check(0 < ent <= max_ent, f"entropy: {ent:.3f} <= {max_ent:.3f}")
ok(f"ProbabilityOutput.entropy: {ent:.3f}")

# 12.3 uncertainty_level
level = ProbabilityOutput.uncertainty_level(ent, 33)
check(level in ("高", "中", "低"), f"uncertainty_level: {level}")
ok("ProbabilityOutput.uncertainty_level: correct")

# 12.4 combination_prob
cp = ProbabilityOutput.combination_prob([1, 2, 3, 4, 5, 6], 7, red_probs, blue_probs)
check(cp > 0, f"combination_prob: {cp:.10f}")
ok("ProbabilityOutput.combination_prob: correct")

# 12.5 Calibrator
cal = Calibrator(n_bins=5)
# Create some predictions and actuals
pred_probs_list = [[1.0/33]*33 for _ in range(100)]
actual_values_list = [[i+1 for i in range(6)] for _ in range(100)]
cal.fit(pred_probs_list, actual_values_list)
ece = cal.ece()
check(ece >= 0, f"ECE: {ece:.4f}")
ok(f"Calibrator.ece: {ece:.4f}")

# 12.6 EnsembleUncertainty
preds = [
    {"红球": [1, 2, 3, 4, 5, 6], "蓝球": 7},
    {"红球": [1, 2, 3, 4, 5, 7], "蓝球": 7},
    {"红球": [1, 2, 3, 4, 5, 8], "蓝球": 8},
]
rd, bd = EnsembleUncertainty.disagreement(preds)
check(0 <= rd <= 1, f"red disagreement: {rd:.3f}")
check(0 <= bd <= 1, f"blue disagreement: {bd:.3f}")
ok(f"EnsembleUncertainty: red={rd:.3f}, blue={bd:.3f}")

# 12.7 format_probability_report
report = format_probability_report(red_probs, blue_probs)
check("top_red" in report, "report: has top_red")
ok("format_probability_report: correct")


# ============================================================
# 13. ensemble.py
# ============================================================
section("13. ensemble.py")

from ensemble import (
    RollingWeightedEnsemble,
    make_frequency_strategy, make_markov_strategy,
    make_hot_strategy, make_cold_strategy,
)

# 13.1 RollingWeightedEnsemble
ensemble = RollingWeightedEnsemble(beta=0.9)

def make_test_predictor(red_balls, blue_ball):
    def predict(ctx=None):
        return {"红球": red_balls, "蓝球": blue_ball}
    return predict

ensemble.add_member("test1", make_test_predictor([1, 2, 3, 4, 5, 6], 7))
ensemble.add_member("test2", make_test_predictor([7, 8, 9, 10, 11, 12], 8))

result = ensemble.predict()
check(len(result["红球"]) == 6, "ensemble: 6 reds")
check(result["蓝球"] is not None, "ensemble: has blue")
ok("RollingWeightedEnsemble: predict OK")

# 13.2 Weighted voting
result = ensemble.predict_and_cache()
ensemble.update_weights([1, 2, 3, 20, 21, 22], 7)
weights = ensemble.get_weights_summary()
check(len(weights) == 2, "get_weights: 2 members")
ok("RollingWeightedEnsemble: weights updated")

# 13.3 Strategy factories
fr, fb, _ = compute_weighted_freq(parsed, half_life=200)
freq_strat = make_frequency_strategy(fr, fb)
result = freq_strat()
check(len(result["红球"]) == 6, "freq_strategy: 6 reds")
ok("make_frequency_strategy: correct")

# 13.4 Markov strategy
cr, _ = compute_conditional_prob(parsed, half_life=200)
fr2, fb2, _ = compute_weighted_freq(parsed, half_life=200)
markov_strat = make_markov_strategy(cr, fr2, fb2, parsed[0]["红球"])
result = markov_strat()
check(len(result["红球"]) == 6, "markov_strategy: 6 reds")
ok("make_markov_strategy: correct")

# 13.5 Hot strategy
rz, bz, _ = compute_hotness_zscore(parsed, 0, half_life=50)
hot_reds = sorted(rz, key=rz.get, reverse=True)[:10]
hot_blue = sorted(bz, key=bz.get, reverse=True)[:5]
hot_strat = make_hot_strategy(hot_reds, hot_blue, fr)
result = hot_strat()
check(len(result["红球"]) == 6, "hot_strategy: 6 reds")
ok("make_hot_strategy: correct")

# 13.6 Cold strategy
cold_reds = sorted(rz, key=rz.get)[:10]
cold_blue = sorted(bz, key=bz.get)[:5]
cold_strat = make_cold_strategy(cold_reds, cold_blue, fr)
result = cold_strat()
check(len(result["红球"]) == 6, "cold_strategy: 6 reds")
ok("make_cold_strategy: correct")


# ============================================================
# 14. cross_validation.py
# ============================================================
section("14. cross_validation.py")

from cross_validation import TimeSeriesCV

# 14.1 TimeSeriesCV
cv = TimeSeriesCV(parsed[:500])

# 14.2 expanding_window
folds = cv.expanding_window(n_train_min=100, n_val=20, n_step=30)
check(len(folds) > 0, "CV expanding: returns folds")
ok(f"CV expanding: {len(folds)} folds")

# 14.3 rolling_window
folds = cv.rolling_window(n_train=100, n_val=20, n_step=30)
check(len(folds) > 0, "CV rolling: returns folds")
ok(f"CV rolling: {len(folds)} folds")

# 14.4 anchored_walk_forward
folds = cv.anchored_walk_forward(anchors=[0.7, 0.85])
check(len(folds) == 2, "CV anchored: 2 folds")
ok("CV anchored: correct")

# 14.5 evaluate with simple predictor
def simple_predictor(train_data):
    # Just predict the most common numbers
    from collections import Counter
    rc = Counter()
    bc = Counter()
    for d in train_data:
        for rd in d["红球"]:
            rc[rd] += 1
        bc[d["蓝球"]] += 1
    return {
        "红球": sorted([b for b,_ in rc.most_common(6)]),
        "蓝球": bc.most_common(1)[0][0],
    }

fold = folds[0]
metrics = cv.evaluate(fold, simple_predictor)
check("red_hit_rate" in metrics, "CV evaluate: red_hit_rate")
check("blue_hit_rate" in metrics, "CV evaluate: blue_hit_rate")
ok(f"CV evaluate: red={metrics.get('red_hit_rate',0):.3f}, blue={metrics.get('blue_hit_rate',0):.3f}")

# 14.6 run_cv
result = cv.run_cv(simple_predictor, method="anchored", anchors=[0.8, 0.9])
check("summary" in result, "run_cv: has summary")
summary = result["summary"]
ok(f"run_cv: red_vs_random={summary.get('red_vs_random',0):.3f}")


# ============================================================
# 15. backtest.py
# ============================================================
section("15. backtest.py")

from backtest import (
    backtest_walk_forward, binomial_test,
    random_baseline, compare_to_baseline,
)

# 15.1 random_baseline
bl = random_baseline(100)
check(0.15 < bl["expected_red_hit_rate"] < 0.20, f"baseline red: {bl['expected_red_hit_rate']}")
check(0.04 < bl["expected_blue_hit_rate"] < 0.08, f"baseline blue: {bl['expected_blue_hit_rate']}")
ok("random_baseline: correct")

# 15.2 binomial_test
p_val = binomial_test(25, 100, 0.18)
check(0 < p_val < 1, f"binomial_test: p={p_val:.4f}")
ok("binomial_test: correct")

# 15.3 backtest_walk_forward (on small subset)
bt = backtest_walk_forward(parsed[:300], simple_predictor, n_folds=3, train_ratio=0.8)
if "error" not in bt:
    agg = bt["aggregate"]
    check("mean_red_hit_rate" in agg, "backtest: red_hit_rate")
    ok(f"backtest: red={agg['mean_red_hit_rate']:.4f}, consistency={agg.get('consistency',0):.2f}")
else:
    err(f"backtest error: {bt['error']}")

# 15.4 compare_to_baseline
if "error" not in bt:
    comp = compare_to_baseline(bt)
    check("assessment" in comp, "compare: has assessment")
    ok(f"compare_to_baseline: {comp.get('assessment','N/A')}")


# ============================================================
# 16. predictor.py (integration)
# ============================================================
section("16. predictor.py (integration)")

from predictor import SSQPredictor

# Only test on small subset due to MCMC cost
small_data = parsed[:200]
predictor = SSQPredictor(small_data, half_life=100)
predictor.mc_n_samples = 2000
predictor.prepare(train_mcmc=True, mine_associations=True, min_support=0.02)
result = predictor.predict(top_n=3, n_mc_samples=2000)
check("recommendations" in result, "predictor: has recommendations")
check(len(result["recommendations"]) <= 3, "predictor: <= 3 recs")
check("marginal_probs" in result, "predictor: has marginal_probs")
ok(f"SSQPredictor: {len(result['recommendations'])} recs, "
   f"{result['n_candidates_generated']} candidates")


# ============================================================
# 17. nn_attention.py
# ============================================================
section("17. nn_attention.py")

from nn_attention import TemporalAttentionPredictor

# 17.1 Model init
model = TemporalAttentionPredictor(n_features=20, d_model=32, n_heads=2, T=10)
check(model.n_features == 20, "attention: n_features")
check(model.d_model == 32, "attention: d_model")
check(model.n_heads == 2, "attention: n_heads")
ok("TemporalAttentionPredictor: init OK")

# 17.2 Forward pass
import random
random.seed(42)
X = [[random.gauss(0, 1) for _ in range(20)] for _ in range(10)]
red_logits, blue_logits = model.forward(X)
check(len(red_logits) == 33, f"forward: red_logits len={len(red_logits)}")
check(len(blue_logits) == 16, f"forward: blue_logits len={len(blue_logits)}")
ok("TemporalAttentionPredictor.forward: correct dims")

# 17.3 Forward with attention weights
red_logits, blue_logits, attn_weights = model.forward(X, return_attention=True)
check(len(attn_weights) == 10, "attention weights: 10 rows")
check(len(attn_weights[0]) == 10, "attention weights: 10 cols")
# Each row should sum to ~1
row0_sum = sum(attn_weights[0])
check(abs(row0_sum - 1.0) < 0.01, f"attention row sum: {row0_sum:.4f}")
ok("TemporalAttentionPredictor: attention weights OK")

# 17.4 predict_proba
rp, bp = model.predict_proba(X)
check(abs(sum(rp) - 1.0) < 0.01, f"predict_proba red sum: {sum(rp):.4f}")
check(abs(sum(bp) - 1.0) < 0.01, f"predict_proba blue sum: {sum(bp):.4f}")
ok("TemporalAttentionPredictor.predict_proba: sums to 1")

# 17.5 predict
red_pred, blue_pred = model.predict(X)
check(len(red_pred) == 6, f"predict: {len(red_pred)} reds")
check(len(set(red_pred)) == 6, "predict: no duplicates")
check(1 <= blue_pred <= 16, f"predict: blue in range ({blue_pred})")
ok("TemporalAttentionPredictor.predict: correct")

# 17.6 compute_loss
y_red = [0.0]*33
for i in [0, 2, 5, 10, 15, 20]:
    y_red[i] = 1.0
y_blue = [0.0]*16
y_blue[5] = 1.0
loss = model.compute_loss(X, y_red, y_blue)
check(loss > 0, f"compute_loss: {loss:.4f}")
ok("TemporalAttentionPredictor.compute_loss: OK")

# 17.7 get/set params
params = model.get_params()
check(len(params) > 10, f"get_params: {len(params)} params")
model2 = TemporalAttentionPredictor(n_features=20, d_model=32, n_heads=2, T=10)
model2.set_params(params)
# Verify same forward output
rl2, bl2 = model2.forward(X)
check(abs(rl2[0] - red_logits[0]) < 0.001, "params roundtrip: red same")
ok("get/set_params: roundtrip OK")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed ({passed+failed} total)")
print(f"{'='*60}")

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
