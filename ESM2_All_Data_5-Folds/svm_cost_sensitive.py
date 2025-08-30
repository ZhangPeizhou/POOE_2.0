#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础版（仅类权重）：python svm_cost_sensitive.py

小网格搜索：python svm_cost_sensitive.py --grid
"""

"""
Cost-Sensitive SVM（与“微调阈值”脚本同落盘风格，支持小网格）
- 模型保存到 MODEL_DIR
- 其他产物保存到 ARTIFACT_DIR/本次运行子目录
- 运行前缀包含脚本名
- 仅在原 SVM 上增加 class_weight（默认 "balanced"）
- 可选：--grid 进行小网格搜索（C/gamma，目标 AUPRC）

路径/命名优先级：
1) 从 config_paths.py 导入 MODEL_DIR / ARTIFACT_DIR / RUN_PREFIX_NAME（若存在）
2) 环境变量 FP_MODEL_DIR / FP_ARTIFACT_DIR
3) 自动探测：models|checkpoints；outputs|results|metrics
4) 默认：models/ 与 outputs/

输入数据（若 .pkl 不存在将自动尝试 .npy）：
  data/train_X.pkl, data/train_y.pkl
  data/ext_X.pkl,   data/ext_y.pkl
"""

import os
import re
import json
import time
import pickle
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
)

# ==================== 兼容你现有工程的路径/命名 ====================

RUN_PREFIX_NAME: Optional[str] = None
MODEL_DIR: Optional[str] = None
ARTIFACT_DIR: Optional[str] = None
try:
    from config_paths import RUN_PREFIX_NAME as _RPN  # type: ignore
    RUN_PREFIX_NAME = _RPN
except Exception:
    pass
try:
    from config_paths import MODEL_DIR as _MD  # type: ignore
    MODEL_DIR = _MD
except Exception:
    pass
try:
    from config_paths import ARTIFACT_DIR as _AD  # type: ignore
    ARTIFACT_DIR = _AD
except Exception:
    pass

MODEL_DIR = os.environ.get("FP_MODEL_DIR", MODEL_DIR) if MODEL_DIR is not None else os.environ.get("FP_MODEL_DIR", None)
ARTIFACT_DIR = os.environ.get("FP_ARTIFACT_DIR", ARTIFACT_DIR) if ARTIFACT_DIR is not None else os.environ.get("FP_ARTIFACT_DIR", None)

def _detect_dir(candidates, default_name):
    for d in candidates:
        if Path(d).exists():
            return d
    return default_name

if MODEL_DIR is None:
    MODEL_DIR = _detect_dir(["models", "checkpoints"], "models")
if ARTIFACT_DIR is None:
    ARTIFACT_DIR = _detect_dir(["outputs", "results", "metrics"], "outputs")

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

# ==================== 数据路径 ====================
TRAIN_X_PATH = "data/train_X.pkl"
TRAIN_Y_PATH = "data/train_y.pkl"
EXT_X_PATH   = "data/ext_X.pkl"
EXT_Y_PATH   = "data/ext_y.pkl"

# ==================== 模型与实验缺省参数 ====================
DEFAULT_KERNEL = "rbf"
DEFAULT_C = 10.0
DEFAULT_GAMMA = 0.25
DEFAULT_CLASS_WEIGHT = "balanced"    # 关键改动
DEFAULT_N_SPLITS = 5
DEFAULT_POS_LABEL = 1
try:
    from config_labels import POS_LABEL as _PL  # type: ignore
    DEFAULT_POS_LABEL = int(_PL)
except Exception:
    pass

# 小网格（可根据需要在这里改，保证可复现）
GRID_C_VALUES = [3.0, 10.0, 30.0]
GRID_GAMMA_VALUES = [0.1, 0.25, 0.5]
GRID_CV_FOLDS = 3
GRID_SCORING = "average_precision"

THRESH_GRID = np.linspace(0.0, 1.0, 1001)

# ==================== 工具函数 ====================
def _safe_load(path: str):
    p = Path(path)
    if p.exists():
        if p.suffix == ".pkl":
            with open(p, "rb") as f:
                return pickle.load(f)
        elif p.suffix == ".npy":
            return np.load(p, allow_pickle=True)
    alt = p.with_suffix(".npy") if p.suffix == ".pkl" else p.with_suffix(".pkl")
    if alt.exists():
        return _safe_load(str(alt))
    raise FileNotFoundError(f"找不到数据：{path} 或 {alt}")

def ensure_2d(x):
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x

def prf_mcc(y_true, y_prob, thr, pos_label=1):
    y_pred = (y_prob >= thr).astype(int)
    if pos_label != 1:
        y_true = (y_true == pos_label).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return p, r, f1, mcc

def sweep_thresholds(y_true, y_prob, pos_label=1):
    rows = []
    best_f1 = (-1.0, None)
    best_mcc = (-2.0, None)
    for t in THRESH_GRID:
        p, r, f1, mcc = prf_mcc(y_true, y_prob, t, pos_label)
        rows.append([t, p, r, f1, mcc])
        if f1 > best_f1[0]:
            best_f1 = (f1, t)
        if mcc > best_mcc[0]:
            best_mcc = (mcc, t)
    df = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "mcc"])
    return df, best_f1, best_mcc

def cv_eval(X, y, model_kwargs, n_splits=DEFAULT_N_SPLITS, pos_label=DEFAULT_POS_LABEL, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for i, (tr, va) in enumerate(skf.split(X, y), 1):
        clf = svm.SVC(**model_kwargs)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[va])[:, 1]
        try:
            roc_auc = roc_auc_score(y[va], prob)
        except Exception:
            roc_auc = np.nan
        try:
            auprc = average_precision_score(y[va], prob, pos_label=pos_label)
        except Exception:
            auprc = np.nan
        p, r, f1, mcc = prf_mcc(y[va], prob, 0.5, pos_label)
        rows.append({
            "fold": i, "roc_auc": roc_auc, "auprc": auprc,
            "precision@0.5": p, "recall@0.5": r, "f1@0.5": f1, "mcc@0.5": mcc
        })
    df = pd.DataFrame(rows)
    return df, df.mean(numeric_only=True).to_dict()

def grid_search(X, y, base_kwargs):
    params = {
        "C": GRID_C_VALUES,
        "gamma": GRID_GAMMA_VALUES,
    }
    est = svm.SVC(
        probability=True,
        kernel=base_kwargs.get("kernel", "rbf"),
        class_weight=base_kwargs.get("class_weight", "balanced"),
    )
    gs = GridSearchCV(
        est, params, scoring=GRID_SCORING, cv=GRID_CV_FOLDS, n_jobs=-1, verbose=1
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def make_run_prefix(script_name: str) -> str:
    stem = Path(script_name).stem
    stem = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = RUN_PREFIX_NAME if RUN_PREFIX_NAME else stem
    return f"{base}_{ts}"

# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser(description="Cost-Sensitive SVM（小网格可选）")
    ap.add_argument("--grid", action="store_true", help="开启小网格搜索（C/gamma，目标 AUPRC）")
    ap.add_argument("--pos_label", type=int, default=DEFAULT_POS_LABEL, help="正类标签")
    args = ap.parse_args()

    X_tr = ensure_2d(_safe_load(TRAIN_X_PATH))
    y_tr = np.asarray(_safe_load(TRAIN_Y_PATH)).astype(int)
    X_ext = ensure_2d(_safe_load(EXT_X_PATH))
    y_ext = np.asarray(_safe_load(EXT_Y_PATH)).astype(int)

    model_kwargs = dict(
        kernel=DEFAULT_KERNEL,
        C=DEFAULT_C,
        gamma=DEFAULT_GAMMA,
        probability=True,
        class_weight=DEFAULT_CLASS_WEIGHT,
    )

    # 交叉验证（报告基础版表现）
    cv_df, cv_summary = cv_eval(X_tr, y_tr, model_kwargs, n_splits=DEFAULT_N_SPLITS, pos_label=args.pos_label)

    # 可选：小网格搜索
    best_params = None
    best_cv_score = None
    if args.grid:
        best_model, best_params, best_cv_score = grid_search(X_tr, y_tr, model_kwargs)
        if "C" in best_params: model_kwargs["C"] = best_params["C"]
        if "gamma" in best_params: model_kwargs["gamma"] = best_params["gamma"]

    # 全量训练 + external
    clf = svm.SVC(**model_kwargs)
    clf.fit(X_tr, y_tr)
    ext_prob = clf.predict_proba(X_ext)[:, 1]

    # 概率级指标
    try:
        ext_roc = roc_auc_score(y_ext, ext_prob)
    except Exception:
        ext_roc = np.nan
    try:
        ext_pr = average_precision_score(y_ext, ext_prob, pos_label=args.pos_label)
    except Exception:
        ext_pr = np.nan

    p05, r05, f105, mcc05 = prf_mcc(y_ext, ext_prob, 0.5, args.pos_label)
    sweep_df, best_f1, best_mcc = sweep_thresholds(y_ext, ext_prob, args.pos_label)
    f1_val, f1_thr = best_f1
    mcc_val, mcc_thr = best_mcc

    p_f1, r_f1, f1_f1, mcc_f1 = prf_mcc(y_ext, ext_prob, f1_thr, args.pos_label)
    p_m, r_m, f1_m, mcc_m = prf_mcc(y_ext, ext_prob, mcc_thr, args.pos_label)

    # ===== 落盘（严格区分模型 vs 其他产物）=====
    run_prefix = make_run_prefix(__file__)
    run_dir = Path(ARTIFACT_DIR) / run_prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) 模型 → MODEL_DIR
    model_name = f"{run_prefix}_model.joblib"
    dump(clf, str(Path(MODEL_DIR) / model_name))

    # 2) 其他 → ARTIFACT_DIR/run_prefix
    np.save(str(run_dir / "ext_prob.npy"), ext_prob)
    cv_df.to_csv(run_dir / "cv_folds.csv", index=False)
    sweep_df.to_csv(run_dir / "threshold_sweep.csv", index=False)

    summary = {
        "run_prefix": run_prefix,
        "paths": {
            "MODEL_DIR": str(Path(MODEL_DIR).resolve()),
            "ARTIFACT_DIR": str(Path(ARTIFACT_DIR).resolve()),
            "model_file": str((Path(MODEL_DIR) / model_name).resolve()),
            "run_dir": str(run_dir.resolve()),
        },
        "model_params_used": model_kwargs,
        "pos_label": args.pos_label,
        "cv_summary_mean": cv_summary,
        "grid_search": {
            "enabled": best_params is not None,
            "best_params": best_params,
            "best_cv_score(auprc)": best_cv_score,
            "grid": {
                "C": GRID_C_VALUES,
                "gamma": GRID_GAMMA_VALUES,
                "cv_folds": GRID_CV_FOLDS,
                "scoring": GRID_SCORING
            }
        },
        "external_test": {
            "roc_auc": float(ext_roc) if ext_roc == ext_roc else None,
            "auprc": float(ext_pr) if ext_pr == ext_pr else None,
            "metrics@0.5": {"precision": p05, "recall": r05, "f1": f105, "mcc": mcc05},
            "best_f1": {"threshold": f1_thr, "precision": p_f1, "recall": r_f1, "f1": f1_f1, "mcc": mcc_f1},
            "best_mcc": {"threshold": mcc_thr, "precision": p_m, "recall": r_m, "f1": f1_m, "mcc": mcc_m},
        },
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== Cost-Sensitive SVM 完成（小网格={}）====".format("ON" if args.grid else "OFF"))
    print(f"模型已保存: {Path(MODEL_DIR) / model_name}")
    print(f"产物目录:   {run_dir}")
    print(f"[CV-mean] AUPRC={cv_summary.get('auprc', np.nan):.4f}, "
          f"ROC-AUC={cv_summary.get('roc_auc', np.nan):.4f}, "
          f"F1@0.5={cv_summary.get('f1@0.5', np.nan):.4f}, "
          f"MCC@0.5={cv_summary.get('mcc@0.5', np.nan):.4f}")
    print(f"[EXT] AUPRC={ext_pr:.4f}, ROC-AUC={ext_roc:.4f}")
    print(f"[EXT @0.5] P={p05:.4f}, R={r05:.4f}, F1={f105:.4f}, MCC={mcc05:.4f}")
    print(f"[EXT best F1] thr={f1_thr:.3f}, F1={f1_f1:.4f}, P={p_f1:.4f}, R={r_f1:.4f}, MCC={mcc_f1:.4f}")
    print(f"[EXT best MCC] thr={mcc_thr:.3f}, MCC={mcc_m:.4f}, P={p_m:.4f}, R={r_m:.4f}, F1={f1_m:.4f}\n")


if __name__ == "__main__":
    main()
