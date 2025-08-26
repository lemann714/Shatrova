#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import DATA_ROOT, MODEL_ROOT
import pathlib
import json
import joblib
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import kurtosis
from scipy.fft import rfft, rfftfreq
from itertools import product
from tqdm import tqdm  
from math import prod             

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.cuda.amp import autocast, GradScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import re
import random

import logging
import sys

LOG_FILE = "monitor.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),   # оставляем вывод в консоль
    ],
)
logger = logging.getLogger(__name__) 
amp_scaler = GradScaler()          # ← отдельный объект, НЕ переопределяется
RANDOM_SEED = 42                     # единственное место, где меняем seed

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CuDNN flags – выбираем один из режимов
    if DEVICE.type == "cuda":
        torch.backends.cudnn.deterministic = True   # воспроизводимость
        torch.backends.cudnn.benchmark = False     # отключаем автоподбор
    else:
        # На CPU эти флаги не имеют смысла
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False


# ------------------- параметры измерения -----------------------
FS = 25_600                     # Hz
WINDOW_SEC = 0.20               # 0.2 s → 5120 точек
WINDOW_SIZE = int(FS * WINDOW_SEC)
STEP_SIZE = WINDOW_SIZE // 2    # 50 % overlap
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_global_seed()            

# --------------------------------------------------------------
# Утилиты (чтение, окна, признаки)
# --------------------------------------------------------------
def load_multiple_csv(folder: pathlib.Path) -> pd.DataFrame:
    pattern = re.compile(r'^current_(?:[1-9]|[1-9]\d|100)\.csv$')
    files = sorted(p for p in folder.iterdir() if p.is_file() and pattern.match(p.name))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        required = {"current_R", "current_S", "current_T"}
        if not required.issubset(df.columns):
            raise ValueError(f"{f} misses required columns {required}")
        dfs.append(df[["current_R", "current_S", "current_T"]])
    full = pd.concat(dfs, ignore_index=True)
    logger.info(f"[load_multiple_csv] {len(files)} files → {len(full)} rows")
    return full


def rolling_window(a: np.ndarray, win: int, step: int):
    """Перекрывающиеся окна без копирования памяти."""
    n = (len(a) - win) // step + 1
    shape = (n, win)
    strides = (a.strides[0] * step, a.strides[0])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def make_dataset(df: pd.DataFrame) -> np.ndarray:
    """DataFrame → (n_windows, 3, win)."""
    windows = []
    for col in ["current_R", "current_S", "current_T"]:
        arr = df[col].values.astype(np.float32)
        windows.append(rolling_window(arr, WINDOW_SIZE, STEP_SIZE))
    return np.stack(windows, axis=1)          # (n, 3, win)


def statistical_features(window: np.ndarray) -> list:
    """Mean, std, RMS, max, min, kurtosis – по каждой фазе."""
    feats = []
    for ph in range(3):
        ch = window[ph]
        feats.append(np.mean(ch))
        feats.append(np.std(ch))
        feats.append(np.sqrt(np.mean(ch ** 2)))   # RMS
        feats.append(np.max(ch))
        feats.append(np.min(ch))
        feats.append(kurtosis(ch))
    return feats


def fft_features(window: np.ndarray) -> list:
    """
    12-мерный вектор:
        – средняя мощность спектра
        – мощность 2-й, 3-й и 5-й гармоник (по частоте вращения 1770 об/мин)
    """
    feats = []
    freqs = rfftfreq(WINDOW_SIZE, d=1 / FS)

    # частоты в Гц, соответствующие 2-й, 3-й и 5-й гармоникам вращения:
    #   f_rot = 1770 об/мин = 29.5 Гц  →  k·f_rot, k=2,3,5
    f_rot = 1770 / 60.0
    idx_2 = np.argmin(np.abs(freqs - 2 * f_rot))
    idx_3 = np.argmin(np.abs(freqs - 3 * f_rot))
    idx_5 = np.argmin(np.abs(freqs - 5 * f_rot))

    for ph in range(3):
        spec = np.abs(rfft(window[ph])) ** 2
        mean_pow = spec.mean()
        pow_2 = spec[idx_2]
        pow_3 = spec[idx_3]
        pow_5 = spec[idx_5]
        feats.extend([mean_pow, pow_2, pow_3, pow_5])
    return feats


def extract_features_for_isolation(windows: np.ndarray) -> np.ndarray:
    """(n, 3, win) → (n, 30) : статистика + FFT."""
    n = windows.shape[0]
    feats = []
    for i in range(n):
        win = windows[i]                     # (3, win)
        feats.append(statistical_features(win) + fft_features(win))
    return np.array(feats, dtype=np.float32)    # (n, 30)


# --------------------------------------------------------------
# IsolationForest + Grid-Search
# --------------------------------------------------------------
def _unsupervised_scorer(estimator, X, y=None):
    """
    Возвращает **отрицательное** среднее значение `score_samples`.
    Чем меньше значение, тем хуже модель (поэтому берём отрицательное,
    чтобы GridSearchCV стремилась к максимуму).
    """
    # score_samples → чем больше, тем «более нормальный» объект.
    # Мы хотим минимизировать количество аномалий → берём отрицательное среднее.
    return -estimator.score_samples(X).mean()


# Оборачиваем в sklearn-совместимый объект, чтобы можно было передать в GridSearchCV
# UNSUPERVISED_SCORER = make_scorer(_unsupervised_scorer, greater_is_better=True)


def grid_search_isolation(df: pd.DataFrame,
                         param_grid: dict | None = None,
                         cv: int = 3,
                         scoring=_unsupervised_scorer):
    """
    Обучает IsolationForest с помощью GridSearchCV.
    Параметр `scoring` по-умолчанию – наш unsupervised-скорер,
    поэтому пользователь может вызвать функцию без указания `scoring`.
    """
    if param_grid is None:
        param_grid = {
            "iso__n_estimators": [150, 300],
            "iso__contamination": [0.005, 0.01],
            "iso__max_samples": ["auto", 0.7],
        }

    # ------------------- подготовка признаков -------------------
    windows = make_dataset(df)                               # (n, 3, win)
    X = extract_features_for_isolation(windows)             # (n, 30)

    # Делим только X – y-ов у нас нет.
    X_train, X_val = train_test_split(
        X, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    # ------------------- пайплайн -------------------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iso", IsolationForest(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    # ------------------- GridSearchCV -------------------
    # Обратите внимание: передаём **функцию**, а не объект make_scorer!
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,          # ← наш обычный callable
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )
    gs.fit(X_train)                # y-ов не передаём

    logger.info("\n=== IsolationForest GridSearch ===")
    logger.info(f"Best score (neg‑mean‑score_samples): {gs.best_score_:.6f}")
    logger.info(f"Best params: {gs.best_params_}")

    best_pipe = gs.best_estimator_
    # Переобучаем на всём наборе (train+val)
    best_pipe.fit(X)

    return best_pipe, gs.best_params_


def predict_isolation_forest(test_path: pathlib.Path,
                             model_dir: pathlib.Path) -> pd.DataFrame:
    """Возвращает DataFrame с колонкой iso_score ∈[0,1] (чем больше – аномалия)."""
    scaler = joblib.load(model_dir / "scaler_iso.pkl")
    iso = joblib.load(model_dir / "iso_forest.pkl")

    df = pd.read_csv(test_path)
    windows = make_dataset(df)
    X = extract_features_for_isolation(windows)
    X_scaled = scaler.transform(X)

    raw = iso.decision_function(X_scaled)                # <0 – аномалия
    iso_score = MinMaxScaler().fit_transform(raw.reshape(-1, 1)).ravel()

    timestamps = np.arange(0, len(raw) * STEP_SIZE) / FS
    return pd.DataFrame({
        "timestamp_s": timestamps,
        "iso_score": iso_score,
        "iso_raw": raw,
    })

class CurrentsDataset(Dataset):
    """(n, 3, win) → (n, win, 3) без копирования."""
    def __init__(self, windows: np.ndarray):
        # windows: (n, 3, win) → (n, win, 3)
        arr = np.transpose(windows, (0, 2, 1))
        self.x = torch.from_numpy(arr).float()   # без копии

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]
# --------------------------------------------------------------
# LSTM-AutoEncoder
# --------------------------------------------------------------
class LazyWindowsDataset(Dataset):
    def __init__(self, arr: np.ndarray, scaler: StandardScaler):
        self.arr = arr.astype(np.float32)          # (N, 3)
        self.scaler = scaler
        self.n_samples = (len(self.arr) - WINDOW_SIZE) // STEP_SIZE + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * STEP_SIZE
        win = self.arr[start:start + WINDOW_SIZE]          # (win, 3)
        win = self.scaler.transform(win)                   # (win, 3)
        return torch.from_numpy(win)                         # (win, 3)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int = 3, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=n_features,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder(x)
        # используем последний слой скрытого состояния как вектор-латент
        latent = h_n[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(latent)
        return out

def grid_search_lstm_ae(df: pd.DataFrame,
                        param_grid: dict | None = None,
                        patience: int = 2,
                        num_workers: int = 0) -> tuple:
    """
    Перебор гиперпараметров LSTM‑AE.
    Возвращает (final_model, feat_scaler, best_cfg, best_score)
    """
    if param_grid is None:
        param_grid = {
            "latent_dim": [64, 128],
            "batch_size": [64, 128],
            "lr": [5e-4, 1e-4],
            "epochs": [5, 10],
        }

    # ---------- 1️⃣  скейлер признаков ----------
    raw_arr = df[["current_R", "current_S", "current_T"]].values.astype(np.float32)
    sample_n = min(100_000, len(raw_arr))
    sample_idx = np.random.choice(len(raw_arr), size=sample_n, replace=False)
    feat_scaler = StandardScaler()
    feat_scaler.fit(raw_arr[sample_idx])

    # ---------- 2️⃣  датасет ----------
    full_dataset = LazyWindowsDataset(raw_arr, feat_scaler)

    # ---------- 3️⃣  разбиение ----------
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    split = int(0.85 * len(full_dataset))
    train_idx, val_idx = indices[:split], indices[split:]

    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)

    # ---------- 4️⃣  перебор конфигураций ----------
    keys, values = zip(*param_grid.items())
    total_cfg = prod([len(v) for v in values])   # без лишних списков
    logger.info(f"[grid_search_lstm_ae] {total_cfg} конфигураций")

    best_score = np.inf
    best_cfg   = None
    best_state = None

    for cfg in tqdm(product(*values), desc="LSTM‑AE grid"):
        cfg_dict = dict(zip(keys, cfg))

        # DataLoader'ы – единственное, что зависит от batch_size
        train_loader = DataLoader(
            train_set,
            batch_size=cfg_dict["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=DEVICE.type == "cuda",
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg_dict["batch_size"],
            shuffle=False,
            pin_memory=DEVICE.type == "cuda",
            num_workers=num_workers,
        )

        # модель
        model = LSTMAutoEncoder(
            seq_len=WINDOW_SIZE,
            n_features=3,
            latent_dim=cfg_dict["latent_dim"]
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg_dict["lr"])
        loss_fn   = nn.MSELoss()

        # ---------- обучение ----------
        for epoch in range(1, cfg_dict["epochs"] + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                with autocast():
                    recon = model(batch)
                    loss = loss_fn(recon, batch)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()

            # ---------- валидация ----------
            model.eval()
            val_err = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE, non_blocking=True)
                    recon = model(batch)
                    mse = ((recon - batch) ** 2).mean(dim=(1, 2))
                    val_err.append(mse.cpu().numpy())
            val_score = np.concatenate(val_err).mean()

            # ---------- early‑stop ----------
            if val_score < best_score:
                best_score = val_score
                best_cfg   = cfg_dict
                best_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # очистка GPU‑памяти
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # 5️⃣  Финальное обучение на всём наборе (используем best_state)
    # --------------------------------------------------------------
    logger.info(f"\n=== Лучший cfg: {best_cfg} → val MSE {best_score:.6f}")

    full_loader = DataLoader(
        full_dataset,
        batch_size=best_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=DEVICE.type == "cuda",
        num_workers=num_workers,
        persistent_workers=False,
    )

    final_model = LSTMAutoEncoder(
        seq_len=WINDOW_SIZE,
        n_features=3,
        latent_dim=best_cfg["latent_dim"]
    ).to(DEVICE)

    # Если хотите продолжить обучение от лучшего состояния:
    if best_state is not None:
        final_model.load_state_dict(best_state)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_cfg["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(1, best_cfg["epochs"] + 1):
        final_model.train()
        epoch_loss = 0.0
        for batch in full_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            recon = final_model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        logger.info(
            f"[final LSTM‑AE] epoch {epoch:02d} | loss {epoch_loss/len(full_loader.dataset):.6f}"
        )

    return final_model, feat_scaler, best_cfg, best_score

def predict_lstm_ae(test_path: pathlib.Path,
                    model_dir: pathlib.Path) -> pd.DataFrame:
    """Возвращает DataFrame с колонкой lstm_error ∈[0,1] (чем больше – аномалия)."""
    scaler = joblib.load(model_dir / "scaler_lstm.pkl")
    model = LSTMAutoEncoder(seq_len=WINDOW_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(model_dir / "lstm_ae.pt",
                                    map_location=DEVICE))
    model.eval()

    df = pd.read_csv(test_path)
    windows = make_dataset(df)

    # нормализация тем же скейлером
    flat = np.transpose(windows, (0, 2, 1)).reshape(-1, 3)
    flat = scaler.transform(flat)
    windows_norm = flat.reshape(windows.shape[0],
                                WINDOW_SIZE, 3).transpose(0, 2, 1)

    ds = CurrentsDataset(windows_norm)
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    errors = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(DEVICE)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2))
            errors.append(mse.cpu().numpy())
    errors = np.concatenate(errors)

    lstm_error = MinMaxScaler().fit_transform(errors.reshape(-1, 1)).ravel()
    timestamps = np.arange(0, len(errors) * STEP_SIZE) / FS
    return pd.DataFrame({
        "timestamp_s": timestamps,
        "lstm_error": lstm_error,
        "lstm_raw": errors,
    })


# --------------------------------------------------------------
# Объединение предсказаний (offline)
# --------------------------------------------------------------
def ensemble_predictions(iso_df: pd.DataFrame,
                         lstm_df: pd.DataFrame,
                         alarm_threshold: float = 0.4) -> pd.DataFrame:
    """Считаем ensemble_score = iso_score * lstm_error и бинарный alarm."""
    assert len(iso_df) == len(lstm_df), "Размеры не совпадают!"
    ensemble = iso_df["iso_score"].values * lstm_df["lstm_error"].values
    alarm = ensemble > alarm_threshold
    return pd.DataFrame({
        "timestamp_s": iso_df["timestamp_s"],
        "iso_score": iso_df["iso_score"],
        "lstm_error": lstm_df["lstm_error"],
        "ensemble_score": ensemble,
        "alarm": alarm.astype(int),
    })


def compute_threshold_from_train(iso_df: pd.DataFrame,
                                 lstm_df: pd.DataFrame,
                                 q: float = 0.99) -> float:
    """Возвращает q-й перцентиль ensemble-score на тренировочном наборе."""
    ensemble = iso_df["iso_score"].values * lstm_df["lstm_error"].values
    return np.quantile(ensemble, q)


# --------------------------------------------------------------
# Онлайн-инференс (генератор)
# --------------------------------------------------------------
def real_time_ensemble_predictor(model_dir: pathlib.Path,
                                 alarm_threshold: float = 0.4):
    """
    Генератор, который принимает новые измерения (R, S, T) через .send()
    и каждый STEP_SIZE-й отсчёт выдаёт кортеж:
        (timestamp, iso_score, lstm_error, ensemble_score, alarm)
    """
    # ----- загрузка моделей -----
    scaler_iso = joblib.load(model_dir / "scaler_iso.pkl")
    iso = joblib.load(model_dir / "iso_forest.pkl")

    scaler_lstm = joblib.load(model_dir / "scaler_lstm.pkl")
    lstm = LSTMAutoEncoder(seq_len=WINDOW_SIZE).to(DEVICE)
    lstm.load_state_dict(torch.load(model_dir / "lstm_ae.pt",
                                   map_location=DEVICE))
    lstm.eval()

    # ----- буферы -----
    buf_R = deque(maxlen=WINDOW_SIZE)
    buf_S = deque(maxlen=WINDOW_SIZE)
    buf_T = deque(maxlen=WINDOW_SIZE)

    counter = 0  # количество полученных отсчётов

    def _inner():
        nonlocal counter
        while True:
            sample = (yield)          # получаем (R, S, T) через .send()
            r, s, t = sample
            buf_R.append(r)
            buf_S.append(s)
            buf_T.append(t)

            if len(buf_R) < WINDOW_SIZE:
                continue

            counter += 1
            # выдаём только раз в STEP_SIZE отсчётов
            if counter % STEP_SIZE != 0:
                continue

            # ---------------- Isolation ----------------
            win = np.stack([np.array(buf_R),
                            np.array(buf_S),
                            np.array(buf_T)], axis=0)   # (3, win)

            feats = np.array(statistical_features(win) + fft_features(win)).reshape(1, -1)
            feats_scaled = scaler_iso.transform(feats)
            iso_raw = iso.decision_function(feats_scaled)[0]
            iso_score = MinMaxScaler().fit_transform(np.array([iso_raw]).reshape(-1, 1)).ravel()[0]

            # ---------------- LSTM ----------------
            flat = win.T.reshape(-1, 3)
            flat = scaler_lstm.transform(flat)
            win_norm = flat.reshape(1, WINDOW_SIZE, 3)
            torch_win = torch.tensor(win_norm, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                recon = lstm(torch_win)
                lstm_raw = ((recon - torch_win) ** 2).mean().item()
            lstm_error = MinMaxScaler().fit_transform(np.array([lstm_raw]).reshape(-1, 1)).ravel()[0]

            # ---------------- Ensemble ----------------
            ensemble_score = iso_score * lstm_error
            alarm = ensemble_score > alarm_threshold

            # timestamp – начало окна
            timestamp = (counter - STEP_SIZE) / FS
            yield (timestamp, iso_score, lstm_error, ensemble_score, int(alarm))

    gen = _inner()
    next(gen)                     # «прокачиваем» до первого yield
    return gen


# --------------------------------------------------------------
# 6️⃣  Обучение на всей папке и сохранение моделей
# --------------------------------------------------------------
def train_on_folder(data_folder: pathlib.Path,
                    model_folder: pathlib.Path,
                    iso_grid: dict = None,
                    lstm_grid: dict = None) -> dict:
    """
    1) Читает все csv из data_folder.
    2) Делает grid-search для IsolationForest и LSTM-AE.
    3) Сохраняет лучшие модели и их параметры.
    4) Возвращает словарь с найденными гиперпараметрами.
    """
    df = load_multiple_csv(data_folder)

    # ---------- IsolationForest ----------
    iso_pipe, iso_best = grid_search_isolation(df, param_grid=iso_grid)
    model_folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso_pipe.named_steps["scaler"], model_folder / "scaler_iso.pkl")
    joblib.dump(iso_pipe.named_steps["iso"], model_folder / "iso_forest.pkl")
    with open(model_folder / "iso_params.json", "w") as f:
        json.dump(iso_best, f, indent=2)

    # ---------- LSTM-AE ----------
    lstm_model, lstm_scaler, lstm_best, lstm_best_val = \
        grid_search_lstm_ae(df, param_grid=lstm_grid, num_workers=4)

    torch.save(lstm_model.state_dict(), model_folder / "lstm_ae.pt")
    joblib.dump(lstm_scaler, model_folder / "scaler_lstm.pkl")
    with open(model_folder / "lstm_params.json", "w") as f:
        json.dump(lstm_best, f, indent=2)

    logger.info("\n=== Обучение завершено ===")
    logger.info("IsolationForest params : %s", iso_best)
    logger.info("LSTM‑AE params        : %s", lstm_best)
    logger.info(f"LSTM‑AE validation MSE: {lstm_best_val:.6f}")

    return {
        "iso_params": iso_best,
        "lstm_params": lstm_best,
        "lstm_val_mse": lstm_best_val,
    }


if __name__ == "__main__":
    # Папка, где лежат все тренировочные csv-файлы

    iso_grid = {
        "iso__n_estimators": [200, 400, 600],
        "iso__contamination": [0.001, 0.003, 0.005, 0.01],
        "iso__max_samples": ["auto", 0.5, 0.7],
    }

    lstm_grid = {
        "latent_dim": [64, 128, 256],
        "batch_size": [64, 128, 256],
        "lr": [1e-3, 5e-4, 1e-4],
        "epochs": [10, 20, 30],
    }

    # ------------------- обучение -------------------
    best_cfg = train_on_folder(DATA_ROOT, 
                               MODEL_ROOT,
                               iso_grid=iso_grid,
                               lstm_grid=lstm_grid)
