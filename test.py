import pandas as pd
from train import real_time_ensemble_predictor, predict_isolation_forest, predict_lstm_ae, compute_threshold_from_train, ensemble_predictions
from constants import DATA_ROOT, MODEL_ROOT, TEST_PATH

if __name__ == "__main__":

    # ------------------- offline-оценка -------------------
    
    if TEST_PATH.exists():
        iso_res  = predict_isolation_forest(TEST_PATH, MODEL_ROOT)
        lstm_res = predict_lstm_ae(TEST_PATH, MODEL_ROOT)

        # Порог берём как 99-й перцентиль ensemble-score на тренировочном наборе
        thr = compute_threshold_from_train(
            predict_isolation_forest(DATA_ROOT / "train.csv", MODEL_ROOT),
            predict_lstm_ae(DATA_ROOT / "train.csv", MODEL_ROOT),
            q=0.99,
        )
        ensemble_df = ensemble_predictions(iso_res, lstm_res,
                                           alarm_threshold=thr)
        ensemble_df.to_csv("ensemble_offline_predictions.csv", index=False)
        print("\nOffline-предсказания сохранены в ensemble_offline_predictions.csv")

    # ------------------- онлайн-демо -------------------
    print("\n=== Онлайн‑демо (чтение test.csv построчно) ===")
    gen = real_time_ensemble_predictor(MODEL_ROOT, alarm_threshold=thr)
    for _, row in pd.read_csv(TEST_PATH, chunksize=1).iterrows():
        sample = (row["current_R"], row["current_S"], row["current_T"])
        out = gen.send(sample)           # может вернуть None, пока не сформировано окно
        if out is not None:
            ts, iso_sc, lstm_sc, ens_sc, al = out
            print(f"[{ts:7.3f}s] iso={iso_sc:.3f} lstm={lstm_sc:.3f} "
                  f"ens={ens_sc:.3f} alarm={al}")

    print("\nКонец файла")