"""Adaptive Machine Learning Pipeline for Binary Direction Forecasting.

Employs LightGBM (with robust fallback) for binary classification,
walk-forward cross-validation, Expected Value (EV) estimation, and
rolling periodic retraining with checkpoint retention management.
"""

from datetime import datetime, timezone
import glob
import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV

# Try LightGBM, fallback to XGBoost or RandomForest
HAS_LGBM = False
HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    pass

if not HAS_LGBM:
    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        pass

from config import settings
from feature_engine import build_features, get_feature_columns


class AdaptiveModel:
    """Manages ML training, walk-forward validation, EV computation, and persistence."""

    def __init__(self, asset: str, model_dir: Optional[str] = None):
        self.asset = asset
        self.model_dir = model_dir or settings.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.feature_columns = get_feature_columns()
        self.latest_metrics: Dict[str, float] = {}
        self.model_path: Optional[str] = None

    def calculate_expected_value(self, proba_up: float, payout: float) -> Tuple[str, float, float]:
        """Calculate the Expected Value (EV) for both CALL and PUT directions.

        Formula:
            EV_call = p_up * payout - (1 - p_up) * 1.0
            EV_put  = (1 - p_up) * payout - p_up * 1.0

        Returns:
            Tuple of (best_direction, best_confidence, best_ev)
        """
        proba_down = 1.0 - proba_up
        ev_call = (proba_up * payout) - (proba_down * 1.0)
        ev_put = (proba_down * payout) - (proba_up * 1.0)

        if ev_call >= ev_put:
            return "CALL", proba_up, ev_call
        else:
            return "PUT", proba_down, ev_put

    def _init_estimator(self):
        """Instantiate the primary, ensemble, and/or calibrated classifier."""
        base_models = []
        if HAS_LGBM:
            lgb_clf = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=100,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=24,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=-1
            )
            base_models.append(("lgb", lgb_clf))
        elif HAS_XGB:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=5,
                eval_metric="logloss",
                random_state=42
            )
            base_models.append(("xgb", xgb_clf))

        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        base_models.append(("rf", rf_clf))

        et_clf = ExtraTreesClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        base_models.append(("et", et_clf))

        if getattr(settings, "USE_ENSEMBLE", True):
            clf = VotingClassifier(estimators=base_models, voting="soft")
        else:
            clf = base_models[0][1]

        if getattr(settings, "CALIBRATE_PROBABILITIES", False):
            # Wrap in isotonic probability calibration
            clf = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=3)

        return clf

    def walk_forward_validate(
        self, df: pd.DataFrame, n_splits: int = 5, default_payout: float = 0.85
    ) -> Dict[str, float]:
        """Perform rolling walk-forward validation across expanding historical windows.

        Measures precision, recall, accuracy, Brier score, and average Expected Value (EV).
        """
        features_df = build_features(df)
        if features_df.empty or len(features_df) < 150:
            logger.warning("Insufficient samples for walk-forward validation.")
            return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "mean_ev": 0.0, "brier_score": 0.25}

        X = features_df[self.feature_columns].values
        y = features_df['target'].values

        fold_size = len(X) // (n_splits + 1)
        accuracies = []
        precisions = []
        recalls = []
        brier_scores = []
        evs = []

        for i in range(1, n_splits + 1):
            train_idx = i * fold_size
            val_idx = min((i + 1) * fold_size, len(X))

            X_train, y_train = X[:train_idx], y[:train_idx]
            X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]

            if len(X_val) < 10 or len(np.unique(y_train)) < 2:
                continue

            clf = self._init_estimator()
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)[:, 1]

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            brier = brier_score_loss(y_val, y_proba)

            # Calculate EV on validation predictions
            fold_evs = []
            for p, actual in zip(y_proba, y_val):
                _, conf, ev = self.calculate_expected_value(p, default_payout)
                fold_evs.append(ev)

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            brier_scores.append(brier)
            evs.append(np.mean(fold_evs))
            logger.info(
                f"Fold {i}/{n_splits} -> Test Samples: {len(X_val)} | "
                f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Brier: {brier:.4f} | EV: {np.mean(fold_evs):+.4f}"
            )

        metrics = {
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.5,
            "precision": float(np.mean(precisions)) if precisions else 0.5,
            "recall": float(np.mean(recalls)) if recalls else 0.5,
            "brier_score": float(np.mean(brier_scores)) if brier_scores else 0.25,
            "mean_ev": float(np.mean(evs)) if evs else 0.0,
            "per_fold": [
                {
                    "fold": i + 1,
                    "acc": round(accuracies[i], 4),
                    "prec": round(precisions[i], 4),
                    "rec": round(recalls[i], 4),
                    "brier": round(brier_scores[i], 4),
                    "ev": round(evs[i], 4),
                }
                for i in range(len(accuracies))
            ]
        }
        logger.info(
            f"Walk-forward validation [{self.asset}] -> Acc: {metrics['accuracy']:.3f}, "
            f"Prec: {metrics['precision']:.3f}, Rec: {metrics['recall']:.3f}, "
            f"Brier: {metrics['brier_score']:.4f}, EV: {metrics['mean_ev']:.4f}"
        )
        return metrics

    def train(self, df: pd.DataFrame, default_payout: float = 0.85) -> str:
        """Train model on historical candle dataframe, run validation, and save checkpoint."""
        logger.info(f"Training adaptive model for {self.asset}...")
        features_df = build_features(df)
        if features_df.empty or len(features_df) < 50:
            raise ValueError(f"Insufficient data to train model for {self.asset}.")

        # Run walk-forward validation
        self.latest_metrics = self.walk_forward_validate(df, default_payout=default_payout)

        # Train final model on all available history
        X = features_df[self.feature_columns].values
        y = features_df['target'].values

        self.model = self._init_estimator()
        self.model.fit(X, y)

        # Feature importances
        importances = {}
        underlying = self.model
        if hasattr(underlying, "calibrated_classifiers_") and len(underlying.calibrated_classifiers_) > 0:
            underlying = getattr(underlying.calibrated_classifiers_[0], "estimator", getattr(underlying.calibrated_classifiers_[0], "base_estimator", None))

        if hasattr(underlying, "estimators_"):
            tree_imps = []
            for sub_est in underlying.estimators_:
                if hasattr(sub_est, "feature_importances_"):
                    raw = sub_est.feature_importances_
                    s = sum(raw)
                    if s > 0:
                        tree_imps.append(raw / s)
            if tree_imps:
                avg_imp = np.mean(tree_imps, axis=0)
                importances = {col: float(val) for col, val in zip(self.feature_columns, avg_imp)}
        elif hasattr(underlying, "feature_importances_"):
            raw_imp = underlying.feature_importances_
            total = sum(raw_imp) if sum(raw_imp) > 0 else 1.0
            importances = {col: float(val / total) for col, val in zip(self.feature_columns, raw_imp)}

        if not importances:
            try:
                from sklearn.inspection import permutation_importance
                sub_X = X[-min(len(X), 1000):]
                sub_y = y[-min(len(y), 1000):]
                perm = permutation_importance(self.model, sub_X, sub_y, n_repeats=2, random_state=42)
                raw_imp = [max(0.0, float(v)) for v in perm.importances_mean]
                total = sum(raw_imp) if sum(raw_imp) > 0 else 1.0
                importances = {col: float(val / total) for col, val in zip(self.feature_columns, raw_imp)}
            except Exception as e:
                logger.debug(f"Could not compute permutation importance: {e}")

        if importances:
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            print("\n" + "=" * 50)
            print("         TOP 10 FEATURE IMPORTANCES")
            print("=" * 50)
            for col, imp in sorted_imp[:10]:
                flag = " ⚠️ [SUSPICIOUS >40%]" if imp > 0.40 else ""
                print(f"  {col:22s}: {imp * 100:5.2f}%{flag}")
            print("=" * 50 + "\n")
            self.latest_metrics["feature_importances"] = dict(sorted_imp[:10])

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{self.asset}_{timestamp_str}.joblib"
        self.model_path = os.path.join(self.model_dir, model_filename)

        # Save model and metadata
        payload = {
            "asset": self.asset,
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metrics": self.latest_metrics,
            "timestamp": timestamp_str
        }
        joblib.dump(payload, self.model_path)
        logger.info(f"Saved model checkpoint to {self.model_path}")

        # Purge stale checkpoints
        self.purge_old_models()
        return self.model_path

    def load_latest(self) -> bool:
        """Load the most recent model checkpoint for this asset from disk."""
        pattern = os.path.join(self.model_dir, f"model_{self.asset}_*.joblib")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if not files:
            logger.debug(f"No existing model checkpoints found for asset {self.asset}.")
            return False

        latest_file = files[0]
        try:
            payload = joblib.load(latest_file)
            self.model = payload["model"]
            self.feature_columns = payload.get("feature_columns", self.feature_columns)
            self.latest_metrics = payload.get("metrics", {})
            self.model_path = latest_file
            logger.info(f"Loaded existing model for {self.asset} from {latest_file}")
            return True
        except Exception as e:
            logger.error(f"Failed loading model from {latest_file}: {e}")
            return False

    def predict_proba(self, candle_features: pd.DataFrame) -> float:
        """Predict the probability of next candle being bullish (target=1).

        Returns:
            float: probability between 0.0 and 1.0 (defaults to 0.5 on error).
        """
        if self.model is None:
            if not self.load_latest():
                return 0.5

        try:
            X = candle_features[self.feature_columns].iloc[[-1]].values
            proba = self.model.predict_proba(X)[0][1]
            return float(proba)
        except Exception as e:
            logger.error(f"Prediction failed for {self.asset}: {e}")
            return 0.5

    def purge_old_models(self):
        """Keep only the latest MODEL_PURGE_KEEP checkpoints on disk."""
        pattern = os.path.join(self.model_dir, f"model_{self.asset}_*.joblib")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if len(files) > settings.MODEL_PURGE_KEEP:
            for old_file in files[settings.MODEL_PURGE_KEEP:]:
                try:
                    os.remove(old_file)
                    logger.debug(f"Purged stale model checkpoint: {old_file}")
                except Exception as e:
                    logger.debug(f"Could not remove old model file {old_file}: {e}")

