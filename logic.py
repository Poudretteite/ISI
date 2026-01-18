# noinspection PyUnresolvedReferences
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, LeaveOneOut, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, balanced_accuracy_score, precision_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io


# --- 1. WCZYTYWANIE ---
def load_data(file_path, no_header=False):
    try:
        header_opt = None if no_header else 'infer'
        df = pd.read_csv(file_path, header=header_opt, engine='python')

        if df.shape[1] < 2:
            df = pd.read_csv(file_path, sep=';', header=header_opt, engine='python')

        if no_header:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
        else:
            df.columns = df.columns.str.strip()

        return df
    except Exception:
        return None


def get_categorical_targets(df):
    candidates = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 30:
            candidates.append(col)
    return candidates


# --- 2. PRZETWARZANIE ---
# noinspection PyBroadException
def preprocess_data(df, target_column, selected_features=None):
    data = df.copy()

    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].str.replace(',', '.').astype(float)
            except ValueError:
                pass

    data = data.dropna()

    if selected_features:
        selected_features = list(selected_features)
        available_feats = [c for c in selected_features if c in data.columns]
        if not available_feats:
            raise ValueError("Wybrane kolumny nie istnieją!")
        X = data[available_feats]
    else:
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
        else:
            X = data

    y = data[target_column].astype(str)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    feature_meta = []
    for col in X.columns:
        if col in categorical_features:
            opts = sorted(X[col].unique().astype(str).tolist())
            feature_meta.append({"name": col, "type": "cat", "options": opts})
        else:
            feature_meta.append({"name": col, "type": "num"})

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ],
        verbose_feature_names_out=False
    )

    return X, y, preprocessor, feature_meta


# --- 3. METRYKI (ROZSZERZONE) ---
def calculate_metrics(y_true, y_pred, classes, model_loss=None):
    """Oblicza bogaty zestaw metryk."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # average='macro' liczy średnią nie ważoną liczebnością klas (traktuje każdą klasę równo)
    sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Swoistość (Specificity)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_specificity = tn / (tn + fp)
        class_specificity = np.nan_to_num(class_specificity)

    specificity = np.mean(class_specificity)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "best_loss": model_loss if model_loss is not None else 0.0
    }


# --- 4. TRENING I EWALUACJA ---
def train_and_evaluate(X, y, preprocessor, hidden_layers, activation, max_iter, alpha, solver,
                       method="split", k_folds=5):
    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    try:
        layers = tuple(map(int, hidden_layers.split(',')))
    except ValueError:
        layers = (100,)

    # Model bazowy (szablon)
    model_params = {
        "hidden_layer_sizes": layers,
        "activation": activation,
        "solver": solver,
        "alpha": float(alpha),
        "max_iter": int(max_iter),
        "random_state": 42
    }

    model = MLPClassifier(**model_params)

    y_true_for_metrics = []
    y_pred_for_metrics = []

    # --- LOGIKA EWALUACJI ---
    if method == "split":
        x_tr, x_te, y_tr, y_te = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_te)
        y_true_for_metrics = y_te
        y_pred_for_metrics = y_pred
        final_model = model

    elif method == "cv":
        cv = StratifiedKFold(n_splits=int(k_folds), shuffle=True, random_state=42)
        y_pred_for_metrics = cross_val_predict(model, X_processed, y, cv=cv)
        y_true_for_metrics = y
        final_model = MLPClassifier(**model_params)
        final_model.fit(X_processed, y)

    elif method == "shuffle":
        # ShuffleSplit - losowe permutacje (np. 10 powtórzeń)
        # cross_val_predict tutaj działa nieco inaczej, zbierzemy predykcje z losowań
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        # Dla uproszczenia w Shuffle użyjemy też cross_val_predict (zwróci predykcje dla instancji, które trafiły do testu)
        y_pred_for_metrics = cross_val_predict(model, X_processed, y, cv=cv)
        y_true_for_metrics = y  # Uwaga: w ShuffleSplit nie każdy element musi trafić do testu, sklearn sobie z tym radzi

        final_model = MLPClassifier(**model_params)
        final_model.fit(X_processed, y)

    elif method == "loo":
        cv = LeaveOneOut()
        y_pred_for_metrics = cross_val_predict(model, X_processed, y, cv=cv)
        y_true_for_metrics = y
        final_model = MLPClassifier(**model_params)
        final_model.fit(X_processed, y)

    else:
        raise ValueError("Nieznana metoda ewaluacji")

    # Pobieramy best_loss_ jeśli dostępny
    loss_val = getattr(final_model, 'best_loss_', None)
    if loss_val is None and hasattr(final_model, 'loss_'):  # Dla LBFGS
        loss_val = final_model.loss_

    metrics = calculate_metrics(y_true_for_metrics, y_pred_for_metrics, final_model.classes_, model_loss=loss_val)

    return final_model, metrics, y_true_for_metrics, y_pred_for_metrics


# --- 5. SERIALIZACJA ---
def save_model_pipeline(model, preprocessor, feature_meta):
    pipeline_obj = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_meta": feature_meta
    }
    buffer = io.BytesIO()
    joblib.dump(pipeline_obj, buffer)
    buffer.seek(0)
    return buffer


# --- 6. WYKRESY ---
# noinspection PyUnresolvedReferences
def get_confusion_matrix_plot(y_true, y_pred, classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=classes, yticklabels=classes)
    ax.set_title("Macierz Pomyłek")
    ax.set_xlabel("Przewidywana")
    ax.set_ylabel("Prawdziwa")
    return fig


# noinspection PyUnresolvedReferences
def get_loss_curve_plot(model):
    fig, ax = plt.subplots(figsize=(5, 4))
    if hasattr(model, 'loss_curve_'):
        ax.plot(model.loss_curve_)
        ax.set_title("Krzywa Straty (Uczenie)")
        ax.set_xlabel("Iteracje")
        ax.set_ylabel("Strata")
        ax.grid(True, linestyle='--', alpha=0.7)
    else:
        ax.text(0.5, 0.5, "Brak krzywej (Solver LBFGS)", ha='center')
        ax.set_axis_off()
    return fig