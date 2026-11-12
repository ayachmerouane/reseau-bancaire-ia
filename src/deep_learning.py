"""Approfondissement Deep Learning avec Keras (TensorFlow).

Deux architectures :
  - MLP : couches denses, BatchNorm et Dropout. C'est le réseau de base pour des données tabulaires.
  - CNN 1D : traite le vecteur de caractéristiques comme une séquence et apprend des motifs
    locaux. Souvent cité dans la littérature IDS : à comparer honnêtement au MLP et à XGBoost.

TensorFlow est lourd (environ 500 Mo) : il n'est installé que si on fait cette partie
(pip install tensorflow). Sur Colab ou Kaggle, il est déjà installé.
"""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .pretraitement import LogSigne


def _keras():
    import tensorflow as tf
    tf.random.set_seed(config.GRAINE)
    return tf.keras


def construire_mlp(n_entrees: int, n_classes: int):
    keras = _keras()
    return keras.Sequential([
        keras.layers.Input(shape=(n_entrees,)),
        keras.layers.Dense(256, activation="relu"), keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"), keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])


def construire_cnn1d(n_entrees: int, n_classes: int):
    keras = _keras()
    return keras.Sequential([
        keras.layers.Input(shape=(n_entrees, 1)),
        keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(), keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(), keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])


class ModeleKeras:
    """Enveloppe qui donne au réseau la même interface que scikit-learn
    (fit / predict / predict_proba), pour réutiliser src/evaluation.py tel quel."""

    def __init__(self, architecture: str = "mlp", epoques: int = 30, taille_lot: int = 1024):
        self.architecture = architecture
        self.epoques = epoques
        self.taille_lot = taille_lot
        self.prep = make_pipeline(LogSigne(), StandardScaler())

    def _forme(self, X):
        X = self.prep.transform(X).astype("float32")
        return X[..., None] if self.architecture == "cnn1d" else X

    def fit(self, X, y, X_val=None, y_val=None, poids_classes: bool = True):
        keras = _keras()
        y = np.asarray(y)
        self.n_classes_ = int(y.max()) + 1
        self.prep.fit(X)
        Xt = self._forme(X)
        construire = construire_cnn1d if self.architecture == "cnn1d" else construire_mlp
        self.reseau = construire(np.asarray(X).shape[1], self.n_classes_)
        self.reseau.compile(optimizer=keras.optimizers.Adam(1e-3),
                            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        poids = None
        if poids_classes:
            cl = np.unique(y)
            poids = dict(zip(cl, compute_class_weight("balanced", classes=cl, y=y)))
        val = (self._forme(X_val), np.asarray(y_val)) if X_val is not None else None
        arret = keras.callbacks.EarlyStopping(monitor="val_loss" if val else "loss", patience=4,
                                              restore_best_weights=True)
        self.historique = self.reseau.fit(Xt, y, validation_data=val, epochs=self.epoques,
                                          batch_size=self.taille_lot, class_weight=poids,
                                          callbacks=[arret], verbose=2)
        return self

    def predict_proba(self, X):
        return self.reseau.predict(self._forme(X), batch_size=4096, verbose=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
