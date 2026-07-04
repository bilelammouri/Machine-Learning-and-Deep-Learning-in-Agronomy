# 🌾 Machine Learning & Deep Learning in Agronomy

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/Keras%2FTensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green?logo=scikit-learn)

> Applied machine learning and deep learning course materials for agronomy — from classical regression to LSTM forecasting and CNN classification, implemented **in parallel with both Keras/TensorFlow and PyTorch**.

---

## 🎯 Ce que ce repo démontre

Ce repo prouve une maîtrise **opérationnelle** des deux frameworks deep learning industriels sur les **mêmes cas d'usage agronomiques**, ce qui permet de comparer directement leurs API (définition de l'architecture, boucle d'entraînement, gestion des tenseurs, visualisation).

| Compétence | Keras / TensorFlow | PyTorch |
|---|:---:|:---:|
| **Réseau de neurones** — classification (2 & 3 couches cachées) | ✅ TP3 | ✅ TP4 |
| **LSTM** — forecasting de rendement agricole | ✅ TP5 | ✅ TP6 |
| **CNN 1D** — classification de type de culture | ✅ TP7 | ✅ TP8 |
| **ML classique** — régression crop yield | TP1 (sklearn) | TP1 (sklearn) |
| **ML classique** — classification pesticides | TP2 (sklearn) | TP2 (sklearn) |
| **NN from scratch** — rétropropagation NumPy pur | `.py` | `.py` |

> **Exemple d'analyse comparative Keras vs PyTorch** (utile en entretien) :
> - En Keras, la boucle d'entraînement est encapsulée dans `model.fit()` avec des callbacks ; en PyTorch, elle est explicite (forward → loss → backward → optimizer.step()), ce qui donne un contrôle total sur le gradient.
> - La définition des LSTM diffère : en Keras, `LSTM(64, return_sequences=True)` gère automatiquement les états cachés ; en PyTorch, `nn.LSTM` retourne explicitement `(output, (h_n, c_n))`, forçant à manipuler directement l'état caché.
> - Pour les CNN : `Conv1D` en Keras vs `nn.Conv1d` en PyTorch (différence d'ordre des dimensions : `(batch, steps, features)` vs `(batch, features, steps)`).

---

## 📁 Structure du repo

```
├── TP_1_Regression --forecasts crop yield--.ipynb        # Sklearn — régression rendement
├── TP_2_Classification --predict Pesticides--.ipynb      # Sklearn — classification pesticides
├── TP_3_Create_neural_network_models_with_Keras.ipynb    # Keras — NN classification
├── TP_4_Create_neural_network_models_with_PyTorch.ipynb  # PyTorch — NN classification (miroir TP3)
├── TP_5_LSTM_Crop_Yield_Forecasting_Keras.ipynb          # Keras — LSTM forecasting
├── TP_6_LSTM_Crop_Yield_Forecasting_PyTorch.ipynb        # PyTorch — LSTM forecasting (miroir TP5)
├── TP_7_CNN_1D_Crop_Classification_Keras.ipynb           # Keras — CNN 1D classification
├── TP_8_CNN_1D_Crop_Classification_PyTorch.ipynb         # PyTorch — CNN 1D classification (miroir TP7)
├── Neural network -- with numpy --.py                    # NN from scratch (NumPy)
├── Neural network models with Keras - object class -.py  # NN OOP Keras
├── Neural network models with Pytorch - object class -.py# NN OOP PyTorch
├── Data/
│   └── crop_csv_file.xlsx                                # Dataset principal (50k lignes, 1997–2014)
├── Courses-pdf/                                          # Slides de cours
└── Support courses M-D Learning.pdf                      # Support pédagogique complet
```

---

## 📚 Contenu pédagogique

### TP 1 — Régression : Prédiction de rendement agricole (`TP_1`)
Exploration et modélisation de la prédiction de rendement à l'aide de techniques de régression supervisée :
- Analyse exploratoire des données (EDA), traitement des valeurs manquantes et outliers
- Encodage des variables catégorielles (One-Hot, Label, Target Encoding)
- Sélection de features (SelectKBest) et scaling (Min-Max, Standardisation)
- Modèles : Linear, Lasso, Ridge, Decision Tree, Random Forest, SVR, Bayesian Regression

### TP 2 — Classification : Prédiction de l'impact des pesticides (`TP_2`)
Classification multi-classe de l'impact des pesticides sur la santé humaine :
- EDA approfondie, traitement du déséquilibre de classes (SMOTE)
- Modèles : XGBoost, CatBoost, LightGBM, Random Forest, SVM, SGD, Logistic Regression
- Évaluation : classification report, F1-score, comparaison des modèles

### TP 3 & 4 — Réseaux de neurones : Keras vs PyTorch (`TP_3` / `TP_4`)
Implémentation **miroir** d'un réseau de neurones pour la classification binaire :
- Architecture avec 2 et 3 couches cachées, activation ReLU/Softmax
- Visualisation de l'architecture (`plot_model` / `torchviz`)
- Visualisation des frontières de décision apprises

### TP 5 & 6 — LSTM : Forecasting de rendement (`TP_5` / `TP_6`)
Prédiction de la production agricole par séries temporelles avec LSTM :
- Préparation des séquences temporelles (sliding window)
- Architecture : LSTM(64) → LSTM(32) → Dense(1)
- **Keras** : `model.fit()`, `EarlyStopping`, `ModelCheckpoint`
- **PyTorch** : boucle d'entraînement manuelle, gestion explicite de `(h_n, c_n)`
- Comparaison des courbes de loss et des prédictions vs réalité

### TP 7 & 8 — CNN 1D : Classification de type de culture (`TP_7` / `TP_8`)
Classification du type de culture (Kharif / Rabi / Whole Year) à partir des features agronomiques :
- Encodage des séquences de features en tenseurs 1D
- Architecture : `Conv1D(32)` → `MaxPool` → `Conv1D(64)` → `Dense`
- **Keras** : `Conv1D`, `MaxPooling1D`, `GlobalAveragePooling1D`
- **PyTorch** : `nn.Conv1d`, `nn.MaxPool1d`, boucle d'évaluation manuelle
- Analyse des feature maps et courbes d'apprentissage

---

## ⚙️ Installation

```bash
# Cloner le repo
git clone https://github.com/<your-username>/Machine-Learning-and-Deep-Learning-in-Agronomy.git
cd Machine-Learning-and-Deep-Learning-in-Agronomy

# Installer les dépendances
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow keras
pip install torch torchviz
pip install xgboost catboost lightgbm imbalanced-learn openpyxl
```

---

## 📊 Données

Le dataset principal (`Data/crop_csv_file.xlsx`) contient **~50 000 enregistrements** de production agricole en Inde (1997–2014), avec les features : `State_Name`, `District_Name`, `Crop_Year`, `Season`, `Crop`, `Temperature`, `humidity`, `soil moisture`, `area`, `Production`.

---

*Support de cours complet disponible dans `Support courses M-D Learning.pdf`.*
