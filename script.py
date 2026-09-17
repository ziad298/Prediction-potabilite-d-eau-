import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score, make_scorer

# =========================
# 1. CHARGEMENT
# =========================
df = pd.read_csv('water_potability.csv')

print(df.info())
print(df.isnull().sum())

# =========================
# 2. VISUALISATION
# =========================
sns.countplot(x='Potability', data=df)
plt.title('Répartition de la Potabilité')
plt.show()

# =========================
# 3. ANALYSE DES VALEURS MANQUANTES
# =========================
missing_ratio = df.isnull().mean()
print("\nRatio de valeurs manquantes:")
print(missing_ratio)

# =========================
# 4. PHASE 1 - IMPORTANCE DES VARIABLES
# =========================
print("\n=== PHASE 1: Calcul de l'importance des variables ===")

df_temp = df.copy()
for col in df_temp.columns:
    if df_temp[col].isnull().sum() > 0:
        df_temp[col] = df_temp[col].fillna(df_temp[col].median())

X_temp = df_temp.drop('Potability', axis=1)
y_temp = df_temp['Potability']

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_temp, y_temp)

importances = pd.Series(tree.feature_importances_, index=X_temp.columns).sort_values(ascending=False)
print("\nImportance des variables:")
print(importances)

plt.figure(figsize=(8, 4))
importances.plot(kind="bar")
plt.title("Importance des variables (DecisionTree)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

print("\n--- APPROCHE: Garder toutes les colonnes ---")
print("Les deux strategies de gestion des NaN seront testees directement:")
print("  1. Imputation mediane (garde tous les exemples)")
print("  2. Suppression lignes NaN (garde uniquement les exemples complets)")
print("\nAucune colonne n'est supprimee - toutes sont testees.")

# =========================
# 5. CONFIGURATION DES MODELES
# =========================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', random_state=42)
}

print("\nModeles configures:", ", ".join(models.keys()))

# =========================
# 6. FONCTION D'EVALUATION DES STRATEGIES (CROSS-VALIDATION MANUELLE)
# =========================
def evaluate_strategy(df_input, strategy_name):
    X_local = df_input.drop('Potability', axis=1)
    y_local = df_input['Potability']

    best_model_local = None
    best_score_local = -1
    
    # Dictionnaires pour accumuler les résultats par modèle
    model_results = {name: {
        'recall_0_scores': [],
        'accuracy_scores': [],
        'f1_scores': [],
        'roc_auc_scores': [],
        'all_y_true': [],
        'all_y_pred': [],
        'all_y_score': [],
        'cms': []
    } for name in models.keys()}

    print(f"\n==============================")
    print(f"Strategie: {strategy_name}")
    print(f"Validation: 3-Fold Cross-Validation")
    print(f"Taille du dataset: {df_input.shape}")
    print(f"==============================")

    # StratifiedKFold pour garder les proportions des classes
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_num = 0

    for train_idx, test_idx in cv.split(X_local, y_local):
        fold_num += 1
        X_train_fold = X_local.iloc[train_idx]
        X_test_fold = X_local.iloc[test_idx]
        y_train_fold = y_local.iloc[train_idx]
        y_test_fold = y_local.iloc[test_idx]

        # Normalisation DANS chaque fold (pas de fuite de données)
        scaler_fold = StandardScaler()
        X_train_fold = scaler_fold.fit_transform(X_train_fold)
        X_test_fold = scaler_fold.transform(X_test_fold)

        for name, model in models.items():
            # Entraînement
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)

            # Scores de probabilité/décision
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test_fold)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test_fold)
            else:
                y_score = y_pred

            # Calcul des métriques
            acc = accuracy_score(y_test_fold, y_pred)
            f1 = f1_score(y_test_fold, y_pred)
            cm = confusion_matrix(y_test_fold, y_pred)
            tn, fp, fn, tp = cm.ravel()
            recall_0 = tn / (tn + fp)  # Spécificité

            # ROC-AUC
            try:
                auc_value = roc_auc_score(y_test_fold, y_score)
            except:
                auc_value = 0

            # Accumulation des résultats
            model_results[name]['recall_0_scores'].append(recall_0)
            model_results[name]['accuracy_scores'].append(acc)
            model_results[name]['f1_scores'].append(f1)
            model_results[name]['roc_auc_scores'].append(auc_value)
            model_results[name]['all_y_true'].extend(y_test_fold)
            model_results[name]['all_y_pred'].extend(y_pred)
            model_results[name]['all_y_score'].extend(y_score)
            model_results[name]['cms'].append(cm)

    # Affichage et sélection du meilleur modèle
    for name, results in model_results.items():
        mean_recall_0 = np.mean(results['recall_0_scores'])
        std_recall_0 = np.std(results['recall_0_scores'])
        mean_acc = np.mean(results['accuracy_scores'])
        mean_f1 = np.mean(results['f1_scores'])
        mean_auc = np.mean(results['roc_auc_scores'])

        print(f"\n--- {name} ({strategy_name}) ---")
        print(f"Recall classe 0: {mean_recall_0:.3f} (+/- {std_recall_0:.3f})")
        print(f"  Scores par fold: {[f'{s:.3f}' for s in results['recall_0_scores']]}")
        print(f"Accuracy: {mean_acc:.3f}")
        print(f"F1: {mean_f1:.3f}")
        print(f"ROC-AUC: {mean_auc:.3f}")

        if mean_recall_0 > best_score_local:
            best_score_local = mean_recall_0
            best_model_local = name

    # ======= GRAPHIQUES =======
    # 1. CONFUSION MATRICES
    print(f"\n--- Confusion Matrices par fold ---")
    for name, results in model_results.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Confusion Matrices - {name} ({strategy_name})', fontsize=14)
        for fold_idx, cm in enumerate(results['cms']):
            ax = axes[fold_idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'Fold {fold_idx + 1}')
            ax.set_ylabel('True')
            ax.set_xlabel('Pred')
        plt.tight_layout()
        plt.show()

    # 2. COURBES ROC AGGREGÉES
    print(f"\n--- Courbes ROC ---")
    plt.figure(figsize=(10, 7))
    for name, results in model_results.items():
        y_true_all = np.array(results['all_y_true'])
        y_score_all = np.array(results['all_y_score'])
        
        try:
            fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
            auc_value = roc_auc_score(y_true_all, y_score_all)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")
        except:
            pass

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aleatoire')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Courbes ROC - {strategy_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.2)
    plt.show()

    # Réentraîner le meilleur modèle sur TOUT le dataset pour la sauvegarde
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X_local)
    best_model_local = models[best_model_local]
    best_model_local.fit(X_scaled_final, y_local)

    return best_model_local, scaler_final, best_score_local

# =========================
# 7. PREPARATION DONNEES - DEUX STRATEGIES
# =========================

# Strategie 1: Imputation mediane
print("\n=== STRATEGIE 1: Imputation mediane ===")
df_median = df.copy()
for col in df_median.columns:
    if df_median[col].isnull().sum() > 0:
        df_median[col] = df_median[col].fillna(df_median[col].median())

print(f"Dataset apres imputation: {df_median.shape}")
print(f"Valeurs manquantes restantes: {df_median.isnull().sum().sum()}")

# Strategie 2: Suppression des lignes avec NaN
print("\n=== STRATEGIE 2: Suppression lignes NaN ===")
df_dropna = df.copy().dropna().reset_index(drop=True)

if df_dropna.empty:
    raise ValueError("Apres suppression des NaN, le dataset est vide.")

print(f"Dataset original: {df.shape}")
print(f"Dataset apres suppression NaN: {df_dropna.shape}")
print(f"Lignes supprimees: {df.shape[0] - df_dropna.shape[0]}")

# =========================
# 8. EXECUTION DES DEUX STRATEGIES
# =========================

print("\n" + "="*50)
print("EXECUTION DES DEUX STRATEGIES")
print("="*50)

best_model_median, scaler_median, best_score_median = evaluate_strategy(df_median, "imputation_mediane")
best_model_dropna, scaler_dropna, best_score_dropna = evaluate_strategy(df_dropna, "suppression_lignes_nan")

# =========================
# 9. COMPARAISON FINALE ET SELECTION
# =========================

print("\n=== COMPARAISON FINALE ===")
print(f"Meilleur Recall classe 0 (imputation mediane): {best_score_median:.3f}")
print(f"Meilleur Recall classe 0 (suppression NaN): {best_score_dropna:.3f}")

if best_score_median >= best_score_dropna:
    best_model = best_model_median
    scaler = scaler_median
    selected_strategy = "imputation_mediane"
    selected_score = best_score_median
else:
    best_model = best_model_dropna
    scaler = scaler_dropna
    selected_strategy = "suppression_lignes_nan"
    selected_score = best_score_dropna

print(f"\nStrategie retenue: {selected_strategy}")
print(f"Recall classe 0 = {selected_score:.3f}")

# =========================
# 10. SAUVEGARDE DES ARTEFACTS
# =========================

joblib.dump(best_model, 'best_model_water.pkl')
joblib.dump(scaler, 'scaler_water.pkl')

print("\nArtefacts sauvegardes:")
print("- best_model_water.pkl")
print("- scaler_water.pkl")
print("\nScript termine avec succes!")
