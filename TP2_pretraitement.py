import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# PARTIE 1 : EXPLORER ET VERIFIER LES DONNEES
# ==========================================
# Q1 : Importation des données
df = pd.read_csv("transactions_TP2.csv")
# Affichage des 5 premières lignes
print("Les 5 premières lignes sont:")
print(df.head())
# Affichage des dimensions
print("\nLes dimensions du dataset sont:")
print(df.shape)
# Affichage des types de données
print("\nLes types de données de chaque colonne sont:")
print(df.dtypes)
# Q2 : Valeurs manquantes
# Nombre de valeurs manquantes par colonne
nb_manquantes = df.isnull().sum()
# Pourcentage de valeurs manquantes
missing_percentage = df.isnull().sum() / len(df) * 100
# Création du tableau récapitulatif
recap_table = pd.DataFrame({
    'Nombre de valeurs manquantes': nb_manquantes,
    'Pourcentage de valeurs manquantes': missing_percentage
})
# Trier par ordre décroissant
recap_table = recap_table.sort_values(
    by='Pourcentage de valeurs manquantes',
    ascending=False
)
print("\nTableau récapitulatif des valeurs manquantes:")
print(recap_table)
# Q3 : Recherche et suppression des doublons
nb_doublons = df.duplicated(subset=["id_operation"]).sum()
print("\nLe nombre de doublons est:", nb_doublons)
# Suppression des doublons
df = df.drop_duplicates(subset=["id_operation"])
print("\nLe nombre de lignes après suppression des doublons:")
print(len(df))
# Q4 : Histogramme et Boxplot du montant
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['montant_dh'].dropna(), bins=30, edgecolor="black")
plt.title("Histogramme des montants")
plt.xlabel("Montant (DH)")
plt.ylabel("Fréquence")
plt.subplot(1, 2, 2)
plt.boxplot(df['montant_dh'].dropna())
plt.title("Boxplot des montants")
plt.ylabel("Montant (DH)")
plt.tight_layout()
plt.show()
# Q5 : Répartition des classes
count_per_class = df['fraude'].value_counts()
print("\nLe nombre d'opérations par classe:")
print(count_per_class)
# Calcul des proportions
proportion_per_class = df['fraude'].value_counts(normalize=True) * 100
print("\nLa proportion des opérations par classe:")
print(proportion_per_class)
# Diagramme des classes
df["fraude"].value_counts().sort_index().plot(
    kind="bar",
    edgecolor="black"
)
plt.title("Nombre d'opérations par classe")
plt.xlabel("Classe")
plt.ylabel("Nombre d'opérations")
plt.xticks(rotation=0)
plt.show()
# Q6 : Identification des colonnes à exclure
colonnes_a_exclure = [
    "id_operation",
    "id_client",
    "resultat_enquete"
]
print("\nLes colonnes à exclure de l'analyse sont:")
print(colonnes_a_exclure)
print("\nLes identifiants servent au suivi des opérations et des clients.")
print("La colonne resultat_enquete est connue après l'enquête.")
print("Elle doit être exclue pour éviter la fuite d'information.")
# ==========================================
# PARTIE 2 : PREPARER LES DONNEES SANS FUITE
# ==========================================
# Q7 : Création de X et y
X = df.drop(columns=[
    "id_operation",
    "id_client",
    "resultat_enquete",
    "fraude"
])
y = df["fraude"]
print("\nLes dimensions de X sont:", X.shape)
print("Les dimensions de y sont:", y.shape)
print("\nLes colonnes de X sont:")
print(X.columns.tolist())
# Q8 : Division du dataset en train et test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
print("\nLes dimensions du groupe de référence:")
print(X_train.shape)
print("\nLes dimensions du groupe de vérification:")
print(X_test.shape)
# Comparaison des proportions de fraude
print("\nProportion de fraude dans le groupe de référence:")
print(y_train.value_counts(normalize=True) * 100)
print("\nProportion de fraude dans le groupe de vérification:")
print(y_test.value_counts(normalize=True) * 100)
# Q9 : Variables numériques et catégorielles
variables_numeriques = X.select_dtypes(
    include=['number']
).columns.tolist()
variables_categorielles = X.select_dtypes(
    include=['object']
).columns.tolist()
print("\nLes variables numériques sont:")
print(variables_numeriques)
print("\nLes variables catégorielles sont:")
print(variables_categorielles)
# Q10 : Pipeline numérique
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
pipeline_numerique = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Ajustement sur le groupe de référence
x_ref_numerique = pipeline_numerique.fit_transform(
    X_train[variables_numeriques]
)
# Transformation du groupe de vérification
x_verif_numerique = pipeline_numerique.transform(
    X_test[variables_numeriques]
)
print("\nGroupe de référence après transformation numérique:")
print(x_ref_numerique.shape)
print("\nGroupe de vérification après transformation numérique:")
print(x_verif_numerique.shape)
# Q11 : Pipeline catégoriel
from sklearn.preprocessing import OneHotEncoder
pipeline_categorielle = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
# Ajustement sur le groupe de référence
x_ref_categorielle = pipeline_categorielle.fit_transform(
    X_train[variables_categorielles]
)
# Transformation du groupe de vérification
x_verif_categorielle = pipeline_categorielle.transform(
    X_test[variables_categorielles]
)
print("\nGroupe de référence après transformation catégorielle:")
print(x_ref_categorielle.shape)
print("\nGroupe de vérification après transformation catégorielle:")
print(x_verif_categorielle.shape)
# Q12 : ColumnTransformer
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', pipeline_numerique, variables_numeriques),
    ('cat', pipeline_categorielle, variables_categorielles)
])
# Ajustement sur le groupe de référence
x_ref_preprocessed = preprocessor.fit_transform(X_train)
# Transformation du groupe de vérification
x_verif_preprocessed = preprocessor.transform(X_test)
print("\nGroupe de référence après prétraitement:")
print(x_ref_preprocessed.shape)
print("\nGroupe de vérification après prétraitement:")
print(x_verif_preprocessed.shape)
# Q13 : Comparaison StandardScaler et RobustScaler
from sklearn.preprocessing import RobustScaler
# Préparation de la variable montant_dh
imputer_montant = SimpleImputer(strategy='median')
X_train_montant = imputer_montant.fit_transform(
    X_train[['montant_dh']]
)
# Choisir un montant ordinaire et un montant très élevé
montant_ordinaire = X_train['montant_dh'].median()
montant_eleve = X_train['montant_dh'].max()
montants_exemples = pd.DataFrame({
    'montant_dh': [montant_ordinaire, montant_eleve]
})
montants_imputed = imputer_montant.transform(montants_exemples)
# StandardScaler
scaler_standard = StandardScaler()
scaler_standard.fit(X_train_montant)
montants_standard = scaler_standard.transform(montants_imputed)
# RobustScaler
scaler_robust = RobustScaler()
scaler_robust.fit(X_train_montant)
montants_robust = scaler_robust.transform(montants_imputed)
print("\nComparaison StandardScaler et RobustScaler:")
resultat_comparaison = pd.DataFrame({
    'Montant original': montants_exemples['montant_dh'],
    'StandardScaler': montants_standard.ravel(),
    'RobustScaler': montants_robust.ravel()
})
print(resultat_comparaison.round(3))
# Q14 : Explication du Data Leakage
print("\nSi on calcule les paramètres sur toutes les données,")
print("on risque une fuite d'information (Data Leakage).")
print("Il faut calculer les paramètres sur le groupe de référence")
print("uniquement, puis les appliquer au groupe de vérification.")
# ==========================================
# PARTIE 3 : COMPARER LES METHODES DE SCALING
# ==========================================
# Q15 : Calcul des quartiles et de l'IQR
montants_non_manquants = X_train['montant_dh'].dropna()
Q1 = montants_non_manquants.quantile(0.25)
Q3 = montants_non_manquants.quantile(0.75)
IQR = Q3 - Q1
print("\nCalcul des quartiles et de l'IQR:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
# Q16 : Détection des valeurs aberrantes
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = montants_non_manquants[
    (montants_non_manquants < lower_bound) |
    (montants_non_manquants > upper_bound)
]
print("\nLimite inférieure:", lower_bound)
print("Limite supérieure:", upper_bound)
print("\nNombre de valeurs aberrantes détectées:")
print(len(outliers))
print("\nQuelques valeurs aberrantes détectées:")
print(outliers.sort_values(ascending=False).head())
# Q17 : Conservation ou limitation des valeurs extrêmes
# Copie des montants originaux
montants_original = X_train['montant_dh'].copy()
# Limitation des valeurs extrêmes
montants_cap = montants_original.clip(
    lower=lower_bound,
    upper=upper_bound
)
# Comparaison des deux Boxplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.boxplot(montants_original.dropna())
plt.title("Montants originaux")
plt.subplot(1, 2, 2)
plt.boxplot(montants_cap.dropna())
plt.title("Montants après limitation")
plt.tight_layout()
plt.show()
# Q18 : Comparaison des trois Scalers
from sklearn.preprocessing import MinMaxScaler
# Imputation des valeurs manquantes
imputer_montant = SimpleImputer(strategy='median')
X_train_imputed = imputer_montant.fit_transform(
    X_train[['montant_dh']]
)
X_test_imputed = imputer_montant.transform(
    X_test[['montant_dh']]
)
# Création des trois Scalers
scalers = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler()
}
# Dictionnaire pour conserver les résultats
resultats = {}
for nom, scaler in scalers.items():
    # Ajustement sur le groupe de référence
    scaler.fit(X_train_imputed)
    # Transformation du groupe de référence
    X_train_scaled = scaler.transform(X_train_imputed)
    # Transformation du groupe de vérification
    X_test_scaled = scaler.transform(X_test_imputed)
    resultats[nom] = (X_train_scaled, X_test_scaled)
    print(f"\nRésultats après transformation avec {nom}:")
    print("Référence:", X_train_scaled.shape)
    print("Vérification:", X_test_scaled.shape)
# Q19 : Tableau comparatif des transformations
# Choisir trois montants habituels et un montant très élevé
valeurs = pd.DataFrame({
    "montant_dh": [
        X_train['montant_dh'].quantile(0.25),
        X_train['montant_dh'].quantile(0.50),
        X_train['montant_dh'].quantile(0.75),
        X_train['montant_dh'].max()
    ]
})
# Imputation des valeurs choisies
valeurs_imputed = imputer_montant.transform(valeurs)
# Création du tableau comparatif
resultat = pd.DataFrame({
    "Montant_original": valeurs["montant_dh"],
    "MinMax": scalers["MinMaxScaler"].transform(
        valeurs_imputed
    ).ravel(),
    "Standard": scalers["StandardScaler"].transform(
        valeurs_imputed
    ).ravel(),
    "Robust": scalers["RobustScaler"].transform(
        valeurs_imputed
    ).ravel()
})
print("\nTableau comparatif des transformations:")
print(resultat.round(3))
# Q20 : Histogrammes des montants transformés
plt.figure(figsize=(15, 4))
for i, (nom, scaler) in enumerate(scalers.items(), 1):
    # Récupérer les montants transformés du groupe de vérification
    X_test_scaled = resultats[nom][1]
    plt.subplot(1, 3, i)
    plt.hist(
        X_test_scaled,
        bins=30,
        edgecolor="black"
    )
    plt.title(nom)
    plt.xlabel("Montant transformé")
    plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()
# Q21 : Nouvelle transaction supérieure au maximum du groupe de référence
# Récupérer le maximum du groupe de référence
maximum = X_train['montant_dh'].max()
# Créer un nouveau montant supérieur au maximum
nouveau_montant = pd.DataFrame({
    'montant_dh': [maximum * 1.5]
})
# Imputation
nouveau_montant_imputed = imputer_montant.transform(nouveau_montant)
# Transformation avec MinMaxScaler
nouveau_montant_scaled = scalers['MinMaxScaler'].transform(
    nouveau_montant_imputed
)
print("\nQ21 : Nouvelle transaction:")
print("Montant original:", nouveau_montant.iloc[0, 0])
print("Montant transformé:", nouveau_montant_scaled[0, 0])
# Q22 : Nombre de lignes de chaque classe dans le groupe de référence
count_classes = y_train.value_counts().sort_index()
print("\nNombre de lignes de chaque classe:")
print(count_classes)
# Calcul du rapport entre les deux classes
rapport_classes = count_classes[0] / count_classes[1]
print("\nRapport entre les classes:")
print(rapport_classes)
# Q23 : Application de RandomOverSampler
from imblearn.over_sampling import RandomOverSampler
# Création du suréchantillonneur
ros = RandomOverSampler(random_state=42)
# Application sur le groupe de référence uniquement
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
# Comparaison avant et après
print("\nAvant le suréchantillonnage:")
print(y_train.value_counts())
print("\nAprès le suréchantillonnage:")
print(y_train_over.value_counts())
# Q24 : Application de RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
# Création du sous-échantillonneur
rus = RandomUnderSampler(random_state=42)
# Application sur le groupe de référence initial
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
# Comparaison avant et après
print("\nAvant le sous-échantillonnage:")
print(y_train.value_counts())
print("\nAprès le sous-échantillonnage:")
print(y_train_under.value_counts())
# Nombre de lignes supprimées
lignes_supprimees = len(X_train) - len(X_train_under)
print("\nNombre de lignes supprimées:")
print(lignes_supprimees)
# Q25 : Illustration de SMOTE avec deux variables numériques
from imblearn.over_sampling import SMOTE
# Création d'un petit exemple
X_demo = pd.DataFrame({
    'montant': [1, 2, 3, 4, 1, 2],
    'anciennete': [1, 2, 3, 4, 2, 3]
})
y_demo = pd.Series([0, 0, 0, 0, 1, 1])
print("\nDonnées originales:")
print(X_demo)
print("\nClasses originales:")
print(y_demo.value_counts())
# Application de SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_demo_smote, y_demo_smote = smote.fit_resample(
    X_demo, y_demo
)
# Affichage des nouvelles données
print("\nDonnées après SMOTE:")
print(X_demo_smote)
print("\nClasses après SMOTE:")
print(y_demo_smote.value_counts())
# Affichage des observations synthétiques
print("\nObservations synthétiques:")
print(X_demo_smote.iloc[len(X_demo):])
# Q26 : Explication de SMOTE avec les variables catégorielles
print("\nSMOTE peut créer des valeurs intermédiaires.")
print("Cela peut poser un problème avec les variables catégorielles")
print("codées en OneHotEncoder.")
print("\nUne méthode adaptée aux données numériques et catégorielles:")
print("SMOTENC")
# Q27 : Comparaison des effectifs des classes
tableau_equilibrage = pd.DataFrame({
    'Avant rééquilibrage': y_train.value_counts(),
    'Après RandomOverSampler': y_train_over.value_counts(),
    'Après RandomUnderSampler': y_train_under.value_counts()
})
print("\nTableau comparatif des effectifs:")
print(tableau_equilibrage)
# Vérification du groupe de vérification
print("\nNombre de lignes du groupe de vérification:")
print(len(X_test))
print("\nRépartition du groupe de vérification:")
print(y_test.value_counts())
# Q28 : Création du prétraitement final
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# Pipeline numérique
pipeline_numerique_final = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
# Pipeline catégoriel
pipeline_categorielle_final = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
# Combinaison des deux pipelines
preprocessor_final = ColumnTransformer(transformers=[
    ('num', pipeline_numerique_final, variables_numeriques),
    ('cat', pipeline_categorielle_final, variables_categorielles)
])
print("\nLe prétraitement final est créé.")
# Q29 : Ajustement sur le groupe de référence
# Fit uniquement sur le groupe de référence
X_train_final = preprocessor_final.fit_transform(X_train)
# Transformation du groupe de vérification
X_test_final = preprocessor_final.transform(X_test)
# Affichage des dimensions
print("\nDimensions du groupe de référence après prétraitement:")
print(X_train_final.shape)
print("\nDimensions du groupe de vérification après prétraitement:")
print(X_test_final.shape)
# Vérification des valeurs manquantes
from scipy import sparse
if sparse.issparse(X_train_final):
    valeurs_manquantes_train = np.isnan(X_train_final.data).sum()
    valeurs_manquantes_test = np.isnan(X_test_final.data).sum()
else:
    valeurs_manquantes_train = np.isnan(X_train_final).sum()
    valeurs_manquantes_test = np.isnan(X_test_final).sum()
print("\nValeurs manquantes dans le groupe de référence:")
print(valeurs_manquantes_train)
print("\nValeurs manquantes dans le groupe de vérification:")
print(valeurs_manquantes_test)
# Q30 : Simulation d'une nouvelle catégorie inconnue
# Création d'une nouvelle transaction
nouvelle_transaction = X_train.iloc[[0]].copy()
# Ajouter une catégorie inconnue
nouvelle_transaction['canal'] = 'canal_inconnu'
# Transformation avec le prétraitement final
nouvelle_transaction_transformee = preprocessor_final.transform(
    nouvelle_transaction
)
print("\nNouvelle transaction transformée:")
print(nouvelle_transaction_transformee.shape)
print("\nLa transformation a été effectuée sans erreur.")
# Q31 : Enregistrement du prétraitement avec joblib
import joblib
# Enregistrer le prétraitement
joblib.dump(preprocessor_final, "preprocessor_TP2.joblib")
print("\nLe prétraitement est enregistré.")
# Recharger le prétraitement
preprocessor_charge = joblib.load("preprocessor_TP2.joblib")
print("\nLe prétraitement est rechargé.")
# Choisir un petit échantillon
echantillon = X_test.iloc[:5]
# Transformation avec les deux prétraitements
resultat_original = preprocessor_final.transform(echantillon)
resultat_charge = preprocessor_charge.transform(echantillon)
# Comparaison des résultats
if sparse.issparse(resultat_original):
    comparaison = np.allclose(
        resultat_original.toarray(),
        resultat_charge.toarray()
    )
else:
    comparaison = np.allclose(
        resultat_original,
        resultat_charge
    )
print("\nLes résultats sont identiques:")
print(comparaison)
# Q32 : Bilan général du TP2
print("\n========== BILAN DU TP2 ==========")
# Nombre de doublons supprimés
print("\nNombre de doublons supprimés:")
print(nb_doublons)
# Nombre de valeurs manquantes
print("\nNombre de valeurs manquantes dans X_train:")
print(X_train.isnull().sum().sum())
# Nombre de valeurs aberrantes
print("\nNombre de valeurs aberrantes détectées:")
print(len(outliers))
# Méthodes de prétraitement utilisées
print("\nMéthodes de prétraitement utilisées:")
print("- Imputation par la médiane")
print("- Imputation par la valeur la plus fréquente")
print("- StandardScaler")
print("- MinMaxScaler")
print("- RobustScaler")
print("- OneHotEncoder")
# Effet du rééquilibrage
print("\nEffectifs avant rééquilibrage:")
print(y_train.value_counts())
print("\nEffectifs après suréchantillonnage:")
print(y_train_over.value_counts())
print("\nEffectifs après sous-échantillonnage:")
print(y_train_under.value_counts())
print("\n==================================")