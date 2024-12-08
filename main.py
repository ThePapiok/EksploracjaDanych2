
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

# wczytanie danych z separatorem ,
df = pd.read_csv("./loan_data.csv", sep=",")

# wyświetlenie danych
print(df)

# wyświetalnie dodatkowych informacji
print(df.info())

pd.set_option("display.max_columns",None)

# zmieniamy typ na int gdyż nie potrzebujemy floatow
df["person_age"] = df["person_age"].astype(int)
df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].astype(int)

before_length = len(df)

# wyswietlamy unikalne wartosci dla każdej kolumny
for column in df.columns:
    print(column + ": ", df[column].unique())

# sprawdzamy czy nie ma takich rekordów, którzy więcej pracowali lub mieli dłuższą historie kredytową niż ich długość życia
found = False
for index, row in df.iterrows():
    if row["person_age"] < row["person_emp_exp"]:
        found = True
        break
    if row["person_age"] < row["cb_person_cred_hist_length"]:
        found = True
        break
if found:
    print("Są takie obserwacje, których czas pracy lub dlugość historii kredytowej jest większa niż długość życia")
else:
    print("Nie ma obserwacji, których czas pracy lub dlugość historii kredytowej jest większa niż długość życia")

# sprawdzamy czy są duplikaty
print("Ilość duplikatów - ", len(df[df.duplicated()]))

# usuwamy obserwacje gdzie wiek jest powyżej 100 lat
df = df[df["person_age"] <= 100]
print("Usunięto ", before_length - len(df), " wyników, gdzie wiek większy niż 100")

# służy nam do wyznaczenia podstawowych statystyk
print(df.describe())

# sprawdzamy jakiej częsci osób został udzielony kredyt
print(df["loan_status"].value_counts(normalize = True))

# sprawdzmy czy są jakieś zależności, między zmiennymi
print(pd.crosstab(df["loan_status"], df["person_gender"], normalize="columns"))
print(pd.crosstab(df["loan_status"], df["person_education"], normalize="columns"))
print(pd.crosstab(df["loan_status"], df["previous_loan_defaults_on_file"], normalize = "columns"))

plt.figure(figsize=(15, 6))
sns.countplot(x='person_home_ownership', hue='loan_status', data=df)
plt.title('Własność a status pożyczki')
plt.xlabel('Własność')
plt.ylabel('Liczba osób')
plt.show()

plt.figure(figsize=(15, 6))
sns.countplot(x='loan_intent', hue='loan_status', data=df)
plt.title('Cel pożyczki a status pożyczki')
plt.xlabel('Cel Pożyczki')
plt.ylabel('Liczba osób')
plt.show()

#obliczamy korelacje
print("Korelacja: ", (df["loan_amnt"] * 100 / df["person_income"]).corr(df["loan_percent_income"]))

# sprawdzamy zależności między zmiennymi liczbowymi
plt.figure(figsize=(10,7))
sns.set_theme(font_scale = 1.3)
sns.heatmap(df.select_dtypes(include="number").corr()["loan_status"].to_frame(), cbar = True, annot = True, square = True, cmap = "Blues")
plt.show()

# odrzucamy możliwe nieznaczące zmienne
df = df.drop(["person_gender", "person_education", "loan_percent_income", "person_age", "person_emp_exp", "cb_person_cred_hist_length", "credit_score"], axis=1)

# wyciągamy predytkory oraz zmienne celu
y = df["loan_status"]
X = df.drop(["loan_status"], axis=1)

# dzielimy na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 313128)

# dzielimy zmienne na numeryczne i kategoryczne
sel_num = make_column_selector(dtype_include=["int64", "float64"])
sel_cat = make_column_selector(dtype_include="object")

# zmienne numeryczne normalizujemy, a z katogrycznymi dzielimy na osobne kolumny dla kazdej wartości
preprocessor = ColumnTransformer(transformers =
                                [("num", MinMaxScaler(feature_range = (0, 1)), sel_num),
                                 ("cat", OneHotEncoder(handle_unknown = "ignore"), sel_cat)])

# tworzymy potok, który na początku wykona preprocessor, który utworzyliśmy wyżej a potem juz klasyfikator oparty o sieci neuronowe, parametry zostały dobrane dzięki grid search
pipeline = Pipeline(steps = [("prep", preprocessor),
                        ("siec", MLPClassifier(random_state=313128, max_iter=1000, hidden_layer_sizes=(50, 50)))])
pipeline.fit(X_train, y_train)

# uzyskujemy trafność modelu
print("Trafność: ")
print("Zbiór uczący: ", pipeline.score(X_train, y_train))
print("Zbiór testowy: ", pipeline.score(X_test, y_test))

# uzyskujemy macierze pomylek dla uczącej i testowej
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Macierze pomyłek: ")
print("Zbiór uczący: ")
print(pd.DataFrame(confusion_matrix_test))
print("Zbiór testowy: ")
print(pd.DataFrame(confusion_matrix_train))

# wyznaczamy tp, tn, fp, fn
tn_train, fp_train, fn_train, tp_train = confusion_matrix_train.ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix_test.ravel()

# uzyskujemy czułość modelu
sensitivity_train = tp_train/(tp_train + fn_train)
sensitivity_test = tp_test/(tp_test + fn_test)
print("Czułość: ")
print("Zbiór uczący: ", sensitivity_train)
print("Zbiór testowy: ", sensitivity_test)

# uzyskujemy swoistość modelu
print("Swoistość: ")
print("Zbiór uczący: ", tn_train/(tn_train + fp_train))
print("Zbiór testowy: ", tn_test/(tn_test + fp_test))

# uzyskujemy precyzje modelu
precision_train = tp_train/(tp_train + fp_train)
precision_test = tp_test/(tp_test + fp_test)
print("Precyzja: ")
print("Zbiór uczący: ", precision_train)
print("Zbiór testowy: ", precision_test)

# uzyskujemy f1 modelu
print("F1: ")
print("Zbiór uczący: ", 2*sensitivity_train*precision_train/(sensitivity_train + precision_train))
print("Zbiór testowy: ", 2*sensitivity_test*precision_test/(sensitivity_test + precision_test))

# uzyskujemy krzywą roc wraz z auc dla zbioru uczącego
y_train_prob = pipeline.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob, pos_label = 1)
auc = roc_auc_score(y_train, y_train_prob)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specyficzność')
plt.ylabel('Czułość')
plt.title('Krzywa ROC - zbiór uczący')
plt.legend(loc="lower right")
plt.show()

#uzyskujemy krzywą roc wraz z auc dla zbioru testowego
y_test_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob, pos_label = 1)
auc = roc_auc_score(y_test, y_test_prob)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specyficzność')
plt.ylabel('Czułość')
plt.title('Krzywa ROC - zbiór testowy')
plt.legend(loc="lower right")
plt.show()

# zmienne katogryczne dzielimy na osobne kolumny dla kazdej wartości
preprocessor = ColumnTransformer(transformers =
                                [('num', 'passthrough', sel_num),
                                 ("cat", OneHotEncoder(handle_unknown = "ignore"), sel_cat)])

# tworzymy potok, który na początku wykona preprocessor, który utworzyliśmy wyżej a potem juz klasyfikator oparty o xgboost, parametry zostały dobrane dzięki grid search
pipeline = Pipeline(steps = [("prep", preprocessor),
                      ("las", XGBClassifier(objective='binary:logistic', gamma=1,  max_depth=7, n_estimators=500, subsample=0.8, learning_rate=0.1, random_state=313128))])
pipeline.fit(X_train, y_train)

# uzyskujemy trafność modelu
print("Trafność: ")
print("Zbiór uczący: ", pipeline.score(X_train, y_train))
print("Zbiór testowy: ", pipeline.score(X_test, y_test))

# uzyskujemy macierze pomylek dla uczącej i testowej
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
confusion_matrix_test = confusion_matrix(y_train, y_train_pred)
confusion_matrix_train = confusion_matrix(y_test, y_test_pred)
print("Macierze pomyłek: ")
print("Zbiór uczący: ")
print(pd.DataFrame(confusion_matrix_test))
print("Zbiór testowy: ")
print(pd.DataFrame(confusion_matrix_train))

# wyznaczamy tp, tn, fp, fn
tn_train, fp_train, fn_train, tp_train = confusion_matrix_train.ravel()
tn_test, fp_test, fn_test, tp_test = confusion_matrix_test.ravel()

# uzyskujemy czułość modelu
sensitivity_train = tp_train/(tp_train + fn_train)
sensitivity_test = tp_test/(tp_test + fn_test)
print("Czułość: ")
print("Zbiór uczący: ", sensitivity_train)
print("Zbiór testowy: ", sensitivity_test)

# uzyskujemy swoistość modelu
print("Swoistość: ")
print("Zbiór uczący: ", tn_train/(tn_train + fp_train))
print("Zbiór testowy: ", tn_test/(tn_test + fp_test))

# uzyskujemy precyzje modelu
precision_train = tp_train/(tp_train + fp_train)
precision_test = tp_test/(tp_test + fp_test)
print("Precyzja: ")
print("Zbiór uczący: ", precision_train)
print("Zbiór testowy: ", precision_test)

# uzyskujemy f1 modelu
print("F1: ")
print("Zbiór uczący: ", 2*sensitivity_train*precision_train/(sensitivity_train + precision_train))
print("Zbiór testowy: ", 2*sensitivity_test*precision_test/(sensitivity_test + precision_test))

# Wyświetlenie wykresu ważności cech
importances = pipeline["las"].feature_importances_
plt.figure(figsize=(15,5))
plt.barh(range(len(importances)), importances)
plt.yticks(range(len(importances)), pipeline["prep"].get_feature_names_out())
plt.xlabel('Ważność cech')
plt.show()

# uzyskujemy krzywą roc wraz z auc dla zbioru uczącego
y_train_prob = pipeline.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob, pos_label = 1)
auc = roc_auc_score(y_train, y_train_prob)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specyficzność')
plt.ylabel('Czułość')
plt.title('Krzywa ROC - zbiór uczący')
plt.legend(loc="lower right")
plt.show()

# uzyskujemy krzywą roc wraz z auc dla zbioru testowego
y_test_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob, pos_label = 1)
auc = roc_auc_score(y_test, y_test_prob)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specyficzność')
plt.ylabel('Czułość')
plt.title('Krzywa ROC - zbiór testowy')
plt.legend(loc="lower right")
plt.show()