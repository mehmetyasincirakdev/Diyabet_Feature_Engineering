import warnings

from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore")

import matplotlib.pyplot as plt
import pandas
import seaborn
import numpy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

pandas.set_option("display.max_columns", None)
pandas.set_option("display.width", 170)
pandas.set_option("display.max_rows", 20)
pandas.set_option("display.float_format", lambda x: "%.3f" % x)

dataframe = pandas.read_csv("Datasets/diabetes.csv")


def check_dataframe(dataframee, head=5):
    print("############################# Shape ############################# ")
    print(dataframee.shape)
    print("############################# Dtypes ############################# ")
    print(dataframee.dtypes)
    print("############################# Head ############################# ")
    print(dataframee.head(head))
    print("############################# Tail ############################# ")
    print(dataframee.tail(head))
    print("############################# NA ############################# ")
    print(dataframee.isnull().sum())
    print("############################# Quantiles ############################# ")
    print(dataframee.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_dataframe(dataframe)


def grab_col_names(dataframee, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframee.columns if dataframee[col].dtypes == "0"]
    num_but_cat = [col for col in dataframee.columns if dataframee[col].nunique() < cat_th and dataframee[col].dtypes != "0"]
    cat_but_car = [col for col in dataframee.columns if dataframee[col].nunique() > car_th and dataframee[col].dtypes == "0"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframee.columns if dataframee[col].dtypes != "0"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Obvervations:{dataframee.shape[0]}")
    print(f"Variables: {dataframee.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)


def cat_summary(dataframee, col_name, plot=False):
    print(pandas.DataFrame({col_name: dataframee[col_name].value_counts(), "Ratio": 100 * dataframee[col_name].value_counts() / len(dataframee)}))

    print("#######################################################################")
    if plot:
        seaborn.countplot(x=dataframee[col_name], data=dataframee)
        plt.show()


cat_summary(dataframe, "Outcome")


def num_summary(dataframee, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframee[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


# for col in num_cols:
#     num_summary(dataframe, col, plot=True)


def target_summary_with_num(dataframee, target, numerical_col):
    print(dataframee.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(dataframe, "Outcome", col)

##################################
# KORELASYON
##################################

# Korelasyon, olasılık kuramı ve istatistikte iki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir

dataframe.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
seaborn.heatmap(dataframe.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# BASE model kurulumu
#####################

y = dataframe["Outcome"]
X = dataframe.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_prediction = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_prediction, y_test), 2)}")
print(f"Recall: {round(recall_score(y_prediction, y_test), 3)}")
print(f"Precision: {round(precision_score(y_prediction, y_test), 2)}")
print(f"F1: {round(f1_score(y_prediction, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_prediction, y_test), 2)}")


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pandas.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    seaborn.set(font_scale=1)
    seaborn.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

# Bir insanda Pregnancies ve Outcome dışındaki değişken değerleri 0 olamayacağı bilinmektedir.
# Bundan dolayı bu değerlerle ilgili aksiyon kararı alınmalıdır. 0 olan değerlere NaN atanabilir .
zero_columns = [col for col in dataframe.columns if (dataframe[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns

# Gözlem birimlerinde 0 olan degiskenlerin her birisine gidip 0 iceren gozlem degerlerini NaN ile değiştirdik.
for col in zero_columns:
    dataframe[col] = numpy.where(dataframe[col] == 0, numpy.nan, dataframe[col])

# Eksik Gözlem Analizi
dataframe.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_dataframe = pandas.concat([n_miss, numpy.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_dataframe, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(dataframe, na_name=True)


# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
def missing_vs_target(dataframe, target, na_columns):
    temp_dataframe = dataframe.copy()
    for col in na_columns:
        temp_dataframe[col + '_NA_FLAG'] = numpy.where(temp_dataframe[col].isnull(), 1, 0)
    na_flags = temp_dataframe.loc[:, temp_dataframe.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pandas.DataFrame({"TARGET_MEAN": temp_dataframe.groupby(col)[target].mean(),
                                "Count": temp_dataframe.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(dataframe, "Outcome", na_columns)

# Eksik Değerlerin Doldurulması
for col in zero_columns:
    dataframe.loc[dataframe[col].isnull(), col] = dataframe[col].median()

dataframe.isnull().sum()


##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframee, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframee, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframee, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in dataframe.columns:
    print(col, check_outlier(dataframe, col))
    if check_outlier(dataframe, col):
        replace_with_thresholds(dataframe, col)

for col in dataframe.columns:
    print(col, check_outlier(dataframe, col))

##################################
# ÖZELLİK ÇIKARIMI
##################################

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
dataframe.loc[(dataframe["Age"] >= 21) & (dataframe["Age"] < 50), "NEW_AGE_CAT"] = "mature"
dataframe.loc[(dataframe["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
dataframe['NEW_BMI'] = pandas.cut(x=dataframe['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glukoz degerini kategorik değişkene çevirme
dataframe["NEW_GLUCOSE"] = pandas.cut(x=dataframe["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma 3 kırılım yakalandı
dataframe.loc[(dataframe["BMI"] < 18.5) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
dataframe.loc[(dataframe["BMI"] < 18.5) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
dataframe.loc[((dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
dataframe.loc[((dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25)) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
dataframe.loc[((dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
dataframe.loc[((dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30)) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
dataframe.loc[(dataframe["BMI"] > 18.5) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
dataframe.loc[(dataframe["BMI"] > 18.5) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
dataframe.loc[(dataframe["Glucose"] < 70) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
dataframe.loc[(dataframe["Glucose"] < 70) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
dataframe.loc[((dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
dataframe.loc[((dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100)) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
dataframe.loc[((dataframe["Glucose"] >= 100) & (dataframe["Glucose"] <= 125)) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
dataframe.loc[((dataframe["Glucose"] >= 100) & (dataframe["Glucose"] <= 125)) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
dataframe.loc[(dataframe["Glucose"] > 125) & ((dataframe["Age"] >= 21) & (dataframe["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
dataframe.loc[(dataframe["Glucose"] > 125) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


# İnsulin Değeri ile Kategorik değişken türetmek
def set_insulin(dataframee, col_name="Insulin"):
    if 16 <= dataframee[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

dataframe["NEW_INSULIN_SCORE"] = dataframe.apply(set_insulin, axis=1)

dataframe["NEW_GLUCOSE*INSULIN"] = dataframe["Glucose"] * dataframe["Insulin"]

# sıfır olan değerler dikkat!!!!
dataframe["NEW_GLUCOSE*PREGNANCIES"] = dataframe["Glucose"] * dataframe["Pregnancies"]
#dataframe["NEW_GLUCOSE*PREGNANCIES"] = dataframe["Glucose"] * (1+ dataframe["Pregnancies"])


# Kolonların büyültülmesi
dataframe.columns = [col.upper() for col in dataframe.columns]

dataframe.head()

##################################
# ENCODING
##################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

# LABEL ENCODING
def label_encoder(dataframee, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() == 2]
binary_cols

for col in binary_cols:
    dataframe = label_encoder(dataframe, col)

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframee, categorical_cols, drop_first=False):
    dataframe = pandas.get_dummies(dataframee, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)

dataframe.head()

##################################
# STANDARTLAŞTIRMA
##################################

num_cols

scaler = StandardScaler()
dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

dataframe.head()
dataframe.shape



##################################
# MODELLEME
##################################

y = dataframe["OUTCOME"]
X = dataframe.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.79
# Recall: 0.711
# Precision: 0.67
# F1: 0.69
# Auc: 0.77

# Base Model
# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75



##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pandas.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    seaborn.set(font_scale=1)
    seaborn.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)
print(dataframe)
print(y_prediction)