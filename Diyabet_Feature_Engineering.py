import warnings

warnings.simplefilter(action="ignore")

import matplotlib.pyplot as plt
import pandas
import seaborn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)
