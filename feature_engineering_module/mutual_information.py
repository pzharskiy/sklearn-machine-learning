import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression

filepath = "C:/Users/Pavel_Zharski/IdeaProjects/" \
                             "learning/python/" \
                             "sklearn-machine-learning/data/ames.csv"

# Settings to show all columns, not limit them while using head()
pd.set_option('display.max_columns', None)


# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


data = pd.read_csv(filepath)

features = ["YearBuilt", "MoSold", "ScreenPorch"]

sns.relplot(
    x="value",
    y="SalePrice",
    col="variable",
    data=data.melt(id_vars="SalePrice", value_vars=features),
    facet_kws=dict(sharex=False),
)

#plt.show()

X = data.copy()
y = X.pop('SalePrice')

mi_scores = make_mi_scores(X, y)
print(mi_scores.head(20))
# print(mi_scores.tail(20))  # uncomment to see bottom 20
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))

sns.catplot(x="BldgType", y="SalePrice", data=data, kind="boxen");

feature = "GrLivArea"
sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=data, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
);

