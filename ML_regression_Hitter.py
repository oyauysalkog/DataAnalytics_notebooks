## HITTERS_DATASET_FINAL_ML ##

import warnings
import pandas as pd
#import missingno as msno
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from helpers.data_prep_oya import *
from helpers.eda_oya import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Tum Base Modeller
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df = pd.read_csv(r"C:\Users\oyauy\Desktop\DSMLDS\WEEK7_Lineer-Logistic_Regression\hitters.csv")

check_df(df)

# Target_variable
sns.distplot(df.Salary)
plt.show()

# Categorical_variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Analyse Categorical_variables
#rare_analyser(df, "Salary", cat_cols)

# Analyse Outliers
for col in num_cols:
    print(col, check_outlier(df, col))

# Replace Outliers
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# Correlations:
corr=df.corr()


# Correlation matrix:
f, ax = plt.subplots(figsize=[20, 15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="Blues")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# most correlated features with target:
corr['Salary'].sort_values(ascending=False)
'''
Out[20]: 
Salary     1.000
CRBI       0.618
CRuns      0.617
CHits      0.598
CHmRun     0.588
CWalks     0.584
CAtBat     0.571
RBI        0.460
Walks      0.460
Hits       0.458
Runs       0.441
Years      0.433
AtBat      0.420
HmRun      0.359
PutOuts    0.269
Assists    0.026
Errors    -0.002'''

df.columns

# Data Preprocessing

df['NEW_HitRatio'] = df['Hits'] / df['AtBat']
df['NEW_RunRatio'] = df['HmRun'] / df['Runs']
df['NEW_CHitRatio'] = df['CHits'] / df['CAtBat']
df['NEW_CRunRatio'] = df['CHmRun'] / df['CRuns']

df['NEW_Avg_AtBat'] = df['CAtBat'] / df['Years']
df['NEW_Avg_Hits'] = df['CHits'] / df['Years']
df['NEW_Avg_HmRun'] = df['CHmRun'] / df['Years']
df['NEW_Avg_Runs'] = df['CRuns'] / df['Years']
df['NEW_Avg_RBI'] = df['CRBI'] / df['Years']
df['NEW_Avg_Walks'] = df['CWalks'] / df['Years']

df.groupby(["NewLeague","Division","League"])["Salary"].mean()

df["NEW_League_Division_NewLeague"] = df["League"] + df["Division"] + df["NewLeague"]
df["NEW_League_Division_NewLeague"].value_counts()
df.groupby("NEW_League_Division_NewLeague")["Salary"].mean()


df['New_Year'] = pd.cut(x=df['Years'], bins=[0, 1, 3, 5, 10, 14, 25],
                        labels=["A", "B", "C", "D", "E", "F"]).astype("O")
pd.crosstab(df["Years"], df["New_Year"])

df["Overall_performance"] = (df["AtBat"] * 10 + df["Hits"] * 20 + df["HmRun"] * 30 + df["Runs"] * 20 + df["RBI"] * 10 + df["Walks"] * 10 ) / 100

df["Triple_interaction"] = df["HmRun"] * df["RBI"] * df["Hits"] / df["AtBat"]
df["CTriple_interaction"] = (df["CHmRun"] / df["Years"]) * (df["CRBI"] / df["Years"]) * df["CHits"] / df["CAtBat"]
df["NEW_FIELDING"] = (df["PutOuts"] + df["Assists"] + df["Errors"])


# One Hot Encoder

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)
df.columns
df.head()

check_df(df)
# Model:
y = df['Salary']
X = df.drop("Salary", axis=1)

rf = RandomForestRegressor(random_state=42).fit(X,y)
rf_CV = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
rf_CV.best_params_
rf_final_model = RandomForestRegressor(**rf_CV.best_params_, random_state=42).fit(X,y)


# Feature importance
def plot_importance(model, features, num=len(X), save=True):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final_model, X)
plt.show()

feature_imp = pd.DataFrame({'Value': rf_final_model.feature_importances_, 'Feature': X.columns})
feature_imp.columns
feature_imp.sort_values("Value")
selected_features = feature_imp[feature_imp["Value"] > 0.03]["Feature"]

a = selected_features.to_list()
a.append("Salary")

df_copy.head()
################################

def val_curve_params(model, X, y, param_name, param_range, scoring="neg_mean_squared_error", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(np.sqrt(-train_score), axis=1)
    mean_test_score = np.mean(np.sqrt(-test_score), axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()



rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestRegressor(random_state=17)


for i in range(len(rf_val_params)):
    val_curve_params(rf_final_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_CV.best_params_








X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def all_models(X, y, test_size=0.33, random_state=42, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df


all_models = all_models(X, y, test_size=0.33, random_state=42, classification=False)


        '''  name  RMSE_TRAIN  RMSE_TEST
9      XGBoost       0.001    236.037
11    CatBoost       4.943    236.291
6           RF      79.783    237.343
8          GBM      34.468    239.913
10    LightGBM      60.860    250.785
4          KNN     197.616    288.496
5         CART       0.000    296.391
3   ElasticNet     214.215    304.764
2        Lasso     210.687    312.356
0           LR     208.521    312.607
1        Ridge     209.929    312.912
7          SVR     392.921    375.949  '''




# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı    400 / 10 = 40, son yıldaki değişim 30/40
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı


##################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

check_df(df)

# One Hot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

models1 = [('RF', RandomForestRegressor()),
           ("LightGBM", LGBMRegressor()),
           (('LR', LinearRegression())]

for name, regressor in models1:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# Automated Hyperparameter Optimization

rf_params = {"max_depth": [15, 20],
             "max_features": [3, 5],
             "min_samples_split": [5, 8],
             "n_estimators": [500, 700]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [ 200, 300],
                   "colsample_bytree": [0.5, 0.7]}

regressor1 = [("RF", RandomForestRegressor(), rf_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]


best_models = {}

for name, regressor, params in regressor1:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


