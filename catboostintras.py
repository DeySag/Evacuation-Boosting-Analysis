# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

!unzip evacuation-readiness-who-makes-it-out.zip

train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

test_ids = test_df["RecordID"]
drop_cols = ["RecordID", "AccessCode"]

train_df = train_df.drop(columns=drop_cols)
test_df  = test_df.drop(columns=drop_cols)
for df in [train_df, test_df]:
    df["Title"] = df["ClientAlias"].str.extract(
        r",\s*([^\.]+)\.", expand=False
    )

    df["Title"] = df["Title"].replace(
        ["Mlle", "Ms"], "Miss"
    ).replace(
        ["Mme"], "Mrs"
    )

    rare_titles = [
        "Dr", "Rev", "Major", "Col",
        "Sir", "Lady", "Countess",
        "Jonkheer", "Don"
    ]

    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    df.drop(columns="ClientAlias", inplace=True)
for df in [train_df, test_df]:
    df["FamilySize"] = (
        df["DependentsOnboard"] +
        df["DependentsAtDestination"] + 1
    )

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
train_df["Years"] = train_df.groupby(
    ["ServiceTier", "Gender"]
)["Years"].transform(
    lambda x: x.fillna(x.median())
)

years_median = train_df.groupby(
    ["ServiceTier", "Gender"]
)["Years"].median()

def fill_years(row):
    if pd.isna(row["Years"]):
        return years_median.loc[
            row["ServiceTier"], row["Gender"]
        ]
    return row["Years"]

test_df["Years"] = test_df.apply(fill_years, axis=1)
fare_median = train_df["TransactionValue"].median()

train_df["TransactionValue"] = train_df["TransactionValue"].fillna(fare_median)
test_df["TransactionValue"]  = test_df["TransactionValue"].fillna(fare_median)
import numpy as np

for df in [train_df, test_df]:
    df["LogTransactionValue"] = np.log1p(df["TransactionValue"])

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
X = train_df.drop("Outcome", axis=1)
y = train_df["Outcome"]

cat_features = ["Gender", "Title"]
cat_idx = [X.columns.get_loc(c) for c in cat_features]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probs = np.zeros(len(X))
test_probs = np.zeros(len(test_df))
models=[]
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=1500,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        l2_leaf_reg=3,
        subsample=0.9,
        random_seed=42 + fold,
        verbose=False
    )

    model.fit(
        X_tr, y_tr,
        cat_features=cat_idx,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100
    )

    oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
    test_probs += model.predict_proba(test_df)[:, 1]
    models.append(model)
test_probs /= skf.n_splits

import numpy as np
from sklearn.metrics import accuracy_score

thresholds = np.arange(0.45, 0.65, 0.01)
scores = []

for t in thresholds:
    preds = (oof_probs >= t).astype(int)
    scores.append(accuracy_score(y, preds))

best_t = thresholds[np.argmax(scores)]
best_acc = max(scores)

best_t, best_acc

val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))

test_probs = np.zeros(len(test_df))
test_X = test_df[X.columns]

for model in models:
    test_probs += model.predict_proba(test_X)[:, 1]

test_probs /= len(models)
FINAL_THRESHOLD = 0.61
final_preds = (test_probs >= FINAL_THRESHOLD).astype(int)

submission = pd.DataFrame({
    "RecordID": test_ids,
    "Outcome": final_preds
})

submission.to_csv("submission.csv", index=False)
