# Competetion - Single Nucleotide Polymorphism Classification(+ reveiw)

## 후기

  처음으로 진행중인 데이콘 컴피티션에 참여해보았다. 

  작년 초에 처음으로 머신러닝과 딥러닝을 접하고 조금씩이라도 공부를 하려고 했지만 잘 안되었다. 그동안은 컴피티션을 시도해본적은 있지만 끝까지 진행하지 못했고 누군가의 코드공유를 따라하기만 했었다.

  이번에 머신러닝, 딥러닝 강의를 들을 기회가 생겨서 강의를 듣고 강의 내용들을 직접 적용해보면서 하루하루 진행했다.  

  수업 진행 내용에 따라 분류모델 중 DecisionTree, RandomForest, XGBoost, Lightgbm을 적용해 보았고, 교차검증과 그리드서치, 랜덤서치를 시도해보았다. 그리고 강사님의 추천으로 강의에서 배우지는 않았지만 Hyperotp라는 모듈을 공부하여 성능을 향상시켜보려고 하였다. 

(진행한 코드가 궁금하다면 아래 깃허브를 참고해주길 바랍니다!)

[Dacon_Repo/SNP_Classification at main · Byeon-MJ/Dacon_Repo](https://github.com/Byeon-MJ/Dacon_Repo/tree/main/SNP_Classification)

  결론적으로는 적은 데이터셋으로 진행된 대회였기 때문에 기본적인 LightGBM에서 CrossValidation만 진행을 해주어도 내가 받은 최고점인 97.1이 나와주는 모습을 보였다.

  이후 파라미터 튜닝, 모델 변경 등등을 시도해보았지만 여전히 점수들은 계속 97.1점만 나왔다. 데이터의 수가 많지 않고, feature engineering 을 깊게 진행하지 않아서 그런지 파라미터 튜닝에서 큰 점수 상승을 보이지는 못했다… 내가 할 수 있는 다양한 시도들을 해봤지만 public Score 기준으로 97.1점이 한계였다.   (물론 데이터가 더 많은 Private에서는 점수가 조금씩은 달랐을 수도 있겠지만..) 대회가 끝나고 후기들을 보다보니 다들 나와같은 점수의 문턱에 걸려서 고생들을 한 것 같았다.

## 대회 종료 후…

  Feature Engineerin은 어떻게 진행해야할지 방향을 잡지 못했고 저 점수에서 마무리를 지었다.  최종적으로 private Score는 0.962점, 전체 715명 중에서 224등을 받았다. 50등부터 0.97점이 넘어가는 것을 보면 다들 점수가 비슷비슷했던 것 같다. 마지막에 제출 선택을 못바꿨는데 튜닝 열심히 한 것으로 잘 냈더라면 조금 더 잘 받을수 있지 않았을까 하는 생각도 들었다. 첫 대회에서 이정도면 뭐 나름 만족은 한다! 여러가지 시도를 많이 해보면서 공부가 많이 되었다.

 지금은 상위권 분들의 코드공유가 올라와서 하나씩 참고하면서 새로이 공부를 하는 중이다. snp_info.CSV를 활용을 해야할것 같은 느낌적인 느낌은 있었는데 정말 잘 활용하셨더라…다양한 기술적인 방법들과 많은 노력들이 눈에 보여서 더 열심히 공부해야겠다고 느꼈다.

## 시작

### [Dacon Competetion 사이트]

[유전체 정보 품종 분류 AI 경진대회](https://dacon.io/competitions/official/236035/overview/description)

### [배경]

유전체 염기서열에서 획득한 유전체 변이 정보인 Single Nucleotide Polymorphism 정보는 특정 개체 및 특정 품종에 따라 다른 변이 양상을 나타낼 수 있기 때문에 동일개체를 확인하거나,

동일 품종을 구분하는데 활용이 가능합니다. 따라서 이번 경진대회에서는 개체 정보와 SNP 정보를 이용하여 A, B, C 품종을 분류하는 **최고의 품종구분 정확도**를 획득하는 것이 목표입니다.

  위 사이트와 배경에서도 알 수 있듯 15개의 염기서열을 가지고 3가지 분류를 해야하는 다중분류 분제이다.

## 데이터

### Load Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from sklearn import preprocessing

from google.colab import drive
drive.mount('/content/gdrive')

train = pd.read_csv('/content/gdrive/MyDrive/Project/Dacon_SNP/dataset/train.csv')
test = pd.read_csv('/content/gdrive/MyDrive/Project/Dacon_SNP/dataset/test.csv')
info = pd.read_csv('/content/gdrive/MyDrive/Project/Dacon_SNP/dataset/snp_info.csv')

train.info()
```

  데이터는 ID와 3개의 개체정보, 15개의 SNP정보, 그리고 Class로 총 21개의 컬럼을 가지고 있다.

그리고 전체 데이터의 수는 262개 정도이고 Null값은 보이지 않았다. 

![Untitled](https://user-images.githubusercontent.com/69300448/218361111-2918f54c-bb88-4b5d-bf89-9d803dbcc20e.png)

## 베이스라인

  데이콘에서 최초 제시해주는 Baseline 코드 공유를 따라서 우선 진행해보았다. 데이터에 레이블 인코딩만 적용하고 RandomForest 모델을 사용하여 분류를 진행하였다. Baseline을 따라하면서 데이터셋과 전체적인 흐름들을 파악하였다. 점수는 `F1 Score`로 측정을 하였다. 기준이 되는 Baseline의 F1 Score는 0.9442점이 나왔다.

```python
# Label-Encoding
class_le = preprocessing.LabelEncoder()
snp_le = preprocessing.LabelEncoder()
snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1,16)]

snp_data = []
for col in snp_col:
    snp_data += list(train_x[col].values)

train_y = class_le.fit_transform(train_y)
snp_le.fit(snp_data)

for col in train_x.columns:
    if col in snp_col:
        train_x[col] = snp_le.transform(train_x[col])
        test_x[col] = snp_le.transform(test_x[col])

# Model Fit
clf = RandomForestClassifier(random_state=CFG.SEED)
clf.fit(train_x, train_y)
```

## 전처리

### 1. Unused Columns Drop

  전처리는 필요없는 컬럼을 제거해주는 작업만 했다. 아래 이미지에서 보이듯 [’father’, ‘mother’, ‘gender’] 컬럼은 데이터가 모두 0값만 들어있어서, 성능에 영향을 주지 않을거라는 판단에 삭제를 해 주었다.

```python
train.describe(include='all')

train.drop(['father', 'mother', 'gender'], axis=1, inplace=True)
test.drop(['father', 'mother', 'gender'], axis=1, inplace=True)
```

![Untitled 1](https://user-images.githubusercontent.com/69300448/218361786-9e99398b-e399-4276-8a80-7c7a75dd9856.png)

### 2. Label Encoding

  분류 모델은 트리기반의 모델들을 주로 사용할 것이기때문에 `One-Hot Encoding` 보다는 `Label Encoding`을 선택하였다.

```python
class_le = preprocessing.LabelEncoder()
snp_le = preprocessing.LabelEncoder()
snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1,16)]

train_data = []
for col in snp_col:
    train_data += list(train_x[col].values)

train_y = class_le.fit_transform(train_y)
snp_le.fit(train_data)

for col in train_x.columns:
    if col in snp_col:
        train_x[col] = snp_le.transform(train_x[col])
        test_x[col] = snp_le.transform(test_x[col])

	train_x.head()
```

![Untitled 2](https://user-images.githubusercontent.com/69300448/218361806-3a43881b-ebbb-4d95-9488-1de864027ebe.png)

## 모델링 & 하이퍼 파라미터 튜닝

  모델은 하루하루 강의를 들으며 배운 내용들을 적용해보려고 하였고, 최종적으로 내가 사용해본 모델은 DecisionTree, XGBoost, LightGBM 총 3가지였다. 

  하이퍼 파라미터 튜닝은 기본적으로 GridSearchCV를 진행하였고, 이후 RandomSearch와 HyperOpt를 적용해보았다. Parameter 선택은 사이킷런 공식문서를 참조하였다.

[API Reference](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)

### 1. Decision Tree

```python
# Model Fit
model = DecisionTreeClassifier(max_depth=8)
model.fit(train_x, train_y)

train_pred = model.predict(train_x)
val_pred = model.predict(val_x)
test_pred = model.predict(test_x)

print(f'Train Score : {accuracy_score(train_y, train_pred)}')
print(f'Validation Score : {accuracy_score(val_y, val_pred)}')

# HyperParameter Tuning
def test_depth(depth):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    val_pred = model.predict(val_x)
    print(f"depth : {depth}")
    print(f'Train Score : {accuracy_score(train_y, train_pred)}')
    print(f'Validation Score : {accuracy_score(val_y, val_pred)}')

for i in range(1, 20):
    test_depth(i)
```

### 2. XGBoost

```python
# XGBoost Base
import xgboost as xgb

model = xgb.XGBClassifier()

model.fit(train_x, train_y)

train_pred = model.predict(train_x)
val_pred = model.predict(val_x)
test_pred = model.predict(test_x)

print(f'Train Score : {accuracy_score(train_y, train_pred)}')
print(f'Validation Score : {accuracy_score(val_y, val_pred)}')

# K-Fold
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

train_acc_total = []
val_acc_total = []

model = xgb.XGBClassifier()

for train_index, val_index in kf.split(train_x):
    X_train, X_val = train_x.loc[train_index], train_x.loc[val_index]
    y_train, y_val = train_y[train_index], train_y[val_index]

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    print(f'Train Score : {accuracy_score(y_train, train_pred)}')
    print(f'Validation Score : {accuracy_score(y_val, val_pred)}')
    
    train_acc_total.append(accuracy_score(y_train, train_pred))
    val_acc_total.append(accuracy_score(y_val, val_pred))

test_pred = model.predict(test_x)

# XGBoost Grid Search
from sklearn.model_selection import GridSearchCV, StratifiedKFold

parameter = {
    'learning_rate' : [0.01, 0.1, 0.3, 0.5, 0.7], # 학습률 수정
    'max_depth' : [5, 7, 10, 30, 50],           # 트리 깊이 제한
    'subsample' : [0.5, 0.6, 0.7, 0.8, 1],        # subsample 비율에 따라 부분 추출
    'n_estimators' : [100, 200, 300, 500, 1000]   # 트리 개수
}

model = xgb.XGBClassifier()

# 모델, 파라미터 dict, n_jobs, scoring='f1', cv= kfold -> folding 횟수
gs_model = GridSearchCV(
    estimator = model, param_grid = parameter, scoring='f1_macro', cv=5
)

gs_model.fit(train_x, train_y)

test_pred = gs_model.predict(test_x)

print(gs_model.best_params_)
print(gs_model.best_score_)
```

### 3. LightGBM

```python
# simple LightGBM
import lightgbm as lgb

lgb_clf = LGBMClassifier(random_state=42)

lgb_clf.fit(train_x, train_y)

test_pred_simple = lgb_clf.predict(test_x)

# LightGBM Grid Search
parameter = {
    'learning_rate' : [0.01, 0.1, 0.3, 0.5, 0.7],
    'max_depth' : [5, 7, 10, 30, 50],
    'subsample' : [0.5, 0.6, 0.7, 0.8, 1],
    'n_estimators' : [100, 200, 300, 500, 1000]
}

lgb_clf = LGBMClassifier(random_state=42)

gs_model = GridSearchCV(
    estimator = lgb_clf, param_grid = parameter, scoring='f1_macro', cv=5
)

gs_model.fit(train_x, train_y)

test_pred_gs = gs_model.predict(test_x)

# LightGBM HyperOpt
from hyperopt import hp
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK, fmin, tpe, Trials

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

lgb_search_space = {'n_estimators' : hp.quniform('n_estimators', 100, 1000, 50),
                    'learning_rate':hp.uniform('learning_rate', 0.01, 0.2),
                    'max_depth':hp.quniform('max_depth', 5, 20, 1),
                    'min_child_weight':hp.quniform('min_child_weight', 1, 2, 1),
                    'colsample_bytree':hp.uniform('colsample_bytree', 0.5, 1),
                    'subsample' : hp.quniform('subsample', 0.5, 1, 0.1),
                    'num_leaves' : hp.quniform('num_leaves', 10, 50, 5)
                    # 'lambda_l1' : 
                    # 'lambda_l2' :
                    }

def objective_func(search_space):
    lgb_clf = LGBMClassifier(n_estimators=int(search_space['n_estimators']),
                             learning_rate = search_space['learning_rate'],
                             max_depth=int(search_space['max_depth']),
                             min_child_weight=int(search_space['min_child_weight']),
                             colsample_bytree=search_space['colsample_bytree'],
                             subsample = search_space['subsample'],
                             num_leaves = int(search_space['num_leaves']),
                             eval_metric='logloss')
    accuracy = cross_val_score(lgb_clf, X_train, y_train, scoring='f1_macro', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    return {'loss':-1 * np.mean(accuracy), 'status': STATUS_OK}

trial_val = Trials()
best = fmin(fn=objective_func, space=lgb_search_space,
            algo=tpe.suggest, max_evals=50, trials=trial_val, rstate=np.random.seed(42))
print('best:', best)

lgb_clf = LGBMClassifier(colsample_bytree = best['colsample_bytree'],
                         learning_rate = best['learning_rate'],
                         max_depth = int(best['max_depth']),
                         min_child_weight = int(best['min_child_weight']),
                         n_estimators = int(best['n_estimators']),
                         num_leaves = int(best['num_leaves']),
                         subsample = best['subsample']
                         )

lgb_clf.fit(train_x, train_y, early_stopping_rounds=50, eval_metric='logloss', 
            eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

train_pred = lgb_clf.predict(X_train)
val_pred = lgb_clf.predict(X_val)

train_f1 = f1_score(y_train, train_pred, average='macro')
val_f1 = f1_score(y_val, val_pred, average='macro')

print(train_f1)
print(val_f1)

test = lgb_clf.predict(test_x)
```

## Reference

[유전체 정보 품종 분류 AI 경진대회](https://dacon.io/competitions/official/236035/overview/description)

[scikit-learn](https://scikit-learn.org/stable/index.html)

[XGBoost Documentation — xgboost 1.7.3 documentation](https://xgboost.readthedocs.io/en/stable/)

[Welcome to LightGBM’s documentation! — LightGBM 3.3.2 documentation](https://lightgbm.readthedocs.io/en/v3.3.2/)

[Hyperopt Documentation](http://hyperopt.github.io/hyperopt/)
