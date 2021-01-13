import pandas as pd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import toad


def model_test(model, X_test, Y_test):
    test_pred_proba = model.predict_proba(X_test)
    roc_auc_test = roc_auc_score(Y_test, test_pred_proba[:, 1])
    get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    ks_test = get_ks(test_pred_proba[:, 1], Y_test)
    print('roc_auc_test =', roc_auc_test)
    print('ks_test =', ks_test)



def get_lr_model(name, train_df, test_df, columns_selected, target_col, early_stop=False, params={}):
    X_train = train_df[columns_selected].fillna(0)
    Y_train = train_df[target_col]
    X_test = test_df[columns_selected].fillna(0)
    Y_test = test_df[target_col]

    print(X_train.shape, X_test.shape, np.sum(Y_train), np.sum(Y_test))
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    if early_stop == False:
        model = LogisticRegression(**params).fit(X_train, Y_train)
    else:
        model = LogisticRegression(**params).fit(X_train, Y_train, eval_metric=ks_metric, early_stopping_rounds=100,
                                                 eval_set=[(X_test, Y_test)])

    train_pred_proba = model.predict_proba(X_train)
    test_pred_proba = model.predict_proba(X_test)
    roc_auc_test = roc_auc_score(Y_test, test_pred_proba[:, 1])
    roc_auc_train = roc_auc_score(Y_train, train_pred_proba[:, 1])
    get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    ks_train = get_ks(train_pred_proba[:, 1], Y_train)
    ks_test = get_ks(test_pred_proba[:, 1], Y_test)
    print('task name =', name)
    print('roc_auc_train =', roc_auc_train)
    print('roc_auc_test =', roc_auc_test)
    print('ks_train =', ks_train)
    print('ks_test =', ks_test)
    return model



cate_features = ['employmentTitle', 'employmentLength_bin', 'purpose', 'postCode', 'earliesCreditLine_bin', \
                 'regionCode', 'title', 'issueDate_bin', 'term_bin', 'homeOwnership_bin']
train_data = pd.read_csv("data/dk/train_new.csv")[:7000]
test_data = pd.read_csv("data/dk/test_new.csv")[:1000]
feat_lst = list(test_data.columns[1:])

feat_lst.remove('installment_homeOwnership_ratio')
feat_lst.remove('installment_purpose_ratio')
feat_lst.remove('revolBal_issueDate_ratio')
feat_lst.remove('revolBal_loanAmnt')
feat_lst.remove('annualIncome_installment')
feat_lst.remove('installment_issueDate_ratio')
feat_lst.remove('installment_employmentLength_ratio')
feat_lst.remove('revolUtil_issueDate_ratio')
feat_lst.remove('revolBal_purpose_ratio')
feat_lst.remove('revolBal_homeOwnership_ratio')
feat_lst.remove('revolBal_employmentLength_ratio')
feat_lst.remove('dti_issueDate_ratio')

n_feat_lst = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']
for col in n_feat_lst:
    feat_lst.remove(col)

# X_train = pd.read_csv("data/dk/X_train.csv", index_col=0)
# X_validation = pd.read_csv("data/dk/X_validation.csv", index_col=0)
# y_train = pd.read_csv("data/dk/y_train.csv", index_col=0)
# y_validation = pd.read_csv("data/dk/y_validation.csv", index_col=0)

lgb_params = {'metric': 'auc',
              'lambda_l1': 0,
              'lambda_l2': 3,
              'num_leaves': 10,
              'feature_fraction': 0.7,
              'bagging_fraction': 0.7,
              'bagging_freq': 3,
              'min_child_samples': 50,
              'learning_rate': 0.1,
              'num_round': 1000}

target_col = 'isDefault'
# X_train, X_validation, y_train, y_validation = train_test_split(
#     train_data.loc[:, ['id', target_col] + feat_lst].fillna(0),
#     train_data.loc[:, target_col],
#     test_size=0.125, random_state=1000)
X_train = pd.read_csv('./data/dk/X_train_TS.csv')
y_train = pd.read_csv('./data/dk/y_train.csv')[target_col]
X_validation = pd.read_csv('./data/dk/X_validation_TS.csv')
y_validation = pd.read_csv('./data/dk/y_validation.csv')[target_col]

lgb_model = LGBMClassifier(**lgb_params).fit(X_train[feat_lst], y_train, eval_metric='AUC',
                                             early_stopping_rounds=100,
                                             eval_set=[(X_validation[feat_lst], y_validation)],
                                             verbose=False)
model_test(lgb_model, X_validation[feat_lst], y_validation)

lr_params = {'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 1, 'solver': 'lbfgs'}
get_lr_model("normal LR", X_train, X_validation, feat_lst, target_col, early_stop=False, params=lr_params)
# get_lr_model_with_woe("woe + LR", X_train, X_validation, True, False, cate_features, target_col, params=lr_params)
# get_lr_model("ordered_TS + LR", X_train, X_validation, numerial_cate_features, target_col, early_stop=False,
#              params=lr_params)