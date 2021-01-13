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


def Ordered_TS(train_data, test_data, target, cate_features):
    #     print(train_data.head())
    train_data['reindex'] = np.random.permutation(train_data.shape[0])
    train_data['istest'] = 0
    test_data['reindex'] = train_data.shape[0] + np.random.permutation(test_data.shape[0])
    test_data['istest'] = 1
    data_df = pd.concat([train_data, test_data])
    a = 1
    p = train_data[target].mean()
    numerial_feat_name_lst = []
    for feat in cate_features:
        numerial_feat_name = 'numerial_' + feat
        numerial_feat_name_lst.append(numerial_feat_name)
        numerial_feat_value_lst = []
        cate_feat_df = data_df.sort_values(by=[feat, 'reindex'])[[feat, target, 'reindex', 'istest']]
        y_sum = 0
        y_cnt = 0
        elem_pre = -1e9
        for indx in range(cate_feat_df.shape[0]):
            elem = cate_feat_df.iloc[indx, 0]
            y = cate_feat_df.iloc[indx, 1]
            reindex = cate_feat_df.iloc[indx, 2]
            istest = cate_feat_df.iloc[indx, 3]
            if elem != elem_pre:
                y_sum = 0
                y_cnt = 0
            val = (y_sum + a * p) / (y_cnt + a)
            if istest == 0:
                y_sum += y
                y_cnt += 1
            numerial_feat_value_lst.append(val)
            elem_pre = elem
        cate_feat_df[numerial_feat_name] = numerial_feat_value_lst
        if numerial_feat_name in train_data.columns:
            del train_data[numerial_feat_name]
        train_data = train_data.merge(cate_feat_df.loc[cate_feat_df['istest'] == 0, ['reindex', numerial_feat_name]],
                                      on='reindex', how='left')
        if numerial_feat_name in test_data.columns:
            del test_data[numerial_feat_name]
        test_data = test_data.merge(cate_feat_df.loc[cate_feat_df['istest'] == 1, ['reindex', numerial_feat_name]],
                                    on='reindex', how='left')
    return train_data[["id"] + cate_features + numerial_feat_name_lst], \
           test_data[["id"] + cate_features + numerial_feat_name_lst], \
           numerial_feat_name_lst  # , data_df, cate_feat_df, train_data, test_data


def Ordered_TS_Transform(splited_train_data, splited_test_data, target, cate_features, s):
    #     X_train, X_validation, y_train, y_validation = train_test_split(mt_data_201909_df.loc[:, ['id', 'target']+feat_lst].fillna(0),
    #                                                                     mt_data_201909_df.loc[:, 'target'],
    #                                                                     test_size=0.2 , random_state=i*1000)
    #     splited_train_data = X_train
    #     splited_test_data = X_validation
    #     target = 'target'
    #     cate_features = ['2000040013']
    #     s = 1
    ordered_cate_feat_dfs = []
    test_cate_feat_dfs = []
    numerial_cate_features_dict = {}
    for i in range(s):
        print(i)
        ordered_cate_feat_df, test_cate_feat_df, numerial_cate_features = \
            Ordered_TS(splited_train_data, splited_test_data, target, cate_features)
        ordered_cate_feat_dfs.append(ordered_cate_feat_df)
        test_cate_feat_dfs.append(test_cate_feat_df)
    splited_train_data.reset_index(drop=True, inplace=True)
    splited_test_data.reset_index(drop=True, inplace=True)
    for feat in numerial_cate_features:
        splited_train_data[feat] = 0
        splited_test_data[feat] = 0
        for r in range(s):
            splited_train_data[feat] += ordered_cate_feat_dfs[r][feat] / s
            splited_test_data[feat] += test_cate_feat_dfs[r][feat] / s
        numerial_cate_features_dict[feat] = [splited_train_data[feat].mean(), splited_train_data[feat].std()]
    return numerial_cate_features, numerial_cate_features_dict  # ,splited_train_data,splited_test_data


def Get_Noised_Test_Data(X_train, X_validation, ):
    noised_numerial_cate_features = []
    numerial_cate_features_dict = {}
    for feat in cate_features:
        numerial_feat = "numerial_" + feat
        noised_feat = "noised_" + numerial_feat
        noised_numerial_cate_features.append(noised_feat)
        X_train[noised_feat] = X_train[numerial_feat]
        X_validation[noised_feat] = X_validation[numerial_feat]
        numerial_cate_features_dict[numerial_feat] \
            = X_train[[feat, numerial_feat]].groupby(by=feat).agg(['mean', 'std']).to_dict()
        mean_key = (numerial_feat, 'mean')
        std_key = (numerial_feat, 'std')
        for k in numerial_cate_features_dict[numerial_feat][mean_key]:
            mu = numerial_cate_features_dict[numerial_feat][mean_key][k]
            sigma = numerial_cate_features_dict[numerial_feat][std_key][k]
            if np.isnan(sigma):
                sigma = 0.0001
            sz = X_validation.loc[X_validation[feat] == k, noised_feat].shape
            X_validation.loc[X_validation[feat] == k, noised_feat] = np.random.normal(mu, sigma, sz)
    return noised_numerial_cate_features


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


def get_lr_model_with_woe(name, train_df, test_df, woe_transform, step_wise, columns_selected, target_col, params={}):
    # initialise
    c = toad.transform.Combiner()

    to_drop = []

    if woe_transform == False:
        scaler = StandardScaler().fit(train_df[columns_selected])
        train_df[columns_selected] = scaler.transform(train_df[columns_selected])
        test_df[columns_selected] = scaler.transform(test_df[columns_selected])
        train_df[columns_selected] = np.nan_to_num(train_df[columns_selected])
        test_df[columns_selected] = np.nan_to_num(test_df[columns_selected])

    columns_selected.append(target_col)
    if woe_transform == True:
        # Train binning with the selected features from previous; use reliable Chi-squared binning, and control that each bucket has at least 5% sample.
        c.fit(train_df[columns_selected], y=target_col, method='chi', min_samples=0.05,
              exclude=to_drop)  # empty_separate = False
        # Initialise
        transer = toad.transform.WOETransformer()

        # transer.fit_transform() & combiner.transform(). Remember to exclude target
        train_woe = transer.fit_transform(c.transform(train_df[columns_selected]), train_df[target_col],
                                          exclude=to_drop + [target_col])
        test_woe = transer.transform(c.transform(test_df[columns_selected]))
    else:
        train_woe = train_df[columns_selected]
        test_woe = test_df[columns_selected]

    print(train_woe.shape)
    # print(train_woe.describe())

    if step_wise == True:
        # Apply stepwise regression on the WOE-transformed data
        final_data = toad.selection.stepwise(train_woe, target=target_col, estimator='ols', direction='both',
                                             criterion='aic', exclude=to_drop)
        #  Place the selected features to test / OOT sample
        final_test = test_woe[final_data.columns]
        columns_selected = list(final_data.drop(to_drop + [target_col], axis=1).columns)
    else:
        final_data = train_woe
        final_test = test_woe
        columns_selected.remove(target_col)
    print(columns_selected)
    X_train, Y_train = final_data[columns_selected], final_data[target_col]
    X_test, Y_test = final_test[columns_selected], final_test[target_col]

    print(final_data.shape)  # Out of 31 features, stepwise regression selected 10 of them.

    model = LogisticRegression(**params).fit(X_train, Y_train)

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


if __name__ == '__main__':


    cate_features = ['employmentTitle', 'employmentLength_bin', 'purpose', 'postCode', 'earliesCreditLine_bin', \
                     'regionCode', 'title', 'issueDate_bin', 'term_bin', 'homeOwnership_bin']
    train_data = pd.read_csv("data/dk/train_new.csv")
    test_data = pd.read_csv("data/dk/test_new.csv")
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
    X_train, X_validation, y_train, y_validation = train_test_split(
        train_data.loc[:, ['id', target_col] + feat_lst].fillna(0),
        train_data.loc[:, target_col],
        test_size=0.125, random_state=1000)
    lgb_model = LGBMClassifier(**lgb_params).fit(X_train[cate_features].fillna(0), y_train, eval_metric='AUC',
                                                 early_stopping_rounds=100,
                                                 eval_set=[(X_validation[cate_features].fillna(0), y_validation)],
                                                 verbose=False)
    model_test(lgb_model, X_validation[cate_features].fillna(0), y_validation)
    cat_model = CatBoostClassifier(iterations=1000, cat_features=cate_features, eval_metric='AUC', logging_level='Verbose',
                                   learning_rate=0.1, depth=5, l2_leaf_reg=3, loss_function='CrossEntropy', thread_count=8)
    cat_model.fit(X_train.loc[:, cate_features].fillna(0), y_train,
                  eval_set=(X_validation.loc[:, cate_features].fillna(0), y_validation), plot=False, verbose=False)
    model_test(cat_model, X_validation[cate_features].fillna(0), y_validation)

    numerial_cate_features, numerial_cate_features_dict = Ordered_TS_Transform(X_train, X_validation, target_col,
                                                                               cate_features, 10)
    X_train.to_csv('./data/dk/X_train_TS.csv', index=False)
    y_train.to_csv('./data/dk/y_train.csv', index=False)
    X_validation.to_csv('./data/dk/X_validation_TS.csv', index=False)
    y_validation.to_csv('./data/dk/y_validation.csv', index=False)

    lgb_model = LGBMClassifier(**lgb_params).fit(X_train[numerial_cate_features].fillna(0), y_train, eval_metric='AUC',
                                                 early_stopping_rounds=100,
                                                 eval_set=[
                                                     (X_validation[numerial_cate_features].fillna(0), y_validation)], verbose=False)
    model_test(lgb_model, X_validation[numerial_cate_features].fillna(0), y_validation)
    lr_params = {'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 1, 'solver': 'lbfgs'}
    get_lr_model("normal LR", X_train, X_validation, cate_features, target_col, early_stop=False, params=lr_params)
    get_lr_model_with_woe("woe + LR", X_train, X_validation, True, False, cate_features, target_col, params=lr_params)
    get_lr_model("ordered_TS + LR", X_train, X_validation, numerial_cate_features, target_col, early_stop=False,
                 params=lr_params)
    noised_numerial_cate_features = Get_Noised_Test_Data(X_train, X_validation)
    X_train.to_csv('./data/dk/X_train_noised_TS.csv', index=False)
    y_train.to_csv('./data/dk/y_train.csv', index=False)
    X_validation.to_csv('./data/dk/X_validation_noised_TS.csv', index=False)
    y_validation.to_csv('./data/dk/y_validation.csv', index=False)

    model_test(lgb_model, X_validation[noised_numerial_cate_features].fillna(0), y_validation)
    get_lr_model("noised_ordered_TS + LR", X_train, X_validation, noised_numerial_cate_features, target_col,
                 early_stop=False, params=lr_params)
