from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import gc

# 特征工程

grade_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

employmentLength_dict = {'1 year': 1,
                         '10+ years': 10,
                         '2 years': 2,
                         '3 years': 3,
                         '4 years': 4,
                         '5 years': 5,
                         '6 years': 6,
                         '7 years': 7,
                         '8 years': 8,
                         '9 years': 9,
                         '< 1 year': 0}


def get_sub_grade(grade, sub):
    return grade * 10 + int(sub[1])


def trans_issueDate(issueDate):
    year, month, day = issueDate.split('-')
    return int(year) * 12 + int(month) - 1


def trans_earliesCreditLine(earliesCreditLine):
    month_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
                  "Nov": 11, "Dec": 12}
    month, year = earliesCreditLine.split('-')
    month = month_dict[month]
    return int(year) * 12 + month - 1


# 类别特征
cate_features = ['employmentTitle', 'employmentLength_bin', 'purpose', 'postCode', 'subGrade', 'earliesCreditLine_bin',
                 'regionCode', 'title', 'issueDate_bin', 'term_bin', 'interestRate_bin', 'annualIncome_bin',
                 'loanAmnt_bin', 'homeOwnership_bin', 'revolBal_bin', 'dti_bin', 'installment_bin', 'revolBal_bin',
                 'revolUtil_bin']


def process(dfs):
    for df in dfs:
        print(df.shape)
        df['grade'] = df['grade'].apply(lambda x: x if x not in grade_dict else grade_dict[x])
        df['subGrade'] = df.apply(lambda row: get_sub_grade(row['grade'], row['subGrade']), axis=1)
        df['employmentLength'] = df['employmentLength'].apply(
            lambda x: x if x not in employmentLength_dict else employmentLength_dict[x])
        df['issueDate'] = df['issueDate'].apply(lambda x: trans_issueDate(x))
        df['earliesCreditLine'] = df['earliesCreditLine'].apply(lambda x: trans_earliesCreditLine(x))
        df['date_Diff'] = df['issueDate'] - df['earliesCreditLine']
        df['installment_term_revolBal'] = df['installment'] * 12 * df['term'] / (df['revolBal'] + 0.1)
        df['revolUtil_revolBal'] = df['revolUtil'] / (df['revolBal'] + 0.1)
        df['openAcc_totalAcc'] = df['openAcc'] / df['totalAcc']
        df['dti'] = np.abs(df['dti'].fillna(1000))
        df['loanAmnt_dti_annualIncome'] = df['loanAmnt'] / (np.abs(df['dti']) * df['annualIncome'] + 0.1)
        df['employmentLength_bin'] = df['employmentLength']
        df['issueDate_bin'] = df['issueDate']
        df['earliesCreditLine_bin'] = df['earliesCreditLine']
        df['term_bin'] = df['term']
        df['homeOwnership_bin'] = df['homeOwnership']
        df['annualIncome_loanAmnt'] = df['annualIncome'] / (df['loanAmnt'] + 0.1)
        df['revolBal_loanAmnt'] = df['revolBal'] / (df['loanAmnt'] + 0.1)
        df['revolBal_installment'] = df['revolBal'] / (df['installment'] + 0.1)
        df['annualIncome_installment'] = df['annualIncome'] / (df['installment'] + 0.1)
    concated_df = pd.concat(dfs)
    label_lst = []
    # 把分箱后的特征做为类别特征处理
    bin_number = 10
    for i in range(bin_number):
        label_lst.append(i)
    dfs[0]['annualIncome_bin'] = pd.qcut(concated_df['annualIncome'], bin_number, labels=label_lst, duplicates='drop')[
                                 :dfs[0].shape[0]]
    dfs[0]['loanAmnt_bin'] = pd.qcut(concated_df['loanAmnt'], bin_number, labels=label_lst, duplicates='drop')[
                             :dfs[0].shape[0]]
    dfs[1]['annualIncome_bin'] = pd.qcut(concated_df['annualIncome'], bin_number, labels=label_lst, duplicates='drop')[
                                 dfs[0].shape[0]:]
    dfs[1]['loanAmnt_bin'] = pd.qcut(concated_df['loanAmnt'], bin_number, labels=label_lst, duplicates='drop')[
                             dfs[0].shape[0]:]

    label_lst = []
    bin_number = 100
    for i in range(bin_number):
        label_lst.append(i)
    dfs[0]['interestRate_bin'] = pd.qcut(concated_df['revolBal'], bin_number, labels=label_lst, duplicates='drop')[
                                 :dfs[0].shape[0]]
    dfs[0]['dti_bin'] = pd.qcut(concated_df['dti'], bin_number, labels=label_lst, duplicates='drop')[:dfs[0].shape[0]]
    dfs[0]['installment_bin'] = pd.qcut(concated_df['installment'], bin_number, labels=label_lst, duplicates='drop')[
                                :dfs[0].shape[0]]
    dfs[0]['revolBal_bin'] = pd.qcut(concated_df['revolBal'], bin_number, labels=label_lst, duplicates='drop')[
                             :dfs[0].shape[0]]
    dfs[0]['revolUtil_bin'] = pd.qcut(concated_df['revolUtil'], bin_number, labels=label_lst, duplicates='drop')[
                              :dfs[0].shape[0]]

    dfs[1]['interestRate_bin'] = pd.qcut(concated_df['revolBal'], bin_number, labels=label_lst, duplicates='drop')[
                                 dfs[0].shape[0]:]
    dfs[1]['dti_bin'] = pd.qcut(concated_df['dti'], bin_number, labels=label_lst, duplicates='drop')[dfs[0].shape[0]:]
    dfs[1]['installment_bin'] = pd.qcut(concated_df['installment'], bin_number, labels=label_lst, duplicates='drop')[
                                dfs[0].shape[0]:]
    dfs[1]['revolBal_bin'] = pd.qcut(concated_df['revolBal'], bin_number, labels=label_lst, duplicates='drop')[
                             dfs[0].shape[0]:]
    dfs[1]['revolUtil_bin'] = pd.qcut(concated_df['revolUtil'], bin_number, labels=label_lst, duplicates='drop')[
                              dfs[0].shape[0]:]

    for df in dfs:
        for cate in cate_features:
            df[cate] = df[cate].fillna(0).astype('int')
    issueDate_lst = list(set(concated_df['issueDate']))
    ratio_feat_lst = ['loanAmnt', 'installment', 'interestRate', 'annualIncome', 'dti', 'openAcc', \
                      'revolBal', 'revolUtil', 'totalAcc']
    issueDate_lst = list(set(concated_df['issueDate']))
    employmentLength_lst = list(set(concated_df['employmentLength']))
    purpose_lst = list(set(concated_df['purpose']))
    homeOwnership_lst = list(set(concated_df['homeOwnership']))
    for feat in ratio_feat_lst:
        issueDate_median = {}
        issueDate_item_rank = {}
        issueDate_label_mean = {}
        for dt in issueDate_lst:
            # 取最近6个月
            mask = (concated_df['issueDate'] >= dt - 3) & (concated_df['issueDate'] <= dt + 3)
            # 取最近6个月除去当前月份
            mask_1 = (concated_df['issueDate'] >= dt - 3) & (concated_df['issueDate'] <= dt + 3) & (
                    concated_df['issueDate'] != dt)
            item_series = concated_df.loc[mask, feat]
            label_series = concated_df.loc[mask_1, 'isDefault']
            # 取最近6个月的中位数
            issueDate_median[dt] = item_series.median()
            issueDate_label_mean[dt] = label_series.mean()
            item_rank = item_series.rank() / len(item_series)
            issueDate_item_rank[dt] = {}
            for item, rank in zip(item_series, item_rank):
                issueDate_item_rank[dt][item] = rank
        employmentLength_median = {}
        for et in employmentLength_lst:
            mask = concated_df['employmentLength'] == et
            item_series = concated_df.loc[mask, feat]
            employmentLength_median[et] = item_series.median()
        purpose_median = {}
        for pp in purpose_lst:
            mask = concated_df['purpose'] == pp
            item_series = concated_df.loc[mask, feat]
            purpose_median[pp] = item_series.median()
        homeOwnership_median = {}
        for ho in homeOwnership_lst:
            mask = concated_df['homeOwnership'] == ho
            item_series = concated_df.loc[mask, feat]
            homeOwnership_median[ho] = item_series.median()
        for df in dfs:
            print(feat, df.shape)
            df['label_issueDate_mean'] = df['issueDate'].apply(lambda x: issueDate_label_mean[x])
            df[feat + '_issueDate_median'] = df['issueDate'].apply(lambda x: issueDate_median[x])
            df[feat + '_issueDate_ratio'] = df.fillna(0).apply(lambda r: issueDate_item_rank[r['issueDate']][r[feat]],
                                                               axis=1)
            df[feat + '_employmentLength_ratio'] = df.fillna(0).apply(
                lambda r: r[feat] / employmentLength_median[r['employmentLength']], axis=1)
            df[feat + '_purpose_ratio'] = df.fillna(0).apply(lambda r: r[feat] / purpose_median[r['purpose']], axis=1)
            df[feat + '_homeOwnership_ratio'] = df.fillna(0).apply(
                lambda r: r[feat] / homeOwnership_median[r['homeOwnership']], axis=1)
            print(feat, df.shape)
    return dfs[0], dfs[1]
train_data = pd.read_csv('data/dk/train.csv')[:1000]
test_data = pd.read_csv('data/dk/testA.csv')[:100]

process([train_data, test_data])

feat_lst = list(test_data.columns[1:])

# 删除PSI高的特征
# installment_homeOwnership_ratio        2.633922e-01
# installment_purpose_ratio              2.703396e-01
# revolBal_issueDate_ratio               2.992158e-01
# revolBal_loanAmnt                      3.035800e-01
# annualIncome_installment               3.122066e-01
# installment_issueDate_ratio            3.255245e-01
# installment_employmentLength_ratio     3.367582e-01
# revolUtil_issueDate_ratio              3.394691e-01
# revolBal_purpose_ratio                 3.754704e-01
# revolBal_homeOwnership_ratio           3.816513e-01
# revolBal_employmentLength_ratio        4.892058e-01
# dti_issueDate_ratio                    5.064817e-01

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

# 缺失值预测补全

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model

lgb_params = {'objective': 'regression', 'metric': 'mae', 'lambda_l1': 0,
              'lambda_l2': 0.02,
              'num_leaves': 80,
              'feature_fraction': 0.5,
              'bagging_fraction': 1,
              'bagging_freq': 3,
              'min_child_samples': 50,
              'learning_rate': 0.1, 'n_estimators': 2000}

n_feat_lst = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']

full_df = pd.concat([train_data, test_data])

for col in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']:
    x = full_df[feat_lst].drop(n_feat_lst, axis=1).values
    y = full_df[col].values
    X_train, X_test, y_train, y_test = train_test_split(x[full_df[col].isnull() == False, :],
                                                        y[full_df[col].isnull() == False],
                                                        test_size=0.1, random_state=42)

    print(col, full_df[feat_lst].shape, X_train.shape, X_test.shape, y_train.shape)
    # reg = MLPRegressor(verbose=True)
    reg = LGBMRegressor(**lgb_params)
    # reg = linear_model.Lasso()
    reg.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="mae", eval_set=[(X_test, y_test)])
    col_new = col + '_new'
    print(col_new)
    for data_df in [train_data, test_data]:
        print(data_df.shape, np.sum(data_df[col].isnull()))
        if np.sum(data_df[col].isnull()) == 0:
            continue
        data_df[col_new] = data_df[col]
        x = data_df[feat_lst].drop(n_feat_lst, axis=1).values
        y_test_pred = reg.predict(x[data_df[col_new].isnull(), :])
        data_df.loc[data_df[col_new].isnull(), col_new] = y_test_pred
        print(y_test_pred)
        gc.collect()
for col in n_feat_lst:
    col_new = col + '_new'
    feat_lst.remove(col)
    feat_lst.append(col_new)

full_model = CatBoostClassifier(iterations=3000, depth=6, cat_features=cate_features, learning_rate=0.06,
                                loss_function='CrossEntropy',
                                thread_count=8, logging_level='Verbose', l2_leaf_reg=5)
test_data['isDefault'] = 0
full_model.fit(train_data[feat_lst], train_data['isDefault'], plot=False)
preds = full_model.predict_proba(test_data[feat_lst])
test_data['isDefault'] = preds[:, 1]
test_data[['id', 'isDefault']].to_csv('submit.csv', index=False)
