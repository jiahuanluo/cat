import pandas as pd
X_train = pd.read_csv("data/dk/X_train_noised.csv")
X_validation = pd.read_csv("data/dk/X_validation_noised.csv")

print(X_train.shape)
print(X_validation.shape)

cate_features = ['employmentTitle', 'employmentLength_bin', 'purpose', 'postCode', 'subGrade', 'earliesCreditLine_bin',
                 'regionCode', 'title', 'issueDate_bin', 'term_bin', 'interestRate_bin', 'annualIncome_bin',
                 'loanAmnt_bin', 'homeOwnership_bin', 'revolBal_bin', 'dti_bin', 'installment_bin', 'revolBal_bin',
                 'revolUtil_bin']
# X_train = X_train[:1000]
train_m_s = pd.DataFrame(columns=cate_features)
for feat in cate_features:
    print(feat, len(X_train[feat].unique()))
    # numerial_feat = "numerial_" + feat
    # noised_feat = "noised_" + numerial_feat
    # a = X_train[[feat, numerial_feat, noised_feat]].groupby(by=feat).agg(['mean', 'std'])
    # b = X_validation[[feat, numerial_feat, noised_feat]].groupby(by=feat).agg(['mean', 'std'])
