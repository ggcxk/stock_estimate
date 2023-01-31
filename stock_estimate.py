import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded,dtype='float64')
        for new_column in excluded:
            model = sm.OLS(y, pd.DataFrame(X[included+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.index[new_pval.argmin()]
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, pd.DataFrame(X[included])).fit()
        pvalues = model.pvalues
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.index[pvalues.argmax()]
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# 导入并处理数据
path = '.\\close_price.xlsx'
data = pd.read_excel(path)
data = data.set_index('date')
data = data.fillna(method='ffill')   # 填充缺失值
y = data.iloc[:,0]
X = data.iloc[:,1:]

# 通过逐步回归进行变量选择
selected_feature = stepwise_selection(X, y)
print('selected features:')
print(selected_feature)

# 按照选出的变量进行回归
best_X = data.loc[:,selected_feature]
model = sm.OLS(y, best_X).fit()
res = model.summary()
print(res)

# 绘图
plt.xticks(rotation=45)
plt.plot(y)
pred_y = model.predict()
plt.plot(pd.DataFrame(pred_y,index=y.index))
plt.legend(['real','predict'])
plt.show()