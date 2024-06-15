import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from lifelines.statistics import logrank_test
import os
import pandas as pd
import numpy as np
import json

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import shap

def load_data(args):
    cli_df = pd.read_csv(os.path.join(args.dataset_dir, f"{args.data_name}.csv"))
    cli_df = cli_df.drop_duplicates("case_id").drop(["slide_id", "group"], axis=1).reset_index(drop=True)
    tests = pd.read_csv(os.path.join(args.split_dir, f"{args.data_name}/splits_0.csv"))["test"].dropna().values
    cli_df["split"] = [pd.NA] * len(cli_df)
    cli_df.loc[cli_df["case_id"].isin(tests), "split"] = "test"
    cli_df.loc[~cli_df["case_id"].isin(tests), "split"] = "train"

    if args.target_data != "cli":
        target_df = pd.read_csv(os.path.join(args.dataset_dir, f"{args.data_name}_{args.target_data}.csv.zip"), compression="zip")
        target_df = pd.merge(target_df, cli_df[["case_id", "split", "survival_months", "event"]], on="case_id")
    else:
        target_df = cli_df

    target_df.reset_index(drop=True, inplace=True)
    assert not target_df.columns.duplicated().any(), "Duplicated features are found."
    assert not target_df.columns.isna().any(), "NaN features are found."
    print(f"Loaded {args.target_data} data (shape: {target_df.shape}, missing values: {target_df.isna().any().any()})")
    
    return target_df

def split_data(df):
    train_df = df[df["split"] != "test"].drop(["split"], axis=1)
    test_df = df[df["split"] == "test"].drop(["split"], axis=1)
    train_df = train_df.drop_duplicates("case_id").reset_index(drop=True)
    test_df = test_df.drop_duplicates("case_id").reset_index(drop=True)

    X_train = train_df.drop(["case_id", "event", "survival_months"], axis=1)
    y_train_time = train_df["survival_months"]
    y_train_event = train_df["event"]

    X_test = test_df.drop(["case_id", "event", "survival_months"], axis=1)
    y_test = test_df["survival_months"]
    print(f"Train set shape: {X_train.shape}, test set shape: {X_test.shape}\n")
    print("Train survival times:")
    print(y_train_time.describe())
    print("\nTest survival times:")
    print(y_test.describe())
    return X_train, y_train_time, y_train_event

def fill_missing(X, fill="median"):
    indep_vars = X.columns
    if X.isna().any().any():
        medians = X.median() if fill=="median" else X.mean()
        print("Filling missing values with ", fill)
        for i, col in enumerate(indep_vars):
            if i % 1000 == 0:
                print(i, "/", len(indep_vars))
            if X[col].isna().any():
                X[col] = X[col].fillna(medians[col])
        print("Done!\n")
    return X

def var_filter(X, thresh=0.01):
    indep_vars = X.columns
    var_sel = VarianceThreshold(thresh)
    X = var_sel.fit_transform(X)
    X = pd.DataFrame(X, columns=indep_vars[var_sel.get_support()])
    print("\t**After variance threshold: ", X.shape)
    removed_cols = [col for col in indep_vars if col not in X.columns]
    return X, removed_cols

def norm(X):
    sc = StandardScaler()
    print("\nNormalization:")
    print("> Before min: {:.4f} | max: {:.4f}" .format(X.min().min(), X.max().max()))
    X_norm = sc.fit_transform(X)
    X_norm = pd.DataFrame(X_norm, columns=X.columns)
    print("> After min: {:.4f} | max: {:.4f}\n\n" .format(X_norm.min().min(), X_norm.max().max()))
    return X_norm

class MultiCollinearityEliminator:
    def __init__(self, df, target, threshold):
        self.df = df
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = pd.DataFrame(target)
        self.threshold = threshold

    def createCorrMatrix(self, include_target = False):
        if (include_target == True):
            df_target = pd.concat([self.df, self.target], axis=1)
            corrMatrix = df_target.corr(method='pearson', min_periods=30).abs()
        elif (include_target == False):
            corrMatrix = self.df.corr(method='pearson', min_periods=30).abs()
        return corrMatrix

    def createCorrMatrixWithTarget(self):
        corrMatrix = self.createCorrMatrix(include_target = True)
        corrWithTarget = pd.DataFrame(corrMatrix.loc[:,self.target.columns[0]]).drop([self.target.columns[0]], axis = 0).sort_values(by = self.target.columns[0])                    
        # print(corrWithTarget, '\n')
        return corrWithTarget

    
    def createCorrelatedFeaturesList(self):
        corrMatrix = self.createCorrMatrix(include_target = False)                          
        colCorr = []
        for column in corrMatrix.columns:
            for idx, row in corrMatrix.iterrows(): 
                if (row[column]>self.threshold) and (row[column]<1):
                    
                    if (idx not in colCorr):
                        colCorr.append(idx)
                        # print(idx, column, row[column], '\n')
                    if (column not in colCorr):
                        colCorr.append(column)
        # print(colCorr, '\n')
        return colCorr

    def deleteFeatures(self, colCorr):
        corrWithTarget = self.createCorrMatrixWithTarget()                                  
        for idx, row in corrWithTarget.iterrows():
            # print(idx, '\n')
            if (idx in colCorr):
                self.df = self.df.drop(idx, axis =1)
                break
        return self.df

    def autoEliminateMulticollinearity(self):
        colCorr = self.createCorrelatedFeaturesList()                                       
        while colCorr != []:
            self.df = self.deleteFeatures(colCorr)
            colCorr = self.createCorrelatedFeaturesList()                                     
        return self.df
    
def multicol_filter(X, y, thresh=0.7):
    # corr > 0.7 indicates multicollinearity
    # https://blog.clairvoyantsoft.com/correlation-and-collinearity-how-they-can-make-or-break-a-model-9135fbe6936a#:~:text=Multicollinearity%20is%20a%20situation%20where,indicates%20the%20presence%20of%20multicollinearity.
    
    df = pd.concat([X, y], axis=1)
    corr = df.corr().abs()
    indep_vars = X.columns

    if len(indep_vars) < 1000:
        cor_sel = MultiCollinearityEliminator(df=X, target=y, threshold=thresh)
        X = cor_sel.autoEliminateMulticollinearity()
        removed_cols = [i for i in indep_vars if i not in X.columns]
    else:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        removed_cols = [column for column in upper.columns if any(upper[column] > thresh)]
        X = X.drop(columns=removed_cols, axis=1)
    print("\t**After collinearity elimination: ", X.shape)
    return X, removed_cols

def logrank(X, y):
    indep_vars = X.columns
    data = pd.concat([X, y], axis=1)
    results_df = {}
    sign_features = []

    for feature in X.columns:
        if len(X[feature].unique()) > 2:
            data["group"] = data[feature].apply(lambda x: x > data[feature].median())
        else:
            data["group"] = data[feature].apply(lambda x: x > 0)
        
        T = data['survival_months']
        E = data['event']
        ix = data["group"]
        results = logrank_test(T[ix], T[~ix], event_observed_A=E[ix], event_observed_B=E[~ix])
        
        results_df[feature] = results.p_value
        if results.p_value < 0.05:
            sign_features.append(feature)
            
    results_df = pd.DataFrame.from_dict(results_df, orient='index', columns=['p-value'])

    removed_cols = [i for i in indep_vars if i not in sign_features]
    X = X[sign_features]
    print("\t**After log-rank test: ", X.shape)
    return X, results_df, removed_cols

def feature_importance(X, y):
    cph = CoxPHFitter(penalizer=.1)
    cph.fit(pd.concat([X, y], axis=1), 'survival_months', 'event')

    explainer = shap.Explainer(cph.predict_partial_hazard, X, max_evals=2 * X.shape[1] + 1)
    shap_values = explainer(X)
    feature_importance = np.abs(shap_values.values).mean(axis=0)

    return pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

def _cv_surv(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    c_index_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Combine features and target for CoxPHFitter
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Fit the Cox Proportional Hazards model
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_data, 'survival_months', 'event')
        
        # Predict partial hazard for test set
        partial_hazard = cph.predict_partial_hazard(test_data)
        
        # Calculate concordance index
        c_index = concordance_index(test_data['survival_months'], -partial_hazard, test_data['event'])
        c_index_scores.append(c_index)
    
    return np.mean(c_index_scores), np.std(c_index_scores)

def cross_validate_survival_model(X, y, feature_importance_df, save_path=None):
    results = []
    len_features = X.shape[1]
    print_every = 50 if len_features > 100 else 10

    max_value = 0
    counter = 0
    for i in range(1, len_features+1):
        selected_features = feature_importance_df.loc[:i, "feature"].values
        
        mean_c_index, std_c_index = _cv_surv(X[selected_features], y)
        results.append((i, mean_c_index, std_c_index))
        if mean_c_index > max_value:
            max_value = mean_c_index
            counter = 0
        else:
            counter += 1

        if i % print_every == 0:
            print(f'Number of features: {i} | Cross-validated C-index: {mean_c_index:.4f} ± {std_c_index:.4f}')
            if save_path is not None:
                pd.DataFrame(results, columns=["Number of features", "Mean C-index", "Std C-index"]).to_csv(save_path)

        if i > 300 and counter > 50:
            print("No improvement since {} for the last {} iterations. Early stopping..." .format(max_value, counter))
            break

    results_df = pd.DataFrame(results, columns=["Number of features", "Mean C-index", "Std C-index"])
    max_results = results_df[results_df["Mean C-index"] == results_df["Mean C-index"].max()].reset_index(drop=True)
    print("\nBest results with {} features: C-index {:.4f} ± {:.4f}" .format(max_results["Number of features"].iloc[0], max_results["Mean C-index"].iloc[0], max_results["Std C-index"].iloc[0]))

    return results_df
