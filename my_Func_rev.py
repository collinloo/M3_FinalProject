import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_auc_score
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import time
from time import process_time

def fea_churn_plot(df, fea, index):
    '''
    signature:   fea_churn_plot(df=dataframe, fea=array/list, index=tuple).
    docstring:   plot numerical columns by churn histogram.
    parameters:  take in a dataframe, a list of column features and a tuple for
                 calculating subplots ncols value.
    returns:     a plt plot.
    '''
    # plot column features by churn histogram
    with plt.style.context('Solarize_Light2'):
        fig, axes = plt.subplots(nrows=1, ncols=index[1]-index[0], figsize=(16,5))
        for xcol, ax in zip(fea[index[0]:index[1]], axes):
            df[df['churn'] == 0][xcol].plot(kind='hist', alpha=0.7,
                                            ax=ax, color='#7FAFCE',
                                            label='not churn',
                                            density=True
                                           )
            df[df['churn'] == 1][xcol].plot(kind='hist', alpha=0.7,
                                            ax=ax, color='#F9C764',
                                            label='churn',
                                            density=True
                                           )
            ax.legend()
            ax.set_xlabel(xcol)
            ax.set_title(f'Churn by {xcol}', size=14)
            
            
def pd_ohe(df):
    '''
    signature:     pd_ohe(df=X_data)
    docstring:     perform one hot encoding using pandas get_dummies
    parameters:    X data set
    returns:       dataframe 
    '''
    # create category list
    cat_col = df.select_dtypes(object).columns
    
    # perform OHE and concat dataframes
    dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
    x_ohe = df.drop(cat_col, axis=1)
    x_ohe = pd.concat([dummies, x_ohe], axis=1)
    
    return x_ohe


def fit_eval_clf(clfs, df_list, disp_cm=True):
    '''
    signature:     fit_eval_clf(clfs=array/list, X_train=dataframe, y_train=dataframe
                   X_test=dataframe, y_test=dataframe, disp_cm=bool)
    docstring:     run each classifier in the list through the pipeline
    parameters:    takes in a list of classifier, X and y training and testing dataset,
                   option to output the confusion matrix plot
    return:        classification report data in dataframe and disctionary format.
    '''
                                               
    # assign dfs
    X_train, y_train, X_test, y_test = df_list
                                        
    # declare dict obj to store classification report
    clf_rpt_dict = {}
    # declare list to store fit classifiers results
    classifiers = []
    # fit model via pipeline
    for clf in clfs:
        clf_pipe =  Pipeline([('scaler', StandardScaler()),
                      (type(clf).__name__, clf)
                     ])
        clf_pipe.fit(X_train, y_train)
        y_hat_test = clf_pipe.predict(X_test)
        # convert classification report to dictionary
        clf_rpt_dict[type(clf).__name__] = pd.DataFrame(classification_report(y_test, y_hat_test,
                                    target_names=['Not Churn', 'Churn'], output_dict=True)).T
        classifiers.append(clf_pipe)
        
    # plot confusion matrix
    if disp_cm == True:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,9))
        for class_, name, ax in zip(classifiers, clfs, axes.flatten()):
            plot_confusion_matrix(class_, X_test, y_test, cmap='Blues', normalize='true', ax=ax)
            ax.title.set_text(f"Normalized Confusion Matrix - {type(name).__name__}")
            plt.subplots_adjust(bottom=0.1, top=0.9)
        plt.show()
        return pd.concat(clf_rpt_dict.values(), axis=1, keys=clf_rpt_dict.keys()), clf_rpt_dict
    else:
        return pd.concat(clf_rpt_dict.values(), axis=1, keys=clf_rpt_dict.keys()), clf_rpt_dict
         

def plot_clf_rpt(df_clf_rpt, clf_dict):
    '''
    signature:     plot_clf_rpt(df_clf_rpt=dataframe, clf_dict=dictionary).
    doctring:      plot classification report scores by models by labels.
    parameters:    take in the outputs from fit_eval_clf(), a classification report data in
                   dataframe and disctionary format.
    return:        a bar plot.
    '''
    
    # define new dataframe column names
    col_names = ['precision', 'recall', 'f1-score']
    churn_dict = {'not_churn': 0, 'churn':1}
    dfs = []
    
    # separate classification report by 'Not Churn' & 'Churn'
    for k, v in churn_dict.items():
        scores_lst = {}
        # loop thru' list of classifiers
        for key in clf_dict.keys():
            tmp = []
            # loop thru level 1 column name
            for name in col_names:
                tmp.append(df_clf_rpt[(key,name)][v])
        
            # append accuracy score
            tmp.append(df_clf_rpt[(key,name)][2])
            scores_lst[key] = tmp
        dfs.append(pd.DataFrame(scores_lst, index=['precision', 'recall', 'f1-score', 'accuracy']))
     
    # comparision scores plot for the models
    with plt.style.context('seaborn-paper'):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,16))
        for df, ax, title in zip(dfs, axes, churn_dict.keys()):
            df.plot(kind='barh', ax=ax, alpha=.7)
            ax.set_title(f'{title} Classification Scores by Classifiers', size=24)
            ax.tick_params(axis='x', labelsize=14, )
            ax.tick_params(axis='y', labelsize=14)
            ax.set_xlabel('Score Value', fontsize=18)
            ax.set_ylabel('Score Name', fontsize=18)
            ax.grid(True, axis='x')
            ax.legend(loc='upper left', fontsize=14)


def plot_gain_loss(new_rpt, base_rpt, columns, comp):
    '''
    signature:     plot_gain_loss(new_rpt=df, base_rpt=df, columns=dict)
    doctring:      compute differences between two dataframe that contain the classification
                   report scores.
    parameters:    takes in two classification reports in df format and a dictionary that contains
                   classification labels.
    return:        plt plots
    '''
    dfs = []
    # separate churn and not churn into two dataframes
    for colname, colvalue in columns.items():
        scores_diff = (new_rpt - base_rpt).T.unstack()
        colname = scores_diff.loc[:, colvalue].copy()
        colname.rename(columns={'support': 'accuracy'}, inplace=True)
        # get the classifier accuracy store
        ser = (new_rpt - base_rpt).loc['accuracy'].unstack().support.values
        colname['accuracy'] = [x for x in ser]
        dfs.append(colname.T)
      
    with plt.style.context('seaborn-paper'):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,16))
        for df, ax, title in zip(dfs, axes, columns.values()):
            df.plot(kind='barh', ax=ax, alpha=.7)
            ax.set_title(f'{title} Performance Gain/Loss - {comp}', size=24)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_xlabel('Score Value', fontsize=18)
            ax.set_ylabel('Score Name', fontsize=18)
            ax.grid(True, axis='x')
            ax.legend(loc='upper left', fontsize=14)              
            

def fit_evl_grdsch(alg_dict, params, df_list):
# def fit_evl_grdsch(alg_dict, params, X_data, y_data):                                        
    '''
    signature:     fit_tuned_model(alg_dict=dictionary, params=array/list
                   , X_data=df, y_data=df).
    docstring:     run GridSearchCV to find the best combination of parameters.      
    parameters:    take in a classifier objects stored in dictionary list, the grid params for
                   GridSearchCV, X training data and y training data.
    return:        A pandas dataframe   
    '''
                   
    # assign dfs
    X_train, y_train, X_test, y_test = [df for df in df_list]
    results = []
    
    for (alg_name, alg), (p_name, param) in zip(alg_dict.items(), params.items()):
        # start timer
        start_time = process_time()
              
        # calculate total of number of models per classification
        no_mdl = 1
        for val in param.values():
            no_mdl *= len(val)
        
        # Perform GridSearch 
        print(f'running gridsearch for {alg_name}')
        clf_pipe =  Pipeline([('scaler', StandardScaler()),
                      (alg_name, alg)
                     ])
        p_name = GridSearchCV(clf_pipe, param_grid=param, cv=5, scoring='roc_auc')
        p_name.fit(X_train, y_train)
        best_param = p_name.best_params_
        best_score = p_name.best_score_
        best_est = p_name.best_estimator_
        
        # compute total exec time
        end_time = process_time()
        total_time = time.gmtime(end_time - start_time)
        exec_time = time.strftime("%H:%M:%S",total_time)
        print(f'total elapsed time {exec_time}')
        
        # append results to list
        results.append([no_mdl, exec_time, best_param, best_est, best_score])
                
    return pd.DataFrame(results, columns=['no_model', 'exec_time', 'best_params', 'best_est', 'best_score'],
                        index=alg_dict.keys())                
            
            
def plot_roc_cur(y_test, y_test_pred, sec_y_test=None, sec_y_pred=None, sec=None):
    '''
    signature:     plot_roc_cur(y_test=pd.series, y_test_pred=array, sec_y_test=pd.series,
                   sec_y_pred=array, sec=bool)
    docstring:     calculate AUC score and plot the ROC curve
    parameters:    take in actual and predicted y values or/and second set of actual and predicted y values  
    return:        print message of AUC score(s) and a plt plot
    '''
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    if sec:
        fpr_s, tpr_s, th_s = roc_curve(sec_y_test, sec_y_pred)
        print(f'\n***** Model A AUC Score: {round(roc_auc_score(sec_y_test, sec_y_pred)*100,2)}% *****')
        print(f'\n***** Model B AUC Score: {round(roc_auc_score(y_test, y_test_pred)*100,2)}% *****')
    else: print(f'\n***** AUC Score: {round(roc_auc_score(y_test, y_test_pred)*100,2)}% *****')

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9,6))
        if sec:
            plt.plot(fpr_s, tpr_s, label="Model A ROC curve", color='#04B404')
            plt.plot(fpr, tpr, label="Model B ROC curve", color='#C87F21')
        else:
            plt.plot(fpr, tpr, label="ROC curve", color='#C87F21')
        plt.plot([0,1],[0,1], color='#3380B2', linestyle='--')
        plt.xlim([0.0, 1.0], )
        plt.ylim([0.0, 1.05])
        plt.yticks([i/20.0 for i in range(21)])
        plt.xticks([i/20.0 for i in range(21)])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) Curve')
        plt.legend()
        plt.show()           
            
            
def mdl_validation(clf_name, clf, df_list):
# def mdl_validation(clf_name, clf, X_train, y_train, X_test, y_test):                                        
    '''
    signature:     mdl_validation(clf_name=sring, clf=classifier object, X_train=df, y_train=df,
                   X_test=df, y_test=df)
    docstring      set up pipeline and fit model to calculate y predicted values and the
                   accuracy score.
    parameters:    take in name of classifier, the classifier object, X and y train, test dataset
    return:        y test predicted value, train and test accuracy scores
    '''
    
    # assign dfs
    X_train, y_train, X_test, y_test = [df for df in df_list]                                       
                                        
    # set up pipe
    valid_pipe = Pipeline([('scaler', StandardScaler()),
                           (clf_name, clf)])
    
    # fit the model
    valid_pipe.fit(X_train, y_train)
    
    # predict
    y_train_pred = valid_pipe.predict(X_train)
    y_test_pred = valid_pipe.predict(X_test)
    y_train_acc = accuracy_score(y_train, y_train_pred)
    y_test_acc = accuracy_score(y_test, y_test_pred)
    
    return y_test_pred, round(y_train_acc,4), round(y_test_acc, 4)
            
            
# *** Not Used ***            
# def exec_ohe(x_train, x_test):
#     # get categorical column names
#     cat_names = x_train.select_dtypes(object).columns

#     # separate numerical columns
#     x_train_num = x_train.select_dtypes('number').reset_index(drop=True)
#     x_test_num = x_test.select_dtypes('number').reset_index(drop=True)
#     # instantiate OHE object
#     enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
#     # perform OHE
#     train_tmp = enc.fit_transform(x_train[cat_names])
#     test_tmp = enc.transform(x_test[cat_names])
    
#     # OHE column names
#     ohe_cat_names = enc.get_feature_names(cat_names)
        
#     # transform array to dataframe
#     X_train_ohe = pd.DataFrame(train_tmp, columns=[*ohe_cat_names])
#     X_test_ohe = pd.DataFrame(test_tmp, columns=[*ohe_cat_names])
    
#     # combine with input dataframes on numerical columns
#     X_train_ohe = pd.concat([X_train_ohe, x_train_num], axis=1)
#     X_test_ohe = pd.concat([X_test_ohe, x_test_num], axis=1)
    
#     return X_train_ohe, X_test_ohe      


# def eval_smote_ratio(df_list, clfs):
#     '''
#     signature:     eval_smote_size(df_list=list, clfs=list)
#     docstring:     loop thru' the ratio list to find the best AUC from fitting the classifiers
#     parameters:    take in list of trainning and testing dataframe and list of classifiers
#     return:        a dictionary that stores the best ratio for each classifier
#     '''
#     # list of ratios to test
#     ratios = [0.25, 0.33, 0.5, 0.7, 1]
#     names = ['0.25', '0.33','0.5','0.7','1']
#     score_dict = {}
    
#     # unpack df in list
#     X_train, y_train, X_test, y_test = df_list
    
#     for n, ratio in enumerate(ratios):
#         smote = SMOTE(sampling_strategy=ratio, random_state=36)
#         X_train_resamp, y_train_resamp = smote.fit_sample(X_train, y_train)
#         #fit a model
#         temp = []
#         # loop thru' list of classifier and calculate auc score
#         for clf in clfs:
#             clf_name = type(clf).__name__ 
#             clf_name = clf.fit(X_train_resamp, y_train_resamp)
#             y_hat_test = clf.predict(X_test)
#             fpr, tpr, thresholds = roc_curve(y_test, y_hat_test)
#             auc = roc_auc_score(y_test, y_hat_test)
#             # append auc at ratio value
#             temp.append(auc)
#         # update dict
#         score_dict[names[n]] = temp
   
#     # convert score dict to df
#     pd_col = [type(x).__name__ for x in clfs]
#     df_auc = pd.DataFrame.from_dict(score_dict).T
#     df_auc.columns = pd_col
    
#     # get the max auc
#     best_ratio = {}
#     for name in df_auc.columns:
#         max_v = df_auc[name].max()
#         best_ratio[name] = float(df_auc.index[df_auc[name] == max_v].values[0])
        
#     return best_ratio
            

# def fit_eval_clf_smote(clfs, df_list, best_ratio, disp_cm=True):
# # def fit_eval_clf(clfs, X_train, y_train, X_test, y_test, disp_cm=True):
#     '''
#     signature:     fit_eval_clf(clfs=array/list, df_list=array/list, best_ratio=dictionary, disp_cm=bool)
#     docstring:     run SMOTE with best ratio then run each classifier in the list through the pipeline
#     parameters:    takes in a list of classifier, X and y training and testing dataset,
#                    option to output the confusion matrix plot
#     return:        classification report data in dataframe and disctionary format and y_pred from each clf.
#     '''
                                               
#     # assign dfs
#     b_X_train, b_y_train, X_test, y_test = df_list
                                        
#     # declare dict obj to store classification report
#     clf_rpt_dict = {}
#     # declare list to store fit classifiers results
#     classifiers = []
#     # declare list to store y_pred
#     y_pred_lst = []
#     # fit model via pipeline
#     for clf, ratio in zip(clfs, best_ratio.values()):
#         smote = SMOTE(random_state=36, sampling_strategy=ratio)
#         X_train, y_train = smote.fit_sample(b_X_train, b_y_train)
#         clf_pipe =  Pipeline([('scaler', StandardScaler()),
#                       (type(clf).__name__, clf)
#                      ])
#         clf_pipe.fit(X_train, y_train)
#         # predict y
#         y_hat_test = clf_pipe.predict(X_test)
#         # append pred y
#         y_pred_lst.append((y_test, y_hat_test))
#         # convert classification report to dictionary
#         clf_rpt_dict[type(clf).__name__] = pd.DataFrame(classification_report(y_test, y_hat_test,
#                                     target_names=['Not Churn', 'Churn'], output_dict=True)).T
#         classifiers.append(clf_pipe)
        
#     # plot confusion matrix
#     if disp_cm == True:
#         fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,9))
#         for class_, name, ax in zip(classifiers, clfs, axes.flatten()):
#             plot_confusion_matrix(class_, X_test, y_test, cmap='Blues', normalize='true', ax=ax)
#             ax.title.set_text(f"Normalized Confusion Matrix - {type(name).__name__}")
#             plt.subplots_adjust(bottom=0.1, top=0.9)
#         plt.show()
#         return pd.concat(clf_rpt_dict.values(), axis=1, keys=clf_rpt_dict.keys()), clf_rpt_dict, y_pred_lst
#     else:
#         return pd.concat(clf_rpt_dict.values(), axis=1, keys=clf_rpt_dict.keys()), clf_rpt_dict, y_pred_lst    
    
    
# def plot_clf_rpt(df_clf_rpt, clf_dict):
#     '''
#     signature:     plot_clf_rpt(df_clf_rpt=dataframe, clf_dict=dictionary).
#     doctring:      plot classification report scores by models by labels.
#     parameters:    take in the outputs from fit_eval_clf(), a classification report data in
#                    dataframe and disctionary format.
#     return:        a bar plot.
#     '''
    
#     # define new dataframe column names
#     col_names = ['precision', 'recall', 'f1-score']
#     churn_dict = {'not_churn': 0, 'churn':1}
#     dfs = []
    
#     # separate classification report by 'Not Churn' & 'Churn'
#     for k, v in churn_dict.items():
#         scores_lst = {}
#         # loop thru' list of classifiers
#         for key in clf_dict.keys():
#             tmp = []
#             # loop thru level 1 column name
#             for name in col_names:
#                 tmp.append(df_clf_rpt[(key,name)][v])
        
#             # append accuracy score
#             tmp.append(df_clf_rpt[(key,name)][2])
#             scores_lst[key] = tmp
#         dfs.append(pd.DataFrame(scores_lst, index=['precision', 'recall', 'f1-score', 'accuracy']))
     
#     # comparision scores plot for the models
#     with plt.style.context('seaborn-paper'):
#         fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9))
#         for df, ax, title in zip(dfs, axes, churn_dict.keys()):
#             df.plot(kind='bar', ax=ax, alpha=.7)
#             ax.set_title(f'{title} Classification Scores by Classifiers', size=14)
#             ax.set_xlabel('Score Names')
#             ax.set_ylabel('Score Values')
#             ax.grid(True, axis='y')
#             ax.legend(loc='lower right')  


# def plot_gain_loss(new_rpt, base_rpt, columns, comp):
#     '''
#     signature:     plot_gain_loss(new_rpt=df, base_rpt=df, columns=dict)
#     doctring:      compute differences between two dataframe that contain the classification
#                    report scores.
#     parameters:    takes in two classification reports in df format and a dictionary that contains
#                    classification labels.
#     return:        plt plots
#     '''
#     dfs = []
#     # separate churn and not churn into two dataframes
#     for colname, colvalue in columns.items():
#         scores_diff = (new_rpt - base_rpt).T.unstack()
#         colname = scores_diff.loc[:, colvalue].copy()
#         colname.rename(columns={'support': 'accuracy'}, inplace=True)
#         # get the classifier accuracy store
#         ser = (new_rpt - base_rpt).loc['accuracy'].unstack().support.values
#         colname['accuracy'] = [x for x in ser]
#         dfs.append(colname.T)
      
#     with plt.style.context('seaborn-paper'):
#         fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9))
#         for df, ax, title in zip(dfs, axes, columns.values()):
#             df.plot(kind='bar', ax=ax, alpha=.7)
#             ax.set_title(f'{title} Performance Gain/Loss - {comp}', size=14)
#             ax.set_xlabel('Score Names')
#             ax.set_ylabel('Score Values')
#             ax.grid(True, axis='y')
#             ax.legend(loc='lower right')  


# def fit_evl_grdsch(alg_dict, params, df_list, best_ratio):
#     '''
#     signature:     fit_tuned_model(alg_dict=dictionary, params=array/list
#                    , best_eastimator=dictionary).
#     docstring:     run GridSearchCV to find the best combination of parameters.      
#     parameters:    take in a classifier objects stored in dictionary list, the grid params for
#                    GridSearchCV, train & test df in list and smote ratio in dict form.
#     return:        A pandas dataframe   
#     '''
                   
#     # assign dfs
#     b_X_train, b_y_train, X_test, y_test = [df for df in df_list]
#     results = []
    
#     # create scorer
#     recall_scorer = make_scorer(recall_score)
    
#     for (alg_name, alg), (p_name, param), ratio in zip(alg_dict.items(), params.items(), best_ratio.values()):
#         # start timer
#         start_time = process_time()
              
#         # calculate total of number of models per classification
#         no_mdl = 1
#         for val in param.values():
#             no_mdl *= len(val)
        
#         # Perform GridSearch 
#         print(f'running gridsearch for {alg_name}')
#         smote = SMOTE(random_state=36, sampling_strategy=ratio)
#         X_train, y_train = smote.fit_sample(b_X_train, b_y_train)
#         clf_pipe =  Pipeline([('scaler', StandardScaler()),
#                       (alg_name, alg)
#                      ])
#         # using roc_auc: 0.98:0.83
#         p_name = GridSearchCV(clf_pipe, param_grid=param, cv=5, scoring='roc_auc')
#         # using recall_scorer: 0.98:0.81
# #         p_name = GridSearchCV(clf_pipe, param_grid=param, cv=5, scoring=recall_scorer)
#         p_name.fit(X_train, y_train)
#         best_param = p_name.best_params_
#         best_score = p_name.best_score_
#         best_est = p_name.best_estimator_
        
#         # compute total exec time
#         end_time = process_time()
#         total_time = time.gmtime(end_time - start_time)
#         exec_time = time.strftime("%H:%M:%S",total_time)
#         print(f'total elapsed time {exec_time}')
        
#         # append results to list
#         results.append([no_mdl, exec_time, best_param, best_est, best_score])
                
#     return pd.DataFrame(results, columns=['no_model', 'exec_time', 'best_params', 'best_est', 'best_score'],
#                         index=alg_dict.keys())   
            

# def fit_evl_grdsch(alg_dict, params, df_list):
# # def fit_evl_grdsch(alg_dict, params, X_data, y_data):                                        
#     '''
#     signature:     fit_tuned_model(alg_dict=dictionary, params=array/list
#                    , X_data=df, y_data=df).
#     docstring:     run GridSearchCV to find the best combination of parameters.      
#     parameters:    take in a classifier objects stored in dictionary list, the grid params for
#                    GridSearchCV, X training data and y training data.
#     return:        A pandas dataframe   
#     '''
                   
#     # assign dfs
#     X_train, y_train, X_test, y_test = [df for df in df_list]
#     results = []
    
#     for (alg_name, alg), (p_name, param) in zip(alg_dict.items(), params.items()):
#         # start timer
#         start_time = process_time()
              
#         # calculate total of number of models per classification
#         no_mdl = 1
#         for val in param.values():
#             no_mdl *= len(val)
        
#         # Perform GridSearch 
#         print(f'running gridsearch for {alg_name}')
# #         alg_name = alg
#         clf_pipe =  Pipeline([('scaler', StandardScaler()),
#                       (alg_name, alg)
#                      ])
#         p_name = GridSearchCV(clf_pipe, param_grid=param, cv=5, scoring='roc_auc')
#         p_name.fit(X_train, y_train)
#         best_param = p_name.best_params_
#         best_score = p_name.best_score_
#         best_est = p_name.best_estimator_
        
#         # compute total exec time
#         end_time = process_time()
#         total_time = time.gmtime(end_time - start_time)
#         exec_time = time.strftime("%H:%M:%S",total_time)
#         print(f'total elapsed time {exec_time}')
        
#         # append results to list
#         results.append([no_mdl, exec_time, best_param, best_est, best_score])
                
#     return pd.DataFrame(results, columns=['no_model', 'exec_time', 'best_params', 'best_est', 'best_score'],
#                         index=alg_dict.keys())      


# def mdl_validation(clf_name, clf, df_list, best_ratio):
#     '''
#     signature:     mdl_validation(clf_name=sring, clf=classifier object, df_list=array/list,                                          best_ratio=float)
#     docstring      set up pipeline and fit model to calculate y predicted values and the
#                    accuracy score.
#     parameters:    take in name of classifier, the classifier object, X and y train, test dataset
#                    and ratio for smote
#     return:        y test predicted value, train and test accuracy scores
#     '''
    
#     # assign dfs
#     b_X_train, b_y_train, X_test, y_test = [df for df in df_list]                                       
    
#     # fit smote
#     smote = SMOTE(random_state=36, sampling_strategy=best_ratio)
#     X_train, y_train = smote.fit_sample(b_X_train, b_y_train)
#     # set up pipe
#     valid_pipe = Pipeline([('scaler', StandardScaler()),
#                            (clf_name, clf)])
    
#     # fit smote with best ratio
    
#     # fit the model
#     valid_pipe.fit(X_train, y_train)
    
#     # predict
#     y_train_pred = valid_pipe.predict(X_train)
#     y_test_pred = valid_pipe.predict(X_test)
#     y_train_acc = accuracy_score(y_train, y_train_pred)
#     y_test_acc = accuracy_score(y_test, y_test_pred)
    
#     return y_test_pred, round(y_train_acc,4), round(y_test_acc, 4)