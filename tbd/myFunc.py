import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import time
from time import process_time


def fea_churn_plot(df, fea, index):
    '''
    signature:   fea_churn_plot(df=dataframe, fea=array/list, index=tuple).
    docstring:   plot numerical columns by churn histogram.
    parameters:  take in a dataframe, a list of column features and slicing indexes as tuple.
    returns:     a plt plot.
    '''
    # plot column features by churn histogram
    with plt.style.context('Solarize_Light2'):
        fig, axes = plt.subplots(nrows=1, ncols=index[1]-index[0], figsize=(16,5))
        for xcol, ax in zip(fea[index[0]:index[1]], axes):
            df[df['churn'] == 0][xcol].plot(kind='hist', alpha=0.7,
                                                                    ax=ax, color='#7FAFCE')
            df[df['churn'] == 1][xcol].plot(kind='hist', alpha=0.7,
                                                                    ax=ax, color='#F9C764')
            ax.set_xlabel(xcol)
            ax.set_title(f'Churn by {xcol}', size=10)
            
            
def fit_model(clf, X_train, y_train, X_test, y_test):
    '''
    signature:   eva_base_model(clf=estimator, X_train=X_train, y_trainn=y_train, X_test=X_test, y_test=y_test).
    docstring:   evaluate logistic regression model.
    parameters:  take in a classifier, train and test data sets.
    returns:     classification report and confusion matrix plot.
    '''
    
    # fit the model
    clf = clf
    clf.fit(X_train, y_train)
    
    # predict y
    #y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    # create classification report
    clf_rpt = pd.DataFrame(classification_report(y_test, y_pred_test,
                                    target_names=['Not Churn', 'Churn'], output_dict=True)).T
    
    return clf_rpt


def clf_rpt_by_label(df_clf_rpt, clf_dict, col_names, churn_key):
    '''
    signature:   clf_rpt_by_label(df_clf_rpt=dataframe, clf_dic=dictionary, col_names=array/list of string,                        churn_key=label indice in dataframe.
    docstring:   get precison, recall, f1_score and accuracy values from multi level dataframe and
                 store them in single level dataframe for ease of ploting.
    parameters:  take in a classifincation report in dictionary format, a list of level 2 columns
                 names and row indices.
    return:      pandas dataframe.
    '''
    prec_lst = {}
    for key in clf_dict.keys():
        tmp = []
        for name in col_names:
            tmp.append(df_clf_rpt[(key,name)][churn_key])
        # append accuracy score
        tmp.append(df_clf_rpt[(key,name)][2])
        prec_lst[key] = tmp
    
    return pd.DataFrame(prec_lst, index=['precision', 'recall', 'f1-score', 'accuracy'])


def plot_clf_rpt(dfs, titles=['Not Churn', 'Churn']):
    '''
    signature:     plot_clf_rpt(dfs=list of dataframe, titles=list of plot titles).
    doctring:      plot classification report scores by models by labels.
    parameters:    take in classification report scores stored in dataframe and list of titles.
    return:        matplotlib bar plot.
    '''
    # Comparision scores plot for the models
    with plt.style.context('seaborn-paper'):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9))
        for df, ax, title in zip(dfs, axes, titles):
            df.plot(kind='bar', ax=ax, alpha=.7)
            ax.set_title(f'{title} Classification Scores by Classifiers')
            ax.set_xlabel('Score Names')
            ax.set_ylabel('Score Values')
            ax.grid(True, axis='y')
            ax.legend(loc='lower right')
        
        
def plot_conf_mat(classifiers, x_test, y_test):
    '''
    signature:     plot_conf_mat(classifiers=array/list of classifiers, x_test=x test dataset, y_test=y test                          datatset).
    doctring:      plot the confusion matrix.
    parameters:    take in a list of classifiers, test dataset.
    return:        confusion matrix figure.
    '''
    # Plot confusion matrix for the 4 classifiers
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,9))
    for clf, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(clf, x_test, y_test, cmap='Blues', normalize='true', ax=ax)
        ax.title.set_text(f"Normalized Confusion Matrix - {type(clf).__name__}")
        plt.subplots_adjust(bottom=0.1, top=0.9)

        
def fit_tuned_model(alg_dict, params, X_data, y_data):
    '''
    signature:     fit_tuned_model(alg_dict=dictionary obj, params=GridSearch Params
                   , X_data=X train smote, y_data=Y train smote).
    docstring:     run GridSearchCV to find the best combination of parameters.      
    parameters:    take in a classifier objects stored in dictionary list, the grid params for
                   GridSearchCV, X training data and y training data.
    return:        A pandas dataframe   
    '''
    # start timer
    start_time = process_time()
    results = []
    
    for (alg_name, alg), (p_name, param) in zip(alg_dict.items(), params.items()):
        # calculate total of number of models per classification
        no_mdl = 1
        for val in param.values():
            no_mdl *= len(val)
        
        # Perform GridSearch 
        alg_name = alg
        p_name = GridSearchCV(alg_name, param_grid=param, cv=3, scoring='roc_auc')
        p_name.fit(X_data, y_data)
        best_param = p_name.best_params_
        best_score = p_name.best_score_
        best_est = p_name.best_estimator_
        
        # compute total exec time
        end_time = process_time()
        total_time = time.gmtime(end_time - start_time)
        exec_time = time.strftime("%H:%M:%S",total_time)
        
        # append results to list
        results.append([no_mdl, exec_time, best_param, best_est, best_score])
                
    return pd.DataFrame(results, columns=['no_model', 'exec_time', 'best_params', 'best_est', 'best_score'],
                        index=alg_dict.keys())

