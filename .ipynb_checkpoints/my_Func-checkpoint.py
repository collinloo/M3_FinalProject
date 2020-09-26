{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "import time\n",
    "from time import process_time\n",
    "\n",
    "def fea_churn_plot(df, fea, index):\n",
    "    '''\n",
    "    signature:   fea_churn_plot(df=dataframe, fea=array/list, index=tuple).\n",
    "    docstring:   plot numerical columns by churn histogram.\n",
    "    parameters:  take in a dataframe, a list of column features and slicing indexes as tuple.\n",
    "    returns:     a plt plot.\n",
    "    '''\n",
    "    # plot column features by churn histogram\n",
    "    with plt.style.context('Solarize_Light2'):\n",
    "        fig, axes = plt.subplots(nrows=1, ncols=index[1]-index[0], figsize=(16,5))\n",
    "        for xcol, ax in zip(fea[index[0]:index[1]], axes):\n",
    "            df[df['churn'] == 0][xcol].plot(kind='hist', alpha=0.7,\n",
    "                                                                    ax=ax, color='#7FAFCE')\n",
    "            df[df['churn'] == 1][xcol].plot(kind='hist', alpha=0.7,\n",
    "                                                                    ax=ax, color='#F9C764')\n",
    "            ax.set_xlabel(xcol)\n",
    "            ax.set_title(f'Churn by {xcol}', size=10)\n",
    "\n",
    "\n",
    "def get_clf_rpt(clfs, X_train, y_train, X_test, y_test):\n",
    "    # declare dict obj to store classification report\n",
    "    clf_rpt_dict = {}\n",
    "    for cls in clfs:\n",
    "        clf_pipe =  Pipeline([('scaler', StandardScaler()),\n",
    "                      (type(clf).__name__, clf)\n",
    "                     ])\n",
    "        v.fit(X_train, y_train)\n",
    "        y_hat_test = clf_pipe.predict(X_test)\n",
    "        clfd[type(clf).__name__] = pd.DataFrame(classification_report(y_test, y_hat_test,\n",
    "                                    target_names=['Not Churn', 'Churn'], output_dict=True)).T\n",
    "    \n",
    "    return pd.concat(clf_rpt_dict.values(), axis=1, keys=clf_rpt_dict.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
