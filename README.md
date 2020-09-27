## Table of contents
* [Repo Structure](#Repo-Structure)
* [How to Run the Codes](#How-to-Run-the-Codes)
* [Introduction](#Introduction)
* [Business Case](#Business-Case)
* [Approach](#Approach)
* [The OSE Part](#The-OSE-Part)
* [The M Part](#The-M-Part)
* [Model Evaluation](#Model-Evaluation)
* [Alternate Model Model B](#Alternate-Model-Model-B)
* [Model B Evaluation](#Model-B-Evaluation)
* [Interpretaion](#Interpretation)
* [Conclusion](#Conclusion)
* [Recommendation](#Recommendation)
* [Future Work](#Future-Work)

## Repo Structure
* Root of the repo contains the main jupyter notebook and a python file of my fuctions
* A PDF file delivers the presentation of this project
* img folder holds all the images for the repo README.md and presentation.
* csv_files folder houses the data source and mapping of States to Region file.

## How to Run the Codes
<ul>
    <li>Use git clone to create a local repo.  URL is: ???</li>
    <li>Required Libraries:
        <ul>
            <li>time, process_time</li>
            <li>pandas</li>
            <li>numpy</li>
            <li>matplotlib.pyplot</li>
            <li>sklearn plus various modules</li>
            <li>myFunc.py</li>
            <li>Shap</li>
            <li>xgboost</li>
        </ul>
    </li>
</ul>

## Introduction
The objective of this project is to produce a classification model that will predict if customers will cease their telephone services with Syria Tel.

## Business Case
The targeted audience will be the sales and customer services of the company. The model will provides the audience with insights as to why we might lose a customer's subscription. Knowing these factors is critical as we can take corrective and preventive actions to retain the customers and prevent revenue loss.

## Approach
Out of the popular data science processes, such as CRISP-DM or KDD, we have selected the OSEMN framework. Cleaning and analyzing the data set according to the guidelines defined in the OSEMN framework.

In addition, we will evaluate four classifiers at different stages, base model, data resampling and fine tuning the hyperparameters.  The one that produces the higher recall score for both labels will be our final model.


## The OSE Part
In the first 3 parts of the OSEMN framework, the following procedures were performed:
1. Drop irrelevant columns.
2. Group States by Regions.
3. Handle null values if any.
4. Examine Churn column distribution.
5. Explore correlation between explanatory variables and the churn target.
6. Explore correlation between independent variables.
7. Study feature columns distribution in relation to target.
8. Drop columns showing strong multicollinearity.
9. Perform one hot encoding with pandas get_dummies

## The M Part
### Baseline Model 
Create a baseline Model that composes of 4 classifiers, Logistic Regression, KNN, Random Forest and XGBoost.  Fit the model with defalut parameters.  

Baselilne model classification scores
![](/img/baseline_scores.png?raw=true)

Churn Column examination reveals class imbalance issue.

Churn distribution plot
![](/img/sld_churn.png?raw=true)

### Transform training data with SMOTE
To address the class imblance issue, we use SMOTE to transform the training data and fit the model.

Scores changes SMOTE scores vs baseline
![](/img/smote_baseline_comp.png?raw=true)


### Hypterparameter Tuning
Define grid search parameter and set cross validation to 5.

Tuned model with SMOTE classifcation scores
![](/img/tuned_baseline_scores.png?raw=true)

Performance gain/loss - tuend vs baseline
![](/img/tuned_baseline_comp.png?raw=true)

### Conclusion
Based on the classification report charts, XGBoost is clearly the superior model. We will use it for the next step, model evaluation.

## Model Evaluation
Comparing the train and test accuracy scores shows that this model might be over fitting.  We have 100% in train accuracy score, while test accuracy score is only 92.09%.  However, test AUC score is pretty good at 83.36% which means there is a 83.36% probability that the model will correctly distinguish between the two labels.  We will refer to this model as Model A.

ROC Curve
![](/img/ROC.png?raw=true)


## Alternate Model Model B
The sample data includes States where the customers reside. Perhaps, the geographic data might provide additional insights as to why and where the churn happen. Here we will construct another model without grouping States by Region. It will be referred to as model B.

### Baseline
Model B baseline classification scores
![](/img/mdl_b_baseline_score.png?raw=true)

### Tuned
Model B classification scores after tuning
![](/img/mdl_b_tuned_comp.png?raw=true)

### Tuned vs Baseline
Model B performance gain/loss - tuend vs baseline
![](/img/mdl_b_tuned_baseline_comp.png?raw=true)

## Model B Evaluation
There is a difference of about 5% between the train (99.91%) and test (94.84%) accuracy scores which suggests Model B is less over fitting than Model A.  Furthermore, Model B improves vastly in the AUC score, 88.06% vs 83.36%.

ROC Curve
![](/img/mdl_b_roc.png?raw=true)

Performance gain/loss between Model B and Model A
![](/img/mdl_b_vs_mdl_a.png?raw=true)

### Conclusion
The XGBoost in Model B outperformed Model A in all aspect of the scores.

Model B will be our final model and we will use in our next step, model interpretation.

## Interpretation
We will use Shap to help us explain our model because it provides a clearer picture as to how each independent variable impacts the model outcomes.

### Feature Important
![](/img/shap_fea_imp.png?raw=true)

Observations
* The 'total day minutes' and 'voice mail plan_yes' features have the biggest impact on the model outcome. It changes the predicted churn probability by about 120% (approx. 1.2 on the x-axis) and 98% on average respectively.
* The 'number vmail messages' and 'customer service calls' came in third and fourth as the most influential features.

### Shap values distribution by features
![](/img/shap_val_by_fea.png?raw=true)

Observations
* In general, the higher the actual feature values in 'total day minutes' (red dots), the higher the prediction in the 'churn' label.  On the hand, the lower their actual feature values (blue dots), reduce the prediction for the 'churn' label.

* Customers with voice mail plan tend not to leave while customers without a voice mail plan contribute more toward the the churn label prediction.

* High feature values in 'number vmial messages' has a positive impact in predicting the 'churn' label in the model outcomes.

* For feature 'customer service calls', the higher the number of calls to the customer service line, the higher the risk that the customers will leave.

### Examine individual instance
![](/img/ind_inst.PNG?raw=true)

Observations
* In this instance, the actual churn rate is False or 0. Here we can how the the features in the model is working together to come to a false prediction.
* In this case, the 'state_NV', 'total day minutes' and 'customer services calls' are pushing toward a False churn rate while the features 'voice mail plan_yes' and 'total intl minutes' are acting in the opposite direction. Since the negative forces outweigh the positive forces, they push the model toward a False prediction.



## Conclusion
### Our Classifier

All of the selected supervised learning methods in this project are capable of tackling binary classification. Each have their strengths and weaknesses for different business situations. In our case of predicting the Churn rate, the XGBosst classifier out performed the other three classifiers in term of the accuracy, f1_score, precision and recall scores. Especially the recall score because we need to identify all potential customers that are like to cancel their services.

We picked Model B because Model B have a higher scorings in the classification report than Model A. Furthermore, Model B seems to be less overfitting than Model A. Lastly, Model B AUC score is significantly higher than Model A.

## Recommendation
Based on the available data, our model is able to identify which explanatory variables have the biggest impact on the Churn rate. We will based our recommendations based on our top four features.

1. 'Total day minutes': Customers with high number of minutes used during the day, have a greater chance of churn. A remedy to such problem would be to review the pricing structure for total day minutes. Perhaps, we can micro-segment the pricing structure for total day minutes charges.  
Another approach we can implement is to model what the airline industry is offering their customer with the 'frequent-flyer miles'. We could grand the heavy day time usage customers with VIP status and they will receive certain amount of benefits when they reach certain usage level. Thus providing the justification to make more or longer call without feeling spending more.

2. 'Voice mail plan_yes': Customers without a voice mail plan tend to end their services with the company. It is possible that they find other competitor offering such feature without addition charges. We should carry out a comparative analysis to see if we can match the competitions, in order to keep the customers happy.

3. 'No vmail messages': Are we charging customers when they exceed certain amount of voice mails? Do we have a limit as to how many voice mails or fixed length of time slot they are allow to use? Understanding what our competitors are offering is crucial so we can provide competitive offerings. Furthermore, if our current infrastructure reach a bottleneck, capacity and usage analysis will need to be carried out so that new offerings by our company will not over tax on our daily operations.

4. 'Customer service calls': The model indicates that customers with high number of service calls show higher risk of churn. A analysis of the call nature will help us prioritize area of improvements. Questions like is the call information or technical? Are customers calling with the same questions? If yes, it would suggest that more training are needed for the customer service agents.

## Future Work
1. In this project, we have selected four different types of supervised learning algorithms namely, Logistic Regression, K-Nearest Neighbors, Random Forest and XGBosst, because each classifier employs different algorithm to optimize predictions. It would be interesting to see what type of results we could obtain, if we were to use other algorithms, such as Bayesian Classification or the Support Vector Machines.
2. We can also try different techniques to deal with class imbalance problems, for instance, using oversampling and undersampling and see how well each of our models performs.
3. Each classifier come with number of parameters for fine tuning. Finding the best combination of parameters can be computationally expensive. In this project we selected the most common parameters for fine tuning. If time allows, we can introduce more hyperparameters for each classifier to the grid search method can see where it will bring us in term of the model performances.






