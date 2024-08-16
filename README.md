# Data Scientist - Technical Challenge

Technical Challenge for the Data Scientist position at Volkswagen Group Services

- Jupyter notebook: [Notebook/Prueba_VSG.ipynb](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Notebook/Prueba_VSG.ipynb) 

- pdf version of the notebook: [Prueba_VSG.pdf](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Prueba_VSG.pdf)

You can find detailed information on each step in the Jupyter notebook
Here I only provide a summary of the deliverables.

## Part 1: Exploratory data analysis
After the data cleaning, the dataset containing information about Customer transactions has dimensions (989, 8).

The columns and its corrected data types are:

| Variable                      | Type           |
|-------------------------------|----------------|
| customer_id                   | int64          |
| transaction_date              | datetime64[ns] |
| amount                        | float64        |
| product_category              | object         |
| payment_method                | object         |
| customer_age                  | int64          |
| customer_income               | float64        |
| dtype                         | object         |

There are three unique categories for Products ('groceries' 'clothing' 'electronics') and for Payment Methods ('debit card' 'paypal' 'credit card').

Main characteristics of the data

|      | transaction_date | amount      | customer_age | customer_income |
|------|------------------|-------------|--------------|-----------------|
| mean | 2024-01-22       | 988.841729  | 43.6         | 71141.64        |
| min  | 2023-07-31       | 248.789798  | 18           | 20111.77        |
| 25%  | 2023-10-22       | 733.200329  | 31           | 46407.07        |
| 50%  | 2024-01-20       | 982.027657  | 43           | 70481.60        |
| 75%  | 2024-04-24       | 1252.443139 | 57           | 96152.29        |
| max  | 2024-07-29       | 1679.681855 | 69           | 119941.30       |
| std  | NaN              | 333.239302  | 15.0         | 28889.57        |


### Data Visualizations
First, histograms for the numerical variables are created (Figure 1). The correlations among these are studied in Figure 2. The high correlation between the transaction amount and the customer income is further analysed in Figure 3. The histogram for categorical features is presented in Figures 4 and 5.
The amount spent per product category is balanced across the three classes (see plot in the notebook).
Finally, Figure 6 presents the monthly sales over time. The latter, while being relative stable, presents a slight negative slope. A linear fit is a simpel way to quantify this: y = (-56.78 ± 16.57)x + (42034230.86 ± 12244833.75). 


![histograms for numerical variables](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/15_VariableHistos.png)

Figure 1: Numerical variables. All of them present a relatively uniform distribution, except the 'amount' variable.


![Correlation between variables](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/11_Correlations.png)

Figure 2: Correlation among numerical features. Note that customer_id and customer_income are not related at all. This is surprising because one might assume that 'customer_income' refers to the income of the customer identified by 'customer_id'.


![High correlation](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/12_HighCorrelation.png)

Figure 3: Scatter plot presenting the correlation between the transaction amount and the customer income.


![histograms for Product Categories](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/13_Categorical_A.png)

Figure 4: Histogram for Product Categories. All categories are balanced.


![histograms for Payment Methods](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/13_Categorical_A.png)

Figure 5: Histogram for Payment Methods. All categories are balanced.


![Monthly sales](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/16_TransactionsTime.png)

Figure 6: Monthly sales over time. While the total amount per month is a relatively stable quantity, it has a global minimum on February. 


## Part 2: Predictive modeling
### Predict customer churn

In this exercise, it is suggested to study customer churn, i.e., whenever a customer leaves the service.
Figures 7 and 8 provide insights into customer behavior regarding churn. The histogram in Figure 7 has been fitted to a Gaussian distribution. 

| Fit                 | Mean | std  |
|---------------------|------|------|
| Full data           | 9.89 | 3.16 |
| N transactions < 18 | 9.71 | 2.93 |

![Occurrences of customer id](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/21_Churn.png)

Figure 7: Histogram for number of transactions per customer. It can be seen that all clients had made several transactions within the service. The most common value is 9, and the maximum is 20. The solid lines correspond to two different Gaussian fits; one to all the data (black) and another removing the outliers (red).

![Distribution of Days Between Transactions](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/22_DaysBetweenTransactions.png)

Figure 8: Histogram for the time difference between transactions. 

To produce a predictive model, the first step is to define criteria in the dataset for labeling a client as churned. Defining customer churn based on "no transactions within a specific time frame" is a common approach. The overall average transaction frequency is 35.75 days. 
As an approximation, one could consider that a client has churned using the 75th percentile (Q3). Therefore, if it has been more than 'days_75th_percentile' (46.0 days) without transactions, it is considered churn. Choosing Q3 is an arbitrary threshold, and higher percentiles (e.g., 80th or 90th) could also be considered.

Once the churn label has been added, the corresponding ML algorithm for prediction can be trained. 
In this case, a sklearn-based LSTM model has been used. 
While LSTMs can capture dependencies over time, they need a relatively large amount of data to train.
The hyperparameters have been optimised with a **grid search**. To run the grid optimization in the notebook, set the RunGridSearch to True (it is currently disabled to reduce the size of the document).



| Best LSTM Model Evaluation            | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| **No churn (0)**  | 0.75      | 0.97   | 0.85     | 118     |
| **Churn (1)**     | 0.00      | 0.00   | 0.00     | 39      |
| **Accuracy**      |           |        | 0.73     | 157     |
| **Macro Avg**     | 0.37      | 0.49   | 0.42     | 157     |
| **Weighted Avg**  | 0.56      | 0.73   | 0.64     | 157     |

**ROC-AUC: 0.487**



The precision, recall, and F1-score for the 'Churn' class are all 0.00, indicating that the model fails to correctly identify any customers who churn. The low ROC value suggests that the model is not learning. To fix this, a simpler algorithm could be useful.
<!--Autoregression can be a viable alternative to neural networks. While it may not capture complex patterns as effectively as more sophisticated models, it's often more interpretable and easier to implement. -->
A linear regression can be a viable alternative to neural networks. While it may not capture complex patterns as effectively as more sophisticated models, it's often more interpretable and easier to implement. A SMOTE (Synthetic Minority Over-sampling Technique) has been applied to correct the unbalnce between classes. The results of this model are:

| Logistic Regression Model Evaluation            | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| **Class 0 (No Churn)** | 0.80      | 0.59   | 0.68     | 124     |
| **Class 1 (Churn)**    | 0.23      | 0.45   | 0.30     | 33      |
| **Accuracy**       |           |        | 0.56     | 157     |
| **Macro Avg**      | 0.51      | 0.52   | 0.49     | 157     |
| **Weighted Avg**   | 0.68      | 0.56   | 0.60     | 157     |

**ROC-AUC: 0.54**

The updated model shows improved recall for the Churn class, indicating a more balanced consideration of both classes. 
However, it still has low precision for the Churn class, leading to a risk of false positives, which could cause unnecessary actions in real-world applications.

I believe the poor results in predicting churn primarily stem from how the churn variable is defined. 
Additionally, I think having more data could significantly improve the model's training and performance.

### Predict transaction amount
The transaction amount distribution is presented and fitted in Figure 9. First to a Gaussian and then to a Beta distribution. The latter seems to better describe the data.

![Transaction amount fit](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/25_TransactionFit_merged.png)

Figure 9: Histograms of the transaction amounts fitted to (left) a Gaussian and (right) a Beta distributions. The fit parameters are shown on the stat box. 


Metrics employed for model evaluation

- Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.

- Mean Squared Error (MSE): Measures the average of the squares of the errors. It gives more weight to larger errors.

- Root Mean Squared Error (RMSE): The square root of the MSE. Sensitive to outliers.

- R² Score: Coefficient of Determination.

- Mean Absolute Percentage Error (MAPE): Measures the accuracy of a forecasting method in predicting values.

- Median Absolute Error (MdAE): Measures the median of the absolute errors. Less sensitive to outliers.


| Regression model     | MAE    | MSE      | RMSE   | R² Score | MAPE   | MdAE    |
|----------------------|--------|----------|--------|----------|--------|--------|
| Linear               | 121.44 | 19979.74 | 141.35 | 0.8158   | 14.72% | 114.83 |
| Random Forest        | 126.99 | 22709.51 | 150.7  | 0.7906   | 15.22% | 117.76 |
| XGBoost              | 134.97 | 26853.5  | 163.87 | 0.7524   | 16.05% | 124.1  |
| Support Vector       | 260.65 | 97412.64 | 312.11 | 0.1019   | 33.45% | 235.0  |
| Neural Network (MLP) | 127.63 | 22823.34 | 151.07 | 0.7896   | 15.39% | 115.61 |




## Part 3: Natural language processing
In this part, the dataset with the information about Customer reviews is employed.
For NLP we will be using the [Natural Language Toolkit (NLTK)](https://www.nltk.org/index.html), a leading platform for building Python programs to work with human language data.


The distribution of scores is presented in Figure 9, and the average score customer in Figure 10. The evolution of the sentiment score per week and per month can be checked in the notebook.



![Sentiment Score](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/31_SentimentScore.png)

Figure 10: Histograms of the sentiment score distribution. Note that the NLTK tool does not provide uniformly distributed scores, which is due to the fact that the same reviews are copy-pasted throughout the dataset. In a more realistic scenario, this distribution would look more uniform. It can be seen that the majority of reviews are deemed as positive.


![Avg Score](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/32_AvgSentimentScore.png)

Figure 11:  Plot the sentiment score vs customer ID. The entries have being oredered from largest to lowest average sentiment score. The majority of customers have an average positive experience.

## Part 4: Real-world scenario
**Topic modelling**

**Positive ::**  Key terms such as "fast", "works", "delivery", "perfectly", "quality" suggest that the positive feedback is focused on fast delivery, the product working perfectly, and the overall quality. The terms acceptable", "expectations", "exceeded", "happy", "purchase" indicate that, in some skituations,the products or services exceeded the customer's expectations.



**Negative ::**  Some customers are highly dissatisfied with the quality of the products, stating that they are poorly made or do not match the descriptions provided (key Terms: "terrible", "described", "poor", "quality", "recommend"). The wide usage of terms such us "arrived", "damaged", "unhelpful", "described", and "terrible" highlights problems with products arriving damaged and customer service being unhelpful in resolving these issues. 
Other issue is that customers are reporting that products are breaking easily, leaving them unhappy with their purchases("broke", "one", "unhappy", "use").

____

**Actionable Insights for Business Improvement**


1. Enhance product quality control: 
    - Issue: Discrepancies between product descriptions and what customers receive were reported in 14.42% of negative comments.
    <!-- Consistent complaints about poor product quality and items not matching descriptions. This is mentioned in 14.4% of negative comments. -->

    - Action: Ensure that product descriptions are accurate.  In order to build trust, be transparent about product limitations or potential issues in descriptions.
____
2. Improve the shipping process:
    - Issue: Frequent reports of products arriving damaged.

    - Action: Improve the packaging standards to better protect products during shipping.
____
3. Maintain the delivery speed:
    - Issue: Almost 16% of positive comments referred to the fast delivery time.

    - Action: Continue with the current delivery strategy.
    
    
____
4. Revamp customer service training and policies:
    - Issue: 22.1% of negative comments mentioned how unhelpful the customer service was.

    - Action: Invest in customer service training focused on empathy, problem-solving, and effective communication. Introduce more customer-friendly return, exchange, and complaint resolution policies to address issues promptly and to customer satisfaction.
____


