# Data Scientist - Technical Challenge

Technical Challenge for the Data Scientist position at Volkswagen Group Services

- Jupyter notebook: [Notebook/Prueba_VSG.ipynb](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Notebook/Prueba_VSG.ipynb) 

- pdf version of the notebook: [Prueba_VSG.pdf](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Prueba_VSG.pdf)

You can find detailed information on each step in the Jupyter notebook
Here I only provide a summary of the deliverables.

## Part 1: Exploratory data analysis
After the data cleaning, the dataset containing information about Customer transactions has dimensions (989, 8).

The columns and its data types are:

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
Finally, Figure 6 presents the monthly sales over time.


![histograms for numerical variables](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/15_VariableHistos.png)
Figure 1: Numerical variables. All of them present a relatively uniform distribution, except the 'amount' variable.


![Correlation between variables](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/11_Correlations.png)
Figure 2: Correlation among numerical features. Note that customer_id and customer_income are not related at all. This is surprising because one might assume that 'customer_income' refers to the income of the customer identified by 'customer_id'.


![High correlation](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/12_HighCorrelation.png)
Figure 3: Correlation between the transaction amount and the customer income.


![histograms for Product Categories](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/13_Categorical_A.png)
Figure 4: Histogram for Product Categories. All categories are balanced.


![histograms for Payment Methods](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/13_Categorical_A.png)
Figure 5: Histogram for Payment Methods. All categories are balanced.


![Monthly sales](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/16_TransactionsTime.png)
Figure 6: Monthly sales over time. While the total amount per month is a relatively stable quantity, it has a global minimum on February. 


## Part 2: Predictive modeling
In this exercise, it is suggested to study customer churn, i.e., whenever a customer leaves the service.
Figures 7 and 8 provide insights about the behaviour of the clients regarding churn.

![Ocurrences of customer id](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/21_Churn.png)
Figure 7: Number of transactions per customer. It can be seen that all clients had made several transactions within the service. The most common value is 9, and the maximum is 20.

![Distribution of Days Between Transactions](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/21_Churn.png)
Figure 8: Time difference between transactions. 

To produce a predictive model, the first step is to define criteria in the dataset for labeling a client as churned. Defining customer churn based on "no transactions within a specific time frame" is a common approach. The overall average transaction frequency is 35,75 days. 
As an approximation, one could consider that a client has churned using the 75th percentile (Q3). Therefore, if it has been more than 'days_75th_percentile' (46.0 days) without transactions, it is considered churn. Choosing Q3 is an arbitrary threshold, and higher percentiles (e.g., 80th or 90th) could also be considered.

Once that the churn label has been added, the corresponding ML algorithm for prediction can be trained. 
In this case, a sklearn-based LSTM model has been used. The hyperparameters have been optimised with a grid search. 


| Best LSTM Model Evaluation:        |               |        |          |         |
|------------------------------------|---------------|--------|----------|---------|
|                                    | precision     | recall | f1-score | support |
| No churn (0)                       | 0.75          | 0.97   | 0.85     | 118     |
| Churn (1)                          | 0.00          | 0.00   | 0.00     | 0.00    |
|                                    |               |        |          |         |
| accuracy                           |               |        | 0.73     | 157     |
| macro avg                          | 0.37          | 0.49   | 0.42     | 157     |
| weighted avg                       | 0.56          | 0.73   | 0.64     | 157     |
|                                    |               |        |          |         |
| ROC-AUC: 0.487                     |               |        |          |         |


The precision, recall, and F1-score for the 'Churn' class are all 0.00, indicating that the model fails to correctly identify any customers who churn. The low ROC value suggests that the model is not learning. To fix this, a more simple algorithm could be useull.
Autoregression can be a viable alternative to neural networks. While it may not capture complex patterns as effectively as more sophisticated models, it's often more interpretable and easier to implement. 




## Part 3: Natural language processing

## Part 4: Real-world scenario

**Actionable Insights for Business Improvement**

____
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


