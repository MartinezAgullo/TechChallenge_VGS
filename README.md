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
Figure 6: Monthly sales over time. While the total amount per month is a relatively stable quantity, it has a global minimum on February 


## Part 2: Predictive modeling

## Part 3: Natural language processing

## Part 4: Real-world scenario

### Actionable Insights for Business Improvement

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


