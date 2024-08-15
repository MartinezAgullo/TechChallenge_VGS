# # Data Scientist - Technical Challenge

Technical Challenge for the Data Scientist position at Volkswagen Group Services

Jupyter notebook: Notebook/Prueba_VSG.ipynb
Results in pdf: Prueba_VSG.pdf

You can find the jupyeter notebook Notebook/Prueba_VSG.ipynb the detailed information of each step.
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

There three unique categories for Products ('groceries' 'clothing' 'electronics') and for Payment Methods ('debit card' 'paypal' 'credit card').

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




## Part 2: Predictive modeling

## Part 3: Natural language processing

## Part 4: Real-world scenario


