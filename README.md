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
| Nº transactions < 18 | 9.71 | 2.93 |

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
A linear regression can be a viable alternative to neural networks. While it may not capture complex patterns as effectively as more sophisticated models, it's often more interpretable and easier to implement. SMOTE (Synthetic Minority Over-sampling Technique) has been applied to correct the unbalance between classes. The results of this model are:

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



**ML implementations** 
 There are several possible ML-based implementations for this task. Firstly we will build a few basic models to test and compare them.
 
 1. **Linear Regression**
   - **Use Case**: Best for simple, linear relationships.
   - **Advantages**: Easy to interpret and works well with large datasets.
   - **Disadvantages**: May not perform well with complex datasets due to its assumption of linearity.

2. **Random Forest Regression**
   - **Use Case**: Suitable for capturing non-linear relationships.
   - **Advantages**: Handles both linear and non-linear relationships, provides feature importance (Figure 10), and reduces overfitting.
   - **Disadvantages**: Less interpretable than linear models.

3. **Gradient Boosting Machines (e.g., XGBoost, LightGBM)**
   - **Use Case**: Ideal when high accuracy is needed.
   - **Advantages**: Often delivers state-of-the-art performance for regression and handles non-linear relationships effectively.
   - **Disadvantages**: More complex to tune and interpret, and sensitive to hyperparameters.

4. **Support Vector Regression (SVR)**
   - **Use Case**: Good for data with many outliers or when capturing complex relationships is crucial.
   - **Advantages**: Effective in high-dimensional spaces and robust to outliers.
   - **Disadvantages**: Computationally expensive and requires careful hyperparameter tuning.

5. **Neural Networks**
   - **Use Case**: Best for large datasets with complex relationships.
   - **Advantages**: Can model highly complex relationships and is flexible in architecture and hyperparameters.
   - **Disadvantages**: Requires significant computational resources, large amounts of data, and careful tuning.


![Feature ranking](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/26_FeatureImportance.png)

Figure 10: Ranking of the most relevant variables for the RandomForestRegressor model.

| **Feature**                         | **Importance**|
|---------------------------------|-----------|
|                customer_income  |  0.863298 |
|                 transaction_id  |  0.043099 |
|                   customer_age  |  0.039149 |
|                    customer_id  |  0.036174 |
|          payment_method_paypal  |  0.004898 |
|   product_category_electronics  |  0.004737 |
|     product_category_groceries  |  0.004695 |
|       payment_method_debit card |  0.003950 |



Metrics employed for model evaluation of the different models:

- Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.

- Mean Squared Error (MSE): Measures the average of the squares of the errors. It gives more weight to larger errors.

- Root Mean Squared Error (RMSE): The square root of the MSE. Sensitive to outliers.

- R² Score: Coefficient of Determination.

- Mean Absolute Percentage Error (MAPE): Measures the accuracy of a forecasting method in predicting values.

- Median Absolute Error (MdAE): Measures the median of the absolute errors. Less sensitive to outliers.

Evaluation of metrics:

| **Regression model**     | **MAE**    | **MSE**      | **RMSE**   | **R² Score** | **MAPE**   | **MdAE**    |
|----------------------|--------|----------|--------|----------|--------|--------|
| **Linear**               | 121.44 | 19979.74 | 141.35 | 0.8158   | 14.72% | 114.83 |
| **Random Forest**        | 126.99 | 22709.51 | 150.7  | 0.7906   | 15.22% | 117.76 |
| **XGBoost**              | 134.97 | 26853.5  | 163.87 | 0.7524   | 16.05% | 124.1  |
| **Support Vector**       | 260.65 | 97412.64 | 312.11 | 0.1019   | 33.45% | 235.0  |
| **Neural Network (MLP)** | 127.63 | 22823.34 | 151.07 | 0.7896   | 15.39% | 115.61 |


Before any optimisation, the linear regression appears to have the best performance. Nevertheless, both the Random Forest (RF) and the Neural Networks (NN) present a comparable performance. RF and NN could be further explored with hyperparameter tuning to potentially improve their performance. XGBoost might also benefit from hyperparameter tuning, but initial tests suggest worse performance compared to LR, RF, or NN. SVR should be deprioritized due to its poor performance.


In order to improve the performance of the model a Genetic Algorithm (GA) is used. A GA is a search heuristic inspired by the process of natural selection that is used to find approximate solutions to optimization and search problems. The workflow of the GA is described in Figure 11.
To implement the GA on the notebook, set the RunGA option to True. 

![Genetic algorithm](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/27_GeneticAlgorithm.png)

Figure 11: The evolutionary cycle of a typical evolutionary algorithm. Each block represents an operation on a population of candidate solutions.

** Optimized Regression Models Performance** 

The results of the GA are presented in the table below.
 
| **Metric**                          | **Linear Regression** | **Random Forest Regression** | **Neural Network (MLP) Regression** |
|-------------------------------------|-----------------------|------------------------------|-------------------------------------|
| **MAE**                             | 121.44                | 124.59                       | 126.49                              |
| **MSE**                             | 19979.74              | 20759.12                     | 22433.54                            |
| **RMSE**                            | 141.35                | 144.08                       | 149.78                              |
| **R² Score**                        | 0.8158                | 0.8086                       | 0.7932                              |
| **MAPE**                            | 14.72%                | 15.15%                       | 14.45%                              |
| **MdAE**                            | 114.83                | 121.83                       | 126.03                              |


- **Linear Regression** performs the best overall, with the lowest errors (MAE, MSE, RMSE) and the highest R² score, making it a strong, simple model for this dataset.
  
- **Random Forest Regression** shows slightly higher errors but still performs well, making it a good choice if more complex relationships or robustness are needed.

- **Neural Network (MLP) Regression** has slightly higher errors and lower R² compared to the other models. It might benefit from more data or further tuning but offers slightly better accuracy on a percentage basis (MAPE).

Therefore, **Linear Regression** is the most suitable model given the current dataset.



## Part 3: Natural language processing
In this part, the dataset with the information about Customer reviews is employed.
First, the dataset is preprocessed:
    - Tokenization: Break down the review text into individual words.
    - Stopword Removal: Remove common words like 'the', 'is', etc., that don't contribute to the meaning.
    - Stemming/Lemmatization: Reduce words to their root form.
    - Remove rows with empty 'review_text'
For NLP we will be using the [Natural Language Toolkit (NLTK)](https://www.nltk.org/index.html), a leading platform for building Python programs to work with human language data.

The distribution of scores is presented in Figure 12, and the average score customer in Figure 13. The evolution of the [Sentiment Score per week](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/33_AvgSentimentPerWeek.png) and per month can be checked in the notebook. The scores have been obtained with the nltk::SentimentIntensityAnalyzer()::polarity_scores() function.
By setting the option InspectScores to True in the notebook, a printout presents each review with the assigned score, allowing to manually evaluate the the behaviour of the polarity_scores() function.



![Sentiment Score](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/31_SentimentScore.png)

Figure 12: Histograms of the sentiment score distribution. Note that the NLTK tool does not provide uniformly distributed scores, which is due to the fact that the same reviews are copy-pasted throughout the dataset. In a more realistic scenario, this distribution would look more uniform. It can be seen that the majority of reviews are deemed as positive.


![Avg Score](https://github.com/MartinezAgullo/TechChallenge_VGS/blob/main/Images/32_AvgSentimentScore.png)

Figure 13:  Plot the sentiment score vs customer ID. The entries have being oredered from largest to lowest average sentiment score. The majority of customers have an average positive experience.

## Part 4: Real-world scenario
###Topic modelling
[Latent Dirichlet Allocation (LDA)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) is one of the most popular topic modeling methods [1].
The 'CountVectorize' function Converts the text data into a document-term matrix (DTM). The parameters of CountVectorizer(max_df=0.9, min_df=2, stop_words='english') are:
    - max_df=0.9: Ignores words that appear in more than 90% of the documents, considering them too common to be informative.

    - min_df=2: Ignores words that appear in fewer than 2 documents, considering them too rare.

    - stop_words='english': Removes common English stop words (like "the", "is", etc.). I created a custom_stop_words_list for the stop words, to remove those that I consider uninteresting for topic modelling.

he parameters of CountVectorizer and n_components (number of topics to be found) can be tweaked, allowing exploration of how different configurations yield different topic modelling results. As an example, the output of the topic extraction with four topics, max_dfs=0.6 and min_dfs=4:

| Topic | Keyworkds                                                             |
|-------|------------------------------------------------------------------------|
| 0     | broke, unhappy, amazing, definitely, poor, recommend, quality          |
| 1     | excellent, unhelpful, damaged, customer, arrived, improved, acceptable |
| 2     | decent, available, service, excellent, highly, recommend, product      |
| 3     | long, arrive, perfectly, fast, delivery, terrible, work                |

More configuration are available in the notebook.

Overall, the keywords that define the topics can be separated in positive and negative:

- **Positive ::**  Key terms such as "fast", "works", "delivery", "perfectly", "quality" suggest that the positive feedback is focused on fast delivery, the product working perfectly, and the overall quality. The terms acceptable", "expectations", "exceeded", "happy", "purchase" indicate that, in some situations,the products or services exceeded the customer's expectations.
- **Negative ::**  Some customers are highly dissatisfied with the quality of the products, stating that they are poorly made or do not match the descriptions provided (key Terms: "terrible", "described", "poor", "quality", "recommend"). The wide usage of terms such as "arrived", "damaged", "unhelpful", "described", and "terrible" highlights problems with products arriving damaged and customer service being unhelpful in resolving these issues. 
Another issue is that customers are reporting that products are breaking easily, leaving them unhappy with their purchases("broke", "one", "unhappy", "use").



###Actionable Insights for Business Improvement

____
1. Enhance product quality control: 
    - **Issue**: Discrepancies between product descriptions and what customers receive were reported in 14.42% of negative comments.
    <!-- Consistent complaints about poor product quality and items not matching descriptions. This is mentioned in 14.4% of negative comments. -->

    - **Action**: Ensure that product descriptions are accurate.  In order to build trust, be transparent about product limitations or potential issues in descriptions.
____
2. Improve the shipping process:
    - **Issue**: Frequent reports of products arriving damaged.

    - **Action**: Improve the packaging standards to better protect products during shipping.
____
3. Maintain the delivery speed:
    - **Issue**: Almost 16% of positive comments referred to the fast delivery time.

    - **Action**: Continue with the current delivery strategy.
    
    
____
4. Revamp customer service training and policies:
    - **Issue**: 22.1% of negative comments mentioned how unhelpful the customer service was.

    - **Action**: Invest in customer service training focused on empathy, problem-solving, and effective communication. Introduce more customer-friendly return, exchange, and complaint resolution policies to address issues promptly and to customer satisfaction.
____




## References

[1] Blei, D., Ng, A., & Jordan, M. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3(4-5), 993–1022. Paper presented at the 18th International Conference on Machine Learning, Williamstown, Massachusetts, June 28-July 1, 2001. https://doi.org/10.1162/jmlr.2003.3.4-5.993
