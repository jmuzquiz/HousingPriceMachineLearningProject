# California Housing Price Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Google Colab Notebook](#google-colab-notebook)
- [Data Description](#data-description)
- [Dataset Source](#dataset-source)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
  - [Linear Regression](#linear-regression)
  - [Random Forest](#random-forest)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Visualizations](#visualizations)
- [Insights and Interpretations](#insights-and-interpretations)
- [Project Goals Met](#project-goals-met)
- [Limitations](#limitations)
- [Future Steps](#future-steps)
- [Acknowledgements](#acknowledgements)
- [Requirements](#requirements)
- [License](#license)

## Project Overview
This project aimed to predict housing prices in California using machine learning techniques, utilizing the California Housing Prices dataset which contains information from the 1990 census. Although the dataset reflects historical data, it provided a valuable platform for applying and demonstrating various machine learning methodologies. The analysis involved several key steps:

1. **Data Exploration and Preprocessing**:
   - Investigated the dataset to understand feature distributions and relationships.
   - Performed data cleaning and preprocessing, including handling missing values, encoding categorical variables, and scaling features.

2. **Model Development and Evaluation**:
   - Developed a baseline Linear Regression model and evaluated its performance.
   - Trained a Random Forest model to enhance predictive accuracy and compared it against the linear model.

3. **Hyperparameter Tuning**:
   - Optimized the Random Forest model using hyperparameter tuning techniques such as RandomizedSearchCV to identify the best model parameters.

4. **Model Assessment**:
   - Assessed model performance using metrics such as R², RMSE, and MAE to determine the effectiveness of the models.
   - Analyzed feature importance to understand the key factors influencing housing prices.

5. **Visualization and Insights**:
   - Created visualizations to present the model results and feature importances clearly.
   - Provided insights based on the analysis to interpret the factors driving housing prices and suggested possible model improvements.

The goal of this project was to develop a robust predictive model for housing prices and derive actionable insights from the data. While the dataset represents historical housing conditions, the project demonstrated proficiency in data preprocessing, model building, and performance evaluation, providing a solid foundation for more complex real-world applications.

## Google Colab Notebook
- [Link to the Google Colab Notebook](insert link here)
This notebook contains the Python code used for the housing price prediction analysis in this project.

The notebook includes data preprocessing steps, feature engineering, model development, and evaluation. It covers the implementation of both Linear Regression and Random Forest models, along with hyperparameter tuning and performance assessment. The notebook also features visualizations and insights related to predicting California housing prices based on 1990 data, helping to interpret the historical trends and understand the factors influencing housing values during that period.

## Statistical Tools
For the housing price prediction project, I utilized the following tools and platforms:

- **Python**: Employed for data preprocessing, feature engineering, model development, evaluation, and creating visualizations.
- **Google Colab**: Used for its cloud-based Jupyter notebook environment, which facilitated efficient code execution and provided an accessible platform for potential collaboration.
- **Git and GitHub**: Leveraged for version control, enabling effective tracking of changes, managing project iterations, and sharing the work.
- **NumPy and Pandas**: Essential libraries for data manipulation and analysis, aiding in handling and preparing the dataset.
- **Matplotlib and Seaborn**: Utilized for creating visualizations that illustrate model performance, feature importance, and insights from the data.
- **Scikit-Learn**: Applied for splitting the dataset into training and test sets, implementing machine learning models, performing hyperparameter tuning, and evaluating model performance.

These tools and libraries were crucial for conducting a thorough analysis of housing prices, enabling effective data processing, modeling, and interpretation of results.

## Dataset Source
The dataset used in this project is the California Housing Prices dataset, sourced from [Kaggle](https://www.kaggle.com/datasets). This dataset contains housing attributes from the 1990 California census and is employed to showcase proficiency in linear and nonlinear regression techniques.

## Dataset Description
**Dataset Overview:**
The California Housing Prices dataset includes housing attributes from the 1990 census, serving as a historical reference for analyzing factors that influence housing prices. Although it does not reflect current market conditions, it provides a practical dataset for demonstrating regression analysis skills.

**Features:**
The dataset contains the following features:
- `longitude`: A measure of how far west a house is; a higher value is farther west.
- `latitude`: A measure of how far north a house is; a higher value is farther north.
- `house_median_age`: Median age of a house within a block; a lower number is a newer building.
- `total_rooms`: Total number of rooms within a block.
- `total_bedrooms`: Total number of bedrooms within a block.
- `population`: Total number of people residing within a block.
- `households`: Total number of households, a group of people residing within a home unit, for a block.
- `median_income`: Median income for households within a block of houses (measured in tens of thousands of US Dollars).
- `median_house_value`: Median house value for households within a block (measured in US Dollars).
- `ocean_proximity`: Location of the house w.r.t ocean/sea.

**Feature Engineering and Preprocessing:**
- **One-Hot Encoding**: Categorical variable `ocean_proximity` was one-hot encoded into the following features:
  - `<1H OCEAN`
  - `INLAND`
  - `NEAR OCEAN`
  - `NEAR BAY`
  - `ISLAND`
- **Feature Engineering**:
  - `bedroom_ratio`: Ratio of total bedrooms to total rooms.
  - `rooms_per_household`: Ratio of total rooms to total households.
- **Log Transformations**: Applied to variables with skewed distributions (e.g., `total_rooms`, `total_bedrooms`, `population`, `households`).

**Target Variable:**
- `median_house_value`: The median house value for each block is the target variable for the regression analysis.

**Size and Format:**
- **Number of Observations**: 20,640
- **Number of Features**: 10
- **File Format**: CSV

**Usage:**
This dataset is employed to develop and evaluate predictive models for housing prices based on historical data from the 1990 California census. It helps in understanding how various features influence housing values and demonstrates the application of regression techniques in a real-world context.

**Notes:**
- Missing values were addressed during preprocessing.
- The dataset was split into training and test sets for model evaluation.

## Data Preprocessing and Exploration
- **Initial Setup**:
Before diving into the analysis, it was essential to import the necessary libraries and load the dataset for processing and exploration.
  
  ```python
  #importing essential libraries
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  #loading the dataset
  data = pd.read_csv("housing.csv")
  ```

- **Handling Missing Values**:
Upon initial inspection of the dataset, it was found that the `total_bedrooms` column had missing values. Out of 20,640 entries, only 207 (approximately 1%) were missing. Since this accounts for less than 5% of the dataset, it was deemed acceptable to drop the rows with missing values.

  ```python
  #check for missing values
  data.info()
  
  #drop missing values
  data.dropna(inplace = True)
  ```

- **Splitting Data into Train and Test Sets**:
To evaluate the performance of predictive models effectively, the dataset was split into training and test sets. This approach allowed for training the models on one portion of the data and evaluating them on a separate, unseen portion to ensure the models generalized well to new data.
  ```python
  #split the data into test and train data and x and y data
  from sklearn.model_selection import train_test_split
  
  X = data.drop(['median_house_value'], axis = 1) #say axis =1 since we are dropping column
  y = data['median_house_value']
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #80/20 split
  ```
- **Exploratory Data Analysis**:
Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset and uncovering underlying patterns, trends, and relationships among the features. This process helps in formulating hypotheses and guiding further analysis. In this project, EDA involved several key steps:
  - **Combining Training Data**:
  To facilitate a thorough exploration of the training data, the target variable was recombined with the predictor variables. This combined dataset was used for various EDA techniques.
  ```python
  #combine to make the full train data
  train_data = X_train.join(y_train)
  ```
  - **Visualizing Distributions of Numerical Variables**:
  Plotted histograms for numerical variables to understand their distributions. Some variables appeared skewed and required later transformation.
  ```python
  #basic exploration of numeric variables (not ocean proximity)
  train_data.hist(figsize = (15,8))
  ```
  *show visual
  
  - **Correlation of Numerical Variables with Target Variable**:
   Calculated correlation coefficients and visualized the correlation heatmap to identify relationships between variables. Correlation coefficients measure the strength and direction of the linear relationship between two variables. They range from -1 to 1, where:
    - **1** indicates a perfect positive linear relationship,
    - **-1** indicates a perfect negative linear relationship,
    - **0** indicates no linear relationship.

    In this analysis, `median_income` emerged as the most positively correlated with `median_house_value`, with a correlation coefficient of 0.69. This suggests that the income level of the area plays a significant role in determining housing prices. Other notable correlations include `latitude`, which had a negative correlation of -0.14, indicating that houses located further north tend to have lower values, and `total_rooms`, which showed a positive correlation of 0.14, suggesting that more rooms are associated with higher house values.

  ```python
  #now do correlations with target variable
  numerical_train_data = train_data.select_dtypes(include = np.number)
  correlation_matrix = numerical_train_data.corr()
  plt.figure(figsize = (15,8))
  sns.heatmap(correlation_matrix, annot = True, cmap = "YlGnBu")
  ```
  *show visual
  - **Log Transformation of Variables**:
  Applied log transformations to several numerical variables to address right-skewed distributions. Log transformations are often used to normalize skewed data, making it more suitable for modeling. Adding 1 to the values before transformation avoids taking the logarithm of zero, which is undefined.

    After applying the transformations, the histograms of the affected variables displayed more normalized distributions, indicating a reduction in skewness. This normalization helps improve the performance and reliability of statistical and machine learning models.
  ```python
  #many hists are right-skewed (take log)
  train_data['total_rooms'] = np.log(train_data['total_rooms'] +1) #+1 to avoid zero values
  train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] +1)
  train_data['population'] = np.log(train_data['population'] +1)
  train_data['households'] = np.log(train_data['households'] +1)

  train_data.hist(figsize = (15,8)) #the transformed 4 variables look more normal now
  ```
  *show visual
  - **Visualizing Distribution of Categorical Variable**:
  Visualized the distribution of the ocean_proximity values to understand the frequency of each category.
  ```python
  #checked the distribution of categories
  train_data['ocean_proximity'].value_counts() #island has very few counts
  #visualizing the value_counts
  train_data['ocean_proximity'].value_counts().plot(kind='bar')
  plt.xlabel('Ocean Proximity')
  plt.ylabel('Count')
  plt.title('Distribution of Ocean Proximity')
  plt.show()
  ```
  *show visual
  
  - **Applying One-Hot Encoding**:
  Applied one-hot encoding to the `ocean_proximity` variable to convert it into a format suitable for machine learning models. This process includes creating binary columns for each category.
  ```python
  #one hot encoding
  dummies = pd.get_dummies(train_data.ocean_proximity)
  #convert the boolean values to integers (0 and 1)
  dummies = dummies.astype(int)

  #make sure it worked
  print(dummies)

  #add new columns
  #no longer need ocean_proximity
  train_data = train_data.join(dummies).drop('ocean_proximity', axis = 1)
  ```
  - **Correlation Heatmap with One-Hot Encoded Variables**:
  A new heatmap was generated to visualize the correlations between variables after applying one-hot encoding. The heatmap revealed that the ocean_proximity categories <1H OCEAN and INLAND had the strongest correlations with median_house_value, with coefficients of 0.25 and -0.48, respectively. This indicates that a house’s proximity to the ocean has a significant influence on its value, with houses closer to the ocean tending to have higher values, while inland houses tend to have lower values.
  ```python
  plt.figure(figsize = (15,8))
  sns.heatmap(train_data.corr(), annot = True, cmap = "YlGnBu")
  ```
  *show visual
  
  - **Visualization of Coordinates**:
  To investigate how geographical location influences house prices, a scatter plot of `latitude` versus `longitude` was created. This visualization aimed to identify any spatial patterns related to `median_house_value`. The scatter plot showed that higher house values (indicated by red) were concentrated along the coast, while lower house values (indicated by blue) were more frequently observed inland. This pattern corroborates the earlier findings from the correlation heatmap, highlighting the impact of geographical proximity to the coast on house prices.
  ```python
  #visualization of the coordinates first
  plt.figure(figsize = (15,8))
  sns.scatterplot(x = 'latitude', y = 'longitude', data = train_data, 
                  hue = 'median_house_value', palette='coolwarm')
  #red is touching the coast (more expensive), up and to the right is more inland
  ```
  *show visual
  
  - **Feature Engineering**:
  New features were created to provide additional insights into the data. Specifically, the `bedroom_ratio` was calculated as the ratio of `total_bedrooms` to `total_rooms`, and `rooms_per_household` was derived as the ratio of `total_rooms` to the number of `households`.  These additional features were intended to provide more insightful information that could improve model performance.
  ```python
  #create two new columns
  train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
  train_data['rooms_per_household'] = train_data['total_rooms'] / train_data['households']
  ```
  - **Correlation Heatmap with Engineered Features**:
   A new heatmap was created to visualize the correlations between the newly engineered features and the target variable. The updated correlations indicated that while some original features had relatively weak relationships with the target variable, the engineered features exhibited more significant correlations. Specifically:
    - `total_rooms` showed a correlation of 0.15 with median_house_value.
    - `total_bedrooms` had a correlation of 0.048.
    - `bedroom_ratio` had a correlation of -0.20, suggesting that houses with a higher proportion of bedrooms relative to total rooms, possibly due to conversions of other spaces into bedrooms, might influence home values.
    - `households` showed a correlation of 0.067.
    - `rooms_per_household` had a correlation of 0.11.
      
    These observations suggest that the engineered features, particularly bedroom_ratio, might better capture aspects of the housing data that affect values, such as the impact of converted rooms or garages on home prices.

  ```python
  #correlation heatmap with newly engineered features
  plt.figure(figsize = (15,8))
  sns.heatmap(train_data.corr(), annot = True, cmap = "YlGnBu")
  ```
  *show visual
  
  - **Scaling Data**
  To ensure that all features contributed equally to the model training, data scaling was performed. Scaling standardizes the range of the features, which is particularly important for algorithms that are sensitive to feature magnitudes.

    Since new features were added during previous steps, the dataset was re-split into feature variables (X_train) and the target variable (y_train). The StandardScaler was then applied to standardize the features in the training data. This process transforms the features so they have a mean of 0 and a standard deviation of 1, which helps in improving the performance and convergence of many machine learning algorithms.

  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  
  #we don't have to do the train test split again, but since we added more features
  #we have to do the x y split again
  X_train, y_train = train_data.drop('median_house_value', axis = 1),
  train_data['median_house_value']
  #scale the train data
  X_train_s = scaler.fit_transform(X_train)
  ```
  *start here later

## Model Training
### Linear Regression
- Implementation and evaluation of the Linear Regression model

### Random Forest
- Implementation and evaluation of the Random Forest model

## Hyperparameter Tuning
- Use of RandomizedSearchCV to find optimal parameters for the Random Forest model

## Model Evaluation
- Comparison of R^2, RMSE, and MAE metrics for different models
- Visualization of predicted vs. actual values

## Feature Importance
- Analysis of feature importance using the Random Forest model
- Visualization of feature importances

## Visualizations
- Scatter plots of predicted vs. actual values
- Bar plots for feature importances

## Insights and Interpretations
- Key findings from the model predictions and data analysis
- Insights into the relationships between features and target variable

## Project Goals Met
- Summary of the goals set at the beginning of the project
- Discussion on how the project objectives were achieved

## Limitations
- Limitations encountered during the analysis
- Challenges faced and their impact on the results

## Future Steps
- Suggestions for further improvements
- Potential extensions and additional analyses

## Acknowledgements
- Credits to any resources, libraries, or individuals who contributed to the project

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, scikit-learn, [other libraries used]

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
