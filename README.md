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

## Data Preprocessing
- **Initial Setup**
  
  ```python
#importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
data = pd.read_csv("housing.csv")
  ```

- **Handling Missing Values**
Upon inspection, the `total_bedrooms` column had missing values. Out of 20,640 entries, only 207 (approximately 1%) were missing. Since this is less than 5% of the dataset, it is acceptable to drop the rows with missing values.

```python
#check for missing values
data.info()

#drop missing values
data.dropna(inplace = True)
```

- Encoding categorical variables
- Scaling numerical features
- Feature engineering (e.g., creating new features)

## Exploratory Data Analysis
- Statistical summaries
- Data visualizations to understand feature distributions and relationships

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
