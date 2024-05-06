# AI-Mini-Project
Heart Disease Detection using Data Analysis and Machine Learning
This script appears to be an exploratory data analysis (EDA) and machine learning pipeline for a heart disease dataset. Let's break down the steps:
1. Importing Libraries: The code starts by importing necessary libraries such as pandas for data manipulation, numpy for numerical operations, matplotlib and seaborn for visualization, and scikit-learn for machine learning functionalities.
2. Data Acquisition and Exploration: The dataset (heart.csv) is loaded using pandas' read_csv function. Basic exploratory analysis is performed:
    * data.head(): Displays the first few rows of the dataset.
    * data.shape: Prints the dimensions of the dataset (number of rows and columns).
    * data.columns: Lists the column names.
    * data.info(): Provides information about the dataset, including data types and missing values.
    * data.describe().transpose(): Generates summary statistics for numerical columns.
    * data.isnull().sum(): Counts missing values in each column.
    * sns.heatmap(data.isnull(), cmap='magma', cbar=False): Visualizes missing values using a heatmap.
3. Data Visualization:
    * Heatmaps are used to visualize the mean values of certain features for individuals with and without heart disease.
    * Distributions of categorical and numerical features are plotted.
    * Pie charts show the percentage of individuals with and without heart disease.
    * Various categorical features are analyzed in relation to the presence of heart disease.
    * Scatterplots and stripplots visualize relationships between numerical features and heart disease.
4. Feature Engineering:
    * Categorical features are encoded using LabelEncoder from scikit-learn.
5. Data Scaling:
    * MinMaxScaler and StandardScaler from scikit-learn are used to scale numerical features.
6. Feature Selection:
    * Chi-squared and ANOVA tests are performed to select the most important categorical and numerical features, respectively.
7. Machine Learning Model Building:
    * The dataset is split into training and testing sets.
    * Logistic Regression and RandomForestClassifier models are trained and evaluated using accuracy, cross-validation scores, ROC-AUC scores, and confusion matrices.
Overall, this code provides a comprehensive analysis of the heart disease dataset, including data exploration, visualization, preprocessing, and model building.
