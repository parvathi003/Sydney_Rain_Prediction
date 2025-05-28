Project Overview
This project, titled "ABC-XYZ Inventory Classification using Machine Learning," was designed to enhance inventory management by categorising items based on their sales value and demand variability. The goal was to help businesses prioritise stock control and improve decision-making processes.
Methodology and Results
Using the CRISP-DM framework, the project involved:
Business Understanding: Identifying the need for better inventory classification to optimise stock management.
Data Understanding and Preparation: Analysing a dataset of 1,000 items with details like monthly demand and sales value, the dataset was cleaned by checking for missing values and duplicates. Features were engineered for ABC (using sales value quantiles) and XYZ (using Coefficient of Variation) classifications, combined into a single feature. Numerical features were standardised using StandardScaler, and the dataset was split into 80% training and 20% testing sets..
Key Technologies Used: The project leveraged Python with libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn, executed in a Google Colab environment. Machine learning models included Decision Tree, Random Forest, K-Nearest Neighbour (KNN), Logistic Regression, Gradient Boosting, Support Vector Machine (SVM), and Naive Bayes. Data preprocessing involved the use of StandardScaler and LabelEncoder, with evaluation metrics including accuracy, precision, recall, F1-score, and a confusion matrix.
Modelling: Implementing various machine learning models, including Decision Tree and Random Forest, with Decision Tree achieving 98% accuracy.
Evaluation: Assessing models using metrics like Accuracy, Precision, and F1-score, confirming the Decision Tree's effectiveness.

Project Overview
This project, titled "Predicting the Sales of Products of a Retail Chain using Supervised Learning," predicts future sales for a large Indian retail chain in Maharashtra, Telangana, and Kerala. The goal is to enhance inventory management, reduce costs, and improve strategic planning through accurate sales forecasts.
Methodology and Results
The project followed the CRISP-DM framework, involving:
Business Understanding: Identifying the need for sales forecasting to optimize inventory and decision-making.
Data Preparation: Using datasets like train data, test data, product prices, and date mappings, with features like date, product ID, and sales. Data was cleaned, merged, and split into 70% training and 30% testing sets.
Modeling: Applied models like Linear Regression, Decision Tree, and Random Forest, with Random Forest achieving the best results due to the lowest Root Mean Squared Error (RMSE).
Evaluation: Used metrics like RMSE, Mean Absolute Error (MAE), and R², confirming Random Forest's effectiveness.
Key Technologies
The project used Python with libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn, run in Google Colab for execution.


Project Overview
This project, titled "Rainfall Prediction in Sydney Using Machine Learning," focuses on developing predictive models to forecast whether it will rain tomorrow in Sydney based on historical weather data from 2008 to 2017. The goal is to enhance decision-making in sectors like agriculture and transportation by improving prediction accuracy using ensemble machine learning techniques.
Methodology and Results
The project followed a structured approach:
Data Preparation: The dataset was cleaned by handling missing values (e.g., median imputation for numerical columns) and converting categorical variables to numerical formats. The date was split into year, month, and day, and features were standardized.
Modeling: Three models were used—Decision Tree, Random Forest, and Gradient Boosting. Random Forest performed best with 82.63% accuracy, compared to 75.90% for Decision Tree and 82.34% for Gradient Boosting.
Evaluation: Models were assessed using accuracy scores and confusion matrices, with Random Forest selected for its robustness.
Technologies Used
Python with libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
Executed in Google Colab for cloud-based processing.

