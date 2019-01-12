Procedure to run this code:

The following softwares and libraries need to be installed:

1. Ananconda.
2. Python 3.6.
3. Pandas for data manipulation.
4. imblearn for data balancing.
5. Scikit for Learn for predictive modelling for machine learning algorithms and evaluation. 
6. NumPy for data manipulation.
7. Matplotlib and Seaborn for Data Visualization. 

These can be done using the anaconda navigator -> Environment-> search the uninstalled libraries and install it or use pip by going C:\Anaconda\Scripts.

Once these libraries are installed post the Anaconda environment setup:

1. Start the anaconda console with 3.6 version.
2. Go to folder with 'code' subfolder and run the 'AnomalyDetectionInCreditCardTransactions' ipynb.
3. Do not run each cell, as many cells have predictive models developed and they would take a lot of time to build. 
Hence, the notebook with data and required explanation are already saved with output.
4. Also, then do not remove the or change contents of any folder.

Project Workflow - CRISP DM:

1. Business Understanding: Build a model using the credit card transactional data that can help a financial institution predict fraudulent transactions.
2. Data Understanding: 492 fake credit card transactions out 284,807 transactions so need to perform Data balancing.
3. Data Preparation: Data is cleaned, processed and prepared using ETL and EDA.
4. Modelling: Feature selection and iterating over models to select machine learning algorithms that are a good fit to give a predictive power. 
5. Model evaluation: Select which model suits the business requirement the best using various mathematical measures like Accuracy, AUC etc.
6. Model deployment : Deployed in the Anaconda Ipython Notebook.

Data Understanding: (Data/creditcard.csv)

Dataset:  https://www.kaggle.com/mlg-ulb/creditcardfraud
COLUMNS
Time - Number of seconds elapsed between this transaction and the first transaction in the dataset.
Amount - Transaction amount. (Not Transformed Data)
V1, V2, V3…V28 – PRINCIPAL COMPONENTS obtained through PCA (Renamed for security).
Class1 for fraudulent transactions, 0 otherwise
It’s a classification model.

Model Evaluation techniques:

1. Random Forest
2. K Nearest Neighbour
3. Support Vector Machine
4. XGBoost
5. AdaBoost
6. Voting Ensemble
7. Stochastic Gradient Descent
8. Logistic Regression. 


Team members: ( Please contact incase og any enquiries)
1.  Sree Gowri Addepalli - sga297@nyu.edu
2.  Sree Lakshmi Addepalli - sla410@nyu.edu

