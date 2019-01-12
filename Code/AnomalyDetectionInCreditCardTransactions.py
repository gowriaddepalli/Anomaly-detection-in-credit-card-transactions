
# coding: utf-8

# # Anomaly Detection In Credit Card Transactions.
# 
# Trying to achieve the objective of identifying fraudulent credit card transactions using CRISP-DM process.
# 
# - Handle imbalanced dataset using Random Undersampling, Random Oversampling, Random Oversampling using SMOTE analysis.
# - Perform exploratory data analysis for data visualization and feature selection.
# - Cluster Data into fraudulent and non-fraudulent transactions using dimensionality reduction technique with t-distributed Stochastic Neighbour Embedding.
# - Build predictive models using Logistic Regression, K-Nearest Neighbours, Support Vector Machine, Stochastic Gradient Descent.
# - Use Ensemble modelling technique and performed Voting Ensemble, bagging with Random Forest Classifier, boosting with XGBoost and AdaBoost.
# - Perform model evaluation and comparison of the models using Area under the ROC Curve (Accuracy) and time for speed of detection.

# ## Business Understanding
# 
# + Credit cards are used by many customers of various financial institutions to perform various online, ATM transactions. 
# + These transactions can sometimes be fraudulent done by people who aim to gain monetary benefit without authorization. 
# + This leads to financial losses for the banks and creates a sense of mistrust between the bank and customer and could be a major source for banks losing their customers and trust. 
# + Hence, it becomes necessary for financial institutions to identify and hold such transactions accountable for security purposes. 
# + The properties of a good fraud detection system are:
# 
#      1) It should identify the frauds accurately.
#      
#      2) It should detecting the frauds quickly. 
#      
#      3) It should not classify a genuine transaction as fraud.
#      

# In[1]:


# Import the required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import time


# ## Workflow Diagram

# <img src="Workflow_diag.png">

# ## Data Gathering and Preprocessing (Data understanding)

# In[2]:


# Read the data.
df = pd.read_csv("creditcard.csv")
df.shape


# In[3]:


# Showing the sample data.
df.head(5)


# In[4]:


# Checking for null value.
df.isnull().values.any()


# In[5]:


# Adding a column based on labels.
df['ClassLabel'] = np.where(df['Class']==1, 'Fraud', 'Normal')
df.head(5)


# ## Exploratory Data Analysis
# 
# Understanding the data by asking a few questions that helps the business.

# In[6]:


# Splitting the data into different data sets of classes.
Fraud_data = df[df.Class == 1]
Normal_data = df[df.Class == 0]


# In[7]:


# Checking the size of the normal transaction data.
Normal_data.shape


# In[8]:


# Checking the size of the fraudulent transaction data.
Fraud_data.shape


# In[9]:


# function to check percentage of a class 
def calculatePercentage(data):
# now let us check in the number of Percentage
    CountNormal = len(data[data["Class"]==0]) # normal transaction are repersented by 0
    CountFraud = len(data[data["Class"]==1]) # fraud by 1
    NormalTransactionPercentage = CountNormal/(CountNormal+CountFraud)
    print("Normal transacation % is",NormalTransactionPercentage*100)
    FraudTransacationPercentage= CountFraud/(CountNormal+CountFraud)
    print("Fraud transaction %",FraudTransacationPercentage*100)
    
calculatePercentage(df)


# ##### Statistics of the datasets according to the class.

# In[10]:


# Gathering the sense of all data distribution in the fraudulent transaction dataset.
Fraud_data.describe()


# In[11]:


# Gathering the sense of all data distribution in the normal transaction dataset.
Normal_data.describe()


# In[12]:


# Seeing the visual representation of fradulent transactions (Class==1) and normal ones(Class== 0)
df['ClassLabel'].value_counts().plot('bar')


# #### We can see that the data is unbalanced meaning there are very few fraudelent transactions as compared to normal transactional data.

# ### Data Visualization

# In[13]:


# Seeing the visual representation of fraudulent transactions (Class==1) with time.
sns.distplot(df.Time[df.Class == 1],color= 'R');


# In[14]:


# Seeing the visual representation of normal transactions(Class== 0) with time.
sns.distplot(df.Time[df.Class == 0], color= 'G');


# - Observation:  Time doesnt seem to be a significant factor that affects the transactions being classified as fraudulent or normal.

# In[15]:


# Plotting Time and Amount with fraudulent class.
sns.jointplot(x="Time", y="Amount", data=df[df.Class==1]);


# In[16]:


# Plotting Time and Amount with non-fraudulent class.
sns.jointplot(x="Time", y="Amount", data=df[df.Class==0], color='pink');


# In[17]:


# Transaction amount distribution.
bins = np.linspace(200, 2500, 100)
plt.hist(Normal_data.Amount, bins, alpha=1, normed=True, label='Normal', color = 'Red')
plt.hist(Fraud_data.Amount, bins, alpha=0.6, normed=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of transactions.")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions (%)");
plt.show()


# In[18]:


# Transaction Distribution by time.
bins = np.linspace(0, 48, 48) #48 hours
plt.hist((Normal_data.Time/(60*60)), bins, alpha=1, normed=True, label='Normal', color = 'G')
plt.hist((Fraud_data.Time/(60*60)), bins, alpha=0.6, normed=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Percentage of transactions by hour")
plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
plt.ylabel("Percentage of transactions (%)");
plt.show()


# In[19]:


# Correlation between features of the dataset.
correlation = df.corr()
sns.heatmap(correlation, 
            xticklabels=correlation.columns.values,
            yticklabels=correlation.columns.values)


# In[20]:


correlation.style.background_gradient()


# #### Intra Correlation between features of the dataset with respect to different classes.

# In[21]:


#Select only the anonymized features.
Vfeatures = df.iloc[:,1:29].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, feature in enumerate(df[Vfeatures]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[feature][df.Class == 1], bins=50, color = 'Brown')
    sns.distplot(df[feature][df.Class == 0], bins=50, color = 'yellow')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(feature))
plt.show()


# #### From this we can see that features 'V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8' are redundant and do not vary with class 'Fradulent' and 'Normal'.
# #### So the major contributing features that distinguish between the fraudulent transactions and non fraudulent ones are V1-V7, V9, V10, V11, V12, V14, V16- V19, V21

# ### Feature Scaling

# ##### As the other columns are transformed into a standard normal form using PCA we need to transform the columns Time and Amount too into standard normal form.

# Reference: https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
# 
# To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation.
# 
# x′=x−μσ
# 
# You do that on the training set of data. But then you have to apply the same transformation to your testing set (e.g. in cross-validation), or to newly obtained examples before forecast. But you have to use the same two parameters μ and σ (values) that you used for centering the training set.
# 
# Hence, every sklearn's transform's fit() just calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state. Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.
# 
# fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.

# Reference: https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
# 
# The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 
# and standard deviation of 1. Given the distribution of the data, each value in the dataset will have the sample mean value 
# subtracted, and then divided by the standard deviation of the whole dataset.

# In[22]:


# Trying to normalize the values.
NormalizedDF = df
NormalizedDF['Time'] = StandardScaler().fit_transform(NormalizedDF['Time'].values.reshape(-1, 1))
NormalizedDF['Amount'] = StandardScaler().fit_transform(NormalizedDF['Amount'].values.reshape(-1, 1))
NormalizedDF.head(5)


# ### Data Preparation - Dealing with Imbalanced Data.
# 
# We use the following techniques to handle imbalance in the dataset to make it balanced:
# 
# 1. Random Undersampling
# 2. Random Oversampling
# 3. Random Oversampling with SMOTE.
# 
# ###### The need to do balance data:
# 
# 1. It can cause overfitting of the data and assume the major class as the output for the testing set.
# 2. We can fail to understand the correlations between the features due to these anomalies as they are in insignificant amount compared to the major class.

# In[23]:


# Dropping the columns that don't add to the predictive value.
df_dropped = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8', 'Time', 'Amount', 'ClassLabel'], axis =1)
df_drop = df.drop(['ClassLabel'], axis =1)


# In[24]:


# Remove 'class' columns
data_without_labels = df_drop.columns[:-1]
#print(labels)

# Preparing the data for applying the predictive models.

X = df_drop[data_without_labels]
y = df_drop['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[25]:


X.head(5)


# In[26]:


y.head(5)


# In[27]:


X_train.head(5) 


# In[28]:


X_test.head(5)


# In[29]:


y_train.head(5)


# In[30]:


y_test.head(5)


# In[31]:


# Original dataset showing the imbalance in the dataset.
import numpy as np
P = np.asarray(X)
colors = ['green' if v == 0 else 'Red' if v == 1 else 'blue' for v in y]
kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
fig = plt.Figure(figsize=(12,6))
plt.scatter(P[:, 0], P[:, 1], c=colors, **kwarg_params)
sns.despine()


# Random Undersampling may not be better than Random oversampling as we might loose important data, which is maintained in oversampling. Outside of this case however, the performance of the one or the other will be most indistinguishable. 
# Sampling doesn't introduce new information in the dataset, it merely shifts it around so as to increase the "numerical stability" of the resulting models.

# In[32]:


# imblearn implements over-sampling and under-sampling using dedicated classes.
# undersampling the data to balance out the data.
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_undersampled, y_undersampled = rus.fit_resample(X, y)
colors = ['green' if v == 0 else 'Red' if v == 1 else 'blue' for v in y_undersampled]
plt.scatter(X_undersampled[:, 0], X_undersampled[:, 1], c=colors, linewidth=0.5, edgecolor='black')
sns.despine()


# In[33]:


# Getting the count for label =1 in the undersampled data
count = 0
for x in list(y_undersampled):
    if x == 1:
        count=count+1
print(count)


# In[34]:


# Getting the count for label =0 in the oversampled data
count = 0
for x in list(y_undersampled):
    if x == 0:
        count=count+1
print(count)


#  #### We can see that the data has been balanced, and the majority class has been undersampled.

# In[35]:


# imblearn implements over-sampling and under-sampling using dedicated classes.
# oversampling the data to balance out the data.
from imblearn.over_sampling import RandomOverSampler
rusO = RandomOverSampler(random_state=0)
X_oversampled, y_oversampled = rusO.fit_resample(X, y)
colors = ['green' if v == 0 else 'Red' if v == 1 else 'blue' for v in y_oversampled]
plt.scatter(X_oversampled[:, 0], X_oversampled[:, 1], c=colors, linewidth=0.5, edgecolor='black')
sns.despine()


# In[36]:


# Getting the count for label =1 in the undersampled data
count = 0
for x in list(y_oversampled):
    if x == 1:
        count=count+1
print(count)


# In[37]:


# Getting the count for label =1 in the undersampled data
count = 0
for x in list(y_oversampled):
    if x == 1:
        count=count+1
print(count)


#  #### We can see that the data has been balanced, and the majority class has been oversampled.

# In[38]:


# SMOTE ANALYSIS:  Informed Over Sampling: Synthetic Minority Over-sampling Technique

from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_oversampled_sm, y_oversampled_sm = smote.fit_sample(X, y)
colors = ['green' if v == 0 else 'Red' if v == 1 else 'blue' for v in y_oversampled_sm]
plt.scatter(X_oversampled_sm[:, 0], X_oversampled_sm[:, 1], c=colors, linewidth=0.5, edgecolor='black')
sns.despine()


# In[39]:


# Getting the count for label = 1 in the oversampled data
count = 0
for x in list(y_oversampled_sm):
    if x == 1:
        count=count+1
print(count)


# In[40]:


# Getting the count for label = 0 in the oversampled data
count = 0
for x in list(y_oversampled_sm):
    if x == 0:
        count=count+1
print(count)


#  #### We can see that the data has been balanced, and the majority class has been oversampled using SMOTE.

# ###### Reference: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
# 
# #### Few ways of handling imbalanced datasets:
# 1.  Collect more data to remove imbalance.
# 2.  Changing performance metric as accuracy isn't the correct measure and gives the great feeling of your model being good as it majorly predicts the major class as the predicted value. The following could be considered as metrics for imbalanced data.
# - Kappa (or Cohen’s kappa): Classification accuracy normalized by the imbalance of the classes in the data.
# - ROC Curves: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.
# 3. Sampling the dataset helps as follows:
# - You can add copies of instances from the under-represented class called over-sampling.
# - You can delete instances from the over-represented class, called under-sampling.
# 4. Try Generate Synthetic Samples using to randomly sample the attributes from instances in the minority class using SMOTE or the Synthetic Minority Over-sampling Technique. SMOTE is an oversampling method. It works by creating synthetic samples from the minor class instead of creating copies. The algorithm selects two or more similar instances (using a distance measure) and perturbing an instance one attribute at a time by a random amount within the difference to the neighboring instances.
# 5. Try different machine learning algorithms like decision trees etc as decision trees often perform well on imbalanced datasets. The splitting rules that look at the class variable used in the creation of the trees, can force both classes to be addressed.
# 6. Trying penalised models. Penalized classification imposes an additional cost on the model for making classification mistakes on the minority class during training. These penalties can bias the model to pay more attention to the minority class. 
# 7. Change in perspective.
#  - Anomaly detection is the detection of rare events. This might be a machine malfunction indicated through its vibrations or a malicious activity by a program indicated by it’s sequence of system calls. The events are rare and when compared to normal operation.This shift in thinking considers the minor class as the outliers class which might help you think of new ways to separate and classify samples.
# - Change detection is similar to anomaly detection except rather than looking for an anomaly it is looking for a change or difference. This might be a change in behavior of a user as observed by usage patterns or bank transactions.

# ### Applying predictive models on imbalanced data

# In[41]:


# Applying K Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)


# In[42]:


accuracyKNN = accuracy_score(y_test, y_pred)
print("Accuracy of KNN: %.2f%%" % (accuracyKNN * 100.0))


# In[43]:


aucKNN = roc_auc_score(y_test, y_pred)
print("Area under the ROC of KNN: %.4f" % (aucKNN))


# In[44]:


# Applying SVM
from sklearn.svm import SVC
SVMclf = SVC()
SVMclf.fit(X_train, y_train)
y_pred = SVMclf.predict(X_test)


# In[45]:


accuracySVM = accuracy_score(y_test, y_pred)
print("Accuracy of SVM: %.2f%%" % (accuracySVM * 100.0))


# In[46]:


aucSVM = roc_auc_score(y_test, y_pred)
print("Area under the ROC of SVM: %.4f" % (aucSVM))


# In[47]:


# Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
Logisticclf = LogisticRegression(random_state=0, solver='lbfgs')
Logisticclf.fit(X_train, y_train)
y_pred = Logisticclf.predict(X_test)


# In[48]:


accuracyLR = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression: %.2f%%" % (accuracyLR * 100.0))


# In[49]:


aucLR = roc_auc_score(y_test, y_pred)
print("Area under the ROC of Logistic Regression: %.4f" % (aucLR))


# In[50]:


from sklearn import linear_model
SGDclf = linear_model.SGDClassifier(max_iter=1000)
SGDclf.fit(X_train, y_train)
y_pred = SGDclf.predict(X_test)


# In[51]:


accuracySGD = accuracy_score(y_test, y_pred)
print("Accuracy of Stochastic Gradient Descent: %.2f%%" % (accuracySGD * 100.0))


# In[53]:


aucSGD = roc_auc_score(y_test, y_pred)
print("Area under the ROC of Stochastic Gradient Descent: %.4f" % (aucSGD))


# ### Predictive Modelling with various machine learning algorithms on balanced data using Random Oversampling with SMOTE.

# We will be using the data generated with Random oversampling with SMOTE as Random Undersampling may not be better than Random oversampling as we might loose important data, which is maintained in oversampling. Outside of this case however, the performance of the one or the other will be most indistinguishable. 
# Sampling doesn't introduce new information in the dataset, it merely shifts it around so as to increase the "numerical stability" of the resulting models.
# 
# Of all the sampling techniques, Random Oversampling with SMOTE is the one that offers the best advantages.

# In[54]:


# Applying K Nearest Neighbour with balanced data.
from sklearn.neighbors import KNeighborsClassifier
t0 = time.time()
neighROS = KNeighborsClassifier(n_neighbors=3)
neighROS.fit(X_oversampled_sm, y_oversampled_sm)
y_predKNN = neighROS.predict(X_test)
t1 = time.time()
print("KNN took {:.2} s".format(t1 - t0))


# In[55]:


accuracyKNNROS = accuracy_score(y_test, y_predKNN)
print("Accuracy of KNN on balanced data is : %.2f%%" % (accuracyKNNROS * 100.0))


# In[56]:


aucKNNRoS = roc_auc_score(y_test, y_predKNN)
print("Area under the ROC of KNN on balanced data is : %.4f" % (aucKNNRoS))


# In[57]:


# Applying SVM
from sklearn.svm import SVC
t0 = time.time()
SVMclfROS = SVC()
SVMclfROS.fit(X_oversampled_sm, y_oversampled_sm)
y_predSVM = SVMclfROS.predict(X_test)
t1 = time.time()
print("SVM took {:.2} s".format(t1 - t0))


# In[58]:


accuracySVMRoS = accuracy_score(y_test, y_predSVM)
print("Accuracy of SVM on balanced data is: %.2f%%" % (accuracySVMRoS * 100.0))


# In[59]:


aucSVMRoS = roc_auc_score(y_test, y_predSVM)
print("Area under the ROC of SVM on balanced data is: %.4f" % (aucSVMRoS))


# In[60]:


# Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
t0 = time.time()
LogisticclfROS = LogisticRegression(random_state=0, solver='lbfgs')
LogisticclfROS.fit(X_oversampled_sm, y_oversampled_sm)
y_predLR = LogisticclfROS.predict(X_test)
t1 = time.time()
print("Logistic Regression took {:.2} s".format(t1 - t0))


# In[61]:


accuracyLRROS = accuracy_score(y_test, y_predLR)
print("Accuracy of Logistic Regression on balanced data is: %.2f%%" % (accuracyLRROS * 100.0))


# In[62]:


aucLR = roc_auc_score(y_test, y_predLR)
print("Area under the ROC of Logistic Regression on balanced data is: %.4f" % (aucLR))


# In[63]:


from sklearn import linear_model
t0 = time.time()
SGDclfRoS = linear_model.SGDClassifier(max_iter=1000)
SGDclfRoS.fit(X_oversampled_sm, y_oversampled_sm)
y_predSGD = SGDclfRoS.predict(X_test)
t1 = time.time()
print("SGD took {:.2} s".format(t1 - t0))


# In[64]:


accuracySGDRoS = accuracy_score(y_test, y_predSGD)
print("Accuracy of Stochastic Gradient Descent on balanced data is: %.2f%%" % (accuracySGDRoS * 100.0))


# In[65]:


aucSGDRos = roc_auc_score(y_test, y_predSGD)
print("Area under the ROC of Stochastic Gradient Descent on balanced data is: %.4f" % (aucSGDRos))


# In[71]:


# Drawing the confusion  matrix
from sklearn.metrics import confusion_matrix
def plot_conf_matrix(y_test_data, y_pred_data, classifier):
    labels = [0, 1]
    cm = confusion_matrix(y_test_data, y_pred_data, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the %s classifier'% classifier)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    plt.show()


# In[72]:


plot_conf_matrix(y_test, y_predKNN, "K Nearest Neighbour")


# In[73]:


plot_conf_matrix(y_test, y_predSVM, "Support Vector Machine")


# In[74]:


plot_conf_matrix(y_test, y_predLR, "Logistic Regression")


# In[76]:


plot_conf_matrix(y_test, y_predSGD, "Stochastic Gradient Descent")


# ### Feature Importance

# ### Clustering using Dimensionality Reduction.

# In[77]:


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches


# In[78]:


# T-SNE with Original Data.
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)


# In[79]:


plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
plt.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='rainbow', label='Fraud', linewidths=2)
plt.legend(handles=[mpatches.Patch(color='#0A0AFF', label='No Fraud'), mpatches.Patch(color='#AF0000', label='Fraud')])
plt.title('t-SNE with normal data', fontsize=14)
plt.show()


# In[80]:


# T-SNE with Random undersampled balanced data
X_reduced_tsne_undersample = TSNE(n_components=2, random_state=42).fit_transform(X_undersampled)


# In[82]:


# T-SNE with Undersampled Data.
plt.scatter(X_reduced_tsne_undersample[:,0], X_reduced_tsne_undersample[:,1], c=(y_undersampled == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
plt.scatter(X_reduced_tsne_undersample[:,0], X_reduced_tsne_undersample[:,1], c=(y_undersampled == 1), cmap='coolwarm', label='Fraud', linewidths=2)
plt.legend(handles=[mpatches.Patch(color='#0A0AFF', label='No Fraud'), mpatches.Patch(color='#AF0000', label='Fraud')])
plt.title('t-SNE with undersampled data', fontsize=14)
plt.show()


# ### Ensemble Learning
# 
# The main objective of ensemble methodology is to improve the performance of single classifiers. The approach involves constructing several two stage classifiers from the original data and then aggregate their predictions.

# ### Bagging
# 
# - Random Forest

# In[83]:


# Type of bagging ensemble learning with SMOTE data.
from sklearn.ensemble import RandomForestClassifier 
t0 = time.time()
forest = RandomForestClassifier(n_estimators = 100)
forest.fit(X_oversampled_sm, y_oversampled_sm)
y_predRF = forest.predict(X_test)
t1 = time.time()
print("Random Forest took {:.2} s".format(t1 - t0))
#print('Score: ', forest.score(X_test, y_test))


# In[84]:


accuracyRFRoS = accuracy_score(y_test, y_predRF)
print("Accuracy of Random Forest on balanced data is: %.2f%%" % (accuracyRFRoS * 100.0))


# In[85]:


aucRFRos = roc_auc_score(y_test, y_predRF)
print("Area under the ROC of Random Forest on balanced data is: %.4f" % (aucRFRos))


# In[86]:


plot_conf_matrix(y_test, y_predRF, "Random Forest")


# ### Boosting:
# - XGBoost
# - Ada- Boost

# In[131]:


# Converting numpy array to Dataframe for Xgboost.
X_oversampleDF_sm = pd.DataFrame(data=X_oversampled_sm[0:,0:],   
             columns=X_train.columns)  


# In[133]:


# Converting numpy array to Dataframe for Xgboost.
y_oversampleDF_sm = pd.DataFrame(data=y_oversampled_sm) 


# In[134]:


# Type of boosting ensemble learning with SMOTE data.
from xgboost import XGBClassifier
t0 = time.time()
XGBmodel = XGBClassifier()
XGBmodel.fit(X_oversampleDF_sm, y_oversampleDF_sm)
y_predXGB = XGBmodel.predict(X_test)
t1 = time.time()
print("Xtreme Gradient Boosting took {:.2} s".format(t1 - t0))


# In[135]:


accuracyXGBRoS = accuracy_score(y_test, y_predXGB)
print("Accuracy of Xtreme Gradient Boosting on balanced data is: %.2f%%" % (accuracyXGBRoS * 100.0))


# In[136]:


aucXGBRos = roc_auc_score(y_test, y_predXGB)
print("Area under the ROC of Xtreme Gradient Boosting on balanced data is: %.4f" % (aucXGBRos))


# In[137]:


plot_conf_matrix(y_test, y_predXGB, "Xtreme Gradient Boosting")


# In[97]:


from sklearn.ensemble import AdaBoostClassifier
t0 = time.time()
ada = AdaBoostClassifier(n_estimators = 100)
ada.fit(X_oversampled_sm, y_oversampled_sm)
y_predAda = ada.predict(X_test)
t1 = time.time()
print("Ada Boosting took {:.2} s".format(t1 - t0))


# In[98]:


accuracyAdaRoS = accuracy_score(y_test, y_predAda)
print("Accuracy of Ada Boosting on balanced data is: %.2f%%" % (accuracyAdaRoS * 100.0))


# In[99]:


aucAdaRos = roc_auc_score(y_test, y_predAda)
print("Area under the ROC of Ada Boosting on balanced data is: %.4f" % (aucAdaRos))


# In[100]:


plot_conf_matrix(y_test, y_predAda, "Ada Boosting")


# ### Voting Ensemble

# In[102]:


from sklearn.ensemble import VotingClassifier


# In[105]:


t0 = time.time()
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = linear_model.SGDClassifier(max_iter=1000)
estimators.append(('SGD', model2))
model3 = SVC()
estimators.append(('SVM', model3))
model4 = KNeighborsClassifier(n_neighbors=3)
estimators.append(('KNN', model4))
# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_oversampled_sm, y_oversampled_sm)
y_predEnsemble = ensemble.predict(X_test)
t1 = time.time()
print("Voting Ensemble took {:.2} s".format(t1 - t0))


# In[106]:


accuracyEnsembleRoS = accuracy_score(y_test, y_predEnsemble)
print("Accuracy of voting ensemble on balanced data is: %.2f%%" % (accuracyEnsembleRoS * 100.0))


# In[107]:


aucEnsembleRos = roc_auc_score(y_test, y_predEnsemble)
print("Area under the ROC of ensemble learning on balanced data is: %.4f" % (aucEnsembleRos))


# In[187]:


plot_conf_matrix(y_test, y_predEnsemble, "Voting Ensemble")


# ### Comparison of various classification models with respect to Area Under the ROC Curve.
# 
# <img src="accRoc.png">

# ###  Comparison of various classification models with respect to Time (Speed).
# <img src="graph.png">

# ## Feature Importance

# In[108]:


feature_weight = []
feature_names = []
feature_weight_name = {}
for i, j in sorted(zip(X.columns, forest.feature_importances_)):
    feature_weight.append(i)
    feature_names.append(j)
    feature_weight_name[i] = j

plt.figure(figsize = (17,5))
plt.bar(feature_weight, feature_names, align='center', alpha=0.2)
plt.ylabel('Feature Weight')
plt.title('Feature Weightage')
plt.show()


# ## Model Evaluation (Comparisons of various models)

# In[150]:


from sklearn.metrics import roc_curve, auc

def getROCcurve(test_data, predicted_data, clasModel):
    fpr, tpr, thresholds = roc_curve(test_data,predicted_data)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for %s' % clasModel)
    plt.legend(loc="lower right")
    plt.show()


# In[159]:


knnROC = getROCcurve(y_test, y_predKNN, 'K Nearest Neighbour')


# In[160]:


svmROC = getROCcurve(y_test, y_predSVM, 'Support Vector Machine')


# In[170]:


lrROC = getROCcurve(y_test, y_predLR, "Logistic Regression")


# In[171]:


sgdROC = getROCcurve(y_test, y_predSGD, "Stochastic Gradient Descent")


# In[172]:


rfROC = getROCcurve(y_test, y_predRF, 'Random Forest')


# In[173]:


xgbROC = getROCcurve(y_test, y_predXGB, 'Xtreme Gradient Boosting')


# In[174]:


adaROC = getROCcurve(y_test, y_predAda, 'Ada Boosting')


# In[175]:


veROC = getROCcurve(y_test, y_predEnsemble, 'Voting Ensemble')


# ### Conclusions:
# 
# - In most cases, synthetic techniques like SMOTE and MSMOTE will outperform the conventional oversampling and undersampling methods.
# - For better results, one can use synthetic sampling methods like SMOTE and MSMOTE along with advanced boosting methods like Gradient boosting and XG Boost.

# ### References:
# 
# https://www.infoworld.com/article/2907877/machine-learning/how-paypal-reduces-fraud-with-machine-learning.html
# https://www.3pillarglobal.com/insights/credit-card-fraud-detection
# https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd
# https://www.datascience.com/blog/fraud-detection-with-tensorflow
# https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/
# https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
# https://www.kaggle.com/gargmanish/how-to-handle-imbalance-data-study-in-detail
# https://weiminwang.blog/2017/06/23/credit-card-fraud-detection-using-auto-encoder-in-tensorflow-2/
# https://www.kaggle.com/azzion/credit-card-fraud-detection-using-neural-network
# https://www.linkedin.com/pulse/analyzing-transaction-data-like-scientist-taha-mokfi/
# https://datascience.stackexchange.com/questions/32818/train-test-split-of-unbalanced-dataset-classification
# https://github.com/phatak-dev/spark-ml-kaggle/blob/master/python/credit_card_class_imbalance.ipynb
# https://www.youtube.com/watch?v=m-S9Hojj1as
# https://www.youtube.com/watch?v=EuBBz3bI-aA
# https://www.datascience.com/blog/fraud-detection-with-tensorflow
# https://towardsdatascience.com/detecting-financial-fraud-using-machine-learning-three-ways-of-winning-the-war-against-imbalanced-a03f8815cce9
# https://www.kaggle.com/zhouhq/credit-fraud-detection-the-power-of-ensemble/notebook
# https://github.com/nilaysen/Credit-Card-Fraud-Detection/blob/master/credit_card_fraud.ipynb
# https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
# https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
# https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
# https://qiita.com/bmj0114/items/460424c110a8ce22d945
