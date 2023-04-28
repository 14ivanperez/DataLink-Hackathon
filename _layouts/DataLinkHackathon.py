import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Read CSV file
df = pd.read_csv('C:/Users/ivvan/Desktop/Python/credit_traindata.csv')

# Exploratory Data Analysis
print(df.head())
print(df.describe())
print(df.info())

# Select numerical variables
numeric_vars = ['duration', 'credit_amount', 'employment', 'personal_status', 'num_dependents']

# Distribution graphs
for col in numeric_vars:
    sns.displot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'{col}_distplot.jpg')
    plt.show()

# Correlation graph
corr = df[numeric_vars].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.jpg')
plt.show()

# Descriptive statistics
desc_stats = df[numeric_vars].describe()
desc_stats.to_csv('desc_stats.csv')

#Save descriptive statistics as an image
desc_stats.plot(kind='bar').get_figure().savefig("estadisticas_descriptivas.jpg")


# Divide data in training and test
# Define variables X and Y and separate variable to predict
y = df["class"]

# select all variables except variable to predict
X = df.drop("class", axis=1)

# select categorical variables and apply one-hot codification
from sklearn.preprocessing import OneHotEncoder
categorical_cols = X.select_dtypes(include=["object"]).columns
encoder = OneHotEncoder()
X_cat = encoder.fit_transform(X[categorical_cols])

# select numerical variables
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
X_num = X[numerical_cols]

# put together both types of variables
import scipy.sparse as sp
X_encoded = sp.hstack((X_cat, X_num))

# split data intro training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Model with logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions with testing data
y_pred = model.predict(X_test)

# Count both good and bad loans (logistic regression)
count_good = 0
count_bad = 0
for i in y_pred:
    if i == 'good':
        count_good += 1
    else:
        count_bad += 1

# Produce bar graph of results
plt.bar(['Good', 'Bad'], [count_good, count_bad])
plt.title('Predicted Credit Classification with Logistic Regression')
plt.xlabel('Credit class')
plt.ylabel('Count')
plt.show()

# Decision trees model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Count both good and bad loans (Decision trees)
count_good = 0
count_bad = 0
for i in y_pred_dt:
    if i == 'good':
        count_good += 1
    else:
        count_bad += 1

# Produce bar graph of results
plt.bar(['Good', 'Bad'], [count_good, count_bad])
plt.title('Predicted Credit Classification with Decision trees')
plt.xlabel('Credit class')
plt.ylabel('Count')
plt.show()


#  Model Validation
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
print("Logistic regression")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
le = LabelEncoder()
y = le.fit_transform(y_pred)
y1 = le.fit_transform(y_test) 
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y, y1)
roc_auc_lr = auc(fpr_lr, tpr_lr)
print("Area under ROC curve: ", roc_auc_lr)

print("Decision tree")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
le = LabelEncoder()
y3 = le.fit_transform(y_pred_dt)
y4 = le.fit_transform(y_test) 
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y3, y4)
roc_auc_dt = auc(fpr_dt, tpr_dt)
print("Area under ROC curve:", roc_auc_dt)

