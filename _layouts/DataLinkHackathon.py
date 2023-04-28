import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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
# Define variables X and Y
X = df.iloc[:, :-1]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
