import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df = pd.read_csv('C:/Users/ivvan/Desktop/Python/credit_traindata.csv')

# Exploratory Data Analysis

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
