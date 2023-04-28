import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

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


# Entrenamiento del modelo: Regresión logística
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

modelo = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Predecir los valores de la variable de salida para los datos de prueba
y_pred = modelo.predict(X_test)



# Definir variables X e y
X = df.iloc[:, :-1] # Todas las variables excepto la última
y = df.iloc[:, -1] # Variable objetivo "Loan_Status"

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar modelo de regresión logística
model = LogisticRegression()

# Ajustar modelo con datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones con datos de prueba
y_pred = model.predict(X_test)

# Evaluar rendimiento del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))