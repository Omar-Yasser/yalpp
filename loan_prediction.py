import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("loan_data.csv")
data.head()

data.info()

print(data['Gender'].value_counts(), '\n')
print(data['Married'].value_counts(), '\n')
print(data['Dependents'].value_counts(), '\n')
print(data['Education'].value_counts(), '\n')
print(data['Self_Employed'].value_counts(), '\n')
print(data['Property_Area'].value_counts(), '\n')
print(data['Loan_Status'].value_counts())

"""# Preprocessing"""

data['Gender'] = data['Gender'].replace(['Female', 'Male'], [0, 1])
data['Married'] = data['Married'].replace(['No', 'Yes'], [0, 1])
data['Dependents'] = data['Dependents'].replace(['0', '1', '2', '3+'], [0, 1, 2, 3])
data['Education'] = data['Education'].replace(['Not Graduate', 'Graduate'], [0, 1])
data['Self_Employed'] = data['Self_Employed'].replace(['No', 'Yes'], [0, 1])
data['Property_Area'] = data['Property_Area'].replace(['Rural', 'Semiurban', 'Urban'], [1, 2, 3])
data['Loan_Status'] = data['Loan_Status'].replace(['N', 'Y'], [0, 1])

data.head()

data.describe()

"""Correlation Matrix"""

corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').format(precision=2)

# Dropping weakly correlated columns

data.drop(['Loan_ID'], axis=1, inplace=True)
data.drop(['Dependents'], axis=1, inplace=True)
data.drop(['Self_Employed'], axis=1, inplace=True)
data.drop(['ApplicantIncome'], axis=1, inplace=True)

data.isnull().sum()

"""Replacing Missing Values"""

# .mode() -> return object and we need the frist val so .mode().iloc[0] or just mode()[0]
data['Gender'].fillna(data.Gender.mode()[0], inplace=True)
data['Married'].fillna(data.Married.mode()[0], inplace=True)
data['LoanAmount'].fillna(data.LoanAmount.std(), inplace=True)
data['Loan_Amount_Term'].fillna(data.Loan_Amount_Term.mode()[0], inplace=True)
data['Credit_History'].fillna(data.Credit_History.mode()[0], inplace=True)

data.isnull().sum()

"""# Splitting Data"""

X = data.drop('Loan_Status', axis=1)
Y = data['Loan_Status']
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1)

"""# Data scaling"""

scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)
x_train.head()

"""# Principal Component Analysis"""

pca = PCA(n_components=3)

X_train = pca.fit_transform(x_train)
X_test = pca.fit_transform(x_test)

explained_variance = pca.explained_variance_ratio_

"""# Logistic Regression
(with PCA)
"""

lr = LogisticRegression(solver='liblinear', random_state=1)
lr = lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)
print("Classification Report: \n", classification_report(y_test, y_pred))

print("Accuracy:", lr.score(X_test, y_test))
print("\nMean Squared Error: ", mean_squared_error(y_test, y_pred))

"""# Logistic Regression
(without PCA)
"""

lr = LogisticRegression(solver='liblinear', random_state=1)
lr = lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("Accuracy:", lr.score(x_test, y_test))
print("\nMean Squared Error: ", mean_squared_error(y_test, y_pred))

"""# SVM"""

classifier = svm.LinearSVC()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)
print("Classification Report: \n", classification_report(y_test, y_pred))

print("Accuracy:", classifier.score(x_test, y_test))
print("\nMean Squared Error: ", mean_squared_error(y_test, y_pred))

"""# Decision Tree"""

clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)
print("Classification Report: \n", classification_report(y_test, y_pred))

print("Accuracy:", clf.score(x_test, y_test))
print("\nMean Squared Error: ", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(25, 10))
a = plot_tree(clf,
              feature_names=x_train.columns,
              filled=True,
              rounded=True,
              fontsize=14)

"""# Random Forest"""

clf = RandomForestClassifier(n_estimators=2000, random_state=1, max_depth=4)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy:", clf.score(x_test, y_test))
print("\nMean Squared Error: ", mean_squared_error(y_test, y_pred))

"""# Naive bayes"""

clf = BernoulliNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy:", clf.score(x_test, y_test))
print("\nMean Squared Error: ", mean_squared_error(y_test, y_pred))

"""# KNN"""

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Accuracy:", knn.score(x_test, y_test))
print("\nMean Squared Error: ", mean_squared_error(y_test, y_pred))
