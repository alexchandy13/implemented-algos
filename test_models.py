import csv
import models
from sklearn.model_selection import train_test_split

file = open('BankNote_Authentication.csv')
filereader = csv.reader(file)
X = []
y = []
header = next(filereader)
for row in filereader:
    x = row[:-1]
    for i in range(len(x)):
        x[i] = float(x[i])        
    X.append(x)
    y.append(int(row[-1]))
    
train_size = 0.8
test_size = 0.2
    
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size)

print('----k Nearest Neighbors----')
k_model = models.kNN()
k_model.fit(X_train,y_train)
k_model.predict(X_test,y_test)
print()

print('----Logistic Regression----')
k_model = models.LogisticRegression()
k_model.fit(X_train,y_train)
k_model.predict(X_test,y_test)