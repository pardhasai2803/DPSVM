import pandas as pd
import numpy as np
from numpy import linalg
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC

df = pd.read_csv('diabetes.csv')


X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for i in range(len(y_train)):
    if(y_train.values[i] == 0):
        y_train.values[i] = -1

for i in range(len(y_test)):
    if(y_test.values[i] == 0):
        y_test.values[i] = -1

mean = X_train.mean()
sd = X_train.std()

X_train_normalized = (X_train - mean) / sd
X_test_normalized = (X_test - mean) / sd

class SVM:
    def __init__(self, C, kernel='linear', degree=2, gamma=0.2):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma

    def gen_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2)
        elif self.kernel == 'poly':
            return (1 + np.dot(X1, X2)) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * (np.linalg.norm(X1-X2) ** 2))

    def compute(self, X, y):
        # Solving using Scipy.optimize.minimize
        self.X = X
        self.y = y

        self.get_samples, self.get_features = X.shape
        self.K = np.zeros((self.get_samples, self.get_samples))
        for i in range(self.get_samples):
            for j in range(self.get_samples):
                self.K[i, j] = self.gen_kernel(X[i], X[j])
        def objective(alpha):
            t1 = np.sum(alpha)
            t2 = np.sum(alpha[:, None] * alpha[None, :] * y[:, None] * y[None, :] * self.K)
            t2 = t2*0.5
            t3 = np.dot(alpha, alpha.T)
            t3 = (t3*0.5)/self.C
            return t2+t3-t1

        def stat_cond(alpha):
            return np.dot(alpha, self.y)

        bounds = [(0, self.C) for i in range(self.get_samples)]
        alpha_guess = np.zeros(self.get_samples)
        constraint = {'type': 'eq', 'fun': stat_cond}
        options = {'maxiter': 10}
        res = minimize(objective, alpha_guess, method='SLSQP', bounds=bounds, constraints=constraint, options=options)

        self.alpha = res.x

        sv = self.alpha > 1e-5
        t = np.arange(len(self.alpha))[sv]
        self.num_sv = sum(sv)
        self.support_vectors = self.X[sv]
        self.support_vectors_y = self.y[sv]
        self.alpha = self.alpha[sv]
        self.bias = 0
        for i in range(self.num_sv):
            self.bias += self.support_vectors_y[i]
            self.bias -= np.sum(self.alpha * self.support_vectors_y * self.K[t[i], sv])
        self.bias /= self.num_sv
        if self.kernel == 'linear':
            self.weights = np.zeros(self.get_features)
            for i in range(self.num_sv):
                self.weights += self.alpha[i] * self.support_vectors_y[i] * self.support_vectors[i]
        else:
            self.weights = None

    def predict(self, X):
        if self.weights is not None:
            y_pred = np.dot(X, self.weights) + self.bias
        else:
            y_pred = np.zeros((len(X)))
            for i in range(len(X)):
                val = 0
                for a, sv_y, sv in zip(self.alpha, self.support_vectors_y, self.support_vectors):
                    val += a * sv_y * self.gen_kernel(X[i], sv)
                y_pred[i] = val
            y_pred = y_pred + self.bias

        for i in range(len(y_pred)):
            if y_pred[i] <= 0:
                y_pred[i] = -1
            else:
                y_pred[i] = 1
        return y_pred



X_train_train, X_train_validation, y_train_train, y_train_validation = train_test_split(X_train_normalized, y_train, stratify=y_train, test_size=0.3, random_state=42)

print('Linear Kernel')
for C in [0.1, 1, 10, 100]:
    # print(f'C={C}')
    svm = SVM(C=C, kernel="linear")
    svm.compute(X_train_train.values, y_train_train.values)
    y_pred = np.array(svm.predict(X_train_validation.values))
    accuracy = accuracy_score(y_pred, y_train_validation.values)
    print(f'C={C} Accuracy = {accuracy}')


print('Polynomial Kernel')
for C in [0.1, 1, 10, 100]:
    for deg in [2,3]:
        svm = SVM(C=C, kernel="poly", degree=deg)
        svm.compute(X_train_train.values, y_train_train.values)
        y_pred = np.array(svm.predict(X_train_validation.values))
        accuracy = accuracy_score(y_pred, y_train_validation.values)
        print(f'C={C}, degree={deg} Accuracy = {accuracy}')

print('RBF Kernel')
for C in [0.1, 1, 10, 100]:
    for gam in [0.1,0.2]:
        svm = SVM(C=C, kernel="rbf", gamma=gam)
        svm.compute(X_train_train.values, y_train_train.values)
        y_pred = np.array(svm.predict(X_train_validation.values))
        accuracy = accuracy_score(y_pred, y_train_validation.values)
        print(f'C={C}, gamma={gam} Accuracy = {accuracy}')

svm_L = SVM(C=0.1, kernel="linear")
svm_L.compute(X_train_normalized.values, y_train.values)
pred_L = svm_L.predict(X_test_normalized.values)
accuracy_L = accuracy_score(pred_L, y_test)
print(classification_report(pred_L, y_test))
print(f'Accuracy of our svm for linear kernel = {accuracy_L}')

svm_inbuilt = LinearSVC(C=0.1)
svm_inbuilt.fit(X_train_normalized, y_train)
y_pred_inbuilt = svm_inbuilt.predict(X_test_normalized)
accuracy_inbuilt = accuracy_score(y_pred_inbuilt, y_test)
print(classification_report(y_pred_inbuilt, y_test))
print(f'Accuracy using svm from sklearn = {accuracy_inbuilt}')


svm_P = SVM(C=0.1, kernel="poly", degree=2)
svm_P.compute(X_train_normalized.values, y_train.values)
pred_P = svm_P.predict(X_test_normalized.values)
accuracy_P = accuracy_score(pred_P, y_test)
print(classification_report(pred_P, y_test))
print(f'Accuracy of our svm for polynomial kernel= {accuracy_P}')


svm_R = SVM(C=0.1, kernel="rbf", gamma=0.2)
svm_R.compute(X_train_normalized.values, y_train.values)
pred_R = svm_R.predict(X_test_normalized.values)
accuracy_R = accuracy_score(pred_R, y_test)
print(classification_report(pred_R, y_test))
print(f'Accuracy of our svm for rbf kernel= {accuracy_R}')

