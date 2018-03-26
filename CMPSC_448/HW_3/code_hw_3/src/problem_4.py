from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# load data in LibSVM sparse data format
X_data, y_data = load_svmlight_file("../data/a9a")
X_valid, y_valid = load_svmlight_file("../data/a9a.t")

# Split data to training and test sets
seed = 6
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=seed)

############################
# USING SVM
############################

# Defaults: 84.70%
# svm_model = svm.SVC(C=1.0, kernel='rbf', gamma='auto')

# Using linear: 84.81
svm_model = svm.SVC(C=1.0, kernel='linear', gamma='auto')

# Using Sigmoid: 84.50
# svm_model = svm.SVC(C=1.0, kernel='sigmoid', gamma='auto')

# Using polynomial: 75.95
# svm_model = svm.SVC(C=1.0, kernel='poly', gamma='auto')

# Using linear, increase C: 84.76
# svm_model = svm.SVC(C=1.2, kernel='linear', gamma='auto')

# Using linear, decrease C: 84.80
# svm_model = svm.SVC(C=0.8, kernel='linear', gamma='auto')

svm_model.fit(X_train, y_train)

svm_train_predict = svm_model.predict(X_train)
svm_train_predict = [round(value) for value in svm_train_predict]
svm_train_accuracy = accuracy_score(y_train, svm_train_predict)
print("SVM Training Accuracy: %.2f%%" % (svm_train_accuracy * 100.0))


svm_predict = svm_model.predict(X_test)
svm_predict = [round(value) for value in svm_predict]
svm_accuracy = accuracy_score(y_test, svm_predict)
print("SVM Holdout Accuracy: %.2f%%" % (svm_accuracy * 100.0))

############################
# USING Random Forests
############################

# Defaults: 76.83
# rf_model = RandomForestClassifier(bootstrap=True, max_depth=2, min_impurity_decrease=0.0, min_samples_leaf=1,
#                                   n_estimators=10)

# Increased depth: 79.16
# rf_model = RandomForestClassifier(bootstrap=True, max_depth=3, min_impurity_decrease=0.0, min_samples_leaf=1,
#                                   n_estimators=10)

# Increased depth and n_estimators: 79.87
# rf_model = RandomForestClassifier(bootstrap=True, max_depth=3, min_impurity_decrease=0.0, min_samples_leaf=1,
#                                   n_estimators=15)

# Increased depth more, increased n_estimators: 82.29
# rf_model = RandomForestClassifier(bootstrap=True, max_depth=5, min_impurity_decrease=0.0, min_samples_leaf=1,
#                                   n_estimators=15)

# Further increased depth, default estimators: 84.14
# rf_model = RandomForestClassifier(bootstrap=True, max_depth=15, min_impurity_decrease=0.0, min_samples_leaf=1,
#                                   n_estimators=10)

# Further increased depth, default estimators, increased min impurity: 84.17
# rf_model = RandomForestClassifier(bootstrap=True, max_depth=15, min_impurity_decrease=0.00001, min_samples_leaf=1,
#                                   n_estimators=10)

# Further increased depth, default estimators, increased min impurity, increased min samples: 84.
rf_model = RandomForestClassifier(bootstrap=True, max_depth=15, min_impurity_decrease=0.00001, min_samples_leaf=2,
                                  n_estimators=10)


rf_model.fit(X_train, y_train)

rf_train_predict = rf_model.predict(X_train)
rf_train_predict = [round(value) for value in rf_train_predict]
rf_train_accuracy = accuracy_score(y_train, rf_train_predict)
print("RF Training Accuracy: %.2f%%" % (rf_train_accuracy * 100.0))


rf_predict = rf_model.predict(X_test)
rf_predict = [round(value) for value in rf_predict]
rf_accuracy = accuracy_score(y_test, rf_predict)
print("RF Holdout Accuracy: %.2f%%" % (rf_accuracy * 100.0))

############################
# Validation Tests
############################
svm_valid_predict = svm_model.predict(X_valid)
svm_valid_predict = [round(value) for value in svm_valid_predict]
svm_valid_accuracy = accuracy_score(y_valid, svm_valid_predict)

rf_valid_predict = rf_model.predict(X_valid)
rf_valid_predict = [round(value) for value in rf_valid_predict]
rf_valid_accuracy = accuracy_score(y_valid, rf_valid_predict)
print("SVM Validation Accuracy: %.2f%%" % (svm_valid_accuracy * 100.0))
print("RF Validation Accuracy: %.2f%%" % (rf_valid_accuracy * 100.0))
