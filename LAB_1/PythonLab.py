from asyncio import wait
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "top universities.csv"
df = pd.read_csv(file)

print("Columns: ", df.columns)
print("Dataset size: ", df.shape)
print(df.head())
print(df.info())


print("\nNumber of missing values in each column before processing:")
print(df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna("unknown", inplace=True)
print("\nNumber of missing values in each column after processing:")
print(df.isnull().sum())

df_top100 = df.head(100)
pivot_table = df_top100.pivot_table(
    index="Country",
    columns="City",
    values="Global Rank",
    aggfunc="mean",
    fill_value=0
)

plt.figure(figsize=(20, 10))
sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".1f", cbar_kws={'label': 'Average Global Rank'})
plt.title("Average University Global Rank by Country and City (Top 100 Rows)")
plt.xlabel("City")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

country_avg_rank = df.groupby('Country')['Global Rank'].mean().sort_values().head(20)
plt.figure(figsize=(15, 8))
country_avg_rank.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average University Global Rank by Country')
plt.xlabel('Country')
plt.ylabel('Average Global Rank')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

country_univ_count = df['Country'].value_counts().head(20)
plt.figure(figsize=(15, 8))
country_univ_count.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Number of Universities by Country (Top 20)')
plt.xlabel('Country')
plt.ylabel('Number of Universities')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

ukraine_data = df[df['Country'] == 'Ukraine']
city_univ_count = ukraine_data['City'].value_counts()
plt.figure(figsize=(15, 8))
city_univ_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Universities by City in Ukraine')
plt.xlabel('City')
plt.ylabel('Number of Universities')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

df['Rank_Range'] = pd.cut(
    df['Global Rank'],
    bins=[0, 1000, 2000, 3000, 4000, 5000, float('inf')],
    labels=['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5001+']
)
sns.boxplot(x="Rank_Range", y="Global Rank", data=df, order=['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5001+'])
plt.title('Global Rank Distribution by Rank Ranges')
plt.xlabel('Rank Range')
plt.ylabel('Global Rank')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

categorical_columns = ['Country', 'City']
numeric_columns = ['Global Rank']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[categorical_columns])
X = pd.concat([
    pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns)),
    df[numeric_columns]
], axis=1)
y = df['Global Rank']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)
print("Train features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Train target shape:", y_train.shape)
print("Test target shape:", y_test.shape)

bins = [0, 1000, 5000, 10000, float('inf')]
labels = ['Top 1000', '1001-5000', '5001-10000', '10000+']
df['Rank_Group'] = pd.cut(df['Global Rank'], bins=bins, labels=labels)
categorical_columns = ['Country', 'City']
numeric_columns = []
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[categorical_columns])
X = pd.concat([
    pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns)),
    df[numeric_columns]
], axis=1)
y = df['Rank_Group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
decision_tree = DecisionTreeClassifier(random_state=1)
decision_tree.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn, zero_division=0))
print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nKNN Accuracy Score:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
plt.figure(figsize=(20, 20))
plot_tree(decision_tree, max_depth=4, fontsize=9, feature_names=X.columns, class_names=labels, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

y_pred = decision_tree.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=decision_tree.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

svm_alg = SVC(kernel='linear', random_state=1)
svm_alg.fit(X_train_scaled, y_train)
y_pred = svm_alg.predict(X_test_scaled)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nSVM Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_alg.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("SVM Confusion Matrix")
plt.show()

rand_forest = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=None)
rand_forest.fit(X_train_scaled, y_train)
y_pred = rand_forest.predict(X_test_scaled)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nRandom Forest Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rand_forest.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Random Forest Confusion Matrix")
plt.show()

adaboost = AdaBoostClassifier(n_estimators=50, random_state=1, algorithm='SAMME')
adaboost.fit(X_train_scaled, y_train)
y_pred = adaboost.predict(X_test_scaled)
print("AdaBoost Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nAdaBoost Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = []
k_values = range(1, 40)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy.append(accuracy_score(y_test, y_pred))
best_k = accuracy.index(max(accuracy)) + 1
print(f"Best accuracy: {max(accuracy):.4f} when K = {best_k}")
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy, marker='o', linestyle='-')
plt.title('Accuracy vs. K Value for kNN')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(range(1, 40, 2))
plt.grid()
plt.show()
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("\nOptimal kNN Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nOptimal kNN Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

X_train_small, _, y_train_small, _ = train_test_split(
    X_train_scaled, y_train, test_size=0.8, random_state=1, stratify=y_train
)
print("Original training size:", X_train_scaled.shape[0])
print("Reduced training size:", X_train_small.shape[0])
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
grid = list(ParameterGrid(param_grid))
total_combinations = len(grid)
print(f"Total combinations: {total_combinations}")
progress = 0
best_params = None
best_score = -1
for params in grid:
    progress += 1
    print(f"Progress: {progress}/{total_combinations} ({(progress / total_combinations) * 100:.2f}%)")
    svm_model = SVC(random_state=1, **params)
    svm_model.fit(X_train_small, y_train_small)
    score = svm_model.score(X_test_scaled, y_test)
    if score > best_score:
        best_score = score
        best_params = params
print("\nBest parameters:", best_params)
print(f"Best accuracy: {best_score:.4f}")
best_svm = SVC(random_state=1, **best_params)
best_svm.fit(X_train_scaled, y_train)
y_pred = best_svm.predict(X_test_scaled)
print("\nOptimal SVM Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nOptimal SVM Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svm.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=plt.gca())
plt.title("Optimal SVM Confusion Matrix")
plt.show()
wait(1000)