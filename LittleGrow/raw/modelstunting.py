import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the preprocessed dataset
preprocessed_csv_filename = 'stunted_growth_dataset.csv'
df = pd.read_csv(preprocessed_csv_filename)

# Split the data into features (X) and conditions (y)
X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features (optional, depending on the algorithm)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42, n_estimators=100)
classifier.fit(X_train_scaled, y_train)

# Track accuracy during training
train_accuracies = []
test_accuracies = []

for i, estimator in enumerate(classifier.estimators_):
    # Predict on the training set
    y_train_pred = classifier.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)

    # Predict on the test set
    y_test_pred = classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)

    print(f"Tree {i + 1} - Training Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

# Plotting accuracy over trees
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.title('Accuracy Over Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
