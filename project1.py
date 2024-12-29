from flask import Flask, render_template
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data (standardization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test)) * 100
dt_conf_matrix = confusion_matrix(y_test, dt_model.predict(X_test))

# Train Logistic Regression Classifier
lr_model = LogisticRegression(max_iter=200, random_state=38)
lr_model.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test)) * 100
lr_conf_matrix = confusion_matrix(y_test, lr_model.predict(X_test))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Visualization and result route
@app.route('/predict', methods=['POST'])
def predict():
    # Select a random test sample
    random_index = np.random.randint(0, X_test.shape[0])
    input_data = X_test[random_index].reshape(1, -1)
    true_class = y_test[random_index]

    # Predictions
    dt_class_prediction = dt_model.predict(input_data)[0]
    lr_class_prediction = lr_model.predict(input_data)[0]
    formatted_input_data = [abs(int(round(x))) for x in input_data[0]]


    # Create bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(['Decision Tree Accuracy', 'Logistic Regression Accuracy'], [dt_accuracy, lr_accuracy], color=['blue', 'green'])
    plt.title("Comparison of Classifier Accuracies")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 120)
    for i, v in enumerate([dt_accuracy, lr_accuracy]):
        plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=12)

    # Convert bar plot to PNG and encode it
    bar_img = io.BytesIO()
    plt.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_plot_url = base64.b64encode(bar_img.getvalue()).decode()

    # Create confusion matrix plot for Decision Tree
    plt.figure(figsize=(8, 5))
    sns.heatmap(dt_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Decision Tree Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    dt_cm_img = io.BytesIO()
    plt.savefig(dt_cm_img, format='png')
    dt_cm_img.seek(0)
    dt_cm_plot_url = base64.b64encode(dt_cm_img.getvalue()).decode()

    # Create confusion matrix plot for Logistic Regression
    plt.figure(figsize=(8, 5))
    sns.heatmap(lr_conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    lr_cm_img = io.BytesIO()
    plt.savefig(lr_cm_img, format='png')
    lr_cm_img.seek(0)
    lr_cm_plot_url = base64.b64encode(lr_cm_img.getvalue()).decode()

    return render_template(
        'result.html',
        input_data=input_data[0],
        true_class=true_class,
        dt_class_prediction=dt_class_prediction,
        lr_class_prediction=lr_class_prediction,
        dt_accuracy=f"{dt_accuracy:.2f}%",
        lr_accuracy=f"{lr_accuracy:.2f}%",
        bar_plot_url=bar_plot_url,
        dt_cm_plot_url=dt_cm_plot_url,
        lr_cm_plot_url=lr_cm_plot_url
    )

if __name__ == '__main__':
    app.run(debug=True)
