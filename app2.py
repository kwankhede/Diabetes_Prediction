import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# Load and preprocess your dataset (you can modify this as needed)
def load_data(filename):
    original_data = pd.read_csv(filename, sep=",")
    return original_data


def replace_zeros_with_nan(data):
    columns_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    data[columns_to_replace] = data[columns_to_replace].replace(0, np.NaN)
    return data


def fill_missing_values(data, key):
    for var in data.columns[:-1]:
        temp = data[data[var].notnull()]
        temp = temp.groupby([key])[var].median().reset_index()
        data.loc[(data[key] == 0) & (data[var].isnull()), var] = temp[temp[key] == 0][
            var
        ].values[0]
        data.loc[(data[key] == 1) & (data[var].isnull()), var] = temp[temp[key] == 1][
            var
        ].values[0]
    return data


original_data = load_data("diabetes.csv")
original_data = replace_zeros_with_nan(original_data)
original_data = fill_missing_values(original_data, "Outcome")

X = original_data.iloc[:, :-1]
y = original_data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create a dictionary of the classifiers
classifiers_dict = {
    "Logistic Classifier": LogisticRegression(class_weight="balanced"),
    "Decision_Tree Classifier": DecisionTreeClassifier(class_weight="balanced"),
    "Random_Forest Classifier": RandomForestClassifier(class_weight="balanced"),
    "SVM Classifier": SVC(probability=True, gamma="scale"),
    "GaussianNB Classifier": GaussianNB(),
    "KNN Classifiers": KNeighborsClassifier(),
    "GB Classifier": GradientBoostingClassifier(loss="deviance"),
    "XGB Classifier": XGBClassifier(scale_pos_weight=2),
}

# Streamlit app
st.title("Diabetes Prediction Using Machine Learning")
st.markdown(
    "This app predicts whether a person has diabetes or not based on user's input features."
)

# Sidebar with user input features
st.sidebar.header("User Input Features")
st.sidebar.markdown("Enter the following patient information:")

# st.image(
# "https://www.arkanalabs.com/wp-content/uploads/2021/10/diabetes.png",
# use_column_width=True,
# )

# Sidebar with user input features
st.sidebar.header("User Input Features")
st.sidebar.markdown("Enter the following patient information:")

pregnancies = st.sidebar.number_input(
    "Pregnancies (Range: 0 - 20)", min_value=0, max_value=20, value=0, step=1
)
glucose = st.sidebar.number_input(
    "Glucose (Range: 44 - 300 mg/dL)", min_value=44, max_value=300, value=120, step=1
)
blood_pressure = st.sidebar.number_input(
    "Blood Pressure (Range: 24 - 222 mm Hg)",
    min_value=24,
    max_value=222,
    value=120,
    step=1,
)
skin_thickness = st.sidebar.number_input(
    "Skin Thickness (Range: 7 - 199 mm)", min_value=7, max_value=199, value=20, step=1
)
insulin = st.sidebar.number_input(
    "Insulin (Range: 14 - 1200 mu U/ml)", min_value=14, max_value=1200, value=79, step=1
)
bmi = st.sidebar.number_input(
    "BMI (Range: 18.2 - 67.1)", min_value=18.2, max_value=67.1, value=32.0, step=0.1
)
diabetes_pedigree_function = st.sidebar.number_input(
    "Diabetes Pedigree Function (Range: 0.078 - 3.42)",
    min_value=0.078,
    max_value=3.42,
    value=0.25,
    step=0.001,
)
# ... (similar inputs for other features)

if st.button("Train and Predict"):
    classifier1_name = st.sidebar.selectbox(
        "Select Classifier 1", list(classifiers_dict.keys())
    )
    classifier2_name = st.sidebar.selectbox(
        "Select Classifier 2", list(classifiers_dict.keys())
    )

    classifier1 = classifiers_dict[classifier1_name]
    classifier2 = classifiers_dict[classifier2_name]

    # Train and predict with Classifier 1
    classifier1.fit(X_train, y_train)
    y_pred1 = classifier1.predict(X_test)

    # Train and predict with Classifier 2
    classifier2.fit(X_train, y_train)
    y_pred2 = classifier2.predict(X_test)

    # Get classification report for Classifier 1
    report1 = classification_report(y_test, y_pred1, output_dict=True)

    # Get classification report for Classifier 2
    report2 = classification_report(y_test, y_pred2, output_dict=True)

    # Display classification results side by side in a nice tabular format
    st.write("### Classification Results")
    st.write(
        pd.DataFrame.from_dict(
            {
                f"{classifier1_name}": [
                    round(report1["accuracy"], 3),
                    round(report1["1"]["precision"], 3),
                    round(report1["1"]["recall"], 3),
                    round(report1["1"]["f1-score"], 3),
                ],
                f"{classifier2_name}": [
                    round(report2["accuracy"], 3),
                    round(report2["1"]["precision"], 3),
                    round(report2["1"]["recall"], 3),
                    round(report2["1"]["f1-score"], 3),
                ],
            },
            index=["Accuracy", "Precision", "Recall", "F1"],
        )
    )
