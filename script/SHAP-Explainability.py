import shap
import matplotlib.pyplot as plt

Assume `rf` is your trained best model (Random Forest in this case)
# and X_train is the training features BEFORE scaling/SMOTE (or the scaled one used in training)
def shap_explain_model(model, X_train, feature_names=None):
    # Initialize explainer
    explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values
    shap_values = explainer(X_train)

    # Summary plot: global feature importance & effect
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names)

    # Force plot for first 5 samples (local explanations)
    for i in range(5):
        shap.force_plot(explainer.expected_value, shap_values[i].values, 
                        features=X_train[i], feature_names=feature_names, matplotlib=True)

# Usage example, assuming Task 2 outputs:

# You should pass the training data that the model was trained on.
# Typically, the scaled and SMOTE-resampled X_train_res.
# For SHAP, better to use a subset to save resources.

# For example, use first 100 samples of X_train_res:

X_train_subset = X_train[:100]  # if scaled numpy array
feature_names = X_fraud.columns.tolist()  # or X_cc.columns.tolist() for credit card data

# Run SHAP explainability on Fraud dataset Random Forest
shap_explain_model(rf, X_train_subset, feature_names=feature_names)
