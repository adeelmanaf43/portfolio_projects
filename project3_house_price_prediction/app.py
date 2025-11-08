# Importing Required libraries
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 1. Page Configuration
st.set_page_config(
    page_title='🏠 Hose Price Prediction by Adeel Manaf',
    page_icon='🏠',
    layout='wide'
)


# Set the title
st.title('🏠 House Price Prediction App')

# adding details
st.markdown("""
Welcome!
This app predicts **house sale prices** using a trained Random Forest regression.
""")

# Reading data from csv file
df = pd.read_csv("data/house_price.csv")
# Separating Input and Output variable

X = df[['OverallQual', 'GrLivArea', 'GarageCars',
        'TotalBsmtSF', 'YearBuilt', 'Neighborhood']]
y = df['SalePrice']

# Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Separating numerical features and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include='object').columns


# Building numeric features transformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Building categorical features transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Building preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# making pipeline for random forest model
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# fitting pipe on training data
pipe.fit(X_train, y_train)


# 3. Sidebar Inputs

st.sidebar.header("🔧 Input House Features")
OverallQual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.sidebar.number_input(
    'Above Ground Living Area (Sq Ft)', df['GrLivArea'].min(), df['GrLivArea'].max(), 1500)
GarageCars = st.sidebar.slider(
    'Garage Cars Capacity', df['GarageCars'].min(), df['GarageCars'].max(), 2)
TotalBsmtSF = st.sidebar.number_input(
    'Total Basement Area (sq ft)', df['TotalBsmtSF'].min(), df['TotalBsmtSF'].max(), 900)
YearBuilt = st.sidebar.number_input(
    'Year Built', df['YearBuilt'].min(), df['YearBuilt'].max(), 2000)
Neighborhood = st.sidebar.selectbox('Neighborhood', [
    'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
    'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
    'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
    'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
    'Blueste'
])

# Creating input dictionary
input_dict = {
    'OverallQual': OverallQual,
    'GrLivArea': GrLivArea,
    'GarageCars': GarageCars,
    'TotalBsmtSF': TotalBsmtSF,
    'YearBuilt': YearBuilt,
    'Neighborhood': Neighborhood
}

# Creating input dataframe
input_df = pd.DataFrame([input_dict])


# Prediction Button
st.write("---")
if st.button("💰 Predict House Price"):
    pred = pipe.predict(input_df)[0]
    st.success(f"🏡 **Predicted Price:** ${pred:,.0f}")

    # Feature Importance plot
    rf_model = pipe.named_steps['model']

    # Get Feature Names
    feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

    # Get feature importances
    importances = rf_model.feature_importances_

    # Combine them in a dataframe
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    # Plottin feature importance
    fig, ax = plt.subplots()
    ax.barh(imp_df['Feature'], imp_df['Importance'], color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.invert_yaxis()
    st.pyplot(fig)

    # Shap Graph
    # Get the model from pipeline
    rf_model = pipe.named_steps['model']
    preprocessor = pipe.named_steps['preprocessor']

    # Transform input data
    input_transformed = preprocessor.transform(input_df)
    if hasattr(input_transformed, "toarray"):  # if sparse
        input_transformed = input_transformed.toarray()

    input_transformed = input_transformed.astype(float)

    # Create SHAP explainer for RandomForest
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_transformed)

    # Display SHAP force plot
    shap.initjs()
    # Modern and clean fix
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                         base_values=explainer.expected_value,
                                         data=input_transformed[0],
                                         feature_names=feature_names))
    st.pyplot(fig)
    plt.close(fig)
