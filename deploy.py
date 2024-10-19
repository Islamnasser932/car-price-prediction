import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st


# Read Data
car = pd.read_csv('CarPrice_Assignment.csv')

# Configure page
st.set_page_config(layout='wide')

# Sidebar options
option = st.sidebar.selectbox("Pick a choice:", ['Home', 'EDA', 'Model'])

if option == 'Home':
    st.title("Car App")
    st.dataframe(car.head(20))
    
elif option == 'EDA':
    st.title("Car EDA")
    
    # Symboling Distribution
    st.header("Symboling Distribution")
    car_counts = car['symboling'].value_counts().reset_index()
    car_counts.columns = ['symboling', 'Count']
    fig1 = px.bar(car_counts, x='symboling', y='Count', color='symboling', text='Count', template='plotly_dark')
    st.plotly_chart(fig1)
    
    st.header("fueltype  of Car")
    car_counts = car['fueltype'].value_counts().reset_index()
    car_counts.columns = ['fueltype', 'Count']
    fig1 = px.bar(car_counts, x='fueltype', y='Count', color='fueltype',color_discrete_sequence=['#96CEB4', '#F6E96B'], template='plotly_dark')
    st.plotly_chart(fig1)    
    
    
    st.header("doornumber  of Car")
    car_counts = car['doornumber'].value_counts().reset_index()
    car_counts.columns = ['doornumber', 'Count']
    fig1 = px.bar(car_counts, x='doornumber', y='Count', color='doornumber', color_discrete_sequence=['#C7253E', '#FABC3F','#C7253E','#FABC3F','#CD5C08'],template='plotly_dark')
    st.plotly_chart(fig1)    
    
    
    st.header("carbody of Car")
    car_counts = car['carbody'].value_counts().reset_index()
    car_counts.columns = ['carbody', 'Count']
    fig1 = px.bar(car_counts, x='carbody', y='Count', color='carbody',color_discrete_sequence=['#6A9C89', '#78B7D0','#C7253E','#FABC3F','#CD5C08'], text='Count', template='plotly_dark')
    st.plotly_chart(fig1)
    
    
    st.header("enginetype of Car")
    car_counts = car['enginetype'].value_counts().reset_index()
    car_counts.columns = ['enginetype', 'Count']
    fig1 = px.bar(car_counts, x='enginetype', y='Count', color='enginetype',color_discrete_sequence=['#6A9C89', '#78B7D0','#C7253E','#FABC3F','#CD5C08'],text='Count', template='plotly_dark')
    st.plotly_chart(fig1)   
    
     
    st.header("enginetype of Car")
    car_counts = car['enginetype'].value_counts().reset_index()
    car_counts.columns = ['enginetype', 'Count']
    fig1 = px.bar(car_counts, x='enginetype', y='Count', color='enginetype',color_discrete_sequence=['#6A9C89', '#78B7D0','#C7253E','#FABC3F','#CD5C08'],text='Count', template='plotly_dark')
    st.plotly_chart(fig1)       
    
    
    
elif option == 'Model':
    st.title("ML Car Prediction")
    st.text("In this app, we will predict the price of the car")
    st.text("Please enter the following values:")
    btn = st.button("Submit")

    carbody = st.selectbox("Select carbody", ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'])
    fuel_type = st.selectbox("Select fuel_type", ['gas', 'diesel'])

    # Select features and target variable (add 'fueltype' and 'carbody' to features)
    new_data = car[['carlength', 'carwidth', 'carheight', 'curbweight', 'stroke', 'fueltype', 'carbody', 'price']]  # Added 'fueltype' and 'carbody'
    
    # Apply Label Encoding only to categorical columns
    le_fueltype = LabelEncoder()
    le_carbody = LabelEncoder()

    new_data['fueltype'] = le_fueltype.fit_transform(new_data['fueltype'])
    new_data['carbody'] = le_carbody.fit_transform(new_data['carbody'])
    
    # Standardize features (X columns, not the target column 'price')
    ss = StandardScaler()
    new_data.iloc[:, :-1] = ss.fit_transform(new_data.iloc[:, :-1])  # Scale X columns (but not the target)
        
    def handling_outliers(car, lst_of_col):
        for i in lst_of_col:
            q1 = car[i].quantile(0.25)
            q3 = car[i].quantile(0.75)
            iqr = q3 - q1
            upper_limit = q3 + 1.5 * iqr
            lower_limit = q1 - 1.5 * iqr

            for col in range(0, car.shape[0]):
                if car[i][col] < lower_limit:
                    car[i][col] = lower_limit
                elif car[i][col] > upper_limit:
                    car[i][col] = upper_limit
                else:
                    continue

        return car    
    
    new_data = handling_outliers(new_data, new_data.columns)
    
    # Prepare X and y
    X = new_data.drop(columns=['price'])  # 'fueltype' and 'carbody' are part of X
    y = new_data['price']

    # Save column names before scaling
    column_names = X.columns

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Prediction logic
    if btn:
        try:
            # Encoding carbody and fuel_type values manually
            carbody_encoded = le_carbody.transform([carbody])[0]
            fuel_type_encoded = le_fueltype.transform([fuel_type])[0]
            
            # Prepare input data
            input_data = np.zeros((1, X_train.shape[1]))
            
            # Use the saved column names to get 'carbody' and 'fuel_type' index
            carbody_col_index = list(column_names).index('carbody')
            fuel_type_col_index = list(column_names).index('fueltype')  # Use correct column name 'fueltype'
            
            input_data[0, carbody_col_index] = carbody_encoded
            input_data[0, fuel_type_col_index] = fuel_type_encoded
            
            # Scale the input data
            input_data_scaled = ss.transform(input_data)

            # Predict the price using Random Forest Regressor
            result = rf.predict(input_data_scaled)
            
            # Display result
            st.write(f"Predicted Car Price: ${result[0]:.2f}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
