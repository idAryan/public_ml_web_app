# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:01:15 2022

@author: siddhardhan
"""

import pickle
import streamlit as st


# Load the saved diabetes model
diabetes_model = pickle.load(open('trained_model.sav', 'rb'))


# Diabetes Prediction Page
def main():
    st.title('Diabetes Prediction using ML')

    # Collecting user input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
        
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1)
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=150, step=1)
    
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, step=1)
    
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0, max_value=900, step=1)
    
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, max_value=70.0, step=0.1)
    
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, step=0.01)
    
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, step=1)
    
    # Code for Prediction
    diab_diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
                
            st.success(diab_diagnosis)
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    main()
