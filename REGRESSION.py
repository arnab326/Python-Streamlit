import streamlit as st
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Page layout
st.set_page_config(page_title="Machine Learning(Linear Regression) App", layout="centered")
st.write("""
      # Machine Learning App   
         """)
st.subheader("""             
       In this implementation , the Ordinary Least Squares() function is used to build a linear regression model.      
             """)

#Model Variable
model= None

#Model building
def build_model(df, param1, param2):
    global model
    #Write the linear regression formula
    ols_formula =f"{param1} ~ {param2}"
    
    #implement OLS
    OLS =ols(formula=ols_formula, data=df)
    
    #fit the model to the data
    model = OLS.fit()
    
    #get fitted values
    fitted_values = model.predict(df[param2])
    
    #get summary of the results
    model_summary = model.summary()
    
    return fitted_values, model_summary


#function to calculate sales based on radio promotion budget , slope and y-intercept
def calculate_sales(radio_budget, slope, y_intercept):
    sales= slope * radio_budget + y_intercept
    return sales

#Sidebar - collects user input features into dataframe
with st.sidebar.header('Please upload your CSV data'):
    uploaded_file=st.sidebar.file_uploader('Upload your csv file', type=['CSV'])
    
    st.sidebar.title("OLS Summary Parameters:")
    param1 = st.sidebar.text_input("Parameter y", value="column y")
    param2 = st.sidebar.text_input("Parameter x", value="column x")
    
#main content
st.title("OLS SUMMARY")

#sidebar -collect user input features into dataframe
st.sidebar.header('y = slop * x + y-intercept') 
slope = st.sidebar.number_input("Enter the slope with 2 or more decimal places:", value=0.00, step=0.01)
y_intercept = st.sidebar.number_input("Enter the y-intercept with 2 or more decimal places:", value=0.00, step=0.01)


#Input for radio promotion budget
radio_budget= st.sidebar.number_input("Enter the x value for planning")

#calculate sales based on the entered values
sales= calculate_sales(radio_budget, slope, y_intercept)

#display the calculated sales
st.sidebar.write(f"Projected dependent value (y):{sales}")

#display the dataset
st.subheader("1.Dataset")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('***1.1 Overview of dataset***')
    st.write(df.head(10))
    
    st.write("Keys:",df.keys())
    st.write("Shape",df.shape)
    st.write("data types:", df.dtypes)
    
    st.write("Check missing values:")
    st.write(df.isna().sum())
    
    st.write("Missing values along colum", df.isna().any(axis=1).sum())
    
    st.write("Drop rows with missing values:", df.dropna(axis=0))
    st.write("Missing Values:",df.isnull().sum())
    
    
    #build and display the linear regrassion model
    fitted_values, model_summary = build_model(df, param1, param2)
    st.subheader('2.Model Summary')
    st.text_area('Model Summary', model_summary, height=700)
    
    st.subheader("3. Create a regression plot using seaborn")
    fig1, ax = plt.subplots()
    sns.regplot(x=param2, y=param1, data=df, logistic=True, ci=None, ax=ax )
    st.pyplot(fig1)
    
    st.subheader("4. Crreate pairplot using seaborn")
    st.subheader("Pairplot")
    pairplot_fig = sns.pairplot(df)
    st.pyplot(pairplot_fig)
    
    #get the residuals from model
    residuals = model.resid
    
    st.subheader("5. Visualize the distribution of the residuals")
    
    fig3, ax3= plt.subplots()
    sns.histplot(residuals, ax=ax3)
    ax3.set_xlabel("Residual Value")
    ax3.set_title("Histogram of residuals")
    
    #display the plot using st.pyplot()
    st.pyplot(fig3)
    st.write("Check if the distribution is normal one")
    
    #display the q-q plot
    st.subheader('6. Create 0-0 plot')
    fig4, ax4 = plt.subplots()
    sm.qqplot(residuals, line="s", ax=ax4)
    ax4.set_title("0-0 plot of Residuals")
    
    st.pyplot(fig4)
    st.write("normality assumption is met when the points follow a straight diagonal line")
    
    st.subheader("7. Create a scatterplot of the residuals against fitted values")
    fig5 = sns.scatterplot(x=fitted_values, y= residuals)
    fig5.axhline(0, color='red' , linestyle='--')
    fig5.set_xlabel("Fitted Values")
    fig5.set_ylabel("Residuals")
    
    st.pyplot(fig4.figure)
    