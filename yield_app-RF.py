from gettext import install
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
#import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from PIL import Image
import pickle


st.markdown("<h1 style='text-align:center; color:black;'>Yield Prediction</h2>", unsafe_allow_html=True)

df = pd.read_csv('EDA_clean_yield.csv')

# show image
img = Image.open("bowery.png")
new_img=img.resize((750, 150))
st.image(new_img)

# add button
if st.checkbox("Show Data") :
    st.write(df.head())

# add warning
# st.error(":point_left:Please input the features of component **using sidebar**, before making yield prediction!!!")

st.sidebar.title("Please select features of component")

# Collects user input features into dataframe
component_name = st.sidebar.selectbox("Component Name", df["component_name"].unique())
cultivar_name=st.sidebar.selectbox("Cultivar Name", df.loc[df.component_name==component_name]['cultivar_name'].unique())
temperature_c = st.sidebar.slider("Temperature", min_value=20.8, max_value=26.8, value=23.9, step=0.1)
humidity_rh = st.sidebar.slider("Humidity", min_value=52.7, max_value=75.9, value=64.8, step=0.1)
vapor_pressure_deficit = st.sidebar.slider("Vapor Pressure Deficit", min_value=0.43, max_value=1.83, value=1.15, step=0.01)
co2_ppm = st.sidebar.slider("CO2 ppm", min_value=996, max_value=1262, value=1134, step=1)
fan_intensity = st.sidebar.slider("Fan Intensity", min_value=149, max_value=254, value=203, step=1)
light_intensity = st.sidebar.slider("Light Intensity", min_value=117, max_value=255, value=188, step=1)
light_on_minutes = st.sidebar.slider("Light on (min)", min_value=20, max_value=60, value=40, step=1)
irrigation_minutes = st.sidebar.slider("Irrigation (min)", min_value=30, max_value=60, value=45, step=1)
grow_room = st.sidebar.selectbox("Grow Room", df["grow_room"].unique())

data = {
    "component_name" : component_name,
    "cultivar_name" : cultivar_name,
    "temperature_c" : temperature_c,
    "humidity_rh" : humidity_rh,
    "vapor_pressure_deficit" : vapor_pressure_deficit,
    "co2_ppm" : co2_ppm,
    "fan_intensity" : fan_intensity,
    "light_intensity" : light_intensity,
    "light_on_minutes" : light_on_minutes,
    "irrigation_minutes" : irrigation_minutes,
    "grow_room" : grow_room
}
     
df2 = pd.DataFrame([data])

df3 = df2.copy()

df3.rename(columns={'component_name':'Component', 'cultivar_name':'Cultivar','temperature_c':'Temp',
                   'humidity_rh':'Humidity','vapor_pressure_deficit':'Vapor_Pres','co2_ppm':'CO2',
                    'fan_intensity':'Fan_Intens.', 'light_intensity':'Light_Intens.', 
                    'light_on_minutes':'Light_on(m)', 'irrigation_minutes':'Irrigation(m)',
                    'grow_room':'GR'
                   }, inplace=True)

df3['Temp'] = df3['Temp'].apply(lambda x: format(float(x),".2f"))
df3['Humidity'] = df3['Humidity'].apply(lambda x: format(float(x),".2f"))
df3['Vapor_Pres'] = df3['Vapor_Pres'].apply(lambda x: format(float(x),".2f"))

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """



# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)


st.subheader("The feature of component is below")
            
st.table(df3)


# Read saved model
yield_model = pickle.load(open('RF_model', 'rb'))

#st.subheader("Press predict if configuration is okay")
# Apply model to make predictions
if st.button('Predict Yield'):
    prediction = yield_model.predict(df2)
    st.success("The estimate yield of **{}** is {:.2f} ".format(df2.iloc[:,0][0], float(prediction)))

#st.markdown("Thank you for visiting **Yield Prediction** page.")

