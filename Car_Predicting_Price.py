import streamlit as st
import pandas as pd
import pickle

# Upload Data

data=pickle.load(open(r"C:\Users\hp\Cars_Predictions.sav","rb"))

# Streamlit Page

st.title("Car Price Prediction")

st.sidebar.header("Feature Selecting")
st.sidebar.info("Easy Application For Predicting Cares_Price")

st.image(r"C:\Users\hp\my_project\pexels-auto-2179220_1920.jpg")
#---------------------------------------------------------------------------------------------------------------------------------------------------- --                                                                    

m1=['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA',
       'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN',
       'AUDI', 'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA',
       'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC', 'FIAT', 'INFINITI',
       'ALFA ROMEO', 'SUZUKI', 'ACURA', 'LINCOLN', 'VAZ', 'GAZ',
       'CITROEN', 'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR',
       'ISUZU', 'SKODA', 'DAIHATSU', 'BUICK', 'TESLA', 'CADILLAC',
       'PEUGEOT', 'BENTLEY', 'VOLVO', 'სხვა', 'HAVAL', 'HUMMER', 'SCION',
       'UAZ', 'MERCURY', 'ZAZ', 'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH',
       'MASERATI', 'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE',
       'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL']
m2=[16, 12, 17, 43, 27, 45, 35, 31,  6, 41,  9,  3, 21, 30, 40, 26, 14,
       11, 42, 24, 32,  2,  8, 29, 10, 23, 20,  0, 44, 19, 39,  7, 25,  4,
       33, 47, 15,  5, 38, 18, 34, 22, 28, 36, 46,  1, 37, 13]
Manu_maping=dict(zip(m1,m2))
Manu1=st.selectbox("Manufacturer",m1)
Manu2=Manu_maping[Manu1]
#---------------------------------------------------------------------------------------------------------------------------------------------
mm1=["RX 450","Equinox","FIT","E 230 124","RX 450 F SPORT","Prius C aqua"]
mm2=[890, 458, 477, 485, 470, 833]
Model_maping=dict(zip(m1,m2))
Model1=st.selectbox("Model",m1)
Model=Model_maping[Model1]
#---------------------------------------------------------------------------------------------------------------------------------------------
c1=["Jeep", "Hatchback", "Sedan", "Microbus", "Goods wagon", "Universal", "Coupe", "Minivan", "Cabriolet", "Limousine", "Pickup"]
c2=[4, 3, 9, 2, 10, 7, 0, 1, 6, 8, 5]
Category_maping=dict(zip(c1,c2))
Category1=st.selectbox("Category",c1)
Category=Category_maping[Category1]
#----------------------------------------------------------------------------------------------------------------------------------------------------
l1=["Yes", "No"]
l2=[1, 2]
Leather_maping=dict(zip(l1,l2))
Leather1=st.selectbox("Leather",l1)
Leather=Leather_maping[Leather1]
#--------------------------------------------------------------------------------------------------------------------------------------------------------
f1=["Hybrid", "Petrol", "Diesel","CNG", "Plug-in Hybrid", "LPG", "Hydrogen"]
f2=[2, 5, 1, 6, 4, 0, 3]
Fuel_maping=dict(zip(f1,f2))
Fuel1=st.selectbox("Fuel",f1)
Fuel=Fuel_maping[Fuel1]
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
g1=["Automatic", "Tiptronic", "Variator", "Manual"]
g2=[0, 2, 3, 1]
Gear_maping=dict(zip(g1,g2))
Gear1=st.selectbox("Gear",g1)
Gear=Gear_maping[Gear1]
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
d1=["4x4", "Front", "Rear"]
d2=[1, 2]
Drive_maping=dict(zip(d1,d2))
Drive1=st.selectbox("Drive",d1)
Drive=Drive_maping[Drive1]
#-------------------------------------------------------------------------------------------------------
w1=["Left wheel", "Right-hand drive"]
w2=[0, 1]
Wheel_maping=dict(zip(w1,w2))
Wheel1=st.selectbox("Wheel",w1)
Wheel=Wheel_maping[Wheel1]
#--------------------------------------------------------------------------------------------------------------------------------------------------------
cc1=["Silver", "Black", "White", "Grey", "Blue", "Red", "Sky blue", "Orange", "Yellow", "Brown", "Golden", "Beige", "Carnelian red", "Purple", "Pink"]
cc2=[12, 1, 14, 7, 13, 11, 8, 6, 15, 3, 5, 0, 4, 10, 9]
Color_maping=dict(zip(cc1,cc2))
Color1=st.selectbox("Color",cc1)
Color=Color_maping[Color1]
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
Cylinders = st.selectbox("Cylinders", [4, 6, 8, 12, 1, 2, 3, 5, 10, 16])
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Engine=st.selectbox("Engine volume", [3.5, 3. , 1.3, 2.5, 2. , 1.8, 2.4, 1,6, 2.2, 1.5, 3.3, 1.4, 2.3, 3.2, 1.2, 1.7, 2.9, 1.9, 2.7, 2.8, 2.1, 1. , 0.8, 3.4, 2.6, 1.1])
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Airbags=st.selectbox("Airbags", [12, 8, 2, 0, 4, 6, 10, 3, 1, 16, 5, 7, 9, 11, 14, 15, 13])
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Age=st.number_input("Age")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Mileage=st.number_input("Mileage")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Levy=st.number_input("Levy")
#-----------------------------------------------------------------------------------------
df=pd.DataFrame({"Manufacturer":Manu2, "Model":Model, "Category":Category, "Leather interior":Leather,
               "Fuel type":Fuel, "Mileage":Mileage,  "Gear box type":Gear, "Drive wheels":Drive, "Wheel":Wheel, "Color":Color, "Levy":Levy,
               "Engine volume":Engine, "Cylinders":Cylinders,
               "Airbags":Airbags, "Age":Age },index=[0])

p=st.sidebar.button("Predict Price")
if p:
    pre=data.predict(df)
    st.sidebar.write("Price Is :", pre)



