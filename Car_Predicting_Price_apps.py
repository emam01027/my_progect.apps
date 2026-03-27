import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from datetime import datetime

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Car Price Predictor (Pro)",
    layout="wide", 
    page_icon="📊"
)

# --- إدارة التنقل بين الصفحات ---
if "page" not in st.session_state:
    st.session_state.page = "prediction"

col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Dashboard"):
        st.session_state.page = "dashboard"
with col2:
    if st.button("🚗 Prediction"):
        st.session_state.page = "prediction"

# --- التحميل المسبق للموديل ---
@st.cache_resource
def load_model():
    return pickle.load(open("Cars_Predictions.sav", "rb"))

try:
    model_data = load_model()
except:
    model_data = None

# ---------------------------------------------------------
# 1.Dashboard (Turbo)
# ---------------------------------------------------------
if st.session_state.page == "dashboard":
    st.header("📊 Car Market Data Analysis")
    
    FILE_PATH = "car_price_prediction.csv" 

    try:
        df_csv = pd.read_csv(FILE_PATH)
        
        if 'Engine volume' in df_csv.columns:
            
            df_csv['Engine volume'] = df_csv['Engine volume'].astype(str).str.replace('Turbo', '', case=False).str.strip()
            df_csv['Engine volume'] = pd.to_numeric(df_csv['Engine volume'], errors='coerce')
            
        if 'Price' in df_csv.columns:
            df_csv['Price'] = pd.to_numeric(df_csv['Price'], errors='coerce')

        df_csv = df_csv.dropna(subset=['Engine volume', 'Price'])

        # ---Sidebar ---
        with st.sidebar:
            st.header("🔍 Dashboard Filters")
            st.image("pexels-auto-2179220_1920.jpg")
            
            selected_brands = st.multiselect("Select Brands", options=sorted(df_csv['Manufacturer'].unique()), default=df_csv['Manufacturer'].unique()[:5])
            selected_fuels = st.multiselect("Fuel Type", 
                               options=sorted(df_csv['Fuel type'].unique()), 
                               default=['Petrol', 'Diesel']) 
            selected_gears = st.multiselect("Transmission Type", 
                               options=sorted(df_csv['Gear box type'].unique()), 
                               default=['Automatic'])

        
        df_filtered = df_csv[
            (df_csv['Manufacturer'].isin(selected_brands)) &
            (df_csv['Fuel type'].isin(selected_fuels)) &
            (df_csv['Gear box type'].isin(selected_gears))
        ]

        if df_filtered.empty:
            st.warning("No data matches your filters.")
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("Selected Cars", len(df_filtered))
            k2.metric("Avg Price", f"{df_filtered['Price'].mean():,.0f}")
            k3.metric("Avg Engine", f"{df_filtered['Engine volume'].mean():.1f}L")

            
            row1_col1, row1_col2 = st.columns(2)
            
            fig1 = px.bar(df_filtered['Manufacturer'].value_counts(), title="Cars by Brand", color_discrete_sequence=['#636EFA'])
            row1_col1.plotly_chart(fig1, use_container_width=True)

            fig2 = px.scatter(df_filtered, x='Prod. year', y='Price', color='Manufacturer', title="Year vs Price")
            row1_col2.plotly_chart(fig2, use_container_width=True)

            fig3 = px.pie(df_filtered, names='Fuel type', title="Fuel Distribution", hole=0.3)
            st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

# ---------------------------------------------------------
# 2.Prediction
# ---------------------------------------------------------
elif st.session_state.page == "prediction":
    with st.sidebar:
        st.header("Car Price Prediction")
        st.image("pexels-auto-2179220_1920.jpg")
        st.write("""
        Enter the car specifications to get an estimated market price 
        based on our trained Machine Learning model.
        
        The prediction is powered by an optimized Random Forest model
        trained on real used car market data.
        """)

        st.subheader("🧠 Model Info")
        st.write("• Model: Random Forest Regressor")
        st.write("• Tuned with GridSearchCV")
        st.write("• Evaluation Metric: R² Score")
        st.write("• Preprocessing: StandardScaler + Pipeline")
        st.caption("This prediction is for estimation purposes only and may vary from actual market prices.")

    st.title("CAR PRICE PREDICTION")
    st.image("pexels-auto-2179220_1920.jpg")
    st.write("Enter the car details to predict its selling price")
    
    m1=['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN', 'AUDI', 'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC', 'FIAT', 'INFINITI', 'ALFA ROMEO', 'SUZUKI', 'ACURA', 'LINCOLN', 'VAZ', 'GAZ', 'CITROEN', 'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR', 'ISUZU', 'SKODA', 'DAIHATSU', 'BUICK', 'TESLA', 'CADILLAC', 'PEUGEOT', 'BENTLEY', 'VOLVO', 'სხვა', 'HAVAL', 'HUMMER', 'SCION', 'UAZ', 'MERCURY', 'ZAZ', 'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH', 'MASERATI', 'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE', 'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL']
    m2=[16, 12, 17, 43, 27, 45, 35, 31, 6, 41, 9, 3, 21, 30, 40, 26, 14, 11, 42, 24, 32, 2, 8, 29, 10, 23, 20, 0, 44, 19, 39, 7, 25, 4, 33, 47, 15, 5, 38, 18, 34, 22, 28, 36, 46, 1, 37, 13]
    Manu_maping = dict(zip(m1, m2))

    c_left, c_right = st.columns(2)
    with c_left:
        Manu1 = st.selectbox("Manufacturer", m1)
        Model1 = st.selectbox("Model", m1)
        Category1 = st.selectbox("Category", ["Jeep", "Hatchback", "Sedan", "Microbus", "Goods wagon", "Universal", "Coupe", "Minivan", "Cabriolet", "Limousine", "Pickup"])
        Fuel1 = st.selectbox("Fuel type", ["Hybrid", "Petrol", "Diesel","CNG", "Plug-in Hybrid", "LPG", "Hydrogen"])
        Gear1 = st.selectbox("Gear box type", ["Automatic", "Tiptronic", "Variator", "Manual"])

    with c_right:
        Leather1 = st.selectbox("Leather interior", ["Yes", "No"])
        Drive1 = st.selectbox("Drive wheels", ["4x4", "Front", "Rear"])
        Wheel1 = st.selectbox("Wheel", ["Left wheel", "Right-hand drive"])
        Color1 = st.selectbox("Color", ["Silver", "Black", "White", "Grey", "Blue", "Red", "Sky blue", "Orange", "Yellow", "Brown", "Golden", "Beige", "Carnelian red", "Purple", "Pink"])
        Cylinders = st.selectbox("Cylinders", [4, 6, 8, 12, 1, 2, 3, 5, 10, 16])

    st.markdown("---")
    col_num1, col_num2, col_num3 = st.columns(3)
    with col_num1: Engine = st.number_input("Engine volume", value=2.0)
    with col_num2: Airbags = st.number_input("Airbags", value=4)
    with col_num3: Age_input = st.number_input("Age", value=5)
    
    Mileage = st.number_input("Mileage", value=0)
    Levy = st.number_input("Levy", value=0)
   
   
    btn = st.button("💰 Predict Price")

    if btn:
        if model_data:
            cat_map = dict(zip(["Jeep", "Hatchback", "Sedan", "Microbus", "Goods wagon", "Universal", "Coupe", "Minivan", "Cabriolet", "Limousine", "Pickup"], [4, 3, 9, 2, 10, 7, 0, 1, 6, 8, 5]))
            fuel_map = dict(zip(["Hybrid", "Petrol", "Diesel","CNG", "Plug-in Hybrid", "LPG", "Hydrogen"], [2, 5, 1, 6, 4, 0, 3]))
            gear_map = dict(zip(["Automatic", "Tiptronic", "Variator", "Manual"], [0, 2, 3, 1]))
            drive_map = dict(zip(["4x4", "Front", "Rear"], [1, 2, 0]))
            wheel_map = dict(zip(["Left wheel", "Right-hand drive"], [0, 1]))
            color_map = dict(zip(["Silver", "Black", "White", "Grey", "Blue", "Red", "Sky blue", "Orange", "Yellow", "Brown", "Golden", "Beige", "Carnelian red", "Purple", "Pink"], [12, 1, 14, 7, 13, 11, 8, 6, 15, 3, 5, 0, 4, 10, 9]))

            df_input = pd.DataFrame({
                "Manufacturer": [Manu_maping[Manu1]],
                "Model": [Manu_maping[Model1]],
                "Category": [cat_map[Category1]],
                "Leather interior": [1 if Leather1 == "Yes" else 2],
                "Fuel type": [fuel_map[Fuel1]],
                "Mileage": [Mileage],
                "Gear box type": [gear_map[Gear1]],
                "Drive wheels": [drive_map[Drive1]],
                "Wheel": [wheel_map[Wheel1]],
                "Color": [color_map[Color1]],
                "Levy": [Levy],
                "Engine volume": [Engine],
                "Cylinders": [Cylinders],
                "Airbags": [Airbags],
                "Age": [Age_input]
            }, index=[0])

            pre = model_data.predict(df_input)
            st.success(f"### Price Is: \n {pre[0]:,.2f}")
            st.balloons()
