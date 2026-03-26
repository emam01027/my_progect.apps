import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# 1. إعدادات الصفحة (يفضل وضعها في البداية)
st.set_page_config(page_title="Car Price Dashboard", layout="wide")

# تحميل الموديل
@st.cache_resource # استخدام الكاش لتسريع التطبيق
def load_model():
    return pickle.load(open(r"C:\Users\hp\Cars_Predictions.sav", "rb"))

model = load_model()

# --- الهيكل الرئيسي للتطبيق (Tabs) ---
tab1, tab2 = st.tabs(["🔮 Price Prediction", "📊 Data Dashboard"])

with tab1:
    st.title("Car Price Prediction")
    st.image(r"C:\Users\hp\my_project\pexels-auto-2179220_1920.jpg")
    
    # تنظيم المدخلات في أعمدة لجعل الشكل أفضل
    col1, col2, col3 = st.columns(3)
    
    with col1:
        m1 = ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'BMW', 'KIA'] # اختصرت القائمة للمثال
        Manu_maping = {name: i for i, name in enumerate(m1)} # طريقة أسرع للمابينج
        Manu1 = st.selectbox("Manufacturer", m1)
        Manu2 = Manu_maping.get(Manu1, 0)

        Fuel1 = st.selectbox("Fuel", ["Hybrid", "Petrol", "Diesel", "LPG"])
        Fuel = {"Hybrid": 2, "Petrol": 5, "Diesel": 1, "LPG": 0}[Fuel1]

    with col2:
        Category1 = st.selectbox("Category", ["Jeep", "Hatchback", "Sedan", "Coupe"])
        Category = {"Jeep": 4, "Hatchback": 3, "Sedan": 9, "Coupe": 0}[Category1]
        
        Gear1 = st.selectbox("Gear box", ["Automatic", "Tiptronic", "Manual"])
        Gear = {"Automatic": 0, "Tiptronic": 2, "Manual": 1}[Gear1]

    with col3:
        Engine = st.number_input("Engine volume", value=2.0)
        Airbags = st.slider("Airbags", 0, 16, 4)
        Age = st.number_input("Age", value=5)

    # زر التوقع في القائمة الجانبية كما كنت تفعل
    st.sidebar.header("Execution")
    if st.sidebar.button("Predict Price"):
        # بناء الـ DataFrame (تأكد من ترتيب الأعمدة كما تدرب الموديل)
        df_input = pd.DataFrame({
            "Manufacturer": [Manu2], "Fuel type": [Fuel], "Category": [Category],
            "Gear box type": [Gear], "Engine volume": [Engine], "Airbags": [Airbags], "Age": [Age]
        })
        # ملاحظة: يجب أن يحتوي df_input على كل الأعمدة الـ 15 التي في كودك الأصلي
        prediction = model.predict(df_input)
        st.sidebar.success(f"Estimated Price: ${prediction[0]:,.2f}")

with tab2:
    st.title("🚗 Cars Market Insights")
    
    # هنا نصنع داشبورد تفاعلي
    # سأصنع بيانات وهمية كمثال، استبدلها بـ pd.read_csv("your_data.csv")
    chart_data = pd.DataFrame({
        'Manufacturer': ['Toyota', 'Lexus', 'BMW', 'Hyundai', 'Ford'],
        'Average Price': [20000, 45000, 50000, 18000, 25000],
        'Count': [150, 80, 60, 120, 90]
    })

    # صف الأعمدة العلوية (Metrics)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Models", "65")
    m2.metric("Avg Market Price", "$28,500", "+5%")
    m3.metric("Most Popular", "Toyota")

    st.markdown("---")

    # صف الرسوم البيانية
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Price Distribution by Brand")
        fig1 = px.bar(chart_data, x='Manufacturer', y='Average Price', color='Manufacturer', template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Market Share")
        fig2 = px.pie(chart_data, values='Count', names='Manufacturer', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Engine Volume vs Price Analysis")
    # مثال لرسم منتشر (Scatter Plot)
    fig3 = px.scatter(chart_data, x="Average Price", y="Count", size="Count", color="Manufacturer", hover_name="Manufacturer")
    st.plotly_chart(fig3, use_container_width=True)