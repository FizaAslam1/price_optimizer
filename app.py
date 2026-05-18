import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================
st.set_page_config(page_title="PriceOptimizer AI", page_icon="🎯", layout="wide")

# ============================================
# DARK THEME — PROPER CONTRAST
# ============================================
st.markdown("""
<style>
    /* Force dark background with white text */
    .stApp, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117 !important;
        color: #ffffff !important;
    }
    
    /* All text white */
    p, span, label, div, h1, h2, h3, h4, h5 {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Metric values - Bright cyan */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #c9d1d9 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #7ee787 !important;
    }
    
    /* Headers */
    h1 {
        color: #58a6ff !important;
        font-size: 2.5rem !important;
    }
    h2, h3 {
        color: #f0f6fc !important;
    }
    
    /* Cards with dark bg + border */
    .info-card {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        padding: 18px !important;
        margin: 8px 0 !important;
    }
    
    .card-green {
        background: #0d3320 !important;
        border-left: 4px solid #3fb950 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 8px 0 !important;
    }
    .card-green * { color: #ffffff !important; }
    
    .card-red {
        background: #3d1f1f !important;
        border-left: 4px solid #f85149 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 8px 0 !important;
    }
    .card-red * { color: #ffffff !important; }
    
    .card-orange {
        background: #3d2e0f !important;
        border-left: 4px solid #d2991d !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 8px 0 !important;
    }
    .card-orange * { color: #ffffff !important; }
    
    .card-blue {
        background: #0d2137 !important;
        border-left: 4px solid #58a6ff !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 8px 0 !important;
    }
    .card-blue * { color: #ffffff !important; }
    
    /* Buttons */
    .stButton > button {
        background: #238636 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: #2ea043 !important;
        border-color: #58a6ff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22 !important;
        border-radius: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #c9d1d9 !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    
    /* Sliders */
    [data-testid="stSliderThumb"] {
        background: #58a6ff !important;
    }
    
    /* Horizontal rule */
    hr {
        border-color: #30363d !important;
    }
    
    /* Number inputs */
    input[type="number"] {
        background: #0d1117 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
    }
    
    /* Select boxes */
    select {
        background: #0d1117 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.title("🎯 PriceOptimizer AI")
st.markdown('<p style="color:#8b949e; font-size:1.1rem;">AI-Powered Pricing Engine — Maximize Profit with Demand Prediction</p>', 
            unsafe_allow_html=True)
st.markdown("---")

# ============================================
# DATA + MODEL
# ============================================
@st.cache_resource
def load_or_create_model():
    if os.path.exists("demand_model.pkl") and os.path.exists("feature_ranges.json"):
        with open("demand_model.pkl", "rb") as f:
            data = pickle.load(f)
        with open("feature_ranges.json") as f:
            ranges = json.load(f)
        return data["model"], data["features"], ranges
    
    with st.spinner("🔄 Training AI model... (one-time setup)"):
        np.random.seed(42)
        n = 3000
        df = pd.DataFrame({
            'price': np.random.choice(['$8,500','$15,000','$22,000','$35,000','$50,000','$75,000'], n),
            'brand': np.random.choice(['Toyota','Honda','BMW','Mercedes','Ford','Hyundai','Audi'], n),
            'model': 'M' + pd.Series(np.random.randint(1, 30, n)).astype(str),
            'model_year': np.random.randint(2010, 2024, n),
            'milage': np.random.choice(['25,000 mi.','50,000 mi.','80,000 mi.','120,000 mi.'], n),
            'fuel_type': np.random.choice(['Gasoline','Diesel','Hybrid'], n),
            'engine': np.random.choice(['180HP 2.0L I4','250HP 3.0L V6','350HP 4.0L V8','120HP 1.5L I3'], n),
            'transmission': np.random.choice(['Automatic','Manual'], n),
            'ext_col': np.random.choice(['Black','White','Silver','Blue','Red','Gray'], n),
            'int_col': np.random.choice(['Black','Gray','Beige','Brown'], n),
            'accident': np.random.choice(['None reported','At least 1 accident'], n, p=[0.7, 0.3]),
            'clean_title': np.random.choice(['Yes', np.nan], n, p=[0.8, 0.2])
        })
        
        df['price'] = df['price'].str.replace('$','',regex=False).str.replace(',','',regex=False).astype(float)
        df.columns = df.columns.str.lower().str.replace(' ','_')
        if 'milage' in df.columns:
            df.rename(columns={'milage':'mileage'}, inplace=True)
        
        df['mileage'] = df['mileage'].astype(str).str.replace(',','',regex=False).str.extract(r'(\d+)').astype(float)
        df['horsepower'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
        df['engine_size'] = df['engine'].str.extract(r'(\d+\.?\d*)\s*L').astype(float)
        df['cylinders'] = df['engine'].str.extract(r'V(\d+)').astype(float)
        df.drop('engine', axis=1, inplace=True)
        
        for col in ['brand','model','fuel_type','transmission','ext_col','int_col']:
            counts = df[col].value_counts()
            df[f'{col}_code'] = df[col].map(counts)
            df.drop(col, axis=1, inplace=True)
        
        df['accident_flag'] = df['accident'].str.contains('accident|damage', case=False, na=False).astype(int)
        df.drop('accident', axis=1, inplace=True)
        df['clean_title_flag'] = df['clean_title'].map({'Yes':1}).fillna(0).astype(int)
        df.drop('clean_title', axis=1, inplace=True)
        df = df.fillna(df.median(numeric_only=True))
        df = df[(df['price'] > df['price'].quantile(0.01)) & (df['price'] < df['price'].quantile(0.99))]
        
        df['cost_price'] = (df['price'] * 0.80).round(0)
        for i in range(1,4):
            df[f'comp{i}_price'] = (df['price'] * np.random.uniform(0.90, 1.10, len(df))).round(0)
        df['competitor_avg'] = df[['comp1_price','comp2_price','comp3_price']].mean(axis=1)
        df['price_diff'] = df['competitor_avg'] - df['price']
        df['units_sold'] = (50 + (df['price_diff']/1000)*10 + (df['model_year']-2015)*2 
                            - (df['mileage']/10000)*1 - (df['horsepower']/100)*0.5)
        df['units_sold'] *= np.random.normal(1, 0.05, len(df))
        df['units_sold'] = df['units_sold'].clip(lower=5, upper=200).round(0).astype(int)
        df.drop('price_diff', axis=1, inplace=True)
        
        FEATURES = ['model_year','mileage','horsepower','engine_size','cylinders',
                    'brand_code','model_code','fuel_code','transmission_code',
                    'ext_col_code','int_col_code','accident_flag','clean_title_flag','competitor_avg']
        
        from xgboost import XGBRegressor
        X = df[FEATURES].fillna(0)
        y = df['units_sold']
        model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
        model.fit(X, y)
        
        ranges = {col: {'min':float(X[col].min()), 'max':float(X[col].max()), 'avg':float(X[col].mean())} 
                  for col in FEATURES}
        
        with open("demand_model.pkl", "wb") as f:
            pickle.dump({"model":model, "features":FEATURES}, f)
        with open("feature_ranges.json", "w") as f:
            json.dump(ranges, f)
        
        return model, FEATURES, ranges

model, FEATURES, ranges = load_or_create_model()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🚗 Vehicle Specs")
    model_year = st.slider("Model Year", 2010, 2024, 2019)
    mileage = st.slider("Mileage", 0, 200000, 45000, 1000)
    horsepower = st.slider("Horsepower", 100, 500, 200, 10)
    engine_size = st.slider("Engine Size (L)", 1.0, 6.0, 2.0, 0.5)
    cylinders = st.selectbox("Cylinders", [4, 6, 8])
    
    st.subheader("📊 Market Data")
    fuel = st.selectbox("Fuel Type", [0, 1, 2], format_func=lambda x: ["⛽ Gasoline", "🛢️ Diesel", "⚡ Hybrid"][x])
    transmission = st.selectbox("Transmission", [0, 1], format_func=lambda x: ["🔄 Automatic", "🕹️ Manual"][x])
    brand_pop = st.slider("Brand Popularity", 0, 500, 250)
    model_pop = st.slider("Model Popularity", 0, 200, 50)
    
    st.subheader("🎨 Appearance")
    ext_pop = st.slider("Exterior Color Popularity", 0, 300, 150)
    int_pop = st.slider("Interior Color Popularity", 0, 300, 150)
    
    st.subheader("📋 History")
    accident = st.checkbox("⚠️ Had Accident?", False)
    clean_title = st.checkbox("✅ Clean Title?", True)
    
    st.markdown("---")
    st.subheader("💲 Pricing")
    competitor_avg = st.number_input("Competitor Avg Price ($)", 3000, 150000, 20000, 500)
    cost_price = st.number_input("Your Cost Price ($)", 3000, 100000, 15000, 500)
    current_price = st.slider("Your Current Price ($)", int(cost_price*0.8), int(cost_price*2), int(cost_price*1.2), 100)

# ============================================
# PREDICT + OPTIMIZE
# ============================================
features = pd.DataFrame([[
    model_year, mileage, horsepower, engine_size, cylinders,
    brand_pop, model_pop, fuel, transmission,
    ext_pop, int_pop, int(accident), int(clean_title), competitor_avg
]], columns=FEATURES)

current_demand = max(0, model.predict(features)[0])
current_profit = max(0, (current_price - cost_price) * current_demand)

price_range = np.linspace(max(cost_price*0.75, 500), cost_price*2.5, 80)
results = []
for price in price_range:
    demand = max(0, model.predict(features)[0])
    profit = (price - cost_price) * demand
    margin = ((price - cost_price) / price * 100) if price > 0 else 0
    results.append({"Price": price, "Demand": demand, "Profit": profit, "Margin": margin})

df_results = pd.DataFrame(results)
best_idx = df_results['Profit'].idxmax()
best_price = df_results.loc[best_idx, 'Price']
best_demand = df_results.loc[best_idx, 'Demand']
best_profit = df_results.loc[best_idx, 'Profit']
best_margin = df_results.loc[best_idx, 'Margin']

# ============================================
# KPI CARDS
# ============================================
st.markdown("---")
profit_change = ((best_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
competitor_gap = ((current_price - competitor_avg) / competitor_avg * 100)
position = "Premium 📈" if competitor_gap > 5 else ("Market ⚖️" if competitor_gap > -5 else "Budget 📉")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Optimal Price", f"${best_price:,.0f}", f"{profit_change:+.1f}% profit")

with col2:
    st.metric("📈 Demand Forecast", f"{best_demand:.0f} units", f"Margin: {best_margin:.1f}%")

with col3:
    st.metric("💰 Max Profit", f"${best_profit:,.0f}")

with col4:
    st.metric("🏷️ Market Position", position, f"{competitor_gap:+.1f}%")

st.markdown("---")

# ============================================
# CHARTS
# ============================================
tab1, tab2, tab3 = st.tabs(["📉 Demand Curve", "💰 Profit Optimization", "🔄 What-If Simulator"])

with tab1:
    col_c1, col_c2 = st.columns([3, 1])
    
    with col_c1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_results['Price'], y=df_results['Demand'],
                                  mode='lines', name='Demand Curve',
                                  line=dict(color='#58a6ff', width=3),
                                  fill='tozeroy', fillcolor='rgba(88,166,255,0.15)'))
        fig1.add_trace(go.Scatter(x=[best_price], y=[best_demand],
                                  mode='markers+text',
                                  marker=dict(size=18, color='#3fb950', symbol='star', line=dict(width=2, color='white')),
                                  text=[f'<b>OPTIMAL</b><br>${best_price:,.0f}'],
                                  textposition='top center',
                                  textfont=dict(color='#3fb950', size=13),
                                  name='Optimal'))
        fig1.add_trace(go.Scatter(x=[current_price], y=[current_demand],
                                  mode='markers+text',
                                  marker=dict(size=14, color='#f85149', symbol='diamond'),
                                  text=[f'Current<br>${current_price:,.0f}'],
                                  textposition='bottom center',
                                  textfont=dict(color='#f85149', size=11),
                                  name='Current'))
        fig1.update_layout(title=dict(text='Price vs Demand', font=dict(color='white', size=18)),
                           xaxis=dict(title='Price ($)', color='#c9d1d9', gridcolor='#21262d'),
                           yaxis=dict(title='Demand (Units)', color='#c9d1d9', gridcolor='#21262d'),
                           plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
                           height=400, hovermode='x unified',
                           legend=dict(font=dict(color='#c9d1d9')))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_c2:
        st.markdown("### 📊 Insights")
        st.markdown(f"""
        <div class="card-blue">
            <strong>🎯 Optimal</strong><br>
            <span style="font-size:1.4rem; font-weight:700;">${best_price:,.0f}</span><br>
            <small>+{profit_change:.1f}% profit gain</small>
        </div>
        <div class="info-card">
            <strong>💡 Rule:</strong><br>
            Higher price = Lower demand<br>
            <small>Find the sweet spot</small>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_results['Price'], y=df_results['Profit'],
                              mode='lines', name='Profit',
                              line=dict(color='#d2991d', width=4),
                              fill='tozeroy', fillcolor='rgba(210,153,29,0.12)'))
    fig2.add_trace(go.Scatter(x=[best_price], y=[best_profit],
                              mode='markers+text',
                              marker=dict(size=20, color='#d2991d', symbol='star', line=dict(width=3, color='white')),
                              text=[f'<b>PEAK</b><br>${best_profit:,.0f}'],
                              textposition='top center',
                              textfont=dict(color='#d2991d', size=13),
                              name='Max Profit'))
    fig2.update_layout(title=dict(text='Profit Optimization', font=dict(color='white', size=18)),
                       xaxis=dict(title='Price ($)', color='#c9d1d9', gridcolor='#21262d'),
                       yaxis=dict(title='Total Profit ($)', color='#c9d1d9', gridcolor='#21262d'),
                       plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown(f"""
    <div class="card-blue">
        <strong>💰 At Optimal:</strong>
        Per Unit: <b>${best_price - cost_price:,.0f}</b> | 
        Volume: <b>{best_demand:.0f} units</b> | 
        Total: <b>${best_profit:,.0f}</b> | 
        Margin: <b>{best_margin:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.subheader("🔄 Simulate Competitor Price Changes")
    
    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        new_comp = st.slider("Competitor adjusts price to:", 
                            int(competitor_avg*0.5), int(competitor_avg*1.5), 
                            int(competitor_avg), 100)
    
    f2 = features.copy()
    f2["competitor_avg"] = new_comp
    new_demand = max(0, model.predict(f2)[0])
    new_profit = (current_price - cost_price) * new_demand
    
    demand_delta = new_demand - current_demand
    profit_delta = new_profit - current_profit
    
    col_w3, col_w4, col_w5 = st.columns(3)
    
    with col_w3:
        cls = "card-green" if demand_delta >= 0 else "card-red"
        st.markdown(f'<div class="{cls}"><strong>📈 Demand</strong><br><span style="font-size:1.4rem;">{new_demand:.0f} units</span><br><small>{demand_delta:+.0f} vs now</small></div>', unsafe_allow_html=True)
    
    with col_w4:
        cls = "card-green" if profit_delta >= 0 else "card-red"
        st.markdown(f'<div class="{cls}"><strong>💰 Profit</strong><br><span style="font-size:1.4rem;">${new_profit:,.0f}</span><br><small>${profit_delta:+,.0f} vs now</small></div>', unsafe_allow_html=True)
    
    with col_w5:
        if new_comp < competitor_avg:
            action, cls = "📉 Lower Price", "card-orange"
        elif new_comp > competitor_avg:
            action, cls = "📈 Raise Price", "card-green"
        else:
            action, cls = "⚖️ Hold Price", "card-blue"
        st.markdown(f'<div class="{cls}"><strong>🎯 Strategy</strong><br><span style="font-size:1.2rem;">{action}</span></div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center; color:#484f58;">🚀 PriceOptimizer AI — XGBoost + Streamlit | Real-time Demand Prediction</p>', unsafe_allow_html=True)
