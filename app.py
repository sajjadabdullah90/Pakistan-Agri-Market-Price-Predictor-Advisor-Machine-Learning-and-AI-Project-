import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
try:
    import nltk
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except (LookupError, Exception):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
except Exception:
    def word_tokenize(text):
        return re.findall(r'\b\w+\b', str(text))

st.set_page_config(page_title="Agri-Market Predictor", page_icon="🌱", layout="wide")


st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(46, 123, 50, 0.2);
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(46, 204, 113, 0.15);
    border-color: #2e7b32;
}
div[data-testid="stMetricValue"] {
    color: #2e7b32;
    font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    font-weight: 600;
    border-radius: 8px 8px 0px 0px;
    padding: 8px 16px;
}
</style>
""", unsafe_allow_html=True)



st.title("Pakistan Agri-Market Price Predictor & Advisor 🌱")

import requests
from streamlit_lottie import st_lottie

# 1. Improved Loader with a Try/Except block so it never crashes the app
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=1.5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_url = "https://assets8.lottiefiles.com/packages/lf20_qp1q7mct.json"
lottie_plant = load_lottieurl(lottie_url)

@st.cache_data
def load_data():
    df = pd.read_csv('producer-prices_pak.csv')
    df = df[df['Element'] == 'Producer Price (LCU/tonne)']
    df = df[['Year', 'Item', 'Value']].dropna()
    
    animal_keywords = ['meat', 'egg', 'milk', 'wool', 'honey', 'silk', 'skin', 'hide', 'fat', 'cattle', 'chicken', 'sheep', 'goat', 'buffalo', 'camel', 'beef', 'mutton', 'poultry', 'beeswax', 'hair', 'grease', 'silk-worm']
    pattern = '|'.join(animal_keywords)
    df = df[~df['Item'].str.contains(pattern, case=False, na=False)]
    
    df['Price_Per_Maund'] = df['Value'] / 25
    return df

df = load_data()

def clean_for_prolog(text):
    base = text.lower().split(',')[0].strip().replace(' ', '_')
    return re.sub(r'[^a-z0-9_]', '', base)

major_crops_kb = {
    'wheat': {'shelf': 'long', 'sow': 'rabi_nov_dec', 'harvest': 'apr_may', 'water': 'moderate_4_to_5_irrigations', 'fert': 'dap_and_urea', 'disease': 'rust_fungus'},
    'rice': {'shelf': 'long', 'sow': 'kharif_may_jun', 'harvest': 'oct_nov', 'water': 'very_high_flooded_fields', 'fert': 'zinc_and_nitrogen', 'disease': 'stem_borer'},
    'cotton': {'shelf': 'long', 'sow': 'kharif_apr_may', 'harvest': 'aug_oct', 'water': 'moderate_crucial_at_flowering', 'fert': 'phosphorus_and_potassium', 'disease': 'whitefly_and_pink_bollworm'},
    'sugarcane': {'shelf': 'long', 'sow': 'spring_feb_mar', 'harvest': 'nov_mar', 'water': 'high_frequent_irrigation', 'fert': 'heavy_nitrogen', 'disease': 'red_rot'},
    'maize': {'shelf': 'medium', 'sow': 'spring_or_kharif', 'harvest': 'jun_or_nov', 'water': 'moderate_sensitive_to_drought', 'fert': 'nitrogen_rich', 'disease': 'shoot_fly'},
    'apples': {'shelf': 'long', 'sow': 'spring_feb_mar', 'harvest': 'aug_oct', 'water': 'moderate_drip_preferred', 'fert': 'npk_mix', 'disease': 'apple_scab'},
    'bananas': {'shelf': 'short', 'sow': 'spring_or_autumn', 'harvest': 'all_year', 'water': 'high_needs_moist_soil', 'fert': 'potassium_rich', 'disease': 'bunchy_top_virus'},
    'onions': {'shelf': 'medium', 'sow': 'rabi_oct_nov', 'harvest': 'apr_may', 'water': 'light_but_frequent', 'fert': 'sulfur_and_phosphorus', 'disease': 'purple_blotch'},
    'tomatoes': {'shelf': 'short', 'sow': 'spring_or_autumn', 'harvest': 'multiple_seasons', 'water': 'moderate_avoid_leaf_wetness', 'fert': 'calcium_and_potassium', 'disease': 'blight_and_leaf_curl'},
    'potatoes': {'shelf': 'medium', 'sow': 'autumn_sep_oct', 'harvest': 'jan_feb', 'water': 'moderate_consistent_moisture', 'fert': 'phosphorus_heavy', 'disease': 'late_blight'},
    'almonds': {'shelf': 'long', 'sow': 'winter_dec_jan', 'harvest': 'aug_sep', 'water': 'low_drought_tolerant', 'fert': 'nitrogen_and_zinc', 'disease': 'hull_rot'},
    'cabbages': {'shelf': 'short', 'sow': 'winter_oct_nov', 'harvest': 'jan_mar', 'water': 'high_consistent_moisture', 'fert': 'nitrogen_rich', 'disease': 'diamondback_moth'},
    'apricots': {'shelf': 'short', 'sow': 'winter_dec_jan', 'harvest': 'may_jul', 'water': 'low_well_drained_soil', 'fert': 'nitrogen_in_spring', 'disease': 'brown_rot'},
    'barley': {'shelf': 'long', 'sow': 'rabi_oct_nov', 'harvest': 'apr_may', 'water': 'low_drought_resistant', 'fert': 'moderate_nitrogen', 'disease': 'powdery_mildew'},
    'beans': {'shelf': 'long', 'sow': 'spring_feb_mar', 'harvest': 'may_jun', 'water': 'moderate_at_pod_formation', 'fert': 'low_nitrogen_fixes_own', 'disease': 'anthracnose'},
    'carrots_and_turnips': {'shelf': 'medium', 'sow': 'autumn_sep_oct', 'harvest': 'dec_feb', 'water': 'moderate_deep_watering', 'fert': 'low_nitrogen_high_potassium', 'disease': 'root_knot_nematodes'}
}

def get_crop_kb(name):
    clean = clean_for_prolog(name)
    if clean in major_crops_kb:
        return major_crops_kb[clean]
    for k, v in major_crops_kb.items():
        if k in clean or clean in k:
            return v
    return {
        'shelf': 'medium',
        'sow': 'varies_by_region',
        'harvest': 'depends_on_sowing_time',
        'water': 'moderate_depends_on_soil',
        'fert': 'standard_npk',
        'disease': 'general_pests'
    }

PROLOG_READY = False
try:
    from pyswip import Prolog
    prolog = Prolog()

    unique_crops = df['Item'].unique()
    for item in unique_crops:
        safe_name = clean_for_prolog(item)
        crop_info = get_crop_kb(safe_name)
        shelf = crop_info['shelf']
        sow = crop_info['sow']
        harvest = crop_info['harvest']
        water = crop_info['water']
        fert = crop_info['fert']
        disease = crop_info['disease']
            
        prolog.assertz(f"shelf_life({safe_name}, {shelf})")
        prolog.assertz(f"sowing_season({safe_name}, {sow})")
        prolog.assertz(f"harvesting_season({safe_name}, {harvest})")
        prolog.assertz(f"water_req({safe_name}, {water})")
        prolog.assertz(f"fert_req({safe_name}, {fert})")
        prolog.assertz(f"disease_risk({safe_name}, {disease})")

    prolog.assertz("subsidy(dap_fertilizer, available_in_punjab_sindh)")
    prolog.assertz("subsidy(solar_tubewell, available_for_small_farmers)")
    prolog.assertz("subsidy(certified_seeds, available_for_wheat_and_cotton)")
    
    prolog.assertz("advise_storage(Crop) :- price_trend(Crop, falling), shelf_life(Crop, long)")
    prolog.assertz("advise_sowing(Crop, Season) :- sowing_season(Crop, Season)")
    prolog.assertz("advise_harvest(Crop, Season) :- harvesting_season(Crop, Season)")
    prolog.assertz("advise_water(Crop, Req) :- water_req(Crop, Req)")
    prolog.assertz("advise_fert(Crop, Req) :- fert_req(Crop, Req)")
    prolog.assertz("advise_disease(Crop, Risk) :- disease_risk(Crop, Risk)")
    
    PROLOG_READY = True
except ImportError:
    pass
except Exception as e:
    st.sidebar.warning("Running AI Advisor via built-in expert engine.")

with st.sidebar:
    # 3. The Safeguard: Only show the animation if it successfully downloaded
    if lottie_plant is not None:
        st_lottie(lottie_plant, height=150, key="plant_animation")
    else:
        # If the internet fails, it falls back to the static image safely
        st.image("https://cdn-icons-png.flaticon.com/512/2885/2885069.png", width=80)
        
    st.markdown("## ⚙️ Market Settings")
with st.sidebar:
    crop_list = sorted(list(df['Item'].unique()))
    default_crop = st.query_params.get("crop", "Wheat")
    wheat_idx = crop_list.index(default_crop) if default_crop in crop_list else (crop_list.index('Wheat') if 'Wheat' in crop_list else 0)
    selected_crop = st.selectbox("Select a Crop/Vegetable:", crop_list, index=wheat_idx)
    model_choice = st.radio("Prediction Algorithm:", ("Linear Regression", "Decision Tree"))
    st.markdown("---")
    st.info("Developed by:\n- Syeda Ummelaila Raza\n- Noor ul Hassan\n- Syed Abdullah")

crop_data = df[df['Item'] == selected_crop].sort_values(by='Year')

if len(crop_data) > 5:
    X = crop_data[['Year']].values
    y = crop_data['Price_Per_Maund'].values
    
    model = LinearRegression() if model_choice == "Linear Regression" else DecisionTreeRegressor()
    model.fit(X, y)
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ... inside your model fitting logic ...
    y_pred = model.predict(X)
    accuracy_score = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred)) # Standard deviation of residuals
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    last_year = int(crop_data['Year'].max())
    future_years = np.array([[last_year + 1], [last_year + 2], [last_year + 3]])
    predictions = model.predict(future_years)
    
    accuracy_score = r2_score(y, model.predict(X))
    trend = "falling" if predictions[-1] < y[-1] else "rising"
    trend_ur = "gir raha" if trend == "falling" else "barh raha"

tab1, tab2, tab3 = st.tabs(["📈 ML Price Forecast", "🤖 AI Advisor Chat", "📊 Raw Data & Export"])

with tab1:
    st.subheader(f"Market Overview: {selected_crop.title()}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Latest Recorded Price (Per Maund)", value=f"{y[-1]:,.2f} PKR")
    with col2:
        price_diff = predictions[-1] - y[-1]
        st.metric(label="Average 3-Year Forecasted Price", value=f"{predictions[-1]:,.2f} PKR", delta=f"{price_diff:,.2f} PKR")
    with col3:
        st.markdown("---")
    st.write("### 🔍 Model Reliability Metrics")
    
    # Second Row: Evaluation
    eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
    
    with eval_col1:
        st.metric(label="R² Score (Fit)", value=f"{accuracy_score:.2f}", help="1.0 is a perfect fit. Lower means more volatility.")
    
    with eval_col2:
        st.metric(label="Avg Error (MAE)", value=f"{mae:,.2f} PKR", help="On average, the prediction is off by this many Rupees.")
        
    with eval_col3:
        st.metric(label="Percentage Error", value=f"{mape:.1f}%", help="The average percentage the model deviates from real prices.")

    with eval_col4:
        st.metric(label="RMSE", value=f"{rmse:,.2f}", help="Root Mean Squared Error: Highlights if there are very large 'misses' in the data.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=crop_data['Year'], y=crop_data['Price_Per_Maund'], mode='lines+markers', name='Historical Data', line=dict(color='#2e7b32', width=3)))
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=predictions, mode='lines+markers', name='ML Forecast', line=dict(color='#d32f2f', width=3, dash='dash')))
    fig.update_layout(xaxis_title="Year", yaxis_title="Price (PKR per 40kg Maund)", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Chat with the Agri-Advisor about {selected_crop.title()}")
    
    # 1. Create a scrollable container with a fixed height
    # This ensures the input bar stays pinned at the bottom
    chat_container = st.container()

    # 2. Display existing messages inside the scrollable container
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Assalam o Alaikum! I am your AI Agri-Advisor. Ask me about pricing, storage, sowing, water, fertilizer, or diseases in English or Roman Urdu."}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.markdown("**Quick Inquiries:**")
    prompt_cols = st.columns(4)
    quick_choice = None
    with prompt_cols[0]:
        if st.button("📦 Store ya bechun?", key="btn_store"):
            quick_choice = "Fasal store karun ya abhi bechun?"
    with prompt_cols[1]:
        if st.button("🌱 Sowing Season?", key="btn_sow"):
            quick_choice = "Boai ka behtareen mausam kab hai?"
    with prompt_cols[2]:
        if st.button("💧 Water Needs?", key="btn_water"):
            quick_choice = "Pani kitna dena chahiye?"
    with prompt_cols[3]:
        if st.button("🏛️ Subsidies?", key="btn_subsidy"):
            quick_choice = "Hukoomat ki subsidies kya hain?"

    chat_input_val = st.chat_input("Message the AI Advisor...")
    prompt = quick_choice or chat_input_val
    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately in the container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # --- Your Processing Logic ---
        tokens = [word.lower() for word in word_tokenize(prompt)]
        urdu_indicators = ['kya', 'hai', 'ki', 'kab', 'kaise', 'keemat', 'rakhun', 'lagaun', 'katai', 'madad', 'bhao', 'boai', 'aur', 'mein', 'ko', 'k', 'se', 'yeh', 'woh', 'khaad', 'pani', 'beemari', 'dun', 'dalun','mashwara', 'bechun']
        is_urdu = any(word in tokens for word in urdu_indicators)
        
        # 2. Add mashwara to intent_storage since that triggers the Advice logic
        intent_storage = any(word in tokens for word in ['store', 'storage', 'rakhun', 'save', 'mahfooz', 'mashwara', 'bechun', 'hold'])
        intent_price = any(word in tokens for word in ['price', 'keemat', 'rate', 'bhao','qeemat', 'cost', 'forecast'])
        intent_sowing = any(word in tokens for word in ['sow', 'sowing', 'lagaun', 'boai', 'plant', 'seed'])
        intent_harvest = any(word in tokens for word in ['harvest', 'katai', 'cut', 'picking'])
        intent_subsidy = any(word in tokens for word in ['subsidy', 'madad', 'scheme', 'government', 'kisan', 'hukoomat'])
        intent_water = any(word in tokens for word in ['water', 'pani', 'irrigation', 'aabpashi', 'dun'])
        intent_fert = any(word in tokens for word in ['fertilizer', 'khaad', 'urea', 'dap', 'dalun', 'potash'])
        intent_disease = any(word in tokens for word in ['disease', 'pest', 'beemari', 'keerey', 'spray', 'virus', 'kungi', 'borer'])
        
        safe_crop_name = clean_for_prolog(selected_crop)
        pretty_crop_name = selected_crop.split(',')[0].title()
        crop_kb = get_crop_kb(selected_crop)
        
        reply = ""

        # Logic Branching (Prolog / ML / Expert KB)
        if intent_price:
            current_price = y[-1] if len(crop_data) > 5 else 0
            if is_urdu: reply = f"**{pretty_crop_name}** ki aakhri record karda keemat **{current_price:,.2f} PKR** fi Maund thi. ML forecast ke mutabiq market ka trend **{trend_ur}** hai."
            else: reply = f"The last recorded price for **{pretty_crop_name}** was **{current_price:,.2f} PKR** per Maund. The ML forecast trend is **{trend}**."
            
        elif intent_storage:
            should_store = False
            if PROLOG_READY:
                prolog.retractall("price_trend(_,_)")
                prolog.assertz(f"price_trend({safe_crop_name}, {trend})")
                advice_query = list(prolog.query(f"advise_storage({safe_crop_name})"))
                should_store = bool(advice_query)
            else:
                should_store = (trend == "falling" and crop_kb['shelf'] == 'long')

            if should_store:
                reply = f"🧠 **Agronomy Advisory:** Keemat gir rahi hai aur **{pretty_crop_name}** ki shelf life lambi hai. Mashwara yeh hai ke fasal ko **store kar lein** taqay baad mein behtar rate milay." if is_urdu else f"🧠 **Agronomy Advisory:** Price trend is softening and **{pretty_crop_name}** has a durable shelf life. It is advised to **store the crop** in dry storage and sell when market supplies tighten."
            else:
                reply = f"🧠 **Agronomy Advisory:** Abhi lambay arsay ke liye store karna munasib nahi hai. Fasal bechne par ghaur karein taqay kharab na ho." if is_urdu else f"🧠 **Agronomy Advisory:** Long-term storage is not strictly recommended right now. Consider staged selling to capture present market value."

        elif intent_sowing:
            sow_season = None
            if PROLOG_READY:
                sow_query = list(prolog.query(f"advise_sowing({safe_crop_name}, Season)"))
                if sow_query:
                    sow_season = sow_query[0]['Season'].replace('_', ' ').title()
            if not sow_season:
                sow_season = crop_kb['sow'].replace('_', ' ').title()
            reply = f"🌱 **Agronomy Advisory:** **{pretty_crop_name}** ko lagane (sowing) ka behtareen mausam **{sow_season}** hai." if is_urdu else f"🌱 **Agronomy Advisory:** The optimal sowing season for **{pretty_crop_name}** is **{sow_season}**."

        elif intent_harvest:
            harv_season = None
            if PROLOG_READY:
                harv_query = list(prolog.query(f"advise_harvest({safe_crop_name}, Season)"))
                if harv_query:
                    harv_season = harv_query[0]['Season'].replace('_', ' ').title()
            if not harv_season:
                harv_season = crop_kb['harvest'].replace('_', ' ').title()
            reply = f"🌾 **Agronomy Advisory:** **{pretty_crop_name}** ki katai (harvest) ka behtareen waqt **{harv_season}** hai." if is_urdu else f"🌾 **Agronomy Advisory:** The harvest timeframe for **{pretty_crop_name}** is **{harv_season}**."

        elif intent_water:
            water_req = None
            if PROLOG_READY:
                water_query = list(prolog.query(f"advise_water({safe_crop_name}, Req)"))
                if water_query:
                    water_req = water_query[0]['Req'].replace('_', ' ').title()
            if not water_req:
                water_req = crop_kb['water'].replace('_', ' ').title()
            reply = f"💧 **Agronomy Advisory:** **{pretty_crop_name}** ke liye pani ki zaroorat: **{water_req}**." if is_urdu else f"💧 **Agronomy Advisory:** Water requirement for **{pretty_crop_name}**: **{water_req}**."

        elif intent_fert:
            fert_req = None
            if PROLOG_READY:
                fert_query = list(prolog.query(f"advise_fert({safe_crop_name}, Req)"))
                if fert_query:
                    fert_req = fert_query[0]['Req'].replace('_', ' ').title()
            if not fert_req:
                fert_req = crop_kb['fert'].replace('_', ' ').title()
            reply = f"🌱 **Agronomy Advisory:** **{pretty_crop_name}** ke liye behtareen khaad: **{fert_req}**." if is_urdu else f"🌱 **Agronomy Advisory:** Recommended fertilizer for **{pretty_crop_name}**: **{fert_req}**."

        elif intent_disease:
            disease_risk = None
            if PROLOG_READY:
                disease_query = list(prolog.query(f"advise_disease({safe_crop_name}, Risk)"))
                if disease_query:
                    disease_risk = disease_query[0]['Risk'].replace('_', ' ').title()
            if not disease_risk:
                disease_risk = crop_kb['disease'].replace('_', ' ').title()
            reply = f"🐛 **Agronomy Advisory:** **{pretty_crop_name}** ko in beemariyon/keeron ka khatra hai: **{disease_risk}**. Waqt par spray aur field scouting karein." if is_urdu else f"🐛 **Agronomy Advisory:** Common pest/disease threats for **{pretty_crop_name}**: **{disease_risk}**."

        elif intent_subsidy:
            if not is_urdu:
                reply = "🏛️ **Agricultural Subsidies (Pakistan):**\n- **DAP Fertilizer**: Rs 1,000–1,500 subsidy on Kisan/Hari cards\n- **Solar Tubewells**: 80% government grant for small farmers (<12.5 acres)\n- **Certified Seeds**: Subsidized certified wheat & Bt cotton seeds\n"
            else:
                reply = "🏛️ **سرکاری زرعی سبسڈیز:**\n- **ڈی اے پی کھاد**: کسان کارڈ پر 1000 سے 1500 روپے سبسڈی\n- **سولر ٹیوب ویل**: چھوٹے کسانوں کے لیے 80 فیصد حکومتی امداد\n- **تصدیق شدہ بیج**: گندم اور کپاس کے بیجوں پر خصوصی رعایت\n"
                
        else:
            if is_urdu: reply = "Main **keemat**, **mashwara**, **lagaun**, **katai**, **pani**, **khaad**, **beemari**, ya **madad** ke bare mein sawalat samajh sakta hoon."
            else: reply = "I can process intents regarding **price**, **storage**, **sowing**, **harvest**, **water**, **fertilizer**, **disease**, or **subsidy**."

        # Display assistant response in the container
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(reply)
        
        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": reply})
        
        # Force a rerun to scroll to the bottom and refresh state
        st.rerun()
with tab3:
    # Calculate residuals
    residuals = y - model.predict(X)

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
    x=crop_data['Year'], 
    y=residuals, 
    mode='markers',
    marker=dict(color='#ff5252', size=10),
    name='Residuals'
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="white")
    fig_res.update_layout(
        title="Error Analysis (Residuals)",
        xaxis_title="Year",
        yaxis_title="Error (Actual - Predicted)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig_res.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig_res.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    st.plotly_chart(fig_res, use_container_width=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=crop_data['Price_Per_Maund'],
        nbinsx=15,
        marker_color='#2ecc71',
        opacity=0.75
    ))
    fig_dist.update_layout(
        title="Historical Price Distribution",
        xaxis_title="Price Range (PKR)",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig_dist.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig_dist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    st.plotly_chart(fig_dist, use_container_width=True)
    st.subheader(f"Underlying Market Data for {selected_crop.title()}")
    st.dataframe(crop_data[['Year', 'Item', 'Value', 'Price_Per_Maund']], use_container_width=True)
    
    st.markdown("### 📥 Export Data")
    
    # 1. Prepare Forecast CSV
    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Price_PKR': predictions})
    forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')
    
    # 2. Prepare Raw Historical CSV
    raw_csv = crop_data[['Year', 'Item', 'Value', 'Price_Per_Maund']].to_csv(index=False).encode('utf-8')
    
    # 3. Create columns for side-by-side buttons
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        st.download_button(
            label="📈 Download 3-Year Forecast (CSV)", 
            data=forecast_csv, 
            file_name=f"{clean_for_prolog(selected_crop)}_forecast.csv", 
            mime="text/csv",
            use_container_width=True
        )
        
    with dl_col2:
        st.download_button(
            label="📊 Download Raw Historical Data (CSV)", 
            data=raw_csv, 
            file_name=f"{clean_for_prolog(selected_crop)}_historical_data.csv", 
            mime="text/csv",
            use_container_width=True
        )