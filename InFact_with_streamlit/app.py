import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration with custom theme
st.set_page_config(
    page_title="InFact - Smart Food Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Load animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.warning(f"Failed to load animation: {e}")
        return None

# Add a default placeholder image URL - using a more reliable source
PLACEHOLDER_IMAGE = "https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/assets/streamlit-mark-color.png"

# Update the safe_lottie function
def safe_lottie(animation_data, **kwargs):
    try:
        if animation_data is not None:
            st_lottie(animation_data, **kwargs)
        else:
            # Fallback to placeholder image with error handling
            try:
                st.image(PLACEHOLDER_IMAGE, use_container_width=True)
            except:
                # If even the placeholder fails, show a colored box with text
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        margin: 10px 0;
                    ">
                        <p style="color: #31333F; margin: 0;">Visualization Placeholder</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.warning(f"Error displaying animation: {str(e)}")
        try:
            st.image(PLACEHOLDER_IMAGE, use_container_width=True)
        except:
            st.error("Failed to display visualization")

# Update animations dictionary with working Lottie URLs
animations = {
    'food': load_lottieurl("https://lottie.host/c99f6338-a7aa-48c4-ad19-94f8f0c73a40/3DI4hKzM4k.json"),
    'search': load_lottieurl("https://lottie.host/0912f275-b615-4ab9-91e6-e8ac187b5424/DPOCx7gsJu.json"),
    'analytics': load_lottieurl("https://lottie.host/8a90f966-4c87-4fab-8dee-c486e2efa024/WPcIU2i71Y.json"),
    'loading': load_lottieurl("https://lottie.host/91f2fcc4-7c92-4a5b-85e6-f4134b6691f9/8slKrm2rPN.json")
}

# Enhanced CSS with theme support and animations
def get_css():
    return f"""
    <style>
    /* Global Theme Variables */
    :root {{
        --bg-primary: {('#1a1a2e' if st.session_state.theme == 'dark' else '#ffffff')};
        --bg-secondary: {('#16213e' if st.session_state.theme == 'dark' else '#f8f9fa')};
        --text-primary: {('#e6e6e6' if st.session_state.theme == 'dark' else '#1a1a2e')};
        --accent-color: #4CAF50;
        --accent-hover: #45a049;
    }}

    /* Global Styles */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
        color: var(--text-primary);
        transition: all 0.3s ease;
    }}
    
    /* Modern Cards */
    .modern-card {{
        background: {('rgba(255, 255, 255, 0.05)' if st.session_state.theme == 'dark' else 'rgba(255, 255, 255, 0.9)')};
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid {('rgba(255, 255, 255, 0.1)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.1)')};
        box-shadow: 0 8px 32px {('rgba(0, 0, 0, 0.1)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.05)')};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .modern-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px {('rgba(0, 0, 0, 0.2)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.1)')};
    }}
    
    /* Buttons */
    .stButton button {{
        background: linear-gradient(45deg, var(--accent-color), var(--accent-hover));
        color: white;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }}
    
    /* Search Box */
    .stTextInput input {{
        background: {('rgba(255, 255, 255, 0.05)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.05)')};
        border: 2px solid rgba(76, 175, 80, 0.3);
        border-radius: 15px;
        color: var(--text-primary);
        padding: 1rem;
        transition: all 0.3s ease;
    }}
    
    .stTextInput input:focus {{
        border-color: var(--accent-color);
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.2);
        transform: translateY(-2px);
    }}
    
    /* Select Box */
    .stSelectbox select {{
        background: {('rgba(255, 255, 255, 0.05)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.05)')};
        border-radius: 15px;
        border: 2px solid rgba(76, 175, 80, 0.3);
        color: var(--text-primary);
        transition: all 0.3s ease;
    }}
    
    /* Navigation */
    .nav-link {{
        background: {('rgba(255, 255, 255, 0.05)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.05)')} !important;
        border-radius: 10px !important;
        margin: 5px !important;
        transition: all 0.3s ease !important;
    }}
    
    .nav-link:hover {{
        background: rgba(76, 175, 80, 0.1) !important;
        transform: translateY(-2px) !important;
    }}
    
    .nav-link.active {{
        background: linear-gradient(45deg, var(--accent-color), var(--accent-hover)) !important;
        color: white !important;
    }}

    /* Loading Animation */
    @keyframes skeleton-loading {{
        0% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0 50%; }}
    }}

    .skeleton {{
        background: linear-gradient(90deg, 
            {('rgba(255, 255, 255, 0.05)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.05)')} 25%, 
            {('rgba(255, 255, 255, 0.1)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.1)')} 37%, 
            {('rgba(255, 255, 255, 0.05)' if st.session_state.theme == 'dark' else 'rgba(0, 0, 0, 0.05)')} 63%);
        background-size: 400% 100%;
        animation: skeleton-loading 1.4s ease infinite;
    }}

    /* Toast Notifications */
    .toast {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 1rem 2rem;
        border-radius: 10px;
        background: var(--accent-color);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.3s ease forwards;
    }}

    @keyframes slideIn {{
        from {{ transform: translateX(100%); }}
        to {{ transform: translateX(0); }}
    }}

    /* Theme Toggle */
    .theme-toggle {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .modern-card {{
            padding: 1rem;
        }}
        
        .stButton button {{
            width: 100%;
        }}
    }}

    /* Header Styles */
    .header {{
        background: linear-gradient(45deg, var(--accent-color), var(--accent-hover));
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }}

    .header h1 {{
        margin: 0;
        font-size: 2rem;
    }}

    .header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }}
    </style>
    """

# Inject CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Theme toggle
with st.sidebar:
    if st.button("🌓 Toggle Theme"):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.experimental_rerun()

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('food_data_updated.csv')
    data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
    if 'is_harmful?' not in data.columns:
        data['is_harmful?'] = 'No'
    data.fillna("N/A", inplace=True)
    data['harmful_ingredient_count'] = pd.to_numeric(data.get('harmful_ingredient_count', 0), errors='coerce').fillna(0).astype(int)
    data['total_ingredients'] = pd.to_numeric(data.get('total_ingredients', 0), errors='coerce').fillna(0).astype(int)
    return data

# Show loading animation
with st.spinner("Loading data..."):
    safe_lottie(animations['loading'], height=200, key="loading")
    data = load_data()
    time.sleep(1)  # Simulate loading for demo

# Modern Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "Search", "Analytics", "About"],
    icons=["house-heart-fill", "search-heart", "graph-up", "info-circle-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"font-size": "1.2rem", "margin-right": "8px"},
        "nav-link": {
            "font-size": "1rem", 
            "text-align": "center", 
            "margin": "0px", 
            "--hover-color": "#4CAF50"
        },
        "nav-link-selected": {"background-color": "#4CAF50"},
    }
)

if selected == "Home":
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
            <div class="header">
                <h1>InFact: Your Smart Food Guide 🍎</h1>
                <p>Make informed decisions about your food choices</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="modern-card">
                <p style="font-size: 1.2rem; margin-top: 20px;">
                    Discover what's really in your food with our advanced analysis:
                </p>
                <ul style="font-size: 1.1rem; margin-top: 15px;">
                    <li>🔍 Ingredient composition analysis</li>
                    <li>⚠️ Harmful ingredients detection</li>
                    <li>📊 Nutritional impact assessment</li>
                    <li>🥗 Healthy alternatives suggestions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Quick Stats
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Products", f"{len(data):,}", "Updated daily")
        with col_stat2:
            harmful_count = len(data[data['is_harmful?'].str.lower() == 'yes'])
            st.metric("Harmful Products", f"{harmful_count:,}", f"{(harmful_count/len(data))*100:.1f}%")
        with col_stat3:
            safe_count = len(data[data['is_harmful?'].str.lower() == 'no'])
            st.metric("Safe Products", f"{safe_count:,}", f"{(safe_count/len(data))*100:.1f}%")
    
    with col2:
        safe_lottie(animations['food'], height=400, key="food_animation")

elif selected == "Search":
    st.markdown("""
        <div class="header">
            <h1>Smart Product Search 🔍</h1>
            <p>Find detailed information about food products</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced search interface
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input("", placeholder="Search for a product...", help="Enter product name")
    with col2:
        category_filter = st.selectbox(
            "Category",
            options=["All"] + sorted(data["category"].dropna().unique().tolist())
        )
    with col3:
        harmful_filter = st.radio(
            "Show harmful?",
            ["All", "Yes", "No"],
            horizontal=True
        )

    # Filter data
    filtered_data = data.copy()
    if category_filter != "All":
        filtered_data = filtered_data[filtered_data["category"] == category_filter]
    if harmful_filter != "All":
        filtered_data = filtered_data[filtered_data["is_harmful?"].str.lower() == harmful_filter.lower()]

    # Search functionality
    if query:
        with st.spinner("Searching..."):
            filtered_data = data.copy()
            if category_filter != "All":
                filtered_data = filtered_data[filtered_data['category'] == category_filter]
            if harmful_filter != "All":
                filtered_data = filtered_data[filtered_data['is_harmful?'].str.lower() == harmful_filter.lower()]
            
            # Improved search logic
            mask = filtered_data['product_name'].str.contains(query, case=False, na=False)
            search_results = filtered_data[mask]
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching products")
                
                # Display top 5 matches
                for _, result in search_results.head(5).iterrows():
                    with st.container():
                        st.markdown(f"""
                            <div class="modern-card">
                                <h3>{result['product_name']}</h3>
                                <p><strong>Brand:</strong> {result['brand']}</p>
                                <p><strong>Category:</strong> {result['category']}</p>
                                <p><strong>Safety Status:</strong> 
                                    <span style="color: {'#ef4444' if result['is_harmful?'].lower() == 'yes' else '#4CAF50'}">
                                        {result['is_harmful?']}
                                    </span>
                                </p>
                                <div class="neumorphic" style="margin-top: 1rem; padding: 1rem;">
                                    <p><strong>Total Ingredients:</strong> {result['total_ingredients']}</p>
                                    <p><strong>Harmful Ingredients:</strong> {result['harmful_ingredient_count']}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show ingredient composition
                        col1, col2 = st.columns(2)
                        with col1:
                            if result.get('nutritional_impact'):
                                st.info(f"**Nutritional Impact:** {result['nutritional_impact']}")
                        with col2:
                            if result.get('healthy_alternatives'):
                                st.success(f"**Healthy Alternative:** {result['healthy_alternatives']}")
            else:
                st.warning(f"No products found matching '{query}'. Try a different search term or adjust filters.")
                # Show suggested searches
                similar_products = filtered_data[
                    filtered_data['product_name'].str.contains('|'.join(query.split()), case=False, na=False)
                ]
                if not similar_products.empty:
                    st.markdown("### You might be interested in:")
                    for _, prod in similar_products.head(3).iterrows():
                        st.markdown(f"- {prod['product_name']} ({prod['brand']})")

elif selected == "Analytics":
    st.markdown("""
        <div class="header">
            <h1>Product Analytics 📊</h1>
            <p>Insights and trends from our food database</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Interactive filters
    col1, col2 = st.columns(2)
    with col1:
        selected_categories = st.multiselect(
            "Filter by Categories",
            options=sorted(data["category"].unique()),
            default=sorted(data["category"].unique())[:5]
        )
    with col2:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar", "Pie", "Line", "Scatter"]
        )
    
    # Filtered data
    filtered_data = data[data['category'].isin(selected_categories)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category Distribution
        category_counts = filtered_data['category'].value_counts()
        if chart_type == "Bar":
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Products by Category",
                labels={'x': 'Category', 'y': 'Count'}
            )
        elif chart_type == "Pie":
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Category Distribution"
            )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=('white' if st.session_state.theme == 'dark' else 'black'))
        )
        st.plotly_chart(fig)
    
    with col2:
        # Harmful vs Non-harmful
        harmful_counts = filtered_data['is_harmful?'].value_counts()
        fig = px.pie(
            values=harmful_counts.values,
            names=harmful_counts.index,
            title="Harmful vs Non-harmful Products",
            hole=0.3
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=('white' if st.session_state.theme == 'dark' else 'black'))
        )
        st.plotly_chart(fig)
    
    # Brand Analysis
    st.markdown("""
        <div class="modern-card">
            <h3>Top Brands Analysis</h3>
        </div>
    """, unsafe_allow_html=True)
    
    top_brands = filtered_data['brand'].value_counts().head(10)
    fig = px.bar(
        x=top_brands.index,
        y=top_brands.values,
        title="Top 10 Brands by Product Count",
        labels={'x': 'Brand', 'y': 'Number of Products'}
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=('white' if st.session_state.theme == 'dark' else 'black'))
    )
    st.plotly_chart(fig)

elif selected == "About":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="header">
                <h1>About InFact 🌟</h1>
                <p>Your trusted companion for food safety</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="modern-card">
                <p style="font-size: 1.1em; margin-top: 20px;">
                    InFact is your trusted companion in making informed decisions about your food choices. 
                    Our mission is to provide transparent, accurate information about food products, 
                    helping you maintain a healthy lifestyle through informed choices.
                </p>
            </div>
            
            <div class="modern-card">
                <h3>Our Mission</h3>
                <p>To empower consumers with knowledge about their food choices through:</p>
                <ul>
                    <li>🔬 Comprehensive ingredient analysis</li>
                    <li>⚠️ Identification of potentially harmful ingredients</li>
                    <li>🥗 Suggesting healthy alternatives</li>
                    <li>📚 Promoting food safety awareness</li>
                </ul>
            </div>
            
            <div class="modern-card">
                <h3>Contact Us</h3>
                <p>📧 Email: <a href="mailto:infactsap2025@gmail.com">infactsap2025@gmail.com</a></p>
                <p>📍 PDEA COEM, Pune</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        safe_lottie(animations['analytics'], height=300, key="about_animation")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px;">`
        <p style="color: rgba(255,255,255,0.6);">Made with ❤️ by Team InFact</p>
        <p style="color: rgba(255,255,255,0.4);">© 2025 All rights reserved | PDEA COEM Pune</p>
    </div>
""", unsafe_allow_html=True)