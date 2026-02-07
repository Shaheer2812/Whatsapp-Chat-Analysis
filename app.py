import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- Page Config ---
st.set_page_config(
    page_title="WhatsApp Chat Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #f8f9fb; }
    .stApp { background-color: #f8f9fb; }
    .block-container { padding: 2rem 3rem; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e6e9ef; }
    section[data-testid="stSidebar"] > div { background-color: #ffffff; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #128C7E !important; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stText, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: #31333F !important; }
    
    /* Headers */
    h1, h2, h3 { color: #1a1d24 !important; }
    h1 { font-weight: 700 !important; }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label { color: #5f6368 !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1a1d24 !important; font-size: 1.8rem !important; font-weight: 600 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #f1f3f4; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent; 
        border-radius: 6px; 
        color: #5f6368; 
        font-weight: 500;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #128C7E !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(37, 211, 102, 0.3); }
    
    /* File uploader */
    section[data-testid="stFileUploader"] { 
        border: 2px dashed #25D366; 
        border-radius: 12px; 
        padding: 1rem;
        background: rgba(37, 211, 102, 0.05);
    }
    
    /* Divider */
    hr { border-color: #e6e9ef !important; margin: 1.5rem 0 !important; }
    
    /* Info boxes */
    .stAlert { border-radius: 8px; }
    
    /* Progress bar */
    .stProgress > div > div { background-color: #25D366 !important; }
</style>
""", unsafe_allow_html=True)

# --- Download NLTK data ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# --- Parser Function ---
def parse_whatsapp_chat(file_content):
    """Parses WhatsApp chat file content and returns a DataFrame."""
    patterns = [
        r'^\[(\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2})\] (.*?): (.*)$',
        r'^(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.*?): (.*)$'
    ]
    
    lines = file_content.decode('utf-8').split('\n')
    data = []
    message_buffer = []
    date_str, author = None, None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        match = None
        for p in patterns:
            match = re.match(p, line)
            if match:
                break
        
        if match:
            if author:
                data.append([date_str, author, ' '.join(message_buffer)])
            message_buffer = []
            date_str = match.group(1)
            author = match.group(2)
            message = match.group(3)
            message_buffer.append(message)
        else:
            if author:
                message_buffer.append(line)

    if author:
        data.append([date_str, author, ' '.join(message_buffer)])

    df = pd.DataFrame(data, columns=['DateTime', 'Author', 'Message'])
    
    df['temp_date'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y, %H:%M', errors='coerce')
    mask = df['temp_date'].isna()
    if mask.any():
        df.loc[mask, 'temp_date'] = pd.to_datetime(df.loc[mask, 'DateTime'], format='%d/%m/%y, %H:%M:%S', errors='coerce')
    df['DateTime'] = df['temp_date']
    df.drop(columns=['temp_date'], inplace=True)
    
    return df

# --- Helper Functions ---
def get_talkativeness_rating(percentage):
    if percentage >= 30: return "🔥 Very Talkative"
    elif percentage >= 20: return "😄 Talkative"
    elif percentage >= 10: return "😊 Moderate"
    elif percentage >= 5: return "🤫 Quiet"
    else: return "🤐 Very Quiet"

def count_media(msg):
    media = {'images': 0, 'videos': 0, 'gifs': 0, 'stickers': 0, 'voice': 0, 'links': 0, 'deleted': 0}
    msg_lower = msg.lower()
    if 'image omitted' in msg_lower: media['images'] = 1
    if 'video omitted' in msg_lower: media['videos'] = 1
    if 'gif omitted' in msg_lower: media['gifs'] = 1
    if 'sticker omitted' in msg_lower: media['stickers'] = 1
    if 'audio omitted' in msg_lower or 'ptt omitted' in msg_lower: media['voice'] = 1
    if 'http://' in msg or 'https://' in msg: media['links'] = 1
    if 'this message was deleted' in msg_lower or 'you deleted this message' in msg_lower: media['deleted'] = 1
    return media

# --- Sidebar ---
with st.sidebar:
    st.markdown("# 🔍 Chat Analysis")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Upload WhatsApp Export", type=['txt'], help="Export from WhatsApp: Settings → Chats → Export Chat → Without Media")
    
    st.markdown("---")
    st.markdown("### 📖 How to Export")
    st.markdown("""
    1. Open WhatsApp Chat
    2. Tap ⋮ → More → Export Chat
    3. Choose **Without Media**
    4. Upload the `.txt` file here
    """)

# --- Landing Page ---
if uploaded_file is None:
    st.markdown("# 🔍 WhatsApp Chat Analysis")
    st.markdown("### Uncover hidden patterns in your conversations")
    st.markdown("---")
    
    st.markdown("#### Features:")
    st.markdown("""
    *   📊 Message statistics and trends
    *   👥 User comparisons and rankings
    *   ⏰ Activity patterns by time and day
    *   💬 Response time analysis
    """)
    
    st.markdown("#### Getting Started:")
    st.markdown("""
    1. Export your chat from WhatsApp (without media)
    2. Upload the .txt file using the sidebar
    3. Select participants and language
    4. Click \"Analyze Chat\"
    """)
    
    st.markdown("---")
    
    with st.expander("📱 How to Export Your Chat"):
        st.markdown("#### On iPhone:")
        st.markdown("""
        1. Open the chat in WhatsApp
        2. Tap the contact/group name at the top
        3. Scroll down and tap **Export Chat**
        4. Choose **Without Media**
        5. Save or share the .txt file
        """)
        
        st.markdown("#### On Android:")
        st.markdown("""
        1. Open the chat in WhatsApp
        2. Tap the three dots ( : ) menu
        3. Select **More** → **Export Chat**
        4. Choose **Without Media**
        5. Save or share the .txt file
        """)
    
    st.stop()

# --- Load and Parse Data ---
with st.spinner("🔍 Analyzing your chat..."):
    df = parse_whatsapp_chat(uploaded_file.getvalue())
    df = df[df['Author'] != 'Meta AI'].copy()
    
    # Save uploaded file
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    with open(os.path.join(upload_dir, uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getvalue())

# Filter out system messages
df_text = df[~df['Message'].str.contains('<Media omitted>|image omitted|video omitted|sticker omitted|audio omitted', case=False, na=False, regex=True)].copy()

# --- Header with key metrics ---
st.markdown(f"# 🔍 {uploaded_file.name.replace('.txt', '').replace('WhatsApp Chat with ', '')}")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("💬 Messages", f"{len(df):,}")
with col2:
    st.metric("👥 Participants", df['Author'].nunique())
with col3:
    days = max(1, (df['DateTime'].max() - df['DateTime'].min()).days)
    st.metric("📅 Days Active", f"{days:,}")
with col4:
    st.metric("📈 Msgs/Day", f"{len(df) // days:,}")
with col5:
    first_msg = df['DateTime'].min()
    st.metric("🗓️ Since", first_msg.strftime('%b %Y') if pd.notna(first_msg) else "N/A")

st.markdown("---")

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "⏰ Activity", "👥 Users"])

# ============================================
# TAB 1: OVERVIEW
# ============================================
with tab1:
    st.markdown("## 📊 Message Overview")
    
    st.markdown("### 💬 Messages per User")
    user_counts = df['Author'].value_counts().reset_index()
    user_counts.columns = ['User', 'Messages']
    user_counts['Percentage'] = (user_counts['Messages'] / user_counts['Messages'].sum() * 100).round(1)
    user_counts['Rating'] = user_counts['Percentage'].apply(get_talkativeness_rating)
    
    fig = px.bar(user_counts, x='Messages', y='User', orientation='h', 
                 color='Messages', color_continuous_scale='Viridis',
                 text=user_counts.apply(lambda x: f"{x['Messages']:,} ({x['Percentage']}%)", axis=1))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a1d24'), showlegend=False, coloraxis_showscale=False,
        yaxis=dict(categoryorder='total ascending'),
        height=max(300, len(user_counts) * 40),
        xaxis_title='', yaxis_title=''
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 User Statistics")
    stats_df = user_counts[['User', 'Messages', 'Percentage', 'Rating']].copy()
    stats_df['Percentage'] = stats_df['Percentage'].apply(lambda x: f"{x}%")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    st.markdown("### 📈 Message Trend Over Time")
    df['YearMonth'] = df['DateTime'].dt.to_period('M').astype(str)
    monthly = df.groupby('YearMonth').size().reset_index(name='Messages')
    fig_trend = px.area(monthly, x='YearMonth', y='Messages', 
                        color_discrete_sequence=['#1f77b4'])
    fig_trend.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a1d24'), xaxis_title='', yaxis_title=''
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ============================================
# TAB 2: ACTIVITY
# ============================================
with tab2:
    st.markdown("## ⏰ Activity Patterns")
    
    st.markdown("### 🔥 Activity Heatmap")
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    heatmap_data = df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='Count')
    pivot = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values='Count').fillna(0)
    pivot = pivot.reindex(days_order)
    
    # Peak detection
    peak_val = int(heatmap_data['Count'].max())
    peak_row = heatmap_data[heatmap_data['Count'] == peak_val].iloc[0]
    st.info(f"**Peak Engagement:** Most active on **{peak_row['DayOfWeek']}s** around **{peak_row['Hour']}:00** ({peak_val} messages).")

    fig_heat = px.imshow(pivot, 
                         labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
                         color_continuous_scale='Viridis',
                         aspect='auto',
                         text_auto=False) # Keep it clean
    fig_heat.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a1d24'), 
        height=350, 
        xaxis_title='Hour of Day (0-23)', 
        yaxis_title='',
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⏱️ Messages by Hour")
        hourly = df.groupby('Hour').size().reset_index(name='Messages')
        fig_hour = px.bar(hourly, x='Hour', y='Messages', color='Messages',
                          color_continuous_scale='Viridis')
        fig_hour.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1a1d24'), showlegend=False, coloraxis_showscale=False,
            xaxis_title='', yaxis_title=''
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 Messages by Day")
        daily = df.groupby('DayOfWeek').size().reindex(days_order).reset_index()
        daily.columns = ['Day', 'Messages']
        fig_day = px.bar(daily, x='Day', y='Messages', color='Messages',
                         color_continuous_scale='Viridis')
        fig_day.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1a1d24'), showlegend=False, coloraxis_showscale=False,
            xaxis_title='', yaxis_title=''
        )
        st.plotly_chart(fig_day, use_container_width=True)

# ============================================
# TAB 3: AUTHORS
# ============================================
with tab3:
    st.markdown("## 👥 User Analysis")
    
    # --- Response Time Analysis ---
    st.markdown("### ⏱️ Response Time Analysis")
    df_resp = df.sort_values('DateTime').reset_index(drop=True)
    df_resp['PrevAuthor'] = df_resp['Author'].shift(1)
    df_resp['PrevTime'] = df_resp['DateTime'].shift(1)
    
    # A reply is from a different author within 1 hour
    mask = (df_resp['Author'] != df_resp['PrevAuthor']) & (df_resp['PrevAuthor'].notna())
    df_resp = df_resp[mask].copy()
    df_resp['ResponseTimeMinutes'] = (df_resp['DateTime'] - df_resp['PrevTime']).dt.total_seconds() / 60
    
    # Filter out responses longer than 1 hour to avoid skewing (e.g. overnight)
    df_resp = df_resp[df_resp['ResponseTimeMinutes'] <= 60]
    
    if not df_resp.empty:
        resp_stats = df_resp.groupby('Author')['ResponseTimeMinutes'].agg(['mean']).reset_index()
        resp_stats.columns = ['Author', 'Mean Response (min)']
        resp_stats = resp_stats.sort_values('Mean Response (min)')
        
        # Display as a bar chart for fast comparison
        fig_resp = px.bar(resp_stats, x='Mean Response (min)', y='Author', orientation='h',
                          title="Mean Response Time (Lower is Faster)",
                          color='Mean Response (min)', color_continuous_scale='Viridis_r')
        fig_resp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1a1d24'), coloraxis_showscale=False,
            yaxis=dict(categoryorder='total descending'), xaxis_title='Minutes', yaxis_title=''
        )
        st.plotly_chart(fig_resp, use_container_width=True)
        
        # Table with mean stats
        st.dataframe(resp_stats.style.format({'Mean Response (min)': '{:.2f}'}), 
                     use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data to calculate response times.")

    st.markdown("---")
    
    st.markdown("### 🔗 Top Interactions (Most Frequent Replies)")
    interactions = defaultdict(int)
    # Using the same logic as before but with the already calculated df_resp if possible, 
    # but let's stick to the threshold logic for consistency
    df_sort = df.sort_values('DateTime').reset_index(drop=True)
    df_sort['Diff'] = df_sort['DateTime'].diff().dt.total_seconds()
    REPLY_THRESHOLD = 120
    for i in range(1, len(df_sort)):
        if df_sort.loc[i, 'Author'] != df_sort.loc[i-1, 'Author'] and df_sort.loc[i, 'Diff'] <= REPLY_THRESHOLD:
            interactions[f"{df_sort.loc[i, 'Author']} ➔ {df_sort.loc[i-1, 'Author']}"] += 1
    
    int_df = pd.DataFrame(interactions.items(), columns=['Interaction', 'Count']).sort_values('Count', ascending=False).head(10)
    
    if not int_df.empty:
        fig_int = px.bar(int_df, x='Count', y='Interaction', orientation='h',
                         color='Count', color_continuous_scale='Viridis')
        fig_int.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1a1d24'), showlegend=False, coloraxis_showscale=False,
            yaxis=dict(categoryorder='total ascending'), xaxis_title='', yaxis_title=''
        )
        st.plotly_chart(fig_int, use_container_width=True)
    else:
        st.info("Not enough interaction data found.")
    
    st.markdown("### 🎤 Conversation Starters")
    SILENCE_THRESHOLD = 7200
    starters = df_sort[(df_sort['Diff'] > SILENCE_THRESHOLD) | (df_sort['Diff'].isna())]['Author'].value_counts().reset_index()
    starters.columns = ['Author', 'Count']
    
    fig_starters = px.bar(starters, x='Count', y='Author', orientation='h',
                          color='Count', color_continuous_scale='Viridis')
    fig_starters.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a1d24'), coloraxis_showscale=False,
        yaxis=dict(categoryorder='total ascending'), xaxis_title='', yaxis_title=''
    )
    st.plotly_chart(fig_starters, use_container_width=True)
