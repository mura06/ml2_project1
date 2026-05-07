import streamlit as st
import pandas as pd
import re

# Set page config
st.set_page_config(page_title="Steam Recommendation Engine", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Steam aesthetics
st.markdown("""
<style>
    /* Global Background and Text */
    .stApp {
        background-color: #1b2838;
        color: #c7d5e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: "Motiva Sans", Sans-serif;
        font-weight: 300;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Style the metrics/cards headers */
    div[data-testid="stMarkdownContainer"] h4 {
        color: #66c0f4;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #171a21;
    }
    
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #8f98a0;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff;
        background-color: #2a475e;
        border-bottom: 2px solid #66c0f4;
    }
    
    /* Hide top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Area
col1, col2 = st.columns([1, 5])
with col1:
    # A simple placeholder for a logo
    st.markdown("""
    <div style="background: linear-gradient(135deg, #171a21 0%, #1b2838 100%); border-radius: 50%; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; border: 2px solid #66c0f4; margin-top: 10px;">
        <span style="font-size: 40px;">🎮</span>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.title("STEAM DISCOVERY QUEUE")
    st.markdown("<p style='color: #8f98a0; font-size: 1.1rem;'>Your personalized ML-powered recommendation engine based on playtime, reviews, and genres.</p>", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #2a475e; margin-top: 0;'>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("clustered_games.csv")
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Could not find 'clustered_games.csv'. Please run `python train_model.py` first to generate the clustered dataset.")
    st.stop()

# Sidebar for inputs
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    game_list = sorted(df['name'].dropna().unique().tolist())
    selected_game = st.selectbox("Select a base game:", game_list, index=None, placeholder="Search for a game...", help="We'll find games similar to this one.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    algorithm = st.radio(
        "Clustering Algorithm:",
        ("K-Means Clustering", "Self-Organizing Maps (SOM)"),
        help="Choose the underlying model used to cluster games."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='background-color: rgba(0,0,0,0.2); padding: 15px; border-radius: 5px;'><span style='color: #66c0f4; font-weight: bold;'>Stats</span><br>Total Games: {}<br>Algorithm Ready</div>".format(len(df)), unsafe_allow_html=True)

# Main logic
if not selected_game:
    st.markdown("""
    <div style="background-color: #171a21; padding: 30px; border-radius: 8px; text-align: center; border: 1px dashed #2a475e; margin-top: 20px;">
        <h3 style="color: #8f98a0; font-size: 24px;">Welcome to the Discovery Queue</h3>
        <p style="color: #66c0f4; font-size: 16px;">👈 Select a base game from the sidebar to generate personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

cluster_col = 'KMeans_Cluster' if algorithm == "K-Means Clustering" else 'SOM_Cluster'
game_info = df[df['name'] == selected_game].iloc[0]
game_cluster = game_info[cluster_col]

kmeans_names = {
    0: "Premium & Mainstream Hits",
    1: "The Megahit Anomaly (CS2)",
    2: "Racing & Sports",
    3: "Free-to-Play & Budget Indies",
    4: "Utilities & Software",
    5: "Niche Micro-Indies"
}

som_names = {
    0: "Premium RPGs & Strategy",
    1: "Popular Free-to-Play",
    2: "Sports & Competitive",
    3: "Utilities & Software",
    4: "Casual & Budget Indies",
    5: "Mainstream Classics",
    6: "Unpopular/Niche Titles",
    7: "Racing & Driving",
    8: "Simulation & Builders"
}

cluster_name = kmeans_names.get(game_cluster, f"Cluster {game_cluster}") if algorithm == "K-Means Clustering" else som_names.get(game_cluster, f"Cluster {game_cluster}")

# Helper functions
def get_genres(row):
    genres = [col.replace('genre_', '') for col in df.columns if col.startswith('genre_') and row[col] == 1]
    return ", ".join(genres)

# Layout: Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Top Matches", "📋 Full Cluster Results", "🔗 Sequels & Franchise"])

# --- TAB 1: Top Matches ---
with tab1:
    st.markdown(f"#### Because you play **{selected_game}**")
    st.markdown(f"<span style='color: #8f98a0;'>Algorithm placed this in: </span><span style='color: #66c0f4; font-weight: 500;'>{cluster_name}</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    recommendations = df[(df[cluster_col] == game_cluster) & (df['name'] != selected_game)].copy()
    recommendations = recommendations.sort_values(by=['positive_review_ratio', 'total_reviews'], ascending=[False, False])
    top_recs = recommendations.head(9) # Give 9 for a 3x3 grid
    
    if len(top_recs) == 0:
        st.info("No other games found in this specific cluster.")
    else:
        top_recs['Genres'] = top_recs.apply(get_genres, axis=1)
        
        # Display as a grid of cards
        cols = st.columns(3)
        for i, (_, game) in enumerate(top_recs.iterrows()):
            col = cols[i % 3]
            with col:
                st.markdown(f'''
                <div style="background-color: #171a21; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.4); border-left: 4px solid #66c0f4; margin-bottom: 20px; height: 160px; display: flex; flex-direction: column; justify-content: space-between; transition: transform 0.2s;">
                    <div>
                        <h4 style="color: #ffffff; margin-top: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{game['name']}">{game['name']}</h4>
                        <p style="margin-bottom: 3px; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><strong>Genres:</strong> <span style="color: #8f98a0;">{game['Genres'] if game['Genres'] else 'Mixed'}</span></p>
                        <p style="margin-bottom: 3px; font-size: 13px;"><strong>Price:</strong> <span style="color: #a3cc27;">${game['price_usd']:.2f}</span></p>
                        <p style="margin-bottom: 3px; font-size: 13px;"><strong>Rating:</strong> <span style="color: #66c0f4;">{game['positive_review_ratio']*100:.1f}% Positive</span></p>
                    </div>
                    <div>
                        <p style="margin-bottom: 0px; font-size: 13px;"><strong>Players:</strong> <span style="color: #8f98a0;">~{int(game['avg_estimated_owners']):,}</span></p>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

# --- TAB 2: Full Recommendations ---
with tab2:
    st.markdown(f"#### Complete Cluster View: {cluster_name}")
    st.markdown(f"Explore all games that share similar traits within the **{cluster_name}** cluster.")
    
    if len(recommendations) == 0:
        st.info("No other games found in this specific cluster.")
    else:
        display_df = recommendations[['name', 'price_usd', 'positive_review_ratio', 'avg_playtime_forever', 'release_year']].copy()
        display_df['Genres'] = recommendations.apply(get_genres, axis=1)
        display_df = display_df[['name', 'Genres', 'price_usd', 'positive_review_ratio', 'avg_playtime_forever', 'release_year']]
        
        display_df.columns = ['Game Name', 'Genres', 'Price ($)', 'Positive Review %', 'Avg Playtime (hrs)', 'Release Year']
        
        display_df['Positive Review %'] = (display_df['Positive Review %'] * 100).round(1).astype(str) + '%'
        display_df['Avg Playtime (hrs)'] = (display_df['Avg Playtime (hrs)'] / 60).round(1)
        display_df['Release Year'] = display_df['Release Year'].fillna(0).astype(int).astype(str).replace('0', 'Unknown')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

# --- TAB 3: Sequels ---
with tab3:
    st.markdown("#### Franchise & Related Titles")
    st.markdown("Games sharing similar titles or part of the same franchise.")
    
    alphanumeric_words = re.findall(r'[a-zA-Z0-9]+', str(selected_game))
    stop_words = {"the", "a", "an", "of", "and", "in", "to"}
    
    search_term = ""
    for w in alphanumeric_words:
        if w.lower() not in stop_words and len(w) > 2:
            search_term = w.lower()
            break
            
    if not search_term and len(alphanumeric_words) > 0:
        search_term = alphanumeric_words[0].lower()
        
    if search_term:
        sequels = df[df['name'].str.lower().str.contains(search_term, na=False) & (df['name'] != selected_game)].copy()
        
        if len(sequels) == 0:
            st.info(f"No related games found for '{search_term}'.")
        else:
            st.success(f"Found {len(sequels)} related games matching '{search_term}'!")
            sequels = sequels.sort_values(by='total_reviews', ascending=False)
            
            sequels['Genres'] = sequels.apply(get_genres, axis=1)
            display_seq_df = sequels[['name', 'Genres', 'price_usd', 'positive_review_ratio', 'release_year']].copy()
            display_seq_df.columns = ['Game Name', 'Genres', 'Price ($)', 'Positive Review %', 'Release Year']
            display_seq_df['Positive Review %'] = (display_seq_df['Positive Review %'] * 100).round(1).astype(str) + '%'
            display_seq_df['Release Year'] = display_seq_df['Release Year'].fillna(0).astype(int).astype(str).replace('0', 'Unknown')
            
            st.dataframe(display_seq_df, use_container_width=True, hide_index=True)
