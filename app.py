import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="Steam Recommendation Engine", layout="wide")

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
    
    /* Steam-like gradient for primary buttons */
    .stButton > button {
        background: linear-gradient( to right, #47bfff 5%, #1a44c2 60%);
        border: none;
        color: white;
        border-radius: 2px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient( to right, #47bfff 5%, #1a44c2 60%);
        filter: brightness(1.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("STEAM GAME RECOMMENDATION ENGINE")
st.markdown("Discover your next favorite game. This engine uses Machine Learning to find games similar to the ones you already love based on features like price, playtime, player base, and genres.")

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
st.sidebar.header("Find Recommendations")

# Game selection
game_list = sorted(df['name'].dropna().unique().tolist())
selected_game = st.sidebar.selectbox("Select a game you like:", game_list)

# Algorithm selection
algorithm = st.sidebar.radio(
    "Choose Recommendation Algorithm:",
    ("K-Means Clustering", "Self-Organizing Maps (SOM)")
)

cluster_col = 'KMeans_Cluster' if algorithm == "K-Means Clustering" else 'SOM_Cluster'

if st.sidebar.button("Get Recommendations!", type="primary"):
    # Find the cluster of the selected game
    game_info = df[df['name'] == selected_game].iloc[0]
    game_cluster = game_info[cluster_col]
    
    # Define descriptive names for clusters based on our analysis
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
    
    st.subheader(f"Because you liked **{selected_game}**...")
    st.markdown(f"*Algorithm used: {algorithm} ({cluster_name})*")
    
    # Filter games in the same cluster
    recommendations = df[(df[cluster_col] == game_cluster) & (df['name'] != selected_game)].copy()
    
    # Sort recommendations by quality/popularity
    # We'll use positive_review_ratio and total_reviews to find the best games in the cluster
    recommendations = recommendations.sort_values(by=['positive_review_ratio', 'total_reviews'], ascending=[False, False])
    
    # Get top 10
    top_recs = recommendations.head(10)
    
    if len(top_recs) == 0:
        st.info("No other games found in this specific cluster.")
    else:
        # Display the results nicely
        
        # Add primary genre if available (just format it nicely)
        def get_genres(row):
            genres = [col.replace('genre_', '') for col in df.columns if col.startswith('genre_') and row[col] == 1]
            return ", ".join(genres)
            
        top_recs['Genres'] = top_recs.apply(get_genres, axis=1)
        
        # Format the display dataframe
        display_df = top_recs[['name', 'Genres', 'price_usd', 'positive_review_ratio', 'avg_playtime_forever', 'release_year']].copy()
        display_df.columns = ['Game Name', 'Genres', 'Price ($)', 'Positive Review %', 'Avg Playtime (hrs)', 'Release Year']
        
        # Convert ratio to percentage
        display_df['Positive Review %'] = (display_df['Positive Review %'] * 100).round(1).astype(str) + '%'
        display_df['Avg Playtime (hrs)'] = (display_df['Avg Playtime (hrs)'] / 60).round(1) # Assuming playtime is in minutes
        display_df['Release Year'] = display_df['Release Year'].fillna(0).astype(int).astype(str).replace('0', 'Unknown')
        
        # Show top 3 in cards
        st.markdown("### Top 3 Matches")
        cols = st.columns(min(3, len(top_recs)))
        for i, col in enumerate(cols):
            game = top_recs.iloc[i]
            with col:
                st.markdown(f"#### {game['name']}")
                st.markdown(f"**Genres:** {game['Genres'] if game['Genres'] else 'Mixed'}")
                st.markdown(f"**Price:** ${game['price_usd']:.2f}")
                st.markdown(f"**Rating:** {game['positive_review_ratio']*100:.1f}% Positive")
                st.markdown(f"**Players:** ~{int(game['avg_estimated_owners']):,}")
        
        st.markdown("### Top 10 Recommendations")
        # Show as a dataframe
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# Sequels / Related Games Button
if st.sidebar.button("Find Sequels & Related", type="secondary"):
    st.subheader(f"Games related to **{selected_game}**")
    st.markdown("*Searching by title similarity...*")
    
    # Extract the first significant word to avoid apostrophe/encoding mismatches
    import re
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
        
        # Find matches ignoring case
        sequels = df[df['name'].str.lower().str.contains(search_term, na=False) & (df['name'] != selected_game)].copy()
        
        if len(sequels) == 0:
            st.info("No related games or sequels found in the dataset.")
        else:
            st.success(f"Found {len(sequels)} related games!")
            # Sort by total reviews to show the most popular ones first
            sequels = sequels.sort_values(by='total_reviews', ascending=False)
            
            # Formatting similar to the recommendations
            def get_genres_seq(row):
                genres = [col.replace('genre_', '') for col in df.columns if col.startswith('genre_') and row[col] == 1]
                return ", ".join(genres)
                
            sequels['Genres'] = sequels.apply(get_genres_seq, axis=1)
            display_df = sequels[['name', 'Genres', 'price_usd', 'positive_review_ratio', 'release_year']].copy()
            display_df.columns = ['Game Name', 'Genres', 'Price ($)', 'Positive Review %', 'Release Year']
            display_df['Positive Review %'] = (display_df['Positive Review %'] * 100).round(1).astype(str) + '%'
            display_df['Release Year'] = display_df['Release Year'].fillna(0).astype(int).astype(str).replace('0', 'Unknown')
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
