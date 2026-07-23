import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Spotify AI Recommendation",
    page_icon="🎧",
    layout="wide"
)


# =========================
# Spotify Style CSS
# =========================

st.markdown(
    """
    <style>

    .stApp {
        background-color:#121212;
        color:white;
    }


    h1 {
        color:#1DB954;
        font-size:48px;
        font-weight:900;
    }


    .subtitle {
        color:#b3b3b3;
        font-size:18px;
    }


    .song-card {

        background:#181818;

        padding:20px;

        border-radius:15px;

        margin-bottom:15px;

        border:1px solid #282828;

    }


    .song-title {

        color:#1DB954;

        font-size:22px;

        font-weight:bold;

    }


    .artist {

        color:white;

        font-size:18px;

    }


    .info {

        color:#b3b3b3;

        line-height:1.8;

    }


    .stButton button {

        background:#1DB954;

        color:white;

        border-radius:30px;

        font-weight:bold;

    }


    </style>
    """,
    unsafe_allow_html=True
)



# =========================
# Load Data
# =========================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "notebooks/spotify_songs_clean.csv"
    )


    pop_scaler = MinMaxScaler()


    df["pop_score"] = pop_scaler.fit_transform(
        df[["popularity"]]
    )


    return df



@st.cache_resource
def load_model():

    model = joblib.load(
        "notebooks/spotify_knn_model.pkl"
    )


    scaler = joblib.load(
        "notebooks/spotify_scaler.pkl"
    )


    return model, scaler



df = load_data()

model, scaler = load_model()



# =========================
# Dashboard Sidebar
# =========================

st.sidebar.title(
    "📊 Spotify Dataset"
)


st.sidebar.metric(
    "🎵 Total Songs",
    f"{len(df):,}"
)


st.sidebar.metric(
    "👤 Artists",
    f"{df['artists'].nunique():,}"
)


st.sidebar.metric(
    "📅 Year Range",
    f"{df['release_year'].min()} - {df['release_year'].max()}"
)


st.sidebar.metric(
    "⭐ Average Popularity",
    round(df["popularity"].mean(),2)
)



# =========================
# Feature
# =========================

features = [

    'danceability',
    'energy',
    'loudness',
    'acousticness',
    'instrumentalness',
    'valence',
    'tempo',
    'popularity'

]


X = scaler.transform(
    df[features]
)



# =========================
# Header
# =========================

st.markdown(
    """
    <h1>
    🎧 Spotify AI Recommendation
    </h1>

    <div class="subtitle">

    AI Music Recommendation System
    <br>
    Powered by KNN + Hybrid Ranking

    </div>

    """,
    unsafe_allow_html=True
)



# =========================
# Dashboard Charts
# =========================

st.markdown(
    "## 📈 Music Analytics"
)


col1,col2 = st.columns(2)



with col1:


    year_data = (

        df.groupby(
            "release_year"
        )
        ["popularity"]
        .mean()
        .reset_index()

    )


    fig = px.line(

        year_data,

        x="release_year",

        y="popularity",

        title="Average Popularity By Year"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



with col2:


    top_artist = (

        df.groupby(
            "artists"
        )
        ["popularity"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()

    )


    fig2 = px.bar(

        top_artist,

        x="popularity",

        y="artists",

        orientation="h",

        title="Top Artists"

    )


    st.plotly_chart(

        fig2,

        use_container_width=True

    )



# =========================
# Search Song
# =========================


st.markdown(
    "## 🔍 Find Similar Songs"
)


song_name = st.text_input(

    "Search Song",

    "Blinding Lights"

)



songs = df[

    df["name"]

    .str.contains(

        song_name,

        case=False,

        na=False

    )

]



if len(songs)==0:

    st.warning(
        "Song not found"
    )

    st.stop()



selected_song = st.selectbox(

    "🎵 Choose Song",

    songs["name"].unique()

)



artists = df[

    df["name"] == selected_song

]["artists"].unique()



selected_artist = st.selectbox(

    "👤 Choose Artist",

    artists

)



# =========================
# Recommendation
# =========================

if st.button(
    "🎵 Generate Recommendation"
):


    with st.spinner(
        "AI searching..."
    ):


        target = df[

            (df["name"]==selected_song)

            &

            (df["artists"]==selected_artist)

        ]


        index = target.index[0]



        distance, indices = model.kneighbors(

            X[index].reshape(1,-1),

            n_neighbors=50

        )



        result = df.iloc[
            indices[0]
        ].copy()



        result["similarity"] = (

            1-distance[0]

        )



        result["hybrid_score"] = (

            result["similarity"]*0.7

            +

            (result["artists"]==selected_artist)
            .astype(int)*0.2

            +

            result["pop_score"]*0.1

        )



        result = (

            result

            .sort_values(
                "hybrid_score",
                ascending=False
            )

            .query(
                "name != @selected_song"
            )

            .head(5)

        )



    st.markdown(
        "## 🔥 Recommended Songs"
    )



    for _,row in result.iterrows():


        st.markdown(

            f"""

            <div class="song-card">

            <div class="song-title">
            🎵 {row['name']}
            </div>


            <div class="artist">
            👤 {row['artists']}
            </div>


            <div class="info">

            📅 Release:
            {row['release_year']}

            <br>

            ⭐ Popularity:
            {row['popularity']}

            <br>

            🔥 Similarity:
            {row['similarity']:.2f}

            </div>

            </div>

            """,

            unsafe_allow_html=True

        )