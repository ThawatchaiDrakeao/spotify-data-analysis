# 🎧 Spotify AI Recommendation System

An AI-powered music recommendation system using Machine Learning.

This project analyzes Spotify music data and recommends similar songs using **K-Nearest Neighbors (KNN) + Hybrid Ranking Algorithm**.

---

# 🚀 Project Overview

Spotify AI Recommendation System is a machine learning project that recommends songs based on audio features and user-selected tracks.

The system analyzes:

- Song characteristics
- Audio features
- Popularity trends
- Artist similarity

Then generates personalized song recommendations.

---

# ✨ Features

✅ Search songs  
✅ AI-based song recommendation  
✅ KNN similarity search  
✅ Hybrid ranking algorithm  
✅ Spotify-inspired dark UI  
✅ Music analytics dashboard  
✅ Popularity trend visualization  
✅ Top artists analysis  

---

# 📊 Dataset

Dataset contains:

- 🎵 586,601 songs
- 👤 Thousands of artists
- 📅 Music records from 1900 - 2021


Main audio features:

```
danceability
energy
loudness
acousticness
instrumentalness
valence
tempo
popularity
```

---

# 🧠 Machine Learning Workflow


```
Spotify Dataset

        ↓

Data Cleaning

        ↓

Exploratory Data Analysis

        ↓

Feature Engineering

        ↓

Feature Scaling

        ↓

KNN Model Training

        ↓

Hybrid Ranking

        ↓

Streamlit Recommendation App
```

---

# 🔎 Exploratory Data Analysis (EDA)

Analyzed:

- Music popularity by year
- Artist popularity
- Audio feature distribution
- Relationship between audio features and popularity


Key findings:

- Modern songs have higher average popularity
- Energy and loudness show positive correlation with popularity
- Acousticness tends to decrease as popularity increases

---

# 🤖 Recommendation Model


## K-Nearest Neighbors (KNN)

The model finds songs with similar audio characteristics using distance-based similarity.


Features used:

```
danceability
energy
loudness
acousticness
instrumentalness
valence
tempo
popularity
```

---

# 🔥 Hybrid Ranking System


The recommendation ranking combines:


### 1. Similarity Score

Weight:

```
70%
```


Measures how similar songs are based on audio features.


---

### 2. Artist Preference

Weight:

```
20%
```


Boost songs from the same artist.


---

### 3. Popularity Score

Weight:

```
10%
```


Considers overall song popularity.


---

## Formula


```
Hybrid Score =

(Similarity × 0.7)

+

(Artist Match × 0.2)

+

(Popularity Score × 0.1)
```

---

# 🖥️ Application Architecture


```
User

 ↓

Streamlit Web App

 ↓

Recommendation Engine

 ↓

KNN Model

 ↓

Similar Songs

```

---

# 🛠️ Tech Stack


## Programming Language

- Python


## Data Analysis

- Pandas
- NumPy


## Machine Learning

- Scikit-learn
- KNN Algorithm
- Feature Scaling


## Visualization

- Plotly


## Application

- Streamlit


## Model Storage

- Joblib

---

# 📂 Project Structure


```
spotify-data-analysis/

│
├── app.py
│
├── notebooks/
│   │
│   ├── analysis.ipynb
│   ├── spotify_knn_model.pkl
│   ├── spotify_scaler.pkl
│   └── spotify_songs_clean.csv
│
├── README.md
│
└── requirements.txt
```

---

# ▶️ Installation


Clone repository:


```bash
git clone https://github.com/ThawatchaiDrakeao/spotify-data-analysis.git
```


Install dependencies:


```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application


Start Streamlit:


```bash
streamlit run app.py
```

Application will open:

```
http://localhost:8501
```

---

# 🎵 Example Recommendation


Input:

```
Song:
Blinding Lights

Artist:
The Weeknd
```


Recommended:


```
UN DIA - J Balvin

Waves - Dean Lewis

Mercy - Shawn Mendes

Ghost - Justin Bieber

Cover Me In Sunshine - P!nk
```

---

# 📸 Application Preview


Spotify-inspired AI recommendation dashboard.


Features:

- Dark Spotify UI
- Recommendation cards
- Analytics dashboard
- Similarity score display


---

# 🎯 Future Improvements


- Spotify API integration
- Album cover images
- User login system
- Deep Learning recommendation model
- Deploy on cloud platform


---

# 👨‍💻 Author


**Thawatchai Dakaew**

Full Stack Developer | AI & Data Science Enthusiast


GitHub:

https://github.com/ThawatchaiDrakeao

Portfolio:

https://fengpixel-worldportfolio-project.vercel.app/
