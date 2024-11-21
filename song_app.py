import streamlit as st
from IPython.display import IFrame
import random
import pandas as pd
import pickle
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config import client_secret, client_id

# --- Caching data loading ---
@st.cache_data
def load_data():
    """
    Load and cache the hot_tracks and track_database data.
    """
    hot_tracks = pd.read_csv("scraped_hot_tracks.csv")
    track_database = pd.read_csv("spotify_database.csv")
    return hot_tracks, track_database

# --- Spotify Initialization ---
def initialize_spotify():
    global sp
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                               client_secret=client_secret))

# --- Audio Features Fetching ---
def af_user_input(user_song, user_artist):
    result = sp.search(q=f'{user_song} {user_artist}')
    if len(result["tracks"]["items"]) == 0:
        st.error("Sorry, we couldn't find that song. Please try again.")
        st.stop()
    user_id = result["tracks"]["items"][0]["id"]
    user_af = sp.audio_features(user_id)
    return pd.DataFrame(user_af)

# --- Model and Scaler Loading ---
def load_model(filename="kmeans_26.pickle"): 
    try: 
        with open(filename, "rb") as f: 
            return pickle.load(f) 
    except FileNotFoundError: 
        st.error("Model file not found!")
        st.stop()

def load_scaler(filename="scaler_alt.pickle"): 
    try: 
        with open(filename, "rb") as f: 
            return pickle.load(f) 
    except FileNotFoundError: 
        st.error("Scaler file not found!")
        st.stop()

# --- Recommender Function ---
def song_playlist_recommender(user_af, user_song, hot_tracks, database):
    if user_song.lower() in [song.lower() for song in list(hot_tracks["song"])]:
        rec_track = random.choice(list(hot_tracks[hot_tracks["song"] != user_song]["id"]))
        rec_playlist = [rec_track]
    else:
        model = load_model(filename="kmeans_26.pickle")
        scaler = load_scaler(filename="scaler_alt.pickle")
        user_af_norm = scaler.transform(user_af.drop(columns=[
            "type", "id", "uri", "track_href", "analysis_url", 
            "duration_ms", "time_signature", "liveness", "loudness", "speechiness"
        ]))
        rec_track_cluster = model.predict(user_af_norm)[0]
        rec_playlist = random.sample(
            list(database[(database["id"] != user_af["id"][0]) & 
                          (database["cluster"] == rec_track_cluster)]["id"]), 
            15
        )
    return rec_playlist

# --- Spotify Track Embed ---
def play_song(track_id):
    track_url = f"https://open.spotify.com/embed/track/{track_id}"
    st.components.v1.iframe(src=track_url, width=400, height=230)

# --- Main App ---
def main():
    st.title("🎵 Song Recommender App")
    st.write("Enter your favorite song and artist to receive a playlist of recommendations!")

    # Load data
    hot_tracks, track_database = load_data()

    # Initialize Spotify connection
    initialize_spotify()

    # User input
    user_song = st.text_input("Enter the song name:")
    user_artist = st.text_input("Enter the artist name:")

    if st.button("Get Recommendations"):
        if not user_song or not user_artist:
            st.error("Please fill out both the song and artist fields.")
            st.stop()

        # Get user audio features
        user_af = af_user_input(user_song=user_song, user_artist=user_artist)

        # Get recommendations
        rec_playlist = song_playlist_recommender(
            user_af=user_af, user_song=user_song, 
            hot_tracks=hot_tracks, database=track_database
        )

        # Display results
        st.write("### You chose this song:")
        play_song(user_af["id"][0])

        st.write("### You might also like:")
        play_song(rec_playlist[0])

        st.write("### Here's a complete playlist for you to check out:")
        for song_id in rec_playlist[1:]:
            play_song(song_id)

# Run the app
if __name__ == "__main__":
    main()
