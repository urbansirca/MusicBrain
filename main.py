import spotipy
from dotenv import load_dotenv, dotenv_values
from neurosity import NeurositySDK
from spotipy.oauth2 import SpotifyOAuth
import requests
import sqlite3
from datetime import datetime
import os
from neurosity_class import NeurosityVectorizer
from spotify_class import SpotifyAPI
import numpy as np
import time




load_dotenv("environment.env")




# get the secret from the environment
SPOTIFY_SECRET = os.getenv("SPOTIFY_SECRET")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")

email = os.environ.get('NEUROSITY_EMAIL')
password = os.environ.get("NEUROSITY_PASSWORD")
device_id = os.environ.get("NEUROSITY_DEVICE_ID")


neurosity = NeurositySDK({
    "device_id": device_id})
neurosity.login({
    "email": email,
    "password": password})



DB = "music.db"


spotify_api = SpotifyAPI(SPOTIFY_CLIENT_ID, SPOTIFY_SECRET)
neurosity_vectorizer = NeurosityVectorizer(neurosity)

# clear database
spotify_api.clear_database(DB)

# if database doesnt exist, create it
if not os.path.exists(DB):
    spotify_api.initialize_database(DB)



def monitor_song_and_collect_data(spotify_api, neurosity_vectorizer, db_name="music_focus.db"):
    """
    Monitors the currently playing song on Spotify, collects EEG data while the song is playing,
    and saves the collected data to the database when the song ends or is skipped.

    Args:
        spotify_api (SpotifyAPI): Instance of the SpotifyAPI class.
        neurosity_vectorizer (NeurosityVectorizer): Instance of the NeurosityVectorizer class.
        db_name (str): Name of the SQLite database.
    """
    first_song = spotify_api.get_current_song_metrics()


    print(f"Now playing: {first_song['track_name']} by {', '.join(first_song['artists'])}")
    current_song_id = first_song["id"]
    eeg_data = []
    stop = False
    added = False

    while not stop:
        # Get the currently playing song
        neurosity_vectorizer.gather_eeg_samples_during_song()  # Start gathering EEG samples

        current_song = spotify_api.get_current_song_metrics() # Get the currently playing song
        if not added:
            spotify_api.add_song_to_database(current_song, db_name)
            added = True

        # If the song has changed, finalize the previous song's data
        if current_song_id != current_song["id"]:
            eeg_dict = neurosity_vectorizer.get_current_song_eeg_data()
            neurosity_vectorizer.reset_current_song_eeg_data()


            if current_song_id and neurosity_vectorizer.status=="online" and not neurosity_vectorizer.charging:
                # Add the EEG data to the database for the previous song
                spotify_api.add_eeg_metrics(current_song_id, eeg_dict, db_name)
            elif neurosity_vectorizer.status=="offline":
                print("EEG data not added - Device is offline")
            elif not current_song_id:
                print("EEG data not added - No song playing")
            elif neurosity_vectorizer.charging:
                print("EEG data not added - Device is charging")


            # Update the current song ID
            old_song_id = current_song_id
            current_song_id = current_song["id"]
            added = False # Reset the added flag
            stop = True

            print(f"Now playing: {current_song['track_name']} by {', '.join(current_song['artists'])}")

        time.sleep(1/4)
    return old_song_id





old_song_id = monitor_song_and_collect_data(spotify_api, neurosity_vectorizer, DB)

eeg_data = spotify_api.get_eeg_data_from_DB(old_song_id, DB)

print("Data collection complete")
