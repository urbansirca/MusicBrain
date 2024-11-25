import json
import sqlite3
from os import path, mkdir, listdir
from time import sleep, time
import requests

import numpy as np

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
from datetime import datetime
import os
import pandas as pd

class SpotifyAPI:

    def __init__(self, client_id, client_secret):
        self.sp_oauth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="https://open.spotify.com/",
            scope="user-read-playback-state user-read-currently-playing"
        )
        self.API = spotipy.Spotify(auth_manager=self.sp_oauth)

        self.user_id = self.get_current_user()["id"]
        self.user_name = self.get_current_user()["display_name"]
    def get_current_user(self):
        """
        Retrieves the current user's information from Spotify.

        Returns:
            dict: A dictionary containing the current user's information.
        """
        try:
            user_info = self.API.current_user()
            # print("user info: ", user_info)
            return user_info
        except Exception as e:
            print(f"An error occurred while retrieving the current user: {e}")
            return None

    def get_audio_features(self, track_id):
        # Retrieve audio features for the given track
        features = self.API.audio_features(track_id)[0]


        if features:
            audio_features = {
                "acousticness": features.get("acousticness"),
                "danceability": features.get("danceability"),
                "duration_ms": features.get("duration_ms"),
                "energy": features.get("energy"),
                "id": features.get("id"),
                "instrumentalness": features.get("instrumentalness"),
                "key": features.get("key"),
                "liveness": features.get("liveness"),
                "loudness": features.get("loudness"),
                "mode": features.get("mode"),
                "speechiness": features.get("speechiness"),
                "tempo": features.get("tempo"),
                "time_signature": features.get("time_signature"),
                "track_href": features.get("track_href"),
                "type": features.get("type"),
                "uri": features.get("uri"),
                "valence": features.get("valence")
            }
            return audio_features
        else:
            print("No audio features found for the given track ID.")
            return None

    def get_audio_analysis(self, track_id):
        # Refresh token if necessary
        token_info = self.sp_oauth.get_access_token()
        access_token = token_info['access_token']

        # Get audio features, including the analysis_url
        features = self.API.audio_features(track_id)[0]
        analysis_url = features.get("analysis_url")

        if analysis_url:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(analysis_url, headers=headers)

            if response.status_code == 200:
                audio_analysis = response.json()
                return audio_analysis
            else:
                print("Error accessing audio analysis:", response.status_code, response.text)
                return None
        else:
            print("Analysis URL not found for the track.")
            return None


    def delete_database(self, db_name="music_focus.db"):
        try:
            if os.path.exists(db_name):
                os.remove(db_name)
                print(f"Database '{db_name}' deleted successfully.")
            else:
                print(f"Database '{db_name}' does not exist.")
        except Exception as e:
            print(f"An error occurred while deleting the database: {e}")

    def clear_database(self, db_name="music_focus.db"):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Clear all data from the tables
        cursor.execute('''
            DELETE FROM song_metrics
        ''')
        cursor.execute('''
            DELETE FROM artists
        ''')
        cursor.execute('''
            DELETE FROM eeg_metrics
        ''')

        conn.commit()
        conn.close()
        print("Database cleared successfully.")

    def initialize_database(self, db_name="music_focus.db"):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create table for song metrics (without date and time fields)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS song_metrics (
                id TEXT PRIMARY KEY,
                track_name TEXT,
                acousticness REAL,
                danceability REAL,
                duration_ms INTEGER,
                energy REAL,
                instrumentalness REAL,
                key INTEGER,
                liveness REAL,
                loudness REAL,
                mode INTEGER,
                speechiness REAL,
                tempo REAL,
                time_signature INTEGER,
                valence REAL,
                uri TEXT,
                user_id TEXT,
                user_name TEXT
            )
        ''')

        # Create a table for artists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS artists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id TEXT,
                artist_name TEXT,
                FOREIGN KEY (song_id) REFERENCES song_metrics (id)
            )
        ''')

        # Create a table for EEG metrics with date and time fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS eeg_metrics (
                song_id TEXT,
                timestamp TIMESTAMP,
                psd BLOB,    -- Stores PSD as a vector of any length
                alpha BLOB,  -- Stores alpha as a vector of length 8
                beta BLOB,   -- Stores beta as a vector of length 8
                delta BLOB,  -- Stores delta as a vector of length 8
                gamma BLOB,  -- Stores gamma as a vector of length 8
                theta BLOB,  -- Stores theta as a vector of length 8
                focus_score REAL,   -- Stores focus score as a float
                calm_score REAL,    -- Stores calm score as a float
                user_id TEXT,
                user_name TEXT,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                weekday INTEGER,
                hour INTEGER,
                FOREIGN KEY (song_id) REFERENCES song_metrics (id)
            )
        ''')

        conn.commit()
        conn.close()
        print("Database initialized successfully with song, artist, and EEG tables.")

    def add_eeg_metrics(self, song_id, eeg_data, db_name="music_focus.db"):
        # Debug prints to confirm function is called with correct values

        # print("EEG Data:", eeg_data)

        conn = sqlite3.connect(db_name)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
        cursor = conn.cursor()

        # Current timestamp and date/time fields
        timestamp = datetime.now()
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        hour = timestamp.hour

        print(f"Inserting EEG data for song {song_id} with timestamp: {timestamp}")

        # Convert vectors to binary (BLOB format)
        psd_blob = sqlite3.Binary(eeg_data["psd"].tobytes())
        alpha_blob = sqlite3.Binary(eeg_data["alpha"].tobytes())
        beta_blob = sqlite3.Binary(np.array(eeg_data["beta"]).tobytes())
        delta_blob = sqlite3.Binary(np.array(eeg_data["delta"]).tobytes())
        gamma_blob = sqlite3.Binary(np.array(eeg_data["gamma"]).tobytes())
        theta_blob = sqlite3.Binary(np.array(eeg_data["theta"]).tobytes())

        cursor.execute('''
            INSERT INTO eeg_metrics (
                song_id, timestamp, psd, alpha, beta, delta, gamma, theta, 
                focus_score, calm_score, user_id, user_name, year, month, day, weekday, hour
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            song_id, timestamp, psd_blob, alpha_blob, beta_blob, delta_blob, gamma_blob, theta_blob,
            eeg_data["focus"], eeg_data["calm"], self.user_id, self.user_name, year, month, day, weekday, hour
        ))

        # Commit changes to ensure data is saved
        conn.commit()
        conn.close()
        print("EEG metrics successfully added to database.")


    def add_song_to_database(self, song_data, db_name="music_focus.db"):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Insert song data into song_metrics (without date and time fields)
        cursor.execute('''
            INSERT OR REPLACE INTO song_metrics (
                id, track_name, acousticness, danceability, duration_ms, energy,
                instrumentalness, key, liveness, loudness, mode, speechiness, tempo,
                time_signature, valence, uri, user_id, user_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            song_data['id'], song_data['track_name'], song_data['acousticness'],
            song_data['danceability'], song_data['duration_ms'], song_data['energy'],
            song_data['instrumentalness'], song_data['key'], song_data['liveness'],
            song_data['loudness'], song_data['mode'], song_data['speechiness'],
            song_data['tempo'], song_data['time_signature'], song_data['valence'],
            song_data['uri'], self.user_id, self.user_name,

        ))

        # Insert each artist into the artists table only if not already present for this song
        for artist in song_data['artists']:
            # Check if artist is already in the table for this song_id
            cursor.execute('''
                SELECT 1 FROM artists WHERE song_id = ? AND artist_name = ?
            ''', (song_data['id'], artist))
            if cursor.fetchone() is None:
                # Artist is not present, so insert it
                cursor.execute('''
                    INSERT INTO artists (song_id, artist_name) 
                    VALUES (?, ?)
                ''', (song_data['id'], artist))

        conn.commit()
        conn.close()
        print(f"Song '{song_data['track_name']}' with artists added to database.")

    # Example function to get current song metrics and add to database with artists
    def get_current_song_metrics(self):
        current_track = self.API.current_playback()
        if current_track and current_track['item']:
            track_id = current_track['item']['id']
            track_name = current_track['item']['name']
            artists = [artist['name'] for artist in current_track['item']['artists']]

            features = self.API.audio_features(track_id)[0]
            return {
                "id": features.get("id"),
                "track_name": track_name,
                "artists": artists,  # List of artist names
                "acousticness": features.get("acousticness"),
                "danceability": features.get("danceability"),
                "duration_ms": features.get("duration_ms"),
                "energy": features.get("energy"),
                "instrumentalness": features.get("instrumentalness"),
                "key": features.get("key"),
                "liveness": features.get("liveness"),
                "loudness": features.get("loudness"),
                "mode": features.get("mode"),
                "speechiness": features.get("speechiness"),
                "tempo": features.get("tempo"),
                "time_signature": features.get("time_signature"),
                "valence": features.get("valence"),
                "uri": features.get("uri")
            }
        else:
            print("No track currently playing.")
            return None

    def get_eeg_data_from_DB(self, song_id, db_name="music_focus.db"):
        """
        Fetch and process EEG data for a given song ID from the database.

        Args:
            song_id (str): The ID of the song to fetch EEG data for.
            db_name (str): Name of the SQLite database.

        Returns:
            pd.DataFrame: A DataFrame containing the processed EEG data.
        """
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Fetch EEG data for the specified song ID
        cursor.execute('''
            SELECT song_id, timestamp, psd, alpha, beta, delta, gamma, theta, 
                   focus_score, calm_score, year, month, day, weekday, hour 
            FROM eeg_metrics 
            WHERE song_id = ?
        ''', (song_id,))
        eeg_data = cursor.fetchall()
        conn.close()

        # Process the data
        processed_data = []
        for row in eeg_data:
            (
                song_id, timestamp, psd_blob, alpha_blob, beta_blob, delta_blob, gamma_blob, theta_blob,
                focus_score, calm_score, year, month, day, weekday, hour
            ) = row

            # Decode binary blobs into numpy arrays
            psd = np.frombuffer(psd_blob, dtype=np.float64)
            alpha = np.frombuffer(alpha_blob, dtype=np.float64)
            beta = np.frombuffer(beta_blob, dtype=np.float64)
            delta = np.frombuffer(delta_blob, dtype=np.float64)
            gamma = np.frombuffer(gamma_blob, dtype=np.float64)
            theta = np.frombuffer(theta_blob, dtype=np.float64)

            # Append structured data
            processed_data.append({
                "song_id": song_id,
                "timestamp": timestamp,
                "psd": psd,
                "alpha": alpha,
                "beta": beta,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "focus_score": focus_score,
                "calm_score": calm_score,
                "year": year,
                "month": month,
                "day": day,
                "weekday": weekday,
                "hour": hour
            })

        if len(processed_data) == 0:
            print(f"No EEG data found for song ID: {song_id}")
            return None

        return processed_data
