import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
import sqlite3
from datetime import datetime
import os
DB = "music_focus.db"


sp_oauth = SpotifyOAuth(
    client_id="218f2ba7042d49c5adda2036f2a99200",
    client_secret="4da476a0609d4de5a154d444acd61b29",
    redirect_uri="https://open.spotify.com/",
    scope="user-read-playback-state user-read-currently-playing"
)

sp = spotipy.Spotify(auth_manager=sp_oauth)

def get_audio_features(track_id):
    # Retrieve audio features for the given track
    features = sp.audio_features(track_id)[0]

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
def get_audio_analysis(track_id):
    # Refresh token if necessary
    token_info = sp_oauth.get_access_token()
    access_token = token_info['access_token']

    # Get audio features, including the analysis_url
    features = sp.audio_features(track_id)[0]
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


def delete_database(db_name="music_focus.db"):
    try:
        if os.path.exists(db_name):
            os.remove(db_name)
            print(f"Database '{db_name}' deleted successfully.")
        else:
            print(f"Database '{db_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the database: {e}")

def initialize_database(db_name="music_focus.db"):
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
            uri TEXT
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
            alpha REAL,
            beta REAL,
            gamma REAL,
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


def add_eeg_metrics(song_id, eeg_data, db_name="music_focus.db"):
    # Debug prints to confirm function is called with correct values
    print(f"Adding EEG metrics for song ID: {song_id}")
    print("EEG Data:", eeg_data)

    conn = sqlite3.connect(db_name)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
    cursor = conn.cursor()

    # Insert each EEG metric with associated song_id and date/time information
    for data in eeg_data:
        timestamp = data['timestamp']
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        hour = timestamp.hour

        print(f"Inserting EEG data with timestamp: {timestamp}")

        cursor.execute('''
            INSERT INTO eeg_metrics (song_id, timestamp, alpha, beta, gamma, year, month, day, weekday, hour)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            song_id, timestamp, data['alpha'], data['beta'], data['gamma'],
            year, month, day, weekday, hour
        ))

    # Commit changes to ensure data is saved
    conn.commit()
    conn.close()
    print("EEG metrics successfully added to database.")
def add_song_to_database(song_data, db_name="music_focus.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Insert song data into song_metrics (without date and time fields)
    cursor.execute('''
        INSERT OR REPLACE INTO song_metrics (
            id, track_name, acousticness, danceability, duration_ms, energy,
            instrumentalness, key, liveness, loudness, mode, speechiness, tempo,
            time_signature, valence, uri
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        song_data['id'], song_data['track_name'], song_data['acousticness'],
        song_data['danceability'], song_data['duration_ms'], song_data['energy'],
        song_data['instrumentalness'], song_data['key'], song_data['liveness'],
        song_data['loudness'], song_data['mode'], song_data['speechiness'],
        song_data['tempo'], song_data['time_signature'], song_data['valence'],
        song_data['uri']
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
def get_current_song_metrics():
    current_track = sp.current_playback()
    if current_track and current_track['item']:
        track_id = current_track['item']['id']
        track_name = current_track['item']['name']
        artists = [artist['name'] for artist in current_track['item']['artists']]

        features = sp.audio_features(track_id)[0]
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


# delete_database("music_focus.db")
# initialize_database(DB)

# Example usage: add current song to the database with artists
song_data = get_current_song_metrics()
if song_data:
    add_song_to_database(song_data)  # Add song data first to create the record

    # Example EEG data with datetime.now() for timestamp
    eeg_data = [
        {"timestamp": datetime.now(), "alpha": 0.5, "beta": 0.3, "gamma": 0.7}
    ]
    add_eeg_metrics(song_data["id"], eeg_data)  # Add associated EEG data

