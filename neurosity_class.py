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







class NeurosityVectorizer:
    """
    This class is used to vectorize the data from the Neurosity hardware or a simulator
    """
    def __init__(self, simulator):
        self.simulator = simulator
        self.latest_raw = None
        self.latest_raw_unfiltered = None
        self.latest_psd = None
        self.latest_power_by_band = None
        self.latest_focus = None
        self.latest_calm = None

        # Subscribe to all data streams
        self.simulator.brainwaves_raw(self.update_raw)  # Shape: (8, 16)
        self.simulator.brainwaves_raw_unfiltered(self.update_raw_unfiltered)  # Shape: (8, 16)
        self.simulator.brainwaves_psd(self.update_psd)  # Shape: (8, 64)
        self.simulator.brainwaves_power_by_band(self.update_power_by_band)  # Shape: (5, 8)
        self.simulator.focus(self.update_focus)  # Shape: (1, 1)
        self.simulator.calm(self.update_calm)  # Shape: (1, 1)

        # Quality bool
        self.quality_met = False
        self.max_fails = 3
        self.status = self.get_status()["state"] # online or offline
        self.charging =  self.get_status()["charging"] # True or False

        self.current_song_psd = np.zeros((0,8,26))
        self.current_song_power_by_band = np.zeros((0,5,8))
        self.current_song_focus = np.array([])
        self.current_song_calm = np.array([])

        # sleep(1.5) # to get 808 vectors straight away

    def get_status(self):
        return self.simulator.status_once()

    def update_raw(self, data):
        # self.latest_raw = np.array(data['data'])
        self.latest_raw = data['data']
    def update_raw_unfiltered(self, data):
        # self.latest_raw_unfiltered = np.array(data['data'])
        self.latest_raw_unfiltered = data['data']
    def update_psd(self, data):
        # self.latest_psd = np.array(data['psd'])
        self.latest_psd = data['psd']
    def update_power_by_band(self, data):
        # self.latest_power_by_band = np.array(list(data['data'].values()))
        self.latest_power_by_band = list(data['data'].values())

    def update_focus(self, data):
        self.latest_focus = data["probability"]

    def update_calm(self, data):
        self.latest_calm = data["probability"]

    def export_np(self):
        """
        Returns all the latest data as a single numpy array.
        The array is in the shape (808,)
        Sections of the array are as follows:
        0:128 - raw
        128:256 - raw unfiltered
        256:768 - psd
        768:808 - power by band
        """
        # Convert to numpy arrays
        latest_raw = np.array(self.latest_raw)
        latest_raw_unfiltered = np.array(self.latest_raw_unfiltered)
        latest_psd = np.array(self.latest_psd)
        latest_power_by_band = np.array(self.latest_power_by_band)
        # print("latest_raw", len(latest_raw))
        # print("latest_raw_unfiltered", len(latest_raw_unfiltered))
        # print("latest_psd", len(latest_psd))
        # print("latest_power_by_band", len(latest_power_by_band))
        # print("sum of lengths", len(latest_raw) + len(latest_raw_unfiltered) + len(latest_psd) + len(latest_power_by_band))

        # Concatenate and flatten all the latest data
        all_data = [latest_raw, latest_raw_unfiltered, latest_psd, latest_power_by_band]
        vector_808 = np.concatenate([data.flatten() for data in all_data if data is not None])
        if len(vector_808) != 808:
            print(f"WARNING: vector_808 is not 808 long (length: {len(vector_808)})")
            return
        elif len(vector_808) == 808:
            return vector_808

    def export_json(self):
        return json.dumps(self.export_np().tolist())
    def export_image(self):
        """
        For viewing convenience, this function exports brain data as a heatmap like image
        The data is normally in the shape (808,) but this function slices the data to an (800,) shape
        then exports it as a 40x20 image in grayscale
        """
        # Get the data
        data = self.export_np()
        # Slice the data to an (800,) shape
        data = data[0:800]
        # Minmax scale each segment of the data
        data[0:128] = (data[0:128] - data[0:128].min()) / (data[0:128].max() - data[0:128].min())
        data[128:256] = (data[128:256] - data[128:256].min()) / (data[128:256].max() - data[128:256].min())
        data[256:768] = (data[256:768] - data[256:768].min()) / (data[256:768].max() - data[256:768].min())
        data[768:800] = (data[768:800] - data[768:800].min()) / (data[768:800].max() - data[768:800].min())

        # Reshape the data to a (40, 20) shape
        data = data.reshape((40, 20))
        # Return the data
        return data

    def signal_quality_callback(self, data):
        fails = 0
        for status in data:
            # print(data)
            print(status["status"], status["standardDeviation"])
            if status["status"] not in ["good", "great"]:
                fails += 1

        if fails > self.max_fails:
            print("Signal quality still not good enough (" + str(fails) + " fails)")
            return
        self.quality_met = True

    def ensure_quality(self, max_fails=3):
        """Ensures a good signal quality for data collection and inference. Continually checks on the signal quality until it's good enough.
        Parameters:
            max_fails (int): The maximum number of times the signal quality can fail before this function returns False. Default: 3"""

        signal_quality_unsub = self.simulator.signal_quality(self.signal_quality_callback)
        while not self.quality_met:
            sleep(1)
        signal_quality_unsub()
        return True

    def gather_samples(self, num_inputs, samples_per_input, sample_rate, data_label, data_class):
        """
        Gathers training data for a specified number of inputs, each with a specified number of samples.
        For example, if num_inputs is 2 and samples_per_input is 3, then 6 samples will be gathered.
        To expand on this example, if the sample rate is 250 (expressed as Hz), then 6 samples will be gathered with a 1/250 second interval between each sample,
        for a total of 1.5 seconds of data.
        For each input taken, one .npy file will be produced in the appropriate training data folder.

        Parameters:
        num_inputs (int): The number of inputs to gather data for.
        samples_per_input (int): The number of samples to gather for each input.
        sample_rate (int): The sample rate, expressed in Hz.
        data_label (str): The label for the data. For example, "doing_math"
        data_class (str): The class for the data. For example, "doing_math" vs "not_doing_math".

        Returns:
        None
        """
        # sleep(1)
        # Calculate the length of time to gather data for and print it for the user
        sampling_time_total = float(num_inputs * samples_per_input) / sample_rate
        print(f"Sampling for {round(sampling_time_total, 5)} seconds...")
        actual_start_time = time()

        # Make a training_data directory if it doesn't exist
        if not path.exists("training_data"):
           mkdir("training_data")
        # Make a directory for the label if it doesn't exist
        if not path.exists(f"training_data/{data_label}"):
           mkdir(f"training_data/{data_label}")
        # Make a directory for the class if it doesn't exist
        if not path.exists(f"training_data/{data_label}/{data_class}"):
           mkdir(f"training_data/{data_label}/{data_class}")

        # Gather all the samples. They'll be split into inputs later.
        all_samples = []
        for i in range(num_inputs * samples_per_input):
            start_time = time()

            next_time = start_time + (1. / sample_rate)

            all_samples.append(self.export_np()) # appends a vector (808,) to all_samples
            # sleep(1. / sample_rate)
            print(f"Sample {i + 1} of {num_inputs * samples_per_input} complete")


            # New method - if there's now enough data in all_samples to make an input, save it
            if len(all_samples) >= samples_per_input:
                # Get the first samples_per_input samples
                input_samples = all_samples[0:samples_per_input]
                # Remove those samples from all_samples
                all_samples = all_samples[samples_per_input:]
                # Save the input
                ts = int(round(time() * 1000))
                t1 = time()
                # np.save(f"training-data/{data_label}/{data_class}/{ts}.npy", input_samples)
                # Old method was using np.save, but it's too slow. Run it in a thread instead.
                # Thread(target=np.save, args=(f"training-data/{data_label}/{data_class}/{ts}.npy", input_samples)).start()
                # newer method is to use a queue
                # save_queue.put((f"training_data/{data_label}/{data_class}/{ts}.npy", input_samples)) # each input is a 2d array of shape (samples_per_input, 808)
                # newest method is to use sqlite
                # Will be accessed like: timestamp, data_label, data_class, data = sqlite_queue.get()

            # naprinti si vse podatke k grejo v queue da vidis ker zajebe
            #     print("ts", ts)
            #     print("data_label", data_label)
            #     print("data_class", data_class)
                sqlite_queue.put((ts, data_label, data_class, input_samples))
                t2 = time()
                # print(f"Saved input {ts}.npy to training-data/{data_label}/{data_class}/ (took {round(t2-t1, 5)} seconds)")
                print(f"Saved input {ts} to sqlite (took {round(t2 - t1, 5)} seconds)")

            # Determine how long to sleep for
            sleep_time = next_time - time()
            # If sleep_time is positive, sleep until it's time to take the next sample
            # print("DEBUG: sleep_time =", sleep_time)
            if sleep_time > 0:
                sleep(sleep_time)

        # Print status
        print(f"Saved {num_inputs * samples_per_input} samples to training_data/{data_label}/{data_class}/")
        print(f"total time taken: {round(time() - actual_start_time, 5)} seconds")
        # Return nothing. Read from file instead.
        return None

    def sample(self, num_samples, sample_rate):
        """Similar to gather_samples, but returns the samples instead of saving them to a file.
        Intended for use in real-time applications.
        """
        # Calculate the length of time to gather data for and print it for the user
        sampling_time_total = float(num_samples) / sample_rate
        # print(f"Sampling for {round(sampling_time_total,5)} seconds...")

        # Gather all the samples. They'll be split into inputs later.
        all_samples = []
        for i in range(num_samples):
            all_samples.append(self.export_np())
            sleep(1. / sample_rate)
            # print(f"Sample {i + 1} of {num_samples} complete")
        # Return the samples
        return np.array(all_samples)

    def validate_samples(self, data_label):
        """
        Validates the samples for all classes belonging to a specified label. It does this by loading each .npy file in the appropriate folder and checking its shape.
        If the shape of any file is different from the others, this function will return False. Otherwise, it will return True.

        Parameters:
        data_label (str): The label for the data. For example, "doing_math"

        Returns:
        bool: True if the samples are valid, False otherwise.
        """
        # Get the directory for the label
        directory = f"training_data/{data_label}/"
        # Get the list of classes for the label
        classes = [f for f in listdir(directory) if path.isdir(path.join(directory, f))]
        # Get the shape of the first file
        first_file = listdir(path.join("training_data/", data_label, classes[0]))[0]
        shape = np.load(path.join(directory, classes[0], first_file)).shape
        # Check the shape of each file
        for c in classes:
            for f in listdir(path.join(directory, c)):
                if np.load(path.join(directory, c, f)).shape != shape:
                    return False
        # If all the shapes are the same, return True
        return True

    def convert_to_sqlite(self, db_name):
        # Make sure that the DB exists and has the proper tables
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute(
            "CREATE TABLE IF NOT EXISTS training_data_table (id INTEGER PRIMARY KEY, timestamp REAL, label TEXT, class TEXT, data BLOB)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_training_data ON training_data_table (timestamp, label, class)")

        # Get a list of all the directories in training-data
        labels = [f for f in listdir("training_data/") if path.isdir(path.join("training_data/", f))]
        # Iterate over each label
        for label in labels:
            # Get all the classes for the label
            classes = [f for f in listdir(path.join("training_data/", label)) if
                       path.isdir(path.join("training_data/", label, f))]
            # Iterate over each class
            for data_class in classes:
                counter = 0
                print("Converting", label + "/" + data_class)
                # Get all the files for the class
                files = [f for f in listdir(path.join("training_data/", label, data_class)) if f.endswith(".npy")]
                # Iterate over each file
                for f in files:
                    try:
                        # Load the file
                        data = np.load(path.join("training_data/", label, data_class, f))
                        # Flatten to a 1d array, since that's my new standard rather than time series
                        data = data.flatten()
                        # Convert the data to a buffer
                        data = data.tobytes()
                        # See if a row already exists with  this timestamp, label, and class
                        c.execute("SELECT * FROM training_data WHERE timestamp=? AND label=? AND class=?",
                                  (float(f.replace(".npy", "")) / 1000., label, data_class))
                        # If so, don't insert it again
                        if c.fetchone() is not None:
                            continue
                        # Insert the data into the DB
                        c.execute("INSERT INTO training_data (timestamp, label, class, data) VALUES (?, ?, ?, ?)",
                                  (float(f.replace(".npy", "")) / 1000., label, data_class, data))
                        # Increment the counter and print a status every 1k files
                        counter += 1
                        if counter % 1000 == 0:
                            print("Processed", counter, "files in training_data/" + label + "/" + data_class)
                            # Commit the changes
                            conn.commit()
                    except Exception as e:
                        print("Error loading file", f, e)
        # Commit the changes and close the connection
        conn.commit()
        # Close the connection
        conn.close()
        print("Done converting to sqlite")

    def gather_eeg_samples_during_song(self):
        # append current song data to the current song data arrays

        # convert psd to numpy array
        psd = np.array(self.latest_psd)
        psd = psd[:, 0:26]
        self.current_song_psd = np.concatenate((self.current_song_psd, psd[np.newaxis, ...]), axis=0)

        bands = np.array(self.latest_power_by_band)
        self.current_song_power_by_band = np.concatenate((self.current_song_power_by_band, bands[np.newaxis, ...]), axis=0)

        if self.latest_focus is not None:
            self.current_song_focus = np.append(self.current_song_focus, self.latest_focus)

        if self.latest_calm is not None:
            self.current_song_calm = np.append(self.current_song_calm, self.latest_calm)





    def get_current_song_eeg_data(self):
        self.current_song_psd = np.mean(self.current_song_psd, axis=0)
        self.current_song_power_by_band = np.mean(self.current_song_power_by_band, axis=0)
        # split the power by band into individual bands
        self.current_song_alpha = self.current_song_power_by_band[0,:]
        self.current_song_beta = self.current_song_power_by_band[1,:]
        self.current_song_delta = self.current_song_power_by_band[2,:]
        self.current_song_gamma = self.current_song_power_by_band[3,:]
        self.current_song_theta = self.current_song_power_by_band[4,:]

        self.current_song_focus = np.mean(self.current_song_focus)
        self.current_song_calm = np.mean(self.current_song_calm)

        print("Current song EEG data gathered")

        # put in a dict and return

        eeg_dict = {
            "psd": self.current_song_psd,
            "alpha": self.current_song_alpha,
            "beta": self.current_song_beta,
            "delta": self.current_song_delta,
            "gamma": self.current_song_gamma,
            "theta": self.current_song_theta,
            "focus": self.current_song_focus,
            "calm": self.current_song_calm
        }
        return eeg_dict
        # return self.current_song_psd, self.current_song_alpha,self.current_song_beta, self.current_song_delta, self.current_song_gamma,self.current_song_theta, self.current_song_focus, self.current_song_calm

    def reset_current_song_eeg_data(self):
        self.current_song_psd = np.zeros((0,8,26))
        self.current_song_power_by_band = np.zeros((0,5,8))
        self.current_song_focus = np.array([])
        self.current_song_calm = np.array([])

        print("Current song EEG data reset")
