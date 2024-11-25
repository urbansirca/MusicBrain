from dotenv import load_dotenv, dotenv_values
from neurosity import NeurositySDK
import os
import time
import numpy as np
from matplotlib import pyplot as plt
# import pygame
from neurosity_class import NeurosityVectorizer





load_dotenv("environment.env")

email = os.environ.get('NEUROSITY_EMAIL')
password = os.environ.get("NEUROSITY_PASSWORD")
device_id = os.environ.get("NEUROSITY_DEVICE_ID")

# email = os.getenv("NEUROSITY_EMAIL")
# password = os.getenv("NEUROSITY_PASSWORD")
# device_id = os.getenv("NEUROSITY_DEVICE_ID")


# print(device_id)


neurosity = NeurositySDK({
    "device_id": device_id})
neurosity.login({
    "email": email,
    "password": password})



# Create a vectorizer
vectorizer = NeurosityVectorizer(neurosity)

# create code so that when esure?quality is becomes true, continue with gathering samples
while not vectorizer.ensure_quality():
    continue
print("Quality ensured")
time.sleep(1.5)


# def callback(data):
#     print("data", data)
#
# unsubscribe = neurosity.brainwaves_psd(callback)