# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Perform test request
"""

import pprint

import requests

DETECTION_URL = 'http://localhost:8080/v1/object-detection/model'
IMAGE = 'zidane.jpg'

# Read image
with open(IMAGE, 'rb') as f:
    image_data = f.read()

response = requests.post(DETECTION_URL, files={'image': image_data}).json()

pprint.pprint(response)
