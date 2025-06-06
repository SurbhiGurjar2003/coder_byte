# -*- coding: utf-8 -*-
"""test_server.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Yu7vzcblg28vc3BX9msqNZ-0xwiT75Te
"""

import requests
import argparse
from model import ImagePreprocessor
import json

def test_deployment(image_path, server_url):
    prep = ImagePreprocessor()
    img_tensor = prep.preprocess(image_path)
    payload = img_tensor.numpy().tolist()
    response = requests.post(server_url, json={"input": payload})
    print("Prediction:", response.text)

def run_custom_tests():
    # Extend this as needed
    print("Running custom deployment tests...")
    # Simulate latency test or uptime check

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Path to the image')
    parser.add_argument('--url', type=str, help='Deployed model server URL')
    parser.add_argument('--custom', action='store_true', help='Run custom platform tests')
    args = parser.parse_args()

    if args.custom:
        run_custom_tests()
    else:
        test_deployment(args.img, args.url)