#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv env

echo "Created virtual environment"
echo "Activating virtual environment..."
source ./env/bin/activate

echo "Activated virtual environment"
echo "Installing dependencies..."
pip3 install -r requirements.txt
