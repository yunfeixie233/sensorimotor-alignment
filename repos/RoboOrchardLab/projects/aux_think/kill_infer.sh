#!/bin/bash

echo "Killing all run_api.py processes..."
pkill -f run_api.py

echo "Killing all infer.py processes..."
pkill -f infer.py

echo "All API servers and inference jobs have been terminated."
