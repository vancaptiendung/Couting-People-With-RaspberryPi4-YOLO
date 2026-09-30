import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts.warning=false"

import cv2
import ncnn
import numpy as np
import threading
import time
import json
import glob
import queue
import re
import shutil
import sqlite3
import requests
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request
from picamera2 import Picamera2

import log_handle
import const

#system infomation
TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM = log_handle.load_data()

latest_frame = None
output_frame_bgr = None
frame_lock = threading.Lock()
recording_enabled = False
cam_fps = 0
ai_fps = 0

VIDEO_DIR = "videos"
MAX_VIDEOS = 24
CHUNK_DURATION = 30 * 60 
if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)

#setup system
log_handle.init_db()
log_handle.log_event("SYSTEM_START", details="Hệ thống Camera AI Pi 5 khởi động")