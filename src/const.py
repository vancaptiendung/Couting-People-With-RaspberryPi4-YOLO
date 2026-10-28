import os

DATA_FILE = "counter_data.json"
DB_FILE = "cctv_logs.db"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(ROOT_DIR, "../models")
BIN_FILE_PATH = os.path.join(MODELS_DIR, "yolo-fastest-1.1.bin")
PARAM_FILE_PATH = os.path.join(MODELS_DIR, "yolo-fastest-1.1.param")

AI_SIZE = 320 
DISPLAY_W, DISPLAY_H = 640, 480 
LINE_X = DISPLAY_W // 2 
MAX_DISAPPEARED = 30