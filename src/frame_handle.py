import os
import cv2
import threading
import queue
import time

class ThreadedVideoWriter:
    def __init__(self, filename, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.q = queue.Queue(maxsize=128)
        self.stopped = False
        
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()
        
    def write(self, frame):
        if not self.stopped and not self.q.full(): 
            self.q.put(frame.copy())
            
    def _write(self):
        while not self.stopped or not self.q.empty():
            if not self.q.empty(): 
                self.writer.write(self.q.get())
            else: 
                time.sleep(0.01) 
        self.writer.release()
                
    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()