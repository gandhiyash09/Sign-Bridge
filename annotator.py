"""
annotator.py

Simple OpenCV-based annotation GUI to correct per-frame keypoints and re-save JSONs.
Usage:
    python annotator.py --json path/to/video_YYYY.json --video path/to/video_YYYY.mp4

Controls:
- Left mouse drag: move nearest keypoint
- n: next frame
- p: previous frame
- s: save JSON
- q or ESC: quit

The tool overlays the canonical 48 nodes (with indices). Click-drag a point to reposition it.
When saved, it updates the JSON file in place.

"""

import argparse
import json
import cv2
import numpy as np

# load nodes ordering from the json

def load_data(json_path, video_path=None):
    with open(json_path, 'r') as f:
        j = json.load(f)
    frames = j['frames']
    # convert to numpy arrays
    F = len(frames)
    V = len(frames[0])
    arr = np.zeros((F, V, 3), dtype=np.float32)
    for t in range(F):
        for v in range(V):
            x,y,c = frames[t][v]
            arr[t,v,0] = -1 if x is None else x
            arr[t,v,1] = -1 if y is None else y
            arr[t,v,2] = c
    return j, arr


def save_data(json_path, j, arr):
    F = arr.shape[0]
    V = arr.shape[1]
    frames = []
    for t in range(F):
        fr = []
        for v in range(V):
            x = None if arr[t,v,0] < 0 else float(arr[t,v,0])
            y = None if arr[t,v,1] < 0 else float(arr[t,v,1])
            c = float(arr[t,v,2])
            fr.append([x,y,c])
        frames.append(fr)
    j['frames'] = frames
    with open(json_path, 'w') as f:
        json.dump(j, f, indent=2)
    print('[Annotator] saved', json_path)


class Annotator:
    def __init__(self, json_path, video_path=None):
        self.json_path = json_path
        self.video_path = video_path
        self.j, self.arr = load_data(json_path, video_path)
        self.F = self.arr.shape[0]
        self.V = self.arr.shape[1]
        self.t = 0
        self.dragging = False
        self.drag_idx = None
        self.window = 'Annotator'
        self.frame_img = None
        # if video provided, open for frame background; else black
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = None
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def get_frame_image(self, t):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, img = self.cap.read()
            if not ret:
                # fallback black frame
                img = np.zeros((480,640,3), dtype=np.uint8)
        else:
            img = np.zeros((480,640,3), dtype=np.uint8)
        return img

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # find nearest keypoint in current frame
            pts = self.arr[self.t,:, :2]
            dists = np.linalg.norm(pts - np.array([x,y]), axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < 40:
                self.dragging = True
                self.drag_idx = idx
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.drag_idx is not None:
                self.arr[self.t, self.drag_idx, 0] = x
                self.arr[self.t, self.drag_idx, 1] = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_idx = None

    def draw(self):
        img = self.get_frame_image(self.t)
        overlay = img.copy()
        for v in range(self.V):
            x,y,c = self.arr[self.t, v]
            if x >= 0 and y >= 0:
                cv2.circle(overlay, (int(x), int(y)), 4, (0,255,0), -1)
                cv2.putText(overlay, str(v), (int(x)+3, int(y)+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(overlay, f'Frame {self.t+1}/{self.F}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return overlay

    def run(self):
        while True:
            img = self.draw()
            cv2.imshow(self.window, img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                self.t = min(self.F-1, self.t+1)
            elif key == ord('p'):
                self.t = max(0, self.t-1)
            elif key == ord('s'):
                save_data(self.json_path, self.j, self.arr)
            elif key == ord('q') or key == 27:
                break
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True)
    parser.add_argument('--video', default=None)
    args = parser.parse_args()
    ann = Annotator(args.json, args.video)
    ann.run()
