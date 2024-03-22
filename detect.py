import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

basename = sys.argv[1]
video_path = f"videos/{basename}.mov"
capture = cv2.VideoCapture(video_path)

if not capture.isOpened():
    print(f"Invalid filename: {video_path}")
    sys.exit()

# 156 px = 6,6 cm

def color_bounds(color, offset):
    offset_v = np.repeat(offset, 3)
    lower = color - np.minimum(color, offset_v)
    upper = color + np.minimum(255 - color, offset_v)
    return lower, upper

color_yellow = np.array([44, 159, 185], dtype=np.uint8)
yellow_lb, yellow_ub = color_bounds(color_yellow, 40)

color_green = np.array([61, 48, 7], dtype=np.uint8)
green_lb, green_ub = color_bounds(color_green, 45)

yellow_positions, green_positions = [], []

fps = 30
frame_index = 0

try:
    while True:
        is_available, frame = capture.read()
        if not is_available:
            print("No more frames available")
            break

        section = frame[200:500,:]

        yellow_mask = cv2.inRange(section, yellow_lb, yellow_ub)
        yellow_pos = np.mean(np.nonzero(yellow_mask), axis=1)

        green_mask = cv2.inRange(section, green_lb, green_ub)
        green_pos = np.mean(np.nonzero(green_mask), axis=1)

        yellow_positions.append(yellow_pos)
        green_positions.append(green_pos)

        frame_index += 1
        if frame_index % (fps * 20) == 0:
            print("20 seconds have been proccesed")

except KeyboardInterrupt:
    pass

capture.release()

def calculate_angle(pos):
    return np.arcsin(pos[1] / 1332.0036)

yellow_angles = [calculate_angle(pos - yellow_positions[0]) for pos in yellow_positions]
green_angles = [calculate_angle(pos - green_positions[0]) for pos in green_positions]

with open(f"data/{basename}.txt", "w") as file:
    file.write(",".join([str(n) for n in yellow_angles]) + "\n")
    file.write(",".join([str(n) for n in green_angles]))

time = np.linspace(0, len(yellow_angles) / fps, len(yellow_angles))
plt.plot(time, yellow_angles, c="gold")
plt.plot(time, green_angles, c="limegreen")
plt.savefig(f"data/{basename}.png")
