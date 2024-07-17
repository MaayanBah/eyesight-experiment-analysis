import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('assets/fixation_average_sorted_by_time_real_data.csv', delimiter="\t", header=None)
graph_data = {line_index: list(line) for line_index, line in data.iterrows()}

# Video path
video_path = r'assets/Tom_and_Jerry_cut.mp4'

# Set up the video capture
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Delay in milliseconds

# Set up the plot
fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(12, 10))  # Changed to (2, 1) layout

# Function to update the plot
def update_plot(frame_number):
    # Get current time in seconds from the video
    current_time_sec = frame_number / fps

    # Determine the segment of graph data to display (latest 200 points)
    segment_length = 200
    start_index = max(0, int(current_time_sec) - segment_length)
    end_index = int(current_time_sec)

    # Ensure indices do not exceed data length
    start_index = max(start_index, 0)
    end_index = min(end_index, len(graph_data[0]))

    y_data_good_graph = graph_data[0][start_index:end_index]  # good
    y_data_bad_graph = graph_data[1][start_index:end_index]  # bad
    x_data = range(len(y_data_good_graph))

    ax2.clear()  # Clear the previous plot
    ax2.plot(x_data, y_data_good_graph, color='green', label='Graph 1')
    ax2.plot(x_data, y_data_bad_graph, color='red', label='Graph 2')
    ax2.set_xlim(0, segment_length)  # Set x-axis limit to 200 points
    ax2.set_title(f'Graph Display (Time: {current_time_sec:.2f} sec)')
    ax2.legend()

# Function to update video frame
def update_frame(frame_number):
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        ax1.clear()  # Clear the previous video frame
        ax1.imshow(frame)
        ax1.set_title('Video Display')
        ax1.axis('off')  # Turn off axis for better video display

# Create animation
def animate(i):
    update_plot(i)
    update_frame(i)
    return ax1, ax2

# Ensure the number of frames is appropriate
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=frame_delay, blit=False)

# Save the animation as a video file
output_video_path = 'output_video.mp4'  # Specify the output file path
writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
ani.save(output_video_path, writer=writer)

# Release video capture object
cap.release()
