import os
import cv2
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import highway_env
import ffmpeg
import time
import io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import csv


def save_video_with_ffmpeg(frame_sequence, video_path, fps, frame_size):
    """Save the frames as a video using ffmpeg."""
    process = (
        ffmpeg
        .input('pipe:0', format='rawvideo', pix_fmt='rgb24', s=f'{frame_size[0]}x{frame_size[1]}', r=fps)
        .output(video_path, vcodec='vp8', pix_fmt='yuv420p', crf=23)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in frame_sequence:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()


def save_plot_as_image_with_metrics(metrics):
    """Generate the plot image with episode metrics overlaid."""
    fig = plt.figure(figsize=(10, 8))  # Increased height for space
    spec = GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[2, 1])  # 2 parts: plot (2) and text (1)

    # Plotting section (top part)
    ax = fig.add_subplot(spec[0])  # Top part for the plot
    ax.set_xlabel("Episode")
    if metrics["Episode"]:
        ax.set_title(f"Episode {metrics['Episode'][-1]} Metrics", fontsize=18)
    ax.grid()

    # Plot the different metrics with specified colors and markers
    ax.plot(metrics["Episode"], metrics["Average Speed"], color="blue", label="Average Speed", linestyle='-', marker='o')
    ax.plot(metrics["Episode"], metrics["Number of Arrivals"], color="green", label="Number of Arrivals", linestyle='-', marker='x')
    ax.plot(metrics["Episode"], metrics["Number of Collisions"], color="red", label="Number of Collisions", linestyle='-', marker='s')
    
    ax.legend()

    # Text section (bottom part)
    ax_text = fig.add_subplot(spec[1])  # Bottom part for the text
    ax_text.axis('off')  # Disable the axis since we don't need it for text

    # Prepare text content
    metrics_text_left = []
    metrics_text_right = []

    if metrics["Episode Length"]:
        metrics_text_left.append(f"Episode Length: {metrics['Episode Length'][-1]}")
    if metrics["Average Speed"]:
        metrics_text_left.append(f"Avg. Speed: {metrics['Average Speed'][-1]:.2f}")
    if metrics["Number of Arrivals"]:
        metrics_text_right.append(f"Arrivals: {metrics['Number of Arrivals'][-1]}")
    if metrics["Number of Collisions"]:
        metrics_text_right.append(f"Collisions: {metrics['Number of Collisions'][-1]}")

    # Add text to the bottom part, using two columns
    y_start = 0.9  # Starting point for text (scaled to 0-1)
    line_spacing = 0.1  # Vertical space between text lines
    text_fontsize = 18  # Larger font size for better readability

    # Left column text
    for i, line in enumerate(metrics_text_left):
        ax_text.text(0.1, y_start - i * line_spacing, line, ha='left', va='center', fontsize=text_fontsize, color='black')

    # Right column text
    for i, line in enumerate(metrics_text_right):
        ax_text.text(0.6, y_start - i * line_spacing, line, ha='left', va='center', fontsize=text_fontsize, color='black')

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.tight_layout(pad=3.0)  # Add more padding to avoid overlapping
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    # Read the image from the buffer using OpenCV
    img_array = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    
    # Convert from BGR to RGB (matplotlib format)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    buf.close()

    return img_array


if __name__ == "__main__":

    plt.style.use('default')  # Reset to the default style

    # Directory to save the video and plots
    video_dir = "videos/"
    os.makedirs(video_dir, exist_ok=True)

    algorithm="ppo"
    policy="cnn"
    traffic="sparse"
    n_episodes = 100

    # Paths for videos
    env_video_path = os.path.join(video_dir, f"environment_{algorithm}_{policy}_{traffic}.webm")
    plot_video_path = os.path.join(video_dir, f"speed_{algorithm}_{policy}_{traffic}.webm")

    # Load the trained model
    model = PPO.load(f"intersection_{algorithm}_{policy}_multi/model")

    # Create the environment
    env = gym.make(
        "intersection-v0",
        render_mode="rgb_array",
        config={
            'initial_vehicle_count': 0,
            'controlled_vehicles': 4,
            'destination': 'o1',
            "observation": {
                "vehicles_count": 0,
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75,
            },
        },
    )

    # Reset the environment and get dimensions for video
    obs, _ = env.reset()
    frame = env.render()
    height, width, _ = frame.shape
    frame_size = (width, height)
    fps = env.metadata.get("render_fps", 30)
    fps = 6

    # Frame sequences for videos
    env_frame_sequence = []
    plot_frame_sequence = []

    # Metrics to plot
    metrics = {"Episode": [], "Episode Length": [], "Average Speed": [], "Number of Arrivals": [], "Number of Collisions": []}

    # Initialize accumulators for global metrics
    total_speed_sum = 0
    total_step_count = 0
    total_arrivals = 0
    total_collisions = 0
    avg_episode_length=0

    # Evaluation loop
   
    for episode in range(n_episodes):
        obs, _ = env.reset()
        speed_sum, step_count = 0, 0
        avg_speed = 0.0
        done = False
        total_arrived_reward = 0
        final_agents_terminated = None
        number_of_arrivals = 0
        number_of_collisions = 0

        while not done:
            # Predict action using the trained model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, infos = env.step(action)
            done = terminated or truncated

            # Update metrics
            speed_sum += infos['speed']
            step_count += 1

            # Render environment frame
            env_frame = env.render()
            env_frame_sequence.append(env_frame)

            # Update the plot frame with metrics
            plot_frame = save_plot_as_image_with_metrics(metrics)

            plot_frame_resized = cv2.resize(plot_frame, (width, height))
            plot_frame_sequence.append(plot_frame_resized)

        # Calculate episode metrics
        avg_speed = speed_sum / step_count if step_count > 0 else 0
        total_arrived_reward = infos["rewards"]["arrived_reward"]
        number_of_arrivals = total_arrived_reward / 0.25

        if not truncated:
            final_agents_terminated = infos["agents_terminated"]
            number_of_collisions = sum(final_agents_terminated)

        # Log episode results
        print(f"\nEpisode {episode + 1}: Avg. Speed = {avg_speed:.2f}, Arrivals = {number_of_arrivals}, Collisions = {number_of_collisions}")

        # Update cumulative metrics
        metrics["Episode"].append(episode + 1)
        metrics["Episode Length"].append(step_count)
        metrics["Average Speed"].append(avg_speed)
        metrics["Number of Arrivals"].append(number_of_arrivals)
        metrics["Number of Collisions"].append(number_of_collisions)

        total_speed_sum += speed_sum
        total_step_count += step_count
        total_arrivals += number_of_arrivals
        total_collisions += number_of_collisions
        

    # Compute global metrics
    global_avg_speed = total_speed_sum / total_step_count if total_step_count > 0 else 0
    avg_arrivals_per_episode = total_arrivals / n_episodes
    avg_collisions_per_episode = total_collisions / n_episodes
    avg_episode_length = total_step_count / n_episodes

   
    # Define the output CSV file path
    csv_file_path = os.path.join(video_dir, "global_metrics.csv")

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Check if the file is empty, if so, write the header
        if csv_file.tell() == 0:  # File is empty, so write the header
            csv_writer.writerow(["algorithm", "policy", "traffic", "avg_episode_length", "global_avg_speed", "avg_arrivals_per_episode", "avg_collisions_per_episode"])
        
        # Write the data (metrics)
        csv_writer.writerow([algorithm, policy, traffic, avg_episode_length, global_avg_speed, avg_arrivals_per_episode, avg_collisions_per_episode])

    print(f"Global metrics appended to {csv_file_path}")


    # Save the environment video
    save_video_with_ffmpeg(env_frame_sequence, env_video_path, fps, frame_size)

    # Save the plot video with global metrics
    save_video_with_ffmpeg(plot_frame_sequence, plot_video_path, fps, frame_size)












