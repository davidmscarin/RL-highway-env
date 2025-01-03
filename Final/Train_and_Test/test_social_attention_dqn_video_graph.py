'''
To run the code:
rl_agents/trainer/evaluation.py change the line line 79 to 
self.wrapped_env = RecordVideo(env,
                                self.run_directory,
                                episode_trigger=(lambda e: True))
since changing the episode trigger after the evaluation object is created was not working.
moviepy was not working on my machine so I had to install ImageMagick and update the path in the code.
'''


import gymnasium as gym
import sys
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import csv
import cv2
import io
import ffmpeg

sys.path.insert(0, 'C:/Users/tomas/Documents/University/Master/Year1/MS/MIA_MOD_SIM-main/rl-agents')

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

from moviepy.config import change_settings
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips


def join_videos(num_episodes, video_path):
    
    print("Joining videos...")
    video_dir = r'out\IntersectionEnv\DQNAgent\videos'
    # join all videos into one
    clips = []
    for i in range(num_episodes):
        video = os.path.join(video_dir, f'rl-video-episode-{i}.mp4')
        clip = VideoFileClip(video)
        clips.append(clip)
    
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(video_path)
    final_clip.close()


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
    #fig.patch.set_facecolor('white')
    spec = GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[2, 1])  # 2 parts: plot (2) and text (1)

    # Plotting section (top part)
    ax = fig.add_subplot(spec[0])  # Top part for the plot
    ax.set_facecolor('white')
    
    ax.set_xlabel("Episode")
    if metrics["Episode"]:
        ax.set_title(f"Episode {metrics['Episode'][-1]} Metrics", fontsize=18)
    ax.grid(True,color = 'gray')

    # Plot the different metrics with specified colors and markers
    ax.plot(metrics["Episode"], metrics["Average Speed"], color="blue", label="Average Speed", linestyle='-', marker='o')
    ax.plot(metrics["Episode"], metrics["Number of Arrivals"], color="green", label="Number of Arrivals", linestyle='-', marker='x')
    ax.plot(metrics["Episode"], metrics["Number of Collisions"], color="red", label="Number of Collisions", linestyle='-', marker='s')
    
    ax.legend()

    # Text section (bottom part)
    ax_text = fig.add_subplot(spec[1])  
    ax_text.axis('off')  

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
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
    y_start = 0.9  
    line_spacing = 0.1  
    text_fontsize = 18  

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

    # Save each plot as an image file

    # Read the image from the buffer using OpenCV
    img_array = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    
    # Convert from BGR to RGB (matplotlib format)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    buf.close()

    return img_array

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

if __name__ == '__main__':

    


    # Get the environment and agent configurations from the rl-agents repository
    #%cd /content/rl-agents/scripts/

    logger = logging.getLogger(__name__)

    env_config = 'configs/IntersectionEnv/env.json'
    agent_config = 'configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json'
    pretrained_model_path = r'C:\Users\tomas\Documents\University\Master\Year1\MS\MIA_MOD_SIM-main\rl-agents\scripts\out\IntersectionEnv\DQNAgent\run_20250101-214428_26200\checkpoint-best.tar'

    env = load_environment(env_config)
    agent = load_agent(agent_config, env)


    algorithm="dqn"
    policy="social attention"
    traffic="dense"
    NUM_EPISODES = 100  #@param {type: "integer"}

    eval = Evaluation(
        env=env,
        agent=agent,
        run_directory="videos",
        display_env=True,
        display_agent=True,
        num_episodes=NUM_EPISODES,
        recover=pretrained_model_path  # Load the pretrained model
    )

    print(f"Ready to test {agent} on {env}")

    #obs, _ = env.reset()
    #frame = env.render()
    height, width = 600,600
    frame_size = (width, height)
    #fps = env.metadata.get("render_fps", 30)
    fps = 30

    video_dir = "videos/"
    os.makedirs(video_dir, exist_ok=True)

    env_video_path = os.path.join(video_dir, f"environment_{algorithm}_{policy}_{traffic}.mp4")
    plot_video_path = os.path.join(video_dir, f"speed_{algorithm}_{policy}_{traffic}.webm")
    plot_frame_sequence = []

    # Metrics to plot
    metrics = {"Episode": [], "Episode Length": [], "Average Speed": [], "Number of Arrivals": [], "Number of Collisions": []}


    total_speed_sum = 0
    total_step_count = 0
    total_arrivals = 0
    total_collisions = 0
    avg_episode_length=0

    plot_frame = save_plot_as_image_with_metrics(metrics)

    plot_frame_resized = cv2.resize(plot_frame, (width, height))
    plot_frame_sequence.append(plot_frame_resized)
    
    for eval.episode in range(eval.num_episodes):
        
        speed_sum, step_count = 0, 0
        avg_speed = 0.0
        terminal = False
        total_arrived_reward = 0
        final_agents_terminated = None
        number_of_arrivals = 0
        number_of_collisions = 0



        eval.reset(seed=eval.episode)

    
        while not terminal:
            # Step
            actions = eval.agent.plan(eval.observation)
            previous_observation, action = eval.observation, actions[0]
            transition = eval.wrapped_env.step(action)
            eval.observation, reward, done, truncated, info = transition

            speed_sum += info['speed']
            step_count += 1

            terminal = done or truncated



        plot_frame = save_plot_as_image_with_metrics(metrics)

        plot_frame_resized = cv2.resize(plot_frame, (width, height))
        #plot_frame_sequence.append(plot_frame_resized)

        # Calculate episode metrics
        avg_speed = speed_sum / step_count if step_count > 0 else 0
        total_arrived_reward = info["rewards"]["arrived_reward"]
        number_of_arrivals = total_arrived_reward / 0.25

        if not truncated:
            #final_agents_terminated = info["agents_terminated"]
            #number_of_collisions = sum(final_agents_terminated)
            final_agents_terminated = info["rewards"]["collision_reward"]
            number_of_collisions = final_agents_terminated / 0.25

        # Log episode results
        print(f"\nEpisode {eval.episode + 1}: Avg. Speed = {avg_speed:.2f}, Arrivals = {number_of_arrivals}, Collisions = {number_of_collisions}")

        # Update cumulative metrics
        metrics["Episode"].append(eval.episode + 1)
        metrics["Episode Length"].append(step_count)
        metrics["Average Speed"].append(avg_speed)
        metrics["Number of Arrivals"].append(number_of_arrivals)
        metrics["Number of Collisions"].append(number_of_collisions)
        
        total_speed_sum += speed_sum
        total_step_count += step_count
        total_arrivals += number_of_arrivals
        total_collisions += number_of_collisions

        
        print("!!!!!!!!!")
        print(eval.episode)

        # write the frames
        env_video_path_local = r'out\IntersectionEnv\DQNAgent\videos'
        env_video_path_local = os.path.join(env_video_path_local, f'rl-video-episode-{eval.episode}.mp4')
        frames_needed = get_video_duration(env_video_path_local)

        for _ in range(frames_needed):
            plot_frame_sequence.append(plot_frame_resized)


     # Compute global metrics
    global_avg_speed = total_speed_sum / total_step_count if total_step_count > 0 else 0
    avg_arrivals_per_episode = total_arrivals / NUM_EPISODES
    avg_collisions_per_episode = total_collisions / NUM_EPISODES
    avg_episode_length = total_step_count / NUM_EPISODES

    
    # Define the output CSV file path
    csv_file_path = os.path.join(video_dir, "global_metrics.csv")

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Check if the file is empty, if so, write the header
        if csv_file.tell() == 0:  
            csv_writer.writerow(["algorithm", "policy", "traffic", "avg_episode_length", "global_avg_speed", "avg_arrivals_per_episode", "avg_collisions_per_episode"])
        
        # Write the data (metrics)
        csv_writer.writerow([algorithm, policy, traffic, avg_episode_length, global_avg_speed, avg_arrivals_per_episode, avg_collisions_per_episode])

    print(f"Global metrics appended to {csv_file_path}")

    join_videos(NUM_EPISODES,env_video_path)

    save_video_with_ffmpeg(plot_frame_sequence, plot_video_path, fps, frame_size)




