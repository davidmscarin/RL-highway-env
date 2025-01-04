import streamlit as st
import os
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit.components.v1 as components
import sys
from PIL import Image
import io

st.set_page_config(
    page_title="RL Highway-Env",  
    page_icon="🌟",      
    layout="wide"        
)

# Add the 'latex' subdirectory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'latex'))

# Now you can import the mermaid module from the 'latex' directory
from latex.mermaid import decision_flow_code, training_sequence_code, causal_loop_code, \
    interaction_diagram_code, entity_relationship_code, simulation_goals_code, \
    system_variables_code,system_representation, main_entities, operation_policy, system_code_metrics,system_code_scenarios


# Read the external HTML file
with open("style.html", "r") as file:
    styles = file.read()

st.markdown(styles, unsafe_allow_html=True)


def load_latex_arrays(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Use a regular expression to capture all blocks between \begin{array} and \end{array}
    arrays = []
    pattern = r"\\begin{array}{l}(.*?)\\end{array}"
    
    # Find all matches for the LaTeX arrays
    matches = re.findall(pattern, content, re.DOTALL)  # DOTALL allows matching across multiple lines

    for match in matches:
        # Add the r""" at the start and """ at the end for correct format in Streamlit
        array_content = r""" \begin{array}{l}""" + match + r"""\end{array} """
        arrays.append(array_content)
    
    return arrays



def load_svg_file(file_path):
    with open(file_path, "r") as file:
        return file.read()
            

# Function to plot metrics for each traffic type
def plot_metrics(df, metrics, traffic_type):
    plots_dir = "global_metrics_plots"
    os.makedirs(plots_dir, exist_ok=True)  
    df_filtered = df[df['traffic'] == traffic_type]

    plot_filenames = []

    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.barplot(
            data=df_filtered, 
            x="policy", 
            y=metric, 
            hue="algorithm", 
            ax=ax,
            palette="muted"
        )

        # Set titles and labels with styling
        ax.set_title(metric.replace("_", " ").capitalize(), fontsize=14, color="#333333", pad=10)
        ax.set_xlabel("Policy", fontsize=12, color="#333333")
        ax.set_ylabel(metric.replace("_", " ").capitalize(), fontsize=12, color="#333333")

        # Customize legend
        ax.legend(title="Algorithm", fontsize=10, title_fontsize=12, frameon=True, 
                  loc='upper left', bbox_to_anchor=(1, 1))

        # Customize axes spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

        plot_filename = os.path.join(plots_dir, f"plot_{metric}_{traffic_type}.png")
        plot_filenames.append(plot_filename)
        fig.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close(fig) 

    num_plots = len(plot_filenames)
    
    for i in range(0, num_plots, 4):  
        cols = st.columns(4) 
        for j, col in enumerate(cols):
            if i + j < num_plots:
                plot_filename = plot_filenames[i + j]
                col.markdown(
                    f"""
                    <div class="image-container">
                        <img src="data:image/png;base64,{get_base64_image(plot_filename)}" alt="Plot" width="100%">
                    </div>
                    """, 
                    unsafe_allow_html=True
                )


# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def render_mermaid(mermaid_code, height=400):
    mermaid_html = f"""
    <html>
    <head>
        <style>
            /* Custom box style */
                    .shadow-box {{
                        border-radius: 15px;
                        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
                        margin: 20px auto; 
                        padding: 20px;
                        background-color: #ffffff; 
                        max-width: 95%; 
                        text-align: center;
                    }}

                    .shadow-box img {{
                        width: 100%; /* Adjust image size */
                        margin: 0 auto;
                    }}
            
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/9.4.3/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </head>
    <body>
        <div class="shadow-box">
            <div class="mermaid">
                {mermaid_code}
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(mermaid_html, height=height)


# Function to load specific items from the markdown file
def parse_markdown_sections(file_path):
    """
    Parse the markdown file into sections based on `###` headers.
    Returns a dictionary mapping section headers to their content.
    """
    with open(file_path, "r") as file:
        content = file.read()

    sections = {}
    current_section = None
    current_content = []

    for line in content.splitlines():
        if line.startswith("###"):  # Found a section header
            if current_section:
                sections[current_section] = "\n".join(current_content)
            current_section = line.strip()
            current_content = []
        else:
            current_content.append(line)

    # Add the last section
    if current_section:
        sections[current_section] = "\n".join(current_content)

    return sections

# Initialize session state for video control
if "play_videos" not in st.session_state:
    st.session_state.play_videos = False
if "video_played" not in st.session_state:
    st.session_state.video_played = False

# Create session state for global metrics visibility
if "metrics_visibility" not in st.session_state:
    st.session_state.metrics_visibility = {}

with st.sidebar:
    selected_algorithm = st.radio("Please select an RL algorithm", ["Base Line","DQN", "PPO"])

    st.markdown('<hr>', unsafe_allow_html=True)

    st.write("Please select an Agent Policy:")
    none_selected = st.checkbox("None", value=True)
    cnn_selected = st.checkbox("Cnn Policy", value=False)
    mlp_selected = st.checkbox("Mlp Policy", value=False)
    social_attention_selected = st.checkbox("Social Attention", value=False)

    st.markdown('<hr>', unsafe_allow_html=True)

    st.write("Please select one Operation Policy:")
    selection = st.selectbox("Traffic Density:", ["sparse", "dense"])

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Start Videos"):
        st.session_state.play_videos = True
        st.session_state.video_played = False  

with col2:
    if st.button("Stop Videos"):
        st.session_state.play_videos = False
        st.session_state.video_played = False  


tabs = st.tabs(["Introduction", "Training", "Simulation","Conclusions"])

# First tab: Introduction
with tabs[0]:
    markdown_file = "latex/intro.md"

    # Parse the markdown file into sections
    sections = parse_markdown_sections(markdown_file)
    st.markdown("#### 1. Project Description")
    st.markdown(sections["### 1. Project Description"])
    st.markdown("##### Ojective:")
    st.markdown(sections["#### Ojective:"])
    st.markdown("##### Focus On:")
    st.markdown(sections["#### Focus On:"])

    st.markdown("#### 2.Goals of the Simulation Project:")
    render_mermaid(simulation_goals_code,300)

    col1,col2=st.columns(2)
    with col1:
        st.markdown("#### 3.Models of Decision Support Considered:")
        st.markdown(sections["### 3.Models of Decision Support Considered"])

    with col2:
        st.markdown("#### 4. Model Characteristics:")
        st.markdown(sections["### 4. Model Characteristics:"])

    st.markdown("#### 5.Main Entities of the System:")
    render_mermaid(system_representation,500)

    st.markdown("#### 6.System Variables:")
    render_mermaid(system_variables_code,500)

    st.markdown("#### 7.Operation Policy:")
    render_mermaid(operation_policy,400)

    st.markdown("#### 8.Experimental Scenarios:")
    render_mermaid(system_code_scenarios,300)

    st.markdown("#### 9.Performance Metrics:")
    render_mermaid(system_code_metrics,250)



# Second tab: Training
with tabs[1]:
    st.markdown("##### Training Process - Sequential Diagram:")
   
    try:
        # Load and display the PNG image
        image = Image.open("images/Training_Process.png")
        
        # Resize the image (optional, you can adjust the size)
        image = image.resize((600, 400))  # Resize image to 600x400 pixels, adjust as needed
        
        # Save the image to a BytesIO buffer to get the base64 string for embedding
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Convert the image to a base64 string
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        # Display the image with a caption and center it using custom HTML
        st.markdown(
             f"""
            <html>
            <head>
                <style>
                    /* Custom box style */
                    .shadow-box {{
                        border-radius: 15px;
                        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
                        margin: 20px auto; 
                        padding: 20px;
                        background-color: #f9f9f9; 
                        max-width: 40%; /* Container width is 60% */
                        text-align: center;
                    }}

                    /* Center image within the box */
                    .shadow-box img {{
                        width: 100%; /* Adjust image size */
                        margin: 0 auto;
                    }}
                </style>
            </head>
            <body>
                <div class="shadow-box">
                    <img src="data:image/png;base64,{img_base64}" alt="Sequence Diagram"/>
                </div>
            </body>
            </html>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Error loading or displaying the image: {e}")


    st.markdown("##### Deep Reinforcement Learning:")
    latex_file_path = "latex/dqn_algorithm.tex" 
    latex_arrays = load_latex_arrays(latex_file_path)

    col1,col2,col3 =st.columns(3)
    with col1: 
        
        display_order = [3] 
        for index in display_order:
            if index < len(latex_arrays):
                try:
                    st.latex(latex_arrays[index])
                except Exception as e:
                    st.error(f"Error rendering LaTeX at index {index}: {str(e)}")
            else:
                st.error(f"Invalid index: {index}. Only {len(latex_arrays)} LaTeX arrays are available.")
    with col2:
        display_order = [1] 
        for index in display_order:
            if index < len(latex_arrays):
                try:
                    st.latex(latex_arrays[index])
                except Exception as e:
                    st.error(f"Error rendering LaTeX at index {index}: {str(e)}")
            else:
                st.error(f"Invalid index: {index}. Only {len(latex_arrays)} LaTeX arrays are available.")

    with col3:
        display_order = [2] 
        for index in display_order:
            if index < len(latex_arrays):
                try:
                    st.latex(latex_arrays[index])
                except Exception as e:
                    st.error(f"Error rendering LaTeX at index {index}: {str(e)}")
            else:
                st.error(f"Invalid index: {index}. Only {len(latex_arrays)} LaTeX arrays are available.")
        
    col1, col2=st.columns(2)
    with col1:
        st.markdown(f'<div style="text-align: left;"><b>DQN CnnPolicy Train:</b></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="text-align: left;"><b>DQN MlpPolicy Train:</b></div>', unsafe_allow_html=True)

    #DQN TRAIN
    col1,col2,col3, col4 =st.columns(4)
    with col1:
       
        svg_plot = load_svg_file("images/dqn_cnn_ep_rew_mean.svg")

        st.markdown(f'<div style="text-align: center;">Average Episode Reward:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)

    with col2:

        svg_plot = load_svg_file("images/dqn_cnn_train_loss.svg")

        st.markdown(f'<div style="text-align: center;">Train Loss:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)


    with col3:
 
        svg_plot = load_svg_file("images/dqn_mlp_ep_rew_mean.svg")

        st.markdown(f'<div style="text-align: center;">Average Episode Reward:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)

    with col4:

        svg_plot = load_svg_file("images/dqn_mlp_train_loss.svg")

        st.markdown(f'<div style="text-align: center;">Train Loss:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)


    #PPO TRAIN
    latex_file_path = "latex/ppo_algorithm.tex" 
    latex_arrays = load_latex_arrays(latex_file_path)

    col1,col2,col3 =st.columns(3)
    with col1: 
        display_order = [1] 
        for index in display_order:
            if index < len(latex_arrays):
                try:
                    st.latex(latex_arrays[index])
                except Exception as e:
                    st.error(f"Error rendering LaTeX at index {index}: {str(e)}")
            else:
                st.error(f"Invalid index: {index}. Only {len(latex_arrays)} LaTeX arrays are available.")
        
    with col2: 
        display_order = [0] 
        for index in display_order:
            if index < len(latex_arrays):
                try:
                    st.latex(latex_arrays[index])
                except Exception as e:
                    st.error(f"Error rendering LaTeX at index {index}: {str(e)}")
            else:
                st.error(f"Invalid index: {index}. Only {len(latex_arrays)} LaTeX arrays are available.")


    col1, col2=st.columns(2)
    with col1:
        st.markdown(f'<div style="text-align: left;"><b>PPO CnnPolicy Train:</b></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="text-align: left;"><b>PPO MlpPolicy Train:</b></div>', unsafe_allow_html=True)


    col1,col2,col3, col4 =st.columns(4)
    with col1:
       
        svg_plot = load_svg_file("images/ppo_cnn_ep_rew_mean.svg")

        st.markdown(f'<div style="text-align: center;">Average Episode Reward:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)

    with col2:

        svg_plot = load_svg_file("images/ppo_cnn_train_loss.svg")

        st.markdown(f'<div style="text-align: center;">Train Loss:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)


    with col3:
 
        svg_plot = load_svg_file("images/ppo_mlp_ep_rew_mean.svg")

        st.markdown(f'<div style="text-align: center;">Average Episode Reward:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)

    with col4:

        svg_plot = load_svg_file("images/ppo_mlp_train_loss.svg")

        st.markdown(f'<div style="text-align: center;">Train Loss:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)


    #Social Attention TRAIN
    latex_file_path = "latex/social_attention_algorithm.tex" 
    latex_arrays = load_latex_arrays(latex_file_path)

    col1,col2 =st.columns(2)
    with col1: 
        display_order = [0] 
        for index in display_order:
            if index < len(latex_arrays):
                try:
                    st.latex(latex_arrays[index])
                except Exception as e:
                    st.error(f"Error rendering LaTeX at index {index}: {str(e)}")
            else:
                st.error(f"Invalid index: {index}. Only {len(latex_arrays)} LaTeX arrays are available.")


    [col1] =st.columns(1)
    col1.markdown(f'<div style="text-align: left;"><b>DQN Social Attention Train:</b></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
       
        svg_plot = load_svg_file("images/social_episode_totalreward.svg")

        st.markdown(f'<div style="text-align: center;">Average Episode Reward:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)

    
    with col2:
        # Load and resize the PNG image
        png_image = Image.open("images/social_episode_reward3d.png")
        png_image = png_image.resize((600, 365))
        
        # Save the image to a BytesIO buffer
        buffer = io.BytesIO()
        png_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        st.markdown(f'<div style="text-align: center;">Average Episode Reward (3D View):</div>', unsafe_allow_html=True)
        
        st.markdown(
            f'<div class=image-container style="text-align: center;"><img src="data:image/png;base64,{base64.b64encode(buffer.read()).decode()}" width="600"></div>',
            unsafe_allow_html=True
        )



# Third tab: Simulation
with tabs[2]:
    st.write("#### Exploring the performance of RL algorithms under different operating policies")

    if selection and st.session_state.play_videos:
        if selected_algorithm == "Base Line":
            selected_algorithm = "baseline"
        if selected_algorithm == "DQN":
            selected_algorithm = "dqn"
        elif selected_algorithm == "PPO":
            selected_algorithm = "ppo"
        
        selected_policies = []
        if none_selected:
            selected_policies.append("none")
        if cnn_selected:
            selected_policies.append("cnn")
        if mlp_selected:
            selected_policies.append("mlp")
        if social_attention_selected:
            selected_policies.append("social attention")
        
        if selected_algorithm and selected_policies:
            st.write(f"#### Algorithm: {selected_algorithm.upper()}     Traffic Density: {selection}")
            video_dir = "videos/"
            
            num_columns = 4
            columns = st.columns(num_columns)  
            
            # Always display baseline videos in the first column
            baseline_env_video_path = os.path.join(video_dir, f"environment_baseline_none_{selection}.webm")
            baseline_table_video_path = os.path.join(video_dir, f"speed_baseline_none_{selection}.webm")


            with columns[0]:
                if os.path.exists(baseline_env_video_path) and os.path.exists(baseline_table_video_path):
                    def get_base64_video(video_path):
                        with open(video_path, "rb") as file:
                            video_bytes = file.read()
                            return base64.b64encode(video_bytes).decode("utf-8")

                    baseline_env_video_data = get_base64_video(baseline_env_video_path)
                    baseline_table_video_data = get_base64_video(baseline_table_video_path)

                    st.markdown(
                        """
                        <div style="text-align: center; font-size: 14px; font-weight: bold; margin-bottom: 10px;">
                            BASELINE - NONE
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"""
                        <div class="video-container">
                            <video autoplay>
                                <source src="data:video/webm;base64,{baseline_env_video_data}" type="video/webm">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div class="video-container">
                            <video autoplay muted>
                                <source src="data:video/webm;base64,{baseline_table_video_data}" type="video/webm">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"Baseline videos are missing for Traffic Density: {selection}.")

            # Display videos for the selected algorithm and policies
            for idx, policy in enumerate(selected_policies): 
                env_video_path = os.path.join(video_dir, f"environment_{selected_algorithm}_{policy}_{selection}.webm")
                table_video_path = os.path.join(video_dir, f"speed_{selected_algorithm}_{policy}_{selection}.webm")
                global_metrics_file = "videos/global_metrics.csv"

                if os.path.exists(env_video_path) and os.path.exists(table_video_path):
                    with columns[idx]:
                        def get_base64_video(video_path):
                            with open(video_path, "rb") as file:
                                video_bytes = file.read()
                                return base64.b64encode(video_bytes).decode("utf-8")

                        env_video_data = get_base64_video(env_video_path)
                        table_video_data = get_base64_video(table_video_path)

                        st.markdown(
                            f"""
                            <div style="text-align: center; font-size: 14px; font-weight: bold; margin-bottom: 10px;">
                                {policy.upper()}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="video-container">
                                <video autoplay>
                                    <source src="data:video/webm;base64,{env_video_data}" type="video/webm">
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"""
                            <div class="video-container">
                                <video autoplay muted>
                                    <source src="data:video/webm;base64,{table_video_data}" type="video/webm">
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if f"{policy}_metrics_visible" not in st.session_state.metrics_visibility:
                            st.session_state.metrics_visibility[f"{policy}_metrics_visible"] = False

                        if st.button(f"Show Indicators - {policy.upper()}", key=f"metrics_{idx}"):
                            st.session_state.metrics_visibility[f"{policy}_metrics_visible"] = not st.session_state.metrics_visibility[f"{policy}_metrics_visible"]

                        # Show or hide metrics based on the state
                        if st.session_state.metrics_visibility[f"{policy}_metrics_visible"]:
                            if os.path.exists(global_metrics_file):
                                df = pd.read_csv(global_metrics_file)
                                filtered_metrics = df[
                                    (df['algorithm'] == selected_algorithm) &
                                    (df['policy'] == policy) &
                                    (df['traffic'] == selection)
                                ]
                                if not filtered_metrics.empty:
                                    avg_episode_length = filtered_metrics['avg_episode_length'].values[0]
                                    global_avg_speed = filtered_metrics['global_avg_speed'].values[0]
                                    avg_arrivals_per_episode = filtered_metrics['avg_arrivals_per_episode'].values[0]
                                    avg_collisions_per_episode = filtered_metrics['avg_collisions_per_episode'].values[0]

                                    st.write(f"### {policy.upper()} - Indicators")
                                    st.write(f"**Average Episode Length:** {avg_episode_length}")
                                    st.write(f"**Global Average Speed:** {global_avg_speed:.2f}")
                                    st.write(f"**Average Arrivals per Episode:** {avg_arrivals_per_episode}")
                                    st.write(f"**Average Collisions per Episode:** {avg_collisions_per_episode}")
                                else:
                                    st.warning(f"No global metrics found for {selected_algorithm.upper()} | {policy.upper()} | {selection}.")
                            else:
                                st.warning("Global metrics CSV file is missing.")
                else:
                    st.warning(f"Some videos are missing for Algorithm: {selected_algorithm.upper()} and Policy: {policy.upper()}. Please make sure all environment and table videos for {selection} are generated.")
        else:
            st.warning("Please select at least one policy.")
    else:
        st.info("Press 'Start Videos' in the sidebar to play the videos.")


    
# Last tab: Conclusions
with tabs[3]:
    st.title("Conclusions")
    global_metrics_file = 'videos/global_metrics.csv'  

    df = pd.read_csv(global_metrics_file)
   
    styled_df = df.style.set_table_attributes('class="dataframe"') \
                        .format({"avg_episode_length": "{:.2f}", 
                                "global_avg_speed": "{:.2f}", 
                                "avg_arrivals_per_episode": "{:.2f}", 
                                "avg_collisions_per_episode": "{:.2f}"}) \
                        .set_properties(**{'text-align': 'center', 'font-size': '12px'})
    
    df_html = styled_df.to_html(index=False, escape=False)
    st.write("### Global Indicators: ")

    st.markdown(f"<div class='dataframe'>{df_html}</div>", unsafe_allow_html=True)

    metrics = ["avg_episode_length", "global_avg_speed", "avg_arrivals_per_episode", "avg_collisions_per_episode"]
    
    for traffic_type in df["traffic"].unique():
        st.write(f"### Metrics for {traffic_type.capitalize()} Traffic")
        plot_metrics(df, metrics, traffic_type)








        
   

    




   
  









