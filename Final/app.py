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

sys.path.append(os.path.join(os.path.dirname(__file__), 'latex'))

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
            palette="plasma"  # Use a different colormap
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

def render_mermaid(mermaid_code, height=400, font_size="18px"):
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

            /* Modify the font size of the mermaid diagram */
            .mermaid {{
                font-size: {font_size}; /* Set the font size here */
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
        if line.startswith("###"):  
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

def render_shadow_box(content, flex_direction="row", font_size="18px"):
    st.markdown(
        f"""
        <div style="
            border-radius: 15px; 
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3); 
            margin: 20px auto; 
            padding: 20px; 
            background-color: #ffffff; 
            max-width: 95%; 
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: {flex_direction};
            font-size: {font_size};">
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initialize session state for video control
if "play_videos" not in st.session_state:
    st.session_state.play_videos = False
if "video_played" not in st.session_state:
    st.session_state.video_played = False

# Create session state for global metrics visibility
if "metrics_visibility" not in st.session_state:
    st.session_state.metrics_visibility = {}

with st.sidebar:
    selected_algorithm = st.radio("Please select an RL algorithm", ["Base Line","DQN", "PPO"],  index=1)

    st.markdown('<hr>', unsafe_allow_html=True)

    st.write("Please select an Agent Policy:")
    none_selected = st.checkbox("None", value=True)
    cnn_selected = st.checkbox("Cnn Policy", value=True)
    mlp_selected = st.checkbox("Mlp Policy", value=True)
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


tabs = st.tabs(["Introduction", "Training", "Simulation","Results & Conclusions"])

# First tab: Introduction
with tabs[0]:
    markdown_file = "latex/intro.md"

    sections = parse_markdown_sections(markdown_file)
    st.markdown("#### 1. Project Description")
    
    st.markdown("##### Ojective:")
    render_shadow_box(sections["#### Ojective:"])
    
    st.markdown("##### Focus On:")
    render_shadow_box(sections["#### Focus On:"],flex_direction="column")

    st.markdown("#### 2.Goals of the Simulation Project:")
    render_mermaid(simulation_goals_code,300)

    col1,col2=st.columns(2)
    with col1:
        st.markdown("#### 3.Models of Decision Support Considered:")
        render_shadow_box(sections["### 3.Models of Decision Support Considered"],flex_direction="row")

    with col2:
        st.markdown("#### 4. Model Characteristics:")
        render_shadow_box(sections["### 4. Model Characteristics:"],flex_direction="row")

    st.markdown("#### 5.Main Entities of the System:")
    render_mermaid(system_representation, height=500, font_size="24px")

    st.markdown("#### 6.System Variables:")
    render_mermaid(system_variables_code, height=500)

    st.markdown("#### 7.Operation Policy:")
    render_mermaid(operation_policy, height=400,font_size="24px")

    st.markdown("#### 8.Experimental Scenarios:")
    render_mermaid(system_code_scenarios, height=300)

    st.markdown("#### 9.Performance Metrics:")
    render_mermaid(system_code_metrics, height=250)



# Second tab: Training
with tabs[1]:
    st.markdown("##### Training Process - Sequential Diagram:")
   
    try:
        image = Image.open("images/Training_Process.png")
        
        image = image.resize((600, 400)) 
        
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


    #Social Attention TRAIN
    latex_file_path = "latex/social_attention_algorithm.tex" 
    latex_arrays = load_latex_arrays(latex_file_path)

    col1,col2,col3 =st.columns(3)
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


    [col1] =st.columns(1)
    col1.markdown(f'<div style="text-align: left;"><b>DQN Social Attention Train:</b></div>', unsafe_allow_html=True)

    col1, col2, col3,col4= st.columns(4)

    with col2:
       
        svg_plot = load_svg_file("images/social_episode_totalreward.svg")

        st.markdown(f'<div style="text-align: center;">Average Episode Reward:</div>', unsafe_allow_html=True)

        st.markdown(f'<div class=image-container style="text-align: center;">{svg_plot}</div>', unsafe_allow_html=True)

    
    with col3:
        png_image = Image.open("images/social_episode_reward3d.png")
        png_image = png_image.resize((600, 365))
        
        buffer = io.BytesIO()
        png_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        st.markdown(f'<div style="text-align: center;">Average Episode Reward (3D View):</div>', unsafe_allow_html=True)
        
        st.markdown(
            f'<div class=image-container style="text-align: center;"><img src="data:image/png;base64,{base64.b64encode(buffer.read()).decode()}" width="600"></div>',
            unsafe_allow_html=True
        )


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


#Social Influence
    import streamlit as st

    text = """

    ### Decentralized Training with Social Influence

    Each agent is represented by its own network (and therefore optimizer, during training). 
    to approximate our theoretical $Q$ function, which we use to pick the maximum value action for each state. 
    We assume that every $Q$ function obeys the Bellman Equation:
    """

    st.markdown(text)

    st.latex(r"Q^\pi(s, a) = r + \gamma Q^\pi(s', \pi(s'))")

    st.markdown("From this, we get the temporal difference error, \\( \delta \\), which we attempt to minimize:")

    st.latex(r"\delta = Q(s, a) - \left(r + \gamma \max_{a'} Q(s', a')\right)")

    st.markdown("""
    Loss is calculated from a batch of transitions:
    """)

    st.latex(r"L = \frac{1}{|B|} \sum_{(s, a, s', r) \in B} L(\delta)")

    st.markdown("where:")

    st.latex(r"""
    L(\delta) =
    \begin{cases} 
    \frac{1}{2}\delta^2 & \text{for } |\delta| \leq 1, \\
    |\delta| - \frac{1}{2} & \text{otherwise}.
    \end{cases}
    """)

    st.markdown("""
    where \\( (s, a, s', r) \\) is a single transition.

    We chose to implement what is known as the Basic Social Influence mechanism, 
    which shifts an agent's rewards using counterfactuals, so that it becomes:
    """)

    st.latex(r"r^k_t = \alpha e^k_t + \beta c^k_t")

    st.markdown("""
    where \\( e^k_t \\) is the extrinsic or environmental reward, and \\( c^k_t \\) is the causal 
    influence reward. Essentially, agent \\( k \\) asks the question: 
    *"How would \\( j \)'s action change if I had acted differently in this situation?"* 

    This causal influence reward is obtained by calculating the divergence between 
    the marginal policy of \\( j \\) (if \\( j \\) did not consider \\( k \\)) and the 
    conditional policy of \\( j \\) (when \\( j \\) does consider \\( k \\)).
    """)

    # Display the image separately
    st.image("images/social_influence.png", caption="Chain of social influence")

    st.image("images/dqn_SI_metrics.png", caption="Training of Social Influence Decentralized Agents")
    

# Third tab: Simulation
with tabs[2]:
    st.write("#### Exploring the performance of RL algorithms under different operating policies")
    
    global_metrics_file = "videos/global_metrics.csv"

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

                    if "none_metrics_visible" not in st.session_state.metrics_visibility:
                        st.session_state.metrics_visibility["none_metrics_visible"] = False

                    if st.button(f"Show Indicators - {selected_policies[0].upper()}", key=f"metrics_0"):
                        st.session_state.metrics_visibility["none_metrics_visible"] = not st.session_state.metrics_visibility["none_metrics_visible"]

                    # Show or hide metrics based on the state
                    if st.session_state.metrics_visibility["none_metrics_visible"]:
                        if os.path.exists(global_metrics_file):
                            df = pd.read_csv(global_metrics_file)
                            filtered_metrics = df[
                                (df['algorithm'] == "baseline") &
                                (df['policy'] == "none") &
                                (df['traffic'] == selection)
                            ]
                            if not filtered_metrics.empty:
                                avg_episode_length = filtered_metrics['avg_episode_length'].values[0]
                                global_avg_speed = filtered_metrics['global_avg_speed'].values[0]
                                avg_arrivals_per_episode = filtered_metrics['avg_arrivals_per_episode'].values[0]
                                avg_collisions_per_episode = filtered_metrics['avg_collisions_per_episode'].values[0]

                                st.write(f"### {selected_policies[0].upper()} - Indicators")
                                st.write(f"**Average Episode Length:** {avg_episode_length}")
                                st.write(f"**Global Average Speed:** {global_avg_speed:.2f}")
                                st.write(f"**Average Arrivals per Episode:** {avg_arrivals_per_episode}")
                                st.write(f"**Average Collisions per Episode:** {avg_collisions_per_episode}")
                            else:
                                st.warning(f"No global metrics found for {selected_algorithm.upper()} | {selected_policies[0].upper()} | {selection}.")
                        else:
                            st.warning("Global metrics CSV file is missing.")

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
            st.warning("Please select at least one policy.")
    else:
        st.info("Press 'Start Videos' in the sidebar to play the videos.")


    
# Last tab: Conclusions
with tabs[3]:
    st.title("Results")
    global_metrics_file = 'videos/global_metrics.csv'  

    df = pd.read_csv(global_metrics_file)
   
    styled_df = df.style.set_table_attributes('class="dataframe"') \
                        .format({"avg_episode_length": "{:.2f}", 
                                "global_avg_speed": "{:.2f}", 
                                "avg_arrivals_per_episode": "{:.2f}", 
                                "avg_collisions_per_episode": "{:.2f}"}) \
                        .set_properties(**{'text-align': 'center', 'font-size': '12px'})
    
    df_html = styled_df.to_html(index=False, escape=False)
    st.write("### Global Metrics: ")

    st.markdown(f"<div class='dataframe'>{df_html}</div>", unsafe_allow_html=True)

    metrics = ["avg_episode_length", "global_avg_speed", "avg_arrivals_per_episode", "avg_collisions_per_episode"]
    
    for traffic_type in df["traffic"].unique():
        st.write(f"### Metrics for {traffic_type.capitalize()} Traffic")
        plot_metrics(df, metrics, traffic_type)

    st.write("## Performance Indicators: ")


    import pandas as pd
    import streamlit as st

    # Load the data from the CSV file
    file_path = 'videos/global_metrics.csv'
    df = pd.read_csv(file_path)

    # Calculate the required metrics
    # 1. Calculate Average Speed per Arrived Vehicle (Efficiency)
    df['efficiency'] = df['global_avg_speed'] / df['avg_arrivals_per_episode']

    # 2. Calculate Collision Rate per 100 Steps (Safety)
    df['collision_rate_per_100_steps'] = (df['avg_collisions_per_episode'] / df['avg_episode_length']) * 100

    # 3. Calculate Efficiency-to-Safety Tradeoff Index
    df['tradeoff_index'] = df['efficiency'] / df['collision_rate_per_100_steps']

    # Split data into sparse and dense traffic
    df_sparse = df[df['traffic'] == 'sparse']
    df_dense = df[df['traffic'] == 'dense']

    # Function to highlight the best values for a column
    def highlight_best(s, ascending=True):
        is_best = s == (s.min() if ascending else s.max())
        return ['background-color: #FFA500;' if v else '' for v in is_best]

    # Create the styled dataframes for sparse traffic
    styled_sparse = df_sparse[['algorithm', 'policy', 'traffic', 'efficiency', 'collision_rate_per_100_steps', 'tradeoff_index']].style \
        .apply(highlight_best, subset=['efficiency', 'tradeoff_index'], ascending=False, axis=0) \
        .apply(highlight_best, subset=['collision_rate_per_100_steps'], ascending=True, axis=0) \
        .format({
            'efficiency': '{:.2f}',
            'collision_rate_per_100_steps': '{:.2f}',
            'tradeoff_index': '{:.2f}'
        }) \
        .set_properties(**{'text-align': 'center', 'font-size': '12px'}) \
        .set_table_attributes('class="dataframe"')

    # Create the styled dataframes for dense traffic
    styled_dense = df_dense[['algorithm', 'policy', 'traffic', 'efficiency', 'collision_rate_per_100_steps', 'tradeoff_index']].style \
        .apply(highlight_best, subset=['efficiency', 'tradeoff_index'], ascending=False, axis=0) \
        .apply(highlight_best, subset=['collision_rate_per_100_steps'], ascending=True, axis=0) \
        .format({
            'efficiency': '{:.2f}',
            'collision_rate_per_100_steps': '{:.2f}',
            'tradeoff_index': '{:.2f}'
        }) \
        .set_properties(**{'text-align': 'center', 'font-size': '12px'}) \
        .set_table_attributes('class="dataframe"')

    # Calculate Adaptability and Stress Safety Index by algorithm and policy
    # 5.2.3 Adaptability to Varying Traffic Conditions

    # Adaptability: Variance / Mean for each algorithm and policy
    adaptability = df.groupby(['algorithm', 'policy'])['avg_arrivals_per_episode'].agg(['var', 'mean'])
    adaptability['adaptability'] = adaptability['var'] / adaptability['mean']

    # Load Adaptability: Dense / Sparse for each algorithm and policy
    load_adaptability = df_dense.groupby(['algorithm', 'policy'])['avg_arrivals_per_episode'].mean() / df_sparse.groupby(['algorithm', 'policy'])['avg_arrivals_per_episode'].mean()

    # Stress Safety Index: Collision Rate (Dense) / Collision Rate (Sparse) for each algorithm and policy
    collision_rate_dense = df_dense.groupby(['algorithm', 'policy'])['collision_rate_per_100_steps'].mean()
    collision_rate_sparse = df_sparse.groupby(['algorithm', 'policy'])['collision_rate_per_100_steps'].mean()
    stress_safety_index = collision_rate_dense / collision_rate_sparse

    # Combine these metrics into a single DataFrame
    adaptability_table = pd.DataFrame({
        'Adaptability': adaptability['adaptability'],
        'Load Adaptability': load_adaptability,
        'Stress Safety Index': stress_safety_index
    })

    # Reset the index to remove multi-index and create a single-level index
    adaptability_table.reset_index(inplace=True)

    # Apply highlighting to the Adaptability, Load Adaptability, and Stress Safety Index columns
    styled_adaptability = adaptability_table.style \
        .apply(highlight_best, subset=['Adaptability'], ascending=True, axis=0) \
        .apply(highlight_best, subset=['Load Adaptability'], ascending=False, axis=0) \
        .apply(highlight_best, subset=['Stress Safety Index'], ascending=True, axis=0) \
        .format({
            'Adaptability': '{:.2f}',
            'Load Adaptability': '{:.2f}',
            'Stress Safety Index': '{:.2f}'
        }) \
        .set_properties(**{'text-align': 'center', 'font-size': '12px'}) \
        .set_table_attributes('class="dataframe"')

    # Convert styled DataFrames to HTML
    df_sparse_html = styled_sparse.to_html(index=False, escape=False)
    df_dense_html = styled_dense.to_html(index=False, escape=False)
    df_adaptability_html = styled_adaptability.to_html(index=False, escape=False)

    # Display in Streamlit
    st.write("### Efficiency / Security Analysis")

    # Create three columns for the formulas
    col1, col2, col3 = st.columns(3)

    # First column: Efficiency Formula
    with col1:
        st.latex(r'''
        \scriptsize
        \text{Efficiency} = \frac{\text{Average Speed}}{\text{Average Arrivals per Episode}}
        ''')

    # Second column: Collision Rate Formula
    with col2:
        st.latex(r'''
        \scriptsize
        \text{Collision Rate per 100 Steps} = \frac{\text{Average Collisions per Episode}}{\text{Average Episode Length}} \times 100
        ''')

    # Third column: Tradeoff Index Formula
    with col3:
        st.latex(r'''
        \scriptsize
        \text{Tradeoff Index} = \frac{\text{Average Speed per Arrived Vehicle}}{\text{Collision Rate per 100 Steps}}
        ''')

    # Create columns in Streamlit for displaying Sparse and Dense traffic data
    col1, col2 = st.columns(2)

    with col1:
        # Display Sparse Traffic Dataframe
        st.write("##### Sparse Traffic Metrics:")
        st.markdown(df_sparse_html, unsafe_allow_html=True)

    with col2:
        # Display Dense Traffic Dataframe
        st.write("##### Dense Traffic Metrics:")
        st.markdown(df_dense_html, unsafe_allow_html=True)

    # Create a column for Adaptability, Load Adaptability, and Stress Safety Index
    st.write("#### Adaptability to Varying Traffic Conditions:")
    col1, col2, col3 = st.columns(3)

    # First column: Relative Variance in Throughput (Adaptability)
    with col1:
        st.latex(r'''
        \scriptsize
        \text{Adaptability} = \frac{\text{Variance}(\text{Average Arrivals per Episode})}{\text{Mean}(\text{Average Arrivals per Episode})}
        ''')

    # Second column: Traffic Load Adaptability
    with col2:
        st.latex(r'''
        \scriptsize
        \text{Load Adaptability} = \frac{\text{Average Arrivals per Episode (Dense Traffic)}}{\text{Average Arrivals per Episode(Sparse Traffic)}}
        ''')

    # Third column: Stress Safety Index
    with col3:
        st.latex(r'''
        \scriptsize
        \text{Stress Safety Index} = \frac{\text{Collision Rate (Dense Traffic)}}{\text{Collision Rate (Sparse Traffic)}}
        ''')

    st.markdown(df_adaptability_html, unsafe_allow_html=True)


    import pandas as pd
    import numpy as np
    import streamlit as st

    # Data
    sparse_metrics = pd.DataFrame({
        'algorithm': ['baseline', 'dqn', 'dqn', 'ppo', 'ppo', 'dqn'],
        'policy': ['none', 'cnn', 'mlp', 'cnn', 'mlp', 'social attention'],
        'traffic': ['sparse'] * 6,
        'efficiency': [3.25, 2.03, 1.30, 1.73, 1.54, 2.79],
        'collision_rate_per_100_steps': [16.63, 4.23, 4.47, 7.48, 6.24, 9.63],
        'tradeoff_index': [0.20, 0.48, 0.29, 0.23, 0.25, 0.29]
    })

    dense_metrics = pd.DataFrame({
        'algorithm': ['baseline', 'dqn', 'dqn', 'ppo', 'ppo', 'dqn'],
        'policy': ['none', 'cnn', 'mlp', 'cnn', 'mlp', 'social attention'],
        'traffic': ['dense'] * 6,
        'efficiency': [4.83, 3.71, 1.98, 3.30, 1.86, 5.43],
        'collision_rate_per_100_steps': [20.83, 11.29, 10.95, 15.00, 8.74, 17.50],
        'tradeoff_index': [0.23, 0.33, 0.18, 0.22, 0.21, 0.31]
    })

    adaptability_metrics = pd.DataFrame({
        'algorithm': ['baseline', 'dqn', 'dqn', 'dqn', 'ppo', 'ppo'],
        'policy': ['none', 'cnn', 'mlp', 'social attention', 'cnn', 'mlp'],
        'Adaptability': [0.11, 0.24, 0.12, 0.26, 0.20, 0.05],
        'Load Adaptability': [0.67, 0.60, 0.70, 0.54, 0.59, 0.80],
        'Stress Safety Index': [1.25, 2.67, 2.45, 1.82, 2.01, 1.40]
    })

    # Combine all dataframes
    sparse_metrics = sparse_metrics.rename(columns={
        'efficiency': 'efficiency_sparse',
        'collision_rate_per_100_steps': 'collision_rate_sparse',
        'tradeoff_index': 'tradeoff_sparse'
    })
    dense_metrics = dense_metrics.rename(columns={
        'efficiency': 'efficiency_dense',
        'collision_rate_per_100_steps': 'collision_rate_dense',
        'tradeoff_index': 'tradeoff_dense'
    })

    combined_df = pd.merge(sparse_metrics, dense_metrics, on=['algorithm', 'policy'])
    combined_df = pd.merge(combined_df, adaptability_metrics, on=['algorithm', 'policy'])

    # Define metric classification function
    def classify_metric(metric_values, ascending=True):
        quantiles = np.percentile(metric_values, [33, 67])
        if ascending:
            # Lower is better
            conditions = [
                metric_values <= quantiles[0],
                (metric_values > quantiles[0]) & (metric_values <= quantiles[1]),
                metric_values > quantiles[1],
            ]
        else:
            # Higher is better
            conditions = [
                metric_values >= quantiles[1],
                (metric_values > quantiles[0]) & (metric_values < quantiles[1]),
                metric_values <= quantiles[0],
            ]
        categories = ["High", "Medium", "Low"]
        return np.select(conditions, categories, default="Unknown")

    # Apply classification for each metric
    combined_df["efficiency_category"] = classify_metric(combined_df["efficiency_sparse"], ascending=False)
    combined_df["safety_category"] = classify_metric(combined_df["collision_rate_sparse"], ascending=True)
    combined_df["adaptability_category"] = classify_metric(combined_df["Adaptability"], ascending=True)

    # Create a styled dataframe
    def highlight_category(val):
        if val == "High":
            return "background-color: #f8d7da; color: black;"  # Light red
            
        elif val == "Medium":
            return "background-color: #d4edda; color: black;"  # Light green
            
        elif val == "Low":
            return "background-color: #fff3cd; color: black;"  # Light yellow
        return ""

    styled_combined = combined_df[[
        "algorithm", "policy", "efficiency_category", "safety_category", "adaptability_category"
        ]].style \
            .map(highlight_category, subset=["efficiency_category", "safety_category", "adaptability_category"]) \
            .set_properties(**{'text-align': 'center', 'font-size': '12px'}) \
            .set_table_attributes('class="dataframe"')



    # Streamlit display
    st.write("### Algorithm Classification Table")
    st.write("These indicators are classified using **quantiles (33%, 67%)** to categorize them into High, Medium, and Low categories.")  
 
    st.markdown(styled_combined.to_html(index=False), unsafe_allow_html=True)



























        
   

    




   
  









