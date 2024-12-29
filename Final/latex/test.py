import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="RL Highway-Env",  
    page_icon="🌟",      
    layout="wide"        
)

# Define Mermaid diagram code for Experimental Scenarios
system_code_scenarios = """
flowchart TD
    subgraph A[Experimental Scenarios]
        B[High Traffic Density]
        C[Low Traffic Density]

        %% Descriptions for Experimental Scenarios
        B1[Increased vehicles entering intersection]
        C1[Sparse vehicle presence]

        B --> B1
        C --> C1
    end
"""

# Define Mermaid diagram code for Performance Metrics
system_code_metrics = """
flowchart TD
    subgraph H[Performance Metrics]
        I[Efficiency]
        I1[Average wait time]
        I2[Throughput]
        I3[Average speed]

        J[Safety]
        J1[Collision rates]
        J2[Near-misses]
        J3[Compliance with traffic rules]

        K[Adaptability]
        K1[Generalizing learned behaviors]
        K2[Adaptability to new traffic scenarios]

        %% Linking Metrics to Details
        I --> I1
        I --> I2
        I --> I3
        J --> J1
        J --> J2
        J --> J3
        K --> K1
        K --> K2
    end
"""

# HTML template to include Mermaid.js and render the diagrams
html_code_scenarios = f"""
  <html>
    <head>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{startOnLoad:true}});
      </script>
    </head>
    <body>
      <div class="mermaid">
        {mermaid_code_scenarios}
      </div>
    </body>
  </html>
"""

html_code_metrics = f"""
  <html>
    <head>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{startOnLoad:true}});
      </script>
    </head>
    <body>
      <div class="mermaid">
        {mermaid_code_metrics}
      </div>
    </body>
  </html>
"""

# Use Streamlit to render the HTML for each graph
st.subheader("Experimental Scenarios")
components.html(html_code_scenarios, height=400)

st.subheader("Performance Metrics")
components.html(html_code_metrics, height=600)




