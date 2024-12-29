import streamlit as st
import streamlit.components.v1 as components

# Define the fixed flowchart code
flowchart_code = """
graph LR
    A[Agent] -->|Send Current State| T[Trainer]
    T -->|Calculate Action Based on Policy| A
    A -->|Perform Action| R[Reward System]
    R -->|Return Reward| A
    A -->|Update Policy Based on Reward| T
    T -->|Evaluate Performance| T

"""

# Build the Mermaid HTML content
mermaid_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/9.3.0/mermaid.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', () => {{
            try {{
                mermaid.initialize({{ startOnLoad: true }});
            }} catch (error) {{
                console.error("Mermaid initialization error:", error);
            }}
        }});
    </script>
</head>
<body>
    <div class="mermaid">
    {flowchart_code}
    </div>
</body>
</html>
"""

# Use Streamlit tabs
tabs = st.tabs(["Tab 1", "Tab 2", "Diagram Tab", "Tab 4"])

with tabs[2]:  # The tab where the flowchart is rendered
    try:
        components.html(mermaid_html, height=500, scrolling=True)
    except Exception as e:
        st.error(f"Error rendering flowchart: {e}")






