import streamlit as st
import streamlit.components.v1 as components

# Define the diagrams in Mermaid syntax

# 1. Agent Behavior - Decision Flow
decision_flow_code = """
graph TD
    A[Start] --> B[Observe Traffic Conditions]
    B --> C{Collision Risk?}
    C -->|Yes| D[Slow Down]
    C -->|No| E[Maintain Speed]
    D --> F[Continue Driving]
    E --> F
    F --> G[End]
"""

# 2. Agent Training Process - Sequence Diagram
training_sequence_code = """
sequenceDiagram
    participant A as Agent
    participant T as Trainer
    participant R as Reward System
    A->>T: Send Current State
    T->>A: Calculate Action Based on Policy
    A->>R: Perform Action
    R->>A: Return Reward
    A->>T: Update Policy Based on Reward
    T->>T: Evaluate Performance
"""

# 3. System Variables and Feedback Loops - Causal Loop Diagram
causal_loop_code = """
flowchart TD
    A[Traffic Flow] --> B[Collision Rate]
    B --> C[Safety Policy]
    C --> A
    A --> D[System Throughput]
    D --> B
    C --> D
"""

# 4. Interaction Between Multiple Agents - Interaction Diagram
interaction_diagram_code = """
graph TD
    A[Vehicle 1] -->|Move Forward| B[Intersection]
    B --> C[Vehicle 2]
    C -->|Yield to Vehicle 1| D[Vehicle 1 Passes]
    D --> E[Vehicle 2 Moves]
    E --> B
"""

# 5. System Components - Entity Relationship Diagram
entity_relationship_code = """
erDiagram
    VEHICLE {
        string id
        float speed
        string position
        string direction
        string currentLane
    }
    INTERSECTION {
        string id
        int numLanes
        string layout
        int laneCapacity
    }
    TRAFFIC_FLOW {
        float arrivalRate
        float trafficDensity
        string vehicleTypes
    }
    AGENT_POLICY {
        string learnedPolicy
        string decisionRules
        string rewardFunction
    }

    VEHICLE ||--o| AGENT_POLICY: assigned
    VEHICLE ||--o| INTERSECTION: interactsWith
    VEHICLE ||--o| TRAFFIC_FLOW: contributesTo
    TRAFFIC_FLOW ||--o| INTERSECTION: configures
"""

simulation_goals_code = """
flowchart TD
    A[Goals of the Simulation Project] --> B[1. Successfully Training the Agents]
    B --> C[Apply DRL algorithms to train autonomous agents]
    A --> D[2. Introducing System Perturbations]
    D --> E[Systematically vary key factors: Traffic flow and agent policies]
    A --> F[3. Evaluating System Scalability and Robustness]
    F --> G[Evaluate scalability and robustness of MAS under extreme conditions]
"""

main_entities_code = """
graph TD
    %% Main Entities
    subgraph Main_Entities
        AV["Autonomous Vehicles (Agents)"]
        Intersection["Intersection (Road Infrastructure)"]
        Traffic_Flow["Traffic Flow"]
        Agent_Policies["Agent Policies (Decision-Making Models)"]
    end

    %% Attributes and Resources of Main Entities
    subgraph Attributes_Resources
        AV_Attributes["Attributes: Speed, Position, Direction, Assigned Policy, Current Lane"]
        AV_Resources["Resources: Road Space, Crossing Priority, Lanes"]
        Intersection_Attributes["Attributes: Number of Lanes, Intersection Layout"]
        Intersection_Resources["Resources: Lane Capacity"]
        Traffic_Flow_Attributes["Attributes: Arrival Rate, Traffic Density, Vehicle Types"]
        Traffic_Flow_Resources["Resources: Access to Intersection, Road Segments"]
        Agent_Policies_Attributes["Attributes: Learned Policy, Decision Rules, Reward Function"]
        Agent_Policies_Resources["Resources: Computational Resources, Control over Behavior"]
    end

    %% Entity Relationships
    AV -->|Attributes| AV_Attributes
    AV -->|Competes for| AV_Resources
    Intersection -->|Attributes| Intersection_Attributes
    Intersection -->|Competes for| Intersection_Resources
    Traffic_Flow -->|Attributes| Traffic_Flow_Attributes
    Traffic_Flow -->|Competes for| Traffic_Flow_Resources
    Agent_Policies -->|Attributes| Agent_Policies_Attributes
    Agent_Policies -->|Competes for| Agent_Policies_Resources

    %% Interaction between entities
    AV -->|Interacts with| Intersection
    AV -->|Influences| Traffic_Flow
    AV -->|Guided by| Agent_Policies
    Traffic_Flow -->|Influences| Intersection
    Traffic_Flow -->|Affected by| Agent_Policies
    Intersection -->|Affects| Traffic_Flow
    Agent_Policies -->|Affects| AV
"""

system_variables_code = """
    graph TD
        %% System Variables
        subgraph Exogenous_Variables
            Non_Controllable["Non-Controllable Variables"]
            Traffic_Density["Traffic Density"]
            Intersection_Config["Intersection Configuration (Lane Layout)"]
            Controllable["Controllable Variables"]
            Controlled_Vehicles["Number of Controlled Vehicles"]
            Agent_Policy["Agent's Policy"]
        end

        subgraph Endogenous_Variables
            Average_Speed["Average Speed"]
            Collision_Rate["Collision Rate"]
            Throughput["System Throughput"]
        end

        %% Exogenous Variables - Non-Controllable
        Non_Controllable --> Traffic_Density
        Non_Controllable --> Intersection_Config

        %% Exogenous Variables - Controllable
        Controllable --> Controlled_Vehicles
        Controllable --> Agent_Policy

        %% Endogenous Variables
        Average_Speed --> Endogenous_Variables
        Collision_Rate --> Endogenous_Variables
        Throughput --> Endogenous_Variables

        %% Relationships between variables
        Traffic_Density -->|Affects| Average_Speed
        Traffic_Density -->|Affects| Collision_Rate
        Intersection_Config -->|Affects| Average_Speed
        Intersection_Config -->|Affects| Collision_Rate
        Controlled_Vehicles -->|Affects| Throughput
        Agent_Policy -->|Affects| Average_Speed
        Agent_Policy -->|Affects| Collision_Rate
        Agent_Policy -->|Affects| Throughput
"""
main_entities="""
graph TD
    %% Main Entities of the System
    subgraph Entities
        Autonomous_Vehicles["Autonomous Vehicles (Agents)"]
        Intersection["Intersection (Road Infrastructure)"]
        Traffic_Flow["Traffic Flow"]
        Agent_Policies["Agent Policies (Decision-Making Models)"]
    end

    %% Attributes and Resources for Autonomous Vehicles
    subgraph Autonomous_Vehicles_Details
        Attributes_AV["Attributes: Speed, Position, Direction, Assigned Policy, Current Lane"]
        Resources_AV["Resources: Road Space, Crossing Priority, Lanes"]
        Autonomous_Vehicles --> Attributes_AV
        Autonomous_Vehicles --> Resources_AV
    end

    %% Attributes and Resources for Intersection
    subgraph Intersection_Details
        Attributes_Intersection["Attributes: Number of Lanes, Intersection Layout"]
        Resources_Intersection["Resources: Lane Capacity"]
        Intersection --> Attributes_Intersection
        Intersection --> Resources_Intersection
    end

    %% Attributes and Resources for Traffic Flow
    subgraph Traffic_Flow_Details
        Attributes_Traffic["Attributes: Arrival Rate, Traffic Density, Vehicle Types"]
        Resources_Traffic["Resources: Access to Intersection, Road Segments"]
        Traffic_Flow --> Attributes_Traffic
        Traffic_Flow --> Resources_Traffic
    end

    %% Attributes and Resources for Agent Policies
    subgraph Agent_Policies_Details
        Attributes_Policies["Attributes: Learned Policy, Decision Rules, Reward Function"]
        Resources_Policies["Resources: Computational Resources, Control Over Vehicle Behavior"]
        Agent_Policies --> Attributes_Policies
        Agent_Policies --> Resources_Policies
    end
"""

system_representation = """
graph TD
    subgraph Main_Entities
        AV["Autonomous Vehicles (Agents)"]
        Intersection["Intersection (Road Infrastructure)"]
        Traffic_Flow["Traffic Flow"]
        Agent_Policies["Agent Policies (Decision-Making Models)"]
    end

    subgraph Variables
        Exogenous["Exogenous Variables"]
        Endogenous["Endogenous Variables"]
    end

    subgraph Exogenous_Variables
        Traffic_Density["Traffic Density"]
        Intersection_Config["Intersection Configuration"]
        Controlled_Vehicles["Number of Controlled Vehicles"]
        Agent_Policy["Agent's Policy"]
    end

    subgraph Endogenous_Variables
        Average_Speed["Average Speed"]
        Collision_Rate["Collision Rate"]
        Throughput["System Throughput"]
    end

    AV -->|"Attributes: Speed, Position, Direction"| Intersection
    AV -->|"Compete for: Road space, Priority"| Traffic_Flow
    Intersection -->|"Attributes: Number of lanes, Layout"| Traffic_Flow
    Traffic_Flow -->|"Attributes: Arrival rate, Density"| Exogenous
    Agent_Policies -->|"Guide agent behavior"| AV

    Exogenous -->|"Non-Controllable"| Traffic_Density
    Exogenous -->|"Non-controllable"| Intersection_Config
    Exogenous -->|"Controllable"| Controlled_Vehicles
    Exogenous -->|"Controllable"| Agent_Policy

    Endogenous -->|""| Average_Speed
    Endogenous -->|"Includes"| Collision_Rate
    Endogenous -->|"Includes"| Throughput
"""

# Generate Mermaid HTML content for each diagram
def render_mermaid(mermaid_code):
    mermaid_html = f"""
    <div class="mermaid" style="width: 100%; height: 100%; min-height: 1500px;">
    {mermaid_code}
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/9.4.3/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    """
    components.html(mermaid_html,height=1500)

# Streamlit app setup
st.title("Autonomous Vehicle Simulation System Diagrams")

# Render the diagrams
st.header("1. Agent Behavior - Decision Flow")
render_mermaid(decision_flow_code)

st.header("2. Agent Training Process - Sequence Diagram")
render_mermaid(training_sequence_code)

st.header("3. System Variables and Feedback Loops - Causal Loop Diagram")
render_mermaid(causal_loop_code)

st.header("4. Interaction Between Multiple Agents - Interaction Diagram")
render_mermaid(interaction_diagram_code)

st.header("5. System Components - Entity Relationship Diagram")
render_mermaid(entity_relationship_code)

render_mermaid(simulation_goals_code)

render_mermaid(main_entities_code)

render_mermaid(system_variables_code)

render_mermaid(main_entities)

render_mermaid(system_representation)

