
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

        %% Global font size for all text
        classDef defaultFontSize font-size:12px;
        class Non_Controllable,Traffic_Density,Intersection_Config,Controllable,Controlled_Vehicles,Agent_Policy,Average_Speed,Collision_Rate,Throughput defaultFontSize;
        classDef defaultLinkSize font-size:12px;
        class Traffic_Density,Intersection_Config,Controlled_Vehicles,Agent_Policy,Average_Speed,Collision_Rate,Throughput defaultLinkSize;

        %% Apply the same font size to all edges and their labels
        linkStyle default stroke-width:2px;
        classDef nodeStyle fill:#f9f, stroke:#333, stroke-width:2px, font-size:12px, border-radius:10px;
"""



system_representation = """
    graph TD
        subgraph Main_Entities
            AV["Autonomous Vehicles (Agents)"]
            Intersection["Intersection (Road Infrastructure)"]
            Traffic_Flow["Traffic Flow"]
            Agent_Policies["Agent Policies (Decision-Making Models)"]
        end

        subgraph Autonomous_Vehicles_Details
            Attributes_AV["Attributes: Speed, Position, Direction, Assigned Policy, Current Lane"]
            Resources_AV["Resources: Road Space, Crossing Priority, Lanes"]
            AV --> Attributes_AV
            AV --> Resources_AV
        end

        subgraph Intersection_Details
            Attributes_Intersection["Attributes: Number of Lanes, Intersection Layout"]
            Resources_Intersection["Resources: Lane Capacity"]
            Intersection --> Attributes_Intersection
            Intersection --> Resources_Intersection
        end

        subgraph Traffic_Flow_Details
            Attributes_Traffic["Attributes: Arrival Rate, Traffic Density, Vehicle Types"]
            Resources_Traffic["Resources: Access to Intersection, Road Segments"]
            Traffic_Flow --> Attributes_Traffic
            Traffic_Flow --> Resources_Traffic
        end

        subgraph Agent_Policies_Details
            Attributes_Policies["Attributes: Learned Policy, Decision Rules, Reward Function"]
            Resources_Policies["Resources: Computational Resources, Control Over Vehicle Behavior"]
            Agent_Policies --> Attributes_Policies
            Agent_Policies --> Resources_Policies
        end

        %% Relations between main entities
        AV -->|"Attributes: Speed, Position, Direction"| Intersection
        AV -->|"Resources: Road space, Priority"| Traffic_Flow
        Intersection -->|"Attributes: Number of lanes, Layout"| Traffic_Flow
        Traffic_Flow -->|"Attributes: Arrival rate, Density"| AV
        Agent_Policies -->|"Guide agent behavior"| AV
"""



main_entities = """
graph TD
    A[Main Entities] --> B[Autonomous Vehicles (Agents)]
    A --> C[Intersection (Road Infrastructure)]
    A --> D[Traffic Flow]
    A --> E[Agent Policies (Decision-Making Models)]

    B --> B1[Attributes: Speed, Position, Direction, Assigned Policy, Current Lane]
    B --> B2[Resources: Road Space, Crossing Priority, Lanes]

    C --> C1[Attributes: Number of Lanes, Intersection Layout]
    C --> C2[Resources: Lane Capacity]

    D --> D1[Attributes: Arrival Rate, Traffic Density, Vehicle Types]
    D --> D2[Resources: Access to Intersection, Road Segments]

    E --> E1[Attributes: Learned Policy, Decision Rules, Reward Function]
    E --> E2[Resources: Computational Resources, Control Over Vehicle Behavior]
"""

operation_policy = """
    graph TD
    %% Operation Policies to be Tested (scenarios)
    subgraph Policies
        MlpPolicy["MlpPolicy (Multi-Layer Perceptron)"]
        CnnPolicy["CnnPolicy (Convolutional Neural Network)"]
        SocialAttention["Social Attention Mechanisms"]
    end

    %% DQN Algorithm
    DQN["DQN Algorithm (Deep Q-Network)"]
    DQN --> Policies

    %% MlpPolicy Scenario and Objective
    subgraph MlpPolicy_Scenario
        Scenario_Mlp["Scenario: Agents use MlpPolicy to navigate intersections based on immediate surroundings."]
        Objective_Mlp["Objective: Assess efficiency (throughput, wait times) & safety (collision rates)."]
    end
    MlpPolicy --> MlpPolicy_Scenario
    MlpPolicy --> MlpPolicy_Scenario
    Scenario_Mlp --> Objective_Mlp

    %% CnnPolicy Scenario and Objective
    subgraph CnnPolicy_Scenario
        Scenario_Cnn["Scenario: Agents interpret visual/spatial data for lane changes, merging, and crossing orders."]
        Objective_Cnn["Objective: Evaluate ability to recognize spatial configurations and respond to traffic."]
    end
    CnnPolicy --> CnnPolicy_Scenario
    CnnPolicy --> Objective_Cnn
    Scenario_Cnn --> Objective_Cnn

    %% Social Attention Mechanism Scenario and Objective
    subgraph SocialAttention_Scenario
        Scenario_Social["Scenario: Agents observe and interpret behaviors of nearby vehicles."]
        Objective_Social["Objective: Investigate social attention’s effect on cooperation and traffic flow."]
    end
    SocialAttention --> SocialAttention_Scenario
    SocialAttention --> Objective_Social
    Scenario_Social --> Objective_Social

    %% Agents Grouping (Three Groups of Agents)
    subgraph Agent_Groups
        Group1["Group 1 (MlpPolicy agents)"]
        Group2["Group 2 (CnnPolicy agents)"]
        Group3["Group 3 (Social Attention Mechanism agents)"]
    end
    Group1 --> MlpPolicy
    Group2 --> CnnPolicy
    Group3 --> SocialAttention
"""

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

system_code_metrics = """
flowchart TD
    subgraph H[Performance Metrics]
        I[Efficiency]
        I2[Throughput]
        I3[Average speed]

        J[Safety]
        J1[Collision rates]
        J3[Compliance with traffic rules]

        K[Adaptability]
        K1[Generalizing learned behaviors]
        K2[Adaptability to new traffic scenarios]

        %% Linking Metrics to Details
        I --> I2
        I --> I3
        J --> J1
        J --> J3
        K --> K1
        K --> K2
    end
"""



