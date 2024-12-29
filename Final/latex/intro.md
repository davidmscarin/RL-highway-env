### 1. Project Description

#### Ojective:
 To simulate and evaluate reinforcement learning algorithms that enable autonomous vehicles to perform complex driving tasks while ensuring safety and optimizing traffic flow. 
#### Focus On:
**Road Intersection** simulation scenarios, using an agent-based simulation framework ***Highway-Env***, at  where multiple autonomous vehicles, trained with different Deep Reinforcement Learning (DRL) policies, must determine the order of crossing and appropriate speeds, aiming to **maximize traffic efficiency and minimize collision risks**.  

The agents will be tested under various combinations of environmental variables (e.g., traffic density vehicle speed) to evaluate their behavior and decision-making processes in real-time.  


### 2.Goals of the Simulation Project
3 key stages:

> ##### 1.Successfully training the agents
> Apply selected Deep Reinforcement Learning (DRL) algorithms to train autonomous agents in navigating a road intersection.
           
> ##### 2.Introducing System Perturbations
> Systematically vary key factors: traffic flow and agents policy
           
> ##### 3.Evaluating system scalability and robustness
> Evaluating the scalability and robustness of the MAS under more challenging or extreme conditions. 

      
### 3.Models of Decision Support Considered

> **Descriptive**

> **Predictive**

> **Prescriptive**


### 4. Model Characteristics:

> **Dynamic**
      
> **Stochastic**
      
> **Discrete**
      

### 5.Main Entities of the System

Main entities are the objects of interest that interact with each other and the environment dynamically. 
The primary entities of the system are:

    
**Autonomous Vehicles (Agents):**
Each autonomous vehicle represents an agent in the simulation with decision-making capabilities. 
> **Attributes**: Speed, position, direction, assigned policy (decision-making model), current lane.
          
> **Resources** they compete for: Road space, crossing priority, lanes.
          
**Intersection (Road Infrastructure)**
The road intersection is a static entity but a key part of the system. It defines where vehicles meet and interact. 
          
>**Attributes**: Number of lanes, intersection layout.
          
> **Resources** they compete for: Lane capacity (the number of vehicles that can use a lane or section of road at a time).
          
**Traffic Flow:**
Represents the overall movement of vehicles through the system. This is a dynamic entity in terms of the rate at which vehicles arrive at and depart from the intersection.
          
>**Attributes**: Arrival rate of vehicles, traffic density, vehicle types (e.g. vehicles, ego-vehicles).
          
>**Resources** they compete for: Access to the intersection, road segments.
          
**Agent Policies (Decision-Making Models):**
Each autonomous vehicle (agent) operates based on a decision-making policy (learned behavior from DRL). 
These policies guide how each vehicle responds to other vehicles and environmental factors.
          
>**Attributes**: Learned policy, decision rules, reward function (for reinforcement learning).
          
>**Resources**they compete for: Computational resources for decision-making (though implicit in the model), control over vehicle behavior.

---   

### 6.Variables of the System

**Exogenous**
>**Non-Controllable**
>>**Traffic Density**:
Reflects the overall number of vehicles within the system or passing through the intersection at any given time.
                  
>>**Intersection Configuration (Lane Layout):**
Represents the structure of the intersection, such as the number of lanes or the presence of dedicated turn lanes.

>**Controllable**

>>**Number of Controlled Vehicles**

>>**Agent's Policy**
          
**Endogenous**

>**Average Speed:**
Controlled Agents average speed.

>**Collision Rate:**
A key variable that reflects the number of collisions or near-collisions occurring in the system.
          
>**System Throughput:**
The number of vehicles successfully passing through the intersection over a given period.

### 7.Operation Policies to be Tested (scenarios)
          
