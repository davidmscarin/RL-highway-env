```mermaid
flowchart TD
    A[Autonomous Vehicles] --> B[Intersection]
    B --> C[Traffic Flow]
    A --> D[Decision-Making (DRL Policies)]
    D --> B
    C --> E[System Throughput]
    C --> F[Collision Rate]
    E --> G[Optimization Metrics]
    F --> G
    G --> H[Evaluation]
    H --> I[Final Report]
