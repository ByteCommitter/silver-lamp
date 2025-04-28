# Densest Subgraph Discovery: Algorithm Implementation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DAA Project](https://img.shields.io/badge/Course-DAA_Design_%26_Analysis_of_Algorithms-blueviolet)]()

Implementation of exact and optimized densest subgraph discovery algorithms from "*Efficient Algorithms for Densest Subgraph Discovery*" (Fang et al., PVLDB 2019) for the DAA course project.

## Key Features
- **Exact Algorithm** (Algorithm 1): Traditional maximum flow approach with binary search
- **CoreExact** (Algorithm 4): Optimized version using (k,Ψ)-core decomposition
- **Core-Based Optimizations**: 
  - Progressive core refinement
  - Adaptive flow networks
  - Component-specific thresholds
- **Pattern Support**: Edge-density and h-clique density calculations
- **Dataset Handling**: Preprocessing for real-world network datasets

## Installation
git clone https://github.com/yourusername/densest-subgraph-daa.git
cd densest-subgraph-daa

Build with Gradle
./gradlew build

Run Algorithm 1
./gradlew run -Palgorithm=dsd.ExactDensestSubgraph

Run Algorithm 4
./gradlew run -Palgorithm=dsd.CoreExact

text

## Datasets
We tested on four network datasets:
1. **CA-HepTh** (9,877 nodes, 51K edges) - Collaboration network
2. **AS-733** (6.4K nodes, 13K edges) - Internet topology
3. **CAIDA AS** (26K nodes, 53K edges) - Infrastructure relationships
4. **NetScience** (379 nodes, 914 edges) - Co-authorship network

## Getting Started
### Prerequisites
- Java JDK 17+
- Gradle 7.6+
- Python 3.8+ (for dataset preprocessing)

### Usage Example
// Sample input graph initialization
Graph network = GraphLoader.loadFromFile("data/CA-HepTh.txt");

// Run Exact Algorithm
DensestSubgraph exactResult = new ExactDensestSubgraph().find(network);

// Run CoreExact Optimization
DensestSubgraph coreExactResult = new CoreExact().find(network);

text

## Algorithm Overview
### Algorithm 1: Exact Approach
- Binary search over density parameters
- Flow network construction with:
  - Source/sink nodes
  - Vertex capacity constraints
  - Intermediate clique nodes
- Minimum cut analysis for density verification

### Algorithm 4: CoreExact
- **Core Decomposition** pre-processing
- Progressive core refinement:
while (currentCore.density() > threshold) {
core = core.refine();
buildReducedNetwork(core);
}

text
- Adaptive flow network sizing (60-90% reduction in later stages)

| Feature              | Algorithm 1 | CoreExact   |
|----------------------|-------------|-------------|
| Time Complexity      | O(n|Λ| + ...) | O(|R_k|³)   |
| Space Complexity     | O(n + |Λ|)  | O(|R_k| + |Λ_C|) |
| Best For             | Small graphs| Large networks |

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-optimization`
3. Commit changes: `git commit -m 'Add some optimization'`
4. Push to branch: `git push origin feature/new-optimization`
5. Open a Pull Request

### Code of Conduct
- Follow Google Java Style Guide
- Include JUnit tests for new features
- Document complex algorithms with flowcharts in `/docs`

## Credits
**Project Team:**
- Gurumurthy V- Core algorithm implementation, Report making
- Aarush - Core algorithm implementation, Report making, Benchmarking & Analysis
- Pranav V A - Core algorithm implementation, Report Making, Benchmarking & Analysis
- Snehal M S - Core algorithm implementation, Report Making
- Sriram Jatin - Dataset Search

**Advisors:** [Course Instructor Name(s)]

## License
Apache License 2.0 - See [LICENSE](LICENSE) for details

## Project Status
- ✅ Exact Algorithm implemented
- ✅ CoreExact optimization complete
- ✅ Dataset preprocessing scripts
- 🔄 Future Work: 
- Distributed computation support
- GPU acceleration for flow networks
- Interactive visualization module
