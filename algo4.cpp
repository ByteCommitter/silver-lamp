#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include <chrono>
using namespace std;

class Pattern {
private:
    int h; 
    bool isClique; 
    vector<vector<int>> adjMatrix; 
    
public:
    
    Pattern(int size) {
        h = size;
        isClique = true;
        adjMatrix.resize(h, vector<int>(h, 1));
        for (int i = 0; i < h; i++) {
            adjMatrix[i][i] = 0; 
        }
    }
    
    
    Pattern(vector<vector<int>>& adj) {
        adjMatrix = adj;
        h = adj.size();
        isClique = true;
        
        
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < h; j++) {
                if (i != j && adjMatrix[i][j] == 0) {
                    isClique = false;
                    break;
                }
            }
            if (!isClique) break;
        }
    }
    
    int getSize() const {
        return h;
    }
    
    bool isCliquePattern() const {
        return isClique;
    }
    
    const vector<vector<int>>& getAdjMatrix() const {
        return adjMatrix;
    }
};

class Graph {
private:
    int n; 
    int m; 
    vector<vector<int>> adj; 
    vector<int> vertexMap; 
    vector<int> reverseMap; 
    
    
    vector<vector<int>> capacity;
    vector<vector<int>> flow;
    
    
    int componentSize;
    vector<int> component;
    
public:
    
    Graph(int vertices) {
        n = vertices;
        m = 0;
        adj.resize(n);
        reverseMap.resize(n);
        
        for (int i = 0; i < n; i++) {
            reverseMap[i] = i;
        }
    }
    
    
    Graph(const string& filename) {
        readGraphFromFile(filename);
    }
    
    
    void readGraphFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Unable to open file " << filename << endl;
            exit(1);
        }
        
        cout << "Reading graph from file: " << filename << endl;
        
        string line;
        unordered_map<int, int> idMap; 
        set<pair<int, int>> edgeSet; 
        
        
        n = 0;
        m = 0;
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                
                if (line.find("Nodes:") != string::npos && line.find("Edges:") != string::npos) {
                    sscanf(line.c_str(), "# Nodes: %d Edges: %d", &n, &m);
                    cout << "Found metadata: " << n << " nodes, " << m << " edges" << endl;
                }
                continue;
            }
            break; 
        }
        
        
        if (n > 0) {
            adj.resize(n);
            vertexMap.resize(n);
            reverseMap.resize(n);
            for (int i = 0; i < n; i++) {
                vertexMap[i] = i;
                reverseMap[i] = i;
            }
        }
        
        
        do {
            if (line.empty() || line[0] == '#') continue;
            
            istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) {
                cerr << "Error parsing line: " << line << endl;
                continue;
            }
            
            
            if (idMap.find(u) == idMap.end()) {
                int idx = idMap.size();
                idMap[u] = idx;
                if (idx >= reverseMap.size()) {
                    reverseMap.push_back(u);
                } else {
                    reverseMap[idx] = u;
                }
            }
            
            if (idMap.find(v) == idMap.end()) {
                int idx = idMap.size();
                idMap[v] = idx;
                if (idx >= reverseMap.size()) {
                    reverseMap.push_back(v);
                } else {
                    reverseMap[idx] = v;
                }
            }
            
            
            int mappedU = idMap[u];
            int mappedV = idMap[v];
            
            if (mappedU != mappedV) { 
                edgeSet.insert({min(mappedU, mappedV), max(mappedU, mappedV)});
            }
        } while (getline(file, line));
        
        
        if (n == 0) {
            n = idMap.size();
            m = edgeSet.size();
            
            adj.resize(n);
            vertexMap.resize(n);
            
            
            for (const auto& pair : idMap) {
                vertexMap[pair.second] = pair.first;
            }
        }
        
        
        for (const auto& edge : edgeSet) {
            int u = edge.first;
            int v = edge.second;
            
            adj[u].push_back(v);
            adj[v].push_back(u); 
        }
        
        file.close();
        
        cout << "Graph loaded: " << n << " vertices, " << m << " edges" << endl;
    }
    
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); 
        m++;
    }
    
    
    int getNumVertices() const {
        return n;
    }
    
    
    int getNumEdges() const {
        return m;
    }
    
    
    const vector<vector<int>>& getAdjList() const {
        return adj;
    }
    
    
    int getOriginalID(int v) const {
        if (v >= 0 && v < reverseMap.size()) {
            return reverseMap[v];
        }
        return -1; 
    }
    
    
    vector<vector<int>> enumerateCliques(int h) {
        vector<vector<int>> cliques;
        vector<int> currentClique;
        
        
        for (int v = 0; v < n; v++) {
            currentClique.push_back(v);
            enumerateCliquesRecursive(v, 1, h, currentClique, cliques);
            currentClique.pop_back();
        }
        
        return cliques;
    }
    
    
    void enumerateCliquesRecursive(int startVertex, int depth, int h, vector<int>& currentClique, vector<vector<int>>& cliques) {
        if (depth == h) {
            cliques.push_back(currentClique);
            return;
        }
        
        
        vector<int> candidates;
        for (int neighbor : adj[startVertex]) {
            if (neighbor > startVertex) { 
                bool isCandidate = true;
                
                
                for (int i = 0; i < depth; i++) {
                    if (currentClique[i] != startVertex && 
                        find(adj[neighbor].begin(), adj[neighbor].end(), currentClique[i]) == adj[neighbor].end()) {
                        isCandidate = false;
                        break;
                    }
                }
                
                if (isCandidate) {
                    candidates.push_back(neighbor);
                }
            }
        }
        
        
        for (int candidate : candidates) {
            currentClique.push_back(candidate);
            enumerateCliquesRecursive(candidate, depth + 1, h, currentClique, cliques);
            currentClique.pop_back();
        }
    }
    
    
    int cliqueDegree(int v, const Pattern& pattern) {
        if (v < 0 || v >= n) {
            return 0; 
        }
        
        if (pattern.isCliquePattern()) {
            int h = pattern.getSize();
            if (h == 2) {
                
                return adj[v].size();
            } else {
                
                int count = 0;
                vector<int> currentClique = {v};
                countCliquesContainingVertex(v, 1, h, currentClique, count);
                return count;
            }
        } else {
            
            
            return 0; 
        }
    }
    
    
    void countCliquesContainingVertex(int startVertex, int depth, int h, vector<int>& currentClique, int& count) {
        if (depth == h) {
            count++;
            return;
        }
        
        
        vector<int> candidates;
        for (int neighbor : adj[startVertex]) {
            if (neighbor > currentClique[0]) { 
                bool isCandidate = true;
                
                
                for (int i = 0; i < depth; i++) {
                    if (find(adj[neighbor].begin(), adj[neighbor].end(), currentClique[i]) == adj[neighbor].end()) {
                        isCandidate = false;
                        break;
                    }
                }
                
                if (isCandidate) {
                    candidates.push_back(neighbor);
                }
            }
        }
        
        
        for (int candidate : candidates) {
            currentClique.push_back(candidate);
            countCliquesContainingVertex(candidate, depth + 1, h, currentClique, count);
            currentClique.pop_back();
        }
    }
    
    
    vector<int> coreDecomposition(const Pattern& pattern) {
        vector<int> core(n, 0);
        vector<int> degrees(n);
        
        
        for (int i = 0; i < n; i++) {
            degrees[i] = cliqueDegree(i, pattern);
        }
        
        
        int maxDegree = 0;
        for (int i = 0; i < n; i++) {
            maxDegree = max(maxDegree, degrees[i]);
        }
        
        vector<vector<int>> bins(maxDegree + 1);
        for (int i = 0; i < n; i++) {
            bins[degrees[i]].push_back(i);
        }
        
        vector<bool> removed(n, false);
        int remaining = n;
        
        
        for (int d = 0; d <= maxDegree && remaining > 0; d++) {
            while (!bins[d].empty()) {
                int v = bins[d].back();
                bins[d].pop_back();
                
                if (removed[v] || degrees[v] > d) continue;
                
                core[v] = d;
                removed[v] = true;
                remaining--;
                
                
                for (int u : adj[v]) {
                    if (!removed[u] && degrees[u] > d) {
                        
                        auto it = find(bins[degrees[u]].begin(), bins[degrees[u]].end(), u);
                        if (it != bins[degrees[u]].end()) {
                            bins[degrees[u]].erase(it);
                        }
                        
                        
                        degrees[u]--;
                        
                        
                        bins[degrees[u]].push_back(u);
                    }
                }
            }
        }
        
        return core;
    }
    
    
    vector<vector<int>> findConnectedComponents(const vector<int>& vertices) {
        vector<vector<int>> components;
        if (vertices.empty()) return components;
        
        unordered_set<int> vertexSet(vertices.begin(), vertices.end());
        vector<bool> visited(n, false);
        
        for (int v : vertices) {
            if (!visited[v]) {
                vector<int> component;
                queue<int> q;
                q.push(v);
                visited[v] = true;
                
                while (!q.empty()) {
                    int u = q.front();
                    q.pop();
                    component.push_back(u);
                    
                    for (int neighbor : adj[u]) {
                        if (vertexSet.find(neighbor) != vertexSet.end() && !visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
                
                components.push_back(component);
            }
        }
        
        return components;
    }
    
    
    void buildFlowNetwork(double alpha, const vector<int>& component, const Pattern& pattern) {
        int componentSize = component.size();
        if (componentSize == 0) return; 
        
        int patternSize = pattern.getSize();
        
        
        capacity.clear();
        flow.clear();
        int networkSize = componentSize * 2 + 2;
        capacity.resize(networkSize, vector<int>(networkSize, 0));
        flow.resize(networkSize, vector<int>(networkSize, 0));
        
        int s = 0;  
        int t = componentSize * 2 + 1;  
        
        
        unordered_map<int, int> vertexToNetworkIndex;
        for (int i = 0; i < componentSize; i++) {
            vertexToNetworkIndex[component[i]] = i + 1;  
        }
        
        
        for (int i = 0; i < componentSize; i++) {
            int v = component[i];
            int node = i + 1;  
            
            
            capacity[s][node] = cliqueDegree(v, pattern);
            
            
            capacity[node][t] = ceil(alpha * patternSize);
            
            
            for (int u : adj[v]) {
                
                auto it = vertexToNetworkIndex.find(u);
                if (it != vertexToNetworkIndex.end()) {
                    int uNode = it->second;
                    capacity[node][uNode] = 1;
                }
            }
        }
    }
    
    
    bool bfs(int s, int t, vector<int>& parent) {
        if (s >= capacity.size() || t >= capacity.size()) return false; 
        
        fill(parent.begin(), parent.end(), -1);
        vector<bool> visited(parent.size(), false);
        queue<int> q;
        q.push(s);
        visited[s] = true;
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (int v = 0; v < parent.size(); v++) {
                
                if (u < capacity.size() && v < capacity[u].size() && 
                    !visited[v] && capacity[u][v] > flow[u][v]) {
                    q.push(v);
                    parent[v] = u;
                    visited[v] = true;
                    
                    if (v == t) return true;
                }
            }
        }
        
        return visited[t];
    }
    
    
    int maxFlow(int s, int t, vector<int>& minCut) {
        if (capacity.empty() || s >= capacity.size() || t >= capacity.size()) {
            minCut.clear();
            return 0; 
        }
        
        int u, v;
        vector<int> parent(capacity.size(), -1);
        
        
        for (u = 0; u < capacity.size(); u++) {
            for (v = 0; v < capacity[u].size(); v++) {
                flow[u][v] = 0;
            }
        }
        
        int max_flow = 0;
        
        
        while (bfs(s, t, parent)) {
            int path_flow = numeric_limits<int>::max();
            
            
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                path_flow = min(path_flow, capacity[u][v] - flow[u][v]);
            }
            
            
            for (v = t; v != s; v = parent[v]) {
                u = parent[v];
                flow[u][v] += path_flow;
                flow[v][u] -= path_flow;
            }
            
            max_flow += path_flow;
        }
        
        
        minCut.clear();
        vector<bool> visited(capacity.size(), false);
        queue<int> q;
        q.push(s);
        visited[s] = true;
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (int v = 0; v < capacity.size(); v++) {
                if (v < capacity[u].size() && !visited[v] && capacity[u][v] > flow[u][v]) {
                    q.push(v);
                    visited[v] = true;
                }
            }
        }
        
        
        for (int i = 1; i <= componentSize && i < capacity.size(); i++) {
            if (visited[i]) {
                
                if (i-1 < component.size()) {
                    minCut.push_back(component[i-1]);
                }
            }
        }
        
        return max_flow;
    }
    
    
    vector<int> coreExact(const Pattern& pattern) {
        cout << "Starting CoreExact algorithm..." << endl;
        auto startTime = chrono::high_resolution_clock::now();
        
        
        cout << "Performing core decomposition..." << endl;
        vector<int> coreNumbers = coreDecomposition(pattern);
        
        auto coreDecompTime = chrono::high_resolution_clock::now();
        cout << "Core decomposition completed in " 
             << chrono::duration_cast<chrono::milliseconds>(coreDecompTime - startTime).count() 
             << " ms" << endl;
        
        
        int kMax = 0;
        for (int k : coreNumbers) {
            kMax = max(kMax, k);
        }
        
        cout << "Maximum core number: " << kMax << endl;
        
        
        int kDoublePrime = ceil(kMax / 2.0);
        
        
        vector<int> kDoublePrimeCore;
        for (int i = 0; i < n; i++) {
            if (coreNumbers[i] >= kDoublePrime) {
                kDoublePrimeCore.push_back(i);
            }
        }
        
        cout << "Size of (k'', Ψ)-core: " << kDoublePrimeCore.size() << endl;
        
        
        double rho_prime = 0.0;
        vector<bool> removed(n, false);
        vector<int> degrees = coreNumbers;
        
        
        vector<pair<int, int>> vertexCorePairs;
        for (int i = 0; i < n; i++) {
            vertexCorePairs.push_back({i, coreNumbers[i]});
        }
        
        sort(vertexCorePairs.begin(), vertexCorePairs.end(), 
             [](const pair<int, int>& a, const pair<int, int>& b) {
                 return a.second < b.second;
             });
        
        
        vector<int> residualGraph;
        for (int i = 0; i < n; i++) {
            residualGraph.push_back(i);
        }
        
        for (const auto& pair : vertexCorePairs) {
            int v = pair.first;
            removed[v] = true;
            
            
            residualGraph.erase(remove(residualGraph.begin(), residualGraph.end(), v), residualGraph.end());
            
            
            if (!residualGraph.empty()) {
                double density = calculateDensity(residualGraph, pattern);
                rho_prime = max(rho_prime, density);
            }
        }
        
        cout << "Maximum density of residual graphs (rho'): " << rho_prime << endl;
        
        
        int kPrime = ceil(rho_prime);
        
        
        vector<vector<int>> components = findConnectedComponents(kDoublePrimeCore);
        
        cout << "Number of connected components in (k'', Ψ)-core: " << components.size() << endl;
        
        
        double rho_double_prime = 0.0;
        for (const auto& component : components) {
            double density = calculateDensity(component, pattern);
            rho_double_prime = max(rho_double_prime, density);
        }
        
        cout << "Maximum density of connected components (rho''): " << rho_double_prime << endl;
        
        
        int kDoublePrimeNew = std::max(kPrime, static_cast<int>(ceil(rho_double_prime)));
        
        
        double l = rho_double_prime;  
        double u = kMax;  
        vector<int> D;  
        double bestDensity = 0;
        
        cout << "Initial bounds: l = " << l << ", u = " << u << endl;
        
        
        for (const auto& C : components) {
            if (C.empty()) {
                cout << "Skipping empty component" << endl;
                continue;
            }
            
            
            vector<int> currentC = C;
            if (l > kDoublePrimeNew) {
                int kCeil = ceil(l);
                vector<int> higherCore;
                for (int v : C) {
                    if (coreNumbers[v] >= kCeil) {
                        higherCore.push_back(v);
                    }
                }
                currentC = higherCore;
            }
            
            if (currentC.empty()) {
                cout << "Skipping component (empty after core filtering)" << endl;
                continue;
            }
            
            cout << "Processing component of size " << currentC.size() << endl;
            
            
            componentSize = currentC.size(); 
            component = currentC; 
            buildFlowNetwork(l, currentC, pattern);
            
            
            vector<int> minCut;
            int maxFlowValue = maxFlow(0, currentC.size() * 2 + 1, minCut);
            
            
            if (minCut.empty()) {
                cout << "Skipping component (S = {s})" << endl;
                continue;
            }
            
            
            double localL = l;
            double localU = u;
            vector<int> U = minCut;
            
            int iteration = 0;
            while (localU - localL >= 1.0 / (currentC.size() * (currentC.size() - 1))) {
                iteration++;
                double alpha = (localL + localU) / 2.0;
                
                cout << "Binary search iteration " << iteration << ": alpha = " << alpha << endl;
                
                
                buildFlowNetwork(alpha, currentC, pattern);
                
                
                vector<int> newMinCut;
                maxFlowValue = maxFlow(0, currentC.size() * 2 + 1, newMinCut);
                
                if (newMinCut.empty()) {
                    
                    localU = alpha;
                    cout << "  S = {s}, updating upper bound to " << localU << endl;
                } else {
                    
                    if (alpha > ceil(localL)) {
                        
                        vector<int> newC;
                        for (int v : currentC) {
                            if (find(newMinCut.begin(), newMinCut.end(), v) != newMinCut.end()) {
                                newC.push_back(v);
                            }
                        }
                        currentC = newC;
                        cout << "  Reduced component size to " << currentC.size() << endl;
                    }
                    
                    localL = alpha;
                    U = newMinCut;
                    cout << "  Updating lower bound to " << localL << " and U size to " << U.size() << endl;
                }
            }
            
            
            double density = calculateDensity(U, pattern);
            cout << "Component density: " << density << endl;
            
            
            if (density > bestDensity) {
                bestDensity = density;
                D = U;
                cout << "New best density: " << bestDensity << " with " << D.size() << " vertices" << endl;
            }
        }
        
        auto endTime = chrono::high_resolution_clock::now();
        cout << "CoreExact completed in " 
             << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() 
             << " ms" << endl;
        
        return D;
    }
    
    
    double calculateDensity(const vector<int>& vertices, const Pattern& pattern) {
        if (vertices.empty()) return 0;
        
        if (pattern.isCliquePattern() && pattern.getSize() == 2) {
            
            int edges = 0;
            unordered_set<int> vertexSet(vertices.begin(), vertices.end());
            
            for (int u : vertices) {
                for (int v : adj[u]) {
                    if (vertexSet.find(v) != vertexSet.end() && u < v) {
                        edges++;
                    }
                }
            }
            
            return static_cast<double>(edges) / vertices.size();
        } else {
            
            int patternInstances = 0;
            
            if (pattern.isCliquePattern()) {
                
                int h = pattern.getSize();
                vector<int> currentClique;
                
                for (int v : vertices) {
                    currentClique.clear();
                    currentClique.push_back(v);
                    
                    
                    unordered_set<int> vertexSet(vertices.begin(), vertices.end());
                    countPatternsInSubgraph(v, 1, h, currentClique, patternInstances, vertexSet);
                }
                
                
                patternInstances /= h;
            } else {
                
                
                
            }
            
            return static_cast<double>(patternInstances) / vertices.size();
        }
    }
    
    
    void countPatternsInSubgraph(int startVertex, int depth, int h, vector<int>& currentPattern, 
                                int& count, const unordered_set<int>& vertexSet) {
        if (depth == h) {
            count++;
            return;
        }
        
        
        for (int neighbor : adj[startVertex]) {
            if (vertexSet.find(neighbor) != vertexSet.end() && neighbor > currentPattern[0]) {
                bool isCandidate = true;
                
                
                for (int i = 0; i < depth; i++) {
                    if (find(adj[neighbor].begin(), adj[neighbor].end(), currentPattern[i]) == adj[neighbor].end()) {
                        isCandidate = false;
                        break;
                    }
                }
                
                if (isCandidate) {
                    currentPattern.push_back(neighbor);
                    countPatternsInSubgraph(neighbor, depth + 1, h, currentPattern, count, vertexSet);
                    currentPattern.pop_back();
                }
            }
        }
    }
    
    
    void printDensestSubgraph(const vector<int>& densestSubgraph, const Pattern& pattern) {
        cout << "Densest Subgraph:" << endl;
        cout << "Number of vertices: " << densestSubgraph.size() << endl;
        
        cout << "Vertices: ";
        for (int v : densestSubgraph) {
            cout << getOriginalID(v) << " ";
        }
        cout << endl;
        
        cout << "Density: " << calculateDensity(densestSubgraph, pattern) << endl;
        
        
        if (pattern.isCliquePattern() && pattern.getSize() == 2) {
            
            int edges = 0;
            unordered_set<int> vertexSet(densestSubgraph.begin(), densestSubgraph.end());
            
            for (int u : densestSubgraph) {
                for (int v : adj[u]) {
                    if (vertexSet.find(v) != vertexSet.end() && u < v) {
                        edges++;
                    }
                }
            }
            
            cout << "Number of edges: " << edges << endl;
        } else {
            
            int patternInstances = 0;
            
            if (pattern.isCliquePattern()) {
                
                int h = pattern.getSize();
                vector<int> currentClique;
                
                for (int v : densestSubgraph) {
                    currentClique.clear();
                    currentClique.push_back(v);
                    
                    
                    unordered_set<int> vertexSet(densestSubgraph.begin(), densestSubgraph.end());
                    countPatternsInSubgraph(v, 1, h, currentClique, patternInstances, vertexSet);
                }
                
                
                patternInstances /= h;
                
                cout << "Number of " << h << "-cliques: " << patternInstances << endl;
            } else {
                
                cout << "Number of pattern instances: " << patternInstances << endl;
            }
        }
    }
};
int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            cerr << "Usage: " << argv[0] << " <input_file> [pattern_size]" << endl;
            return 1;
        }
        
        string inputFile = argv[1];
        int patternSize = 2;  
        
        if (argc >= 3) {
            patternSize = stoi(argv[2]);
        }
        
        cout << "Loading graph from " << inputFile << "..." << endl;
        Graph g(inputFile);
        
        cout << "Using " << patternSize << "-clique pattern" << endl;
        Pattern pattern(patternSize);
        
        cout << "Finding densest subgraph..." << endl;
        vector<int> densestSubgraph = g.coreExact(pattern);
        
        g.printDensestSubgraph(densestSubgraph, pattern);
    } catch (const exception& e) {
        cerr << "Exception caught: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown exception caught" << endl;
        return 1;
    }
    
    return 0;
}
