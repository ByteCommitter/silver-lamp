#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <set>
#include <functional>

using namespace std;

class Graph {
private:
    int numVertices;
    vector<unordered_set<int>> adjacency_list;

    mutable vector<vector<int>> mapVertexCliques;
    mutable bool cacheInitialized = false;
    mutable vector<vector<int>> hCache;
    mutable vector<vector<int>> hMinus1Cache;
    
    bool isVertexFullyConnected(int e2, const vector<int>& current) const {
        if (e2 < 0 || e2 >= numVertices) return false;
        
        for (int e1 : current) {
            if (e1 < 0 || e1 >= numVertices || adjacency_list[e2].find(e1) == adjacency_list[e2].end()) {
                return false;
            }
        }
        return true;
    }

    void findTriangles(vector<vector<int>>& cliques) const {
        cliques.clear();
        
        cout << "Finding 3-cliques (triangles) with optimisatised way" << flush;
        for (int e1 = 0; e1 < numVertices; e1++) {
            for (int e2 : adjacency_list[e1]) {
                if (e2 <= e1) continue;
                
                for (int w : adjacency_list[e1]) {
                    if (w <= e2) continue;
                    
                    if (adjacency_list[e2].find(w) != adjacency_list[e2].end()) {
                        cliques.push_back({e1, e2, w});
                    }
                }
            }
        }
        
        cout << " Found " << cliques.size() << " triangles." << endl;
    }
    
    void findCliques(int h, vector<vector<int>>& cliques) const {
        cliques.clear();
        
        if (h == 3) {
            findTriangles(cliques);
            return;
        }
        
        cout << "Finding " << h << "-cliques with backtracking " << flush;
        
        const size_t MAX_CLIQUES = 10000000;
        size_t maxIterations = 1000000000;
        size_t iterations = 0;
        
        function<void(vector<int>&, int)> backtrack = [&](vector<int>& current, int start) {
            iterations++;
            
            if (iterations % 100000 == 0) {
                cout << "." << flush;
            }
        
            if (cliques.size() >= MAX_CLIQUES || iterations >= maxIterations) {
                return;
            }
            
            if (current.size() == h) {
                cliques.push_back(current);
                return;
            }
            int i = start; 
            while (i < numVertices && cliques.size() < MAX_CLIQUES) {
                if (isVertexFullyConnected(i, current)) {
                    current.push_back(i);
                    backtrack(current, i + 1);
                    current.pop_back();
                }
                i++;
            }
        };
        
        vector<int> current;
        backtrack(current, 0);
        
        cout << " Found " << cliques.size() << " cliques" 
             << (cliques.size() >= MAX_CLIQUES ? " (limit reached)" : "") 
             << "." << endl;
    }
    
public:
    Graph(int vertices) : numVertices(vertices) {
        if (vertices <= 0) {
            numVertices = 0;
            cerr << "Invalid number of vertices, using empty graph" << endl;
        }
        adjacency_list.resize(numVertices);
    }
    
    void addEdge(int e1, int e2) {
        if (e1 < 0 || e1 >= numVertices || e2 < 0 || e2 >= numVertices) {
            return;
        }
        adjacency_list[e1].insert(e2);
        adjacency_list[e2].insert(e1);
    }
    
    int getNumVertices() const {
        return numVertices;
    }
    
    bool hasEdge(int e1, int e2) const {
        if (e1 < 0 || e1 >= numVertices || e2 < 0 || e2 >= numVertices) return false;
        return adjacency_list[e1].find(e2) != adjacency_list[e1].end();
    }
    
    void initCache(int h) const {
        if (cacheInitialized) return;
        
        cout << "Precomputing cliques for h=" << h << "..." << flush;
        auto start = chrono::high_resolution_clock::now();
        
        hCache.clear();
        hMinus1Cache.clear();
        mapVertexCliques.resize(numVertices);
        
        try {
            findCliques(h, hCache);
            
            if (h > 1) {
                findCliques(h-1, hMinus1Cache);
            }
            
            size_t i = 0;
            while (i < hCache.size()) {
                for (int e2 : hCache[i]) {
                    if (e2 >= 0 && e2 < numVertices) {
                        mapVertexCliques[e2].push_back(i);
                    }
                }
                i++;
            }
            
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            
            cout << "Found" << hCache.size() << " h-cliques, " 
                 << hMinus1Cache.size() << " (h-1)-cliques in " << duration << "ms" << endl;
            
            cacheInitialized = true;
        }
        catch (const exception& e) {
            cout << "Error: " << e.what() << endl;
            hCache.clear();
            hMinus1Cache.clear();
        }
    }
    
    const vector<vector<int>>& getHCliques(int h) const {
        initCache(h);
        return hCache;
    }
    
    const vector<vector<int>>& getHMinus1Cliques(int h) const {
        initCache(h);
        return hMinus1Cache;
    }
    
    int cliqueDegree(int e2, int h) const {
        if (e2 < 0 || e2 >= numVertices) return 0;
        initCache(h);
        return mapVertexCliques[e2].size();
    }
    
    int getMaxDegreeClique(int h) const {
        initCache(h);
        
        int degreeMax = 0;
        int e2 = 0; 
        while (e2 < numVertices) {
            degreeMax = max(degreeMax, static_cast<int>(mapVertexCliques[e2].size()));
            e2++;
        }
        return degreeMax;
    }
    
    int countCliques(int h) const {
        initCache(h);
        return hCache.size();
    }
    
    double cliqueDensity(int h) const {
        int cliqueCount = countCliques(h);
        if (numVertices == 0) return 0.0;
        return static_cast<double>(cliqueCount) / numVertices;
    }
    
    Graph findIndSubgraph(const vector<int>& vertices) const {
        Graph subG(vertices.size());
        unordered_map<int, int> indexMap;
        
        size_t i = 0; 
        while (i < vertices.size()) {
            if (vertices[i] >= 0 && vertices[i] < numVertices) {
                indexMap[vertices[i]] = i;
            }
            i++;
        }

        i = 0;
        while (i < vertices.size()) {
            for (size_t j = i + 1; j < vertices.size(); j++) {
                int e1 = vertices[i];
                int e2 = vertices[j];
                if (e1 >= 0 && e1 < numVertices && e2 >= 0 && e2 < numVertices && hasEdge(e1, e2)) {
                    subG.addEdge(indexMap[e1], indexMap[e2]);
                }
            }
            i++;
        }
        
        return subG;
    }
};

int fordFulkerson(const vector<vector<int>>& capacity, int s, int t, vector<int>& minCut) {
    int numVertices = capacity.size();
    if (s < 0 || s >= numVertices || t < 0 || t >= numVertices) {
        cerr << "Source Invalid" << endl;
        return 0;
    }
    
    vector<vector<int>> res(numVertices, vector<int>(numVertices, 0));
    
    int i = 0;
    while (i < numVertices) {
        int j = 0;
        while (j < numVertices) {
            res[i][j] = capacity[i][j];
            j++;
        }
        i++;
    }
    
    vector<int> parent(numVertices);
    int maxFlow = 0;
    
    auto bfs = [&](vector<int>& parent) {
        fill(parent.begin(), parent.end(), -1);
        vector<bool> visited(numVertices, false);
        queue<int> q;
        
        q.push(s);
        visited[s] = true;
        parent[s] = -1;
        
        while (!q.empty()) {
            int e1 = q.front();
            q.pop();
            int e2 = 0;
            while (e2 < numVertices) {
                if (!visited[e2] && res[e1][e2] > 0) {
                    q.push(e2);
                    parent[e2] = e1;
                    visited[e2] = true;
                    if (e2 == t) return true;
                }
                e2++;
            }
        }
        
        return visited[t] == true;
    };
    
    int ffIterations = 0;
    
    while (bfs(parent)) {
        ffIterations++;
        if (ffIterations % 100 == 0) {
            cout << "." << flush;
        }
        
        int pathFlow = numeric_limits<int>::max();
        int e2 = t;
        while (e2 != s) {
            int e1 = parent[e2];
            pathFlow = min(pathFlow, res[e1][e2]);
            e2 = parent[e2];
        }
        
        e2 = t;
        while (e2 != s) {
            int e1 = parent[e2];
            res[e1][e2] -= pathFlow;
            res[e2][e1] += pathFlow;
            e2 = parent[e2];
        }
        
        maxFlow += pathFlow;
    }
    
    vector<bool> visited(numVertices, false);
    queue<int> q;
    q.push(s);
    visited[s] = true;
    
    while (!q.empty()) {
        int e1 = q.front();
        q.pop();
        
        int e2 = 0;
        while (e2 < numVertices) {
            if (res[e1][e2] > 0 && !visited[e2]) {
                visited[e2] = true;
                q.push(e2);
            }
            e2++;
        }
    }
    
    minCut.clear();
    i = 0;
    while (i < numVertices) {
        if (visited[i]) {
            minCut.push_back(i);
        }
        i++;
    }
    
    return maxFlow;
}

Graph findCDS(const Graph& G, int h) {
    int numVertices = G.getNumVertices();
    cout << "Finding CDS for grapth with" << numVertices << " vertices for " << h << "-clique densest subgraph" << endl;
    
    if (numVertices <= 0) {
        cerr << "Graph empty" << endl;
        return G;
    }
    
    
    int maxDegree = G.getMaxDegreeClique(h);
    cout << "Maximum cliques degree: " << maxDegree << endl;
    
    if (maxDegree == 0) {
        cout << "No " << h << "-cliques found in the graph" << endl;
        return G;
    }
    
    const auto& hCliques = G.getHCliques(h);
    const auto& hMinus1Cliques = G.getHMinus1Cliques(h);
    
    if (hCliques.empty() || (h > 1 && hMinus1Cliques.empty())) {
        cout << "Not enough cliques found for analysis." << endl;
        return G;
    }
    

    double l = 0;
    double e1 = maxDegree;
    double precision = 1.0 / (numVertices * numVertices);
    
    vector<int> F;
    vector<int> bestF;
    F.reserve(numVertices);
    bestF.reserve(numVertices);
    
    double bestDensity = 0;
    
    int iterCount = 0;
    
    const int MAX_ITERATIONS = 20;
    
    try {
        while (e1 - l >= precision && iterCount < MAX_ITERATIONS) {
            iterCount++;
            double progress = (e1 - l) / maxDegree * 100.0;
            
            double alpha = (l + e1) / 2;
            
            cout << "\nBuilding flow network for α=" << alpha << "... " << flush;
            
            size_t maxCliquesToProcess = min(hMinus1Cliques.size(), numVertices > 10000000 ? (size_t)500 : (size_t)10000);
            size_t numNodes = 1 + numVertices + min(maxCliquesToProcess, (size_t)10000) + 1;
            
            if (numNodes > 10000000) {
                cout << "Too many nodes (" << numNodes << " nodes), using sampling approach." << endl;
                numNodes = 500000;  // Limit network size even more
            }
            
            vector<vector<pair<int, int>>> capacitySparse(numNodes);
            
            int s = 0;
            int t = numNodes - 1;
            int offsetVertex = 1;
            int offsetClique = offsetVertex + numVertices;
            
            int e2 = 0;
            while (e2 < numVertices) {
                int cap = G.cliqueDegree(e2, h);
                if (cap > 0) {
                    capacitySparse[s].push_back({offsetVertex + e2, cap});
                }
                e2++;
            }
            
            e2=0;
            while (e2 < numVertices) {
                capacitySparse[offsetVertex + e2].push_back({t, ceil(alpha * h)});
                e2++;
            }
            
            const size_t MAX_CLIQUES_TO_PROCESS = min(hMinus1Cliques.size(), 
                                                     numVertices > 10000000 ? (size_t)50000 : 
                                                     (numVertices > 1000000 ? (size_t)500000 : (size_t)1000000));
            
            size_t batchSize = 100;
            for (size_t i = 0; i < MAX_CLIQUES_TO_PROCESS && i < hMinus1Cliques.size(); i++) {
                if (offsetClique + i >= numNodes) break; 
                
                const auto& clique = hMinus1Cliques[i];
                
                for (int e2 : clique) {
                    if (e2 >= 0 && e2 < numVertices && offsetVertex + e2 < numNodes) {
                        capacitySparse[offsetClique + i].push_back({offsetVertex + e2, numeric_limits<int>::max()});
                    }
                }
                
                for (int e2 = 0; e2 < numVertices; e2 += 1) { 
                    if (find(clique.begin(), clique.end(), e2) != clique.end()) continue;
                    
                    bool canExtend = true;
                    
                    const size_t MAX_CHECKS = min(clique.size(), (size_t)50); 
                    for (size_t j = 0; j < MAX_CHECKS; j++) {
                        int e1 = clique[j];
                        if (!G.hasEdge(e2, e1)) {
                            canExtend = false;
                            break;
                        }
                    }
                    
                    if (canExtend && offsetVertex + e2 < numNodes && offsetClique + i < numNodes) {
                        capacitySparse[offsetVertex + e2].push_back({offsetClique + i, 1});
                    }
                }
                
                if (i % batchSize == batchSize - 1) {
                    vector<int> tempVector(100);
                    vector<int>().swap(tempVector);

                    if (i > 1000) {
                        for (size_t j = i - 1000; j < i - 500; j++) {
                            if (j < capacitySparse.size() && offsetClique + j < numNodes) {
                                vector<pair<int, int>>().swap(capacitySparse[offsetClique + j]);
                            }
                        }
                    }
                }
            }
        
            vector<vector<int>> capacity;
            
            if (numNodes > 50000) {
                cout << "Using sparse max-flow implementation for large network..." << endl;
            
                unordered_set<int> activeNodes;
                activeNodes.insert(s);
                activeNodes.insert(t);
                
            
                for (int e1 = 0; e1 < numNodes; e1++) {
                    if (!capacitySparse[e1].empty()) {
                        activeNodes.insert(e1);
                        for (const auto& edge : capacitySparse[e1]) {
                            activeNodes.insert(edge.first);
                        }
                    }
                }
                
        
                unordered_map<int, int> nodeMap;
                int idx = 0;
                for (int node : activeNodes) {
                    nodeMap[node] = idx++;
                }
                
                try {
                    cout << "Creating compact capacity matrix with " << idx << " nodes..." << endl;
                    capacity.resize(idx);
                    for (int i = 0; i < idx; i++) {
                        capacity[i].resize(idx, 0);
                    }
                    
                    for (int e1 = 0; e1 < numNodes; e1++) {
                        if (activeNodes.find(e1) != activeNodes.end()) {
                            for (const auto& edge : capacitySparse[e1]) {
                                if (activeNodes.find(edge.first) != activeNodes.end()) {
                                    capacity[nodeMap[e1]][nodeMap[edge.first]] = edge.second;
                                }
                            }
                        }
                    }
                    
                    s = nodeMap[s];
                    t = nodeMap[t];
                }
                catch (const bad_alloc& ba) {
                    cout << "Memory allocation failed for capacity matrix: " << ba.what() << endl;
                    cout << "Trying with smaller network..." << endl;
                   
                    if (activeNodes.size() > 10000) {
                        vector<int> activeNodesList(activeNodes.begin(), activeNodes.end());
                        random_shuffle(activeNodesList.begin(), activeNodesList.end());
                        activeNodes.clear();
                
                        activeNodes.insert(s);
                        activeNodes.insert(t);
                        for (int i = 0; i < min((size_t)10000, activeNodesList.size()); i++) {
                            activeNodes.insert(activeNodesList[i]);
                        }
                
                        nodeMap.clear();
                        idx = 0;
                        for (int node : activeNodes) {
                            nodeMap[node] = idx++;
                        }
                        
                        capacity.clear();
                        capacity.resize(idx, vector<int>(idx, 0));
                        
                        for (int e1 = 0; e1 < numNodes; e1++) {
                            if (activeNodes.find(e1) != activeNodes.end()) {
                                for (const auto& edge : capacitySparse[e1]) {
                                    if (activeNodes.find(edge.first) != activeNodes.end()) {
                                        capacity[nodeMap[e1]][nodeMap[edge.first]] = edge.second;
                                    }
                                }
                            }
                        }
                        
                        s = nodeMap[s];
                        t = nodeMap[t];
                    }
                }
            } else {

                try {
                    capacity.resize(numNodes, vector<int>(numNodes, 0));
                    for (int e1 = 0; e1 < numNodes; e1++) {
                        for (const auto& edge : capacitySparse[e1]) {
                            capacity[e1][edge.first] = edge.second;
                        }
                    }
                }
                catch (const bad_alloc& ba) {
                    cout << "Memory allocation failed for capacity matrix: " << ba.what() << endl;
                    size_t reducedSize = numNodes / 2;
                    cout << "Trying with reduced size: " << reducedSize << endl;
                    
                    capacity.clear();
                    capacity.resize(reducedSize, vector<int>(reducedSize, 0));
                    
                    for (int e1 = 0; e1 < reducedSize; e1++) {
                        if (e1 < capacitySparse.size()) {
                            for (const auto& edge : capacitySparse[e1]) {
                                if (edge.first < reducedSize) {
                                    capacity[e1][edge.first] = edge.second;
                                }
                            }
                        }
                    }
                }
            }
            
            for (auto& e2 : capacitySparse) {
                vector<pair<int, int>>().swap(e2);
            }
            vector<vector<pair<int, int>>>().swap(capacitySparse);
            
            vector<int> minCut;
            try {
                fordFulkerson(capacity, s, t, minCut);
            }
            catch (const exception& e) {
                cout << "Error in Ford-Fulkerson: " << e.what() << endl;
                continue;
            }
            
            for (auto& e2 : capacity) {
                vector<int>().swap(e2);
            }
            vector<vector<int>>().swap(capacity);
            
            if (minCut.size() <= 1) {
                e1 = alpha;
                cout << "Cut contains only source. Reducing upper bound to " << e1 << endl;
            } else {
                l = alpha;
                

                F.clear();
                for (int node : minCut) {
                    if (node != s && node >= offsetVertex && node < offsetClique) {
                        int originalVertex = node - offsetVertex;
                        if (originalVertex >= 0 && originalVertex < numVertices) {
                            F.push_back(originalVertex);
                        }
                    }
                }
                
                if (!F.empty()) {
                    if (F.size() < 1000) {
                        Graph subgraph = G.findIndSubgraph(F);
                        double density = subgraph.cliqueDensity(h);
                        if (density > bestDensity) {
                            bestDensity = density;
                            bestF = F;
                        }
                        cout << "Cut contains " << F.size() << " vertices with density " << density << ". Increasing lower bound to " << l << endl;
                    } else {
                        bestF = F;
                        cout << "Cut contains " << F.size() << " vertices. Increasing lower bound to " << l << endl;
                    }
                }
            }
            
            vector<int>().swap(minCut);
            
            if (iterCount % 2 == 0) {
                vector<int> dummy(1000);
                vector<int>().swap(dummy);
            }
        }
    }
    catch (const bad_alloc& ba) {
        cout << "Memory allocation failed: " << ba.what() << endl;
        cout << "Trying to continue with reduced precision..." << endl;
        
        if (bestF.empty() && !F.empty()) {
            bestF = F;
        }
    }
    catch (const exception& e) {
        cout << "Error during binary search: " << e.what() << endl;
        cout << "Using best subgraph found so far..." << endl;
    }
    
    cout << "\nBinary search complete. Final density estimate: " << l << endl;
    
    if (!bestF.empty()) {
        return G.findIndSubgraph(bestF);
    } else if (!F.empty()) {
        return G.findIndSubgraph(F);
    } else {
        return G;
    }
}
int main(int argc, char* argv[]) {
    try {
        ifstream inputFile;
        string path = "input.txt";
        if (argc > 1) {
            path = argv[1];
        } else {
            cout << "No path provided" << endl;
        }
        
        cout <<"Reading file " << path << endl;

        inputFile.open(path);
        
        int numVertices, numEdges, h;
        inputFile >> numVertices >> numEdges >> h;
        
        if (numVertices <= 0 || numEdges < 0 || h <= 0) {
            cout << "Invalid parameters" << endl;
            return 1;
        }
        
        if (numVertices > 10000000) {
            cout << "Number of vertices too large (greater than 10,000,000)" << endl;
            return 1;
        }
        
        //Creating a graph with numVertices
        Graph G(numVertices);
    
        int invalidEdges = 0;
        for (int i = 0; i < numEdges; i++) {
            int e1, e2;
            if (!(inputFile >> e1 >> e2)) {
                cerr << "Error reading edge"<< endl;
                break;
            }
            
            if (e1 < 0 || e1 >= numVertices || e2 < 0 || e2 >= numVertices) {
                invalidEdges++;
                if (invalidEdges < 10) {
                    cerr << "Warning: Invalid edge (" << e1 << ", " << e2 << ")" << endl;
                }
                continue;
            }
            
            G.addEdge(e1, e2);
        }
        inputFile.close();
        
        if (invalidEdges > 0) {
            cerr << "Warning: " << invalidEdges << " invalid edges were ignored" << endl;
        }
        
        
        cout << "Graph with " << numVertices << " vertices, " << numEdges << " edges created." << endl;
        
        if (numVertices > 10000 && h > 3) {
            cout << "Very large graph(" << numVertices << " vertices). Using h=3 instead of " << h << endl;
            h = 3;
        }
        
        cout << "Looking for " << h << "-clique densest subgraph..." << endl;
        
        auto startTime = chrono::high_resolution_clock::now();
    
        Graph F = findCDS(G, h);
        
        auto endTime = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
        
        cout << "\nCompleted: " << duration << " seconds" << endl;
        cout << "CDS found with " << F.getNumVertices() << " vertices" << endl;
        
        if (F.getNumVertices() < 1000000) {
            cout << "Number of " << h << "-cliques in CDS: " << F.countCliques(h) << endl;
            cout << h << "-clique density of CDS: " << F.cliqueDensity(h) << endl;
        } else {
            cout << "Large subgraph, skipping analysis to save memory." << endl;
        }
    }
    catch (const bad_alloc& ba) {
        cerr << "Fatal memory error: " << ba.what() << endl;
        cerr << "Try smaller graph or lower value of h" << endl;
        return 1;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown error occurred" << endl;
    }
    
    return 0;
}