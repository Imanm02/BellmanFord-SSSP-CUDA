# BellmanFord SSSP in CUDA

Here's the CUDA Bellman-Ford algorithm implementation for the Single-Source Shortest Path (SSSP) problem

## Bellman-Ford Algorithm on CUDA: Code Walkthrough

### Headers and Namespaces

```cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include <fstream>
#include <climits>
#include <cuda_runtime.h>
#include <sstream>
#include <algorithm>
```

We start by including necessary header files:
- I/O operations (`<cstdio>`, `<iostream>`, `<fstream>`)
- Common data structures (`<vector>`)
- C++ utilities (`<climits>`, `<sstream>`, `<algorithm>`)
- CUDA-specific operations (`<cuda_runtime.h>`)

### CUDA Kernel

```cpp
__global__ void bellmanFordKernel(int* d_V, int* d_E, int* d_W, int* d_dist, int V, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < E) {
        int u = d_V[idx];
        int v = d_E[idx];
        int w = d_W[idx];
        
        if (d_dist[u] != INT_MAX/2 && d_dist[u] + w < d_dist[v]) {
            atomicMin(&d_dist[v], d_dist[u] + w);
        }
    }
}
```

This is the heart of the CUDA implementation: the kernel. Here's what each part does:

1. Calculate the global thread index with `int idx = blockIdx.x * blockDim.x + threadIdx.x;`.
2. Each thread is responsible for relaxing one edge. Thus, check if the current thread's index is within bounds (`idx < E`).
3. Fetch the source vertex (`u`), destination vertex (`v`), and weight (`w`) for the edge the thread is responsible for.
4. If the potential new distance to vertex `v` through vertex `u` is shorter, update it using `atomicMin`.

### Helper Functions

#### Writing to File

```cpp
void writeToFile(const vector<int> &distances, const char *filename) {
    ofstream outfile(filename);
    for (int distance : distances) {
        outfile << distance << endl;
    }
    outfile.close();
}
```

This function writes the calculated shortest-path distances to an output file. It simply iterates through the distances vector and writes each distance to a new line.

#### Reading from File

```cpp
void readGraphFromFile(const char *filename, vector<int> &V, vector<int> &E, vector<int> &W) {
    ifstream infile(filename);
    string line;
    int u, v, w;
    char c;

    while (getline(infile, line)) {
        stringstream ss(line);
        ss >> c;
        
        if (c == 'a') {
            ss >> u >> v >> w;
            V.push_back(u - 1);
            E.push_back(v - 1);
            W.push_back(w);
        }
    }

    infile.close();
}
```

This function reads the graph data from a `.gr` file format. It differentiates edge data lines by checking the starting character (`a`). It then pushes the vertices and weights to their respective vectors.

### Main Function

Inside the `main` function:

1. **Reading the Graph**:

    We declare vectors to hold vertices (`h_V`), edges (`h_E`), and weights (`h_W`), and then we read the graph data from a file.

    ```cpp
    vector<int> h_V, h_E, h_W;
    readGraphFromFile("USA-road-d.CAL.gr", h_V, h_E, h_W);
    ```

2. **Initialize Distances**:

    The number of vertices (`V`) and edges (`E`) are determined, and a vector to hold distances (`h_dist`) is initialized. The source vertex (vertex 0) is initialized with a distance of 0, while others are initialized to half the maximum possible value to avoid overflow.

    ```cpp
    int V = *max_element(h_V.begin(), h_V.end()) + 1;
    int E = h_V.size();
    vector<int> h_dist(V, INT_MAX/2);
    h_dist[0] = 0;
    ```

3. **Memory Allocation & Data Transfer to GPU**:

    Allocate GPU memory for vertices, edges, weights, and distances and then transfer the data from host to GPU.

    ```cpp
    int *d_V, *d_E, *d_W, *d_dist;
    cudaMalloc(&d_V, E * sizeof(int));
    // ... (similar for d_E, d_W, d_dist)
    cudaMemcpy(d_V, h_V.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    // ... (similar for d_E, d_W, d_dist)
    ```
    
4. **Kernel Execution**:

    The Bellman-Ford algorithm is iterative, and the kernel is launched `V-1` times. Each iteration aims to further relax the edges.

    ```cpp
    int threadsPerBlock = 256;
    int blocksPerGrid = (E + threadsPerBlock - 1) / threadsPerBlock;
    for(int i = 0; i < V - 1; ++i) {
        bellmanFordKernel<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_E, d_W, d_dist, V, E);
        cudaDeviceSynchronize();
    }
    ```

5. **Retrieve Results & Cleanup**:

    We then copy the shortest path distances back to the host memory from the GPU and write these distances to an output file. Lastly, we free up the GPU memory.

    ```cpp
    cudaMemcpy(h_dist.data(), d_dist, V * sizeof(int), cudaMemcpyDeviceToHost);
    writeToFile(h_dist, "shortest_path_distances.txt");
    cudaFree(d_V);
    // ... (similar for d_E, d_W, d_dist)
    ```

And that wraps up our CUDA implementation of the Bellman-Ford algorithm!

## Maintainer

- [Iman Mohammadi](https://github.com/Imanm02)
