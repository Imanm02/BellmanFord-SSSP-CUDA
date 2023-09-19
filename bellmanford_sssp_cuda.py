# -*- coding: utf-8 -*-
"""BellmanFord-SSSP-CUDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nZvKWlnrUEUkcfE5Lb0EoPnI_dCDZ9hI
"""

# Check GPU availability
!nvidia-smi

from google.colab import files
uploaded = files.upload()

# Commented out IPython magic to ensure Python compatibility.
# # Write the modified CUDA code to a file
# %%writefile bellman_ford_cuda.cu
# 
# #include <algorithm>
# #include <cstdio>
# #include <iostream>
# #include <vector>
# #include <fstream>
# #include <climits>
# #include <cuda_runtime.h>
# #include <sstream>
# 
# using namespace std;
# 
# // CUDA kernel to perform edge relaxation.
# __global__ void bellmanFordKernel(int* d_V, int* d_E, int* d_W, int* d_dist, int V, int E) {
#     int idx = blockIdx.x * blockDim.x + threadIdx.x;
#     if(idx < E) {
#         int u = d_V[idx];
#         int v = d_E[idx];
#         int w = d_W[idx];
# 
#         // Check for possible overflow and perform relaxation only if it's safe.
#         if (d_dist[u] != INT_MAX/2 && d_dist[u] + w < d_dist[v]) {
#             atomicMin(&d_dist[v], d_dist[u] + w);
#         }
#     }
# }
# 
# // Function to write shortest path distances to an output file.
# void writeToFile(const vector<int> &distances, const char *filename) {
#     ofstream outfile(filename);
#     for (int distance : distances) {
#         outfile << distance << endl;
#     }
#     outfile.close();
# }
# 
# // Function to read graph from a .gr file, handling its specific format.
# void readGraphFromFile(const char *filename, vector<int> &V, vector<int> &E, vector<int> &W) {
#     ifstream infile(filename);
#     string line;
#     int u, v, w;
#     char c;
# 
#     while (getline(infile, line)) {
#         stringstream ss(line);
#         ss >> c;  // Read the first character to determine the line type.
# 
#         if (c == 'a') { // Only process lines that describe edges.
#             ss >> u >> v >> w;
#             V.push_back(u - 1);  // Convert 1-based index to 0-based.
#             E.push_back(v - 1);  // Convert 1-based index to 0-based.
#             W.push_back(w);
#         }
#     }
# 
#     infile.close();
# }
# 
# int main() {
#     vector<int> h_V, h_E, h_W;
# 
#     // Read graph from .gr format.
#     readGraphFromFile("USA-road-d.CAL.gr", h_V, h_E, h_W);
# 
#     int V = *max_element(h_V.begin(), h_V.end()) + 1; // Number of vertices
#     int E = h_V.size(); // Number of edges
# 
#     vector<int> h_dist(V, INT_MAX/2); // Initialize distances.
#     h_dist[0] = 0; // Source vertex (0-based index).
# 
#     // Allocate device memory.
#     int *d_V, *d_E, *d_W, *d_dist;
#     cudaMalloc(&d_V, E * sizeof(int));
#     cudaMalloc(&d_E, E * sizeof(int));
#     cudaMalloc(&d_W, E * sizeof(int));
#     cudaMalloc(&d_dist, V * sizeof(int));
# 
#     // Copy data to device.
#     cudaMemcpy(d_V, h_V.data(), E * sizeof(int), cudaMemcpyHostToDevice);
#     cudaMemcpy(d_E, h_E.data(), E * sizeof(int), cudaMemcpyHostToDevice);
#     cudaMemcpy(d_W, h_W.data(), E * sizeof(int), cudaMemcpyHostToDevice);
#     cudaMemcpy(d_dist, h_dist.data(), V * sizeof(int), cudaMemcpyHostToDevice);
# 
#     int threadsPerBlock = 256;
#     int blocksPerGrid = (E + threadsPerBlock - 1) / threadsPerBlock;
# 
#     // Main loop for edge relaxation.
#     for(int i = 0; i < V - 1; ++i) {
#         bellmanFordKernel<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_E, d_W, d_dist, V, E);
#         cudaDeviceSynchronize();
#     }
# 
#     // Copy results back to host.
#     cudaMemcpy(h_dist.data(), d_dist, V * sizeof(int), cudaMemcpyDeviceToHost);
# 
#     // Write results to an output file.
#     writeToFile(h_dist, "shortest_path_distances.txt");
# 
#     // Free device memory.
#     cudaFree(d_V);
#     cudaFree(d_E);
#     cudaFree(d_W);
#     cudaFree(d_dist);
# 
#     return 0;
# }

!nvcc bellman_ford_cuda.cu -o bellman_ford_cuda
!./bellman_ford_cuda

# Read and print the output file.
!cat shortest_path_distances.txt