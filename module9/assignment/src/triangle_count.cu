#include <nvgraph.h>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <chrono>


void check(nvgraphStatus_t status) {
	if(status != NVGRAPH_STATUS_SUCCESS) {
		std::cout << "[ERROR]: " << status << " in " << __FILE__ << " : " << __LINE__ << std::endl;
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char **argv) {
	// nvgraph variables
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCSRTopology32I_t CSR_input;

	// Init host data
	CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));

	// Undirected graph:
	// 0       2-------4       
	//  \     / \     / \
	//   \   /   \   /   \
	//    \ /     \ /     \
	//     1-------3-------5
	//      \     / \     /
	//       \   /   \   /
	//        \ /     \ /
	//         6 ----- 7
	// 6 triangles

	// CSR of lower triangular of adjacency matrix:
	const size_t no_vertices = 8, no_edges = 13;
	int source_offsets[] = {0, 0, 1, 2, 4, 6, 8, 10, 13};
	int destination_indices[] = {0, 1, 1, 2, 2, 3, 3, 4, 1, 3, 3, 5, 6};
	check(nvgraphCreate(&handle));
	check(nvgraphCreateGraphDescr(handle, &graph));
	CSR_input->nvertices = no_vertices; 
	CSR_input->nedges = no_edges;
	CSR_input->source_offsets = source_offsets;
	CSR_input->destination_indices = destination_indices;

	// Set graph connectivity
	check(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input, NVGRAPH_CSR_32));

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	// Count triangles
	uint64_t triangle_count = 0;
	check(nvgraphTriangleCount(handle, graph, &triangle_count));

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << "Time to count triangles [" << triangle_count << "] samples (" << 1000*diff.count() << ") ms\n";

	// Free resources
	free(CSR_input);
	check(nvgraphDestroyGraphDescr(handle, graph));
	check(nvgraphDestroy(handle));
	return EXIT_SUCCESS;
}
