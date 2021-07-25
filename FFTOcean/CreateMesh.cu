#include "Vertex.h"
#include <helper_cuda.h>
#include <helper_math.h>
__global__ void meshPosKernel(int2 oceanMeshSize, Vertex* vertex)
{
	int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	int idxy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idxx < oceanMeshSize.x && idxy < oceanMeshSize.y)
	{
		float posx = (float)idxx / (float)(oceanMeshSize.x - 1) * 2.0f - 1.0f;
		float posy = (float)idxy / (float)(oceanMeshSize.y - 1) * 2.0f - 1.0f;
		vertex[idxy * oceanMeshSize.x + idxx].pos = make_float3(posx, 0.0f, posy);
		vertex[idxy * oceanMeshSize.x + idxx].norm = make_float3(0.0f);
	}
}
__global__ void meshIndexKernel(int2 oceanMeshSize, uint* indices)
{
	int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	int idxy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idxx < oceanMeshSize.x - 1 && idxy < oceanMeshSize.y - 1)
	{
		uint basicIndex = idxy * oceanMeshSize.x + idxx;
		int cnt = 6 * basicIndex;
		indices[cnt++] = basicIndex;
		indices[cnt++] = basicIndex + oceanMeshSize.x;
		indices[cnt++] = basicIndex + 1;
		indices[cnt++] = basicIndex + 1;
		indices[cnt++] = basicIndex + oceanMeshSize.x;
		indices[cnt++] = basicIndex + oceanMeshSize.x + 1;
	}
}
extern "C" void createMesh(dim3 blockSize, dim3 gridSize, int2 oceanMeshSize, uint * indicesHost, Vertex * vertHost, Vertex*vertDev)
{
	uint* indexDev;
	uint numVertex = oceanMeshSize.x * oceanMeshSize.y;
	// 每个矩形由两个三角形即四个顶点六个索引表示
	uint numIndex = (oceanMeshSize.x - 1) * (oceanMeshSize.y - 1) * 6;

	checkCudaErrors(cudaMalloc((void**)&indexDev, numIndex * sizeof(uint)));

	meshPosKernel << <gridSize, blockSize >> > (oceanMeshSize, vertDev);
	meshIndexKernel << <gridSize, blockSize >> > (oceanMeshSize, indexDev);

	checkCudaErrors(cudaMemcpy(vertHost, vertDev, numVertex * sizeof(Vertex), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(indicesHost, indexDev, numIndex * sizeof(uint), cudaMemcpyDeviceToHost));

	cudaFree(indexDev);
}