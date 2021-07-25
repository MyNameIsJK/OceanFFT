#include "OceanFFT.h"
#include <math_constants.h>
extern "C" void createMesh(dim3 blockSize, dim3 gridSize, int2 oceanMeshSize,
	uint * indicesHost, Vertex * vertHost, Vertex * vertDev);
extern "C" void generateReverseIndex(uint * reverseIndex,
	uint arrayLen, uint binaryLen, dim3 blockSize, dim3 gridSize);
extern "C" void generateH0(float2 * h0Dev, float Vdir, float V, float A, float dir_depend,
	int2 oceanMeshSize, dim3 gridSize, dim3 blockSize);
extern "C" void generateWnk(float2 * wnk, int arrayLen, dim3 blockSize, dim3 gridSize);

extern "C" void updateVertex(float gameTime, float2 * h0, float2 * ht, float2 * tmph, Vertex * vertices,
	int2 oceanMeshSize, float2 * wnk, uint * reverseIndex, cufftHandle & fftPlan,
	dim3 gridSize, dim3 blockSize, dim3 gridRow, dim3 blockRow);
int iDivUp(int x, int y)
{
	int ret = x / y;
	if (x % y != 0)
		ret++;
	return ret;
}

OceanFFT::OceanFFT(const int2& oceanMeshSize, float windSpeed, float windDir, float A, float dirDepend):
	oceanMeshSize(oceanMeshSize),windSpeed(windSpeed),windDir(windDir),A(A),dirDepend(dirDepend)
{
	numVertex = oceanMeshSize.x * oceanMeshSize.y;
	vertices = new Vertex[numVertex];
	numVertexBytes = numVertex * sizeof(Vertex);
	numIndex = (oceanMeshSize.x - 1) * (oceanMeshSize.y - 1) * 6;
	indices = new uint[numIndex];
	checkCudaErrors(cudaMalloc((void**)&verticesDev, numVertexBytes));
	checkCudaErrors(cudaMalloc((void**)&h0Dev, numVertex * sizeof(float2)));
	checkCudaErrors(cudaMalloc((void**)&htDev, numVertex * sizeof(float2)));
	checkCudaErrors(cudaMalloc((void**)&tmphDev, numVertex * sizeof(float2)));
	
	blockSize = dim3(256);
	gridSize = dim3(iDivUp(oceanMeshSize.x, 256));
	uint indexArrayLen = oceanMeshSize.x;
	uint binaryLen = 0;
	int x = indexArrayLen - 1;
	do
	{
		binaryLen++;
		x /= 2;
	} while (x != 0);
	checkCudaErrors(cudaMalloc((void**)&reverseIndexDev, indexArrayLen * sizeof(uint)));
	generateReverseIndex(reverseIndexDev,
		indexArrayLen, binaryLen, blockSize, gridSize);
	/*
	uint* ri = new uint[indexArrayLen];
	checkCudaErrors(cudaMemcpy(ri, reverseIndexDev, indexArrayLen * sizeof(uint), cudaMemcpyDeviceToHost));
	for (int i = 0; i < indexArrayLen; i++)
		cout << ri[i] << " ";
	cout << endl;
	*/
	cufftPlan2d(&fftPlan, oceanMeshSize.x , oceanMeshSize.y, CUFFT_C2C);
	uint wnkLen = oceanMeshSize.x - 1;
	gridSize = dim3(iDivUp(oceanMeshSize.x, 256));
	checkCudaErrors(cudaMalloc((void**)&wnkDev, wnkLen * sizeof(float2)));
	generateWnk(wnkDev, wnkLen, blockSize, gridSize);
	
	blockSize = dim3(16, 16);
	gridSize = dim3(iDivUp(oceanMeshSize.x, 16), iDivUp(oceanMeshSize.y, 16));
	createMesh(blockSize, gridSize, oceanMeshSize, indices, vertices, verticesDev);
	generateH0(h0Dev, windDir, windSpeed, A, dirDepend,
		oceanMeshSize, gridSize, blockSize);
	
	blockRow = dim3(256);
	gridRow = dim3(iDivUp(oceanMeshSize.x, 256));
}

OceanFFT::~OceanFFT()
{
	delete[]vertices;
	delete[]indices;
	cudaFree(verticesDev);
}

void OceanFFT::bufferVertexData(uint& VAO, uint& VBO)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, numVertexBytes, vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(float3)));
}

void OceanFFT::bufferIndexData(uint& EBO, uint& VAO)
{
	glBindVertexArray(VAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndex * sizeof(uint), indices, GL_STATIC_DRAW);
}

void OceanFFT::update(float t)
{
	updateVertex(t, h0Dev, htDev, tmphDev, verticesDev,
		oceanMeshSize, wnkDev, reverseIndexDev,fftPlan,
		gridSize, blockSize, gridRow, blockRow);
	checkCudaErrors(cudaMemcpy(vertices, verticesDev, numVertex * sizeof(Vertex), cudaMemcpyDeviceToHost));
}
