#pragma once
#ifndef OCEAN_FFT_H
#define OCEAN_FFT_H

#include "Vertex.h"
#include "MyOpenGL.h"
#include <helper_math.h>
#include <helper_cuda.h>
#include "cufft.h"

class OceanFFT
{
private:
	Vertex* vertices;
	uint* indices;

	int2 oceanMeshSize;
	float windSpeed;
	float windDir;
	float A;
	float dirDepend;

	uint numVertex;
	uint numIndex;
	size_t numVertexBytes;

	dim3 gridSize;
	dim3 blockSize;
	dim3 gridRow;
	dim3 blockRow;

	// device data
	Vertex* verticesDev;

	float2* h0Dev;
	float2* htDev;
	float2* tmphDev;
	float2* wnkDev;
	uint* reverseIndexDev;

	cufftHandle fftPlan;

public:
	OceanFFT(const int2& oceanMeshSize, float windSpeed, float windDir, float A, float dirDepend);
	OceanFFT() = delete;
	OceanFFT(const OceanFFT& offt) = delete;
	OceanFFT& operator = (const OceanFFT& offt) = delete;
	~OceanFFT();

	void bufferVertexData(uint& VAO, uint& VBO);
	void bufferIndexData(uint& EBO, uint& VAO);
	void update(float t);
};

#endif