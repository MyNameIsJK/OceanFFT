#include "Vertex.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include <math_constants.h>

#include <cufft.h>
typedef unsigned int uint;
__device__ __host__ float2 conjugate(float2 originComplex)
{
	return make_float2(originComplex.x, -originComplex.y);
}
__device__ __host__ float2 complexAdd(float2 a, float2 b)
{
	return a + b;
}
__device__ __host__ float2 complexMul(float2 a, float2 b)
{
	return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + b.x * a.y);
}
__device__ __host__ float2 complexExp(float a)
{
	return make_float2(cosf(a), sinf(a));
}
__global__ void updateHtKernel(float2* h0, float2* ht, float t, int2 oceanMeshSize, int2 oceanRealSize)
{
	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxz = blockIdx.y * blockDim.y + threadIdx.y;
	if (idxx < oceanMeshSize.x && idxz < oceanMeshSize.y)
	{
		float2 K = (-make_float2(oceanMeshSize) / 2.0f + make_float2(idxx, idxz))
			* (2.0f * CUDART_PI_F / make_float2(oceanRealSize));
		int idx = idxz * oceanMeshSize.x + idxx;
		int mirrorIdx = (oceanMeshSize.y-1 - idxz) * oceanMeshSize.x + (oceanMeshSize.x-1 - idxx);
		float2 h = h0[idx];
		float2 hs = h0[mirrorIdx];
		float omega = sqrtf(length(K) * 9.81f);
		ht[idx] = complexAdd(complexMul(h, complexExp(omega * t)),
			complexMul(hs, complexExp(omega * t)));
	}
}
__global__ void fftRow(int2 oceanMeshSize, float2* height, float2* tmph,
	float2* wnk, uint* reverseIndex)
{
	int idxy = threadIdx.x + blockIdx.x * blockDim.x;
	// 对idxy行的数据进行fft
	if (idxy < (oceanMeshSize.y))
	{
		// 从height中去当前行的数据存到tmph中
		uint basicIndex = idxy * oceanMeshSize.x;
		for (int i = 0; i < oceanMeshSize.x; i++)
		{
			tmph[basicIndex + i] = height[basicIndex + reverseIndex[i]];
			//tmph[basicIndex + i] = make_float2(0.0f);
		}
		int m;
		float2 t;
		for (int l = 2; l <= oceanMeshSize.x; l *= 2)
		{
			m = l / 2;
			for (int i = 0; i < oceanMeshSize.x; i += l)
			{
				for (int j = i; j < i + m; j++)
				{
					//t = tmph[basicIndex + j + m];
					// m-1,l-2
					t = complexMul(wnk[j % m + m - 1], tmph[basicIndex + j + m]);
					//t = make_float2(0.0f);
					tmph[basicIndex + j + m] = tmph[basicIndex + j] - t;
					tmph[basicIndex + j] += t;
				}
			}
		}
	}
}
__global__ void fftColumn(int2 oceanMeshSize, float2* height, float2* tmph,
	float2* wnk, uint* reverseIndex)
{
	int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	// 对idxx列的数据进行fft
	if (idxx < (oceanMeshSize.x))
	{
		// 从height中去当前列的数据存到tmph中
		for (int i = 0; i < oceanMeshSize.y; i++)
		{
			tmph[idxx+i*oceanMeshSize.x] = height[idxx + reverseIndex[i] * oceanMeshSize.x];
		}
		int m;
		float2 t;
		for (int l = 2; l <= oceanMeshSize.y; l *= 2)
		{
			m = l / 2;
			for (int i = 0; i < oceanMeshSize.y; i += l)
			{
				for (int j = i; j < i + m; j++)
				{
					// m-1,l-2
					t = complexMul(wnk[j % m + m - 1], tmph[idxx + (j + m) * oceanMeshSize.x]);
					tmph[idxx + (j + m) * oceanMeshSize.x] = tmph[idxx + j * oceanMeshSize.x] - t;
					tmph[idxx + j * oceanMeshSize.x] += t;
				}
			}
		}
	}
}
__global__ void updateVertexKernel(Vertex* vertexDev, int2 meshSize, float2*height)
{
	int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	int idxy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idxx < (meshSize.x) && idxy < (meshSize.y))
	{
		uint basicIndex = idxy * meshSize.x + idxx;
		vertexDev[basicIndex].norm = make_float3(0.0f);
		float sign_correction = ((idxx + idxy) & 0x01) ? -1.0f : 1.0f;
		vertexDev[basicIndex].pos.y = height[basicIndex].x * sign_correction*0.1f;
		if ((idxx > 0 && idxx < meshSize.x - 1) && (idxy > 0 && idxy < meshSize.y - 1))
		{
			float3 xgrad = vertexDev[basicIndex + 1].pos - vertexDev[basicIndex - 1].pos;
			float3 zgrad = vertexDev[basicIndex + meshSize.x].pos - vertexDev[basicIndex - meshSize.x].pos;
			vertexDev[basicIndex].norm = normalize(cross(zgrad, xgrad));
		}
	}
}

extern "C" void updateVertex(float gameTime, float2 * h0, float2 * ht, float2 * tmph, Vertex * vertices,
	int2 oceanMeshSize, float2 * wnk, uint * reverseIndex,cufftHandle&fftPlan,
	dim3 gridSize, dim3 blockSize, dim3 gridRow, dim3 blockRow)
{
	updateHtKernel << <gridSize, blockSize >> > (h0, ht, gameTime, oceanMeshSize, oceanMeshSize);
	//fftRow << <gridRow, blockRow >> > (oceanMeshSize, ht, tmph, wnk, reverseIndex);
	//checkCudaErrors(cudaMemcpy(ht, tmph, oceanMeshSize.x * oceanMeshSize.y * sizeof(float2), cudaMemcpyDeviceToDevice));
	//fftColumn << <gridRow, blockRow >> > (oceanMeshSize, ht, tmph, wnk, reverseIndex);
	//checkCudaErrors(cudaMemcpy(ht, tmph, oceanMeshSize.x * oceanMeshSize.y * sizeof(float2), cudaMemcpyDeviceToDevice));
	cufftExecC2C(fftPlan, ht, ht, CUFFT_INVERSE);
	updateVertexKernel << <gridSize, blockSize >> > (vertices, oceanMeshSize, ht);
}
