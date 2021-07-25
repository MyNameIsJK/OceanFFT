#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <helper_math.h>

__device__ float2 cExp(float f)
{
	return make_float2(cosf(f), sinf(f));
}
__global__ void generateWnkKernel(float2* wnk, int arrayLen)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < arrayLen)
	{
		// ÕÒ³öN
		int n = 2;
		while (n < (idx + 2))
		{
			n <<= 1;
		}
		int k = idx + 1 - n / 2;
		wnk[idx] = cExp(2.0f * CUDART_PI_F * (float)k / (float)n);
	}
}
extern "C" void generateWnk (float2 * wnk, int arrayLen, dim3 blockSize, dim3 gridSize)
{
	generateWnkKernel << <gridSize, blockSize >> > (wnk, arrayLen);
}