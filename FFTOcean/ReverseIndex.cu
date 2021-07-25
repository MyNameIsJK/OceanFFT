#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_math.h>

// binaryLen表示要转换的二进制长度
__device__ int reverseDev(uint index, uint binaryLen)
{
	uint ret = 0;
	for (int i = 0; i < binaryLen; i++)
	{
		ret = (ret * 2) + (index & 1);
		index /= 2;
	}
	return ret;
}

__global__ void reverseIndexKernel(uint* reverseIndex, uint arrayLen, uint binaryLen)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < arrayLen)
	{
		reverseIndex[idx] = reverseDev(idx, binaryLen);
	}
}

extern "C" void generateReverseIndex(uint * reverseIndex, uint arrayLen, uint binaryLen, dim3 blockSize, dim3 gridSize)
{
	reverseIndexKernel << <gridSize, blockSize >> > (reverseIndex, arrayLen, binaryLen);
}