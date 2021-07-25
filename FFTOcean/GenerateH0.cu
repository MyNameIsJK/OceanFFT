#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <math_constants.h>
#include <stdlib.h>


float urand()
{
	return rand() / (float)RAND_MAX;
}
// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss()
{
	float u1 = urand();
	float u2 = urand();

	if (u1 < 1e-6f)
	{
		u1 = 1e-6f;
	}

	return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
}
void randomComplex(int2 oceanSize, float2* randomComplex)
{
	for (int i = 0; i < oceanSize.x; i++)
	{
		for (int j = 0; j < oceanSize.y; j++)
		{
			randomComplex[j * oceanSize.x + i].x = gauss();
			randomComplex[j * oceanSize.x + i].y = gauss();
		}
	}
}

__device__ float phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend)
{
	float k_squared = Kx * Kx + Ky * Ky;

	if (k_squared == 0.0f)
	{
		return 0.0f;
	}
	float g = 9.81f;
	// largest possible wave from constant wind of velocity v
	float L = V * V / g;

	float k_x = Kx / sqrtf(k_squared);
	float k_y = Ky / sqrtf(k_squared);
	float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

	float phillips = A * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

	// filter out waves moving opposite to wind
	if (w_dot_k < 0.0f)
	{
		phillips *= dir_depend;
	}

	// damp out waves with very small length w << l
	//float w = L / 10000;
	//phillips *= expf(-k_squared * w * w);

	return phillips;
}
__global__ void generating(float Vdir, float V, float A, float dir_depend,
	float2* h0, int2 oceanMeshSize, float2*randomComplex)
{
	int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	int idxy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idxx < oceanMeshSize.x - 1 && idxy < oceanMeshSize.y - 1)
	{
		uint basicIndex = idxy * oceanMeshSize.x + idxx;
		float Kx = (-(float)oceanMeshSize.x / 2.0f + (float)idxx) * (2.0f * CUDART_PI_F / (float)oceanMeshSize.x);
		float Ky = (-(float)oceanMeshSize.y / 2.0f + (float)idxy) * (2.0f * CUDART_PI_F / (float)oceanMeshSize.y);
		float P = sqrtf(phillips(Kx, Ky, Vdir, V, A, dir_depend));
		if (Kx == 0.0f && Ky == 0.0f)
			P = 0.0f;
		h0[basicIndex] = randomComplex[basicIndex] * P * CUDART_SQRT_HALF_F;
	}
}
extern "C" void generateH0(float2 * h0Dev, float Vdir, float V, float A, float dir_depend, 
	int2 oceanMeshSize, dim3 gridSize,dim3 blockSize)
{
	int numVertex = oceanMeshSize.x * oceanMeshSize.y;
	float2* complexHost = new float2[numVertex];
	randomComplex(oceanMeshSize, complexHost);

	float2* complexDev;
	checkCudaErrors(cudaMalloc((void**)&complexDev, numVertex * sizeof(float2)));
	checkCudaErrors(cudaMemcpy(complexDev, complexHost, numVertex * sizeof(float2), cudaMemcpyHostToDevice));

	generating << <gridSize, blockSize >> > (Vdir, V, A, dir_depend, h0Dev, oceanMeshSize, complexDev);

	cudaFree(complexDev);
	delete[]complexHost;
}
