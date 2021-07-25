#pragma once
#ifndef VERTEX_H
#define VERTEX_H
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
struct Vertex
{
	float3 pos;
	float3 norm;
};

#endif