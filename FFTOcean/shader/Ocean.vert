#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNorm;
//layout (location = 2) in vec3 aNorm;

out vec3 fragPos;
out vec3 fragNorm;
uniform mat4 transform;

void main()
{
	fragPos=aPos;
	fragNorm=aNorm;
	//fragNorm=vec3(0.0f,1.0f,0.0f);
	gl_Position = transform*vec4(fragPos, 1.0);
}