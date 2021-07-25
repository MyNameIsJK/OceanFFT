#version 460 core

out vec4 FragColor;
in vec3 fragPos;
in vec3 fragNorm;

uniform vec4 deepColor;
uniform vec4 skyColor;
uniform vec3 lightDir;
uniform vec3 eyePos;
uniform samplerCube skybox;

void main()
{
	vec3 lightdir=normalize(lightDir);
	vec3 eyedir=normalize(eyePos-fragPos);
	vec3 fragnorm=normalize(fragNorm);
	float diffuse=max(0.0f,dot(lightdir,fragnorm));
	float facing=1.0f-max(0.0f,dot((eyedir+lightdir)/2.0f,fragnorm));
	float frenel=pow(facing,5.0f);
	vec4 waterColor=vec4(texture(skybox,reflect(eyedir,fragnorm).rgb,1.0f));
	FragColor=waterColor*diffuse+frenel*skyColor;
	//FragColor=frenel*skyColor;
	//FragColor=deepColor*diffuse;
	//FragColor=vec4(0.0f,0.0f,1.0f,1.0f);
}