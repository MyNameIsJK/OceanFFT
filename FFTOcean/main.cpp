#include <helper_math.h>
#include <string>
#include "OceanFFT.h"
using namespace std;
int main()
{
	float durationTime = 0.0f;

	const float PI = 3.1415926f;
	const float A = 1e-7f;
	float windSpeed = 100.0f;
	float windDir = PI / 3.0f;
	float dirDepend = 0.07f;

	int winWidth = 800;
	int winHeight = 600;
	string winName = "Ocean";
	MyOpenGL myGL;
	Shader* myShader;
	GLFWwindow* window;
	uint VAO, VBO, EBO;
	myGL.initGLFW();
	window = myGL.createWindow(winWidth, winHeight, winName);
	myGL.initGLAD();
	myShader = new Shader("./shader/Ocean.vert", "./shader/Ocean.Frag");
	array<string, 6>cubeMapPath = {
		"./skybox/right.jpg",
		"./skybox/left.jpg",
		"./skybox/top.jpg",
		"./skybox/bottom.jpg",
		"./skybox/front.jpg",
		"./skybox/back.jpg"
	};

	uint meshSize = 256;
	OceanFFT ocean(make_int2(meshSize), windSpeed, windDir, A, dirDepend);
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	ocean.bufferVertexData(VAO, VBO);
	ocean.bufferIndexData(EBO, VAO);
	//draw(M, N, vertices);
	glEnable(GL_DEPTH_TEST);
	cout << "初始化成功" << endl;
	glm::vec3 translateVec;
	glm::vec3 rotateVec;
	glm::mat4 model;
	glm::mat4 E(1.0f);
	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 proj = glm::perspective(glm::radians(90.0f), (float)winWidth / (float)winHeight, 0.1f, 10.0f);
	unsigned int skyboxTexture;
	myGL.createCubeMapFromFile(cubeMapPath, skyboxTexture);
	myShader->use();
	myShader->setVec4("deepColor", glm::vec4(0.0f, 0.1f, 0.4f, 1.0f));
	myShader->setVec4("skyColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
	myShader->setVec3("lightDir", glm::vec3(0.0f, 1.0f, 0.0f));
	myShader->setVec3("eyePos", glm::vec3(0.0f, 0.0f, 5.0f));

	while (!glfwWindowShouldClose(window))
	{
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//draw(M, N, vertices);
		durationTime += 0.1f;
		ocean.update(durationTime);
		ocean.bufferVertexData(VAO, VBO);
		translateVec = myGL.getTrasVec();
		rotateVec = myGL.getRotateVec();
		model = translate(E, translateVec);
		model = glm::rotate(model, glm::radians(rotateVec.x), glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, glm::radians(rotateVec.y), glm::vec3(0.0f, 1.0f, 0.0f));
		myShader->use();
		glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);
		myShader->setMat4("transform", proj * view * model);
		glDrawElements(GL_TRIANGLES, (meshSize - 1) * (meshSize - 1) * 6, GL_UNSIGNED_INT, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glfwTerminate();
	return 0;
}