#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>
#include <array>
#include "Shader.h"
using namespace std;


class MyOpenGL
{
public:
	MyOpenGL() = default;
	~MyOpenGL() = default;
	MyOpenGL(const MyOpenGL& mo) = delete;
	MyOpenGL& operator = (const MyOpenGL& mo) = delete;
	//void mouseMoveCallBack(GLFWwindow* window, double xpos, double ypos);
	//void mouseScrollCallBack(GLFWwindow* window, double xoffset, double yoffset);
	void processInput(GLFWwindow* window);
	void initGLFW();
	GLFWwindow* createWindow(int winWidth, int winHeight, string winName);
	bool initGLAD();
	bool createTexture(string imgPath, unsigned int& texture);
	bool createCubeMapFromFile(array<string, 6>& cubeMapPath, unsigned int& texture);
	glm::vec3 getRotateVec();
	glm::vec3 getTrasVec();
};

