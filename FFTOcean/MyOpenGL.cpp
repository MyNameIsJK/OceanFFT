#include "MyOpenGL.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool leftMousePress = false;
bool rightMousePress = false;
glm::vec3 translateVec = glm::vec3(0.0f, 0.0f, -2.0f);
glm::vec3 rotateVec = glm::vec3(20.0f, 0.0f, 0.0f);
double lastMouseX = 400;
double lastMouseY = 300;
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}
void mouseClickCallBack(GLFWwindow* window, int buttonId, int buttonState, int mod)
{
	if (buttonId == GLFW_MOUSE_BUTTON_LEFT)
	{
		if (buttonState == GLFW_PRESS)
		{
			leftMousePress = true;
			glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
		}
		else if (buttonState == GLFW_RELEASE)
			leftMousePress = false;

	}
	else if (buttonId == GLFW_MOUSE_BUTTON_RIGHT)
	{
		if (buttonState == GLFW_PRESS)
		{
			rightMousePress = true;
			glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
		}
		else if (buttonState == GLFW_RELEASE)
			rightMousePress = false;

	}
}
void mouseMoveCallBack(GLFWwindow* window, double xpos, double ypos)
{
	if (leftMousePress)
	{
		double dx = xpos - lastMouseX;
		double dy = ypos - lastMouseY;
		rotateVec.x += dy * 0.2f;
		rotateVec.y += dx * 0.2f;
	}
	else if (rightMousePress)
	{
		double dx = xpos - lastMouseX;
		double dy = ypos - lastMouseY;
		translateVec.z += dy * 0.01f;
	}
	lastMouseX = xpos;
	lastMouseY = ypos;
}
void MyOpenGL::processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void MyOpenGL::initGLFW()
{
	//初始化glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

GLFWwindow* MyOpenGL::createWindow(int winWidth, int winHeight, string winName)
{
	GLFWwindow* window;
	window = glfwCreateWindow(winWidth, winHeight, winName.c_str(), NULL, NULL);
	if (!window)
	{
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return nullptr;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouseMoveCallBack);
	glfwSetMouseButtonCallback(window, mouseClickCallBack);
	return window;
}

bool MyOpenGL::initGLAD()
{
	//初始化glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return false;
	}
	return true;
}

bool MyOpenGL::createTexture(string imgPath, unsigned int& texture)
{
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// load img data
	// load image, create texture and generate mipmaps
	int width, height, nrChannels;
	stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
											// The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
	unsigned char* data = stbi_load(imgPath.c_str(), &width, &height, &nrChannels, 0);
	if (data)
	{
		if (nrChannels == 3)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		else if (nrChannels == 4)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);
	return true;
}

glm::vec3 MyOpenGL::getRotateVec()
{
	return rotateVec;
}

glm::vec3 MyOpenGL::getTrasVec()
{
	return translateVec;
}
bool MyOpenGL::createCubeMapFromFile(array<string, 6>& cubeMapPath, unsigned int& texture)
{
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
	int width, height, nrChannels;
	unsigned char* data;
	for (unsigned int i = 0; i < cubeMapPath.size(); i++)
	{
		data = stbi_load(cubeMapPath[i].c_str(), &width, &height, &nrChannels, 0);
		glTexImage2D(
			GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
			0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
		);
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	return true;
}