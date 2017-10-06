#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include "Kernel.h"
#include <fstream>
#include <cstdio>
#include <SDL.h>

#define CHECK_ERROR(X) checkError(__FILE__, __LINE__, X)

float d = 120;

float blurMask[7][7] = {
		{ 1/d, 1/d, 2/d, 2/d, 2/d, 1/d, 1/d },
		{ 1/d, 2/d, 2/d, 4/d, 2/d, 2/d, 1/d },
		{ 2/d, 2/d, 4/d, 8/d, 4/d, 2/d, 2/d },
		{ 2/d, 4/d, 8/d, 1/d, 8/d, 4/d, 2/d },
		{ 2/d, 2/d, 4/d, 8/d, 4/d, 2/d, 2/d },
		{ 1/d, 2/d, 2/d, 4/d, 2/d, 2/d, 1/d },
		{ 1/d, 1/d, 2/d, 2/d, 2/d, 1/d, 1/d }
};

Kernel::Kernel(PixelPacket* pixels, unsigned int width, unsigned int height) {
	err = CL_SUCCESS;

	numPlatforms = 0;
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	CHECK_ERROR(err);
	printf("Found %i platforms\n", (int)numPlatforms);
	if (numPlatforms == 0)
		exit(EXIT_FAILURE);

	platforms = new cl_platform_id[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, platforms, NULL);
	CHECK_ERROR(err);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	CHECK_ERROR(err);

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	CHECK_ERROR(err);

	commands = clCreateCommandQueue(context, device_id, 0, &err);
	CHECK_ERROR(err);

	// get the source code
	std::ifstream kernelFile("kernel.cl", std::ios::in | std::ios::binary);
	kernelFile.seekg(0, std::ios::end);
	size_t fileSize = kernelFile.tellg();
	kernelFile.seekg(0, std::ios::beg);

	char* kernelSource = new char[fileSize + 1];
	kernelFile.read(kernelSource, fileSize);
	kernelSource[fileSize] = '\0';
	kernelFile.close();

	program = clCreateProgramWithSource(context, 1, (const char **)& kernelSource, NULL, &err);
	CHECK_ERROR(err);

	// build the program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(EXIT_FAILURE);
	}
	delete[] kernelSource;

	initKernelNames();

	this->pixels = pixels;
	this->size = width * height;
	this->width = width;
	this->height = height;

	input = 0;
	output = 1;

	// allocate memory on the device
	clMems[input] = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		size * sizeof(unsigned char) * sizeof(PixelPacket),
		NULL,
		&err);
	CHECK_ERROR(err);

	clMems[output] = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		size * sizeof(unsigned char) * sizeof(PixelPacket),
		NULL,
		&err);
	CHECK_ERROR(err);

	// Copy data to the device memory
	err = clEnqueueWriteBuffer(
		commands,
		clMems[input],
		CL_TRUE,
		0,
		size * sizeof(unsigned char) * sizeof(PixelPacket),
		this->pixels,
		0,
		NULL,
		NULL);
	CHECK_ERROR(err);

	setupGaussianBlur();
}

void Kernel::setupGaussianBlur() {
	gaussianBlurMask = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		49 * sizeof(float),
		NULL,
		&err);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(
		commands,
		gaussianBlurMask,
		CL_TRUE,
		0,
		49 * sizeof(float),
		blurMask,
		0,
		NULL,
		NULL);
	CHECK_ERROR(err);
}

Kernel::~Kernel() {
	clReleaseProgram(program);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	delete[] platforms;

	deinitKernelNames();

	clReleaseMemObject(clMems[input]);
	clReleaseMemObject(clMems[output]);
}

void Kernel::performKernel(FilterCommand filter) {
	kernel = clCreateKernel(program, kernelNames[filter], &err);
	CHECK_ERROR(err);

	//Prepare to call the kernel
	//Set the arguments to our compute kernel
	err = 0;
	if (filter == FilterCommand::GAUSSIAN_BLUR) {
		this->size *= 4;
		err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clMems[input]);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &clMems[output]);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &gaussianBlurMask);
		err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), NULL);
		err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &width);
		err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &size);
		CHECK_ERROR(err);
		this->size /= 4;
	}
	else{
		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clMems[input]);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &clMems[output]);
		err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &size);
		CHECK_ERROR(err);
	}

	size_t global;  // number of thread blocks
	size_t local;   // thread block size

	//Get the maximum work group size for executing the kernel on the device
	err = clGetKernelWorkGroupInfo(
		kernel, 
		device_id, 
		CL_KERNEL_WORK_GROUP_SIZE, 
		sizeof(local), 
		&local, 
		NULL);
	CHECK_ERROR(err);

	// Execute the kernel over the entire range of our 1d input data set
	// using the maximum number of work group items for this device
	global = size;
	err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Wait for the commands in the queue to finish before reading back results
	clFinish(commands);

	// Read back the results from the device
	err = clEnqueueReadBuffer(
		commands,
		clMems[output],
		CL_TRUE,
		0,
		size * sizeof(unsigned char) * sizeof(PixelPacket),
		this->pixels,
		0,
		NULL,
		NULL);
	CHECK_ERROR(err);

	std::swap(input, output);

	//clReleaseMemObject(input);
	//clReleaseMemObject(output);
}

inline void Kernel::checkError(const char* file, int line, cl_int error) {
	if (error != CL_SUCCESS) {
		printf("error %i in file %s, line %i\n", error, file, line);
		exit(EXIT_FAILURE);
	}
}

void Kernel::initKernelNames() {
	kernelNames.resize(FILTER_COUNT);

	int len = strlen("invertFilterKernel") + 1;
	kernelNames[INVERT] = new char[len + 1];
	memcpy(kernelNames[INVERT], "invertFilterKernel", len);
	kernelNames[INVERT][len - 1] = '\0';

	len = strlen("grayFilterKernel") + 1;
	kernelNames[GRAY] = new char[len + 1];
	memcpy(kernelNames[GRAY], "grayFilterKernel", len);
	kernelNames[GRAY][len - 1] = '\0';

	len = strlen("grayToBinaryFilterKernel") + 1;
	kernelNames[GRAYTOBINARY] = new char[len + 1];
	memcpy(kernelNames[GRAYTOBINARY], "grayToBinaryFilterKernel", len);
	kernelNames[GRAYTOBINARY][len - 1] = '\0';

	len = strlen("acosFilterKernel") + 1;
	kernelNames[ACOS] = new char[len + 1];
	memcpy(kernelNames[ACOS], "acosFilterKernel", len);
	kernelNames[ACOS][len - 1] = '\0';

	len = strlen("sepiaFilterKernel") + 1;
	kernelNames[SEPIA] = new char[len + 1];
	memcpy(kernelNames[SEPIA], "sepiaFilterKernel", len);
	kernelNames[SEPIA][len - 1] = '\0';

	len = strlen("gaussianBlurFilterKernel") + 1;
	kernelNames[GAUSSIAN_BLUR] = new char[len + 1];
	memcpy(kernelNames[GAUSSIAN_BLUR], "gaussianBlurFilterKernel", len);
	kernelNames[GAUSSIAN_BLUR][len - 1] = '\0';

	len = strlen("redChannelFilterKernel") + 1;
	kernelNames[RED_CHANNEL] = new char[len + 1];
	memcpy(kernelNames[RED_CHANNEL], "redChannelFilterKernel", len);
	kernelNames[RED_CHANNEL][len - 1] = '\0';

	len = strlen("greenChannelFilterKernel") + 1;
	kernelNames[GREEN_CHANNEL] = new char[len + 1];
	memcpy(kernelNames[GREEN_CHANNEL], "greenChannelFilterKernel", len);
	kernelNames[GREEN_CHANNEL][len - 1] = '\0';

	len = strlen("blueChannelFilterKernel") + 1;
	kernelNames[BLUE_CHANNEL] = new char[len + 1];
	memcpy(kernelNames[BLUE_CHANNEL], "blueChannelFilterKernel", len);
	kernelNames[BLUE_CHANNEL][len - 1] = '\0';
}

void Kernel::deinitKernelNames() {
	for (size_t i = 0; i < FILTER_COUNT; ++i) {
		delete[] kernelNames[i];
	}
}

void Kernel::printMenu() {
	printf("Select filter effect number: \n");
	printf("1. Invert\n");
	printf("2. Grayscale\n");
	printf("3. Gray to binary\n");
	printf("4. Acos\n");
	printf("5. Sepia\n");
	printf("6. Gaussian blur\n");
	printf("7. Red channel\n");
	printf("8. Green channel\n");
	printf("9. Blue channel\n");
	printf("Select 0 for this menu\n");
}