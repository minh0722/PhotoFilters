#pragma once

#ifdef __APPLE__
#   include "OpenCL/cl.h"
#else
#include <stdlib.h>
#include "CL/cl.h"
#endif

#include <cstdio>
#include <vector>

#include <Magick++.h>
using namespace Magick;

enum FilterCommand {
	INVERT,
	GRAY,
	GRAYTOBINARY,
	ACOS,
	SEPIA,
	GAUSSIAN_BLUR,
	RED_CHANNEL,
	GREEN_CHANNEL,
	BLUE_CHANNEL,
	FILTER_COUNT
};

class Kernel {
public:
	Kernel(PixelPacket* pixels, unsigned int width, unsigned int height);
	~Kernel();

	void performKernel(FilterCommand filter);
	void printMenu();

private:
	void checkError(const char* file, int line, cl_int error);
	void initKernelNames();
	void deinitKernelNames();
	void setupGaussianBlur();

	cl_int err;
	cl_device_id device_id;
	cl_uint numPlatforms;
	cl_platform_id* platforms;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;
	cl_mem clMems[2];			// input and output arrays in device memory for our calculation
	size_t input, output;
	std::vector<char*> kernelNames;
	PixelPacket* pixels;
	unsigned int size;
	unsigned int width, height;

	/* GAUSSIAN BLUR */
	cl_mem gaussianBlurMask;
};

