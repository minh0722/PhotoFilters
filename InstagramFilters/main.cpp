#define _CRT_SECURE_NO_WARNINGS

#include <Magick++.h>
#include <SDL.h>
#include <iostream>
#include <cstdint>
#include <ctime>
#include "Kernel.h"
using namespace Magick;

#define WIDTH 700
#define HEIGHT 700

/** Magick++
*	http://www.imagemagick.org/download/
*   SDL
*   https://www.libsdl.org/download-2.0.php
*/

SDL_Window* window = nullptr;
SDL_Surface* windowSurface = nullptr;
SDL_Surface* displaySurface = nullptr;

bool initSDL() {
	bool success = true;

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cout << "Can't init SDL! SDL_Error: " << SDL_GetError() << std::endl;
		success = false;
	}
	else {
		window = SDL_CreateWindow(
			"Instagram filters",
			SDL_WINDOWPOS_UNDEFINED,
			SDL_WINDOWPOS_UNDEFINED,
			WIDTH,
			HEIGHT,
			SDL_WINDOW_SHOWN);

		if (window == NULL) {
			std::cout << "Can't init SDL! SDL_Error: " << SDL_GetError() << std::endl;
			success = false;
		}
		else {
			windowSurface = SDL_GetWindowSurface(window);
		}
	}

	return success;
}

void deinitSDL() {
	SDL_FreeSurface(displaySurface);
	displaySurface = nullptr;

	//Destroy window
	SDL_DestroyWindow(window);
	window = NULL;

	//Quit SDL subsystems
	SDL_Quit();
}

void display(PixelPacket* imagepixels, size_t width, size_t height) {

	if (!displaySurface) {
		displaySurface = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
	}

	// Get pointer to pixels of surface.
	Uint32* surfacepixels = (Uint32*)displaySurface->pixels;

	// Copy image pixels to surface.
	Uint32 color;

	SDL_LockSurface(displaySurface);

	for (int row = 0; row < displaySurface->h; row++) {
		for (int column = 0; column < displaySurface->w; column++) {
			color = SDL_MapRGB(displaySurface->format,
						imagepixels->red,
						imagepixels->green,
						imagepixels->blue);

			*surfacepixels = color;

			// Increment pointers.
			surfacepixels++;
			imagepixels++;
		}
	}
	SDL_UnlockSurface(displaySurface);

	SDL_BlitScaled(displaySurface, nullptr, windowSurface, nullptr);

	// Update surface.
	SDL_UpdateWindowSurface(window);

	return;
}

int main(int argc, char** argv) {

	InitializeMagick("");
	initSDL();

	Image image;
	try {
		image.read("nature.jpg");

		size_t w = image.columns();
		size_t h = image.rows();

		PixelPacket* pixels = image.getPixels(0, 0, w, h);
		display(pixels, w, h);

		Kernel kernel(pixels, w, h);

		// processing input event
		SDL_Event inputEvent;
		bool quit = false;
		bool isValidSelection = false;
		clock_t begin, end;

		kernel.printMenu();
		while (!quit) {
			while (SDL_PollEvent(&inputEvent)) {
				switch (inputEvent.type) {
				case SDL_QUIT:
					quit = true;
					break;
				case SDL_TEXTINPUT:
					switch (inputEvent.text.text[0]) {
					case '0':
						isValidSelection = true;
						kernel.printMenu();
						break;
					case '1':
						isValidSelection = true;
						begin = clock();

						kernel.performKernel(INVERT);
						
						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					case '2':
						isValidSelection = true;
						begin = clock();
						
						kernel.performKernel(GRAY);
						
						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					case '3':
						isValidSelection = true;
						begin = clock();
						
						kernel.performKernel(GRAYTOBINARY);
						
						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					case '4':
						isValidSelection = true;
						begin = clock();
						
						kernel.performKernel(ACOS);
						
						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					case '5':
						isValidSelection = true;
						begin = clock();
						
						kernel.performKernel(SEPIA);
						
						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					case '6':
						isValidSelection = true;
						begin = clock();

						kernel.performKernel(GAUSSIAN_BLUR);

						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					case '7':
						isValidSelection = true;
						begin = clock();

						kernel.performKernel(RED_CHANNEL);

						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					case '8':
						isValidSelection = true;
						begin = clock();

						kernel.performKernel(GREEN_CHANNEL);

						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					case '9':
						isValidSelection = true;
						begin = clock();

						kernel.performKernel(BLUE_CHANNEL);

						end = clock();
						printf("Kernel execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						break;
					}

					if (isValidSelection) {
						begin = clock();

						display(pixels, w, h);
						
						end = clock();
						printf("Display execution took: %f\n", double(end - begin) / CLOCKS_PER_SEC);
						isValidSelection = false;
					}
				}
			}
		}
		
		// Write the image to a file
		printf("Do you want to save it? (y/n): ");
		char answer;
		do{
			scanf("%c", &answer);
		} while (answer != 'y' && answer != 'n');

		if (answer == 'y') {
			image.write("result.jpg");
		}
	}
	catch (Exception &error_) {
		std::cout << "Caught exception: " << error_.what() << std::endl;
		return 1;
	}

	deinitSDL();

	return 0;
}