__kernel void invertFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		out[i * 4] = 255 - in[i * 4];
		out[i * 4 + 1] = 255 - in[i * 4 + 1];
		out[i * 4 + 2] = 255 - in[i * 4 + 2];
		out[i * 4 + 3] = 255 - in[i * 4 + 3];
	}
}

__kernel void grayFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		unsigned char gray = (in[i * 4] + in[i * 4 + 1] + in[i * 4 + 2]) / 3;

		out[i * 4] = gray;
		out[i * 4 + 1] = gray;
		out[i * 4 + 2] = gray;
		out[i * 4 + 3] = in[i * 4 + 3];
	}
}

__kernel void grayToBinaryFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		unsigned char gray = (in[i * 4] + in[i * 4 + 1] + in[i * 4 + 2]) / 3;

		if (gray > 100) {		// 100 is adjustment value
			gray = 255;
		}

		out[i * 4] = gray;
		out[i * 4 + 1] = gray;
		out[i * 4 + 2] = gray;
		out[i * 4 + 3] = in[i * 4 + 3];
	}
}

__kernel void acosFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		float p1 = in[i * 4] + in[i * 4 + 1] + in[i * 4 + 2] + in[i * 4 + 3];
		p1 /= 220;				// 220 is adjustment value
		p1 = acos(p1);
		p1 *= 220;

		out[i * 4] = p1;
		out[i * 4 + 1] = p1;
		out[i * 4 + 2] = p1;
		out[i * 4 + 3] = p1;
	}
}

__kernel void sepiaFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		unsigned char b = in[i * 4];
		unsigned char g = in[i * 4 + 1];
		unsigned char r = in[i * 4 + 2];

		unsigned char sepiaR = add_sat( add_sat((unsigned char)((float)r * 0.393), (unsigned char)((float)g * 0.769)), (unsigned char)((float)b * 0.189) );	
		unsigned char sepiaG = add_sat( add_sat((unsigned char)((float)r * 0.349), (unsigned char)((float)g * 0.686)), (unsigned char)((float)b * 0.168) );
		unsigned char sepiaB = add_sat( add_sat((unsigned char)((float)r * 0.272), (unsigned char)((float)g * 0.534)), (unsigned char)((float)b * 0.131) );
		
		out[i * 4] = sepiaB;
		out[i * 4 + 1] = sepiaG;
		out[i * 4 + 2] = sepiaR;
		out[i * 4 + 3] = in[i * 4 + 3];
	}
}

__kernel void gaussianBlurFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		__global float* blurMask,
		__local float* sharedMask,
		int width,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		float b = 0;
		float g = 0;
		float r = 0;
		
		int channel;

/*
		// really slows down the kernel if we use barrier
		for(int x = 0; x < 7; ++x) {
			for(int y = 0; y < 7; ++y) {
				sharedMask[x * 7 + y] = blurMask[x * 7 + y];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
*/


		for(int x = -3; x <= 3; ++x) {
			for(int y = -3; y <= 3; ++y) {

				channel = (i + x * width + y) * 4;
				if (channel < count) {
					sharedMask[(x + 3) * 3 + y + 3] = blurMask[(x + 3) * 3 + y + 3];

					b += (float)in[channel]		* sharedMask[(x + 3) * 3 + y + 3];
					g += (float)in[channel + 1] * sharedMask[(x + 3) * 3 + y + 3];
					r += (float)in[channel + 2] * sharedMask[(x + 3) * 3 + y + 3];
				}
			}
		}

		out[i * 4] = b;
		out[i * 4 + 1] = g;
		out[i * 4 + 2] = r;
	}
}

__kernel void redChannelFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		out[i * 4] = 0;
		out[i * 4 + 1] = 0;
		out[i * 4 + 2] = in[i * 4 + 2];
		out[i * 4 + 3] = 255;
	}
}

__kernel void greenChannelFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		out[i * 4] = 0;
		out[i * 4 + 1] = in[i * 4 + 1];
		out[i * 4 + 2] = 0;
		out[i * 4 + 3] = 255;
	}
}

__kernel void blueChannelFilterKernel(
		__global const unsigned char* in,
		__global unsigned char* out,
		unsigned int count)
{
	int i = get_global_id(0);
	if(i < count) {
		out[i * 4] = in[i * 4];
		out[i * 4 + 1] = 0;
		out[i * 4 + 2] = 0;
		out[i * 4 + 3] = 255;
	}
}