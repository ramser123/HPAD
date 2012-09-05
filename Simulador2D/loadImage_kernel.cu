/*
	Device Query Driver API
	Global Memory 766705664 bytes
	Multiprocessor: 12 Multiprocessors X 8 Cores (96 Cores)
	Constant Memory: 65536 bytes
	Shared Memory per Block: 16384 bytes
	Registers available per block: 8192
	Warp Size: 32
	Maximum memory pitch: 2147483647 bytes
	Texture alignment: 256 bytes
	Clocl rate: 1.2GHz
	Max threads: 512
	Dimension maxima de un bloque: 512x512x64
	Dimension maxima de un grid: 65535x65535x1
	
	gridDim.x (2D) Constante, define la cantidad de bloques del Grid en cada dimensión.
	blockDim.x (3D) Define el bloque actual (en tres dimensiones) de trabajo. 
	threadIdx.x (3D) Define el Thread actual (en tres dimensiones) de trabajo.

	Importante, X es la dimension que mas rapido se mueve, y Z la mas 'lenta' cuando se ejecuta de manera sincronizada, aunque todo va en paralelo....
	*/

__global__ void Copy(
    TColor *dst,
    int imageW,
    int imageH
){
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if(ix < imageW && iy < imageH){
        float4 fresult = tex2D(texImage, x, y); //valores de 0 a 1
		dst[imageW * iy + ix] = make_color(fresult.x, fresult.y, fresult.z, 0);
    }
	/*CUPRINTF("\t[%i, %i] \t Bloques:[%d,%d] \tThreads: [%d,%d,%d]\n", ix, iy, blockIdx.x, blockIdx.y,threadIdx.x, threadIdx.y, threadIdx.z);*/
}

extern "C" void copy_Image(TColor *d_dst, int imageW, int imageH)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    Copy<<<grid, threads>>>(d_dst, imageW, imageH);
}