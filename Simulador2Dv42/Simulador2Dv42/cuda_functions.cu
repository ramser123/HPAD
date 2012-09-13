/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



/*
 * This sample demonstrates two adaptive image denoising technqiues: 
 * KNN and NLM, based on computation of both geometric and color distance 
 * between texels. While both techniques are already implemented in the 
 * DirectX SDK using shaders, massively speeded up variation 
 * of the latter techique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulatorGL.h"

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture reference and channel descriptor for image texture
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

//CUDA array descriptor
cudaArray *a_Src;

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Max(float x, float y){
    return (x > y) ? x : y;
}

float Min(float x, float y){
    return (x < y) ? x : y;
}

int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float lerpf(float a, float b, float c){
    return a + (b - a) * c;
}

__device__ float vecLen(float4 a, float4 b){
    return (
        (b.x - a.x) * (b.x - a.x) +
        (b.y - a.y) * (b.y - a.y) +
        (b.z - a.z) * (b.z - a.z)
    );
}

__device__ TColor make_color(float r, float g, float b, float a){
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

__device__ int absMax(int a, int b){
	int c = (a>0) ? a : -a;
	int d = (b>0) ? b : -b;
	return (c > d) ? c : d;
}

__device__ int absi(int f){
	return (f > 0) ? f : -f;
}

__device__ float absf(float f){
	return (f > 0) ? f : -f;
}




/***********CALCULATE NEXT DESIRED JUMP**************/
__device__ bool getNextDirection(int id, int &x, int &y, int **devLocalX, int **devLocalY, int *devLocalStep, int *devNextX, int *devNextY, int type){

	float div = 0.2f;
	if(type != 1){
		div = 0.13f;
	}
	
	int nextTrasX=devLocalX[id][5*(devLocalStep[id])+4];
	int nextTrasY=devLocalY[id][5*(devLocalStep[id])+4];

	float disX=(float)(nextTrasX-devNextX[id]);
	float disY=(float)(nextTrasY-devNextY[id]);

	int maximo = absMax(disX,disY);
	if (maximo == 0){
		devLocalStep[id]++;
		return false;
	}
	float hyp =sqrt(disX*disX+disY*disY);

	if (absf(disX/hyp)>div){
		x = disX > 0 ? 1 : -1;
	}
	if (absf(disY/hyp)>div){
		y = disY > 0 ? 1 : -1;
	}
	return true;
}

//mirar como incluir a los peatones sin complicar mucho el codigo......
__device__ int isSemaphoreRed(float4 nextFresult, bool semaphoreState, int type){ // simple, si esta lejos, aumente el contador de zonas en un valor, prueba y error, para que "BAJE" la velocidad, en el caso mas cercano se salga del semaforo, sin contador que salga como se hace actualmente.
	
	//return 0;

	int delay=10;
	if(type==2)
		delay=8;
	else if (type==3)
		delay=4;

	int b = (int)(nextFresult.z * 255.0f);
	
	bool orientationSemaphore = (b & 0x80) >> 7;
	int close = (b & 0x60) >> 5;

	if (type!=1){//vehiculos
		if (orientationSemaphore == semaphoreState){
			if (close==3) //muy cerca
				return -1;
			if (close==2) //cerca
				return delay*2;
			if (close==1) //lejos
				return delay;
		}
		return 0; //muy lejos
	}else{//peatones
		if (! (orientationSemaphore == semaphoreState) ){
			if(close>1)
				return -1;
			if (close==3) //muy cerca
				return -1;
			if (close==2) //cerca
				return delay*2;
			if (close==1) //lejos
				return delay;
		}
		return 0; //muy lejos
	}
	return 0;
}

//OK
__device__ void eraseOneTrace(TColor *dst, int imageW, int imageH, int px, int py, int rx, int ry, int leftSize, int rightSize){

	if (px >= imageW || px < 1)
		return;
	if (py >= imageH || py < 1)
		return;

	float4 result=tex2D(texImage, (float)px + 0.5f, (float)py + 0.5f);
	TColor color= make_color(result.x, result.y, result.z, 0.f);
	dst[imageW*(py) + (px)]=color;
	if (rx==0){ //para direccion arriba-abajo
		for(int n=1; n<rightSize+1; n++){
			result = tex2D(texImage, (float)(px+n*ry) + 0.5f, (float)py + 0.5f);
			color=make_color(result.x, result.y, result.z, 0.f);
			dst[imageW*(py) + (px+n*ry)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			result = tex2D(texImage, (float)(px-n*ry) + 0.5f, (float)py + 0.5f);
			color=make_color(result.x, result.y, result.z, 0.f);
			dst[imageW*(py) + (px-n*ry)]=color;
		}
	}else if (ry==0){ //para direccion izquierda-derecha
		for(int n=1; n<rightSize+1; n++){
			result = tex2D(texImage, (float)(px) + 0.5f, (float)(py-n*rx) + 0.5f);
			color=make_color(result.x, result.y, result.z, 0.f);
			dst[imageW*(py-n*rx) + (px)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			result = tex2D(texImage, (float)(px) + 0.5f, (float)(py+n*rx) + 0.5f);
			color=make_color(result.x, result.y, result.z, 0.f);
			dst[imageW*(py+n*rx) + (px)]=color;
		}
	}else if (rx==ry){ //para diagonal so-ne
		for(int n=1; n<rightSize+1; n++){
			result = tex2D(texImage, (float)(px+n*rx) + 0.5f, (float)(py-n*ry) + 0.5f);
			color=make_color(result.x, result.y, result.z, 0.f);
			dst[imageW*(py-n*ry) + (px+n*rx)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			result = tex2D(texImage, (float)(px-n*rx) + 0.5f, (float)(py+n*ry) + 0.5f);
			color=make_color(result.x, result.y, result.z, 0.f);
			dst[imageW*(py+n*ry) + (px-n*rx)]=color;
		}
	}else if (rx==-ry){ //para diagonal se-no
		for(int n=1; n<rightSize+1; n++){
			result = tex2D(texImage, (float)(px-n*rx) + 0.5f, (float)(py+n*ry) + 0.5f);
			color=make_color(result.x, result.y, result.z, 0.f);
			dst[imageW*(py+n*ry) + (px-n*rx)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			result = tex2D(texImage, (float)(px+n*rx) + 0.5f, (float)(py-n*ry) + 0.5f);
			color=make_color(result.x, result.y, result.z, 0.f);
			dst[imageW*(py-n*ry) + (px+n*rx)]=color;
		}
	}

}

//OK
__device__ void drawOneTrace(TColor *dst, TColor color, int imageW, int imageH, int px, int py, int rx, int ry, int leftSize, int rightSize){
	if (px >= imageW || px < 1)
		return;
	if (py >= imageH || py < 1)
		return;

	dst[imageW*(py) + (px)]=color;

	if (rx==0){ //para direccion arriba-abajo
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py) + (px+n*ry)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py) + (px-n*ry)]=color;
		}
	}else if (ry==0){ //para direccion izquierda-derecha
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py-n*rx) + (px)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py+n*rx) + (px)]=color;
		}
	}else if (rx==ry){ //para diagonal so-ne
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py-n*ry) + (px+n*rx)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py+n*ry) + (px-n*rx)]=color;
		}
	}else if (rx==-ry){ //para diagonal se-no
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py+n*ry) + (px-n*rx)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py-n*ry) + (px+n*rx)]=color;
		}
	}
}

//OK talves mirar para no tener que redibujar TODO el vehiculo cada vez.
__device__ void drawAllTrace(TColor *dst, TColor color, int imageW, int imageH, int x, int y, int dx, int dy, int *devTraceX, int *devTraceY, int *devTraceRotX, int *devTraceRotY, float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout){
	
	if (x==devTraceX[0] && y==devTraceY[0])
		return;

	int size = (dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[0]-1)/2;
	int res = (( (float)dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[0] - 1.f )/2.f - (float)size) * 2;//residuo, cuando no es impar da 1
	int sizeZ = dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[1];
	int leftSize=size;
	int rightSize=size+res;

	
	eraseOneTrace(dst, imageW, imageH, devTraceX[sizeZ-1], devTraceY[sizeZ-1], devTraceRotX[sizeZ-1], devTraceRotY[sizeZ-1], leftSize, rightSize);
	//for (int n=0; n<sizeZ; n++){
	//	eraseOneTrace(dst, imageW, imageH, devTraceX[n], devTraceY[n], devTraceRotX[n], devTraceRotY[n], leftSize, rightSize);
	//}

	/** Esto es para hacer corrimiento del trace, para actualizarlo **/
	for (int n=sizeZ-1; n > 0; n--){
		devTraceX[n]=devTraceX[n-1];
		devTraceY[n]=devTraceY[n-1];
		devTraceRotX[n]=devTraceRotX[n-1];
		devTraceRotY[n]=devTraceRotY[n-1];
	}
	devTraceX[0]=x;
	devTraceY[0]=y;
	//devTraceRotX[0]=dx;
	devTraceRotX[0]=0;
	devTraceRotY[0]=dy;
	
	/** se dibuja el trace actual **/
	drawOneTrace(dst, color, imageW, imageH, devTraceX[0], devTraceY[0], devTraceRotX[0], devTraceRotY[0], leftSize, rightSize);
	//for (int n=0; n < sizeZ ; n++){
	//	drawOneTrace(dst, color, imageW, imageH, devTraceX[n], devTraceY[n], devTraceRotX[n], devTraceRotY[n], leftSize, rightSize);
	//}
}





////////////////////////////////////////////////////////////////////////////////
// Filtering kernels
////////////////////////////////////////////////////////////////////////////////
#include "loadImage_kernel.cu"
#include "pedestrian_kernel.cu"
#include "vehicle_kernel.cu"
#include "transmilenio_kernel.cu"

extern "C"
cudaError_t CUDA_Bind2TextureArray()
{
    return cudaBindTextureToArray(texImage, a_Src);
}

extern "C"
cudaError_t CUDA_UnbindTexture()
{
    return cudaUnbindTexture(texImage);
}

extern "C" 
cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH)
{
    cudaError_t error;

    error = cudaMallocArray(&a_Src, &uchar4tex, imageW, imageH);
    error = cudaMemcpyToArray(a_Src, 0, 0,
                              *h_Src, imageW * imageH * sizeof(uchar4),
                              cudaMemcpyHostToDevice
                              );

    return error;
}


extern "C"
cudaError_t CUDA_FreeArray()
{
    return cudaFreeArray(a_Src);    
}

