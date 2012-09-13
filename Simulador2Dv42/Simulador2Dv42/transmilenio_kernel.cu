#include <cuda.h>
#include <stdio.h>
//#include <cuPrintf.cu>
#include <shrUtils.h>
#include "cutil_inline.h"
//#define CUPRINTF cuPrintf

// ok
__device__ bool isColissionForTransmilenio(TColor *dst, int id, int imageW, int imageH, int x, int y, int dx, int dy){

	int nx=x+dx;
	int ny=y+dy;

	if (nx >= imageW || nx < 1)
		return true;
	if (ny >= imageH || ny < 1)
		return true;
		
	TColor color = dst[imageW * ny + nx];
	int r = (color >> 0) & 0xFF;
	int g = (color >> 8) & 0xFF;
	int b = (color >> 16) & 0xFF;
	int a = (color >> 24) & 0xFF;

	int area= r & 0xE0;
	if ( (area >> 5) == 7) //hay un edificio alli
		return true;
	if ( (area >> 5) == 6) //hay una estacion alli
		return true;
	if ( (area >> 5) == 5) //via peatonal
		return true;
	
	if ( ((area >> 5) != 4) && (area >> 5) != 3) //no es una via de transmilenio/carro
		return true;
	
	if ( a != id && b==255) //hay un vehiculo, peaton o transmilenio ocupando el sitio.
		return true;

	if (r == 0) //es un punto negro, lo aprovecho dado que el espacio de las vias de transmilenio son pequenias
		return false;

	bool up = ((g & 0x80) >> 7) == 1;
	bool down = ((g & 0x40) >> 6) == 1;
	bool left = ((g & 0x20) >> 5) == 1;
	bool right = ((g & 0x10) >> 4) == 1;
	
	if ((dy>-1 && up) || (dy<1 && down) || (dx>-1 && right) || (dx<1 && left))
		return false;
	else
		return true;
}

//ok
__device__ void getFirstStepForTransmilenio(TColor *dst, int id, int imageW, int imageH, int x, int y, int &px, int &py){

	if (isColissionForTransmilenio(dst,id,imageW,imageH, px, py, x, y) ){// de frente

		if (x==0){ //para direccion arriba-abajo
			//asumiendo direcion hacia arriba
			if ( !isColissionForTransmilenio(dst,id,imageW,imageH,px,py,y,y) ){ // (+,+) - derecha de frente
				px+=y;
				py+=y;
			}else if (!isColissionForTransmilenio(dst,id,imageW,imageH,px,py,-y,y)){ // (-,+) - izquierda de frente
				px-=y;
				py+=y;
			}
		}else if (y==0){ //para direccion izquierda-derecha
			//asumiendo direccion hacia la derecha
			if ( !isColissionForTransmilenio(dst,id,imageW,imageH,px,py,x,-x) ){ // (+,-) - diagonal derecha
				px+=x;
				py-=x;
			}else if (!isColissionForTransmilenio(dst,id,imageW,imageH,px,py,x,x)){ // (+,+) - diagonal izquierda
				px+=x;
				py+=x;
			}
		}else if (x==y){ //para diagonal so-ne
			// tomando como direccion (1,1) derecha-arriba
			if ( !isColissionForTransmilenio(dst,id,imageW,imageH,px,py,x,0) ){ // (+,0) - miro diagonal derecha
				px+=x;
			}else if (!isColissionForTransmilenio(dst,id,imageW,imageH,px,py,0,y)){ // (0,+) - miro diagonal izquierda
				py+=y;
			}
		}else if (x==-y){ //para diagonal se-no
			//asumiendo como direccion (1,-1) derecha-abajo
			if ( !isColissionForTransmilenio(dst,id,imageW,imageH,px,py,0,y) ){ // (0,-) - miro diagonal derecha (asumo y=-1)
				py+=y;
			}else if (!isColissionForTransmilenio(dst,id,imageW,imageH,px,py,x,0)){ // (0,+) - miro diagonal izquierda (asumo x=1)
				px+=x;
			}
		}
	}else{
		px+=x;
		py+=y;
	}
}


//OK
__device__ void frontSidersForTransmilenio(int id, int rx, int ry, int &dx, int &dy){

	dy=0;
	dx=ry;

	// inicialmente soportaba el movimiento a cualquir direccion, pero despues de que alguna de las dimensiones del vehiculo aumentara 
	// a mas de 2 pixeles, la lógica no servía, asi que se simplificó a solo dejar movimiento de arriba-abajo
	/*if (rx==0){ //para direccion arriba-abajo
		dy=0;
		dx=ry;
	}else if (ry==0){ //para direccion izquierda-derecha
		dy=-rx;
		dx=0;
	}else if (rx==ry){ //para diagonal so-ne
		dy=-ry;
		dx=rx;
	}else if (rx==-ry){ //para diagonal se-no
		dy=ry;
		dx=-rx;
	}*/
}

//OK
__device__ bool isFrontCollisionForTransmilenio(TColor *dst, int id, int imageW, int imageH, int px, int py, int x, int y, int dx, int dy, int rightSize, int leftSize){
	
	if (isColissionForTransmilenio(dst,id,imageW,imageH, px, py, x, y))
		return true;

	for(int n=1; n<rightSize+1; n++){
		if(isColissionForTransmilenio(dst,id,imageW,imageH, px+x, py+y, n*dx, n*dy))
			return true;
	}
	for(int n=1; n<leftSize+1; n++){
		if(isColissionForTransmilenio(dst,id,imageW,imageH, px+x, py+y, -n*dx, -n*dy))
			return true;
	}

	return false;
}

__device__ void getNextStepForTransmilenio(TColor *dst, int id, int imageW, int imageH, int x, int y, int &px, int &py, int *devTraceX, int *devTraceY, float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout){

	if (devTraceX[0]<0 && devTraceY[0]<0){
		getFirstStepForTransmilenio(dst, id, imageW, imageH, x, y, px, py);
		return;
	}

	int size = (dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[0]-1)/2;
	int res = (( (float)dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[0] - 1.f )/2.f - (float)size) * 2;
	int sizeZ = dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[1];
	int leftSize=size;
	int rightSize=size+res;

	int dx=0;
	int dy=0;
	frontSidersForTransmilenio(id, x, y, dx, dy);
	if ( isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, x, y, dx, dy, rightSize, leftSize) ){// de frente

		if (x==0){ //para direccion arriba-abajo		
			//asumiendo direcion hacia arriba
			frontSidersForTransmilenio(id, y, y, dx, dy);
			if ( !isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, y, y, dx, dy, rightSize, leftSize) ){ // (+,+) - derecha de frente
				px+=y;
				py+=y;
			}else{
				frontSidersForTransmilenio(id, -y, y, dx, dy);
				if (!isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, -y, y, dx, dy, rightSize, leftSize) ){ // (-,+) - izquierda de frente
					px-=y;
					py+=y;
				}
			}
			
		}else if (y==0){ //para direccion izquierda-derecha
				//asumiendo direccion hacia la derecha
			frontSidersForTransmilenio(id, x, -x, dx, dy);
			if ( !isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, x, -x, dx, dy, rightSize, leftSize) ){ // (+,-) - diagonal derecha
				px+=x;
				py-=x;
			}else{
				frontSidersForTransmilenio(id, x, x, dx, dy);
				if (!isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, x, x, dx, dy, rightSize, leftSize) ){ // (+,+) - diagonal izquierda
					px+=x;
					py+=x;
				}
			}
			
		}else if (x==y){ //para diagonal so-ne
			// tomando como direccion (1,1) derecha-arriba
			frontSidersForTransmilenio(id, x, 0, dx, dy);
			if ( !isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, x, 0, dx, dy, rightSize, leftSize) ){ // (+,0) - miro diagonal derecha
				px+=x;
			}else{
				frontSidersForTransmilenio(id, 0, y, dx, dy);
				if (!isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, 0, y, dx, dy, rightSize, leftSize) ){ // (0,+) - miro diagonal izquierda
					py+=y;
				}
			}

		}else if (x==-y){ //para diagonal se-no
			//asumiendo como direccion (1,-1) derecha-abajo
			frontSidersForTransmilenio(id, 0, y, dx, dy);
			if ( !isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, 0, y, dx, dy, rightSize, leftSize) ){ // (0,-) - miro diagonal derecha (asumo y=-1)
				py+=y;
			}else{
				frontSidersForTransmilenio(id, x, 0, dx, dy);
				if (!isFrontCollisionForTransmilenio(dst, id, imageW, imageH, px, py, x, 0, dx, dy, rightSize, leftSize)){ // (0,+) - miro diagonal izquierda (asumo x=1)
					px+=x;
				}
			}
		}
	}else{
		px+=x;
		py+=y;
	}
}
























//
//
// AQUI EMPIEZAN LAS FASES DEL PASO DE SIMULACION
//
//


/********* SPEED MANAGER *******/
__global__ void TransmilenioPhase0(
	int maxTransmilenio,
	int *devTimeOut,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout
){
	const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id < maxTransmilenio)
	{
		if(devTimeOut[id]==-10) //solo entra la primera vez.
		{
			int cellSize=maxTransmilenio*dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[2];
			int cellNumber=id/cellSize;
			devTimeOut[id]=cellNumber;
			return;
		}
		if(devTimeOut[id]<0)
		{
			devTimeOut[id]=1.f/dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[2]-1;
		}else{
			devTimeOut[id]--;
		}
	}
}


/*********  PHASE 1: P'=f(p) *******/
__global__ void TransmilenioPhase1(
    TColor *dst,
    int imageW,
    int imageH,
	int maxTransmilenio,
	bool semaphore,

    int **devLocalX,
	int **devLocalY,
	int *devLocalStep,
	int *devMaxLocalStep,

	int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,

	int **devTraceX,
	int **devTraceY,
	int **devTraceRotX,
	int **devTraceRotY,

	int *devConflicted,
	int *devTimeOut,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout
){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id < maxTransmilenio){
		
		if(devTimeOut[id]!=0)
			return;

		if (devLocalStep[id]<0){
			devLocalStep[id]++;
			return;
		}

		int delay=isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f), semaphore, 3);
		if(delay==-1)
			return;

		//if (isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f)
		///	, semaphore, 3) ){
		//	return;
		//}

		if (devLocalStep[id]==0){
			devPreviousX[id]=devLocalX[id][5*devLocalStep[id] + 4];
			devPreviousY[id]=devLocalY[id][5*devLocalStep[id] + 4];
			devNextX[id]=devPreviousX[id];
			devNextY[id]=devPreviousY[id];
			devLocalStep[id]++;
			return; //comentar?
		}

        if (devLocalStep[id]==devMaxLocalStep[id]){
            float4 nextFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
		    dst[imageW * devNextY[id] + devNextX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);

            devLocalStep[id]=0;
            return; //comentar?
        }

		int x=0;
		int y=0;
		if (!getNextDirection(id , x, y, devLocalX, devLocalY, devLocalStep, devNextX, devNextY, 3))
			return;

		int px=devNextX[id];
		int py=devNextY[id];
		getNextStepForTransmilenio(dst, id, imageW, imageH, x, y, px, py, devTraceX[id], devTraceY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);

		if (px != devNextX[id] || py != devNextY[id]){ //nueva posicion
			
			devPreviousX[id]=devNextX[id];
			devPreviousY[id]=devNextY[id];
			devNextX[id]=px;
			devNextY[id]=py;


			dst[imageW * devNextY[id] + devNextX[id]] = make_color(1.f, 0.f, 1.f, ((float)id)/255.0f); //prueba REVISAR !!!


			
			float disX=(float)(devLocalX[id][5*(devLocalStep[id]) + 4]-devNextX[id]);
			float disY=(float)(devLocalY[id][5*(devLocalStep[id]) + 4]-devNextY[id]);
			float hyp=sqrt(disX*disX+disY*disY);
			if ( hyp < 2.f ){
				devLocalStep[id]++;
				if (devLocalStep[id]!=devMaxLocalStep[id]){
					devTimeOut[id]+=dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[4];
					
					float4 nextFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
					dst[imageW * devNextY[id] + devNextX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
				}
			}
			devConflicted[imageW * devNextY[id] + devNextX[id]] = id; //no es necesario hacerlo con todas las partes dado que solo se mueve el frente, el resto queda quieto.
		}
    }
}


/*********  PHASE 2: Se intenta solucionar conflictos en paralelo********/
__global__ void TransmilenioPhase2( 
    TColor *dst,
    int imageW,
    int imageH,
	int maxTransmilenio,
	bool semaphore,

	int **devLocalX,
	int **devLocalY,
	int *devLocalStep,

    int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	
	int **devTraceX,
	int **devTraceY,
	int **devTraceRotX,
	int **devTraceRotY,

	int *devConflicted,
	int *devTimeOut,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout
){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if((id < maxTransmilenio) && (devLocalStep[id] >= 0)){

		if(devTimeOut[id]!=0)
			return;

		int delay=isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f), semaphore, 3);
		if(delay==-1)
			return;

//		if (isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f)
//			, semaphore, 3) ){
//			return;
//		}

		if (devConflicted[imageW * devNextY[id] + devNextX[id]] == id){ //talves tenga conflicto, pero tiene prioridad sobre los demas
			//float4 nextFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			//dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
			
			TColor color=make_color(1.f, 0.f, 1.f, ((float)id)/255.0f);
			drawAllTrace(dst, color, imageW, imageH, devNextX[id], devNextY[id], 0, devNextY[id]-devPreviousY[id],
				devTraceX[id],	devTraceY[id],	devTraceRotX[id], devTraceRotY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
			return;
		}
		if ( (devNextX[id]==devPreviousX[id]) && (devNextY[id]==devPreviousY[id]) )
			//esta en conflicto pero no se ha movido (no deberia pasar nunca)
			return;

		int x=0;
		int y=0;
		if (!getNextDirection(id , x, y, devLocalX, devLocalY, devLocalStep, devNextX, devNextY, 3))
			return;

		int px=devPreviousX[id];
		int py=devPreviousY[id];

		getNextStepForTransmilenio(dst, id, imageW, imageH, x, y, px, py, devTraceX[id], devTraceY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
		
		//borro la posicion siguiente ya que el id de mas prioridad lo ocupo.
		//float4 nextFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
		//dst[imageW * devNextY[id] + devNextX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
		//guardo mi nueva posicion, no sobreescribo la anterior.
		devNextX[id]=px;
		devNextY[id]=py;

		if ( (px!=devPreviousX[id]) || (py!=devPreviousY[id]) ){
			// si me pude mover, me muevo a mi nueva coordenada, aunque este movimiento puede generar colisiones.
			float4 newFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
			dst[imageW * devNextY[id] + devNextX[id]] = make_color(1.f, 0.f, 1.f, ((float)id)/255.0f);
			devConflicted[imageW * devNextY[id] + devNextX[id]] = id;
		}else{
			//el peaton no se pudo mover. no hago nada.
		}

    }
}


/*********  PHASE 3: Se solucionan los conflictos (serial) que no se pudo resolver en la fase 3********/
__global__ void TransmilenioPhase3( 
    TColor *dst,
    int imageW,
    int imageH,
	int maxTransmilenio,
	bool semaphore,

	int *devLocalStep,
    int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,

	int **devTraceX,
	int **devTraceY,
	int **devTraceRotX,
	int **devTraceRotY,

	int *devConflicted,
	int *devTimeOut,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout
){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if((id < maxTransmilenio) && (devLocalStep[id] >= 0)){

		if(devTimeOut[id]!=0)
			return;
	
		if (devLocalStep[id]<=0)
			return;

		int delay=isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f), semaphore, 3);
		if(delay!=-1)
			devTimeOut[id]+=delay;

		if (devConflicted[imageW * devNextY[id] + devNextX[id]] == id){ //talves tenga conflicto, pero tiene prioridad sobre los demas
			// no se ha iniciado o no tiene conflicto
			//float4 nextFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			//dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
			
			TColor color=make_color(1.f, 0.f, 1.f, ((float)id)/255.0f);
			drawAllTrace(dst, color, imageW, imageH, devNextX[id], devNextY[id], 0, devNextY[id]-devPreviousY[id],
				devTraceX[id],	devTraceY[id],	devTraceRotX[id], devTraceRotY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
			return;
		}
		
		devNextX[id]=devPreviousX[id];
		devNextY[id]=devPreviousY[id];

    }
}


























/*********  PHASE 2: Detecto Colisiones (serial)********/
__global__ void TransmilenioCollision(
    TColor *dst,
    int imageW,
    int imageH,
	int maxTransmilenio,
	bool semaphore,

	int *devClass,
	int *devLocalStep,
	int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,

	int **devTraceX,
	int **devTraceY,
	int **devTraceRotX,
	int **devTraceRotY,

	int *devTimeOut,
	float *devSpeed,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout

){
    //const int id = blockDim.x * blockIdx.x + threadIdx.x;

	for (int id=0; id < maxTransmilenio; id++){
		devConflicted[id]=-1;
		if(devTimeOut[id]!=0)
			continue;
		if (devLocalStep[id] >= 0){
			if (devClass[imageW*devNextY[id] + devNextX[id]]==-1)
				devClass[imageW*devNextY[id] + devNextX[id]]=id;
			else
				devConflicted[id]=devClass[imageW*devNextY[id] + devNextX[id]];
		}
	}
}


/*********  PHASE 3: Se intenta solucionar conflictos en paralelo********/
__global__ void TransmilenioPhase2OLD( 
    TColor *dst,
    int imageW,
    int imageH,
	int maxTransmilenio,
	bool semaphore,

	int **devLocalX,
	int **devLocalY,
	int *devLocalStep,

    int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,

	int **devTraceX,
	int **devTraceY,
	int **devTraceRotX,
	int **devTraceRotY,

	int *devClass,
	int *devTimeOut,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout
){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if((id < maxTransmilenio) && (devLocalStep[id] >= 0)){

		if(devTimeOut[id]!=0)
			return;

		int delay=isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f), semaphore, 3);
		if(delay==-1)
			return;
		
		//if (isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f)
		//	, semaphore, 3) ){
		//	return;
		//}

		if (devConflicted[id]==-1 || devConflicted[id]==id){ //no tiene conflicto, borro con confianza el paso anterior
			//float4 nextFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			//dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
			
			TColor color=make_color(1.f, 0.f, 1.f, ((float)id)/255.0f);
			drawAllTrace(dst, color, imageW, imageH, devNextX[id], devNextY[id], 0, devNextY[id]-devPreviousY[id],
				devTraceX[id],	devTraceY[id],	devTraceRotX[id], devTraceRotY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
			return;
		}
		if ( (devNextX[id]==devPreviousX[id]) && (devNextY[id]==devPreviousY[id]) )
			//esta en conflicto pero no se ha movido (no deberia pasar nunca)
			return;

		int x=0;
		int y=0;
		if (!getNextDirection(id , x, y, devLocalX, devLocalY, devLocalStep, devNextX, devNextY, 3))
			return;

		int px=devPreviousX[id];
		int py=devPreviousY[id];

		getNextStepForTransmilenio(dst, id, imageW, imageH, x, y, px, py, devTraceX[id], devTraceY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
		
		//borro la posicion siguiente ya que el id de mas prioridad lo ocupo.
		//float4 nextFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
		//dst[imageW * devNextY[id] + devNextX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
		//guardo mi nueva posicion, no sobreescribo la anterior.
		devNextX[id]=px;
		devNextY[id]=py;

		if ( (px!=devPreviousX[id]) || (py!=devPreviousY[id]) ){
			// si me pude mover, me muevo a mi nueva coordenada, aunque este movimiento puede generar colisiones.
			float4 newFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
			dst[imageW * devNextY[id] + devNextX[id]] = make_color(1.f, 0.f, 1.f, ((float)id)/255.0f);
			devConflicted[imageW * devNextY[id] + devNextX[id]] = id;
			devClass[0]=100; //esta variable no la uso mas, asi que tomo la primera posicion para indicar que al menos hubo un movimiento
		}else{
			//el peaton no se pudo mover. no hago nada.
		}

    }
}


/*********  PHASE 4: Se solucionan los conflictos (serial) que no se pudo resolver en la fase 3********/
__global__ void TransmilenioPhase3OLD( 
    TColor *dst,
    int imageW,
    int imageH,
	int maxTransmilenio,
	bool semaphore,

	int *devLocalStep,

    int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,

	int **devTraceX,
	int **devTraceY,
	int **devTraceRotX,
	int **devTraceRotY,

	int *devClass,
	int *devTimeOut,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout
){
	if(devClass[0]!=100) 
		return; //no hubo conflictos en la fase 2, asi que esta fase sobra.
    
    for (int id=0; id < maxTransmilenio; id++){

		if(devTimeOut[id]!=0)
			continue;
	
		if (devLocalStep[id]<=0)
			continue;
		
		int delay=isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f), semaphore, 3);
		if(delay==-1)
			return;
		else
			devTimeOut[id]+=delay;
		//if (isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f)
		//	, semaphore, 3) ){
		//	continue;
		//}

		if (devConflicted[id]==-1 || devConflicted[id]==id){
			// no se ha iniciado o no tiene conflicto
			//float4 nextFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			//dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
			
			TColor color=make_color(1.f, 0.f, 1.f, ((float)id)/255.0f);
			drawAllTrace(dst, color, imageW, imageH, devNextX[id], devNextY[id], 0, devNextY[id]-devPreviousY[id],
				devTraceX[id],	devTraceY[id],	devTraceRotX[id], devTraceRotY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
			continue;
		}
		
		if ( (devNextX[id]==devPreviousX[id]) && (devNextY[id]==devPreviousY[id]) )
			//esta en conflicto pero no se ha movido (no deberia pasar nunca)
			continue;

		int x=devNextX[id]-devPreviousX[id];
		int y=devNextY[id]-devPreviousY[id];

		int px=devPreviousX[id];
		int py=devPreviousY[id];

		getNextStepForTransmilenio(dst, id, imageW, imageH, x, y, px, py, devTraceX[id], devTraceY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
		
		//borro la posicion siguiente ya que el id de mas prioridad lo ocupo.
		//float4 nextFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
		//dst[imageW * devNextY[id] + devNextX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
		//guardo mi nueva posicion, no sobreescribo la anterior.
		devNextX[id]=px;
		devNextY[id]=py;

		if ( (px!=devPreviousX[id]) || (py!=devPreviousY[id]) ){
			// si me pude mover, me muevo a mi nueva coordenada, aunque este movimiento puede generar colisiones.
			//float4 newFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
			//dst[imageW * devNextY[id] + devNextX[id]] = make_color(1.f, 0.f, 1.f, ((float)id)/255.0f);

			//float4 preFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			//dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(preFresult.x, preFresult.y, preFresult.z, 0.0f);

			TColor color=make_color(1.f, 0.f, 1.f, ((float)id)/255.0f);
			drawAllTrace(dst, color, imageW, imageH, devNextX[id], devNextY[id], 0, devNextY[id]-devPreviousY[id],
				devTraceX[id],	devTraceY[id],	devTraceRotX[id], devTraceRotY[id], dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
		}else{
			//el peaton no se pudo mover. no hago nada.
		}

    }
}






extern "C" void run_Transmilenio(
    TColor *d_dst,
	int *devClass,
    int imageW,
    int imageH,
	int maxTransmilenio,
	bool parallelDetection,
	bool semaphore,

    int **devLocalX,
	int **devLocalY,
	int *devLocalStep,
	int *devMaxLocalStep,

	int *devCurrentX, //para phase 1
	int *devCurrentY,
	int *devPreviousX,
	int *devPreviousY,

	int **devTraceX,
	int **devTraceY,
	int **devTraceRotX,
	int **devTraceRotY,

	int *devConflicted, //para phase 2 y 4
	int **devRelated, //para phase 3

	int *devTimeOut,
	float *devSpeed,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout
){
	/******PARA IMPRIMIR EN CONSOLA******/
	//cudaPrintfInit(); //descomentar esta linea para activar la impresion por consola
	//CUPRINTF("aqui va el mensaje, funciona igual que un printf"); //copiar este comando en los kernels donde se desee imprimir

	//cudaMemset(devConflicted,-1,imageW*imageH*sizeof(int));  //REVISAR ESTO !!!! :OOOOOOOOOOOOOO
	cudaMemset(devClass,-1,imageW*imageH*sizeof(int));  //REVISAR ESTO !!!! :OOOOOOOOOOOOOO

	dim3 dimGrid(maxTransmilenio, 1); //define en total cuantas veces se ejecuta. (multiplicacion de ambos)
	dim3 dimBlock(1, 1, 1); // para saber cuantos se ejecuta, solo multiplique todos los valores, no use Z.

	if(parallelDetection){
		//Fase 0: Control de velocidad, determino si en este paso de simulacion intento moverme o no.
		TransmilenioPhase0<<<dimGrid, dimBlock>>>(maxTransmilenio, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
		
		//Fase 1: realizo movimientos así hayan colisiones.
		TransmilenioPhase1<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxTransmilenio, semaphore, 
												  devLocalX, devLocalY, devLocalStep, devMaxLocalStep, 
												  devCurrentX, devCurrentY, devPreviousX, devPreviousY,
												  devTraceX, devTraceY, devTraceRotX, devTraceRotY,
												  devConflicted, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);

		//Fase 2: detecto y corrigo movimientos de peatones en conflicto con nuevos movimientos, aun asi pueden haber conflictos en los nuevos movimientos
		TransmilenioPhase2<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxTransmilenio, semaphore, 
												  devLocalX, devLocalY, devLocalStep,
												  devCurrentX, devCurrentY, devPreviousX, devPreviousY,
												  devTraceX, devTraceY, devTraceRotX, devTraceRotY,
												  devConflicted, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);

		//Fase 3: detecto y corrigo movimientos de peatones en conflicto, pero no genero movimientos nuevos, me devuelvo mejor a un estado estable.
		TransmilenioPhase3<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxTransmilenio, semaphore,
												  devLocalStep,
												  devCurrentX, devCurrentY, devPreviousX, devPreviousY,
												  devTraceX, devTraceY, devTraceRotX, devTraceRotY,
												  devConflicted,devTimeOut,dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);



	}else{
		//Fase 0: Control de velocidad, determino si en este paso de simulacion intento moverme o no.
		TransmilenioPhase0<<<dimGrid, dimBlock>>>(maxTransmilenio, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);

		//Fase 1: realizo movimientos así hayan colisiones.
		TransmilenioPhase1<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxTransmilenio, semaphore, 
												  devLocalX, devLocalY, devLocalStep, devMaxLocalStep, 
												  devCurrentX, devCurrentY, devPreviousX, devPreviousY,
												  devTraceX, devTraceY, devTraceRotX, devTraceRotY,
												  devConflicted, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);

		//Fase intermedia: detecto colisiones. (solo se detectan, no se arreglan)
		cudaMemset(devClass,-1,imageW*imageH*sizeof(int));
		TransmilenioCollision<<<1,1>>>(d_dst, imageW, imageH, maxTransmilenio, semaphore, 
									   devClass, devLocalStep,
									   devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted,
									   devTraceX, devTraceY, devTraceRotX, devTraceRotY,
									   devTimeOut, devSpeed, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
		
		//Fase 2: detecto y corrigo movimientos de peatones en conflicto con nuevos movimientos, aun asi pueden haber conflictos en los nuevos movimientos
		TransmilenioPhase2OLD<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxTransmilenio, semaphore, 
													 devLocalX, devLocalY, devLocalStep,
													 devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted,
													 devTraceX, devTraceY, devTraceRotX, devTraceRotY,
													 devClass, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
	
		//Fase intermedia: detecto nuevas colisiones, que quedaron de la fase anterior.
		cudaMemset(devClass,-1,imageW*imageH*sizeof(int));
		TransmilenioCollision<<<1,1>>>(d_dst, imageW, imageH, maxTransmilenio, semaphore, 
									   devClass, devLocalStep,
									   devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted,
									   devTraceX, devTraceY, devTraceRotX, devTraceRotY,
									   devTimeOut, devSpeed, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
		
		//Fase 3: arreglo las colisiones resultantes de manera serial.
		TransmilenioPhase3OLD<<<1,1>>>(d_dst, imageW, imageH, maxTransmilenio, semaphore, 
									   devLocalStep,
									   devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted,
									   devTraceX, devTraceY, devTraceRotX, devTraceRotY,
									   devClass, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);

    }
	
	/***********PARA IMPRIMIR EN CONSOLA********/
	//cudaPrintfDisplay(stdout, true); //descomentar esta linea para imprimer por consola, no modificar los atributos.
	//cudaPrintfEnd(); //descomentar esta linea para finalizar la impresion por consola.
}