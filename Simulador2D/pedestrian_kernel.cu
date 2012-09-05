#include <cuda.h>
#include <stdio.h>
#include <cuPrintf.cu>
#include <shrUtils.h>
#include "cutil_inline.h"
#define CUPRINTF cuPrintf

/***********DETECTS COLISSION**************/
__device__ bool isCollision(TColor *dst, int imageW, int imageH, int x, int y){

	if (x >= imageW || x < 1)
		return true;
	if (y >= imageH || y < 1)
		return true;
		
	TColor color = dst[imageW * y + x];
	int r = (color >> 0) & 0xFF;
	int g = (color >> 8) & 0xFF;
	int b = (color >> 16) & 0xFF;
	int a = (color >> 24) & 0xFF;

	if (b==255) //hay un vehiculo, peaton o transmilenio ocupando el sitio.
		return true;

	int area= r & 0xE0;
	if ( (area >> 5) == 7) //hay un edificio alli
		return true;
	if ( (area >> 6) == 3) //hay una estacion alli
		return true;
	
	return false;
}

/***********CALCULATE NEXT JUMP**************/
__device__ void getNextStep(TColor *dst, int imageW, int imageH, int x, int y, int &px, int &py){

	if (isCollision(dst,imageW,imageH, px+x, py+y) ){// de frente

		if (x==0){ //para direccion arriba-abajo
				//asumiendo direcion hacia arriba
			if ( !isCollision(dst,imageW,imageH,px+y,py+y) ){ // (+,+) - derecha de frente
				px+=y;
				py+=y;
			}else if (!isCollision(dst,imageW,imageH,px-y,py+y)){ // (-,+) - izquierda de frente
				px-=y;
				py+=y;
			}else if (!isCollision(dst,imageW,imageH,px+y,py)){ // (-,0) - derecha
				px+=y;
			}else if (!isCollision(dst,imageW,imageH,px-y,py)){ // (-,0) - izquierda
				px-=y;
			}else if (!isCollision(dst,imageW,imageH,px+y,py-y)){ // (+,-) - diagonal atras derecha
				px+=y;
				py-=y;
			}else if (!isCollision(dst,imageW,imageH,px-y,py-y)){ // (-,-) - diagonal atras izquierda
				px-=y;
				py-=y;
			}else if (!isCollision(dst,imageW,imageH,px,py-y)){ // (0,-)atras
				py-=y;
			}
		}else if (y==0){ //para direccion izquierda-derecha
				//asumiendo direccion hacia la derecha
			if ( !isCollision(dst,imageW,imageH,px+x,py-x) ){ // (+,-) - diagonal derecha
				px+=x;
				py-=x;
			}else if (!isCollision(dst,imageW,imageH,px+x,py+x)){ // (+,+) - diagonal izquierda
				px+=x;
				py+=x;
			}else if (!isCollision(dst,imageW,imageH,px,py-x)){ // (0,-) - derecha
				py-=x;
			}else if (!isCollision(dst,imageW,imageH,px,py+x)){ // (0,+) - izquierda
				py+=x;
			}else if (!isCollision(dst,imageW,imageH,px-x,py-x)){ // (-,-) - diagonal atras derecha
				px-=x;
				py-=x;
			}else if (!isCollision(dst,imageW,imageH,px-x,py+x)){ // (-,+) - diagonal atras izquierda
				px-=x;
				py+=x;
			}else if (!isCollision(dst,imageW,imageH,px-x,py)){ // (-,0) - atras
				px-=x;
			}
		}else if (x==y){ //para diagonal so-ne
			// tomando como direccion (1,1) derecha-arriba
			if ( !isCollision(dst,imageW,imageH,px+x,py) ){ // (+,0) - miro diagonal derecha
				px+=x;
			}else if (!isCollision(dst,imageW,imageH,px,py+y)){ // (0,+) - miro diagonal izquierda
				py+=y;
			}else if (!isCollision(dst,imageW,imageH,px+x,py-y)){ // (+,-) - derecha
				px+=x;
				py-=y;
			}else if (!isCollision(dst,imageW,imageH,px-x,py+y)){ // (-,+) - izquierda
				px-=x;
				py+=y;
			}else if (!isCollision(dst,imageW,imageH,px,py-y)){ // (0,-) - diagonal atras derecha
				py-=y;
			}else if (!isCollision(dst,imageW,imageH,px-x,py)){ // (-,0) - diagonal atras izquierda
				px-=x;
			}else if (!isCollision(dst,imageW,imageH,px-x,py-y)){ // (-,-) - atras
				px-=x;
				py-=y;
			}
		}else if (x==-y){ //para diagonal se-no
			//asumiendo como direccion (1,-1) derecha-abajo
			if ( !isCollision(dst,imageW,imageH,px,py+y) ){ // (0,-) - miro diagonal derecha (asumo y=-1)
				py+=y;
			}else if (!isCollision(dst,imageW,imageH,px+x,py)){ // (0,+) - miro diagonal izquierda (asumo x=1)
				px+=x;
			}else if (!isCollision(dst,imageW,imageH,px-x,py+y)){ // (-,-) - derecha
				px-=x;
				py+=y;
			}else if (!isCollision(dst,imageW,imageH,px+x,py-y)){ // (+,+) - izquierda
				px+=x;
				py-=y;
			}else if (!isCollision(dst,imageW,imageH,px-x,py)){ // (-,0) - diagonal atras derecha
				px-=x;
			}else if (!isCollision(dst,imageW,imageH,px,py-y)){ // (0,+) - diagonal atras izquierda
				py-=y;
			}else if (!isCollision(dst,imageW,imageH,px-x,py-y)){ // (-,+) - atras
				px-=x;
				py-=y;
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
__global__ void PedestrianPhase0(
	int maxPediestran,
	int *devTimeOut,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout
){
	const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id < maxPediestran){

		if(devTimeOut[id]==-10){ //first time only
			int cellSize=maxPediestran*dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[2];
			int cellNumber=id/cellSize;
			devTimeOut[id]=cellNumber;
			return;
		}

		if(devTimeOut[id]<0){
			devTimeOut[id]=1.f/dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[2]-1;
		}else{
			devTimeOut[id]--;
		}
	}
}

/*********  PHASE 1: P'=f(p) *******/
__global__ void PedestrianPhase1(
    TColor *dst,
    int imageW,
    int imageH,
	int maxPediestran,
	bool semaphore,

    int **devLocalX,
	int **devLocalY,
	int *devLocalStep,
	int *devMaxLocalStep,

	int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,
	int *devTimeOut
){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
	
    if(id < maxPediestran){

		int delay=isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f), semaphore, 1);
		if(delay==-1 && devTimeOut[id]>=0){
				devTimeOut[id]=devTimeOut[id]+1;
		}
			

		if(devTimeOut[id]!=0)
			return;

		if (devLocalStep[id]<0){
			devLocalStep[id]++;
			return;
		}

		if (devLocalStep[id]==0){
			devPreviousX[id]=devLocalX[id][5*devLocalStep[id] + 4];
			devPreviousY[id]=devLocalY[id][5*devLocalStep[id] + 4];
			devNextX[id]=devPreviousX[id];
			devNextY[id]=devPreviousY[id];
			devLocalStep[id]++;
			return;
		}
		
		//INVIERTE METAS LOCALES UNA VEZ HA LLEGADO A LA META GLOBAL
		if (devLocalStep[id]==devMaxLocalStep[id]){
			
			for (int i = 0; i < devMaxLocalStep[id]/2; i++){
				for (int n = 0; n < 5; n++){

					int tempX=devLocalX[id][5*(devMaxLocalStep[id]-i-1)+n];
					devLocalX[id][5*(devMaxLocalStep[id]-i-1)+n]=devLocalX[id][5*i+n];
					devLocalX[id][5*i+n]=tempX;

					int tempY=devLocalY[id][5*(devMaxLocalStep[id]-i-1)+n];
					devLocalY[id][5*(devMaxLocalStep[id]-i-1)+n]=devLocalY[id][5*i+n];
					devLocalY[id][5*i+n]=tempY;
				}
			}
			devLocalStep[id]=0;
		}

		int x=0;
		int y=0;
		if (!getNextDirection(id , x, y, devLocalX, devLocalY, devLocalStep, devNextX, devNextY, 1))
			return;

		int px=devNextX[id];
		int py=devNextY[id];
		getNextStep(dst,imageW,imageH,x,y,px,py);		

		if (px != devNextX[id] || py != devNextY[id]){ //nueva posicion
			
			devPreviousX[id]=devNextX[id];
			devPreviousY[id]=devNextY[id];
			devNextX[id]=px;
			devNextY[id]=py;

			dst[imageW * devNextY[id] + devNextX[id]] = make_color(0.f, 1.f, 1.f, 1.f);

			float disX=(float)(devLocalX[id][5*(devLocalStep[id]) + 4]-devNextX[id]);
			float disY=(float)(devLocalY[id][5*(devLocalStep[id]) + 4]-devNextY[id]);
			float hyp=sqrt(disX*disX+disY*disY);
			if ( hyp < 1.f ){
				devLocalStep[id]++;
			}
			devConflicted[imageW * devNextY[id] + devNextX[id]] = id;
		}
    }
}

/*********  PHASE 2: Se intenta solucionar conflictos en paralelo********/
__global__ void PedestrianPhase2( 
    TColor *dst,
    int imageW,
    int imageH,
	int maxPediestran,
	bool semaphore,

	int **devLocalX,
	int **devLocalY,
	int *devLocalStep,

    int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,
	int *devTimeOut,
	int *devClass
){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if((id < maxPediestran) && (devLocalStep[id] >= 0)){

		if(devTimeOut[id]!=0)
			return;

		//int delay=isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f), semaphore, 1);
		//if(delay==-1)
		//	return;

		if (devConflicted[imageW * devNextY[id] + devNextX[id]] == id){ //talves tenga conflicto, pero tiene prioridad sobre los demas
			float4 nextFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
			return;
		}
		if ( (devNextX[id]==devPreviousX[id]) && (devNextY[id]==devPreviousY[id]) )
			//esta en conflicto pero no se ha movido (no deberia pasar nunca)
			return;

		int x=0;
		int y=0;
		if (!getNextDirection(id, x, y, devLocalX, devLocalY, devLocalStep, devNextX, devNextY, 1))
			return;

		int px=devPreviousX[id];
		int py=devPreviousY[id];
		
		getNextStep(dst,imageW,imageH,x,y,px,py);

		//borro la posicion siguiente ya que el id de mas prioridad lo ocupo.
		float4 nextFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
		dst[imageW * devNextY[id] + devNextX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
		//guardo mi nueva posicion, no sobreescribo la anterior.
		devNextX[id]=px;
		devNextY[id]=py;

		if ( (px!=devPreviousX[id]) || (py!=devPreviousY[id]) ){
			// si me pude mover, me muevo a mi nueva coordenada, aunque este movimiento puede generar colisiones.
			float4 newFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
			dst[imageW * devNextY[id] + devNextX[id]] = make_color(0.f, 1.f, 1.f, 1.f);
			devConflicted[imageW * devNextY[id] + devNextX[id]] = id;
		}else{
			//el peaton no se pudo mover. no hago nada.
		}
    }
}

/*********  PHASE 3: Se intenta solucionar conflictos en paralelo********/
__global__ void PedestrianPhase3( 
    TColor *dst,
    int imageW,
    int imageH,
	int maxPediestran,
	bool semaphore,

	int *devLocalStep,
    int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,
	int *devTimeOut,
	int *devClass
){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if((id < maxPediestran) && (devLocalStep[id] >= 0)){

		if(devTimeOut[id]!=0)
			return;

		//int delay=isSemaphoreRed(tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f), semaphore, 1);
		//if(delay==-1)
		//	return;
		//else
		//	devTimeOut[id]+=delay;

		if (devConflicted[imageW * devNextY[id] + devNextX[id]] == id){ //talves tenga conflicto, pero tiene prioridad sobre los demas
			float4 nextFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
			return;
		}

		devNextX[id]=devPreviousX[id];
		devNextY[id]=devPreviousY[id];
    }
}





// SOLUCION INICIAL PARA RESOLUCION DE CONFLICTOS
/*********  PHASE 2: Detecto Colisiones (serial)********/
__global__ void PedestrianCollision(
    TColor *dst,
    int imageW,
    int imageH,
	int maxPediestran,

	int *devClass,
	int *devLocalStep,
	int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,
	int *devTimeOut

){
	for (int id=0; id < maxPediestran; id++){
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

__global__ void PedestrianPhase2Old( 
    TColor *dst,
    int imageW,
    int imageH,
	int maxPediestran,

	int **devLocalX,
	int **devLocalY,
	int *devLocalStep,

    int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,
	int *devTimeOut,
	int *devClass
){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if((id < maxPediestran) && (devLocalStep[id] >= 0)){

		if(devTimeOut[id]!=0)
			return;

		if (devConflicted[id]==-1 || devConflicted[id]==id){ //no tiene conflicto o tiene prioridad asi que borro con confianza el paso anterior
			float4 nextFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
			return;
		}
		if ( (devNextX[id]==devPreviousX[id]) && (devNextY[id]==devPreviousY[id]) )
			//esta en conflicto pero no se ha movido (no deberia pasar nunca)
			return;

		int x=0;
		int y=0;
		if (!getNextDirection(id, x, y, devLocalX, devLocalY, devLocalStep, devNextX, devNextY, 1))
			return;

		int px=devPreviousX[id];
		int py=devPreviousY[id];
		
		getNextStep(dst,imageW,imageH,x,y,px,py);

		//borro la posicion siguiente ya que el id de mas prioridad lo ocupo.
		float4 nextFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
		dst[imageW * devNextY[id] + devNextX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
		//guardo mi nueva posicion, no sobreescribo la anterior.
		devNextX[id]=px;
		devNextY[id]=py;

		if ( (px!=devPreviousX[id]) || (py!=devPreviousY[id]) ){
			// si me pude mover, me muevo a mi nueva coordenada, aunque este movimiento puede generar colisiones.
			float4 newFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
			dst[imageW * devNextY[id] + devNextX[id]] = make_color(0.f, 1.f, 1.f, 1.f);
			devConflicted[imageW * devNextY[id] + devNextX[id]] = id;
			devClass[0]=100; //esta variable no la uso mas, asi que tomo la primera posicion para indicar que al menos hubo un movimiento
		}else{
			//el peaton no se pudo mover. no hago nada.
		}
    }
}


/*********  PHASE 3: Se solucionan los conflictos (serial) que no se pudo resolver en la fase anterior********/
__global__ void PedestrianPhase3Old( 
    TColor *dst,
    int imageW,
    int imageH,
	int maxPediestran,

	int *devLocalStep,
    int *devNextX,
	int *devNextY,
	int *devPreviousX,
	int *devPreviousY,
	int *devConflicted,
	int *devTimeOut,
	int *devClass
){
	if(devClass[0]!=100) 
		return; //no hubo conflictos en la fase 2, asi que esta fase sobra.

    for (int id=0; id < maxPediestran; id++){
		
		if(devTimeOut[id]!=0)
			continue;

		if (devLocalStep[id]<=0)
			continue;
		
		if (devConflicted[id]==-1 || devConflicted[id]==id){ //talves tenga conflicto, pero tiene prioridad sobre los demas
			//borro paso anterior.
			float4 nextFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
			continue;
		}
		
		if ( (devNextX[id]==devPreviousX[id]) && (devNextY[id]==devPreviousY[id]) )
			//esta en conflicto pero no se ha movido (no deberia pasar nunca)
			continue;

		int x=devNextX[id]-devPreviousX[id];
		int y=devNextY[id]-devPreviousY[id];

		int px=devPreviousX[id];
		int py=devPreviousY[id];

		getNextStep(dst,imageW,imageH,x,y,px,py);
		
		//borro la posicion siguiente ya que el id de mas prioridad lo ocupo.
		float4 nextFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
		dst[imageW * devNextY[id] + devNextX[id]] = make_color(nextFresult.x, nextFresult.y, nextFresult.z, 0.0f);
		//guardo mi nueva posicion, no sobreescribo la anterior.
		devNextX[id]=px;
		devNextY[id]=py;

		if ( (px!=devPreviousX[id]) || (py!=devPreviousY[id]) ){
			//me muevo al siguiente punto
			float4 newFresult = tex2D(texImage, (float)devNextX[id] + 0.5f, (float)devNextY[id] + 0.5f);
			dst[imageW * devNextY[id] + devNextX[id]] = make_color(0.f, 1.f, 1.f, 1.f);	
			//borro el punto anterior
			float4 preFresult = tex2D(texImage, (float)devPreviousX[id] + 0.5f, (float)devPreviousY[id] + 0.5f);
			dst[imageW * devPreviousY[id] + devPreviousX[id]] = make_color(preFresult.x, preFresult.y, preFresult.z, 0.0f);
		}else{
			//el peaton no se pudo mover. no hago nada.
		}
	}
}



extern "C" void run_Pedestrian(
    TColor *d_dst,
	int *devClass,
    int imageW,
    int imageH,
	int maxPediestran,
	bool parallelDetection,
	bool semaphore,

    int **devLocalX,
	int **devLocalY,
	int *devLocalStep,
	int *devMaxLocalStep,

	int *devCurrentX,
	int *devCurrentY,
	int *devPreviousX,
	int *devPreviousY,

	int *devConflicted,
	int **devRelated,

	int *devTimeOut,
	float *devSpeed,
	float *dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout

){

	/******PARA IMPRIMIR EN CONSOLA******/
	//cudaPrintfInit(); //descomentar esta linea para activar la impresion por consola
	//CUPRINTF("aqui va el mensaje, funciona igual que un printf"); //copiar este comando en los kernels donde se desee imprimir

	cudaMemset(devConflicted,-1,imageW*imageH*sizeof(int));
	dim3 dimGrid(maxPediestran, 1);
	dim3 dimBlock(1, 1, 1); // para saber cuantos se ejecuta, solo multiplique todos los valores, no use Z.

	if(parallelDetection){
		//Fase 0: Control de velocidad, determino si en este paso de simulacion intento moverme o no.
		PedestrianPhase0<<<dimGrid, dimBlock>>>(maxPediestran, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);

		//Fase 1: realizo movimientos así hayan colisiones.
		PedestrianPhase1<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxPediestran, semaphore,
												devLocalX, devLocalY, devLocalStep, devMaxLocalStep, 
												devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted, devTimeOut);
		
		//Fase 2: detecto y corrigo movimientos de peatones en conflicto con nuevos movimientos, aun asi pueden haber conflictos en los nuevos movimientos
		PedestrianPhase2<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxPediestran, semaphore,
												devLocalX, devLocalY, devLocalStep,
												devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted, devTimeOut, devClass);
		
		//Fase 3: detecto y corrigo movimientos de peatones en conflicto, pero no genero movimientos nuevos, me devuelvo mejor a un estado estable.
		PedestrianPhase3<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxPediestran, semaphore,
												devLocalStep,
												devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted, devTimeOut, devClass);
	}else{
		//Fase 0: Control de velocidad, determino si en este paso de simulacion intento moverme o no.
		PedestrianPhase0<<<dimGrid, dimBlock>>>(maxPediestran, devTimeOut, dev_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);

		//Fase 1: realizo movimientos así hayan colisiones.
		PedestrianPhase1<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxPediestran, semaphore,
												devLocalX, devLocalY, devLocalStep, devMaxLocalStep, 
												devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted, devTimeOut);
		
		//Fase intermedia: detecto colisiones. (solo se detectan, no se arreglan)
		cudaMemset(devClass,-1,imageW*imageH*sizeof(int));
		PedestrianCollision<<<1,1>>>(d_dst, imageW, imageH, maxPediestran, 
									 devClass,devLocalStep,
									 devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted, devTimeOut);
		
		//Fase 2: detecto y corrigo movimientos de peatones en conflicto con nuevos movimientos, aun asi pueden haber conflictos en los nuevos movimientos
		PedestrianPhase2Old<<<dimGrid, dimBlock>>>(d_dst, imageW, imageH, maxPediestran, 
													devLocalX, devLocalY, devLocalStep,
													devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted, devTimeOut, devClass);

	
		//Fase intermedia: detecto nuevas colisiones, que quedaron de la fase anterior.
		cudaMemset(devClass,-1,imageW*imageH*sizeof(int));
		PedestrianCollision<<<1,1>>>(d_dst, imageW, imageH, maxPediestran, 
									 devClass,devLocalStep,
									 devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted, devTimeOut);
		
		//Fase 3: arreglo las colisiones resultantes de manera serial.
		PedestrianPhase3Old<<<1,1>>>(d_dst, imageW, imageH, maxPediestran, 
									 devLocalStep,
									 devCurrentX, devCurrentY, devPreviousX, devPreviousY, devConflicted, devTimeOut, devClass);
	}
	
	/***********PARA IMPRIMIR EN CONSOLA********/
	//cudaPrintfDisplay(stdout, true); //descomentar esta linea para imprimer por consola, no modificar los atributos.
	//cudaPrintfEnd(); //descomentar esta linea para finalizar la impresion por consola.


}