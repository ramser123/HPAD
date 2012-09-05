/*
 * Copyright 2012 Diego Hernando Rodriguez Gaitan.  Licenced by GPL.
 *
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "MobileManager.h"

//documentar de donde salen estos valores en GIS
const float cosiente=109728.16f;
const float minY=0.94514084f;
const float minX=78.57597f;

bool semaphoreState=true;
int semaphoreCounter=0;

Pedestrian *pedestrian=NULL;
Zone *pediestranZones=NULL;
int **pediestranX=NULL;
int **pediestranY=NULL;
int *pediestranStep=NULL;
int *pediestranMaxStep=NULL;
int **devCPUPediestranX=NULL;
int **devCPUPediestranY=NULL;
float *pediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout=NULL;
map<int,Zone> idPediestranZones;




Vehicle *vehicle;
Zone *vehicleZones;
int **vehicleX;
int **vehicleY;
int *vehicleStep;
int *vehicleMaxStep;
int **devCPUVehicleX;
int **devCPUVehicleY;
int **devCPUVehicleTraceX;
int **devCPUVehicleTraceY;
int **devCPUVehicleTraceRotX;
int **devCPUVehicleTraceRotY;
float *vehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout=NULL;
map<int,Zone> idVehicleZones;

Transmilenio *transmilenio;
Zone *transmilenioZones;
int **transmilenioX;
int **transmilenioY;
int *transmilenioStep;
int *transmilenioMaxStep;
int **devCPUTransmilenioX;
int **devCPUTransmilenioY;

int **devCPUTransmilenioTraceX;
int **devCPUTransmilenioTraceY;
int **devCPUTransmilenioTraceRotX;
int **devCPUTransmilenioTraceRotY;


float *transmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout=NULL;
map<int,Zone> idTransmilenioZones;


MobileManager::MobileManager() 
{
	devPediestranX=NULL;
	devPediestranY=NULL;
	devPediestranStep=NULL;
	devPediestranMaxStep=NULL;
	devPediestranClass=NULL;
	devPediestranCurrentX=NULL;
	devPediestranCurrentY=NULL;
	devPediestranPreviousX=NULL;
	devPediestranPreviousY=NULL;
	devPediestranConflicted=NULL;
	devPediestranRelated=NULL;

	devVehicleX=NULL;
	devVehicleY=NULL;
	devVehicleStep=NULL;
	devVehicleMaxStep=NULL;
	devVehicleClass=NULL;
	devVehicleCurrentX=NULL;
	devVehicleCurrentY=NULL;
	devVehiclePreviousX=NULL;
	devVehiclePreviousY=NULL;
	devVehicleConflicted=NULL;
	devVehicleRelated=NULL;

	devTransmilenioX=NULL;
	devTransmilenioY=NULL;
	devTransmilenioStep=NULL;
	devTransmilenioMaxStep=NULL;
	devTransmilenioClass=NULL;
	devTransmilenioCurrentX=NULL;
	devTransmilenioCurrentY=NULL;
	devTransmilenioPreviousX=NULL;
	devTransmilenioPreviousY=NULL;
	devTransmilenioConflicted=NULL;
	devTransmilenioRelated=NULL;
}

MobileManager::~MobileManager()
{

	if (devPediestranStep){
		cudaFree(devPediestranStep);
		devPediestranStep=NULL;
	}
	if (devPediestranMaxStep){
		cudaFree(devPediestranMaxStep);
		devPediestranMaxStep=NULL;
	}
	if (devPediestranClass){
		cudaFree(devPediestranClass);
		devPediestranClass=NULL;
	}
	if (devPediestranCurrentX){
		cudaFree(devPediestranCurrentX);
		devPediestranCurrentX=NULL;
	}
	if (devPediestranCurrentY){
		cudaFree(devPediestranCurrentY);
		devPediestranCurrentY=NULL;
	}
	if (devPediestranPreviousX){
		cudaFree(devPediestranPreviousX);
		devPediestranPreviousX=NULL;
	}
	if (devPediestranPreviousY){
		cudaFree(devPediestranPreviousY);
		devPediestranPreviousY=NULL;
	}
	if (devPediestranConflicted){
		cudaFree(devPediestranConflicted);
		devPediestranConflicted=NULL;
	}

	if (pedestrian){
		delete[] pedestrian;
		pedestrian=NULL;
	}
	if (pediestranZones){
		delete[] pediestranZones;
		pediestranZones=NULL;
	}

	if (pediestranStep){
		delete[] pediestranStep;
		pediestranStep=NULL;
	}
	if (pediestranMaxStep){
		delete[] pediestranMaxStep;
		pediestranMaxStep=NULL;
	}
	if (devPediestranRelated){
		// ahora no lo estoy utilizando, pero yo supondria que es...
		for (int i=0;i<pedestrianNumber;i++){
			cudaFree(devPediestranRelated[i]);
		}
		cudaFree(devPediestranRelated);
		devPediestranRelated=NULL;
	}
	if (pediestranX){
		for (int i=0;i<pedestrianNumber;i++){
			delete[] pediestranX[i];
		}
		delete[] pediestranX;
		pediestranX=NULL;
	}
	if (pediestranY){
		for (int i=0;i<pedestrianNumber;i++){
			delete[] pediestranY[i];
		}
		delete[] pediestranY;
		pediestranY=NULL;
	}

	if (devCPUPediestranX){
		for (int i=0; i<pedestrianNumber;i++){
			cudaFree(devCPUPediestranX[i]);
		}
		delete[] devCPUPediestranX;
		devCPUPediestranX=NULL;
	}
	if (devPediestranX){//no hago un lazo de borrado ya que he borrado las direcciones en la variable anterior
		cudaFree(devPediestranX);
	}
	if (devCPUPediestranY){
		for (int i=0; i<pedestrianNumber;i++){
			cudaFree(devCPUPediestranY[i]);
		}
		delete[] devCPUPediestranY;
		devCPUPediestranY=NULL;
	}
	if (devPediestranY){//no hago un lazo de borrado ya que he borrado las direcciones en la variable anterior
		cudaFree(devPediestranY);
	}

}

void MobileManager::loadFlags(void){

	char *propertyVehicleFile = "./SIMULATOR.properties";   
	ifstream archivo(propertyVehicleFile);
    char linea[128];

	if(archivo.fail()){
		cerr << "Error al abrir el archivo " << propertyVehicleFile << endl;
		cin.get();
		exit(1);
	}else{
		while(!archivo.eof()){
			archivo.getline(linea, sizeof(linea));
			string key;
			key=linea;
			if (key.compare(0,1,"#")==0)
				continue;
			char *t1;
			bool i=true;
			int value=-1;
			key="";
			for ( t1=strtok(linea,"="); t1!=NULL; t1=strtok(NULL,"=") ){
				if (i){
					key=t1;
				}else{
					sscanf(t1, "%i", &value);
					if (key.compare("pedestrianNumber") == 0){
						pedestrianNumber=value;
					}else if (key.compare("vehicleNumber") == 0){
						vehicleNumber=value;
					}else if (key.compare("transmilenioNumber") == 0){
						transmilenioNumber=value;
					}else if (key.compare("maxSemaphoreTime") == 0){
						maxSemaphoreTime=value;
					}else if (key.compare("desiredFPS") == 0){
						desiredFPS=value;
					}
					else if (key.compare("stepByStep") == 0){
						stepByStep = (value == 1);
					}else if (key.compare("Kernel") == 0){
						if (value==0){
							runsOnGPU=false;
							parallelDetection=false;
						}else if (value==1){
							runsOnGPU=true;
							parallelDetection=false;
						}else if (value==2){
							runsOnGPU=true;
							parallelDetection=true;
						}
					}
					value=-1;
					key="";
				}
				i=!i;
			}
		}
	}
    archivo.close();

	/*printf("[pedes-%i]\n", pedestrianNumber);
	printf("[vehi-%i]\n", vehicleNumber);
	printf("[trans-%i]\n", transmilenioNumber);
	printf("[fps-%i]\n", desiredFPS);

	printf("[stepByStep-%s]\n", stepByStep ? "true" : "false");
	printf("[Kernel1-%s]\n", runsOnGPU ? "true" : "false");
	printf("[Kernel2-%s]\n", parallelDetection ? "true" : "false");*/
}

void MobileManager::createZones(char *fileName,Zone *zones, map<int,Zone> &idMap){

	ifstream archivo(fileName);
    char linea[128];
	map<string,float> tempZones;
	vector<string> zoneName;

	if(archivo.fail()){
		cerr << "Error al abrir el archivo " << fileName << endl;
		cin.get();
		exit(1);
	}else{
		while(!archivo.eof()){
			
			archivo.getline(linea, sizeof(linea));
			char *t1;
			bool i=true;
			string key;
			float value;
			for ( t1=strtok(linea,"="); t1!=NULL; t1=strtok(NULL,"=") ){
				if (i){
					key=t1;
					if (key.compare(0,4,"name") == 0){
						zoneName.push_back(key);
					}
				}else{
					sscanf(t1, "%f", &value);
					tempZones[key]=value;
				}
				i=!i;
			}
		}
	}
    archivo.close();

	zones = new Zone[zoneName.size()];
	for(size_t i=0;i<zoneName.size();i++){
		zones[i].parseZone(tempZones , zoneName[i]);
		idMap[zones[i].id]=zones[i];
	}
}



void MobileManager::initPedestrian(void){
	createPedestrianZones();
	createPedestrian();
	allocatePedestrianCPU2GPUMemory();
}

void MobileManager::initVehicle(void){
	createVehicleZones();
	createVehicle();
	allocateVehicleCPU2GPUMemory();
}

void MobileManager::initTransmilenio(void){
	createTransmilenioZones();
	createTransmilenio();
	allocateTransmilenioCPU2GPUMemory();
}

void MobileManager::pedestrianOneStepSimulation(TColor *textureMap, TColor *h_dst, int imageW, int imageH, int maxPediestran, bool semaphore){
	for (int n = 0; n < pedestrianNumber; n++)
		pedestrian[n].oneStepSimulation(textureMap, h_dst, n, imageW, imageH, maxPediestran, semaphore);
}

void MobileManager::vehicleOneStepSimulation(TColor *textureMap, TColor *h_dst, int imageW, int imageH, int maxVehicle, bool semaphore){
	for (int n = 0; n < maxVehicle; n++)
		vehicle[n].oneStepSimulation(textureMap, h_dst, n, imageW, imageH, maxVehicle, semaphore);
}

void MobileManager::transmilenioOneStepSimulation(TColor *textureMap, TColor *h_dst, int imageW, int imageH, int maxTransmilenio, bool semaphore){
	for (int n = 0; n < maxTransmilenio; n++)
		transmilenio[n].oneStepSimulation(textureMap, h_dst, n, imageW, imageH, maxTransmilenio, semaphore);
}


/**************CREACION DE PEDESTRIAN*************/
void MobileManager::createPedestrianZones(void){
	createZones(Pedestrian::getPropertyFileName(), pediestranZones, idPediestranZones);
}

void MobileManager::createPedestrian(void){
	pedestrian = new Pedestrian[pedestrianNumber];

	map<int,int> initZones;
	printf("por aqui\n");
	for (int n = 0; n < pedestrianNumber; n++){
		pedestrian[n].createRoute(idPediestranZones,initZones,1);
	}
}

void MobileManager::allocatePedestrianCPU2GPUMemory(void){

	pediestranX = new int*[pedestrianNumber];
	pediestranY = new int*[pedestrianNumber];
	pediestranStep = new int[pedestrianNumber];
	pediestranMaxStep = new int[pedestrianNumber];
	pediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout = new float[5];

	for (int i=0;i<pedestrianNumber;i++){
		pediestranStep[i] = pedestrian[i].delay;
		pediestranX[i]=pedestrian[i].x;
		pediestranY[i]=pedestrian[i].y;
		pediestranMaxStep[i]=pedestrian[i].size;
	}

	pediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[0]=pedestrian[0].sizeX;
	pediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[1]=pedestrian[0].sizeZ;
	pediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[2]=pedestrian[0].maxSpeed;
	pediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[3]=pedestrian[0].maxAcceleration;
	pediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[4]=(float)pedestrian[0].timeOut;

	/*****BLOQUE DE COPIA para APUNTADOR DE APUNTADORES****/
	devCPUPediestranX= new int*[pedestrianNumber];
	devCPUPediestranY= new int*[pedestrianNumber];

	for (int i=0; i<pedestrianNumber;i++){
		int m=5*pedestrian[i].size;

		cudaMalloc ( (void**)&devCPUPediestranX[i] , m*sizeof(int) );
		cudaMemcpy (devCPUPediestranX[i], pediestranX[i], m*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc ( (void**)&devCPUPediestranY[i] , m*sizeof(int) );
		cudaMemcpy (devCPUPediestranY[i], pediestranY[i], m*sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaMalloc ( (void**)&devPediestranX , pedestrianNumber*sizeof(int*) );
	cudaMemcpy (devPediestranX, devCPUPediestranX, pedestrianNumber*sizeof(int*), cudaMemcpyHostToDevice);
	
	cudaMalloc ( (void**)&devPediestranY , pedestrianNumber*sizeof(int*) );
	cudaMemcpy (devPediestranY, devCPUPediestranY, pedestrianNumber*sizeof(int*), cudaMemcpyHostToDevice);
	/****************************************************/

	cudaMalloc ( (void**)&devPediestranStep , pedestrianNumber*sizeof(int) );
	cudaMemcpy(devPediestranStep, pediestranStep, pedestrianNumber*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devPediestranMaxStep , pedestrianNumber*sizeof(int) );
	cudaMemcpy(devPediestranMaxStep, pediestranMaxStep, pedestrianNumber*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devPediestranClass , imageW*imageH*sizeof(int));

	cudaMalloc ( (void**)&devPediestranCurrentX , pedestrianNumber*sizeof(int) );
	cudaMalloc ( (void**)&devPediestranCurrentY , pedestrianNumber*sizeof(int) );

	cudaMalloc ( (void**)&devPediestranPreviousX , pedestrianNumber*sizeof(int) );
	cudaMalloc ( (void**)&devPediestranPreviousY , pedestrianNumber*sizeof(int) );

	cudaMalloc ( (void**)&devPediestranConflicted , imageW*imageH*sizeof(int) );

	/*************************/
	cudaMalloc ( (void**)&devPediestranTimeOut , pedestrianNumber*sizeof(int) );
	cudaMemset(devPediestranTimeOut,-10,pedestrianNumber*sizeof(int));

	cudaMalloc ( (void**)&devPediestranSpeed , pedestrianNumber*sizeof(float) );
	cudaMemset(devPediestranSpeed,0,pedestrianNumber*sizeof(int));

	cudaMalloc ( (void**)&devPediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout , pedestrianNumber*sizeof(float) );
	cudaMemcpy(devPediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout, pediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout, 5*sizeof(float), cudaMemcpyHostToDevice);


	//cudaMemset(devClass,-1,imageW*imageH*sizeof(int));

}


/**********CREACION DE VEHICULOS*********/

void MobileManager::createVehicleZones(void){
	createZones(Vehicle::getPropertyFileName(), vehicleZones, idVehicleZones);
}


void MobileManager::createVehicle(void){
	vehicle = new Vehicle[vehicleNumber];
	map<int,int> initZones;
	
	for (int n = 0; n < vehicleNumber; n++)
		vehicle[n].createRoute(idVehicleZones,initZones,2);
	
}

void MobileManager::allocateVehicleCPU2GPUMemory(void){
	
	vehicleX = new int*[vehicleNumber];
	vehicleY = new int*[vehicleNumber];
	vehicleStep = new int[vehicleNumber];
	vehicleMaxStep = new int[vehicleNumber];
	vehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout=new float[5];

	for (int i=0;i<vehicleNumber;i++){
		vehicleStep[i] = vehicle[i].delay;
		vehicleX[i]=vehicle[i].x;
		vehicleY[i]=vehicle[i].y;
		vehicleMaxStep[i]=vehicle[i].size;
	}

	vehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[0]=vehicle[0].sizeX;
	vehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[1]=vehicle[0].sizeZ;
	vehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[2]=vehicle[0].maxSpeed;
	vehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[3]=vehicle[0].maxAcceleration;
	vehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[4]=(float)vehicle[0].timeOut;

	/*****BLOQUE DE COPIA para APUNTADOR DE APUNTADORES****/
	devCPUVehicleX= new int*[vehicleNumber];
	devCPUVehicleY= new int*[vehicleNumber];
	
	for (int i=0; i<vehicleNumber;i++){
		int m=5*vehicle[i].size;

		cudaMalloc ( (void**)&devCPUVehicleX[i] , m*sizeof(int) );
		cudaMemcpy (devCPUVehicleX[i], vehicleX[i], m*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc ( (void**)&devCPUVehicleY[i] , m*sizeof(int) );
		cudaMemcpy (devCPUVehicleY[i], vehicleY[i], m*sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaMalloc ( (void**)&devVehicleX , vehicleNumber*sizeof(int*) );
	cudaMemcpy (devVehicleX, devCPUVehicleX, vehicleNumber*sizeof(int*), cudaMemcpyHostToDevice);
	
	cudaMalloc ( (void**)&devVehicleY , vehicleNumber*sizeof(int*) );
	cudaMemcpy (devVehicleY, devCPUVehicleY, vehicleNumber*sizeof(int*), cudaMemcpyHostToDevice);
	/*****************************************************/
	/*****PARA MANEJO DEL TAMANO DEL TRANSMILENIO*****/
	devCPUVehicleTraceX= new int*[vehicleNumber];
	devCPUVehicleTraceY= new int*[vehicleNumber];
	devCPUVehicleTraceRotX= new int*[vehicleNumber];
	devCPUVehicleTraceRotY= new int*[vehicleNumber];
	for (int i=0; i<vehicleNumber;i++){
		int m=vehicle[i].sizeZ;
		cudaMalloc ( (void**)&devCPUVehicleTraceX[i] , m*sizeof(int) );
		cudaMemset(devCPUVehicleTraceX[i],-10,m*sizeof(int));

		cudaMalloc ( (void**)&devCPUVehicleTraceY[i] , m*sizeof(int) );
		cudaMemset(devCPUVehicleTraceY[i],-10,m*sizeof(int));

		cudaMalloc ( (void**)&devCPUVehicleTraceRotX[i] , m*sizeof(int) );
		cudaMemset(devCPUVehicleTraceRotX[i],-10,m*sizeof(int));

		cudaMalloc ( (void**)&devCPUVehicleTraceRotY[i] , m*sizeof(int) );
		cudaMemset(devCPUVehicleTraceRotY[i],-10,m*sizeof(int));
	}

	cudaMalloc ( (void**)&devVehicleTraceX , vehicleNumber*sizeof(int*) );
	cudaMemcpy (devVehicleTraceX, devCPUVehicleTraceX, vehicleNumber*sizeof(int*), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devVehicleTraceY , vehicleNumber*sizeof(int*) );
	cudaMemcpy (devVehicleTraceY, devCPUVehicleTraceY, vehicleNumber*sizeof(int*), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devVehicleTraceRotX , vehicleNumber*sizeof(int*) );
	cudaMemcpy (devVehicleTraceRotX, devCPUVehicleTraceRotX, vehicleNumber*sizeof(int*), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devVehicleTraceRotY , vehicleNumber*sizeof(int*) );
	cudaMemcpy (devVehicleTraceRotY, devCPUVehicleTraceRotY, vehicleNumber*sizeof(int*), cudaMemcpyHostToDevice);
	/*****************************************************/

	
	cudaMalloc ( (void**)&devVehicleStep , vehicleNumber*sizeof(int) );
	cudaMemcpy(devVehicleStep, vehicleStep, vehicleNumber*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devVehicleMaxStep , vehicleNumber*sizeof(int) );
	cudaMemcpy(devVehicleMaxStep, vehicleMaxStep, vehicleNumber*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devVehicleClass , imageW*imageH*sizeof(int));

	cudaMalloc ( (void**)&devVehicleCurrentX , vehicleNumber*sizeof(int) );
	cudaMalloc ( (void**)&devVehicleCurrentY , vehicleNumber*sizeof(int) );

	cudaMalloc ( (void**)&devVehiclePreviousX , vehicleNumber*sizeof(int) );
	cudaMalloc ( (void**)&devVehiclePreviousY , vehicleNumber*sizeof(int) );

	cudaMalloc ( (void**)&devVehicleConflicted , imageW*imageH*sizeof(int) );

	/*************************/
	cudaMalloc ( (void**)&devVehicleTimeOut , vehicleNumber*sizeof(int) );
	cudaMemset(devVehicleTimeOut,-10,vehicleNumber*sizeof(int));

	cudaMalloc ( (void**)&devVehicleSpeed , vehicleNumber*sizeof(float) );
	cudaMemset(devVehicleSpeed,0,vehicleNumber*sizeof(int));

	cudaMalloc ( (void**)&devVehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout , vehicleNumber*sizeof(float) );
	cudaMemcpy(devVehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout, vehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout, 5*sizeof(float), cudaMemcpyHostToDevice);
	
}




/**********CREACION DE TRANSMILENIO*********/
void MobileManager::createTransmilenioZones(void){
	createZones(Transmilenio::getPropertyFileName(), transmilenioZones, idTransmilenioZones);
}


void MobileManager::createTransmilenio(void){
	transmilenio = new Transmilenio[transmilenioNumber];
	map<int,int> initZones;
	
	for (int n = 0; n < transmilenioNumber; n++)
		transmilenio[n].createRoute(idTransmilenioZones,initZones,3);
	
}

void MobileManager::allocateTransmilenioCPU2GPUMemory(void){
	
	transmilenioX = new int*[transmilenioNumber];
	transmilenioY = new int*[transmilenioNumber];
	transmilenioStep = new int[transmilenioNumber];
	transmilenioMaxStep = new int[transmilenioNumber];
	transmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout = new float[5];

	for (int i=0;i<transmilenioNumber;i++){
		transmilenioStep[i] = transmilenio[i].delay;
		transmilenioX[i]=transmilenio[i].x;
		transmilenioY[i]=transmilenio[i].y;
		transmilenioMaxStep[i]=transmilenio[i].size;
	}

	transmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[0]=transmilenio[0].sizeX;
	transmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[1]=transmilenio[0].sizeZ;
	transmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[2]=transmilenio[0].maxSpeed;
	transmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[3]=transmilenio[0].maxAcceleration;
	transmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout[4]=(float)transmilenio[0].timeOut;

	/*****BLOQUE DE COPIA para APUNTADOR DE APUNTADORES****/
	devCPUTransmilenioX= new int*[transmilenioNumber];
	devCPUTransmilenioY= new int*[transmilenioNumber];
	
	for (int i=0; i<transmilenioNumber;i++){
		int m=5*transmilenio[i].size;

		cudaMalloc ( (void**)&devCPUTransmilenioX[i] , m*sizeof(int) );
		cudaMemcpy (devCPUTransmilenioX[i], transmilenioX[i], m*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc ( (void**)&devCPUTransmilenioY[i] , m*sizeof(int) );
		cudaMemcpy (devCPUTransmilenioY[i], transmilenioY[i], m*sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaMalloc ( (void**)&devTransmilenioX , transmilenioNumber*sizeof(int*) );
	cudaMemcpy (devTransmilenioX, devCPUTransmilenioX, transmilenioNumber*sizeof(int*), cudaMemcpyHostToDevice);
	
	cudaMalloc ( (void**)&devTransmilenioY , transmilenioNumber*sizeof(int*) );
	cudaMemcpy (devTransmilenioY, devCPUTransmilenioY, transmilenioNumber*sizeof(int*), cudaMemcpyHostToDevice);
	/*****************************************************/

	/*****PARA MANEJO DEL TAMANO DEL TRANSMILENIO*****/
	devCPUTransmilenioTraceX= new int*[transmilenioNumber];
	devCPUTransmilenioTraceY= new int*[transmilenioNumber];
	devCPUTransmilenioTraceRotX= new int*[transmilenioNumber];
	devCPUTransmilenioTraceRotY= new int*[transmilenioNumber];
	for (int i=0; i<transmilenioNumber;i++){
		int m=transmilenio[i].sizeZ;
		cudaMalloc ( (void**)&devCPUTransmilenioTraceX[i] , m*sizeof(int) );
		cudaMemset(devCPUTransmilenioTraceX[i],-10,m*sizeof(int));

		cudaMalloc ( (void**)&devCPUTransmilenioTraceY[i] , m*sizeof(int) );
		cudaMemset(devCPUTransmilenioTraceY[i],-10,m*sizeof(int));

		cudaMalloc ( (void**)&devCPUTransmilenioTraceRotX[i] , m*sizeof(int) );
		cudaMemset(devCPUTransmilenioTraceRotX[i],-10,m*sizeof(int));

		cudaMalloc ( (void**)&devCPUTransmilenioTraceRotY[i] , m*sizeof(int) );
		cudaMemset(devCPUTransmilenioTraceRotY[i],-10,m*sizeof(int));
	}

	cudaMalloc ( (void**)&devTransmilenioTraceX , transmilenioNumber*sizeof(int*) );
	cudaMemcpy (devTransmilenioTraceX, devCPUTransmilenioTraceX, transmilenioNumber*sizeof(int*), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devTransmilenioTraceY , transmilenioNumber*sizeof(int*) );
	cudaMemcpy (devTransmilenioTraceY, devCPUTransmilenioTraceY, transmilenioNumber*sizeof(int*), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devTransmilenioTraceRotX , transmilenioNumber*sizeof(int*) );
	cudaMemcpy (devTransmilenioTraceRotX, devCPUTransmilenioTraceRotX, transmilenioNumber*sizeof(int*), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devTransmilenioTraceRotY , transmilenioNumber*sizeof(int*) );
	cudaMemcpy (devTransmilenioTraceRotY, devCPUTransmilenioTraceRotY, transmilenioNumber*sizeof(int*), cudaMemcpyHostToDevice);
	/*****************************************************/

	cudaMalloc ( (void**)&devTransmilenioStep , transmilenioNumber*sizeof(int) );
	cudaMemcpy(devTransmilenioStep, transmilenioStep, transmilenioNumber*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devTransmilenioMaxStep , transmilenioNumber*sizeof(int) );
	cudaMemcpy(devTransmilenioMaxStep, transmilenioMaxStep, transmilenioNumber*sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc ( (void**)&devTransmilenioClass , imageW*imageH*sizeof(int));

	cudaMalloc ( (void**)&devTransmilenioCurrentX , transmilenioNumber*sizeof(int) );
	cudaMalloc ( (void**)&devTransmilenioCurrentY , transmilenioNumber*sizeof(int) );

	cudaMalloc ( (void**)&devTransmilenioPreviousX , transmilenioNumber*sizeof(int) );
	cudaMalloc ( (void**)&devTransmilenioPreviousY , transmilenioNumber*sizeof(int) );

	cudaMalloc ( (void**)&devTransmilenioConflicted , imageW*imageH*sizeof(int) );
	
	/*************************/
	cudaMalloc ( (void**)&devTransmilenioTimeOut , transmilenioNumber*sizeof(int) );
	cudaMemset(devTransmilenioTimeOut,-10,transmilenioNumber*sizeof(int));

	cudaMalloc ( (void**)&devTransmilenioSpeed , transmilenioNumber*sizeof(float) );
	cudaMemset(devTransmilenioSpeed,0,transmilenioNumber*sizeof(int));

	cudaMalloc ( (void**)&devTransmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout , transmilenioNumber*sizeof(float) );
	cudaMemcpy(devTransmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout, transmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout, 5*sizeof(float), cudaMemcpyHostToDevice);

}


bool MobileManager::getSemaphoreState(void){
	return semaphoreState;
}

void MobileManager::updateSemaphore(void){
	semaphoreCounter++;
	if (semaphoreCounter>maxSemaphoreTime){
		semaphoreCounter=0;
		semaphoreState = !semaphoreState;
	}
}


/***** por ahora no utilizarlos parece haber problemas con las direcciones de los apuntadores.... ****/
void MobileManager::allocateDeviceMemory(int *devicePointer, int size){
	cudaMalloc ( (void**)&devicePointer , size*sizeof(int) );
}
void MobileManager::copyHostToDeviceLevel1(int *hostPointer, int *devicePointer, int size){
	cudaMalloc ( (void**)&devicePointer , size*sizeof(int) );
	cudaMemcpy(devicePointer, hostPointer, size*sizeof(int), cudaMemcpyHostToDevice);
}

void MobileManager::copyHostToDeviceLevel2(int **hostPointer, int **devicePointer, int size){
	cudaMalloc ( (void**)&devicePointer , size*sizeof(int*) );
	cudaMemcpy(devicePointer, hostPointer, size*sizeof(int*), cudaMemcpyHostToDevice);
}

void MobileManager::copyDeviceToHostLevel1(int *hostPointer, int *devicePointer, int size){
	cudaMemcpy(devicePointer, hostPointer, size*sizeof(int), cudaMemcpyDeviceToHost);
}

