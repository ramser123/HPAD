#ifndef __MOBILE_MANAGER_H__ //Guardas
#define __MOBILE_MANAGER_H__

#include <map>

#include "Pedestrian.h"
#include "Vehicle.h"
#include "Transmilenio.h"
//typedef unsigned int TColor;

/*int pedestrianNumber=0;//6932;
int vehicleNumber=0;
int transmilenioNumber=0;
int maxSemaphoreTime=0;

int desiredFPS=0;
bool stepByStep=false;
bool runsOnGPU=true;
bool parallelDetection=true;*/


class MobileManager{
public:
	MobileManager();
	~MobileManager();
	std::map<int,Vehicle> id_vehicles;
	std::map<int,Pedestrian> id_Pedestrian;

	void loadFlags(void);
	void initPedestrian(void);
	void initVehicle(void);
	void initTransmilenio(void);

	void pedestrianOneStepSimulation(TColor *textureMap, TColor *h_dst, int imageW, int imageH, int maxPediestran, bool semaphore);
	void vehicleOneStepSimulation(TColor *textureMap, TColor *h_dst, int imageW, int imageH, int maxVehicle, bool semaphore);
	void transmilenioOneStepSimulation(TColor *textureMap, TColor *h_dst, int imageW, int imageH, int maxTransmilenio, bool semaphore);

	bool getSemaphoreState(void);
	void updateSemaphore(void);

	int pedestrianNumber;
	int vehicleNumber;
	int	transmilenioNumber;
	int maxSemaphoreTime;
	int desiredFPS;
	bool stepByStep;
	bool runsOnGPU;
	bool parallelDetection;

	//Grupo de variables para las metas es de peatones.

	int **devPediestranX;
	int **devPediestranY;
	int *devPediestranStep;
	int *devPediestranMaxStep;
	int *devPediestranClass; //para solucion de conflictos
	int *devPediestranCurrentX;
	int *devPediestranCurrentY;
	int *devPediestranPreviousX;
	int *devPediestranPreviousY;
	int *devPediestranConflicted; //para solucion de conflictos
	int **devPediestranRelated; //para solucion de conflictos
	int *devPediestranTimeOut;
	float *devPediestranSpeed;
	float *devPediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout;

	int **devVehicleX;
	int **devVehicleY;
	int *devVehicleStep;
	int *devVehicleMaxStep;
	int *devVehicleClass;
	int *devVehicleCurrentX;
	int *devVehicleCurrentY;
	int *devVehiclePreviousX;
	int *devVehiclePreviousY;
	int *devVehicleConflicted;
	int **devVehicleRelated;
	int *devVehicleTimeOut;
	int **devVehicleTraceX;
	int **devVehicleTraceY;
	int **devVehicleTraceRotX;
	int **devVehicleTraceRotY;
	float *devVehicleSpeed;
	float *devVehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout;

	int **devTransmilenioX;
	int **devTransmilenioY;
	int *devTransmilenioStep;
	int *devTransmilenioMaxStep;
	int *devTransmilenioClass;
	int *devTransmilenioCurrentX;
	int *devTransmilenioCurrentY;
	int *devTransmilenioPreviousX;
	int *devTransmilenioPreviousY;
	int *devTransmilenioConflicted;
	int **devTransmilenioRelated;
	int *devTransmilenioTimeOut;
	int **devTransmilenioTraceX;
	int **devTransmilenioTraceY;
	int **devTransmilenioTraceRotX;
	int **devTransmilenioTraceRotY;
	float *devTransmilenioSpeed;
	float *devTransmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout;
	

private:

	void createZones(char *fileName,Zone *zones, map<int,Zone> &idMap);

	void createPedestrianZones(void);
	void createPedestrian(void);
	void allocatePedestrianCPU2GPUMemory(void);

	void createVehicleZones(void);
	void createVehicle(void);
	void allocateVehicleCPU2GPUMemory(void);

	void createTransmilenioZones(void);
	void createTransmilenio(void);
	void allocateTransmilenioCPU2GPUMemory(void);

	void allocateDeviceMemory(int *devicePointer, int size);
	void copyHostToDeviceLevel1(int *hostPointer, int *devicePointer, int size);
	void copyHostToDeviceLevel2(int **hostPointer, int **devicePointer, int size);
	void copyDeviceToHostLevel1(int *hostPointer, int *devicePointer, int size);

};

#endif