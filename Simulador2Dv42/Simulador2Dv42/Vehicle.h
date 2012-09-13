#ifndef __VEHICLE_H__ //Guardas
#define __VEHICLE_H__

#include <map>
#include <vector>

#include "Mobile.h"
#include "Zone.h"

using namespace std;

class Vehicle : public Mobile {
public:
	Vehicle();
	~Vehicle();
	static char *getPropertyFileName(void);

	static int getExcludeSize(void);
	static int *getExcludePoint(void);

	static int getStartingSize(void);
	static int *getStartingPoint(void);

	static int getFinishSize(void);
	static int *getFinishPoint(void);

	static int *getDelay(void);

	void oneStepSimulation(TColor *textureMap, TColor *dst, int id, int imageW, int imageH, int maxPediestran, bool semaphore);
	bool isColissionForVehicle(TColor *dst, int id, int imageW, int imageH, int x, int y, int dx, int dy);
	void getFirstStepForVehicle(TColor *dst, int id, int imageW, int imageH, int x, int y, int &px, int &py);
	void frontSidersForVehicle(int id, int rx, int ry, int &dx, int &dy);
	bool isFrontCollisionForVehicle(TColor *dst, int id, int imageW, int imageH, int px, int py, int x, int y, int dx, int dy, int rightSize, int leftSize);
	void getNextStepForVehicle(TColor *dst, int id, int imageW, int imageH, int x, int y, int &px, int &py);
};
#endif