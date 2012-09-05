/*
 * Copyright 2012 Diego Hernando Rodriguez Gaitan.  Licenced by GPL.
 *
 *
 */
#ifndef __PEDESTRIAN_H__ //Guardas
#define __PEDESTRIAN_H__

#include <map>
#include <vector>

#include "Mobile.h"
#include "Zone.h"

class Pedestrian : public Mobile {
public:
	Pedestrian();
	~Pedestrian();
	static char *getPropertyFileName(void);
	
	static int getExcludeSize(void);
	static int *getExcludePoint(void);

	static int getStartingSize(void);
	static int *getStartingPoint(void);

	static int getFinishSize(void);
	static int *getFinishPoint(void);

	static int *getDelay(void);

	void oneStepSimulation(TColor *textureMap, TColor *dst, int id, int imageW, int imageH, int maxPediestran, bool semaphore);
	bool isColission(TColor *dst, int imageW, int imageH, int x, int y);
};

#endif