/*
 * Copyright 2012 Diego Hernando Rodriguez Gaitan.  Licenced by GPL.
 *
 *
 */
#ifndef __TRANSMILENIO_H__ //Guardas
#define __TRANSMILENIO_H__

#include <map>
#include <vector>

#include "Mobile.h"
#include "Zone.h"

//using namespace std;

class Transmilenio : public Mobile {
public:
	Transmilenio();
	~Transmilenio();
	static char *getPropertyFileName(void);
	
	static int getExcludeSize(void);
	static int *getExcludePoint(void);

	static int getStartingSize(void);
	static int *getStartingPoint(void);

	static int getFinishSize(void);
	static int *getFinishPoint(void);

	static int *getDelay(void);

	void oneStepSimulation(TColor *textureMap, TColor *dst, int id, int imageW, int imageH, int maxPediestran, bool semaphore);
	bool isColissionForTransmilenio(TColor *dst, int id, int imageW, int imageH, int x, int y, int dx, int dy);
	void getFirstStepForTransmilenio(TColor *dst, int id, int imageW, int imageH, int x, int y, int &px, int &py);
	void frontSidersForTransmilenio(int id, int rx, int ry, int &dx, int &dy);
	bool isFrontCollisionForTransmilenio(TColor *dst, int id, int imageW, int imageH, int px, int py, int x, int y, int dx, int dy, int rightSize, int leftSize);
	void getNextStepForTransmilenio(TColor *dst, int id, int imageW, int imageH, int x, int y, int &px, int &py);

};

#endif