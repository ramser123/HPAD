/*
 * Copyright 2012 Diego Hernando Rodriguez Gaitan.  Licenced by GPL.
 *
 *
 */


#ifndef __MOBILE_H__ //Guardas
#define __MOBILE_H__

#include <map>
#include <vector>
#include <math.h>

#include "Zone.h"

typedef unsigned int TColor;

class Mobile {
public:
	Mobile();
	~Mobile();
	Zone iniZone; //aqui solo guardo la zona inicial
    Zone finZone; //aqui solo guardo la zona final
	float sizeX;
	float sizeY;
	float sizeZ;
	float maxSpeed;
	float maxAcceleration;
	int timeOut;
	int timeCounter;

	int delay;
	int *x;
	int *y;
	int size;

	int nextX;
	int nextY;
	int previousX;
	int previousY;

	int *traceX;
	int *traceY;
	int *traceRotX;
	int *traceRotY;

	std::vector<Zone> localGoalsVector; //lista de metas locales (vector primera vez)
	std::vector<Coordinate> stepSimulationVector; //lista de metas locales (vector primera vez)
	void createRoute(std::map<int,Zone> &idZones, std::map<int,int> &initCounter, int type);

	bool containsValue(int value, int *data, int size);
	TColor make_color(float r, float g, float b, float a);
	float absf(float f);
	int absMax(int a, int b);

	void eraseOneTrace(TColor *textureMap, TColor *dst, int imageW, int imageH, int px, int py, int rx, int ry, int leftSize, int rightSize);
	void drawOneTrace(TColor *dst, TColor color, int imageW, int imageH, int px, int py, int rx, int ry, int leftSize, int rightSize);
	void drawAllTrace(TColor *textureMap, TColor *dst, TColor color, int imageW, int imageH, int x, int y, int dx, int dy, int *traceX, int *traceY, int *traceRotX, int *traceRotY, float sizeX, float sizeZ);

};

#endif