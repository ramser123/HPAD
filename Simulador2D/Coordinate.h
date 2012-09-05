/*
 * Copyright 2012 Diego Hernando Rodriguez Gaitan.  Licenced by GPL.
 *
 *
 */

#ifndef __COORDINATE_H__ //Guardas
#define __COORDINATE_H__

extern int imageW, imageH;
extern const float cosiente;
extern const float minY;
extern const float minX;

class Coordinate {

public:
	Coordinate();
	~Coordinate();
	int pixelX;
	int pixelY;
	float valueX;
	float valueY;
	void setCoordinateValue(float x, float y);

};

#endif