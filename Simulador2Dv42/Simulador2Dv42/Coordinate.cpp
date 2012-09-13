#include "Coordinate.h"

Coordinate::Coordinate() {pixelX=0; pixelX=0;}
//Coordinate::Coordinate(int x, int y) : x(x),y(y) {}
//Coordinate::Coordinate(int x, int y) {this->x=x; this->y=y;}
Coordinate::~Coordinate() {}
void Coordinate::setCoordinateValue(float x, float y) {
	valueX = x;
	valueY = y;
	pixelX = imageW - (int)(cosiente*( x - minX));
	pixelY = (int)(cosiente*( y - minY));
}


