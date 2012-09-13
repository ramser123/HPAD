#ifndef __ZONE_H__ //Guardas
#define __ZONE_H__

#include <map>
#include <string>

#include "Coordinate.h"

extern int imageW, imageH;
extern const float cosiente;
extern const float minY;
extern const float minX;

class Zone {

public:
    int id;
    int northID;
	int southID;
	int westID;
	int eastID;
	Coordinate coor[4];
	Coordinate center;
	Zone();
	~Zone();
	void parseZone(std::map<std::string,float> &pedestrianZones, std::string name);
	std::string convertInt(int number);
	static int getNextZone(std::map<int,Zone> &idZones, Zone &zone);
	static int getNextZone(std::map<int,Zone> &idZones, Zone &tempZone, Zone &finZone);
	static float getDistance(Zone &z1, Zone &z2);
};

#endif