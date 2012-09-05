/*
 * Copyright 2012 Diego Hernando Rodriguez Gaitan.  Licenced by GPL.
 *
 *
 */
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iomanip>
#include <fstream>
#include <conio.h>
#include <vector>
#include <sstream>
#include <cstdlib>
using namespace std;

#include "Zone.h"

Zone::Zone()
	{
		id= -1;
		northID= -1;
		southID= -1;
		westID= -1;
		eastID= -1;
	}

Zone::~Zone()
	{
	}

void Zone::parseZone(map<string,float> &pedestrianZones, string name){
	id=(int) (pedestrianZones.find(name)->second);

	string temp="north."+convertInt(id);
	northID=(int) (pedestrianZones.find(temp)->second);

	temp="south."+convertInt(id);
	southID=(int) (pedestrianZones.find(temp)->second);

	temp="west."+convertInt(id);
	westID=(int) (pedestrianZones.find(temp)->second);

	temp="east."+convertInt(id);
	eastID=(int) (pedestrianZones.find(temp)->second);

	float sumX=0;
	float sumY=0;
	for (int i = 0; i < 4; i++ ){
		float x=0;
		float y=0;
		temp="zone."+convertInt(id)+".x"+convertInt(i);
		x = -pedestrianZones.find(temp)->second;
		temp="zone."+convertInt(id)+".y"+convertInt(i);
		y = pedestrianZones.find(temp)->second;
		coor[i].setCoordinateValue(x,y);
		sumX+=x;
		sumY+=y;
	}
	center.setCoordinateValue(sumX/4.f,sumY/4.f);

}

string Zone::convertInt(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

int Zone::getNextZone(map<int,Zone> &idZones, Zone &tempZone, Zone &finZone){

	float d1=1000000000.f;
	float d2=1000000000.f;
	float d3=1000000000.f;
	float d4=1000000000.f;

	map<int,Zone>::iterator iter=idZones.find(tempZone.northID);
	if (iter != idZones.end() )
		d1=getDistance(idZones.find(tempZone.northID)->second,finZone);

	iter=idZones.find(tempZone.southID);
	if (iter != idZones.end() )
		d2=getDistance(idZones.find(tempZone.southID)->second,finZone);

	iter=idZones.find(tempZone.westID);
	if (iter != idZones.end() )
		d3=getDistance(idZones.find(tempZone.westID)->second,finZone);

	iter=idZones.find(tempZone.eastID);
	if (iter != idZones.end() )
		d4=getDistance(idZones.find(tempZone.eastID)->second,finZone);

	if (d1<d2 && d1<d3 && d1<d4)
		return tempZone.northID;
	if (d2<d1 && d2<d3 && d2<d4)
		return tempZone.southID;
	if (d3<d2 && d3<d1 && d3<d4)
		return tempZone.westID;
	if (d4<d2 && d4<d3 && d4<d1)
		return tempZone.eastID;
		
	return -1;
}

int Zone::getNextZone(map<int,Zone> &idZones, Zone &zone){

	int id = (int) ((float)4*(float)rand()/(float)RAND_MAX);
	if (id>3)
		id=3;
	
	switch(id){
		case 0:
			return zone.northID;
		break;
		case 1:
			return zone.southID;
		break;
		case 2:
			return zone.westID;
		break;
		case 3:
			return zone.eastID;
		break;
		/*default:
			return zone.eastID;*/
	}
	return 0;
}


float Zone::getDistance(Zone &z1, Zone &z2){
	float x = pow( (float)(z1.center.pixelX-z2.center.pixelX) , 2);
	float y = pow( (float)(z1.center.pixelY-z2.center.pixelY) , 2);
	return sqrt(x+y);
}