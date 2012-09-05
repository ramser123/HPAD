/*
 * Copyright 2012 Diego Hernando Rodriguez Gaitan.  Licenced by GPL.
 *
 *
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <conio.h>
#include <string>
#include <sstream>
#include <cstdlib>
using namespace std;

#include "Mobile.h"
#include "Pedestrian.h"
#include "Vehicle.h"
#include "Transmilenio.h"

const int maxGoals = 50;
int delayCounter=0;

Mobile::Mobile() 
{
	x=NULL;
	y=NULL;
}
Mobile::~Mobile()
{
	if (x){
		delete[] x;
		x=NULL;
	}
	if (y){
		delete[] y;
		y=NULL;
	}
}


void Mobile::createRoute(map<int,Zone> &idZones, map<int,int> &initCounter, int type) {
	delay=0;
	int excludeSize;
	int *excludePoint;

	int startingSize;
	int *startingPoint;

	int finishSize;
	int *finishPoint;

	if (type==1){
		excludeSize=Pedestrian::getExcludeSize();
		excludePoint=Pedestrian::getExcludePoint();
		startingSize=Pedestrian::getStartingSize();
		startingPoint=Pedestrian::getStartingPoint();
		finishSize=Pedestrian::getFinishSize();
		finishPoint=Pedestrian::getFinishPoint();
		delay=*Pedestrian::getDelay();
	}else if (type==2){
		excludeSize=Vehicle::getExcludeSize();
		excludePoint=Vehicle::getExcludePoint();
		startingSize=Vehicle::getStartingSize();
		startingPoint=Vehicle::getStartingPoint();
		finishSize=Vehicle::getFinishSize();
		finishPoint=Vehicle::getFinishPoint();
		delay=*Vehicle::getDelay();
	}else{
		excludeSize=Transmilenio::getExcludeSize();
		excludePoint=Transmilenio::getExcludePoint();
		startingSize=Transmilenio::getStartingSize();
		startingPoint=Transmilenio::getStartingPoint();
		finishSize=Transmilenio::getFinishSize();
		finishPoint=Transmilenio::getFinishPoint();
		delay=*Transmilenio::getDelay();
	}
	
	int zonesSize= (int) idZones.size();
	float random=(float)rand()/(float)RAND_MAX;

	int idInicial = (int)( 1+(float)zonesSize*random);

	//verifico si tiene que cumplir con un punto de inicio punto de inicio. -1 si no importa
	if (startingPoint[0]!=-1){
		int pos=(int) (0.5f+(float)startingSize*random);
		pos = pos>startingSize-1 ?  startingSize-1 : pos;
		idInicial=startingPoint[pos];
	}
	//verifico si es un punto para omitir
	while ( containsValue(idInicial,excludePoint,excludeSize) ){
		if (excludePoint[0]==-1)
			break;
		random=(float)rand()/(float)RAND_MAX;
		idInicial= (int)( 1+(float)zonesSize*random);
	}

	int tries=0;
	int maxSizeLocal= (int) ((maxGoals+(float)maxGoals*(float)rand()/(float)RAND_MAX)/2);
	
	localGoalsVector.push_back(idZones.find(idInicial)->second);

	for (int n=0;n<maxSizeLocal;n++){
		int id=Zone::getNextZone(idZones,localGoalsVector.back());

		if (id!=0){
			bool repeat=false;

			for (int m=0;m<localGoalsVector.size();m++){
				if (id==localGoalsVector[m].id){
					repeat=true;
					tries++;
					n--;
					break;
				}
			}
			if (!repeat){
				tries=0;
				localGoalsVector.push_back(idZones.find(id)->second);
			}
			
		}else{
			tries++;
			n--;
		}
		//agrego si es un punto final, lo ultimo del if
		if (localGoalsVector.size() > (maxSizeLocal-1) || tries > maxSizeLocal ||containsValue(id,finishPoint,finishSize) )
			break;
	}
	
	map<int,int>::iterator iter = initCounter.find(idInicial);
	if (iter != initCounter.end() ){
		int value = iter->second;
		//delay=-3*value;
		iter->second++;
        //cout << "Value is: " << iter->second << '\n';
	}else{
		initCounter[idInicial]=1;
	}

	size=(int) localGoalsVector.size();

	x= new int[5*size];
	y= new int[5*size];

	for (int i=0;i<localGoalsVector.size();i++){
		x[5*i+0]=localGoalsVector[i].coor[0].pixelX;
		y[5*i+0]=localGoalsVector[i].coor[0].pixelY;

		x[5*i+1]=localGoalsVector[i].coor[1].pixelX;
		y[5*i+1]=localGoalsVector[i].coor[1].pixelY;

		x[5*i+2]=localGoalsVector[i].coor[2].pixelX;
		y[5*i+2]=localGoalsVector[i].coor[2].pixelY;

		x[5*i+3]=localGoalsVector[i].coor[3].pixelX;
		y[5*i+3]=localGoalsVector[i].coor[3].pixelY;

		x[5*i+4]=localGoalsVector[i].center.pixelX;
		y[5*i+4]=localGoalsVector[i].center.pixelY;
	}

}


bool Mobile::containsValue(int value, int *data, int size){
	for (int n=0; n < size; n++){
		if (data[n]==value)
			return true;
	}
	return false;
}

TColor Mobile::make_color(float r, float g, float b, float a){
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

float Mobile::absf(float f){
	return (f > 0) ? f : -f;
}

int Mobile::absMax(int a, int b){
	int c = (a>0) ? a : -a;
	int d = (b>0) ? b : -b;
	return (c > d) ? c : d;
}

void Mobile::eraseOneTrace(TColor *textureMap, TColor *dst, int imageW, int imageH, int px, int py, int rx, int ry, int leftSize, int rightSize){

	if (px >= imageW || px < 1)
		return;
	if (py >= imageH || py < 1)
		return;

	dst[imageW*(py) + (px)]=textureMap[imageW*(py) + (px)];
	if (rx==0){ //para direccion arriba-abajo
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py) + (px+n*ry)]=textureMap[imageW*(py) + (px+n*ry)];
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py) + (px-n*ry)]=textureMap[imageW*(py) + (px-n*ry)];
		}
	}else if (ry==0){ //para direccion izquierda-derecha
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py-n*rx) + (px)]=textureMap[imageW*(py-n*rx) + (px)];
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py+n*rx) + (px)]=textureMap[imageW*(py+n*rx) + (px)];
		}
	}else if (rx==ry){ //para diagonal so-ne
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py-n*ry) + (px+n*rx)]=textureMap[imageW*(py-n*ry) + (px+n*rx)];
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py+n*ry) + (px-n*rx)]=textureMap[imageW*(py+n*ry) + (px-n*rx)];
		}
	}else if (rx==-ry){ //para diagonal se-no
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py+n*ry) + (px-n*rx)]=textureMap[imageW*(py+n*ry) + (px-n*rx)];
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py-n*ry) + (px+n*rx)]=textureMap[imageW*(py-n*ry) + (px+n*rx)];
		}
	}

}

void Mobile::drawOneTrace(TColor *dst, TColor color, int imageW, int imageH, int px, int py, int rx, int ry, int leftSize, int rightSize){
	if (px >= imageW || px < 1)
		return;
	if (py >= imageH || py < 1)
		return;

	dst[imageW*(py) + (px)]=color;

	if (rx==0){ //para direccion arriba-abajo
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py) + (px+n*ry)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py) + (px-n*ry)]=color;
		}
	}else if (ry==0){ //para direccion izquierda-derecha
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py-n*rx) + (px)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py+n*rx) + (px)]=color;
		}
	}else if (rx==ry){ //para diagonal so-ne
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py-n*ry) + (px+n*rx)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py+n*ry) + (px-n*rx)]=color;
		}
	}else if (rx==-ry){ //para diagonal se-no
		for(int n=1; n<rightSize+1; n++){
			dst[imageW*(py+n*ry) + (px-n*rx)]=color;
		}
		for(int n=1; n<leftSize+1; n++){
			dst[imageW*(py-n*ry) + (px+n*rx)]=color;
		}
	}
}

void Mobile::drawAllTrace(TColor *textureMap, TColor *dst, TColor color, int imageW, int imageH, int x, int y, int dx, int dy, int *traceX, int *traceY, int *traceRotX, int *traceRotY, float sizeX, float sizeZ){
	
	if (x==traceX[0] && y==traceY[0])
		return;

	int size = (sizeX-1)/2;
	int res = (( sizeX - 1.f )/2.f - (float)size) * 2;//residuo, cuando no es impar da 1
	int leftSize=size;
	int rightSize=size+res;
	
	//eraseOneTrace(textureMap, dst, imageW, imageH, traceX[(int)sizeZ-1], traceY[(int)sizeZ-1], traceRotX[(int)sizeZ-1], traceRotY[(int)sizeZ-1], leftSize, rightSize);
	for (int n=0; n<sizeZ; n++){
		eraseOneTrace(textureMap, dst, imageW, imageH, traceX[n], traceY[n], traceRotX[n], traceRotY[n], leftSize, rightSize);
	}

	/** Esto es para hacer corrimiento del trace, para actualizarlo **/
	for (int n=sizeZ-1; n > 0; n--){
		traceX[n]=traceX[n-1];
		traceY[n]=traceY[n-1];
		traceRotX[n]=traceRotX[n-1];
		traceRotY[n]=traceRotY[n-1];
	}
	traceX[0]=x;
	traceY[0]=y;
	//traceRotX[0]=dx;
	traceRotX[0]=0;
	traceRotY[0]=dy;
	
	/** se dibuja el trace actual **/
	//drawOneTrace(dst, color, imageW, imageH, traceX[0], traceY[0], traceRotX[0], traceRotY[0], leftSize, rightSize);
	for (int n=0; n < sizeZ ; n++){
		drawOneTrace(dst, color, imageW, imageH, traceX[n], traceY[n], traceRotX[n], traceRotY[n], leftSize, rightSize);
	}
}