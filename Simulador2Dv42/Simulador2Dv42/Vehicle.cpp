#include "Vehicle.h"

char *propertyVehicleFile = "./prop/TRAFFIC_GOAL.properties";

int *excludeVehicle=NULL;
const int excludeVehicleSize=1;
const int excludeVehicleIds[excludeVehicleSize] = {-1};

int *startingVehicle=NULL;
const int startingVehicleSize=6;
const int startingVehicleIds[startingVehicleSize] = { 8,69,10,68,70,42};

int *finishVehicle=NULL;
const int finishVehicleSize=1;
const int finishVehicleIds[finishVehicleSize] = {-1};

static int *delayVehicle=NULL;

Vehicle::Vehicle() 
{
	sizeX=1.f;
	sizeY=2.f;
	sizeZ=3.f;
	maxSpeed=0.5f;
	//maxSpeed=0.05f;
	maxAcceleration=0.1f;
	timeOut=0;

	timeCounter=-10;
	nextX=0;
	nextY=0;
	previousX=0;
	previousY=0;

	traceX=new int[(int)sizeZ];
	traceY=new int[(int)sizeZ];
	traceRotX=new int[(int)sizeZ];
	traceRotY=new int[(int)sizeZ];
	for(int n=0;n++;n<sizeZ){
		traceX[n]=-10;
		traceY[n]=-10;
		traceRotX[n]=-10;
		traceRotY[n]=-10;
	}
}
Vehicle::~Vehicle()
{
	if (excludeVehicle)
		delete[] excludeVehicle;
	if (startingVehicle)
		delete[] startingVehicle;
	if (finishVehicle)
		delete[] finishVehicle;
	if (delayVehicle)
		delete delayVehicle;

}

char *Vehicle::getPropertyFileName(void){
	return propertyVehicleFile;
}

int Vehicle::getExcludeSize(void){
	return excludeVehicleSize;
}
int *Vehicle::getExcludePoint(void){
	if(!excludeVehicle){
		excludeVehicle = new int[excludeVehicleSize];
		for (int n=0; n<excludeVehicleSize; n++)
			excludeVehicle[n]=excludeVehicleIds[n];

	}
	return excludeVehicle;
}


int Vehicle::getStartingSize(void){
	return startingVehicleSize;
}
int *Vehicle::getStartingPoint(void){
	if(!startingVehicle){
		startingVehicle = new int[startingVehicleSize];
		for (int n=0; n<startingVehicleSize; n++)
			startingVehicle[n]=startingVehicleIds[n];

	}
	return startingVehicle;
}


int Vehicle::getFinishSize(void){
	return finishVehicleSize;
}
int *Vehicle::getFinishPoint(void){
	if(!finishVehicle){
		finishVehicle = new int[finishVehicleSize];
		for (int n=0; n<finishVehicleSize; n++)
			finishVehicle[n]=finishVehicleIds[n];

	}
	return finishVehicle;
}


int *Vehicle::getDelay(void){
	if(!delayVehicle){
		delayVehicle = new int(0);
		return delayVehicle;
	}
	*delayVehicle-=20;
	return delayVehicle;
}

void Vehicle::oneStepSimulation(TColor *textureMap, TColor *dst, int id, int imageW, int imageH, int maxPediestran, bool semaphore){

	/*int id;
	TColor *dst;
    int imageW;
    int imageH;
	int maxPediestran;
	bool semaphore;*/

	/*
    int *x; //int **devLocalX; 
	int *y; //int **devLocalY;
	int delay;//int *devLocalStep;
	int size;//int *devMaxLocalStep;

	//estos cuatro no los habia inicializado......solo crearlos y ya.....
	int nextX; //int *devNextX; //para phase 1
	int nextY; //int *devNextY;
	int previousX; //int *devPreviousX;
	int previousY; //int *devPreviousY;

	
	int timeout; //int *devTimeOut; //crear inicializando en -10.....
	float maxSpeed; //float *devSpeed; //crear en 0
	*/

		if(timeCounter==-10) //solo entra la primera vez.
		{
			int cellSize=maxPediestran*maxSpeed;
			int cellNumber=id/cellSize;
			timeCounter=cellNumber;
			return;
		}
		if(timeCounter<0)
		{
			timeCounter=1.f/maxSpeed-1;
		}else{
			timeCounter--;
		}



		if(timeCounter!=0)
			return;
		
		if (delay<0){
			delay++;
			return;
		}

		if (delay==0){
			previousX=x[5*delay + 4];
			previousY=y[5*delay + 4];
			nextX=previousX;
			nextY=previousY;
			delay++;
			//return;
		}
		
		//invierte metas locales OK funciona
		if (delay==size){
			delay=0;
			//return;
		}

		int nextTrasX=x[5*(delay+0) + 4];
		int nextTrasY=y[5*(delay+0) + 4];

		float disX=(float) (nextTrasX-nextX);
		float disY=(float) (nextTrasY-nextY);

		int maximo = absMax(disX,disY);
		if (maximo == 0){
			delay++;
			return;
		}

		float hyp =sqrt(disX*disX+disY*disY);
		int dx=0;
		int dy=0;
		if (absf(disX/hyp)>0.13f){
			dx = disX > 0 ? 1 : -1;
		}
		if (absf(disY/hyp)>0.13f){
			dy = disY > 0 ? 1 : -1;
		}

		int px=nextX;
		int py=nextY;

		getNextStepForVehicle(dst, id, imageW, imageH, dx, dy, px, py);

		if (px != nextX || py != nextY){ //nueva posicion
			
			TColor color=make_color(1.f, 1.f, 1.f, ((float)id)/255.0f);
			drawAllTrace(textureMap, dst, color, imageW, imageH, nextX, nextY, 0, nextY-previousY,
							traceX,	traceY,	traceRotX, traceRotY, sizeX, sizeZ);

			//cambiar esto a area.
			disX=(float) (x[5*(delay) + 4]-px);
			disY=(float) (y[5*(delay) + 4]-py);
			hyp=sqrt(disX*disX+disY*disY);
			//if ( px==x[5*(delay) + 4] && py == y[5*(delay) + 4] ){
			if ( hyp < 2.f ){
				delay++;
				if (delay!=size){
					timeCounter+=timeOut;
				}
			}
			previousX=nextX;
			previousY=nextY;
			nextX=px;
			nextY=py;
		}
}

bool Vehicle::isColissionForVehicle(TColor *dst, int id, int imageW, int imageH, int x, int y, int dx, int dy){

	int nx=x+dx;
	int ny=y+dy;

	if (nx >= imageW || nx < 1)
		return true;
	if (ny >= imageH || ny < 1)
		return true;
		
	TColor color = dst[imageW * ny + nx];
	int r = (color >> 0) & 0xFF;
	int g = (color >> 8) & 0xFF;
	//int b = (color >> 16) & 0xFF;
	int a = (color >> 24) & 0xFF;

	int area= r & 0xE0;
	if ( (area >> 5) == 7) //hay un edificio alli
		return true;
	if ( (area >> 5) == 6) //hay una estacion alli
		return true;
	if ( (area >> 5) == 5) //via peatonal
		return true;
	
	if ( ((area >> 5) != 4) && (area >> 5) != 2) //no es una via de transmilenio/carro
		return true;
	
	if ( a != id && r==1.f) //hay un vehiculo, peaton o transmilenio ocupando el sitio.
		return true;

	bool up = ((g & 0x80) >> 7) == 1;
	bool down = ((g & 0x40) >> 6) == 1;
	bool left = ((g & 0x20) >> 5) == 1;
	bool right = ((g & 0x10) >> 4) == 1;
	
	if ((dy>-1 && up) || (dy<1 && down) || (dx>-1 && right) || (dx<1 && left))
		return false;
	else
		return true;

	return false;
}

void Vehicle::getFirstStepForVehicle(TColor *dst, int id, int imageW, int imageH, int x, int y, int &px, int &py){

	if (isColissionForVehicle(dst,id,imageW,imageH, px, py, x, y) ){// de frente

		if (x==0){ //para direccion arriba-abajo
			//asumiendo direcion hacia arriba
			if ( !isColissionForVehicle(dst,id,imageW,imageH,px,py,y,y) ){ // (+,+) - derecha de frente
				px+=y;
				py+=y;
			}else if (!isColissionForVehicle(dst,id,imageW,imageH,px,py,-y,y)){ // (-,+) - izquierda de frente
				px-=y;
				py+=y;
			}
		}else if (y==0){ //para direccion izquierda-derecha
			//asumiendo direccion hacia la derecha
			if ( !isColissionForVehicle(dst,id,imageW,imageH,px,py,x,-x) ){ // (+,-) - diagonal derecha
				px+=x;
				py-=x;
			}else if (!isColissionForVehicle(dst,id,imageW,imageH,px,py,x,x)){ // (+,+) - diagonal izquierda
				px+=x;
				py+=x;
			}
		}else if (x==y){ //para diagonal so-ne
			// tomando como direccion (1,1) derecha-arriba
			if ( !isColissionForVehicle(dst,id,imageW,imageH,px,py,x,0) ){ // (+,0) - miro diagonal derecha
				px+=x;
			}else if (!isColissionForVehicle(dst,id,imageW,imageH,px,py,0,y)){ // (0,+) - miro diagonal izquierda
				py+=y;
			}
		}else if (x==-y){ //para diagonal se-no
			//asumiendo como direccion (1,-1) derecha-abajo
			if ( !isColissionForVehicle(dst,id,imageW,imageH,px,py,0,y) ){ // (0,-) - miro diagonal derecha (asumo y=-1)
				py+=y;
			}else if (!isColissionForVehicle(dst,id,imageW,imageH,px,py,x,0)){ // (0,+) - miro diagonal izquierda (asumo x=1)
				px+=x;
			}
		}
	}else{
		px+=x;
		py+=y;
	}
}

void Vehicle::frontSidersForVehicle(int id, int rx, int ry, int &dx, int &dy){
	dy=0;
	dx=ry;
}

bool Vehicle::isFrontCollisionForVehicle(TColor *dst, int id, int imageW, int imageH, int px, int py, int x, int y, int dx, int dy, int rightSize, int leftSize){
	
	if (isColissionForVehicle(dst,id,imageW,imageH, px, py, x, y))
		return true;

	for(int n=1; n<rightSize+1; n++){
		if(isColissionForVehicle(dst,id,imageW,imageH, px+x, py+y, n*dx, n*dy))
		;//	return true;
	}
	for(int n=1; n<leftSize+1; n++){
		if(isColissionForVehicle(dst,id,imageW,imageH, px+x, py+y, -n*dx, -n*dy))
		;//return true;
	}

	return false;
}

void Vehicle::getNextStepForVehicle(TColor *dst, int id, int imageW, int imageH, int x, int y, int &px, int &py){

	 // esto debe estar en esta clase.....int *traceX, int *traceY

	if (traceX[0]<0 && traceY[0]<0){
		getFirstStepForVehicle(dst, id, imageW, imageH, x, y, px, py);
		return;
	}

	int size = (sizeX-1)/2;
	int res = (( (float)sizeX - 1.f )/2.f - (float)size) * 2;
	int leftSize=size;
	int rightSize=size+res;

	int dx=0;
	int dy=0;
	frontSidersForVehicle(id, x, y, dx, dy);
	if ( isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, x, y, dx, dy, rightSize, leftSize) ){// de frente

		if (x==0){ //para direccion arriba-abajo		
			//asumiendo direcion hacia arriba
			frontSidersForVehicle(id, y, y, dx, dy);
			if ( !isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, y, y, dx, dy, rightSize, leftSize) ){ // (+,+) - derecha de frente
				px+=y;
				py+=y;
			}else{
				frontSidersForVehicle(id, -y, y, dx, dy);
				if (!isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, -y, y, dx, dy, rightSize, leftSize) ){ // (-,+) - izquierda de frente
					px-=y;
					py+=y;
				}
			}
			
		}else if (y==0){ //para direccion izquierda-derecha
				//asumiendo direccion hacia la derecha
			frontSidersForVehicle(id, x, -x, dx, dy);
			if ( !isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, x, -x, dx, dy, rightSize, leftSize) ){ // (+,-) - diagonal derecha
				px+=x;
				py-=x;
			}else{
				frontSidersForVehicle(id, x, x, dx, dy);
				if (!isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, x, x, dx, dy, rightSize, leftSize) ){ // (+,+) - diagonal izquierda
					px+=x;
					py+=x;
				}
			}
			
		}else if (x==y){ //para diagonal so-ne
			// tomando como direccion (1,1) derecha-arriba
			frontSidersForVehicle(id, x, 0, dx, dy);
			if ( !isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, x, 0, dx, dy, rightSize, leftSize) ){ // (+,0) - miro diagonal derecha
				px+=x;
			}else{
				frontSidersForVehicle(id, 0, y, dx, dy);
				if (!isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, 0, y, dx, dy, rightSize, leftSize) ){ // (0,+) - miro diagonal izquierda
					py+=y;
				}
			}

		}else if (x==-y){ //para diagonal se-no
			//asumiendo como direccion (1,-1) derecha-abajo
			frontSidersForVehicle(id, 0, y, dx, dy);
			if ( !isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, 0, y, dx, dy, rightSize, leftSize) ){ // (0,-) - miro diagonal derecha (asumo y=-1)
				py+=y;
			}else{
				frontSidersForVehicle(id, x, 0, dx, dy);
				if (!isFrontCollisionForVehicle(dst, id, imageW, imageH, px, py, x, 0, dx, dy, rightSize, leftSize)){ // (0,+) - miro diagonal izquierda (asumo x=1)
					px+=x;
				}
			}
		}
	}else{
		px+=x;
		py+=y;
	}
}

