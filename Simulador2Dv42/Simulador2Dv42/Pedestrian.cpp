#include "Pedestrian.h"

char *propertyPedestrianFile = "./prop/USER_GOAL.properties";

int *excludePedestrian=NULL;
const int excludePedestrianSize=7;
const int excludePedestrianIds[excludePedestrianSize] = {32,33,34,38,39,40,87};

int *startingPedestrian=NULL;
const int startingPedestrianSize=1;
const int startingPedestrianIds[startingPedestrianSize] = {-1};

int *finishPedestrian=NULL;
const int finishPedestrianSize=1;
const int finishPedestrianIds[finishPedestrianSize] = {-1};

static int *delayPedestrian=NULL;

Pedestrian::Pedestrian() 
{
	sizeX=1.f;
	sizeY=2.f;
	sizeZ=1.f;
	//maxSpeed=0.004f;
	maxSpeed=0.1f;
	maxAcceleration=0.f;
	timeOut=-10;
	nextX=0;
	nextY=0;
	previousX=0;
	previousY=0;
}
Pedestrian::~Pedestrian()
{
	if (excludePedestrian)
		delete[] excludePedestrian;
	if (startingPedestrian)
		delete[] startingPedestrian;
	if (finishPedestrian)
		delete[] finishPedestrian;
	if (delayPedestrian)
		delete delayPedestrian;
}

char *Pedestrian::getPropertyFileName(void){
	return propertyPedestrianFile;
}

int Pedestrian::getExcludeSize(void){
	return excludePedestrianSize;
}
int *Pedestrian::getExcludePoint(void){
	if(!excludePedestrian){
		excludePedestrian = new int[excludePedestrianSize];
		for (int n=0; n<excludePedestrianSize; n++)
			excludePedestrian[n]=excludePedestrianIds[n];

	}
	return excludePedestrian;
}


int Pedestrian::getStartingSize(void){
	return startingPedestrianSize;
}
int *Pedestrian::getStartingPoint(void){
	if(!startingPedestrian){
		startingPedestrian = new int[startingPedestrianSize];
		for (int n=0; n<startingPedestrianSize; n++)
			startingPedestrian[n]=startingPedestrianIds[n];

	}
	return startingPedestrian;
}


int Pedestrian::getFinishSize(void){
	return finishPedestrianSize;
}
int *Pedestrian::getFinishPoint(void){
	if(!finishPedestrian){
		finishPedestrian = new int[finishPedestrianSize];
		for (int n=0; n<finishPedestrianSize; n++)
			finishPedestrian[n]=finishPedestrianIds[n];

	}
	return finishPedestrian;
}

int *Pedestrian::getDelay(void){
	if(!delayPedestrian){
		delayPedestrian = new int(0);
		return delayPedestrian;
	}
	*delayPedestrian-=1;
	return delayPedestrian;
}

void Pedestrian::oneStepSimulation(TColor *textureMap, TColor *dst, int id, int imageW, int imageH, int maxPediestran, bool semaphore){

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

		if(timeOut==-10) //solo entra la primera vez.
		{
			int cellSize=maxPediestran*maxSpeed;
			int cellNumber=id/cellSize;
			timeOut=cellNumber;
			return;
		}
		if(timeOut<0)
		{
			timeOut=1.f/maxSpeed-1;
		}else{
			timeOut--;
		}



		if(timeOut!=0)
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
			return;
		}
		
		//invierte metas locales OK funciona
		if (delay==size){
			//return;
			for (int i = 0; i < size/2; i++){
				for (int n = 0; n < 5; n++){

					int tempX=x[5*(size-i-1)+n];
					x[5*(size-i-1)+n]=x[5*i+n];
					x[5*i+n]=tempX;

					int tempY=y[5*(size-i-1)+n];
					y[5*(size-i-1)+n]=y[5*i+n];
					y[5*i+n]=tempY;
				}
			}
			delay=0;
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
		if (absf(disX/hyp)>0.2f){
			dx = disX > 0 ? 1 : -1;
		}
		if (absf(disY/hyp)>0.2f){
			dy = disY > 0 ? 1 : -1;
		}

		int px=nextX;
		int py=nextY;

		if (isColission(dst,imageW,imageH, px+dx, py+dy) ){// de frente

			if (dx==0){ //para direccion arriba-abajo

				//asumiendo direcion hacia arriba
				if ( !isColission(dst,imageW,imageH,px+dy,py+dy) ){ // (+,+) - derecha de frente
					px+=dy;
					py+=dy;
				}else if (!isColission(dst,imageW,imageH,px-dy,py+dy)){ // (-,+) - izquierda de frente
					px-=dy;
					py+=dy;
				}else if (!isColission(dst,imageW,imageH,px+dy,py)){ // (-,0) - derecha
					px+=dy;
				}else if (!isColission(dst,imageW,imageH,px-dy,py)){ // (-,0) - izquierda
					px-=dy;
				}else if (!isColission(dst,imageW,imageH,px+dy,py-dy)){ // (+,-) - diagonal atras derecha
					px+=dy;
					py-=dy;
				}else if (!isColission(dst,imageW,imageH,px-dy,py-dy)){ // (-,-) - diagonal atras izquierda
					px-=dy;
					py-=dy;
				}else if (!isColission(dst,imageW,imageH,px,py-dy)){ // (0,-)atras
					py-=dy;
				}
			}else if (dy==0){ //para direccion izquierda-derecha

				//asumiendo direccion hacia la derecha
				if ( !isColission(dst,imageW,imageH,px+dx,py-dx) ){ // (+,-) - diagonal derecha
					px+=dx;
					py-=dx;
				}else if (!isColission(dst,imageW,imageH,px+dx,py+dx)){ // (+,+) - diagonal izquierda
					px+=dx;
					py+=dx;
				}else if (!isColission(dst,imageW,imageH,px,py-dx)){ // (0,-) - derecha
					py-=dx;
				}else if (!isColission(dst,imageW,imageH,px,py+dx)){ // (0,+) - izquierda
					py+=dx;
				}else if (!isColission(dst,imageW,imageH,px-dx,py-dx)){ // (-,-) - diagonal atras derecha
					px-=dx;
					py-=dx;
				}else if (!isColission(dst,imageW,imageH,px-dx,py+dx)){ // (-,+) - diagonal atras izquierda
					px-=dx;
					py+=dx;
				}else if (!isColission(dst,imageW,imageH,px-dx,py)){ // (-,0) - atras
					px-=dx;
				}
			}else if (dx==dy){ //para diagonal so-ne
				// tomando como direccion (1,1) derecha-arriba
				if ( !isColission(dst,imageW,imageH,px+dx,py) ){ // (+,0) - miro diagonal derecha
					px+=dx;
				}else if (!isColission(dst,imageW,imageH,px,py+dy)){ // (0,+) - miro diagonal izquierda
					py+=dy;
				}else if (!isColission(dst,imageW,imageH,px+dx,py-dy)){ // (+,-) - derecha
					px+=dx;
					py-=dy;
				}else if (!isColission(dst,imageW,imageH,px-dx,py+dy)){ // (-,+) - izquierda
					px-=dx;
					py+=dy;
				}else if (!isColission(dst,imageW,imageH,px,py-dy)){ // (0,-) - diagonal atras derecha
					py-=dy;
				}else if (!isColission(dst,imageW,imageH,px-dx,py)){ // (-,0) - diagonal atras izquierda
					px-=dx;
				}else if (!isColission(dst,imageW,imageH,px-dx,py-dy)){ // (-,-) - atras
					px-=dx;
					py-=dy;
				}
			}else if (dx==-dy){ //para diagonal se-no
				//asumiendo como direccion (1,-1) derecha-abajo
				if ( !isColission(dst,imageW,imageH,px,py+dy) ){ // (0,-) - miro diagonal derecha (asumo y=-1)
					py+=dy;
				}else if (!isColission(dst,imageW,imageH,px+dx,py)){ // (0,+) - miro diagonal izquierda (asumo x=1)
					px+=dx;
				}else if (!isColission(dst,imageW,imageH,px-dx,py+dy)){ // (-,-) - derecha
					px-=dx;
					py+=dy;
				}else if (!isColission(dst,imageW,imageH,px+dx,py-dy)){ // (+,+) - izquierda
					px+=dx;
					py-=dy;
				}else if (!isColission(dst,imageW,imageH,px-dx,py)){ // (-,0) - diagonal atras derecha
					px-=dx;
				}else if (!isColission(dst,imageW,imageH,px,py-dy)){ // (0,+) - diagonal atras izquierda
					py-=dy;
				}else if (!isColission(dst,imageW,imageH,px-dx,py-dy)){ // (-,+) - atras
					px-=dx;
					py-=dy;
				}
			}
		}else{
			px+=dx;
			py+=dy;
		}

		if (px != nextX || py != nextY){ //nueva posicion
			//float4 curFresult = tex2D(texImage, (float)nextX + 0.5f, (float)nextY + 0.5f);
			//dst[imageW * nextY + nextX] = make_color(curFresult.x, curFresult.y, curFresult.z, 0.0f);
			dst[imageW * py + px] = make_color(0.f, 1.f, 1.f, 1.f); //prueba
			//float4 preFresult = tex2D(texImage, (float)previousX + 0.5f, (float)previousY + 0.5f);
			dst[imageW * previousY + previousX] = textureMap[imageW * previousY + previousX];//make_color(preFresult.x, preFresult.y, preFresult.z, 0.0f);

			//cambiar esto a area.
			disX=(float) (x[5*(delay) + 4]-px);
			disY=(float) (y[5*(delay) + 4]-py);
			hyp=sqrt(disX*disX+disY*disY);
			//if ( px==x[5*(delay) + 4] && py == y[5*(delay) + 4] ){
			if ( hyp < 2.f ){
				delay++;
			}
			previousX=nextX;
			previousY=nextY;
			nextX=px;
			nextY=py;
		}
}

bool Pedestrian::isColission(TColor *dst, int imageW, int imageH, int x, int y){

	if (x >= imageW || x < 1)
		return true;
	if (y >= imageH || y < 1)
		return true;
		
	TColor color = dst[imageW * y + x];
	int r = (color >> 0) & 0xFF;
	int g = (color >> 8) & 0xFF;
	int b = (color >> 16) & 0xFF;
	int a = (color >> 24) & 0xFF;

	if ( a == 1.f) //hay un vehiculo, peaton o transmilenio ocupando el sitio.
		return true;

	int area= r & 0xE0;
	if ( (area >> 5) == 7) //hay un edificio alli
		return true;
	if ( (area >> 6) == 3) //hay una estacion alli
		return true;
	
	return false;
}