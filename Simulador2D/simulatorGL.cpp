/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



/*
 * This sample demonstrates two adaptive image denoising technqiues: 
 * KNN and NLM, based on computation of both geometric and color distance 
 * between texels. While both techniques are already implemented in the 
 * DirectX SDK using shaders, massively speeded up variation 
 * of the latter techique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <conio.h>
#include <map>

#include "MobileManager.h"

#include <vector>
using namespace std;



#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
// #include <GL/glut.h>
#include <GL/freeglut.h>
#endif

#include "simulatorGL.h"
#include <rendercheck_gl.h>

static char *sSDKsample = "TRANSMILENIO simulator2D"; 

//metodo(pedestrian) paso
//recibo metodo(map &nombre)

// Constantes
/*int pedestrianNumber;
int vehicleNumber;
int transmilenioNumber;
int desiredFPS;
bool stepByStep;
bool runsOnGPU;
bool parallelDetection;*/

MobileManager *mobileManager = NULL;

static char *filterMode[] = {
    "Texture",
    "Pedestrian",
    "Vehicle",
    "Transmilenio",
	"Simulation",
	"OneStepSimulation",
    NULL
};

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "image_passthru.ppm",
    "image_knn.ppm",
    "image_nlm.ppm",
    "image_nlm2.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_passthru.ppm",
    "ref_knn.ppm",
    "ref_nlm.ppm",
    "ref_nlm2.ppm",
    NULL
};

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
int imageW, imageH;
TColor *textureMap;
TColor *h_dst;

GLuint shader;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int  g_Kernel = 0;
bool    g_FPS = false;
bool   g_Diag = false;
unsigned int hTimer;

//Algorithms global parameters
const float noiseStep = 0.025f;
const float  lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float    lerpC = 0.2f;


const int frameN = 24;
int frameCounter = 0;


#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_AutoQuit = false;
bool g_Verify = false;
bool g_bQAReadback = false;
bool g_bOpenGLQA   = false;
bool g_bFBODisplay = false;

// CheckFBO/BackBuffer class objects
CFrameBufferObject  *g_FrameBufferObject = NULL;
CheckRender         *g_CheckRender       = NULL;

#define MAX_EPSILON_ERROR 5

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        sprintf(temp, "%s<%s>", "[AutoTest]:", filterMode[g_Kernel]);  
	    glutSetWindowTitle(temp);

        g_Kernel++;

        if (g_Kernel > 3) {
            g_Kernel = 0;
            printf("Summary: %d errors!\n", g_TotalErrors);
            printf("\tTEST: %s\n", (g_TotalErrors==0) ? "OK" : "FAILURE");
            exit(0);
        }
    }
}


void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit-1) {
        g_Verify = true;
    }
    if (fpsCount == fpsLimit) {
        char fps[256];
   		float ifps = 1.f / (cutGetAverageTimerValue(hTimer) / 1000.f);
		sprintf(fps, "%s <%s: %s>: %0.0f fps", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "[AutoTest]:" : ""),
				mobileManager->runsOnGPU ? "GPU" : "CPU", filterMode[g_Kernel], ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        cutilCheckError(cutResetTimer(hTimer));  
        AutoQATest();
    }
}



//aqui está el procesamiento en CUDA
void runImageFilters(TColor *d_dst)
{
	__int64 initTime=currentTimeMillis();
    switch(g_Kernel){
        case 0:
            runImage(d_dst);
        break;
        case 1:
			runPedestrian(d_dst);
        break;
        case 2:
			runVehicle(d_dst);
        break;
        case 3:
            runTransmilenio(d_dst);
        break;
		case 4:
			runPedestrian(d_dst);
			runVehicle(d_dst);
			runTransmilenio(d_dst);
        break;
    }
	
	if(!mobileManager->runsOnGPU)
		cudaMemcpy (d_dst, h_dst, imageW*imageH*sizeof(TColor), cudaMemcpyHostToDevice);
	
	if (g_Kernel!=0 && g_Kernel!=5)
		mobileManager->updateSemaphore();

	if (mobileManager->stepByStep)
		g_Kernel=5;

	__int64 finalTime=currentTimeMillis();
	
	int sleepTime = (int) (((float)1/(float)(mobileManager->desiredFPS+16))*1000.f - (int)(finalTime-initTime));
	sleepTime = sleepTime < 0 ? 0 : sleepTime;
	if (mobileManager->desiredFPS!=0)
		Sleep(sleepTime);

    cutilCheckMsg("Filtering kernel execution failed.\n");
}
void runImage(TColor *d_dst){
	copy_Image(d_dst, imageW, imageH);
	if (!mobileManager->runsOnGPU){
		cudaMemcpy (textureMap, d_dst, imageW*imageH*sizeof(TColor), cudaMemcpyDeviceToHost);
		cudaMemcpy (h_dst, d_dst, imageW*imageH*sizeof(TColor), cudaMemcpyDeviceToHost);
	}
}

void runPedestrian(TColor *d_dst){
	if (mobileManager->runsOnGPU){
		run_Pedestrian(d_dst,
				mobileManager->devPediestranClass,
				imageW,
				imageH,
				mobileManager->pedestrianNumber,
				mobileManager->parallelDetection,
				mobileManager->getSemaphoreState(),
				mobileManager->devPediestranX,
				mobileManager->devPediestranY,
				mobileManager->devPediestranStep,
				mobileManager->devPediestranMaxStep,
				mobileManager->devPediestranCurrentX,
				mobileManager->devPediestranCurrentY,
				mobileManager->devPediestranPreviousX,
				mobileManager->devPediestranPreviousY,
				mobileManager->devPediestranConflicted,
				mobileManager->devPediestranRelated,
				mobileManager->devPediestranTimeOut,
				mobileManager->devPediestranSpeed,
				mobileManager->devPediestran_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
	}else{
		mobileManager->pedestrianOneStepSimulation(textureMap,
			h_dst, 
			imageW, 
			imageH, 
			mobileManager->pedestrianNumber, 
			mobileManager->getSemaphoreState());
	}
}
void runVehicle(TColor *d_dst){
	if (mobileManager->runsOnGPU){
		run_Vehicle(d_dst,
				mobileManager->devVehicleClass,
				imageW,
				imageH,
				mobileManager->vehicleNumber,
				mobileManager->parallelDetection,
				mobileManager->getSemaphoreState(),
				mobileManager->devVehicleX,
				mobileManager->devVehicleY,
				mobileManager->devVehicleStep,
				mobileManager->devVehicleMaxStep,
				mobileManager->devVehicleCurrentX,
				mobileManager->devVehicleCurrentY,
				mobileManager->devVehiclePreviousX,
				mobileManager->devVehiclePreviousY,
				mobileManager->devVehicleTraceX,
				mobileManager->devVehicleTraceY,
				mobileManager->devVehicleTraceRotX,
				mobileManager->devVehicleTraceRotY,
				mobileManager->devVehicleConflicted,
				mobileManager->devVehicleRelated,
				mobileManager->devVehicleTimeOut,
				mobileManager->devVehicleSpeed,
				mobileManager->devVehicle_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
	}else{
		mobileManager->vehicleOneStepSimulation(textureMap,
			h_dst, 
			imageW, 
			imageH, 
			mobileManager->vehicleNumber, 
			mobileManager->getSemaphoreState());
	}
}
void runTransmilenio(TColor *d_dst){
	if (mobileManager->runsOnGPU){
		run_Transmilenio(d_dst,
				mobileManager->devTransmilenioClass,
				imageW,
				imageH,
				mobileManager->transmilenioNumber,
				mobileManager->parallelDetection,
				mobileManager->getSemaphoreState(),
				mobileManager->devTransmilenioX,
				mobileManager->devTransmilenioY,
				mobileManager->devTransmilenioStep,
				mobileManager->devTransmilenioMaxStep,
				mobileManager->devTransmilenioCurrentX,
				mobileManager->devTransmilenioCurrentY,
				mobileManager->devTransmilenioPreviousX,
				mobileManager->devTransmilenioPreviousY,
				mobileManager->devTransmilenioTraceX,
				mobileManager->devTransmilenioTraceY,
				mobileManager->devTransmilenioTraceRotX,
				mobileManager->devTransmilenioTraceRotY,
				mobileManager->devTransmilenioConflicted,
				mobileManager->devTransmilenioRelated,
				mobileManager->devTransmilenioTimeOut,
				mobileManager->devTransmilenioSpeed,
				mobileManager->devTransmilenio_sizeX_sizeZ_maxSpeed_maxAcce_maxTimeout);
	}else{
		mobileManager->transmilenioOneStepSimulation(textureMap,
			h_dst, 
			imageW, 
			imageH, 
			mobileManager->transmilenioNumber, 
			mobileManager->getSemaphoreState());
	}
}



// funcion de visualizacion, tener presente para inicializacion de variables.....
void displayFunc(void){
	cutStartTimer(hTimer);
    TColor *d_dst = NULL;
	size_t num_bytes;

    if(frameCounter++ == 0) cutResetTimer(hTimer);
    // DEPRECATED: cutilSafeCall(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));
    cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	cutilCheckMsg("cudaGraphicsMapResources failed");
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource));
	cutilCheckMsg("cudaGraphicsResourceGetMappedPointer failed");

    cutilSafeCall( CUDA_Bind2TextureArray()                      );

    runImageFilters(d_dst);

    cutilSafeCall( CUDA_UnbindTexture()     );
    // DEPRECATED: cutilSafeCall(cudaGLUnmapBufferObject(gl_PBO));
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	if (g_bFBODisplay) {
		g_FrameBufferObject->bindRenderPath();
	}

    // Common display code path
	{
        glClear(GL_COLOR_BUFFER_BIT);

        glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0) );
        glBegin(GL_TRIANGLES);
            glTexCoord2f(0, 0); glVertex2f(-1, -1);
            glTexCoord2f(2, 0); glVertex2f(+3, -1);
            glTexCoord2f(0, 2); glVertex2f(-1, +3);
        glEnd();
        glFinish();
    }

	if (g_bFBODisplay) {
		g_FrameBufferObject->unbindRenderPath();
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        printf("> (Frame %d) readback BackBuffer\n", frameCount);
        if (g_bFBODisplay) {
            g_CheckRender->readback( imageW, imageH, g_FrameBufferObject->getFbo() );
        } else {
            g_CheckRender->readback( imageW, imageH );
        }
        g_CheckRender->savePPM ( sOriginal[g_Kernel], true, NULL );
        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Kernel], sReference[g_Kernel], MAX_EPSILON_ERROR, 0.15f)) {
            g_TotalErrors++;
        }
        g_Verify = false;
    }

    if(frameCounter == frameN){
        frameCounter = 0;
        if(g_FPS){
            printf("FPS: %3.1f\n", frameN / (cutGetTimerValue(hTimer) * 0.001) );
            g_FPS = false;
        }
    }

	glutSwapBuffers();

	cutStopTimer(hTimer);
	computeFPS();

	glutPostRedisplay();
}



void shutDown(unsigned char k, int /*x*/, int /*y*/)
{
    switch (k){
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");

			if (mobileManager)
				delete mobileManager;

            cutilCheckError( cutStopTimer(hTimer)   );
            cutilCheckError( cutDeleteTimer(hTimer) );
			// DEPRECATED: cutilSafeCall( cudaGLRegisterBufferObject(gl_PBO) );
			cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, 
									   cudaGraphicsMapFlagsWriteDiscard));
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
            glDeleteBuffers(1, &gl_PBO);
            glDeleteTextures(1, &gl_Tex);

            cutilSafeCall( CUDA_FreeArray() );
            free(h_Src);
            printf("Shutdown done.\n");
            cudaThreadExit();
            exit(0);
        break;

        case '1':
            printf("Original Texture.\n");
            g_Kernel = 0;
        break;

        case '2':
            //printf("Pedestrian simulation \n");
            g_Kernel = 1;
        break;

        case '3':
            //printf("Vehicle simulation \n");
            g_Kernel = 2;
        break;

        case '4':
            //printf("Transmilenio simulation \n");
            g_Kernel = 3;
        break;

		case '5':
            printf("ALL simulation \n");
            g_Kernel = 4;
        break;

        case ' ':
            
            g_Diag = !g_Diag;
        break;

        case 'n':
            
            knnNoise -= noiseStep;
            nlmNoise -= noiseStep;
        break;

        case 'N':
            
            knnNoise += noiseStep;
            nlmNoise += noiseStep;
        break;

        case 'l':
            
            lerpC = MAX(lerpC - lerpStep, 0.0f);
        break;

        case 'L':
            
            lerpC = MIN(lerpC + lerpStep, 1.0f);
        break;

        case 'f' : case 'F':
            g_FPS = true;
        break;

        case '?':
            
        break;
    }
}


int initGL( int *argc, char **argv )
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
    glutCreateWindow(argv[0]);
    printf("OpenGL window created.\n");

    glewInit();
    printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
	if (g_bFBODisplay) {
        if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
            fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
            fprintf(stderr, "This sample requires:\n");
            fprintf(stderr, "  OpenGL version 2.0\n");
            fprintf(stderr, "  GL_ARB_fragment_program\n");
            fprintf(stderr, "  GL_EXT_framebuffer_object\n");
            exit(-1);
        }
	} else {
		if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
			fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
			fprintf(stderr, "This sample requires:\n");
			fprintf(stderr, "  OpenGL version 1.5\n");
			fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
			fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
            fflush(stderr);
            return CUTFalse;
		}
	}

    return 0;
}

// shader for displaying floating-point texture
static const char *shader_code = 
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if (error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}

void initOpenGLBuffers() //aqui se carga la textura a OpenGL
{
    printf("Creating GL texture...\n");
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &gl_Tex);
        glBindTexture(GL_TEXTURE_2D, gl_Tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
        glGenBuffers(1, &gl_PBO);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src, GL_STREAM_COPY);
        //While a PBO is registered to CUDA, it can't be used 
        //as the destination for OpenGL drawing calls.
        //But in our particular case OpenGL is only used 
        //to display the content of the PBO, specified by CUDA kernels,
        //so we need to register/unregister it only once.
	// DEPRECATED: cutilSafeCall( cudaGLRegisterBufferObject(gl_PBO) );
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, 
					       cudaGraphicsMapFlagsWriteDiscard));
        CUT_CHECK_ERROR_GL();
    printf("PBO created.\n");

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

    if (g_FrameBufferObject) {
		delete g_FrameBufferObject; g_FrameBufferObject = NULL;
	}
	if (g_bFBODisplay) {
		g_FrameBufferObject = new CFrameBufferObject(imageW, imageH, 32, false, GL_TEXTURE_2D);
	}
}


void cleanup()
{
    cutilCheckError( cutDeleteTimer( hTimer));

    glDeleteProgramsARB(1, &shader);

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
    if (g_FrameBufferObject) {
        delete g_FrameBufferObject; g_FrameBufferObject = NULL;
    }
}

void runAutoTest(int argc, char **argv)
{
	int devID = 0;
    printf("[%s] - (automated testing w/ readback)\n", sSDKsample);

	devID = cutilChooseCudaDevice(argc, argv);

    // First load the image, so we know what the size of the image (imageW and imageH)
    printf("Allocating host and CUDA memory and loading image file...\n");
        const char *image_path = cutFindFilePath("portrait_noise.bmp", argv[0]);
        LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
    printf("Data init done.\n");

    cutilSafeCall( CUDA_MallocArray(&h_Src, imageW, imageH) );

    g_CheckRender       = new CheckBackBuffer(imageW, imageH, sizeof(TColor), false);
    g_CheckRender->setExecPath(argv[0]);

    TColor *d_dst = NULL;
    cutilSafeCall( cudaMalloc( (void **)&d_dst, imageW*imageH*sizeof(TColor)) );

    while (g_Kernel <= 3) {
        
        printf("[AutoTest]:%s <%s>\n", sSDKsample, filterMode[g_Kernel]);

        cutilSafeCall( CUDA_Bind2TextureArray()                      );

        runImageFilters(d_dst);

        cutilSafeCall( CUDA_UnbindTexture()     );

        cutilSafeCall( cudaThreadSynchronize() );

        cudaMemcpy(g_CheckRender->imageData(), d_dst, imageW*imageH*sizeof(TColor), cudaMemcpyDeviceToHost); //aqui se copia de GPU a CPU....

        g_CheckRender->savePPM(sOriginal[g_Kernel], true, NULL);

        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Kernel], sReference[g_Kernel], MAX_EPSILON_ERROR, 0.15f)) {
            g_TotalErrors++;
        }
        g_Kernel++;
    }

    cutilSafeCall( CUDA_FreeArray() );
    free(h_Src);

    cutilSafeCall( cudaFree( d_dst ) );
    delete g_CheckRender;

	printf("\n[%s] -> Test Results: %d errors\n", sSDKsample, g_TotalErrors);

    if (!g_TotalErrors) 
        printf("PASSED\n");
    else 
        printf("FAILED\n");
}

__int64 currentTimeMillis(void)
{
static const __int64 magic = 116444736000000000; // 1970/1/1
SYSTEMTIME st;
GetSystemTime(&st);
FILETIME ft;
SystemTimeToFileTime(&st,&ft); // in 100-nanosecs...
__int64 t;
memcpy(&t,&ft,sizeof t);
return (t - magic)/10000; // scale to millis.
}


int main(int argc, char **argv)
{

    if (argc > 1) {
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest") ||
            cutCheckCmdLineFlag(argc, (const char **)argv, "noprompt")) 
		{
            g_bQAReadback = true;
            fpsLimit = frameCheckNumber;
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "glverify"))
		{
            g_bOpenGLQA = true;
            g_bFBODisplay = false;
            fpsLimit = frameCheckNumber;
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "fbo")) {
            g_bFBODisplay = true;
            fpsLimit = frameCheckNumber;
        }
    }

    if (g_bQAReadback) {
        runAutoTest(argc, argv);
        cutilExit(argc, argv);
        exit(0);
    } else { //esto al parecer no se cumple nunca
        printf("[%s] ", sSDKsample);
        if (g_bFBODisplay) printf("[FBO Display] ");
        if (g_bOpenGLQA)   printf("[OpenGL Readback Comparisons] ");
        printf("\n");

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device")) {
			printf("[%s]\n", argv[0]);
			printf("   Does not explicitly support -device=n in OpenGL mode\n");
			printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
			printf(" > %s -device=n -qatest\n", argv[0]);
			printf("exiting...\n");
            exit(0);
		}

	    // First load the image, so we know what the size of the image (imageW and imageH)
        printf("Allocating host and CUDA memory and loading image file...\n");
            const char *image_path = cutFindFilePath("portrait_noise.bmp", argv[0]);
            LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
        printf("Data init done.\n");
		
		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		initGL( &argc, argv );
	    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

        cutilSafeCall( CUDA_MallocArray(&h_Src, imageW, imageH) );

        initOpenGLBuffers();

        // Creating the Auto-Validation Code
        if (g_bOpenGLQA) {
            if (g_bFBODisplay) {
                g_CheckRender = new CheckFBO(imageW, imageH, 4);
            } else {
                g_CheckRender = new CheckBackBuffer(imageW, imageH, 4);
            }
            g_CheckRender->setPixelFormat(GL_RGBA);
            g_CheckRender->setExecPath(argv[0]);
            g_CheckRender->EnableQAReadback(g_bOpenGLQA);
        }
    }

	textureMap = new TColor[imageW*imageH];
	h_dst = new TColor[imageW*imageH];

	mobileManager = new MobileManager();
	mobileManager->loadFlags();
	mobileManager->initPedestrian();
	mobileManager->initVehicle();
	mobileManager->initTransmilenio();

    printf("Starting GLUT main loop...\n");
    printf("Press [1] to view texture image\n");
    printf("Press [2] to view one step simulation\n");
    printf("Press [q] to exit\n");

    glutIdleFunc(displayFunc);
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(shutDown);
    cutilCheckError( cutCreateTimer(&hTimer) );
    cutilCheckError( cutStartTimer(hTimer)   );
    glutMainLoop();

    cutilExit(argc, argv);

    cudaThreadExit();
}
