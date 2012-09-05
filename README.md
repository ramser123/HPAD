HPAD
====

Ejemplos del curso de HPAD

- La Solución se llama Simulador2D. Para abrir el proyecto es necesario tener instalado CUDA 4.2 (ver http://developer.nvidia.com/cuda/cuda-downloads). Y copiar los archivos .rules de CUDA (en una instalación por defecto se encuentran en C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\extras\visual_studio_integration\rules) en la instalación de Visual Studio (por defecto en C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\VCProjectDefaults)

- Los valores iniciales de la simulación como número de peatones, vehículos, transmilenios, tiempos de semáforos, FPS deseado, y tipo de Kernel se configuran en el archivo "SIMULATOR.properties", explicación de los parámetros se encuentran en el archivo mismo.

-La simulación por CPU se encuentra en los métodos pedestrianOneStepSimulation(...), vehicleOneStepSimulation(...) y transmilenioOneStepSimulation (...) de MobileManager.cpp

- La simulación por GPU se encuentra en pedestrian_kernel.cu, transmilenio_kernel.cu y vehicle_kernel.cu (ver los métodos run_Pedestrian, run_Transmilenio y run_Vehicle respectivamente), cada paso de simulación se realiza en 4 etapas para GPU con deteccion paralela y 6 etapas para GPU con deteccion serial.

-El manejo de OpenGL y llamado de los kernels se realiza en simuladorGL.cpp