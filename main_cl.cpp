
#include <iostream>
#include <Cl/cl.h>

std::string kernel_code =
    "__kernel void suma_opencl(__global float *a, __global  float *b, __global float *c, const unsigned int size) {"
    "      int index = get_global_id(0);        "
    "      if (index < size)                    "
    "           c[index] = a[index] + b[index]; "
    "}"
;
int main()
{
//Listamos el número de dispositivos gráficos
    cl_platform_id* platforms= nullptr;
    cl_uint numPlatforms=0;

    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::printf("No. plataformas: %d\n", numPlatforms);

    platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, nullptr);
    for (int i=0; i<numPlatforms; i++)
    {
        char vendor[1024];
        char version[1024];
        char name[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        std::printf("Plataforma %d: %s\n\t %s - %s\n", i, vendor, version, name);

        std::printf("\t*** Dispositivos ***\n");
        cl_uint numDevices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        std::printf("\tNo. : %d\n", numDevices);

        cl_device_id* devices = new cl_device_id[numDevices];

        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, nullptr);
        for(int di=0; di<numDevices; di++)
        {
            char deviceName[1024];
            clGetDeviceInfo(devices[di], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            std::printf("\tId %d - Nombre: %s\n", di, deviceName);
        }
    }

    //-------------- Seleccionamos el dispositivo que vamos a usar (GPU)
    std::printf("\n---------------------------------------------\n");
    cl_platform_id platform_id = platforms[1]; //Es el índice de la plataforma nvidia
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue commands_queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_int error = 0;

    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    char buffer[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
    std::printf("Dispositivo seleccionado: %s\n", buffer);

    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &error);
    commands_queue = clCreateCommandQueue(context, device_id, 0, &error);

    //Creamos el programa
    const char* src_program = kernel_code.c_str();
    program = clCreateProgramWithSource(context, 1, &src_program, nullptr, &error);
    if(program == nullptr)
    {
        std::printf("Error al cargar el recurso\n");
        exit(1);
    }
    //Compilamos el programa
    error = clBuildProgram(program,1, &device_id, nullptr, nullptr, nullptr);
    if(error!=CL_SUCCESS)
    {
        size_t len = 0;
        char buffer[1024];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::printf("Error al compilar el programa: %s\n", buffer);
        exit(1);
    }
    std::string kernel_name = "suma_opencl";
    kernel = clCreateKernel(program, kernel_name.c_str(), &error);
    if(kernel == nullptr)
    {
        std::printf("Error al crear el kernel: %s\n", kernel_name.c_str());
        exit(1);
    }

    ///----------------------------
    //Operaciones hostToDevice y viceversa
    //Allocate memory for the host vectors
    const size_t VECTOR_SIZE =1024*1024*256;
    float* h_A= new float[VECTOR_SIZE];
    float* h_B= new float[VECTOR_SIZE];
    float* h_C = new float[VECTOR_SIZE];

    memset(h_C, 0, VECTOR_SIZE * sizeof(float));
    for(int i=0; i<VECTOR_SIZE; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    //Allocate memory for the device vectors
    size_t size_bytes = VECTOR_SIZE * sizeof(float);
    cl_mem d_A= clCreateBuffer(context, CL_MEM_READ_ONLY, size_bytes, nullptr, &error);
    cl_mem d_B= clCreateBuffer(context, CL_MEM_READ_ONLY, size_bytes, nullptr, &error);
    cl_mem d_C= clCreateBuffer(context, CL_MEM_READ_ONLY, size_bytes, nullptr, &error);


    //Copy the host vectors to the device
    clEnqueueWriteBuffer(commands_queue, d_A, CL_TRUE, 0, size_bytes, h_A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commands_queue, d_B, CL_TRUE, 0, size_bytes, h_B, 0, nullptr, nullptr);

    //Invocamos el kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &VECTOR_SIZE);

    size_t global_work_size = VECTOR_SIZE;
    size_t local_work_size = 256; //256 hilos por bloque
    clEnqueueNDRangeKernel(
        commands_queue,
        kernel
        , 1 //Dimensión del grupo de hilos
        ,0 //Offset
        , &global_work_size
        , &local_work_size
        , 0, nullptr, nullptr);
    clFinish(commands_queue);

    //Copy the device to host
    clEnqueueReadBuffer(commands_queue, d_C, CL_TRUE, 0, size_bytes, h_C, 0, nullptr, nullptr);
    for(int i=0; i<1024; i++)
    {
        std::printf("%.0f", h_C[i]);
    }

    //Liberamos recursos
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands_queue);
    clReleaseContext(context);


    return 0;
}
