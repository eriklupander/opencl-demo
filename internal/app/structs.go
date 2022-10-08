package app

import (
	"fmt"
	"github.com/jgillich/go-opencl/cl"
	"github.com/sirupsen/logrus"
	"time"
	"unsafe"
)

// __attribute__((packed, aligned(16)))
var printRayStructSrc = `

typedef struct __attribute__((aligned(128))) tag_mystruct{
	double4 origin;       // 32 bytes
	double4 direction;    // 32 bytes
    float4 extra;         // 16 bytes
    //char padding[48];     // 48 bytes
} mystruct;

__kernel void printRayStruct(
   __global mystruct* input1,
   __global double* output)
{
   int i = get_global_id(0);
	printf("Job: %d Origin: %f %f %f %f\n", i, input1[i].origin.x, input1[i].origin.y, input1[i].origin.z, input1[i].origin.w);
    printf("Job: %d Direct: %f %f %f %f\n", i, input1[i].direction.x, input1[i].direction.y, input1[i].direction.z, input1[i].direction.w);
    printf("Job: %d Extra: %f %f %f %f\n", i, input1[i].extra.x, input1[i].extra.y, input1[i].extra.z, input1[i].extra.w);
    
	output[i] = 1.0;
}

`

type MyStruct struct {
	Origin    [4]float64 // 32 bytes
	Direction [4]float64 // 32 bytes
	Extra     [4]float32 // 16 bytes
	Padding   [48]byte   // 48 bytes => Total 128 bytes
}

func Structs(deviceIndex int) {
	wgSize := 256
	// add first arg
	input1 := make([]MyStruct, 0)
	for i := int32(0); i < int32(wgSize); i++ {
		input1 = append(input1, MyStruct{
			Origin:    [4]float64{float64(1 + i), float64(2 * i), float64(3 * i), float64(4 * i)},
			Direction: [4]float64{float64(5 + i), float64(6 * i), float64(7 * i), float64(8 * i)},
			Extra:     [4]float32{999, 888, 777, 666}, // added just to demonstrate alignment
			//Padding:   [48]byte{}, // not needed
		})
	}

	sx := unsafe.Sizeof(input1[0])
	ss := nextPowOf2(int(sx))

	fmt.Printf("size of a MyStruct is: %d bytes\n", ss)
	platforms, err := cl.GetPlatforms()
	if err != nil {
		logrus.Fatalf("Failed to get platforms: %+v", err)
	}
	platform := platforms[0]

	devices, err := platform.GetDevices(cl.DeviceTypeAll)
	if err != nil {
		logrus.Fatalf("Failed to get devices: %+v", err)
	}
	if len(devices) == 0 {
		logrus.Fatalf("GetDevices returned no devices")
	}
	if deviceIndex < 0 {
		deviceIndex = 0
	}
	device := devices[deviceIndex]
	logrus.Infof("Using device %d %v", deviceIndex, device.Name())

	// 1. Select a device to use. On my mac: 0 == CPU, 1 == Iris GPU, 2 == GeForce 750M GPU
	// Use selected device to create an OpenCL context
	context, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		logrus.Fatalf("CreateContext failed: %+v", err)
	}

	// 2. Create a "Command Queue" bound to the selected device
	queue, err := context.CreateCommandQueue(device, 0)
	if err != nil {
		logrus.Fatalf("CreateCommandQueue failed: %+v", err)
	}

	// 3.1 Create an OpenCL "program" from the source code.
	program, err := context.CreateProgramWithSource([]string{printRayStructSrc})
	if err != nil {
		logrus.Fatalf("CreateProgramWithSource failed: %+v", err)
	}

	// 3.2 Build the OpenCL program (compile it?)
	if err := program.BuildProgram(nil, ""); err != nil {
		logrus.Fatalf("BuildProgram failed: %+v", err)
	}

	// 3.3 Create the actual Kernel with a name, the Kernel is what we call when we want to execute something.
	kernel, err := program.CreateKernel("printRayStruct")
	if err != nil {
		logrus.Fatalf("CreateKernel failed: %+v", err)
	}

	// 4. Some kind of error-check where we make sure the parameters passed are supported?
	for i := 0; i < 2; i++ {
		name, err := kernel.ArgName(i)
		if err == cl.ErrUnsupported {
			logrus.Errorf("GetKernelArgInfo for arg: %d ErrUnsupported", i)
			break
		} else if err != nil {
			logrus.Errorf("GetKernelArgInfo for name failed: %+v", err)
			break
		} else {
			logrus.Infof("Kernel arg %d: %s", i, name)
		}
	}

	// 5. Time to start loading data into GPU memory

	// 5.1 create OpenCL buffers (memory) for the input data.
	param1, err := context.CreateEmptyBuffer(cl.MemReadOnly, ss*len(input1))
	if err != nil {
		logrus.Fatalf("CreateBuffer failed for matrices input: %+v", err)
	}

	// 5.2 create OpenCL buffers (memory) for the output data
	output, err := context.CreateEmptyBuffer(cl.MemWriteOnly, len(input1)*8)
	if err != nil {
		logrus.Fatalf("CreateBuffer failed for output: %+v", err)
	}

	// 5.3 This is where we connect our input to the command queue, and upload the actual data into GPU memory
	//     The dataPtr:s seems to be a point to the first element of the input,
	//     while dataSize should be the total length of the data.
	dataPtr := unsafe.Pointer(&input1[0])
	dataSize := ss * len(input1)
	if _, err := queue.EnqueueWriteBuffer(param1, true, 0, dataSize, dataPtr, nil); err != nil {
		logrus.Fatalf("EnqueueWriteBuffer failed: %+v", err)
	}

	// 5.4 Kernel is our program and here we explicitly bind our 3 parameters to it
	if err := kernel.SetArgs(param1, output); err != nil {
		logrus.Fatalf("SetKernelArgs failed: %+v", err)
	}

	// 6. Determine device's WorkGroup size. This is probably how many items the GPU can process at a time.
	local, err := kernel.WorkGroupSize(device)
	if err != nil {
		logrus.Fatalf("WorkGroupSize failed: %+v", err)
	}
	logrus.Infof("Work group size: %d", local)
	size, _ := kernel.PreferredWorkGroupSizeMultiple(nil)
	logrus.Infof("Preferred Work Group Size Multiple: %d", size)

	// 6.1 calc local/global sizes. This stuff is passed on to the "enqueue". I think it's purpose is to handle
	//     cases where the data set size isn't divideable by the WG size
	global := len(input1) // number of items to process, e.g. 32768
	d := global % local   // given the preferred WG size, d is
	logrus.Infof("Global: %d, D: %d", global, d)
	if d != 0 {
		global += local - d
	}
	logrus.Infof("Global after applying D: %d, D: %d", global, d)

	st := time.Now()

	// 7. Finally, start work! Enqueue executes the loaded args on the specified kernel.
	if _, err := queue.EnqueueNDRangeKernel(kernel, nil, []int{len(input1)}, nil, nil); err != nil {
		logrus.Fatalf("EnqueueNDRangeKernel failed: %+v", err)
	}

	// 8. Finish() blocks the main goroutine until the OpenCL queue is empty, i.e. all calculations are done
	if err := queue.Finish(); err != nil {
		logrus.Fatalf("Finish failed: %+v", err)
	}

	logrus.Infof("Took: %v", time.Since(st))

	// 9. Allocate storage for loading the output from the OpenCL program
	results := make([]int32, len(input1)/2)

	// 10. The EnqueueReadBuffer copies the data in the OpenCL "output" buffer into the "results" slice.
	dataPtrOut := unsafe.Pointer(&results[0])
	dataSizeOut := int(unsafe.Sizeof(results[0])) * len(results)
	if _, err := queue.EnqueueReadBuffer(output, true, 0, dataSizeOut, dataPtrOut, nil); err != nil {
		logrus.Fatalf("EnqueueReadBuffer failed: %+v", err)
	}
}
