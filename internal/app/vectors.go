package app

import (
	"github.com/jgillich/go-opencl/cl"
	"github.com/sirupsen/logrus"
	"strings"
	"unsafe"
)

var vectorsSrc = `
__kernel void multiply2(
   __global double* input1,
   __global double* input2,
   __global double* output,
   const unsigned int count)
{
   int i = get_global_id(0);

   if(i < count) {
		unsigned int vOffset = i * 4;
        unsigned int mOffset = i * 16;
		for (unsigned int row = 0; row < 4; row++) {
			double a = input2[mOffset + (row*4)+0] * input1[vOffset];
			double b = input2[mOffset + (row*4)+1] * input1[vOffset + 1];
			double c = input2[mOffset + (row*4)+2] * input1[vOffset + 2];
			double d = input2[mOffset + (row*4)+3] * input1[vOffset + 3];
			output[vOffset + row] = a + b + c + d;
		}
   }
}
`

func Vectors(deviceIndex int) {
	// add 1024 vectors
	vectors1 := make([]float64, 0)
	for i := 0; i < 64; i++ {
		for j := 0; j < 4; j++ {
			vectors1 = append(vectors1, float64(i+j))
		}
	}

	// add 1024 matrices
	matrices := make([]float64, 0)
	for i := 0; i < 64; i++ {
		for j := 0; j < 16; j++ {
			matrices = append(matrices, float64(j))
		}
	}

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

	// Check for double precision support
	if !strings.Contains(device.Extensions(), "cl_khr_fp64") {
		logrus.Errorf("device does not support double-precision floating point: extensions supported: %s", device.Extensions())
		return
	}

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
	program, err := context.CreateProgramWithSource([]string{vectorsSrc})
	if err != nil {
		logrus.Fatalf("CreateProgramWithSource failed: %+v", err)
	}

	// 3.2 Build the OpenCL program (compile it?)
	if err := program.BuildProgram(nil, ""); err != nil {
		logrus.Fatalf("BuildProgram failed: %+v", err)
	}

	// 3.3 Create the actual Kernel with a name, the Kernel is what we call when we want to execute something.
	kernel, err := program.CreateKernel("multiply2")
	if err != nil {
		logrus.Fatalf("CreateKernel failed: %+v", err)
	}

	// 4. Some kind of error-check where we make sure the parameters passed are supported?
	for i := 0; i < 4; i++ {
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

	// 5.1 create OpenCL buffers (memory) for the input data. Note that we're allocating 9x bytes the size of data.
	//     since each float64 uses 8 bytes.
	inputVector1, err := context.CreateEmptyBuffer(cl.MemReadOnly, 8*len(vectors1))
	if err != nil {
		logrus.Fatalf("CreateBuffer failed for matrices input: %+v", err)
	}
	inputMatrix2, err := context.CreateEmptyBuffer(cl.MemReadOnly, 8*len(matrices))
	if err != nil {
		logrus.Fatalf("CreateBuffer failed for vectors input: %+v", err)
	}

	// 5.2 create OpenCL buffers (memory) for the output data
	output, err := context.CreateEmptyBuffer(cl.MemReadOnly, 8*len(vectors1))
	if err != nil {
		logrus.Fatalf("CreateBuffer failed for output: %+v", err)
	}

	// 5.3 This is where we connect our input to the command queue, and upload the actual data into GPU memory
	//     The dataPtr:s seems to be a point to the first element of the input,
	//     while dataSize should be the total length of the data.
	dataPtr := unsafe.Pointer(&vectors1[0])
	dataSize := int(unsafe.Sizeof(vectors1[0])) * len(vectors1)
	if _, err := queue.EnqueueWriteBuffer(inputVector1, true, 0, dataSize, dataPtr, nil); err != nil {
		logrus.Fatalf("EnqueueWriteBufferFloat32 failed: %+v", err)
	}
	dataPtrVec2 := unsafe.Pointer(&matrices[0])
	dataSizeVec2 := int(unsafe.Sizeof(matrices[0])) * len(matrices)
	if _, err := queue.EnqueueWriteBuffer(inputMatrix2, true, 0, dataSizeVec2, dataPtrVec2, nil); err != nil {
		logrus.Fatalf("EnqueueWriteBufferFloat32 matrices failed: %+v", err)
	}

	// 5.4 Kernel is our program and here we explicitly bind our 3 parameters to it
	if err := kernel.SetArgs(inputVector1, inputMatrix2, output, uint32(len(vectors1))); err != nil {
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
	global := len(vectors1) // number of items to process, e.g. 32768
	d := global % local     // given the preferred WG size, d is
	logrus.Infof("Global: %d, D: %d", global, d)
	if d != 0 {
		global += local - d
	}
	logrus.Infof("Global after applying D: %d, D: %d", global, d)

	// 7. Finally, start work! Enqueue executes the loaded args on the specified kernel.
	if _, err := queue.EnqueueNDRangeKernel(kernel, nil, []int{len(vectors1) / 4}, []int{4}, nil); err != nil {
		logrus.Fatalf("EnqueueNDRangeKernel failed: %+v", err)
	}

	// 8. Finish() blocks the main goroutine until the OpenCL queue is empty, i.e. all calculations are done
	if err := queue.Finish(); err != nil {
		logrus.Fatalf("Finish failed: %+v", err)
	}

	// 9. Allocate storage for loading the output from the OpenCL program
	results := make([]float64, len(vectors1))

	// 10. The EnqueueReadBuffer copies the data in the OpenCL "output" buffer into the "results" slice.
	dataPtrOut := unsafe.Pointer(&results[0])
	dataSizeOut := int(unsafe.Sizeof(results[0])) * len(results)
	if _, err := queue.EnqueueReadBuffer(output, true, 0, dataSizeOut, dataPtrOut, nil); err != nil {
		logrus.Fatalf("EnqueueReadBuffer failed: %+v", err)
	}

	logrus.Infof("%v", vectors1)
	logrus.Infof("%v", matrices)
	logrus.Infof("%v", results)
}
