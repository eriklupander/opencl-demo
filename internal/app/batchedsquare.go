package app

import (
	"fmt"
	"github.com/jgillich/go-opencl/cl"
	"time"
	"unsafe"
)

var batchedSquareSrc = `__kernel void square(
   __global int* input,
   __global int* output)
{
   int i = get_global_id(0);
   int localSize = get_local_size(0);
   int local_id = get_local_id(0);
   for (int n = 0;n < localSize;n++) {
       	int localIndex = i * localSize+n;
       	output[localIndex] = input[localIndex] * input[localIndex];
   }
}`

func BatchedSquare(deviceIndex int) {
	// First, get hold of a Platform
	platforms, _ := cl.GetPlatforms()

	// Next, get all devices from the first platform. Check so there's at least one device
	devices, _ := platforms[0].GetDevices(cl.DeviceTypeAll)
	if len(devices) == 0 {
		panic("GetDevices returned no devices")
	}
	for i := range devices {
		fmt.Printf("Device %d - %s: max work group size: %d\n", i, devices[i].Name(), devices[i].MaxWorkGroupSize())
	}

	// Select a device to use. On my mac: 0 == CPU, 1 == Iris GPU, 2 == GeForce 750M GPU
	// Use selected device to create an OpenCL context
	context, _ := cl.CreateContext([]*cl.Device{devices[deviceIndex]})
	defer context.Release()
	fmt.Println(devices[deviceIndex].Name())

	// Create a "Command Queue" bound to the selected device
	queue, _ := context.CreateCommandQueue(devices[deviceIndex], 0)
	defer queue.Release()

	// Create an OpenCL "program" from the source code. (batchedSquareSrc is declared elsewhere)
	program, _ := context.CreateProgramWithSource([]string{batchedSquareSrc})

	// Build the OpenCL program
	if err := program.BuildProgram(nil, ""); err != nil {
		panic("BuildProgram failed: " + err.Error())
	}

	// Create the actual Kernel with a name, the Kernel is what we call when we want to execute something.
	kernel, err := program.CreateKernel("square")
	if err != nil {
		panic("CreateKernel failed: " + err.Error())
	}
	defer kernel.Release()

	// Prepare data, note explicit use of int32 which we know are 4 bytes each.
	elemCount := 16777216
	numbers := make([]int32, elemCount)
	for i := 0; i < elemCount; i++ {
		numbers[i] = int32(i)
	}

	// Prepare for loading data into Device memory by creating an empty OpenCL buffers (memory)
	// for the input data. Note that we're allocating 4x bytes the size of data since each int32
	// uses 4 bytes.
	inputBuffer, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(numbers))
	if err != nil {
		panic("CreateBuffer failed for matrices input: " + err.Error())
	}
	defer inputBuffer.Release()

	// Do the same for the output. We'll expect to get int32's back, the same number
	// of items we passed in the input.
	outputBuffer, err := context.CreateEmptyBuffer(cl.MemReadOnly, 4*len(numbers))
	if err != nil {
		panic("CreateBuffer failed for output: " + err.Error())
	}
	defer outputBuffer.Release()

	// Note that we haven't loaded anything into those buffers yet!

	// This is where we pass the "numbers" to the command queue by filling the write buffer, e.g. upload the actual data
	// into Device memory. The inputDataPtr is a CGO pointer to the first element of the input, so OpenCL
	// will know from where to begin reading memory into the buffer, while inputDataTotalSize tells us the length (in bytes)
	// of the data we want to pass. It's 1024 elements x 4 bytes each, but we can also calculate it on the
	// fly using unsafe.Sizeof.
	inputDataPtr := unsafe.Pointer(&numbers[0])
	inputDataTotalSize := int(unsafe.Sizeof(numbers[0])) * len(numbers) // 1024 x 4
	if _, err := queue.EnqueueWriteBuffer(inputBuffer, true, 0, inputDataTotalSize, inputDataPtr, nil); err != nil {
		panic("EnqueueWriteBuffer failed: " + err.Error())
	}
	fmt.Printf("Enqueed %d bytes into the write buffer\n", inputDataTotalSize)

	// Kernel is our program and here we explicitly bind our 2 parameters to it, first the input and
	// then the output. This matches the signature of our OpenCL kernel:
	// __kernel void square(__global int* input, __global int* output)
	if err := kernel.SetArgs(inputBuffer, outputBuffer); err != nil {
		panic("SetKernelArgs failed: " + err.Error())
	}

	size, _ := kernel.PreferredWorkGroupSizeMultiple(nil)

	maxWGSize := devices[deviceIndex].MaxWorkGroupSize()
	maxWISize := devices[deviceIndex].MaxWorkItemSizes()[0]
	fmt.Printf("Preferred Work Group Size Multiple: %d, MaxWG: %d, MaxWI: %d\n", size, maxWGSize, maxWISize)
	// For fun, time how long the execution takes
	localSizes := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
	fmt.Println(
		`| Device        | Work group size [ Local size | Result  |
| ------------- |:-------------:| -----:| -----:|`)
	iterations := int64(16)
	for _, lz := range localSizes {
		if lz > maxWGSize {
			continue
		}
		var sum = int64(0)
		for it := 0; it < int(iterations); it++ {
			st := time.Now()

			// Finally, start work! Enqueue executes the loaded args on the specified kernel.
			if _, err := queue.EnqueueNDRangeKernel(kernel, nil, []int{elemCount / lz}, []int{lz}, nil); err != nil {
				panic("EnqueueNDRangeKernel failed: " + err.Error())
			}

			// Finish() blocks the main goroutine until the OpenCL queue is empty, i.e. all calculations are done.
			// The results have been written to the outputBuffer.
			if err := queue.Finish(); err != nil {
				panic("Finish failed: %" + err.Error())
			}
			//fmt.Printf("LZ: %d Took: %v\n", lz, time.Since(st))
			sum += time.Since(st).Microseconds()
		}
		fmt.Printf("| %s   | %d | %d | %v |\n", devices[deviceIndex].Name(), 1048576/lz, lz, (time.Microsecond * time.Duration(sum/iterations)).String())
	}

	// Allocate storage for loading the output from the OpenCL program. Remember, we expect
	// the same number of elements and type as the input.
	results := make([]int32, len(numbers))

	// The EnqueueReadBuffer copies the data in the OpenCL "output" buffer into the "results" slice.
	outputDataPtrOut := unsafe.Pointer(&results[0])
	outputDataSizeOut := int(unsafe.Sizeof(results[0])) * len(results)
	if _, err := queue.EnqueueReadBuffer(outputBuffer, true, 0, outputDataSizeOut, outputDataPtrOut, nil); err != nil {
		panic("EnqueueReadBuffer failed: " + err.Error())
	}
	//for i := 0; i < elemCount; i++ {
	//	fmt.Printf("%d ", results[i])
	//}
}
