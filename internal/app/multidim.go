package app

import (
	"fmt"
	"github.com/jgillich/go-opencl/cl"
	"time"
	"unsafe"
)

var squareRootSrc = `
__kernel void squareRoot(
   __global float* input,
   __global float* output)
{
   int groupId_col = get_group_id(0);
   int groupId_row = get_group_id(1);

   int row = get_global_id(1);         // get row from second dimension
   int col = get_global_id(0);         // get col from first dimension
   int colCount = get_global_size(0);  // get number of columns
   int index = row * colCount + col;   // calculate 1D index
   int localId = get_local_id(0);
   printf("row: %d, col: %d, i: %d, local id: %d, groupId_row: %d, groupId_col: %d\n", row, col, index, localId, groupId_row, groupId_col);
   output[index] = sqrt(input[index]);
}
`

func MultiDim(deviceIndex int) {
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

	// Create an OpenCL "program" from the source code. (squareSrc is declared elsewhere)
	program, _ := context.CreateProgramWithSource([]string{squareRootSrc})

	// Build the OpenCL program
	if err := program.BuildProgram(nil, ""); err != nil {
		panic("BuildProgram failed: " + err.Error())
	}

	// Create the actual Kernel with a name, the Kernel is what we call when we want to execute something.
	kernel, err := program.CreateKernel("squareRoot")
	if err != nil {
		panic("CreateKernel failed: " + err.Error())
	}
	defer kernel.Release()

	// Prepare data, note explicit use of float32 which we know are 4 bytes each.
	elems := 4
	elemCount := elems * elems
	numbers := make([]float32, elemCount)
	for i := 0; i < elemCount; i++ {
		numbers[i] = float32(i)
	}
	for i := 0; i < elems; i++ {
		for j := 0; j < elems; j++ {
			fmt.Printf("|%f", numbers[i*4+j])
		}
		fmt.Printf("|\n")
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

	wgSize, err := kernel.WorkGroupSize(devices[deviceIndex])
	if err != nil {
		panic("get work group size: " + err.Error())
	}
	preferredMultiple, err := kernel.PreferredWorkGroupSizeMultiple(devices[deviceIndex])
	if err != nil {
		panic("get preferred multiple: " + err.Error())
	}
	fmt.Printf("WorkGroupSize: %d\n", wgSize)
	fmt.Printf("Preferred multiple: %d\n", preferredMultiple)
	fmt.Printf("Work item sizes: %v\n", devices[deviceIndex].MaxWorkItemSizes())
	fmt.Printf("Max compute units: %v\n", devices[deviceIndex].MaxComputeUnits())
	fmt.Printf("Max samplers: %v\n", devices[deviceIndex].MaxSamplers())

	st := time.Now()

	// Finally, start work! Enqueue executes the loaded args on the specified kernel.
	if _, err := queue.EnqueueNDRangeKernel(kernel, nil, []int{elems, elems}, []int{1, 1}, nil); err != nil {
		panic("EnqueueNDRangeKernel failed: " + err.Error())
	}

	// Finish() blocks the main goroutine until the OpenCL queue is empty, i.e. all calculations are done.
	// The results have been written to the outputBuffer.
	if err := queue.Finish(); err != nil {
		panic("Finish failed: %" + err.Error())
	}

	fmt.Printf("Took: %v\n", time.Since(st))

	// Allocate storage for loading the output from the OpenCL program. Remember, we expect
	// the same number of elements and type as the input.
	results := make([]float32, len(numbers))

	// The EnqueueReadBuffer copies the data in the OpenCL "output" buffer into the "results" slice.
	outputDataPtrOut := unsafe.Pointer(&results[0])
	outputDataSizeOut := int(unsafe.Sizeof(results[0])) * len(results)
	if _, err := queue.EnqueueReadBuffer(outputBuffer, true, 0, outputDataSizeOut, outputDataPtrOut, nil); err != nil {
		panic("EnqueueReadBuffer failed: " + err.Error())
	}
	for i := 0; i < elems; i++ {
		for j := 0; j < elems; j++ {
			fmt.Printf("|%f", results[i*4+j])
		}
		fmt.Printf("|\n")
	}
}
