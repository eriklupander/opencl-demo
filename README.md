# opencl-demo
OpenCL with Go demo app

### Usage
Use `-device=<deviceIndex>` and `-op=<opname>` to select device and which "demo" to run.

Available demos:
* square - Hello-world like, squares the passed input.
* batched-square - Benchmarks the square scenario using various workgroup sizes
* structs - How to pass a Go struct into a C struct
* vectors - Simple vector x matrix multiplication
* multidim - Showcases use of multi-dimensional work group counts

```shell
make build
./bin/opencl-demo -device=0 -op=square

Device 0 - Intel(R) Core(TM) i7-4870HQ CPU @ 2.50GHz: max work group size: 1024
Device 1 - Iris Pro: max work group size: 512
Device 2 - GeForce GT 750M: max work group size: 1024
Intel(R) Core(TM) i7-4870HQ CPU @ 2.50GHz
Enqueed 4096 bytes into the write buffer
Took: 117.661µs
0 1 4 9 16 25 36 49 64 81 100 121 144 ... rest omitted
```

## Sources
See /internal/app for the various demos. Each example has full boilerplate.