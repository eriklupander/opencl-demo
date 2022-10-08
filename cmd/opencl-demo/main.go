package main

import (
	"flag"
	"fmt"
	"github.com/eriklupander/ocltest/internal/app"
)

func main() {
	op := flag.String("op", "square", "Demo to run: square, square-local, structs, multidim, vectors, batched-square, benchmark")
	deviceIndex := flag.Int("device", 0, "OpenCL device index")
	flag.Parse()

	switch *op {
	case "structs":
		app.Structs(*deviceIndex)
	case "multidim":
		app.MultiDim(*deviceIndex)
	case "vectors":
		app.Vectors(*deviceIndex)
	case "batched-square":
		app.BatchedSquare(*deviceIndex)
	case "square":
		app.Square(*deviceIndex)
	case "square-local":
		app.SquareLocalSize(*deviceIndex)
	case "benchmark":
		app.Benchmark(*deviceIndex)
	case "benchmark2":
		app.Benchmark2(*deviceIndex)
	default:
		fmt.Printf("Unknown op: %s. Options: square, square-local, structs, multidim, vectors, batched-square, benchmark\n", *op)
	}
}
