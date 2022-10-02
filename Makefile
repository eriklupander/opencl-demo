all: fmt build test

.PHONY: build
build:
	go build -o bin/opencl-demo cmd/opencl-demo/main.go

.PHONY: run
run: build
	./bin/opencl-demo -device=0 -op=square

.PHONY: test
test:
	go test ./... -count=1 -v

.PHONY: fmt
fmt:
	go fmt ./...