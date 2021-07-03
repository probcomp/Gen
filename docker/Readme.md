# Gen Containerization

## Quickstart

### Docker Setup
Please follow the [official Docker installation instructions](https://docs.docker.com/engine/install/) to setup Docker on your target platform. If you would like to use GPU acceleration, please also install the [NVIDIA Container Toolkit / nvidia-docker](https://github.com/NVIDIA/nvidia-docker).


### Building Container
You can build Docker containers for Gen that either target CPU-only or GPU-enabled machines.
- Please run `docker build -f docker/GenCPU.dockerfile -t gen:cpu-ubuntu20.04 .` for the CPU image
- Please run `docker build -f docker/GenGPU.dockerfile -t gen:gpu-ubuntu20.04 .` if you want to use GPU acceleration
Both commands are supposed to be run from the main Gen.jl folder, not this subfolder.

### Running Container
You can run `docker run -it gen:cpu-ubuntu20.04 bash` to run the CPU image and `docker run -it --gpus all gen:gpu-ubuntu20.04 bash` to run the GPU image. Afterwards, you can just execute `julia`, type `using Gen` and start developing with Gen. If you would like to remove the container, please add the flag `--rm` to your run command.

## Known Issues

### Run or Build Script Not Executable
If you get an error like
```
bash: ./docker_build_gpu.sh: Permission denied
```
Then please make the script executable via `chmod +x docker_build_gpu.sh` (analogous for CPU).

### Known Issues with Docker on macOS
The Docker container might currently fail on macOS with the following error:
```
julia> using Gen
[ Info: Precompiling Gen [ea4f424c-a589-11e8-07c0-fd5c91b9da4a]
ERROR: Failed to precompile Gen [ea4f424c-a589-11e8-07c0-fd5c91b9da4a] to /root/.julia/compiled/v1.4/Gen/OEZG1_t5nDi.ji.
Stacktrace:
 [1] error(::String) at ./error.jl:33
 [2] compilecache(::Base.PkgId, ::String) at ./loading.jl:1272
 [3] _require(::Base.PkgId) at ./loading.jl:1029
 [4] require(::Base.PkgId) at ./loading.jl:927
 [5] require(::Module, ::Symbol) at ./loading.jl:922
 ```

 This can even happen when running inside a VM on macOS. While this bug is confirmed and reproducable, there is currently no solution for it, cmp. [issue 311](https://github.com/probcomp/Gen.jl/issues/311).
