using CUDA
CUDA.versioninfo()
using BenchmarkTools

# define Constants
const dim = 100_000_000
const a = Float32(3.1416)

# # SAXPY (CPU)
# x = ones(Float32,dim)
# y = ones(Float32,dim)
# z = zeros(Float32,dim)
# typeof(x)
# # serieal op on cpu
# @btime z .= a .* x .+ y
# # cpu_time = 232.091 ms

# # SAXPY on (GPU)
# x = CUDA.ones(Float32,dim)
# y = CUDA.ones(Float32,dim)
# z = CUDA.zeros(Float32,dim)
# typeof(x)
# @btime CUDA.@sync z .= a .* x .+ y
# # gpu_time = 8.456

# compare
232.091 / 8.456

cpu_time = 232.091
gpu_time = 8.456
broadcast_time = 2.931

# SAXPY on (GPU)
x = CUDA.ones(Float32,dim)
y = CUDA.ones(Float32,dim)
z = CUDA.zeros(Float32,dim)
typeof(x)

# CUDA structure
# threads then block then grid
# 1 thread per task
# 1 block can have max 1024 threads
# grid is the whole job all the block to do the job 


nthreads = CUDA.attribute(
    device(),
    CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK

)
# calc #block need to to task using ceiling division to round up
# or # of block per grid
nblocks = cld(dim,nthreads)

# define CUDA kernel
# CUDA uses 0 indexing for array
# Julia uses 1 indexing 
# CUDA dimension uses 1 for vector 
#           2 for matrix
#           3 for 3D tensor
#      thread index start with 1 -> max
# CUDA is working in paralell by default make sure that
# the task is independ for each thread.
# there no need for loop or broadcasting construct

function saxpy_gpu_kernel!(z,a,x,y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(z)
        # remove array bound checking for performance
        @inbounds z[i] = a * x[i] + y[i]
    end
    return nothing
end

# launch the CUDA kernel
# using @sync to force cpu to wait for GPU task to complete
@btime CUDA.@sync @cuda(
    threads = nthreads,
    blocks = nblocks,
    saxpy_gpu_kernel!(z,a,x,y)
)
cuda_kernel_time = 9.294
z

cuda_vs_cpu = cpu_time / cuda_kernel_time
cuda_vs_broadcast = broadcast_time / cuda_kernel_time

# using CUDA library
using CUDA.CUBLAS

CUBLAS.axpy!(dim,a,x,y)
# output is in y
y

y = CUDA.ones(Float32,dim)
@btime CUDA.@sync CUBLAS.axpy!(dim,a,x,y)

cubla_time = 7.948


cuda_vs_cpu = cpu_time / cubla_time
cuda_vs_broadcast = broadcast_time / cubla_time
cublas_lib_time = cuda_kernel_time / cubla_time

