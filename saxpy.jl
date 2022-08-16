using CUDA
CUDA.versioninfo()
using BenchmarkTools

# define Constants
const dim = 100_000_000
const a = 3.1416

# SAXPY (CPU)
x = ones(Float32,dim)
y = ones(Float32,dim)
z = zeros(Float32,dim)
typeof(x)
# serieal op on cpu
@btime z .= a .* x .+ y
# cpu_time = 232.091 ms

# SAXPY on (GPU)
x = CUDA.ones(Float32,dim)
y = CUDA.ones(Float32,dim)
z = CUDA.zeros(Float32,dim)
typeof(x)
@btime CUDA.@sync z .= a .* x .+ y
# gpu_time = 8.456

# compare
232.091 / 8.456