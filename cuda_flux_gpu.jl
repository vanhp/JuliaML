using Flux, MLDatasets, CUDA

using Flux: crossentropy, flatten,onehotbatch,params,train!
using Random

Random.seed!(1)

# to use gpu for ML 
# 1. Load CUDA
# 2. Load data into gpu with |> gpu
# 3. Load model into gpu with |> gpu
# Flux will do the rest


# load data 
# MLDatasets into gpu
X_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:] |> gpu
X_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:] |> gpu
typeof(X_train_raw)

# flatten input to be 1D
X_train = flatten(X_train_raw)
X_test = flatten(X_test_raw)

# one-hot encode the labels
y_train = onehotbatch(y_train_raw,0:9)
y_test = onehotbatch(y_test_raw,0:9)

# define architecture
model = Chain(
    Dense(28 * 28,32,relu),
    Dense(32,10),
    softmax
) |> gpu
typeof(model)

# define loss function
loss(x,y) = crossentropy(model(x),y) 

# track parameters
ps = params(model)

# optimizer to improve performance of the model
learning_rate = Float32(0.01)
optim = ADAM(learning_rate)

# train the model
epochs = 500

@time CUDA.@sync for epoch in 1:epochs
    train!(loss,ps,[(X_train,y_train)],optim)
end

gpu_time = 43.457
gpu_compile_time = gpu_time * 67.57 /100
gpu_run_time = gpu_time - gpu_compile_time 

# CPU vs GPU time

elapse_time = cpu_time / gpu_time
compile_time = cpu_compile_time /gpu_compile_time
run_time = cpu_run_time / gpu_run_time