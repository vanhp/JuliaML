################################################################################################################
# deep Learning with flux library
#################################3#########################################

using Flux,Images,MLDatasets, Plots
using Flux: crossentropy,onecold,onehotbatch,train!
using LinearAlgebra,Random,Statistics

# start with random seed 
Random.seed!(1)
# load data  MNIST or already split 60000 train, 10000 test set
X_train_raw,y_train_raw = MLDatasets.MNIST.traindata(Float32)
X_test_raw,y_test_raw = MLDatasets.MNIST.testdata(Float32)
X_train_raw
index = 1
img = X_train_raw[:,:,index]

# image is loaded horizontally so it must be transpose with '
colorview(Gray,img')

# check the label for the image
y_train_raw
y_train_raw[index]

# check test data for the image 
img_test = X_test_raw[:,:,index]
colorview(Gray,img')
y_test_raw[index]

# preprocessing data    
# data need to be flatten from 3D to 2D matrix for the API 
X_train = Flux.flatten(X_train_raw)
X_test = Flux.flatten(X_test_raw)

# then do the onehot encoding step and auto concatenate into 10x60000 onehot matrix 
# of 0,1 boolean
y_train = onehotbatch(y_train_raw,0:9)
y_test = onehotbatch(y_test_raw,0:9)

# create model 
model = Chain(Dense(28*28,32,relu),
                Dense(32,10),
                softmax)

# loss function  
loss(x,y) = crossentropy(model(x),y)

########################################################################################
# Optimization 
# Flux support many optimizers 
# Gradient descent optimizer Flux.Optimise.Descent
# Momentum use previous update to calculate future update
# ADAM (adaptive moment estimation) a subset of stochastic gradient descent
# it calculate gradient base on random subset of a batch rather than whole batch
# in addtion it start with initial learning rate but let it decay over time
# this let it reach the objective faster than SGD 
########################################################################################

# set the optimizer
learning_rate = 0.01
optim = ADAM(learning_rate)
# track the parameters
# the param function create a Param Object that point to
# its trainable parameters
# The parameters are store in memory in 4 Arrays
ps = Flux.params(model)

# check the value stored in the param Arrays
# ps[1]
# check the range of value with extrema(ps[3]) function


loss_history = []
epochs = 500
# train the modul
for epoch in 1:epochs
    train!(loss,ps,[(X_train,y_train)],optim)
    train_loss = loss(X_train,y_train)
    push!(loss_history,train_loss)
    println("Epoch = $epoch :  Traing loss = $train_loss")
end

# making prediction 
ŷ_raw = model(X_test)

# using the onecold function to reverse the onehot to understand result easier 
# it convert the matrix of output into column vector containing index number
# of the highest probability value, then adjust for 0 indexing
ŷ = onecold(ŷ_raw) .- 1 
y = y_test_raw 
# compare the prediction label index with the actual label index 
# to see the accuracy
mean(ŷ .== y)

# show the result
check = [ŷ[i] == y[i] for i in 1:length(y)]
index = collect(1:length(y))
check_display = [index ŷ y check]
vscodedisplay(check_display)

# look at the misclassified
mis_class_index = 9 
img = X_test_raw[:,:,mis_class_index]
colorview(Gray,img')
y[mis_class_index]
ŷ[mis_class_index]

# let plot the loss_history
gr(size=(600,600))

# plot the learning curve see how it's learning
pl_curve = plot(1:epochs,
            loss_history,
            xlabel = "Epochs",
            ylabel= "Loss",
            title="Learning Curve",
            lengend= false,
            color=:blue,
            linewidth=2)

savefig(pl_curve,"learnCurve.svg")