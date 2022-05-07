########################################################
# Classification work with data and output discrete data
# e.g. true, false,yes,no,1,0
# Logistic Regression is used for classification
########################################################

using Plots, CSV
gr(size=(600,600))

raw"""
    ``` math
    sigmoid = \frac{1}{(1+ e^{-x})} 
    ```
"""
sigmoid(x) = 1/(1+exp(-x))
p_sigmoid = plot(-6:0.1:6, sigmoid,
            xlabel = "Input (x)",
            ylabel = "Output (y)",
            title = "Sigmoid Function",
            legend = false,
            color= :blue
)

# modified sigmoid function
θ_0 = 0.0   #  (b) y-intercept try 1.0 to -1
θ_1 = -.5   #  (m) slope try 0.5 to -0.5

# hypothesis function (linear regression) y = mx + b
z(x) = θ_0 .+ θ_1*x



# sigmoid function 
# with x replaced by z linear function
h(x) = 1 ./(1 .+ exp.(-z(x)))

## replot
plot!(h,color = :magenta,linestyle = :dash)



data = CSV.File("wolfspider.csv")
Y_temp = data.class
X = data.feature

Y =[]
# convert class text to 0,1
for i in 1:length(Y_temp)
    if Y_temp[i] == "present"
        y = 1
    else
        y = 0
    end
    push!(Y,y)
end

p_data = scatter(X,Y,
            xlabel = "Size Grain of sand (mm)",
            ylabel = "Probability of Observation (Absent = 0 | Present = 1",
            title = "Wolf Spider Present Classification",
            legend = false,
            color = :blue,
            markersize = 5
)

# workflow
# Logistic Regression model 
# 1. Initialize the model parameters
# 2. Define the hypothesis function
# 3. Define the cost function
# 4. Define the optimization algorithm
# 5. Initialize hyperparameters
# 6. Change the value of parameters
# 7. recalculate the cost function
# 8. Repeat steps 3-7 until convergence

# 1
θ_0 = 0.0
θ_1 = 0.0
# track value history
t0_history = []
t1_history = []
push!(t0_history,θ_0)
push!(t1_history,θ_1)
#2
z(x) = θ_0 .+ θ_1*x # linear function
h(x) = 1 ./(1 .+ exp.(-z(x))) # sigmoid function
plot!(0:0.1:1.2,h,color = :magenta,linestyle = :dash)
#3  
m = length(X)
y_hat = h(X)

function cost()
    (-1/m) .* sum(Y .* log.(y_hat) + (1 .- Y) .* log.(1 .- y_hat))
end
J = cost()
# track cost function over time
J_history = []
push!(J_history,J)
#4 define optimization algorithm (batch gradient descent)
# use partial derivatives to update theta
function partial_θ_0()
    sum(y_hat - Y)    
end

function partial_θ_1()
    sum((y_hat - Y) .* X)
end
#5 learning rate
α = 0.01
epochs = 0

# looping until convergence
for i in 1:10000
    # calc partial derivative
    θ_0_t = partial_θ_0()
    θ_1_t = partial_θ_1()

    # 6 change the value of parameters
    θ_0 -= α * θ_0_t
    θ_1 -= α * θ_1_t
    push!(t0_history,θ_0)
    push!(t1_history,θ_1)

    #7 update
    y_hat = h(X)
    J = cost()
    push!(J_history,J)

    epochs += 1
    plot!(0:0.1:1.2,h,color = :green,alpha = 0.025,
            linewidth=3,
            title = "Wolf Spider Present Classification (epochs = $epochs)")
end
p_data

# for i in 1:2000000
#     θ_0_t = partial_θ_0()
#     θ_1_t = partial_θ_1()

#     θ_0 -= α * θ_0_t
#     θ_1 -= α * θ_1_t
#     push!(t0_history,θ_0)
#     push!(t1_history,θ_1)

#     y_hat = h(X)
#     J = cost()
#     push!(J_history,J)

#     epochs += 1
#     plot!(0:0.1:1.2,h,color = :green,alpha = 0.025,
#             linewidth=3,
#             title = "Wolf Spider Present Classification (epochs = $epochs)")
# end
# p_data

# plot the learning curve
p_learn = plot(0:epochs,J_history,
            xlabel = "Epochs",
            ylabel = "Cost Function",
            title = "Wolf Spider Present Classification Learning Curve",
            legend = false,
            color = :blue,
            linewidth = 2
)

# plot parameters
p_param = scatter(t1_history,t0_history,
            xlabel = "θ_1",
            ylabel = "θ_0",
            title = "Gradient descent Path",
            legend = false,
            color = :blue,
            linewidth = 2,
            alpha = 0.05
)

# make predition
# Probability of wolfspider present in this grain of sand
newX = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
h(newX)

# Logistic regression function
σ(X, θ) = 1 ./ (1 .+ exp.(-X*θ))

predict(f, x) = f(x) > 0.5

function logistic_regression(X, y; T=1000, α=1e-4)
    𝐗 = mapreduce(x->[1;x]', vcat, X)
    θ = zeros(2)
    for iteration in 1:T
        θ = θ + α * 𝐗' * (y - σ(𝐗,θ))
    end
    return x -> σ([1;x]', θ)
end


