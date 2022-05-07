using CSV,GLM,Plots,TypedTables

###############################################################
# Regress work with the data and produce output 
# that are continous e.g. 13.5,2799.
# Using GLM and OLS approach to predict 
# the price of a house base on size
###############################################################
data = CSV.File("housedata.csv")
X = data.size
Y = round.(Int,data.price/1000)
t = Table(X=X,Y=Y)
gr(size=(600,600))
p_scatter = scatter(X,Y, 
            xlim=(0,5000), 
            ylim=(0,800), 
            xlabel="Size (sqft)", 
            ylabel="Price (kdollar)", 
            title="Scatter Plot Housing Price in Portland",
            legend = false,
            color = :red 
)

# GLM (Generalized linear model) package for linear regression model
# ordinary least squares (OLS) takes the squares of vertical distances
# between predicted value and actual value and adds them together.
# GLM will return value of the y-intercept and slope of the line that minimizes
# the sum of squares of the vertical distances between the predicted value and
# the actual value.

ols = lm(@formula(Y ~ X), t)
plot!(X,predict(ols),color=:green,linewidth=3)

# predict price base on new value for size
newX = Table(X=[1250])
predict(ols,newX)

###############################################################
# Machine Learning approach
###############################################################
epochs = 10
p_scatter = scatter(X,Y, 
            xlim=(0,5000), 
            ylim=(0,800), 
            xlabel="Size (sqft)", 
            ylabel="Price (kdollar)", 
            title="Scatter Plot Housing Price in Portland ($epochs)",
            legend = false,
            color = :red 
)
# parameters for the model in ML
# m is weight (slope)
# b is bias (y-intercept)
Θ_0 = 0.0   # (b) y intercept from y = mx + b
θ_1 = 0.0   # m slope

# linear regression model h = mx+b
h(x) = Θ_0 .+ θ_1*x

# cost function ``` $ J(θ_0,θ_1) = 1/2m ∑i=1 m (h(xi)-yi)^2 ```
# cost function J(theta) = 1/2m * sum(h(x) - y)^2
plot!(X,h(X),color=:green,linewidth=3)
m = length(X)
y_hat = h(X)

function cost(X,Y)
    (1/(2*m)) * sum((y_hat - Y).^2)
    
end

# save cost value that change over time
J = cost(X,Y)
J_history = []
push!(J_history,J)

# gradient descent
# gradient descent is a method of minimizing a function by iteratively
# taking steps along the direction of the negative of the gradient
# of the function at the current point.
# gradient descent is a simple and efficient way to find the minimum
# of a function.
# in this case, we make a small adjustment to input by calculating the
# loss function (cost function) 

function bias_theta_0(X,Y)
    (1/m) * sum(y_hat - Y)
end

    
function weight_theta_1(X,Y)
    (1/m) * sum((y_hat - Y) .* X)
end

# learning rate α (very small value)
α = 0.09  
α1  = 0.00000008

###############################################################
# calculate partial derivative 
# partial derivative of cost function with respect to theta_0
# partial derivative of cost function with respect to theta_1
# θ_1 = θ_1 - α * 1/m Σ (i=0,m) (h(x) - y)* xi 

θ_0_t = bias_theta_0(X,Y)
θ_1_t = weight_theta_1(X,Y)

# update theta
θ_0 -= α*θ_0_t
θ_1 -= α1*θ_1_t

# recheck the progress of the model
y_hat = h(X)
J = cost(X,Y)
push!(J_history,J)

epochs += 1
plot!(X,y_hat,color=:green,linewidth=3,alpha=0.5,
        title="house price in Portland (epochs = $epochs)"
)

###############################################################
# compare to OLS GLM approach
plot!(X,predict(ols),color=:blue,linewidth=3,alpha=0.5)

# plot learning curve
gr(size=(600,600))
p_line = plot(0:epochs,J_history,color=:red,linewidth=3,
xlabel="Epochs",ylabel="Cost Function",title="Learning Curve",legend=false)

newX_ml = [1250]
h(newX_ml)

###############################################################
# Using GLM and OLS approach to predict the price of a house base on size
predict(ols,newX)