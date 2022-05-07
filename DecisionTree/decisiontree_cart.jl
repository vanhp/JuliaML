using DecisionTree
using Random,Statistics

X, y = load_data("iris")

# assign type to dataset
X = float.(X)
y = string.(y)
iris = [X y]

 vscodedisplay(iris)

function perlabel_splits(label,percent)
    uniq_class = unique(label)
    keep_index = []
    for class in uniq_class
        class_index = findall(label .== class)
        row_index = randsubseq(class_index,percent)
        push!(keep_index,row_index...)
    end
    return keep_index
end

# calc how to split labels using 67% into train,test portion
Random.seed!(1)
train_label = perlabel_splits(y,0.67)
test_label = setdiff(1:length(y),train_label)

# split data set features input
X_train = X[train_label,:]
X_test = X[test_label,:]

# split classes labels
y_train = y[train_label]
y_test = y[test_label]


####################################################################
# Build a Dicision tree model
# and run it
####################################################################

model = DecisionTreeClassifier(max_depth=-1)
fit!(model,X_train,y_train)

print_tree(model)

# check the output data
train_data = [X_train y_train ]
 vscodedisplay(train_data)

right_train =  train_data[train_data[:,4].>0.8,:]
 vscodedisplay(right_train)

ŷ = predict(model,X_test)
accuracy = mean(ŷ .== y_test)

#####################################################################################
# <span style="color:lightblue"> Analyze the result </span>
# using confusion matrix to see where it went wrong
# How to read the matrix
# 1 argument is the row
# 2 argument is the column 
#  the diagonal is the correct value
# Kappa coefficients indicate more nuance assessment of the prediction
# to help remove the correct prediction purely by chance 
#
# Decision tree assign probability for its prediction
#
# It's not very good at making prediction, although it provides lot of 
# informative data 
#
# Bias-Variance trade-off technique may help in reducing the inaccuracy of
# Decision trees
# 1. underfitting: don't have enough relation information about the input and output
# 2. overfitting: too much information about the particular 
#    input and output but not applicable to other data
# 3. just right: understanding relationship information to applicable to other data
#
# Decision trees are generally produce high variance with low bias. Since it has a
# the tendency to overfit the data because it might continue to split the
# data until it satisfies. The lower right quadrant
#  where lower left is the best low variance and low bias,higher right is the worst with higher
#  bias and variance. Most model generally fit in between low variance and high bias or low variance
#  with high bias
# 
# To alleviate the overfitting tendency of Dicision tree use Ensemble learning
# 1. Bagging  (bootstrap Aggregating)
#    1.  reduce variance by 
#       a. increase the independence of features
#       b. Increase number of models
#       c. combine both technique into meta-model that result in lower variance but higher bias
#    Random Forest
#    use independence feature
#    use multiple decision tree model
#    use randomness by only consider a fraction of the features at every split
#    drawback it's blackbox model where detail of how the dicision get made is not shown
# 2. Boosting (Ensemble 2) use to reduce bias
#    a. AdaBoost (adaptive Boosting)
#       start with a Strump (a tree with only root node and 2 leaf nodes)
#       work with multi classes classification.
#    b. work by give more weight to the error so it can pay more attend to it 
#       in the next round of stump
#    c. the idea is focus more on harder problem
#    d. repeat this process until the end result is satisfies
######################################################################################
confusion_matrix(y_test,ŷ)

# show the result
result = [ŷ[i] == y_test[i] for i in 1:length(ŷ)]

# vscodedisplay([ŷ y_test result])

probab = predict_proba(model,X_test)
# vscodedisplay(probab)

# using Literate
# Literate.notebook("decisiontree_cart.jl")

######################################################################################
# Random forest (bagging) less confident than decision tree
model = RandomForestClassifier(n_trees = 20)
fit!(model,X_train,y_train)
ŷ = predict(model,X_test)
accuracy = mean(ŷ .== y_test)

confusion_matrix(y_test,ŷ)
# check the mistake
error = [ŷ[i] == y_test[i] for i in 1:length(ŷ)]
show_error = [ŷ y_test error]
vscodedisplay(show_error)

# show probability of each prediction
probab = predict_proba(model,X_test)

######################################################################################
# AdaBoost (boosting) less confident than random forest
model = AdaBoostStumpClassifier(n_iterations = 20)
fit!(model,X_train,y_train)
ŷ = predict(model,X_test)
accuracy = mean(ŷ .== y_test)

confusion_matrix(y_test,ŷ)
# check the mistake
error = [ŷ[i] == y_test[i] for i in 1:length(ŷ)]
show_error = [ŷ y_test error]
vscodedisplay(show_error)

# show probability of each prediction
probab = predict_proba(model,X_test)