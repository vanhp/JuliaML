######################################################
# Support vector machine is a method of
# machine learning
#
######################################################

using LIBSVM, RDatasets
using LinearAlgebra, Random, Statistics

iris = dataset("datasets","iris")
vscodedisplay(iris)

######################################################
# cross validation a method of spliting the dataset
# into training and testing (validation) set
# use the validation set to comfirm the working
# of the training
######################################################

X = Matrix(iris[:,1:4])
y = iris.Species

# define function to split dataset into train and test set
function perclass_splits(y,percent)
    uniq_class = unique(y)
    keep_index = []
    for class in uniq_class
        class_index = findall(y .== class)
        row_index = randsubseq(class_index,percent)
        push!(keep_index,row_index...)
    end
    return keep_index
end

Random.seed!(1)
train_index = perclass_splits(y,0.67)
test_index = setdiff(1:length(y),train_index)

#########################################################################
# SVM model
#########################################################################
Xtrain = X[train_index,:]
Xtest = X[test_index,:]
ytrain = y[train_index]
ytest = y[test_index]

# the LIBSVM package requires the orientation of the feature dataset
# for both train and test set to be in horizotal instead of vertical
# meaning each row should contain feature ande each column should contain
# the samples.

# this require the data must be transpose
Xtrain_t = Xtrain'
Xtest_t = Xtest'

# run the model
model = svmtrain(Xtrain_t,ytrain)

# make prediction
ŷ,decision_value = svmpredict(model,Xtest_t)
# check accuracy
accuracy = mean(ŷ .== ytest)
# display the result
check = [ŷ[i] == ytest[i] for i in 1:length(ŷ)]
check_display = [ŷ ytest check]
vscodedisplay(check_display)

##########################################
# feature scaling technique
# 1. Rescaling(min-max normalization)
#       x' = x - min(x)/max(x) - min(x)
# 2. Standardization (Z-score Normalization)
#       x' = (x - x̄)/σ
# it might not be improve the accuracy
###########################################

# try method 1 min-max normalization
# split features into separate vectors
f1 = iris.SepalLength
f2 = iris.SepalWidth
f3 = iris.PetalLength
f4 = iris.PetalWidth
# mean nomalization
f1_min = minimum(f1)
f2_min = minimum(f2)
f3_min = minimum(f3)
f4_min = minimum(f4)

f1_max = maximum(f1)
f2_max = maximum(f2)
f3_max = maximum(f3)
f4_max = maximum(f4)

f1_norm = (f1 .- f1_min) ./ (f1_max - f1_min)
f2_norm = (f2 .- f2_min) ./ (f2_max - f2_min)
f3_norm = (f3 .- f3_min) ./ (f3_max - f3_min)
f4_norm = (f4 .- f4_min) ./ (f4_max - f4_min)

X = [f1_norm f2_norm f3_norm f4_norm]
vscodedisplay(X)

# try method 2 z-score
f1_bar = mean(f1)
f2_bar = mean(f2)
f3_bar = mean(f3)
f4_bar = mean(f4)

# standard deviation
f1_std = std(f1)
f2_std = std(f2)
f3_std = std(f3)
f4_std = std(f4)

# guausian distribution
f1_s = (f1 .- f1_bar) ./ f1_std
f2_s = (f2 .- f2_bar) ./ f2_std
f3_s = (f3 .- f3_bar) ./ f3_std
f4_s = (f4 .- f4_bar) ./ f4_std
X = [f1_s f2_s f3_s f4_s]
vscodedisplay(X)