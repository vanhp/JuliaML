############################################################
# applying knn to iris dataset
# the selection of k value is important because it may 
# have different accuracy results.
# Tip: it should not be too low or too high
#     should odd number to avoid tie vote 
#     should be k > # of classes
#     try different value of k to see results
#     feature scaling is important if they are different
# It's non-probabilistic and transparent it easier to 
# try to understand how it works
############################################################

using RDatasets, StatsBase,Statistics

iris = dataset("datasets","iris")
X = Matrix(iris[:,1:4])
y = Vector{String}(iris.Species)


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

# identify index for train and test set
Random.seed!(1)
index_train = perlabel_splits(y,0.67)
index_test = setdiff(1:length(y),index_train)
# split data to train and test set
X_train = X[index_train,:]
X_test = X[index_test,:]
y_train = y[index_train]
y_test = y[index_test]
# transpose data to match function arguments
X_train_transpose = permutedims(X_train)
X_test_transpose = permutedims(X_test)
kdTree = KDTree(X_train_transpose)
k = 11
index_knn, distance = knn(kdTree,X_test_transpose,k,true)
output = [index_test index_knn distance]
vscodedisplay(output)

# converting to classification 
index_knn_matrix = hcat(index_knn...)
index_knn_matrix_transpose = permutedims(index_knn_matrix)
vscodedisplay(index_knn_matrix_transpose)

# replace numbers with label
knn_classes = y_train[index_knn_matrix_transpose]
vscodedisplay(knn_classes)

# make predictions with StatsBase 
ŷ = [argmax(countmap(knn_classes[i,:]))
    for i in 1:length(y_test)
]
# how countmap and argmax work 
demo = knn_classes[53,:]
# countmap is like histogram by create Dict
cm_demo = countmap(demo)
# the argmax return the index of the largest item
amax = argmax(cm_demo)

# maximum return the larger sorted order alphabetically
mmax = maximum(cm_demo)

# check accuracy
accuracy = mean(ŷ .== y_test)
check = [ŷ[i] == y_test[i] for i in 1:length(ŷ)]
check_display = [ŷ y_test check]
vscodedisplay(check_display)
