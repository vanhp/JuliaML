##########################################################################################################
# K nearest neighbors 
# k stand for number of item to identify
# measuring distance from interest point to the closest item
# then sort them inorder from low (closest) to high (farthest)
#
# k-dimensional Tree is a generalization of binary search tree algorithm 
# for multi-dimensional structure combine with knn make ML practical
# 
# use in many field e.g. ML,robotic...
# any field that require analysis base on distance
# not practical for large number of dataset(millions)
#
################################################################################################
using NearestNeighbors, Plots
using Random

# Initialize the plot 
gr(size = (600,600))
# generate random point usin random seed to be able to reproduce the result
Random.seed!(1)
f1_train = rand(100)
f2_train = rand(100)
p_knn = scatter(f1_train, f2_train,
            xlabel = "Feature 1",
            ylabel = "Feature 2",
            title = "k-NN & k-D Tree Demo",
            legend = false,
            color=:blue)

# build Tree

X_train = [f1_train f2_train]
# transpose the matrix using permutedims function
X_train_t = permutedims(X_train)

# KDTree constructor map the data to more efficient data structure
# to used for analysis
kdtree = KDTree(X_train_t)

# initialize the k of k-NN 
# try to find the 11 point that nearest euclidian distance points are
# to the test data point
k = 11

# generate random point for testing
f1_test = rand()
f2_test = rand()
X_test = [f1_test, f2_test]

# the plot function need x,y coordinate in array coordinates
scatter!([f1_test], [f2_test],color= :red,markersize=10)

# find the nearest neighbors using k-NN & k-d tree 
# the knn returned the 11 nearest neighbor of test point
index_knn,distance = knn(kdtree,X_test,k,true)
output = [index_knn distance]
vscodedisplay(output)

# ploting the 11 neighbors
f1_knn = [f1_train[i] for i in index_knn]
f2_knn = [f2_train[i] for i in index_knn]
scatter!(f1_knn,f2_knn,color=:yellow,markersize=10,alpha=0.5)

# connect neighbors to test points with a line
for i in 1:k
    plot!([f1_test,f1_knn[i]],[f2_test,f2_knn[i]],color=:green)
end
p_knn

savefig(p_knn,"knn_plot.svg")

