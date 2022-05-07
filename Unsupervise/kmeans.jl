# Unsupervise learning 
# To learn from unlabel dataset 

# K-means is an unsupervise learning field that take the whole dataset 
# and partition it into k number of cluster of the average value 
# like-property togethers
# K >= to 1 but K < number of data
# The goal of K-means is to classify all data into one of the cluster
# to minimize the within-cluster sum of squares value
# cost function to minimize ∑ ₁ⁿ ‖Xᵢ - μᵥ‖² w.r.t (μ,v)
# because it uses the distance calculation it similar to 
# K-Nearest neighbors algorithm, where k-means k refer number of Cluster
# where k in K-nn refer to number of data that nearest to point
# this is also an iterative process that try to minimize the cost function

# The K-means only take the input but provide no output as all unsupervise 
# learning algorithm works
#  comparison of the K algorithm
#       K-NN                        K-means
#     supervised                  unsupervised
#     Classification              Clustering
#     Labeled                     unlabeled
#     K number near a point       number of Cluster 
#     prediction                  no prediction
    
# K-means is used for when accuracy is not require or data without label
# use to take a quick peek into the data to some idea of how to handle them



using Clustering, Plots, Random
gr(size = (600,600))

Random.seed!(1)
f1 = rand(100)
f2 = rand(100)
p_rand = scatter(f1, f2,
        xlabel = "feature 1", 
        ylabel = "feature 2",
        title = "Random data",
        legend = false)

# to use k-means the input data must be in row(horizontal)
# each row is the feature and each column is the sample data   

# transpose into 2 row and 1 column
X = [f1 f2]'
# cluster groups
k = 5

# k-means is self iterate no need to write loop. It will stop when 
# it found the solution, depend on data it may take too long, so 
# it's good idea to set max loop to prevent infinite
iter = 100
Random.seed!(1)
results = kmeans(X,k;maxiter=iter,display=:iter)

# which cluster sample was assigned to
a = assignments(results)
c = counts(results)
# center of the cluster
μ = results.centers

p_kmeans = scatter(f1, f2,
        xlabel = "feature 1", 
        ylabel = "feature 2",
        title = "Random data",
        legend = false,
        group = a,
        markersize=10,
        alpha =0.7)

# show centers cluser group
p_center = scatter!(μ[1,:], μ[2,:],
        color=:yellow,
        markersize=20,
        alpha =0.7,
        legend = false)
savefig(p_kmeans,"plotKmeans.svg")

################################################################
# Application of kmeans 
################################################################

using RDatasets
cats = dataset("boot","catsM")
vscodedisplay(cats)

p_cats = scatter(cats.BWt,cats.HWt,
xlabel = "Body Weight (kg)",
ylabel = "Heart weight (g)",
title = "weight of cat body vs heart",
legend= true)

###############################################
# normalization (scale feature)
# since the data are in difference format
# both axes are in difference scale 
# must nomalize them to the same scale
f1 = cats.BWt 
f2 = cats.HWt 
f1_min= minimum(f1)
f2_min= minimum(f2)

f1_max= maximum(f1)
f2_max= maximum(f2)

f1_norm = (f1 .- f1_min) ./ (f1_max .- f1_min)
f2_norm = (f2 .- f2_min) ./ (f2_max .- f2_min)
X = [f1_norm f2_norm]'

p_cats = scatter(f1_norm,f2_norm,
xlabel = "Body Weight (kg)",
ylabel = "Heart weight (g)",
title = "weight of cat body vs heart (normalization)",
legend= true)

k = 3
iter = 5
Random.seed!(1)
result = kmeans(X,k;maxiter=iter,display=:iter)
a = assignments(result)
c = counts(result)
μ = result.centers

p_kmeans2 = scatter(f1_norm, f2_norm,
        xlabel = "Male cat body weight", 
        ylabel = "Male cat heart weight",
        title = "domestic male cat body weight  vs heart weight",
        legend = false,
        group = a,
        markersize=10,
        alpha =0.7)

# look at the centroid value
p_center2 = scatter!(μ[1,:], μ[2,:],
        color=:yellow,
        markersize=20,
        alpha =0.7,
        legend = false)
 savefig(p_kmeans,"plotKmeans2.svg")
