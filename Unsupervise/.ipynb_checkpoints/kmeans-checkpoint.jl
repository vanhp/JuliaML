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
# transpose into 2 row and 1 column
X = [f1 f2]'
# 2 groups
k = 5

# k-means is self iterate must set max loop to prevent infinite
iter = 100
Random.seed!(1)
results = kmeans(X,k;maxiter=iter,display=:iter)
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

###############################
# normalization (scale feature)
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

p_center2 = scatter!(μ[1,:], μ[2,:],
        color=:yellow,
        markersize=20,
        alpha =0.7,
        legend = false)
 savefig(p_kmeans,"plotKmeans2.svg")
