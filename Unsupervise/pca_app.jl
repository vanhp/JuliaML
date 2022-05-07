# application of PCA

using RDatasets
iris = dataset("datasets","iris")

# pre-process data
X = Matrix(iris[:,1:4])'
y = Vector{String}(iris.Species)
species = reshape(unique(iris.Species),(1,3))

# generate PCA model 
# show original dimension,new dimension, info retain
model = fit(PCA,X;maxoutdim=3)

# tranform data
X_transform = transform(model,X)
PC1 = X_transform[1,:]
PC2 = X_transform[2,:]
PC3 = X_transform[3,:]

# plot the tranformed data 
p_transform = scatter(PC1,PC2,PC3,
                        xlabel = "PC1",ylabel= "PC2",zlabel="PC3",
                        title = "Iris dataset for PCA transformation",
                        markersize = 2,
                        group = y,
                        label = species,
                        legend = true)
savefig(p_transform,"iris_PCA_plot.svg")                        
