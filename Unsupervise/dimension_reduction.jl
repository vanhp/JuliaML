# Dimensionality Reduction

# A transformation of data from high-dimension into lower-dimension while maintain meaningful
# property of the original data.

# In ML this high-dimension is reprented by features of dataset. The more features in the dataset 
# it's mean the more dimension. If many of these feature can be removed without affecting the property 
# of the dataset then it's a very useful process indeed. These would simplify the understandinng of dataset 
# and also reduce the use of computational resource. It's use to clean up the data in many application in
# data compression,speech recognition, signal processing,neuroinfomatic,bioinfomatics, before analysis, 
# and computation processing. 

# PCA (principal component analysis) is one of the field in dimensional reduction. PCA is used to 
# compress instead of remove feature of the high dimension data e.g. Iris flower dataset 
# which has 4 features down to 3 but still have all the important property of the original data
# so it can be used to plot in 3D. 

# Concept 
using MultivariateStats, Plots
using Random

# use seed so it can be reproduced the result
Random.seed!(1)
f0 = collect(0.50:0.50:50)
f1 = f0 .+ rand(100)
f2 = f0 .+ rand(100)

# plotlyjs is buggie but has interactive 3D 
plotlyjs(size = (480,480))
p_random = scatter(f1,f2,
                    xlabel= "Feature 1",
                    ylabel="Feature 2",
                    title="Random data",
                    legend=false
                )

# pre-processing data using PCA
# PCA require data to be in row format instead of column 
# that the row reprented feature of data 
# the data should be transpose into horizontal style
X = [f1 f2]'

# PCA only take input and produce no output
model = fit(PCA,X;maxoutdim=1)