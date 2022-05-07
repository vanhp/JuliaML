##############################################################
# Binary classification
#
##############################################################

using DelimitedFiles

data = readdlm("/home/vanh/S256GB/Learn/Languages/Julia/ML/classification/NaiveBayes/tennis.csv",
                ',';
                skipstart=1)

x1 = data[:,1]
x2 = data[:,2]
x3 = data[:,3]
x4 = data[:,4]

y = data[:,5]

# identify unique element
unique_x1 = unique(x1)
unique_x2 = unique(x2)
unique_x3 = unique(x3)
unique_x4 = unique(x4)
unique_y = unique(y)

# probability of yes or no output
len_y = length(y)
len_yes = count(x -> x == "yes", y)
len_no = count(x -> x == "no", y)
prob_yes = len_yes / len_y
prob_no = len_no / len_y

# split yes, no into separate matrices
data_yes = data[data[:,5] .== "yes",:]
data_no = data[data[:,5] .== "no",:]

# count features in data_yes
# outlook
count_sunny_yes = count(x -> x == unique_x1[1], data_yes)
count_overcast_yes = count(x -> x == unique_x1[2], data_yes)
count_rainy_yes = count(x -> x == unique_x1[3], data_yes)

# temperature
count_hot_yes = count(x -> x == unique_x2[1], data_yes)
count_mild_yes = count(x -> x == unique_x2[2], data_yes)
count_cool_yes = count(x -> x == unique_x2[3], data_yes)

# humidity
count_high_yes = count(x -> x == unique_x3[1], data_yes)
count_normal_yes = count(x -> x == unique_x3[2], data_yes)

# wind condition
count_nowind_yes = count(x -> x == unique_x4[1], data_yes)
count_windy_yes = count(x -> x == unique_x4[2], data_yes)

# playing
count_notplay_yes = count(x -> x == unique_y[1], data_yes)
count_play_yes = count(x -> x == unique_y[2], data_yes)

# data no
count_sunny_no = count(x -> x == unique_x1[1], data_no)
count_overcast_no = count(x -> x == unique_x1[2], data_no)
count_rainy_no = count(x -> x == unique_x1[3], data_no)

# temperature
count_hot_no = count(x -> x == unique_x2[1], data_no)
count_mild_no = count(x -> x == unique_x2[2], data_no)
count_cool_no = count(x -> x == unique_x2[3], data_no)

# humidity
count_high_no = count(x -> x == unique_x3[1], data_no)
count_normal_no = count(x -> x == unique_x3[2], data_no)

# wind condition
count_nowind_no = count(x -> x == unique_x4[1], data_no)
count_windy_no = count(x -> x == unique_x4[2], data_no)

# playing
count_notplay_no = count(x -> x == unique_y[1], data_no)
count_play_no = count(x -> x == unique_y[2], data_no)

#############################################################
# naive bayes classification
# It's inspired by Bayes theorem
# P(A|B) = P(B|A) P(A)/ P(B)
# It's assume all features are independent of each other (naive)
#
#
# Work by try to understand data then put the data into
# an equation
# apply this to the tennis playing problem
#   P(yes|newX) = P(newX |yes)P(yes) / P(newX)
#   where A is yes
#         B is newX
#
#   It's simple to implement and fast
#   it's output is not the best
#   It's work for simple problem

#############################################################

# prediction #1 newX=["sunny","hot"]
pred_yes_newX = (count_sunny_yes / len_yes) * (count_hot_yes / len_yes) * prob_yes
pred_no_newX = (count_sunny_no / len_no) * (count_hot_no/ len_no) * prob_no

# value don't add up to 100 so normalize value
prob_yes_newX_norm = pred_yes_newX / (pred_yes_newX + pred_no_newX)
prob_no_newX_norm = pred_no_newX / (pred_yes_newX + pred_no_newX)

# prediction #2 with 4 features newX=["sunny","cool","high","windy"]
pred2_yes_newX = (count_sunny_yes / len_yes) * 
                 (count_high_yes / len_yes) *
                 (count_cool_yes / len_yes) * 
                 (count_windy_yes / len_yes) *  prob_yes

pred2_no_newX = (count_sunny_no / len_no) * 
                 (count_high_no / len_no) *
                 (count_cool_no / len_no) * 
                 (count_windy_no / len_no) *  prob_no
# normalize
prob2_yes_newX_norm = pred2_yes_newX / (pred2_yes_newX + pred2_no_newX)
prob2_no_newX_norm = pred2_no_newX / (pred2_yes_newX + pred2_no_newX)