# """
# Line of best fit 
# http://fluxml.ai/Flux.jl/stable/models/overview/


# This example will predict the output of the function 4x + 2.

# """

using Flux
# using Distributions # for the noise
using Flux: train!



# True function 
actual(x) = 4x + 2

x_train, x_test = hcat(0.0:5.0...), hcat(6.0:10...)

# in their example they don't have noise but I added noise 
y_train_true, y_test_true = actual.(x_train), actual.(x_test)
y_train, y_test = deepcopy(y_train_true),deepcopy(y_test_true)

println("Type of the data is " , typeof(x_train)," y is ",typeof(y_train))


x_train, x_test = Float16.(x_train),Float16.(x_test)
y_train, y_test = Float16.(y_train), Float16.(y_test)

println("Type of the data is " , typeof(x_train)," y is ",typeof(y_train))


# model 
## 

# I can name it predict or keep it model
# predict = Dense(1 => 1)

predict = Dense(rand(Float16,1,1),true) # true is for bias, 1 input and 1 output

# The dimensions of these model parameters depend on the number 
# of inputs and outputs. Since models can have hundreds of inputs and several layers, it helps to have a function 
# to collect the parameters into the data structure Flux expects:



#### Type change
# idea https://discourse.julialang.org/t/a-possible-way-to-improve-training-in-flux/44629/6


# julia> m = Chain(Dense(3,4), Dense(4,5));

# julia> typeof.(params(m))
# 4-element Array{DataType,1}:
#  Array{Float32,2}
#  Array{Float32,1}
#  Array{Float32,2}
#  Array{Float32,1}

# julia> cfun(x::AbstractArray) = Float64.(x); 

# julia> cfun(x) = x; Noop for stuff which is not arrays (e.g. activation functions)

# julia> m64 = Flux.fmap(cfun, m)
# Chain(Dense(3, 4), Dense(4, 5))

# julia> typeof.(params(m64))
# 4-element Array{DataType,1}:
#  Array{Float64,2}
#  Array{Float64,1}
#  Array{Float64,2}
#  Array{Float64,1}


#changing the type from Float16 to Float64
parameters = Flux.params(predict)

println("type of the parameters", typeof.(parameters)) 

cfun(x::AbstractArray) = Float64.(x); 
cfun(x) = x; #Noop for stuff which is not arrays (e.g. activation functions)
m64 = Flux.fmap(cfun, predict);

println("type of the parameters", typeof.(Flux.params(m64))) 


f16(m) = Flux.paramtype(Float16, m) # similar to https://github.com/FluxML/Flux.jl/blob/d21460060e055dca1837c488005f6b1a8e87fa1b/src/functor.jl#L217

m16 = f16(m64)
println("type of the parameters", typeof.(Flux.params(m16))) 




# The loss function  Here is MSE
loss(x, y) = Flux.Losses.mse(predict(x), y);

# Choosing the Optimiser
opt = Descent()

# data 
data = [(x_train, y_train)]



print("LOSS: ")
println(loss(x_train, y_train))
# training 
for epoch in 1:200
    train!(loss, parameters, data, opt)
    if epoch%10==0
        println("Epoch ", epoch)
    end
end

print("LOSS: ")
println(loss(x_train, y_train))
println(predict(x_test),"\n",y_test)