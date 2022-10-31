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

predict = Dense(rand(Float16,1,1),true) # true is for bias

# The dimensions of these model parameters depend on the number 
# of inputs and outputs. Since models can have hundreds of inputs and several layers, it helps to have a function 
# to collect the parameters into the data structure Flux expects:
parameters = Flux.params(predict)
println("type of the parameters", parameters) #TODO fix this and print the type

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