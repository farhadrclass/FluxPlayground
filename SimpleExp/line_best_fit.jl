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

#noise added
map!(vi->vi+(rand(1:100)/1000),y_train,y_train) 
map!(vi->vi+(rand(1:100)/1000),y_test,y_test) 


# model 
## 
model = Dense(1 => 1)
println( model.weight)
println(model.bias)

# I can name it predict or keep it model
predict = Dense(1 => 1)

# The loss function  Here is MSE
loss(x, y) = Flux.Losses.mse(predict(x), y);

# Choosing the Optimiser
opt = Descent()

# data 
data = [(x_train, y_train)]

# The dimensions of these model parameters depend on the number 
# of inputs and outputs. Since models can have hundreds of inputs and several layers, it helps to have a function 
# to collect the parameters into the data structure Flux expects:
parameters = Flux.params(predict)

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
predict(x_test,"\n",y_test)