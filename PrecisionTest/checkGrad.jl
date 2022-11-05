using Flux
using Flux: train!

# the goal is to keep track of the grad and also check if 
# we can do it in different precisions
# we will make one similar to Knetnlpproblems

# From https://fluxml.ai/Flux.jl/stable/models/basics/

f(x, y) = sum((x .- y).^2);

# no parameters
gradient(f, [2, 1], [2, 0])



x = [2, 1];
y = [2, 0];

gs = gradient(Flux.params(x, y)) do
         f(x, y)
       end

println(gs[x])
println(gs[y])


# models 
W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b

# function loss(x, y)
#   ŷ = predict(x)
#   sum((y .- ŷ).^2)
# end

loss(x, y) = Flux.Losses.mse(predict(x), y);


x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 3


gs = gradient(() -> loss(x, y), Flux.params(W, b))

## Now that we have gradients, we can pull them out and update W to train the model.

W̄ = gs[W]

W .-= 0.1 .* W̄

loss(x, y) # ~ 2.5











################################


# True function 
actual(x) = 4x + 2

# x_train, x_test = hcat(0.0:5.0...), hcat(6.0:10...)

# # in their example they don't have noise but I added noise 
# y_train_true, y_test_true = actual.(x_train), actual.(x_test)
# y_train, y_test = deepcopy(y_train_true),deepcopy(y_test_true)

# println("Type of the data is " , typeof(x_train)," y is ",typeof(y_train))


# x_train, x_test = Float16.(x_train),Float16.(x_test)
# y_train, y_test = Float16.(y_train), Float16.(y_test)

# println("Type of the data is " , typeof(x_train)," y is ",typeof(y_train))

T = Float16

x = rand(T,1,1)
y = actual.(x)
## using linear model 

loss(x, y) = Flux.Losses.mse(x, y);

model = Dense(rand(T,1,1),true) # true is for bias, 1 input and 1 output


#changing the type from Float16 to Float64
θ = Flux.params(model)

println("type of the parameters", typeof.(θ)) 




θ_bar = gradient(() -> loss(model(x), y), θ)


for p in θ
    println(p, θ_bar[p])
end











cfun(x::AbstractArray) = Float64.(x); 
cfun(x) = x; #Noop for stuff which is not arrays (e.g. activation functions)
m64 = Flux.fmap(cfun, model);

println("type of the parameters", typeof.(Flux.params(m64))) 


f16(m) = Flux.paramtype(Float16, m) # similar to https://github.com/FluxML/Flux.jl/blob/d21460060e055dca1837c488005f6b1a8e87fa1b/src/functor.jl#L217

m16 = f16(m64)
println("type of the parameters", typeof.(Flux.params(m16))) 



##########3
model2 = Chain(
  Dense(10 => 5, σ),
  Dense(5 => 2),
  softmax)

y = model2(rand(10)) # => 2-element vector
x = rand(10)

#changing the type from Float16 to Float64
θ2 = Flux.params(model2)

println("type of the parameters", typeof.(θ2)) 




θ2_bar = gradient(() -> loss(model2(x), y), θ2)


for p in θ2
    println(p, θ2_bar[p])
end


#turn it into Float16

f16(m) = Flux.paramtype(Float16, m) # similar to https://github.com/FluxML/Flux.jl/blob/d21460060e055dca1837c488005f6b1a8e87fa1b/src/functor.jl#L217

m16 = f16(model2)


x = Float16.(rand(10))

y = m16(x) # => 2-element vector


#changing the type from Float16 to Float64
θ3 = Flux.params(m16)

println("type of the parameters", typeof.(θ3)) 


θ3_bar = gradient(() -> loss(m16(x), y), θ3)


for p in θ3
    println(p, θ3_bar[p])
end