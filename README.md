# FluxPl

## Useful links

* https://github.com/FluxML/model-zoo/tree/master/tutorials/dataloader
* Small sample https://github.com/FluxML/model-zoo/blob/master/tutorials/60-minute-blitz/60-minute-blitz.jl


## Todo
- [x] Train simple model on Flux
- [x] Try it in Float32 and Float64, 
- [ ] Float32SR
- [ ] Grad in Flux --Zygote  slow(check if we can overridde)
- [ ] Zygote  is limited such as inplace gradient and can be slow
- [ ] SR gradian
- [x] Float16
- [ ] Move to GPU
- [ ] Data loader
- [ ] MNIST data 
- [ ] Float32SR
- [ ] Float16 
- [ ] GPU MNIST 

## Examples 
- Line of best fit 
  - Dense(1 => 1) also implements the function σ(Wx+b) where W and b are the weights and biases. σ is an activation function (more on activations later).  

## gradient:
- using https://fluxml.ai/Flux.jl/stable/models/basics/
- we will use the param function to allow us to change to gpu,type and more robust..

 have been working on the Flux grad , good news is that we have access to it similar to how knetnlpmodel does 

Flux (Float116 and 64)
```julia
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
```
The way that KnetNlpModels uses this:
```julia
 L = Knet.@diff nlp.chain(nlp.current_training_minibatch)
  vars = Knet.params(nlp.chain)
  for (index, wᵢ) in enumerate(vars)
    nlp.layers_g[index] = Param(Knet.grad(L, wᵢ))
  end
  g .= Vector(vcat_arrays_vector(nlp.layers_g))
  return g
```


## Notes:
- for Optimizer details http://fluxml.ai/Flux.jl/stable/training/optimisers/
- Parameters
  - The dimensions of these model parameters depend on the number of inputs and outputs. Since models can have hundreds of inputs and several layers, it helps to have a function to collect the parameters into the data structure Flux expects:
    ```
    parameters = Flux.params(predict)
    ```
  - The dimensions of these model parameters depend on the number of inputs and outputs. Since models can have hundreds of inputs and several layers, it helps to have a function to collect the parameters into the data structure Flux expects:
    ```
    julia> parameters = Flux.params(predict)
    Params([Float32[0.9066542], Float32[0.0]])
    ```
    These are the parameters Flux will change, one step at a time, to improve predictions. At each step, the contents of this Params object changes too, since it is just a collection of references to the mutable arrays inside the model:
 - To change the type 
    ```
    
    predict = Dense(rand(Float16,1,1),true) # true is for bias

    # The dimensions of these model parameters depend on the number 
    # of inputs and outputs. Since models can have hundreds of inputs and several layers, it helps to have a function 
    # to collect the parameters into the data structure Flux expects:
    parameters = Flux.params(predict)
    println("type of the parameters", parameters) #TODO fix this and print the type

    # The loss function  Here is MSE
    loss(x, y) = Flux.Losses.mse(predict(x), y);
    ```
    - Cuda support for Flux https://github.com/FluxML/NNlibCUDA.jl/blob/master/src/cudnn/conv.jl#L11
    - 

    - useful Have not tried but https://github.com/FluxML/NNlibCUDA.jl/blob/master/src/cudnn/conv.jl#L11 looks promising. For Flux you’d probably want an f16 like https://github.com/FluxML/Flux.jl/blob/d21460060e055dca1837c488005f6b1a8e87fa1b/src/functor.jl#L217 to change all types
  - To change the type 
      ```
      predict = Dense(rand(Float16,1,1),true) # true is for bias, 1 input and 1 output

      #changing the type from Float16 to Float64
        parameters = Flux.params(predict)

        println("type of the parameters", typeof.(parameters)) 

        cfun(x::AbstractArray) = Float64.(x); 
        cfun(x) = x; #Noop for stuff which is not arrays (e.g. activation functions)
        m64 = Flux.fmap(cfun, predict);

        println("type of the parameters", typeof.(Flux.params(m64))) 


        f16(m) = Flux.paramtype(Float16, m) # similar to https://github.com/FluxML/Flux.jl/blob/d21460060e055dca1837c488005f6b1a8e87fa1b/src/functor.jl#L217

        m16 = f16(m64)
        println("type of the parameters", typeof.(Flux.params(m16)))```