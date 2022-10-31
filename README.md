# FluxPl

## Useful links

* https://github.com/FluxML/model-zoo/tree/master/tutorials/dataloader
* Small sample https://github.com/FluxML/model-zoo/blob/master/tutorials/60-minute-blitz/60-minute-blitz.jl


## Todo
- [x] Train simple model on Flux
- [x] Try it in Float32 and Float64, 
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