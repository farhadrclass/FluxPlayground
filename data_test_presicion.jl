using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON
using CUDA

# We set default values for the arguments for the function `train`:

Base.@kwdef mutable struct Args
    η = 3e-4             ## learning rate
    λ = 0                ## L2 regularizer param, implemented as weight decay
    batchsize = 128      ## batch size
    epochs = 10          ## number of epochs
    seed = 0             ## set seed > 0 for reproducibility
    use_cuda = true      ## if true use cuda (if available)
    infotime = 1 	     ## report every `infotime` epochs
    checktime = 5        ## Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      ## log training with tensorboard
    savepath = "runs/"   ## results path
end

# ## Data

# We create the function `get_data` to load the MNIST train and test data from [MLDatasets](https://github.com/JuliaML/MLDatasets.jl) and reshape them so that they are in the shape that Flux expects. 

function get_data(args,T=Float32)

    xtrain, ytrain = MLDatasets.MNIST(Tx=T, split=:train)[:]
    xtest, ytest = MLDatasets.MNIST(Tx=T,split=:test)[:]

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=128, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=128)
    
    return train_loader, test_loader
end




function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end


function myModel(; imgsize=(28,28,1), nclasses=10,T=Float32) 
    
    # return Chain(
    #     Dense(rand(Float16,prod(imgsize), 32),true, relu),
    #     Dense(rand(Float16,32, nclasses),true)
    #     )
        return Chain( Dense(prod(imgsize), 32, relu),
                  Dense(32, nclasses))
        
        # Dense(rand(Float16, 784,500),true,relu),
        # Dense(rand(Float16, 500,200),true,relu),
        # Dense(rand(Float16, 200, 84), true,relu), 
        # Dense(rand(Float16,84, nclasses),true)
        # )
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        print(size(x))
        ŷ = model(x)
        println("Type of y after model  is ", typeof(ŷ ))
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

# ## Utility functions
# We need a couple of functions to obtain the total number of the model's parameters. Also, we create a function to round numbers to four digits.

num_params(model) = sum(length, Flux.params(model)) 
round4(x) = round(x, digits=4)


#now testing the type
function myTest()
    use_cuda =  CUDA.functional()
    
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    ## DATA
    train_loader, test_loader = get_data(25,Float16)
    # @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    # model = LeNet5() |> device
    model = myModel(T=Float16) |> device
    @info "The model: $(num_params(model)) trainable params"    
    
    ps = Flux.params(model)  


    train = eval_loss_accuracy(train_loader, model, device)


end

myTest()