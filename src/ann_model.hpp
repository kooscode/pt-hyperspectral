#include <torch/torch.h>

struct ann_model : torch::nn::Module 
{
    // Constructor to register all layers.
    ann_model() 
    {
        // construct and register your layers
        fc1 = register_module("fc1",torch::nn::Linear(7, 128));
        fc2 = register_module("fc2",torch::nn::Linear(128, 32));
        fc3 = register_module("fc3",torch::nn::Linear(32, 2));
    }

    // the forward operation (how data will flow from layer to layer)
    torch::Tensor forward(torch::Tensor x)
    {
        //Forward pass..
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::sigmoid(fc3->forward(x));
 
        return x;
    }

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};
