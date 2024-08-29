import torch

def compute_jacobian(function, args, var_name, eps=1e-5):
    """
    Computes the Jacobian of the function with respect to the variable specified by var_name.

    :param function: The function to differentiate. It should accept keyword arguments.
    :param args: A dictionary of arguments to pass to the function.
    :param var_name: The name of the variable in args with respect to which the derivative is computed.
    :param eps: A small value for finite difference approximation.
    :return: A Jacobian matrix where each row corresponds to the derivative of one output component with respect to each input component.
    """
    
    # Extract the target variable by name and ensure it's a torch tensor
    target = torch.tensor(args[var_name], dtype=torch.float32, requires_grad=True)
    
    # Update the args with the tensor version of the target
    args[var_name] = target
    
    # Evaluate the function to get the shape of the output
    output = function(**args)
    
    # If output is a scalar, convert it to a tensor with a single element
    if output.ndim == 0:
        output = output.unsqueeze(0)
    
    num_outputs = output.numel()
    
    # Initialize the Jacobian matrix with the shape (num_outputs, len(target))
    jacobian = torch.zeros(num_outputs, target.numel())
    
    # Iterate over each element in the output
    for i in range(num_outputs):
        # Clear any existing gradients
        if target.grad is not None:
            target.grad.zero_()
        
        # Compute the gradient of the i-th output component with respect to the target
        output_flat = output.flatten()
        output_flat[i].backward(retain_graph=True)
        
        # Store the gradient in the Jacobian matrix
        jacobian[i] = target.grad.clone().flatten()
    
    # Reshape the Jacobian if the original output is multi-dimensional
    if output.ndim > 1:
        jacobian = jacobian.view(*output.shape, target.numel())
    
    return jacobian

if __name__ == "__main__":
    # Example usage:
    # Define a function that returns a vector output
    def example_func(x, y):
        return torch.tensor([torch.sin(x) + torch.cos(y), x * y])

    # Arguments as a dictionary
    args = {'x': torch.tensor([3.0]), 'y': torch.tensor([2.0])}

    # Compute the Jacobian with respect to 'x'
    jacobian_x = compute_jacobian(example_func, args, var_name='x')
    print("Jacobian with respect to x:\n", jacobian_x)

    # Compute the Jacobian with respect to 'y'
    jacobian_y = compute_jacobian(example_func, args, var_name='y')
    print("Jacobian with respect to y:\n", jacobian_y)
