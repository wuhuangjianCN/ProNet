function [gradients,state,loss,dlY] = modelGradients_balance(dlX,dlY_target,parameters,state,weight)

    [dlY,state] = model_balance(dlX,parameters,state);
    loss = partialLoss(dlY,dlY_target,weight);
    gradients = dlgradient(loss,parameters);
end