function allGradients=getMyVarGradientFC(dlX,parameters,h_model,option)
    if nargin>4 && isfield(option,'num_partion');  num_partion = option.num_partion;else; num_partion = 1;end
    if num_partion == 1;dispWaitBar=false;else;dispWaitBar=true;end
    % ######################################## Loop over epochs ##########################################
    if dispWaitBar;WaitMessage = parfor_wait(num_partion,'Waitbar',true,'Title','getMyVarGradientFC');end
    XCell=dataPartion(num_partion,dlX);
    outCell=cell(size(XCell));
    for n_cell =1:num_partion
        dlX_curr=XCell{n_cell};
        dlX_curr=gpuArray(dlX_curr);
        [gradients,~] = dlfeval(@myVarGradients,dlX_curr, parameters,h_model);
        outCell{n_cell}=gather(extractdata(gradients));
    if dispWaitBar;WaitMessage.Send;end
    end
    if dispWaitBar;WaitMessage.Destroy;end
    allGradients=cell2mat(outCell);
end
function XCell=dataPartion(num_partion,Xin)
    XCell=cell(1,num_partion);
    xbars=linspace(0,1,num_partion+1);
    xRand=rand(1,size(Xin,2));
    for n_partion=1:num_partion
        q_curr=xRand >= xbars(n_partion) & xRand < xbars(n_partion+1);
        XCell{n_partion}=Xin(:,q_curr);
    end
end

function [gradients,dlY] = myVarGradients(dlX,parameters,h_trainingModel)
    dlY = h_trainingModel(dlX,parameters);
    gradients = dlgradient(mean(dlY,'all'),dlX);
end

