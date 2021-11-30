function dlY_Predicted = modelAstart_ST(xVarName,dlX_init,dlX,para,state)
% init ---------------------
    dlX_init=dlarray(dlX_init);
    %if canUseGPU;    dlX_init=gpuArray(dlX_init);end
    [~,state] = model_ST(dlX_init,para,state);
% forecast --------------
    q_before=contains(xVarName,'Before');
    dlY_Predicted=0.*dlX(:,:,1,:,:);
    for nLead=2:size(dlX,5)
        % dlY_Target ------------------
        % prepare part of dlX -------------------------------------------
        dlX_e=dlX(:,:,:,:,nLead);
        if nLead>2;dlX_e(:,:,q_before,:,:)=dlY_predict_current;end
        
        dlX_e=dlarray(dlX_e);
        % do the neural network part ----------------------------------
        if canUseGPU;    dlX_e=gpuArray(dlX_e);end
        [dlY_predict_current,state] = model_ST(dlX_e,para,state);
        dlY_Predicted(:,:,:,:,nLead)=gather(extractdata(dlY_predict_current));
    end
end
