function [bErr_Predicted,ErrAll_Predicted] = modelAstart_Balance(xVarName,dlX_init,dlX_fore,ErrAll_init,u_aStart,v_aStart,con_ori_aStart,q_grid2predict,para,state)
    dx=15e3;dy=15e3;
% init ------------------------------------------------------
    dlX_init=dlarray(dlX_init);
    if canUseGPU;    dlX_init=gpuArray(dlX_init); end
    [~,state] = model_ST(dlX_init,para,state);clear dlX_init
        
% gradient durinig forecast ------------------------------
    q_advErr=contains(xVarName,'advErr');
    q_erdErr=contains(xVarName,'erdErrPredicted');
    erdErrPredicted=dlX_fore(:,:,q_erdErr,:,:);
    errPre=ErrAll_init;
    bErr_Predicted=nan(size(dlX_fore(:,:,1,:,:)),'single');   bErr_Predicted(:,:,:,:,1) =0;
    ErrAll_Predicted=nan(size(bErr_Predicted),'single');   ErrAll_Predicted(:,:,:,:,1) =ErrAll_init;
    for nLead=2:size(dlX_fore,5)
        % transport the get the erdErr_Target,advErr --------------------
        con_ori=con_ori_aStart(:,:,:,:,nLead);
        u=u_aStart(:,:,:,:,nLead);
        v=v_aStart(:,:,:,:,nLead);
        advErr=advDiffAnHour_wuhj(errPre, u, v, dx, dy);
        % prepare dlX -------------------------------------------------
        dlX_curr=dlX_fore(:,:,:,:,nLead);
        dlX_curr(:,:,q_advErr,:,:)=advErr;
        % do the neural network part ----------------------------------
        dlX_curr=dlarray(dlX_curr); 
        if canUseGPU;    dlX_curr=gpuArray(dlX_curr);end
        [dlY_predict_curr,state] = model_ST(dlX_curr,para,state);
        dlY_predict_curr=gather(extractdata(dlY_predict_curr));
        
        % postprocess for the next iteration --------------------
        % add limit to ensure the nonagative of predicted concentration:
        erdb_aSetp=erdErrPredicted(:,:,:,:,nLead)+dlY_predict_curr;
        erdb_aSetp(~q_grid2predict)=0;
        errAll_aStep=advErr+erdb_aSetp;
        errAll_aStep([1 end],:)=0;errAll_aStep(:,[1 end])=0;
        q_exceed=con_ori-errAll_aStep<0;
        errAll_aStep(q_exceed)=-con_ori(q_exceed);                
        errPre=errAll_aStep;
        
        ErrAll_Predicted(:,:,:,:,nLead)=errPre;
        bErr_Predicted(:,:,:,:,nLead)=dlY_predict_curr;
    end
end