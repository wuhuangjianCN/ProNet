function [erdBalance_aStart,errAll_predicted_aStart,state]=modelFrame_balance(matData,varName_init,varName_forecast,varName_balance,erdPredict_previous_aStart,...
                        parameters, state,gridWeight_predict)
    % parameters *****************************************************    
    q_zero=gridWeight_predict==0;
    dx=15e3;dy=15e3;         
    pollTarget=strjoin(extractBefore(extractAfter(varName_forecast,'erd_'),'Err'),'');
    [~,ind_erd_init]=ismember(['erd_' pollTarget 'Err'],varName_init);             
    [~,ind_erdPredicted_balance]=ismember(['erdPredicted_' pollTarget 'Err'],varName_balance);
    [~,ind_advErr_balance]=ismember(['adv_' pollTarget 'Err'],varName_balance);
    
    % spin up to update the model state ******************************
    dlX=nan(size(matData.init,1),size(matData.init,2),numel(varName_balance),1,size(matData.init,5),'single');
    [lia,lob]=ismember(varName_balance,varName_init);
    dlX(:,:,lia,:,:)=matData.init(:,:,lob(lia),:,:);
    dlX(:,:,ind_erdPredicted_balance,:,:)=matData.init(:,:,ind_erd_init,:,:);
    dlX=dlarray(dlX);
    for nt=1:size(dlX,5)
        dlX_aStep=dlX(:,:,:,:,nt);
        if canUseGPU;    dlX_aStep=gpuArray(dlX_aStep); end
        [~,state] = model_balance(dlX_aStep,parameters,state);
    end

    % do the forecast ***************************************************
        
    % prepare u v errAllTarget_aStart, and part of dlX_aStart ------
    con_ori_aStart=matData.forecast(:,:,strcmp(varName_forecast,pollTarget),:,:);
    u_aStart=matData.forecast(:,:,strcmp(varName_forecast,'u'),:,:);
    v_aStart=matData.forecast(:,:,strcmp(varName_forecast,'v'),:,:);
    
    % allocate output data and dlX_aStart
    dlX_aStart=nan(size(matData.forecast,1),size(matData.forecast,2),numel(varName_balance),1,size(matData.forecast,5),'single');
    erdBalance_aStart=nan(size(matData.forecast,1),size(matData.forecast,2),1,1,size(matData.forecast,5),'single');
    errAll_predicted_aStart=nan(size(matData.forecast,1),size(matData.forecast,2),1,1,size(matData.forecast,5),'single');
    erdBalance_aStart(:,:,:,:,1)=0;
    
    % prepare part of dlX_aStart ------------------------------------
    [lia,lob]=ismember(varName_balance,varName_forecast);
    dlX_aStart(:,:,lia,:,:)=matData.forecast(:,:,lob(lia),:,:);
    dlX_aStart(:,:,ind_erdPredicted_balance,:,:)=erdPredict_previous_aStart;  
    dlX_aStart=dlarray(dlX_aStart); 
    
    
    errPre=con_ori_aStart(:,:,:,:,1)-matData.forecast(:,:,strcmp(varName_forecast,[pollTarget 'Analysis']),:,1); 
    errAll_predicted_aStart(:,:,:,:,1)=errPre;
    for nt_curr=2:size(errAll_predicted_aStart,5)        
        % transport the get the erdErr_Target,advErr --------------------
        con_ori=con_ori_aStart(:,:,:,:,nt_curr);
        u=u_aStart(:,:,:,:,nt_curr);
        v=v_aStart(:,:,:,:,nt_curr);
        advErr=advDiffAnHour_wuhj(errPre, u, v, dx, dy);   
        
        % prepare part of dlX -------------------------------------------
        dlX=dlX_aStart(:,:,:,:,nt_curr);
        dlX(:,:,ind_advErr_balance,:,:)=advErr;
        % do the neural network part ----------------------------------
        if canUseGPU;    dlX=gpuArray(dlX); end    
        [dlY_predict,state] = model_balance(dlX,parameters,state);
        dlY_predict=gather(extractdata(dlY_predict));
        erdBalance_aStart(:,:,:,:,nt_curr)=dlY_predict;
        %  postprocess for the next iteration --------------------
        erd_aStep=erdPredict_previous_aStart(:,:,:,:,nt_curr);
        erd_aStep(q_zero)=0;
        errAll_aStep=advErr+dlY_predict+erd_aStep;
        errAll_aStep([1 end],:)=0;errAll_aStep(:,[1 end])=0;
        q_exceed=con_ori-errAll_aStep<0;
        errAll_aStep(q_exceed)=-con_ori(q_exceed);                
        errPre=errAll_aStep;        
        errAll_predicted_aStart(:,:,:,:,nt_curr)=errPre;
    end
end
