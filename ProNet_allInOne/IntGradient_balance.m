function  [gradients,lossAccumulated] = IntGradient_balance(matData,varName_init,varName_forecast,varName_balance,erdPredict_previous_aStart,...
                        parameters, state,gridWeight_predict,lossRadius)
    % parameters *****************************************************
    lead2startGradient=-24*2;
    leads_init=(1-size(matData.init,5)):0;
    leads_forecast=0:(size(matData.forecast,5)-1);
    dx=15e3;dy=15e3;
    q_zero=gridWeight_predict==0;
    q_erd_init= startsWith(varName_init,'erd_') & endsWith(varName_init,'Err');
    q_advErr_balance= startsWith(varName_balance,'adv_') & endsWith(varName_balance,'Err');
    q_erdPredicted_balance= startsWith(varName_balance,'erdPredicted_') & endsWith(varName_balance,'Err');
    pollTarget=strjoin(extractBefore(varName_forecast,'Analysis'),'');
    
    % spin up to update the model state ******************************
    dlX=nan(size(matData.init,1),size(matData.init,2),numel(varName_balance),1,size(matData.init,5),'single');
    [lia,lob]=ismember(varName_balance,varName_init);
    dlX(:,:,lia,:,:)=matData.init(:,:,lob(lia),:,:);
    dlX(:,:,q_erdPredicted_balance,:,:)=matData.init(:,:,q_erd_init,:,:);
    dlX=dlarray(dlX);
    % spin up for the state --------------------------------
    for lead=min(leads_init):(lead2startGradient-1)
        li_step=leads_init==lead;
        dlX_aStep=dlX(:,:,:,:,li_step);
        if canUseGPU;    dlX_aStep=gpuArray(dlX_aStep); end
        [~,state] = model_balance(dlX_aStep,parameters,state);
    end
    % calculate the gradients of some intial steps----------
    
    lossWeight=matData.init(:,:,strcmp(varName_init,[pollTarget 'minValidDist']),:,:);
    lossWeight=double(lossWeight<lossRadius);    lossWeight(:,:,:,:,2:end)=min(lossWeight(:,:,:,:,1:(end-1)),lossWeight(:,:,:,:,2:end));
    
    %gridWeight_loss=lossWeight;
    nGradient=0;gradients=[];lossAccumulated=[];
    for lead=lead2startGradient:0
        li_step=leads_init==lead;
        lossWeight_curr=lossWeight(:,:,:,:,li_step);
        dlX_aStep=dlX(:,:,:,:,li_step);
        dlY_Target=0.*dlX_aStep(:,:,1,:,:);
        if canUseGPU;    dlX_aStep=gpuArray(dlX_aStep);dlY_Target=gpuArray(dlY_Target); end
        [gradients,state,lossAccumulated,~,nGradient]=gradient_aStep(dlX_aStep,dlY_Target, parameters, state,lossWeight_curr,nGradient,gradients,lossAccumulated);
    end

    % do the forecast ***************************************************
    
    % prepare u v errAllTarget_aStart, and part of dlX_aStart ------
    con_ori_aStart=matData.forecast(:,:,strcmp(varName_forecast,pollTarget),:,:);
    u_aStart=matData.forecast(:,:,strcmp(varName_forecast,'u'),:,:);
    v_aStart=matData.forecast(:,:,strcmp(varName_forecast,'v'),:,:);
    errAllTarget_aStart=con_ori_aStart-matData.forecast(:,:,strcmp(varName_forecast,[pollTarget 'Analysis']),:,:);
    
    % prepare part of dlX_aStart ------------------------------------
    dlX_aStart=nan(size(matData.forecast,1),size(matData.forecast,2),numel(varName_balance),1,size(matData.forecast,5),'single');
    [lia,lob]=ismember(varName_balance,varName_forecast);
    dlX_aStart(:,:,lia,:,:)=matData.forecast(:,:,lob(lia),:,:);
    dlX_aStart(:,:,q_erdPredicted_balance,:,:)=erdPredict_previous_aStart;
    dlX_aStart=dlarray(dlX_aStart); 
    
    lossWeight=matData.forecast(:,:,strcmp(varName_forecast,[pollTarget 'minValidDist']),:,:);
    lossWeight=double(lossWeight<lossRadius);    lossWeight(:,:,:,:,2:end)=min(lossWeight(:,:,:,:,1:(end-1)),lossWeight(:,:,:,:,2:end));
    % intitialization for the forecast
    errPre=con_ori_aStart(:,:,:,:,1)-matData.forecast(:,:,strcmp(varName_forecast,[pollTarget 'Analysis']),:,1);
    nts2forecast=1:max(leads_forecast);lossAccumulated=0;
    for nt_curr=nts2forecast
        % transport the get the erdErr_Target,advErr --------------------
        con_ori=con_ori_aStart(:,:,:,:,nt_curr);
        u=u_aStart(:,:,:,:,nt_curr);
        v=v_aStart(:,:,:,:,nt_curr);
        advErr=advDiffAnHour_wuhj(errPre, u, v, dx, dy);    
        
        
        % prepare dlY_Target -------------------------------------------
        erdErr_Target=errAllTarget_aStart(:,:,:,:,nt_curr)-advErr-erdPredict_previous_aStart(:,:,:,:,nt_curr);    
        dlY_Target=dlarray(erdErr_Target); 
        
        % prepare dlX -------------------------------------------------
        dlX=dlX_aStart(:,:,:,:,nt_curr);
        dlX(:,:,q_advErr_balance,:,:)=advErr;
        lossWeight_curr=lossWeight(:,:,:,:,nt_curr);
        % do the neural network part ----------------------------------
        if canUseGPU;    dlX=gpuArray(dlX);   dlY_Target=gpuArray(dlY_Target);   end
        [gradients,state,lossAccumulated,dlY_predict,nGradient]=gradient_aStep(dlX,dlY_Target, parameters, state,lossWeight_curr,nGradient,gradients,lossAccumulated);
        dlY_predict=gather(extractdata(dlY_predict));
        
        % postprocess for the next iteration --------------------
        % add limit to ensure the nonagative of predicted concentration:
        erd_aSetp=erdPredict_previous_aStart(:,:,:,:,nt_curr);
        erd_aSetp(q_zero)=0;
        errAll_aStep=advErr+erd_aSetp+dlY_predict;
        errAll_aStep([1 end],:)=0;errAll_aStep(:,[1 end])=0;
        q_exceed=con_ori-errAll_aStep<0;
        errAll_aStep(q_exceed)=-con_ori(q_exceed);                
        errPre=errAll_aStep;        
    end
	lossAccumulated=lossAccumulated/numel(nts2forecast);
    gradients=fieldfun(@(x) x/nGradient,gradients);
end
function [gradients,state,lossAccumulated,dlY_predict,nGradient]=gradient_aStep(dlX,dlY_Target, parameters, state,gridWeight_loss,nGradient,gradients,lossAccumulated)
    if 0==sum(gridWeight_loss)
        [~,state] = model_balance(dlX,parameters,state);
        dlY_predict=0.*dlX(:,:,1,:,:);
    else
        [tmp_gradients,state,tmp_loss,dlY_predict] = dlfeval(@modelGradients_balance,dlX,dlY_Target, parameters, state,gridWeight_loss);
        if isnan(tmp_loss);error('nan loss found');end
        if isempty(gradients)
            gradients=tmp_gradients;        lossAccumulated=tmp_loss;
        else
            lossAccumulated=tmp_loss+lossAccumulated;
            gradients=fieldfun(@(x,y) x+y,gradients,tmp_gradients);
        end
        nGradient=nGradient+1;
    end
end
