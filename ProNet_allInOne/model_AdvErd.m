function [DSaStart,errAll_predicted_aStart]=model_AdvErd(u_aStart,v_aStart,con_ori_aStart,erdErr_pred_aStart,errAll_init,gridWeight_predict)
    % parameters *****************************************************                  
    dx=15e3;dy=15e3;
    q_zero=gridWeight_predict==0;
    % prepare part of dlX_aStart ------------------------------------
    sizeOriCon=size(con_ori_aStart);
    DSaStart=nan(sizeOriCon,'single');
    errAll_predicted_aStart=nan(sizeOriCon,'single');

    errPre=errAll_init;
    errAll_predicted_aStart(:,:,:,:,1)=errPre;
    DSaStart(:,:,:,:,1)=con_ori_aStart(:,:,:,:,1)-errAll_init;
    for nt_curr=2:size(erdErr_pred_aStart,5)
        % transport the get the erdErr_Target,advErr --------------------
        con_ori=con_ori_aStart(:,:,:,:,nt_curr);
        u=u_aStart(:,:,:,:,nt_curr);
        v=v_aStart(:,:,:,:,nt_curr);
        advErr=advDiffAnHour_wuhj(errPre, u, v, dx, dy);   
        
        %dlY_predict=extractdata(dlY_predict);
        erd_aStep=erdErr_pred_aStart(:,:,:,:,nt_curr);
        erd_aStep(q_zero)=0;
        errAll_aStep=advErr+erd_aStep;
        % postprocess for the next iteration --------------------
        errAll_aStep([1 end],:)=0;errAll_aStep(:,[1 end])=0;
        q_exceed=con_ori-errAll_aStep<0;
        errAll_aStep(q_exceed)=-con_ori(q_exceed);                
        errPre=errAll_aStep; 
        % add boundary limit
        errAll_predicted_aStart(:,:,:,:,nt_curr)=errPre;
        DSaStart(:,:,:,:,nt_curr)=con_ori-errPre;
    end
end
