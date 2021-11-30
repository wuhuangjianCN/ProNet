function [dlY,state] = model_balance(dlX,parameters,state)
    % dlX is a 2-d image sequence: h-by-w-by-c-by-N-by-S, 
    % where h, w, and c correspond to the height, width, and number of channels of the images respectively, N is the number of observations, and S is the sequence length.
    dlY=0.*dlX(:,:,1,:,:);
    for n_obs=1:size(dlX,4)
        for index_sequence=1:size(dlX,5)
            dlX_aStep=dlX(:,:,:,n_obs,index_sequence);
            % conv_b1
            dlX_aStep=dlconv(dlX_aStep,parameters.conv_b1.Weights,parameters.conv_b1.Bias,'Padding','same','DataFormat','SSCBT');
            % convLSTM_b1
            [dlX_aStep,state.convLSTM_b1]=ConvLSTM(dlX_aStep,state.convLSTM_b1,parameters.convLSTM_b1);
            % convLSTM_b2
            [~,state.convLSTM_b2]=ConvLSTM(dlX_aStep,state.convLSTM_b2,parameters.convLSTM_b2);
            % conv_b2
            dlX_aStep=dlconv(cat(3,state.convLSTM_b1.h,state.convLSTM_b2.h),parameters.conv_b2.Weights,parameters.conv_b2.Bias,'Padding','same','DataFormat','SSCBT');  
            dlY(:,:,:,n_obs,index_sequence)=dlX_aStep;
        end
    end
end
function [dlY,state]=ConvLSTM(dlX,state,paraConvLSTM)
    myconv=@(x,weight) dlconv(x,weight,0,'Padding',"same",'DataFormat','SSCBT');
    dlY=myconv(dlX,paraConvLSTM.wxo)+paraConvLSTM.bo;
    tmp_f=myconv(dlX,paraConvLSTM.wxf)+paraConvLSTM.bf;
    tmp_i=myconv(dlX,paraConvLSTM.wxi)+paraConvLSTM.bi;
    tmp_i = sigmoid(tmp_i+...
        myconv(state.h,paraConvLSTM.whi)+paraConvLSTM.wci.*state.c);
    tmp_f = sigmoid(tmp_f+...
        myconv(state.h,paraConvLSTM.whf)+paraConvLSTM.wcf.*state.c);
    state.c=tmp_f.*state.c+...
        tmp_i.*tanh(myconv(dlX,paraConvLSTM.wxc)+myconv(state.h,paraConvLSTM.whc)+paraConvLSTM.bc);
    dlY = sigmoid(dlY+...
        myconv(state.h,paraConvLSTM.who)+paraConvLSTM.wco.*state.c);
    state.h = dlY.*tanh(state.c);
end