function [dlY,state]=SaConvLSTM(dlX,state,paraSaConvLSTM,q_gridRef)
    [dlY,state.H,state.C]=ConvLSTM(dlX,state.H,state.C,paraSaConvLSTM.ConvLSTM);
    
%     dlX_aStep=dlconv(cat(3,H1,H2),parameters.conv_2.Weights,parameters.conv_2.Bias,'Padding','same','DataFormat','SSCBT'); 
%     H=cat(3,paraSaConvLSTM.Wh*H,
    % attention
%     sizeH_full=size(state.H);
%     H=avgpool(dlarray(state.H,'SSB'),[5 5],'stride',3);
%     sizeH_small=size(H);
%     H=reshape(H,prod(sizeH_small(1:2)),sizeH_small(3))';
%     if nargin>3
%         [H(:,q_gridRef),state.M]=SelfAttentionMemory(H(:,q_gridRef),state.M,paraSaConvLSTM.SAM);
%         H(:,~q_gridRef)=0;
%     else
%         [H,state.M]=SelfAttentionMemory(H,state.M,paraSaConvLSTM.SAM);
%     end
%     H1=reshape(H',sizeH_small);
%     H = dltranspconv(H1,paraSaConvLSTM.tconv.Weights,paraSaConvLSTM.tconv.Bias,'Stride',3,'Cropping',2,'DataFormat','SSB');
%  %   H = dltranspconv(H1,paraSaConvLSTM.tconv.Weights,paraSaConvLSTM.tconv.Bias,'Stride',3,'Cropping',-1,'DataFormat','SSB');
%     state.H=H;
end