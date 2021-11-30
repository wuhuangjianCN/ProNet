function [dlY,H,C]=ConvLSTM(dlX,H,C,paraConvLSTM)
    myconv=@(x,weight) dlconv(x,weight,0,'Padding',"same",'DataFormat','SSCBT');
    dlY=myconv(dlX,paraConvLSTM.wxo)+paraConvLSTM.bo;
    tmp_f=myconv(dlX,paraConvLSTM.wxf)+paraConvLSTM.bf;
    tmp_i=myconv(dlX,paraConvLSTM.wxi)+paraConvLSTM.bi;
    tmp_i = sigmoid(tmp_i+...
        myconv(H,paraConvLSTM.whi)+paraConvLSTM.wci.*C);
    tmp_f = sigmoid(tmp_f+...
        myconv(H,paraConvLSTM.whf)+paraConvLSTM.wcf.*C);
    C=tmp_f.*C+...
        tmp_i.*tanh(myconv(dlX,paraConvLSTM.wxc)+myconv(H,paraConvLSTM.whc)+paraConvLSTM.bc);
    dlY = sigmoid(dlY+...
        myconv(H,paraConvLSTM.who)+paraConvLSTM.wco.*C);
    H = dlY.*tanh(C);
end