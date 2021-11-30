function [dlY,state_st,dlY_st] = model_STCS(dlX_st,dlX_cs,para_st,state_st,para_cs)

    % for model_ed
    [dlY_st,state_st] = model_ST(dlX_st,para_st,state_st);
    
    % for model_r
    sizeOri=size(dlX_cs);sizeOut=sizeOri;sizeOut(3)=1;
    dlX_cs=permute(dlX_cs,[3 1 2 4 5]);%'CSSBT'
    dlX_cs=reshape(dlX_cs,sizeOri(3),sizeOri(1)*sizeOri(2),[]);
    dlY_cs = model_CS(dlX_cs,para_cs);
    dlY_cs=reshape(dlY_cs,sizeOut);
    
    dlY=dlY_st+dlY_cs;
end
