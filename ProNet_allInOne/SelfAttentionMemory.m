function [H,M]=SelfAttentionMemory(H,M,para)
    % dimension in H M (Chanel,allGrid)
    % for H ---------------------
    Vh=para.Whv*H;% (CN)
    Kh=para.Whk*H;% (CtN)
    Qh=para.Whq*H;% (CtN)    
    Zh=getZ(Vh,Kh,Qh); % (CN)

    % for M ---------------------
    Vm=para.Wmv*M;% (CN)
    Km=para.Wmk*M;% (CtN)
    Qm=para.Wmq*M;% (CtN)
    Zm=getZ(Vm,Km,Qm); % (CN)
    
    Z=para.Wz*[Zh;Zm];
    
    % update M --------
    i=sigmoid(para.Wzi*Z+para.Whi*H+para.bi);
    g=tanh(para.Wzg*Z+para.Whg*H+para.bg);
    M=(1-i).*M+i.*g;
    
    % update H
    o=sigmoid(para.Wzo*Z+para.Who*H+para.bo);
    H=o.*M;
end
function Z=getZ(V,K,Q)
    e=Q'*K;
    alpha=softmax(e,'DataFormat','CB');
    Z=V*alpha;
%     Z=0.*V;
%     for n_grid=1:size(Z,2)
%         e_i=Q(:,n_grid)'*K; % (1,N)
%         alpha_i=softmax(e_i','DataFormat','CB'); % (N,1)
%         Z(:,n_grid)=sum(V*alpha_i);
%     end
end