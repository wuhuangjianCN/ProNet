function lossSeries=calLossSeries(DSinAstart,analysis_aStart,gridWeight_loss)
    tmp=nan(size(DSinAstart,5),1);
    for n_tmp=1:numel(tmp)
        tmp(n_tmp)=mean(abs(DSinAstart(:,:,:,:,n_tmp)-analysis_aStart(:,:,:,:,n_tmp)).*gridWeight_loss,'all');
    end
    lossSeries=numel(gridWeight_loss)/sum(gridWeight_loss,'all').*tmp;
end