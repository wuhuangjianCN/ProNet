function XDATA=myCSDATAFun(meshObjIn,xVarName_cs,varName2portion,normalizer_CS,leads2extract)
%     xVarName_cs={ 'erd_PM25' 'ANH4' 'BC'  'ANO3' 'ASO4' 'OC'  'SOA'  'temp' 'rh'};
%     varName2portion={'ANH4' 'BC'  'ANO3' 'ASO4' 'OC'  'SOA' };
    if nargin<5 ;leads2extract=meshObjIn.leadTimes;end

    % dlX_e 
    XDATA=meshObjIn.get([],leads2extract,xVarName_cs,[]);
    q2portion=ismember(xVarName_cs,varName2portion);
    if sum(q2portion)>0
        dataDenominator=meshObjIn.get([],leads2extract,{'PM25'},[]);
        XDATA(:,:,q2portion,:,:)=XDATA(:,:,q2portion,:,:)./repmat(dataDenominator,1,1,sum(q2portion),1,1);
    end
    if nargin>3 && ~isempty(normalizer_CS)
        XDATA=normalizer_CS.normData(XDATA,xVarName_cs);
    end
end