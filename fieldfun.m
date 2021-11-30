function out=fieldfun(funHandle,in1,in2)
    subFields=fields(in1);
    if nargin==2
        out=dealWithALevel(subFields,funHandle,in1);
    else
        out=dealWithALevel(subFields,funHandle,in1,in2);
    end
end
function out=dealWithALevel(FieldsAll,funHandle,in1,in2)
    for n_field=1:numel(FieldsAll)
        cur_field=FieldsAll{n_field};
        eval(['q_isstruct=isstruct(in1.' cur_field ');'])
        if ~q_isstruct
            % dealWithAField
            if nargin==3
                eval(['out.' cur_field '=dealWithAField(funHandle,in1.' cur_field ');']);
            else
                eval(['out.' cur_field '=dealWithAField(funHandle,in1.' cur_field ',in2.' cur_field ');']);
            end
        else
            % explor subFields
            eval(['subFields=fields(in1.' cur_field ');'])
            if nargin==3
                eval(['out.' cur_field '=dealWithALevel(subFields,funHandle,in1.' cur_field ');']);
            else
                eval(['out.' cur_field '=dealWithALevel(subFields,funHandle,in1.' cur_field ',in2.' cur_field ');']);
            end
        end
    end
end
function out=dealWithAField(funHandle,in1,in2)
    if nargin==2
        out=funHandle(in1);
    else
        out=funHandle(in1,in2);
    end
end