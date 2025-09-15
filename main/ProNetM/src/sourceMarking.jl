
export markInfoStr,markERD
mutable struct markInfoStr
    idSets::AbstractArray
    tSets::AbstractArray
    Times::AbstractArray
    idMat::AbstractArray
end

function markERD(erdIn,iT,iS,markInfo)
    # skip if iT or iS not in the candidate list
    if !( (0 < iT <= length(markInfo.tSets)) && (0 < iS <= length(markInfo.idSets)) )
        println("markERD: iT=$iT, iS=$iS, out of range")
        return 0 .* erdIn
    end
    # prepare q_validT--------
    q_validT=falses(size(erdIn,3))
    q_validT[markInfo.tSets[iT]] .= true
    # prepare q_validGrid--------
    q_validGrid=falses(size(markInfo.idMat))
    idValid_all=markInfo.idSets[iS]
    q_validGrid= in.(markInfo.idMat,[idValid_all])
    # mark erdIn to erdOut--------
    erdOut=zeros(size(erdIn))
    for (it,isValidT) in enumerate(q_validT)
        if isValidT
            erdOut[:,:,it] = erdIn[:,:,it]
            erdCurr=view(erdOut, :, :, it)
            erdCurr[.!q_validGrid] .= 0
        end
    end
    return erdOut
end