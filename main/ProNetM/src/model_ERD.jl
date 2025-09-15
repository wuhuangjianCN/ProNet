export getEmisModel,getDeposModel,getReactModel,EmisModelCell,DeposModelCell,ReactModelCell

# ======================= Emission Model ============================
mutable struct EmisModelCell
    encSpaceInit::Chain
    encSpaceFore::Chain
    hidMap_inToHid::Conv
    hidMap_foreToHid::Conv
    hid::Chain
    tMapper::Conv
    decSpace::Chain
end
function (m::EmisModelCell)(x_STinit,x_STfore)    
    W_init,H_init,C_init,T_init,B= size(x_STinit)
    W_fore,H_fore,C_fore,T_fore,B= size(x_STfore)
    # encoder ------------------    
    # encode init
    x_STinit,inT,B=toSpaceForm(x_STinit)
    x_STinit=m.encSpaceInit[1](x_STinit)
    skip_init,H_,C_space=spaceToTimeForm(x_STinit,inT,B)
    x_STinit=m.encSpaceInit[2:end](x_STinit)
    x_STinit,_,_=spaceToHidForm(x_STinit,T_init,B)
    x_STinit=m.hidMap_inToHid(x_STinit)
    # encode fore
    x_STfore,outT,B=toSpaceForm(x_STfore)
    x_STfore=m.encSpaceFore[1](x_STfore)
    skip_fore=copy(x_STfore)
    x_STfore=m.encSpaceFore[2:end](x_STfore)
    x_STfore,_,_=spaceToHidForm(x_STfore,T_fore,B)
    x_STfore=m.hidMap_foreToHid(x_STfore)
    
    # hidden ------------------
    x=cat(x_STinit,x_STfore;dims=3)
    x=m.hid(x)
    # x=backFromHiddenForm(x,C,T)
    x,_,_=hidToSpaceForm(x,C_space,T_fore)

    # bridge between inT and outT ------------------
    skip_init=m.tMapper(skip_init)
    skip_init,T_,B_=timeToSpaceForm(skip_init,H_init,C_space)

    # decoder ------------------ 
    x=m.decSpace[1:end-2](x)
    # x=m.decSpace[end](cat(x+skip_fore,x+skip_init;dims=3))
    x=m.decSpace[end-1:end](cat(x+skip_fore,x+skip_init;dims=3))
    x=backFromSpaceForm( x ,outT,B)
    return  x
end
Flux.@layer EmisModelCell

function getEmisModel(inT,outT,inC_STinit,inC_STfore,outC,hid_S,hid_T;nGroup_space=2,nGroup_hid=8,drop_prob=0.2)
    # encoder decoder for Space -----------------------
    encSpaceInit =Chain(
        Conv((3, 3),  inC_STinit => hid_S;stride=1,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=2,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=1,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=2,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish)
    )# |> device
    encSpaceFore =Chain(
        Conv((3, 3),  inC_STfore => hid_S;stride=1,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=2,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=1,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=2,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish)
    ) 

    tMapper=Conv((1, 1), inT => outT, swish) 

    decSpace = Chain(
        Conv((3, 3), hid_S => hid_S*4;stride=1,pad = 1),
        PixelShuffle(2),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=1,pad = (1,1)),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S*4;stride=1,pad = 1),
        PixelShuffle(2),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S*2 => hid_S, swish;stride=1,pad = 1),
        Conv((3, 3), hid_S => outC ;stride=1,pad = 1)
    ) 
    # hidden -----------------------
    hidMap_inToHid=Conv((1, 1), inT*hid_S => hid_T)# |> device
    hidMap_foreToHid=Conv((1, 1), outT*hid_S => hid_T)# |> device
    hid = Chain(
        GASubBlock(2*hid_T; kernel_size=21, drop_prob=drop_prob,nGroup=nGroup_hid),
        Conv((1, 1), 2*hid_T => hid_T), # reduce channel to hid_T
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/2,nGroup=nGroup_hid),
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/4,nGroup=nGroup_hid),
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/6,nGroup=nGroup_hid),
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/8,nGroup=nGroup_hid),
        GASubBlock(hid_T; kernel_size=21, drop_prob=0.0,nGroup=nGroup_hid),
        Conv((1, 1), hid_T=>outT*hid_S ), # reduce channel to hid_T
    )
    modelST=EmisModelCell(encSpaceInit,encSpaceFore,hidMap_inToHid,hidMap_foreToHid,hid,tMapper,decSpace) 
    return modelST
end


# ======================= Deposition Model ============================
mutable struct DeposModelCell
    encSpaceFore::Chain
    hidMap_foreToHid::Conv
    hid::Chain
    decSpace::Chain
end
function (m::DeposModelCell)(x_STfore)
    T=eltype(x_STfore)
    W_fore,H_fore,C_fore,T_fore,B= size(x_STfore)
    # encoder ------------------    
    x_STfore,outT,B=toSpaceForm(x_STfore)
    x_STfore=m.encSpaceFore[1](x_STfore)
    skip_fore=copy(x_STfore)
    x_STfore=m.encSpaceFore[2:end](x_STfore)
    x_STfore,C_space,_=spaceToHidForm(x_STfore,T_fore,B)
    # hidden ------------------
    x=m.hidMap_foreToHid(x_STfore)
    x=m.hid(x)
    x,_,_=hidToSpaceForm(x,C_space,T_fore)

    # decoder ------------------ 
    x=m.decSpace[1:end-2](x)
    x=m.decSpace[end-1:end](x+skip_fore)
    x=backFromSpaceForm( x,outT,B)
    x= softsign(x) # limit the output ( i.e. error of deposition error to [-1 1])
    return  x
end
Flux.@layer DeposModelCell


function getDeposModel(outT,inC_STfore,outC,hid_S,hid_T;nGroup_space=2,nGroup_hid=8,drop_prob=0.2)
    # encoder decoder for Space -----------------------

    encSpaceFore =Chain(
        Conv((3, 3),  inC_STfore => hid_S;stride=1,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=2,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=1,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=2,pad = 1),
        GroupNorm(hid_S,nGroup_space, swish)
    ) 

    decSpace = Chain(
        Conv((3, 3), hid_S => hid_S*4;stride=1,pad = 1),
        PixelShuffle(2),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S;stride=1,pad = (1,1)),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S*4;stride=1,pad = 1),
        PixelShuffle(2),
        GroupNorm(hid_S,nGroup_space, swish),
        Conv((3, 3), hid_S => hid_S, swish;stride=1,pad = 1),
        Conv((3, 3), hid_S => outC,  softsign;stride=1,pad = 1)
    ) 
    # hidden -----------------------
    hidMap_foreToHid=Conv((1, 1), outT*hid_S => hid_T)# |> device
    hid = Chain(
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob,nGroup=nGroup_hid),
        # Conv((1, 1), 2*hid_T => hid_T), # reduce channel to hid_T
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/2,nGroup=nGroup_hid),
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/4,nGroup=nGroup_hid),
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/6,nGroup=nGroup_hid),
        GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/8,nGroup=nGroup_hid),
        GASubBlock(hid_T; kernel_size=21, drop_prob=0.0,nGroup=nGroup_hid),
        Conv((1, 1), hid_T=>outT*hid_S ), # reduce channel to hid_T
    )
    modelST=DeposModelCell(encSpaceFore,hidMap_foreToHid,hid,decSpace) 
    return modelST
end


# ======================= Reaction Model ============================
mutable struct ReactModelCell
    CSNet::Chain
end
function toCSForm(x)
    W,H,C,T,B=size(x)
    x=permutedims(x,[3,1,2,4,5])
    x=reshape(x,C,W*H*T*B)
    return x,W,H,T,B
end
function backFromCSForm(x,W,H,T,B)
    C,WHTB=size(x)
    x=reshape(x,C,W,H,T,B)
    x=permutedims(x,[2,3,1,4,5])
    return x
end
function (m::ReactModelCell)(x_CSfore)
    x_CSfore,W,H,T,B=toCSForm(x_CSfore)
    x_CSfore=m.CSNet(x_CSfore)
    x_CSfore=backFromCSForm(x_CSfore,W,H,T,B)
    return x_CSfore
end
Flux.@layer ReactModelCell
function getReactModel(inC_CSfore,hid_CS,outC;ratio_drop=[0.2,0.1,0.01])
    CSNet=Chain(
        Dense(inC_CSfore=>hid_CS, relu),
        Dropout(ratio_drop[1]), 
        Dense(hid_CS=>hid_CS, swish),
        Dropout(ratio_drop[2]), 
        Dense(hid_CS=>hid_CS, swish),
        Dropout(ratio_drop[3]), 
        Dense(hid_CS=>hid_CS, swish),
        Dense(hid_CS=>outC)
    )
    modelCS=ReactModelCell(CSNet) 
    return modelCS
end