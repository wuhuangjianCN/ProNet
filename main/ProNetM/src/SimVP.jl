using Flux,MLUtils
# julia version of the SimVP model


# ======================== MixMlp ========================
function buildMixMlp(in_features; hidden_features=nothing, out_features=nothing, act_layer=gelu, drop=0.0)

    out_features = isnothing(out_features) ? in_features : out_features
    hidden_features = isnothing(hidden_features) ? in_features : hidden_features
    MixMlp=Chain(
        Conv((1, 1), in_features => hidden_features), # fc1
        Conv((3, 3), hidden_features => hidden_features,act_layer,pad=1, groups=hidden_features), # dwconv + act
        Dropout(drop), # drop
        Conv((1, 1), hidden_features => out_features), # fc2
        Dropout(drop) # drop
    )

    return MixMlp
end

# ======================== AttentionModule ========================
mutable struct AttentionModuleCell
    conv0::Conv
    conv_spatial::Conv
    conv1::Conv
end
function AttentionModuleCell(num_Channel::T; kernel_size=21, dilation=3) where T
    d_k = 2 * dilation - 1
    d_p = floor((d_k - 1) / 2) |> T
    dd_k = div(kernel_size, dilation) + (div(kernel_size, dilation) % 2 - 1) |> T
    dd_p = floor(dilation * (dd_k - 1)/2 ) |> T

    conv0 = Conv((d_k, d_k), num_Channel => num_Channel, pad=d_p, groups=num_Channel)
    conv_spatial = Conv((dd_k, dd_k), num_Channel => num_Channel; stride=1, pad=dd_p, groups=num_Channel, dilation=dilation)
    conv1 = Conv((1, 1), num_Channel => 2*num_Channel)

    return AttentionModuleCell(conv0, conv_spatial, conv1)
end

function (m::AttentionModuleCell)(x)
    x = m.conv0(x)  # depth-wise convolution
    x = m.conv_spatial(x)  # depth-wise dilation convolution
    
    f_g = m.conv1(x)
    f_x, g_x = MLUtils.chunk(f_g,2,dims=ndims(f_g)-1) # split the channel dimension into two parts
    return σ.(g_x) .* f_x  # σ is the sigmoid function in Julia
end

# Flux.@functor AttentionModuleCell
Flux.@layer AttentionModuleCell
"""
AttentionModuleCell(640)
"""

# ======================== SpatialAttention ========================
mutable struct SpatialAttentionCell
    proj_1::Conv
    spatial_gating_unit::AttentionModuleCell
    proj_2::Conv
    attn_shorcut::Bool
end
function SpatialAttentionCell(num_Channel::T; kernel_size=21,  attn_shorcut=true) where T
    proj_1 = Conv((1, 1), num_Channel => num_Channel,gelu)
    spatial_gating_unit = AttentionModuleCell(num_Channel; kernel_size=kernel_size)
    proj_2 = Conv((1, 1), num_Channel => num_Channel)
    return SpatialAttentionCell(proj_1, spatial_gating_unit, proj_2, attn_shorcut)
end
function (m::SpatialAttentionCell)(x)
    
    if m.attn_shorcut
        shortcut = copy(x)
    end
    x = m.proj_1(x)
    x = m.spatial_gating_unit(x)
    x = m.proj_2(x)
    if m.attn_shorcut
        return x + shortcut
    else
        return x_attn
    end
end
# Flux.@functor SpatialAttentionCell
Flux.@layer SpatialAttentionCell
"""
m=SpatialAttentionCell(640)
trainable(m)
Flux.params(m)
"""

# ======================== MetaBlock ========================
mutable struct GASubBlock
    attn::SpatialAttentionCell
    drop_path::Dropout
    norm::GroupNorm
    mlp::Chain
end
function GASubBlock(num_Channel::T; kernel_size=21, drop_prob=0.01,nGroup=8) where T
    attn = SpatialAttentionCell(num_Channel; kernel_size=kernel_size)
    drop_path = Dropout(drop_prob) 
    norm = GroupNorm(num_Channel,nGroup)
    mlp = buildMixMlp(num_Channel)
    return GASubBlock(attn, drop_path,norm, mlp)
end
function (m::GASubBlock)(x)
    x = m.attn(x)
    x = m.drop_path(x)
    x = m.norm(x)
    x = m.mlp(x)
    return x
end
# Flux.@functor GASubBlock
Flux.@layer GASubBlock
# """

function toSpaceForm(x)
    W,H,C,T,B=size(x)
    x = reshape(x, (W,H,C,T*B)) # treat time as sample
    return x,T,B
end
function backFromSpaceForm(x,T,B)    
    W_,H_,C_,_=size(x)
    x=reshape(x, (W_,H_,C_,T,B)) # reverse to input format
    return x
end
function spaceToTimeForm(x,T,B)
    W,H,C,T_B_=size(x)
    x=reshape(x, (W,H*C,T,B)) # treat time as channel
    return x,H,C
end
function timeToSpaceForm(x,H,C)
    W_,_,T_,B_=size(x)
    x=reshape(x, (W_,H,C,T_*B_)) # reverse to input format
    return x,T_,B_
end
function spaceToHidForm(x,T,B)
    W,H,C,T_B_=size(x)
    x=reshape(x, (W,H,C*T,B)) # treat time as channel
    return x,C,T
end
function hidToSpaceForm(x,C,T)
    W,H,_,B=size(x)
    x=reshape(x, (W,H,C,T*B)) # reverse to input format
    return x,T,B
end
function toTimeForm(x)
    W,H,C,T,B=size(x)
    x = reshape(x,(W,H*C,T,B)) # treat time as channel and channel as sample
    return x,H,C
end
function backFromTimeForm(x,H,C)
    W_,_,T_,B_=size(x)
    x=reshape(x, (W_,H,C,T_,B_)) # reverse to input format
    return x
end
function toHiddenForm(x)
    W,H,C,T,B=size(x)
    x=reshape(x, (W,H,C*T,B)) # treat time as channel
    return x,C,T
end
function backFromHiddenForm(x,C,T)
    W_,H_,_,B_=size(x)
    x=reshape(x, (W_,H_,C,T,B_)) # reverse to input format
    return x
end

