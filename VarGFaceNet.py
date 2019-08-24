# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 8/24/19 11:25 AM
# @Author : mengao.zhao
# @File : VarGFaceNet.py

import mxnet as mx


def Act(data, act_type, name):
    if act_type == 'prelu':
      body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


def get_setting_params(**kwargs):
    # bn_params
    bn_mom = kwargs.get('bn_mom', 0.9)
    bn_eps = kwargs.get('bn_eps', 2e-5)
    fix_gamma = kwargs.get('fix_gamma', False)
    use_global_stats = kwargs.get('use_global_stats', False)
    # net_setting param
    workspace = kwargs.get('workspace', 512)
    act_type = kwargs.get('act_type', 'prelu')
    use_se = kwargs.get('use_se', True)
    se_ratio = kwargs.get('se_ratio', 4)
    group_base = kwargs.get('group_base', 8)

    setting_params = {}
    setting_params['bn_mom'] = bn_mom
    setting_params['bn_eps'] = bn_eps
    setting_params['fix_gamma'] = fix_gamma
    setting_params['use_global_stats'] = use_global_stats
    setting_params['workspace'] = workspace
    setting_params['act_type'] = act_type
    setting_params['use_se'] = use_se
    setting_params['se_ratio'] = se_ratio
    setting_params['group_base'] = group_base

    return setting_params


def separable_conv2d(data,
                     in_channels,
                     out_channels,
                     kernel,
                     pad,
                     setting_params,
                     stride=(1, 1),
                     factor=1,
                     bias=False,
                     bn_dw_out=True,
                     act_dw_out=True,
                     bn_pw_out=True,
                     act_pw_out=True,
                     dilate=1,
                     name=None):
    bn_mom = setting_params['bn_mom']
    bn_eps = setting_params['bn_eps']
    fix_gamma = setting_params['fix_gamma']
    use_global_stats = setting_params['use_global_stats']
    workspace = setting_params['workspace']
    group_base = setting_params['group_base']
    act_type = setting_params['act_type']
    assert in_channels % group_base == 0

    # depthwise
    dw_out = mx.sym.Convolution(data=data,
                                num_filter=int(in_channels * factor),
                                kernel=kernel,
                                pad=pad,
                                stride=stride,
                                no_bias=False if bias else True,
                                num_group=int(in_channels / group_base),
                                dilate=(dilate, dilate),
                                workspace=workspace,
                                name=name + '_conv2d_depthwise')
    if bn_dw_out:
        dw_out = mx.sym.BatchNorm(data=dw_out,
                                  fix_gamma=fix_gamma,
                                  eps=bn_eps,
                                  momentum=bn_mom,
                                  use_global_stats=use_global_stats,
                                  name=name + '_conv2d_depthwise_bn')
    if act_dw_out:
        dw_out = Act(data=dw_out,
                     act_type=act_type,
                     name=name + '_conv2d_depthwise_act')
    # pointwise
    pw_out = mx.sym.Convolution(data=dw_out,
                                num_filter=out_channels,
                                kernel=(1, 1),
                                stride=(1, 1),
                                pad=(0, 0),
                                num_group=1,
                                no_bias=False if bias else True,
                                workspace=workspace,
                                name=name + '_conv2d_pointwise')
    if bn_pw_out:
        pw_out = mx.sym.BatchNorm(data=pw_out,
                                  fix_gamma=fix_gamma,
                                  eps=bn_eps,
                                  momentum=bn_mom,
                                  use_global_stats=use_global_stats,
                                  name=name + '_conv2d_pointwise_bn')
    if act_pw_out:
        pw_out = Act(data=pw_out,
                     act_type=act_type,
                     name=name + '_conv2d_pointwise_act')
    return pw_out


def vargnet_block(data,
                  n_out_ch1,
                  n_out_ch2,
                  n_out_ch3,
                  setting_params,
                  factor=2,
                  dim_match=True,
                  multiplier=1,
                  kernel=(3, 3),
                  stride=(1, 1),
                  dilate=1,
                  with_dilate=False,
                  name=None):
    use_se = setting_params['use_se']
    se_ratio = setting_params['se_ratio']
    act_type = setting_params['act_type']

    out_channels_1 = int(n_out_ch1 * multiplier)
    out_channels_2 = int(n_out_ch2 * multiplier)
    out_channels_3 = int(n_out_ch3 * multiplier)

    pad = (((kernel[0] - 1) * dilate + 1) // 2,
           ((kernel[1] - 1) * dilate + 1) // 2)

    if with_dilate:
        stride = (1, 1)
    if dim_match:
        short_cut = data
    else:
        short_cut = separable_conv2d(data=data,
                                     in_channels=out_channels_1,
                                     out_channels=out_channels_3,
                                     kernel=kernel,
                                     pad=pad,
                                     setting_params=setting_params,
                                     stride=stride,
                                     factor=factor,
                                     bias=False,
                                     act_pw_out=False,
                                     dilate=dilate,
                                     name=name + '_shortcut')
    sep1_data = separable_conv2d(data=data,
                                 in_channels=out_channels_1,
                                 out_channels=out_channels_2,
                                 kernel=kernel,
                                 pad=pad,
                                 setting_params=setting_params,
                                 stride=stride,
                                 factor=factor,
                                 bias=False,
                                 dilate=dilate,
                                 name=name + '_sep1_data')
    sep2_data = separable_conv2d(data=sep1_data,
                                 in_channels=out_channels_2,
                                 out_channels=out_channels_3,
                                 kernel=kernel,
                                 pad=pad,
                                 setting_params=setting_params,
                                 stride=(1, 1),
                                 factor=factor,
                                 bias=False,
                                 dilate=dilate,
                                 act_pw_out=False,
                                 name=name + '_sep2_data')

    if use_se:
        body = mx.sym.Pooling(data=sep2_data,
                              global_pool=True,
                              pool_type='avg',
                              name=name + '_se_pool1')
        body = mx.sym.Convolution(data=body,
                                  num_filter=out_channels_3 // se_ratio,
                                  kernel=(1, 1),
                                  stride=(1, 1),
                                  pad=(0, 0),
                                  name=name + "_se_conv1",
                                  )
        body = Act(data=body, act_type=act_type, name=name + '_se_act1')

        body = mx.sym.Convolution(data=body,
                                  num_filter=out_channels_3,
                                  kernel=(1, 1),
                                  stride=(1, 1),
                                  pad=(0, 0),
                                  name=name + "_se_conv2",
                                  )
        body = mx.symbol.Activation(data=body,
                                    act_type='sigmoid',
                                    name=name + "_se_sigmoid")
        sep2_data = mx.symbol.broadcast_mul(sep2_data, body)

    out_data = sep2_data + short_cut
    out_data = Act(data=out_data, act_type=act_type, name=name + '_out_data_act')
    return out_data


def vargnet_branch_merge_block(data,
                               n_out_ch1,
                               n_out_ch2,
                               n_out_ch3,
                               setting_params,
                               factor=2,
                               dim_match=False,
                               multiplier=1,
                               kernel=(3, 3),
                               stride=(2, 2),
                               dilate=1,
                               with_dilate=False,
                               name=None):
    act_type = setting_params['act_type']

    out_channels_1 = int(n_out_ch1 * multiplier)
    out_channels_2 = int(n_out_ch2 * multiplier)
    out_channels_3 = int(n_out_ch3 * multiplier)

    pad = (((kernel[0] - 1) * dilate + 1) // 2,
           ((kernel[1] - 1) * dilate + 1) // 2)

    if with_dilate:
        stride = (1, 1)
    if dim_match:
        short_cut = data
    else:
        short_cut = separable_conv2d(data=data,
                                     in_channels=out_channels_1,
                                     out_channels=out_channels_3,
                                     kernel=kernel,
                                     pad=pad,
                                     setting_params=setting_params,
                                     stride=stride,
                                     factor=factor,
                                     bias=False,
                                     act_pw_out=False,
                                     dilate=dilate,
                                     name=name + '_shortcut')
    sep1_data_brach1 = separable_conv2d(data=data,
                                        in_channels=out_channels_1,
                                        out_channels=out_channels_2,
                                        kernel=kernel,
                                        pad=pad,
                                        setting_params=setting_params,
                                        stride=stride,
                                        factor=factor,
                                        bias=False,
                                        dilate=dilate,
                                        act_pw_out=False,
                                        name=name + '_sep1_data_branch')
    sep1_data_brach2 = separable_conv2d(data=data,
                                        in_channels=out_channels_1,
                                        out_channels=out_channels_2,
                                        kernel=kernel,
                                        pad=pad,
                                        setting_params=setting_params,
                                        stride=stride,
                                        factor=factor,
                                        bias=False,
                                        dilate=dilate,
                                        act_pw_out=False,
                                        name=name + '_sep2_data_branch')
    sep1_data = sep1_data_brach1 + sep1_data_brach2
    sep1_data = Act(data=sep1_data, act_type=act_type, name=name + '_sep1_data_act')
    sep2_data = separable_conv2d(data=sep1_data,
                                 in_channels=out_channels_2,
                                 out_channels=out_channels_3,
                                 kernel=kernel,
                                 pad=pad,
                                 setting_params=setting_params,
                                 stride=(1, 1),
                                 factor=factor,
                                 bias=False,
                                 dilate=dilate,
                                 act_pw_out=False,
                                 name=name + '_sep2_data')
    out_data = sep2_data + short_cut
    out_data = Act(data=out_data, act_type=act_type, name=name + '_out_data_act')
    return out_data


def add_vargnet_conv_block(data,
                           stage,
                           units,
                           in_channels,
                           out_channels,
                           setting_params,
                           kernel=(3, 3),
                           stride=(2, 2),
                           multiplier=1,
                           factor=2,
                           dilate=1,
                           with_dilate=False,
                           name=None):
    assert stage >= 2, 'stage is {}, stage must be set >=2'.format(stage)
    data = vargnet_branch_merge_block(data=data,
                                      n_out_ch1=in_channels,
                                      n_out_ch2=out_channels,
                                      n_out_ch3=out_channels,
                                      setting_params=setting_params,
                                      factor=factor,
                                      dim_match=False,
                                      multiplier=multiplier,
                                      kernel=kernel,
                                      stride=stride,
                                      dilate=dilate,
                                      with_dilate=with_dilate,
                                      name=name + 'stage_{}_unit_1'.format(stage))
    for i in range(units - 1):
        data = vargnet_block(data=data,
                             n_out_ch1=out_channels,
                             n_out_ch2=out_channels,
                             n_out_ch3=out_channels,
                             setting_params=setting_params,
                             factor=factor,
                             dim_match=True,
                             multiplier=multiplier,
                             kernel=kernel,
                             stride=(1, 1),
                             dilate=dilate,
                             with_dilate=with_dilate,
                             name=name + 'stage_{}_unit_{}'.format(stage, i + 2))
    return data


def add_head_block(data,
                   num_filter,
                   setting_params,
                   multiplier,
                   head_pooling=False,
                   kernel=(3, 3),
                   stride=(2, 2),
                   pad=(1, 1),
                   name=None):
    bn_mom = setting_params['bn_mom']
    bn_eps = setting_params['bn_eps']
    fix_gamma = setting_params['fix_gamma']
    use_global_stats = setting_params['use_global_stats']
    workspace = setting_params['workspace']
    act_type = setting_params['act_type']
    channels = int(num_filter * multiplier)

    conv1 = mx.sym.Convolution(data=data,
                               num_filter=channels,
                               kernel=kernel,
                               pad=pad,
                               stride=stride,
                               no_bias=True,
                               num_group=1,
                               workspace=workspace,
                               name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1,
                           fix_gamma=fix_gamma,
                           eps=bn_eps,
                           momentum=bn_mom,
                           use_global_stats=use_global_stats,
                           name=name + '_conv1_bn')

    act1 = Act(data=bn1, act_type=act_type, name=name + '_conv1_act')

    if head_pooling:
        head_data = mx.symbol.Pooling(data=act1,
                                      kernel=(3, 3),
                                      stride=(2, 2),
                                      pad=(1, 1),
                                      pool_type='max',
                                      name=name + '_max_pooling')
    else:
        head_data = vargnet_block(data=act1,
                                  n_out_ch1=num_filter,
                                  n_out_ch2=num_filter,
                                  n_out_ch3=num_filter,
                                  setting_params=setting_params,
                                  factor=1,
                                  dim_match=False,
                                  multiplier=multiplier,
                                  kernel=kernel,
                                  stride=(2, 2),
                                  dilate=1,
                                  with_dilate=False,
                                  name=name + '_head_pooling')
    return head_data


def add_emb_block(data,
                  input_channels,
                  last_channels,
                  emb_size,
                  setting_params,
                  bias=False,
                  name=None):
    bn_mom = setting_params['bn_mom']
    bn_eps = setting_params['bn_eps']
    fix_gamma = setting_params['fix_gamma']
    use_global_stats = setting_params['use_global_stats']
    workspace = setting_params['workspace']
    act_type = setting_params['act_type']
    group_base = setting_params['group_base']
    # last channels
    if input_channels != last_channels:
        data = mx.sym.Convolution(data=data,
                                  num_filter=last_channels,
                                  kernel=(1, 1),
                                  pad=(0, 0),
                                  stride=(1, 1),
                                  no_bias=False if bias else True,
                                  workspace=workspace,
                                  name=name + '_convx')
        data = mx.sym.BatchNorm(data=data,
                                fix_gamma=fix_gamma,
                                eps=bn_eps,
                                momentum=bn_mom,
                                use_global_stats=use_global_stats,
                                name=name + '_convx_bn')
        data = Act(data=data, act_type=act_type, name=name + '_convx_act')
    # depthwise
    convx_depthwise = mx.sym.Convolution(data=data,
                                         num_filter=last_channels,
                                         num_group=int(last_channels / group_base),
                                         kernel=(7, 7),
                                         pad=(0, 0),
                                         stride=(1, 1),
                                         no_bias=False if bias else True,
                                         workspace=workspace,
                                         name=name + '_convx_depthwise')
    convx_depthwise = mx.sym.BatchNorm(data=convx_depthwise,
                                       fix_gamma=fix_gamma,
                                       eps=bn_eps,
                                       momentum=bn_mom,
                                       use_global_stats=use_global_stats,
                                       name=name + '_convx_depthwise_bn')
    # pointwise
    convx_pointwise = mx.sym.Convolution(data=convx_depthwise,
                                         num_filter=last_channels // 2,
                                         kernel=(1, 1),
                                         pad=(0, 0),
                                         stride=(1, 1),
                                         no_bias=False if bias else True,
                                         workspace=workspace,
                                         name=name + '_convx_pointwise')
    convx_pointwise = mx.sym.BatchNorm(data=convx_pointwise,
                                       fix_gamma=fix_gamma,
                                       eps=bn_eps,
                                       momentum=bn_mom,
                                       use_global_stats=use_global_stats,
                                       name=name + '_convx_lpointwise_bn')
    convx_pointwise = Act(data=convx_pointwise,
                          act_type=act_type,
                          name=name + '_convx_pointwise_act')
    emb_feat = mx.sym.FullyConnected(data=convx_pointwise, num_hidden=emb_size)
    emb_feat = mx.sym.BatchNorm(data=emb_feat,
                                fix_gamma=fix_gamma,
                                eps=bn_eps,
                                momentum=bn_mom,
                                use_global_stats=use_global_stats,
                                name='fc1')
    return emb_feat


def get_symbol(**kwargs):
    setting_params = get_setting_params(**kwargs)
    multiplier = kwargs.get('multiplier', 1.25)
    emb_size = kwargs.get('emb_size', 512)
    factor = kwargs.get('factor', 2)
    head_pooling = kwargs.get('head_pooling', False)
    num_stage = 3
    stage_list = [2, 3, 4]
    units = [3, 7, 4]
    filter_list = [32, 64, 128, 256]
    last_channels = 1024
    dilate_list = [1, 1, 1]
    with_dilate_list = [False, False, False]

    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = data - 127.5
    data = data * 0.0078125

    body = add_head_block(data=data,
                          num_filter=filter_list[0],
                          setting_params=setting_params,
                          multiplier=multiplier,
                          head_pooling=head_pooling,
                          kernel=(3, 3),
                          stride=(1, 1),
                          pad=(1, 1),
                          name="vargface_head")

    for i in range(num_stage):
        body = add_vargnet_conv_block(data=body,
                                      stage=stage_list[i],
                                      units=units[i],
                                      in_channels=filter_list[i],
                                      out_channels=filter_list[i + 1],
                                      setting_params=setting_params,
                                      kernel=(3, 3),
                                      stride=(2, 2),
                                      multiplier=multiplier,
                                      factor=factor,
                                      dilate=dilate_list[i],
                                      with_dilate=with_dilate_list[i],
                                      name="vargface")
        
    emb_feat = add_emb_block(data=body,
                             input_channels=filter_list[3],
                             last_channels=last_channels,
                             emb_size=emb_size,
                             setting_params=setting_params,
                             bias=False,
                             name='embed')
    emb_feat.save('VarGFace-symbol.json')
    return emb_feat

if __name__ == '__main__':
    get_symbol()
