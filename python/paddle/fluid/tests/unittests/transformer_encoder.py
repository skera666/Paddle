
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.initializer import NumpyArrayInitializer
def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.,
                           name=''):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out_dtype = out.dtype
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float32")
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=fluid.initializer.Constant(0.)))
            if out_dtype == fluid.core.VarDesc.VarType.FP16:
                out = layers.cast(x=out, dtype="float16")
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out
pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer
def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_attrs,
                              param_initializer=None,
                              name='ffn'):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act=hidden_act,
                       param_attr=param_attrs[name + '_fc_0.w_0'],
                       bias_attr=param_attrs[name + '_fc_0.b_0'])
                       #param_attr=fluid.ParamAttr(
                       #    name=name + '_fc_0.w_0',
                       #    initializer=param_initializer),
                       #bias_attr=name + '_fc_0.b_0')
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)
    out = layers.fc(input=hidden,
                    size=d_hid,
                    num_flatten_dims=2,
                    param_attr=param_attrs[name + '_fc_1.w_0'],
                    bias_attr=param_attrs[name + '_fc_1.b_0'])
                    #param_attr=fluid.ParamAttr(
                    #    name=name + '_fc_1.w_0', initializer=param_initializer),
                    #bias_attr=name + '_fc_1.b_0')
    return out
def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head,
                         dropout_rate,
                         cache,
                         param_attrs,
                         param_initializer=None,
                         name='multi_head_att'):
  """
  Multi-Head Attention. Note that attn_bias is added to the logit before
  computing softmax activiation to mask certain selected positions so that
  they will not considered in attention weights.
  """
  keys = queries if keys is None else keys
  values = keys if values is None else values
  
  if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
      raise ValueError(
          "Inputs: quries, keys and values should all be 3-D tensors.")
  def __compute_qkv(queries, keys, values, n_head, d_key, d_value, param_attrs, name):
      """
      Add linear projection to queries, keys, and values.
      """
      q = layers.fc(input=queries,
                    size=d_key * n_head,
                    num_flatten_dims=2,
                    param_attr=param_attrs[name + "_query_fc.w_0"],
                    #param_attr=param_attr_query_fc_w_0,
                    bias_attr=param_attrs[name + "_query_fc.b_0"])
                    #bias_attr=param_attr_query_fc_b_0)
      k = layers.fc(input=keys,
                    size=d_key * n_head,
                    num_flatten_dims=2,
                    param_attr=param_attrs[name + "_key_fc.w_0"],
                    #param_attr=fluid.ParamAttr(
                    #    name=name + '_key_fc.w_0',
                    #    initializer=NumpyArrayInitializer(attr_kernel_k)),
                    bias_attr=param_attrs[name + "_key_fc.b_0"])
                    #bias_attr=fluid.ParamAttr(
                    #  name=name + '_key_fc.b_0',
                    #  initializer=NumpyArrayInitializer(attr_bias_q)))
      v = layers.fc(input=values,
                    size=d_value * n_head,
                    num_flatten_dims=2,
                    param_attr=param_attrs[name + "_value_fc.w_0"],
                    #param_attr=fluid.ParamAttr(
                    #    name=name + '_value_fc.w_0',
                    #    initializer=NumpyArrayInitializer(attr_kernel_v)),
                    bias_attr=param_attrs[name + "_value_fc.b_0"])
                    #bias_attr=fluid.ParamAttr(
                    #  name=name + '_value_fc.b_0',
                    #  initializer=NumpyArrayInitializer(attr_bias_v)))
      return q, k, v
  
  def __split_heads(x, n_head):
      """
      Reshape the last dimension of inpunt tensor x so that it becomes two
      dimensions and then transpose. Specifically, input a tensor with shape
      [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
      with shape [bs, n_head, max_sequence_length, hidden_dim].
      """
      hidden_size = x.shape[-1]
      # The value 0 in shape attr means copying the corresponding dimension
      # size of the input as the output dimension size.
      reshaped = layers.reshape(
          x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)
  
      # permuate the dimensions into:
      # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
      return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])
  
  def __combine_heads(x):
      """
      Transpose and then reshape the last two dimensions of inpunt tensor x
      so that it becomes one dimension, which is reverse to __split_heads.
      """
      if len(x.shape) == 3: return x
      if len(x.shape) != 4:
          raise ValueError("Input(x) should be a 4-D Tensor.")
  
      trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
      # The value 0 in shape attr means copying the corresponding dimension
      # size of the input as the output dimension size.
      return layers.reshape(
          x=trans_x,
          shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
          inplace=True)
  
   
  def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
      """
      Scaled Dot-Product Attention
      """
      scaled_q = layers.scale(x=q, scale=d_key**-0.5)
      product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
      if attn_bias:
          product += attn_bias
      weights = layers.softmax(product)
      if dropout_rate:
          weights = layers.dropout(
              weights,
              dropout_prob=dropout_rate,
              dropout_implementation="upscale_in_train",
              is_test=False)
      out = layers.matmul(weights, v)
      return out
  
  q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value, param_attrs, name)
  
  q = __split_heads(q, n_head)
  k = __split_heads(k, n_head)
  v = __split_heads(v, n_head)
  
  
  #ctx_multiheads = scaled_dot_product_attention(q, k, v, input_mask_new, d_key,
  ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                dropout_rate)
  
  out = __combine_heads(ctx_multiheads)
  
  # Project back to the model size.
  proj_out = layers.fc(input=out,
                       size=d_model,
                       num_flatten_dims=2,
                       param_attr=param_attrs[name + "_output_fc.w_0"],
                       bias_attr=param_attrs[name + "_output_fc.b_0"])
                       #param_attr=fluid.ParamAttr(
                       #    name=name + '_output_fc.w_0',
                       #    initializer=param_initializer),
                       #bias_attr=name + '_output_fc.b_0')
  return proj_out
def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_attrs=None,
                  param_initializer=None,
                  name=''):
  """The encoder layers that can be stacked to form a deep encoder.
  This module consits of a multi-head (self) attention followed by
  position-wise feed-forward networks and both the two components companied
  with the post_process_layer to add residual connection, layer normalization
  and droput.
  """
  attn_output = multi_head_attention(enc_input, None, None,
      attn_bias, d_key, d_value, d_model, n_head, attention_dropout,
      None, param_attrs, param_initializer=None, name=name + 'multi_head_att')
  
  attn_output = post_process_layer(
      enc_input,
      attn_output,
      postprocess_cmd,
      prepostprocess_dropout,
      name=name + 'post_att')
  ffd_output = positionwise_feed_forward(
      pre_process_layer(
          attn_output,
          preprocess_cmd,
          prepostprocess_dropout,
          name=name + 'pre_ffn'),
      d_inner_hid,
      d_model,
      relu_dropout,
      hidden_act,
      param_attrs,
      param_initializer=None,
      name=name + 'ffn')
  return post_process_layer(
      attn_output,
      ffd_output,
      postprocess_cmd,
      prepostprocess_dropout,
      name=name + 'post_ffn')
#n_head = 12
#d_key = 64
#d_value = 64
#d_model = 768
#enc_input = fluid.layers.data(name="enc_input", shape=[batch_size, from_seq_len, hidden_dim], dtype="float32",append_batch_size=False)
#input_mask_new = fluid.layers.data(name="input_mask_new", shape=[batch_size, from_seq_len, from_seq_len], dtype="float32",append_batch_size=False)
#n_head_self_attn_mask = fluid.layers.stack(
#    x=[input_mask_new] * n_head, axis=1)
#
#from_tensor = np.random.random((batch_size, from_seq_len, hidden_dim)).astype("float32")
#
##input_mask = fluid.layers.data(name="input_mask", shape=[batch_size, from_seq_len, 1], dtype="float32",append_batch_size=False)
##self_attn_mask = fluid.layers.matmul(
##    x=input_mask, y=input_mask, transpose_y=True)
##self_attn_mask = fluid.layers.scale(
##    x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
##print ("self_attn_mask~~~~~~~~~~~~~~~", self_attn_mask)
##n_head_self_attn_mask = fluid.layers.stack(
##    x=[self_attn_mask] * n_head, axis=1)
##print ("n_head_self_attn_mask~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", n_head_self_attn_mask)
#
#prepostprocess_dropout = 0
#emb_size=768
#d_inner_hid=emb_size * 4
#attention_dropout=0
#relu_dropout=0
## TODO
#hidden_act='gelu'
##attn_bias = n_head_self_attn_mask
#preprocess_cmd="n"
#postprocess_cmd="da"
#param_initializer=None
#
#enc_output = encoder_layer(
#    enc_input,
#    n_head_self_attn_mask,
#    #input_mask_new,
#    n_head,
#    d_key,
#    d_value,
#    d_model,
#    d_inner_hid,
#    prepostprocess_dropout,
#    attention_dropout,
#    relu_dropout,
#    hidden_act,
#    preprocess_cmd,
#    postprocess_cmd,
#    param_initializer=param_initializer,
#    name="")
#
##print(fluid.default_main_program().to_string(True))
#
#place = fluid.CPUPlace()
#exe = fluid.Executor(place)
#
#exe.run(fluid.default_startup_program())
#result = exe.run(fluid.default_main_program(),
#    #feed={"enc_input": from_tensor, "input_mask": attr_mask_old},
#    feed={"enc_input": from_tensor, "input_mask_new": attr_mask},
#    fetch_list=[enc_output],
#    return_numpy=True)
#print (result)
