       �K"	  �*���Abrain.Event:2B����      D(�	�Ֆ*���A"��
~
PlaceholderPlaceholder*$
shape:���������*
dtype0*/
_output_shapes
:���������
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:���������
*
shape:���������

R
Placeholder_2Placeholder*
dtype0
*
_output_shapes
:*
shape:
�
.conv2d/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@conv2d/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
,conv2d/kernel/Initializer/random_uniform/minConst* 
_class
loc:@conv2d/kernel*
valueB
 *����*
dtype0*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@conv2d/kernel*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv2d/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv2d/kernel
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
conv2d/kernel
VariableV2*
dtype0*&
_output_shapes
:*
shared_name * 
_class
loc:@conv2d/kernel*
	container *
shape:
�
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv2d/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d/kernel/readIdentityconv2d/kernel*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
U
ReluReluconv2d/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *�[q�*
dtype0*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *�[q>*
dtype0*
_output_shapes
: 
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_1/kernel
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:*
T0
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
conv2d_1/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:*
T0
g
conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
Y
Relu_1Reluconv2d_2/Conv2D*/
_output_shapes
:���������*
T0
�
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_2/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_2/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *��:�*
dtype0*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *��:>*
dtype0*
_output_shapes
: 
�
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0
�
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_2/kernel
�
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:*
T0
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:*
T0
�
conv2d_2/kernel
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_2/kernel*
	container *
shape:
�
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_2/kernel
g
conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_3/Conv2DConv2DRelu_1conv2d_2/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
Y
Relu_2Reluconv2d_3/Conv2D*/
_output_shapes
:���������*
T0
�
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_3/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *��*
dtype0*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *�>*
dtype0*
_output_shapes
: 
�
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 
�
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
conv2d_3/kernel
VariableV2*"
_class
loc:@conv2d_3/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
�
conv2d_3/kernel/AssignAssignconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_3/kernel/readIdentityconv2d_3/kernel*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
g
conv2d_4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_4/Conv2DConv2DRelu_2conv2d_3/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Y
Relu_3Reluconv2d_4/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_4/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *HY�*
dtype0*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *HY>*
dtype0*
_output_shapes
: 
�
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_4/kernel*
seed2 
�
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
conv2d_4/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_4/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
conv2d_4/kernel/AssignAssignconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_4/kernel/readIdentityconv2d_4/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
g
conv2d_5/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_5/Conv2DConv2DRelu_3conv2d_4/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
Y
Relu_4Reluconv2d_5/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_5/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_5/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel*
valueB
 *d��
�
.conv2d_5/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *d�=*
dtype0*
_output_shapes
: 
�
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@conv2d_5/kernel*
seed2 *
dtype0*&
_output_shapes
:
�
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
�
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
�
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_5/kernel
�
conv2d_5/kernel
VariableV2*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_5/kernel*
	container *
shape:*
dtype0
�
conv2d_5/kernel/AssignAssignconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
�
conv2d_5/kernel/readIdentityconv2d_5/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_5/kernel
g
conv2d_6/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_6/Conv2DConv2DRelu_4conv2d_5/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
Y
Relu_5Reluconv2d_6/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_6/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_6/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_6/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_6/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
.conv2d_6/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@conv2d_6/kernel*
valueB
 *���=*
dtype0
�
8conv2d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_6/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_6/kernel*
seed2 
�
.conv2d_6/kernel/Initializer/random_uniform/subSub.conv2d_6/kernel/Initializer/random_uniform/max.conv2d_6/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_6/kernel*
_output_shapes
: 
�
.conv2d_6/kernel/Initializer/random_uniform/mulMul8conv2d_6/kernel/Initializer/random_uniform/RandomUniform.conv2d_6/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
�
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
�
conv2d_6/kernel
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_6/kernel*
	container 
�
conv2d_6/kernel/AssignAssignconv2d_6/kernel*conv2d_6/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_6/kernel/readIdentityconv2d_6/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
g
conv2d_7/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Y
Relu_6Reluconv2d_7/Conv2D*
T0*/
_output_shapes
:���������
^
Reshape/shapeConst*
valueB"����\  *
dtype0*
_output_shapes
:
j
ReshapeReshapeRelu_6Reshape/shape*(
_output_shapes
:����������
*
T0*
Tshape0
�
-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"\  d   
�
+dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *����*
dtype0
�
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�
d*

seed *
T0*
_class
loc:@dense/kernel*
seed2 
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	�
d*
T0*
_class
loc:@dense/kernel
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d
�
dense/kernel
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	�
d*
dtype0*
_output_shapes
:	�
d
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	�
d
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d
�
dense/MatMulMatMulReshapedense/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
N
Relu_7Reludense/MatMul*
T0*'
_output_shapes
:���������d
�
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
valueB"d   
   *
dtype0*
_output_shapes
:
�
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
valueB
 *�'o�*
dtype0*
_output_shapes
: 
�
-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
valueB
 *�'o>*
dtype0*
_output_shapes
: 
�
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:d
*

seed *
T0*!
_class
loc:@dense_1/kernel
�
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
�
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

�
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:d
*
T0*!
_class
loc:@dense_1/kernel
�
dense_1/kernel
VariableV2*
	container *
shape
:d
*
dtype0*
_output_shapes

:d
*
shared_name *!
_class
loc:@dense_1/kernel
�
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:d

{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

�
dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
dense_1/bias
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@dense_1/bias*
	container *
shape:

�
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
q
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:
*
T0*
_class
loc:@dense_1/bias
�
dense_2/MatMulMatMulRelu_7dense_1/kernel/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

h
logistic_loss/zeros_like	ZerosLikedense_2/BiasAdd*
T0*'
_output_shapes
:���������

�
logistic_loss/GreaterEqualGreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:���������

�
logistic_loss/SelectSelectlogistic_loss/GreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:���������

[
logistic_loss/NegNegdense_2/BiasAdd*
T0*'
_output_shapes
:���������

�
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negdense_2/BiasAdd*'
_output_shapes
:���������
*
T0
j
logistic_loss/mulMuldense_2/BiasAddPlaceholder_1*
T0*'
_output_shapes
:���������

s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*'
_output_shapes
:���������

b
logistic_loss/ExpExplogistic_loss/Select_1*'
_output_shapes
:���������
*
T0
a
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:���������

n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:���������

V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
`
MeanMeanlogistic_lossConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
conv_loss/tagsConst*
_output_shapes
: *
valueB B	conv_loss*
dtype0
Q
	conv_lossScalarSummaryconv_loss/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
f
gradients/Mean_grad/ShapeShapelogistic_loss*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������

h
gradients/Mean_grad/Shape_1Shapelogistic_loss*
out_type0*
_output_shapes
:*
T0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
�
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:���������

s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
out_type0*
_output_shapes
:*
T0
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
�
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
�
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
�
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:���������

�
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1
z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
_output_shapes
:*
T0*
out_type0
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
_output_shapes
:*
T0*
out_type0
�
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
�
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
�
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:���������

�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:���������

�
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:���������

�
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:���������

�
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:���������
*
T0
~
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikedense_2/BiasAdd*
T0*'
_output_shapes
:���������

�
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:���������

�
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*'
_output_shapes
:���������
*
T0
�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
�
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������

�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������

u
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
u
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
�
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/mul_grad/mulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Placeholder_1*
T0*'
_output_shapes
:���������

�
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
&gradients/logistic_loss/mul_grad/mul_1Muldense_2/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������

�
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
�
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:���������

�
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:���������

�
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*'
_output_shapes
:���������
*
T0
�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:���������

�
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������

�
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:���������

�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
�
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select*'
_output_shapes
:���������

�
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1
�
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������

�
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:���������
*
T0
�
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:

u
/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN+^gradients/dense_2/BiasAdd_grad/BiasAddGrad
�
7gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������

�
9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_2/BiasAdd_grad/BiasAddGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

�
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_77gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:d
*
transpose_a(*
transpose_b( *
T0
�
.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1
�
6gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_2/MatMul_grad/MatMul/^gradients/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*7
_class-
+)loc:@gradients/dense_2/MatMul_grad/MatMul
�
8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:d

�
gradients/Relu_7_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_7*'
_output_shapes
:���������d*
T0
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_7_grad/ReluGraddense/kernel/read*
T0*(
_output_shapes
:����������
*
transpose_a( *
transpose_b(
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/Relu_7_grad/ReluGrad*
transpose_b( *
T0*
_output_shapes
:	�
d*
transpose_a(
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������
*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�
d*
T0
b
gradients/Reshape_grad/ShapeShapeRelu_6*
T0*
out_type0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
gradients/Relu_6_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_6*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_7/Conv2D_grad/ShapeNShapeNRelu_5conv2d_6/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0
�
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_5'gradients/conv2d_7/Conv2D_grad/ShapeN:1gradients/Relu_6_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
/gradients/conv2d_7/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
gradients/Relu_5_grad/ReluGradReluGrad7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyRelu_5*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_6/Conv2D_grad/ShapeNShapeNRelu_4conv2d_5/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4'gradients/conv2d_6/Conv2D_grad/ShapeN:1gradients/Relu_5_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
/gradients/conv2d_6/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_6/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_6/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*E
_class;
97loc:@gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput
�
9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_6/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
gradients/Relu_4_grad/ReluGradReluGrad7gradients/conv2d_6/Conv2D_grad/tuple/control_dependencyRelu_4*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_5/Conv2D_grad/ShapeNShapeNRelu_3conv2d_4/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_4/kernel/readgradients/Relu_4_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
/gradients/conv2d_5/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_5/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter
�
gradients/Relu_3_grad/ReluGradReluGrad7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyRelu_3*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNRelu_2conv2d_3/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/Relu_3_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2'gradients/conv2d_4/Conv2D_grad/ShapeN:1gradients/Relu_3_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
/gradients/conv2d_4/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
gradients/Relu_2_grad/ReluGradReluGrad7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyRelu_2*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
�
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/Relu_2_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
/gradients/conv2d_3/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*F
_class<
:8loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyRelu_1*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
/gradients/conv2d_2/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_2/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
gradients/Relu_grad/ReluGradReluGrad7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:���������
�
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/Relu_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
�
5gradients/conv2d/Conv2D_grad/tuple/control_dependencyIdentity0gradients/conv2d/Conv2D_grad/Conv2DBackpropInput.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1Identity1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
b
GradientDescent/learning_rateConst*
valueB
 *o;*
dtype0*
_output_shapes
: 
�
9GradientDescent/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelGradientDescent/learning_rate7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
use_locking( 
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:*
use_locking( 
�
;GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentApplyGradientDescentconv2d_3/kernelGradientDescent/learning_rate9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_4/kernel
�
;GradientDescent/update_conv2d_5/kernel/ApplyGradientDescentApplyGradientDescentconv2d_5/kernelGradientDescent/learning_rate9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_6/kernel/ApplyGradientDescentApplyGradientDescentconv2d_6/kernelGradientDescent/learning_rate9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_6/kernel
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d*
use_locking( 
�
:GradientDescent/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelGradientDescent/learning_rate8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

�
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( *
T0
�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_3/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_4/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_5/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_6/kernel/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
�
ArgMaxArgMaxdense_2/BiasAddArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
P
CastCastEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_1MeanCastConst_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
N
Merge/MergeSummaryMergeSummary	conv_loss*
_output_shapes
: *
N"Nz�      ?�K	�~�*���AJ��
��
9
Add
x"T
y"T
z"T"
Ttype:
2	
T
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	��
�
ApplyGradientDescent
var"T�

alpha"T

delta"T
out"T�"
Ttype:
2	"
use_lockingbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type"
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
+
Exp
x"T
y"T"
Ttype:	
2
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
-
Log1p
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
9
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.12v1.4.0-19-ga52c8d9��
~
PlaceholderPlaceholder*
dtype0*/
_output_shapes
:���������*$
shape:���������
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:���������
*
shape:���������

R
Placeholder_2Placeholder*
dtype0
*
_output_shapes
:*
shape:
�
.conv2d/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@conv2d/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
,conv2d/kernel/Initializer/random_uniform/minConst* 
_class
loc:@conv2d/kernel*
valueB
 *����*
dtype0*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@conv2d/kernel*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
T0
�
conv2d/kernel
VariableV2* 
_class
loc:@conv2d/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
�
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@conv2d/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d/kernel/readIdentityconv2d/kernel*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
e
conv2d/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
U
ReluReluconv2d/Conv2D*/
_output_shapes
:���������*
T0
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel*%
valueB"            
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *�[q�*
dtype0*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *�[q>*
dtype0*
_output_shapes
: 
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
conv2d_1/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
g
conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Y
Relu_1Reluconv2d_2/Conv2D*/
_output_shapes
:���������*
T0
�
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@conv2d_2/kernel*%
valueB"            *
dtype0
�
.conv2d_2/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *��:�*
dtype0*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *��:>*
dtype0*
_output_shapes
: 
�
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 
�
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
T0
�
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
conv2d_2/kernel
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_2/kernel
�
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
g
conv2d_3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_3/Conv2DConv2DRelu_1conv2d_2/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
Y
Relu_2Reluconv2d_3/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_3/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
valueB
 *��*
dtype0
�
.conv2d_3/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *�>*
dtype0*
_output_shapes
: 
�
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_3/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
�
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:*
T0
�
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
conv2d_3/kernel
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_3/kernel
�
conv2d_3/kernel/AssignAssignconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
conv2d_3/kernel/readIdentityconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:*
T0
g
conv2d_4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_4/Conv2DConv2DRelu_2conv2d_3/kernel/read*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Y
Relu_3Reluconv2d_4/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_4/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *HY�*
dtype0*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *HY>*
dtype0*
_output_shapes
: 
�
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_4/kernel*
seed2 *
dtype0
�
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
T0
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
�
conv2d_4/kernel
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_4/kernel*
	container *
shape:
�
conv2d_4/kernel/AssignAssignconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel
�
conv2d_4/kernel/readIdentityconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
T0
g
conv2d_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_5/Conv2DConv2DRelu_3conv2d_4/kernel/read*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Y
Relu_4Reluconv2d_5/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_5/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_5/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *d��*
dtype0*
_output_shapes
: 
�
.conv2d_5/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel*
valueB
 *d�=
�
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@conv2d_5/kernel*
seed2 *
dtype0*&
_output_shapes
:
�
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
�
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
�
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
�
conv2d_5/kernel
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_5/kernel*
	container 
�
conv2d_5/kernel/AssignAssignconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
g
conv2d_6/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_6/Conv2DConv2DRelu_4conv2d_5/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
Y
Relu_5Reluconv2d_6/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_6/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_6/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_6/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_6/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
.conv2d_6/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_6/kernel*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
8conv2d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_6/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_6/kernel*
seed2 
�
.conv2d_6/kernel/Initializer/random_uniform/subSub.conv2d_6/kernel/Initializer/random_uniform/max.conv2d_6/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_6/kernel*
_output_shapes
: *
T0
�
.conv2d_6/kernel/Initializer/random_uniform/mulMul8conv2d_6/kernel/Initializer/random_uniform/RandomUniform.conv2d_6/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
�
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:*
T0
�
conv2d_6/kernel
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_6/kernel*
	container 
�
conv2d_6/kernel/AssignAssignconv2d_6/kernel*conv2d_6/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_6/kernel/readIdentityconv2d_6/kernel*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
g
conv2d_7/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
Y
Relu_6Reluconv2d_7/Conv2D*/
_output_shapes
:���������*
T0
^
Reshape/shapeConst*
_output_shapes
:*
valueB"����\  *
dtype0
j
ReshapeReshapeRelu_6Reshape/shape*(
_output_shapes
:����������
*
T0*
Tshape0
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"\  d   *
dtype0*
_output_shapes
:
�
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *����*
dtype0*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *���=
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@dense/kernel*
seed2 *
dtype0*
_output_shapes
:	�
d
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel*
_output_shapes
:	�
d*
T0
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	�
d*
T0*
_class
loc:@dense/kernel
�
dense/kernel
VariableV2*
dtype0*
_output_shapes
:	�
d*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	�
d
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	�
d
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d
�
dense/MatMulMatMulReshapedense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
N
Relu_7Reludense/MatMul*'
_output_shapes
:���������d*
T0
�
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
valueB"d   
   *
dtype0*
_output_shapes
:
�
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
valueB
 *�'o�*
dtype0*
_output_shapes
: 
�
-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
valueB
 *�'o>*
dtype0*
_output_shapes
: 
�
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:d
*

seed *
T0*!
_class
loc:@dense_1/kernel
�
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
�
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:d
*
T0*!
_class
loc:@dense_1/kernel
�
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:d
*
T0*!
_class
loc:@dense_1/kernel
�
dense_1/kernel
VariableV2*
	container *
shape
:d
*
dtype0*
_output_shapes

:d
*
shared_name *!
_class
loc:@dense_1/kernel
�
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:d
*
use_locking(
{
dense_1/kernel/readIdentitydense_1/kernel*
_output_shapes

:d
*
T0*!
_class
loc:@dense_1/kernel
�
dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
dense_1/bias
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@dense_1/bias*
	container 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:

q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

�
dense_2/MatMulMatMulRelu_7dense_1/kernel/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

h
logistic_loss/zeros_like	ZerosLikedense_2/BiasAdd*
T0*'
_output_shapes
:���������

�
logistic_loss/GreaterEqualGreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:���������

�
logistic_loss/SelectSelectlogistic_loss/GreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:���������

[
logistic_loss/NegNegdense_2/BiasAdd*
T0*'
_output_shapes
:���������

�
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negdense_2/BiasAdd*'
_output_shapes
:���������
*
T0
j
logistic_loss/mulMuldense_2/BiasAddPlaceholder_1*
T0*'
_output_shapes
:���������

s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*'
_output_shapes
:���������
*
T0
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:���������

a
logistic_loss/Log1pLog1plogistic_loss/Exp*'
_output_shapes
:���������
*
T0
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:���������

V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
`
MeanMeanlogistic_lossConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
X
conv_loss/tagsConst*
valueB B	conv_loss*
dtype0*
_output_shapes
: 
Q
	conv_lossScalarSummaryconv_loss/tagsMean*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
f
gradients/Mean_grad/ShapeShapelogistic_loss*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������

h
gradients/Mean_grad/Shape_1Shapelogistic_loss*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/ConstConst*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Const_1Const*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:���������

s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0*
_output_shapes
:
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
�
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
Tshape0*'
_output_shapes
:���������
*
T0
�
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
�
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:���������

�
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:���������

z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*
_output_shapes
:
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0*
_output_shapes
:
�
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
�
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
�
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape
�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:���������

�
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:���������

�
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*'
_output_shapes
:���������
*
T0
�
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������

~
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikedense_2/BiasAdd*'
_output_shapes
:���������
*
T0
�
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:���������

�
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������

�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
�
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������

�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������

u
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
u
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
�
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/mul_grad/mulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Placeholder_1*'
_output_shapes
:���������
*
T0
�
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
�
&gradients/logistic_loss/mul_grad/mul_1Muldense_2/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������

�
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:���������
*
T0
�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
�
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:���������

�
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:���������

�
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*'
_output_shapes
:���������
*
T0
�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:���������

�
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*'
_output_shapes
:���������
*
T0
�
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:���������

�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
�
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select
�
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1
�
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*'
_output_shapes
:���������
*
T0
�
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:���������

�
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
_output_shapes
:
*
T0
u
/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN+^gradients/dense_2/BiasAdd_grad/BiasAddGrad
�
7gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
�
9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_2/BiasAdd_grad/BiasAddGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

�
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
�
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_77gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:d
*
transpose_a(*
transpose_b( 
�
.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1
�
6gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_2/MatMul_grad/MatMul/^gradients/dense_2/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_2/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:d

�
gradients/Relu_7_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_7*
T0*'
_output_shapes
:���������d
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_7_grad/ReluGraddense/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������
*
transpose_a( 
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/Relu_7_grad/ReluGrad*
T0*
_output_shapes
:	�
d*
transpose_a(*
transpose_b( 
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������
*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	�
d*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1
b
gradients/Reshape_grad/ShapeShapeRelu_6*
_output_shapes
:*
T0*
out_type0
�
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
gradients/Relu_6_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_6*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_7/Conv2D_grad/ShapeNShapeNRelu_5conv2d_6/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
�
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_5'gradients/conv2d_7/Conv2D_grad/ShapeN:1gradients/Relu_6_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
/gradients/conv2d_7/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*F
_class<
:8loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
�
gradients/Relu_5_grad/ReluGradReluGrad7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyRelu_5*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_6/Conv2D_grad/ShapeNShapeNRelu_4conv2d_5/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
�
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4'gradients/conv2d_6/Conv2D_grad/ShapeN:1gradients/Relu_5_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
/gradients/conv2d_6/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_6/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_6/Conv2D_grad/tuple/group_deps*E
_class;
97loc:@gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
�
9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_6/Conv2D_grad/tuple/group_deps*F
_class<
:8loc:@gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
�
gradients/Relu_4_grad/ReluGradReluGrad7gradients/conv2d_6/Conv2D_grad/tuple/control_dependencyRelu_4*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_5/Conv2D_grad/ShapeNShapeNRelu_3conv2d_4/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
�
2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_4/kernel/readgradients/Relu_4_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
/gradients/conv2d_5/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_5/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
gradients/Relu_3_grad/ReluGradReluGrad7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyRelu_3*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNRelu_2conv2d_3/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
�
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/Relu_3_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2'gradients/conv2d_4/Conv2D_grad/ShapeN:1gradients/Relu_3_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
�
/gradients/conv2d_4/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*E
_class;
97loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput
�
9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter
�
gradients/Relu_2_grad/ReluGradReluGrad7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyRelu_2*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0
�
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/Relu_2_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
�
/gradients/conv2d_3/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
gradients/Relu_1_grad/ReluGradReluGrad7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyRelu_1*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
�
/gradients/conv2d_2/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_2/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*E
_class;
97loc:@gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput
�
9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_2/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter
�
gradients/Relu_grad/ReluGradReluGrad7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyRelu*/
_output_shapes
:���������*
T0
�
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/Relu_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
�
5gradients/conv2d/Conv2D_grad/tuple/control_dependencyIdentity0gradients/conv2d/Conv2D_grad/Conv2DBackpropInput.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1Identity1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
b
GradientDescent/learning_rateConst*
valueB
 *o;*
dtype0*
_output_shapes
: 
�
9GradientDescent/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelGradientDescent/learning_rate7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentApplyGradientDescentconv2d_3/kernelGradientDescent/learning_rate9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
use_locking( *
T0
�
;GradientDescent/update_conv2d_5/kernel/ApplyGradientDescentApplyGradientDescentconv2d_5/kernelGradientDescent/learning_rate9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_6/kernel/ApplyGradientDescentApplyGradientDescentconv2d_6/kernelGradientDescent/learning_rate9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�
d*
use_locking( *
T0*
_class
loc:@dense/kernel
�
:GradientDescent/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelGradientDescent/learning_rate8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d
*
use_locking( 
�
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_3/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_4/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_5/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_6/kernel/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
ArgMaxArgMaxdense_2/BiasAddArgMax/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
P
CastCastEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_1MeanCastConst_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: ""�
trainable_variables��
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
q
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:0
q
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02,conv2d_3/kernel/Initializer/random_uniform:0
q
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02,conv2d_4/kernel/Initializer/random_uniform:0
q
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02,conv2d_5/kernel/Initializer/random_uniform:0
q
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02,conv2d_6/kernel/Initializer/random_uniform:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"
	summaries

conv_loss:0"
train_op

GradientDescent"�
	variables��
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
q
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:0
q
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02,conv2d_3/kernel/Initializer/random_uniform:0
q
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02,conv2d_4/kernel/Initializer/random_uniform:0
q
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02,conv2d_5/kernel/Initializer/random_uniform:0
q
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02,conv2d_6/kernel/Initializer/random_uniform:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0ԙDg       `/�#	�f�*���A*

	conv_loss��0?ze9J       QKD	���*���A*

	conv_loss�0?h�E�       QKD	���*���A*

	conv_loss�0?�	�       QKD	�/�*���A*

	conv_loss]�0?���R       QKD	h�*���A*

	conv_loss��0?��4�       QKD	ܚ�*���A*

	conv_loss��0?���       QKD	9��*���A*

	conv_loss��0?		�       QKD	��*���A*

	conv_loss9�0?c:=�       QKD	5�*���A*

	conv_loss �0?v�K�       QKD	�l�*���A	*

	conv_loss��0?̮F       QKD	A��*���A
*

	conv_loss��0?M�#�       QKD	V��*���A*

	conv_loss<�0?�L��       QKD	�*���A*

	conv_loss��0?}���       QKD	KC�*���A*

	conv_loss1�0?����       QKD	v�*���A*

	conv_lossU�0?RH=       QKD	}��*���A*

	conv_loss%�0?���       QKD	���*���A*

	conv_loss��0?����       QKD	��*���A*

	conv_loss��0?��U       QKD	�R�*���A*

	conv_loss��0?�1��       QKD	m��*���A*

	conv_loss��0?y��       QKD	���*���A*

	conv_loss��0?b��       QKD	���*���A*

	conv_loss��0?g�Y       QKD	V!�*���A*

	conv_loss4�0?n
�L       QKD	?W�*���A*

	conv_loss��0?XC��       QKD	d��*���A*

	conv_loss"�0?���       QKD	���*���A*

	conv_loss��0?���       QKD	���*���A*

	conv_loss��0?G�H�       QKD	�*���A*

	conv_loss�0?��7�       QKD	P�*���A*

	conv_loss��0?�"|�       QKD	p��*���A*

	conv_loss�0?,Sa�       QKD	���*���A*

	conv_loss��0?��с       QKD	���*���A*

	conv_loss��0?S�u�       QKD	�*���A *

	conv_loss��0?����       QKD	mP�*���A!*

	conv_loss��0?k��9       QKD	���*���A"*

	conv_lossƙ0?��S�       QKD	ѵ�*���A#*

	conv_loss´0?i��       QKD	��*���A$*

	conv_lossF�0?��!`       QKD	��*���A%*

	conv_lossY�0?p��       QKD	sF�*���A&*

	conv_loss��0?&G�.       QKD	_v�*���A'*

	conv_lossc�0?���       QKD	���*���A(*

	conv_loss�0?[|��       QKD	h��*���A)*

	conv_loss��0?�?'       QKD	��*���A**

	conv_loss�0?��:�       QKD	�5�*���A+*

	conv_loss�0?��ю       QKD	�l�*���A,*

	conv_loss.�0?zj/&       QKD	��*���A-*

	conv_loss��0?����       QKD	���*���A.*

	conv_losso�0?X���       QKD	� �*���A/*

	conv_lossF�0?�=��       QKD	oV�*���A0*

	conv_loss��0?�d[r       QKD	���*���A1*

	conv_loss�~0?�aj$       QKD	U��*���A2*

	conv_loss�0?M���       QKD	�$�*���A3*

	conv_lossa�0?¥)�       QKD	9X�*���A4*

	conv_loss%�0?���W       QKD	��*���A5*

	conv_loss�f0?��A       QKD	 ��*���A6*

	conv_loss^f0?���C       QKD	&��*���A7*

	conv_loss�{0?���       QKD	=�*���A8*

	conv_loss�|0?��e�       QKD	[o�*���A9*

	conv_lossap0?�^A�       QKD	9��*���A:*

	conv_lossVR0?h=�       QKD	s��*���A;*

	conv_lossr_0?[2��       QKD	u�*���A<*

	conv_lossDc0?}GX�       QKD		I�*���A=*

	conv_loss�Q0?���       QKD	�x�*���A>*

	conv_loss�Y0?'�B       QKD	���*���A?*

	conv_losscM0?^хs       QKD	q��*���A@*

	conv_lossPr0?Z(�	       QKD	B�*���AA*

	conv_loss�K0?� F       QKD	�\�*���AB*

	conv_lossW0?�gIt       QKD	̌�*���AC*

	conv_lossA0?^~_Y       QKD	Ž�*���AD*

	conv_loss�?0?[t��       QKD	���*���AE*

	conv_loss�K0?����       QKD	��*���AF*

	conv_lossA0?��       QKD	�R�*���AG*

	conv_lossj80?�$a       QKD	���*���AH*

	conv_loss�?0?��e       QKD	���*���AI*

	conv_loss�20?���_       QKD	/��*���AJ*

	conv_loss�I0?��$t       QKD	��*���AK*

	conv_loss�C0?���       QKD	�^�*���AL*

	conv_lossA)0?���n       QKD	Y��*���AM*

	conv_loss\;0?
a�       QKD	
��*���AN*

	conv_loss�/0?@K�       QKD	���*���AO*

	conv_loss#0?׽�       QKD	~*�*���AP*

	conv_loss�40?��{       QKD	Kf�*���AQ*

	conv_loss�+0?d�$1       QKD	���*���AR*

	conv_loss�10?'/�p       QKD	���*���AS*

	conv_lossK&0?�3U�       QKD	��*���AT*

	conv_loss�&0?�d�       QKD	,�*���AU*

	conv_loss60?ME �       QKD	L��*���AV*

	conv_loss�!0?p1B�       QKD	 ��*���AW*

	conv_loss�(0?��P       QKD	��*���AX*

	conv_loss� 0?���       QKD	�B�*���AY*

	conv_loss�&0?�C�=       QKD	�z�*���AZ*

	conv_loss�"0?zf�       QKD	��*���A[*

	conv_loss80?\��       QKD	B��*���A\*

	conv_loss�0?Qy�       QKD	��*���A]*

	conv_loss�0?�+h       QKD	�;�*���A^*

	conv_loss0?ܮ�       QKD	[p�*���A_*

	conv_loss�0?��]       QKD	���*���A`*

	conv_loss�0?T!�       QKD	W��*���Aa*

	conv_loss�0?�2��       QKD	���*���Ab*

	conv_loss�0?M"�K       QKD	z2�*���Ac*

	conv_loss��/?j��       QKD	�i�*���Ad*

	conv_loss�/?G㭝       QKD	��*���Ae*

	conv_loss��/?��O/       QKD	���*���Af*

	conv_loss��/?�攡       QKD	@�*���Ag*

	conv_loss��/?��=       QKD	NG�*���Ah*

	conv_loss0?�;k�       QKD	��*���Ai*

	conv_loss�/?��x�       QKD	T��*���Aj*

	conv_loss��/?o>�       QKD	���*���Ak*

	conv_loss��/?�.e       QKD	Z�*���Al*

	conv_loss=�/?5�u       QKD	K�*���Am*

	conv_loss��/?��<       QKD	���*���An*

	conv_loss
�/?��-       QKD	Ҷ�*���Ao*

	conv_loss��/?��1       QKD	���*���Ap*

	conv_loss�/?�m�y       QKD	(�*���Aq*

	conv_loss�/?�L�       QKD	�Y�*���Ar*

	conv_loss0�/?b,�n       QKD	���*���As*

	conv_loss��/?�|�       QKD	���*���At*

	conv_loss��/?��       QKD	���*���Au*

	conv_loss��/?��t�       QKD	z�*���Av*

	conv_loss��/?ϙV�       QKD	�L�*���Aw*

	conv_loss��/?�s��       QKD	H��*���Ax*

	conv_loss��/?L�Sx       QKD	��*���Ay*

	conv_lossY�/?i���       QKD	���*���Az*

	conv_loss��/?���       QKD	��*���A{*

	conv_loss$�/?��       QKD	�C�*���A|*

	conv_loss��/? �       QKD	f}�*���A}*

	conv_loss�/?� 4�       QKD	̰�*���A~*

	conv_loss^�/?�+]�       QKD	���*���A*

	conv_lossR�/?~�	        )��P	��*���A�*

	conv_lossh�/?P8`�        )��P	/H�*���A�*

	conv_loss��/?=���        )��P	�z�*���A�*

	conv_loss
�/?�rX�        )��P	���*���A�*

	conv_loss4�/?5O>�        )��P	R��*���A�*

	conv_loss(�/?c�Q�        )��P	�*���A�*

	conv_loss�/?�_r        )��P	>G�*���A�*

	conv_loss��/?]w�E        )��P	�y�*���A�*

	conv_loss2�/?G���        )��P	��*���A�*

	conv_loss	�/?�)W         )��P	���*���A�*

	conv_loss7�/?����        )��P	�*���A�*

	conv_lossp�/?��w�        )��P	 N�*���A�*

	conv_loss�/?�_�2        )��P	<�*���A�*

	conv_loss�/?%C�        )��P	��*���A�*

	conv_loss<�/?�L��        )��P	'��*���A�*

	conv_loss��/?&'f        )��P	��*���A�*

	conv_loss5�/?5��8        )��P	�D�*���A�*

	conv_loss��/?^�3        )��P	�s�*���A�*

	conv_lossv�/?ôQ.        )��P	{��*���A�*

	conv_loss��/?���        )��P	R��*���A�*

	conv_lossz/?;L#p        )��P	j��*���A�*

	conv_loss&�/?�Y�        )��P	�-�*���A�*

	conv_loss�{/?6��W        )��P	�_�*���A�*

	conv_loss�l/?u[�        )��P	���*���A�*

	conv_loss p/?B�!�        )��P	q��*���A�*

	conv_loss|/?��s�        )��P	��*���A�*

	conv_lossw�/?!k��        )��P	�C�*���A�*

	conv_loss��/?�        )��P	5u�*���A�*

	conv_loss�t/?��        )��P	ɤ�*���A�*

	conv_loss�^/?�°�        )��P	���*���A�*

	conv_loss u/?��3I        )��P	N�*���A�*

	conv_loss�q/?�/Ef        )��P	<D�*���A�*

	conv_lossv/?�N�F        )��P	L{�*���A�*

	conv_loss�V/?E��        )��P	��*���A�*

	conv_loss�b/?~���        )��P	���*���A�*

	conv_loss�g/?�K�        )��P	�*���A�*

	conv_loss�s/?2t�s        )��P	pK�*���A�*

	conv_lossXn/?nX�        )��P	�{�*���A�*

	conv_loss�x/?Zx68        )��P	���*���A�*

	conv_loss*W/?1�{�        )��P	i��*���A�*

	conv_loss�[/?�t�        )��P	��*���A�*

	conv_loss�\/?��n�        )��P	>:�*���A�*

	conv_loss]E/?Bǰ�        )��P	ri�*���A�*

	conv_loss�P/?�@�        )��P	ߛ�*���A�*

	conv_lossFQ/?��<	        )��P	z��*���A�*

	conv_loss�P/?�y#        )��P	q��*���A�*

	conv_loss�T/?���        )��P	�-�*���A�*

	conv_loss"I/?N�YY        )��P	�\�*���A�*

	conv_loss�+/?E�~        )��P	 ��*���A�*

	conv_lossS+/?� ��        )��P	���*���A�*

	conv_loss+2/?�9�|        )��P	/��*���A�*

	conv_lossX//?��+        )��P	��*���A�*

	conv_loss� /?�as�        )��P	�H�*���A�*

	conv_loss.C/?+��        )��P	�w�*���A�*

	conv_loss�!/?P�Mb        )��P	2��*���A�*

	conv_lossR/?U��!        )��P	���*���A�*

	conv_loss9@/?:�I        )��P	�*���A�*

	conv_loss�(/?G՚}        )��P	T7�*���A�*

	conv_lossf/?j&        )��P	\g�*���A�*

	conv_loss�/?v`b�        )��P	��*���A�*

	conv_loss2/?8=         )��P	[��*���A�*

	conv_loss�2/?6���        )��P	���*���A�*

	conv_loss�/?���D        )��P	T'�*���A�*

	conv_loss/?\;��        )��P	�V�*���A�*

	conv_loss�/?�OI�        )��P	��*���A�*

	conv_loss/?͔�        )��P	U��*���A�*

	conv_loss/?!^�k        )��P	1��*���A�*

	conv_loss��.?�8�        )��P	(�*���A�*

	conv_loss�/?��        )��P	^H�*���A�*

	conv_loss7/?`:�q        )��P	Kz�*���A�*

	conv_lossF/?��G�        )��P	��*���A�*

	conv_loss�/?4t��        )��P	Q��*���A�*

	conv_lossp/?��a/        )��P	�*���A�*

	conv_loss
/?���_        )��P	ћ�*���A�*

	conv_loss��.?R�        )��P	���*���A�*

	conv_loss�.?d�F        )��P	���*���A�*

	conv_loss{�.?oľ        )��P	�/�*���A�*

	conv_loss �.?�ń�        )��P	`�*���A�*

	conv_loss �.?��,D        )��P	���*���A�*

	conv_loss}/?�Kԩ        )��P	��*���A�*

	conv_lossG�.?���        )��P	���*���A�*

	conv_loss��.?wB��        )��P	�5�*���A�*

	conv_lossq�.?8io5        )��P	}q�*���A�*

	conv_loss]�.?;��        )��P	ʢ�*���A�*

	conv_loss��.?��;        )��P	r��*���A�*

	conv_loss��.?��v        )��P	L�*���A�*

	conv_loss��.?ٗ�-        )��P	�:�*���A�*

	conv_loss��.?�^�        )��P	�h�*���A�*

	conv_loss��.?/7�!        )��P	���*���A�*

	conv_loss'�.?06n        )��P	���*���A�*

	conv_loss��.?)a�        )��P	���*���A�*

	conv_loss��.?�I�        )��P	t(�*���A�*

	conv_loss��.?�ݳ%        )��P	TW�*���A�*

	conv_loss�.?�"
        )��P	���*���A�*

	conv_loss��.?�,��        )��P	���*���A�*

	conv_loss��.?]a{        )��P	���*���A�*

	conv_lossJ�.?��        )��P	P'�*���A�*

	conv_lossr�.?Hx�C        )��P	�[�*���A�*

	conv_loss��.?6�a�        )��P	f��*���A�*

	conv_loss��.?R.�        )��P	���*���A�*

	conv_lossz�.?�T�        )��P	���*���A�*

	conv_lossW�.?@cxH        )��P	W�*���A�*

	conv_loss��.?��Τ        )��P	�L�*���A�*

	conv_lossx�.?���.        )��P	&{�*���A�*

	conv_loss*�.?��<        )��P	���*���A�*

	conv_loss<�.?�t        )��P	���*���A�*

	conv_lossm�.?�^N�        )��P	��*���A�*

	conv_loss*�.?0kj�        )��P	�A�*���A�*

	conv_loss��.?1��        )��P	q�*���A�*

	conv_loss �.?��,�        )��P	͠�*���A�*

	conv_loss!�.?�j�1        )��P	��*���A�*

	conv_loss�.?2�{        )��P	*��*���A�*

	conv_loss��.?�w�        )��P	#1�*���A�*

	conv_loss��.?�[�`        )��P	�h�*���A�*

	conv_lossҋ.?��t        )��P	z��*���A�*

	conv_loss�.?��[�        )��P	Y��*���A�*

	conv_lossg�.?�h��        )��P	���*���A�*

	conv_loss�}.?�&        )��P	u-�*���A�*

	conv_loss^�.?�)�        )��P	"_�*���A�*

	conv_loss�.?ԍ�e        )��P	%��*���A�*

	conv_loss>�.?�6�        )��P	y��*���A�*

	conv_loss��.?u�n�        )��P	J��*���A�*

	conv_lossr.?���        )��P	�!�*���A�*

	conv_lossZ|.?t�B        )��P	%e�*���A�*

	conv_loss��.?8Z'�        )��P	���*���A�*

	conv_loss$�.?`V�        )��P	4��*���A�*

	conv_loss�.?N޹I        )��P	V��*���A�*

	conv_loss�}.?�S��        )��P	�'�*���A�*

	conv_losss^.?'�,�        )��P	e�*���A�*

	conv_loss�.?ܐ�        )��P	��*���A�*

	conv_loss�j.?ˮ2        )��P	���*���A�*

	conv_loss�n.?���F        )��P	� �*���A�*

	conv_lossch.?m%^X        )��P	�7�*���A�*

	conv_loss�P.?���D        )��P	
t�*���A�*

	conv_lossb].?R9�        )��P	5��*���A�*

	conv_loss�R.?��r1        )��P	��*���A�*

	conv_loss�d.? ٯ        )��P	`�*���A�*

	conv_loss�\.?`�UW        )��P	�<�*���A�*

	conv_loss�S.?���        )��P	��*���A�*

	conv_loss�N.?�H��        )��P	Ͱ�*���A�*

	conv_lossrS.?�}        )��P	���*���A�*

	conv_loss)h.?��        )��P	�*���A�*

	conv_loss�`.?2��
        )��P	)A�*���A�*

	conv_loss�S.?��        )��P	�o�*���A�*

	conv_lossZ.?>"4�        )��P	��*���A�*

	conv_loss:].?Gn�        )��P	.��*���A�*

	conv_lossT.?3��        )��P	� +���A�*

	conv_loss�I.?KH��        )��P	z9 +���A�*

	conv_lossY.?�ܫ|        )��P	9y +���A�*

	conv_loss�-.?��p        )��P	(� +���A�*

	conv_loss�9.?����        )��P	�� +���A�*

	conv_loss|8.?�E�1        )��P	(+���A�*

	conv_loss�=.?���        )��P	�9+���A�*

	conv_loss�".?K��L        )��P	Mi+���A�*

	conv_losscC.?ˋ��        )��P	��+���A�*

	conv_loss�).?�#��        )��P	��+���A�*

	conv_loss�#.?��^6        )��P	3�+���A�*

	conv_loss?'.?>���        )��P	d.+���A�*

	conv_lossc1.?4��        )��P	�^+���A�*

	conv_loss�.?�Ǎ�        )��P	8�+���A�*

	conv_loss\.?m�B        )��P	m�+���A�*

	conv_loss�*.?e���        )��P	��+���A�*

	conv_loss�(.?sp�        )��P	� +���A�*

	conv_loss�.?�
�h        )��P	QS+���A�*

	conv_loss2.?��?8        )��P	r�+���A�*

	conv_loss
.?ӧ�        )��P	��+���A�*

	conv_loss.?���z        )��P	��+���A�*

	conv_loss�$.?<��        )��P	 +���A�*

	conv_loss�.?�h�        )��P	P+���A�*

	conv_loss��-?�mH        )��P	�+���A�*

	conv_loss�-?e�[&        )��P	;�+���A�*

	conv_loss�.?�MV�        )��P	��+���A�*

	conv_loss'.?see        )��P	�+���A�*

	conv_loss��-?�s��        )��P	�P+���A�*

	conv_loss�
.?�N�        )��P	�+���A�*

	conv_loss��-?mjrR        )��P	�+���A�*

	conv_loss��-?C�N        )��P	��+���A�*

	conv_lossV
.?8"�L        )��P	+���A�*

	conv_lossX�-?%���        )��P	�E+���A�*

	conv_loss��-?�F��        )��P	�u+���A�*

	conv_lossB	.?ak&�        )��P	T�+���A�*

	conv_loss��-??��        )��P	��+���A�*

	conv_loss��-?0V        )��P	~+���A�*

	conv_loss��-?��L�        )��P	|W+���A�*

	conv_lossc�-?��        )��P	��+���A�*

	conv_lossQ�-?1C6�        )��P	�+���A�*

	conv_loss0�-?��        )��P	\�+���A�*

	conv_loss��-?!��        )��P	�*+���A�*

	conv_loss��-?�2�        )��P	�[+���A�*

	conv_loss��-?K1        )��P	E�+���A�*

	conv_loss�-?��z�        )��P	X�+���A�*

	conv_loss��-?2��        )��P	��+���A�*

	conv_loss�-?mx��        )��P	y&	+���A�*

	conv_loss��-?�SO        )��P	�Z	+���A�*

	conv_lossD�-?��`L        )��P	e�	+���A�*

	conv_loss��-?vY��        )��P	V�	+���A�*

	conv_lossu�-?Äv�        )��P	m�	+���A�*

	conv_loss��-?�"�-        )��P	Y
+���A�*

	conv_lossw�-?
a�"        )��P	�M
+���A�*

	conv_loss��-?�~�        )��P	4~
+���A�*

	conv_loss�-?�SR�        )��P	v�
+���A�*

	conv_loss�-?���        )��P	��
+���A�*

	conv_loss~�-?9R        )��P	8+���A�*

	conv_loss$�-?	v�        )��P	�M+���A�*

	conv_loss��-?�u�7        )��P	;~+���A�*

	conv_lossњ-?�X&W        )��P	��+���A�*

	conv_loss��-?:#�        )��P	��+���A�*

	conv_loss=�-?�SL;        )��P	;+���A�*

	conv_loss³-?,�K        )��P	�B+���A�*

	conv_loss�-?~&8        )��P	Av+���A�*

	conv_loss"�-?`D�        )��P	�+���A�*

	conv_lossI�-?xGj        )��P	�+���A�*

	conv_lossޖ-?�h��        )��P	�+���A�*

	conv_loss|�-?MS`�        )��P	7+���A�*

	conv_loss/�-?���/        )��P	'i+���A�*

	conv_loss��-?#��        )��P	��+���A�*

	conv_loss�{-?'�j        )��P	
�+���A�*

	conv_loss*z-?��=        )��P	�+���A�*

	conv_loss�v-?�GpI        )��P	�@+���A�*

	conv_loss�x-?���K        )��P	r+���A�*

	conv_loss�-?�z�        )��P	�+���A�*

	conv_loss�w-?pHF        )��P	��+���A�*

	conv_lossV�-?��>        )��P	+���A�*

	conv_lossIy-?�ЦF        )��P	=[+���A�*

	conv_lossvr-?MLx�        )��P	�+���A�*

	conv_loss�f-?�4�         )��P	�+���A�*

	conv_loss�\-?p��?        )��P	��+���A�*

	conv_loss_l-?O+�o        )��P	�&+���A�*

	conv_loss�Q-?���j        )��P	5X+���A�*

	conv_loss�u-?�ӎ�        )��P	��+���A�*

	conv_loss�c-?36b�        )��P	��+���A�*

	conv_lossXC-?�{��        )��P	�+���A�*

	conv_loss�[-?}/�>        )��P	�)+���A�*

	conv_loss�T-?�߰        )��P	�b+���A�*

	conv_loss{T-?�2�        )��P	��+���A�*

	conv_loss4-?J�Ò        )��P	�+���A�*

	conv_lossO-?���~        )��P	'+���A�*

	conv_loss"Z-?>b��        )��P	�8+���A�*

	conv_loss�S-?{��        )��P	Cj+���A�*

	conv_lossY^-?�ذ�        )��P	��+���A�*

	conv_lossS-?�E/�        )��P	��+���A�*

	conv_loss�C-?�w��        )��P	1+���A�*

	conv_loss�a-?F���        )��P	18+���A�*

	conv_loss?O-?}D-9        )��P	j+���A�*

	conv_loss4Y-?���]        )��P	��+���A�*

	conv_loss�:-?��jM        )��P	9�+���A�*

	conv_loss�H-?�F        )��P	+���A�*

	conv_loss8-?񅄘        )��P	�4+���A�*

	conv_lossA=-?��`        )��P	Me+���A�*

	conv_lossZM-?Q�[        )��P	D�+���A�*

	conv_loss0<-?]�9:        )��P	`�+���A�*

	conv_loss� -?�~I        )��P	9+���A�*

	conv_loss�--?����        )��P	�5+���A�*

	conv_losss&-?8>�A        )��P	�h+���A�*

	conv_lossy2-?����        )��P	L�+���A�*

	conv_loss�!-?��#        )��P	��+���A�*

	conv_loss�-?mP{+        )��P	#�+���A�*

	conv_loss$-?}��        )��P	�/+���A�*

	conv_loss�-?\�        )��P	�b+���A�*

	conv_loss��,?���        )��P	T�+���A�*

	conv_loss�#-?ڣ}g        )��P	��+���A�*

	conv_lossP-?�*.�        )��P	�+���A�*

	conv_loss�-?RI"        )��P	�0+���A�*

	conv_lossQ�,?��a        )��P	�e+���A�*

	conv_loss5-?(=n�        )��P	��+���A�*

	conv_loss~-?"��        )��P	m�+���A�*

	conv_loss�-?��ɻ        )��P	��+���A�*

	conv_loss�-?Ԯb�        )��P	~++���A�*

	conv_loss��,?`]��        )��P	�[+���A�*

	conv_loss��,?qU�j        )��P	��+���A�*

	conv_lossv�,?R�        )��P	I�+���A�*

	conv_lossd-?��R�        )��P	�+���A�*

	conv_loss��,?3Ie�        )��P	�&+���A�*

	conv_lossd�,?�Ui        )��P	��+���A�*

	conv_lossX�,?A�&�        )��P	5�+���A�*

	conv_loss2�,?��        )��P	E+���A�*

	conv_loss��,?���u        )��P	�M+���A�*

	conv_loss��,?bt?�        )��P	O~+���A�*

	conv_loss��,?����        )��P	��+���A�*

	conv_loss��,?���        )��P	C�+���A�*

	conv_loss��,?q�r�        )��P	�+���A�*

	conv_loss��,?�ek        )��P	'K+���A�*

	conv_loss�,?�,$        )��P	f+���A�*

	conv_loss��,?3�"        )��P	į+���A�*

	conv_loss �,?4O[�        )��P	��+���A�*

	conv_loss��,?D�16        )��P	H+���A�*

	conv_loss��,?,�        )��P	iH+���A�*

	conv_loss��,?��4�        )��P	8x+���A�*

	conv_loss<�,?Wc        )��P	�+���A�*

	conv_lossǵ,?i?��        )��P	��+���A�*

	conv_loss��,?�}�        )��P	�+���A�*

	conv_lossz�,?Y��Q        )��P	�I+���A�*

	conv_loss�,?@��        )��P	 {+���A�*

	conv_loss��,?3!��        )��P	f�+���A�*

	conv_loss�,?��A$        )��P	��+���A�*

	conv_loss��,?���        )��P	�+���A�*

	conv_loss<�,?���\        )��P	�E+���A�*

	conv_loss˚,?�X�        )��P	^y+���A�*

	conv_loss\�,?\F�O        )��P	��+���A�*

	conv_loss�,?����        )��P	��+���A�*

	conv_loss(�,?��2        )��P	t +���A�*

	conv_loss>�,?:F�        )��P	~E +���A�*

	conv_losss�,?97ڌ        )��P	�v +���A�*

	conv_lossb�,?F�        )��P	Ԩ +���A�*

	conv_losse�,?�k�        )��P	�� +���A�*

	conv_loss��,?�*�$        )��P	 	!+���A�*

	conv_loss��,?�w�        )��P	=!+���A�*

	conv_loss,m,?̭c        )��P	�m!+���A�*

	conv_lossL�,?4���        )��P	2�!+���A�*

	conv_loss��,?�m        )��P	��!+���A�*

	conv_loss��,?-,        )��P	"�!+���A�*

	conv_lossF�,?kD
-        )��P	�,"+���A�*

	conv_lossXd,?�{�_        )��P	�f"+���A�*

	conv_loss\~,?ἕN        )��P	\�"+���A�*

	conv_lossCr,?�w�W        )��P	��"+���A�*

	conv_loss��,?�ZE*        )��P	��"+���A�*

	conv_loss�,?�kTh        )��P	�4#+���A�*

	conv_loss@V,?�~�4        )��P	�g#+���A�*

	conv_loss�s,?�p1U        )��P	��#+���A�*

	conv_lossp},?�{�R        )��P	l�#+���A�*

	conv_loss�k,?U���        )��P	��#+���A�*

	conv_loss�Z,?Կc        )��P	�*$+���A�*

	conv_lossRx,?��w�        )��P	�[$+���A�*

	conv_lossPd,?��́        )��P	��$+���A�*

	conv_loss�d,?��>�        )��P	��$+���A�*

	conv_lossZ,?ݷ~Z        )��P	��$+���A�*

	conv_loss�E,?�
e        )��P	�/%+���A�*

	conv_loss�d,?�F        )��P	�f%+���A�*

	conv_lossx;,?��        )��P	�%+���A�*

	conv_loss?X,?f>�y        )��P	r�%+���A�*

	conv_loss�U,?�h�        )��P	�&+���A�*

	conv_loss�B,?q�r�        )��P	&8&+���A�*

	conv_loss�3,?�U        )��P	hn&+���A�*

	conv_loss�9,?�,"�        )��P	��&+���A�*

	conv_lossif,?jPc        )��P	
�&+���A�*

	conv_loss�6,?���        )��P	��&+���A�*

	conv_loss]T,?�f@p        )��P	�.'+���A�*

	conv_loss�C,?2�e        )��P	�l'+���A�*

	conv_loss�5,?�ޙ        )��P	r�'+���A�*

	conv_loss�F,?u{��        )��P	��'+���A�*

	conv_loss�",?F���        )��P	5
(+���A�*

	conv_lossN0,?�y�s        )��P	h>(+���A�*

	conv_loss�-,?t�V        )��P	4o(+���A�*

	conv_loss�A,?Ҩ�`        )��P	��(+���A�*

	conv_loss�*,?Oc�"        )��P	M�(+���A�*

	conv_loss�-,?����        )��P	� )+���A�*

	conv_lossr,?!k�D        )��P	1)+���A�*

	conv_loss;,?�e�        )��P	�b)+���A�*

	conv_loss�(,?���        )��P	�)+���A�*

	conv_loss� ,?�@=        )��P	`�)+���A�*

	conv_loss�,?�LC        )��P	��)+���A�*

	conv_loss�,?�d�        )��P	#*+���A�*

	conv_lossa,?$E`]        )��P	tS*+���A�*

	conv_loss�,?��        )��P	?�*+���A�*

	conv_losss,?���        )��P	<�*+���A�*

	conv_lossi,?/��=        )��P	��*+���A�*

	conv_loss>	,?�~&�        )��P	�++���A�*

	conv_loss� ,?&=E�        )��P	1H++���A�*

	conv_losss,?G�m        )��P	�w++���A�*

	conv_loss��+?@1bZ        )��P	z�++���A�*

	conv_loss��+?�!\P        )��P	��++���A�*

	conv_loss�,?�?z�        )��P	
,+���A�*

	conv_lossV�+?�@F        )��P	�;,+���A�*

	conv_loss��+?�Ar�        )��P	k,+���A�*

	conv_lossg�+?�B�s        )��P	��,+���A�*

	conv_lossQ�+?�_�        )��P	��,+���A�*

	conv_loss��+?�@ET        )��P	h-+���A�*

	conv_loss��+?M���        )��P	6-+���A�*

	conv_lossG�+?�N��        )��P	�g-+���A�*

	conv_lossn�+?4���        )��P	M�-+���A�*

	conv_loss��+?��
�        )��P	�-+���A�*

	conv_lossп+?�*b�        )��P	�-+���A�*

	conv_lossy�+?/�X�        )��P	�2+���A�*

	conv_lossq�+?T0�        )��P	��2+���A�*

	conv_loss��+?9̶�        )��P	-3+���A�*

	conv_loss�+?�.��        )��P	w\3+���A�*

	conv_loss��+?v�g-        )��P	��3+���A�*

	conv_loss��+?IQ#        )��P	��3+���A�*

	conv_lossy�+?�k        )��P	�3+���A�*

	conv_lossܫ+?H	G�        )��P	�"4+���A�*

	conv_loss�+?��qz        )��P	8P4+���A�*

	conv_loss��+?�-5�        )��P	��4+���A�*

	conv_loss��+?كJ�        )��P	g�4+���A�*

	conv_loss��+?��f�        )��P	O�4+���A�*

	conv_lossٿ+?�.}�        )��P	�/5+���A�*

	conv_loss��+?+��        )��P	�_5+���A�*

	conv_loss��+?�1q0        )��P	`�5+���A�*

	conv_loss,�+?��b        )��P	��5+���A�*

	conv_loss��+?����        )��P	�5+���A�*

	conv_lossp�+?�ؤ�        )��P	�(6+���A�*

	conv_loss1�+?F�-%        )��P	~W6+���A�*

	conv_lossHz+?!��        )��P	��6+���A�*

	conv_lossts+?dZ��        )��P	��6+���A�*

	conv_lossO�+?�IL        )��P	H�6+���A�*

	conv_lossN�+?�s�        )��P	�7+���A�*

	conv_loss0�+?�~��        )��P	uQ7+���A�*

	conv_loss��+?�3�6        )��P	�7+���A�*

	conv_loss�+?�Av        )��P	�7+���A�*

	conv_loss��+?�7n         )��P	,�7+���A�*

	conv_lossct+?:�~C        )��P	#8+���A�*

	conv_lossRv+?$��        )��P	�H8+���A�*

	conv_loss��+?����        )��P	�v8+���A�*

	conv_loss +?�d�#        )��P	J�8+���A�*

	conv_loss��+?ɚ��        )��P	��8+���A�*

	conv_loss`|+?�jJ�        )��P	�9+���A�*

	conv_loss��+?�X�l        )��P	�?9+���A�*

	conv_lossE\+?��        )��P	n9+���A�*

	conv_loss/n+?�{�        )��P	
�9+���A�*

	conv_lossJ\+?l ir        )��P	s�9+���A�*

	conv_losse+?�!ܦ        )��P	�9+���A�*

	conv_loss�J+?%��        )��P	-:+���A�*

	conv_loss6x+?���        )��P	\:+���A�*

	conv_loss�~+?�sS        )��P	n�:+���A�*

	conv_loss�`+?��ַ        )��P	ʼ:+���A�*

	conv_lossEx+?9)��        )��P	N�:+���A�*

	conv_loss�I+?��G'        )��P	�;+���A�*

	conv_loss
U+?�I,        )��P	2N;+���A�*

	conv_lossW+?�J�        )��P	�~;+���A�*

	conv_lossU+?��U�        )��P	D�;+���A�*

	conv_loss&=+?,��        )��P	��;+���A�*

	conv_loss�G+?��        )��P	U<+���A�*

	conv_loss�D+?�WY�        )��P	y;<+���A�*

	conv_loss�2+?�#�        )��P	'l<+���A�*

	conv_loss�W+?�:mT        )��P	^�<+���A�*

	conv_lossN4+?�\p=        )��P	��<+���A�*

	conv_loss�/+?P�!�        )��P	Z=+���A�*

	conv_loss�2+?��u]        )��P	^E=+���A�*

	conv_loss"6+?F�*�        )��P	v=+���A�*

	conv_lossc:+?�        )��P	��=+���A�*

	conv_loss�L+?�	ĥ        )��P	��=+���A�*

	conv_loss:=+?���        )��P	->+���A�*

	conv_loss="+?���        )��P	�F>+���A�*

	conv_loss�+?�^�5        )��P	.z>+���A�*

	conv_loss�'+?� !�        )��P	�>+���A�*

	conv_lossf5+?�RE0        )��P	(�>+���A�*

	conv_loss�+?�L�        )��P	 ?+���A�*

	conv_loss/�*?��y�        )��P	�K?+���A�*

	conv_loss@+?�2T        )��P	A?+���A�*

	conv_loss;+?Z�p        )��P	*�?+���A�*

	conv_loss��*?-md�        )��P	,�?+���A�*

	conv_loss/
+?:�Y|        )��P	G@+���A�*

	conv_loss�+?�+�        )��P	wC@+���A�*

	conv_loss�+?��=        )��P	2s@+���A�*

	conv_loss[�*?�}a        )��P	�@+���A�*

	conv_lossP�*?�W��        )��P	��@+���A�*

	conv_lossm+?RM�%        )��P	bA+���A�*

	conv_lossD�*?+Ӿ(        )��P	�;A+���A�*

	conv_loss�*?�:��        )��P	+iA+���A�*

	conv_loss$�*?��6�        )��P	Y�A+���A�*

	conv_loss{�*?�C�I        )��P		�A+���A�*

	conv_loss��*?;��e        )��P	��A+���A�*

	conv_loss�+?s8�        )��P	�)B+���A�*

	conv_loss�+?�TƏ        )��P	\B+���A�*

	conv_loss��*?Mb��        )��P	�B+���A�*

	conv_loss1�*?}]w(        )��P	��B+���A�*

	conv_loss̵*?��]T        )��P	��B+���A�*

	conv_loss��*?��!        )��P	�C+���A�*

	conv_loss��*?)w�F        )��P	�HC+���A�*

	conv_lossA�*?�N        )��P	͋C+���A�*

	conv_lossL�*?e�*        )��P		�C+���A�*

	conv_loss��*?i){�        )��P	��C+���A�*

	conv_lossJ�*?���$        )��P	D+���A�*

	conv_loss��*?g ��        )��P	SMD+���A�*

	conv_loss��*?�e͞        )��P	�|D+���A�*

	conv_loss��*?�XU        )��P	��D+���A�*

	conv_loss�*?�D<�        )��P	��D+���A�*

	conv_loss>�*?%X|        )��P	`E+���A�*

	conv_lossW�*?O Jy        )��P	:DE+���A�*

	conv_lossn�*?�g|        )��P	;tE+���A�*

	conv_loss(�*?͊P        )��P	ݦE+���A�*

	conv_loss`�*?����        )��P	j�E+���A�*

	conv_loss�*?�o?s        )��P	�F+���A�*

	conv_lossڞ*?�fɛ        )��P	�9F+���A�*

	conv_loss�*?�M��        )��P	�G+���A�*

	conv_lossA�*?B��        )��P	� H+���A�*

	conv_loss��*?����        )��P	6>H+���A�*

	conv_loss�*?/ �%        )��P	�vH+���A�*

	conv_loss�n*?��3�        )��P	�H+���A�*

	conv_loss
�*?�$        )��P	��H+���A�*

	conv_lossI�*?p�r        )��P	=I+���A�*

	conv_loss�r*?H���        )��P	RI+���A�*

	conv_lossa�*?;��2        )��P	d�I+���A�*

	conv_loss\w*?��5        )��P	��I+���A�*

	conv_loss�l*?+2g        )��P	�I+���A�*

	conv_loss�\*?�U�Z        )��P	"J+���A�*

	conv_loss��*?]�l�        )��P	�TJ+���A�*

	conv_loss�x*?�F�        )��P	6�J+���A�*

	conv_lossY*?���        )��P	B�J+���A�*

	conv_lossd*?+��        )��P	��J+���A�*

	conv_loss.e*?t*(�        )��P	I!K+���A�*

	conv_loss�*?B��R        )��P	]`K+���A�*

	conv_loss�]*?�h/B        )��P	%�K+���A�*

	conv_loss�o*?bVB        )��P	��K+���A�*

	conv_loss,d*?ZVQ        )��P	��K+���A�*

	conv_loss	d*?�l,        )��P	0*L+���A�*

	conv_loss�h*?�N�        )��P	8iL+���A�*

	conv_lossiU*?��u�        )��P	6�L+���A�*

	conv_losszC*?Ӳ�v        )��P	��L+���A�*

	conv_loss1/*?�C        )��P	J�L+���A�*

	conv_lossy *?J`]�        )��P	F-M+���A�*

	conv_lossRP*?l��        )��P	�iM+���A�*

	conv_loss�*?�Ye�        )��P	m�M+���A�*

	conv_loss�T*?�        )��P	��M+���A�*

	conv_loss�O*?&��        )��P	-�M+���A�*

	conv_loss1@*?�mP�        )��P	�+N+���A�*

	conv_loss�!*?��1        )��P	#iN+���A�*

	conv_losse1*?�1/        )��P	ךN+���A�*

	conv_loss7*?�T�        )��P	w�N+���A�*

	conv_lossz*?��        )��P	X�N+���A�*

	conv_lossL$*?E]�G        )��P	�-O+���A�*

	conv_loss*?Ԕ�        )��P	/lO+���A�*

	conv_loss"*?��p        )��P	l�O+���A�*

	conv_loss2(*?�d;(        )��P	��O+���A�*

	conv_loss"*?����        )��P	�O+���A�*

	conv_loss^*?7��        )��P	�.P+���A�*

	conv_lossw*?�c%        )��P	�_P+���A�*

	conv_loss�)?�ŝ        )��P	��P+���A�*

	conv_loss�)?�T"�        )��P	}�P+���A�*

	conv_loss��)?}�<        )��P	��P+���A�*

	conv_loss��)?���        )��P	�"Q+���A�*

	conv_loss��)?V:�        )��P	-TQ+���A�*

	conv_loss�(*?hh        )��P	O�Q+���A�*

	conv_loss�)?0x�        )��P	��Q+���A�*

	conv_loss�)?L���        )��P	>�Q+���A�*

	conv_loss��)?A���        )��P	�-R+���A�*

	conv_loss� *?:��n        )��P	�_R+���A�*

	conv_loss��)?��Dl        )��P	�R+���A�*

	conv_loss��)?7{(        )��P	��R+���A�*

	conv_lossz�)?`y        )��P	E�R+���A�*

	conv_losss�)?�<�y        )��P	�0S+���A�*

	conv_loss��)?�T�x        )��P	�cS+���A�*

	conv_loss�)?��Cl        )��P	��S+���A�*

	conv_loss��)?����        )��P	��S+���A�*

	conv_lossf�)?���        )��P	T+���A�*

	conv_loss�)?p�#        )��P	z7T+���A�*

	conv_loss�)?��~x        )��P	�hT+���A�*

	conv_loss��)?�h 
        )��P	P�T+���A�*

	conv_loss��)?���        )��P	��T+���A�*

	conv_loss��)?��A�        )��P	�U+���A�*

	conv_lossx�)?����        )��P	89U+���A�*

	conv_loss��)?�M�        )��P	�qU+���A�*

	conv_loss��)?�j}q        )��P	��U+���A�*

	conv_lossU�)?�4=�        )��P	T�U+���A�*

	conv_loss�)?���5        )��P	�V+���A�*

	conv_loss�)?�-Q�        )��P	9BV+���A�*

	conv_lossT�)?�t�        )��P	sV+���A�*

	conv_lossu�)?tTwz        )��P	v�V+���A�*

	conv_loss©)?���!        )��P	��V+���A�*

	conv_losss�)?�c!"        )��P	�W+���A�*

	conv_loss��)?)a$�        )��P	�CW+���A�*

	conv_loss�v)?}q��        )��P	tW+���A�*

	conv_loss|A)?��P        )��P	��W+���A�*

	conv_loss�s)? ��        )��P	w�W+���A�*

	conv_loss�w)?�Ӟ         )��P	�X+���A�*

	conv_loss�v)?���        )��P	>X+���A�*

	conv_lossi{)?c���        )��P	�rX+���A�*

	conv_loss�w)?�9�        )��P	��X+���A�*

	conv_lossH�)?�[o�        )��P	��X+���A�*

	conv_loss��)?i+��        )��P	wY+���A�*

	conv_loss�k)?/x�|        )��P	�=Y+���A�*

	conv_lossUH)?]j�        )��P	�mY+���A�*

	conv_loss�b)?}�n+        )��P	��Y+���A�*

	conv_lossK)?d��        )��P	)�Y+���A�*

	conv_loss;{)?�?S        )��P	  Z+���A�*

	conv_loss�C)?���        )��P	\2Z+���A�*

	conv_losss\)?�hdh        )��P	�eZ+���A�*

	conv_losso)?��        )��P	N�Z+���A�*

	conv_loss�G)?uO        )��P	��Z+���A�*

	conv_loss�1)?r� �        )��P	��Z+���A�*

	conv_loss�C)?l�mS        )��P	�)[+���A�*

	conv_loss�)?�Z��        )��P	�][+���A�*

	conv_lossD)?��*W        )��P	΍[+���A�*

	conv_loss 8)?�{�8        )��P	@�[+���A�*

	conv_loss(G)?-糯        )��P	p\+���A�*

	conv_loss�5)?�2 �        )��P	[3\+���A�*

	conv_loss�0)?��        )��P	te\+���A�*

	conv_loss�9)?�k��        )��P	�\+���A�*

	conv_loss�4)?��U        )��P	��\+���A�*

	conv_loss�E)?�2y�        )��P	)]+���A�*

	conv_loss6)?��@        )��P	�@]+���A�*

	conv_loss�)?#�=|        )��P	�r]+���A�*

	conv_loss�)?XM�]        )��P	�]+���A�*

	conv_loss�)?���        )��P	��]+���A�*

	conv_losss�(?��T        )��P	� ^+���A�*

	conv_lossJ)?	�Hr        )��P	�S^+���A�*

	conv_loss%)?k`s        )��P	�^+���A�*

	conv_loss�#)?���        )��P	&�^+���A�*

	conv_lossp�(?A��        )��P	;�^+���A�*

	conv_lossz�(?�̙W        )��P	V_+���A�*

	conv_loss��(?#�;        )��P	yP_+���A�*

	conv_loss��(?F��        )��P	 �_+���A�*

	conv_loss��(?�p�o        )��P	$�_+���A�*

	conv_loss��(?�w�        )��P	5`+���A�*

	conv_loss��(?��@�        )��P	Y3`+���A�*

	conv_losso�(?�`q        )��P	�c`+���A�*

	conv_loss[�(?���        )��P	E�`+���A�*

	conv_loss��(?���T        )��P	M�`+���A�*

	conv_loss �(?b|aF        )��P	��`+���A�*

	conv_loss�(?"
�?        )��P	3-a+���A�*

	conv_lossy�(?N�;7        )��P	
_a+���A�*

	conv_loss��(?�^p        )��P	R�a+���A�*

	conv_loss��(?4�;D        )��P	W�a+���A�*

	conv_loss�(?We�        )��P	D�a+���A�*

	conv_lossȢ(?v�{s        )��P	\'b+���A�*

	conv_loss��(?u�>�        )��P	�Yb+���A�*

	conv_losst�(?��]        )��P	3�b+���A�*

	conv_loss��(?��        )��P	��b+���A�*

	conv_loss��(?�=L        )��P	�b+���A�*

	conv_loss'�(?�U/�        )��P	z(c+���A�*

	conv_lossx�(?A�y�        )��P	QZc+���A�*

	conv_loss��(?D��V        )��P	-�c+���A�*

	conv_lossF�(?gCՍ        )��P	�c+���A�*

	conv_loss6�(?�p�        )��P	�c+���A�*

	conv_loss�v(?fC��        )��P	�d+���A�*

	conv_loss��(?u�~�        )��P	�Pd+���A�*

	conv_lossnx(?Q�u        )��P	S�d+���A�*

	conv_loss�(?e��        )��P	̷d+���A�*

	conv_loss�(?�)�        )��P	��d+���A�*

	conv_loss~�(?��6;        )��P	�*e+���A�*

	conv_loss5�(?�d�"        )��P	r\e+���A�*

	conv_lossCr(?]=�k        )��P	��e+���A�*

	conv_lossA2(?��"�        )��P	^�e+���A�*

	conv_loss�y(?�J�        )��P	m�e+���A�*

	conv_lossyM(?�Y��        )��P	<4f+���A�*

	conv_loss�s(?���        )��P	ff+���A�*

	conv_loss]o(?��Y�        )��P	�f+���A�*

	conv_loss�[(?}R��        )��P	�f+���A�*

	conv_lossRe(?��pw        )��P	)�f+���A�*

	conv_loss�w(?��Aw        )��P	�;g+���A�*

	conv_loss�@(?í�8        )��P	�lg+���A�*

	conv_loss�#(?��k�        )��P	��g+���A�*

	conv_lossJq(?�b        )��P	��g+���A�*

	conv_loss�P(?��"        )��P	�h+���A�*

	conv_loss�7(?��+^        )��P	sHh+���A�*

	conv_lossX5(?A��        )��P	Rzh+���A�*

	conv_loss�2(?o���        )��P	7�h+���A�*

	conv_loss6(?�-�        )��P	��h+���A�*

	conv_loss�E(?��        )��P	i+���A�*

	conv_loss�"(?��t�        )��P	�Xi+���A�*

	conv_loss5�'?�        )��P	V�i+���A�*

	conv_lossWK(?����        )��P	z�i+���A�*

	conv_loss�(?���        )��P	��i+���A�*

	conv_lossX/(?ަ�d        )��P	�>j+���A�*

	conv_loss*(?ѐ�        )��P	^oj+���A�*

	conv_loss�'(?;��        )��P	�j+���A�*

	conv_loss�((?4��0        )��P	��j+���A�*

	conv_loss�
(?d@B�        )��P	'k+���A�*

	conv_loss�'?�U�        )��P	�6k+���A�*

	conv_loss	(?�Y>        )��P	�pk+���A�*

	conv_lossT(?�'h�        )��P	�k+���A�*

	conv_loss�(?�:�\        )��P	�k+���A�*

	conv_loss �'?���
        )��P	�
l+���A�*

	conv_lossT�'?�>�+        )��P	�<l+���A�*

	conv_loss6�'?�(M�        )��P	�pl+���A�*

	conv_loss��'?4��J        )��P	|�l+���A�*

	conv_loss��'?�r�        )��P	��l+���A�*

	conv_losso�'?�k]        )��P	m+���A�*

	conv_loss�'?Sppq        )��P	�5m+���A�*

	conv_loss'�'?82�        )��P	Kpm+���A�*

	conv_loss��'?�A�        )��P	��m+���A�*

	conv_losso�'?b��"        )��P	k�m+���A�*

	conv_loss$�'?�N�        )��P	�n+���A�*

	conv_loss��'?����        )��P	�:n+���A�*

	conv_loss��'?��>v        )��P	�jn+���A�*

	conv_loss��'?A�N        )��P	��n+���A�*

	conv_loss��'?�å        )��P	��n+���A�*

	conv_loss�'?�F�3        )��P	\ o+���A�*

	conv_loss_�'?L��        )��P	�2o+���A�*

	conv_loss�~'?VV��        )��P	+mo+���A�*

	conv_loss@�'?��        )��P	ۢo+���A�*

	conv_loss�'?!��3        )��P	��o+���A�*

	conv_loss�{'?lcr�        )��P	Sp+���A�*

	conv_lossΡ'?D���        )��P	J9p+���A�*

	conv_loss;�'?��        )��P	��q+���A�*

	conv_loss�v'?��WC        )��P	&r+���A�*

	conv_loss!�'?$�S�        )��P	{>r+���A�*

	conv_lossT�'?D�#�        )��P	Dsr+���A�*

	conv_lossT{'?�Nv        )��P	��r+���A�*

	conv_loss�y'?/3�        )��P	��r+���A�*

	conv_loss�v'?9>ӷ        )��P	(s+���A�*

	conv_lossWj'?r�!        )��P	d?s+���A�*

	conv_loss�n'?����        )��P	w�s+���A�*

	conv_loss_J'?��/        )��P	��s+���A�*

	conv_loss�`'?��w        )��P	;�s+���A�*

	conv_loss�)'?{f�        )��P	Zt+���A�*

	conv_loss5'?N��        )��P	�Pt+���A�*

	conv_loss�('?#�E        )��P	g�t+���A�*

	conv_loss�'?#.̲        )��P	��t+���A�*

	conv_loss/8'?��=�        )��P	��t+���A�*

	conv_loss5@'?�-�/        )��P	�u+���A�*

	conv_lossP'?�P+V        )��P	PPu+���A�*

	conv_loss�'?A�        )��P	l�u+���A�*

	conv_loss�N'?��        )��P	V�u+���A�*

	conv_loss$'?�I{        )��P	N�u+���A�*

	conv_lossN<'?;�        )��P	�v+���A�*

	conv_loss�O'?JU        )��P	�Yv+���A�*

	conv_loss>\'?��~3        )��P	��v+���A�*

	conv_loss�'?����        )��P	�v+���A�*

	conv_loss��&?����        )��P	6�v+���A�*

	conv_loss��&?�Iۑ        )��P	�$w+���A�*

	conv_loss��&?j'7�        )��P	�Tw+���A�*

	conv_loss��&?��        )��P	ڐw+���A�*

	conv_loss��&?�m�c        )��P	��w+���A�*

	conv_loss��&?-
5&        )��P	��w+���A�*

	conv_loss|'?"�F�        )��P	�*x+���A�*

	conv_loss`�&?Vc��        )��P	c]x+���A�*

	conv_loss�&?�V�h        )��P	�x+���A�*

	conv_loss��&?���        )��P	�x+���A�*

	conv_lossh�&?��y,        )��P	��x+���A�*

	conv_loss��&?M�+$        )��P	�)y+���A�*

	conv_loss}'?$J�D        )��P	$Zy+���A�*

	conv_loss(�&?��b        )��P	��y+���A�*

	conv_loss��&?'��]        )��P	t�y+���A�*

	conv_lossq�&?u��        )��P		�y+���A�*

	conv_loss�&?�$R�        )��P	:z+���A�*

	conv_loss(Z&?>�L�        )��P	�Oz+���A�*

	conv_loss �&?�f,        )��P	��z+���A�*

	conv_loss�&?�k��        )��P	�z+���A�*

	conv_loss��&?e�o        )��P	��z+���A�*

	conv_loss�&?;���        )��P	�{+���A�*

	conv_loss�V&?OB �        )��P	�O{+���A�*

	conv_lossf�&?p��_        )��P	�{+���A�*

	conv_lossq�&?J��        )��P	��{+���A�*

	conv_loss�f&?�?��        )��P	��{+���A�*

	conv_loss�t&?J��        )��P	�+|+���A�*

	conv_lossv&?uۨ�        )��P	s`|+���A�*

	conv_lossO&?"��,        )��P	�|+���A�*

	conv_loss�l&?֐#�        )��P	;�|+���A�*

	conv_loss33&?!+�        )��P	 �|+���A�*

	conv_loss�`&?RiO�        )��P	Z)}+���A�*

	conv_loss�&?�z�        )��P	^}+���A�*

	conv_loss&?�17        )��P	��}+���A�*

	conv_lossrQ&?��b        )��P	L�}+���A�*

	conv_loss�f&?Y���        )��P	&~+���A�*

	conv_loss�6&?+��        )��P	�W~+���A�*

	conv_loss{5&?b3�         )��P	Ո~+���A�*

	conv_loss� &?^;�}        )��P	q�~+���A�*

	conv_lossM�%?�	        )��P	�~+���A�*

	conv_lossR4&?�\��        )��P	�,+���A�*

	conv_loss�&??H        )��P	�\+���A�*

	conv_lossc�%?I�        )��P	G�+���A�*

	conv_loss�%&?��җ        )��P	��+���A�*

	conv_loss,
&?��Sl        )��P	��+���A�*

	conv_lossݿ%?��0�        )��P	X,�+���A�*

	conv_loss��%?ڍ�        )��P	o�+���A�*

	conv_loss/&?[�F^        )��P	���+���A�*

	conv_loss:�%?�[Ԡ        )��P	�Հ+���A�*

	conv_loss�&?2���        )��P	d	�+���A�*

	conv_loss=�%?��m        )��P	�;�+���A�*

	conv_loss�&?	��0        )��P	,n�+���A�*

	conv_loss�&?.-yw        )��P	���+���A�*

	conv_loss��%?�a�	        )��P	jс+���A�*

	conv_loss��%?�Y�        )��P	��+���A�*

	conv_loss��%?�"��        )��P	5�+���A�*

	conv_loss��%?��nz        )��P	g�+���A�*

	conv_loss��%?��ɑ        )��P	L��+���A�*

	conv_lossp�%?Y�6        )��P	�˂+���A�*

	conv_loss��%?*Ȥ        )��P	���+���A�*

	conv_loss�%?��q        )��P	�/�+���A�*

	conv_loss�%?�!�        )��P	�a�+���A�*

	conv_loss��%?о�f        )��P	�+���A�*

	conv_lossa�%?_��d        )��P	�ȃ+���A�*

	conv_loss7-%?���~        )��P	���+���A�*

	conv_lossU%?���        )��P	0�+���A�*

	conv_loss�%?ȁJ_        )��P	~c�+���A�*

	conv_losst�%?(�݅        )��P	���+���A�*

	conv_loss�Z%?��C        )��P	�Ȅ+���A�*

	conv_loss5{%?oAht        )��P	���+���A�*

	conv_loss�?%?Z���        )��P	�-�+���A�*

	conv_loss8E%?\��        )��P	Z^�+���A�*

	conv_lossa�%?o�f�        )��P	���+���A�*

	conv_loss>%?�["&        )��P	�ą+���A�*

	conv_losscI%?d���        )��P	���+���A�*

	conv_losse%?��P        )��P	�;�+���A�*

	conv_loss��$?KL[�        )��P	�n�+���A�*

	conv_loss[%?�Z        )��P	Q��+���A�*

	conv_loss�o%?��\        )��P	��+���A�*

	conv_loss�N%?g��        )��P	��+���A�*

	conv_loss��$?��        )��P	�P�+���A�*

	conv_lossX'%?"��(        )��P	��+���A�*

	conv_loss++%?�9g        )��P	��+���A�*

	conv_loss�F%?ě�        )��P	b�+���A�*

	conv_loss��$?-N�a        )��P	�=�+���A�*

	conv_lossd�$?ݠY        )��P	q�+���A�*

	conv_loss��$?�5��        )��P	䣈+���A�*

	conv_loss��$?����        )��P	HՈ+���A�*

	conv_loss/A%?�dR�        )��P	��+���A�*

	conv_loss��$?ϣ`*        )��P	�:�+���A�*

	conv_loss��$?$�	        )��P	m�+���A�*

	conv_loss}$?�@~c        )��P	d��+���A�*

	conv_loss�Z$?ѫs        )��P	�׉+���A�*

	conv_loss�$?J��        )��P	L�+���A�*

	conv_loss��$?����        )��P	YM�+���A�*

	conv_lossZ�$?b���        )��P	��+���A�*

	conv_loss��$?���        )��P	���+���A�*

	conv_loss��$?�o�9        )��P	%��+���A�*

	conv_lossL$?`��        )��P	&*�+���A�*

	conv_loss�$?3<8�        )��P	G\�+���A�*

	conv_loss#q$?��L$        )��P	K��+���A�*

	conv_loss)$?���G        )��P	�+���A�*

	conv_loss�U$?���        )��P	N��+���A�*

	conv_loss�h$?���        )��P	�3�+���A�*

	conv_loss�"$?�K        )��P	Qe�+���A�*

	conv_loss�r$?�g�        )��P	
��+���A�*

	conv_loss�=$?�C�        )��P	�ь+���A�*

	conv_loss�9$?��{Q        )��P	q�+���A�*

	conv_loss�O$?P�U�        )��P	�B�+���A�*

	conv_loss:�#?��#Q        )��P	�u�+���A�*

	conv_lossU4$?!^ �        )��P	+���A�*

	conv_loss�7$?JU�        )��P	�ٍ+���A�*

	conv_loss�$?��˯        )��P	�+���A�*

	conv_lossB,$?����        )��P	C>�+���A�*

	conv_loss�$?M�*>        )��P	�o�+���A�*

	conv_loss�;$?3��        )��P	ɡ�+���A�*

	conv_loss��#?W$̋        )��P	�Ҏ+���A�*

	conv_loss�$?�'        )��P	��+���A�*

	conv_lossy�#?        )��P	g5�+���A�*

	conv_lossA�#?
���        )��P	<f�+���A�*

	conv_loss7�#?��3�        )��P	!��+���A�*

	conv_lossa�#?���O        )��P	�ˏ+���A�*

	conv_loss2k#?�?,        )��P	[��+���A�*

	conv_loss��#? �'�        )��P	�-�+���A�*

	conv_lossJ�#?z���        )��P	E_�+���A�*

	conv_loss=�#?�ɨ�        )��P	���+���A�*

	conv_loss�#?WxU        )��P	_Ԑ+���A�*

	conv_loss>)#?5ɶ�        )��P	
�+���A�*

	conv_loss|#?cc��        )��P	�=�+���A�*

	conv_loss-�#?���z        )��P	Io�+���A�*

	conv_loss��#?w�`&        )��P	 ��+���A�*

	conv_losswY#?��        )��P	�֑+���A�*

	conv_loss=�"?3]�O        )��P	*�+���A�*

	conv_lossi#?����        )��P	P�+���A�*

	conv_loss�~#?ɮ|        )��P	剒+���A�*

	conv_loss�e#?���        )��P	���+���A�*

	conv_loss�A#?1��/        )��P	��+���A�*

	conv_loss�T#?Pk4�        )��P	��+���A�*

	conv_loss�z#?�e�H        )��P	�[�+���A�*

	conv_loss��"?\��        )��P	Ԓ�+���A�*

	conv_loss�N#?m�e�        )��P	GǓ+���A�*

	conv_loss#?U/Su        )��P	���+���A�*

	conv_loss�"?�g6        )��P	n)�+���A�*

	conv_loss:#?cr�}        )��P	Zm�+���A�*

	conv_loss"	#?<���        )��P	m��+���A�*

	conv_loss6"#?-�ý        )��P	�۔+���A�*

	conv_lossh�"?{�H?        )��P	S�+���A�*

	conv_loss)�"?�;�i        )��P	C@�+���A�*

	conv_loss�#?����        )��P	E{�+���A�*

	conv_loss��"?��        )��P	��+���A�*

	conv_loss��"?2�        )��P	W�+���A�*

	conv_loss��"?e���        )��P	|�+���A�*

	conv_loss��"?�D�        )��P	qF�+���A�*

	conv_loss.�"?���<        )��P	慖+���A�*

	conv_loss/�"?����        )��P	���+���A�*

	conv_loss{"?�V��        )��P	��+���A�*

	conv_lossӇ"?����        )��P	QC�+���A�*

	conv_lossJ_"?g        )��P	�v�+���A�*

	conv_loss"?Z2�6        )��P	z��+���A�*

	conv_loss b"?|��        )��P	��+���A�*

	conv_loss�!?��^W        )��P	��+���A�*

	conv_loss|="?d���        )��P	&F�+���A�*

	conv_lossH"?�,�9        )��P	zw�+���A�*

	conv_loss�5"?���        )��P	߱�+���A�*

	conv_lossU"?R�G        )��P	�+���A�*

	conv_loss."?���        )��P	��+���A�*

	conv_lossL"?TХ        )��P	�N�+���A�*

	conv_loss��!?��t�        )��P	(��+���A�*

	conv_lossVR"?\���        )��P	�+���A�*

	conv_lossJ"?��Q[        )��P	���+���A�*

	conv_lossR�!?C���        )��P	�'�+���A�*

	conv_lossQ�!?�|�l        )��P	`Y�+���A�*

	conv_loss�!?�P�        )��P	y��+���A�*

	conv_lossB=!?{�<        )��P	1ɚ+���A�*

	conv_loss��!??�Jz        )��P	���+���A�*

	conv_loss�!?	\�        )��P	n6�+���A�*

	conv_loss�!?#R��        )��P	�k�+���A�*

	conv_lossBW!?���~        )��P	���+���A�*

	conv_loss��!?���        )��P	Dۡ+���A�*

	conv_lossb�!?1��        )��P	��+���A�*

	conv_loss�N!?�C3        )��P	nM�+���A�*

	conv_loss� !?IFb�        )��P	�}�+���A�*

	conv_loss�z!?[s�\        )��P	J��+���A�*

	conv_loss?g!??��        )��P	�+���A�*

	conv_loss)|!?3���        )��P	D�+���A�*

	conv_loss�� ?GJ�        )��P	�H�+���A�*

	conv_loss�3!?v(��        )��P	z�+���A�*

	conv_loss�� ?o�>~        )��P	ԯ�+���A�*

	conv_loss�D!?-Kt        )��P	(�+���A�*

	conv_loss � ?��v`        )��P	l�+���A�*

	conv_loss�� ?*�U�        )��P	�N�+���A�*

	conv_loss�u ?S���        )��P	�~�+���A�*

	conv_lossr* ?��&p        )��P	Q��+���A�*

	conv_loss�� ?JG�Z        )��P	v�+���A�*

	conv_loss�P ?��V2        )��P	L�+���A�*

	conv_loss�!?���        )��P	mH�+���A�*

	conv_loss#� ?䖒�        )��P	�x�+���A�*

	conv_loss�. ?UF�        )��P	X��+���A�*

	conv_loss�n ?���        )��P	�٥+���A�*

	conv_loss` ?�;�        )��P	�"�+���A�*

	conv_loss�d ?���        )��P	�]�+���A�*

	conv_loss/� ?��]        )��P	C��+���A�*

	conv_loss�  ?��/        )��P	`��+���A�*

	conv_loss�a ?�!&�        )��P	��+���A�*

	conv_loss� ?�g��        )��P	�'�+���A�*

	conv_loss$�?��        )��P	3Y�+���A�*

	conv_loss4�?�&h        )��P	ņ�+���A�*

	conv_lossgV?��        )��P	3��+���A�*

	conv_loss.�?0C+2        )��P	��+���A�*

	conv_lossm�?Yu.        )��P	��+���A�*

	conv_loss�u?V�g        )��P	a?�+���A�*

	conv_lossG ?ޘ�;        )��P	�m�+���A�*

	conv_loss6�?u��        )��P	���+���A�*

	conv_loss�H?�A�        )��P	�٨+���A�*

	conv_loss�?�V��        )��P	��+���A�*

	conv_lossԄ?�+        )��P	�7�+���A�*

	conv_loss�1?�V         )��P	�g�+���A�*

	conv_loss�?�Ǉ�        )��P	ҙ�+���A�*

	conv_loss� ?��        )��P	�̩+���A�*

	conv_loss�?�AT!        )��P	���+���A�*

	conv_loss�?�M'�        )��P	�)�+���A�*

	conv_loss�?�7�=        )��P	�^�+���A�*

	conv_loss�	?j�C�        )��P	ꏪ+���A�*

	conv_loss��?�x        )��P	{��+���A�*

	conv_lossi�?��P�        )��P	@�+���A�*

	conv_loss$0?�A_�        )��P	�/�+���A�*

	conv_loss��?�Ќ        )��P	�^�+���A�*

	conv_loss��?ԡx        )��P	���+���A�*

	conv_lossp?���.        )��P	��+���A�*

	conv_loss�2?�ʺ        )��P	��+���A�*

	conv_loss��?)#�        )��P	%�+���A�*

	conv_lossrw?-�:         )��P	�X�+���A�*

	conv_loss�9?A'�        )��P	���+���A�*

	conv_loss�.?b^w        )��P	<��+���A�*

	conv_loss�B?`�j        )��P	%�+���A�*

	conv_lossΖ?�2��        )��P	��+���A�*

	conv_lossS�?<�\�        )��P	Y�+���A�*

	conv_loss�?ϥ�}        )��P	��+���A�*

	conv_loss3�?�J|O        )��P	�+���A�*

	conv_loss��?�j.        )��P	b��+���A�*

	conv_loss�?c+�D        )��P	\-�+���A�*

	conv_lossll?����        )��P	�[�+���A�*

	conv_loss^U?qDL        )��P	���+���A�*

	conv_loss�o?�>
        )��P	g��+���A�*

	conv_loss�r?�5�        )��P	��+���A�*

	conv_loss�?u�u�        )��P	��+���A�*

	conv_losss�?�4r        )��P	�G�+���A�*

	conv_loss�m?���        )��P	�x�+���A�*

	conv_loss}"?��w}        )��P	���+���A�*

	conv_loss?�:        )��P	6կ+���A�*

	conv_losseg?���        )��P	��+���A�*

	conv_loss��?��J        )��P	|1�+���A�*

	conv_loss�?��,i        )��P	�_�+���A�*

	conv_loss[s?�1�%        )��P	{��+���A�*

	conv_lossEY?O��$        )��P	���+���A�*

	conv_loss�?_��c        )��P	9�+���A�*

	conv_loss5D?(�ޢ        )��P	-�+���A�*

	conv_loss�g?�0�        )��P	�N�+���A�*

	conv_loss31?龄e        )��P	��+���A�*

	conv_loss��?�C��        )��P	���+���A�*

	conv_loss�.?�=t�        )��P	�ܱ+���A�*

	conv_loss�?@A        )��P	��+���A�*

	conv_loss`�?K��b        )��P	:�+���A�*

	conv_loss��?0�7�        )��P	�h�+���A�*

	conv_loss��?j��        )��P	|��+���A�*

	conv_lossi?5G��        )��P	�ǲ+���A�*

	conv_loss��?jCM�        )��P	���+���A�*

	conv_losse�?� �         )��P	�&�+���A�*

	conv_loss�-?X�        )��P	'W�+���A�*

	conv_lossR�?V�$I        )��P	���+���A�*

	conv_loss�{?D��{        )��P	���+���A�*

	conv_loss��?v��        )��P	�+���A�*

	conv_loss�w?;�3        )��P	��+���A�*

	conv_loss/m?��*T        )��P	�H�+���A�*

	conv_loss�Z?���        )��P	�z�+���A�*

	conv_loss��?c�o        )��P	���+���A�*

	conv_loss?�?2��        )��P	6�+���A�*

	conv_loss�t?�-�t        )��P	�&�+���A�*

	conv_loss[�?� �        )��P	o_�+���A�*

	conv_loss?��/        )��P	ٛ�+���A�*

	conv_loss�b?���        )��P	pе+���A�*

	conv_loss��?����        )��P	��+���A�*

	conv_lossB?���3        )��P	d:�+���A�*

	conv_loss'8?��Q        )��P	p�+���A�*

	conv_loss��?*f�        )��P	���+���A�*

	conv_lossWg?砊�        )��P	)�+���A�*

	conv_lossr�?q�R�        )��P	��+���A�*

	conv_loss��?����        )��P	�L�+���A�*

	conv_loss�?�ė        )��P	��+���A�*

	conv_loss�?[
��        )��P	^��+���A�*

	conv_loss;v?O��        )��P	���+���A�*

	conv_lossl�?a�I
        )��P	>*�+���A�*

	conv_loss�?Ӆ<�        )��P	a�+���A�*

	conv_loss�x?A&�        )��P	I��+���A�*

	conv_loss�=?k�D�        )��P	4ȸ+���A�*

	conv_loss>k?8���        )��P	���+���A�*

	conv_loss�?�>�/        )��P	 3�+���A�*

	conv_loss9\?S�.�        )��P	+f�+���A�*

	conv_loss�>?���        )��P	^��+���A�*

	conv_loss��?JI�        )��P	Sй+���A�*

	conv_losss<?���m        )��P	��+���A�*

	conv_lossչ?���        )��P	9�+���A�*

	conv_loss�?Nk��        )��P	�m�+���A�*

	conv_loss5�?��1        )��P	顺+���A�*

	conv_loss8�?�&6�        )��P	uֺ+���A�*

	conv_loss�/?�v�        )��P	�	�+���A�*

	conv_loss��?���        )��P	�>�+���A�*

	conv_loss�d?���        )��P	�t�+���A�*

	conv_loss=?B��=        )��P	8��+���A�*

	conv_loss4�?w��]        )��P	�+���A�*

	conv_loss�v?�)%        )��P	��+���A�*

	conv_loss|�?���        )��P	~H�+���A�*

	conv_loss0m?g�01        )��P	�{�+���A�*

	conv_loss��?��X        )��P	f��+���A�*

	conv_lossj�?W�!        )��P	��+���A�*

	conv_loss��?�P��        )��P	��+���A�*

	conv_loss��?�m��        )��P	 P�+���A�*

	conv_loss��?�u�        )��P	��+���A�*

	conv_loss��?&a�        )��P	;��+���A�*

	conv_loss7T?Z�9        )��P	��+���A�*

	conv_loss�A?]�~;        )��P	�#�+���A�*

	conv_loss;�?�I�        )��P	IV�+���A�*

	conv_loss�J?ej��        )��P	��+���A�*

	conv_lossG?�g>P        )��P	�+���A�*

	conv_loss=�?���@        )��P	;��+���A�*

	conv_loss�M?`=�        )��P	�;�+���A�*

	conv_loss��?���        )��P	�p�+���A�	*

	conv_loss;?\�+        )��P	���+���A�	*

	conv_loss��?0V�        )��P	�ݿ+���A�	*

	conv_loss��?÷�I        )��P	��+���A�	*

	conv_loss��?z���        )��P	IE�+���A�	*

	conv_loss��?�:`        )��P	;y�+���A�	*

	conv_loss�#?���J        )��P	��+���A�	*

	conv_lossa?���9        )��P	���+���A�	*

	conv_lossæ?��*        )��P	0�+���A�	*

	conv_loss_ ?��        )��P	�f�+���A�	*

	conv_loss4�?�Vl        )��P	��+���A�	*

	conv_lossT!?#{݌        )��P	���+���A�	*

	conv_loss�,?	K��        )��P	-�+���A�	*

	conv_loss��?6�        )��P	'K�+���A�	*

	conv_lossd�?G-��        )��P	�~�+���A�	*

	conv_loss"�?nm        )��P	��+���A�	*

	conv_loss�?.��        )��P	���+���A�	*

	conv_loss�]?�H�        )��P	�+���A�	*

	conv_loss]�?��"y        )��P	�P�+���A�	*

	conv_loss*?T�=        )��P	���+���A�	*

	conv_loss�?�s��        )��P	ϼ�+���A�	*

	conv_loss�]?��&�        )��P	]��+���A�	*

	conv_loss_?�o�        )��P	$�+���A�	*

	conv_loss;q?T��d        )��P	�Y�+���A�	*

	conv_loss��?{���        )��P	]��+���A�	*

	conv_loss�;?�'��        )��P	��+���A�	*

	conv_lossO�?߇Ga        )��P	]��+���A�	*

	conv_lossm?�2�        )��P	-�+���A�	*

	conv_loss�[
?��        )��P	Cb�+���A�	*

	conv_loss�-?s��        )��P	_��+���A�	*

	conv_lossR??j��        )��P	���+���A�	*

	conv_loss�	?��]H        )��P	���+���A�	*

	conv_loss�Q	?� ��        )��P	�3�+���A�	*

	conv_loss[�	?K<�        )��P	�i�+���A�	*

	conv_lossQ�	?�        )��P	���+���A�	*

	conv_loss�?�d�        )��P	���+���A�	*

	conv_loss��?��        )��P	f�+���A�	*

	conv_loss��	?S�x        )��P	8�+���A�	*

	conv_lossx?u��L        )��P	�m�+���A�	*

	conv_lossf�?�)�d        )��P	K��+���A�	*

	conv_loss�O?"�        )��P	���+���A�	*

	conv_loss(�?a��	        )��P	�
�+���A�	*

	conv_loss�?I���        )��P	>?�+���A�	*

	conv_loss*?GA�
        )��P	Br�+���A�	*

	conv_loss,3?I�bj        )��P	��+���A�	*

	conv_lossC}??��        )��P	U��+���A�	*

	conv_lossWG?u!��        )��P	a�+���A�	*

	conv_loss��? AjE        )��P	?F�+���A�	*

	conv_loss.a?'P��        )��P	�{�+���A�	*

	conv_lossg�?�z�        )��P	��+���A�	*

	conv_loss��?�]�        )��P	�L�+���A�	*

	conv_loss��?���        )��P	f��+���A�	*

	conv_loss�?w�%�        )��P	j��+���A�	*

	conv_lossk?Ҷ�@        )��P	H�+���A�	*

	conv_loss��?O=��        )��P	�B�+���A�	*

	conv_loss�� ?���?        )��P	[v�+���A�	*

	conv_lossH�?r���        )��P	)��+���A�	*

	conv_loss?RR1e        )��P	i��+���A�	*

	conv_loss)�?��        )��P	l<�+���A�	*

	conv_lossN�>V�a        )��P	Lo�+���A�	*

	conv_loss���>����        )��P	_��+���A�	*

	conv_loss�� ?�.�        )��P	��+���A�	*

	conv_loss�8�>�8�        )��P	�
�+���A�	*

	conv_loss�~�>L\�&        )��P	.@�+���A�	*

	conv_loss���>�*D�        )��P	x�+���A�	*

	conv_lossJ�>'sp         )��P	��+���A�	*

	conv_loss�E�>�v5�        )��P	���+���A�	*

	conv_loss��>���        )��P	o�+���A�	*

	conv_loss��>6�m(        )��P	H�+���A�	*

	conv_loss�,�>����        )��P	�|�+���A�	*

	conv_lossqW�>��        )��P	b��+���A�	*

	conv_lossD�>���        )��P	���+���A�	*

	conv_loss�1�>���        )��P	e�+���A�	*

	conv_loss�v�>쏷R        )��P	uO�+���A�	*

	conv_loss��>�        )��P	u��+���A�	*

	conv_loss�)�>����        )��P	4��+���A�	*

	conv_loss���>��XN        )��P	���+���A�	*

	conv_lossj
�>uv        )��P	!�+���A�	*

	conv_loss/�>�V�s        )��P	�V�+���A�	*

	conv_loss��>1,��        )��P	H��+���A�	*

	conv_loss�c�>���1        )��P	q��+���A�	*

	conv_lossΊ�>���        )��P	���+���A�	*

	conv_loss\��>��r        )��P	�*�+���A�	*

	conv_loss)��>³I�        )��P	�^�+���A�	*

	conv_loss���>���~        )��P	��+���A�	*

	conv_loss��>��        )��P	���+���A�	*

	conv_loss���>s�@        )��P	N��+���A�	*

	conv_loss3n�>qޚ�        )��P	�0�+���A�	*

	conv_loss�7�>�7�        )��P	)d�+���A�	*

	conv_loss�:�>���b        )��P	���+���A�	*

	conv_loss[��>���>        )��P	Q��+���A�	*

	conv_loss`�>׍�         )��P	=�+���A�	*

	conv_loss���>�.�/        )��P	7�+���A�	*

	conv_lossBC�>	��o        )��P	�l�+���A�	*

	conv_loss!=�>�"\        )��P	C��+���A�	*

	conv_loss�g�>U�        )��P	���+���A�	*

	conv_loss���>�&|2        )��P	;�+���A�	*

	conv_loss��>�޸        )��P	�E�+���A�	*

	conv_lossz�>����        )��P	8z�+���A�	*

	conv_loss���>F�KW        )��P	0��+���A�	*

	conv_loss�A�>��K        )��P	<��+���A�	*

	conv_loss*��>NN        )��P	�/�+���A�	*

	conv_loss�|�>{�NT        )��P	�h�+���A�	*

	conv_loss/�>�R~        )��P	���+���A�	*

	conv_lossWW�>�c�        )��P	���+���A�	*

	conv_loss=0�>�g^�        )��P	)�+���A�	*

	conv_loss���>�&��        )��P	L<�+���A�	*

	conv_loss��>��y        )��P	�o�+���A�	*

	conv_loss�	�>1�2        )��P	���+���A�	*

	conv_loss�]�>��A*        )��P	���+���A�	*

	conv_loss���>}�ԛ        )��P	P �+���A�	*

	conv_loss	�>��        )��P	SU�+���A�	*

	conv_lossId�>��؞        )��P	���+���A�	*

	conv_loss��>����        )��P	��+���A�	*

	conv_lossq��>^��}        )��P	��+���A�	*

	conv_loss��>��q8        )��P	�;�+���A�	*

	conv_loss���>��K�        )��P	np�+���A�	*

	conv_loss��>{�ܣ        )��P	��+���A�	*

	conv_loss?�>�Q�        )��P	���+���A�	*

	conv_loss���>n7��        )��P	D�+���A�	*

	conv_loss8F�>⪓�        )��P	dI�+���A�	*

	conv_loss�0�>���        )��P	�}�+���A�	*

	conv_lossV�>����        )��P	���+���A�	*

	conv_loss�(�>���        )��P	���+���A�	*

	conv_lossg�>P�x        )��P	�$�+���A�	*

	conv_loss.�>��T        )��P	�X�+���A�	*

	conv_loss��>*n�        )��P	ދ�+���A�	*

	conv_lossJi�>�k0        )��P	R��+���A�	*

	conv_lossG��>���        )��P	#��+���A�
*

	conv_loss�>
�-        )��P	�/�+���A�
*

	conv_loss���>�h��        )��P	Kd�+���A�
*

	conv_loss���>��TS        )��P	��+���A�
*

	conv_loss,�>�a�        )��P	9��+���A�
*

	conv_loss�F�>u�        )��P	A�+���A�
*

	conv_loss���>�C�        )��P	�;�+���A�
*

	conv_loss%��>U �_        )��P	Zo�+���A�
*

	conv_lossѸ�>Q��k        )��P	9��+���A�
*

	conv_loss�׻>F���        )��P	y��+���A�
*

	conv_loss���>~D�        )��P	��+���A�
*

	conv_loss(u�>��        )��P	�C�+���A�
*

	conv_loss�N�>{�d�        )��P	<{�+���A�
*

	conv_loss�Կ>+���        )��P	p��+���A�
*

	conv_loss1�>�1�        )��P	��+���A�
*

	conv_loss�A�>6��F        )��P	N�+���A�
*

	conv_loss\��>����        )��P	�L�+���A�
*

	conv_loss�ȹ>}�        )��P	��+���A�
*

	conv_lossK��>���X        )��P	X��+���A�
*

	conv_lossSU�>1
��        )��P	���+���A�
*

	conv_loss�ͽ>�Gv�        )��P	|(�+���A�
*

	conv_loss�n�>(��D        )��P	h{�+���A�
*

	conv_losss�>�e�z        )��P	���+���A�
*

	conv_lossU��>J�b        )��P	,��+���A�
*

	conv_lossǷ>[U        )��P	:�+���A�
*

	conv_lossS�>��(         )��P	z�+���A�
*

	conv_loss͇�>Ƚ�        )��P	���+���A�
*

	conv_loss�ݶ>���G        )��P	�+���A�
*

	conv_lossm�>�~�a        )��P	�;�+���A�
*

	conv_loss��>}XS        )��P	�q�+���A�
*

	conv_lossZ�>�4        )��P	���+���A�
*

	conv_loss�r�>��        )��P	>��+���A�
*

	conv_loss>H�>��Dy        )��P	�+���A�
*

	conv_loss��>��#�        )��P	�M�+���A�
*

	conv_loss�b�>����        )��P	|�+���A�
*

	conv_losst��>���        )��P	U��+���A�
*

	conv_loss��>��Q        )��P	<��+���A�
*

	conv_loss�L�>��        )��P	��+���A�
*

	conv_loss<��>QB�,        )��P	rG�+���A�
*

	conv_loss�v�>�O��        )��P	�u�+���A�
*

	conv_loss$�>ٟ��        )��P	5��+���A�
*

	conv_loss�$�>A�ș        )��P	���+���A�
*

	conv_lossJ��>�
%�        )��P	��+���A�
*

	conv_loss{�>I+��        )��P	K�+���A�
*

	conv_loss=��>���p        )��P	r��+���A�
*

	conv_loss ��>[Y�I        )��P	G��+���A�
*

	conv_loss���>)�8�        )��P	���+���A�
*

	conv_lossI��>1�w        )��P	��+���A�
*

	conv_loss�ڲ>��e�        )��P	�X�+���A�
*

	conv_losseĵ>��)�        )��P	É�+���A�
*

	conv_loss��>�3*        )��P	и�+���A�
*

	conv_loss��>04O        )��P	=��+���A�
*

	conv_loss���>�C��        )��P	!�+���A�
*

	conv_lossぱ>zA�        )��P	~T�+���A�
*

	conv_lossF�>���l        )��P	��+���A�
*

	conv_losse��>N�tm        )��P	ӽ�+���A�
*

	conv_loss�5�>Դ�        )��P	���+���A�
*

	conv_loss�n�>�f�        )��P	*�+���A�
*

	conv_loss��>Ǜ��        )��P	�^�+���A�
*

	conv_loss-�>��~        )��P	��+���A�
*

	conv_loss��>&J�j        )��P	��+���A�
*

	conv_loss�x�>���        )��P	J��+���A�
*

	conv_loss]�>ɬ��        )��P	c*�+���A�
*

	conv_loss`��>�B�        )��P	�Z�+���A�
*

	conv_loss���>�Q)B        )��P	���+���A�
*

	conv_loss檳>�x�        )��P	*��+���A�
*

	conv_loss"��>��t�        )��P	���+���A�
*

	conv_loss�>Z�H        )��P	=5�+���A�
*

	conv_loss5e�>��̽        )��P	�f�+���A�
*

	conv_lossL��> s��        )��P	E��+���A�
*

	conv_loss���>bG��        )��P	���+���A�
*

	conv_loss�7�>�WWZ        )��P	w
�+���A�
*

	conv_loss�>���g        )��P	�E�+���A�
*

	conv_loss��>+        )��P	;u�+���A�
*

	conv_lossT��>')s�        )��P	k��+���A�
*

	conv_loss0Ů>���>        )��P	8��+���A�
*

	conv_loss�,�>C�zs        )��P	x�+���A�
*

	conv_loss e�>��        )��P	b5�+���A�
*

	conv_loss@F�>%ܮ!        )��P	�j�+���A�
*

	conv_loss��>0v��        )��P	F��+���A�
*

	conv_loss2�>L v�        )��P	��+���A�
*

	conv_loss�B�>��        )��P	��+���A�
*

	conv_loss��>���        )��P	�4�+���A�
*

	conv_loss;c�>�P��        )��P	qn�+���A�
*

	conv_loss/U�>)�1�        )��P	ޝ�+���A�
*

	conv_loss��>@�3w        )��P	V��+���A�
*

	conv_lossY��>H΁        )��P	���+���A�
*

	conv_loss�e�>��?�        )��P	�0�+���A�
*

	conv_loss��>�߯o        )��P	f�+���A�
*

	conv_lossMW�>���         )��P	3��+���A�
*

	conv_loss�`�>�h:�        )��P	 ��+���A�
*

	conv_loss��>�        )��P	���+���A�
*

	conv_loss� �>k<�        )��P	4*�+���A�
*

	conv_loss�]�>�X�%        )��P	s^�+���A�
*

	conv_loss>�GY         )��P	m��+���A�
*

	conv_loss���>�;�        )��P	C��+���A�
*

	conv_loss��>��Ņ        )��P	���+���A�
*

	conv_lossv�>�O        )��P	��+���A�
*

	conv_loss��>���        )��P	N�+���A�
*

	conv_loss�F�>MC(        )��P	J��+���A�
*

	conv_loss�i�>ٱ>        )��P	���+���A�
*

	conv_loss�I�>��        )��P	���+���A�
*

	conv_loss��>�}        )��P	��+���A�
*

	conv_loss���>;K3        )��P	4D�+���A�
*

	conv_losskʭ>Im�        )��P	�r�+���A�
*

	conv_lossЍ�>����        )��P	Q��+���A�
*

	conv_loss��>�]&�        )��P	O��+���A�
*

	conv_loss➫>*�g        )��P	
�+���A�
*

	conv_loss�D�>�e7        )��P	�7�+���A�
*

	conv_loss�ԭ>#���        )��P	�g�+���A�
*

	conv_loss	\�>8��        )��P	&��+���A�
*

	conv_loss���>��t        )��P	���+���A�
*

	conv_loss���>�$4�        )��P	���+���A�
*

	conv_loss�Q�>�ω]        )��P	$�+���A�
*

	conv_loss_$�>���        )��P	�R�+���A�
*

	conv_loss�>� �        )��P	A��+���A�
*

	conv_loss�>�6�#        )��P	ʱ�+���A�
*

	conv_loss�b�>�)�        )��P	���+���A�
*

	conv_lossv�>�/        )��P	�+���A�
*

	conv_loss�h�>���q        )��P	�<�+���A�
*

	conv_loss ?�>A��        )��P	�j�+���A�
*

	conv_lossL�>Q��        )��P	;��+���A�
*

	conv_loss���>J͊        )��P	p)�+���A�
*

	conv_loss���>�y�/        )��P	�W�+���A�
*

	conv_loss��>��)        )��P	��+���A�
*

	conv_loss�
�>��R        )��P	B��+���A�
*

	conv_loss�9�>iߪ�        )��P	���+���A�
*

	conv_loss��>͍�        )��P	��+���A�
*

	conv_lossع�>��        )��P	BJ�+���A�*

	conv_lossq��>R�7        )��P	8x�+���A�*

	conv_loss���> s        )��P	(��+���A�*

	conv_loss/s�>v��w        )��P	���+���A�*

	conv_losss�>.*��        )��P	��+���A�*

	conv_loss� �>!)*        )��P	�G�+���A�*

	conv_loss��>�        )��P	qz�+���A�*

	conv_loss���>i��        )��P	S��+���A�*

	conv_lossF@�>��        )��P	~��+���A�*

	conv_loss�>�6        )��P	��+���A�*

	conv_lossl��>�{�j        )��P	AO�+���A�*

	conv_loss�_�>|��        )��P	'��+���A�*

	conv_loss��>�x��        )��P	s��+���A�*

	conv_loss
ɪ>�I�        )��P	���+���A�*

	conv_loss�U�>����        )��P	�-�+���A�*

	conv_lossbȬ>�<�N        )��P	n^�+���A�*

	conv_loss"̬>-Q        )��P	ȏ�+���A�*

	conv_loss�.�>�m�P        )��P	���+���A�*

	conv_loss�Y�>���        )��P	���+���A�*

	conv_loss�g�>t��        )��P	��+���A�*

	conv_loss4�>���        )��P	qK�+���A�*

	conv_losse5�>����        )��P	U|�+���A�*

	conv_lossn�>����        )��P	���+���A�*

	conv_loss��>�A?        )��P	~��+���A�*

	conv_loss�v�>['}        )��P	�
�+���A�*

	conv_loss"[�>2Z�s        )��P	�9�+���A�*

	conv_loss�Ϭ>�$�!        )��P	�j�+���A�*

	conv_loss�K�>͌�        )��P	���+���A�*

	conv_lossR��>��        )��P	%��+���A�*

	conv_lossF�>��Ur        )��P	���+���A�*

	conv_loss�Ы>���$        )��P	�%�+���A�*

	conv_loss"#�>)��        )��P	U�+���A�*

	conv_loss��>��        )��P	���+���A�*

	conv_loss�H�>E�        )��P	Ѵ�+���A�*

	conv_loss��>/��        )��P	���+���A�*

	conv_loss�_�>q⌈        )��P	��+���A�*

	conv_loss/>�>5���        )��P	�A�+���A�*

	conv_loss��>s��L        )��P	s�+���A�*

	conv_losss�>�        )��P	$��+���A�*

	conv_loss��>���        )��P	��+���A�*

	conv_lossb�>��Xf        )��P	� �+���A�*

	conv_loss��>� �        )��P	r2�+���A�*

	conv_loss��>�Y��        )��P	Ta�+���A�*

	conv_loss�:�>�0M�        )��P	3��+���A�*

	conv_loss�C�>ľ��        )��P	���+���A�*

	conv_loss�]�>�H�        )��P	 ,���A�*

	conv_loss��>$�d�        )��P	�7 ,���A�*

	conv_loss鉫>�]܄        )��P	l ,���A�*

	conv_loss�W�>|0�        )��P	�� ,���A�*

	conv_loss��>���t        )��P	�� ,���A�*

	conv_lossdi�>Q�T        )��P	W,���A�*

	conv_loss\�>o˄p        )��P	�B,���A�*

	conv_losso2�>�5�        )��P	fw,���A�*

	conv_loss���>�"�        )��P	z�,���A�*

	conv_loss�3�>�`E        )��P	��,���A�*

	conv_loss�ʪ>C��v        )��P	�,���A�*

	conv_lossƠ�>''��        )��P	�B,���A�*

	conv_loss��>�yG�        )��P	�q,���A�*

	conv_loss,!�>I-S�        )��P	�,���A�*

	conv_loss"ݫ>�\�        )��P	��,���A�*

	conv_lossۯ>����        )��P	�,���A�*

	conv_loss��>y��h        )��P	�<,���A�*

	conv_loss�P�>670�        )��P	w,���A�*

	conv_loss�@�> �#         )��P	1�,���A�*

	conv_lossӷ�>��I�        )��P	��,���A�*

	conv_loss��>H��(        )��P	�,���A�*

	conv_loss9��> ]P�        )��P	�<,���A�*

	conv_loss�è>�j�        )��P	@l,���A�*

	conv_lossdW�>�L�        )��P	۞,���A�*

	conv_loss�*�>�A#�        )��P	,�,���A�*

	conv_loss��>62��        )��P	u,���A�*

	conv_loss�B�>� S�        )��P	�1,���A�*

	conv_loss��>V�o        )��P	)a,���A�*

	conv_loss�3�>K���        )��P	��,���A�*

	conv_loss$�>���        )��P	B�,���A�*

	conv_lossaǬ>�Dd�        )��P	v�,���A�*

	conv_loss��>v�        )��P	#,���A�*

	conv_loss~۩>�V�i        )��P	cS,���A�*

	conv_loss�{�>���^        )��P	��,���A�*

	conv_losse�>A�#�        )��P	�,���A�*

	conv_loss��>Nb�Z        )��P	��,���A�*

	conv_losss��>�M�        )��P	�,���A�*

	conv_loss�>�>�a��        )��P	�H,���A�*

	conv_loss�&�>�E��        )��P	�x,���A�*

	conv_loss{��>5�F        )��P	g�,���A�*

	conv_loss+��>/X��        )��P	��,���A�*

	conv_loss#C�>��;        )��P	�,���A�*

	conv_loss�٪>�'        )��P	�8,���A�*

	conv_loss�Ȧ> 1R        )��P	Zs,���A�*

	conv_loss�֮>Q�0�        )��P	ϧ,���A�*

	conv_loss��>۹�        )��P	��,���A�*

	conv_loss���>Dm�        )��P	�	,���A�*

	conv_lossb�>x;�&        )��P	��,���A�*

	conv_loss� �>AG�>        )��P	��,���A�*

	conv_lossH$�>G3X        )��P	�,���A�*

	conv_loss"ԩ>#��        )��P	�",���A�*

	conv_loss��>����        )��P	�R,���A�*

	conv_loss�(�>|kQ        )��P	{�,���A�*

	conv_loss%�>IQ�t        )��P	��,���A�*

	conv_lossJ�>��v�        )��P	��,���A�*

	conv_loss<��>!R�        )��P	4,���A�*

	conv_loss탪>t ��        )��P	�R,���A�*

	conv_lossߐ�>�m�m        )��P	�,���A�*

	conv_lossl!�>�s��        )��P	��,���A�*

	conv_loss:9�>̋�        )��P	��,���A�*

	conv_loss
��>�)        )��P	�,���A�*

	conv_loss���>pB�        )��P	z_,���A�*

	conv_lossN|�>�/+�        )��P	��,���A�*

	conv_loss���>|a�        )��P	��,���A�*

	conv_loss�Z�>Չ        )��P	��,���A�*

	conv_lossqӬ>wKJ/        )��P	F,���A�*

	conv_lossw5�>R�xW        )��P	P,���A�*

	conv_loss��>�)})        )��P	O~,���A�*

	conv_lossj�>�;        )��P	r�,���A�*

	conv_lossn�>���        )��P	*�,���A�*

	conv_loss/M�>{h�        )��P	K,���A�*

	conv_lossu�>�	��        )��P	o?,���A�*

	conv_loss��>��0        )��P	�o,���A�*

	conv_loss.h�>c'R�        )��P	1�,���A�*

	conv_loss�j�>&~��        )��P	^�,���A�*

	conv_loss��>�~�Q        )��P	K�,���A�*

	conv_lossC�>�'l        )��P	%-,���A�*

	conv_loss�5�>vɌ        )��P	�^,���A�*

	conv_loss��>��z"        )��P	%�,���A�*

	conv_lossVt�>q��'        )��P	Ѻ,���A�*

	conv_loss�Ԫ>�         )��P	n�,���A�*

	conv_loss��>�p        )��P	\,���A�*

	conv_loss��>U<��        )��P	�E,���A�*

	conv_lossm�>��n�        )��P	t,���A�*

	conv_loss?r�>Ca�        )��P	��,���A�*

	conv_loss嚨>P}�        )��P	�,���A�*

	conv_loss��>���        )��P	�,���A�*

	conv_loss"��>3�?�        )��P	*6,���A�*

	conv_lossI�>����        )��P	�d,���A�*

	conv_lossǘ�>?��.        )��P	M�,���A�*

	conv_loss�ɬ>��$�        )��P	
�,���A�*

	conv_loss�,�>�q��        )��P	��,���A�*

	conv_loss��>�Zy        )��P	�$,���A�*

	conv_loss�6�>cN�&        )��P	T,���A�*

	conv_lossd��>ۃ        )��P	E�,���A�*

	conv_loss��>biU        )��P	H�,���A�*

	conv_losslB�>�DS        )��P	E�,���A�*

	conv_loss�l�>	���        )��P	,���A�*

	conv_loss��>A7g         )��P	�Z,���A�*

	conv_loss��>��1        )��P	ٍ,���A�*

	conv_loss�:�>&�(�        )��P	�,���A�*

	conv_loss���>��q)        )��P	��,���A�*

	conv_loss��>)0�<        )��P	^',���A�*

	conv_loss%ʩ>���/        )��P	�V,���A�*

	conv_loss���>��n        )��P	Ќ,���A�*

	conv_lossҌ�>�2X        )��P	��,���A�*

	conv_lossʠ�>���        )��P	��,���A�*

	conv_loss ��>"��        )��P	�',���A�*

	conv_lossh�>��S!        )��P	9W,���A�*

	conv_loss���>�        )��P	{�,���A�*

	conv_loss�S�>[�+|        )��P	=�,���A�*

	conv_losse?�>�;R�        )��P	��,���A�*

	conv_lossf��>BE�        )��P	T-,���A�*

	conv_losswЪ>�c��        )��P	�\,���A�*

	conv_lossm�>��(�        )��P	��,���A�*

	conv_loss�b�>*B�        )��P	ɹ,���A�*

	conv_lossw�>W|p        )��P	�,���A�*

	conv_loss2��>��E�        )��P	�,���A�*

	conv_loss7Z�>��1�        )��P	�M,���A�*

	conv_loss�ͩ>��B}        )��P		�,���A�*

	conv_loss��>���        )��P	�,���A�*

	conv_loss���>�x˛        )��P	��,���A�*

	conv_loss�>H��!        )��P	,���A�*

	conv_lossឩ>^�2�        )��P	(F,���A�*

	conv_loss�2�>.K �        )��P	mv,���A�*

	conv_loss�L�>�'��        )��P	�,���A�*

	conv_lossC��>��9        )��P	x�,���A�*

	conv_loss�ݪ>��j        )��P	�,���A�*

	conv_lossYJ�>�9�        )��P	�B,���A�*

	conv_loss�X�>�x��        )��P	7v,���A�*

	conv_lossv�>3�R        )��P	q�,���A�*

	conv_loss�>�β        )��P	
�,���A�*

	conv_losse��>J΅�        )��P	b	,���A�*

	conv_loss,��> ��|        )��P	�9,���A�*

	conv_loss� �>B��|        )��P	�k,���A�*

	conv_lossۧ>�ĉ         )��P	��,���A�*

	conv_lossTݩ>���        )��P	��,���A�*

	conv_loss64�>�\=�        )��P	s�,���A�*

	conv_lossa�>��p�        )��P	�+,���A�*

	conv_loss3�>/�G�        )��P	�\,���A�*

	conv_loss��>2�0        )��P	��,���A�*

	conv_loss��>�4��        )��P	й,���A�*

	conv_lossѶ�>+S|B        )��P	��,���A�*

	conv_loss��>�MqZ        )��P	� ,���A�*

	conv_lossb�>5bR6        )��P	�G ,���A�*

	conv_loss	�>-QI        )��P	%z ,���A�*

	conv_loss�ϩ>Yg        )��P	�� ,���A�*

	conv_loss���>�V        )��P	�� ,���A�*

	conv_losst/�>��        )��P	qr",���A�*

	conv_lossɂ�>�U�@        )��P	��",���A�*

	conv_loss���>6JQ        )��P	E�",���A�*

	conv_loss*��>:�P�        )��P	�#,���A�*

	conv_lossT��>�݉        )��P	�6#,���A�*

	conv_loss�>:��        )��P	�e#,���A�*

	conv_loss\ɨ>���        )��P	 �#,���A�*

	conv_loss(
�>1
�r        )��P	��#,���A�*

	conv_lossv�>/i�'        )��P	�$,���A�*

	conv_loss��>6��I        )��P	G$,���A�*

	conv_loss�Ϧ>W��        )��P	�v$,���A�*

	conv_loss�T�>���"        )��P	ӧ$,���A�*

	conv_loss��>�LΩ        )��P	~�$,���A�*

	conv_loss~J�>m�NW        )��P	�	%,���A�*

	conv_loss��>6w        )��P	�8%,���A�*

	conv_loss���>�s        )��P	�k%,���A�*

	conv_loss�I�>�?B�        )��P	��%,���A�*

	conv_loss��>���p        )��P	x�%,���A�*

	conv_loss��>��by        )��P	��%,���A�*

	conv_lossIΨ>���        )��P	-&,���A�*

	conv_loss��>�T        )��P	�\&,���A�*

	conv_loss�¨>���f        )��P	1�&,���A�*

	conv_loss`-�>۰��        )��P	��&,���A�*

	conv_loss�O�>I�5>        )��P	��&,���A�*

	conv_loss=٪>�6��        )��P	)',���A�*

	conv_lossW]�>p��n        )��P	wY',���A�*

	conv_loss	}�>o@^�        )��P	ň',���A�*

	conv_loss���>�l�f        )��P	��',���A�*

	conv_loss#s�>�̮        )��P	��',���A�*

	conv_loss9�>j��         )��P	�(,���A�*

	conv_lossSL�>�|�        )��P	�P(,���A�*

	conv_lossl��>��n�        )��P	 �(,���A�*

	conv_loss��>8R^        )��P	x�(,���A�*

	conv_loss��>�Ck�        )��P	k�(,���A�*

	conv_loss���>�?U�        )��P	�),���A�*

	conv_loss�t�>����        )��P	�>),���A�*

	conv_lossS&�>I��        )��P	�n),���A�*

	conv_loss���>�6��        )��P	C�),���A�*

	conv_losso�>���        )��P	'�),���A�*

	conv_lossܱ�>��>%        )��P	
�),���A�*

	conv_lossק�>6��        )��P	C.*,���A�*

	conv_loss]$�>�ܾQ        )��P	d]*,���A�*

	conv_lossI�>'��        )��P	��*,���A�*

	conv_loss�N�>�l��        )��P	��*,���A�*

	conv_loss*��>�9�        )��P	��*,���A�*

	conv_losse�>�2 ?        )��P	b +,���A�*

	conv_lossx�>�`��        )��P	oQ+,���A�*

	conv_loss4�>�Ye�        )��P	��+,���A�*

	conv_loss׃�>�fG        )��P	ޯ+,���A�*

	conv_loss��>�ܧ�        )��P	��+,���A�*

	conv_loss�ʥ>}�        )��P	�!,,���A�*

	conv_loss/e�>V�~�        )��P	�Q,,���A�*

	conv_loss���>�NV        )��P	J�,,���A�*

	conv_lossW-�>���        )��P	�,,���A�*

	conv_loss�m�>a�8�        )��P	��,,���A�*

	conv_loss��>�V�z        )��P	t-,���A�*

	conv_loss�S�>s�{�        )��P	�N-,���A�*

	conv_loss�d�>�go$        )��P	�~-,���A�*

	conv_loss�ݦ>(>��        )��P	��-,���A�*

	conv_loss��>
M�;        )��P	}�-,���A�*

	conv_loss�x�>k�X�        )��P	* .,���A�*

	conv_loss�C�>�\,�        )��P	jY.,���A�*

	conv_loss��>i�_�        )��P	��.,���A�*

	conv_loss�S�>�ԙ�        )��P	��.,���A�*

	conv_loss�ͨ>=(��        )��P	��.,���A�*

	conv_loss��>��r        )��P	!/,���A�*

	conv_loss#X�>&�        )��P	PR/,���A�*

	conv_loss��>�A��        )��P	X�/,���A�*

	conv_loss�{�>�        )��P	��/,���A�*

	conv_loss��>��@        )��P	T�/,���A�*

	conv_loss���>=� �        )��P	0,���A�*

	conv_loss��>�=p        )��P	mL0,���A�*

	conv_loss�>�Z��        )��P	I|0,���A�*

	conv_lossF�>-g�p        )��P	Ǭ0,���A�*

	conv_loss�>�>-�4o        )��P	��0,���A�*

	conv_lossf��>�k��        )��P	1,���A�*

	conv_losszc�>s5lH        )��P	�K1,���A�*

	conv_loss^��>�}H�        )��P	{}1,���A�*

	conv_loss饥>C��        )��P	��1,���A�*

	conv_lossT|�>h���        )��P	1�1,���A�*

	conv_loss�ڨ>V,        )��P	�2,���A�*

	conv_loss��>����        )��P	�I2,���A�*

	conv_lossU�>d?��        )��P	�w2,���A�*

	conv_loss���>����        )��P	e�2,���A�*

	conv_loss�5�>7��        )��P	��2,���A�*

	conv_loss�F�>&��        )��P	�3,���A�*

	conv_loss7Ϧ> �؛        )��P	93,���A�*

	conv_loss��>��!�        )��P	�h3,���A�*

	conv_loss�Z�>�O9        )��P	ɚ3,���A�*

	conv_loss�ȩ>����        )��P	]�3,���A�*

	conv_loss��>�F4        )��P	R�3,���A�*

	conv_lossa`�>�E�v        )��P	)/4,���A�*

	conv_loss#z�>+��        )��P	ha4,���A�*

	conv_lossLD�>d�QV        )��P	ړ4,���A�*

	conv_loss�R�>����        )��P	��4,���A�*

	conv_loss�>�Zo        )��P	b�4,���A�*

	conv_loss��>��&�        )��P	�+5,���A�*

	conv_loss|Ũ>���        )��P	�\5,���A�*

	conv_loss��>�<�/        )��P	'�5,���A�*

	conv_losss1�>�~�+        )��P	�5,���A�*

	conv_lossd�>��        )��P	u6,���A�*

	conv_loss���>�!vL        )��P	A6,���A�*

	conv_loss���>����        )��P	�r6,���A�*

	conv_loss>I�> o�t        )��P	�6,���A�*

	conv_loss��>)[5        )��P	�6,���A�*

	conv_loss?�>ѫ
        )��P	/7,���A�*

	conv_loss�#�>���q        )��P	�M7,���A�*

	conv_loss�d�>WV �        )��P	Q~7,���A�*

	conv_loss�Y�>��M�        )��P	��7,���A�*

	conv_loss���>#���        )��P	0�7,���A�*

	conv_lossF�>D���        )��P	�8,���A�*

	conv_loss;��>����        )��P	=U8,���A�*

	conv_loss!��>�N�`        )��P	��8,���A�*

	conv_loss��>*�D�        )��P	M�8,���A�*

	conv_loss���>O�+`        )��P	&�8,���A�*

	conv_loss�ʥ>��|;        )��P	<9,���A�*

	conv_loss\��>ƙpW        )��P	�W9,���A�*

	conv_loss0	�>*��*        )��P	r�9,���A�*

	conv_loss�`�>5=b�        )��P	��9,���A�*

	conv_lossj��>����        )��P	��9,���A�*

	conv_lossJ�>o��        )��P	�(:,���A�*

	conv_losss�>����        )��P	�[:,���A�*

	conv_loss�W�>�qp        )��P	�:,���A�*

	conv_lossӦ>]��        )��P	�:,���A�*

	conv_lossNz�>��̐        )��P	��:,���A�*

	conv_loss��>]�f%        )��P	?";,���A�*

	conv_loss;Ϩ>;��        )��P	�U;,���A�*

	conv_loss�8�>}P�3        )��P	{�;,���A�*

	conv_loss/ߦ>�y8�        )��P	��;,���A�*

	conv_loss6n�>��u�        )��P	�;,���A�*

	conv_lossW9�>,#XN        )��P	]<,���A�*

	conv_loss���>��.l        )��P	�X<,���A�*

	conv_loss˓�>-��        )��P	׉<,���A�*

	conv_loss��>��b        )��P	�<,���A�*

	conv_loss���>�DPz        )��P	��<,���A�*

	conv_lossCJ�>Q��        )��P	==,���A�*

	conv_lossg¥>D�_�        )��P	�O=,���A�*

	conv_loss��>%��        )��P	�=,���A�*

	conv_loss"��>9Ӷm        )��P	ɱ=,���A�*

	conv_lossO��>��ծ        )��P	�=,���A�*

	conv_lossb�>�	��        )��P	+>,���A�*

	conv_loss>�>�=$�        )��P	H>,���A�*

	conv_lossj
�>�X        )��P	�~>,���A�*

	conv_loss��>�^�        )��P	U�>,���A�*

	conv_loss�#�>AJd        )��P	��>,���A�*

	conv_lossC�>@���        )��P	R?,���A�*

	conv_loss}D�>���        )��P	�E?,���A�*

	conv_loss�o�>u`B�        )��P	�~?,���A�*

	conv_loss�x�>C̬        )��P	�?,���A�*

	conv_loss���>�@�Q        )��P	w�?,���A�*

	conv_loss龥>�?m        )��P	�'@,���A�*

	conv_loss���>��+9        )��P	EZ@,���A�*

	conv_loss^�>6š        )��P	@�@,���A�*

	conv_loss���>��:        )��P	S�@,���A�*

	conv_lossXĪ>�$�e        )��P	h�@,���A�*

	conv_lossXx�>]-A#        )��P	�'A,���A�*

	conv_loss��>c(�+        )��P	�ZA,���A�*

	conv_loss�\�>OH�~        )��P	c�A,���A�*

	conv_loss�<�>H���        )��P	��A,���A�*

	conv_loss���>Jww        )��P	�B,���A�*

	conv_loss���>C�+i        )��P	e<B,���A�*

	conv_loss�|�>�W�8        )��P	DB,���A�*

	conv_loss�:�>^�C�        )��P	��B,���A�*

	conv_lossC�>! x�        )��P	��B,���A�*

	conv_loss��>/4�        )��P	KC,���A�*

	conv_loss4�>u�        )��P	�IC,���A�*

	conv_loss��>�ڝ$        )��P	e{C,���A�*

	conv_loss�m�>Ͽ�)        )��P	íC,���A�*

	conv_loss[�>���        )��P	v�C,���A�*

	conv_lossmC�>��v        )��P	D,���A�*

	conv_lossɰ�>��^x        )��P	�AD,���A�*

	conv_loss�ɢ>[��        )��P	wsD,���A�*

	conv_loss7�>4���        )��P	��D,���A�*

	conv_loss/.�>bL�        )��P	�D,���A�*

	conv_loss��>�S�        )��P	{E,���A�*

	conv_loss�٢>@!"E        )��P	�BE,���A�*

	conv_loss�;�>|���        )��P	KuE,���A�*

	conv_lossC�>�?V�        )��P	e�E,���A�*

	conv_lossP|�>��!        )��P	)�E,���A�*

	conv_loss��>O�28        )��P	JF,���A�*

	conv_loss��>�6*�        )��P	1EF,���A�*

	conv_loss{�>%���        )��P	�vF,���A�*

	conv_lossc�>o�        )��P	�F,���A�*

	conv_loss��>}��        )��P	8�F,���A�*

	conv_loss�0�>��\�        )��P	aG,���A�*

	conv_loss��>&&��        )��P	�HG,���A�*

	conv_loss8�>�|0�        )��P	�|G,���A�*

	conv_loss���>  ,B        )��P	ͺG,���A�*

	conv_loss� �>�g�        )��P	]�G,���A�*

	conv_lossЦ>i��        )��P	KH,���A�*

	conv_loss�@�>�Y�        )��P	QH,���A�*

	conv_loss�A�>��6        )��P	g�H,���A�*

	conv_lossl�>��<T        )��P	��H,���A�*

	conv_loss�s�>�ڳ�        )��P	f�H,���A�*

	conv_loss���>���        )��P	�&I,���A�*

	conv_loss-�>���#        )��P	�WI,���A�*

	conv_lossX��>��x�        )��P	��I,���A�*

	conv_loss8�>����        )��P	��I,���A�*

	conv_loss$w�>SP�        )��P	��I,���A�*

	conv_loss,ߨ>�Y�        )��P	�,J,���A�*

	conv_loss�B�>��@        )��P	d�K,���A�*

	conv_loss�٤>��y        )��P	�L,���A�*

	conv_loss2#�>��[�        )��P	�7L,���A�*

	conv_loss�>�&�        )��P	�mL,���A�*

	conv_loss�A�>��#}        )��P	ӡL,���A�*

	conv_loss��>CSK�        )��P	�L,���A�*

	conv_lossw�>�x��        )��P	�M,���A�*

	conv_lossJ��>�박        )��P	FEM,���A�*

	conv_loss�d�>�V�        )��P	9xM,���A�*

	conv_loss�G�>���        )��P	��M,���A�*

	conv_loss"�>�=#�        )��P	(�M,���A�*

	conv_loss���>m��q        )��P	�N,���A�*

	conv_loss.W�>q�Aw        )��P	�GN,���A�*

	conv_loss��>Gv6�        )��P	4{N,���A�*

	conv_lossna�>��F        )��P	(�N,���A�*

	conv_loss�2�>�Tՠ        )��P	��N,���A�*

	conv_loss���>[�ǖ        )��P	-O,���A�*

	conv_loss
p�>4�        )��P	�`O,���A�*

	conv_loss�ۤ>���        )��P	�O,���A�*

	conv_losse�>s��        )��P	p�O,���A�*

	conv_loss�j�>��K�        )��P	��O,���A�*

	conv_loss���>�Z��        )��P	'P,���A�*

	conv_loss�)�> l�	        )��P	:ZP,���A�*

	conv_loss �>c4�*        )��P	��P,���A�*

	conv_loss�B�>���U        )��P	h�P,���A�*

	conv_lossF��>d��        )��P	��P,���A�*

	conv_loss�ť>�h	\        )��P	\'Q,���A�*

	conv_loss�N�>ү�        )��P	�XQ,���A�*

	conv_loss<��>G�ߗ        )��P	�Q,���A�*

	conv_lossX��>�I�H        )��P	ʻQ,���A�*

	conv_loss�y�>� �	        )��P	5�Q,���A�*

	conv_lossŰ�>�V��        )��P	� R,���A�*

	conv_loss��>��"�        )��P	�TR,���A�*

	conv_loss���>a���        )��P	��R,���A�*

	conv_loss��>ċ�        )��P	4�R,���A�*

	conv_lossԪ�>�Y�"        )��P	 �R,���A�*

	conv_loss�L�>JrJ8        )��P	�S,���A�*

	conv_lossĥ>�m|        )��P	RQS,���A�*

	conv_loss���>Z�&        )��P	��S,���A�*

	conv_losse��>�N�        )��P	��S,���A�*

	conv_loss��>��H        )��P	J�S,���A�*

	conv_lossќ�>�&��        )��P	-T,���A�*

	conv_loss���>B�ʅ        )��P		_T,���A�*

	conv_loss���>ه*�        )��P	u�T,���A�*

	conv_lossc@�>O�        )��P	
�T,���A�*

	conv_lossw�>�S�I        )��P	%�T,���A�*

	conv_loss��>D�        )��P	�#U,���A�*

	conv_loss,
�>6�u        )��P	�VU,���A�*

	conv_loss�'�>	�^6        )��P	�U,���A�*

	conv_lossC9�>ʊ��        )��P	��U,���A�*

	conv_loss��>���        )��P	�V,���A�*

	conv_loss�v�>L��\        )��P	�4V,���A�*

	conv_loss��>���!        )��P	zgV,���A�*

	conv_loss{��>����        )��P	��V,���A�*

	conv_loss�H�>� 1        )��P	��V,���A�*

	conv_loss�ǥ>��]�        )��P	��V,���A�*

	conv_loss}��>�ذ�        )��P	�6W,���A�*

	conv_lossӳ�>��+        )��P	SjW,���A�*

	conv_loss�>��\        )��P	ɢW,���A�*

	conv_loss���>�ޅ�        )��P	�W,���A�*

	conv_lossn�>��So        )��P	�
X,���A�*

	conv_loss��>d8�        )��P	�AX,���A�*

	conv_lossT8�>�>)        )��P	zrX,���A�*

	conv_loss���>c��        )��P	��X,���A�*

	conv_lossD��><6m         )��P	��X,���A�*

	conv_loss~f�>BV�w        )��P	�Y,���A�*

	conv_loss�`�>_���        )��P	�FY,���A�*

	conv_loss}�>E��        )��P	�xY,���A�*

	conv_lossF̣>}���        )��P	׫Y,���A�*

	conv_loss.��>�:�q        )��P	a�Y,���A�*

	conv_loss�s�>��+�        )��P	�Z,���A�*

	conv_loss:4�>�{�        )��P	yEZ,���A�*

	conv_loss�P�>Sio         )��P	4xZ,���A�*

	conv_loss�}�>����        )��P	h�Z,���A�*

	conv_loss��>��        )��P	s�Z,���A�*

	conv_loss�>u֬        )��P	 [,���A�*

	conv_loss���>�        )��P	=[,���A�*

	conv_loss���>��/        )��P	o[,���A�*

	conv_loss�J�>��        )��P	j�[,���A�*

	conv_loss]��>W�2�        )��P	�[,���A�*

	conv_loss}��>�� �        )��P	\,���A�*

	conv_loss�L�>8�(p        )��P	�7\,���A�*

	conv_lossx�>2�<        )��P	7j\,���A�*

	conv_lossmR�>�A��        )��P	�\,���A�*

	conv_loss��>"?�        )��P	�\,���A�*

	conv_loss�T�>DbL�        )��P	��\,���A�*

	conv_loss~�>���S        )��P	0],���A�*

	conv_loss�ڣ>o��V        )��P	ia],���A�*

	conv_loss�>�	�        )��P	��],���A�*

	conv_loss`��>9G��        )��P	��],���A�*

	conv_loss���>��        )��P	{�],���A�*

	conv_lossM��>F��        )��P	�(^,���A�*

	conv_lossֿ�>Y��        )��P	@[^,���A�*

	conv_loss��>�^;^        )��P	`�^,���A�*

	conv_loss��>6�=
        )��P	��^,���A�*

	conv_lossc�>s_�!        )��P	��^,���A�*

	conv_lossEV�>��        )��P	&!_,���A�*

	conv_lossL�><ߌ�        )��P	�Q_,���A�*

	conv_loss�'�>�G�i        )��P	=�_,���A�*

	conv_loss�g�>�>�        )��P	��_,���A�*

	conv_loss ��><�"        )��P	��_,���A�*

	conv_loss-�>�Â        )��P	/`,���A�*

	conv_loss`�>�F
        )��P	$``,���A�*

	conv_lossl��>sR�o        )��P	�`,���A�*

	conv_loss�i�>�l�        )��P	(�`,���A�*

	conv_loss���>��;�        )��P	+�`,���A�*

	conv_lossZ��>�Y�!        )��P	'-a,���A�*

	conv_loss���>6|�        )��P	*ka,���A�*

	conv_lossY��>7���        )��P	��a,���A�*

	conv_loss�>���        )��P	z�a,���A�*

	conv_loss�̤>V�R        )��P	Db,���A�*

	conv_loss#��>����        )��P	{>b,���A�*

	conv_lossI$�>�ya        )��P	k}b,���A�*

	conv_loss8�>�5j/        )��P	��b,���A�*

	conv_loss��>GM��        )��P	��b,���A�*

	conv_loss�֤>�\X�        )��P	#c,���A�*

	conv_loss��>Ů#V        )��P	9Qc,���A�*

	conv_loss⻣>ď@�        )��P	A�c,���A�*

	conv_loss'�>�ļ        )��P	��c,���A�*

	conv_loss��>Z�j        )��P	!�c,���A�*

	conv_loss� �>�<GD        )��P	�!d,���A�*

	conv_lossX��>�y��        )��P	Sd,���A�*

	conv_loss�g�>���6        )��P	��d,���A�*

	conv_loss��>X^y        )��P	�d,���A�*

	conv_lossk��>�        )��P	��d,���A�*

	conv_loss��>�J�M        )��P	�e,���A�*

	conv_loss�-�>;9��        )��P	}Le,���A�*

	conv_loss�S�>Z0�        )��P	�e,���A�*

	conv_lossGp�>G�x�        )��P	�e,���A�*

	conv_loss�š>���P        )��P	��e,���A�*

	conv_lossPj�>`��        )��P	�f,���A�*

	conv_lossNѣ>ڨ��        )��P	=Ff,���A�*

	conv_lossGt�>�)��        )��P	�yf,���A�*

	conv_loss�>f��        )��P	߬f,���A�*

	conv_loss8
�>����        )��P	P�f,���A�*

	conv_loss-��>�Q        )��P	�g,���A�*

	conv_loss�à>gKel        )��P	Dg,���A�*

	conv_loss�z�>#L1�        )��P	:zg,���A�*

	conv_loss��>#&(�        )��P	�g,���A�*

	conv_lossh|�>bl�        )��P	��g,���A�*

	conv_loss��>�7Y        )��P	0h,���A�*

	conv_loss�٣>��
v        )��P	CAh,���A�*

	conv_lossj�>3Ce        )��P	luh,���A�*

	conv_lossx!�>��'�        )��P	�h,���A�*

	conv_loss�ע>�B��        )��P	��h,���A�*

	conv_lossM<�>� i�        )��P	ii,���A�*

	conv_loss0��>�,J�        )��P	�Ci,���A�*

	conv_lossˍ�>A?dk        )��P	�yi,���A�*

	conv_lossH�>��i�        )��P	�i,���A�*

	conv_lossd�>6�@�        )��P	��i,���A�*

	conv_loss4��>t��        )��P	�$j,���A�*

	conv_lossuk�>�V�        )��P	&Wj,���A�*

	conv_loss�ס>�k��        )��P	��j,���A�*

	conv_loss���>�u7         )��P	L�j,���A�*

	conv_lossNo�>���q        )��P	��j,���A�*

	conv_loss	��>�$3,        )��P	&'k,���A�*

	conv_lossp}�>Cw        )��P	Xk,���A�*

	conv_loss"��>a�&        )��P	��k,���A�*

	conv_loss�(�>Q��        )��P	��k,���A�*

	conv_losss|�>�F�        )��P	��k,���A�*

	conv_loss��>�OKD        )��P	�/l,���A�*

	conv_loss�G�>,7X7        )��P	al,���A�*

	conv_loss/L�>=vD�        )��P	#�l,���A�*

	conv_loss._�>"HH�        )��P	\�l,���A�*

	conv_lossu��>��\�        )��P	Y�l,���A�*

	conv_loss�ݥ>	�/^        )��P	1m,���A�*

	conv_loss�ԥ>?t3        )��P	�fm,���A�*

	conv_lossy'�>���\        )��P	m�m,���A�*

	conv_lossɤ>h4�%        )��P	f�m,���A�*

	conv_loss$��>�=�        )��P	�n,���A�*

	conv_loss�ѣ>�l�        )��P	�3n,���A�*

	conv_loss� >��ӎ        )��P	afn,���A�*

	conv_loss���>�ʌ�        )��P	��n,���A�*

	conv_loss��>�*L&        )��P	�n,���A�*

	conv_loss��>E>8�        )��P	r�n,���A�*

	conv_loss��>5��        )��P	�0o,���A�*

	conv_lossݤ�>po��        )��P	#co,���A�*

	conv_lossܤ>B        )��P	��o,���A�*

	conv_loss�>)�?�        )��P	��o,���A�*

	conv_lossg��>*c�r        )��P	<�o,���A�*

	conv_loss�Ƣ>��Y�        )��P	T'p,���A�*

	conv_loss�ҡ>���        )��P	5Xp,���A�*

	conv_loss���>�W��        )��P	ɉp,���A�*

	conv_loss[��>��x        )��P	�p,���A�*

	conv_loss���>�TE�        )��P	��p,���A�*

	conv_lossWS�>����        )��P	Cq,���A�*

	conv_loss���>]1��        )��P	$Rq,���A�*

	conv_loss>;�>����        )��P	��q,���A�*

	conv_lossq��>���        )��P	��q,���A�*

	conv_loss���>�w��        )��P	��q,���A�*

	conv_lossDC�>�W/<        )��P	wr,���A�*

	conv_loss1�>�Q         )��P	PJr,���A�*

	conv_lossE�>�QB        )��P	�|r,���A�*

	conv_lossu}�>[�L        )��P	��r,���A�*

	conv_loss-��>�>M�        )��P	_�r,���A�*

	conv_loss*D�>���T        )��P	�s,���A�*

	conv_loss>�<&"        )��P	BGs,���A�*

	conv_loss��>bSGC        )��P	�xs,���A�*

	conv_loss5�>�I [        )��P	`�s,���A�*

	conv_loss/-�>��;        )��P	�x,���A�*

	conv_loss B�>Me        )��P	N'z,���A�*

	conv_loss��>�\�        )��P	�Wz,���A�*

	conv_loss�n�>C���        )��P	Ԇz,���A�*

	conv_lossܢ>͠�9        )��P	q�z,���A�*

	conv_loss�̡>	�+�        )��P	��z,���A�*

	conv_loss�>���x        )��P	9({,���A�*

	conv_loss�7�>��	        )��P	*Z{,���A�*

	conv_loss�̢>x�H6        )��P	ӈ{,���A�*

	conv_loss�^�>�[z#        )��P	8�{,���A�*

	conv_lossn?�>�Ti        )��P	��{,���A�*

	conv_loss<?�>�/�        )��P	�|,���A�*

	conv_loss��>���*        )��P	Y|,���A�*

	conv_loss<�>�Z�r        )��P	��|,���A�*

	conv_loss�l�>���_        )��P	�|,���A�*

	conv_lossƬ�>�U��        )��P	��|,���A�*

	conv_loss0\�>,ɰU        )��P	�},���A�*

	conv_loss
v�>2\��        )��P	@[},���A�*

	conv_loss	��>���I        )��P	�},���A�*

	conv_loss2��>*        )��P	P�},���A�*

	conv_lossݓ�>�q�u        )��P	X�},���A�*

	conv_loss�+�>�:��        )��P	�~,���A�*

	conv_lossJM�>�L�        )��P	)K~,���A�*

	conv_loss"Ν>V/�        )��P	Qz~,���A�*

	conv_lossǛ�>�Q�        )��P	��~,���A�*

	conv_loss㕢>P�8�        )��P	��~,���A�*

	conv_loss/��>��        )��P	<,���A�*

	conv_loss~i�>� �        )��P	_G,���A�*

	conv_loss���>���r        )��P	2x,���A�*

	conv_loss��>U��        )��P	&�,���A�*

	conv_loss��>�գ1        )��P	�,���A�*

	conv_loss"��>p62        )��P	�	�,���A�*

	conv_loss}��>�J�        )��P	�7�,���A�*

	conv_loss<�>��.�        )��P	&g�,���A�*

	conv_lossUX�>�	[�        )��P	���,���A�*

	conv_loss4I�>��#        )��P	eȀ,���A�*

	conv_lossբ>���        )��P	��,���A�*

	conv_losstx�>o�{        )��P	h*�,���A�*

	conv_loss�>��1�        )��P	{Z�,���A�*

	conv_lossAo�>c�        )��P	��,���A�*

	conv_loss�֡>���        )��P	m��,���A�*

	conv_loss7#�>;=G        )��P	��,���A�*

	conv_loss���>���        )��P	"�,���A�*

	conv_lossL�>X3-�        )��P	oP�,���A�*

	conv_loss���>�~[        )��P	n�,���A�*

	conv_lossؚ�>;�ě        )��P	���,���A�*

	conv_loss7��>��Y�        )��P	j�,���A�*

	conv_loss�ա>B�^�        )��P	��,���A�*

	conv_losse:�>	        )��P	�E�,���A�*

	conv_lossB��>�u�        )��P	�u�,���A�*

	conv_loss.�>�D        )��P	W��,���A�*

	conv_losszF�>�o?�        )��P	��,���A�*

	conv_loss��>>l��        )��P	{�,���A�*

	conv_loss��>^��/        )��P	~G�,���A�*

	conv_loss�`�>j�BN        )��P	�w�,���A�*

	conv_loss���>�� �        )��P	q��,���A�*

	conv_loss���>mkf�        )��P	�ڄ,���A�*

	conv_loss1��>�L�3        )��P	r�,���A�*

	conv_loss(}�>��:        )��P	�>�,���A�*

	conv_loss��>��kx        )��P	@o�,���A�*

	conv_lossW��>z1�        )��P	�,���A�*

	conv_loss�G�>�nj�        )��P	�م,���A�*

	conv_loss��>��         )��P	�	�,���A�*

	conv_lossy��>	%        )��P	�=�,���A�*

	conv_loss�U�>]��        )��P	�t�,���A�*

	conv_loss���>
��        )��P	I��,���A�*

	conv_loss��>n���        )��P	"ކ,���A�*

	conv_loss�f�>�(3�        )��P	��,���A�*

	conv_loss�x�>�suv        )��P	�?�,���A�*

	conv_loss6�>eS\        )��P	o�,���A�*

	conv_losso��>w
k        )��P	0��,���A�*

	conv_loss}ՠ>���        )��P	͇,���A�*

	conv_lossC�>����        )��P	���,���A�*

	conv_loss�Q�>�R�        )��P	4*�,���A�*

	conv_loss�O�>�}[O        )��P	�X�,���A�*

	conv_loss���>gWl�        )��P	艈,���A�*

	conv_loss:�>!_5        )��P	ݺ�,���A�*

	conv_loss_��>��        )��P	q�,���A�*

	conv_loss�>�a�        )��P	(�,���A�*

	conv_loss�۠>-���        )��P	H�,���A�*

	conv_loss�,�>z�4        )��P	�w�,���A�*

	conv_loss���>��kg        )��P	ɦ�,���A�*

	conv_lossp�>�F%_        )��P	�։,���A�*

	conv_lossq��>AR��        )��P	��,���A�*

	conv_loss���>o|��        )��P	�5�,���A�*

	conv_lossn�>7�=�        )��P	dg�,���A�*

	conv_lossd��>��B        )��P	×�,���A�*

	conv_loss�Ӣ>�B��        )��P	]Ǌ,���A�*

	conv_loss�>�W�        )��P	x��,���A�*

	conv_losse��>�#'�        )��P	�$�,���A�*

	conv_loss���>~;de        )��P	{S�,���A�*

	conv_loss j�>��        )��P	 ��,���A�*

	conv_loss��>k4��        )��P	(��,���A�*

	conv_lossW)�>�eʹ        )��P	f�,���A�*

	conv_lossn�>oZ}�        )��P	i�,���A�*

	conv_loss��>�:�        )��P	�D�,���A�*

	conv_loss�k�>�|�0        )��P	�s�,���A�*

	conv_lossd�>xv>        )��P	>��,���A�*

	conv_loss�x�>)o�        )��P	_Ԍ,���A�*

	conv_loss �>�Y`p        )��P	�,���A�*

	conv_loss՘�>����        )��P	�4�,���A�*

	conv_loss�>��o�        )��P	�x�,���A�*

	conv_loss� >U�0        )��P	n��,���A�*

	conv_loss���>�6@>        )��P	vڍ,���A�*

	conv_loss?
�>8֋�        )��P	�,���A�*

	conv_lossI̡>�QR        )��P	4@�,���A�*

	conv_loss))�>(��        )��P	�r�,���A�*

	conv_lossOk�>��%        )��P	6��,���A�*

	conv_lossҺ�>���        )��P	�֎,���A�*

	conv_lossN�>�bP�        )��P	��,���A�*

	conv_loss�|�>E�_h        )��P	7H�,���A�*

	conv_loss�>B��        )��P	�z�,���A�*

	conv_lossڴ�>�JG        )��P	���,���A�*

	conv_loss�L�>5JN�        )��P	I�,���A�*

	conv_loss��>���        )��P	��,���A�*

	conv_lossH��>9|�K        )��P	bS�,���A�*

	conv_loss̫�>�:�        )��P	���,���A�*

	conv_lossR��>�ޗ        )��P	E��,���A�*

	conv_loss��>��A�        )��P	��,���A�*

	conv_loss4�>�%�:        )��P	��,���A�*

	conv_loss\F�>�x�P        )��P	lP�,���A�*

	conv_loss�@�>���5        )��P	E��,���A�*

	conv_loss��>k�=�        )��P	k��,���A�*

	conv_lossЀ�>c"�l        )��P	�,���A�*

	conv_loss��>p�F$        )��P	8�,���A�*

	conv_loss_֠>�]��        )��P	�C�,���A�*

	conv_loss{ޞ>���        )��P	�s�,���A�*

	conv_loss2�>,a��        )��P	5��,���A�*

	conv_loss(M�>�ۖ�        )��P	_Ӓ,���A�*

	conv_loss�g�>��        )��P	��,���A�*

	conv_loss�Ơ>u�(        )��P	k4�,���A�*

	conv_loss�H�>R�U        )��P	�e�,���A�*

	conv_loss�S�>�=��        )��P	��,���A�*

	conv_loss�Ο>Nv�        )��P	
Ɠ,���A�*

	conv_loss���>�_�1        )��P	���,���A�*

	conv_lossI�>��9�        )��P	�%�,���A�*

	conv_lossSM�>'v�        )��P	�U�,���A�*

	conv_loss釞>c��        )��P	���,���A�*

	conv_loss�X�>r�q        )��P	2��,���A�*

	conv_loss��>�r�        )��P	_�,���A�*

	conv_lossN�>��:P        )��P	��,���A�*

	conv_loss�}�>ȁ��        )��P	�J�,���A�*

	conv_losseǝ>��3        )��P	�{�,���A�*

	conv_loss�Ĝ>���        )��P	���,���A�*

	conv_loss3�>��JA        )��P	U��,���A�*

	conv_lossؚ�>Hi�        )��P	f�,���A�*

	conv_loss&c�>}��*        )��P	PF�,���A�*

	conv_loss�p�>hx�        )��P	�v�,���A�*

	conv_lossJ�>b�H)        )��P	���,���A�*

	conv_loss"�>%9,�        )��P	�ז,���A�*

	conv_loss�۞>��qr        )��P	��,���A�*

	conv_loss���>�8��        )��P	�K�,���A�*

	conv_loss�9�>"��        )��P	�|�,���A�*

	conv_loss�ՠ>�~p        )��P	���,���A�*

	conv_loss��>ȵ��        )��P	t�,���A�*

	conv_loss��>���        )��P	w!�,���A�*

	conv_loss鼝>��        )��P	�T�,���A�*

	conv_loss�)�>��5-        )��P	��,���A�*

	conv_lossM��>��Ir        )��P	W��,���A�*

	conv_loss���>R�W        )��P	!�,���A�*

	conv_lossr�>�Y~�        )��P	�,�,���A�*

	conv_lossk�>��t3        )��P	 ]�,���A�*

	conv_lossn�>b�X        )��P	���,���A�*

	conv_loss5�>m�]        )��P	d��,���A�*

	conv_lossn�>_�;Y        )��P	��,���A�*

	conv_lossU��>�TM2        )��P	J'�,���A�*

	conv_loss��>v%�=        )��P	,X�,���A�*

	conv_loss�t�>T��        )��P	��,���A�*

	conv_loss 	�>$�Q�        )��P	�Ú,���A�*

	conv_loss'=�>!���        )��P	���,���A�*

	conv_losss��>݂��        )��P	�%�,���A�*

	conv_losspğ>8S        )��P	�U�,���A�*

	conv_loss�j�>F��_        )��P	��,���A�*

	conv_lossT�>Y��        )��P	Q��,���A�*

	conv_loss=�>�Hb1        )��P	�,���A�*

	conv_loss��>λJ�        )��P	��,���A�*

	conv_loss�V�>`l�        )��P	sJ�,���A�*

	conv_loss�>u���        )��P	_|�,���A�*

	conv_lossFם>ip؇        )��P	r��,���A�*

	conv_loss_��> `�        )��P	�ܜ,���A�*

	conv_loss7�>Zg�f        )��P	�,���A�*

	conv_loss/��>�a�?        )��P	�@�,���A�*

	conv_loss�{�>z>#        )��P	�t�,���A�*

	conv_lossr��>��s        )��P	��,���A�*

	conv_loss�֟>��Cz        )��P	"ٝ,���A�*

	conv_loss�>��g�        )��P	��,���A�*

	conv_loss"��>,QG�        )��P	J:�,���A�*

	conv_loss��>e!e�        )��P	�m�,���A�*

	conv_loss��>��m�        )��P	z��,���A�*

	conv_loss��>�        )��P	�Ҟ,���A�*

	conv_lossX��>ܼQx        )��P	Z�,���A�*

	conv_lossoZ�>���        )��P	h5�,���A�*

	conv_losse��>i�@w        )��P	�f�,���A�*

	conv_lossr�>Sk�.        )��P	�,���A�*

	conv_loss�h�>�_ B        )��P	�ǟ,���A�*

	conv_loss��>�nB        )��P	^��,���A�*

	conv_loss�U�>Rz�}        )��P	�,�,���A�*

	conv_lossݍ�>�^�z        )��P	�_�,���A�*

	conv_loss���>J�5        )��P	�,���A�*

	conv_loss�t�>�ٶ7        )��P	rĠ,���A�*

	conv_loss&2�>=�ݺ        )��P	`��,���A�*

	conv_lossIg�>n��        )��P	���,���A�*

	conv_loss���>�^�D        )��P	�̢,���A�*

	conv_lossD��>O�Y�        )��P	U��,���A�*

	conv_loss�8�>#��d        )��P	�-�,���A�*

	conv_loss�Q�>Gc�
        )��P	�]�,���A�*

	conv_lossꓞ>���        )��P		��,���A�*

	conv_loss��>��4m        )��P	uͣ,���A�*

	conv_losss�>Q��{        )��P	��,���A�*

	conv_loss,��>���        )��P	y6�,���A�*

	conv_loss�#�>�E(�        )��P	�h�,���A�*

	conv_loss�P�>��wq        )��P	��,���A�*

	conv_loss��>��p�        )��P	�ߤ,���A�*

	conv_loss��>�f7�        )��P	>�,���A�*

	conv_loss2�>$��        )��P	JC�,���A�*

	conv_loss���>��n"        )��P	Yt�,���A�*

	conv_loss�~�>7{��        )��P	E��,���A�*

	conv_loss�y�>f=-        )��P	�֥,���A�*

	conv_lossN�>�.�c        )��P		�,���A�*

	conv_loss��>3��        )��P	Z<�,���A�*

	conv_loss��>��        )��P	�l�,���A�*

	conv_lossj�>�5A        )��P	�,���A�*

	conv_loss`�>tc        )��P	MϦ,���A�*

	conv_lossQ��>&���        )��P	���,���A�*

	conv_loss��>v�D&        )��P	�/�,���A�*

	conv_loss�`�>��        )��P	A`�,���A�*

	conv_loss�I�>J �b        )��P	���,���A�*

	conv_losscz�>�@�l        )��P	�ç,���A�*

	conv_loss��>���        )��P	���,���A�*

	conv_lossG'�>�y��        )��P	o(�,���A�*

	conv_losset�>���T        )��P	�X�,���A�*

	conv_loss�>2���        )��P	��,���A�*

	conv_loss<�>�bm�        )��P	���,���A�*

	conv_loss?:�>��t�        )��P	�,���A�*

	conv_loss�ʝ>���        )��P	�!�,���A�*

	conv_lossjƝ>2co�        )��P	R�,���A�*

	conv_loss)�>jn�        )��P	1��,���A�*

	conv_loss��>CB�R        )��P	��,���A�*

	conv_lossx՟><A Q        )��P	#�,���A�*

	conv_loss붛>�3	        )��P	%�,���A�*

	conv_loss
Ǜ>�;k        )��P	JH�,���A�*

	conv_loss�:�>[џh        )��P	�x�,���A�*

	conv_loss���>[k�q        )��P	A��,���A�*

	conv_lossm�>W}2�        )��P	�ܪ,���A�*

	conv_lossJ^�>���V        )��P	D�,���A�*

	conv_loss�ț>��r        )��P	�@�,���A�*

	conv_loss��>�(=�        )��P	�q�,���A�*

	conv_loss�q�>
�         )��P	>��,���A�*

	conv_losssv�>�'k�        )��P	�ӫ,���A�*

	conv_loss�w�>h��        )��P	%�,���A�*

	conv_loss��>ぢ�        )��P	O:�,���A�*

	conv_loss`j�>N��g        )��P	U��,���A�*

	conv_lossF�>�w        )��P	Ѹ�,���A�*

	conv_lossGQ�>�*        )��P	��,���A�*

	conv_loss��>֛��        )��P	�"�,���A�*

	conv_lossq3�>w�Oj        )��P	�V�,���A�*

	conv_loss_��>�8��        )��P	x��,���A�*

	conv_loss�b�>�@�X        )��P	X��,���A�*

	conv_loss���>$ٓ        )��P	Z��,���A�*

	conv_loss�ӛ>x6w�        )��P	2�,���A�*

	conv_loss�>��	�        )��P	�f�,���A�*

	conv_loss�d�>k�ws        )��P	I��,���A�*

	conv_loss���>|��        )��P	�خ,���A�*

	conv_loss��>9�>        )��P	��,���A�*

	conv_loss;%�>�ݨ�        )��P	sL�,���A�*

	conv_loss�$�>��        )��P	���,���A�*

	conv_loss�+�>�]        )��P	=��,���A�*

	conv_loss �>qP        )��P	��,���A�*

	conv_loss�6�>Il�~        )��P	\�,���A�*

	conv_lossز�>�C�        )��P	�J�,���A�*

	conv_loss�&�>j�e        )��P	,~�,���A�*

	conv_lossÁ�>�[Y,        )��P	(��,���A�*

	conv_loss�Μ>��'�        )��P	l�,���A�*

	conv_loss���>&�+�        )��P	N�,���A�*

	conv_loss�>(���        )��P	L�,���A�*

	conv_loss@�>nl4        )��P	
~�,���A�*

	conv_loss��>�S��        )��P	���,���A�*

	conv_loss�h�>�5,�        )��P	1�,���A�*

	conv_loss��>E�M�        )��P	A�,���A�*

	conv_loss�a�>/4j�        )��P	�L�,���A�*

	conv_loss��>��g{        )��P	��,���A�*

	conv_loss�`�>�du        )��P	��,���A�*

	conv_lossgU�>�V/�        )��P	�,���A�*

	conv_loss/�>0�9�        )��P	g�,���A�*

	conv_loss�Š>�d        )��P	.H�,���A�*

	conv_loss\�>����        )��P	�y�,���A�*

	conv_loss��>3w�h        )��P	ڬ�,���A�*

	conv_lossL��>�'U�        )��P	m�,���A�*

	conv_loss$?�>q���        )��P		�,���A�*

	conv_loss��>X���        )��P	�K�,���A�*

	conv_loss���>���O        )��P	�,���A�*

	conv_loss�қ>N��L        )��P	���,���A�*

	conv_lossu �>�)<�        )��P	7�,���A�*

	conv_loss٫�>ĭx�        )��P	��,���A�*

	conv_lossK�>u	�m        )��P	fL�,���A�*

	conv_lossϓ�>���
        )��P	��,���A�*

	conv_loss��>���=        )��P	7��,���A�*

	conv_lossj��>�YQ�        )��P	��,���A�*

	conv_lossӫ�>W~�        )��P	g�,���A�*

	conv_lossl`�>wE��        )��P	L�,���A�*

	conv_loss��>���[        )��P	�,���A�*

	conv_loss"�>3��        )��P	�Ŷ,���A�*

	conv_lossc��>�@        )��P	M��,���A�*

	conv_lossN�>'��        )��P	�*�,���A�*

	conv_loss��>a_9�        )��P	b�,���A�*

	conv_loss�]�>��A�        )��P	K��,���A�*

	conv_loss�^�> �a        )��P	�ɷ,���A�*

	conv_loss�>�A��        )��P	���,���A�*

	conv_loss�v�>���        )��P	J<�,���A�*

	conv_lossM3�>t�&:        )��P	�s�,���A�*

	conv_loss�>�        )��P	,���A�*

	conv_loss�5�>L��F        )��P	��,���A�*

	conv_loss!��>��\.        )��P	��,���A�*

	conv_loss��>���        )��P	�W�,���A�*

	conv_loss&�>SfU	        )��P	`��,���A�*

	conv_loss:��>\/��        )��P		ù,���A�*

	conv_lossZ��>�xP�        )��P	O��,���A�*

	conv_lossZ�>YM?        )��P	3)�,���A�*

	conv_lossH��>iP��        )��P	\�,���A�*

	conv_loss5��>B^Ll        )��P	���,���A�*

	conv_loss��>1���        )��P	<ĺ,���A�*

	conv_lossPA�>�k�        )��P	:��,���A�*

	conv_loss�Y�>�X@�        )��P	�-�,���A�*

	conv_loss��>���        )��P	�d�,���A�*

	conv_loss Ś>�:        )��P	X��,���A�*

	conv_loss�%�>,BX        )��P	Gλ,���A�*

	conv_loss� �>-���        )��P	��,���A�*

	conv_lossX��>K��3        )��P	67�,���A�*

	conv_loss�d�>���        )��P	�j�,���A�*

	conv_loss゙>��nQ        )��P	���,���A�*

	conv_loss�d�>�        )��P	�μ,���A�*

	conv_loss���>�̅
        )��P	3�,���A�*

	conv_loss5��>Bp�(        )��P	F3�,���A�*

	conv_lossN{�>�{7        )��P	�f�,���A�*

	conv_lossZo�>�v�q        )��P	���,���A�*

	conv_loss��>�V�        )��P	�ҽ,���A�*

	conv_loss��>�k~        )��P	��,���A�*

	conv_loss���>�dz        )��P	J:�,���A�*

	conv_loss���>�O�*        )��P	6m�,���A�*

	conv_loss�S�>�I�        )��P	U��,���A�*

	conv_loss��>8X%        )��P	eվ,���A�*

	conv_loss=%�>)c��        )��P	y	�,���A�*

	conv_lossx:�>Sqfd        )��P	�=�,���A�*

	conv_loss�g�>Y��A        )��P	;q�,���A�*

	conv_loss��>.U��        )��P	��,���A�*

	conv_lossb�>k��        )��P	�׿,���A�*

	conv_loss��>�h�p        )��P	6�,���A�*

	conv_lossԘ>W&r$        )��P	�@�,���A�*

	conv_loss7��>r�8        )��P	vu�,���A�*

	conv_loss�ڙ>Rd�        )��P	���,���A�*

	conv_loss�t�>�C6�        )��P	��,���A�*

	conv_loss���>O��}        )��P	�&�,���A�*

	conv_loss"�>/��        )��P	R[�,���A�*

	conv_lossl�>���        )��P	���,���A�*

	conv_loss"�>WD9c        )��P	���,���A�*

	conv_loss��>0��        )��P	���,���A�*

	conv_loss��>]��        )��P	j)�,���A�*

	conv_losso��>���b        )��P	�[�,���A�*

	conv_loss��>�UR�        )��P	ҕ�,���A�*

	conv_lossVM�>>zv�        )��P	3��,���A�*

	conv_loss��>�3�        )��P	�,���A�*

	conv_loss��>�_E        )��P	�H�,���A�*

	conv_loss���>��3        )��P	���,���A�*

	conv_lossݷ�>�Ϙ        )��P	ѽ�,���A�*

	conv_lossXT�>��bD        )��P	���,���A�*

	conv_loss���>؀[�        )��P	�"�,���A�*

	conv_lossJ5�>�O�        )��P	pU�,���A�*

	conv_lossO��>��        )��P	q��,���A�*

	conv_lossn��>A���        )��P	���,���A�*

	conv_loss�ɜ>a�-�        )��P	���,���A�*

	conv_loss�/�>-oX#        )��P	�$�,���A�*

	conv_lossyQ�>@�<=        )��P	^�,���A�*

	conv_loss�ɝ>�6f�        )��P	ѓ�,���A�*

	conv_loss�j�>�+�>        )��P	���,���A�*

	conv_loss�>�m�        )��P	Q��,���A�*

	conv_loss��>�zK�        )��P	�0�,���A�*

	conv_lossϛ>���y        )��P	�d�,���A�*

	conv_loss�Z�>ǚ        )��P	ݖ�,���A�*

	conv_lossb��>D>�        )��P	_��,���A�*

	conv_loss�~�>�        )��P	&�,���A�*

	conv_loss>���        )��P	9�,���A�*

	conv_lossih�>�ݚ�        )��P	m�,���A�*

	conv_loss�[�>[ �        )��P	_��,���A�*

	conv_loss��>��~�        )��P	r��,���A�*

	conv_loss?O�>��        )��P	*"�,���A�*

	conv_lossԕ>t��        )��P	 W�,���A�*

	conv_loss!�>	]$�        )��P	e��,���A�*

	conv_loss�k�>�7z        )��P	h��,���A�*

	conv_loss�B�>���u        )��P	��,���A�*

	conv_loss)v�>�:n        )��P	�"�,���A�*

	conv_loss�ė>ܔ�        )��P	�V�,���A�*

	conv_lossb�>���W        )��P	Ί�,���A�*

	conv_lossv��>��&�        )��P	W��,���A�*

	conv_loss���>�G�w        )��P	��,���A�*

	conv_loss(�>ª��        )��P	�&�,���A�*

	conv_lossC��>���        )��P	Y�,���A�*

	conv_loss�D�>���4        )��P	��,���A�*

	conv_loss
��>�0 �        )��P	_��,���A�*

	conv_loss��>���        )��P	��,���A�*

	conv_loss"1�>.��        )��P	E'�,���A�*

	conv_loss��>�<^        )��P	�Z�,���A�*

	conv_loss���>4k�{        )��P	���,���A�*

	conv_loss�G�>���        )��P	�"�,���A�*

	conv_loss7�>�@�        )��P	X�,���A�*

	conv_loss�v�>��6G        )��P	��,���A�*

	conv_loss�e�>H�j�        )��P	 ��,���A�*

	conv_loss,�>��ǁ        )��P	���,���A�*

	conv_lossx)�>��>�        )��P	�+�,���A�*

	conv_loss�Ɯ><t
�        )��P	z_�,���A�*

	conv_loss�f�>���        )��P	��,���A�*

	conv_loss�K�>��~�        )��P	]��,���A�*

	conv_loss�՘>��;�        )��P	��,���A�*

	conv_loss�Q�>�s�5        )��P	�8�,���A�*

	conv_lossޣ�>P��        )��P	vq�,���A�*

	conv_loss1t�>Mߐ�        )��P	���,���A�*

	conv_loss=>Yp*�        )��P	���,���A�*

	conv_lossnF�>���        )��P	�,���A�*

	conv_loss*��>�P.        )��P	�B�,���A�*

	conv_lossuӛ>���        )��P	^u�,���A�*

	conv_loss�Y�>��         )��P	U��,���A�*

	conv_lossd�>|�ޮ        )��P	���,���A�*

	conv_loss�̛>���~        )��P	V�,���A�*

	conv_loss�=�>�EĜ        )��P	�@�,���A�*

	conv_loss(ޙ>��&        )��P	�s�,���A�*

	conv_lossN?�>����        )��P	��,���A�*

	conv_loss�P�>�E        )��P	���,���A�*

	conv_loss^��>�u��        )��P	��,���A�*

	conv_lossj��>"�ܿ        )��P	B�,���A�*

	conv_lossp��>�V].        )��P	�v�,���A�*

	conv_loss�G�>�Ʈ�        )��P	2��,���A�*

	conv_loss�j�>ك�?        )��P	���,���A�*

	conv_loss�l�>�r�e        )��P	��,���A�*

	conv_loss�q�>��'�        )��P	�A�,���A�*

	conv_loss�K�>���1        )��P	�u�,���A�*

	conv_loss>	�>k�2E        )��P	���,���A�*

	conv_lossE��>G���        )��P	���,���A�*

	conv_loss���>8���        )��P	o�,���A�*

	conv_lossx|�>}�F�        )��P	�C�,���A�*

	conv_loss:�>����        )��P	Sv�,���A�*

	conv_loss㼙>�Պ>        )��P	��,���A�*

	conv_loss��>�J��        )��P	���,���A�*

	conv_loss@H�>!=c�        )��P	��,���A�*

	conv_loss�9�>�s�        )��P	�A�,���A�*

	conv_losssY�>b�/        )��P	�u�,���A�*

	conv_lossȅ�>���        )��P	���,���A�*

	conv_loss鏙>��^�        )��P	a��,���A�*

	conv_loss- �>~[2        )��P	�,���A�*

	conv_loss}Ԙ>
G�        )��P	E�,���A�*

	conv_loss���>���?        )��P	Oz�,���A�*

	conv_loss��>��        )��P	4��,���A�*

	conv_loss�R�>��	        )��P	u��,���A�*

	conv_loss�ޙ>9��        )��P	�)�,���A�*

	conv_loss�A�>6L�        )��P	N\�,���A�*

	conv_lossV��>����        )��P	���,���A�*

	conv_loss1}�>��        )��P	J��,���A�*

	conv_lossߖ>� .        )��P	]�,���A�*

	conv_loss�f�>����        )��P	6F�,���A�*

	conv_loss���>��U�        )��P	�y�,���A�*

	conv_lossth�>��|�        )��P	���,���A�*

	conv_loss�U�>e���        )��P	t��,���A�*

	conv_loss �>�g��        )��P	1/�,���A�*

	conv_loss���>�ː�        )��P	�c�,���A�*

	conv_loss�ט>JH�Q        )��P	ϡ�,���A�*

	conv_loss��>���        )��P	��,���A�*

	conv_loss�Z�>�T�p        )��P	��,���A�*

	conv_loss>��>M��        )��P	�@�,���A�*

	conv_loss��>Q�M        )��P	�r�,���A�*

	conv_loss|W�>�lƔ        )��P	��,���A�*

	conv_loss�g�>����        )��P	��,���A�*

	conv_loss(�>���(        )��P	��,���A�*

	conv_loss�E�>*�        )��P	�A�,���A�*

	conv_loss��>��A�        )��P	�t�,���A�*

	conv_loss.��>�Z�        )��P	,��,���A�*

	conv_loss�͘>�o�<        )��P	��,���A�*

	conv_loss˓>z���        )��P	�,���A�*

	conv_loss�H�>�|        )��P	B�,���A�*

	conv_lossf��>���9        )��P	�t�,���A�*

	conv_lossÔ>��af        )��P	g��,���A�*

	conv_lossӪ�>{N        )��P	��,���A�*

	conv_loss�Ę>�qt�        )��P	.�,���A�*

	conv_lossy<�>�#        )��P	�@�,���A�*

	conv_losso�>��U�        )��P	�r�,���A�*

	conv_loss�H�>
�e�        )��P	ɥ�,���A�*

	conv_loss�9�>��+        )��P	���,���A�*

	conv_loss�ϗ>�8�        )��P	b�,���A�*

	conv_loss=�>t%9�        )��P	�>�,���A�*

	conv_loss,2�>@��        )��P	q�,���A�*

	conv_loss���>\��        )��P	j��,���A�*

	conv_loss��>�Ӎ        )��P	���,���A�*

	conv_loss��>�}        )��P	�
�,���A�*

	conv_loss1\�>����        )��P	@�,���A�*

	conv_losseI�>��\�        )��P	�t�,���A�*

	conv_loss�^�>��7�        )��P	���,���A�*

	conv_lossk`�>�>c        )��P	���,���A�*

	conv_loss~Ù>�[�        )��P	7�,���A�*

	conv_loss���>)2V�        )��P	�B�,���A�*

	conv_loss	�>����        )��P	Bv�,���A�*

	conv_loss�1�>9�        )��P	��,���A�*

	conv_lossF&�>W�        )��P	���,���A�*

	conv_lossJ�>+m�f        )��P	)�,���A�*

	conv_loss�y�>	���        )��P	���,���A�*

	conv_loss'z�>�� D        )��P	�/�,���A�*

	conv_lossX�>��
A        )��P	�`�,���A�*

	conv_loss>��>����        )��P	;��,���A�*

	conv_loss�)�>;��"        )��P	d��,���A�*

	conv_losse2�>N�1�        )��P	���,���A�*

	conv_loss,-�>6�-        )��P	~�,���A�*

	conv_lossb1�>�7��        )��P	�M�,���A�*

	conv_loss���>�q0�        )��P	'��,���A�*

	conv_loss�>Fz3        )��P	���,���A�*

	conv_lossq��>��        )��P	,��,���A�*

	conv_loss@��>Z��        )��P	��,���A�*

	conv_loss�H�>����        )��P	SC�,���A�*

	conv_lossQ#�>��˔        )��P	{�,���A�*

	conv_loss�g�>�9"J        )��P	#��,���A�*

	conv_loss���>Ô��        )��P	v��,���A�*

	conv_loss�̕>���        )��P	��,���A�*

	conv_lossdÓ>�c��        )��P	eF�,���A�*

	conv_loss/�>oaz6        )��P	�u�,���A�*

	conv_loss鱔>鹰Z        )��P	���,���A�*

	conv_lossэ�>��P�        )��P	2��,���A�*

	conv_loss�9�>��        )��P	�,���A�*

	conv_loss�_�>��1�        )��P	�5�,���A�*

	conv_loss�*�>���4        )��P	qg�,���A�*

	conv_loss�I�>��>F        )��P	,��,���A�*

	conv_loss���>��|        )��P	���,���A�*

	conv_lossr�>��?        )��P	���,���A�*

	conv_loss�ݔ>H�I�        )��P	c �,���A�*

	conv_loss���>��a�        )��P	�O�,���A�*

	conv_loss�_�>tZs        )��P	}�,���A�*

	conv_loss4��>Ҙ�        )��P	���,���A�*

	conv_loss���>���        )��P	U��,���A�*

	conv_loss��>%Fk        )��P	�	�,���A�*

	conv_loss��>�C�        )��P	 8�,���A�*

	conv_loss��>��\�        )��P	"i�,���A�*

	conv_loss�A�>�l
"        )��P	2��,���A�*

	conv_lossb_�>J5S        )��P	���,���A�*

	conv_loss⍔>D#�        )��P	���,���A�*

	conv_loss���>v��        )��P	Q'�,���A�*

	conv_loss��>�        )��P	qY�,���A�*

	conv_lossqs�>k��(        )��P	���,���A�*

	conv_lossma�>� �        )��P	���,���A�*

	conv_lossV�>R���        )��P	��,���A�*

	conv_loss�0�>Oz��        )��P	�,�,���A�*

	conv_lossq�>�6�S        )��P	�\�,���A�*

	conv_losseg�>��*        )��P	���,���A�*

	conv_loss��>��~.        )��P	?��,���A�*

	conv_lossRɘ>'4�        )��P	���,���A�*

	conv_loss�Ӑ>�E!        )��P	��,���A�*

	conv_loss��>6�        )��P	J�,���A�*

	conv_loss���>^=ߑ        )��P	z�,���A�*

	conv_loss*M�>�1*        )��P	0��,���A�*

	conv_lossu0�>z���        )��P	���,���A�*

	conv_loss�ٍ>�OB        )��P	�,���A�*

	conv_losscV�>1��        )��P	�N�,���A�*

	conv_lossx��>�V�        )��P	5��,���A�*

	conv_loss���>4��        )��P	G��,���A�*

	conv_loss*��>���C        )��P	��,���A�*

	conv_loss�+�>��*        )��P	��,���A�*

	conv_loss�Ñ>���        )��P	G�,���A�*

	conv_lossm��>�;fY        )��P	Yy�,���A�*

	conv_lossU}�>ІS        )��P	ݨ�,���A�*

	conv_lossF�>^�}        )��P	6��,���A�*

	conv_loss胕>��0�        )��P	��,���A�*

	conv_loss�ڕ>��z�        )��P	�B�,���A�*

	conv_loss7E�>G��        )��P	�q�,���A�*

	conv_loss<�>B�-        )��P	���,���A�*

	conv_lossَ�>��"�        )��P	���,���A�*

	conv_loss�!�>	#�        )��P	1�,���A�*

	conv_loss�,�>¤��        )��P	�<�,���A�*

	conv_loss���>K�b�        )��P	+m�,���A�*

	conv_lossfϐ>�9�h        )��P	���,���A�*

	conv_losse�>A淚        )��P	���,���A�*

	conv_loss	>�>�#�        )��P	i��,���A�*

	conv_loss9��>r��        )��P	,�,���A�*

	conv_loss^�>�'ޖ        )��P	�[�,���A�*

	conv_loss���>�A|�        )��P	l��,���A�*

	conv_loss�x�>�+        )��P	3��,���A�*

	conv_loss��>W���        )��P	b��,���A�*

	conv_lossC_�>0T%�        )��P	m�,���A�*

	conv_loss��>�a�        )��P	jP�,���A�*

	conv_loss��>��U;        )��P	a~�,���A�*

	conv_loss.Ғ>�/jX        )��P	*��,���A�*

	conv_loss��>��7        )��P	��,���A�*

	conv_lossu�>��        )��P	)�,���A�*

	conv_loss^��>;4.,        )��P	�=�,���A�*

	conv_loss��>��X�        )��P	l�,���A�*

	conv_loss�Ő>�5Q        )��P	*��,���A�*

	conv_lossk@�>�&�        )��P	���,���A�*

	conv_lossJ2�>!%��        )��P	���,���A�*

	conv_loss���>��d        )��P	y%�,���A�*

	conv_lossX��>���        )��P	�T�,���A�*

	conv_lossF�>hd         )��P	؄�,���A�*

	conv_loss]g�>v�        )��P	���,���A�*

	conv_loss�>�        )��P	���,���A�*

	conv_loss��>t�D2        )��P	��,���A�*

	conv_loss���>]�5G        )��P	�I�,���A�*

	conv_loss?��>�x�        )��P	{�,���A�*

	conv_loss^I�>u��<        )��P	ݬ�,���A�*

	conv_lossR2�>1�?        )��P	���,���A�*

	conv_lossH�>���K        )��P	g�,���A�*

	conv_loss��>	���        )��P	��,���A�*

	conv_loss�ɔ>Ft�        )��P	���,���A�*

	conv_lossIa�>/�b�        )��P	y�,���A�*

	conv_loss1��>[/�        )��P	�2�,���A�*

	conv_loss�	�>�� �        )��P	�k�,���A�*

	conv_loss�v�>�SX        )��P	b��,���A�*

	conv_loss���>+>a        )��P	4��,���A�*

	conv_loss���>���D        )��P	���,���A�*

	conv_loss���>��q        )��P	�8�,���A�*

	conv_loss軗>$         )��P	/o�,���A�*

	conv_loss,U�>-aK�        )��P	��,���A�*

	conv_loss޶�>�i�        )��P	���,���A�*

	conv_lossvɓ>�s"�        )��P	��,���A�*

	conv_loss�-�>��        )��P	`I�,���A�*

	conv_loss���>�!�        )��P	�{�,���A�*

	conv_loss$)�>��        )��P	c��,���A�*

	conv_loss[��>!�^        )��P	���,���A�*

	conv_loss�D�>^��        )��P	��,���A�*

	conv_loss�>g1��        )��P	_;�,���A�*

	conv_loss���>O�Ξ        )��P	`j�,���A�*

	conv_lossB��>�!��        )��P	���,���A�*

	conv_loss2��>����        )��P	���,���A�*

	conv_lossE�>Ù�B        )��P	��,���A�*

	conv_loss�ڔ>�tc1        )��P	�+�,���A�*

	conv_loss� �>k��        )��P	U]�,���A�*

	conv_lossؓ>��3.        )��P	��,���A�*

	conv_loss��>�S�[        )��P	���,���A�*

	conv_lossY܏>�Yt�        )��P	@��,���A�*

	conv_loss��>��L        )��P	& -���A�*

	conv_lossa��>�LC        )��P	�V -���A�*

	conv_loss��>�h$�        )��P	-� -���A�*

	conv_loss>K�>�q,        )��P	� -���A�*

	conv_loss�h�>8�1        )��P	0� -���A�*

	conv_loss�>a&�        )��P	e-���A�*

	conv_loss���>����        )��P	�F-���A�*

	conv_loss�h�>��bU        )��P	z-���A�*

	conv_loss���>P�,�        )��P	��-���A�*

	conv_lossA͎>��0         )��P	7�-���A�*

	conv_loss�p�>���        )��P	�-���A�*

	conv_loss n�>�4��        )��P	�7-���A�*

	conv_loss�#�>(�        )��P	s-���A�*

	conv_loss��>f΍0        )��P	N�-���A�*

	conv_loss�_�>��z        )��P	=�-���A�*

	conv_loss���>Ջ�        )��P	Q-���A�*

	conv_lossT��>��V        )��P	�2-���A�*

	conv_loss�Ε>�*��        )��P	�`-���A�*

	conv_loss�ߏ>,��=        )��P	W�-���A�*

	conv_lossL��>��        )��P	��-���A�*

	conv_loss���>'KFx        )��P	�-���A�*

	conv_lossH�>�"�D        )��P	�!-���A�*

	conv_loss��>���G        )��P	c-���A�*

	conv_loss�j�>����        )��P	O�-���A�*

	conv_loss"8�>7ܥ�        )��P	�-���A�*

	conv_loss��>�I�        )��P	/�-���A�*

	conv_lossc��>��h        )��P	�/-���A�*

	conv_loss�J�>�d�        )��P	^-���A�*

	conv_loss���>渮�        )��P	R�-���A�*

	conv_loss�M�>��        )��P	�-���A�*

	conv_loss���>�tg        )��P	�-���A�*

	conv_losst��>(�h        )��P	4-���A�*

	conv_lossZ�>�v�%        )��P	e-���A�*

	conv_lossL0�>�_�        )��P	��-���A�*

	conv_loss6o�>2v�        )��P	f�-���A�*

	conv_lossN��>rk�        )��P	-���A�*

	conv_loss�ȏ>l*y�        )��P	�?-���A�*

	conv_loss��>�A�        )��P	!o-���A�*

	conv_loss"،>[\[        )��P	0�-���A�*

	conv_lossu��>�HTP        )��P	6�-���A�*

	conv_loss���>���S        )��P	Z�-���A�*

	conv_loss�;�>\�`�        )��P	�/-���A�*

	conv_loss���>ѐ�        )��P	'^-���A�*

	conv_loss�}�>�o        )��P	`�-���A�*

	conv_loss�R�>��T        )��P	x�-���A�*

	conv_loss�ӊ>��em        )��P	~�-���A�*

	conv_loss��>��        )��P	C 	-���A�*

	conv_lossv��>/���        )��P	\Q	-���A�*

	conv_loss�6�>�        )��P	��	-���A�*

	conv_loss�z�>Kq}�        )��P	̲	-���A�*

	conv_loss�3�>Ș	        )��P	��	-���A�*

	conv_loss��>	ǐ        )��P	c
-���A�*

	conv_loss)#�>MmI�        )��P	-C
-���A�*

	conv_lossu;�>-O_�        )��P	�u
-���A�*

	conv_lossuٍ>Q�	        )��P	7�
-���A�*

	conv_loss�ʍ>`}m        )��P	)�
-���A�*

	conv_losss��>�vB�        )��P	P-���A�*

	conv_loss�/�>gEJ�        )��P	�7-���A�*

	conv_loss]�>�T        )��P	`k-���A�*

	conv_loss2�>����        )��P	��-���A�*

	conv_loss���>���J        )��P	=�-���A�*

	conv_lossb��>�͛        )��P	z�-���A�*

	conv_loss�V�>�F�        )��P	&--���A�*

	conv_loss�#�>�M        )��P	$]-���A�*

	conv_lossCߊ>0e        )��P	��-���A�*

	conv_loss8�>N�U�        )��P	ҽ-���A�*

	conv_lossd5�>�Ĵv        )��P	��-���A�*

	conv_loss�!�>䰢b        )��P	-���A�*

	conv_lossÓ>j���        )��P	�L-���A�*

	conv_loss�:�>�[�o        )��P	�-���A�*

	conv_lossڲ�>w8_�        )��P	�-���A�*

	conv_losso�>eʌ�        )��P	��-���A�*

	conv_lossSݔ>-��        )��P	�!-���A�*

	conv_lossU��>���        )��P	9Q-���A�*

	conv_lossm��>	���        )��P	U�-���A�*

	conv_loss�0�>��        )��P	��-���A�*

	conv_loss�>�>c��        )��P	��-���A�*

	conv_loss1��>���X        )��P	-���A�*

	conv_loss)�>syĠ        )��P	�C-���A�*

	conv_lossH�>�C�        )��P	t-���A�*

	conv_loss��>!        )��P	��-���A�*

	conv_loss��>�R"�        )��P	��-���A�*

	conv_losse��>��p        )��P	;-���A�*

	conv_loss���>��<~        )��P	kH-���A�*

	conv_lossݷ�>3dg�        )��P	mz-���A�*

	conv_loss�m�>K���        )��P	5�-���A�*

	conv_lossA��>��ae        )��P	@�-���A�*

	conv_loss���>5�5        )��P	�)-���A�*

	conv_lossFg�>L��        )��P	X[-���A�*

	conv_loss?�>����        )��P	;�-���A�*

	conv_lossp�>�T�        )��P	�-���A�*

	conv_loss���>��M        )��P	P�-���A�*

	conv_loss=j�>��]�        )��P	�-���A�*

	conv_loss`f�>��_[        )��P	M-���A�*

	conv_loss[��>�E��        )��P	�~-���A�*

	conv_loss��>P�+        )��P	�-���A�*

	conv_lossc��>y�n�        )��P	��-���A�*

	conv_lossJ��>�1�s        )��P	�-���A�*

	conv_loss�>7�h�        )��P	�A-���A�*

	conv_loss+��>LU}        )��P	r-���A�*

	conv_loss�Q�>o���        )��P	`�-���A�*

	conv_loss�j�>2�:�        )��P	��-���A�*

	conv_loss��>�Z        )��P	�-���A�*

	conv_losst��>n4(1        )��P	�@-���A�*

	conv_lossQ��>JE�        )��P	p-���A�*

	conv_loss��>��        )��P	��-���A�*

	conv_loss/z�>���        )��P	�-���A�*

	conv_loss�/�>�        )��P	-���A�*

	conv_loss���>[�        )��P	N4-���A�*

	conv_loss%��>	w7        )��P	�c-���A�*

	conv_loss
��>q�        )��P	��-���A�*

	conv_loss�p�>���        )��P	J�-���A�*

	conv_loss(U�>_�x�        )��P	��-���A�*

	conv_lossI��>ʒ        )��P	�3-���A�*

	conv_loss�`�>��.        )��P	�c-���A�*

	conv_loss�@�>���        )��P	)�-���A�*

	conv_loss.d�>^n �        )��P	�-���A�*

	conv_loss/�>v4�        )��P	0�-���A�*

	conv_loss�Ό>�ʒa        )��P	 --���A�*

	conv_loss	g�>=+O�        )��P	�^-���A�*

	conv_lossj�>����        )��P	r�-���A�*

	conv_loss�7�>� ��        )��P	N�-���A�*

	conv_loss�ڎ>&,9�        )��P	-���A�*

	conv_loss�>>���        )��P	�7-���A�*

	conv_lossG�>��k�        )��P	6g-���A�*

	conv_loss%�>���        )��P	��-���A�*

	conv_lossr��>0�@�        )��P	"�-���A�*

	conv_lossa�>g��8        )��P	��-���A�*

	conv_loss¹�>h%j        )��P	�2-���A�*

	conv_loss�~�>��t        )��P	|b-���A�*

	conv_loss(m�>w+�f        )��P	��-���A�*

	conv_lossI�>�R��        )��P	��-���A�*

	conv_lossX=�>�##�        )��P	�-���A�*

	conv_loss�<�>�[[{        )��P	x<-���A�*

	conv_loss"�>&��h        )��P	m-���A�*

	conv_lossLN�>z�%�        )��P	3�-���A�*

	conv_loss�{�>c���        )��P	k�-���A�*

	conv_loss��>� ��        )��P	]-���A�*

	conv_loss�p�>�j'�        )��P	T>-���A�*

	conv_loss��>`��        )��P	 p-���A�*

	conv_loss빈>d��        )��P	��-���A�*

	conv_loss�Ƈ>��B        )��P	��-���A�*

	conv_loss�\�>���h        )��P	�-���A�*

	conv_loss���>^���        )��P	m3-���A�*

	conv_loss�f�>��%�        )��P	Pb-���A�*

	conv_loss!�>�W�        )��P	�-���A�*

	conv_loss�d�>N��,        )��P	�-���A�*

	conv_loss@�>I��2        )��P	L�-���A�*

	conv_loss+��>/H�O        )��P	�!-���A�*

	conv_lossĆ�>Eѳz        )��P	�R-���A�*

	conv_loss&��>�i��        )��P	��-���A�*

	conv_loss��>U-�?        )��P	ݲ-���A�*

	conv_loss◇>M��        )��P	^�-���A�*

	conv_lossԤ�>���        )��P	�-���A�*

	conv_lossc�>��1        )��P	C-���A�*

	conv_loss�
�>�sbb        )��P	�r-���A�*

	conv_lossZ��>��        )��P	;�-���A�*

	conv_loss}�>�-��        )��P	x�-���A�*

	conv_loss�j�>�y�        )��P	�-���A�*

	conv_loss�]�>i�G�        )��P	�0-���A�*

	conv_loss���>����        )��P	b-���A�*

	conv_lossJ5�>�F|�        )��P	�-���A�*

	conv_loss���>�[�i        )��P	��-���A�*

	conv_loss�$�>���E        )��P	b�-���A�*

	conv_loss���>#���        )��P	�  -���A�*

	conv_loss)�>���        )��P	|P -���A�*

	conv_lossh��> ���        )��P	�� -���A�*

	conv_loss#%�>dk��        )��P	k� -���A�*

	conv_loss�b�>�\��        )��P	�� -���A�*

	conv_loss<F�>���(        )��P	�!-���A�*

	conv_lossW�>���[        )��P	F!-���A�*

	conv_loss�>dY��        )��P	Dy!-���A�*

	conv_loss�>t
��        )��P	�#-���A�*

	conv_loss}T�>��@�        )��P	s7#-���A�*

	conv_lossSh�>�t�        )��P	,h#-���A�*

	conv_lossc��>���$        )��P	�#-���A�*

	conv_loss�a�>���/        )��P	5�#-���A�*

	conv_loss+l�>(k+        )��P	��#-���A�*

	conv_loss6͈>A�Р        )��P	D0$-���A�*

	conv_loss�F�>���        )��P	�k$-���A�*

	conv_lossuF�>%�@        )��P	��$-���A�*

	conv_lossh�>R�;        )��P	k�$-���A�*

	conv_lossX-�>D1��        )��P	 %-���A�*

	conv_loss}x�>���        )��P	�C%-���A�*

	conv_lossq�>'$�        )��P	�y%-���A�*

	conv_loss���>�)��        )��P	��%-���A�*

	conv_loss	U�>���         )��P	o�%-���A�*

	conv_loss��>;�͂        )��P	�
&-���A�*

	conv_loss�>P�`j        )��P	�9&-���A�*

	conv_loss�[�>(�N�        )��P	 h&-���A�*

	conv_lossХ�>��I        )��P	��&-���A�*

	conv_loss�3�>=�b        )��P	M�&-���A�*

	conv_loss���>A��_        )��P	��&-���A�*

	conv_losso��>&R/         )��P	�)'-���A�*

	conv_loss,؉>���        )��P	`Y'-���A�*

	conv_loss��>Ӽ�         )��P	È'-���A�*

	conv_loss�l�>�N�        )��P	�'-���A�*

	conv_loss)�>�A�        )��P	��'-���A�*

	conv_loss���>��F�        )��P	9(-���A�*

	conv_lossʂ>i�u        )��P	�K(-���A�*

	conv_loss!�>�1q�        )��P	�~(-���A�*

	conv_lossvh�>C��t        )��P	G�(-���A�*

	conv_lossS�>����        )��P	K�(-���A�*

	conv_loss��>��z�        )��P	0)-���A�*

	conv_loss�ȉ>����        )��P	�>)-���A�*

	conv_loss̃>Y%#        )��P	n)-���A�*

	conv_lossOm�>���;        )��P	��)-���A�*

	conv_lossA��>g�I        )��P	#�)-���A�*

	conv_loss�ʆ>���        )��P	.�)-���A�*

	conv_lossM��>%�@�        )��P	x0*-���A�*

	conv_loss�˅>Q��:        )��P	%`*-���A�*

	conv_loss��>���        )��P	��*-���A�*

	conv_loss�ҋ>DX        )��P	��*-���A�*

	conv_loss���>��%�        )��P	��*-���A�*

	conv_loss��>���        )��P	K$+-���A�*

	conv_loss�9�>���        )��P	�S+-���A�*

	conv_loss$�>D9�Z        )��P	��+-���A�*

	conv_loss�=�>g��p        )��P	��+-���A�*

	conv_loss���>�N�        )��P	z�+-���A�*

	conv_loss�Ћ>��G        )��P	�,-���A�*

	conv_lossX=�>R�8�        )��P	�G,-���A�*

	conv_loss�(�>�B=y        )��P	z,-���A�*

	conv_loss}V�>W��        )��P	ƹ,-���A�*

	conv_lossÄ>=:@c        )��P	c�,-���A�*

	conv_loss��>C�8        )��P	.--���A�*

	conv_loss��>M�̬        )��P	KK--���A�*

	conv_lossj`�>rN|_        )��P	�~--���A�*

	conv_loss��>7��        )��P	k�--���A�*

	conv_loss�D�>� Z        )��P	[�--���A�*

	conv_loss$f�>�Nn        )��P	v.-���A�*

	conv_lossJ��>�UQ        )��P	qS.-���A�*

	conv_lossy�> L�        )��P	��.-���A�*

	conv_lossJi�>P�&?        )��P	��.-���A�*

	conv_loss"��>�ʘ�        )��P	�.-���A�*

	conv_loss��>��Y�        )��P	q(/-���A�*

	conv_loss	0�>R�A        )��P	Z]/-���A�*

	conv_lossu��>U�pz        )��P	�/-���A�*

	conv_loss���>����        )��P	��/-���A�*

	conv_loss�/�>�ϵ�        )��P	�0-���A�*

	conv_loss�|�>�@�        )��P	10-���A�*

	conv_loss�n�>
s�         )��P	:a0-���A�*

	conv_lossW0�>�/        )��P	��0-���A�*

	conv_loss�w�>�hǔ        )��P	��0-���A�*

	conv_loss�>y�G        )��P	��0-���A�*

	conv_loss���>�        )��P	Q%1-���A�*

	conv_lossz�>�P        )��P	�U1-���A�*

	conv_loss��>��        )��P	��1-���A�*

	conv_lossoԃ>��`        )��P	��1-���A�*

	conv_lossW�>���@        )��P	;�1-���A�*

	conv_loss�Y�>s���        )��P	2-���A�*

	conv_loss��>��m        )��P	*F2-���A�*

	conv_loss�4�>,Ǣ�        )��P	�u2-���A�*

	conv_loss^V�>��        )��P	��2-���A�*

	conv_loss���>܄        )��P	��2-���A�*

	conv_loss=��>��        )��P	�3-���A�*

	conv_loss�K�>%��R        )��P	63-���A�*

	conv_loss*��>X��`        )��P	�e3-���A�*

	conv_lossN�>��r        )��P	ٕ3-���A�*

	conv_lossp��>m�        )��P	�3-���A�*

	conv_loss�>�{-        )��P	��3-���A�*

	conv_loss2��>�{��        )��P	(4-���A�*

	conv_loss��~>M��'        )��P	�Y4-���A�*

	conv_losst�>����        )��P	߈4-���A�*

	conv_loss8o�>9NY        )��P	=�4-���A�*

	conv_lossˌ�>~z/W        )��P	��4-���A�*

	conv_loss���>�{W        )��P	]5-���A�*

	conv_lossJp�>X<�        )��P	�J5-���A�*

	conv_lossD��>:^�        )��P	�z5-���A�*

	conv_loss�T�>��a�        )��P	O�5-���A�*

	conv_loss\��>I��        )��P	��5-���A�*

	conv_loss�Ҋ>���r        )��P	6-���A�*

	conv_lossS,�>I��:        )��P	�=6-���A�*

	conv_lossX��>+�2�        )��P	��6-���A�*

	conv_lossX�>n��        )��P	�6-���A�*

	conv_loss�^�>�,�        )��P	��6-���A�*

	conv_loss��>�f�        )��P	!7-���A�*

	conv_loss��>��;        )��P	�L7-���A�*

	conv_loss���>T��U        )��P	u}7-���A�*

	conv_loss���>.E�D        )��P	�7-���A�*

	conv_loss���>��6�        )��P	K�7-���A�*

	conv_loss�x�>�uo\        )��P	�8-���A�*

	conv_loss� �>��ߥ        )��P	tS8-���A�*

	conv_loss��>�dM*        )��P	ք8-���A�*

	conv_loss/l�>{        )��P	��8-���A�*

	conv_loss�̆>�P�?        )��P	��8-���A�*

	conv_loss��>Ja�        )��P	")9-���A�*

	conv_loss7k�>ӛ1�        )��P	�[9-���A�*

	conv_loss(��>���x        )��P	
�9-���A�*

	conv_lossz�>&EU�        )��P	[�9-���A�*

	conv_loss��>�E�        )��P	��9-���A�*

	conv_loss2Ѐ>�X��        )��P	-%:-���A�*

	conv_losse�>KN        )��P	=W:-���A�*

	conv_loss߿�>�ܚ        )��P	8�:-���A�*

	conv_loss���>�#�        )��P	�:-���A�*

	conv_loss��>Nm3        )��P	R�:-���A�*

	conv_loss�̅>�)�        )��P	y&;-���A�*

	conv_lossR�>���        )��P	WX;-���A�*

	conv_loss���>��=        )��P	_�;-���A�*

	conv_loss���>���?        )��P	!�;-���A�*

	conv_lossD��>�xm�        )��P	��;-���A�*

	conv_lossU�>J7        )��P	�*<-���A�*

	conv_loss�{>��        )��P	�\<-���A�*

	conv_loss�>�H�V        )��P	��<-���A�*

	conv_lossJ}�>�a"�        )��P	=�<-���A�*

	conv_loss�>x�~�        )��P	[�<-���A�*

	conv_loss��{>���        )��P	) =-���A�*

	conv_lossࠄ>/[a�        )��P	�P=-���A�*

	conv_loss�݅>��U�        )��P	�=-���A�*

	conv_lossT�>w��        )��P	T�=-���A�*

	conv_loss�#�>r���        )��P	
�=-���A�*

	conv_loss؀�>Uٺ�        )��P	�>-���A�*

	conv_loss��x>@�]�        )��P	bH>-���A�*

	conv_loss飂>�T֑        )��P	�{>-���A�*

	conv_loss5L�>�&�3        )��P	��>-���A�*

	conv_loss��y>����        )��P	3�>-���A�*

	conv_loss�J�>:�8        )��P	�?-���A�*

	conv_loss@��>��Rg        )��P	�;?-���A�*

	conv_loss�\z>��H        )��P	�j?-���A�*

	conv_lossr.�>@�~        )��P	��?-���A�*

	conv_loss�j�>��(�        )��P	��?-���A�*

	conv_loss�>��{        )��P	��?-���A�*

	conv_loss���>q��9        )��P	�)@-���A�*

	conv_loss�7�>��1�        )��P	uj@-���A�*

	conv_loss���>���        )��P	W�@-���A�*

	conv_loss�+�>��        )��P	��@-���A�*

	conv_loss�r>_i k        )��P	�A-���A�*

	conv_lossJ`�>R�/�        )��P	SCA-���A�*

	conv_loss\�>3�~        )��P	�uA-���A�*

	conv_loss�ۂ>�p        )��P	��A-���A�*

	conv_loss�F�>qq�q        )��P	`�A-���A�*

	conv_loss݃�>���4        )��P	-B-���A�*

	conv_loss�Ă>B]�        )��P	CIB-���A�*

	conv_loss$��>�v��        )��P	~B-���A�*

	conv_lossh	�>���        )��P	V�B-���A�*

	conv_lossXf�>pzGP        )��P	��B-���A�*

	conv_loss�T�>��         )��P	�C-���A�*

	conv_loss���>f5(1        )��P	8\C-���A�*

	conv_lossƈ>?F�        )��P	ɓC-���A�*

	conv_loss��>�K�!        )��P	��C-���A�*

	conv_loss+Ă>g�21        )��P	�C-���A�*

	conv_lossi��> =��        )��P	7.D-���A�*

	conv_lossh�}>�$�        )��P	�eD-���A�*

	conv_loss߲�>��        )��P	�D-���A�*

	conv_loss�t~>����        )��P	1�D-���A�*

	conv_loss��>�B         )��P	��D-���A�*

	conv_loss<�>�L��        )��P	�3E-���A�*

	conv_lossb"�>|�y        )��P	8gE-���A�*

	conv_loss��>��[(        )��P	x�E-���A�*

	conv_loss�w}>���N        )��P	��E-���A�*

	conv_loss��~>-WV�        )��P	�F-���A�*

	conv_lossn�{>����        )��P	�3F-���A�*

	conv_loss#s�>��+        )��P	&gF-���A�*

	conv_loss�~>�V@�        )��P	��F-���A�*

	conv_loss�ɀ>��i        )��P	z�F-���A�*

	conv_loss�>㷝�        )��P	�G-���A�*

	conv_loss:u>�<fn        )��P	.6G-���A�*

	conv_loss�~>�e        )��P	�jG-���A�*

	conv_loss�>S._�        )��P	�G-���A�*

	conv_loss�Zv>2��        )��P	n�G-���A�*

	conv_loss+��>��!�        )��P	�H-���A�*

	conv_losssه>��,�        )��P	�9H-���A�*

	conv_loss�>�1D�        )��P	�kH-���A�*

	conv_loss.��>�=i�        )��P	�H-���A�*

	conv_lossM�>�{�}        )��P	��H-���A�*

	conv_loss�g|>��        )��P	II-���A�*

	conv_loss�Y�>���        )��P	�7I-���A�*

	conv_loss�̀>k��}        )��P	&kI-���A�*

	conv_lossr�>���        )��P	ӞI-���A�*

	conv_loss�8�>6���        )��P	��I-���A�*

	conv_lossr�x>�`x        )��P	�J-���A�*

	conv_loss-�}>�
�)        )��P	�6J-���A�*

	conv_losssT�>�
y�