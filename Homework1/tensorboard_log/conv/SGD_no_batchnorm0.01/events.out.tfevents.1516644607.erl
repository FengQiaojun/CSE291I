       �K"	  �?���Abrain.Event:2ɜ���      D(�	�	�?���A"��
~
PlaceholderPlaceholder*$
shape:���������*
dtype0*/
_output_shapes
:���������
p
Placeholder_1Placeholder*
shape:���������
*
dtype0*'
_output_shapes
:���������

R
Placeholder_2Placeholder*
shape:*
dtype0
*
_output_shapes
:
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
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0* 
_class
loc:@conv2d/kernel
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
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0* 
_class
loc:@conv2d/kernel
�
conv2d/kernel
VariableV2*
shared_name * 
_class
loc:@conv2d/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
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
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
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
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:
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
Relu_1Reluconv2d_2/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_2/kernel*%
valueB"            
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

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*&
_output_shapes
:
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
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
conv2d_2/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_2/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(
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
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 *
dtype0
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
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_3/kernel
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
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_4/Conv2DConv2DRelu_2conv2d_3/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
Y
Relu_3Reluconv2d_4/Conv2D*/
_output_shapes
:���������*
T0
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
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
T0
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
conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
g
conv2d_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_5/Conv2DConv2DRelu_3conv2d_4/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
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
.conv2d_5/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *d�=*
dtype0*
_output_shapes
: 
�
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_5/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
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
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:*
T0
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
0conv2d_6/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@conv2d_6/kernel*%
valueB"            *
dtype0
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
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:*
T0
�
conv2d_6/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_6/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
conv2d_6/kernel/AssignAssignconv2d_6/kernel*conv2d_6/kernel/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel
�
conv2d_6/kernel/readIdentityconv2d_6/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
g
conv2d_7/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
_class
loc:@dense/kernel*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_output_shapes
:	�
d*

seed *
T0*
_class
loc:@dense/kernel*
seed2 *
dtype0
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
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
_output_shapes
:	�
d*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	�
d*
dtype0
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
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@dense_1/kernel*
valueB"d   
   
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
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes

:d
*

seed *
T0
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
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

�
dense_1/kernel
VariableV2*!
_class
loc:@dense_1/kernel*
	container *
shape
:d
*
dtype0*
_output_shapes

:d
*
shared_name 
�
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:d
*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
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
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@dense_1/bias
�
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
q
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:
*
T0*
_class
loc:@dense_1/bias
�
dense_2/MatMulMatMulRelu_7dense_1/kernel/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
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
logistic_loss/GreaterEqualGreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:���������
*
T0
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
	conv_lossScalarSummaryconv_loss/tagsMean*
T0*
_output_shapes
: 
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
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
f
gradients/Mean_grad/ShapeShapelogistic_loss*
_output_shapes
:*
T0*
out_type0
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
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
�
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( *

Tidx0
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
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:���������
*
T0
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
out_type0*
_output_shapes
:*
T0
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
out_type0*
_output_shapes
:*
T0
�
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

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
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*
_output_shapes
:
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
_output_shapes
:*
T0*
out_type0
�
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:���������
*
T0
�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1
�
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������

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
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������

�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
�
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������

u
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
u
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
�
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$gradients/logistic_loss/mul_grad/mulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Placeholder_1*
T0*'
_output_shapes
:���������

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
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
�
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:���������
*
T0
�
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:���������

�
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:���������

�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*'
_output_shapes
:���������
*
T0
�
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������

�
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*'
_output_shapes
:���������
*
T0
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
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:���������

�
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������

�
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:���������

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
9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_2/BiasAdd_grad/BiasAddGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*
T0*=
_class3
1/loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad
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
8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:d
*
T0*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1
�
gradients/Relu_7_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_7*
T0*'
_output_shapes
:���������d
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_7_grad/ReluGraddense/kernel/read*
T0*(
_output_shapes
:����������
*
transpose_a( *
transpose_b(
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/Relu_7_grad/ReluGrad*
_output_shapes
:	�
d*
transpose_a(*
transpose_b( *
T0
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:����������
*
T0
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
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
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
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_5'gradients/conv2d_7/Conv2D_grad/ShapeN:1gradients/Relu_6_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
gradients/Relu_3_grad/ReluGradReluGrad7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyRelu_3*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNRelu_2conv2d_3/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/Relu_3_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2'gradients/conv2d_4/Conv2D_grad/ShapeN:1gradients/Relu_3_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter
�
gradients/Relu_2_grad/ReluGradReluGrad7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyRelu_2*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/Relu_2_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*E
_class;
97loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput
�
9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
gradients/Relu_1_grad/ReluGradReluGrad7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyRelu_1*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1Identity1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter.^gradients/conv2d/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*D
_class:
86loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
b
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
�
9GradientDescent/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelGradientDescent/learning_rate7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0* 
_class
loc:@conv2d/kernel
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel
�
;GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentApplyGradientDescentconv2d_3/kernelGradientDescent/learning_rate9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_5/kernel/ApplyGradientDescentApplyGradientDescentconv2d_5/kernelGradientDescent/learning_rate9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_5/kernel
�
;GradientDescent/update_conv2d_6/kernel/ApplyGradientDescentApplyGradientDescentconv2d_6/kernelGradientDescent/learning_rate9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d
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
ArgMaxArgMaxdense_2/BiasAddArgMax/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
T
ArgMax_1/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
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
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: "�s���      ?�K	\��?���AJ��
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
Placeholder_1Placeholder*'
_output_shapes
:���������
*
shape:���������
*
dtype0
R
Placeholder_2Placeholder*
dtype0
*
_output_shapes
:*
shape:
�
.conv2d/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d/kernel*%
valueB"            
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
dtype0*
_output_shapes
: * 
_class
loc:@conv2d/kernel*
valueB
 *���>
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 *
dtype0*&
_output_shapes
:
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv2d/kernel
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0* 
_class
loc:@conv2d/kernel
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
conv2d/kernel
VariableV2*&
_output_shapes
:*
shared_name * 
_class
loc:@conv2d/kernel*
	container *
shape:*
dtype0
�
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform* 
_class
loc:@conv2d/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
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
ReluReluconv2d/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel*%
valueB"            *
dtype0
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
valueB
 *�[q�*
dtype0
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
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 *
dtype0
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_1/kernel
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
conv2d_1/kernel
VariableV2*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape:*
dtype0
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
conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_1/kernel
g
conv2d_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
.conv2d_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *��:�
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
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_2/kernel
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
conv2d_2/kernel
VariableV2*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_2/kernel*
	container *
shape:*
dtype0
�
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
g
conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_3/Conv2DConv2DRelu_1conv2d_2/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
.conv2d_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
valueB
 *��
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
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_3/kernel*
	container 
�
conv2d_3/kernel/AssignAssignconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(
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
Relu_3Reluconv2d_4/Conv2D*/
_output_shapes
:���������*
T0
�
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_4/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_4/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel*
valueB
 *HY�
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
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_4/kernel
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
�
conv2d_4/kernel
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_4/kernel*
	container 
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
conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
g
conv2d_5/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
conv2d_5/Conv2DConv2DRelu_3conv2d_4/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_5/kernel
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
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_5/kernel*
	container *
shape:
�
conv2d_5/kernel/AssignAssignconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
g
conv2d_6/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
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
8conv2d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_6/kernel/Initializer/random_uniform/shape*"
_class
loc:@conv2d_6/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0
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
VariableV2*
shared_name *"
_class
loc:@conv2d_6/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
�
conv2d_6/kernel/AssignAssignconv2d_6/kernel*conv2d_6/kernel/Initializer/random_uniform*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(
�
conv2d_6/kernel/readIdentityconv2d_6/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
g
conv2d_7/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
Y
Relu_6Reluconv2d_7/Conv2D*/
_output_shapes
:���������*
T0
^
Reshape/shapeConst*
valueB"����\  *
dtype0*
_output_shapes
:
j
ReshapeReshapeRelu_6Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������

�
-dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"\  d   *
dtype0
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
VariableV2*
_class
loc:@dense/kernel*
	container *
shape:	�
d*
dtype0*
_output_shapes
:	�
d*
shared_name 
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
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel*
_output_shapes
:	�
d*
T0
�
dense/MatMulMatMulReshapedense/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
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
dtype0*
_output_shapes

:d
*

seed *
T0*!
_class
loc:@dense_1/kernel*
seed2 
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
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

�
dense_1/kernel
VariableV2*
_output_shapes

:d
*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape
:d
*
dtype0
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
dense_1/bias/Initializer/zerosConst*
_output_shapes
:
*
_class
loc:@dense_1/bias*
valueB
*    *
dtype0
�
dense_1/bias
VariableV2*
shared_name *
_class
loc:@dense_1/bias*
	container *
shape:
*
dtype0*
_output_shapes
:

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
logistic_loss/GreaterEqualGreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:���������
*
T0
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
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negdense_2/BiasAdd*
T0*'
_output_shapes
:���������

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
MeanMeanlogistic_lossConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
conv_loss/tagsConst*
dtype0*
_output_shapes
: *
valueB B	conv_loss
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
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
r
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
f
gradients/Mean_grad/ShapeShapelogistic_loss*
out_type0*
_output_shapes
:*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:���������
*

Tmultiples0*
T0
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
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( *

Tidx0
�
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
_output_shapes
:*
T0*
out_type0
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
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
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
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*'
_output_shapes
:���������
*
T0
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
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
u
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1*
out_type0*
_output_shapes
:*
T0
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
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
&gradients/logistic_loss/mul_grad/mul_1Muldense_2/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������

�
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
�
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape
�
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:���������
*
T0
�
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*'
_output_shapes
:���������
*
T0
�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*'
_output_shapes
:���������
*
T0
�
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*'
_output_shapes
:���������
*
T0
�
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*'
_output_shapes
:���������
*
T0
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
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:���������

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
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:

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
9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_2/BiasAdd_grad/BiasAddGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*
T0*=
_class3
1/loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad
�
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
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
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	�
d*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1
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
gradients/Relu_6_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_6*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_7/Conv2D_grad/ShapeNShapeNRelu_5conv2d_6/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_5'gradients/conv2d_7/Conv2D_grad/ShapeN:1gradients/Relu_6_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter
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
7gradients/conv2d_6/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_6/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_6/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter
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
2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_4/kernel/readgradients/Relu_4_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*E
_class;
97loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput
�
9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*F
_class<
:8loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
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
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
�
5gradients/conv2d/Conv2D_grad/tuple/control_dependencyIdentity0gradients/conv2d/Conv2D_grad/Conv2DBackpropInput.^gradients/conv2d/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*C
_class9
75loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropInput
�
7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1Identity1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter.^gradients/conv2d/Conv2D_grad/tuple/group_deps*D
_class:
86loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
b
GradientDescent/learning_rateConst*
valueB
 *
�#<*
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
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:*
use_locking( 
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
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
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
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d*
use_locking( 
�
:GradientDescent/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelGradientDescent/learning_rate8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:d
*
use_locking( *
T0*!
_class
loc:@dense_1/kernel
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
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
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
Const_1Const*
_output_shapes
:*
valueB: *
dtype0
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
N""�
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
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0����       `/�#	�@���A*

	conv_loss�1?lg[�       QKD	��"@���A*

	conv_lossv�0?���A       QKD	��"@���A*

	conv_losss�0?:�cn       QKD	(#@���A*

	conv_loss��0?�!�)       QKD	�D#@���A*

	conv_loss��0?*�>�       QKD	��#@���A*

	conv_loss�0?��0       QKD	��#@���A*

	conv_loss��0?77�       QKD	�	$@���A*

	conv_loss�0?�ט�       QKD	�D$@���A*

	conv_lossԮ0?���       QKD	N�$@���A	*

	conv_lossЦ0?A��       QKD	��$@���A
*

	conv_loss.�0?j��        QKD	�%@���A*

	conv_lossR�0?Ez��       QKD	�C%@���A*

	conv_loss��0?W1"       QKD	ځ%@���A*

	conv_loss
�0?d��       QKD	�%@���A*

	conv_loss�p0?��%�       QKD	�&@���A*

	conv_loss�b0?b�V�       QKD	�<&@���A*

	conv_loss�W0?����       QKD	�v&@���A*

	conv_lossqO0?]       QKD	��&@���A*

	conv_loss�F0?����       QKD	�'@���A*

	conv_loss�:0?���       QKD	�:'@���A*

	conv_loss�70?��'�       QKD	Sq'@���A*

	conv_loss/0?��T$       QKD	n�'@���A*

	conv_loss�0?E-�        QKD	!�'@���A*

	conv_loss�0?���       QKD	�3(@���A*

	conv_loss��/?e���       QKD	uj(@���A*

	conv_loss��/?ST�       QKD	��(@���A*

	conv_loss��/?���^       QKD	��(@���A*

	conv_loss6�/?����       QKD	�)@���A*

	conv_loss{�/?w�l�       QKD	C)@���A*

	conv_loss�/?;ʊ       QKD	}|)@���A*

	conv_loss�/?���       QKD	3�)@���A*

	conv_loss"�/?PI��       QKD	��)@���A *

	conv_loss�/?��+E       QKD	�#*@���A!*

	conv_loss��/?�x΋       QKD	*^*@���A"*

	conv_loss��/?ŀԢ       QKD	'�*@���A#*

	conv_loss>�/?x�f�       QKD	��*@���A$*

	conv_lossix/?)u�,       QKD	�+@���A%*

	conv_lossV�/?��f�       QKD	iI+@���A&*

	conv_loss7m/?]nl�       QKD	�~+@���A'*

	conv_loss�[/?kV�&       QKD	�+@���A(*

	conv_loss�O/?2e(�       QKD	��+@���A)*

	conv_loss�S/?�-�"       QKD	�9,@���A**

	conv_losspB/?�"R1       QKD	�p,@���A+*

	conv_loss:/?�Jq       QKD	�,@���A,*

	conv_loss/?�<��       QKD	��,@���A-*

	conv_lossN/?���b       QKD	�-@���A.*

	conv_loss
/?�<MJ       QKD	]-@���A/*

	conv_loss��.?�ߨ�       QKD	-�-@���A0*

	conv_loss</?��\�       QKD	e�-@���A1*

	conv_loss$�.?����       QKD	�.@���A2*

	conv_loss��.?Y�*L       QKD	�W.@���A3*

	conv_loss�.?�B�-       QKD	6�.@���A4*

	conv_loss��.?��       QKD	��.@���A5*

	conv_loss��.?E��       QKD	�	/@���A6*

	conv_loss5�.?��w:       QKD	gA/@���A7*

	conv_loss�.?/ph�       QKD	v~/@���A8*

	conv_loss2�.?�3�^       QKD	�/@���A9*

	conv_loss��.?\i*�       QKD	4�/@���A:*

	conv_loss��.?e;R^       QKD	�40@���A;*

	conv_lossD�.?��r       QKD	/n0@���A<*

	conv_lossW�.?1�_�       QKD	^�0@���A=*

	conv_loss�n.?��m�       QKD	f�0@���A>*

	conv_loss�c.?���       QKD	$1@���A?*

	conv_loss�[.?��/�       QKD	H^1@���A@*

	conv_lossY.?�!��       QKD	��1@���AA*

	conv_loss�I.?��       QKD	g�1@���AB*

	conv_loss�E.?;R�S       QKD	2@���AC*

	conv_loss..?�*�       QKD	Uq2@���AD*

	conv_loss�2.?�P)"       QKD	��2@���AE*

	conv_loss�.?f\�       QKD	��2@���AF*

	conv_loss/.?.<M       QKD	�3@���AG*

	conv_loss"	.?��s       QKD	I3@���AH*

	conv_loss��-?G��H       QKD	�z3@���AI*

	conv_lossR�-??K�       QKD	d�3@���AJ*

	conv_lossG�-?	q��       QKD	A�3@���AK*

	conv_loss5�-?.V �       QKD	!4@���AL*

	conv_loss��-?B䘤       QKD	�Z4@���AM*

	conv_lossN�-?g�>B       QKD	��4@���AN*

	conv_loss��-?&_#U       QKD	��4@���AO*

	conv_lossɨ-?�>�       QKD	��4@���AP*

	conv_loss�-?�Ar       QKD	x.5@���AQ*

	conv_loss�-?�n��       QKD	a5@���AR*

	conv_loss-?w�       QKD	{�5@���AS*

	conv_lossy�-?��(       QKD	��5@���AT*

	conv_lossjy-?���k       QKD	6@���AU*

	conv_lossAv-?�<       QKD	�<6@���AV*

	conv_loss�g-?���d       QKD	�m6@���AW*

	conv_loss
]-?y&ş       QKD	3�6@���AX*

	conv_lossmI-?ń��       QKD	q�6@���AY*

	conv_loss�<-?�0�       QKD	�7@���AZ*

	conv_lossf>-?��       QKD	�S7@���A[*

	conv_loss*-?�v��       QKD	D�7@���A\*

	conv_lossE*-?���8       QKD	�7@���A]*

	conv_loss�-?s��       QKD	��7@���A^*

	conv_lossA	-?���a       QKD	8@���A_*

	conv_loss�-?:��m       QKD	uH8@���A`*

	conv_loss��,?	�7�       QKD	�x8@���Aa*

	conv_loss1�,?��U       QKD	�8@���Ab*

	conv_loss��,?ꪣ�       QKD	�8@���Ac*

	conv_loss��,?+��Y       QKD	9@���Ad*

	conv_loss��,?�=��       QKD	2^9@���Ae*

	conv_loss��,?2�6�       QKD	R�9@���Af*

	conv_lossR�,?L�ǟ       QKD	��9@���Ag*

	conv_loss��,?"��       QKD	:@���Ah*

	conv_loss��,?!���       QKD	�A:@���Ai*

	conv_lossȏ,?�Ǯ       QKD	�w:@���Aj*

	conv_lossI�,?3o��       QKD	��:@���Ak*

	conv_loss|,?ֻ.       QKD	��:@���Al*

	conv_losszu,?Q{��       QKD	!;@���Am*

	conv_loss�f,?���{       QKD	�T;@���An*

	conv_lossn\,?�౪       QKD	��;@���Ao*

	conv_loss�X,?ou��       QKD	��;@���Ap*

	conv_loss�F,?��a       QKD	��;@���Aq*

	conv_lossp%,?��z�       QKD	�'<@���Ar*

	conv_loss�3,?�:�t       QKD	�a<@���As*

	conv_loss�+,?	g�       QKD	l�<@���At*

	conv_loss#,?U�j�       QKD	 �<@���Au*

	conv_loss",?w��       QKD	j�<@���Av*

	conv_lossW,?��       QKD	�-=@���Aw*

	conv_loss��+?ϔpP       QKD	�\=@���Ax*

	conv_loss]�+?(OLL       QKD	p�=@���Ay*

	conv_lossR�+?@�b       QKD	��=@���Az*

	conv_loss��+?��4�       QKD	"�=@���A{*

	conv_lossS�+?�]5�       QKD	6>@���A|*

	conv_loss��+?V�       QKD	h>@���A}*

	conv_loss��+?��ԫ       QKD	۞>@���A~*

	conv_loss��+?��GR       QKD	��>@���A*

	conv_loss-�+?�)�        )��P	?@���A�*

	conv_loss�+?Ae        )��P	�L?@���A�*

	conv_loss�+?�bp�        )��P	�~?@���A�*

	conv_loss݀+?�o�        )��P	?�?@���A�*

	conv_loss�x+?o�4t        )��P	��?@���A�*

	conv_loss�i+?�(m        )��P	�@@���A�*

	conv_loss�l+?��`�        )��P	�O@@���A�*

	conv_loss<_+?�{Ւ        )��P	ˑ@@���A�*

	conv_loss�C+?.f��        )��P	`�@@���A�*

	conv_loss�I+?�]�        )��P	��@@���A�*

	conv_lossTF+?R���        )��P	$1A@���A�*

	conv_losse2+?����        )��P	�wA@���A�*

	conv_lossq.+?t�        )��P	R�A@���A�*

	conv_loss/'+?�^Ƀ        )��P	}�A@���A�*

	conv_loss�+?s��        )��P	�
B@���A�*

	conv_loss�+?��c�        )��P	�<B@���A�*

	conv_loss��*?,�	�        )��P	�{B@���A�*

	conv_lossp�*?�U�         )��P	��B@���A�*

	conv_loss��*?u�        )��P	��B@���A�*

	conv_loss�*?S�.�        )��P	C@���A�*

	conv_loss��*?��]        )��P	�LC@���A�*

	conv_loss��*?,Hx�        )��P	��C@���A�*

	conv_loss��*?�K\�        )��P	�C@���A�*

	conv_lossY�*?{g~�        )��P	�D@���A�*

	conv_loss�*??�@�        )��P	�:D@���A�*

	conv_lossܜ*?��%        )��P	�mD@���A�*

	conv_lossx*?A�h        )��P	�D@���A�*

	conv_loss]y*?�5C        )��P	U�D@���A�*

	conv_lossDk*?���9        )��P	�E@���A�*

	conv_loss�q*?��w        )��P	�1E@���A�*

	conv_lossUc*?��&�        )��P	�mE@���A�*

	conv_loss�g*?9��        )��P	�E@���A�*

	conv_lossI*?3@N�        )��P	��E@���A�*

	conv_loss�A*?���]        )��P	�F@���A�*

	conv_lossR#*?��        )��P	�EF@���A�*

	conv_loss�.*? %�        )��P	X�F@���A�*

	conv_lossb#*?�-ܟ        )��P	ռF@���A�*

	conv_loss�*?�F        )��P	��F@���A�*

	conv_lossi*?�͉        )��P	$G@���A�*

	conv_loss� *?��        )��P	�VG@���A�*

	conv_loss'�)?8��        )��P	 �G@���A�*

	conv_loss/�)?��rX        )��P	d�G@���A�*

	conv_loss]�)?p��        )��P	��G@���A�*

	conv_lossG�)?�y��        )��P	�0H@���A�*

	conv_loss�)?l�L        )��P	cH@���A�*

	conv_loss��)?m>p�        )��P	s�H@���A�*

	conv_loss.�)?����        )��P	��H@���A�*

	conv_lossF�)?��        )��P	}I@���A�*

	conv_loss\�)?E�ܩ        )��P	�2I@���A�*

	conv_loss�)?�        )��P	fI@���A�*

	conv_loss�)?OU�        )��P	��I@���A�*

	conv_loss0p)?���        )��P	K�I@���A�*

	conv_loss�s)?���        )��P	�J@���A�*

	conv_loss�o)?ʾ�        )��P	�BJ@���A�*

	conv_loss/Z)?��Z4        )��P	�sJ@���A�*

	conv_loss�?)?��(�        )��P	�J@���A�*

	conv_loss[L)?�N=        )��P	��J@���A�*

	conv_loss�9)?L�`        )��P	�K@���A�*

	conv_lossU4)??�k�        )��P	�5K@���A�*

	conv_loss)?��<        )��P	ClK@���A�*

	conv_lossk*)?�	��        )��P	��K@���A�*

	conv_loss/)?��         )��P	��K@���A�*

	conv_lossG)?����        )��P	p	L@���A�*

	conv_loss��(?�^�        )��P	�<L@���A�*

	conv_loss�(?���E        )��P	xoL@���A�*

	conv_loss��(?1��        )��P	ˬL@���A�*

	conv_loss5�(?L��        )��P	��L@���A�*

	conv_loss��(?���x        )��P	M@���A�*

	conv_loss��(?�A(        )��P	�OM@���A�*

	conv_loss��(?	��4        )��P	E�M@���A�*

	conv_loss��(?�Jsc        )��P	��M@���A�*

	conv_loss}�(?��'        )��P	�M@���A�*

	conv_loss��(?��B�        )��P	G)N@���A�*

	conv_loss��(?�)n        )��P	��O@���A�*

	conv_lossΒ(?��F        )��P	��O@���A�*

	conv_loss�e(?>���        )��P	h!P@���A�*

	conv_loss�Z(?n�1�        )��P	`WP@���A�*

	conv_loss>^(?�\�V        )��P	��P@���A�*

	conv_loss"R(?B��        )��P	��P@���A�*

	conv_loss]2(?�{�        )��P	�P@���A�*

	conv_lossU/(?bu��        )��P	�*Q@���A�*

	conv_loss�$(?����        )��P	WkQ@���A�*

	conv_loss�"(?�f�        )��P	N�Q@���A�*

	conv_loss(?���z        )��P	��Q@���A�*

	conv_loss�'?� �        )��P	�R@���A�*

	conv_loss��'?���        )��P	�BR@���A�*

	conv_loss��'?r�|]        )��P	byR@���A�*

	conv_loss�'?�mm        )��P	�R@���A�*

	conv_loss1�'?�<�.        )��P	��R@���A�*

	conv_loss_�'?	��        )��P	ZS@���A�*

	conv_loss��'?�U'        )��P	�SS@���A�*

	conv_loss��'?9�G        )��P	�S@���A�*

	conv_loss�'?p�n        )��P	�S@���A�*

	conv_loss��'?����        )��P	��S@���A�*

	conv_loss�'?���        )��P	MT@���A�*

	conv_loss��'?��[        )��P	,ST@���A�*

	conv_loss��'?}�7        )��P	z�T@���A�*

	conv_loss�j'?#�Z
        )��P	��T@���A�*

	conv_lossW'?|<�        )��P	��T@���A�*

	conv_lossEX'?ԭ�        )��P	X,U@���A�*

	conv_loss�Y'??��        )��P	q`U@���A�*

	conv_loss�4'?�lz        )��P	ɛU@���A�*

	conv_loss�@'?���        )��P	��U@���A�*

	conv_loss04'?3��        )��P	�V@���A�*

	conv_loss�-'?���        )��P	�?V@���A�*

	conv_loss�'?[-��        )��P	�pV@���A�*

	conv_loss�'?B��        )��P	ۣV@���A�*

	conv_loss"'?���
        )��P	��V@���A�*

	conv_loss?�&?��A�        )��P	>W@���A�*

	conv_loss�&?I��        )��P	�MW@���A�*

	conv_loss��&?��h�        )��P	�}W@���A�*

	conv_lossG�&?�$2        )��P	K�W@���A�*

	conv_loss\�&?%b-�        )��P	M�W@���A�*

	conv_loss��&?a���        )��P	9X@���A�*

	conv_loss��&?��;�        )��P	ZX@���A�*

	conv_lossz�&?ہ�        )��P	��X@���A�*

	conv_loss͙&?B~k1        )��P	J�X@���A�*

	conv_loss��&?e}        )��P	��X@���A�*

	conv_loss�u&?�/v�        )��P	�)Y@���A�*

	conv_loss'z&?�F�}        )��P	�\Y@���A�*

	conv_loss~k&?��ZM        )��P	��Y@���A�*

	conv_lossIW&?�C�.        )��P	��Y@���A�*

	conv_loss�[&?{�B�        )��P	4Z@���A�*

	conv_loss4?&?�ݾ        )��P	�KZ@���A�*

	conv_loss�B&?���K        )��P	U�Z@���A�*

	conv_loss�&?�y        )��P	��Z@���A�*

	conv_lossE/&?Ū��        )��P	1�Z@���A�*

	conv_loss�&?�8c�        )��P	�[@���A�*

	conv_loss$&?���        )��P	�P[@���A�*

	conv_loss\ &?3�o        )��P	�~[@���A�*

	conv_loss&?f��p        )��P	4�[@���A�*

	conv_loss��%?m�b8        )��P	��[@���A�*

	conv_loss3�%?���        )��P	$\@���A�*

	conv_loss@�%?�>��        )��P	�Y\@���A�*

	conv_lossv�%?9�e�        )��P	��\@���A�*

	conv_loss2�%?�A��        )��P	;�\@���A�*

	conv_lossŲ%?�!�        )��P	��\@���A�*

	conv_lossJ�%?d�s�        )��P	-4]@���A�*

	conv_loss��%?*Ԟ<        )��P	�o]@���A�*

	conv_loss)�%?���h        )��P	��]@���A�*

	conv_lossEo%?f6a        )��P	#�]@���A�*

	conv_loss�`%?퉲�        )��P	�^@���A�*

	conv_lossD%?@] �        )��P	�>^@���A�*

	conv_loss�G%?��"�        )��P	�x^@���A�*

	conv_loss�.%?��Q�        )��P	Ϯ^@���A�*

	conv_loss=F%?����        )��P	a�^@���A�*

	conv_lossl%?��(        )��P	_@���A�*

	conv_loss�%?Jj�        )��P	�R_@���A�*

	conv_loss>�$?d��        )��P	��_@���A�*

	conv_loss��$?L�D<        )��P	ʿ_@���A�*

	conv_loss%�$?4��        )��P	�_@���A�*

	conv_loss�$?ٚ�;        )��P	�$`@���A�*

	conv_loss�$?��8�        )��P	Y`@���A�*

	conv_loss|�$?�F:�        )��P	%�`@���A�*

	conv_lossk�$?v���        )��P	�`@���A�*

	conv_loss�$?}���        )��P	��`@���A�*

	conv_losse�$?����        )��P	�#a@���A�*

	conv_lossk�$?-k�Q        )��P	
[a@���A�*

	conv_loss�t$?�/�        )��P	a@���A�*

	conv_losss�$?����        )��P	��a@���A�*

	conv_lossk$?b��n        )��P		�a@���A�*

	conv_lossEr$?��1q        )��P	�(b@���A�*

	conv_lossS;$?���        )��P	3bb@���A�*

	conv_lossU]$?�ݶ        )��P	��b@���A�*

	conv_loss_6$?�d/�        )��P	��b@���A�*

	conv_loss�*$?v;t�        )��P	+�b@���A�*

	conv_loss�7$?Cy��        )��P	�$c@���A�*

	conv_loss��#?�C`�        )��P	�^c@���A�*

	conv_loss�$?����        )��P	�c@���A�*

	conv_loss��#?n@dW        )��P	6�c@���A�*

	conv_loss��#?��e�        )��P	�c@���A�*

	conv_lossJ�#?��;j        )��P	�d@���A�*

	conv_lossj�#?l�ӯ        )��P	�Od@���A�*

	conv_loss�#?p�6t        )��P	�d@���A�*

	conv_loss:�#?ˈ}        )��P	��d@���A�*

	conv_loss�#?�#        )��P	� e@���A�*

	conv_lossb�#?}��        )��P	�2e@���A�*

	conv_loss*d#?�~r�        )��P	�oe@���A�*

	conv_loss8d#?<��<        )��P	�e@���A�*

	conv_loss�r#?=�))        )��P	��e@���A�*

	conv_lossPC#?!�        )��P	T	f@���A�*

	conv_loss=#?;���        )��P	�<f@���A�*

	conv_loss�#?��K�        )��P	vf@���A�*

	conv_loss�'#?>k��        )��P	�f@���A�*

	conv_loss�"?�tUN        )��P	z�f@���A�*

	conv_loss��"?��I@        )��P	mg@���A�*

	conv_loss.�"?,+        )��P	�>g@���A�*

	conv_loss��"?��        )��P	��g@���A�*

	conv_loss�"?�E�        )��P	ٰg@���A�*

	conv_loss	�"?!�G        )��P	��g@���A�*

	conv_lossϷ"?ڡ�        )��P	dh@���A�*

	conv_loss�"?r)[.        )��P	1Eh@���A�*

	conv_loss��"?��        )��P	��h@���A�*

	conv_loss�b"?��6        )��P	q�h@���A�*

	conv_loss�Q"?��Ѻ        )��P	~�h@���A�*

	conv_loss�Y"?>6�        )��P	!i@���A�*

	conv_loss�e"?����        )��P	�Fi@���A�*

	conv_loss�"?��y�        )��P	ivi@���A�*

	conv_lossX#"?��e        )��P	L�i@���A�*

	conv_loss*"?x���        )��P	_�i@���A�*

	conv_loss"?ͮ&�        )��P	�j@���A�*

	conv_loss��!?L�/        )��P	�Kj@���A�*

	conv_loss��!?z@=        )��P	b�j@���A�*

	conv_loss&�!?eq(z        )��P	��j@���A�*

	conv_loss��!?>GP        )��P	�j@���A�*

	conv_loss��!?���        )��P	ik@���A�*

	conv_loss5p!?|�N        )��P	�Qk@���A�*

	conv_loss�x!?��        )��P	K�k@���A�*

	conv_loss�W!?��        )��P	��k@���A�*

	conv_lossQf!?�Aj�        )��P	n�k@���A�*

	conv_lossZ;!?f(:         )��P	Il@���A�*

	conv_lossx!?L%:        )��P	�[l@���A�*

	conv_loss�!?�*        )��P	 �l@���A�*

	conv_loss"!?S��t        )��P	��l@���A�*

	conv_lossv� ?����        )��P	4�l@���A�*

	conv_loss"� ?�CL        )��P	-m@���A�*

	conv_loss�� ?n�.]        )��P	�^m@���A�*

	conv_loss+� ?eX�5        )��P	ўm@���A�*

	conv_loss�� ?N��        )��P	��m@���A�*

	conv_loss�[ ?C�t�        )��P	Q�m@���A�*

	conv_loss�h ?�E4        )��P	/n@���A�*

	conv_loss�* ?�p         )��P		`n@���A�*

	conv_loss� ?^Ow,        )��P	ɒn@���A�*

	conv_loss�E ?f��7        )��P	��n@���A�*

	conv_loss' ?;23�        )��P	a3o@���A�*

	conv_lossi9 ?[���        )��P	Vjo@���A�*

	conv_loss� ?K$S�        )��P	͛o@���A�*

	conv_lossb�?n ��        )��P	L�o@���A�*

	conv_loss�?���        )��P	B�o@���A�*

	conv_loss��?�W�        )��P	�,p@���A�*

	conv_lossT�?�{c�        )��P	+[p@���A�*

	conv_lossX?V��        )��P	]�p@���A�*

	conv_lossUm?r.m�        )��P	��p@���A�*

	conv_lossM?�h��        )��P	��p@���A�*

	conv_lossx�?�"�X        )��P	n,q@���A�*

	conv_loss�?p�g�        )��P	�eq@���A�*

	conv_loss��?�2EJ        )��P	3�q@���A�*

	conv_loss��?8㾍        )��P	�q@���A�*

	conv_loss�?�u�        )��P	E	r@���A�*

	conv_lossdj?f@��        )��P	�7r@���A�*

	conv_lossbS?��M        )��P	�gr@���A�*

	conv_loss�&?s�A~        )��P	��r@���A�*

	conv_loss�?��K\        )��P	��r@���A�*

	conv_losso"?�^Ԑ        )��P	��r@���A�*

	conv_loss��?�t�(        )��P	-'s@���A�*

	conv_loss$�?ޕ�4        )��P	Ws@���A�*

	conv_loss�?i�Db        )��P	4�s@���A�*

	conv_lossGj?���        )��P	Ŵs@���A�*

	conv_loss��?�Ua�        )��P	��s@���A�*

	conv_loss�*?��]        )��P	w$t@���A�*

	conv_loss�3?��G        )��P	8Ut@���A�*

	conv_loss�X?�r��        )��P	��t@���A�*

	conv_loss8�?}E�        )��P	γt@���A�*

	conv_loss.�?u*�        )��P	B�t@���A�*

	conv_loss��?K1�+        )��P	mu@���A�*

	conv_loss^�?ƍ        )��P	�@u@���A�*

	conv_loss4B?\(��        )��P	�nu@���A�*

	conv_lossC�?73�q        )��P	R�u@���A�*

	conv_lossߵ?=�Y        )��P	*�u@���A�*

	conv_loss�[?+^h1        )��P	A�u@���A�*

	conv_lossݗ?��M�        )��P	B*v@���A�*

	conv_losse+?C��0        )��P	yZv@���A�*

	conv_loss6'?B��        )��P	9�v@���A�*

	conv_loss�w?�SO�        )��P	��v@���A�*

	conv_loss�^?��*        )��P	 �v@���A�*

	conv_loss�C?+`�        )��P	fw@���A�*

	conv_loss�?��        )��P	Hw@���A�*

	conv_loss7p?��,j        )��P	\ww@���A�*

	conv_lossգ?�f�S        )��P	ݤw@���A�*

	conv_loss<�?���        )��P	��w@���A�*

	conv_loss)?	��        )��P	�x@���A�*

	conv_loss��?<��_        )��P	�1x@���A�*

	conv_loss�u?���        )��P	�bx@���A�*

	conv_loss??���/        )��P	w�y@���A�*

	conv_lossY?�<�        )��P	�&z@���A�*

	conv_loss��?c���        )��P	�Vz@���A�*

	conv_loss(�?9���        )��P	`�z@���A�*

	conv_loss��?��        )��P	k�z@���A�*

	conv_loss/�?�ԙ6        )��P	\�z@���A�*

	conv_lossyT?ݭ�e        )��P	q {@���A�*

	conv_lossڬ?dcl        )��P	�^{@���A�*

	conv_loss��?aTX�        )��P	L�{@���A�*

	conv_loss�?R��        )��P	�{@���A�*

	conv_lossj?I���        )��P	�|@���A�*

	conv_loss?�S<-        )��P	�C|@���A�*

	conv_loss�?��        )��P	�v|@���A�*

	conv_loss��?���        )��P	��|@���A�*

	conv_loss�j?s>�        )��P	"�|@���A�*

	conv_lossP:?�Du        )��P	;}@���A�*

	conv_loss�|?��        )��P	>}@���A�*

	conv_loss�?�I.�        )��P	n}@���A�*

	conv_loss-�?�9z�        )��P	<�}@���A�*

	conv_loss�k?�V,         )��P	��}@���A�*

	conv_lossC�?2%�        )��P	�~@���A�*

	conv_lossN�?ѭ>        )��P	eJ~@���A�*

	conv_loss�3?�n��        )��P	�|~@���A�*

	conv_loss`�?��q�        )��P	��~@���A�*

	conv_loss=V?:@X�        )��P	H�~@���A�*

	conv_lossg{
?�W)�        )��P	l@���A�*

	conv_lossCX
?Dm�        )��P	ZD@���A�*

	conv_loss	?���s        )��P	�t@���A�*

	conv_loss�P?+>��        )��P	�@���A�*

	conv_loss[�?��        )��P	��@���A�*

	conv_loss6�?�%�        )��P	
�@���A�*

	conv_loss%�?�!�        )��P	�:�@���A�*

	conv_loss��?�\�:        )��P	l�@���A�*

	conv_lossƍ??<�        )��P	ݜ�@���A�*

	conv_lossp?B�7�        )��P	�̀@���A�*

	conv_loss��>Rz�        )��P	���@���A�*

	conv_loss#��>�]M6        )��P	,�@���A�*

	conv_loss8z�>�.��        )��P	�\�@���A�*

	conv_loss;�>r���        )��P	i��@���A�*

	conv_loss�e�>��	�        )��P	���@���A�*

	conv_loss�4�>���v        )��P	��@���A�*

	conv_loss�Z�>&v��        )��P	��@���A�*

	conv_lossG��>�F=�        )��P	aO�@���A�*

	conv_loss���>��i        )��P	���@���A�*

	conv_loss�J�>�N�        )��P	7��@���A�*

	conv_lossc��>
$c�        )��P	S�@���A�*

	conv_loss:��>�A��        )��P	o�@���A�*

	conv_loss`�>h|R�        )��P	�K�@���A�*

	conv_lossV�>�Ԙ`        )��P	�|�@���A�*

	conv_loss�Q�>���{        )��P	Z��@���A�*

	conv_loss�>�yn5        )��P	��@���A�*

	conv_loss_��>(?CZ        )��P	\"�@���A�*

	conv_lossI%�>��!A        )��P	=T�@���A�*

	conv_lossV��>
u�        )��P	ׇ�@���A�*

	conv_loss1�>.�        )��P	z��@���A�*

	conv_lossgk�>��&r        )��P	�@���A�*

	conv_loss"��>5�Q�        )��P	~!�@���A�*

	conv_loss���>Dp        )��P	R�@���A�*

	conv_loss���>���        )��P	@���A�*

	conv_loss_Ĺ>���        )��P	�х@���A�*

	conv_loss���>p44        )��P	D�@���A�*

	conv_lossի�>��S�        )��P	g2�@���A�*

	conv_loss���>���V        )��P	�a�@���A�*

	conv_loss�~�>�zk�        )��P	R��@���A�*

	conv_loss*�>Z@�G        )��P	OȆ@���A�*

	conv_lossV�>�5�_        )��P	u��@���A�*

	conv_loss��>3� �        )��P	�'�@���A�*

	conv_lossu�>����        )��P	`Y�@���A�*

	conv_loss��>�YZ�        )��P	���@���A�*

	conv_losse��>YѲ�        )��P	��@���A�*

	conv_lossXѵ>���        )��P	���@���A�*

	conv_loss�7�>vUj�        )��P	�,�@���A�*

	conv_loss_��>�3h�        )��P	c�@���A�*

	conv_loss���>�
��        )��P	Y��@���A�*

	conv_lossR۲>U{�;        )��P	Bǈ@���A�*

	conv_loss`�>5I�        )��P	���@���A�*

	conv_lossD��>:��        )��P	�(�@���A�*

	conv_lossJ8�>�*�        )��P	�[�@���A�*

	conv_loss���>���        )��P	ۍ�@���A�*

	conv_loss9ײ>d���        )��P	���@���A�*

	conv_lossv�>?	�        )��P	 �@���A�*

	conv_loss:r�>vIB        )��P	�"�@���A�*

	conv_lossa֬>ݾ58        )��P	aS�@���A�*

	conv_loss�g�>��#        )��P	��@���A�*

	conv_lossƶ�>J�        )��P	���@���A�*

	conv_loss"��>2B��        )��P	��@���A�*

	conv_lossy/�>�w        )��P	��@���A�*

	conv_lossQɳ>ee:�        )��P	�H�@���A�*

	conv_loss扰>;��        )��P	�z�@���A�*

	conv_lossv��>����        )��P	��@���A�*

	conv_loss��>�@|�        )��P	�݋@���A�*

	conv_loss�ݯ>���        )��P	:�@���A�*

	conv_lossU��>����        )��P	@�@���A�*

	conv_loss�s�>m��        )��P	jp�@���A�*

	conv_loss��>�T��        )��P	���@���A�*

	conv_loss5�>T*�        )��P	�Ҍ@���A�*

	conv_loss~l�>�䞄        )��P	�@���A�*

	conv_loss�X�>CR>        )��P	33�@���A�*

	conv_loss�>�>����        )��P	Dd�@���A�*

	conv_loss�;�>��u        )��P	�@���A�*

	conv_lossCo�>M�$�        )��P	IU�@���A�*

	conv_lossMC�>�
j�        )��P	��@���A�*

	conv_loss�G�>s�F        )��P	a��@���A�*

	conv_loss���>�r        )��P	��@���A�*

	conv_loss��>K��L        )��P	A�@���A�*

	conv_loss��>~�P        )��P	~G�@���A�*

	conv_lossMı>O<�&        )��P	Ru�@���A�*

	conv_loss0�>��2�        )��P	���@���A�*

	conv_loss��>~<�         )��P	�Г@���A�*

	conv_losss��>kz`        )��P	��@���A�*

	conv_loss�(�>},E�        )��P	SA�@���A�*

	conv_lossr��>�        )��P	Oo�@���A�*

	conv_loss�*�>��o4        )��P	��@���A�*

	conv_loss2˰>��7        )��P	�ؔ@���A�*

	conv_loss��>0+�Q        )��P	��@���A�*

	conv_loss���>�p��        )��P	�>�@���A�*

	conv_loss��>��iF        )��P	�n�@���A�*

	conv_loss���>|W�m        )��P	���@���A�*

	conv_lossry�>�<��        )��P	�˕@���A�*

	conv_loss��>�[x/        )��P	���@���A�*

	conv_loss��>��]N        )��P	)�@���A�*

	conv_lossj��>���C        )��P	Z�@���A�*

	conv_loss<��>Z�/        )��P	ȉ�@���A�*

	conv_lossp��>zA�        )��P	E��@���A�*

	conv_lossN��>~��	        )��P	V�@���A�*

	conv_lossH��>�&�        )��P	$�@���A�*

	conv_loss�7�>̶m        )��P	LJ�@���A�*

	conv_loss���>����        )��P	�z�@���A�*

	conv_loss��>)L:        )��P		��@���A�*

	conv_loss���>�$d�        )��P	�ڗ@���A�*

	conv_loss̬>\"��        )��P	�	�@���A�*

	conv_loss¸�>��/G        )��P	f8�@���A�*

	conv_loss|p�>'�-�        )��P	�g�@���A�*

	conv_loss=��>TV�        )��P	���@���A�*

	conv_loss{�>�        )��P	Ș@���A�*

	conv_loss:ڪ>t�ġ        )��P	y��@���A�*

	conv_loss��>&��g        )��P	�%�@���A�*

	conv_loss���>��SS        )��P	.Y�@���A�*

	conv_lossԒ�>"'�        )��P	���@���A�*

	conv_loss�.�>�B        )��P	���@���A�*

	conv_loss=��>��q�        )��P	G�@���A�*

	conv_loss+��>j4T        )��P	;�@���A�*

	conv_loss쌭>��z�        )��P	�E�@���A�*

	conv_loss64�>P���        )��P	ju�@���A�*

	conv_loss�>��ԇ        )��P	0��@���A�*

	conv_loss�!�>e	+H        )��P	6՚@���A�*

	conv_loss�q�>k"��        )��P	��@���A�*

	conv_loss��>x�շ        )��P	J4�@���A�*

	conv_loss��>OE�        )��P	�b�@���A�*

	conv_loss�ի>�j��        )��P	Α�@���A�*

	conv_lossSܬ>k���        )��P	@֛@���A�*

	conv_loss�e�>�v-!        )��P	e�@���A�*

	conv_lossk��>��b        )��P	�5�@���A�*

	conv_loss��>�P�        )��P	Kf�@���A�*

	conv_lossQ�>���|        )��P	��@���A�*

	conv_loss�R�>g��        )��P	t̜@���A�*

	conv_loss�7�>�.��        )��P	���@���A�*

	conv_lossP�>HVhP        )��P	|-�@���A�*

	conv_loss���>�_        )��P	�k�@���A�*

	conv_lossO;�>��U#        )��P	o��@���A�*

	conv_loss䚩>�[�         )��P	�֝@���A�*

	conv_loss��>��Y        )��P	�	�@���A�*

	conv_losso�>��4        )��P	�:�@���A�*

	conv_lossPW�>�:%d        )��P	s�@���A�*

	conv_loss�a�>ǼY�        )��P	���@���A�*

	conv_loss$�> ��%        )��P	�ڞ@���A�*

	conv_lossyڭ>����        )��P	��@���A�*

	conv_loss�ެ>Mt0        )��P	�F�@���A�*

	conv_loss�>��;�        )��P	<z�@���A�*

	conv_loss�c�>��        )��P	p��@���A�*

	conv_loss ]�>��|�        )��P	ܟ@���A�*

	conv_lossk��>��B_        )��P	�@���A�*

	conv_loss=�>d,�        )��P	\>�@���A�*

	conv_loss`e�>�PJv        )��P	�m�@���A�*

	conv_loss���>Li��        )��P	%��@���A�*

	conv_loss�c�>����        )��P	�Ϡ@���A�*

	conv_loss5ܪ>GF        )��P	��@���A�*

	conv_lossY�>8VA�        )��P	C4�@���A�*

	conv_loss�0�>��5        )��P	Eg�@���A�*

	conv_loss+��>�{�v        )��P	��@���A�*

	conv_loss�!�>jZ��        )��P	�̡@���A�*

	conv_lossW��>��Ek        )��P	��@���A�*

	conv_loss#U�>UO]�        )��P	�2�@���A�*

	conv_loss��>Rq        )��P	�d�@���A�*

	conv_loss�«>�'�        )��P	���@���A�*

	conv_loss�%�>����        )��P	�Ң@���A�*

	conv_loss�˫>Kh�        )��P	o
�@���A�*

	conv_loss��>�\�        )��P	`@�@���A�*

	conv_loss�q�>�ݻb        )��P	E��@���A�*

	conv_loss¸�>�/�        )��P	£@���A�*

	conv_lossr:�>hI�        )��P	�@���A�*

	conv_losss:�>7�O        )��P	*<�@���A�*

	conv_loss���>	y��        )��P	/o�@���A�*

	conv_lossD�>{�        )��P	h��@���A�*

	conv_loss�6�>螋S        )��P	�פ@���A�*

	conv_lossi�>B.N�        )��P	�@���A�*

	conv_loss�ɩ>+��        )��P	JE�@���A�*

	conv_lossT�>=xf        )��P	�x�@���A�*

	conv_lossC��>���        )��P	i��@���A�*

	conv_loss���>ϤN        )��P	��@���A�*

	conv_loss%`�>s�N�        )��P	ϙ�@���A�*

	conv_loss�q�>��        )��P	�ѧ@���A�*

	conv_loss! �>��        )��P	�@���A�*

	conv_lossht�>�T��        )��P	U8�@���A�*

	conv_lossX��>�aR�        )��P	�n�@���A�*

	conv_lossa�>fT~        )��P	g��@���A�*

	conv_loss~�>��iW        )��P	���@���A�*

	conv_loss�-�>��&        )��P	':�@���A�*

	conv_loss}۬>�        )��P	�k�@���A�*

	conv_loss���>|h("        )��P	���@���A�*

	conv_loss/"�>j�E�        )��P	Y֩@���A�*

	conv_loss���>]�        )��P	x	�@���A�*

	conv_loss��>����        )��P	q9�@���A�*

	conv_loss?	�>��hi        )��P	�q�@���A�*

	conv_lossb�>|�c*        )��P	���@���A�*

	conv_lossJΨ>_�QH        )��P	}٪@���A�*

	conv_loss��>z�]        )��P	?
�@���A�*

	conv_loss��>���        )��P	:�@���A�*

	conv_loss��>8�M�        )��P	�o�@���A�*

	conv_losssΪ>�ህ        )��P	���@���A�*

	conv_loss��>"Bc        )��P	$�@���A�*

	conv_loss�d�>_79+        )��P	M�@���A�*

	conv_lossu��>X�        )��P	F�@���A�*

	conv_lossXߨ>���        )��P	�x�@���A�*

	conv_loss�>�> v        )��P	���@���A�*

	conv_lossܯ�>&�9�        )��P	��@���A�*

	conv_losst��>��.�        )��P	'�@���A�*

	conv_losss|�>N�F        )��P	�M�@���A�*

	conv_lossa��>����        )��P	W��@���A�*

	conv_loss�>����        )��P	+��@���A�*

	conv_loss�թ>�Eb        )��P	���@���A�*

	conv_loss���>�K�        )��P	g%�@���A�*

	conv_loss�c�>���y        )��P	�W�@���A�*

	conv_loss(�>���        )��P	߉�@���A�*

	conv_loss�ܨ>�(��        )��P	)®@���A�*

	conv_loss�ا>
��        )��P	���@���A�*

	conv_loss[Q�>�L�-        )��P	�1�@���A�*

	conv_loss<�>�ۮ�        )��P	Jb�@���A�*

	conv_loss.է>֮�"        )��P	`��@���A�*

	conv_loss���>W5        )��P	�ů@���A�*

	conv_loss�c�>P��        )��P	� �@���A�*

	conv_loss��>�]�        )��P	�/�@���A�*

	conv_loss�?�>&���        )��P	Qd�@���A�*

	conv_loss7��>��5        )��P	��@���A�*

	conv_loss0e�>c6��        )��P	�ΰ@���A�*

	conv_loss�|�>�:�k        )��P	�@���A�*

	conv_loss�v�>��>�        )��P	�>�@���A�*

	conv_loss/V�>KS[        )��P	Zn�@���A�*

	conv_lossHި>9.|�        )��P	6��@���A�*

	conv_loss�a�>b�"        )��P	�ϱ@���A�*

	conv_loss���>#��        )��P	��@���A�*

	conv_loss��>��F        )��P	�H�@���A�*

	conv_loss�ީ>�o\�        )��P	�y�@���A�*

	conv_loss
��>����        )��P	���@���A�*

	conv_loss��>Km��        )��P	��@���A�*

	conv_loss@�>}\�        )��P	?)�@���A�*

	conv_loss���>��x        )��P	�]�@���A�*

	conv_loss��>�X�        )��P	���@���A�*

	conv_losseЫ>��n        )��P	�@���A�*

	conv_loss�©>0?�         )��P	�@���A�*

	conv_loss�8�>���        )��P	�)�@���A�*

	conv_loss�X�>�M8�        )��P	;[�@���A�*

	conv_loss���>y��-        )��P	*��@���A�*

	conv_losso��>�p��        )��P	0��@���A�*

	conv_lossXw�>/�        )��P	��@���A�*

	conv_loss���>+�        )��P	u$�@���A�*

	conv_loss�Y�>>s}�        )��P	�V�@���A�*

	conv_loss�>���        )��P	���@���A�*

	conv_loss7��>���        )��P	�ʵ@���A�*

	conv_lossV��>�Oy        )��P	���@���A�*

	conv_loss�p�>M�b        )��P	�/�@���A�*

	conv_loss�ϧ>����        )��P	�a�@���A�*

	conv_loss���>�N:        )��P	��@���A�*

	conv_loss}I�>�j�        )��P	�Ŷ@���A�*

	conv_loss1�>c��        )��P	���@���A�*

	conv_lossco�>.��:        )��P	76�@���A�*

	conv_loss9�>8%M�        )��P	cj�@���A�*

	conv_loss�H�>�OA        )��P	[��@���A�*

	conv_lossK�>�J��        )��P	�з@���A�*

	conv_loss��>a;        )��P	��@���A�*

	conv_loss��>���        )��P	B�@���A�*

	conv_lossק>�Aau        )��P	�y�@���A�*

	conv_lossp�>�ڱP        )��P	���@���A�*

	conv_loss[^�>��^%        )��P	|ٸ@���A�*

	conv_loss�ө>w�~�        )��P	��@���A�*

	conv_loss�j�>��"�        )��P	�D�@���A�*

	conv_loss�\�>���R        )��P	�v�@���A�*

	conv_loss�n�>�X��        )��P	몹@���A�*

	conv_loss�̩>Ei�t        )��P	�ܹ@���A�*

	conv_loss%��>*ͼ        )��P	��@���A�*

	conv_lossNr�>S�8�        )��P	BH�@���A�*

	conv_loss���>�V�        )��P	m��@���A�*

	conv_lossB6�>�r�        )��P	�ƺ@���A�*

	conv_lossf4�>-2��        )��P	���@���A�*

	conv_loss���>O\K        )��P	0+�@���A�*

	conv_loss���>a���        )��P	�b�@���A�*

	conv_lossp��>n��        )��P	���@���A�*

	conv_loss�7�>r���        )��P	�λ@���A�*

	conv_loss�N�>�{        )��P	X �@���A�*

	conv_loss�o�>��        )��P	%4�@���A�*

	conv_loss0�>�J�        )��P	y�@���A�*

	conv_loss���>9Y��        )��P	<��@���A�*

	conv_loss�Ǧ>�%�        )��P	-�@���A�*

	conv_loss�B�>z��        )��P	%%�@���A�*

	conv_loss腦>˖*�        )��P	�h�@���A�*

	conv_loss0��>����        )��P	w��@���A�*

	conv_loss�V�>s���        )��P	~ʽ@���A�*

	conv_lossnm�>�I�+        )��P	���@���A�*

	conv_loss��>ȧB        )��P	;.�@���A�*

	conv_loss}!�>e.ϛ        )��P	�f�@���A�*

	conv_loss:A�>g��	        )��P	̙�@���A�*

	conv_loss���>rgF�        )��P	Gʾ@���A�*

	conv_lossJ*�>�x�_        )��P	d��@���A�*

	conv_loss=��>դ7�        )��P	.�@���A�*

	conv_loss��>}3�        )��P	�l�@���A�*

	conv_loss�@�>4��j        )��P	T��@���A�*

	conv_loss�ɧ>\�*�        )��P	��@���A�*

	conv_loss���>-��e        )��P	o�@���A�*

	conv_loss7�>qW��        )��P	�A�@���A�*

	conv_lossե�>�%l        )��P	���@���A�*

	conv_loss��>�X��        )��P	��@���A�*

	conv_loss���>�u7�        )��P	u��@���A�*

	conv_lossl��>gJ�*        )��P	d�@���A�*

	conv_loss�E�>��P        )��P	�J�@���A�*

	conv_lossx��>�F��        )��P	���@���A�*

	conv_loss̧>(�        )��P	���@���A�*

	conv_lossu(�>8a�
        )��P	���@���A�*

	conv_loss�a�>���        )��P	_-�@���A�*

	conv_lossi��>"8��        )��P	�e�@���A�*

	conv_lossJV�>u��        )��P	W��@���A�*

	conv_lossqE�>�� �        )��P	���@���A�*

	conv_lossV��>J|�z        )��P	U��@���A�*

	conv_loss�N�>�e5        )��P	�&�@���A�*

	conv_loss�ʨ>Osė        )��P	�X�@���A�*

	conv_loss�>�2�        )��P	��@���A�*

	conv_loss�>=�H        )��P	r��@���A�*

	conv_lossȥ>e�/a        )��P	g��@���A�*

	conv_loss���>`��        )��P	�2�@���A�*

	conv_lossa�>���        )��P	�h�@���A�*

	conv_loss�X�>�tJ-        )��P	׫�@���A�*

	conv_loss��>L�Z�        )��P	3��@���A�*

	conv_loss�%�>5��        )��P	
�@���A�*

	conv_loss�g�>uȱ�        )��P	�R�@���A�*

	conv_lossd-�>�vc        )��P	���@���A�*

	conv_lossC,�>�F�        )��P	���@���A�*

	conv_loss�q�>�O�v        )��P	���@���A�*

	conv_losso��>;Н        )��P	�.�@���A�*

	conv_loss�>��        )��P	�a�@���A�*

	conv_loss��>ό�        )��P	���@���A�*

	conv_lossp>�>�"	�        )��P	O��@���A�*

	conv_loss2��>��R        )��P	��@���A�*

	conv_loss���>`���        )��P	�P�@���A�*

	conv_loss!�>�+        )��P	q��@���A�*

	conv_loss�$�>G��        )��P	���@���A�*

	conv_loss�
�>�&�        )��P	>��@���A�*

	conv_loss��>:�M        )��P	�.�@���A�*

	conv_loss��>��@i        )��P	7c�@���A�*

	conv_loss�r�>� �m        )��P	��@���A�*

	conv_loss�W�>�"�        )��P	(��@���A�*

	conv_loss\�>ة&�        )��P	��@���A�*

	conv_loss���>��        )��P	?6�@���A�*

	conv_loss�Ҥ>���        )��P	�h�@���A�*

	conv_loss�a�>�         )��P	���@���A�*

	conv_loss���>I�G�        )��P	���@���A�*

	conv_lossYp�>1�x�        )��P	��@���A�*

	conv_loss���>�Vt�        )��P	2@�@���A�*

	conv_losszd�>۠��        )��P	hv�@���A�*

	conv_loss-5�>$I�        )��P	4��@���A�*

	conv_loss���>��g        )��P	���@���A�*

	conv_loss�D�>�^��        )��P	b�@���A�*

	conv_lossMӤ>�R         )��P	D�@���A�*

	conv_loss\x�>�e        )��P	?v�@���A�*

	conv_loss��>���        )��P	c��@���A�*

	conv_loss虤>�w�        )��P	>��@���A�*

	conv_lossv�>Ұ�        )��P	T�@���A�*

	conv_loss��>��.        )��P	�Q�@���A�*

	conv_losss&�>�]��        )��P	���@���A�*

	conv_lossN�>[�E�        )��P	��@���A�*

	conv_loss�L�>f�        )��P	%��@���A�*

	conv_loss$H�>t���        )��P	)1�@���A�*

	conv_loss��>�O]        )��P	Ef�@���A�*

	conv_loss���>��k        )��P	��@���A�*

	conv_lossL��>�`        )��P	��@���A�*

	conv_loss�K�>��        )��P	�@���A�*

	conv_lossP�>���        )��P	(M�@���A�*

	conv_loss��> X
E        )��P	6��@���A�*

	conv_lossCզ>�^&t        )��P	g��@���A�*

	conv_lossL��>�4L�        )��P	4�@���A�*

	conv_lossZ|�>�5        )��P	/7�@���A�*

	conv_loss�Ф>�&�        )��P	�i�@���A�*

	conv_loss�ԣ>lU�        )��P	2��@���A�*

	conv_loss68�>a�>l        )��P	���@���A�*

	conv_loss���>�mB        )��P	��@���A�*

	conv_loss"3�>w��        )��P	�@�@���A�*

	conv_loss�'�>�m�h        )��P	�q�@���A�*

	conv_loss��>`�2�        )��P	9��@���A�*

	conv_lossQ�>I�C        )��P	���@���A�*

	conv_loss]F�>�qJ        )��P	L�@���A�*

	conv_lossk�>hގ*        )��P	�Q�@���A�*

	conv_loss��>�/I        )��P	i��@���A�*

	conv_lossRY�>ֿ��        )��P	�H�@���A�*

	conv_loss���>z�?        )��P	�|�@���A�*

	conv_lossa��>�G        )��P	���@���A�*

	conv_lossL��>��W        )��P	���@���A�*

	conv_loss��>�R��        )��P	� �@���A�*

	conv_lossʿ�>�iד        )��P	<^�@���A�*

	conv_loss=d�>��v        )��P	W��@���A�*

	conv_loss}^�>(bv�        )��P	1��@���A�*

	conv_lossg�>F:lP        )��P	���@���A�*

	conv_loss��>еl        )��P	�,�@���A�*

	conv_loss-��>�7*X        )��P	>h�@���A�*

	conv_loss-�>�}o�        )��P	��@���A�*

	conv_loss�p�>�XTy        )��P	���@���A�*

	conv_loss���>R�        )��P	���@���A�*

	conv_lossgh�>�o0�        )��P	x.�@���A�*

	conv_lossn��>�#*�        )��P	�m�@���A�*

	conv_loss?@�>iYa|        )��P	\��@���A�*

	conv_loss�1�>�v��        )��P	���@���A�*

	conv_loss�X�>�_�        )��P	��@���A�*

	conv_loss0��>#�*x        )��P	�1�@���A�*

	conv_loss'
�>Q�n4        )��P	�d�@���A�*

	conv_lossbI�>Zg         )��P	��@���A�*

	conv_loss�]�>��>        )��P	���@���A�*

	conv_lossb��>����        )��P	A��@���A�*

	conv_loss���>�6�)        )��P	�%�@���A�*

	conv_lossL�>n�        )��P	T�@���A�*

	conv_loss��>�,�        )��P	ل�@���A�*

	conv_loss��>֬G�        )��P	��@���A�*

	conv_loss���>�pe�        )��P	���@���A�*

	conv_loss;ڡ>���        )��P	�#�@���A�*

	conv_loss٢>���        )��P	U�@���A�*

	conv_lossdV�>}�/�        )��P	F��@���A�*

	conv_loss���>�'�        )��P	^��@���A�*

	conv_loss��>�k��        )��P	���@���A�*

	conv_lossA��>��c        )��P	
�@���A�*

	conv_loss�>n�u        )��P	|G�@���A�*

	conv_loss���>	�)�        )��P	|u�@���A�*

	conv_lossP��>v��        )��P	���@���A�*

	conv_loss�y�>���        )��P	���@���A�*

	conv_loss���>R^�$        )��P	U�@���A�*

	conv_lossWX�>�a        )��P	�5�@���A�*

	conv_loss�ޡ>(<�g        )��P	�e�@���A�*

	conv_loss}�>M��        )��P	��@���A�*

	conv_lossJj�>���        )��P	���@���A�*

	conv_loss7G�>A�k�        )��P	���@���A�*

	conv_lossۡ�>���        )��P	�&�@���A�*

	conv_loss"�>c��        )��P	�W�@���A�*

	conv_loss��>���\        )��P	���@���A�*

	conv_lossU��>T��7        )��P	��@���A�*

	conv_lossX}�>��d        )��P	���@���A�*

	conv_loss���>J��        )��P	�!�@���A�*

	conv_loss�n�>1lK         )��P	�P�@���A�*

	conv_loss�A�>��        )��P	L��@���A�*

	conv_lossѓ�>@�h        )��P	C��@���A�*

	conv_lossJP�>?E�        )��P	%��@���A�*

	conv_loss��>or�        )��P	��@���A�*

	conv_loss�4�>��u        )��P	�B�@���A�*

	conv_loss��>M�        )��P	<r�@���A�*

	conv_lossJ��>��,3        )��P	Ԩ�@���A�*

	conv_lossu@�>���        )��P	,��@���A�*

	conv_lossW�>.[L�        )��P	��@���A�*

	conv_losswf�>Z�$        )��P	�F�@���A�*

	conv_loss�S�>��.k        )��P	Yx�@���A�*

	conv_loss�@�>!��        )��P	��@���A�*

	conv_loss9s�>�0�        )��P	_��@���A�*

	conv_loss	��>��`�        )��P	5�@���A�*

	conv_loss���>D��p        )��P	�U�@���A�*

	conv_loss��>���t        )��P	d��@���A�*

	conv_loss�Ţ>����        )��P	���@���A�*

	conv_loss�>�U�@        )��P	���@���A�*

	conv_loss�>���N        )��P	�-�@���A�*

	conv_loss�h�>�(-�        )��P	)]�@���A�*

	conv_loss�7�>�b2�        )��P	6��@���A�*

	conv_loss���>�N��        )��P	5��@���A�*

	conv_lossd8�>�8@        )��P	���@���A�*

	conv_loss�Ӡ> x̖        )��P	,�@���A�*

	conv_loss�D�>_��        )��P	�_�@���A�*

	conv_loss�
�>�ʰ        )��P	'��@���A�*

	conv_loss}��>�+*�        )��P	@��@���A�*

	conv_lossb��>{���        )��P	N�@���A�*

	conv_loss3d�>V*qb        )��P	�G�@���A�*

	conv_loss�^�>`�W�        )��P	J|�@���A�*

	conv_loss�(�>��xD        )��P	���@���A�*

	conv_loss�1�> �1        )��P	���@���A�*

	conv_loss.Q�>���        )��P	X%�@���A�*

	conv_lossm�>+ZD        )��P	�W�@���A�*

	conv_lossSG�>��B        )��P	���@���A�*

	conv_lossŦ�>(g��        )��P	���@���A�*

	conv_loss-R�>�=��        )��P	���@���A�*

	conv_loss�ݟ>)�'�        )��P	�?�@���A�*

	conv_lossCӡ>a���        )��P	p�@���A�*

	conv_lossK�>y �        )��P	���@���A�*

	conv_loss�y�>z}�v        )��P	��@���A�*

	conv_loss�"�>�0��        )��P	y/�@���A�*

	conv_loss�C�>�u��        )��P	�h�@���A�*

	conv_loss�}�>C���        )��P	���@���A�*

	conv_loss$h�>���        )��P	N��@���A�*

	conv_loss��>0�ge        )��P	��@���A�*

	conv_lossH�>dq8X        )��P	$:�@���A�*

	conv_lossi١>�l�        )��P	�m�@���A�*

	conv_loss�>&R�        )��P	���@���A�*

	conv_loss?�>x/�        )��P	-��@���A�*

	conv_loss'�>Gƨ@        )��P	�4�@���A�*

	conv_loss�_�>����        )��P	�h�@���A�*

	conv_loss��>��K�        )��P	3��@���A�*

	conv_lossĉ�>jx
�        )��P	���@���A�*

	conv_lossB�>�(��        )��P	��@���A�*

	conv_loss�%�> R�        )��P	�I�@���A�*

	conv_loss�W�>rl%�        )��P	V��@���A�*

	conv_loss�L�>n�        )��P	���@���A�*

	conv_loss'ݞ>��ң        )��P	���@���A�*

	conv_loss�K�>�s�        )��P	��@���A�*

	conv_lossx�>4���        )��P	;K�@���A�*

	conv_loss-��>:^5!        )��P	'��@���A�*

	conv_loss�̜>�x�        )��P	K��@���A�*

	conv_lossr؝>G\e�        )��P	��@���A�*

	conv_lossP��>P��Q        )��P	t,�@���A�*

	conv_loss���>��H        )��P	�Y�@���A�*

	conv_lossʹ�>����        )��P	A��@���A�*

	conv_lossĝ>,�Q        )��P	X��@���A�*

	conv_loss���>��34        )��P	:��@���A�*

	conv_loss�Ν>��        )��P	+�@���A�*

	conv_lossC)�>�`��        )��P	4Z�@���A�*

	conv_loss�.�>��M        )��P	U��@���A�*

	conv_loss ֜>?�Ϥ        )��P	��@���A�*

	conv_loss�<�>a7�q        )��P	$��@���A�*

	conv_lossĝ>�Ѡ        )��P	z)�@���A�*

	conv_loss���>��r        )��P	c]�@���A�*

	conv_loss1�>J��;        )��P	{��@���A�*

	conv_loss���>�O}Q        )��P	<��@���A�*

	conv_loss��>�ĳ�        )��P	_��@���A�*

	conv_lossX$�>� !        )��P	L4�@���A�*

	conv_loss��>��DG        )��P	�f�@���A�*

	conv_loss�z�>C	�        )��P	��@���A�*

	conv_loss†>@J��        )��P	���@���A�*

	conv_loss6Ĝ>k�a        )��P	�@���A�*

	conv_loss�@�>{��        )��P	8<�@���A�*

	conv_loss�,�>Y�h�        )��P	�l�@���A�*

	conv_lossLٜ>�.��        )��P	��@���A�*

	conv_loss�ם>����        )��P	���@���A�*

	conv_loss�̞>h!�        )��P	��@���A�*

	conv_loss:�>��        )��P	�B�@���A�*

	conv_loss{��>�e        )��P	;t�@���A�*

	conv_loss�њ>E��        )��P	��@���A�*

	conv_loss�>m�4z        )��P	���@���A�*

	conv_lossa~�>)�p�        )��P	��@���A�*

	conv_lossN�>�j�        )��P	M�@���A�*

	conv_loss�>��
2        )��P	4}�@���A�*

	conv_loss���>�2�        )��P	��@���A�*

	conv_loss��>�a        )��P	���@���A�*

	conv_loss7S�>O��D        )��P	C�@���A�*

	conv_lossOl�>'"�        )��P	�t�@���A�*

	conv_lossy,�>����        )��P	X��@���A�*

	conv_lossW͚>�An�        )��P	���@���A�*

	conv_loss�M�>�!#        )��P	��@���A�*

	conv_loss��>�)�Z        )��P	cJ�@���A�*

	conv_loss_�>-q|�        )��P	%}�@���A�*

	conv_loss3t�>$�mF        )��P	N��@���A�*

	conv_loss�Ǜ>�;J        )��P	���@���A�*

	conv_loss5��>`$X        )��P	�&�@���A�*

	conv_loss�ߝ>���        )��P	�Z�@���A�*

	conv_loss{�>>���        )��P	H��@���A�*

	conv_loss�(�>-e�        )��P		��@���A�*

	conv_lossm�>t��        )��P	���@���A�*

	conv_lossK̛>1��        )��P	p0�@���A�*

	conv_loss_�>��o�        )��P	g�@���A�*

	conv_loss��>%l#�        )��P	���@���A�*

	conv_loss�>�>�w�        )��P	���@���A�*

	conv_loss;��>�l�        )��P	s��@���A�*

	conv_losss9�>f�{�        )��P	�;�@���A�*

	conv_loss���>���         )��P	>n�@���A�*

	conv_lossZ�>G,sR        )��P	ޟ�@���A�*

	conv_loss��>2(U        )��P	���@���A�*

	conv_losss��>���d        )��P	�@���A�*

	conv_lossw�>�S[        )��P	B;�@���A�*

	conv_loss���>���*        )��P	ju�@���A�*

	conv_lossXX�>�U�        )��P	���@���A�*

	conv_loss'�>HS�        )��P	��@���A�*

	conv_loss�9�>B.0@        )��P	��@���A�*

	conv_losse �>��Gw        )��P	�E�@���A�*

	conv_loss0l�>&(��        )��P	�y�@���A�*

	conv_loss
�>����        )��P	���@���A�*

	conv_lossM�>}x�1        )��P	��@���A�*

	conv_loss�ǚ> m�        )��P	��@���A�*

	conv_loss�Ș>�5%        )��P	�?�@���A�*

	conv_lossoM�>XU�Q        )��P	�u�@���A�*

	conv_lossu×>/̞�        )��P	���@���A�*

	conv_lossⳛ>��0�        )��P	5��@���A�*

	conv_lossL��>�+@7        )��P	A�@���A�*

	conv_loss�L�>��g�        )��P	SF�@���A�*

	conv_lossԜ�>7�>        )��P	9}�@���A�*

	conv_loss�V�>��<�        )��P	���@���A�*

	conv_loss﫛>���        )��P	��@���A�*

	conv_loss'L�>]�        )��P	:�@���A�*

	conv_loss�n�>� �        )��P	�C�@���A�*

	conv_loss���>k�L"        )��P	o��@���A�*

	conv_loss8ؗ>l�=�        )��P	���@���A�*

	conv_loss�[�>�}��        )��P	`
�@���A�*

	conv_loss��>	S%*        )��P	�<�@���A�*

	conv_lossj?�>��ή        )��P	vA���A�*

	conv_loss�@�>���        )��P	��A���A�*

	conv_loss!4�>��z        )��P	�A���A�*

	conv_lossÐ�>_·        )��P	9A���A�*

	conv_losss�>y��        )��P	nA���A�*

	conv_loss�}�>C��        )��P	��A���A�*

	conv_loss�~�>�;        )��P	��A���A�*

	conv_loss}	�>�:�*        )��P	)+A���A�*

	conv_lossȕ>v!A        )��P	MdA���A�*

	conv_lossP�>�˸        )��P	��A���A�*

	conv_lossV�>�D�        )��P	��A���A�*

	conv_lossew�>��        )��P	�A���A�*

	conv_loss��>Z��v        )��P	�8A���A�*

	conv_loss�є>Wa��        )��P	kA���A�*

	conv_loss���>���d        )��P	�A���A�*

	conv_loss�h�>��D        )��P	��A���A�*

	conv_loss�w�>x/�        )��P	�A���A�*

	conv_loss>�>�A�#        )��P	�EA���A�*

	conv_loss��>�vH        )��P	��A���A�*

	conv_loss|�>�;R�        )��P	�A���A�*

	conv_loss���>"D�        )��P	+�A���A�*

	conv_loss��>T��V        )��P	�$A���A�*

	conv_loss�ϒ>ID.        )��P	�WA���A�*

	conv_lossG�>'#|�        )��P	n�A���A�*

	conv_lossJ�>&���        )��P	y�A���A�*

	conv_loss���>�>W�        )��P	5�A���A�*

	conv_loss)��>@�Kt        )��P	A/A���A�*

	conv_loss쿓>@�
�        )��P	waA���A�*

	conv_loss~=�>�	��        )��P	��A���A�*

	conv_loss���>�+��        )��P	�A���A�*

	conv_loss/��>v4�        )��P	�	A���A�*

	conv_lossZ��>
��g        )��P	`;	A���A�*

	conv_loss�!�>� )`        )��P	�r	A���A�*

	conv_loss�|�>0�K        )��P	!�	A���A�*

	conv_lossN.�>Ѭ!�        )��P	=�	A���A�*

	conv_loss�X�>�g��        )��P	w
A���A�*

	conv_loss��>yR�t        )��P	H5
A���A�*

	conv_loss7+�>x��|        )��P	�e
A���A�*

	conv_loss_�>�a��        )��P	�
A���A�*

	conv_loss�f�>bH��        )��P	7�
A���A�*

	conv_lossn�>��V�        )��P	BA���A�*

	conv_loss6G�>�� �        )��P	]7A���A�*

	conv_loss=�>j��        )��P	�rA���A�*

	conv_loss��>��t�        )��P	�A���A�*

	conv_lossP�>$T7        )��P	�A���A�*

	conv_loss�q�>��s�        )��P	A���A�*

	conv_loss�	�>8vO        )��P	m7A���A�*

	conv_loss�d�>ت        )��P	�iA���A�*

	conv_loss�H�>�s��        )��P	��A���A�*

	conv_loss���>^U<5        )��P	
�A���A�*

	conv_lossyǒ>v{�        )��P	�A���A�*

	conv_loss�&�>f'�K        )��P	�WA���A�*

	conv_loss��>�B�U        )��P	M�A���A�*

	conv_loss:2�>���        )��P	��A���A�*

	conv_loss���>��&y        )��P	��A���A�*

	conv_loss*�>�@        )��P	}5A���A�*

	conv_loss;��>Y�;        )��P	<gA���A�*

	conv_lossv��> �1        )��P	H�A���A�*

	conv_loss�݉>�@Q�        )��P	��A���A�*

	conv_lossu�>H�`m        )��P	ZA���A�*

	conv_lossX��>���#        )��P	>A���A�*

	conv_loss�܎>�"        )��P	?oA���A�*

	conv_loss��>p�z�        )��P	5�A���A�*

	conv_loss���>���        )��P	��A���A�*

	conv_loss���>��=        )��P	�A���A�*

	conv_loss���> ��        )��P	DA���A�*

	conv_lossH�>��"        )��P	H�A���A�*

	conv_loss���>it�        )��P	R�A���A�*

	conv_loss���>$ө�        )��P	��A���A�*

	conv_loss>[�>�B�1        )��P	�*A���A�*

	conv_loss�>���        )��P	-aA���A�*

	conv_loss�>U,�        )��P	��A���A�*

	conv_loss6��>I�"X        )��P	��A���A�*

	conv_loss�P�>FU��        )��P	��A���A�*

	conv_loss()�>� �        )��P	p5A���A�*

	conv_lossX�>����        )��P	�hA���A�*

	conv_loss�È>�3$        )��P	�A���A�*

	conv_loss�D�>��        )��P	�A���A�*

	conv_loss[��>�L�        )��P	A���A�*

	conv_lossr,�>�Y�        )��P	>A���A�*

	conv_lossv��>��G        )��P	�pA���A�*

	conv_loss�>�-��        )��P	ңA���A�*

	conv_loss?�>�Nz�        )��P	��A���A�*

	conv_loss��>�ڞ8        )��P	�A���A�*

	conv_lossd&�>� ��        )��P	M@A���A�*

	conv_loss�ʉ>|�o        )��P	uA���A�*

	conv_loss�|�>�+        )��P	�A���A�*

	conv_loss���>"        )��P	��A���A�*

	conv_loss���>#��        )��P	^A���A�*

	conv_loss0�>eJ�U        )��P	o@A���A�*

	conv_loss"�>��,�        )��P	PxA���A�*

	conv_lossZ��>]�        )��P	��A���A�*

	conv_loss�1�>YN�        )��P	�A���A�*

	conv_lossF�><,w<        )��P	9A���A�*

	conv_loss2�>g��        )��P	FA���A�*

	conv_loss{�>�l/H        )��P	�vA���A�*

	conv_lossq{�>JU�        )��P	��A���A�*

	conv_loss:��>����        )��P	��A���A�*

	conv_loss�>_��e        )��P	A���A�*

	conv_lossւ>ʆ�        )��P	�RA���A�*

	conv_loss���>��        )��P	A�A���A�*

	conv_loss�G�>?%�        )��P	0�A���A�*

	conv_lossO��> �i        )��P	7 A���A�*

	conv_loss_X�>S�C;        )��P	87A���A�*

	conv_lossd��>���	        )��P	�tA���A�*

	conv_loss:��>��1b        )��P	��A���A�*

	conv_loss��>wޘ        )��P	C�A���A�*

	conv_loss;�>>�        )��P	7A���A�*

	conv_loss��>R��8        )��P	):A���A�*

	conv_loss���>��        )��P	�tA���A�*

	conv_loss�z>@��        )��P	��A���A�*

	conv_lossV��>ƙ        )��P	��A���A�*

	conv_loss�҅>
NJ�        )��P	4A���A�*

	conv_lossz�|>%�k        )��P	>BA���A�*

	conv_lossJZ�>�ç�        )��P	��A���A�*

	conv_lossŽ�>&UM        )��P	s�A���A�*

	conv_loss)>LHU        )��P	��A���A�*

	conv_loss#�y>OR�N        )��P	Z-A���A�*

	conv_loss�U}>�~        )��P	�jA���A�*

	conv_loss]Fw>�l?�        )��P	��A���A�*

	conv_lossE�x>��[�        )��P	��A���A�*

	conv_losssr>I�n"        )��P	�A���A�*

	conv_loss��~>��L        )��P	�9A���A�*

	conv_loss�Xs>(�B        )��P	�yA���A�*

	conv_loss�Ls>��        )��P	+�A���A�*

	conv_lossO�q>;Eˌ        )��P	-�A���A�*

	conv_loss4|{>8GB�        )��P	A���A�*

	conv_loss!Sq>�'�        )��P	�IA���A�*

	conv_lossh�v>4�C-        )��P	�A���A�*

	conv_loss�o>��        )��P	��A���A�*

	conv_lossV<q>�(��        )��P	��A���A�*

	conv_loss�F�>At�        )��P	A���A�*

	conv_loss'�p>�/�        )��P	JA���A�*

	conv_loss��f>��        )��P	�A���A�*

	conv_loss0�o>�6]�        )��P	��A���A�*

	conv_loss�v>}��        )��P	�A���A�*

	conv_loss}.n>N�Bf        )��P	`A���A�*

	conv_lossn�r>B���        )��P	�IA���A�*

	conv_loss�#t>,ؾ        )��P	x}A���A�*

	conv_loss�ty>�#)y        )��P	c�A���A�*

	conv_loss�7t>�Z         )��P	��A���A�*

	conv_loss��h>� �        )��P	/ A���A�*

	conv_loss�ku>��>        )��P	�a A���A�*

	conv_loss�o>~�j        )��P	�� A���A�*

	conv_loss�Vj>SR        )��P	�� A���A�*

	conv_lossQ�m>	��.        )��P	�� A���A�*

	conv_lossri>��        )��P	�)!A���A�*

	conv_loss�_>~�5o        )��P	&]!A���A�*

	conv_lossfZl>�-�        )��P	i�!A���A�*

	conv_loss��e>	�)�        )��P	r�!A���A�*

	conv_loss]>�٨�        )��P	� "A���A�*

	conv_loss�o>Ol�        )��P	�I"A���A�*

	conv_loss�_> ]�y        )��P	8}"A���A�	*

	conv_loss�d>_(\        )��P	G�"A���A�	*

	conv_lossX�_>A�E        )��P	+�"A���A�	*

	conv_loss��Z>I��-        )��P	#A���A�	*

	conv_loss�9f>�ﶓ        )��P	{D#A���A�	*

	conv_loss��j>#�        )��P	v#A���A�	*

	conv_lossna>����        )��P	��#A���A�	*

	conv_loss�U>+�܌        )��P	��#A���A�	*

	conv_loss�*d>�xc        )��P	s$A���A�	*

	conv_loss��S>���        )��P	�Q$A���A�	*

	conv_loss�b>Q��z        )��P	��$A���A�	*

	conv_loss�<g>�s�        )��P	i�$A���A�	*

	conv_loss��S>(+�4        )��P	��$A���A�	*

	conv_loss�yQ>���%        )��P	�2%A���A�	*

	conv_lossÝ\> _��        )��P	�c%A���A�	*

	conv_loss�[>յa        )��P	I�%A���A�	*

	conv_loss�O>�1�8        )��P	��%A���A�	*

	conv_lossR�i>�Ѧ        )��P	�&A���A�	*

	conv_loss%�]>��	|        )��P	L9&A���A�	*

	conv_loss�O> �j@        )��P	hi&A���A�	*

	conv_lossl�Q>g� x        )��P	a�&A���A�	*

	conv_loss`[R>�C�        )��P	f�&A���A�	*

	conv_loss�/^>9���        )��P	/�&A���A�	*

	conv_lossS�O>��s2        )��P	D)'A���A�	*

	conv_loss��G>姴�        )��P	wh'A���A�	*

	conv_loss�nO>���        )��P	f�'A���A�	*

	conv_loss@ab>�@Ů        )��P	a�'A���A�	*

	conv_lossͲ[>�^&        )��P	�(A���A�	*

	conv_loss,�S>a4?�        )��P	9=(A���A�	*

	conv_loss&�V>F�!�        )��P	Xo(A���A�	*

	conv_loss�^>��:        )��P	��(A���A�	*

	conv_loss>�R>���        )��P	��(A���A�	*

	conv_loss��V>Y�F�        )��P	�)A���A�	*

	conv_lossMMU>u`%        )��P	�2)A���A�	*

	conv_loss�M>[��(        )��P	f)A���A�	*

	conv_loss�c>pz�        )��P	��)A���A�	*

	conv_lossTU>��X�        )��P	��)A���A�	*

	conv_lossޔG>aㄭ        )��P	"*A���A�	*

	conv_loss��J>���1        )��P	>H*A���A�	*

	conv_loss�iY>}�
�        )��P	�~*A���A�	*

	conv_loss�@>��*        )��P	��*A���A�	*

	conv_loss��F>:��(        )��P	��*A���A�	*

	conv_loss{�f>���F        )��P	2+A���A�	*

	conv_lossQ�T>_�4        )��P	�J+A���A�	*

	conv_lossA�G> YI        )��P	�z+A���A�	*

	conv_lossWGI>p��        )��P	*�+A���A�	*

	conv_loss�z@>x��        )��P	��+A���A�	*

	conv_lossHE>r���        )��P	�,A���A�	*

	conv_loss,J>J<��        )��P	�?,A���A�	*

	conv_lossR@W>��
�        )��P	;�-A���A�	*

	conv_loss�CN>��        )��P	�'.A���A�	*

	conv_lossctH>���        )��P	[.A���A�	*

	conv_loss`W?>���h        )��P	�.A���A�	*

	conv_loss� =>ހ%�        )��P	^�.A���A�	*

	conv_loss�7X>��i        )��P	/A���A�	*

	conv_loss=�R>\�]        )��P	�E/A���A�	*

	conv_loss��>>�$        )��P	v/A���A�	*

	conv_lossyAD>�z�V        )��P	>�/A���A�	*

	conv_loss�H>e���        )��P	��/A���A�	*

	conv_lossiP>�а         )��P	H0A���A�	*

	conv_loss@>0s�4        )��P	�M0A���A�	*

	conv_losshjI>U��0        )��P	|0A���A�	*

	conv_loss
�.>��Yj        )��P		�0A���A�	*

	conv_loss�3>G`�{        )��P	��0A���A�	*

	conv_loss��1>:��<        )��P	�1A���A�	*

	conv_lossHK6>7Xp�        )��P	�S1A���A�	*

	conv_loss�&:>��1�        )��P	d�1A���A�	*

	conv_loss[3;>eS��        )��P	��1A���A�	*

	conv_lossz�D>n�s*        )��P	M�1A���A�	*

	conv_loss�I>�h�>        )��P	�32A���A�	*

	conv_loss75K>q��        )��P	�e2A���A�	*

	conv_loss�8S>A� z        )��P	Ǘ2A���A�	*

	conv_lossu�\>~�ֹ        )��P	��2A���A�	*

	conv_loss�nK>�g&        )��P	��2A���A�	*

	conv_loss��9>�<z        )��P	�53A���A�	*

	conv_loss8A>1f�        )��P	:f3A���A�	*

	conv_loss}�?>�N��        )��P	��3A���A�	*

	conv_loss��3>���        )��P	�3A���A�	*

	conv_losseK3>�>^         )��P	�4A���A�	*

	conv_loss��<>����        )��P	�@4A���A�	*

	conv_loss�RJ>M>�>        )��P	 q4A���A�	*

	conv_loss�y4>ɾ�        )��P		�4A���A�	*

	conv_loss�1:>� ]P        )��P	�4A���A�	*

	conv_loss�5>3@         )��P	�5A���A�	*

	conv_lossf9>ViZ        )��P	�?5A���A�	*

	conv_loss�C>�{r�        )��P	�r5A���A�	*

	conv_loss�>>n�c        )��P	�5A���A�	*

	conv_loss!.B>n�d�        )��P	��5A���A�	*

	conv_loss�04>)=        )��P	-6A���A�	*

	conv_loss"�2>��ƨ        )��P	>6A���A�	*

	conv_loss�>1>ͥ��        )��P	tz6A���A�	*

	conv_lossa�'>���        )��P	q�6A���A�	*

	conv_loss�;I>�5�        )��P	��6A���A�	*

	conv_loss7>7��X        )��P	�7A���A�	*

	conv_loss3�?>!��	        )��P	�P7A���A�	*

	conv_lossFL>:��        )��P	Z�7A���A�	*

	conv_loss�N->y�66        )��P	��7A���A�	*

	conv_lossh�,>�u�        )��P	��7A���A�	*

	conv_losss/>8�9�        )��P	�&8A���A�	*

	conv_lossg�/>7���        )��P	/v8A���A�	*

	conv_lossʦ/>qBeb        )��P	¦8A���A�	*

	conv_loss^�(>�q��        )��P	��8A���A�	*

	conv_loss�q0>0���        )��P	e9A���A�	*

	conv_lossdB>�v        )��P	<Y9A���A�	*

	conv_loss��0>aM        )��P	�9A���A�	*

	conv_lossF�5>���        )��P	Ļ9A���A�	*

	conv_loss�25>t�V        )��P	��9A���A�	*

	conv_loss��+>0�        )��P	O:A���A�	*

	conv_loss��>�)֒        )��P	X:A���A�	*

	conv_loss��B>D$        )��P	Џ:A���A�	*

	conv_loss^�A>��o�        )��P	�:A���A�	*

	conv_loss?!#>�=7g        )��P	%�:A���A�	*

	conv_loss3)>@0G�        )��P	�;A���A�	*

	conv_loss�>n�S�        )��P	\^;A���A�	*

	conv_loss�;>-�        )��P	W�;A���A�	*

	conv_loss
0>��Z�        )��P	[�;A���A�	*

	conv_loss`�$>����        )��P	<A���A�	*

	conv_loss�}#>���        )��P	�:<A���A�	*

	conv_lossz1>A�x        )��P	]o<A���A�	*

	conv_lossr�>\=�1        )��P	p�<A���A�	*

	conv_loss �*>�:�,        )��P	c�<A���A�	*

	conv_lossXM>��CB        )��P	�=A���A�	*

	conv_loss�,>���        )��P	�:=A���A�	*

	conv_lossr	+>�h&:        )��P	�u=A���A�	*

	conv_loss�>��        )��P	��=A���A�	*

	conv_lossF^<>�=�        )��P	g�=A���A�	*

	conv_loss3c!>�ʵ�        )��P	� >A���A�	*

	conv_lossA >���Q        )��P	�U>A���A�	*

	conv_loss��>�w�        )��P	�>A���A�
*

	conv_loss0�%>+A �        )��P	ֿ>A���A�
*

	conv_loss�"&>���R        )��P	��>A���A�
*

	conv_loss�&3>��	[        )��P	�-?A���A�
*

	conv_loss�">=��        )��P	bp?A���A�
*

	conv_loss.�/>��"T        )��P	
�?A���A�
*

	conv_loss�*>�9        )��P	��?A���A�
*

	conv_loss�$>'"��        )��P	e@A���A�
*

	conv_loss��=>�`R        )��P	09@A���A�
*

	conv_loss�.A>��&        )��P	Av@A���A�
*

	conv_loss�->Gk��        )��P	q�@A���A�
*

	conv_loss�#>�'e        )��P	��@A���A�
*

	conv_loss�.>#{7        )��P	$%AA���A�
*

	conv_lossu�>����        )��P	�WAA���A�
*

	conv_lossЉ->Kw�t        )��P	��AA���A�
*

	conv_lossQ�>ΘH        )��P	ؽAA���A�
*

	conv_loss(;%>$Lq�        )��P	$�AA���A�
*

	conv_loss)�>���        )��P	�/BA���A�
*

	conv_loss@�$>a�u        )��P	�cBA���A�
*

	conv_loss�y">tPe        )��P	~�BA���A�
*

	conv_loss/"1>?i��        )��P	g�BA���A�
*

	conv_lossS�>EІ�        )��P	�&CA���A�
*

	conv_loss�!>��H�        )��P	�aCA���A�
*

	conv_loss;#>݂�C        )��P	Z�CA���A�
*

	conv_loss��">ǹ��        )��P	7�CA���A�
*

	conv_loss]n#>��u"        )��P	�DA���A�
*

	conv_loss�>�m��        )��P	iCDA���A�
*

	conv_lossD�7>.���        )��P	sDA���A�
*

	conv_lossAS>^,I        )��P	��DA���A�
*

	conv_lossc >�E$        )��P	��DA���A�
*

	conv_lossl�>��o        )��P	jEA���A�
*

	conv_loss��.>-JEk        )��P	�CEA���A�
*

	conv_loss}V%>Q�2�        )��P	�EA���A�
*

	conv_loss�'>�"��        )��P	��EA���A�
*

	conv_loss��>Qm�L        )��P	�FA���A�
*

	conv_lossy�=>s��b        )��P	�4FA���A�
*

	conv_lossJ�:>ↇI        )��P	oiFA���A�
*

	conv_loss�>3O�        )��P	8�FA���A�
*

	conv_loss٣>�H�        )��P	��FA���A�
*

	conv_loss��>MB^        )��P	�GA���A�
*

	conv_loss��">E�D        )��P	JBGA���A�
*

	conv_loss=t>/w�        )��P	�rGA���A�
*

	conv_lossO >�d�        )��P	X�GA���A�
*

	conv_lossC\>V��        )��P	�GA���A�
*

	conv_loss�)>���        )��P	�HA���A�
*

	conv_loss�(>u7z        )��P	
HHA���A�
*

	conv_loss��>��݅        )��P	]zHA���A�
*

	conv_loss��>�U�~        )��P	L�HA���A�
*

	conv_loss�>�i^        )��P	<�HA���A�
*

	conv_lossW�>��        )��P	�IA���A�
*

	conv_loss/�>j"�        )��P	�GIA���A�
*

	conv_loss�>�
�        )��P	�~IA���A�
*

	conv_loss߸$>T�v#        )��P	ʾIA���A�
*

	conv_loss=�	>�͒�        )��P	��IA���A�
*

	conv_loss6�>'         )��P	M&JA���A�
*

	conv_lossE�> ��\        )��P	6[JA���A�
*

	conv_loss�,>��>�        )��P	t�JA���A�
*

	conv_loss��B>��H�        )��P	�JA���A�
*

	conv_loss�1>,�]        )��P	V�JA���A�
*

	conv_loss��%>p�ad        )��P	�-KA���A�
*

	conv_loss��0>�n�]        )��P	�^KA���A�
*

	conv_loss��5>\���        )��P	ߓKA���A�
*

	conv_loss�=@>�׎        )��P	s�KA���A�
*

	conv_lossJ>�DSq        )��P	�LA���A�
*

	conv_loss��>�y        )��P	�5LA���A�
*

	conv_loss�f3>_I.        )��P	ufLA���A�
*

	conv_lossC�7>�+��        )��P	�LA���A�
*

	conv_loss �#>/��P        )��P	��LA���A�
*

	conv_loss��#>��&        )��P	L�LA���A�
*

	conv_loss��>�TQE        )��P	<)MA���A�
*

	conv_loss*>F�@        )��P	�^MA���A�
*

	conv_loss\>�E�        )��P	z�MA���A�
*

	conv_loss�>�I        )��P	��MA���A�
*

	conv_loss��>,a��        )��P	NA���A�
*

	conv_loss�>Xv��        )��P	jQNA���A�
*

	conv_loss��>%�}        )��P	�NA���A�
*

	conv_loss+V2>i]d�        )��P	�NA���A�
*

	conv_loss�=>�ĳs        )��P	��NA���A�
*

	conv_loss0V$>NH��        )��P	j!OA���A�
*

	conv_loss� >%J��        )��P	<VOA���A�
*

	conv_lossn>��         )��P	��OA���A�
*

	conv_loss��$>���        )��P	!�OA���A�
*

	conv_loss*��=/�l        )��P	�PA���A�
*

	conv_losszi>@��s        )��P	�3PA���A�
*

	conv_lossu>�r
�        )��P	kPA���A�
*

	conv_loss�->��|�        )��P	M�PA���A�
*

	conv_lossY>>���        )��P	��PA���A�
*

	conv_lossa>q��        )��P	CQA���A�
*

	conv_loss�>���6        )��P	!:QA���A�
*

	conv_loss�>���        )��P	�mQA���A�
*

	conv_lossR�>Y��O        )��P	��QA���A�
*

	conv_loss<>��O�        )��P	��QA���A�
*

	conv_loss�>�.        )��P	?RA���A�
*

	conv_lossJE>����        )��P	�ERA���A�
*

	conv_loss4�>H�5        )��P	�|RA���A�
*

	conv_loss15>���(        )��P	ڰRA���A�
*

	conv_loss�(>���        )��P	�RA���A�
*

	conv_loss�b>���9        )��P	?SA���A�
*

	conv_loss@�>f_d         )��P	'HSA���A�
*

	conv_loss��>NA�v        )��P	$wSA���A�
*

	conv_loss�>�@�        )��P	�SA���A�
*

	conv_loss�t(>$^��        )��P	��SA���A�
*

	conv_loss�W�=���9        )��P	�TA���A�
*

	conv_loss_9#>���m        )��P	DGTA���A�
*

	conv_loss]>�W�        )��P	�wTA���A�
*

	conv_loss�
>���        )��P	��TA���A�
*

	conv_lossG�>cq$�        )��P	l�TA���A�
*

	conv_loss�>�%��        )��P	7 UA���A�
*

	conv_lossJ�>��"4        )��P	WPUA���A�
*

	conv_loss�o >1�S�        )��P	9�UA���A�
*

	conv_lossm�>/�x�        )��P	s�UA���A�
*

	conv_lossO>L�        )��P	��UA���A�
*

	conv_lossXe>��F�        )��P	!VA���A�
*

	conv_loss_N>���        )��P	�YVA���A�
*

	conv_loss8�>;1�        )��P	��VA���A�
*

	conv_loss�Y >"�ҷ        )��P	�VA���A�
*

	conv_loss0
�=��_        )��P	��VA���A�
*

	conv_loss��>A]�        )��P	�WA���A�
*

	conv_loss1E$>ʵ�4        )��P	jYWA���A�
*

	conv_loss��>��c        )��P	��WA���A�
*

	conv_loss(
>'d�_        )��P	f�WA���A�
*

	conv_lossu>�mX        )��P	^cYA���A�
*

	conv_loss�B>�?_        )��P	ϖYA���A�
*

	conv_lossl�>���        )��P	��YA���A�
*

	conv_losss>ЯI�        )��P	`ZA���A�
*

	conv_loss94>��        )��P	AIZA���A�
*

	conv_loss)0�=���S        )��P	�ZA���A�
*

	conv_loss�}>̗u�        )��P	�ZA���A�
*

	conv_loss�0>f9�        )��P	��ZA���A�*

	conv_losss,>e�j        )��P	�[A���A�*

	conv_lossW�
>t��5        )��P	gV[A���A�*

	conv_loss�>�`5        )��P	��[A���A�*

	conv_loss���=y.��        )��P	��[A���A�*

	conv_loss~��=OOwy        )��P	�[A���A�*

	conv_loss�i>f� J        )��P	�&\A���A�*

	conv_loss��=���        )��P	/`\A���A�*

	conv_loss/*	>:�Eu        )��P	��\A���A�*

	conv_loss��>>}��        )��P	R�\A���A�*

	conv_loss>�>�{YP        )��P	N]A���A�*

	conv_loss��>8�l�        )��P	}>]A���A�*

	conv_loss�>�X)�        )��P	�t]A���A�*

	conv_loss���=?;�        )��P	:�]A���A�*

	conv_loss��=��B        )��P	.�]A���A�*

	conv_lossN>>�[k�        )��P	Z^A���A�*

	conv_loss�q>5hT        )��P	�C^A���A�*

	conv_loss��>bc        )��P	�|^A���A�*

	conv_lossyb>
(K        )��P	{�^A���A�*

	conv_loss�/>~��        )��P	��^A���A�*

	conv_loss��>M���        )��P	_A���A�*

	conv_lossK�>�k@        )��P	uG_A���A�*

	conv_loss�I>�2�u        )��P	d}_A���A�*

	conv_loss�>�        )��P	8�_A���A�*

	conv_loss��>u�
        )��P	��_A���A�*

	conv_loss��>��M        )��P	+`A���A�*

	conv_lossZ�>Bs�_        )��P	�N`A���A�*

	conv_lossb�>�tB�        )��P	O�`A���A�*

	conv_loss�>,��$        )��P	Z�`A���A�*

	conv_lossh�>�ӛU        )��P	��`A���A�*

	conv_loss�>T�6        )��P	^aA���A�*

	conv_loss�� >�B+        )��P	 NaA���A�*

	conv_loss��#>��|o        )��P	ӘaA���A�*

	conv_loss�_>14.�        )��P	^�aA���A�*

	conv_loss��>��ͅ        )��P	~bA���A�*

	conv_loss�.>���         )��P	\2bA���A�*

	conv_loss�:>��        )��P	�abA���A�*

	conv_loss�O!>�A�J        )��P	��bA���A�*

	conv_lossem>�A�=        )��P	^�bA���A�*

	conv_loss��=	�        )��P	 cA���A�*

	conv_loss�>p!��        )��P	b3cA���A�*

	conv_loss�v>k�m�        )��P	�ecA���A�*

	conv_lossZ�>�-la        )��P	ћcA���A�*

	conv_loss�D>6�?h        )��P	H�cA���A�*

	conv_loss�S>ҧ�        )��P	�dA���A�*

	conv_loss��>�
K        )��P	�OdA���A�*

	conv_lossR+>�\z7        )��P	Q�dA���A�*

	conv_lossՄ>���H        )��P	-�dA���A�*

	conv_lossE��=GbH�        )��P	u�dA���A�*

	conv_loss[ >m(2	        )��P	[eA���A�*

	conv_lossn�>Օ��        )��P	�=eA���A�*

	conv_lossY>���
        )��P	vweA���A�*

	conv_loss��>�EX�        )��P	�eA���A�*

	conv_loss�=�:9�        )��P	��eA���A�*

	conv_loss�>�KS3        )��P	fA���A�*

	conv_loss�>7x2        )��P	LfA���A�*

	conv_loss92>�[�        )��P	fA���A�*

	conv_loss�x�= �C.        )��P	l�fA���A�*

	conv_loss�>�&�        )��P	��fA���A�*

	conv_loss�3>�o        )��P	v*gA���A�*

	conv_loss$��=���        )��P	�\gA���A�*

	conv_loss��	>n_��        )��P	��gA���A�*

	conv_loss��>31�S        )��P	��gA���A�*

	conv_loss�l>�2�        )��P	%�gA���A�*

	conv_loss�	�=}�h�        )��P	-.hA���A�*

	conv_loss���=�7�X        )��P	�fhA���A�*

	conv_loss�>b��        )��P	 �hA���A�*

	conv_loss���=jҸ�        )��P	��hA���A�*

	conv_loss� >�)�B        )��P	�iA���A�*

	conv_lossv�>���        )��P	J>iA���A�*

	conv_loss�>f�A$        )��P	�viA���A�*

	conv_loss�v>���        )��P	��iA���A�*

	conv_loss�k>(���        )��P	��iA���A�*

	conv_loss�!>6�f;        )��P	jA���A�*

	conv_loss�h>�M�        )��P	"AjA���A�*

	conv_loss=>���        )��P	�qjA���A�*

	conv_lossr�>�O-�        )��P	��jA���A�*

	conv_loss�>��3�        )��P	��jA���A�*

	conv_loss~�>U���        )��P	kkA���A�*

	conv_lossZb>o� ;        )��P	�<kA���A�*

	conv_loss�w�=�i��        )��P	�pkA���A�*

	conv_loss���=@L�F        )��P	)�kA���A�*

	conv_loss�>Vɹ]        )��P	��kA���A�*

	conv_loss��>v\�        )��P	+lA���A�*

	conv_lossw�>E�m%        )��P	$`lA���A�*

	conv_loss@�	>�Iœ        )��P	i�lA���A�*

	conv_loss%�>]�p�        )��P	��lA���A�*

	conv_lossf�	>H"O�        )��P		mA���A�*

	conv_loss��>̵:k        )��P	BmA���A�*

	conv_lossh�>��{        )��P	�umA���A�*

	conv_loss��>ڕK�        )��P	�mA���A�*

	conv_loss��=�(�        )��P	��mA���A�*

	conv_loss��><��        )��P	�rA���A�*

	conv_loss�Z>e���        )��P	��rA���A�*

	conv_lossø">�*�_        )��P	�sA���A�*

	conv_loss�0>�B�        )��P	�7sA���A�*

	conv_lossg>ɭ@�        )��P	vsA���A�*

	conv_loss���=��        )��P	ĨsA���A�*

	conv_loss�c>C��U        )��P	U�sA���A�*

	conv_lossNk>-|z        )��P	YtA���A�*

	conv_loss��>Oh�        )��P	FtA���A�*

	conv_loss�>���7        )��P	^vtA���A�*

	conv_lossbD>:��        )��P	��tA���A�*

	conv_loss6��=+�i�        )��P	N�tA���A�*

	conv_loss�U>����        )��P	�uA���A�*

	conv_loss���=�)��        )��P	�BuA���A�*

	conv_lossK��='f?�        )��P	�suA���A�*

	conv_loss�>v��        )��P	��uA���A�*

	conv_loss;z�=���7        )��P	d�uA���A�*

	conv_loss�$>�"�+        )��P	�vA���A�*

	conv_lossW��=�        )��P	�AvA���A�*

	conv_loss4��=�WD�        )��P	spvA���A�*

	conv_loss�*�=��\        )��P	q�vA���A�*

	conv_loss�p�=<��1        )��P	�vA���A�*

	conv_loss�y�=+�Z�        )��P	��vA���A�*

	conv_lossz�>���        )��P	/wA���A�*

	conv_losss��=�[Ib        )��P	�]wA���A�*

	conv_lossE >�q��        )��P	@�wA���A�*

	conv_losso�=Lj?�        )��P	k�wA���A�*

	conv_loss�>����        )��P	H�wA���A�*

	conv_loss���=dS�&        )��P	�xA���A�*

	conv_loss���=d8�        )��P	�IxA���A�*

	conv_lossN�>�<��        )��P	֎xA���A�*

	conv_loss���=[CZ�        )��P	��xA���A�*

	conv_loss��>�PgN        )��P	i�xA���A�*

	conv_lossi��=u<�        )��P	{ yA���A�*

	conv_lossU >�X        )��P	;OyA���A�*

	conv_lossN�>@/�D        )��P	H}yA���A�*

	conv_loss_$�=�h        )��P	�yA���A�*

	conv_loss�%>{�s        )��P	��yA���A�*

	conv_loss�m>�?�T        )��P	�zA���A�*

	conv_lossm >U��b        )��P	(<zA���A�*

	conv_loss>4;C�        )��P	okzA���A�*

	conv_loss	Z>(%��        )��P	��zA���A�*

	conv_loss ��=a��        )��P	N�zA���A�*

	conv_losso�>m���        )��P	-{A���A�*

	conv_losscM>��        )��P	�={A���A�*

	conv_loss���=W�$G        )��P	<p{A���A�*

	conv_loss�4>H1`�        )��P	٤{A���A�*

	conv_loss���=�w        )��P	��{A���A�*

	conv_loss<��=Q�V        )��P	W	|A���A�*

	conv_loss��>R��        )��P	qK|A���A�*

	conv_loss��
>��        )��P	�}|A���A�*

	conv_lossL.>�qj        )��P	��|A���A�*

	conv_loss���=]�G�        )��P	��|A���A�*

	conv_loss��>��џ        )��P	�$}A���A�*

	conv_loss��=�9��        )��P	d}A���A�*

	conv_lossQP>���        )��P	|�}A���A�*

	conv_loss. >D{(        )��P	��}A���A�*

	conv_loss��>xop*        )��P	��}A���A�*

	conv_loss���=�AW@        )��P	�,~A���A�*

	conv_loss�=U-_P        )��P	 k~A���A�*

	conv_losse��=���|        )��P	�~A���A�*

	conv_loss�z�=��ˡ        )��P	G�~A���A�*

	conv_loss�=�P�        )��P	�A���A�*

	conv_loss8��=(��!        )��P	�MA���A�*

	conv_lossnA>wQ�        )��P	�A���A�*

	conv_lossR��=����        )��P	��A���A�*

	conv_loss���=c�I        )��P	��A���A�*

	conv_loss��>���d        )��P	#�A���A�*

	conv_loss��=![�U        )��P	V�A���A�*

	conv_loss��>��        )��P	X��A���A�*

	conv_loss^>��-�        )��P	���A���A�*

	conv_loss��=C��E        )��P	�A���A�*

	conv_loss�g�=��J        )��P	r�A���A�*

	conv_loss��="�EB        )��P	�Z�A���A�*

	conv_lossR��=	"        )��P	ь�A���A�*

	conv_lossʧ�=��0�        )��P	|��A���A�*

	conv_lossQ>����        )��P	E�A���A�*

	conv_loss�� >����        )��P	(�A���A�*

	conv_loss�r�=�vd�        )��P	�\�A���A�*

	conv_loss5I�=�T        )��P	���A���A�*

	conv_loss;>>7�        )��P	yɂA���A�*

	conv_losso">�}��        )��P	���A���A�*

	conv_lossx>�!�S        )��P	�1�A���A�*

	conv_lossX��=N&O2        )��P	�p�A���A�*

	conv_loss4��=��x        )��P	8��A���A�*

	conv_lossP7�=���        )��P	�ՃA���A�*

	conv_loss��=�P        )��P	^�A���A�*

	conv_loss9�=�tG�        )��P	�H�A���A�*

	conv_loss���=��s        )��P	�|�A���A�*

	conv_loss\�=�?DH        )��P	���A���A�*

	conv_lossBO>�;�        )��P	�ڄA���A�*

	conv_loss���=�\�        )��P	��A���A�*

	conv_lossV��=;��        )��P	gS�A���A�*

	conv_loss���=�Mޭ        )��P	J��A���A�*

	conv_loss@"	>g�{P        )��P	���A���A�*

	conv_loss�=#<E�        )��P	���A���A�*

	conv_loss>�	�        )��P	J(�A���A�*

	conv_lossAR>ȗ_�        )��P	�[�A���A�*

	conv_loss�h>Q���        )��P	͗�A���A�*

	conv_lossh>S��        )��P	�ʆA���A�*

	conv_loss~	>�`6j        )��P	���A���A�*

	conv_lossy
>�RI        )��P	=��A���A�*

	conv_loss��=��T�        )��P	q�A���A�*

	conv_loss<�>;���        )��P	��A���A�*

	conv_loss�c>	u6�        )��P	�O�A���A�*

	conv_loss�<->�C�        )��P	��A���A�*

	conv_losst`>��_L        )��P	��A���A�*

	conv_loss/��=��Y        )��P	^�A���A�*

	conv_loss��=��        )��P	�'�A���A�*

	conv_loss���=�q��        )��P	L^�A���A�*

	conv_loss��>W��        )��P	��A���A�*

	conv_lossR �=;q�0        )��P	�͊A���A�*

	conv_loss���=CD�_        )��P	���A���A�*

	conv_loss���=�G��        )��P	E1�A���A�*

	conv_lossY�=���        )��P	9a�A���A�*

	conv_loss���=����        )��P	���A���A�*

	conv_loss�@�=�M�K        )��P	ЋA���A�*

	conv_loss��=���U        )��P	���A���A�*

	conv_loss���=��<�        )��P	0�A���A�*

	conv_loss���=wO�        )��P	_�A���A�*

	conv_loss��=�tM        )��P	���A���A�*

	conv_loss��=��)q        )��P	�̌A���A�*

	conv_loss���=ٍ�        )��P	��A���A�*

	conv_loss$��=���        )��P	>D�A���A�*

	conv_loss:�>��e        )��P	-v�A���A�*

	conv_lossP��=�h�E        )��P	h��A���A�*

	conv_losss��=�y�        )��P	�ލA���A�*

	conv_lossD9�=����        )��P	C�A���A�*

	conv_loss8��=���        )��P	�D�A���A�*

	conv_loss^w�=G�ř        )��P	�w�A���A�*

	conv_lossB��=���i        )��P	L��A���A�*

	conv_loss�=��{�        )��P	2ݎA���A�*

	conv_loss��>�=�e        )��P	H�A���A�*

	conv_loss���=k)��        )��P	E�A���A�*

	conv_loss��=�E�p        )��P	�y�A���A�*

	conv_loss<�=���`        )��P	|��A���A�*

	conv_lossuT�=�Sg�        )��P	��A���A�*

	conv_loss�T�=�a�        )��P	D�A���A�*

	conv_loss�O�=/;-        )��P	U�A���A�*

	conv_loss��=	��        )��P	��A���A�*

	conv_losss��=3�L�        )��P	���A���A�*

	conv_loss�;�=O|��        )��P	"�A���A�*

	conv_losse��=��C�        )��P	��A���A�*

	conv_loss�P�=�abf        )��P	 P�A���A�*

	conv_loss}�=��L�        )��P	P��A���A�*

	conv_loss�>>���1        )��P	״�A���A�*

	conv_loss0 >OB�O        )��P	I��A���A�*

	conv_loss��=�t&%        )��P	(�A���A�*

	conv_loss��>9��        )��P	�Y�A���A�*

	conv_lossK�>f�/Q        )��P	Ȋ�A���A�*

	conv_lossT�>�iǺ        )��P	���A���A�*

	conv_loss��
>���        )��P	��A���A�*

	conv_losss��=����        )��P		@�A���A�*

	conv_loss���= X        )��P	&s�A���A�*

	conv_loss)�=Dӽ�        )��P	���A���A�*

	conv_loss���=bcb        )��P	�ޓA���A�*

	conv_loss�3�=vğ�        )��P	��A���A�*

	conv_loss���=����        )��P	�S�A���A�*

	conv_loss��=���u        )��P	M��A���A�*

	conv_loss%F�=���        )��P	wA���A�*

	conv_loss�"�=j�y        )��P	���A���A�*

	conv_loss۹�=�N�        )��P	a-�A���A�*

	conv_loss�j�=ݳ        )��P	�c�A���A�*

	conv_loss7��=��
        )��P	��A���A�*

	conv_loss�_�=���w        )��P	]ȕA���A�*

	conv_loss{�=
���        )��P	���A���A�*

	conv_loss��>���m        )��P	�5�A���A�*

	conv_lossԀ#>s h        )��P	Fx�A���A�*

	conv_loss�M�=�Sw        )��P	⼖A���A�*

	conv_losss�>�˲        )��P	��A���A�*

	conv_loss�{�=j� �        )��P	�1�A���A�*

	conv_loss���=���)        )��P	�l�A���A�*

	conv_loss���=��)�        )��P	���A���A�*

	conv_lossY��=�3v        )��P	zؗA���A�*

	conv_lossS�=ܡc        )��P	Z	�A���A�*

	conv_loss���=`m�        )��P	U<�A���A�*

	conv_loss���=����        )��P	z�A���A�*

	conv_loss�N�=��v        )��P	2��A���A�*

	conv_lossq��=��KC        )��P	��A���A�*

	conv_loss���=W��        )��P	��A���A�*

	conv_loss���=��j�        )��P	�W�A���A�*

	conv_loss�;�=�ۑ�        )��P	���A���A�*

	conv_loss��=��=h        )��P	;əA���A�*

	conv_loss�B�=��t        )��P	���A���A�*

	conv_loss�L�=���        )��P	�*�A���A�*

	conv_loss�"�=l��[        )��P	�^�A���A�*

	conv_loss}w�=}�??        )��P	z��A���A�*

	conv_loss*�>n�P        )��P	ߚA���A�*

	conv_loss���=w z�        )��P	 �A���A�*

	conv_loss���=����        )��P	�A�A���A�*

	conv_loss�Q�=9��A        )��P	u�A���A�*

	conv_loss���=���        )��P	���A���A�*

	conv_loss���=�ch�        )��P	g�A���A�*

	conv_loss#��=�{�3        )��P	�7�A���A�*

	conv_loss��=�L��        )��P	zp�A���A�*

	conv_loss�#>���        )��P	���A���A�*

	conv_loss��=0J�        )��P	��A���A�*

	conv_lossNz>��f�        )��P	n-�A���A�*

	conv_loss���=+(��        )��P	=\�A���A�*

	conv_loss��>����        )��P	]��A���A�*

	conv_loss��>Q��        )��P	�ʝA���A�*

	conv_loss��>�op_        )��P	�%�A���A�*

	conv_loss��>0{�         )��P	2g�A���A�*

	conv_loss�B>|=Q        )��P	���A���A�*

	conv_lossY�=�	�        )��P	jԞA���A�*

	conv_lossǄ�=��J        )��P	��A���A�*

	conv_losse�=���        )��P	�B�A���A�*

	conv_lossL�=9         )��P	Zu�A���A�*

	conv_loss�5�=��ۯ        )��P	~��A���A�*

	conv_loss��=~<�{        )��P	ޟA���A�*

	conv_loss%��=� �s        )��P	1�A���A�*

	conv_loss�s�=�ye�        )��P	�R�A���A�*

	conv_loss�=�{�        )��P	w��A���A�*

	conv_loss���=,T;V        )��P	���A���A�*

	conv_loss���=X�h1        )��P	���A���A�*

	conv_loss˞�=?��        )��P	�)�A���A�*

	conv_loss6�=I��        )��P	f�A���A�*

	conv_loss��=җ��        )��P	M��A���A�*

	conv_lossAF�=mxt
        )��P	GơA���A�*

	conv_loss[��=�k�[        )��P	���A���A�*

	conv_loss�.�=!���        )��P	�'�A���A�*

	conv_loss��>�&(�        )��P	�e�A���A�*

	conv_loss�&>[�        )��P	���A���A�*

	conv_lossaC�=Z@1\        )��P	�ĢA���A�*

	conv_loss�!> ���        )��P	m��A���A�*

	conv_loss���=�+�        )��P	(�A���A�*

	conv_loss���=)rPj        )��P	�p�A���A�*

	conv_loss��=S�        )��P	6��A���A�*

	conv_loss[��=g���        )��P	�ԣA���A�*

	conv_losswڿ=^+�g        )��P	��A���A�*

	conv_loss��=���\        )��P	�=�A���A�*

	conv_lossH��=��        )��P	j|�A���A�*

	conv_loss&�=TM��        )��P	-��A���A�*

	conv_loss��>�[��        )��P	?�A���A�*

	conv_loss�"�=��,        )��P	��A���A�*

	conv_loss�u	>���	        )��P	�M�A���A�*

	conv_loss7��=u�<N        )��P	E��A���A�*

	conv_lossJ�=wi         )��P	ǹ�A���A�*

	conv_loss?P>��\�        )��P	R��A���A�*

	conv_lossJ��=cE�        )��P	c)�A���A�*

	conv_loss���=��3�        )��P	�Z�A���A�*

	conv_loss�( >`O�        )��P	H��A���A�*

	conv_loss��>�        )��P	�ȦA���A�*

	conv_loss=l�=sD�        )��P	���A���A�*

	conv_loss��=����        )��P	/�A���A�*

	conv_lossp:�=��;@        )��P	8b�A���A�*

	conv_loss��=�r�        )��P	��A���A�*

	conv_loss_b�=d�ڳ        )��P	HܧA���A�*

	conv_loss�A�=��Z�        )��P	��A���A�*

	conv_lossї�=��b�        )��P	�A�A���A�*

	conv_loss�s�={��        )��P	���A���A�*

	conv_loss��=z"��        )��P	\ШA���A�*

	conv_loss�$�=��ؓ        )��P	��A���A�*

	conv_loss���=Ǫ��        )��P	�1�A���A�*

	conv_loss$�=1x�        )��P	(g�A���A�*

	conv_lossz]�=a?��        )��P	˚�A���A�*

	conv_loss�*>_U�        )��P	�˩A���A�*

	conv_loss�{�=����        )��P	]��A���A�*

	conv_loss�S�=(�Bb        )��P	n-�A���A�*

	conv_loss"��=���k        )��P	df�A���A�*

	conv_loss��=���        )��P	���A���A�*

	conv_loss<�=,/\�        )��P	�A���A�*

	conv_loss-'�=���        )��P	F�A���A�*

	conv_loss�C�=).U�        )��P	-N�A���A�*

	conv_lossp�=M��        )��P	O��A���A�*

	conv_loss�
�=M�S5        )��P	���A���A�*

	conv_lossy��=����        )��P	��A���A�*

	conv_loss���=�        )��P	d'�A���A�*

	conv_loss��=���        )��P	Z�A���A�*

	conv_lossT��=���        )��P	���A���A�*

	conv_loss6\�=�7:�        )��P	�ɬA���A�*

	conv_losskc�=l2��        )��P	d��A���A�*

	conv_loss�ף=�`_h        )��P	`,�A���A�*

	conv_lossNv�=���        )��P	�_�A���A�*

	conv_loss~��=YD��        )��P	A���A�*

	conv_lossc��=��o        )��P	��A���A�*

	conv_loss���=j�7�        )��P	��A���A�*

	conv_lossH��=����        )��P	<H�A���A�*

	conv_loss{
�=���        )��P	Jx�A���A�*

	conv_loss�E�=6�L�        )��P	8��A���A�*

	conv_loss�~�=pu~        )��P	�ٮA���A�*

	conv_loss���=�bl        )��P	R	�A���A�*

	conv_loss��=�,�        )��P	�8�A���A�*

	conv_loss�}�=�u2        )��P	fh�A���A�*

	conv_loss���=�Ks0        )��P	��A���A�*

	conv_loss1�>a���        )��P	�ůA���A�*

	conv_lossQk	>��j>        )��P	��A���A�*

	conv_loss�
>�u�        )��P	�!�A���A�*

	conv_loss~��=��ɏ        )��P	�O�A���A�*

	conv_loss=���k        )��P	~�A���A�*

	conv_loss/�= �        )��P	⬰A���A�*

	conv_loss��>�?&�        )��P	�ݰA���A�*

	conv_lossx�>�        )��P	��A���A�*

	conv_loss��>�g�        )��P	`=�A���A�*

	conv_lossh�=y�%N        )��P	m�A���A�*

	conv_loss�=��        )��P	��A���A�*

	conv_loss���=�˩P        )��P	�˱A���A�*

	conv_loss���=�x�        )��P	 �A���A�*

	conv_loss��=cU�        )��P	B0�A���A�*

	conv_lossc��=�"`        )��P	�b�A���A�*

	conv_loss�D�=�=��        )��P	7��A���A�*

	conv_loss���=�3q        )��P	d=�A���A�*

	conv_lossW;�=t�A        )��P	�n�A���A�*

	conv_loss��=��X        )��P	���A���A�*

	conv_loss���=c,r�        )��P	�ɴA���A�*

	conv_loss��=�2�        )��P	���A���A�*

	conv_loss�N�=���"        )��P	�1�A���A�*

	conv_loss�7�=i�J�        )��P	mc�A���A�*

	conv_loss�g�=8�b�        )��P	���A���A�*

	conv_loss1��=>��        )��P	õA���A�*

	conv_losss�=��$:        )��P	��A���A�*

	conv_loss�>�=|
hN        )��P	�4�A���A�*

	conv_loss"r�=�)�u        )��P	Ve�A���A�*

	conv_loss��=+��@        )��P	)��A���A�*

	conv_loss�)�=�e}�        )��P	J¶A���A�*

	conv_loss��=Oj�        )��P	3��A���A�*

	conv_loss{˱=o�J�        )��P	�,�A���A�*

	conv_lossы�=|W�        )��P	^a�A���A�*

	conv_lossii�=�M9'        )��P	͘�A���A�*

	conv_loss!��=���[        )��P	�̷A���A�*

	conv_losst��==x        )��P	���A���A�*

	conv_losso��={�W�        )��P	�,�A���A�*

	conv_loss���=|�6        )��P	Sc�A���A�*

	conv_loss�j�=� �\        )��P	l��A���A�*

	conv_loss4��=G"�i        )��P	YƸA���A�*

	conv_loss�֡=�d8�        )��P	H �A���A�*

	conv_loss�=c�	�        )��P	�-�A���A�*

	conv_lossp�=i�        )��P	^�A���A�*

	conv_loss�=�`m        )��P	��A���A�*

	conv_loss:�=~$�        )��P	߼�A���A�*

	conv_loss��=�e        )��P	��A���A�*

	conv_loss���=i;�X        )��P	��A���A�*

	conv_loss:�=odw�        )��P	�S�A���A�*

	conv_lossqԱ=pc�        )��P	؁�A���A�*

	conv_loss���=��7�        )��P	���A���A�*

	conv_losstR�=E��        )��P	��A���A�*

	conv_loss3��=�9�{        )��P	��A���A�*

	conv_loss
�=���        )��P	>�A���A�*

	conv_loss��=��rw        )��P	[n�A���A�*

	conv_lossE.�=w	F        )��P	[��A���A�*

	conv_loss,v�=ڞz�        )��P	�ϻA���A�*

	conv_lossLݵ=��)�        )��P	��A���A�*

	conv_loss���=u��        )��P	e4�A���A�*

	conv_loss�W�=5R�        )��P	�c�A���A�*

	conv_loss�6�=9���        )��P	�A���A�*

	conv_loss��=$[�        )��P	�¼A���A�*

	conv_loss�1�=���4        )��P	��A���A�*

	conv_loss�W�=���        )��P	"%�A���A�*

	conv_loss�|w=�)��        )��P	:S�A���A�*

	conv_loss�ۗ=�ӑ�        )��P	���A���A�*

	conv_lossx;�=ฝ        )��P	���A���A�*

	conv_loss �=�dD�        )��P	L�A���A�*

	conv_loss���=3��        )��P	�$�A���A�*

	conv_loss���=��        )��P	3T�A���A�*

	conv_loss�S�=N�߲        )��P	څ�A���A�*

	conv_lossT�>�H�        )��P	
��A���A�*

	conv_loss�Z�=Ul�        )��P	C�A���A�*

	conv_loss�$�=���        )��P	�0�A���A�*

	conv_loss9�=��        )��P	|e�A���A�*

	conv_loss?��= f2&        )��P	ל�A���A�*

	conv_lossz�=D�^�        )��P	�ͿA���A�*

	conv_loss�~�=>'��        )��P	b��A���A�*

	conv_loss���=�oIr        )��P	�=�A���A�*

	conv_loss8
�=�Pw        )��P	�l�A���A�*

	conv_loss���=l�A        )��P	���A���A�*

	conv_loss�ڳ=��        )��P	���A���A�*

	conv_lossМ�=�QW�        )��P	���A���A�*

	conv_loss�ȩ=����        )��P	w.�A���A�*

	conv_loss���=j�(�        )��P	�]�A���A�*

	conv_lossя�=H�.k        )��P	���A���A�*

	conv_loss�7�=O�Ų        )��P	���A���A�*

	conv_loss���=�;�        )��P	���A���A�*

	conv_lossl,�=��N+        )��P	e$�A���A�*

	conv_lossz�=���        )��P	�Q�A���A�*

	conv_loss�]�=�[<.        )��P	�~�A���A�*

	conv_lossu��=��Fm        )��P	���A���A�*

	conv_loss���=�/�)        )��P	���A���A�*

	conv_loss��=G�h        )��P	��A���A�*

	conv_lossZ��=+'P        )��P	�<�A���A�*

	conv_loss	��=���        )��P	\l�A���A�*

	conv_loss�;�=/k�        )��P	V��A���A�*

	conv_loss�ʐ=�0>-        )��P	���A���A�*

	conv_lossh��=�@�        )��P	��A���A�*

	conv_loss�D�=��r        )��P	R6�A���A�*

	conv_loss	0�=o�        )��P	Bn�A���A�*

	conv_loss��=SA�        )��P	q��A���A�*

	conv_loss�$�=��
�        )��P	���A���A�*

	conv_lossິ=�Ay;        )��P	�A���A�*

	conv_loss��=���        )��P	�4�A���A�*

	conv_loss�4�=���.        )��P	:e�A���A�*

	conv_loss�w�=�=��        )��P	��A���A�*

	conv_losso�=,4G        )��P	��A���A�*

	conv_loss�י=x�        )��P	/��A���A�*

	conv_loss"��=��j�        )��P	h5�A���A�*

	conv_loss6~�=V��        )��P	�w�A���A�*

	conv_loss�O�=/3�        )��P	?��A���A�*

	conv_loss�Z�=���.        )��P	^��A���A�*

	conv_loss���=r�        )��P	��A���A�*

	conv_loss*c�=�ʳ�        )��P	U�A���A�*

	conv_loss@O�=��S(        )��P	���A���A�*

	conv_loss��=�O=.        )��P	���A���A�*

	conv_loss��=��gA        )��P	6�A���A�*

	conv_lossM�	>�F�(        )��P	`1�A���A�*

	conv_loss�$>a'�        )��P	xa�A���A�*

	conv_loss���=
��        )��P	G��A���A�*

	conv_lossۖ�=�F!�        )��P	���A���A�*

	conv_lossW��=��.2        )��P	-��A���A�*

	conv_lossS=�=����        )��P	%�A���A�*

	conv_lossTm�=��F        )��P	�Z�A���A�*

	conv_loss�K�=�*ٞ        )��P	��A���A�*

	conv_lossg�=8;J�        )��P	���A���A�*

	conv_loss���=Z��g        )��P	A�A���A�*

	conv_loss��=1@�        )��P	�;�A���A�*

	conv_loss���=Z��        )��P	tp�A���A�*

	conv_losseB�=��
�        )��P	h��A���A�*

	conv_loss2Q�=�L֗        )��P	���A���A�*

	conv_lossL�=��        )��P	��A���A�*

	conv_loss���=ҏ4        )��P	\H�A���A�*

	conv_loss���=�d|        )��P	���A���A�*

	conv_lossU��=&l�        )��P	Z��A���A�*

	conv_loss�ش=���        )��P	��A���A�*

	conv_loss ��=�{tA        )��P	`:�A���A�*

	conv_loss�=k���        )��P	�o�A���A�*

	conv_loss�a�=��        )��P	<��A���A�*

	conv_lossW"�=y�5�        )��P	t��A���A�*

	conv_loss ��=Ѡ��        )��P	��A���A�*

	conv_loss�Z�=�$�G        )��P	�Y�A���A�*

	conv_loss�.�=w�)3        )��P	]��A���A�*

	conv_loss��=.�        )��P	���A���A�*

	conv_loss��=j�;        )��P	��A���A�*

	conv_loss?*�=Ǐ\        )��P	j7�A���A�*

	conv_loss{p�=8<�        )��P	�t�A���A�*

	conv_loss���=	���        )��P	2��A���A�*

	conv_loss���=[��        )��P	���A���A�*

	conv_loss���=G�        )��P	l(�A���A�*

	conv_loss�"�=c�        )��P	�Z�A���A�*

	conv_loss�|�=	        )��P	��A���A�*

	conv_loss|~�=���i        )��P	'��A���A�*

	conv_loss��=�	{]        )��P	%�A���A�*

	conv_loss��=��/�        )��P	24�A���A�*

	conv_loss�a�=���        )��P	�g�A���A�*

	conv_loss�L�=-�x        )��P	���A���A�*

	conv_loss�}�=X�A4        )��P	���A���A�*

	conv_losshk�=T�@        )��P	��A���A�*

	conv_loss���=9�-#        )��P	�6�A���A�*

	conv_loss��=����        )��P	�f�A���A�*

	conv_loss=�=�H)�        )��P	y��A���A�*

	conv_loss=��=#^��        )��P	&��A���A�*

	conv_loss"��=@�T>        )��P	4��A���A�*

	conv_loss�;�=��         )��P	}3�A���A�*

	conv_loss�=]��,        )��P	f�A���A�*

	conv_lossv�=��\�        )��P	���A���A�*

	conv_loss��>��        )��P	��A���A�*

	conv_loss�6�=�9�"        )��P	��A���A�*

	conv_loss��=!�vS        )��P	�D�A���A�*

	conv_loss<b�=ֽly        )��P	4�A���A�*

	conv_lossJO�=�ڒ        )��P	|��A���A�*

	conv_loss6�=>os�        )��P	���A���A�*

	conv_loss��=�-�x        )��P	r'�A���A�*

	conv_loss(v�=c�@        )��P	0\�A���A�*

	conv_lossN�=ɸ��        )��P	p��A���A�*

	conv_loss ^�=����        )��P	3��A���A�*

	conv_loss�{�=���S        )��P	���A���A�*

	conv_loss���=�WE        )��P	�2�A���A�*

	conv_loss�^�=�%��        )��P	�f�A���A�*

	conv_lossp��=��+�        )��P	ؖ�A���A�*

	conv_loss��=$��        )��P	���A���A�*

	conv_loss�[>��e�        )��P	���A���A�*

	conv_loss�h>No7�        )��P	�A�A���A�*

	conv_loss ��=+D��        )��P	�t�A���A�*

	conv_loss*>zi;�        )��P	���A���A�*

	conv_loss^��=lEAm        )��P	���A���A�*

	conv_loss���=���        )��P	��A���A�*

	conv_loss��=�C�&        )��P	�K�A���A�*

	conv_loss���=���        )��P	�}�A���A�*

	conv_loss	�= ��        )��P	���A���A�*

	conv_loss��=l��        )��P	���A���A�*

	conv_loss�²=�
        )��P	~�A���A�*

	conv_lossJ��="��p        )��P	�U�A���A�*

	conv_lossC�=�i'        )��P	7��A���A�*

	conv_loss8�=fUu        )��P	���A���A�*

	conv_loss���=@�        )��P	���A���A�*

	conv_loss�ʨ=c	��        )��P	��A���A�*

	conv_loss���=�f�        )��P	bV�A���A�*

	conv_lossu4�=��        )��P	���A���A�*

	conv_loss -�=��^        )��P	K��A���A�*

	conv_loss�'�=�y*s        )��P	���A���A�*

	conv_loss�=2��        )��P	2�A���A�*

	conv_loss5l�=�D�'        )��P	U�A���A�*

	conv_loss忴=�^��        )��P	*��A���A�*

	conv_loss�n�=��        )��P	G��A���A�*

	conv_lossM�=�ڗ        )��P	���A���A�*

	conv_loss���=���>        )��P	�*�A���A�*

	conv_lossir�=�K        )��P	�\�A���A�*

	conv_lossDh�=�I�|        )��P	���A���A�*

	conv_loss1�=n�H1        )��P	��A���A�*

	conv_loss��=��X        )��P	��A���A�*

	conv_loss�
�=�|�I        )��P	�"�A���A�*

	conv_loss42�=�=$�        )��P	wQ�A���A�*

	conv_loss9�o=,�        )��P	��A���A�*

	conv_loss�G�=J6        )��P	�?�A���A�*

	conv_loss˄�=UՒ�        )��P	��A���A�*

	conv_loss�=s���        )��P	�>�A���A�*

	conv_losshV�=�g��        )��P	�p�A���A�*

	conv_loss��=���e        )��P	��A���A�*

	conv_loss:!�=�9Pq        )��P	���A���A�*

	conv_loss�a�=V���        )��P	 �A���A�*

	conv_loss�x�=i��`        )��P	(G�A���A�*

	conv_loss��=�Z'�        )��P	�z�A���A�*

	conv_loss�Z�=�S�        )��P	���A���A�*

	conv_loss[�=ni�p        )��P	��A���A�*

	conv_loss%	�=���4        )��P	�:�A���A�*

	conv_loss6�=�+�        )��P	o�A���A�*

	conv_loss@��=�HHA        )��P	���A���A�*

	conv_loss`��=d+�b        )��P	���A���A�*

	conv_loss��=3{        )��P	��A���A�*

	conv_loss�&�=t���        )��P	'A�A���A�*

	conv_loss�!�=y��        )��P	2u�A���A�*

	conv_losstޘ=;��L        )��P	ͨ�A���A�*

	conv_lossS�=KH�W        )��P	���A���A�*

	conv_loss���=�#e        )��P	
�A���A�*

	conv_loss�=P�        )��P	^M�A���A�*

	conv_lossÃ�=�>�	        )��P	��A���A�*

	conv_lossھ�=���        )��P	���A���A�*

	conv_loss[�=n��        )��P	���A���A�*

	conv_lossK�=;�)        )��P	�5�A���A�*

	conv_loss7��=LC
        )��P	oh�A���A�*

	conv_loss�X�=� �C        )��P	���A���A�*

	conv_loss��=����        )��P	���A���A�*

	conv_loss���=����        )��P	R�A���A�*

	conv_lossb��=0x$        )��P	-C�A���A�*

	conv_lossyܲ=\���        )��P	�v�A���A�*

	conv_loss��=b���        )��P	���A���A�*

	conv_loss���=�yq�        )��P	-��A���A�*

	conv_lossl�=�8lw        )��P	��A���A�*

	conv_lossõ�=J|H�        )��P	�S�A���A�*

	conv_loss�q�=�h~O        )��P	���A���A�*

	conv_loss��=���)        )��P	h��A���A�*

	conv_loss��=�b�
        )��P	���A���A�*

	conv_loss*m�=<1x�        )��P	�#�A���A�*

	conv_loss��=Tg        )��P	�Z�A���A�*

	conv_loss7�=���        )��P	��A���A�*

	conv_lossW�[=�9�        )��P	���A���A�*

	conv_loss���=��        )��P	:��A���A�*

	conv_loss��=Y�&�        )��P	`%�A���A�*

	conv_loss���=�'��        )��P	�e�A���A�*

	conv_lossQ�>Pȧ=        )��P	���A���A�*

	conv_loss�x�=�)n        )��P	>��A���A�*

	conv_lossx��=n��@        )��P	�A���A�*

	conv_lossk}�=�s��        )��P	�X�A���A�*

	conv_loss�	�=�-a�        )��P	���A���A�*

	conv_loss�Ԩ=k{�        )��P	��A���A�*

	conv_loss�e�=M���        )��P		�A���A�*

	conv_loss
y�=����        )��P	5�A���A�*

	conv_lossW��=	�Hv        )��P	$o�A���A�*

	conv_loss�κ=��        )��P	��A���A�*

	conv_loss�=�=��{�        )��P	���A���A�*

	conv_lossf��=VP�N        )��P	�
�A���A�*

	conv_lossX&�=�ո�        )��P	�C�A���A�*

	conv_loss���=x�{        )��P	9|�A���A�*

	conv_loss7��=G��<        )��P	!��A���A�*

	conv_loss@,�=�         )��P	!��A���A�*

	conv_lossPҫ=�a"        )��P	%�A���A�*

	conv_lossY�=��s�        )��P	7I�A���A�*

	conv_loss%*�=�ם        )��P	���A���A�*

	conv_loss�ܓ=����        )��P	3��A���A�*

	conv_loss���=�фP        )��P	���A���A�*

	conv_lossp��=�a�        )��P	��A���A�*

	conv_loss�/�=��6�        )��P	:Q�A���A�*

	conv_loss�l�=t��        )��P	&��A���A�*

	conv_lossH��=��&        )��P	���A���A�*

	conv_lossJ��=�|O        )��P	�A���A�*

	conv_loss�r�=�Ùz        )��P	�G�A���A�*

	conv_loss�س=T�%        )��P	x��A���A�*

	conv_lossݸ�=-        )��P	D��A���A�*

	conv_loss���=qW        )��P	P��A���A�*

	conv_lossL��=�66h        )��P	��A���A�*

	conv_lossm��=��	�        )��P	�R�A���A�*

	conv_lossz��=����        )��P	���A���A�*

	conv_loss���=���n        )��P	���A���A�*

	conv_loss{��=9�w�        )��P	.��A���A�*

	conv_lossy��=�G��        )��P	C2�A���A�*

	conv_loss
o=�a1w        )��P	�s�A���A�*

	conv_loss��=.��g        )��P	'��A���A�*

	conv_lossW�=$��        )��P	��A���A�*

	conv_loss�=$&�        )��P	s�A���A�*

	conv_loss(�=����        )��P	�>�A���A�*

	conv_loss���=�ڣ�        )��P	�o�A���A�*

	conv_lossΨ=�y�v        )��P	x��A���A�*

	conv_loss"g�=�        )��P	���A���A�*

	conv_loss�_�=��,1        )��P	�
�A���A�*

	conv_loss��=�        )��P	g:�A���A�*

	conv_lossRn�=��H�        )��P	�k�A���A�*

	conv_loss��=���        )��P	ߜ�A���A�*

	conv_loss?��=�\        )��P	���A���A�*

	conv_lossO��=JB��        )��P	���A���A�*

	conv_lossB9�=줺*        )��P	3�A���A�*

	conv_lossP�=��s�        )��P	j�A���A�*

	conv_loss�i�=����        )��P	_��A���A�*

	conv_loss>��=�;7        )��P	���A���A�*

	conv_loss0}=Z��}        )��P	}�A���A�*

	conv_loss�ϑ=g���        )��P	NP�A���A�*

	conv_lossV��=Y��        )��P	9��A���A�*

	conv_lossY��=嚰�        )��P	��A���A�*

	conv_loss�7i=���^        )��P	���A���A�*

	conv_loss���=����        )��P	�%�A���A�*

	conv_loss��=�qO        )��P	c�A���A�*

	conv_loss��=�&�d        )��P	��A���A�*

	conv_lossT��=Xk?        )��P	���A���A�*

	conv_loss;��=��$        )��P	@�A���A�*

	conv_loss�h�=Ӥ�        )��P	D�A���A�*

	conv_loss
m�=h���        )��P	�v�A���A�*

	conv_lossc�=�h�        )��P	��A���A�*

	conv_loss�W�=��k�        )��P	���A���A�*

	conv_loss�=緾�        )��P	��A���A�*

	conv_loss2ޢ=�.�e        )��P	{J�A���A�*

	conv_loss��=;��        )��P	v��A���A�*

	conv_loss�ϯ=���        )��P	���A���A�*

	conv_loss��=S!5�        )��P	�A���A�*

	conv_loss�=o�ԣ        )��P	�3�A���A�*

	conv_lossN�=�g��        )��P	�c�A���A�*

	conv_loss̡�=ҁ[�        )��P	6��A���A�*

	conv_lossˉ�=B\4        )��P	���A���A�*

	conv_loss��=�?�        )��P	��A���A�*

	conv_loss���=�]��        )��P	�(�A���A�*

	conv_loss3g�=ʸ�R        )��P	�V�A���A�*

	conv_loss���=.��        )��P	ӆ�A���A�*

	conv_lossa�=�E��        )��P	���A���A�*

	conv_loss�1�=ݒ�        )��P	���A���A�*

	conv_loss�i�=���!        )��P	�!�A���A�*

	conv_loss��=[�-�        )��P	�X�A���A�*

	conv_loss�l�=���        )��P	J��A���A�*

	conv_loss��=)T�        )��P	���A���A�*

	conv_loss鄙=T[�        )��P	��A���A�*

	conv_losst+�=���v        )��P	w5�A���A�*

	conv_loss��=��ף        )��P	�f�A���A�*

	conv_loss�%o=I��X        )��P	l��A���A�*

	conv_loss���=���l        )��P	���A���A�*

	conv_loss���=��#0        )��P	� B���A�*

	conv_lossG�=�3��        )��P	�B B���A�*

	conv_loss���=G��        )��P	�t B���A�*

	conv_lossU��=��9        )��P	�� B���A�*

	conv_lossH(�=H��        )��P	�� B���A�*

	conv_lossW��=��-K        )��P	 %B���A�*

	conv_lossg��=#ӧ�        )��P	~VB���A�*

	conv_loss��=мϊ        )��P	��B���A�*

	conv_lossP�=^8^        )��P	�B���A�*

	conv_lossJ�=�J�<        )��P	��B���A�*

	conv_loss/��=��vD        )��P	�.B���A�*

	conv_loss7��=/!        )��P	�`B���A�*

	conv_loss8x�=�ݷ�        )��P	i�B���A�*

	conv_loss�2�=�#��        )��P	J�B���A�*

	conv_loss�N�=�RA"        )��P	I+B���A�*

	conv_loss��=c��        )��P	ofB���A�*

	conv_loss��=-:|        )��P	��B���A�*

	conv_loss8C�=_c*�        )��P	��B���A�*

	conv_loss��x=x�        )��P	��B���A�*

	conv_loss|�=�w._        )��P	?3B���A�*

	conv_loss�j�=�قE        )��P	6fB���A�*

	conv_lossXK�=8�        )��P	��B���A�*

	conv_lossu�=�a1�        )��P	?�B���A�*

	conv_loss�z�=����        )��P	�B���A�*

	conv_loss�~�=��r        )��P	[CB���A�*

	conv_loss�i�=��        )��P	�vB���A�*

	conv_loss�>L���        )��P	��B���A�*

	conv_loss���=�޵�        )��P	��B���A�*

	conv_loss��=�֗�        )��P	A B���A�*

	conv_losspI�=d���        )��P	�ZB���A�*

	conv_loss�,�=L���        )��P	��B���A�*

	conv_loss}x�=۔S�        )��P	��B���A�*

	conv_loss���=W}�Y        )��P	��B���A�*

	conv_loss���=O!�        )��P	I+B���A�*

	conv_loss��=���        )��P	�[B���A�*

	conv_loss�P�=�
�        )��P	�B���A�*

	conv_loss���=Q�l�        )��P	��B���A�*

	conv_loss�Ey=ۘ£        )��P	��B���A�*

	conv_lossm�=h��        )��P	N#B���A�*

	conv_loss\��=����        )��P	�[B���A�*

	conv_loss�pm=cʻ        )��P	�B���A�*

	conv_loss%��=���        )��P	��B���A�*

	conv_lossn
�=w�%�        )��P	��B���A�*

	conv_loss{��=N��        )��P	r-	B���A�*

	conv_lossŠ�=>�)[        )��P	�]	B���A�*

	conv_loss Ђ=��-�        )��P	��	B���A�*

	conv_lossQ��=
��O        )��P	�	B���A�*

	conv_loss%�=O�6�        )��P	��	B���A�*

	conv_lossw/�=)7n�        )��P	m 
B���A�*

	conv_loss*�=��g        )��P	NO
B���A�*

	conv_lossX�=�y�R        )��P	�}
B���A�*

	conv_loss0ͺ=b�k�        )��P	�
B���A�*

	conv_loss��q=w5\H        )��P	��
B���A�*

	conv_lossƢ�==>�        )��P	QB���A�*

	conv_loss�5�=3�A        )��P	QJB���A�*

	conv_loss��=GYȨ        )��P	�~B���A�*

	conv_loss���=���        )��P	�B���A�*

	conv_lossǄ�=�/`7        )��P	��B���A�*

	conv_loss�'�=�3        )��P	NB���A�*

	conv_loss"P�=���Z        )��P	�OB���A�*

	conv_loss���=)�        )��P	߁B���A�*

	conv_loss#�=~[I�        )��P	ĺB���A�*

	conv_loss���=�3J        )��P	>�B���A�*

	conv_loss{�=
Y        )��P	^�B���A�*

	conv_lossI��=w�o"        )��P	��B���A�*

	conv_lossky�=d���        )��P	�B���A�*

	conv_loss��k=,)�         )��P	�$B���A�*

	conv_lossFT�=Q�|}        )��P	VB���A�*

	conv_lossYƦ="2�m        )��P	ƒB���A�*

	conv_lossq܈=uj�        )��P	�B���A�*

	conv_loss�˷=\�$R        )��P	��B���A�*

	conv_loss�-�=��        )��P	�5B���A�*

	conv_loss�ɿ=(��l        )��P	�iB���A�*

	conv_loss���=~��        )��P	1�B���A�*

	conv_lossQ�=<��        )��P	��B���A�*

	conv_loss���=�U9        )��P	@B���A�*

	conv_loss[��=@�K        )��P	�EB���A�*

	conv_loss9<�=�o��        )��P	�vB���A�*

	conv_loss�ˤ=ס�        )��P	��B���A�*

	conv_loss!;�=�[Z        )��P	��B���A�*

	conv_lossXό=}�'�        )��P	gB���A�*

	conv_loss�I�=[`��        )��P	�KB���A�*

	conv_lossK�=�9�;        )��P	�|B���A�*

	conv_lossz$�=��m        )��P	U�B���A�*

	conv_loss�=��        )��P	L�B���A�*

	conv_loss�a�=��~�        )��P	�B���A�*

	conv_lossJ �=��-�        )��P	�LB���A�*

	conv_loss]�=����        )��P	�}B���A�*

	conv_loss�A�=},�y        )��P	�B���A�*

	conv_loss�=���        )��P	��B���A�*

	conv_loss��=zP>?        )��P	�B���A�*

	conv_loss0��=��!:        )��P	-<B���A�*

	conv_loss��=O��z        )��P	�mB���A�*

	conv_loss��=��        )��P	�B���A�*

	conv_loss�G�=T��        )��P	��B���A�*

	conv_loss��=/�,        )��P	�B���A�*

	conv_lossA�=p�U        )��P	.B���A�*

	conv_loss�Ǔ=?v��        )��P	s\B���A�*

	conv_loss�V�=] �"        )��P	H�B���A�*

	conv_loss(|�=2�G        )��P	һB���A�*

	conv_loss,C~=�GC`        )��P	��B���A�*

	conv_loss�)�=�\��        )��P	B���A�*

	conv_loss��=}G;        )��P	dKB���A�*

	conv_lossD��=ٌ��        )��P	�zB���A�*

	conv_loss�2�=����        )��P	��B���A�*

	conv_losssF�=�g�        )��P	��B���A�*

	conv_loss�\�=$l��        )��P	*B���A�*

	conv_loss�(f=W��`        )��P	�=B���A�*

	conv_loss���=,R�;        )��P	�mB���A�*

	conv_loss��=����        )��P	E�B���A�*

	conv_loss�I�=�y�@        )��P	��B���A�*

	conv_lossЫ=�#Ni        )��P	R
B���A�*

	conv_loss��u=`�ZS        )��P	t>B���A�*

	conv_loss2��=q�|W        )��P	��B���A�*

	conv_loss�T�=I6HY        )��P	k�B���A�*

	conv_loss��=Q���        )��P	��B���A�*

	conv_loss�w�=���3        )��P	�B���A�*

	conv_loss/$�=CJb        )��P	�HB���A�*

	conv_loss���=5o        )��P	|B���A�*

	conv_loss�}�=�S#        )��P	�B���A�*

	conv_lossŚ�=���.        )��P	��B���A�*

	conv_loss��=�t��        )��P	B���A�*

	conv_loss�޿=��:�        )��P	�VB���A�*

	conv_loss���=�7�        )��P	g�B���A�*

	conv_loss�o�=�r �        )��P	;�B���A�*

	conv_loss��=̳�        )��P	A�B���A�*

	conv_lossO��=v�0�        )��P	�)B���A�*

	conv_loss���=r�c�        )��P	�cB���A�*

	conv_loss�V�=���        )��P	��B���A�*

	conv_loss���=֛O�        )��P	
�B���A�*

	conv_lossf�=u���        )��P	)	B���A�*

	conv_lossi9�=���)        )��P	�<B���A�*

	conv_losse��=8��r        )��P	�nB���A�*

	conv_lossm�=�Tʰ        )��P	R�B���A�*

	conv_lossuH�=W}��        )��P	v�B���A�*

	conv_lossZ�=4F.Y        )��P	dB���A�*

	conv_loss�!�=G^��        )��P	�JB���A�*

	conv_loss-��=zlm        )��P	�~B���A�*

	conv_loss��=�@�        )��P	��B���A�*

	conv_lossΙ�=�R�I        )��P	�B���A�*

	conv_loss�C�=���        )��P	�#B���A�*

	conv_loss�)�=���r        )��P	%UB���A�*

	conv_loss�=rG�        )��P	�B���A�*

	conv_loss��=)n\"        )��P	�B���A�*

	conv_loss���=c���        )��P	G�B���A�*

	conv_loss{��=��(        )��P	KB���A�*

	conv_loss	��=�IKj        )��P	NB���A�*

	conv_loss��=E
G        )��P	)�B���A�*

	conv_loss���=��^�        )��P	0�B���A�*

	conv_losse��=I�)�        )��P	��B���A�*

	conv_loss�m�=`w��        )��P	�$ B���A�*

	conv_lossbF�=��7"        )��P	�Y B���A�*

	conv_loss�؆=hH2        )��P	`� B���A�*

	conv_loss�,�=vA�        )��P	�� B���A�*

	conv_lossT�=��~>        )��P	.!B���A�*

	conv_loss�8�=��xl        )��P	�d!B���A�*

	conv_loss�ְ=L���        )��P	��!B���A�*

	conv_lossx>�=���        )��P	:�!B���A�*

	conv_lossEV�=Y�        )��P	-"B���A�*

	conv_lossrm�=�Rr        )��P	R"B���A�*

	conv_loss�Wv=բ��        )��P	r�"B���A�*

	conv_loss�9�=����        )��P	Ƕ"B���A�*

	conv_loss�&�=Ot�        )��P	r�"B���A�*

	conv_loss�p�=m�x�        )��P	_:#B���A�*

	conv_lossU�=�=�        )��P	�o#B���A�*

	conv_lossM4�=<a q        )��P	�#B���A�*

	conv_loss�̵=S�=        )��P	>�#B���A�*

	conv_lossn�=P(l_        )��P	!$B���A�*

	conv_lossjL�=�!��        )��P	g^$B���A�*

	conv_loss��=/.-�        )��P	V�$B���A�*

	conv_loss���=-��<        )��P		�$B���A�*

	conv_loss��=�d�        )��P	�%B���A�*

	conv_loss&@d=m��        )��P	�6%B���A�*

	conv_loss¢g=hQ#        )��P		m%B���A�*

	conv_loss�c�=��i�        )��P	 �%B���A�*

	conv_loss���=�-W        )��P	��%B���A�*

	conv_loss��=y?K        )��P	&B���A�*

	conv_lossC�=����        )��P	�8&B���A�*

	conv_loss==��x        )��P	�t&B���A�*

	conv_loss"��=��8        )��P	(�&B���A�*

	conv_loss�d�=D/v        )��P	��&B���A�*

	conv_loss���=�_�7        )��P	�/'B���A�*

	conv_loss�	�=�S�        )��P	�m'B���A�*

	conv_loss��z=^�;�        )��P	k�'B���A�*

	conv_lossH[�=���        )��P	4�'B���A�*

	conv_loss���=d8�>        )��P	�(B���A�*

	conv_loss=��='	��        )��P	j3(B���A�*

	conv_loss��=8��{        )��P	�k(B���A�*

	conv_loss�=��        )��P	��(B���A�*

	conv_loss 1�=�/.p        )��P	��(B���A�*

	conv_loss:s�=��,�        )��P	�)B���A�*

	conv_loss�0p=��'z        )��P	�7)B���A�*

	conv_loss���=��C        )��P	�|)B���A�*

	conv_lossD$�=�_M        )��P	��)B���A�*

	conv_loss�=�&�        )��P	7�)B���A�*

	conv_lossHN�=|?        )��P	Q*B���A�*

	conv_lossjM�=��%N        )��P	�H*B���A�*

	conv_loss��u=F[��        )��P	�*B���A�*

	conv_loss�¬=��'�        )��P	��*B���A�*

	conv_loss���=�4��        )��P		�*B���A�*

	conv_loss�ě=3�ػ        )��P	�+B���A�*

	conv_loss���=��s�        )��P	TL+B���A�*

	conv_loss��=��a        )��P	&�+B���A�*

	conv_loss��= ��        )��P	��+B���A�*

	conv_loss���=����        )��P	��+B���A�*

	conv_loss�{=o���        )��P	�(,B���A�*

	conv_loss̍�=��%�        )��P	\X,B���A�*

	conv_lossQ.=5�R        )��P	��,B���A�*

	conv_loss��^=$_a        )��P	�,B���A�*

	conv_loss�i=��&'        )��P	��,B���A�*

	conv_loss�l�=�	W        )��P	�.-B���A�*

	conv_loss �=�z�        )��P	�a-B���A�*

	conv_loss�5�=:�=        )��P	 �-B���A�*

	conv_loss�\�=!:/�        )��P	@�-B���A�*

	conv_lossE�u=15��        )��P	�.B���A�*

	conv_lossDb�=bwM�        )��P	�<.B���A�*

	conv_loss�D�=���G        )��P	Vu.B���A�*

	conv_loss=��=VH��        )��P	h�.B���A�*

	conv_lossH+�=���        )��P	��.B���A�*

	conv_loss��=b<"/        )��P	�/B���A�*

	conv_loss�ݻ=�ǿ�        )��P	�N/B���A�*

	conv_lossk��=��!        )��P	D�/B���A�*

	conv_loss���=�Ѕ�        )��P	��/B���A�*

	conv_loss�M�=|�        )��P	0�/B���A�*

	conv_loss"�=����        )��P	.0B���A�*

	conv_loss�O�=t�q        )��P	?a0B���A�*

	conv_loss�a�=p��y        )��P	e�0B���A�*

	conv_loss~��=�� t        )��P	��0B���A�*

	conv_lossĹ�=��W        )��P	!%1B���A�*

	conv_loss���=ZĽ�        )��P	}Y1B���A�*

	conv_loss��=M#V�        )��P	Ќ1B���A�*

	conv_loss6�=j��Z        )��P	��1B���A�*

	conv_loss+�=����        )��P	��1B���A�*

	conv_loss
3�=��s        )��P	B.2B���A�*

	conv_loss}�[=C2�        )��P	|_2B���A�*

	conv_lossڕ�=�cj        )��P	}�2B���A�*

	conv_lossid=c��r        )��P	��2B���A�*

	conv_lossu�=E��        )��P	#�2B���A�*

	conv_lossl�}=XU��        )��P	..3B���A�*

	conv_loss�H�=�]j        )��P	�\3B���A�*

	conv_loss�\�=��uO        )��P	ޖ3B���A�*

	conv_loss�7�=c�&        )��P	8�3B���A�*

	conv_loss�Ҝ=��^�        )��P	�3B���A�*

	conv_loss��=�p        )��P	]+4B���A�*

	conv_loss��=3�E        )��P	�X4B���A�*

	conv_loss4�|=[��        )��P	A�4B���A�*

	conv_loss�s�=WԪ        )��P	��4B���A�*

	conv_loss��=΂|        )��P	��4B���A�*

	conv_loss!Y�=�qw        )��P	�5B���A�*

	conv_loss��=#�T        )��P		E5B���A�*

	conv_loss�\�=��\        )��P	 s5B���A�*

	conv_lossx��=���        )��P	��5B���A�*

	conv_lossj�=9��'        )��P	��5B���A�*

	conv_loss�.�=�a)b        )��P	�6B���A�*

	conv_loss�<�=z�l        )��P	y36B���A�*

	conv_loss�f�=��A�        )��P	b6B���A�*

	conv_lossj��=O���        )��P	N�6B���A�*

	conv_loss�V�=�J�r        )��P	��6B���A�*

	conv_loss��=��S        )��P	@�6B���A�*

	conv_loss���=2�d�        )��P	/ 7B���A�*

	conv_loss���=��
�        )��P	P7B���A�*

	conv_loss<E�=L��f        )��P	p�7B���A�*

	conv_loss/q=���        )��P	�7B���A�*

	conv_loss� �=M��        )��P	�B9B���A�*

	conv_loss�=���        )��P	�r9B���A�*

	conv_loss���=^��8        )��P	��9B���A�*

	conv_losse	�=���        )��P	&�9B���A�*

	conv_loss�a�=E�N#        )��P	�	:B���A�*

	conv_lossN�=��         )��P	(?:B���A�*

	conv_loss��=1���        )��P	�q:B���A�*

	conv_loss��=`�        )��P	��:B���A�*

	conv_loss��=�uiB        )��P	3�:B���A�*

	conv_loss���=�t�b        )��P	�);B���A�*

	conv_loss_�=��F        )��P	ic;B���A�*

	conv_loss.�=�FA        )��P	f�;B���A�*

	conv_lossA��=Jg��        )��P	>�;B���A�*

	conv_lossl �=?p͕        )��P	��;B���A�*

	conv_lossO,�=�        )��P	0<B���A�*

	conv_loss��j=�^$f        )��P	�_<B���A�*

	conv_loss!o�=O\F        )��P	H�<B���A�*

	conv_loss��c=P���        )��P	a�<B���A�*

	conv_loss���=]k�9        )��P	'�<B���A�*

	conv_lossv,�=�a�        )��P	�-=B���A�*

	conv_loss!�{=c��        )��P	�`=B���A�*

	conv_loss���=+5X        )��P	�=B���A�*

	conv_losse0=y՗b        )��P	�=B���A�*

	conv_loss�!�=41�        )��P	�>B���A�*

	conv_loss\Y�=���'        )��P	 ;>B���A�*

	conv_loss�i�=��3�        )��P	�o>B���A�*

	conv_lossR%:=��         )��P	��>B���A�*

	conv_loss0��=���        )��P	��>B���A�*

	conv_loss�we=�K^        )��P	�?B���A�*

	conv_loss�0�=�~�        )��P	WB?B���A�*

	conv_loss:��=��        )��P	v?B���A�*

	conv_loss�}�=z.�4        )��P	��?B���A�*

	conv_lossӏ�=�ӟ        )��P	��?B���A�*

	conv_lossAx�=�M}        )��P	U@B���A�*

	conv_lossQ�a=x��t        )��P	�L@B���A�*

	conv_loss�0�=��d        )��P	�@B���A�*

	conv_loss�G�=w�9        )��P	��@B���A�*

	conv_lossz=w�l        )��P	6�@B���A�*

	conv_lossr�w=���b        )��P	,AB���A�*

	conv_loss�8{= �t�        )��P	D`AB���A�*

	conv_loss�
w=�x�!        )��P	��AB���A�*

	conv_loss��C=�1`        )��P	��AB���A�*

	conv_loss�R�=2vԢ        )��P	��AB���A�*

	conv_loss6z=ܡm
        )��P	!5BB���A�*

	conv_loss�/�=�q��        )��P	�eBB���A�*

	conv_loss�a|=y��        )��P	0�BB���A�*

	conv_loss��=�E�r        )��P	��BB���A�*

	conv_loss&�=�Y�        )��P	�CB���A�*

	conv_loss"��=7l�        )��P	�CCB���A�*

	conv_loss���=yba        )��P	�wCB���A�*

	conv_lossј�=3	,        )��P	��CB���A�*

	conv_loss���=�AL	        )��P	��CB���A�*

	conv_lossj'�=��R�        )��P	4DB���A�*

	conv_loss|Un=�uG        )��P	�gDB���A�*

	conv_loss�-z=3�/        )��P	�DB���A�*

	conv_loss�x�=d�^�        )��P	W�DB���A�*

	conv_loss�x�=��        )��P	�EB���A�*

	conv_loss�4�=�5$        )��P	9XEB���A�*

	conv_loss9�w=h��7        )��P	+�EB���A�*

	conv_loss�T�=}�I
        )��P	N�EB���A�*

	conv_loss��=�	�        )��P	��EB���A�*

	conv_loss�
{=��Y�        )��P	�"FB���A�*

	conv_loss(|�=
uv�        )��P	�[FB���A�*

	conv_loss�x=�
k        )��P	2�FB���A�*

	conv_loss7��=ި j        )��P	��FB���A�*

	conv_loss�I=�=�!        )��P	��FB���A�*

	conv_loss�.�=��V;        )��P	$GB���A�*

	conv_loss��=XN��        )��P	�bGB���A�*

	conv_loss��=B���        )��P	��GB���A�*

	conv_lossL�=���        )��P	[�GB���A�*

	conv_loss0{="0��        )��P	�HB���A�*

	conv_lossf	�=��h        )��P	v:HB���A�*

	conv_loss��=�xX�        )��P	�vHB���A�*

	conv_lossb΀=�         )��P	u�HB���A�*

	conv_loss���=�B�/        )��P	n�HB���A�*

	conv_loss��=�tR5        )��P	fIB���A�*

	conv_loss��w=����        )��P	 6IB���A�*

	conv_loss��=CP�        )��P	srIB���A�*

	conv_loss��=�'�        )��P	��IB���A�*

	conv_lossݑ�=�n��        )��P	��IB���A�*

	conv_lossSi=9�        )��P	�JB���A�*

	conv_loss�Ҋ=<|gT        )��P	xMJB���A�*

	conv_loss���=��?�        )��P	юJB���A�*

	conv_loss���=�Ja�        )��P	8�JB���A�*

	conv_loss>Ź=���        )��P	D�JB���A�*

	conv_loss���=�=�s        )��P	�%KB���A�*

	conv_loss��o=�>�        )��P	�UKB���A�*

	conv_lossp,�=7��        )��P	��KB���A�*

	conv_loss���=Թv�        )��P	'�KB���A�*

	conv_losscT�=���        )��P	�LB���A�*

	conv_loss�J�=F�2�        )��P	3LB���A�*

	conv_loss^��=0A�        )��P	�bLB���A�*

	conv_loss��q=�J�        )��P	s�LB���A�*

	conv_loss��c=�I        )��P	y�LB���A�*

	conv_lossl�=+        )��P	?MB���A�*

	conv_loss�<�=���        )��P	?MB���A�*

	conv_loss���=�S�,        )��P	�MB���A�*

	conv_loss0^�=tk        )��P	-�MB���A�*

	conv_loss<G�=S�        )��P	��MB���A�*

	conv_loss��=��K        )��P	FGRB���A�*

	conv_loss�=J�H        )��P	�RB���A�*

	conv_loss�%�=�_,        )��P	O�RB���A�*

	conv_lossi��=��޿        )��P	��RB���A�*

	conv_loss��=�F��        )��P	CSB���A�*

	conv_loss�f�=dk�        )��P	#HSB���A�*

	conv_loss��=��ڋ        )��P	�zSB���A�*

	conv_lossa�.={�-Q        )��P	�SB���A�*

	conv_loss��=!,�2        )��P	w�SB���A�*

	conv_loss�Y�=�`�        )��P	�TB���A�*

	conv_loss���=i���        )��P	�FTB���A�*

	conv_loss;ֳ='�Ko        )��P	�TB���A�*

	conv_lossS#o=��G�        )��P	߽TB���A�*

	conv_lossP��=��<�        )��P	��TB���A�*

	conv_loss��{=�$�6        )��P	�UB���A�*

	conv_lossk`�= ��        )��P	�NUB���A�*

	conv_loss�7=y~�        )��P	�~UB���A�*

	conv_lossJx�=Oo	V        )��P	7�UB���A�*

	conv_loss��=��[�        )��P	��UB���A�*

	conv_loss��=�˽�        )��P	[VB���A�*

	conv_loss���=�}lK        )��P	[=VB���A�*

	conv_loss^]�=.��s        )��P	�kVB���A�*

	conv_loss�\�=R���        )��P	
�VB���A�*

	conv_loss�Ӊ=,�5        )��P	��VB���A�*

	conv_loss!�=pl-�        )��P	��VB���A�*

	conv_loss�)m=�%H�        )��P	8WB���A�*

	conv_loss�_�=&h��        )��P	\kWB���A�*

	conv_loss��=���        )��P	/�WB���A�*

	conv_lossI��=�1�        )��P	��WB���A�*

	conv_loss�@�=���p        )��P	*�WB���A�*

	conv_loss���=���        )��P	.XB���A�*

	conv_loss�5-=���        )��P	�]XB���A�*

	conv_loss��=�hy`        )��P	�XB���A�*

	conv_lossM�=�S��        )��P	�XB���A�*

	conv_loss\ō=XV        )��P	��XB���A�*

	conv_lossryX=�L�+        )��P	�YB���A�*

	conv_loss���=��,�        )��P	DOYB���A�*

	conv_lossFr�=����        )��P	�YB���A�*

	conv_loss=��=RF��        )��P	6�YB���A�*

	conv_loss:�>=�R�        )��P	a�YB���A�*

	conv_lossQN=��Ԙ        )��P	�ZB���A�*

	conv_loss#R�=N�        )��P	hDZB���A�*

	conv_lossy�=�w�        )��P	�vZB���A�*

	conv_loss��=�]�9        )��P	1�ZB���A�*

	conv_loss�ɘ=�&�f        )��P	��ZB���A�*

	conv_loss���=�=        )��P	~[B���A�*

	conv_loss2ӝ=�w>1        )��P	�:[B���A�*

	conv_lossZ�=Εd�        )��P	.i[B���A�*

	conv_losss�u=���        )��P	e�[B���A�*

	conv_loss2��=_���        )��P	f�[B���A�*

	conv_losso�=�(�M        )��P	��[B���A�*

	conv_loss�9�=g���        )��P	.<\B���A�*

	conv_loss��=�=z�        )��P	j\B���A�*

	conv_loss� u=�^'�        )��P	M�\B���A�*

	conv_loss>��=�\u        )��P	��\B���A�*

	conv_lossfˎ=��        )��P	�\B���A�*

	conv_loss|M�=���        )��P	�/]B���A�*

	conv_loss{^�=a���        )��P	D_]B���A�*

	conv_lossUa�=�x�        )��P	��]B���A�*

	conv_loss�}�=�'s&        )��P	ɾ]B���A�*

	conv_lossR��=�Gn�        )��P	5�]B���A�*

	conv_loss���=��a�        )��P	,+^B���A�*

	conv_loss��=��p        )��P	�_^B���A�*

	conv_loss���=x�,        )��P	��^B���A�*

	conv_loss��=���        )��P	z�^B���A�*

	conv_lossv��=��R        )��P	�_B���A�*

	conv_loss5[=�'Wd        )��P	�=_B���A�*

	conv_lossa0�=��<(        )��P	�l_B���A�*

	conv_lossm_~=~�4        )��P	�_B���A�*

	conv_loss`�=�yٯ        )��P	v�_B���A�*

	conv_lossն�=-� �        )��P	��_B���A�*

	conv_loss��=v�        )��P	�1`B���A�*

	conv_lossߖ=��        )��P	_b`B���A�*

	conv_lossצ�=��F�        )��P	�`B���A�*

	conv_loss���=�zh�        )��P	��`B���A�*

	conv_lossH[e=���        )��P	+�`B���A�*

	conv_lossނ�=���        )��P	�'aB���A�*

	conv_loss�j=*MU�        )��P	�WaB���A�*

	conv_loss.�r=���        )��P	��aB���A�*

	conv_loss��d=tU�(        )��P	�aB���A�*

	conv_loss�=s�jh        )��P	��aB���A�*

	conv_lossxW~=���        )��P	/bB���A�*

	conv_loss��=7��'        )��P	nJbB���A�*

	conv_loss�?�=.��        )��P	<{bB���A�*

	conv_loss��=�-��        )��P	��bB���A�*

	conv_loss+��="��        )��P	�bB���A�*

	conv_loss:4T=A�        )��P	cB���A�*

	conv_lossFĄ=@��        )��P	�OcB���A�*

	conv_loss���=ߨԁ        )��P	-�cB���A�*

	conv_loss��=�%�%        )��P	F�cB���A�*

	conv_loss�޻==H�V        )��P	��cB���A�*

	conv_loss)�=x�{�        )��P	�$dB���A�*

	conv_loss�&�=n��r        )��P	�UdB���A�*

	conv_loss[H�=O�O        )��P	��dB���A�*

	conv_loss�
�=�Q��        )��P	`�dB���A�*

	conv_loss�E�=���        )��P	i�dB���A�*

	conv_loss�=�6eI        )��P	KeB���A�*

	conv_loss���=�}�        )��P	�EeB���A�*

	conv_losslx�=*L��        )��P	DxeB���A�*

	conv_loss/t=빨�        )��P	��eB���A�*

	conv_loss��=/'�        )��P	a�eB���A�*

	conv_loss��{=���'        )��P	�kgB���A�*

	conv_loss.=�L��        )��P	8�gB���A�*

	conv_loss�`�=�9`�        )��P	�gB���A�*

	conv_loss���=�-��        )��P	y hB���A�*

	conv_lossK�h=	�V        )��P	�4hB���A�*

	conv_loss��=�[�        )��P	�ehB���A�*

	conv_loss���=��        )��P	��hB���A�*

	conv_lossia�=Y��        )��P	�hB���A�*

	conv_loss�8�=��$        )��P	�iB���A�*

	conv_loss�"�=�6è        )��P	VFiB���A�*

	conv_loss���=d"�        )��P	�yiB���A�*

	conv_loss]6L=�yP3        )��P	s�iB���A�*

	conv_losspft=ʻ'�        )��P	 �iB���A�*

	conv_loss�>=0�#�        )��P	�
jB���A�*

	conv_loss�<�=��        )��P	KjB���A�*

	conv_loss�H�= ��        )��P	�zjB���A�*

	conv_loss6�=���        )��P	�jB���A�*

	conv_loss��=eQQ�        )��P	��jB���A�*

	conv_loss��q=�j��        )��P	VkB���A�*

	conv_loss!;�=Y���        )��P	�JkB���A�*

	conv_loss�j�=LW9        )��P	�kB���A�*

	conv_loss|U=�-�|        )��P	��kB���A�*

	conv_loss���=��:        )��P	�kB���A�*

	conv_loss�T�=���M        )��P	IlB���A�*

	conv_loss�f=��y�        )��P	BHlB���A�*

	conv_loss�v�=�=�        )��P	�ylB���A�*

	conv_loss1��=�>��        )��P	��lB���A�*

	conv_loss�1�=�b�        )��P	��lB���A�*

	conv_losssْ=T���        )��P	 mB���A�*

	conv_loss��=E��&        )��P	&>mB���A�*

	conv_loss���=��b�        )��P	GumB���A�*

	conv_loss愐=��hy        )��P	©mB���A�*

	conv_lossN��=�S�        )��P	2�mB���A�*

	conv_loss�C�=]<!�        )��P	�nB���A�*

	conv_loss=�%�        )��P	EnB���A�*

	conv_lossn��=��:�        )��P	�wnB���A�*

	conv_loss�qS=�V�        )��P	�nB���A�*

	conv_loss�T�=�QK�        )��P	�nB���A�*

	conv_lossb��=�%:�        )��P	�oB���A�*

	conv_loss�U�=fn�        )��P	�7oB���A�*

	conv_loss+2k=|h�        )��P	�hoB���A�*

	conv_loss9�=��s�        )��P	�oB���A�*

	conv_loss�Fa=��_{        )��P	��oB���A�*

	conv_loss�ye=(���        )��P	H�oB���A�*

	conv_loss�H�=GX�        )��P	�'pB���A�*

	conv_loss��=
Y��        )��P	�WpB���A�*

	conv_loss�"�=�.��        )��P	1�pB���A�*

	conv_loss�s�=����        )��P	��pB���A�*

	conv_loss<��=��(�        )��P	��pB���A�*

	conv_loss%M<=3�=        )��P	� qB���A�*

	conv_loss��=�4K        )��P	�fqB���A�*

	conv_loss�]�=��I        )��P	՘qB���A�*

	conv_lossly�=i*�        )��P	O�qB���A�*

	conv_lossr�Z=^�        )��P	c�qB���A�*

	conv_loss�2K=F�        )��P	�6rB���A�*

	conv_lossN�d=�8�        )��P	fgrB���A�*

	conv_loss�:�=h�p        )��P	ޡrB���A�*

	conv_loss��=��%        )��P	��rB���A�*

	conv_loss�=��        )��P	usB���A�*

	conv_loss�=z]        )��P	�KsB���A�*

	conv_lossn֒=�F        )��P	D�sB���A�*

	conv_loss8�k=��9        )��P	��sB���A�*

	conv_loss*v=�{{        )��P	��sB���A�*

	conv_loss��Q=v�Da        )��P	k$tB���A�*

	conv_loss�
x=���        )��P	
YtB���A�*

	conv_loss$S�=��f        )��P	o�tB���A�*

	conv_lossa%m=r�v        )��P	��tB���A�*

	conv_loss�M�=3'�z        )��P	��tB���A�*

	conv_loss �]=U���        )��P	�$uB���A�*

	conv_loss�A=żw�        )��P	�[uB���A�*

	conv_loss���=�7�        )��P	;�uB���A�*

	conv_loss�^�=�e        )��P	��uB���A�*

	conv_loss�Pk=
E��        )��P	�uB���A�*

	conv_loss���=��#        )��P	�(vB���A�*

	conv_loss�A�=,�2        )��P	�_vB���A�*

	conv_lossz[w=�}�        )��P	{�vB���A�*

	conv_lossy�c=�m�        )��P	-�vB���A�*

	conv_loss�܂=I��        )��P	�wB���A�*

	conv_lossZ	�=�""�        )��P	-CwB���A�*

	conv_loss���=v��        )��P	�wwB���A�*

	conv_loss*��=�/�        )��P	��wB���A�*

	conv_loss���=�p        )��P	��wB���A�*

	conv_loss���=eU�        )��P	�&xB���A�*

	conv_loss��}=2��~        )��P	�[xB���A�*

	conv_lossg�v=z���        )��P	X�xB���A�*

	conv_loss�z�=D.?�        )��P	\�xB���A�*

	conv_loss���=8�        )��P	yyB���A�*

	conv_loss�%�=*��Y        )��P	5yB���A�*

	conv_lossT=9���        )��P	iyB���A�*

	conv_loss���=~,C�        )��P	 �yB���A�*

	conv_loss�5�=��l�        )��P	��yB���A�*

	conv_lossR,�=���]        )��P	�zB���A�*

	conv_loss�l�=j��        )��P	�GzB���A�*

	conv_loss7�R=�3�j        )��P	L~zB���A�*

	conv_lossh�=m�}�        )��P	7�zB���A�*

	conv_loss+�u=��        )��P	a�zB���A�*

	conv_loss҈J=�^X        )��P	c,{B���A�*

	conv_loss�of=��C�        )��P	�^{B���A�*

	conv_loss婗=��V�        )��P	��{B���A�*

	conv_loss赩=���         )��P	��{B���A�*

	conv_loss�	�=��|�        )��P	4|B���A�*

	conv_loss[�r=�rK        )��P	:?|B���A�*

	conv_lossf0�=Fr        )��P	-p|B���A�*

	conv_loss�`�=m�        )��P	Ц|B���A�*

	conv_loss�[�=��V        )��P	R�|B���A�*

	conv_loss汳=,,�        )��P	�+}B���A�*

	conv_lossI��=��a?        )��P	�[}B���A�*

	conv_loss�y|=.;y        )��P	
�}B���A�*

	conv_loss���=ӓz        )��P	��}B���A�*

	conv_loss��=����        )��P	��}B���A�*

	conv_loss�S�=~�0E        )��P	�*~B���A�*

	conv_lossR�=�        )��P	�^~B���A�*

	conv_loss���=�`�n        )��P	��~B���A�*

	conv_loss�Ӂ=�5o        )��P	W�~B���A�*

	conv_loss�y=���8        )��P	B���A�*

	conv_loss+bq=�,�N        )��P	�5B���A�*

	conv_loss/��=%�Ӹ        )��P	��B���A�*

	conv_lossF��=y��        )��P	�B���A�*

	conv_loss���=8UT        )��P	��B���A�*

	conv_loss�g�=��^�        )��P	�+�B���A�*

	conv_lossb?�=7�.        )��P	�c�B���A�*

	conv_lossh�=�	�6        )��P	A��B���A�*

	conv_loss�g=ƍ"Q        )��P	�ȀB���A�*

	conv_lossBhR=�s�        )��P	N	�B���A�*

	conv_losszJ�=ϝ�        )��P	;�B���A�*

	conv_loss˖=[��        )��P	&n�B���A�*

	conv_loss�~�=:i�-        )��P	���B���A�*

	conv_lossk?t=�Q*        )��P	�ρB���A�*

	conv_loss�t\=x?1        )��P	�	�B���A�*

	conv_loss�i�=
U�n        )��P	�B�B���A�*

	conv_loss���=Y�        )��P	�z�B���A�*

	conv_loss���=Τ<�        )��P	d��B���A�*

	conv_loss�a�=i��2        )��P	T؂B���A�*

	conv_lossLx=mF]I        )��P	��B���A�*

	conv_loss�n�=��        )��P	�N�B���A�*

	conv_lossU�K=��P+        )��P	S��B���A�*

	conv_loss/d�=���        )��P	貃B���A�*

	conv_loss���=�pģ        )��P	��B���A�*

	conv_lossت�=�\0�        )��P	O%�B���A�*

	conv_loss�=�?{        )��P	�W�B���A�*

	conv_lossxWs=�n�        )��P	F��B���A�*

	conv_loss�Z�=��        )��P	�̈́B���A�*

	conv_loss�|�= \�u        )��P	s�B���A�*

	conv_loss_f�=�4K        )��P	=1�B���A�*

	conv_loss�D9=j�1        )��P	�b�B���A�*

	conv_loss<�=O�D�        )��P	Ø�B���A�*

	conv_lossZW�=�ρ�        )��P	˅B���A�*

	conv_losso�=�2ח        )��P	���B���A�*

	conv_loss�V�=IZc�        )��P	t9�B���A�*

	conv_lossb�>=���y        )��P	�l�B���A�*

	conv_loss2�U=w��        )��P	+��B���A�*

	conv_loss���=7Kx�        )��P	��B���A�*

	conv_loss���=�"��        )��P	#�B���A�*

	conv_loss�؊=4�D0        )��P	�[�B���A�*

	conv_lossR�r=�#8,        )��P	���B���A�*

	conv_loss�xI=ה         )��P	���B���A�*

	conv_loss���=V��p        )��P	G�B���A�*

	conv_loss�=;��V        )��P	�1�B���A�*

	conv_loss��o=!�(        )��P	�f�B���A�*

	conv_loss��]=���        )��P	���B���A�*

	conv_loss���=i+�.        )��P	шB���A�*

	conv_loss=��=h!��        )��P	��B���A�*

	conv_lossn��=���        )��P	�=�B���A�*

	conv_loss=��=��        )��P	^p�B���A�*

	conv_loss�	k=��        )��P	���B���A�*

	conv_loss�M�=�;�        )��P	M�B���A�*

	conv_loss��o=�k6x        )��P	`�B���A�*

	conv_lossUk�=�<��        )��P	@Y�B���A�*

	conv_loss�e=.�ov        )��P	#��B���A�*

	conv_loss�x�=eSN        )��P	���B���A�*

	conv_loss��b=c��@        )��P	k�B���A�*

	conv_loss�_�=��        )��P	�!�B���A�*

	conv_loss��m=��        )��P	X\�B���A�*

	conv_lossO��=��P�        )��P	�B���A�*

	conv_loss��={v(�        )��P	ƋB���A�*

	conv_lossȳ�=���        )��P	���B���A�*

	conv_loss��=����        )��P	];�B���A�*

	conv_lossp��=�4�{        )��P	$t�B���A�*

	conv_loss�R�=>s��        )��P	���B���A�*

	conv_loss�)�=�6�        )��P	�،B���A�*

	conv_loss�[�=���        )��P	}�B���A�*

	conv_loss|w�=��>�        )��P	�P�B���A�*

	conv_lossX�=a5(        )��P	�B���A�*

	conv_loss� �=s�ڡ        )��P	���B���A�*

	conv_loss�d^=���        )��P	��B���A�*

	conv_loss�}�=��        )��P	 !�B���A�*

	conv_loss�iv=����        )��P	 U�B���A�*

	conv_loss��I=���        )��P	���B���A�*

	conv_loss�S[=h8        )��P	�B���A�*

	conv_loss��q=͟�        )��P	W��B���A�*

	conv_lossT�=O��        )��P	4,�B���A�*

	conv_loss\P=L�O        )��P	Q`�B���A�*

	conv_loss�K�=�	h�        )��P		��B���A�*

	conv_losso.:=b�        )��P	ˏB���A�*

	conv_loss�pw=7��        )��P	��B���A�*

	conv_lossgؒ=��;�        )��P	�0�B���A�*

	conv_losse=�=�"�        )��P	Rd�B���A�*

	conv_loss�=͏q�        )��P	��B���A�*

	conv_loss�=ʼv2        )��P	�̐B���A�*

	conv_loss�i=q��J        )��P	��B���A�*

	conv_loss�=Yɚ        )��P	�˒B���A�*

	conv_loss<6�=
�&p        )��P	��B���A�*

	conv_loss.X=,u�[        )��P	h.�B���A�*

	conv_lossHOl=�'�        )��P	e�B���A�*

	conv_losstN=i��        )��P	�B���A�*

	conv_loss�:m=4�q;        )��P	OݓB���A�*

	conv_lossR�;=���a        )��P	��B���A�*

	conv_loss8č=�,��        )��P	h?�B���A�*

	conv_loss�Y8=1Vn        )��P	p�B���A�*

	conv_loss�:r=��x�        )��P	���B���A�*

	conv_loss	?S=�iy        )��P	�ՔB���A�*

	conv_loss^�l=�#��        )��P	L�B���A�*

	conv_loss[�J=�{�        )��P	�4�B���A�*

	conv_loss�J=:�        )��P	d�B���A�*

	conv_loss͹=8=�        )��P	���B���A�*

	conv_loss��=7�        )��P	^��B���A�*

	conv_loss5W�=3�n        )��P	'��B���A�*

	conv_loss�z�=T�V�        )��P	1�B���A�*

	conv_loss�ҡ=�܌        )��P	}k�B���A�*

	conv_loss�[q=䅗        )��P	���B���A�*

	conv_lossQ��=V2E        )��P	YΖB���A�*

	conv_lossv/�=��9�        )��P	k��B���A�*

	conv_loss�hP=�"\        )��P	U.�B���A�*

	conv_loss�=G�?�        )��P	�_�B���A�*

	conv_loss��L=G���        )��P	��B���A�*

	conv_loss��.=��!        )��P	���B���A�*

	conv_loss1�9=F�        )��P	��B���A�*

	conv_loss���=�9��        )��P	�B���A�*

	conv_lossX��=7,�        )��P	xM�B���A�*

	conv_loss�BC=�ڡh        )��P	�}�B���A�*

	conv_losssH�=atKl        )��P	��B���A�*

	conv_loss*��=��\        )��P	�ۘB���A�*

	conv_losss+�=�P�        )��P	5�B���A�*

	conv_losscuG=����        )��P	�<�B���A�*

	conv_loss�-�=��2        )��P	�n�B���A�*

	conv_lossʐ�=_�i        )��P	���B���A�*

	conv_loss�e�=ON.{        )��P	�ΙB���A�*

	conv_loss�ob=�甪        )��P	H��B���A�*

	conv_losseP�=H=�!        )��P	�/�B���A�*

	conv_loss}�=u��c        )��P	�a�B���A�*

	conv_lossX�Z=�>��        )��P	8��B���A�*

	conv_loss��h="�mO        )��P	ÚB���A�*

	conv_lossJ�q=p0�        )��P	G��B���A�*

	conv_loss�^�=��7�        )��P	V$�B���A�*

	conv_loss�f�=��        )��P	�R�B���A�*

	conv_loss�9=���.        )��P	v��B���A�*

	conv_loss��9=�RȾ        )��P	���B���A�*

	conv_lossZGX=_        )��P	l�B���A�*

	conv_loss��=��i        )��P	f�B���A�*

	conv_lossO��=?\-        )��P	
B�B���A�*

	conv_loss�L�= ��_        )��P	��B���A�*

	conv_loss��Y=El��        )��P	б�B���A�*

	conv_lossެ�=�)tH        )��P	���B���A�*

	conv_losss|=�M�        )��P	��B���A�*

	conv_lossy2P=	�[t        )��P	�D�B���A�*

	conv_loss��8=�X�        )��P	�r�B���A�*

	conv_loss�dc=���{        )��P	���B���A�*

	conv_loss��=�XwG        )��P	�ܝB���A�*

	conv_loss<$�=�j�R        )��P	5 �B���A�*

	conv_loss㠧=_<V�        )��P	�T�B���A�*

	conv_lossF��=��ҏ        )��P	���B���A�*

	conv_loss�1_=\5#Z        )��P	���B���A�*

	conv_loss4t=�`/        )��P	��B���A�*

	conv_loss3�n=��=        )��P	@(�B���A�*

	conv_loss���=��        )��P	X�B���A�*

	conv_lossy�r=�?�        )��P	���B���A�*

	conv_lossj��=�c'd        )��P	\��B���A�*

	conv_lossѽo=<E        )��P	��B���A�*

	conv_loss�Y=j1��        )��P	��B���A�*

	conv_loss�?D=0��        )��P	�O�B���A�*

	conv_loss�Ɯ=�#>        )��P	���B���A�*

	conv_lossl�E=���        )��P	j��B���A�*

	conv_loss3a3=�e�w        )��P	��B���A�*

	conv_loss��=�
0�        )��P	�#�B���A�*

	conv_lossg&�=䞠y        )��P	�R�B���A�*

	conv_loss�vi=R��w        )��P	��B���A�*

	conv_loss��}=^4T>        )��P	���B���A�*

	conv_loss�h=&�3W        )��P	K�B���A�*

	conv_loss�\P=x��{        )��P	�B���A�*

	conv_lossG�X=2�L�        )��P	VB�B���A�*

	conv_loss�|t=@��:        )��P	�q�B���A�*

	conv_loss�С=U�J�        )��P	h��B���A�*

	conv_loss���=���        )��P	�ԢB���A�*

	conv_loss?��=��0k        )��P	�B���A�*

	conv_loss��=�״:        )��P	B2�B���A�*

	conv_loss{d�=��֪        )��P	�`�B���A�*

	conv_lossb.[=��>        )��P	ё�B���A�*

	conv_loss�l�=c���        )��P	�£B���A�*

	conv_lossAYm=+VҾ        )��P	��B���A�*

	conv_loss�Л=�1        )��P	� �B���A�*

	conv_loss�$s=+-͂        )��P	�S�B���A�*

	conv_loss�ֈ=?�"�        )��P	؄�B���A�*

	conv_loss5C=|���        )��P	���B���A�*

	conv_loss���=^�$        )��P	7�B���A�*

	conv_loss���=���h        )��P	C�B���A�*

	conv_lossR)[=�_��        )��P	T�B���A�*

	conv_loss]�=��        )��P	煥B���A�*

	conv_loss�S�=_���        )��P	o��B���A�*

	conv_loss��p=���        )��P	��B���A�*

	conv_lossmђ=N�        )��P	|�B���A�*

	conv_loss|=�v�        )��P	�\�B���A�*

	conv_loss�K=s��        )��P	D��B���A�*

	conv_loss �b=(��        )��P	�ǦB���A�*

	conv_loss��V=��p        )��P	!��B���A�*

	conv_loss��=T��        )��P	y,�B���A�*

	conv_lossg~=��]        )��P	�d�B���A�*

	conv_loss��|=��.�        )��P	|��B���A�*

	conv_loss�Q�=�a�	        )��P	)ɧB���A�*

	conv_loss��d=�v        )��P	��B���A�*

	conv_loss=���~        )��P	�<�B���A�*

	conv_lossT�=�ݐx        )��P	�u�B���A�*

	conv_losss��=(}�        )��P	���B���A�*

	conv_loss�ab=��x<        )��P	OݨB���A�*

	conv_losse��=���x        )��P	��B���A�*

	conv_lossK"�=~N "        )��P	�V�B���A�*

	conv_loss�z=�T$�        )��P	]��B���A�*

	conv_loss$�=���        )��P	s��B���A�*

	conv_losst<�=,��        )��P	��B���A�*

	conv_loss���=��6C        )��P	h#�B���A�*

	conv_loss�_=�2�        )��P	�V�B���A�*

	conv_lossi�=��n        )��P	G��B���A�*

	conv_loss�%=��        )��P	�ʪB���A�*

	conv_loss��T=�]��        )��P		��B���A�*

	conv_loss�A�=mq�        )��P	,�B���A�*

	conv_loss��=ؓB�        )��P	Z[�B���A�*

	conv_lossa�=Me/�        )��P	p��B���A�*

	conv_loss���=E"�        )��P	�«B���A�*

	conv_loss"�=c[        )��P	��B���A�*

	conv_lossa�=}iV        )��P	�%�B���A�*

	conv_loss\�4=��1        )��P	�V�B���A�*

	conv_loss�l=�tχ        )��P	׈�B���A�*

	conv_loss24b=8�\�        )��P	���B���A�*

	conv_loss��S=P���        )��P	6��B���A�*

	conv_loss�8E=|�pn        )��P	G0�B���A�*

	conv_loss��:=p&.        )��P	�c�B���A�*

	conv_loss�O�=�}	         )��P	,��B���A�*

	conv_loss��h=�M��        )��P	dƭB���A�*

	conv_loss�=5���        )��P	���B���A�*

	conv_loss�Ӗ=�0l�        )��P	�'�B���A�*

	conv_loss�A�=�b��        )��P	�X�B���A�*

	conv_loss/J=��        )��P	:��B���A�*

	conv_loss���=�N��        )��P	���B���A�*

	conv_loss�=٤        )��P	M�B���A�*

	conv_loss�=�7�         )��P		�B���A�*

	conv_loss��l=��{        )��P	_O�B���A�*

	conv_loss#s=]���        )��P	��B���A�*

	conv_lossT�p=�~        )��P	ů�B���A�*

	conv_lossv��=ܹ�[        )��P	D߯B���A�*

	conv_lossI�=���        )��P	T�B���A�*

	conv_loss�s=��G        )��P	zD�B���A�*

	conv_loss�s?=J�}        )��P	���B���A�*

	conv_lossf �=�lj�        )��P	���B���A�*

	conv_lossQ��=0         )��P	��B���A�*

	conv_loss��=�<��        )��P	,!�B���A�*

	conv_lossC�=twd        )��P	T�B���A�*

	conv_loss��=P\�"        )��P	���B���A�*

	conv_loss�v=�        )��P	��B���A�*

	conv_loss=�        )��P	���B���A�*

	conv_loss���=8	�        )��P	$9�B���A�*

	conv_loss��=.H�        )��P	Ul�B���A�*

	conv_lossJ��=ې��        )��P	8��B���A�*

	conv_lossz|�= ��        )��P	!ղB���A�*

	conv_loss�w=��G�        )��P	]	�B���A�*

	conv_loss��=��f        )��P	a8�B���A�*

	conv_loss�Z�=���        )��P	�j�B���A�*

	conv_loss?P�=,B�4        )��P	��B���A�*

	conv_lossK.=�K?        )��P	�ɳB���A�*

	conv_loss_,s=��J        )��P	 ��B���A�*

	conv_lossۡ=3��        )��P	�-�B���A�*

	conv_loss˞w=��m        )��P	�a�B���A�*

	conv_loss�ȁ=��ˍ        )��P	���B���A�*

	conv_loss�Շ={̀�        )��P	�ҴB���A�*

	conv_loss׽2=se1�        )��P	{�B���A�*

	conv_loss�l�=�
E�        )��P	�2�B���A�*

	conv_lossrv�=��&        )��P	{c�B���A�*

	conv_loss�^s=h��        )��P	 ��B���A�*

	conv_lossnR�=��0        )��P	ƵB���A�*

	conv_loss�Ȉ=}E�        )��P	���B���A�*

	conv_loss���=���        )��P	�*�B���A�*

	conv_loss'�f=~q?        )��P	�]�B���A�*

	conv_lossܤ=-.�        )��P	h��B���A�*

	conv_loss��S=�6�        )��P	r��B���A�*

	conv_loss^ N=�}��        )��P	��B���A�*

	conv_lossbN�=��Ir        )��P	�!�B���A�*

	conv_loss��=Ȃ��        )��P	:R�B���A�*

	conv_losssɁ=���o        )��P	���B���A�*

	conv_loss���=B�        )��P	���B���A�*

	conv_loss��="�!        )��P	��B���A�*

	conv_lossͪf=���b        )��P	��B���A�*

	conv_loss[�m=���        )��P	�I�B���A�*

	conv_loss��Z=^��<        )��P	ay�B���A�*

	conv_loss|IM=��;�        )��P	]��B���A�*

	conv_loss�T=�8�        )��P	�ڸB���A�*

	conv_loss���=�L��        )��P	a
�B���A�*

	conv_loss�,=C�dw        )��P	�<�B���A�*

	conv_lossۼ�=�ֺh        )��P	lo�B���A�*

	conv_lossO�=O{�        )��P	��B���A�*

	conv_loss���=b\�D        )��P	2ѹB���A�*

	conv_lossLXl==�
�        )��P	�B���A�*

	conv_loss&=�=�G�        )��P	fm�B���A�*

	conv_loss��=~��        )��P	� �B���A�*

	conv_loss�1�=p%f�        )��P	�0�B���A�*

	conv_loss�fr=0��u        )��P	`�B���A�*

	conv_loss�K�=�5ڗ        )��P	2��B���A�*

	conv_loss
��=R��j        )��P	M��B���A�*

	conv_loss+��=]�J�        )��P	���B���A�*

	conv_loss,��=� �        )��P	�3�B���A�*

	conv_loss��q=��~@        )��P	^h�B���A�*

	conv_loss NE=�U�+        )��P	��B���A�*

	conv_loss�Ō=t�        )��P	[��B���A�*

	conv_loss<x=�%	        )��P	\�B���A�*

	conv_loss=�\=�(�        )��P	�;�B���A�*

	conv_loss�و=x��        )��P	�j�B���A�*

	conv_loss�v=���        )��P	8��B���A�*

	conv_lossKc�=�Av�        )��P	7��B���A�*

	conv_loss�B=���        )��P	a�B���A�*

	conv_loss;��=x��	        )��P	�=�B���A�*

	conv_lossȝZ=�4�-        )��P	gn�B���A�*

	conv_lossb`=UHh�        )��P	'��B���A�*

	conv_loss��]=ğ��        )��P	L��B���A�*

	conv_loss��9=���D        )��P	�B���A�*

	conv_loss�g=�#�6        )��P	�<�B���A�*

	conv_lossQzb=K�6�        )��P	�m�B���A�*

	conv_loss��Z=m��        )��P	O��B���A�*

	conv_loss�5C=E��h        )��P	��B���A�*

	conv_lossI�=u�k        )��P	=��B���A�*

	conv_loss<D6=��Pp        )��P	�,�B���A�*

	conv_lossg��=z97        )��P	[�B���A�*

	conv_lossAZ�=ml0�        )��P	3��B���A�*

	conv_loss���=[�        )��P	���B���A�*

	conv_loss�	�=Vq�l        )��P	���B���A�*

	conv_loss�S�=?���        )��P	�B���A�*

	conv_loss�5x=T'�S        )��P	?I�B���A�*

	conv_loss)�=pM��        )��P	�x�B���A�*

	conv_losst��=���3        )��P	3��B���A�*

	conv_loss� ;=L?q.        )��P	���B���A�*

	conv_loss��h=w�ZO        )��P	�B���A�*

	conv_loss��=��e�        )��P	�5�B���A�*

	conv_loss��=���        )��P	d�B���A�*

	conv_lossD=���        )��P	Ւ�B���A�*

	conv_lossqv=�yA�        )��P	&��B���A�*

	conv_loss�v�=�a"        )��P	���B���A�*

	conv_lossI��=a��'        )��P	��B���A�*

	conv_loss��D=�oE#        )��P	oL�B���A�*

	conv_lossf�u=�bE        )��P	{�B���A�*

	conv_loss��C=�	z�        )��P	[��B���A�*

	conv_loss�&�=��WJ        )��P	���B���A�*

	conv_loss���='�:R        )��P	��B���A�*

	conv_lossI��=�{        )��P	n?�B���A�*

	conv_lossd��=��a�        )��P	7o�B���A�*

	conv_lossu��=�`        )��P	��B���A�*

	conv_loss��k=��:        )��P	��B���A�*

	conv_loss�>p= G�        )��P	�B���A�*

	conv_loss��=�֚�        )��P	�C�B���A�*

	conv_loss�L�=FE}        )��P	}�B���A�*

	conv_loss�P�=�
k        )��P	���B���A�*

	conv_loss=Cp=bi�i        )��P	��B���A�*

	conv_loss�4w=<ҏY        )��P	��B���A�*

	conv_loss��=04�        )��P	4]�B���A�*

	conv_lossK�=�w��        )��P	J��B���A�*

	conv_loss2�C=4K�        )��P	a��B���A�*

	conv_lossͼ�=��        )��P	���B���A�*

	conv_loss�w�=��2�        )��P	��B���A�*

	conv_loss�@�=�Z7�        )��P	�P�B���A�*

	conv_loss�z=�D�?        )��P	O��B���A�*

	conv_loss�W�=��C        )��P	��B���A�*

	conv_loss��=!��F        )��P	���B���A�*

	conv_loss,�]=Y���        )��P	g�B���A�*

	conv_loss��H=Mr'        )��P	sH�B���A�*

	conv_loss�R=O��        )��P	�x�B���A�*

	conv_loss���=XԜ>        )��P	��B���A�*

	conv_loss��=�Ej;        )��P	���B���A�*

	conv_loss>��=	�N        )��P	��B���A�*

	conv_loss;�=2'y�        )��P	�F�B���A�*

	conv_lossK{�=f�        )��P	�u�B���A�*

	conv_loss?��=����        )��P	���B���A�*

	conv_loss�
�=��        )��P	U��B���A�*

	conv_loss���=�bo6        )��P	@�B���A�*

	conv_loss@3==��m        )��P	�1�B���A�*

	conv_lossMQW=1Z�        )��P	!a�B���A�*

	conv_loss�W=��;        )��P	���B���A�*

	conv_lossMw�=9���        )��P	��B���A�*

	conv_loss{0{=TN�        )��P	���B���A�*

	conv_loss Rg=��iR        )��P	X�B���A�*

	conv_loss�C�=8_�        )��P	�Q�B���A�*

	conv_loss���=l�%U        )��P	l��B���A�*

	conv_lossqp=Ej��        )��P	���B���A�*

	conv_loss#�s=ʀ4        )��P	��B���A�*

	conv_loss�o�=F��W        )��P	V�B���A�*

	conv_loss%�m=���        )��P	�?�B���A�*

	conv_lossJ�a=���A        )��P	�p�B���A�*

	conv_loss�֔=M9B        )��P	 ��B���A�*

	conv_loss쒣=��        )��P	@��B���A�*

	conv_loss�O=����        )��P	;�B���A�*

	conv_loss�P�=�9�\        )��P	�2�B���A�*

	conv_loss	4�=N�)        )��P	6b�B���A�*

	conv_lossn�W=b�I�        )��P	��B���A�*

	conv_loss�5=�Ěk        )��P	t��B���A�*

	conv_loss�2�=2{��        )��P	��B���A�*

	conv_loss�S=
�1�        )��P	�!�B���A�*

	conv_loss��x=����        )��P	a�B���A�*

	conv_loss�x�=�N�w        )��P	��B���A�*

	conv_loss���=FxT        )��P	���B���A�*

	conv_lossg�J=̡nn        )��P	���B���A�*

	conv_loss��=h�+�        )��P	�$�B���A�*

	conv_lossl�T=��N�        )��P	�U�B���A�*

	conv_loss��&=�r��        )��P	���B���A�*

	conv_loss`]�=b?O        )��P	ٿ�B���A�*

	conv_loss`n=�^�        )��P	H��B���A�*

	conv_lossyί=%)9B        )��P	j+�B���A�*

	conv_loss!�=6��H        )��P	.d�B���A�*

	conv_loss2*u=S?�        )��P	+��B���A�*

	conv_lossG��=
�N        )��P	���B���A�*

	conv_lossH�=�u        )��P	�B���A�*

	conv_loss��=�tW        )��P	�3�B���A�*

	conv_lossXFq=:q�        )��P	b�B���A�*

	conv_loss�z=4ɽA        )��P	Q��B���A�*

	conv_loss�@j=8 �        )��P	���B���A�*

	conv_lossy��=E>	�        )��P	���B���A�*

	conv_loss�):=H�+�        )��P	��B���A�*

	conv_lossP7�="�0        )��P	GM�B���A�*

	conv_loss�W=���        )��P	�}�B���A�*

	conv_lossJl=q��h        )��P	��B���A�*

	conv_loss�N{=|�e(        )��P	���B���A�*

	conv_loss�R�=zk        )��P	��B���A�*

	conv_loss�ǯ=�s��        )��P	5O�B���A�*

	conv_lossu)�=A:�d        )��P	&�B���A�*

	conv_loss��o=]K        )��P	��B���A�*

	conv_loss���= �$9        )��P	���B���A�*

	conv_lossW�p=���        )��P	��B���A�*

	conv_losst�Y=��B        )��P	�<�B���A�*

	conv_lossŌr=1���        )��P	�k�B���A�*

	conv_loss�6=Y�        )��P	���B���A�*

	conv_loss��B=$(�*        )��P	���B���A�*

	conv_loss��f=���        )��P	���B���A�*

	conv_loss��}=���        )��P	y,�B���A�*

	conv_loss.#b=��4]        )��P	[�B���A�*

	conv_loss�V=
�r        )��P	��B���A�*

	conv_loss���=\h        )��P	L��B���A�*

	conv_losso+0=׾W�        )��P	���B���A�*

	conv_loss�:�=�L        )��P	m�B���A�*

	conv_loss?
W=W5�#        )��P	
K�B���A�*

	conv_lossv��=��>        )��P	|�B���A�*

	conv_loss��`=ؽ        )��P	a��B���A�*

	conv_lossǕ =�y��        )��P	���B���A�*

	conv_loss �W=��l�        )��P	z�B���A�*

	conv_loss�G:=l�Ls        )��P	YB�B���A�*

	conv_loss�i�=�j��        )��P	ft�B���A�*

	conv_lossފ@=H	��        )��P	C��B���A�*

	conv_lossH(�=��        )��P	���B���A�*

	conv_lossa=R��        )��P	��B���A�*

	conv_loss�i=�vh        )��P	C�B���A�*

	conv_loss��=;?        )��P	�t�B���A�*

	conv_loss�;}=�H�        )��P	���B���A�*

	conv_lossp�=�76        )��P	<��B���A�*

	conv_loss��~="�        )��P	��B���A�*

	conv_lossJ�M=����        )��P	*B�B���A�*

	conv_loss���=��5�        )��P	�q�B���A�*

	conv_loss�fs=p9��        )��P	M��B���A�*

	conv_lossh�O=�1��        )��P	]��B���A�*

	conv_loss�=����        )��P	Q�B���A�*

	conv_loss �V=�5��        )��P	�G�B���A�*

	conv_loss��t=#T�7        )��P	_��B���A�*

	conv_loss�.=�R�f        )��P	N��B���A�*

	conv_loss��S=ٜ��        )��P	���B���A�*

	conv_loss���=�QH&        )��P	��B���A�*

	conv_loss<H�=��'I        )��P	�@�B���A�*

	conv_loss^w�=U��        )��P	Up�B���A�*

	conv_loss�ut=ot�        )��P	,��B���A�*

	conv_losso=�V�        )��P	���B���A�*

	conv_loss��.=���"        )��P	���B���A�*

	conv_loss�5=�D�         )��P	.�B���A�*

	conv_lossLJy=g�Gr        )��P	6_�B���A�*

	conv_loss�#0=���        )��P	ӎ�B���A�*

	conv_lossjXO=��0        )��P	9��B���A�*

	conv_loss�.=5��        )��P	���B���A�*

	conv_lossUц=P�E        )��P	��B���A�*

	conv_loss)��=a��^        )��P	�M�B���A�*

	conv_loss�[7=3o	_        )��P	\}�B���A�*

	conv_loss�?�=���        )��P	3��B���A�*

	conv_loss=�=\� �        )��P	j��B���A�*

	conv_loss��T=+gŊ        )��P		�B���A�*

	conv_losszl�=�K0        )��P	�:�B���A�*

	conv_loss��a=ͩI        )��P	Mh�B���A�*

	conv_loss<�f=����        )��P	��B���A�*

	conv_loss5�\=�nY        )��P	��B���A�*

	conv_loss�=�O�        )��P	!��B���A�*

	conv_lossJ?w=q��G        )��P	�'�B���A�*

	conv_loss�`�=߃pQ        )��P	>W�B���A�*

	conv_loss�:�=�	U�        )��P	��B���A�*

	conv_loss(��=hb<�        )��P	(��B���A�*

	conv_loss�ŕ=�?t�        )��P	\��B���A�*

	conv_loss�=;�p        )��P	��B���A�*

	conv_lossVI�=���}        )��P	�D�B���A�*

	conv_loss�W=Nȑ�        )��P	�s�B���A�*

	conv_lossd:=T��        )��P	���B���A�*

	conv_loss���=��d        )��P	R��B���A�*

	conv_loss1�D=��C        )��P	��B���A�*

	conv_loss�r=f�"        )��P	�2�B���A�*

	conv_lossڔ�=&=L        )��P	b�B���A�*

	conv_lossHf=#�3�        )��P	N��B���A�*

	conv_lossS=�	        )��P	�!�B���A�*

	conv_loss߬�=�06
        )��P	;R�B���A�*

	conv_loss��I=)��        )��P	���B���A�*

	conv_lossj"m=���        )��P	���B���A�*

	conv_loss.y2=����        )��P	��B���A�*

	conv_lossJ�u=��        )��P	��B���A�*

	conv_loss�Tw=q�0        )��P	S^�B���A�*

	conv_loss�=�D�        )��P	d��B���A�*

	conv_lossIz�=�!        )��P	N��B���A�*

	conv_loss�Kf=��        )��P	���B���A�*

	conv_loss�s�=$�B�        )��P	�+�B���A�*

	conv_loss�u�=<~�:        )��P	bg�B���A�*

	conv_loss��C=�d�        )��P	Z��B���A�*

	conv_lossr=�?�g        )��P	(��B���A�*

	conv_lossڈq=1:5
        )��P	$��B���A�*

	conv_loss�(=|��        )��P	$#�B���A�*

	conv_lossY�b='���        )��P	zR�B���A�*

	conv_loss��a=us�        )��P	��B���A�*

	conv_losss}�=���        )��P	V��B���A�*

	conv_loss>n=����        )��P	F��B���A�*

	conv_lossyO=ǂ;        )��P	"�B���A�*

	conv_loss�d=Fq        )��P	cU�B���A�*

	conv_loss�z=ۺ�        )��P	m��B���A�*

	conv_loss.�=�j�S        )��P	���B���A�*

	conv_loss�Q�=��c        )��P	���B���A�*

	conv_lossO#�=+[��        )��P	��B���A�*

	conv_loss�+�=��|�        )��P	eL�B���A�*

	conv_loss��S=��        )��P	�B���A�*

	conv_loss��g=:��        )��P	u��B���A�*

	conv_lossp��=��Xv        )��P	���B���A�*

	conv_loss/8@=v]k        )��P	v�B���A�*

	conv_loss��_=^9d�        )��P	x;�B���A�*

	conv_loss&EQ=U��Y        )��P	Vl�B���A�*

	conv_lossMj�=��M�        )��P	9��B���A�*

	conv_loss��P=��~        )��P	���B���A�*

	conv_loss=�E=��}        )��P	d�B���A�*

	conv_lossѾt=c�E-        )��P	U6�B���A�*

	conv_loss��=؞(�        )��P	lh�B���A�*

	conv_loss�2=�;        )��P	`��B���A�*

	conv_loss"#R=�        )��P	���B���A�*

	conv_loss�|�=~Xg�        )��P	���B���A�*

	conv_loss#rE=��5        )��P	�,�B���A�*

	conv_loss@R=��E�        )��P	�[�B���A�*

	conv_loss)�=rm�E        )��P	��B���A�*

	conv_loss���=���        )��P	���B���A�*

	conv_loss��n=p!�u        )��P	���B���A�*

	conv_loss�b=8��        )��P	�B���A�*

	conv_loss_pq=P�+n        )��P	�M�B���A�*

	conv_loss��#=$-��        )��P	N~�B���A�*

	conv_loss4�N=�1LB        )��P	L��B���A�*

	conv_loss���=���        )��P	0��B���A�*

	conv_lossjQ�=���!        )��P	�"�B���A�*

	conv_loss��=8��        )��P	�X�B���A�*

	conv_lossp/=)T�        )��P	)��B���A�*

	conv_loss�r=��%�        )��P	a��B���A�*

	conv_lossy{7=���        )��P	��B���A�*

	conv_loss�$�==��        )��P	��B���A�*

	conv_loss���=!��        )��P	K�B���A�*

	conv_loss�'{=��{�        )��P	��B���A�*

	conv_lossmT�=�y��        )��P	���B���A�*

	conv_loss�kh=ݸ��        )��P	���B���A�*

	conv_lossHa=rR��        )��P	i$�B���A�*

	conv_loss5!q=`R�        )��P	�T�B���A�*

	conv_loss=~=3ǂ        )��P	7��B���A�*

	conv_lossN�B=�T�/        )��P	Ͼ�B���A�*

	conv_loss��R=���3        )��P	��B���A�*

	conv_loss+�=��N        )��P	&�B���A�*

	conv_loss�p�=ޠ�	        )��P	ZP�B���A�*

	conv_loss�V=���l        )��P	-��B���A�*

	conv_loss�_=t�g�        )��P	޴�B���A�*

	conv_loss�x=`�:�        )��P	>��B���A�*

	conv_loss<#8=�Uq;        )��P	��B���A�*

	conv_lossu�H= N@        )��P	�[�B���A�*

	conv_loss�9�=�|�1        )��P	��B���A�*

	conv_loss��I=?=��        )��P	ƽ�B���A�*

	conv_loss���=o�        )��P	���B���A�*

	conv_loss�/7=�d<�        )��P	Y%�B���A�*

	conv_loss/SX=�CU�        )��P	�Y�B���A�*

	conv_losss	*=io�"        )��P	��B���A�*

	conv_loss�O=�7        )��P	���B���A�*

	conv_lossxb�="~80        )��P	���B���A�*

	conv_loss�==�bc        )��P	w?�B���A�*

	conv_loss�f=q��        )��P	�z�B���A�*

	conv_loss��4=�߄        )��P	M��B���A�*

	conv_loss/�=79�i        )��P	���B���A�*

	conv_loss`wZ=�@�@        )��P	�.�B���A�*

	conv_loss+ p=H%��        )��P	6k�B���A�*

	conv_lossT=+і        )��P	���B���A�*

	conv_loss�GS=�ŭM        )��P	U��B���A�*

	conv_loss��=H@+�        )��P	>�B���A�*

	conv_lossT�Y=���5        )��P	�<�B���A�*

	conv_losset=�%�,        )��P	|m�B���A�*

	conv_lossˌH=RĘ	        )��P	���B���A�*

	conv_loss�==<�r�        )��P	C��B���A�*

	conv_losss��=�c��        )��P	� �B���A�*

	conv_loss�T�=XND�        )��P	�7�B���A�*

	conv_loss�wk=���        )��P	~|�B���A�*

	conv_loss�Ε=;��        )��P	o��B���A�*

	conv_loss�Ir=��'        )��P	6��B���A�*

	conv_loss��=�!�        )��P	/�B���A�*

	conv_lossb"=�K��        )��P	�O�B���A�*

	conv_loss��M=j�GD        )��P	d}�B���A�*

	conv_loss�^`=WD��        )��P	ˬ�B���A�*

	conv_loss"&�=��:B        )��P	8��B���A�*

	conv_loss(�Z="Ώ        )��P	p�B���A�*

	conv_lossLU4=;��        )��P		M�B���A�*

	conv_loss�,^=H���        )��P	$��B���A�*

	conv_loss�_=zC��        )��P	$��B���A�*

	conv_loss���=� �        )��P	���B���A�*

	conv_lossem<=�ˑ        )��P	�*�B���A�*

	conv_lossޠ,=Ԉ�^        )��P	1a�B���A�*

	conv_loss]j=�3�        )��P	��B���A�*

	conv_loss��Z=A��        )��P	���B���A�*

	conv_lossN�U=��G�        )��P	��B���A�*

	conv_loss���=L�w        )��P	� �B���A�*

	conv_loss���=z��        )��P	P�B���A�*

	conv_loss~N=��        )��P	���B���A�*

	conv_loss��=�#�`        )��P	ʰ�B���A�*

	conv_loss���=%��3        )��P	2��B���A�*

	conv_loss���==�&P        )��P	 C���A�*

	conv_loss�Ð=�B        )��P	�M C���A�*

	conv_loss%�=�%�        )��P	�� C���A�*

	conv_loss<(R=�N        )��P	�� C���A�*

	conv_lossiA^=�j��        )��P	`� C���A�*

	conv_lossҦn=�Ⲡ        )��P	�C���A�*

	conv_loss"�=�k|M        )��P	y?C���A�*

	conv_loss��=��EO        )��P	�lC���A�*

	conv_loss�DX=1F��        )��P	�C���A�*

	conv_lossW&L=ЦD        )��P	Q�C���A�*

	conv_loss�̍=ȥ��        )��P	y�C���A�*

	conv_loss�z�=6�B*        )��P	�)C���A�*

	conv_loss�IU=.���        )��P	�WC���A�*

	conv_loss��E=c��g        )��P	C�C���A�*

	conv_loss�Z=)��        )��P	v�C���A�*

	conv_lossi��=*`�{        )��P	W�C���A�*

	conv_loss?�f=�`�[        )��P	]C���A�*

	conv_loss�Ǆ=*��M        )��P	�ZC���A�*

	conv_lossif=��W        )��P	p�C���A�*

	conv_loss��(=3�>[        )��P	��C���A�*

	conv_loss��=�v�8        )��P	 �C���A�*

	conv_loss�<m=*|        )��P	E!C���A�*

	conv_loss3�[=;/        )��P	�cC���A�*

	conv_loss���=}Q        )��P	��C���A�*

	conv_loss�gy=��W�        )��P	��C���A�*

	conv_lossl�~=�l��        )��P	��C���A�*

	conv_loss�"�=�&��        )��P	u2C���A�*

	conv_loss��I=��8�        )��P	"sC���A�*

	conv_loss^u�=�	�2        )��P	ݣC���A�*

	conv_loss�z�=4k��        )��P	��C���A�*

	conv_loss$�O=���v        )��P	�C���A�*

	conv_loss��u=k�-        )��P	IC���A�*

	conv_loss�+�=�        )��P	ԂC���A�*

	conv_lossZP=���i        )��P	R�C���A�*

	conv_lossDx=W6        )��P	��C���A�*

	conv_loss�6�=u��        )��P	�C���A�*

	conv_loss9�=y�w�        )��P	pLC���A�*

	conv_loss}��=;�ղ        )��P	G�C���A�*

	conv_losse҄=X��        )��P	L�C���A�*

	conv_lossn[b=���z        )��P	dC���A�*

	conv_lossKϟ=<�ܦ        )��P	h9C���A�*

	conv_loss��=��#        )��P	�qC���A�*

	conv_loss�J=h	��        )��P	֪C���A�*

	conv_losssD:=g��        )��P	$�C���A�*

	conv_loss�p�=Suuj        )��P	�"	C���A�*

	conv_loss��=���        )��P	�\	C���A�*

	conv_loss�h=y��        )��P	=�	C���A�*

	conv_loss��+=��;C        )��P	^�	C���A�*

	conv_loss��u=��`        )��P	��	C���A�*

	conv_loss��4=7�y        )��P	�2
C���A�*

	conv_loss,�H=�~�        )��P	ot
C���A�*

	conv_loss�oZ=�@��        )��P	�
C���A�*

	conv_loss.��=2�        )��P	)&C���A�*

	conv_loss�o�=��FH        )��P	qfC���A�*

	conv_loss6@�=��        )��P	A�C���A�*

	conv_lossI��=|D��        )��P	��C���A�*

	conv_lossw$�=:~W�        )��P	�C���A�*

	conv_loss�ٔ=Cp�        )��P	�%C���A�*

	conv_loss��^=ͥ�)        )��P	�fC���A�*

	conv_loss6�T=�6�        )��P	˙C���A�*

	conv_lossc8Y=�Dk�        )��P	��C���A�*

	conv_loss�u�=f�S�        )��P	�C���A�*

	conv_loss�k=��Y�        )��P	�0C���A�*

	conv_loss��=�3��        )��P	�jC���A�*

	conv_loss!U=�	��        )��P	۰C���A�*

	conv_loss�_�=Jx[�        )��P	��C���A�*

	conv_loss/�R=9,T�        )��P	nC���A�*

	conv_loss�mD=�* �        )��P	AC���A�*

	conv_lossa1=:?��        )��P	�C���A�*

	conv_loss��o=J�aB        )��P	|�C���A�*

	conv_lossz��=v
         )��P	<�C���A�*

	conv_lossd�Z=Q��        )��P	�C���A�*

	conv_loss#[=W\._        )��P	cDC���A�*

	conv_lossAXS=$̦        )��P	C���A�*

	conv_loss6iR=)�I�        )��P	��C���A�*

	conv_loss��=��W�        )��P	[�C���A�*

	conv_lossC� =_oj        )��P	�C���A�*

	conv_loss|?=]�q�        )��P	�=C���A�*

	conv_loss�d�=g`�@        )��P	�yC���A�*

	conv_losse��=u�:        )��P	o�C���A�*

	conv_lossw�=��        )��P	~AC���A�*

	conv_loss�ȅ=�1i�        )��P	�sC���A�*

	conv_loss�C=�-$�        )��P	S�C���A�*

	conv_loss��s=f�R�        )��P	��C���A�*

	conv_loss���=>��        )��P	�2C���A�*

	conv_lossr�C=��}�        )��P	EgC���A�*

	conv_loss7�=M�i+        )��P	;�C���A�*

	conv_lossQ�C=�in        )��P	~�C���A�*

	conv_lossu�h=���        )��P	&"C���A�*

	conv_loss�rT=��m�        )��P	�YC���A�*

	conv_loss��M=�B��        )��P	��C���A�*

	conv_loss�QN=Z�=�        )��P	H�C���A�*

	conv_lossZ�4=���        )��P	��C���A�*

	conv_losss��=b�%1        )��P	�1C���A�*

	conv_loss�m=y,�        )��P	SkC���A�*

	conv_lossg�f=���C        )��P	��C���A�*

	conv_lossH� =�cv�        )��P	��C���A�*

	conv_loss�ł=1G        )��P	&C���A�*

	conv_loss0�Z==7�        )��P	19C���A�*

	conv_lossUɗ=Ul��        )��P	�qC���A�*

	conv_lossV=�מ        )��P	�C���A�*

	conv_loss߀p=DI��        )��P	��C���A�*

	conv_loss"Ԙ=8:|�        )��P	C���A�*

	conv_loss�
=$!�        )��P	jHC���A�*

	conv_loss�s =d�~{        )��P	�wC���A�*

	conv_loss\��=��V         )��P	�C���A�*

	conv_lossh>R=y]>�        )��P	h�C���A�*

	conv_loss�4o=�Nm@        )��P	�C���A�*

	conv_loss�IY=<�d        )��P	:GC���A�*

	conv_lossZSP=�!%V        )��P	CzC���A�*

	conv_loss��v=6Љ        )��P	ͫC���A�*

	conv_loss�`�=��        )��P	��C���A�*

	conv_loss�HS=��?o        )��P	FC���A�*

	conv_loss�u�=t�@�        )��P	�FC���A�*

	conv_loss=JG=0ƍ�        )��P	�|C���A�*

	conv_loss��]=��O�        )��P	��C���A�*

	conv_loss�e�={pQ        )��P	��C���A�*

	conv_loss��m=����        )��P	#C���A�*

	conv_loss�7�=��3�        )��P	RVC���A�*

	conv_lossD�=9V�        )��P	6�C���A�*

	conv_loss0C=�D*
        )��P	\�C���A�*

	conv_loss��L=ͨ��        )��P	��C���A�*

	conv_loss�k�=7��        )��P	�'C���A�*

	conv_loss�d=��>        )��P	�WC���A�*

	conv_loss^-|=�        )��P	r�C���A�*

	conv_loss>�4=�g�V        )��P	��C���A�*

	conv_loss �b=�ea        )��P	��C���A�*

	conv_losss��=��8a        )��P	_0C���A�*

	conv_loss�A=�� �        )��P	bC���A�*

	conv_loss��=U���        )��P	@�C���A�*

	conv_loss%\=A1<        )��P	��C���A�*

	conv_loss��=����        )��P	LC���A�*

	conv_loss>6=F"/        )��P	v5C���A�*

	conv_lossX5=*�        )��P	$cC���A�*

	conv_loss<��=���        )��P	ЖC���A�*

	conv_lossg��=�c��        )��P	��C���A�*

	conv_loss��=��H�        )��P	��C���A�*

	conv_loss{9�=����        )��P	0C���A�*

	conv_loss�T=c�E�        )��P	�bC���A�*

	conv_loss��U=�$�        )��P	T�C���A�*

	conv_loss��l=�#��        )��P	>�C���A�*

	conv_lossX�F=�!g�        )��P	/�C���A�*

	conv_loss��4=� D        )��P	8C���A�*

	conv_lossD�=�=�w        )��P	�fC���A�*

	conv_loss�u�=����        )��P	�C���A�*

	conv_loss��=�c�        )��P	I�C���A�*

	conv_loss���=У�        )��P	��C���A�*

	conv_loss��=��^        )��P	t$ C���A�*

	conv_loss�JE=j�(F        )��P	S C���A�*

	conv_lossK�B=(�=�        )��P	Ђ C���A�*

	conv_loss�T=^]؇        )��P	� C���A�*

	conv_lossU=_WIL        )��P	�� C���A�*

	conv_lossVڂ=j͚        )��P	A!C���A�*

	conv_loss��`=`�\�        )��P	K!C���A�*

	conv_loss�Ms=w��"        )��P	�{!C���A�*

	conv_loss��=i��?        )��P	%�!C���A�*

	conv_loss�l
=���^        )��P	$�!C���A�*

	conv_loss�p:=
��
        )��P	 "C���A�*

	conv_loss�v=-��        )��P	�B"C���A�*

	conv_loss�	t=�3^        )��P	�q"C���A�*

	conv_loss"M�=�P&K        )��P	O�"C���A�*

	conv_lossn�=ۉ�^        )��P	��"C���A�*

	conv_loss��r=nh��        )��P	� #C���A�*

	conv_loss�q=�%$Q        )��P	i/#C���A�*

	conv_loss//D=k�=�        )��P	L^#C���A�*

	conv_loss��'=e�o�        )��P	�#C���A�*

	conv_lossƛm=Z��        )��P	��#C���A�*

	conv_loss@�=���9        )��P	��#C���A�*

	conv_loss��F= ���        )��P	�$C���A�*

	conv_loss���="	�        )��P	�J$C���A�*

	conv_loss|�=���f        )��P	�z$C���A�*

	conv_lossQ/==�Y        )��P	�$C���A�*

	conv_lossK��=Fq)        )��P	+�$C���A�*

	conv_loss��'=�m�'        )��P	%C���A�*

	conv_lossV�1=8��        )��P	$?%C���A�*

	conv_lossJ�= �,        )��P	o%C���A�*

	conv_loss� .=� 0        )��P	9�%C���A�*

	conv_lossW�4=�B��        )��P	�%C���A�*

	conv_loss�0Z=��}        )��P	S&C���A�*

	conv_loss�S5=17��        )��P	H�*C���A�*

	conv_loss��M=�1L        )��P	>�*C���A�*

	conv_loss��y=���        )��P	B+C���A�*

	conv_loss�K=n        )��P	|6+C���A�*

	conv_lossh|z=��z;        )��P	g+C���A�*

	conv_loss��p=�1��        )��P	�+C���A�*

	conv_losstd`=�OH        )��P	��+C���A�*

	conv_loss)�N=
��        )��P	o�+C���A�*

	conv_loss�\l=�e�        )��P	:-,C���A�*

	conv_lossޙ4=免�        )��P	s\,C���A�*

	conv_loss�Vw= U��        )��P	�,C���A�*

	conv_lossZʀ=���        )��P	s�,C���A�*

	conv_losss4=)�        )��P	|�,C���A�*

	conv_lossK�b=#�v        )��P	�"-C���A�*

	conv_losse�=���        )��P	)Q-C���A�*

	conv_loss�r=m��q        )��P	�-C���A�*

	conv_loss�&S=s?�X        )��P	��-C���A�*

	conv_loss��P=2�3*        )��P	�-C���A�*

	conv_loss�$=���        )��P	X.C���A�*

	conv_loss�]f=}D�        )��P	L.C���A�*

	conv_lossX)=�X\s        )��P	~.C���A�*

	conv_lossOa*=����        )��P	լ.C���A�*

	conv_loss�Ń=�5��        )��P	%�.C���A�*

	conv_loss���=� �        )��P	
/C���A�*

	conv_loss�F6=g6&�        )��P	6=/C���A�*

	conv_loss8.�=f�Ӭ        )��P	n/C���A�*

	conv_lossmu=ﱜ�        )��P	l�/C���A�*

	conv_loss�s=���        )��P	\�/C���A�*

	conv_loss\'=no�        )��P	i�/C���A�*

	conv_lossL=9���        )��P	2,0C���A�*

	conv_loss !U=�]�        )��P	[0C���A�*

	conv_lossq){=�>�?        )��P	��0C���A�*

	conv_lossL� =�նG        )��P	ɽ0C���A�*

	conv_loss{�X=U͢H        )��P	$�0C���A�*

	conv_loss�uh=�H�        )��P	1C���A�*

	conv_losse�z= ��        )��P	HR1C���A�*

	conv_lossZJ=�ze9        )��P	ق1C���A�*

	conv_loss=$z��        )��P	��1C���A�*

	conv_loss�T =�׃�        )��P	��1C���A�*

	conv_loss*:=�1yH        )��P	2C���A�*

	conv_loss!�<=��l        )��P	�H2C���A�*

	conv_lossD1=
z`�        )��P	ly2C���A�*

	conv_loss�T`=D��        )��P	s�2C���A�*

	conv_loss`W~=�"�w        )��P	��2C���A�*

	conv_loss �U=�Ԟ�        )��P	!3C���A�*

	conv_loss9L�=%�To        )��P	�=3C���A�*

	conv_lossI~t=6OEJ        )��P	�n3C���A�*

	conv_loss�:=���f        )��P	��3C���A�*

	conv_loss/A[=�#��        )��P	'�3C���A�*

	conv_lossա�=ƴ^�        )��P	#4C���A�*

	conv_loss���=|��        )��P	�64C���A�*

	conv_lossu�a=l�&�        )��P	Qy4C���A�*

	conv_loss�!r=>Q$        )��P	۩4C���A�*

	conv_loss��/=$���        )��P	��4C���A�*

	conv_loss'�w=���        )��P	�5C���A�*

	conv_loss��3=��X�        )��P	PC5C���A�*

	conv_lossUB�=�f��        )��P	Mu5C���A�*

	conv_loss��d=�M�        )��P	ަ5C���A�*

	conv_losst�6=��:s        )��P	��5C���A�*

	conv_loss��Z=;���        )��P	�6C���A�*

	conv_loss��U=�N�        )��P	3O6C���A�*

	conv_loss*�= 鱓        )��P	��6C���A�*

	conv_loss�Um=k6.h        )��P	��6C���A�*

	conv_lossA�Q=�x�        )��P	q�6C���A�*

	conv_lossʤ�=�/	�        )��P	F$7C���A�*

	conv_loss�z~=�/(�        )��P	�V7C���A�*

	conv_loss��w=��@�        )��P	�7C���A�*

	conv_lossQ�U=�$=�        )��P	�7C���A�*

	conv_loss5�=��        )��P	]�7C���A�*

	conv_loss�	l=u��        )��P	0&8C���A�*

	conv_loss�R`=�5�        )��P		Y8C���A�*

	conv_loss|o=�z�        )��P	��8C���A�*

	conv_lossL@�=r�V�        )��P	z�8C���A�*

	conv_loss҇=z�3�        )��P	��8C���A�*

	conv_loss!@`=���        )��P	�!9C���A�*

	conv_lossj��=̠�        )��P	aX9C���A�*

	conv_loss��|=��6�        )��P	�9C���A�*

	conv_loss��=??{�        )��P	{�9C���A�*

	conv_loss��=�.�        )��P	��9C���A�*

	conv_loss =�ĝ�        )��P	\8:C���A�*

	conv_loss�%6=E�j`        )��P	�|:C���A�*

	conv_lossv�)=��D�        )��P	��:C���A�*

	conv_loss��=��y�        )��P	��:C���A�*

	conv_loss�|�="�ذ        )��P	W;C���A�*

	conv_loss�/=��Wq        )��P	�F;C���A�*

	conv_loss<�v=/�        )��P	�z;C���A�*

	conv_loss��R=3�2        )��P	}�;C���A�*

	conv_loss1�J=-�k        )��P	0�;C���A�*

	conv_loss��=��o        )��P	A<C���A�*

	conv_loss;x=�"�        )��P	�J<C���A�*

	conv_lossj.=xT9P        )��P	%<C���A�*

	conv_lossT:_=w�{        )��P	�<C���A�*

	conv_loss/�=���        )��P	��<C���A�*

	conv_loss�Ϧ=���        )��P	�=C���A�*

	conv_lossC�=�%��        )��P	�B=C���A�*

	conv_loss"�=�O�j        )��P	�q=C���A�*

	conv_loss�=���)        )��P	k�=C���A�*

	conv_loss��=/�w�        )��P	�=C���A�*

	conv_lossxc`=��o        )��P	:>C���A�*

	conv_lossń(=[�Ox        )��P	�1>C���A�*

	conv_loss��l=�Y�9        )��P	�`>C���A�*

	conv_loss��|=��Ӭ        )��P	�?C���A�*

	conv_loss���=���[        )��P	�#@C���A�*

	conv_loss��j=ȶ�E        )��P	U@C���A�*

	conv_lossg�u=�	��        )��P	�@C���A�*

	conv_loss��<�=g        )��P	N�@C���A�*

	conv_lossBJk=��G�        )��P	��@C���A�*

	conv_loss��=M �        )��P	~AC���A�*

	conv_loss�k\=(<<�        )��P	�NAC���A�*

	conv_loss���=����        )��P	ȋAC���A�*

	conv_loss�y�=���        )��P	��AC���A�*

	conv_lossr6m=�-�        )��P	j�AC���A�*

	conv_loss�-a=t/xm        )��P	Z;BC���A�*

	conv_loss׷v=�_Z�        )��P	�jBC���A�*

	conv_loss�NC=�9c�        )��P	��BC���A�*

	conv_loss�=� 	        )��P	��BC���A�*

	conv_loss�=��p�        )��P	=�BC���A�*

	conv_loss�R=��T�        )��P	�'CC���A�*

	conv_lossD[9=q�`�        )��P	qVCC���A�*

	conv_lossQ1U=͐TJ        )��P	ŅCC���A�*

	conv_loss�e=5l1�        )��P	+�CC���A�*

	conv_loss�\�<����        )��P	��CC���A�*

	conv_loss�w_=��K�        )��P	3DC���A�*

	conv_loss��Y=H�d        )��P	BDC���A�*

	conv_loss�b=0��        )��P	GxDC���A�*

	conv_loss&�w=�H�        )��P	��DC���A�*

	conv_loss0S=`�        )��P	��DC���A�*

	conv_loss��a=D:        )��P	�EC���A�*

	conv_loss��k=�� <        )��P	�6EC���A�*

	conv_loss��=hL9@        )��P	eEC���A�*

	conv_loss��=B�!        )��P	F�EC���A�*

	conv_loss'�=���        )��P	.�EC���A�*

	conv_loss�_|=����        )��P	��EC���A�*

	conv_lossy0=wP�        )��P	T"FC���A�*

	conv_lossX�?=���        )��P	+PFC���A�*

	conv_loss��=R�         )��P	aFC���A�*

	conv_loss�=��X        )��P	a�FC���A�*

	conv_loss�a�=�QL�        )��P	8�FC���A�*

	conv_lossf
o=�Uh        )��P	GC���A�*

	conv_loss�&={���        )��P	�:GC���A�*

	conv_loss4�c=^���        )��P	jGC���A�*

	conv_loss��r=R�-        )��P	{�GC���A�*

	conv_loss�S=��#Q        )��P	�GC���A�*

	conv_loss_�=��k�        )��P	��GC���A�*

	conv_loss7�=b�
`        )��P	�(HC���A�*

	conv_loss�X3=`��,        )��P	WHC���A�*

	conv_loss��a=	�y:        )��P	��HC���A�*

	conv_loss	��=�@Y6        )��P	A�HC���A�*

	conv_lossE�0=���M        )��P	b�HC���A�*

	conv_loss�
p=[�        )��P	�IC���A�*

	conv_lossn�@=��.        )��P	1FIC���A�*

	conv_lossI`;=�%��        )��P	��IC���A�*

	conv_loss�$�=5�y        )��P	g�IC���A�*

	conv_loss=cf=񸓿        )��P	��IC���A�*

	conv_loss�k=�Kw�        )��P	�JC���A�*

	conv_loss="=��*        )��P	XJJC���A�*

	conv_loss�5=Te�        )��P	��JC���A�*

	conv_loss%�S=|o�        )��P	̵JC���A�*

	conv_loss���=��96        )��P	h�JC���A�*

	conv_loss"o=c�S�        )��P	�KC���A�*

	conv_lossh3X=�h�        )��P	5KKC���A�*

	conv_loss�W=3�@        )��P	�KC���A�*

	conv_loss	�q=5�I        )��P	�KC���A�*

	conv_loss�#=J>(�        )��P	r�KC���A�*

	conv_loss��'=e��         )��P	b2LC���A�*

	conv_loss�vA=�x�e        )��P	�eLC���A�*

	conv_loss�6=@2.        )��P	W�LC���A�*

	conv_loss x=���        )��P	F�LC���A�*

	conv_losse�+=�L_        )��P	[�LC���A�*

	conv_loss�'@=I�j        )��P	�)MC���A�*

	conv_loss��W=P���        )��P	�[MC���A�*

	conv_loss�Ap=R�'        )��P	��MC���A�*

	conv_lossm�X=ySJ        )��P	��MC���A�*

	conv_lossR�=
z:        )��P	f�MC���A�*

	conv_lossK@�=͕�)        )��P	�NC���A�*

	conv_loss��=�
        )��P	�PNC���A�*

	conv_lossF�u=��n�        )��P	
�NC���A�*

	conv_lossb�|=�SY        )��P	״NC���A�*

	conv_loss�i= �Ĉ        )��P	N�NC���A�*

	conv_loss"�5=�1��        )��P	�OC���A�*

	conv_loss��(=��u�        )��P	kLOC���A�*

	conv_loss���<a��        )��P	�{OC���A�*

	conv_lossD�=h��        )��P	��OC���A�*

	conv_loss5� =�r%        )��P	��OC���A�*

	conv_loss!J=��lF        )��P	�PC���A�*

	conv_lossh��=�w�.        )��P	�FPC���A�*

	conv_lossa`I=^#��        )��P	�zPC���A�*

	conv_loss=o^=-A��        )��P	{�PC���A�*

	conv_loss� �=�:Ň        )��P	��PC���A�*

	conv_loss7�=��V�        )��P	�
QC���A�*

	conv_loss��Y=�g�        )��P	b;QC���A�*

	conv_loss�h=���        )��P	&kQC���A�*

	conv_loss�Pt=ǜ{x        )��P	ǛQC���A�*

	conv_loss�g=�0�L        )��P	�QC���A�*

	conv_loss/\=��%1        )��P	��QC���A�*

	conv_loss�c=�p/d        )��P	U,RC���A�*

	conv_loss�oG=i?�Q        )��P	~]RC���A�*

	conv_lossZ!`=h��        )��P	�RC���A�*

	conv_lossm�O=���        )��P	,�RC���A�*

	conv_loss��X=�x�        )��P	�RC���A�*

	conv_losst9=h�.Z        )��P	D%SC���A�*

	conv_loss�V=Uc�        )��P	�hSC���A�*

	conv_loss��=�Y#        )��P	�SC���A�*

	conv_loss+�=����        )��P	7�SC���A�*

	conv_loss�E�=X�        )��P	�SC���A�*

	conv_loss:>=�hD5        )��P	�/TC���A�*

	conv_loss�U�=�s8�        )��P	�cTC���A�*

	conv_lossWJ�=��p�        )��P	7�TC���A�*

	conv_loss��=��l�        )��P	.�TC���A�*

	conv_lossF�M=���        )��P	(UC���A�*

	conv_loss'��=�m�        )��P	I8UC���A�*

	conv_lossl��=�v        )��P	}mUC���A�*

	conv_loss�x=��B�        )��P	&�UC���A�*

	conv_loss,eA=o\&        )��P	r�UC���A�*

	conv_loss�Ê=?$�q        )��P	lVC���A�*

	conv_loss��.=D݃�        )��P	�IVC���A�*

	conv_lossX�W=�s<~        )��P	�zVC���A�*

	conv_loss�Q=��        )��P	�VC���A�*

	conv_loss�S0=�H?�        )��P	��VC���A�*

	conv_loss�n�=��|�        )��P	2WC���A�*

	conv_loss��;=sVd        )��P	�?WC���A�*

	conv_loss��3=����        )��P	�pWC���A�*

	conv_loss��=r�Z        )��P	[�WC���A�*

	conv_loss�+=���        )��P	��WC���A�*

	conv_loss7#=��Di        )��P	7XC���A�*

	conv_loss=PZ�        )��P	�8XC���A�*

	conv_loss+�h=Mjj        )��P	�kXC���A�*

	conv_loss�YE=�W$�        )��P	�XC���A�*

	conv_loss�3�=�J        )��P	��XC���A�*

	conv_lossR�=j!��        )��P	�YC���A�*

	conv_loss�w=���V        )��P	45YC���A�*

	conv_loss�u=�eI�        )��P	fYC���A�*

	conv_loss:J.=��V        )��P	�YC���A�*

	conv_loss��A=�{..        )��P	��YC���A�*

	conv_loss�I=Y�'        )��P	��YC���A�*

	conv_loss�a�=X��        )��P	�'ZC���A�*

	conv_loss.Y=��@V        )��P	�\ZC���A�*

	conv_lossՆ?=6�        )��P	��ZC���A�*

	conv_loss�P=��b�        )��P	�ZC���A�*

	conv_loss�GQ=���        )��P	��ZC���A�*

	conv_lossv�=��&#        )��P	�#[C���A�*

	conv_loss��~=Ѽ�p        )��P	_S[C���A�*

	conv_lossƺQ=�K�        )��P	o�[C���A�*

	conv_loss�]K=߽�X        )��P	�[C���A�*

	conv_lossZ�V=�˅        )��P	��[C���A�*

	conv_loss_ˌ=7�Y�        )��P	\C���A�*

	conv_loss0
=��Qt        )��P	�H\C���A�*

	conv_loss�9=�Mi        )��P	My\C���A�*

	conv_loss��o=��q�        )��P	~�\C���A�*

	conv_loss:�#=B�4[        )��P	�\C���A�*

	conv_loss$#G=��Q�        )��P	�]C���A�*

	conv_lossoV�=>y'        )��P	=Y]C���A�*

	conv_lossq=�=���        )��P	{�]C���A�*

	conv_loss)�,=�hi�        )��P	��]C���A�*

	conv_lossJ�=F��5        )��P	�]C���A�*

	conv_loss�*p=tMA        )��P	m&^C���A�*

	conv_loss�&=�
�+        )��P	DW^C���A�*

	conv_loss�5�=�ND        )��P	D�^C���A�*

	conv_loss�_=I|�<        )��P	��^C���A�*

	conv_loss��=�N        )��P	�_C���A�*

	conv_loss%��=�;�;        )��P	u5_C���A�*

	conv_loss)c�=:�1�        )��P	eg_C���A�*

	conv_loss��e=���T        )��P	��_C���A�*

	conv_loss���=���        )��P	S�_C���A�*

	conv_loss~l�=��E�        )��P	`C���A�*

	conv_loss�f�=�rS�        )��P	H`C���A�*

	conv_loss#�t=܊�        )��P	�|`C���A�*

	conv_loss��/=~��y        )��P	��`C���A�*

	conv_loss�=-�S%        )��P	��`C���A�*

	conv_loss=;4=@pR5        )��P	EaC���A�*

	conv_loss�*=Y\�,        )��P	�MaC���A�*

	conv_loss� �=���'        )��P	T�aC���A�*

	conv_loss\~�=�#gj        )��P	�aC���A�*

	conv_loss։=�M�,        )��P	��aC���A�*

	conv_loss�q=��`        )��P	� bC���A�*

	conv_loss��k=��j        )��P	UbC���A�*

	conv_loss�6S=z�~        )��P	\�bC���A�*

	conv_loss�E=���        )��P	R�bC���A�*

	conv_losstԀ=�H��        )��P	��bC���A�*

	conv_loss,�E=)�(�        )��P	)cC���A�*

	conv_loss��y=����        )��P	DPcC���A�*

	conv_loss��^=F��"        )��P	сcC���A�*

	conv_loss0=�Q!�        )��P	c�cC���A�*

	conv_lossoC='�        )��P	��cC���A�*

	conv_loss
ט=hN�        )��P	�dC���A�*

	conv_lossi�C=�}��        )��P	�HdC���A�*

	conv_loss��g=G��        )��P	�ydC���A�*

	conv_lossǦr=ȗO,        )��P	�dC���A�*

	conv_loss��>=�598        )��P	��dC���A�*

	conv_loss<=�JK�        )��P	eC���A�*

	conv_loss�S=e]!s        )��P	�BeC���A�*

	conv_loss~�=(�5U        )��P	JteC���A�*

	conv_loss��q=�[{        )��P	�eC���A�*

	conv_loss�؆=�~�O        )��P	��eC���A�*

	conv_lossO0a=7N��        )��P	�fC���A�*

	conv_loss?��=�}�        )��P	�9fC���A�*

	conv_loss�=�6x        )��P	sjfC���A�*

	conv_loss==G=�ZG        )��P	�fC���A�*

	conv_lossG0=�6Pq        )��P	��fC���A�*

	conv_loss�_7=�9�|        )��P	S gC���A�*

	conv_loss�O=^&a        )��P	�2gC���A�*

	conv_loss9F�=�H        )��P	��hC���A�*

	conv_loss�K=�N3        )��P	iC���A�*

	conv_lossʜ�=�u�c        )��P	�4iC���A�*

	conv_lossʎH=(J�        )��P	�giC���A�*

	conv_loss7�w=�te        )��P	��iC���A�*

	conv_lossu
o=K�         )��P	I�iC���A�*

	conv_loss/�"=m���        )��P	7 jC���A�*

	conv_loss�lT=!5��        )��P	@jC���A�*

	conv_lossj�@=K���        )��P	(yjC���A�*

	conv_loss�w=L�i�        )��P	�jC���A�*

	conv_lossҵ>=��6�        )��P	��jC���A�*

	conv_lossab=0-Ot        )��P	ekC���A�*

	conv_loss�=Z�͔        )��P	�UkC���A�*

	conv_lossH�&=n���        )��P	��kC���A�*

	conv_loss��Y=�>        )��P	ֹkC���A�*

	conv_loss��=���        )��P	��kC���A�*

	conv_loss�B�=~��        )��P	�lC���A�*

	conv_loss��\=Y��        )��P	;LlC���A�*

	conv_loss���<���        )��P	��lC���A�*

	conv_lossJ�o=X7s|        )��P	߹lC���A�*

	conv_loss�	�<��o�        )��P	3�lC���A�*

	conv_loss�o=0�J        )��P	�mC���A�*

	conv_loss�`=���>        )��P	�KmC���A�*

	conv_loss��b=|��        )��P	�~mC���A�*

	conv_lossְ�=*R��        )��P	z�mC���A�*

	conv_losse�=�!&�        )��P	��mC���A�*

	conv_loss��<��t�        )��P	ZnC���A�*

	conv_loss��=�5^        )��P	JAnC���A�*

	conv_lossC�-=;7I�        )��P	6rnC���A�*

	conv_loss�S+=�4��        )��P	|�nC���A�*

	conv_loss<Ye={w�!        )��P	��nC���A�*

	conv_loss�/I=����        )��P	�oC���A�*

	conv_loss�.=9w��        )��P	\9oC���A�*

	conv_loss>G=���        )��P	]joC���A�*

	conv_loss �1=�        )��P	�oC���A�*

	conv_loss�K=:���        )��P	��oC���A�*

	conv_lossZ��=�Ł(        )��P	;�oC���A�*

	conv_loss�==)D�        )��P	-pC���A�*

	conv_loss��C=���         )��P	_^pC���A�*

	conv_loss��=@ic         )��P	6�pC���A�*

	conv_lossD�=S���        )��P	m�pC���A�*

	conv_loss�6=#�q        )��P	7�pC���A�*

	conv_loss��9=�޺         )��P	!qC���A�*

	conv_loss$�D=��        )��P	�UqC���A�*

	conv_loss��T=��        )��P	��qC���A�*

	conv_loss� N= ��o        )��P	�qC���A�*

	conv_losspg@=�-�        )��P	��qC���A�*

	conv_lossO�~=n~|w        )��P	�rC���A�*

	conv_loss՝|=yo	        )��P	�LrC���A�*

	conv_lossg�=}w��        )��P	�|rC���A�*

	conv_loss�O|=�8��        )��P	��rC���A�*

	conv_loss�y=x8�        )��P	��rC���A�*

	conv_loss/== ��9        )��P	�"sC���A�*

	conv_loss�= )�Q        )��P	�UsC���A�*

	conv_loss
�6=���        )��P	U�sC���A�*

	conv_loss"{S=��U}        )��P	��sC���A�*

	conv_loss���=(���        )��P	b�sC���A�*

	conv_lossR�m=�k��        )��P	�%tC���A�*

	conv_loss#Xj=u$U        )��P	�dtC���A�*

	conv_loss��7=_Pz�        )��P	e�tC���A�*

	conv_loss��#=[���        )��P	��tC���A�*

	conv_loss�"Q=m$�g        )��P	��tC���A�*

	conv_loss�q9=.��        )��P	#-uC���A�*

	conv_loss\�a=�@`        )��P	,`uC���A�*

	conv_lossl�=���        )��P	ݘuC���A�*

	conv_loss٨X=��        )��P	��uC���A�*

	conv_loss\-#=���N        )��P	c�uC���A�*

	conv_lossy5=]�{*        )��P	�-vC���A�*

	conv_lossz�G=)���        )��P	�`vC���A�*

	conv_loss�==��        )��P	W�vC���A�*

	conv_loss��=��        )��P	��vC���A�*

	conv_lossSvO=>�_F        )��P	hwC���A�*

	conv_loss�[=$k�G        )��P	�4wC���A�*

	conv_lossR=���         )��P	sfwC���A�*

	conv_lossp�G=�vru        )��P	��wC���A�*

	conv_loss�ܭ=+�        )��P	��wC���A�*

	conv_loss�m8=�u�q        )��P	��wC���A�*

	conv_loss��=T�9�        )��P	i*xC���A�*

	conv_loss�m!=���^        )��P	�]xC���A�*

	conv_loss�c= ���        )��P	M�xC���A�*

	conv_lossH,�=S��X        )��P	��xC���A�*

	conv_lossѳ�=԰|�        )��P	��xC���A�*

	conv_losswb�=ɭ�        )��P	2!yC���A�*

	conv_loss2��=n�2        )��P	@QyC���A�*

	conv_loss��i=�<�        )��P	уyC���A�*

	conv_lossJ3=��(        )��P	L�yC���A�*

	conv_loss�_[=��        )��P	��yC���A�*

	conv_lossKo==��=        )��P	czC���A�*

	conv_loss�Ad=��        )��P	�PzC���A�*

	conv_lossbyw=�Wp        )��P	_�zC���A�*

	conv_loss^\G=%6f?        )��P	��zC���A�*

	conv_loss��1=��6�        )��P	M�zC���A�*

	conv_loss95.=��s:        )��P	/{C���A�*

	conv_loss:�O=D��        )��P	Ee{C���A�*

	conv_loss?^'=y/}        )��P	y�{C���A�*

	conv_loss�(N=�h4        )��P	w�{C���A�*

	conv_loss�p=l�ц        )��P	�|C���A�*

	conv_loss��W=�+;        )��P	;=|C���A�*

	conv_loss�4�=��        )��P	�t|C���A�*

	conv_loss��x=�IL�        )��P	H�|C���A�*

	conv_lossM3q=�3�V        )��P	�|C���A�*

	conv_loss�=�܎o        )��P	i%}C���A�*

	conv_loss=�X=�L*        )��P	V}C���A�*

	conv_loss��=�s�C        )��P	��}C���A�*

	conv_loss	2w=�b        )��P	�}C���A�*

	conv_loss�a=l�a�        )��P	��}C���A�*

	conv_loss���=��         )��P	9~C���A�*

	conv_loss�=�u�w        )��P	p~C���A�*

	conv_loss��=��$]        )��P	B�~C���A�*

	conv_lossJ�=+W��        )��P	��~C���A�*

	conv_lossiK=��_&        )��P	kC���A�*

	conv_lossθd=vC�\        )��P	kHC���A�*

	conv_loss�ub=��t        )��P	?zC���A�*

	conv_loss1� =��+        )��P	��C���A�*

	conv_loss�m/=Q�W        )��P	��C���A�*

	conv_loss���=���N        )��P	��C���A�*

	conv_lossJR= y�s        )��P	�M�C���A�*

	conv_lossn,�=S27�        )��P	��C���A�*

	conv_loss�zd=�p�        )��P	#΀C���A�*

	conv_loss {=�z�J        )��P	t�C���A�*

	conv_loss6�=T��        )��P	B7�C���A�*

	conv_loss�N=x{�        )��P	�k�C���A�*

	conv_loss$}&=�^��        )��P	ğ�C���A�*

	conv_loss�Cc=���        )��P	��C���A�*

	conv_loss�f=���        )��P	y�C���A�*

	conv_loss/��=m��        )��P	9L�C���A�*

	conv_loss�H�=�*�`        )��P	>��C���A�*

	conv_loss��~=V�6{        )��P	.��C���A�*

	conv_lossKp=�F�        )��P	��C���A�*

	conv_loss{K�=~�`�        )��P	�)�C���A�*

	conv_loss��<=C�'�        )��P	�`�C���A�*

	conv_losso5r=��ޤ        )��P	q��C���A�*

	conv_loss�$U=v8��        )��P	wȃC���A�*

	conv_lossH�Q=E*�        )��P	+ �C���A�*

	conv_lossat]=�d�        )��P	91�C���A�*

	conv_loss���=CV�R        )��P	qg�C���A�*

	conv_lossRaQ=�"        )��P	꜄C���A�*

	conv_lossQt8=�f�\        )��P	CʄC���A�*

	conv_loss�=Ы�w        )��P	�C���A�*

	conv_loss,�{=��6        )��P	^4�C���A�*

	conv_loss�MQ=�
-W        )��P	9d�C���A�*

	conv_loss���=6���        )��P	���C���A�*

	conv_loss��5=��p        )��P	�ȅC���A�*

	conv_loss���=)�o        )��P	h��C���A�*

	conv_loss�6=_Z�        )��P	�1�C���A�*

	conv_loss�q-={r        )��P		b�C���A�*

	conv_loss �%=w���        )��P	p��C���A�*

	conv_lossW�,=e�B        )��P	���C���A�*

	conv_lossh(�=���/        )��P	4��C���A�*

	conv_loss��G=��        )��P	�(�C���A�*

	conv_loss�V�=��Nw        )��P	�i�C���A�*

	conv_lossH�S=��ܟ        )��P	~��C���A�*

	conv_loss��a=�CO�        )��P	KȇC���A�*

	conv_loss�#)=2g��        )��P	���C���A�*

	conv_loss	�}=y�1        )��P	2�C���A�*

	conv_lossCT=QNo�        )��P	�h�C���A�*

	conv_lossJ8V=���F        )��P	l��C���A�*

	conv_loss��;=�^�        )��P	�ȈC���A�*

	conv_loss�R�= J�        )��P	��C���A�*

	conv_loss� �=����        )��P	M>�C���A�*

	conv_loss��H=U�2�        )��P	pn�C���A�*

	conv_lossY�=։ĕ        )��P	+��C���A�*

	conv_loss�/=��a�        )��P	j͉C���A�*

	conv_loss"љ=�P��        )��P	h��C���A�*

	conv_loss�u�=H�j        )��P	�5�C���A�*

	conv_loss��x=��        )��P	�d�C���A�*

	conv_loss==�vc        )��P	��C���A�*

	conv_loss�FS=���|        )��P	LȊC���A�*

	conv_lossX}^=��O�        )��P	���C���A�*

	conv_loss�Kz=N�.�        )��P	�-�C���A�*

	conv_loss� i=	
�.        )��P	�\�C���A�*

	conv_loss�!=��J�        )��P	y��C���A�*

	conv_loss/�2=�UY        )��P	y��C���A�*

	conv_loss��[=Xo�0        )��P	[��C���A�*

	conv_loss��X=
0        )��P	|(�C���A�*

	conv_lossX�/=;Y&�        )��P	�\�C���A�*

	conv_loss�-=�:,7        )��P	���C���A�*

	conv_loss�|U=GM        )��P	K��C���A�*

	conv_loss��l=K���        )��P	��C���A�*

	conv_loss^�!=���        )��P	��C���A�*

	conv_loss�=39��        )��P	UU�C���A�*

	conv_loss=�ő        )��P	���C���A�*

	conv_loss�N�=��,        )��P	9��C���A�*

	conv_loss�L=�s��        )��P	H�C���A�*

	conv_loss�]=���        )��P	��C���A�*

	conv_loss�s_=��
b        )��P	fN�C���A�*

	conv_loss�ć= i        )��P	|}�C���A�*

	conv_lossBVW=�J`        )��P	龜C���A�*

	conv_loss ?5=���        )��P	 ێC���A�*

	conv_loss��6=�2��        )��P	L�C���A�*

	conv_losslL#=��        )��P	u:�C���A�*

	conv_loss�
=i��        )��P	�i�C���A�*

	conv_loss�-l=�ͥ�        )��P	М�C���A�*

	conv_loss��K=.*=�        )��P	�ʏC���A�*

	conv_lossE�!=o��        )��P	���C���A�*

	conv_loss��!=}�=        )��P	?)�C���A�*

	conv_loss>_K=�*�        )��P	dW�C���A�*

	conv_loss X�=�k��        )��P	���C���A�*

	conv_lossƸ�=c:�        )��P	W��C���A�*

	conv_lossq�,=}!��        )��P	��C���A�*

	conv_lossh;B=��*        )��P	Y��C���A�*

	conv_lossZI�=u�҃        )��P	�ږC���A�*

	conv_loss��n=o�'        )��P	��C���A�*

	conv_loss�=����        )��P	�>�C���A�*

	conv_loss�D=�y�1        )��P	�{�C���A�*

	conv_loss})F=0��        )��P	2��C���A�*

	conv_lossI�=(Au�        )��P	^ۗC���A�*

	conv_loss�3y=��_�        )��P	�
�C���A�*

	conv_loss�iW=f(�        )��P	�C�C���A�*

	conv_loss0h�=ZL�z        )��P	�~�C���A�*

	conv_loss�D;=��L�        )��P	b��C���A�*

	conv_lossG�$=r�        )��P	tޘC���A�*

	conv_lossđ@=FU��        )��P	8�C���A�*

	conv_loss��_=B*=        )��P	�>�C���A�*

	conv_lossM�\=J� X        )��P	�p�C���A�*

	conv_lossǓ�=����        )��P	c��C���A�*

	conv_lossXU=��0        )��P	G֙C���A�*

	conv_loss�}p=p��        )��P	�
�C���A�*

	conv_loss4d=�[��        )��P	l@�C���A�*

	conv_loss+ =���        )��P	p�C���A�*

	conv_lossh=�D3!        )��P	��C���A�*

	conv_loss���=+7��        )��P	XΚC���A�*

	conv_loss�yb=�[��        )��P	+��C���A�*

	conv_lossI�}=5�<S        )��P	�0�C���A�*

	conv_lossO9n=��ś        )��P	(d�C���A�*

	conv_lossj$<=X�Jy        )��P	���C���A�*

	conv_loss�M�=��j        )��P	l͛C���A�*

	conv_loss�VG=�J        )��P	0��C���A�*

	conv_loss1=��;        )��P	t/�C���A�*

	conv_loss�=�h        )��P	�a�C���A�*

	conv_loss(a!=K.�        )��P	��C���A�*

	conv_loss�*P=��-�        )��P	�ۜC���A�*

	conv_loss�<�#*�        )��P	�C���A�*

	conv_loss�Ar=�/_        )��P	>�C���A�*

	conv_loss��=�-�        )��P	1q�C���A�*

	conv_lossX�n=�c�x        )��P	%��C���A�*

	conv_loss�(�=���V        )��P	�НC���A�*

	conv_loss�T=��/�        )��P	i �C���A�*

	conv_loss��_=�#q�        )��P	x1�C���A�*

	conv_loss�}>=ePB7        )��P	�b�C���A�*

	conv_lossQ=���Q        )��P	啞C���A�*

	conv_lossPU6=^'�        )��P	�ǞC���A�*

	conv_loss�'=�-Z5        )��P	���C���A�*

	conv_loss�v2=Y���        )��P	9,�C���A�*

	conv_lossK�M=~�|_        )��P	k]�C���A�*

	conv_loss^q*=@��^        )��P	���C���A�*

	conv_loss�?B=�B�        )��P	���C���A�*

	conv_loss�@=���>        )��P	��C���A�*

	conv_loss�8=Q�        )��P	��C���A�*

	conv_loss+B=�i�        )��P	�Q�C���A�*

	conv_lossH��==��        )��P	���C���A�*

	conv_loss��=��5�        )��P	ĠC���A�*

	conv_loss��(=(���        )��P	F��C���A�*

	conv_loss��`=�,�        )��P	�,�C���A�*

	conv_loss$�L=SW��        )��P	�_�C���A�*

	conv_losszy�=�[H�        )��P	֗�C���A�*

	conv_losstg=x�"`        )��P	�ݡC���A�*

	conv_loss�68=�S;7        )��P	��C���A�*

	conv_loss��F=4��e        )��P	�C�C���A�*

	conv_lossE̟=X�J        )��P	w�C���A�*

	conv_loss�,0=��         )��P	d��C���A�*

	conv_loss}ڙ=p���        )��P	�ޢC���A�*

	conv_loss(%�=x	�        )��P	��C���A�*

	conv_loss�@="��k        )��P	�?�C���A�*

	conv_loss�*4=p�;O        )��P	�o�C���A�*

	conv_loss�
0=m>|        )��P	���C���A�*

	conv_loss�d=:        )��P	�ѣC���A�*

	conv_lossxmv=:�        )��P	��C���A�*

	conv_loss�=L�2        )��P	TL�C���A�*

	conv_loss�7=���        )��P	��C���A�*

	conv_loss?�s=X���        )��P	J��C���A�*

	conv_lossg�?=�h�t        )��P	��C���A�*

	conv_losso<=�/��        )��P	_�C���A�*

	conv_loss��c=K�0Z        )��P	/F�C���A�*

	conv_loss�	�=�H        )��P	�v�C���A�*

	conv_lossb�7=u}��        )��P	y��C���A�*

	conv_loss8xw=?~W�        )��P	0٥C���A�*

	conv_loss3w�<��]c        )��P	�
�C���A�*

	conv_lossl�=}�Õ        )��P	�=�C���A�*

	conv_lossAc=�{g7        )��P	�n�C���A�*

	conv_lossIR=����        )��P	u��C���A�*

	conv_lossY,=�7v�        )��P	�ϦC���A�*

	conv_loss�}w=�7        )��P	��C���A�*

	conv_losseɑ=R�k�        )��P	�5�C���A�*

	conv_lossz�:=QYE        )��P	Xg�C���A�*

	conv_lossY<;=�"�V        )��P	˙�C���A�*

	conv_lossK=A�"�        )��P	�̧C���A�*

	conv_loss�&i=���        )��P	^��C���A�*

	conv_loss%��=��P        )��P	62�C���A�*

	conv_loss|�\=����        )��P	h�C���A�*

	conv_loss��L=��ʴ        )��P	=��C���A�*

	conv_loss��5=73I!        )��P	p̨C���A�*

	conv_loss�,H=hs�        )��P	s��C���A�*

	conv_loss�=�JX6        )��P	�2�C���A�*

	conv_loss�&<=[���        )��P	5h�C���A�*

	conv_loss�z=�#�        )��P	��C���A� *

	conv_loss�5P=�	�R        )��P	eߩC���A� *

	conv_lossF�A=�8��        )��P	��C���A� *

	conv_lossx�E=:8й        )��P	�J�C���A� *

	conv_loss{a<=\�U�        )��P	~�C���A� *

	conv_loss~�p=rA8f        )��P	���C���A� *

	conv_loss�z=��n�        )��P	��C���A� *

	conv_loss��N=�Qj        )��P	:�C���A� *

	conv_loss��=�bnX        )��P	�J�C���A� *

	conv_loss���<w�P        )��P	���C���A� *

	conv_lossZqf=`m:�        )��P	.��C���A� *

	conv_loss�+=�׽�        )��P	��C���A� *

	conv_loss�f}=-���        )��P	Q/�C���A� *

	conv_lossvd=��V        )��P	q�C���A� *

	conv_loss��/=�1~        )��P	Ģ�C���A� *

	conv_loss�,=?��        )��P	LѬC���A� *

	conv_loss��o=t�m        )��P	� �C���A� *

	conv_loss�p"=CE�        )��P	�/�C���A� *

	conv_loss�*X=E��        )��P	k^�C���A� *

	conv_lossp��=+m|�        )��P	��C���A� *

	conv_loss"gh=�        )��P	�ŭC���A� *

	conv_loss�n=*�Q        )��P	���C���A� *

	conv_loss?b=l�K        )��P	�(�C���A� *

	conv_loss^b=��        )��P	g�C���A� *

	conv_loss]��=#�.�        )��P	=��C���A� *

	conv_loss�C3=�5<�        )��P	 ˮC���A� *

	conv_loss�%v=<L��        )��P	���C���A� *

	conv_losspY=�/5L        )��P	=,�C���A� *

	conv_loss7�=� ��        )��P	�Z�C���A� *

	conv_loss�U=��:�        )��P	d��C���A� *

	conv_lossj!_=�f�t        )��P	�ïC���A� *

	conv_loss�dh=��l        )��P	��C���A� *

	conv_loss_�S=6���        )��P	:�C���A� *

	conv_loss�<!=�T�        )��P	�N�C���A� *

	conv_loss��J=)y�        )��P	ۄ�C���A� *

	conv_loss%C=����        )��P	[��C���A� *

	conv_loss�>=&'Fj        )��P	��C���A� *

	conv_loss�*'=>I��        )��P	H�C���A� *

	conv_losss�7=e�        )��P	�O�C���A� *

	conv_lossq&a=%�{        )��P	�C���A� *

	conv_lossϋ=��3        )��P	���C���A� *

	conv_loss��[=�e��        )��P	ݱC���A� *

	conv_loss�'�<rn��        )��P	*�C���A� *

	conv_loss�v�=�@$�        )��P	�@�C���A� *

	conv_loss�ZH=<r�        )��P	7o�C���A� *

	conv_lossw�=�QT�        )��P	x��C���A� *

	conv_loss}�'=��:�        )��P	�ϲC���A� *

	conv_loss^�<�.�R        )��P	���C���A� *

	conv_loss�k�=�fe        )��P	S/�C���A� *

	conv_loss�1=���G        )��P	�_�C���A� *

	conv_loss��x=�v��        )��P	���C���A� *

	conv_loss��=S�"&        )��P	-ĳC���A� *

	conv_loss=0=�xLl        )��P	n��C���A� *

	conv_loss�7=U�'�        )��P	�%�C���A� *

	conv_losswb^=Ѽ�{        )��P	-U�C���A� *

	conv_loss��'=���        )��P	��C���A� *

	conv_loss�F�=��@�        )��P	�ŴC���A� *

	conv_loss	�[=QtA�        )��P	��C���A� *

	conv_lossG-=#?r]        )��P	�"�C���A� *

	conv_lossŚ=OD�        )��P	�U�C���A� *

	conv_loss،H=�ts!        )��P	o��C���A� *

	conv_loss��=!��        )��P	һ�C���A� *

	conv_losst}=Fq�a        )��P	���C���A� *

	conv_loss�-=��Mi        )��P	1�C���A� *

	conv_loss�7L=�̂        )��P	
f�C���A� *

	conv_lossiB=��p        )��P	���C���A� *

	conv_loss�d`=֘Js        )��P	ŶC���A� *

	conv_loss4=���        )��P	7��C���A� *

	conv_loss:#E=_�D�        )��P	*�C���A� *

	conv_loss�)=�<��        )��P	�Z�C���A� *

	conv_loss��V=N��        )��P	���C���A� *

	conv_lossI�K=b2�-        )��P	h��C���A� *

	conv_loss��f=Ꚑl        )��P	��C���A� *

	conv_lossRBo=��        )��P	�C���A� *

	conv_loss��;=j��4        )��P	�N�C���A� *

	conv_loss�R={f��        )��P	N��C���A� *

	conv_loss��>=����        )��P	��C���A� *

	conv_loss��u=x,��        )��P	��C���A� *

	conv_loss�0�=j|��        )��P	.�C���A� *

	conv_loss�1L=��#        )��P	�B�C���A� *

	conv_lossp�@=-Xح        )��P	Sp�C���A� *

	conv_loss=`=�%�        )��P	���C���A� *

	conv_loss�t8=��n        )��P	�͹C���A� *

	conv_loss$�Q= �io        )��P	���C���A� *

	conv_loss�K=Ơ2{        )��P	�+�C���A� *

	conv_loss�P=m(m        )��P	�Z�C���A� *

	conv_loss��H=��x@        )��P	\��C���A� *

	conv_lossl�2=䵪�        )��P	8��C���A� *

	conv_loss��=��GU        )��P	�C���A� *

	conv_loss�%l=���        )��P	��C���A� *

	conv_loss:4E=����        )��P	ZI�C���A� *

	conv_loss8H?=��{�        )��P	f{�C���A� *

	conv_loss���=nc��        )��P	驻C���A� *

	conv_loss�"=z��}        )��P	ڻC���A� *

	conv_loss/=�nu�        )��P	�	�C���A� *

	conv_lossk�e=��A        )��P	�8�C���A� *

	conv_loss"@=YĈj        )��P	di�C���A� *

	conv_loss�;=��!�        )��P	���C���A� *

	conv_lossD�O=��        )��P	0ɼC���A� *

	conv_loss��=%���        )��P	���C���A� *

	conv_loss�Ch=���        )��P	�)�C���A� *

	conv_loss��Z=���g        )��P	Y�C���A� *

	conv_lossl3=�Q��        )��P	��C���A� *

	conv_loss)2=]��Q        )��P	��C���A� *

	conv_loss�6�=bU�        )��P	��C���A� *

	conv_loss �$=L�HG        )��P	�x�C���A� *

	conv_loss��U=�?d�        )��P	���C���A� *

	conv_loss�<=*�        )��P	�ֿC���A� *

	conv_loss�=|7��        )��P	7�C���A� *

	conv_loss�Y<=�J�        )��P	 9�C���A� *

	conv_loss_$=���        )��P	ui�C���A� *

	conv_loss��@=���        )��P	���C���A� *

	conv_loss��=7�Q         )��P	���C���A� *

	conv_loss�V�=��G        )��P	&�C���A� *

	conv_loss �M=��wI        )��P	'5�C���A� *

	conv_loss��=�7s        )��P	�d�C���A� *

	conv_loss��J=����        )��P	��C���A� *

	conv_loss��=��\P        )��P	���C���A� *

	conv_loss��=����        )��P	��C���A� *

	conv_lossA"=��         )��P	!�C���A� *

	conv_lossV`Q=�/vf        )��P	�b�C���A� *

	conv_loss�=�B�6        )��P	���C���A� *

	conv_lossށ�=�E?�        )��P	���C���A� *

	conv_loss��Z=GiU        )��P	u��C���A� *

	conv_loss�G=*���        )��P	9-�C���A� *

	conv_lossu�=�6/        )��P	G_�C���A� *

	conv_lossę9=�q}        )��P	��C���A� *

	conv_loss92=���        )��P	���C���A� *

	conv_loss�D=ܦ�        )��P	���C���A�!*

	conv_lossK�A=�Ja�        )��P	�"�C���A�!*

	conv_loss�,r=o�        )��P	�T�C���A�!*

	conv_loss,i6=��        )��P	��C���A�!*

	conv_loss�Y)=C��"        )��P	;��C���A�!*

	conv_lossein=}���        )��P	Q��C���A�!*

	conv_loss+)=`�`'        )��P	�C���A�!*

	conv_loss@'"=-��        )��P	9D�C���A�!*

	conv_loss�A�=�k]        )��P	�s�C���A�!*

	conv_loss�|=����        )��P	��C���A�!*

	conv_loss[=�o        )��P	#��C���A�!*

	conv_loss��R=��)=        )��P	{�C���A�!*

	conv_loss��==LW+        )��P	_0�C���A�!*

	conv_loss�"S=�踼        )��P	_�C���A�!*

	conv_lossi�\=�3�V        )��P	���C���A�!*

	conv_loss��=����        )��P	e��C���A�!*

	conv_loss��L=�&�(        )��P	���C���A�!*

	conv_loss�j=��#        )��P	7�C���A�!*

	conv_lossa��=���D        )��P	�I�C���A�!*

	conv_loss!��=���        )��P	Hw�C���A�!*

	conv_lossDh=>�p�        )��P	ҥ�C���A�!*

	conv_lossW�=��[        )��P	Q��C���A�!*

	conv_loss���=���H        )��P	F�C���A�!*

	conv_loss5W�=���*        )��P	 3�C���A�!*

	conv_loss�d=���        )��P	8c�C���A�!*

	conv_loss�V/=�1=�        )��P	��C���A�!*

	conv_loss��C=��o�        )��P	U��C���A�!*

	conv_lossc=Ju�        )��P	��C���A�!*

	conv_loss�L=�
�G        )��P	D1�C���A�!*

	conv_loss�!#=�\�        )��P	�`�C���A�!*

	conv_loss�L�=f{�j        )��P	���C���A�!*

	conv_loss��5=&w�        )��P	M��C���A�!*

	conv_lossT�=
���        )��P	���C���A�!*

	conv_lossA�?=	�d        )��P	e&�C���A�!*

	conv_loss�_=	�&�        )��P	�V�C���A�!*

	conv_loss�QO=1R�g        )��P	,��C���A�!*

	conv_lossE�=q���        )��P	r��C���A�!*

	conv_lossU�Z=�$��        )��P	g��C���A�!*

	conv_lossTBp=�T&f        )��P	�#�C���A�!*

	conv_lossӊS=��G�        )��P	�S�C���A�!*

	conv_loss��r=P�Q        )��P	?��C���A�!*

	conv_loss�-=��Ն        )��P	��C���A�!*

	conv_lossp�0=2��A        )��P	���C���A�!*

	conv_lossf�E=�g��        )��P	
�C���A�!*

	conv_loss�=�0KA        )��P	fH�C���A�!*

	conv_lossw*y=J�Z        )��P	S}�C���A�!*

	conv_loss�R5=ع�        )��P	Q��C���A�!*

	conv_loss5�=�5P        )��P	��C���A�!*

	conv_loss�F1=��3n        )��P	"�C���A�!*

	conv_loss��0=�B[�        )��P	:Q�C���A�!*

	conv_lossO�3=�a{        )��P	���C���A�!*

	conv_loss�>D=S���        )��P	��C���A�!*

	conv_loss�nN=�'}�        )��P	���C���A�!*

	conv_loss��z=��9�        )��P	w�C���A�!*

	conv_loss) ]=���        )��P	�D�C���A�!*

	conv_loss��G=4�        )��P	�s�C���A�!*

	conv_losss~�=?�K        )��P	4��C���A�!*

	conv_loss��=1s�M        )��P	���C���A�!*

	conv_loss]FV=k�d        )��P	��C���A�!*

	conv_loss�4=�Ѡ/        )��P	-7�C���A�!*

	conv_losse�9=���        )��P	�f�C���A�!*

	conv_loss�p==��        )��P	���C���A�!*

	conv_lossRxP=�K�        )��P	7��C���A�!*

	conv_loss��1=)l�        )��P	���C���A�!*

	conv_loss��k=޸a        )��P	�*�C���A�!*

	conv_loss��Q=F�$i        )��P	�Z�C���A�!*

	conv_loss��.=7�h        )��P	B��C���A�!*

	conv_loss��=2B5�        )��P	ù�C���A�!*

	conv_lossjxe=`~��        )��P	��C���A�!*

	conv_loss�s,=!�^F        )��P	��C���A�!*

	conv_loss�g�=�gp�        )��P	�M�C���A�!*

	conv_loss�q0=�Ll�        )��P	��C���A�!*

	conv_loss��=�<��        )��P	;��C���A�!*

	conv_loss�$=���        )��P	J��C���A�!*

	conv_lossϽm=�Q�        )��P	�C���A�!*

	conv_loss�\=I�        )��P	=B�C���A�!*

	conv_loss�=���        )��P	�t�C���A�!*

	conv_loss��Z=��3�        )��P	��C���A�!*

	conv_loss���<�]�        )��P	���C���A�!*

	conv_lossC:�<�4)        )��P	<�C���A�!*

	conv_loss�6=	�        )��P	�K�C���A�!*

	conv_loss>�=r���        )��P	}|�C���A�!*

	conv_loss��R=­9<        )��P	x��C���A�!*

	conv_loss�8=�8=�        )��P	��C���A�!*

	conv_loss��\=a<�        )��P	��C���A�!*

	conv_loss�%=�ae�        )��P	�A�C���A�!*

	conv_loss �<=E��        )��P	t�C���A�!*

	conv_loss��0=���        )��P	���C���A�!*

	conv_loss]=�rr�        )��P	��C���A�!*

	conv_lossaa=r�        )��P	��C���A�!*

	conv_loss�*d=D�         )��P	mD�C���A�!*

	conv_loss�_=Q�7        )��P	�t�C���A�!*

	conv_loss��3=��J�        )��P	l��C���A�!*

	conv_loss|=�:�)        )��P	���C���A�!*

	conv_loss}o=��<�        )��P	��C���A�!*

	conv_loss�.=��@�        )��P	�@�C���A�!*

	conv_loss3�L="��        )��P	�y�C���A�!*

	conv_loss�I=\��f        )��P	ޮ�C���A�!*

	conv_loss!#=X��        )��P	���C���A�!*

	conv_loss��=���7        )��P	A�C���A�!*

	conv_loss�i=ӷ�        )��P	�F�C���A�!*

	conv_lossMF={k�        )��P	�u�C���A�!*

	conv_loss�I=ɽZ        )��P	&��C���A�!*

	conv_loss((=W��{        )��P	���C���A�!*

	conv_loss��N=@N�        )��P	��C���A�!*

	conv_lossn�=\��{        )��P	;5�C���A�!*

	conv_loss��==D��I        )��P	�i�C���A�!*

	conv_loss;=���        )��P	��C���A�!*

	conv_loss��=V���        )��P	Q��C���A�!*

	conv_loss^�%=Xws�        )��P	N��C���A�!*

	conv_loss�Yg==☑        )��P	�0�C���A�!*

	conv_loss�.9=I���        )��P	&b�C���A�!*

	conv_lossJiV=E�4�        )��P	��C���A�!*

	conv_loss"~g=���        )��P	7��C���A�!*

	conv_loss���=�P�L        )��P	���C���A�!*

	conv_loss'FR=���        )��P	)�C���A�!*

	conv_loss�4)=5�x]        )��P	�Y�C���A�!*

	conv_lossu9=��        )��P	o��C���A�!*

	conv_lossO`Y=$~��        )��P	��C���A�!*

	conv_lossoB=y瑩        )��P	���C���A�!*

	conv_loss�rI=a	с        )��P	�C���A�!*

	conv_loss(o�=�S/x        )��P	�O�C���A�!*

	conv_loss���=%H��        )��P	g��C���A�!*

	conv_lossYWU=�^{�        )��P	���C���A�!*

	conv_loss�H[=���        )��P	K��C���A�!*

	conv_loss�3=Ϟ
        )��P	]�C���A�!*

	conv_loss�!j=N��?        )��P	�K�C���A�!*

	conv_lossP��=`��c        )��P	э�C���A�!*

	conv_losscH=ƪ��        )��P	���C���A�"*

	conv_loss/ks=���        )��P	M��C���A�"*

	conv_loss�=�AL        )��P	*�C���A�"*

	conv_loss̶=�2�I        )��P	M[�C���A�"*

	conv_loss�9<=;Ïd        )��P	0��C���A�"*

	conv_lossFe	=��e        )��P	���C���A�"*

	conv_loss�{c=��g        )��P	���C���A�"*

	conv_loss�=ƣ�        )��P	�=�C���A�"*

	conv_loss��8=F�U/        )��P	�s�C���A�"*

	conv_loss^�S=�"[�        )��P	���C���A�"*

	conv_loss��)=̚@R        )��P	���C���A�"*

	conv_loss��=td�[        )��P	*
�C���A�"*

	conv_loss�(E=ykO        )��P	�B�C���A�"*

	conv_loss~4=����        )��P	�s�C���A�"*

	conv_loss��=��        )��P	���C���A�"*

	conv_loss�qb=�q��        )��P	���C���A�"*

	conv_loss�^Y=L.�        )��P	p
�C���A�"*

	conv_loss�Q=	tE        )��P	<�C���A�"*

	conv_loss�A=L�T        )��P	�u�C���A�"*

	conv_loss�q�=;�C        )��P	���C���A�"*

	conv_loss��=���,        )��P	���C���A�"*

	conv_loss�=P�w�        )��P	��C���A�"*

	conv_lossC8=�7-        )��P	�A�C���A�"*

	conv_loss}b=&�        )��P	�q�C���A�"*

	conv_lossZ�N=��a        )��P	���C���A�"*

	conv_loss �<�3�!        )��P	���C���A�"*

	conv_losss�=#�        )��P	��C���A�"*

	conv_loss��=� �E        )��P	;�C���A�"*

	conv_loss�t[=��)        )��P	:m�C���A�"*

	conv_loss�i=�?�        )��P	C��C���A�"*

	conv_loss|Y=��֞        )��P	m��C���A�"*

	conv_loss��=X_�I        )��P	J��C���A�"*

	conv_loss�4=� ��        )��P	1�C���A�"*

	conv_loss�3= �~         )��P	�b�C���A�"*

	conv_lossCq=�}7        )��P	���C���A�"*

	conv_lossw!=�]��        )��P	��C���A�"*

	conv_lossb=�rn�        )��P	{��C���A�"*

	conv_loss�4=ȼ�k        )��P	Z'�C���A�"*

	conv_lossa�!=B���        )��P	@Y�C���A�"*

	conv_loss
�=%�ʡ        )��P	���C���A�"*

	conv_loss�m=��m        )��P	ȷ�C���A�"*

	conv_lossM'�<��ּ        )��P	���C���A�"*

	conv_loss�y=����        )��P	��C���A�"*

	conv_lossB{c=��=J        )��P	�K�C���A�"*

	conv_loss�=Z��T        )��P	<��C���A�"*

	conv_lossF[=��u�        )��P	���C���A�"*

	conv_loss8�=���\        )��P	���C���A�"*

	conv_loss��a=3�̀        )��P	�C���A�"*

	conv_loss�9,=����        )��P	�G�C���A�"*

	conv_lossaO6=7�c�        )��P	���C���A�"*

	conv_loss�(=�T�        )��P	��C���A�"*

	conv_loss�e=.s	-        )��P	S<�C���A�"*

	conv_loss���=���        )��P	�w�C���A�"*

	conv_loss�v�=4�5        )��P	h��C���A�"*

	conv_loss�\=��uA        )��P	��C���A�"*

	conv_loss�� =��I        )��P	L�C���A�"*

	conv_lossi�<��4        )��P	�<�C���A�"*

	conv_loss͕�=���@        )��P	�|�C���A�"*

	conv_lossAE=��        )��P	��C���A�"*

	conv_loss��S=$S�        )��P	���C���A�"*

	conv_lossN$.=[혃        )��P	��C���A�"*

	conv_loss^=�Ե�        )��P	�A�C���A�"*

	conv_loss��Y=r�`�        )��P	[r�C���A�"*

	conv_loss!t�=r��        )��P	;��C���A�"*

	conv_loss��Y=w/��        )��P	���C���A�"*

	conv_loss�a=ǐ�v        )��P	��C���A�"*

	conv_loss[9<=�t��        )��P	�B�C���A�"*

	conv_loss�2=oe        )��P	fr�C���A�"*

	conv_loss�=μ��        )��P	 ��C���A�"*

	conv_lossD=�        )��P	?��C���A�"*

	conv_loss�
=���        )��P	�C���A�"*

	conv_loss"B�<#��        )��P	xC�C���A�"*

	conv_loss��8=����        )��P	Qt�C���A�"*

	conv_loss��7=X�B        )��P	���C���A�"*

	conv_lossG�f="Zy�        )��P	���C���A�"*

	conv_loss�[=H        )��P	��C���A�"*

	conv_loss�S�=���        )��P	�3�C���A�"*

	conv_lossyt2=2���        )��P	�b�C���A�"*

	conv_loss��g=+4c�        )��P	ґ�C���A�"*

	conv_loss2=���        )��P	\��C���A�"*

	conv_lossBt=��a�        )��P	7��C���A�"*

	conv_loss�d=G��        )��P	�'�C���A�"*

	conv_lossg
R=�g�        )��P	�W�C���A�"*

	conv_loss�mH=sΆ'        )��P	υ�C���A�"*

	conv_loss@x=!��!        )��P	U��C���A�"*

	conv_losseY=A��p        )��P	K��C���A�"*

	conv_lossՙ=$h�        )��P	��C���A�"*

	conv_lossD�=�-b*        )��P	^G�C���A�"*

	conv_loss!zF=��        )��P	�v�C���A�"*

	conv_loss�_=o�Υ        )��P	Ħ�C���A�"*

	conv_losse<=�-��        )��P	���C���A�"*

	conv_loss6{F=h�u        )��P	
�C���A�"*

	conv_lossm=a��a        )��P	�9�C���A�"*

	conv_loss%'=6:��        )��P	yh�C���A�"*

	conv_loss�=��&        )��P	(��C���A�"*

	conv_loss�j=�Ա�        )��P	���C���A�"*

	conv_loss�@=Sl(�        )��P	7��C���A�"*

	conv_loss�_=�L�        )��P	�)�C���A�"*

	conv_loss0D=^���        )��P	cY�C���A�"*

	conv_loss��=��w        )��P	f��C���A�"*

	conv_loss�m�=��        )��P	u��C���A�"*

	conv_loss��=����        )��P	��C���A�"*

	conv_loss�=��Va        )��P	�8�C���A�"*

	conv_loss�Ɋ=G��        )��P	�m�C���A�"*

	conv_loss�y=�y        )��P	k��C���A�"*

	conv_loss�|�=�I�%        )��P	���C���A�"*

	conv_lossq�=80��        )��P	�C���A�"*

	conv_loss"=��o        )��P	�C�C���A�"*

	conv_loss~�:=��+�        )��P	){�C���A�"*

	conv_lossF�=s�0R        )��P	��C���A�"*

	conv_loss���=nנ�        )��P	���C���A�"*

	conv_loss܋ =ӯa`        )��P	p�C���A�"*

	conv_loss��=��W�        )��P	�G�C���A�"*

	conv_lossj�O=6��        )��P	�{�C���A�"*

	conv_loss��N=Y���        )��P	��C���A�"*

	conv_loss�4=�+�        )��P	���C���A�"*

	conv_loss}M=�F�        )��P	��C���A�"*

	conv_loss㭔=����        )��P	.N�C���A�"*

	conv_loss�?=�8�        )��P	%�C���A�"*

	conv_loss�T]=��.�        )��P	Ѯ�C���A�"*

	conv_loss3�=�׋�        )��P	���C���A�"*

	conv_loss�=�<��Z        )��P	}�C���A�"*

	conv_loss��A=�[�^        )��P	lN�C���A�"*

	conv_loss*~0=��}        )��P	>�C���A�"*

	conv_loss�"@=ι�        )��P	)��C���A�"*

	conv_loss��2=cN%�        )��P	��C���A�"*

	conv_loss|�E=�g        )��P	��C���A�"*

	conv_loss)G=��1�        )��P	�M�C���A�"*

	conv_lossfH=�t�        )��P	�~�C���A�#*

	conv_loss�88=��=        )��P	���C���A�#*

	conv_loss�)=�ɞl        )��P	���C���A�#*

	conv_loss�%�<�.��        )��P	��C���A�#*

	conv_loss/!�<r�{�        )��P	>�C���A�#*

	conv_loss�9�<�T�        )��P	�n�C���A�#*

	conv_loss=��O2        )��P	���C���A�#*

	conv_loss�v�<�44�        )��P	@��C���A�#*

	conv_lossB`5=��B[        )��P	� �C���A�#*

	conv_loss�O=a��E        )��P	J1�C���A�#*

	conv_lossv3=�쟥        )��P	}a�C���A�#*

	conv_loss�=~Þk        )��P	c��C���A�#*

	conv_loss$�+=�"m'        )��P		��C���A�#*

	conv_loss�A=���        )��P	��C���A�#*

	conv_loss�\=�� �        )��P	B2�C���A�#*

	conv_loss�XJ=��76        )��P	hc�C���A�#*

	conv_loss"u�=�y        )��P	E��C���A�#*

	conv_loss�5k=O=>�        )��P	���C���A�#*

	conv_loss�?=cE        )��P	���C���A�#*

	conv_loss�<�=J%�        )��P	�#�C���A�#*

	conv_loss�t<=ӧ�        )��P	o��C���A�#*

	conv_losscX�=v�ӷ        )��P	���C���A�#*

	conv_loss�x-=[��        )��P	� D���A�#*

	conv_loss�r=�ki�        )��P	�I D���A�#*

	conv_lossn8=�a�        )��P	�x D���A�#*

	conv_loss��=���6        )��P	q� D���A�#*

	conv_lossaT=��        )��P	E� D���A�#*

	conv_loss��=
�x        )��P	`D���A�#*

	conv_lossk8=��C�        )��P	ND���A�#*

	conv_loss�J)=p�y        )��P	�D���A�#*

	conv_loss�,�=�JxO        )��P	��D���A�#*

	conv_loss� =&�'        )��P	n�D���A�#*

	conv_loss��=X��        )��P	�D���A�#*

	conv_lossm-�<YA�        )��P	YD���A�#*

	conv_loss� ,=����        )��P	��D���A�#*

	conv_loss�in=etr        )��P	��D���A�#*

	conv_loss��,=,��;        )��P	��D���A�#*

	conv_losss	=߽O        )��P	�D���A�#*

	conv_lossXZ>=���        )��P	KD���A�#*

	conv_loss�6;=�F �        )��P	*|D���A�#*

	conv_loss+`=*ӗ�        )��P	��D���A�#*

	conv_lossy:6=���        )��P	^�D���A�#*

	conv_lossg@=���        )��P	#D���A�#*

	conv_loss��=r�^]        )��P	:GD���A�#*

	conv_loss?zl=����        )��P	�vD���A�#*

	conv_loss#,I=��x        )��P	
�D���A�#*

	conv_loss
yA=Z"L�        )��P	y�D���A�#*

	conv_loss��<�!�        )��P	XD���A�#*

	conv_lossW3,=c���        )��P	1D���A�#*

	conv_loss�]%=���t        )��P	��D���A�#*

	conv_loss,L=���<        )��P	�D���A�#*

	conv_loss�D=�ᓠ        )��P	=�D���A�#*

	conv_loss�T�=���        )��P	�D���A�#*

	conv_loss��@=�m��        )��P	܃D���A�#*

	conv_loss�e"=�h��        )��P	��D���A�#*

	conv_loss A=�8�        )��P	cpD���A�#*

	conv_loss=�Ȕ�        )��P	��D���A�#*

	conv_loss��=�9�X        )��P	_^	D���A�#*

	conv_loss�7=@/�        )��P	��	D���A�#*

	conv_loss��W=��vq        )��P	�J
D���A�#*

	conv_loss%}=�>C        )��P	�
D���A�#*

	conv_lossc,=�7�        )��P	K1D���A�#*

	conv_loss���=I�r        )��P	D�D���A�#*

	conv_loss��=���        )��P	:D���A�#*

	conv_lossC��<����        )��P	w�D���A�#*

	conv_lossm�=��x�        )��P	D���A�#*

	conv_lossXB;=<��        )��P	�zD���A�#*

	conv_loss��{=��W/        )��P	��D���A�#*

	conv_lossc�=?��H        )��P	 hD���A�#*

	conv_loss��=d~x?        )��P	`�D���A�#*

	conv_loss){�<�?�        )��P	�SD���A�#*

	conv_loss�Q
=h-��        )��P	5D���A�#*

	conv_loss<�;=�ٍ        )��P	��D���A�#*

	conv_loss�p=B�ܦ        )��P	�D���A�#*

	conv_loss%C�=��vA        )��P	RyD���A�#*

	conv_loss��!=m�\        )��P	��D���A�#*

	conv_loss�=���        )��P	�pD���A�#*

	conv_loss
#�=_�O�        )��P	}�D���A�#*

	conv_lossmW=�cN        )��P	PhD���A�#*

	conv_loss��T=A�6�        )��P	9D���A�#*

	conv_loss�J=Re��        )��P	�zD���A�#*

	conv_loss��a=�St        )��P	�D���A�#*

	conv_loss�u=���        )��P	�D���A�#*

	conv_loss�BH={[~�        )��P	�QD���A�#*

	conv_losss$�=��V        )��P	��D���A�#*

	conv_loss�As=���        )��P	X�D���A�#*

	conv_loss�mR=|��        )��P	�D���A�#*

	conv_lossS�2=$}��        )��P	�]D���A�#*

	conv_lossn�w=�e}        )��P	��D���A�#*

	conv_lossǫG=Ω�9        )��P	��D���A�#*

	conv_loss!?_=Դ�        )��P	%D���A�#*

	conv_loss�n=)�]        )��P	�gD���A�#*

	conv_loss�PS='�j        )��P	��D���A�#*

	conv_loss[�=Y&�        )��P	�D���A�#*

	conv_loss���<����        )��P	�5D���A�#*

	conv_loss�wE=b�q�        )��P	�wD���A�#*

	conv_lossƹ�=Pց�        )��P	��D���A�#*

	conv_lossG-=�	��        )��P	r�D���A�#*

	conv_loss�TR=��r        )��P	�<D���A�#*

	conv_loss:�'=n_        )��P	>D���A�#*

	conv_loss@*=~. �        )��P	�D���A�#*

	conv_loss(<=H��        )��P	ED���A�#*

	conv_loss���<!��        )��P	/CD���A�#*

	conv_loss��=�h+�        )��P	|�D���A�#*

	conv_loss�|�<��}r        )��P	��D���A�#*

	conv_lossS5=�� 1        )��P	?D���A�#*

	conv_loss��<7�x        )��P	�ND���A�#*

	conv_loss�&=��_�        )��P	��D���A�#*

	conv_lossA~�<lc>�        )��P	��D���A�#*

	conv_loss�)=�7eP        )��P	VD���A�#*

	conv_loss��"=���P        )��P	�WD���A�#*

	conv_loss��4=��n        )��P	�D���A�#*

	conv_loss��v=�        )��P	5�D���A�#*

	conv_lossk�Q=���C        )��P	\!D���A�#*

	conv_loss�B=�:b#        )��P	"eD���A�#*

	conv_loss�X+=Yp�W        )��P	3�D���A�#*

	conv_loss]��=�{        )��P	��D���A�#*

	conv_loss�!-=��ɝ        )��P	�.D���A�#*

	conv_loss��N=SD        )��P	1qD���A�#*

	conv_loss8b=�j�        )��P	��D���A�#*

	conv_loss�)M=3���        )��P	��D���A�#*

	conv_loss��={��        )��P	�� D���A�#*

	conv_lossZ,=� U        )��P	� D���A�#*

	conv_loss[%I==�s�        )��P	�3!D���A�#*

	conv_loss�*=��Ȃ        )��P	�p!D���A�#*

	conv_loss^3<=�#��        )��P	��!D���A�#*

	conv_loss���<kը�        )��P	�!D���A�#*

	conv_lossq=~�        )��P	�!"D���A�#*

	conv_loss�S3=��H        )��P	0c"D���A�$*

	conv_loss?�^=�B        )��P	L�"D���A�$*

	conv_loss}V3=���[        )��P	��"D���A�$*

	conv_loss�c=��5        )��P	#D���A�$*

	conv_loss@~d=\�b>        )��P	�L#D���A�$*

	conv_loss�PI=�\�        )��P	�#D���A�$*

	conv_loss)�1=x#�)        )��P	��#D���A�$*

	conv_loss�h;=��&	        )��P	��#D���A�$*

	conv_loss�O=}q��        )��P	Q)$D���A�$*

	conv_loss}K= ��        )��P	�b$D���A�$*

	conv_lossЏ$=*$(�        )��P	��$D���A�$*

	conv_loss�&�<u�O        )��P	��$D���A�$*

	conv_loss�~=��DG        )��P	$%D���A�$*

	conv_losso7=��_�        )��P	(J%D���A�$*

	conv_loss�=��Z        )��P	��%D���A�$*

	conv_lossc{B=��K        )��P	�%D���A�$*

	conv_loss\d==!�        )��P	��%D���A�$*

	conv_loss} =��Y        )��P	�%&D���A�$*

	conv_loss�Bn=�r�        )��P	t\&D���A�$*

	conv_lossY�U=	L�         )��P	�&D���A�$*

	conv_loss�=E�        )��P	��&D���A�$*

	conv_loss�A=�b�F        )��P	c�&D���A�$*

	conv_loss!A=p�w        )��P	'6'D���A�$*

	conv_loss��6=�\ז        )��P	5j'D���A�$*

	conv_lossZ�F=���        )��P	�'D���A�$*

	conv_loss�==��6�        )��P	�'D���A�$*

	conv_loss�rc=����        )��P	�(D���A�$*

	conv_lossb�Y=G��        )��P	,D(D���A�$*

	conv_loss��J=T]a        )��P	�{(D���A�$*

	conv_loss��;=ݴ+        )��P	��(D���A�$*

	conv_loss!�a=65�        )��P	+�(D���A�$*

	conv_loss)$+=�).        )��P	s)D���A�$*

	conv_loss.-.=ī��        )��P	1Y)D���A�$*

	conv_loss��Y=�Eh�        )��P	z�)D���A�$*

	conv_loss�Q=sNu�        )��P	��)D���A�$*

	conv_loss	�;=���Y        )��P	,�)D���A�$*

	conv_loss/��=�$�        )��P	k4*D���A�$*

	conv_loss�	�=w��_        )��P	�k*D���A�$*

	conv_loss4?=tX        )��P	d�*D���A�$*

	conv_lossO]K=	&�        )��P	�*D���A�$*

	conv_loss��=�K��        )��P	�+D���A�$*

	conv_loss��=��        )��P	WE+D���A�$*

	conv_loss���<r�<w        )��P	E{+D���A�$*

	conv_loss���<�z�        )��P	�+D���A�$*

	conv_loss�>0=��E~        )��P	r�+D���A�$*

	conv_lossha6=��f�        )��P	�2,D���A�$*

	conv_loss>�r=�o�        )��P	�n,D���A�$*

	conv_loss�,�=�ݣp        )��P	�,D���A�$*

	conv_loss�4W=T@9H        )��P	�,D���A�$*

	conv_loss��E=�a��        )��P	�-D���A�$*

	conv_lossi�-=�>��        )��P	nF-D���A�$*

	conv_loss�
n=�p޹        )��P	k�-D���A�$*

	conv_loss��9=��l�        )��P	��-D���A�$*

	conv_loss��J=$�QK        )��P	��-D���A�$*

	conv_loss��M=�@!9        )��P	f5.D���A�$*

	conv_loss&�o=��f        )��P	�j.D���A�$*

	conv_loss�]W=��Sv        )��P	�.D���A�$*

	conv_loss�V=�)�\        )��P	h�.D���A�$*

	conv_lossW\"=�dx�        )��P	9*/D���A�$*

	conv_loss��(=ʳM[        )��P	<b/D���A�$*

	conv_lossޤ=J?�        )��P	�/D���A�$*

	conv_loss��=?��P        )��P	7�/D���A�$*

	conv_lossd�<=@\W        )��P	�0D���A�$*

	conv_loss�I=]CS        )��P	�>0D���A�$*

	conv_loss�k�=K�cV        )��P	�t0D���A�$*

	conv_lossϚ"=�&mw        )��P	<�0D���A�$*

	conv_lossN�q=���9        )��P	E�0D���A�$*

	conv_loss��g=�}'P        )��P	D1D���A�$*

	conv_lossg�?=Z��"        )��P	�K1D���A�$*

	conv_lossx�g=oo        )��P	s�1D���A�$*

	conv_loss&"=�H�4        )��P	�1D���A�$*

	conv_loss%#=��j        )��P	��1D���A�$*

	conv_loss��4=�;xs        )��P	�'2D���A�$*

	conv_loss��=�Z�-        )��P	r^2D���A�$*

	conv_lossa?M=�#        )��P	��2D���A�$*

	conv_loss[5D=��        )��P	��2D���A�$*

	conv_lossR�6=��N(        )��P	� 3D���A�$*

	conv_lossS*=���        )��P	�73D���A�$*

	conv_lossz%{=�*        )��P	�n3D���A�$*

	conv_loss`>n=��	        )��P	˥3D���A�$*

	conv_loss�b�<�B�        )��P	��3D���A�$*

	conv_loss-y�=����        )��P	�4D���A�$*

	conv_loss�<�=��<�        )��P	�M4D���A�$*

	conv_loss��1=o��9        )��P	5�4D���A�$*

	conv_loss��<�ĺ�        )��P	˺4D���A�$*

	conv_lossر?=h���        )��P	��4D���A�$*

	conv_loss�2h=��H�        )��P	�'5D���A�$*

	conv_lossS�!=	�r�        )��P	9^5D���A�$*

	conv_lossl%m=�� w        )��P	;�5D���A�$*

	conv_lossL��=$�w        )��P	�5D���A�$*

	conv_loss�>=�}��        )��P	�6D���A�$*

	conv_loss�,=RS��        )��P	�86D���A�$*

	conv_loss8_9=�%�        )��P	*o6D���A�$*

	conv_loss�\C=�PQ        )��P	8�6D���A�$*

	conv_loss��>=~n�        )��P	��6D���A�$*

	conv_loss�A3=TE�        )��P	&7D���A�$*

	conv_loss,�=��%?        )��P	ea7D���A�$*

	conv_loss��'=�Y��        )��P	]�7D���A�$*

	conv_loss��=���        )��P	{�7D���A�$*

	conv_loss�Y =d��@        )��P	P8D���A�$*

	conv_loss̒Q=��
        )��P	�Q8D���A�$*

	conv_loss�m:=�7mC        )��P	8D���A�$*

	conv_loss�DX=�/`        )��P	��8D���A�$*

	conv_loss�H=�u"A        )��P	>9D���A�$*

	conv_loss�yT=!        )��P	E9D���A�$*

	conv_loss��=a�7>        )��P	H|9D���A�$*

	conv_lossT�'=�5�        )��P	6�9D���A�$*

	conv_loss�OT=r�N        )��P	��9D���A�$*

	conv_loss�,=�[        )��P	R!:D���A�$*

	conv_loss��]=A��w        )��P	�X:D���A�$*

	conv_loss� I=�|�        )��P	�:D���A�$*

	conv_loss|s=^Al�        )��P	��:D���A�$*

	conv_loss�9=I�sP        )��P	�;D���A�$*

	conv_lossJXa=奪v        )��P	f8;D���A�$*

	conv_loss�Ϗ=��N        )��P	�p;D���A�$*

	conv_loss�=��.|        )��P	��;D���A�$*

	conv_lossُ	=c��        )��P	��;D���A�$*

	conv_loss��M=t�+        )��P	�<D���A�$*

	conv_loss�A=ds#�        )��P	>M<D���A�$*

	conv_loss�{�<k���        )��P	!�<D���A�$*

	conv_loss*�p=�QP�        )��P	��<D���A�$*

	conv_lossK�k=*f^�        )��P	e�<D���A�$*

	conv_loss�_i=&��        )��P	l*=D���A�$*

	conv_loss!=f[0�        )��P	_=D���A�$*

	conv_loss��=a�q        )��P	��=D���A�$*

	conv_losst�<!�
�        )��P	O�=D���A�$*

	conv_loss�)�=�#��        )��P	>D���A�$*

	conv_lossr`�<+gr�        )��P	=>D���A�$*

	conv_loss9tD=�a        )��P	�t>D���A�%*

	conv_loss�qn=D���        )��P	I�>D���A�%*

	conv_loss[z�<x�Ԧ        )��P	*�>D���A�%*

	conv_lossጋ=Za8.        )��P	I ?D���A�%*

	conv_lossK6=���        )��P	'T?D���A�%*

	conv_loss��@= c^�        )��P	��?D���A�%*

	conv_loss�W�=��@        )��P	��?D���A�%*

	conv_lossҁ5=�e �        )��P	>�?D���A�%*

	conv_lossT�=5���        )��P	�@D���A�%*

	conv_lossmq�=,�&        )��P	J@D���A�%*

	conv_lossG��<1���        )��P	�y@D���A�%*

	conv_loss�R=��ɺ        )��P	l�@D���A�%*

	conv_loss��
=RK��        )��P	��@D���A�%*

	conv_loss=�)=GEu�        )��P	�AD���A�%*

	conv_loss8<A=�'��        )��P	R8AD���A�%*

	conv_loss��=wte        )��P	
xAD���A�%*

	conv_loss�c�=E��        )��P	J�AD���A�%*

	conv_lossԹ*=���        )��P	��AD���A�%*

	conv_loss =1=�k	        )��P	qBD���A�%*

	conv_loss��"=׃?�        )��P	_ABD���A�%*

	conv_loss� |=X� �        )��P	�rBD���A�%*

	conv_loss)C=x��G        )��P	Z�BD���A�%*

	conv_lossE]=K��        )��P	'�BD���A�%*

	conv_loss5�[=xK�^        )��P	�CD���A�%*

	conv_loss��g=�(        )��P	XCD���A�%*

	conv_loss�{/=;nsv        )��P	��CD���A�%*

	conv_loss�KC=����        )��P	��CD���A�%*

	conv_loss�|�<9X�        )��P	�CD���A�%*

	conv_loss�#=Q��        )��P	<&DD���A�%*

	conv_loss��k=�(�        )��P	�TDD���A�%*

	conv_loss�B=ƫ�N        )��P	��DD���A�%*

	conv_lossJv=�Ы�        )��P	��DD���A�%*

	conv_loss&p4=��t        )��P	��DD���A�%*

	conv_loss:^M=�a�[        )��P	�ED���A�%*

	conv_loss؈�=+g�b        )��P	�BED���A�%*

	conv_loss�%?=���V        )��P	Y}ED���A�%*

	conv_loss�{=�N        )��P	V�ED���A�%*

	conv_lossKa7=����        )��P	F�ED���A�%*

	conv_loss^5�=iA��        )��P	�+FD���A�%*

	conv_lossG�*=�\��        )��P	 _FD���A�%*

	conv_loss	�N=���        )��P	?�FD���A�%*

	conv_loss��b=Ed�|        )��P	 �FD���A�%*

	conv_loss���=�{d�        )��P	z�FD���A�%*

	conv_loss�vY=p2�        )��P	�!GD���A�%*

	conv_lossԟ_=\R�E        )��P	XGD���A�%*

	conv_loss�8H=�ō        )��P	/�GD���A�%*

	conv_loss+%=\�e        )��P	{�GD���A�%*

	conv_losso�:=zؐl        )��P	H�GD���A�%*

	conv_loss-�<��        )��P	�HD���A�%*

	conv_loss�d*=�        )��P	QHD���A�%*

	conv_loss��D=�0W        )��P	P�HD���A�%*

	conv_lossk2=�ŋ�        )��P	5�HD���A�%*

	conv_loss9�m=1�W        )��P	,�HD���A�%*

	conv_loss�0=�p_�        )��P	2ID���A�%*

	conv_loss=�N=!�p        )��P	W>ID���A�%*

	conv_loss��5=D�\�        )��P	pID���A�%*

	conv_loss�^=V�L�        )��P	��ID���A�%*

	conv_loss��=n��        )��P	r�ID���A�%*

	conv_loss�:=C�w�        )��P	~�ID���A�%*

	conv_loss|O+=^�\         )��P	K,JD���A�%*

	conv_loss�6A=x$z�        )��P	�YJD���A�%*

	conv_loss^Ff=ȓ҃        )��P	��JD���A�%*

	conv_lossSG�<X�ӆ        )��P	�JD���A�%*

	conv_losseB{=9�>�        )��P	�JD���A�%*

	conv_loss��=!�t�        )��P	KD���A�%*

	conv_loss���<T��        )��P	��LD���A�%*

	conv_loss�ZL=Ծ	        )��P	R�LD���A�%*

	conv_loss1|=�)�        )��P	5MD���A�%*

	conv_losseZ=/-<        )��P	�LMD���A�%*

	conv_lossݢA=Ǭލ        )��P	]�MD���A�%*

	conv_loss��=?ң        )��P	��MD���A�%*

	conv_lossׯ�<V�/�        )��P	��MD���A�%*

	conv_loss�Q=Br�]        )��P	ND���A�%*

	conv_lossC�|=��u        )��P	�SND���A�%*

	conv_loss��=h�S        )��P	��ND���A�%*

	conv_loss��=���        )��P	�ND���A�%*

	conv_loss�Tu=4��        )��P	��ND���A�%*

	conv_lossR{e=�e�T        )��P	`!OD���A�%*

	conv_loss�1�<�Ȏ�        )��P	�OOD���A�%*

	conv_loss���=��<R        )��P	}�OD���A�%*

	conv_loss�>=c��s        )��P	R�OD���A�%*

	conv_loss�H�<�$��        )��P	��OD���A�%*

	conv_loss�f1=�:q        )��P	� PD���A�%*

	conv_loss]r*=�Е        )��P	cPPD���A�%*

	conv_loss
�=��        )��P	6PD���A�%*

	conv_loss�3=d�w�        )��P	��PD���A�%*

	conv_loss�Q=o�#        )��P	��PD���A�%*

	conv_loss�~�<���        )��P	cQD���A�%*

	conv_loss �=�M��        )��P	�BQD���A�%*

	conv_loss�)7=�^V        )��P	�rQD���A�%*

	conv_loss�4X=H�u�        )��P	��QD���A�%*

	conv_loss�V6=��+�        )��P	��QD���A�%*

	conv_lossS<T=�4aj        )��P	O�QD���A�%*

	conv_loss%9m=���        )��P	x.RD���A�%*

	conv_loss��=����        )��P	�_RD���A�%*

	conv_lossZ;�=td��        )��P	~�RD���A�%*

	conv_loss�\=�šY        )��P	�RD���A�%*

	conv_loss߲B=��/        )��P	��RD���A�%*

	conv_loss�==�C        )��P	�SD���A�%*

	conv_lossR�S==΍�        )��P	�LSD���A�%*

	conv_loss�Ea=���        )��P	�{SD���A�%*

	conv_lossŰc=v�7�        )��P	��SD���A�%*

	conv_loss,�>=��m�        )��P	e�SD���A�%*

	conv_loss�Zv=]v2        )��P	�
TD���A�%*

	conv_loss��<1��        )��P	�<TD���A�%*

	conv_loss=�=��        )��P		nTD���A�%*

	conv_lossK.5=��"�        )��P	��TD���A�%*

	conv_loss��=]:��        )��P	[�TD���A�%*

	conv_loss��l=���        )��P	Z�TD���A�%*

	conv_lossĽ/=�r�        )��P	�,UD���A�%*

	conv_loss"
#=J��%        )��P	[UD���A�%*

	conv_loss�4=b��        )��P	��UD���A�%*

	conv_loss��E=Z�-�        )��P	��UD���A�%*

	conv_lossʯ=SȄ        )��P	��UD���A�%*

	conv_loss�o�=.0n�        )��P	$VD���A�%*

	conv_loss9=M=�&Ӆ        )��P	�YVD���A�%*

	conv_lossQ�=�8Q        )��P	��VD���A�%*

	conv_loss;�=JҡB        )��P	��VD���A�%*

	conv_loss�(�<E;�,        )��P	e�VD���A�%*

	conv_loss��<�	�3        )��P	S WD���A�%*

	conv_lossi�J=���        )��P	�OWD���A�%*

	conv_loss.�=C��C        )��P	WD���A�%*

	conv_loss݁[=T�Ru        )��P	6�WD���A�%*

	conv_loss(Da=���        )��P	��WD���A�%*

	conv_loss;4=�;=�        )��P	(XD���A�%*

	conv_loss��6=��o�        )��P	�IXD���A�%*

	conv_loss��=�~٢        )��P	/yXD���A�%*

	conv_lossk`I=6:�z        )��P	�XD���A�%*

	conv_lossK��<�#�        )��P	��XD���A�&*

	conv_lossKc4=;W)�        )��P	SYD���A�&*

	conv_loss��(=�r�        )��P	lAYD���A�&*

	conv_loss�ym=v�        )��P	�rYD���A�&*

	conv_loss�w5=7�0+        )��P	a�YD���A�&*

	conv_loss��<&b�,        )��P	{�YD���A�&*

	conv_lossx=$=�#(        )��P	HZD���A�&*

	conv_loss(�f=���o        )��P	MDZD���A�&*

	conv_loss�� =}n%        )��P	xZD���A�&*

	conv_loss'FZ=�r�*        )��P	%�ZD���A�&*

	conv_lossy:N=��8        )��P	��ZD���A�&*

	conv_loss|�=����        )��P	�[D���A�&*

	conv_lossW��<��}w        )��P	�>[D���A�&*

	conv_lossf�B=�$rN        )��P	�o[D���A�&*

	conv_losspX=�]ݑ        )��P	G�[D���A�&*

	conv_loss+�=��D�        )��P	��[D���A�&*

	conv_loss��a=�K�f        )��P	\D���A�&*

	conv_loss��#=�:O�        )��P	n6\D���A�&*

	conv_loss��H=���        )��P	Bi\D���A�&*

	conv_lossN�2=|܄�        )��P	Ӛ\D���A�&*

	conv_lossr�V=�Ϥ�        )��P	 �\D���A�&*

	conv_loss���<qv�        )��P	��\D���A�&*

	conv_loss̟7=c��P        )��P	�,]D���A�&*

	conv_loss��=�!�        )��P	�\]D���A�&*

	conv_lossx�=�]�        )��P	R�]D���A�&*

	conv_loss��v=���W        )��P	�]D���A�&*

	conv_loss�_=�p1L        )��P	��]D���A�&*

	conv_loss���=s�u        )��P	R^D���A�&*

	conv_loss�3&=�y�L        )��P	N^D���A�&*

	conv_losstP�=�j�        )��P	�}^D���A�&*

	conv_loss7]W=�Ę        )��P	L�^D���A�&*

	conv_lossZ.$=�,��        )��P	h�^D���A�&*

	conv_loss�!=?���        )��P	�_D���A�&*

	conv_loss�� =��A@        )��P	FD_D���A�&*

	conv_loss� 2=�V�,        )��P	�u_D���A�&*

	conv_loss�==NU�,        )��P	��_D���A�&*

	conv_loss?a=ε�        )��P	��_D���A�&*

	conv_loss�=T-O2        )��P	�`D���A�&*

	conv_loss{I3=��kv        )��P	MJ`D���A�&*

	conv_losss�=���        )��P	l~`D���A�&*

	conv_loss�M4=l��        )��P	T�`D���A�&*

	conv_loss��|=gw�        )��P	O�`D���A�&*

	conv_loss��1=q|{        )��P	%aD���A�&*

	conv_loss�=�9��        )��P	�baD���A�&*

	conv_loss"�&=�"�c        )��P	�aD���A�&*

	conv_loss�'&={b~T        )��P	m�aD���A�&*

	conv_loss��X=�<�        )��P	��aD���A�&*

	conv_loss��#=���.        )��P	�5bD���A�&*

	conv_lossD�F=Έ|!        )��P	�fbD���A�&*

	conv_loss��5=v+_%        )��P	��bD���A�&*

	conv_loss���=\tھ        )��P	��bD���A�&*

	conv_lossMx'=��*        )��P	�bD���A�&*

	conv_lossJJ�=`#M\        )��P	�0cD���A�&*

	conv_loss�=f݈v        )��P	NacD���A�&*

	conv_loss�d=��        )��P	�cD���A�&*

	conv_loss�c=��        )��P	��cD���A�&*

	conv_loss�O=����        )��P	�dD���A�&*

	conv_loss�8=o�;;        )��P	�9dD���A�&*

	conv_losshȃ=��o0        )��P	�kdD���A�&*

	conv_loss��W=V\[        )��P	�dD���A�&*

	conv_loss�bg=�o�w        )��P	1�dD���A�&*

	conv_loss,0=�X��        )��P	b�dD���A�&*

	conv_loss"�7=�c�H        )��P	q.eD���A�&*

	conv_loss���<����        )��P	�_eD���A�&*

	conv_loss=�s=���        )��P	�eD���A�&*

	conv_loss�1W=_��        )��P	�eD���A�&*

	conv_lossMN =Դ�        )��P	��eD���A�&*

	conv_loss��t=�	b�        )��P	#fD���A�&*

	conv_loss�"=�Am        )��P	�RfD���A�&*

	conv_loss�=�غ        )��P	(�fD���A�&*

	conv_loss��/=�        )��P	��fD���A�&*

	conv_loss��I=�6T�        )��P	��fD���A�&*

	conv_losso =���        )��P	_gD���A�&*

	conv_loss���<��2h        )��P	�EgD���A�&*

	conv_loss�!�=W�        )��P	�xgD���A�&*

	conv_lossL�C="��        )��P	 �gD���A�&*

	conv_losse�I=�^��        )��P	��gD���A�&*

	conv_loss�d=���        )��P	�hD���A�&*

	conv_loss�O=�#9        )��P	�ChD���A�&*

	conv_loss��i=����        )��P	�shD���A�&*

	conv_loss�Z7=Zב        )��P	��hD���A�&*

	conv_lossp�=eT9�        )��P	�hD���A�&*

	conv_loss-�=�6G        )��P	OiD���A�&*

	conv_loss�Y=yTP�        )��P	�7iD���A�&*

	conv_loss��+=�G�;        )��P	hiD���A�&*

	conv_lossQ!r=BG�s        )��P	��iD���A�&*

	conv_loss��X=o$GO        )��P	��iD���A�&*

	conv_loss��S=R�D        )��P	$jD���A�&*

	conv_loss��2= �r        )��P	L?jD���A�&*

	conv_loss��<�`�        )��P	4ojD���A�&*

	conv_loss{�=�s�        )��P	�jD���A�&*

	conv_loss��$=0r�        )��P	`�jD���A�&*

	conv_loss�3%=���^        )��P	okD���A�&*

	conv_losseR�=07�P        )��P	�BkD���A�&*

	conv_loss�Q�<�	�\        )��P	��kD���A�&*

	conv_loss�@0=<�        )��P	�kD���A�&*

	conv_loss�=��9�        )��P	B�kD���A�&*

	conv_loss,�7=�%        )��P	�!lD���A�&*

	conv_loss�Z=��Y        )��P	�RlD���A�&*

	conv_loss<A=~h��        )��P	�lD���A�&*

	conv_loss��A=$	�v        )��P	��lD���A�&*

	conv_loss=SB=�uۨ        )��P	 �lD���A�&*

	conv_loss�w=��G        )��P	�$mD���A�&*

	conv_lossX�E=j���        )��P	�XmD���A�&*

	conv_loss|"!=J�e�        )��P	��mD���A�&*

	conv_loss)�=�@`a        )��P	l�mD���A�&*

	conv_loss�A=��Ձ        )��P	�mD���A�&*

	conv_loss��<T��9        )��P	�-nD���A�&*

	conv_loss��)=�`�	        )��P	VanD���A�&*

	conv_loss�=J_�~        )��P	�nD���A�&*

	conv_loss��+=�P��        )��P	��nD���A�&*

	conv_loss{w=�g�        )��P	,�nD���A�&*

	conv_loss3:�=x�Z        )��P	:,oD���A�&*

	conv_lossB�=�۸�        )��P	�`oD���A�&*

	conv_loss�*=��o$        )��P	��oD���A�&*

	conv_loss\�8=C*��        )��P	��oD���A�&*

	conv_loss��=�)_        )��P	_�oD���A�&*

	conv_loss�2W=�'��        )��P	�*pD���A�&*

	conv_lossR?2=7�e        )��P	�`pD���A�&*

	conv_loss� L=�K��        )��P	5�pD���A�&*

	conv_loss�N =13)        )��P	��pD���A�&*

	conv_lossH�<�H�        )��P	�qD���A�&*

	conv_loss��<c7�o        )��P	(2qD���A�&*

	conv_loss��K=%�_+        )��P	�dqD���A�&*

	conv_lossv| =�=�        )��P	ģqD���A�&*

	conv_loss�V=RN%�        )��P	��qD���A�&*

	conv_loss�*=��;        )��P	�rD���A�&*

	conv_loss��I=���        )��P	�7rD���A�&*

	conv_lossot=S��        )��P	�irD���A�'*

	conv_loss�R;=�V�        )��P	;�rD���A�'*

	conv_lossF�"=�yt�        )��P	P�rD���A�'*

	conv_lossu=c.        )��P	T�rD���A�'*

	conv_loss��1=I���        )��P	�+sD���A�'*

	conv_loss[=�Ϙ        )��P	�[sD���A�'*

	conv_loss�2=���        )��P	��sD���A�'*

	conv_loss�.='���        )��P	˼sD���A�'*

	conv_loss(��<��Պ        )��P	�[xD���A�'*

	conv_loss��F=���        )��P	��yD���A�'*

	conv_lossi�u=�Ȁ�        )��P	w&zD���A�'*

	conv_loss�&!=�c��        )��P	@UzD���A�'*

	conv_loss�=<��        )��P	u�zD���A�'*

	conv_lossslF=����        )��P	��zD���A�'*

	conv_loss��u=-��        )��P	��zD���A�'*

	conv_loss�x==k�        )��P	�!{D���A�'*

	conv_loss�<�\$        )��P	�P{D���A�'*

	conv_loss�xd=R{�H        )��P	]{D���A�'*

	conv_loss�>.= ��        )��P	-�{D���A�'*

	conv_lossH=�}�        )��P	�{D���A�'*

	conv_lossw0=��Ϗ        )��P	�|D���A�'*

	conv_loss��<�i�        )��P	LL|D���A�'*

	conv_loss�B�<�z6Z        )��P	{|D���A�'*

	conv_loss�ʋ=3�        )��P	J�|D���A�'*

	conv_loss-�{=��ع        )��P	�|D���A�'*

	conv_loss/��<%dB�        )��P	�#}D���A�'*

	conv_loss�gu=���        )��P	hW}D���A�'*

	conv_loss{`=��v�        )��P	:�}D���A�'*

	conv_loss�J�<ćE        )��P	Q�}D���A�'*

	conv_loss/G2={        )��P	��}D���A�'*

	conv_loss��)=Y��        )��P	�$~D���A�'*

	conv_lossA�=*��C        )��P	_T~D���A�'*

	conv_loss�F�=����        )��P	S�~D���A�'*

	conv_loss���<�}r�        )��P	L�~D���A�'*

	conv_loss՛�<�S�*        )��P	��~D���A�'*

	conv_loss�t3=�o        )��P	9D���A�'*

	conv_loss̨U=m5#C        )��P	�ED���A�'*

	conv_loss�Y\=D��.        )��P	CwD���A�'*

	conv_loss��5=� NN        )��P	��D���A�'*

	conv_lossB4=Ӆ>�        )��P	 �D���A�'*

	conv_loss�D=��p�        )��P	��D���A�'*

	conv_loss��!=��G        )��P	�K�D���A�'*

	conv_lossk$=#�*        )��P	�y�D���A�'*

	conv_loss,��<��&\        )��P	)��D���A�'*

	conv_lossX�=ӱ&�        )��P	�؀D���A�'*

	conv_loss�u�=�^ݓ        )��P	��D���A�'*

	conv_losse=l��C        )��P	7A�D���A�'*

	conv_lossk�5=���Q        )��P	�s�D���A�'*

	conv_loss�;~=���        )��P	o��D���A�'*

	conv_loss=(�¦        )��P	��D���A�'*

	conv_lossZ�=# �        )��P	^ �D���A�'*

	conv_lossr�{=��_        )��P	!O�D���A�'*

	conv_loss�R=n@-K        )��P	P�D���A�'*

	conv_loss�z�<��T        )��P	5��D���A�'*

	conv_loss��	=ȉ��        )��P	���D���A�'*

	conv_lossLA3=O{v        )��P	� �D���A�'*

	conv_loss 0x=���        )��P	�R�D���A�'*

	conv_loss�/A=P���        )��P	y��D���A�'*

	conv_lossP�?=�/d        )��P	!��D���A�'*

	conv_loss
=r��        )��P	�D���A�'*

	conv_loss?>1=H(<        )��P	*�D���A�'*

	conv_loss{^C=��}�        )��P	�i�D���A�'*

	conv_loss��f=ڋ        )��P	ʟ�D���A�'*

	conv_lossP�	=@��        )��P	�ЄD���A�'*

	conv_loss٣N=�        )��P	���D���A�'*

	conv_loss���=#�f�        )��P	R8�D���A�'*

	conv_loss�zZ=�Q~        )��P	�k�D���A�'*

	conv_loss��3= �V        )��P	�D���A�'*

	conv_loss.4=xϿ        )��P	�ͅD���A�'*

	conv_loss<�<�#        )��P	�
�D���A�'*

	conv_lossn�=��Ef        )��P	=J�D���A�'*

	conv_lossE+=��UJ        )��P	{}�D���A�'*

	conv_loss�~D=	�d�        )��P	E��D���A�'*

	conv_lossM�d=/(y�        )��P	��D���A�'*

	conv_loss�=�dx�        )��P	��D���A�'*

	conv_loss(�=�ΐ        )��P	RW�D���A�'*

	conv_loss��=�-�a        )��P	N��D���A�'*

	conv_loss��4=��i�        )��P	4��D���A�'*

	conv_lossEŋ=���}        )��P	�D���A�'*

	conv_loss=��        )��P	=�D���A�'*

	conv_loss��=B        )��P	n\�D���A�'*

	conv_loss>�3=��7        )��P	��D���A�'*

	conv_loss��Z=c,z�        )��P	AÈD���A�'*

	conv_loss�/=�7��        )��P	��D���A�'*

	conv_loss:v,=�4�<        )��P	~!�D���A�'*

	conv_loss�:�<&-��        )��P	CS�D���A�'*

	conv_lossT��=�@�^        )��P	��D���A�'*

	conv_loss�>=�V�+        )��P	�ЉD���A�'*

	conv_loss�́=��1h        )��P	��D���A�'*

	conv_loss��=���        )��P	_2�D���A�'*

	conv_lossO�=y��        )��P	�b�D���A�'*

	conv_loss�J=���        )��P	��D���A�'*

	conv_loss�=^�5Q        )��P	�͊D���A�'*

	conv_lossY=�z��        )��P	���D���A�'*

	conv_loss;=����        )��P	�*�D���A�'*

	conv_lossҤ.=M��        )��P	�Z�D���A�'*

	conv_loss4=f-we        )��P	��D���A�'*

	conv_lossۖ�<�|Or        )��P	���D���A�'*

	conv_loss�==�&w�        )��P	B��D���A�'*

	conv_loss1�f=s�MP        )��P	�0�D���A�'*

	conv_loss53 =d-�        )��P	6`�D���A�'*

	conv_loss�[=;cб        )��P	;��D���A�'*

	conv_loss&�=��y�        )��P	�ǌD���A�'*

	conv_loss�=��s        )��P	-��D���A�'*

	conv_loss��8=�݄        )��P	�(�D���A�'*

	conv_lossP��<�@L        )��P	qW�D���A�'*

	conv_lossv�f=�_e2        )��P	ِ�D���A�'*

	conv_loss�y�=s��        )��P	ōD���A�'*

	conv_lossjE=|�/        )��P	��D���A�'*

	conv_loss�=�׈        )��P	n7�D���A�'*

	conv_loss��(=K�:        )��P	<l�D���A�'*

	conv_loss^H
=L��        )��P	g��D���A�'*

	conv_lossp=���        )��P	4ԎD���A�'*

	conv_loss�D=�.f�        )��P	�	�D���A�'*

	conv_loss;�=��S        )��P	 =�D���A�'*

	conv_lossg� =F�         )��P	l�D���A�'*

	conv_loss�f9=�1��        )��P	���D���A�'*

	conv_loss�U=V��u        )��P	�ݏD���A�'*

	conv_loss��f=J$��        )��P	��D���A�'*

	conv_loss
�=���        )��P	�F�D���A�'*

	conv_loss�R#=���        )��P	*u�D���A�'*

	conv_loss*=�Kצ        )��P	7��D���A�'*

	conv_loss<�<�C��        )��P	��D���A�'*

	conv_lossѩ�=��=4        )��P	F+�D���A�'*

	conv_loss��7=����        )��P	�]�D���A�'*

	conv_lossz�T=w��        )��P	�D���A�'*

	conv_loss�M7=��Ѓ        )��P	�ƑD���A�'*

	conv_lossj�=��:M        )��P	���D���A�'*

	conv_lossP�<�0u        )��P	.�D���A�(*

	conv_loss��=���        )��P	�^�D���A�(*

	conv_loss/O�< �>        )��P	q��D���A�(*

	conv_lossR�"=!��0        )��P	�ɒD���A�(*

	conv_loss�0=&��.        )��P	���D���A�(*

	conv_loss9�k=B��[        )��P	_2�D���A�(*

	conv_lossTX1=|)�        )��P	<h�D���A�(*

	conv_loss��<�Ɩ3        )��P	N��D���A�(*

	conv_loss��==�.c�        )��P	�ՓD���A�(*

	conv_lossǴk=��n        )��P	��D���A�(*

	conv_loss��{=E�	5        )��P	�6�D���A�(*

	conv_loss+ =gy��        )��P	�j�D���A�(*

	conv_loss��H=-r��        )��P	B��D���A�(*

	conv_loss.�1=O�y_        )��P	�ҔD���A�(*

	conv_loss�|=Q�        )��P	a�D���A�(*

	conv_loss�8=M]��        )��P	�4�D���A�(*

	conv_loss=`
=���        )��P	�e�D���A�(*

	conv_loss5v=���        )��P	 ��D���A�(*

	conv_loss�Z!= qժ        )��P	UƕD���A�(*

	conv_losss��=�tkg        )��P	p��D���A�(*

	conv_loss�=�;)�        )��P	\E�D���A�(*

	conv_loss07=�S�8        )��P	w}�D���A�(*

	conv_loss�O=��O�        )��P	-��D���A�(*

	conv_losslV�<أ>X        )��P	�D���A�(*

	conv_loss:b=䓏         )��P	b�D���A�(*

	conv_loss�`=��Q�        )��P	�C�D���A�(*

	conv_loss<UV=t.%�        )��P	@t�D���A�(*

	conv_loss+�<9sY!        )��P	N��D���A�(*

	conv_loss�$�<���        )��P	�ݗD���A�(*

	conv_loss�=�<�,�t        )��P	��D���A�(*

	conv_lossW�J=�w��        )��P	�;�D���A�(*

	conv_loss�5=�Y�        )��P	D}�D���A�(*

	conv_lossZ={�	Q        )��P	��D���A�(*

	conv_loss��	=Κ��        )��P	I�D���A�(*

	conv_loss�-=�]�        )��P	6!�D���A�(*

	conv_loss/� =i:W�        )��P	%Q�D���A�(*

	conv_lossxV[=|���        )��P	���D���A�(*

	conv_loss���=t��#        )��P	f��D���A�(*

	conv_loss�H9=/        )��P	{�D���A�(*

	conv_lossk"=�'(�        )��P	��D���A�(*

	conv_loss�r =����        )��P	P�D���A�(*

	conv_loss��<i�"        )��P	��D���A�(*

	conv_loss��!=��        )��P	U��D���A�(*

	conv_lossiEy=JB��        )��P	*��D���A�(*

	conv_loss'"=7h�        )��P	%-�D���A�(*

	conv_loss�z:=N�h�        )��P	�^�D���A�(*

	conv_loss"=�� k        )��P	ꏛD���A�(*

	conv_loss}�M=��        )��P	�ǛD���A�(*

	conv_lossD	= _z        )��P	O�D���A�(*

	conv_loss��<����        )��P	�2�D���A�(*

	conv_losss�N=I��0        )��P	�b�D���A�(*

	conv_loss��<���        )��P	���D���A�(*

	conv_loss�k=� @,        )��P	ŜD���A�(*

	conv_loss�� =����        )��P	o�D���A�(*

	conv_losso�=_c��        )��P	�5�D���A�(*

	conv_loss�d�<���        )��P	�h�D���A�(*

	conv_loss���<��        )��P	[��D���A�(*

	conv_loss���<a�2        )��P	TҝD���A�(*

	conv_loss?�!=�%        )��P	r�D���A�(*

	conv_loss3)�<��^        )��P	�C�D���A�(*

	conv_lossZ�P=P��        )��P	)s�D���A�(*

	conv_loss�2=��S�        )��P	椞D���A�(*

	conv_loss0=V��'        )��P	�֞D���A�(*

	conv_loss�<=h;��        )��P	��D���A�(*

	conv_loss=�=�Sc        )��P	(H�D���A�(*

	conv_loss��<�aC`        )��P	�v�D���A�(*

	conv_loss��<=�1�        )��P	ᥟD���A�(*

	conv_loss~.,=�+�)        )��P	0۟D���A�(*

	conv_loss|4=5���        )��P	Q�D���A�(*

	conv_lossBs�=RJ<        )��P	�P�D���A�(*

	conv_loss^}P=?沓        )��P	{��D���A�(*

	conv_lossU��<8���        )��P	��D���A�(*

	conv_loss�0F=Y���        )��P	��D���A�(*

	conv_loss}�=�A�5        )��P	)�D���A�(*

	conv_lossz�=2        )��P		Z�D���A�(*

	conv_lossl�=�n��        )��P	���D���A�(*

	conv_loss�{== S�        )��P	w��D���A�(*

	conv_loss4=�;        )��P	�D���A�(*

	conv_loss�!=�M��        )��P	�!�D���A�(*

	conv_loss�7E=�U�1        )��P	�S�D���A�(*

	conv_loss�l&=��sg        )��P	z��D���A�(*

	conv_loss�c�={�7B        )��P	�Z�D���A�(*

	conv_loss�= 	��        )��P	V��D���A�(*

	conv_loss|�#=��)        )��P	���D���A�(*

	conv_loss�z=@���        )��P	�D���A�(*

	conv_loss~�{=`�(Q        )��P	g�D���A�(*

	conv_loss*�T=F��'        )��P	�h�D���A�(*

	conv_loss&�p=����        )��P	D���A�(*

	conv_loss0`P="v;�        )��P	kΥD���A�(*

	conv_lossȚ$=B�;	        )��P	��D���A�(*

	conv_loss5 =W|�M        )��P	�9�D���A�(*

	conv_lossÿ*=e�y        )��P	fu�D���A�(*

	conv_loss��<���        )��P	��D���A�(*

	conv_loss�)=*��        )��P	i�D���A�(*

	conv_loss�3=g<�        )��P	k�D���A�(*

	conv_loss�[[=�h�        )��P	rS�D���A�(*

	conv_loss��=(=�        )��P	U��D���A�(*

	conv_loss�r=�qI�        )��P	伧D���A�(*

	conv_loss��=*�hp        )��P	>��D���A�(*

	conv_loss�+=ѓ�        )��P	o*�D���A�(*

	conv_loss$"=+2�^        )��P	�[�D���A�(*

	conv_loss��v=|.        )��P	��D���A�(*

	conv_loss��<!�        )��P	�ĨD���A�(*

	conv_loss�_=��
�        )��P	!��D���A�(*

	conv_loss�m2=��-h        )��P	�,�D���A�(*

	conv_loss�k=0|c        )��P	`�D���A�(*

	conv_loss��=���        )��P	���D���A�(*

	conv_loss��@=5�        )��P	ѩD���A�(*

	conv_loss/�+=�a        )��P	��D���A�(*

	conv_loss#�=gw	�        )��P	`4�D���A�(*

	conv_loss�;==96�        )��P	ml�D���A�(*

	conv_loss�]=�阶        )��P	`��D���A�(*

	conv_lossW�!=�P�        )��P	��D���A�(*

	conv_loss/�=?�l        )��P	I;�D���A�(*

	conv_lossU�=�G?        )��P	�l�D���A�(*

	conv_loss��<L�jo        )��P	W��D���A�(*

	conv_loss�!=��V�        )��P	ګD���A�(*

	conv_loss�+�<i��N        )��P		
�D���A�(*

	conv_loss4#=�BH�        )��P	v8�D���A�(*

	conv_loss��=nw�        )��P	j�D���A�(*

	conv_loss�4=��/�        )��P	���D���A�(*

	conv_lossϫ5=~Rp        )��P	�ݬD���A�(*

	conv_lossJ	=!��,        )��P	��D���A�(*

	conv_loss�<4hʄ        )��P	DK�D���A�(*

	conv_loss�D=�#W        )��P	�z�D���A�(*

	conv_loss��	=����        )��P	n��D���A�(*

	conv_loss�=rGJ>        )��P	Y�D���A�(*

	conv_lossp|2=��-�        )��P	��D���A�(*

	conv_losss��=�!EB        )��P	�G�D���A�)*

	conv_lossI�;=����        )��P	�w�D���A�)*

	conv_loss�B=�k��        )��P	���D���A�)*

	conv_loss9�<o���        )��P	v�D���A�)*

	conv_loss-!N=��ސ        )��P	{"�D���A�)*

	conv_loss��<��        )��P	�W�D���A�)*

	conv_loss��<�ok        )��P	���D���A�)*

	conv_loss��;=��Os        )��P	>دD���A�)*

	conv_lossӊM=�1        )��P	�	�D���A�)*

	conv_loss��;=#n        )��P	?:�D���A�)*

	conv_loss�.=�&�        )��P	�j�D���A�)*

	conv_loss R�<��'�        )��P	=��D���A�)*

	conv_loss�=S��        )��P	R�D���A�)*

	conv_loss��8=��FV        )��P	��D���A�)*

	conv_lossL�<�&��        )��P	�G�D���A�)*

	conv_lossxG�<\�H        )��P	�u�D���A�)*

	conv_lossQU�<�+�G        )��P	S��D���A�)*

	conv_loss!�<s�,d        )��P	K�D���A�)*

	conv_loss�=��w8        )��P	��D���A�)*

	conv_loss�f=4�nc        )��P	�F�D���A�)*

	conv_loss�b=����        )��P	 v�D���A�)*

	conv_loss��"=M���        )��P	���D���A�)*

	conv_loss��1=ߟ/        )��P	��D���A�)*

	conv_lossK	t=��L        )��P	_�D���A�)*

	conv_loss�?=���b        )��P	;E�D���A�)*

	conv_lossH#=zs�X        )��P	�t�D���A�)*

	conv_loss��,=×        )��P	O��D���A�)*

	conv_loss�)E=m�R>        )��P	�ѳD���A�)*

	conv_loss��r=]���        )��P	 �D���A�)*

	conv_loss�[7=M;        )��P	�/�D���A�)*

	conv_loss�b=Jw@N        )��P	�`�D���A�)*

	conv_lossv$=z�16        )��P	퐴D���A�)*

	conv_loss���<���I        )��P	���D���A�)*

	conv_loss���=�G��        )��P	��D���A�)*

	conv_losslW-=,5�        )��P	�!�D���A�)*

	conv_lossDl =�;        )��P	�P�D���A�)*

	conv_loss'q$=��H�        )��P	��D���A�)*

	conv_loss�FT= �]        )��P	ղ�D���A�)*

	conv_lossJcR=KC_y        )��P	��D���A�)*

	conv_lossf�=+[        )��P	�D���A�)*

	conv_lossˎ�<r���        )��P	>�D���A�)*

	conv_loss�'=����        )��P	�l�D���A�)*

	conv_lossq�/=p��        )��P	��D���A�)*

	conv_lossא�=�O        )��P	k̶D���A�)*

	conv_loss܋V=`��        )��P	���D���A�)*

	conv_lossV�X=@��        )��P	X+�D���A�)*

	conv_loss�w=]R�        )��P	�Z�D���A�)*

	conv_loss"_=�~��        )��P	��D���A�)*

	conv_loss1�N=�ef�        )��P	K��D���A�)*

	conv_loss�s=0���        )��P	-�D���A�)*

	conv_loss8J*=j�        )��P	<�D���A�)*

	conv_lossL��<woԾ        )��P	 H�D���A�)*

	conv_loss�g-=���        )��P	x�D���A�)*

	conv_lossY`=^�]l        )��P	e��D���A�)*

	conv_loss.�5=`H��        )��P	���D���A�)*

	conv_loss��,=�ky        )��P	��D���A�)*

	conv_loss�= #~�        )��P	Q�D���A�)*

	conv_loss���<<���        )��P	}��D���A�)*

	conv_loss1vc=��v3        )��P	t��D���A�)*

	conv_lossf+=�)�-        )��P	4�D���A�)*

	conv_loss�N=���        )��P	0�D���A�)*

	conv_loss�k�<ɟC�        )��P	+V�D���A�)*

	conv_loss'� =�қ�        )��P	"��D���A�)*

	conv_loss��U=��g�        )��P	�ĺD���A�)*

	conv_loss�`=�1+        )��P	��D���A�)*

	conv_loss�A&=�!�T        )��P	>+�D���A�)*

	conv_loss0\h=��7u        )��P	)\�D���A�)*

	conv_loss?��<�8sD        )��P	b��D���A�)*

	conv_loss7�p=�Yp        )��P	g��D���A�)*

	conv_loss��=���<        )��P	���D���A�)*

	conv_loss�/=�q�        )��P	�-�D���A�)*

	conv_loss��&=!��v        )��P	�\�D���A�)*

	conv_losseW=�m#        )��P	ދ�D���A�)*

	conv_loss�9=\��7        )��P	.��D���A�)*

	conv_lossY�B=c��        )��P	��D���A�)*

	conv_loss
��<��zw        )��P	O�D���A�)*

	conv_loss^p�<�u�L        )��P	M�D���A�)*

	conv_loss%G=�Ed�        )��P	�|�D���A�)*

	conv_lossH<1=����        )��P	{��D���A�)*

	conv_loss7=�.`{        )��P	�߽D���A�)*

	conv_lossD�J=z�ւ        )��P	��D���A�)*

	conv_lossM�M=5^�:        )��P	1D�D���A�)*

	conv_loss�I�<�s        )��P	rw�D���A�)*

	conv_loss�>=�Ә�        )��P	}��D���A�)*

	conv_loss�=�*        )��P	�׾D���A�)*

	conv_loss�{]={���        )��P	�	�D���A�)*

	conv_loss�?&=�*        )��P	\:�D���A�)*

	conv_loss�=!W�        )��P	(k�D���A�)*

	conv_loss\�=���        )��P	њ�D���A�)*

	conv_loss�,k=F�1,        )��P	ʿD���A�)*

	conv_loss��K=1f�_        )��P	��D���A�)*

	conv_loss��=Z��
        )��P	~*�D���A�)*

	conv_loss;$=?�S�        )��P	�]�D���A�)*

	conv_lossѳh=-?Z        )��P	k��D���A�)*

	conv_loss��<��#�        )��P	3��D���A�)*

	conv_loss+=��2�        )��P	��D���A�)*

	conv_lossvH?=��g        )��P	�5�D���A�)*

	conv_loss�D"=�T��        )��P	+g�D���A�)*

	conv_loss���<9�        )��P	=��D���A�)*

	conv_lossX(�<a`�S        )��P	���D���A�)*

	conv_loss��M=~'k9        )��P	���D���A�)*

	conv_lossj=E��        )��P	�(�D���A�)*

	conv_loss��=�x��        )��P	�Y�D���A�)*

	conv_loss�=��^        )��P	B��D���A�)*

	conv_loss�� =���        )��P	B��D���A�)*

	conv_loss��=g+�        )��P	Y �D���A�)*

	conv_loss�qX=(��        )��P	u3�D���A�)*

	conv_loss��=6�i        )��P	�e�D���A�)*

	conv_lossV��="��        )��P	���D���A�)*

	conv_loss��'=��S�        )��P	���D���A�)*

	conv_loss�==�#$        )��P	o�D���A�)*

	conv_loss�X=1
�        )��P	k8�D���A�)*

	conv_loss��=��00        )��P	
{�D���A�)*

	conv_lossJ;r=3.0r        )��P	��D���A�)*

	conv_loss�S=        )��P	���D���A�)*

	conv_loss �u=v        )��P	��D���A�)*

	conv_lossF'=��x        )��P	>�D���A�)*

	conv_loss��=�� /        )��P	�r�D���A�)*

	conv_loss^:=�m|K        )��P	���D���A�)*

	conv_loss=g3=${T�        )��P	T��D���A�)*

	conv_loss�
=��x        )��P	��D���A�)*

	conv_loss��=��        )��P	0=�D���A�)*

	conv_lossF�a=���        )��P	 n�D���A�)*

	conv_losse!=���        )��P	���D���A�)*

	conv_lossK�C=jv�%        )��P	5��D���A�)*

	conv_lossʤ!=�WP*        )��P	-��D���A�)*

	conv_lossCX=91U        )��P	m.�D���A�)*

	conv_loss�=k�J�        )��P	]`�D���A�)*

	conv_lossA9=�?&[        )��P	s��D���A�**

	conv_loss?R="��        )��P	U��D���A�**

	conv_loss=��=�D�        )��P	���D���A�**

	conv_losst��<���        )��P	O)�D���A�**

	conv_lossb =�ֿ        )��P	�Z�D���A�**

	conv_loss��=ͧ��        )��P	�D���A�**

	conv_loss��Y=M�P        )��P	E��D���A�**

	conv_loss�[=y- �        )��P	���D���A�**

	conv_lossw�P=k`_        )��P	�!�D���A�**

	conv_loss��=�h�        )��P	=P�D���A�**

	conv_loss?f�<���        )��P	���D���A�**

	conv_lossY�<<c��        )��P	[��D���A�**

	conv_loss�(=I���        )��P	L��D���A�**

	conv_loss�}�<0b�w        )��P	��D���A�**

	conv_loss��F=ʬ        )��P	hI�D���A�**

	conv_lossT8=��        )��P	�z�D���A�**

	conv_loss\�M=��        )��P	���D���A�**

	conv_loss��l=���x        )��P	���D���A�**

	conv_loss`s*=K?        )��P	��D���A�**

	conv_loss��"=�X��        )��P	�H�D���A�**

	conv_loss��!=`��        )��P	�~�D���A�**

	conv_loss��c=��N!        )��P	���D���A�**

	conv_loss˴6=x�QG        )��P	���D���A�**

	conv_loss��=[
�        )��P	��D���A�**

	conv_loss�Q	=��M        )��P	�F�D���A�**

	conv_loss���<v��        )��P	e��D���A�**

	conv_loss��%=�0�v        )��P	D�D���A�**

	conv_loss�,=VT֏        )��P	s<�D���A�**

	conv_lossz�`=0�f�        )��P	no�D���A�**

	conv_loss��=鄌�        )��P	���D���A�**

	conv_loss�QG=�(�#        )��P	+��D���A�**

	conv_loss=�^        )��P	��D���A�**

	conv_loss���=�X׻        )��P	C2�D���A�**

	conv_lossѮ3=�D��        )��P	�l�D���A�**

	conv_loss+8A=/��        )��P	���D���A�**

	conv_loss�i==��        )��P	!��D���A�**

	conv_loss��<��C�        )��P	��D���A�**

	conv_loss�=#��        )��P	�F�D���A�**

	conv_loss='S=�C��        )��P	%}�D���A�**

	conv_loss��<}Q��        )��P	���D���A�**

	conv_loss�b2=�=        )��P	���D���A�**

	conv_loss|*=� �        )��P	/-�D���A�**

	conv_loss��.=��M_        )��P	�]�D���A�**

	conv_loss�=�=;�        )��P	���D���A�**

	conv_lossG�/=��        )��P	 ��D���A�**

	conv_loss��=�O��        )��P	���D���A�**

	conv_lossL��=DS        )��P	�!�D���A�**

	conv_loss'8=����        )��P	�S�D���A�**

	conv_loss��=ry�        )��P	`��D���A�**

	conv_lossE=��r        )��P	���D���A�**

	conv_loss��=-Wz{        )��P	��D���A�**

	conv_loss�]=�_�        )��P	��D���A�**

	conv_lossJ �<G���        )��P	oK�D���A�**

	conv_loss��)=(�R�        )��P	�{�D���A�**

	conv_loss��n=50��        )��P	U��D���A�**

	conv_lossJ�)=)�x        )��P	X��D���A�**

	conv_loss��q=�{�w        )��P	��D���A�**

	conv_loss�� =F�        )��P	�A�D���A�**

	conv_loss�='o��        )��P	�r�D���A�**

	conv_loss/�]=g��        )��P	̢�D���A�**

	conv_loss�!P=aJ�        )��P	���D���A�**

	conv_lossR@�<�N        )��P	��D���A�**

	conv_loss�=M?        )��P	�6�D���A�**

	conv_loss��<={�|O        )��P	�g�D���A�**

	conv_lossS�2=�X��        )��P	7��D���A�**

	conv_loss~6,=\n&�        )��P	���D���A�**

	conv_loss�RU=a�m�        )��P	���D���A�**

	conv_loss/�C=��        )��P	'+�D���A�**

	conv_loss��=e�w�        )��P	�]�D���A�**

	conv_loss�f=P�xt        )��P	p��D���A�**

	conv_loss#�V=oi5�        )��P	k��D���A�**

	conv_loss޸!=|P��        )��P	���D���A�**

	conv_loss;#=i�^�        )��P	��D���A�**

	conv_lossn�=����        )��P	UN�D���A�**

	conv_loss �<j-]        )��P	J~�D���A�**

	conv_loss+$=���        )��P	��D���A�**

	conv_loss��0=P���        )��P	{��D���A�**

	conv_loss�l}=Ir-;        )��P	&�D���A�**

	conv_lossY+J=���3        )��P	P\�D���A�**

	conv_loss� =;l��        )��P	��D���A�**

	conv_loss��I=Q��i        )��P	���D���A�**

	conv_loss,3�<Y�ۨ        )��P	���D���A�**

	conv_lossEJ=�F�        )��P	�(�D���A�**

	conv_losss�'=-���        )��P	+b�D���A�**

	conv_lossu�<A���        )��P	���D���A�**

	conv_loss@�=â�2        )��P	���D���A�**

	conv_lossЫd=��]F        )��P	b�D���A�**

	conv_loss  =�1E        )��P	�Q�D���A�**

	conv_lossD�0=!���        )��P	��D���A�**

	conv_loss�=ʞm{        )��P	��D���A�**

	conv_loss��=��	�        )��P	=��D���A�**

	conv_loss=&�<��U        )��P	*�D���A�**

	conv_loss���<���        )��P	`O�D���A�**

	conv_loss8R=�i2        )��P	��D���A�**

	conv_lossD=��͋        )��P	I��D���A�**

	conv_loss���<�)�        )��P	���D���A�**

	conv_loss�P�<����        )��P	��D���A�**

	conv_losst�!=zy�\        )��P	3M�D���A�**

	conv_loss��d=9�/l        )��P	�~�D���A�**

	conv_loss�2=HN�N        )��P	h��D���A�**

	conv_lossi�<beK}        )��P	���D���A�**

	conv_lossW�=�|��        )��P	=�D���A�**

	conv_loss`\P=�N��        )��P	aI�D���A�**

	conv_loss�r=F��m        )��P	-z�D���A�**

	conv_loss��=��p�        )��P	���D���A�**

	conv_loss%�=��.        )��P	o��D���A�**

	conv_loss�¥<���        )��P	4#�D���A�**

	conv_loss��=X�6        )��P	�R�D���A�**

	conv_loss��=���        )��P	ք�D���A�**

	conv_lossln�<E�T�        )��P	��D���A�**

	conv_loss�s=���        )��P	���D���A�**

	conv_loss*�!=_���        )��P	e�D���A�**

	conv_loss��=���h        )��P	H�D���A�**

	conv_loss��=b >�        )��P	hw�D���A�**

	conv_loss`�#=r��        )��P		��D���A�**

	conv_lossXl2=x޲        )��P	J��D���A�**

	conv_loss�b=n�R�        )��P	.	�D���A�**

	conv_loss��=�,        )��P	�:�D���A�**

	conv_loss�m9=�*)        )��P	�i�D���A�**

	conv_loss�c=�t�=        )��P	K��D���A�**

	conv_loss'�}=~��\        )��P	p��D���A�**

	conv_loss?�R=��        )��P	���D���A�**

	conv_loss�� =x&��        )��P	/.�D���A�**

	conv_loss��Y=%7J'        )��P	a�D���A�**

	conv_loss��=-$g�        )��P	�=�D���A�**

	conv_loss@�=m���        )��P	��D���A�**

	conv_loss	.=۸�k        )��P	���D���A�**

	conv_loss�=�1~        )��P	���D���A�**

	conv_loss��=
�6�        )��P	��D���A�+*

	conv_loss#�<=���        )��P	�F�D���A�+*

	conv_lossf��<M���        )��P	A��D���A�+*

	conv_loss5y$=5[�v        )��P	��D���A�+*

	conv_lossDe=F�$�        )��P	���D���A�+*

	conv_loss}K�<B�]`        )��P	��D���A�+*

	conv_loss�x=�&P        )��P	�P�D���A�+*

	conv_loss��F=�I��        )��P	���D���A�+*

	conv_loss��U=�1f        )��P	ڱ�D���A�+*

	conv_lossX�C=O��        )��P	��D���A�+*

	conv_loss��m=���        )��P	,�D���A�+*

	conv_loss�+%=��        )��P	�A�D���A�+*

	conv_loss���<���        )��P	N��D���A�+*

	conv_lossH��<t	9        )��P	M��D���A�+*

	conv_loss�)=���        )��P	���D���A�+*

	conv_lossX� =�zw�        )��P	�#�D���A�+*

	conv_lossWI =z:        )��P	V�D���A�+*

	conv_loss̥<=|Ղ�        )��P	S��D���A�+*

	conv_lossme=����        )��P	���D���A�+*

	conv_loss��;=	��J        )��P	<��D���A�+*

	conv_lossbEo=VC�        )��P	��D���A�+*

	conv_loss�g8=`��        )��P	�A�D���A�+*

	conv_loss��<�^��        )��P	
p�D���A�+*

	conv_loss"E'=獉�        )��P	��D���A�+*

	conv_loss581=pe��        )��P	e��D���A�+*

	conv_loss���<z��        )��P	� �D���A�+*

	conv_loss�t=[�*        )��P	,2�D���A�+*

	conv_loss�]=��`j        )��P	b�D���A�+*

	conv_lossr!O=4��        )��P	e��D���A�+*

	conv_loss�=_Ta[        )��P	X��D���A�+*

	conv_loss�C=        )��P	��D���A�+*

	conv_loss73;=�!�        )��P	�$�D���A�+*

	conv_loss�.=�o        )��P	U�D���A�+*

	conv_loss:t=�v��        )��P	��D���A�+*

	conv_loss�_D=4��x        )��P	���D���A�+*

	conv_loss��K=�%��        )��P	K��D���A�+*

	conv_loss�]P=d7^�        )��P	k�D���A�+*

	conv_loss�=��1�        )��P	VE�D���A�+*

	conv_loss��8=��6        )��P	�w�D���A�+*

	conv_loss���=c#�a        )��P	���D���A�+*

	conv_lossi�=��        )��P	[��D���A�+*

	conv_lossw|�<���H        )��P	��D���A�+*

	conv_lossYs=�Z�^        )��P	L6�D���A�+*

	conv_lossϹK=���%        )��P	'e�D���A�+*

	conv_loss$�/=(s��        )��P	'��D���A�+*

	conv_lossl�=�Y�C        )��P	G��D���A�+*

	conv_loss���<\H        )��P	j��D���A�+*

	conv_loss�O=��Z:        )��P	�6�D���A�+*

	conv_loss��7=�<�q        )��P	�g�D���A�+*

	conv_loss�=!�f        )��P	��D���A�+*

	conv_loss���<zM�[        )��P	8��D���A�+*

	conv_loss1z�=#|�$        )��P	���D���A�+*

	conv_lossN8=w�i        )��P	'�D���A�+*

	conv_loss~';=*A�        )��P	�U�D���A�+*

	conv_loss��=JH��        )��P	τ�D���A�+*

	conv_lossT2q=؃�        )��P	���D���A�+*

	conv_loss���<Ɍ/        )��P	 �D���A�+*

	conv_loss&L=��M        )��P	-0�D���A�+*

	conv_loss��!=j�        )��P	a�D���A�+*

	conv_lossV`=�G �        )��P	~��D���A�+*

	conv_loss�=s�        )��P	���D���A�+*

	conv_loss��<K1��        )��P	���D���A�+*

	conv_loss
1=�s��        )��P	�'�D���A�+*

	conv_loss��&=�U        )��P	�W�D���A�+*

	conv_loss_X�<�䖖        )��P	���D���A�+*

	conv_loss�y�<         )��P	H��D���A�+*

	conv_loss9O=�2�        )��P	���D���A�+*

	conv_loss,�=R��        )��P	(�D���A�+*

	conv_lossL0=��        )��P	�W�D���A�+*

	conv_loss��=�C         )��P	݈�D���A�+*

	conv_loss��<����        )��P	���D���A�+*

	conv_lossH=���@        )��P	b��D���A�+*

	conv_loss�y=�Y0�        )��P	Z�D���A�+*

	conv_loss��+=���        )��P	5K�D���A�+*

	conv_loss9�F=���*        )��P	{�D���A�+*

	conv_loss���< ��(        )��P	���D���A�+*

	conv_lossIՃ=ݤ.l        )��P	d��D���A�+*

	conv_loss�=�Yu!        )��P	p
�D���A�+*

	conv_loss|#�=(O        )��P	�9�D���A�+*

	conv_loss5�'=����        )��P	i�D���A�+*

	conv_loss��=D���        )��P	4��D���A�+*

	conv_loss5F=����        )��P	3��D���A�+*

	conv_loss@=M���        )��P	U��D���A�+*

	conv_loss��=b���        )��P	�+�D���A�+*

	conv_lossVCR=���        )��P	�[�D���A�+*

	conv_lossL6=���&        )��P	��D���A�+*

	conv_loss3��<����        )��P	K��D���A�+*

	conv_loss��6=D;p        )��P	?��D���A�+*

	conv_lossva�<�{�        )��P	��D���A�+*

	conv_lossӼ>=��41        )��P	>Q�D���A�+*

	conv_loss��T=���        )��P	���D���A�+*

	conv_loss�X=�Rý        )��P	��D���A�+*

	conv_loss��"=��4        )��P	��D���A�+*

	conv_loss]�=�r�        )��P	��D���A�+*

	conv_loss�me=T�t�        )��P	9H�D���A�+*

	conv_loss��!=���        )��P	bv�D���A�+*

	conv_lossq�=� ��        )��P	���D���A�+*

	conv_lossh�(=%�        )��P	U\�D���A�+*

	conv_lossf�=�7��        )��P	���D���A�+*

	conv_loss�hF=�ܛ�        )��P	C��D���A�+*

	conv_loss�Q=��e�        )��P	d��D���A�+*

	conv_loss�=!�.'        )��P	�,�D���A�+*

	conv_loss{dr=M�	        )��P	A]�D���A�+*

	conv_loss���<=�!�        )��P	���D���A�+*

	conv_loss1�<3�p�        )��P	u��D���A�+*

	conv_lossy�=�$�        )��P	���D���A�+*

	conv_loss��r=���        )��P	�)�D���A�+*

	conv_lossM�:=���        )��P	@Z�D���A�+*

	conv_lossS=eR�z        )��P	���D���A�+*

	conv_loss@=|=#-��        )��P	���D���A�+*

	conv_loss�IG=ܮl        )��P	��D���A�+*

	conv_loss��8="�l        )��P	�3�D���A�+*

	conv_loss�I=����        )��P	+c�D���A�+*

	conv_loss
�0=v�        )��P	���D���A�+*

	conv_lossF{+=��>�        )��P	���D���A�+*

	conv_loss��R=��Ƹ        )��P	&��D���A�+*

	conv_loss7�5=�P��        )��P	].�D���A�+*

	conv_lossD0(=F� �        )��P	�^�D���A�+*

	conv_losse�-=�W[P        )��P	���D���A�+*

	conv_lossd�l=���        )��P	���D���A�+*

	conv_loss���<�"j        )��P	���D���A�+*

	conv_loss���<��>        )��P	�- E���A�+*

	conv_losse�d=�O|�        )��P	Cd E���A�+*

	conv_loss��=�q�        )��P	k� E���A�+*

	conv_loss�vH=і        )��P	B� E���A�+*

	conv_loss���<4͛        )��P	q� E���A�+*

	conv_loss�M =�@�        )��P	�8E���A�+*

	conv_loss��E=��        )��P	jE���A�+*

	conv_lossf�:=x���        )��P	)�E���A�,*

	conv_loss��	=�.Q"        )��P	��E���A�,*

	conv_loss�0=��)        )��P	0�E���A�,*

	conv_loss�]�<��        )��P	�4E���A�,*

	conv_loss���<�J��        )��P	xhE���A�,*

	conv_loss��X=����        )��P	��E���A�,*

	conv_loss�a9=X�        )��P	��E���A�,*

	conv_loss=�/=^Y�        )��P	�
E���A�,*

	conv_lossq"=`,+�        )��P	FE���A�,*

	conv_lossܥF=�T��        )��P	�zE���A�,*

	conv_loss:y=���        )��P	��E���A�,*

	conv_loss�O�<��j        )��P	��E���A�,*

	conv_loss���<�lWs        )��P	E���A�,*

	conv_loss��<}z��        )��P	�DE���A�,*

	conv_loss�N='/{        )��P	S}E���A�,*

	conv_loss�=���        )��P	j�E���A�,*

	conv_loss"�4=<�+        )��P	��E���A�,*

	conv_loss@ =U��        )��P	�E���A�,*

	conv_loss��,=�|.�        )��P	rNE���A�,*

	conv_lossp�=��@R        )��P	��E���A�,*

	conv_loss��
=��e�        )��P	��E���A�,*

	conv_loss7I=]��        )��P	g�E���A�,*

	conv_loss���<N a        )��P	/E���A�,*

	conv_loss�0�<r��v        )��P	�gE���A�,*

	conv_lossaL=�p�        )��P	םE���A�,*

	conv_loss��	=�9�G        )��P	��E���A�,*

	conv_lossv^=O�E7        )��P	 E���A�,*

	conv_loss+I/=d4<�        )��P	S8E���A�,*

	conv_loss��<�hk        )��P	�pE���A�,*

	conv_lossM$=(�wB        )��P	l�E���A�,*

	conv_loss��A=]��        )��P	��E���A�,*

	conv_lossoIL=�i�        )��P	�E���A�,*

	conv_loss;.=7���        )��P	�TE���A�,*

	conv_loss8�$=M9�&        )��P	e�E���A�,*

	conv_loss-�=�.!�        )��P	��E���A�,*

	conv_loss[q@=�I�n        )��P	��E���A�,*

	conv_lossq7-=��[        )��P	�,	E���A�,*

	conv_lossS/=�%�:        )��P	*g	E���A�,*

	conv_lossu�L=����        )��P	��	E���A�,*

	conv_loss��
=s�!�        )��P	�	E���A�,*

	conv_loss\��<O�J        )��P	
E���A�,*

	conv_loss%�+=&�N#        )��P	�S
E���A�,*

	conv_loss
�<G#_�        )��P	�
E���A�,*

	conv_loss@% =[]4        )��P	��
E���A�,*

	conv_lossֲ=4
\�        )��P	��
E���A�,*

	conv_loss��<�<Y�        )��P	b-E���A�,*

	conv_loss��%=�5�        )��P	3^E���A�,*

	conv_loss�7=��k;        )��P	��E���A�,*

	conv_lossC�=��        )��P	��E���A�,*

	conv_loss�]N=�C��        )��P	��E���A�,*

	conv_loss�= �         )��P	�)E���A�,*

	conv_loss ��=���        )��P	�[E���A�,*

	conv_loss�=�fm{        )��P	׏E���A�,*

	conv_loss�==�        )��P	Z�E���A�,*

	conv_loss��=S_S�        )��P	��E���A�,*

	conv_loss�:�<��)�        )��P	�5E���A�,*

	conv_loss��>=+���        )��P	�kE���A�,*

	conv_loss%�8=�w`        )��P	��E���A�,*

	conv_loss��<6U��        )��P	��E���A�,*

	conv_lossZ�=<�#        )��P	DE���A�,*

	conv_loss[�=��,�        )��P	�9E���A�,*

	conv_lossw=��7        )��P	jE���A�,*

	conv_loss�E=�/,        )��P	��E���A�,*

	conv_lossS�=�Pw        )��P	�E���A�,*

	conv_lossQ�>=�DW	        )��P	�E���A�,*

	conv_loss�wZ=�5P        )��P	=E���A�,*

	conv_loss�*=l��        )��P	�{E���A�,*

	conv_loss�:=��~t        )��P	ζE���A�,*

	conv_loss]e2=�cS        )��P	��E���A�,*

	conv_loss��<&`w�        )��P	�4E���A�,*

	conv_loss�w=y�-�        )��P	>kE���A�,*

	conv_loss~��< F�I        )��P	2�E���A�,*

	conv_loss��= *F        )��P	��E���A�,*

	conv_loss��c=�]
        )��P	o
E���A�,*

	conv_loss�=�V�        )��P	�<E���A�,*

	conv_loss�=O,��        )��P	/kE���A�,*

	conv_lossq�\=l��O        )��P	��E���A�,*

	conv_lossD=p�~�        )��P	R�E���A�,*

	conv_loss+M=���        )��P	iE���A�,*

	conv_loss���=s1�C        )��P	yHE���A�,*

	conv_loss~��=���        )��P	UyE���A�,*

	conv_losse�#=Zi@�        )��P	$�E���A�,*

	conv_loss =A���        )��P	W�E���A�,*

	conv_loss	=���        )��P	�E���A�,*

	conv_loss�&=8rm�        )��P	�BE���A�,*

	conv_loss��=��4        )��P	=tE���A�,*

	conv_losss�2=�׉        )��P	��E���A�,*

	conv_loss�v�<�2@        )��P	:�E���A�,*

	conv_loss��O=>�X�        )��P	�E���A�,*

	conv_loss��,=tT�~        )��P	ME���A�,*

	conv_lossX=;=�o        )��P	��E���A�,*

	conv_loss�J	=�m�         )��P	�E���A�,*

	conv_lossAT1=�xX0        )��P	t�E���A�,*

	conv_loss�i=�%,�        )��P	JE���A�,*

	conv_lossѬe=m�jl        )��P	�GE���A�,*

	conv_lossMj=%�&V        )��P	[vE���A�,*

	conv_lossc2�<��l4        )��P	ĨE���A�,*

	conv_loss���<��        )��P	O�E���A�,*

	conv_lossZC=v4�        )��P	�	E���A�,*

	conv_lossw�;=�'�        )��P	�8E���A�,*

	conv_loss��:=U�%�        )��P	gE���A�,*

	conv_loss��=�)�J        )��P	ΕE���A�,*

	conv_lossnzX=��2�        )��P	a�E���A�,*

	conv_loss�38=j�\�        )��P	X�E���A�,*

	conv_loss��	=�p         )��P	�+E���A�,*

	conv_loss�!=���0        )��P	�ZE���A�,*

	conv_loss\y=d x�        )��P	\�E���A�,*

	conv_loss�6=�1W�        )��P	��E���A�,*

	conv_lossH�<��0        )��P	��E���A�,*

	conv_loss�j="g�?        )��P	�E���A�,*

	conv_loss�`=O��        )��P	�JE���A�,*

	conv_loss �N=         )��P	x{E���A�,*

	conv_loss�A�<��b        )��P	��E���A�,*

	conv_loss�_=n��        )��P	��E���A�,*

	conv_loss���<�a        )��P	~E���A�,*

	conv_loss�=�:�        )��P	�6E���A�,*

	conv_loss�c�<w؂�        )��P	(dE���A�,*

	conv_loss��'=����        )��P	�E���A�,*

	conv_loss�]=7/��        )��P	��E���A�,*

	conv_loss*M=N��        )��P	�
E���A�,*

	conv_lossJ$N=Ɇ�0        )��P	�CE���A�,*

	conv_loss�s =R`|        )��P	nsE���A�,*

	conv_loss_ =F�6        )��P	��E���A�,*

	conv_loss@M=h��        )��P	$�E���A�,*

	conv_loss;�=�{�        )��P	�E���A�,*

	conv_loss��<��|�        )��P	�EE���A�,*

	conv_loss��=��mC        )��P	ewE���A�,*

	conv_loss�5@=�o:        )��P	ƱE���A�,*

	conv_loss�E=K!	X        )��P	L�E���A�-*

	conv_lossn�H=�6        )��P	0E���A�-*

	conv_lossaB=A�@�        )��P	�cE���A�-*

	conv_losse�=S�|�        )��P	=�E���A�-*

	conv_loss�=X=���        )��P	��E���A�-*

	conv_loss�q%=���        )��P	��E���A�-*

	conv_lossG�=e��        )��P	1E���A�-*

	conv_loss�=7��        )��P	�iE���A�-*

	conv_loss;5=��        )��P	#�E���A�-*

	conv_loss�r1=Җ'S        )��P	��E���A�-*

	conv_lossD8 =��e        )��P	E���A�-*

	conv_lossF�=)I�\        )��P	�4E���A�-*

	conv_loss�!=�,B�        )��P	)lE���A�-*

	conv_lossO@=���        )��P	��E���A�-*

	conv_loss�c=y04        )��P	��E���A�-*

	conv_losso@=�L��        )��P		E���A�-*

	conv_lossW�=f��        )��P	X9E���A�-*

	conv_loss��9=f�p�        )��P	�wE���A�-*

	conv_loss���=U�S        )��P	2�E���A�-*

	conv_loss��=��q        )��P	��E���A�-*

	conv_loss�@=�;*�        )��P	[ E���A�-*

	conv_loss��:=rn�        )��P	"B E���A�-*

	conv_loss�=R=���        )��P	ss E���A�-*

	conv_lossn]�<��1�        )��P	r� E���A�-*

	conv_loss� $=��        )��P	�� E���A�-*

	conv_loss\M�=T!��        )��P	�!E���A�-*

	conv_loss�5_=�C<        )��P	�<!E���A�-*

	conv_loss��=�%        )��P	Up!E���A�-*

	conv_lossV��<?�U�        )��P	��!E���A�-*

	conv_lossV
=n��        )��P	��!E���A�-*

	conv_loss8�
=��~�        )��P	)"E���A�-*

	conv_loss;�=�(U�        )��P	�>"E���A�-*

	conv_loss�-=Z�        )��P	$s"E���A�-*

	conv_loss��=��{�        )��P	۬"E���A�-*

	conv_lossҶ�<���        )��P	��"E���A�-*

	conv_loss��<	�n        )��P	&#E���A�-*

	conv_loss��=�|y        )��P	�K#E���A�-*

	conv_loss5%!=%0��        )��P	A�#E���A�-*

	conv_loss{=
�t        )��P	�#E���A�-*

	conv_loss���<>�σ        )��P	g$E���A�-*

	conv_loss�6A=�(�T        )��P	�2$E���A�-*

	conv_loss��!=����        )��P	��%E���A�-*

	conv_lossJ	=��s�        )��P	�&E���A�-*

	conv_lossƣ=.SG�        )��P	'D&E���A�-*

	conv_loss�@4=bl�;        )��P	Qx&E���A�-*

	conv_lossp��<E:��        )��P	k�&E���A�-*

	conv_loss�#=��F        )��P	��&E���A�-*

	conv_loss��<Z��1        )��P	~'E���A�-*

	conv_lossb+=�7Ra        )��P	�C'E���A�-*

	conv_lossA�<g��/        )��P	�w'E���A�-*

	conv_lossi�<=�f1        )��P	 �'E���A�-*

	conv_lossU��=);��        )��P	|�'E���A�-*

	conv_loss���<#�o[        )��P	�'(E���A�-*

	conv_loss>�W=)�7�        )��P	GW(E���A�-*

	conv_loss��@=�@�        )��P	`�(E���A�-*

	conv_loss�op=b��        )��P		�(E���A�-*

	conv_loss˃4=�c�:        )��P	1�(E���A�-*

	conv_loss\S*=�(#S        )��P	�0)E���A�-*

	conv_loss���<��g        )��P	�_)E���A�-*

	conv_lossX	=܉k        )��P	k�)E���A�-*

	conv_loss�T(=kw�        )��P	��)E���A�-*

	conv_lossrR�<D=�        )��P	�)E���A�-*

	conv_lossH3!=>��2        )��P	�&*E���A�-*

	conv_loss��f=���[        )��P	!V*E���A�-*

	conv_losssO=�l�        )��P	\�*E���A�-*

	conv_loss1=�a�N        )��P	��*E���A�-*

	conv_loss�=�U9        )��P	��*E���A�-*

	conv_loss^�=�ă        )��P	X:+E���A�-*

	conv_lossNq=���        )��P	Pk+E���A�-*

	conv_loss?[=�G)o        )��P	6�+E���A�-*

	conv_loss�Y�<�ȯ�        )��P	*�+E���A�-*

	conv_lossqo6=�V��        )��P	p,E���A�-*

	conv_loss�ڈ=K��        )��P	4,E���A�-*

	conv_loss�"2=�#�|        )��P		c,E���A�-*

	conv_loss/p=�b�        )��P	x�,E���A�-*

	conv_lossJ�7=��        )��P	�,E���A�-*

	conv_loss�v	=|u�6        )��P	�,E���A�-*

	conv_loss� =}�        )��P	;*-E���A�-*

	conv_loss?�-=����        )��P	YY-E���A�-*

	conv_loss|��<!�,�        )��P	.�-E���A�-*

	conv_loss��=�D�        )��P	g�-E���A�-*

	conv_loss��<��        )��P	��-E���A�-*

	conv_losse��<�S        )��P	o..E���A�-*

	conv_lossrb=�ܯ        )��P	T_.E���A�-*

	conv_loss�=��M        )��P	ޏ.E���A�-*

	conv_loss*e�<�3X	        )��P	f�.E���A�-*

	conv_loss\)=1~�        )��P	�.E���A�-*

	conv_loss4Jw=���        )��P	 -/E���A�-*

	conv_loss��0=�zF�        )��P	�]/E���A�-*

	conv_lossU�<���        )��P	[�/E���A�-*

	conv_loss�=���h        )��P	Ľ/E���A�-*

	conv_loss1�=,��%        )��P	�0E���A�-*

	conv_loss@�5=�A"y        )��P	n@0E���A�-*

	conv_lossӊ=c,��        )��P	�q0E���A�-*

	conv_loss��=�R�        )��P	c�0E���A�-*

	conv_loss��=xJ        )��P	��0E���A�-*

	conv_lossd�2=0��r        )��P	�1E���A�-*

	conv_loss�"=?��        )��P	�61E���A�-*

	conv_loss�d�<�1ԥ        )��P	jg1E���A�-*

	conv_lossб!=��&u        )��P	B�1E���A�-*

	conv_loss���<�d]�        )��P	��1E���A�-*

	conv_loss��<]H�U        )��P	�2E���A�-*

	conv_losswF6=(M	r        )��P	�>2E���A�-*

	conv_losse�<��x        )��P	m2E���A�-*

	conv_loss�\c=����        )��P	6�2E���A�-*

	conv_lossЪ=�3&�        )��P	��2E���A�-*

	conv_losswR�<��        )��P	�
3E���A�-*

	conv_loss@�=+`|        )��P	�<3E���A�-*

	conv_loss��<�-        )��P	�k3E���A�-*

	conv_loss� �<Q�        )��P	^�3E���A�-*

	conv_loss�Z=6��P        )��P	e�3E���A�-*

	conv_lossO�=swB        )��P	�4E���A�-*

	conv_lossI�)=,~��        )��P	f74E���A�-*

	conv_loss2�Y=�*~�        )��P	-g4E���A�-*

	conv_loss�JD=�� �        )��P	��4E���A�-*

	conv_lossm<�<3���        )��P	��4E���A�-*

	conv_loss0=$j�5        )��P	�4E���A�-*

	conv_loss�� =�m        )��P	�"5E���A�-*

	conv_loss��=ű��        )��P	�U5E���A�-*

	conv_loss΋�<	��|        )��P	2�5E���A�-*

	conv_loss���<��ng        )��P	��5E���A�-*

	conv_loss��M=�߈�        )��P	��5E���A�-*

	conv_loss�-=_^3        )��P	?6E���A�-*

	conv_loss�T=�a�        )��P	O6E���A�-*

	conv_loss�O=Bs]�        )��P	�6E���A�-*

	conv_lossy�#=Gj�         )��P	M�6E���A�-*

	conv_loss��G=�G�^        )��P	��6E���A�-*

	conv_loss)��<+O�        )��P	}7E���A�-*

	conv_loss��<��I�        )��P	�C7E���A�.*

	conv_loss-�P=�O�        )��P	Cr7E���A�.*

	conv_loss8&=�O~I        )��P	F�7E���A�.*

	conv_loss��T=� P        )��P	�7E���A�.*

	conv_loss1VQ=k���        )��P	%8E���A�.*

	conv_loss�B!=�;�}        )��P	�08E���A�.*

	conv_loss�%=u�]R        )��P	s_8E���A�.*

	conv_losst2�<��_�        )��P	[�8E���A�.*

	conv_loss�N�<���        )��P	_�8E���A�.*

	conv_loss���<g��        )��P	��8E���A�.*

	conv_lossj�=<��j        )��P	�9E���A�.*

	conv_lossQ�d=��GX        )��P	7P9E���A�.*

	conv_loss�p3=�kc        )��P	1�9E���A�.*

	conv_lossȱ-=,��        )��P	E�9E���A�.*

	conv_loss�j�</�        )��P	Q�9E���A�.*

	conv_loss��=�?�        )��P	�':E���A�.*

	conv_loss -=)���        )��P	=]:E���A�.*

	conv_loss=y=����        )��P	��:E���A�.*

	conv_lossϣ=����        )��P	��:E���A�.*

	conv_loss�\�<�m�        )��P	��:E���A�.*

	conv_loss	ͺ<;Q        )��P	Q;E���A�.*

	conv_loss���<Ɏi{        )��P	c\;E���A�.*

	conv_loss��!=Aw�        )��P	J�;E���A�.*

	conv_loss��=��\        )��P	��;E���A�.*

	conv_losso�<�Ͱ        )��P	��;E���A�.*

	conv_loss}9!=޸��        )��P	�'<E���A�.*

	conv_loss��=�p	        )��P	�]<E���A�.*

	conv_loss���<-��v        )��P	ޑ<E���A�.*

	conv_loss��G=yI��        )��P	�<E���A�.*

	conv_loss��=-�k        )��P	�=E���A�.*

	conv_loss7�<�f        )��P	5=E���A�.*

	conv_loss��%=��N>        )��P	�d=E���A�.*

	conv_loss���<�."        )��P	��=E���A�.*

	conv_loss���<H_�F        )��P	��=E���A�.*

	conv_loss�*=`m4        )��P	5�=E���A�.*

	conv_loss�=��.�        )��P	m*>E���A�.*

	conv_loss���<�$�;        )��P	7[>E���A�.*

	conv_loss;=Gs޽        )��P	�>E���A�.*

	conv_loss�,�<b:�]        )��P	��>E���A�.*

	conv_loss�\=l{H�        )��P	��>E���A�.*

	conv_loss�n&=�f        )��P	�!?E���A�.*

	conv_loss���<��j}        )��P	7S?E���A�.*

	conv_loss��;==?�        )��P	N�?E���A�.*

	conv_lossU_u=MD��        )��P	��?E���A�.*

	conv_loss��<$��        )��P	��?E���A�.*

	conv_lossX��<޽�        )��P	�@E���A�.*

	conv_losse�=
fw        )��P	9N@E���A�.*

	conv_loss6��<m�        )��P	^�@E���A�.*

	conv_loss��==[p}        )��P	��@E���A�.*

	conv_lossJ��<
�7        )��P	��@E���A�.*

	conv_loss^j#=��v        )��P	�AE���A�.*

	conv_loss�Z=�5�x        )��P	�GAE���A�.*

	conv_loss� /=w׋�        )��P	IzAE���A�.*

	conv_loss��<1/>B        )��P	S�AE���A�.*

	conv_loss��=��        )��P	��AE���A�.*

	conv_loss��D=i��/        )��P	�BE���A�.*

	conv_loss:z�<J��m        )��P	OBBE���A�.*

	conv_loss���<E�4        )��P	�sBE���A�.*

	conv_loss�s�<� x�        )��P	��BE���A�.*

	conv_loss7=f�k�        )��P	Z�BE���A�.*

	conv_loss�0$=Cꔓ        )��P	�CE���A�.*

	conv_loss���<���        )��P	<CE���A�.*

	conv_lossz�=0��X        )��P	)mCE���A�.*

	conv_loss���<�B4�        )��P	��CE���A�.*

	conv_loss�Ѐ=>G�6        )��P	��CE���A�.*

	conv_lossV6=p�c�        )��P	�DE���A�.*

	conv_loss�2;=�]��        )��P	�JDE���A�.*

	conv_lossx=Xu��        )��P	��DE���A�.*

	conv_loss<$=v�xJ        )��P	��DE���A�.*

	conv_loss��=��        )��P	H�DE���A�.*

	conv_lossh%=���        )��P	qEE���A�.*

	conv_loss�<=6i+�        )��P	mNEE���A�.*

	conv_loss��=1%��        )��P	؂EE���A�.*

	conv_loss�x�<~�r8        )��P	[�EE���A�.*

	conv_lossy��<�'
        )��P	��EE���A�.*

	conv_loss�C�<�o3{        )��P	�FE���A�.*

	conv_loss�==�s
        )��P	`GFE���A�.*

	conv_loss�as=�I=�        )��P	7vFE���A�.*

	conv_loss�mB=�EN�        )��P	߻FE���A�.*

	conv_loss�&k=�9ܨ        )��P	��FE���A�.*

	conv_lossv� =�Al        )��P	'GE���A�.*

	conv_loss��0=�Ͳ�        )��P	�YGE���A�.*

	conv_lossr�=٪q�        )��P	>�GE���A�.*

	conv_lossR�<H��|        )��P	ټGE���A�.*

	conv_lossie=�F�^        )��P	3�GE���A�.*

	conv_lossq6=T<T        )��P	�HE���A�.*

	conv_loss�Zu=X��%        )��P	�PHE���A�.*

	conv_loss�'=3�        )��P	��HE���A�.*

	conv_loss6�A=>9ʗ        )��P	��HE���A�.*

	conv_loss�P�<"�@        )��P	#�HE���A�.*

	conv_loss�!�< ɖ        )��P	vIE���A�.*

	conv_loss"Y=��        )��P	�GIE���A�.*

	conv_lossL�O=��w         )��P	�zIE���A�.*

	conv_lossH��<T�        )��P	��IE���A�.*

	conv_lossiT=���5        )��P	:�IE���A�.*

	conv_lossU�=[�֚        )��P	�JE���A�.*

	conv_loss��<�Z&+        )��P	�>JE���A�.*

	conv_loss,,=��        )��P	;qJE���A�.*

	conv_loss�XX=i��k        )��P	��JE���A�.*

	conv_lossH��<d�~�        )��P	��JE���A�.*

	conv_loss�X=�b5        )��P	.�JE���A�.*

	conv_loss�J=��5        )��P	�1KE���A�.*

	conv_loss�O=֫��        )��P	*cKE���A�.*

	conv_loss"�<��J        )��P	a�KE���A�.*

	conv_loss���<D<��        )��P	��KE���A�.*

	conv_loss��=6�@=        )��P	��KE���A�.*

	conv_lossQ0=�5�,        )��P	y(LE���A�.*

	conv_loss�_-=�Ȑ        )��P	�WLE���A�.*

	conv_loss��<��>�        )��P	��LE���A�.*

	conv_loss �f=��        )��P	r�LE���A�.*

	conv_loss}�=��S�        )��P	�LE���A�.*

	conv_lossdu�<�dS        )��P	2ME���A�.*

	conv_loss%=� 