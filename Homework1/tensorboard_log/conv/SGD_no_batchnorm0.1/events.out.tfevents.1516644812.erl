       �K"	   s���Abrain.Event:2N�W���      D(�	�-s���A"��
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
dtype0*
_output_shapes
: * 
_class
loc:@conv2d/kernel*
valueB
 *���>
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
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
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
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
T0* 
_class
loc:@conv2d/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
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
ReluReluconv2d/Conv2D*/
_output_shapes
:���������*
T0
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
valueB
 *�[q�
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
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container *
shape:
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
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
�
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_2/kernel
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
.conv2d_3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *��*
dtype0*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
valueB
 *�>*
dtype0
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
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_3/kernel*
	container *
shape:
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
conv2d_4/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
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
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
conv2d_4/kernel
VariableV2*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_4/kernel*
	container *
shape:*
dtype0
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
valueB"      *
dtype0*
_output_shapes
:
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
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@conv2d_5/kernel*%
valueB"            *
dtype0
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
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_5/kernel*
seed2 
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
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_5/kernel
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
Relu_5Reluconv2d_6/Conv2D*/
_output_shapes
:���������*
T0
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
8conv2d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_6/kernel/Initializer/random_uniform/shape*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_6/kernel*
seed2 *
dtype0
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
VariableV2*"
_class
loc:@conv2d_6/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
�
conv2d_6/kernel/AssignAssignconv2d_6/kernel*conv2d_6/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
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
dtype0*
_output_shapes
:*
valueB"����\  
j
ReshapeReshapeRelu_6Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������

�
-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"\  d   
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
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	�
d*
T0*
_class
loc:@dense/kernel
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
:	�
d*
T0
�
dense/kernel
VariableV2*
shape:	�
d*
dtype0*
_output_shapes
:	�
d*
shared_name *
_class
loc:@dense/kernel*
	container 
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	�
d*
use_locking(*
T0*
_class
loc:@dense/kernel
v
dense/kernel/readIdentitydense/kernel*
_output_shapes
:	�
d*
T0*
_class
loc:@dense/kernel
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
VariableV2*
dtype0*
_output_shapes

:d
*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape
:d

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
_class
loc:@dense_1/bias*
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name 
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
logistic_loss/zeros_like	ZerosLikedense_2/BiasAdd*'
_output_shapes
:���������
*
T0
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
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negdense_2/BiasAdd*
T0*'
_output_shapes
:���������

j
logistic_loss/mulMuldense_2/BiasAddPlaceholder_1*'
_output_shapes
:���������
*
T0
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
MeanMeanlogistic_lossConst*

Tidx0*
	keep_dims( *
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
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:���������
*

Tmultiples0*
T0
h
gradients/Mean_grad/Shape_1Shapelogistic_loss*
_output_shapes
:*
T0*
out_type0
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
: *

Tidx0*
	keep_dims( *
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
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
_output_shapes
:*
T0*
out_type0
�
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
�
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:���������

z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
out_type0*
_output_shapes
:*
T0
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
:*

Tidx0*
	keep_dims( 
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
�
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:���������
*
T0
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
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1
u
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
u
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
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
:*

Tidx0*
	keep_dims( *
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
:*

Tidx0*
	keep_dims( 
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
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1
�
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:���������

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
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
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
6gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_2/MatMul_grad/MatMul/^gradients/dense_2/MatMul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dense_2/MatMul_grad/MatMul*'
_output_shapes
:���������d*
T0
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
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:����������

�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�
d
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
%gradients/conv2d_7/Conv2D_grad/ShapeNShapeNRelu_5conv2d_6/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
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
3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_5'gradients/conv2d_7/Conv2D_grad/ShapeN:1gradients/Relu_6_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
%gradients/conv2d_5/Conv2D_grad/ShapeNShapeNRelu_3conv2d_4/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
�
2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_4/kernel/readgradients/Relu_4_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
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
7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_5/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*E
_class;
97loc:@gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput
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
gradients/Relu_2_grad/ReluGradReluGrad7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyRelu_2*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
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
gradients/Relu_1_grad/ReluGradReluGrad7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyRelu_1*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
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
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *���=
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
;GradientDescent/update_conv2d_6/kernel/ApplyGradientDescentApplyGradientDescentconv2d_6/kernelGradientDescent/learning_rate9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_6/kernel
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
:GradientDescent/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelGradientDescent/learning_rate8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

�
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@dense_1/bias
�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_3/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_4/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_5/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_6/kernel/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
ArgMaxArgMaxdense_2/BiasAddArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
�
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
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
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_1MeanCastConst_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: "�9dH�      ?�K	��-s���AJ��
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
shape:*
dtype0
*
_output_shapes
:
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
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0* 
_class
loc:@conv2d/kernel
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
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
conv2d/kernel/readIdentityconv2d/kernel*&
_output_shapes
:*
T0* 
_class
loc:@conv2d/kernel
e
conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
�
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
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
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
valueB
 *�[q>
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_1/kernel
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
VariableV2*"
_class
loc:@conv2d_1/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
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
conv2d_2/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
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
Relu_1Reluconv2d_2/Conv2D*
T0*/
_output_shapes
:���������
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
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_2/kernel
�
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
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
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_3/Conv2DConv2DRelu_1conv2d_2/kernel/read*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
.conv2d_3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *��*
dtype0*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
valueB
 *�>
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
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_3/kernel
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
VariableV2*
shared_name *"
_class
loc:@conv2d_3/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
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
.conv2d_4/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel*
valueB
 *HY>*
dtype0
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
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
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
conv2d_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
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
Relu_4Reluconv2d_5/Conv2D*/
_output_shapes
:���������*
T0
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

seed *
T0*"
_class
loc:@conv2d_5/kernel*
seed2 *
dtype0*&
_output_shapes
:
�
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_5/kernel
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
conv2d_5/kernel/AssignAssignconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(
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
conv2d_6/Conv2DConv2DRelu_4conv2d_5/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Y
Relu_5Reluconv2d_6/Conv2D*/
_output_shapes
:���������*
T0
�
0conv2d_6/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_6/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_6/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@conv2d_6/kernel*
valueB
 *���*
dtype0
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
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
�
conv2d_6/kernel
VariableV2*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_6/kernel*
	container *
shape:*
dtype0
�
conv2d_6/kernel/AssignAssignconv2d_6/kernel*conv2d_6/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
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
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC
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
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
seed2 *
dtype0*
_output_shapes
:	�
d*

seed 
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d
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
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d
�
dense/MatMulMatMulReshapedense/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
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
-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *�'o�
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
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes

:d
*

seed 
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
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes

:d
*
T0
�
dense_1/kernel
VariableV2*
shared_name *!
_class
loc:@dense_1/kernel*
	container *
shape
:d
*
dtype0*
_output_shapes

:d

�
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
_output_shapes

:d
*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes

:d
*
T0
�
dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
_class
loc:@dense_1/bias*
valueB
*    
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
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

�
dense_2/MatMulMatMulRelu_7dense_1/kernel/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*'
_output_shapes
:���������
*
T0*
data_formatNHWC
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
MeanMeanlogistic_lossConst*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
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
: *

Tidx0*
	keep_dims( *
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
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
gradients/Mean_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0*
_output_shapes
:
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
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
�
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:���������
*
T0
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
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
u
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
�
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$gradients/logistic_loss/mul_grad/mulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Placeholder_1*'
_output_shapes
:���������
*
T0
�
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
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
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

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
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1
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
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_77gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:d
*
transpose_a(
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
8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:d
*
T0
�
gradients/Relu_7_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_7*
T0*'
_output_shapes
:���������d
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_7_grad/ReluGraddense/kernel/read*(
_output_shapes
:����������
*
transpose_a( *
transpose_b(*
T0
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
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:����������

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
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:���������*
T0
�
gradients/Relu_6_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_6*/
_output_shapes
:���������*
T0
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
9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*F
_class<
:8loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
�
gradients/Relu_5_grad/ReluGradReluGrad7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyRelu_5*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_6/Conv2D_grad/ShapeNShapeNRelu_4conv2d_5/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0
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
3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4'gradients/conv2d_6/Conv2D_grad/ShapeN:1gradients/Relu_5_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
gradients/Relu_4_grad/ReluGradReluGrad7gradients/conv2d_6/Conv2D_grad/tuple/control_dependencyRelu_4*
T0*/
_output_shapes
:���������
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
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNRelu_2conv2d_3/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0
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
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2'gradients/conv2d_4/Conv2D_grad/ShapeN:1gradients/Relu_3_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
/gradients/conv2d_3/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*E
_class;
97loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
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
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
N* 
_output_shapes
::*
T0*
out_type0
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/Relu_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
 *���=*
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
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel
�
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:*
use_locking( *
T0
�
;GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentApplyGradientDescentconv2d_3/kernelGradientDescent/learning_rate9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_3/kernel
�
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
use_locking( *
T0
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
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�
d*
use_locking( *
T0*
_class
loc:@dense/kernel
�
:GradientDescent/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelGradientDescent/learning_rate8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

�
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@dense_1/bias
�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_3/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_4/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_5/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_6/kernel/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
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
: *

Tidx0*
	keep_dims( *
T0
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: ""
	summaries

conv_loss:0"�
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
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0���       `/�#	��Vs���A*

	conv_loss��0?.�Э       QKD	9N^s���A*

	conv_lossD00?��*y       QKD	�^s���A*

	conv_lossx�/?i��       QKD	A�^s���A*

	conv_loss�Y/?<��~       QKD	��^s���A*

	conv_loss��.?aYp�       QKD	$-_s���A*

	conv_loss�l.?�8�       QKD	|b_s���A*

	conv_loss[.?U�æ       QKD	Y�_s���A*

	conv_lossrw-?���       QKD	��_s���A*

	conv_loss3-?�Z[`       QKD	A`s���A	*

	conv_loss&�,?��`�       QKD	=Q`s���A
*

	conv_loss�,?���9       QKD	�`s���A*

	conv_loss��+?05H�       QKD	��`s���A*

	conv_loss�+?^�	�       QKD	��`s���A*

	conv_loss3�*?���       QKD	}4as���A*

	conv_loss��)?�!X�       QKD	�jas���A*

	conv_loss
Y)?�.�Q       QKD	ޡas���A*

	conv_loss��(?��a       QKD	��as���A*

	conv_lossX
(?iq�4       QKD	�bs���A*

	conv_loss��'?Υ�-       QKD	�Ebs���A*

	conv_loss~�&?�E��       QKD	ʄbs���A*

	conv_loss��%?�'W�       QKD	�bs���A*

	conv_lossf�$?J���       QKD	�cs���A*

	conv_loss��#?�
9       QKD	\8cs���A*

	conv_lossc�"?���       QKD	�ncs���A*

	conv_lossT� ?W_�       QKD	��cs���A*

	conv_loss"�?���       QKD	��cs���A*

	conv_loss9�?�^�d       QKD	�ds���A*

	conv_loss$;?��|       QKD	�Jds���A*

	conv_loss-?I�       QKD	�~ds���A*

	conv_lossb�	?|�       QKD	2�ds���A*

	conv_loss��>��?       QKD	O�ds���A*

	conv_loss�Z�>2+��       QKD	es���A *

	conv_lossN�>���"       QKD	!Oes���A!*

	conv_loss?^�>�e��       QKD	Ӏes���A"*

	conv_loss���>5"�-       QKD	��es���A#*

	conv_loss���>��\       QKD	X�es���A$*

	conv_lossh�>��lf       QKD	�fs���A%*

	conv_lossI��>6x��       QKD	2Nfs���A&*

	conv_loss��>t�l       QKD	�fs���A'*

	conv_loss�<�>E       QKD	#�fs���A(*

	conv_lossl��>4��>       QKD	:�fs���A)*

	conv_loss)��>���       QKD	
"gs���A**

	conv_loss&�>ɈB       QKD	�Vgs���A+*

	conv_loss�t�>�tH5       QKD	��gs���A,*

	conv_loss�d�>�"�~       QKD	V�gs���A-*

	conv_loss���>���       QKD	��gs���A.*

	conv_loss֙�>ފR       QKD	A/hs���A/*

	conv_loss���>�E2       QKD	�ehs���A0*

	conv_lossKU�>kڒ�       QKD	v�hs���A1*

	conv_lossU�>:by�       QKD	��hs���A2*

	conv_lossNͤ>�ѣ"       QKD	�is���A3*

	conv_loss_��>�#�       QKD	�Ois���A4*

	conv_loss�b�>
.<�       QKD	d�is���A5*

	conv_lossB��>h�:|       QKD	��is���A6*

	conv_lossC�>�!=       QKD	��is���A7*

	conv_loss>Z<V�       QKD	�4js���A8*

	conv_loss� �>2�7       QKD	�xjs���A9*

	conv_loss�ء>��'       QKD	-�js���A:*

	conv_loss��>�V8�       QKD	��js���A;*

	conv_loss�`�>��-�       QKD	t*ks���A<*

	conv_loss��>)       QKD	�mks���A=*

	conv_loss�>7F�)       QKD	#�ks���A>*

	conv_loss�>�?K�       QKD	��ks���A?*

	conv_lossRb�>
(V/       QKD	mls���A@*

	conv_loss���>�(�       QKD	�Bls���AA*

	conv_loss^J�>�P:�       QKD	u{ls���AB*

	conv_losso<�>Nu�&       QKD	_�ls���AC*

	conv_losse�>��
�       QKD	 �ls���AD*

	conv_loss��>��4x       QKD	)ms���AE*

	conv_loss>��>���       QKD	q]ms���AF*

	conv_loss=T�>`/}        QKD	|�ms���AG*

	conv_loss�A�>W�Hm       QKD	&�ms���AH*

	conv_losss՚>FZ�U       QKD	m�ms���AI*

	conv_loss�e�>=A�q       QKD	�/ns���AJ*

	conv_lossҢ�>�á�       QKD	�cns���AK*

	conv_loss�;�>n�C1       QKD	/�ns���AL*

	conv_loss��>�H h       QKD	��ns���AM*

	conv_losss��>���       QKD	os���AN*

	conv_loss!��><��       QKD	�8os���AO*

	conv_lossU�>e�k       QKD	�los���AP*

	conv_lossh(�>�3�"       QKD	��os���AQ*

	conv_loss��>;���       QKD	�os���AR*

	conv_lossC�>	�|�       QKD	�	ps���AS*

	conv_lossML�>���       QKD	�;ps���AT*

	conv_loss ��>�X�E       QKD	�mps���AU*

	conv_loss��>Ԃ��       QKD	�ps���AV*

	conv_lossN��>�P�       QKD	��ps���AW*

	conv_loss���>gh�       QKD	.qs���AX*

	conv_loss�;�>��#       QKD		9qs���AY*

	conv_loss��}>���9       QKD	�kqs���AZ*

	conv_loss�Ry>��@       QKD	��qs���A[*

	conv_lossJn>��^       QKD	�qs���A\*

	conv_loss�x~>��Jx       QKD	irs���A]*

	conv_loss�k�>��       QKD	�5rs���A^*

	conv_loss"2�>�O��       QKD	bgrs���A_*

	conv_loss�@�>�nWB       QKD	ʘrs���A`*

	conv_lossh�s>E�       QKD	!�rs���Aa*

	conv_loss��j>H�˘       QKD	K�rs���Ab*

	conv_loss��o>q�1/       QKD	�2ss���Ac*

	conv_lossx�u>��g
       QKD	�fss���Ad*

	conv_loss�wf>H�@`       QKD	��ss���Ae*

	conv_loss�Q>��V       QKD	��ss���Af*

	conv_loss�,M>��m       QKD	Qts���Ag*

	conv_loss��a>�g2|       QKD	Fts���Ah*

	conv_loss�ކ>g��       QKD	Rxts���Ai*

	conv_loss�y�>�_��       QKD	K�ts���Aj*

	conv_loss��{>2x��       QKD	A�ts���Ak*

	conv_loss�f>k�d�       QKD	us���Al*

	conv_loss�eP>�<�j       QKD	�Vus���Am*

	conv_loss��J>�E��       QKD	܉us���An*

	conv_loss
J>��.4       QKD	��us���Ao*

	conv_loss.�I>��7E       QKD	��us���Ap*

	conv_losss}Y>o�4�       QKD	�0vs���Aq*

	conv_loss�?e>�Q�       QKD	�dvs���Ar*

	conv_lossד<>8�L�       QKD	W�vs���As*

	conv_loss"�=>#��       QKD	��vs���At*

	conv_loss�)>�m�       QKD	�ws���Au*

	conv_loss�@>�!��       QKD	d@ws���Av*

	conv_loss�_L>�9İ       QKD	Vsws���Aw*

	conv_loss��p>�       QKD	e�ws���Ax*

	conv_loss�v>\*	�       QKD	��ws���Ay*

	conv_loss�gF>#��#       QKD	�	xs���Az*

	conv_loss�0>.�T�       QKD	�<xs���A{*

	conv_loss��6>��       QKD	�oxs���A|*

	conv_lossj�*>C욘       QKD	̟xs���A}*

	conv_loss9�,>�>v{       QKD	�xs���A~*

	conv_loss��:>�y��       QKD	�ys���A*

	conv_loss2�;>�|�        )��P	�8ys���A�*

	conv_loss޹V>�ʒ�        )��P	Plys���A�*

	conv_loss�R>Xy�4        )��P	]�ys���A�*

	conv_loss1hO>#��"        )��P	��ys���A�*

	conv_loss��W>���        )��P	��ys���A�*

	conv_loss��J>�~�1        )��P		0zs���A�*

	conv_loss��>>�U        )��P	"`zs���A�*

	conv_loss�7>M��        )��P	��zs���A�*

	conv_loss:)>R��?        )��P	��zs���A�*

	conv_loss%�!>~��        )��P	��zs���A�*

	conv_loss�X=>���        )��P	K){s���A�*

	conv_loss�6>���        )��P	[{s���A�*

	conv_loss=L>�F        )��P	f�{s���A�*

	conv_lossE0)>���        )��P	[�{s���A�*

	conv_loss�: >v��'        )��P	�{s���A�*

	conv_losstA>�ԁ�        )��P	$%|s���A�*

	conv_loss��>��=        )��P	%U|s���A�*

	conv_loss ?>}f�        )��P	b�|s���A�*

	conv_lossс>��2        )��P	̹|s���A�*

	conv_loss!>��}        )��P	��|s���A�*

	conv_loss�X>/#�        )��P	K}s���A�*

	conv_loss��>���        )��P	wO}s���A�*

	conv_loss�P>d�n        )��P	*�}s���A�*

	conv_lossB�">U�8�        )��P	��}s���A�*

	conv_loss#�8>An��        )��P	�}s���A�*

	conv_loss�1 >�D!�        )��P	A.~s���A�*

	conv_loss��	>���t        )��P	b~s���A�*

	conv_loss��>�n�        )��P	��~s���A�*

	conv_loss��>P��        )��P	��~s���A�*

	conv_loss��>�q*        )��P	r�~s���A�*

	conv_lossk/>��3        )��P	�,s���A�*

	conv_loss��>u��J        )��P	�fs���A�*

	conv_loss��>��y�        )��P	�s���A�*

	conv_loss�7>Ns�        )��P	R�s���A�*

	conv_loss�*>ꉂ        )��P	�	�s���A�*

	conv_loss�(>>F�        )��P	S;�s���A�*

	conv_lossf�>�u�R        )��P	*m�s���A�*

	conv_loss�5�=�#�l        )��P	?��s���A�*

	conv_lossK��=NΤ�        )��P	؀s���A�*

	conv_loss�Y�=ܻ7�        )��P	n�s���A�*

	conv_loss��>4��        )��P	>�s���A�*

	conv_loss�7>߇�P        )��P	xn�s���A�*

	conv_loss���=b��        )��P	���s���A�*

	conv_loss��>�|H�        )��P	�Ӂs���A�*

	conv_lossK>�Up�        )��P	��s���A�*

	conv_loss�r�=��J�        )��P	B<�s���A�*

	conv_loss5�>�A        )��P	No�s���A�*

	conv_loss@1.>3P�        )��P	젂s���A�*

	conv_loss�?>��SB        )��P	�҂s���A�*

	conv_loss�">g{n�        )��P	5�s���A�*

	conv_losst?�=�p�q        )��P	l6�s���A�*

	conv_loss��=�c�Z        )��P	�k�s���A�*

	conv_loss���=�)��        )��P	]��s���A�*

	conv_loss;s�=���        )��P	�Ӄs���A�*

	conv_lossJ��=�C�        )��P	s�s���A�*

	conv_lossy�=^�        )��P	D>�s���A�*

	conv_loss��=(Z��        )��P	�o�s���A�*

	conv_loss^O�=h͇X        )��P	8��s���A�*

	conv_loss�^�=�B.X        )��P	�фs���A�*

	conv_loss��>Bj<�        )��P	Q�s���A�*

	conv_loss��>jMO�        )��P	�3�s���A�*

	conv_lossE7 >��VB        )��P	�d�s���A�*

	conv_loss��=��<y        )��P	���s���A�*

	conv_loss�>�\�K        )��P	�хs���A�*

	conv_loss,��=�l6�        )��P	�	�s���A�*

	conv_loss!��=z���        )��P	�:�s���A�*

	conv_loss��=�l��        )��P	el�s���A�*

	conv_lossy(�=t�        )��P	��s���A�*

	conv_loss ^�=M���        )��P	�͆s���A�*

	conv_lossn#>�pb        )��P	���s���A�*

	conv_lossX�@>!�        )��P	3�s���A�*

	conv_loss*7>4��h        )��P	/i�s���A�*

	conv_lossƫ>����        )��P	ޚ�s���A�*

	conv_loss�!�='���        )��P	�D�s���A�*

	conv_loss���=���Y        )��P	~y�s���A�*

	conv_loss�K�=�4        )��P	��s���A�*

	conv_loss8۱=��B(        )��P	.��s���A�*

	conv_loss�3�=g�q        )��P	�s���A�*

	conv_loss��=_-��        )��P	G�s���A�*

	conv_lossC��=^�_�        )��P	�{�s���A�*

	conv_loss�z�=�ȇ        )��P	N��s���A�*

	conv_loss}��=]V7        )��P	�s���A�*

	conv_loss�v�=�h{�        )��P	#�s���A�*

	conv_loss���=��A        )��P	�X�s���A�*

	conv_loss!��=�1A~        )��P	ω�s���A�*

	conv_loss|<�= J�        )��P	5��s���A�*

	conv_loss��=k'�y        )��P	��s���A�*

	conv_loss�Y�=f;sC        )��P	��s���A�*

	conv_loss\ϥ=����        )��P	�K�s���A�*

	conv_loss�1�=%HF�        )��P	d|�s���A�*

	conv_lossg>ҍJR        )��P	��s���A�*

	conv_lossb	>�"�        )��P	�ڌs���A�*

	conv_loss.��=b���        )��P	��s���A�*

	conv_loss���=x��        )��P	0=�s���A�*

	conv_loss�ޗ=GG�K        )��P	Ut�s���A�*

	conv_loss���=0�[�        )��P	���s���A�*

	conv_loss�0�=�lJ�        )��P	�܍s���A�*

	conv_loss��=Zr6�        )��P	��s���A�*

	conv_lossb+�=%��        )��P	�>�s���A�*

	conv_loss'�=�J        )��P	�m�s���A�*

	conv_loss>��=�T��        )��P	)��s���A�*

	conv_loss�8�=��        )��P	�Ɏs���A�*

	conv_loss)v�=��݆        )��P	���s���A�*

	conv_loss�x�=����        )��P	|.�s���A�*

	conv_loss�k�=��        )��P	�]�s���A�*

	conv_loss.��=J0H        )��P	`��s���A�*

	conv_loss�B�=��]�        )��P	��s���A�*

	conv_loss}��=Y��        )��P	��s���A�*

	conv_lossV��=)7��        )��P	��s���A�*

	conv_loss�]�=k5[r        )��P	�K�s���A�*

	conv_loss�ɠ=�N��        )��P	?{�s���A�*

	conv_loss6֛=m>?�        )��P	���s���A�*

	conv_lossW-�=�S\�        )��P	�ېs���A�*

	conv_loss\�>��|�        )��P	]
�s���A�*

	conv_loss���=�O�        )��P	�;�s���A�*

	conv_loss.�=EM��        )��P	�j�s���A�*

	conv_loss��=��        )��P	r��s���A�*

	conv_loss���=E�k�        )��P	�ˑs���A�*

	conv_lossa�=��؄        )��P	-��s���A�*

	conv_loss���=D�Z�        )��P	N/�s���A�*

	conv_loss�:�=�3�A        )��P	�^�s���A�*

	conv_loss�|�=�ܩ        )��P	.��s���A�*

	conv_loss�d�=��#�        )��P	�Ēs���A�*

	conv_lossM�=S�        )��P	(�s���A�*

	conv_loss�o�=�w�        )��P	8�s���A�*

	conv_loss鑤=}�n�        )��P	�g�s���A�*

	conv_lossݯ�=�ZI�        )��P	���s���A�*

	conv_loss���=͵!�        )��P	�Γs���A�*

	conv_loss���=���        )��P	���s���A�*

	conv_loss���=����        )��P	�1�s���A�*

	conv_loss���=\�        )��P	Dc�s���A�*

	conv_loss��=�]         )��P	o��s���A�*

	conv_lossv7�=�rF�        )��P	=ܔs���A�*

	conv_loss<��=Nq��        )��P	��s���A�*

	conv_loss��=X{Ļ        )��P	3B�s���A�*

	conv_loss��=�(�        )��P	fu�s���A�*

	conv_loss��=&���        )��P	r��s���A�*

	conv_loss�=ZԠ�        )��P	"��s���A�*

	conv_lossR�=b�mN        )��P	�s���A�*

	conv_loss|��=l        )��P	t@�s���A�*

	conv_loss��=y���        )��P	-p�s���A�*

	conv_loss�9Y=���        )��P	e��s���A�*

	conv_lossCߴ=" p        )��P	�Жs���A�*

	conv_losse
�=��        )��P	� �s���A�*

	conv_loss?ʬ=ۈ        )��P	�/�s���A�*

	conv_loss
*�=>�^i        )��P	�g�s���A�*

	conv_loss2��=~�        )��P	�s���A�*

	conv_lossp��=����        )��P	eϗs���A�*

	conv_loss8��=�Y��        )��P	���s���A�*

	conv_loss�μ=��#X        )��P	/�s���A�*

	conv_losso�=��v�        )��P	`�s���A�*

	conv_loss�Ś=�K��        )��P	���s���A�*

	conv_loss��=���A        )��P	e��s���A�*

	conv_loss-��=�i        )��P	0�s���A�*

	conv_loss� �=7�l�        )��P	7�s���A�*

	conv_lossyP�=�t��        )��P	O�s���A�*

	conv_loss�љ=�Ż$        )��P	y~�s���A�*

	conv_losss��=r�        )��P	���s���A�*

	conv_lossuB�=��        )��P	Gߙs���A�*

	conv_loss���=�[p        )��P	�s���A�*

	conv_lossM��=Oy��        )��P	�@�s���A�*

	conv_lossؤ�=/��}        )��P	(q�s���A�*

	conv_loss���=��I�        )��P	١�s���A�*

	conv_loss�"}=^Y-V        )��P	�њs���A�*

	conv_loss�ט=��O        )��P	��s���A�*

	conv_loss��==
�        )��P	r;�s���A�*

	conv_loss�ۥ=e53�        )��P	Jk�s���A�*

	conv_loss��=��%\        )��P	M��s���A�*

	conv_loss,�Q=�        )��P	�՛s���A�*

	conv_lossб=�Ŗ�        )��P	��s���A�*

	conv_loss��=�zޭ        )��P	2:�s���A�*

	conv_lossŐ�=R���        )��P	kt�s���A�*

	conv_loss.��=J�        )��P	d��s���A�*

	conv_loss�R{=��        )��P	;�s���A�*

	conv_loss���=��F1        )��P	��s���A�*

	conv_loss�Ԣ=n!/*        )��P	]M�s���A�*

	conv_loss'9^=�5�_        )��P	E}�s���A�*

	conv_loss���=�Щ�        )��P	��s���A�*

	conv_loss��=^��        )��P	��s���A�*

	conv_loss{�\=��Ѡ        )��P	��s���A�*

	conv_loss>w�=i���        )��P	�G�s���A�*

	conv_lossw��=bk�E        )��P	�x�s���A�*

	conv_lossem�=R���        )��P	i��s���A�*

	conv_loss�Б=\�Ƥ        )��P	��s���A�*

	conv_lossY{�=��e�        )��P	��s���A�*

	conv_loss���=�b+�        )��P	lJ�s���A�*

	conv_loss�v�=f���        )��P	�z�s���A�*

	conv_loss�Ƣ=.7�B        )��P	h��s���A�*

	conv_loss�8`=_��V        )��P	�ٟs���A�*

	conv_lossǿ=���W        )��P	��s���A�*

	conv_loss�8{=!B�        )��P	�E�s���A�*

	conv_lossW:�=�&�e        )��P	4w�s���A�*

	conv_loss�=.	B1        )��P	:��s���A�*

	conv_loss�;�=1A�$        )��P	�ܠs���A�*

	conv_loss]7R=�yl�        )��P	��s���A�*

	conv_lossE�=�#�*        )��P	>�s���A�*

	conv_loss��=\�y,        )��P	�w�s���A�*

	conv_lossLֆ=�S4,        )��P	V��s���A�*

	conv_loss�U�=2ڑd        )��P	�ߡs���A�*

	conv_loss�mt=�K{e        )��P	S�s���A�*

	conv_lossu�=���        )��P	_@�s���A�*

	conv_lossP-A=$xO�        )��P	�p�s���A�*

	conv_loss�=�Jw        )��P	_��s���A�*

	conv_loss�ED=��i�        )��P	�Ԣs���A�*

	conv_lossrH.=�73        )��P	��s���A�*

	conv_lossM$l=���        )��P	{8�s���A�*

	conv_loss^ܯ=�AV�        )��P	�k�s���A�*

	conv_loss��g=�0��        )��P	L��s���A�*

	conv_loss���=��/�        )��P	.̣s���A�*

	conv_lossU�=�o�f        )��P	��s���A�*

	conv_loss�O=Asc        )��P	0�s���A�*

	conv_loss@�m=�H�4        )��P	z`�s���A�*

	conv_lossWɖ= �ʁ        )��P	Y��s���A�*

	conv_loss�f<=��u        )��P	���s���A�*

	conv_loss��=����        )��P	��s���A�*

	conv_loss`
l=�|��        )��P	� �s���A�*

	conv_loss(��=[�        )��P	�P�s���A�*

	conv_lossA~`=�%�        )��P	P��s���A�*

	conv_loss��x=3��!        )��P	���s���A�*

	conv_loss�0=�9��        )��P	��s���A�*

	conv_loss��u=�        )��P	��s���A�*

	conv_loss�C�=㢬�        )��P	�F�s���A�*

	conv_loss�=�o�        )��P	w�s���A�*

	conv_losss;s=��        )��P	���s���A�*

	conv_loss*e�=q��        )��P	'�s���A�*

	conv_lossZ�=n|        )��P	��s���A�*

	conv_lossS�=�!f        )��P	�F�s���A�*

	conv_loss�=�JC        )��P	 {�s���A�*

	conv_lossQ��=�b$        )��P	�s���A�*

	conv_loss��\=v        )��P	9ߧs���A�*

	conv_loss�Ճ=s�=_        )��P	��s���A�*

	conv_loss�{a=��a�        )��P	�S�s���A�*

	conv_loss�.�=��ߛ        )��P	X��s���A�*

	conv_loss��={���        )��P	���s���A�*

	conv_loss���=�!�        )��P	i��s���A�*

	conv_loss�A�=�<[�        )��P	5(�s���A�*

	conv_lossm��=��*        )��P	�o�s���A�*

	conv_loss�"v= r�B        )��P	���s���A�*

	conv_loss��=Z�a�        )��P	-ةs���A�*

	conv_loss�i�=����        )��P	^�s���A�*

	conv_loss*=���        )��P	�9�s���A�*

	conv_loss&5n=
_:�        )��P	�k�s���A�*

	conv_loss�[=vs��        )��P	ʛ�s���A�*

	conv_loss@��=��dO        )��P	H̪s���A�*

	conv_lossʼ�=t��        )��P	���s���A�*

	conv_loss�!�=��        )��P	�0�s���A�*

	conv_lossr�i=����        )��P	(a�s���A�*

	conv_loss��o=ΩsE        )��P	Z��s���A�*

	conv_lossۖ:='�        )��P	(ūs���A�*

	conv_lossbM?=�A�        )��P	���s���A�*

	conv_loss���=�5��        )��P	['�s���A�*

	conv_loss'ho=7-        )��P	BW�s���A�*

	conv_loss`ĉ=�<��        )��P	���s���A�*

	conv_loss���=g��F        )��P	���s���A�*

	conv_loss/��=�vc        )��P	�s���A�*

	conv_lossF,B=(���        )��P	q�s���A�*

	conv_loss9�f=���r        )��P	Q�s���A�*

	conv_loss�4�=��>�        )��P	�s���A�*

	conv_loss.�P=^?-4        )��P	��s���A�*

	conv_loss\io=^俗        )��P	6�s���A�*

	conv_loss�o=ē�L        )��P	��s���A�*

	conv_lossT*[=F
��        )��P	�A�s���A�*

	conv_loss��=q�i        )��P	�r�s���A�*

	conv_lossNΜ=%+o        )��P	���s���A�*

	conv_loss�=}�q        )��P	�Ӯs���A�*

	conv_loss��=F)�        )��P	��s���A�*

	conv_lossoh=-P�	        )��P	66�s���A�*

	conv_lossጆ=���        )��P	�f�s���A�*

	conv_loss���=�6%        )��P	R��s���A�*

	conv_loss��g=#l��        )��P	�˯s���A�*

	conv_loss�^=����        )��P	���s���A�*

	conv_lossJHc=����        )��P	`.�s���A�*

	conv_lossA�=f˹B        )��P	�_�s���A�*

	conv_loss�s�=֢H        )��P	P��s���A�*

	conv_loss�l�=���        )��P	+&�s���A�*

	conv_loss��[=zt��        )��P	�V�s���A�*

	conv_loss ?k=�1�        )��P	d��s���A�*

	conv_loss1�5=$�B�        )��P	��s���A�*

	conv_loss���=q�H�        )��P	%�s���A�*

	conv_loss�}�=b���        )��P	�(�s���A�*

	conv_loss?=�^s�        )��P	�_�s���A�*

	conv_loss�%�=����        )��P	Ҝ�s���A�*

	conv_lossL2q=e�r        )��P	�ͳs���A�*

	conv_loss��j=��$�        )��P	���s���A�*

	conv_loss�A�=$@*        )��P	�/�s���A�*

	conv_loss���=�A�}        )��P	�b�s���A�*

	conv_lossA��=Q���        )��P	���s���A�*

	conv_lossB�]=���        )��P	δs���A�*

	conv_lossb]\=wI��        )��P	���s���A�*

	conv_losse��=�M��        )��P	%2�s���A�*

	conv_lossV�=�t        )��P	�c�s���A�*

	conv_lossC`v=_&�        )��P	H��s���A�*

	conv_lossu�.=�m        )��P	Nҵs���A�*

	conv_loss+��=�~��        )��P	��s���A�*

	conv_loss�N9=pܙ�        )��P	,:�s���A�*

	conv_loss�D=�5�        )��P	7k�s���A�*

	conv_loss��4=E8�        )��P	̝�s���A�*

	conv_loss%X=#��n        )��P	=ζs���A�*

	conv_lossy�Z=���        )��P	��s���A�*

	conv_lossR�@=�2�        )��P	62�s���A�*

	conv_loss���==�D�        )��P	c�s���A�*

	conv_lossZy�=�nj�        )��P	x��s���A�*

	conv_loss!�=y�s�        )��P	ŷs���A�*

	conv_lossD�=1��        )��P	���s���A�*

	conv_loss�3�=/@��        )��P	�&�s���A�*

	conv_loss�M=�6`         )��P	�X�s���A�*

	conv_lossz"�=`(��        )��P	���s���A�*

	conv_loss�-=P�k�        )��P	��s���A�*

	conv_loss)��<���        )��P	��s���A�*

	conv_loss��=u�-�        )��P	 �s���A�*

	conv_loss�3R=ۃ�Z        )��P	
Q�s���A�*

	conv_loss
�&=��        )��P	��s���A�*

	conv_loss�Q=ՓY        )��P	���s���A�*

	conv_loss�]=���        )��P	o�s���A�*

	conv_lossFn;=4�`        )��P	��s���A�*

	conv_lossw=���        )��P	�D�s���A�*

	conv_loss�4b=DD        )��P	�w�s���A�*

	conv_loss�W=�8�        )��P	n��s���A�*

	conv_lossq�R=���        )��P	�غs���A�*

	conv_loss��q=����        )��P	"�s���A�*

	conv_loss�P=�+-	        )��P	C;�s���A�*

	conv_loss��=*�2        )��P	�n�s���A�*

	conv_loss��Z=���        )��P	>��s���A�*

	conv_loss�""=(K�9        )��P	��s���A�*

	conv_loss�|E=�jK�        )��P	��s���A�*

	conv_loss�z=ZrL�        )��P	gG�s���A�*

	conv_loss8Å=�|:�        )��P	}�s���A�*

	conv_lossggY=�۵7        )��P	꯼s���A�*

	conv_loss�	]=�[:        )��P	M�s���A�*

	conv_loss��0=b��        )��P	��s���A�*

	conv_loss���=}�        )��P	Y�s���A�*

	conv_lossq�J=5DX�        )��P	���s���A�*

	conv_losse<�=:�        )��P	Žs���A�*

	conv_loss�r�=�R��        )��P	I��s���A�*

	conv_loss�"=�c��        )��P	�-�s���A�*

	conv_lossmI=��        )��P	X_�s���A�*

	conv_lossS1E=q�nn        )��P	��s���A�*

	conv_loss��<���        )��P	�žs���A�*

	conv_loss�*)=�ۙ�        )��P	$��s���A�*

	conv_loss&�d=Ǖf�        )��P	C*�s���A�*

	conv_loss5�=юk[        )��P	�[�s���A�*

	conv_loss=��<���        )��P	X��s���A�*

	conv_loss�=+�Rr        )��P	�ʿs���A�*

	conv_lossX�}= ]k�        )��P	��s���A�*

	conv_loss�҈=��\        )��P	:9�s���A�*

	conv_loss�x�=���        )��P	+l�s���A�*

	conv_loss#�(=�5�c        )��P	��s���A�*

	conv_loss)43=Õz~        )��P	���s���A�*

	conv_losstu=��7        )��P	��s���A�*

	conv_loss��=���        )��P	�/�s���A�*

	conv_loss�� =�;8        )��P	ze�s���A�*

	conv_loss�kL=�X%�        )��P	���s���A�*

	conv_loss��/=��[�        )��P	���s���A�*

	conv_lossd��=c�!^        )��P	�s���A�*

	conv_loss�f=�e��        )��P	�=�s���A�*

	conv_loss��7=Κ��        )��P	�s�s���A�*

	conv_loss�=���d        )��P	��s���A�*

	conv_lossްj=��        )��P	��s���A�*

	conv_loss �=M�e        )��P	�
�s���A�*

	conv_lossa4}=�[y        )��P	Y>�s���A�*

	conv_loss��M=O�6        )��P	4o�s���A�*

	conv_loss�-x=��        )��P	p��s���A�*

	conv_lossD�>=��:�        )��P	���s���A�*

	conv_lossf =֩#<        )��P	� �s���A�*

	conv_loss��<w���        )��P	
2�s���A�*

	conv_lossucN={���        )��P	kc�s���A�*

	conv_loss՗"=�ed�        )��P	���s���A�*

	conv_loss�J=I���        )��P	t��s���A�*

	conv_lossgQ:=A��,        )��P	L��s���A�*

	conv_loss�~C=���        )��P	8(�s���A�*

	conv_loss��\=-j        )��P	�W�s���A�*

	conv_loss���<p�$        )��P	7��s���A�*

	conv_loss���=�7�"        )��P	�6�s���A�*

	conv_loss�=��m        )��P	�w�s���A�*

	conv_lossX��=x4}        )��P	���s���A�*

	conv_loss���=��~�        )��P	���s���A�*

	conv_lossk=�=g�G        )��P	�
�s���A�*

	conv_loss�+= �Q�        )��P	�?�s���A�*

	conv_loss,5==g'C�        )��P	p�s���A�*

	conv_lossu�e=�b�        )��P	+��s���A�*

	conv_lossq	=Qhvh        )��P	���s���A�*

	conv_losse =��+�        )��P	��s���A�*

	conv_loss��R=�c4�        )��P	5�s���A�*

	conv_loss��n=)H��        )��P	qx�s���A�*

	conv_loss1+=�L�<        )��P	J��s���A�*

	conv_loss��+=Az�z        )��P	#��s���A�*

	conv_loss��W=�n�        )��P	��s���A�*

	conv_loss��=l^�g        )��P	7�s���A�*

	conv_loss2�=��qz        )��P	Fn�s���A�*

	conv_loss��A=��        )��P	8��s���A�*

	conv_loss�3N=�+c�        )��P	[��s���A�*

	conv_loss��K=�7X�        )��P	��s���A�*

	conv_loss�=����        )��P	�,�s���A�*

	conv_loss��=@Q��        )��P	L\�s���A�*

	conv_loss��=akM        )��P	���s���A�*

	conv_loss�\-=�癊        )��P	���s���A�*

	conv_lossZ�/=��L�        )��P	y��s���A�*

	conv_loss�V;=�D        )��P	�'�s���A�*

	conv_lossQ�I=A��*        )��P	?W�s���A�*

	conv_loss��=�h��        )��P	j��s���A�*

	conv_loss]'�=���        )��P	+��s���A�*

	conv_loss|RG=:Wf        )��P	��s���A�*

	conv_lossar=M�~        )��P	��s���A�*

	conv_lossǊg=�gq        )��P	�>�s���A�*

	conv_loss�*�=�� �        )��P	�l�s���A�*

	conv_lossXRD=���        )��P	���s���A�*

	conv_loss�H=�        )��P	��s���A�*

	conv_loss8A=�CQ        )��P	A��s���A�*

	conv_lossF�=Ǆ#�        )��P	�)�s���A�*

	conv_loss-�(=�5r        )��P	lX�s���A�*

	conv_loss��=s[�        )��P	���s���A�*

	conv_loss;Wo=��
        )��P	���s���A�*

	conv_lossu5=���/        )��P	T��s���A�*

	conv_lossms�=�	�W        )��P	��s���A�*

	conv_loss��d=V�Q        )��P	�@�s���A�*

	conv_loss3{H=n�T�        )��P	qo�s���A�*

	conv_loss�d6=�        )��P	��s���A�*

	conv_loss'7P=oH�        )��P	+��s���A�*

	conv_loss��@=h��l        )��P	���s���A�*

	conv_loss5Q0=��N�        )��P	,�s���A�*

	conv_losss�S={�S�        )��P	�\�s���A�*

	conv_loss
V�=�3�        )��P	���s���A�*

	conv_loss��=���        )��P	��s���A�*

	conv_lossچq=���        )��P	���s���A�*

	conv_lossQw�<����        )��P	�*�s���A�*

	conv_loss-�=B�1m        )��P	dY�s���A�*

	conv_lossZ�s=�_fG        )��P	��s���A�*

	conv_loss7�-=`��)        )��P		��s���A�*

	conv_lossп=�x�        )��P	���s���A�*

	conv_loss��=I�+4        )��P	1�s���A�*

	conv_losse(=<��s        )��P	lO�s���A�*

	conv_loss"o=�G�e        )��P	�~�s���A�*

	conv_loss�� =���%        )��P	
��s���A�*

	conv_loss��=�2:�        )��P	���s���A�*

	conv_lossYt =b���        )��P	��s���A�*

	conv_loss!�2=ÿ��        )��P	0Q�s���A�*

	conv_loss"=j�/�        )��P	��s���A�*

	conv_loss�[b=卙�        )��P	I��s���A�*

	conv_loss�	u=�r/>        )��P	6��s���A�*

	conv_loss�*]=F�_�        )��P	��s���A�*

	conv_loss
�P=��PW        )��P	�O�s���A�*

	conv_loss�M,=0K��        )��P	��s���A�*

	conv_lossѱL=�#�        )��P	f��s���A�*

	conv_loss*_�<;D��        )��P	���s���A�*

	conv_lossȥ�<Z��Q        )��P	 �s���A�*

	conv_loss�^	=�ZBu        )��P	�C�s���A�*

	conv_loss�0=�-ї        )��P	�s�s���A�*

	conv_loss�z=c7        )��P	G��s���A�*

	conv_lossR&=0        )��P	��s���A�*

	conv_lossO�=Q�{�        )��P	>
�s���A�*

	conv_loss"o=XJC        )��P	�;�s���A�*

	conv_loss��7=$���        )��P	�k�s���A�*

	conv_lossn[3=�LJ�        )��P	$��s���A�*

	conv_loss��=���.        )��P	���s���A�*

	conv_loss��%=K
�f        )��P	���s���A�*

	conv_loss�%=��        )��P	�.�s���A�*

	conv_loss)=p#-j        )��P	�_�s���A�*

	conv_loss�$>=!{|        )��P	���s���A�*

	conv_loss^�%=BG˼        )��P	L��s���A�*

	conv_lossm}P=_�k        )��P	m��s���A�*

	conv_loss�?=ȇ�`        )��P	� �s���A�*

	conv_lossUS=[���        )��P	ZQ�s���A�*

	conv_loss�r=o��        )��P	T��s���A�*

	conv_loss��+=8W�`        )��P	d��s���A�*

	conv_loss�x!=�6o�        )��P	���s���A�*

	conv_loss	�*=�^'        )��P	��s���A�*

	conv_loss� =Ta�{        )��P	4@�s���A�*

	conv_loss�=)e��        )��P	�q�s���A�*

	conv_loss93S=�@b        )��P	��s���A�*

	conv_loss:l�<�G��        )��P	���s���A�*

	conv_loss�$�=�w��        )��P	S�s���A�*

	conv_loss��@=@���        )��P	�3�s���A�*

	conv_lossBK=�;u        )��P	Xe�s���A�*

	conv_lossVR =�ݨ        )��P	���s���A�*

	conv_loss�L=���        )��P	�-�s���A�*

	conv_lossϯ�<���        )��P	�[�s���A�*

	conv_lossʹ�<ǲ�V        )��P	$��s���A�*

	conv_loss�m�<���        )��P	n��s���A�*

	conv_lossDr=Z+�        )��P	��s���A�*

	conv_loss]*I=<s[�        )��P	�$�s���A�*

	conv_loss�C?=�4��        )��P	<[�s���A�*

	conv_loss*+P=��,�        )��P	��s���A�*

	conv_lossO�9= ��        )��P	���s���A�*

	conv_loss&�=�7}�        )��P	;�s���A�*

	conv_loss9�H=&�CZ        )��P	a:�s���A�*

	conv_loss?�&=�߰�        )��P	�h�s���A�*

	conv_loss9ku=�9�>        )��P	���s���A�*

	conv_loss�=	=hx|{        )��P	 ��s���A�*

	conv_lossjR=	�ƶ        )��P		��s���A�*

	conv_lossA�Q=34[�        )��P	;(�s���A�*

	conv_loss�\F=[`�        )��P	�V�s���A�*

	conv_loss���=d*��        )��P	���s���A�*

	conv_loss80�<�|�        )��P	���s���A�*

	conv_loss=���_        )��P	\��s���A�*

	conv_loss��d=��x        )��P	d�s���A�*

	conv_lossή=��^+        )��P	�N�s���A�*

	conv_loss�"2=i3v        )��P	~�s���A�*

	conv_loss��<�I�        )��P	Я�s���A�*

	conv_loss�=Vg�/        )��P	;��s���A�*

	conv_loss)�b=ԕD/        )��P	��s���A�*

	conv_loss�%=2.�	        )��P	�>�s���A�*

	conv_loss���<< �.        )��P	Zp�s���A�*

	conv_loss�h=�I)        )��P	M��s���A�*

	conv_loss�	=�}�        )��P	��s���A�*

	conv_loss��<G?g        )��P	���s���A�*

	conv_lossK2=�F�        )��P	�*�s���A�*

	conv_loss\=0��        )��P	�Y�s���A�*

	conv_lossl�"=N�!�        )��P	��s���A�*

	conv_loss�<�Y��        )��P	M��s���A�*

	conv_loss��-=P�        )��P	���s���A�*

	conv_loss>�f=��b�        )��P	�"�s���A�*

	conv_loss��<����        )��P	�U�s���A�*

	conv_loss��1=$vF        )��P	ʅ�s���A�*

	conv_lossz�F=��&%        )��P	Ҹ�s���A�*

	conv_loss��E=%s�R        )��P	.��s���A�*

	conv_loss�<}��C        )��P	+�s���A�*

	conv_lossD� =62��        )��P	QG�s���A�*

	conv_loss��9=�o=        )��P	=w�s���A�*

	conv_loss'O6=���U        )��P	
��s���A�*

	conv_loss�SM=��3i        )��P	���s���A�*

	conv_loss�y=��cd        )��P	��s���A�*

	conv_lossyN=f��'        )��P	wK�s���A�*

	conv_loss��@=3�3�        )��P	�x�s���A�*

	conv_lossG�=F���        )��P	A��s���A�*

	conv_lossiKG=�=�m        )��P	���s���A�*

	conv_loss��l=L�2�        )��P	w�s���A�*

	conv_lossst=|��        )��P	�M�s���A�*

	conv_loss��=��SF        )��P	w��s���A�*

	conv_lossQ�A=�޼        )��P	��s���A�*

	conv_lossz�=��{�        )��P	���s���A�*

	conv_lossw��<��?        )��P	�
�s���A�*

	conv_lossH�!=����        )��P	�C�s���A�*

	conv_lossXH=�*��        )��P	j�s���A�*

	conv_loss�T9=�S�P        )��P	p��s���A�*

	conv_lossQJ�<���        )��P	���s���A�*

	conv_loss/a=I���        )��P	p#�s���A�*

	conv_loss�=椪�        )��P	�T�s���A�*

	conv_loss*4=��)        )��P	���s���A�*

	conv_loss�Y3=be�{        )��P	��s���A�*

	conv_loss�=�9H        )��P	/��s���A�*

	conv_lossF�g=�4*q        )��P	f�s���A�*

	conv_loss��=�s�        )��P	�D�s���A�*

	conv_loss_�(=Y��Q        )��P	tv�s���A�*

	conv_loss�z=)        )��P	s��s���A�*

	conv_loss�G�<��d�        )��P	���s���A�*

	conv_lossw�\=�˽;        )��P	�
�s���A�*

	conv_loss�c"=���        )��P	�=�s���A�*

	conv_loss�1!=�        )��P	�o�s���A�*

	conv_loss�'(=;d        )��P	ԝ�s���A�*

	conv_loss�P�<��F        )��P	���s���A�*

	conv_loss�_�<�)�        )��P	���s���A�*

	conv_loss
	+=�Uj�        )��P	/�s���A�*

	conv_lossz8=�髏        )��P	�]�s���A�*

	conv_loss��8=�)HX        )��P	[��s���A�*

	conv_loss�� =�AX4        )��P	e��s���A�*

	conv_loss�&,=ՙMx        )��P	��s���A�*

	conv_loss>K�<��        )��P	;�s���A�*

	conv_lossm�=;Eg        )��P	�I�s���A�*

	conv_loss�uZ=z��        )��P	�y�s���A�*

	conv_loss�2=h�+        )��P	���s���A�*

	conv_loss��@=� �d        )��P	@��s���A�*

	conv_loss���<yYQ        )��P	��s���A�*

	conv_loss3�=G��f        )��P	�6�s���A�*

	conv_loss�|�<�, �        )��P	�e�s���A�*

	conv_loss�<Y=�         )��P	<��s���A�*

	conv_lossG9J=��-�        )��P	��s���A�*

	conv_loss��<*z��        )��P	���s���A�*

	conv_loss7# =��:�        )��P	p#�s���A�*

	conv_loss��.=6��)        )��P	-T�s���A�*

	conv_lossg��<�
��        )��P	��s���A�*

	conv_loss�'=�8�2        )��P	���s���A�*

	conv_loss(;�<��V�        )��P	R��s���A�*

	conv_loss=\=�R�        )��P	H�s���A�*

	conv_loss�S=3UF        )��P	R�s���A�*

	conv_lossr{(=��]�        )��P	6��s���A�*

	conv_lossWv=>8s�        )��P	\��s���A�*

	conv_lossY	=�}%t        )��P	a��s���A�*

	conv_loss�}�<	ϕ�        )��P	��s���A�*

	conv_lossX=�eo        )��P	�Q�s���A�*

	conv_loss��=��X�        )��P	���s���A�*

	conv_loss��"=q�hR        )��P	Ƚ�s���A�*

	conv_loss�K�=5�        )��P	N��s���A�*

	conv_loss�=1e��        )��P	�-�s���A�*

	conv_lossi�=GS        )��P	Lc�s���A�*

	conv_loss�O=t�I�        )��P	��s���A�*

	conv_lossڐ�<�!=�        )��P	���s���A�*

	conv_loss&��<�g�	        )��P	�s���A�*

	conv_loss)=m���        )��P	�G�s���A�*

	conv_loss�7=�q��        )��P	d{�s���A�*

	conv_loss�=�G?0        )��P	\��s���A�*

	conv_loss� ;=rj�         )��P	���s���A�*

	conv_lossM�<m��        )��P	��s���A�*

	conv_loss�-N=�+�K        )��P	�K�s���A�*

	conv_loss��<O�        )��P	���s���A�*

	conv_loss��=�M        )��P	B��s���A�*

	conv_loss/	=�Y��        )��P	T��s���A�*

	conv_loss�~�<��        )��P	��s���A�*

	conv_loss�3=���G        )��P	�Q�s���A�*

	conv_loss"�	=���        )��P	p��s���A�*

	conv_lossǃ=�}�        )��P	Y��s���A�*

	conv_loss�=���	        )��P	v��s���A�*

	conv_lossGF�<#���        )��P	> �s���A�*

	conv_loss=<6=,8_E        )��P	�S�s���A�*

	conv_loss���<�?j�        )��P	!��s���A�*

	conv_loss0p�<�F��        )��P	���s���A�*

	conv_loss�o�<C	w        )��P	���s���A�*

	conv_lossey�<         )��P	]%�s���A�*

	conv_loss�-=�`�        )��P	�Y�s���A�*

	conv_loss��=��m!        )��P	���s���A�*

	conv_loss}d)=�۱]        )��P	E��s���A�*

	conv_loss�O=|�        )��P	���s���A�*

	conv_loss8�=���        )��P	&�s���A�*

	conv_loss�.8=�eD         )��P	�X�s���A�*

	conv_loss���<��        )��P	݌�s���A�*

	conv_loss��=�??        )��P	X��s���A�*

	conv_loss�MQ=	��        )��P	���s���A�*

	conv_lossf=2���        )��P	$�s���A�*

	conv_lossr�=2�
        )��P	uY�s���A�*

	conv_loss� =��I3        )��P	r��s���A�*

	conv_loss�=A׀        )��P	���s���A�*

	conv_loss���<0�        )��P	���s���A�*

	conv_lossC�=��X�        )��P	�%�s���A�*

	conv_loss(h=�]�         )��P	X�s���A�*

	conv_loss6�1=#~�        )��P	o��s���A�*

	conv_loss�c�<���        )��P	���s���A�*

	conv_lossg5R=��b        )��P	��s���A�*

	conv_loss1s==��3        )��P	$?�s���A�*

	conv_loss��a=�<S�        )��P	 s�s���A�*

	conv_loss��<��        )��P	��s���A�*

	conv_lossD�=WM        )��P	���s���A�*

	conv_loss�i=���]        )��P	m�s���A�*

	conv_loss!N�<X���        )��P	�K�s���A�*

	conv_loss~S�<z�z        )��P	{��s���A�*

	conv_losss�;=#�z+        )��P	���s���A�*

	conv_lossq�\=�Q��        )��P	��s���A�*

	conv_loss�j�<�}׷        )��P	�4�s���A�*

	conv_lossh�=~�        )��P	�h�s���A�*

	conv_loss��6=�        )��P	\��s���A�*

	conv_loss�Z�<���R        )��P	,��s���A�*

	conv_loss,=gl��        )��P	� t���A�*

	conv_loss�#=��        )��P	_7 t���A�*

	conv_loss+f�<�He�        )��P	�j t���A�*

	conv_lossYL�<�9�)        )��P	z� t���A�*

	conv_loss��{=�r�        )��P	y� t���A�*

	conv_loss�!�<�h&�        )��P	st���A�*

	conv_lossּ�<IE�        )��P	�9t���A�*

	conv_loss�}�<��'        )��P	�nt���A�*

	conv_loss-�<J��        )��P	��t���A�*

	conv_loss�#�<�_�        )��P	�t���A�*

	conv_loss�`=�]�        )��P	�t���A�*

	conv_loss�(= mC        )��P	rAt���A�*

	conv_loss2$�<�qm�        )��P	%tt���A�*

	conv_loss��=1�g�        )��P	W�t���A�*

	conv_lossP �<� �<        )��P	��t���A�*

	conv_loss=O�N        )��P	�t���A�*

	conv_loss3�!=v�+�        )��P	�At���A�*

	conv_loss4��<ˢ�        )��P	+xt���A�*

	conv_loss�N�<Z=�        )��P	�t���A�*

	conv_loss�D=^Z�        )��P	��t���A�*

	conv_loss��\=�s�        )��P	�t���A�*

	conv_loss*{H=����        )��P	�Dt���A�*

	conv_loss���<VR��        )��P	�wt���A�*

	conv_loss�<��N        )��P	�t���A�*

	conv_loss/|�<�</I        )��P	��t���A�*

	conv_loss!�'=I�3�        )��P	ht���A�*

	conv_loss��=�j(�        )��P	�Gt���A�*

	conv_loss�'�<HE~        )��P	zzt���A�*

	conv_loss:�Q=�nr�        )��P	��t���A�*

	conv_lossE=*_p�        )��P	x�t���A�*

	conv_loss��D=��b        )��P	9t���A�*

	conv_lossJ�)=1F        )��P	�Gt���A�*

	conv_lossC��<�d��        )��P	{t���A�*

	conv_lossɆ�<���        )��P	l�t���A�*

	conv_loss��(=�/}�        )��P	�Ct���A�*

	conv_loss���<�(�        )��P	�rt���A�*

	conv_loss�M=w��$        )��P	��t���A�*

	conv_loss�ڋ=I�7�        )��P	B�t���A�*

	conv_loss��=WA�        )��P	}	t���A�*

	conv_loss� =�
��        )��P	�D	t���A�*

	conv_loss�&=U�dw        )��P	sw	t���A�*

	conv_loss�}*=ۦLI        )��P	��	t���A�*

	conv_loss�0=�J��        )��P	��	t���A�*

	conv_loss�?�<^��        )��P	e
t���A�*

	conv_loss�'�<�d�         )��P	*A
t���A�*

	conv_loss� =>�Es        )��P	�z
t���A�*

	conv_loss�D=s��        )��P	�
t���A�*

	conv_loss�
V=��=�        )��P	�
t���A�*

	conv_loss��3=��{        )��P	z
t���A�*

	conv_loss�<�
        )��P	�>t���A�*

	conv_losss�=�{�        )��P	��t���A�*

	conv_loss��=<�`        )��P	�t���A�*

	conv_loss�� =|_��        )��P	��t���A�*

	conv_lossT�=|�o        )��P	�t���A�*

	conv_loss��<Ir�        )��P	�Et���A�*

	conv_loss�/�<��f        )��P	�tt���A�*

	conv_loss�I�<{I�H        )��P	 �t���A�*

	conv_loss��$=�w8�        )��P	��t���A�*

	conv_loss,+�<���s        )��P	U t���A�*

	conv_loss��<cY�y        )��P	>1t���A�*

	conv_lossw�<J)~�        )��P	;`t���A�*

	conv_losss�U=�-�Y        )��P	t�t���A�*

	conv_loss�=��G�        )��P	��t���A�*

	conv_loss8��<P���        )��P	��t���A�*

	conv_loss\�2=ll��        )��P	�!t���A�*

	conv_lossC}=U-        )��P	�Qt���A�*

	conv_loss"B
=r��        )��P	��t���A�*

	conv_loss�~�<1#�`        )��P	��t���A�*

	conv_loss4�3=,��/        )��P	��t���A�*

	conv_loss���<B�jj        )��P	t���A�*

	conv_loss�;d=�X��        )��P	�=t���A�*

	conv_loss�d�<        )��P	�mt���A�*

	conv_lossj�[=��j�        )��P	6�t���A�*

	conv_loss��%=F�J        )��P	R�t���A�*

	conv_loss�?Q=i��        )��P	Ft���A�*

	conv_lossgEp<�`X�        )��P	�<t���A�*

	conv_loss�[�<rn2�        )��P	srt���A�*

	conv_loss��<��m        )��P	�t���A�*

	conv_lossm�G=���        )��P	��t���A�*

	conv_loss�V%=; ��        )��P	lt���A�*

	conv_loss&�=���        )��P	y8t���A�*

	conv_loss<�<1��        )��P	5ht���A�*

	conv_lossk�=�%v)        )��P	^�t���A�*

	conv_losslq=e�ߴ        )��P	�t���A�*

	conv_loss��=�1��        )��P	Zt���A�*

	conv_lossn�!=}�        )��P	�9t���A�*

	conv_loss�w =���<        )��P	it���A�*

	conv_loss��<A'        )��P	��t���A�*

	conv_loss|�=0�a$        )��P	/�t���A�*

	conv_lossJ�=���8        )��P	�	t���A�*

	conv_loss4�=�n        )��P	G9t���A�*

	conv_loss��8=��a4        )��P	lt���A�*

	conv_loss��#=��p        )��P	@�t���A�*

	conv_lossLw#=�_=5        )��P	+�t���A�*

	conv_loss���< n        )��P	Lt���A�*

	conv_loss0Y4=C���        )��P	�;t���A�*

	conv_loss�� =���        )��P	`lt���A�*

	conv_lossk�==F���        )��P	`�t���A�*

	conv_loss�|�<�L�        )��P	�t���A�*

	conv_loss��<J˴8        )��P	�	t���A�*

	conv_loss�(=���        )��P	@t���A�*

	conv_loss�(=��e        )��P	6wt���A�*

	conv_lossZ�=�4�        )��P	 �t���A�*

	conv_loss�6�<�E�        )��P	R�t���A�*

	conv_loss��=��M�        )��P	 
t���A�*

	conv_loss�<=�i3	        )��P	�8t���A�*

	conv_lossL=*=���        )��P	�lt���A�*

	conv_loss��<�J�v        )��P	�t���A�*

	conv_lossކ�<�/�        )��P	��t���A�*

	conv_lossrJ�<lH�R        )��P	M�t���A�*

	conv_loss!�=6ӯb        )��P	w.t���A�*

	conv_loss�T�<��j        )��P	�`t���A�*

	conv_lossb9v<�'<        )��P	��t���A�*

	conv_loss�L:=Q��        )��P	��t���A�*

	conv_loss]��<ě[        )��P	 �t���A�*

	conv_lossR�<��B        )��P	�t���A�*

	conv_loss��	=n�P$        )��P	PNt���A�*

	conv_lossl�=w3TN        )��P	�}t���A�*

	conv_loss*�=��p8        )��P	p�t���A�*

	conv_loss.=�G�        )��P	n�t���A�*

	conv_loss��P=�&�1        )��P	�t���A�*

	conv_loss��=��?        )��P	�Bt���A�*

	conv_lossUc=QY!�        )��P	�tt���A�*

	conv_loss"�E=0@�        )��P	��t���A�*

	conv_loss��=��<        )��P	�t���A�*

	conv_loss��[=�E*�        )��P	�t���A�*

	conv_loss�Y(=�f�        )��P	�6t���A�*

	conv_loss�*=�dW        )��P	�ft���A�*

	conv_loss��1=z��        )��P	Ɩt���A�*

	conv_loss�Y=R��        )��P	��t���A�*

	conv_loss�Q=�y        )��P	��t���A�*

	conv_loss�[6=1/�y        )��P	�'t���A�*

	conv_loss�"M=�{�        )��P	mWt���A�*

	conv_lossyH=�KB�        )��P	��t���A�*

	conv_loss43�<�&�        )��P	O�t���A�*

	conv_loss�i�<�J�        )��P	L�t���A�*

	conv_lossU[�<�-_8        )��P	�*t���A�*

	conv_lossj=(j��        )��P	_t���A�*

	conv_loss��!=ɤ��        )��P	�t���A�*

	conv_lossØB=�S�X        )��P	�t���A�*

	conv_lossgFg=Bd        )��P	+t���A�*

	conv_lossv��<K�x=        )��P	%Gt���A�*

	conv_loss��=�dg6        )��P	�xt���A�*

	conv_lossX%2=��M�        )��P	*�t���A�*

	conv_loss��8=� �        )��P	��t���A�*

	conv_lossse=wa��        )��P	it���A�*

	conv_loss�|�<+��V        )��P	�It���A�*

	conv_lossx�<)�k�        )��P	{t���A�*

	conv_loss��_<��7e        )��P	Z�t���A�*

	conv_loss�K�<��w�        )��P	��t���A�*

	conv_loss/}	=B�        )��P	�t���A�*

	conv_loss�S�<�?\r        )��P	Dt���A�*

	conv_loss��<td+�        )��P	}t���A�*

	conv_loss&�6=X�h�        )��P	h�t���A�*

	conv_loss�=H�X,        )��P	}�t���A�*

	conv_loss��<> ��        )��P	 t���A�*

	conv_losss	=���        )��P	HM t���A�*

	conv_loss���<��K
        )��P	�| t���A�*

	conv_loss>7=e[��        )��P	� t���A�*

	conv_loss��=YE|@        )��P	� t���A�*

	conv_loss��<X���        )��P	�!t���A�*

	conv_loss��</���        )��P	g<!t���A�*

	conv_lossy%=���        )��P	(m!t���A�*

	conv_loss`�"=���        )��P	��!t���A�*

	conv_loss��<����        )��P	k�!t���A�*

	conv_lossgp=�T]        )��P	�"t���A�*

	conv_loss�+*=C��        )��P	�2"t���A�*

	conv_loss6�=���        )��P	Jc"t���A�*

	conv_loss��=�Jʃ        )��P	d�"t���A�*

	conv_lossZ	=d��w        )��P	��"t���A�*

	conv_loss¦�<�yD        )��P	)�"t���A�*

	conv_losse�=���        )��P	"#t���A�*

	conv_loss��<�ɘ�        )��P	�R#t���A�*

	conv_loss�� =��^        )��P	�#t���A�*

	conv_lossXoB=~��y        )��P	V�#t���A�*

	conv_loss� =����        )��P	��#t���A�*

	conv_loss���<��?~        )��P	�$t���A�*

	conv_loss{�=/�H�        )��P	F$t���A�*

	conv_losst^�<�\I        )��P	u{$t���A�*

	conv_loss_֎<��        )��P	.�$t���A�*

	conv_lossڔ�<�Mv�        )��P	Z�$t���A�*

	conv_loss��Q=�@        )��P	�%t���A�*

	conv_lossqF=��"1        )��P	2@%t���A�*

	conv_loss�'=�5J�        )��P	�q%t���A�*

	conv_loss�Z�<o	R        )��P	[�%t���A�*

	conv_loss8!=s�^        )��P	��%t���A�*

	conv_loss�(=~ȤQ        )��P	�&t���A�*

	conv_loss��=���f        )��P	�O&t���A�*

	conv_loss�(�<�
�d        )��P	��&t���A�*

	conv_loss�}=[ ��        )��P	��&t���A�*

	conv_loss�7=Avy        )��P	��&t���A�*

	conv_loss��=O�BJ        )��P	�+'t���A�*

	conv_loss���<^��E        )��P	�c't���A�*

	conv_lossF=���        )��P	ߓ't���A�*

	conv_loss�7=f��        )��P	L�'t���A�*

	conv_loss��,=�l�        )��P	!�'t���A�*

	conv_lossx��<z�$�        )��P	D5(t���A�*

	conv_loss3��<�N�        )��P	of(t���A�*

	conv_lossJh=�&��        )��P	 �(t���A�*

	conv_lossʩ1=��{b        )��P	��(t���A�*

	conv_loss��=Lum�        )��P	
�(t���A�*

	conv_loss���<P�.        )��P	<))t���A�*

	conv_loss�E�<��K8        )��P	�\)t���A�*

	conv_loss�^�<؞�E        )��P	M�)t���A�*

	conv_loss��&=��00        )��P	��)t���A�*

	conv_loss�A=Im�<        )��P	-�)t���A�*

	conv_loss+=Y]7�        )��P	�,*t���A�*

	conv_loss��=c�r        )��P	�\*t���A�*

	conv_loss�%=�*�        )��P	#�*t���A�*

	conv_loss�X=��        )��P	�*t���A�*

	conv_loss=�=O�!�        )��P	��*t���A�*

	conv_loss��K=j��        )��P	"+t���A�*

	conv_loss!q=M:�\        )��P	�T+t���A�*

	conv_loss(;	=5���        )��P	1�+t���A�*

	conv_lossfW =݅>        )��P	>�+t���A�*

	conv_loss���<��d�        )��P	V�+t���A�*

	conv_loss��<4��        )��P	t,t���A�*

	conv_loss��<U쥈        )��P	J,t���A�*

	conv_loss��W=^�K�        )��P	Oz,t���A�*

	conv_loss���<# ��        )��P	��,t���A�*

	conv_losscn=��y        )��P	��,t���A�*

	conv_loss��P=�Ҩ{        )��P	�-t���A�*

	conv_loss�|=���c        )��P	�?-t���A�*

	conv_lossw��<:���        )��P	�p-t���A�*

	conv_loss[�<���        )��P	t�-t���A�*

	conv_lossF;#=�PkZ        )��P	G�-t���A�*

	conv_loss	<=޷O\        )��P	.t���A�*

	conv_loss˸$=2�	�        )��P	^2.t���A�*

	conv_loss��<H��        )��P	�a.t���A�*

	conv_loss/��<$7
        )��P	E�.t���A�*

	conv_lossu�=<I�        )��P	�.t���A�*

	conv_loss��<�S�        )��P	g�.t���A�*

	conv_loss���<��ݷ        )��P	�'/t���A�*

	conv_loss]�=�\��        )��P	m�3t���A�*

	conv_loss==j���        )��P	G�5t���A�*

	conv_lossn��<���        )��P	��5t���A�*

	conv_loss�{�<��w<        )��P	��5t���A�*

	conv_lossq��<Ͱ��        )��P	�/6t���A�*

	conv_loss��=��e&        )��P	�a6t���A�*

	conv_loss�Z=� �        )��P	$�6t���A�*

	conv_loss�O3=k��        )��P	�6t���A�*

	conv_loss��<�7fM        )��P	��6t���A�*

	conv_lossP��<�qLy        )��P	{.7t���A�*

	conv_loss�Uf=����        )��P	�_7t���A�*

	conv_loss���<âu        )��P	��7t���A�*

	conv_loss2~�<>�1        )��P	��7t���A�*

	conv_losse�<}��        )��P	|�7t���A�*

	conv_loss"��<�·        )��P	$!8t���A�*

	conv_loss)ޕ<й�        )��P	�O8t���A�*

	conv_loss"4�<��S�        )��P	�8t���A�*

	conv_lossn!=���         )��P	r�8t���A�*

	conv_loss��=��5�        )��P	��8t���A�*

	conv_loss��<153        )��P	�9t���A�*

	conv_loss=�=/D�        )��P	�J9t���A�*

	conv_loss�6�<<ָ        )��P	3x9t���A�*

	conv_loss�w=X��s        )��P	,�9t���A�*

	conv_loss=b=Dm_        )��P	��9t���A�*

	conv_loss���<�        )��P	�:t���A�*

	conv_loss��=���        )��P	�4:t���A�*

	conv_loss#��</3n�        )��P	�g:t���A�*

	conv_lossp��<�Z�#        )��P	/�:t���A�*

	conv_lossY��<9��        )��P	��:t���A�*

	conv_loss�h	=��>n        )��P	��:t���A�*

	conv_loss]�<�ܪ        )��P	> ;t���A�*

	conv_loss�-=-���        )��P	�O;t���A�*

	conv_loss�8Q=��jS        )��P	t};t���A�*

	conv_loss���<�G��        )��P	��;t���A�*

	conv_lossM�_=�Cۣ        )��P	k�;t���A�*

	conv_loss��<��?�        )��P	�<t���A�*

	conv_lossB��<��hM        )��P	I;<t���A�*

	conv_loss��<���        )��P	cj<t���A�*

	conv_loss11=.� k        )��P	0�<t���A�*

	conv_loss�L	=��        )��P	I�<t���A�*

	conv_loss�X=�{�        )��P	��<t���A�*

	conv_loss�Ԇ<��=H        )��P	@)=t���A�*

	conv_loss�H=���        )��P	~V=t���A�*

	conv_loss��=L^��        )��P	X�=t���A�*

	conv_lossD�7=�        )��P	T�=t���A�*

	conv_loss�F�<kބ�        )��P	��=t���A�*

	conv_loss�q=qג�        )��P	�>t���A�*

	conv_loss@$=��        )��P	CB>t���A�*

	conv_loss���<.T�l        )��P	�o>t���A�*

	conv_loss�ۍ<���        )��P	ɝ>t���A�*

	conv_loss��1=a�        )��P	��>t���A�*

	conv_loss�JN=ut��        )��P	[?t���A�*

	conv_loss�/=4˼        )��P	h??t���A�*

	conv_loss���<���        )��P	Do?t���A�*

	conv_lossQ�<
�1        )��P	��?t���A�*

	conv_loss��7=_�        )��P	��?t���A�*

	conv_lossQ��<.%�        )��P	l@t���A�*

	conv_loss��=��&�        )��P	|C@t���A�*

	conv_loss��<+��S        )��P	�s@t���A�*

	conv_loss��	=-2	�        )��P	��@t���A�*

	conv_loss�##=f�Z�        )��P	�@t���A�*

	conv_lossE\�<�p�        )��P	"At���A�*

	conv_lossW�<ЇN�        )��P	;At���A�*

	conv_loss�<=c>~�        )��P	�iAt���A�*

	conv_lossB`�<ĚY�        )��P	��At���A�*

	conv_loss ��<�(,"        )��P	��At���A�*

	conv_loss��=�X9�        )��P	�Bt���A�*

	conv_loss�8�<bc�        )��P	&5Bt���A�*

	conv_loss���<����        )��P	�bBt���A�*

	conv_loss��<H5�        )��P	��Bt���A�*

	conv_loss��<�5 �        )��P	 �Bt���A�*

	conv_loss�+�<�m�        )��P	H�Bt���A�*

	conv_lossK��<��b=        )��P	�.Ct���A�*

	conv_loss%0�=��y        )��P	v^Ct���A�*

	conv_lossٓ�<{'        )��P	ތCt���A�*

	conv_loss�	�<<?�        )��P	v�Ct���A�*

	conv_lossݻ�<ܪ=        )��P	��Ct���A�*

	conv_loss��j<D44+        )��P	�Dt���A�*

	conv_losst��<�!È        )��P	�LDt���A�*

	conv_loss�7�<�t        )��P	�}Dt���A�*

	conv_loss}�<o/�         )��P	c�Dt���A�*

	conv_lossj��<��}�        )��P	��Dt���A�*

	conv_loss$;=�E�P        )��P	�Et���A�*

	conv_loss�#=	�\        )��P	AEt���A�*

	conv_loss�?=�8>&        )��P	3pEt���A�*

	conv_loss;�<(z��        )��P	}�Et���A�*

	conv_lossU#=�,C        )��P	��Et���A�*

	conv_lossV
�< l        )��P	��Et���A�*

	conv_loss �<`.��        )��P	j,Ft���A�*

	conv_loss���<�R9        )��P	aZFt���A�*

	conv_loss(��<$lB        )��P	��Ft���A�*

	conv_lossV2�<�]��        )��P	t�Ft���A�*

	conv_loss���<��6        )��P	��Ft���A�*

	conv_lossR/	=S�`9        )��P	�Gt���A�*

	conv_loss
��<�]x        )��P	,JGt���A�*

	conv_loss)��<�b-c        )��P	xGt���A�*

	conv_loss�G=�Ch3        )��P	�Gt���A�*

	conv_loss"Gw<)G.c        )��P	��Gt���A�*

	conv_loss��<� �        )��P	�Ht���A�*

	conv_loss?��<���        )��P	�2Ht���A�*

	conv_loss�A=F��        )��P	`dHt���A�*

	conv_loss�C =վ�
        )��P	��Ht���A�*

	conv_loss�S=���v        )��P	��Ht���A�*

	conv_lossq!�<A�;g        )��P	'It���A�*

	conv_loss�7�<�/��        )��P	r0It���A�*

	conv_lossp��<�J�        )��P	�bIt���A�*

	conv_loss��=���        )��P	��It���A�*

	conv_loss@z�<tK�v        )��P	�It���A�*

	conv_loss��d=[�        )��P	��It���A�*

	conv_loss��=A!#�        )��P	72Jt���A�*

	conv_loss��<��qe        )��P	�mJt���A�*

	conv_loss"��<�        )��P	D�Jt���A�*

	conv_loss� <,(��        )��P	~�Jt���A�*

	conv_loss|I=�=�]        )��P	��Jt���A�*

	conv_loss�G=UNiX        )��P	�:Kt���A�*

	conv_loss��=h�G�        )��P	�iKt���A�*

	conv_lossza%=l�y�        )��P	��Kt���A�*

	conv_lossi�R=��,�        )��P	�Kt���A�*

	conv_loss��(=��u        )��P	`�Kt���A�*

	conv_loss�y�<KH        )��P	i)Lt���A�*

	conv_loss%�=��ܓ        )��P	�XLt���A�*

	conv_loss
��<>��        )��P	׊Lt���A�*

	conv_lossb=YY�s        )��P	v�Lt���A�*

	conv_loss���<4ՇN        )��P	��Lt���A�*

	conv_loss�-=�|�        )��P	T*Mt���A�*

	conv_loss8��<�<=�        )��P	ZMt���A�*

	conv_loss�Ǝ<x���        )��P	 �Mt���A�*

	conv_loss���<m�<�        )��P	b�Mt���A�*

	conv_loss{��<~%�        )��P	�Mt���A�*

	conv_loss&�=�|�        )��P	Nt���A�*

	conv_loss�  =Q��        )��P	aNNt���A�*

	conv_loss: �</[        )��P	��Nt���A�*

	conv_loss���<`5�        )��P	�Nt���A�*

	conv_loss��<q�        )��P	��Nt���A�*

	conv_loss��<s�        )��P	ROt���A�*

	conv_loss��<HZ55        )��P	�COt���A�*

	conv_loss?$=��        )��P	7tOt���A�*

	conv_loss1��<�!�        )��P	��Ot���A�*

	conv_loss�=���        )��P	��Ot���A�*

	conv_lossDx=��Z�        )��P	� Pt���A�*

	conv_loss��<��%        )��P	�0Pt���A�*

	conv_lossr��<<1�        )��P	�aPt���A�*

	conv_loss���< '�        )��P	B�Pt���A�*

	conv_lossS0�<�P��        )��P	��Pt���A�*

	conv_loss|<�L��        )��P	��Pt���A�*

	conv_loss��<c�M1        )��P	R!Qt���A�*

	conv_loss��`<� �=        )��P	�RQt���A�*

	conv_loss��=pܭI        )��P	�Qt���A�*

	conv_lossV�n=�"�        )��P	��Qt���A�*

	conv_loss�I	=�h�y        )��P	�Qt���A�*

	conv_loss�A�<�R�\        )��P	�Rt���A�*

	conv_loss��	=��3�        )��P	+TRt���A�*

	conv_loss�m�<�qV�        )��P	��Rt���A�	*

	conv_loss� �<I%-�        )��P	�Rt���A�	*

	conv_lossvg�<�f߀        )��P	��Rt���A�	*

	conv_loss���<�R�        )��P	�St���A�	*

	conv_loss�?=d�C�        )��P	�GSt���A�	*

	conv_lossi�={tS�        )��P	�~St���A�	*

	conv_losst�S<�o�U        )��P	ҮSt���A�	*

	conv_loss�T�<@wx�        )��P	��St���A�	*

	conv_loss��;<�úb        )��P	dTt���A�	*

	conv_loss���<���        )��P	�ITt���A�	*

	conv_loss�?�<����        )��P	��Tt���A�	*

	conv_loss���<;=��        )��P	��Tt���A�	*

	conv_loss�e?<�}        )��P	�Tt���A�	*

	conv_loss�L�<��7        )��P	�Ut���A�	*

	conv_loss[g�<��F�        )��P	�MUt���A�	*

	conv_loss�B�<$��        )��P	\�Ut���A�	*

	conv_loss�l�<�KUv        )��P	��Ut���A�	*

	conv_loss�o�<�N[�        )��P	��Ut���A�	*

	conv_loss�}*=��e�        )��P	�Vt���A�	*

	conv_lossA�={�        )��P	�AVt���A�	*

	conv_losst�<X(��        )��P	�qVt���A�	*

	conv_loss�	�<���        )��P	ϠVt���A�	*

	conv_lossq[�<�e=        )��P	��Vt���A�	*

	conv_loss��=Ȏ�        )��P	�Wt���A�	*

	conv_lossӁ�<�1�U        )��P	UHWt���A�	*

	conv_loss�rD<>��v        )��P	�wWt���A�	*

	conv_loss�#�<�_X        )��P	:�Wt���A�	*

	conv_loss�"=���        )��P	��Wt���A�	*

	conv_loss��<=ڗ�        )��P	pXt���A�	*

	conv_lossV�=��ug        )��P	7Xt���A�	*

	conv_lossE�<9dl�        )��P	�eXt���A�	*

	conv_lossh<i��        )��P	��Xt���A�	*

	conv_lossr�<��b�        )��P	h�Xt���A�	*

	conv_lossԺ�<]A�7        )��P	"�Xt���A�	*

	conv_lossW�<��[|        )��P	$$Yt���A�	*

	conv_loss�H�<a|}k        )��P	�UYt���A�	*

	conv_loss
�<�7�        )��P	h�Yt���A�	*

	conv_loss*=��Ӯ        )��P	q�Yt���A�	*

	conv_loss��`<���R        )��P	�Yt���A�	*

	conv_loss��$=p�Ձ        )��P	�Zt���A�	*

	conv_loss��<cgs        )��P	(LZt���A�	*

	conv_loss]�<_��        )��P	8Zt���A�	*

	conv_loss�=���        )��P	��Zt���A�	*

	conv_loss�R0=�/u�        )��P	��Zt���A�	*

	conv_loss���<=�        )��P	A[t���A�	*

	conv_loss���<��ѝ        )��P	F[t���A�	*

	conv_loss*[�<_��I        )��P	xv[t���A�	*

	conv_loss[3�<���        )��P	�[t���A�	*

	conv_lossW� =q�x        )��P	��[t���A�	*

	conv_loss��<Ru
�        )��P	T�]t���A�	*

	conv_loss�<n���        )��P	�]t���A�	*

	conv_lossr�<���        )��P	��]t���A�	*

	conv_loss���<���        )��P	i!^t���A�	*

	conv_loss�k�<��        )��P	*T^t���A�	*

	conv_loss�J.=ߧ2j        )��P	�^t���A�	*

	conv_lossXG=���        )��P	��^t���A�	*

	conv_loss=�<����        )��P	�^t���A�	*

	conv_loss>��<�-��        )��P	�0_t���A�	*

	conv_loss��<7�{�        )��P	�a_t���A�	*

	conv_loss`��<�u'        )��P	ϕ_t���A�	*

	conv_loss��=b	��        )��P	��_t���A�	*

	conv_loss�	=7�7H        )��P	��_t���A�	*

	conv_lossҮ�<Na��        )��P	|-`t���A�	*

	conv_loss�X)=ڲ+�        )��P	``t���A�	*

	conv_loss�=���        )��P	�`t���A�	*

	conv_lossW�P<��˖        )��P	��`t���A�	*

	conv_loss �=z        )��P	��`t���A�	*

	conv_loss*�<�s        )��P	�1at���A�	*

	conv_loss��<K��        )��P	<cat���A�	*

	conv_loss)hj<�<&�        )��P	��at���A�	*

	conv_loss�p<��J�        )��P	`�at���A�	*

	conv_loss�r==�q        )��P	}�at���A�	*

	conv_lossf<j�F�        )��P	�&bt���A�	*

	conv_loss��<���F        )��P	uVbt���A�	*

	conv_loss=Ѹ<�;˨        )��P	 �bt���A�	*

	conv_loss�=�ǖ�        )��P	��bt���A�	*

	conv_loss�n�<,�8�        )��P	��bt���A�	*

	conv_lossޝ�<<�        )��P	Nct���A�	*

	conv_lossH	�<]@ @        )��P	�Kct���A�	*

	conv_loss�4�<{���        )��P	��ct���A�	*

	conv_loss*Q=��;        )��P	d�ct���A�	*

	conv_lossn��<8�         )��P	��ct���A�	*

	conv_loss��<��f�        )��P	tdt���A�	*

	conv_lossͲ�<�"T        )��P	�Edt���A�	*

	conv_loss���<���j        )��P	-udt���A�	*

	conv_lossM�=7$        )��P	-�dt���A�	*

	conv_loss��= I�        )��P	��dt���A�	*

	conv_loss)n�<�p�        )��P	s	et���A�	*

	conv_lossCɌ<���        )��P	?9et���A�	*

	conv_loss��a=�
�        )��P	iet���A�	*

	conv_loss��<�5�        )��P	�et���A�	*

	conv_lossl6�<�)��        )��P	��et���A�	*

	conv_lossA��<�&o~        )��P	V�et���A�	*

	conv_loss�m�<�5        )��P	J-ft���A�	*

	conv_lossL`=��q        )��P	�]ft���A�	*

	conv_loss�=�V\�        )��P	/�ft���A�	*

	conv_loss<:�<j�J�        )��P	�ft���A�	*

	conv_loss��<�[        )��P	f�ft���A�	*

	conv_loss���<��cw        )��P	;gt���A�	*

	conv_loss�z<py        )��P	#_gt���A�	*

	conv_loss�<��&'        )��P	��gt���A�	*

	conv_loss*q=��9�        )��P	��gt���A�	*

	conv_loss���<���         )��P	�gt���A�	*

	conv_loss|m�<Ҫvv        )��P	�"ht���A�	*

	conv_lossq�<u胨        )��P	hWht���A�	*

	conv_loss���<٤�{        )��P	C�ht���A�	*

	conv_losssC�<��%        )��P	��ht���A�	*

	conv_loss���<�u�|        )��P	�
it���A�	*

	conv_loss�&n<G�e�        )��P	<it���A�	*

	conv_loss7$=�^+�        )��P	�lit���A�	*

	conv_loss/��<�'��        )��P	��it���A�	*

	conv_loss)wO=�֕�        )��P	��it���A�	*

	conv_loss�U�<���        )��P		jt���A�	*

	conv_loss!��<vx        )��P	�>jt���A�	*

	conv_loss�N<4M�H        )��P	nnjt���A�	*

	conv_loss���<�m�#        )��P	��jt���A�	*

	conv_loss�q<>}�        )��P	y�jt���A�	*

	conv_lossT�<ff�/        )��P	Q	kt���A�	*

	conv_loss!u�<M��        )��P	5Ckt���A�	*

	conv_loss �=��H�        )��P	*wkt���A�	*

	conv_loss9�M=�dOq        )��P	i�kt���A�	*

	conv_lossa��<ZKŰ        )��P	��kt���A�	*

	conv_loss*��<��        )��P	�lt���A�	*

	conv_loss$��<��'d        )��P	x6lt���A�	*

	conv_loss�E�<�?�        )��P	�flt���A�	*

	conv_lossI=��        )��P	�lt���A�	*

	conv_losst(�<hL=�        )��P	��lt���A�	*

	conv_loss�V�<�eGU        )��P	c�lt���A�	*

	conv_loss�'�<�a�        )��P	�$mt���A�
*

	conv_lossן�<jrm        )��P	�Tmt���A�
*

	conv_losseZ=���        )��P	��mt���A�
*

	conv_loss�t�<��f*        )��P	6�mt���A�
*

	conv_loss"�=|�,         )��P	�mt���A�
*

	conv_lossB��<�P        )��P	snt���A�
*

	conv_loss�=�y�z        )��P	eHnt���A�
*

	conv_loss,��<g��=        )��P	�xnt���A�
*

	conv_loss��<kU        )��P	n�nt���A�
*

	conv_loss���<&�*@        )��P	~�nt���A�
*

	conv_lossb�=���=        )��P	%	ot���A�
*

	conv_losso��<���        )��P	�9ot���A�
*

	conv_loss�� ==:        )��P	.kot���A�
*

	conv_loss�
�<v?c^        )��P	&�ot���A�
*

	conv_loss��<�s �        )��P	��ot���A�
*

	conv_loss��<��        )��P	��ot���A�
*

	conv_loss�[)=]�9�        )��P	�+pt���A�
*

	conv_lossФ=YKH{        )��P	�[pt���A�
*

	conv_losse�=H.O�        )��P	'�pt���A�
*

	conv_loss��=׻         )��P	��pt���A�
*

	conv_loss�s=��[        )��P	��pt���A�
*

	conv_lossY`c<d��a        )��P	Q-qt���A�
*

	conv_loss���<c�s        )��P	'^qt���A�
*

	conv_loss��<�m�3        )��P	Z�qt���A�
*

	conv_loss=Vi=Jl�        )��P	7�qt���A�
*

	conv_loss$*%=@��        )��P	��qt���A�
*

	conv_loss�Z�<`suw        )��P	�'rt���A�
*

	conv_loss��<4��        )��P	�brt���A�
*

	conv_lossLc=�_u�        )��P	̔rt���A�
*

	conv_lossج`<7�/        )��P	F�rt���A�
*

	conv_lossR1�<��S        )��P	�st���A�
*

	conv_loss��<IZ�        )��P	�>st���A�
*

	conv_loss�C�<���        )��P	`ust���A�
*

	conv_lossPj=�
8        )��P	}�st���A�
*

	conv_loss��<�M        )��P	e�st���A�
*

	conv_loss�lz<�+�        )��P	�tt���A�
*

	conv_loss!˞<��h�        )��P	�5tt���A�
*

	conv_lossb�=O`�b        )��P	igtt���A�
*

	conv_loss�F=�W        )��P	H�tt���A�
*

	conv_loss�= ��        )��P	��tt���A�
*

	conv_loss_�<�{B        )��P		�tt���A�
*

	conv_loss���<�I�        )��P	�-ut���A�
*

	conv_loss�<�x�        )��P	�but���A�
*

	conv_loss�<PA�$        )��P	ەut���A�
*

	conv_loss��<Є�        )��P	��ut���A�
*

	conv_loss�	e=�*x        )��P	��ut���A�
*

	conv_lossN =b�bD        )��P	�(vt���A�
*

	conv_lossٜ�<ϕK        )��P	�Vvt���A�
*

	conv_loss�,�<G�        )��P	��vt���A�
*

	conv_loss9j�<G�@�        )��P	�vt���A�
*

	conv_loss���<�נ�        )��P	(�vt���A�
*

	conv_loss��8=�HT?        )��P	cwt���A�
*

	conv_lossc�<N�\k        )��P	�Dwt���A�
*

	conv_loss���<���[        )��P	�twt���A�
*

	conv_lossg\�<x�!N        )��P	��wt���A�
*

	conv_lossti�<��)�        )��P	��wt���A�
*

	conv_losse��<��        )��P	�xt���A�
*

	conv_loss��=a���        )��P	�7xt���A�
*

	conv_lossM��<��=r        )��P	�jxt���A�
*

	conv_loss=1�<�8�        )��P	�xt���A�
*

	conv_lossJ=U��]        )��P	��xt���A�
*

	conv_loss��<�^��        )��P	��xt���A�
*

	conv_loss��<�u��        )��P	v,yt���A�
*

	conv_lossk��<m6�.        )��P	j\yt���A�
*

	conv_loss���<'+�y        )��P	a�yt���A�
*

	conv_lossa�:<~�@�        )��P	S�yt���A�
*

	conv_lossA@�<��@        )��P	��yt���A�
*

	conv_loss�©<%79        )��P	vzt���A�
*

	conv_loss*��<ax�        )��P	4Ozt���A�
*

	conv_loss���<���M        )��P	�}zt���A�
*

	conv_loss�R=V�T�        )��P	��zt���A�
*

	conv_loss���<�PZ�        )��P	��zt���A�
*

	conv_loss�=�<X�        )��P	I{t���A�
*

	conv_lossJ޼<IZp        )��P	�R{t���A�
*

	conv_loss��<�ͺ3        )��P	Ŋ{t���A�
*

	conv_loss�P�<�s*�        )��P	��{t���A�
*

	conv_lossں�<�,DD        )��P	��{t���A�
*

	conv_loss�N<�YYx        )��P	2|t���A�
*

	conv_loss�?U<�m�        )��P	�P|t���A�
*

	conv_loss x<��        )��P	�|t���A�
*

	conv_loss�Y�<�$Y�        )��P	��|t���A�
*

	conv_loss�V=�,({        )��P	}t���A�
*

	conv_lossO��<���        )��P	�3}t���A�
*

	conv_lossGB�<n�4y        )��P	^c}t���A�
*

	conv_lossF��<���        )��P	w�}t���A�
*

	conv_loss��<���        )��P	`�}t���A�
*

	conv_loss��D=��g�        )��P	��}t���A�
*

	conv_loss=B�<'���        )��P	�0~t���A�
*

	conv_loss���<�d�o        )��P	Mb~t���A�
*

	conv_loss2�<���        )��P	ʓ~t���A�
*

	conv_lossݚ<�{        )��P	5�~t���A�
*

	conv_loss���<��|�        )��P	��~t���A�
*

	conv_loss=�=WD\�        )��P	�*t���A�
*

	conv_loss�d�<4"�        )��P	bt���A�
*

	conv_lossm	�<�A �        )��P	i�t���A�
*

	conv_loss�XA=��        )��P	��t���A�
*

	conv_lossƄ�<-���        )��P	Z �t���A�
*

	conv_loss�
=��[9        )��P	=/�t���A�
*

	conv_loss93�<�[�`        )��P	_�t���A�
*

	conv_loss�4==۰p        )��P	͎�t���A�
*

	conv_lossdA=��_        )��P	���t���A�
*

	conv_lossW��<7�F�        )��P	c�t���A�
*

	conv_loss^�<a�K<        )��P	?�t���A�
*

	conv_loss���<��o        )��P	O�t���A�
*

	conv_loss0�<!}h        )��P	x�t���A�
*

	conv_loss<�r�8        )��P	1��t���A�
*

	conv_loss0��<��ȷ        )��P	 ��t���A�
*

	conv_loss�p=�g2        )��P	�)�t���A�
*

	conv_loss;<_(��        )��P	r\�t���A�
*

	conv_loss���<!�B�        )��P	;��t���A�
*

	conv_loss�<�>R        )��P	dłt���A�
*

	conv_loss$�<���        )��P	���t���A�
*

	conv_lossH��<B˙�        )��P	�'�t���A�
*

	conv_lossX<%Ē        )��P	�V�t���A�
*

	conv_loss�
<J���        )��P	݆�t���A�
*

	conv_lossDd�<�}��        )��P	��t���A�
*

	conv_loss��=̘b        )��P	��t���A�
*

	conv_loss�@1=����        )��P	B�t���A�
*

	conv_loss ��<��        )��P	�G�t���A�
*

	conv_loss���<vH\        )��P	�y�t���A�
*

	conv_loss���<�1�        )��P	�t���A�
*

	conv_loss���<��o�        )��P	�>�t���A�
*

	conv_loss���<�l-�        )��P	�q�t���A�
*

	conv_loss9G�<�a        )��P	N��t���A�
*

	conv_loss��O<g�        )��P	H׆t���A�
*

	conv_loss��=Gn��        )��P	$
�t���A�
*

	conv_loss��i<��+w        )��P	�:�t���A�
*

	conv_loss��<)�        )��P	�l�t���A�
*

	conv_lossW0�<�?�]        )��P	��t���A�*

	conv_loss΅�<`�	X        )��P	�ڇt���A�*

	conv_loss��/<ɦ�%        )��P	y�t���A�*

	conv_lossw��<�ڠ        )��P	@�t���A�*

	conv_loss]�n<���        )��P	�o�t���A�*

	conv_loss���;�3�8        )��P	3��t���A�*

	conv_loss�J�<Ǿ�Y        )��P	�݈t���A�*

	conv_loss�,�<>h��        )��P	��t���A�*

	conv_loss���<��M        )��P	�A�t���A�*

	conv_loss?`�<&C        )��P	xv�t���A�*

	conv_lossN1�<��ׂ        )��P	*��t���A�*

	conv_loss^`�<��        )��P	U܉t���A�*

	conv_loss��=��sd        )��P	?�t���A�*

	conv_loss�R�<4�<_        )��P	@�t���A�*

	conv_loss+�<�/(�        )��P	�o�t���A�*

	conv_lossD�=�E��        )��P	���t���A�*

	conv_lossF&�<�BB�        )��P	fϊt���A�*

	conv_loss���<n4�        )��P	���t���A�*

	conv_loss���<�ᴘ        )��P	(0�t���A�*

	conv_loss���<�]�        )��P	i`�t���A�*

	conv_loss5��;e1p        )��P	ّ�t���A�*

	conv_loss�z�<��M0        )��P	dċt���A�*

	conv_loss���<�م�        )��P	K�t���A�*

	conv_loss��;=�        )��P	�"�t���A�*

	conv_loss=�z<x��        )��P	�R�t���A�*

	conv_loss�aj<��        )��P	"��t���A�*

	conv_loss^w�<��on        )��P	���t���A�*

	conv_loss�V=���v        )��P	��t���A�*

	conv_loss,��<:���        )��P	��t���A�*

	conv_loss��<�j�        )��P	GG�t���A�*

	conv_loss��=�i]        )��P	�x�t���A�*

	conv_loss�<H|h        )��P	$��t���A�*

	conv_loss� |<G��        )��P	d܍t���A�*

	conv_loss:z�<��G�        )��P	�t���A�*

	conv_loss��=�b_g        )��P	?�t���A�*

	conv_loss<h�<4�@        )��P	�o�t���A�*

	conv_losso�<���        )��P	��t���A�*

	conv_loss<��<�n�        )��P	�Ҏt���A�*

	conv_lossǳ�<�'`        )��P	�t���A�*

	conv_loss�?=�P�\        )��P	52�t���A�*

	conv_lossì�<{�[�        )��P	Wb�t���A�*

	conv_lossBџ<��8�        )��P	Ē�t���A�*

	conv_loss���<dN�`        )��P	[t���A�*

	conv_loss���<���        )��P	K�t���A�*

	conv_loss��5=>���        )��P	27�t���A�*

	conv_lossش�<�4�l        )��P	�f�t���A�*

	conv_loss���<����        )��P	o��t���A�*

	conv_loss=z�<���        )��P	jѐt���A�*

	conv_loss�:�<�=@L        )��P	G
�t���A�*

	conv_losso�<��9�        )��P	`I�t���A�*

	conv_lossZ�b<���S        )��P	|�t���A�*

	conv_loss=χ<�� �        )��P	଑t���A�*

	conv_loss72=��9        )��P	��t���A�*

	conv_loss�;�<��n�        )��P	��t���A�*

	conv_loss 9='t�H        )��P	-M�t���A�*

	conv_lossd{�<7
��        )��P	�}�t���A�*

	conv_loss�kc<U�o        )��P	|��t���A�*

	conv_lossa*=�^�        )��P	�ޒt���A�*

	conv_loss���<��5        )��P	��t���A�*

	conv_loss��=%1x<        )��P	C�t���A�*

	conv_loss�]�<3��        )��P	�z�t���A�*

	conv_loss/�<�~ާ        )��P	ү�t���A�*

	conv_lossI/=��C        )��P	)�t���A�*

	conv_loss�a<O��        )��P	��t���A�*

	conv_loss7X�<�Qr        )��P	yC�t���A�*

	conv_loss�P�<7���        )��P	�t�t���A�*

	conv_lossN�< %�        )��P	꥔t���A�*

	conv_loss� =Q�z        )��P	�הt���A�*

	conv_lossL�<2�G        )��P	�	�t���A�*

	conv_loss��<����        )��P	�:�t���A�*

	conv_loss�<A�Q�        )��P	�j�t���A�*

	conv_loss/M�<
K��        )��P	1��t���A�*

	conv_lossb��<�@�        )��P	�͕t���A�*

	conv_loss�'<<�$m}        )��P	G �t���A�*

	conv_loss�g=ۭ>�        )��P	^4�t���A�*

	conv_lossJ�<�觞        )��P	�e�t���A�*

	conv_lossci�<��ј        )��P	���t���A�*

	conv_loss��
=5̶        )��P	pȖt���A�*

	conv_lossǨt<|!�        )��P	���t���A�*

	conv_loss���<�p�        )��P	�-�t���A�*

	conv_loss2@�<,H�        )��P	9_�t���A�*

	conv_lossj�:=&}�        )��P	���t���A�*

	conv_loss���<)>�6        )��P	��t���A�*

	conv_loss�j=���        )��P	(�t���A�*

	conv_loss���<��D        )��P	�#�t���A�*

	conv_loss���<:rC        )��P	�U�t���A�*

	conv_loss���<�Z��        )��P	���t���A�*

	conv_loss���<�u2O        )��P	��t���A�*

	conv_lossT:�<�JK        )��P	�t���A�*

	conv_loss���<�o��        )��P	�t���A�*

	conv_lossR��<�'�        )��P	N�t���A�*

	conv_loss^u
=Y���        )��P	?~�t���A�*

	conv_lossڜ�<I�y        )��P	��t���A�*

	conv_lossａ<l.�        )��P	�[�t���A�*

	conv_loss2%�<�;�        )��P	G��t���A�*

	conv_loss���<�vC        )��P	���t���A�*

	conv_loss�޿<�oi~        )��P	��t���A�*

	conv_loss�	=Wz�&        )��P	�"�t���A�*

	conv_loss��<Ց��        )��P	�U�t���A�*

	conv_loss/��<Jsq        )��P	���t���A�*

	conv_loss#�=t��        )��P	>ȟt���A�*

	conv_loss({o<&�        )��P	,�t���A�*

	conv_loss�=̛d�        )��P	�;�t���A�*

	conv_loss�*<
�7        )��P	'p�t���A�*

	conv_lossj�6<^�A        )��P	��t���A�*

	conv_loss6xC<wA        )��P	9Ѡt���A�*

	conv_loss^��<15�        )��P	j�t���A�*

	conv_lossE��<�@m        )��P	44�t���A�*

	conv_loss��y<���        )��P	g�t���A�*

	conv_loss?G='���        )��P	B��t���A�*

	conv_loss��G=F"U        )��P	�ơt���A�*

	conv_lossc�<��        )��P	:��t���A�*

	conv_loss��<b9%v        )��P	�(�t���A�*

	conv_loss$hg<2��g        )��P	uV�t���A�*

	conv_lossz��<w��        )��P	T��t���A�*

	conv_loss}��<�J;        )��P	o��t���A�*

	conv_lossK��<9<.�        )��P	
�t���A�*

	conv_loss��B<%|�        )��P	�$�t���A�*

	conv_loss4�<�;7b        )��P	S�t���A�*

	conv_lossz��<��B�        )��P	E��t���A�*

	conv_lossad�;���         )��P	ɱ�t���A�*

	conv_loss�h�<`���        )��P	G�t���A�*

	conv_loss�ؚ<=�        )��P	��t���A�*

	conv_lossޯ<ԍ=u        )��P	�B�t���A�*

	conv_loss3fd<��5        )��P	�p�t���A�*

	conv_loss0��<݌!        )��P	d��t���A�*

	conv_loss!��<���        )��P	�Ϥt���A�*

	conv_loss���<a�r        )��P	���t���A�*

	conv_loss "~<^�}        )��P	�+�t���A�*

	conv_loss(c�<)$/        )��P	�Z�t���A�*

	conv_loss �~<{U�        )��P	0��t���A�*

	conv_loss!m<�ʘ4        )��P	b��t���A�*

	conv_loss�~=̺g         )��P	j�t���A�*

	conv_lossҥ<@�OP        )��P	��t���A�*

	conv_loss-�<��q�        )��P	_J�t���A�*

	conv_loss|��<SqN�        )��P	�w�t���A�*

	conv_lossn��;��;�        )��P	Ŧ�t���A�*

	conv_loss�B<�A�        )��P	צt���A�*

	conv_losss��<$$%�        )��P	S�t���A�*

	conv_loss��=�n_        )��P	�7�t���A�*

	conv_lossMs4<7���        )��P	\e�t���A�*

	conv_lossP��<��ha        )��P	c��t���A�*

	conv_lossYu�;����        )��P	�çt���A�*

	conv_loss	x�<�k[�        )��P	��t���A�*

	conv_loss�o�<�RUw        )��P	�3�t���A�*

	conv_loss��<�g!r        )��P	�`�t���A�*

	conv_loss&t<��Q�        )��P	{��t���A�*

	conv_loss�+�<t�h�        )��P	ǿ�t���A�*

	conv_lossG<r���        )��P	�t���A�*

	conv_loss�*= ���        )��P	�"�t���A�*

	conv_loss�R�<�8L�        )��P	�U�t���A�*

	conv_lossM5�<�|        )��P	���t���A�*

	conv_lossŢ<c=ұ        )��P	��t���A�*

	conv_loss
��<R P        )��P	��t���A�*

	conv_loss�W�<�>r�        )��P	'�t���A�*

	conv_loss�H�<��        )��P	V]�t���A�*

	conv_losst<#LV�        )��P	���t���A�*

	conv_loss��=<w�        )��P	ªt���A�*

	conv_loss��|<����        )��P	��t���A�*

	conv_loss0��<K�v�        )��P	�!�t���A�*

	conv_loss���<~:oA        )��P	�Q�t���A�*

	conv_loss�<���        )��P	N��t���A�*

	conv_losse[�<���        )��P	���t���A�*

	conv_loss��w<�P�        )��P	/ޫt���A�*

	conv_lossG�3=L���        )��P	��t���A�*

	conv_loss��<�M�        )��P	{<�t���A�*

	conv_loss�Q�<�h��        )��P	Sk�t���A�*

	conv_loss�o�<���}        )��P	f��t���A�*

	conv_loss7ę<%]�        )��P	{ˬt���A�*

	conv_loss]�=�S0        )��P	��t���A�*

	conv_lossX��<˺2        )��P	E,�t���A�*

	conv_loss ��<�5}�        )��P	[�t���A�*

	conv_lossn�<^N�        )��P	F��t���A�*

	conv_loss�w�<޶�<        )��P	*��t���A�*

	conv_loss/��<>4�        )��P	+�t���A�*

	conv_loss넍=���x        )��P	��t���A�*

	conv_loss�7�<OA_�        )��P	�P�t���A�*

	conv_loss�֍<ꭴN        )��P	���t���A�*

	conv_loss�"�<��        )��P	���t���A�*

	conv_lossy`�<��G        )��P	��t���A�*

	conv_loss���<��`�        )��P	��t���A�*

	conv_loss�0=���        )��P	ZH�t���A�*

	conv_loss6S<v��_        )��P	�|�t���A�*

	conv_loss���<�B        )��P	���t���A�*

	conv_loss�v =��J        )��P	Hޯt���A�*

	conv_loss��<���        )��P	��t���A�*

	conv_loss���<��S        )��P	�>�t���A�*

	conv_loss���< ���        )��P	zn�t���A�*

	conv_loss�C<��O        )��P	���t���A�*

	conv_loss��<��/        )��P	Ѱt���A�*

	conv_lossF��<�jY        )��P	���t���A�*

	conv_losszO�<�$�        )��P	�0�t���A�*

	conv_loss�y"<3�ݨ        )��P	La�t���A�*

	conv_loss3�<c:�U        )��P	���t���A�*

	conv_loss���<�tq        )��P	G,�t���A�*

	conv_loss��l<��<        )��P	�^�t���A�*

	conv_lossH��<ҋ�'        )��P	o��t���A�*

	conv_loss�=�<���T        )��P	�³t���A�*

	conv_loss3i�<44�        )��P	3�t���A�*

	conv_lossgy�<:�R        )��P	�#�t���A�*

	conv_loss�=/{A�        )��P	u�t���A�*

	conv_loss\�<#O�        )��P	0��t���A�*

	conv_loss-?1<��q        )��P	�ܴt���A�*

	conv_loss�>�<l��        )��P	��t���A�*

	conv_loss��<����        )��P	�=�t���A�*

	conv_lossYT�<�g�        )��P	>r�t���A�*

	conv_loss��=nN�        )��P	���t���A�*

	conv_lossR2�<u��        )��P	vصt���A�*

	conv_loss�<�*�        )��P	5�t���A�*

	conv_loss��<>8M        )��P	�:�t���A�*

	conv_losss��<��S�        )��P	�i�t���A�*

	conv_loss_�[<���        )��P	ʗ�t���A�*

	conv_loss���<e�_m        )��P	Mжt���A�*

	conv_lossS�G=*I��        )��P	#�t���A�*

	conv_loss�X�<l&`0        )��P	)J�t���A�*

	conv_loss5�=}�y        )��P	�y�t���A�*

	conv_loss0ޟ<��        )��P	Q��t���A�*

	conv_loss"��<�D��        )��P	5طt���A�*

	conv_loss/�<���        )��P	�t���A�*

	conv_loss{0(<���        )��P	�7�t���A�*

	conv_lossvM�<BԾr        )��P	(h�t���A�*

	conv_loss�,<2�)[        )��P	��t���A�*

	conv_lossq�t<j�2�        )��P	�Ƹt���A�*

	conv_loss�<xk�        )��P	-��t���A�*

	conv_loss��<�LU�        )��P	Q&�t���A�*

	conv_loss��-<f�6f        )��P	�V�t���A�*

	conv_loss܀�<(��w        )��P	3��t���A�*

	conv_loss�J=ِ-�        )��P	�t���A�*

	conv_loss��<���        )��P	��t���A�*

	conv_loss���<���j        )��P	��t���A�*

	conv_loss.��<etN        )��P	6L�t���A�*

	conv_loss��<����        )��P	�{�t���A�*

	conv_loss��<���        )��P	J��t���A�*

	conv_loss�p�<->8`        )��P	�ܺt���A�*

	conv_lossb|"=p���        )��P	��t���A�*

	conv_loss΃<ͳ�,        )��P	_=�t���A�*

	conv_lossю�<Pa7�        )��P	5n�t���A�*

	conv_lossL��<�g�        )��P	 ��t���A�*

	conv_lossn�<�Ԩ2        )��P	�̻t���A�*

	conv_lossu��<8w�        )��P	� �t���A�*

	conv_loss���<l�3^        )��P	O2�t���A�*

	conv_loss3!=}�'�        )��P	wa�t���A�*

	conv_lossW�<UE�        )��P	}��t���A�*

	conv_loss��<?��7        )��P	%мt���A�*

	conv_loss�KK<bv�        )��P	z��t���A�*

	conv_lossÊ�<��y        )��P	/�t���A�*

	conv_loss=|�<X�}�        )��P	m^�t���A�*

	conv_loss�"�<vZwm        )��P	둽t���A�*

	conv_loss7@�<�-}        )��P	rʽt���A�*

	conv_loss&��<���I        )��P	]��t���A�*

	conv_loss��<�$�        )��P	�-�t���A�*

	conv_loss�~�<��g�        )��P	6l�t���A�*

	conv_loss3ӏ<)ҫ�        )��P	P��t���A�*

	conv_loss�tS<���&        )��P	�Ծt���A�*

	conv_lossـ�<mzf�        )��P	m�t���A�*

	conv_lossR<�<E�}�        )��P	�6�t���A�*

	conv_lossf�<5ntk        )��P	�e�t���A�*

	conv_loss�ȑ<&�6�        )��P	˔�t���A�*

	conv_loss��<��3        )��P	�˿t���A�*

	conv_loss���<^	pF        )��P	���t���A�*

	conv_loss= =�
�        )��P	|/�t���A�*

	conv_lossdg�<�3�        )��P	�^�t���A�*

	conv_loss=�[<��I'        )��P	���t���A�*

	conv_loss�><E�	        )��P	f��t���A�*

	conv_lossT��<���P        )��P	���t���A�*

	conv_loss1d�<H��        )��P	�1�t���A�*

	conv_loss�x><��        )��P	�f�t���A�*

	conv_loss۞=��K        )��P	g��t���A�*

	conv_loss��<X��        )��P	���t���A�*

	conv_loss�<���        )��P	h��t���A�*

	conv_loss�#�<����        )��P	�&�t���A�*

	conv_loss�<���x        )��P	�W�t���A�*

	conv_loss<��<��4W        )��P	���t���A�*

	conv_loss�<Ąg�        )��P	6��t���A�*

	conv_lossj�<g�L�        )��P	���t���A�*

	conv_loss�k<���e        )��P	��t���A�*

	conv_loss/rm<
��        )��P	�M�t���A�*

	conv_loss���<R&        )��P	�}�t���A�*

	conv_loss���<4�pM        )��P	���t���A�*

	conv_loss�uX<t�;        )��P	���t���A�*

	conv_loss[o�<�{_�        )��P	�t���A�*

	conv_loss��+<����        )��P	�?�t���A�*

	conv_loss���<,��        )��P	�o�t���A�*

	conv_loss�<�Q��        )��P	͠�t���A�*

	conv_loss�ˑ<���        )��P	x��t���A�*

	conv_loss
<�*�        )��P	��t���A�*

	conv_lossh,r<�ĥB        )��P	�6�t���A�*

	conv_loss��=��        )��P	�i�t���A�*

	conv_loss���<	sp�        )��P	ߘ�t���A�*

	conv_loss�h�<�{        )��P	���t���A�*

	conv_losswS=�f         )��P	���t���A�*

	conv_loss��S=�-Ņ        )��P	**�t���A�*

	conv_loss���<���        )��P	
Z�t���A�*

	conv_loss�U<_-�o        )��P	y��t���A�*

	conv_loss��N<k�e�        )��P	?��t���A�*

	conv_loss��<E!��        )��P	D�t���A�*

	conv_loss���<�'��        )��P	�5�t���A�*

	conv_loss�Q<!+��        )��P	�h�t���A�*

	conv_loss�h�<&�        )��P	I��t���A�*

	conv_loss�y�<(w�y        )��P	B��t���A�*

	conv_loss�o�<S��r        )��P	G�t���A�*

	conv_lossXC�<���j        )��P	E�t���A�*

	conv_loss%��<Q륅        )��P	݄�t���A�*

	conv_lossJ��<�/%        )��P	��t���A�*

	conv_loss��<�        )��P	���t���A�*

	conv_loss%=;.Y        )��P	�.�t���A�*

	conv_loss��<��(t        )��P	0a�t���A�*

	conv_loss�T=ӄ_�        )��P	d��t���A�*

	conv_loss�K�<�X�        )��P	���t���A�*

	conv_loss^�<IZD�        )��P	i��t���A�*

	conv_lossP��<y L        )��P	�'�t���A�*

	conv_loss ��<��?�        )��P	EX�t���A�*

	conv_loss,�u<R���        )��P	m��t���A�*

	conv_loss^7�<}�+?        )��P	L��t���A�*

	conv_loss�b�<X��l        )��P	��t���A�*

	conv_loss��Q<!��!        )��P	�)�t���A�*

	conv_loss���<֥�}        )��P	�`�t���A�*

	conv_loss�5=굆Q        )��P	Ж�t���A�*

	conv_loss���<��my        )��P	��t���A�*

	conv_lossKL�<@�         )��P	&��t���A�*

	conv_loss/:0<O�l        )��P	-�t���A�*

	conv_loss�X[<���        )��P	�_�t���A�*

	conv_lossq`<P�r�        )��P	r��t���A�*

	conv_loss�S<��ԫ        )��P	3��t���A�*

	conv_loss|�<�%�        )��P	|��t���A�*

	conv_loss�m<�P        )��P	@'�t���A�*

	conv_loss�د<RFi        )��P	�Y�t���A�*

	conv_loss9��<=k��        )��P	V��t���A�*

	conv_lossu�5=��IT        )��P	���t���A�*

	conv_loss[�<�PE�        )��P	��t���A�*

	conv_loss�u�<6pw�        )��P	$�t���A�*

	conv_lossb)�<�t��        )��P	<V�t���A�*

	conv_loss���<&AN�        )��P	��t���A�*

	conv_loss��1<0�\        )��P	V��t���A�*

	conv_loss��<j =�        )��P	���t���A�*

	conv_loss�<J�-f        )��P	�*�t���A�*

	conv_loss� �<��כ        )��P	�Z�t���A�*

	conv_loss�T�</�,        )��P	��t���A�*

	conv_loss8q�<YUE�        )��P	3��t���A�*

	conv_loss�Ľ<2 ��        )��P	���t���A�*

	conv_lossv�	=�9W        )��P	�(�t���A�*

	conv_lossPZ�<ݛ.�        )��P	�c�t���A�*

	conv_loss��<Ǥ�        )��P	ޕ�t���A�*

	conv_loss��<Ӹk+        )��P	��t���A�*

	conv_loss�c�<3��c        )��P	i�t���A�*

	conv_lossH�<=aDD        )��P	�?�t���A�*

	conv_loss���<2���        )��P	
t�t���A�*

	conv_loss���<����        )��P	%��t���A�*

	conv_loss�=��        )��P	���t���A�*

	conv_loss��0=:��x        )��P	,
�t���A�*

	conv_loss�5�<D�>�        )��P	�F�t���A�*

	conv_loss:R�<	G        )��P	���t���A�*

	conv_loss���<KkL        )��P	���t���A�*

	conv_loss�%%<�/�e        )��P	I��t���A�*

	conv_lossyX�<	S(        )��P	�,�t���A�*

	conv_loss�<�Q�"        )��P	�e�t���A�*

	conv_loss�B�<��        )��P	���t���A�*

	conv_lossPj3<D�Z�        )��P	���t���A�*

	conv_loss�K�<���h        )��P	���t���A�*

	conv_loss:d	<Dens        )��P	,�t���A�*

	conv_loss�a�<�.0�        )��P	�]�t���A�*

	conv_loss_�<K�V        )��P	��t���A�*

	conv_lossP�a<r��`        )��P	���t���A�*

	conv_loss���<̀��        )��P	@��t���A�*

	conv_loss��\<�ê@        )��P	�.�t���A�*

	conv_lossW��<dUP�        )��P	8e�t���A�*

	conv_loss���<8A�        )��P	���t���A�*

	conv_loss ��<踲@        )��P	k��t���A�*

	conv_loss�Ig<-3t	        )��P	A��t���A�*

	conv_loss��/<�}�k        )��P	�,�t���A�*

	conv_lossР�<27]        )��P	;a�t���A�*

	conv_loss̗�<�.��        )��P	6��t���A�*

	conv_loss�(�<(�2�        )��P	��t���A�*

	conv_loss��<Ȯu�        )��P	���t���A�*

	conv_loss�1�<g�        )��P	`'�t���A�*

	conv_loss�h1<c�=�        )��P	�Y�t���A�*

	conv_loss��<vť1        )��P	N��t���A�*

	conv_losswp�<�xg�        )��P	O��t���A�*

	conv_loss�ߜ<�_��        )��P	���t���A�*

	conv_lossB��<�e�        )��P	4#�t���A�*

	conv_lossʖ<�gY        )��P	V�t���A�*

	conv_loss�R=�<�	        )��P	���t���A�*

	conv_lossn��<�5L7        )��P	���t���A�*

	conv_loss��<s
S3        )��P	P��t���A�*

	conv_loss�P$=�_!        )��P	��t���A�*

	conv_loss�Y�<�ф        )��P	�O�t���A�*

	conv_loss�Ӑ<\rh<        )��P	���t���A�*

	conv_loss�w�<����        )��P	%��t���A�*

	conv_loss�W=�X}        )��P	��t���A�*

	conv_loss�T=���        )��P	��t���A�*

	conv_loss�m�<l{�        )��P	G�t���A�*

	conv_loss���<=,�        )��P	�{�t���A�*

	conv_lossF�<���        )��P	j��t���A�*

	conv_loss�Y�<I4�`        )��P	�>�t���A�*

	conv_loss-�<�07�        )��P	To�t���A�*

	conv_loss+�<N)f�        )��P	Ɵ�t���A�*

	conv_loss�T+<�>�        )��P	o��t���A�*

	conv_loss���<����        )��P	��t���A�*

	conv_lossCŧ<����        )��P	�5�t���A�*

	conv_loss/v�<c���        )��P	�d�t���A�*

	conv_loss��<�{%#        )��P	4��t���A�*

	conv_loss�#�<Ț��        )��P	]��t���A�*

	conv_loss���<\�!        )��P	�t���A�*

	conv_loss�U�<��S        )��P	�5�t���A�*

	conv_loss���<\���        )��P	�f�t���A�*

	conv_loss�K�<K'Hy        )��P	ݖ�t���A�*

	conv_loss��<�c�        )��P	���t���A�*

	conv_loss*��<�,        )��P	���t���A�*

	conv_loss"w= �0G        )��P	76�t���A�*

	conv_lossT��<�=`�        )��P	�j�t���A�*

	conv_loss���<i�X        )��P	ۜ�t���A�*

	conv_lossL�<�V��        )��P	���t���A�*

	conv_lossc�<-��        )��P	���t���A�*

	conv_loss�Ί<E� ^        )��P	�/�t���A�*

	conv_loss�x�<�%o�        )��P	�`�t���A�*

	conv_loss��<��        )��P	9��t���A�*

	conv_loss@E�<�x��        )��P	���t���A�*

	conv_loss]|<&�v�        )��P	���t���A�*

	conv_loss̙u<�/^�        )��P	��t���A�*

	conv_loss�Ѐ<2�        )��P	Q�t���A�*

	conv_loss���<��ig        )��P	/��t���A�*

	conv_loss��K<@�g@        )��P	���t���A�*

	conv_loss���<f�        )��P	���t���A�*

	conv_loss�R�<����        )��P	��t���A�*

	conv_loss#0=9��        )��P		>�t���A�*

	conv_loss���<����        )��P	�l�t���A�*

	conv_loss�?<�[        )��P	Ǟ�t���A�*

	conv_lossRh�<9��{        )��P	6��t���A�*

	conv_loss�6�<e&iU        )��P	# �t���A�*

	conv_loss���<��?�        )��P	�2�t���A�*

	conv_loss��<��r        )��P	7f�t���A�*

	conv_loss?��<rh�9        )��P	N��t���A�*

	conv_loss�=5�R�        )��P	i��t���A�*

	conv_loss�_x<��g        )��P	v��t���A�*

	conv_loss=�G(O        )��P	�.�t���A�*

	conv_loss^��<�7�E        )��P	�_�t���A�*

	conv_loss)܇<~?�        )��P	��t���A�*

	conv_loss���<f*�        )��P	���t���A�*

	conv_loss/*=��B        )��P	 �t���A�*

	conv_lossU��<C�#�        )��P	�4�t���A�*

	conv_lossH��<zw�        )��P	h�t���A�*

	conv_loss+�"<�        )��P	Ɯ�t���A�*

	conv_loss�L�<�Q        )��P	(��t���A�*

	conv_loss�7<hS�        )��P	��t���A�*

	conv_loss+��<X��B        )��P	kR�t���A�*

	conv_loss��8<��E        )��P	v��t���A�*

	conv_loss>��<l㸣        )��P	P��t���A�*

	conv_loss�<r��        )��P	H��t���A�*

	conv_loss%08<���p        )��P	�3�t���A�*

	conv_loss�.=8?��        )��P	h�t���A�*

	conv_lossT|�<	0��        )��P	1��t���A�*

	conv_loss���<�}�        )��P	
��t���A�*

	conv_loss�l�<_�5        )��P	�t���A�*

	conv_loss�]<
�10        )��P	7M�t���A�*

	conv_lossD�<	np�        )��P	���t���A�*

	conv_loss c�<�NSQ        )��P	��t���A�*

	conv_loss��<���        )��P	(��t���A�*

	conv_loss�;(8g�        )��P	6 �t���A�*

	conv_loss>�s<�F        )��P	;^�t���A�*

	conv_loss=�=�/�@        )��P	���t���A�*

	conv_loss���<�k|        )��P	���t���A�*

	conv_loss��d<!Vn
        )��P	��t���A�*

	conv_lossC�<k$�        )��P	3?�t���A�*

	conv_loss� �<���        )��P	�s�t���A�*

	conv_loss6�<���P        )��P	b��t���A�*

	conv_lossB�i<��        )��P	���t���A�*

	conv_loss-�=h��        )��P	h�t���A�*

	conv_lossfz�<���	        )��P	�J�t���A�*

	conv_loss���<�h)        )��P	�t���A�*

	conv_loss�I�<:N��        )��P	[��t���A�*

	conv_loss��<���o        )��P	@��t���A�*

	conv_loss7G�<^���        )��P	� �t���A�*

	conv_lossA��<�չ�        )��P	!V�t���A�*

	conv_loss���<�U�        )��P	"��t���A�*

	conv_loss�=�N�$        )��P	z��t���A�*

	conv_loss�խ<Y���        )��P	���t���A�*

	conv_loss�]�<|�^        )��P	�)�t���A�*

	conv_loss`w9<#8��        )��P	#^�t���A�*

	conv_loss��<J�7�        )��P	T��t���A�*

	conv_loss7k=��k]        )��P	 ��t���A�*

	conv_lossxIu<�]        )��P	���t���A�*

	conv_loss��<�	�_        )��P	B3�t���A�*

	conv_lossZC�<��	        )��P	h�t���A�*

	conv_loss���<��g        )��P	_��t���A�*

	conv_loss�<����        )��P	���t���A�*

	conv_loss�e<��t        )��P	n�t���A�*

	conv_loss���<r���        )��P	�:�t���A�*

	conv_lossm@�<��xC        )��P	�o�t���A�*

	conv_loss߮m<�!��        )��P	��t���A�*

	conv_loss"V<�<�        )��P	���t���A�*

	conv_lossݿ<�]{�        )��P	��t���A�*

	conv_lossx!=r�Zg        )��P	YA�t���A�*

	conv_lossM��<?�J�        )��P	�t�t���A�*

	conv_losss�<y"57        )��P	���t���A�*

	conv_loss	Q�<^*        )��P	���t���A�*

	conv_loss��=,Ad�        )��P	^(�t���A�*

	conv_loss���</<8        )��P	c�t���A�*

	conv_loss)E�<�3�        )��P	z��t���A�*

	conv_lossbC=�.�        )��P	���t���A�*

	conv_loss ��<�� "        )��P	�t���A�*

	conv_loss
	�<�1�@        )��P	
;�t���A�*

	conv_loss���<�D        )��P	�s�t���A�*

	conv_loss�}�<)$8        )��P	§�t���A�*

	conv_loss�#=,��e        )��P	���t���A�*

	conv_lossZ<�<S��        )��P	6�t���A�*

	conv_loss~��<�W��        )��P	�[�t���A�*

	conv_lossDK�<mr�[        )��P	���t���A�*

	conv_loss��'=�#n        )��P	���t���A�*

	conv_loss���<��]B        )��P	
�t���A�*

	conv_loss��<��$        )��P	;>�t���A�*

	conv_loss�� <���V        )��P	�q�t���A�*

	conv_lossՎK<��|�        )��P	���t���A�*

	conv_loss���<�k�v        )��P	���t���A�*

	conv_loss���<THm        )��P	��t���A�*

	conv_loss�Y =e0�        )��P	M�t���A�*

	conv_loss�W=V���        )��P	)��t���A�*

	conv_loss���<�S�v        )��P	?��t���A�*

	conv_lossN�h<�$_        )��P	��t���A�*

	conv_loss��<�9J�        )��P	G4�t���A�*

	conv_loss���<�        )��P	k�t���A�*

	conv_lossİ�<�h�        )��P	z��t���A�*

	conv_loss���<f^�        )��P	5��t���A�*

	conv_lossL�<� �B        )��P	k�t���A�*

	conv_loss���<�#�F        )��P	E=�t���A�*

	conv_loss8��;7�*�        )��P	r�t���A�*

	conv_lossMI�<�~        )��P	}��t���A�*

	conv_loss�"�<�<o6        )��P	���t���A�*

	conv_loss`ݥ<.�gd        )��P	=�t���A�*

	conv_loss���<�r��        )��P	�J�t���A�*

	conv_loss���<�d��        )��P	M��t���A�*

	conv_loss��<6A�        )��P	��t���A�*

	conv_loss@�f<?
�        )��P	���t���A�*

	conv_loss��-<w�Q�        )��P	 �t���A�*

	conv_loss㇍<b�h�        )��P	#T�t���A�*

	conv_loss^o<O<I        )��P	���t���A�*

	conv_loss�R#<Ћ        )��P	��t���A�*

	conv_loss���<�]1        )��P	��t���A�*

	conv_loss���<2��        )��P	�'�t���A�*

	conv_loss=�<d�        )��P	E\�t���A�*

	conv_loss�ӟ;���        )��P	F��t���A�*

	conv_loss��=|ԑ�        )��P	���t���A�*

	conv_loss2{=�5C7        )��P	���t���A�*

	conv_loss�"�<�{*�        )��P	�/�t���A�*

	conv_loss�S�<\���        )��P	rw�t���A�*

	conv_loss�p�<���u        )��P	r��t���A�*

	conv_loss:�<��B�        )��P	���t���A�*

	conv_loss���;/���        )��P	C�t���A�*

	conv_loss[v�<����        )��P	�P�t���A�*

	conv_loss��@<��        )��P	u��t���A�*

	conv_lossC��<A�*�        )��P	��t���A�*

	conv_loss*��<�~��        )��P	�	�t���A�*

	conv_loss��<�&e�        )��P	IB�t���A�*

	conv_loss�$�<�0�S        )��P	=��t���A�*

	conv_loss��;�k��        )��P	���t���A�*

	conv_loss/p<&�\        )��P	���t���A�*

	conv_loss�0<�rƍ        )��P	�(�t���A�*

	conv_loss��,=�}V�        )��P	1^�t���A�*

	conv_lossh\�<�ǵ        )��P	ݓ�t���A�*

	conv_loss�$�<��f        )��P	��t���A�*

	conv_lossہ<�Ӿ�        )��P	���t���A�*

	conv_loss1.<5�C�        )��P	f3�t���A�*

	conv_lossY.u<y���        )��P	@h�t���A�*

	conv_loss�<$ʕ        )��P	!��t���A�*

	conv_loss���<V��        )��P	D��t���A�*

	conv_loss���<C٣k        )��P	o u���A�*

	conv_loss�0�<NH&�        )��P	K u���A�*

	conv_loss���<%�9        )��P	�� u���A�*

	conv_loss3��<{M��        )��P	� u���A�*

	conv_loss��<* �J        )��P	�� u���A�*

	conv_loss���<nCN�        )��P	O%u���A�*

	conv_loss�7�<��i        )��P	�[u���A�*

	conv_lossC�:<3H�p        )��P	p�u���A�*

	conv_lossn��<̓��        )��P	R�u���A�*

	conv_loss��<�\��        )��P	��u���A�*

	conv_loss���<��N        )��P	1u���A�*

	conv_loss�Æ<�;��        )��P	�gu���A�*

	conv_loss¦z<�क़        )��P	V�u���A�*

	conv_loss�
�<��Y_        )��P	Y�u���A�*

	conv_loss+��<q���        )��P	�u���A�*

	conv_loss���<m��        )��P	f;u���A�*

	conv_lossaZ5<�9��        )��P	�ru���A�*

	conv_loss2�=FuL        )��P	��u���A�*

	conv_loss�ݢ<�y$b        )��P	D�u���A�*

	conv_loss��<D�8        )��P	�u���A�*

	conv_loss���<�AQ6        )��P	�Cu���A�*

	conv_loss���<���        )��P	Hyu���A�*

	conv_loss��<NLo        )��P	c�u���A�*

	conv_lossaZ�<�1<�        )��P	�u���A�*

	conv_lossXB=d        )��P	Au���A�*

	conv_loss��<3ç�        )��P	>Qu���A�*

	conv_loss�@y<Ag�Z        )��P	J�u���A�*

	conv_lossQ��<A�d�        )��P	��u���A�*

	conv_loss�[�<~��X        )��P	d�
u���A�*

	conv_loss�|�<��
        )��P	m"u���A�*

	conv_loss�h�<�Z�R        )��P	�Ru���A�*

	conv_loss��=qy�3        )��P	�u���A�*

	conv_loss�ޯ<o.e        )��P	h�u���A�*

	conv_loss\=�6        )��P	��u���A�*

	conv_loss�v�<bW�0        )��P	�u���A�*

	conv_loss��<�w�|        )��P	Ou���A�*

	conv_loss���</e�        )��P	%�u���A�*

	conv_loss��<Dǧ�        )��P	h�u���A�*

	conv_loss�t<q�fr        )��P	X�u���A�*

	conv_loss�&<�Mx]        )��P	$'u���A�*

	conv_loss��=Щ��        )��P	Uu���A�*

	conv_lossbi�<�Fw,        )��P	a�u���A�*

	conv_loss/��< ��        )��P	E�u���A�*

	conv_loss�
<_�         )��P	��u���A�*

	conv_lossB�<`u�l        )��P	�u���A�*

	conv_lossHq�<fhJK        )��P	jHu���A�*

	conv_loss��<Z�h'        )��P	\yu���A�*

	conv_loss�={Q,i        )��P	:�u���A�*

	conv_loss{�<҈�h        )��P	>�u���A�*

	conv_loss�F�<����        )��P	Du���A�*

	conv_lossg��<3|��        )��P	14u���A�*

	conv_lossl�<o�K        )��P	�fu���A�*

	conv_lossJ�<��x        )��P	ߕu���A�*

	conv_loss���<W&n�        )��P	�u���A�*

	conv_loss�X�<!e�        )��P	��u���A�*

	conv_loss�<�Ll�        )��P	su���A�*

	conv_lossu(=las        )��P	aNu���A�*

	conv_loss/�<�D^         )��P	�}u���A�*

	conv_loss��<09n        )��P	��u���A�*

	conv_loss�?<о�<        )��P	0�u���A�*

	conv_loss�%�;4�7        )��P	�	u���A�*

	conv_loss��[<�*'1        )��P	(:u���A�*

	conv_loss�#�<J���        )��P	ju���A�*

	conv_lossP~�<':E        )��P	w�u���A�*

	conv_loss���<�1i        )��P	�u���A�*

	conv_loss<�7ܯ        )��P	��u���A�*

	conv_lossD>�<+�i        )��P	�*u���A�*

	conv_loss�L<#?�{        )��P	�Zu���A�*

	conv_loss��<�k�r        )��P	�u���A�*

	conv_loss�f�;Ns�X        )��P	��u���A�*

	conv_loss_H�<��'        )��P	��u���A�*

	conv_loss��<D�eV        )��P	ju���A�*

	conv_lossY�u<����        )��P	Lu���A�*

	conv_loss�V<*0�        )��P	l|u���A�*

	conv_loss�L:<=`%�        )��P	��u���A�*

	conv_loss�hi<�O$!        )��P	6�u���A�*

	conv_loss�a�<3�        )��P	�u���A�*

	conv_loss�<�T��        )��P	<u���A�*

	conv_loss-Í<�UFU        )��P	�ou���A�*

	conv_loss�m�<�c�x        )��P	��u���A�*

	conv_loss���<�~[�        )��P	�u���A�*

	conv_lossvw�<��        )��P	�u���A�*

	conv_loss��;����        )��P	�Fu���A�*

	conv_lossd�<��|T        )��P	�}u���A�*

	conv_lossJ�<��b        )��P	�u���A�*

	conv_lossh.�<��re        )��P	��u���A�*

	conv_loss�<p��        )��P	�u���A�*

	conv_lossg��<.�dz        )��P	�[u���A�*

	conv_loss=��<'t�        )��P	��u���A�*

	conv_loss��p<���        )��P	��u���A�*

	conv_loss� �<
���        )��P	��u���A�*

	conv_loss�3�<A��;        )��P	�$u���A�*

	conv_loss˳�<���        )��P	�[u���A�*

	conv_loss|��<�fth        )��P	]�u���A�*

	conv_losstt�<�`�I        )��P	8�u���A�*

	conv_loss2��<���H        )��P	��u���A�*

	conv_loss9�~<V���        )��P	�#u���A�*

	conv_lossH�<���F        )��P	�[u���A�*

	conv_loss{�\<�Y1*        )��P	o�u���A�*

	conv_lossY�_<0�	�        )��P	��u���A�*

	conv_lossh�<�y;        )��P	`�u���A�*

	conv_loss��<'��        )��P	�*u���A�*

	conv_loss�u�<���        )��P	P[u���A�*

	conv_loss�W�<�gS        )��P	��u���A�*

	conv_loss圣<?,�        )��P	V�u���A�*

	conv_loss�J$<&CSH        )��P	C�u���A�*

	conv_lossz�<���        )��P	� u���A�*

	conv_loss�-8<z�        )��P	pRu���A�*

	conv_loss���<}�U\        )��P	q�u���A�*

	conv_loss�,�<���        )��P	��u���A�*

	conv_lossTU�<�*7�        )��P	��u���A�*

	conv_loss%��<)�=�        )��P	[u���A�*

	conv_loss��=��k�        )��P	dLu���A�*

	conv_loss�z�<��+�        )��P	�u���A�*

	conv_loss�b[<C�A        )��P	'�u���A�*

	conv_loss�?S<�m�6        )��P	��u���A�*

	conv_loss@q�<V"��        )��P	Qu���A�*

	conv_loss�$�<�Ox�        )��P	�Hu���A�*

	conv_loss��<��F�        )��P	Fyu���A�*

	conv_loss�|�<B�        )��P	�u���A�*

	conv_loss�t�<�O9d        )��P	L�u���A�*

	conv_loss�*�<38�g        )��P	�u���A�*

	conv_lossв <h��        )��P	@Ju���A�*

	conv_loss`�J<��        )��P	B}u���A�*

	conv_lossE�<���e        )��P	"�u���A�*

	conv_loss)	�<r~j        )��P		�u���A�*

	conv_loss==s�$H        )��P	�u���A�*

	conv_loss�?�<�/�6        )��P	�Du���A�*

	conv_loss���<]�Dd        )��P	�wu���A�*

	conv_loss���</��        )��P	c�u���A�*

	conv_loss�Xo<^��@        )��P	:�u���A�*

	conv_loss�$:<����        )��P	� u���A�*

	conv_loss��<ݧ�s        )��P	�S u���A�*

	conv_loss��$<ր�        )��P	� u���A�*

	conv_loss<پ<���        )��P	�� u���A�*

	conv_loss�2<*"Qx        )��P	u� u���A�*

	conv_loss��}<$�        )��P	�/!u���A�*

	conv_loss]��<��U�        )��P	�e!u���A�*

	conv_lossÉ�<2&3p        )��P	ɞ!u���A�*

	conv_loss�dR<'�J�        )��P	��!u���A�*

	conv_loss⊨<⫓&        )��P	�"u���A�*

	conv_loss`�j<�ol        )��P	�2"u���A�*

	conv_loss��<#*ϱ        )��P	�g"u���A�*

	conv_loss~��<���        )��P	W�"u���A�*

	conv_lossS9=ق6>        )��P	#�"u���A�*

	conv_loss�WN<"�%�        )��P	 #u���A�*

	conv_loss�~F<:��        )��P	v8#u���A�*

	conv_lossm}&=�al        )��P	�m#u���A�*

	conv_loss�<ޜ�        )��P	��#u���A�*

	conv_loss��<��/�        )��P	T�#u���A�*

	conv_loss$��<9o�        )��P	$u���A�*

	conv_loss��w<Y�v        )��P	n3$u���A�*

	conv_lossV<�<�3�        )��P	Nd$u���A�*

	conv_lossY.�<"�֣        )��P	~�$u���A�*

	conv_loss���<Y�ݿ        )��P	c�$u���A�*

	conv_loss���<;�b        )��P	��$u���A�*

	conv_loss#�X<���        )��P	�,%u���A�*

	conv_loss���<3�S        )��P	�`%u���A�*

	conv_lossK<��Q        )��P	��%u���A�*

	conv_loss�-<�+�        )��P	J�%u���A�*

	conv_loss�k�<�Si2        )��P	k�%u���A�*

	conv_losst�<���        )��P	z(&u���A�*

	conv_loss1;8<R��6        )��P	�Y&u���A�*

	conv_loss���<^{�        )��P	��&u���A�*

	conv_loss�Q<���        )��P	��&u���A�*

	conv_lossN�<P��w        )��P	=�&u���A�*

	conv_loss_ i;��        )��P	"'u���A�*

	conv_lossb��<X|J�        )��P	cS'u���A�*

	conv_loss��<?�r�        )��P	J�'u���A�*

	conv_loss�$==#~        )��P	p�'u���A�*

	conv_loss�y�<��4        )��P	�'u���A�*

	conv_loss?�<]i�        )��P	Y(u���A�*

	conv_loss�m6=��r        )��P	L(u���A�*

	conv_loss=g�<蘛        )��P	v}(u���A�*

	conv_lossR Q<����        )��P	ƭ(u���A�*

	conv_lossA�<�|h�        )��P	�(u���A�*

	conv_loss��d<VA�z        )��P	W)u���A�*

	conv_loss�N�<7Cl        )��P	}E)u���A�*

	conv_loss"ǭ;�?=�        )��P	dx)u���A�*

	conv_lossc�<��E        )��P	�)u���A�*

	conv_loss��}<W�{        )��P	/�)u���A�*

	conv_loss���;q�        )��P	Z!*u���A�*

	conv_lossTc<�׽�        )��P	W*u���A�*

	conv_loss��y<�_2�        )��P	��*u���A�*

	conv_lossϨ�<�8�(        )��P	1�*u���A�*

	conv_losso/�<QM��        )��P	2+u���A�*

	conv_loss��<;��        )��P	d8+u���A�*

	conv_loss/J=�W��        )��P	�l+u���A�*

	conv_loss�<�N�        )��P	��+u���A�*

	conv_lossR��<vR�        )��P	��+u���A�*

	conv_loss�^<�L��        )��P	g,u���A�*

	conv_loss$�<ו��        )��P	><,u���A�*

	conv_loss%��<3 �        )��P	��,u���A�*

	conv_loss���<c��Q        )��P	Գ,u���A�*

	conv_loss5|�<s��        )��P	�,u���A�*

	conv_lossx��<f��        )��P	[-u���A�*

	conv_loss��|<I�
[        )��P	vN-u���A�*

	conv_loss�Xy<���0        )��P	Ԑ-u���A�*

	conv_lossN�G<�        )��P	��-u���A�*

	conv_loss}��;
	��        )��P	�-u���A�*

	conv_loss���<,���        )��P	B'.u���A�*

	conv_losste�<���        )��P	Z.u���A�*

	conv_losses�<����        )��P	O�.u���A�*

	conv_loss�ކ<܈Q�        )��P	�.u���A�*

	conv_loss�s�<�(��        )��P	I�.u���A�*

	conv_loss�H7<cJ�        )��P	�'/u���A�*

	conv_loss��<�ǟ/        )��P	l\/u���A�*

	conv_lossY��<[Ԫ        )��P	'�/u���A�*

	conv_loss��<��x�        )��P	��/u���A�*

	conv_lossI!<n��        )��P	1�/u���A�*

	conv_loss���;��K�        )��P	[0u���A�*

	conv_loss��<.�'        )��P	XP0u���A�*

	conv_loss�} <��        )��P	ف0u���A�*

	conv_loss�s<��z�        )��P	q�0u���A�*

	conv_loss(:�<���        )��P	9�0u���A�*

	conv_lossi�<2��,        )��P	�1u���A�*

	conv_loss�n�<��G_        )��P	JE1u���A�*

	conv_loss"l(=��@        )��P	tu1u���A�*

	conv_loss��}<�g��        )��P	S�1u���A�*

	conv_loss�d�<S�Ӊ        )��P	3�1u���A�*

	conv_lossG
M<-E�K        )��P	2u���A�*

	conv_loss���<���        )��P	A2u���A�*

	conv_loss��.=170�        )��P	�s2u���A�*

	conv_loss
g�<�k��        )��P	c�2u���A�*

	conv_loss�<f<'ʐ^        )��P	��2u���A�*

	conv_loss���<
#f        )��P	�
3u���A�*

	conv_loss�v<E�1        )��P	�<3u���A�*

	conv_lossÀ�<�        )��P	�o3u���A�*

	conv_lossY�
<�7        )��P	��3u���A�*

	conv_lossw5�<L��-        )��P	k55u���A�*

	conv_loss� �<�8        )��P	�e5u���A�*

	conv_loss{� =�j�        )��P	q�5u���A�*

	conv_lossu�i<UI        )��P	�5u���A�*

	conv_loss�7N<Hp<        )��P	��5u���A�*

	conv_lossA�<��;        )��P	976u���A�*

	conv_loss \�<�{(|        )��P	Xf6u���A�*

	conv_lossk>�<��$        )��P	��6u���A�*

	conv_lossFN<����        )��P	L�6u���A�*

	conv_loss��<[CUw        )��P	�7u���A�*

	conv_loss�_�<]e�        )��P	!?7u���A�*

	conv_loss��<�^�        )��P	�v7u���A�*

	conv_lossVڻ<��xw        )��P	�7u���A�*

	conv_loss& �<�*�        )��P	��7u���A�*

	conv_loss�I�<���        )��P	�8u���A�*

	conv_loss:L<@&߮        )��P	�;8u���A�*

	conv_loss��<��         )��P	�k8u���A�*

	conv_lossp�=LOX        )��P	@�8u���A�*

	conv_loss�܆<�'�/        )��P	��8u���A�*

	conv_lossW�<��o�        )��P	��8u���A�*

	conv_loss��<���        )��P	F09u���A�*

	conv_loss|q�<G��        )��P	�`9u���A�*

	conv_loss�<�Y�        )��P	א9u���A�*

	conv_loss���;��d        )��P	��9u���A�*

	conv_loss�u�<{^�        )��P	
�9u���A�*

	conv_loss$=���        )��P	�%:u���A�*

	conv_lossn�X<k�du        )��P	�T:u���A�*

	conv_loss�I�<�u��        )��P	4�:u���A�*

	conv_loss�>�<i�}�        )��P	Ƶ:u���A�*

	conv_lossJU<�{�        )��P	��:u���A�*

	conv_loss��^<�[A5        )��P	>;u���A�*

	conv_loss�:�<h˝d        )��P	�G;u���A�*

	conv_lossU�	=�l�        )��P	�x;u���A�*

	conv_loss]�u<$�$�        )��P	h�;u���A�*

	conv_loss��|<97we        )��P	+�;u���A�*

	conv_loss�4<N�        )��P	�<u���A�*

	conv_loss���<���l        )��P	�;<u���A�*

	conv_loss���<�7�X        )��P	�m<u���A�*

	conv_loss�u�<LC�~        )��P	Ξ<u���A�*

	conv_loss�m�<K	�        )��P	��<u���A�*

	conv_loss��=J�        )��P	�<u���A�*

	conv_loss�X#=�CBi        )��P	�-=u���A�*

	conv_loss˚�<צH        )��P	\_=u���A�*

	conv_loss;~<.�Q        )��P	ɏ=u���A�*

	conv_loss�*�<���        )��P	,�=u���A�*

	conv_lossRG�<�b��        )��P	��=u���A�*

	conv_loss倯<%�@k        )��P	� >u���A�*

	conv_loss��}<��1        )��P	�S>u���A�*

	conv_lossu<cu#        )��P	��>u���A�*

	conv_loss�1�<a��:        )��P	A�>u���A�*

	conv_loss��<�E��        )��P	��>u���A�*

	conv_loss�q<�=�        )��P	*?u���A�*

	conv_loss��<>�x        )��P	�[?u���A�*

	conv_loss���<��6W        )��P	p�?u���A�*

	conv_losso6'<�.��        )��P	q�?u���A�*

	conv_loss�H�<�F�        )��P	��?u���A�*

	conv_loss�a�<�<�         )��P	�&@u���A�*

	conv_loss3Ե<*>�        )��P	�b@u���A�*

	conv_loss2�<�[��        )��P	��@u���A�*

	conv_lossR:<�K9�        )��P	}�@u���A�*

	conv_loss��<K�Q        )��P	$�@u���A�*

	conv_lossܱ�<n��        )��P	u/Au���A�*

	conv_lossqW< ��8        )��P	`sAu���A�*

	conv_loss�*#<y���        )��P	��Au���A�*

	conv_lossr�<�x�        )��P	��Au���A�*

	conv_loss��r<�H�L        )��P	KBu���A�*

	conv_loss�
�<�5��        )��P	�<Bu���A�*

	conv_loss'C<���        )��P	mBu���A�*

	conv_loss��<���        )��P	�Bu���A�*

	conv_loss�k�<td+'        )��P	��Bu���A�*

	conv_lossP|�<eẹ        )��P	Cu���A�*

	conv_loss�S#<��/�        )��P	�:Cu���A�*

	conv_loss���<9*�J        )��P	}Cu���A�*

	conv_loss���<� �        )��P	V�Cu���A�*

	conv_lossc�<�!�F        )��P	��Cu���A�*

	conv_lossH�<h{�        )��P	}Du���A�*

	conv_loss�<yiM�        )��P	iLDu���A�*

	conv_loss:��<21�        )��P	f|Du���A�*

	conv_loss|��<2�L        )��P	�Du���A�*

	conv_loss�Q�<����        )��P	��Du���A�*

	conv_loss�g�;m
&�        )��P	�
Eu���A�*

	conv_loss-�&<�+��        )��P		:Eu���A�*

	conv_losskXR<#h8        )��P	�hEu���A�*

	conv_loss��<�E9        )��P	3�Eu���A�*

	conv_loss/�<�A�        )��P	�Eu���A�*

	conv_loss�b�;cAmX        )��P	P�Eu���A�*

	conv_losse��<��e�        )��P	�+Fu���A�*

	conv_lossr�<Yg�4        )��P	\Fu���A�*

	conv_loss=��<Rt(�        )��P	��Fu���A�*

	conv_loss@�<8d�        )��P	�Fu���A�*

	conv_loss]�<pm��        )��P	/�Fu���A�*

	conv_loss h<����        )��P	YGu���A�*

	conv_lossO/w<�B        )��P	�NGu���A�*

	conv_loss]A�<W�s�        )��P	��Gu���A�*

	conv_loss!�<	.�N        )��P	׸Gu���A�*

	conv_loss1��<
���        )��P	��Gu���A�*

	conv_lossZv�<C���        )��P	�Hu���A�*

	conv_lossx�b<��|        )��P	SKHu���A�*

	conv_loss�=���;        )��P	�|Hu���A�*

	conv_loss)��<�"�        )��P	E�Hu���A�*

	conv_loss���<�_|R        )��P	&�Hu���A�*

	conv_loss�>�<��g        )��P	�"Iu���A�*

	conv_lossáz<9J��        )��P	RIu���A�*

	conv_loss=��<0�3        )��P	u�Iu���A�*

	conv_loss<�J<�ٱ5        )��P	�Iu���A�*

	conv_loss{<�7[        )��P	��Iu���A�*

	conv_loss��=d]��        )��P	TJu���A�*

	conv_loss�I�<=�        )��P	�HJu���A�*

	conv_loss��C<b�        )��P	xJu���A�*

	conv_loss4?<�2        )��P	ܸJu���A�*

	conv_lossηL<J�$        )��P	[�Ju���A�*

	conv_lossÌ�<�6�        )��P	PKu���A�*

	conv_loss��<n\�5        )��P	�IKu���A�*

	conv_loss�te<�uG@        )��P	�Ku���A�*

	conv_lossH��<�Y"�        )��P	;�Ku���A�*

	conv_loss��<U�x�        )��P	#�Ku���A�*

	conv_loss�+@<)*�        )��P	�"Lu���A�*

	conv_loss��<a��
        )��P	�SLu���A�*

	conv_loss�'<Rў5        )��P	��Lu���A�*

	conv_loss(�<J�-        )��P	ڳLu���A�*

	conv_lossm�<s�]�        )��P	��Lu���A�*

	conv_lossZ)�<�1L�        )��P	�Mu���A�*

	conv_loss�FZ<� A~        )��P	�FMu���A�*

	conv_lossCO$=��        )��P	�vMu���A�*

	conv_loss�y<r��g        )��P	%�Mu���A�*

	conv_loss�d�<��fV        )��P	��Mu���A�*

	conv_loss�f<�"f�        )��P	1Nu���A�*

	conv_loss���<q�3        )��P	88Nu���A�*

	conv_loss(��<c΅        )��P	�gNu���A�*

	conv_lossv]�;rs<�        )��P	��Nu���A�*

	conv_loss��"<j��        )��P	��Nu���A�*

	conv_loss���<Ϙ��        )��P	��Nu���A�*

	conv_lossWAi<�kڼ        )��P	�+Ou���A�*

	conv_loss�R<�U�        )��P	J[Ou���A�*

	conv_loss�Ð<���        )��P	ۋOu���A�*

	conv_lossB�=Hdx�        )��P	ƼOu���A�*

	conv_loss��<V�5        )��P	_�Ou���A�*

	conv_loss7�\<P��x        )��P	1Pu���A�*

	conv_lossnU<��        )��P	YOPu���A�*

	conv_lossʍ�<6�        )��P	�~Pu���A�*

	conv_losst�V<��(@        )��P	�Pu���A�*

	conv_loss
�<���        )��P	'�Pu���A�*

	conv_loss�Z�<�kF}        )��P	Qu���A�*

	conv_loss9 �<�6        )��P	�>Qu���A�*

	conv_loss�%<Z��-        )��P	eoQu���A�*

	conv_loss"�<�%��        )��P	ѢQu���A�*

	conv_loss2�<|�}w        )��P	��Qu���A�*

	conv_loss���;�k�        )��P	/Ru���A�*

	conv_loss{,#<e&��        )��P	�4Ru���A�*

	conv_loss�2�;�hn        )��P	�fRu���A�*

	conv_loss"�x<<��        )��P	�Ru���A�*

	conv_loss"�<�i*        )��P	��Ru���A�*

	conv_loss�Ʈ<��3�        )��P	d
Su���A�*

	conv_loss�Y<�T�        )��P	�?Su���A�*

	conv_loss�g�<\ڛ        )��P	tSu���A�*

	conv_loss�$�<���        )��P	��Su���A�*

	conv_loss8b�;�v�        )��P	��Su���A�*

	conv_loss	=s_�        )��P	�Tu���A�*

	conv_lossj<-�~=        )��P	$JTu���A�*

	conv_loss��<$R�        )��P	�Tu���A�*

	conv_loss䮫<*5�        )��P	��Tu���A�*

	conv_loss��<eI�        )��P	h�Tu���A�*

	conv_loss�<�\        )��P	�Uu���A�*

	conv_loss�q�<O��H        )��P	�NUu���A�*

	conv_loss=ĉ<�_��        )��P	�Uu���A�*

	conv_loss���<�k�        )��P	��Uu���A�*

	conv_lossl<�n�<        )��P	q�Uu���A�*

	conv_loss��<6lP	        )��P	�Vu���A�*

	conv_loss�Qg<�*Ua        )��P	OVu���A�*

	conv_loss<E<��?�        )��P	WVu���A�*

	conv_loss��@<����        )��P	�Vu���A�*

	conv_loss�k<ka��        )��P	��Vu���A�*

	conv_loss�{�<"���        )��P	 Wu���A�*

	conv_loss���<��l        )��P	BWu���A�*

	conv_loss�g�<Y'22        )��P	�rWu���A�*

	conv_lossw<�n�N        )��P	g�Wu���A�*

	conv_loss�o'<���        )��P	N�Wu���A�*

	conv_loss�{�;��S�        )��P	�Xu���A�*

	conv_loss��4<��Fm        )��P	�4Xu���A�*

	conv_loss�=��'�        )��P	�dXu���A�*

	conv_loss C�<~�P        )��P	#�Xu���A�*

	conv_loss�D9<F��M        )��P	x�Xu���A�*

	conv_loss(�[<?g��        )��P	��Xu���A�*

	conv_loss/�p<�6M�        )��P	�(Yu���A�*

	conv_loss�P;<;ck        )��P	F]Yu���A�*

	conv_loss�.�<�G�B        )��P	��Yu���A�*

	conv_loss�u<� ��        )��P	@�Yu���A�*

	conv_loss��<P��o        )��P	��Yu���A�*

	conv_loss���<W:O        )��P	c$Zu���A�*

	conv_loss�d<S�C�        )��P	�TZu���A�*

	conv_loss�q�;,��\        )��P	>�Zu���A�*

	conv_lossq��<X��        )��P	��Zu���A�*

	conv_loss�DF<i���        )��P	��Zu���A�*

	conv_loss��=�T �        )��P	�[u���A�*

	conv_loss�iy<���b        )��P	4I[u���A�*

	conv_loss�[m<�Es]        )��P		}[u���A�*

	conv_loss��;�P�        )��P	�[u���A�*

	conv_loss��	<���O        )��P	��[u���A�*

	conv_lossQ= U��        )��P	�\u���A�*

	conv_lossN�e<Xa9        )��P	L>\u���A�*

	conv_lossMJ�<��+        )��P	��]u���A�*

	conv_loss�M�;�`�        )��P	^u���A�*

	conv_loss��;�p        )��P	76^u���A�*

	conv_lossQI<�A�F        )��P	Nl^u���A�*

	conv_lossY@S<��ƒ        )��P	�^u���A�*

	conv_loss+:=l�n�        )��P	��^u���A�*

	conv_lossJ��<0�c�        )��P	 _u���A�*

	conv_losse�<
#M        )��P	�5_u���A�*

	conv_loss��<e���        )��P	�k_u���A�*

	conv_loss�}0<��5�        )��P	��_u���A�*

	conv_loss-<�        )��P	��_u���A�*

	conv_loss�<�׳�        )��P	b`u���A�*

	conv_lossm0�<���        )��P	<G`u���A�*

	conv_lossn�B<`y��        )��P	�{`u���A�*

	conv_loss��K<C%�F        )��P	�`u���A�*

	conv_loss2�Q<D6y�        )��P	�`u���A�*

	conv_loss�t�<��I�        )��P	d
au���A�*

	conv_lossC�]<?��        )��P	x:au���A�*

	conv_loss�J!<���        )��P	�lau���A�*

	conv_lossz�<�o�        )��P	ܝau���A�*

	conv_loss��A<Q�=�        )��P	��au���A�*

	conv_loss���<W)x        )��P	� bu���A�*

	conv_loss�Q�<l�T�        )��P	�9bu���A�*

	conv_loss�M=Ŭl�        )��P	�mbu���A�*

	conv_lossP��<��T�        )��P	Z�bu���A�*

	conv_loss�^�<�J��        )��P	K�bu���A�*

	conv_loss ~o<�u}�        )��P	�cu���A�*

	conv_loss�U<3���        )��P	 3cu���A�*

	conv_lossI��<�2@k        )��P	^ecu���A�*

	conv_loss+f_<�>��        )��P	��cu���A�*

	conv_loss��X<�w��        )��P	��cu���A�*

	conv_loss�f�;@C)�        )��P	��cu���A�*

	conv_loss�N�<~h#�        )��P	,du���A�*

	conv_lossPha<il�        )��P	�[du���A�*

	conv_loss��=4�{        )��P	x�du���A�*

	conv_loss���<���        )��P	��du���A�*

	conv_lossX!t<�w�        )��P	[�du���A�*

	conv_loss06]<�cѫ        )��P	�eu���A�*

	conv_loss���<��Hq        )��P	�Leu���A�*

	conv_loss*k<�$�,        )��P	y�eu���A�*

	conv_loss0ʴ<��E        )��P	��eu���A�*

	conv_loss{�=o��w        )��P	��eu���A�*

	conv_loss?��<Ml        )��P	ufu���A�*

	conv_loss��<��        )��P	�Afu���A�*

	conv_loss��;�:�        )��P	4qfu���A�*

	conv_loss�n�<c@ƛ        )��P	��fu���A�*

	conv_loss�N<�R�        )��P	2�fu���A�*

	conv_lossK�<&$5e        )��P	]gu���A�*

	conv_lossp\�<�ƽ�        )��P	�5gu���A�*

	conv_loss)·<���        )��P	�egu���A�*

	conv_loss~�1<ڶ�	        )��P	��gu���A�*

	conv_loss�w<���        )��P	��gu���A�*

	conv_lossk��<!���        )��P	�	hu���A�*

	conv_loss[�<�P�        )��P	g<hu���A�*

	conv_lossI�Y<M�<�        )��P	�phu���A�*

	conv_loss��<����        )��P	��hu���A�*

	conv_loss��F<R�        )��P	"�hu���A�*

	conv_lossC��<���d        )��P	Qiu���A�*

	conv_loss���<��        )��P	W2iu���A�*

	conv_loss1�<���        )��P	piu���A�*

	conv_loss�۽<J��a        )��P	6�iu���A�*

	conv_loss0W)=�K��        )��P	��iu���A�*

	conv_loss;"<s�,        )��P	5ju���A�*

	conv_loss�k�<�S/�        )��P	mLju���A�*

	conv_loss�<~~�        )��P	?�ju���A�*

	conv_lossp(_=�g��        )��P	��ju���A�*

	conv_loss��<���x        )��P	��ju���A�*

	conv_lossX�<� ��        )��P	Sku���A�*

	conv_loss65�<����        )��P	�Bku���A�*

	conv_loss@�V<a��        )��P	tsku���A�*

	conv_lossr�<���        )��P	��ku���A�*

	conv_loss�y�<���        )��P	��ku���A�*

	conv_loss��<�!�        )��P	�	lu���A�*

	conv_loss�<�yE!        )��P	7:lu���A�*

	conv_loss�Z<D0v        )��P	�klu���A�*

	conv_lossU�w<'�۟        )��P	��lu���A�*

	conv_losserj<|��A        )��P	��lu���A�*

	conv_lossϚ;��C�        )��P	d�lu���A�*

	conv_loss��<;�r        )��P	�0mu���A�*

	conv_loss�߷<[<�        )��P	�amu���A�*

	conv_loss��W<&#��        )��P	ǒmu���A�*

	conv_loss7?N<q.�Y        )��P	��mu���A�*

	conv_loss��6=\6�        )��P	$�mu���A�*

	conv_loss���<<wO�        )��P	&nu���A�*

	conv_lossBi�<�̜�        )��P	-Xnu���A�*

	conv_loss��1<�	f        )��P	��nu���A�*

	conv_lossF��<�[��        )��P	t�nu���A�*

	conv_lossW��<IK˧        )��P	��nu���A�*

	conv_lossLm�<����        )��P	qou���A�*

	conv_loss�΂<!���        )��P	qMou���A�*

	conv_loss��<+"z5        )��P	�}ou���A�*

	conv_loss�T�<c�        )��P	S�ou���A�*

	conv_loss��<�l��        )��P	L�ou���A�*

	conv_lossw�<c�b�        )��P	pu���A�*

	conv_loss�<� �g        )��P	�>pu���A�*

	conv_loss{��<Q8�?        )��P	�mpu���A�*

	conv_loss��u<f��        )��P	�pu���A�*

	conv_loss�nk<���        )��P	��pu���A�*

	conv_loss��%<[$�c        )��P	� qu���A�*

	conv_loss�©</��        )��P	��uu���A�*

	conv_lossQW<|��        )��P	�uu���A�*

	conv_loss���;��U        )��P	>�uu���A�*

	conv_loss�p�<���?        )��P	�,vu���A�*

	conv_loss2du<�!b<        )��P	�[vu���A�*

	conv_loss�ZQ<w�z        )��P	��vu���A�*

	conv_losse��<�/�u        )��P	P�vu���A�*

	conv_loss��<܍~        )��P	��vu���A�*

	conv_loss�<,��        )��P	�+wu���A�*

	conv_lossUi�<�}�d        )��P	�`wu���A�*

	conv_lossTϵ<	�I        )��P	n�wu���A�*

	conv_loss���< +        )��P	V�wu���A�*

	conv_loss?<B>�/        )��P	%xu���A�*

	conv_loss���<�֬�        )��P	!1xu���A�*

	conv_loss��P<>�Vk        )��P	�^xu���A�*

	conv_loss���<r���        )��P	�xu���A�*

	conv_loss��<<� Dd        )��P	��xu���A�*

	conv_loss��<�N�        )��P	N�xu���A�*

	conv_loss��<8üN        )��P	
.yu���A�*

	conv_loss
hV<Pc��        )��P	_yu���A�*

	conv_lossw^<�E,�        )��P	�yu���A�*

	conv_loss#�k<vQ��        )��P	��yu���A�*

	conv_lossn�<V���        )��P	��yu���A�*

	conv_loss��<�ە!        )��P	lzu���A�*

	conv_lossAQ�<:�+9        )��P	�Nzu���A�*

	conv_loss�^<N*wa        )��P	��zu���A�*

	conv_loss�q<��c        )��P	��zu���A�*

	conv_lossՁ�<u�n�        )��P	��zu���A�*

	conv_lossǤ�<hu~�        )��P	7{u���A�*

	conv_loss}J�;�� �        )��P	>B{u���A�*

	conv_lossk<�*�[        )��P	�r{u���A�*

	conv_loss[�<���;        )��P	
�{u���A�*

	conv_loss��G<��        )��P	F�{u���A�*

	conv_lossq}�<3�f        )��P	w|u���A�*

	conv_lossk<��w        )��P	)3|u���A�*

	conv_loss�ئ<R�E�        )��P	�b|u���A�*

	conv_loss��_<L�B�        )��P	]�|u���A�*

	conv_lossM�y<1(�
        )��P	��|u���A�*

	conv_loss���<��߅        )��P	��|u���A�*

	conv_lossW�K<؇�        )��P	n#}u���A�*

	conv_loss���<��J�        )��P	�V}u���A�*

	conv_loss6,$<�}�        )��P	��}u���A�*

	conv_loss+��<��Ⱥ        )��P	�}u���A�*

	conv_loss�<��';        )��P	?�}u���A�*

	conv_losse]�<��7        )��P	!~u���A�*

	conv_loss#Z�<jh��        )��P	pG~u���A�*

	conv_loss���<8���        )��P	Dy~u���A�*

	conv_loss�?�<@�e�        )��P	{�~u���A�*

	conv_loss�0<j���        )��P	`�~u���A�*

	conv_loss&̖<�]�         )��P	au���A�*

	conv_loss��z<b��^        )��P	<u���A�*

	conv_loss��<d�}�        )��P	��u���A�*

	conv_loss#�<y�;6        )��P	̴u���A�*

	conv_loss(�C<��        )��P	��u���A�*

	conv_loss�\<�sCV        )��P	��u���A�*

	conv_loss�G<
4�        )��P	�L�u���A�*

	conv_loss��<��vr        )��P	o��u���A�*

	conv_loss�� <�yu�        )��P	��u���A�*

	conv_loss���;L@�|        )��P	�u���A�*

	conv_losst��;m>�        )��P	�0�u���A�*

	conv_loss}�<'Z�        )��P	Pd�u���A�*

	conv_lossOU�<��f�        )��P	
��u���A�*

	conv_loss�ϻ<t�        )��P	j́u���A�*

	conv_loss&S�<@X�        )��P	[��u���A�*

	conv_loss�l<@9�?        )��P	�=�u���A�*

	conv_lossv��<�_0        )��P	So�u���A�*

	conv_loss�$�<��W+        )��P	��u���A�*

	conv_loss4$<n6�|        )��P	�ςu���A�*

	conv_loss���;�+��        )��P	e�u���A�*

	conv_loss�0"<{��        )��P	L<�u���A�*

	conv_lossJ�R<�4        )��P	�o�u���A�*

	conv_loss�5�;΀aT        )��P	���u���A�*

	conv_lossƵ<�#<        )��P	w΃u���A�*

	conv_lossCX�<��        )��P	���u���A�*

	conv_loss���<�~�        )��P	�-�u���A�*

	conv_loss�w<A��*        )��P	_�u���A�*

	conv_lossX۴<�4h�        )��P	���u���A�*

	conv_loss"v|<�ن(        )��P	e��u���A�*

	conv_loss]�<�@y        )��P	 �u���A�*

	conv_loss�<6�Q        )��P	��u���A�*

	conv_lossS.�<��`+        )��P	�I�u���A�*

	conv_lossu�<�W"�        )��P	�z�u���A�*

	conv_loss��{<��?�        )��P	���u���A�*

	conv_loss'�<�_`�        )��P	�مu���A�*

	conv_loss"�n<�%�;        )��P	R�u���A�*

	conv_lossv<T<.�g�        )��P	�8�u���A�*

	conv_loss^ �;�3��        )��P	5h�u���A�*

	conv_loss"�<��p        )��P	2��u���A�*

	conv_loss�=W<���w        )��P	�ņu���A�*

	conv_lossC��;�EA        )��P	��u���A�*

	conv_loss��<���        )��P	U#�u���A�*

	conv_loss1�<���        )��P	�S�u���A�*

	conv_loss���;I{d�        )��P	��u���A�*

	conv_loss�z`<
�0�        )��P	���u���A�*

	conv_loss��;���>        )��P	��u���A�*

	conv_lossY�Z<�� �        )��P	e�u���A�*

	conv_loss���<dry        )��P	�H�u���A�*

	conv_lossd�;\�Q�        )��P	�x�u���A�*

	conv_loss�I�;�e<        )��P	S��u���A�*

	conv_loss�e_<�FL!        )��P	*Ոu���A�*

	conv_loss��<�l        )��P	��u���A�*

	conv_loss��<�w=        )��P	���u���A�*

	conv_loss�]T<��Rw        )��P	�Ŋu���A�*

	conv_loss�'J<��|        )��P	X��u���A�*

	conv_loss�KE<��        )��P	A$�u���A�*

	conv_loss�v^<�fh�        )��P	�Z�u���A�*

	conv_loss�0V<��l<        )��P	���u���A�*

	conv_loss�$k<]$&Z        )��P	p��u���A�*

	conv_loss�u.<y���        )��P	��u���A�*

	conv_loss�qe<��        )��P	n�u���A�*

	conv_loss���<� ��        )��P	�N�u���A�*

	conv_loss;�=�Ng�        )��P	���u���A�*

	conv_loss���< ؆�        )��P	�u���A�*

	conv_lossG�G<f�(�        )��P	��u���A�*

	conv_loss^U�<��C        )��P	��u���A�*

	conv_loss���</�        )��P	�T�u���A�*

	conv_loss���<rN�=        )��P	��u���A�*

	conv_loss��k<��        )��P	��u���A�*

	conv_loss./�;�Y�        )��P	
�u���A�*

	conv_loss��7<񈡋        )��P	��u���A�*

	conv_loss�9�<��_        )��P	�E�u���A�*

	conv_loss��=S,�        )��P	ru�u���A�*

	conv_loss���;��,        )��P	⧎u���A�*

	conv_lossZ�a< ß        )��P	�ގu���A�*

	conv_lossGVw<;8.        )��P	b�u���A�*

	conv_loss;�<���        )��P	K<�u���A�*

	conv_loss2=<��`        )��P	�i�u���A�*

	conv_loss��;U�f�        )��P	A��u���A�*

	conv_loss� <���        )��P	�ߏu���A�*

	conv_loss�8�<{��        )��P	��u���A�*

	conv_loss���<ҡb        )��P	WB�u���A�*

	conv_loss Z�<on�z        )��P	r�u���A�*

	conv_lossh{<��        )��P	_��u���A�*

	conv_loss"e�<\�P5        )��P	�Ӑu���A�*

	conv_loss���<�*ȫ        )��P	U�u���A�*

	conv_loss�,<�,�        )��P	�3�u���A�*

	conv_loss6mP<����        )��P	!b�u���A�*

	conv_loss�8�;J��S        )��P	���u���A�*

	conv_lossH��<���        )��P	���u���A�*

	conv_loss��k<,U��        )��P	��u���A�*

	conv_loss̗=<;*25        )��P	E"�u���A�*

	conv_losstg�<��w�        )��P	�P�u���A�*

	conv_loss&�<�!�        )��P	倒u���A�*

	conv_lossk�S<�/<(        )��P	&��u���A�*

	conv_loss� <e$7�        )��P	?�u���A�*

	conv_loss���<��p        )��P	��u���A�*

	conv_loss��<T�Z~        )��P	7>�u���A�*

	conv_lossb<H*        )��P	�l�u���A�*

	conv_lossx��<ir$~        )��P	g��u���A�*

	conv_loss�,y<��j        )��P	͓u���A�*

	conv_loss�j<w�3�        )��P	g��u���A�*

	conv_loss�8<7�_B        )��P	�:�u���A�*

	conv_loss(�g<<���        )��P	i�u���A�*

	conv_loss�+2<�_        )��P	M��u���A�*

	conv_lossi��<�z��        )��P	jȔu���A�*

	conv_loss=<<u��u        )��P	f�u���A�*

	conv_loss<�(<��6�        )��P	�=�u���A�*

	conv_loss�ж<�[�        )��P	#p�u���A�*

	conv_loss�pQ<}~��        )��P	r��u���A�*

	conv_lossόr<��        )��P	�ޕu���A�*

	conv_loss�n<ƙ��        )��P	��u���A�*

	conv_loss|�H<Eh��        )��P	?A�u���A�*

	conv_loss<���R        )��P	p�u���A�*

	conv_loss�_�<��        )��P	P��u���A�*

	conv_lossa�v<��        )��P	�ߖu���A�*

	conv_lossQ
<���        )��P	H�u���A�*

	conv_lossH�(<E��        )��P	$M�u���A�*

	conv_lossy�<DX~        )��P	9�u���A�*

	conv_loss`o�<y�uC        )��P	a��u���A�*

	conv_loss��<��V�        )��P	�ݗu���A�*

	conv_loss�<���b        )��P	S�u���A�*

	conv_loss��<��e        )��P	L=�u���A�*

	conv_loss$j�<� ��        )��P		l�u���A�*

	conv_lossD�\< �<        )��P	h��u���A�*

	conv_lossf'<0�GL        )��P	�Θu���A�*

	conv_loss�P<���q        )��P	w��u���A�*

	conv_loss"�J<dK�{        )��P	�*�u���A�*

	conv_lossK͆<z�>        )��P	�\�u���A�*

	conv_loss]w�<J@�        )��P	�u���A�*

	conv_loss1��<|�J�        )��P	༙u���A�*

	conv_loss��><5�fc        )��P	��u���A�*

	conv_loss �5<-��        )��P	��u���A�*

	conv_loss��o<����        )��P	S�u���A�*

	conv_loss&M�<��O        )��P	Q��u���A�*

	conv_loss��=��        )��P	���u���A�*

	conv_lossC��<$![�        )��P	��u���A�*

	conv_loss�QO<�̚�        )��P	e�u���A�*

	conv_lossK��;��th        )��P	TB�u���A�*

	conv_loss?�<���        )��P	mq�u���A�*

	conv_loss�k<��)�        )��P	���u���A�*

	conv_loss?�;�+�        )��P	�ћu���A�*

	conv_lossí�;-�Y�        )��P	��u���A�*

	conv_loss�$�<��L        )��P	33�u���A�*

	conv_loss�԰<zfʠ        )��P	c�u���A�*

	conv_loss)�<�&��        )��P	���u���A�*

	conv_lossh��<�[�~        )��P	�Ĝu���A�*

	conv_loss_�<�8��        )��P	O��u���A�*

	conv_losseC�<0c�        )��P	�'�u���A�*

	conv_loss��m<�X�u        )��P	[�u���A�*

	conv_loss�P�<>)A�        )��P	���u���A�*

	conv_loss�%<k���        )��P	���u���A�*

	conv_loss���<OiJ�        )��P	���u���A�*

	conv_loss�G<�ϒ&        )��P	�-�u���A�*

	conv_loss��<��a6        )��P	$]�u���A�*

	conv_loss*��<D�        )��P	���u���A�*

	conv_lossOX�<�3@        )��P	�Þu���A�*

	conv_loss8�<6(�<        )��P	.�u���A�*

	conv_loss��<R�sK        )��P	�,�u���A�*

	conv_loss+%�;bct        )��P	�[�u���A�*

	conv_lossU�+<���        )��P	���u���A�*

	conv_loss̒�;^Yp        )��P	 ȟu���A�*

	conv_lossFw<��̓        )��P	/��u���A�*

	conv_loss���;� �        )��P	�/�u���A�*

	conv_loss`;�<���        )��P	�b�u���A�*

	conv_lossAa�<�[�[        )��P	���u���A�*

	conv_loss�܆<����        )��P	Ǡu���A�*

	conv_lossAE�<��^�        )��P	���u���A�*

	conv_lossm�<P�.�        )��P	F,�u���A�*

	conv_loss��<<.        )��P	�a�u���A�*

	conv_lossS�%<��        )��P	m��u���A�*

	conv_loss�]W<��d�        )��P	�ѡu���A�*

	conv_loss�0=.I�L        )��P	F�u���A�*

	conv_loss*v�<x�!        )��P	&5�u���A�*

	conv_loss�P;<���        )��P	 h�u���A�*

	conv_loss��<x�)+        )��P	㚢u���A�*

	conv_loss�o�;��        )��P	H΢u���A�*

	conv_loss�V_<�9v        )��P	�u���A�*

	conv_loss��<���        )��P	3�u���A�*

	conv_loss�9u<AH>        )��P	:e�u���A�*

	conv_loss�<{J^`        )��P	Q��u���A�*

	conv_loss=`u<��H        )��P	{ʣu���A�*

	conv_loss�j�<����        )��P	a��u���A�*

	conv_loss�%s<z�o�        )��P	P,�u���A�*

	conv_loss)�< |�        )��P	�^�u���A�*

	conv_lossե;<�&        )��P	(��u���A�*

	conv_loss��< 9�        )��P	�Ĥu���A�*

	conv_loss7�<�2ͮ        )��P	���u���A�*

	conv_loss.�;�ŭ�        )��P	t&�u���A�*

	conv_lossxGr<���        )��P	�W�u���A�*

	conv_lossA��<�:`        )��P	x��u���A�*

	conv_loss$b�<f��J        )��P	��u���A�*

	conv_losss�0<��@        )��P	��u���A�*

	conv_loss���<���2        )��P	�$�u���A�*

	conv_loss>� <��        )��P	8V�u���A�*

	conv_loss��;R!��        )��P	6��u���A�*

	conv_loss���<��        )��P	ַ�u���A�*

	conv_loss.%<�s\        )��P	3�u���A�*

	conv_lossAx�<_I�        )��P	��u���A�*

	conv_lossO��;)���        )��P	�O�u���A�*

	conv_loss���<b���        )��P	E��u���A�*

	conv_lossc<$�O�        )��P	���u���A�*

	conv_loss��h<3H�        )��P	I��u���A�*

	conv_loss' <�U}Y        )��P	�.�u���A�*

	conv_lossZ��<��        )��P	�_�u���A�*

	conv_loss ��;_��        )��P	���u���A�*

	conv_loss5�l<����        )��P	�Ȩu���A�*

	conv_loss���<�4jj        )��P	���u���A�*

	conv_loss���;��
        )��P	?/�u���A�*

	conv_loss�9r<�        )��P	�^�u���A�*

	conv_loss&&<�{z        )��P	˗�u���A�*

	conv_loss�<*M        )��P	�˩u���A�*

	conv_loss�w�<���        )��P	m��u���A�*

	conv_loss�[=�w0H        )��P	�0�u���A�*

	conv_loss���<�T        )��P	�a�u���A�*

	conv_loss5�J<�'�        )��P	t��u���A�*

	conv_loss��x<Υ��        )��P	˪u���A�*

	conv_lossx�<ec{i        )��P	L��u���A�*

	conv_lossB��<�TB        )��P	�0�u���A�*

	conv_loss�(�<�S�        )��P	�f�u���A�*

	conv_lossIHr<��^K        )��P	���u���A�*

	conv_loss��d< Zt        )��P	�֫u���A�*

	conv_lossX�e<K24        )��P	��u���A�*

	conv_loss?,�<��&        )��P	�:�u���A�*

	conv_loss#g<"���        )��P	�l�u���A�*

	conv_loss��;�=~u        )��P	���u���A�*

	conv_loss�e<�HzU        )��P	�άu���A�*

	conv_lossA��<���        )��P	&��u���A�*

	conv_lossB��;2��        )��P	�0�u���A�*

	conv_loss��\<v�q:        )��P	�`�u���A�*

	conv_loss�q<;),#        )��P	"��u���A�*

	conv_loss/�<hrIg        )��P	�ĭu���A�*

	conv_lossQ=t�˥        )��P	|��u���A�*

	conv_loss�:�<��        )��P	+�u���A�*

	conv_loss��<�*ީ        )��P	
_�u���A�*

	conv_loss�#�<�⥙        )��P	���u���A�*

	conv_lossag=^,=m        )��P	�®u���A�*

	conv_loss �<���g        )��P	��u���A�*

	conv_loss��<�9�T        )��P	�'�u���A�*

	conv_lossP�<?��s        )��P	�X�u���A�*

	conv_losssyy<i�m�        )��P	�u���A�*

	conv_loss�X�<@���        )��P	A��u���A�*

	conv_loss�HU<�2��        )��P	��u���A�*

	conv_lossC��<���o        )��P	��u���A�*

	conv_loss?*<C�s�        )��P	�K�u���A�*

	conv_loss+:�<Z?�J        )��P	��u���A�*

	conv_lossZ��<=��1        )��P	(��u���A�*

	conv_loss�=f<}8uK        )��P	��u���A�*

	conv_lossɞ<b�s�        )��P	��u���A�*

	conv_loss�jd<���        )��P	�I�u���A�*

	conv_loss��<���        )��P	M{�u���A�*

	conv_lossP2�<��
        )��P	�u���A�*

	conv_loss"��<2�V�        )��P	e@�u���A�*

	conv_loss���;@W        )��P	�r�u���A�*

	conv_lossZ�T<X��g        )��P	p��u���A�*

	conv_loss��<:�b        )��P	�ڳu���A�*

	conv_loss"=�<�ɵ        )��P	��u���A�*

	conv_loss[�<�i'        )��P	�=�u���A�*

	conv_loss��o<��ʹ        )��P	�p�u���A�*

	conv_lossm�<w��        )��P	B��u���A�*

	conv_loss�ph<���        )��P	I޴u���A�*

	conv_loss�ԃ<
1I         )��P	��u���A�*

	conv_loss�'t</��        )��P	nG�u���A�*

	conv_loss�� <����        )��P	��u���A�*

	conv_loss��J<ā!        )��P	���u���A�*

	conv_loss�^Z<u@        )��P	@�u���A�*

	conv_losso�<�#�H        )��P	K(�u���A�*

	conv_loss\�<Dg	9        )��P	Y�u���A�*

	conv_loss֍f<��d�        )��P	Έ�u���A�*

	conv_loss�W�<�Rx�        )��P	e��u���A�*

	conv_loss�}<��.        )��P	�u���A�*

	conv_loss�=��&t        )��P	!�u���A�*

	conv_loss�Н<�?|�        )��P	�Q�u���A�*

	conv_loss��\<Z��5        )��P	��u���A�*

	conv_loss��T<yd�n        )��P	C��u���A�*

	conv_loss��==��        )��P	��u���A�*

	conv_loss�8<�N)�        )��P	��u���A�*

	conv_lossU��;��5        )��P	�I�u���A�*

	conv_lossnf<0��        )��P	�u���A�*

	conv_loss�J�<�V*        )��P	���u���A�*

	conv_lossxW�<eƕ�        )��P	o�u���A�*

	conv_loss�=�i�l        )��P	��u���A�*

	conv_loss#�4<� �w        )��P	�H�u���A�*

	conv_loss(K<m7g�        )��P	({�u���A�*

	conv_loss�8�<m�Uh        )��P	i��u���A�*

	conv_loss%<�N=�        )��P	�ݹu���A�*

	conv_lossl��<�?�>        )��P	K�u���A�*

	conv_loss��=ր�        )��P	�B�u���A�*

	conv_lossed<0_�        )��P	�w�u���A�*

	conv_loss�g<�R��        )��P	D��u���A�*

	conv_loss!��<���        )��P	ۺu���A�*

	conv_lossV�=8/�        )��P	��u���A�*

	conv_loss�b�<��q�        )��P	�A�u���A�*

	conv_lossA �<�sޫ        )��P	8z�u���A�*

	conv_loss�W�<����        )��P	��u���A�*

	conv_loss�K�<Ȧ��        )��P	ݻu���A�*

	conv_loss�g�<�1��        )��P	��u���A�*

	conv_loss�<h��X        )��P	�M�u���A�*

	conv_lossX�;��?�        )��P	�~�u���A�*

	conv_loss�t�< /��        )��P	ڰ�u���A�*

	conv_loss�=F���        )��P	Q�u���A�*

	conv_loss�$<��^�        )��P	b�u���A�*

	conv_lossG��<h�G�        )��P	�i�u���A�*

	conv_lossԵ�<$���        )��P	���u���A�*

	conv_lossaB<<0�+�        )��P	*˽u���A�*

	conv_loss��<AF�x        )��P	���u���A�*

	conv_loss��=òu        )��P	�/�u���A�*

	conv_loss.�6<(!$7        )��P	�a�u���A�*

	conv_loss�q�<T���        )��P	��u���A�*

	conv_loss;�1=�f>�        )��P	�žu���A�*

	conv_loss���<:27        )��P	�u���A�*

	conv_loss��<J�&4        )��P	EE�u���A�*

	conv_lossf�<���        )��P	��u���A�*

	conv_lossɤp<��y�        )��P	���u���A�*

	conv_loss���<���        )��P	x�u���A�*

	conv_loss;�.<��3�        )��P	O1�u���A�*

	conv_loss`g�<��8        )��P	�a�u���A�*

	conv_lossL�<�1oq        )��P	���u���A�*

	conv_loss�0�<�rQ        )��P	���u���A�*

	conv_losse�I<�~g        )��P	��u���A�*

	conv_lossw��<��v        )��P	�,�u���A�*

	conv_loss��<�k�        )��P	�]�u���A�*

	conv_loss̇<����        )��P	S��u���A�*

	conv_lossW�n<��j        )��P	��u���A�*

	conv_loss%�<Zzd        )��P	���u���A�*

	conv_loss�n<���^        )��P	�#�u���A�*

	conv_loss�:�<+<��        )��P	;V�u���A�*

	conv_loss�R2<Z�-T        )��P		��u���A�*

	conv_loss���<����        )��P	���u���A�*

	conv_loss�<9U�2        )��P	���u���A�*

	conv_loss/ۣ<��R        )��P	q�u���A�*

	conv_loss�m�<!B=        )��P	qL�u���A�*

	conv_loss"�"<p=f�        )��P	�}�u���A�*

	conv_loss��;�(q        )��P	��u���A�*

	conv_loss��<���        )��P	���u���A�*

	conv_loss�9<�ڼ        )��P	�u���A�*

	conv_lossc*w<��+�        )��P	�C�u���A�*

	conv_loss.+
=�:@A        )��P	nv�u���A�*

	conv_loss��<��`P        )��P	��u���A�*

	conv_loss�S�<��"F        )��P	��u���A�*

	conv_loss�zB<?Y�        )��P	e	�u���A�*

	conv_loss\��<F�_�        )��P	^:�u���A�*

	conv_loss�:�<Z;g�        )��P	ck�u���A�*

	conv_lossʝ�<�H��        )��P	���u���A�*

	conv_loss�mH<7��        )��P	u��u���A�*

	conv_loss�ȃ<��r        )��P	= �u���A�*

	conv_loss�Ȓ;rE�        )��P	[2�u���A�*

	conv_loss�QO<AS��        )��P	�b�u���A�*

	conv_lossO��;���        )��P	ە�u���A�*

	conv_loss��=<_�&�        )��P	w��u���A�*

	conv_loss�x<��ײ        )��P	���u���A�*

	conv_loss�[i<+:�x        )��P	�+�u���A�*

	conv_lossl�<�_/N        )��P	�p�u���A�*

	conv_lossJ�<�T(�        )��P	a��u���A�*

	conv_loss���<�n��        )��P	��u���A�*

	conv_loss��J<�Cl�        )��P	�u���A�*

	conv_loss*�<�TtC        )��P	Q?�u���A�*

	conv_lossV<R        )��P	|�u���A�*

	conv_lossB��<��k�        )��P	��u���A�*

	conv_lossVt�;����        )��P	���u���A�*

	conv_losstyV<�
��        )��P	5�u���A�*

	conv_lossC�<D���        )��P	�K�u���A�*

	conv_lossw�a<�R79        )��P	U��u���A�*

	conv_loss�'�<����        )��P	7��u���A�*

	conv_loss��G<�`Q�        )��P	 �u���A�*

	conv_loss�1</��        )��P	 2�u���A�*

	conv_loss��<����        )��P	�a�u���A�*

	conv_loss�<*W�$        )��P	��u���A�*

	conv_loss	�r<jg��        )��P	���u���A�*

	conv_lossk%�;S�ok        )��P	���u���A�*

	conv_loss���;�Ln        )��P	W%�u���A�*

	conv_loss��!<�~��        )��P	\V�u���A�*

	conv_loss�g�<�F��        )��P	S��u���A�*

	conv_lossgN�<��        )��P	���u���A�*

	conv_loss�1�;y�Ɵ        )��P	���u���A�*

	conv_loss��<�G��        )��P	�u���A�*

	conv_loss��;	-��        )��P	�H�u���A�*

	conv_loss^�<�ݽ�        )��P	 y�u���A�*

	conv_loss��<         )��P	��u���A�*

	conv_lossB{�<�i��        )��P	��u���A�*

	conv_lossl�B<��Ҫ        )��P	��u���A�*

	conv_loss/��<��Y	        )��P	@�u���A�*

	conv_loss*�Y<�߳C        )��P	�q�u���A�*

	conv_loss�DO<\�ޥ        )��P	��u���A�*

	conv_loss�: =�L�        )��P	m��u���A�*

	conv_loss���<:��        )��P	F
�u���A�*

	conv_loss�e<�<��        )��P	x<�u���A�*

	conv_loss~�J;M�Hr        )��P	mn�u���A�*

	conv_loss�O<�(0<        )��P	���u���A�*

	conv_loss�L<3��        )��P	��u���A�*

	conv_loss��<�$y�        )��P	0�u���A�*

	conv_losshT�<֮�|        )��P	�7�u���A�*

	conv_lossa:�<�Pd�        )��P	�h�u���A�*

	conv_loss�Xx<y+q�        )��P	i��u���A�*

	conv_loss�^�<;"w�        )��P	$��u���A�*

	conv_loss(QI<:)j        )��P	���u���A�*

	conv_lossD�;�#        )��P	�,�u���A�*

	conv_loss�r+<63@}        )��P	�^�u���A�*

	conv_lossX�;<��'        )��P	���u���A�*

	conv_loss3 <:V��        )��P	߿�u���A�*

	conv_loss�_�;Z6�        )��P	���u���A�*

	conv_loss�0< ��#        )��P	�"�u���A�*

	conv_lossBe�;����        )��P	Bg�u���A�*

	conv_lossh x<+���        )��P	d��u���A�*

	conv_loss�o<\ԣ�        )��P	���u���A�*

	conv_loss)��<�Op�        )��P	���u���A�*

	conv_loss�<�<���F        )��P	�0�u���A�*

	conv_lossݙK<�t0        )��P	^d�u���A�*

	conv_loss]��<�o5�        )��P	��u���A�*

	conv_loss�Zx<��W�        )��P	t��u���A�*

	conv_loss�,�<,>��        )��P	�
�u���A�*

	conv_lossX��;��~        )��P	C@�u���A�*

	conv_loss�<�$�i        )��P	Yt�u���A�*

	conv_loss�&=1���        )��P	c��u���A�*

	conv_lossM??=��xK        )��P	k��u���A�*

	conv_loss���<����        )��P	k%�u���A�*

	conv_loss�n�<�p��        )��P	_Y�u���A�*

	conv_loss| <�,�        )��P	ً�u���A�*

	conv_loss�Q�<z'��        )��P	r��u���A�*

	conv_loss��<w)g�        )��P	���u���A�*

	conv_lossw�<ȅ��        )��P	�%�u���A�*

	conv_loss�XX<R>B�        )��P	W�u���A�*

	conv_lossk��<O�sL        )��P	g��u���A�*

	conv_loss�<�Y�        )��P	Ż�u���A�*

	conv_loss��<�sm�        )��P	T��u���A�*

	conv_loss���<�6W�        )��P	y$�u���A�*

	conv_loss���<M��z        )��P	�X�u���A�*

	conv_loss|�q<���E        )��P	���u���A�*

	conv_loss���<����        )��P	���u���A�*

	conv_lossS��<&�}        )��P	.��u���A�*

	conv_loss�@�;�~)j        )��P	P(�u���A�*

	conv_lossӞ;<�I,        )��P	�Z�u���A�*

	conv_lossJ	<���\        )��P	���u���A�*

	conv_lossG�<kM"z        )��P	���u���A�*

	conv_loss��E<�a3�        )��P	u��u���A�*

	conv_loss?r�;��        )��P	]'�u���A�*

	conv_lossV�:<���        )��P	�X�u���A�*

	conv_loss�b�<��U        )��P	_��u���A�*

	conv_loss�<`�S}        )��P	˽�u���A�*

	conv_lossc�</g�        )��P	^��u���A�*

	conv_loss�&<Z3�*        )��P	r%�u���A�*

	conv_loss <�^3�        )��P	Z�u���A�*

	conv_loss��-<�A�        )��P	Ì�u���A�*

	conv_loss�5�<�9T        )��P	Ͽ�u���A�*

	conv_loss��<^�         )��P	U��u���A�*

	conv_loss5�=�v�        )��P	6&�u���A�*

	conv_lossj]�<Nq*�        )��P	�X�u���A�*

	conv_lossW�]<q2��        )��P	ˌ�u���A�*

	conv_loss�0�<Э>        )��P	��u���A�*

	conv_lossR:<r�gF        )��P	'��u���A�*

	conv_loss��`<�I�        )��P	&�u���A�*

	conv_loss9Z�<�A�        )��P	U��u���A�*

	conv_lossK�s<+��,        )��P	 }�u���A�*

	conv_loss3<t�]�        )��P	9��u���A�*

	conv_lossB�<E��        )��P	���u���A�*

	conv_loss+ѷ;J�F�        )��P	�u���A�*

	conv_loss٥<-���        )��P	 7�u���A�*

	conv_loss�m�<Xh&�        )��P	?h�u���A�*

	conv_lossރ�<���        )��P	��u���A�*

	conv_loss�[l<ˤ�*        )��P	���u���A�*

	conv_losst%�;�b        )��P	���u���A�*

	conv_lossU�L<���A        )��P	52�u���A�*

	conv_loss�XF;A��[        )��P	�`�u���A�*

	conv_loss`�v< C        )��P	U��u���A�*

	conv_lossz��<���        )��P	���u���A�*

	conv_loss�Ob<?�ZG        )��P	u��u���A�*

	conv_loss=�|<�ٸP        )��P	��u���A�*

	conv_loss�Y	<��J�        )��P	�N�u���A�*

	conv_loss?L<T@X	        )��P	\|�u���A�*

	conv_losseÁ<<��        )��P	��u���A�*

	conv_lossuT<T�[n        )��P	O��u���A�*

	conv_loss���<*k��        )��P	��u���A�*

	conv_lossG"><�"        )��P	(B�u���A�*

	conv_loss?��<c��%        )��P	wx�u���A�*

	conv_loss���<z�S�        )��P	=��u���A�*

	conv_loss?�/<�#�        )��P	Y��u���A�*

	conv_lossU�<|Z��        )��P	Z�u���A�*

	conv_loss�Q]<�w��        )��P	03�u���A�*

	conv_loss�6�<yes        )��P	5`�u���A�*

	conv_loss�z�<ͧ�I        )��P	F��u���A�*

	conv_loss�dN<�I�.        )��P	��u���A�*

	conv_lossV<���        )��P	���u���A�*

	conv_loss̼<� �        )��P	#�u���A�*

	conv_loss��o<����        )��P	�S�u���A�*

	conv_losslh}<RU��        )��P	r��u���A�*

	conv_lossU�r<�n��        )��P	���u���A�*

	conv_loss�ݿ;H�3�        )��P	9��u���A�*

	conv_loss��<Dm�        )��P	_�u���A�*

	conv_lossO�<�}^�        )��P	D�u���A�*

	conv_loss��8<����        )��P	�r�u���A�*

	conv_loss$9�<�sT        )��P	K��u���A�*

	conv_loss��?<w��2        )��P	���u���A�*

	conv_loss�\<3*xY        )��P	I��u���A�*

	conv_lossm<��y        )��P	2+�u���A�*

	conv_lossq�}<�N��        )��P	'Y�u���A�*

	conv_lossk�<p�G        )��P	̉�u���A�*

	conv_losss�.<Bv�        )��P	��u���A�*

	conv_loss>Kd<���        )��P	���u���A�*

	conv_loss��<22�        )��P	��u���A�*

	conv_lossJ�'<����        )��P	�F�u���A�*

	conv_loss��<���        )��P	%v�u���A�*

	conv_loss$
9<^�        )��P	���u���A�*

	conv_loss�';<n�        )��P	J��u���A�*

	conv_lossA�`<����        )��P	�u���A�*

	conv_loss��4<Gyh;        )��P	H�u���A�*

	conv_loss5n&<56�        )��P	x�u���A�*

	conv_loss��<DV�7        )��P	���u���A�*

	conv_loss��<2A�,        )��P	z��u���A�*

	conv_loss��;N���        )��P	��u���A�*

	conv_loss͎�;���        )��P	G?�u���A�*

	conv_losss�d<d        )��P	+o�u���A�*

	conv_loss�=�<⦨E        )��P	���u���A�*

	conv_loss�m�;?�S        )��P	���u���A�*

	conv_lossP�k<���        )��P	� �u���A�*

	conv_loss�]�<y��        )��P	�U�u���A�*

	conv_lossJ��;w0U�        )��P	���u���A�*

	conv_loss"�D<+#��        )��P	H��u���A�*

	conv_loss�G(<����        )��P	���u���A�*

	conv_loss��<�\~�        )��P	�-�u���A�*

	conv_loss��<����        )��P	�^�u���A�*

	conv_lossw��<��W        )��P	j��u���A�*

	conv_loss�C�<%y�        )��P	���u���A�*

	conv_loss7�;/��j        )��P	���u���A�*

	conv_lossyuZ<�X�@        )��P	-#�u���A�*

	conv_loss�G<�p��        )��P	�\�u���A�*

	conv_loss�
<�(�        )��P	 ��u���A�*

	conv_lossz}r<�R_        )��P	v��u���A�*

	conv_loss�w;�¡'        )��P	6��u���A�*

	conv_loss�m<��45        )��P	�*�u���A�*

	conv_loss��<}%L�        )��P	<\�u���A�*

	conv_loss��<��*B        )��P	U��u���A�*

	conv_loss��;�,D�        )��P	��u���A�*

	conv_loss-~^<�E�        )��P	~��u���A�*

	conv_loss��%<�         )��P	
!�u���A�*

	conv_loss�]�<�)�        )��P	T�u���A�*

	conv_loss�x<@7�l        )��P	˃�u���A�*

	conv_loss�=�0�        )��P	���u���A�*

	conv_loss!�<q���        )��P	��u���A�*

	conv_lossL&�<U��        )��P	a�u���A�*

	conv_loss��2<"��        )��P	,E�u���A�*

	conv_loss=ʘ;�;'        )��P	u�u���A�*

	conv_loss�_a<�Z�	        )��P	���u���A�*

	conv_loss}<��mJ        )��P	-��u���A�*

	conv_loss��;l��        )��P	�u���A�*

	conv_loss	\�<茓�        )��P	XK�u���A�*

	conv_loss!�<���X        )��P	I{�u���A�*

	conv_lossR<���        )��P	֬�u���A�*

	conv_loss�l�<
�e        )��P	8��u���A�*

	conv_loss�W<T�`�        )��P	��u���A�*

	conv_loss�j<G��-        )��P	C�u���A�*

	conv_loss���;��-l        )��P	#s�u���A�*

	conv_loss�4�<)9�G        )��P	��u���A�*

	conv_loss�+<��~        )��P	.��u���A�*

	conv_lossN<�7�        )��P	��u���A�*

	conv_loss"�<a��        )��P	�G�u���A�*

	conv_loss�5<����        )��P	�|�u���A�*

	conv_loss�LD<p� �        )��P	,��u���A�*

	conv_lossypq<W)        )��P	���u���A�*

	conv_loss��l<���        )��P	�u���A�*

	conv_loss��%=�T0        )��P	gO�u���A�*

	conv_loss��<=A=_        )��P	��u���A�*

	conv_loss�}D<��1b        )��P	u��u���A�*

	conv_lossbf�<f���        )��P	1�u���A�*

	conv_loss�<�&        )��P	�<�u���A�*

	conv_loss��B<ۂ6        )��P	"p�u���A�*

	conv_loss��z<�}RM        )��P	#��u���A�*

	conv_loss庁<%o�        )��P	q��u���A�*

	conv_lossqu<w;s�        )��P	X�u���A�*

	conv_loss>�=�Y2]        )��P	�8�u���A�*

	conv_lossE�3<���a        )��P	ei�u���A�*

	conv_loss/�;���'        )��P	ʙ�u���A�*

	conv_loss��&<l�A        )��P	U��u���A�*

	conv_loss��<%*JM        )��P	���u���A�*

	conv_loss0$�<'�.        )��P	S1�u���A�*

	conv_loss��<LZϨ        )��P	Ud�u���A�*

	conv_loss8�<d9+9        )��P	���u���A�*

	conv_loss��<�a�k        )��P		��u���A�*

	conv_loss.m|<N%d�        )��P	/��u���A�*

	conv_loss/�<�]��        )��P	N+�u���A�*

	conv_lossꃍ<G�.        )��P	b\�u���A�*

	conv_lossSю<���        )��P	���u���A�*

	conv_loss��x<���c        )��P	4��u���A�*

	conv_loss�m8<��P        )��P	���u���A�*

	conv_loss-�	<�$        )��P	e#�u���A�*

	conv_loss���<�g        )��P	�S�u���A�*

	conv_lossP></��        )��P	���u���A�*

	conv_loss�<�]!�        )��P	���u���A�*

	conv_loss�
<e��        )��P	��u���A�*

	conv_lossr%\<�(	        )��P	c�u���A�*

	conv_loss"��<wfx9        )��P	xO�u���A�*

	conv_loss�e�<d�]        )��P	��u���A�*

	conv_losstz�<�%�x        )��P	-��u���A�*

	conv_loss@\H;��f@        )��P	��u���A�*

	conv_loss�.�<ŝY        )��P	}�u���A�*

	conv_loss!�;��C        )��P	�M�u���A�*

	conv_loss��E<c��        )��P	e��u���A�*

	conv_loss�o<�b9        )��P	���u���A�*

	conv_lossz�e<�ؔ        )��P	%��u���A�*

	conv_loss
��<�R!         )��P	��u���A�*

	conv_loss�|�<��<        )��P	�J�u���A�*

	conv_lossQYb<�s�        )��P	N~�u���A�*

	conv_lossA�1<m��        )��P	Q��u���A�*

	conv_lossĳ�;q޴%        )��P	�u���A�*

	conv_loss��<�2��        )��P	�6�u���A�*

	conv_loss �D<YG?        )��P	�h�u���A�*

	conv_loss��M<�~C0        )��P	˝�u���A�*

	conv_loss�z�<�?        )��P	J��u���A�*

	conv_loss���<��        )��P	�
 v���A�*

	conv_loss�*<�l�        )��P	r> v���A�*

	conv_loss���;��
        )��P	�p v���A�*

	conv_loss���;�
{        )��P	0� v���A�*

	conv_loss��R<�S-<        )��P	�� v���A�*

	conv_lossp��;D�        )��P	�v���A�*

	conv_loss� a<�Ȏ�        )��P	?Rv���A�*

	conv_lossu�;��n�        )��P	��v���A�*

	conv_losse|!<��y�        )��P	��v���A�*

	conv_lossz<�_�J        )��P	��v���A�*

	conv_loss��<�G��        )��P	�v���A�*

	conv_loss�ۉ<��˟        )��P	~Lv���A�*

	conv_loss�r<���        )��P	�~v���A�*

	conv_loss�B=<�)�b        )��P	�v���A�*

	conv_loss	"/<���        )��P	`�v���A�*

	conv_loss��<��k        )��P	�v���A�*

	conv_loss��;o        )��P	�Iv���A�*

	conv_loss��;��b�        )��P	�{v���A�*

	conv_loss|�G;�
Qn        )��P	��v���A�*

	conv_loss~��<D�v�        )��P	^�v���A�*

	conv_loss�+�<qtAc        )��P	�v���A�*

	conv_loss1�;���        )��P	gEv���A�*

	conv_loss��T<Oy��        )��P	hxv���A�*

	conv_loss��<iኼ        )��P	��v���A�*

	conv_loss��~<���        )��P	w�v���A�*

	conv_loss�#=��	e        )��P	qv���A�*

	conv_lossa>q<z�̒        )��P	�>v���A�*

	conv_lossI%�<J8�        )��P	@rv���A�*

	conv_lossx�X<'�[        )��P	t�v���A�*

	conv_loss��<:�1(        )��P	o�v���A�*

	conv_loss���<_�        )��P	v���A�*

	conv_loss��W<F,Z        )��P	7v���A�*

	conv_loss~v<�@�        )��P	Mjv���A�*

	conv_loss���;8�        )��P	�v���A�*

	conv_loss�3�<�
�        )��P	�v���A�*

	conv_loss&��<|���        )��P	:v���A�*

	conv_loss9�S=*�        )��P	�3v���A�*

	conv_lossT=^���        )��P	fv���A�*

	conv_loss֭�;�-�W        )��P	�v���A�*

	conv_loss��s<��        )��P	�v���A�*

	conv_lossKMB<�`fW        )��P	��v���A�*

	conv_loss#<v�*�        )��P	�0v���A�*

	conv_loss���;�J(�        )��P	zhv���A�*

	conv_loss�}�;�u�(        )��P	c�v���A�*

	conv_lossK/<����        )��P	U�v���A�*

	conv_loss�	-<�t��        )��P	s
v���A�*

	conv_loss��<j���        )��P	��
v���A�*

	conv_loss.�@;!S��        )��P	��
v���A�*

	conv_loss@L<>��h        )��P	v���A�*

	conv_loss��h;�5�A        )��P	�=v���A�*

	conv_loss�q<��@        )��P		|v���A�*

	conv_lossY�E<�RSn        )��P	�v���A�*

	conv_loss$�<)��        )��P	��v���A�*

	conv_loss�^]<D�U	        )��P	�v���A�*

	conv_loss4�T<�(��        )��P	*Fv���A�*

	conv_loss�<a��I        )��P	�v���A�*

	conv_loss��K<����        )��P	��v���A�*

	conv_loss:x�<@�;A        )��P	f�v���A�*

	conv_loss��[<��        )��P	 v���A�*

	conv_loss��<�HG�        )��P	�Fv���A�*

	conv_loss��<�a�L        )��P	P�v���A�*

	conv_loss"$�<0m\        )��P	��v���A�*

	conv_lossh�D<�2��        )��P	G�v���A�*

	conv_loss��<��D�        )��P	$0v���A�*

	conv_loss�a<H��        )��P	Vdv���A�*

	conv_loss�#6<zwE         )��P	B�v���A�*

	conv_loss���;���        )��P	�v���A�*

	conv_lossԡb<�?�h        )��P	1�v���A�*

	conv_lossQZ+<0��        )��P	�1v���A�*

	conv_loss�^<s        )��P	^bv���A�*

	conv_loss��<�%��        )��P	$�v���A�*

	conv_loss}p<	d��        )��P	5�v���A�*

	conv_loss��X<���        )��P	m�v���A�*

	conv_lossJM=�o�y        )��P	'.v���A�*

	conv_loss���;��        )��P	�av���A�*

	conv_loss־�<�An        )��P	��v���A�*

	conv_loss�;��u        )��P	-�v���A�*

	conv_lossOQ�<Cd?~        )��P	tv���A�*

	conv_lossnf?<�|��        )��P	�<v���A�*

	conv_loss��I<d�n        )��P	@ov���A�*

	conv_loss5w�<�~8_        )��P	��v���A�*

	conv_loss!�<=k        )��P	v�v���A�*

	conv_lossg`<�菬        )��P	�v���A�*

	conv_loss��*<@>'        )��P	�?v���A�*

	conv_loss<�X<Ck�        )��P	�tv���A�*

	conv_loss�_i<�U�4        )��P	�v���A�*

	conv_loss�6�<4�B        )��P	+�v���A�*

	conv_loss�o�<;߃�        )��P	�v���A�*

	conv_lossr�h<��~�        )��P	Jv���A�*

	conv_loss���<��פ        )��P	{v���A�*

	conv_lossy��<���        )��P	S�v���A�*

	conv_loss��<2�U�        )��P	9�v���A�*

	conv_lossv�7<��6        )��P	�v���A�*

	conv_lossb�Q<I<,�        )��P	�@v���A�*

	conv_loss۴<�!`�        )��P	|rv���A�*

	conv_loss��<)>/�        )��P		�v���A�*

	conv_loss-�<�M�l        )��P	��v���A�*

	conv_lossW�<�z�l        )��P	�v���A�*

	conv_lossɞ�<��3�        )��P	kHv���A�*

	conv_loss�3k<���b        )��P	��v���A�*

	conv_loss m�<�t{        )��P	��v���A�*

	conv_loss��<۬�r        )��P	�v���A�*

	conv_loss��A<���        )��P	+Hv���A�*

	conv_losseb�<�v"        )��P	�v���A�*

	conv_losst�O<M�        )��P	N�v���A�*

	conv_lossc2�<��o        )��P	��v���A�*

	conv_loss���;Y���        )��P	�0v���A�*

	conv_loss��g<W�1�        )��P	�fv���A�*

	conv_loss�t`<���        )��P	N�v���A�*

	conv_loss�<��\4        )��P	k�v���A�*

	conv_loss���;�9��        )��P	t#v���A�*

	conv_losss��<�j�Z        )��P	�Wv���A�*

	conv_lossOe~<�1��        )��P	#�v���A�*

	conv_loss��<`tq�        )��P	F�v���A�*

	conv_loss0�9<#9��        )��P	��v���A�*

	conv_lossUJ�<�e�        )��P	_v���A�*

	conv_loss���<�D        )��P	xLv���A�*

	conv_loss�ǎ<|��        )��P	�~v���A�*

	conv_lossA�5<F�%7        )��P	U�v���A�*

	conv_loss	'�;Rh7�        )��P	��v���A�*

	conv_loss��I<>��         )��P		v���A�*

	conv_lossލp<tǑ�        )��P	�Dv���A�*

	conv_lossn �;��B/        )��P	�~v���A�*

	conv_lossq�;0�w�        )��P	�v���A�*

	conv_loss��;��=        )��P	.�v���A�*

	conv_lossӴP<���        )��P	�v���A�*

	conv_loss�l�<J        )��P	�^v���A�*

	conv_lossm�:<�`��        )��P	.�v���A�*

	conv_loss�>H<pzJ�        )��P	��v���A�*

	conv_loss��><�;]�        )��P	;�v���A�*

	conv_lossOW�;�˅[        )��P	)v���A�*

	conv_loss�<��fF        )��P	�]v���A�*

	conv_loss���;�+"�        )��P	��v���A�*

	conv_lossv�;Xe�        )��P	U�v���A�*

	conv_loss�p<����        )��P	��v���A�*

	conv_loss�PZ;���V        )��P	�-v���A�*

	conv_lossS��<PV.�        )��P	�av���A�*

	conv_loss�'H<=��G        )��P	�v���A�*

	conv_loss�
�<za}        )��P	2�v���A�*

	conv_loss���<�/�        )��P	Z�v���A�*

	conv_loss B�;��a        )��P	�1v���A�*

	conv_loss4P<�`�        )��P	 gv���A�*

	conv_loss���<���        )��P	ܜv���A�*

	conv_loss\�<xĹ�        )��P	!�v���A�*

	conv_loss�I<�~�        )��P	v���A�*

	conv_loss;h;<Z���        )��P	E\v���A�*

	conv_loss1=vc�        )��P	'�v���A�*

	conv_loss~U�;?x΀        )��P	��v���A�*

	conv_lossI>w<$�"        )��P	��v���A�*

	conv_loss�s�< +�K        )��P	%4 v���A�*

	conv_loss'�_<���2        )��P	<e v���A�*

	conv_loss5��<��c2        )��P	�� v���A�*

	conv_loss�ɍ<���        )��P	k� v���A�*

	conv_lossޚ�<;�        )��P	�� v���A�*

	conv_loss�v�<N�Y        )��P	�9!v���A�*

	conv_loss�h�;�GT6        )��P	�l!v���A�*

	conv_lossg�;�F�        )��P	1�!v���A�*

	conv_loss?�<����        )��P	��!v���A�*

	conv_loss�˒<5�8        )��P	h"v���A�*

	conv_loss�l(<�::b        )��P	�Z"v���A�*

	conv_loss�cR<M���        )��P	&�"v���A�*

	conv_lossy��;���8        )��P	i�"v���A�*

	conv_loss�M<[~�        )��P	k�"v���A�*

	conv_loss�2q<��%�        )��P	�,#v���A�*

	conv_loss�}�<���G        )��P	nd#v���A�*

	conv_loss���<�8�j        )��P	]�#v���A�*

	conv_loss��<-G��        )��P	��#v���A�*

	conv_lossڥ(<��7�        )��P	�$v���A�*

	conv_loss�L�<�_�        )��P	7$v���A�*

	conv_loss���<Hƭ        )��P	�{$v���A�*

	conv_lossT�2<
�        )��P	s�$v���A�*

	conv_loss�~�;��        )��P	��$v���A�*

	conv_loss+��<T��        )��P	w%v���A�*

	conv_loss]x�;�H��        )��P	/Q%v���A�*

	conv_losss�7<m}�:        )��P	=�%v���A�*

	conv_loss,�<�ē7        )��P	��%v���A�*

	conv_loss��<k���        )��P	��%v���A�*

	conv_loss�x�;�?b�        )��P	�*&v���A�*

	conv_loss�"�<��W�        )��P	5a&v���A�*

	conv_loss2%<2�n        )��P	>�&v���A�*

	conv_loss�l<����        )��P	��&v���A�*

	conv_loss	^x<G/e�        )��P	�'v���A�*

	conv_loss���;ď�        )��P	�N'v���A�*

	conv_loss�y�;��y        )��P	�'v���A�*

	conv_lossH<��\�        )��P	�'v���A�*

	conv_loss�<�;XU5�        )��P	1�'v���A�*

	conv_loss��;׉B        )��P	81(v���A�*

	conv_loss��<�A�        )��P	�g(v���A�*

	conv_loss`N<vMH�        )��P	�(v���A�*

	conv_loss.<燶�        )��P	��(v���A�*

	conv_loss�Zk<�Y�A        )��P	�)v���A�*

	conv_loss_�	<���        )��P	�C)v���A�*

	conv_loss«i<���        )��P	f})v���A�*

	conv_lossz%=�.�        )��P	��)v���A�*

	conv_losscB]<̦ٗ        )��P	��)v���A�*

	conv_loss�b[<%0t        )��P	�/*v���A�*

	conv_lossc��<k�        )��P	�_*v���A�*

	conv_loss�o<Z19        )��P	�*v���A�*

	conv_lossd�;#�d,        )��P	�*v���A�*

	conv_lossݝI;T�Q�        )��P		+v���A�*

	conv_loss:��<=�        )��P	�@+v���A�*

	conv_loss�X
<o��z        )��P	�q+v���A�*

	conv_loss�g<pʓ        )��P	�+v���A�*

	conv_lossqp<�W�#        )��P	��+v���A�*

	conv_loss��L<��s7        )��P	�,v���A�*

	conv_loss;� <l��@        )��P	�V,v���A�*

	conv_loss@;�;���[        )��P	)�,v���A�*

	conv_loss���;��w        )��P	W�,v���A�*

	conv_lossY��;M=3        )��P	��,v���A�*

	conv_lossk��<����        )��P	�--v���A�*

	conv_loss?��;���        )��P	�g-v���A�*

	conv_lossZs<���        )��P	˙-v���A�*

	conv_loss��<��@4        )��P	<�-v���A�*

	conv_lossA׀<�$��        )��P	9.v���A�*

	conv_lossfw<���        )��P	�=.v���A�*

	conv_loss�<��Y�        )��P	t.v���A�*

	conv_loss�U<�pW        )��P	��.v���A�*

	conv_loss:<ˑ_6        )��P	L�.v���A�*

	conv_lossn.�;���        )��P	7!/v���A�*

	conv_lossa�A<��wD        )��P	>[/v���A�*

	conv_loss��<H��(        )��P	q�/v���A�*

	conv_loss�J-<���F        )��P	4�/v���A�*

	conv_lossYpQ<ϵ�:        )��P	s0v���A�*

	conv_lossB<>�tQ        )��P	�40v���A�*

	conv_loss��<=�'�        )��P	�k0v���A�*

	conv_lossT��;[��        )��P	�0v���A�*

	conv_loss��<@�t�        )��P	j�0v���A�*

	conv_lossBY�;�)X�        )��P	�1v���A�*

	conv_loss��o<� �        )��P	�N1v���A�*

	conv_loss��{<܌��        )��P	��1v���A�*

	conv_lossva]<�pu�        )��P	}�1v���A�*

	conv_loss��n<� s        )��P	��1v���A�*

	conv_loss�,�<56�        )��P	C 2v���A�*

	conv_loss��<��m	        )��P	�T2v���A�*

	conv_loss4\�;ꅃ        )��P	@�2v���A�*

	conv_lossoW�;s�p        )��P	�2v���A�*

	conv_loss��R<ܕ�a        )��P	�2v���A�*

	conv_loss�u�;4��C        )��P	�#3v���A�*

	conv_loss%�
<�bT�        )��P	V3v���A�*

	conv_lossr��<��Ͼ        )��P	8�3v���A�*

	conv_loss�;�~K        )��P	��3v���A�*

	conv_loss�ۓ;&g�        )��P	�4v���A�*

	conv_lossv�
<&��         )��P	n@4v���A�*

	conv_lossP�<����        )��P	�q4v���A�*

	conv_loss�*w<m;        )��P	j�4v���A�*

	conv_loss5J�<�xƸ        )��P	%J6v���A�*

	conv_loss�K�;�V��        )��P	2�6v���A�*

	conv_loss'��;ԛ+0        )��P	��6v���A�*

	conv_loss��U<��T\        )��P	�7v���A�*

	conv_loss���;x��        )��P	�57v���A�*

	conv_lossn��;b�o�        )��P	He7v���A�*

	conv_loss:7�;O�v        )��P	�7v���A�*

	conv_loss��{<&�?        )��P	�7v���A�*

	conv_lossAQ;%�I�        )��P	�8v���A�*

	conv_loss�u<c�m%        )��P	�<8v���A�*

	conv_lossOٗ<;�t�        )��P	Xk8v���A�*

	conv_loss?$�<M�-�        )��P	(�8v���A�*

	conv_loss#�<��1        )��P	9�8v���A�*

	conv_loss�rd<��wT        )��P	�!9v���A�*

	conv_lossh�<�}Dk        )��P	�O9v���A�*

	conv_loss�q<x���        )��P	o~9v���A�*

	conv_loss�U<�L�        )��P	հ9v���A�*

	conv_lossso=R��e        )��P	��9v���A�*

	conv_loss��b<'%�        )��P	:v���A�*

	conv_loss,><�ǹ        )��P	�L:v���A�*

	conv_loss��;I�i�        )��P	�|:v���A�*

	conv_loss��<d<1V        )��P	u�:v���A�*

	conv_loss6W<`=>�        )��P	�:v���A�*

	conv_loss5��<���_        )��P	f;v���A�*

	conv_loss�y�<��S�        )��P	P;v���A�*

	conv_loss�w^<ڞ��        )��P	݂;v���A�*

	conv_loss/Q�<"D[�        )��P	۰;v���A�*

	conv_loss/a<2�         )��P	��;v���A�*

	conv_losseB<�F��        )��P	0<v���A�*

	conv_loss�?<�Og        )��P	?<v���A�*

	conv_loss� M;'�6R        )��P	 m<v���A�*

	conv_lossk�<�g�        )��P	u�<v���A�*

	conv_loss0m�<Ｕ0        )��P	��<v���A�*

	conv_loss���;7[��        )��P	�=v���A�*

	conv_losss��<�        )��P	w3=v���A�*

	conv_loss�S�<*w��        )��P	fa=v���A�*

	conv_loss͆q<qP��        )��P	��=v���A�*

	conv_loss��<g���        )��P	[�=v���A�*

	conv_lossI'�<P(         )��P	��=v���A�*

	conv_loss��<��        )��P	�>v���A�*

	conv_loss_�=�:0X        )��P	S>v���A�*

	conv_loss���<�
)b        )��P	>v���A�*

	conv_loss�J[<���        )��P	s�>v���A�*

	conv_lossH?<�v�{        )��P	�>v���A�*

	conv_lossx1�;���E        )��P	�?v���A�*

	conv_loss�;+<��I        )��P	?@?v���A�*

	conv_loss�f�<���        )��P	=r?v���A�*

	conv_lossDi�<���J        )��P	K�?v���A�*

	conv_loss�0P<ѾCW        )��P	��?v���A�*

	conv_loss�><�(�$        )��P	o@v���A�*

	conv_loss-�T<�d�        )��P	"H@v���A�*

	conv_lossb_b<���        )��P	!~@v���A�*

	conv_loss,2�<0��*        )��P	�@v���A�*

	conv_lossG�<��        )��P	/�@v���A�*

	conv_loss�sv<�r�        )��P	�!Av���A�*

	conv_loss��A<�ܮ�        )��P	GQAv���A�*

	conv_lossj��;Qa��        )��P	��Av���A�*

	conv_loss��<�!�f        )��P	��Av���A�*

	conv_loss�,<�M        )��P	`�Av���A�*

	conv_lossc!�<6RC�        )��P	.Bv���A�*

	conv_loss��y<����        )��P	�^Bv���A�*

	conv_loss�V�<0���        )��P	ߎBv���A�*

	conv_loss�݈<CY[�        )��P	��Bv���A�*

	conv_loss��<�^��        )��P	 �Bv���A�*

	conv_loss�ʵ<k�0        )��P	 4Cv���A�*

	conv_lossu#h<��,�        )��P	�cCv���A�*

	conv_loss��<�R��        )��P	!�Cv���A�*

	conv_lossZ�@<�18        )��P	��Cv���A�*

	conv_loss�K<�9�        )��P	��Cv���A�*

	conv_lossc��<�cb        )��P	�*Dv���A�*

	conv_loss�:R<��v�        )��P	�]Dv���A�*

	conv_loss��u;v!%e        )��P	��Dv���A�*

	conv_loss`�<͵6(        )��P	��Dv���A�*

	conv_loss��<����        )��P	X�Dv���A�*

	conv_loss�S�<�n�        )��P	|#Ev���A�*

	conv_lossȰ;��        )��P	�YEv���A�*

	conv_loss�<�{h*        )��P	q�Ev���A�*

	conv_loss�~�<�o�        )��P	I�Ev���A�*

	conv_loss�+�<Yt�_        )��P	��Ev���A�*

	conv_loss츭<�<��        )��P	TFv���A�*

	conv_lossɡ�<��#        )��P	5NFv���A�*

	conv_loss��"<��e        )��P	JFv���A�*

	conv_losst<(<�{�        )��P	�Fv���A�*

	conv_loss��!<�k{        )��P	��Fv���A�*

	conv_lossC�@<�&��        )��P	�Gv���A�*

	conv_loss%j�;Uz!�        )��P	�AGv���A�*

	conv_loss�E�<g        )��P	
qGv���A�*

	conv_loss��<�!x        )��P	:�Gv���A�*

	conv_loss�H<�Y�        )��P	�Gv���A�*

	conv_loss�b�<�{ey        )��P	�Hv���A�*

	conv_loss���<7㚁        )��P	�6Hv���A�*

	conv_loss�0�<h0�        )��P	�hHv���A�*

	conv_loss�<(�al        )��P	q�Hv���A�*

	conv_lossc2�<Z/6Q        )��P	��Hv���A�*

	conv_losseW�<6!t        )��P	��Hv���A�*

	conv_loss}�<��b�        )��P	�.Iv���A�*

	conv_lossu�e<76��        )��P	^Iv���A�*

	conv_loss��<�w��        )��P	�Iv���A�*

	conv_lossS�<��}        )��P	ɿIv���A�*

	conv_loss�Tm<��l        )��P	DeNv���A�*

	conv_loss��I<���        )��P	V�Nv���A�*

	conv_loss���;~        )��P	�Nv���A�*

	conv_loss,5<<+�a�        )��P	�
Ov���A�*

	conv_loss�{,<�0�        )��P	�>Ov���A�*

	conv_loss�P'<W.3h        )��P	�uOv���A�*

	conv_lossvG2<Yj��        )��P	��Ov���A�*

	conv_lossdHI<��f'        )��P	��Ov���A�*

	conv_loss5)�<"�x_        )��P	)Pv���A�*

	conv_loss|+�<N��        )��P	|4Pv���A�*

	conv_lossm'�;�	�        )��P	IcPv���A�*

	conv_lossS�;{19        )��P	u�Pv���A�*

	conv_loss��v<%ఌ        )��P	�Pv���A�*

	conv_loss%�!<��}�        )��P	:Qv���A�*

	conv_loss���<�ݬ^        )��P	0Qv���A�*

	conv_lossj�<�IK        )��P	<^Qv���A�*

	conv_loss�kw<
8A        )��P	Y�Qv���A�*

	conv_loss�<,�        )��P	ϾQv���A�*

	conv_loss���;�Vb�        )��P	�Qv���A�*

	conv_loss��@<��ʔ        )��P	�!Rv���A�*

	conv_lossn��<_��Z        )��P	7ORv���A�*

	conv_loss���<V�        )��P	�|Rv���A�*

	conv_loss�J�<%ǃ�        )��P	��Rv���A�*

	conv_lossJs0<U>Z        )��P	5�Rv���A�*

	conv_lossU�<�Y-R        )��P	�Sv���A�*

	conv_loss�p8<d�:        )��P	r;Sv���A�*

	conv_lossH��;���        )��P	�iSv���A�*

	conv_loss{�<�e��        )��P	��Sv���A�*

	conv_lossp6�;<
��        )��P	��Sv���A�*

	conv_loss~��<�D�        )��P	r�Sv���A�*

	conv_loss��G<}��        )��P	�'Tv���A�*

	conv_loss�N�<ȚC�        )��P	8UTv���A�*

	conv_lossJ޴;n�S        )��P	ӅTv���A�*

	conv_loss���<{��        )��P	Z�Tv���A�*

	conv_lossz��<gR�        )��P	��Tv���A�*

	conv_lossA>w<�Fe        )��P	�Uv���A�*

	conv_lossc�G<>�ߙ        )��P	�IUv���A�*

	conv_loss�,�<
�ey        )��P	�xUv���A�*

	conv_loss��<��,�        )��P	��Uv���A�*

	conv_loss�R<K�g�        )��P	�Uv���A�*

	conv_loss��<�ą        )��P	�Vv���A�*

	conv_loss0T<�,jH        )��P	�AVv���A�*

	conv_loss�<Kx��        )��P	�rVv���A�*

	conv_lossn��;X��u        )��P	��Vv���A�*

	conv_loss+ǆ<��&        )��P	S�Vv���A�*

	conv_loss�U�<��e�        )��P	Wv���A�*

	conv_loss)vq<M��        )��P	�FWv���A�*

	conv_loss��U<P@�        )��P	�vWv���A�*

	conv_loss���;�n�        )��P	�Wv���A�*

	conv_lossU��;��        )��P	��Wv���A�*

	conv_loss�WH<��p        )��P	�Xv���A�*

	conv_loss�<г8$        )��P	�EXv���A�*

	conv_loss�nj<�K�        )��P	xXv���A�*

	conv_lossz�<D��        )��P	Y�Xv���A�*

	conv_loss,ԁ<��         )��P	�Xv���A�*

	conv_loss���;�R:7        )��P	�Yv���A�*

	conv_loss�A<3�ʗ        )��P	�<Yv���A�*

	conv_loss�;<�$^        )��P	�sYv���A�*

	conv_loss��<��)        )��P	�Yv���A�*

	conv_loss��;nt{        )��P	��Yv���A�*

	conv_loss��<+/!        )��P	�Zv���A�*

	conv_loss���<�UJ        )��P	�KZv���A�*

	conv_loss~u`<Ȼ��        )��P	�yZv���A�*

	conv_loss��;GU��        )��P	�Zv���A�*

	conv_lossT��;1�U        )��P	r�Zv���A�*

	conv_loss�-�;��#        )��P	a[v���A�*

	conv_loss���;�F^�        )��P	QD[v���A�*

	conv_loss�iw<�K@        )��P	r[v���A�*

	conv_loss��=�5�        )��P	&�[v���A�*

	conv_loss	�<�c�T        )��P	��[v���A�*

	conv_loss��=1�i�        )��P	E\v���A�*

	conv_loss�� <㫂�        )��P	�6\v���A�*

	conv_loss�6�;54}�        )��P	�h\v���A�*

	conv_loss1 4<'G��        )��P	�\v���A�*

	conv_loss���;A�U=        )��P	�\v���A�*

	conv_loss�;:�Hk        )��P	��\v���A�*

	conv_lossr3�<�!�
        )��P	5']v���A�*

	conv_lossVC<<�(U�        )��P	�W]v���A�*

	conv_loss*}R<��=�        )��P	�]v���A�*

	conv_loss�Z<�F��        )��P	
�]v���A�*

	conv_lossM�F;;��=        )��P	��]v���A�*

	conv_loss!�b<�"��        )��P	N%^v���A�*

	conv_loss���<R��O        )��P	vT^v���A�*

	conv_loss�Ag<2���        )��P	�^v���A�*

	conv_loss�<0��{        )��P	q�^v���A�*

	conv_loss��3<@20        )��P	��^v���A�*

	conv_loss���;��U        )��P	�_v���A�*

	conv_loss��b<}�2        )��P	n@_v���A�*

	conv_loss1<%s��        )��P	o_v���A�*

	conv_lossj�<E�Q        )��P	�_v���A�*

	conv_loss$@�<)A��        )��P	5�_v���A�*

	conv_loss<
M<��>        )��P	��_v���A�*

	conv_loss�'�;#�a�        )��P	,-`v���A�*

	conv_loss�u<���N        )��P	``v���A�*

	conv_loss�ߛ<t��        )��P	�`v���A�*

	conv_loss���;����        )��P	:�`v���A�*

	conv_lossZ�;7�aw        )��P	F�`v���A�*

	conv_lossG4n<���T        )��P	Xav���A�*

	conv_loss��0<�z�        )��P	�Nav���A�*

	conv_lossRj<��Cy        )��P	J~av���A�*

	conv_loss
��<�eM        )��P	X�av���A�*

	conv_loss;��;e�        )��P	z<cv���A�*

	conv_loss�Q8<�=��        )��P	�kcv���A�*

	conv_loss�C?<WX�        )��P	$�cv���A�*

	conv_loss�]n<㺈�        )��P	��cv���A�*

	conv_loss�a<áA�        )��P	��cv���A�*

	conv_loss���;�b�E        )��P	�/dv���A�*

	conv_loss�o<M��G        )��P	fdv���A�*

	conv_loss�W<����        )��P	�dv���A�*

	conv_loss���<�#k�        )��P	��dv���A�*

	conv_loss��=<��]         )��P	\�dv���A�*

	conv_loss�+C<�2�        )��P	�)ev���A�*

	conv_loss���;���        )��P	zcev���A�*

	conv_losse�7<�g         )��P	��ev���A�*

	conv_loss�*<�V�        )��P	0�ev���A�*

	conv_loss��<�,��        )��P	�ev���A�*

	conv_loss���;����        )��P	�(fv���A�*

	conv_loss�:�<�|�'        )��P	�[fv���A�*

	conv_loss�:<�8�        )��P	��fv���A�*

	conv_losso=<�c.        )��P	��fv���A�*

	conv_loss��;��Z        )��P	��fv���A�*

	conv_loss[ȥ<��f        )��P	�gv���A�*

	conv_loss��@<
Ě        )��P	3Lgv���A�*

	conv_lossA��<M�fg        )��P	�{gv���A�*

	conv_loss1|�;h1        )��P	٫gv���A�*

	conv_lossz��<|��        )��P	�gv���A�*

	conv_lossq	:<y��{        )��P	�	hv���A�*

	conv_loss��{<���.        )��P	}9hv���A�*

	conv_loss�^�<f�l        )��P	slhv���A�*

	conv_loss�8<.��        )��P	w�hv���A�*

	conv_lossX�<<���        )��P	��hv���A�*

	conv_loss"1�<g�yx        )��P	] iv���A�*

	conv_loss���;WF��        )��P	</iv���A�*

	conv_loss���<!��S        )��P	�^iv���A�*

	conv_loss�s{< i�        )��P	�iv���A�*

	conv_lossR�<O�Z        )��P	��iv���A�*

	conv_loss���;J���        )��P	��iv���A�*

	conv_loss&Å;W	�        )��P	�"jv���A�*

	conv_lossٵ2<�ٷ        )��P	�Pjv���A�*

	conv_lossqIz<a�j0        )��P	Wjv���A�*

	conv_loss�)%<O���        )��P	��jv���A�*

	conv_loss!m-<A�[H        )��P	�jv���A�*

	conv_loss^�<%��        )��P	�kv���A�*

	conv_lossR�&<x�S        )��P	�?kv���A�*

	conv_lossA�2<�nT        )��P	5okv���A�*

	conv_losskn�<��        )��P	�kv���A�*

	conv_loss��<p"f        )��P	L�kv���A�*

	conv_loss5{�<@��        )��P	 lv���A�*

	conv_loss�V<2���        )��P	�3lv���A�*

	conv_lossw�<��~�        )��P	}dlv���A�*

	conv_loss{�}<���        )��P	��lv���A�*

	conv_loss�^<p!��        )��P	�lv���A�*

	conv_loss�L�<<7-�        )��P	a	mv���A�*

	conv_loss��a<c�`        )��P	0;mv���A�*

	conv_loss���;�$�        )��P	inmv���A�*

	conv_lossKa�;r�u~        )��P	��mv���A�*

	conv_loss��;	�Y�        )��P	��mv���A�*

	conv_lossC�><�1U$        )��P	ynv���A�*

	conv_loss �!<��        )��P	j6nv���A�*

	conv_loss�v�<�l        )��P	grnv���A�*

	conv_loss���;<��b        )��P	��nv���A�*

	conv_loss�y�;%?$        )��P	��nv���A�*

	conv_loss��O<{E`�        )��P	�ov���A�*

	conv_lossL{�<{���        )��P	�Cov���A�*

	conv_losse+Q;�Փ        )��P	tvov���A�*

	conv_loss2�}<�Bֱ        )��P	O�ov���A�*

	conv_loss��!<"���        )��P	��ov���A�*

	conv_loss	�<�!/�        )��P		$pv���A�*

	conv_loss�<f �<        )��P	MXpv���A�*

	conv_loss�b�;OA��        )��P	��pv���A�*

	conv_lossi�<1YoA        )��P	`�pv���A�*

	conv_loss�<�?�        )��P	��pv���A�*

	conv_lossۦ�;B�a�        )��P	�&qv���A�*

	conv_lossM��;��        )��P		Yqv���A�*

	conv_loss	N^<p�h�        )��P	��qv���A�*

	conv_loss�o><6         )��P	a�qv���A�*

	conv_loss�vt<u��;        )��P	�qv���A�*

	conv_loss6d�;V:        )��P	� rv���A�*

	conv_lossL�<�?        )��P	�Rrv���A�*

	conv_loss��;KQ�        )��P	��rv���A�*

	conv_lossJ0�<S� =        )��P	V�rv���A�*

	conv_loss���;
@�Z        )��P	��rv���A�*

	conv_loss��
;IBK$        )��P	�sv���A�*

	conv_lossn6�:��%        )��P	yGsv���A�*

	conv_lossl�<e檹        )��P	Jxsv���A�*

	conv_loss"�;U���        )��P	X�sv���A�*

	conv_loss2��;Xa�%        )��P	1�sv���A�*

	conv_loss;`5<�ݗ�        )��P	 tv���A�*

	conv_lossX v<���        )��P	Dtv���A�*

	conv_loss���;Of9        )��P	'ytv���A�*

	conv_lossL�H;:���        )��P	K�tv���A�*

	conv_loss���<`��R        )��P	?�tv���A�*

	conv_loss���;�7�        )��P	ruv���A�*

	conv_loss���;`Y�        )��P	�@uv���A�*

	conv_loss�+�;K'�        )��P	�ruv���A�*

	conv_loss0�7<��@        )��P	\�uv���A�*

	conv_loss��<!Ԭ        )��P	��uv���A�*

	conv_loss?��;�w��        )��P	�vv���A�*

	conv_loss0X<�|*        )��P	�7vv���A�*

	conv_loss�2*;�5�        )��P	=kvv���A�*

	conv_lossꂍ<�м#        )��P	ߟvv���A�*

	conv_loss"�H<0\3�        )��P	f�vv���A�*

	conv_loss�_<'�}�        )��P	jwv���A�*

	conv_loss�	<6�"        )��P	`Jwv���A�*

	conv_loss�P�<s�
2        )��P	�}wv���A�*

	conv_loss�t�<�aWO        )��P	��wv���A�*

	conv_loss�)c<�x��        )��P	��wv���A�*

	conv_loss��;�p��        )��P	 xv���A�*

	conv_lossj�%<��<�        )��P	yQxv���A�*

	conv_loss��Q<��D        )��P	�xv���A�*

	conv_loss�<� T~        )��P	�xv���A�*

	conv_lossWY�<< ��        )��P	�yv���A�*

	conv_lossg�;�ȴb        )��P	�?yv���A�*

	conv_loss��y<.�W�        )��P	�pyv���A�*

	conv_loss�(C<o���        )��P	ߡyv���A�*

	conv_loss DG<{��        )��P	��yv���A�*

	conv_loss��0<08�P        )��P	�zv���A�*

	conv_loss�BT<�3O�        )��P	�Bzv���A�*

	conv_loss�<��&        )��P	!tzv���A�*

	conv_loss�<xM4�        )��P	@�zv���A�*

	conv_lossCx[<t���        )��P	��zv���A�*

	conv_lossQ �<���	        )��P	�{v���A�*

	conv_loss��<Əǟ        )��P	�8{v���A�*

	conv_loss"4�;nsR        )��P	�i{v���A�*

	conv_loss�͓<�h�         )��P	��{v���A�*

	conv_loss�<���        )��P	�{v���A�*

	conv_loss7��<��%        )��P	�
|v���A�*

	conv_lossev�;�=d\        )��P	�;|v���A�*

	conv_lossX��<�rM<        )��P	Cn|v���A�*

	conv_loss�QS<c�D        )��P	�|v���A�*

	conv_loss�/[<w:�        )��P	��|v���A�*

	conv_loss�_�;S��        )��P	�}v���A�*

	conv_lossN~�;�u�c        )��P	�?}v���A�*

	conv_loss�!<R+�.        )��P	�o}v���A�*

	conv_lossѶm<1�J        )��P	(�}v���A�*

	conv_loss�<�l�        )��P	P�}v���A�*

	conv_lossa�<	���        )��P	�~v���A�*

	conv_loss+j<�g�        )��P	�4~v���A�*

	conv_loss1X<�V��        )��P	ve~v���A�*

	conv_losstC(<ˬWZ        )��P	��~v���A�*

	conv_loss1Oe<��        )��P	�~v���A�*

	conv_loss��t<V/$y        )��P	�~v���A�*

	conv_lossњ�; �v}        )��P	�-v���A�*

	conv_lossM_�<��a�        )��P	>`v���A�*

	conv_loss'�<��.=        )��P	�v���A�*

	conv_loss(E+<E"        )��P	��v���A�*

	conv_lossf�<��]c        )��P	~�v���A�*

	conv_loss���<��        )��P	�$�v���A�*

	conv_loss���;�H,         )��P	dX�v���A�*

	conv_lossJf�;���q        )��P	䉀v���A�*

	conv_lossHc*<�	�        )��P	a��v���A�*

	conv_lossf�;T�l�        )��P	��v���A�*

	conv_lossD-�<���        )��P	54�v���A�*

	conv_loss�e;<ez�        )��P	2i�v���A�*

	conv_loss�t<JQ�/        )��P	���v���A�*

	conv_loss���;���        )��P	�Ёv���A�*

	conv_loss�3�<U ��        )��P	 �v���A�*

	conv_loss���<R��        )��P	�:�v���A�*

	conv_loss�֨<Ha�3        )��P	7n�v���A�*

	conv_loss��<��]�        )��P	6��v���A�*

	conv_lossE.<�m�        )��P	?��v���A�*

	conv_loss���< �߼        )��P	��v���A�*

	conv_loss���<ʗ        )��P	O�v���A�*

	conv_loss���;����        )��P	��v���A�*

	conv_loss� �<�{��        )��P	 ��v���A�*

	conv_lossڈ�;>�w        )��P	L�v���A�*

	conv_loss�<l���        )��P	��v���A�*

	conv_losst��<z�;        )��P	�L�v���A�*

	conv_loss	(�;�d	        )��P	��v���A�*

	conv_loss�$< 8��        )��P	f��v���A�*

	conv_lossW�;rwÿ        )��P	r�v���A�*

	conv_loss���;V���        )��P	��v���A�*

	conv_lossT�<��        )��P	cK�v���A�*

	conv_loss�1@<l�G�        )��P	C|�v���A�*

	conv_loss���;m@`�        )��P	=��v���A�*

	conv_loss�x�<�}�	        )��P	i݅v���A�*

	conv_loss/�<e��A        )��P	��v���A�*

	conv_lossB8�;{&R        )��P	�B�v���A�*

	conv_lossQ�G<��        )��P	�s�v���A�*

	conv_loss�#0<0~��        )��P	���v���A�*

	conv_loss�U<�9�	        )��P	\؆v���A�*

	conv_loss'�<��XZ        )��P	<	�v���A�*

	conv_lossh,<�&P7        )��P	�:�v���A�*

	conv_loss��<�Q        )��P	�m�v���A�*

	conv_loss��;ZXV�        )��P	���v���A�*

	conv_loss!�x<%�F        )��P	�Їv���A�*

	conv_loss�5<�"�        )��P	��v���A�*

	conv_lossB�<�<��        )��P	�6�v���A�*

	conv_lossv�<k�        )��P	Ph�v���A�*

	conv_loss��j<�b�        )��P	���v���A�*

	conv_losspTj<�G        )��P	�̈v���A�*

	conv_loss0�;?��        )��P	���v���A�*

	conv_loss�É<�jA�        )��P	�4�v���A�*

	conv_lossF�<�N        )��P	�f�v���A�*

	conv_lossQ��;&-��        )��P	���v���A�*

	conv_loss���<�y�j        )��P	�ʉv���A�*

	conv_lossR�$<�%]�        )��P	��v���A�*

	conv_loss��<��(Q        )��P	�+�v���A�*

	conv_loss��5<���e        )��P	a]�v���A�*

	conv_lossQ9�<O�a        )��P	)��v���A�*

	conv_loss�H�<�4�        )��P	濊v���A�*

	conv_lossŸ�;�[�        )��P	W�v���A�*

	conv_loss��<�\�        )��P	��v���A�*

	conv_loss�2�<P��        )��P	2��v���A�*

	conv_loss��+<����        )��P	��v���A�*

	conv_loss0^-=���        )��P	�!�v���A�*

	conv_lossd�\<X���        )��P	'Q�v���A�*

	conv_loss#�<�<H�        )��P	Ɂ�v���A�*

	conv_loss�6<��+<        )��P	㸍v���A�*

	conv_lossA��<�dh�        )��P	��v���A�*

	conv_losso<����        )��P	�)�v���A�*

	conv_loss���<tLG{        )��P	�^�v���A�*

	conv_lossĲ;�s��        )��P	���v���A�*

	conv_loss"�<��h        )��P	(̎v���A�*

	conv_loss��<@^��        )��P	���v���A�*

	conv_loss���<�פ�        )��P	$+�v���A�*

	conv_lossM�[<���E        )��P	�Z�v���A�*

	conv_loss�.�<�({'        )��P	0��v���A�*

	conv_loss�r}<*��R        )��P	߹�v���A�*

	conv_loss�g�<SexN        )��P	��v���A�*

	conv_loss~�Z<U#��        )��P	��v���A�*

	conv_loss�$g<��u        )��P	-K�v���A�*

	conv_lossft�<�9        )��P	�}�v���A�*

	conv_lossiUT<����        )��P	���v���A�*

	conv_loss�jt<�W��        )��P	��v���A�*

	conv_lossT�;+Mn        )��P	��v���A�*

	conv_loss�(}<���        )��P	�B�v���A�*

	conv_loss�2<���        )��P	Yq�v���A�*

	conv_lossě<w���        )��P	���v���A�*

	conv_loss1�.<B��        )��P	ёv���A�*

	conv_lossC8<<F��        )��P	b��v���A�*

	conv_loss�3<��[        )��P	�0�v���A�*

	conv_lossN�$<�        )��P	|^�v���A�*

	conv_loss�ٍ<U�\�        )��P	���v���A�*

	conv_loss��<��6�        )��P	(��v���A�*

	conv_loss�E�<��T6        )��P	��v���A�*

	conv_loss|��;��"        )��P	!�v���A�*

	conv_lossR�<���R        )��P	O�v���A�*

	conv_lossx��<@t~        )��P	���v���A�*

	conv_loss��<�+^�        )��P	���v���A�*

	conv_lossY�<N��        )��P	*�v���A�*

	conv_loss��<�Tx        )��P	|�v���A�*

	conv_loss=<�7f�        )��P	ZC�v���A�*

	conv_loss��Q<��U�        )��P	r�v���A�*

	conv_loss6�M< x�q        )��P	���v���A�*

	conv_lossִ<>}��        )��P	�ߔv���A�*

	conv_loss�y<����        )��P	D�v���A�*

	conv_lossJ�<u�#        )��P	W>�v���A�*

	conv_lossAF�;X*;        )��P	r�v���A�*

	conv_lossY`�<ON�        )��P	R��v���A�*

	conv_loss�;�<����        )��P	ؕv���A�*

	conv_lossYd)<�a}�        )��P	��v���A�*

	conv_loss�ʹ<�k,^        )��P	�L�v���A�*

	conv_loss�ƭ<��%�        )��P	�~�v���A�*

	conv_loss:�N<�hV�        )��P	r��v���A�*

	conv_loss�&�<u��        )��P	��v���A�*

	conv_lossΗ[<� �        )��P	Y�v���A�*

	conv_loss�'�;ǣ�O        )��P	�I�v���A�*

	conv_loss�%A<����        )��P	e}�v���A�*

	conv_loss
!�<-�5>        )��P	��v���A�*

	conv_loss�~<��~�        )��P	��v���A�*

	conv_loss�ij<S��        )��P	a�v���A�*

	conv_loss�/+<��        )��P	#S�v���A�*

	conv_loss��<.Ϯ�        )��P	,��v���A�*

	conv_loss�r�;��        )��P	v���A�*

	conv_loss�C<�)e�        )��P	��v���A�*

	conv_loss�c�;�Qt        )��P	�*�v���A�*

	conv_lossd>"<r��        )��P	$Z�v���A�*

	conv_loss�{.<<lM        )��P	X��v���A�*

	conv_loss��x<��o        )��P	o��v���A�*

	conv_loss���<t:�C        )��P	��v���A�*

	conv_lossP+<�avJ        )��P	j �v���A�*

	conv_lossu��;���}        )��P	�Q�v���A�*

	conv_losse?h<��h        )��P	���v���A�*

	conv_loss��=����        )��P	O��v���A�*

	conv_loss���;���{        )��P	��v���A�*

	conv_loss܅�<j���        )��P	��v���A�*

	conv_lossZ�M<Z�E�        )��P	eK�v���A�*

	conv_loss�D�;�.�4        )��P	 ~�v���A�*

	conv_lossD<T��        )��P	'��v���A�*

	conv_loss.1�<�t/K        )��P	�ݛv���A�*

	conv_loss@{<ځż        )��P	%�v���A�*

	conv_loss\�<�@`�        )��P	�B�v���A�*

	conv_lossp�F<���        )��P	s�v���A�*

	conv_loss�9<�W�B        )��P	;��v���A�*

	conv_loss��C<�o�        )��P	M֜v���A�*

	conv_lossƶf<O��        )��P	��v���A�*

	conv_loss�<[3	�        )��P	�8�v���A�*

	conv_lossa�;��
@        )��P	�j�v���A�*

	conv_lossk/�;�&��        )��P	j��v���A�*

	conv_loss�[s;��s        )��P	�Νv���A�*

	conv_loss�ʐ<c:ab        )��P	���v���A�*

	conv_loss�|m<{�un        )��P	p-�v���A�*

	conv_loss��g<��S�        )��P	�\�v���A�*

	conv_lossZ�<b�        )��P	���v���A�*

	conv_loss��=�η        )��P	#��v���A�*

	conv_losswh�;ty        )��P	9�v���A�*

	conv_loss�Q�;��~�        )��P	>�v���A�*

	conv_loss��;���c        )��P	R�v���A�*

	conv_loss��<�M%        )��P	x��v���A�*

	conv_loss�o�;�w�        )��P	���v���A�*

	conv_lossvs:<<�q        )��P	���v���A�*

	conv_loss_��<��*        )��P	.&�v���A�*

	conv_loss��.<ÖL�        )��P	�V�v���A�*

	conv_loss�H�;�ks�        )��P	0��v���A�*

	conv_loss0<�y+        )��P	h��v���A�*

	conv_losst�0<�І�        )��P	��v���A�*

	conv_loss@A�<�9�U        )��P	�+�v���A�*

	conv_loss�<�ڞ|        )��P	\�v���A�*

	conv_loss�O;�s��        )��P	e��v���A�*

	conv_lossqm<Px��        )��P	�ʡv���A�*

	conv_loss7�f<Q` �        )��P	���v���A�*

	conv_lossJWF<�Q�        )��P	t4�v���A�*

	conv_lossu<����        )��P	�f�v���A�*

	conv_loss"�<l7�]        )��P	��v���A�*

	conv_lossÆ�< ��)        )��P	բv���A�*

	conv_lossƫ�;3�7        )��P	��v���A�*

	conv_loss�)�;P�G        )��P	�8�v���A�*

	conv_loss�I�;�#�        )��P	Pj�v���A�*

	conv_loss��+<�        )��P	L��v���A�*

	conv_lossJ�;���        )��P	�ӣv���A�*

	conv_losswe�;���V        )��P	2�v���A�*

	conv_loss�Ё;m�q        )��P	&F�v���A�*

	conv_lossZ��<l�&G        )��P	Jx�v���A�*

	conv_loss��A;�^�@        )��P	��v���A�*

	conv_loss�~�<*��        )��P	�ڤv���A�*

	conv_loss�9B<�BXe        )��P	��v���A�*

	conv_loss-�/<	��        )��P	yF�v���A�*

	conv_lossߥH</Ud�        )��P	�v�v���A�*

	conv_loss��k<}�Wg        )��P	��v���A�*

	conv_loss7�<>T2:        )��P	�إv���A�*

	conv_loss���;�,�z        )��P	x	�v���A�*

	conv_loss}��<:ٵ        )��P		;�v���A�*

	conv_loss%�}<�5^        )��P	]k�v���A�*

	conv_lossG��<��$        )��P	��v���A�*

	conv_loss`�<����        )��P	�ɦv���A�*

	conv_lossk�G<'�)        )��P	���v���A�*

	conv_loss؂�<��        )��P	�*�v���A�*

	conv_loss=H<n��	        )��P	�[�v���A�*

	conv_loss��h<YEN�        )��P	���v���A�*

	conv_loss���;�C	�        )��P	���v���A�*

	conv_loss<��;�XiZ        )��P	m�v���A�*

	conv_loss�ɝ;��]1        )��P	H!�v���A�*

	conv_loss�<Xhx�        )��P	�Q�v���A�*

	conv_loss�m<D;�C        )��P	h��v���A�*

	conv_loss�+|<9�-^        )��P	���v���A�*

	conv_lossY�E<�0�        )��P	��v���A�*

	conv_loss?�<S���        )��P	��v���A�*

	conv_lossII(<��]F        )��P	\A�v���A�*

	conv_loss��;���D        )��P	s�v���A�*

	conv_lossL��<��̱        )��P	���v���A�*

	conv_loss7�=�|�        )��P	��v���A�*

	conv_loss���<�&n.        )��P	��v���A�*

	conv_loss3�<��        )��P	,H�v���A�*

	conv_lossB��;2��        )��P	�|�v���A�*

	conv_loss��~<b*�        )��P	��v���A�*

	conv_loss��<i�ԟ        )��P	�ߪv���A�*

	conv_lossS�^<h`:�        )��P	;�v���A�*

	conv_loss�I-<�c�j        )��P	dC�v���A�*

	conv_loss�Z�;�E��        )��P	�|�v���A�*

	conv_losst9�;�($d        )��P	t��v���A�*

	conv_loss}�<�25�        )��P	7�v���A�*

	conv_lossp;-w��        )��P	y�v���A�*

	conv_lossq:<`���        )��P	UD�v���A�*

	conv_loss�qT</��        )��P	�|�v���A�*

	conv_lossf�<4�        )��P	ų�v���A�*

	conv_lossEE�<�MX        )��P	��v���A�*

	conv_loss��<���        )��P	g�v���A�*

	conv_lossX =�[�        )��P	_I�v���A�*

	conv_loss�<ua!�        )��P	z�v���A�*

	conv_loss�<<	�        )��P	M��v���A�*

	conv_losse��;�\d�        )��P	@٭v���A�*

	conv_lossU�<I��        )��P	�
�v���A�*

	conv_loss%�<�}u�        )��P	�=�v���A�*

	conv_lossa5�<��3�        )��P	{o�v���A�*

	conv_loss8Y�<qJR�        )��P	���v���A�*

	conv_loss>�<�>*        )��P	'Үv���A�*

	conv_loss;$�8        )��P	<�v���A�*

	conv_loss/x�<��        )��P	�4�v���A�*

	conv_loss�<S�5�        )��P	f�v���A�*

	conv_lossY�;�(,�        )��P	ᖯv���A�*

	conv_loss&��;���        )��P	OƯv���A�*

	conv_loss,i!;|���        )��P	)��v���A�*

	conv_loss��Z<;�8,        )��P	�*�v���A�*

	conv_loss�Ar<
��        )��P	V\�v���A�*

	conv_lossgt<��J        )��P	���v���A�*

	conv_lossL��<��T�        )��P	�v���A�*

	conv_lossG<U^P        )��P	���v���A�*

	conv_loss�w�<!{��        )��P	��v���A�*

	conv_lossef<�}"�        )��P	L�v���A�*

	conv_loss�0<��>�        )��P	�{�v���A�*

	conv_lossGI<��        )��P	j��v���A�*

	conv_loss[Y:<Q	��        )��P	Sݱv���A�*

	conv_loss�3!<(�{        )��P	-�v���A�*

	conv_loss���<RŅo        )��P	�=�v���A�*

	conv_loss��<1/^        )��P	o�v���A�*

	conv_lossU�<����        )��P	���v���A�*

	conv_loss��;ԕ�V        )��P	�ײv���A�*

	conv_loss��g<`ރ�        )��P	v�v���A�*

	conv_loss�YZ<���        )��P	�<�v���A�*

	conv_loss�M<�~�        )��P	��v���A�*

	conv_loss0�@<����        )��P	~�v���A�*

	conv_loss1�<Ɣ��        )��P	���v���A�*

	conv_loss�*<�� Z        )��P	$ܹv���A�*

	conv_loss�\<=��        )��P	d�v���A�*

	conv_lossi)�<AУ�        )��P	^@�v���A�*

	conv_loss��<��_        )��P	�p�v���A�*

	conv_loss��[<�	'�        )��P	4��v���A�*

	conv_loss_Z�;/��        )��P	[ݺv���A�*

	conv_lossۃ;���        )��P	��v���A�*

	conv_loss���;��X�        )��P	$J�v���A�*

	conv_loss �z<ٵ,�        )��P	.y�v���A�*

	conv_loss�� <�!2        )��P	��v���A�*

	conv_loss�|?<��O�        )��P	��v���A�*

	conv_loss|4�;�<��        )��P	7�v���A�*

	conv_loss~3< %�        )��P	�L�v���A�*

	conv_loss���<�Z��        )��P	�{�v���A�*

	conv_loss��K<�u}�        )��P	�v���A�*

	conv_loss�%�<��З        )��P	�v���A�*

	conv_lossFe�;�x�/        )��P	��v���A�*

	conv_loss��<,g�S        )��P	�@�v���A�*

	conv_loss{�<ZŎ�        )��P	1o�v���A�*

	conv_loss��;�9]        )��P	❽v���A�*

	conv_losscm<A�2�        )��P	�νv���A�*

	conv_lossc�
<o��h        )��P	��v���A�*

	conv_loss�	<���        )��P	�-�v���A�*

	conv_loss���<p"�        )��P	�\�v���A�*

	conv_loss��<Il1�        )��P	ފ�v���A�*

	conv_loss8s�;lZ�        )��P	���v���A�*

	conv_losss�><��>�        )��P	��v���A�*

	conv_loss�|J<X���        )��P	l�v���A�*

	conv_loss0W�;�2��        )��P	�C�v���A�*

	conv_loss)�<B��y        )��P	�s�v���A�*

	conv_loss���; dhq        )��P	X��v���A�*

	conv_losszf<�M��        )��P	SϿv���A�*

	conv_lossد<�b��        )��P	��v���A�*

	conv_loss��;z�xj        )��P	�.�v���A�*

	conv_loss�U�;��Jw        )��P	�]�v���A�*

	conv_loss9�</3\        )��P	���v���A�*

	conv_lossB�;Y���        )��P	���v���A�*

	conv_lossV�;��x        )��P	���v���A�*

	conv_loss�7�;�qP        )��P	��v���A�*

	conv_loss�Et;G��4        )��P	�I�v���A�*

	conv_losse��;p(M,        )��P	>y�v���A�*

	conv_loss�į<sv�*        )��P	��v���A�*

	conv_lossM	�;�%�
        )��P	Y��v���A�*

	conv_loss ��;�jS        )��P	o�v���A�*

	conv_loss+8�;.���        )��P	'5�v���A�*

	conv_loss�%@<DrC        )��P	;g�v���A�*

	conv_loss�6�;7��        )��P	Q��v���A�*

	conv_lossc�;���        )��P	v��v���A�*

	conv_loss�Ѫ<)��`        )��P	;�v���A�*

	conv_loss�'�;�8�}        )��P	4�v���A�*

	conv_loss-��<Qzz%        )��P	�c�v���A�*

	conv_lossNf�<�|1S        )��P	��v���A�*

	conv_loss�<y�)        )��P	���v���A�*

	conv_lossr��<O^<�        )��P	z��v���A�*

	conv_loss뵩<�V"�        )��P	 )�v���A�*

	conv_lossO7I<*?�e        )��P	Y�v���A�*

	conv_lossX�u;�m�        )��P	]��v���A�*

	conv_loss)J<�Y        )��P	���v���A�*

	conv_loss��<�g&        )��P	���v���A�*

	conv_loss̎:<s�)�        )��P	"�v���A�*

	conv_loss���<zXո        )��P	4R�v���A�*

	conv_lossݓ�<ݛ(P        )��P	�~�v���A�*

	conv_lossp<�ߚq        )��P	���v���A�*

	conv_lossnw<���        )��P	9��v���A�*

	conv_lossZ��;��        )��P	��v���A�*

	conv_loss{Z�<���        )��P	-O�v���A�*

	conv_loss�F�;D�t^        )��P	(~�v���A�*

	conv_loss��;�Z?�        )��P	���v���A�*

	conv_loss�<㗊        )��P	���v���A�*

	conv_loss?rv<��        )��P	D�v���A�*

	conv_loss��<�        )��P	�B�v���A�*

	conv_loss]3;���        )��P	p�v���A�*

	conv_loss�u4<B�Π        )��P	!��v���A�*

	conv_loss��;M�t        )��P	,��v���A�*

	conv_lossc�<t��        )��P	 �v���A�*

	conv_loss�K;I#߶        )��P	\0�v���A�*

	conv_lossP);���        )��P	�d�v���A�*

	conv_loss�<��        )��P	���v���A�*

	conv_loss���;�.n�        )��P	���v���A�*

	conv_loss�<	�@�        )��P	���v���A�*

	conv_loss7X�<_,$�        )��P	7�v���A�*

	conv_lossi�z<�0�^        )��P	Pf�v���A�*

	conv_loss/Tn<9Q�]        )��P	��v���A�*

	conv_losst��;VHS�        )��P	R��v���A�*

	conv_loss�
<n��        )��P	���v���A�*

	conv_lossdH<�q�=        )��P	d"�v���A�*

	conv_loss�y�;l�ҧ        )��P	�O�v���A�*

	conv_loss�h�;HY�_        )��P	1�v���A�*

	conv_loss�i<<:��        )��P	 ��v���A�*

	conv_loss�x�;k        )��P	���v���A�*

	conv_lossun�;-j��        )��P	�"�v���A�*

	conv_loss�+;�h/�        )��P	:S�v���A�*

	conv_loss;<��e&        )��P	S��v���A�*

	conv_loss��_;�@�        )��P	ӱ�v���A� *

	conv_loss-��<m� .        )��P	O��v���A� *

	conv_lossCn<�N��        )��P	��v���A� *

	conv_loss*H`<�w��        )��P	�=�v���A� *

	conv_loss/�]<�d��        )��P	�m�v���A� *

	conv_loss�e�;��o        )��P	���v���A� *

	conv_loss?j�;V-P�        )��P	k��v���A� *

	conv_lossԀ<���W        )��P	N�v���A� *

	conv_lossZr=� �<        )��P	�?�v���A� *

	conv_loss�}<�        )��P	�s�v���A� *

	conv_loss��<Eϼ�        )��P	h��v���A� *

	conv_loss��<�T�C        )��P	��v���A� *

	conv_loss�<-&�i        )��P	}�v���A� *

	conv_loss �3<�s        )��P	E5�v���A� *

	conv_lossNs�;nK�        )��P	wu�v���A� *

	conv_lossѤ<�CW        )��P	(��v���A� *

	conv_loss���;���{        )��P	��v���A� *

	conv_loss��	<qL�        )��P	%�v���A� *

	conv_loss�('<�[��        )��P	1�v���A� *

	conv_loss��X<��	        )��P	�j�v���A� *

	conv_lossۇi<ZW��        )��P	��v���A� *

	conv_loss�	�;��0        )��P	���v���A� *

	conv_lossa�k<�:"        )��P	�v���A� *

	conv_loss��H<'� �        )��P	2�v���A� *

	conv_loss���<d�τ        )��P	_d�v���A� *

	conv_loss�m><<��        )��P	���v���A� *

	conv_loss��m<�~-        )��P	���v���A� *

	conv_loss	�<<��g�        )��P	M��v���A� *

	conv_lossX�<ܛQ�        )��P	�$�v���A� *

	conv_loss{^1<f��S        )��P	�T�v���A� *

	conv_loss��<�OG        )��P	���v���A� *

	conv_lossζ�;��յ        )��P	#��v���A� *

	conv_lossG+<kF�        )��P	R��v���A� *

	conv_loss�t5<O�,        )��P	��v���A� *

	conv_loss�l�;���V        )��P	�:�v���A� *

	conv_loss���;S���        )��P	4i�v���A� *

	conv_lossd�<���        )��P	/��v���A� *

	conv_loss�RV;y�6        )��P	0��v���A� *

	conv_loss$6<DR        )��P	��v���A� *

	conv_loss"��<G/m        )��P	!�v���A� *

	conv_loss�Ҏ<�n�        )��P	(R�v���A� *

	conv_loss��<�VA�        )��P	 ��v���A� *

	conv_loss6��<Č        )��P	F��v���A� *

	conv_loss:�<����        )��P	���v���A� *

	conv_lossp��;�")        )��P	 �v���A� *

	conv_loss��;<���        )��P	�?�v���A� *

	conv_lossN�<��O        )��P	~n�v���A� *

	conv_loss5��<�E��        )��P	G��v���A� *

	conv_loss(9<)mD�        )��P	���v���A� *

	conv_loss"Q<�m�1        )��P	���v���A� *

	conv_lossJ�,<e|{        )��P	y-�v���A� *

	conv_lossғ�<	��        )��P	�]�v���A� *

	conv_loss��;#m�        )��P	z��v���A� *

	conv_lossy��<(���        )��P	i��v���A� *

	conv_loss���<l���        )��P	��v���A� *

	conv_loss�Hc<��.        )��P	�7�v���A� *

	conv_lossU�<���        )��P	i�v���A� *

	conv_loss�r�;'�r        )��P	ʙ�v���A� *

	conv_lossgy<�w>        )��P	��v���A� *

	conv_lossJ5�;���        )��P	��v���A� *

	conv_loss<#)<���l        )��P	�5�v���A� *

	conv_lossv��;}�_        )��P	}q�v���A� *

	conv_loss�<峺�        )��P	E��v���A� *

	conv_lossW��;���        )��P	-��v���A� *

	conv_loss�.<��0�        )��P	c�v���A� *

	conv_lossr�<`˂:        )��P	qF�v���A� *

	conv_loss�4L<#"��        )��P	�y�v���A� *

	conv_loss9��;�.4%        )��P	Ӷ�v���A� *

	conv_lossݷ'=�Y�
        )��P	��v���A� *

	conv_loss,7<<��C�        )��P	��v���A� *

	conv_loss�<���g        )��P	M�v���A� *

	conv_lossh!5<��(.        )��P	�}�v���A� *

	conv_loss/�< /ު        )��P	[��v���A� *

	conv_loss��(<��|        )��P	���v���A� *

	conv_loss�ݝ;�>��        )��P	��v���A� *

	conv_lossHa<'F��        )��P	�P�v���A� *

	conv_loss�٥<q�        )��P	��v���A� *

	conv_loss��;��T        )��P	��v���A� *

	conv_lossaD<��H        )��P	0��v���A� *

	conv_lossuy�;��!        )��P	 �v���A� *

	conv_loss�<���i        )��P	�I�v���A� *

	conv_loss�<���U        )��P	J{�v���A� *

	conv_loss ص;��Ǝ        )��P	w��v���A� *

	conv_loss��?<y�d�        )��P	���v���A� *

	conv_loss�#�<7��        )��P	=�v���A� *

	conv_loss�<h3�        )��P	�@�v���A� *

	conv_loss�J�<�ġ_        )��P	�r�v���A� *

	conv_loss�?5<�>��        )��P	y��v���A� *

	conv_loss�oE<P�	'        )��P	R��v���A� *

	conv_loss��;���F        )��P	<
�v���A� *

	conv_lossg�;c��2        )��P	�;�v���A� *

	conv_lossxA <�h�E        )��P	�m�v���A� *

	conv_loss>��;O�[Y        )��P	��v���A� *

	conv_loss���<�	V�        )��P	>��v���A� *

	conv_losstl<р4�        )��P	v�v���A� *

	conv_lossػ;=�Ϫ        )��P	3�v���A� *

	conv_lossf�<�9ܿ        )��P	g�v���A� *

	conv_lossS �;g3��        )��P	S��v���A� *

	conv_lossf�m<�y�u        )��P	��v���A� *

	conv_loss!�E<ܝS�        )��P	6�v���A� *

	conv_loss��<|�u        )��P	�7�v���A� *

	conv_losskj<R[9�        )��P	Oj�v���A� *

	conv_loss�e<`Svn        )��P	ĝ�v���A� *

	conv_loss�B�;H �        )��P	��v���A� *

	conv_loss��&<��[R        )��P	i �v���A� *

	conv_loss9�W<��-�        )��P	���v���A� *

	conv_loss�8�<0��        )��P	S��v���A� *

	conv_lossY�F<R���        )��P	D��v���A� *

	conv_loss�};�d��        )��P	A-�v���A� *

	conv_loss�#!<O@�        )��P	^�v���A� *

	conv_loss��<�fW�        )��P	��v���A� *

	conv_loss�H<3C'        )��P	ݿ�v���A� *

	conv_lossK<�.��        )��P	}��v���A� *

	conv_loss�]�<��x        )��P	�6�v���A� *

	conv_loss\A
<y�c�        )��P	�o�v���A� *

	conv_lossp��<��p�        )��P	 �v���A� *

	conv_loss{�0<@��        )��P	���v���A� *

	conv_loss�?J<�<EK        )��P	�v���A� *

	conv_loss�c�<�u(�        )��P	�J�v���A� *

	conv_loss	�=<5[��        )��P	j}�v���A� *

	conv_loss�&�<
��K        )��P	k��v���A� *

	conv_loss� $<����        )��P	���v���A� *

	conv_loss��S<�#M        )��P	z�v���A� *

	conv_loss/e�;Ћ&w        )��P	RD�v���A� *

	conv_loss};�;���R        )��P	�x�v���A� *

	conv_loss�W<���        )��P	��v���A� *

	conv_loss��*<$��_        )��P	��v���A� *

	conv_loss���<��!        )��P	��v���A� *

	conv_loss�M<؅��        )��P	�B�v���A�!*

	conv_lossB��;�,�        )��P	r�v���A�!*

	conv_loss֢�<{�B1        )��P	���v���A�!*

	conv_loss���;�ُ�        )��P	���v���A�!*

	conv_loss+�=k���        )��P	��v���A�!*

	conv_loss|��<-X��        )��P	�5�v���A�!*

	conv_lossg{v<ձ�        )��P	j�v���A�!*

	conv_loss�7s<��}T        )��P	���v���A�!*

	conv_losszR<�&D0        )��P	���v���A�!*

	conv_loss�]{<B�9        )��P	A��v���A�!*

	conv_loss=��<&�.�        )��P	�0�v���A�!*

	conv_lossB�;��wC        )��P	�a�v���A�!*

	conv_lossY@<aV��        )��P	ߔ�v���A�!*

	conv_loss3�<k���        )��P	���v���A�!*

	conv_loss��<���        )��P	���v���A�!*

	conv_loss��;��t�        )��P	�.�v���A�!*

	conv_lossL�<�i�$        )��P	�_�v���A�!*

	conv_loss�'�<ݓZ        )��P	P��v���A�!*

	conv_loss?�<�i@�        )��P	���v���A�!*

	conv_loss8	<֛�        )��P	���v���A�!*

	conv_loss���;'K        )��P	�&�v���A�!*

	conv_loss���<bR        )��P	�V�v���A�!*

	conv_loss���;I�6�        )��P	���v���A�!*

	conv_loss
�<W�&        )��P	��v���A�!*

	conv_loss��D<{�        )��P	���v���A�!*

	conv_loss��<���J        )��P	��v���A�!*

	conv_lossj��;��        )��P	I�v���A�!*

	conv_lossـ<eY�k        )��P	L��v���A�!*

	conv_loss��<M'��        )��P	���v���A�!*

	conv_loss;<=�s�        )��P	<��v���A�!*

	conv_loss��4<���        )��P	�'�v���A�!*

	conv_lossY�;���        )��P	kX�v���A�!*

	conv_loss��E<�'?        )��P	v��v���A�!*

	conv_losspB<��z�        )��P	��v���A�!*

	conv_loss}� <�wa        )��P	��v���A�!*

	conv_loss�y<��        )��P	w&�v���A�!*

	conv_lossl.�<��v        )��P	sc�v���A�!*

	conv_loss�`�<8�v        )��P	���v���A�!*

	conv_loss�O<���        )��P	���v���A�!*

	conv_loss;�<t�G�        )��P	O��v���A�!*

	conv_loss�j&<L�E�        )��P	56�v���A�!*

	conv_loss#�<J&��        )��P	w�v���A�!*

	conv_loss��;�W;�        )��P	5��v���A�!*

	conv_loss�7�<ɓQ�        )��P	���v���A�!*

	conv_lossa�<O���        )��P	�v���A�!*

	conv_loss4�;ݗ��        )��P	�@�v���A�!*

	conv_loss�$n;���1        )��P	)��v���A�!*

	conv_loss@��;C]        )��P	ɲ�v���A�!*

	conv_loss�<��Z�        )��P	��v���A�!*

	conv_loss�Ҋ<_�w�        )��P	��v���A�!*

	conv_lossQ��;	�6�        )��P	�J�v���A�!*

	conv_loss�<���        )��P	R|�v���A�!*

	conv_lossG��<��U        )��P	��v���A�!*

	conv_loss��<���t        )��P	���v���A�!*

	conv_lossBt<p�2        )��P	:�v���A�!*

	conv_loss�4<�L��        )��P	�L�v���A�!*

	conv_loss@mq<���        )��P	D|�v���A�!*

	conv_loss2�;(���        )��P	ͬ�v���A�!*

	conv_loss�{�;�>�        )��P	j��v���A�!*

	conv_loss�5�;+�        )��P	��v���A�!*

	conv_lossٱq<ηe�        )��P	�D�v���A�!*

	conv_loss(Ur<<2�U        )��P	v�v���A�!*

	conv_loss�U�;ڠA4        )��P	���v���A�!*

	conv_loss,�;�+�f        )��P	���v���A�!*

	conv_loss�<C�        )��P	E	�v���A�!*

	conv_lossR�;X��~        )��P	&9�v���A�!*

	conv_loss�$<�#ۗ        )��P	�h�v���A�!*

	conv_loss�
-;�xi        )��P	��v���A�!*

	conv_loss��<�$i�        )��P	+��v���A�!*

	conv_losswA�:�2}s        )��P	���v���A�!*

	conv_loss��<�$D        )��P	�.�v���A�!*

	conv_lossh<��wm        )��P	�^�v���A�!*

	conv_loss��5<�`��        )��P	���v���A�!*

	conv_lossn��<5�K-        )��P	���v���A�!*

	conv_loss��_<;���        )��P	��v���A�!*

	conv_loss�U�<�y��        )��P	�!�v���A�!*

	conv_loss�ɴ;_���        )��P	AS�v���A�!*

	conv_loss3
<ڪ��        )��P	���v���A�!*

	conv_lossX��<n\��        )��P	x��v���A�!*

	conv_loss4��<�Rz        )��P	���v���A�!*

	conv_lossF�c<�_m�        )��P	\0�v���A�!*

	conv_loss�1�;\���        )��P	�b�v���A�!*

	conv_loss�~�;�$��        )��P	��v���A�!*

	conv_loss��<�q��        )��P	���v���A�!*

	conv_loss7��<*7�$        )��P	>��v���A�!*

	conv_loss9H�;W=e�        )��P	6�v���A�!*

	conv_lossM�<���g        )��P	�l�v���A�!*

	conv_loss�1r<�T�        )��P	4��v���A�!*

	conv_loss��<�煚        )��P	k��v���A�!*

	conv_loss/��;�x�$        )��P	C
�v���A�!*

	conv_loss
��<��-z        )��P	�@�v���A�!*

	conv_loss��<;�        )��P	sv�v���A�!*

	conv_loss���;���        )��P	��v���A�!*

	conv_lossઈ<*À        )��P	@��v���A�!*

	conv_lossZ�<1B�        )��P	4�v���A�!*

	conv_loss�h�;��        )��P	�@�v���A�!*

	conv_loss�x:<0�rr        )��P	�q�v���A�!*

	conv_loss�BQ;��;;        )��P	���v���A�!*

	conv_loss�L<��ɩ        )��P	2��v���A�!*

	conv_loss���;�o�        )��P	E
�v���A�!*

	conv_loss�q;��        )��P	�;�v���A�!*

	conv_loss{V<���a        )��P	Um�v���A�!*

	conv_loss-�<����        )��P	C��v���A�!*

	conv_loss jm<P��G        )��P	���v���A�!*

	conv_loss)��;��        )��P	��v���A�!*

	conv_loss�V�;i.�        )��P	�2�v���A�!*

	conv_loss�w<�o��        )��P	c�v���A�!*

	conv_lossY�;lДI        )��P	
��v���A�!*

	conv_loss_�6<)w��        )��P	���v���A�!*

	conv_loss��?<��f�        )��P	���v���A�!*

	conv_loss�ʟ<m	UA        )��P	)�v���A�!*

	conv_loss��P<��a.        )��P	�[�v���A�!*

	conv_losss�1<�XX�        )��P	���v���A�!*

	conv_loss� <�E�        )��P	���v���A�!*

	conv_lossMf<1�S        )��P	<��v���A�!*

	conv_loss�+�<r<�q        )��P	/2�v���A�!*

	conv_loss'�_<��y        )��P	oe�v���A�!*

	conv_loss�P5<��}        )��P	���v���A�!*

	conv_loss��=�?�        )��P	;��v���A�!*

	conv_loss�2�<�z�        )��P	��v���A�!*

	conv_lossp�k<�o�        )��P	�5�v���A�!*

	conv_loss2�<��U�        )��P	�i�v���A�!*

	conv_lossO�;���;        )��P	2��v���A�!*

	conv_loss�2�<~�e�        )��P	���v���A�!*

	conv_lossQ]~<!>��        )��P	��v���A�!*

	conv_lossKJ$<�f!        )��P	�9�v���A�!*

	conv_loss��; �m�        )��P	�n�v���A�!*

	conv_loss��<��@F        )��P	���v���A�!*

	conv_loss�\<�S̞        )��P	���v���A�"*

	conv_loss�^k<x���        )��P	�! w���A�"*

	conv_loss��R<���Y        )��P	]V w���A�"*

	conv_lossUc<Å        )��P	� w���A�"*

	conv_lossZ|5<�P:�        )��P	Ծ w���A�"*

	conv_loss���<��        )��P	�� w���A�"*

	conv_loss��<x��~        )��P	1w���A�"*

	conv_lossR�<�@Q        )��P	�vw���A�"*

	conv_lossC=S<���        )��P	��w���A�"*

	conv_loss37K<�_        )��P	��w���A�"*

	conv_loss�G�<��F        )��P	!w���A�"*

	conv_loss�@I<6O�         )��P	�Ww���A�"*

	conv_loss���;��;        )��P	q�w���A�"*

	conv_loss*6�<����        )��P	��w���A�"*

	conv_loss��B<	)�        )��P	�w���A�"*

	conv_loss�6�<Y�Щ        )��P	RGw���A�"*

	conv_loss�J<#�]�        )��P	q{w���A�"*

	conv_loss�b4<��        )��P	a�w���A�"*

	conv_loss�;�9��        )��P	��w���A�"*

	conv_losssh�;�x�~        )��P	�w���A�"*

	conv_loss2TF<ճ��        )��P	�Mw���A�"*

	conv_loss�o�<0��        )��P	�w���A�"*

	conv_loss�A�<Ih��        )��P	��w���A�"*

	conv_loss�:<�?��        )��P	��w���A�"*

	conv_loss�mi<p�*�        )��P	w���A�"*

	conv_loss��v<	2��        )��P	�Rw���A�"*

	conv_loss�}�<�7'        )��P	�w���A�"*

	conv_lossM�Y<3χ_        )��P	X�w���A�"*

	conv_loss(#<9N/K        )��P	t�w���A�"*

	conv_loss&w�;��o        )��P	�$w���A�"*

	conv_loss�H�;`�~�        )��P	�Yw���A�"*

	conv_loss��;�~��        )��P	P�w���A�"*

	conv_lossJ�<<0�%        )��P	��w���A�"*

	conv_loss�<Zq��        )��P	G�w���A�"*

	conv_lossrHu<�w��        )��P	A,w���A�"*

	conv_lossk)�<����        )��P	2_w���A�"*

	conv_loss��<uq��        )��P	˓w���A�"*

	conv_lossɨ<<���+        )��P	(�w���A�"*

	conv_loss��<(Fc[        )��P	��w���A�"*

	conv_loss>5�<:��M        )��P	�0w���A�"*

	conv_lossR<�^7        )��P	
fw���A�"*

	conv_loss�W�;JH��        )��P	5�w���A�"*

	conv_loss��H<#�H        )��P	p�w���A�"*

	conv_loss{\�<KmPM        )��P		w���A�"*

	conv_lossV*<�0+)        )��P	7	w���A�"*

	conv_lossw�;��        )��P	�k	w���A�"*

	conv_loss3R;��Z        )��P	$�	w���A�"*

	conv_lossmi�<!�P�        )��P	��	w���A�"*

	conv_loss��=�2��        )��P	J
w���A�"*

	conv_loss��P<�^�        )��P	l�w���A�"*

	conv_lossn��<́p�        )��P	4�w���A�"*

	conv_loss�k�<��x*        )��P	&	w���A�"*

	conv_lossOx<N�P�        )��P	wEw���A�"*

	conv_loss3�f<b��<        )��P	�w���A�"*

	conv_loss�v�<dT�        )��P	(�w���A�"*

	conv_lossq�M<u��        )��P	�w���A�"*

	conv_loss�7<'/�a        )��P	Aw���A�"*

	conv_loss:��;!kH�        )��P	F]w���A�"*

	conv_loss�W;ii        )��P	��w���A�"*

	conv_lossz�;�Ӧ�        )��P	��w���A�"*

	conv_loss]<ׂL        )��P	��w���A�"*

	conv_loss�՟<{��o        )��P	�/w���A�"*

	conv_lossu� <\d��        )��P	�ew���A�"*

	conv_loss��3<�!a        )��P	ŗw���A�"*

	conv_loss�q<� B        )��P	q�w���A�"*

	conv_lossd̈;�N�T        )��P	{w���A�"*

	conv_loss�U�<!u��        )��P	3Aw���A�"*

	conv_loss��K<a�1        )��P	�vw���A�"*

	conv_loss��F<�c t        )��P	�w���A�"*

	conv_loss��O<fk+i        )��P	��w���A�"*

	conv_loss���;��);        )��P	�w���A�"*

	conv_loss��<��3        )��P	�Fw���A�"*

	conv_loss� <|�b�        )��P	�{w���A�"*

	conv_loss`A�<[^ o        )��P	ıw���A�"*

	conv_loss�4�;�k9        )��P	��w���A�"*

	conv_loss\}<��!        )��P	
w���A�"*

	conv_loss��<`�9v        )��P	mPw���A�"*

	conv_loss/��<;��        )��P	�w���A�"*

	conv_loss}&<���        )��P	�w���A�"*

	conv_loss�<�M�        )��P	��w���A�"*

	conv_lossRh8<�e��        )��P	#w���A�"*

	conv_loss�|;��r�        )��P	Ww���A�"*

	conv_loss��R<�̭        )��P	��w���A�"*

	conv_lossv��<@��'        )��P	�w���A�"*

	conv_lossF�h<x/�        )��P	�w���A�"*

	conv_loss�a< �        )��P	�&w���A�"*

	conv_loss�	<k�        )��P	�Yw���A�"*

	conv_loss�~�;�se        )��P	t�w���A�"*

	conv_loss��;1$b>        )��P	��w���A�"*

	conv_lossn�<d9#}        )��P	��w���A�"*

	conv_loss���;Q� �        )��P	�@w���A�"*

	conv_loss&��;���]        )��P	�uw���A�"*

	conv_loss"c�<��fP        )��P	s�w���A�"*

	conv_loss�V`<�$>�        )��P	��w���A�"*

	conv_loss�W<p�h�        )��P	�w���A�"*

	conv_loss��=<_I�r        )��P	LGw���A�"*

	conv_loss�x<�K�        )��P	�w���A�"*

	conv_loss�H�;����        )��P	q�w���A�"*

	conv_loss�	"<a2�        )��P	��w���A�"*

	conv_losss`^<t��        )��P	�9w���A�"*

	conv_losss��<
>w9        )��P	�rw���A�"*

	conv_loss[9<3{�        )��P	�w���A�"*

	conv_loss�P<�T�        )��P	��w���A�"*

	conv_loss�I<Yv��        )��P	/w���A�"*

	conv_loss��<ކ�        )��P	�Sw���A�"*

	conv_loss�)�;���        )��P	܇w���A�"*

	conv_loss��;68�        )��P	��w���A�"*

	conv_loss��;2'k�        )��P	�w���A�"*

	conv_loss��B<ז��        )��P	,,w���A�"*

	conv_loss���;�@��        )��P	�bw���A�"*

	conv_lossh.<��9        )��P	~�w���A�"*

	conv_lossZ�m;c�@�        )��P	��w���A�"*

	conv_lossa��;��        )��P	ww���A�"*

	conv_loss��<K@��        )��P	Lw���A�"*

	conv_lossJ?<��U�        )��P	s�w���A�"*

	conv_loss���<�J�        )��P	�w���A�"*

	conv_loss���;#�	        )��P	��w���A�"*

	conv_loss��
<��l        )��P	�w���A�"*

	conv_loss�5�<�O��        )��P	:Rw���A�"*

	conv_loss0�*<��*        )��P	��w���A�"*

	conv_loss��<<9n        )��P	ʻw���A�"*

	conv_loss�5<��{        )��P	��w���A�"*

	conv_loss���;��        )��P	�%w���A�"*

	conv_loss���<b���        )��P	�Yw���A�"*

	conv_loss��;��o@        )��P	]�w���A�"*

	conv_lossc�(<�7Wg        )��P	��w���A�"*

	conv_lossÂ;v�8l        )��P	��w���A�"*

	conv_loss�`�;Y�oP        )��P	c+w���A�"*

	conv_lossJ��<�9�0        )��P	�_w���A�#*

	conv_loss�;!S        )��P	I�w���A�#*

	conv_loss��;��        )��P	��w���A�#*

	conv_loss�ԓ<�N�I        )��P	�w���A�#*

	conv_lossbK�;g.�        )��P	~1w���A�#*

	conv_lossTY�<φ��        )��P	�cw���A�#*

	conv_loss9�2<dBn�        )��P	Ɩw���A�#*

	conv_loss�6<��,�        )��P	|�w���A�#*

	conv_loss�T<R�        )��P	��w���A�#*

	conv_loss*�	=��K        )��P	e1w���A�#*

	conv_loss�:<Ɏ�i        )��P	�cw���A�#*

	conv_loss�<н�E        )��P	M�w���A�#*

	conv_lossEӢ<��`�        )��P	�w���A�#*

	conv_lossJ��;���        )��P	 w���A�#*

	conv_loss�i[<
�ކ        )��P	�5w���A�#*

	conv_loss��=<",V        )��P	9iw���A�#*

	conv_loss�C�<@�[M        )��P	}�w���A�#*

	conv_loss���;�S�i        )��P	�w���A�#*

	conv_loss�<��wC        )��P	^	 w���A�#*

	conv_lossD'|<h}�        )��P	�; w���A�#*

	conv_losse;f<!�?        )��P	a5%w���A�#*

	conv_lossʴ;�	��        )��P	u%w���A�#*

	conv_loss�<��~        )��P	Ѥ%w���A�#*

	conv_loss�j$<�+�        )��P	i�%w���A�#*

	conv_loss��<�M��        )��P	&w���A�#*

	conv_loss�<�r��        )��P	�6&w���A�#*

	conv_loss��G<r��        )��P	�h&w���A�#*

	conv_loss�3s<'���        )��P	|�&w���A�#*

	conv_lossJ��;�'UR        )��P	��&w���A�#*

	conv_lossI�x<�=�        )��P	Y�&w���A�#*

	conv_loss��"<� 0�        )��P	�,'w���A�#*

	conv_lossۙ<|+r        )��P	^'w���A�#*

	conv_loss_�;/�p        )��P	ɏ'w���A�#*

	conv_lossS4�;5���        )��P	��'w���A�#*

	conv_loss�J<���x        )��P	��'w���A�#*

	conv_loss �;L<�        )��P	1(w���A�#*

	conv_loss��,<�m��        )��P	4d(w���A�#*

	conv_loss��W<z���        )��P	7�(w���A�#*

	conv_loss��F<�캦        )��P	��(w���A�#*

	conv_lossUYB<���w        )��P	Q)w���A�#*

	conv_loss���;�?T        )��P	�5)w���A�#*

	conv_loss�6�;E8!Z        )��P	�d)w���A�#*

	conv_loss�6}<�n�        )��P	A�)w���A�#*

	conv_lossmA�;EeƖ        )��P	C�)w���A�#*

	conv_loss�P�;�>��        )��P	z�)w���A�#*

	conv_lossBp<0��|        )��P	~"*w���A�#*

	conv_lossO]�;xRb�        )��P	�P*w���A�#*

	conv_loss�k�;��6�        )��P	��*w���A�#*

	conv_lossF�<BYpq        )��P	�*w���A�#*

	conv_loss�7<�M��        )��P	��*w���A�#*

	conv_loss�rO<�_��        )��P	�+w���A�#*

	conv_loss�r<H/�        )��P	�>+w���A�#*

	conv_lossц<K��f        )��P	,r+w���A�#*

	conv_loss�/<�$        )��P	��+w���A�#*

	conv_loss(\<��L�        )��P	D�+w���A�#*

	conv_loss���;K�~_        )��P	O�+w���A�#*

	conv_loss��0<K﫦        )��P	-,w���A�#*

	conv_loss�ܰ;#J�q        )��P	Z,w���A�#*

	conv_loss�R;��N        )��P	��,w���A�#*

	conv_loss|l�<�<�6        )��P	��,w���A�#*

	conv_lossZ~j<�<٤        )��P	!�,w���A�#*

	conv_loss���<8��(        )��P	�-w���A�#*

	conv_loss=Ǩ<�
�        )��P	I-w���A�#*

	conv_loss��<�o�A        )��P	�v-w���A�#*

	conv_lossK�&<�0�        )��P	��-w���A�#*

	conv_loss��A;�ө        )��P	��-w���A�#*

	conv_loss��<���        )��P	�.w���A�#*

	conv_loss�<?��        )��P	L3.w���A�#*

	conv_lossY�<��Z@        )��P	a.w���A�#*

	conv_lossv$<u��        )��P	��.w���A�#*

	conv_loss���;x>�&        )��P	f�.w���A�#*

	conv_loss
a�<�ފ?        )��P	�.w���A�#*

	conv_loss"5�< #C        )��P	�//w���A�#*

	conv_lossN
a<7�!�        )��P	�_/w���A�#*

	conv_loss�l�;����        )��P	��/w���A�#*

	conv_loss��;�9         )��P	�/w���A�#*

	conv_lossTB:<�aj        )��P	z�/w���A�#*

	conv_loss �!<�q��        )��P	&0w���A�#*

	conv_loss��e;�C#�        )��P	U0w���A�#*

	conv_loss�Aw;��y        )��P	�0w���A�#*

	conv_loss�S-<���7        )��P	d�0w���A�#*

	conv_loss���;�:Z        )��P	��0w���A�#*

	conv_loss9�=�p�        )��P	|(1w���A�#*

	conv_loss�1�;�$Q�        )��P	�X1w���A�#*

	conv_loss�(�<���        )��P	��1w���A�#*

	conv_loss��<�GY        )��P	��1w���A�#*

	conv_lossv-|<Yp�W        )��P	��1w���A�#*

	conv_loss��	<��X�        )��P	d02w���A�#*

	conv_loss��H<g:��        )��P	�j2w���A�#*

	conv_losst��;=��        )��P	��2w���A�#*

	conv_loss`�<_O�        )��P	@�2w���A�#*

	conv_loss�8<K}_        )��P	a3w���A�#*

	conv_loss��<�S��        )��P	363w���A�#*

	conv_loss봢<�F��        )��P	ni3w���A�#*

	conv_loss��><�B|        )��P	��3w���A�#*

	conv_lossS�<�@�V        )��P	4�3w���A�#*

	conv_loss�=�;��L�        )��P	��3w���A�#*

	conv_loss��<�L0        )��P	�+4w���A�#*

	conv_lossX�;��^�        )��P	�^4w���A�#*

	conv_loss���<L���        )��P	��4w���A�#*

	conv_loss=�p<&�K        )��P	\�4w���A�#*

	conv_loss���;ߴA�        )��P	P�4w���A�#*

	conv_loss6�<T���        )��P	�)5w���A�#*

	conv_loss��;ӱq�        )��P	BZ5w���A�#*

	conv_lossw�;��`        )��P	9�5w���A�#*

	conv_loss��=<U
��        )��P	E�5w���A�#*

	conv_loss�{C<.�Y        )��P	p�5w���A�#*

	conv_loss���;c�3        )��P	v!6w���A�#*

	conv_loss�<����        )��P	�R6w���A�#*

	conv_lossՅ<�� ]        )��P	��6w���A�#*

	conv_loss�8�<QՒ�        )��P	��6w���A�#*

	conv_loss:�(<��<�        )��P	F�6w���A�#*

	conv_lossxȳ;?��z        )��P	�7w���A�#*

	conv_loss���;��
�        )��P	,H7w���A�#*

	conv_loss+3<ɬ�        )��P	yz7w���A�#*

	conv_loss}�<�?J        )��P	Ŭ7w���A�#*

	conv_loss�7�<>��        )��P	R�7w���A�#*

	conv_lossv�<d��        )��P	�8w���A�#*

	conv_loss��<�	        )��P	�A8w���A�#*

	conv_lossr�/<���        )��P	�s8w���A�#*

	conv_lossO!�<R�v        )��P	��8w���A�#*

	conv_loss �<�q��        )��P	`7:w���A�#*

	conv_loss�ˀ;��        )��P	�j:w���A�#*

	conv_loss.��;&V�        )��P	��:w���A�#*

	conv_loss�;5        )��P	V�:w���A�#*

	conv_loss��;1��        )��P	��:w���A�#*

	conv_loss"14;��w        )��P	�,;w���A�#*

	conv_lossD�<��U�        )��P	_;w���A�#*

	conv_lossx�<�G��        )��P	�;w���A�$*

	conv_lossrT�<E8�        )��P	h�;w���A�$*

	conv_losse��<��ü        )��P	I<w���A�$*

	conv_loss��<����        )��P	�8<w���A�$*

	conv_lossX�r<���        )��P	�p<w���A�$*

	conv_loss<ƶ<���        )��P	X�<w���A�$*

	conv_loss�<�Q        )��P	6�<w���A�$*

	conv_lossQ��;�>�        )��P	�=w���A�$*

	conv_loss�#<2���        )��P	B=w���A�$*

	conv_loss/V~<��OU        )��P	�r=w���A�$*

	conv_lossQ��;�=�Z        )��P	�=w���A�$*

	conv_loss�d<z�9�        )��P	�=w���A�$*

	conv_loss\f<�-        )��P	>w���A�$*

	conv_loss:B�;�<4�        )��P	6>w���A�$*

	conv_loss^�;n�
�        )��P	�e>w���A�$*

	conv_loss�29<���#        )��P	֔>w���A�$*

	conv_loss�X;���        )��P	��>w���A�$*

	conv_loss���:@�#        )��P	!�>w���A�$*

	conv_loss��;<�֢d        )��P	�"?w���A�$*

	conv_loss�<�o�9        )��P	S?w���A�$*

	conv_loss��.<�(�        )��P	ց?w���A�$*

	conv_loss	�4<��Dt        )��P	e�?w���A�$*

	conv_lossC2q<��2         )��P	2�?w���A�$*

	conv_loss�)X<�a�        )��P	�@w���A�$*

	conv_loss6vs<'�+�        )��P	N<@w���A�$*

	conv_lossºA<�C��        )��P	�j@w���A�$*

	conv_lossђ�;�-ҵ        )��P	�@w���A�$*

	conv_loss8�,<�Pq�        )��P	��@w���A�$*

	conv_lossm3�;I���        )��P	�@w���A�$*

	conv_lossV!<�j<t        )��P	�&Aw���A�$*

	conv_losss��<~�e�        )��P	�TAw���A�$*

	conv_loss�zn;s��        )��P	τAw���A�$*

	conv_loss�2�;_��        )��P	 �Aw���A�$*

	conv_lossvr�;kt�U        )��P	�Aw���A�$*

	conv_lossר�<	u �        )��P	>Bw���A�$*

	conv_losso�;��T        )��P	{CBw���A�$*

	conv_loss�=;�>�        )��P	2sBw���A�$*

	conv_loss*MX<���U        )��P	c�Bw���A�$*

	conv_loss���<����        )��P	x�Bw���A�$*

	conv_loss�E�<DyFI        )��P	� Cw���A�$*

	conv_lossn�<�K        )��P	�/Cw���A�$*

	conv_loss��; ��        )��P	�`Cw���A�$*

	conv_loss��e<B�30        )��P	^�Cw���A�$*

	conv_loss�F<�5�        )��P	��Cw���A�$*

	conv_lossD�<���,        )��P	6�Cw���A�$*

	conv_loss���<8>��        )��P	�+Dw���A�$*

	conv_loss���;�aĨ        )��P	�YDw���A�$*

	conv_loss ;=<��i        )��P	��Dw���A�$*

	conv_loss}��<c��(        )��P	ǽDw���A�$*

	conv_loss0j<��Z.        )��P	��Dw���A�$*

	conv_loss�<*ɽ�        )��P	�Ew���A�$*

	conv_loss�}�<��ד        )��P	-PEw���A�$*

	conv_losst.�;2�A�        )��P	C�Ew���A�$*

	conv_lossW�L;�p        )��P	��Ew���A�$*

	conv_loss�E</��z        )��P	��Ew���A�$*

	conv_lossϻ;�	�        )��P	;Fw���A�$*

	conv_loss�q<����        )��P	CHFw���A�$*

	conv_lossgW<�@aq        )��P	cFw���A�$*

	conv_loss-/<m���        )��P	��Fw���A�$*

	conv_loss6j�<@�%>        )��P	X�Fw���A�$*

	conv_lossf`�<��@$        )��P	�Gw���A�$*

	conv_loss1~K<��J$        )��P	�LGw���A�$*

	conv_loss4�<A        )��P	�}Gw���A�$*

	conv_loss�� <�a?        )��P	t�Gw���A�$*

	conv_losskd�;���        )��P	��Gw���A�$*

	conv_losst�L<�)��        )��P	Hw���A�$*

	conv_loss��k<�i�s        )��P	�>Hw���A�$*

	conv_loss��;�'�        )��P	�oHw���A�$*

	conv_loss�><��1        )��P	e�Hw���A�$*

	conv_loss�n�;�        )��P	?�Hw���A�$*

	conv_loss�%<�Ky        )��P	FIw���A�$*

	conv_loss��P<�P�O        )��P	�1Iw���A�$*

	conv_loss�:�<�U*]        )��P	1�Iw���A�$*

	conv_lossXq<|�7�        )��P	k!Jw���A�$*

	conv_loss�܋;)_t�        )��P	��Jw���A�$*

	conv_loss���;���,        )��P	�Kw���A�$*

	conv_lossn��;,�        )��P	�Kw���A�$*

	conv_loss|Av;c��_        )��P	Lw���A�$*

	conv_loss�I�;��Z�        )��P	ZzLw���A�$*

	conv_lossg��<o��0        )��P	��Lw���A�$*

	conv_lossJT�<4*>�        )��P	pjMw���A�$*

	conv_loss6r�;�]u�        )��P	��Mw���A�$*

	conv_loss"<"�4,        )��P	�[Nw���A�$*

	conv_loss��<f��d        )��P	��Nw���A�$*

	conv_loss|�K<N��        )��P	QHOw���A�$*

	conv_loss�	<�X�        )��P	�Ow���A�$*

	conv_loss�|<.2�        )��P	O3Pw���A�$*

	conv_loss�?<�$�        )��P	�Pw���A�$*

	conv_lossp�;���l        )��P	�Qw���A�$*

	conv_loss���;�/+�        )��P	�Qw���A�$*

	conv_loss�Z<�B$        )��P	�
Rw���A�$*

	conv_lossdǂ<� X�        )��P	?�Rw���A�$*

	conv_loss��;���        )��P	y�Rw���A�$*

	conv_loss4�>;���        )��P	ʞSw���A�$*

	conv_loss�*<�n%        )��P	�Tw���A�$*

	conv_lossx<%P�        )��P	�Tw���A�$*

	conv_loss1f�<��$�        )��P	Uw���A�$*

	conv_loss7]n<��E        )��P	O�Uw���A�$*

	conv_loss.��;(��        )��P	#Vw���A�$*

	conv_loss�1<�OL        )��P	e�Vw���A�$*

	conv_lossTh�<>3B)        )��P	Ww���A�$*

	conv_loss��;���7        )��P	u�Ww���A�$*

	conv_loss�lA<�/        )��P	�
Xw���A�$*

	conv_lossfgQ<�mO�        )��P	�~Xw���A�$*

	conv_loss=*�;B�?        )��P	��Xw���A�$*

	conv_lossI��<��u�        )��P	�sYw���A�$*

	conv_loss�*�<R
��        )��P	;�Yw���A�$*

	conv_lossM�C<��4�        )��P	�Zw���A�$*

	conv_lossV"	<Aĥ        )��P	�^Zw���A�$*

	conv_loss-./<+U�        )��P	0�Zw���A�$*

	conv_lossEH<1xR        )��P	F�Zw���A�$*

	conv_lossp��<I��?        )��P	�&[w���A�$*

	conv_loss֤i<��>        )��P	8i[w���A�$*

	conv_loss=�K<f�K�        )��P	֫[w���A�$*

	conv_loss��;b���        )��P	�[w���A�$*

	conv_loss]}<�\�        )��P	x2\w���A�$*

	conv_loss��<��2        )��P	iv\w���A�$*

	conv_lossch�<�r��        )��P	��\w���A�$*

	conv_loss�N;�3Q        )��P	�\w���A�$*

	conv_loss׻�;��pw        )��P	�A]w���A�$*

	conv_loss7�<k��4        )��P	�]w���A�$*

	conv_loss�%u;��q        )��P	��]w���A�$*

	conv_lossa}�;��s�        )��P	U
^w���A�$*

	conv_loss��;���~        )��P	�M^w���A�$*

	conv_loss^�<���        )��P	-�^w���A�$*

	conv_loss��/<��(H        )��P	��^w���A�$*

	conv_loss�)�;g�0        )��P	�_w���A�$*

	conv_lossN�e;�yt        )��P	WW_w���A�$*

	conv_loss��:��U	        )��P	(�_w���A�%*

	conv_loss�H�<�~J�        )��P	��_w���A�%*

	conv_lossP� <
5�        )��P	"`w���A�%*

	conv_loss���;Ё/�        )��P	rc`w���A�%*

	conv_loss�<��'�        )��P	&�`w���A�%*

	conv_loss�}<E[�        )��P	k�`w���A�%*

	conv_loss�i<��        )��P	a,aw���A�%*

	conv_loss��K<?T��        )��P	�oaw���A�%*

	conv_loss݄8;򩅋        )��P	��aw���A�%*

	conv_loss̤(<��^_        )��P	��aw���A�%*

	conv_lossb�0<
��k        )��P	�?bw���A�%*

	conv_loss���;��}        )��P	6�bw���A�%*

	conv_loss�z@<��w&        )��P	��bw���A�%*

	conv_loss) <v�MG        )��P	�cw���A�%*

	conv_loss��R<���        )��P	i[cw���A�%*

	conv_loss��<��        )��P	G�cw���A�%*

	conv_loss}=<����        )��P	< dw���A�%*

	conv_lossu�W<?�{2        )��P	�Hdw���A�%*

	conv_lossMI<��)        )��P	)�dw���A�%*

	conv_loss,�B<�M�[        )��P	��dw���A�%*

	conv_loss�b�<�;�        )��P	[7ew���A�%*

	conv_loss�H�;@�:�        )��P	k�ew���A�%*

	conv_loss$z^<��ĳ        )��P	�ew���A�%*

	conv_loss��;���        )��P	�fw���A�%*

	conv_loss�h<l��        )��P	c\fw���A�%*

	conv_loss7�.<u7ԉ        )��P	՞fw���A�%*

	conv_lossꔔ;+�w�        )��P	��fw���A�%*

	conv_loss��<��$        )��P	J"gw���A�%*

	conv_loss��<��n(        )��P	�ugw���A�%*

	conv_loss�m;��        )��P	j�gw���A�%*

	conv_loss�<�z�        )��P	_hw���A�%*

	conv_loss�2�;8>�        )��P	,Qhw���A�%*

	conv_loss�n�;5�r0        )��P	�hw���A�%*

	conv_loss<�;� 4        )��P	�hw���A�%*

	conv_lossn��;i`)�        )��P	iw���A�%*

	conv_loss���;�         )��P	�[iw���A�%*

	conv_loss�u�;�44�        )��P	�iw���A�%*

	conv_loss���;���        )��P	-�iw���A�%*

	conv_loss�E;��J-        )��P	,#jw���A�%*

	conv_loss���<7�Y9        )��P	�fjw���A�%*

	conv_lossh]j;�	^         )��P	^�jw���A�%*

	conv_loss/�-<��-�        )��P	2�jw���A�%*

	conv_loss��g<��e�        )��P	�-kw���A�%*

	conv_loss���;@�5�        )��P	�qkw���A�%*

	conv_lossH�<RU��        )��P	u�kw���A�%*

	conv_lossVg;5_�$        )��P	��kw���A�%*

	conv_loss�'<n�'        )��P	�<lw���A�%*

	conv_loss�P;��Oq        )��P	�lw���A�%*

	conv_lossի<��_�        )��P	�lw���A�%*

	conv_loss�<���        )��P	Smw���A�%*

	conv_lossQ�;;1�        )��P	�Hmw���A�%*

	conv_loss0д<#�z�        )��P	��mw���A�%*

	conv_loss��;G.��        )��P	��mw���A�%*

	conv_loss׎"<^-�7        )��P	^nw���A�%*

	conv_loss=�v<7fD)        )��P	 Qnw���A�%*

	conv_lossFɪ;�;�{        )��P	�nw���A�%*

	conv_loss�F<8�S        )��P	�nw���A�%*

	conv_loss
�;q-V        )��P	Uow���A�%*

	conv_loss�7�;f�5%        )��P	�\ow���A�%*

	conv_loss�?;'�#        )��P	��ow���A�%*

	conv_losstQ,;P��        )��P	��ow���A�%*

	conv_loss13<�5        )��P	T#pw���A�%*

	conv_loss�D�<���\        )��P	�fpw���A�%*

	conv_lossG�;<���        )��P	i�pw���A�%*

	conv_lossm�<�c        )��P	��pw���A�%*

	conv_loss���<�m��        )��P	��rw���A�%*

	conv_lossDU�<�~�U        )��P	+�rw���A�%*

	conv_loss�O\<���g        )��P	�
sw���A�%*

	conv_loss'1<���        )��P	�Fsw���A�%*

	conv_loss�Y%<�ۧc        )��P	�}sw���A�%*

	conv_lossz��<h�c�        )��P	��sw���A�%*

	conv_loss���;EN�        )��P	��sw���A�%*

	conv_loss���;6Q�        )��P	vBtw���A�%*

	conv_loss�Ŏ;'�B        )��P	�{tw���A�%*

	conv_lossY�S<�QM        )��P	�tw���A�%*

	conv_lossL~<!y        )��P	��tw���A�%*

	conv_loss��;yY��        )��P	�)uw���A�%*

	conv_loss���<F���        )��P	�_uw���A�%*

	conv_loss�x;)��h        )��P	��uw���A�%*

	conv_loss�=<~:        )��P	v�uw���A�%*

	conv_lossw[k<��F�        )��P	�vw���A�%*

	conv_loss�J�;Vh�        )��P	<vw���A�%*

	conv_loss�<�2�X        )��P	L{vw���A�%*

	conv_loss>c$<y���        )��P	�vw���A�%*

	conv_loss���;O��        )��P	��vw���A�%*

	conv_loss@��;���        )��P	�&ww���A�%*

	conv_lossgd4<��z�        )��P	I^ww���A�%*

	conv_loss|<o�j�        )��P	d�ww���A�%*

	conv_lossh��<��V        )��P	�ww���A�%*

	conv_loss�c�<�S!�        )��P	q�ww���A�%*

	conv_loss�H<N�        )��P	�4xw���A�%*

	conv_loss��:�u;D        )��P	�lxw���A�%*

	conv_lossu�F<p�(l        )��P	��xw���A�%*

	conv_loss3C�;U��        )��P	d�xw���A�%*

	conv_lossɈ(<Αp�        )��P	vyw���A�%*

	conv_loss�a<��1�        )��P	IIyw���A�%*

	conv_loss��-<T��z        )��P	S�yw���A�%*

	conv_loss9�;#V	�        )��P	d�yw���A�%*

	conv_loss�0a<�#        )��P	��yw���A�%*

	conv_loss ��;���        )��P	�#zw���A�%*

	conv_loss��\; F�v        )��P	�Yzw���A�%*

	conv_loss�~�;�Tm        )��P	y�zw���A�%*

	conv_loss���<���        )��P	��zw���A�%*

	conv_loss�<Z���        )��P	4�zw���A�%*

	conv_loss�[=��#�        )��P	N/{w���A�%*

	conv_loss�ޖ<h��K        )��P	�d{w���A�%*

	conv_loss�H<���        )��P	P�{w���A�%*

	conv_loss�n<ḿV        )��P	*�{w���A�%*

	conv_loss&C�<\��/        )��P	|w���A�%*

	conv_lossE��;�I�        )��P	�<|w���A�%*

	conv_loss���;U�        )��P	St|w���A�%*

	conv_loss�:;��        )��P	�|w���A�%*

	conv_loss���;�ҩx        )��P	F�|w���A�%*

	conv_lossQ�=	�u�        )��P	_}w���A�%*

	conv_lossĂ<v�        )��P	�M}w���A�%*

	conv_lossx��;�h�        )��P	��}w���A�%*

	conv_loss�F<��x�        )��P	�}w���A�%*

	conv_loss���;�Y�        )��P	�~w���A�%*

	conv_loss��;�CvY        )��P	E=~w���A�%*

	conv_loss#<�R�        )��P	�r~w���A�%*

	conv_lossnIJ<��q        )��P	N�~w���A�%*

	conv_loss8�;C�        )��P	��~w���A�%*

	conv_loss���< �:        )��P	Hw���A�%*

	conv_lossXu�<�eaX        )��P	�Rw���A�%*

	conv_lossw�;y�        )��P	��w���A�%*

	conv_loss�@H<�H        )��P	�w���A�%*

	conv_loss��<�*A        )��P	5�w���A�%*

	conv_lossЛ&<͝�8        )��P	J0�w���A�%*

	conv_loss+�4;�k�d        )��P	 g�w���A�&*

	conv_loss��<�(��        )��P	血w���A�&*

	conv_lossr1�;�Lm�        )��P	�ڀw���A�&*

	conv_loss�@<����        )��P	i�w���A�&*

	conv_loss��<U��        )��P	J�w���A�&*

	conv_loss�?<e!��        )��P	5��w���A�&*

	conv_loss��<4���        )��P	���w���A�&*

	conv_lossf <�ˬ�        )��P	}�w���A�&*

	conv_loss�S�;�X�@        )��P	s%�w���A�&*

	conv_loss��
<�\�U        )��P	�Z�w���A�&*

	conv_loss�SK;�g��        )��P	���w���A�&*

	conv_loss��R;�$�q        )��P	�ʂw���A�&*

	conv_loss��<��ԣ        )��P	b�w���A�&*

	conv_lossGʱ<�Hݵ        )��P	�9�w���A�&*

	conv_loss�<����        )��P	Qo�w���A�&*

	conv_loss���;��        )��P	ߧ�w���A�&*

	conv_loss�;b+�C        )��P	�߃w���A�&*

	conv_loss�(C<[8n\        )��P	[�w���A�&*

	conv_loss�97<�#��        )��P	�L�w���A�&*

	conv_lossP�:<�vMi        )��P	ӂ�w���A�&*

	conv_loss���;6(<\        )��P	���w���A�&*

	conv_loss?��;[
�        )��P	��w���A�&*

	conv_loss,��:�g�A        )��P	1#�w���A�&*

	conv_loss቗;*�<�        )��P	5Z�w���A�&*

	conv_loss���;zV�        )��P	4��w���A�&*

	conv_loss���<2�uk        )��P	�ƅw���A�&*

	conv_loss�N*<[JjA        )��P	y��w���A�&*

	conv_loss��;�k�        )��P	�4�w���A�&*

	conv_lossT��<1��t        )��P	xj�w���A�&*

	conv_loss��P;��E�        )��P	�w���A�&*

	conv_loss���<����        )��P	&نw���A�&*

	conv_loss�[�;iwX        )��P	�w���A�&*

	conv_lossK�y<L,g�        )��P	�B�w���A�&*

	conv_loss(�;y�.8        )��P	bw�w���A�&*

	conv_loss���;�gc        )��P	���w���A�&*

	conv_loss��;l��        )��P	'�w���A�&*

	conv_loss�z+<���
        )��P	f�w���A�&*

	conv_lossŧK<��/        )��P	Ih�w���A�&*

	conv_loss�#*<�w��        )��P	}��w���A�&*

	conv_lossX�<�ig5        )��P	�ֈw���A�&*

	conv_loss9�:;���Q        )��P	m�w���A�&*

	conv_loss�g<����        )��P	9@�w���A�&*

	conv_lossm�;���S        )��P	Zv�w���A�&*

	conv_loss�ղ;ʁN�        )��P	��w���A�&*

	conv_loss��;k��W        )��P	��w���A�&*

	conv_loss���<;Ϝ�        )��P	�!�w���A�&*

	conv_loss#�<-�P        )��P	�h�w���A�&*

	conv_loss��;��D        )��P	ʥ�w���A�&*

	conv_loss�ݎ;/O�        )��P	X�w���A�&*

	conv_losss�z<���        )��P	��w���A�&*

	conv_loss���;��Y        )��P	YQ�w���A�&*

	conv_loss7��;�/�r        )��P	0��w���A�&*

	conv_lossn��:�@�        )��P	�ċw���A�&*

	conv_loss]
<^���        )��P	���w���A�&*

	conv_loss=�;��#        )��P	�5�w���A�&*

	conv_lossj�;�� .        )��P	{v�w���A�&*

	conv_loss><ҁ�y        )��P	䭌w���A�&*

	conv_lossA�<�̘        )��P	B�w���A�&*

	conv_losso�	<����        )��P	�(�w���A�&*

	conv_lossjp<�]�        )��P	�h�w���A�&*

	conv_loss�d�;��        )��P	���w���A�&*

	conv_lossь@<oUo&        )��P	�ԍw���A�&*

	conv_lossoVN<����        )��P	�
�w���A�&*

	conv_loss�e�<rY�x        )��P	A�w���A�&*

	conv_loss8��: �?3        )��P		x�w���A�&*

	conv_lossOd�;+��        )��P	P��w���A�&*

	conv_lossŠ8<�7#        )��P	��w���A�&*

	conv_lossF�<Ʀ�E        )��P	,�w���A�&*

	conv_loss��*<)3QO        )��P	�P�w���A�&*

	conv_loss��
<�ø        )��P	���w���A�&*

	conv_losssO<� e�        )��P	0��w���A�&*

	conv_loss�EZ<�r)        )��P	��w���A�&*

	conv_loss�:<rwy        )��P	F(�w���A�&*

	conv_loss��V;?�73        )��P	Z]�w���A�&*

	conv_loss�k<��        )��P	[��w���A�&*

	conv_lossO��;&�        )��P	Ȑw���A�&*

	conv_loss���<���        )��P	���w���A�&*

	conv_loss2�<�#[        )��P	 5�w���A�&*

	conv_loss�T�;���        )��P	�j�w���A�&*

	conv_loss���;|_�        )��P	��w���A�&*

	conv_loss�� <M��        )��P	�ّw���A�&*

	conv_loss��?<���        )��P	�w���A�&*

	conv_loss-�;�A�        )��P	�F�w���A�&*

	conv_loss�I�<�P��        )��P	�|�w���A�&*

	conv_loss��<>p
	        )��P	+��w���A�&*

	conv_lossb'�<���        )��P	�w���A�&*

	conv_lossT
�<�uI�        )��P	"�w���A�&*

	conv_lossʯ<���        )��P	�n�w���A�&*

	conv_loss=��;Ԟ�        )��P	��w���A�&*

	conv_loss=j�;�S        )��P	rړw���A�&*

	conv_loss���;0�`F        )��P	��w���A�&*

	conv_loss"��;�5�        )��P	�L�w���A�&*

	conv_loss��<���        )��P	���w���A�&*

	conv_loss�L<u��p        )��P	/ʔw���A�&*

	conv_lossY}�<j�g        )��P	��w���A�&*

	conv_loss��V<���7        )��P	�D�w���A�&*

	conv_loss�n�;�/�h        )��P	�|�w���A�&*

	conv_loss�p<辞�        )��P	�ĕw���A�&*

	conv_loss���;��x        )��P	���w���A�&*

	conv_lossF<�d��        )��P	92�w���A�&*

	conv_loss^�;!a        )��P	ti�w���A�&*

	conv_lossl�'<@~ua        )��P	j��w���A�&*

	conv_loss{�;�W�E        )��P	�Ӗw���A�&*

	conv_loss���;�^��        )��P	,�w���A�&*

	conv_lossx� <�ҧ�        )��P	K�w���A�&*

	conv_loss[��;�8"e        )��P	���w���A�&*

	conv_loss��<Wu��        )��P	tėw���A�&*

	conv_loss�,;Χy�        )��P	���w���A�&*

	conv_loss�;U�>g        )��P	M2�w���A�&*

	conv_lossh)V<.6��        )��P	�i�w���A�&*

	conv_loss��;<��o5        )��P	�w���A�&*

	conv_loss�h;�c6{        )��P	.ؘw���A�&*

	conv_loss(�<�cW        )��P	��w���A�&*

	conv_loss�t)<��޽        )��P	E�w���A�&*

	conv_loss�\i<�h�#        )��P	�z�w���A�&*

	conv_loss�ޯ;�]
0        )��P	��w���A�&*

	conv_loss�� <����        )��P	d�w���A�&*

	conv_lossc�<���        )��P	�w���A�&*

	conv_loss]�:�ҵ�        )��P	U�w���A�&*

	conv_lossB<<��S�        )��P	��w���A�&*

	conv_loss��`<Ke�9        )��P	���w���A�&*

	conv_loss)�z<��8$        )��P	��w���A�&*

	conv_loss"��:/l�D        )��P	�*�w���A�&*

	conv_loss���<%�;�        )��P	Y`�w���A�&*

	conv_lossB5<�saQ        )��P	2��w���A�&*

	conv_loss�v<���        )��P	�˛w���A�&*

	conv_lossw�	<,��        )��P	��w���A�&*

	conv_loss���<j�#�        )��P	�8�w���A�&*

	conv_loss;���        )��P	�p�w���A�'*

	conv_loss�kM<K�;�        )��P	���w���A�'*

	conv_loss�\.<��އ        )��P	?ڜw���A�'*

	conv_loss} �;�3�        )��P	��w���A�'*

	conv_loss���<)�t�        )��P	�G�w���A�'*

	conv_loss�Y�;�_�>        )��P	�w���A�'*

	conv_losshV<5�\        )��P	쳝w���A�'*

	conv_lossv",<�        )��P	��w���A�'*

	conv_loss㤍<���r        )��P	-�w���A�'*

	conv_lossI�;t�        )��P	З�w���A�'*

	conv_loss�*�<+v�        )��P	�Ǥw���A�'*

	conv_loss>=0<~�&        )��P	H��w���A�'*

	conv_loss���:�@�        )��P	�&�w���A�'*

	conv_loss���;3HA�        )��P	yZ�w���A�'*

	conv_loss���;����        )��P	�w���A�'*

	conv_lossj �;�D�L        )��P	�w���A�'*

	conv_loss��';�{�b        )��P	w�w���A�'*

	conv_loss�U�;�4̂        )��P	4�w���A�'*

	conv_loss˛;Z��        )��P	�_�w���A�'*

	conv_loss�d�<FP#        )��P	+��w���A�'*

	conv_loss��<�6��        )��P	�¦w���A�'*

	conv_loss�_;�ل(        )��P	=��w���A�'*

	conv_loss��\<�p{        )��P	�#�w���A�'*

	conv_loss�j<,        )��P	z`�w���A�'*

	conv_loss�<唀�        )��P	���w���A�'*

	conv_loss�+<CH��        )��P	˾�w���A�'*

	conv_loss��t<+I��        )��P	���w���A�'*

	conv_loss
�R<\6��        )��P	�w���A�'*

	conv_loss�Ҧ;��؏        )��P	�K�w���A�'*

	conv_loss�iA<a�        )��P	�{�w���A�'*

	conv_loss���;���        )��P	ٯ�w���A�'*

	conv_loss|/<�)ډ        )��P	A�w���A�'*

	conv_lossA^<��        )��P	{�w���A�'*

	conv_loss#�;rCl�        )��P	'J�w���A�'*

	conv_losst�;���l        )��P	�z�w���A�'*

	conv_loss���;���        )��P	ĩ�w���A�'*

	conv_loss��v<���        )��P	%٩w���A�'*

	conv_losse��;'�S_        )��P	��w���A�'*

	conv_loss� �;��        )��P	�4�w���A�'*

	conv_loss8<���        )��P	�a�w���A�'*

	conv_loss/H*<e1n3        )��P	y��w���A�'*

	conv_loss+>m;Y�        )��P	#��w���A�'*

	conv_loss̡<gA��        )��P	��w���A�'*

	conv_loss�Gk;�am�        )��P	e�w���A�'*

	conv_lossT�;��o�        )��P	J�w���A�'*

	conv_loss<<���        )��P	Wy�w���A�'*

	conv_loss�+<�ͮ=        )��P	J��w���A�'*

	conv_loss�{;�n�        )��P	�ثw���A�'*

	conv_loss���;�)
        )��P	��w���A�'*

	conv_loss|<9�@        )��P	 8�w���A�'*

	conv_loss�9�;����        )��P	�e�w���A�'*

	conv_loss�:�;��f        )��P	g��w���A�'*

	conv_loss�q<�=}        )��P	�ìw���A�'*

	conv_loss�2�<��J�        )��P	��w���A�'*

	conv_loss�v|;�ԵZ        )��P	y �w���A�'*

	conv_loss��;���H        )��P	:O�w���A�'*

	conv_loss	��;��R�        )��P	��w���A�'*

	conv_loss��<��2        )��P	���w���A�'*

	conv_lossS*<s��        )��P	<�w���A�'*

	conv_lossj��;{�7U        )��P	�$�w���A�'*

	conv_loss�e�;؏rH        )��P	�T�w���A�'*

	conv_loss��<���/        )��P	S��w���A�'*

	conv_lossE��;�i�        )��P	��w���A�'*

	conv_loss)�`<<4<�        )��P	Z�w���A�'*

	conv_lossZ�;��        )��P	��w���A�'*

	conv_lossi�V<n'�        )��P	$N�w���A�'*

	conv_loss)<ڏ�d        )��P	$��w���A�'*

	conv_loss��<X�	        )��P	�Ưw���A�'*

	conv_loss��;�d�        )��P	:��w���A�'*

	conv_lossO�9<�21�        )��P	�0�w���A�'*

	conv_loss��j<��Z�        )��P	�j�w���A�'*

	conv_loss��;ݣ�        )��P	��w���A�'*

	conv_loss���;��        )��P	 �w���A�'*

	conv_lossf�	<e-��        )��P	��w���A�'*

	conv_loss���;���        )��P	�R�w���A�'*

	conv_loss��<�~Qd        )��P	��w���A�'*

	conv_loss�	�;��u        )��P	$��w���A�'*

	conv_loss��<���        )��P	l��w���A�'*

	conv_loss��;U�        )��P	� �w���A�'*

	conv_lossOk<p�"Q        )��P	 V�w���A�'*

	conv_lossn��;s<�        )��P	��w���A�'*

	conv_lossE��;�¹        )��P	A��w���A�'*

	conv_loss��<n'U        )��P	f�w���A�'*

	conv_loss�!<���        )��P	v$�w���A�'*

	conv_loss8�;�/$�        )��P	X�w���A�'*

	conv_loss(b�<R�;        )��P	.��w���A�'*

	conv_lossgm3<lJ5        )��P	6��w���A�'*

	conv_loss\{<"�SF        )��P	y�w���A�'*

	conv_lossۊ];f���        )��P	t%�w���A�'*

	conv_loss�b�;���        )��P	�X�w���A�'*

	conv_loss}@<;x�'        )��P	)��w���A�'*

	conv_loss�x6<l%         )��P	���w���A�'*

	conv_lossŠ�<S�XZ        )��P	$�w���A�'*

	conv_lossR�;�I��        )��P	n&�w���A�'*

	conv_loss�Y<���l        )��P	�Y�w���A�'*

	conv_loss@<2H�x        )��P	���w���A�'*

	conv_lossD��;+���        )��P	��w���A�'*

	conv_lossY�<:��k        )��P	���w���A�'*

	conv_loss��l<ǘ	        )��P	�)�w���A�'*

	conv_loss�o�<$KO&        )��P	K^�w���A�'*

	conv_loss3�;�}��        )��P	㏶w���A�'*

	conv_loss�i�<���        )��P	���w���A�'*

	conv_loss��<��Eg        )��P	5��w���A�'*

	conv_loss�Po;bt�_        )��P	*�w���A�'*

	conv_loss�<<�5�        )��P	�^�w���A�'*

	conv_lossQRJ<
���        )��P	`��w���A�'*

	conv_loss�E<uH�y        )��P	{÷w���A�'*

	conv_loss"�9<���        )��P	~��w���A�'*

	conv_loss�>�;1��X        )��P	�'�w���A�'*

	conv_loss6P<9:��        )��P	Zl�w���A�'*

	conv_lossN�<N�6        )��P	k��w���A�'*

	conv_lossX<o�$�        )��P	1Ըw���A�'*

	conv_lossV� <I=��        )��P	�w���A�'*

	conv_loss��<���<        )��P	pA�w���A�'*

	conv_losshB{<×��        )��P	{u�w���A�'*

	conv_loss��^;Н4        )��P	©�w���A�'*

	conv_loss��;Ъ*        )��P	��w���A�'*

	conv_lossԒ�<l��S        )��P	��w���A�'*

	conv_loss
<��%        )��P	NQ�w���A�'*

	conv_loss@��<I@�        )��P	퉺w���A�'*

	conv_lossI8{;W'�$        )��P	1��w���A�'*

	conv_lossNd�<�w\g        )��P	���w���A�'*

	conv_lossU�
<x۝        )��P	[2�w���A�'*

	conv_lossR
�<CߟW        )��P	g�w���A�'*

	conv_loss��<�)_&        )��P	��w���A�'*

	conv_loss"�;�>W)        )��P	�ϻw���A�'*

	conv_loss��<���        )��P	��w���A�'*

	conv_loss=�<;���        )��P	�7�w���A�'*

	conv_loss���;�s�        )��P	�i�w���A�(*

	conv_loss�ҋ<�{        )��P	!��w���A�(*

	conv_loss�!<��Ų        )��P	Gͼw���A�(*

	conv_losseS�<�>5         )��P	��w���A�(*

	conv_loss��;�GS        )��P	t8�w���A�(*

	conv_loss
�<��*        )��P	�l�w���A�(*

	conv_lossp#<u4o        )��P	��w���A�(*

	conv_loss�v�;�[�W        )��P	�ҽw���A�(*

	conv_loss���;1i��        )��P	n�w���A�(*

	conv_loss�<8��        )��P	�:�w���A�(*

	conv_loss�90<��R~        )��P	An�w���A�(*

	conv_loss���;��B�        )��P	꠾w���A�(*

	conv_loss���;�yP        )��P	�Ӿw���A�(*

	conv_loss�:�;�&�k        )��P	��w���A�(*

	conv_loss�J;l)�        )��P	�8�w���A�(*

	conv_loss"�_;�S�        )��P	�j�w���A�(*

	conv_loss�w�<B�        )��P	w���A�(*

	conv_loss-\w<�o}l        )��P	?пw���A�(*

	conv_loss�3<d<nx        )��P	��w���A�(*

	conv_lossK�<a��        )��P	
7�w���A�(*

	conv_lossO��;��?d        )��P	�l�w���A�(*

	conv_loss�o�;C�	M        )��P	��w���A�(*

	conv_lossDٱ;�Mx�        )��P	��w���A�(*

	conv_loss�K�;X�!�        )��P	�w���A�(*

	conv_loss6<��E�        )��P	�8�w���A�(*

	conv_lossl�;��X�        )��P	Nj�w���A�(*

	conv_loss��2<�#s        )��P	ܞ�w���A�(*

	conv_loss�݂<JM��        )��P	���w���A�(*

	conv_loss�W<"�9        )��P	:�w���A�(*

	conv_loss�,�;��S�        )��P	:�w���A�(*

	conv_loss��3;���        )��P	�m�w���A�(*

	conv_loss~�;d��J        )��P	J��w���A�(*

	conv_loss9�w<���        )��P	���w���A�(*

	conv_loss�<
�z        )��P	��w���A�(*

	conv_loss�m�;3��        )��P	GT�w���A�(*

	conv_loss/D�<��F        )��P	��w���A�(*

	conv_loss:5<���        )��P	p��w���A�(*

	conv_loss�[<��-        )��P	}��w���A�(*

	conv_lossy�J<{��H        )��P	'+�w���A�(*

	conv_loss=��;G2�H        )��P	x`�w���A�(*

	conv_lossf��;��G        )��P	���w���A�(*

	conv_loss9!�:�F        )��P	���w���A�(*

	conv_loss�h];֚;�        )��P	��w���A�(*

	conv_losso�Q<��܉        )��P	�G�w���A�(*

	conv_loss�<:M=�        )��P	�z�w���A�(*

	conv_loss��<�O        )��P	��w���A�(*

	conv_lossӀ�<|��        )��P	-��w���A�(*

	conv_lossYy<X=)�        )��P	�w���A�(*

	conv_loss�N�;�If        )��P	�E�w���A�(*

	conv_loss&�;W[�        )��P	�w�w���A�(*

	conv_loss7y�:Y�B        )��P	:��w���A�(*

	conv_loss�k<,�'        )��P	���w���A�(*

	conv_loss�.�<�nN        )��P	��w���A�(*

	conv_lossXIm<c�]        )��P	R�w���A�(*

	conv_loss�><6q?�        )��P	���w���A�(*

	conv_lossJ�<�wWj        )��P	���w���A�(*

	conv_lossFB<̑p�        )��P	���w���A�(*

	conv_loss���<w�T�        )��P	�!�w���A�(*

	conv_loss���;�� '        )��P	SU�w���A�(*

	conv_loss#2<B��        )��P	���w���A�(*

	conv_loss/�;�o��        )��P	��w���A�(*

	conv_loss!�&<���        )��P	���w���A�(*

	conv_loss�R#<*S~�        )��P	��w���A�(*

	conv_loss{�g;C�{        )��P	T�w���A�(*

	conv_lossҍ�<��C�        )��P	��w���A�(*

	conv_loss�A�;Kh:t        )��P	��w���A�(*

	conv_loss���;p�ZB        )��P	���w���A�(*

	conv_loss��/<�?Y�        )��P	/#�w���A�(*

	conv_loss���;Y#0        )��P	�V�w���A�(*

	conv_loss7.<�%Y        )��P	��w���A�(*

	conv_loss��;L�"�        )��P	���w���A�(*

	conv_loss���;t.        )��P	z��w���A�(*

	conv_lossMI�;2S%�        )��P	I"�w���A�(*

	conv_loss���;���        )��P	�T�w���A�(*

	conv_loss �;5�OY        )��P	���w���A�(*

	conv_loss`�;L=��        )��P	H��w���A�(*

	conv_loss�r<��        )��P	a��w���A�(*

	conv_lossw��;6�-�        )��P	#�w���A�(*

	conv_loss�,<�ϊ         )��P	sV�w���A�(*

	conv_loss�%#<�]h        )��P	���w���A�(*

	conv_loss���;�q�2        )��P	z��w���A�(*

	conv_loss>�<�'�	        )��P	SR�w���A�(*

	conv_lossn�`<{�#        )��P	Z��w���A�(*

	conv_lossN��;S6P7        )��P	/��w���A�(*

	conv_lossb�G<l��        )��P	���w���A�(*

	conv_loss���; J��        )��P	[�w���A�(*

	conv_loss,x�<k���        )��P	4K�w���A�(*

	conv_loss��/<�i�<        )��P	��w���A�(*

	conv_loss7h�<��#        )��P	��w���A�(*

	conv_loss2��;���        )��P	���w���A�(*

	conv_lossR�&<Nkk        )��P	�(�w���A�(*

	conv_lossY�5<")�        )��P	�X�w���A�(*

	conv_lossOv<�@        )��P	���w���A�(*

	conv_lossX�<���        )��P	��w���A�(*

	conv_lossރ<��%o        )��P	D��w���A�(*

	conv_lossA^7;�(��        )��P	O*�w���A�(*

	conv_loss^1N<�&��        )��P	f_�w���A�(*

	conv_loss='<d��         )��P	_��w���A�(*

	conv_loss�Q&<�P�        )��P	���w���A�(*

	conv_lossO�@<����        )��P	x�w���A�(*

	conv_loss�;����        )��P	B5�w���A�(*

	conv_loss篋;���        )��P	�d�w���A�(*

	conv_loss�q<C��        )��P	q��w���A�(*

	conv_loss��;I�Ѥ        )��P	���w���A�(*

	conv_loss�<���z        )��P	���w���A�(*

	conv_lossJ�<O�u�        )��P	�+�w���A�(*

	conv_lossfP�;��"        )��P	�\�w���A�(*

	conv_loss�a\<&�3        )��P	��w���A�(*

	conv_loss��<�Ğ        )��P	��w���A�(*

	conv_loss{E8<��o�        )��P	���w���A�(*

	conv_loss9��;g��^        )��P	��w���A�(*

	conv_loss�i<_*
�        )��P	=R�w���A�(*

	conv_loss�U�<"�i�        )��P	J��w���A�(*

	conv_loss/�<OP�        )��P	C��w���A�(*

	conv_lossp^�;+��        )��P	���w���A�(*

	conv_loss��<\���        )��P	��w���A�(*

	conv_loss�?�;J��        )��P	�O�w���A�(*

	conv_loss� <!��        )��P	���w���A�(*

	conv_loss�/�<Q��        )��P	n��w���A�(*

	conv_lossd/<Ŧ�        )��P	���w���A�(*

	conv_loss��<a        )��P	��w���A�(*

	conv_loss�<%���        )��P	�T�w���A�(*

	conv_loss9O�;���8        )��P	Ɉ�w���A�(*

	conv_lossz�<朞�        )��P	>��w���A�(*

	conv_lossim"<n⒥        )��P	8��w���A�(*

	conv_loss��<<�',        )��P	^#�w���A�(*

	conv_loss�
&<���        )��P	LS�w���A�(*

	conv_loss}#;R̡~        )��P	D��w���A�(*

	conv_lossL!<GK�b        )��P	j��w���A�)*

	conv_lossE�]<�b��        )��P	���w���A�)*

	conv_lossGg0<B��8        )��P	j�w���A�)*

	conv_loss6%.<'l        )��P	&^�w���A�)*

	conv_loss`�<\�Q.        )��P	���w���A�)*

	conv_loss�T<�        )��P	M��w���A�)*

	conv_loss�<a=l        )��P	���w���A�)*

	conv_loss��W<hL��        )��P	�!�w���A�)*

	conv_loss�j�:���        )��P	:T�w���A�)*

	conv_loss��;<9�_        )��P	ȃ�w���A�)*

	conv_loss�fD<T�ާ        )��P	��w���A�)*

	conv_lossm9�;d��        )��P	���w���A�)*

	conv_loss���;.��(        )��P	C)�w���A�)*

	conv_loss���<~���        )��P	4Z�w���A�)*

	conv_loss�U<�	P        )��P	���w���A�)*

	conv_lossa��;��        )��P	W��w���A�)*

	conv_loss:T;m���        )��P	-��w���A�)*

	conv_loss�(�;!Q��        )��P	.�w���A�)*

	conv_loss�B�;�r/G        )��P	�c�w���A�)*

	conv_loss�w<�f��        )��P	��w���A�)*

	conv_loss'��;�8/&        )��P	���w���A�)*

	conv_loss�J
<�-40        )��P	��w���A�)*

	conv_loss6l;�go        )��P	�)�w���A�)*

	conv_loss���;tH�        )��P	]�w���A�)*

	conv_loss�.�;e.$        )��P	M��w���A�)*

	conv_loss=<)���        )��P	���w���A�)*

	conv_loss3�`;�P�1        )��P	���w���A�)*

	conv_loss��;��$6        )��P	�/�w���A�)*

	conv_loss?<2�$�        )��P	W`�w���A�)*

	conv_lossh��<���        )��P	v��w���A�)*

	conv_loss���<�̈́        )��P	���w���A�)*

	conv_lossY��;�.�.        )��P	��w���A�)*

	conv_lossL�h<;	��        )��P	;$�w���A�)*

	conv_loss���;��-�        )��P	fR�w���A�)*

	conv_lossΚ�;�?x&        )��P	��w���A�)*

	conv_loss��<�ZH�        )��P	��w���A�)*

	conv_lossM��:�ce�        )��P	���w���A�)*

	conv_loss�A�;þ�        )��P	��w���A�)*

	conv_loss@�<�0��        )��P	F�w���A�)*

	conv_loss�!<����        )��P	@u�w���A�)*

	conv_lossx`�<�ht�        )��P	Φ�w���A�)*

	conv_loss�C�<ǧ��        )��P	���w���A�)*

	conv_lossߘ<ۂp^        )��P	��w���A�)*

	conv_loss�<\�^6        )��P	7�w���A�)*

	conv_loss�A<��+        )��P	�f�w���A�)*

	conv_loss���;���        )��P	���w���A�)*

	conv_loss]�I<J�B�        )��P	���w���A�)*

	conv_loss�$';t��o        )��P	���w���A�)*

	conv_loss�ն;�9�E        )��P	0�w���A�)*

	conv_loss"~<3���        )��P	�`�w���A�)*

	conv_loss�O<xY�n        )��P	���w���A�)*

	conv_lossp�@;[�0M        )��P	���w���A�)*

	conv_losscΈ<�g        )��P	���w���A�)*

	conv_lossы<�Fm�        )��P	9�w���A�)*

	conv_loss�`R<hY��        )��P	kj�w���A�)*

	conv_loss"��;'.�        )��P	o��w���A�)*

	conv_lossβ<3�l�        )��P	!��w���A�)*

	conv_loss��;s���        )��P	��w���A�)*

	conv_lossn�<u��        )��P	�>�w���A�)*

	conv_loss�;v��0        )��P	@q�w���A�)*

	conv_loss��)<K��         )��P	���w���A�)*

	conv_loss���;X�,�        )��P	���w���A�)*

	conv_lossmh�;*Lm        )��P	j�w���A�)*

	conv_loss咽;�v�)        )��P	bE�w���A�)*

	conv_loss�#�;^�        )��P	�u�w���A�)*

	conv_loss��B<�Q�        )��P	��w���A�)*

	conv_loss��<.�~-        )��P	���w���A�)*

	conv_loss�\�<J�;�        )��P		�w���A�)*

	conv_loss�	><M�4        )��P	�H�w���A�)*

	conv_loss2��;�|��        )��P	�~�w���A�)*

	conv_loss�<<^D>        )��P	%��w���A�)*

	conv_lossr��;����        )��P	��w���A�)*

	conv_lossMƁ<�5�        )��P	��w���A�)*

	conv_loss�U/<6d�        )��P	�F�w���A�)*

	conv_lossu;�;�d�        )��P	Ox�w���A�)*

	conv_loss~?�<��(d        )��P	���w���A�)*

	conv_loss��<��ua        )��P	���w���A�)*

	conv_loss�p<���        )��P	��w���A�)*

	conv_loss�1�;��        )��P	iF�w���A�)*

	conv_loss�m<T1�\        )��P	�z�w���A�)*

	conv_loss?e�<�v(        )��P	f��w���A�)*

	conv_loss4G�;�vQ        )��P	���w���A�)*

	conv_loss	�;���        )��P	��w���A�)*

	conv_loss7�M<�b!        )��P	�=�w���A�)*

	conv_loss���;	�7�        )��P	Ro�w���A�)*

	conv_lossU9A;"�c        )��P	(��w���A�)*

	conv_lossǪ�;��8        )��P	r��w���A�)*

	conv_loss;[�;���8        )��P	h�w���A�)*

	conv_loss��<B�        )��P	�6�w���A�)*

	conv_losso��;��kZ        )��P	*m�w���A�)*

	conv_loss�u$<��YW        )��P	���w���A�)*

	conv_loss���:N�        )��P	-��w���A�)*

	conv_lossC]�<r�6        )��P	�w���A�)*

	conv_loss--<ӷ        )��P	�1�w���A�)*

	conv_loss��<eϻ�        )��P	c�w���A�)*

	conv_loss� ,<�QpA        )��P	���w���A�)*

	conv_lossx��;G��        )��P	l��w���A�)*

	conv_lossJ�;�	�        )��P	���w���A�)*

	conv_loss���;�.��        )��P	z*�w���A�)*

	conv_lossc�;�H�        )��P	�^�w���A�)*

	conv_loss:��;l;��        )��P	���w���A�)*

	conv_loss/�<�{�*        )��P	j��w���A�)*

	conv_loss|��;�+Ī        )��P	y��w���A�)*

	conv_lossi�s<W?~        )��P	.>�w���A�)*

	conv_loss�rC<�e��        )��P	�o�w���A�)*

	conv_loss�@<���        )��P	D��w���A�)*

	conv_lossnŋ;�=`�        )��P	���w���A�)*

	conv_loss��;��C        )��P	��w���A�)*

	conv_lossZ4�<�~;        )��P	j<�w���A�)*

	conv_loss�*< �=        )��P	bw�w���A�)*

	conv_lossw��<���W        )��P	��w���A�)*

	conv_lossS]<!���        )��P	q��w���A�)*

	conv_loss��!<�v!H        )��P	)�w���A�)*

	conv_lossX�;ܭ�v        )��P	�H�w���A�)*

	conv_loss.�;�
        )��P	ف�w���A�)*

	conv_loss� 1<ܸ�        )��P	���w���A�)*

	conv_loss�F<蓧�        )��P	���w���A�)*

	conv_loss�,
<7�|        )��P	��w���A�)*

	conv_loss��m<�X{        )��P	LF�w���A�)*

	conv_lossV�<��,        )��P	P}�w���A�)*

	conv_loss�>�<�AA�        )��P	���w���A�)*

	conv_loss�u<c��o        )��P	��w���A�)*

	conv_loss��;��        )��P	��w���A�)*

	conv_loss߾|<Hu�        )��P	�G�w���A�)*

	conv_loss4�<%�-�        )��P	R|�w���A�)*

	conv_loss�<F�b        )��P	{��w���A�)*

	conv_loss�C]<�|�O        )��P	u��w���A�)*

	conv_loss�;��/�        )��P	Z�w���A�)*

	conv_loss��Y;p|<        )��P	�C�w���A�**

	conv_loss);�*��        )��P	1v�w���A�**

	conv_lossɇ�<���S        )��P	���w���A�**

	conv_loss�<+ ��        )��P	���w���A�**

	conv_loss�AS<�	        )��P	v�w���A�**

	conv_loss�?t<1�:�        )��P	{<�w���A�**

	conv_loss�s;�i�`        )��P	en�w���A�**

	conv_loss;�<�?oK        )��P	���w���A�**

	conv_loss��j<��B�        )��P	`��w���A�**

	conv_loss�R<�[��        )��P	B�w���A�**

	conv_loss��;�~jC        )��P	�3�w���A�**

	conv_loss%�<�4        )��P	�d�w���A�**

	conv_lossq�I;歙�        )��P	���w���A�**

	conv_loss��?<�g�6        )��P	���w���A�**

	conv_lossf�;<,v�S        )��P	/��w���A�**

	conv_loss��=<%�`�        )��P	q.�w���A�**

	conv_lossn��;<���        )��P	+_�w���A�**

	conv_lossf�<��d        )��P	���w���A�**

	conv_loss��3<��)        )��P	��w���A�**

	conv_losss�<a;@        )��P	C��w���A�**

	conv_loss92<}�]        )��P	�%�w���A�**

	conv_lossԘ6<7I�=        )��P	OZ�w���A�**

	conv_loss?�(<��q�        )��P	��w���A�**

	conv_loss�$8<�%�s        )��P	w��w���A�**

	conv_loss��;U�        )��P	���w���A�**

	conv_loss �D<�1�        )��P	K��w���A�**

	conv_loss�<<��%        )��P	��w���A�**

	conv_lossZ�o<6|��        )��P	y��w���A�**

	conv_loss��p<CQ8�        )��P	M�w���A�**

	conv_loss���;�{ч        )��P	WP�w���A�**

	conv_lossR.�;d���        )��P	���w���A�**

	conv_loss7x:<F"�        )��P	���w���A�**

	conv_lossG7�;�]�        )��P	|��w���A�**

	conv_lossrV<XI��        )��P	�&�w���A�**

	conv_loss*��<F�{w        )��P	 _�w���A�**

	conv_loss�YA;9�+K        )��P	6��w���A�**

	conv_lossw8{<T�4`        )��P	���w���A�**

	conv_loss��
<m�#�        )��P	���w���A�**

	conv_loss��<�L�        )��P	
+�w���A�**

	conv_loss�7�;�Pm        )��P	E[�w���A�**

	conv_lossp�;�'        )��P	���w���A�**

	conv_lossG~�;Q���        )��P	���w���A�**

	conv_loss�X<.M&        )��P	��w���A�**

	conv_loss��O<2n��        )��P	v�w���A�**

	conv_lossƻ<�K��        )��P	�K�w���A�**

	conv_loss�6�;ѣ�8        )��P	�{�w���A�**

	conv_loss杻;�y��        )��P	���w���A�**

	conv_loss9<�22�        )��P	��w���A�**

	conv_loss¡�<���        )��P	��w���A�**

	conv_loss5I�<��        )��P	0L�w���A�**

	conv_loss�0�<���k        )��P	�{�w���A�**

	conv_loss�"0<��9�        )��P	��w���A�**

	conv_loss5�<C��        )��P	)��w���A�**

	conv_lossb<(/�x        )��P	��w���A�**

	conv_loss�><[�Mj        )��P	PB�w���A�**

	conv_loss�p;Ż�n        )��P	�s�w���A�**

	conv_loss=�s;6�=�        )��P	ڢ�w���A�**

	conv_loss�i< ���        )��P	;��w���A�**

	conv_loss%��;���        )��P	l�w���A�**

	conv_loss��;!�-�        )��P	)4�w���A�**

	conv_lossm��;���        )��P	�f�w���A�**

	conv_loss�K;�,w"        )��P	t��w���A�**

	conv_lossY�;/��        )��P	���w���A�**

	conv_loss���</{w        )��P	��w���A�**

	conv_loss���;e�o�        )��P	�9�w���A�**

	conv_loss�=<��$"        )��P	ck�w���A�**

	conv_loss�z�;�F�        )��P	��w���A�**

	conv_loss�`:;�t;�        )��P	I��w���A�**

	conv_loss���<�<�        )��P	H��w���A�**

	conv_loss��<����        )��P	�( x���A�**

	conv_loss���;�� �        )��P	W x���A�**

	conv_losso�<O&H1        )��P	� x���A�**

	conv_loss�<k�Y�        )��P	�� x���A�**

	conv_loss��<`��*        )��P	U� x���A�**

	conv_loss��	<_�e        )��P	�x���A�**

	conv_loss#|<u.�        )��P	u\x���A�**

	conv_loss���;NS�
        )��P	<�x���A�**

	conv_loss��;�;\        )��P	�x���A�**

	conv_lossB"<àZ�        )��P	`�x���A�**

	conv_lossi,W<��I        )��P	:x���A�**

	conv_loss[�<��)F        )��P	�Ox���A�**

	conv_loss��x;J8X        )��P	�~x���A�**

	conv_loss">,<Q���        )��P	�x���A�**

	conv_loss-�;��        )��P	��x���A�**

	conv_lossR�J<%�P        )��P	�x���A�**

	conv_lossÇ$<pZ�        )��P	PEx���A�**

	conv_loss�M,<�k^H        )��P	�x���A�**

	conv_loss�;��5�        )��P	;�x���A�**

	conv_lossK��;k�?        )��P	�x���A�**

	conv_loss�*�:�(�e        )��P	�x���A�**

	conv_lossy#W;%��0        )��P	�Lx���A�**

	conv_loss��<�M        )��P	�{x���A�**

	conv_loss�d=<?3�        )��P	9�x���A�**

	conv_loss�d�;�~        )��P	�x���A�**

	conv_lossv�;��\        )��P	^x���A�**

	conv_loss#T6<O=�        )��P	H9x���A�**

	conv_loss7�!<���'        )��P	xhx���A�**

	conv_lossÚ�;;��N        )��P	��x���A�**

	conv_loss]��;=0C�        )��P	��x���A�**

	conv_loss��q<N %�        )��P	�x���A�**

	conv_loss1R;�Ix�        )��P	/*x���A�**

	conv_loss"<< kX        )��P	]Yx���A�**

	conv_loss֊�<)\#        )��P	��x���A�**

	conv_loss�Z�<!��        )��P	i�x���A�**

	conv_loss�R�;R,�        )��P	��x���A�**

	conv_lossEӢ;SQ��        )��P	x���A�**

	conv_loss��<�W��        )��P	fIx���A�**

	conv_loss{1<�[J=        )��P	vyx���A�**

	conv_loss�(:<q�I        )��P	��x���A�**

	conv_loss�B<���        )��P	��x���A�**

	conv_loss$��; �yU        )��P	/x���A�**

	conv_lossz�v;�2��        )��P	�7x���A�**

	conv_lossfD6<ʒ(        )��P	�ex���A�**

	conv_loss�o<O;�        )��P	��x���A�**

	conv_loss�E<s��        )��P	��x���A�**

	conv_loss��{;�d{        )��P	��x���A�**

	conv_loss�H�<̦�        )��P	*'	x���A�**

	conv_loss+�;��=b        )��P	6V	x���A�**

	conv_loss�T�;A=K        )��P	��	x���A�**

	conv_loss̻<���        )��P	#�	x���A�**

	conv_loss�:�<�#�M        )��P	�	x���A�**

	conv_loss��;��~        )��P	& 
x���A�**

	conv_loss�u[<��o        )��P	FQ
x���A�**

	conv_loss�m�;��ct        )��P	��
x���A�**

	conv_loss��;M�A        )��P	*1x���A�**

	conv_lossՋ�:� o        )��P	Nsx���A�**

	conv_loss)�;](e        )��P	Ѡx���A�**

	conv_loss|�<@z�        )��P	J�x���A�**

	conv_loss#<6n         )��P	F�x���A�+*

	conv_loss�#F<�B�        )��P	3x���A�+*

	conv_lossuDk<��        )��P	�px���A�+*

	conv_loss�, <�o��        )��P	��x���A�+*

	conv_loss���;u��        )��P	��x���A�+*

	conv_loss��:�v4        )��P	�x���A�+*

	conv_loss�j<r�y        )��P	�0x���A�+*

	conv_loss��<�Kq        )��P	Iax���A�+*

	conv_loss�q�;041        )��P	�x���A�+*

	conv_loss(&4;�x	        )��P	�x���A�+*

	conv_lossN�;�O��        )��P	��x���A�+*

	conv_loss|<>���        )��P	�x���A�+*

	conv_loss��+<wO�        )��P	`x���A�+*

	conv_lossq��:wk�U        )��P	b�x���A�+*

	conv_loss�z#<�{��        )��P	�x���A�+*

	conv_loss���<PD�        )��P	��x���A�+*

	conv_loss|/j<ꓟ�        )��P	L,x���A�+*

	conv_loss�&�<Py�        )��P	F\x���A�+*

	conv_loss��;P�5g        )��P	M�x���A�+*

	conv_loss�<S8�<        )��P	��x���A�+*

	conv_loss�Bn;p��        )��P	[�x���A�+*

	conv_lossAz<mU�D        )��P	�x���A�+*

	conv_loss�2a<�!��        )��P	�Kx���A�+*

	conv_loss��,;c��:        )��P	�zx���A�+*

	conv_lossl�<��'        )��P	ݩx���A�+*

	conv_loss��;�s�z        )��P	��x���A�+*

	conv_loss�_<Za�S        )��P	�x���A�+*

	conv_loss�Z<o폿        )��P	67x���A�+*

	conv_loss .�<Y��        )��P	fx���A�+*

	conv_loss���;�6�I        )��P	ǔx���A�+*

	conv_lossZ(B<��{�        )��P	��x���A�+*

	conv_loss��<���        )��P	�x���A�+*

	conv_lossu�W;`q�d        )��P	f#x���A�+*

	conv_loss��1<f�H�        )��P	Sx���A�+*

	conv_lossIP�;�!*l        )��P	 �x���A�+*

	conv_loss�W�;	W��        )��P	��x���A�+*

	conv_loss�x;�c]        )��P	|�x���A�+*

	conv_loss�u�;��        )��P	x���A�+*

	conv_loss�k&<���        )��P	?x���A�+*

	conv_loss_{�;&�        )��P	�nx���A�+*

	conv_lossS9�;�#        )��P	0�x���A�+*

	conv_loss��e;V�?�        )��P	2�x���A�+*

	conv_lossWܻ;�w��        )��P	��x���A�+*

	conv_loss�l�<E˕{        )��P	�,x���A�+*

	conv_loss�Q�;?ín        )��P	F\x���A�+*

	conv_loss�A�;�4IY        )��P	�x���A�+*

	conv_loss�.<nf��        )��P	��x���A�+*

	conv_loss�;H<.��        )��P	��x���A�+*

	conv_loss�CL<�	�        )��P	o+x���A�+*

	conv_loss?Q<~�G        )��P	�Yx���A�+*

	conv_loss�;�;�^��        )��P	��x���A�+*

	conv_loss/p�;Ȟ\�        )��P	k�x���A�+*

	conv_loss,�&<��+        )��P	)�x���A�+*

	conv_lossk)�;��:        )��P	x���A�+*

	conv_lossZ�C<Ox��        )��P	Rx���A�+*

	conv_loss��;��Q`        )��P	?�x���A�+*

	conv_loss��;݋ְ        )��P	��x���A�+*

	conv_loss���;����        )��P	��x���A�+*

	conv_loss�d�;��        )��P	^&x���A�+*

	conv_loss�!&<�N        )��P	YSx���A�+*

	conv_loss�ٹ;+0��        )��P	��x���A�+*

	conv_loss�J<O��        )��P	ҿx���A�+*

	conv_loss�{h:�[3,        )��P	r�x���A�+*

	conv_loss��:��@�        )��P	�x���A�+*

	conv_loss�<<���,        )��P	0Ox���A�+*

	conv_lossO*�;����        )��P	�x���A�+*

	conv_loss��:��p        )��P	��x���A�+*

	conv_loss s�;�7�        )��P	�x���A�+*

	conv_losss�'<,�)        )��P	x���A�+*

	conv_loss��;��C�        )��P	PIx���A�+*

	conv_lossw� <���        )��P	yx���A�+*

	conv_loss)<��̚        )��P	y�x���A�+*

	conv_loss�u�;4��        )��P	��x���A�+*

	conv_loss<{�;l�i�        )��P	�x���A�+*

	conv_loss��;;���        )��P	�4x���A�+*

	conv_loss�C�;��X.        )��P	"dx���A�+*

	conv_loss#e;:�b        )��P	�x���A�+*

	conv_loss9�<�t�G        )��P	g�x���A�+*

	conv_loss�US<'��        )��P	��x���A�+*

	conv_loss-X<�|
        )��P	n%x���A�+*

	conv_loss��<�V�        )��P	�Ux���A�+*

	conv_loss��;�        )��P	�x���A�+*

	conv_loss��s<&<o        )��P	��x���A�+*

	conv_loss���;��        )��P	��x���A�+*

	conv_lossr��;>�Q        )��P	 x���A�+*

	conv_loss�4�:�h        )��P	XA x���A�+*

	conv_lossm&<����        )��P	�p x���A�+*

	conv_loss^-<xA�q        )��P	\� x���A�+*

	conv_loss
�K;|0Y�        )��P	�� x���A�+*

	conv_loss�b; p��        )��P	^� x���A�+*

	conv_loss"0<F�        )��P		+!x���A�+*

	conv_loss�G<�E<�        )��P	{Z!x���A�+*

	conv_lossmX<ɪ��        )��P	�!x���A�+*

	conv_loss5]e<o�,�        )��P	T�!x���A�+*

	conv_loss���<��        )��P	z�!x���A�+*

	conv_loss�<�`�        )��P	�"x���A�+*

	conv_loss�H<07d'        )��P	sL"x���A�+*

	conv_loss��<B�V0        )��P	M�#x���A�+*

	conv_loss�v;ƃ�0        )��P	�$x���A�+*

	conv_loss�9<�k�        )��P	*S$x���A�+*

	conv_lossH�;��n         )��P	�$x���A�+*

	conv_lossv�<�U�        )��P	w�$x���A�+*

	conv_loss��;lEZ        )��P	R�$x���A�+*

	conv_loss�);go�}        )��P	�%%x���A�+*

	conv_loss�&o<F?}        )��P	 U%x���A�+*

	conv_lossK<*�#G        )��P	P�%x���A�+*

	conv_loss
1�;�'�H        )��P	N�%x���A�+*

	conv_losskV�;��        )��P	��%x���A�+*

	conv_losstY;P��u        )��P	n%&x���A�+*

	conv_lossB�B;�}         )��P	W&x���A�+*

	conv_losslD<�п        )��P	i�&x���A�+*

	conv_loss���<1�         )��P	��&x���A�+*

	conv_loss��<<Z�9        )��P	��&x���A�+*

	conv_loss��;�U��        )��P	�$'x���A�+*

	conv_lossV�<��=�        )��P	]U'x���A�+*

	conv_loss_�<^؎         )��P	��'x���A�+*

	conv_lossTK-<�D]G        )��P	��'x���A�+*

	conv_loss��;P��        )��P	�'x���A�+*

	conv_loss�|;<|T�        )��P	�(x���A�+*

	conv_loss�m�;�l�v        )��P	LD(x���A�+*

	conv_loss��<���        )��P	�u(x���A�+*

	conv_lossзZ<Ʉ�         )��P	��(x���A�+*

	conv_loss7�:#��$        )��P	��(x���A�+*

	conv_lossa�@<��H�        )��P	�)x���A�+*

	conv_loss+��<q-O        )��P	�8)x���A�+*

	conv_loss�8x<��*�        )��P	Xg)x���A�+*

	conv_loss���<�6�        )��P	��)x���A�+*

	conv_loss�y*<�x��        )��P	�)x���A�+*

	conv_loss�j�;�̨�        )��P	��)x���A�,*

	conv_loss�UF<���        )��P	Z-*x���A�,*

	conv_loss�A�;����        )��P	�]*x���A�,*

	conv_loss4]R;�p�        )��P	"�*x���A�,*

	conv_loss1�p;����        )��P	μ*x���A�,*

	conv_loss�<<�^q�        )��P	��*x���A�,*

	conv_loss���;'��        )��P	�+x���A�,*

	conv_loss�I�;�jK�        )��P	�O+x���A�,*

	conv_loss��;��y�        )��P	�+x���A�,*

	conv_loss��W<���'        )��P	c�+x���A�,*

	conv_lossJ�;�u��        )��P	��+x���A�,*

	conv_lossƬ�;�Q��        )��P	g,x���A�,*

	conv_loss<;���        )��P	xA,x���A�,*

	conv_loss�� ;�q�W        )��P	sq,x���A�,*

	conv_loss|w�;�>Q�        )��P	u�,x���A�,*

	conv_loss�Y<Q3I        )��P	��,x���A�,*

	conv_lossW5\<�n��        )��P	>-x���A�,*

	conv_loss���;~��        )��P	3-x���A�,*

	conv_loss��<�\��        )��P	=f-x���A�,*

	conv_loss�֤;x��        )��P	6�-x���A�,*

	conv_loss� �<�s�S        )��P	�-x���A�,*

	conv_lossxg�;ڡb�        )��P	).x���A�,*

	conv_loss�O�;�0�$        )��P	
C.x���A�,*

	conv_loss��;�2d�        )��P	qt.x���A�,*

	conv_lossc;�;��V�        )��P	I�.x���A�,*

	conv_loss�ɨ<�(�        )��P	��.x���A�,*

	conv_loss:�;X���        )��P	u/x���A�,*

	conv_loss�M�<esfm        )��P	�B/x���A�,*

	conv_loss���;�j�E        )��P	v/x���A�,*

	conv_loss{�N<��r        )��P	ͦ/x���A�,*

	conv_loss.��;ﴓ>        )��P	?�/x���A�,*

	conv_lossN��; ���        )��P	�0x���A�,*

	conv_lossNY<�W��        )��P	780x���A�,*

	conv_loss�pR<�p$�        )��P	Ko0x���A�,*

	conv_loss�p�;�̼�        )��P	�0x���A�,*

	conv_loss�8;��B        )��P	��0x���A�,*

	conv_loss��<�=��        )��P	�1x���A�,*

	conv_loss�¯;>���        )��P	�91x���A�,*

	conv_lossP7�;��Y�        )��P	�m1x���A�,*

	conv_loss��];�K�        )��P	ʝ1x���A�,*

	conv_loss��+<���        )��P	��1x���A�,*

	conv_loss�p2<#arh        )��P	�2x���A�,*

	conv_lossՑ<�tC        )��P	�42x���A�,*

	conv_lossvw<����        )��P	ad2x���A�,*

	conv_lossS�\;�e�x        )��P	�2x���A�,*

	conv_loss���<J$�        )��P	��2x���A�,*

	conv_loss�rc<�=K        )��P	Z�2x���A�,*

	conv_loss�&<�J��        )��P	�#3x���A�,*

	conv_loss�,�;�K�        )��P	�R3x���A�,*

	conv_loss1�;m0%        )��P	�3x���A�,*

	conv_loss��<��+�        )��P	��3x���A�,*

	conv_loss}J;�c�        )��P	��3x���A�,*

	conv_lossh><)�H        )��P	w4x���A�,*

	conv_loss��B<���l        )��P	G4x���A�,*

	conv_loss^I�;ac��        )��P	�v4x���A�,*

	conv_loss;I<�e�        )��P	D�4x���A�,*

	conv_lossq�<O�&�        )��P	��4x���A�,*

	conv_lossU��<Tl��        )��P	�5x���A�,*

	conv_loss��<Z6�6        )��P	>35x���A�,*

	conv_lossO��;�R        )��P	c5x���A�,*

	conv_loss���<r�7        )��P	��5x���A�,*

	conv_lossUKC<t+�        )��P	��5x���A�,*

	conv_lossx��;�3f        )��P	��5x���A�,*

	conv_lossZ�;�K�y        )��P	�+6x���A�,*

	conv_lossO��;�|�#        )��P	�\6x���A�,*

	conv_lossvC�;���        )��P	B�6x���A�,*

	conv_lossC%�;��        )��P	��6x���A�,*

	conv_loss��;Q�        )��P	��6x���A�,*

	conv_loss|��;��?j        )��P	�7x���A�,*

	conv_loss�F�;����        )��P	�_7x���A�,*

	conv_loss��;̟�        )��P	"�7x���A�,*

	conv_loss��7<	�4�        )��P	��7x���A�,*

	conv_lossâ�;:�~&        )��P	B�7x���A�,*

	conv_loss�<n�G�        )��P	2&8x���A�,*

	conv_loss�^<�@��        )��P	"\8x���A�,*

	conv_loss�h�<d�u	        )��P	�8x���A�,*

	conv_loss�2<*/��        )��P	1�8x���A�,*

	conv_loss/ޔ;��l         )��P	
9x���A�,*

	conv_loss��=<8x��        )��P	�;9x���A�,*

	conv_loss��<�>�f        )��P	Zl9x���A�,*

	conv_loss;~<��F        )��P	��9x���A�,*

	conv_loss�@�;�        )��P	|�9x���A�,*

	conv_loss�(<<ң        )��P	v:x���A�,*

	conv_loss�_�<=�ڶ        )��P	W4:x���A�,*

	conv_loss2��;�R��        )��P	$c:x���A�,*

	conv_lossU�<�d�        )��P	�:x���A�,*

	conv_loss��;����        )��P	��:x���A�,*

	conv_lossi+G;-Y�]        )��P	g�:x���A�,*

	conv_loss]L�;����        )��P	�-;x���A�,*

	conv_loss�,�;��/%        )��P	.a;x���A�,*

	conv_loss�.�<[���        )��P	9�;x���A�,*

	conv_loss�^�;_�Y        )��P	!�;x���A�,*

	conv_lossŪ;G�&�        )��P	��;x���A�,*

	conv_loss��s<�0^        )��P	=%<x���A�,*

	conv_loss;N<�rl        )��P	^V<x���A�,*

	conv_loss�<��C        )��P	��<x���A�,*

	conv_lossE-<Eb�K        )��P	�<x���A�,*

	conv_lossp��<�u�2        )��P	R�<x���A�,*

	conv_loss��;����        )��P	�=x���A�,*

	conv_loss'�F;�x��        )��P	L=x���A�,*

	conv_lossy�;�R        )��P	}=x���A�,*

	conv_loss�D!<F:{        )��P	��=x���A�,*

	conv_loss�M�;m��,        )��P	'�=x���A�,*

	conv_loss�X2<1o!�        )��P	�
>x���A�,*

	conv_loss<��g�        )��P	9>x���A�,*

	conv_loss�1<<��O        )��P	Kj>x���A�,*

	conv_loss�]�;SΥ�        )��P	�>x���A�,*

	conv_lossN�<_o��        )��P	2�>x���A�,*

	conv_lossy8M<����        )��P	��>x���A�,*

	conv_loss+[�;�>[�        )��P	c&?x���A�,*

	conv_loss�(&<e�F�        )��P	�Y?x���A�,*

	conv_lossU�;��~�        )��P	��?x���A�,*

	conv_loss��*;b�z�        )��P	�?x���A�,*

	conv_loss<*�;ƀ��        )��P	�?x���A�,*

	conv_lossU�<ҝ|        )��P	�@x���A�,*

	conv_lossf�j<vF`        )��P	�K@x���A�,*

	conv_loss��X;�Ά�        )��P	|@x���A�,*

	conv_loss:b
<��V        )��P	��@x���A�,*

	conv_lossе6<�6|�        )��P	/�@x���A�,*

	conv_loss�r8<��        )��P	q"Ax���A�,*

	conv_loss��;���U        )��P	�QAx���A�,*

	conv_loss�n�:&1u        )��P	�Ax���A�,*

	conv_loss;+;�S�        )��P	:�Ax���A�,*

	conv_lossw6|<��&�        )��P	��Ax���A�,*

	conv_loss%(A<F���        )��P	�Bx���A�,*

	conv_loss�b<:�t�        )��P	GBx���A�,*

	conv_loss%^.<�d��        )��P	*}Bx���A�,*

	conv_loss�ԫ<����        )��P	�Bx���A�,*

	conv_loss�YS<�i{        )��P	�Bx���A�-*

	conv_loss�^�;�2W�        )��P	m,Cx���A�-*

	conv_lossV&3;_��        )��P	=^Cx���A�-*

	conv_lossq��:�WSf        )��P	A�Cx���A�-*

	conv_loss�e�<[�P        )��P	��Cx���A�-*

	conv_lossά;�2�        )��P	J�Cx���A�-*

	conv_lossHg�<��C        )��P	�*Dx���A�-*

	conv_loss�X1<ExD�        )��P	�\Dx���A�-*

	conv_loss�>�;Æ��        )��P	��Dx���A�-*

	conv_loss�)<���        )��P	�Dx���A�-*

	conv_loss J<nش        )��P	<�Dx���A�-*

	conv_loss��;���        )��P	�&Ex���A�-*

	conv_lossuq;,��        )��P	i^Ex���A�-*

	conv_lossv�<���        )��P	ԑEx���A�-*

	conv_loss7#<�o|        )��P	�Ex���A�-*

	conv_lossN��;��=�        )��P	�Ex���A�-*

	conv_loss��<2,е        )��P	�!Fx���A�-*

	conv_lossz-<�Es,        )��P	�RFx���A�-*

	conv_lossV��;�� R        )��P	��Fx���A�-*

	conv_lossc�<;���A        )��P	��Fx���A�-*

	conv_loss��;ǩ,        )��P	q�Fx���A�-*

	conv_loss�K7<[�ؖ        )��P	KGx���A�-*

	conv_loss��;9        )��P	�LGx���A�-*

	conv_loss�,<[�^�        )��P	�~Gx���A�-*

	conv_loss�I�;���        )��P	R�Gx���A�-*

	conv_loss��;�e        )��P	!�Gx���A�-*

	conv_loss]��;�GN        )��P	�Hx���A�-*

	conv_lossKdH=�j*�        )��P	rBHx���A�-*

	conv_loss�<H��        )��P	�sHx���A�-*

	conv_loss��:*        )��P		�Hx���A�-*

	conv_loss���;4�H�        )��P	��Hx���A�-*

	conv_lossǤ<>��        )��P	LIx���A�-*

	conv_loss�`�;B�}        )��P	�FIx���A�-*

	conv_lossӌ<)��        )��P	�wIx���A�-*

	conv_loss�<ֻ         )��P	<�Ix���A�-*

	conv_loss�u<����        )��P	�Ix���A�-*

	conv_lossh�.<v�Y        )��P	�Jx���A�-*

	conv_loss	�w;���K        )��P	�NJx���A�-*

	conv_loss6�<�G�        )��P	�Jx���A�-*

	conv_loss`��;��        )��P	�Jx���A�-*

	conv_loss�<��e�        )��P	��Jx���A�-*

	conv_loss��<nŨ�        )��P	>�Lx���A�-*

	conv_lossM�y;ז�!        )��P	�Lx���A�-*

	conv_loss�A<�s7�        )��P	��Lx���A�-*

	conv_loss�F><v��        )��P	eMx���A�-*

	conv_loss9�;��/A        )��P	�IMx���A�-*

	conv_loss&�<��>        )��P	�yMx���A�-*

	conv_loss���<N3        )��P	��Mx���A�-*

	conv_loss>��;�&^�        )��P	0�Mx���A�-*

	conv_loss�G;��t�        )��P	:Nx���A�-*

	conv_loss��<@�~        )��P	NINx���A�-*

	conv_lossC=;��s|        )��P	�zNx���A�-*

	conv_loss)x_;��^�        )��P	�Nx���A�-*

	conv_loss�;M�v�        )��P	W�Nx���A�-*

	conv_lossX;�1�        )��P	{Ox���A�-*

	conv_loss>��;:{�        )��P	ROx���A�-*

	conv_loss��;�ʃ�        )��P	��Ox���A�-*

	conv_loss�	<<��X        )��P	%�Ox���A�-*

	conv_loss��<�ģr        )��P	I�Ox���A�-*

	conv_lossX�<E���        )��P	GPx���A�-*

	conv_loss+�%<�|�        )��P	�ZPx���A�-*

	conv_loss�Ο;�ES        )��P	ʋPx���A�-*

	conv_loss�;�]�        )��P	��Px���A�-*

	conv_lossf
p;��M�        )��P	��Px���A�-*

	conv_loss��'<�o5�        )��P	�Qx���A�-*

	conv_loss�-<����        )��P	.GQx���A�-*

	conv_loss��<��`        )��P	o�Qx���A�-*

	conv_loss��=<��        )��P	s�Qx���A�-*

	conv_loss�;����        )��P	 �Qx���A�-*

	conv_loss���;f�+�        )��P	"Rx���A�-*

	conv_lossp�;;��        )��P	tBRx���A�-*

	conv_loss�<U%!I        )��P	`{Rx���A�-*

	conv_loss�*�;��        )��P	��Rx���A�-*

	conv_loss}�<�p��        )��P	��Rx���A�-*

	conv_loss.��;9�&�        )��P	SSx���A�-*

	conv_lossR��;���I        )��P	�6Sx���A�-*

	conv_loss�r�;�e�5        )��P	
tSx���A�-*

	conv_loss���:C�z)        )��P	��Sx���A�-*

	conv_loss��;[i��        )��P	��Sx���A�-*

	conv_loss!_:<"�_�        )��P	�Tx���A�-*

	conv_loss�<�:*��T        )��P	�3Tx���A�-*

	conv_loss���<'/        )��P	�cTx���A�-*

	conv_loss�NQ<�D�        )��P	њTx���A�-*

	conv_loss�<r�;        )��P	��Tx���A�-*

	conv_loss��x;r�7p        )��P	��Tx���A�-*

	conv_loss�(;t��5        )��P	�+Ux���A�-*

	conv_loss= �;�^�        )��P	�ZUx���A�-*

	conv_loss7Չ<��#        )��P	��Ux���A�-*

	conv_loss(�8;�Y��        )��P	��Ux���A�-*

	conv_loss���<�ƥ        )��P	��Ux���A�-*

	conv_loss��6<
R��        )��P	T%Vx���A�-*

	conv_loss��:�?��        )��P	�gVx���A�-*

	conv_lossD�,<3߈*        )��P	ҢVx���A�-*

	conv_lossQ`'<@��V        )��P	��Vx���A�-*

	conv_loss�du<��        )��P	�Wx���A�-*

	conv_loss�=-<rB�        )��P	R3Wx���A�-*

	conv_loss���;�G_W        )��P	�aWx���A�-*

	conv_lossѾ�;���        )��P	X�Wx���A�-*

	conv_loss!̀<�Z�        )��P	2�Wx���A�-*

	conv_lossD)<+�m�        )��P	Y	Xx���A�-*

	conv_loss��Q<Eڏ�        )��P	�>Xx���A�-*

	conv_loss��;�;\*        )��P	�pXx���A�-*

	conv_loss��;��        )��P	>�Xx���A�-*

	conv_lossTd�;���d        )��P	2�Xx���A�-*

	conv_loss#nW;s� �        )��P	�Yx���A�-*

	conv_loss�$<�o�        )��P	w>Yx���A�-*

	conv_lossQW;�?f        )��P	~sYx���A�-*

	conv_loss$V�;��0        )��P	�Yx���A�-*

	conv_lossw;i<�C�        )��P	��Yx���A�-*

	conv_lossgE<a��        )��P	YZx���A�-*

	conv_lossl��<����        )��P	�AZx���A�-*

	conv_loss�#�<���        )��P	JrZx���A�-*

	conv_loss=L�<U*��        )��P	W�Zx���A�-*

	conv_loss�i�;b|�E        )��P	U�Zx���A�-*

	conv_lossz�;�R��        )��P	I[x���A�-*

	conv_loss�1'<��|        )��P	�@[x���A�-*

	conv_loss�u�<��        )��P	q[x���A�-*

	conv_loss��%<�*7�        )��P	٠[x���A�-*

	conv_loss�$^<��7        )��P	��[x���A�-*

	conv_loss���;�7��        )��P	�\x���A�-*

	conv_lossAv;��
X        )��P	:\x���A�-*

	conv_loss=��;��M3        )��P	�j\x���A�-*

	conv_lossq��;0G,        )��P	c�\x���A�-*

	conv_loss�d�;~��        )��P	x�\x���A�-*

	conv_loss�k�;�j�        )��P	 ]x���A�-*

	conv_loss��;���        )��P	4]x���A�-*

	conv_lossf��;����        )��P	�f]x���A�-*

	conv_loss�m�;�t        )��P	.�]x���A�-*

	conv_loss��;���        )��P	G�]x���A�.*

	conv_loss�.�;���        )��P	�]x���A�.*

	conv_lossa�k<���        )��P	)^x���A�.*

	conv_loss��e<�[0V        )��P	�[^x���A�.*

	conv_loss�;���H        )��P	��^x���A�.*

	conv_lossf@<Ȏ��        )��P	m�^x���A�.*

	conv_loss���;��=        )��P	E�^x���A�.*

	conv_lossÇ<�=�        )��P	(_x���A�.*

	conv_lossQ�2<��)        )��P	�L_x���A�.*

	conv_loss��'<�k        )��P	�~_x���A�.*

	conv_lossƖ�;d��4        )��P	+�_x���A�.*

	conv_losseM;�Q{        )��P	t�_x���A�.*

	conv_loss�m2<�h��        )��P	T`x���A�.*

	conv_loss8,<QO��        )��P	�U`x���A�.*

	conv_loss��<�	�g        )��P	�`x���A�.*

	conv_loss%bp<�L�        )��P	��`x���A�.*

	conv_loss��<�Y�        )��P	:�`x���A�.*

	conv_loss���<�:9        )��P	Aax���A�.*

	conv_lossT�<�/]�        )��P	�Oax���A�.*

	conv_loss3=<�Bɳ        )��P	"�ax���A�.*

	conv_loss�B�;U���        )��P	ոax���A�.*

	conv_loss�S?<�s�        )��P	q�ax���A�.*

	conv_loss���;���        )��P	x&bx���A�.*

	conv_loss� <�]��        )��P	�Vbx���A�.*

	conv_loss�zd<,���        )��P	��bx���A�.*

	conv_lossQ�	<Ы�        )��P	�bx���A�.*

	conv_losss�><�>7�        )��P	�bx���A�.*

	conv_loss���;�?S(        )��P	&cx���A�.*

	conv_loss
< )�1        )��P	�[cx���A�.*

	conv_losse��<�ͥ        )��P	/�cx���A�.*

	conv_loss�a4<���        )��P	9�cx���A�.*

	conv_loss��;���        )��P	y�cx���A�.*

	conv_lossDX`:���        )��P	�)dx���A�.*

	conv_loss8��;��L        )��P	b[dx���A�.*

	conv_loss�[;OGS        )��P	o�dx���A�.*

	conv_loss��;�n�I        )��P	�dx���A�.*

	conv_loss_<]�         )��P	F�dx���A�.*

	conv_loss��;�4�P        )��P	#ex���A�.*

	conv_loss���;�CY        )��P	-Rex���A�.*

	conv_loss4@:�O	�        )��P	؅ex���A�.*

	conv_loss�U6;4�t        )��P	�ex���A�.*

	conv_loss9��;���        )��P	��ex���A�.*

	conv_loss�t<�(��        )��P	9fx���A�.*

	conv_losssW(;�        )��P	JNfx���A�.*

	conv_loss��;k�        )��P	D~fx���A�.*

	conv_loss�u<���4        )��P	�fx���A�.*

	conv_loss�/<�T�+        )��P	��fx���A�.*

	conv_loss�ӑ;����        )��P	�gx���A�.*

	conv_loss��o<��z         )��P	�Agx���A�.*

	conv_loss=g�;��a�        )��P	�sgx���A�.*

	conv_loss�*�;_)~?        )��P	H�gx���A�.*

	conv_loss���<Z��        )��P	��gx���A�.*

	conv_loss�;
�        )��P	4	hx���A�.*

	conv_lossJ=f<<��        )��P	m:hx���A�.*

	conv_loss�L\<|���        )��P	Dlhx���A�.*

	conv_loss��;3�u        )��P	�hx���A�.*

	conv_loss�a<�c�Q        )��P	��hx���A�.*

	conv_lossq�w;�.�J        )��P	uix���A�.*

	conv_loss50<W�        )��P	`5ix���A�.*

	conv_lossꠋ<�g�        )��P	iix���A�.*

	conv_loss�߄;��        )��P	�ix���A�.*

	conv_loss���;#tG        )��P	��ix���A�.*

	conv_lossdO<�аQ        )��P	� jx���A�.*

	conv_loss0�C<)�]        )��P	�Kjx���A�.*

	conv_loss�uc;����        )��P	�}jx���A�.*

	conv_loss}��;iX�U        )��P	հjx���A�.*

	conv_loss���;��mc        )��P	�jx���A�.*

	conv_lossȞ<��
%        )��P	�kx���A�.*

	conv_loss�'<�б�        )��P	�Zkx���A�.*

	conv_loss�}�;�um        )��P	Ɍkx���A�.*

	conv_loss���;�o�        )��P	��kx���A�.*

	conv_losswܖ<'�        )��P	�kx���A�.*

	conv_loss��;'O        )��P	�(lx���A�.*

	conv_loss�z�<|8        )��P	B`lx���A�.*

	conv_loss~9n<e�p}        )��P	��lx���A�.*

	conv_losspy<�2Վ        )��P	��lx���A�.*

	conv_losswݔ<��cO        )��P	�lx���A�.*

	conv_loss��<����        )��P	p)mx���A�.*

	conv_loss�%
<X"i�        )��P	Jamx���A�.*

	conv_losse��;��Ģ        )��P	k�mx���A�.*

	conv_loss�<��s        )��P	�mx���A�.*

	conv_loss��k;�Ff�        )��P	enx���A�.*

	conv_loss5o�<&�u�        )��P	5nx���A�.*

	conv_loss��<�c*"        )��P	�enx���A�.*

	conv_lossEN.<���R        )��P	{�nx���A�.*

	conv_loss�NW<M\��        )��P	��nx���A�.*

	conv_loss��I<���        )��P	w�nx���A�.*

	conv_lossZ�<���        )��P	�0ox���A�.*

	conv_loss	��;�伐        )��P	6cox���A�.*

	conv_loss�x�;�=8�        )��P	ox���A�.*

	conv_loss��;����        )��P	��ox���A�.*

	conv_loss�'4<��        )��P	��ox���A�.*

	conv_loss&�;fN�        )��P	I*px���A�.*

	conv_loss�y�<N��        )��P	�^px���A�.*

	conv_loss���<����        )��P	��px���A�.*

	conv_losso#<D ��        )��P	�px���A�.*

	conv_loss<�<Z| 2        )��P	�px���A�.*

	conv_loss�<���        )��P	�*qx���A�.*

	conv_loss��;ڽ�        )��P	�]qx���A�.*

	conv_loss�{<�@�~        )��P	ۑqx���A�.*

	conv_lossee<x�/�        )��P	��qx���A�.*

	conv_lossx;���        )��P	?�qx���A�.*

	conv_lossݷq;�B%        )��P	�&rx���A�.*

	conv_loss��;�No%        )��P	(Yrx���A�.*

	conv_lossꁕ;i�m�        )��P	C�rx���A�.*

	conv_loss�f<�Y�6        )��P	ƿrx���A�.*

	conv_loss  �;(ft        )��P	��rx���A�.*

	conv_loss��1<V-K        )��P	u$sx���A�.*

	conv_loss.;Dg�        )��P	`Wsx���A�.*

	conv_loss�;�X[[        )��P	 �sx���A�.*

	conv_loss"r�;��        )��P	m�sx���A�.*

	conv_loss'�;x��        )��P	��sx���A�.*

	conv_loss���<����