       �K"	  @J���Abrain.Event:2Q{4���      D(�	��mJ���A"��
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
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*&
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2d/kernel*
validate_shape(
�
conv2d/kernel/readIdentityconv2d/kernel*&
_output_shapes
:*
T0* 
_class
loc:@conv2d/kernel
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
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
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_1/kernel
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
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
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
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
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@conv2d_3/kernel*%
valueB"            *
dtype0
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
conv2d_3/kernel/AssignAssignconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
conv2d_3/kernel/readIdentityconv2d_3/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_3/kernel
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
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_4/kernel*%
valueB"            
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
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
T0
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
dtype0*
_output_shapes
:*
valueB"      
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
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:*
T0
�
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
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
8conv2d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_6/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_6/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
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
conv2d_7/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
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
ReshapeReshapeRelu_6Reshape/shape*
Tshape0*(
_output_shapes
:����������
*
T0
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
+dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *���=*
dtype0
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
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d
�
dense/kernel
VariableV2*
	container *
shape:	�
d*
dtype0*
_output_shapes
:	�
d*
shared_name *
_class
loc:@dense/kernel
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	�
d*
use_locking(*
T0
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
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
�
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes

:d
*
T0
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
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:

q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
:
*
T0
�
dense_2/MatMulMatMulRelu_7dense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
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
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:���������

n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*'
_output_shapes
:���������
*
T0
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
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
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
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
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
:*
	keep_dims( *

Tidx0
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
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*'
_output_shapes
:���������
*
T0
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
&gradients/logistic_loss/mul_grad/mul_1Muldense_2/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������
*
T0
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
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�
d
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
%gradients/conv2d_7/Conv2D_grad/ShapeNShapeNRelu_5conv2d_6/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
�
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
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
9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_6/Conv2D_grad/tuple/group_deps*F
_class<
:8loc:@gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
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
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNRelu_2conv2d_3/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
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
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
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
 *
ף<
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
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
use_locking( 
�
;GradientDescent/update_conv2d_5/kernel/ApplyGradientDescentApplyGradientDescentconv2d_5/kernelGradientDescent/learning_rate9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:*
use_locking( 
�
;GradientDescent/update_conv2d_6/kernel/ApplyGradientDescentApplyGradientDescentconv2d_6/kernelGradientDescent/learning_rate9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_6/kernel
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
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
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
Mean_1MeanCastConst_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: "sdu2�      ?�K	ԔnJ���AJ��
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
.conv2d/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:* 
_class
loc:@conv2d/kernel*%
valueB"            *
dtype0
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
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
T0
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
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel*%
valueB"            
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

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 *
dtype0*&
_output_shapes
:
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
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_1/kernel*
	container 
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
dtype0*
_output_shapes
:*
valueB"      
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
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_2/kernel*
	container 
�
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
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

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 *
dtype0*&
_output_shapes
:
�
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
T0
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
VariableV2*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_3/kernel*
	container *
shape:*
dtype0
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
Relu_3Reluconv2d_4/Conv2D*
T0*/
_output_shapes
:���������
�
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_4/kernel*%
valueB"            
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
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_4/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
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
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_5/kernel
�
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
�
conv2d_5/kernel
VariableV2*"
_class
loc:@conv2d_5/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
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
conv2d_6/Conv2DConv2DRelu_4conv2d_5/kernel/read*/
_output_shapes
:���������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
.conv2d_6/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_6/kernel*
valueB
 *���=
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
.conv2d_6/kernel/Initializer/random_uniform/mulMul8conv2d_6/kernel/Initializer/random_uniform/RandomUniform.conv2d_6/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
�
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
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
Relu_6Reluconv2d_7/Conv2D*
T0*/
_output_shapes
:���������
^
Reshape/shapeConst*
_output_shapes
:*
valueB"����\  *
dtype0
j
ReshapeReshapeRelu_6Reshape/shape*
Tshape0*(
_output_shapes
:����������
*
T0
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"\  d   *
dtype0
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
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�
d
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
VariableV2*
	container *
shape:	�
d*
dtype0*
_output_shapes
:	�
d*
shared_name *
_class
loc:@dense/kernel
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	�
d*
use_locking(
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
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*'
_output_shapes
:���������
*
T0*
data_formatNHWC
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
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*'
_output_shapes
:���������
*
T0
V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
`
MeanMeanlogistic_lossConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
gradients/Mean_grad/Shape_1Shapelogistic_loss*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
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
: *
	keep_dims( *

Tidx0*
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
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
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
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
�
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
�
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:���������
*
T0
�
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:���������
*
T0
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
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
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
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape
�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1
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
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*'
_output_shapes
:���������
*
T0
�
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*'
_output_shapes
:���������
*
T0
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
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
&gradients/logistic_loss/mul_grad/mul_1Muldense_2/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������
*
T0
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
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:���������
*
T0
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
6gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_2/MatMul_grad/MatMul/^gradients/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*7
_class-
+)loc:@gradients/dense_2/MatMul_grad/MatMul
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
gradients/Reshape_grad/ShapeShapeRelu_6*
out_type0*
_output_shapes
:*
T0
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
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*E
_class;
97loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
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
7gradients/conv2d_6/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_6/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������
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
%gradients/conv2d_5/Conv2D_grad/ShapeNShapeNRelu_3conv2d_4/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
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
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/Relu_3_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_4/Conv2D_grad/tuple/group_deps*F
_class<
:8loc:@gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
�
gradients/Relu_2_grad/ReluGradReluGrad7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyRelu_2*
T0*/
_output_shapes
:���������
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0
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
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
GradientDescent/learning_rateConst*
valueB
 *
ף<*
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
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentApplyGradientDescentconv2d_3/kernelGradientDescent/learning_rate9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:*
use_locking( *
T0
�
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_4/kernel
�
;GradientDescent/update_conv2d_5/kernel/ApplyGradientDescentApplyGradientDescentconv2d_5/kernelGradientDescent/learning_rate9gradients/conv2d_6/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:*
use_locking( *
T0
�
;GradientDescent/update_conv2d_6/kernel/ApplyGradientDescentApplyGradientDescentconv2d_6/kernelGradientDescent/learning_rate9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:*
use_locking( 
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
:GradientDescent/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelGradientDescent/learning_rate8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
_output_shapes

:d
*
use_locking( *
T0
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
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
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
Mean_1MeanCastConst_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: ""
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
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"�
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

conv_loss:0�[�s       `/�#	b�J���A*

	conv_loss�Y1?c٨       QKD	���J���A*

	conv_losssE1?��       QKD	 �J���A*

	conv_loss_%1?�zj�       QKD	(�J���A*

	conv_lossB1?�^M       QKD	5O�J���A*

	conv_lossD�0?z�v       QKD	���J���A*

	conv_loss��0?���       QKD	 ��J���A*

	conv_loss��0?���       QKD	G�J���A*

	conv_loss�0?�OAZ       QKD	�'�J���A*

	conv_loss��0?qS>V       QKD	H`�J���A	*

	conv_loss��0?uR�A       QKD	���J���A
*

	conv_loss��0?�91�       QKD	�ʠJ���A*

	conv_loss�p0?���       QKD	O��J���A*

	conv_losswX0?�=�       QKD	u1�J���A*

	conv_loss�E0?�s�%       QKD	b�J���A*

	conv_loss,0?SZ8       QKD	y��J���A*

	conv_loss�0?S�u       QKD	uġJ���A*

	conv_loss<0?$y�M       QKD	_��J���A*

	conv_lossY�/?��g       QKD	�(�J���A*

	conv_losso�/?�ф       QKD	w[�J���A*

	conv_losss�/?9���       QKD	O��J���A*

	conv_lossĵ/?���       QKD	�ƢJ���A*

	conv_lossJ�/?D�{C       QKD	���J���A*

	conv_loss�/?���       QKD	2-�J���A*

	conv_losswv/?��|       QKD	�]�J���A*

	conv_loss�Z/?=�?       QKD	⍣J���A*

	conv_loss"E/?~n��       QKD	�ƣJ���A*

	conv_loss�//?z0�       QKD	���J���A*

	conv_loss�/?5	+�       QKD	�+�J���A*

	conv_loss�
/?��]:       QKD	�]�J���A*

	conv_loss�.?��G       QKD	���J���A*

	conv_loss��.?CG#�       QKD	��J���A*

	conv_loss��.?��Y�       QKD	4�J���A *

	conv_lossN�.?"��       QKD	�)�J���A!*

	conv_loss��.?؂�f       QKD	�Y�J���A"*

	conv_losso�.?�7       QKD	֊�J���A#*

	conv_loss�x.?鑽�       QKD	���J���A$*

	conv_loss�^.?nu�h       QKD	��J���A%*

	conv_loss�K.?���       QKD	��J���A&*

	conv_loss�1.?uR�&       QKD	sZ�J���A'*

	conv_loss�.?�"�Q       QKD		��J���A(*

	conv_loss?.?�DJ�       QKD	D��J���A)*

	conv_loss�-?c���       QKD	�J���A**

	conv_loss��-?���       QKD	8�J���A+*

	conv_loss��-?�T�       QKD	�Q�J���A,*

	conv_lossg�-?��k�       QKD	���J���A-*

	conv_loss��-?��D       QKD	ΰ�J���A.*

	conv_losse�-?���
       QKD	��J���A/*

	conv_loss/w-?Q�'�       QKD	��J���A0*

	conv_loss�m-?*A�       QKD	uA�J���A1*

	conv_lossV-?�G       QKD	�p�J���A2*

	conv_loss"@-?�F�       QKD	���J���A3*

	conv_loss�0-?D�Sr       QKD	��J���A4*

	conv_loss�-?T�V       QKD	��J���A5*

	conv_loss��,?��       QKD	oI�J���A6*

	conv_loss��,?'�1�       QKD	���J���A7*

	conv_loss"�,?ܨ�       QKD	���J���A8*

	conv_loss��,?(�r       QKD	��J���A9*

	conv_lossD�,?�0       QKD	n �J���A:*

	conv_loss��,?I���       QKD	a^�J���A;*

	conv_loss"�,?b>�       QKD	m��J���A<*

	conv_loss�t,?��I       QKD	�ʪJ���A=*

	conv_losssf,?[o6       QKD	P��J���A>*

	conv_loss�L,? y�       QKD	�-�J���A?*

	conv_loss3,?�HT�       QKD	y_�J���A@*

	conv_loss$,?3��B       QKD	I��J���AA*

	conv_lossE,?�2�"       QKD	�ǫJ���AB*

	conv_loss��+?��       QKD	A��J���AC*

	conv_loss��+?��
       QKD	/�J���AD*

	conv_lossd�+?�2�H       QKD	>`�J���AE*

	conv_loss��+?1��       QKD	=��J���AF*

	conv_loss�+?�24�       QKD	�ϬJ���AG*

	conv_lossd�+?j,��       QKD	U�J���AH*

	conv_lossf�+?�       QKD	�1�J���AI*

	conv_loss�i+?��&u       QKD	�a�J���AJ*

	conv_lossWX+?���O       QKD	���J���AK*

	conv_lossbC+?ֶ�       QKD	0��J���AL*

	conv_loss�2+?`s��       QKD	��J���AM*

	conv_loss+?7�0       QKD	X�J���AN*

	conv_loss_+?7܉       QKD	�O�J���AO*

	conv_loss��*?�<*�       QKD	W��J���AP*

	conv_loss��*?����       QKD	%��J���AQ*

	conv_loss��*?��V�       QKD	��J���AR*

	conv_loss�*?���~       QKD	�J���AS*

	conv_loss��*?~��`       QKD	RF�J���AT*

	conv_loss]�*?�;C       QKD	�u�J���AU*

	conv_loss��*?��       QKD	ޱ�J���AV*

	conv_loss�q*?_φ$       QKD	�J���AW*

	conv_lossZ]*?���       QKD	E�J���AX*

	conv_loss[B*?Ү�S       QKD	&E�J���AY*

	conv_loss8*?�[_       QKD	�u�J���AZ*

	conv_loss�+*?*j       QKD	��J���A[*

	conv_loss�*?��,k       QKD	�J���A\*

	conv_loss��)?[��M       QKD	9�J���A]*

	conv_loss]�)?���       QKD	�A�J���A^*

	conv_lossT�)?)#�       QKD	�s�J���A_*

	conv_loss>�)?9i�s       QKD	���J���A`*

	conv_lossԪ)?�>"       QKD	�ױJ���Aa*

	conv_loss��)?�&�       QKD	�J���Ab*

	conv_loss~�)?�6��       QKD	s;�J���Ac*

	conv_loss�k)?�d�       QKD	�m�J���Ad*

	conv_loss�_)?�$�       QKD	H��J���Ae*

	conv_loss�C)?%\��       QKD	0�J���Af*

	conv_lossp.)?��       QKD	��J���Ag*

	conv_loss�&)?"���       QKD	l@�J���Ah*

	conv_loss�)?�6�q       QKD	u�J���Ai*

	conv_lossE�(?}��       QKD	���J���Aj*

	conv_loss,�(?��e       QKD	��J���Ak*

	conv_loss�(?�1�       QKD	I�J���Al*

	conv_loss
�(?{6       QKD	�S�J���Am*

	conv_lossի(?�\��       QKD	Љ�J���An*

	conv_loss��(?ʿ�       QKD	�ôJ���Ao*

	conv_loss�(?:aP�       QKD	���J���Ap*

	conv_loss�w(?�9σ       QKD	�&�J���Aq*

	conv_loss,[(?�ΥH       QKD	�V�J���Ar*

	conv_lossM(?P�bA       QKD	=��J���As*

	conv_loss�8(?դ�R       QKD	'��J���At*

	conv_loss�"(?�H��       QKD	��J���Au*

	conv_loss�(?�jk1       QKD	��J���Av*

	conv_loss��'?��3/       QKD	 H�J���Aw*

	conv_loss�'?��G�       QKD	&z�J���Ax*

	conv_loss��'?�'��       QKD	q¶J���Ay*

	conv_loss��'?�-��       QKD	���J���Az*

	conv_loss=�'?Ü
�       QKD	�)�J���A{*

	conv_loss��'?E�J       QKD	�Y�J���A|*

	conv_loss9}'?��(\       QKD	��J���A}*

	conv_loss�t'?cO�       QKD	���J���A~*

	conv_lossoi'?���       QKD	D��J���A*

	conv_lossP'?�,�D        )��P	��J���A�*

	conv_loss�8'?�>��        )��P	�O�J���A�*

	conv_lossV('?��        )��P	��J���A�*

	conv_loss'?C��B        )��P	j��J���A�*

	conv_loss'?M-�        )��P	
�J���A�*

	conv_lossB�&?�s:�        )��P	M�J���A�*

	conv_loss��&?M��        )��P	"B�J���A�*

	conv_loss��&?γ�q        )��P	�r�J���A�*

	conv_lossʿ&?����        )��P	]��J���A�*

	conv_lossΝ&?/P�I        )��P	rӹJ���A�*

	conv_lossÍ&?g��D        )��P	��J���A�*

	conv_lossK{&??��        )��P	�4�J���A�*

	conv_loss�o&?�s&L        )��P	�c�J���A�*

	conv_loss�[&?,iܗ        )��P	 ��J���A�*

	conv_loss�A&?B�v        )��P	�ºJ���A�*

	conv_loss�.&?;K�y        )��P	��J���A�*

	conv_loss� &?;9a        )��P	h �J���A�*

	conv_loss�&?=�         )��P	�O�J���A�*

	conv_loss=�%?4i9        )��P	��J���A�*

	conv_loss��%?�G        )��P	箻J���A�*

	conv_loss;�%?lt�P        )��P	o޻J���A�*

	conv_lossH�%?'��?        )��P	{�J���A�*

	conv_loss��%?|��        )��P	�A�J���A�*

	conv_loss�%?���        )��P	N��J���A�*

	conv_loss
�%?H��f        )��P	���J���A�*

	conv_lossXs%?�.�d        )��P	�J���A�*

	conv_lossbe%?Sut}        )��P	��J���A�*

	conv_lossN%?L�ҽ        )��P	uO�J���A�*

	conv_loss}9%?YMy�        )��P	���J���A�*

	conv_loss=(%?��3S        )��P	���J���A�*

	conv_loss%?>�6�        )��P	��J���A�*

	conv_loss� %?)�ϴ        )��P	�*�J���A�*

	conv_loss��$?��Z        )��P	 i�J���A�*

	conv_loss��$?QІY        )��P	���J���A�*

	conv_loss��$?[vuw        )��P	6оJ���A�*

	conv_lossQ�$?��        )��P	� �J���A�*

	conv_lossG�$?�U,        )��P	9�J���A�*

	conv_lossU�$?V���        )��P	rk�J���A�*

	conv_lossc�$?EON        )��P	i��J���A�*

	conv_loss7p$?�̢%        )��P	�˿J���A�*

	conv_loss�Z$?��U        )��P	���J���A�*

	conv_loss�I$?pюW        )��P	�-�J���A�*

	conv_loss+$?ouL        )��P	�^�J���A�*

	conv_loss�'$?����        )��P	ӎ�J���A�*

	conv_loss�$?���g        )��P	���J���A�*

	conv_loss��#?<�j        )��P	e��J���A�*

	conv_loss��#?˲��        )��P	�,�J���A�*

	conv_loss��#?��?        )��P	�\�J���A�*

	conv_loss��#?����        )��P	&��J���A�*

	conv_lossܨ#?�.        )��P	˽�J���A�*

	conv_lossǜ#?�٢�        )��P	-��J���A�*

	conv_loss��#?�=�I        )��P	a�J���A�*

	conv_loss�~#?�ʉ�        )��P	Q�J���A�*

	conv_loss�q#?=�"w        )��P	O��J���A�*

	conv_loss�T#?\x�#        )��P	K��J���A�*

	conv_loss<>#?�k��        )��P	H��J���A�*

	conv_loss$5#?��/|        )��P	��J���A�*

	conv_loss]*#?U��        )��P	�C�J���A�*

	conv_loss�
#?���        )��P	_r�J���A�*

	conv_loss}#?h��        )��P	��J���A�*

	conv_loss��"?_	Q�        )��P	���J���A�*

	conv_loss��"?+h�        )��P	��J���A�*

	conv_loss6�"?��ś        )��P	�6�J���A�*

	conv_loss��"?_��        )��P	cf�J���A�*

	conv_loss��"?ƝT�        )��P	`��J���A�*

	conv_lossU�"?/���        )��P	��J���A�*

	conv_loss�z"?K�0        )��P	2��J���A�*

	conv_lossbl"?��	        )��P	*,�J���A�*

	conv_loss�U"?        )��P	�\�J���A�*

	conv_loss	I"?��        )��P	}��J���A�*

	conv_loss�6"?T+�V        )��P	q��J���A�*

	conv_loss�"?�=��        )��P	���J���A�*

	conv_loss�"?�=�        )��P	� �J���A�*

	conv_loss|"?3|�        )��P	���J���A�*

	conv_lossx�!?G �9        )��P	v��J���A�*

	conv_lossB�!?.��.        )��P	_�J���A�*

	conv_lossL�!?�w�        )��P	_J�J���A�*

	conv_loss�!?���        )��P	�|�J���A�*

	conv_loss9�!??��        )��P	��J���A�*

	conv_lossҏ!?��!�        )��P	D��J���A�*

	conv_loss�v!?�^�        )��P	W�J���A�*

	conv_loss<h!?k��        )��P	�O�J���A�*

	conv_lossGW!?�;z        )��P	��J���A�*

	conv_loss�H!?���        )��P	���J���A�*

	conv_loss�5!?(��0        )��P	���J���A�*

	conv_loss !?�.        )��P	��J���A�*

	conv_loss�!?���        )��P	�R�J���A�*

	conv_loss�� ?G~�        )��P	���J���A�*

	conv_loss0� ?^�`        )��P	���J���A�*

	conv_lossL� ?�av�        )��P	���J���A�*

	conv_loss�� ?�`�b        )��P	�!�J���A�*

	conv_lossҢ ?d��        )��P	�S�J���A�*

	conv_lossr� ?�&�        )��P	ڄ�J���A�*

	conv_lossҜ ?B��        )��P	��J���A�*

	conv_loss�� ?�$'N        )��P	F��J���A�*

	conv_loss k ?�        )��P	[�J���A�*

	conv_loss�L ?�ZFN        )��P	�F�J���A�*

	conv_loss�D ?4rߡ        )��P	�v�J���A�*

	conv_loss 8 ?6�        )��P	
��J���A�*

	conv_loss$ ?:���        )��P	���J���A�*

	conv_loss ?^�        )��P	��J���A�*

	conv_loss��?����        )��P	�8�J���A�*

	conv_lossB�?�w�5        )��P	@j�J���A�*

	conv_loss��?�.p]        )��P	Л�J���A�*

	conv_loss��?J�k�        )��P	 ��J���A�*

	conv_loss�?���        )��P	'��J���A�*

	conv_loss>�?��<V        )��P	X*�J���A�*

	conv_loss2�?���        )��P	�Z�J���A�*

	conv_loss�w?��yJ        )��P	b��J���A�*

	conv_loss�c?����        )��P	d��J���A�*

	conv_loss�[?Ϻ        )��P	���J���A�*

	conv_lossOJ?yxB�        )��P	��J���A�*

	conv_loss�3?_��[        )��P	�M�J���A�*

	conv_lossU?ͥ�        )��P	X�J���A�*

	conv_loss�?��J�        )��P	f��J���A�*

	conv_loss�?1�U}        )��P	(��J���A�*

	conv_loss@�?qx��        )��P	�J���A�*

	conv_lossx�?�9�A        )��P	+?�J���A�*

	conv_loss��?��        )��P	/n�J���A�*

	conv_lossα?�d�        )��P	���J���A�*

	conv_loss�?��4�        )��P	Y��J���A�*

	conv_loss�?���v        )��P	��J���A�*

	conv_losss�?��n�        )��P	�4�J���A�*

	conv_loss)h?7H�        )��P	>y�J���A�*

	conv_loss�R?'��5        )��P	s��J���A�*

	conv_loss�>?�F*        )��P	x��J���A�*

	conv_loss�;?��	        )��P		�J���A�*

	conv_loss�?.-�        )��P	;@�J���A�*

	conv_loss?�GG        )��P	{�J���A�*

	conv_lossF?m���        )��P	f��J���A�*

	conv_loss��?�O�        )��P	���J���A�*

	conv_lossa�?OS��        )��P	"�J���A�*

	conv_loss2�?�fk�        )��P	=R�J���A�*

	conv_lossǾ?`Tx        )��P	x��J���A�*

	conv_loss��?Kz        )��P	���J���A�*

	conv_loss�|?NR`        )��P	��J���A�*

	conv_loss�u?k�g}        )��P	b�J���A�*

	conv_loss�g?�Ε�        )��P	�I�J���A�*

	conv_loss�P?� ��        )��P	|�J���A�*

	conv_loss�6?����        )��P	���J���A�*

	conv_loss�5?Qg�h        )��P	J��J���A�*

	conv_loss*?���        )��P	��J���A�*

	conv_loss�?�.�2        )��P	zG�J���A�*

	conv_loss��?�1E        )��P	�y�J���A�*

	conv_loss��?�u�        )��P	���J���A�*

	conv_loss��?�8B        )��P	���J���A�*

	conv_loss��?��        )��P	w
�J���A�*

	conv_loss��?���        )��P	3:�J���A�*

	conv_loss;�?��H        )��P	�j�J���A�*

	conv_loss=x?&Y�,        )��P	���J���A�*

	conv_loss|]?3�IW        )��P	t��J���A�*

	conv_loss_Z?�(�        )��P	���J���A�*

	conv_lossbI?L�1t        )��P	b-�J���A�*

	conv_loss�?�R�        )��P	�_�J���A�*

	conv_loss=?��"        )��P	{��J���A�*

	conv_lossE?�FZ:        )��P	ۿ�J���A�*

	conv_lossx�?x��        )��P	_��J���A�*

	conv_loss�?]�>�        )��P	��J���A�*

	conv_lossd�?��]        )��P	�O�J���A�*

	conv_loss#�?R6�        )��P	�~�J���A�*

	conv_loss��?E�T6        )��P	���J���A�*

	conv_loss�?A��        )��P	���J���A�*

	conv_loss/�?�uy�        )��P	d�J���A�*

	conv_lossu?I�b4        )��P	�=�J���A�*

	conv_lossFK?��v�        )��P	$m�J���A�*

	conv_loss"N?il��        )��P	��J���A�*

	conv_loss�0?���        )��P	b��J���A�*

	conv_loss�!?)_@�        )��P	�J���A�*

	conv_loss<�?H���        )��P	=4�J���A�*

	conv_loss��?","        )��P	xd�J���A�*

	conv_loss �?>A��        )��P	x��J���A�*

	conv_loss�?p�P�        )��P	��J���A�*

	conv_loss��?�q|        )��P	���J���A�*

	conv_lossh�?����        )��P	�0�J���A�*

	conv_loss�?�-�         )��P	s`�J���A�*

	conv_lossCp?����        )��P	���J���A�*

	conv_loss;z?��        )��P	���J���A�*

	conv_loss#J?NZ|�        )��P	���J���A�*

	conv_losso??��H        )��P	3)�J���A�*

	conv_loss�?�,y:        )��P	[�J���A�*

	conv_loss??�3        )��P	d��J���A�*

	conv_loss��?��(�        )��P	���J���A�*

	conv_loss��?;��x        )��P	��J���A�*

	conv_loss6�?� �        )��P	G�J���A�*

	conv_lossv�?HE��        )��P	�y�J���A�*

	conv_loss6�?�d7@        )��P		��J���A�*

	conv_loss5}?�/�z        )��P	N��J���A�*

	conv_loss�|?��p�        )��P	^�J���A�*

	conv_loss�H?�-��        )��P	M�J���A�*

	conv_loss!??�2sl        )��P	7}�J���A�*

	conv_loss�4?(�F_        )��P	��J���A�*

	conv_loss�?�o%�        )��P	��J���A�*

	conv_loss:�?��d�        )��P	��J���A�*

	conv_lossZ�?��j        )��P	:N�J���A�*

	conv_loss��?y��4        )��P	���J���A�*

	conv_loss"�?�f�p        )��P	M��J���A�*

	conv_loss�?+AH        )��P	���J���A�*

	conv_lossه?5�)        )��P	~�J���A�*

	conv_loss�r?��	�        )��P	�M�J���A�*

	conv_lossnc?Ȼ        )��P	�}�J���A�*

	conv_loss	(?b+        )��P	P��J���A�*

	conv_loss�?ԁ�s        )��P	D��J���A�*

	conv_loss3?,H        )��P	o�J���A�*

	conv_loss2�?B��!        )��P	�@�J���A�*

	conv_lossQ�?ȑ��        )��P	q�J���A�*

	conv_loss��?���        )��P	���J���A�*

	conv_lossN�?���        )��P	��J���A�*

	conv_loss�?X@
        )��P	��J���A�*

	conv_loss��?��V        )��P	09�J���A�*

	conv_lossDj?���        )��P	em�J���A�*

	conv_loss�>?r�֮        )��P	:��J���A�*

	conv_loss�'?��	        )��P	���J���A�*

	conv_lossd?�p�        )��P	���J���A�*

	conv_loss
�?�~�$        )��P	@0�J���A�*

	conv_loss��?��;        )��P	�a�J���A�*

	conv_lossܔ?�p3        )��P	��J���A�*

	conv_lossj?���        )��P	]��J���A�*

	conv_lossT?���        )��P	~��J���A�*

	conv_loss:?��G        )��P	�+�J���A�*

	conv_loss?b�ӌ        )��P	�\�J���A�*

	conv_loss=?GWU:        )��P	��J���A�*

	conv_loss!�?�df�        )��P	W��J���A�*

	conv_lossW�?w��        )��P	���J���A�*

	conv_loss،?/H9�        )��P	%2�J���A�*

	conv_lossYu?����        )��P	�e�J���A�*

	conv_lossyF?�Ɉ7        )��P	��J���A�*

	conv_loss�?���L        )��P	9��J���A�*

	conv_loss�?m�J~        )��P	��J���A�*

	conv_loss/�?-��        )��P	�3�J���A�*

	conv_lossN�?*X&        )��P	n�J���A�*

	conv_lossD?Z�        )��P	���J���A�*

	conv_loss]h?�(Q�        )��P	z��J���A�*

	conv_loss�.?��E�        )��P	��J���A�*

	conv_loss"�?<	�        )��P	�D�J���A�*

	conv_loss�}?�5�        )��P	�|�J���A�*

	conv_loss9�?4B	        )��P	���J���A�*

	conv_loss�)?��T�        )��P	x��J���A�*

	conv_loss��?V��        )��P	M�J���A�*

	conv_loss�?��        )��P	�=�J���A�*

	conv_loss��?��        )��P	�n�J���A�*

	conv_loss(�?<=8�        )��P	���J���A�*

	conv_loss`B?o�-"        )��P	*��J���A�*

	conv_loss#�?�        )��P	��J���A�*

	conv_loss��?����        )��P	:�J���A�*

	conv_loss�g?:s�        )��P	Go�J���A�*

	conv_loss2�?y��J        )��P	y��J���A�*

	conv_loss��?^��(        )��P	��J���A�*

	conv_loss�q?f��        )��P	A�J���A�*

	conv_losss�?�_�S        )��P	�5�J���A�*

	conv_loss� ?�{�        )��P	bf�J���A�*

	conv_loss�D?r�I�        )��P	��J���A�*

	conv_lossg�?1�2�        )��P	2��J���A�*

	conv_loss�?�9t�        )��P	��J���A�*

	conv_lossV?-�:        )��P	�.�J���A�*

	conv_loss�`?�>S�        )��P	]�J���A�*

	conv_loss�?S�h        )��P	���J���A�*

	conv_loss�?�3�        )��P	:��J���A�*

	conv_loss.�?V��         )��P	���J���A�*

	conv_lossex
?���        )��P	� �J���A�*

	conv_loss|L
?����        )��P	�P�J���A�*

	conv_loss؆	?��K        )��P	x��J���A�*

	conv_loss�j?[���        )��P	���J���A�*

	conv_loss6�?=}Sd        )��P	I��J���A�*

	conv_lossP1?;��I        )��P	�J���A�*

	conv_lossu�?K6��        )��P	YJ�J���A�*

	conv_loss��?�w'_        )��P	�{�J���A�*

	conv_loss��?��s^        )��P	���J���A�*

	conv_loss^�?Xa�        )��P	b��J���A�*

	conv_lossX?}�o�        )��P	#�J���A�*

	conv_lossb��>z���        )��P	@�J���A�*

	conv_loss�u�>��        )��P	Hp�J���A�*

	conv_loss���>T         )��P	��J���A�*

	conv_lossqC�>��r        )��P	���J���A�*

	conv_loss���>F�K        )��P	�g�J���A�*

	conv_loss/�>�1s        )��P	���J���A�*

	conv_loss_i�>���@        )��P	���J���A�*

	conv_loss7n�>���        )��P	V �J���A�*

	conv_loss�0�>�&�        )��P	�1�J���A�*

	conv_loss��>���p        )��P	ib�J���A�*

	conv_loss"�>�,�N        )��P	���J���A�*

	conv_loss�K�>Q�\        )��P	2��J���A�*

	conv_loss_�>#q�        )��P	%�J���A�*

	conv_lossղ�>`Ye        )��P	�8�J���A�*

	conv_lossɝ�>���        )��P	hk�J���A�*

	conv_loss��>�/^        )��P	ߛ�J���A�*

	conv_loss���>&1�        )��P	D��J���A�*

	conv_loss'm�>���        )��P	-�J���A�*

	conv_loss�G�>�x�t        )��P	�:�J���A�*

	conv_loss,��>Vծn        )��P	�m�J���A�*

	conv_lossW�>J���        )��P	���J���A�*

	conv_loss���>C�Kt        )��P	���J���A�*

	conv_loss�Ǭ>���        )��P	��J���A�*

	conv_loss�>�>H~e        )��P	-�J���A�*

	conv_loss�X�>߰&#        )��P	g\�J���A�*

	conv_loss=�>�#r        )��P	Ќ�J���A�*

	conv_loss"�>�.��        )��P	k��J���A�*

	conv_loss��>�Be�        )��P	���J���A�*

	conv_lossX��>�9$        )��P	9�J���A�*

	conv_loss���>�`c        )��P	�O�J���A�*

	conv_loss���>��O        )��P	5�J���A�*

	conv_loss�"�>V�/#        )��P	��J���A�*

	conv_loss�d�>��'        )��P	���J���A�*

	conv_lossx��>Xbߢ        )��P	�J���A�*

	conv_lossѯ>��Y        )��P	�A�J���A�*

	conv_losso�>����        )��P	\t�J���A�*

	conv_lossq�>2�U�        )��P	æ�J���A�*

	conv_losseR�>��        )��P	g��J���A�*

	conv_lossI��>��-;        )��P	
�J���A�*

	conv_loss�k�>��"        )��P	;�J���A�*

	conv_loss:Q�>L��        )��P	m�J���A�*

	conv_loss���>0��Y        )��P	���J���A�*

	conv_loss�q�>�^�        )��P	��J���A�*

	conv_loss��>2z��        )��P	7�J���A�*

	conv_losszV�>�	t�        )��P	�4�J���A�*

	conv_lossը>���        )��P	Kh�J���A�*

	conv_loss�U�>n�L�        )��P	Z��J���A�*

	conv_loss�Ш>��S�        )��P	���J���A�*

	conv_loss^��>�S�>        )��P	@�J���A�*

	conv_loss�ŧ>�7"^        )��P	}:�J���A�*

	conv_loss�g�> �7V        )��P	�l�J���A�*

	conv_lossy~�>�Up�        )��P	��J���A�*

	conv_loss�ۨ>���Z        )��P	���J���A�*

	conv_loss��>�˃�        )��P	��J���A�*

	conv_loss���>;��@        )��P	R�J���A�*

	conv_loss��>x�        )��P	���J���A�*

	conv_loss+�>�W�Z        )��P	ٸ�J���A�*

	conv_lossO9�>�]�        )��P	7��J���A�*

	conv_lossl��>���        )��P	2$�J���A�*

	conv_loss�C�>k�Ա        )��P	�W�J���A�*

	conv_loss�5�>·�        )��P	���J���A�*

	conv_loss��>���J        )��P	��J���A�*

	conv_loss��>� �        )��P	{�J���A�*

	conv_lossuҦ>���        )��P	�8�J���A�*

	conv_loss�z�>`�O        )��P	.l�J���A�*

	conv_loss��>03�)        )��P	'��J���A�*

	conv_loss��>�1�q        )��P	p��J���A�*

	conv_loss�¦>�J�        )��P	��J���A�*

	conv_lossJ��>�h�        )��P	�K�J���A�*

	conv_lossԦ�>85��        )��P	!��J���A�*

	conv_lossME�>i�        )��P	o��J���A�*

	conv_lossͤ>�۱�        )��P		��J���A�*

	conv_loss9��>$P�Z        )��P	<�J���A�*

	conv_loss(��>�xw�        )��P	�S�J���A�*

	conv_lossx�>�7��        )��P	��J���A�*

	conv_loss���>|�W�        )��P	���J���A�*

	conv_loss�H�>��F�        )��P	P��J���A�*

	conv_loss�f�>ϼ�        )��P	z"�J���A�*

	conv_lossCϤ>��I�        )��P	cV�J���A�*

	conv_loss�>�>��Q        )��P	��J���A�*

	conv_loss�M�>\f¬        )��P	��J���A�*

	conv_loss�Z�>���        )��P	���J���A�*

	conv_lossn�>�7n        )��P	d, K���A�*

	conv_loss�>�e�
        )��P	/a K���A�*

	conv_loss�>���        )��P	M� K���A�*

	conv_loss]�>�'S        )��P	=� K���A�*

	conv_loss�W�>C�(        )��P	�� K���A�*

	conv_loss��>���K        )��P	�3K���A�*

	conv_loss�;�>����        )��P	�gK���A�*

	conv_loss�V�>�W�*        )��P	��K���A�*

	conv_lossmɦ>�2l        )��P	`�K���A�*

	conv_loss�
�>���        )��P	�K���A�*

	conv_loss���>J�        )��P	x5K���A�*

	conv_loss!Q�>=���        )��P	
kK���A�*

	conv_loss,N�>H%]�        )��P	ٝK���A�*

	conv_lossѣ>��Q        )��P	�K���A�*

	conv_loss�¥>Rr�        )��P	�K���A�*

	conv_losst3�>n��        )��P	\<K���A�*

	conv_lossj�>���        )��P	�oK���A�*

	conv_loss'�>�|Ĳ        )��P	1�K���A�*

	conv_loss��>m��3        )��P	��K���A�*

	conv_loss�l�>4>�        )��P	#K���A�*

	conv_loss���>���        )��P	�CK���A�*

	conv_loss��>�vW�        )��P	F?	K���A�*

	conv_losse/�>:��,        )��P	Y�	K���A�*

	conv_lossm��>6B        )��P	�	K���A�*

	conv_lossU��>?��        )��P	\�	K���A�*

	conv_loss�Ѣ>U��0        )��P	�
K���A�*

	conv_loss啟>M��        )��P	�E
K���A�*

	conv_loss9�>�칍        )��P	�{
K���A�*

	conv_loss�L�>q�5        )��P	'�
K���A�*

	conv_lossՁ�>���a        )��P	�
K���A�*

	conv_loss�\�>����        )��P	�K���A�*

	conv_loss|�>N(��        )��P	\9K���A�*

	conv_loss���>o��        )��P	wK���A�*

	conv_loss��>� r"        )��P	=�K���A�*

	conv_loss�֠>��r�        )��P	K�K���A�*

	conv_lossjw�>�R��        )��P	�K���A�*

	conv_loss�a�>nd��        )��P	�9K���A�*

	conv_loss�
�>GƑ        )��P	�nK���A�*

	conv_loss���>i(p�        )��P	c�K���A�*

	conv_loss��>���        )��P	��K���A�*

	conv_loss�I�>��0        )��P	�K���A�*

	conv_loss�f�>.*+�        )��P	�5K���A�*

	conv_loss;x�>�m,        )��P	fK���A�*

	conv_loss�F�>�F�:        )��P	��K���A�*

	conv_loss�ݞ>�6	        )��P	��K���A�*

	conv_loss��>j1�4        )��P	E�K���A�*

	conv_lossݛ�>���        )��P	'#K���A�*

	conv_loss��>kMu]        )��P	RK���A�*

	conv_loss�՞>�|Z�        )��P	}�K���A�*

	conv_loss�f�>����        )��P	��K���A�*

	conv_loss��>�5��        )��P	��K���A�*

	conv_loss��>�Q        )��P	�K���A�*

	conv_lossॠ>��_        )��P	)GK���A�*

	conv_loss���>�        )��P	�uK���A�*

	conv_loss��>�(��        )��P	U�K���A�*

	conv_loss���>��#        )��P	\�K���A�*

	conv_loss�5�>�X�        )��P	�K���A�*

	conv_losssH�>�B��        )��P	�5K���A�*

	conv_loss���>�b��        )��P	 eK���A�*

	conv_loss���>̨4        )��P	��K���A�*

	conv_loss��>	��O        )��P	��K���A�*

	conv_loss$+�>�E�        )��P	��K���A�*

	conv_losss�>�J�7        )��P	�&K���A�*

	conv_lossN��>[���        )��P	qVK���A�*

	conv_loss��>Mݖp        )��P	ÄK���A�*

	conv_loss?$�>&��        )��P	ɶK���A�*

	conv_loss�b�>|�d        )��P	��K���A�*

	conv_lossK��>j�N�        )��P	�K���A�*

	conv_loss3�>�:�        )��P	4HK���A�*

	conv_loss=2�>����        )��P	rzK���A�*

	conv_loss�L�>Ҧ�        )��P	Q�K���A�*

	conv_loss���>�bv�        )��P	 �K���A�*

	conv_lossǉ�>�]T�        )��P	' K���A�*

	conv_loss���>{��`        )��P	 RK���A�*

	conv_loss��>���F        )��P	)�K���A�*

	conv_loss��>��&�        )��P	��K���A�*

	conv_losssp�>	Au        )��P	��K���A�*

	conv_losshÚ>�`Q5        )��P	$K���A�*

	conv_loss$g�>x^�        )��P	,QK���A�*

	conv_lossE4�>�ŵ        )��P	�K���A�*

	conv_loss]�>��,/        )��P	��K���A�*

	conv_lossw��>,׺�        )��P	~�K���A�*

	conv_loss�p�>V�        )��P	=&K���A�*

	conv_loss�%�>9�͍        )��P	�VK���A�*

	conv_lossj�>cV�        )��P	�K���A�*

	conv_loss��>q��B        )��P	:�K���A�*

	conv_loss�f�>�宀        )��P	��K���A�*

	conv_loss�>�j�~        )��P	�#K���A�*

	conv_lossU��>�',        )��P	�WK���A�*

	conv_loss�ə>��j(        )��P	�K���A�*

	conv_loss]T�>�;Ч        )��P	��K���A�*

	conv_loss���>��x        )��P	��K���A�*

	conv_loss�~�>��-        )��P	u(K���A�*

	conv_loss��>=fl�        )��P	P[K���A�*

	conv_losspW�>J���        )��P	�K���A�*

	conv_loss���>q4�f        )��P	F�K���A�*

	conv_lossǙ>�g�        )��P	d�K���A�*

	conv_loss�˕>��ү        )��P	�"K���A�*

	conv_loss���>���        )��P	XTK���A�*

	conv_loss�j�>�O�        )��P	ԇK���A�*

	conv_loss�N�>Gi�        )��P	v�K���A�*

	conv_lossS�>5Q�m        )��P	R�K���A�*

	conv_loss4~�>l�]v        )��P	�K���A�*

	conv_losssi�>��        )��P	�NK���A�*

	conv_loss'��>�5t�        )��P	�K���A�*

	conv_lossE�>� �        )��P	�K���A�*

	conv_loss)՘>���        )��P	��K���A�*

	conv_loss{�>�_�        )��P	�K���A�*

	conv_loss�ݖ>4Zp'        )��P	&KK���A�*

	conv_loss�>>�ka        )��P	ZzK���A�*

	conv_lossd�>���U        )��P	ȭK���A�*

	conv_lossbh�>*?        )��P	�K���A�*

	conv_loss�9�>3 ��        )��P	�K���A�*

	conv_loss�͘>;xG�        )��P	�@K���A�*

	conv_lossW�>��D        )��P	>rK���A�*

	conv_loss���>".8�        )��P	��K���A�*

	conv_loss�f�>;���        )��P	��K���A�*

	conv_loss�>Cx�        )��P	�K���A�*

	conv_loss�J�>Lʵ        )��P	�;K���A�*

	conv_loss�z�>>�0�        )��P	yoK���A�*

	conv_loss��>ܱ|A        )��P	��K���A�*

	conv_loss`I�>��W        )��P	��K���A�*

	conv_lossuL�>N"ܕ        )��P	uoK���A�*

	conv_loss˼�>�%��        )��P	|�K���A�*

	conv_loss�/�>��uj        )��P	&�K���A�*

	conv_loss�>}!y        )��P	K���A�*

	conv_loss��>}g�        )��P	@K���A�*

	conv_loss���>6���        )��P		vK���A�*

	conv_loss��>��ww        )��P	�K���A�*

	conv_loss���>���        )��P	f�K���A�*

	conv_loss���>�5T        )��P	F K���A�*

	conv_loss4ؓ>����        )��P	�c K���A�*

	conv_loss�>�b�        )��P	۞ K���A�*

	conv_loss��>�[��        )��P	� K���A�*

	conv_loss{��>ji#y        )��P	8!K���A�*

	conv_loss��>ϑ�        )��P	�C!K���A�*

	conv_lossڒ>M�;j        )��P	�u!K���A�*

	conv_loss�t�>]=.�        )��P	g�!K���A�*

	conv_loss�d�>r�<        )��P	R�!K���A�*

	conv_loss���>�l�        )��P	g"K���A�*

	conv_loss���>�,̠        )��P	�G"K���A�*

	conv_loss���>#��M        )��P	 {"K���A�*

	conv_loss9�>����        )��P	d�"K���A�*

	conv_loss��>��95        )��P	L�"K���A�*

	conv_loss��>ʰ��        )��P	#K���A�*

	conv_lossL��>|�@O        )��P	�J#K���A�*

	conv_loss@?�>ú        )��P	�}#K���A�*

	conv_loss/[�>`*�R        )��P	G�#K���A�*

	conv_loss��>Թi[        )��P	��#K���A�*

	conv_loss�0�>���        )��P	�$K���A�*

	conv_loss/ԋ>ה-L        )��P	�K$K���A�*

	conv_loss$��>�Cs�        )��P	0~$K���A�*

	conv_lossDB�>F��        )��P	�$K���A�*

	conv_lossF/�>cK�Q        )��P	C�$K���A�*

	conv_loss�g�>���W        )��P	0%K���A�*

	conv_loss�ŋ>��xw        )��P	L%K���A�*

	conv_loss#�>�L        )��P	��%K���A�*

	conv_loss/�>p�        )��P	��%K���A�*

	conv_loss�߇>���        )��P	��%K���A�*

	conv_loss ��>qQq        )��P	�&K���A�*

	conv_loss�9�>r�        )��P	UR&K���A�*

	conv_loss�i�>_|1        )��P	��&K���A�*

	conv_loss���>IV4�        )��P	��&K���A�*

	conv_loss�e�>��x^        )��P	��&K���A�*

	conv_loss�F�>x�        )��P	�'K���A�*

	conv_loss���>m"c        )��P	�Q'K���A�*

	conv_lossy��>zF�9        )��P	i�'K���A�*

	conv_loss�̅>t	        )��P	i�'K���A�*

	conv_loss>or2        )��P	��'K���A�*

	conv_loss�c�>q���        )��P	�!(K���A�*

	conv_lossȄ>��{v        )��P		V(K���A�*

	conv_loss�s�>8|X        )��P	(K���A�*

	conv_loss���>�`��        )��P	��(K���A�*

	conv_lossD��>դ��        )��P	�)K���A�*

	conv_loss���>�k3        )��P	�8)K���A�*

	conv_loss�}y>�Un        )��P	vm)K���A�*

	conv_loss���>�Q        )��P	��)K���A�*

	conv_loss^~>E��        )��P	$�)K���A�*

	conv_lossN�>����        )��P	�	*K���A�*

	conv_lossi'�>�h�        )��P	�K*K���A�*

	conv_loss���>&v�t        )��P	ǋ*K���A�*

	conv_loss�>���        )��P	��*K���A�*

	conv_loss�|>�|�        )��P	s�*K���A�*

	conv_lossOY{>��=/        )��P	�*+K���A�*

	conv_loss�zy>u���        )��P	R_+K���A�*

	conv_loss���>����        )��P	�+K���A�*

	conv_losssC�>�B,        )��P	�+K���A�*

	conv_loss��>&��)        )��P	��+K���A�*

	conv_loss�wu>l��        )��P	|1,K���A�*

	conv_lossu�>O�|        )��P	;f,K���A�*

	conv_loss2dw>�1 �        )��P	��,K���A�*

	conv_loss��p>��J        )��P	��,K���A�*

	conv_loss�Aq>�x��        )��P	v-K���A�*

	conv_loss�s>e۱q        )��P	�:-K���A�*

	conv_loss8$x>�k�v        )��P	oq-K���A�*

	conv_loss��p>sA��        )��P	�-K���A�*

	conv_loss��v>o�A        )��P	��-K���A�*

	conv_loss�o>�谳        )��P	�	.K���A�*

	conv_loss�Ac>���2        )��P	�<.K���A�*

	conv_loss�e>M!ݨ        )��P	�o.K���A�*

	conv_loss��[>7Q�s        )��P	T�.K���A�*

	conv_loss��t>=lv�        )��P	r�.K���A�*

	conv_loss��\>�3�        )��P	E	/K���A�*

	conv_lossV�f>�
�        )��P	�:/K���A�*

	conv_loss5�Z>Ӭ՛        )��P	�n/K���A�*

	conv_lossG�e>�u��        )��P	��/K���A�*

	conv_loss\�g>+��        )��P	3�/K���A�*

	conv_loss�3\>���        )��P	�	0K���A�*

	conv_loss�\>7_�        )��P	*@0K���A�*

	conv_loss��e>,56        )��P	�s0K���A�*

	conv_loss�z\>�'ݕ        )��P	y�0K���A�*

	conv_loss�Af>�1߳        )��P	j�0K���A�*

	conv_loss~�X>$�        )��P	n1K���A�*

	conv_loss�,b>|��        )��P	M=1K���A�*

	conv_loss�]>ҩ�        )��P	�p1K���A�*

	conv_loss\>����        )��P	��1K���A�*

	conv_loss��Y>$r�        )��P	��1K���A�*

	conv_loss�[R>����        )��P	�2K���A�*

	conv_loss��O>'�ê        )��P	7@2K���A�*

	conv_loss��W>�>�        )��P	�r2K���A�*

	conv_loss�+Q>�Ԃ        )��P	ϧ2K���A�*

	conv_loss�[>�Z �        )��P	N�2K���A�*

	conv_loss��I>#u�        )��P	)#3K���A�*

	conv_loss�V>G�        )��P	�W3K���A�*

	conv_loss�CN>_�6        )��P	&�3K���A�*

	conv_losszfV>p8�        )��P	|�3K���A�*

	conv_lossN�K>���        )��P	��3K���A�*

	conv_loss��]>�q�A        )��P	�)4K���A�*

	conv_lossXDM>ͪ��        )��P	�^4K���A�*

	conv_loss+�p>1��<        )��P	6�4K���A�*

	conv_loss��c>O�IK        )��P	�4K���A�*

	conv_loss�yD>��/'        )��P	_5K���A�*

	conv_lossf�Y>L�b�        )��P	�A5K���A�*

	conv_loss��N>���O        )��P	�u5K���A�*

	conv_loss��8>\�Q        )��P	��5K���A�*

	conv_loss�FM> Ft�        )��P	��5K���A�*

	conv_loss��F>�*�        )��P	�6K���A�*

	conv_loss^�@>%�g        )��P	G6K���A�*

	conv_lossґT>l��        )��P	.y6K���A�*

	conv_lossMX?>-�ؓ        )��P	6�6K���A�*

	conv_lossJ�O>K ]�        )��P	��6K���A�*

	conv_loss�N?>fk        )��P	f7K���A�*

	conv_loss�ZB>6��        )��P	N7K���A�*

	conv_loss��8>K�        )��P	�7K���A�*

	conv_loss�}>>�N�        )��P	��7K���A�*

	conv_loss,3>7��        )��P	��7K���A�*

	conv_loss5/>G�"�        )��P	/8K���A�*

	conv_loss�BB>�=��        )��P	�T8K���A�*

	conv_loss��8>j�)=        )��P	�8K���A�*

	conv_lossHG0>OC��        )��P	��8K���A�*

	conv_loss��8>C��        )��P	e�8K���A�*

	conv_loss��F>�V�d        )��P	�%9K���A�*

	conv_lossu<>i{kI        )��P	�Y9K���A�*

	conv_loss��K>�n-�        )��P	�9K���A�*

	conv_loss�@>�0�        )��P	��9K���A�*

	conv_loss�<>�V��        )��P	�9K���A�*

	conv_loss7�;>�6�        )��P	�$:K���A�*

	conv_loss��5>S*�L        )��P	X:K���A�*

	conv_loss}�/>)U        )��P	��:K���A�*

	conv_loss�&g>��/        )��P	
�:K���A�*

	conv_loss �W>I`��        )��P	��:K���A�*

	conv_losslI>W�W        )��P	/&;K���A�*

	conv_lossh�=>�8��        )��P	�Z;K���A�*

	conv_loss�[6>�Pټ        )��P	x�;K���A�*

	conv_loss��4>���        )��P	{�;K���A�*

	conv_loss'.X>R_S        )��P	X�;K���A�*

	conv_loss�n7>,���        )��P	�)<K���A�*

	conv_loss*�$>;Rw|        )��P	�]<K���A�*

	conv_loss�T,>q���        )��P	V�<K���A�*

	conv_lossؙ6>�2�        )��P	��<K���A�*

	conv_loss�q!>u]�C        )��P	��<K���A�*

	conv_loss�|/>�J�        )��P	a-=K���A�*

	conv_lossD�#>��_�        )��P	Ss=K���A�*

	conv_loss�)&>�YA�        )��P	Ԧ=K���A�*

	conv_loss�O*>i~	�        )��P	h�=K���A�*

	conv_loss�8.>���I        )��P	>K���A�*

	conv_lossE�.>
�E�        )��P	<D>K���A�*

	conv_loss3@>H�
         )��P	Ww>K���A�*

	conv_lossS":>��x        )��P	_�>K���A�*

	conv_loss8�*>��P        )��P	X�>K���A�*

	conv_lossϓ,>w�#q        )��P	�%?K���A�*

	conv_loss~�'>��]        )��P	o?K���A�*

	conv_loss�I>�(0        )��P	��?K���A�*

	conv_loss�Wm>�s��        )��P	)�?K���A�*

	conv_loss �D>�[j�        )��P	@K���A�*

	conv_loss�V$>���2        )��P	A@K���A�*

	conv_loss�H$>�_��        )��P	st@K���A�*

	conv_loss��<>>�7�        )��P	�@K���A�*

	conv_loss��P>�2�        )��P	 �@K���A�*

	conv_loss/I>9���        )��P	�AK���A�*

	conv_loss@4>\ʏ�        )��P	GAK���A�*

	conv_loss7=>�j�        )��P	TAK���A�*

	conv_lossC~N>Ġ��        )��P	ԲAK���A�*

	conv_loss�/$>���        )��P	/�AK���A�*

	conv_loss��D>j���        )��P	�BK���A�*

	conv_loss��>;��U        )��P	&LBK���A�*

	conv_loss�}4>q��        )��P	�}BK���A�*

	conv_loss�>���        )��P	9�BK���A�*

	conv_loss��#>��(I        )��P	��BK���A�*

	conv_loss/&'>��>N        )��P	CK���A�*

	conv_loss�#>�E,        )��P	�MCK���A�*

	conv_loss�*>��'        )��P	�CK���A�*

	conv_loss�>����        )��P	`�CK���A�*

	conv_lossUK)><}�        )��P	��CK���A�*

	conv_lossS\>�u�R        )��P	QDK���A�*

	conv_loss��>����        )��P	�PDK���A�*

	conv_loss�2>h['�        )��P	��DK���A�*

	conv_loss��2>G�        )��P	a�DK���A�*

	conv_lossƋ3>t4}        )��P	W�DK���A�*

	conv_loss��`>���#        )��P	� EK���A�*

	conv_loss�4f>�w�        )��P	VEK���A�*

	conv_loss6�(>V͏0        )��P	�EK���A�*

	conv_loss@T>Q� �        )��P	��EK���A�*

	conv_lossi�1>i�3        )��P	S�EK���A�*

	conv_loss�c>L�S�        )��P	�!FK���A�*

	conv_loss!�>8f2�        )��P	�SFK���A�*

	conv_lossJ{)>��	l        )��P	�FK���A�*

	conv_lossCV>V#`M        )��P	{�FK���A�*

	conv_loss�8>�H?{        )��P	J�FK���A�*

	conv_loss]>e4M�        )��P	�$GK���A�*

	conv_loss��>��3�        )��P	�ZGK���A�*

	conv_loss�E>��x        )��P	�GK���A�*

	conv_loss��>��.�        )��P	&IK���A�*

	conv_loss��>���S        )��P	6XIK���A�*

	conv_lossu)>r��        )��P	ۋIK���A�*

	conv_loss%p>�ݭ        )��P	��IK���A�*

	conv_lossE>R        )��P	��IK���A�*

	conv_loss&8>���        )��P	+JK���A�*

	conv_loss��i>�P2�        )��P	�dJK���A�*

	conv_loss��5>�}�        )��P	ܣJK���A�*

	conv_lossQU">�u�h        )��P	�JK���A�*

	conv_loss��+>��         )��P	�KK���A�*

	conv_loss��>Ų*�        )��P	AKK���A�*

	conv_loss`�#>�)�        )��P	>�KK���A�*

	conv_loss=S>�5j=        )��P	�KK���A�*

	conv_loss���=ըt�        )��P	u�KK���A�*

	conv_loss'4> q��        )��P	,LK���A�*

	conv_loss��>�        )��P	�PLK���A�*

	conv_loss�!>Z�        )��P	ÃLK���A�*

	conv_loss��$>A��        )��P	׷LK���A�*

	conv_loss�Q>���        )��P	��LK���A�*

	conv_loss7>��ŭ        )��P	#MK���A�*

	conv_loss��>^�ci        )��P	[ZMK���A�*

	conv_lossX�>�&        )��P	$�MK���A�*

	conv_lossۓ�=���        )��P	N�MK���A�*

	conv_lossf�>���m        )��P	��MK���A�*

	conv_lossw�:>*�t\        )��P	�(NK���A�*

	conv_loss��K>�/�        )��P	T\NK���A�*

	conv_loss��>�6�6        )��P	̍NK���A�*

	conv_losss7	>�)        )��P	[�NK���A�*

	conv_lossN3>�2)�        )��P	��NK���A�*

	conv_loss��>߰�t        )��P	!)OK���A�*

	conv_loss�>'G��        )��P	o]OK���A�*

	conv_lossXh>���G        )��P	d�OK���A�*

	conv_loss[�>r �        )��P	%�OK���A�*

	conv_loss*�>2�h/        )��P	t�OK���A�*

	conv_loss`�%>VS��        )��P	4PK���A�*

	conv_loss��>��w        )��P	$oPK���A�*

	conv_loss]��=�#�        )��P	�PK���A�*

	conv_loss�c>Ȱ�4        )��P	��PK���A�*

	conv_loss���=�j        )��P	�QK���A�*

	conv_loss�m>� W�        )��P	�JQK���A�*

	conv_lossÁ>kŴ�        )��P	�QK���A�*

	conv_loss|�>'�        )��P	ɵQK���A�*

	conv_loss&�	>��}|        )��P	W�QK���A�*

	conv_loss!�>��        )��P	�RK���A�*

	conv_loss��>G��        )��P	QRK���A�*

	conv_loss�G0>q\L�        )��P	c�RK���A�*

	conv_loss*�7>1i��        )��P	��RK���A�*

	conv_loss�3>����        )��P	�RK���A�*

	conv_loss>�>�SJ�        )��P	� SK���A�*

	conv_lossf>q���        )��P	eSSK���A�*

	conv_loss}<a>1��        )��P	
�SK���A�*

	conv_lossv�;>����        )��P	��SK���A�*

	conv_loss��O>w�0~        )��P	(TK���A�*

	conv_lossG�>��]�        )��P	[7TK���A�*

	conv_loss�u>��*Z        )��P	{jTK���A�*

	conv_loss�=��ۀ        )��P	��TK���A�*

	conv_loss$>g���        )��P	��TK���A�*

	conv_loss�{>(Ѣ�        )��P	�UK���A�*

	conv_lossW>6��        )��P	2@UK���A�*

	conv_loss�)%>�� �        )��P	�zUK���A�*

	conv_lossBE!>���        )��P	�UK���A�*

	conv_loss�kC>���        )��P	��UK���A�*

	conv_loss�2>��,        )��P	[,VK���A�*

	conv_loss��7>��g�        )��P	�_VK���A�*

	conv_loss��=�el(        )��P	�VK���A�*

	conv_loss�#>���        )��P	}�VK���A�*

	conv_lossCp>١�        )��P	��VK���A�*

	conv_losss��=2�g        )��P	Q.WK���A�*

	conv_loss��>���h        )��P	�`WK���A�*

	conv_loss>,�ԛ        )��P	��WK���A�*

	conv_loss���=�/�t        )��P	k�WK���A�*

	conv_losszy�=�aRl        )��P	��WK���A�*

	conv_loss���=C�g        )��P	�:XK���A�*

	conv_loss�r�=����        )��P	pXK���A�*

	conv_loss<��=��        )��P	&�XK���A�*

	conv_loss�>�<�e        )��P	��XK���A�*

	conv_loss:>{Z{�        )��P	�YK���A�*

	conv_loss�=[<�9        )��P	�:YK���A�*

	conv_loss�|>��KM        )��P	WoYK���A�*

	conv_loss�K>�n�        )��P	��YK���A�*

	conv_loss<�=9��        )��P	��YK���A�*

	conv_loss5O�=��        )��P	ZK���A�*

	conv_loss>��=�ˀ�        )��P	�?ZK���A�*

	conv_lossw<>�¥�        )��P	�rZK���A�*

	conv_loss>��ͭ        )��P	;�ZK���A�*

	conv_loss�9>+��{        )��P	X�ZK���A�*

	conv_lossX��=���        )��P	�[K���A�*

	conv_lossi�=hd�d        )��P	
?[K���A�*

	conv_loss���=�&\p        )��P	3r[K���A�*

	conv_loss�E >��۾        )��P	g�[K���A�*

	conv_loss�i">q.��        )��P	�[K���A�*

	conv_loss�N�=��S        )��P	�\K���A�*

	conv_loss��=�Z�        )��P	^B\K���A�*

	conv_lossk��=�J+        )��P	Jv\K���A�*

	conv_lossl�=^��        )��P	�\K���A�*

	conv_loss;��=�Q��        )��P	/�\K���A�*

	conv_lossd�>Rj�6        )��P	!]K���A�*

	conv_loss��>�ɭ�        )��P	DF]K���A�*

	conv_loss��=��U        )��P	�y]K���A�*

	conv_loss�@�=6,^,        )��P	��]K���A�*

	conv_lossT*�=g֫�        )��P	X�]K���A�*

	conv_loss�A>��4?        )��P	�%^K���A�*

	conv_lossz��=u^�~        )��P	�Z^K���A�*

	conv_loss���=�ӑ        )��P	8�^K���A�*

	conv_loss��=���V        )��P	�^K���A�*

	conv_loss/��=3}��        )��P	�_K���A�*

	conv_loss�m�=���X        )��P	�<_K���A�*

	conv_loss]	>f�6�        )��P	Ko_K���A�*

	conv_loss#&�=���        )��P	�_K���A�*

	conv_losseD>� d        )��P	x�_K���A�*

	conv_loss"
>��l�        )��P	�%`K���A�*

	conv_lossn�=��[�        )��P	�Z`K���A�*

	conv_loss��=@6�k        )��P	��`K���A�*

	conv_loss�-#>��?B        )��P	q�`K���A�*

	conv_loss�b>�#B        )��P	O�`K���A�*

	conv_loss�k�=˷��        )��P	�+aK���A�*

	conv_loss��=�$y�        )��P	�^aK���A�*

	conv_loss���=���j        )��P	�aK���A�*

	conv_losso�'>�k�6        )��P	�aK���A�*

	conv_loss�6�=��m-        )��P	c�aK���A�*

	conv_loss�>��q        )��P	�2bK���A�*

	conv_lossD.�=4P��        )��P	�tbK���A�*

	conv_loss���=��]        )��P	?�bK���A�*

	conv_loss���=���g        )��P	�bK���A�*

	conv_loss� >����        )��P	2cK���A�*

	conv_lossJ��=<��        )��P	�CcK���A�*

	conv_lossW��=����        )��P	ucK���A�*

	conv_loss �=M�c~        )��P	�cK���A�*

	conv_loss�ݽ=�T'        )��P	��cK���A�*

	conv_loss�>��W        )��P	�dK���A�*

	conv_loss��=��?T        )��P	�BdK���A�*

	conv_loss�>�xl!        )��P	�udK���A�*

	conv_loss�>V�ą        )��P	ͨdK���A�*

	conv_lossJZ�=�]�1        )��P	U�dK���A�*

	conv_loss4f>��        )��P	�eK���A�*

	conv_loss���==RPi        )��P	xEeK���A�*

	conv_loss���=�\Z�        )��P	wweK���A�*

	conv_lossVe�=��L�        )��P	%�eK���A�*

	conv_loss�T�=���        )��P	#�eK���A�*

	conv_loss��=��J�        )��P	�fK���A�*

	conv_lossw>�(qM        )��P	�FfK���A�*

	conv_loss5��=�F��        )��P	�zfK���A�*

	conv_loss��=�
�        )��P	��fK���A�*

	conv_loss��=Yɚj        )��P	%�fK���A�*

	conv_loss5��==��G        )��P	�gK���A�*

	conv_loss��=F�|H        )��P	�GgK���A�*

	conv_loss̈�=��	{        )��P	f}gK���A�*

	conv_loss�3�=3��X        )��P	��gK���A�*

	conv_loss7��=����        )��P		�gK���A�*

	conv_loss�إ=���        )��P	phK���A�*

	conv_loss��=n���        )��P	�]hK���A�*

	conv_loss�;�=����        )��P	�hK���A�*

	conv_loss(�
>�]%�        )��P	��hK���A�*

	conv_loss"�=�;�v        )��P	�hK���A�*

	conv_loss`��=N�Q        )��P	�+iK���A�*

	conv_loss)?�=n��         )��P	�giK���A�*

	conv_loss�=�=��8j        )��P	=�iK���A�*

	conv_loss>7�=a�v        )��P	?�iK���A�*

	conv_loss���=��n�        )��P	�jK���A�*

	conv_loss���=
Rm�        )��P	�CjK���A�*

	conv_loss�z�=��        )��P	�{jK���A�*

	conv_loss��=[9\#        )��P	��jK���A�*

	conv_loss�S>��G@        )��P	��jK���A�*

	conv_losss�=�Ť�        )��P	tkK���A�*

	conv_loss���=O"+�        )��P	�JkK���A�*

	conv_loss���=����        )��P	�~kK���A�*

	conv_loss���= �H�        )��P	ײkK���A�*

	conv_loss���=<��        )��P	��kK���A�*

	conv_loss�) >�q��        )��P	�lK���A�*

	conv_loss~#�=U>��        )��P	�VlK���A�*

	conv_loss~a�=1Um        )��P	q�lK���A�*

	conv_loss�׺=F/W�        )��P	P�lK���A�*

	conv_loss�P�=[�	�        )��P	��lK���A�*

	conv_loss�'�=q�j�        )��P	�*mK���A�*

	conv_loss��=;        )��P	�]mK���A�*

	conv_loss���=b��U        )��P	�mK���A�*

	conv_lossSv�=����        )��P	��mK���A�*

	conv_lossӏ�=��s"        )��P	��mK���A�*

	conv_loss'g�=q
�        )��P	%,nK���A�*

	conv_loss��=̴#6        )��P	manK���A�*

	conv_loss@6�=��)7        )��P	��nK���A�*

	conv_loss��=*n<        )��P	v�nK���A�*

	conv_loss��=���        )��P	D�nK���A�*

	conv_loss?,�=$N&�        )��P	^/oK���A�*

	conv_loss�>Xrݭ        )��P	�boK���A�*

	conv_loss�~�=X<�}        )��P	v�oK���A�*

	conv_lossȂ�=ƣ�T        )��P	��oK���A�*

	conv_loss�2�=n��        )��P	��oK���A�*

	conv_loss�D�=�M��        )��P	�2pK���A�*

	conv_loss���=_�        )��P	$gpK���A�*

	conv_loss�W>�aR        )��P	@�pK���A�*

	conv_lossdҭ=��m�        )��P	��pK���A�*

	conv_losse8�=t��        )��P	$qK���A�*

	conv_loss��=__        )��P	�5qK���A�*

	conv_loss��>]�;�        )��P	�gqK���A�*

	conv_lossգ�=>{�        )��P	�qK���A�*

	conv_lossJ�=E]�,        )��P	d�qK���A�*

	conv_loss��=у'         )��P	�rK���A�*

	conv_losse>�_��        )��P	�8rK���A�*

	conv_loss��>(���        )��P	twK���A�*

	conv_loss�>5>2        )��P	��xK���A�*

	conv_loss)�=�Ș        )��P	)�xK���A�*

	conv_lossk��=o���        )��P	zyK���A�*

	conv_loss�c�=���        )��P	NyK���A�*

	conv_loss`��=�le        )��P	�}yK���A�*

	conv_loss�=��U�        )��P	l�yK���A�*

	conv_loss#��=+�]        )��P	_�yK���A�*

	conv_lossL]�=��        )��P	zK���A�*

	conv_loss���=\&�        )��P	)QzK���A�*

	conv_loss��=V�L        )��P	J�zK���A�*

	conv_lossJF�=Ɉ+�        )��P	��zK���A�*

	conv_loss3(�=��        )��P	��zK���A�*

	conv_loss��	>����        )��P	[{K���A�*

	conv_lossV�=�_��        )��P	bV{K���A�*

	conv_loss=|�=���        )��P	އ{K���A�*

	conv_loss���=r��        )��P	4�{K���A�*

	conv_lossX��=�y�}        )��P	��{K���A�*

	conv_loss%j�=%~�        )��P	�|K���A�*

	conv_lossµ=q_4�        )��P	�F|K���A�*

	conv_lossȭ�=ɃC        )��P	�v|K���A�*

	conv_loss�p�=�         )��P	�|K���A�*

	conv_loss0�=7Z�        )��P	,�|K���A�*

	conv_loss�2�=̂_        )��P	T}K���A�*

	conv_lossG��=b7-        )��P	�8}K���A�*

	conv_loss=I�=53��        )��P	�j}K���A�*

	conv_lossx�=b�G        )��P	R�}K���A�*

	conv_lossH��=S�XQ        )��P	��}K���A�*

	conv_loss@)�=(�'�        )��P	K�}K���A�*

	conv_loss\��=�g�        )��P	�.~K���A�*

	conv_loss�3�=��        )��P	k^~K���A�*

	conv_loss"��=��N�        )��P	Ռ~K���A�*

	conv_loss��=��        )��P	��~K���A�*

	conv_lossi�=�7��        )��P	l�~K���A�*

	conv_lossǲ=�o$R        )��P	�K���A�*

	conv_loss��=���$        )��P	�NK���A�*

	conv_loss���=VI��        )��P	t�K���A�*

	conv_loss"��=Jn�        )��P	��K���A�*

	conv_loss6��=���t        )��P	��K���A�*

	conv_loss�K�=�N�c        )��P	��K���A�*

	conv_loss���=��M        )��P	�@�K���A�*

	conv_loss5�=e]��        )��P	�s�K���A�*

	conv_lossn~�=����        )��P	���K���A�*

	conv_lossK�=�h�        )��P	A�K���A�*

	conv_loss�$�=�5�E        )��P	��K���A�*

	conv_loss��=�'��        )��P	�H�K���A�*

	conv_loss���=��R        )��P	Kw�K���A�*

	conv_loss��=�"k�        )��P	穁K���A�*

	conv_loss��=�m�        )��P	�ځK���A�*

	conv_loss�w�=-xLe        )��P	n�K���A�*

	conv_lossD��=�/�\        )��P	2@�K���A�*

	conv_loss�N�=1��        )��P	ǂ�K���A�*

	conv_loss���=��!        )��P	���K���A�*

	conv_loss	��=	�Y        )��P	��K���A�*

	conv_loss�]�=�$f�        )��P	��K���A�*

	conv_loss���=&���        )��P	�H�K���A�*

	conv_loss��=e��W        )��P	�z�K���A�*

	conv_loss��=?��r        )��P	q��K���A�*

	conv_loss� �=W"U        )��P	<܃K���A�*

	conv_loss��=S|�        )��P	��K���A�*

	conv_lossLV�=��C        )��P	,R�K���A�*

	conv_loss��=W��5        )��P	���K���A�*

	conv_loss�^�=2Kܠ        )��P	ִ�K���A�*

	conv_lossʤ�= ���        )��P	)�K���A�*

	conv_loss�W�=e�        )��P	%�K���A�*

	conv_lossJَ=�\�        )��P	�]�K���A�*

	conv_lossKJ�=i��P        )��P	ޕ�K���A�*

	conv_loss<��=�&D        )��P	�ŅK���A�*

	conv_loss���=c�!�        )��P	��K���A�*

	conv_loss��=��M        )��P	v%�K���A�*

	conv_lossMp�=i�ߪ        )��P	�V�K���A�*

	conv_loss�L�=l�%�        )��P	K���A�*

	conv_loss��=���        )��P	ڹ�K���A�*

	conv_loss�o�=�nD        )��P	��K���A�*

	conv_loss�T�=�BF        )��P	�"�K���A�*

	conv_loss﷬=r        )��P	+S�K���A�*

	conv_loss�^�==��x        )��P	䅇K���A�*

	conv_loss|�=)���        )��P	���K���A�*

	conv_loss���=����        )��P	��K���A�*

	conv_loss���=2��        )��P	!�K���A�*

	conv_loss${�=�        )��P	zE�K���A�*

	conv_loss��=�&�        )��P	�w�K���A�*

	conv_loss�Z�=��0        )��P	è�K���A�*

	conv_loss�'�=�f�        )��P	�؈K���A�*

	conv_loss���=_m�        )��P	�
�K���A�*

	conv_loss�϶=%�C        )��P	�=�K���A�*

	conv_loss;��=�p��        )��P	�n�K���A�*

	conv_lossi]�=J�o        )��P	���K���A�*

	conv_loss���=����        )��P	+ӉK���A�*

	conv_lossz�=Ν�        )��P	��K���A�*

	conv_loss�>
�>*        )��P	�7�K���A�*

	conv_lossRں=4��        )��P	+k�K���A�*

	conv_loss��=u���        )��P	���K���A�*

	conv_loss���=�)n        )��P	ԊK���A�*

	conv_loss�h�=<rk        )��P	��K���A�*

	conv_loss���=�Q�        )��P	a=�K���A�*

	conv_lossmu�=b���        )��P	�q�K���A�*

	conv_loss���=�u�        )��P	���K���A�*

	conv_loss@R�=H�x        )��P	@݋K���A�*

	conv_lossڦ�=�,t        )��P	5�K���A�*

	conv_lossb��=����        )��P	DH�K���A�*

	conv_lossޱ�=E�P{        )��P	呌K���A�*

	conv_loss�|�="�[F        )��P	�ƌK���A�*

	conv_lossť=,���        )��P	b��K���A�*

	conv_loss���=��Ho        )��P	f1�K���A�*

	conv_loss�}�=�{��        )��P	�f�K���A�*

	conv_loss�L�=���}        )��P	뜍K���A�*

	conv_loss	��=�篐        )��P	�ҍK���A�*

	conv_loss
U�=�J�X        )��P	{�K���A�*

	conv_lossh��=�N        )��P	�?�K���A�*

	conv_loss@�=�r��        )��P	�y�K���A�*

	conv_lossu}�=��w        )��P	N��K���A�*

	conv_loss��=o�/        )��P	k�K���A�*

	conv_loss�j�=\^�        )��P	5�K���A�*

	conv_loss���=�$�        )��P	K�K���A�*

	conv_loss�C�=Ul�        )��P	u��K���A�*

	conv_loss�Һ=�7        )��P	ƏK���A�*

	conv_loss���=��9�        )��P	���K���A�*

	conv_loss�>�V�        )��P	�1�K���A�*

	conv_loss��=l        )��P	�h�K���A�*

	conv_loss䭾=�B�        )��P	H��K���A�*

	conv_loss7Y�=���        )��P	`ҐK���A�*

	conv_loss��=��        )��P	��K���A�*

	conv_loss��=���Z        )��P	�=�K���A�*

	conv_lossZ!�=&?`        )��P	r�K���A�*

	conv_loss��=0���        )��P	���K���A�*

	conv_loss�ѻ=�s�        )��P	�ܑK���A�*

	conv_loss�ј=��Q        )��P	��K���A�*

	conv_loss�ε=�Я        )��P	�E�K���A�*

	conv_loss�ך=�F��        )��P	�z�K���A�*

	conv_loss�-=Wv�        )��P	[��K���A�*

	conv_loss2O�=��M�        )��P	�K���A�*

	conv_loss�K�=��50        )��P	��K���A�*

	conv_loss�ˌ={<�r        )��P	U�K���A�*

	conv_loss�߁=���
        )��P	+��K���A�*

	conv_loss���=�벟        )��P	���K���A�*

	conv_lossX��=Sp��        )��P	��K���A�*

	conv_lossGī=���        )��P	q'�K���A�*

	conv_loss��=ĹŚ        )��P	�Z�K���A�*

	conv_lossW�=�m��        )��P		��K���A�*

	conv_loss���=��?�        )��P	LĔK���A�*

	conv_losse��=��)        )��P	���K���A�*

	conv_loss��=iL�:        )��P	�,�K���A�*

	conv_loss��=�I��        )��P	b_�K���A�*

	conv_loss?{=��s�        )��P	���K���A�*

	conv_loss�p�=��5�        )��P	ɕK���A�*

	conv_loss�X�=���6        )��P	���K���A�*

	conv_loss��=k�̹        )��P	�0�K���A�*

	conv_loss�6�=���        )��P	d�K���A�*

	conv_loss=�=6��        )��P	h��K���A�*

	conv_loss'�=p��U        )��P	ΖK���A�*

	conv_losssF�=���	        )��P	?�K���A�*

	conv_lossǚ=�y7        )��P	vK�K���A�	*

	conv_loss���=X�!O        )��P	��K���A�	*

	conv_loss��=-|�        )��P	��K���A�	*

	conv_loss��=~�        )��P	<�K���A�	*

	conv_loss�H�=�0�        )��P	#�K���A�	*

	conv_loss3!�=�ˇ        )��P	X�K���A�	*

	conv_loss���=i�/O        )��P	�K���A�	*

	conv_loss�؇=�$M�        )��P	�ΘK���A�	*

	conv_lossS�=�,        )��P	>�K���A�	*

	conv_loss�V�=o�qU        )��P	�8�K���A�	*

	conv_loss��=.?��        )��P	�m�K���A�	*

	conv_lossy�=�	+        )��P	ʧ�K���A�	*

	conv_loss�=g�?�        )��P	��K���A�	*

	conv_loss���=���L        )��P	b�K���A�	*

	conv_loss���=4e        )��P	�O�K���A�	*

	conv_loss�e�=���        )��P	���K���A�	*

	conv_lossH�=0W��        )��P	���K���A�	*

	conv_loss\�=#��        )��P	�K���A�	*

	conv_lossr�=�c$-        )��P	�#�K���A�	*

	conv_loss�Ԕ=�NH�        )��P	NW�K���A�	*

	conv_loss�|�=��}        )��P	Ԋ�K���A�	*

	conv_loss���=���f        )��P	���K���A�	*

	conv_lossc$�=9�1        )��P	��K���A�	*

	conv_lossb��=��E�        )��P	c(�K���A�	*

	conv_lossД=9��~        )��P	�]�K���A�	*

	conv_loss�O�=�2        )��P	���K���A�	*

	conv_lossM2�=1;�        )��P	�ɜK���A�	*

	conv_loss��=����        )��P	���K���A�	*

	conv_loss0�=        )��P	3�K���A�	*

	conv_lossb��=V��H        )��P	�g�K���A�	*

	conv_loss9��=�F�<        )��P	���K���A�	*

	conv_loss��=g�        )��P	�ҝK���A�	*

	conv_lossR0�=b��        )��P	�
�K���A�	*

	conv_loss�,�=[�$�        )��P	�@�K���A�	*

	conv_loss8�=!�@=        )��P	�v�K���A�	*

	conv_loss�U�=L
�        )��P	���K���A�	*

	conv_loss���=��        )��P	c�K���A�	*

	conv_loss�=�5��        )��P	�K���A�	*

	conv_lossKt�=�G��        )��P	Q�K���A�	*

	conv_loss�D�=�        )��P	�K���A�	*

	conv_loss��=3�'{        )��P	T��K���A�	*

	conv_losso+�=F0?�        )��P	��K���A�	*

	conv_lossܯ�=MiL�        )��P	"�K���A�	*

	conv_loss+��=~��        )��P	�V�K���A�	*

	conv_loss�T�=v�7        )��P	X��K���A�	*

	conv_loss}��=p���        )��P	� K���A�	*

	conv_lossK`�=ew��        )��P	[��K���A�	*

	conv_lossh��=��9�        )��P	�-�K���A�	*

	conv_loss�%�=��r�        )��P	�c�K���A�	*

	conv_loss�=�]�X        )��P	���K���A�	*

	conv_loss��=��C        )��P	�-�K���A�	*

	conv_loss=6�=0r2�        )��P	<a�K���A�	*

	conv_loss'�=�/        )��P	���K���A�	*

	conv_loss��|=�B��        )��P	(ˣK���A�	*

	conv_lossJ.�=I��        )��P	?��K���A�	*

	conv_loss��=���        )��P	!5�K���A�	*

	conv_loss���=QV0        )��P	�h�K���A�	*

	conv_lossƎ�=:�	        )��P	O��K���A�	*

	conv_loss�|v=�y�        )��P	�٤K���A�	*

	conv_loss�N�=5i��        )��P	��K���A�	*

	conv_loss���=
�        )��P	fO�K���A�	*

	conv_lossow�=��_�        )��P	���K���A�	*

	conv_loss4ԛ=W�        )��P	"��K���A�	*

	conv_losss�=�˓�        )��P	��K���A�	*

	conv_losse��=�O"        )��P	��K���A�	*

	conv_loss�Ĳ=@4[�        )��P	\W�K���A�	*

	conv_loss5Ɠ=��Q2        )��P	(��K���A�	*

	conv_loss#�=K��        )��P	���K���A�	*

	conv_loss���=�5<[        )��P	���K���A�	*

	conv_loss�0�=�m�        )��P	�0�K���A�	*

	conv_loss�ؑ=��\        )��P	Ud�K���A�	*

	conv_lossڇ�=�vb�        )��P	v��K���A�	*

	conv_loss���=S!j        )��P	$ʧK���A�	*

	conv_lossB�=(�?�        )��P	���K���A�	*

	conv_lossBU�=(�	�        )��P	)2�K���A�	*

	conv_loss�H�=t�        )��P	uc�K���A�	*

	conv_loss��=U]��        )��P	*��K���A�	*

	conv_lossƳ�="Rf|        )��P	IɨK���A�	*

	conv_loss�2�=�N��        )��P	'��K���A�	*

	conv_loss6f�=�#��        )��P	�/�K���A�	*

	conv_loss�@�=']�F        )��P	�c�K���A�	*

	conv_loss0��=^�2-        )��P	T��K���A�	*

	conv_lossJL�=2�o        )��P	�ʩK���A�	*

	conv_loss�s=6_	        )��P	���K���A�	*

	conv_loss��C=fI'�        )��P	+/�K���A�	*

	conv_loss���=<�        )��P	nd�K���A�	*

	conv_loss���=�c�        )��P	g��K���A�	*

	conv_lossz �=�        )��P	ɪK���A�	*

	conv_loss�N�=%8�        )��P	���K���A�	*

	conv_loss��=���W        )��P	�.�K���A�	*

	conv_losse��=�.\�        )��P	�a�K���A�	*

	conv_lossp�=Q�8        )��P	���K���A�	*

	conv_lossP��=��]        )��P	tūK���A�	*

	conv_loss�j�=��4�        )��P	��K���A�	*

	conv_lossz�=g0�        )��P	/�K���A�	*

	conv_loss>5�=�%        )��P	kb�K���A�	*

	conv_loss^��=�]�        )��P	g��K���A�	*

	conv_loss��=pw�K        )��P	�ɬK���A�	*

	conv_loss�ժ=���        )��P	���K���A�	*

	conv_loss��=i��        )��P	�A�K���A�	*

	conv_loss$�=U�K;        )��P	�u�K���A�	*

	conv_loss�̱=��e        )��P	:��K���A�	*

	conv_lossg=�=�*��        )��P	�ޭK���A�	*

	conv_loss���=�)�G        )��P	B�K���A�	*

	conv_loss�^�=Nܰ�        )��P	XB�K���A�	*

	conv_loss�Z�=C��        )��P	T��K���A�	*

	conv_lossP �=@��        )��P	���K���A�	*

	conv_loss�`�=�b}�        )��P	/�K���A�	*

	conv_loss�<�=�h��        )��P	W%�K���A�	*

	conv_loss -�=��J�        )��P	 X�K���A�	*

	conv_loss�ƨ=�hx        )��P	N��K���A�	*

	conv_loss�=��u�        )��P	3ïK���A�	*

	conv_lossY��=
�=B        )��P	���K���A�	*

	conv_loss�=tˌ�        )��P	,�K���A�	*

	conv_lossY��=|S��        )��P	}d�K���A�	*

	conv_loss���=����        )��P	��K���A�	*

	conv_loss���=�]��        )��P	�аK���A�	*

	conv_lossHCr=��        )��P	[�K���A�	*

	conv_loss�.�=��+�        )��P	L6�K���A�	*

	conv_loss3N�=�(�        )��P	�g�K���A�	*

	conv_loss^��=�^�        )��P	��K���A�	*

	conv_loss���=(N&        )��P	ͱK���A�	*

	conv_loss䯩=r��        )��P	l��K���A�	*

	conv_loss�@�=��ޯ        )��P	'2�K���A�	*

	conv_lossؘ�=%*�        )��P	]e�K���A�	*

	conv_lossW�=fx>        )��P	���K���A�	*

	conv_loss�7�=�J;4        )��P	b̲K���A�	*

	conv_loss	�=��^n        )��P	���K���A�	*

	conv_loss�i�=6i�        )��P	�2�K���A�
*

	conv_loss�b�=�+��        )��P	�f�K���A�
*

	conv_lossx��=�9�        )��P	:��K���A�
*

	conv_loss筡=O�8        )��P	%˳K���A�
*

	conv_loss�$p=s���        )��P	��K���A�
*

	conv_loss.φ=c?�$        )��P	�-�K���A�
*

	conv_loss$�=���        )��P	�_�K���A�
*

	conv_loss䇟=�	m�        )��P	7��K���A�
*

	conv_loss:�=��I}        )��P	�ôK���A�
*

	conv_loss��=� �~        )��P	���K���A�
*

	conv_loss�V^=�B��        )��P	,�K���A�
*

	conv_loss<�=m��        )��P	j^�K���A�
*

	conv_loss��=��J        )��P	���K���A�
*

	conv_loss�ư=�n�        )��P	�ĵK���A�
*

	conv_lossEZ�=����        )��P	X��K���A�
*

	conv_losstii=���        )��P	�-�K���A�
*

	conv_loss�s�=���        )��P	ka�K���A�
*

	conv_loss��=V�'E        )��P	,��K���A�
*

	conv_loss�9�=���1        )��P	�ǶK���A�
*

	conv_loss���="җi        )��P	���K���A�
*

	conv_loss�J�=	���        )��P	�.�K���A�
*

	conv_lossv��=��        )��P	zu�K���A�
*

	conv_lossa�=J�}�        )��P	Χ�K���A�
*

	conv_loss�L�=�I�&        )��P	�طK���A�
*

	conv_loss��=�tK�        )��P	��K���A�
*

	conv_loss��S=u툽        )��P	hG�K���A�
*

	conv_loss� �=�E9�        )��P	wz�K���A�
*

	conv_loss�%�=!6ȕ        )��P	.��K���A�
*

	conv_losssV�=�=rP        )��P	��K���A�
*

	conv_losse��=��        )��P	J�K���A�
*

	conv_loss�~�=��%7        )��P	�V�K���A�
*

	conv_loss���=&� �        )��P	���K���A�
*

	conv_loss�]�=G�w        )��P	&˹K���A�
*

	conv_loss��=M�!5        )��P	r��K���A�
*

	conv_loss��d=ŀ�        )��P	�2�K���A�
*

	conv_loss;*�=�EC�        )��P	�j�K���A�
*

	conv_loss��^=��c        )��P	d��K���A�
*

	conv_losso�=/	�	        )��P	�ֺK���A�
*

	conv_loss�>�=�<�        )��P	��K���A�
*

	conv_loss�f�=��        )��P	K@�K���A�
*

	conv_loss�/�=�+H"        )��P	u�K���A�
*

	conv_lossǼ�=��r	        )��P	I��K���A�
*

	conv_lossP��=)%|        )��P	JڻK���A�
*

	conv_loss裨=_k+        )��P	��K���A�
*

	conv_loss�x�=S�Ѝ        )��P	NA�K���A�
*

	conv_loss�,�=���        )��P	�s�K���A�
*

	conv_lossh��=�j�[        )��P	å�K���A�
*

	conv_loss���=���        )��P	=ټK���A�
*

	conv_losskC�=����        )��P	��K���A�
*

	conv_lossx#�=�6o�        )��P	@�K���A�
*

	conv_loss���=M�Q,        )��P	cr�K���A�
*

	conv_loss�څ=���Z        )��P	���K���A�
*

	conv_loss��=�@�        )��P	�׽K���A�
*

	conv_lossk5l=����        )��P	�	�K���A�
*

	conv_loss7�=�Jv�        )��P	�;�K���A�
*

	conv_loss<S=�T�)        )��P	�n�K���A�
*

	conv_lossJ�m=H��5        )��P	Z��K���A�
*

	conv_loss�=|�>        )��P	�ѾK���A�
*

	conv_lossǔ�=ԭI�        )��P	=�K���A�
*

	conv_loss!��=S#�        )��P	�6�K���A�
*

	conv_lossf�=W��c        )��P	}i�K���A�
*

	conv_lossB�`=��4�        )��P	���K���A�
*

	conv_loss>��=�� <        )��P	�пK���A�
*

	conv_loss�~�=���        )��P	E�K���A�
*

	conv_lossU�=�Q$        )��P	59�K���A�
*

	conv_losspc�=\+��        )��P	�l�K���A�
*

	conv_losseAs=���}        )��P	���K���A�
*

	conv_loss���='ol        )��P	%��K���A�
*

	conv_loss�D�=�e�        )��P	��K���A�
*

	conv_loss�=w��J        )��P	48�K���A�
*

	conv_loss��=�P*        )��P	�k�K���A�
*

	conv_loss�*�=�E|        )��P	#��K���A�
*

	conv_loss�-u=���        )��P	���K���A�
*

	conv_loss�&E=�_�        )��P	�K���A�
*

	conv_loss���=e�U.        )��P	�K�K���A�
*

	conv_lossٍ�=�4z�        )��P	1��K���A�
*

	conv_loss���=�1?�        )��P	��K���A�
*

	conv_loss^�=���        )��P	��K���A�
*

	conv_lossx�=���        )��P	�'�K���A�
*

	conv_loss�͂= ��        )��P	�^�K���A�
*

	conv_lossoΙ=�<8        )��P	Җ�K���A�
*

	conv_loss�fp=4���        )��P	#��K���A�
*

	conv_loss9k�=4�`$        )��P	1��K���A�
*

	conv_loss攋=� St        )��P	5�K���A�
*

	conv_lossA�=���N        )��P	�r�K���A�
*

	conv_loss�H�=Mqm        )��P	ç�K���A�
*

	conv_loss�=ևI        )��P	���K���A�
*

	conv_loss�i�=QHh�        )��P	��K���A�
*

	conv_loss�3{=�5�;        )��P	.A�K���A�
*

	conv_lossg�\=B��        )��P	qv�K���A�
*

	conv_loss�bg=��d        )��P	���K���A�
*

	conv_loss�=^2        )��P	l��K���A�
*

	conv_loss���=�A�        )��P	D�K���A�
*

	conv_loss��=>E��        )��P	�>�K���A�
*

	conv_loss(�p=y�W[        )��P	�o�K���A�
*

	conv_lossdQ=�c;�        )��P	���K���A�
*

	conv_loss_k=��!d        )��P	���K���A�
*

	conv_loss(l�=�C        )��P	��K���A�
*

	conv_loss�A�=�I�        )��P	�;�K���A�
*

	conv_loss<�|='�[�        )��P	n�K���A�
*

	conv_loss���=y��        )��P	��K���A�
*

	conv_loss&��=dȽ        )��P	���K���A�
*

	conv_loss\�^=&��        )��P	<	�K���A�
*

	conv_loss���=ʾ��        )��P	f=�K���A�
*

	conv_losssc�=(�        )��P	[o�K���A�
*

	conv_loss�=���        )��P	��K���A�
*

	conv_loss$!�=��O        )��P	���K���A�
*

	conv_loss�#�=�I��        )��P	�	�K���A�
*

	conv_loss*��=T{�        )��P	:<�K���A�
*

	conv_loss۫�=�j�        )��P	p�K���A�
*

	conv_loss't�=6nP:        )��P	z��K���A�
*

	conv_loss�=�]�        )��P	D��K���A�
*

	conv_loss�(�=���|        )��P	c	�K���A�
*

	conv_loss�*�=� �k        )��P	�;�K���A�
*

	conv_loss�h�=��h�        )��P	�o�K���A�
*

	conv_loss���=E5�        )��P	���K���A�
*

	conv_lossU=틪        )��P	���K���A�
*

	conv_loss4��=�m�        )��P	_�K���A�
*

	conv_lossH��=�c
3        )��P	P7�K���A�
*

	conv_loss��=�48:        )��P	k�K���A�
*

	conv_loss챟=�s |        )��P	���K���A�
*

	conv_loss7a�=@��        )��P	4�K���A�
*

	conv_loss/-Y=V�B�        )��P	�g�K���A�
*

	conv_loss�,�=��        )��P	̚�K���A�
*

	conv_loss��=���        )��P	���K���A�
*

	conv_loss���=5OL        )��P	m�K���A�
*

	conv_loss-WR=���        )��P	�3�K���A�
*

	conv_lossL[=�eN4        )��P	u�K���A�
*

	conv_loss�=�Z�        )��P	��K���A�*

	conv_loss飖=ǵ�        )��P	]��K���A�*

	conv_loss��= ��        )��P	(�K���A�*

	conv_loss7M�=գ,8        )��P	8d�K���A�*

	conv_loss��=��3�        )��P	���K���A�*

	conv_loss̛q=����        )��P	���K���A�*

	conv_lossR��=8�3d        )��P	'��K���A�*

	conv_loss��=@ �        )��P	r0�K���A�*

	conv_loss��=j��        )��P	Ib�K���A�*

	conv_loss��=��        )��P	���K���A�*

	conv_loss͗�=��{        )��P	���K���A�*

	conv_lossb+�=��Ѐ        )��P	� �K���A�*

	conv_losss�I=z��z        )��P	�6�K���A�*

	conv_loss b6=6��        )��P	Um�K���A�*

	conv_loss��=;]��        )��P	K��K���A�*

	conv_lossӍo=��0�        )��P	<��K���A�*

	conv_loss<�T=�Ϊ�        )��P	��K���A�*

	conv_loss#`f=���,        )��P	C;�K���A�*

	conv_loss�	�=4G�        )��P	�o�K���A�*

	conv_losscN�=0�F�        )��P	á�K���A�*

	conv_loss�=�N�        )��P	���K���A�*

	conv_loss �=}ʗ�        )��P	O�K���A�*

	conv_losso��=@�ؗ        )��P	w1�K���A�*

	conv_loss�V|=z�&        )��P	�a�K���A�*

	conv_loss�@=�V�)        )��P	#��K���A�*

	conv_loss���=q0	4        )��P	���K���A�*

	conv_loss���=�Q�        )��P	a��K���A�*

	conv_loss���=��t>        )��P	��K���A�*

	conv_lossg�z=){s&        )��P	I�K���A�*

	conv_loss4�=�n��        )��P	�v�K���A�*

	conv_lossf}=��h�        )��P	���K���A�*

	conv_loss�r=nz	�        )��P	���K���A�*

	conv_loss�"�=TZ��        )��P	u�K���A�*

	conv_loss�6{=���        )��P	�3�K���A�*

	conv_loss��=�Ė�        )��P	Xc�K���A�*

	conv_loss�D�=EK=        )��P	���K���A�*

	conv_lossQ�y=���        )��P	 ��K���A�*

	conv_loss�>�=���S        )��P	\��K���A�*

	conv_lossh��=��:x        )��P	<"�K���A�*

	conv_loss��=��&        )��P	VR�K���A�*

	conv_lossWP�=��}        )��P	o��K���A�*

	conv_lossk2�=�R3s        )��P	��K���A�*

	conv_loss-�^=�_[        )��P	���K���A�*

	conv_loss�+p=�$�        )��P	��K���A�*

	conv_loss�[�=4��*        )��P	O�K���A�*

	conv_loss`�=��        )��P	,�K���A�*

	conv_loss3$�=w��_        )��P	Ϭ�K���A�*

	conv_lossl��=��,        )��P	[��K���A�*

	conv_loss���=G��        )��P	)�K���A�*

	conv_loss=
p=�R7�        )��P	QG�K���A�*

	conv_lossv��=��d        )��P	)y�K���A�*

	conv_loss$��=�         )��P	���K���A�*

	conv_loss��=,t$        )��P	��K���A�*

	conv_loss��{=���        )��P	�%�K���A�*

	conv_loss�=�]��        )��P	�T�K���A�*

	conv_loss���=�<�5        )��P	m��K���A�*

	conv_loss/u�=�[d        )��P	G��K���A�*

	conv_loss롈="���        )��P	^��K���A�*

	conv_loss��H=\�;c        )��P	��K���A�*

	conv_loss�"�=���        )��P	�B�K���A�*

	conv_lossKQ�=Ϋ/�        )��P	�o�K���A�*

	conv_lossI��=����        )��P	ĝ�K���A�*

	conv_loss6��=��|        )��P	#��K���A�*

	conv_loss��=0�#        )��P	$��K���A�*

	conv_loss��0=���        )��P	:1�K���A�*

	conv_loss��=W�Y        )��P	c�K���A�*

	conv_lossc��=[ ]        )��P	~��K���A�*

	conv_loss[zo=<_@�        )��P	X��K���A�*

	conv_loss+��=���h        )��P	"��K���A�*

	conv_lossPZ�=W�'z        )��P	#�K���A�*

	conv_loss��=:���        )��P	iR�K���A�*

	conv_loss��L=/F�        )��P	���K���A�*

	conv_loss��=g��        )��P	��K���A�*

	conv_loss0�=�N"        )��P	���K���A�*

	conv_loss4b�=��s        )��P	��K���A�*

	conv_loss���=lZ        )��P	�>�K���A�*

	conv_loss�F�=����        )��P	#o�K���A�*

	conv_loss_Ύ=�I!|        )��P	��K���A�*

	conv_loss���=����        )��P	���K���A�*

	conv_loss�w}=qط%        )��P	m��K���A�*

	conv_loss��:=8OO        )��P	�/�K���A�*

	conv_loss؁�=���        )��P	8`�K���A�*

	conv_loss�zr=��n        )��P	я�K���A�*

	conv_losse�=ŭ��        )��P	ƾ�K���A�*

	conv_loss	7�={���        )��P	���K���A�*

	conv_loss>}=���        )��P	�!�K���A�*

	conv_loss�ki=�2�+        )��P	)U�K���A�*

	conv_losseR�=ck��        )��P	���K���A�*

	conv_loss�~=�x�        )��P	Ը�K���A�*

	conv_lossޅP=�2�        )��P	G��K���A�*

	conv_loss��=:l�        )��P	��K���A�*

	conv_loss�}=x�G        )��P	�N�K���A�*

	conv_loss�1�=�K#        )��P	���K���A�*

	conv_lossXؤ=�}��        )��P	��K���A�*

	conv_losszr�=I��        )��P	�:�K���A�*

	conv_loss�=����        )��P	`j�K���A�*

	conv_loss�r�=����        )��P	��K���A�*

	conv_loss���=�~�O        )��P	���K���A�*

	conv_loss*э=�+R�        )��P	��K���A�*

	conv_lossJ��=N���        )��P	R7�K���A�*

	conv_loss��=�w�#        )��P	�e�K���A�*

	conv_lossAU=Ҕަ        )��P	���K���A�*

	conv_loss�^==�6�k        )��P	��K���A�*

	conv_loss] �=o�,        )��P	j��K���A�*

	conv_loss��=M�#        )��P	�,�K���A�*

	conv_losskԧ=޿yQ        )��P	�[�K���A�*

	conv_loss7�x=���        )��P	���K���A�*

	conv_loss���=t�r�        )��P	��K���A�*

	conv_loss�-�=�
 1        )��P	���K���A�*

	conv_lossKA�=�~�h        )��P	7#�K���A�*

	conv_loss�,�=Si�$        )��P	Q�K���A�*

	conv_loss�S=�%�-        )��P	i�K���A�*

	conv_lossX�U=�r        )��P	v��K���A�*

	conv_lossƀ�=��t1        )��P	$��K���A�*

	conv_loss��~=��|�        )��P	��K���A�*

	conv_loss�ؘ=o�J        )��P	x>�K���A�*

	conv_loss��n=4�8�        )��P	n�K���A�*

	conv_lossB�f=��)T        )��P	&��K���A�*

	conv_loss���=����        )��P	��K���A�*

	conv_loss$ϔ=���        )��P	���K���A�*

	conv_loss��h=�F�        )��P	�B�K���A�*

	conv_loss^��=L��        )��P	Gr�K���A�*

	conv_loss��=t���        )��P	t��K���A�*

	conv_loss�lV=p�{        )��P	���K���A�*

	conv_loss�kb=���        )��P	���K���A�*

	conv_loss�:�=�+p        )��P	�*�K���A�*

	conv_loss[t]=���n        )��P	�Y�K���A�*

	conv_loss�l�=[^V�        )��P	���K���A�*

	conv_lossy=��        )��P	+��K���A�*

	conv_loss%�Y=f��[        )��P	���K���A�*

	conv_lossF�=�K�4        )��P	��K���A�*

	conv_lossL��=B���        )��P	}I�K���A�*

	conv_lossG:u=���X        )��P	z�K���A�*

	conv_loss��t=��h        )��P	���K���A�*

	conv_loss~��=x�        )��P	���K���A�*

	conv_loss��=l=�f        )��P	��K���A�*

	conv_lossA-�=ky�a        )��P	6�K���A�*

	conv_loss�X=�U��        )��P	gc�K���A�*

	conv_loss�O�=�ns        )��P	4��K���A�*

	conv_loss޾E=���        )��P	w��K���A�*

	conv_loss"n�=���^        )��P	;��K���A�*

	conv_loss�%x=�3~�        )��P	��K���A�*

	conv_loss �B=K�{        )��P	<O�K���A�*

	conv_loss�ݟ=N�+        )��P	���K���A�*

	conv_loss���=��+        )��P		��K���A�*

	conv_lossy�=}��        )��P	"��K���A�*

	conv_loss?u:=8�^�        )��P	|%�K���A�*

	conv_lossrq=���        )��P	�[�K���A�*

	conv_loss�w�=���        )��P	+��K���A�*

	conv_loss�X=����        )��P	���K���A�*

	conv_loss�u=��P        )��P	=��K���A�*

	conv_loss�~~=L��:        )��P	r�K���A�*

	conv_loss�q�=��#0        )��P	�X�K���A�*

	conv_loss�	�=}�,h        )��P	��K���A�*

	conv_loss�I�=�`l�        )��P	���K���A�*

	conv_lossܮ�=)�.U        )��P	��K���A�*

	conv_loss
�;=�F��        )��P	��K���A�*

	conv_loss�G=���        )��P	~Z�K���A�*

	conv_loss�a�=�6._        )��P	���K���A�*

	conv_loss+�=�m��        )��P	���K���A�*

	conv_loss��}=�L�        )��P	��K���A�*

	conv_loss�y�=���V        )��P	~0�K���A�*

	conv_loss��=͕L8        )��P	=d�K���A�*

	conv_loss��y=VQ�        )��P	��K���A�*

	conv_lossm|=O.��        )��P	k��K���A�*

	conv_loss_GS=�~�`        )��P	���K���A�*

	conv_loss�'2=붏�        )��P	0�K���A�*

	conv_loss	��=�;        )��P	:a�K���A�*

	conv_loss���=�E�Z        )��P	V��K���A�*

	conv_loss���=W�        )��P	���K���A�*

	conv_lossn�=�hQ`        )��P	���K���A�*

	conv_lossh��=�C�        )��P	})�K���A�*

	conv_loss��=��+        )��P	kZ�K���A�*

	conv_loss^�'=՘�        )��P	���K���A�*

	conv_loss��,=�X��        )��P	0��K���A�*

	conv_loss�P=��"        )��P	���K���A�*

	conv_loss���=�:�        )��P	��K���A�*

	conv_losst^�=�M�        )��P	�Q�K���A�*

	conv_loss�1S=�!�        )��P	'��K���A�*

	conv_loss�G|=�)�        )��P	���K���A�*

	conv_loss���=`5��        )��P	���K���A�*

	conv_loss���=��F�        )��P	��K���A�*

	conv_losst��=���        )��P	�C�K���A�*

	conv_loss��=5�q        )��P	'v�K���A�*

	conv_lossÏ�=����        )��P	��K���A�*

	conv_loss
��=���U        )��P	���K���A�*

	conv_lossª|=�q�        )��P	��K���A�*

	conv_lossk�{=�g�        )��P	�<�K���A�*

	conv_lossY[g=��tk        )��P	�q�K���A�*

	conv_loss�m�=Ť�:        )��P	��K���A�*

	conv_loss	�p=#��(        )��P	���K���A�*

	conv_loss�)�=%���        )��P	u�K���A�*

	conv_lossw�=¨��        )��P	�7�K���A�*

	conv_lossb�S=��%,        )��P	)��K���A�*

	conv_lossW4�=��g�        )��P	��K���A�*

	conv_loss��e=�*5�        )��P	4�K���A�*

	conv_loss��?=�DP        )��P	h�K���A�*

	conv_lossde^=��h�        )��P	Ø�K���A�*

	conv_loss�̟=�
�.        )��P	���K���A�*

	conv_loss̱�=M�I8        )��P	���K���A�*

	conv_loss��=缾v        )��P	�0�K���A�*

	conv_lossJ�\=�Թ�        )��P	 j�K���A�*

	conv_lossEge=��0�        )��P	r��K���A�*

	conv_lossF�=��{�        )��P	���K���A�*

	conv_loss��=�6        )��P	M�K���A�*

	conv_loss��=���        )��P	7P�K���A�*

	conv_loss|�x=��/�        )��P	y��K���A�*

	conv_loss��=]r��        )��P	��K���A�*

	conv_loss�P=�Z�        )��P	���K���A�*

	conv_loss��u=k3�        )��P	��K���A�*

	conv_loss��M=Z�\�        )��P	�N�K���A�*

	conv_loss.}x=|�+�        )��P	��K���A�*

	conv_loss ^�=�^$W        )��P	��K���A�*

	conv_loss��=Sm�6        )��P	���K���A�*

	conv_lossx��=�a~w        )��P	��K���A�*

	conv_loss�Q=@��]        )��P	'J�K���A�*

	conv_loss�7�=�Tε        )��P	�|�K���A�*

	conv_loss�+O=_���        )��P	���K���A�*

	conv_loss!u�=� f�        )��P	���K���A�*

	conv_loss���=.`N�        )��P	��K���A�*

	conv_loss��Z=t��        )��P	�E�K���A�*

	conv_loss�=�%A�        )��P	�v�K���A�*

	conv_lossc�R=�AO�        )��P	���K���A�*

	conv_loss��=�0�        )��P	���K���A�*

	conv_loss��=���J        )��P	� L���A�*

	conv_loss�ǜ=�D�        )��P	RB L���A�*

	conv_lossoZ=�<        )��P	�s L���A�*

	conv_loss�M�=l&{�        )��P	�� L���A�*

	conv_loss��F=gz"        )��P	M� L���A�*

	conv_lossn�q=�A1c        )��P	�L���A�*

	conv_loss;A�=����        )��P	2?L���A�*

	conv_loss�j=W�B        )��P	�pL���A�*

	conv_losss�l=�շ�        )��P	ơL���A�*

	conv_loss�Rc=}8�        )��P	��L���A�*

	conv_loss
�=2�2�        )��P	ZL���A�*

	conv_loss���=c]p        )��P	:L���A�*

	conv_loss�"=f�F�        )��P	=lL���A�*

	conv_loss�g�= `�        )��P	'�L���A�*

	conv_loss�Ya=#[=�        )��P	[�L���A�*

	conv_loss|M�=��I        )��P	L���A�*

	conv_loss�ׁ='z-p        )��P	)6L���A�*

	conv_loss�T=�vU        )��P	~hL���A�*

	conv_lossJ�V=y,��        )��P	�L���A�*

	conv_loss���=O��         )��P	��L���A�*

	conv_loss�"�=�4L�        )��P	}L���A�*

	conv_loss���=m�U        )��P	�DL���A�*

	conv_lossяO=�z�        )��P	�zL���A�*

	conv_loss!j=�gq2        )��P	��L���A�*

	conv_loss���=Q��        )��P	��L���A�*

	conv_loss�ΐ=���        )��P	�L���A�*

	conv_lossed�=�|��        )��P	@NL���A�*

	conv_loss�1-=*�ߑ        )��P	4�L���A�*

	conv_loss4�u=E	(�        )��P	޽L���A�*

	conv_lossb��=��B�        )��P	L���A�*

	conv_loss�%e=�Ĩ        )��P	�4L���A�*

	conv_loss2mH=՜��        )��P	ekL���A�*

	conv_losszw}=�м"        )��P	�L���A�*

	conv_loss�m�=Q3�        )��P	��L���A�*

	conv_lossx�p=w*3�        )��P	�L���A�*

	conv_losse��=�L7         )��P	I8L���A�*

	conv_lossʋo=�\^�        )��P	)hL���A�*

	conv_lossxݼ=���d        )��P	ÛL���A�*

	conv_loss���=sn��        )��P	��L���A�*

	conv_loss�%g=�&�        )��P	�L���A�*

	conv_lossW�z=���$        )��P	l.L���A�*

	conv_loss��_=.��%        )��P	aL���A�*

	conv_lossQb�=�n��        )��P	�L���A�*

	conv_loss��b=�Ō        )��P	��L���A�*

	conv_loss�lp=����        )��P	_�L���A�*

	conv_lossE�=����        )��P	�)	L���A�*

	conv_loss�X=�        )��P	�[	L���A�*

	conv_losse�=�(��        )��P	�	L���A�*

	conv_loss+7:=�<�Z        )��P	��	L���A�*

	conv_loss�ʲ=5`�        )��P	��	L���A�*

	conv_loss��=��|�        )��P	$
L���A�*

	conv_loss��v=r�W�        )��P	�U
L���A�*

	conv_loss�%=4�y�        )��P	��
L���A�*

	conv_lossRX�=$�&�        )��P	�
L���A�*

	conv_loss�dA=�G        )��P	�
L���A�*

	conv_lossk�+=S�ŉ        )��P	�$L���A�*

	conv_loss�@=n�.�        )��P	~VL���A�*

	conv_loss%�y=���m        )��P	ÈL���A�*

	conv_lossg��=���        )��P	^�L���A�*

	conv_loss��=|2��        )��P	_�L���A�*

	conv_loss�u�=�
��        )��P	�L���A�*

	conv_loss�7{=�"�        )��P	�OL���A�*

	conv_loss=Y�=���        )��P	�L���A�*

	conv_loss�='=�8�R        )��P	_�L���A�*

	conv_loss`j�=A=�        )��P	m�L���A�*

	conv_lossw�=��        )��P	L���A�*

	conv_loss��J=1t�        )��P	^HL���A�*

	conv_loss�sM=HyB�        )��P	�xL���A�*

	conv_loss���={��X        )��P	��L���A�*

	conv_loss�X�=����        )��P	��L���A�*

	conv_lossC+X=�y�        )��P	�"L���A�*

	conv_loss#�=s�E�        )��P	bUL���A�*

	conv_loss�	�=����        )��P	ČL���A�*

	conv_losse�i=�u�.        )��P	�L���A�*

	conv_loss�6�=T1�        )��P	��L���A�*

	conv_loss{K=�_��        )��P	�6L���A�*

	conv_loss �J=}4��        )��P	KgL���A�*

	conv_loss~��=��V�        )��P	]�L���A�*

	conv_loss��m=(M        )��P	��L���A�*

	conv_loss_G�=��u�        )��P	hL���A�*

	conv_lossJ�=,c��        )��P	XL���A�*

	conv_loss�Ò=qW��        )��P	�L���A�*

	conv_loss��^=��|�        )��P	��L���A�*

	conv_loss�UZ=�D        )��P	(�L���A�*

	conv_loss�=�:�p        )��P	�$L���A�*

	conv_loss≘=Ls`o        )��P	]VL���A�*

	conv_loss%J�=��E3        )��P	��L���A�*

	conv_loss�8=!B�        )��P	��L���A�*

	conv_lossk!P=�k�        )��P	��L���A�*

	conv_lossǮt=�	H�        )��P	#L���A�*

	conv_lossWx�=�r�{        )��P	]L���A�*

	conv_lossl�)=~��U        )��P	��L���A�*

	conv_loss��}=��        )��P	��L���A�*

	conv_lossF{^=�j��        )��P	9L���A�*

	conv_loss��=+l��        )��P	�6L���A�*

	conv_loss�(S={�M        )��P	�gL���A�*

	conv_lossR4n=LO�6        )��P	E�L���A�*

	conv_loss �=y��        )��P	V�L���A�*

	conv_loss��W=%륯        )��P	O�L���A�*

	conv_losss\�=0�?�        )��P	|.L���A�*

	conv_loss2T�=��        )��P	aL���A�*

	conv_lossG=:ʦ        )��P	��L���A�*

	conv_loss�=]���        )��P	]�L���A�*

	conv_loss��B=���        )��P	��L���A�*

	conv_loss*�=AC��        )��P	U.L���A�*

	conv_loss\��=��b        )��P	$`L���A�*

	conv_loss�-V=����        )��P	/�L���A�*

	conv_loss�}@=����        )��P	��L���A�*

	conv_lossru=4��        )��P	r�L���A�*

	conv_loss�D=?�D        )��P	�$L���A�*

	conv_loss��=� ��        )��P	�WL���A�*

	conv_loss��=�;`�        )��P	Z�L���A�*

	conv_loss�o�<��K`        )��P	ܹL���A�*

	conv_loss�i=��q?        )��P	a�L���A�*

	conv_loss�Wu=u�        )��P	�L���A�*

	conv_loss�ld=��(�        )��P	7QL���A�*

	conv_loss�ŏ=+�c�        )��P	��L���A�*

	conv_loss���=�WH        )��P	��L���A�*

	conv_loss�u.=�#ԇ        )��P	O�L���A�*

	conv_loss�G�=�l�        )��P	�*L���A�*

	conv_lossa	==�        )��P	�]L���A�*

	conv_loss�͒=�< o        )��P	�L���A�*

	conv_loss�=�Z��        )��P	8�L���A�*

	conv_lossj�=t��q        )��P	:�L���A�*

	conv_loss�P=u��U        )��P	�-L���A�*

	conv_lossU�#=��N6        )��P	O`L���A�*

	conv_loss5AT=S_�l        )��P	ǚL���A�*

	conv_loss��=�L��        )��P	��L���A�*

	conv_loss؏=;�        )��P	L���A�*

	conv_lossZ	s=��	        )��P	PLL���A�*

	conv_loss�>k=j�Ǌ        )��P	0�L���A�*

	conv_lossk*=lQTu        )��P	��L���A�*

	conv_losss�H=����        )��P	��L���A�*

	conv_loss(ie=���        )��P	� L���A�*

	conv_loss=�U=rX�c        )��P	RL���A�*

	conv_losswvn=�__#        )��P	��L���A�*

	conv_loss�XK=`�        )��P	��L���A�*

	conv_loss�HO=���        )��P	:�L���A�*

	conv_loss�)o=���        )��P	%L���A�*

	conv_loss�.�=�P��        )��P	dQL���A�*

	conv_loss��X=a-]        )��P	ąL���A�*

	conv_loss�Kc=d?�        )��P	��L���A�*

	conv_loss<$G=r���        )��P	�L���A�*

	conv_loss��C=z�ܥ        )��P	i*L���A�*

	conv_losss~=[�X"        )��P	�ZL���A�*

	conv_lossY�= t��        )��P	E�L���A�*

	conv_loss��=D1b�        )��P	b�L���A�*

	conv_loss���=ʕ�D        )��P	��L���A�*

	conv_loss�hn=P��        )��P	�%L���A�*

	conv_loss���=��P        )��P	GYL���A�*

	conv_loss-Rn=u��t        )��P	�L���A�*

	conv_lossq�c=yP��        )��P	��L���A�*

	conv_loss��S=10��        )��P	��L���A�*

	conv_loss�Hg=���i        )��P	n L���A�*

	conv_loss�F�=+aȃ        )��P	�QL���A�*

	conv_loss��=�S~        )��P	J�L���A�*

	conv_loss�`q=U�`�        )��P	�L���A�*

	conv_loss�N=Qk@=        )��P	�L���A�*

	conv_loss���=󐟆        )��P	� L���A�*

	conv_loss�F=�J	!        )��P	\N L���A�*

	conv_lossJ8�=-|Z        )��P	� L���A�*

	conv_loss�s=��:A        )��P	� L���A�*

	conv_loss�x=��2        )��P	�� L���A�*

	conv_lossJ�}=���N        )��P	�!L���A�*

	conv_loss��u=O�޸        )��P	�G!L���A�*

	conv_loss�ܓ=��        )��P	3{!L���A�*

	conv_loss���=*-�        )��P	�!L���A�*

	conv_loss;P�=�4?�        )��P	��!L���A�*

	conv_loss�=z���        )��P	8"L���A�*

	conv_lossm��<i�>        )��P	��#L���A�*

	conv_loss���=��YK        )��P	��#L���A�*

	conv_loss�=        )��P	$L���A�*

	conv_loss��F=�E�$        )��P	;$L���A�*

	conv_lossh�I=~���        )��P	�o$L���A�*

	conv_loss��=ޗW        )��P	Š$L���A�*

	conv_loss��s=u��$        )��P	�$L���A�*

	conv_loss�?\=|C�2        )��P	%L���A�*

	conv_loss�=��        )��P	;I%L���A�*

	conv_loss�*�=�,�        )��P	z}%L���A�*

	conv_loss�N=�T�        )��P	q�%L���A�*

	conv_loss�=���        )��P	��%L���A�*

	conv_loss�T�=�ꘄ        )��P	X&L���A�*

	conv_lossٽs=iO�        )��P	vE&L���A�*

	conv_loss��=%��^        )��P	�u&L���A�*

	conv_loss?�<=�,^�        )��P	��&L���A�*

	conv_lossS�=���        )��P		�&L���A�*

	conv_loss�lB=���        )��P	�'L���A�*

	conv_loss�/�=�7��        )��P	^F'L���A�*

	conv_loss��6=uX�        )��P	{y'L���A�*

	conv_loss���=n>1�        )��P	m�'L���A�*

	conv_loss��V=�.��        )��P	u�'L���A�*

	conv_loss�1=�!��        )��P	�(L���A�*

	conv_lossx�0=�.�v        )��P	�<(L���A�*

	conv_lossL?X=��ك        )��P	�o(L���A�*

	conv_loss1+M=j�f�        )��P	 �(L���A�*

	conv_loss��h=�u�C        )��P	�(L���A�*

	conv_loss�Y=4��         )��P	��(L���A�*

	conv_lossS=x=��        )��P	�.)L���A�*

	conv_lossu�=J���        )��P	�])L���A�*

	conv_loss_�r=o2+�        )��P	w�)L���A�*

	conv_loss/0L=W���        )��P	4�)L���A�*

	conv_loss%3�=��        )��P	0�)L���A�*

	conv_loss*'=R�2        )��P	�#*L���A�*

	conv_lossm
Z=���f        )��P	�S*L���A�*

	conv_loss�=�IrL        )��P	��*L���A�*

	conv_loss!�=Hx-        )��P	S�*L���A�*

	conv_loss9��=���J        )��P	|�*L���A�*

	conv_lossl��=e�sY        )��P	�+L���A�*

	conv_loss3�=!��n        )��P	�F+L���A�*

	conv_loss�%2=��        )��P	v+L���A�*

	conv_lossI3G=.'�        )��P	��+L���A�*

	conv_loss��K=��ԯ        )��P	G�+L���A�*

	conv_lossth/=����        )��P	�,L���A�*

	conv_loss��,=)L]
        )��P	�6,L���A�*

	conv_losseӉ=�3J�        )��P	�d,L���A�*

	conv_loss��G=QVڡ        )��P	�,L���A�*

	conv_loss���=<@Sj        )��P	v�,L���A�*

	conv_loss譔=ٕ��        )��P	��,L���A�*

	conv_loss0k='Ua�        )��P	 $-L���A�*

	conv_loss�(�=s|${        )��P	�c-L���A�*

	conv_loss��W=�c�        )��P	��-L���A�*

	conv_lossG�I=��7        )��P	4�-L���A�*

	conv_loss�P�=�3�        )��P	O�-L���A�*

	conv_loss3R�=�|�        )��P	�$.L���A�*

	conv_loss�r=��h�        )��P	"\.L���A�*

	conv_loss[lh=�\J
        )��P	D�.L���A�*

	conv_lossv6]=w�=.        )��P	�.L���A�*

	conv_loss�xO=��        )��P	<�.L���A�*

	conv_loss8Z=A[=Y        )��P	�/L���A�*

	conv_lossT�5=��:        )��P	{O/L���A�*

	conv_lossUB�=�/��        )��P	E/L���A�*

	conv_lossj�=lD�        )��P	M�/L���A�*

	conv_loss==�9�        )��P	�/L���A�*

	conv_loss��=O�W        )��P	K0L���A�*

	conv_losseE>=")        )��P	�E0L���A�*

	conv_loss�_7=q���        )��P	�t0L���A�*

	conv_loss$�F=6͟j        )��P	��0L���A�*

	conv_lossSy=w��        )��P	��0L���A�*

	conv_lossC�W=����        )��P	�1L���A�*

	conv_loss�_?=h��B        )��P	�D1L���A�*

	conv_loss-�=�V�        )��P	?u1L���A�*

	conv_loss��c=e�hw        )��P	��1L���A�*

	conv_loss�Q1=�N�w        )��P	f�1L���A�*

	conv_loss�4=e� C        )��P	�2L���A�*

	conv_loss,Bi=U�Z�        )��P	�12L���A�*

	conv_lossa�<=h�{�        )��P	=`2L���A�*

	conv_loss�x=���        )��P	�2L���A�*

	conv_lossrv=�3        )��P	�2L���A�*

	conv_loss��K=�_�        )��P	��2L���A�*

	conv_loss��n='�P�        )��P	*!3L���A�*

	conv_loss=�`=�k�        )��P		Q3L���A�*

	conv_loss� L=*;�        )��P	C�3L���A�*

	conv_loss/5�=<�V        )��P	�3L���A�*

	conv_lossn0\=#�x        )��P	��3L���A�*

	conv_loss��)=�"�        )��P	�4L���A�*

	conv_lossM�S=�W�        )��P	3F4L���A�*

	conv_lossJF�=ʈ�        )��P	b|4L���A�*

	conv_loss��J=���        )��P	2�4L���A�*

	conv_loss��==g��H        )��P	��4L���A�*

	conv_lossn�V=�˸�        )��P	�5L���A�*

	conv_loss��==�n-K        )��P	�<5L���A�*

	conv_loss��=H�x        )��P	�u5L���A�*

	conv_loss��r=|<�        )��P	��5L���A�*

	conv_lossrn9=d 4        )��P	��5L���A�*

	conv_loss�-�=K�%�        )��P	�6L���A�*

	conv_loss"9S=�ii�        )��P	�56L���A�*

	conv_loss�K�=��"s        )��P	�h6L���A�*

	conv_loss;U[=;Dg�        )��P	ӟ6L���A�*

	conv_loss�~J=�L        )��P	��6L���A�*

	conv_loss�|=�W�        )��P	%7L���A�*

	conv_loss��u=�Uq        )��P	E7L���A�*

	conv_losso�)=�	�        )��P	ނ7L���A�*

	conv_lossla)=^�t        )��P	�7L���A�*

	conv_loss��=o �        )��P	��7L���A�*

	conv_loss�7�=�$0        )��P	C8L���A�*

	conv_loss��M=�LR        )��P	*Q8L���A�*

	conv_loss�y_=�?�        )��P	C�8L���A�*

	conv_loss��a=���T        )��P	��8L���A�*

	conv_lossh�1=�l�.        )��P	_�8L���A�*

	conv_loss�O=�<��        )��P	K#9L���A�*

	conv_loss>P=!\�?        )��P	�S9L���A�*

	conv_lossF�P=]N]�        )��P	��9L���A�*

	conv_loss�R�="J�}        )��P	�9L���A�*

	conv_loss��R=�[l#        )��P	V�9L���A�*

	conv_lossRn=>R5P        )��P	6:L���A�*

	conv_loss�8=�)6b        )��P	�J:L���A�*

	conv_lossK�M=�n�}        )��P	n�:L���A�*

	conv_loss4zC=M�\        )��P	>�:L���A�*

	conv_lossY�h=��        )��P	@�:L���A�*

	conv_loss#fV=5�[�        )��P	, ;L���A�*

	conv_loss��n=�<        )��P	�Q;L���A�*

	conv_loss��=��f�        )��P	@�;L���A�*

	conv_loss��H=��0M        )��P	г;L���A�*

	conv_lossyr=�(��        )��P	]�;L���A�*

	conv_loss�G=C�[�        )��P	�<L���A�*

	conv_loss��}=Q���        )��P	RD<L���A�*

	conv_lossf�\=d�,s        )��P	�s<L���A�*

	conv_loss��l=sn_        )��P	��<L���A�*

	conv_loss��0=��e�        )��P	��<L���A�*

	conv_loss�Rh=�A�        )��P	L=L���A�*

	conv_loss�2n=15&�        )��P	^@=L���A�*

	conv_loss��h=L�        )��P	�o=L���A�*

	conv_loss�F.=9ƴ9        )��P	I�=L���A�*

	conv_lossm=X��        )��P	��=L���A�*

	conv_lossm�3=h��2        )��P	3>L���A�*

	conv_loss
�=B�?        )��P	�5>L���A�*

	conv_loss9=e���        )��P	Hd>L���A�*

	conv_lossg�<�~0�        )��P	��>L���A�*

	conv_lossۙ==�C �        )��P	~�>L���A�*

	conv_loss�}[=�u        )��P	C�>L���A�*

	conv_loss��=��q-        )��P	v-?L���A�*

	conv_loss�=RT��        )��P	�\?L���A�*

	conv_lossW�V="�~        )��P	�?L���A�*

	conv_loss!��=�Hg�        )��P	��?L���A�*

	conv_loss�'=:1��        )��P	#�?L���A�*

	conv_loss�In=��6o        )��P	�!@L���A�*

	conv_loss�s~=�9C        )��P	#S@L���A�*

	conv_loss}
}=�0�J        )��P	M�@L���A�*

	conv_loss�1=���        )��P	4�@L���A�*

	conv_loss�<^=DC�        )��P	�AL���A�*

	conv_losss΀=&��#        )��P	�;AL���A�*

	conv_loss �h=�"��        )��P	�oAL���A�*

	conv_loss,��=�Y.[        )��P	*�AL���A�*

	conv_lossDtU=�ǰ        )��P	�AL���A�*

	conv_loss���<���"        )��P	RBL���A�*

	conv_loss��X=�B@�        )��P	KGBL���A�*

	conv_loss%�=g���        )��P	̂BL���A�*

	conv_loss�u=>7��        )��P	��BL���A�*

	conv_loss�5<=�
�d        )��P	z�BL���A�*

	conv_loss�'~=Q���        )��P	J0CL���A�*

	conv_lossʋr=#΅        )��P	hcCL���A�*

	conv_loss�q=q�S        )��P	b�CL���A�*

	conv_loss��t=��_!        )��P	D�CL���A�*

	conv_loss�Y=Ĩ*        )��P	JDL���A�*

	conv_loss�x=2���        )��P	4DL���A�*

	conv_loss7#q=���        )��P	miDL���A�*

	conv_loss��Z=���        )��P		�DL���A�*

	conv_loss4�=���g        )��P	��DL���A�*

	conv_loss�"=�q        )��P	@EL���A�*

	conv_loss�?\=O[�        )��P	�PEL���A�*

	conv_loss��=�_s�        )��P	R�EL���A�*

	conv_loss�.c=��        )��P	�EL���A�*

	conv_lossY�w=#��w        )��P	C�EL���A�*

	conv_lossb"�=}�s�        )��P	�+FL���A�*

	conv_loss�#;=���E        )��P	!^FL���A�*

	conv_loss���=�~�        )��P	Q�FL���A�*

	conv_loss��K=$<�        )��P	��FL���A�*

	conv_lossBj=��8�        )��P	��FL���A�*

	conv_lossEd�=0z��        )��P	3GL���A�*

	conv_lossE�=��Q�        )��P	.fGL���A�*

	conv_loss��3=B��A        )��P	��GL���A�*

	conv_loss%�#=vU��        )��P	��GL���A�*

	conv_loss%��=�        )��P	{�GL���A�*

	conv_loss-e�=�&�k        )��P	S0HL���A�*

	conv_loss<�&=[�ͧ        )��P	`HL���A�*

	conv_loss�#_=�i��        )��P	��HL���A�*

	conv_losse�8=�2=P        )��P	��HL���A�*

	conv_lossq�=Z�'        )��P	5IL���A�*

	conv_loss�[=�*(c        )��P	�4IL���A�*

	conv_lossq��=�9[        )��P	�dIL���A�*

	conv_loss=�T=u��k        )��P	��IL���A�*

	conv_loss��=�a�        )��P	��IL���A�*

	conv_lossq�K=�"�        )��P	�IL���A�*

	conv_loss�r=l"��        )��P	n0JL���A�*

	conv_loss�==��$z        )��P	 bJL���A�*

	conv_loss��R=�!A`        )��P	�JL���A�*

	conv_loss��9=qR.        )��P	��JL���A�*

	conv_loss�L=�B�?        )��P	�KL���A�*

	conv_losse�]=�8�        )��P	��OL���A�*

	conv_loss0�2=�u�        )��P	�bQL���A�*

	conv_loss�/X=���        )��P	��QL���A�*

	conv_loss��8=��        )��P	�QL���A�*

	conv_loss��g=�`��        )��P	j�QL���A�*

	conv_loss��.=i��        )��P	�&RL���A�*

	conv_lossñG=�[�`        )��P	�^RL���A�*

	conv_loss��?=��        )��P	ʌRL���A�*

	conv_loss|T=0�[        )��P	�RL���A�*

	conv_loss�Z	=��X        )��P	V�RL���A�*

	conv_loss�yl=�	�%        )��P	�*SL���A�*

	conv_lossM�=BE��        )��P	�`SL���A�*

	conv_loss�3\=8IΠ        )��P	�SL���A�*

	conv_lossV�=��        )��P	p�SL���A�*

	conv_loss���=��%        )��P	��SL���A�*

	conv_loss��=����        )��P	f&TL���A�*

	conv_loss�fy=+G	`        )��P	+dTL���A�*

	conv_loss�8=�ub�        )��P	.�TL���A�*

	conv_lossB�=��        )��P	�TL���A�*

	conv_lossWHL=i��a        )��P	��TL���A�*

	conv_lossrh=�<�        )��P	�)UL���A�*

	conv_lossõS=�d�        )��P	�cUL���A�*

	conv_loss��P=�`��        )��P	��UL���A�*

	conv_loss��J=�wP,        )��P	��UL���A�*

	conv_lossZ�=/�n�        )��P	��UL���A�*

	conv_loss��7=��8)        )��P	#VL���A�*

	conv_loss��p=��M�        )��P	t]VL���A�*

	conv_lossög=`2g*        )��P	ߌVL���A�*

	conv_loss"�O=���e        )��P	H�VL���A�*

	conv_loss�LL=�#�        )��P	��VL���A�*

	conv_lossY {=���        )��P	wWL���A�*

	conv_loss�h=���        )��P	�XWL���A�*

	conv_loss% ^=k3E        )��P	@�WL���A�*

	conv_loss�=�Iq�        )��P	l�WL���A�*

	conv_loss.�=@��        )��P	�WL���A�*

	conv_loss�Hh=R1'�        )��P	�XL���A�*

	conv_loss�5F= @!        )��P	JXL���A�*

	conv_lossp�#=P�        )��P	wXL���A�*

	conv_loss��O=��\Z        )��P	�XL���A�*

	conv_loss��D=�q4u        )��P	��XL���A�*

	conv_loss6z=���        )��P	]YL���A�*

	conv_loss�)�=-�c�        )��P	�2YL���A�*

	conv_loss�)V=u���        )��P	qeYL���A�*

	conv_loss�f=dIw        )��P	��YL���A�*

	conv_loss�V=/yz;        )��P	��YL���A�*

	conv_lossgj2=L$        )��P	-�YL���A�*

	conv_loss��"=N��        )��P	w&ZL���A�*

	conv_loss|Q�=�� �        )��P	�WZL���A�*

	conv_lossFT=����        )��P	��ZL���A�*

	conv_loss���=O\�]        )��P	��ZL���A�*

	conv_loss-5e=���2        )��P	
�ZL���A�*

	conv_lossH�=��        )��P	�0[L���A�*

	conv_loss�dp=_�:        )��P	q_[L���A�*

	conv_loss�n2=_֤�        )��P	�[L���A�*

	conv_lossЫ$=����        )��P	l�[L���A�*

	conv_loss�3O=�;�6        )��P	E�[L���A�*

	conv_lossX$=��w        )��P	|.\L���A�*

	conv_loss2=��Ȇ        )��P	�^\L���A�*

	conv_loss9�B=��p�        )��P	ו\L���A�*

	conv_loss�Y=X��        )��P	u�\L���A�*

	conv_lossb{@=�U�        )��P	�]L���A�*

	conv_loss��==�b"        )��P	�5]L���A�*

	conv_lossD�r=h>�G        )��P	�i]L���A�*

	conv_loss)H=�v�)        )��P	�]L���A�*

	conv_loss �e=�V�@        )��P	��]L���A�*

	conv_loss{�1=��K*        )��P	�^L���A�*

	conv_loss��3=7��        )��P	�<^L���A�*

	conv_lossBi=�iM�        )��P	Dp^L���A�*

	conv_loss�84=K��i        )��P	�^L���A�*

	conv_loss =㐓        )��P	Y�^L���A�*

	conv_loss�1]=����        )��P	�_L���A�*

	conv_loss��`=��}�        )��P	/?_L���A�*

	conv_loss���=I<�        )��P	9n_L���A�*

	conv_loss}= �5�        )��P	��_L���A�*

	conv_lossQD=�*}�        )��P	��_L���A�*

	conv_loss���=�&        )��P	�	`L���A�*

	conv_loss2t9=�M,}        )��P	�:`L���A�*

	conv_loss�T@=���h        )��P	Ui`L���A�*

	conv_loss��?=z�.�        )��P	h�`L���A�*

	conv_loss��J=�V�        )��P	��`L���A�*

	conv_loss2Hm=���        )��P	]aL���A�*

	conv_loss��=�9��        )��P	/CaL���A�*

	conv_lossޚ.=�ˇ�        )��P	�raL���A�*

	conv_loss͈�<�V)        )��P	�aL���A�*

	conv_loss�s=�颗        )��P	Z�aL���A�*

	conv_loss"O=t1_        )��P	IbL���A�*

	conv_lossӛ=�E��        )��P	26bL���A�*

	conv_loss�;)=߀e        )��P	�fbL���A�*

	conv_loss�b=�{:�        )��P	��bL���A�*

	conv_loss�H!=���        )��P	]�bL���A�*

	conv_lossx=B�iQ        )��P	�cL���A�*

	conv_loss��=01|b        )��P	:cL���A�*

	conv_loss&�?=j�mu        )��P	�icL���A�*

	conv_loss(=$��        )��P	��cL���A�*

	conv_loss��m=�rAJ        )��P	��cL���A�*

	conv_loss+^i=LZe�        )��P	J�cL���A�*

	conv_loss�3-=k���        )��P	^/dL���A�*

	conv_loss�=���        )��P	�_dL���A�*

	conv_loss��H=�;�        )��P	ٍdL���A�*

	conv_loss��=���        )��P	��dL���A�*

	conv_loss�(=e:�        )��P	��dL���A�*

	conv_loss]�,=��q        )��P	�FeL���A�*

	conv_loss"�X= %         )��P	�veL���A�*

	conv_loss"�a=#eV        )��P	W�eL���A�*

	conv_loss��6=�7�        )��P	K�eL���A�*

	conv_lossZ�i=`�	-        )��P	�fL���A�*

	conv_loss��e=�p�        )��P	xHfL���A�*

	conv_loss��=����        )��P	�xfL���A�*

	conv_losst��=aʲ        )��P	��fL���A�*

	conv_loss�TX=G|�A        )��P	�fL���A�*

	conv_loss(E=�qu        )��P	�gL���A�*

	conv_loss��_=��         )��P	#KgL���A�*

	conv_loss�[�=�'��        )��P	_zgL���A�*

	conv_loss�|b=7�R�        )��P	L�gL���A�*

	conv_lossZ3:=�Vk�        )��P	N�gL���A�*

	conv_loss�u�='�u        )��P	�hL���A�*

	conv_lossNpn=ǐ=�        )��P	�VhL���A�*

	conv_loss�S=�jR        )��P	
�hL���A�*

	conv_loss�*=�sF        )��P	g�hL���A�*

	conv_loss��g=�צ?        )��P	X�hL���A�*

	conv_lossSY=J��        )��P	O!iL���A�*

	conv_loss�=�_��        )��P	QiL���A�*

	conv_loss/�G=�UG�        )��P	@�iL���A�*

	conv_loss��?=��n        )��P	@�iL���A�*

	conv_loss���=�h@&        )��P	<�iL���A�*

	conv_loss�&=���L        )��P	ejL���A�*

	conv_loss��m=� �l        )��P	|HjL���A�*

	conv_lossj�==H��Q        )��P	oyjL���A�*

	conv_lossm-D=��Q�        )��P	�jL���A�*

	conv_loss�^="���        )��P	��jL���A�*

	conv_loss��+=��$�        )��P	XkL���A�*

	conv_loss�k=�П�        )��P	�=kL���A�*

	conv_loss"�i=a��        )��P	okL���A�*

	conv_loss{Y`=E@o�        )��P	��kL���A�*

	conv_loss
�L=�ņt        )��P	1�kL���A�*

	conv_loss��$=�E��        )��P	 lL���A�*

	conv_loss��@=�:2        )��P	�ClL���A�*

	conv_loss|$?=["�N        )��P	�slL���A�*

	conv_loss��B=V�        )��P	��lL���A�*

	conv_loss��W=��x�        )��P	��lL���A�*

	conv_loss��=�a        )��P	�mL���A�*

	conv_loss(�=��E        )��P	�9mL���A�*

	conv_lossK�E=L��        )��P	wkmL���A�*

	conv_loss�Oa=fs�        )��P	>�mL���A�*

	conv_loss��=��Q�        )��P	O�mL���A�*

	conv_loss��v=����        )��P	�nL���A�*

	conv_loss&�=`�        )��P	�8nL���A�*

	conv_loss|2B=�{�v        )��P	8inL���A�*

	conv_loss.=��>        )��P	��nL���A�*

	conv_loss��w=vK6�        )��P	��nL���A�*

	conv_loss�D=�Ű�        )��P	yoL���A�*

	conv_loss��=/y��        )��P	�FoL���A�*

	conv_loss�FS=x`!a        )��P	xoL���A�*

	conv_loss~-+=(�        )��P	F�oL���A�*

	conv_losst�}=���<        )��P	j�oL���A�*

	conv_loss�l7=��ŀ        )��P	�pL���A�*

	conv_lossv�=����        )��P	�PpL���A�*

	conv_loss1�%=D�ku        )��P	��pL���A�*

	conv_loss�|=;[b�        )��P	�pL���A�*

	conv_loss�=���k        )��P	c�pL���A�*

	conv_lossƁ==Ʊ��        )��P	.#qL���A�*

	conv_loss�g=d��        )��P	_VqL���A�*

	conv_loss�=D�"]        )��P	�qL���A�*

	conv_loss�^?=?�,r        )��P	��qL���A�*

	conv_lossg8=*        )��P	>�qL���A�*

	conv_lossc\(=Co�        )��P	R-rL���A�*

	conv_loss�x=���        )��P	�crL���A�*

	conv_loss\�#=P.-�        )��P	�rL���A�*

	conv_loss��;=�R        )��P	�rL���A�*

	conv_lossk�-=���        )��P	�sL���A�*

	conv_loss;�M=GN��        )��P	�5sL���A�*

	conv_loss�U=�(w�        )��P	�gsL���A�*

	conv_loss���=�G�2        )��P	i�sL���A�*

	conv_loss��7=2<�        )��P	t�sL���A�*

	conv_loss��v=� ��        )��P	� tL���A�*

	conv_loss^=?�"�        )��P	P3tL���A�*

	conv_loss��"='$^l        )��P	�ftL���A�*

	conv_loss��)=I>�        )��P	\�tL���A�*

	conv_loss�9=#�_        )��P	q�tL���A�*

	conv_loss�=:��        )��P	n�tL���A�*

	conv_lossE.=-��)        )��P	{.uL���A�*

	conv_lossF7=[���        )��P	�`uL���A�*

	conv_loss�~x=!��        )��P	֓uL���A�*

	conv_loss-
&=f�,        )��P	��uL���A�*

	conv_lossI�=4	+        )��P	W�uL���A�*

	conv_loss*:P=�%=S        )��P	�+vL���A�*

	conv_lossÆ/=`�8�        )��P	#]vL���A�*

	conv_loss{�f=t_�t        )��P	ɍvL���A�*

	conv_loss�n=c(�m        )��P	�vL���A�*

	conv_loss^n�=��?
        )��P	�vL���A�*

	conv_loss�r<=e8g        )��P	�#wL���A�*

	conv_loss"mf=��9�        )��P	�TwL���A�*

	conv_loss]�=+�        )��P	��wL���A�*

	conv_loss��!=_.|�        )��P	޶wL���A�*

	conv_loss��'=���        )��P	��wL���A�*

	conv_loss��/=�q�        )��P	�xL���A�*

	conv_loss� E=���l        )��P	�OxL���A�*

	conv_loss�]=��1        )��P	��xL���A�*

	conv_loss8pJ=:$��        )��P	p�xL���A�*

	conv_loss9�=��        )��P	a�xL���A�*

	conv_losspk�=i ��        )��P	(yL���A�*

	conv_loss��1=a��        )��P	ǲzL���A�*

	conv_lossR�=�h1�        )��P	9�zL���A�*

	conv_loss^c=0'�        )��P	{L���A�*

	conv_loss�+�=jdYr        )��P	�L{L���A�*

	conv_loss366=�AC�        )��P	�~{L���A�*

	conv_loss�
=�m�        )��P	��{L���A�*

	conv_lossE�P=��QC        )��P	&�{L���A�*

	conv_loss<��=����        )��P	�|L���A�*

	conv_losss*=�j8        )��P	�J|L���A�*

	conv_loss��=4���        )��P	.�|L���A�*

	conv_loss܊>=���        )��P	��|L���A�*

	conv_losse�=���        )��P	��|L���A�*

	conv_loss0��=YE��        )��P	/-}L���A�*

	conv_loss��<���        )��P	�_}L���A�*

	conv_loss	.U=e��        )��P	Ֆ}L���A�*

	conv_lossӃo=��nY        )��P	�}L���A�*

	conv_loss�B=f�        )��P	j�}L���A�*

	conv_loss{m�<��        )��P	�/~L���A�*

	conv_lossm�E=4!C�        )��P	�c~L���A�*

	conv_loss��3=.���        )��P	S�~L���A�*

	conv_losse�s=�kr        )��P	�~L���A�*

	conv_loss�|�=�t�        )��P	� L���A�*

	conv_loss��=,,&8        )��P	J5L���A�*

	conv_loss�7&=@�D        )��P	�fL���A�*

	conv_loss�P=���R        )��P	��L���A�*

	conv_loss�$r=�J�        )��P	��L���A�*

	conv_loss 4=
,��        )��P	��L���A�*

	conv_loss��<��s        )��P	\8�L���A�*

	conv_loss��1=D㦮        )��P	�j�L���A�*

	conv_loss/R=�:�9        )��P	9��L���A�*

	conv_loss-sd=?���        )��P	@ҀL���A�*

	conv_lossx1N=W��H        )��P	N�L���A�*

	conv_lossv�i=h        )��P	6�L���A�*

	conv_lossp�h=�b�        )��P	
g�L���A�*

	conv_loss��E=��ϙ        )��P	˙�L���A�*

	conv_loss���=� �        )��P	ÉL���A�*

	conv_loss��s=�\ڴ        )��P	#��L���A�*

	conv_loss�lZ="��B        )��P	�.�L���A�*

	conv_loss��2=@JF        )��P	�_�L���A�*

	conv_loss�J^=9�ˌ        )��P	��L���A�*

	conv_loss+Qd=:��        )��P	�тL���A�*

	conv_losso��=A�I�        )��P	f�L���A�*

	conv_loss�()=烥U        )��P	�7�L���A�*

	conv_loss�b3=�X1        )��P	9k�L���A�*

	conv_lossr�1=b�Q        )��P	J��L���A�*

	conv_loss���=�?q        )��P	�ރL���A�*

	conv_loss�lV=��        )��P	p�L���A�*

	conv_loss���<Um�        )��P	VN�L���A�*

	conv_loss�ed=���        )��P	+��L���A�*

	conv_loss�@=��e�        )��P	��L���A�*

	conv_lossL8<=bSnx        )��P	5��L���A�*

	conv_loss�^=C��        )��P	�4�L���A�*

	conv_loss��%=�X��        )��P	Xh�L���A�*

	conv_loss49%=�wX�        )��P	ŝ�L���A�*

	conv_loss$w"=�Լ�        )��P	IمL���A�*

	conv_lossJDp=���        )��P	��L���A�*

	conv_loss�'=N���        )��P	B�L���A�*

	conv_loss�
.= �Ѓ        )��P	�|�L���A�*

	conv_loss��G=�ạ        )��P	н�L���A�*

	conv_lossh�=�ǥ�        )��P	���L���A�*

	conv_loss�=�rF        )��P	�+�L���A�*

	conv_loss}�d=����        )��P	�^�L���A�*

	conv_loss��s=���        )��P	ז�L���A�*

	conv_lossB=��ߋ        )��P	xՇL���A�*

	conv_lossɶ2=&��        )��P	+�L���A�*

	conv_loss�=ŗE        )��P	�D�L���A�*

	conv_loss0�=gLѰ        )��P	0v�L���A�*

	conv_lossj�=�"        )��P	���L���A�*

	conv_loss��i=�_Z\        )��P	���L���A�*

	conv_lossĄP=��        )��P	��L���A�*

	conv_lossb��=��fJ        )��P	~V�L���A�*

	conv_lossò1=�AD        )��P	ڋ�L���A�*

	conv_loss��j=���        )��P	���L���A�*

	conv_loss�7�=��(�        )��P	���L���A�*

	conv_loss�V=:��\        )��P	�)�L���A�*

	conv_loss�h=2�x        )��P	�[�L���A�*

	conv_loss� =�"�        )��P	D��L���A�*

	conv_loss�6=�Oh/        )��P	���L���A�*

	conv_lossY�R=��        )��P	]��L���A�*

	conv_lossßN=F=��        )��P	�+�L���A�*

	conv_loss\HR=�,0�        )��P	x]�L���A�*

	conv_lossLi�=��.�        )��P	W��L���A�*

	conv_lossm �=����        )��P	aыL���A�*

	conv_loss�<�?��        )��P	C�L���A�*

	conv_loss�H=���4        )��P	�:�L���A�*

	conv_loss�0=1)R        )��P	[m�L���A�*

	conv_loss)��<�w�        )��P	��L���A�*

	conv_loss��L=��l        )��P	�ҌL���A�*

	conv_loss-�a=1�ߏ        )��P	�	�L���A�*

	conv_loss9w�=�3�        )��P	�;�L���A�*

	conv_loss�4G=s ��        )��P	Sq�L���A�*

	conv_loss��=��T        )��P	��L���A�*

	conv_loss|�`=9���        )��P	zԍL���A�*

	conv_lossa=)=u���        )��P	�
�L���A�*

	conv_losso6=�#�        )��P	v;�L���A�*

	conv_loss�8=d�u�        )��P	�n�L���A�*

	conv_loss��g=���        )��P	U��L���A�*

	conv_lossQ2^=CK�i        )��P	!�L���A�*

	conv_lossG<;=u�DT        )��P	/�L���A�*

	conv_lossIFs=�v��        )��P	.L�L���A�*

	conv_loss��e=)�m�        )��P	y��L���A�*

	conv_loss��0=�85D        )��P	�ÏL���A�*

	conv_loss[| =�=�        )��P	���L���A�*

	conv_loss��B=��|        )��P	o,�L���A�*

	conv_loss��A='�w        )��P	Nd�L���A�*

	conv_loss��#=���        )��P	!��L���A�*

	conv_lossI�(=� G-        )��P	�ӐL���A�*

	conv_loss�t=2�b        )��P	�L���A�*

	conv_lossC]#=���        )��P	�D�L���A�*

	conv_loss�	!=d��        )��P	'}�L���A�*

	conv_loss��G=m�X�        )��P	0��L���A�*

	conv_loss�Ds=���        )��P	��L���A�*

	conv_loss�G�=�1�        )��P	��L���A�*

	conv_lossޢ�=�7�        )��P	Q�L���A�*

	conv_loss�=����        )��P	@��L���A�*

	conv_lossc�/=y�        )��P	���L���A�*

	conv_loss؄V=m6b        )��P	_�L���A�*

	conv_loss���=)΅        )��P	�-�L���A�*

	conv_loss'P2=2�2�        )��P	&c�L���A�*

	conv_loss��W=���        )��P	���L���A�*

	conv_lossL=�v�O        )��P	�ɓL���A�*

	conv_loss�B\=OFQ%        )��P	���L���A�*

	conv_loss�oe=�N��        )��P	y9�L���A�*

	conv_loss?bv=�N��        )��P	o�L���A�*

	conv_loss"�O=����        )��P	'��L���A�*

	conv_loss2�=$��v        )��P	�הL���A�*

	conv_loss�;=�V        )��P	��L���A�*

	conv_loss;i�=��0        )��P	�F�L���A�*

	conv_loss��7=��kD        )��P	x�L���A�*

	conv_loss�A1=�x�        )��P	ƪ�L���A�*

	conv_loss,�j=q�ה        )��P	�ޕL���A�*

	conv_loss/�4=�1ڙ        )��P	��L���A�*

	conv_losse�=�u�        )��P	�C�L���A�*

	conv_loss�.==Ha        )��P	�s�L���A�*

	conv_loss�ځ=����        )��P	���L���A�*

	conv_loss�5�<X~        )��P	זL���A�*

	conv_lossm(==��xm        )��P	&�L���A�*

	conv_loss��N=ȊP%        )��P	J�L���A�*

	conv_loss�)=%��        )��P	�~�L���A�*

	conv_lossp�Q=�f�6        )��P	o��L���A�*

	conv_loss��Z=�<�        )��P	]�L���A�*

	conv_losst�i=�Q/a        )��P	��L���A�*

	conv_loss2#=s�        )��P	[J�L���A�*

	conv_loss�=\s,*        )��P	y{�L���A�*

	conv_loss97=���        )��P	ᬘL���A�*

	conv_loss��<=g��        )��P	jޘL���A�*

	conv_loss�;7=��
        )��P	��L���A�*

	conv_loss#6i=I�F        )��P	�Q�L���A�*

	conv_lossO~+=A�        )��P	ˈ�L���A�*

	conv_loss#�v=6�.g        )��P	7ÙL���A�*

	conv_loss�|L=����        )��P	�
�L���A�*

	conv_loss2�=38�c        )��P	&@�L���A�*

	conv_loss�lX=,{>9        )��P	e|�L���A�*

	conv_lossU�=�F+�        )��P	���L���A�*

	conv_loss�\\=L�L        )��P	���L���A�*

	conv_lossY+:=��/N        )��P	�+�L���A�*

	conv_loss=�y=�k�        )��P	Li�L���A�*

	conv_loss�|=R�#        )��P	���L���A�*

	conv_loss��=���        )��P	@ޛL���A�*

	conv_loss��6=�+        )��P	��L���A�*

	conv_loss�W=
�<!        )��P	�D�L���A�*

	conv_loss/�T==�Q*        )��P	X|�L���A�*

	conv_loss�&8=��M        )��P	X��L���A�*

	conv_loss�7%=���n        )��P	�L���A�*

	conv_loss*	-=�̋        )��P	�"�L���A�*

	conv_loss��=.z�        )��P	�\�L���A�*

	conv_lossy�j=��u=        )��P	嚝L���A�*

	conv_loss��<�q!        )��P	uϝL���A�*

	conv_loss�Q�=�        )��P	��L���A�*

	conv_loss��L=S2,6        )��P	`:�L���A�*

	conv_loss�=�Q�        )��P	�|�L���A�*

	conv_loss�3{=sJ�        )��P	<��L���A�*

	conv_loss'ͅ=��
�        )��P	��L���A�*

	conv_lossh�E=ݫ�=        )��P	��L���A�*

	conv_loss=6!=m���        )��P	�I�L���A�*

	conv_loss�!0=T[�        )��P	�{�L���A�*

	conv_loss�>q=:6[        )��P	9��L���A�*

	conv_loss��M=��q�        )��P	��L���A�*

	conv_loss$�=�%M�        )��P	�)�L���A�*

	conv_loss!�E=���        )��P	]�L���A�*

	conv_lossr>A=>�9�        )��P	��L���A�*

	conv_lossb�5=x��)        )��P	}͠L���A�*

	conv_loss5�u=+N�        )��P	I�L���A�*

	conv_loss3�=�V?        )��P	Y7�L���A�*

	conv_lossLw=�'�        )��P	�i�L���A�*

	conv_loss��E=�G�         )��P	`��L���A�*

	conv_loss��T=�)�p        )��P	<ҡL���A�*

	conv_loss��=�b&        )��P	c�L���A�*

	conv_lossVA�=d⨏        )��P	g@�L���A�*

	conv_loss��'=Q�I        )��P	�{�L���A�*

	conv_loss:�R=X�        )��P	챢L���A�*

	conv_loss�0=���t        )��P	��L���A�*

	conv_loss[3=����        )��P	��L���A�*

	conv_loss�g=&n;        )��P	H�L���A�*

	conv_loss�D=1�*.        )��P	0{�L���A�*

	conv_loss�u=(�2I        )��P	\��L���A�*

	conv_loss�4_=��E�        )��P	K�L���A�*

	conv_loss_�,=��g�        )��P	�L���A�*

	conv_loss��==��7�        )��P	S�L���A�*

	conv_loss��<=�i]�        )��P	���L���A�*

	conv_loss�=ޜ��        )��P	�9�L���A�*

	conv_loss�^�<��"�        )��P	*m�L���A�*

	conv_lossZu{=j��S        )��P	��L���A�*

	conv_loss)B=/�0        )��P	�٦L���A�*

	conv_loss�=ҳ��        )��P	�
�L���A�*

	conv_lossX�E=���(        )��P	�>�L���A�*

	conv_lossw�+=�w��        )��P	�w�L���A�*

	conv_loss�S=�d�        )��P	6��L���A�*

	conv_loss�*�=/c]S        )��P	�ߧL���A�*

	conv_lossB�=����        )��P	��L���A�*

	conv_lossgF=����        )��P	E�L���A�*

	conv_loss�^=33�        )��P	�w�L���A�*

	conv_loss�K=4Z�        )��P	(��L���A�*

	conv_loss�I=�|eK        )��P	9�L���A�*

	conv_loss��0=���2        )��P	��L���A�*

	conv_loss��*=e��        )��P	�D�L���A�*

	conv_lossK=���        )��P	Ax�L���A�*

	conv_loss��X=wu#�        )��P	e��L���A�*

	conv_loss���=[E�        )��P	��L���A�*

	conv_loss��%=Ȃa�        )��P	
'�L���A�*

	conv_loss���=ez�        )��P	uY�L���A�*

	conv_loss5.8=@�H        )��P	���L���A�*

	conv_loss!_=?��        )��P	\��L���A�*

	conv_loss�6=��/        )��P	���L���A�*

	conv_loss��@=��}�        )��P	�%�L���A�*

	conv_loss�d=�{ë        )��P	.T�L���A�*

	conv_lossK�C=����        )��P	���L���A�*

	conv_loss\@=V��O        )��P	;��L���A�*

	conv_loss�5=z@�        )��P	K��L���A�*

	conv_loss��0=B	��        )��P	%�L���A�*

	conv_loss
�=�u        )��P	�o�L���A�*

	conv_lossJ��<l@�        )��P	��L���A�*

	conv_loss'=6�M�        )��P	�ѬL���A�*

	conv_loss���<�̅@        )��P	��L���A�*

	conv_loss��=:�        )��P	r5�L���A�*

	conv_loss<�<�^�        )��P	�i�L���A�*

	conv_loss1)y=/<V        )��P	���L���A�*

	conv_loss�=g�        )��P	�ǭL���A�*

	conv_loss��@=kÕ�        )��P	G��L���A�*

	conv_lossH(=�N�        )��P	K*�L���A�*

	conv_loss�=���z        )��P	+b�L���A�*

	conv_loss�xA=��(Q        )��P	ؕ�L���A�*

	conv_lossd�c=�_��        )��P	�ϮL���A�*

	conv_loss��5=Ul��        )��P	��L���A�*

	conv_loss?�J=�m@        )��P	E>�L���A�*

	conv_losss�7=Lȍ�        )��P	O~�L���A�*

	conv_loss9Z�=�%?Q        )��P	һ�L���A�*

	conv_loss��G=\��        )��P	#��L���A�*

	conv_loss~G2=,��        )��P	�6�L���A�*

	conv_loss��=�d�B        )��P	�v�L���A�*

	conv_lossIV�=�:�        )��P	ϼ�L���A�*

	conv_loss%�6=���l        )��P	���L���A�*

	conv_loss�O?=X��        )��P	$�L���A�*

	conv_loss��=V:�        )��P	�O�L���A�*

	conv_loss�V=�!ɛ        )��P	��L���A�*

	conv_loss3�?=�,q�        )��P	�˱L���A�*

	conv_loss�[=����        )��P	\�L���A�*

	conv_loss�=?r(l        )��P	�K�L���A�*

	conv_lossw�}=ޙ�5        )��P	���L���A�*

	conv_loss}�'=         )��P	/��L���A�*

	conv_loss�o=2��        )��P	���L���A�*

	conv_loss��K=l��U        )��P	�%�L���A�*

	conv_loss^3p=���        )��P	�U�L���A�*

	conv_losse2=ی�a        )��P	]��L���A�*

	conv_loss�7=w� p        )��P	'��L���A�*

	conv_lossAY=���^        )��P	��L���A�*

	conv_loss�K4=�3        )��P	j5�L���A�*

	conv_loss^�l=�@�c        )��P	)o�L���A�*

	conv_lossz�<��|        )��P	���L���A�*

	conv_loss5�K=��O�        )��P	�״L���A�*

	conv_lossU�V=R>
�        )��P	��L���A�*

	conv_loss��-=�,	        )��P	�=�L���A�*

	conv_lossr�c=+�        )��P	%o�L���A�*

	conv_loss�(f=�1��        )��P	��L���A�*

	conv_lossx�=:���        )��P	�ѵL���A�*

	conv_loss�wR=
W6        )��P		�L���A�*

	conv_loss�)o=�z        )��P	�>�L���A�*

	conv_loss-;=����        )��P	�y�L���A�*

	conv_loss�-=_<�{        )��P	Z��L���A�*

	conv_loss �=<?�z        )��P	��L���A�*

	conv_loss���<FK�        )��P	�(�L���A�*

	conv_loss��=�Y��        )��P	�Y�L���A�*

	conv_loss�-=�M��        )��P	͈�L���A�*

	conv_lossEI�=J���        )��P	�ϷL���A�*

	conv_loss�O=�p*	        )��P	���L���A�*

	conv_loss0�N=���        )��P	�7�L���A�*

	conv_loss��=�*�        )��P	�h�L���A�*

	conv_lossN�E=����        )��P	���L���A�*

	conv_loss/A=jQ        )��P	<ȸL���A�*

	conv_loss\A=��*H        )��P	g �L���A�*

	conv_loss� T=c���        )��P	Z2�L���A�*

	conv_loss�{K=��}H        )��P	�`�L���A�*

	conv_loss�5�= Ē        )��P	[��L���A�*

	conv_loss��4=�[b        )��P	-ƹL���A�*

	conv_lossh0=d�         )��P	&��L���A�*

	conv_lossi�4=)�A�        )��P	�"�L���A�*

	conv_loss��=3��        )��P	�Q�L���A�*

	conv_loss��=�16�        )��P	���L���A�*

	conv_lossn0=M=ف        )��P	t��L���A�*

	conv_lossgN=`Q�        )��P	#��L���A�*

	conv_loss9s=F�0�        )��P	n?�L���A�*

	conv_lossF	`=�c        )��P	}q�L���A�*

	conv_loss"�!=���        )��P	J��L���A�*

	conv_loss'�<
G�        )��P	�ֿL���A�*

	conv_loss_v�=,�        )��P	�	�L���A�*

	conv_lossL�<ݩ�M        )��P	�D�L���A�*

	conv_loss�x= ~|        )��P	�y�L���A�*

	conv_lossQ;�<Y���        )��P	���L���A�*

	conv_loss�Q3=<e��        )��P	A��L���A�*

	conv_loss(N=+[)[        )��P	L�L���A�*

	conv_lossg�4=�s�R        )��P	�S�L���A�*

	conv_loss��=Y�1        )��P	���L���A�*

	conv_loss��.=�Սf        )��P	T��L���A�*

	conv_loss {=V��        )��P	W��L���A�*

	conv_loss��h=}Yj�        )��P		�L���A�*

	conv_loss�/N=�,{        )��P	�H�L���A�*

	conv_lossY=i�W"        )��P	|x�L���A�*

	conv_loss1�/=��j        )��P	���L���A�*

	conv_loss�Y=OqD        )��P	���L���A�*

	conv_loss.�:=u��        )��P	y�L���A�*

	conv_lossJW�=�Awu        )��P	�E�L���A�*

	conv_loss�+=�t��        )��P	,v�L���A�*

	conv_loss8&M=�d��        )��P	5��L���A�*

	conv_loss2�=f�        )��P	���L���A�*

	conv_loss�*r=�
��        )��P	��L���A�*

	conv_loss4�+=D��        )��P	�5�L���A�*

	conv_loss�8={n�V        )��P	�d�L���A�*

	conv_loss<<=���E        )��P	���L���A�*

	conv_lossRe�=sf~        )��P	���L���A�*

	conv_lossU�=�܈        )��P	���L���A�*

	conv_loss�,=婗        )��P	�"�L���A�*

	conv_loss�=�q        )��P	�P�L���A�*

	conv_lossX�=NIU�        )��P	���L���A�*

	conv_loss�=��5�        )��P	[��L���A�*

	conv_loss�U=��        )��P	P��L���A�*

	conv_loss���<�Q��        )��P	��L���A�*

	conv_loss�26=$��        )��P	yB�L���A�*

	conv_loss_�?=��I�        )��P	�r�L���A�*

	conv_loss.r[=B��M        )��P	��L���A�*

	conv_loss&�=;�%        )��P	���L���A�*

	conv_loss��=�y{�        )��P	a�L���A�*

	conv_lossK>=��6        )��P	G8�L���A�*

	conv_loss��A=�<�        )��P	�j�L���A�*

	conv_lossi�;=l�2        )��P	���L���A�*

	conv_loss5�7=$��        )��P	��L���A�*

	conv_loss]�=�$�`        )��P	��L���A�*

	conv_loss�z!=�4؏        )��P	�&�L���A�*

	conv_loss��=���        )��P	"W�L���A�*

	conv_loss8g =ʓ�        )��P	���L���A�*

	conv_loss��=��o�        )��P	���L���A�*

	conv_lossW�6=�>/        )��P	��L���A�*

	conv_loss�7_=�m�J        )��P	�+�L���A�*

	conv_loss�V=���S        )��P	:^�L���A�*

	conv_losseH =�h��        )��P	���L���A�*

	conv_loss8�=N?Wz        )��P	���L���A�*

	conv_loss��@=�`�4        )��P	���L���A�*

	conv_loss�K�<gl�U        )��P	�1�L���A�*

	conv_loss�3=�
��        )��P	�c�L���A�*

	conv_loss��={�cz        )��P	G��L���A�*

	conv_lossUM�=\�Q�        )��P	Z��L���A�*

	conv_loss�@-=�iO        )��P	��L���A�*

	conv_loss���=���        )��P	`H�L���A�*

	conv_loss�Rm=�#	        )��P	{�L���A�*

	conv_lossd=8=����        )��P	��L���A�*

	conv_loss�8D=�?        )��P	5��L���A�*

	conv_loss�Y=ݜM�        )��P	��L���A�*

	conv_lossz7=���        )��P	HO�L���A�*

	conv_loss�*=p�G�        )��P	���L���A�*

	conv_lossI�8=	�        )��P	A��L���A�*

	conv_loss�Ig=�Z�        )��P	Z��L���A�*

	conv_lossEu<=��>        )��P	R$�L���A�*

	conv_loss�r=N�!        )��P	}f�L���A�*

	conv_loss���=�xp        )��P	q��L���A�*

	conv_lossT�=�Tn�        )��P	��L���A�*

	conv_lossx�L=�#�        )��P	2�L���A�*

	conv_loss$~S=��A�        )��P	==�L���A�*

	conv_loss��=#�4z        )��P	Br�L���A�*

	conv_loss�=U��        )��P	3��L���A�*

	conv_loss{U=4o�h        )��P	R��L���A�*

	conv_loss+��<���        )��P	��L���A�*

	conv_lossfb=R%��        )��P	�:�L���A�*

	conv_loss&�2=��j�        )��P	%p�L���A�*

	conv_loss�~�<{��        )��P	­�L���A�*

	conv_lossm��<��ll        )��P	\��L���A�*

	conv_loss�G=fK<�        )��P	/�L���A�*

	conv_lossv8[=OS@)        )��P	�K�L���A�*

	conv_loss;
E=�K��        )��P	�~�L���A�*

	conv_loss!{�<Z���        )��P	��L���A�*

	conv_lossGzM=��1�        )��P	\��L���A�*

	conv_loss�y=��F�        )��P	��L���A�*

	conv_loss�:=_        )��P	2O�L���A�*

	conv_loss�w7=x���        )��P	���L���A�*

	conv_loss9sM=W�p$        )��P	Q��L���A�*

	conv_loss*=˰#h        )��P	W��L���A�*

	conv_losss.G=�v        )��P	��L���A�*

	conv_lossS�D=�M>        )��P	aP�L���A�*

	conv_loss���= z�        )��P	B��L���A�*

	conv_loss�S)=���        )��P	ڷ�L���A�*

	conv_loss��!=���        )��P	X��L���A�*

	conv_lossHE5=���K        )��P	��L���A�*

	conv_loss&�=O��        )��P	C��L���A�*

	conv_loss�F=b ��        )��P	H��L���A�*

	conv_loss/�<=
���        )��P	��L���A�*

	conv_loss�=�=��Ұ        )��P	�J�L���A�*

	conv_lossƑ=cz�        )��P	{�L���A�*

	conv_loss!&=��A�        )��P	���L���A�*

	conv_loss�@=�~�        )��P	��L���A�*

	conv_loss�9=��        )��P	��L���A�*

	conv_loss�~=�}        )��P	J�L���A�*

	conv_loss^�='��        )��P	@��L���A�*

	conv_loss�k=8��        )��P	s��L���A�*

	conv_lossU��<��        )��P	J��L���A�*

	conv_loss-P�<<9��        )��P	�#�L���A�*

	conv_loss=�%=�@FH        )��P	xZ�L���A�*

	conv_loss�j�<�Yc+        )��P	(��L���A�*

	conv_loss;5�<�%k3        )��P	���L���A�*

	conv_lossW��<�ܣG        )��P	���L���A�*

	conv_loss2lI=��+c        )��P	��L���A�*

	conv_loss�*={��^        )��P	N�L���A�*

	conv_loss�AX=���x        )��P	v~�L���A�*

	conv_loss΋=�M�        )��P	���L���A�*

	conv_loss�;=�E�V        )��P	��L���A�*

	conv_loss(W@=cE�u        )��P	��L���A�*

	conv_lossɋ=����        )��P	T;�L���A�*

	conv_loss���<�c�        )��P	Ak�L���A�*

	conv_loss5�j=tw�%        )��P	Ù�L���A�*

	conv_loss�Q8=���        )��P	C��L���A�*

	conv_loss��=��%�        )��P	S��L���A�*

	conv_loss���<y�        )��P	�'�L���A�*

	conv_loss�v=h?�        )��P	lV�L���A�*

	conv_loss�=Em�*        )��P	4��L���A�*

	conv_lossE==E�n        )��P	´�L���A�*

	conv_loss�E=�^�T        )��P	B��L���A�*

	conv_loss�MM=�>�Q        )��P	6�L���A�*

	conv_loss�ݨ<Ryf�        )��P	�C�L���A�*

	conv_loss�We=����        )��P	�r�L���A�*

	conv_loss}�.=��        )��P	#��L���A�*

	conv_lossg�,=�b|�        )��P	���L���A�*

	conv_loss��F=����        )��P	y	�L���A�*

	conv_lossۊ=Y	�        )��P	Z9�L���A�*

	conv_loss��=��'z        )��P	 i�L���A�*

	conv_lossvo=�<��        )��P	��L���A�*

	conv_loss�=9�        )��P	v��L���A�*

	conv_losse�X=�V.        )��P	���L���A�*

	conv_losspJ =[̹        )��P	�+�L���A�*

	conv_loss�'= ��        )��P	Z�L���A�*

	conv_loss̝<=��i        )��P	w��L���A�*

	conv_lossS��<ȌT?        )��P	_��L���A�*

	conv_loss��b=�d��        )��P	���L���A�*

	conv_loss"�$=Ϸ�        )��P	��L���A�*

	conv_loss2�=�[��        )��P	e[�L���A�*

	conv_lossP�Z=�Ⱦd        )��P	���L���A�*

	conv_loss���<j        )��P	b��L���A�*

	conv_loss<$o=���        )��P	���L���A�*

	conv_lossfV�=o��D        )��P	"�L���A�*

	conv_loss�ƕ=�4��        )��P	�Z�L���A�*

	conv_loss^vU=RM*        )��P	��L���A�*

	conv_loss�KI=�*@        )��P	q��L���A�*

	conv_lossգ)=��1�        )��P	���L���A�*

	conv_loss"n=�~q�        )��P	?�L���A�*

	conv_lossW=�        )��P	ft�L���A�*

	conv_loss´=Xh/        )��P	@��L���A�*

	conv_loss�W=G�:a        )��P	j��L���A�*

	conv_loss��i=��_        )��P	��L���A�*

	conv_loss
�<��`P        )��P		O�L���A�*

	conv_loss�gL=���e        )��P	J��L���A�*

	conv_lossI�Y=:�        )��P	���L���A�*

	conv_loss�$$=��t        )��P	���L���A�*

	conv_lossU^H=)¤        )��P	9(�L���A�*

	conv_loss��Z=�b�        )��P	k^�L���A�*

	conv_loss�T0=����        )��P	���L���A�*

	conv_lossb�=+�%�        )��P	L��L���A�*

	conv_loss
�D=,�@        )��P	���L���A�*

	conv_lossf�=D	�`        )��P	61�L���A�*

	conv_loss#�l=�p�        )��P	�f�L���A�*

	conv_loss��=�&v        )��P	ƚ�L���A�*

	conv_loss��6=vwp        )��P	q��L���A�*

	conv_lossT�h=Nׯ        )��P	��L���A�*

	conv_lossVM�<��+K        )��P	�;�L���A�*

	conv_loss�,O=��7�        )��P	�s�L���A�*

	conv_lossV�N=Ɗ�l        )��P	"��L���A�*

	conv_loss��-=�h�v        )��P	;��L���A�*

	conv_lossп2=z�U        )��P	�L���A�*

	conv_loss�ӊ=cN$�        )��P		]�L���A�*

	conv_lossK�0==j�        )��P	��L���A�*

	conv_loss��=v5�        )��P	���L���A�*

	conv_lossI�=��        )��P	o �L���A�*

	conv_loss��=���        )��P	�:�L���A�*

	conv_loss>$M=誤�        )��P	�{�L���A�*

	conv_lossJ`s=��,�        )��P	��L���A�*

	conv_loss�xX=���W        )��P	���L���A�*

	conv_losstw�<���        )��P	
)�L���A�*

	conv_loss�`=�z�A        )��P	�n�L���A�*

	conv_loss́b=���        )��P	j��L���A�*

	conv_loss��'=F/�        )��P	>��L���A�*

	conv_loss�(=`�'�        )��P	��L���A�*

	conv_loss�@=���        )��P	tJ�L���A�*

	conv_lossX�==�bP        )��P	P��L���A�*

	conv_loss��*=eU\        )��P	��L���A�*

	conv_lossAS�<X���        )��P	"��L���A�*

	conv_loss��==��        )��P	V�L���A�*

	conv_loss(g�<<�4-        )��P	+��L���A�*

	conv_loss��N=�Se�        )��P	���L���A�*

	conv_loss��=P �        )��P	���L���A�*

	conv_lossd?=gV�        )��P	�/�L���A�*

	conv_lossTT3=�
        )��P	�y�L���A�*

	conv_loss�T="<c        )��P	f��L���A�*

	conv_lossC�.=�O�        )��P	a��L���A�*

	conv_loss-�=��F        )��P	,%�L���A�*

	conv_loss&%=Q�-U        )��P	~l�L���A�*

	conv_lossb��<��Q�        )��P	��L���A�*

	conv_loss1=bz�        )��P	C��L���A�*

	conv_loss}4=�M�k        )��P	��L���A�*

	conv_loss��<.���        )��P	yT�L���A�*

	conv_lossj�=��~        )��P	ؚ�L���A�*

	conv_lossu&=�嵼        )��P	:��L���A�*

	conv_loss
$1=����        )��P	F�L���A�*

	conv_loss/_=#�ә        )��P	�=�L���A�*

	conv_loss���<�}��        )��P	x�L���A�*

	conv_lossI�L=��`�        )��P	��L���A�*

	conv_lossà�=8B�~        )��P	��L���A�*

	conv_loss2O=u���        )��P	O.�L���A�*

	conv_loss	�	=����        )��P	�m�L���A�*

	conv_loss��_='��j        )��P	���L���A�*

	conv_loss7�=_��-        )��P	�
�L���A�*

	conv_losscJ=��(        )��P	�=�L���A�*

	conv_loss�+ =����        )��P	�o�L���A�*

	conv_loss��\=���.        )��P	j��L���A�*

	conv_losstZ=#�        )��P	q��L���A�*

	conv_loss�=L=+@��        )��P	i�L���A�*

	conv_loss�u<=��        )��P	TA�L���A�*

	conv_loss��=�V��        )��P	~|�L���A�*

	conv_loss�H=�cb�        )��P	D��L���A�*

	conv_loss�,>=`J��        )��P	��L���A�*

	conv_loss��==z&(        )��P	?�L���A�*

	conv_loss_�=hϝX        )��P	�r�L���A�*

	conv_loss�/N=6E��        )��P	��L���A�*

	conv_loss��J=2�        )��P	���L���A�*

	conv_loss�	=d\�        )��P	
�L���A�*

	conv_loss?73=�|:Q        )��P	�:�L���A�*

	conv_loss���<G�ck        )��P	5h�L���A�*

	conv_loss�X=�i�        )��P	[��L���A�*

	conv_loss��=�u�        )��P	+��L���A�*

	conv_loss��k=2�L        )��P	v �L���A�*

	conv_loss@�<8�=_        )��P	�1�L���A�*

	conv_lossܴ=�~s�        )��P	Ua�L���A�*

	conv_loss��D=��        )��P	Õ�L���A�*

	conv_lossA��<����        )��P	��L���A�*

	conv_loss@4=����        )��P	a�L���A�*

	conv_loss'j�=B�qN        )��P	g@�L���A�*

	conv_loss�Y�<���N        )��P	��L���A�*

	conv_loss�)=�k�        )��P	7��L���A�*

	conv_loss=e=��w�        )��P	���L���A�*

	conv_loss�A=���b        )��P	�L���A�*

	conv_lossg�K=ܚu'        )��P	�G�L���A�*

	conv_loss�-8=����        )��P	�{�L���A�*

	conv_loss�=���5        )��P	=��L���A�*

	conv_loss��;=��:�        )��P	j��L���A�*

	conv_loss��=)+4        )��P	>;�L���A�*

	conv_loss#m=�.Gf        )��P	�m�L���A�*

	conv_loss};=��ߐ        )��P	���L���A�*

	conv_lossm=�;�B        )��P	���L���A�*

	conv_loss'�*=���        )��P	��L���A�*

	conv_lossR$=����        )��P	sF�L���A�*

	conv_loss��<=B9�V        )��P	2x�L���A�*

	conv_loss�(=�.ӗ        )��P	n��L���A�*

	conv_lossF�= ��>        )��P	J��L���A�*

	conv_loss�KT=���        )��P	#%�L���A�*

	conv_loss�A2=���        )��P	�g�L���A�*

	conv_lossXx*="�č        )��P	���L���A�*

	conv_loss�ȅ=%�z        )��P	S��L���A�*

	conv_loss��]=��+0        )��P	��L���A�*

	conv_loss�#c=v���        )��P	�@�L���A�*

	conv_lossZ@=��^        )��P	_r�L���A�*

	conv_loss�bl=v�4        )��P	4��L���A�*

	conv_lossa�G=�d�>        )��P	���L���A�*

	conv_lossU�<r�y�        )��P	��L���A�*

	conv_loss�Az=�>�        )��P	xM�L���A�*

	conv_loss�=�K~�        )��P	N��L���A�*

	conv_loss��:= {32        )��P	��L���A�*

	conv_loss�xV=�6N        )��P	���L���A�*

	conv_loss?/=;��]        )��P	�4�L���A�*

	conv_lossޙ=3�w        )��P	�f�L���A�*

	conv_loss��,=$AV�        )��P	���L���A�*

	conv_loss3]= ���        )��P	���L���A�*

	conv_loss�Z:=3@�        )��P	>��L���A�*

	conv_loss[��<�Y7�        )��P	�4�L���A�*

	conv_lossc��<�zA        )��P	g�L���A�*

	conv_loss]^=�+�r        )��P	���L���A�*

	conv_loss��v=%���        )��P	���L���A�*

	conv_lossW�;=3�]�        )��P	�L���A�*

	conv_loss�?(=H�Χ        )��P	rE�L���A�*

	conv_loss|��<8���        )��P	n��L���A�*

	conv_lossm1u=��؎        )��P	���L���A�*

	conv_loss�4=��!s        )��P	���L���A�*

	conv_loss��<�D�o        )��P	D�L���A�*

	conv_loss�q#=���C        )��P	�M�L���A�*

	conv_loss"�'=�6�        )��P	Q��L���A�*

	conv_lossp�C=�Py�        )��P	���L���A�*

	conv_loss�6=����        )��P	o��L���A�*

	conv_loss�
=.�9        )��P	ɛ M���A�*

	conv_loss��<H��        )��P	� M���A�*

	conv_loss���<1 ��        )��P	�M���A�*

	conv_lossy�<=eT��        )��P	%7M���A�*

	conv_loss�y�<��fW        )��P	�nM���A�*

	conv_loss�0=P��f        )��P	��M���A�*

	conv_loss�]=�$        )��P	��M���A�*

	conv_loss��D=���B        )��P	{M���A�*

	conv_loss��=�I        )��P	�SM���A�*

	conv_loss�~=W���        )��P	��M���A�*

	conv_loss�4=�R�H        )��P	˾M���A�*

	conv_loss�-=
�˷        )��P	6M���A�*

	conv_loss��;=�=�        )��P	�7M���A�*

	conv_loss^�=��,@        )��P	�kM���A�*

	conv_loss��E=���u        )��P	��M���A�*

	conv_loss���<J:        )��P	d�M���A�*

	conv_loss�b=����        )��P	M���A�*

	conv_loss�8=�;�        )��P	NM���A�*

	conv_loss5�C=��
�        )��P	�M���A�*

	conv_loss���<A~��        )��P	�M���A�*

	conv_losst-�<.I�g        )��P	��M���A�*

	conv_loss�7Q=�4��        )��P	y0M���A�*

	conv_loss��X=��<�        )��P	�bM���A�*

	conv_loss�n==�lx        )��P	�M���A�*

	conv_lossQ�7=�⩂        )��P	��M���A�*

	conv_loss�;�<p#��        )��P	j�M���A�*

	conv_lossՈ=�z@        )��P	2M���A�*

	conv_loss���<nC�        )��P	�cM���A�*

	conv_losss"-=rs        )��P	�M���A�*

	conv_loss��d=�*.        )��P	�M���A�*

	conv_losssC�<C>.x        )��P	!�M���A�*

	conv_loss��D=}�B�        )��P	fFM���A�*

	conv_loss��G=.?9�        )��P	�{M���A�*

	conv_loss!�=%�e�        )��P	İM���A�*

	conv_loss�}=ܰ��        )��P	c�M���A�*

	conv_lossI�{=rȂj        )��P	�M���A�*

	conv_lossE�w=|H        )��P	�PM���A�*

	conv_loss�^�<���        )��P	>�M���A�*

	conv_loss� =,k&        )��P	��M���A�*

	conv_lossIn/=ߑ�        )��P	c�M���A�*

	conv_loss�K=�p�        )��P	�	M���A�*

	conv_loss�\:=S/��        )��P	�Z	M���A�*

	conv_loss�1=f�        )��P	�	M���A�*

	conv_loss�=�bw        )��P	��	M���A�*

	conv_loss�jP=�`˽        )��P	'�	M���A�*

	conv_losspq=�:U*        )��P	�?
M���A�*

	conv_loss��y=�'        )��P	Wx
M���A�*

	conv_loss�`�=s6!5        )��P	N�
M���A�*

	conv_loss��E=E7�F        )��P	��
M���A�*

	conv_loss
�l=e�W        )��P	hM���A�*

	conv_loss�U=4E�        )��P	YdM���A�*

	conv_loss�i7=����        )��P	�M���A�*

	conv_loss���<�5        )��P	�M���A�*

	conv_lossJ1 =��t0        )��P	��M���A�*

	conv_lossH�7=..v�        )��P	-2M���A�*

	conv_loss�)8=zܲ�        )��P	�zM���A�*

	conv_loss�	�<�tƺ        )��P	r�M���A�*

	conv_loss�Z*=���        )��P	��M���A�*

	conv_loss�s�< ��        )��P	�M���A�*

	conv_loss���<�M��        )��P	�QM���A�*

	conv_loss��0=��@        )��P	�M���A�*

	conv_loss���<��A<        )��P	��M���A�*

	conv_lossٱ4=���        )��P	~�M���A�*

	conv_loss�4�<O�        )��P	C#M���A�*

	conv_loss��<
:        )��P	�UM���A�*

	conv_loss��K=�m�.        )��P	[�M���A�*

	conv_lossP�-=t;3�        )��P	l�M���A�*

	conv_loss�Q@=J�Z�        )��P	�M���A�*

	conv_loss;�'=�kB�        )��P	G@M���A�*

	conv_loss��@=�Q)�        )��P	u�M���A�*

	conv_loss��/=ڤ��        )��P	_�M���A�*

	conv_loss���<A,��        )��P	��M���A�*

	conv_loss~�= ��W        )��P	,M���A�*

	conv_loss�=I�        )��P	`QM���A�*

	conv_lossR�=ܑ��        )��P	��M���A�*

	conv_lossV0=�$�"        )��P	�M���A�*

	conv_loss�OK=C�b�        )��P	��M���A�*

	conv_lossr��<��*?        )��P	}/M���A�*

	conv_loss��0=�o��        )��P	hM���A�*

	conv_loss��f=�f��        )��P	��M���A�*

	conv_loss@c=̩�I        )��P	�M���A�*

	conv_loss�,=�Zf�        )��P	�M���A�*

	conv_loss=�=�w0        )��P	�BM���A�*

	conv_lossL6�<���        )��P	�pM���A�*

	conv_loss�8;=#g7        )��P	w�M���A�*

	conv_loss{�<Y�b�        )��P	��M���A�*

	conv_loss�=�.�1        )��P	LM���A�*

	conv_lossh�'=A�        )��P	7EM���A�*

	conv_loss��<=ϳn�        )��P	=wM���A�*

	conv_lossgkW=N?Z        )��P	Z�M���A�*

	conv_loss�,:=v�D        )��P	K�M���A�*

	conv_loss1y=��        )��P	�M���A�*

	conv_loss�Xk=��6�        )��P	fQM���A�*

	conv_loss<A==Y8�!        )��P	S�M���A�*

	conv_loss.�`=,)lb        )��P	״M���A�*

	conv_loss�=�R�`        )��P	$�M���A�*

	conv_loss��=����        )��P	�"M���A�*

	conv_loss��M=A�ޏ        )��P	TVM���A�*

	conv_loss'=��,n        )��P	�M���A�*

	conv_loss>A`=j��E        )��P	��M���A�*

	conv_lossaV'=�t�        )��P	�M���A�*

	conv_loss�!=X�        )��P	�>M���A�*

	conv_loss=F =B��        )��P	KxM���A�*

	conv_lossi�d=�:W�        )��P	b�M���A�*

	conv_loss��>=��        )��P	��M���A�*

	conv_loss��!=�˽_        )��P	�(M���A�*

	conv_loss}4=�>        )��P	�ZM���A�*

	conv_losseg=ޱ�        )��P	U�M���A�*

	conv_loss�P�<U��        )��P	m�M���A�*

	conv_loss�`�<�k\Z        )��P	)M���A�*

	conv_losse�6=�r��        )��P	�6M���A�*

	conv_loss��=�7�Z        )��P	�iM���A�*

	conv_loss��=&U2G        )��P	P�M���A�*

	conv_lossN�=����        )��P	��M���A�*

	conv_loss��<D���        )��P	�M���A�*

	conv_lossr(�<���        )��P	�NM���A�*

	conv_loss3"k=a�5�        )��P	��M���A�*

	conv_losslf=���        )��P	��M���A�*

	conv_loss�/	=Ag�^        )��P	��M���A�*

	conv_loss�:�<��        )��P	�"M���A�*

	conv_loss9�@=_B        )��P	FXM���A�*

	conv_loss��=� �        )��P	��M���A�*

	conv_loss"�H=&�K�        )��P	��M���A�*

	conv_loss�J�<梻B        )��P	U�M���A�*

	conv_loss��=�BJY        )��P	c1M���A�*

	conv_loss>n=�6c        )��P	DcM���A�*

	conv_loss�H=��)p        )��P	��M���A�*

	conv_lossÓ=��3C        )��P	��M���A�*

	conv_loss�W'=�5�        )��P	�M���A�*

	conv_lossi�Q=u�9        )��P	�5M���A�*

	conv_loss�S=�6�        )��P	!kM���A�*

	conv_loss7�"=����        )��P	��M���A�*

	conv_loss?�X=[        )��P	�M���A�*

	conv_loss�O=g��        )��P	�M���A�*

	conv_loss�!=8�D�        )��P	?M���A�*

	conv_loss�Y=�]�        )��P	�pM���A�*

	conv_loss��=��        )��P	��M���A�*

	conv_loss@/=*��)        )��P	b�M���A�*

	conv_loss��\=�E        )��P	�	M���A�*

	conv_loss��=����        )��P	�@M���A�*

	conv_loss��^=_���        )��P	SrM���A�*

	conv_loss��=��        )��P	��M���A�*

	conv_loss)��<q7�        )��P	b�M���A�*

	conv_loss��<���]        )��P	.M���A�*

	conv_lossM� =a&�        )��P	KM���A�*

	conv_loss�=�M�        )��P	A|M���A�*

	conv_lossʮ<КW        )��P	��M���A�*

	conv_loss�]R=��s�        )��P	>�M���A�*

	conv_loss��=���        )��P	� M���A�*

	conv_loss�/=����        )��P	uG M���A�*

	conv_loss,�<tu��        )��P	B� M���A�*

	conv_loss �=�RZ-        )��P	;� M���A�*

	conv_lossM?=�Z�
        )��P	}!M���A�*

	conv_loss�<
S        )��P	�9!M���A�*

	conv_loss�=?�`        )��P	�o!M���A�*

	conv_loss�D=�3�        )��P	��!M���A�*

	conv_loss �<E�_#        )��P	}�!M���A�*

	conv_loss�-=�ؼ#        )��P	�"M���A�*

	conv_loss)d[=zz��        )��P	�<"M���A�*

	conv_loss���<k��p        )��P	p"M���A�*

	conv_loss��_=�Nh�        )��P	�"M���A�*

	conv_loss��{=�)5�        )��P	_�"M���A�*

	conv_loss:YB=.�0�        )��P	�#M���A�*

	conv_loss�*`=��%�        )��P	bF#M���A�*

	conv_loss��=X��        )��P	�z#M���A�*

	conv_loss�--=;]U�        )��P	Ԩ#M���A�*

	conv_lossD];=u�l�        )��P	n�#M���A�*

	conv_loss:|�<���t        )��P	�$M���A�*

	conv_loss�c7=q��	        )��P	�B$M���A�*

	conv_loss��S=<��        )��P	�u$M���A�*

	conv_loss��B=���=        )��P	�$M���A�*

	conv_loss�#�<
�t�        )��P	J�$M���A�*

	conv_loss`=[�2        )��P	n%M���A�*

	conv_loss�<���        )��P	E%M���A�*

	conv_loss"7�<�-        )��P	�v%M���A�*

	conv_loss3��<h��        )��P	�%M���A�*

	conv_lossd��<�aq�        )��P	�%M���A�*

	conv_losst!=c0��        )��P	�&M���A�*

	conv_lossQm=d3 �        )��P	KH&M���A�*

	conv_loss��!=�{�        )��P	`x&M���A�*

	conv_loss��=e��I        )��P	�&M���A�*

	conv_loss#G=�z�        )��P	2�&M���A�*

	conv_loss�5�<�߿        )��P	�
'M���A�*

	conv_loss�=���        )��P	�;'M���A�*

	conv_loss��<�_x        )��P	�j'M���A�*

	conv_loss�B=��.*        )��P	S�'M���A�*

	conv_lossx�l==��        )��P	��'M���A�*

	conv_loss��=�-��        )��P	��'M���A�*

	conv_lossg7!=�g��        )��P	u3(M���A�*

	conv_lossB'=w/        )��P	�j(M���A�*

	conv_loss�}4=�%x{        )��P	��(M���A�*

	conv_loss�B=���        )��P	��(M���A�*

	conv_loss��@='r0-        )��P	/)M���A�*

	conv_loss��=���t        )��P	D)M���A�*

	conv_loss&� =��#        )��P	y)M���A�*

	conv_lossۗ�<��a        )��P	D�)M���A�*

	conv_loss���<YH��        )��P	`�)M���A�*

	conv_loss˶=��        )��P	�*M���A�*

	conv_loss2��<k�,        )��P	�L*M���A�*

	conv_loss�.=���N        )��P	}�.M���A�*

	conv_loss��O=��7�        )��P	��0M���A�*

	conv_loss�2�<{]	y        )��P	C�0M���A�*

	conv_loss��5=[|��        )��P	E1M���A�*

	conv_lossH,=���7        )��P	D1M���A�*

	conv_loss%�K=��%c        )��P	,w1M���A�*

	conv_lossU.O=wQ��        )��P	u�1M���A�*

	conv_lossʈ�<��	S        )��P	��1M���A�*

	conv_loss��=��=        )��P	�2M���A�*

	conv_lossO@=3�YM        )��P	�P2M���A�*

	conv_loss�)�<��L�        )��P	Y�2M���A�*

	conv_lossl$E=�+s        )��P	��2M���A�*

	conv_loss��6=�1�]        )��P	d�2M���A�*

	conv_loss=P=xaX�        )��P	�+3M���A�*

	conv_loss�I'=v�Y�        )��P	!^3M���A�*

	conv_loss���<��        )��P	��3M���A�*

	conv_loss�==r��        )��P	�3M���A�*

	conv_loss��<�lϔ        )��P	w�3M���A�*

	conv_loss�@=h)�        )��P	`.4M���A�*

	conv_lossN�*=�W��        )��P	�]4M���A�*

	conv_loss�5�<Ě�C        )��P	7�4M���A�*

	conv_loss�D�=�*��        )��P	1�4M���A�*

	conv_loss�˦<����        )��P	��4M���A�*

	conv_lossc`	=X��        )��P	155M���A�*

	conv_loss&��<�)'s        )��P	Zj5M���A�*

	conv_loss>�=8��        )��P	ܰ5M���A�*

	conv_loss�On=z��}        )��P	a�5M���A�*

	conv_loss8�='�        )��P	�6M���A�*

	conv_loss#CR= ���        )��P	�L6M���A�*

	conv_loss�E=��	        )��P	�6M���A�*

	conv_loss�Q�<.S��        )��P	d�6M���A�*

	conv_loss���=Hڤ�        )��P	��6M���A�*

	conv_loss�e=!�=�        )��P	�7M���A�*

	conv_loss�$=r�=h        )��P	�R7M���A�*

	conv_loss��=���        )��P	��7M���A�*

	conv_loss�F=%���        )��P	��7M���A�*

	conv_loss8�<=��        )��P	 8M���A�*

	conv_loss�=՟�        )��P	^88M���A�*

	conv_lossg%=O��        )��P	�k8M���A�*

	conv_loss>� =�ǘ�        )��P	��8M���A�*

	conv_loss�I)=��0:        )��P	V�8M���A�*

	conv_loss��,=�q�:        )��P	~�8M���A�*

	conv_loss�}5=��8�        )��P	q79M���A�*

	conv_loss�+=N(�        )��P	�f9M���A�*

	conv_loss�h�<h���        )��P	7�9M���A�*

	conv_loss@3=��&�        )��P	��9M���A�*

	conv_loss���<���i        )��P	�:M���A�*

	conv_loss���<�Ň�        )��P	F:M���A�*

	conv_loss,��<�.        )��P	�}:M���A�*

	conv_loss��=���        )��P	V�:M���A�*

	conv_lossc8D=z;J�        )��P	)�:M���A�*

	conv_loss���<��        )��P	�.;M���A�*

	conv_loss��=��d        )��P	b;M���A�*

	conv_loss��:=c���        )��P	�;M���A�*

	conv_losss�<UԒ�        )��P	��;M���A�*

	conv_losso�=��&%        )��P	.<M���A�*

	conv_loss�4�=�i�        )��P	�:<M���A�*

	conv_loss��8=en�        )��P	�v<M���A�*

	conv_loss�=�@        )��P	
�<M���A�*

	conv_loss) �<�ԙ        )��P	D�<M���A�*

	conv_lossln=����        )��P	(=M���A�*

	conv_lossP�4=���        )��P	U=M���A�*

	conv_lossW0=����        )��P	�=M���A�*

	conv_lossL�\=Z�)^        )��P		�=M���A�*

	conv_loss�.4=;��,        )��P	u�=M���A�*

	conv_loss�@=&�        )��P	�&>M���A�*

	conv_loss���<����        )��P		U>M���A�*

	conv_loss�i=｠^        )��P	��>M���A�*

	conv_loss|=�3BP        )��P	7�>M���A�*

	conv_loss0��<�        )��P	;�>M���A�*

	conv_loss�j�<�!�        )��P	�0?M���A�*

	conv_loss�N]=���        )��P	�c?M���A�*

	conv_loss�=Z���        )��P	^�?M���A�*

	conv_loss&��<aq,}        )��P	i�?M���A�*

	conv_loss�Z=��/P        )��P	�@M���A�*

	conv_loss�_=�Ć        )��P	�4@M���A�*

	conv_loss!��<��R        )��P	�g@M���A�*

	conv_lossd1=���        )��P	ț@M���A�*

	conv_loss��/=!��y        )��P	�@M���A�*

	conv_loss�j=�FI�        )��P	�AM���A�*

	conv_loss$�<K[Ò        )��P	_CAM���A�*

	conv_loss���<z�e        )��P	�qAM���A�*

	conv_loss�)'=�        )��P	,�AM���A�*

	conv_loss�V=6a�        )��P	��AM���A�*

	conv_loss۷�<�*@        )��P	�BM���A�*

	conv_lossIF)=[�&        )��P	�HBM���A�*

	conv_loss[=���        )��P	-yBM���A�*

	conv_loss���</K\        )��P	2�BM���A�*

	conv_loss�=hǍ        )��P	��BM���A�*

	conv_loss�=My        )��P	�CM���A�*

	conv_loss�=� (        )��P	z9CM���A�*

	conv_loss�N$=Μm        )��P	%jCM���A�*

	conv_lossjJ'=	��        )��P	 �CM���A�*

	conv_loss��<�(q[        )��P	��CM���A�*

	conv_loss`�:=��        )��P	e	DM���A�*

	conv_loss�d=)�P        )��P	�EDM���A�*

	conv_loss4ä<p�6�        )��P	cyDM���A�*

	conv_loss>nV=@�        )��P	I�DM���A�*

	conv_loss��=�|�        )��P	��DM���A�*

	conv_loss�3=����        )��P	�EM���A�*

	conv_loss5� =��?!        )��P	(LEM���A�*

	conv_loss4K0==��h        )��P	��EM���A�*

	conv_lossI��<���{        )��P	g�EM���A�*

	conv_losse��<�}�b        )��P	��EM���A�*

	conv_lossӌ2=T��1        )��P	k/FM���A�*

	conv_loss�05=��i)        )��P	hbFM���A�*

	conv_loss�G=�z��        )��P	}�FM���A�*

	conv_losst�=1~�U        )��P	��FM���A�*

	conv_loss� L=�u�&        )��P	�FM���A�*

	conv_losseP=�5E�        )��P	r.GM���A�*

	conv_loss��<)�д        )��P	4fGM���A�*

	conv_lossv�=_1��        )��P	7�GM���A�*

	conv_loss�O�<`Mk�        )��P	/�GM���A�*

	conv_loss��=])�*        )��P	��GM���A�*

	conv_loss�n6=�h�        )��P	�"HM���A�*

	conv_loss� ={җY        )��P	[XHM���A�*

	conv_loss� =2:        )��P	؉HM���A�*

	conv_loss���<�ƛ        )��P	�HM���A�*

	conv_lossZl�<u�#�        )��P	��HM���A�*

	conv_loss��z=�#�        )��P	]IM���A�*

	conv_loss�;=;8A        )��P	PVIM���A�*

	conv_loss?=K:��        )��P	�IM���A�*

	conv_loss�9=��L        )��P	ϸIM���A�*

	conv_loss�5:=��oX        )��P	��IM���A�*

	conv_loss�=���        )��P	�JM���A�*

	conv_loss8.=��&j        )��P	�KJM���A�*

	conv_loss͏�<e�H�        )��P	�{JM���A�*

	conv_loss���<�P��        )��P	��JM���A�*

	conv_loss��$=+�Pv        )��P	�JM���A�*

	conv_loss���<4)��        )��P	�KM���A�*

	conv_loss��=8�B�        )��P	�=KM���A�*

	conv_loss�F=��k        )��P	AmKM���A�*

	conv_loss�M=d�F        )��P	��KM���A�*

	conv_loss�=��        )��P	��KM���A�*

	conv_loss3Ѷ<MY&x        )��P	9�KM���A�*

	conv_lossv&=*�-        )��P	�.LM���A�*

	conv_loss|�6=���        )��P	�\LM���A�*

	conv_loss�=DL�|        )��P	<�LM���A�*

	conv_loss��E=�z�        )��P	M�LM���A�*

	conv_losse�z="�        )��P	Z�LM���A�*

	conv_loss�=�,�        )��P	/MM���A�*

	conv_loss[B�<�S��        )��P	�CMM���A�*

	conv_loss=���        )��P	�rMM���A�*

	conv_loss"�=�6        )��P	�MM���A�*

	conv_loss;=W�Ns        )��P	�MM���A�*

	conv_loss �=��=        )��P	� NM���A�*

	conv_loss�d=?�q        )��P	w3NM���A�*

	conv_loss�{�<�K         )��P	�`NM���A�*

	conv_loss�.=c�7�        )��P	�NM���A�*

	conv_loss�U=�Ȕ5        )��P	%�NM���A�*

	conv_loss2�:=҈��        )��P	Q�NM���A�*

	conv_loss�<$�l        )��P	8OM���A�*

	conv_loss0�=kOW@        )��P	2vOM���A�*

	conv_loss��=g_'        )��P	ǩOM���A�*

	conv_loss?�r=�y�8        )��P	��OM���A�*

	conv_lossd�<9*�        )��P	�PM���A�*

	conv_lossgy2=��^+        )��P	�KPM���A�*

	conv_lossd=�J�        )��P	��PM���A�*

	conv_losse�?=��,@        )��P	��PM���A�*

	conv_loss�@=�`��        )��P	�QM���A�*

	conv_lossAj�<3��        )��P	E=QM���A�*

	conv_loss�%`=W�q�        )��P	�{QM���A�*

	conv_lossH�=����        )��P	T�QM���A�*

	conv_loss�$=E�Sw        )��P	�QM���A�*

	conv_loss�b�<�G>�        )��P	URM���A�*

	conv_loss��5==a�        )��P	LORM���A�*

	conv_loss��==\.�        )��P	k�RM���A�*

	conv_loss�[=ۥ2        )��P	Q�RM���A�*

	conv_loss(�,=uI�        )��P	\�RM���A�*

	conv_loss�1�=u��        )��P	�4SM���A�*

	conv_loss��=�|��        )��P	�nSM���A�*

	conv_lossEL0=�c�!        )��P	��SM���A�*

	conv_loss��=��*        )��P	��SM���A�*

	conv_loss���<
Gkw        )��P	�
TM���A�*

	conv_loss��<��=        )��P	ETM���A�*

	conv_loss,�=���        )��P	�TM���A�*

	conv_lossG�=����        )��P	<�TM���A�*

	conv_loss[��<�^.        )��P	r�TM���A�*

	conv_loss��=����        )��P	y$UM���A�*

	conv_lossM�'=��c�        )��P	GWUM���A�*

	conv_loss�+�<T�=        )��P	��UM���A�*

	conv_loss��r=���	        )��P	��UM���A�*

	conv_lossP�=��        )��P	�VM���A�*

	conv_lossV�O=��Vv        )��P	�IVM���A�*

	conv_loss��<��C�        )��P	�}VM���A�*

	conv_loss̏&=��p        )��P	��VM���A�*

	conv_lossC�!=���i        )��P	��VM���A�*

	conv_lossR�<�M�}        )��P	� WM���A�*

	conv_loss��&=B�D�        )��P	�cWM���A�*

	conv_lossx>Z=Q��T        )��P	ՔWM���A�*

	conv_loss1?=��B�        )��P	��WM���A�*

	conv_loss�.=ڔ�        )��P	�WM���A�*

	conv_loss]5=\�,7        )��P	�4XM���A�*

	conv_lossj�C=���        )��P	gvXM���A�*

	conv_lossd�=}%�        )��P	v�XM���A�*

	conv_loss�9=����        )��P	��XM���A�*

	conv_lossI��<��,        )��P	�YM���A�*

	conv_loss٩�<��ß        )��P	?QYM���A�*

	conv_loss���<�<�         )��P	�YM���A�*

	conv_loss?�7=K�>�        )��P	��YM���A�*

	conv_lossZ�$=T^�1        )��P	��YM���A�*

	conv_loss�Q�< �s        )��P	�[M���A�*

	conv_loss��*=�L��        )��P	��[M���A�*

	conv_lossL3=���        )��P	\M���A�*

	conv_loss�21=����        )��P	�H\M���A�*

	conv_loss�F=�k%        )��P	8�\M���A�*

	conv_loss��'=��]�        )��P	��\M���A�*

	conv_lossv=����        )��P	�\M���A�*

	conv_lossL=7��        )��P	�4]M���A�*

	conv_loss�=5�(        )��P	x]M���A�*

	conv_loss��=f��        )��P	b�]M���A�*

	conv_loss��&=�$�Y        )��P	�]M���A�*

	conv_lossK�*=��        )��P	D+^M���A�*

	conv_loss{�=�<�        )��P	�]^M���A�*

	conv_loss��=~�q        )��P	�^M���A�*

	conv_lossb�?=��        )��P	9�^M���A�*

	conv_lossF�-=����        )��P	�_M���A�*

	conv_loss�u�<�[O�        )��P	\6_M���A�*

	conv_loss��<�S�         )��P	�m_M���A�*

	conv_loss�O<=��F�        )��P	U�_M���A�*

	conv_loss���<��X        )��P	��_M���A�*

	conv_loss�4	=-�1�        )��P	�`M���A�*

	conv_loss���<=e��        )��P	�L`M���A�*

	conv_lossq�>=�K?        )��P	�`M���A�*

	conv_loss&�#=vu��        )��P	!�`M���A�*

	conv_lossF�f=���        )��P	7�`M���A�*

	conv_loss�=[��        )��P	�*aM���A�*

	conv_lossW��<��*q        )��P	�[aM���A�*

	conv_loss*I=�|IH        )��P	��aM���A�*

	conv_loss�'=\�y        )��P	��aM���A�*

	conv_loss�F5=*)��        )��P	�bM���A�*

	conv_lossYZ=İj6        )��P	�7bM���A�*

	conv_loss�01=R:Hx        )��P	6rbM���A�*

	conv_loss�z�<1R O        )��P	ӧbM���A�*

	conv_loss]B=�}�        )��P	��bM���A�*

	conv_lossS�/=�{�        )��P	cM���A�*

	conv_loss�q
=a�        )��P	FJcM���A�*

	conv_loss�y=h�c        )��P	�|cM���A�*

	conv_loss�l=��G�        )��P	�cM���A�*

	conv_loss�#=����        )��P	��cM���A�*

	conv_loss��2=��b        )��P	$%dM���A�*

	conv_loss�_:=�k�        )��P	oUdM���A�*

	conv_lossU=�ޭ        )��P	q�dM���A�*

	conv_loss2!=�΂-        )��P	;�dM���A�*

	conv_loss�`d= �C�        )��P	��dM���A�*

	conv_loss<%=�n�        )��P	@6eM���A�*

	conv_lossxb(=�B�M        )��P		heM���A�*

	conv_loss5=�C�        )��P	�eM���A�*

	conv_loss$C=@�>�        )��P	3�eM���A�*

	conv_loss��=7���        )��P	�fM���A�*

	conv_loss�v�<	�        )��P	'FfM���A�*

	conv_loss��1=l��Q        )��P		�fM���A�*

	conv_loss��$=g�P        )��P	�fM���A�*

	conv_loss��=r)#s        )��P	f�fM���A�*

	conv_loss8�(=�dɭ        )��P	C4gM���A�*

	conv_loss$�<w(/�        )��P	SlgM���A�*

	conv_loss��)=��G        )��P	ؤgM���A�*

	conv_loss�y�<o�p�        )��P	��gM���A�*

	conv_loss��
=�U        )��P	�
hM���A�*

	conv_loss)[\=�k        )��P	DhM���A�*

	conv_loss:{=��mU        )��P	vhM���A�*

	conv_loss���<�;E�        )��P	~�hM���A�*

	conv_loss�C�<�?y        )��P	b�hM���A�*

	conv_loss_I=��H        )��P	QiM���A�*

	conv_loss�:=��        )��P	4JiM���A�*

	conv_lossȀ�<Q��        )��P	�{iM���A�*

	conv_loss��=9/tV        )��P	߭iM���A�*

	conv_loss�=}Uf        )��P	z�iM���A�*

	conv_loss3D�<8lV        )��P	�jM���A�*

	conv_lossO�<�1��        )��P	�ZjM���A�*

	conv_loss�fy=@&�        )��P	;�jM���A�*

	conv_loss,�&=0�b        )��P	��jM���A�*

	conv_loss��^=r$Ź        )��P	!�jM���A�*

	conv_loss�9�<�h�        )��P	SkM���A�*

	conv_loss�Q^=�o        )��P	�UkM���A�*

	conv_loss D	=� )        )��P	��kM���A�*

	conv_loss6�3=�lk�        )��P	I�kM���A�*

	conv_loss�*=?( _        )��P	��kM���A�*

	conv_loss[�L=���y        )��P	$lM���A�*

	conv_lossg =-��M        )��P	�WlM���A�*

	conv_lossf�&=NMh�        )��P	i�lM���A�*

	conv_loss�3�<�(�        )��P	��lM���A�*

	conv_loss|�=�;��        )��P	o�lM���A�*

	conv_loss�O.=u�g        )��P	�*mM���A�*

	conv_lossV�2=^��z        )��P	�\mM���A�*

	conv_loss'b=�w�        )��P	��mM���A�*

	conv_loss�=X[V�        )��P	h�mM���A�*

	conv_lossճ+=��x        )��P	@�mM���A�*

	conv_lossV��<��F        )��P	�(nM���A�*

	conv_lossz5=@9�6        )��P	�_nM���A�*

	conv_loss�J=#�d�        )��P	˓nM���A�*

	conv_loss�'=�[�        )��P	y�nM���A�*

	conv_loss%�<�7��        )��P	�nM���A�*

	conv_loss���<����        )��P	�2oM���A�*

	conv_loss��<w�m        )��P	�goM���A�*

	conv_loss}�g=�GX�        )��P	l�oM���A�*

	conv_loss)�(='C�h        )��P	U�oM���A�*

	conv_lossU��<:~��        )��P	EpM���A�*

	conv_loss�b�<6P�S        )��P	�;pM���A�*

	conv_loss=n�<���P        )��P	|tpM���A�*

	conv_loss�`�<�S>�        )��P	*�pM���A�*

	conv_lossf��<��.u        )��P	j�pM���A�*

	conv_loss��<�i        )��P	�qM���A�*

	conv_lossh%=%�        )��P	�QqM���A�*

	conv_lossK#�<4��        )��P	��qM���A�*

	conv_loss�2-=h�e%        )��P	��qM���A�*

	conv_loss1��<��q        )��P	��qM���A�*

	conv_loss֌�< ���        )��P	�.rM���A�*

	conv_loss��&=֍�	        )��P	�^rM���A�*

	conv_loss�)�<ͦC�        )��P	��rM���A�*

	conv_loss�S�<�=�        )��P	�rM���A�*

	conv_lossC��<m��        )��P	�rM���A�*

	conv_lossD�<!u�X        )��P	=2sM���A�*

	conv_loss/Nq=�FR�        )��P	�ssM���A�*

	conv_loss=�=�#b`        )��P	ӦsM���A�*

	conv_loss�p=5�s        )��P	B�sM���A�*

	conv_losst�#=N%ǘ        )��P	rtM���A�*

	conv_loss�a�<Q ��        )��P	�MtM���A�*

	conv_loss9H1=�W8�        )��P	*�tM���A�*

	conv_loss��=TC̳        )��P	�tM���A�*

	conv_loss���<ql�        )��P	��tM���A�*

	conv_loss"�&=>h[        )��P	�&uM���A�*

	conv_loss���</6�
        )��P	ZuM���A�*

	conv_loss��=��        )��P	�uM���A�*

	conv_lossW��<�;�h        )��P	@�uM���A�*

	conv_loss�2=��~�        )��P	��uM���A�*

	conv_lossjd=�y0�        )��P	�$vM���A�*

	conv_loss��u=;�        )��P	�VvM���A�*

	conv_loss�N=��A        )��P	I�vM���A�*

	conv_loss�Ne= N�        )��P	�vM���A�*

	conv_loss�
=JJW>        )��P	��vM���A�*

	conv_loss�=�sd        )��P	-wM���A�*

	conv_loss���<'�tW        )��P	i^wM���A�*

	conv_loss5N:=��B        )��P	M�wM���A�*

	conv_loss�T=U"M_        )��P	��wM���A�*

	conv_loss�1�<�0U�        )��P	.�wM���A�*

	conv_loss��<�<�        )��P	�2xM���A�*

	conv_loss9��<���G        )��P	�exM���A�*

	conv_loss��=UsQ�        )��P	��xM���A�*

	conv_lossr��=�-��        )��P	
�xM���A�*

	conv_loss��<����        )��P	x
yM���A�*

	conv_lossy�=�Z�        )��P	&OyM���A�*

	conv_loss�g�<�`�i        )��P	�yM���A�*

	conv_loss�E=���        )��P	ѼyM���A�*

	conv_lossd��<^Yf        )��P	�yM���A�*

	conv_loss=U�<�~%+        )��P	�zM���A�*

	conv_loss(y=屣�        )��P	�OzM���A�*

	conv_lossX�d=F6�?        )��P	��zM���A�*

	conv_lossF�)=�vLe        )��P	O�zM���A�*

	conv_loss�@"=e��        )��P	�zM���A�*

	conv_loss)!=Ȅ��        )��P	s%{M���A�*

	conv_loss0{	=7W�        )��P	�i{M���A�*

	conv_loss�s)=�x^        )��P	��{M���A�*

	conv_loss��<8�c�        )��P	��{M���A�*

	conv_lossa�C=5e�        )��P	�|M���A�*

	conv_loss6 =� �x        )��P	{?|M���A�*

	conv_loss�=��>�        )��P	v|M���A�*

	conv_loss�==�K�        )��P	��|M���A�*

	conv_loss���<Q�2.        )��P	��|M���A�*

	conv_losst��<Q��.        )��P	f}M���A�*

	conv_loss�PF=d]��        )��P	NM}M���A�*

	conv_loss���<��        )��P	�~}M���A�*

	conv_loss�=q���        )��P	��}M���A�*

	conv_loss�=N�U�        )��P	��}M���A�*

	conv_loss�,�<����        )��P	�,~M���A�*

	conv_loss�}=^r��        )��P	�b~M���A�*

	conv_loss���<���        )��P	�~M���A�*

	conv_loss*w�<�%cE        )��P	��~M���A�*

	conv_loss6f}=Un:�        )��P	QM���A�*

	conv_loss�=�Wg        )��P	:M���A�*

	conv_lossŢ=P�'        )��P	RlM���A�*

	conv_lossu�=.�z\        )��P	��M���A�*

	conv_loss��=vr0        )��P	��M���A�*

	conv_loss��=2�        )��P	�M���A�*

	conv_loss =�)��        )��P	�@�M���A�*

	conv_loss�:�<�vӪ        )��P	�t�M���A�*

	conv_loss���<��A�        )��P	���M���A�*

	conv_loss�!=�ڒv        )��P	���M���A�*

	conv_lossM#2=���        )��P	�)�M���A�*

	conv_loss��<�D�C        )��P	�X�M���A�*

	conv_loss�D=^o͇        )��P	��M���A�*

	conv_loss��=2�s2        )��P	긁M���A�*

	conv_loss+��<2���        )��P	��M���A�*

	conv_loss�)=���,        )��P	h&�M���A�*

	conv_loss��=�l.�        )��P	�Z�M���A�*

	conv_loss��w=N�f=        )��P	v��M���A�*

	conv_lossA=��"2        )��P	S��M���A�*

	conv_loss���<���        )��P	���M���A�*

	conv_loss�%=b�i        )��P	�/�M���A�*

	conv_loss�0�<�Y�        )��P	�i�M���A�*

	conv_lossF#=ޓ��        )��P	K��M���A�*

	conv_lossg=I2X�        )��P	L҃M���A�*

	conv_loss�4=���        )��P	=�M���A�*

	conv_loss�h�<j��        )��P	p9�M���A�*

	conv_loss��=��2        )��P	�k�M���A�*

	conv_loss�w=	���        )��P	���M���A�*

	conv_lossk�=�t�        )��P	фM���A�*

	conv_loss��3=_�(        )��P	��M���A�*

	conv_loss�'=��X�        )��P	�<�M���A�*

	conv_loss��.=����        )��P	"p�M���A�*

	conv_loss�D6=���        )��P	���M���A�*

	conv_loss�)=Z�ny        )��P	_c�M���A�*

	conv_loss�V=�k�        )��P	Ș�M���A�*

	conv_loss���<-wy        )��P	GʇM���A�*

	conv_loss]��<���        )��P	���M���A�*

	conv_losst '=Y2��        )��P	�1�M���A�*

	conv_lossB�<t8�        )��P	lv�M���A�*

	conv_loss$��<�hQ        )��P	���M���A�*

	conv_loss���<"*Ɨ        )��P	�݈M���A�*

	conv_loss��g="�r        )��P	��M���A�*

	conv_lossQ�=e���        )��P	�@�M���A�*

	conv_loss�=�rk"        )��P	���M���A�*

	conv_loss�q=N�Q        )��P	���M���A�*

	conv_loss��-=L�g�        )��P	��M���A�*

	conv_lossP�<���        )��P	Q�M���A�*

	conv_loss|V�<K}��        )��P	?N�M���A�*

	conv_lossZv�<�	��        )��P	��M���A�*

	conv_loss��<|焊        )��P	r��M���A�*

	conv_loss���<��X�        )��P	C��M���A�*

	conv_loss�ɱ<XQv�        )��P	�*�M���A�*

	conv_loss�ݡ<�4�        )��P	Ma�M���A�*

	conv_loss�$=_:        )��P	���M���A�*

	conv_loss��<e���        )��P	�M���A�*

	conv_loss���=����        )��P	��M���A�*

	conv_losss*<=~�p        )��P	�(�M���A�*

	conv_loss~l�<_��J        )��P	�a�M���A�*

	conv_loss�u�<�À        )��P	��M���A�*

	conv_loss��=4�G(        )��P	�ԌM���A�*

	conv_lossu��<Nr�        )��P	.	�M���A�*

	conv_loss �<G���        )��P	�F�M���A�*

	conv_lossDc�<@o�O        )��P	���M���A�*

	conv_loss� �<���        )��P	rōM���A�*

	conv_loss	=R��        )��P	���M���A�*

	conv_loss�>�<�R�        )��P	;,�M���A�*

	conv_lossH�<��'�        )��P	T[�M���A�*

	conv_loss;)=�h�        )��P	܊�M���A�*

	conv_lossDE={;        )��P	tM���A�*

	conv_loss��(=E,
d        )��P	`��M���A�*

	conv_loss8�$=�;��        )��P	�1�M���A�*

	conv_loss`[
=�Pk�        )��P	pj�M���A�*

	conv_loss�?�<��N        )��P	���M���A�*

	conv_loss��=�Z{        )��P	^֏M���A�*

	conv_loss	=�AS        )��P	��M���A�*

	conv_loss�N(=�b�        )��P	I�M���A�*

	conv_loss轿<pG��        )��P	(y�M���A�*

	conv_loss��*=���        )��P	���M���A�*

	conv_loss$�<,�MO        )��P	ܐM���A�*

	conv_lossэj=&�C        )��P	�M���A�*

	conv_loss�K=�Oe�        )��P	�C�M���A�*

	conv_loss��-=V�,o        )��P	�u�M���A�*

	conv_loss�Y?=�ʋ1        )��P	���M���A�*

	conv_loss�)=u�        )��P		�M���A�*

	conv_loss��(=�T9&        )��P	�G�M���A�*

	conv_lossM�:=U�2�        )��P	P}�M���A�*

	conv_loss.��<1��        )��P	���M���A�*

	conv_lossy��<O�^�        )��P	Y��M���A�*

	conv_loss�i$=z��        )��P	=)�M���A�*

	conv_loss��=#씢        )��P	�W�M���A�*

	conv_loss�p�<�Gw�        )��P	W��M���A�*

	conv_lossw�%=H�        )��P	쿓M���A�*

	conv_loss�K=-�        )��P	���M���A�*

	conv_loss#�=<���        )��P	x/�M���A�*

	conv_loss�=8/        )��P	�a�M���A�*

	conv_loss�<,���        )��P	���M���A�*

	conv_loss'"=����        )��P	dɔM���A�*

	conv_loss8n={R�J        )��P	��M���A�*

	conv_loss��F=+��        )��P	�P�M���A�*

	conv_loss��<���        )��P	���M���A�*

	conv_lossk�<F���        )��P	���M���A�*

	conv_lossښ=��v        )��P	\ߕM���A�*

	conv_loss��%=��g        )��P	��M���A�*

	conv_loss���<+�        )��P	�Q�M���A�*

	conv_lossk'�<N��        )��P	��M���A�*

	conv_loss�0=S3Xy        )��P	���M���A�*

	conv_loss���<fܙ        )��P	���M���A�*

	conv_lossU
=�A        )��P	V2�M���A�*

	conv_loss���<=��        )��P	�c�M���A�*

	conv_loss �"=� }�        )��P	��M���A�*

	conv_lossH�<=J�!        )��P	�ЗM���A�*

	conv_loss�֝<���        )��P	��M���A�*

	conv_loss��W=|x&�        )��P	�=�M���A�*

	conv_loss��7=��2�        )��P	�m�M���A�*

	conv_loss'+�<@��$        )��P	��M���A�*

	conv_loss8�W=_�Mu        )��P	ӘM���A�*

	conv_loss.�=�7)0        )��P	-�M���A�*

	conv_loss�=���        )��P	oF�M���A�*

	conv_loss=@�<��        )��P	7x�M���A�*

	conv_loss�7=���        )��P	���M���A�*

	conv_loss/84=�T��        )��P	PڙM���A�*

	conv_loss�k!=��E�        )��P	��M���A�*

	conv_lossN�"=C&        )��P	�P�M���A�*

	conv_loss1��<�U5�        )��P	���M���A�*

	conv_loss���<�z8        )��P	s��M���A�*

	conv_lossso�<=�\d        )��P	��M���A�*

	conv_lossE�<�l�k        )��P	t�M���A�*

	conv_loss�H1=��l        )��P	�Q�M���A�*

	conv_loss�=�<���q        )��P	��M���A�*

	conv_loss�3=F-*�        )��P	_��M���A�*

	conv_loss��<��        )��P	�M���A�*

	conv_loss��=b�գ        )��P	��M���A�*

	conv_loss�<�QM�        )��P	���M���A�*

	conv_loss��=`d�        )��P	@<�M���A�*

	conv_lossQ�F=�;�        )��P	�l�M���A�*

	conv_lossެD=k��)        )��P	១M���A�*

	conv_lossb=7��        )��P	>ڡM���A�*

	conv_loss��M=��        )��P	$ �M���A�*

	conv_loss�c=&E�H        )��P	�`�M���A�*

	conv_loss7��<��ש        )��P	ُ�M���A�*

	conv_loss�	=��I        )��P	���M���A�*

	conv_lossR�=���        )��P	���M���A�*

	conv_lossd�=�J L        )��P	 /�M���A�*

	conv_loss7�=��        )��P	8r�M���A�*

	conv_loss^�<"�2        )��P	�M���A�*

	conv_loss�=��I�        )��P	�ףM���A�*

	conv_loss�1�<�W5�        )��P	��M���A�*

	conv_loss[��<�_�        )��P	I@�M���A�*

	conv_loss�=	�3        )��P	{x�M���A�*

	conv_loss1<=��y�        )��P	]��M���A�*

	conv_loss\�2=e�5        )��P	#٤M���A�*

	conv_loss��.=�]        )��P	�
�M���A�*

	conv_loss�+�<�|u�        )��P	�<�M���A�*

	conv_loss���<!I?        )��P	�z�M���A�*

	conv_loss�<OY�        )��P	���M���A�*

	conv_loss��<����        )��P	�ޥM���A�*

	conv_loss�
=���6        )��P	N�M���A�*

	conv_loss�7�<��@        )��P	JB�M���A�*

	conv_loss+Y=3RD        )��P	�q�M���A�*

	conv_loss<W=+N5i        )��P	q��M���A�*

	conv_loss�:=$��        )��P	%ۦM���A�*

	conv_loss#=PDK�        )��P	��M���A�*

	conv_lossћ�<�sA�        )��P	vP�M���A�*

	conv_loss�=�[��        )��P	��M���A�*

	conv_loss���<�?�        )��P	���M���A�*

	conv_loss�i�<����        )��P	!�M���A�*

	conv_lossI��<]�y�        )��P	�&�M���A�*

	conv_loss�=p�n        )��P	�Z�M���A�*

	conv_loss�M=/]��        )��P	^��M���A�*

	conv_loss��=\�M'        )��P	̨M���A�*

	conv_lossi�=X�Ȯ        )��P	g��M���A�*

	conv_loss�	�<k��        )��P	Y9�M���A�*

	conv_loss��<���        )��P	4w�M���A�*

	conv_lossE��<�;        )��P	 ��M���A�*

	conv_loss��<VpY        )��P	ݩM���A�*

	conv_loss�E�<~�5�        )��P	��M���A�*

	conv_loss��!=b�hC        )��P	_A�M���A�*

	conv_loss��#=HM�        )��P	w�M���A�*

	conv_loss��<�1	�        )��P	寪M���A�*

	conv_lossC�=@1�W        )��P	�ުM���A�*

	conv_lossgS=y���        )��P	��M���A�*

	conv_lossv�<�N6        )��P	�>�M���A�*

	conv_loss�P=�t�        )��P	�p�M���A�*

	conv_loss�=�6�        )��P	X��M���A�*

	conv_loss�=|��        )��P	���M���A�*

	conv_loss��6=<�        )��P	�+�M���A�*

	conv_loss:�=-��        )��P	�c�M���A�*

	conv_lossY�=>�        )��P	��M���A�*

	conv_loss�c=�zG�        )��P	ѬM���A�*

	conv_lossE7=B/��        )��P	��M���A�*

	conv_loss��S=0x	h        )��P	;5�M���A�*

	conv_lossT�"=I6��        )��P	f�M���A�*

	conv_loss�5)=�KR�        )��P	J��M���A�*

	conv_loss5Y=tg��        )��P	�ϭM���A�*

	conv_loss�+=��`a        )��P	���M���A�*

	conv_loss��<k�        )��P	�.�M���A�*

	conv_loss��-='��        )��P	fc�M���A�*

	conv_lossR0=��        )��P	���M���A�*

	conv_losss%0=�c        )��P	�خM���A�*

	conv_lossb�2=[4��        )��P	5
�M���A�*

	conv_loss��<3���        )��P	o;�M���A�*

	conv_loss�A�<G��p        )��P	Ym�M���A�*

	conv_loss�?=��O�        )��P	=��M���A�*

	conv_lossTX=�㜼        )��P	��M���A�*

	conv_lossM�<X�GX        )��P	��M���A�*

	conv_loss���<b��n        )��P	�D�M���A�*

	conv_loss- �<K��b        )��P	u�M���A�*

	conv_loss:=��        )��P	ë�M���A�*

	conv_loss��=H��        )��P	��M���A�*

	conv_loss��<D�n        )��P	��M���A�*

	conv_loss�+ =�I�k        )��P	�C�M���A�*

	conv_loss��=	Xx�        )��P	�s�M���A�*

	conv_lossM�<��}        )��P	.��M���A�*

	conv_loss~�=_��        )��P	��M���A�*

	conv_lossX=���j        )��P	��M���A�*

	conv_loss$v�<J�        )��P	�E�M���A�*

	conv_lossz= u��        )��P	0w�M���A�*

	conv_loss�@�<2!�        )��P	��M���A�*

	conv_loss�#=�        )��P	�ڲM���A�*

	conv_loss�� =��        )��P	�
�M���A�*

	conv_loss��<�G�u        )��P	�:�M���A�*

	conv_loss��=w��        )��P	�l�M���A�*

	conv_loss}L!=��        )��P	��M���A�*

	conv_loss=A=`Ő<        )��P	ԳM���A�*

	conv_loss��=�{!        )��P	M	�M���A�*

	conv_loss=��<l_5�        )��P	�7�M���A�*

	conv_loss-R=����        )��P	�g�M���A�*

	conv_loss=�6=����        )��P	���M���A�*

	conv_loss~5�<��        )��P	cǴM���A�*

	conv_loss��8=�J?        )��P	��M���A�*

	conv_lossL�%=ȁ��        )��P	`'�M���A�*

	conv_loss�_�<�L        )��P	�V�M���A�*

	conv_lossE�5=g�v        )��P	d��M���A�*

	conv_loss~�E=h�~.        )��P	d�M���A�*

	conv_loss�`!=D�x�        )��P	�F�M���A�*

	conv_loss�O4=�q        )��P	�t�M���A�*

	conv_lossu=OT��        )��P	У�M���A�*

	conv_loss���<�Ml2        )��P	dַM���A�*

	conv_loss�<W|��        )��P	��M���A�*

	conv_loss>l�<�?�        )��P	 B�M���A�*

	conv_loss�*=o�        )��P	�v�M���A�*

	conv_loss�=��l�        )��P	���M���A�*

	conv_loss���<��ͱ        )��P	��M���A�*

	conv_loss���<5G        )��P	^)�M���A�*

	conv_loss��<a�@�        )��P	Y�M���A�*

	conv_loss��=��5�        )��P	��M���A�*

	conv_loss���<8�>        )��P	y��M���A�*

	conv_loss�5�<{��        )��P	�M���A�*

	conv_loss��<Y�        )��P	W�M���A�*

	conv_loss��*=#���        )��P	n@�M���A�*

	conv_loss�#=_ym        )��P	�n�M���A�*

	conv_loss�PC=Bi��        )��P	�M���A�*

	conv_loss���<M�2�        )��P	'ͺM���A�*

	conv_loss 
=�b.:        )��P	�M���A�*

	conv_loss&��<Q��D        )��P	�7�M���A�*

	conv_loss�؏<ƙ�,        )��P	�g�M���A�*

	conv_loss�<R�-�        )��P	���M���A�*

	conv_loss�[�<�S|�        )��P	BȻM���A�*

	conv_loss�	=u40�        )��P	P��M���A�*

	conv_loss�iT=$g��        )��P	�'�M���A�*

	conv_loss���<�=1N        )��P	�W�M���A�*

	conv_loss��=�4L�        )��P	_��M���A�*

	conv_loss��,=b^/�        )��P	��M���A�*

	conv_lossP�<ͭc        )��P	�M���A�*

	conv_loss��<�^��        )��P	x�M���A�*

	conv_loss��)=�wӷ        )��P	mJ�M���A�*

	conv_lossE5=�h|�        )��P	8y�M���A�*

	conv_loss��:=�D��        )��P	���M���A�*

	conv_loss�&=����        )��P	�սM���A�*

	conv_lossD	=S-J+        )��P	��M���A�*

	conv_loss�N�<�o�P        )��P	�2�M���A�*

	conv_lossi=w�_�        )��P	.a�M���A�*

	conv_loss5�=)��,        )��P	Ǒ�M���A�*

	conv_loss/�=ߦ�        )��P	���M���A�*

	conv_loss�~=,DO{        )��P	@�M���A�*

	conv_loss#�-=��?�        )��P	�!�M���A�*

	conv_loss��=��B!        )��P	zR�M���A�*

	conv_loss�<}/U        )��P	���M���A�*

	conv_loss��O=G㋊        )��P	b��M���A�*

	conv_lossx��<��]        )��P	��M���A�*

	conv_lossn-=��]        )��P	��M���A�*

	conv_loss�BX=dϳ�        )��P	�C�M���A�*

	conv_lossQ�<���I        )��P	cs�M���A�*

	conv_lossG��<�LF/        )��P	���M���A�*

	conv_loss�0	= ^�V        )��P	���M���A�*

	conv_loss�%�<n)�        )��P	]�M���A�*

	conv_loss$�0=�y9�        )��P	�C�M���A�*

	conv_loss��3=/        )��P	�w�M���A�*

	conv_loss���<�`�        )��P	ا�M���A�*

	conv_loss�A.=�{p�        )��P	���M���A�*

	conv_lossx�<��*-        )��P	��M���A�*

	conv_lossK�<�T�        )��P	�:�M���A�*

	conv_loss_��<�/@        )��P	�{�M���A�*

	conv_lossb�E=�$��        )��P	+��M���A�*

	conv_lossi��<����        )��P	���M���A�*

	conv_loss��<�{�        )��P	"�M���A�*

	conv_loss�%�<�-�        )��P	Ff�M���A�*

	conv_lossŞ =�
��        )��P	��M���A�*

	conv_loss�H�<�8h�        )��P	���M���A�*

	conv_loss�h�<���        )��P	�/�M���A�*

	conv_loss+�<��]�        )��P	go�M���A�*

	conv_loss��=�G�n        )��P	(��M���A�*

	conv_loss�)=d�9        )��P	b��M���A�*

	conv_loss2��<�{z        )��P	}�M���A�*

	conv_lossO�<B�Y        )��P	�K�M���A�*

	conv_loss�B�<���        )��P	W��M���A�*

	conv_loss�@=�2��        )��P	���M���A�*

	conv_loss��=&��1        )��P	��M���A�*

	conv_loss��=`�!c        )��P	F(�M���A�*

	conv_loss�1=��2|        )��P	�]�M���A�*

	conv_loss��x=��]�        )��P	&��M���A�*

	conv_losst#=��L�        )��P	���M���A�*

	conv_loss��<!���        )��P	u��M���A�*

	conv_lossK9s=��O        )��P	Q-�M���A�*

	conv_loss��=���        )��P	j^�M���A�*

	conv_loss���<W2(�        )��P	���M���A�*

	conv_loss�G�<�i�}        )��P	F��M���A�*

	conv_loss�Y�<��k        )��P	C��M���A�*

	conv_loss��-=�u3        )��P	�*�M���A�*

	conv_loss���<k�"        )��P	xa�M���A�*

	conv_lossD=b�P&        )��P	��M���A�*

	conv_loss�
=P��`        )��P	���M���A�*

	conv_loss���<6"g        )��P	���M���A�*

	conv_loss��<����        )��P	?(�M���A�*

	conv_loss�+�<ir_7        )��P	[�M���A�*

	conv_loss$K�<2�v        )��P	���M���A�*

	conv_loss��T=�ʼ        )��P	$��M���A�*

	conv_loss`�=')�}        )��P	I��M���A�*

	conv_lossB>=2I��        )��P	�4�M���A�*

	conv_lossT�<O��        )��P	v�M���A�*

	conv_loss��+=x�G�        )��P	ϰ�M���A�*

	conv_loss�<=$�R�        )��P	���M���A�*

	conv_loss��=2��Y        )��P	�M���A�*

	conv_loss�=�_�2        )��P	T�M���A�*

	conv_loss���<X;Dh        )��P	���M���A�*

	conv_loss:#I=X�{        )��P	|��M���A�*

	conv_lossf�8=�v1        )��P	���M���A�*

	conv_loss���<��9`        )��P	)�M���A�*

	conv_loss�� =��!�        )��P	�[�M���A�*

	conv_loss3�=�&R        )��P	A��M���A�*

	conv_lossWgs<Cjů        )��P	V��M���A�*

	conv_loss�<���x        )��P	��M���A�*

	conv_loss���<݅        )��P	�=�M���A�*

	conv_lossu��<��-        )��P	Nm�M���A�*

	conv_loss��<��        )��P	���M���A�*

	conv_loss�h�<��1G        )��P	.��M���A�*

	conv_lossC��<��        )��P	
�M���A�*

	conv_loss���<�;v        )��P	%:�M���A�*

	conv_loss֑�<��TJ        )��P	�m�M���A�*

	conv_loss��==���e        )��P	��M���A�*

	conv_lossV=mEL        )��P	���M���A�*

	conv_loss��$=��G\        )��P	��M���A�*

	conv_loss��<�>�#        )��P	aG�M���A�*

	conv_lossNw!=_Ίt        )��P	T��M���A�*

	conv_losse�<��D        )��P	���M���A�*

	conv_loss<J=䦙�        )��P	���M���A�*

	conv_loss���<2�K        )��P	��M���A�*

	conv_lossi��<�VB        )��P	J�M���A�*

	conv_loss0�	=��3_        )��P	4|�M���A�*

	conv_losssb'=����        )��P	9��M���A�*

	conv_lossH�)=5{ݣ        )��P	��M���A�*

	conv_loss���<�FnX        )��P	��M���A�*

	conv_loss�|�<��^        )��P	ER�M���A�*

	conv_loss�-=���        )��P	��M���A�*

	conv_lossa?=%*�        )��P	J��M���A�*

	conv_loss�0�<-R:�        )��P	��M���A�*

	conv_loss��<U�h        )��P	q8�M���A�*

	conv_loss��&=;q�%        )��P	i�M���A�*

	conv_loss�u=<q�        )��P	Ƙ�M���A�*

	conv_lossen?<i5��        )��P	���M���A�*

	conv_loss-��<j��        )��P	z�M���A�*

	conv_loss8�=ۗ��        )��P	L7�M���A�*

	conv_loss�%=��.        )��P	Ch�M���A�*

	conv_loss���<�ݷ        )��P	k��M���A�*

	conv_loss-=�        )��P	��M���A�*

	conv_loss��=8�`        )��P	���M���A�*

	conv_loss��<�$��        )��P	�2�M���A�*

	conv_lossP|=)3��        )��P	k�M���A�*

	conv_loss4�\=}�HA        )��P	ա�M���A�*

	conv_lossؒ�<�$�        )��P	���M���A�*

	conv_lossP0(=ST        )��P	� �M���A�*

	conv_loss��j=>��        )��P	�7�M���A�*

	conv_loss�n	=��G        )��P	�l�M���A�*

	conv_lossi��<���        )��P	 ��M���A�*

	conv_loss6(�<���h        )��P	���M���A�*

	conv_loss=L	=��P?        )��P	:�M���A�*

	conv_loss�6�<��        )��P	BQ�M���A�*

	conv_loss;�=�Z3        )��P	��M���A�*

	conv_loss�G=xpv        )��P	<��M���A�*

	conv_loss&@=<�yK�        )��P	1��M���A�*

	conv_lossܱ3=���3        )��P	h-�M���A�*

	conv_loss6��<O���        )��P	h`�M���A�*

	conv_loss#>=zP�3        )��P	��M���A�*

	conv_loss�<+�j�        )��P	���M���A�*

	conv_loss^�=rj&�        )��P	h��M���A�*

	conv_loss��=%s�0        )��P	L)�M���A�*

	conv_loss��=��m|        )��P	�\�M���A�*

	conv_loss��=FBl�        )��P	J��M���A�*

	conv_loss���<���q        )��P	���M���A�*

	conv_loss-=Tt�l        )��P	e��M���A�*

	conv_loss�2B=$��        )��P	z)�M���A�*

	conv_loss�I�<��t[        )��P	`�M���A�*

	conv_loss1M�<����        )��P	ǚ�M���A�*

	conv_loss���<ՄǤ        )��P	o��M���A�*

	conv_loss�9�<u\E�        )��P	���M���A�*

	conv_lossB�=����        )��P	T5�M���A�*

	conv_lossm�=(^O�        )��P	ze�M���A�*

	conv_lossj	"=8k�        )��P	D��M���A�*

	conv_loss�={�d         )��P	���M���A�*

	conv_lossge=0���        )��P	� �M���A�*

	conv_loss��=B_        )��P	�<�M���A�*

	conv_loss#r=߂�        )��P	Bn�M���A�*

	conv_lossq��<����        )��P	۠�M���A�*

	conv_losszG�<��        )��P		��M���A�*

	conv_loss ��<��        )��P	�M���A�*

	conv_loss|{==x�        )��P	>�M���A�*

	conv_loss��<F-        )��P	�l�M���A�*

	conv_loss	�=Ŭ�/        )��P	���M���A�*

	conv_loss��<>[�m        )��P	���M���A�*

	conv_loss>?�=J�        )��P	u�M���A�*

	conv_loss��<��T        )��P	!6�M���A�*

	conv_loss��=N�&�        )��P	�i�M���A�*

	conv_loss�j=7���        )��P	���M���A�*

	conv_loss~$X=Cݪ        )��P	���M���A�*

	conv_loss>�<g���        )��P	\�M���A�*

	conv_lossȑ=�-^�        )��P	�D�M���A�*

	conv_loss�=��M�        )��P	-��M���A�*

	conv_loss��&=1�n        )��P	���M���A�*

	conv_loss=sʃ        )��P	w��M���A�*

	conv_loss%�<,^<        )��P	S�M���A�*

	conv_lossj>�<�<        )��P	qP�M���A�*

	conv_loss�=�%Z�        )��P	���M���A�*

	conv_loss�#=_�M        )��P	���M���A�*

	conv_loss���<$D�?        )��P	�g�M���A�*

	conv_loss��1=]CM        )��P	���M���A�*

	conv_loss�(�<�/�        )��P	{��M���A�*

	conv_loss&ph=��g        )��P	��M���A�*

	conv_loss���<��e        )��P	�:�M���A�*

	conv_loss�=NQ�        )��P	z�M���A�*

	conv_loss��F=v"�        )��P	���M���A�*

	conv_loss��=b|F        )��P	��M���A�*

	conv_loss�/=��0�        )��P	��M���A�*

	conv_loss;�=hCWL        )��P	�G�M���A�*

	conv_losst�F=\�ە        )��P	���M���A�*

	conv_loss�]�<�@t        )��P	���M���A�*

	conv_loss��=E���        )��P	��M���A�*

	conv_loss� �<���        )��P	v=�M���A�*

	conv_loss�K"=�ps        )��P	Qp�M���A�*

	conv_lossc1=�m�r        )��P	V��M���A�*

	conv_loss:��<ûm�        )��P	4��M���A�*

	conv_loss�_=�p�         )��P	��M���A�*

	conv_lossi��<���        )��P	 3�M���A�*

	conv_loss-��<h��/        )��P	�c�M���A�*

	conv_loss�(q<�#"        )��P	��M���A�*

	conv_lossV=�\[        )��P	���M���A�*

	conv_lossJ��<uÔ        )��P	��M���A�*

	conv_loss�5>=�t��        )��P	�9�M���A�*

	conv_loss��<		        )��P	il�M���A�*

	conv_losso�<L�T3        )��P	S��M���A�*

	conv_loss���<��o�        )��P	���M���A�*

	conv_loss���<�CV:        )��P	K	�M���A�*

	conv_loss{��<,xqi        )��P	;�M���A�*

	conv_loss �=�4G�        )��P	�m�M���A�*

	conv_loss��=���4        )��P	ؤ�M���A�*

	conv_lossF�=� x�        )��P	v��M���A�*

	conv_loss/�'=}?˃        )��P	A�M���A�*

	conv_loss�N�<X���        )��P	�?�M���A�*

	conv_loss�:=a�U.        )��P	�z�M���A�*

	conv_loss��=�9��        )��P	��M���A�*

	conv_loss�	=����        )��P	v��M���A�*

	conv_loss�'�<a4+�        )��P	m�M���A�*

	conv_loss��<���
        )��P	kK�M���A�*

	conv_lossZձ<�i��        )��P	À�M���A�*

	conv_loss0 �<x��        )��P	]��M���A�*

	conv_loss�ZS=V�{�        )��P	3��M���A�*

	conv_loss� =�	}        )��P	��M���A�*

	conv_loss\OB=ir        )��P	QK�M���A�*

	conv_lossk�< !%        )��P	{�M���A�*

	conv_loss�nn=49��        )��P	^��M���A�*

	conv_loss��=���2        )��P	O��M���A�*

	conv_loss&,=�NU�        )��P	u�M���A�*

	conv_lossvE=�dW�        )��P	XS�M���A�*

	conv_loss'��<�d8�        )��P	)��M���A�*

	conv_loss��<��S        )��P	{��M���A�*

	conv_loss��<��        )��P	L�M���A�*

	conv_loss��<˝)        )��P	�4�M���A�*

	conv_loss�7=�h=�        )��P	�i�M���A�*

	conv_loss6=b��        )��P	l��M���A�*

	conv_loss�;&=�	Շ        )��P	S��M���A�*

	conv_lossM�<k�_        )��P	��M���A�*

	conv_loss��-=�k        )��P	LI�M���A�*

	conv_loss&e�<>��        )��P	Fy�M���A�*

	conv_loss��/=�K        )��P	���M���A�*

	conv_loss���<"��        )��P	���M���A�*

	conv_loss��<=��g#        )��P	C�M���A�*

	conv_lossw�1=���        )��P	�O�M���A�*

	conv_loss��6=&���        )��P	8��M���A�*

	conv_lossԾ}<��vC        )��P	+��M���A�*

	conv_losslf)=�ϏB        )��P	>��M���A�*

	conv_losseB�<����        )��P	��M���A�*

	conv_loss�
�<=�        )��P	�L�M���A�*

	conv_lossT�3=���        )��P	���M���A�*

	conv_lossp?�<G��        )��P	´�M���A�*

	conv_loss�k@=����        )��P	���M���A�*

	conv_lossĦ*=tQU�        )��P	��M���A�*

	conv_losss��<��~�        )��P	�V�M���A�*

	conv_losssb"= ,        )��P	��M���A�*

	conv_loss�+=�B{        )��P	��M���A�*

	conv_loss���<B ��        )��P	��M���A�*

	conv_lossf�0=Q��'        )��P	�/�M���A�*

	conv_loss�Y�<U�j�        )��P	�a�M���A�*

	conv_loss�=��        )��P	ٓ�M���A�*

	conv_lossز<�:'        )��P	[��M���A�*

	conv_loss�>=$x�        )��P	��M���A�*

	conv_loss��O=�/5        )��P	�:�M���A�*

	conv_loss~�!=3RK        )��P	$l�M���A�*

	conv_loss��z<{W        )��P	��M���A�*

	conv_loss���<��8        )��P	���M���A�*

	conv_loss���<<�L�        )��P	��M���A�*

	conv_loss4��<�`��        )��P	�K�M���A�*

	conv_loss�($=�<:        )��P	�}�M���A�*

	conv_lossj��<1��_        )��P	F��M���A�*

	conv_lossj!k=Ȓ��        )��P	k��M���A�*

	conv_lossy��<���        )��P	�"�M���A�*

	conv_loss"��<���v        )��P	�U�M���A�*

	conv_loss6��<�P~        )��P	���M���A�*

	conv_lossP�=�
}�        )��P	l��M���A�*

	conv_loss�=�F�        )��P	z��M���A�*

	conv_lossfo�<柞        )��P	+�M���A�*

	conv_lossl�=Zڱ;        )��P	�j�M���A�*

	conv_loss�`=�*�	        )��P	}��M���A�*

	conv_loss�t�<�SG        )��P	���M���A�*

	conv_loss*Y�< ━        )��P	3�M���A�*

	conv_loss6�a=��O�        )��P	^f�M���A�*

	conv_loss7�=Q���        )��P	ݗ�M���A�*

	conv_lossk�=\B�        )��P	��M���A�*

	conv_loss�\?=�m��        )��P	���M���A�*

	conv_loss��<�Z�        )��P	�5�M���A�*

	conv_loss���<��        )��P	k�M���A�*

	conv_loss�=��"�        )��P	���M���A�*

	conv_loss��=P��W        )��P	���M���A�*

	conv_losscn�<���        )��P	6�M���A�*

	conv_lossD=�?�        )��P	�H�M���A�*

	conv_loss�4`=��h        )��P	w��M���A�*

	conv_loss��<g��        )��P	|��M���A�*

	conv_loss6ʼ<�.        )��P	l��M���A�*

	conv_loss�=iP�        )��P	�1�M���A�*

	conv_loss�L�<���I        )��P	�d�M���A�*

	conv_lossϒ�<ȝ]X        )��P	ӗ�M���A�*

	conv_loss}r�<� �y        )��P	���M���A�*

	conv_loss U<��Q        )��P	S��M���A�*

	conv_loss�� = �"M        )��P	�%�M���A�*

	conv_loss�'�<����        )��P	�]�M���A�*

	conv_loss&�=��F�        )��P	*��M���A�*

	conv_lossE�<��-        )��P	���M���A�*

	conv_lossp��<�ZaY        )��P	b��M���A�*

	conv_loss��<4���        )��P	�9�M���A�*

	conv_loss"�j=#��k        )��P	k�M���A�*

	conv_loss��<���z        )��P	��M���A�*

	conv_loss,�<�k�        )��P	T��M���A�*

	conv_loss��<�o        )��P	!	�M���A�*

	conv_loss���<ž�2        )��P	�H�M���A�*

	conv_loss���<Xl{        )��P	�y�M���A�*

	conv_loss%z<�9~        )��P	���M���A�*

	conv_loss��p=9��V        )��P	��M���A�*

	conv_loss�=[�,        )��P	7!�M���A�*

	conv_loss�G=T�!�        )��P	WZ�M���A�*

	conv_lossu��<̡�"        )��P	���M���A�*

	conv_loss�n�<�        )��P	R��M���A�*

	conv_loss8�=~��"        )��P	���M���A�*

	conv_loss�_�<NF.�        )��P	�H�M���A�*

	conv_loss�F=��        )��P	(|�M���A�*

	conv_loss�ZU=�D��        )��P	��M���A�*

	conv_loss(�>=�zme        )��P	��M���A�*

	conv_lossv�<��m        )��P	�M���A�*

	conv_loss��4=:x�6        )��P	�d�M���A�*

	conv_loss�=��n        )��P	��M���A�*

	conv_loss�w�<`��Z        )��P	���M���A�*

	conv_loss��=å        )��P	� N���A�*

	conv_lossJ��<�؃�        )��P	|n N���A�*

	conv_loss�T�<W�W]        )��P	ǡ N���A�*

	conv_loss�0=u���        )��P	�� N���A�*

	conv_loss�=~���        )��P	�N���A�*

	conv_loss{G�<Q	��        )��P	�VN���A�*

	conv_loss��C=��w        )��P	U�N���A�*

	conv_loss�cA=��ݟ        )��P	�N���A�*

	conv_loss�S7=HN-        )��P	�N���A�*

	conv_loss�0=�yR�        )��P	0@N���A�*

	conv_lossh�<�[��        )��P	A{N���A�*

	conv_lossղ�=�)�=        )��P	j�N���A�*

	conv_loss!�=�c�9        )��P	��N���A�*

	conv_lossU��<&�Fd        )��P	>N���A�*

	conv_loss�t=��}D        )��P	�FN���A�*

	conv_loss�	b=��        )��P	��N���A�*

	conv_loss�=+v�U        )��P	T�N���A�*

	conv_loss�V	==MN~        )��P	��N���A�*

	conv_lossJ�=���        )��P	�"N���A�*

	conv_loss�D�<���        )��P	�XN���A�*

	conv_loss%a=6��!        )��P	��N���A�*

	conv_loss�%�<��u        )��P	r�N���A�*

	conv_loss�F=��c        )��P	0�N���A�*

	conv_loss�*�<�PR�        )��P	 2N���A�*

	conv_loss�R=_�        )��P	�eN���A�*

	conv_lossL�=5���        )��P	f�N���A�*

	conv_loss�N�<�F�i        )��P	��N���A�*

	conv_lossz�	=�n�G        )��P	E N���A�*

	conv_lossޜ=�|ma        )��P	�/N���A�*

	conv_loss*u=/C        )��P	�dN���A�*

	conv_loss"Ux<*�s�        )��P	�N���A�*

	conv_loss��<�sع        )��P	M�N���A�*

	conv_loss�C�<=~��        )��P	�N���A�*

	conv_loss���<gE�        )��P	U=N���A�*

	conv_loss�~�<�p�        )��P	BwN���A�*

	conv_loss��=��        )��P	��N���A�*

	conv_loss�{ =�E�        )��P	��N���A�*

	conv_loss|e=i���        )��P	�N���A�*

	conv_lossT�<�,�[        )��P	�NN���A�*

	conv_loss��=0�V        )��P	,�N���A�*

	conv_loss���<�s̶        )��P	,�N���A�*

	conv_loss��2=m���        )��P	9�N���A�*

	conv_lossX�h=�*��        )��P	2#	N���A�*

	conv_loss�+U=a8�)        )��P	�U	N���A�*

	conv_loss��<բH        )��P	T�	N���A�*

	conv_loss�B"=� _        )��P	O�	N���A�*

	conv_loss��<�ﵘ        )��P	��	N���A�*

	conv_loss��2=C�s�        )��P	@.
N���A�*

	conv_losso�D=C��V        )��P	�\
N���A�*

	conv_loss#��<��,�        )��P	�
N���A�*

	conv_lossgQ=�(m        )��P	�
N���A�*

	conv_loss?��<��0        )��P	�
N���A�*

	conv_loss��
=�g�Y        )��P	�+N���A�*

	conv_loss=�8w�        )��P	_[N���A�*

	conv_lossl�<B��        )��P	��N���A�*

	conv_lossѬ=��Dl        )��P	�RN���A�*

	conv_loss!t<^l�        )��P	��N���A�*

	conv_loss箺<k�r�        )��P	��N���A�*

	conv_loss���<��e        )��P	��N���A�*

	conv_loss�<xTe        )��P	�+N���A�*

	conv_lossT�o=΢�        )��P	S_N���A�*

	conv_loss%�=�=�Q        )��P	j�N���A�*

	conv_lossr[�<76r<        )��P	��N���A�*

	conv_loss�N&=/t:�        )��P	#�N���A�*

	conv_loss��<�V        )��P	�7N���A�*

	conv_loss���<�
8�        )��P	jN���A�*

	conv_lossw��<`���        )��P	��N���A�*

	conv_loss$�<uk��        )��P	��N���A�*

	conv_loss�|e=9&7<        )��P	WN���A�*

	conv_loss��<Z��P        )��P	zQN���A�*

	conv_loss�;/=�U��        )��P	��N���A�*

	conv_lossQ0?=�
=        )��P	��N���A�*

	conv_lossb��<ght�        )��P	��N���A�*

	conv_losse-�<ZF _        )��P	$&N���A�*

	conv_lossx�C=-��        )��P	�eN���A�*

	conv_loss�<E�F�        )��P	��N���A�*

	conv_loss�2�<	:m�        )��P	]�N���A�*

	conv_loss?٨<'4�        )��P		N���A�*

	conv_lossr�+=v���        )��P	�VN���A�*

	conv_loss�ׄ<8X{i        )��P	�N���A�*

	conv_lossˉ�<`R�g        )��P	.�N���A�*

	conv_loss ��<��<o        )��P	g�N���A�*

	conv_loss"^ =�
�        )��P	"N���A�*

	conv_loss�=�<�w        )��P	3bN���A�*

	conv_loss�D�<�y*�        )��P	�N���A�*

	conv_loss1�<R��        )��P	K�N���A�*

	conv_loss�1=��        )��P	8�N���A�*

	conv_lossV�#=4!5�        )��P	1N���A�*

	conv_loss��-=w7i        )��P	PyN���A�*

	conv_lossJf4=B�>�        )��P	��N���A�*

	conv_loss2�<�P�w        )��P	��N���A�*

	conv_loss:�=ay:�        )��P	�KN���A�*

	conv_lossԼ�<P*        )��P	��N���A�*

	conv_lossV�<�         )��P	��N���A�*

	conv_loss���<t��6        )��P	��N���A�*

	conv_loss���<zv�        )��P	N���A�*

	conv_loss�s/=E��        )��P		QN���A�*

	conv_loss��C= ��Z        )��P	��N���A�*

	conv_loss�h�<TB�        )��P	V�N���A�*

	conv_loss�b	=&B�h        )��P	&�N���A�*

	conv_lossg
�<
(��        )��P	�)N���A�*

	conv_loss��=H��z        )��P		aN���A�*

	conv_lossQ��<*�o�        )��P	ۗN���A�*

	conv_loss$?�<"m�.        )��P	��N���A�*

	conv_loss"VA=����        )��P	�N���A�*

	conv_loss��7==�@�        )��P	HN���A�*

	conv_lossV�$=�rT�        )��P	H�N���A�*

	conv_loss���<��/        )��P	�N���A�*

	conv_loss�{<={毤        )��P	�N���A�*

	conv_loss�l!=cH�f        )��P	@ N���A�*

	conv_lossqL=q��p        )��P	DSN���A�*

	conv_loss'�<k(�        )��P	�N���A�*

	conv_loss��M="��t        )��P	|�N���A�*

	conv_loss��=}\Rx        )��P	�	N���A�*

	conv_loss��=Uإ        )��P	�>N���A�*

	conv_loss�"�<�:�?        )��P	pN���A�*

	conv_lossef�<<��        )��P	'�N���A�*

	conv_loss���<Yh�        )��P	D�N���A�*

	conv_loss� �<��.        )��P	�N���A�*

	conv_lossA�='�        )��P	�GN���A�*

	conv_lossza�<���%        )��P	{�N���A�*

	conv_loss�=jm�        )��P	��N���A�*

	conv_loss�&=^>�        )��P	��N���A�*

	conv_loss��<��4�        )��P	o9 N���A�*

	conv_loss%��<=Q�        )��P	&k N���A�*

	conv_loss��'=Iq�o        )��P	�� N���A�*

	conv_loss��<��*�        )��P	�� N���A�*

	conv_loss|ש<�m�"        )��P	 !N���A�*

	conv_loss���<Z�x�        )��P	_J!N���A�*

	conv_loss6�%=b��I        )��P	&|!N���A�*

	conv_loss���<��(�        )��P	ճ!N���A�*

	conv_loss�e<��za        )��P	1�!N���A�*

	conv_loss�#�<�Pv        )��P	""N���A�*

	conv_loss��<2ү�        )��P	$T"N���A�*

	conv_loss��=S���        )��P	ń"N���A�*

	conv_loss��=��        )��P	?�"N���A�*

	conv_loss6_=���t        )��P	K�"N���A�*

	conv_lossם�< �.�        )��P	s6#N���A�*

	conv_loss4��<��05        )��P	�i#N���A�*

	conv_lossG�&=O��8        )��P	��#N���A�*

	conv_loss�0=����        )��P	\�#N���A�*

	conv_loss =1x��        )��P	?$N���A�*

	conv_loss_5=8��        )��P	)M$N���A�*

	conv_loss.L�<�v�B        )��P	?$N���A�*

	conv_loss@n=���        )��P	��$N���A�*

	conv_loss���<��3b        )��P	g�$N���A�*

	conv_lossY��<�d�        )��P	~ %N���A�*

	conv_lossF	=ƒ(�        )��P	E_%N���A�*

	conv_lossޢ�<{'�        )��P	3�%N���A�*

	conv_loss�T=����        )��P	'�%N���A�*

	conv_lossu�=H��&        )��P	� &N���A� *

	conv_loss~/�<P��        )��P	�9&N���A� *

	conv_loss��<�T�N        )��P	�s&N���A� *

	conv_lossϦ!=s�ae        )��P	�&N���A� *

	conv_loss�n(=&�N        )��P	��&N���A� *

	conv_lossFm#=5\�`        )��P	!'N���A� *

	conv_loss<ݶ<�}��        )��P	�S'N���A� *

	conv_loss�h+='���        )��P	��'N���A� *

	conv_loss��=���        )��P	�'N���A� *

	conv_loss=�=ۦ��        )��P	9(N���A� *

	conv_loss�=�@        )��P	+6(N���A� *

	conv_lossC=i)ܱ        )��P	�l(N���A� *

	conv_losss��</�\        )��P	��(N���A� *

	conv_loss:&�<'k|�        )��P	�(N���A� *

	conv_losso��<���        )��P	)N���A� *

	conv_loss��<��1�        )��P	�K)N���A� *

	conv_lossr�<(NO        )��P	'~)N���A� *

	conv_loss�=��        )��P	ɱ)N���A� *

	conv_loss��=�Ȅ.        )��P	A�)N���A� *

	conv_loss!�=>y�p        )��P	�*N���A� *

	conv_lossPS5=8EC�        )��P	�W*N���A� *

	conv_loss�D=4�Ӽ        )��P	>�*N���A� *

	conv_lossW�#=ie8�        )��P	��*N���A� *

	conv_lossyٲ<%X�        )��P	��*N���A� *

	conv_loss��=�v��        )��P	�1+N���A� *

	conv_loss,�=��l        )��P	"d+N���A� *

	conv_lossc+=b'��        )��P	5�+N���A� *

	conv_loss1��<c���        )��P	r�+N���A� *

	conv_loss�1�<%�        )��P	,N���A� *

	conv_loss	��</K�        )��P	�2,N���A� *

	conv_loss���<E�V�        )��P	8c,N���A� *

	conv_loss���<n��        )��P	֛,N���A� *

	conv_loss8��<�
        )��P	��,N���A� *

	conv_loss���<#+t}        )��P	<#-N���A� *

	conv_losswl=�˰A        )��P	`T-N���A� *

	conv_loss>�$=`�P        )��P	��-N���A� *

	conv_lossѓ�<_�q3        )��P	y�-N���A� *

	conv_loss��!=���        )��P	O�-N���A� *

	conv_loss�e=лi        )��P	f&.N���A� *

	conv_loss�=`rt�        )��P	�Y.N���A� *

	conv_loss�<�<|�i        )��P	\�.N���A� *

	conv_loss�B=�f{w        )��P	~�.N���A� *

	conv_lossmp=:���        )��P	 �.N���A� *

	conv_lossx�=��Zm        )��P	4/N���A� *

	conv_loss���<�,��        )��P	�q/N���A� *

	conv_loss���<d6�        )��P	�/N���A� *

	conv_lossg =�!��        )��P	�/N���A� *

	conv_lossCȲ<U�7        )��P	,0N���A� *

	conv_loss�?�<*�iV        )��P	0P0N���A� *

	conv_loss� =���m        )��P	��0N���A� *

	conv_loss��C=,Z�D        )��P	�0N���A� *

	conv_loss�޴<��        )��P	��0N���A� *

	conv_loss��=_"�        )��P	/&1N���A� *

	conv_loss�s=��/	        )��P	�W1N���A� *

	conv_loss�+=Lw�        )��P	�1N���A� *

	conv_loss��@=���)        )��P	]�1N���A� *

	conv_loss���<�?        )��P	{"2N���A� *

	conv_loss-Ȫ<�+T3        )��P	�Z2N���A� *

	conv_loss%H=Si5�        )��P	��2N���A� *

	conv_loss��;=5~��        )��P	��2N���A� *

	conv_loss�f�<�Wܴ        )��P	��2N���A� *

	conv_losszU=��>�        )��P	[=3N���A� *

	conv_loss��=J!ba        )��P	*u3N���A� *

	conv_loss�]=4R��        )��P	W�3N���A� *

	conv_loss�=���        )��P	5�3N���A� *

	conv_lossM�<�r�        )��P	�4N���A� *

	conv_loss"F�<M��\        )��P	�S4N���A� *

	conv_loss'��<E�6        )��P	�4N���A� *

	conv_lossTu�<���z        )��P	:�4N���A� *

	conv_losslݜ<K��_        )��P	 �4N���A� *

	conv_loss��=-�Ϋ        )��P	�>5N���A� *

	conv_loss���<-��        )��P	�s5N���A� *

	conv_lossn�=j��        )��P	�5N���A� *

	conv_loss�c�<�1+H        )��P	��5N���A� *

	conv_loss��<�_�.        )��P	6N���A� *

	conv_loss�)�<,�@�        )��P	�I6N���A� *

	conv_loss���<{��        )��P	6N���A� *

	conv_loss!�<����        )��P	��6N���A� *

	conv_loss~Q�<�6Ru        )��P	��6N���A� *

	conv_loss�=�m�        )��P	�'7N���A� *

	conv_lossўU=Cx�        )��P	s]7N���A� *

	conv_lossR�"=3_��        )��P	P�7N���A� *

	conv_loss��1=����        )��P	��7N���A� *

	conv_loss���<(�!        )��P	f8N���A� *

	conv_loss��<��2        )��P	H48N���A� *

	conv_lossI�1=/E8�        )��P	�k8N���A� *

	conv_loss�k=!j.1        )��P	��8N���A� *

	conv_loss7~ =�p�o        )��P	�8N���A� *

	conv_loss!
=��i        )��P	�9N���A� *

	conv_lossQ�T=Y.9        )��P	r79N���A� *

	conv_lossl�=���        )��P	�r9N���A� *

	conv_losss�C=�4�&        )��P	+�9N���A� *

	conv_loss�H�<��~}        )��P	>�9N���A� *

	conv_lossw��<҅�q        )��P	9:N���A� *

	conv_loss �<���%        )��P	2Q:N���A� *

	conv_loss��<��        )��P	S:N���A� *

	conv_loss��<�OD�        )��P	�:N���A� *

	conv_loss4<�<{Fd�        )��P	��:N���A� *

	conv_loss�H�<�W��        )��P	~;N���A� *

	conv_lossK��<�s�B        )��P	�;;N���A� *

	conv_loss�^�<V��        )��P	�l;N���A� *

	conv_loss��=��x�        )��P	��;N���A� *

	conv_loss��=�P-�        )��P	F�;N���A� *

	conv_loss��<�].>        )��P	M<N���A� *

	conv_loss�=� �        )��P	�B<N���A� *

	conv_losse� =𸸄        )��P	��=N���A� *

	conv_loss)@�<�/��        )��P	�>N���A� *

	conv_loss�k<���        )��P	$J>N���A� *

	conv_loss!�=���V        )��P	��>N���A� *

	conv_loss]�f<F��?        )��P	 �>N���A� *

	conv_loss�V�<~��2        )��P	K�>N���A� *

	conv_loss-�<U��         )��P	?N���A� *

	conv_loss�=+s��        )��P	�O?N���A� *

	conv_lossc?�<���        )��P	Ԕ?N���A� *

	conv_loss�U=���        )��P	��?N���A� *

	conv_loss_�J=�Ş�        )��P	��?N���A� *

	conv_loss�)=�        )��P	�,@N���A� *

	conv_loss�-=���        )��P	\@N���A� *

	conv_lossI��<O��        )��P	˖@N���A� *

	conv_loss��=߂2�        )��P	��@N���A� *

	conv_lossVs�<���        )��P	� AN���A� *

	conv_lossg=r7�5        )��P	]<AN���A� *

	conv_loss��<���        )��P	woAN���A� *

	conv_loss<�<>�Y�        )��P	d�AN���A� *

	conv_loss=��<ܿK        )��P	��AN���A� *

	conv_loss��<�Z'        )��P	ABN���A� *

	conv_loss�nT<�u\&        )��P	?BN���A� *

	conv_loss!�=�.\�        )��P	RtBN���A� *

	conv_loss�d�<��L�        )��P	S�BN���A�!*

	conv_loss���<�n�	        )��P	��BN���A�!*

	conv_losso!=pw!        )��P	�CN���A�!*

	conv_loss`�=�<t�        )��P	�JCN���A�!*

	conv_loss���<ؾ��        )��P	�CN���A�!*

	conv_loss$�=�E>        )��P	��CN���A�!*

	conv_loss�#=N�m$        )��P	��CN���A�!*

	conv_losska�<���        )��P	�6DN���A�!*

	conv_lossꖴ<��N�        )��P	phDN���A�!*

	conv_lossM=)�(o        )��P	m�DN���A�!*

	conv_loss/��<�D        )��P	��DN���A�!*

	conv_loss���<��        )��P	�	EN���A�!*

	conv_loss!f�<����        )��P	Q=EN���A�!*

	conv_lossh��<��        )��P	#qEN���A�!*

	conv_loss��<�_�        )��P	��EN���A�!*

	conv_loss��<�:�<        )��P	��EN���A�!*

	conv_loss�V�<.�t        )��P	oFN���A�!*

	conv_loss���<�gK        )��P	{_FN���A�!*

	conv_loss��=�
^�        )��P	�FN���A�!*

	conv_loss� =ك[        )��P	H�FN���A�!*

	conv_loss�H2=s�p        )��P	?�FN���A�!*

	conv_loss2�=2�b        )��P	�#GN���A�!*

	conv_lossd$=��        )��P	�ZGN���A�!*

	conv_loss�ߧ<��        )��P	��GN���A�!*

	conv_loss׹�<���        )��P	��GN���A�!*

	conv_lossc	=���        )��P	L HN���A�!*

	conv_loss,�<�{�9        )��P	�3HN���A�!*

	conv_loss��<lsTH        )��P	�vHN���A�!*

	conv_loss|c�<�cP�        )��P	3�HN���A�!*

	conv_loss3�<�W�        )��P		�HN���A�!*

	conv_loss5�?<pH2        )��P	�IN���A�!*

	conv_loss:j@=���        )��P	nGIN���A�!*

	conv_lossXx,=h��        )��P	=xIN���A�!*

	conv_lossZ�=>Ք�        )��P	��IN���A�!*

	conv_loss���<-YJ�        )��P	R�IN���A�!*

	conv_loss�Q�<Ozp�        )��P	>JN���A�!*

	conv_loss�8=G���        )��P	�JJN���A�!*

	conv_loss]��<�6��        )��P	��JN���A�!*

	conv_loss�.=0��        )��P	H�JN���A�!*

	conv_lossGʳ<a�k�        )��P	�JN���A�!*

	conv_loss^\�<GF��        )��P	>!KN���A�!*

	conv_loss�q=˼,�        )��P	�XKN���A�!*

	conv_loss�+�<!~�        )��P	��KN���A�!*

	conv_lossO?�<*�(�        )��P	��KN���A�!*

	conv_lossP�=���        )��P	^�KN���A�!*

	conv_lossw�<U�[_        )��P	�(LN���A�!*

	conv_loss݈<�e��        )��P	�ZLN���A�!*

	conv_loss?3�<����        )��P	��LN���A�!*

	conv_loss�e�=+>5�        )��P	��LN���A�!*

	conv_loss׆h=�{�&        )��P	��LN���A�!*

	conv_loss?��<-�.�        )��P	�)MN���A�!*

	conv_lossF��<��Ք        )��P	�\MN���A�!*

	conv_loss���<�=�R        )��P	��MN���A�!*

	conv_loss���<��N        )��P	��MN���A�!*

	conv_lossK��<�KI�        )��P	�NN���A�!*

	conv_loss�{A=��        )��P	n6NN���A�!*

	conv_lossO�<��U        )��P	dNN���A�!*

	conv_loss�i�<g�d        )��P	˒NN���A�!*

	conv_loss��=���&        )��P	��NN���A�!*

	conv_loss��=��w�        )��P	-�NN���A�!*

	conv_loss��<8��K        )��P	�1ON���A�!*

	conv_lossU�=�N��        )��P	�cON���A�!*

	conv_loss�c�<@�0!        )��P	"�ON���A�!*

	conv_loss;�<㑰        )��P	W�ON���A�!*

	conv_loss��<Ռ.        )��P	T�ON���A�!*

	conv_loss�ӧ<�(        )��P	�/PN���A�!*

	conv_loss&ӎ<���Y        )��P	8nPN���A�!*

	conv_loss��'=��        )��P	�PN���A�!*

	conv_loss��=0S��        )��P	�PN���A�!*

	conv_loss�`p<��        )��P	cQN���A�!*

	conv_loss�[�<�J��        )��P	�BQN���A�!*

	conv_loss�m�<*W�z        )��P	DvQN���A�!*

	conv_loss���<w�R        )��P	|�QN���A�!*

	conv_loss��=:0��        )��P	��QN���A�!*

	conv_loss(9�<���        )��P	RRN���A�!*

	conv_lossnl�<�+�        )��P	�ERN���A�!*

	conv_loss[�=8�v�        )��P	xxRN���A�!*

	conv_lossX�C=\��n        )��P	r�RN���A�!*

	conv_lossu>�<Nk��        )��P	x�RN���A�!*

	conv_loss��
=��        )��P	/SN���A�!*

	conv_loss�=�`�        )��P	icSN���A�!*

	conv_loss���<���        )��P	�SN���A�!*

	conv_loss��.=!�ED        )��P	G�SN���A�!*

	conv_loss��=���        )��P	�TN���A�!*

	conv_losssO�<��&        )��P	<9TN���A�!*

	conv_lossE#�<Kl�8        )��P	flTN���A�!*

	conv_lossZ1�<��5�        )��P	u�TN���A�!*

	conv_lossr�<n�w0        )��P	J�TN���A�!*

	conv_loss_޸<,5dE        )��P	�UN���A�!*

	conv_loss���<�1h`        )��P	~AUN���A�!*

	conv_loss�#=�-�        )��P	iyUN���A�!*

	conv_loss��<��g�        )��P	m�UN���A�!*

	conv_loss��=<p�R�        )��P	�UN���A�!*

	conv_loss/�=���t        )��P	�VN���A�!*

	conv_loss%�=�fq�        )��P	�QVN���A�!*

	conv_loss�=�k        )��P	^�VN���A�!*

	conv_lossj=D���        )��P	��VN���A�!*

	conv_loss�C+=�N��        )��P	��VN���A�!*

	conv_loss�f�<!p".        )��P	�WN���A�!*

	conv_loss��)<��:�        )��P	2NWN���A�!*

	conv_loss̡�<0�.$        )��P	WN���A�!*

	conv_loss�S�<fp¢        )��P	A�WN���A�!*

	conv_lossS� =Kҡ        )��P	:�WN���A�!*

	conv_lossɤ�<e�m        )��P	�+XN���A�!*

	conv_loss���<O��P        )��P	v^XN���A�!*

	conv_lossJ�O=\���        )��P	�XN���A�!*

	conv_loss/&E=�0*G        )��P	D�XN���A�!*

	conv_loss&+=H��        )��P	��XN���A�!*

	conv_lossU=�9�1        )��P	�1YN���A�!*

	conv_loss��<��H�        )��P	�cYN���A�!*

	conv_loss�/R=�^8        )��P	��YN���A�!*

	conv_loss6�<m��        )��P	�YN���A�!*

	conv_loss�
�<s�6�        )��P	�YN���A�!*

	conv_loss�׋<QS        )��P	�-ZN���A�!*

	conv_lossН=��ol        )��P	bZN���A�!*

	conv_loss:N=>�H        )��P	��ZN���A�!*

	conv_loss`�	=Y	l_        )��P	-�ZN���A�!*

	conv_loss%��<���*        )��P	��ZN���A�!*

	conv_loss�ZB=G���        )��P	�)[N���A�!*

	conv_loss�W�<B�T�        )��P	ia[N���A�!*

	conv_loss*_'=
�V        )��P	E�[N���A�!*

	conv_lossbY/=Ei        )��P	��[N���A�!*

	conv_loss Y�<�^��        )��P	��[N���A�!*

	conv_loss2�<����        )��P	�7\N���A�!*

	conv_losspk�</`�        )��P	�k\N���A�!*

	conv_loss�q=E� �        )��P	(�\N���A�!*

	conv_loss�VN=�R8F        )��P	�\N���A�!*

	conv_loss/��<H�0�        )��P	�!]N���A�!*

	conv_loss޶=D{x        )��P	�Z]N���A�"*

	conv_loss���<��        )��P	��]N���A�"*

	conv_lossWME=i���        )��P	��]N���A�"*

	conv_loss#1�<t���        )��P	��]N���A�"*

	conv_loss3��<6        )��P	8^N���A�"*

	conv_loss�=����        )��P	�[^N���A�"*

	conv_loss=�<T"Ϛ        )��P	<�^N���A�"*

	conv_loss"�	=�q��        )��P	I�^N���A�"*

	conv_lossΘ�<l6Y        )��P	��^N���A�"*

	conv_loss"�=Ƹ��        )��P	�,_N���A�"*

	conv_loss���<�ꢲ        )��P	�m_N���A�"*

	conv_losss��<�A        )��P	��_N���A�"*

	conv_loss��<�`��        )��P	�_N���A�"*

	conv_loss"W�< v�.        )��P	�`N���A�"*

	conv_lossFDU=��ͮ        )��P	L9`N���A�"*

	conv_loss+::={�l        )��P	p`N���A�"*

	conv_loss~��<�l�        )��P	b�`N���A�"*

	conv_loss��<�lڶ        )��P	��`N���A�"*

	conv_loss�-�<x>�        )��P	�aN���A�"*

	conv_loss��=�M        )��P	�;aN���A�"*

	conv_loss,�<�c��        )��P	^�aN���A�"*

	conv_loss�e1=���<        )��P	#�aN���A�"*

	conv_lossp��<�ΘP        )��P	��aN���A�"*

	conv_losshE/=�Kq�        )��P	(bN���A�"*

	conv_loss��<.���        )��P	_gbN���A�"*

	conv_loss+�6<�er        )��P	��bN���A�"*

	conv_loss�Y9=ま�        )��P	m�bN���A�"*

	conv_loss/u�<آ)        )��P	�cN���A�"*

	conv_lossǣ=��]        )��P	�=cN���A�"*

	conv_lossl�=Y'SZ        )��P		|cN���A�"*

	conv_loss���<���        )��P	��cN���A�"*

	conv_lossX�<�s#�        )��P	~�cN���A�"*

	conv_loss��<�[�        )��P	>dN���A�"*

	conv_lossŷ<%eǏ        )��P	�7dN���A�"*

	conv_loss�d=�Q�        )��P	/�dN���A�"*

	conv_loss�2�<�m��        )��P	�dN���A�"*

	conv_loss�f�<�W�        )��P	`�dN���A�"*

	conv_loss+��<��	�        )��P	WeN���A�"*

	conv_loss�ĩ<DK�        )��P	�LeN���A�"*

	conv_loss�%[=)C        )��P	ΊeN���A�"*

	conv_lossv�<F%t        )��P	A�eN���A�"*

	conv_lossYq�<F��}        )��P	^�eN���A�"*

	conv_loss�;u<0���        )��P	 fN���A�"*

	conv_loss��=!��        )��P	RfN���A�"*

	conv_loss2��<��        )��P	K�fN���A�"*

	conv_loss��$=E��O        )��P	��fN���A�"*

	conv_lossA�<xD�        )��P	�fN���A�"*

	conv_loss�<�<���        )��P	MgN���A�"*

	conv_loss�M�<j@�        )��P	�DgN���A�"*

	conv_loss���<�"%t        )��P	�hN���A�"*

	conv_loss�2�<���x        )��P	�iN���A�"*

	conv_loss��<"N        )��P	�IiN���A�"*

	conv_loss��/<6�Eh        )��P	��iN���A�"*

	conv_loss�k�<��F]        )��P	�iN���A�"*

	conv_loss�;)=o�gG        )��P	��iN���A�"*

	conv_lossw�<j��        )��P	�)jN���A�"*

	conv_lossXJ=F���        )��P	!\jN���A�"*

	conv_loss��*=���        )��P	��jN���A�"*

	conv_loss���<�v�        )��P	��jN���A�"*

	conv_losshO=���s        )��P	G�jN���A�"*

	conv_loss��<y���        )��P	�0kN���A�"*

	conv_loss"��<����        )��P	�bkN���A�"*

	conv_lossɋ�<�	�        )��P	_�kN���A�"*

	conv_lossH�	=�0(�        )��P	��kN���A�"*

	conv_loss"�<�_x        )��P	�lN���A�"*

	conv_lossIN�<'�kz        )��P	�?lN���A�"*

	conv_loss�C=�I��        )��P	�olN���A�"*

	conv_loss�=a�.�        )��P	�lN���A�"*

	conv_loss�	�<���        )��P	�lN���A�"*

	conv_loss�t�<��        )��P	+mN���A�"*

	conv_loss��=�c�        )��P	�2mN���A�"*

	conv_loss�T=�u        )��P	mamN���A�"*

	conv_loss��C=Y�\        )��P	��mN���A�"*

	conv_loss�n@=�PjX        )��P	'�mN���A�"*

	conv_loss�' =����        )��P	nN���A�"*

	conv_loss���<��        )��P	9nN���A�"*

	conv_lossl�=�Df�        )��P	�pnN���A�"*

	conv_loss��T<d�td        )��P	0�nN���A�"*

	conv_loss�v!=߸y�        )��P	��nN���A�"*

	conv_loss��=���        )��P	z&oN���A�"*

	conv_loss�n=$��        )��P	�[oN���A�"*

	conv_loss}=�u        )��P	��oN���A�"*

	conv_loss�>�<ʻ��        )��P	5�oN���A�"*

	conv_loss8L�<����        )��P	�pN���A�"*

	conv_lossls�<����        )��P	<8pN���A�"*

	conv_loss-<<�oV�        )��P	jpN���A�"*

	conv_loss=���        )��P	T�pN���A�"*

	conv_lossT<m<M�'+        )��P	�pN���A�"*

	conv_loss@�<@q]�        )��P	�qN���A�"*

	conv_loss��=��L�        )��P	^GqN���A�"*

	conv_loss���<c�Ŷ        )��P	�yqN���A�"*

	conv_loss�(�<hO��        )��P	c�qN���A�"*

	conv_losse��<5DL        )��P	C�qN���A�"*

	conv_loss��d=cB�        )��P	rN���A�"*

	conv_loss�l�<�@�        )��P	aMrN���A�"*

	conv_loss?�
=���        )��P	E}rN���A�"*

	conv_loss� =nx�        )��P	��rN���A�"*

	conv_loss?=��`<        )��P	��rN���A�"*

	conv_loss�g�<j�I�        )��P	�sN���A�"*

	conv_loss�V�<���m        )��P	u�sN���A�"*

	conv_loss��<)&T�        )��P	��sN���A�"*

	conv_loss'>=ni�m        )��P	{�sN���A�"*

	conv_lossE�<�"�]        )��P	�.tN���A�"*

	conv_loss{�<*{�4        )��P	'atN���A�"*

	conv_losskދ<3��V        )��P	�tN���A�"*

	conv_losse/=
4�        )��P	l�tN���A�"*

	conv_loss��<<I�}        )��P	TuN���A�"*

	conv_loss���<%(�*        )��P	�MuN���A�"*

	conv_loss "�<[K}/        )��P	�uN���A�"*

	conv_loss =�F��        )��P	��uN���A�"*

	conv_loss-R�<��7�        )��P	��uN���A�"*

	conv_loss���<��c�        )��P	/vN���A�"*

	conv_loss��=��i�        )��P	�cvN���A�"*

	conv_loss��=q]�S        )��P	{�vN���A�"*

	conv_loss"[6=���T        )��P	:�vN���A�"*

	conv_loss.>=�#��        )��P	�wN���A�"*

	conv_loss�P5<�a�        )��P	<IwN���A�"*

	conv_loss�{�<�ǰ�        )��P	�{wN���A�"*

	conv_loss��<��        )��P	m�wN���A�"*

	conv_lossg�=���        )��P	Q�wN���A�"*

	conv_lossY�#=ߋI�        )��P	xN���A�"*

	conv_loss��<AEU�        )��P	�ZxN���A�"*

	conv_loss%*�<�b�J        )��P	'�xN���A�"*

	conv_loss�q�<���        )��P	��xN���A�"*

	conv_lossW�=_��        )��P	E�xN���A�"*

	conv_lossG?=�+9�        )��P	v3yN���A�"*

	conv_loss߬<��<U        )��P	�kyN���A�"*

	conv_loss.��<2�'�        )��P	��yN���A�"*

	conv_lossu��<�"7�        )��P	��yN���A�#*

	conv_lossy
=�(Y        )��P	CzN���A�#*

	conv_lossG	=EW�m        )��P	=zN���A�#*

	conv_loss6� =�o+        )��P	�mzN���A�#*

	conv_loss%��<�7�        )��P	��zN���A�#*

	conv_lossAs<��6        )��P	]�zN���A�#*

	conv_loss�̵<�n�         )��P	{N���A�#*

	conv_loss��<1��I        )��P	�<{N���A�#*

	conv_loss��=���J        )��P	oq{N���A�#*

	conv_lossjI�<�lI�        )��P	_�{N���A�#*

	conv_loss���<3"+�        )��P	��{N���A�#*

	conv_lossr�< q��        )��P	�|N���A�#*

	conv_lossy�=�Y�        )��P	wH|N���A�#*

	conv_loss=��<F��6        )��P	N||N���A�#*

	conv_loss�6�<�� s        )��P	9�|N���A�#*

	conv_loss+�=�U1�        )��P	s}N���A�#*

	conv_loss[=>S
        )��P	�A}N���A�#*

	conv_lossM��<MН        )��P	�v}N���A�#*

	conv_loss7?�<A�;y        )��P	��}N���A�#*

	conv_loss#z<Mpn�        )��P	��}N���A�#*

	conv_loss��=7�i�        )��P	x�N���A�#*

	conv_loss���<��z~        )��P	��N���A�#*

	conv_loss�.�<���        )��P	��N���A�#*

	conv_lossZ*�<���        )��P	)�N���A�#*

	conv_loss|G�<Z��        )��P	^�N���A�#*

	conv_loss%�<)        )��P	+��N���A�#*

	conv_lossyE�<\��=        )��P	���N���A�#*

	conv_loss�Y�<�W�        )��P	,��N���A�#*

	conv_loss��<y5��        )��P	�N���A�#*

	conv_loss�E
=��        )��P	3T�N���A�#*

	conv_loss�� =���        )��P	N���A�#*

	conv_lossX��<��&        )��P	۸�N���A�#*

	conv_loss-��<?���        )��P	M�N���A�#*

	conv_loss|�U=�8�w        )��P	/�N���A�#*

	conv_loss��<��W�        )��P	�J�N���A�#*

	conv_lossĻ�<N~�        )��P	��N���A�#*

	conv_loss#��<���6        )��P	���N���A�#*

	conv_loss<�=��        )��P	��N���A�#*

	conv_loss2_=S	�        )��P	M�N���A�#*

	conv_loss�L�< ;�}        )��P	K�N���A�#*

	conv_loss��<Wf�        )��P	nz�N���A�#*

	conv_loss�g�<���        )��P	���N���A�#*

	conv_loss{rG=&o�        )��P	`؆N���A�#*

	conv_loss�a�<��|�        )��P	��N���A�#*

	conv_loss��<J/Ya        )��P	!6�N���A�#*

	conv_loss���<��        )��P	�e�N���A�#*

	conv_loss}�|<GT�        )��P	���N���A�#*

	conv_loss��	=7��^        )��P	�ÇN���A�#*

	conv_loss�i<�ٓd        )��P	V�N���A�#*

	conv_loss"��<����        )��P	a$�N���A�#*

	conv_loss^��<a���        )��P	�S�N���A�#*

	conv_loss���<�j�        )��P	l��N���A�#*

	conv_loss�~	=/m��        )��P	���N���A�#*

	conv_lossD<w��        )��P	��N���A�#*

	conv_loss�N�<�8�        )��P	I�N���A�#*

	conv_loss�'�<��t�        )��P	D�N���A�#*

	conv_loss��=���`        )��P	=s�N���A�#*

	conv_loss��I=�W1.        )��P	̤�N���A�#*

	conv_loss�Y==-].)        )��P	)ԉN���A�#*

	conv_loss*p'=E٢.        )��P	��N���A�#*

	conv_loss��=l�x        )��P	�1�N���A�#*

	conv_loss'Q1=�n�
        )��P	�a�N���A�#*

	conv_loss|&
=)#$        )��P	l��N���A�#*

	conv_loss7y�<�+�        )��P	rŊN���A�#*

	conv_loss� =����        )��P	���N���A�#*

	conv_loss�<=��I4        )��P	�&�N���A�#*

	conv_lossW%�<�	v�        )��P	�W�N���A�#*

	conv_loss	.�<\V�V        )��P	���N���A�#*

	conv_loss*=D�"3        )��P	a��N���A�#*

	conv_loss�=;&��        )��P	��N���A�#*

	conv_lossu�<sO�        )��P	��N���A�#*

	conv_loss�P�<]5u        )��P	9b�N���A�#*

	conv_loss+wK=�G�<        )��P	�N���A�#*

	conv_loss:=,��        )��P	2ÌN���A�#*

	conv_loss��D=�E        )��P	��N���A�#*

	conv_loss��.=D1�,        )��P	*�N���A�#*

	conv_loss�n�<F��        )��P	�]�N���A�#*

	conv_loss�)=��?�        )��P	���N���A�#*

	conv_loss���<��3}        )��P	�ȍN���A�#*

	conv_loss`��<M1        )��P	���N���A�#*

	conv_lossp\�<ȑ        )��P	�2�N���A�#*

	conv_loss�L=�b�        )��P	'd�N���A�#*

	conv_loss<��<aN��        )��P	H��N���A�#*

	conv_loss���<��[�        )��P	�͎N���A�#*

	conv_lossvK�<���9        )��P	���N���A�#*

	conv_lossN�<~EǍ        )��P	�1�N���A�#*

	conv_lossF��<���        )��P	�h�N���A�#*

	conv_loss��;=��0k        )��P	¡�N���A�#*

	conv_loss2�= �	�        )��P	ՏN���A�#*

	conv_loss��=��6        )��P	a�N���A�#*

	conv_loss�I =�`�        )��P	�;�N���A�#*

	conv_loss~�=T#��        )��P	�n�N���A�#*

	conv_lossi�P<m�)k        )��P	���N���A�#*

	conv_loss��=�4ޞ        )��P	�ՐN���A�#*

	conv_lossU�=GN        )��P	�	�N���A�#*

	conv_loss=?��        )��P	t;�N���A�#*

	conv_loss�@
=h�        )��P	|o�N���A�#*

	conv_loss9��<s�)        )��P	D��N���A�#*

	conv_loss�T=���r        )��P	LّN���A�#*

	conv_loss��<I�        )��P	�
�N���A�#*

	conv_loss�0�<B��:        )��P	K>�N���A�#*

	conv_loss�j�<�`E        )��P	ir�N���A�#*

	conv_loss���<sV�        )��P	���N���A�#*

	conv_loss���<ИW6        )��P	�גN���A�#*

	conv_loss���<��t        )��P	��N���A�#*

	conv_loss���<�l�4        )��P	A�N���A�#*

	conv_loss9��<����        )��P	�v�N���A�#*

	conv_loss�"=���        )��P	⩓N���A�#*

	conv_loss�%%=l�J�        )��P	�ݓN���A�#*

	conv_lossV�<��        )��P	V�N���A�#*

	conv_loss��=�/�        )��P	�C�N���A�#*

	conv_loss�<���        )��P	�w�N���A�#*

	conv_loss<��<G��        )��P	ꭔN���A�#*

	conv_loss&4=-�*�        )��P	��N���A�#*

	conv_loss�ZW=V7ն        )��P	��N���A�#*

	conv_loss���<�p�l        )��P	zG�N���A�#*

	conv_loss#��<i:�s        )��P	Lz�N���A�#*

	conv_loss�T<�b#S        )��P	��N���A�#*

	conv_lossu��<'�        )��P	6�N���A�#*

	conv_loss��<���        )��P	��N���A�#*

	conv_loss�1�<�?;�        )��P	�H�N���A�#*

	conv_loss��=#��        )��P	���N���A�#*

	conv_loss"�=]�[�        )��P	��N���A�#*

	conv_loss���<��        )��P	�K�N���A�#*

	conv_lossg�7=�J��        )��P	&��N���A�#*

	conv_loss!�I=�5 �        )��P	嵘N���A�#*

	conv_loss�=�]��        )��P	��N���A�#*

	conv_loss�H�<���        )��P	}#�N���A�#*

	conv_loss��w<��X�        )��P	ZV�N���A�$*

	conv_loss��=/��        )��P	���N���A�$*

	conv_lossӄ�<�'��        )��P	�ʙN���A�$*

	conv_lossm�<CzY�        )��P	��N���A�$*

	conv_loss�<��FF        )��P	�E�N���A�$*

	conv_loss��<�Ps�        )��P	-{�N���A�$*

	conv_loss��<�G �        )��P	"��N���A�$*

	conv_loss�B=��-�        )��P	��N���A�$*

	conv_loss*�s<AQ�        )��P	�%�N���A�$*

	conv_loss��<�5n�        )��P	X�N���A�$*

	conv_loss���<��*'        )��P	1��N���A�$*

	conv_lossۄ�<fM��        )��P	��N���A�$*

	conv_loss�i�<Z�2        )��P	��N���A�$*

	conv_loss"��<7�         )��P	"1�N���A�$*

	conv_loss�©<K�'        )��P		e�N���A�$*

	conv_lossq9=�#��        )��P	V��N���A�$*

	conv_loss�5="���        )��P	˜N���A�$*

	conv_loss]��<��#�        )��P	6 �N���A�$*

	conv_loss��=�N��        )��P	Q5�N���A�$*

	conv_lossD�=&�%+        )��P	>g�N���A�$*

	conv_loss���<WYi        )��P	���N���A�$*

	conv_lossȞ�<��CJ        )��P	ԝN���A�$*

	conv_loss9�=���        )��P	��N���A�$*

	conv_loss?�<C��        )��P	�9�N���A�$*

	conv_loss=Z�<�Ia�        )��P	3m�N���A�$*

	conv_loss�O�<��b        )��P	���N���A�$*

	conv_lossX2=abW        )��P	jԞN���A�$*

	conv_loss��<� ?+        )��P	��N���A�$*

	conv_lossu��<��m        )��P	iO�N���A�$*

	conv_lossv��<̊_        )��P	a��N���A�$*

	conv_losst�<����        )��P	���N���A�$*

	conv_loss�=%ipc        )��P	s�N���A�$*

	conv_loss�f2=�nc�        )��P	��N���A�$*

	conv_loss���<���        )��P	�R�N���A�$*

	conv_loss���<�6��        )��P	ݎ�N���A�$*

	conv_lossW:=v?��        )��P	���N���A�$*

	conv_loss3.�<AQR        )��P	���N���A�$*

	conv_lossEA�<��v�        )��P	U4�N���A�$*

	conv_loss�pd=�a�        )��P	�m�N���A�$*

	conv_loss^=�8D        )��P	���N���A�$*

	conv_loss�mI=R���        )��P	�ޡN���A�$*

	conv_loss�=�9?        )��P	F�N���A�$*

	conv_loss���<��u�        )��P	�U�N���A�$*

	conv_loss[UA=z7q
        )��P	Ȣ�N���A�$*

	conv_lossL�<R��V        )��P	CۢN���A�$*

	conv_loss=�<>R�        )��P	��N���A�$*

	conv_loss"��<l��        )��P	#V�N���A�$*

	conv_loss5d3=<��        )��P	���N���A�$*

	conv_loss�d�<��ߧ        )��P	~ˣN���A�$*

	conv_loss�	�<���A        )��P	$ �N���A�$*

	conv_loss� �<.���        )��P	�7�N���A�$*

	conv_loss�s�<���
        )��P	�n�N���A�$*

	conv_lossN<+=�ɟ�        )��P	M��N���A�$*

	conv_lossP�=�b        )��P	�ۤN���A�$*

	conv_loss}�2=J���        )��P	_�N���A�$*

	conv_loss'�<�/8        )��P	�=�N���A�$*

	conv_loss���<\B,�        )��P	z�N���A�$*

	conv_loss�<��)�        )��P	��N���A�$*

	conv_loss)\�<_̚        )��P	ܥN���A�$*

	conv_loss98<e��        )��P	��N���A�$*

	conv_lossJ@�<��=�        )��P	7P�N���A�$*

	conv_loss�>�<�/'        )��P	���N���A�$*

	conv_loss���<9O�        )��P	ɦN���A�$*

	conv_loss1�<K]        )��P	���N���A�$*

	conv_loss���<����        )��P	w/�N���A�$*

	conv_loss��<5��        )��P	�q�N���A�$*

	conv_loss�S=�ʢ:        )��P	��N���A�$*

	conv_loss�d�<	_S        )��P	6ڧN���A�$*

	conv_loss�=쌤�        )��P	8�N���A�$*

	conv_loss�
�<��;�        )��P	8>�N���A�$*

	conv_loss�r&=�Ʉ�        )��P	>z�N���A�$*

	conv_loss8��<�3h        )��P	���N���A�$*

	conv_loss��=y���        )��P	z�N���A�$*

	conv_loss���<���        )��P	��N���A�$*

	conv_loss[p=!eI�        )��P	R�N���A�$*

	conv_loss�f�<u7h�        )��P	 ��N���A�$*

	conv_lossr�N=m�!%        )��P	V��N���A�$*

	conv_losse�<9���        )��P	��N���A�$*

	conv_lossw@�<cS#�        )��P	��N���A�$*

	conv_loss��=�Z�s        )��P	�O�N���A�$*

	conv_loss��<»�        )��P	��N���A�$*

	conv_loss0�<&�K�        )��P	Ű�N���A�$*

	conv_loss��=WK�        )��P	�ުN���A�$*

	conv_loss�A�<^}'7        )��P	.�N���A�$*

	conv_loss5�>=�G        )��P	�X�N���A�$*

	conv_loss�V=�[y@        )��P	���N���A�$*

	conv_losspN�<5�2        )��P	�˫N���A�$*

	conv_loss�-L<��0�        )��P	i��N���A�$*

	conv_loss�x�<����        )��P	b3�N���A�$*

	conv_loss�X�<q���        )��P		g�N���A�$*

	conv_loss%m�<)��        )��P	���N���A�$*

	conv_lossڌ�<�}�        )��P	jڬN���A�$*

	conv_loss�!<F��7        )��P	��N���A�$*

	conv_lossS=�s�        )��P	�Q�N���A�$*

	conv_loss���<�o�        )��P	<��N���A�$*

	conv_loss���<��        )��P	�ЭN���A�$*

	conv_loss�AL<���        )��P	;
�N���A�$*

	conv_loss��=h��        )��P	�E�N���A�$*

	conv_loss�ӿ<g��        )��P	�w�N���A�$*

	conv_loss���<�k�         )��P	���N���A�$*

	conv_lossP�=<[���        )��P	��N���A�$*

	conv_loss�K�<�G�W        )��P	�#�N���A�$*

	conv_loss�?!=�se�        )��P	RT�N���A�$*

	conv_loss���<%'?�        )��P	φ�N���A�$*

	conv_loss��<�p��        )��P	���N���A�$*

	conv_loss7I�<�{:�        )��P	��N���A�$*

	conv_loss/�q<���        )��P	�%�N���A�$*

	conv_lossT�j<|z�        )��P	�]�N���A�$*

	conv_loss���<P��z        )��P	b��N���A�$*

	conv_lossP\�<h�        )��P	�ͰN���A�$*

	conv_loss�i�<�^|�        )��P	���N���A�$*

	conv_loss��<�x        )��P	�-�N���A�$*

	conv_loss�4=�#��        )��P	^�N���A�$*

	conv_lossO/�<z��        )��P	��N���A�$*

	conv_loss���<Mٌ        )��P	QƱN���A�$*

	conv_lossr��<��ԭ        )��P	Z��N���A�$*

	conv_loss_�<�G�        )��P	(%�N���A�$*

	conv_loss��<t���        )��P	ZV�N���A�$*

	conv_loss�!�<�3�        )��P	y��N���A�$*

	conv_loss���<��U        )��P	&˲N���A�$*

	conv_losss8�<Wa�        )��P	��N���A�$*

	conv_loss+S�<���        )��P	7�N���A�$*

	conv_lossI�<� �)        )��P	k�N���A�$*

	conv_loss���<sAP	        )��P	���N���A�$*

	conv_lossP��<z��3        )��P	�ֳN���A�$*

	conv_loss�U=J S        )��P	�N���A�$*

	conv_loss�"�<�J
�        )��P	S?�N���A�$*

	conv_lossު3=��1�        )��P	vq�N���A�$*

	conv_loss{<���        )��P	۪�N���A�%*

	conv_lossZ��<!���        )��P	��N���A�%*

	conv_loss"��<�.�`        )��P	O �N���A�%*

	conv_loss{�<4��F        )��P	�[�N���A�%*

	conv_lossRh�<H�s        )��P	���N���A�%*

	conv_loss��<2g��        )��P	�ĵN���A�%*

	conv_lossQ'�</lh2        )��P	���N���A�%*

	conv_loss��<6��	        )��P	�0�N���A�%*

	conv_loss�:<<��4�        )��P	Yc�N���A�%*

	conv_loss�2�<a��        )��P	���N���A�%*

	conv_lossT=X�-�        )��P	�˶N���A�%*

	conv_loss-�s<��1        )��P	H�N���A�%*

	conv_loss6��<���!        )��P	U3�N���A�%*

	conv_loss�߈<hPg�        )��P	.e�N���A�%*

	conv_loss���</fɼ        )��P	g��N���A�%*

	conv_loss:�=��c        )��P	��N���A�%*

	conv_loss�)=��P�        )��P	��N���A�%*

	conv_loss =�fI�        )��P	�K�N���A�%*

	conv_loss'��<��        )��P	H~�N���A�%*

	conv_loss���<ՒA�        )��P	���N���A�%*

	conv_loss+�<��        )��P	5�N���A�%*

	conv_lossZ�<�g        )��P	�#�N���A�%*

	conv_lossG�]=��        )��P	.V�N���A�%*

	conv_loss�&
=~z��        )��P	���N���A�%*

	conv_loss97�<nԆ^        )��P	��N���A�%*

	conv_loss�>�<�;�        )��P	T�N���A�%*

	conv_loss��<���        )��P	�)�N���A�%*

	conv_lossbew<�@��        )��P	e�N���A�%*

	conv_lossJ?�<$oB        )��P	䜺N���A�%*

	conv_loss�{�<��3{        )��P	�кN���A�%*

	conv_loss���<�4��        )��P	��N���A�%*

	conv_lossvA�<�2ca        )��P	�@�N���A�%*

	conv_lossV�<��bb        )��P	�q�N���A�%*

	conv_loss�*�<���        )��P	��N���A�%*

	conv_lossܫ<�-��        )��P	�ӻN���A�%*

	conv_lossA��<��        )��P	t�N���A�%*

	conv_loss��	=�Ɨ*        )��P	\>�N���A�%*

	conv_loss��<�]        )��P	hm�N���A�%*

	conv_lossW<yսr        )��P	N���A�%*

	conv_loss��~<\��        )��P	ʼN���A�%*

	conv_loss�;�<q+�        )��P	� �N���A�%*

	conv_loss�/�<�6Q�        )��P	�;�N���A�%*

	conv_loss�v�<�l��        )��P	�r�N���A�%*

	conv_loss�R�<؅�        )��P	���N���A�%*

	conv_loss��<��r        )��P	E׽N���A�%*

	conv_loss�f0=Tİ�        )��P	��N���A�%*

	conv_loss��<+�\�        )��P	�K�N���A�%*

	conv_loss� =U��        )��P	�}�N���A�%*

	conv_loss��5=�f=�        )��P	i��N���A�%*

	conv_loss>�<�d�        )��P	A�N���A�%*

	conv_loss/;�<���b        )��P	}�N���A�%*

	conv_lossNj=�;]        )��P	�U�N���A�%*

	conv_loss�g�< m]/        )��P	U��N���A�%*

	conv_loss9�/=�{&        )��P	���N���A�%*

	conv_loss�=	=�r҉        )��P	��N���A�%*

	conv_loss���<��i�        )��P	: �N���A�%*

	conv_loss�%�<���        )��P	�[�N���A�%*

	conv_losso�=F��n        )��P	���N���A�%*

	conv_loss�%�<���        )��P	7��N���A�%*

	conv_loss.m=�U�        )��P	���N���A�%*

	conv_loss���<�r        )��P	!�N���A�%*

	conv_loss#Շ<��-        )��P	\T�N���A�%*

	conv_losss�=(��9        )��P	n��N���A�%*

	conv_loss��<mQ��        )��P	��N���A�%*

	conv_loss�=��_        )��P	���N���A�%*

	conv_lossk�?=�An        )��P	V��N���A�%*

	conv_loss�A�<�;:        )��P	i��N���A�%*

	conv_lossΜ�<�g-�        )��P	��N���A�%*

	conv_loss�8=!X        )��P	�;�N���A�%*

	conv_lossBo=�_        )��P	t�N���A�%*

	conv_loss(�<DN��        )��P	���N���A�%*

	conv_loss���<d���        )��P	0��N���A�%*

	conv_loss���<V�8        )��P	Z�N���A�%*

	conv_loss,V\<Gd�        )��P	�A�N���A�%*

	conv_lossY\�<�9�        )��P	�{�N���A�%*

	conv_lossX�=H�        )��P	Ƴ�N���A�%*

	conv_loss5�<<v�        )��P	���N���A�%*

	conv_loss��<��P        )��P	��N���A�%*

	conv_loss��<�>        )��P	_D�N���A�%*

	conv_loss��n<[�'        )��P	)}�N���A�%*

	conv_loss_1@=��F�        )��P	ǫ�N���A�%*

	conv_loss�'�<3$        )��P	���N���A�%*

	conv_loss��=���.        )��P	�N���A�%*

	conv_loss]��<�?�q        )��P	S�N���A�%*

	conv_loss�@i<� �@        )��P	Ȉ�N���A�%*

	conv_losspz?=b{        )��P	��N���A�%*

	conv_lossy�5=P�ݛ        )��P	?��N���A�%*

	conv_lossLJ =8Ep        )��P	k#�N���A�%*

	conv_loss`E2=nU�N        )��P	W�N���A�%*

	conv_loss�W�<9٠T        )��P	<��N���A�%*

	conv_loss�j�<��*        )��P	q��N���A�%*

	conv_loss��=�v�>        )��P	 �N���A�%*

	conv_loss��<��/�        )��P	c9�N���A�%*

	conv_loss���<����        )��P	�|�N���A�%*

	conv_loss�=���k        )��P	��N���A�%*

	conv_lossW=�+��        )��P	���N���A�%*

	conv_losseB�<�G�        )��P	� �N���A�%*

	conv_lossy�<��"�        )��P	�V�N���A�%*

	conv_lossO��<�4'�        )��P	1��N���A�%*

	conv_loss2�<��B�        )��P	���N���A�%*

	conv_loss�`2<S�v        )��P	3��N���A�%*

	conv_loss�=��/        )��P	�+�N���A�%*

	conv_loss��<��S�        )��P	�a�N���A�%*

	conv_loss��J<3ѡx        )��P	̛�N���A�%*

	conv_losshj�<�pQ]        )��P	t��N���A�%*

	conv_lossI-=�޶        )��P	���N���A�%*

	conv_loss.�<� y        )��P	o8�N���A�%*

	conv_loss�%=Y���        )��P	�q�N���A�%*

	conv_lossҕ<��	        )��P	b��N���A�%*

	conv_loss�'P<�~[        )��P	���N���A�%*

	conv_loss���<J�a        )��P	��N���A�%*

	conv_loss/t+=�u�        )��P	�<�N���A�%*

	conv_loss�,=�lڅ        )��P	7n�N���A�%*

	conv_lossk��<�֋        )��P	+��N���A�%*

	conv_loss��<j��        )��P	���N���A�%*

	conv_loss�	�<�NO        )��P	��N���A�%*

	conv_loss��=]��        )��P	!P�N���A�%*

	conv_loss�} =�9]�        )��P	��N���A�%*

	conv_loss��<����        )��P	i��N���A�%*

	conv_loss�Z�<�8�:        )��P	C �N���A�%*

	conv_loss��<��        )��P	Q1�N���A�%*

	conv_loss=��<c���        )��P	Zc�N���A�%*

	conv_loss]�F<9�z�        )��P	��N���A�%*

	conv_loss_�<uS-�        )��P	��N���A�%*

	conv_loss�6�<��@N        )��P	� �N���A�%*

	conv_loss�a�<���)        )��P	�1�N���A�%*

	conv_loss�
='!�        )��P	�a�N���A�%*

	conv_lossq`�<ӗ!4        )��P	'��N���A�%*

	conv_loss�E�<���        )��P	���N���A�&*

	conv_loss�<g@��        )��P	� �N���A�&*

	conv_loss3�<��        )��P	�1�N���A�&*

	conv_loss�>�<���r        )��P	�i�N���A�&*

	conv_lossXt�<���        )��P	H��N���A�&*

	conv_loss8G,=�Y 
        )��P	*��N���A�&*

	conv_loss4�<L�        )��P	5�N���A�&*

	conv_loss���<xų        )��P	9�N���A�&*

	conv_lossƥ�<A�ܶ        )��P	�i�N���A�&*

	conv_lossCR�<b�/~        )��P	N��N���A�&*

	conv_loss���<zP        )��P	���N���A�&*

	conv_loss�b5=�ڿ^        )��P	<�N���A�&*

	conv_loss�_=�wZ        )��P	�H�N���A�&*

	conv_loss���<Y�GR        )��P	�{�N���A�&*

	conv_loss��1=�k        )��P	���N���A�&*

	conv_loss�\�<�O>        )��P	K��N���A�&*

	conv_loss��=��}]        )��P	�.�N���A�&*

	conv_loss�_=��
        )��P	;`�N���A�&*

	conv_lossH;=����        )��P	X��N���A�&*

	conv_lossH��<+�#        )��P	���N���A�&*

	conv_loss<��<�ֶ"        )��P	w�N���A�&*

	conv_loss'�<V�4        )��P	J6�N���A�&*

	conv_loss#T�<%��        )��P	�i�N���A�&*

	conv_loss���<����        )��P	���N���A�&*

	conv_loss�<.ܵ�        )��P	"��N���A�&*

	conv_loss�2�<r��e        )��P	��N���A�&*

	conv_loss]�!=���        )��P	|G�N���A�&*

	conv_lossQ�<g�̭        )��P	~�N���A�&*

	conv_lossJAl<x�5�        )��P	K��N���A�&*

	conv_losszڎ<�K,        )��P	O��N���A�&*

	conv_loss���<.��\        )��P	b*�N���A�&*

	conv_lossi��<R%        )��P	�^�N���A�&*

	conv_loss���<ΧV�        )��P	j��N���A�&*

	conv_loss�=I�r�        )��P	T��N���A�&*

	conv_lossl$&=�z	7        )��P	j �N���A�&*

	conv_loss�B�<ԷK:        )��P	�2�N���A�&*

	conv_loss��<��#        )��P	�c�N���A�&*

	conv_loss�#=U�        )��P	H��N���A�&*

	conv_loss���<���        )��P	��N���A�&*

	conv_loss� =�oo�        )��P	�%�N���A�&*

	conv_loss���<�t��        )��P	�X�N���A�&*

	conv_loss>S�<e�3        )��P	&��N���A�&*

	conv_loss���<����        )��P	���N���A�&*

	conv_loss�i�<�c�        )��P	���N���A�&*

	conv_loss	��<�p        )��P	�0�N���A�&*

	conv_lossX�<���        )��P	3d�N���A�&*

	conv_lossr�<p��        )��P	P��N���A�&*

	conv_loss�6�<����        )��P	2��N���A�&*

	conv_loss8�}<����        )��P	��N���A�&*

	conv_loss&{}<�0:�        )��P	N3�N���A�&*

	conv_loss�;�<��cr        )��P	�q�N���A�&*

	conv_loss�A<�j=9        )��P	��N���A�&*

	conv_lossY��<����        )��P	}��N���A�&*

	conv_lossV��<�	�3        )��P	��N���A�&*

	conv_loss��<���4        )��P	�D�N���A�&*

	conv_loss���<�ܵ        )��P	�v�N���A�&*

	conv_lossH�=u3�        )��P	U��N���A�&*

	conv_loss���<(�Qg        )��P	;��N���A�&*

	conv_loss��<����        )��P	��N���A�&*

	conv_loss`=���        )��P	VK�N���A�&*

	conv_loss���<����        )��P	"~�N���A�&*

	conv_loss��<+��        )��P	M��N���A�&*

	conv_loss5��<��HO        )��P	W��N���A�&*

	conv_lossϗ�<�k�        )��P	  �N���A�&*

	conv_loss_]�<���        )��P	�V�N���A�&*

	conv_loss�X<��<        )��P	��N���A�&*

	conv_lossQ'�<��        )��P	P��N���A�&*

	conv_loss��=5��        )��P	`��N���A�&*

	conv_lossL�w=9� q        )��P	8<�N���A�&*

	conv_loss&�
=��P|        )��P	uu�N���A�&*

	conv_lossJ��<��?        )��P	M��N���A�&*

	conv_loss!]�<dv?L        )��P	��N���A�&*

	conv_losse$�<-�        )��P	J�N���A�&*

	conv_loss�B�<����        )��P	�N�N���A�&*

	conv_loss�R�<��        )��P	���N���A�&*

	conv_loss�n�<��<        )��P	8��N���A�&*

	conv_loss)d�<�)�        )��P	���N���A�&*

	conv_loss90,=9�8        )��P	g%�N���A�&*

	conv_loss=�&�b        )��P	�U�N���A�&*

	conv_loss��<
3�f        )��P	���N���A�&*

	conv_loss�P�<�J��        )��P	���N���A�&*

	conv_loss(K�<����        )��P	���N���A�&*

	conv_loss�D�<�8�Y        )��P	��N���A�&*

	conv_loss�V=��        )��P	�\�N���A�&*

	conv_loss��=Jr�0        )��P	��N���A�&*

	conv_loss���<x>&        )��P	���N���A�&*

	conv_loss8(!=���%        )��P	��N���A�&*

	conv_loss=Q%�        )��P	�L�N���A�&*

	conv_loss&W�<���        )��P	���N���A�&*

	conv_lossv�S<��J�        )��P	���N���A�&*

	conv_lossS"=�        )��P	���N���A�&*

	conv_loss��<�q�        )��P	�!�N���A�&*

	conv_loss�<�4��        )��P	�T�N���A�&*

	conv_loss2�<j2�b        )��P	@��N���A�&*

	conv_loss9��<]w�r        )��P	5��N���A�&*

	conv_loss=�<	�F�        )��P	���N���A�&*

	conv_loss �<�|Q�        )��P	�'�N���A�&*

	conv_loss��<jN�P        )��P	�X�N���A�&*

	conv_loss_��<�U��        )��P	���N���A�&*

	conv_loss�d�<ӷ��        )��P	���N���A�&*

	conv_loss��=_��        )��P	��N���A�&*

	conv_loss��<���        )��P	�.�N���A�&*

	conv_loss��<�+�0        )��P	u^�N���A�&*

	conv_lossd5#=-b2�        )��P	;��N���A�&*

	conv_lossq
=K��        )��P	���N���A�&*

	conv_loss�%=ى��        )��P	���N���A�&*

	conv_lossZ��<��:c        )��P	�&�N���A�&*

	conv_lossvp<��C
        )��P	V�N���A�&*

	conv_loss�E
=����        )��P	X��N���A�&*

	conv_loss�j�<~�D        )��P	]��N���A�&*

	conv_lossZH=�9�N        )��P	8��N���A�&*

	conv_loss��<�G�4        )��P	R�N���A�&*

	conv_loss̓=��q        )��P	lT�N���A�&*

	conv_lossC�=�Km        )��P	��N���A�&*

	conv_loss� �<�         )��P	���N���A�&*

	conv_loss��=Y�?        )��P	���N���A�&*

	conv_lossQ`�<�|6�        )��P	1�N���A�&*

	conv_loss�n�<�.�        )��P	�B�N���A�&*

	conv_losse|�<�G\        )��P	r�N���A�&*

	conv_loss�T�<xE�        )��P	���N���A�&*

	conv_loss炠<�mb�        )��P	,��N���A�&*

	conv_loss:��<�=��        )��P	!��N���A�&*

	conv_lossð�<��y2        )��P	�+�N���A�&*

	conv_loss '=����        )��P	hZ�N���A�&*

	conv_loss��z<�7��        )��P	��N���A�&*

	conv_lossZ =;��        )��P	���N���A�&*

	conv_loss�%�<��Y        )��P	���N���A�&*

	conv_loss��=l
�$        )��P	-�N���A�&*

	conv_loss%&=��        )��P	E�N���A�'*

	conv_loss��=�!Q�        )��P	�t�N���A�'*

	conv_loss{K=S���        )��P	c��N���A�'*

	conv_lossW�t<1up�        )��P	���N���A�'*

	conv_loss3�=���        )��P	I�N���A�'*

	conv_lossDh�<�?��        )��P	�6�N���A�'*

	conv_lossps=l��f        )��P	�f�N���A�'*

	conv_loss�l�<yʳ�        )��P	��N���A�'*

	conv_loss��<	8)�        )��P	��N���A�'*

	conv_loss$��<��Q        )��P	��N���A�'*

	conv_loss5k=@��        )��P	G��N���A�'*

	conv_loss��<�g�j        )��P	� �N���A�'*

	conv_loss\^"=�        )��P	�.�N���A�'*

	conv_loss���<Q���        )��P	[a�N���A�'*

	conv_lossg�0=���M        )��P	��N���A�'*

	conv_lossꝸ<�!        )��P	��N���A�'*

	conv_lossa5=w�k        )��P	���N���A�'*

	conv_lossD��<Q�9:        )��P	)'�N���A�'*

	conv_lossB},<4�        )��P	�W�N���A�'*

	conv_loss��s<���        )��P	���N���A�'*

	conv_loss�+�<���        )��P	���N���A�'*

	conv_loss�U�<��         )��P	���N���A�'*

	conv_loss�.=Æg        )��P	�0�N���A�'*

	conv_loss���<0y�        )��P	'`�N���A�'*

	conv_loss^[=�i_|        )��P	���N���A�'*

	conv_loss	:�<Jz�        )��P	���N���A�'*

	conv_lossj�<���=        )��P	k��N���A�'*

	conv_loss'9*=��7h        )��P	��N���A�'*

	conv_lossD�<0dt        )��P	O�N���A�'*

	conv_loss��<�3	�        )��P	*��N���A�'*

	conv_loss�T�<��        )��P	���N���A�'*

	conv_lossm=͡�        )��P	d��N���A�'*

	conv_lossvس<>���        )��P	`�N���A�'*

	conv_loss��<�]l�        )��P	�>�N���A�'*

	conv_loss���<�)�J        )��P	"n�N���A�'*

	conv_loss=��<�$Y	        )��P	���N���A�'*

	conv_loss6��<�0,;        )��P	��N���A�'*

	conv_lossl2=sw��        )��P	��N���A�'*

	conv_lossR=$��k        )��P	f,�N���A�'*

	conv_loss�f�<�<��        )��P	�[�N���A�'*

	conv_loss�
�<����        )��P	S��N���A�'*

	conv_losst�<��~        )��P	���N���A�'*

	conv_loss�O:=N��<        )��P	���N���A�'*

	conv_lossy�<cdv�        )��P	��N���A�'*

	conv_lossE
=J<-        )��P	F�N���A�'*

	conv_lossX֝<��        )��P	�t�N���A�'*

	conv_loss�п<���        )��P	���N���A�'*

	conv_loss
��<�d�        )��P	b��N���A�'*

	conv_loss[��<��	H        )��P	�N���A�'*

	conv_loss��<��O        )��P	�7�N���A�'*

	conv_loss���<}��        )��P	Ul�N���A�'*

	conv_loss��i<�9�        )��P	1��N���A�'*

	conv_lossP<�<;zh�        )��P	���N���A�'*

	conv_loss�i�<��U        )��P	���N���A�'*

	conv_lossVV�<*ӺA        )��P	{*�N���A�'*

	conv_loss�'�<\a�        )��P	�`�N���A�'*

	conv_loss���<�)�        )��P	��N���A�'*

	conv_loss:�m<�ԥ�        )��P	 ��N���A�'*

	conv_loss+n�<J@Zs        )��P	3��N���A�'*

	conv_loss���<���        )��P	9<�N���A�'*

	conv_loss���<=��r        )��P	�p�N���A�'*

	conv_loss)�<�e        )��P	���N���A�'*

	conv_loss~��<���        )��P	���N���A�'*

	conv_loss��=�z�        )��P	v�N���A�'*

	conv_loss�t	=���        )��P	�8�N���A�'*

	conv_lossZ�=��M        )��P	�n�N���A�'*

	conv_loss=G<�X �        )��P	��N���A�'*

	conv_loss b=\19v        )��P	���N���A�'*

	conv_loss�b�<���        )��P	�	�N���A�'*

	conv_loss�?�<��T�        )��P	�<�N���A�'*

	conv_loss���<�}�:        )��P	�t�N���A�'*

	conv_loss��=�ˤ�        )��P	*��N���A�'*

	conv_loss�w<�R�R        )��P	b��N���A�'*

	conv_lossI��<���s        )��P	��N���A�'*

	conv_loss�:�<����        )��P	rE�N���A�'*

	conv_loss?��<a�4        )��P	�x�N���A�'*

	conv_loss4�<��$>        )��P	J��N���A�'*

	conv_lossl��<�Y3        )��P	'��N���A�'*

	conv_loss��<_4�        )��P	� O���A�'*

	conv_loss�V�<�v��        )��P	�6 O���A�'*

	conv_loss�5�<1"        )��P	2e O���A�'*

	conv_loss�1�<���        )��P		� O���A�'*

	conv_losse3�<�A7C        )��P	/� O���A�'*

	conv_lossO�A=��x(        )��P	*� O���A�'*

	conv_loss�`�<<�V        )��P	:#O���A�'*

	conv_loss��=����        )��P	LTO���A�'*

	conv_lossE��<  �        )��P	h�O���A�'*

	conv_lossy�x<����        )��P	}�O���A�'*

	conv_loss�'�<eB�7        )��P	��O���A�'*

	conv_loss3B�<���g        )��P	�O���A�'*

	conv_loss��<A�        )��P	yDO���A�'*

	conv_loss6��<�&#1        )��P	�tO���A�'*

	conv_loss�t<��K+        )��P	b�O���A�'*

	conv_loss[�<�%�i        )��P	?�O���A�'*

	conv_loss��<�d�        )��P	� O���A�'*

	conv_loss��= ӊ�        )��P	�/O���A�'*

	conv_loss���<LCT        )��P	`_O���A�'*

	conv_loss���<�o
        )��P	R�O���A�'*

	conv_loss~x�<bn�        )��P	(�O���A�'*

	conv_loss�'z<r7��        )��P	��O���A�'*

	conv_loss�K�<��B�        )��P	tO���A�'*

	conv_loss��!=�*j�        )��P	MO���A�'*

	conv_loss2)�<�q�e        )��P	A�O���A�'*

	conv_lossD^<Q�        )��P	 �O���A�'*

	conv_loss�K=���        )��P	J�O���A�'*

	conv_loss@3�<ir�s        )��P	O���A�'*

	conv_lossp�<�C��        )��P	5DO���A�'*

	conv_lossD��<͞å        )��P	rO���A�'*

	conv_loss��=�|�        )��P	��O���A�'*

	conv_loss$��< t�        )��P	4�O���A�'*

	conv_loss~ט<,���        )��P	BO���A�'*

	conv_lossh��<��gS        )��P	�RO���A�'*

	conv_loss�B=�v�        )��P	�O���A�'*

	conv_loss>/�<�pU�        )��P	��O���A�'*

	conv_loss�(�<����        )��P	��O���A�'*

	conv_loss��= ��        )��P	�#O���A�'*

	conv_loss��<�	��        )��P	�RO���A�'*

	conv_lossv<�<�        )��P	��O���A�'*

	conv_loss��< ��)        )��P	��O���A�'*

	conv_loss8�<�fO        )��P	y�O���A�'*

	conv_lossp8f<dv�6        )��P	1"O���A�'*

	conv_loss���<z�8�        )��P	�QO���A�'*

	conv_loss�1�<pp�        )��P	w�O���A�'*

	conv_loss�=&)��        )��P	��O���A�'*

	conv_loss��<_H�        )��P	��O���A�'*

	conv_loss���<��JM        )��P	�,	O���A�'*

	conv_loss㈯<��k�        )��P	�]	O���A�'*

	conv_loss6K(=r�͘        )��P	��	O���A�'*

	conv_loss�� =�aC�        )��P	��	O���A�(*

	conv_loss� =���        )��P	��	O���A�(*

	conv_lossA��<��        )��P	�
O���A�(*

	conv_lossۊ�<�ŦY        )��P	&J
O���A�(*

	conv_loss�=�;�-        )��P	{
O���A�(*

	conv_lossސ�<�-�        )��P	L�
O���A�(*

	conv_loss��<gZ        )��P	�
O���A�(*

	conv_losseE�<���        )��P	#O���A�(*

	conv_loss���<�@i�        )��P	5=O���A�(*

	conv_loss��<�Jk�        )��P	c�O���A�(*

	conv_loss���<B��        )��P	A�O���A�(*

	conv_lossu=�Y;        )��P	E�O���A�(*

	conv_lossꗾ<���        )��P	LO���A�(*

	conv_loss���<]�3~        )��P	�BO���A�(*

	conv_loss���<�(r�        )��P	=sO���A�(*

	conv_loss���<�� �        )��P	��O���A�(*

	conv_loss�l�<���w        )��P	��O���A�(*

	conv_loss9-=�X        )��P	�O���A�(*

	conv_loss��<-CA        )��P	�3O���A�(*

	conv_loss���<�z�        )��P	fO���A�(*

	conv_loss {�<cѶ�        )��P	'�O���A�(*

	conv_loss�<�"�$        )��P	�O���A�(*

	conv_loss�M=��        )��P	��O���A�(*

	conv_loss��=>��w        )��P	�'O���A�(*

	conv_loss�m�<�N�        )��P	%YO���A�(*

	conv_lossw��<��_�        )��P	ȋO���A�(*

	conv_loss,�=���?        )��P	��O���A�(*

	conv_loss�S�<�8mb        )��P	��O���A�(*

	conv_loss|�}<��'        )��P	�O���A�(*

	conv_lossD��<���r        )��P	KKO���A�(*

	conv_lossH��<@��        )��P	�zO���A�(*

	conv_lossF?�<zL��        )��P	
�O���A�(*

	conv_loss�)�<��        )��P	�O���A�(*

	conv_lossq�<:��        )��P	O���A�(*

	conv_loss�	=_Q�*        )��P	!LO���A�(*

	conv_loss���<+��        )��P	�O���A�(*

	conv_loss�E�<��        )��P	��O���A�(*

	conv_loss�s�<C�q�        )��P	��O���A�(*

	conv_lossX��<d��~        )��P	�O���A�(*

	conv_lossZ��<�Ύ!        )��P	�TO���A�(*

	conv_loss�{=�A�        )��P	��O���A�(*

	conv_loss?�#=�� �        )��P	O�O���A�(*

	conv_loss�6=�\        )��P	��O���A�(*

	conv_loss1R<c�ط        )��P	�)O���A�(*

	conv_loss�ˀ<ֳ�]        )��P	�XO���A�(*

	conv_loss��<���        )��P	W�O���A�(*

	conv_loss���<V��        )��P	s�O���A�(*

	conv_loss!_�<-D}g        )��P	��O���A�(*

	conv_lossݥ�<A���        )��P	�.O���A�(*

	conv_loss>W�<ɲg        )��P	$`O���A�(*

	conv_loss�X=��        )��P	�O���A�(*

	conv_loss1�	=<��        )��P	��O���A�(*

	conv_loss)9�<e�<        )��P	��O���A�(*

	conv_lossv��<a�o�        )��P	0O���A�(*

	conv_loss<��<����        )��P	�iO���A�(*

	conv_loss/�
=P�        )��P	��O���A�(*

	conv_loss�\�<Z9H        )��P	��O���A�(*

	conv_loss�G�<��*         )��P	MO���A�(*

	conv_loss��<�f�/        )��P	�NO���A�(*

	conv_loss=�A        )��P	�O���A�(*

	conv_loss{|�<�d�F        )��P	m�O���A�(*

	conv_lossdI�<�F�        )��P	��O���A�(*

	conv_loss��<��        )��P	j?O���A�(*

	conv_loss���<�$��        )��P	2uO���A�(*

	conv_loss�l�<E�O7        )��P	�O���A�(*

	conv_loss�N�<DF<�        )��P	2�O���A�(*

	conv_loss���<^�5�        )��P	�O���A�(*

	conv_losse-�=zAĊ        )��P	qEO���A�(*

	conv_losshC�<�O��        )��P	wO���A�(*

	conv_loss���<�ݒ�        )��P	N�O���A�(*

	conv_loss,�<���        )��P	��O���A�(*

	conv_loss���<�0�u        )��P	>!O���A�(*

	conv_loss�4�<�g        )��P	�XO���A�(*

	conv_loss��<���a        )��P	C�O���A�(*

	conv_loss)2�<���K        )��P	��O���A�(*

	conv_loss/�=��dg        )��P	��O���A�(*

	conv_loss��<+Q�2        )��P	+O���A�(*

	conv_loss8��<�)w0        )��P	LO���A�(*

	conv_loss��<�ܦ        )��P	�{O���A�(*

	conv_loss��=w��F        )��P	
�O���A�(*

	conv_loss���<�k�        )��P	J�O���A�(*

	conv_loss�<�W2        )��P	�kO���A�(*

	conv_loss��=	�B        )��P	��O���A�(*

	conv_loss�r�<�4�f        )��P	��O���A�(*

	conv_loss���<@A��        )��P	�O���A�(*

	conv_losss��<r>�        )��P	t2O���A�(*

	conv_loss��<��4�        )��P	�eO���A�(*

	conv_loss�͓<� ��        )��P	��O���A�(*

	conv_lossUC<�;�        )��P	��O���A�(*

	conv_loss���<ƃ�        )��P	�O���A�(*

	conv_lossR�=�@�        )��P	�JO���A�(*

	conv_loss���<3��        )��P	1|O���A�(*

	conv_loss��<���        )��P	%�O���A�(*

	conv_loss�g	<��
n        )��P	�O���A�(*

	conv_loss|�=���        )��P	 O���A�(*

	conv_lossy.=�GP�        )��P	 RO���A�(*

	conv_lossu��<�w�S        )��P	9�O���A�(*

	conv_loss�H�<�k��        )��P	�O���A�(*

	conv_loss�t)=e�#g        )��P	��O���A�(*

	conv_lossm��<R�mS        )��P	uO���A�(*

	conv_loss���<����        )��P	LIO���A�(*

	conv_lossvt<|NX        )��P	[�O���A�(*

	conv_loss�ª<�        )��P	��O���A�(*

	conv_loss��=����        )��P	��O���A�(*

	conv_loss$��<�,�        )��P	9 O���A�(*

	conv_loss�F�<%�N        )��P	�k O���A�(*

	conv_loss�� =6���        )��P	�� O���A�(*

	conv_loss�;=��r�        )��P	l� O���A�(*

	conv_loss�6<o^�        )��P	�!O���A�(*

	conv_loss��<Oh�        )��P	K!O���A�(*

	conv_losso�$=�Z�        )��P	�!O���A�(*

	conv_loss���<�&(        )��P	n�!O���A�(*

	conv_lossO�C<s�k        )��P	��!O���A�(*

	conv_loss|Q�<��4        )��P	-%"O���A�(*

	conv_loss�=��VD        )��P	�V"O���A�(*

	conv_lossÕ<���        )��P	Έ"O���A�(*

	conv_loss[��<XtH2        )��P	��"O���A�(*

	conv_lossJK>=�pr�        )��P	�1#O���A�(*

	conv_loss�Ѯ<��<�        )��P	�e#O���A�(*

	conv_lossS��<�r��        )��P	<�#O���A�(*

	conv_losse<�S        )��P	�#O���A�(*

	conv_lossT�<1��u        )��P	
$O���A�(*

	conv_lossu��<Ƶ�A        )��P	N8$O���A�(*

	conv_loss
o�<����        )��P	Ih$O���A�(*

	conv_loss�7<͏�"        )��P	��$O���A�(*

	conv_loss p(=�ݿ�        )��P	�$O���A�(*

	conv_loss<�<w�"        )��P	�%O���A�(*

	conv_loss�	�<O��        )��P	h>%O���A�(*

	conv_loss�j�<~�Y        )��P	�q%O���A�)*

	conv_loss�d�<
L��        )��P	ʤ%O���A�)*

	conv_loss�K	=���_        )��P	4�%O���A�)*

	conv_loss��<L�5
        )��P	�+&O���A�)*

	conv_lossTx=rٹ�        )��P	^a&O���A�)*

	conv_loss��6=����        )��P	5�&O���A�)*

	conv_loss�b�<���        )��P	��&O���A�)*

	conv_loss3=����        )��P	^'O���A�)*

	conv_loss&��<�[i        )��P	J'O���A�)*

	conv_loss��<br�        )��P	�~'O���A�)*

	conv_loss��l=-�`�        )��P	L�'O���A�)*

	conv_loss��<�P��        )��P	>�'O���A�)*

	conv_loss@�<�c�o        )��P	�((O���A�)*

	conv_loss��= �6        )��P	�Y(O���A�)*

	conv_lossO =3*[        )��P	I�(O���A�)*

	conv_loss��<g�TM        )��P	
�(O���A�)*

	conv_loss6��<�Z�g        )��P	��(O���A�)*

	conv_lossZZ=�G�w        )��P	�$)O���A�)*

	conv_loss���<sr��        )��P	DY)O���A�)*

	conv_loss���<\�        )��P	Ԋ)O���A�)*

	conv_loss���<�mCj        )��P	�)O���A�)*

	conv_losss�<�"��        )��P	~�)O���A�)*

	conv_loss��<C=5�        )��P	(*O���A�)*

	conv_loss�R�<D�j~        )��P	Z*O���A�)*

	conv_lossd9�<g�cn        )��P	K�*O���A�)*

	conv_lossy\�<m���        )��P	��*O���A�)*

	conv_lossb��<�~�        )��P	H�*O���A�)*

	conv_loss,�=Eŵ        )��P	�+O���A�)*

	conv_loss��=���l        )��P	G+O���A�)*

	conv_loss-�=xO	        )��P	�v+O���A�)*

	conv_loss�k�<�P��        )��P	k�+O���A�)*

	conv_lossF��<�v#8        )��P	$�+O���A�)*

	conv_lossU�<�t        )��P	,O���A�)*

	conv_loss�>=�f��        )��P	7,O���A�)*

	conv_lossN[�<��]�        )��P	>i,O���A�)*

	conv_loss�-l<�^�/        )��P	ޙ,O���A�)*

	conv_loss���<��^@        )��P	��,O���A�)*

	conv_loss��=|@�b        )��P	A�,O���A�)*

	conv_lossz�<��n'        )��P	,(-O���A�)*

	conv_lossM�<��p�        )��P	�Y-O���A�)*

	conv_loss��<��        )��P	F�-O���A�)*

	conv_loss� =~�        )��P	o�-O���A�)*

	conv_lossi<��        )��P	��-O���A�)*

	conv_lossN��<0�G�        )��P	`.O���A�)*

	conv_loss�j�<��^        )��P	�A.O���A�)*

	conv_loss���<.
        )��P	p.O���A�)*

	conv_loss�E=���B        )��P	�.O���A�)*

	conv_lossp��<tP�U        )��P	��.O���A�)*

	conv_lossR�Z<Q$�        )��P	U/O���A�)*

	conv_loss�~<H��        )��P	�2/O���A�)*

	conv_loss��<$��+        )��P	|c/O���A�)*

	conv_loss��<Op6v        )��P	,�/O���A�)*

	conv_loss"��<�}�        )��P	`�/O���A�)*

	conv_loss��-=���u        )��P	40O���A�)*

	conv_loss�ƻ<��H�        )��P	!40O���A�)*

	conv_loss[�<���        )��P	�b0O���A�)*

	conv_loss.�=��,E        )��P	o�0O���A�)*

	conv_loss5��<N�M        )��P	��0O���A�)*

	conv_loss#�<��[        )��P	��0O���A�)*

	conv_loss��=(Xd,        )��P	T=1O���A�)*

	conv_loss���<����        )��P	Dt1O���A�)*

	conv_loss���<6jaH        )��P	��1O���A�)*

	conv_loss�4�<7��]        )��P	_�1O���A�)*

	conv_loss��X<<AK        )��P	�%2O���A�)*

	conv_loss�=aُ        )��P	�X2O���A�)*

	conv_loss��k<W�h        )��P	��2O���A�)*

	conv_loss>� =8fG        )��P	ĸ2O���A�)*

	conv_loss��<s��        )��P	��2O���A�)*

	conv_loss��<���s        )��P	�3O���A�)*

	conv_loss�=��H�        )��P	F3O���A�)*

	conv_loss��<���	        )��P	@x3O���A�)*

	conv_loss��<�e[        )��P	�3O���A�)*

	conv_loss�=5���        )��P	t�3O���A�)*

	conv_loss�a2<��?�        )��P	�4O���A�)*

	conv_loss>y�<ĪH�        )��P	�E4O���A�)*

	conv_lossB�<��q�        )��P	2u4O���A�)*

	conv_lossLV�<���d        )��P	��4O���A�)*

	conv_loss�=j��        )��P	��4O���A�)*

	conv_loss/Z�<Ly\        )��P	�5O���A�)*

	conv_loss��<O���        )��P	�45O���A�)*

	conv_lossP��<C��q        )��P	�d5O���A�)*

	conv_loss�ɣ<��g        )��P	�5O���A�)*

	conv_loss��D<~{20        )��P	��5O���A�)*

	conv_loss�V�<0c(        )��P	I�5O���A�)*

	conv_loss=��<��=        )��P	+ 6O���A�)*

	conv_loss*�<�
        )��P	�O6O���A�)*

	conv_loss��=@        )��P	�6O���A�)*

	conv_lossUJ=�A��        )��P	C�6O���A�)*

	conv_loss���<��        )��P	��6O���A�)*

	conv_loss�x�<P�6�        )��P	�7O���A�)*

	conv_loss�X=3��q        )��P	DG7O���A�)*

	conv_loss��<��n        )��P	ew7O���A�)*

	conv_loss��<���        )��P	��7O���A�)*

	conv_lossg��<h���        )��P	%�7O���A�)*

	conv_lossG.=�Z�%        )��P	H8O���A�)*

	conv_loss�Z^<���        )��P	�88O���A�)*

	conv_loss-�<u�}        )��P	i8O���A�)*

	conv_loss��<�i�        )��P	�8O���A�)*

	conv_loss�1+=5�        )��P	��8O���A�)*

	conv_loss�0�<ٓ1#        )��P	��8O���A�)*

	conv_loss7�%=)���        )��P	9%9O���A�)*

	conv_loss�/"=�=��        )��P	�T9O���A�)*

	conv_loss�q8=��q        )��P	ł9O���A�)*

	conv_lossjZ�<:PH�        )��P	q�9O���A�)*

	conv_loss��<��D        )��P	\�9O���A�)*

	conv_loss� =)	K        )��P	�:O���A�)*

	conv_losse=~��        )��P	�N:O���A�)*

	conv_lossӾ<ꅨ�        )��P	ׂ:O���A�)*

	conv_loss	P&=�q�        )��P	ȶ:O���A�)*

	conv_loss��<�1�5        )��P	��:O���A�)*

	conv_loss��<8���        )��P	n;O���A�)*

	conv_loss�<���U        )��P	�G;O���A�)*

	conv_loss�	�<�;/        )��P	.�;O���A�)*

	conv_loss�$�<��_        )��P	��;O���A�)*

	conv_loss7��<���        )��P	��;O���A�)*

	conv_loss�Se=b�T�        )��P	$<O���A�)*

	conv_loss=Ά=�_�`        )��P	S<O���A�)*

	conv_lossQ�<�ç        )��P	L�<O���A�)*

	conv_loss?O�<��z        )��P	;�<O���A�)*

	conv_loss�x=u���        )��P	��<O���A�)*

	conv_loss�[ =O�'�        )��P	�=O���A�)*

	conv_loss1�=��r�        )��P	�N=O���A�)*

	conv_loss�?�<τ�        )��P	k=O���A�)*

	conv_lossv��<(@�e        )��P	?�=O���A�)*

	conv_loss�-�<���        )��P	e�=O���A�)*

	conv_lossm[<+}��        )��P	�>O���A�)*

	conv_lossid=�hd�        )��P	�Q>O���A�)*

	conv_loss�.�<�&�=        )��P	��>O���A�)*

	conv_loss��<���        )��P	��>O���A�**

	conv_loss���<���        )��P	��>O���A�**

	conv_loss/�=�1c        )��P	?O���A�**

	conv_loss�?�<�ɽ�        )��P	N@?O���A�**

	conv_loss���<Nd�        )��P	o?O���A�**

	conv_losss�<�3��        )��P	C�?O���A�**

	conv_loss���<IG�h        )��P	��?O���A�**

	conv_lossa�<a2I�        )��P	 @O���A�**

	conv_loss	��<4N        )��P	�.@O���A�**

	conv_loss$t�<�!r.        )��P	`@O���A�**

	conv_loss�:�<'I#�        )��P	�@O���A�**

	conv_lossj�<]*t�        )��P	.�@O���A�**

	conv_loss"(�<qpQ}        )��P	�@O���A�**

	conv_loss��<�!Ֆ        )��P	O"AO���A�**

	conv_loss5�<YC\        )��P	�RAO���A�**

	conv_losss=��e        )��P	āAO���A�**

	conv_loss���<j=2        )��P	Q�AO���A�**

	conv_loss�c�<9K�        )��P	3�AO���A�**

	conv_loss��~<Amz�        )��P	&BO���A�**

	conv_loss_ۯ<b.�N        )��P	@[BO���A�**

	conv_lossH
�<e}�        )��P	2�BO���A�**

	conv_loss�)�<����        )��P	��BO���A�**

	conv_loss�z�<?ݽ)        )��P	�CO���A�**

	conv_loss���<)+�        )��P	�OCO���A�**

	conv_loss��<���p        )��P	|�CO���A�**

	conv_loss�_2=�&��        )��P	@NEO���A�**

	conv_loss��<��        )��P	�EO���A�**

	conv_loss��<�IF�        )��P	��EO���A�**

	conv_loss5uG=�=�i        )��P	��EO���A�**

	conv_lossy҇<�qe        )��P	*!FO���A�**

	conv_loss���<F�?         )��P	�RFO���A�**

	conv_loss�<�<`�        )��P	�FO���A�**

	conv_loss@�8=�	        )��P	��FO���A�**

	conv_lossGe�<�A�        )��P	��FO���A�**

	conv_loss��=�f�        )��P	�%GO���A�**

	conv_losske<T�O        )��P	lVGO���A�**

	conv_lossǏ�<�"�        )��P	a�GO���A�**

	conv_loss:4�<����        )��P	W�GO���A�**

	conv_loss�*�<D9�        )��P	�GO���A�**

	conv_loss���<��C�        )��P	0HO���A�**

	conv_loss�^I<�,7        )��P	�_HO���A�**

	conv_loss�� =�B��        )��P	g�HO���A�**

	conv_loss ��<����        )��P	^�HO���A�**

	conv_lossC�=�{@Z        )��P	�IO���A�**

	conv_lossĕ<��?�        )��P	�CIO���A�**

	conv_loss���<W�^"        )��P	�qIO���A�**

	conv_loss�.=���j        )��P	�IO���A�**

	conv_loss?�<�Hu�        )��P	o�IO���A�**

	conv_loss.��<4$��        )��P	JO���A�**

	conv_loss��<C�h        )��P	�EJO���A�**

	conv_loss읪<Pg��        )��P	�|JO���A�**

	conv_loss���<D�        )��P	��JO���A�**

	conv_loss4X	=����        )��P	�JO���A�**

	conv_loss54$<����        )��P	KO���A�**

	conv_lossP�<�"]        )��P	NKO���A�**

	conv_loss���<]�H        )��P	��KO���A�**

	conv_loss:�<̏��        )��P	ٯKO���A�**

	conv_lossX�<��x        )��P	�KO���A�**

	conv_lossEX�<Ok͸        )��P	LO���A�**

	conv_lossd�<���9        )��P	�DLO���A�**

	conv_lossK�%=�PH        )��P	�uLO���A�**

	conv_loss�Ѕ< �7        )��P	E�LO���A�**

	conv_lossF�<��7        )��P	��LO���A�**

	conv_loss ;=��D3        )��P	pMO���A�**

	conv_loss�=L��6        )��P	E;MO���A�**

	conv_loss���<iƛ        )��P	EjMO���A�**

	conv_loss1C{<�p�x        )��P	b�MO���A�**

	conv_loss���<���[        )��P	��MO���A�**

	conv_loss��4<����        )��P	2�MO���A�**

	conv_loss��<�/��        )��P	�'NO���A�**

	conv_loss���<�ZuF        )��P	�ZNO���A�**

	conv_lossdY�<�w,�        )��P	i�NO���A�**

	conv_loss���<��ÿ        )��P	��NO���A�**

	conv_lossO�<��ZK        )��P	��NO���A�**

	conv_loss��<aq�        )��P	�"OO���A�**

	conv_loss��<CY_5        )��P	sfOO���A�**

	conv_lossk!�<�u�2        )��P	[�OO���A�**

	conv_lossn��<���U        )��P	��OO���A�**

	conv_lossZ�=v��        )��P	-PO���A�**

	conv_lossE>�<��;�        )��P	4PO���A�**

	conv_loss��<�x�"        )��P	�dPO���A�**

	conv_loss��Q=�b��        )��P	f�PO���A�**

	conv_lossؠ�<����        )��P	�PO���A�**

	conv_losss~�<�&�$        )��P	QO���A�**

	conv_loss�i=��        )��P	�>QO���A�**

	conv_loss���<�ܲ�        )��P	�nQO���A�**

	conv_loss��o<�̝�        )��P	_�QO���A�**

	conv_loss�%�<�%�        )��P	��QO���A�**

	conv_loss
�=�O��        )��P	�RO���A�**

	conv_loss��A=KfaO        )��P	�LRO���A�**

	conv_loss���<���        )��P	��RO���A�**

	conv_loss�z=u']�        )��P	�RO���A�**

	conv_loss���<F��        )��P	��RO���A�**

	conv_loss>��<���        )��P	SO���A�**

	conv_loss�ܴ<���T        )��P		CSO���A�**

	conv_loss���<�,        )��P	�tSO���A�**

	conv_loss�=>�g        )��P	�SO���A�**

	conv_loss���<p�dz        )��P	t�SO���A�**

	conv_loss��=�~Q        )��P	TO���A�**

	conv_loss��<݂��        )��P	�7TO���A�**

	conv_lossu�<��i�        )��P	�iTO���A�**

	conv_loss5R
=��.        )��P	:�TO���A�**

	conv_loss�i�<��?0        )��P	��TO���A�**

	conv_loss��<���        )��P	��TO���A�**

	conv_lossPI�<]�         )��P	�3UO���A�**

	conv_loss�n<�0��        )��P	�fUO���A�**

	conv_loss~�<�x%        )��P	��UO���A�**

	conv_loss��<!���        )��P	`�UO���A�**

	conv_loss^=�8*�        )��P	�VO���A�**

	conv_losss��<�ǟS        )��P	 8VO���A�**

	conv_loss{�z<}���        )��P	jVO���A�**

	conv_loss���<��        )��P	w�VO���A�**

	conv_loss��<��t        )��P	j�VO���A�**

	conv_lossg��<vx��        )��P	�WO���A�**

	conv_loss���<�{p        )��P	�4WO���A�**

	conv_losss��<�7Ad        )��P	�gWO���A�**

	conv_losst��<v�_�        )��P	D�WO���A�**

	conv_loss$Y�<���        )��P	K�WO���A�**

	conv_loss��<�?"        )��P	��WO���A�**

	conv_loss�}�<.�|�        )��P	D2XO���A�**

	conv_loss!6<%�~        )��P	�eXO���A�**

	conv_loss���<�H{�        )��P	�XO���A�**

	conv_loss���<r)�_        )��P	5�XO���A�**

	conv_lossX.m<"��q        )��P	��XO���A�**

	conv_loss�b$=�'��        )��P	J�]O���A�**

	conv_lossZ�$=;�-        )��P	q�]O���A�**

	conv_loss�<_Q>�        )��P	�^O���A�**

	conv_loss��=C'��        )��P	�3^O���A�**

	conv_lossҙ�<��\�        )��P	j^O���A�+*

	conv_loss�U�<�#��        )��P	U�^O���A�+*

	conv_loss?��<Z��        )��P	�^O���A�+*

	conv_loss�]�<�Ĉ-        )��P	_O���A�+*

	conv_loss��<�UP         )��P	�Q_O���A�+*

	conv_loss(ן<��|        )��P	��_O���A�+*

	conv_loss��<iŖ�        )��P	��_O���A�+*

	conv_loss)
=R>7�        )��P	_�_O���A�+*

	conv_loss/o�<[�4X        )��P	p#`O���A�+*

	conv_loss= �<���        )��P	�S`O���A�+*

	conv_lossq]�<�^}H        )��P	��`O���A�+*

	conv_loss>b=nU�        )��P	��`O���A�+*

	conv_loss@�<F�.�        )��P	z�`O���A�+*

	conv_lossc=؀��        )��P	�aO���A�+*

	conv_loss��<2�]w        )��P	�RaO���A�+*

	conv_loss挹<�� �        )��P	`�aO���A�+*

	conv_lossZ��<Of"�        )��P	Z�aO���A�+*

	conv_loss�y�<l��g        )��P	w�aO���A�+*

	conv_lossPj�<W�?        )��P	cbO���A�+*

	conv_loss�5�<�\\        )��P	�GbO���A�+*

	conv_loss��=%�        )��P	%xbO���A�+*

	conv_loss\��<Z7�        )��P	u�bO���A�+*

	conv_loss4I=J�V        )��P	��bO���A�+*

	conv_loss���<`[y�        )��P	=
cO���A�+*

	conv_loss�7�<��m        )��P	�9cO���A�+*

	conv_loss?L�<��         )��P	djcO���A�+*

	conv_loss	��<g��        )��P	��cO���A�+*

	conv_loss�q�<��        )��P	�cO���A�+*

	conv_lossiO.<�ux*        )��P	��cO���A�+*

	conv_loss�چ<�~�
        )��P	�'dO���A�+*

	conv_loss�!�<��        )��P	QYdO���A�+*

	conv_loss#Ja<���        )��P	=�dO���A�+*

	conv_loss��<���7        )��P	��dO���A�+*

	conv_loss�R�<~��        )��P	�dO���A�+*

	conv_loss��4=Th��        )��P	\eO���A�+*

	conv_loss�D�<�6%�        )��P	cHeO���A�+*

	conv_loss��X<ͳ�J        )��P	�zeO���A�+*

	conv_loss' _<f��l        )��P	$�eO���A�+*

	conv_loss�޶<@��        )��P	+�eO���A�+*

	conv_loss]��<��)�        )��P	�fO���A�+*

	conv_loss(�.<-$f�        )��P	�@fO���A�+*

	conv_loss���<`�-�        )��P	�ofO���A�+*

	conv_loss�R=��8�        )��P	�fO���A�+*

	conv_loss�J<�3�        )��P	��fO���A�+*

	conv_loss���<�kC        )��P	��fO���A�+*

	conv_lossw=��mU        )��P	1,gO���A�+*

	conv_lossw%�<�~        )��P	�ZgO���A�+*

	conv_loss��<W�S        )��P	�gO���A�+*

	conv_loss�w=.N�#        )��P	��gO���A�+*

	conv_lossk��<���        )��P	��gO���A�+*

	conv_loss#�=��7        )��P	�/hO���A�+*

	conv_lossUw�<���        )��P	�chO���A�+*

	conv_loss_1�<9��~        )��P	��hO���A�+*

	conv_loss2;�<~�2g        )��P	��hO���A�+*

	conv_loss�A�<�Y�        )��P	�iO���A�+*

	conv_loss�Td<X��l        )��P	}JiO���A�+*

	conv_loss�<�$        )��P	{iO���A�+*

	conv_loss�b�<�͢i        )��P	ѭiO���A�+*

	conv_loss�ּ<z�D        )��P	��iO���A�+*

	conv_loss���<PmD        )��P	�jO���A�+*

	conv_loss�p�<�"[�        )��P	�CjO���A�+*

	conv_loss�`<���!        )��P	tjO���A�+*

	conv_loss�<65&�        )��P	r�jO���A�+*

	conv_loss���<L0�{        )��P	C�jO���A�+*

	conv_loss!��<2'�        )��P	�kO���A�+*

	conv_losst-�<62�        )��P	�9kO���A�+*

	conv_losskO}<��y        )��P	LtkO���A�+*

	conv_loss��<m,�        )��P	w�kO���A�+*

	conv_loss�]�<�        )��P	J�kO���A�+*

	conv_loss��,<O�}        )��P	�lO���A�+*

	conv_loss؏�<*=        )��P	�4lO���A�+*

	conv_loss���<E1Z�        )��P	dlO���A�+*

	conv_lossV��<����        )��P	�lO���A�+*

	conv_loss.#�<g�o        )��P	�lO���A�+*

	conv_loss[��<��        )��P	��lO���A�+*

	conv_loss��#=`��        )��P	p(mO���A�+*

	conv_loss�1�<���        )��P	�YmO���A�+*

	conv_lossS��<��H        )��P	��mO���A�+*

	conv_loss���<���"        )��P	��mO���A�+*

	conv_lossp�<x��        )��P	�mO���A�+*

	conv_loss�c<���        )��P	�nO���A�+*

	conv_loss���<�QB�        )��P	�AnO���A�+*

	conv_lossX��<��c        )��P	�onO���A�+*

	conv_loss��R<'�        )��P	ޟnO���A�+*

	conv_loss0=	(�        )��P	4�nO���A�+*

	conv_loss�Y=4�c�        )��P	�oO���A�+*

	conv_loss���<�PcP        )��P	�2oO���A�+*

	conv_loss�i=�˘
        )��P	�boO���A�+*

	conv_loss/��<�d��        )��P	:�oO���A�+*

	conv_loss�U<�8�        )��P	j�oO���A�+*

	conv_loss'2�<��c�        )��P	G�oO���A�+*

	conv_loss]��<|?h�        )��P	�$pO���A�+*

	conv_loss�Γ<`|(/        )��P	rSpO���A�+*

	conv_loss�F3=:�+        )��P	*�pO���A�+*

	conv_loss�=k���        )��P	M�pO���A�+*

	conv_loss��<����        )��P	��pO���A�+*

	conv_loss�;�< �8l        )��P	�qO���A�+*

	conv_loss��>=��z�        )��P	o�rO���A�+*

	conv_loss��<)5&�        )��P	��rO���A�+*

	conv_loss���<��:�        )��P	�sO���A�+*

	conv_lossru�<�)2@        )��P	�=sO���A�+*

	conv_loss��=Ax�        )��P	AtsO���A�+*

	conv_lossI�b<0o         )��P	��sO���A�+*

	conv_loss(�=Ã        )��P	��sO���A�+*

	conv_loss+[�<�Ԉ�        )��P	�tO���A�+*

	conv_loss��<M�4        )��P	BtO���A�+*

	conv_lossՓ�<)��Q        )��P	bvtO���A�+*

	conv_lossX4�<�͂        )��P	�tO���A�+*

	conv_losss��<C��0        )��P	4�tO���A�+*

	conv_losskY}<�D        )��P	�uO���A�+*

	conv_lossp)�<+r�        )��P	w?uO���A�+*

	conv_loss�r=@VJy        )��P	�vuO���A�+*

	conv_lossVT�<�'=7        )��P	5�uO���A�+*

	conv_loss�	=Vd��        )��P	��uO���A�+*

	conv_loss�>�<E|        )��P	�vO���A�+*

	conv_lossM�<+���        )��P	�?vO���A�+*

	conv_loss�ޑ<��        )��P	svO���A�+*

	conv_lossx�.=�>        )��P	c�vO���A�+*

	conv_loss� �<�ܺT        )��P	�vO���A�+*

	conv_loss�q<x�+        )��P	wO���A�+*

	conv_loss#�<o�%�        )��P	Z5wO���A�+*

	conv_lossYc8<��(�        )��P	$hwO���A�+*

	conv_loss"K�<�`M        )��P	��wO���A�+*

	conv_lossW6=���(        )��P	��wO���A�+*

	conv_lossb�u<����        )��P	 xO���A�+*

	conv_loss/�=@;��        )��P	�2xO���A�+*

	conv_loss��
=��G        )��P	'exO���A�+*

	conv_loss���<�fv        )��P	d�xO���A�+*

	conv_loss���<�R�        )��P	��xO���A�,*

	conv_lossHu�<S�;        )��P	��xO���A�,*

	conv_loss`�<�p�x        )��P	y0yO���A�,*

	conv_loss�lG<�z�c        )��P	ZfyO���A�,*

	conv_loss�]=-zv�        )��P	H�yO���A�,*

	conv_loss�{�<�ُ�        )��P	��yO���A�,*

	conv_loss�j�<��        )��P	� zO���A�,*

	conv_lossl�<���;        )��P	^5zO���A�,*

	conv_loss��=��d        )��P	�lzO���A�,*

	conv_loss���<���P        )��P	ϠzO���A�,*

	conv_lossO�=��&�        )��P	q�zO���A�,*

	conv_loss==����        )��P	�{O���A�,*

	conv_loss[̀<�{�k        )��P	�@{O���A�,*

	conv_lossj�<�,͐        )��P	�v{O���A�,*

	conv_loss�5�<ev{        )��P	�{O���A�,*

	conv_loss�K:=�a�        )��P	��{O���A�,*

	conv_lossj��<�IY�        )��P	�|O���A�,*

	conv_loss��x<�� w        )��P	4G|O���A�,*

	conv_loss}�u<��        )��P	2z|O���A�,*

	conv_loss��=O;Q        )��P	��|O���A�,*

	conv_loss�_=��        )��P	b�|O���A�,*

	conv_loss~�o<sHb        )��P	",}O���A�,*

	conv_loss��<(d�        )��P	�a}O���A�,*

	conv_lossT\�<�%|�        )��P	4�}O���A�,*

	conv_loss?E=��\�        )��P	�}O���A�,*

	conv_loss�n�<�;�\        )��P	�~O���A�,*

	conv_loss�TD=!M:        )��P	�8~O���A�,*

	conv_lossଦ<y���        )��P	�q~O���A�,*

	conv_lossY��<��0�        )��P	ӭ~O���A�,*

	conv_loss���<��Y�        )��P	��~O���A�,*

	conv_loss��<��v        )��P	%O���A�,*

	conv_loss�y�<_���        )��P	IO���A�,*

	conv_loss�E�<��        )��P	ČO���A�,*

	conv_lossYh<���        )��P	��O���A�,*

	conv_loss�<�S>�        )��P	"�O���A�,*

	conv_loss�<�k�~        )��P	�*�O���A�,*

	conv_loss�j<yX4        )��P	�^�O���A�,*

	conv_loss°�<���A        )��P	2��O���A�,*

	conv_loss�{�<���        )��P	�ŀO���A�,*

	conv_losss��<����        )��P	���O���A�,*

	conv_loss�{�<�j4        )��P	�2�O���A�,*

	conv_loss���<��I        )��P	�e�O���A�,*

	conv_loss	1<wA��        )��P	���O���A�,*

	conv_loss���<Ai�l        )��P	�΁O���A�,*

	conv_lossIr=�b�        )��P	��O���A�,*

	conv_loss���<D�7�        )��P	�5�O���A�,*

	conv_loss�%=䕢Q        )��P	i�O���A�,*

	conv_loss�(�<�sY�        )��P	���O���A�,*

	conv_lossC=�]�|        )��P	FтO���A�,*

	conv_loss)m<#�:�        )��P	t�O���A�,*

	conv_loss���<x�,        )��P	29�O���A�,*

	conv_loss1&�<P��        )��P	�l�O���A�,*

	conv_loss�C�<s�ۡ        )��P	3��O���A�,*

	conv_loss�ʯ<��?        )��P	�ՃO���A�,*

	conv_loss���<�X�        )��P	�	�O���A�,*

	conv_lossf:=q�K        )��P	�@�O���A�,*

	conv_loss���<���R        )��P	Gt�O���A�,*

	conv_loss�v�<�e�9        )��P	���O���A�,*

	conv_loss�t=<��V�        )��P		ۄO���A�,*

	conv_loss��=���        )��P	�O���A�,*

	conv_lossj��<��R�        )��P	)A�O���A�,*

	conv_loss9��<�2�Q        )��P	�r�O���A�,*

	conv_loss%��<�7        )��P	���O���A�,*

	conv_loss��<���        )��P	�؅O���A�,*

	conv_loss�l�<f�.(        )��P	 �O���A�,*

	conv_loss�;~<�謜        )��P	f@�O���A�,*

	conv_loss� =���        )��P	xr�O���A�,*

	conv_loss+��<�i�        )��P	T��O���A�,*

	conv_loss�:�<3a��        )��P	�چO���A�,*

	conv_loss�<`���        )��P	�"�O���A�,*

	conv_lossc{1<�M)        )��P	qV�O���A�,*

	conv_loss��<�X�G        )��P	Ӊ�O���A�,*

	conv_loss�պ<��F        )��P		��O���A�,*

	conv_loss�L�<�$��        )��P	��O���A�,*

	conv_loss)��<�	�        )��P	�(�O���A�,*

	conv_lossJy�<���        )��P	�]�O���A�,*

	conv_loss���<�8	'        )��P	̏�O���A�,*

	conv_loss�A<-��        )��P	�ӈO���A�,*

	conv_loss�=���        )��P	��O���A�,*

	conv_loss�j<N�>�        )��P	@;�O���A�,*

	conv_lossTYX=XJ��        )��P	�n�O���A�,*

	conv_loss8Ԝ<~;��        )��P	���O���A�,*

	conv_loss�=�&�        )��P	��O���A�,*

	conv_loss/I�<|7�        )��P	��O���A�,*

	conv_loss,�=+4��        )��P	�O�O���A�,*

	conv_lossDm�<m��        )��P	}��O���A�,*

	conv_lossM�=�B        )��P	��O���A�,*

	conv_lossÃ)<Gnr8        )��P	{�O���A�,*

	conv_loss�t<;T        )��P	��O���A�,*

	conv_loss�	�<v�        )��P	�T�O���A�,*

	conv_losse6�<�_�<        )��P	���O���A�,*

	conv_loss�x�<�8��        )��P	��O���A�,*

	conv_lossCS=�Ux�        )��P	���O���A�,*

	conv_loss: =��        )��P	9&�O���A�,*

	conv_loss���<Iz��        )��P	�[�O���A�,*

	conv_lossf�s<�SYF        )��P	���O���A�,*

	conv_loss!ֻ<�6i        )��P	�ŌO���A�,*

	conv_loss�M�<��;k        )��P	L��O���A�,*

	conv_loss�{<����        )��P	k*�O���A�,*

	conv_lossտ=l�!,        )��P	�]�O���A�,*

	conv_lossR��<��@r        )��P	א�O���A�,*

	conv_loss�fG<��a�        )��P	�O���A�,*

	conv_loss$�<x{i*        )��P	Q��O���A�,*

	conv_lossǫ�<��        )��P	�(�O���A�,*

	conv_loss,��<-�i        )��P	X[�O���A�,*

	conv_loss���<����        )��P	ύ�O���A�,*

	conv_loss&K�<�#        )��P	ƎO���A�,*

	conv_loss���<a��        )��P	+��O���A�,*

	conv_loss�S=���t        )��P	�1�O���A�,*

	conv_lossV��<Xf7        )��P	qf�O���A�,*

	conv_loss`ȕ<9nS        )��P	O��O���A�,*

	conv_lossS��<b�@�        )��P	gЏO���A�,*

	conv_loss�up<�F�         )��P	}
�O���A�,*

	conv_lossz��<D�a�        )��P	?�O���A�,*

	conv_loss:��<�޻<        )��P	�q�O���A�,*

	conv_loss�'�<�wTV        )��P	馐O���A�,*

	conv_loss/F<ϔ�R        )��P	uِO���A�,*

	conv_lossb��<��H�        )��P	R�O���A�,*

	conv_loss�=�<��g�        )��P	OH�O���A�,*

	conv_loss��=*��        )��P	<��O���A�,*

	conv_loss���<CL�        )��P	�őO���A�,*

	conv_loss��<1�H        )��P	Q��O���A�,*

	conv_lossK�=��XR        )��P	�.�O���A�,*

	conv_loss��<���        )��P	b�O���A�,*

	conv_loss;��<j��        )��P	��O���A�,*

	conv_losss��<�9        )��P	\˒O���A�,*

	conv_loss4��<��v        )��P	n�O���A�,*

	conv_loss�&�<�f�        )��P	KB�O���A�,*

	conv_lossqk<%p,�        )��P	pz�O���A�-*

	conv_loss�� =�2U�        )��P	���O���A�-*

	conv_loss���<;���        )��P	)�O���A�-*

	conv_loss�c=Q��        )��P	o"�O���A�-*

	conv_loss��=��.        )��P	�U�O���A�-*

	conv_loss��=
Oe�        )��P	c��O���A�-*

	conv_loss�=�[y�        )��P	���O���A�-*

	conv_loss$��<^LO        )��P	C�O���A�-*

	conv_loss:�<��1)        )��P	�%�O���A�-*

	conv_loss�(�<�5l�        )��P	�Z�O���A�-*

	conv_loss+�_<׾        )��P	_��O���A�-*

	conv_loss�_�<�dǎ        )��P	�ĕO���A�-*

	conv_lossm�<�\Ð        )��P	���O���A�-*

	conv_loss��<�
Y        )��P	n-�O���A�-*

	conv_loss�;<�/ݙ        )��P	`�O���A�-*

	conv_loss�Ƽ<���_        )��P	���O���A�-*

	conv_loss9�<o�8*        )��P	6ȖO���A�-*

	conv_loss�iv<�^%�        )��P	J��O���A�-*

	conv_loss�K�<����        )��P	�1�O���A�-*

	conv_loss��<��W        )��P	.g�O���A�-*

	conv_lossN&�<��        )��P	���O���A�-*

	conv_loss���<�#4k        )��P	̗O���A�-*

	conv_loss��<���        )��P	V �O���A�-*

	conv_loss@0�<�j��        )��P	3�O���A�-*

	conv_loss�e<�y�        )��P	*e�O���A�-*

	conv_lossm=�*"�        )��P	l��O���A�-*

	conv_losss�<��         )��P	�̘O���A�-*

	conv_loss@�<�Ո�        )��P	|��O���A�-*

	conv_lossV��<��B(        )��P	92�O���A�-*

	conv_loss��<�]�        )��P	�g�O���A�-*

	conv_loss\�=3���        )��P	K��O���A�-*

	conv_loss!y�<*yK        )��P	�͙O���A�-*

	conv_lossc��<Yj4~        )��P	��O���A�-*

	conv_loss��<�Y �        )��P	�7�O���A�-*

	conv_loss���<;���        )��P	�k�O���A�-*

	conv_loss⿻<���        )��P	$��O���A�-*

	conv_lossq�<�*�        )��P	�ӚO���A�-*

	conv_loss�<qM        )��P	 �O���A�-*

	conv_loss��<�Hx�        )��P	�;�O���A�-*

	conv_lossº�<&��        )��P	fo�O���A�-*

	conv_loss�^�<�W1�        )��P	{��O���A�-*

	conv_loss]�<�8r        )��P	;=�O���A�-*

	conv_loss	��<�Ԝ�        )��P	�p�O���A�-*

	conv_loss䁇<���N        )��P	 ��O���A�-*

	conv_loss�;<b�{�        )��P	�۝O���A�-*

	conv_loss!�C<���        )��P	��O���A�-*

	conv_loss�2�<l��        )��P	KP�O���A�-*

	conv_lossg��<v�+{        )��P	���O���A�-*

	conv_losso�'=(�e�        )��P	���O���A�-*

	conv_lossec<�t�        )��P	7�O���A�-*

	conv_loss=OP=�SP        )��P	i#�O���A�-*

	conv_loss2��<���%        )��P	�]�O���A�-*

	conv_loss�/<�`        )��P	\��O���A�-*

	conv_loss&�3=(3�*        )��P	�ğO���A�-*

	conv_loss�<N�"        )��P	$��O���A�-*

	conv_loss��<&��d        )��P	.2�O���A�-*

	conv_loss��!=ղL*        )��P	�k�O���A�-*

	conv_losss��<46v�        )��P	S��O���A�-*

	conv_loss��<�-��        )��P	C֠O���A�-*

	conv_loss��<�`'        )��P	�
�O���A�-*

	conv_lossu�<��3        )��P	=�O���A�-*

	conv_loss��/=���        )��P	�p�O���A�-*

	conv_lossH�<��@        )��P	���O���A�-*

	conv_loss�'M<<���        )��P	T١O���A�-*

	conv_loss���<n��        )��P	��O���A�-*

	conv_lossU�=���        )��P	X@�O���A�-*

	conv_loss�0�<B�Y�        )��P	�s�O���A�-*

	conv_loss�K�<�ն        )��P	}��O���A�-*

	conv_loss� �<�Ŷ5        )��P	$ޢO���A�-*

	conv_losse��<	�9        )��P	��O���A�-*

	conv_loss��x<�P�        )��P	nC�O���A�-*

	conv_loss�U\<��h	        )��P	x�O���A�-*

	conv_loss��<,@?        )��P	Y��O���A�-*

	conv_loss��=�%G        )��P	ݣO���A�-*

	conv_lossT+<'a{�        )��P	n�O���A�-*

	conv_loss�8�<(ư�        )��P	�D�O���A�-*

	conv_lossn��<Ӱ"@        )��P	Jz�O���A�-*

	conv_loss�@Z<�,~�        )��P	���O���A�-*

	conv_loss��-<�ap�        )��P	'�O���A�-*

	conv_lossjf�<�R��        )��P	��O���A�-*

	conv_loss^�q<�Z��        )��P	�G�O���A�-*

	conv_loss¿�<�,$�        )��P	�y�O���A�-*

	conv_loss ~�<ҥ1�        )��P	�O���A�-*

	conv_loss�P�<u��C        )��P	��O���A�-*

	conv_lossJ#:= Y;        )��P	V�O���A�-*

	conv_loss`��<����        )��P	CG�O���A�-*

	conv_loss(5=��5�        )��P	�z�O���A�-*

	conv_lossa#9<�iv        )��P	'��O���A�-*

	conv_lossEr�<��A�        )��P	�O���A�-*

	conv_losss�=�b��        )��P	��O���A�-*

	conv_loss)6�<蒁6        )��P	�L�O���A�-*

	conv_loss�:�<m��        )��P	Ԑ�O���A�-*

	conv_lossQG�<���        )��P	�ħO���A�-*

	conv_lossp�<�#8c        )��P	��O���A�-*

	conv_lossX�<V�}m        )��P	V+�O���A�-*

	conv_loss���<%�?        )��P	@Z�O���A�-*

	conv_loss|��<��        )��P	e��O���A�-*

	conv_loss���<n���        )��P	��O���A�-*

	conv_loss���<7j��        )��P	-��O���A�-*

	conv_loss��=$�
y        )��P	S#�O���A�-*

	conv_loss��D<�Df�        )��P	*Y�O���A�-*

	conv_loss�	k=��f        )��P	}��O���A�-*

	conv_lossI�B<��&�        )��P	�ĩO���A�-*

	conv_loss��<��F        )��P	w�O���A�-*

	conv_loss�<��fk        )��P	�$�O���A�-*

	conv_loss��=뼿�        )��P	�\�O���A�-*

	conv_loss���<��@        )��P		��O���A�-*

	conv_lossf�T<vr
        )��P	���O���A�-*

	conv_loss��=��C        )��P	Q�O���A�-*

	conv_loss��<�
M        )��P	#!�O���A�-*

	conv_lossJf�<y6�i        )��P	{Q�O���A�-*

	conv_loss/��<nW�        )��P	���O���A�-*

	conv_loss��<���        )��P	q��O���A�-*

	conv_loss`��<n��        )��P	��O���A�-*

	conv_lossα�<..�T        )��P	��O���A�-*

	conv_loss���<2�9        )��P	)H�O���A�-*

	conv_loss���<^�+        )��P	x�O���A�-*

	conv_lossD*=��,        )��P	<��O���A�-*

	conv_loss$�<l)��        )��P	3جO���A�-*

	conv_loss�B=??	P        )��P	��O���A�-*

	conv_loss�M�<�f:�        )��P	\6�O���A�-*

	conv_loss�	�<c��>        )��P	�i�O���A�-*

	conv_loss3�<���i        )��P	Z��O���A�-*

	conv_loss4�=?"        )��P	T˭O���A�-*

	conv_lossʫ�<2���        )��P	b��O���A�-*

	conv_loss��{<�>�        )��P	�)�O���A�-*

	conv_loss��=�Wl�        )��P	�W�O���A�-*

	conv_loss�P�<f��V        )��P	���O���A�-*

	conv_loss�U=����        )��P	Ÿ�O���A�.*

	conv_lossҘw<!�m        )��P	��O���A�.*

	conv_loss2/W<c:}        )��P	��O���A�.*

	conv_loss���<���        )��P	�F�O���A�.*

	conv_loss�S�<	�        )��P	w�O���A�.*

	conv_loss���<���        )��P	}��O���A�.*

	conv_loss��=�j,        )��P	�ԯO���A�.*

	conv_loss��g=�`�l        )��P	B�O���A�.*

	conv_loss[r�<�8�J        )��P	J6�O���A�.*

	conv_lossp�<���        )��P	Ce�O���A�.*

	conv_loss$[�<�Ұ�        )��P	Ɠ�O���A�.*

	conv_loss4��<��I7        )��P	 °O���A�.*

	conv_lossA�=0kw�        )��P	��O���A�.*

	conv_loss"�>=����        )��P	v4�O���A�.*

	conv_losss='c�F        )��P	*f�O���A�.*

	conv_loss���<� n�        )��P	�O���A�.*

	conv_losszC�<ͣک        )��P	_˱O���A�.*

	conv_loss���<���!        )��P	w��O���A�.*

	conv_loss:��<Lq��        )��P	�+�O���A�.*

	conv_losst~�<�U�g        )��P	�[�O���A�.*

	conv_loss8��<t��        )��P	���O���A�.*

	conv_lossw�<�<��        )��P	�ͲO���A�.*

	conv_losso�<o~$        )��P	� �O���A�.*

	conv_loss�HS=����        )��P	�1�O���A�.*

	conv_loss\%�<�L�        )��P	�c�O���A�.*

	conv_loss�D�<ԍ�[        )��P	S��O���A�.*

	conv_loss��8=�'z�        )��P		ҳO���A�.*

	conv_loss���<�߫�        )��P	��O���A�.*

	conv_loss�S =(셐        )��P	%2�O���A�.*

	conv_loss^�<8�j8        )��P	4f�O���A�.*

	conv_loss�h�<^Ca�        )��P	c��O���A�.*

	conv_loss�Ѹ<5�n        )��P	 ɴO���A�.*

	conv_losse��<�W�j        )��P	0��O���A�.*

	conv_loss���<�A�        )��P	�'�O���A�.*

	conv_loss�ζ<�Ύ�        )��P	�[�O���A�.*

	conv_loss仞<C+        )��P	)��O���A�.*

	conv_loss>N�<~�]        )��P	m��O���A�.*

	conv_lossUxk<Q2�~        )��P	,��O���A�.*

	conv_loss���<0�T        )��P	e%�O���A�.*

	conv_loss�g*<U��        )��P	�T�O���A�.*

	conv_lossM|�<���        )��P	+��O���A�.*

	conv_loss���<KVQP        )��P	}��O���A�.*

	conv_loss��<�n        )��P	B�O���A�.*

	conv_loss�J�<�^��        )��P	> �O���A�.*

	conv_loss���<�2�h        )��P	3P�O���A�.*

	conv_loss���<s-�        )��P	�O���A�.*

	conv_losss�<���        )��P	+��O���A�.*

	conv_loss��<����        )��P	P۷O���A�.*

	conv_lossN��<��C�        )��P	�O���A�.*

	conv_lossT�<r�s�        )��P	];�O���A�.*

	conv_lossq<W��m        )��P	Ek�O���A�.*

	conv_loss�F�<�s,�        )��P	К�O���A�.*

	conv_loss�Y=�1��        )��P	WʸO���A�.*

	conv_lossA�t<�c        )��P	��O���A�.*

	conv_loss�=���        )��P	�'�O���A�.*

	conv_lossZ}�<T�_        )��P	�W�O���A�.*

	conv_lossz��<����        )��P	O���A�.*

	conv_lossͨ�<%+3�        )��P	=��O���A�.*

	conv_lossn�<Tw��        )��P	��O���A�.*

	conv_lossn�<�.o!        )��P	��O���A�.*

	conv_lossmU�<s2�n        )��P	�H�O���A�.*

	conv_loss/==�S�v        )��P	�z�O���A�.*

	conv_loss?��<����        )��P	3��O���A�.*

	conv_loss۪<�i�        )��P	v�O���A�.*

	conv_lossX�!=����        )��P	��O���A�.*

	conv_loss��<�A�        )��P	.N�O���A�.*

	conv_lossY;�<���3        )��P	V��O���A�.*

	conv_loss
�<����        )��P	ݹ�O���A�.*

	conv_loss��b<WE�        )��P	S�O���A�.*

	conv_loss�#�<��{�        )��P	�)�O���A�.*

	conv_loss�i�<J@�        )��P	�Z�O���A�.*

	conv_loss�Y�<�D=�        )��P	m��O���A�.*

	conv_loss=��<z
        )��P	�ȼO���A�.*

	conv_loss�8�<�W�        )��P	��O���A�.*

	conv_loss���<>��(        )��P	F1�O���A�.*

	conv_lossU�6=F��        )��P	�b�O���A�.*

	conv_loss�?=��        )��P	���O���A�.*

	conv_loss�v�<@�K        )��P	\ýO���A�.*

	conv_loss��<s��.        )��P	]��O���A�.*

	conv_loss�>	=i�?        )��P	'�O���A�.*

	conv_lossCa�<
k�        )��P	c�O���A�.*

	conv_losss��<3Q�l        )��P	u��O���A�.*

	conv_lossW��<z��        )��P	%ɾO���A�.*

	conv_loss�<
*�        )��P	 ��O���A�.*

	conv_loss���<~�J        )��P	Q+�O���A�.*

	conv_lossU�<Ӛ��        )��P	�\�O���A�.*

	conv_loss�s�<��J�        )��P	Ǎ�O���A�.*

	conv_loss␶<R�         )��P	C��O���A�.*

	conv_lossit�<�>$!        )��P	��O���A�.*

	conv_loss�Y�<4��        )��P	$$�O���A�.*

	conv_lossX�<���U        )��P	�X�O���A�.*

	conv_lossb��<�� �        )��P	8��O���A�.*

	conv_loss�g�<�=�        )��P	���O���A�.*

	conv_loss6d<���        )��P	���O���A�.*

	conv_loss�׶<v�        )��P	'"�O���A�.*

	conv_loss��<K��s        )��P	�S�O���A�.*

	conv_losse�z<˕�        )��P	\��O���A�.*

	conv_loss�t<1���        )��P	��O���A�.*

	conv_loss·=	�1        )��P	��O���A�.*

	conv_loss�<����        )��P	L�O���A�.*

	conv_loss��<�Y�v        )��P	�M�O���A�.*

	conv_loss$�=��         )��P	v�O���A�.*

	conv_loss��<q��g        )��P	Y��O���A�.*

	conv_lossJ��<*�        )��P	z��O���A�.*

	conv_loss���<(܋        )��P	��O���A�.*

	conv_loss�!q<�Ro        )��P	�B�O���A�.*

	conv_loss��=�K�S        )��P	6t�O���A�.*

	conv_loss� \<g��        )��P	��O���A�.*

	conv_lossË^<����        )��P	M��O���A�.*

	conv_loss=�x<�G�7        )��P	�	�O���A�.*

	conv_loss��<��[D        )��P	e<�O���A�.*

	conv_lossq�<�O�e        )��P	�m�O���A�.*

	conv_loss7�<�F�