       �K"	  �뉙�Abrain.Event:2��*��      D(�	��뉙�A"��
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
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape* 
_class
loc:@conv2d/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0
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
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
conv2d/kernel
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name * 
_class
loc:@conv2d/kernel*
	container 
�
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2d/kernel
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
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

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
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
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
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 
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
conv2d_3/Conv2DConv2DRelu_1conv2d_2/kernel/read*
T0*
data_formatNHWC*
strides
*
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
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_3/kernel
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
conv2d_3/kernel/AssignAssignconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
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
conv2d_4/Conv2DConv2DRelu_2conv2d_3/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
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
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_4/kernel
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
�
conv2d_4/kernel
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_4/kernel
�
conv2d_4/kernel/AssignAssignconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_4/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
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
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:*
T0
�
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_5/kernel
�
conv2d_5/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_5/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:
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
conv2d_6/Conv2DConv2DRelu_4conv2d_5/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
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
.conv2d_6/kernel/Initializer/random_uniform/subSub.conv2d_6/kernel/Initializer/random_uniform/max.conv2d_6/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_6/kernel
�
.conv2d_6/kernel/Initializer/random_uniform/mulMul8conv2d_6/kernel/Initializer/random_uniform/RandomUniform.conv2d_6/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:*
T0
�
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
�
conv2d_6/kernel
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_6/kernel*
	container *
shape:
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
conv2d_7/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*
data_formatNHWC*
strides
*
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
Reshape/shapeConst*
_output_shapes
:*
valueB"����\  *
dtype0
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
seed2 *
dtype0*
_output_shapes
:	�
d*

seed *
T0*
_class
loc:@dense/kernel
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
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
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_output_shapes
:	�
d*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
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
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

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
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes

:d
*
T0
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
logistic_loss/GreaterEqualGreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:���������

�
logistic_loss/SelectSelectlogistic_loss/GreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:���������
*
T0
[
logistic_loss/NegNegdense_2/BiasAdd*'
_output_shapes
:���������
*
T0
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
: *

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
_output_shapes
:*
T0*
out_type0
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
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:���������

�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:���������
*
T0
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
:*

Tidx0*
	keep_dims( *
T0
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
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_5'gradients/conv2d_7/Conv2D_grad/ShapeN:1gradients/Relu_6_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
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
N* 
_output_shapes
::*
T0*
out_type0
�
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4'gradients/conv2d_6/Conv2D_grad/ShapeN:1gradients/Relu_5_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_4/kernel/readgradients/Relu_4_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/Relu_3_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2'gradients/conv2d_4/Conv2D_grad/ShapeN:1gradients/Relu_3_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
�
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel
�
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
;GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentApplyGradientDescentconv2d_3/kernelGradientDescent/learning_rate9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_3/kernel
�
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
use_locking( 
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
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
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
: "�[v��      ?�K	���뉙�AJ��
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
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0* 
_class
loc:@conv2d/kernel
�
conv2d/kernel
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name * 
_class
loc:@conv2d/kernel*
	container 
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
valueB"      *
dtype0*
_output_shapes
:
�
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*
T0*
data_formatNHWC*
strides
*
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
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_1/kernel
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
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

Y
Relu_1Reluconv2d_2/Conv2D*
T0*/
_output_shapes
:���������
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
.conv2d_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *��:>
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
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_2/kernel
�
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:*
T0
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
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_3/Conv2DConv2DRelu_1conv2d_2/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
T0*
data_formatNHWC*
strides
*
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

seed *
T0*"
_class
loc:@conv2d_4/kernel*
seed2 *
dtype0*&
_output_shapes
:
�
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_4/kernel
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_4/kernel
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
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
.conv2d_5/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel*
valueB
 *d��*
dtype0
�
.conv2d_5/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel*
valueB
 *d�=*
dtype0
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
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_6/Conv2DConv2DRelu_4conv2d_5/kernel/read*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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
8conv2d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_6/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_6/kernel
�
.conv2d_6/kernel/Initializer/random_uniform/subSub.conv2d_6/kernel/Initializer/random_uniform/max.conv2d_6/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_6/kernel
�
.conv2d_6/kernel/Initializer/random_uniform/mulMul8conv2d_6/kernel/Initializer/random_uniform/RandomUniform.conv2d_6/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
�
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
�
conv2d_6/kernel
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *"
_class
loc:@conv2d_6/kernel*
	container *
shape:
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
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
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
ReshapeReshapeRelu_6Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������

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
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
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
_class
loc:@dense/kernel*
_output_shapes
:	�
d*
T0
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
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:d
*
use_locking(*
T0
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
logistic_loss/NegNegdense_2/BiasAdd*'
_output_shapes
:���������
*
T0
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
: *

Tidx0*
	keep_dims( *
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
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
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
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:���������
*
T0
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
_output_shapes
:*
T0*
out_type0
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
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:���������

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
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*'
_output_shapes
:���������
*
T0
�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
�
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������
*
T0
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
$gradients/logistic_loss/mul_grad/mulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Placeholder_1*
T0*'
_output_shapes
:���������

�
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N
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
%gradients/conv2d_7/Conv2D_grad/ShapeNShapeNRelu_5conv2d_6/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
�
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_5'gradients/conv2d_7/Conv2D_grad/ShapeN:1gradients/Relu_6_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
/gradients/conv2d_7/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*E
_class;
97loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInput
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
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4'gradients/conv2d_6/Conv2D_grad/ShapeN:1gradients/Relu_5_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_4/kernel/readgradients/Relu_4_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
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
gradients/Relu_3_grad/ReluGradReluGrad7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyRelu_3*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNRelu_2conv2d_3/kernel/read*
N* 
_output_shapes
::*
T0*
out_type0
�
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/Relu_3_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2'gradients/conv2d_4/Conv2D_grad/ShapeN:1gradients/Relu_3_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/Relu_2_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_2/Conv2D_grad/tuple/group_deps*F
_class<
:8loc:@gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:*
T0
�
gradients/Relu_grad/ReluGradReluGrad7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:���������
�
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
�
ArgMaxArgMaxdense_2/BiasAddArgMax/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
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
: *

Tidx0*
	keep_dims( *
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
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0�E@�       `/�#	�)쉙�A*

	conv_loss�1?i��t       QKD	?}0쉙�A*

	conv_loss{�1?���       QKD	��0쉙�A*

	conv_lossx�1?lJ��       QKD	�0쉙�A*

	conv_loss��1?G@[h       QKD	�&1쉙�A*

	conv_loss
�1?W
޻       QKD	�\1쉙�A*

	conv_loss��1?}�       QKD	��1쉙�A*

	conv_loss{1?���K       QKD	'�1쉙�A*

	conv_loss\u1?d�V       QKD	2쉙�A*

	conv_loss�q1?pԠ�       QKD	fU2쉙�A	*

	conv_lossۅ1?^͹       QKD	#�2쉙�A
*

	conv_losses1?%�2�       QKD	{�2쉙�A*

	conv_losscn1?�I�}       QKD	��2쉙�A*

	conv_loss�n1?�@�P       QKD	n63쉙�A*

	conv_loss�j1?z�(�       QKD	1l3쉙�A*

	conv_loss�x1?|��K       QKD	��3쉙�A*

	conv_losswn1?�u�       QKD	n�3쉙�A*

	conv_lossBc1?���k       QKD	4쉙�A*

	conv_loss�d1?Z���       QKD	jM4쉙�A*

	conv_loss-j1?��       QKD	Ԉ4쉙�A*

	conv_loss�i1?i�h}       QKD	��4쉙�A*

	conv_loss�m1?��s�       QKD	��4쉙�A*

	conv_loss�X1?P�V�       QKD	R05쉙�A*

	conv_loss�^1?òI�       QKD	f5쉙�A*

	conv_loss5S1?��       QKD	8�5쉙�A*

	conv_loss�Z1?s��|       QKD	C�5쉙�A*

	conv_loss�X1?
��7       QKD	�6쉙�A*

	conv_loss�W1?���       QKD	A@6쉙�A*

	conv_loss*Q1?�u�v       QKD	�w6쉙�A*

	conv_lossTK1?F��n       QKD	��6쉙�A*

	conv_loss.H1?�^ѓ       QKD	�6쉙�A*

	conv_lossS1?�[݃       QKD	M7쉙�A*

	conv_lossJ1?�T��       QKD	N7쉙�A *

	conv_loss�91?�_       QKD	_7쉙�A!*

	conv_loss�>1?vkk9       QKD	۰7쉙�A"*

	conv_loss�D1?��[       QKD	��7쉙�A#*

	conv_loss,01?���f       QKD	�8쉙�A$*

	conv_lossw;1?%�#K       QKD	�K8쉙�A%*

	conv_loss�:1?ި�[       QKD	X~8쉙�A&*

	conv_lossk01?oφ�       QKD	�8쉙�A'*

	conv_loss71?��,�       QKD	��8쉙�A(*

	conv_loss�<1?��!       QKD	�9쉙�A)*

	conv_loss{*1?9��       QKD	�I9쉙�A**

	conv_loss)31?h$\       QKD	{{9쉙�A+*

	conv_loss�+1?�/�U       QKD	8�9쉙�A,*

	conv_loss�<1?ҽ��       QKD	=�9쉙�A-*

	conv_loss8#1?�7�       QKD	:쉙�A.*

	conv_loss�61?o���       QKD	�F:쉙�A/*

	conv_loss^1?*x       QKD	z:쉙�A0*

	conv_loss.#1?�y�,       QKD	��:쉙�A1*

	conv_loss1?�8�       QKD	��:쉙�A2*

	conv_lossV1?O�`h       QKD	t';쉙�A3*

	conv_lossn 1?ٞ'�       QKD	3Z;쉙�A4*

	conv_loss.1?�.�       QKD	Ì;쉙�A5*

	conv_loss�1?��|�       QKD	��;쉙�A6*

	conv_loss�1?��       QKD	��;쉙�A7*

	conv_loss.1?��vu       QKD	.+<쉙�A8*

	conv_loss(1?^�;�       QKD	�k<쉙�A9*

	conv_lossJ
1?Nu       QKD	�<쉙�A:*

	conv_loss�1?Fӛ�       QKD	H�<쉙�A;*

	conv_loss��0?Y���       QKD	|=쉙�A<*

	conv_loss��0?�܂�       QKD	�:=쉙�A=*

	conv_lossl�0?�|_       QKD	�y=쉙�A>*

	conv_loss�1?Wkk�       QKD	��=쉙�A?*

	conv_loss1?����       QKD	V�=쉙�A@*

	conv_loss��0?!��       QKD	�>쉙�AA*

	conv_loss��0?�!��       QKD	�L>쉙�AB*

	conv_loss_�0?9�N       QKD	<�>쉙�AC*

	conv_loss��0?��|       QKD	~�>쉙�AD*

	conv_loss��0?�T�       QKD	��>쉙�AE*

	conv_loss��0?hZ_k       QKD	�?쉙�AF*

	conv_loss�0?� �X       QKD	>Q?쉙�AG*

	conv_loss��0?�q�       QKD	R�?쉙�AH*

	conv_loss�0?S�v�       QKD	�?쉙�AI*

	conv_lossI�0?�	�       QKD	D�?쉙�AJ*

	conv_loss�0?���       QKD	z@쉙�AK*

	conv_loss��0?�w r       QKD	�N@쉙�AL*

	conv_loss �0?�P��       QKD	`�@쉙�AM*

	conv_loss#�0?.%��       QKD	·@쉙�AN*

	conv_loss��0?j4       QKD	r�@쉙�AO*

	conv_lossX�0?��,%       QKD	�A쉙�AP*

	conv_lossb�0?�|ռ       QKD	�SA쉙�AQ*

	conv_loss6�0?�F�u       QKD	�A쉙�AR*

	conv_loss��0?��        QKD	�A쉙�AS*

	conv_loss`�0?}vd       QKD	��A쉙�AT*

	conv_loss��0?%^J       QKD	&B쉙�AU*

	conv_loss��0?�ҷ�       QKD	�LB쉙�AV*

	conv_loss��0?��c(       QKD	�B쉙�AW*

	conv_loss��0??�xB       QKD	�B쉙�AX*

	conv_loss4�0?�	�/       QKD	��B쉙�AY*

	conv_loss0�0?9�!�       QKD	IC쉙�AZ*

	conv_lossU�0?�v�6       QKD	�GC쉙�A[*

	conv_loss0�0?R�^       QKD	wyC쉙�A\*

	conv_lossI�0?p���       QKD	��C쉙�A]*

	conv_loss߾0?J�~|       QKD	��C쉙�A^*

	conv_loss¼0?z�(�       QKD	iD쉙�A_*

	conv_lossu�0?<�W�       QKD	�=D쉙�A`*

	conv_loss�0?`���       QKD	HqD쉙�Aa*

	conv_loss-�0?sQbi       QKD	u�D쉙�Ab*

	conv_loss��0?��:(       QKD	W�D쉙�Ac*

	conv_loss��0?�M'       QKD	�E쉙�Ad*

	conv_loss<�0?��       QKD	�LE쉙�Ae*

	conv_lossj�0?����       QKD	�~E쉙�Af*

	conv_loss��0?q۳       QKD	�E쉙�Ag*

	conv_loss��0?+(H       QKD	��E쉙�Ah*

	conv_lossҰ0?���       QKD	�F쉙�Ai*

	conv_loss��0?�M�       QKD	�NF쉙�Aj*

	conv_loss�0?��       QKD	5�F쉙�Ak*

	conv_loss��0?�>�       QKD	�F쉙�Al*

	conv_loss��0?@
]�       QKD	��F쉙�Am*

	conv_loss��0?|���       QKD	� G쉙�An*

	conv_loss��0?��	>       QKD	�QG쉙�Ao*

	conv_lossr�0?z�-'       QKD	ڂG쉙�Ap*

	conv_lossE�0?���       QKD	�G쉙�Aq*

	conv_lossl�0?,Ҷ       QKD	��G쉙�Ar*

	conv_lossU�0?�o��       QKD	wH쉙�As*

	conv_loss֗0?)u#       QKD	�EH쉙�At*

	conv_loss�0?��	       QKD	zyH쉙�Au*

	conv_loss�0?�h��       QKD	*�H쉙�Av*

	conv_lossD�0?W�	       QKD	4�H쉙�Aw*

	conv_loss{�0?o4b       QKD	?I쉙�Ax*

	conv_loss=0?��f�       QKD	
@I쉙�Ay*

	conv_loss��0?�avv       QKD	�qI쉙�Az*

	conv_loss�z0?��s       QKD	]�I쉙�A{*

	conv_lossw�0?����       QKD	u�I쉙�A|*

	conv_lossU|0?A:`       QKD	3J쉙�A}*

	conv_loss�0?�ws       QKD	�0J쉙�A~*

	conv_loss�p0?�Y�       QKD	�^J쉙�A*

	conv_loss}m0?�T        )��P	4�J쉙�A�*

	conv_loss�w0?D��        )��P	پJ쉙�A�*

	conv_loss�u0?��C        )��P	��J쉙�A�*

	conv_loss�s0?q{��        )��P	NK쉙�A�*

	conv_loss�m0?�o         )��P	�NK쉙�A�*

	conv_lossg0?�K�        )��P	z�K쉙�A�*

	conv_lossy0?r�        )��P	�K쉙�A�*

	conv_lossTh0?ݒ�;        )��P	��K쉙�A�*

	conv_lossh0?EMei        )��P	 L쉙�A�*

	conv_loss�d0?=�        )��P	GL쉙�A�*

	conv_losszW0?"���        )��P	wL쉙�A�*

	conv_loss�b0?���        )��P	<�L쉙�A�*

	conv_lossvN0?��JJ        )��P	H�L쉙�A�*

	conv_loss�S0?[�|        )��P	�M쉙�A�*

	conv_loss�j0?���        )��P	�:M쉙�A�*

	conv_loss/P0?�6E        )��P	@kM쉙�A�*

	conv_loss�N0?عI         )��P	l�M쉙�A�*

	conv_loss�R0?D��B        )��P	?�M쉙�A�*

	conv_lossWP0?�m�a        )��P	��M쉙�A�*

	conv_loss�Q0?�a�        )��P	�,N쉙�A�*

	conv_loss#O0?�6�`        )��P	s[N쉙�A�*

	conv_loss�F0?g\        )��P	��N쉙�A�*

	conv_loss�G0?[���        )��P	��N쉙�A�*

	conv_lossIA0?N���        )��P	�N쉙�A�*

	conv_loss[I0?gN�        )��P	�,O쉙�A�*

	conv_loss�@0?��        )��P	�\O쉙�A�*

	conv_loss�@0?r�B�        )��P	��O쉙�A�*

	conv_loss�.0?Q-}        )��P	l�O쉙�A�*

	conv_loss�80?��        )��P	 P쉙�A�*

	conv_loss%.0?��۬        )��P	�<P쉙�A�*

	conv_loss�<0?Ej        )��P	KkP쉙�A�*

	conv_loss"50?0&��        )��P	8�P쉙�A�*

	conv_lossk:0?��u�        )��P	|�P쉙�A�*

	conv_loss"00?	���        )��P	�Q쉙�A�*

	conv_loss20?5��*        )��P	�<Q쉙�A�*

	conv_loss�/0?X�        )��P	�kQ쉙�A�*

	conv_loss�00?��         )��P	E�Q쉙�A�*

	conv_loss�$0?oeD�        )��P	z�Q쉙�A�*

	conv_loss	00?c���        )��P	�Q쉙�A�*

	conv_loss�!0?�ͭj        )��P	�*R쉙�A�*

	conv_loss40?�}#        )��P	�ZR쉙�A�*

	conv_loss�0?��x        )��P	$�R쉙�A�*

	conv_loss�0?��.�        )��P	��R쉙�A�*

	conv_loss�0?z|r�        )��P	��R쉙�A�*

	conv_loss0?�8�2        )��P	+S쉙�A�*

	conv_loss�
0?e���        )��P	oYS쉙�A�*

	conv_loss~0?��e        )��P	r�S쉙�A�*

	conv_loss0?���%        )��P	�S쉙�A�*

	conv_loss�0?MPE        )��P	��S쉙�A�*

	conv_loss�0?�m<        )��P	jT쉙�A�*

	conv_loss�0?t��B        )��P	ALT쉙�A�*

	conv_loss@0?>$��        )��P	�{T쉙�A�*

	conv_loss�0?���        )��P	�T쉙�A�*

	conv_loss�0?�}         )��P	?�T쉙�A�*

	conv_loss�0?C�Z        )��P	WU쉙�A�*

	conv_loss� 0?k3&'        )��P	�=U쉙�A�*

	conv_loss�0?�`;-        )��P	 mU쉙�A�*

	conv_loss��/?���P        )��P	��U쉙�A�*

	conv_loss�/?��nT        )��P	p�U쉙�A�*

	conv_loss*0?�g�W        )��P	��U쉙�A�*

	conv_loss��/?ʻ��        )��P	�-V쉙�A�*

	conv_lossW�/?s�        )��P	>aV쉙�A�*

	conv_loss��/?�q�w        )��P	-�V쉙�A�*

	conv_loss�/?�b��        )��P	;�V쉙�A�*

	conv_loss��/?���        )��P	K�V쉙�A�*

	conv_loss �/?��c        )��P	i#W쉙�A�*

	conv_loss��/?Z��        )��P	�SW쉙�A�*

	conv_loss��/?eOn�        )��P	ȁW쉙�A�*

	conv_loss5�/?�3        )��P	ƴW쉙�A�*

	conv_loss�/?%X5        )��P	~�W쉙�A�*

	conv_lossb�/?w��B        )��P	�X쉙�A�*

	conv_loss��/?�f�        )��P	�DX쉙�A�*

	conv_loss@�/?p(Q�        )��P	�uX쉙�A�*

	conv_loss��/?D��        )��P	�	Z쉙�A�*

	conv_loss�/?�        )��P	;Z쉙�A�*

	conv_lossu�/?繥        )��P	�kZ쉙�A�*

	conv_loss�/?��Ֆ        )��P	.�Z쉙�A�*

	conv_loss�/?D��        )��P	{�Z쉙�A�*

	conv_loss��/?�g&        )��P	�[쉙�A�*

	conv_loss�/?��j�        )��P	I:[쉙�A�*

	conv_loss��/?���g        )��P	k[쉙�A�*

	conv_loss�/?�ms        )��P	¤[쉙�A�*

	conv_loss��/?�I�        )��P	B�[쉙�A�*

	conv_loss��/?z�8�        )��P	C
\쉙�A�*

	conv_lossx�/?x��x        )��P	�H\쉙�A�*

	conv_loss��/?�r��        )��P	�|\쉙�A�*

	conv_loss*�/?�@�        )��P	��\쉙�A�*

	conv_loss?�/?�,\z        )��P	��\쉙�A�*

	conv_lossg�/?�QK1        )��P	=]쉙�A�*

	conv_loss��/?��>        )��P	P]쉙�A�*

	conv_loss��/?�
        )��P	q�]쉙�A�*

	conv_loss�/?��E�        )��P	��]쉙�A�*

	conv_loss��/?�        )��P	e�]쉙�A�*

	conv_loss�/?��        )��P	W^쉙�A�*

	conv_lossѥ/?����        )��P	�H^쉙�A�*

	conv_loss��/?�,�        )��P	Ox^쉙�A�*

	conv_lossݟ/?��ǃ        )��P	��^쉙�A�*

	conv_lossΟ/?Yg�        )��P	��^쉙�A�*

	conv_loss|�/?�� T        )��P		_쉙�A�*

	conv_lossʡ/?��~        )��P	;_쉙�A�*

	conv_loss͛/?�GT        )��P	.l_쉙�A�*

	conv_lossa�/?���	        )��P	y�_쉙�A�*

	conv_loss��/?O�{6        )��P	�_쉙�A�*

	conv_lossƠ/?7�@�        )��P	~�_쉙�A�*

	conv_loss֎/?�@�@        )��P	*.`쉙�A�*

	conv_loss �/?d;R�        )��P	l^`쉙�A�*

	conv_loss�/?���        )��P	1�`쉙�A�*

	conv_loss=�/?w@�        )��P	e�`쉙�A�*

	conv_loss��/?�        )��P	��`쉙�A�*

	conv_loss�/?u�>�        )��P	�a쉙�A�*

	conv_loss�/?`v�        )��P	�Na쉙�A�*

	conv_loss�}/?�m��        )��P	��a쉙�A�*

	conv_loss��/?/n�        )��P	İa쉙�A�*

	conv_loss9�/?�
׹        )��P	��a쉙�A�*

	conv_loss�~/?�m�        )��P	bb쉙�A�*

	conv_loss�/?���        )��P	�Cb쉙�A�*

	conv_loss�{/?P-�7        )��P	�xb쉙�A�*

	conv_loss\/?%��j        )��P	K�b쉙�A�*

	conv_lossVq/?p�$        )��P	�b쉙�A�*

	conv_lossr/?S���        )��P	'c쉙�A�*

	conv_loss�r/?����        )��P	Y?c쉙�A�*

	conv_loss6m/?�/�$        )��P	)rc쉙�A�*

	conv_loss�g/?��&�        )��P	��c쉙�A�*

	conv_loss�y/?B�2Z        )��P	g�c쉙�A�*

	conv_loss�z/?(�        )��P	�d쉙�A�*

	conv_loss�^/?���K        )��P	�Id쉙�A�*

	conv_loss�^/?���        )��P	�~d쉙�A�*

	conv_loss�]/?�E�(        )��P	S�d쉙�A�*

	conv_loss�Y/?4�'�        )��P	=�d쉙�A�*

	conv_loss�Z/? l	�        )��P	�e쉙�A�*

	conv_losse`/?�l(        )��P	�De쉙�A�*

	conv_loss�`/?��ZK        )��P	�e쉙�A�*

	conv_loss�S/?�O        )��P	��e쉙�A�*

	conv_loss�R/??���        )��P	��e쉙�A�*

	conv_lossM/?��e        )��P	� f쉙�A�*

	conv_loss�K/?��՜        )��P	lQf쉙�A�*

	conv_loss�P/?����        )��P	��f쉙�A�*

	conv_loss�T/?���        )��P	�f쉙�A�*

	conv_loss�E/?���        )��P	��f쉙�A�*

	conv_loss�L/?:�bi        )��P	�-g쉙�A�*

	conv_loss�>/?)0�        )��P	r_g쉙�A�*

	conv_loss�O/?��5�        )��P	=�g쉙�A�*

	conv_lossE@/?�k|        )��P	��g쉙�A�*

	conv_loss,F/?�Ž�        )��P	��g쉙�A�*

	conv_loss�?/?b��%        )��P	�#h쉙�A�*

	conv_loss�F/?p�p        )��P	�Th쉙�A�*

	conv_loss�>/?�e��        )��P	��h쉙�A�*

	conv_loss�8/?�P�        )��P	{�h쉙�A�*

	conv_loss�//?~��?        )��P	��h쉙�A�*

	conv_lossD7/?�Qk�        )��P	�i쉙�A�*

	conv_loss�1/?�ܨ6        )��P	Mi쉙�A�*

	conv_loss�6/?�o�%        )��P	q�i쉙�A�*

	conv_lossL /?A�yF        )��P	��i쉙�A�*

	conv_loss,/?s��^        )��P	@�i쉙�A�*

	conv_lossu,/?*Ԯ        )��P	�j쉙�A�*

	conv_loss6 /?UAY^        )��P	�Fj쉙�A�*

	conv_loss[#/?�e�        )��P	wj쉙�A�*

	conv_loss,"/?�>9        )��P	��j쉙�A�*

	conv_lossX"/?���        )��P	��j쉙�A�*

	conv_lossd/?�j[�        )��P	�k쉙�A�*

	conv_lossr/?���        )��P	�Fk쉙�A�*

	conv_loss�/?7��u        )��P	�vk쉙�A�*

	conv_loss6/?:�u7        )��P	��k쉙�A�*

	conv_loss/?�ɔ        )��P	��k쉙�A�*

	conv_loss�/?}�|�        )��P	�l쉙�A�*

	conv_losse/?)�7        )��P	:Ll쉙�A�*

	conv_lossl/?E4`�        )��P	I~l쉙�A�*

	conv_lossG/?�x�        )��P	O�l쉙�A�*

	conv_loss�
/?a��        )��P	~�l쉙�A�*

	conv_loss�
/?�;�        )��P	�m쉙�A�*

	conv_loss�/?�ZP        )��P	�;m쉙�A�*

	conv_lossQ/?�5\f        )��P	nm쉙�A�*

	conv_loss�/?0�l�        )��P	ßm쉙�A�*

	conv_loss:/?�3�        )��P	��m쉙�A�*

	conv_loss��.?�`'�        )��P	in쉙�A�*

	conv_loss[/?��`�        )��P	�Cn쉙�A�*

	conv_loss?�.?"_[?        )��P	un쉙�A�*

	conv_loss��.?�4�        )��P	/�n쉙�A�*

	conv_lossb�.?���        )��P	�n쉙�A�*

	conv_loss��.?�nJ:        )��P	Qo쉙�A�*

	conv_lossA�.?M�	        )��P	�Jo쉙�A�*

	conv_loss��.?� UD        )��P	@�o쉙�A�*

	conv_lossh�.?+�#        )��P	��o쉙�A�*

	conv_loss7�.?o���        )��P	��o쉙�A�*

	conv_loss��.?�_Y        )��P	�'p쉙�A�*

	conv_loss��.?/��?        )��P	9Xp쉙�A�*

	conv_loss�.?��*\        )��P	��p쉙�A�*

	conv_loss��.?2�L�        )��P	<�p쉙�A�*

	conv_loss��.?g�ls        )��P	��p쉙�A�*

	conv_lossL�.??y��        )��P	/q쉙�A�*

	conv_loss��.?�$�n        )��P	�_q쉙�A�*

	conv_lossf�.?��        )��P	%�q쉙�A�*

	conv_loss�.?|�ؐ        )��P	G�q쉙�A�*

	conv_loss�.?Z��        )��P	��q쉙�A�*

	conv_lossY�.?���        )��P	z%r쉙�A�*

	conv_loss��.?�ʥ�        )��P	XWr쉙�A�*

	conv_loss��.?��c        )��P	O�r쉙�A�*

	conv_loss��.?.        )��P	N�r쉙�A�*

	conv_loss�.?��N�        )��P	Z�r쉙�A�*

	conv_loss�.?����        )��P	�s쉙�A�*

	conv_loss��.? �,�        )��P	QHs쉙�A�*

	conv_loss��.?�G7i        )��P	�ys쉙�A�*

	conv_loss��.?,�V        )��P	¨s쉙�A�*

	conv_loss
�.?�d�        )��P	M�s쉙�A�*

	conv_loss��.?��        )��P	3	t쉙�A�*

	conv_loss]�.?VGZ�        )��P	.;t쉙�A�*

	conv_loss��.?�C�        )��P	�kt쉙�A�*

	conv_loss��.?�o�|        )��P	ݜt쉙�A�*

	conv_lossͫ.?� [�        )��P	��t쉙�A�*

	conv_loss��.?�tI        )��P	��t쉙�A�*

	conv_loss�.?.���        )��P	4.u쉙�A�*

	conv_loss �.?�'�U        )��P	�^u쉙�A�*

	conv_loss^�.?�<        )��P	��u쉙�A�*

	conv_loss
�.?�|��        )��P	��u쉙�A�*

	conv_losso�.?�� x        )��P	��u쉙�A�*

	conv_lossģ.?PE�h        )��P	�$v쉙�A�*

	conv_lossΡ.?/cV�        )��P	�Uv쉙�A�*

	conv_loss�.?5Tv�        )��P	ąv쉙�A�*

	conv_loss�.?Q�aD        )��P	c�v쉙�A�*

	conv_loss3�.?��&$        )��P	��v쉙�A�*

	conv_lossO�.?�3$        )��P	aw쉙�A�*

	conv_lossٛ.?��        )��P	�Fw쉙�A�*

	conv_losst�.?���        )��P	�xw쉙�A�*

	conv_loss��.?����        )��P	��w쉙�A�*

	conv_lossס.?@/qm        )��P	��w쉙�A�*

	conv_loss��.? TB        )��P	 %x쉙�A�*

	conv_loss��.?G�        )��P	?Xx쉙�A�*

	conv_loss��.?����        )��P	<�x쉙�A�*

	conv_loss��.?"��0        )��P	��x쉙�A�*

	conv_loss��.?��@�        )��P	��x쉙�A�*

	conv_loss+�.?p��N        )��P	�8y쉙�A�*

	conv_loss$~.?��u        )��P	Ply쉙�A�*

	conv_lossل.?U�]�        )��P	|�y쉙�A�*

	conv_loss+|.?X�        )��P	B�y쉙�A�*

	conv_loss��.?J�G,        )��P	>z쉙�A�*

	conv_loss w.?��Td        )��P	�Fz쉙�A�*

	conv_loss�y.?��=        )��P	2yz쉙�A�*

	conv_loss�y.?Y�Q�        )��P	��z쉙�A�*

	conv_loss�s.?#r�        )��P	p�z쉙�A�*

	conv_lossRu.?�}l        )��P	�{쉙�A�*

	conv_loss�k.?�[s`        )��P	�L{쉙�A�*

	conv_loss�g.?m^��        )��P	�~{쉙�A�*

	conv_loss=o.?c�Q0        )��P	ͱ{쉙�A�*

	conv_loss�n.?A�9�        )��P	��{쉙�A�*

	conv_loss"l.?�${�        )��P	�|쉙�A�*

	conv_loss\`.?`ҿ        )��P	�L|쉙�A�*

	conv_loss�l.?���X        )��P	��|쉙�A�*

	conv_loss�`.?���A        )��P	ĳ|쉙�A�*

	conv_loss�].?� [�        )��P	��|쉙�A�*

	conv_loss�Z.?�n
�        )��P	W}쉙�A�*

	conv_loss�T.?��~        )��P	�O}쉙�A�*

	conv_loss�_.?f�~�        )��P	'�}쉙�A�*

	conv_loss�X.?��e        )��P	��}쉙�A�*

	conv_lossyU.?5�        )��P	f�}쉙�A�*

	conv_loss*Z.?k'$        )��P	1~쉙�A�*

	conv_loss�O.?q%u�        )��P	�R~쉙�A�*

	conv_lossH.?�<��        )��P	K�~쉙�A�*

	conv_loss�:.?�I�        )��P	��~쉙�A�*

	conv_loss�P.?��Ҽ        )��P	d�~쉙�A�*

	conv_loss�D.?��        )��P	* 쉙�A�*

	conv_lossJA.?�w�        )��P	,R쉙�A�*

	conv_loss�P.?ʌ��        )��P	z�쉙�A�*

	conv_loss�<.?1�m         )��P	�쉙�A�*

	conv_loss�F.?�d�\        )��P	.�쉙�A�*

	conv_loss�=.?;S�        )��P	~�쉙�A�*

	conv_loss=.?ܮ~�        )��P	�P�쉙�A�*

	conv_loss�9.?$�q        )��P	7��쉙�A�*

	conv_loss�-.?�3�i        )��P	7��쉙�A�*

	conv_loss�7.?�)�        )��P	��쉙�A�*

	conv_loss.4.?�L��        )��P	��쉙�A�*

	conv_loss�8.?�;��        )��P	�O�쉙�A�*

	conv_loss�+.?����        )��P	i��쉙�A�*

	conv_lossZ-.?��C�        )��P	�쉙�A�*

	conv_lossz.?Ǧ�        )��P	�M�쉙�A�*

	conv_loss�.?&��X        )��P	�~�쉙�A�*

	conv_loss�#.?�{�b        )��P	���쉙�A�*

	conv_loss�.?ϡ5        )��P	�/�쉙�A�*

	conv_lossQ.?��o        )��P	�쉙�A�*

	conv_loss0".?C��0        )��P	�L�쉙�A�*

	conv_loss�.?�.�        )��P	sх쉙�A�*

	conv_loss�$.?�fѿ        )��P	RG�쉙�A�*

	conv_lossy .?$��        )��P	ټ�쉙�A�*

	conv_loss�.?L$�        )��P	3�쉙�A�*

	conv_loss.?�G        )��P	�ч쉙�A�*

	conv_loss�.?�B�        )��P	SI�쉙�A�*

	conv_loss�.?rߍ�        )��P	���쉙�A�*

	conv_loss�.?�0�        )��P	�5�쉙�A�*

	conv_loss�.?|d��        )��P	쉙�A�*

	conv_loss.?�>k�        )��P	"!�쉙�A�*

	conv_loss�.?��T        )��P	���쉙�A�*

	conv_loss�.?�9>R        )��P	m
�쉙�A�*

	conv_loss�.?���        )��P	��쉙�A�*

	conv_losss.?�n�        )��P	���쉙�A�*

	conv_loss�.?�oC        )��P	k�쉙�A�*

	conv_loss��-?W'qy        )��P	�ތ쉙�A�*

	conv_loss��-?m��        )��P	�S�쉙�A�*

	conv_loss+.?�φ�        )��P	Mō쉙�A�*

	conv_loss��-?�
p�        )��P	�:�쉙�A�*

	conv_loss��-?tQ�        )��P	���쉙�A�*

	conv_loss�-?Z'4�        )��P	�(�쉙�A�*

	conv_loss�-?>        )��P	���쉙�A�*

	conv_loss��-?(C�        )��P	=�쉙�A�*

	conv_loss	�-?2H5        )��P	\��쉙�A�*

	conv_loss��-?�Fl�        )��P	,"�쉙�A�*

	conv_loss��-?�)��        )��P	G��쉙�A�*

	conv_loss��-?�YH�        )��P	�쉙�A�*

	conv_lossr�-?u�i        )��P	��쉙�A�*

	conv_loss�-?�!��        )��P	-�쉙�A�*

	conv_loss�-?˘L3        )��P	�{�쉙�A�*

	conv_loss��-?����        )��P	��쉙�A�*

	conv_lossh�-?6�^        )��P	sh�쉙�A�*

	conv_loss��-?�`--        )��P	�ޔ쉙�A�*

	conv_loss
�-?Z^�l        )��P	�R�쉙�A�*

	conv_loss6�-?��        )��P	�ʕ쉙�A�*

	conv_loss��-?���        )��P	6>�쉙�A�*

	conv_loss��-?z�D�        )��P	���쉙�A�*

	conv_losso�-?��h^        )��P	�'�쉙�A�*

	conv_loss��-?x��        )��P	̚�쉙�A�*

	conv_loss��-?<;��        )��P	w�쉙�A�*

	conv_loss^�-?��=        )��P	͂�쉙�A�*

	conv_loss|�-?��_        )��P	���쉙�A�*

	conv_lossB�-?�/��        )��P	yn�쉙�A�*

	conv_loss��-?�_        )��P	��쉙�A�*

	conv_loss�-?��i        )��P	v��쉙�A�*

	conv_loss��-?���        )��P	��쉙�A�*

	conv_loss��-?wN        )��P	.��쉙�A�*

	conv_lossF�-?��Z\        )��P	0*�쉙�A�*

	conv_lossc�-?#L�"        )��P	���쉙�A�*

	conv_lossº-?�O        )��P	!�쉙�A�*

	conv_lossx�-?!;+�        )��P	Q��쉙�A�*

	conv_loss3�-?K�a�        )��P	e	�쉙�A�*

	conv_loss&�-?u�7        )��P	冞쉙�A�*

	conv_loss[�-?�s�        )��P	^��쉙�A�*

	conv_lossi�-?7���        )��P	s�쉙�A�*

	conv_loss��-?	9M�        )��P	�쉙�A�*

	conv_loss��-?'8)Y        )��P	�[�쉙�A�*

	conv_loss��-?]�I        )��P	*Ҡ쉙�A�*

	conv_loss��-?�w�U        )��P	�H�쉙�A�*

	conv_loss��-?�#m�        )��P	׾�쉙�A�*

	conv_loss\�-?�"�        )��P	4�쉙�A�*

	conv_lossю-?��        )��P	ǩ�쉙�A�*

	conv_loss�-?y��!        )��P	U"�쉙�A�*

	conv_loss"�-?s�\        )��P	��쉙�A�*

	conv_loss��-?x���        )��P	w�쉙�A�*

	conv_loss��-?�&J        )��P	��쉙�A�*

	conv_loss��-?Pm��        )��P	m��쉙�A�*

	conv_loss!�-?����        )��P	l�쉙�A�*

	conv_loss�-?� 0&        )��P	�쉙�A�*

	conv_loss��-?�	��        )��P	�V�쉙�A�*

	conv_lossS�-?e=,.        )��P	%ͦ쉙�A�*

	conv_loss��-?��         )��P	AE�쉙�A�*

	conv_loss��-?�
�j        )��P	���쉙�A�*

	conv_lossQ�-?�P��        )��P	$2�쉙�A�*

	conv_lossG�-?����        )��P	���쉙�A�*

	conv_lossK|-?��	        )��P	��쉙�A�*

	conv_loss�-?ݨ�        )��P	F��쉙�A�*

	conv_loss4�-?͘(        )��P	2�쉙�A�*

	conv_loss��-?��f�        )��P	��쉙�A�*

	conv_loss�~-?s�|v        )��P	��쉙�A�*

	conv_loss�w-?�7�        )��P	�W�쉙�A�*

	conv_loss0s-? qR7        )��P	\��쉙�A�*

	conv_loss�s-?z�        )��P	[�쉙�A�*

	conv_losss-?,�Wj        )��P	Z�쉙�A�*

	conv_loss�t-? ^Ӹ        )��P	δ�쉙�A�*

	conv_lossyk-? K�        )��P	)�쉙�A�*

	conv_loss�m-?�Q�        )��P	�g�쉙�A�*

	conv_loss�b-?��ݞ        )��P	��쉙�A�*

	conv_loss�e-?�C�        )��P	��쉙�A�*

	conv_loss�d-?����        )��P	h�쉙�A�*

	conv_loss�Z-?�n
�        )��P	v��쉙�A�*

	conv_loss8X-?���.        )��P	0�쉙�A�*

	conv_lossrd-?��_�        )��P	oi�쉙�A�*

	conv_loss�[-?�v�j        )��P	IC�쉙�A�*

	conv_losssM-?��L        )��P	���쉙�A�*

	conv_lossdT-?K�v�        )��P	o��쉙�A�*

	conv_loss�c-?��K        )��P	��쉙�A�*

	conv_loss6J-?5rB�        )��P	��쉙�A�*

	conv_lossT-?�3�        )��P	RF�쉙�A�*

	conv_loss�@-?ơ        )��P	�t�쉙�A�*

	conv_loss�L-?�O��        )��P	Σ�쉙�A�*

	conv_loss(@-?��A�        )��P	"�쉙�A�*

	conv_loss#C-?���,        )��P	��쉙�A�*

	conv_loss�K-?��        )��P	�N�쉙�A�*

	conv_loss�H-?{�=)        )��P	5~�쉙�A�*

	conv_lossVC-?]_D        )��P	���쉙�A�*

	conv_loss�?-?�4��        )��P	*��쉙�A�*

	conv_loss8@-?R�+[        )��P	�(�쉙�A�*

	conv_loss$=-?)        )��P	yW�쉙�A�*

	conv_loss�<-?{}�        )��P	腸쉙�A�*

	conv_loss�7-?�⩱        )��P	���쉙�A�*

	conv_loss�6-?�0�;        )��P	M�쉙�A�*

	conv_lossd4-?�b�        )��P	u�쉙�A�*

	conv_loss}0-?_~�        )��P	qD�쉙�A�*

	conv_loss0-?�oCJ        )��P	_w�쉙�A�*

	conv_loss5-?g�K        )��P	���쉙�A�*

	conv_loss-=-?]I��        )��P	x׹쉙�A�*

	conv_loss1-?�w��        )��P	e�쉙�A�*

	conv_loss�-?5jzd        )��P	�8�쉙�A�*

	conv_lossj-?�6^        )��P	Tf�쉙�A�*

	conv_loss�(-?�*�        )��P	���쉙�A�*

	conv_loss� -?T���        )��P	�ź쉙�A�*

	conv_loss�-?&��        )��P	���쉙�A�*

	conv_loss)-?���        )��P	�'�쉙�A�*

	conv_loss/*-?F��        )��P	0X�쉙�A�*

	conv_loss%-?�T�        )��P	܆�쉙�A�*

	conv_lossE-?K�|�        )��P	���쉙�A�*

	conv_loss�-?=N��        )��P	��쉙�A�*

	conv_lossh-?���G        )��P	l�쉙�A�*

	conv_lossn-?��O         )��P	�D�쉙�A�*

	conv_loss�-?�֢I        )��P	Qw�쉙�A�*

	conv_loss��,?�v        )��P	0��쉙�A�*

	conv_loss�-?*���        )��P	W׼쉙�A�*

	conv_lossQ-?i�q�        )��P	o�쉙�A�*

	conv_loss�-?��        )��P	�4�쉙�A�*

	conv_lossO-?@%.�        )��P	�c�쉙�A�*

	conv_losse�,?��5R        )��P	"��쉙�A�*

	conv_loss� -?���        )��P	���쉙�A�*

	conv_loss�-?N�@A        )��P	��쉙�A�*

	conv_loss��,?!~I�        )��P	c�쉙�A�*

	conv_loss� -?-\Ǣ        )��P	J�쉙�A�*

	conv_loss��,?0b)        )��P	&y�쉙�A�*

	conv_loss�,?�{�3        )��P	��쉙�A�*

	conv_loss�-?�5��        )��P	�ؾ쉙�A�*

	conv_loss0�,?V��        )��P	f�쉙�A�*

	conv_loss��,?V�E7        )��P	H�쉙�A�*

	conv_loss��,?
���        )��P	�y�쉙�A�*

	conv_loss��,?�N��        )��P	V��쉙�A�*

	conv_loss��,?W.q�        )��P	A߿쉙�A�*

	conv_loss��,?���y        )��P	=�쉙�A�*

	conv_loss3�,?JP��        )��P	�N�쉙�A�*

	conv_loss�,?l85        )��P	�~�쉙�A�*

	conv_loss��,?�)U        )��P	���쉙�A�*

	conv_lossW�,?����        )��P	���쉙�A�*

	conv_loss��,?��B�        )��P	��쉙�A�*

	conv_loss��,?,F��        )��P	nM�쉙�A�*

	conv_loss|�,?�*�G        )��P	0{�쉙�A�*

	conv_loss��,?Nҕ�        )��P	���쉙�A�*

	conv_loss��,?��.s        )��P	���쉙�A�*

	conv_loss��,?V��        )��P	��쉙�A�*

	conv_loss��,? ��        )��P	%I�쉙�A�*

	conv_lossV�,?��>        )��P	�y�쉙�A�*

	conv_loss��,?��        )��P	���쉙�A�*

	conv_loss��,?J�Ӧ        )��P	
��쉙�A�*

	conv_loss��,?��&        )��P	��쉙�A�*

	conv_loss��,?oUr�        )��P	6�쉙�A�*

	conv_loss��,?�EY        )��P	�d�쉙�A�*

	conv_lossھ,?YY�         )��P	���쉙�A�*

	conv_loss��,?��;        )��P	���쉙�A�*

	conv_lossҼ,?�Nr        )��P	Q��쉙�A�*

	conv_loss�,?��HD        )��P	$�쉙�A�*

	conv_lossX�,?__��        )��P	T�쉙�A�*

	conv_loss�,? m��        )��P	��쉙�A�*

	conv_loss��,?\b	^        )��P	���쉙�A�*

	conv_lossʯ,?�r�9        )��P	T��쉙�A�*

	conv_loss'�,?���        )��P	�쉙�A�*

	conv_loss��,?YT�S        )��P	�G�쉙�A�*

	conv_loss��,?�1�M        )��P	y�쉙�A�*

	conv_loss��,?�5�        )��P	&��쉙�A�*

	conv_loss�,?�`        )��P	���쉙�A�*

	conv_loss��,?Xߑ        )��P	��쉙�A�*

	conv_loss"�,?�&I        )��P	�<�쉙�A�*

	conv_loss��,?�m\�        )��P	�l�쉙�A�*

	conv_loss�,?��n�        )��P	Q��쉙�A�*

	conv_loss��,?��H        )��P	��쉙�A�*

	conv_loss4�,?�X��        )��P	���쉙�A�*

	conv_loss��,?��b�        )��P	�.�쉙�A�*

	conv_loss�,?^�O        )��P	D_�쉙�A�*

	conv_lossW�,?� yw        )��P	ɏ�쉙�A�*

	conv_lossǜ,?$��        )��P	F��쉙�A�*

	conv_loss��,?��n�        )��P	!��쉙�A�*

	conv_loss[�,?��`�        )��P	��쉙�A�*

	conv_loss�,?�=        )��P	
O�쉙�A�*

	conv_lossx�,?`�l        )��P	9�쉙�A�*

	conv_loss9�,?}�9�        )��P	`�쉙�A�*

	conv_loss��,?��        )��P	 >�쉙�A�*

	conv_loss)~,?��JJ        )��P	Zl�쉙�A�*

	conv_loss��,?�.��        )��P	��쉙�A�*

	conv_loss�,?H\�k        )��P	���쉙�A�*

	conv_loss=u,?����        )��P	���쉙�A�*

	conv_lossbo,?�=_        )��P	�+�쉙�A�*

	conv_losses,?_�N�        )��P	[�쉙�A�*

	conv_loss��,?�	�        )��P	��쉙�A�*

	conv_loss|,?Ze�        )��P	
��쉙�A�*

	conv_loss�{,?{��w        )��P	u�쉙�A�*

	conv_loss�o,?�U�        )��P	�4�쉙�A�*

	conv_loss�f,?�O�p        )��P	�d�쉙�A�*

	conv_loss$j,?Q���        )��P	��쉙�A�*

	conv_loss�g,?V�k*        )��P	���쉙�A�*

	conv_lossc,?7}��        )��P	C��쉙�A�*

	conv_loss�k,?��e�        )��P	�(�쉙�A�*

	conv_loss�],?����        )��P	Y�쉙�A�*

	conv_loss&_,?=�^R        )��P	��쉙�A�*

	conv_loss�_,?���\        )��P	��쉙�A�*

	conv_loss_\,?��f        )��P	���쉙�A�*

	conv_loss�U,?<t        )��P	n�쉙�A�*

	conv_loss\],?��sj        )��P	pJ�쉙�A�*

	conv_lossW,?��yp        )��P	1��쉙�A�*

	conv_loss�P,?��d�        )��P	2��쉙�A�*

	conv_loss T,?�i        )��P	i��쉙�A�*

	conv_loss�U,?⎖        )��P	��쉙�A�*

	conv_loss�S,?��        )��P	�D�쉙�A�*

	conv_loss?C,?����        )��P	�t�쉙�A�*

	conv_loss F,?A�<        )��P	O��쉙�A�*

	conv_loss�A,?��X        )��P	���쉙�A�*

	conv_loss�E,?o`        )��P	x��쉙�A�*

	conv_loss�3,?��s        )��P	{+�쉙�A�*

	conv_loss�;,?���         )��P	C\�쉙�A�*

	conv_loss B,?X���        )��P	��쉙�A�*

	conv_loss�C,?c~��        )��P	g��쉙�A�*

	conv_loss�@,?bp�9        )��P	���쉙�A�*

	conv_loss�.,?PaL|        )��P	��쉙�A�*

	conv_loss�9,?���        )��P	O�쉙�A�*

	conv_loss�8,?\*�        )��P	��쉙�A�*

	conv_loss�<,?��        )��P	���쉙�A�*

	conv_lossE4,?2�;f        )��P	���쉙�A�*

	conv_loss2,?Z3�        )��P	��쉙�A�*

	conv_loss�+,?�~��        )��P	�@�쉙�A�*

	conv_loss6),?t^��        )��P	q�쉙�A�*

	conv_loss�',?�]�        )��P	ޤ�쉙�A�*

	conv_loss|$,?K:�o        )��P	���쉙�A�*

	conv_loss�$,?�(G        )��P	��쉙�A�*

	conv_loss>#,?�R�        )��P	6�쉙�A�*

	conv_lossJ),?ϓwX        )��P	�h�쉙�A�*

	conv_loss�',?�y�        )��P	���쉙�A�*

	conv_loss'',?E�G�        )��P	���쉙�A�*

	conv_loss�,?Y�Rj        )��P	�	�쉙�A�*

	conv_loss�,? ��        )��P	�9�쉙�A�*

	conv_loss�,?n %_        )��P	3n�쉙�A�*

	conv_loss�,?�CB        )��P	��쉙�A�*

	conv_loss=,?�=Q        )��P	|��쉙�A�*

	conv_loss�,?�Z��        )��P	 �쉙�A�*

	conv_lossB,?d��        )��P	�4�쉙�A�*

	conv_loss�,?���s        )��P	�e�쉙�A�*

	conv_lossJ,?�O�        )��P	(��쉙�A�*

	conv_loss<,?I�        )��P	���쉙�A�*

	conv_loss�,?	d�+        )��P	��쉙�A�*

	conv_lossk,?F,\+        )��P	�7�쉙�A�*

	conv_loss�,?{��        )��P	@i�쉙�A�*

	conv_loss�,?���K        )��P	P��쉙�A�*

	conv_loss~	,?�햯        )��P	x��쉙�A�*

	conv_loss��+?yx`        )��P	���쉙�A�*

	conv_loss��+?�`�%        )��P	@0�쉙�A�*

	conv_lossl�+?^�        )��P	:_�쉙�A�*

	conv_loss��+?S��        )��P	���쉙�A�*

	conv_loss��+?,�        )��P	���쉙�A�*

	conv_loss��+?;l��        )��P	���쉙�A�*

	conv_loss��+?)�        )��P	�쉙�A�*

	conv_loss��+?�/�        )��P	AO�쉙�A�*

	conv_loss�+?�W	S        )��P	���쉙�A�*

	conv_loss��+?���        )��P	E��쉙�A�*

	conv_lossw�+?	\�        )��P	���쉙�A�*

	conv_loss�+?AX�        )��P	w�쉙�A�*

	conv_loss8�+?�N11        )��P	C�쉙�A�*

	conv_loss�+?�>`�        )��P	�r�쉙�A�*

	conv_loss��+?u���        )��P		��쉙�A�*

	conv_loss5�+?�H>�        )��P	���쉙�A�*

	conv_losss�+?�cj        )��P	��쉙�A�*

	conv_loss��+?�|�        )��P	�1�쉙�A�*

	conv_loss��+?-���        )��P	c�쉙�A�*

	conv_lossf�+?�ҟ-        )��P	���쉙�A�*

	conv_loss��+?݉�X        )��P	2��쉙�A�*

	conv_loss��+?��18        )��P	q��쉙�A�*

	conv_loss]�+?pd�m        )��P	j�쉙�A�*

	conv_loss��+?���7        )��P	�N�쉙�A�*

	conv_lossA�+?7�ʊ        )��P	n}�쉙�A�*

	conv_loss��+?%F5P        )��P	@��쉙�A�*

	conv_loss��+?H��        )��P	���쉙�A�*

	conv_loss��+?w�?0        )��P	��쉙�A�*

	conv_lossw�+?˞��        )��P	y;�쉙�A�*

	conv_lossѽ+?=�        )��P	Cl�쉙�A�*

	conv_loss��+?Wä!        )��P	���쉙�A�*

	conv_lossi�+?X        )��P	���쉙�A�*

	conv_lossʺ+?k�0�        )��P	e��쉙�A�*

	conv_lossV�+?*�{        )��P	�A�쉙�A�*

	conv_loss�+?��        )��P	q�쉙�A�*

	conv_loss��+?l��        )��P	���쉙�A�*

	conv_lossj�+?�AY        )��P	@��쉙�A�*

	conv_loss�+?}T        )��P	��쉙�A�*

	conv_lossѲ+?x6LT        )��P	 4�쉙�A�*

	conv_loss�+?��'        )��P	�f�쉙�A�*

	conv_lossj�+??�b�        )��P	��쉙�A�*

	conv_lossj�+?��r        )��P	���쉙�A�*

	conv_loss��+?n5        )��P	��쉙�A�*

	conv_loss��+?8��        )��P	�4�쉙�A�*

	conv_loss��+?�d�        )��P	�c�쉙�A�*

	conv_loss	�+?y�S         )��P	d��쉙�A�*

	conv_lossl�+?�چ�        )��P	��쉙�A�*

	conv_lossĕ+?"J5        )��P	.��쉙�A�*

	conv_loss��+?�|�{        )��P	�2�쉙�A�*

	conv_loss/�+?�ʁ        )��P	�f�쉙�A�*

	conv_lossD�+?'���        )��P	���쉙�A�*

	conv_loss�+?a        )��P	Q��쉙�A�*

	conv_loss
�+?�}��        )��P	���쉙�A�*

	conv_loss��+?:�؂        )��P	�(�쉙�A�*

	conv_lossՏ+?OB=�        )��P	
Y�쉙�A�*

	conv_loss�+?b:$�        )��P	��쉙�A�*

	conv_lossn�+?ĥ��        )��P	���쉙�A�*

	conv_loss�+?�H΁        )��P	���쉙�A�*

	conv_lossl�+?����        )��P	��쉙�A�*

	conv_loss��+?�Xnh        )��P	_M�쉙�A�*

	conv_loss�y+?yr��        )��P	D|�쉙�A�*

	conv_loss�+?v�J        )��P	d��쉙�A�*

	conv_loss�~+?��\        )��P	I��쉙�A�*

	conv_loss�~+?���        )��P	�
�쉙�A�*

	conv_lossz+?MX�        )��P	�;�쉙�A�*

	conv_loss.q+?��z        )��P	l�쉙�A�*

	conv_loss�q+?�Z'        )��P	���쉙�A�*

	conv_loss}q+?�7Q        )��P	*��쉙�A�*

	conv_loss�s+?�h �        )��P	v�쉙�A�*

	conv_lossk+?۰��        )��P	�:�쉙�A�*

	conv_loss9p+?�2�        )��P	�k�쉙�A�*

	conv_loss4n+?�.|�        )��P	��쉙�A�*

	conv_loss+a+?�,kK        )��P	S��쉙�A�*

	conv_lossj+?#Y�x        )��P	��쉙�A�*

	conv_loss&d+?�"�F        )��P	09�쉙�A�*

	conv_lossSb+?p�MH        )��P	l�쉙�A�*

	conv_loss<i+?X��        )��P	0��쉙�A�*

	conv_loss]+?�wy        )��P	��쉙�A�*

	conv_loss�_+?P�t?        )��P	6
�쉙�A�*

	conv_lossW+?b<��        )��P	�?�쉙�A�*

	conv_lossV]+?���        )��P	 u�쉙�A�*

	conv_loss+N+?O6�        )��P	w��쉙�A�*

	conv_loss3T+?��4        )��P	o��쉙�A�*

	conv_lossG+?Jg�        )��P	t$�쉙�A�*

	conv_loss�D+?�/
f        )��P	%Y�쉙�A�*

	conv_lossVN+?9+�b        )��P	O��쉙�A�*

	conv_loss>+?��I        )��P	G��쉙�A�*

	conv_loss�I+?bx/b        )��P	���쉙�A�*

	conv_loss�?+?��        )��P	/�쉙�A�*

	conv_loss�K+?���,        )��P	g�쉙�A�*

	conv_loss�E+?ȁ�m        )��P	[��쉙�A�*

	conv_lossc:+?�v�H        )��P	 ��쉙�A�*

	conv_loss�C+?����        )��P	}�쉙�A�*

	conv_loss�=+?U@a�        )��P	�8�쉙�A�*

	conv_lossLC+?t�Y�        )��P	�u�쉙�A�*

	conv_lossl6+?�8��        )��P	���쉙�A�*

	conv_lossu1+?�	�        )��P	~��쉙�A�*

	conv_loss�:+?�!>        )��P	&�쉙�A�*

	conv_loss:-+?d��        )��P	@P�쉙�A�*

	conv_loss�6+?�h��        )��P	���쉙�A�*

	conv_loss)+?��2z        )��P	���쉙�A�*

	conv_loss�'+?�	'n        )��P	���쉙�A�*

	conv_loss�2+?�f$$        )��P	K&�쉙�A�*

	conv_loss&+?�Xs        )��P	5Y�쉙�A�*

	conv_loss�+?=ٯ[        )��P	\��쉙�A�*

	conv_loss�%+?N�G*        )��P	���쉙�A�*

	conv_losss.+?�OE        )��P	���쉙�A�*

	conv_loss�+?ݥ6)        )��P	/*�쉙�A�*

	conv_loss+?�J$�        )��P	`^�쉙�A�*

	conv_loss{+?t+��        )��P	R��쉙�A�*

	conv_loss+?z�=�        )��P	=��쉙�A�*

	conv_loss�+?\��u        )��P	+��쉙�A�*

	conv_loss;+?� �        )��P	<1�쉙�A�*

	conv_loss+?L7u        )��P	�c�쉙�A�*

	conv_loss4+?�&=n        )��P	��쉙�A�*

	conv_loss_+? $�        )��P	)��쉙�A�*

	conv_lossx	+?�DF        )��P	�쉙�A�*

	conv_loss�	+?��6@        )��P	27�쉙�A�*

	conv_loss)+?$M	�        )��P	li�쉙�A�*

	conv_loss�	+?F�ɺ        )��P	@��쉙�A�*

	conv_loss�+?º        )��P	A��쉙�A�*

	conv_loss��*?!��        )��P	��쉙�A�*

	conv_lossL�*?�/��        )��P	]8�쉙�A�*

	conv_lossb +?BdL        )��P	4j�쉙�A�*

	conv_losst+?ͨ�        )��P	��쉙�A�*

	conv_loss�+?@�A{        )��P	��쉙�A�*

	conv_loss��*?��r        )��P	�쉙�A�*

	conv_loss��*?f VN        )��P	�7�쉙�A�*

	conv_loss&�*?��+        )��P	�j�쉙�A�*

	conv_loss��*?��@}        )��P	 ��쉙�A�*

	conv_loss��*?�N1        )��P	���쉙�A�*

	conv_loss��*?�c        )��P	`�쉙�A�*

	conv_loss�*?��e        )��P	O<�쉙�A�*

	conv_lossY�*?X���        )��P	U��쉙�A�*

	conv_loss"�*?\.�r        )��P	�
�쉙�A�*

	conv_loss��*?�;��        )��P	�:�쉙�A�*

	conv_lossM�*?�S��        )��P	�n�쉙�A�*

	conv_loss��*?%�R�        )��P	���쉙�A�*

	conv_lossn�*?J�*b        )��P	���쉙�A�*

	conv_loss�*?�!v'        )��P	h�쉙�A�*

	conv_loss��*?��=        )��P	�8�쉙�A�*

	conv_loss��*?�s�        )��P	�t�쉙�A�*

	conv_loss��*?��$�        )��P	���쉙�A�*

	conv_loss��*?��!        )��P	l��쉙�A�*

	conv_loss�*?C:Q-        )��P	�쉙�A�*

	conv_loss��*?LU0        )��P	BA�쉙�A�*

	conv_loss��*?%C�        )��P	q�쉙�A�*

	conv_lossW�*?��L1        )��P	��쉙�A�*

	conv_lossi�*?���        )��P	���쉙�A�*

	conv_loss6�*?���        )��P	]�쉙�A�*

	conv_loss��*?׮�.        )��P	1A�쉙�A�*

	conv_lossG�*?ftS        )��P	�s�쉙�A�*

	conv_loss��*?sǐ\        )��P	ި�쉙�A�*

	conv_loss��*?�㷮        )��P	v��쉙�A�*

	conv_loss^�*?��        )��P	d�쉙�A�*

	conv_losst�*?uiL        )��P	�A�쉙�A�*

	conv_loss��*?�qԬ        )��P	�q�쉙�A�*

	conv_loss[�*?L�        )��P	3��쉙�A�*

	conv_loss�*?F�        )��P	]��쉙�A�*

	conv_lossӽ*?��        )��P	��쉙�A�*

	conv_loss<�*?�iQ        )��P	M5�쉙�A�*

	conv_loss�*?U��        )��P	�c�쉙�A�*

	conv_lossȰ*?�&�        )��P	N��쉙�A�*

	conv_lossس*?N�#�        )��P	���쉙�A�*

	conv_loss�*?��W�        )��P	d��쉙�A�*

	conv_lossj�*?`�{�        )��P	�#�쉙�A�*

	conv_loss��*?͑H�        )��P	V�쉙�A�*

	conv_loss_�*?[�y        )��P	���쉙�A�*

	conv_lossT�*?VJ֫        )��P	Ķ�쉙�A�*

	conv_loss�*?�        )��P	���쉙�A�*

	conv_loss�*?EN �        )��P	��쉙�A�*

	conv_loss�*?!l�        )��P	K�쉙�A�*

	conv_loss�*?��|4        )��P	�|�쉙�A�*

	conv_lossS�*?Jm��        )��P	���쉙�A�*

	conv_loss�*?�DV�        )��P	���쉙�A�*

	conv_loss3�*?�c��        )��P	�쉙�A�*

	conv_loss��*?�b��        )��P	�@�쉙�A�*

	conv_loss8�*?�"��        )��P	np�쉙�A�*

	conv_loss=�*?{��f        )��P	���쉙�A�*

	conv_lossވ*?�P�E        )��P	C��쉙�A�*

	conv_loss��*?����        )��P	r �쉙�A�*

	conv_loss^�*?�ě�        )��P	�1�쉙�A�*

	conv_loss&�*?���        )��P	bc�쉙�A�*

	conv_loss�t*?q�C        )��P	���쉙�A�*

	conv_lossv*?��        )��P	���쉙�A�*

	conv_loss}*?�Q3�        )��P	!�쉙�A�*

	conv_lossm|*?��        )��P	�A�쉙�A�*

	conv_lossQt*?T�H5        )��P	js�쉙�A�*

	conv_lossu*?��!�        )��P	���쉙�A�*

	conv_loss�q*?�U�o        )��P	_��쉙�A�*

	conv_lossex*?���I        )��P	\�쉙�A�*

	conv_loss�t*?��>�        )��P	QV�쉙�A�*

	conv_lossu*?��        )��P	l��쉙�A�*

	conv_lossG|*?�ߛ�        )��P	��쉙�A�*

	conv_loss6y*?��5�        )��P	]��쉙�A�*

	conv_loss�i*?��3        )��P	c$�쉙�A�*

	conv_loss�h*?���4        )��P	 V�쉙�A�*

	conv_loss�c*?j�;�        )��P	���쉙�A�*

	conv_loss&]*?=��        )��P	���쉙�A�*

	conv_lossg`*?�y_(        )��P	���쉙�A�*

	conv_lossJg*?���         )��P	� 퉙�A�*

	conv_losszY*?�x�        )��P	eF 퉙�A�*

	conv_loss�\*?l<h�        )��P	Sv 퉙�A�*

	conv_lossX_*?�w�R        )��P	Y� 퉙�A�*

	conv_loss%^*?s[dB        )��P	7� 퉙�A�*

	conv_loss Y*?�hA        )��P	퉙�A�*

	conv_loss[*?��        )��P	F퉙�A�*

	conv_loss�R*?�NIT        )��P	�u퉙�A�*

	conv_loss�M*?4�p�        )��P	�퉙�A�*

	conv_lossF*?��T�        )��P	|�퉙�A�*

	conv_loss�N*?t         )��P	7퉙�A�*

	conv_loss.Q*?܁��        )��P	�5퉙�A�*

	conv_lossjL*?_         )��P	-d퉙�A�*

	conv_loss�M*?l�[        )��P	�퉙�A�*

	conv_loss�K*?���        )��P	��퉙�A�*

	conv_lossuG*?,��        )��P	�퉙�A�*

	conv_loss�;*?&NZa        )��P	 퉙�A�*

	conv_loss�<*?���t        )��P	�O퉙�A�*

	conv_lossy@*?�Y�+        )��P	`~퉙�A�*

	conv_loss�7*?,�.P        )��P	>�퉙�A�*

	conv_loss�;*?�0�R        )��P	��퉙�A�*

	conv_loss�5*?��        )��P	=퉙�A�*

	conv_loss�0*?O�u�        )��P	�J퉙�A�*

	conv_lossC4*? ��        )��P	�{퉙�A�*

	conv_loss1*?A'	        )��P	5�퉙�A�*

	conv_loss%0*?��        )��P	8�퉙�A�*

	conv_loss	4*?{�p<        )��P	T퉙�A�*

	conv_lossA1*?����        )��P	�?퉙�A�*

	conv_loss�$*?m1�k        )��P	p퉙�A�*

	conv_loss�/*?H&�z        )��P	��퉙�A�*

	conv_loss�-*?%ѿ        )��P	�퉙�A�*

	conv_lossq*?��        )��P	T퉙�A�*

	conv_loss!*?�d        )��P	#D퉙�A�*

	conv_lossI"*?���        )��P	��퉙�A�*

	conv_loss�*?iPK�        )��P	,�퉙�A�*

	conv_loss�*?�U        )��P	�퉙�A�*

	conv_loss�*?}mN        )��P	!퉙�A�*

	conv_loss"*?��k�        )��P	�P퉙�A�*

	conv_loss*?l�=�        )��P	"�퉙�A�*

	conv_loss\*?K���        )��P	�퉙�A�*

	conv_loss*?�s�        )��P	8�퉙�A�*

	conv_lossR*?�I�        )��P	�#퉙�A�*

	conv_loss6*?�'�        )��P	_퉙�A�*

	conv_loss5*?��k        )��P	�퉙�A�*

	conv_loss*?l�Q�        )��P	�퉙�A�*

	conv_loss�*?��!        )��P	I�퉙�A�*

	conv_loss[*?�7��        )��P	S3	퉙�A�*

	conv_loss*?��        )��P	�c	퉙�A�*

	conv_lossr*?��        )��P	n�	퉙�A�*

	conv_loss|*?.��S        )��P	n�	퉙�A�*

	conv_loss'�)?���        )��P	��	퉙�A�*

	conv_loss�)?l��        )��P	�!
퉙�A�*

	conv_lossL�)?���y        )��P	jS
퉙�A�*

	conv_loss��)?�ϛ�        )��P	��
퉙�A�*

	conv_lossk�)?�ۨ	        )��P	V�
퉙�A�*

	conv_loss��)?=��        )��P	=�
퉙�A�*

	conv_losse�)?L�ԕ        )��P	�!퉙�A�*

	conv_lossE�)?P\�        )��P	�Q퉙�A�*

	conv_loss��)?b�`�        )��P	T�퉙�A�*

	conv_loss��)?&eM        )��P	��퉙�A�*

	conv_loss��)?9E        )��P	B�퉙�A�*

	conv_loss��)?���        )��P	x퉙�A�*

	conv_lossi�)?�H        )��P	}?퉙�A�*

	conv_loss��)?L4�u        )��P	�n퉙�A�*

	conv_lossZ�)?�:��        )��P	X�퉙�A�*

	conv_loss/�)?�Ľ        )��P	�퉙�A�*

	conv_lossr�)?���u        )��P	��퉙�A�*

	conv_loss��)?�Cu�        )��P	Z.퉙�A�*

	conv_loss��)?VP�2        )��P	E]퉙�A�*

	conv_loss��)?C8h        )��P	ُ퉙�A�*

	conv_lossx�)?$c�        )��P	+�퉙�A�*

	conv_lossI�)?�]�M        )��P	�퉙�A�*

	conv_lossu�)?Y��
        )��P	�"퉙�A�*

	conv_loss��)?Eo��        )��P	�P퉙�A�*

	conv_loss�)?a\�Q        )��P	�}퉙�A�*

	conv_lossH�)?��o        )��P	�퉙�A�*

	conv_loss��)?Ã�)        )��P	Q�퉙�A�*

	conv_loss��)?Qw&�        )��P	�퉙�A�*

	conv_loss��)?�r        )��P	t:퉙�A�*

	conv_loss�)?C@�b        )��P	Fi퉙�A�*

	conv_lossa�)?�b-        )��P	C�퉙�A�*

	conv_lossu�)?�h|        )��P	�퉙�A�*

	conv_loss��)?�+o        )��P	5�퉙�A�*

	conv_loss��)?t���        )��P	�6퉙�A�*

	conv_loss#�)?�:V�        )��P	�h퉙�A�*

	conv_loss�)?��j        )��P	˗퉙�A�*

	conv_loss>�)?�`�        )��P	��퉙�A�*

	conv_lossN�)?���        )��P	?�퉙�A�*

	conv_loss+�)?�T�        )��P	B,퉙�A�*

	conv_loss�)?��uG        )��P	�g퉙�A�*

	conv_loss��)?��p�        )��P	��퉙�A�*

	conv_lossD�)?E���        )��P	��퉙�A�*

	conv_loss*�)?:[R�        )��P	�퉙�A�*

	conv_loss��)?o#�5        )��P	�5퉙�A�*

	conv_loss��)?<�        )��P	�x퉙�A�*

	conv_lossC�)?�Q)b        )��P	z�퉙�A�*

	conv_losss�)?̳�H        )��P	��퉙�A�*

	conv_loss�)? 7�        )��P	`퉙�A�*

	conv_loss��)?`bĦ        )��P	wF퉙�A�*

	conv_lossƫ)?��E�        )��P	�u퉙�A�*

	conv_lossċ)?K���        )��P	��퉙�A�*

	conv_loss�)?v}^        )��P	�퉙�A�*

	conv_loss��)?ǐh        )��P	{퉙�A�*

	conv_loss��)?idN�        )��P	�4퉙�A�*

	conv_loss��)?�)�        )��P	Lf퉙�A�*

	conv_loss<�)?����        )��P	��퉙�A�*

	conv_loss)?ɢ֩        )��P	x�퉙�A�*

	conv_loss&�)?�}�7        )��P	3�퉙�A�*

	conv_lossd�)?��"�        )��P	:!퉙�A�*

	conv_lossy)?���        )��P	�Q퉙�A�*

	conv_loss�r)?8�E        )��P	=�퉙�A�*

	conv_loss�)?�!��        )��P	F�퉙�A�*

	conv_loss�|)?H�&7        )��P	�퉙�A�*

	conv_loss|)?93�        )��P	�퉙�A�*

	conv_lossz)?zs~�        )��P	�C퉙�A�*

	conv_loss�n)?���        )��P	t퉙�A�*

	conv_loss4y)?rb��        )��P	d�퉙�A�*

	conv_lossOq)?��L�        )��P	��퉙�A�*

	conv_lossSi)?4tJ�        )��P	t퉙�A�*

	conv_loss�m)?@"�        )��P	�2퉙�A�*

	conv_lossfr)?	�c�        )��P	;b퉙�A�*

	conv_loss�x)?*�        )��P	��퉙�A�*

	conv_loss�f)?��        )��P	��퉙�A�*

	conv_loss�c)?�B        )��P	��퉙�A�*

	conv_lossam)?��        )��P	n$퉙�A�*

	conv_loss-j)?B_"        )��P	!U퉙�A�*

	conv_loss�`)?��X        )��P	U�퉙�A�*

	conv_lossea)?4i�h        )��P	6�퉙�A�*

	conv_loss�_)?F(��        )��P	��퉙�A�*

	conv_loss=[)?őH@        )��P	m퉙�A�*

	conv_loss�^)?��b        )��P	�H퉙�A�*

	conv_lossZ)?��        )��P	�w퉙�A�*

	conv_loss�Y)?�9~�        )��P	��퉙�A�*

	conv_lossZS)?��        )��P	�w퉙�A�*

	conv_lossH)?�m	        )��P	��퉙�A�*

	conv_lossT)?����        )��P	��퉙�A�*

	conv_losshR)?f�k�        )��P	a 퉙�A�*

	conv_loss�A)?�/z        )��P	�5 퉙�A�*

	conv_loss�E)?~�Ak        )��P	�e 퉙�A�*

	conv_losse@)?1f�        )��P	.� 퉙�A�*

	conv_loss�G)?�<,        )��P	�� 퉙�A�*

	conv_lossNK)?g��        )��P	�!퉙�A�*

	conv_loss)>)?'-&�        )��P	�4!퉙�A�*

	conv_loss�>)?{( 4        )��P	�f!퉙�A�*

	conv_loss`C)?��        )��P	<�!퉙�A�*

	conv_lossr9)?6��        )��P	e�!퉙�A�*

	conv_loss�=)?�|��        )��P	|"퉙�A�*

	conv_loss4)?,̒        )��P	B"퉙�A�*

	conv_loss�3)?Ó�	        )��P	�q"퉙�A�*

	conv_loss&)?���        )��P	��"퉙�A�*

	conv_lossV')?���        )��P	-�"퉙�A�*

	conv_loss�2)?��        )��P	T�"퉙�A�*

	conv_loss�.)?���L        )��P	�,#퉙�A�*

	conv_loss�1)?����        )��P	a[#퉙�A�*

	conv_loss�0)?�W�[        )��P	��#퉙�A�*

	conv_loss[,)?)/��        )��P	��#퉙�A�*

	conv_loss�()?~��        )��P	6�#퉙�A�*

	conv_loss�)?�n�        )��P	$퉙�A�*

	conv_loss
')?��        )��P	�E$퉙�A�*

	conv_loss�)?��X        )��P	�u$퉙�A�*

	conv_loss1)?����        )��P	 �$퉙�A�*

	conv_loss*)?H��        )��P	��$퉙�A�*

	conv_loss)?x˦        )��P	�%퉙�A�*

	conv_loss� )?��7�        )��P	�1%퉙�A�*

	conv_loss)?z�>w        )��P	_%퉙�A�*

	conv_loss)?��a�        )��P	�%퉙�A�*

	conv_loss�)?�)g        )��P	.�%퉙�A�*

	conv_loss)?����        )��P	��%퉙�A�*

	conv_loss�)?9�d�        )��P	-"&퉙�A�*

	conv_lossv)?�%?�        )��P	xS&퉙�A�*

	conv_loss�)?�v@2        )��P	'�&퉙�A�*

	conv_loss� )?W��F        )��P	j�&퉙�A�*

	conv_lossZ)?hDr)        )��P	��&퉙�A�*

	conv_loss�)?�7�b        )��P	�'퉙�A�*

	conv_lossY�(?��B�        )��P	�G'퉙�A�*

	conv_loss��(?��        )��P	�y'퉙�A�*

	conv_lossz�(?)6�v        )��P	̪'퉙�A�*

	conv_lossj)? (�        )��P	��'퉙�A�*

	conv_lossu�(?>;�d        )��P	�(퉙�A�*

	conv_loss��(?        )��P	�<(퉙�A�*

	conv_loss8�(?�j�W        )��P	�n(퉙�A�*

	conv_loss�(?l��        )��P	a�(퉙�A�*

	conv_lossB�(?�=ߝ        )��P	�(퉙�A�*

	conv_lossR�(?�RN2        )��P	�)퉙�A�*

	conv_loss��(?��u�        )��P	wG)퉙�A�*

	conv_lossI�(?%�?        )��P	-{)퉙�A�*

	conv_loss~�(?FB�        )��P	C�)퉙�A�*

	conv_loss��(?f�        )��P	��)퉙�A�*

	conv_loss��(?����        )��P	*퉙�A�*

	conv_lossP�(?-��"        )��P	�I*퉙�A�*

	conv_loss�(?����        )��P	gy*퉙�A�*

	conv_loss"�(?0�h�        )��P	F�*퉙�A�*

	conv_loss��(?��G~        )��P	�*퉙�A�*

	conv_loss��(?�Ïg        )��P	�+퉙�A�*

	conv_loss��(?o(޵        )��P	9U+퉙�A�*

	conv_loss�(?S<�        )��P	6�+퉙�A�*

	conv_loss��(?M�w@        )��P	��+퉙�A�*

	conv_loss��(?�|g�        )��P	� ,퉙�A�*

	conv_loss��(?P�t}        )��P	52,퉙�A�*

	conv_loss�(?-6�Z        )��P	(b,퉙�A�*

	conv_loss��(?k�V        )��P	��,퉙�A�*

	conv_loss��(?i�x�        )��P	`�,퉙�A�*

	conv_loss�(?|1nN        )��P	n�,퉙�A�*

	conv_loss�(?�`��        )��P	i'-퉙�A�*

	conv_loss��(?�f�        )��P	�W-퉙�A�*

	conv_losso�(?Y|l�        )��P	,�-퉙�A�*

	conv_lossG�(?j�P�        )��P	�-퉙�A�*

	conv_lossA�(?ű,        )��P	��-퉙�A�*

	conv_loss��(?�!��        )��P	�.퉙�A�*

	conv_loss�(?�2�        )��P	H.퉙�A�*

	conv_loss��(?����        )��P	gz.퉙�A�*

	conv_loss��(?�o��        )��P	_�.퉙�A�*

	conv_loss͸(?)<`        )��P	W�.퉙�A�*

	conv_lossd�(?vf�        )��P	e/퉙�A�*

	conv_lossD�(?Ē�        )��P	=/퉙�A�*

	conv_lossɰ(?�u�        )��P	�m/퉙�A�*

	conv_loss��(?�A�        )��P	�/퉙�A�*

	conv_lossp�(?��N9        )��P	��/퉙�A�*

	conv_loss��(?�@�        )��P	w0퉙�A�*

	conv_loss �(?�\�        )��P	�10퉙�A�*

	conv_losst�(?a{�4        )��P	�c0퉙�A�*

	conv_loss�(?��Js        )��P	͓0퉙�A�*

	conv_loss��(?�~��        )��P	)�0퉙�A�*

	conv_lossi�(?�)�        )��P	M�0퉙�A�*

	conv_lossΕ(?�f��        )��P	&1퉙�A�*

	conv_loss�(?Q��        )��P	iV1퉙�A�*

	conv_loss��(?8�,        )��P	�1퉙�A�*

	conv_loss��(?+���        )��P	*�1퉙�A�*

	conv_loss��(?8V1�        )��P	��1퉙�A�*

	conv_lossM�(?�?��        )��P	R2퉙�A�*

	conv_loss��(?��4b        )��P	�I2퉙�A�*

	conv_loss>�(?�g/�        )��P	�y2퉙�A�*

	conv_losse�(?D�r�        )��P	Ϋ2퉙�A�*

	conv_loss7�(?�@Lq        )��P	_�2퉙�A�*

	conv_loss��(?��0�        )��P	l3퉙�A�*

	conv_loss�(?l�        )��P	~N3퉙�A�*

	conv_loss��(?��[        )��P	L�3퉙�A�*

	conv_lossb�(?U,��        )��P	V�3퉙�A�*

	conv_lossÄ(?b�0�        )��P	��3퉙�A�*

	conv_loss�s(?6��        )��P	�4퉙�A�*

	conv_lossXw(?1-�        )��P	�J4퉙�A�*

	conv_lossT�(?�~C        )��P	ˈ4퉙�A�*

	conv_loss�x(?�t'�        )��P	}�4퉙�A�*

	conv_loss�w(?v �        )��P	��4퉙�A�*

	conv_loss�o(?���        )��P	I!5퉙�A�*

	conv_loss�c(?��0^        )��P	�Q5퉙�A�*

	conv_lossQl(?��:        )��P	��5퉙�A�*

	conv_lossa(?��#        )��P	�5퉙�A�*

	conv_loss�n(?l���        )��P	v 6퉙�A�*

	conv_loss�l(?7u�        )��P	�26퉙�A�*

	conv_lossng(?�6��        )��P	Ve6퉙�A�*

	conv_lossXp(?��\�        )��P	��6퉙�A�*

	conv_loss�p(?`v�G        )��P	��6퉙�A�*

	conv_loss/i(?\�F�        )��P	\�6퉙�A�*

	conv_loss�Z(?~�'        )��P	�)7퉙�A�*

	conv_lossbj(?N��        )��P	�[7퉙�A�*

	conv_loss�](?)F        )��P	��7퉙�A�*

	conv_loss�Z(?'��        )��P	н7퉙�A�*

	conv_lossHY(?&�|        )��P	��7퉙�A�*

	conv_lossWP(?*��        )��P	�!8퉙�A�*

	conv_lossNO(?�F31        )��P	�U8퉙�A�*

	conv_loss�U(?f;        )��P	Q�8퉙�A�*

	conv_loss�[(?k6�        )��P	�8퉙�A�*

	conv_loss�F(?G��        )��P	��8퉙�A�*

	conv_loss"E(?�L4�        )��P	8 9퉙�A�*

	conv_loss�H(?-mi        )��P	Q9퉙�A�*

	conv_loss�D(?�B��        )��P	^�9퉙�A�*

	conv_loss�>(?꾪r        )��P	��9퉙�A�*

	conv_losslA(?rFY        )��P	��9퉙�A�*

	conv_loss�O(?2�
Q        )��P	�:퉙�A�*

	conv_loss�F(?�I�        )��P	�I:퉙�A�*

	conv_lossD(?hX@,        )��P	�{:퉙�A�*

	conv_loss�C(?�        )��P	1�:퉙�A�*

	conv_loss�<(?ޓ�@        )��P	��:퉙�A�*

	conv_lossY8(?�y?�        )��P	�;퉙�A�*

	conv_loss�B(?�f�c        )��P	w@;퉙�A�*

	conv_loss9:(?�>�;        )��P	�q;퉙�A�*

	conv_loss::(?	 OU        )��P	
�;퉙�A�*

	conv_loss+(?C�        )��P	��;퉙�A�*

	conv_loss�3(?�g
        )��P	<퉙�A�*

	conv_loss�*(?YnJ@        )��P	d:<퉙�A�*

	conv_lossh0(?���        )��P	0u<퉙�A�*

	conv_lossD-(?o䚕        )��P	Y�<퉙�A�*

	conv_loss�0(?����        )��P	>�<퉙�A�*

	conv_loss�(?�K(        )��P	n4=퉙�A�	*

	conv_loss�(?��%�        )��P	+m=퉙�A�	*

	conv_losst(?%_4        )��P	��=퉙�A�	*

	conv_loss�(?X�        )��P	�=퉙�A�	*

	conv_loss(?�'�        )��P	%
>퉙�A�	*

	conv_loss�(?���        )��P	=>퉙�A�	*

	conv_lossE(?�        )��P	�n>퉙�A�	*

	conv_loss�(?Q�Y        )��P	!�>퉙�A�	*

	conv_lossk(?��        )��P	��>퉙�A�	*

	conv_lossL
(?a!e�        )��P	&?퉙�A�	*

	conv_loss?(?|�W        )��P	�B?퉙�A�	*

	conv_loss�	(? ���        )��P	�s?퉙�A�	*

	conv_loss/(?fE�        )��P	��?퉙�A�	*

	conv_loss�(?��8�        )��P	��?퉙�A�	*

	conv_lossU(?��`        )��P	�#@퉙�A�	*

	conv_loss�(?,r]y        )��P	`W@퉙�A�	*

	conv_loss! (?��        )��P	ˉ@퉙�A�	*

	conv_loss(?���h        )��P	�@퉙�A�	*

	conv_loss�(?��        )��P	�@퉙�A�	*

	conv_loss� (?�'Zk        )��P	[!A퉙�A�	*

	conv_loss�(?<��M        )��P	�TA퉙�A�	*

	conv_loss��'?!�SK        )��P	��A퉙�A�	*

	conv_loss{�'?��X        )��P	`�A퉙�A�	*

	conv_loss^�'? 	�        )��P	5�A퉙�A�	*

	conv_loss��'?��:J        )��P	�B퉙�A�	*

	conv_loss>�'? @�b        )��P	BQB퉙�A�	*

	conv_loss��'?W�#        )��P	b�B퉙�A�	*

	conv_loss��'?:_[        )��P	�B퉙�A�	*

	conv_loss��'?��b        )��P	��B퉙�A�	*

	conv_loss�'?�|�        )��P	C퉙�A�	*

	conv_loss�'?���        )��P	WLC퉙�A�	*

	conv_loss��'?zf6�        )��P	2}C퉙�A�	*

	conv_loss��'?�PI        )��P	��C퉙�A�	*

	conv_loss&�'?4�@�        )��P	��C퉙�A�	*

	conv_loss-�'?���        )��P	~D퉙�A�	*

	conv_loss��'?�I]�        )��P	CD퉙�A�	*

	conv_loss��'?��        )��P	uD퉙�A�	*

	conv_loss&�'?�@w�        )��P	��D퉙�A�	*

	conv_lossp�'?Y�oF        )��P	^�D퉙�A�	*

	conv_loss��'?��        )��P	E퉙�A�	*

	conv_loss8�'?oݍ#        )��P	1?E퉙�A�	*

	conv_lossP�'?^I��        )��P	=qE퉙�A�	*

	conv_loss��'?�̂�        )��P	��E퉙�A�	*

	conv_loss��'?%�9�        )��P	��E퉙�A�	*

	conv_loss.�'?��        )��P	�F퉙�A�	*

	conv_loss��'?U�        )��P	�9F퉙�A�	*

	conv_loss3�'?��`        )��P	HlF퉙�A�	*

	conv_loss��'?-Ip
        )��P	G�F퉙�A�	*

	conv_loss��'?��(�        )��P	��F퉙�A�	*

	conv_loss��'?y=�        )��P	(eH퉙�A�	*

	conv_loss��'?K���        )��P	4�H퉙�A�	*

	conv_loss0�'?��֘        )��P	:�H퉙�A�	*

	conv_loss0�'?�5�        )��P	��H퉙�A�	*

	conv_loss��'?�=-�        )��P	<-I퉙�A�	*

	conv_loss �'?W��G        )��P	�]I퉙�A�	*

	conv_loss�'?���        )��P	��I퉙�A�	*

	conv_lossX�'?��E�        )��P	��I퉙�A�	*

	conv_loss�'?�L�        )��P	�	J퉙�A�	*

	conv_lossû'?f8�        )��P	.BJ퉙�A�	*

	conv_lossl�'?�k��        )��P	�sJ퉙�A�	*

	conv_loss��'?$m        )��P	��J퉙�A�	*

	conv_lossC�'?�ɝ        )��P	��J퉙�A�	*

	conv_lossī'?��=�        )��P	�K퉙�A�	*

	conv_loss��'?��        )��P	�>K퉙�A�	*

	conv_loss�'?����        )��P	wqK퉙�A�	*

	conv_lossۨ'?��N1        )��P	�K퉙�A�	*

	conv_loss��'?�eZ        )��P	G�K퉙�A�	*

	conv_lossқ'?x]�        )��P	�L퉙�A�	*

	conv_lossN�'?te�q        )��P	4L퉙�A�	*

	conv_loss��'?�Ġ        )��P	�hL퉙�A�	*

	conv_loss��'?�mB        )��P	��L퉙�A�	*

	conv_lossڍ'?Y��        )��P	d�L퉙�A�	*

	conv_lossJ�'?�-��        )��P	WM퉙�A�	*

	conv_loss(�'?��Y�        )��P	5M퉙�A�	*

	conv_loss��'?�~@        )��P	�dM퉙�A�	*

	conv_loss5�'?vv        )��P	{�M퉙�A�	*

	conv_losss�'?IDV        )��P	��M퉙�A�	*

	conv_loss��'?�0^        )��P	��M퉙�A�	*

	conv_lossx�'?C�e        )��P	0&N퉙�A�	*

	conv_loss�'?����        )��P	qUN퉙�A�	*

	conv_loss�z'?���1        )��P	T�N퉙�A�	*

	conv_lossV�'?T6��        )��P	O�N퉙�A�	*

	conv_lossՉ'?�W�#        )��P	��N퉙�A�	*

	conv_lossz'?<��'        )��P	|O퉙�A�	*

	conv_loss�w'?�e��        )��P	mQO퉙�A�	*

	conv_loss�|'?��        )��P	u�O퉙�A�	*

	conv_loss�p'?�_7�        )��P	ôO퉙�A�	*

	conv_lossl'?kO�        )��P	t�O퉙�A�	*

	conv_lossit'?)P�7        )��P	�P퉙�A�	*

	conv_loss�o'?�^�        )��P	;GP퉙�A�	*

	conv_loss�o'?2��l        )��P	�yP퉙�A�	*

	conv_lossfi'?lp*>        )��P	"�P퉙�A�	*

	conv_loss�g'?�6�        )��P	)�P퉙�A�	*

	conv_loss�j'?�鮷        )��P	KQ퉙�A�	*

	conv_loss�j'?z_�P        )��P	#JQ퉙�A�	*

	conv_lossc'?�S
S        )��P	q�Q퉙�A�	*

	conv_loss-h'?�3�        )��P	�Q퉙�A�	*

	conv_loss�g'?�~�l        )��P	c�Q퉙�A�	*

	conv_loss�_'?���        )��P	� R퉙�A�	*

	conv_loss�\'?�)V�        )��P	TfR퉙�A�	*

	conv_loss�_'?�MH        )��P	՘R퉙�A�	*

	conv_loss�^'?-�[        )��P	��R퉙�A�	*

	conv_loss�Q'?��O�        )��P	��R퉙�A�	*

	conv_losslP'?�q��        )��P	.S퉙�A�	*

	conv_lossyW'?#=j�        )��P	{_S퉙�A�	*

	conv_losskS'?j�o
        )��P	ќS퉙�A�	*

	conv_loss�N'?��W&        )��P	�S퉙�A�	*

	conv_loss�I'?1���        )��P	�T퉙�A�	*

	conv_loss+['?��r!        )��P	�LT퉙�A�	*

	conv_loss|;'?l�Y�        )��P	:~T퉙�A�	*

	conv_lossH'?���        )��P	��T퉙�A�	*

	conv_loss�P'?��D�        )��P	�T퉙�A�	*

	conv_loss�>'?^ta�        )��P	SU퉙�A�	*

	conv_loss3'?�9��        )��P	�CU퉙�A�	*

	conv_lossM@'?�l�>        )��P	9tU퉙�A�	*

	conv_loss:'?0�        )��P	F�U퉙�A�	*

	conv_lossv<'?�9l�        )��P	��U퉙�A�	*

	conv_loss�:'?��j�        )��P	V퉙�A�	*

	conv_loss�2'?b��)        )��P	.6V퉙�A�	*

	conv_loss�0'?���\        )��P	AkV퉙�A�	*

	conv_loss6'?
��        )��P	i�V퉙�A�	*

	conv_loss!,'?���        )��P	f�V퉙�A�	*

	conv_loss�/'?_X�        )��P	<W퉙�A�	*

	conv_loss�*'?���        )��P	7W퉙�A�	*

	conv_losse,'?���g        )��P	hW퉙�A�	*

	conv_loss}/'?h@��        )��P	�W퉙�A�	*

	conv_loss"'?�W�        )��P	��W퉙�A�	*

	conv_loss�!'?�        )��P	��W퉙�A�	*

	conv_loss�)'?���        )��P	�*X퉙�A�
*

	conv_lossT'?f�~        )��P	�[X퉙�A�
*

	conv_loss�'?�D�        )��P	ƌX퉙�A�
*

	conv_loss4('?�c�]        )��P	��X퉙�A�
*

	conv_loss '?M^        )��P	�X퉙�A�
*

	conv_loss'?j�o�        )��P	�!Y퉙�A�
*

	conv_loss�'?�<        )��P	�QY퉙�A�
*

	conv_lossI'?@`ID        )��P	�Y퉙�A�
*

	conv_loss�	'?�K        )��P	N�Y퉙�A�
*

	conv_loss�'?`"��        )��P	�Y퉙�A�
*

	conv_loss�'?Go��        )��P	BZ퉙�A�
*

	conv_loss('?.�j        )��P	YKZ퉙�A�
*

	conv_loss�'?�f`        )��P	dZ퉙�A�
*

	conv_loss0'?Ȉ�        )��P	Y�Z퉙�A�
*

	conv_loss�'?��d�        )��P	�Z퉙�A�
*

	conv_loss�'?�6�        )��P	�[퉙�A�
*

	conv_loss��&?'�q        )��P	,D[퉙�A�
*

	conv_lossI	'?Y=F        )��P	Ku[퉙�A�
*

	conv_loss�'?�)�        )��P	�[퉙�A�
*

	conv_lossQ	'?�'�        )��P	��[퉙�A�
*

	conv_loss>�&?�;�        )��P	b\퉙�A�
*

	conv_loss��&?�G        )��P	
M\퉙�A�
*

	conv_loss��&?ۻ��        )��P	>\퉙�A�
*

	conv_loss��&?��N�        )��P	��\퉙�A�
*

	conv_loss��&?uh��        )��P	$�\퉙�A�
*

	conv_loss��&?�߯�        )��P	�]퉙�A�
*

	conv_lossT�&?j+�S        )��P	E]퉙�A�
*

	conv_lossd�&?y��        )��P	$v]퉙�A�
*

	conv_loss]�&?�zn{        )��P	ܩ]퉙�A�
*

	conv_lossV�&?c5/�        )��P	J�]퉙�A�
*

	conv_loss�&?���        )��P	9!^퉙�A�
*

	conv_loss��&?j.�         )��P	�V^퉙�A�
*

	conv_loss��&?+�"�        )��P	��^퉙�A�
*

	conv_loss�&?m��        )��P	��^퉙�A�
*

	conv_loss��&?��[        )��P	��^퉙�A�
*

	conv_loss�&?��?F        )��P	�!_퉙�A�
*

	conv_loss��&?��t�        )��P	�U_퉙�A�
*

	conv_loss��&?"�:        )��P	��_퉙�A�
*

	conv_loss��&?��O        )��P	Z�_퉙�A�
*

	conv_loss��&? �5�        )��P	��_퉙�A�
*

	conv_loss�&?G���        )��P	]`퉙�A�
*

	conv_loss{�&?NA�        )��P	�F`퉙�A�
*

	conv_loss�&?Y�!�        )��P	��`퉙�A�
*

	conv_lossD�&?D��        )��P	V�`퉙�A�
*

	conv_loss*�&?�auN        )��P	��`퉙�A�
*

	conv_loss��&?��        )��P	Aa퉙�A�
*

	conv_loss��&?�j��        )��P	YNa퉙�A�
*

	conv_loss�&?7��        )��P	�|a퉙�A�
*

	conv_loss��&?V�C        )��P	�a퉙�A�
*

	conv_loss��&?-�ͽ        )��P	_�a퉙�A�
*

	conv_loss[�&?65��        )��P	�b퉙�A�
*

	conv_loss��&?�P��        )��P	�@b퉙�A�
*

	conv_loss|�&?Sٖ�        )��P	�ob퉙�A�
*

	conv_lossL�&?lzD�        )��P	��b퉙�A�
*

	conv_loss �&?xG�        )��P	��b퉙�A�
*

	conv_losst�&?��r�        )��P	�c퉙�A�
*

	conv_loss&�&?�`�        )��P	�6c퉙�A�
*

	conv_loss�&?6A�$        )��P	hc퉙�A�
*

	conv_loss��&?>��g        )��P	��c퉙�A�
*

	conv_loss��&?�9�,        )��P	B�c퉙�A�
*

	conv_loss�&?�q        )��P	��c퉙�A�
*

	conv_loss�&?L�fz        )��P	#,d퉙�A�
*

	conv_loss|�&?c2�C        )��P	�[d퉙�A�
*

	conv_loss�&?<�@3        )��P	!�d퉙�A�
*

	conv_lossL�&?����        )��P	��d퉙�A�
*

	conv_lossi�&?Ň��        )��P	��d퉙�A�
*

	conv_loss��&?�d'        )��P	~e퉙�A�
*

	conv_loss,�&?J/��        )��P	�Ne퉙�A�
*

	conv_loss��&?e2�        )��P	�~e퉙�A�
*

	conv_loss��&? �bZ        )��P	��e퉙�A�
*

	conv_loss×&?ѺQF        )��P	:�e퉙�A�
*

	conv_loss��&?� 4�        )��P	t"f퉙�A�
*

	conv_lossO�&?a�7        )��P	�Sf퉙�A�
*

	conv_loss��&?�&{        )��P	օf퉙�A�
*

	conv_loss��&?{M�^        )��P	f�f퉙�A�
*

	conv_loss�&?�-�        )��P	7�f퉙�A�
*

	conv_lossГ&?�Bʂ        )��P	�!g퉙�A�
*

	conv_loss"�&?�7h�        )��P	Rg퉙�A�
*

	conv_loss��&?��Q�        )��P	*�g퉙�A�
*

	conv_loss#�&?)i��        )��P	e�g퉙�A�
*

	conv_loss%�&?�r�~        )��P	X�g퉙�A�
*

	conv_loss�&?_$�        )��P	�*h퉙�A�
*

	conv_loss�x&?A9�Q        )��P	oah퉙�A�
*

	conv_lossy&?�qG?        )��P	Ӓh퉙�A�
*

	conv_loss�y&?��5�        )��P	+�h퉙�A�
*

	conv_loss�}&?����        )��P	�h퉙�A�
*

	conv_lossH&?eu!I        )��P	�2i퉙�A�
*

	conv_loss!�&?���        )��P	~ei퉙�A�
*

	conv_loss�t&?�-I        )��P	��i퉙�A�
*

	conv_loss%q&?�T�M        )��P	��i퉙�A�
*

	conv_loss�}&?��/�        )��P	��i퉙�A�
*

	conv_losseh&?̔Z�        )��P	@%j퉙�A�
*

	conv_loss�u&?7$3�        )��P	�Wj퉙�A�
*

	conv_lossi&?��l�        )��P	�j퉙�A�
*

	conv_loss�o&?��$�        )��P	F�j퉙�A�
*

	conv_loss�d&?��!        )��P	0�j퉙�A�
*

	conv_loss�[&?	{Z        )��P	/*k퉙�A�
*

	conv_loss~S&?�U�        )��P	�Zk퉙�A�
*

	conv_loss�\&?5v�        )��P	ފk퉙�A�
*

	conv_loss�V&?��i        )��P		�k퉙�A�
*

	conv_loss�U&?�M�        )��P	h�k퉙�A�
*

	conv_loss_&?W�	        )��P	�l퉙�A�
*

	conv_loss�a&?s���        )��P	�Nl퉙�A�
*

	conv_loss�M&?)���        )��P	d}l퉙�A�
*

	conv_lossX&?y��g        )��P	�l퉙�A�
*

	conv_loss�I&?�%,�        )��P	��l퉙�A�
*

	conv_lossuR&?�cK�        )��P	m퉙�A�
*

	conv_losssM&?k�lf        )��P	|@m퉙�A�
*

	conv_loss�^&?s[I�        )��P	Drm퉙�A�
*

	conv_loss�E&?V��        )��P	�m퉙�A�
*

	conv_lossIG&?�w�P        )��P	��m퉙�A�
*

	conv_losslL&?����        )��P	�n퉙�A�
*

	conv_lossD&?���        )��P	�9n퉙�A�
*

	conv_loss�?&?X)`�        )��P	mln퉙�A�
*

	conv_loss"I&?��G�        )��P	l�n퉙�A�
*

	conv_loss�D&?���B        )��P	�n퉙�A�
*

	conv_loss�:&?p���        )��P	�o퉙�A�
*

	conv_loss�5&?���,        )��P	�1o퉙�A�
*

	conv_loss�3&?�C�        )��P	�ao퉙�A�
*

	conv_loss>&?��%4        )��P	>�o퉙�A�
*

	conv_lossc5&?�VM�        )��P	'�o퉙�A�
*

	conv_loss�0&?�        )��P	�Yq퉙�A�
*

	conv_loss(2&?0s�"        )��P	ԉq퉙�A�
*

	conv_loss72&?ӣ�        )��P		�q퉙�A�
*

	conv_loss^1&?y�c�        )��P	Q�q퉙�A�
*

	conv_loss�7&?Cts        )��P	�r퉙�A�
*

	conv_loss"*&?ԧ��        )��P	MYr퉙�A�
*

	conv_loss�(&?h���        )��P	T�r퉙�A�
*

	conv_loss[ &?H��9        )��P	ĸr퉙�A�*

	conv_loss'&?��O        )��P	��r퉙�A�*

	conv_loss:$&?��        )��P	�(s퉙�A�*

	conv_loss�&?MN��        )��P	�fs퉙�A�*

	conv_loss�)&?1AF�        )��P	��s퉙�A�*

	conv_loss�&?B�H�        )��P	��s퉙�A�*

	conv_loss�"&?q���        )��P	��s퉙�A�*

	conv_loss�&?(HN�        )��P	-t퉙�A�*

	conv_loss�&?�h\        )��P	]t퉙�A�*

	conv_loss&?�6}�        )��P	��t퉙�A�*

	conv_loss�!&?b�(o        )��P	f�t퉙�A�*

	conv_loss�&?���        )��P	��t퉙�A�*

	conv_loss &?���%        )��P	q/u퉙�A�*

	conv_loss�&?��n        )��P	l^u퉙�A�*

	conv_loss�
&?��        )��P	��u퉙�A�*

	conv_loss=&?'���        )��P	߾u퉙�A�*

	conv_loss|&?��_9        )��P	z�u퉙�A�*

	conv_loss��%?���        )��P	v퉙�A�*

	conv_loss>&?	ݶ�        )��P	yMv퉙�A�*

	conv_lossN�%?6��Y        )��P	�v퉙�A�*

	conv_lossU�%?.��        )��P	��v퉙�A�*

	conv_loss��%?��R        )��P	��v퉙�A�*

	conv_lossX�%?,��S        )��P	_w퉙�A�*

	conv_loss��%?j�mL        )��P	�Iw퉙�A�*

	conv_loss��%?%��        )��P	�yw퉙�A�*

	conv_loss��%?pU9�        )��P	бw퉙�A�*

	conv_loss��%?<:�        )��P	��w퉙�A�*

	conv_loss��%?h2�        )��P	Qx퉙�A�*

	conv_loss[�%?�Ozd        )��P	�@x퉙�A�*

	conv_loss��%?�jv�        )��P	$px퉙�A�*

	conv_loss!�%?�Z��        )��P	��x퉙�A�*

	conv_loss�%?.ll        )��P	��x퉙�A�*

	conv_loss��%?\        )��P	<y퉙�A�*

	conv_loss��%?<�K        )��P	�1y퉙�A�*

	conv_lossK�%?�a3        )��P	e`y퉙�A�*

	conv_loss�%?Q�5        )��P	��y퉙�A�*

	conv_loss8�%?�f~m        )��P	��y퉙�A�*

	conv_lossz�%?��        )��P	� z퉙�A�*

	conv_loss��%?u*��        )��P	1z퉙�A�*

	conv_loss��%?b�9�        )��P	�`z퉙�A�*

	conv_loss��%?Y.        )��P	?�z퉙�A�*

	conv_loss�%?�|F        )��P	��z퉙�A�*

	conv_loss��%?���         )��P	��z퉙�A�*

	conv_loss�%?UT�|        )��P	�1{퉙�A�*

	conv_loss��%?+��        )��P	�`{퉙�A�*

	conv_loss��%?�/{        )��P	�{퉙�A�*

	conv_loss��%?@��        )��P	��{퉙�A�*

	conv_loss��%?n(        )��P	��{퉙�A�*

	conv_lossq�%?��@        )��P	n(|퉙�A�*

	conv_lossA�%?w���        )��P	�_|퉙�A�*

	conv_loss��%?F!        )��P	x�|퉙�A�*

	conv_loss��%?���        )��P	��|퉙�A�*

	conv_loss�%?
E        )��P	w}퉙�A�*

	conv_losss�%?Z[�H        )��P	C>}퉙�A�*

	conv_lossU�%?�]{"        )��P	�r}퉙�A�*

	conv_loss��%?'���        )��P	��}퉙�A�*

	conv_lossl�%?v(��        )��P	#�}퉙�A�*

	conv_loss��%?f��        )��P	~~퉙�A�*

	conv_loss��%?�-�
        )��P	f@~퉙�A�*

	conv_loss��%?��        )��P	tp~퉙�A�*

	conv_loss��%?���l        )��P	�~퉙�A�*

	conv_loss~�%?�e7        )��P	R�~퉙�A�*

	conv_loss��%?���        )��P	�퉙�A�*

	conv_loss�%?��%�        )��P	>퉙�A�*

	conv_loss
�%?%k1�        )��P	_n퉙�A�*

	conv_lossq�%?�!�         )��P	�퉙�A�*

	conv_lossߡ%?�͞�        )��P	��퉙�A�*

	conv_lossw�%?���@        )��P	-�퉙�A�*

	conv_loss��%?�@��        )��P	�,�퉙�A�*

	conv_loss�%?�Df�        )��P	�[�퉙�A�*

	conv_lossܛ%?4Ǆ        )��P	&��퉙�A�*

	conv_loss=�%?e��        )��P	��퉙�A�*

	conv_loss��%?Z�A        )��P	�퉙�A�*

	conv_loss%?��)        )��P	C�퉙�A�*

	conv_loss<�%?�B~d        )��P	L�퉙�A�*

	conv_loss��%?>I��        )��P	�~�퉙�A�*

	conv_loss!�%?ka 7        )��P	9��퉙�A�*

	conv_loss�%?����        )��P	�ށ퉙�A�*

	conv_loss��%?�C7�        )��P	��퉙�A�*

	conv_loss��%?�E�        )��P	�>�퉙�A�*

	conv_loss��%?��̶        )��P	�l�퉙�A�*

	conv_loss�w%?�f�]        )��P	��퉙�A�*

	conv_loss΂%?mOa#        )��P	j͂퉙�A�*

	conv_loss�z%?���%        )��P	���퉙�A�*

	conv_loss�w%?�H�        )��P	1�퉙�A�*

	conv_loss�y%?n�P�        )��P	�c�퉙�A�*

	conv_lossKz%?1�م        )��P	���퉙�A�*

	conv_losswm%?�B�        )��P	fÃ퉙�A�*

	conv_loss�x%?}�uq        )��P	#�퉙�A�*

	conv_loss�u%?��Y        )��P	�#�퉙�A�*

	conv_lossp%?�B��        )��P	S�퉙�A�*

	conv_loss�h%?�9        )��P	��퉙�A�*

	conv_loss�u%?���        )��P	��퉙�A�*

	conv_loss{v%?�7��        )��P	�*�퉙�A�*

	conv_loss*c%?�oR{        )��P	;Z�퉙�A�*

	conv_loss�b%?��i        )��P	���퉙�A�*

	conv_loss6`%?4���        )��P	ƶ�퉙�A�*

	conv_loss�d%?�`        )��P	��퉙�A�*

	conv_loss�_%?{��        )��P	��퉙�A�*

	conv_loss�c%?���        )��P	�U�퉙�A�*

	conv_lossj%?�u�k        )��P	��퉙�A�*

	conv_loss�b%?C�3J        )��P	�퉙�A�*

	conv_loss�f%?Z�?        )��P	���퉙�A�*

	conv_loss=P%?ƞf�        )��P	)�퉙�A�*

	conv_loss#^%?���	        )��P	�j�퉙�A�*

	conv_loss+`%?��9�        )��P	u��퉙�A�*

	conv_loss{b%? �        )��P	bՋ퉙�A�*

	conv_lossfW%?0�xf        )��P	l�퉙�A�*

	conv_loss�X%?�r��        )��P	�6�퉙�A�*

	conv_loss�A%?���        )��P	@h�퉙�A�*

	conv_lossQ%?�O        )��P	��퉙�A�*

	conv_loss"M%?-wIe        )��P	ǌ퉙�A�*

	conv_loss5?%?@��@        )��P	a��퉙�A�*

	conv_lossJF%?���        )��P	%%�퉙�A�*

	conv_loss-F%?Eqc�        )��P	[V�퉙�A�*

	conv_loss�?%?i>GU        )��P	���퉙�A�*

	conv_loss�-%?1���        )��P	뵍퉙�A�*

	conv_loss:%?M��z        )��P	��퉙�A�*

	conv_loss0%?��        )��P	�퉙�A�*

	conv_lossX3%?"��J        )��P	F�퉙�A�*

	conv_loss�5%?��<        )��P	$v�퉙�A�*

	conv_loss�:%?\�^o        )��P	��퉙�A�*

	conv_loss�-%?F��        )��P	�Ԏ퉙�A�*

	conv_loss�7%?��r        )��P	$�퉙�A�*

	conv_lossW7%?�ܿg        )��P	�2�퉙�A�*

	conv_lossm,%?C��        )��P	a�퉙�A�*

	conv_lossS/%?̮5U        )��P	ۓ�퉙�A�*

	conv_loss�!%?A{pU        )��P	�Ï퉙�A�*

	conv_loss�%%?��Ȩ        )��P	��퉙�A�*

	conv_lossv%%?z>�        )��P	(&�퉙�A�*

	conv_loss�%?&��3        )��P	U�퉙�A�*

	conv_loss%?�l=        )��P	���퉙�A�*

	conv_loss�%?�$�        )��P	봐퉙�A�*

	conv_loss %?s]Z�        )��P	��퉙�A�*

	conv_lossX%%?�e'        )��P	��퉙�A�*

	conv_loss %?��7        )��P	�G�퉙�A�*

	conv_loss2%?��S        )��P	�w�퉙�A�*

	conv_lossh%?�M�        )��P	���퉙�A�*

	conv_loss�%?�6��        )��P	�ۑ퉙�A�*

	conv_loss�%?�1�        )��P	��퉙�A�*

	conv_loss�%?��&        )��P	�@�퉙�A�*

	conv_loss�%?KG�        )��P	�q�퉙�A�*

	conv_loss8%?f��^        )��P	���퉙�A�*

	conv_loss/%?����        )��P	��퉙�A�*

	conv_loss��$?��        )��P	U�퉙�A�*

	conv_loss��$?[��        )��P	
N�퉙�A�*

	conv_loss�%??:�}        )��P	��퉙�A�*

	conv_loss	%?���        )��P	���퉙�A�*

	conv_loss�
%?�̔O        )��P	p��퉙�A�*

	conv_loss� %? y>�        )��P	�)�퉙�A�*

	conv_loss�$?C+0�        )��P	a�퉙�A�*

	conv_loss�$?#�g�        )��P	���퉙�A�*

	conv_loss��$?�;ڴ        )��P	ڔ퉙�A�*

	conv_lossd�$?�J�        )��P	<�퉙�A�*

	conv_loss�$?ʞ��        )��P	�B�퉙�A�*

	conv_lossm�$?�        )��P	{~�퉙�A�*

	conv_lossz�$?	:B        )��P	Xɕ퉙�A�*

	conv_lossQ�$?H$        )��P	��퉙�A�*

	conv_loss*�$?���e        )��P	�3�퉙�A�*

	conv_loss�$?ny�O        )��P	�h�퉙�A�*

	conv_loss�$?�@��        )��P	���퉙�A�*

	conv_loss��$?�7H�        )��P	zҖ퉙�A�*

	conv_loss�$?!� d        )��P	N�퉙�A�*

	conv_loss?�$?
ɀ        )��P	=�퉙�A�*

	conv_lossL�$?�^��        )��P	�p�퉙�A�*

	conv_lossr�$?�Io�        )��P	]��퉙�A�*

	conv_loss��$?�3jr        )��P	�ܗ퉙�A�*

	conv_loss��$?Z�6        )��P	&�퉙�A�*

	conv_lossx�$?)�pr        )��P	MF�퉙�A�*

	conv_loss�$?�1K        )��P	\{�퉙�A�*

	conv_loss��$?��        )��P	���퉙�A�*

	conv_loss��$?]�d        )��P	�퉙�A�*

	conv_lossm�$?	��~        )��P	��퉙�A�*

	conv_loss�$?\�>        )��P	�P�퉙�A�*

	conv_lossT�$?��xA        )��P	���퉙�A�*

	conv_loss[�$?�s|�        )��P	���퉙�A�*

	conv_loss'�$?7�i$        )��P	��퉙�A�*

	conv_loss�$?�ۀW        )��P	%�퉙�A�*

	conv_loss��$?6���        )��P	�Z�퉙�A�*

	conv_loss�$?=̼6        )��P	Đ�퉙�A�*

	conv_loss�$?�&�        )��P	�Ś퉙�A�*

	conv_loss��$?�U��        )��P	���퉙�A�*

	conv_loss��$?R\�H        )��P	*.�퉙�A�*

	conv_loss�$?*8 �        )��P	�c�퉙�A�*

	conv_loss�$?����        )��P	��퉙�A�*

	conv_lossQ�$?=-~        )��P	�͛퉙�A�*

	conv_loss�$?�DJ�        )��P	��퉙�A�*

	conv_loss"�$?2fH        )��P	�9�퉙�A�*

	conv_lossګ$?�9,6        )��P	�n�퉙�A�*

	conv_lossĩ$?_�Z�        )��P	v��퉙�A�*

	conv_loss�$?RYq        )��P	�֜퉙�A�*

	conv_loss��$?���        )��P	�퉙�A�*

	conv_loss�$?W5��        )��P	�?�퉙�A�*

	conv_loss��$?-�uN        )��P	�퉙�A�*

	conv_loss��$?g��        )��P	��퉙�A�*

	conv_loss0�$?��q]        )��P	�U�퉙�A�*

	conv_loss�$?p�)�        )��P	���퉙�A�*

	conv_lossX�$?!�B�        )��P	��퉙�A�*

	conv_lossr�$?�#g�        )��P	��퉙�A�*

	conv_loss��$?���X        )��P	@S�퉙�A�*

	conv_lossS�$?l��Z        )��P	
��퉙�A�*

	conv_loss��$?�V��        )��P	�Ƞ퉙�A�*

	conv_lossw�$?P�S�        )��P	� �퉙�A�*

	conv_loss{�$??$��        )��P	�4�퉙�A�*

	conv_lossP�$?���.        )��P	ej�퉙�A�*

	conv_loss��$?�h�!        )��P	J��퉙�A�*

	conv_loss��$? "�        )��P	�ӡ퉙�A�*

	conv_loss��$?�L�`        )��P	s�퉙�A�*

	conv_lossĉ$?���        )��P	�M�퉙�A�*

	conv_loss��$?��        )��P	�퉙�A�*

	conv_loss�$?�8X        )��P	{��퉙�A�*

	conv_loss��$?9�y        )��P	��퉙�A�*

	conv_lossֆ$?8`�        )��P	"$�퉙�A�*

	conv_loss�t$?�Ё�        )��P	EW�퉙�A�*

	conv_lossn$?��\}        )��P	5��퉙�A�*

	conv_loss�}$?XD�C        )��P	���퉙�A�*

	conv_loss~z$?�58        )��P	��퉙�A�*

	conv_lossZw$??��        )��P	C&�퉙�A�*

	conv_loss�y$?"Ϥ        )��P	�Z�퉙�A�*

	conv_loss�z$?V�4        )��P	=��퉙�A�*

	conv_loss�|$?�hw        )��P	/��퉙�A�*

	conv_loss1W$?wLS        )��P	J��퉙�A�*

	conv_loss�n$?�ƒt        )��P	�&�퉙�A�*

	conv_loss�p$?�?%�        )��P	�Z�퉙�A�*

	conv_loss�i$?�G��        )��P	s��퉙�A�*

	conv_loss�^$?z���        )��P	å퉙�A�*

	conv_loss�c$?�L�8        )��P	V��퉙�A�*

	conv_loss�d$?
M        )��P	�+�퉙�A�*

	conv_loss�^$?���        )��P	y_�퉙�A�*

	conv_losso[$?.��        )��P	?��퉙�A�*

	conv_loss�q$?]�LL        )��P	�ɦ퉙�A�*

	conv_loss�R$?	�Q        )��P	���퉙�A�*

	conv_lossNY$?��R        )��P	92�퉙�A�*

	conv_loss�Q$?�\��        )��P	Lf�퉙�A�*

	conv_loss�\$?0<F        )��P	$��퉙�A�*

	conv_lossBN$?�ė$        )��P	Nϧ퉙�A�*

	conv_loss
X$?��hc        )��P	/�퉙�A�*

	conv_loss]T$?��5�        )��P	8�퉙�A�*

	conv_lossrD$?YF        )��P	nl�퉙�A�*

	conv_loss^E$?�U"X        )��P	۠�퉙�A�*

	conv_lossNB$?ED��        )��P	�֨퉙�A�*

	conv_loss$B$?��t        )��P	��퉙�A�*

	conv_loss�H$?/��        )��P	jC�퉙�A�*

	conv_loss`B$?�,        )��P	"��퉙�A�*

	conv_loss�A$?i&�        )��P	+é퉙�A�*

	conv_lossdA$?H˱Q        )��P	��퉙�A�*

	conv_loss�3$?�|        )��P	B0�퉙�A�*

	conv_loss�5$?+�        )��P	�i�퉙�A�*

	conv_loss�*$?ꭣ�        )��P	���퉙�A�*

	conv_loss�/$?��I        )��P	�Ҫ퉙�A�*

	conv_loss�/$?�W        )��P	4�퉙�A�*

	conv_loss�'$?p��        )��P	�;�퉙�A�*

	conv_loss4.$?ą�K        )��P	ct�퉙�A�*

	conv_loss�6$?�[�        )��P	���퉙�A�*

	conv_loss�&$?WU��        )��P	�߫퉙�A�*

	conv_loss*$?�@d�        )��P	<�퉙�A�*

	conv_loss/ $?5�	�        )��P	�`�퉙�A�*

	conv_loss�/$?��Q        )��P	蘬퉙�A�*

	conv_loss%$? ͉�        )��P	ά퉙�A�*

	conv_loss~$?rI`�        )��P	��퉙�A�*

	conv_loss�$?־D�        )��P	@7�퉙�A�*

	conv_loss�$?^W<\        )��P	�k�퉙�A�*

	conv_loss�$?�<y        )��P	3��퉙�A�*

	conv_lossp$?t%�        )��P	�խ퉙�A�*

	conv_loss�$?�4t�        )��P	�	�퉙�A�*

	conv_loss�$?P<�"        )��P	t?�퉙�A�*

	conv_loss
$?�         )��P	:v�퉙�A�*

	conv_loss�$?*��        )��P	��퉙�A�*

	conv_lossw$?�J        )��P	��퉙�A�*

	conv_loss�$?'T�        )��P	��퉙�A�*

	conv_loss'	$?����        )��P	wJ�퉙�A�*

	conv_loss�$?�$%�        )��P	��퉙�A�*

	conv_loss�$?��U        )��P	p��퉙�A�*

	conv_loss$?K���        )��P	��퉙�A�*

	conv_loss��#?����        )��P	b�퉙�A�*

	conv_loss��#?S"�        )��P	�P�퉙�A�*

	conv_loss��#?M'F        )��P	퉙�A�*

	conv_loss9$?����        )��P	���퉙�A�*

	conv_lossp�#?�B��        )��P	��퉙�A�*

	conv_loss�#?�'�.        )��P		&�퉙�A�*

	conv_loss��#?�&bW        )��P	[�퉙�A�*

	conv_loss��#?�ԙ7        )��P	���퉙�A�*

	conv_lossI�#?�tU�        )��P	cı퉙�A�*

	conv_lossM�#?e�M        )��P	���퉙�A�*

	conv_loss[�#?����        )��P	S+�퉙�A�*

	conv_lossq�#?� ,�        )��P	�_�퉙�A�*

	conv_lossp�#?�~��        )��P	
��퉙�A�*

	conv_loss�#?w�I        )��P	�˲퉙�A�*

	conv_loss��#?�Yn        )��P	� �퉙�A�*

	conv_loss��#?N,v�        )��P	�5�퉙�A�*

	conv_lossI�#?���        )��P	�j�퉙�A�*

	conv_lossn�#?�e�(        )��P	���퉙�A�*

	conv_loss��#?i��        )��P	�г퉙�A�*

	conv_loss�#?û�^        )��P	A�퉙�A�*

	conv_lossg�#?��        )��P	�L�퉙�A�*

	conv_loss�#?9|�        )��P	j��퉙�A�*

	conv_loss��#?	��        )��P	Ķ�퉙�A�*

	conv_lossI�#?��e�        )��P	y�퉙�A�*

	conv_loss��#?����        )��P	_"�퉙�A�*

	conv_loss��#?j �_        )��P	8X�퉙�A�*

	conv_loss��#?Z|*,        )��P	昵퉙�A�*

	conv_loss��#?Z��[        )��P	�͵퉙�A�*

	conv_loss׽#?�*�        )��P	]�퉙�A�*

	conv_loss��#?���        )��P	~?�퉙�A�*

	conv_loss��#?��UI        )��P	�z�퉙�A�*

	conv_loss;�#?=�>Z        )��P	䵶퉙�A�*

	conv_loss3�#?q	/�        )��P	��퉙�A�*

	conv_loss�#?0~        )��P	�!�퉙�A�*

	conv_lossm�#?L��        )��P	�U�퉙�A�*

	conv_loss��#?n믡        )��P	ڋ�퉙�A�*

	conv_lossó#?��        )��P	���퉙�A�*

	conv_loss�#?X�ȷ        )��P	���퉙�A�*

	conv_loss��#?Y�        )��P	�.�퉙�A�*

	conv_loss^�#?tǝ�        )��P	�b�퉙�A�*

	conv_loss@�#?���        )��P	k��퉙�A�*

	conv_loss2�#?Ə��        )��P	�ʸ퉙�A�*

	conv_lossf�#?р�        )��P	��퉙�A�*

	conv_loss"�#?�ea        )��P	�2�퉙�A�*

	conv_loss֨#?kdro        )��P	Lf�퉙�A�*

	conv_loss �#?�(G        )��P	D��퉙�A�*

	conv_loss4�#?=��        )��P	�ѹ퉙�A�*

	conv_loss�#?~�        )��P	}�퉙�A�*

	conv_loss/�#?���        )��P	
:�퉙�A�*

	conv_loss��#?�>8        )��P	�n�퉙�A�*

	conv_loss��#?b~�}        )��P	���퉙�A�*

	conv_loss�#?b��x        )��P	׺퉙�A�*

	conv_lossޘ#?���        )��P	,�퉙�A�*

	conv_loss�#?P�L        )��P	�A�퉙�A�*

	conv_loss��#?�G��        )��P	/w�퉙�A�*

	conv_loss�#?����        )��P	���퉙�A�*

	conv_loss4�#?cŋ        )��P	�޻퉙�A�*

	conv_loss��#?6n)        )��P	��퉙�A�*

	conv_loss"�#?E&o        )��P	UI�퉙�A�*

	conv_loss��#?�h        )��P	�~�퉙�A�*

	conv_loss%�#?��C`        )��P	���퉙�A�*

	conv_loss�x#?5��        )��P	��퉙�A�*

	conv_loss��#? Q��        )��P	"�퉙�A�*

	conv_lossу#?u���        )��P	~Q�퉙�A�*

	conv_loss�}#?��,        )��P	T��퉙�A�*

	conv_loss
�#?���        )��P	ž�퉙�A�*

	conv_loss{�#?��	Y        )��P	��퉙�A�*

	conv_loss}#?딗        )��P	�&�퉙�A�*

	conv_loss2o#?���        )��P	[�퉙�A�*

	conv_loss�e#?��*�        )��P	���퉙�A�*

	conv_lossx#?�3eB        )��P	׾퉙�A�*

	conv_loss$t#?�PA        )��P	��퉙�A�*

	conv_loss~#?G�]K        )��P	�C�퉙�A�*

	conv_loss�a#?	Xz�        )��P	Fz�퉙�A�*

	conv_loss�`#?��mJ        )��P	᯿퉙�A�*

	conv_loss�o#?��        )��P	=�퉙�A�*

	conv_loss�T#?��        )��P	K�퉙�A�*

	conv_loss�t#?Fh_�        )��P	�^�퉙�A�*

	conv_loss�l#?̣�        )��P	d��퉙�A�*

	conv_loss�\#?N�m        )��P	3��퉙�A�*

	conv_lossJ_#?���        )��P	3�퉙�A�*

	conv_lossf#?�s�)        )��P	)C�퉙�A�*

	conv_loss�P#?�l|        )��P	�w�퉙�A�*

	conv_loss2V#?�%�        )��P	��퉙�A�*

	conv_lossFV#?:%�        )��P	+��퉙�A�*

	conv_loss�[#?���6        )��P	�퉙�A�*

	conv_loss�S#?ڳ�M        )��P	,J�퉙�A�*

	conv_lossZU#?!�#l        )��P	��퉙�A�*

	conv_loss�B#?D5^        )��P	Q��퉙�A�*

	conv_loss�N#?��s�        )��P	��퉙�A�*

	conv_lossI#?�        )��P	0�퉙�A�*

	conv_lossTK#?zm��        )��P	dd�퉙�A�*

	conv_lossxP#?����        )��P	���퉙�A�*

	conv_loss�<#?���        )��P	���퉙�A�*

	conv_loss�<#?Dǹ�        )��P	��퉙�A�*

	conv_lossM9#?��        )��P	�=�퉙�A�*

	conv_loss|3#?󝦬        )��P	Rq�퉙�A�*

	conv_lossk7#?���        )��P	ۤ�퉙�A�*

	conv_loss�6#?�P��        )��P	��퉙�A�*

	conv_lossu(#?۸        )��P	��퉙�A�*

	conv_loss�8#?�b�        )��P	K�퉙�A�*

	conv_loss�/#?>��        )��P	I��퉙�A�*

	conv_loss�2#?���        )��P	}��퉙�A�*

	conv_lossW.#?��#        )��P	&��퉙�A�*

	conv_lossc0#?/�ޒ        )��P	|#�퉙�A�*

	conv_loss�$#?���#        )��P	�X�퉙�A�*

	conv_loss(#?� \^        )��P	���퉙�A�*

	conv_lossc&#?"9        )��P	���퉙�A�*

	conv_lossE$#?�D�B        )��P	���퉙�A�*

	conv_loss1'#?jJW        )��P	a,�퉙�A�*

	conv_loss� #?��        )��P	�`�퉙�A�*

	conv_loss&#?,��#        )��P	_��퉙�A�*

	conv_loss#?]�        )��P	���퉙�A�*

	conv_loss�$#?2C��        )��P	�퉙�A�*

	conv_loss�!#?C�Z�        )��P	�5�퉙�A�*

	conv_loss#?)c��        )��P	Rj�퉙�A�*

	conv_loss%#?T@s<        )��P	M��퉙�A�*

	conv_loss�#?	#
a        )��P	��퉙�A�*

	conv_loss�#?,�X�        )��P	��퉙�A�*

	conv_loss/#?���:        )��P	���퉙�A�*

	conv_lossA#?�Ex�        )��P	���퉙�A�*

	conv_loss�#?���N        )��P	e�퉙�A�*

	conv_lossi#?׊~=        )��P	cE�퉙�A�*

	conv_loss�#?h�;        )��P	1{�퉙�A�*

	conv_loss��"?+X
�        )��P	~��퉙�A�*

	conv_losso�"?��        )��P	���퉙�A�*

	conv_loss*#?^��=        )��P	C&�퉙�A�*

	conv_loss�#?E�e�        )��P	�Y�퉙�A�*

	conv_loss��"?��8        )��P	 ��퉙�A�*

	conv_lossy #?���        )��P	T��퉙�A�*

	conv_lossa�"?r��        )��P	���퉙�A�*

	conv_loss��"?$Eu�        )��P	@?�퉙�A�*

	conv_loss�"?P�        )��P	�v�퉙�A�*

	conv_loss��"?H�        )��P	���퉙�A�*

	conv_lossQ�"?�נ        )��P	��퉙�A�*

	conv_loss-�"?G�h�        )��P	�퉙�A�*

	conv_lossP�"?�=�        )��P	oG�퉙�A�*

	conv_losss�"?�¹+        )��P	Z|�퉙�A�*

	conv_lossH�"?MbW        )��P	8��퉙�A�*

	conv_lossu�"?��N        )��P	���퉙�A�*

	conv_loss��"?��;�        )��P	��퉙�A�*

	conv_loss��"?k4N        )��P	�M�퉙�A�*

	conv_loss��"?���F        )��P	ۄ�퉙�A�*

	conv_lossX�"?���        )��P	E��퉙�A�*

	conv_loss�"?<�!        )��P	���퉙�A�*

	conv_loss��"?/[U�        )��P	P!�퉙�A�*

	conv_loss?�"?�ܘ�        )��P	AV�퉙�A�*

	conv_losse�"?r��        )��P	���퉙�A�*

	conv_loss��"?T���        )��P	���퉙�A�*

	conv_loss��"?�k�        )��P	���퉙�A�*

	conv_lossd�"?鋄�        )��P	u(�퉙�A�*

	conv_loss��"?cx�        )��P	7\�퉙�A�*

	conv_loss��"?�\�        )��P	���퉙�A�*

	conv_loss��"?��        )��P	��퉙�A�*

	conv_loss��"?�u�4        )��P	=��퉙�A�*

	conv_losso�"?\@YK        )��P	�,�퉙�A�*

	conv_loss�"?�Y&=        )��P	�a�퉙�A�*

	conv_loss1�"?         )��P	���퉙�A�*

	conv_loss\�"?����        )��P	���퉙�A�*

	conv_lossB�"?*/_        )��P	p��퉙�A�*

	conv_losso�"?��k�        )��P	2�퉙�A�*

	conv_loss(�"?�CA�        )��P	sf�퉙�A�*

	conv_lossQ�"?uGU�        )��P	0��퉙�A�*

	conv_lossC�"?�_�        )��P	}��퉙�A�*

	conv_loss'�"?.Gw;        )��P	��퉙�A�*

	conv_loss��"?��5c        )��P	�7�퉙�A�*

	conv_loss4�"?d�؊        )��P	�k�퉙�A�*

	conv_loss�"??}[        )��P	8��퉙�A�*

	conv_loss۱"?m���        )��P	��퉙�A�*

	conv_lossɢ"?AD        )��P	M�퉙�A�*

	conv_loss��"?��03        )��P	�P�퉙�A�*

	conv_loss�"?���        )��P	څ�퉙�A�*

	conv_lossԬ"?�Qt�        )��P	��퉙�A�*

	conv_lossT�"?
Ī�        )��P	#��퉙�A�*

	conv_loss�"?#7��        )��P	m$�퉙�A�*

	conv_loss��"?o��        )��P	�c�퉙�A�*

	conv_loss��"?�9n        )��P	���퉙�A�*

	conv_loss��"?Wxj�        )��P	I��퉙�A�*

	conv_lossL�"?hv�        )��P	C�퉙�A�*

	conv_lossޏ"?�=9�        )��P	�^�퉙�A�*

	conv_loss%�"?5�         )��P	Ɠ�퉙�A�*

	conv_loss�"?cX��        )��P	m��퉙�A�*

	conv_lossC�"?�g8�        )��P	���퉙�A�*

	conv_loss\�"?	g��        )��P	�0�퉙�A�*

	conv_lossk�"?���
        )��P	Zf�퉙�A�*

	conv_loss��"?6ề        )��P	r��퉙�A�*

	conv_loss�}"?ީL�        )��P	���퉙�A�*

	conv_loss"�"?��=�        )��P	��퉙�A�*

	conv_lossۑ"?x1��        )��P	�:�퉙�A�*

	conv_loss�u"?��\1        )��P	�y�퉙�A�*

	conv_loss�"?V�4C        )��P	���퉙�A�*

	conv_loss�"?oXh�        )��P	��퉙�A�*

	conv_loss��"?E�4�        )��P	T �퉙�A�*

	conv_lossVy"?��X        )��P	zS�퉙�A�*

	conv_loss�b"?�2�g        )��P	��퉙�A�*

	conv_losst}"?5��        )��P	{��퉙�A�*

	conv_lossXo"?���        )��P	���퉙�A�*

	conv_lossEo"?���        )��P	&$�퉙�A�*

	conv_loss�h"?(^�        )��P	�Y�퉙�A�*

	conv_loss�u"?Px�V        )��P	o��퉙�A�*

	conv_loss^m"?�p�        )��P	j��퉙�A�*

	conv_loss�b"?S�        )��P	���퉙�A�*

	conv_lossE_"?�/        )��P	a.�퉙�A�*

	conv_loss�e"?B�j8        )��P	cc�퉙�A�*

	conv_loss�`"?�o�        )��P	V��퉙�A�*

	conv_loss�p"?��c        )��P	���퉙�A�*

	conv_lossuL"?"r`5        )��P	���퉙�A�*

	conv_loss�V"?�cU�        )��P	d5�퉙�A�*

	conv_lossjU"?�߾        )��P	�j�퉙�A�*

	conv_loss�N"?X�O        )��P	^��퉙�A�*

	conv_loss�X"?]�W        )��P	��퉙�A�*

	conv_loss�X"?����        )��P	��퉙�A�*

	conv_lossnU"?�$��        )��P	:=�퉙�A�*

	conv_lossWW"?��~=        )��P	�r�퉙�A�*

	conv_loss�L"?�}	        )��P	��퉙�A�*

	conv_loss�D"?��i        )��P	���퉙�A�*

	conv_lossG"?����        )��P	��퉙�A�*

	conv_lossiG"?��R�        )��P	�H�퉙�A�*

	conv_loss�@"?��ʲ        )��P	�|�퉙�A�*

	conv_lossl>"?��-o        )��P	���퉙�A�*

	conv_loss�7"?�@��        )��P	���퉙�A�*

	conv_loss�@"?KQ$        )��P	�1�퉙�A�*

	conv_loss5"?�ͧ�        )��P	[k�퉙�A�*

	conv_loss�3"?kWt         )��P	դ�퉙�A�*

	conv_loss@="?*�d        )��P	���퉙�A�*

	conv_loss�/"?��Ve        )��P	��퉙�A�*

	conv_loss4C"?$��        )��P	E\�퉙�A�*

	conv_loss�1"?tr�!        )��P	��퉙�A�*

	conv_loss	("?�r��        )��P	P��퉙�A�*

	conv_loss*"?w�c�        )��P	'�퉙�A�*

	conv_loss�!"?�H�Q        )��P	9�퉙�A�*

	conv_loss�*"?��}        )��P	�n�퉙�A�*

	conv_lossr'"?��q�        )��P	Y��퉙�A�*

	conv_loss�,"?W"L�        )��P	g��퉙�A�*

	conv_loss:"?=K        )��P	��퉙�A�*

	conv_lossB&"?��        )��P	:L�퉙�A�*

	conv_loss�"?d���        )��P	"��퉙�A�*

	conv_lossu,"?z��Y        )��P	=��퉙�A�*

	conv_loss)"?3a        )��P	��퉙�A�*

	conv_loss�"?<�        )��P	�3�퉙�A�*

	conv_loss"?c�b        )��P	g�퉙�A�*

	conv_lossJ"?���        )��P	J��퉙�A�*

	conv_loss�"?ҡG        )��P	6��퉙�A�*

	conv_loss"?�U��        )��P	k�퉙�A�*

	conv_loss"?��        )��P	�9�퉙�A�*

	conv_loss"?;��        )��P	�o�퉙�A�*

	conv_lossP"?��U�        )��P	���퉙�A�*

	conv_loss�"?7,��        )��P	"��퉙�A�*

	conv_loss��!?�6IK        )��P	H�퉙�A�*

	conv_loss�!?��E        )��P	D�퉙�A�*

	conv_loss��!?^�J�        )��P	Sy�퉙�A�*

	conv_loss�	"?�� F        )��P	-��퉙�A�*

	conv_loss�	"?�UaG        )��P	 ��퉙�A�*

	conv_loss��!?�I�M        )��P	��퉙�A�*

	conv_lossz�!?����        )��P	�I�퉙�A�*

	conv_loss��!?N�+�        )��P	�}�퉙�A�*

	conv_lossE�!?��M�        )��P	���퉙�A�*

	conv_loss��!?�a        )��P	$��퉙�A�*

	conv_loss�!?	p[�        )��P	B�퉙�A�*

	conv_loss��!?�_F        )��P	�S�퉙�A�*

	conv_loss��!?��vQ        )��P	̇�퉙�A�*

	conv_loss��!?����        )��P	P��퉙�A�*

	conv_loss��!?�Ρ�        )��P	��퉙�A�*

	conv_lossr�!?-�        )��P	�"�퉙�A�*

	conv_loss)�!?b"7�        )��P	�V�퉙�A�*

	conv_loss�!?I4�-        )��P	؊�퉙�A�*

	conv_loss��!?��@        )��P	q��퉙�A�*

	conv_loss��!?y�i�        )��P	���퉙�A�*

	conv_loss��!?=��F        )��P	O%�퉙�A�*

	conv_lossK�!?q�5*        )��P	Bn�퉙�A�*

	conv_loss��!?�2b�        )��P	/��퉙�A�*

	conv_loss�!?��h�        )��P	���퉙�A�*

	conv_loss%�!?�s�!        )��P	��퉙�A�*

	conv_loss��!?e��        )��P	�F�퉙�A�*

	conv_lossB�!?�@��        )��P	��퉙�A�*

	conv_lossf�!?�R�        )��P	 ��퉙�A�*

	conv_loss��!?K�        )��P	���퉙�A�*

	conv_loss��!?�Ր        )��P	N2�퉙�A�*

	conv_loss*�!?��)
        )��P	Fg�퉙�A�*

	conv_lossĹ!?���        )��P	ɜ�퉙�A�*

	conv_loss�!?��"O        )��P	1��퉙�A�*

	conv_lossP�!?�.�        )��P	��퉙�A�*

	conv_loss±!?U�C�        )��P	j:�퉙�A�*

	conv_loss��!?�/��        )��P	Zm�퉙�A�*

	conv_loss�!?b�qu        )��P	���퉙�A�*

	conv_loss,�!?`�        )��P	��퉙�A�*

	conv_loss�!?����        )��P	]�퉙�A�*

	conv_lossۧ!?��W[        )��P	�P�퉙�A�*

	conv_loss��!?���        )��P	E��퉙�A�*

	conv_loss��!?u2�        )��P	���퉙�A�*

	conv_loss��!?&r�C        )��P	���퉙�A�*

	conv_loss߬!?�^��        )��P	q(�퉙�A�*

	conv_loss��!?C1N�        )��P	s\�퉙�A�*

	conv_lossa�!?��;l        )��P	5��퉙�A�*

	conv_lossr�!?"�_        )��P	���퉙�A�*

	conv_loss��!?H��        )��P	���퉙�A�*

	conv_loss��!?���6        )��P	F.�퉙�A�*

	conv_lossԎ!?��X�        )��P	�b�퉙�A�*

	conv_loss��!?�x�        )��P	���퉙�A�*

	conv_loss�!?_(E        )��P	��퉙�A�*

	conv_loss/�!?o�Å        )��P	��퉙�A�*

	conv_loss��!?���        )��P	v6�퉙�A�*

	conv_loss+�!?�1G	        )��P	�m�퉙�A�*

	conv_loss�~!?���e        )��P	ˠ�퉙�A�*

	conv_loss��!?���        )��P	C��퉙�A�*

	conv_loss2�!?I3�        )��P	!�퉙�A�*

	conv_loss.�!?�\�        )��P	cC�퉙�A�*

	conv_loss8�!?�k�Z        )��P	�x�퉙�A�*

	conv_loss
�!?�g�        )��P	���퉙�A�*

	conv_loss��!?S�        )��P	���퉙�A�*

	conv_loss�}!?F�A�        )��P	�퉙�A�*

	conv_loss?{!?	��        )��P	�K�퉙�A�*

	conv_loss�!?Y��P        )��P	���퉙�A�*

	conv_loss�j!?���        )��P	{��퉙�A�*

	conv_loss0s!?��L�        )��P	���퉙�A�*

	conv_lossՋ!?X�        )��P	��퉙�A�*

	conv_loss<w!?l�-        )��P	O�퉙�A�*

	conv_loss�h!?P��<        )��P	���퉙�A�*

	conv_lossTl!?x��        )��P	��퉙�A�*

	conv_loss�u!?��        )��P	��퉙�A�*

	conv_loss�|!?_�        )��P	�J�퉙�A�*

	conv_lossj|!?����        )��P	�|�퉙�A�*

	conv_loss�j!?�)�@        )��P	���퉙�A�*

	conv_loss{j!?f"�F        )��P	���퉙�A�*

	conv_loss�W!?<��^        )��P	�퉙�A�*

	conv_losskb!?��        )��P	�D�퉙�A�*

	conv_lossWc!?1.
�        )��P	F��퉙�A�*

	conv_loss�T!?i�&�        )��P	��퉙�A�*

	conv_losszZ!?9jZ3        )��P	���퉙�A�*

	conv_loss�h!?��Ƚ        )��P	$�퉙�A�*

	conv_lossWM!?��T�        )��P	�Q�퉙�A�*

	conv_loss�b!?Z��        )��P	���퉙�A�*

	conv_loss�R!?4h�        )��P	���퉙�A�*

	conv_loss�R!?�'[�        )��P	���퉙�A�*

	conv_loss�H!?�:Je        )��P	�퉙�A�*

	conv_loss�J!?�/�        )��P	1G�퉙�A�*

	conv_lossO!?�V�        )��P	nw�퉙�A�*

	conv_loss�M!?j`        )��P	.��퉙�A�*

	conv_loss&E!?4mo�        )��P	T��퉙�A�*

	conv_loss'W!?R,�1        )��P	� �퉙�A�*

	conv_loss�?!?�p        )��P	/1�퉙�A�*

	conv_losso5!?N�o        )��P	�c�퉙�A�*

	conv_loss�D!?nyS        )��P	���퉙�A�*

	conv_lossJ!?�K��        )��P	%��퉙�A�*

	conv_loss�C!?n��        )��P	���퉙�A�*

	conv_lossb3!?���S        )��P	�, �A�*

	conv_loss/>!?�6�        )��P	_ �A�*

	conv_loss�@!?��#�        )��P	� �A�*

	conv_loss�6!?�j��        )��P	�� �A�*

	conv_lossX>!?`��\        )��P	�� �A�*

	conv_lossJ)!?�Cd        )��P	^�A�*

	conv_lossy,!?N>        )��P	�J�A�*

	conv_loss}!?�-�        )��P	�z�A�*

	conv_loss�'!?EJ��        )��P	=��A�*

	conv_lossd!?�s��        )��P	���A�*

	conv_loss!?�Q<r        )��P	x�A�*

	conv_loss�!?�|F�        )��P	c=�A�*

	conv_lossB!?�a�t        )��P		n�A�*

	conv_loss!?H�<r        )��P	_��A�*

	conv_loss�!?q1ݯ        )��P	���A�*

	conv_loss�!?U�        )��P	���A�*

	conv_loss�!?���        )��P	�(�A�*

	conv_loss!?V�        )��P	�X�A�*

	conv_loss�!?*��        )��P	��A�*

	conv_loss7!?�?        )��P	y��A�*

	conv_lossO!?
�{�        )��P	���A�*

	conv_loss�!?{am        )��P	M�A�*

	conv_loss�� ?ɒ�        )��P	�L�A�*

	conv_loss"!?����        )��P	N~�A�*

	conv_loss/� ?�ݰg        )��P	���A�*

	conv_loss!?J��8        )��P	��A�*

	conv_loss�!?=��        )��P	�&�A�*

	conv_lossv� ?*'E        )��P	^�A�*

	conv_lossx� ?�A*        )��P	u��A�*

	conv_loss-� ?%"�v        )��P	���A�*

	conv_lossJ� ?h        )��P	!�A�*

	conv_lossW� ?[�        )��P	�4�A�*

	conv_loss�!?�E��        )��P	o�A�*

	conv_loss�� ?tMV�        )��P	?��A�*

	conv_loss2� ?z�S        )��P	��A�*

	conv_loss"� ?��XB        )��P	]�A�*

	conv_loss� ?~l�        )��P	�K�A�*

	conv_loss�� ?�Lz�        )��P	�~�A�*

	conv_loss]� ?�F�        )��P	 ��A�*

	conv_loss� ?Хx�        )��P	"��A�*

	conv_loss�� ?գ�p        )��P	� �A�*

	conv_lossT� ?��        )��P	U�A�*

	conv_loss�� ?0��        )��P	W��A�*

	conv_loss�� ?�4Ą        )��P	��A�*

	conv_loss�� ?��C        )��P	���A�*

	conv_loss�� ?�d\        )��P	�	�A�*

	conv_lossJ� ?%F�H        )��P	�K	�A�*

	conv_lossM� ?e�        )��P	�~	�A�*

	conv_losss� ?��U        )��P	v�	�A�*

	conv_loss�� ?~�e        )��P	��	�A�*

	conv_loss� ?܆z�        )��P	6
�A�*

	conv_loss� ?鮊�        )��P	�I
�A�*

	conv_lossB� ?���	        )��P	�|
�A�*

	conv_loss� ?j��        )��P	ˬ
�A�*

	conv_lossǳ ?�8\        )��P	��
�A�*

	conv_lossP� ?F�M�        )��P	��A�*

	conv_lossó ?}�R	        )��P	LB�A�*

	conv_loss�� ?�d��        )��P	?v�A�*

	conv_loss�� ?'�;        )��P	J��A�*

	conv_loss� ?boc�        )��P	���A�*

	conv_loss.� ?���        )��P	�A�*

	conv_loss�� ?�=+        )��P	D?�A�*

	conv_lossn� ?_�`�        )��P	&s�A�*

	conv_loss� ?�4z        )��P	��A�*

	conv_loss� ?o��,        )��P	���A�*

	conv_lossg� ?��[
        )��P	�A�*

	conv_loss:� ?D3i�        )��P	�5�A�*

	conv_lossߠ ?@��        )��P	
h�A�*

	conv_loss�� ?���        )��P	��A�*

	conv_loss�� ?%�\
        )��P		��A�*

	conv_lossr� ?(Ɋ�        )��P	���A�*

	conv_loss� ?�JK/        )��P	R/�A�*

	conv_lossC� ?d��        )��P	�`�A�*

	conv_lossݓ ?�/��        )��P	���A�*

	conv_loss�� ?�'Z        )��P	0��A�*

	conv_loss̗ ?�o
        )��P	��A�*

	conv_loss�� ?ì��        )��P	8�A�*

	conv_loss�v ?n �        )��P	�m�A�*

	conv_loss�� ?1ix        )��P	:��A�*

	conv_loss� ?��y�        )��P	��A�*

	conv_loss�� ?���l        )��P	��A�*

	conv_loss� ?d��        )��P	_7�A�*

	conv_loss�~ ?���        )��P	�p�A�*

	conv_loss�� ?��        )��P	���A�*

	conv_loss4� ?zG�i        )��P	��A�*

	conv_lossy� ?�K�q        )��P	8$�A�*

	conv_loss e ?ǃ�        )��P	�W�A�*

	conv_loss�{ ?�WvN        )��P	���A�*

	conv_loss�n ?z�uQ        )��P	x��A�*

	conv_lossju ?g2(        )��P	���A�*

	conv_loss�z ?!�        )��P	M�A�*

	conv_loss�x ?��y&        )��P	�R�A�*

	conv_loss�j ?����        )��P	���A�*

	conv_loss{ ?؄`        )��P	���A�*

	conv_loss�x ?=��        )��P	^��A�*

	conv_loss�j ?��        )��P	�*�A�*

	conv_lossid ?eDv         )��P	�^�A�*

	conv_loss`k ?�X�        )��P	,��A�*

	conv_loss�c ?.���        )��P	���A�*

	conv_lossc ?�4X        )��P	��A�*

	conv_loss�W ?�Ul�        )��P	�5�A�*

	conv_loss9m ?���        )��P	Li�A�*

	conv_lossTj ?*��        )��P	��A�*

	conv_loss%U ?�F�        )��P	���A�*

	conv_loss(S ?N���        )��P	���A�*

	conv_loss�N ?���        )��P	�2�A�*

	conv_lossz[ ?bΘ�        )��P	�d�A�*

	conv_lossZQ ?S�)�        )��P	#��A�*

	conv_loss�M ?���A        )��P	��A�*

	conv_loss99 ?��6�        )��P	.��A�*

	conv_loss�C ?sV�:        )��P	+�A�*

	conv_lossKB ?��        )��P	\�A�*

	conv_loss�B ?����        )��P	���A�*

	conv_lossI6 ?<�Q        )��P	���A�*

	conv_loss�; ?����        )��P	G��A�*

	conv_loss`? ?��:        )��P	D$�A�*

	conv_lossR3 ?}l2S        )��P	�W�A�*

	conv_loss�? ?N��        )��P	��A�*

	conv_lossj> ?9��1        )��P		��A�*

	conv_lossj: ?�Hf�        )��P	i��A�*

	conv_loss| ?�7��        )��P	u"�A�*

	conv_loss�1 ?�&�        )��P	�S�A�*

	conv_loss� ?-��        )��P	|��A�*

	conv_loss�) ?����        )��P	���A�*

	conv_loss�) ?�x��        )��P	'��A�*

	conv_lossN, ?u�$        )��P	�-�A�*

	conv_loss ?��y�        )��P	�^�A�*

	conv_lossI ?YO`        )��P	x��A�*

	conv_loss ?xOq        )��P	���A�*

	conv_loss ?��޻        )��P	���A�*

	conv_lossZ ?VQ��        )��P	�6�A�*

	conv_lossU ?��&�        )��P	�g�A�*

	conv_loss? ?���        )��P	���A�*

	conv_loss� ?��v�        )��P	s��A�*

	conv_lossV ?��l�        )��P	��A�*

	conv_loss� ?��D�        )��P	�J�A�*

	conv_loss��?�pd        )��P	E~�A�*

	conv_loss��?i��|        )��P	q��A�*

	conv_loss�?���        )��P	
��A�*

	conv_loss��?���p        )��P	��A�*

	conv_lossQ ?G��        )��P	�D�A�*

	conv_loss4 ?4���        )��P	v�A�*

	conv_loss��?ʹBi        )��P	��A�*

	conv_loss� ?��t        )��P	c��A�*

	conv_loss? ?�OkZ        )��P	��A�*

	conv_lossd�?H��        )��P	B�A�*

	conv_loss��?%k�g        )��P	v�A�*

	conv_lossu�?
�        )��P	T��A�*

	conv_loss\�?��U�        )��P	���A�*

	conv_loss��?��,�        )��P	�
�A�*

	conv_loss��?�rW�        )��P	�=�A�*

	conv_loss}�?� ��        )��P	�o�A�*

	conv_loss�?��        )��P	��A�*

	conv_loss��?fg��        )��P	���A�*

	conv_loss�?��        )��P	��A�*

	conv_loss��?��        )��P	38�A�*

	conv_loss��?T)�K        )��P	@l�A�*

	conv_loss��?����        )��P	��A�*

	conv_loss_�?��l?        )��P	��A�*

	conv_loss��?�z�]        )��P	� �A�*

	conv_loss�?�a�        )��P	�5 �A�*

	conv_losso�?��'        )��P	�f �A�*

	conv_loss�?d�=�        )��P	�� �A�*

	conv_losso�?�O��        )��P	N� �A�*

	conv_loss8�?S�        )��P	�� �A�*

	conv_loss��?Q�t        )��P	�*!�A�*

	conv_loss#�?"s��        )��P	~[!�A�*

	conv_loss=�?�t�        )��P	��!�A�*

	conv_loss��?���7        )��P	J�!�A�*

	conv_lossc�?�~        )��P	��!�A�*

	conv_loss��?�i)Q        )��P	G#"�A�*

	conv_lossV�?ѻ�        )��P	�U"�A�*

	conv_loss��?ѓ�`        )��P	��"�A�*

	conv_lossǾ?�g%�        )��P	+�"�A�*

	conv_loss��?�^��        )��P	FM$�A�*

	conv_lossX�?gm�=        )��P	�$�A�*

	conv_loss��?H~�        )��P	5�$�A�*

	conv_loss�?�LG�        )��P	H�$�A�*

	conv_lossa�?�&�2        )��P	X%�A�*

	conv_lossծ?*���        )��P	�P%�A�*

	conv_loss�?��v�        )��P	��%�A�*

	conv_loss3�?@��        )��P	B�%�A�*

	conv_loss��?!�*�        )��P	X�%�A�*

	conv_loss�?O��/        )��P	�)&�A�*

	conv_loss}�?��&5        )��P	o[&�A�*

	conv_loss��?�.)$        )��P	�&�A�*

	conv_losss�?VD�i        )��P	:�&�A�*

	conv_loss��?;��@        )��P	��&�A�*

	conv_loss�?��?A        )��P	�*'�A�*

	conv_loss�?jj�Q        )��P	c`'�A�*

	conv_lossț?� �        )��P	��'�A�*

	conv_loss"�?X�1&        )��P	f�'�A�*

	conv_loss̔?WU��        )��P	� (�A�*

	conv_lossi�?t��        )��P	�2(�A�*

	conv_loss6�?�:��        )��P	�c(�A�*

	conv_loss=y?xCyH        )��P	"�(�A�*

	conv_loss�n?onJ        )��P	��(�A�*

	conv_lossIy?���        )��P	�(�A�*

	conv_loss?�|�2        )��P	'))�A�*

	conv_loss�r?����        )��P	�[)�A�*

	conv_lossy?��,        )��P	R�)�A�*

	conv_loss�h?
m�        )��P	?�)�A�*

	conv_lossm}?7AZH        )��P	��)�A�*

	conv_loss�l?����        )��P	�&*�A�*

	conv_loss)r?�;Hr        )��P	�W*�A�*

	conv_loss�\?f�'[        )��P	�*�A�*

	conv_loss�o?�3��        )��P	,�*�A�*

	conv_loss�s?&66I        )��P	��*�A�*

	conv_loss�K?�XI�        )��P	6%+�A�*

	conv_loss�h?��Q        )��P	�U+�A�*

	conv_loss�R?q���        )��P	�+�A�*

	conv_loss�Y?{i�I        )��P	��+�A�*

	conv_loss�W?\3��        )��P	q�+�A�*

	conv_lossi>?��f=        )��P	W,�A�*

	conv_lossab?T��*        )��P	Q,�A�*

	conv_loss�J?"�AW        )��P	a�,�A�*

	conv_lossAU?��X0        )��P	��,�A�*

	conv_lossp<?�8�        )��P	<�,�A�*

	conv_loss�M?m|SK        )��P	�-�A�*

	conv_loss;?��z        )��P	�M-�A�*

	conv_loss�4?�|b        )��P	i�-�A�*

	conv_loss�G?����        )��P	K�-�A�*

	conv_loss\U?q�qK        )��P	��-�A�*

	conv_loss�K?T�p�        )��P	�.�A�*

	conv_lossx7?�6��        )��P	"Y.�A�*

	conv_loss3I?&6V        )��P	��.�A�*

	conv_loss�A?:b�        )��P	�.�A�*

	conv_loss�7?~3        )��P	�.�A�*

	conv_loss�/?"��        )��P	q&/�A�*

	conv_loss�E?��r        )��P	�X/�A�*

	conv_loss�;?I�8�        )��P	"�/�A�*

	conv_loss�-?aP��        )��P	v�/�A�*

	conv_loss/?�'��        )��P	��/�A�*

	conv_loss ?D[2�        )��P	�/0�A�*

	conv_loss#?�S        )��P	�a0�A�*

	conv_lossE?�VK�        )��P	{�0�A�*

	conv_loss%'?'��        )��P	�0�A�*

	conv_loss2?ܪݠ        )��P	@�0�A�*

	conv_loss�?�        )��P	k/1�A�*

	conv_loss�)?�G�        )��P	0e1�A�*

	conv_loss ?�α'        )��P	��1�A�*

	conv_loss"?:�s<        )��P	z�1�A�*

	conv_loss�?-�?        )��P	D2�A�*

	conv_loss�?H�s        )��P	"=2�A�*

	conv_loss0?P�N=        )��P	$o2�A�*

	conv_loss�	?ńW�        )��P	��2�A�*

	conv_loss�?�B        )��P	��2�A�*

	conv_loss�?(c[�        )��P	�3�A�*

	conv_loss�?-�P6        )��P	�53�A�*

	conv_loss\?�0�Z        )��P	f3�A�*

	conv_loss� ?�i}�        )��P	�3�A�*

	conv_loss!?o��        )��P	G�3�A�*

	conv_loss��?����        )��P	m�3�A�*

	conv_loss(�?����        )��P	�-4�A�*

	conv_loss;�?�"˼        )��P	�_4�A�*

	conv_loss��?-k��        )��P	D�4�A�*

	conv_loss��?ƌӨ        )��P	R�4�A�*

	conv_lossZ�?���}        )��P	��4�A�*

	conv_loss��?�>��        )��P	�,5�A�*

	conv_loss��?l���        )��P	�_5�A�*

	conv_loss�?�
�        )��P	ؑ5�A�*

	conv_loss��?��6        )��P	�5�A�*

	conv_loss/�?y[5�        )��P	��5�A�*

	conv_loss;�?A��K        )��P	\(6�A�*

	conv_loss]�?a�        )��P	�Y6�A�*

	conv_loss��?�c��        )��P	N�6�A�*

	conv_loss��?��ߎ        )��P	��6�A�*

	conv_loss��?�N��        )��P	��6�A�*

	conv_loss��?=�U�        )��P	 7�A�*

	conv_losso�?�Y=*        )��P	AT7�A�*

	conv_loss��?���        )��P	��7�A�*

	conv_loss��?�Z�        )��P	ĸ7�A�*

	conv_loss��?�<`)        )��P	�7�A�*

	conv_loss��?�LA        )��P	�$8�A�*

	conv_loss�?�")a        )��P	Ti8�A�*

	conv_lossӣ?VW.d        )��P	B�8�A�*

	conv_loss��?����        )��P	:�8�A�*

	conv_loss~�?E3�q        )��P	o9�A�*

	conv_loss��?��)<        )��P	�?9�A�*

	conv_loss�?�L��        )��P	�z9�A�*

	conv_loss�?���        )��P	2�9�A�*

	conv_loss]�? -�        )��P	e�9�A�*

	conv_loss#�?���J        )��P	/:�A�*

	conv_lossy�?��"b        )��P	�P:�A�*

	conv_lossǬ?�r        )��P	�:�A�*

	conv_loss��?6z        )��P	��:�A�*

	conv_lossW�?:�K        )��P	��:�A�*

	conv_lossʥ?�Ko�        )��P	�;�A�*

	conv_loss�?���o        )��P	<O;�A�*

	conv_loss"�?�@@�        )��P	,�;�A�*

	conv_loss^�?Δ#        )��P	�;�A�*

	conv_lossH�?�P�9        )��P	��;�A�*

	conv_loss�?x^�        )��P	�'<�A�*

	conv_lossf�?�l*        )��P	&\<�A�*

	conv_lossU�?���        )��P	��<�A�*

	conv_lossM�?�Xj        )��P	-�<�A�*

	conv_loss:�?�j0�        )��P	��<�A�*

	conv_lossgz?�Rk        )��P	;!=�A�*

	conv_lossO�?%P�U        )��P	�Q=�A�*

	conv_loss�h?�j�        )��P	#�=�A�*

	conv_loss{�?��j�        )��P	��=�A�*

	conv_lossw�?��        )��P	��=�A�*

	conv_loss.�?J�Am        )��P	>�A�*

	conv_lossk�?<r3        )��P	|M>�A�*

	conv_loss/{?V:�F        )��P	�~>�A�*

	conv_losshx?%*�        )��P	>�>�A�*

	conv_loss2r?i�j�        )��P	��>�A�*

	conv_loss(X?o}��        )��P	W?�A�*

	conv_loss�c?5룿        )��P	D?�A�*

	conv_losss?�z�        )��P	3w?�A�*

	conv_lossf?�驭        )��P	o�?�A�*

	conv_lossJe??c�        )��P	\�?�A�*

	conv_lossSZ?1�/�        )��P	�@�A�*

	conv_lossuQ?{��        )��P	)?@�A�*

	conv_loss�`?L�;c        )��P	gq@�A�*

	conv_loss�\?+�"        )��P	�@�A�*

	conv_lossHU?}_`�        )��P	��@�A�*

	conv_loss*K?�)&        )��P	�A�A�*

	conv_lossHU?	�sf        )��P	�9A�A�*

	conv_loss-]?����        )��P	tkA�A�*

	conv_lossc?��o        )��P	��A�A�*

	conv_loss�B?��l�        )��P	[�A�A�*

	conv_lossQ?2tqv        )��P	B�A�*

	conv_loss�;?V��        )��P	38B�A�*

	conv_loss�9?ϒR        )��P	8�B�A�*

	conv_loss<?�Cl�        )��P	��B�A�*

	conv_loss!C?��U.        )��P	�B�A�*

	conv_loss�A?��]        )��P	�C�A�*

	conv_loss�'?���        )��P	$NC�A�*

	conv_lossp2?y'O        )��P	p�C�A�*

	conv_loss�B?"�-�        )��P	�C�A�*

	conv_loss�=?4̢        )��P	��C�A�*

	conv_loss!?ʬ/        )��P	�&D�A�*

	conv_loss�?]o=�        )��P	R`D�A�*

	conv_loss�(?�bo        )��P	��D�A�*

	conv_lossO7?0��=        )��P	F�D�A�*

	conv_loss�'?a���        )��P	��D�A�*

	conv_lossP?8��        )��P	`&E�A�*

	conv_loss%>?�:�        )��P	�WE�A�*

	conv_lossq?zS`        )��P	��E�A�*

	conv_lossu?]�G�        )��P	W�E�A�*

	conv_loss9?p2�        )��P	8�E�A�*

	conv_loss�*?�Jg�        )��P	�/F�A�*

	conv_loss�?�]�        )��P	�bF�A�*

	conv_lossa?KgKT        )��P	֖F�A�*

	conv_loss?0���        )��P	��F�A�*

	conv_loss�?쓾e        )��P	��F�A�*

	conv_loss�8?\xl        )��P	�"G�A�*

	conv_loss�
?�c"�        )��P	#TG�A�*

	conv_loss��?��B�        )��P	��G�A�*

	conv_loss��?�@��        )��P	^�G�A�*

	conv_loss��?�Q�         )��P	��G�A�*

	conv_loss��?��E~        )��P	�H�A�*

	conv_loss8�?T�vA        )��P	�<H�A�*

	conv_loss_?8)�R        )��P	�kH�A�*

	conv_loss�?�R��        )��P	��H�A�*

	conv_lossX�?�o@]        )��P	V�H�A�*

	conv_loss��?����        )��P	��H�A�*

	conv_loss��?�_��        )��P	�&I�A�*

	conv_loss��?Y8.�        )��P	�VI�A�*

	conv_loss��?���        )��P	��I�A�*

	conv_loss9�?}셮        )��P	�I�A�*

	conv_loss%�?!���        )��P	��I�A�*

	conv_loss��?9gn�        )��P	�J�A�*

	conv_loss��?2���        )��P	�FJ�A�*

	conv_loss/�?���F        )��P	�tJ�A�*

	conv_loss@�?��8k        )��P	5�J�A�*

	conv_loss>�?(��F        )��P	��J�A�*

	conv_loss��?s���        )��P	|K�A�*

	conv_loss��?�.g        )��P	�6K�A�*

	conv_loss=�?�d�        )��P	�fK�A�*

	conv_lossj�?O��        )��P	v�K�A�*

	conv_loss�?cg��        )��P	��K�A�*

	conv_loss�?M���        )��P	[�K�A�*

	conv_losst�?C�d�        )��P	��M�A�*

	conv_loss �?@-\5        )��P	E�M�A�*

	conv_loss��?(�4        )��P	��M�A�*

	conv_lossx�?��S�        )��P	N�A�*

	conv_loss��?`��        )��P	CN�A�*

	conv_loss��?�?�        )��P	�rN�A�*

	conv_loss��?�yc�        )��P	��N�A�*

	conv_loss�?	��        )��P	Q�N�A�*

	conv_lossȒ?�        )��P	2O�A�*

	conv_loss~�?�x�V        )��P	�?O�A�*

	conv_loss��?珇"        )��P	&tO�A�*

	conv_loss��?��C        )��P	u�O�A�*

	conv_loss7�?(M        )��P	>�O�A�*

	conv_lossku?���        )��P	N"P�A�*

	conv_lossы?n"`�        )��P	�SP�A�*

	conv_lossO�?�=��        )��P	~�P�A�*

	conv_loss��?�D�(        )��P	�P�A�*

	conv_losse�?| @        )��P	��P�A�*

	conv_lossd?��A�        )��P	OQ�A�*

	conv_lossQ_?�>U1        )��P	JCQ�A�*

	conv_lossZ�?�>�        )��P	�rQ�A�*

	conv_loss�u?��L�        )��P	g�Q�A�*

	conv_lossuz?��$        )��P	A�Q�A�*

	conv_lossL~?=Ä�        )��P	��Q�A�*

	conv_loss�q?b0]        )��P	A0R�A�*

	conv_loss�k?u�4�        )��P	�^R�A�*

	conv_loss�a?r��L        )��P	��R�A�*

	conv_loss"f?��        )��P	׻R�A�*

	conv_lossji?�X8c        )��P	e�R�A�*

	conv_loss�i?h��        )��P	S�A�*

	conv_loss�g?�-��        )��P	�DS�A�*

	conv_losseL?�rW+        )��P	sS�A�*

	conv_lossMV?W���        )��P	�S�A�*

	conv_lossy?��M;        )��P	��S�A�*

	conv_loss�B?�h�2        )��P	��S�A�*

	conv_lossBH?�NaF        )��P	/T�A�*

	conv_loss�<?��-�        )��P	�\T�A�*

	conv_loss�B?�â        )��P	S�T�A�*

	conv_loss>P?$���        )��P	O�T�A�*

	conv_loss�Q?���        )��P	��T�A�*

	conv_lossR4?2ΐh        )��P	�U�A�*

	conv_loss�O?��P�        )��P	BU�A�*

	conv_loss2M?���        )��P	CuU�A�*

	conv_loss�@?�5K�        )��P	^�U�A�*

	conv_lossz5?(ӭi        )��P	 �U�A�*

	conv_loss=4?�Av�        )��P	V�A�*

	conv_losse?O�Y�        )��P	kCV�A�*

	conv_loss�>?��i�        )��P	vtV�A�*

	conv_loss?&?[��6        )��P	��V�A�*

	conv_lossy$?�7�        )��P	g�V�A�*

	conv_loss�?���        )��P	�W�A�*

	conv_loss6?��LR        )��P	�JW�A�*

	conv_loss�?���        )��P	�{W�A�*

	conv_loss!?(]�?        )��P	�W�A�*

	conv_loss!�?Ό��        )��P	�W�A�*

	conv_loss�?R�        )��P	 X�A�*

	conv_lossl?��I�        )��P	SSX�A�*

	conv_loss#?�=��        )��P	��X�A�*

	conv_loss�?꒦�        )��P	$�X�A�*

	conv_loss;?�-�!        )��P	Y�A�*

	conv_losso?,:�        )��P	9Y�A�*

	conv_lossf?��/	        )��P	4mY�A�*

	conv_loss� ?m6        )��P	Z�Y�A�*

	conv_loss��?9�)        )��P	��Y�A�*

	conv_lossA?S��/        )��P	Z�A�*

	conv_loss@?\���        )��P	DZ�A�*

	conv_loss,�?\�        )��P	�yZ�A�*

	conv_loss��?]kfG        )��P	a�Z�A�*

	conv_loss��?樘        )��P	��Z�A�*

	conv_loss?F�-S        )��P	�[�A�*

	conv_loss��?T�Y        )��P	�F[�A�*

	conv_loss��?4���        )��P	}z[�A�*

	conv_lossb�?'m        )��P	c�[�A�*

	conv_loss.�?�y}�        )��P	��[�A�*

	conv_loss<�?r��        )��P	�\�A�*

	conv_loss��?s�*        )��P	|F\�A�*

	conv_loss��?E���        )��P	�y\�A�*

	conv_loss��?M�'        )��P	�\�A�*

	conv_losse�?LHjU        )��P	��\�A�*

	conv_loss��?�_��        )��P	W]�A�*

	conv_loss��?��^V        )��P	I]�A�*

	conv_lossp�?���_        )��P	�|]�A�*

	conv_loss�?,��        )��P	<�]�A�*

	conv_loss��?W޸        )��P	��]�A�*

	conv_lossϦ?Tq5        )��P	�^�A�*

	conv_loss#�?�E�        )��P	�Q^�A�*

	conv_loss��?>G��        )��P	 �^�A�*

	conv_loss��?�p�8        )��P	��^�A�*

	conv_loss�?{c�l        )��P	��^�A�*

	conv_lossP�?�`        )��P	�!_�A�*

	conv_loss �?�(�        )��P	gX_�A�*

	conv_loss��?�Ձ�        )��P	��_�A�*

	conv_loss˥?'��        )��P	e�_�A�*

	conv_lossM�?ѥ��        )��P	��_�A�*

	conv_loss>�?����        )��P	y(`�A�*

	conv_loss4�?�Ft        )��P	�^`�A�*

	conv_loss��?Xqz2        )��P	D�`�A�*

	conv_lossEw?�"	�        )��P	�`�A�*

	conv_loss~�?��S�        )��P	��`�A�*

	conv_lossi�?#         )��P	�e�A�*

	conv_loss"]?J��f        )��P	�e�A�*

	conv_lossU�?�99        )��P	�.f�A�*

	conv_loss-�?,���        )��P	�]f�A�*

	conv_lossW�?��Q        )��P	1�f�A�*

	conv_loss��?P4�        )��P	üf�A�*

	conv_loss�z?��CU        )��P	��f�A�*

	conv_loss[?��R        )��P	g�A�*

	conv_loss�l?EA�        )��P	Kg�A�*

	conv_loss�~?km#        )��P	D�g�A�*

	conv_loss�?WD�        )��P	�g�A�*

	conv_lossL`?aB:�        )��P	n�g�A�*

	conv_loss��?$g��        )��P	�h�A�*

	conv_loss�Y?��5        )��P	_Jh�A�*

	conv_loss�??aSM        )��P	�h�A�*

	conv_loss�i?�WIi        )��P	��h�A�*

	conv_loss'�?�l        )��P	(�h�A�*

	conv_lossQf?��-        )��P	�i�A�*

	conv_loss]?l��         )��P	*Qi�A�*

	conv_lossJ?�&        )��P	�i�A�*

	conv_loss�f?j�t        )��P	E�i�A�*

	conv_loss T?�xɧ        )��P	@�i�A�*

	conv_lossF?`.�V        )��P	oj�A�*

	conv_loss9$?}�,�        )��P	�Ej�A�*

	conv_lossL1?�'0�        )��P	�sj�A�*

	conv_loss";?Y~�C        )��P	9�j�A�*

	conv_loss9?=��        )��P	��j�A�*

	conv_lossT
?�*��        )��P	�k�A�*

	conv_lossN?�fq        )��P	�4k�A�*

	conv_lossy*?eʩ        )��P	�ek�A�*

	conv_loss??Me�        )��P	ʕk�A�*

	conv_loss�5?|@�r        )��P	�k�A�*

	conv_loss�	?zN1�        )��P	��k�A�*

	conv_loss�?�6!�        )��P	�!l�A�*

	conv_loss.?d�U        )��P	xPl�A�*

	conv_loss�?��2        )��P	�l�A�*

	conv_lossH!?��g        )��P	��l�A�*

	conv_lossE ?�N        )��P	N�l�A�*

	conv_loss5�?mra        )��P	�m�A�*

	conv_loss�?��,k        )��P	�;m�A�*

	conv_loss/?�m~�        )��P	Fjm�A�*

	conv_loss��?����        )��P	��m�A�*

	conv_loss�?cG��        )��P	�m�A�*

	conv_lossG?���        )��P	e�m�A�*

	conv_loss?"�R        )��P	j8n�A�*

	conv_lossv?�vG        )��P	hn�A�*

	conv_loss��?�EK        )��P	c�n�A�*

	conv_loss'�?]Fs3        )��P	�n�A�*

	conv_lossQ�?�Qz^        )��P	��n�A�*

	conv_loss	�?�\��        )��P	A+o�A�*

	conv_loss��?���j        )��P	7Yo�A�*

	conv_loss"�?}DIw        )��P	#�o�A�*

	conv_loss-�?�r<�        )��P	��o�A�*

	conv_losse�?�H(        )��P	h�o�A�*

	conv_loss��?��Z�        )��P	�+p�A�*

	conv_loss��?;�        )��P	�ap�A�*

	conv_loss5�?��qg        )��P	3�p�A�*

	conv_loss�?>�j�        )��P	�p�A�*

	conv_loss��?N/TY        )��P	�q�A�*

	conv_loss��?�4�        )��P	ZBq�A�*

	conv_loss]�?���t        )��P	�tq�A�*

	conv_loss"�?k��+        )��P	Фq�A�*

	conv_loss��?<��'        )��P	;�q�A�*

	conv_lossp�?����        )��P	$r�A�*

	conv_lossN�?;O�        )��P	OAr�A�*

	conv_loss~�?b#]f        )��P	tr�A�*

	conv_loss��?؁�        )��P	S�r�A�*

	conv_loss��?)���        )��P	��r�A�*

	conv_lossX�?���        )��P	Gs�A�*

	conv_loss*�?�J        )��P	�?s�A�*

	conv_loss�z?S�O        )��P	�rs�A�*

	conv_loss@�?�NV�        )��P	��s�A�*

	conv_lossD�?�5        )��P	��s�A�*

	conv_lossР?�By�        )��P	�t�A�*

	conv_loss�?Cv�        )��P	�5t�A�*

	conv_loss�?D�c�        )��P	�et�A�*

	conv_lossPk?g4��        )��P	�t�A�*

	conv_loss�o?���a        )��P	��t�A�*

	conv_lossJW?��t�        )��P	E�t�A�*

	conv_lossY�?͵�f        )��P	� u�A�*

	conv_loss]N?�.f        )��P	XQu�A�*

	conv_loss!t?&�?�        )��P	Հu�A�*

	conv_loss�q?�N��        )��P	�u�A�*

	conv_lossoZ?o`ff        )��P	��u�A�*

	conv_loss�E?�w��        )��P	�v�A�*

	conv_loss3?�q        )��P	m>v�A�*

	conv_lossf~?���e        )��P	�lv�A�*

	conv_loss�,?����        )��P	a�v�A�*

	conv_loss]q?Ow��        )��P	��v�A�*

	conv_loss]?U�b�        )��P		�v�A�*

	conv_lossMY?M�&�        )��P	z-w�A�*

	conv_loss�S?�2�Z        )��P	�^w�A�*

	conv_loss(;?�9�        )��P	�w�A�*

	conv_lossW?���/        )��P	�w�A�*

	conv_lossO]?�%}        )��P	��w�A�*

	conv_loss�??r�T        )��P	�&x�A�*

	conv_loss�?ë'�        )��P	�Vx�A�*

	conv_loss�?s���        )��P	��x�A�*

	conv_lossV ?UN�3        )��P	9�x�A�*

	conv_loss�"?���        )��P	��x�A�*

	conv_loss�B?��e        )��P	y�A�*

	conv_loss�?�=�j        )��P	o�z�A�*

	conv_loss�#?�&��        )��P	[�z�A�*

	conv_lossZ?�_��        )��P	c	{�A�*

	conv_loss71?1uo        )��P	R8{�A�*

	conv_lossL??��        )��P	l{�A�*

	conv_loss��?�w�        )��P	+�{�A�*

	conv_losst?��u        )��P	j�{�A�*

	conv_lossX?�!Pd        )��P	��{�A�*

	conv_lossl?[v�        )��P	p1|�A�*

	conv_loss��?���        )��P	kd|�A�*

	conv_lossB�?OD}�        )��P	��|�A�*

	conv_loss?�^'�        )��P	�|�A�*

	conv_loss��?}�L        )��P	�|�A�*

	conv_loss\�?j^        )��P	�5}�A�*

	conv_loss�?��i�        )��P	o}�A�*

	conv_loss��?�#        )��P	�}�A�*

	conv_loss?�?qo��        )��P	W�}�A�*

	conv_lossj�?�8        )��P	� ~�A�*

	conv_loss��?�6�f        )��P	�/~�A�*

	conv_loss=�?k5��        )��P	�^~�A�*

	conv_loss �?��DA        )��P	��~�A�*

	conv_lossG�?�R        )��P	��~�A�*

	conv_losss�?BW�u        )��P	��~�A�*

	conv_loss��?>g��        )��P	T�A�*

	conv_loss��?�X�        )��P	*L�A�*

	conv_losss�?��        )��P	z{�A�*

	conv_loss��?y6=�        )��P	u��A�*

	conv_loss��?)�F�        )��P	���A�*

	conv_loss�?��e�        )��P	I��A�*

	conv_loss��?�גO        )��P	5>��A�*

	conv_loss͢?�j�S        )��P	�q��A�*

	conv_loss��?򎧊        )��P	L���A�*

	conv_loss��?�u�        )��P	�Ҁ�A�*

	conv_loss/�?x�5i        )��P	��A�*

	conv_loss"�?��0�        )��P	m5��A�*

	conv_loss?�U��        )��P	0e��A�*

	conv_loss�p?���        )��P	y���A�*

	conv_loss,�?��        )��P	�Ł�A�*

	conv_lossX�?r�        )��P	���A�*

	conv_loss�^?���        )��P	�"��A�*

	conv_loss�B?|+�        )��P	<U��A�*

	conv_loss�K?�aMy        )��P	�A�*

	conv_lossMr?���@        )��P	H���A�*

	conv_loss�]?{;�x        )��P	���A�*

	conv_losskj?"���        )��P	���A�*

	conv_loss�e?S�B�        )��P	�@��A�*

	conv_loss1*?��*�        )��P	Kt��A�*

	conv_loss�p?�p��        )��P	䤃�A�*

	conv_loss+b?"���        )��P	,փ�A�*

	conv_loss�D?�+�m        )��P	���A�*

	conv_loss�3?�XJ3        )��P	_I��A�*

	conv_loss	L?H��        )��P	�y��A�*

	conv_loss�8?{=        )��P	r���A�*

	conv_loss5E?漺U        )��P	���A�*

	conv_loss�G??�L�        )��P	l��A�*

	conv_loss�`?/�X         )��P	JH��A�*

	conv_loss="?���e        )��P	����A�*

	conv_loss�7?91<        )��P	����A�*

	conv_loss1?���T        )��P	���A�*

	conv_loss�,?��:        )��P	�&��A�*

	conv_loss�?�Z�        )��P	0i��A�*

	conv_loss!*? �z�        )��P	����A�*

	conv_loss�?Ȁ_�        )��P	�φ�A�*

	conv_loss�?�@M�        )��P	���A�*

	conv_loss�?�LS        )��P	�<��A�*

	conv_loss~�?� 0i        )��P	�s��A�*

	conv_loss��?Ţ	`        )��P	����A�*

	conv_losss ?��^        )��P	����A�*

	conv_loss��?'Q�        )��P	���A�*

	conv_loss��?0�,        )��P	�C��A�*

	conv_loss�?`}��        )��P	Fw��A�*

	conv_loss#	?���        )��P	x���A�*

	conv_loss@�?���        )��P	߈�A�*

	conv_loss"�?H&A\        )��P	���A�*

	conv_loss��?�7'�        )��P	�B��A�*

	conv_loss�?M�        )��P	Hv��A�*

	conv_loss'�?��gU        )��P	m���A�*

	conv_loss�?A9=�        )��P	�ۉ�A�*

	conv_loss��?Ep        )��P	���A�*

	conv_loss��?�,`        )��P	�@��A�*

	conv_loss��?�u�        )��P	)u��A�*

	conv_loss�?@`�        )��P	����A�*

	conv_loss�?�ؗ        )��P	�ۊ�A�*

	conv_loss�?s	}�        )��P	O��A�*

	conv_loss��?�O        )��P	�A��A�*

	conv_loss�??8A,y        )��P	�w��A�*

	conv_loss��?�3��        )��P	����A�*

	conv_loss��?�}�        )��P	�ދ�A�*

	conv_loss�b?2^`�        )��P	���A�*

	conv_loss�?���        )��P	�E��A�*

	conv_loss��?A�.3        )��P	�z��A�*

	conv_loss}?&��<        )��P	ˮ��A�*

	conv_loss[?L�(�        )��P	$��A�*

	conv_loss�=?L�7        )��P	t��A�*

	conv_loss2U?DY        )��P	�L��A�*

	conv_lossyN?�Ռ�        )��P	���A�*

	conv_loss/S?�f�#        )��P	����A�*

	conv_lossp?Ӭ�P        )��P	��A�*

	conv_lossY?���S        )��P	���A�*

	conv_loss�/?|�        )��P	,Q��A�*

	conv_loss>+?�&D        )��P	����A�*

	conv_lossL?ny        )��P	m͎�A�*

	conv_loss6M?at��        )��P	���A�*

	conv_loss�'?���*        )��P	A��A�*

	conv_loss�X?h�S        )��P	�t��A�*

	conv_loss�0?�O         )��P	f���A�*

	conv_loss9?��        )��P	���A�*

	conv_loss ?�=�        )��P	���A�*

	conv_loss�?��^        )��P	kI��A�*

	conv_loss��?4�>        )��P	H~��A�*

	conv_loss��?�B$        )��P	���A�*

	conv_lossT?�<sv        )��P	��A�*

	conv_lossD�?b��u        )��P	�!��A�*

	conv_lossY??���        )��P	�]��A�*

	conv_lossi�?�>�        )��P	����A�*

	conv_loss
�?9�n�        )��P	Cԑ�A�*

	conv_loss)�?�EPW        )��P	���A�*

	conv_loss~?XW{        )��P	;��A�*

	conv_lossm�? �J        )��P	�m��A�*

	conv_lossˢ?RU@�        )��P	ʢ��A�*

	conv_loss��?���        )��P	Uג�A�*

	conv_loss��?V��        )��P	�
��A�*

	conv_loss�?49�        )��P	O>��A�*

	conv_loss?��!        )��P	p��A�*

	conv_loss�?���        )��P	����A�*

	conv_loss�?��9�        )��P	2֓�A�*

	conv_loss�?�2�C        )��P	]��A�*

	conv_lossP�?��        )��P	?��A�*

	conv_loss��?���,        )��P	{q��A�*

	conv_losswv?Qc        )��P	ʢ��A�*

	conv_loss�?i�H        )��P	�Ք�A�*

	conv_loss��?IF�        )��P	���A�*

	conv_loss��?|���        )��P	�:��A�*

	conv_lossyh?�i��        )��P	�n��A�*

	conv_loss�>?,��z        )��P	M���A�*

	conv_loss)X?��        )��P	Lӕ�A�*

	conv_loss�:?�Y�        )��P	��A�*

	conv_loss��?_g��        )��P	�:��A�*

	conv_loss)�?)��        )��P	�l��A�*

	conv_loss�Q?��hN        )��P	)���A�*

	conv_lossN;?�^�        )��P	�Җ�A�*

	conv_loss�U?Äk�        )��P	���A�*

	conv_loss�Q?H�͜        )��P	=:��A�*

	conv_lossY-?��^j        )��P	m��A�*

	conv_lossf?���        )��P	"���A�*

	conv_loss�k?���~        )��P	Eї�A�*

	conv_lossZ?vx        )��P	���A�*

	conv_loss�?��s        )��P	�;��A�*

	conv_loss:(?���        )��P	�n��A�*

	conv_lossBM?���        )��P	����A�*

	conv_loss? ? �'U        )��P	���A�*

	conv_losss�?�~
        )��P	���A�*

	conv_lossJ�?*u��        )��P	bM��A�*

	conv_loss� ?rJ�        )��P	����A�*

	conv_loss�&?�ک+        )��P	����A�*

	conv_loss��?O[�,        )��P	8��A�*

	conv_loss`?.'�        )��P	h!��A�*

	conv_lossg,?Fm��        )��P	V��A�*

	conv_loss�?vT�        )��P	����A�*

	conv_loss��?�U��        )��P	�ƚ�A�*

	conv_loss�?��`�        )��P	����A�*

	conv_loss��?�c�        )��P	a3��A�*

	conv_loss4�?�9s6        )��P	kk��A�*

	conv_loss7�?��9�        )��P	o���A�*

	conv_loss�l?�Go        )��P	�؛�A�*

	conv_lossч?�X�(        )��P	���A�*

	conv_loss�?�xݨ        )��P	@��A�*

	conv_loss�W? ��        )��P	�r��A�*

	conv_loss��?��q        )��P	릜�A�*

	conv_loss��?��        )��P	�ٜ�A�*

	conv_loss�I?·��        )��P	��A�*

	conv_lossِ?Ԕ�        )��P	�A��A�*

	conv_loss�}?���        )��P	0t��A�*

	conv_lossx�?���        )��P	O���A�*

	conv_lossk?�7w3        )��P	3ڝ�A�*

	conv_loss�P?K�o        )��P	P��A�*

	conv_lossL?ᥬ�        )��P	�?��A�*

	conv_loss�V?�s/        )��P	<t��A�*

	conv_loss�P?U�r�        )��P	?���A�*

	conv_lossX�?�f�T        )��P	ݞ�A�*

	conv_lossT8?���T        )��P	���A�*

	conv_losshb?}�        )��P	�D��A�*

	conv_loss��?�ErZ        )��P	�w��A�*

	conv_loss�?/��G        )��P	ê��A�*

	conv_loss�?/E        )��P	�ߟ�A�*

	conv_lossn�?C�        )��P	2��A�*

	conv_loss� ?v��        )��P	�F��A�*

	conv_loss�?��C        )��P	�z��A�*

	conv_loss)?�r9        )��P	����A�*

	conv_loss!�?�<        )��P	���A�*

	conv_loss�?���        )��P	��A�*

	conv_lossw�?�'�#        )��P	�J��A�*

	conv_loss��? ��        )��P	�~��A�*

	conv_loss��?��i        )��P	b���A�*

	conv_lossC?���        )��P	���A�*

	conv_lossd�?
:�        )��P	B��A�*

	conv_lossؘ?� �        )��P	�J��A�*

	conv_lossհ?��76        )��P	s}��A�*

	conv_loss�?�+�e        )��P	{���A�*

	conv_lossxj?�C�v        )��P	���A�*

	conv_lossY�?z�8t        )��P	�|��A�*

	conv_loss!]?g��E        )��P	'���A�*

	conv_loss�{?K�3�        )��P	���A�*

	conv_loss�s?|�ӣ        )��P	���A�*

	conv_loss�?�L&�        )��P	�M��A�*

	conv_lossab?�H         )��P	����A�*

	conv_loss@k?�tZ        )��P	2ϥ�A�*

	conv_loss�I?��        )��P	���A�*

	conv_lossy?I�;        )��P	�N��A�*

	conv_loss-?W©        )��P	n���A�*

	conv_lossZ@?�u�        )��P	u���A�*

	conv_loss:W?��2        )��P	l��A�*

	conv_loss!�?��H�        )��P	���A�*

	conv_loss2�?�
%"        )��P	�R��A�*

	conv_loss�;?h��        )��P	��A�*

	conv_loss�?JK �        )��P	���A�*

	conv_losse?���t        )��P	���A�*

	conv_loss��?�/!        )��P	�*��A�*

	conv_loss2?"�        )��P	wb��A�*

	conv_loss	?ءHm        )��P	d���A�*

	conv_loss�?e�aC        )��P	4ʨ�A�*

	conv_loss��?ZP�        )��P	����A�*

	conv_lossS�?�58        )��P	y/��A�*

	conv_loss �?�S2        )��P	Wb��A�*

	conv_loss��?n��R        )��P	є��A�*

	conv_loss'�?�S2        )��P	�ũ�A�*

	conv_loss	@?�y��        )��P	����A�*

	conv_loss��?�M��        )��P	/��A�*

	conv_loss'?��        )��P	b��A�*

	conv_lossiq?O̞        )��P	����A�*

	conv_loss=?�|E        )��P	�ʪ�A�*

	conv_loss�/?{��        )��P	h���A�*

	conv_loss�`?4�9P        )��P	�2��A�*

	conv_loss�*?�\��        )��P	ve��A�*

	conv_loss�?|�@i        )��P	����A�*

	conv_loss��?�^�        )��P	�˫�A�*

	conv_lossf?)��        )��P	����A�*

	conv_loss��?����        )��P	O2��A�*

	conv_loss�(?�Q��        )��P	�e��A�*

	conv_loss��?���i        )��P	����A�*

	conv_loss��?�V        )��P	�ʬ�A�*

	conv_loss��?HcR�        )��P	����A�*

	conv_loss�?e�o        )��P	�0��A�*

	conv_loss�?����        )��P	�d��A�*

	conv_loss��?�#H�        )��P	i���A�*

	conv_loss��?�^s        )��P	%˭�A�*

	conv_loss.�?@!�        )��P	{���A�*

	conv_lossE�?��5b        )��P	F3��A�*

	conv_loss:�?),        )��P	�g��A�*

	conv_loss.�?���a        )��P	ۙ��A�*

	conv_loss��?�,Hn        )��P	Cݮ�A�*

	conv_loss��?wIi        )��P	x��A�*

	conv_lossG~?a1�        )��P	�F��A�*

	conv_losssE?�YZ        )��P	}��A�*

	conv_loss�?���        )��P	㱯�A�*

	conv_loss4?�N��        )��P	���A�*

	conv_loss��?�lJ        )��P	��A�*

	conv_lossB�?���        )��P	�G��A�*

	conv_losso?H66        )��P	c���A�*

	conv_lossٮ?>�        )��P	����A�*

	conv_lossI?jMC�        )��P	`��A�*

	conv_loss�(?ya�        )��P	�$��A�*

	conv_loss��?=��`        )��P	�Y��A�*

	conv_losse?�:��        )��P	욱�A�*

	conv_loss�N?N-�|        )��P	nͱ�A�*

	conv_loss�?�z�#        )��P	���A�*

	conv_loss޽?�gV        )��P	�?��A�*

	conv_loss�?���        )��P	�z��A�*

	conv_loss�X?�'��        )��P	����A�*

	conv_lossG�?$��)        )��P	���A�*

	conv_lossd?��U�        )��P	Z��A�*

	conv_lossѸ?�ͼK        )��P	RL��A�*

	conv_loss��?���        )��P	=���A�*

	conv_loss�S?F~VM        )��P	���A�*

	conv_lossC?��J        )��P	���A�*

	conv_loss*D?��L        )��P	���A�*

	conv_lossA�?���S        )��P	�O��A�*

	conv_loss�?���        )��P	����A�*

	conv_loss@?t�K�        )��P	�Ŵ�A�*

	conv_lossB2?�I90        )��P	����A�*

	conv_lossG�?���1        )��P	�0��A�*

	conv_lossc�?�-u�        )��P	�c��A�*

	conv_lossx�?�X=�        )��P	���A�*

	conv_lossV�?;�'�        )��P	ʵ�A�*

	conv_loss�w?�e�        )��P	����A�*

	conv_loss9�?̉��        )��P	�-��A�*

	conv_losscz?�v�!        )��P	`��A�*

	conv_loss�O?��_�        )��P	����A�*

	conv_loss��?��u        )��P	KŶ�A�*

	conv_loss�&?�bI        )��P	����A�*

	conv_lossrf?*��u        )��P	�,��A�*

	conv_loss%:?���"        )��P	�_��A�*

	conv_loss�.?@>27        )��P	����A�*

	conv_loss~�?=$Z-        )��P	*Ƿ�A�*

	conv_loss�?٦�R        )��P	r���A�*

	conv_losst?f;_{        )��P		/��A�*

	conv_loss.�?�e/�        )��P	b��A�*

	conv_loss-�?1�=S        )��P	Ĕ��A�*

	conv_loss�?h�J�        )��P	5Ǹ�A�*

	conv_lossPE?Ė}t        )��P	����A�*

	conv_lossI�?�3��        )��P	S<��A�*

	conv_loss�X?�H��        )��P	No��A�*

	conv_lossf�?^>��        )��P	"���A�*

	conv_loss~�?�,�        )��P	�׹�A�*

	conv_loss��?Q���        )��P	�
��A�*

	conv_loss6T?d�{�        )��P	�=��A�*

	conv_lossb�?N��        )��P	�p��A�*

	conv_lossT|?Éw        )��P	p���A�*

	conv_loss�?ʹ�        )��P	���A�*

	conv_loss^;?��l�        )��P	���A�*

	conv_lossy�?�E��        )��P	%K��A�*

	conv_loss �?���        )��P	���A�*

	conv_loss�?N�r:        )��P	���A�*

	conv_loss�?��8�        )��P	��A�*

	conv_lossX�?�"3�        )��P	'&��A�*

	conv_loss��?�8#        )��P	�[��A�*

	conv_loss�\?qՂ�        )��P	钼�A�*

	conv_loss�N?���        )��P	EƼ�A�*

	conv_loss��?���        )��P	8���A�*

	conv_loss2�?vz�        )��P	b-��A�*

	conv_loss<�?�[X�        )��P	�b��A�*

	conv_loss�?�S��        )��P	Ô��A�*

	conv_lossi?Y��        )��P	Wɽ�A�*

	conv_loss
�?�:@        )��P	���A�*

	conv_lossޯ?�!B�        )��P	-��A�*

	conv_loss]�?��ed        )��P	`��A�*

	conv_loss��?t�k        )��P	����A�*

	conv_loss��?�(        )��P	�ɾ�A�*

	conv_loss��?�y�        )��P	M���A�*

	conv_losso5?�G        )��P	�3��A�*

	conv_loss�
?BH.        )��P	%e��A�*

	conv_loss��??bT�        )��P	���A�*

	conv_loss�~?qH��        )��P	�ʿ�A�*

	conv_loss��?9^��        )��P	����A�*

	conv_loss�?��T�        )��P	w0��A�*

	conv_loss�e?�F��        )��P	�d��A�*

	conv_lossy�?��A�        )��P	����A�*

	conv_loss��?��        )��P	����A�*

	conv_loss�B?cw��        )��P	����A�*

	conv_lossO�?�2M�        )��P	�2��A�*

	conv_loss .?C i        )��P	�g��A�*

	conv_loss�|?�«�        )��P	���A�*

	conv_loss�Q?Ӕ��        )��P	����A�*

	conv_lossr�?h�D        )��P	G��A�*

	conv_lossb�?�!��        )��P	�3��A�*

	conv_loss#8?�:�        )��P	�g��A�*

	conv_lossA*?KF�        )��P	���A�*

	conv_loss��?C�n        )��P	����A�*

	conv_loss�?���3        )��P	���A�*

	conv_loss��?�idK        )��P	%5��A�*

	conv_lossM?1��        )��P	�~��A�*

	conv_loss�?͠H�        )��P	���A�*

	conv_lossgY?�4��        )��P	���A�*

	conv_loss�Q?���E        )��P	���A�*

	conv_loss�?h6�D        )��P	�P��A�*

	conv_loss��?5/�y        )��P	���A�*

	conv_loss��?Ko�        )��P	����A�*

	conv_loss��?`v�        )��P	����A�*

	conv_loss��?�;�Z        )��P	u6��A�*

	conv_lossx�?s�vo        )��P	�h��A�*

	conv_loss޹?j9,�        )��P	���A�*

	conv_losssR?��        )��P	���A�*

	conv_loss�g?����        )��P	Z��A�*

	conv_loss�?�ѩY        )��P	"E��A�*

	conv_loss�}?CS�n        )��P	<|��A�*

	conv_losst�?
x\        )��P	ܲ��A�*

	conv_loss �?�	�        )��P	����A�*

	conv_loss��?�H��        )��P	 ��A�*

	conv_loss��?/Z��        )��P	�M��A�*

	conv_loss�L?��-&        )��P	����A�*

	conv_lossU�
?���        )��P	1���A�*

	conv_loss��
?7�        )��P	C���A�*

	conv_loss��
?"2�        )��P	���A�*

	conv_loss�
?�3�        )��P	8M��A�*

	conv_lossz�
?�A�        )��P	B���A�*

	conv_loss �
?�e�        )��P	g���A�*

	conv_loss�
?{>x        )��P	���A�*

	conv_loss��
?��i        )��P	}��A�*

	conv_loss��	?cz�        )��P	\I��A�*

	conv_lossZe
?f(V#        )��P	K{��A�*

	conv_loss�&
?���        )��P	=���A�*

	conv_lossjg	?��q        )��P	���A�*

	conv_lossu/
?a�(`        )��P	���A�*

	conv_loss`&	?�ǚM        )��P	�F��A�*

	conv_loss��?Y=        )��P	dy��A�*

	conv_lossI�?:�bp        )��P	]���A�*

	conv_losssk	?zG�        )��P	P���A�*

	conv_loss4�	?=�ƥ        )��P	���A�*

	conv_loss�#?y{V�        )��P	XE��A�*

	conv_loss��?�e�        )��P	w��A�*

	conv_loss�?ؑE�        )��P	]���A�*

	conv_loss��?4�18        )��P	����A�*

	conv_loss�c?��<        )��P	'��A�*

	conv_lossB?i=�        )��P	�B��A�*

	conv_losso?�ܒ�        )��P	\v��A�*

	conv_loss4e?�p��        )��P	0���A�*

	conv_loss�P?wH�@        )��P	k���A�*

	conv_loss�?��k        )��P	C��A�*

	conv_loss��??�t        )��P	�@��A�*

	conv_loss^�?��        )��P	:��A�*

	conv_loss�{?Իa        )��P	����A�*

	conv_loss��?�+��        )��P	����A�*

	conv_loss:?â
�        )��P	����A�*

	conv_loss�]?�Y �        )��P	�(��A�*

	conv_lossd�?��u�        )��P	�\��A�*

	conv_loss��?�<?        )��P	t���A�*

	conv_loss�q?�.�        )��P	���A�*

	conv_loss��?�\H�        )��P	����A�*

	conv_loss��?mk��        )��P	$1��A�*

	conv_loss�0?Q'�U        )��P	�k��A�*

	conv_loss�T?qi��        )��P	9���A�*

	conv_loss��?���U        )��P	����A�*

	conv_loss��?�M�	        )��P	���A�*

	conv_loss�?+�=�        )��P	'B��A�*

	conv_loss�;?�:Tp        )��P	�q��A�*

	conv_lossj1?�M�#        )��P	����A�*

	conv_loss�H?����        )��P	+���A�*

	conv_lossc�?�$��        )��P	j ��A�*

	conv_loss{c?	`��        )��P	$1��A�*

	conv_loss��?H{�        )��P	�a��A�*

	conv_lossy?갚d        )��P	����A�*

	conv_loss�?y{ӑ        )��P	Q���A�*

	conv_loss��?�w�t        )��P	���A�*

	conv_lossaO?x��        )��P	�.��A�*

	conv_loss/?�ˡ�        )��P	m_��A�*

	conv_loss�3?$d	        )��P	���A�*

	conv_loss&"?�ox        )��P	����A�*

	conv_lossg"?l�`F        )��P	����A�*

	conv_loss�y?�vR�        )��P	W#��A�*

	conv_loss�� ?F=��        )��P	�Q��A�*

	conv_loss�'?2}o�        )��P	����A�*

	conv_loss���>阗:        )��P	
���A�*

	conv_lossG?�AW�        )��P	���A�*

	conv_loss/�>kW~�        )��P	���A�*

	conv_losse� ?*���        )��P	�E��A�*

	conv_loss���>`3�N        )��P	�~��A�*

	conv_loss���>І��        )��P	/���A�*

	conv_loss��>`ED�        )��P	����A�*

	conv_lossP��>z�Æ        )��P	���A�*

	conv_loss�L�>򒓏        )��P	�R��A�*

	conv_lossO��>\�|�        )��P	����A�*

	conv_loss���>@��X        )��P	ð��A�*

	conv_lossF��>� ��        )��P	b���A�*

	conv_lossX�>���>        )��P	���A�*

	conv_loss��>R{        )��P	g@��A�*

	conv_lossG��>�e�        )��P	Tp��A�*

	conv_loss|o�>a�88        )��P	����A�*

	conv_loss)T�>-3�        )��P	.���A�*

	conv_loss{��>���|        )��P	���A�*

	conv_lossè�>ۼ�i        )��P	2��A�*

	conv_loss*E�>���        )��P	�q��A�*

	conv_loss���> A�        )��P	���A�*

	conv_lossv��>�]��        )��P	����A�*

	conv_loss���>����        )��P	� ��A�*

	conv_lossYh�>H��	        )��P	?3��A�*

	conv_loss���>���v        )��P	Hb��A�*

	conv_loss�C�>Vx�        )��P	����A�*

	conv_loss�5�>����        )��P	����A�*

	conv_loss��>��K�        )��P	����A�*

	conv_lossS�>` ��        )��P	Z-��A�*

	conv_lossX��>oE"O        )��P	�b��A�*

	conv_loss�5�>YZ�        )��P	6���A�*

	conv_loss�#�>l���        )��P	U���A�*

	conv_loss�,�>Đ;'        )��P	����A�*

	conv_loss�Z�>�A        )��P	k8��A�*

	conv_loss7��>�A:�        )��P	�g��A�*

	conv_losss�>�夸        )��P	B���A�*

	conv_loss ��>��X�        )��P	����A�*

	conv_loss���>���c        )��P	����A�*

	conv_loss��>
�/M        )��P	�-��A�*

	conv_lossM/�>3�f4        )��P	^]��A�*

	conv_lossk��>�V�        )��P	z���A�*

	conv_loss��>Y~A#        )��P	���A�*

	conv_loss:�>�),�        )��P	����A�*

	conv_loss���>����        )��P	�$��A�*

	conv_lossHt�>C�        )��P	�T��A�*

	conv_loss���>~vQ�        )��P	6���A�*

	conv_loss��>���        )��P	4���A�*

	conv_loss���>>G�        )��P	E���A�*

	conv_loss!�>�M4A        )��P	��A�*

	conv_lossb{�>��D�        )��P	fH��A�*

	conv_loss,�>;��m        )��P	�w��A�*

	conv_loss�9�>��D�        )��P	Q���A�*

	conv_loss�^�>��`�        )��P	\���A�*

	conv_loss�>�?k        )��P	)	��A�*

	conv_loss~�>�|6�        )��P	 :��A�*

	conv_loss�3�>�c�        )��P	k��A�*

	conv_loss/�>�Q        )��P	����A�*

	conv_loss|�>GN        )��P	X���A�*

	conv_loss˔�>�]�k        )��P	����A�*

	conv_loss��>J1:�        )��P	�%��A�*

	conv_loss�K�>�pg        )��P	�T��A�*

	conv_lossR��>xh�/        )��P	w���A�*

	conv_loss���>���        )��P	����A�*

	conv_loss��>�P        )��P	"���A�*

	conv_loss��>�`3�        )��P	t��A�*

	conv_loss���>#��        )��P	�F��A�*

	conv_loss���>2��B        )��P	�v��A�*

	conv_loss ��>Y��t        )��P	����A�*

	conv_lossTW�>�        )��P	����A�*

	conv_lossw0�>�H}:        )��P	��A�*

	conv_loss��>߱e        )��P	�K��A�*

	conv_loss'6�>���        )��P	�y��A�*

	conv_loss6X�>��(K        )��P	`���A�*

	conv_loss���>ګ��        )��P	d���A�*

	conv_lossm�>ګ��        )��P	p��A�*

	conv_loss���>�(�m        )��P	-=��A�*

	conv_lossb�>P�2        )��P	�u��A�*

	conv_losspP�>S�E�        )��P	����A�*

	conv_loss��>Io2        )��P	W���A�*

	conv_loss}��>���         )��P	���A�*

	conv_lossq��>�	{.        )��P	�I��A�*

	conv_loss���>����        )��P	����A�*

	conv_loss���>�R�        )��P	+���A�*

	conv_lossʛ�>�        )��P	����A�*

	conv_loss{��>�.[        )��P	�%��A�*

	conv_loss�0�>�6��        )��P	dT��A�*

	conv_lossm�>?~��        )��P	j���A�*

	conv_loss��>�i�        )��P	����A�*

	conv_loss���>X7��        )��P	���A�*

	conv_lossa�>���        )��P	���A�*

	conv_loss�F�>��c        )��P	�A��A�*

	conv_loss���>^Za        )��P	�t��A�*

	conv_loss���>0��5        )��P	����A�*

	conv_loss�0�>���!        )��P	t���A�*

	conv_loss��>��N�        )��P	i��A�*

	conv_loss���>uF?Z        )��P	<5��A�*

	conv_loss���>���        )��P	!e��A�*

	conv_loss�,�>�        )��P	5���A�*

	conv_loss,��>�>         )��P	^���A�*

	conv_loss?��>��.k        )��P	���A�*

	conv_loss+�>�        )��P	%��A�*

	conv_loss��>Cq:        )��P	�U��A�*

	conv_loss���>�[�        )��P	"���A�*

	conv_loss1��>Ѐ�L        )��P	.���A�*

	conv_loss���>"-��        )��P	|���A�*

	conv_loss���>R|[l        )��P	R��A�*

	conv_loss`�>d���        )��P	M��A�*

	conv_loss'.�>y�A�        )��P	�}��A�*

	conv_loss���>q        )��P	����A�*

	conv_loss���>��`A        )��P	_���A�*

	conv_loss��>F,�#        )��P	5��A�*

	conv_lossXp�>ף�#        )��P	�>��A�*

	conv_loss�a�>	OD�        )��P	�o��A�*

	conv_loss���>�2��        )��P	����A�*

	conv_loss6�>���        )��P	E���A�*

	conv_loss佾>6��        )��P	���A�*

	conv_loss}c�>�10�        )��P	i1��A�*

	conv_loss^�>�?��        )��P	�b��A�*

	conv_lossg��>K��        )��P	ԑ��A�*

	conv_loss5o�>l�nN        )��P	7���A�*

	conv_lossq-�>��P�        )��P	���A�*

	conv_lossoi�>I ��        )��P	_4��A�*

	conv_loss"��>|V6m        )��P	�d��A�*

	conv_loss ��>����        )��P	���A�*

	conv_lossC.�>"�t        )��P	����A�*

	conv_loss�͸>?�H�        )��P	��A�*

	conv_loss)Ʒ>�jj2        )��P	6��A�*

	conv_loss���>���        )��P	
n��A�*

	conv_loss@�>���$        )��P	����A�*

	conv_loss�A�>YDד        )��P	���A�*

	conv_loss�v�>֨��        )��P	���A�*

	conv_loss+ι>�<�        )��P	�=��A�*

	conv_lossOɹ>jX}�        )��P	Oq��A�*

	conv_loss���>���        )��P	+���A�*

	conv_lossђ�>[3O�        )��P	����A�*

	conv_loss> �>j���        )��P	D��A�*

	conv_loss◷>��`        )��P	�A��A�*

	conv_loss��>��h        )��P	u��A�*

	conv_lossv�> �        )��P	g���A�*

	conv_lossC�>|�Z{        )��P	����A�*

	conv_loss$��>+J{�        )��P	��A�*

	conv_loss�K�>�!ъ        )��P	�9��A�*

	conv_lossƵ>C�w        )��P	�i��A�*

	conv_loss���>4`P        )��P	'���A�*

	conv_lossFN�>A�V�        )��P	l���A�*

	conv_loss��>3�iS        )��P	���A�*

	conv_lossaS�>h���        )��P	*��A�*

	conv_loss���>�A��        )��P	�^��A�*

	conv_losss��>����        )��P	����A�*

	conv_loss�>��Ǚ        )��P	����A�*

	conv_loss�a�>3bķ        )��P	����A�*

	conv_loss*��>��Ѹ        )��P	6$��A�*

	conv_loss@��>�`y         )��P	KU��A�*

	conv_lossD[�>i9d�        )��P	���A�*

	conv_loss~��>�uڑ        )��P	B���A�*

	conv_lossb3�>�v�        )��P	����A�*

	conv_lossٳ�>3��q        )��P	[��A�*

	conv_loss�C�>�,{�        )��P	F��A�*

	conv_loss�ǲ>%B*
        )��P	�v��A�*

	conv_loss���>�/�2        )��P	����A�*

	conv_loss�W�>��a�        )��P	���A�*

	conv_lossu	�>y��U        )��P	��A�*

	conv_loss�ߵ>���        )��P	H7��A�*

	conv_loss.�>���        )��P	�h��A�*

	conv_loss��>)!�M        )��P	W���A�*

	conv_loss�g�>��<
        )��P	Z���A�*

	conv_loss���>܀@�        )��P	:���A�*

	conv_loss��>P�{�        )��P	�.��A�*

	conv_lossgٳ>z�        )��P	�`��A�*

	conv_loss�8�>w��        )��P	}���A�*

	conv_lossC}�>G��        )��P	�)��A�*

	conv_loss��>F$�        )��P	�Y��A�*

	conv_lossk׷>�}�\        )��P	����A�*

	conv_losscI�>W�        )��P	5���A�*

	conv_loss��>5S�        )��P	����A�*

	conv_loss̱>�12l        )��P	�/��A�*

	conv_loss>@��        )��P	�c��A�*

	conv_loss���>)�g&        )��P	����A�*

	conv_losss��>q::�        )��P	����A�*

	conv_loss���>��a�        )��P	���A�*

	conv_loss!5�>LG�        )��P	�L��A�*

	conv_loss�Z�>X� �        )��P	;��A�*

	conv_losso��>vcG�        )��P	���A�*

	conv_lossL�>�mv        )��P	����A�*

	conv_loss��>Ā�        )��P	���A�*

	conv_loss�F�>ߞܑ        )��P	3F��A�*

	conv_loss���>l!ù        )��P	Dx��A�*

	conv_loss�ׯ>��7        )��P	۫��A�*

	conv_lossӐ�>#͞T        )��P	{���A�*

	conv_lossh��>lfD�        )��P	�
 �A�*

	conv_loss#��>�3��        )��P	5E �A�*

	conv_loss|ӱ>�#=�        )��P	� �A�*

	conv_loss�t�>��        )��P	� �A�*

	conv_lossgY�>s*��        )��P	�� �A�*

	conv_lossCd�>���3        )��P	��A�*

	conv_lossee�>��p�        )��P	�O�A�*

	conv_loss��>�B        )��P	��A�*

	conv_lossF�>��Y        )��P	���A�*

	conv_losse~�>.��        )��P	���A�*

	conv_loss^F�>�Ɯ        )��P	��A�*

	conv_loss�̭>���B        )��P	�A�A�*

	conv_loss��>Q�        )��P	dp�A�*

	conv_loss9�>5z�        )��P	���A�*

	conv_loss���>��*        )��P	I��A�*

	conv_lossH�>L�>-        )��P	��A�*

	conv_lossi:�>W6ga        )��P	�C�A�*

	conv_loss=�>�,W        )��P	�u�A�*

	conv_lossϊ�>�^h�        )��P	���A�*

	conv_loss��>�3x=        )��P	n��A�*

	conv_lossj֮>�l�        )��P	��A�*

	conv_lossxs�>�S�g        )��P	7�A�*

	conv_lossC�>>qR        )��P	�h�A�*

	conv_loss���>���        )��P	F��A�*

	conv_loss��>����        )��P	���A�*

	conv_loss�C�>�xA�        )��P	w��A�*

	conv_loss�ʱ>�&�         )��P	%-�A�*

	conv_loss!"�>�/�T        )��P	u^�A�*

	conv_loss�Ŭ>�:�        )��P		��A�*

	conv_loss���>����        )��P	1��A�*

	conv_loss"v�>!�!�        )��P	
�A�*

	conv_loss|�>D �E        )��P	?�A�*

	conv_lossҸ�>.�\        )��P	�n�A�*

	conv_losse٭>��C�        )��P	Z��A�*

	conv_loss���>��_�        )��P	��A�*

	conv_loss�c�>�
:�        )��P	��A�*

	conv_lossW��>IH�
        )��P	}6�A�*

	conv_loss7�>=pb        )��P	kh�A�*

	conv_lossBʮ>(֒        )��P	C��A�*

	conv_lossz�>��A�        )��P	*��A�*

	conv_loss���>�^M&        )��P	 �A�*

	conv_lossm7�>��.        )��P	�P�A�*

	conv_loss�x�>K]��        )��P	!��A�*

	conv_loss��>�_��        )��P	ݾ�A�*

	conv_loss.`�>!���        )��P	L��A�*

	conv_loss���>r6\        )��P	�	�A�*

	conv_loss��>��.�        )��P	�S	�A�*

	conv_loss�.�>�~j        )��P	�	�A�*

	conv_loss^Q�>�D�        )��P	�	�A�*

	conv_lossR��>��C        )��P	��	�A�*

	conv_loss�Y�>�m�        )��P	�
�A�*

	conv_lossL|�>)�R�        )��P	�N
�A�*

	conv_loss3��>��!�        )��P	�
�A�*

	conv_lossۮ>�-��        )��P	��
�A�*

	conv_loss�^�>뾶        )��P	��
�A�*

	conv_loss���>��        )��P	��A�*

	conv_loss�9�>�ҏ�        )��P	AS�A�*

	conv_loss��>��)        )��P	���A�*

	conv_loss�ή>�A�        )��P	9��A�*

	conv_loss���>�j��        )��P	���A�*

	conv_loss9�>��        )��P	}#�A�*

	conv_loss�\�>;���        )��P	$Y�A�*

	conv_loss�.�>�W:w        )��P	���A�*

	conv_lossm�>$@N�        )��P	U��A�*

	conv_loss�ӯ>��L�        )��P	���A�*

	conv_loss�O�>�w�        )��P	*�A�*

	conv_loss�*�>��6@        )��P	�[�A�*

	conv_loss"��>8���        )��P	G��A�*

	conv_loss�P�>�D��        )��P	���A�*

	conv_loss޽�>DZ��        )��P	{��A�*

	conv_loss�>U0        )��P	I'�A�*

	conv_loss���>c�}�        )��P	!W�A�*

	conv_loss8F�>��c�        )��P	4��A�*

	conv_loss��>fK�D        )��P	���A�*

	conv_loss�f�>�w#        )��P	���A�*

	conv_loss���>��        )��P	�'�A�*

	conv_lossP�>Gs��        )��P	j^�A�*

	conv_lossE�>��6t        )��P	/��A�*

	conv_lossѮ>�Lv�        )��P	a��A�*

	conv_lossޞ�>�:�        )��P	��A�*

	conv_loss�Ү>f�        )��P	/E�A�*

	conv_loss,��>��1@        )��P	�v�A�*

	conv_loss�>MǥQ        )��P	���A�*

	conv_loss���>YQ        )��P	!��A�*

	conv_lossL�>��k^        )��P	9�A�*

	conv_lossE��>Ȼq        )��P	kH�A�*

	conv_loss� �>��5t        )��P	1{�A�*

	conv_loss�%�>=��g        )��P	[��A�*

	conv_lossX7�>q>,}        )��P	���A�*

	conv_loss(�>���        )��P	'1�A�*

	conv_loss^��>�u��        )��P	�`�A�*

	conv_lossU�>~9�        )��P	A��A�*

	conv_loss�>�"SY        )��P	g��A�*

	conv_loss��>��p�        )��P	��A�*

	conv_loss��>zƩ        )��P	�:�A�*

	conv_loss��>���        )��P	Dk�A�*

	conv_loss4�>��:c        )��P	���A�*

	conv_loss��>h=�        )��P	��A�*

	conv_loss���>�H�        )��P	�	�A�*

	conv_lossg8�>�v        )��P	�D�A�*

	conv_loss�^�>��r�        )��P	���A�*

	conv_lossa��>�x�        )��P	���A�*

	conv_loss&�>�6[        )��P	��A�*

	conv_loss�T�>�m�%        )��P	V+�A�*

	conv_loss�ì>��D�        )��P	]�A�*

	conv_loss�C�>���        )��P	���A�*

	conv_lossna�>�(        )��P	p��A�*

	conv_loss�7�>��L�        )��P	���A�*

	conv_loss
G�>�z        )��P	9�A�*

	conv_loss�ů>M���        )��P	dq�A�*

	conv_loss٭>��6T        )��P	��A�*

	conv_lossک>�p�I        )��P	���A�*

	conv_lossTO�>�m��        )��P	��A�*

	conv_loss�>=���        )��P	�I�A�*

	conv_loss�ά>ߋE�        )��P	�x�A�*

	conv_lossՅ�>�\�        )��P	b��A�*

	conv_lossQ�>�Y��        )��P	���A�*

	conv_loss���>NXY�        )��P	��A�*

	conv_loss��>�w��        )��P	3=�A�*

	conv_loss���>=�1        )��P	�l�A�*

	conv_loss�*�>��7O        )��P		��A�*

	conv_lossLi�>B�?�        )��P	���A�*

	conv_loss5��>~��        )��P	���A�*

	conv_loss�C�>3<�        )��P	L*�A�*

	conv_loss��>�ztF        )��P	�W�A�*

	conv_loss�ĩ>�M�        )��P	���A�*

	conv_lossP��>5B        )��P	X��A�*

	conv_losssp�>"��:        )��P	���A�*

	conv_losspE�>�|�        )��P	��A�*

	conv_loss�r�>Y���        )��P	oG�A�*

	conv_losse�>�ո        )��P	��A�*

	conv_lossk��>�P�        )��P	���A�*

	conv_lossZi�>-=�        )��P	9��A�*

	conv_loss&3�>��ó        )��P	��A�*

	conv_loss\�>���        )��P	�L�A�*

	conv_lossHī>롋o        )��P	!|�A�*

	conv_lossXX�>y��1        )��P	1��A�*

	conv_loss8¯>hV��        )��P	,��A�*

	conv_loss~�>K���        )��P	��A�*

	conv_losse��>��Y        )��P	�Q�A�*

	conv_loss��>~�#        )��P	t��A�*

	conv_loss/�>�!�        )��P	��A�*

	conv_loss�>�1d        )��P	���A�*

	conv_loss�˫>��        )��P	��A�*

	conv_loss�{�><�/�        )��P	�K�A�*

	conv_loss د>�mX        )��P	C}�A�*

	conv_loss��>B��m        )��P	���A�*

	conv_loss�O�>0�`        )��P	���A�*

	conv_loss��>�ժ�        )��P	��A�*

	conv_lossT�>��        )��P	�;�A�*

	conv_loss�ʫ>ǈ�        )��P	�o�A�*

	conv_loss���>��N�        )��P	أ�A�*

	conv_loss��>mh�H        )��P	���A�*

	conv_loss�C�>�>(        )��P	�
�A�*

	conv_loss^��>b�P�        )��P	�8�A�*

	conv_lossW��>����        )��P	�g�A�*

	conv_loss"C�>�.�        )��P	���A�*

	conv_loss���>��C        )��P	E��A�*

	conv_loss��>LrQ<        )��P	���A�*

	conv_loss���>�6        )��P	q% �A�*

	conv_lossJ��>!�9        )��P	�T �A�*

	conv_lossmߩ>�jw        )��P	� �A�*

	conv_loss�a�>#6l        )��P	q� �A�*

	conv_loss?��>��I�        )��P	~� �A�*

	conv_loss�9�>��?        )��P	x!�A�*

	conv_lossP��>����        )��P	�B!�A�*

	conv_lossg!�>Y�-        )��P	'p!�A�*

	conv_loss�a�>o��        )��P	�!�A�*

	conv_lossg!�>��w�        )��P	�!�A�*

	conv_lossF�>L]1�        )��P	��!�A�*

	conv_loss1>�>�e        )��P	�*"�A�*

	conv_loss�5�>V�_�        )��P	?\"�A�*

	conv_loss}7�>E��        )��P	��"�A�*

	conv_lossU٪>Z�2?        )��P	&�"�A�*

	conv_loss(ڮ>�oC�        )��P	��"�A�*

	conv_losse~�>6��        )��P	�#�A�*

	conv_loss�©>g"CU        )��P	LG#�A�*

	conv_lossር>���        )��P	/z#�A�*

	conv_lossW��>*H)h        )��P	��#�A�*

	conv_lossG-�>���        )��P	��#�A�*

	conv_loss��>���        )��P	pj%�A�*

	conv_lossA�>O��        )��P	k�%�A�*

	conv_lossR�>��De        )��P	n�%�A�*

	conv_loss/M�>�P�        )��P	��%�A�*

	conv_loss��>���        )��P	�)&�A�*

	conv_loss�ǭ>dR��        )��P	.W&�A�*

	conv_lossݩ>��        )��P	:�&�A�*

	conv_loss���>�r<�        )��P	E�&�A�*

	conv_loss�Ҭ>*�a        )��P	��&�A�*

	conv_lossm�>;2O        )��P	� '�A�*

	conv_loss��>Ed�        )��P	>S'�A�*

	conv_lossn�>�0�        )��P	^�'�A�*

	conv_lossJ��>:;	        )��P	�'�A�*

	conv_loss�N�>��r        )��P	��'�A�*

	conv_lossOu�>�tM        )��P	"(�A�*

	conv_loss�L�>�2�        )��P	N(�A�*

	conv_lossM�>���]        )��P	K�(�A�*

	conv_lossF��>s�        )��P	��(�A�*

	conv_loss=R�>�W��        )��P	�(�A�*

	conv_loss��>��8�        )��P	�)�A�*

	conv_loss59�>�̣�        )��P	�I)�A�*

	conv_loss���>R�_        )��P	z)�A�*

	conv_loss��>�<T�        )��P	��)�A�*

	conv_loss�$�>�ۺ�        )��P	�)�A�*

	conv_loss择>@(L        )��P	e*�A�*

	conv_loss���>+t��        )��P	%9*�A�*

	conv_loss/�>�:z        )��P	�j*�A�*

	conv_lossoǪ>�Jd        )��P	p�*�A�*

	conv_losso>�>eOu�        )��P	��*�A�*

	conv_lossM��>�tl        )��P	�+�A�*

	conv_loss�W�>��        )��P	u?+�A�*

	conv_loss���>2�a         )��P	2m+�A�*

	conv_loss\,�>�	�         )��P	�+�A�*

	conv_loss���>���N        )��P	8�+�A�*

	conv_loss���>CQnE        )��P	�,�A�*

	conv_loss�/�>�D�        )��P	j5,�A�*

	conv_loss7f�>�^s        )��P	Md,�A�*

	conv_loss;�>~c        )��P	~�,�A�*

	conv_loss��>t�.�        )��P	��,�A�*

	conv_loss�K�>�h(N        )��P	��,�A�*

	conv_loss���>.��        )��P	�&-�A�*

	conv_loss�<�>��        )��P	�W-�A�*

	conv_loss ��>lsa        )��P	��-�A�*

	conv_loss�ͫ>P��K        )��P	�-�A�*

	conv_lossPU�>JL        )��P	��-�A�*

	conv_loss8��>72)        )��P	#.�A�*

	conv_loss�]�> ]3�        )��P	�F.�A�*

	conv_lossR�>R�6�        )��P	[w.�A�*

	conv_loss&�>wF�P        )��P	��.�A�*

	conv_lossy��>R�g        )��P	��.�A�*

	conv_lossa��>�ǩ        )��P	�/�A�*

	conv_loss[�>�Fl�        )��P	&K/�A�*

	conv_loss!�>P��        )��P	3y/�A�*

	conv_loss��>��        )��P	˦/�A�*

	conv_loss�l�>���        )��P	}�/�A�*

	conv_lossi�>;�S        )��P	�	0�A�*

	conv_loss�+�>���        )��P	�?0�A�*

	conv_loss���>��s1        )��P	�o0�A�*

	conv_lossJĪ>���
        )��P	&�0�A�*

	conv_lossF�>G�!        )��P	U�0�A�*

	conv_loss]�>=��        )��P	�1�A�*

	conv_loss70�>�lt        )��P	�B1�A�*

	conv_loss|n�>�i�        )��P	<t1�A�*

	conv_loss�۩>U��        )��P	��1�A�*

	conv_lossr��>�cz        )��P	��1�A�*

	conv_lossū�>��;�        )��P	�2�A�*

	conv_loss�Ƨ>Ȝ
�        )��P	B2�A�*

	conv_lossK�>�U?I        )��P	�t2�A�*

	conv_loss廩>v��        )��P	��2�A�*

	conv_loss��>�1c        )��P	?�2�A�*

	conv_loss0t�>N���        )��P	h3�A�*

	conv_loss�O�>q3S        )��P	�:3�A�*

	conv_loss�p�>��p        )��P	�j3�A�*

	conv_loss�3�>lM�n        )��P	��3�A�*

	conv_losseu�>�-�e        )��P	3�3�A�*

	conv_loss���>q��E        )��P	�4�A�*

	conv_loss`��>�,�v        )��P	�14�A�*

	conv_loss���>���A        )��P	�a4�A�*

	conv_loss�E�>A!�H        )��P	��4�A�*

	conv_loss���>�Z��        )��P	�4�A�*

	conv_loss��>Y��        )��P	��4�A�*

	conv_loss��>�)�        )��P	.)5�A�*

	conv_loss�g�>?���        )��P	'Y5�A�*

	conv_loss-f�>sq��        )��P	=�5�A�*

	conv_loss�j�>�P�        )��P	�5�A�*

	conv_lossK۪>�.�        )��P	��5�A�*

	conv_lossf��>���V        )��P	�6�A�*

	conv_loss��>>xI�        )��P	vD6�A�*

	conv_loss�]�>]0,�        )��P	�v6�A�*

	conv_loss�Z�>Hu��        )��P	�6�A�*

	conv_loss�;�>O�0        )��P	��6�A�*

	conv_loss �>�1}        )��P	�7�A�*

	conv_lossi��>�h��        )��P	�<7�A�*

	conv_loss�H�>�h�l        )��P	q7�A�*

	conv_loss��>�*�h        )��P	��7�A�*

	conv_loss%Щ>t        )��P	;�7�A�*

	conv_loss��>�`�        )��P	'8�A�*

	conv_loss�u�>��^        )��P	08�A�*

	conv_loss��>$pF        )��P	�`8�A�*

	conv_loss�q�>�E��        )��P	�=�A�*

	conv_lossxD�>�MhW        )��P	ZV=�A�*

	conv_loss4�>59��        )��P	��=�A�*

	conv_loss�ة>��	        )��P	��=�A�*

	conv_lossf^�>��o        )��P	��=�A�*

	conv_lossv��>���N        )��P	�>�A�*

	conv_loss�F�>���        )��P	C>�A�*

	conv_loss�]�>/��j        )��P	q>�A�*

	conv_loss`|�>��n�        )��P	��>�A�*

	conv_lossl$�>����        )��P	�>�A�*

	conv_loss���>���^        )��P	�?�A�*

	conv_loss=�>�~X�        )��P	7I?�A�*

	conv_loss�#�>��[�        )��P	�{?�A�*

	conv_loss��>@��A        )��P	լ?�A�*

	conv_loss�>�]�        )��P	��?�A�*

	conv_loss��>ڔ�#        )��P	j	@�A�*

	conv_loss��>�L�|        )��P	�C@�A�*

	conv_loss}D�>_�l�        )��P	r@�A�*

	conv_loss�w�>���M        )��P	=�@�A�*

	conv_loss4��>��9�        )��P	�@�A�*

	conv_lossZ��>��E�        )��P	�A�A�*

	conv_loss�̫>(�#�        )��P	31A�A�*

	conv_loss�I�>��b        )��P	u_A�A�*

	conv_loss4,�>���*        )��P	׏A�A�*

	conv_loss5�>)X�l        )��P	'�A�A�*

	conv_loss���>���8        )��P	��A�A�*

	conv_lossl{�>�рQ        )��P	�)B�A�*

	conv_lossO��>a�        )��P	YB�A�*

	conv_loss�֪>����        )��P	j�B�A�*

	conv_loss�(�>7��        )��P	m�B�A�*

	conv_lossaȩ>��        )��P	��B�A�*

	conv_loss!��>c`�        )��P	>C�A�*

	conv_loss꭫>mA��        )��P	�FC�A�*

	conv_loss��>����        )��P	�wC�A�*

	conv_loss� �>�x�Q        )��P	֦C�A�*

	conv_loss0I�>�9�        )��P	�C�A�*

	conv_lossШ�>O���        )��P	�D�A�*

	conv_loss�;�>���A        )��P	8D�A�*

	conv_loss�h�>_�M\        )��P	�gD�A�*

	conv_losstl�>���        )��P	�D�A�*

	conv_lossd��>4��        )��P	��D�A�*

	conv_lossk��>����        )��P	t�D�A�*

	conv_loss�8�>u,	o        )��P	'+E�A�*

	conv_lossCp�>��%        )��P	}YE�A�*

	conv_loss8��>Hb�k        )��P	 �E�A�*

	conv_loss�٩>p_QU        )��P	��E�A�*

	conv_loss��>��ү        )��P	C�E�A�*

	conv_lossx��>{߹�        )��P	jF�A�*

	conv_loss���>3�)        )��P	�SF�A�*

	conv_loss@�>��m�        )��P	ȅF�A�*

	conv_loss���>�j:        )��P	�F�A�*

	conv_loss���>;n��        )��P	AG�A�*

	conv_lossc9�>�pU        )��P	3G�A�*

	conv_loss=��>�,(�        )��P	�eG�A�*

	conv_lossuը>��        )��P	ܛG�A�*

	conv_loss�5�>�,        )��P	��G�A�*

	conv_loss�~�>��V�        )��P	fH�A�*

	conv_losszE�>*��        )��P	�7H�A�*

	conv_loss��>�}(3        )��P	w{H�A�*

	conv_loss�B�>k�p        )��P	i�H�A�*

	conv_losse0�>�og�        )��P	��H�A�*

	conv_lossUX�>��8�        )��P	}I�A�*

	conv_lossx��>���        )��P	ZLI�A�*

	conv_lossFé>�%        )��P	�I�A�*

	conv_loss��>���        )��P	7�I�A�*

	conv_loss"e�>���G        )��P	��I�A�*

	conv_lossMЩ>@�Z�        )��P	Z0J�A�*

	conv_lossߧ>W֠n        )��P	�bJ�A�*

	conv_loss��>A�u        )��P	A�J�A�*

	conv_loss��>O�        )��P	�J�A�*

	conv_loss���>ָ�        )��P	A�J�A�*

	conv_lossJ�>�ƒ�        )��P	�0K�A�*

	conv_lossT��>t�8A        )��P	�cK�A�*

	conv_lossĨ>oX�        )��P	��K�A�*

	conv_loss��>ou��        )��P	��K�A�*

	conv_lossCd�>��U]        )��P	bL�A�*

	conv_loss(%�> Z        )��P	s@L�A�*

	conv_loss囩>�q�Z        )��P	wL�A�*

	conv_loss�>	6Tv        )��P	ŮL�A�*

	conv_lossެ�>�JN        )��P	��L�A�*

	conv_loss�[�>T%         )��P	�M�A�*

	conv_loss�i�>W��        )��P	VM�A�*

	conv_loss��>���        )��P	l�M�A�*

	conv_loss�ר>�J�        )��P	�M�A�*

	conv_lossB\�>���        )��P	b�M�A�*

	conv_loss'��>R6l!        )��P	[0N�A�*

	conv_lossz:�>����        )��P	�gN�A�*

	conv_loss�	�>��"�        )��P	ОN�A�*

	conv_lossy�>Y�a        )��P	��N�A�*

	conv_loss��>���:        )��P	�O�A�*

	conv_loss�|�>5-��        )��P	6O�A�*

	conv_loss���>�:J        )��P	�gO�A�*

	conv_loss�(�>l���        )��P	%�O�A�*

	conv_loss�ʩ>���Q        )��P	��O�A�*

	conv_loss�ߦ>�	�K        )��P	t�O�A�*

	conv_lossf��>�#��        )��P	�2P�A�*

	conv_loss&��>�t�        )��P		fP�A�*

	conv_lossCX�>!�$        )��P	��P�A�*

	conv_loss݊�>^L_E        )��P	��P�A�*

	conv_lossִ�>���        )��P	��P�A�*

	conv_lossk �>f.��        )��P	�/Q�A�*

	conv_lossϒ�>�9�        )��P	��R�A�*

	conv_loss�!�>�.��        )��P	m�R�A�*

	conv_loss=�>�Q��        )��P	�+S�A�*

	conv_losso!�>��J        )��P	�_S�A�*

	conv_loss$��>	Qk        )��P	7�S�A�*

	conv_loss���>H�        )��P	W�S�A�*

	conv_loss�Ѫ>�U�Z        )��P	/T�A�*

	conv_loss�w�>�&�        )��P	V4T�A�*

	conv_loss��>S̠J        )��P	�nT�A�*

	conv_lossd�>H�+Y        )��P	��T�A�*

	conv_loss���>h���        )��P	��T�A�*

	conv_loss���>k͸�        )��P	U�A�*

	conv_loss�`�>�*�m        )��P	:DU�A�*

	conv_loss�B�>�#��        )��P	�vU�A�*

	conv_loss�>�>i%�=        )��P	>�U�A�*

	conv_loss�E�>_n�        )��P	��U�A�*

	conv_lossT��>��<        )��P	�V�A�*

	conv_loss�&�>jIg        )��P	�NV�A�*

	conv_lossbN�>:`_        )��P	R�V�A�*

	conv_lossx�>Rpx�        )��P	�V�A�*

	conv_loss��>w�d        )��P	��V�A�*

	conv_lossWQ�>9�*�        )��P	�$W�A�*

	conv_loss˨>��q*        )��P	XW�A�*

	conv_lossO/�>�2G        )��P	,�W�A�*

	conv_loss���>�.Y        )��P	��W�A�*

	conv_loss��>͠��        )��P	��W�A�*

	conv_loss ��>�`m�        )��P	� X�A�*

	conv_lossYw�>,�h        )��P	�SX�A�*

	conv_loss���>��        )��P	�X�A�*

	conv_loss�,�>%l��        )��P	��X�A�*

	conv_loss}ʧ>�k�        )��P	��X�A�*

	conv_lossw��>|�g�        )��P	Y�A�*

	conv_lossV�>mJ>�        )��P	�NY�A�*

	conv_loss5�>cl��        )��P	 �Y�A�*

	conv_loss���>��
        )��P	��Y�A�*

	conv_loss�<�>��T�        )��P	a�Y�A�*

	conv_lossĖ�>���        )��P	�Z�A�*

	conv_loss�6�><�#        )��P	 OZ�A�*

	conv_loss@ǧ>Ci�        )��P	x�Z�A�*

	conv_loss�)�>�2o�        )��P	G�Z�A�*

	conv_loss�ۦ>�g��        )��P	f�Z�A�*

	conv_loss���>H�_m        )��P	�[�A�*

	conv_loss8��><:�        )��P	M[�A�*

	conv_loss���>�lt�        )��P	4�[�A�*

	conv_loss�z�>g*�        )��P	��[�A�*

	conv_loss��>�{�"        )��P	��[�A�*

	conv_lossJ��>DH4�        )��P	�\�A�*

	conv_loss]��>V�4        )��P	5N\�A�*

	conv_loss�ͦ>�\Z�        )��P	�\�A�*

	conv_loss��>9E�        )��P	�\�A�*

	conv_loss�>�rh        )��P	�\�A�*

	conv_lossA��>�S�        )��P	a-]�A�*

	conv_lossLΧ>�:�d        )��P	�_]�A�*

	conv_loss1l�>	�        )��P	��]�A�*

	conv_loss]��>���(        )��P	��]�A�*

	conv_lossĨ>8��        )��P	�^�A�*

	conv_lossʇ�>���        )��P	<7^�A�*

	conv_loss�E�>&�g�        )��P	j^�A�*

	conv_lossh��>9��        )��P	�^�A�*

	conv_loss�b�>�i��        )��P	��^�A�*

	conv_loss2�>��x�        )��P	�_�A�*

	conv_lossg��>l�        )��P	�D_�A�*

	conv_loss���>��J        )��P	w_�A�*

	conv_loss�4�>s�jR        )��P	�_�A�*

	conv_loss���><M�        )��P	Y�_�A�*

	conv_loss���>��6        )��P	�`�A�*

	conv_loss�>Y�        )��P	�P`�A�*

	conv_loss�ȩ>8J[�        )��P	�`�A�*

	conv_loss�m�>I CR        )��P	�`�A�*

	conv_loss:�>N_�c        )��P	?�`�A�*

	conv_loss��>��E�        )��P	�%a�A�*

	conv_loss%�>#rZ�        )��P	Za�A�*

	conv_loss��>@0�G        )��P	�a�A�*

	conv_loss6*�>�]�        )��P	��a�A�*

	conv_loss!��>�qa        )��P	�a�A�*

	conv_loss�T�>����        )��P	�'b�A�*

	conv_losshA�>�>6�        )��P	6[b�A�*

	conv_loss�"�>�,�        )��P	��b�A�*

	conv_loss�5�>6�.        )��P	8�b�A�*

	conv_loss�d�>��A        )��P	��b�A�*

	conv_lossKЧ>��M        )��P	D*c�A�*

	conv_loss�å>�4��        )��P	1]c�A�*

	conv_lossWg�>��I�        )��P	��c�A�*

	conv_loss��>��        )��P	��c�A�*

	conv_loss���>�3�S        )��P	d�c�A�*

	conv_lossQ̦>�M1        )��P	X,d�A�*

	conv_loss8L�>��        )��P	_d�A�*

	conv_loss�4�>���d        )��P	>�d�A�*

	conv_loss4ܧ>Iq*�        )��P	��d�A�*

	conv_loss1�>|��        )��P	�d�A�*

	conv_lossG1�>�Y��        )��P	,*e�A�*

	conv_loss6B�>a�	�        )��P	`e�A�*

	conv_loss�S�>Vq�        )��P	r�e�A�*

	conv_loss�h�>j��=        )��P	��e�A�*

	conv_loss���>��g�        )��P	)�e�A�*

	conv_lossg��>�R��        )��P	P.f�A�*

	conv_loss=H�>yJ        )��P	�^f�A�*

	conv_lossU�>�Q�        )��P	�f�A�*

	conv_lossuƧ>O~        )��P	��f�A�*

	conv_lossZ`�>�	�        )��P	��f�A�*

	conv_loss�I�>�xӒ        )��P	�Dg�A�*

	conv_lossXҥ>��}        )��P	=wg�A�*

	conv_lossoƦ>*Moc        )��P	��g�A�*

	conv_loss��>Rg^:        )��P	$�g�A�*

	conv_loss��>
M-�        )��P	�h�A�*

	conv_loss��>����        )��P	�Dh�A�*

	conv_lossz��>M+��        )��P	�h�A�*

	conv_lossdΨ>r��        )��P	ձh�A�*

	conv_loss[�>��Y        )��P	��h�A�*

	conv_loss&F�>���r        )��P	$i�A�*

	conv_loss[�>��t\        )��P	gi�A�*

	conv_loss��>NH�        )��P	ܙi�A�*

	conv_loss��>����        )��P	(�i�A�*

	conv_loss�A�>����        )��P	�j�A�*

	conv_loss��>�<        )��P	�=j�A�*

	conv_loss;��>,�        )��P	�rj�A�*

	conv_lossI��>s��        )��P	��j�A�*

	conv_lossx=�>$z��        )��P	V�j�A�*

	conv_loss�ۧ>aטp        )��P	�k�A�*

	conv_loss2ܦ>4 ��        )��P	�Fk�A�*

	conv_loss�Ħ>c��r        )��P	6{k�A�*

	conv_loss���>���\        )��P	��k�A�*

	conv_lossd��>Dڠ        )��P	D�k�A�*

	conv_losssK�>�k�g        )��P	�l�A�*

	conv_loss�Ѧ>î�        )��P	�Hl�A�*

	conv_loss-�>�ۣ^        )��P	�|l�A�*

	conv_loss��>��L/        )��P	K�l�A�*

	conv_loss%¨>�&72        )��P	�l�A�*

	conv_loss�	�>��        )��P	_m�A�*

	conv_lossiæ>5�D�        )��P	(Em�A�*

	conv_lossl �>�BG         )��P	Exm�A�*

	conv_lossa�>�7a        )��P	G�m�A�*

	conv_loss�|�>��*�        )��P	N�m�A�*

	conv_losszȨ>���        )��P	^n�A�*

	conv_loss��>�xe�        )��P	Cn�A�*

	conv_loss�~�>8        )��P	�un�A�*

	conv_loss*��>y�u        )��P	�n�A�*

	conv_loss�F�>�+H�        )��P	��n�A�*

	conv_loss���>]��        )��P	1o�A�*

	conv_loss"��>>�b        )��P	aAo�A�*

	conv_loss?.�>����        )��P	ato�A�*

	conv_loss�6�>��        )��P	��o�A�*

	conv_loss�V�>����        )��P	N�o�A�*

	conv_loss�	�>&��        )��P	?p�A�*

	conv_loss��>����        )��P	#Ap�A�*

	conv_lossD��>�;TV        )��P	ctp�A�*

	conv_loss�>�Ф�        )��P	��p�A�*

	conv_loss갦>ҽH�        )��P	��p�A�*

	conv_loss>s�>�%        )��P	8q�A�*

	conv_losseŦ>��8        )��P	�Fq�A�*

	conv_loss��>3Y        )��P	ƍq�A�*

	conv_lossf�>�f�n        )��P	l�q�A�*

	conv_lossn��>"K�b        )��P	��q�A�*

	conv_loss�ѥ>o{M        )��P	�&r�A�*

	conv_loss�§>~̗�        )��P	�[r�A�*

	conv_loss�s�>�x        )��P	��r�A�*

	conv_lossp��>҃?�        )��P	��r�A�*

	conv_loss*��>X�Ƙ        )��P	�r�A�*

	conv_loss�#�>ݽ�        )��P	4s�A�*

	conv_loss�Y�>.EK        )��P	�is�A�*

	conv_loss���>��z        )��P	��s�A�*

	conv_loss�>��m�        )��P	P�s�A�*

	conv_loss=��>���        )��P	�
t�A�*

	conv_loss���>SC��        )��P	�Dt�A�*

	conv_lossE7�>ҹ�        )��P	~t�A�*

	conv_loss뵧>��s        )��P	.�t�A�*

	conv_loss�e�>��,o        )��P	Q�t�A�*

	conv_lossF�>��        )��P	 u�A�*

	conv_lossD��>kb�        )��P	kTu�A�*

	conv_loss�ܣ>�:�        )��P	!�u�A�*

	conv_lossZW�>�=T        )��P	.�u�A�*

	conv_loss<s�>c��        )��P	��u�A�*

	conv_loss�O�>lJ�9        )��P	�%v�A�*

	conv_lossY!�>�%        )��P	fWv�A�*

	conv_loss�]�>X�^)        )��P	Y�v�A�*

	conv_loss+Ĩ><��        )��P	��v�A�*

	conv_loss���>���u        )��P	Dw�A�*

	conv_lossa,�>��>�        )��P	�@w�A�*

	conv_loss��>*��        )��P	�tw�A�*

	conv_loss���>i��b        )��P	c�w�A�*

	conv_loss�f�>�r�        )��P	=�w�A�*

	conv_loss��>�1$        )��P	�x�A�*

	conv_loss/��>�.        )��P	:Cx�A�*

	conv_loss �>p%�        )��P	�tx�A�*

	conv_loss�&�>��)�        )��P	C�x�A�*

	conv_lossI�>y�1        )��P	%�x�A�*

	conv_lossn�>��a�        )��P	�y�A�*

	conv_losse��>F�        )��P	.Dy�A�*

	conv_loss��>d`��        )��P	wy�A�*

	conv_loss���>�R�        )��P	��y�A�*

	conv_loss_��>�gy
        )��P	��y�A�*

	conv_lossצ>�*`        )��P	�z�A�*

	conv_loss�O�>M�s�        )��P	aDz�A�*

	conv_lossR�>���        )��P	�wz�A�*

	conv_loss>ѣ>�,�O        )��P	ǫz�A�*

	conv_loss.Ʀ>�ť|        )��P	��z�A�*

	conv_lossK�>l��n        )��P	{�A�*

	conv_loss�p�>��/�        )��P	CE{�A�*

	conv_loss=�>vb �        )��P	[z{�A�*

	conv_loss$Ӥ>�Q�        )��P	֬{�A�*

	conv_loss��>/09O        )��P	�C}�A�*

	conv_loss���>D���        )��P	�v}�A�*

	conv_lossR��>-m-�        )��P	��}�A�*

	conv_loss/��>��:�        )��P	��}�A�*

	conv_loss�1�>���        )��P	�~�A�*

	conv_lossU��>��        )��P	�D~�A�*

	conv_lossu&�>*F�        )��P	�~�A�*

	conv_lossC�>�K�        )��P	ĳ~�A�*

	conv_loss��>]�-        )��P	��~�A�*

	conv_loss�I�>���O        )��P	�$�A�*

	conv_loss�6�>c|        )��P	�Y�A�*

	conv_lossa��>����        )��P	���A�*

	conv_loss�٣>��S        )��P	���A�*

	conv_loss�v�>k�        )��P	?��A�*

	conv_loss�$�>�%��        )��P	�)��A�*

	conv_loss���>�lo�        )��P	2\��A�*

	conv_loss�إ>�U�        )��P	,���A�*

	conv_lossqi�>��)�        )��P	��A�*

	conv_loss쭤>O��        )��P	����A�*

	conv_lossGL�>

P         )��P	o-��A�*

	conv_loss֦>x�y        )��P	�`��A�*

	conv_loss�4�>��RB        )��P	A���A�*

	conv_loss?&�>#M�        )��P	�Á�A�*

	conv_loss4�>p�ɍ        )��P	3��A�*

	conv_lossϼ�>�b�.        )��P	0&��A�*

	conv_loss�c�>� 
Q        )��P	xW��A�*

	conv_lossn��>�B�        )��P	툂�A�*

	conv_loss/}�>�oъ        )��P	����A�*

	conv_loss[@�>~�        )��P	���A�*

	conv_losse�>ҝFi        )��P	 ��A�*

	conv_loss���>����        )��P	'Q��A�*

	conv_lossj��>���C        )��P	\���A�*

	conv_loss�D�> �I�        )��P	O���A�*

	conv_loss�
�>(�H&        )��P	��A�*

	conv_loss쟥>��        )��P	���A�*

	conv_loss���>\r̢        )��P	L��A�*

	conv_loss��>˩":        )��P	|��A�*

	conv_loss���>�bCu        )��P	s���A�*

	conv_loss�Ҥ>|��        )��P	�݄�A�*

	conv_loss���>�if        )��P	F��A�*

	conv_loss�˥>e6��        )��P	xD��A�*

	conv_lossEΤ>�	�        )��P	�u��A�*

	conv_loss3m�>���        )��P	0���A�*

	conv_loss���>p;��        )��P	�ׅ�A�*

	conv_lossO��>�E3        )��P	l
��A�*

	conv_loss�P�>�q�        )��P	�;��A�*

	conv_losst��>��        )��P	(l��A�*

	conv_loss��>��c�        )��P	ᝆ�A�*

	conv_lossSq�>cX�        )��P	.І�A�*

	conv_lossT`�>ѽE�        )��P	���A�*

	conv_loss�!�>��=0        )��P	�E��A�*

	conv_lossί�>F�ߑ        )��P	py��A�*

	conv_loss��>��B�        )��P	q���A�*

	conv_loss^��>�,C        )��P	��A�*

	conv_loss��>��X�        )��P	���A�*

	conv_loss�d�>y(FE        )��P	�N��A�*

	conv_loss��>�p��        )��P	����A�*

	conv_loss:�>���        )��P	����A�*

	conv_loss��>��C        )��P	����A�*

	conv_loss�t�>(�O        )��P	L-��A�*

	conv_loss���>~Z;�        )��P	qf��A�*

	conv_loss���>�ZH�        )��P	W���A�*

	conv_lossG��>ڸ�Q        )��P	-̉�A�*

	conv_lossL�>Fn�T        )��P	-���A�*

	conv_lossX9�>!�c        )��P	A/��A�*

	conv_lossE�>2.=�        )��P	L`��A�*

	conv_loss��>S��n        )��P	����A�*

	conv_lossYg�>�r~�        )��P	����A�*

	conv_loss�4�>zM*�        )��P	���A�*

	conv_lossO=�> ��e        )��P	0��A�*

	conv_loss��>;|��        )��P	�e��A�*

	conv_loss��>*�;F        )��P	����A�*

	conv_loss��>����        )��P	�ȋ�A�*

	conv_loss,*�>E�
�        )��P	����A�*

	conv_loss�"�>�g�        )��P	�3��A�*

	conv_loss�B�>��Ho        )��P	�e��A�*

	conv_losstO�>1V��        )��P	r���A�*

	conv_lossb
�>���@        )��P	1ƌ�A�*

	conv_loss�ԧ>�L�        )��P	���A�*

	conv_lossW��>Gu`�        )��P	�*��A�*

	conv_lossm�>w���        )��P	\Z��A�*

	conv_lossˤ>��Y�        )��P	m���A�*

	conv_lossŁ�>p���        )��P	���A�*

	conv_lossO��>�t�        )��P	���A�*

	conv_loss�q�>��Q�        )��P	���A�*

	conv_lossh��>���        )��P	�N��A�*

	conv_loss�ʥ> G�)        )��P	���A�*

	conv_lossy��>���        )��P	����A�*

	conv_loss:��>�r        )��P	4ߎ�A�*

	conv_loss���>��"
        )��P	���A�*

	conv_lossآ>�I5        )��P	 <��A�*

	conv_loss���>�6�        )��P	1o��A�*

	conv_loss�ˤ>U�i        )��P	a���A�*

	conv_loss�k�>�V��        )��P	7͏�A�*

	conv_loss^��>}*a�        )��P	����A�*

	conv_lossN�>�ʇ�        )��P	d,��A�*

	conv_loss�ܥ>*]wj        )��P	E[��A�*

	conv_loss~h�>�j�        )��P	}���A�*

	conv_lossӯ�>ad�i        )��P	����A�*

	conv_loss%@�>��x        )��P	`��A�*

	conv_lossض�>�Φ        )��P	�*��A�*

	conv_loss��>gߙ%        )��P	�Z��A�*

	conv_loss�;�>���        )��P	���A�*

	conv_lossQ�>{v z        )��P	�ϑ�A�*

	conv_loss�1�>
���        )��P	 ��A�*

	conv_loss�t�>#�y�        )��P	s/��A�*

	conv_lossz�>����        )��P	�_��A�*

	conv_loss+��>��5�        )��P	J���A�*

	conv_losso��>��)�        )��P	Gܒ�A�*

	conv_loss���>6[��        )��P	:��A�*

	conv_lossyK�>��d        )��P	�:��A�*

	conv_lossp,�>;���        )��P	i��A�*

	conv_loss��>f*k        )��P	&���A�*

	conv_loss�&�>�A��        )��P	s̓�A�*

	conv_lossYr�>ɪ��        )��P	+���A�*

	conv_loss<��>ΛnA        )��P	�,��A�*

	conv_loss���>�=�3        )��P	�[��A�*

	conv_loss��>��        )��P	8���A�*

	conv_loss��>w�e�        )��P	����A�*

	conv_loss���>�_�P        )��P	<��A�*

	conv_loss�c�>�CX        )��P	 ��A�*

	conv_loss�]�>qa܆        )��P	�S��A�*

	conv_lossb�>�s[        )��P	���A�*

	conv_loss=(�>���        )��P	l���A�*

	conv_lossA��>�	ő        )��P	$��A�*

	conv_loss0��>�"wU        )��P	���A�*

	conv_loss5x�>���        )��P	:D��A�*

	conv_loss�F�>&�6�        )��P	r��A�*

	conv_loss�$�>B��        )��P	����A�*

	conv_loss�>+        )��P	і�A�*

	conv_loss���>�b�        )��P	� ��A�*

	conv_loss��>t6*d        )��P	v0��A�*

	conv_loss*�>�9�        )��P	�`��A�*

	conv_loss�Q�>�L^j        )��P	ޏ��A�*

	conv_loss��>[ﴣ        )��P	����A�*

	conv_lossT֤>���m        )��P	���A�*

	conv_lossx��>,��A        )��P	���A�*

	conv_loss�ؤ>KtA`        )��P	�I��A�*

	conv_lossvp�>}���        )��P	2x��A�*

	conv_loss�y�>�*:        )��P	m���A�*

	conv_lossX�>���        )��P	RҘ�A�*

	conv_lossԣ>y�e�        )��P	� ��A�*

	conv_lossC�>%e        )��P	�3��A�*

	conv_loss5�>D	K        )��P	;a��A�*

	conv_loss觤>dq�        )��P	����A�*

	conv_loss�(�>��        )��P	п��A�*

	conv_loss�^�>�p'        )��P	���A�*

	conv_loss'ܣ>�7�        )��P	"��A�*

	conv_loss~O�>�z�A        )��P	�P��A�*

	conv_loss�j�>.�N�        )��P	���A�*

	conv_lossa��>]�a        )��P	!Ú�A�*

	conv_loss�?�>�:        )��P	����A�*

	conv_loss�O�>?�        )��P	�(��A�*

	conv_lossr �>��ew        )��P	�Y��A�*

	conv_lossG"�>�8�        )��P	���A�*

	conv_lossc'�>���        )��P	����A�*

	conv_lossգ>�2�        )��P	+��A�*

	conv_loss%6�>��	B        )��P	���A�*

	conv_loss�n�>v���        )��P	"V��A�*

	conv_loss4}�>٢ި        )��P	&���A�*

	conv_loss���>�l        )��P	(˜�A�*

	conv_loss���>�1�r        )��P	���A�*

	conv_loss�Z�> ���        )��P	5��A�*

	conv_lossڥ>e�2�        )��P	�e��A�*

	conv_loss��>�Ӌ        )��P	ꓝ�A�*

	conv_lossRϣ>zf��        )��P	t���A�*

	conv_loss��> =�f        )��P	k��A�*

	conv_loss��>Y�=b        )��P	� ��A�*

	conv_loss&�>5�uV        )��P	ZP��A�*

	conv_loss�>S��X        )��P	f���A�*

	conv_loss���>N��        )��P	����A�*

	conv_lossӵ�>^;D�        )��P	�ݞ�A�*

	conv_lossX �>����        )��P	���A�*

	conv_loss�.�>[���        )��P	�F��A�*

	conv_loss�{�>�z�        )��P	�y��A�*

	conv_lossO�>���        )��P	u���A�*

	conv_lossg٦>����        )��P	&ڟ�A�*

	conv_loss��>y�ʝ        )��P	���A�*

	conv_loss�+�>�\�        )��P	�<��A�*

	conv_lossﻣ>!-��        )��P	m��A�*

	conv_lossȢ�>�A�        )��P	;���A�*

	conv_lossN�>��|        )��P	�ʠ�A�*

	conv_loss�1�>^Cu�        )��P	m���A�*

	conv_loss鋥>�J��        )��P	+��A�*

	conv_loss���>�|j�        )��P	Y��A�*

	conv_lossD��>x�y�        )��P	C���A�*

	conv_loss�	�>�h        )��P	ܺ��A�*

	conv_loss��>�oA�        )��P	���A�*

	conv_lossE�>zP        )��P	@��A�*

	conv_loss��>+Գ        )��P	H��A�*

	conv_loss�ʣ>o�b�        )��P	�v��A�*

	conv_loss)@�>�;BZ        )��P	���A�*

	conv_loss�)�>(np2        )��P	�Ԣ�A�*

	conv_loss�ʤ>���        )��P	���A�*

	conv_loss揤>#��        )��P	�2��A�*

	conv_loss闤>H4o,        )��P	�`��A�*

	conv_lossQ��>���i        )��P	����A�*

	conv_loss~��>�?J        )��P	����A�*

	conv_loss���>�;��        )��P	���A�*

	conv_losss+�>�Ͷ�        )��P	/I��A�*

	conv_loss2�>���B        )��P	zة�A�*

	conv_loss6^�>��t7        )��P	 ��A�*

	conv_loss�Ǥ>`��        )��P	C7��A�*

	conv_loss;"�>�7�        )��P	�e��A�*

	conv_loss1B�>x���        )��P	왪�A�*

	conv_lossf٦>��o        )��P	
Ȫ�A�*

	conv_loss)��>xM�        )��P	����A�*

	conv_loss�Ҥ>��r        )��P	k)��A�*

	conv_loss"=�>>睽        )��P	�[��A�*

	conv_loss9��>�E��        )��P	0���A�*

	conv_loss�>(�&        )��P	�ū�A�*

	conv_lossa��>��UQ        )��P	����A�*

	conv_loss㿣>� es        )��P	--��A�*

	conv_loss%�>]r�        )��P	9i��A�*

	conv_lossŖ�>�?�h        )��P	y���A�*

	conv_loss�K�>�sB        )��P	fǬ�A�*

	conv_loss��>�<ɚ        )��P	-���A�*

	conv_lossG�>s�Q5        )��P	e(��A�*

	conv_lossꍣ>�M�        )��P	W��A�*

	conv_lossc*�>�F        )��P	����A�*

	conv_loss� �>��!        )��P	,���A�*

	conv_loss��>��
�        )��P	���A�*

	conv_loss���>�1;        )��P	���A�*

	conv_loss���>W�U        )��P	�G��A�*

	conv_loss��>b�ʯ        )��P	���A�*

	conv_loss�ף> ��A        )��P	\���A�*

	conv_loss"<�>�@�f        )��P	�߮�A�*

	conv_loss��>�W�F        )��P	���A�*

	conv_loss�آ>w,-(        )��P	A��A�*

	conv_lossou�>S)�        )��P	�o��A�*

	conv_lossɟ�>�'�        )��P	ܞ��A�*

	conv_loss'΢>�M��        )��P	�ί�A�*

	conv_loss���>��5�        )��P	����A�*

	conv_loss
¢>Fy�        )��P	�+��A�*

	conv_loss���>���'        )��P	K]��A�*

	conv_loss���>�        )��P	���A�*

	conv_loss�8�>�� B        )��P	����A�*

	conv_loss��>�Ա�        )��P	[��A�*

	conv_loss�y�>� ��        )��P	=��A�*

	conv_loss���>N(-         )��P	L��A�*

	conv_lossԥ>%��        )��P	W{��A�*

	conv_loss�d�>Ï�        )��P	����A�*

	conv_lossK �>��xz        )��P	Q߱�A�*

	conv_loss��>q/�*        )��P	���A�*

	conv_loss/��>�eh        )��P	�=��A�*

	conv_loss[��>6��        )��P	�n��A�*

	conv_loss���>�K�        )��P	����A�*

	conv_lossS5�>[��        )��P	�β�A�*

	conv_lossmJ�>��oe        )��P	����A�*

	conv_loss|�>��ی        )��P	�/��A�*

	conv_loss@k�>S�YW        )��P	p��A�*

	conv_lossU�>:{q8        )��P	����A�*

	conv_loss)��>#��        )��P	ҳ�A�*

	conv_loss���>� *&        )��P	� ��A�*

	conv_loss��>�AOS        )��P		3��A�*

	conv_loss)��>�98�        )��P	�`��A�*

	conv_loss�X�>B	2
        )��P	H���A�*

	conv_loss8��>xF�m        )��P	�Ĵ�A�*

	conv_lossu͢>���
        )��P	d���A�*

	conv_loss���>y1.        )��P	>(��A�*

	conv_loss	S�>�D�        )��P	]Z��A�*

	conv_lossj١>8��w        )��P	����A�*

	conv_loss+j�>X��        )��P	�˵�A�*

	conv_loss�:�>����        )��P	��A�*

	conv_loss���>���        )��P	H4��A�*

	conv_lossٴ�>i*        )��P	&f��A�*

	conv_loss��>�?�-        )��P	|���A�*

	conv_loss�ʟ>>�JC        )��P	�ʶ�A�*

	conv_loss��>�C�        )��P	����A�*

	conv_lossd�>�}�        )��P	�0��A�*

	conv_lossX�>n��        )��P	�`��A�*

	conv_lossM'�>:��        )��P	t���A�*

	conv_lossa֣>��        )��P	���A�*

	conv_loss��>;H�G        )��P	a��A�*

	conv_loss���>�c�z        )��P	�!��A�*

	conv_loss�4�><Dl[        )��P	R��A�*

	conv_loss���>��~        )��P	��A�*

	conv_loss�>�:Uv        )��P	갸�A�*

	conv_loss��>`o�Y        )��P	��A�*

	conv_loss'�>���        )��P	���A�*

	conv_lossZA�>���}        )��P	FG��A�*

	conv_lossI!�>��:N        )��P	)x��A�*

	conv_loss:բ>��w\        )��P	w���A�*

	conv_loss�1�>9I��        )��P	J۹�A�*

	conv_loss� �>��n�        )��P	��A�*

	conv_loss�%�>�W-�        )��P	z>��A�*

	conv_loss��>,u5        )��P	�p��A�*

	conv_loss���>j��        )��P	����A�*

	conv_loss��>��Y        )��P	�׺�A�*

	conv_losss�>�_A�        )��P	���A�*

	conv_loss�<�>Ӹ*�        )��P	�>��A�*

	conv_loss3Ǣ>�x(�        )��P	yr��A�*

	conv_loss�h�>�o��        )��P	e���A�*

	conv_lossY��>�G�        )��P	���A�*

	conv_loss6�>���        )��P	���A�*

	conv_loss���>�]e5        )��P	uK��A� *

	conv_lossn��>ǿ�p        )��P	N��A� *

	conv_loss��>h;1        )��P		���A� *

	conv_lossH��>���        )��P	���A� *

	conv_loss+�>�rs         )��P	v ��A� *

	conv_loss�r�>iu��        )��P	2i��A� *

	conv_loss�<�>�g        )��P	����A� *

	conv_loss�s�>T��        )��P	�н�A� *

	conv_loss�K�>���        )��P	���A� *

	conv_loss�Ǣ>G�        )��P	�;��A� *

	conv_loss��>����        )��P	Uq��A� *

	conv_loss�~�>f࢛        )��P	x���A� *

	conv_loss�/�>���        )��P	���A� *

	conv_lossBX�>ݧ��        )��P	��A� *

	conv_loss�0�>���F        )��P	�J��A� *

	conv_losse��>�1�R        )��P	����A� *

	conv_loss�C�>�+�        )��P	����A� *

	conv_lossi�>���G        )��P	����A� *

	conv_loss�C�>	�?i        )��P	�3��A� *

	conv_lossM��>
�N�        )��P	�i��A� *

	conv_lossq��>�#        )��P	#���A� *

	conv_loss���>�l�I        )��P	/���A� *

	conv_loss�[�>�lX�        )��P	�
��A� *

	conv_lossZѢ>K[�|        )��P	#A��A� *

	conv_loss=E�>J�w        )��P	�u��A� *

	conv_lossH�>��|�        )��P	����A� *

	conv_loss\�>A��E        )��P	����A� *

	conv_loss�6�>��V        )��P	���A� *

	conv_loss�>!�;\        )��P	�X��A� *

	conv_lossp��>q�!�        )��P	R���A� *

	conv_loss�2�>AVK^        )��P	����A� *

	conv_loss��>~�        )��P	e��A� *

	conv_loss��>C�u        )��P	�?��A� *

	conv_loss��>��$        )��P	�r��A� *

	conv_losssl�>_3��        )��P	Ŧ��A� *

	conv_loss'��>v�        )��P	����A� *

	conv_lossU�>��        )��P	C��A� *

	conv_loss��>��j�        )��P	dB��A� *

	conv_loss��>����        )��P	�v��A� *

	conv_loss�؟>�W        )��P	m���A� *

	conv_loss"�>�(�q        )��P	e���A� *

	conv_lossO[�>4`�/        )��P	���A� *

	conv_losse.�>?V��        )��P	�J��A� *

	conv_losst�>ȡt        )��P	b��A� *

	conv_loss�+�>R�{�        )��P	���A� *

	conv_lossSQ�>3�E        )��P	����A� *

	conv_lossN�>�0�        )��P	���A� *

	conv_loss���>���A        )��P	�S��A� *

	conv_loss��>���        )��P	����A� *

	conv_losse��>+&ޝ        )��P	����A� *

	conv_lossd�>f���        )��P	����A� *

	conv_lossp�>��        )��P	'��A� *

	conv_loss	Π>'�l        )��P	�Z��A� *

	conv_lossQ��>h�g�        )��P	���A� *

	conv_lossJ�>�g�        )��P	���A� *

	conv_loss�Z�>�L        )��P	J	��A� *

	conv_loss B�>X��        )��P	$>��A� *

	conv_lossuY�>0�=        )��P	�r��A� *

	conv_loss��>���        )��P	���A� *

	conv_loss�ڢ>���        )��P	����A� *

	conv_loss�	�>���        )��P	�&��A� *

	conv_lossǟ�>g� 5        )��P	�]��A� *

	conv_lossbe�>��9�        )��P	����A� *

	conv_lossz$�>���        )��P	����A� *

	conv_loss2�>��bU        )��P	<��A� *

	conv_loss�b�>z�l�        )��P	 W��A� *

	conv_loss���>N���        )��P	ъ��A� *

	conv_loss~,�>�0"        )��P	����A� *

	conv_loss�e�>� K        )��P	���A� *

	conv_lossހ�>r���        )��P	7(��A� *

	conv_loss`E�>��        )��P	@[��A� *

	conv_loss�Z�>�M�!        )��P	X���A� *

	conv_loss���>��VQ        )��P	���A� *

	conv_loss�¢>g��        )��P	���A� *

	conv_loss���>d|Z�        )��P	�,��A� *

	conv_lossA��>Ac��        )��P	�e��A� *

	conv_loss�:�>�tah        )��P	H���A� *

	conv_loss)R�>��»        )��P	Y���A� *

	conv_lossd��>$'@�        )��P	O��A� *

	conv_loss	��>�c�        )��P	OE��A� *

	conv_loss0��>Q�
        )��P	V{��A� *

	conv_lossڂ�>LÇ�        )��P	g���A� *

	conv_loss��>�O��        )��P	����A� *

	conv_loss$	�>o\        )��P	l��A� *

	conv_losse��>�G��        )��P	�L��A� *

	conv_losso0�>��nK        )��P	c���A� *

	conv_loss��>�        )��P	���A� *

	conv_loss�G�>O~��        )��P	o���A� *

	conv_loss\�>��A�        )��P	P ��A� *

	conv_loss�)�>]1��        )��P	T��A� *

	conv_lossy�>���        )��P	����A� *

	conv_lossZ�>�]t�        )��P	V���A� *

	conv_loss�9�>�6�        )��P	����A� *

	conv_loss���>/�|`        )��P	|&��A� *

	conv_loss|�>k�]P        )��P	#Y��A� *

	conv_loss*��>v7��        )��P	َ��A� *

	conv_loss(�>8"B�        )��P	����A� *

	conv_loss
��>�Fa        )��P	H���A� *

	conv_loss��>�n�        )��P	n,��A� *

	conv_loss븠>��        )��P	�`��A� *

	conv_loss*��>4�B        )��P	����A� *

	conv_loss;�>D�o        )��P	����A� *

	conv_lossuc�>!	#�        )��P	q���A� *

	conv_loss�ԡ>�p-        )��P	�0��A� *

	conv_loss��>��S        )��P	Ze��A� *

	conv_loss���>��cb        )��P	���A� *

	conv_loss��>\~��        )��P	�6��A� *

	conv_lossX��>��        )��P	�k��A� *

	conv_loss�>��2        )��P	���A� *

	conv_loss�F�>E ��        )��P	����A� *

	conv_loss�Ӣ>��|�        )��P	���A� *

	conv_loss�)�>�H�^        )��P	�@��A� *

	conv_loss@٠>�[O:        )��P	[u��A� *

	conv_loss ��>�l��        )��P	����A� *

	conv_loss���>>���        )��P	6���A� *

	conv_lossb͠>��        )��P	{&��A� *

	conv_lossy��>����        )��P	r_��A� *

	conv_losst��>��        )��P	ϖ��A� *

	conv_loss7�>�?��        )��P	����A� *

	conv_loss���>�΢�        )��P	��A� *

	conv_loss*��>,�n�        )��P	�A��A� *

	conv_loss쪟>�L,0        )��P	Cv��A� *

	conv_lossj�>[Ph>        )��P	����A� *

	conv_loss�X�>P<        )��P	G���A� *

	conv_loss��>�|�        )��P	Q��A� *

	conv_loss�w�>!ڢ        )��P	VF��A� *

	conv_loss���>�g        )��P	�z��A� *

	conv_loss�|�>��?p        )��P	���A� *

	conv_loss6�><b�        )��P	����A�!*

	conv_loss���>�s        )��P	z��A�!*

	conv_lossپ�>T��        )��P	�K��A�!*

	conv_loss���>���4        )��P	3���A�!*

	conv_lossg�>�PK^        )��P	����A�!*

	conv_loss� >;��        )��P	z���A�!*

	conv_loss���>�O��        )��P	O��A�!*

	conv_loss�[�>D,�        )��P	�T��A�!*

	conv_loss�ȡ>�(:N        )��P	E���A�!*

	conv_loss���>����        )��P	����A�!*

	conv_loss���>�<`�        )��P	���A�!*

	conv_loss9�>JËs        )��P	h&��A�!*

	conv_loss�T�>�B$         )��P	Y��A�!*

	conv_loss�b�>8���        )��P	����A�!*

	conv_loss��>w��(        )��P	&���A�!*

	conv_loss��>��H�        )��P	���A�!*

	conv_loss��>�-J�        )��P	�,��A�!*

	conv_loss�˞>����        )��P	�`��A�!*

	conv_loss�ڟ>��k        )��P	���A�!*

	conv_loss2��>#���        )��P	W���A�!*

	conv_lossbΡ>MD�        )��P	��A�!*

	conv_loss3�>���        )��P	w4��A�!*

	conv_loss�z�>�羼        )��P	Qi��A�!*

	conv_lossC �>�C�.        )��P	����A�!*

	conv_loss<B�>�s�z        )��P	����A�!*

	conv_loss[k�>>��        )��P	���A�!*

	conv_loss�J�>@���        )��P	(;��A�!*

	conv_loss�˞>6�H(        )��P	���A�!*

	conv_loss �>r�+�        )��P	����A�!*

	conv_loss�A�>�xw        )��P	����A�!*

	conv_loss�Р>��f        )��P	&)��A�!*

	conv_loss���>j27�        )��P	b��A�!*

	conv_loss_�>X=�F        )��P	���A�!*

	conv_loss�>C���        )��P	\���A�!*

	conv_loss1�>?��        )��P	9��A�!*

	conv_lossc��>i���        )��P	:7��A�!*

	conv_loss�0�>����        )��P	�m��A�!*

	conv_loss#��>���        )��P	v���A�!*

	conv_loss�ޠ>lN        )��P	k���A�!*

	conv_loss�r�>$�g�        )��P	�)��A�!*

	conv_lossaơ>/�ź        )��P	�]��A�!*

	conv_loss�>��?E        )��P	ʒ��A�!*

	conv_loss�7�>5{�         )��P	����A�!*

	conv_loss�Ҟ>(���        )��P	���A�!*

	conv_loss=��>8ar        )��P	38��A�!*

	conv_loss�p�>�\�p        )��P	�l��A�!*

	conv_lossD��>��m�        )��P	ԡ��A�!*

	conv_loss3��>0���        )��P	����A�!*

	conv_losso�>G��d        )��P	7
��A�!*

	conv_loss ��>�ee        )��P	�?��A�!*

	conv_loss:�>:�$g        )��P	t��A�!*

	conv_loss��>�ʹ        )��P	����A�!*

	conv_loss��>����        )��P	 ���A�!*

	conv_loss�c�>��}�        )��P	���A�!*

	conv_loss͛�>U��7        )��P	G��A�!*

	conv_loss���>�^b�        )��P	�|��A�!*

	conv_loss�>�>+2�        )��P	����A�!*

	conv_loss��>��g        )��P	
���A�!*

	conv_loss�y�>3��g        )��P	���A�!*

	conv_loss//�>�0��        )��P	/N��A�!*

	conv_lossS9�>|��'        )��P	����A�!*

	conv_loss>��>C��-        )��P	7���A�!*

	conv_loss]��>��.�        )��P	���A�!*

	conv_lossk�>����        )��P	���A�!*

	conv_lossQ��>7��!        )��P	�U��A�!*

	conv_loss���>��T)        )��P	݈��A�!*

	conv_lossmg�>��N        )��P	����A�!*

	conv_loss8!�>�o
d        )��P	����A�!*

	conv_loss�>?�@        )��P	*��A�!*

	conv_loss���>EKh�        )��P	�^��A�!*

	conv_loss�>�[E@        )��P	֓��A�!*

	conv_loss���>-0�        )��P	����A�!*

	conv_lossxѝ>��         )��P	l���A�!*

	conv_lossz��>Z�v        )��P	�?��A�!*

	conv_lossF��>���a        )��P	�y��A�!*

	conv_lossnP�>��        )��P	���A�!*

	conv_loss���>�]        )��P	O���A�!*

	conv_loss��>B��2        )��P	�.��A�!*

	conv_loss�t�>^        )��P	�b��A�!*

	conv_loss��>���        )��P	���A�!*

	conv_loss�ʟ>X�[�        )��P	d���A�!*

	conv_loss�Q�>��W        )��P	U��A�!*

	conv_lossc�>��-�        )��P	qB��A�!*

	conv_lossT�>9x        )��P	�u��A�!*

	conv_loss��>�!�        )��P	w���A�!*

	conv_losse'�>%��m        )��P	.���A�!*

	conv_loss��>��$�        )��P	(.��A�!*

	conv_loss��>���        )��P	be��A�!*

	conv_loss��>I�V        )��P	ߙ��A�!*

	conv_lossw��>n�,�        )��P	E���A�!*

	conv_loss2�>.�Ek        )��P	���A�!*

	conv_losst��>���        )��P	k<��A�!*

	conv_loss�7�>;�;        )��P	Jq��A�!*

	conv_loss��>���]        )��P	����A�!*

	conv_loss�(�>����        )��P	3���A�!*

	conv_lossMٞ>{�        )��P	���A�!*

	conv_loss�r�>1H��        )��P	}M��A�!*

	conv_loss���>5*        )��P	����A�!*

	conv_loss:��>�Jq�        )��P	����A�!*

	conv_loss�t�>�Ҥ!        )��P	���A�!*

	conv_loss3��>%\g0        )��P	O#��A�!*

	conv_loss�S�>"�.        )��P	�W��A�!*

	conv_loss�>�FE        )��P	T���A�!*

	conv_lossnʝ>��߻        )��P	V���A�!*

	conv_lossDx�>~,��        )��P	���A�!*

	conv_loss���>a�ܽ        )��P	�(��A�!*

	conv_losseA�>2;        )��P	^��A�!*

	conv_loss�%�>ͫ��        )��P	%���A�!*

	conv_lossd�>Db�        )��P	����A�!*

	conv_loss?Π>��        )��P	����A�!*

	conv_loss�>89�        )��P	�5��A�!*

	conv_loss��>�r�        )��P	�h��A�!*

	conv_loss��>�l�        )��P	���A�!*

	conv_lossZɟ>�
�        )��P	���A�!*

	conv_loss`�>�*H\        )��P	p��A�!*

	conv_loss�k�>����        )��P	`7��A�!*

	conv_loss�n�>ɯ'�        )��P	Lj��A�!*

	conv_loss
��>�t$        )��P	P���A�!*

	conv_loss0�>�        )��P	����A�!*

	conv_loss��>���        )��P	2
��A�!*

	conv_lossK۟>�KV        )��P	�>��A�!*

	conv_loss�Ӟ>�z�f        )��P	(t��A�!*

	conv_lossN>�>�n��        )��P	۩��A�!*

	conv_loss<Ƞ>���        )��P	����A�!*

	conv_loss���>�iJ�        )��P	���A�!*

	conv_losswy�>�kض        )��P	J��A�!*

	conv_loss�s�>��        )��P	<}��A�!*

	conv_loss��>�;*^        )��P	����A�!*

	conv_loss��>f8��        )��P	����A�"*

	conv_loss��>.�6@        )��P	�2��A�"*

	conv_loss6=�>�e        )��P	�h��A�"*

	conv_loss�ϟ>#��        )��P	2���A�"*

	conv_loss��>����        )��P	����A�"*

	conv_loss��>��;        )��P	��A�"*

	conv_loss��>�i~�        )��P	qH��A�"*

	conv_loss"a�>l��        )��P	����A�"*

	conv_losse��>�� �        )��P	t���A�"*

	conv_loss���>c���        )��P	���A�"*

	conv_loss��>K�e�        )��P	"'��A�"*

	conv_loss�ɟ>��Ĕ        )��P	Z��A�"*

	conv_loss B�>\0@/        )��P	٘��A�"*

	conv_losst$�>���        )��P	N���A�"*

	conv_lossTc�>>�+@        )��P	Y���A�"*

	conv_lossϒ�>��        )��P	�5��A�"*

	conv_loss���>���        )��P	n��A�"*

	conv_loss���>�(��        )��P	`���A�"*

	conv_loss�i�>�\�        )��P	����A�"*

	conv_loss��>s^        )��P	;��A�"*

	conv_loss;��>#�.�        )��P	K��A�"*

	conv_loss�ӟ>4r$/        )��P	G��A�"*

	conv_lossf��>�:�        )��P	S���A�"*

	conv_loss�d�>t�        )��P	����A�"*

	conv_loss��>�!]E        )��P	���A�"*

	conv_loss�>:�rC        )��P	7S��A�"*

	conv_loss�@�>#��        )��P	`���A�"*

	conv_loss�!�>�8��        )��P	[���A�"*

	conv_loss�>^_K?        )��P	V���A�"*

	conv_loss�j�>�ڶ2        )��P	c#��A�"*

	conv_lossؚ�>vMHd        )��P	X��A�"*

	conv_lossr��>��B�        )��P	+���A�"*

	conv_loss�r�>t��        )��P	T���A�"*

	conv_loss��>:X�        )��P	<���A�"*

	conv_loss�)�>�q!        )��P	+��A�"*

	conv_lossb�>��T�        )��P	�_��A�"*

	conv_losso��>��L�        )��P	���A�"*

	conv_loss�*�>eOX%        )��P	����A�"*

	conv_loss�Q�>t/��        )��P	(���A�"*

	conv_lossY/�>�:|)        )��P	J0��A�"*

	conv_loss��>K_h        )��P	�f��A�"*

	conv_loss�;�>��        )��P	_���A�"*

	conv_lossdF�>��@        )��P	����A�"*

	conv_lossC�>_uk�        )��P	;��A�"*

	conv_loss�7�>���        )��P	�9��A�"*

	conv_lossN��>���O        )��P	!l��A�"*

	conv_loss��>�P        )��P	����A�"*

	conv_loss�۞>if7�        )��P	����A�"*

	conv_loss��>hXcH        )��P	e��A�"*

	conv_loss�ݙ>gp�t        )��P	����A�"*

	conv_lossc(�>wFB        )��P	p���A�"*

	conv_loss8��>���l        )��P	. ����A�"*

	conv_loss.p�>�8�        )��P	�: ����A�"*

	conv_loss�?�>��-e        )��P	�l ����A�"*

	conv_loss�r�>�G i        )��P	*� ����A�"*

	conv_loss!�>Nhۛ        )��P	C� ����A�"*

	conv_loss{��>��2�        )��P	����A�"*

	conv_loss��>̢��        )��P	�;����A�"*

	conv_loss��>*��        )��P	v����A�"*

	conv_loss=M�>�jG        )��P	Ǭ����A�"*

	conv_loss���>yآ        )��P	1�����A�"*

	conv_loss�o�>�A�1        )��P	$"����A�"*

	conv_loss;ŝ>��        )��P	�R����A�"*

	conv_loss\��>��B{        )��P	M�����A�"*

	conv_loss�ʜ>r�qX        )��P	ڵ����A�"*

	conv_loss���>��+�        )��P	������A�"*

	conv_lossEН>$���        )��P	�����A�"*

	conv_loss�>m��W        )��P	�L����A�"*

	conv_lossX0�>���U        )��P	�}����A�"*

	conv_lossuƛ>��7        )��P	M�����A�"*

	conv_loss��>z��<        )��P	�����A�"*

	conv_loss��>O�'        )��P	[����A�"*

	conv_loss�ݝ>��$�        )��P	�L����A�"*

	conv_loss�E�>٪�        )��P	�����A�"*

	conv_loss�X�>�F�        )��P	������A�"*

	conv_loss�z�> �*        )��P	������A�"*

	conv_loss�>� o�        )��P	-����A�"*

	conv_loss���>��1        )��P	RG����A�"*

	conv_loss^R�>컡�        )��P	�w����A�"*

	conv_loss�.�>����        )��P	ҩ����A�"*

	conv_loss��>��>�        )��P	������A�"*

	conv_loss7!�>��v�        )��P	T����A�"*

	conv_loss��>�x@        )��P	�>����A�"*

	conv_loss���>Xyc        )��P	Mq����A�"*

	conv_loss�z�>�NzL        )��P	�����A�"*

	conv_loss��>�n�|        )��P	;�����A�"*

	conv_loss��>��E�        )��P	�����A�"*

	conv_loss�h�>Ln9|        )��P	B5����A�"*

	conv_lossA��>~ ؃        )��P	6e����A�"*

	conv_loss�g�>$%��        )��P	�����A�"*

	conv_losse�><�7&        )��P	������A�"*

	conv_losst5�>4��        )��P	����A�"*

	conv_loss%R�>�ۃ        )��P	�2����A�"*

	conv_loss���>2�,-        )��P	�c����A�"*

	conv_loss�q�>�h܍        )��P	������A�"*

	conv_loss��>=��S        )��P	������A�"*

	conv_loss��>��E�        )��P	~�����A�"*

	conv_loss�l�>��C�        )��P	z)	����A�"*

	conv_loss�?�>�q�        )��P	�Y	����A�"*

	conv_loss�Q�>�1��        )��P	��	����A�"*

	conv_loss�R�>�YS        )��P	/�	����A�"*

	conv_loss���>�_M        )��P	��	����A�"*

	conv_loss��>���G        )��P	3
����A�"*

	conv_loss�l�>�oy�        )��P	ad
����A�"*

	conv_loss­�>��        )��P	��
����A�"*

	conv_loss���>� ��        )��P	��
����A�"*

	conv_lossOj�>���        )��P	L�
����A�"*

	conv_loss���>[{H        )��P	�5����A�"*

	conv_loss0��>�l6�        )��P	Zq����A�"*

	conv_lossM�>ǥ�+        )��P	������A�"*

	conv_loss��>�ݰ.        )��P	������A�"*

	conv_loss&�>'���        )��P	�����A�"*

	conv_loss���>�?�        )��P	sN����A�"*

	conv_loss:�>�r�~        )��P	�����A�"*

	conv_loss�o�>�
��        )��P	u�����A�"*

	conv_loss���>tң�        )��P	������A�"*

	conv_loss@]�>i���        )��P	`����A�"*

	conv_loss*�><���        )��P	*F����A�"*

	conv_loss��>����        )��P	�}����A�"*

	conv_loss�h�>*bL        )��P	������A�"*

	conv_loss-(�>"�w�        )��P	o�����A�"*

	conv_lossg�>G�7(        )��P	�����A�"*

	conv_loss�s�>>u�X        )��P	Z����A�"*

	conv_lossF9�>�5��        )��P	$�����A�"*

	conv_loss�>��{f        )��P	ξ����A�"*

	conv_lossV<�>Ж�        )��P	������A�"*

	conv_lossj�>_��        )��P	X"����A�"*

	conv_loss��>� �        )��P	T����A�"*

	conv_loss�}�>O�        )��P	M�����A�#*

	conv_loss�B�>���        )��P	{�����A�#*

	conv_lossC��>���Q        )��P	������A�#*

	conv_loss��>���V        )��P	.����A�#*

	conv_loss���>S�HF        )��P	BM����A�#*

	conv_loss/H�>6���        )��P	K����A�#*

	conv_loss֟>w��        )��P	u�����A�#*

	conv_loss��>o�"        )��P	�����A�#*

	conv_loss�4�>&��        )��P	����A�#*

	conv_loss�J�>��b        )��P	SB����A�#*

	conv_loss-�>KB�7        )��P	4t����A�#*

	conv_loss?ڞ>�"�        )��P	H�����A�#*

	conv_loss��>���U        )��P	������A�#*

	conv_loss�>�{j        )��P	Y	����A�#*

	conv_lossᶘ>�|        )��P	�9����A�#*

	conv_loss��>I,�        )��P	j����A�#*

	conv_loss3��>0`�        )��P	^�����A�#*

	conv_loss됞>�c�        )��P	�����A�#*

	conv_loss�>��,        )��P	c�����A�#*

	conv_loss:ؚ>}V|        )��P	�1����A�#*

	conv_loss\1�>a�	5        )��P	�����A�#*

	conv_loss��>��l        )��P	4����A�#*

	conv_loss��>�1�        )��P	�<����A�#*

	conv_loss鿜>%Ď�        )��P	�j����A�#*

	conv_loss."�>�l4        )��P	u�����A�#*

	conv_lossô�>�z�P        )��P	�����A�#*

	conv_lossAʛ>�+        )��P	������A�#*

	conv_loss�P�>M`�        )��P	�/����A�#*

	conv_lossU��>#�~        )��P	g_����A�#*

	conv_loss��>�o�        )��P	J�����A�#*

	conv_lossxS�>�*��        )��P	�����A�#*

	conv_loss�&�>N�s	        )��P	�����A�#*

	conv_lossC��>�!��        )��P	3����A�#*

	conv_loss��>,�;        )��P	�c����A�#*

	conv_loss5�>��ӄ        )��P	������A�#*

	conv_losseq�>� ��        )��P	u�����A�#*

	conv_loss�ə>����        )��P	�	����A�#*

	conv_loss��>�SaB        )��P	z9����A�#*

	conv_lossI��>��        )��P	Zh����A�#*

	conv_loss�ך>��Y�        )��P	z�����A�#*

	conv_loss��>R�K        )��P	������A�#*

	conv_loss�>$��        )��P	������A�#*

	conv_lossd��>�D�n        )��P	�$����A�#*

	conv_loss�.�>���        )��P	�V����A�#*

	conv_lossG�>�2ַ        )��P	˃����A�#*

	conv_loss��>�yk        )��P	�����A�#*

	conv_loss�R�>㵊�        )��P	�����A�#*

	conv_lossأ�>
�Δ        )��P	�����A�#*

	conv_loss�0�>��=L        )��P	�E����A�#*

	conv_loss~��>Zܱ`        )��P	+u����A�#*

	conv_losssw�>�"�        )��P	������A�#*

	conv_loss�>�oT        )��P	������A�#*

	conv_loss��>t�2F        )��P	"����A�#*

	conv_lossaț>�u��        )��P	�6����A�#*

	conv_loss���>3���        )��P	yg����A�#*

	conv_loss%o�>�"��        )��P	�����A�#*

	conv_lossi��>�a"        )��P	������A�#*

	conv_lossY�>���C        )��P	$�����A�#*

	conv_loss���>^-�5        )��P	%����A�#*

	conv_loss@�>����        )��P	_T����A�#*

	conv_loss��>�ȧx        )��P	�����A�#*

	conv_loss�>[�!        )��P	H�����A�#*

	conv_loss���>t^qe        )��P	G�����A�#*

	conv_loss�?�>�r�        )��P	� ����A�#*

	conv_loss�ڜ>�F�        )��P	�C ����A�#*

	conv_losswV�>^���        )��P	>t ����A�#*

	conv_lossi��>��<�        )��P	y� ����A�#*

	conv_loss(�>��|$        )��P	n� ����A�#*

	conv_loss���>f�u-        )��P	W!����A�#*

	conv_losss�>�Z�=        )��P	=6!����A�#*

	conv_lossK��>pg~�        )��P	Xg!����A�#*

	conv_loss�B�>�� e        )��P	u�!����A�#*

	conv_loss��>tXq        )��P	a�!����A�#*

	conv_loss�>���J        )��P	�	"����A�#*

	conv_loss���>�H�        )��P	?="����A�#*

	conv_lossl�>j�̇        )��P	�q"����A�#*

	conv_loss21�>>�*        )��P	��"����A�#*

	conv_lossW+�>p�VP        )��P	��"����A�#*

	conv_loss:�>�n(        )��P	#����A�#*

	conv_lossС�>R�        )��P	a?#����A�#*

	conv_lossy�>���V        )��P	<w#����A�#*

	conv_loss�J�>S1�2        )��P	>�#����A�#*

	conv_loss"F�>��'I        )��P	��#����A�#*

	conv_loss���>(���        )��P	�$����A�#*

	conv_loss��>�aU        )��P	>$����A�#*

	conv_lossJ�>iW        )��P	9t$����A�#*

	conv_loss��>a��Q        )��P	��$����A�#*

	conv_loss6�>���        )��P	A�$����A�#*

	conv_loss�s�>���s        )��P	�%����A�#*

	conv_loss���>�i�        )��P	QH%����A�#*

	conv_lossm�>��,        )��P	l|%����A�#*

	conv_loss
0�>��        )��P	\�%����A�#*

	conv_loss=��>)]��        )��P	��%����A�#*

	conv_loss�ٜ>�R+        )��P	&����A�#*

	conv_loss���>:�}        )��P	�E&����A�#*

	conv_loss�r�>*��        )��P	�v&����A�#*

	conv_loss�К>aT��        )��P	�&����A�#*

	conv_loss7��>�i7i        )��P	��&����A�#*

	conv_loss�B�>�zD�        )��P	�'����A�#*

	conv_loss=��>�Ee        )��P	m='����A�#*

	conv_loss��>�cۃ        )��P	�l'����A�#*

	conv_lossuy�>T�        )��P	��'����A�#*

	conv_loss�S�>8�b        )��P	��'����A�#*

	conv_loss,�>���        )��P	�'����A�#*

	conv_loss�c�>� .~        )��P	)(����A�#*

	conv_lossM��>��        )��P	�X(����A�#*

	conv_loss]J�>	R7        )��P	ň(����A�#*

	conv_lossF��>a9��        )��P	Ϸ(����A�#*

	conv_loss� �>���        )��P	��(����A�#*

	conv_loss��>�B�r        )��P	y)����A�#*

	conv_loss�D�>�X}�        )��P	�J)����A�#*

	conv_loss�>��p�        )��P	�{)����A�#*

	conv_loss~��>�qx�        )��P	��)����A�#*

	conv_loss= �>��U        )��P	��)����A�#*

	conv_loss�P�>��ܮ        )��P	�*����A�#*

	conv_loss�ś>���        )��P	n:*����A�#*

	conv_loss�Ę>Z�~S        )��P	Zi*����A�#*

	conv_lossX�>�mRg        )��P	�*����A�#*

	conv_loss`#�>h$�e        )��P	��*����A�#*

	conv_loss�&�>�	v        )��P	y�*����A�#*

	conv_loss�˙>8�        )��P	B&+����A�#*

	conv_losso�>�i�        )��P	��,����A�#*

	conv_loss��>��[        )��P	��,����A�#*

	conv_loss���>h�        )��P	e-����A�#*

	conv_loss��>
$�        )��P	�L-����A�#*

	conv_loss"q�>��s        )��P	�-����A�#*

	conv_loss�ژ>t�6�        )��P	�-����A�#*

	conv_lossA�>��        )��P	��-����A�#*

	conv_lossX��>�ͨ        )��P	�.����A�$*

	conv_loss�>3��        )��P		U.����A�$*

	conv_loss���>�s�        )��P	ϋ.����A�$*

	conv_loss�d�>U�G        )��P	D�.����A�$*

	conv_loss��>�?E        )��P	?�.����A�$*

	conv_loss��>�lG�        )��P	1/����A�$*

	conv_lossSϗ>��Z        )��P	qa/����A�$*

	conv_lossu��>����        )��P	�/����A�$*

	conv_loss���>��`f        )��P	\�/����A�$*

	conv_loss
)�>�Q5�        )��P	E�/����A�$*

	conv_loss��>�x��        )��P	V!0����A�$*

	conv_loss�7�>��l        )��P	=R0����A�$*

	conv_loss���>G���        )��P	�0����A�$*

	conv_loss��>S�}�        )��P	ʯ0����A�$*

	conv_lossYY�>����        )��P	B�0����A�$*

	conv_loss��>���        )��P	"1����A�$*

	conv_lossؠ�>5J�        )��P	kL1����A�$*

	conv_loss���>�j��        )��P	31����A�$*

	conv_loss�9�>u80        )��P	��1����A�$*

	conv_loss΀�>��;        )��P	��1����A�$*

	conv_loss�>�Ҫ�        )��P	|2����A�$*

	conv_loss�g�>��Ѯ        )��P	3=2����A�$*

	conv_loss� �>���        )��P	jn2����A�$*

	conv_loss���>�$��        )��P	Ğ2����A�$*

	conv_loss�>�>��&        )��P	1�2����A�$*

	conv_loss���>���5        )��P	K 3����A�$*

	conv_loss�7�>�~�        )��P	�33����A�$*

	conv_lossB��>ߢtK        )��P	mc3����A�$*

	conv_loss,�>J�        )��P	5�3����A�$*

	conv_loss�8�>�&�
        )��P	�3����A�$*

	conv_loss��>��E(        )��P	��3����A�$*

	conv_loss��>�&�k        )��P	B@4����A�$*

	conv_loss<�>�4�        )��P	�o4����A�$*

	conv_lossVi�>֪>�        )��P	�4����A�$*

	conv_lossd�>���        )��P	��4����A�$*

	conv_loss�Ζ>k�|C        )��P	 5����A�$*

	conv_loss��>�q"�        )��P	15����A�$*

	conv_losss��>4���        )��P	rc5����A�$*

	conv_loss<�>~�Z"        )��P	��5����A�$*

	conv_loss�ؙ>v�e�        )��P	0�5����A�$*

	conv_lossxҖ>,-        )��P	q�5����A�$*

	conv_loss��>���        )��P	�,6����A�$*

	conv_loss�]�>~��G        )��P	Wd6����A�$*

	conv_lossrF�>����        )��P	˧6����A�$*

	conv_loss3N�>���&        )��P	��6����A�$*

	conv_loss�T�>+K�        )��P	�	7����A�$*

	conv_loss�Е>��"^        )��P	�<7����A�$*

	conv_loss��>��1�        )��P	Hn7����A�$*

	conv_lossO�>*��        )��P	��7����A�$*

	conv_loss�ؗ>�g'        )��P	��7����A�$*

	conv_loss��>�7��        )��P	� 8����A�$*

	conv_lossXZ�>+*�        )��P	�18����A�$*

	conv_loss$ܙ>/�p�        )��P	3p8����A�$*

	conv_loss'ޖ><�GC        )��P	j�8����A�$*

	conv_lossb��>x&��        )��P	)�8����A�$*

	conv_loss�ח> ��        )��P	�9����A�$*

	conv_lossrܖ>�h&�        )��P	�A9����A�$*

	conv_loss��>)�W�        )��P	�w9����A�$*

	conv_lossܙ>�~�        )��P	��9����A�$*

	conv_loss��>6�        )��P	��9����A�$*

	conv_loss*̖>T��        )��P	7:����A�$*

	conv_losswܙ>cY��        )��P	�5:����A�$*

	conv_loss���>Ŧ��        )��P	�f:����A�$*

	conv_loss�-�>�L+        )��P	�:����A�$*

	conv_loss���>8�{1        )��P	�:����A�$*

	conv_loss�i�>�p'        )��P	��:����A�$*

	conv_loss�u�>]���        )��P	[*;����A�$*

	conv_lossa��>6�        )��P	�Z;����A�$*

	conv_loss5�>�8`        )��P	�;����A�$*

	conv_loss���> ��(        )��P	׹;����A�$*

	conv_loss�ƙ>t���        )��P	��;����A�$*

	conv_loss��>��\h        )��P	[<����A�$*

	conv_loss��>����        )��P	$H<����A�$*

	conv_loss���>}G�        )��P	�y<����A�$*

	conv_lossߓ>
���        )��P	��<����A�$*

	conv_lossޣ�>~�m;        )��P	��<����A�$*

	conv_losse͗>���        )��P	a=����A�$*

	conv_loss��>U��W        )��P	�>=����A�$*

	conv_loss�w�>A��j        )��P	�n=����A�$*

	conv_loss�9�>y40        )��P	��=����A�$*

	conv_lossN�>IV�        )��P	��=����A�$*

	conv_lossj��>Q�ˑ        )��P	��=����A�$*

	conv_loss���>�M�"        )��P	�/>����A�$*

	conv_lossڔ>�
��        )��P	7`>����A�$*

	conv_loss�ܖ>_1��        )��P	{�>����A�$*

	conv_loss���>5�ܧ        )��P	��>����A�$*

	conv_lossyD�>CW�        )��P	��>����A�$*

	conv_loss�8�>�9�J        )��P	�'?����A�$*

	conv_loss�X�>rB�        )��P	fW?����A�$*

	conv_lossY��>�f��        )��P	�?����A�$*

	conv_losse��>��	�        )��P	j�?����A�$*

	conv_loss�e�>՟
z        )��P	��?����A�$*

	conv_lossި�>I
R�        )��P	{@����A�$*

	conv_loss�x�>�B$2        )��P	^@����A�$*

	conv_loss�C�>�V��        )��P	��@����A�$*

	conv_loss8��>+ªQ        )��P	�@����A�$*

	conv_loss!�>h�E        )��P	��@����A�$*

	conv_loss�u�>d��        )��P		'A����A�$*

	conv_loss�Γ>+E*�        )��P	�VA����A�$*

	conv_loss�~�>�E��        )��P	m�A����A�$*

	conv_loss^�>旷u        )��P	w�A����A�$*

	conv_loss|��>�:+�        )��P	��A����A�$*

	conv_loss��>�q�        )��P	-B����A�$*

	conv_loss"�>��        )��P	Y^B����A�$*

	conv_loss�t�>8Ȯ�        )��P	��B����A�$*

	conv_loss�w�>��p+        )��P	)�B����A�$*

	conv_loss�L�>��%        )��P	�C����A�$*

	conv_loss,��>���<        )��P	c6C����A�$*

	conv_lossK�>��        )��P	 gC����A�$*

	conv_loss��>��2        )��P	ԖC����A�$*

	conv_loss���>j�r        )��P		�C����A�$*

	conv_loss��>Q��        )��P	c�C����A�$*

	conv_lossI��>k���        )��P	()D����A�$*

	conv_loss�ԕ>ڎ4.        )��P	mZD����A�$*

	conv_loss���>�H�        )��P	�D����A�$*

	conv_loss^��>��ܳ        )��P	S�D����A�$*

	conv_loss���>��ǝ        )��P	��D����A�$*

	conv_loss8ڔ>��j        )��P	wE����A�$*

	conv_loss?�>,�P�        )��P	0OE����A�$*

	conv_loss�z�>D�Q        )��P	9�E����A�$*

	conv_lossS�>��        )��P	R�E����A�$*

	conv_loss��>yð        )��P	Y�E����A�$*

	conv_loss㯔>��J�        )��P	�F����A�$*

	conv_loss�ϖ>����        )��P	^@F����A�$*

	conv_loss�ɗ>����        )��P	rF����A�$*

	conv_lossl��>&�@�        )��P	ۤF����A�$*

	conv_loss�>�j�        )��P	�F����A�$*

	conv_losst&�>pIWs        )��P	G����A�$*

	conv_loss"b�>@#	;        )��P	�9G����A�%*

	conv_loss�m�>a�p        )��P	�mG����A�%*

	conv_loss�ߕ>!�ƀ        )��P	ԝG����A�%*

	conv_loss0̕>��\�        )��P	�G����A�%*

	conv_lossl�>�EI�        )��P	��G����A�%*

	conv_loss��>�¸{        )��P	!-H����A�%*

	conv_lossUL�>cf�        )��P	�^H����A�%*

	conv_loss�g�>�        )��P	��H����A�%*

	conv_loss��>�5�        )��P	�H����A�%*

	conv_loss�h�>�f�1        )��P	��H����A�%*

	conv_lossp�>��H0        )��P	�$I����A�%*

	conv_loss[��>@        )��P	�VI����A�%*

	conv_loss0��>ܩr.        )��P	��I����A�%*

	conv_lossҏ�>�Hk�        )��P	h�I����A�%*

	conv_loss]ܕ>���1        )��P	��I����A�%*

	conv_loss�ܕ>?�fo        )��P	4J����A�%*

	conv_lossQ��>�1��        )��P	7hJ����A�%*

	conv_losse��>)qx@        )��P	��J����A�%*

	conv_loss���>����        )��P	��J����A�%*

	conv_losst#�>׊�#        )��P	��J����A�%*

	conv_loss�ԓ>��O        )��P	�2K����A�%*

	conv_loss��>'a�.        )��P	�dK����A�%*

	conv_lossؕ>��N�        )��P	U�K����A�%*

	conv_loss�>:�x�        )��P	�K����A�%*

	conv_loss�M�>���        )��P	PL����A�%*

	conv_loss�3�>�E��        )��P	�HL����A�%*

	conv_lossu��>�H�K        )��P	�zL����A�%*

	conv_lossȱ�>l�j        )��P	�L����A�%*

	conv_loss�>�F�        )��P	��L����A�%*

	conv_loss`�>\�k6        )��P	�!M����A�%*

	conv_lossr|�>�j)        )��P	9SM����A�%*

	conv_loss�I�>m��        )��P	ȄM����A�%*

	conv_loss�>����        )��P	ͶM����A�%*

	conv_lossam�>�&��        )��P	��M����A�%*

	conv_loss2�>Ӷ1�        )��P	E2N����A�%*

	conv_losszl�>&��        )��P	�gN����A�%*

	conv_loss9;�>y�`y        )��P	��N����A�%*

	conv_loss�y�>�پ�        )��P	��N����A�%*

	conv_loss(ȕ>P���        )��P	�O����A�%*

	conv_lossr�>��ɠ        )��P	)8O����A�%*

	conv_loss�[�>�;��        )��P	ZkO����A�%*

	conv_loss:C�>��ʌ        )��P	ӟO����A�%*

	conv_loss�N�>�x7�        )��P	�O����A�%*

	conv_loss��>"��        )��P	�P����A�%*

	conv_loss|��>� v�        )��P	:P����A�%*

	conv_losso��>Z���        )��P	�lP����A�%*

	conv_loss�>�{4�        )��P	��P����A�%*

	conv_loss}��>����        )��P	��P����A�%*

	conv_losss��>���        )��P	�Q����A�%*

	conv_loss<��>��pO        )��P	�;Q����A�%*

	conv_lossǡ�>��5�        )��P	�nQ����A�%*

	conv_loss�є>���        )��P	�Q����A�%*

	conv_loss�ܓ>g��        )��P	��Q����A�%*

	conv_loss��>/J@Q        )��P	�R����A�%*

	conv_loss��>#
�        )��P	�<R����A�%*

	conv_loss��>��ٓ        )��P	�qR����A�%*

	conv_loss�2�>��-        )��P	ۤR����A�%*

	conv_loss�x�>#���        )��P	��R����A�%*

	conv_loss!�>��Q        )��P	5S����A�%*

	conv_loss>��>�@iJ        )��P	�?S����A�%*

	conv_loss���>b�M?        )��P	+sS����A�%*

	conv_loss�D�>-��#        )��P	��S����A�%*

	conv_lossY�>8D<B        )��P	N�S����A�%*

	conv_loss�(�>ǰ�C        )��P	*T����A�%*

	conv_loss��>��        )��P	aBT����A�%*

	conv_loss��>D)j�        )��P	
�U����A�%*

	conv_loss�c�>p        )��P	�V����A�%*

	conv_loss՝�>]�#B        )��P	�=V����A�%*

	conv_loss�ݒ>���        )��P	moV����A�%*

	conv_loss5��>��        )��P	��V����A�%*

	conv_loss�]�>���I        )��P	��V����A�%*

	conv_loss3��>�Œ�        )��P	w	W����A�%*

	conv_loss�A�>Y�n�        )��P	�CW����A�%*

	conv_lossa��>�]0�        )��P	RxW����A�%*

	conv_loss�M�>��7        )��P	C�W����A�%*

	conv_loss���>m��D        )��P	��W����A�%*

	conv_loss��>���        )��P	@X����A�%*

	conv_lossF�>0'�        )��P	~>X����A�%*

	conv_loss��>4:��        )��P	�mX����A�%*

	conv_loss�۔>2b~�        )��P	��X����A�%*

	conv_loss:�>��2M        )��P	.�X����A�%*

	conv_loss���>���        )��P	��X����A�%*

	conv_loss4��>X��6        )��P	<3Y����A�%*

	conv_losseԑ>��S        )��P	"cY����A�%*

	conv_loss�>��C        )��P	��Y����A�%*

	conv_loss��>|�Zl        )��P	.�Y����A�%*

	conv_loss�ב>WO�'        )��P	�Y����A�%*

	conv_loss,w�>�E�        )��P	�*Z����A�%*

	conv_loss^]�>(ӈ        )��P	XbZ����A�%*

	conv_lossK>�>z�-        )��P	�Z����A�%*

	conv_lossh�>�t~        )��P	%�Z����A�%*

	conv_lossX|�>
@W�        )��P	4�Z����A�%*

	conv_loss��>6��        )��P	U"[����A�%*

	conv_lossB�>ڭ	�        )��P	;^[����A�%*

	conv_loss��>Ѭ�        )��P	r�[����A�%*

	conv_lossU��>��MX        )��P	`�[����A�%*

	conv_loss���>a*        )��P	��[����A�%*

	conv_loss��>�W��        )��P	�-\����A�%*

	conv_loss3*�>�ʸO        )��P	�]\����A�%*

	conv_lossNۓ>�2�6        )��P	Ґ\����A�%*

	conv_loss΋�>���        )��P	�\����A�%*

	conv_loss��>]���        )��P	.�\����A�%*

	conv_lossK=�>84'        )��P	� ]����A�%*

	conv_loss(��>�2�!        )��P	R]����A�%*

	conv_loss��>��z�        )��P	|�]����A�%*

	conv_lossz7�>��c        )��P	�]����A�%*

	conv_loss=S�>��Q�        )��P	��]����A�%*

	conv_loss�r�>S���        )��P	e^����A�%*

	conv_lossJ�>�m�#        )��P	�A^����A�%*

	conv_loss��>��J�        )��P	%t^����A�%*

	conv_loss8��>1��        )��P	Ť^����A�%*

	conv_lossO\�>�k�5        )��P	��^����A�%*

	conv_lossђ�>��1        )��P	J_����A�%*

	conv_loss�8�><���        )��P	�8_����A�%*

	conv_lossB�>��dd        )��P	�f_����A�%*

	conv_lossZZ�>	$K�        )��P	�_����A�%*

	conv_loss��>M}�        )��P	O�_����A�%*

	conv_loss�b�>w&�        )��P	7`����A�%*

	conv_loss��>^��        )��P	�7`����A�%*

	conv_loss��>�CZ�        )��P	tk`����A�%*

	conv_loss��>z�fM        )��P	��`����A�%*

	conv_loss^��>`�w        )��P	��`����A�%*

	conv_lossl|�>A*�        )��P	b
a����A�%*

	conv_loss���>R<�)        )��P	OHa����A�%*

	conv_loss�r�>Yǖ�        )��P	oa����A�%*

	conv_loss>�>5�.|        )��P	�a����A�%*

	conv_losss�>s?j        )��P	�a����A�%*

	conv_loss�1�>��ǉ        )��P	tb����A�%*

	conv_loss0%�>�Y�h        )��P	�Pb����A�&*

	conv_loss�\�>�Wt)        )��P	܂b����A�&*

	conv_loss��>*�        )��P	Գb����A�&*

	conv_loss�ߒ>	KƗ        )��P	��b����A�&*

	conv_lossL��>��,        )��P	%c����A�&*

	conv_loss��>6���        )��P	Jc����A�&*

	conv_lossÚ�>��]�        )��P	3c����A�&*

	conv_loss��>ȩ��        )��P	'�c����A�&*

	conv_loss4�>؉@        )��P	r�c����A�&*

	conv_loss[}�>�:C        )��P	�"d����A�&*

	conv_loss�Ϗ>�a�3        )��P	�Wd����A�&*

	conv_lossFm�>zZ��        )��P	)�d����A�&*

	conv_loss��>�sy�        )��P	��d����A�&*

	conv_loss �>��K        )��P	h�d����A�&*

	conv_loss���>�z��        )��P	;!e����A�&*

	conv_lossI��>�}�        )��P	Se����A�&*

	conv_loss�]�>q�m�        )��P	{�e����A�&*

	conv_loss)��>��i�        )��P	P�e����A�&*

	conv_loss���>J?�B        )��P	��e����A�&*

	conv_loss#t�>��ϡ        )��P	u"f����A�&*

	conv_loss��>��        )��P	�Tf����A�&*

	conv_loss���>���=        )��P	�f����A�&*

	conv_loss���>���7        )��P	ùf����A�&*

	conv_loss�m�>|U�j        )��P	��f����A�&*

	conv_loss�ό>TJ        )��P	�!g����A�&*

	conv_loss���>��        )��P	BTg����A�&*

	conv_loss3#�>V�K�        )��P	��g����A�&*

	conv_loss�Z�>�+�        )��P	�g����A�&*

	conv_loss���>*H�J        )��P	��g����A�&*

	conv_loss�c�>��B        )��P	$h����A�&*

	conv_loss��> �>        )��P	WUh����A�&*

	conv_lossd�>���`        )��P	�h����A�&*

	conv_lossNؑ>t���        )��P	3�h����A�&*

	conv_lossT��>.�L        )��P	L�h����A�&*

	conv_lossg�>�Vj        )��P	� i����A�&*

	conv_loss��>���:        )��P	�Si����A�&*

	conv_lossoҎ>M�3        )��P	B�i����A�&*

	conv_loss���>�G��        )��P	��i����A�&*

	conv_loss͏>-��        )��P	��i����A�&*

	conv_loss�Њ>T[^�        )��P	�1j����A�&*

	conv_lossW �>��        )��P	�ej����A�&*

	conv_lossި�>�($        )��P	�j����A�&*

	conv_loss�ߌ>~��        )��P	9�j����A�&*

	conv_lossgM�>�07�        )��P	%k����A�&*

	conv_loss5 �>����        )��P	#Hk����A�&*

	conv_loss9V�>=�.I        )��P	�{k����A�&*

	conv_loss8�>�z��        )��P	��k����A�&*

	conv_loss�a�>��P        )��P	e�k����A�&*

	conv_loss&�>$���        )��P	Bl����A�&*

	conv_loss,��>�\��        )��P	�Ll����A�&*

	conv_lossn�>��F        )��P	�l����A�&*

	conv_loss�"�>݉7�        )��P	��l����A�&*

	conv_loss��>�*�        )��P	��l����A�&*

	conv_lossݡ�>z\{u        )��P	!m����A�&*

	conv_losst�>ĠL        )��P	KQm����A�&*

	conv_loss�Ǝ>o�        )��P	5�m����A�&*

	conv_loss9,�>;�4�        )��P	�m����A�&*

	conv_loss��>��        )��P	��m����A�&*

	conv_lossP�>�(pT        )��P	 /n����A�&*

	conv_loss,�>�oK        )��P	�cn����A�&*

	conv_loss�-�>��J        )��P	ՙn����A�&*

	conv_loss
�>��        )��P	l�n����A�&*

	conv_loss�!�>aj�        )��P	P o����A�&*

	conv_loss�׍>3Zu�        )��P	�4o����A�&*

	conv_loss@�>~��O        )��P	�fo����A�&*

	conv_loss�>9)Ͳ        )��P	��o����A�&*

	conv_loss�P�>*�uy        )��P	��o����A�&*

	conv_loss�-�>M}        )��P	�p����A�&*

	conv_loss�=�>��r�        )��P	4p����A�&*

	conv_loss�Տ>tC'/        )��P	�gp����A�&*

	conv_loss�B�>��        )��P	��p����A�&*

	conv_loss�̑>�v��        )��P	1�p����A�&*

	conv_lossΎ>%�1        )��P	�q����A�&*

	conv_loss��>���        )��P	�5q����A�&*

	conv_loss���>0�|k        )��P	�hq����A�&*

	conv_loss�>(�o        )��P	��q����A�&*

	conv_loss��>'�u        )��P	O�q����A�&*

	conv_loss^C�>j��n        )��P	Z�q����A�&*

	conv_loss���>��v�        )��P	T4r����A�&*

	conv_loss�(�>n��5        )��P	kr����A�&*

	conv_loss���>e�i�        )��P	ȝr����A�&*

	conv_lossJI�>��        )��P	C�r����A�&*

	conv_loss!�>`P��        )��P	"s����A�&*

	conv_loss�b�>�m�        )��P	5s����A�&*

	conv_losszL�>�8B�        )��P	is����A�&*

	conv_loss���>�+^�        )��P	�s����A�&*

	conv_loss�m�> �x(        )��P	��s����A�&*

	conv_loss7�>�&�        )��P	�t����A�&*

	conv_loss���>�O�        )��P	�It����A�&*

	conv_loss���>�jC�        )��P	
{t����A�&*

	conv_loss)o�>V�-�        )��P	��t����A�&*

	conv_loss"�>aU)�        )��P	u�t����A�&*

	conv_lossDX�>�;Ĩ        )��P	�3u����A�&*

	conv_loss�G�>͏        )��P	�ku����A�&*

	conv_loss�k�>���        )��P		�u����A�&*

	conv_loss���>F4^R        )��P	0�u����A�&*

	conv_loss#��>o.H�        )��P	�$v����A�&*

	conv_loss=l�>⏶�        )��P	�Wv����A�&*

	conv_losse��>8D�         )��P	x�v����A�&*

	conv_lossMN�>�͆        )��P	j�v����A�&*

	conv_loss�>� ί        )��P	#�v����A�&*

	conv_loss���>��_�        )��P	$w����A�&*

	conv_loss`��>6��        )��P	�Yw����A�&*

	conv_loss��>iݡ�        )��P	@�w����A�&*

	conv_loss�>�>���        )��P	��w����A�&*

	conv_loss���>-��        )��P	�x����A�&*

	conv_loss
��>�^�>        )��P	pCx����A�&*

	conv_loss���>G��        )��P	^yx����A�&*

	conv_loss�>h�7        )��P	7�x����A�&*

	conv_lossʓ�>{���        )��P	s�x����A�&*

	conv_loss��>�K�        )��P	�y����A�&*

	conv_loss�5�>�'�        )��P	�Gy����A�&*

	conv_loss=��>���        )��P	x{y����A�&*

	conv_loss# �>`<�        )��P	Ԯy����A�&*

	conv_lossh�>���o        )��P	\�y����A�&*

	conv_loss��>c��        )��P	�z����A�&*

	conv_losso]�>)s1�        )��P	�Jz����A�&*

	conv_lossCҍ>�M�        )��P	~z����A�&*

	conv_lossj�>evp�        )��P	��z����A�&*

	conv_loss�/�>���        )��P	F�z����A�&*

	conv_loss]
�>���        )��P	�{����A�&*

	conv_lossF�>���        )��P	'K{����A�&*

	conv_loss}Ύ>1Ap        )��P	�~{����A�&*

	conv_loss�։>�%�'        )��P	>�{����A�&*

	conv_loss���>b�]        )��P	#�{����A�&*

	conv_lossI�>._H>        )��P	G|����A�&*

	conv_loss�҈>�$o        )��P	�M|����A�&*

	conv_loss��>vZ9�        )��P	W|����A�&*

	conv_loss��>V=?-        )��P	�|����A�'*

	conv_loss���>=�Ԁ        )��P	��|����A�'*

	conv_lossR�>�!R        )��P	�}����A�'*

	conv_loss_T�>�x|9        )��P	�P}����A�'*

	conv_loss|p�>�O�Y        )��P	ă}����A�'*

	conv_lossX��>����        )��P	��}����A�'*

	conv_loss��>K��        )��P	"�}����A�'*

	conv_loss�>�M��        )��P	N ~����A�'*

	conv_lossv�>T�#�        )��P	�.�����A�'*

	conv_loss��>��K�        )��P	d������A�'*

	conv_loss�3�>=���        )��P	������A�'*

	conv_loss*��>�Q�        )��P	" �����A�'*

	conv_loss��>6~�        )��P	0Q�����A�'*

	conv_loss�'�>��`        )��P	G������A�'*

	conv_loss�>Q���        )��P	������A�'*

	conv_loss��>[���        )��P	������A�'*

	conv_loss���>\3�        )��P	K�����A�'*

	conv_loss�_�>1/��        )��P	=F�����A�'*

	conv_loss�k�>M3�&        )��P	[w�����A�'*

	conv_loss짊>�g��        )��P	I������A�'*

	conv_loss���>|s��        )��P	������A�'*

	conv_loss)�>�A�        )��P	R�����A�'*

	conv_loss���>�gi        )��P	}O�����A�'*

	conv_loss���>��,�        )��P	l������A�'*

	conv_loss�
�>��        )��P	6������A�'*

	conv_loss��>��-        )��P	E݇����A�'*

	conv_loss��>_o��        )��P	������A�'*

	conv_loss���>����        )��P	h?�����A�'*

	conv_loss�N�>��        )��P	o�����A�'*

	conv_loss�`�>s�(        )��P	������A�'*

	conv_loss玉>�a��        )��P	�ψ����A�'*

	conv_loss{O�><�\�        )��P	�������A�'*

	conv_loss�Ќ>,�~o        )��P	@.�����A�'*

	conv_loss�9�>��        )��P	�\�����A�'*

	conv_loss���>m��        )��P	�������A�'*

	conv_loss�U�>�4�u        )��P	S������A�'*

	conv_loss�c�>��/        )��P	5�����A�'*

	conv_loss"Ћ>����        )��P	5�����A�'*

	conv_loss��>J���        )��P	mF�����A�'*

	conv_loss�W�>z��        )��P	�u�����A�'*

	conv_lossDq�>v�v�        )��P	զ�����A�'*

	conv_loss+f�>7�e        )��P	ي����A�'*

	conv_loss=ԉ>p��        )��P	������A�'*

	conv_loss[�>U�m#        )��P	�8�����A�'*

	conv_loss�6�>�h��        )��P	�f�����A�'*

	conv_loss2։>����        )��P	Е�����A�'*

	conv_lossNw�>^�O        )��P	ċ����A�'*

	conv_lossw!�>�V��        )��P	K�����A�'*

	conv_loss��>��?        )��P	"�����A�'*

	conv_loss.'�>ߜ�        )��P	�O�����A�'*

	conv_loss�,�>-1-        )��P	N�����A�'*

	conv_loss�1�>��p�        )��P	B������A�'*

	conv_lossꩅ>$�[�        )��P	������A�'*

	conv_lossj��>�
��        )��P	������A�'*

	conv_lossa׊>7鎧        )��P	B�����A�'*

	conv_lossDɊ>lH>        )��P	�t�����A�'*

	conv_loss���>9|��        )��P	������A�'*

	conv_loss��>���I        )��P	܍����A�'*

	conv_loss�;�>Ĺ        )��P	M�����A�'*

	conv_loss�N�>q`        )��P	�T�����A�'*

	conv_loss��>�6֋        )��P	)������A�'*

	conv_loss+6�>����        )��P	츎����A�'*

	conv_loss��>�n��        )��P	�����A�'*

	conv_loss��>ݲ_        )��P	 �����A�'*

	conv_lossЩ�>و��        )��P	!R�����A�'*

	conv_loss���>UF�        )��P	킏����A�'*

	conv_loss�/�>k�OV        )��P	M������A�'*

	conv_loss
X�>W_��        )��P	������A�'*

	conv_loss�M�>F�W�        )��P	+'�����A�'*

	conv_loss�Έ>�0΋        )��P	�Z�����A�'*

	conv_loss�~�>c���        )��P	������A�'*

	conv_loss�ڊ>Q��z        )��P	�������A�'*

	conv_loss/�>e�        )��P	#�����A�'*

	conv_loss���>��\H        )��P	�9�����A�'*

	conv_loss�!�>:�"        )��P	�n�����A�'*

	conv_lossȤ�>-�xu        )��P	������A�'*

	conv_loss箋>���        )��P	B֑����A�'*

	conv_lossj��>�S+.        )��P	������A�'*

	conv_loss섇>P�9        )��P	0:�����A�'*

	conv_loss}��>z�m        )��P	@k�����A�'*

	conv_loss�]�>�\83        )��P	✒����A�'*

	conv_loss���>���        )��P	�ϒ����A�'*

	conv_lossm��>b�+�        )��P	������A�'*

	conv_loss�҅>�Xb�        )��P	N5�����A�'*

	conv_loss!��>�˄�        )��P	�i�����A�'*

	conv_loss��>s�V        )��P	&������A�'*

	conv_loss�>p��        )��P	ғ����A�'*

	conv_lossǦ�>X�        )��P	_�����A�'*

	conv_loss��>Ǌ��        )��P	6�����A�'*

	conv_loss\]�>��1�        )��P	�g�����A�'*

	conv_lossc��>$y�        )��P	�������A�'*

	conv_loss
��>��        )��P	�Δ����A�'*

	conv_lossa��>{a�        )��P	�������A�'*

	conv_loss�>n�ʀ        )��P	�1�����A�'*

	conv_loss,��>�k�        )��P	�d�����A�'*

	conv_loss[f�>̍��        )��P	�������A�'*

	conv_loss�?�>c	Sp        )��P	�ƕ����A�'*

	conv_loss"݆>�k�.        )��P	G������A�'*

	conv_loss�g�>�Hޗ        )��P	
+�����A�'*

	conv_lossz+�>�L�        )��P	Z]�����A�'*

	conv_loss �>ς+m        )��P	8������A�'*

	conv_loss�5�>�7�<        )��P	�������A�'*

	conv_lossl$�>�\?�        )��P	R�����A�'*

	conv_loss ��>�Ya+        )��P	d"�����A�'*

	conv_loss�B�>��!X        )��P	*V�����A�'*

	conv_losss�>T!�g        )��P	;������A�'*

	conv_loss\��>%�P�        )��P	�������A�'*

	conv_loss�3�>k�D"        )��P	^�����A�'*

	conv_lossP�>T�x        )��P	������A�'*

	conv_loss��>���(        )��P	:g�����A�'*

	conv_loss��>�_a�        )��P	�������A�'*

	conv_loss�{�>����        )��P	 ʘ����A�'*

	conv_lossĂ>G�
        )��P	� �����A�'*

	conv_loss�Q�>
�k        )��P	?5�����A�'*

	conv_loss�C�>�%E        )��P	?l�����A�'*

	conv_loss���>'��O        )��P	�������A�'*

	conv_lossq��>M,��        )��P	�ϙ����A�'*

	conv_lossKΆ>Þ        )��P	������A�'*

	conv_loss<�>�!~�        )��P	�=�����A�'*

	conv_lossʃ>��,�        )��P	Xx�����A�'*

	conv_loss:�>K���        )��P	A������A�'*

	conv_loss�a�>��y�        )��P	�ۚ����A�'*

	conv_lossױ�>�ׄj        )��P	������A�'*

	conv_loss��>��G        )��P	�I�����A�'*

	conv_loss���>1 8        )��P	�������A�'*

	conv_loss^�>�        )��P	۰�����A�'*

	conv_loss��>�qw�        )��P	0�����A�'*

	conv_loss�`�>+�2/        )��P	'�����A�'*

	conv_lossA�>fz��        )��P	�F�����A�(*

	conv_lossWˁ>/Z@�        )��P	�z�����A�(*

	conv_lossZt�>75��        )��P	e������A�(*

	conv_lossP��>ǏN�        )��P	b������A�(*

	conv_lossJa�>p���        )��P	f�����A�(*

	conv_loss���>���        )��P	zB�����A�(*

	conv_lossD�>Ij        )��P	x�����A�(*

	conv_loss�ƃ>�`C�        )��P	�������A�(*

	conv_lossf�>�Ka        )��P	�ݝ����A�(*

	conv_loss9G�>���        )��P	������A�(*

	conv_lossKe{>?je�        )��P	k?�����A�(*

	conv_loss�(�>~6˔        )��P	�q�����A�(*

	conv_lossJ8�>g��f        )��P	٣�����A�(*

	conv_loss�r{>��
        )��P	՞����A�(*

	conv_lossy@�>�-w�        )��P	������A�(*

	conv_loss�5�>��4�        )��P	7�����A�(*

	conv_loss���>Ů�a        )��P	�i�����A�(*

	conv_loss�}�>d��        )��P	�������A�(*

	conv_lossݍ�>�+�        )��P	qϟ����A�(*

	conv_loss��>4���        )��P	������A�(*

	conv_lossE�>[���        )��P	�2�����A�(*

	conv_lossAh�>��l        )��P	�e�����A�(*

	conv_loss�Ƀ>+jy        )��P	7������A�(*

	conv_lossZI�>*do[        )��P	�Ƞ����A�(*

	conv_loss��>J�(        )��P	D������A�(*

	conv_lossD�>r��f        )��P	�/�����A�(*

	conv_loss�-�>���        )��P	�b�����A�(*

	conv_loss߀>����        )��P	Z������A�(*

	conv_loss��>���        )��P	~á����A�(*

	conv_loss�߀>�EE        )��P	1������A�(*

	conv_loss<��>z`��        )��P	W*�����A�(*

	conv_lossˁ�><!��        )��P	�o�����A�(*

	conv_lossPy�>�30Q        )��P	Ġ�����A�(*

	conv_lossN��>L�@	        )��P	GѢ����A�(*

	conv_loss#��>���#        )��P	������A�(*

	conv_lossw�~>O�p        )��P	�6�����A�(*

	conv_loss��>$��        )��P	�g�����A�(*

	conv_lossP�>!�[        )��P	�������A�(*

	conv_lossmn�>���        )��P	�ͣ����A�(*

	conv_lossԢ�>���        )��P	�
�����A�(*

	conv_lossh�>8	�        )��P	�@�����A�(*

	conv_loss�\�>Y`B�        )��P	�r�����A�(*

	conv_lossWم>V�,        )��P	R������A�(*

	conv_loss"_�>P��        )��P	�ߤ����A�(*

	conv_loss}�>����        )��P	������A�(*

	conv_loss_ �>»t�        )��P	�N�����A�(*

	conv_loss��>۠H�        )��P	T������A�(*

	conv_loss���>+r�^        )��P	5������A�(*

	conv_loss��>_��        )��P	d�����A�(*

	conv_loss��>�̚�        )��P	������A�(*

	conv_lossA||>;e��        )��P	�K�����A�(*

	conv_loss��>|�{�        )��P	�}�����A�(*

	conv_loss���>��:0        )��P	ꮦ����A�(*

	conv_loss{�|>� y        )��P	�ߦ����A�(*

	conv_loss�}�>#CC�        )��P	������A�(*

	conv_lossͻ�>M�G        )��P	�H�����A�(*

	conv_loss��>�a�        )��P	�y�����A�(*

	conv_loss�y�>���        )��P	b������A�(*

	conv_loss���>����        )��P	9������A�(*

	conv_loss��>5l        )��P	+�����A�(*

	conv_loss.3~>j%��        )��P	�]�����A�(*

	conv_loss��>�ɍQ        )��P	,������A�(*

	conv_loss�|>���        )��P	{è����A�(*

	conv_loss��y>�%��        )��P	������A�(*

	conv_lossѢ�>��f<        )��P	*�����A�(*

	conv_loss곆>���B        )��P	�[�����A�(*

	conv_loss^�}>J�͈        )��P	U������A�(*

	conv_loss1z>4�>        )��P	�������A�(*

	conv_loss�3�>��έ        )��P	������A�(*

	conv_loss�O�>�        )��P	D$�����A�(*

	conv_losslԂ>΃�        )��P	�W�����A�(*

	conv_loss l~>���        )��P	�����A�(*

	conv_lossD�>��J        )��P	�������A�(*

	conv_loss8n�>T��        )��P	������A�(*

	conv_loss ��>܎��        )��P	/�����A�(*

	conv_lossg�>tp=        )��P	O�����A�(*

	conv_loss(�}>��n        )��P	M������A�(*

	conv_loss�]}>�F        )��P	�������A�(*

	conv_loss}��>@r^�        )��P	3�����A�(*

	conv_lossz]|>�-q�        )��P	e�����A�(*

	conv_loss"Á>�U��        )��P	�F�����A�(*

	conv_loss^��>���        )��P	i�����A�(*

	conv_loss�x~>ӻ)�        )��P	������A�(*

	conv_loss�w>�D��        )��P	SH�����A�(*

	conv_lossg�y>����        )��P	I������A�(*

	conv_loss��>�\�        )��P	������A�(*

	conv_lossq�>oF�        )��P	������A�(*

	conv_loss��>ݻ��        )��P	\)�����A�(*

	conv_lossJ�~>H0�        )��P	b�����A�(*

	conv_loss�y>u
q        )��P	�������A�(*

	conv_loss~>��<        )��P	b�����A�(*

	conv_lossBon>c��        )��P	�����A�(*

	conv_lossNf�>[%N        )��P	�R�����A�(*

	conv_loss���>�e�<        )��P	ȅ�����A�(*

	conv_loss@�~>�xT8        )��P	ѹ�����A�(*

	conv_loss���>ɇ��        )��P	�������A�(*

	conv_loss��{>ZU/        )��P	e �����A�(*

	conv_loss�>V\�        )��P	�T�����A�(*

	conv_lossͲ�>�Ф�        )��P	�������A�(*

	conv_loss��>mϝ:        )��P	�±����A�(*

	conv_lossB�{>�R��        )��P	������A�(*

	conv_losso�{>��a        )��P	2�����A�(*

	conv_lossv�z>�Ǹ�        )��P	Sg�����A�(*

	conv_loss끃>�~�        )��P	�������A�(*

	conv_loss��|>$}p5        )��P	kͲ����A�(*

	conv_loss�	n>�L�4        )��P	R �����A�(*

	conv_loss1}>sZ�f        )��P	"3�����A�(*

	conv_lossչ|>7 wN        )��P	�e�����A�(*

	conv_lossy s>:z��        )��P	�������A�(*

	conv_loss�ol>W04�        )��P	Eγ����A�(*

	conv_loss��v>�1��        )��P	�����A�(*

	conv_loss�p>
`7%        )��P	�7�����A�(*

	conv_loss$�w>���        )��P	_j�����A�(*

	conv_loss��y>�b��        )��P	�������A�(*

	conv_lossR��>+?v(        )��P	�ϴ����A�(*

	conv_loss�b�>�dQ        )��P	������A�(*

	conv_loss4��>�F�S        )��P	�5�����A�(*

	conv_loss�dy>�[��        )��P	�j�����A�(*

	conv_loss�}{>��!�        )��P	�������A�(*

	conv_lossJIw>�7Mo        )��P	�е����A�(*

	conv_lossy>�,V�        )��P	������A�(*

	conv_loss��q>)CG�        )��P	�7�����A�(*

	conv_loss��|>���        )��P	�l�����A�(*

	conv_loss��}>�&#�        )��P	Z������A�(*

	conv_loss�u>^���        )��P	�Ҷ����A�(*

	conv_loss(�{>3҇G        )��P	w�����A�(*

	conv_lossJ2~>��"        )��P	a8�����A�(*

	conv_loss�{>���        )��P	l�����A�(*

	conv_loss��>�b�5        )��P	Ȟ�����A�)*

	conv_lossTt>��~U        )��P	�ѷ����A�)*

	conv_loss��p>���        )��P	������A�)*

	conv_loss]�w>fH�6        )��P	�I�����A�)*

	conv_loss�(t>K�X.        )��P	d{�����A�)*

	conv_lossƷ~>�D�        )��P	�������A�)*

	conv_loss�q>����        )��P	������A�)*

	conv_lossk'}>b#0�        )��P	������A�)*

	conv_loss��}>]�        )��P	[�����A�)*

	conv_loss�z>~ԡ        )��P	�������A�)*

	conv_loss�2�>�@0�        )��P	�ǹ����A�)*

	conv_loss�y>Onq6        )��P	�������A�)*

	conv_lossYD~>҄��        )��P	�2�����A�)*

	conv_lossE�s>�W�        )��P	�l�����A�)*

	conv_loss0�{>�y�\        )��P	����A�)*

	conv_lossC,v>��	3        )��P	�Ӻ����A�)*

	conv_loss��r>�®�        )��P	������A�)*

	conv_loss/�v>c��        )��P	�;�����A�)*

	conv_lossXRu>m�l�        )��P	~������A�)*

	conv_lossͰo>�h_u        )��P	������A�)*

	conv_lossމo>l��,        )��P	������A�)*

	conv_lossz�p>Ko��        )��P	H%�����A�)*

	conv_loss�y>���B        )��P	$[�����A�)*

	conv_losst�|>��_        )��P	�����A�)*

	conv_loss!S|>gq��        )��P	�ü����A�)*

	conv_loss�Qx>ʣT        )��P	������A�)*

	conv_loss�h{>	��h        )��P	�*�����A�)*

	conv_lossf>1/x�        )��P	�\�����A�)*

	conv_loss�y>���m        )��P	�������A�)*

	conv_loss
Cv>T��        )��P	�ý����A�)*

	conv_loss�'u>����        )��P	������A�)*

	conv_lossgl>f��        )��P	�)�����A�)*

	conv_loss�qx>2""�        )��P	;_�����A�)*

	conv_loss=�n>�)�        )��P	�������A�)*

	conv_loss6�t>��W        )��P	�ľ����A�)*

	conv_loss%�|>,J��        )��P	`������A�)*

	conv_loss}�w>gSB�        )��P	�,�����A�)*

	conv_loss�}o>�        )��P	N_�����A�)*

	conv_lossqkv>�)�9        )��P	�����A�)*

	conv_loss��o>���        )��P	�ſ����A�)*

	conv_loss@	q>NVB        )��P	�������A�)*

	conv_loss=�s>,�7e        )��P	�+�����A�)*

	conv_loss�{m>�g+/        )��P	�`�����A�)*

	conv_loss��l>�Z|�        )��P	�������A�)*

	conv_loss@�k>*�>;        )��P	�������A�)*

	conv_loss��{>�\��        )��P	�������A�)*

	conv_loss.Qt>��.O        )��P	�/�����A�)*

	conv_loss�Vw>&ԅM        )��P	�c�����A�)*

	conv_loss��t>�ԛS        )��P	U������A�)*

	conv_lossj�l>ѣӥ        )��P	C������A�)*

	conv_loss��o>	�}�        )��P	9������A�)*

	conv_loss��j>���p        )��P	0�����A�)*

	conv_loss�+{>V�u�        )��P	�b�����A�)*

	conv_loss�p>��Ze        )��P	S������A�)*

	conv_lossJ�k>h�PJ        )��P	Q������A�)*

	conv_loss��o>V�T        )��P	������A�)*

	conv_loss��y>��@�        )��P	�F�����A�)*

	conv_loss�fm>����        )��P	�y�����A�)*

	conv_loss�Ks>ZV�!        )��P	:������A�)*

	conv_loss��g>��nl        )��P	������A�)*

	conv_loss��g>�D��        )��P	������A�)*

	conv_loss_q>El.�        )��P	�R�����A�)*

	conv_loss�n>�Cg
        )��P	�������A�)*

	conv_lossB�g>e�F        )��P	)������A�)*

	conv_loss9z>�Bw�        )��P	�������A�)*

	conv_lossR#m>�#1        )��P	�"�����A�)*

	conv_loss�te>�d�R        )��P	�V�����A�)*

	conv_loss��v>�y�K        )��P	~������A�)*

	conv_loss��n>Z��        )��P	ľ�����A�)*

	conv_loss�_n>u��y        )��P	�������A�)*

	conv_lossO�g>5� �        )��P	-�����A�)*

	conv_lossO�q>���        )��P	�d�����A�)*

	conv_loss�p>�}��        )��P	 ������A�)*

	conv_loss�i>@&9        )��P	C������A�)*

	conv_lossyd>���        )��P	,������A�)*

	conv_loss=c>�+nG        )��P	A3�����A�)*

	conv_loss�=l>'d��        )��P	�e�����A�)*

	conv_loss�
k>�/        )��P	~������A�)*

	conv_loss��p>�g�x        )��P	�������A�)*

	conv_lossib>*��K        )��P	�������A�)*

	conv_lossui>ӿX        )��P	2�����A�)*

	conv_loss��o>C�        )��P	�d�����A�)*

	conv_loss�v>���        )��P	ƛ�����A�)*

	conv_loss\r>W��        )��P	f������A�)*

	conv_loss�j>�M?M        )��P	������A�)*

	conv_losse�i>���        )��P	�6�����A�)*

	conv_loss�p>����        )��P	5h�����A�)*

	conv_lossC�i><�        )��P	������A�)*

	conv_loss�hk>�7�<        )��P	�������A�)*

	conv_lossY�y>Ï$        )��P	�������A�)*

	conv_loss~iv>g\�r        )��P	�1�����A�)*

	conv_loss]o>=M�        )��P	�c�����A�)*

	conv_lossPhr>'OyZ        )��P	�������A�)*

	conv_loss��r>*Q�        )��P	�������A�)*

	conv_loss,-i>*L�,        )��P	,������A�)*

	conv_loss��s>��(:        )��P	�3�����A�)*

	conv_loss��o>aś/        )��P	d�����A�)*

	conv_loss�i>
��	        )��P	ז�����A�)*

	conv_lossq>���        )��P	s������A�)*

	conv_loss��p>����        )��P	�������A�)*

	conv_loss�@d>8 b        )��P	�.�����A�)*

	conv_lossG�w>z��        )��P	sa�����A�)*

	conv_loss@h>![        )��P	h������A�)*

	conv_loss��o>#8m�        )��P	�������A�)*

	conv_lossp�k>�~�        )��P	������A�)*

	conv_loss�F]>5-�.        )��P	a@�����A�)*

	conv_loss]#m>f�@        )��P	Ut�����A�)*

	conv_loss�_\>���        )��P	9������A�)*

	conv_loss�Dn>x��        )��P	�������A�)*

	conv_loss��_>�� �        )��P	n�����A�)*

	conv_lossՇq>8��        )��P	P�����A�)*

	conv_losspa>��9        )��P	6������A�)*

	conv_loss�n>Y��4        )��P	�������A�)*

	conv_loss�p>Յ�        )��P	�������A�)*

	conv_lossbo>���        )��P	�/�����A�)*

	conv_loss�l>зz        )��P	,d�����A�)*

	conv_loss�^^>z�        )��P	-������A�)*

	conv_loss�,k>f��        )��P	F������A�)*

	conv_lossr�Z>v���        )��P	�����A�)*

	conv_lossJ�h>g��'        )��P	�=�����A�)*

	conv_loss�pn>O���        )��P	*v�����A�)*

	conv_lossO�T>Nȋ        )��P	R������A�)*

	conv_loss��j>�\.`        )��P	P������A�)*

	conv_loss�Y>SS�        )��P	�����A�)*

	conv_loss��k>*?y�        )��P	�F�����A�)*

	conv_loss��^>랓        )��P	�x�����A�)*

	conv_loss�h`>D�\�        )��P	ī�����A�)*

	conv_loss�_>�y�f        )��P	I������A�)*

	conv_loss�q>AW=D        )��P	������A�**

	conv_losss�Y>�]=�        )��P	�D�����A�**

	conv_lossv�^>/n��        )��P	@x�����A�**

	conv_loss�ca>���        )��P	|������A�**

	conv_lossja>��)�        )��P	������A�**

	conv_lossB2`>7��b        )��P	������A�**

	conv_loss�W>[ތ        )��P	�K�����A�**

	conv_loss��_>
þ        )��P	������A�**

	conv_loss+�k>���        )��P	4������A�**

	conv_loss��`>�sh        )��P	)������A�**

	conv_loss�u>Ɂ��        )��P	"�����A�**

	conv_loss}�a>E�-        )��P	MV�����A�**

	conv_loss�=^>�s��        )��P	ފ�����A�**

	conv_loss��f>{�\�        )��P	�������A�**

	conv_loss� c>��        )��P	�������A�**

	conv_loss�i>�7��        )��P	~�����A�**

	conv_loss�c\>9�Sg        )��P	K�����A�**

	conv_losshGi>]�M�        )��P	G{�����A�**

	conv_lossʭ\>����        )��P	/������A�**

	conv_loss�qZ>���        )��P	�������A�**

	conv_loss��`>?=z�        )��P	D�����A�**

	conv_loss>�]>ԟ�        )��P	�F�����A�**

	conv_loss1Pa>a�        )��P	Vu�����A�**

	conv_loss�c>#$T        )��P	/������A�**

	conv_loss�XU>��a�        )��P	�������A�**

	conv_loss�g>�a��        )��P	ff�����A�**

	conv_loss|Y>�        )��P	@������A�**

	conv_loss�W`>{!w�        )��P	�������A�**

	conv_loss	�G> ;�        )��P	������A�**

	conv_loss�iP>���        )��P	�#�����A�**

	conv_loss:	a>�%q*        )��P	�Y�����A�**

	conv_loss��P>5r��        )��P	D������A�**

	conv_loss��]>���        )��P	T������A�**

	conv_lossQ=S>ܞ6        )��P	�������A�**

	conv_lossi�b>c�O�        )��P	�&�����A�**

	conv_lossj�X>+��x        )��P	�[�����A�**

	conv_loss�Qp>e        )��P	������A�**

	conv_loss�B\>/
�u        )��P	l������A�**

	conv_lossֲ`>�H{�        )��P	�������A�**

	conv_lossdcb>JR��        )��P	������A�**

	conv_lossL�_>M<T        )��P	]U�����A�**

	conv_loss	W>��/q        )��P	�������A�**

	conv_loss$][>����        )��P	`������A�**

	conv_loss�.\>G�8�        )��P	�������A�**

	conv_loss��e>�}�d        )��P	������A�**

	conv_lossQ?a>9Q�R        )��P	 D�����A�**

	conv_lossC-^>�[�        )��P	�u�����A�**

	conv_loss�^>Z��c        )��P	�������A�**

	conv_loss��U>�柩        )��P	h������A�**

	conv_loss/v`>RqM#        )��P	_�����A�**

	conv_loss��N>!8�`        )��P	^I�����A�**

	conv_loss�0h>S v        )��P	�w�����A�**

	conv_loss�XX>���/        )��P	 ������A�**

	conv_loss�\>w��        )��P	������A�**

	conv_loss�_>��3�        )��P	������A�**

	conv_loss�3]>����        )��P	C7�����A�**

	conv_loss�|_>~}�o        )��P	Yh�����A�**

	conv_lossu__>�0b5        )��P	�������A�**

	conv_loss��Q>�'��        )��P	�������A�**

	conv_loss�eV>����        )��P	1������A�**

	conv_lossuY>�t:�        )��P	b#�����A�**

	conv_lossZ}R>b�k�        )��P	^T�����A�**

	conv_loss˔U>���
        )��P	҂�����A�**

	conv_lossCCZ>�$�r        )��P	�������A�**

	conv_loss9�V>¶�%        )��P	�������A�**

	conv_lossPI>0��        )��P	������A�**

	conv_loss�i^>%��        )��P	5B�����A�**

	conv_loss�G\>��)c        )��P	�o�����A�**

	conv_loss��M>�I�        )��P	�������A�**

	conv_loss��V>q�        )��P	�������A�**

	conv_loss��\>��#I        )��P	�������A�**

	conv_loss-�[>&�6�        )��P	�-�����A�**

	conv_loss�RF>1�Qd        )��P	�^�����A�**

	conv_loss�oV>>��        )��P	�������A�**

	conv_loss�xS>O�-�        )��P	H������A�**

	conv_loss�pY>����        )��P	�������A�**

	conv_loss�^>�Gah        )��P	M,�����A�**

	conv_loss|\>�h�        )��P	PZ�����A�**

	conv_loss|a>���V        )��P	������A�**

	conv_loss�2V>���v        )��P	^������A�**

	conv_loss�xX>F�I�        )��P	�������A�**

	conv_loss��P>-Jɿ        )��P	������A�**

	conv_loss�!^>�!��        )��P	rQ�����A�**

	conv_loss� X>_Y0L        )��P	�������A�**

	conv_losselU>���        )��P	S������A�**

	conv_loss7\]>A�<@        )��P	v������A�**

	conv_loss�CQ>;z��        )��P	D#�����A�**

	conv_loss�,W>&�L/        )��P	9X�����A�**

	conv_loss�LP>N��        )��P	ܕ�����A�**

	conv_loss�V>����        )��P	�������A�**

	conv_lossJ�O>(�i        )��P	u������A�**

	conv_loss*�O>�t��        )��P	*%�����A�**

	conv_loss��X>��s        )��P	W�����A�**

	conv_loss�yW>ބ        )��P	Q������A�**

	conv_lossH}S>��h�        )��P	������A�**

	conv_loss]>O�4D        )��P	-������A�**

	conv_lossi�O>�Ao�        )��P	������A�**

	conv_loss�S>�^kJ        )��P	�D�����A�**

	conv_loss�#\>g��        )��P	�u�����A�**

	conv_lossJ_>�$p_        )��P	6������A�**

	conv_loss��S>z,��        )��P	������A�**

	conv_loss��U>�#�        )��P	������A�**

	conv_loss2�Q>tku        )��P	�8�����A�**

	conv_loss�@X>��S        )��P	�i�����A�**

	conv_loss��_>?�~u        )��P	�������A�**

	conv_lossX>�&�        )��P	�������A�**

	conv_loss��V>��Z�        )��P	�������A�**

	conv_lossI�I>��u        )��P	)�����A�**

	conv_lossjQ>�T|6        )��P	Y�����A�**

	conv_loss+�Z>�"n        )��P	ʈ�����A�**

	conv_lossf�W>u��        )��P	�������A�**

	conv_lossYW>��w�        )��P	N������A�**

	conv_loss_�W>�7�        )��P	5�����A�**

	conv_loss��S>���        )��P	�M�����A�**

	conv_loss�vI>W���        )��P	)|�����A�**

	conv_loss��>>���        )��P	�������A�**

	conv_loss��I>���        )��P	F������A�**

	conv_loss��Y>z��#        )��P	�	�����A�**

	conv_loss� Y> S�z        )��P	t<�����A�**

	conv_lossmF>m��        )��P	�k�����A�**

	conv_lossj�S>���=        )��P	[������A�**

	conv_lossJ&[>N��        )��P	�������A�**

	conv_loss�C>+��v        )��P	�������A�**

	conv_loss B>�v�]        )��P	�,�����A�**

	conv_loss�@H>Q�`�        )��P	������A�**

	conv_lossXB>��          )��P	-������A�**

	conv_loss�2M>�F�        )��P	;�����A�**

	conv_loss��V>���        )��P	F�����A�**

	conv_loss��Q>r"��        )��P	�t�����A�+*

	conv_loss��a>�T��        )��P	�������A�+*

	conv_loss|HM>7F��        )��P	/������A�+*

	conv_loss��J>J���        )��P	N�����A�+*

	conv_lossgLK>�v"w        )��P	�<�����A�+*

	conv_lossJ�P>ϴ�L        )��P	�m�����A�+*

	conv_loss?"O>^���        )��P	�������A�+*

	conv_lossr�G>n���        )��P	������A�+*

	conv_loss��K>�7��        )��P	������A�+*

	conv_loss�A>#���        )��P	�6�����A�+*

	conv_loss�6<>v���        )��P	f�����A�+*

	conv_loss9T>�j�        )��P	�������A�+*

	conv_loss��L>R|h[        )��P	�������A�+*

	conv_loss�R>���        )��P	�������A�+*

	conv_loss�C^>�/�n        )��P	�-�����A�+*

	conv_lossE�H>L�A        )��P	�a�����A�+*

	conv_loss��U>H�Z        )��P	Z������A�+*

	conv_loss�xA>{q�        )��P	�������A�+*

	conv_loss�r@>+`�        )��P	b������A�+*

	conv_loss��V>���        )��P	�'�����A�+*

	conv_loss�=Q>1u��        )��P	6Y�����A�+*

	conv_loss�rQ>&�o        )��P	������A�+*

	conv_loss�~H>��A�        )��P	�������A�+*

	conv_lossςI>��6f        )��P	�������A�+*

	conv_lossGG>^N��        )��P	h �����A�+*

	conv_loss7I>)��        )��P	Q�����A�+*

	conv_loss�MP>{Q�        )��P	F������A�+*

	conv_lossXdP>G!�        )��P	&������A�+*

	conv_loss��>>����        )��P	�������A�+*

	conv_lossY�X>�t        )��P	�!�����A�+*

	conv_loss�W>W54U        )��P	�R�����A�+*

	conv_loss��B>��ѻ        )��P	{������A�+*

	conv_loss'�N>Q�#        )��P	�������A�+*

	conv_loss��J>hB��        )��P	�������A�+*

	conv_lossJ�.>l=>        )��P	������A�+*

	conv_losss�J>T+�        )��P	�>�����A�+*

	conv_lossP�H>x��        )��P	n�����A�+*

	conv_loss/�N>/j�|        )��P	�������A�+*

	conv_lossc@R>�q=        )��P	u������A�+*

	conv_loss��<>���/        )��P	�������A�+*

	conv_loss��K>�+/        )��P	�*�����A�+*

	conv_loss_9><�5>        )��P	cY�����A�+*

	conv_lossG R>���%        )��P	�������A�+*

	conv_lossj�D>����        )��P	�������A�+*

	conv_loss�Q>??s�        )��P	�������A�+*

	conv_lossۨ3>P�C�        )��P	�����A�+*

	conv_losslE>�ܑ�        )��P	�K�����A�+*

	conv_loss�>:> V�        )��P	�������A�+*

	conv_loss�:>���        )��P	r������A�+*

	conv_loss2�J>"K��        )��P	C������A�+*

	conv_loss\JG>���        )��P	9$�����A�+*

	conv_loss��=>0�)�        )��P	*V�����A�+*

	conv_loss�J>�.'g        )��P	�������A�+*

	conv_loss�uC>�t        )��P	������A�+*

	conv_loss��L>pD1        )��P	3������A�+*

	conv_loss�S>�Y)        )��P	;�����A�+*

	conv_loss|"E>vx�p        )��P	�m�����A�+*

	conv_loss��C>¹1�        )��P	H������A�+*

	conv_loss�?>ЌW�        )��P	�������A�+*

	conv_loss�=>�I��        )��P	�������A�+*

	conv_lossqq8>]�        )��P	:�����A�+*

	conv_loss�J>�$~        )��P	[o�����A�+*

	conv_lossW�;>OX        )��P	[������A�+*

	conv_loss��Q>ӊ8�        )��P	r������A�+*

	conv_lossylU>[�~�        )��P	������A�+*

	conv_lossחN>�8\}        )��P	sF�����A�+*

	conv_loss�P;>�.j�        )��P	Ɂ�����A�+*

	conv_loss�?>*獆        )��P	�������A�+*

	conv_loss��P>�%�V        )��P	H������A�+*

	conv_loss�e?>a��~        )��P	������A�+*

	conv_loss8pG>�Y�7        )��P	.I�����A�+*

	conv_lossu�E><���        )��P	z�����A�+*

	conv_loss��Z>�4^        )��P	ݪ�����A�+*

	conv_loss�_@>�$lL        )��P	�������A�+*

	conv_lossH<O>�jq�        )��P	������A�+*

	conv_loss��F>+0X�        )��P	B>�����A�+*

	conv_loss\�?>�Ti�        )��P	o�����A�+*

	conv_loss$�E>�=        )��P	|������A�+*

	conv_loss[�G>�}�T        )��P	}������A�+*

	conv_lossNF>{H�        )��P	 ��A�+*

	conv_loss��G>#-�:        )��P	,7 ��A�+*

	conv_lossyY>>�^Z        )��P	�g ��A�+*

	conv_loss��?>��Ӊ        )��P	 � ��A�+*

	conv_loss]�F>u���        )��P	�� ��A�+*

	conv_loss�->F�K        )��P	�� ��A�+*

	conv_loss��2>�v|        )��P	�*��A�+*

	conv_loss��W>�Sk	        )��P	Z��A�+*

	conv_loss�a3>�@W        )��P	���A�+*

	conv_loss��L>�j��        )��P	����A�+*

	conv_lossb�9>�Ds        )��P	����A�+*

	conv_loss��N>c���        )��P	��A�+*

	conv_loss�F;>F ��        )��P	�M��A�+*

	conv_loss�67>I6�        )��P	s��A�+*

	conv_loss�$K>
��        )��P	l���A�+*

	conv_loss�8>�[��        )��P	I���A�+*

	conv_loss�?C>2!�         )��P	���A�+*

	conv_loss(�L>�my        )��P	zF��A�+*

	conv_loss}FC>B�Σ        )��P	����A�+*

	conv_loss�NH>ޡ�        )��P	���A�+*

	conv_loss]�7>�Ҥ�        )��P	�>��A�+*

	conv_loss��@>���n        )��P	ds��A�+*

	conv_loss��;>�"�        )��P	l���A�+*

	conv_loss��J>ˆ��        )��P	n���A�+*

	conv_loss4e9>�
T0        )��P	d	��A�+*

	conv_loss[#4>���        )��P	V;��A�+*

	conv_loss�iC>���        )��P	�x��A�+*

	conv_loss��L>�O<        )��P	����A�+*

	conv_lossF�J>�,~        )��P	����A�+*

	conv_loss�O>���        )��P	���A�+*

	conv_loss�I>.�m�        )��P	 I��A�+*

	conv_loss�b->��(u        )��P	����A�+*

	conv_losst9>~-��        )��P	v���A�+*

	conv_loss�k4>�!��        )��P	����A�+*

	conv_lossJ�L>q�        )��P	9!��A�+*

	conv_loss��L>�C#;        )��P	�P��A�+*

	conv_loss��U>��j�        )��P	l���A�+*

	conv_loss/�G>�Ƿ�        )��P	���A�+*

	conv_lossoh:>~'W        )��P	(���A�+*

	conv_loss�A8>(�+        )��P	�	��A�+*

	conv_lossɮ1>�=�        )��P	KE	��A�+*

	conv_loss-8>��&�        )��P	�v	��A�+*

	conv_loss�f7>oZm�        )��P	*�	��A�+*

	conv_loss\�)>,���        )��P	&�	��A�+*

	conv_loss�H7>H.        )��P	�	
��A�+*

	conv_lossEe?>U��[        )��P	�:
��A�+*

	conv_loss&�<>�9�        )��P	j
��A�+*

	conv_loss��/>�dbc        )��P	��
��A�+*

	conv_loss(5=>��;(        )��P	��
��A�+*

	conv_lossaJ>r�K�        )��P	��
��A�,*

	conv_lossoy>>�*�h        )��P	�0��A�,*

	conv_losstD>�&=@        )��P	na��A�,*

	conv_loss��<>�!u        )��P	6���A�,*

	conv_lossX�1>���        )��P	���A�,*

	conv_lossJ�->�J9        )��P	����A�,*

	conv_loss�@>����        )��P	*%��A�,*

	conv_loss�A>�˨,        )��P	.U��A�,*

	conv_loss�?>G�Q        )��P	>���A�,*

	conv_loss&�A>��        )��P	J���A�,*

	conv_loss��>>���e        )��P	*���A�,*

	conv_loss�\H>%�!M        )��P	���A�,*

	conv_loss��:>�v�        )��P	gL��A�,*

	conv_lossD�8>��#�        )��P	���A�,*

	conv_lossq�=>˹�|        )��P	:���A�,*

	conv_lossB>L��!        )��P	����A�,*

	conv_loss/oC>����        )��P	B��A�,*

	conv_loss�J/>���        )��P	:A��A�,*

	conv_loss�:>>��        )��P	cp��A�,*

	conv_loss�HB>7m^        )��P	����A�,*

	conv_lossd�1>��        )��P	����A�,*

	conv_loss�1>���        )��P	���A�,*

	conv_loss�L>�&�        )��P	�@��A�,*

	conv_lossg�6>{|��        )��P	x��A�,*

	conv_loss�l1>�`\�        )��P	'���A�,*

	conv_loss��4>Q��2        )��P	s���A�,*

	conv_lossCW<>��S[        )��P	���A�,*

	conv_lossh�?>^]��        )��P	]P��A�,*

	conv_lossj?>��n        )��P	����A�,*

	conv_loss��:>��Z�        )��P	ǽ��A�,*

	conv_loss=1>�:Ǔ        )��P	����A�,*

	conv_losso<>��r;        )��P	���A�,*

	conv_loss�j0>�        )��P	�S��A�,*

	conv_loss�?>�?<�        )��P	���A�,*

	conv_loss��4>>}��        )��P	����A�,*

	conv_loss�z<>��@        )��P	7���A�,*

	conv_lossd9>��        )��P	� ��A�,*

	conv_loss��=>�w��        )��P	�T��A�,*

	conv_loss"?>��d�        )��P	1���A�,*

	conv_loss��:>̹)+        )��P	ǵ��A�,*

	conv_loss��7>�B�Q        )��P	H���A�,*

	conv_loss��(>٢1        )��P	��A�,*

	conv_loss�!3>i-!        )��P	G��A�,*

	conv_lossZ�6>w��	        )��P	%w��A�,*

	conv_loss};>r�F�        )��P	~���A�,*

	conv_loss��L>�on�        )��P	A���A�,*

	conv_loss_�=>�O.�        )��P	���A�,*

	conv_loss��7>��        )��P	�;��A�,*

	conv_loss��6>�ǰ        )��P	kj��A�,*

	conv_loss��,>��=F        )��P	����A�,*

	conv_loss�<+>g�7�        )��P	����A�,*

	conv_loss�A)>�2�        )��P	����A�,*

	conv_loss�J0>��]        )��P	|.��A�,*

	conv_loss<.->�4}�        )��P	Ja��A�,*

	conv_loss�D>Q��X        )��P	����A�,*

	conv_loss�h#>A��        )��P	 ���A�,*

	conv_loss�.>�k	Q        )��P	����A�,*

	conv_losse7>Sy        )��P	q&��A�,*

	conv_lossj%>��        )��P	!V��A�,*

	conv_loss�T0>�%T        )��P	3���A�,*

	conv_loss>�:>��        )��P	߶��A�,*

	conv_loss��9>n�+�        )��P	����A�,*

	conv_lossǀ >�4%o        )��P	8��A�,*

	conv_lossct2>S�6[        )��P	�J��A�,*

	conv_loss�u/>EI        )��P	�}��A�,*

	conv_loss(QD>{���        )��P	����A�,*

	conv_lossk F>�+�        )��P	T���A�,*

	conv_lossK+;>y��        )��P	��A�,*

	conv_loss%:5>��         )��P	�D��A�,*

	conv_loss�0>�s�#        )��P	���A�,*

	conv_loss`;>)Bi        )��P	0���A�,*

	conv_lossjU/>��;�        )��P	���A�,*

	conv_loss��@>=�3�        )��P	��A�,*

	conv_loss� >��ZG        )��P	N��A�,*

	conv_loss�(>��Jm        )��P	�~��A�,*

	conv_loss��B>yv7�        )��P	#���A�,*

	conv_loss��!>�y��        )��P	����A�,*

	conv_loss8�)>����        )��P	&��A�,*

	conv_loss��,>F��        )��P	-T��A�,*

	conv_loss:�H>�|        )��P	���A�,*

	conv_loss�9>�m        )��P	C���A�,*

	conv_loss!�8>�df        )��P	l���A�,*

	conv_loss��<>�
��        )��P	g+��A�,*

	conv_loss=�F>2���        )��P	t_��A�,*

	conv_lossc�1>|z        )��P	����A�,*

	conv_loss&8>�o4�        )��P	���A�,*

	conv_loss�3>���\        )��P	����A�,*

	conv_loss=5>��0G        )��P	�0��A�,*

	conv_loss�4>�(S        )��P	�a��A�,*

	conv_loss{P7>����        )��P	7���A�,*

	conv_loss)Z3>?읫        )��P	����A�,*

	conv_loss��C>��\         )��P	����A�,*

	conv_loss��4>�)"<        )��P	�0��A�,*

	conv_loss__/>��==        )��P	{`��A�,*

	conv_lossN?>w��        )��P	}���A�,*

	conv_loss��6>�.c        )��P	����A�,*

	conv_loss�y2>V��u        )��P	���A�,*

	conv_loss}f&>d1'�        )��P	�5��A�,*

	conv_lossi�+>9�y�        )��P	?k��A�,*

	conv_lossՅ@>��        )��P	I���A�,*

	conv_loss�;>�@�        )��P	����A�,*

	conv_loss��->���U        )��P	����A�,*

	conv_lossd]9>����        )��P	�2��A�,*

	conv_loss��.>|�;        )��P	�b��A�,*

	conv_loss��'>v�        )��P	*���A�,*

	conv_loss��$>Ps�        )��P	(���A�,*

	conv_loss�)>�N�        )��P	����A�,*

	conv_loss1S=>����        )��P	, ��A�,*

	conv_loss
N6>@�K�        )��P	�] ��A�,*

	conv_lossI�9>��Z        )��P	� ��A�,*

	conv_loss�t<>��Uu        )��P	� ��A�,*

	conv_loss��J>���        )��P	r� ��A�,*

	conv_loss�<>=��        )��P	I"!��A�,*

	conv_loss��0>B�V�        )��P	MS!��A�,*

	conv_loss�8>Sw�        )��P	��!��A�,*

	conv_loss�B;>��d�        )��P	W�!��A�,*

	conv_loss<�&>*�R�        )��P	��!��A�,*

	conv_loss�D<>:��        )��P	m"��A�,*

	conv_lossJ�4>+��&        )��P	�I"��A�,*

	conv_loss�9>=�C        )��P	��"��A�,*

	conv_loss�P3>0��        )��P		�"��A�,*

	conv_loss��+>'��        )��P	E�"��A�,*

	conv_loss�(<>��        )��P	)#��A�,*

	conv_loss��+>h9Y        )��P	�[#��A�,*

	conv_loss�->�M�e        )��P	Ǎ#��A�,*

	conv_loss��%>�i�A        )��P	��#��A�,*

	conv_lossJQ0>xXF        )��P	��#��A�,*

	conv_loss��/>�tl�        )��P	d2$��A�,*

	conv_loss77+>	G��        )��P	Kg$��A�-*

	conv_loss�� >����        )��P	��$��A�-*

	conv_loss�I$>}3�        )��P	��$��A�-*

	conv_loss#�/>O+;        )��P	��$��A�-*

	conv_lossv!>D���        )��P	5%��A�-*

	conv_loss��">����        )��P	�d%��A�-*

	conv_lossQ,>�f@�        )��P	�%��A�-*

	conv_loss��)>����        )��P	E�%��A�-*

	conv_loss_0>j��        )��P	& &��A�-*

	conv_loss�3>�n�        )��P	?2&��A�-*

	conv_loss�h$>Dyi        )��P	�c&��A�-*

	conv_loss�2>̋�        )��P	�&��A�-*

	conv_loss((>��        )��P	k�&��A�-*

	conv_loss��)>�u��        )��P	�&��A�-*

	conv_loss�c3>Bvr        )��P	','��A�-*

	conv_loss��;>�Mu        )��P	0]'��A�-*

	conv_loss��)>2y�        )��P	��'��A�-*

	conv_loss!�,>���        )��P	��'��A�-*

	conv_losssG)>��        )��P	��'��A�-*

	conv_loss��0>Y7��        )��P	�!(��A�-*

	conv_loss3e>���H        )��P	�T(��A�-*

	conv_loss�5&>h�D�        )��P	�(��A�-*

	conv_loss��>�q�        )��P	Y�(��A�-*

	conv_loss�?B>+ʌ�        )��P		�(��A�-*

	conv_lossB3>Ն7�        )��P	e)��A�-*

	conv_loss*�0>>Z�)        )��P	�J)��A�-*

	conv_loss�=(>RH�        )��P	~z)��A�-*

	conv_loss.>CI        )��P	��)��A�-*

	conv_loss�@*>�2s        )��P	W�)��A�-*

	conv_lossw/>Q��        )��P	�!*��A�-*

	conv_loss��>�<        )��P	�S*��A�-*

	conv_loss�+>���        )��P	 �*��A�-*

	conv_loss�>/J�        )��P	��*��A�-*

	conv_lossy�'>��K�        )��P	<�*��A�-*

	conv_loss�$>��FC        )��P	G+��A�-*

	conv_lossA�5>�`�        )��P	�M+��A�-*

	conv_loss�>��J<        )��P	Ɓ+��A�-*

	conv_loss8�'>�x�        )��P	�+��A�-*

	conv_loss��%>��Y�        )��P	4�+��A�-*

	conv_lossY�>��        )��P	�,��A�-*

	conv_loss)>e� �        )��P	�E,��A�-*

	conv_loss�!4>{�#�        )��P	;�-��A�-*

	conv_loss��>�]��        )��P	`.��A�-*

	conv_loss��(>���        )��P	!>.��A�-*

	conv_loss��->���t        )��P	or.��A�-*

	conv_loss@�6>���        )��P	��.��A�-*

	conv_loss��>V��        )��P	��.��A�-*

	conv_loss��>�k�        )��P	T/��A�-*

	conv_loss%-$>(�        )��P	�8/��A�-*

	conv_loss� >�g��        )��P	0/��A�-*

	conv_loss��>P�#        )��P	'�/��A�-*

	conv_lossC�>|鐏        )��P	��/��A�-*

	conv_loss<�>���^        )��P	�+0��A�-*

	conv_loss�>�$�-        )��P	J^0��A�-*

	conv_loss��3>f�Wn        )��P	i�0��A�-*

	conv_loss�<>�?�        )��P	��0��A�-*

	conv_loss�
)>�
��        )��P	�1��A�-*

	conv_loss��1>��
        )��P	%21��A�-*

	conv_lossu(%>l.        )��P	�d1��A�-*

	conv_lossn�.>I�Ź        )��P	O�1��A�-*

	conv_loss��>��g~        )��P	��1��A�-*

	conv_loss�)>j�V"        )��P	A2��A�-*

	conv_loss�]>�d�/        )��P	�72��A�-*

	conv_loss�1$>M ;�        )��P	-m2��A�-*

	conv_loss��$> �        )��P	á2��A�-*

	conv_loss/c>��H�        )��P	��2��A�-*

	conv_loss�&>���        )��P	�3��A�-*

	conv_loss@7>]�_        )��P	�93��A�-*

	conv_loss�V6>pU        )��P	j3��A�-*

	conv_loss�F(>D`�         )��P	�3��A�-*

	conv_loss2*>d:fR        )��P	��3��A�-*

	conv_loss��%>�9"�        )��P	Z4��A�-*

	conv_loss�6>��`        )��P	�64��A�-*

	conv_lossQ�&>Â�J        )��P	�i4��A�-*

	conv_lossF}*>��        )��P	`�4��A�-*

	conv_loss �'>A��        )��P	�4��A�-*

	conv_loss?k/>,�W        )��P	��4��A�-*

	conv_loss�;9>@['�        )��P	X15��A�-*

	conv_losss�>�        )��P	�b5��A�-*

	conv_lossS%&>p�x�        )��P	�5��A�-*

	conv_loss/�(>oZ        )��P	J�5��A�-*

	conv_loss��">��4        )��P	�5��A�-*

	conv_loss��>D�uH        )��P	Y,6��A�-*

	conv_lossCA>,�%�        )��P	�]6��A�-*

	conv_lossߒ>��        )��P	��6��A�-*

	conv_loss��3>KA\        )��P	��6��A�-*

	conv_loss��&>;M
�        )��P	9�6��A�-*

	conv_loss�">�զ�        )��P	K"7��A�-*

	conv_loss�8!>�n=N        )��P	rU7��A�-*

	conv_loss�U>S��i        )��P	�7��A�-*

	conv_lossz�0>��#n        )��P	�7��A�-*

	conv_loss%�%>�g�[        )��P	��7��A�-*

	conv_loss=�+>���        )��P	�/8��A�-*

	conv_loss��'>�;9        )��P	�b8��A�-*

	conv_loss�	)>h;`        )��P	4�8��A�-*

	conv_loss��#>W���        )��P	��8��A�-*

	conv_loss)'>��G        )��P	�9��A�-*

	conv_lossQ>��iA        )��P	�<9��A�-*

	conv_loss��:>���        )��P	�o9��A�-*

	conv_loss��/>1	j:        )��P	��9��A�-*

	conv_lossތ'>�{�M        )��P	��9��A�-*

	conv_lossf�>d}JY        )��P	�!:��A�-*

	conv_lossS�$>�"�k        )��P	FT:��A�-*

	conv_loss�F0>ʰ�        )��P	I�:��A�-*

	conv_loss�JA>�W�        )��P	�:��A�-*

	conv_lossX�+>[Ę�        )��P	
�:��A�-*

	conv_lossƗ>���]        )��P	$;��A�-*

	conv_loss��9>1t��        )��P	$U;��A�-*

	conv_loss!�>Q��;        )��P	%�;��A�-*

	conv_loss��.>�n*        )��P	չ;��A�-*

	conv_loss �->V�~        )��P	�;��A�-*

	conv_loss�~ >��        )��P	�!<��A�-*

	conv_loss��>��)        )��P	wW<��A�-*

	conv_loss�.>a� �        )��P	:�<��A�-*

	conv_loss:�>LO^N        )��P	u�<��A�-*

	conv_loss&�!>>
-�        )��P	��<��A�-*

	conv_loss��(>�)-�        )��P	�,=��A�-*

	conv_loss+�>'�K        )��P	H`=��A�-*

	conv_loss�e.>f��        )��P	�=��A�-*

	conv_loss+�>�Z�        )��P	��=��A�-*

	conv_lossl�>��        )��P	��=��A�-*

	conv_loss��>�Q��        )��P	 0>��A�-*

	conv_loss�j0>����        )��P	9b>��A�-*

	conv_loss�^0>(Y��        )��P	<�>��A�-*

	conv_lossj+>!!@        )��P	l�>��A�-*

	conv_loss��->��>�        )��P	��>��A�-*

	conv_loss#�/>�|��        )��P	t)?��A�-*

	conv_loss�J>��V�        )��P	�Z?��A�-*

	conv_loss�>�h�        )��P	`�?��A�.*

	conv_lossj	>1�"�        )��P	��?��A�.*

	conv_loss��,>����        )��P	R�?��A�.*

	conv_loss�d5>�#        )��P	C%@��A�.*

	conv_loss��>�N�Z        )��P	~V@��A�.*

	conv_loss�~'>��eU        )��P	
�@��A�.*

	conv_loss�M'>�%g        )��P	��@��A�.*

	conv_loss>�'>�0r�        )��P	�@��A�.*

	conv_loss��>�^�        )��P	�"A��A�.*

	conv_loss�r+> ��        )��P	{SA��A�.*

	conv_lossB�=>H�3;        )��P	s�A��A�.*

	conv_lossۊ)>vN+�        )��P	�A��A�.*

	conv_loss�&>_��k        )��P	��A��A�.*

	conv_loss�G>+���        )��P	�0B��A�.*

	conv_loss(�,>�[ry        )��P	/cB��A�.*

	conv_lossu>$+�4        )��P	�B��A�.*

	conv_loss��+>n���        )��P	��B��A�.*

	conv_loss!>�*�,        )��P	m�B��A�.*

	conv_loss	+>�SA'        )��P	�7C��A�.*

	conv_loss��!>`��v        )��P	�hC��A�.*

	conv_lossd�$>7��        )��P	X�C��A�.*

	conv_loss�Q6>N�I        )��P		�C��A�.*

	conv_loss�>	���        )��P	mD��A�.*

	conv_loss�6>�i�        )��P	<PD��A�.*

	conv_lossz>���        )��P	s�D��A�.*

	conv_loss�'>XlK        )��P	�D��A�.*

	conv_lossd�>{��7        )��P	��D��A�.*

	conv_loss�+>�)	        )��P	�E��A�.*

	conv_lossd�>},ݚ        )��P	�GE��A�.*

	conv_loss��>;@|:        )��P	�yE��A�.*

	conv_loss�>~�s        )��P	��E��A�.*

	conv_loss�>�O�a        )��P	��E��A�.*

	conv_loss��>G�S        )��P	�F��A�.*

	conv_loss"�&>��`        )��P	�GF��A�.*

	conv_loss7�4>x̰=        )��P	�F��A�.*

	conv_loss�9>�SW        )��P	H�F��A�.*

	conv_loss5>h�02        )��P	N�F��A�.*

	conv_loss�>bQ�%        )��P	G��A�.*

	conv_loss��	>�j�        )��P	�KG��A�.*

	conv_losso3&>���        )��P	P~G��A�.*

	conv_loss�C>ur�m        )��P	I�G��A�.*

	conv_loss��>B��        )��P	��G��A�.*

	conv_loss�>�>�        )��P	H��A�.*

	conv_loss��>��S�        )��P	FH��A�.*

	conv_lossG�)>�4 �        )��P	�xH��A�.*

	conv_lossB>:YO�        )��P	ĩH��A�.*

	conv_loss��> ���        )��P	��H��A�.*

	conv_loss�>)��        )��P	�I��A�.*

	conv_loss� 0>��5�        )��P	�CI��A�.*

	conv_loss->�S��        )��P	^xI��A�.*

	conv_loss>��%�        )��P	��I��A�.*

	conv_losse3%>�&>        )��P	�I��A�.*

	conv_loss��->����        )��P	�J��A�.*

	conv_loss��>(�J        )��P	�EJ��A�.*

	conv_loss�.&>)��        )��P	EJ��A�.*

	conv_loss9�>�u�r        )��P	<�J��A�.*

	conv_losso;>��        )��P	0�J��A�.*

	conv_lossA'>�O�        )��P	8K��A�.*

	conv_lossS�/><�[�        )��P	�GK��A�.*

	conv_loss�4&>�M��        )��P	3~K��A�.*

	conv_loss\�!>��.�        )��P	3�K��A�.*

	conv_loss��>f+�        )��P	a�K��A�.*

	conv_loss7�%>;߫�        )��P	�L��A�.*

	conv_loss�$>�J.0        )��P	p`L��A�.*

	conv_loss�!>���j        )��P	��L��A�.*

	conv_loss5(2>"�x�        )��P	{�L��A�.*

	conv_loss��>AV��        )��P	
�L��A�.*

	conv_loss��>����        )��P	p.M��A�.*

	conv_loss�c)>��e)        )��P	!bM��A�.*

	conv_lossA�>oVZ�        )��P	/�M��A�.*

	conv_loss0\>��@        )��P	��M��A�.*

	conv_loss�>��1        )��P	nN��A�.*

	conv_loss{m>g�q        )��P	nBN��A�.*

	conv_loss��>���        )��P	�vN��A�.*

	conv_loss7�>6��        )��P	��N��A�.*

	conv_loss��>�s�H        )��P	��N��A�.*

	conv_loss��>G�        )��P	!O��A�.*

	conv_loss� >��p�        )��P	4KO��A�.*

	conv_loss��">ը�I        )��P	�}O��A�.*

	conv_loss'� >�]��        )��P	"�O��A�.*

	conv_loss!�>�I�I        )��P	��O��A�.*

	conv_loss	�>S={{        )��P	�P��A�.*

	conv_lossI�
>zR9k        )��P	SFP��A�.*

	conv_loss=�7>�&��        )��P	XP��A�.*

	conv_loss[�*>?6;        )��P	m�P��A�.*

	conv_lossհ>E�rO        )��P	s�P��A�.*

	conv_loss�l>�ͱ        )��P	�Q��A�.*

	conv_loss/�>��;^        )��P	BOQ��A�.*

	conv_loss\�&>����        )��P	*�Q��A�.*

	conv_loss��)>��O~        )��P	״Q��A�.*

	conv_losssa>$ٵ        )��P	0�Q��A�.*

	conv_loss�|->���'        )��P	hR��A�.*

	conv_loss��>R�[m        )��P	�IR��A�.*

	conv_loss[*0>���        )��P	DzR��A�.*

	conv_lossZt>2���        )��P	��R��A�.*

	conv_loss�f >j�_        )��P	4�R��A�.*

	conv_loss�S>�ڤ�        )��P	'S��A�.*

	conv_lossW�#>f�        )��P	�AS��A�.*

	conv_lossl�*>qe��        )��P	.tS��A�.*

	conv_loss��!>�j�        )��P	��S��A�.*

	conv_lossv�*>�!Ӄ        )��P	H�S��A�.*

	conv_lossP4'>�=�        )��P	T��A�.*

	conv_loss5>��s�        )��P	M<T��A�.*

	conv_loss��>�sB        )��P	�mT��A�.*

	conv_loss]�>fu        )��P	֝T��A�.*

	conv_loss>����        )��P		�T��A�.*

	conv_lossXH>��߉        )��P	9U��A�.*

	conv_loss��>m��        )��P	�6U��A�.*

	conv_loss��>���        )��P	mU��A�.*

	conv_lossT`>�9b�        )��P	¡U��A�.*

	conv_loss�>�LpX        )��P	��U��A�.*

	conv_loss��!>� ��        )��P	�V��A�.*

	conv_loss��>1�