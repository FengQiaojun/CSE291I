       �K"	  @���Abrain.Event:2��T���      D(�	�h���A"��
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
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0* 
_class
loc:@conv2d/kernel
�
conv2d/kernel
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name * 
_class
loc:@conv2d/kernel
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
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
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
.conv2d_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *��:�*
dtype0
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
dtype0*
_output_shapes
:*
valueB"      
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
conv2d_3/kernel/AssignAssignconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_3/kernel*
validate_shape(*&
_output_shapes
:
�
conv2d_3/kernel/readIdentityconv2d_3/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_3/kernel
g
conv2d_4/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
conv2d_4/Conv2DConv2DRelu_2conv2d_3/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
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
conv2d_4/kernel/AssignAssignconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(
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
.conv2d_5/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel*
valueB
 *d�=*
dtype0
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
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_5/kernel
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
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_6/Conv2DConv2DRelu_4conv2d_5/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_6/kernel*&
_output_shapes
:
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
dtype0*
_output_shapes
:*
valueB"      
�
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*
T0*
data_formatNHWC*
strides
*
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
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
�
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes

:d
*
T0
�
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

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
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@dense_1/bias
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
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������
*
T0
h
logistic_loss/zeros_like	ZerosLikedense_2/BiasAdd*'
_output_shapes
:���������
*
T0
�
logistic_loss/GreaterEqualGreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:���������
*
T0
�
logistic_loss/SelectSelectlogistic_loss/GreaterEqualdense_2/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:���������
*
T0
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
logistic_loss/Log1pLog1plogistic_loss/Exp*'
_output_shapes
:���������
*
T0
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
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
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
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*
_output_shapes
:
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
out_type0*
_output_shapes
:*
T0
�
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:���������
*
T0
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
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_5'gradients/conv2d_7/Conv2D_grad/ShapeN:1gradients/Relu_6_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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
9gradients/conv2d_7/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_7/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_7/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
gradients/Relu_5_grad/ReluGradReluGrad7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyRelu_5*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_6/Conv2D_grad/ShapeNShapeNRelu_4conv2d_5/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/Relu_3_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides

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
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/Relu_2_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_3/Conv2D_grad/tuple/group_deps*&
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
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
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
data_formatNHWC*
strides
*
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
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
ף;*
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
: "��      ?�K	3�i���AJ��
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
T0*
data_formatNHWC*
strides
*
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
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
valueB
 *�[q>*
dtype0
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
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
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
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_2/kernel
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
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_3/kernel
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
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_4/kernel
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
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_5/Conv2DConv2DRelu_3conv2d_4/kernel/read*
data_formatNHWC*
strides
*
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
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
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
conv2d_6/Conv2DConv2DRelu_4conv2d_5/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

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
conv2d_7/Conv2DConv2DRelu_5conv2d_6/kernel/read*
T0*
data_formatNHWC*
strides
*
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
-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *�'o>
�
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes

:d
*

seed *
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0
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
dense_2/MatMulMatMulRelu_7dense_1/kernel/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������
*
T0
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
logistic_loss/mulMuldense_2/BiasAddPlaceholder_1*'
_output_shapes
:���������
*
T0
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
MeanMeanlogistic_lossConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
Tshape0*
_output_shapes

:*
T0
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
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
T0
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
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

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
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*'
_output_shapes
:���������
*
T0
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
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select*'
_output_shapes
:���������

�
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:���������
*
T0
�
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*'
_output_shapes
:���������
*
T0
�
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
N*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
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
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������d*
transpose_a( 
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
2gradients/conv2d_7/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_7/Conv2D_grad/ShapeNconv2d_6/kernel/readgradients/Relu_6_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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
gradients/Relu_5_grad/ReluGradReluGrad7gradients/conv2d_7/Conv2D_grad/tuple/control_dependencyRelu_5*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_6/Conv2D_grad/ShapeNShapeNRelu_4conv2d_5/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
3gradients/conv2d_6/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_4'gradients/conv2d_6/Conv2D_grad/ShapeN:1gradients/Relu_5_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
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
%gradients/conv2d_5/Conv2D_grad/ShapeNShapeNRelu_3conv2d_4/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
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
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/Relu_4_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
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
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/Relu_grad/ReluGrad*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/Relu_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
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
 *
ף;*
dtype0*
_output_shapes
: 
�
9GradientDescent/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelGradientDescent/learning_rate7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0* 
_class
loc:@conv2d/kernel
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
:GradientDescent/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelGradientDescent/learning_rate8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:d
*
use_locking( *
T0*!
_class
loc:@dense_1/kernel
�
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( 
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
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0ݝ\�       `/�#	������A*

	conv_lossVB1?p�	�       QKD	JI����A*

	conv_loss?1?OlH`       QKD	�y����A*

	conv_lossd?1?W�i�       QKD	������A*

	conv_lossg@1?�v       QKD	�ߚ���A*

	conv_loss=1?��^o       QKD	����A*

	conv_loss�>1?�[�       QKD	K@����A*

	conv_loss� 1?f���       QKD	n����A*

	conv_loss�1?�3       QKD	������A*

	conv_loss�1?
bC�       QKD	&����A	*

	conv_loss_1?w�Mv       QKD	a����A
*

	conv_loss�1?~��9       QKD	�M����A*

	conv_lossI1?��{�       QKD	�{����A*

	conv_lossj1?�xL�       QKD	������A*

	conv_loss�
1?�H��       QKD	i�����A*

	conv_loss��0?u�<%       QKD	�����A*

	conv_loss��0?�jY       QKD	<>����A*

	conv_loss��0?�mV       QKD	�w����A*

	conv_loss��0?;0�       QKD	[�����A*

	conv_loss��0?,BI�       QKD	sڝ���A*

	conv_lossJ�0?�,�Z       QKD	�����A*

	conv_loss�0?�X�       QKD	�A����A*

	conv_loss��0?@�nO       QKD	�q����A*

	conv_loss=�0?>�v       QKD	�����A*

	conv_loss��0?��X       QKD	w۞���A*

	conv_lossH�0?�       QKD	�����A*

	conv_loss#�0?$���       QKD	=����A*

	conv_lossz�0?g�@�       QKD	�n����A*

	conv_lossL�0?:j       QKD	ݝ����A*

	conv_lossz�0?G �[       QKD	�͟���A*

	conv_loss��0?�)b       QKD	������A*

	conv_loss��0?��G�       QKD	�,����A*

	conv_lossa�0?�O�       QKD	�\����A *

	conv_loss@�0?�2�1       QKD	Y�����A!*

	conv_loss�0?3{��       QKD	k�����A"*

	conv_loss��0?�5��       QKD	(����A#*

	conv_loss�0?'	F6       QKD	z����A$*

	conv_lossR�0?|�͎       QKD	eH����A%*

	conv_loss�u0?�6�O       QKD	Iw����A&*

	conv_lossv~0?���(       QKD	������A'*

	conv_loss�r0?�{s�       QKD	ԡ���A(*

	conv_loss j0?<�ۢ       QKD	�����A)*

	conv_loss�o0?���       QKD	C3����A**

	conv_loss�S0?�8�       QKD	�a����A+*

	conv_lossW0?�/       QKD	�����A,*

	conv_lossU0?�܂F       QKD	������A-*

	conv_lossbb0?@��e       QKD	�����A.*

	conv_loss�P0?5M��       QKD	�����A/*

	conv_loss�R0?&�$       QKD	O����A0*

	conv_loss�@0?��c�       QKD	�����A1*

	conv_loss�70?�m�       QKD	�����A2*

	conv_loss�40?�`       QKD	�����A3*

	conv_loss�+0?<�        QKD	b!����A4*

	conv_loss@)0?E鿍       QKD	&T����A5*

	conv_loss�$0?PA�w       QKD	k�����A6*

	conv_loss�0?' �~       QKD	۾����A7*

	conv_loss�0?���<       QKD	�����A8*

	conv_loss�0?38}       QKD	� ����A9*

	conv_loss�0?�c	N       QKD	P����A:*

	conv_lossS0?���       QKD	J�����A;*

	conv_loss�0?��I�       QKD	������A<*

	conv_loss[0?�C�@       QKD	������A=*

	conv_loss��/? 1v�       QKD	�"����A>*

	conv_lossU�/?}pQ�       QKD	�W����A?*

	conv_loss��/?��آ       QKD	C�����A@*

	conv_loss��/?�i�2       QKD	黦���AA*

	conv_loss��/?��       QKD	������AB*

	conv_loss�/?8 ��       QKD	)����AC*

	conv_loss��/?�Wc3       QKD	Y����AD*

	conv_lossR�/?�`#�       QKD		�����AE*

	conv_loss�/?�3��       QKD	�����AF*

	conv_loss'�/?�j�       QKD	
����AG*

	conv_lossc�/?�Cqq       QKD	�����AH*

	conv_loss��/?���       QKD	�E����AI*

	conv_loss޾/?�pH       QKD	w����AJ*

	conv_lossͳ/?�טb       QKD	+�����AK*

	conv_loss�/?3���       QKD	6�����AL*

	conv_loss%�/?V07       QKD	'����AM*

	conv_loss|�/?�B�       QKD	�P����AN*

	conv_loss�/?e)!       QKD	〩���AO*

	conv_loss�/?@V��       QKD	_�����AP*

	conv_loss��/?��D`       QKD	�����AQ*

	conv_lossp�/?���       QKD	>����AR*

	conv_loss�/?��/       QKD	qA����AS*

	conv_loss��/?��~�       QKD	"p����AT*

	conv_lossp|/?,��       QKD	S�����AU*

	conv_loss+q/?7�j$       QKD	VҪ���AV*

	conv_loss�k/?eq��       QKD	B����AW*

	conv_loss�q/?���(       QKD	tB����AX*

	conv_lossXY/?�[��       QKD	_u����AY*

	conv_loss�^/?��)       QKD	������AZ*

	conv_loss�^/?R� �       QKD	}٫���A[*

	conv_lossQP/?��0�       QKD	G	����A\*

	conv_lossS/?�!�       QKD	q7����A]*

	conv_lossd7/?�k       QKD	�e����A^*

	conv_loss�D/?�       QKD	z�����A_*

	conv_lossA/?��&9       QKD	�¬���A`*

	conv_loss�4/?l>t|       QKD	�����Aa*

	conv_loss�)/?.2Q       QKD	�+����Ab*

	conv_loss�)/?��       QKD	�[����Ac*

	conv_loss�=/?�f$w       QKD	������Ad*

	conv_loss=/?��p(       QKD	Vխ���Ae*

	conv_lossS2/?q�)�       QKD	�����Af*

	conv_loss�/?�[n       QKD	<6����Ag*

	conv_loss�/?���       QKD	�i����Ah*

	conv_loss�/?p�F       QKD	/�����Ai*

	conv_lossL/?�E��       QKD	;ʮ���Aj*

	conv_loss~/?Y)�!       QKD	������Ak*

	conv_loss�/?q�<       QKD	4����Al*

	conv_loss��.?�z��       QKD	Qb����Am*

	conv_lossc�.?��N       QKD	(�����An*

	conv_loss��.?!�WT       QKD	�ï���Ao*

	conv_loss��.?��7       QKD	o����Ap*

	conv_losss�.?m��&       QKD	�%����Aq*

	conv_lossD�.?�ض�       QKD	�d����Ar*

	conv_lossp�.?�2�-       QKD	������As*

	conv_lossi�.?$��       QKD	�̰���At*

	conv_loss��.?��       QKD	������Au*

	conv_loss��.?^K        QKD	G6����Av*

	conv_lossv�.?�h*       QKD	�o����Aw*

	conv_loss��.?���       QKD	ɞ����Ax*

	conv_loss��.?��bl       QKD	3ͱ���Ay*

	conv_loss�.?Qd��       QKD	�����Az*

	conv_loss_�.?�W�s       QKD	�+����A{*

	conv_lossi�.?=���       QKD	�Z����A|*

	conv_loss6�.?��       QKD	�����A}*

	conv_loss��.?x�O       QKD	������A~*

	conv_lossӏ.?@��\       QKD	�����A*

	conv_lossÍ.?��w�        )��P	�'����A�*

	conv_loss��.?H�n�        )��P	�a����A�*

	conv_lossQ�.?�Q]	        )��P	������A�*

	conv_loss̈.?i'PZ        )��P	������A�*

	conv_loss�|.?�89        )��P	�����A�*

	conv_loss�n.?A�*C        )��P	�����A�*

	conv_loss�b.?��        )��P	�L����A�*

	conv_loss�c.?�f�        )��P	�{����A�*

	conv_loss�^.?�<        )��P	;�����A�*

	conv_lossha.?�_�        )��P	�ݴ���A�*

	conv_loss�a.?�r�p        )��P	�����A�*

	conv_loss=\.?L�܁        )��P	�B����A�*

	conv_lossS.?�*�        )��P	Uv����A�*

	conv_loss�O.?�S�        )��P	c�����A�*

	conv_lossbI.?F�;�        )��P	-׵���A�*

	conv_loss�=.?�Y        )��P	�����A�*

	conv_loss	2.?��        )��P	a4����A�*

	conv_loss�).?�ǲ        )��P	;b����A�*

	conv_lossC).?�])        )��P	������A�*

	conv_loss�1.?��<g        )��P	�ƶ���A�*

	conv_loss�..?#ԗK        )��P	������A�*

	conv_loss>.?i��        )��P	%����A�*

	conv_loss<".?�3        )��P	�S����A�*

	conv_loss�.?�D�>        )��P	������A�*

	conv_loss?.?�'�        )��P	�Ƿ���A�*

	conv_loss.?�Ư�        )��P	������A�*

	conv_loss=.?$a�        )��P	�$����A�*

	conv_lossG�-?�Po,        )��P	8V����A�*

	conv_loss��-?8fnK        )��P	������A�*

	conv_loss��-?gQ��        )��P	�����A�*

	conv_loss^�-?P�G�        )��P	�����A�*

	conv_loss��-?L��D        )��P	�����A�*

	conv_loss,�-?�R�        )��P	R����A�*

	conv_loss��-?#���        )��P	0�����A�*

	conv_lossR�-?2L��        )��P	������A�*

	conv_loss��-?�`        )��P	�����A�*

	conv_loss��-?���        )��P	�����A�*

	conv_lossL�-?�?T�        )��P	?M����A�*

	conv_loss̿-?@��        )��P	������A�*

	conv_loss3�-?פhQ        )��P	������A�*

	conv_loss��-?�z�3        )��P	ߺ���A�*

	conv_loss�-?b�@        )��P	
����A�*

	conv_loss��-?,.D2        )��P	�?����A�*

	conv_loss��-?=R�        )��P	�n����A�*

	conv_loss��-?��t        )��P	蟻���A�*

	conv_loss=�-?�
�        )��P	�л���A�*

	conv_loss5�-?h��;        )��P	 ����A�*

	conv_loss{�-?���        )��P	�3����A�*

	conv_loss�-?fư        )��P	;c����A�*

	conv_loss-�-?�\E,        )��P	u�����A�*

	conv_loss�j-?�Oo        )��P	,Ǽ���A�*

	conv_loss�s-?�Ml#        )��P	�����A�*

	conv_loss}r-?P�k	        )��P	*����A�*

	conv_loss�r-?l�er        )��P	+\����A�*

	conv_loss�p-?m&��        )��P	5�����A�*

	conv_loss�b-?p���        )��P	������A�*

	conv_lossi]-?���        )��P	�����A�*

	conv_loss�P-?�v�        )��P	%����A�*

	conv_lossIN-?�h��        )��P	bJ����A�*

	conv_lossxI-?����        )��P	y����A�*

	conv_loss�A-?��=�        )��P	������A�*

	conv_loss�;-?�ë�        )��P	�پ���A�*

	conv_loss�B-?�s�|        )��P	�	����A�*

	conv_loss@=-?f���        )��P	!:����A�*

	conv_lossB--?%�I        )��P	j����A�*

	conv_loss(-?-�XF        )��P	�����A�*

	conv_loss�.-?��8        )��P	˿���A�*

	conv_loss� -?�>�        )��P	������A�*

	conv_loss�/-?j-�        )��P	*����A�*

	conv_lossk-?hY�        )��P	�Y����A�*

	conv_loss�-?�45U        )��P	<�����A�*

	conv_lossr-?ρ}�        )��P	������A�*

	conv_loss�-?� e
        )��P	L�����A�*

	conv_loss�-?�L��        )��P	�~����A�*

	conv_loss��,?Hb        )��P	�����A�*

	conv_lossJ�,?���        )��P	3�����A�*

	conv_loss��,?�&�3        )��P	X����A�*

	conv_loss��,?�A�        )��P	�H����A�*

	conv_loss��,?�@_�        )��P	8{����A�*

	conv_lossr�,?�SY        )��P	J�����A�*

	conv_loss��,?�Q        )��P	m�����A�*

	conv_loss!�,?qb�        )��P	�"����A�*

	conv_loss��,?�}�j        )��P	�c����A�*

	conv_loss��,?J1&S        )��P	������A�*

	conv_loss��,?��.        )��P	������A�*

	conv_lossc�,?�.~w        )��P	�����A�*

	conv_loss��,?���}        )��P	�5����A�*

	conv_loss��,?�wL|        )��P	g����A�*

	conv_loss��,?�4�        )��P	������A�*

	conv_loss1�,?l[�N        )��P	������A�*

	conv_loss*�,?~�o        )��P	o�����A�*

	conv_loss�,?h
�        )��P	�-����A�*

	conv_lossϦ,?�.�        )��P	�`����A�*

	conv_loss2�,?`:U        )��P	V�����A�*

	conv_loss"�,?�	3a        )��P	�����A�*

	conv_loss�,?����        )��P	������A�*

	conv_loss�},?ݟB�        )��P	�-����A�*

	conv_loss!�,?[~��        )��P	�^����A�*

	conv_loss�~,?��-�        )��P	�����A�*

	conv_loss�q,?١�        )��P	������A�*

	conv_loss�t,?%O�+        )��P	1�����A�*

	conv_loss�h,?U2�j        )��P	l&����A�*

	conv_lossEj,?���        )��P	�Z����A�*

	conv_loss~b,?w΍5        )��P	������A�*

	conv_loss�Y,?�f)        )��P	+�����A�*

	conv_loss�W,?ڛ-�        )��P	������A�*

	conv_loss�S,?�f��        )��P	&����A�*

	conv_losshB,?R�φ        )��P	�Y����A�*

	conv_loss�N,?���        )��P	H�����A�*

	conv_loss2,?w���        )��P	������A�*

	conv_lossu7,?�ָ        )��P	,�����A�*

	conv_loss�2,?��c        )��P	� ����A�*

	conv_loss,$,?�dJ        )��P	qR����A�*

	conv_lossp),?��|�        )��P	������A�*

	conv_loss$,?7��<        )��P	#�����A�*

	conv_loss�",?뙊�        )��P	������A�*

	conv_loss� ,?��8        )��P	�"����A�*

	conv_loss�,?���q        )��P	WT����A�*

	conv_loss�,?z�+�        )��P	+�����A�*

	conv_loss�,?x*        )��P	 �����A�*

	conv_loss{,?5        )��P	9�����A�*

	conv_loss�,?�k/�        )��P	@����A�*

	conv_loss��+?�9C        )��P	�K����A�*

	conv_loss<�+?�:z?        )��P	�����A�*

	conv_loss��+?WE~�        )��P	������A�*

	conv_loss��+?� �        )��P	�����A�*

	conv_loss��+?����        )��P	i/����A�*

	conv_lossy�+?���        )��P	�b����A�*

	conv_lossS�+?�o9        )��P	������A�*

	conv_loss��+?�R�|        )��P	�����A�*

	conv_loss>�+?:%�>        )��P	�����A�*

	conv_loss��+?�l�O        )��P	�:����A�*

	conv_lossƵ+?J�        )��P	�x����A�*

	conv_loss��+?
Xj        )��P	̭����A�*

	conv_loss��+?�6+�        )��P	F�����A�*

	conv_loss#�+?��\[        )��P	����A�*

	conv_loss��+?햒        )��P	K����A�*

	conv_loss̞+?�U�        )��P	 �����A�*

	conv_loss˩+?}hާ        )��P	������A�*

	conv_loss��+?�O��        )��P	U�����A�*

	conv_lossJ�+?�ִ7        )��P	H/����A�*

	conv_loss�+?���        )��P	Ca����A�*

	conv_loss�+?�g        )��P	c�����A�*

	conv_losso�+?'+        )��P	@�����A�*

	conv_lossÅ+?����        )��P	(�����A�*

	conv_lossϊ+?o���        )��P	�?����A�*

	conv_losszx+?����        )��P	�s����A�*

	conv_loss1e+?��n        )��P	m�����A�*

	conv_loss�m+?&���        )��P	^�����A�*

	conv_loss�q+?'׌�        )��P	@����A�*

	conv_loss�l+?���        )��P	jF����A�*

	conv_loss8h+?��I
        )��P	]y����A�*

	conv_loss�X+?/���        )��P	������A�*

	conv_lossM]+?�D�        )��P	q�����A�*

	conv_loss�Q+?�P�        )��P	6����A�*

	conv_loss�4+?;���        )��P	�I����A�*

	conv_loss�F+?/���        )��P	Ђ����A�*

	conv_lossW1+?�~��        )��P	������A�*

	conv_lossc&+?�AT�        )��P	)�����A�*

	conv_loss=+?�6$1        )��P	�����A�*

	conv_loss�+?���        )��P	�Q����A�*

	conv_loss�)+?r�        )��P	������A�*

	conv_lossn+?OPA�        )��P	O�����A�*

	conv_loss>+?�*W�        )��P	������A�*

	conv_loss. +?�\��        )��P	����A�*

	conv_loss�
+?M�D{        )��P	�O����A�*

	conv_lossi
+?�ܢ        )��P	/�����A�*

	conv_lossK+?��E        )��P	s�����A�*

	conv_loss�+?\�)        )��P	A�����A�*

	conv_loss;�*?rFns        )��P	I����A�*

	conv_loss3�*?� �        )��P	�F����A�*

	conv_lossF�*? V5�        )��P	#y����A�*

	conv_loss#�*?D�D�        )��P	�����A�*

	conv_loss��*?�Hv2        )��P	������A�*

	conv_loss��*?�=�y        )��P	Y$����A�*

	conv_loss��*?�a��        )��P	Y����A�*

	conv_loss'�*?��ne        )��P	�����A�*

	conv_loss��*?ϴ�0        )��P	Y�����A�*

	conv_loss��*?�1��        )��P	I�����A�*

	conv_loss��*?4��        )��P	$����A�*

	conv_loss$�*?�:�n        )��P	U����A�*

	conv_loss��*?��        )��P	Y�����A�*

	conv_loss+�*?���0        )��P	|�����A�*

	conv_loss��*?<'�        )��P	 ����A�*

	conv_lossk�*?�r        )��P	�8����A�*

	conv_loss�*?n�2        )��P	�z����A�*

	conv_lossE�*?��        )��P	������A�*

	conv_loss1�*?U|��        )��P	�����A�*

	conv_loss��*?q�PG        )��P	#����A�*

	conv_loss�*?̚�W        )��P	�C����A�*

	conv_lossk�*?���        )��P	j�����A�*

	conv_loss�~*?{m�K        )��P	ޱ����A�*

	conv_loss�}*?�u�        )��P	������A�*

	conv_loss�t*?+�3�        )��P	#����A�*

	conv_loss|*?�(@6        )��P	P����A�*

	conv_loss�`*?
�C}        )��P	�����A�*

	conv_loss�r*?k��L        )��P	ú����A�*

	conv_loss|]*?�A�        )��P	������A�*

	conv_loss�W*??��j        )��P	Q����A�*

	conv_loss�Z*?wI��        )��P	�T����A�*

	conv_loss`P*?}�q�        )��P	F�����A�*

	conv_loss�@*?�D�        )��P	������A�*

	conv_loss�O*?y��        )��P	�����A�*

	conv_loss@E*?��yT        )��P	R.����A�*

	conv_loss:*?��<�        )��P	�a����A�*

	conv_loss�C*?���        )��P	������A�*

	conv_loss�1*?�ȭ�        )��P	M�����A�*

	conv_loss�**?��c        )��P	����A�*

	conv_loss� *?|�)        )��P	H4����A�*

	conv_loss�"*?��        )��P	 h����A�*

	conv_lossc*?���        )��P	\�����A�*

	conv_loss@*?�yb�        )��P	������A�*

	conv_loss�*? ���        )��P	�����A�*

	conv_loss�	*?IOy�        )��P	�*����A�*

	conv_loss�*?�w�i        )��P	^\����A�*

	conv_loss��)?E�<�        )��P	������A�*

	conv_loss�*?��6        )��P		�����A�*

	conv_lossi�)?ʏ�L        )��P	 �����A�*

	conv_loss��)?�V        )��P	2����A�*

	conv_loss%�)?<��p        )��P	Lh����A�*

	conv_loss��)?'�=        )��P	ף����A�*

	conv_loss��)?���|        )��P	������A�*

	conv_loss��)?�+�        )��P	����A�*

	conv_loss��)?�XDQ        )��P	�Y����A�*

	conv_loss��)?"�L        )��P	������A�*

	conv_loss}�)?��J        )��P	������A�*

	conv_lossT�)?�7Ӹ        )��P	������A�*

	conv_lossӴ)?�{*]        )��P	�'����A�*

	conv_loss�)?�p�        )��P	�[����A�*

	conv_lossv�)?}�x�        )��P	Г����A�*

	conv_loss�)?&~Z        )��P	������A�*

	conv_loss��)?˹8)        )��P	�����A�*

	conv_loss֨)?��'�        )��P	�:����A�*

	conv_loss��)?d�E�        )��P	nm����A�*

	conv_loss��)?��1�        )��P	������A�*

	conv_loss(�)?랒�        )��P	.�����A�*

	conv_loss"�)?\��P        )��P	�����A�*

	conv_loss�~)?��U�        )��P	�=����A�*

	conv_loss��)?m���        )��P	Fn����A�*

	conv_loss�|)?��0        )��P	U�����A�*

	conv_loss]})?�73        )��P	y�����A�*

	conv_loss�f)?��i        )��P	�����A�*

	conv_loss�x)?��]        )��P	=����A�*

	conv_loss�v)?x�        )��P	8q����A�*

	conv_loss�a)?@�gH        )��P	~�����A�*

	conv_lossVT)?���        )��P	������A�*

	conv_loss�Q)?kW�~        )��P	D����A�*

	conv_lossYL)?@[��        )��P	�:����A�*

	conv_loss5I)?8�ؓ        )��P	?m����A�*

	conv_loss�R)?!�-�        )��P	������A�*

	conv_loss�C)?��-�        )��P	������A�*

	conv_loss�;)?�M�        )��P	�����A�*

	conv_loss!;)? B�m        )��P	�9����A�*

	conv_lossa3)?���;        )��P	�j����A�*

	conv_lossi*)?u�        )��P	E�����A�*

	conv_loss�)?#8�/        )��P	H�����A�*

	conv_loss)?��_<        )��P	�	����A�*

	conv_loss&)?��n        )��P	�;����A�*

	conv_loss|)?/�g@        )��P	Vo����A�*

	conv_loss�)?^kL�        )��P	������A�*

	conv_loss)?t�R`        )��P	������A�*

	conv_loss�)?�$��        )��P	m
����A�*

	conv_loss�)?)�Q	        )��P	�;����A�*

	conv_loss/�(?a=�        )��P	�l����A�*

	conv_loss&�(?�u�+        )��P	�����A�*

	conv_loss�)?�d�=        )��P	������A�*

	conv_loss��(?��        )��P	����A�*

	conv_lossP�(?�P�t        )��P	�3����A�*

	conv_loss��(?&� �        )��P	�l����A�*

	conv_loss��(?`p��        )��P	Ο����A�*

	conv_loss��(?d�v�        )��P	������A�*

	conv_loss��(?��L        )��P	I����A�*

	conv_loss��(?P�c�        )��P	�E����A�*

	conv_lossp�(?���D        )��P	������A�*

	conv_loss��(?p��N        )��P	�-����A�*

	conv_loss��(?����        )��P	�_����A�*

	conv_lossu�(?A��        )��P	������A�*

	conv_loss�(?��%        )��P	�����A�*

	conv_loss��(?Ɉ�1        )��P	������A�*

	conv_lossߥ(?�!}        )��P	T4����A�*

	conv_lossQ�(?ˠ�        )��P	d����A�*

	conv_lossR�(?f#+        )��P	w�����A�*

	conv_loss��(?!=
6        )��P	\�����A�*

	conv_loss�(?���        )��P	�����A�*

	conv_loss��(?�        )��P	�,����A�*

	conv_loss�t(?���{        )��P	r����A�*

	conv_loss�}(?S��a        )��P	������A�*

	conv_loss�m(?�'��        )��P	������A�*

	conv_loss1r(?&�t�        )��P	&����A�*

	conv_loss�n(?�҄
        )��P	
8����A�*

	conv_loss�h(?���        )��P	�g����A�*

	conv_losse(?�f�        )��P	Q�����A�*

	conv_lossBR(?_���        )��P	$�����A�*

	conv_loss�T(?'E�]        )��P	�����A�*

	conv_loss�X(?���H        )��P	�0����A�*

	conv_lossLK(?�%1�        )��P	�n����A�*

	conv_loss�/(?���        )��P	������A�*

	conv_loss�6(?ٝD�        )��P	b�����A�*

	conv_loss 9(?����        )��P	�#����A�*

	conv_loss"(?ݳ�        )��P	<T����A�*

	conv_loss;1(?���o        )��P	ۊ����A�*

	conv_lossJ((?�� �        )��P	������A�*

	conv_loss(?Mp��        )��P	������A�*

	conv_loss�(?[S�        )��P	�����A�*

	conv_loss�(?L� �        )��P	&P����A�*

	conv_loss�(?d\T        )��P	І����A�*

	conv_loss�(?	��        )��P	2�����A�*

	conv_lossd�'?��        )��P	������A�*

	conv_loss�'?d[-d        )��P	2����A�*

	conv_loss��'?�i�        )��P	J����A�*

	conv_loss��'?nk��        )��P	*�����A�*

	conv_loss��'?R�
        )��P	ݴ����A�*

	conv_loss��'?�՝        )��P	�����A�*

	conv_loss��'?�I3�        )��P	�����A�*

	conv_loss��'?_ ��        )��P	K����A�*

	conv_loss��'?��pl        )��P	Y�����A�*

	conv_lossD�'?k*�        )��P	������A�*

	conv_lossr�'?����        )��P	4�����A�*

	conv_lossV�'?���        )��P	S ����A�*

	conv_loss�'?��ҙ        )��P	'S����A�*

	conv_lossJ�'?���        )��P	ҏ����A�*

	conv_lossg�'?��ϼ        )��P	h�����A�*

	conv_lossʘ'?}�Q        )��P	������A�*

	conv_lossN�'?ze�~        )��P	DD����A�*

	conv_loss��'?�[��        )��P	�s����A�*

	conv_lossy�'?�yŏ        )��P	M�����A�*

	conv_loss$�'?�c        )��P	������A�*

	conv_loss��'?ꞏW        )��P	�����A�*

	conv_loss||'?g�M        )��P	F����A�*

	conv_lossi�'?.�         )��P	ow����A�*

	conv_loss�y'?"%$=        )��P	x�����A�*

	conv_loss�d'?����        )��P	T�����A�*

	conv_loss�h'?L�G�        )��P	����A�*

	conv_loss,e'?ey�$        )��P	mJ����A�*

	conv_lossn'?�+�"        )��P	������A�*

	conv_loss�f'?�B>�        )��P	�����A�*

	conv_loss�K'?l��        )��P	������A�*

	conv_lossTR'?�)�F        )��P	)����A�*

	conv_lossF'?S�a        )��P	`[����A�*

	conv_loss"C'?vc        )��P	ʋ����A�*

	conv_loss�1'?1��        )��P	
�����A�*

	conv_loss�F'?����        )��P	������A�*

	conv_loss�C'?|Ts8        )��P	M'����A�*

	conv_lossw/'?�K��        )��P	?`����A�*

	conv_loss�6'?Eo�        )��P	������A�*

	conv_loss�/'?sn��        )��P	������A�*

	conv_loss*'?���        )��P	����A�*

	conv_losse'?���        )��P	s7����A�*

	conv_loss�'?�E�        )��P	i����A�*

	conv_lossZ'?I�D�        )��P	�����A�*

	conv_loss8'?��C�        )��P	y�����A�*

	conv_loss�'?��@�        )��P	2����A�*

	conv_loss�'?�\A        )��P	"5����A�*

	conv_lossa�&?��&        )��P	�j����A�*

	conv_loss��&?�3��        )��P	������A�*

	conv_lossn�&?��!�        )��P	������A�*

	conv_lossj�&?��:        )��P	�����A�*

	conv_loss��&?H��        )��P	�9����A�*

	conv_loss��&?~$_�        )��P	�{����A�*

	conv_loss��&?���>        )��P	³����A�*

	conv_loss�&?�%�        )��P	s�����A�*

	conv_loss��&?�wm�        )��P	�����A�*

	conv_losst�&?wX�        )��P	�G����A�*

	conv_loss��&?�T-(        )��P	y����A�*

	conv_lossq�&?�6�        )��P	������A�*

	conv_loss�&?�F��        )��P	�����A�*

	conv_loss{�&?έF+        )��P	� ���A�*

	conv_loss��&?k�ρ        )��P	(G ���A�*

	conv_loss�}&?r�h        )��P	�y ���A�*

	conv_loss]�&?�su        )��P	� ���A�*

	conv_loss�&?�^�        )��P	L� ���A�*

	conv_loss�~&?���U        )��P	n ���A�*

	conv_loss��&?=r9#        )��P	�����A�*

	conv_loss�&?���p        )��P	���A�*

	conv_loss�v&?���        )��P	�;���A�*

	conv_loss�p&?��~�        )��P	�j���A�*

	conv_lossIy&?%Ud�        )��P	����A�*

	conv_loss�{&?�Sk�        )��P	����A�*

	conv_loss�M&?q�Z        )��P	�
���A�*

	conv_loss�S&?y�KG        )��P	�:���A�*

	conv_losswP&?�"��        )��P	^l���A�*

	conv_lossTE&?�O�        )��P	4����A�*

	conv_lossF&?�%�        )��P	�����A�*

	conv_loss7&?<�[        )��P	���A�*

	conv_loss(A&?4[�        )��P	<B���A�*

	conv_lossL=&?����        )��P	Cw���A�*

	conv_loss�D&??׿�        )��P	����A�*

	conv_lossJ0&?Q�y5        )��P	�����A�*

	conv_loss�'&?ԳoY        )��P	�&	���A�*

	conv_loss\&?o���        )��P	�V	���A�*

	conv_loss�&?ݑ��        )��P	�	���A�*

	conv_loss&?C�R        )��P	.�	���A�*

	conv_lossA&?N�K        )��P	2�	���A�*

	conv_loss�&?(�wJ        )��P	r
���A�*

	conv_loss?&?GI�}        )��P	�F
���A�*

	conv_loss�&?&l�'        )��P	9w
���A�*

	conv_loss��%?��        )��P	�
���A�*

	conv_loss��%?w?WU        )��P	��
���A�*

	conv_loss�%?Ζ{        )��P	=���A�*

	conv_loss��%?�Ws�        )��P	)E���A�*

	conv_loss��%?����        )��P	u���A�*

	conv_loss��%?�k��        )��P	|����A�*

	conv_loss��%?P��P        )��P	�����A�*

	conv_loss��%?ͧ�        )��P	���A�*

	conv_loss��%?͌kQ        )��P	kC���A�*

	conv_lossô%?��        )��P	�r���A�*

	conv_loss��%?KV}�        )��P	�����A�*

	conv_loss��%?��D�        )��P	}����A�*

	conv_lossF�%?��N        )��P	����A�*

	conv_loss�%?���g        )��P	�2���A�*

	conv_loss$�%?+�r        )��P	�d���A�*

	conv_loss�%?'X�=        )��P	����A�*

	conv_loss�%?�SA        )��P	I����A�*

	conv_loss�%?�M        )��P	p����A�*

	conv_lossQ|%?)��        )��P	%���A�*

	conv_lossf%?^�	�        )��P	3[���A�*

	conv_loss�%?�?�        )��P	u����A�*

	conv_lossc%?�Q��        )��P	{����A�*

	conv_loss�[%?r7�        )��P	�����A�*

	conv_loss�a%?�ɔ�        )��P	E���A�*

	conv_lossZY%?p4{        )��P	
W���A�*

	conv_loss�k%?^κ6        )��P	)����A�*

	conv_losseG%?�]�        )��P	����A�*

	conv_loss�;%?���        )��P	-����A�*

	conv_loss�>%?_�9�        )��P	&���A�*

	conv_losse6%?����        )��P	�_���A�*

	conv_loss*%?~Λ\        )��P	B����A�*

	conv_lossQ%?���        )��P	�����A�*

	conv_loss %?�)�        )��P	�����A�*

	conv_loss�/%?��]�        )��P	p$���A�*

	conv_loss�%?�j�        )��P	
_���A�*

	conv_loss!%?��|        )��P	�����A�*

	conv_loss�%?��        )��P	����A�*

	conv_loss��$?i <�        )��P	�����A�*

	conv_loss��$?�r��        )��P	�,���A�*

	conv_loss%?պg�        )��P	�i���A�*

	conv_loss��$?E��        )��P	ۥ���A�*

	conv_loss��$?�V�        )��P	f����A�*

	conv_loss��$?�n
        )��P	����A�*

	conv_loss��$?�#�        )��P	K���A�*

	conv_lossP�$?���        )��P	Ɓ���A�*

	conv_loss �$?�oh�        )��P	�����A�*

	conv_loss�$?���        )��P	�����A�*

	conv_loss��$?Wx�        )��P	����A�*

	conv_loss=�$?W�g�        )��P	�F���A�*

	conv_loss2�$?�$        )��P	^����A�*

	conv_loss��$?tѪ        )��P	�����A�*

	conv_loss��$?�{G        )��P	�����A�*

	conv_loss-�$?"�        )��P	B���A�*

	conv_lossc�$?j���        )��P	�L���A�*

	conv_loss��$?9�N        )��P	X����A�*

	conv_loss%q$?Z�i=        )��P	<����A�*

	conv_loss�m$?���        )��P	.����A�*

	conv_losseb$?5
�        )��P	����A�*

	conv_loss�u$?����        )��P	
E���A�*

	conv_lossfh$?!�h�        )��P	Jw���A�*

	conv_loss�G$?�P�        )��P	����A�*

	conv_loss\a$?��/�        )��P	�����A�*

	conv_lossg$?E�[        )��P	����A�*

	conv_loss�J$?����        )��P	�@���A�*

	conv_loss{T$?���        )��P	�p���A�*

	conv_loss�O$?vD�        )��P	�����A�*

	conv_lossl%$?�%��        )��P	�����A�*

	conv_loss�H$?*�Ǥ        )��P	���A�*

	conv_lossS+$?�rE�        )��P	a<���A�*

	conv_loss�$?�PS�        )��P	�o���A�*

	conv_lossn+$?�;H        )��P	Q����A�*

	conv_loss�$?�U�        )��P	/����A�*

	conv_loss�$?d�        )��P	!���A�*

	conv_loss� $?�n5�        )��P	�6���A�*

	conv_loss��#?l
Ի        )��P	f���A�*

	conv_loss�$?LjqE        )��P	h����A�*

	conv_lossU $?ii�n        )��P	�����A�*

	conv_loss/�#?�z0�        )��P	�c���A�*

	conv_loss��#?��>        )��P	T����A�*

	conv_loss��#?� o        )��P	�����A�*

	conv_loss�#?*N˖        )��P	�����A�*

	conv_loss��#?��6�        )��P	�.���A�*

	conv_loss]�#?���        )��P	Vc���A�*

	conv_loss��#?/�V        )��P	a����A�*

	conv_loss��#?����        )��P	�����A�*

	conv_lossI�#?;��l        )��P	a���A�*

	conv_loss��#?�Uѻ        )��P	�>���A�*

	conv_lossG�#?�Ϫ�        )��P	Hq���A�*

	conv_loss��#?t�0        )��P	�����A�*

	conv_losszf#?�=��        )��P	<����A�*

	conv_lossDm#?<D��        )��P	(���A�*

	conv_loss݅#?��o_        )��P	wE���A�*

	conv_loss�m#?$@z        )��P	w���A�*

	conv_loss�d#?$���        )��P	�����A�*

	conv_loss�`#?Ľ[7        )��P	����A�*

	conv_loss�p#?��#        )��P	>���A�*

	conv_loss�k#?!1        )��P	�H���A�*

	conv_lossEQ#?���        )��P	�}���A�*

	conv_loss�T#?�Q��        )��P	�����A�*

	conv_loss-*#?؏        )��P	�����A�*

	conv_lossm0#?�<�        )��P	0! ���A�*

	conv_loss}J#?,        )��P	�T ���A�*

	conv_loss�/#?W,��        )��P	T� ���A�*

	conv_loss$#?��H        )��P	ĵ ���A�*

	conv_loss.#?��F�        )��P	W� ���A�*

	conv_losso#?�w��        )��P	�!���A�*

	conv_lossD�"?�@�        )��P	I!���A�*

	conv_lossc�"?uܭ;        )��P	0y!���A�*

	conv_loss��"?�H'�        )��P	k�!���A�*

	conv_losse�"?�`��        )��P	��!���A�*

	conv_loss�"?ce�        )��P	�"���A�*

	conv_lossJ�"?�+}        )��P	�N"���A�*

	conv_loss��"?(^]        )��P	#~"���A�*

	conv_loss��"?=�
c        )��P	��"���A�*

	conv_loss��"?����        )��P	$�"���A�*

	conv_loss�"?Z��        )��P	�#���A�*

	conv_loss:�"?��        )��P	>B#���A�*

	conv_loss�"?!�`        )��P	>t#���A�*

	conv_loss�"?��H        )��P	��#���A�*

	conv_loss1s"?^X1        )��P	N�#���A�*

	conv_loss��"?xt        )��P	�$���A�*

	conv_loss�j"?eɵ�        )��P	�N$���A�*

	conv_loss9h"?���p        )��P	ف$���A�*

	conv_loss�p"?�-�f        )��P	�$���A�*

	conv_loss�["?Y�A        )��P	R�$���A�*

	conv_loss7k"?���        )��P	F#%���A�*

	conv_loss�@"?���        )��P	�U%���A�*

	conv_lossU2"?��e�        )��P	֘%���A�*

	conv_lossVF"?�?H�        )��P	��%���A�*

	conv_loss,"?��p]        )��P	1&���A�*

	conv_loss�"?�?-        )��P	�2&���A�*

	conv_loss�"?n<��        )��P	Ld&���A�*

	conv_loss�!?����        )��P	f�&���A�*

	conv_loss�*"?0}��        )��P	0�&���A�*

	conv_lossU�!?
k�        )��P	�'���A�*

	conv_loss��!?�|Mp        )��P	�E'���A�*

	conv_loss��!?\��        )��P	�v'���A�*

	conv_loss��!?l4f�        )��P	�'���A�*

	conv_lossO�!?���        )��P	��'���A�*

	conv_loss9�!?<{��        )��P	'(���A�*

	conv_lossޣ!?v�C�        )��P	G(���A�*

	conv_loss:�!?h�:        )��P	nx(���A�*

	conv_loss��!?��o�        )��P	��(���A�*

	conv_loss��!?�=]        )��P	?�(���A�*

	conv_lossܯ!?3L�        )��P	)���A�*

	conv_loss:�!?�`'�        )��P	B)���A�*

	conv_lossRq!?�B�        )��P	x)���A�*

	conv_lossEh!?���h        )��P	'�)���A�*

	conv_loss��!?���c        )��P	��)���A�*

	conv_losshq!?�i�K        )��P	Q*���A�*

	conv_lossI!?p��        )��P	C*���A�*

	conv_loss)M!?�d�e        )��P	�u*���A�*

	conv_loss�,!?�;-�        )��P	a�*���A�*

	conv_loss�J!?�dM        )��P	��*���A�*

	conv_loss�%!?�֖        )��P	;+���A�*

	conv_loss�#!?�me        )��P	"K+���A�*

	conv_loss�� ?ӷu        )��P	U�+���A�*

	conv_loss�� ?�t;        )��P	��+���A�*

	conv_lossS� ?�        )��P	U�+���A�*

	conv_loss�� ?�!        )��P	',���A�*

	conv_loss�� ?l:�        )��P	IW,���A�*

	conv_loss�� ?��n<        )��P	��,���A�*

	conv_loss�� ?U��        )��P	�,���A�*

	conv_loss�� ?�\`        )��P	��,���A�*

	conv_loss� ?�<x(        )��P	-���A�*

	conv_lossÍ ?}
�        )��P	�L-���A�*

	conv_loss,� ?���        )��P	A}-���A�*

	conv_lossգ ?bt?
        )��P	�-���A�*

	conv_loss� ?]�|�        )��P	��-���A�*

	conv_loss�l ?�Q�2        )��P	2.���A�*

	conv_lossEN ?��b        )��P	g@.���A�*

	conv_loss7 ?���6        )��P	�p.���A�*

	conv_loss�M ?�m$        )��P	��.���A�*

	conv_loss� ?�@�        )��P	��.���A�*

	conv_loss	+ ?r��        )��P	</���A�*

	conv_loss�	 ?�_�;        )��P	4/���A�*

	conv_loss��?7Y��        )��P	1f/���A�*

	conv_loss�& ?#]^�        )��P	��/���A�*

	conv_loss��?� �        )��P	�/���A�*

	conv_lossk�?��,        )��P	�	0���A�*

	conv_lossk�?�;�        )��P	@0���A�*

	conv_loss��?���x        )��P	 r0���A�*

	conv_loss��?��.        )��P	~�0���A�*

	conv_lossJv?S�#�        )��P	2�0���A�*

	conv_loss�p?Ɩ~        )��P	�1���A�*

	conv_loss�o?w���        )��P	�Q1���A�*

	conv_loss�q?��,        )��P	�1���A�*

	conv_lossPL?R_        )��P	�1���A�*

	conv_loss2r?Y��        )��P	��1���A�*

	conv_losssK?P�        )��P	�2���A�*

	conv_loss_?@�4        )��P	&P2���A�*

	conv_loss*�?�MS        )��P	d�2���A�*

	conv_loss�?�T��        )��P	��2���A�*

	conv_loss�>?BM��        )��P	��2���A�*

	conv_loss�F?:�        )��P	�3���A�*

	conv_loss��?!`e�        )��P	6E3���A�*

	conv_loss8�?BIK        )��P	^|3���A�*

	conv_loss�?����        )��P	��3���A�*

	conv_loss�x?��>v        )��P	�3���A�*

	conv_loss��?_�ʗ        )��P	�4���A�*

	conv_loss��?n��        )��P	�J4���A�*

	conv_loss"\?���        )��P	�z4���A�*

	conv_loss�o?�?�        )��P	t�4���A�*

	conv_loss�Z?�d)�        )��P	��4���A�*

	conv_loss'?�
pD        )��P	�5���A�*

	conv_loss?ᖇ�        )��P	�=5���A�*

	conv_loss�?v�a�        )��P	oo5���A�*

	conv_loss��?�č        )��P	֠5���A�*

	conv_loss2�?�*e�        )��P	��5���A�*

	conv_loss5�?���         )��P		6���A�*

	conv_loss��?s��        )��P	�:6���A�*

	conv_loss�|?� �        )��P	�r6���A�*

	conv_lossZ�?@�ц        )��P	�6���A�*

	conv_loss��?I�h7        )��P	�6���A�*

	conv_lossmM?����        )��P	 	7���A�*

	conv_lossJl?��&r        )��P	|97���A�*

	conv_loss�H?N@]        )��P	�s7���A�*

	conv_loss�1?l��        )��P	$�7���A�*

	conv_loss--?��G        )��P	��7���A�*

	conv_loss��?�'�H        )��P	,8���A�*

	conv_loss �?�
�        )��P	Z68���A�*

	conv_loss�q?V        )��P	�q8���A�*

	conv_loss��?��        )��P	��8���A�*

	conv_loss��?Cx��        )��P	��8���A�*

	conv_loss��?��y;        )��P	~9���A�*

	conv_lossTw?ȵ
�        )��P	D49���A�*

	conv_loss�?�xb�        )��P	�t9���A�*

	conv_lossY?l�P        )��P	�9���A�*

	conv_losst�?���        )��P	��9���A�*

	conv_loss'�?r)�        )��P	�):���A�*

	conv_loss��?�mM�        )��P	g_:���A�*

	conv_loss��?BW�        )��P	�:���A�*

	conv_lossk�?�}}        )��P	��:���A�*

	conv_loss�?-��[        )��P	e
;���A�*

	conv_lossf�?�&�        )��P	J>;���A�*

	conv_lossr�?�z��        )��P	�r;���A�*

	conv_loss��?��.�        )��P	�;���A�*

	conv_loss_�?�ڏ�        )��P	�;���A�*

	conv_loss��?�k        )��P	�b<���A�*

	conv_lossۨ?�5�n        )��P	�<���A�*

	conv_lossu^?d&��        )��P	��<���A�*

	conv_loss_�?�.r        )��P	@=���A�*

	conv_loss�?��I�        )��P	 I=���A�*

	conv_loss��?ld�A        )��P	�}=���A�*

	conv_loss�?���C        )��P	�=���A�*

	conv_lossa�?�W        )��P	�=���A�*

	conv_loss/�?���u        )��P	�2>���A�*

	conv_loss�G?T�P        )��P	�n>���A�*

	conv_loss�?5��        )��P	��>���A�*

	conv_loss��?�w�        )��P	�>���A�*

	conv_lossC�?XcN:        )��P	?���A�*

	conv_loss��?�#9K        )��P	�;?���A�*

	conv_lossV�?��rK        )��P	�m?���A�*

	conv_losss^?M��q        )��P	О?���A�*

	conv_lossm�?�#        )��P	��?���A�*

	conv_loss��?6�        )��P	�@���A�*

	conv_loss��? $+        )��P	<<@���A�*

	conv_loss5u?�5�        )��P	�o@���A�*

	conv_lossX�?6t��        )��P	��@���A�*

	conv_lossw%?>p�        )��P	k�@���A�*

	conv_losss�?����        )��P	�A���A�*

	conv_loss5?*,��        )��P	zWA���A�*

	conv_loss�S?�a��        )��P	|�A���A�*

	conv_lossݹ?�5�"        )��P	C�A���A�*

	conv_loss�?���        )��P	� B���A�*

	conv_loss�T?� .        )��P	!5B���A�*

	conv_loss5L?6�B�        )��P	qhB���A�*

	conv_lossx ?�)�        )��P	��B���A�*

	conv_loss0B?��i�        )��P	y�B���A�*

	conv_loss#j?���8        )��P	`C���A�*

	conv_loss��?e���        )��P	�AC���A�*

	conv_loss��?
��        )��P	=rC���A�*

	conv_loss�?��\	        )��P	�C���A�*

	conv_loss�_?L�+�        )��P	%�C���A�*

	conv_loss�?\`�>        )��P	}D���A�*

	conv_loss�E?�+        )��P	y9D���A�*

	conv_losse?e�Q        )��P	�jD���A�*

	conv_lossѓ?��/�        )��P	F���A�*

	conv_lossp,?1Is        )��P	=F���A�*

	conv_loss��?b��        )��P	GzF���A�*

	conv_loss�k?��S        )��P	֯F���A�*

	conv_lossx?;]�        )��P	B�F���A�*

	conv_loss�?ݓ
�        )��P	� G���A�*

	conv_loss!?�.Z�        )��P	�VG���A�*

	conv_loss�J?�&        )��P	͇G���A�*

	conv_loss�?��        )��P	��G���A�*

	conv_loss�=?W͆�        )��P	�H���A�*

	conv_loss[�?�W��        )��P	�EH���A�*

	conv_loss�?��[S        )��P	vH���A�*

	conv_lossT�?
-        )��P	|�H���A�*

	conv_loss�?�蛧        )��P	��H���A�*

	conv_loss��?]���        )��P	�I���A�*

	conv_loss?���        )��P	�NI���A�*

	conv_loss��?.P]         )��P	�I���A�*

	conv_lossW?�K��        )��P	�I���A�*

	conv_loss�@	?�_        )��P	�I���A�*

	conv_loss�M	?r?L�        )��P	GJ���A�*

	conv_loss$?��)        )��P	hTJ���A�*

	conv_loss@2	?Re        )��P	ƅJ���A�*

	conv_loss�?���        )��P	Q�J���A�*

	conv_lossf�?�V��        )��P	l�J���A�*

	conv_lossPu?A��8        )��P	� K���A�*

	conv_loss�?�t        )��P	�QK���A�*

	conv_loss�o?���        )��P	�K���A�*

	conv_lossm?.�|�        )��P	d�K���A�*

	conv_loss?�"�        )��P	��K���A�*

	conv_loss$j?`�F�        )��P	T+L���A�*

	conv_loss��?��8         )��P	�\L���A�*

	conv_loss02?���        )��P	]�L���A�*

	conv_loss� ?\��        )��P	��L���A�*

	conv_loss�K ?s���        )��P	d�L���A�*

	conv_loss�	�>662�        )��P	h4M���A�*

	conv_loss��>��|�        )��P	�dM���A�*

	conv_loss�x�>r�8        )��P	��M���A�*

	conv_loss���>��#<        )��P	��M���A�*

	conv_lossSq�>I���        )��P	��M���A�*

	conv_loss��>���;        )��P	� N���A�*

	conv_loss�<�>X5        )��P	qON���A�*

	conv_loss���>'V��        )��P	x~N���A�*

	conv_lossR��>����        )��P	A�N���A�*

	conv_loss���>ݮ/�        )��P	1�N���A�*

	conv_loss���>�v��        )��P	,	O���A�*

	conv_loss>�>ғ�M        )��P	BO���A�*

	conv_loss|��>"D�        )��P	�qO���A�*

	conv_loss&��>3�        )��P	a�O���A�*

	conv_lossbS�>=�J�        )��P	��O���A�*

	conv_loss��>��+�        )��P	P���A�*

	conv_loss���>����        )��P	,bP���A�*

	conv_loss9��>�JR        )��P	[�P���A�*

	conv_loss�	�>��        )��P	u�P���A�*

	conv_loss�6�>c~y        )��P	��P���A�*

	conv_loss¶�>���        )��P	& Q���A�*

	conv_loss0��>��=�        )��P	[Q���A�*

	conv_loss�(�>o�@        )��P	c�Q���A�*

	conv_loss0��>�-        )��P	]�Q���A�*

	conv_loss3Y�>Y　        )��P	~�Q���A�*

	conv_loss�|�>�p�I        )��P	�.R���A�*

	conv_loss�}�>3*��        )��P	|mR���A�*

	conv_loss���>@I�+        )��P	�R���A�*

	conv_loss��>�h�        )��P	m�R���A�*

	conv_lossԤ�>7"�        )��P	pS���A�*

	conv_lossy��>�ϻ        )��P	�5S���A�*

	conv_lossc��>Y���        )��P	6kS���A�*

	conv_loss��>����        )��P	J�S���A�*

	conv_loss�Ľ>��E        )��P	/�S���A�*

	conv_loss:��>u��        )��P	��S���A�*

	conv_loss��>8=h        )��P	�(T���A�*

	conv_loss/۹>j��        )��P	�YT���A�*

	conv_loss���>��ߪ        )��P	��T���A�*

	conv_lossFٹ>nw�        )��P	��T���A�*

	conv_loss,��>���g        )��P	��T���A�*

	conv_loss�Ҷ>Uĉ9        )��P	�"U���A�*

	conv_loss�|�>�1p8        )��P	�RU���A�*

	conv_lossѵ>�s�/        )��P	��U���A�*

	conv_loss�ߴ>�U�        )��P	��U���A�*

	conv_lossl-�> �T        )��P	[�U���A�*

	conv_lossŘ�>���#        )��P	�"V���A�*

	conv_lossI��>���        )��P	�SV���A�*

	conv_loss�W�>a��        )��P	��V���A�*

	conv_lossOɱ>�h �        )��P	�V���A�*

	conv_loss�߯>��g�        )��P	��V���A�*

	conv_loss�%�>��        )��P	�#W���A�*

	conv_loss"�>�q%�        )��P	�XW���A�*

	conv_loss��>����        )��P	��W���A�*

	conv_loss�G�>�y�?        )��P	 �W���A�*

	conv_loss�q�>���        )��P	G�W���A�*

	conv_loss���>�ޟ        )��P	.X���A�*

	conv_loss��>����        )��P	�JX���A�*

	conv_loss�V�>J,�        )��P	�zX���A�*

	conv_loss���>��xA        )��P	��X���A�*

	conv_loss�g�>���        )��P	k�X���A�*

	conv_lossX��>{!i}        )��P	�Y���A�*

	conv_loss>�>JT]        )��P	&@Y���A�*

	conv_loss{~�>N�8�        )��P	-wY���A�*

	conv_loss���>�V-&        )��P	�Y���A�*

	conv_loss^�>(�f�        )��P	��Y���A�*

	conv_loss$�>���^        )��P	�Z���A�*

	conv_loss�,�>�U�>        )��P	>OZ���A�*

	conv_loss���>(�2�        )��P	?}Z���A�*

	conv_lossu�>��h        )��P	��Z���A�*

	conv_loss���>�w2�        )��P	Y�Z���A�*

	conv_loss��>�]��        )��P	-[���A�*

	conv_loss�"�>�6��        )��P	)M[���A�*

	conv_loss�ϭ>X>�        )��P	�{[���A�*

	conv_loss��>Kd�_        )��P	m�[���A�*

	conv_loss@��>���o        )��P		�[���A�*

	conv_loss��>����        )��P	m%\���A�*

	conv_lossu~�>�Գ�        )��P	�W\���A�*

	conv_loss�y�>2�M        )��P	C�\���A�*

	conv_loss��>�7v        )��P	��\���A�*

	conv_losst¬>���T        )��P	��\���A�*

	conv_loss+7�>�y{�        )��P	(]���A�*

	conv_loss*1�>�}7        )��P	�[]���A�*

	conv_loss:o�>�6�        )��P	[�]���A�*

	conv_loss��>�/�        )��P	'�]���A�*

	conv_loss譪>n�Hr        )��P	��]���A�*

	conv_loss f�>#ؠ0        )��P	]^���A�*

	conv_loss0v�>���[        )��P	�C^���A�*

	conv_loss�4�>�H�        )��P	kt^���A�*

	conv_loss�>�ގ        )��P	)�^���A�*

	conv_lossv�>���        )��P	��^���A�*

	conv_loss�\�>���        )��P	�_���A�*

	conv_loss�>��h        )��P	}E_���A�*

	conv_loss���>+�y*        )��P	�w_���A�*

	conv_loss���>�\��        )��P	Ч_���A�*

	conv_loss���>�Z��        )��P	��_���A�*

	conv_loss䅪>,��        )��P	�`���A�*

	conv_loss�{�>�ܣ        )��P	 M`���A�*

	conv_loss �>ɻBC        )��P	�`���A�*

	conv_lossX��>X$�J        )��P	��`���A�*

	conv_lossӬ�>�3��        )��P	d�`���A�*

	conv_loss>�>��g*        )��P	�'a���A�*

	conv_loss�.�>7h�=        )��P	�Ya���A�*

	conv_loss��>x���        )��P	i�a���A�*

	conv_loss���>	?�        )��P	��a���A�*

	conv_lossNW�>ę�        )��P	��a���A�*

	conv_loss+�>�Ќ        )��P	�b���A�*

	conv_lossRƪ>��P        )��P	xDb���A�*

	conv_loss���>�[RI        )��P	)sb���A�*

	conv_loss�̭>P��U        )��P	Ѡb���A�*

	conv_loss�x�>�!r        )��P	�b���A�*

	conv_lossw
�>!��        )��P	� c���A�*

	conv_loss�D�>�L�        )��P	�1c���A�*

	conv_loss��>��S        )��P	�ac���A�*

	conv_loss:i�>X��k        )��P	O�c���A�*

	conv_loss���>賛w        )��P	C�c���A�*

	conv_lossh�>U�S�        )��P	y�c���A�*

	conv_lossv�>K|        )��P	?5d���A�*

	conv_loss�֫>|ۜ        )��P	�ed���A�*

	conv_lossP2�>b��	        )��P	v�d���A�*

	conv_lossC�>�R*        )��P	�d���A�*

	conv_loss��>g��        )��P	
�d���A�*

	conv_loss�>BVnl        )��P	z,e���A�*

	conv_loss��>��q        )��P	[e���A�*

	conv_loss���>$_B�        )��P	'�e���A�*

	conv_lossa��>�w��        )��P	��e���A�*

	conv_loss���>�$�        )��P	`�e���A�*

	conv_loss�m�>���7        )��P	P1f���A�*

	conv_loss�ǫ>�戨        )��P	�bf���A�*

	conv_loss렪>Њ�        )��P	m�f���A�*

	conv_loss"�>��ҟ        )��P	_�f���A�*

	conv_lossyͫ>xpW8        )��P	h�f���A�*

	conv_lossk��>��        )��P	�)g���A�*

	conv_loss�Z�>rT�        )��P	}[g���A�*

	conv_losslΩ>�        )��P	g���A�*

	conv_lossA*�>��[�        )��P	�g���A�*

	conv_loss���>[�a�        )��P	�g���A�*

	conv_loss��>��|�        )��P	�$h���A�*

	conv_loss �>R�        )��P	�Uh���A�*

	conv_lossۢ�>ͪN�        )��P	��h���A�*

	conv_lossq�>��:        )��P	�h���A�*

	conv_lossq�>��"        )��P	E�h���A�*

	conv_lossD��>T��        )��P	vi���A�*

	conv_lossA�>���        )��P	�Gi���A�*

	conv_loss۾�>>��        )��P	owi���A�*

	conv_lossB3�>xw�        )��P	~�i���A�*

	conv_loss�k�>���        )��P	��i���A�*

	conv_loss��>��>�        )��P	~j���A�*

	conv_lossl��>Ncc[        )��P	5j���A�*

	conv_lossyD�>�1�        )��P	cj���A�*

	conv_loss��>@e{j        )��P	=�j���A�*

	conv_loss&��>���        )��P	N�j���A�*

	conv_loss���>	)�        )��P	!�j���A�*

	conv_loss|��>���        )��P	�#k���A�*

	conv_loss���>AD�5        )��P	�Tk���A�*

	conv_lossI(�>+���        )��P	��k���A�*

	conv_loss���>}�H!        )��P	ڴk���A�*

	conv_loss�P�><�        )��P	��k���A�*

	conv_loss�T�>}<�        )��P	�l���A�*

	conv_loss��>#>��        )��P	�Bl���A�*

	conv_loss
b�>��J        )��P	�rl���A�*

	conv_loss���>�j',        )��P	̢l���A�*

	conv_loss �>?ށ!        )��P	�l���A�*

	conv_loss���>T��        )��P	� m���A�*

	conv_loss�1�>ꉆ        )��P	�0m���A�*

	conv_loss?~�>����        )��P	`m���A�*

	conv_loss1e�>��E        )��P	��q���A�*

	conv_loss~
�>J'�p        )��P	FJs���A�*

	conv_loss�g�>��a�        )��P	��s���A�*

	conv_loss䟩>�v�        )��P	=�s���A�*

	conv_loss��>[�ڏ        )��P	z�s���A�*

	conv_loss�	�>���S        )��P	�t���A�*

	conv_loss臘>�7�        )��P	bKt���A�*

	conv_loss���>���A        )��P	�t���A�*

	conv_loss͌�>Ń�;        )��P	4�t���A�*

	conv_lossM[�>��        )��P	��t���A�*

	conv_loss�>�ml�        )��P	u���A�*

	conv_loss�n�>�՗        )��P	3Vu���A�*

	conv_loss���>���        )��P	��u���A�*

	conv_loss���>����        )��P	G�u���A�*

	conv_loss筩>P���        )��P	n�u���A�*

	conv_loss�թ>���w        )��P	"v���A�*

	conv_loss8�>'Ώ�        )��P	�Zv���A�*

	conv_loss�O�>��k        )��P	�v���A�*

	conv_lossɫ�>�s�        )��P	��v���A�*

	conv_lossC�>�cS�        )��P	0�v���A�*

	conv_loss_h�>�w�        )��P	w���A�*

	conv_loss$˩>?Z�        )��P	�Fw���A�*

	conv_losse�>[�H        )��P	luw���A�*

	conv_losss�>�$�	        )��P	�w���A�*

	conv_loss��>���        )��P	&�w���A�*

	conv_loss��>��        )��P	�
x���A�*

	conv_loss�%�>
�m�        )��P	"<x���A�*

	conv_loss�]�>� �J        )��P	Jjx���A�*

	conv_losss�>�� �        )��P	/�x���A�*

	conv_loss#��>�T�%        )��P	��x���A�*

	conv_loss��>7��z        )��P	x�x���A�*

	conv_loss���>)El�        )��P	S)y���A�*

	conv_loss�ܩ>�;��        )��P	�Yy���A�*

	conv_loss�h�>���        )��P	-�y���A�*

	conv_lossa��>\=��        )��P	p�y���A�*

	conv_loss,��>���        )��P	��y���A�*

	conv_lossz��>��S        )��P	,z���A�*

	conv_loss�Щ>�K:�        )��P	vbz���A�*

	conv_loss$�>����        )��P	|�z���A�*

	conv_loss(D�> m�        )��P	��z���A�*

	conv_lossę>�|h        )��P	4�z���A�*

	conv_loss���>�ۘ        )��P	(&{���A�*

	conv_loss,&�>�7�        )��P	,T{���A�*

	conv_loss�d�>,J�        )��P	&�{���A�*

	conv_loss�2�>,�E�        )��P	��{���A�*

	conv_losst٩>i�        )��P	��{���A�*

	conv_loss���>e.:        )��P	%|���A�*

	conv_lossW9�>�;,f        )��P	<|���A�*

	conv_loss��>����        )��P	k|���A�*

	conv_lossh\�>M"�        )��P	��|���A�*

	conv_lossX��>��,�        )��P	��|���A�*

	conv_loss���>�S^4        )��P	�}���A�*

	conv_loss'�>��5Y        )��P	cA}���A�*

	conv_loss��>�a+�        )��P	\q}���A�*

	conv_loss�D�>�թ�        )��P	7�}���A�*

	conv_loss���>�m�        )��P	9�}���A�*

	conv_lossZ%�>�Nv�        )��P	�~���A�*

	conv_loss��>���        )��P	i4~���A�*

	conv_loss��>')��        )��P	c~���A�*

	conv_lossȋ�>����        )��P	ڒ~���A�*

	conv_lossw��>+�A        )��P	��~���A�*

	conv_loss�A�>Xt��        )��P	��~���A�*

	conv_loss��>:2.	        )��P	�&���A�*

	conv_loss�K�>�ͳ,        )��P	EZ���A�*

	conv_loss�{�>2�ƶ        )��P	�����A�*

	conv_lossb.�>D_��        )��P	�����A�*

	conv_lossO��>���        )��P	 ����A�*

	conv_loss⧦>��6        )��P	!/����A�*

	conv_loss,2�>m�        )��P	�^����A�*

	conv_losspN�>��[        )��P	������A�*

	conv_loss�)�>���        )��P	j̀���A�*

	conv_loss�>X��        )��P	������A�*

	conv_loss���>�d#�        )��P	�)����A�*

	conv_loss���>�`�r        )��P	\����A�*

	conv_loss[�>��N        )��P	͌����A�*

	conv_loss���>L-1
        )��P	�ʁ���A�*

	conv_loss��>��        )��P	����A�*

	conv_loss���>ˑ��        )��P	i8����A�*

	conv_loss��>S�:        )��P	�h����A�*

	conv_loss�>���2        )��P	������A�*

	conv_loss�٩>EL�        )��P	�ł���A�*

	conv_loss���>(��        )��P	(�����A�*

	conv_loss:"�>���r        )��P	B&����A�*

	conv_loss��>��~        )��P	OU����A�*

	conv_lossd=�>ﲄ        )��P	i�����A�*

	conv_loss׏�>p�8�        )��P	������A�*

	conv_loss���>�w�`        )��P	�����A�*

	conv_loss�>����        )��P	s����A�*

	conv_loss�Z�>��$(        )��P	�G����A�*

	conv_loss�>C�L        )��P	�����A�*

	conv_loss}�>�gFC        )��P	������A�*

	conv_loss&��>��H8        )��P	T�����A�*

	conv_loss���>�+��        )��P	�(����A�*

	conv_lossC��>�aG        )��P	�W����A�*

	conv_lossIA�>�Z�6        )��P	X�����A�*

	conv_lossy�>+        )��P	������A�*

	conv_lossѝ�>�?o/        )��P	�����A�*

	conv_lossQ�>yv��        )��P	H+����A�*

	conv_loss���>��F        )��P	�_����A�*

	conv_loss�ѥ>Kė        )��P	Ϗ����A�*

	conv_lossd�>�պ        )��P	\�����A�*

	conv_lossB�>xL{�        )��P	�����A�*

	conv_loss%��>
|z�        )��P	wF����A�*

	conv_lossȕ�>��U�        )��P	�w����A�*

	conv_loss�[�>u\�e        )��P	 �����A�*

	conv_loss�>!&�b        )��P	wۇ���A�*

	conv_loss��>$"��        )��P	}����A�*

	conv_loss�V�>��[)        )��P	�A����A�*

	conv_loss[ݥ>V��        )��P	<q����A�*

	conv_lossF��>3~�G        )��P	H�����A�*

	conv_loss�C�>Q͒�        )��P	�߈���A�*

	conv_loss<ި>_hr        )��P	q����A�*

	conv_loss ��>W^�        )��P	%B����A�*

	conv_loss걤>jg�b        )��P	mp����A�*

	conv_lossC�><���        )��P	������A�*

	conv_lossI��>�HF�        )��P	�؉���A�*

	conv_losst�>
� �        )��P	�����A�*

	conv_loss�^�>�0V�        )��P	F>����A�*

	conv_loss�Ǧ>p�i        )��P	�l����A�*

	conv_loss&ͥ>����        )��P	⛊���A�*

	conv_loss�;�>�5B�        )��P	AɊ���A�*

	conv_loss)B�>���        )��P	M�����A�*

	conv_loss�X�>&�,q        )��P	�&����A�*

	conv_loss�>!Cf�        )��P	)W����A�*

	conv_lossئ>��        )��P	4�����A�*

	conv_loss���>�!�        )��P	������A�*

	conv_loss�q�>���!        )��P	����A�*

	conv_lossQ��>L��        )��P	�����A�*

	conv_loss5E�>�6�        )��P	�C����A�*

	conv_loss�5�>��Ԩ        )��P	�r����A�*

	conv_loss�>�>i�YF        )��P	�����A�*

	conv_loss�ɦ>=&�<        )��P	�Ԍ���A�*

	conv_loss-H�>�te        )��P	����A�*

	conv_lossHҧ>ݪ=~        )��P	�1����A�*

	conv_loss�æ>]�r2        )��P	a����A�*

	conv_lossD�>x!}        )��P	������A�*

	conv_loss�>"�(H        )��P	������A�*

	conv_lossv�>R�-        )��P	�����A�*

	conv_loss�S�>Gq��        )��P	�����A�*

	conv_lossX�>��b�        )��P		O����A�*

	conv_losss��>x>`�        )��P	�}����A�*

	conv_loss'ӣ>�N��        )��P	P�����A�*

	conv_loss1�>�n�        )��P	�ݎ���A�*

	conv_lossñ�>��:�        )��P	����A�*

	conv_loss�b�>�"O        )��P	;����A�*

	conv_loss�;�>�
��        )��P	�k����A�*

	conv_loss���>X�R	        )��P	E�����A�*

	conv_lossAR�>P��&        )��P	Ώ���A�*

	conv_loss�ޤ>˧��        )��P	������A�*

	conv_lossº�>�Y�H        )��P	!,����A�*

	conv_loss�>@�d�        )��P	�[����A�*

	conv_loss=��>&:�        )��P	Ο����A�*

	conv_loss?��>.��>        )��P	Dΐ���A�	*

	conv_loss�=�>�,D        )��P	p�����A�	*

	conv_lossG�>̃�#        )��P	�+����A�	*

	conv_lossVg�>�֒g        )��P	�^����A�	*

	conv_loss�{�>^���        )��P	������A�	*

	conv_lossn��>t:h        )��P	k�����A�	*

	conv_lossaͥ>گK�        )��P	�����A�	*

	conv_loss&:�>�6�        )��P	z����A�	*

	conv_loss@p�>��:        )��P	�T����A�	*

	conv_loss'��>�uw        )��P	H�����A�	*

	conv_loss���>�)�        )��P	������A�	*

	conv_loss�>񽌹        )��P	�����A�	*

	conv_lossB٣>���Z        )��P	1����A�	*

	conv_lossa��>�/��        )��P	MH����A�	*

	conv_lossQ��>��.F        )��P	,w����A�	*

	conv_loss�j�>�ڱ�        )��P	������A�	*

	conv_loss|�>��0        )��P	�ߓ���A�	*

	conv_loss���>�x�        )��P	�����A�	*

	conv_loss$�>�c7s        )��P	/B����A�	*

	conv_loss�D�>v�{        )��P	Sr����A�	*

	conv_losss�>*�,        )��P	#�����A�	*

	conv_loss ��>��3�        )��P	�Ҕ���A�	*

	conv_lossVQ�>W��        )��P	� ����A�	*

	conv_loss�!�>��h        )��P	�1����A�	*

	conv_lossYZ�>gJ �        )��P	xa����A�	*

	conv_lossF��>��        )��P	P�����A�	*

	conv_loss���><�+        )��P	b�����A�	*

	conv_loss(H�>s�>        )��P	����A�	*

	conv_loss"]�>m�L        )��P	�����A�	*

	conv_lossvs�>���'        )��P	�L����A�	*

	conv_loss�/�>��        )��P	q}����A�	*

	conv_lossU]�>�Y I        )��P	������A�	*

	conv_loss闤>��ǣ        )��P	 ܖ���A�	*

	conv_loss�F�>�}�        )��P	Z
����A�	*

	conv_loss�3�>Ex�        )��P	�9����A�	*

	conv_loss��>�9�M        )��P	�h����A�	*

	conv_lossg��>�LUq        )��P	I�����A�	*

	conv_loss`��>sW�        )��P	Ɨ���A�	*

	conv_loss�M�>־�        )��P	w�����A�	*

	conv_loss�~�>�m        )��P	�"����A�	*

	conv_losse�>s�g�        )��P	�R����A�	*

	conv_loss?��>�
        )��P	������A�	*

	conv_loss[��>�F-        )��P	o�����A�	*

	conv_losse��>`��        )��P	�����A�	*

	conv_lossk�>{�
        )��P	�����A�	*

	conv_loss�x�>� d�        )��P	�G����A�	*

	conv_loss��>�)��        )��P	8y����A�	*

	conv_lossq/�>%�l;        )��P	������A�	*

	conv_loss��>V��        )��P	fܙ���A�	*

	conv_lossG(�>���e        )��P	�t����A�	*

	conv_loss6�>���        )��P	㦛���A�	*

	conv_loss���>Rsr�        )��P	؛���A�	*

	conv_loss� �> :�        )��P	�����A�	*

	conv_lossT�>0�~        )��P	B@����A�	*

	conv_lossX}�>�ӆ�        )��P	�}����A�	*

	conv_loss��>~��>        )��P		�����A�	*

	conv_lossՖ�>��        )��P	a����A�	*

	conv_losst��>�$�4        )��P	�����A�	*

	conv_loss"�>�:b$        )��P	qS����A�	*

	conv_loss��>]eǉ        )��P	������A�	*

	conv_loss�$�>���M        )��P	�Ý���A�	*

	conv_loss�.�>���        )��P	#����A�	*

	conv_loss��>={�d        )��P	�9����A�	*

	conv_loss���>B,�        )��P	o����A�	*

	conv_loss��>���        )��P	v�����A�	*

	conv_lossjp�>��        )��P	Ԟ���A�	*

	conv_loss՘�>�["�        )��P	�����A�	*

	conv_loss��>�M�        )��P	6����A�	*

	conv_loss-��>�9��        )��P	�g����A�	*

	conv_loss�ܡ>���        )��P	����A�	*

	conv_loss�p�>E �        )��P	{ʟ���A�	*

	conv_loss$�>�)-a        )��P	������A�	*

	conv_loss�$�>��E�        )��P	�,����A�	*

	conv_loss���>>c|        )��P	`]����A�	*

	conv_loss���>C<[        )��P	I�����A�	*

	conv_loss��>x��"        )��P	�à���A�	*

	conv_loss�z�>�61        )��P	j�����A�	*

	conv_loss���>p7��        )��P	n'����A�	*

	conv_loss(��>9@��        )��P	�Z����A�	*

	conv_loss$��>�<�        )��P	R�����A�	*

	conv_lossEE�>9��        )��P	+¡���A�	*

	conv_lossAa�>��a        )��P	�����A�	*

	conv_lossb�>4        )��P	�%����A�	*

	conv_loss���>(;"�        )��P	X����A�	*

	conv_loss�=�>e�I�        )��P	������A�	*

	conv_loss�բ>8Uqp        )��P	.�����A�	*

	conv_loss�̠>9C8�        )��P	�����A�	*

	conv_loss_�>!�P        )��P	$����A�	*

	conv_loss���>���0        )��P	zU����A�	*

	conv_loss�Q�>����        )��P	8�����A�	*

	conv_loss��>XO4D        )��P	غ����A�	*

	conv_loss�	�>ŀ-L        )��P	W����A�	*

	conv_loss��>�6�        )��P	� ����A�	*

	conv_lossE~�>�\[b        )��P	S����A�	*

	conv_loss��>�A        )��P	ׄ����A�	*

	conv_loss|v�>aR�,        )��P	
�����A�	*

	conv_loss	�>l�M�        )��P	$����A�	*

	conv_loss�`�>��[        )��P	�����A�	*

	conv_loss}��>5$�&        )��P	bI����A�	*

	conv_loss�_�>�_q        )��P	V�����A�	*

	conv_loss=U�> �/        )��P	�˥���A�	*

	conv_loss�.�>+��        )��P	^�����A�	*

	conv_loss��>�5;S        )��P	�6����A�	*

	conv_loss'�>����        )��P	uk����A�	*

	conv_lossJ�>�$}        )��P	'�����A�	*

	conv_loss�>Ff�J        )��P	�Ц���A�	*

	conv_loss�?�>)��
        )��P	�����A�	*

	conv_loss��>e�Q        )��P	8����A�	*

	conv_loss���>�H�        )��P	�m����A�	*

	conv_loss���>ƃ'a        )��P	+�����A�	*

	conv_lossQ)�>'~7�        )��P	�����A�	*

	conv_loss��>O��M        )��P	�����A�	*

	conv_loss��>�A+        )��P	�I����A�	*

	conv_loss���>f�!K        )��P	{����A�	*

	conv_loss��>���        )��P	ɬ����A�	*

	conv_lossP��>5�        )��P	g����A�	*

	conv_loss��>�{        )��P	� ����A�	*

	conv_loss��>2v�        )��P	�S����A�	*

	conv_loss��>����        )��P	�����A�	*

	conv_loss�o�>��X�        )��P	
�����A�	*

	conv_lossF�>7��         )��P	 �����A�	*

	conv_loss�q�>�wP�        )��P	)����A�	*

	conv_loss6/�>�~�        )��P	�^����A�	*

	conv_lossk��> �?�        )��P	[�����A�	*

	conv_loss���>���        )��P	CǪ���A�	*

	conv_loss�Ԣ>P��        )��P	]����A�	*

	conv_lossf�>�w��        )��P	�9����A�	*

	conv_lossx��>*���        )��P	�k����A�	*

	conv_loss�>7uc        )��P	蠫���A�
*

	conv_lossLP�>cI�         )��P	ԫ���A�
*

	conv_loss=�>�N         )��P	�
����A�
*

	conv_loss.ϣ>)I�*        )��P	�=����A�
*

	conv_lossr��>{3hE        )��P	�p����A�
*

	conv_lossz��>���        )��P	������A�
*

	conv_loss�ǣ>��|C        )��P	Yڬ���A�
*

	conv_lossOН>b(��        )��P	N����A�
*

	conv_loss��>����        )��P	^I����A�
*

	conv_loss^٠>11�        )��P	�}����A�
*

	conv_loss�u�>&i�9        )��P	������A�
*

	conv_loss﷡>W'[�        )��P	����A�
*

	conv_loss�f�>�J[
        )��P	4����A�
*

	conv_loss�+�>#Q'        )��P	!L����A�
*

	conv_loss��>�N        )��P	t~����A�
*

	conv_loss�|�> ���        )��P	������A�
*

	conv_loss�U�>s��        )��P	O����A�
*

	conv_loss/��>�2	�        )��P	�!����A�
*

	conv_lossBӟ>t;jE        )��P	�Z����A�
*

	conv_lossuo�>�&�V        )��P	Տ����A�
*

	conv_loss�1�>/f&        )��P	�į���A�
*

	conv_loss1�>�,��        )��P	F����A�
*

	conv_loss�>X��6        )��P	�A����A�
*

	conv_lossn�>�[��        )��P	�s����A�
*

	conv_loss|$�>��(�        )��P	ꥰ���A�
*

	conv_loss��>��U$        )��P	?ٰ���A�
*

	conv_loss<��>�a*�        )��P		����A�
*

	conv_loss脣>��Ej        )��P	$D����A�
*

	conv_loss�z�>ٗ��        )��P	8x����A�
*

	conv_lossn�>��`        )��P	������A�
*

	conv_losso��>�!Zj        )��P	3����A�
*

	conv_loss_��>/�K         )��P	'����A�
*

	conv_loss��>�{        )��P	^����A�
*

	conv_loss�Þ>��'        )��P	������A�
*

	conv_loss{��>��m,        )��P	������A�
*

	conv_loss���>�<}�        )��P	�����A�
*

	conv_loss?t�>n�
x        )��P	�-����A�
*

	conv_loss�x�>I��O        )��P	(_����A�
*

	conv_loss��>1�c�        )��P	������A�
*

	conv_loss���>ܑ�        )��P	�ǳ���A�
*

	conv_loss9/�>����        )��P	V�����A�
*

	conv_loss�>V@        )��P	�,����A�
*

	conv_loss���>V�~m        )��P	Ee����A�
*

	conv_loss:S�>V�1�        )��P	������A�
*

	conv_lossC��>�-�)        )��P	�ʹ���A�
*

	conv_loss�V�>���f        )��P	������A�
*

	conv_loss�ş>��Y�        )��P	*1����A�
*

	conv_loss(R�>I%��        )��P	Gb����A�
*

	conv_losss
�>�L(        )��P	�����A�
*

	conv_loss���>7��        )��P	ȵ���A�
*

	conv_loss��>ίc�        )��P	R�����A�
*

	conv_loss!�>UPy        )��P	�=����A�
*

	conv_loss$נ>,~Y        )��P	@n����A�
*

	conv_lossB\�>���%        )��P	5�����A�
*

	conv_loss�۝>D�.�        )��P	9۶���A�
*

	conv_loss0B�>owO^        )��P	�����A�
*

	conv_lossn�>Е.�        )��P	�L����A�
*

	conv_loss��>�ʧ�        )��P	�}����A�
*

	conv_lossnk�>����        )��P	~�����A�
*

	conv_loss���>���S        )��P	$����A�
*

	conv_loss��>Lj��        )��P	����A�
*

	conv_loss��>�4w        )��P	AC����A�
*

	conv_lossh�>��Q�        )��P	�t����A�
*

	conv_loss�֠>u�m        )��P	������A�
*

	conv_loss�~�>Dj۔        )��P	xڸ���A�
*

	conv_lossF�>F��b        )��P	�����A�
*

	conv_loss���>iI	        )��P	>����A�
*

	conv_lossH%�>V��        )��P	Iq����A�
*

	conv_loss~%�>�T�        )��P	A�����A�
*

	conv_loss	p�>����        )��P	Թ���A�
*

	conv_lossҨ�>�L�5        )��P	�����A�
*

	conv_loss||�>�_��        )��P	{J����A�
*

	conv_lossk��>�c�        )��P	�}����A�
*

	conv_loss��>6BS        )��P	������A�
*

	conv_loss*�>�u��        )��P	e����A�
*

	conv_loss���>��yn        )��P	�����A�
*

	conv_loss�b�>�2        )��P	QG����A�
*

	conv_lossy�>�9�        )��P	^~����A�
*

	conv_losso�>1d��        )��P	�����A�
*

	conv_loss�J�>��@        )��P	�����A�
*

	conv_loss��>*$        )��P	�����A�
*

	conv_loss}�>�ò�        )��P	�S����A�
*

	conv_loss��>l        )��P	����A�
*

	conv_loss�I�>L>�        )��P	һ����A�
*

	conv_loss�k�>�t�&        )��P	�����A�
*

	conv_lossu �>�XL�        )��P	]!����A�
*

	conv_loss��>��ph        )��P	�S����A�
*

	conv_loss��>L�q        )��P	������A�
*

	conv_loss�7�>�X2�        )��P	<�����A�
*

	conv_loss,��>M�F        )��P	�����A�
*

	conv_lossP��>�(�        )��P	L$����A�
*

	conv_loss짠>���        )��P	�V����A�
*

	conv_loss桟>�H�        )��P	y�����A�
*

	conv_loss�>�%D        )��P	;ƾ���A�
*

	conv_loss>�>q`ֿ        )��P	}�����A�
*

	conv_lossa�>����        )��P	S(����A�
*

	conv_loss�l�>&g[�        )��P	:\����A�
*

	conv_lossi<�>:J�        )��P	������A�
*

	conv_lossX�>����        )��P	������A�
*

	conv_loss��>i�
z        )��P	������A�
*

	conv_loss6��>W��l        )��P	e'����A�
*

	conv_loss�E�>c�nq        )��P	;X����A�
*

	conv_loss�0�>!{�*        )��P	3�����A�
*

	conv_loss��>�?��        )��P	������A�
*

	conv_loss�b�>v�        )��P	�����A�
*

	conv_loss���>���        )��P	� ����A�
*

	conv_lossz�>Tn<�        )��P	JR����A�
*

	conv_loss=>�>��M        )��P	������A�
*

	conv_lossh�>����        )��P	e�����A�
*

	conv_losss'�>����        )��P	S�����A�
*

	conv_loss��>t��        )��P	�%����A�
*

	conv_lossS6�>�彗        )��P	�Z����A�
*

	conv_lossH�>����        )��P	2�����A�
*

	conv_lossJ��>m��        )��P	y�����A�
*

	conv_loss�:�>҃        )��P	������A�
*

	conv_loss�s�>º�V        )��P	�%����A�
*

	conv_loss��>����        )��P	@Y����A�
*

	conv_loss&p�>[�i;        )��P	0�����A�
*

	conv_lossX@�>� +^        )��P	������A�
*

	conv_loss4��>�:o        )��P	T�����A�
*

	conv_loss�<�>�9&L        )��P	p)����A�
*

	conv_loss�"�>�__k        )��P	������A�
*

	conv_lossCR�>�g�        )��P	v�����A�
*

	conv_loss�<�>�K�        )��P	O ����A�
*

	conv_loss�P�>�:�        )��P	+T����A�
*

	conv_loss ��>�cr        )��P	������A�
*

	conv_loss���>-�        )��P	O�����A�
*

	conv_loss�ɜ>R��        )��P	������A�
*

	conv_loss���>1U        )��P	����A�*

	conv_loss�>Km�        )��P	9T����A�*

	conv_loss���>g�i        )��P	f�����A�*

	conv_lossb�>tf 3        )��P	8�����A�*

	conv_loss�˟>��%=        )��P	@�����A�*

	conv_lossqs�>�U�`        )��P	�!����A�*

	conv_loss�j�>T?�A        )��P	�Q����A�*

	conv_lossm�>�        )��P	�����A�*

	conv_loss�Қ>rw�#        )��P	������A�*

	conv_loss���>�ķ�        )��P	�����A�*

	conv_loss1�>h�u        )��P	w"����A�*

	conv_lossv{�>^.�        )��P	�S����A�*

	conv_loss�М>�/�        )��P	������A�*

	conv_loss\	�>��;        )��P	������A�*

	conv_loss �>tg��        )��P	������A�*

	conv_loss�:�>T���        )��P	r����A�*

	conv_loss���>��Z        )��P	WM����A�*

	conv_loss��>"|â        )��P	x}����A�*

	conv_loss���>�c�        )��P	O�����A�*

	conv_loss���>���        )��P	,�����A�*

	conv_loss���>�D~�        )��P	�����A�*

	conv_loss^��>kp��        )��P	�>����A�*

	conv_loss�>�D�        )��P	ho����A�*

	conv_loss�G�>���        )��P	�����A�*

	conv_loss]_�>_4]$        )��P	������A�*

	conv_loss�x�>�bX        )��P	�����A�*

	conv_loss,^�>�D�V        )��P	�<����A�*

	conv_lossX�>d�=        )��P	l����A�*

	conv_loss��>�ܦ         )��P	�����A�*

	conv_lossI�>����        )��P	������A�*

	conv_loss���>g8[        )��P	�����A�*

	conv_loss��>7d        )��P	v6����A�*

	conv_lossS�>���        )��P	=f����A�*

	conv_lossfY�>�X        )��P	 �����A�*

	conv_loss�?�>�S"        )��P	2�����A�*

	conv_loss���>�Ư        )��P	z����A�*

	conv_loss�G�>Y�p        )��P	�?����A�*

	conv_loss�̙>�u�        )��P	o����A�*

	conv_loss�z�>��8
        )��P	\�����A�*

	conv_loss��>�(
j        )��P	 �����A�*

	conv_loss7ŗ>�{N@        )��P	������A�*

	conv_loss���>�S@        )��P	�1����A�*

	conv_loss>�S��        )��P	�c����A�*

	conv_loss*�>cN?�        )��P	ߦ����A�*

	conv_loss�8�>��s        )��P	g�����A�*

	conv_loss�%�>XV�y        )��P	q����A�*

	conv_loss�ŗ>^r�        )��P	�K����A�*

	conv_loss�ʙ>��
^        )��P	g}����A�*

	conv_loss�R�>pnj1        )��P	F�����A�*

	conv_loss���>e_��        )��P	`�����A�*

	conv_loss�ǜ>�K��        )��P	����A�*

	conv_loss��>�K@        )��P	�N����A�*

	conv_loss��>��        )��P	4����A�*

	conv_loss�Z�>����        )��P	r�����A�*

	conv_loss-��>�\|        )��P	�����A�*

	conv_loss�%�>RG��        )��P	�����A�*

	conv_loss쑚>wP        )��P	[Q����A�*

	conv_lossHi�>��%�        )��P	N�����A�*

	conv_loss��>w�Ϡ        )��P	ݷ����A�*

	conv_loss��>)�E�        )��P	������A�*

	conv_loss;��>�
	k        )��P	?%����A�*

	conv_loss��>�nD�        )��P	�W����A�*

	conv_loss���>���        )��P	@�����A�*

	conv_loss_�>0��v        )��P	������A�*

	conv_loss�z�>8��D        )��P	,�����A�*

	conv_lossQ��>v�.        )��P	�!����A�*

	conv_loss�_�>�d@        )��P	SU����A�*

	conv_loss�ܛ>L���        )��P	������A�*

	conv_losss�>���D        )��P	Z�����A�*

	conv_loss�F�>���        )��P	������A�*

	conv_lossn��>��
�        )��P	9-����A�*

	conv_loss6n�>>��        )��P	�`����A�*

	conv_loss���>P�z�        )��P	������A�*

	conv_loss㴚>����        )��P	������A�*

	conv_loss��>���>        )��P	������A�*

	conv_loss �>"�U        )��P	�*����A�*

	conv_lossRz�>)㚊        )��P	N[����A�*

	conv_loss�%�>��ۧ        )��P	r�����A�*

	conv_loss��>��        )��P	�����A�*

	conv_loss��>&��}        )��P	������A�*

	conv_loss.�>�)g�        )��P	|����A�*

	conv_loss��>�]        )��P	>T����A�*

	conv_lossC@�>��+�        )��P	*�����A�*

	conv_lossk��>����        )��P	ܼ����A�*

	conv_loss�H�>�о�        )��P	������A�*

	conv_losswR�>	fN        )��P	�#����A�*

	conv_loss�z�>z�pN        )��P	�S����A�*

	conv_loss�V�>�P��        )��P	�����A�*

	conv_loss%6�><~��        )��P	�����A�*

	conv_loss{��>/��z        )��P	]�����A�*

	conv_loss�;�>{]2�        )��P	g����A�*

	conv_loss���>��L        )��P	gK����A�*

	conv_loss�'�>��<L        )��P	X�����A�*

	conv_losssC�>sw��        )��P	5
����A�*

	conv_loss���>W�,�        )��P	:����A�*

	conv_loss ��>0u
�        )��P	(h����A�*

	conv_lossڒ�>���%        )��P	�����A�*

	conv_lossK�>�EG�        )��P	������A�*

	conv_loss���>�NP        )��P	U�����A�*

	conv_loss዗>@���        )��P	[.����A�*

	conv_loss���>��$b        )��P	�c����A�*

	conv_loss��>�T/G        )��P	������A�*

	conv_loss}x�>�w+}        )��P	������A�*

	conv_lossٸ�>��W�        )��P	

����A�*

	conv_lossM_�>[ڿ|        )��P	�;����A�*

	conv_lossp��>�u�        )��P	<j����A�*

	conv_loss���>����        )��P	D�����A�*

	conv_loss�Ø>�g        )��P	������A�*

	conv_loss�X�>���5        )��P	� ����A�*

	conv_loss,��>Ne�5        )��P	�1����A�*

	conv_loss���>y��z        )��P	�_����A�*

	conv_loss�%�>���        )��P	i�����A�*

	conv_loss5ĕ>�&��        )��P	=�����A�*

	conv_loss�%�>ᢌ�        )��P	%�����A�*

	conv_loss�̖> )�        )��P	�,����A�*

	conv_loss�>3iu�        )��P	�\����A�*

	conv_losst��>6=         )��P	������A�*

	conv_loss�3�>�ʅ,        )��P	�����A�*

	conv_lossO��>�*��        )��P	������A�*

	conv_loss�N�>b{�(        )��P	�����A�*

	conv_lossX�>�+        )��P	�H����A�*

	conv_loss�H�>=�п        )��P	x����A�*

	conv_loss@�>�>�Z        )��P	������A�*

	conv_loss"u�>�S��        )��P	�����A�*

	conv_loss1�>��s        )��P	Y	����A�*

	conv_lossЫ�>Ė�        )��P	8����A�*

	conv_loss��>�5        )��P	_h����A�*

	conv_loss2�>���        )��P	n�����A�*

	conv_loss˛>�A�        )��P	w�����A�*

	conv_loss�V�>dp!�        )��P	r�����A�*

	conv_lossxl�>����        )��P	�(����A�*

	conv_lossg@�>���        )��P	^\����A�*

	conv_loss���>����        )��P	`�����A�*

	conv_loss���>*U0�        )��P	u�����A�*

	conv_loss�`�>�Y        )��P	=�����A�*

	conv_loss�F�>�t�d        )��P	$ ����A�*

	conv_loss>R�n�        )��P	qR����A�*

	conv_loss��>�/nk        )��P	؇����A�*

	conv_lossE�>R-T"        )��P	o�����A�*

	conv_loss�k�>��3[        )��P	n�����A�*

	conv_loss��>���        )��P	F����A�*

	conv_lossws�>ltK!        )��P	�I����A�*

	conv_loss�֑>s*�        )��P	{����A�*

	conv_loss�>Q�pX        )��P	@�����A�*

	conv_loss�P�>�_��        )��P	������A�*

	conv_loss�\�>���        )��P	�!����A�*

	conv_lossg��>�Atb        )��P	�W����A�*

	conv_loss�̑>�7&e        )��P	*�����A�*

	conv_lossӖ�>�fl�        )��P	L�����A�*

	conv_loss��>��        )��P	������A�*

	conv_loss���>yR��        )��P	�.����A�*

	conv_loss\�>�?��        )��P	nu����A�*

	conv_lossC��>�8��        )��P	^�����A�*

	conv_loss��>~y�        )��P	������A�*

	conv_loss%n�>ׂ[        )��P	�����A�*

	conv_loss���>���        )��P	SS����A�*

	conv_loss��>A�        )��P	�����A�*

	conv_loss=q�>�1�        )��P	ʷ����A�*

	conv_loss�m�>���        )��P	������A�*

	conv_loss�P�>�(        )��P	�����A�*

	conv_loss�ߖ>~w��        )��P	�L����A�*

	conv_loss�n�>v
[        )��P	�����A�*

	conv_loss<��>7	$�        )��P	������A�*

	conv_loss�>˪ez        )��P	(�����A�*

	conv_loss�>��ӭ        )��P	�$����A�*

	conv_loss1 �>��R�        )��P	�Y����A�*

	conv_loss|ݒ>��        )��P	�����A�*

	conv_loss��>A���        )��P	~�����A�*

	conv_loss^��>7���        )��P	������A�*

	conv_loss6�>K��        )��P	:$����A�*

	conv_lossep�> �V        )��P	�T����A�*

	conv_loss��>�'R�        )��P	������A�*

	conv_lossU@�>Uh�        )��P	������A�*

	conv_lossK%�>�C�        )��P	������A�*

	conv_lossT��>rc	�        )��P	����A�*

	conv_loss���>)r        )��P	�S����A�*

	conv_loss}��>��X        )��P	˃����A�*

	conv_loss	��>�O"�        )��P	B�����A�*

	conv_loss駑>�.�        )��P	������A�*

	conv_loss�>˲��        )��P	�����A�*

	conv_losscl�>�^f        )��P	�M����A�*

	conv_lossO��>럁        )��P	�~����A�*

	conv_loss��>��1        )��P	ð����A�*

	conv_loss�M�>��-        )��P	������A�*

	conv_loss���>����        )��P	�����A�*

	conv_lossA��>�ֵW        )��P	�D����A�*

	conv_loss$+�>����        )��P	�u����A�*

	conv_loss3Ŗ>�R��        )��P	������A�*

	conv_lossHH�>^Ժ�        )��P	v�����A�*

	conv_loss`#�>d���        )��P	����A�*

	conv_loss���>���I        )��P	�@����A�*

	conv_loss�ː>u���        )��P	�q����A�*

	conv_losscV�>
Q8        )��P	������A�*

	conv_loss*��>��f�        )��P	�?����A�*

	conv_loss"�>��i        )��P	�t����A�*

	conv_loss��>��m        )��P	r�����A�*

	conv_loss��>���        )��P	������A�*

	conv_loss,��>,<��        )��P	����A�*

	conv_loss���>�z�        )��P	�G����A�*

	conv_lossh��>�	2e        )��P	�����A�*

	conv_loss���>�O(l        )��P	N�����A�*

	conv_lossk��>x h�        )��P	I����A�*

	conv_loss��>�~�        )��P	�;����A�*

	conv_loss���>�e�        )��P	q����A�*

	conv_losss�>M:�        )��P	 �����A�*

	conv_lossdG�>�Qz$        )��P	������A�*

	conv_loss��>��6<        )��P	�����A�*

	conv_lossD�><�h�        )��P	�K����A�*

	conv_lossc;�> ��        )��P	�����A�*

	conv_loss�N�>f7�5        )��P	�����A�*

	conv_lossU�>���r        )��P	������A�*

	conv_lossQ�>�3	        )��P	�����A�*

	conv_loss��>����        )��P	�K����A�*

	conv_lossu�>QKy        )��P	-����A�*

	conv_loss��>��        )��P	������A�*

	conv_loss�:�>���        )��P	������A�*

	conv_lossrC�>�d@        )��P	����A�*

	conv_loss���>�T�        )��P	�L����A�*

	conv_loss$��>#c�        )��P	������A�*

	conv_lossק�>F~��        )��P	�����A�*

	conv_loss�Վ>���`        )��P	������A�*

	conv_loss��>�p�        )��P	�%����A�*

	conv_lossٔ>9)        )��P	�\����A�*

	conv_loss8�>˩X#        )��P	�����A�*

	conv_loss�o�>��:j        )��P	������A�*

	conv_loss]�>�K        )��P	������A�*

	conv_losscF�>``+        )��P	�%����A�*

	conv_loss>�>Mq}e        )��P	�[����A�*

	conv_loss���>3��]        )��P	Ύ����A�*

	conv_loss�d�>��)�        )��P	ھ����A�*

	conv_loss�A�>��5�        )��P	U�����A�*

	conv_loss�}�>�y,�        )��P	,+����A�*

	conv_loss�G�>?�        )��P	�f����A�*

	conv_loss��>P��*        )��P	o�����A�*

	conv_loss���>�ιG        )��P	j�����A�*

	conv_lossZF�>A �        )��P	C����A�*

	conv_loss�>աw        )��P	�5����A�*

	conv_loss���>�'.        )��P	Wh����A�*

	conv_loss� �>��^        )��P	������A�*

	conv_loss��>���        )��P	������A�*

	conv_loss�)�>x�&�        )��P	�����A�*

	conv_loss���>��        )��P	�8����A�*

	conv_loss>�>�'\        )��P	fn����A�*

	conv_loss9��>�1�        )��P	Ž����A�*

	conv_lossC�>T�U�        )��P	�����A�*

	conv_lossA��>��&p        )��P	S*����A�*

	conv_loss���>T0��        )��P	jc����A�*

	conv_loss�ۉ>�5*�        )��P	J�����A�*

	conv_loss���>[��        )��P	O�����A�*

	conv_loss���>/�fA        )��P	T����A�*

	conv_loss��>��        )��P	y7����A�*

	conv_loss��>����        )��P	�i����A�*

	conv_loss�>|q�g        )��P	������A�*

	conv_loss���>$6Z�        )��P	�����A�*

	conv_loss�W�>����        )��P	� ���A�*

	conv_lossJ��>��`        )��P	�H ���A�*

	conv_loss�*�>�fy        )��P	�} ���A�*

	conv_loss��> ��        )��P	�� ���A�*

	conv_lossk��>�.G�        )��P	�� ���A�*

	conv_loss��>��4        )��P	b���A�*

	conv_lossV�>O�q�        )��P	G���A�*

	conv_loss�~�> ��        )��P	�x���A�*

	conv_loss� �>�Q��        )��P	����A�*

	conv_loss���>j���        )��P	�����A�*

	conv_lossg:�>Az�.        )��P	>���A�*

	conv_loss�y�>����        )��P	%N���A�*

	conv_lossa"�>�5��        )��P	.����A�*

	conv_lossϢ�>�7\$        )��P	f����A�*

	conv_lossR�>5��        )��P	N����A�*

	conv_loss�;�>2���        )��P	�0���A�*

	conv_loss���>o�1�        )��P	Sa���A�*

	conv_loss�!�>+�>�        )��P	Ӓ���A�*

	conv_loss�>^
F�        )��P	y����A�*

	conv_loss�ć>}�Z�        )��P	<����A�*

	conv_loss��>	��        )��P	d/���A�*

	conv_loss"ň>�	        )��P	�`���A�*

	conv_loss�)�>�(U�        )��P	�����A�*

	conv_lossz�>rn�*        )��P	�����A�*

	conv_loss#��>T�n        )��P	�����A�*

	conv_lossΈ>�_J        )��P	H-���A�*

	conv_loss�u�>t��        )��P	1n���A�*

	conv_loss�؇>GS        )��P	�����A�*

	conv_lossň>����        )��P	6����A�*

	conv_lossȏ�>;h�        )��P	����A�*

	conv_loss?��><>�M        )��P	�6���A�*

	conv_loss��>��d�        )��P	Wi���A�*

	conv_loss��>��        )��P	|����A�*

	conv_lossd	�>#4RN        )��P	����A�*

	conv_losskІ>jh��        )��P	� ���A�*

	conv_lossJ��>� �h        )��P	�0���A�*

	conv_loss8�>���        )��P	�a���A�*

	conv_loss]A�>a��'        )��P	%����A�*

	conv_loss�Յ>+c��        )��P	�����A�*

	conv_loss�<�>��g         )��P	h���A�*

	conv_loss�5�>�E7�        )��P	�8���A�*

	conv_loss���>�>�        )��P	�j���A�*

	conv_loss�-�>$^S@        )��P	x����A�*

	conv_loss�J�>z�^�        )��P	����A�*

	conv_loss���>�W��        )��P	p	���A�*

	conv_loss��>�`�        )��P	�D	���A�*

	conv_loss;�>�xz        )��P	pv	���A�*

	conv_loss|�>��2        )��P	\�	���A�*

	conv_loss��>�X(+        )��P	��	���A�*

	conv_loss���>����        )��P	�
���A�*

	conv_loss���>����        )��P	R
���A�*

	conv_loss�ׇ>_��9        )��P	��
���A�*

	conv_loss}�>):�        )��P	r�
���A�*

	conv_loss�Ʌ>ç�T        )��P	�
���A�*

	conv_loss�y�>���        )��P	)���A�*

	conv_lossO��>OJ�        )��P	�[���A�*

	conv_loss���>d�b�        )��P	����A�*

	conv_loss��>%\,$        )��P	�����A�*

	conv_loss>V�>8|�.        )��P	I����A�*

	conv_loss_�>5��U        )��P	<)���A�*

	conv_loss��>"�b*        )��P	�Z���A�*

	conv_loss�'�>�nVo        )��P	c����A�*

	conv_losscP�>�f��        )��P	Q����A�*

	conv_lossl��>lce        )��P	�����A�*

	conv_lossf6�>�zo        )��P	U$���A�*

	conv_loss�>08D        )��P	�U���A�*

	conv_loss��>�,U�        )��P	ى���A�*

	conv_loss[ �>w��        )��P	�����A�*

	conv_loss��>d5?l        )��P	�����A�*

	conv_loss�΄>��x�        )��P	�!���A�*

	conv_lossC��>�D��        )��P	�T���A�*

	conv_loss"Ճ>-G�!        )��P	)����A�*

	conv_lossnm�>R.�        )��P	�����A�*

	conv_lossg�>&�B�        )��P	�����A�*

	conv_loss��>� 1        )��P	����A�*

	conv_lossv��>ږah        )��P	)K���A�*

	conv_loss�'z>ޮ�        )��P	g|���A�*

	conv_loss���>׷k�        )��P	�����A�*

	conv_loss1��>-��        )��P	@����A�*

	conv_loss=�>����        )��P	����A�*

	conv_lossv��>m�        )��P	�D���A�*

	conv_loss\�z>ʲW�        )��P	�u���A�*

	conv_loss�z�>��`b        )��P	�����A�*

	conv_loss�]�>F�        )��P	�����A�*

	conv_loss���>��        )��P	����A�*

	conv_lossK�>��h�        )��P	HB���A�*

	conv_loss�fz>�)Cv        )��P	�t���A�*

	conv_loss��>�L�        )��P	�����A�*

	conv_lossC�>�>B        )��P	�����A�*

	conv_lossÀ>��K�        )��P	����A�*

	conv_losst�}>3�,;        )��P	cN���A�*

	conv_loss/r�>���@        )��P	 ����A�*

	conv_loss���>��        )��P	����A�*

	conv_lossÁn>����        )��P	s����A�*

	conv_loss��y>�g޷        )��P	�*���A�*

	conv_loss�	y>'�f        )��P	�\���A�*

	conv_loss(�>�Pc        )��P	`����A�*

	conv_loss�P�>Gg        )��P	z����A�*

	conv_loss �z>�z�        )��P	j���A�*

	conv_loss^mv>�Z        )��P	�D���A�*

	conv_loss�b�>oR_d        )��P	ry���A�*

	conv_loss��|>��(        )��P	"����A�*

	conv_loss�3s>�&�        )��P	�����A�*

	conv_loss�>�ض�        )��P	����A�*

	conv_loss�Jq>��k`        )��P	K���A�*

	conv_lossn>}>��t        )��P	�}���A�*

	conv_loss��v>��	        )��P	r����A�*

	conv_loss�s>�%@�        )��P	-����A�*

	conv_lossn�~>\a�        )��P	����A�*

	conv_lossj�~>5��        )��P	�T���A�*

	conv_loss_D}>�'h        )��P	�����A�*

	conv_loss�Q�>����        )��P	����A�*

	conv_lossp̀>�i�L        )��P	�����A�*

	conv_loss�{n>�j�]        )��P	�3���A�*

	conv_loss�|>*��        )��P	�g���A�*

	conv_loss��>��W        )��P	ؚ���A�*

	conv_loss�y>�EC1        )��P	����A�*

	conv_loss'�t>��?        )��P	P���A�*

	conv_loss3!z>h*�R        )��P	�R���A�*

	conv_loss�Mo>���        )��P	�����A�*

	conv_loss/{i>���        )��P	߹���A�*

	conv_loss�g{>)M�$        )��P	����A�*

	conv_lossM�x>�Ƿj        )��P	&���A�*

	conv_losszYr>��7        )��P	0_���A�*

	conv_loss�yv>�m��        )��P	����A�*

	conv_loss	bx>�zJg        )��P	����A�*

	conv_lossgh>��f        )��P	+����A�*

	conv_lossY"�>�ۨ�        )��P	R.���A�*

	conv_loss�u>���        )��P	Wh���A�*

	conv_loss�uh>��8�        )��P	�����A�*

	conv_lossv+z>U�!o        )��P	����A�*

	conv_lossr>^>�&R{        )��P	$ ���A�*

	conv_losse�u>m�p�        )��P	�1���A�*

	conv_lossm��>W�[=        )��P	Nr���A�*

	conv_loss��z>2G?�        )��P	ڤ���A�*

	conv_losslfj>qgo�        )��P	�����A�*

	conv_loss7�x>u~        )��P	����A�*

	conv_loss�)n>��        )��P	F9���A�*

	conv_loss��f>��|�        )��P	Eo���A�*

	conv_lossIx>OTr�        )��P	�	���A�*

	conv_lossg�u>lǊ�        )��P	@���A�*

	conv_lossf�x>KQ(�        )��P	t���A�*

	conv_loss�et>x��        )��P	n����A�*

	conv_loss��n>͕Շ        )��P	=����A�*

	conv_loss��g>Y�i�        )��P	����A�*

	conv_loss�eq>�C�O        )��P	�B���A�*

	conv_loss.	j>��N        )��P	Ut���A�*

	conv_loss�dn>�"�        )��P	�����A�*

	conv_lossߪ}>5S�W        )��P	�����A�*

	conv_loss�l>q�a�        )��P	� ���A�*

	conv_loss�<i>D�-        )��P	�E ���A�*

	conv_loss��i>PAXy        )��P	�z ���A�*

	conv_loss˸p>�*,�        )��P	� ���A�*

	conv_lossxp>����        )��P	�� ���A�*

	conv_lossK�g>z�\        )��P	=-!���A�*

	conv_loss]�d>�NCX        )��P	`!���A�*

	conv_loss4Xi>��        )��P	��!���A�*

	conv_loss�p>Zedo        )��P	G�!���A�*

	conv_loss�Ah>��        )��P	?"���A�*

	conv_lossS�n>�W        )��P	�2"���A�*

	conv_loss@]>��        )��P	�b"���A�*

	conv_lossz%h>ǀ��        )��P	��"���A�*

	conv_loss�]e>J��        )��P	2�"���A�*

	conv_lossFc>;�        )��P	��"���A�*

	conv_lossm�c>b&9�        )��P	\'#���A�*

	conv_loss�Bp>�
�        )��P	K_#���A�*

	conv_lossCpo>�6o�        )��P	��#���A�*

	conv_loss�2n>0m�        )��P	��#���A�*

	conv_loss:i>�A�#        )��P	��#���A�*

	conv_lossuh>T{f        )��P	�1$���A�*

	conv_loss��Y>�pX�        )��P	Ac$���A�*

	conv_loss�d>�5,�        )��P	n�$���A�*

	conv_lossKU>�Ôj        )��P	��$���A�*

	conv_lossn%i>E>h�        )��P	N%���A�*

	conv_lossN�V>�.4�        )��P	�5%���A�*

	conv_losslfm>���
        )��P	f%���A�*

	conv_loss��\>�6�        )��P	N�%���A�*

	conv_loss�f>��˘        )��P	+�%���A�*

	conv_losswbg>!=z        )��P	E
&���A�*

	conv_loss��f>�Q        )��P	�>&���A�*

	conv_loss��h>w��        )��P	<q&���A�*

	conv_loss��`>�75        )��P	��&���A�*

	conv_loss`a>̳�\        )��P	@�&���A�*

	conv_lossU�j>j�"        )��P	6'���A�*

	conv_loss0�a>|1v�        )��P	|>'���A�*

	conv_loss��U>~)!�        )��P	�n'���A�*

	conv_loss�nW>�>.�        )��P	֠'���A�*

	conv_losscd>~B�        )��P	��'���A�*

	conv_loss׍b>=��W        )��P	m
(���A�*

	conv_lossK�e>�K��        )��P	R(���A�*

	conv_loss��c>akM        )��P	m�(���A�*

	conv_loss��W>^+�w        )��P	7�(���A�*

	conv_lossXAV>��O�        )��P	$�(���A�*

	conv_loss:�^>Xѱ        )��P	�")���A�*

	conv_lossh>�        )��P	�Q)���A�*

	conv_loss4Y>8��        )��P	�)���A�*

	conv_lossǄe>Ě)        )��P	t�)���A�*

	conv_losss�^>p~`�        )��P	?�)���A�*

	conv_loss�"_> {n�        )��P	�%*���A�*

	conv_loss��R>����        )��P	�W*���A�*

	conv_loss�j_>��K        )��P	��*���A�*

	conv_loss�`>�D�        )��P	��*���A�*

	conv_loss��]>�14        )��P	�*���A�*

	conv_loss��b>Lp�        )��P	�B+���A�*

	conv_loss�XS>��m         )��P	`v+���A�*

	conv_lossͼ[>#T�+        )��P	�+���A�*

	conv_lossS>��&=        )��P	��+���A�*

	conv_loss;�W>�@�`        )��P	L,���A�*

	conv_loss k>ɗ��        )��P	rF,���A�*

	conv_loss�T>0f2�        )��P	�w,���A�*

	conv_loss�5k>����        )��P	��,���A�*

	conv_loss��W>0Ԕ�        )��P	��,���A�*

	conv_loss�Nb>���r        )��P	�-���A�*

	conv_lossfV>v�<K        )��P	�G-���A�*

	conv_loss�~]>
��        )��P	�-���A�*

	conv_lossZ	R>pI��        )��P	��-���A�*

	conv_loss��U>y|(N        )��P	`�-���A�*

	conv_loss�PF>��t        )��P	 .���A�*

	conv_losszfd>]J��        )��P	�M.���A�*

	conv_losshqM>��v�        )��P	k~.���A�*

	conv_loss�V>��        )��P	 �.���A�*

	conv_loss�`W>)�1|        )��P	�.���A�*

	conv_loss��F>�mI        )��P	�/���A�*

	conv_lossw_>8<�a        )��P	�B/���A�*

	conv_loss�UV>���        )��P	�s/���A�*

	conv_loss��X>c�$�        )��P	�/���A�*

	conv_loss�X>H�+        )��P	9�/���A�*

	conv_loss��T>�.�        )��P	�0���A�*

	conv_lossN	O>bL#x        )��P	u40���A�*

	conv_loss�'X>�<�        )��P	�e0���A�*

	conv_loss��Q>� ^�        )��P	^�0���A�*

	conv_lossدZ>%���        )��P	��0���A�*

	conv_lossh�Y>�	D        )��P	��0���A�*

	conv_loss��O>�;2�        )��P	�'1���A�*

	conv_lossȩM>�,��        )��P	0W1���A�*

	conv_lossJ@>�ׄ�        )��P	̉1���A�*

	conv_lossT�H>�8��        )��P	��1���A�*

	conv_loss��Y>׃�        )��P	�1���A�*

	conv_loss�(R>��B�        )��P	g2���A�*

	conv_loss��Z>��        )��P	�\2���A�*

	conv_loss��J>*s��        )��P	��2���A�*

	conv_loss�XU>�z�9        )��P	��2���A�*

	conv_loss��P>�Qݰ        )��P	��2���A�*

	conv_loss�;O>C�@�        )��P	�&3���A�*

	conv_loss��T>-U�        )��P	�W3���A�*

	conv_loss��L>�ֱ�        )��P	5�3���A�*

	conv_loss�^>���g        )��P	ظ3���A�*

	conv_lossC�O>��ؐ        )��P	��3���A�*

	conv_loss� O>�#��        )��P	�-4���A�*

	conv_lossw�G>�!�        )��P	Sb4���A�*

	conv_loss�@>� �        )��P	;�4���A�*

	conv_loss/�?>!ݕ&        )��P	?�4���A�*

	conv_loss��C>!�Ym        )��P	1�4���A�*

	conv_loss�@>ٯ        )��P	�85���A�*

	conv_loss=�G>ع��        )��P	 l5���A�*

	conv_lossV�P>	S��        )��P	m�5���A�*

	conv_loss/-C>m��        )��P	��5���A�*

	conv_loss��Y>����        )��P	�6���A�*

	conv_loss��:>���        )��P	�F6���A�*

	conv_loss�<O>Z�p        )��P	"}6���A�*

	conv_losse�X>u2�Q        )��P	��6���A�*

	conv_loss�T>k�IP        )��P	y�6���A�*

	conv_loss�@@>�4u9        )��P	U7���A�*

	conv_lossY�>>F>         )��P	�>7���A�*

	conv_loss� Q>Z�+        )��P	zo7���A�*

	conv_loss�F>G8q�        )��P	��7���A�*

	conv_loss�6>� ��        )��P	S�7���A�*

	conv_loss��C>����        )��P	!8���A�*

	conv_loss;�O>��1�        )��P	Q8���A�*

	conv_loss��R>�,<�        )��P	�8���A�*

	conv_lossw�G>��6        )��P	�8���A�*

	conv_loss��F>��K�        )��P	z�8���A�*

	conv_loss��H>�n�        )��P	�9���A�*

	conv_loss��3>eX U        )��P	BD9���A�*

	conv_lossw\>ӛ*        )��P	�t9���A�*

	conv_lossa5>^<օ        )��P	��9���A�*

	conv_loss�*@>o�3�        )��P	:�9���A�*

	conv_loss��U>��8�        )��P	�:���A�*

	conv_lossj�@>�[        )��P	pO:���A�*

	conv_lossF�:>�&��        )��P	��:���A�*

	conv_loss/<L>�kي        )��P	�:���A�*

	conv_loss�XF>rx�        )��P	�:���A�*

	conv_loss�EB>zȬ�        )��P	~;���A�*

	conv_loss��A>�G        )��P	N;���A�*

	conv_losspX@>�2i|        )��P	=~;���A�*

	conv_lossyO>$+�        )��P	�;���A�*

	conv_lossTPG>�9��        )��P	��;���A�*

	conv_lossp};>����        )��P	�<���A�*

	conv_loss[�H>�        )��P	�@<���A�*

	conv_loss�L>5#�F        )��P	��<���A�*

	conv_loss`;>�Ul�        )��P	ִ<���A�*

	conv_loss�B>��ޚ        )��P	��<���A�*

	conv_loss=?>R�9�        )��P	�=���A�*

	conv_loss#�0>�        )��P	tO=���A�*

	conv_loss�]9>���/        )��P	|=���A�*

	conv_loss��(>��Σ        )��P	��=���A�*

	conv_loss�&4>�2?[        )��P	9�=���A�*

	conv_lossxo<>k�/b        )��P	�">���A�*

	conv_loss@W>>kq}        )��P	zY>���A�*

	conv_loss�o,>w��F        )��P	Љ>���A�*

	conv_loss��/>^��        )��P	)�>���A�*

	conv_lossPPM>�ɨ�        )��P	o�>���A�*

	conv_loss$�5>	X�        )��P	$/?���A�*

	conv_loss��=>�{%        )��P	5c?���A�*

	conv_loss�A>�v�        )��P	��?���A�*

	conv_lossi�F>���        )��P	��?���A�*

	conv_loss��A>\�H�        )��P	e�?���A�*

	conv_loss}y&>�o�        )��P	-@���A�*

	conv_loss�X!>�`�        )��P	W\@���A�*

	conv_loss�G>��N=        )��P	͊@���A�*

	conv_loss��=><l�        )��P	�@���A�*

	conv_loss~jB>��i�        )��P	:�@���A�*

	conv_loss �5>��`        )��P	'A���A�*

	conv_loss�X3>��)        )��P	WA���A�*

	conv_lossB�>>���        )��P	d�A���A�*

	conv_lossy&B>_�        )��P	��A���A�*

	conv_loss,�2>
>[�        )��P	�A���A�*

	conv_loss�?>��l)        )��P	�B���A�*

	conv_loss��&>X���        )��P	�JB���A�*

	conv_loss{�B>�*        )��P	�zB���A�*

	conv_loss� :>��        )��P	}�B���A�*

	conv_loss�&>�"�        )��P	��B���A�*

	conv_lossn\:>i1��        )��P	�C���A�*

	conv_lossAe/>$���        )��P	�;C���A�*

	conv_loss�:+>��bW        )��P	�kC���A�*

	conv_lossD>�J_�        )��P	j�C���A�*

	conv_loss�$;>_٨�        )��P	��C���A�*

	conv_lossBf.>���        )��P	��C���A�*

	conv_loss�|(>��]�        )��P	�,D���A�*

	conv_loss��<>���        )��P	�_D���A�*

	conv_lossD�&>	Gg:        )��P	��D���A�*

	conv_lossZ�">j��U        )��P	
�D���A�*

	conv_lossx&4>H\�[        )��P	9�D���A�*

	conv_loss�%>�Fj        )��P	�$E���A�*

	conv_loss�I$>.!D�        )��P	�UE���A�*

	conv_lossel<>~��&        )��P	�E���A�*

	conv_loss�&>Y��        )��P	��E���A�*

	conv_loss=�0>g�U        )��P	0�E���A�*

	conv_lossK9>Q`        )��P	X}J���A�*

	conv_lossWD8>�_��        )��P	�L���A�*

	conv_loss�c=>C��?        )��P	�BL���A�*

	conv_lossR�A>�`��        )��P	rL���A�*

	conv_loss��>����        )��P	q�L���A�*

	conv_loss�A>]]�        )��P	7�L���A�*

	conv_loss��+>�"�        )��P	AM���A�*

	conv_loss4�,>O�<G        )��P	0CM���A�*

	conv_loss�%>���I        )��P	�sM���A�*

	conv_loss/�(>?�w�        )��P	7�M���A�*

	conv_loss̭$>
���        )��P	��M���A�*

	conv_loss�%>�}�        )��P	�N���A�*

	conv_loss�>6>7���        )��P	VN���A�*

	conv_lossc�(>��        )��P		�N���A�*

	conv_loss"C>�|��        )��P	��N���A�*

	conv_lossl7>~ΔL        )��P	��N���A�*

	conv_loss]+>�D�V        )��P	**O���A�*

	conv_loss�R'>�XR        )��P	jYO���A�*

	conv_lossY� >��p        )��P	��O���A�*

	conv_loss�;>Hڢ�        )��P	ʸO���A�*

	conv_loss�=>]5޵        )��P	�O���A�*

	conv_loss�+>=A        )��P	�(P���A�*

	conv_loss�t9>��        )��P	�\P���A�*

	conv_loss-�0>���        )��P	�P���A�*

	conv_loss�*>Ħ�        )��P	�P���A�*

	conv_loss��/>����        )��P	��P���A�*

	conv_loss��,>��{�        )��P	�Q���A�*

	conv_loss��4>M�2`        )��P	lGQ���A�*

	conv_loss�V*>�m�L        )��P	�vQ���A�*

	conv_loss/2(>�6M�        )��P	H�Q���A�*

	conv_loss��5>G�3�        )��P	��Q���A�*

	conv_loss��)>[�~        )��P	6R���A�*

	conv_loss,�2>/�5�        )��P	u?R���A�*

	conv_loss�	>��r*        )��P	�pR���A�*

	conv_loss�e/>�[A�        )��P	2�R���A�*

	conv_loss-_1>�m��        )��P	��R���A�*

	conv_loss��W>�(R#        )��P	{�R���A�*

	conv_losss�1>���        )��P	G-S���A�*

	conv_loss�,>{"�c        )��P	<\S���A�*

	conv_loss�5>�1"        )��P	M�S���A�*

	conv_loss�:*>��ܦ        )��P	y�S���A�*

	conv_loss�0> n>	        )��P	��S���A�*

	conv_loss�I0>9^�h        )��P	2#T���A�*

	conv_loss��$>�9�        )��P	{YT���A�*

	conv_loss�>	�        )��P	�T���A�*

	conv_loss��%>� �        )��P	��T���A�*

	conv_loss~//>:        )��P	|�T���A�*

	conv_lossޝ2>:�M        )��P	�U���A�*

	conv_loss�*>/�!�        )��P	QU���A�*

	conv_lossyx#>T��        )��P	c�U���A�*

	conv_loss:�/>|�        )��P	#�U���A�*

	conv_loss��>sG^�        )��P	�V���A�*

	conv_loss=@>ͥ�(        )��P	\3V���A�*

	conv_loss��>jb        )��P	iiV���A�*

	conv_lossT� >&�        )��P	��V���A�*

	conv_lossnH5>�o�        )��P	$�V���A�*

	conv_loss��3>�!        )��P	�W���A�*

	conv_lossl�3>tIO�        )��P	�1W���A�*

	conv_loss�&>{�ҿ        )��P	iW���A�*

	conv_loss�3!>����        )��P	ԞW���A�*

	conv_loss3I4>���        )��P	e�W���A�*

	conv_loss7� >��@        )��P	NX���A�*

	conv_loss��'>��        )��P	�1X���A�*

	conv_lossV�)>�6A        )��P	tX���A�*

	conv_loss�7>��f�        )��P	�X���A�*

	conv_loss7�>��        )��P	/�X���A�*

	conv_loss7�0>�S�        )��P	;Y���A�*

	conv_loss�Z!>�6W        )��P	�<Y���A�*

	conv_loss�/)>����        )��P	�lY���A�*

	conv_lossL�#>�;�        )��P	6�Y���A�*

	conv_lossJ;&>��,[        )��P	]�Y���A�*

	conv_lossޞ?>[�c$        )��P	gZ���A�*

	conv_lossE�'>�ߘ?        )��P	 0Z���A�*

	conv_loss�� >p{*w        )��P	R^Z���A�*

	conv_loss�T!>MT'        )��P	,�Z���A�*

	conv_loss�s:>^<��        )��P	��Z���A�*

	conv_loss�X;>�}�        )��P	��Z���A�*

	conv_loss��>��U4        )��P	 2[���A�*

	conv_loss�>9Ec�        )��P	�e[���A�*

	conv_loss�y(>����        )��P	�[���A�*

	conv_loss�>�B��        )��P	��[���A�*

	conv_loss��>1��%        )��P	�[���A�*

	conv_loss]>2>_
3�        )��P	�'\���A�*

	conv_loss[s>\�        )��P	�V\���A�*

	conv_loss�}+>��%G        )��P	i�\���A�*

	conv_loss�4>����        )��P	[�\���A�*

	conv_loss�*>���        )��P	��\���A�*

	conv_loss�R>�|        )��P	�']���A�*

	conv_lossC+>�b�        )��P	�Y]���A�*

	conv_loss-&>6��6        )��P	��]���A�*

	conv_loss�>U$W�        )��P	F�]���A�*

	conv_loss2/>W�{R        )��P	�]���A�*

	conv_loss�L!>��V        )��P	�)^���A�*

	conv_loss��(>O��        )��P	�Y^���A�*

	conv_lossL�;>�(�        )��P	��^���A�*

	conv_loss:s:>��X        )��P	�^���A�*

	conv_losse�> #�        )��P	(�^���A�*

	conv_loss�>�3�
        )��P	/_���A�*

	conv_loss��>�ko�        )��P	qG_���A�*

	conv_lossR<>�-�`        )��P	v_���A�*

	conv_loss�!>dJR�        )��P	��_���A�*

	conv_loss�> b�        )��P	U�_���A�*

	conv_loss.�>���        )��P	�`���A�*

	conv_loss'�">T�P�        )��P	EI`���A�*

	conv_loss?>�6|        )��P	c}`���A�*

	conv_lossq#>X��        )��P	̺`���A�*

	conv_loss�G#>?�{�        )��P	��`���A�*

	conv_loss/^)>A1t        )��P	%a���A�*

	conv_loss�M5>����        )��P	5Pa���A�*

	conv_loss[B>"/�        )��P	0�a���A�*

	conv_loss�4>�>">        )��P	L�a���A�*

	conv_losse�)>��ʌ        )��P	�a���A�*

	conv_loss4k	>�)��        )��P	"b���A�*

	conv_loss�>IH�        )��P	lTb���A�*

	conv_loss.�>��Y        )��P	b�b���A�*

	conv_loss�/>-��o        )��P	��b���A�*

	conv_loss>���        )��P	&�b���A�*

	conv_loss��5>��fg        )��P	�*c���A�*

	conv_loss�k!>1H�^        )��P	[c���A�*

	conv_loss�6>A�        )��P	ƌc���A�*

	conv_loss^�(>[��        )��P	K�c���A�*

	conv_lossw�>9�        )��P	O�c���A�*

	conv_lossG�>p�4�        )��P	nd���A�*

	conv_loss)>z�        )��P	'Jd���A�*

	conv_loss�>(1v�        )��P	rzd���A�*

	conv_loss>��A        )��P	��d���A�*

	conv_loss�Q>̈��        )��P	��d���A�*

	conv_lossX�>	@1�        )��P	�e���A�*

	conv_lossm�>9�?�        )��P	�Ie���A�*

	conv_loss�,>Fn{M        )��P	ce���A�*

	conv_loss/>�9        )��P	k�e���A�*

	conv_lossї0>ο��        )��P	�e���A�*

	conv_loss�>�/.�        )��P	�f���A�*

	conv_loss�;>Td        )��P	�@f���A�*

	conv_loss�>��B&        )��P	qf���A�*

	conv_loss�>��H;        )��P	Ġf���A�*

	conv_loss4�&>a�u        )��P	��f���A�*

	conv_loss��>,v�j        )��P	�g���A�*

	conv_loss��!>��^        )��P	�2g���A�*

	conv_lossn�>�"P�        )��P	�dg���A�*

	conv_loss��>��>        )��P	͔g���A�*

	conv_loss{�#>[�9        )��P	t�g���A�*

	conv_loss.�>�S��        )��P	N�g���A�*

	conv_loss��>���b        )��P	�(h���A�*

	conv_lossq�">~�d        )��P	PWh���A�*

	conv_loss��>��$        )��P	��h���A�*

	conv_lossѼ>H�        )��P	�h���A�*

	conv_lossU�%>�M�        )��P	��h���A�*

	conv_loss=�!>�T�        )��P	�i���A�*

	conv_lossр>FFΫ        )��P	CKi���A�*

	conv_loss8>>Fm[        )��P	{i���A�*

	conv_loss
N>��J        )��P	��i���A�*

	conv_loss��$>G�J]        )��P	x�i���A�*

	conv_loss�>Ƅ�z        )��P	j���A�*

	conv_loss��">_^�        )��P	hPj���A�*

	conv_loss��>��y        )��P	V�j���A�*

	conv_loss`�>��        )��P	e�j���A�*

	conv_loss3� >�B��        )��P	��j���A�*

	conv_lossR>˦�r        )��P	k���A�*

	conv_lossؿ">�g        )��P	�Tk���A�*

	conv_loss��&>����        )��P	��k���A�*

	conv_lossÉ>I�r�        )��P	:�k���A�*

	conv_loss�5>A>        )��P	��k���A�*

	conv_loss��>Ë�1        )��P	 /l���A�*

	conv_loss�M&>�S�        )��P	G`l���A�*

	conv_lossM�*>P�M�        )��P	�l���A�*

	conv_losst�>�{ �        )��P	��l���A�*

	conv_loss��>-�%        )��P	`m���A�*

	conv_lossW>���        )��P	�=m���A�*

	conv_loss�>���        )��P	om���A�*

	conv_loss8u>����        )��P	ßm���A�*

	conv_loss�
>̣��        )��P	��m���A�*

	conv_loss
�>�E�        )��P	An���A�*

	conv_loss��=�['B        )��P	R1n���A�*

	conv_lossT�$>�b-;        )��P	T`n���A�*

	conv_loss�}>C�s�        )��P	��n���A�*

	conv_loss0r4>����        )��P	˾n���A�*

	conv_loss!>4��        )��P	��n���A�*

	conv_loss�Z>'�U�        )��P	�'o���A�*

	conv_loss�>�g�Y        )��P	�eo���A�*

	conv_loss��>���        )��P	�o���A�*

	conv_loss�>湜�        )��P	��o���A�*

	conv_loss�>�H        )��P	5�o���A�*

	conv_loss=�>���        )��P	�'p���A�*

	conv_loss�>���        )��P	�Wp���A�*

	conv_loss�S>|�<8        )��P	Јp���A�*

	conv_losst�>$�@        )��P	`�p���A�*

	conv_loss���=R�!        )��P	�p���A�*

	conv_loss��
>ċ        )��P	3!q���A�*

	conv_loss=>��ZG        )��P	4Pq���A�*

	conv_loss9�>v        )��P	N�q���A�*

	conv_lossY�>1�V        )��P	I�q���A�*

	conv_loss�^>��C�        )��P	.�q���A�*

	conv_loss�=���}        )��P	"r���A�*

	conv_lossiG>櫽�        )��P	�>r���A�*

	conv_losslR+>�o��        )��P	�nr���A�*

	conv_lossd�>5��        )��P	��r���A�*

	conv_lossF�$>�a        )��P	��r���A�*

	conv_lossG�>��(�        )��P	�s���A�*

	conv_loss��>�
H�        )��P	�3s���A�*

	conv_loss��>n_��        )��P	xbs���A�*

	conv_loss�>Q�i�        )��P	��t���A�*

	conv_loss�4>�=        )��P	�%u���A�*

	conv_loss>��Q        )��P	�Tu���A�*

	conv_loss�X>�Rp        )��P	`�u���A�*

	conv_loss�>3i        )��P	Ƿu���A�*

	conv_loss�(>N/9        )��P	?�u���A�*

	conv_loss�1 >6ƹ�        )��P	v���A�*

	conv_loss��>�G        )��P	�Gv���A�*

	conv_loss��>u`X�        )��P	�}v���A�*

	conv_loss��>b�X5        )��P	z�v���A�*

	conv_loss�>�G7�        )��P	��v���A�*

	conv_lossu�>��M+        )��P	�(w���A�*

	conv_loss�,>�S        )��P	"ew���A�*

	conv_loss��>� ��        )��P	��w���A�*

	conv_loss�m>�4*f        )��P	��w���A�*

	conv_lossX�>��`        )��P	��w���A�*

	conv_loss~H>���        )��P	'(x���A�*

	conv_loss�U>�n�        )��P	Zx���A�*

	conv_loss�>��2�        )��P	�x���A�*

	conv_lossU� >�J        )��P	g�x���A�*

	conv_loss�] >^���        )��P	W�x���A�*

	conv_loss�A>�HV�        )��P	!y���A�*

	conv_lossq�">]��z        )��P	<Wy���A�*

	conv_lossH�>-.�3        )��P	��y���A�*

	conv_lossR~>�¦�        )��P	!�y���A�*

	conv_loss�M>1�N        )��P	(�y���A�*

	conv_loss��	>ZR9        )��P	�#z���A�*

	conv_loss���=��i        )��P	BUz���A�*

	conv_loss�r>�73�        )��P	a�z���A�*

	conv_loss=�>/�g        )��P	�z���A�*

	conv_loss>g>I���        )��P	�z���A�*

	conv_loss��
>����        )��P	�{���A�*

	conv_loss�>Rp�        )��P	�F{���A�*

	conv_loss0)>�}�        )��P	v{���A�*

	conv_lossv�>�'
�        )��P	��{���A�*

	conv_loss��	>��E�        )��P	��{���A�*

	conv_loss�>��T�        )��P	�|���A�*

	conv_loss�!>_�        )��P	{;|���A�*

	conv_loss�g.>�a��        )��P	{l|���A�*

	conv_loss!� >�3%�        )��P	n�|���A�*

	conv_loss��>�� e        )��P	��|���A�*

	conv_loss0�>�'�        )��P	[�|���A�*

	conv_lossL�>��3�        )��P	�1}���A�*

	conv_loss->��        )��P	�`}���A�*

	conv_loss2>�#�        )��P	2�}���A�*

	conv_lossQ6>���        )��P	P�}���A�*

	conv_loss�&>���        )��P	��}���A�*

	conv_loss�
>���n        )��P	!~���A�*

	conv_loss\��=�ذ�        )��P	�O~���A�*

	conv_loss9>7:�        )��P	\�~���A�*

	conv_loss��>ma��        )��P	|�~���A�*

	conv_loss�D>���        )��P	��~���A�*

	conv_lossc >�Cx:        )��P	�%���A�*

	conv_loss�.�=���
        )��P	8U���A�*

	conv_loss��>h-?t        )��P	S����A�*

	conv_lossj8>Μֶ        )��P	�����A�*

	conv_loss>S�6�        )��P	0����A�*

	conv_losss\>�y+�        )��P	z,����A�*

	conv_loss��=���        )��P	�d����A�*

	conv_loss89>��E        )��P	�����A�*

	conv_loss��>>�S\        )��P	�����A�*

	conv_loss�>()x9        )��P	�����A�*

	conv_loss�Y>3Ǎ        )��P	�T����A�*

	conv_loss�>�:;        )��P	D�����A�*

	conv_loss���=��n�        )��P	������A�*

	conv_loss�>>�փ�        )��P	����A�*

	conv_lossDa�=ˍ�Y        )��P	^����A�*

	conv_loss��>
��        )��P	�I����A�*

	conv_lossz.>��d        )��P	�z����A�*

	conv_loss
�$>��        )��P	ʯ����A�*

	conv_loss�@'>�T��        )��P	�����A�*

	conv_loss��=��g$        )��P	%����A�*

	conv_losse�>E�        )��P	_D����A�*

	conv_lossW�>�I��        )��P	�s����A�*

	conv_lossZ�	>�mo�        )��P	O�����A�*

	conv_loss
E�=ClH�        )��P	ك���A�*

	conv_loss6�=�Q��        )��P	
	����A�*

	conv_lossՅ>���        )��P	�;����A�*

	conv_lossRe>	���        )��P	Fq����A�*

	conv_losslH>�#K        )��P	������A�*

	conv_loss��>��        )��P	�ބ���A�*

	conv_loss��>�$.�        )��P	�����A�*

	conv_loss5� >��M�        )��P	�C����A�*

	conv_loss�&>���        )��P	�u����A�*

	conv_lossv��=M!��        )��P	������A�*

	conv_loss*{>�y�}        )��P	P߅���A�*

	conv_loss=I>��"        )��P	�����A�*

	conv_lossq!>�?F�        )��P	�C����A�*

	conv_loss��>����        )��P	}����A�*

	conv_loss��>��r�        )��P	C�����A�*

	conv_lossy�>_��R        )��P	�����A�*

	conv_loss�i>���x        )��P	� ����A�*

	conv_loss�D>��~        )��P	5Q����A�*

	conv_loss��>QW��        )��P	�����A�*

	conv_loss��=�P�        )��P	i�����A�*

	conv_lossc{>s�K        )��P	�����A�*

	conv_loss|v>�)w�        )��P	#"����A�*

	conv_loss�x>�t�C        )��P	~Y����A�*

	conv_loss$�>ǖ��        )��P	5�����A�*

	conv_loss5	>�_��        )��P	[�����A�*

	conv_loss]�>1J�~        )��P	h����A�*

	conv_loss��>��F2        )��P	�6����A�*

	conv_loss���=����        )��P	ji����A�*

	conv_loss�Q>����        )��P	8�����A�*

	conv_loss�O>ܓ��        )��P	5Ή���A�*

	conv_loss���=Ԫ��        )��P	"����A�*

	conv_loss;>]jk        )��P	�8����A�*

	conv_loss�~>o�;        )��P	�l����A�*

	conv_lossxY
>zZG        )��P	������A�*

	conv_loss�>އ�_        )��P	�����A�*

	conv_loss�>
���        )��P	V����A�*

	conv_lossۻ�=�8�        )��P	�Q����A�*

	conv_lossqm	>�pN`        )��P	`�����A�*

	conv_loss�v>��+�        )��P	1�����A�*

	conv_loss�U>�9E        )��P	n����A�*

	conv_loss">6�L�        )��P	J����A�*

	conv_loss�| >�1�C        )��P	�N����A�*

	conv_loss�>ꙹe        )��P	������A�*

	conv_lossB>� �        )��P	�����A�*

	conv_loss�I>���        )��P	H����A�*

	conv_lossQ�>B��l        )��P	�����A�*

	conv_loss�;>�6)W        )��P	BI����A�*

	conv_loss��=�x)�        )��P	?�����A�*

	conv_lossl�>s��        )��P	P�����A�*

	conv_loss��>�y�        )��P	����A�*

	conv_loss���=�I        )��P	q����A�*

	conv_loss�&>-]�        )��P	=F����A�*

	conv_loss��=A5�        )��P	�v����A�*

	conv_loss�i�=���{        )��P	+�����A�*

	conv_loss�� >��v        )��P	ڎ���A�*

	conv_loss���=K���        )��P	l����A�*

	conv_loss�_>^�        )��P	�;����A�*

	conv_loss�	�=w��        )��P	�n����A�*

	conv_loss6J�=s�        )��P	t�����A�*

	conv_loss�w>�k�        )��P	�я���A�*

	conv_loss�>Ù�)        )��P	R����A�*

	conv_loss0�>�f�        )��P	�4����A�*

	conv_loss��>4�O        )��P	?e����A�*

	conv_loss?�>���        )��P	f�����A�*

	conv_loss�H�=iZ�        )��P	�ǐ���A�*

	conv_loss�6>Pa��        )��P	������A�*

	conv_loss�D�=-V��        )��P	�)����A�*

	conv_loss2b>�"��        )��P	!Y����A�*

	conv_losst>c='�        )��P	�����A�*

	conv_loss[�= �h6        )��P	W�����A�*

	conv_loss���=���        )��P	�����A�*

	conv_loss�/>mi�-        )��P	)����A�*

	conv_loss��>�=        )��P	�L����A�*

	conv_loss��>�k�%        )��P	c����A�*

	conv_losst�>�@��        )��P	˯����A�*

	conv_loss/�
>S���        )��P	�����A�*

	conv_lossTD>�        )��P	�%����A�*

	conv_lossS� >4�^S        )��P	�V����A�*

	conv_loss�>m�u0        )��P	�����A�*

	conv_lossIw�=�c	        )��P	������A�*

	conv_loss��=�z�[        )��P	�����A�*

	conv_loss�> N�        )��P	�%����A�*

	conv_loss�>���8        )��P	 V����A�*

	conv_lossf��=h5�i        )��P	`�����A�*

	conv_loss��>�p�        )��P	����A�*

	conv_loss�>���        )��P	������A�*

	conv_loss��>��1�        )��P	G����A�*

	conv_lossr�>�i?#        )��P	K|����A�*

	conv_lossԯ>vX}        )��P	������A�*

	conv_lossۋ>�{�[        )��P	������A�*

	conv_lossh>��j        )��P	�����A�*

	conv_loss���=�1��        )��P	C����A�*

	conv_loss3�>�T�        )��P	/u����A�*

	conv_loss��>��        )��P	è����A�*

	conv_loss��>�c3�        )��P	ۖ���A�*

	conv_loss���=-��        )��P	�����A�*

	conv_loss5�
>'��        )��P	=����A�*

	conv_loss�
>y�^�        )��P	Qm����A�*

	conv_loss�W>u�+�        )��P	����A�*

	conv_loss
>*5e�        )��P	�ϗ���A�*

	conv_loss{�>k���        )��P	������A�*

	conv_loss��>�j��        )��P	0����A�*

	conv_loss1F>9a g        )��P	�`����A�*

	conv_loss��>��R        )��P	蒘���A�*

	conv_loss��>�18�        )��P	}Ø���A�*

	conv_loss�	>N�ڧ        )��P	�����A�*

	conv_loss�z�=�R�        )��P	�$����A�*

	conv_loss\�=/���        )��P	�W����A�*

	conv_loss�3�=䪟H        )��P		�����A�*

	conv_lossZ;>CZ��        )��P	3�����A�*

	conv_loss���=�o�        )��P	o����A�*

	conv_loss�T�="�fv        )��P	&����A�*

	conv_lossD�>�5$        )��P	UZ����A�*

	conv_lossy�>�}h        )��P	ܒ����A�*

	conv_loss�a>Ҕ�        )��P	�Ś���A�*

	conv_loss���=���        )��P	%�����A�*

	conv_loss���=����        )��P	�)����A�*

	conv_lossn�>~+%        )��P	�[����A�*

	conv_lossF>�Spn        )��P	b�����A�*

	conv_loss�A�=Qpr        )��P	ě���A�*

	conv_lossܺ=��_        )��P	������A�*

	conv_lossk��=3J        )��P	d%����A�*

	conv_loss���=0ӗ�        )��P	V����A�*

	conv_loss�6>2��        )��P	ˆ����A�*

	conv_loss2�=�&        )��P	������A�*

	conv_loss��=�g@�        )��P	�Q����A�*

	conv_loss-��=9o�s        )��P	C�����A�*

	conv_lossb�>���        )��P	������A�*

	conv_loss�>�Y�        )��P	s����A�*

	conv_loss�#�=����        )��P	8����A�*

	conv_loss-B>%}�        )��P	N����A�*

	conv_loss�a>�& \        )��P	�����A�*

	conv_loss7; >ؾ"        )��P	������A�*

	conv_lossLb�=�V�         )��P	�����A�*

	conv_loss-H>Xz{�        )��P	h&����A�*

	conv_loss*2�=r(@        )��P	3X����A�*

	conv_loss�U�=u@a�        )��P	И����A�*

	conv_loss	,�=����        )��P	9ʠ���A�*

	conv_lossks�=�[�        )��P	������A�*

	conv_loss��
>���2        )��P	{1����A�*

	conv_loss�&�=�JG�        )��P	b����A�*

	conv_losspz�=#ā�        )��P	������A�*

	conv_lossv�>`)^e        )��P	�ס���A�*

	conv_loss��=��¢        )��P	�
����A�*

	conv_loss��>�ɻ�        )��P	�?����A�*

	conv_lossB>M��        )��P	�s����A�*

	conv_losso��=�*�        )��P	������A�*

	conv_loss�> >6��        )��P	�����A�*

	conv_loss��>�y        )��P	�����A�*

	conv_loss���=O|��        )��P	�@����A�*

	conv_lossg&�=��i�        )��P	�s����A�*

	conv_losskt>Un�u        )��P	)�����A�*

	conv_loss�\�=M��Y        )��P	�գ���A�*

	conv_loss�=�9sg        )��P	q����A�*

	conv_loss<�=3�        )��P	G9����A�*

	conv_loss�>zO�p        )��P	$~����A�*

	conv_lossK��=�{�        )��P	�����A�*

	conv_loss��>���        )��P	������A�*

	conv_loss���=p�T        )��P	c����A�*

	conv_loss�C�=�G�y        )��P	5P����A�*

	conv_loss�s�=N^��        )��P	 �����A�*

	conv_loss��=-J��        )��P	������A�*

	conv_loss*.>��s        )��P	����A�*

	conv_loss���=�A2�        )��P	}����A�*

	conv_lossJr >DU$�        )��P	�J����A�*

	conv_lossJc >���        )��P	%�����A�*

	conv_loss�Z>�,|^        )��P	̿����A�*

	conv_lossh>	�$,        )��P	#����A�*

	conv_loss	�>ᩕ�        )��P	,$����A�*

	conv_loss
��=��8e        )��P	�V����A�*

	conv_loss�-�=D��        )��P	戧���A�*

	conv_loss�.>a��        )��P	�˧���A�*

	conv_loss/S�=�U�G        )��P	������A�*

	conv_lossߦ�=��        )��P	b,����A�*

	conv_loss/�=y��2        )��P	]^����A�*

	conv_loss��=7��Q        )��P	U�����A�*

	conv_loss�N�=��;�        )��P	k����A�*

	conv_lossX��=
Ix        )��P	j#����A�*

	conv_loss	��=ޗ٭        )��P	�W����A�*

	conv_loss���=�@`.        )��P	������A�*

	conv_loss���=D���        )��P	������A�*

	conv_loss���=�
�<        )��P	������A�*

	conv_loss���=v�m        )��P	f(����A�*

	conv_loss�>Q0I)        )��P	Z����A�*

	conv_loss(�>s�d}        )��P	������A�*

	conv_loss�6>�5�]        )��P	.�����A�*

	conv_loss<�>_��h        )��P	�����A�*

	conv_losszF>L���        )��P	)����A�*

	conv_loss��>h�        )��P	Z����A�*

	conv_lossks�=��˘        )��P	8�����A�*

	conv_loss���=��xy        )��P	Hū���A�*

	conv_loss��==h�v        )��P	������A�*

	conv_loss���=�fh        )��P	�/����A�*

	conv_loss7�>�2        )��P	�_����A�*

	conv_lossL��=>��        )��P	������A�*

	conv_losslw�=nR:�        )��P	�Ĭ���A�*

	conv_lossh�=��_1        )��P	������A�*

	conv_loss�p�=wE)]        )��P	�/����A�*

	conv_loss��>�ʮ�        )��P	�b����A�*

	conv_loss���=�J1        )��P	������A�*

	conv_lossf��=Zt�        )��P	Rƭ���A�*

	conv_loss̗�=�~�        )��P	������A�*

	conv_loss�,�=��w?        )��P	L6����A�*

	conv_lossþ�=a/E�        )��P	�m����A�*

	conv_lossc�>�Q��        )��P	������A�*

	conv_lossv�>��        )��P	Ѯ���A�*

	conv_loss�{�==a�P        )��P	�����A�*

	conv_loss���=����        )��P	.C����A�*

	conv_loss>�>�L-        )��P	Cu����A�*

	conv_loss>��        )��P	�����A�*

	conv_lossLE�=�BWY        )��P	�ׯ���A�*

	conv_losst�=f��        )��P	�����A�*

	conv_loss���=��QR        )��P	mI����A�*

	conv_lossS�>c�2        )��P	]z����A�*

	conv_loss��>SE��        )��P	������A�*

	conv_loss/��=��f@        )��P	����A�*

	conv_loss���=Lyi�        )��P	#����A�*

	conv_loss��='�G        )��P	�T����A�*

	conv_loss�e >p��7        )��P	������A�*

	conv_loss-�=p�-        )��P	ܷ����A�*

	conv_loss��=���.        )��P	m����A�*

	conv_loss�7 >�m�        )��P	 (����A�*

	conv_loss�=�	�]        )��P	�Y����A�*

	conv_lossS��=q��J        )��P	������A�*

	conv_lossB{�=�� .        )��P	�����A�*

	conv_loss���=,        )��P	�]����A�*

	conv_loss��=�*        )��P	ݎ����A�*

	conv_lossG?�=B��O        )��P	������A�*

	conv_loss�0�=^L/        )��P	�����A�*

	conv_lossu�=�EY\        )��P	/ ����A�*

	conv_loss�o�=�	
�        )��P	�e����A�*

	conv_loss���=���Q        )��P	2�����A�*

	conv_loss(+�=�!�Q        )��P	�ĸ���A�*

	conv_loss��=��B�        )��P	D�����A�*

	conv_loss���=X���        )��P	)1����A�*

	conv_loss>H�
�        )��P	�o����A�*

	conv_lossi�>�v        )��P	ힹ���A�*

	conv_loss:��=m��!        )��P	+Ϲ���A�*

	conv_loss���=��M�        )��P	������A�*

	conv_lossy?�=��        )��P	�-����A�*

	conv_loss���=���        )��P	�^����A�*

	conv_loss�\�=ޘ�b        )��P	T�����A�*

	conv_loss��>�@M        )��P	[ں���A�*

	conv_loss^�=�d�        )��P	[����A�*

	conv_loss;�>6z `        )��P	�?����A�*

	conv_loss���=���        )��P	�����A�*

	conv_loss��>�a`        )��P	c�����A�*

	conv_lossT'�=t��a        )��P	�����A�*

	conv_lossZ��=�k>�        )��P	C����A�*

	conv_loss��>ڿ6        )��P	9I����A�*

	conv_loss���=�Q2b        )��P	������A�*

	conv_loss$,�=|��        )��P	|�����A�*

	conv_loss��>�#��        )��P	�����A�*

	conv_loss)��=.�E        )��P	�����A�*

	conv_loss�=C?        )��P	7C����A�*

	conv_loss'��=�{�        )��P	Á����A�*

	conv_loss[�=!�c        )��P	屽���A�*

	conv_loss��>5I��        )��P	�����A�*

	conv_loss�>nm�        )��P	L����A�*

	conv_lossv>��ҽ        )��P	�@����A�*

	conv_loss$>����        )��P	/o����A�*

	conv_loss�(�=���        )��P	$�����A�*

	conv_loss0
�=G}�        )��P	�Ӿ���A�*

	conv_lossm!>0�
�        )��P	�����A�*

	conv_loss�F�=��p         )��P	03����A�*

	conv_loss�g�=<��        )��P	�b����A�*

	conv_losss!>�j|�        )��P	t�����A�*

	conv_lossG��=V4��        )��P	�ÿ���A�*

	conv_loss���=�Z[        )��P	Y����A�*

	conv_loss[4�='��u        )��P	�!����A�*

	conv_lossv�=���a        )��P	Q����A�*

	conv_loss��=�?�;        )��P	�����A�*

	conv_loss�W�=S�?�        )��P	�����A�*

	conv_loss�?�=���a        )��P	������A�*

	conv_lossrQ�=�        )��P	m����A�*

	conv_loss1�=�-'�        )��P	|T����A�*

	conv_lossS0>ԕC        )��P	�����A�*

	conv_loss֗�=Zߦ        )��P	������A�*

	conv_loss �	>�B�        )��P	������A�*

	conv_lossh�>���        )��P	�����A�*

	conv_loss�=�=����        )��P	�O����A�*

	conv_loss��=�e�B        )��P	������A�*

	conv_loss�'�=���Z        )��P	V�����A�*

	conv_loss���=_\�        )��P	������A�*

	conv_loss>��=����        )��P	�0����A�*

	conv_loss�>��1�        )��P	�c����A�*

	conv_loss�>�=d�V        )��P	۔����A�*

	conv_loss�"�=�8��        )��P	������A�*

	conv_loss%��=�Y��        )��P	\�����A�*

	conv_loss��=�ve�        )��P	�,����A�*

	conv_loss]��=���        )��P	�]����A�*

	conv_loss��=^��        )��P	������A�*

	conv_lossm��=f���        )��P	9�����A�*

	conv_loss���=��>        )��P	������A�*

	conv_lossu�>�I�R        )��P	�0����A�*

	conv_loss�D�=ǣV�        )��P	e����A�*

	conv_loss�{�=��s0        )��P	ҕ����A�*

	conv_loss��=N�H�        )��P	������A�*

	conv_loss���=*@�@        )��P	������A�*

	conv_loss���=��~        )��P	�(����A�*

	conv_lossh��=9}.        )��P	�X����A�*

	conv_loss&+�=G=ŵ        )��P	�����A�*

	conv_loss}=> /�        )��P	7�����A�*

	conv_loss�4�=��"        )��P	������A�*

	conv_loss��=}%�6        )��P	0(����A�*

	conv_loss#��=���        )��P	�X����A�*

	conv_lossk�=5#��        )��P	������A�*

	conv_loss^��=����        )��P	3�����A�*

	conv_loss��=cQ�$        )��P	������A�*

	conv_loss���=���?        )��P	7.����A�*

	conv_lossA�>�ij        )��P	5^����A�*

	conv_loss���=҆�        )��P	ߎ����A�*

	conv_loss��=P_�        )��P	������A�*

	conv_loss��=�=�        )��P	8�����A�*

	conv_loss���=	{6        )��P	�"����A�*

	conv_loss:��=����        )��P	4S����A�*

	conv_loss|q�=/�U�        )��P	s�����A�*

	conv_loss��=�!C        )��P	j�����A�*

	conv_loss���=Q�֣        )��P	W�����A�*

	conv_lossa��=�S�s        )��P	L����A�*

	conv_lossp��=Eq[T        )��P	CI����A�*

	conv_loss�z�=��?�        )��P	I|����A�*

	conv_lossx�=P���        )��P	�����A�*

	conv_loss	��=C[!Q        )��P	�����A�*

	conv_loss��=�
�        )��P	f����A�*

	conv_lossR��=�׋        )��P	������A�*

	conv_loss8A�=��        )��P	v�����A�*

	conv_loss@,>�ఊ        )��P	�"����A�*

	conv_loss���=����        )��P	U����A�*

	conv_loss��=ʡ�1        )��P	������A�*

	conv_loss#��=�))�        )��P	`�����A�*

	conv_loss��>3��@        )��P	�����A�*

	conv_loss��=�7P        )��P	�'����A�*

	conv_loss�>���k        )��P	�Z����A�*

	conv_lossp��=���        )��P	������A�*

	conv_lossȵ>ɮv{        )��P	������A�*

	conv_loss���=���b        )��P	�����A�*

	conv_loss�s�=	A�J        )��P	#<����A�*

	conv_loss�~�=����        )��P	>p����A�*

	conv_loss��	>�0,         )��P	�����A�*

	conv_loss��=��        )��P	e�����A�*

	conv_loss�0�=Ǘ��        )��P	�����A�*

	conv_lossJ�>��6�        )��P	�?����A�*

	conv_lossܺ>�e�        )��P	�p����A�*

	conv_lossp�=�2�        )��P	/�����A�*

	conv_loss��=1��        )��P	������A�*

	conv_lossD�=B�jU        )��P	����A�*

	conv_loss���=GO~        )��P	�K����A�*

	conv_lossπ�=In6�        )��P	�����A�*

	conv_loss��=E��        )��P	������A�*

	conv_loss��=����        )��P	1�����A�*

	conv_loss+`>�`��        )��P	/����A�*

	conv_loss�k�=��        )��P	MN����A�*

	conv_loss�=LҶ        )��P	�����A�*

	conv_loss���=�g�        )��P	W�����A�*

	conv_loss|��=`�@        )��P	]�����A�*

	conv_loss��=׼R�        )��P	j/����A�*

	conv_loss���=��(        )��P	�_����A�*

	conv_loss�R�=�b�C        )��P	������A�*

	conv_loss���=6qAS        )��P	5�����A�*

	conv_loss7�>ǲzq        )��P	|�����A�*

	conv_loss�3�=����        )��P	�>����A�*

	conv_loss�6>T	        )��P	hq����A�*

	conv_loss�@�=)�.        )��P	ѡ����A�*

	conv_loss��=S�z        )��P	5�����A�*

	conv_loss�	>�=�        )��P	�����A�*

	conv_lossS�>^��        )��P	>>����A�*

	conv_loss4�=l߭�        )��P	�p����A�*

	conv_loss��==�        )��P	4�����A�*

	conv_lossb��=�<�>        )��P	�����A�*

	conv_loss_ >��G�        )��P	E����A�*

	conv_loss7�=2���        )��P	�8����A�*

	conv_loss}q�=�dH        )��P	mh����A�*

	conv_loss;��=ͮ�/        )��P	������A�*

	conv_lossE��=�~        )��P	#�����A�*

	conv_loss�J�=��n        )��P	�����A�*

	conv_loss/&�=loz        )��P	�@����A�*

	conv_loss�>�=���        )��P	�p����A�*

	conv_loss�2�=��t�        )��P	������A�*

	conv_loss�E�=���        )��P	6�����A�*

	conv_loss���=9J�y        )��P	�����A�*

	conv_lossp �=uj-�        )��P	�K����A�*

	conv_loss��=��D        )��P	Ą����A�*

	conv_loss��=�)�e        )��P	������A�*

	conv_losskB�=b        )��P	������A�*

	conv_loss���=�l�        )��P	3%����A�*

	conv_loss`]�=�*�q        )��P	�^����A�*

	conv_loss.[ >C��d        )��P	w�����A�*

	conv_lossQ<�=�� s        )��P	������A�*

	conv_lossS�>~���        )��P	������A�*

	conv_lossaQ>��D
        )��P	|(����A�*

	conv_loss���=z��        )��P	�X����A�*

	conv_loss�j>ѷq�        )��P	������A�*

	conv_loss���=#Vo�        )��P	������A�*

	conv_loss=e�=�rH        )��P	>�����A�*

	conv_lossm��=��ޢ        )��P	�����A�*

	conv_loss��=-��]        )��P	�O����A�*

	conv_loss|��=�/n        )��P	������A�*

	conv_loss`��=в�        )��P	W�����A�*

	conv_lossl�=lp�x        )��P	������A�*

	conv_loss�=��=e        )��P	=����A�*

	conv_lossY\�=X�R        )��P	�E����A�*

	conv_lossC	�= 0�d        )��P	�w����A�*

	conv_loss���=_�        )��P	d�����A�*

	conv_loss��=r��        )��P	������A�*

	conv_loss7'�=i�&        )��P	!����A�*

	conv_loss'�=�!�w        )��P	�G����A�*

	conv_losss�="�p�        )��P	Ax����A�*

	conv_lossY�>YV��        )��P	Ѫ����A�*

	conv_loss��=�CKy        )��P	������A�*

	conv_lossI��=:�r�        )��P	�����A�*

	conv_loss=��=-p�        )��P	�<����A�*

	conv_lossY�=�jM        )��P	�o����A�*

	conv_loss���=pb        )��P	������A�*

	conv_loss� >��j        )��P	������A�*

	conv_loss)�=v�lL        )��P	�����A�*

	conv_loss���=m]')        )��P	5����A�*

	conv_loss�n�=K�#R        )��P	�d����A�*

	conv_loss��=���        )��P	�����A�*

	conv_loss��=��        )��P	j�����A�*

	conv_loss��>A�g�        )��P	������A�*

	conv_loss˱>��d        )��P	F1����A�*

	conv_loss�#�=��        )��P	�a����A�*

	conv_lossՃ�=��KJ        )��P	������A�*

	conv_loss��=�+�0        )��P	������A�*

	conv_loss}��=�6�M        )��P	����A�*

	conv_loss0��=���M        )��P	7����A�*

	conv_loss�S�=�Mo�        )��P	�g����A�*

	conv_loss0��=Z�#`        )��P	(�����A�*

	conv_loss�I >.�g�        )��P	������A�*

	conv_loss��=�pJ�        )��P	�����A�*

	conv_loss��>
��L        )��P	�A����A�*

	conv_loss � >u�a�        )��P	�s����A�*

	conv_lossg5�=ֹ        )��P	������A�*

	conv_loss�J�=��        )��P	F�����A�*

	conv_loss�x�=��@<        )��P	�����A�*

	conv_loss���=�>�P        )��P		Q����A�*

	conv_lossG��=��ׅ        )��P	@�����A�*

	conv_loss�6�=���        )��P	������A�*

	conv_lossf~�= �S        )��P	������A�*

	conv_loss���=�GQ�        )��P	�'����A�*

	conv_loss��=V�F        )��P	�X����A�*

	conv_lossY��=�5�'        )��P	5�����A�*

	conv_loss5�>�z�        )��P	������A�*

	conv_loss���=�O��        )��P	A�����A�*

	conv_loss���=��^d        )��P	&����A�*

	conv_loss��=\��        )��P	=L����A�*

	conv_loss���=	U��        )��P	a|����A�*

	conv_loss��=��.        )��P	�����A�*

	conv_loss"��=󷦑        )��P	������A�*

	conv_loss7)�=?��        )��P	�����A�*

	conv_loss�=ŉ��        )��P	?����A�*

	conv_lossF1�=�9        )��P	Vo����A�*

	conv_lossv>�        )��P	~�����A�*

	conv_loss�9�=�1w�        )��P	:�����A�*

	conv_loss	��=��        )��P	�����A�*

	conv_loss�|�=PF;_        )��P	�2����A�*

	conv_loss��=�dDD        )��P	�f����A�*

	conv_lossKE�=���        )��P	�����A�*

	conv_lossAe�=��JJ        )��P	,�����A�*

	conv_lossK�=�XI�        )��P	������A�*

	conv_lossJ��=Qݬ        )��P	�-����A�*

	conv_loss
C�=�NU}        )��P	#]����A�*

	conv_loss㵶=OY}        )��P	�����A�*

	conv_loss#>ʋ�j        )��P	������A�*

	conv_lossj��=0q��        )��P	������A�*

	conv_loss ��=v��]        )��P	#����A�*

	conv_losse�=�	        )��P	 T����A�*

	conv_lossv�=���        )��P	k�����A�*

	conv_loss8G�=#��N        )��P	������A�*

	conv_loss��=��?�        )��P	^�����A�*

	conv_lossR�=��        )��P	�����A�*

	conv_lossS��=�j��        )��P	�M����A�*

	conv_loss# >�G7]        )��P	����A�*

	conv_lossD�=oGV        )��P	�����A�*

	conv_loss.d >�x�        )��P	'����A�*

	conv_loss�:
>���        )��P	�=����A�*

	conv_loss���=�Nn=        )��P	�n����A�*

	conv_loss��=l*l
        )��P	=�����A�*

	conv_lossj�=�r�m        )��P	������A�*

	conv_loss�y�=�w         )��P	c����A�*

	conv_loss�J�=���m        )��P	�B����A�*

	conv_lossS}�=�8�7        )��P	�t����A�*

	conv_loss=W�=}e�        )��P	l�����A�*

	conv_loss���=�B5        )��P	e�����A�*

	conv_loss�K�=;54g        )��P	J����A�*

	conv_losse)>���        )��P	4M����A�*

	conv_loss�b�=Nf��        )��P	�����A�*

	conv_loss	V�=4{��        )��P	}�����A�*

	conv_loss�#�=�MPn        )��P	������A�*

	conv_lossi1�=	N        )��P	����A�*

	conv_loss˴�=���3        )��P	*L����A�*

	conv_loss���=�'�        )��P	�����A�*

	conv_lossFɡ=rzځ        )��P	D�����A�*

	conv_loss�f�=���Q        )��P	�����A�*

	conv_loss�->p�t	        )��P	�����A�*

	conv_loss��=f�=        )��P	�J����A�*

	conv_loss@N�=3#�I        )��P	�}����A�*

	conv_loss(��=�N��        )��P	9�����A�*

	conv_loss���=���l        )��P	������A�*

	conv_loss4��=���        )��P	�����A�*

	conv_loss%@�=.��H        )��P	B����A�*

	conv_loss��=y�4�        )��P	�s����A�*

	conv_loss)��=���        )��P	������A�*

	conv_loss���=9��W        )��P	������A�*

	conv_loss���=I�q�        )��P	;����A�*

	conv_loss�H�=��        )��P	OF����A�*

	conv_loss���=��Y        )��P	Mx����A�*

	conv_loss���=��}        )��P	 �����A�*

	conv_loss2�=̅[        )��P	������A�*

	conv_lossTM�=�~�~        )��P	�����A�*

	conv_loss)�=
p�`        )��P	�?����A�*

	conv_loss� �=��^7        )��P	fy����A�*

	conv_loss��=D�ow        )��P	m�����A�*

	conv_lossٲ�=�3        )��P	�����A�*

	conv_lossd�>&t\        )��P	3����A�*

	conv_loss���=>9��        )��P	2I����A�*

	conv_loss��=W7i�        )��P	3{����A�*

	conv_losssU�=�Ηg        )��P	ݭ����A�*

	conv_lossnʚ=lÔ�        )��P	������A�*

	conv_lossE"�=�[/        )��P	����A�*

	conv_loss�>�Xf        )��P	SF����A�*

	conv_loss"`�=(�,5        )��P	�x����A�*

	conv_loss���=�k��        )��P	"�����A�*

	conv_loss��=f�)�        )��P	�����A�*

	conv_loss/��=)	�        )��P	�����A�*

	conv_loss�`�=��%        )��P	������A�*

	conv_loss=�=2�%�        )��P	Y�����A�*

	conv_loss��>�ݥ        )��P	�����A�*

	conv_lossޡ�=X�{        )��P	!Y����A�*

	conv_loss�w�=gM��        )��P	������A�*

	conv_losslh�=��        )��P	������A�*

	conv_lossѽ�=}���        )��P	������A�*

	conv_loss��='��        )��P	 2����A�*

	conv_loss��=%%x�        )��P	uj����A�*

	conv_lossX��=�8��        )��P	������A�*

	conv_loss���=l�e{        )��P	�����A�*

	conv_loss�o�=²        )��P	�����A�*

	conv_loss��=U�-�        )��P	�4����A�*

	conv_loss2�>��Ȉ        )��P	�j����A�*

	conv_loss.��=1��R        )��P	b�����A�*

	conv_loss��=E��        )��P	k�����A�*

	conv_loss�>>Ŏ�w        )��P	�����A�*

	conv_loss3��=��        )��P	FD����A�*

	conv_loss�f >��T2        )��P	�}����A�*

	conv_loss���=���        )��P	������A�*

	conv_loss��=7���        )��P	������A�*

	conv_loss���=���        )��P	����A�*

	conv_loss���=�yV        )��P	�I����A�*

	conv_loss	��=��U        )��P	�~����A�*

	conv_loss���=�        )��P	в����A�*

	conv_loss���=�Q�        )��P	������A�*

	conv_loss��=�0?|        )��P	�����A�*

	conv_lossE�=U��o        )��P	[H����A�*

	conv_lossP$�=�f�        )��P	�����A�*

	conv_loss"�=͍er        )��P	~�����A�*

	conv_loss���={�+�        )��P	I�����A�*

	conv_loss-x�=C��;        )��P	�(����A�*

	conv_loss���=�^�        )��P	9Z����A�*

	conv_lossNF�=�A�S        )��P	������A�*

	conv_loss{{�=0�U�        )��P	�����A�*

	conv_lossx �= ~�e        )��P	������A�*

	conv_loss�e�=��&        )��P	/����A�*

	conv_loss�5�=��%�        )��P	�`����A�*

	conv_loss��=�'�9        )��P	B�����A�*

	conv_loss&H>8��4        )��P	������A�*

	conv_lossR��=g`�        )��P	�����A�*

	conv_loss`A�=�31X        )��P	�C����A�*

	conv_loss	��=l�TY        )��P	�v����A�*

	conv_loss���=���>        )��P	:�����A�*

	conv_loss��=[��        )��P	K�����A�*

	conv_loss��=�8V        )��P	� ���A�*

	conv_loss��=��é        )��P	LF ���A�*

	conv_lossK��=o,K        )��P	}w ���A�*

	conv_loss{$�=9��        )��P	x� ���A�*

	conv_loss�l�=o�        )��P	�� ���A�*

	conv_loss���=�Ȉ        )��P	53���A�*

	conv_loss�;>K��j        )��P	�i���A�*

	conv_loss>m �        )��P	Ħ���A�*

	conv_lossA��=�#�y        )��P	�����A�*

	conv_loss��=���        )��P	����A�*

	conv_loss��=��_        )��P	�A���A�*

	conv_loss���=�+z        )��P	;s���A�*

	conv_losse��=��,        )��P	����A�*

	conv_loss��=�5�        )��P	@����A�*

	conv_losss&�=�BP�        )��P	����A�*

	conv_loss>��b�        )��P	PH���A�*

	conv_loss�o�=rt4�        )��P	ry���A�*

	conv_loss�ڽ=_���        )��P	�����A�*

	conv_loss�T�=H���        )��P	�����A�*

	conv_loss��=�Ä�        )��P	����A�*

	conv_loss�J�=��L�        )��P	�R���A�*

	conv_lossF��=��        )��P	�����A�*

	conv_loss���=s��'        )��P	3����A�*

	conv_lossW	�=�0�0        )��P	�����A�*

	conv_lossވ�=!��        )��P	����A�*

	conv_loss���=��p�        )��P	=R���A�*

	conv_lossc��=41��        )��P	����A�*

	conv_loss"p�=�M        )��P	k����A�*

	conv_loss盼=��Bl        )��P	�����A�*

	conv_loss��=P`��        )��P	���A�*

	conv_loss@�=���        )��P	�M���A�*

	conv_loss�K�='�        )��P	���A�*

	conv_lossM>�ĺR        )��P	8����A�*

	conv_loss��=,p��        )��P	4����A�*

	conv_loss?��=,�cU        )��P	p���A�*

	conv_lossc��=��X        )��P	]L���A�*

	conv_lossz�>��        )��P	�|���A�*

	conv_loss���=��Q/        )��P	�����A�*

	conv_lossxb�=�z4�        )��P	�����A�*

	conv_loss���=h��        )��P	����A�*

	conv_loss0�=�	7�        )��P	nD���A�*

	conv_lossk��=�B�        )��P	>v���A�*

	conv_lossѹ�=�d�        )��P	"����A�*

	conv_lossx�=�L��        )��P	����A�*

	conv_loss�a�= :��        )��P	�	���A�*

	conv_loss�_�= �!*        )��P	�>	���A�*

	conv_loss�&�=o�@u        )��P	|p	���A�*

	conv_loss���=��b+        )��P	��	���A�*

	conv_lossZŧ=�&&        )��P	��	���A�*

	conv_lossj��=��        )��P	'

���A�*

	conv_loss���=�j�J        )��P	�:
���A�*

	conv_loss$	�=jH�/        )��P	ml
���A�*

	conv_loss���=69�m        )��P	�
���A�*

	conv_loss���=��        )��P	V�
���A�*

	conv_lossY�=��e        )��P	����A�*

	conv_loss�Y�=�`�k        )��P	�H���A�*

	conv_losstL�=,�8        )��P	�z���A�*

	conv_loss�C>1!�P        )��P	����A�*

	conv_loss�|�=H��$        )��P	�����A�*

	conv_losso��=J�\        )��P	m%���A�*

	conv_loss�ɴ="T1        )��P	V���A�*

	conv_loss<��=��*s        )��P	�����A�*

	conv_loss��=�        )��P	i����A�*

	conv_loss2��={�c        )��P	�����A�*

	conv_lossA��=���5        )��P	x/���A�*

	conv_lossM��=�%�        )��P	�b���A�*

	conv_loss��=/���        )��P	�����A�*

	conv_lossrO�=gi��        )��P	G����A�*

	conv_loss�h�=�Qm�        )��P	
����A�*

	conv_lossX��=2�&�        )��P	�1���A�*

	conv_lossY��=��K        )��P	{j���A�*

	conv_loss?_�=np        )��P	�����A�*

	conv_loss���=�Z+        )��P	�����A�*

	conv_lossX��=��v�        )��P	����A�*

	conv_loss��=O��W        )��P	%4���A�*

	conv_loss��=oUn        )��P	�f���A�*

	conv_lossl�=�:��        )��P	�����A�*

	conv_lossJu�=����        )��P	�����A�*

	conv_loss� >ȋ�j        )��P	�����A�*

	conv_loss�&�=lVXK        )��P	�.���A�*

	conv_loss{0�=b �        )��P	m`���A�*

	conv_loss��=�&�5        )��P	w����A�*

	conv_loss���=v��W        )��P	^����A�*

	conv_lossAZ�=�e��        )��P	�����A�*

	conv_loss�1�=X��N        )��P	O-���A�*

	conv_loss��=kI��        )��P	a���A�*

	conv_loss�:�=j5�        )��P	����A�*

	conv_loss~��='�ϖ        )��P	�����A�*

	conv_loss�	�=�V6        )��P	,����A�*

	conv_loss���=W��d        )��P	�,���A�*

	conv_loss/�=��f�        )��P	�_���A�*

	conv_loss�q�=�س        )��P	d����A�*

	conv_lossۛ=��"�        )��P	�����A�*

	conv_loss�B�=�v�(        )��P	�����A�*

	conv_lossC#�=�        )��P	i%���A�*

	conv_loss9�=y�        )��P	�V���A�*

	conv_loss��=��1�        )��P	�����A�*

	conv_loss� �=X�w�        )��P	}����A�*

	conv_loss<L�=�ܙ�        )��P	�����A�*

	conv_loss�~�=��V�        )��P	�#���A�*

	conv_loss͐�=p���        )��P	�T���A�*

	conv_lossU{�=�M�c        )��P	����A�*

	conv_loss�:�=�]t        )��P	\����A�*

	conv_loss/-�=��p        )��P	�����A�*

	conv_loss���=��ʞ        )��P	�/���A�*

	conv_loss���=�N��        )��P	�`���A�*

	conv_loss[z�=u҇�        )��P	�����A�*

	conv_lossd�="�T        )��P	����A�*

	conv_loss�Q�=/K�        )��P	M����A�*

	conv_lossXh�=VΧ�        )��P	�:���A�*

	conv_loss@�=ͼ�        )��P	(o���A�*

	conv_loss�I�=�x2�        )��P	�����A�*

	conv_loss1J�=�Dn�        )��P	L����A�*

	conv_loss�=oe��        )��P	
���A�*

	conv_loss���=��E        )��P	QD���A�*

	conv_loss>�=Q�,#        )��P	v���A�*

	conv_loss���=�p��        )��P	ݧ���A�*

	conv_losss��=�0^        )��P	����A�*

	conv_loss���=�d�/        )��P	���A�*

	conv_loss�H�=MvtG        )��P	1W���A�*

	conv_loss���=3 ��        )��P	�����A�*

	conv_loss�#�=�.�        )��P	3����A�*

	conv_loss=��=x|        )��P	 ���A�*

	conv_loss���=��:        )��P	�9���A�*

	conv_lossEڹ=��=�        )��P	�m���A�*

	conv_loss���=���        )��P	0����A�*

	conv_loss�N�=L�.0        )��P	a����A�*

	conv_loss���=w4%        )��P	Z���A�*

	conv_loss\��=f���        )��P	42���A�*

	conv_loss���=8R6�        )��P	Uj���A�*

	conv_loss���=��)        )��P	Ϣ���A�*

	conv_loss�C><��3        )��P	'����A�*

	conv_loss��=�ꠝ        )��P	p���A�*

	conv_loss�y�=�45        )��P	J���A�*

	conv_loss�<�=��c        )��P	Gz���A�*

	conv_loss�v�=PВ�        )��P	����A�*

	conv_loss��=B�_�        )��P	|����A�*

	conv_loss��=����        )��P	Y���A�*

	conv_lossq�='�L        )��P	$F���A�*

	conv_loss�D�=ps�k        )��P	�y���A�*

	conv_loss�U�=Zfx        )��P	�����A�*

	conv_lossC�=�.�j        )��P	m����A�*

	conv_loss��=z�Y�        )��P	���A�*

	conv_loss@o�=�Kex        )��P	�@���A�*

	conv_loss�7> �:�        )��P	kr���A�*

	conv_loss�r�=�         )��P	!����A�*

	conv_loss��=��s�        )��P	�����A�*

	conv_loss�=a	6        )��P	����A�*

	conv_loss�>�=���<        )��P	�6���A�*

	conv_loss�k�=�V�        )��P	�i���A�*

	conv_loss$��=���        )��P	b����A�*

	conv_loss���=���        )��P	�����A�*

	conv_loss�r�=�+�,        )��P	����A�*

	conv_lossJ>�=P�K�        )��P	��#���A�*

	conv_loss�P�=�=��        )��P	�'%���A�*

	conv_loss��=��
D        )��P	0Y%���A�*

	conv_loss�;�=��d�        )��P	��%���A�*

	conv_loss?(�=l�WE        )��P		�%���A�*

	conv_losse��=۴uK        )��P	��%���A�*

	conv_loss^Y�=G�D�        )��P	`!&���A�*

	conv_loss���=���"        )��P	�Q&���A�*

	conv_loss���=��t�        )��P	�&���A�*

	conv_loss�;�=;�?        )��P		�&���A�*

	conv_loss���=LMW~        )��P	>�&���A�*

	conv_loss5,>�)#D        )��P	-'���A�*

	conv_lossN�>�̕         )��P	�g'���A�*

	conv_lossW,>��U�        )��P	��'���A�*

	conv_loss�@>f)�        )��P	��'���A�*

	conv_loss��=���c        )��P	q(���A�*

	conv_loss��=�pO        )��P	�3(���A�*

	conv_loss	��=���a        )��P	<b(���A�*

	conv_loss�e�=.�        )��P	��(���A�*

	conv_loss��=ػ�        )��P	��(���A�*

	conv_loss��=ӌWj        )��P	b�(���A�*

	conv_loss}��=��#        )��P	K)���A�*

	conv_loss���=*��b        )��P	�M)���A�*

	conv_loss��=h��:        )��P	 �)���A�*

	conv_loss���=y�        )��P	5�)���A�*

	conv_loss�c�=��|�        )��P	��)���A�*

	conv_loss�<�=��0�        )��P	�*���A�*

	conv_loss���=�S�"        )��P	1>*���A�*

	conv_loss��==;�        )��P	�r*���A�*

	conv_lossd��=�<��        )��P	{�*���A�*

	conv_loss�u�=C��        )��P	��*���A�*

	conv_loss=Z�=�Ĕ�        )��P	�+���A�*

	conv_loss���=_���        )��P	�;+���A�*

	conv_loss2a�=�`�        )��P	Ai+���A�*

	conv_lossv,�=wV˭        )��P	Ɯ+���A�*

	conv_lossT~�=|q��        )��P	c�+���A�*

	conv_loss�J�='H��        )��P	��+���A�*

	conv_loss���=�?�A        )��P	�),���A�*

	conv_loss+��=��c        )��P	(^,���A�*

	conv_loss���=��CI        )��P	�,���A�*

	conv_loss�5�=����        )��P	/�,���A�*

	conv_loss���=��V\        )��P	��,���A�*

	conv_loss��=���
        )��P	� -���A�*

	conv_loss���=y�M]        )��P	�S-���A�*

	conv_loss��=�{��        )��P	�-���A�*

	conv_lossTa�=P��        )��P	��-���A�*

	conv_lossB��=���        )��P	��-���A�*

	conv_loss-��=���        )��P	�$.���A�*

	conv_loss5X�= �N        )��P	CT.���A�*

	conv_loss섿=�9��        )��P	Ʉ.���A�*

	conv_loss�ʮ=�/��        )��P	��.���A�*

	conv_lossJn�=�6�d        )��P	l	/���A�*

	conv_loss�2�=x�֙        )��P	�;/���A�*

	conv_lossS�=�OE�        )��P	"j/���A�*

	conv_loss��=��F�        )��P	՜/���A�*

	conv_loss��=���        )��P	��/���A�*

	conv_lossa�=e~3        )��P	) 0���A�*

	conv_lossGk�=Y.�U        )��P	�.0���A�*

	conv_loss5�=q�<
        )��P	�\0���A�*

	conv_loss>��=,E��        )��P	ϔ0���A�*

	conv_loss���=�C��        )��P	��0���A�*

	conv_loss?��=�=k�        )��P	�1���A�*

	conv_loss�7�=>ݓ        )��P	71���A�*

	conv_loss���=Ը�        )��P	�j1���A�*

	conv_loss�	�=d'�        )��P	-�1���A�*

	conv_loss�A�=Ny�        )��P	�1���A�*

	conv_loss��=��m�        )��P	�2���A�*

	conv_loss��=m2        )��P	M52���A�*

	conv_loss��=��2�        )��P	�f2���A�*

	conv_loss�=i	P�        )��P	a�2���A�*

	conv_lossM��=.��J        )��P	w�2���A�*

	conv_loss�.�=d�W�        )��P	��2���A�*

	conv_lossAr�=I���        )��P	P%3���A�*

	conv_lossP��=_]/|        )��P	"W3���A�*

	conv_loss��=���        )��P	m�3���A�*

	conv_loss��=��&H        )��P	�3���A�*

	conv_losssl�=>��        )��P	��3���A�*

	conv_loss��=�g��        )��P	B4���A�*

	conv_loss'>7��X        )��P	D4���A�*

	conv_loss�}>�1�5        )��P	�v4���A�*

	conv_loss��=��G/        )��P	?�4���A�*

	conv_lossH��=�:`^        )��P	�4���A�*

	conv_loss|��=��|�        )��P	y5���A�*

	conv_loss���=T3��        )��P	�C5���A�*

	conv_loss�/�=4���        )��P	�r5���A�*

	conv_loss;L�=E��3        )��P	��5���A�*

	conv_loss���=��,        )��P	O�5���A�*

	conv_lossh�=�Ǝ�        )��P	��5���A�*

	conv_loss�y�=7��
        )��P	�66���A�*

	conv_loss�D�=�&        )��P	�l6���A�*

	conv_loss���=<��U        )��P	c�6���A�*

	conv_loss{��=4/G�        )��P	:�6���A�*

	conv_loss�y�=(x        )��P	/7���A�*

	conv_loss�?�=���        )��P	Q77���A�*

	conv_lossaP�=F8G�        )��P	�d7���A�*

	conv_loss���=���        )��P	B�7���A�*

	conv_loss�ۯ=�s?�        )��P	�7���A�*

	conv_loss:�=^�R�        )��P	'�7���A�*

	conv_loss�y�=��4        )��P	�%8���A�*

	conv_loss5��=:nڍ        )��P	�T8���A�*

	conv_loss���=��!�        )��P	h�8���A�*

	conv_loss�j�=�k
�        )��P	�8���A�*

	conv_lossCe�=
KP        )��P	�8���A�*

	conv_loss�v�=&)K        )��P	�$9���A�*

	conv_loss^��=�oBh        )��P	�R9���A�*

	conv_lossx��=A�        )��P	!�9���A�*

	conv_loss=�=�sК        )��P	�9���A�*

	conv_loss���=��        )��P	4�9���A�*

	conv_loss��=�Z�        )��P	A:���A�*

	conv_losst!�=�x(        )��P	4M:���A�*

	conv_lossZT�=G��        )��P	C~:���A�*

	conv_lossl��=a}H�        )��P	��:���A�*

	conv_lossW�=B�mx        )��P	~�:���A�*

	conv_loss��=7��        )��P	�;���A�*

	conv_loss.]�=6���        )��P	�K;���A�*

	conv_loss���=TO        )��P	�~;���A�*

	conv_loss> �=J<�P        )��P	{�;���A�*

	conv_lossS�>���F        )��P	��;���A�*

	conv_loss�~�=�$6X        )��P	<���A�*

	conv_lossKq�=��*        )��P	�K<���A�*

	conv_loss�y�=t_@        )��P	`z<���A�*

	conv_loss%�=n$��        )��P	n�<���A�*

	conv_loss�M�= �K        )��P	,�<���A�*

	conv_loss��=��8        )��P	�	=���A�*

	conv_loss.��=�)�]        )��P	�:=���A�*

	conv_lossQ��=��l        )��P	*k=���A�*

	conv_lossEQ�=�?v        )��P	4�=���A�*

	conv_lossX��=Ъ�4        )��P	��=���A�*

	conv_lossr�=_fK        )��P	��=���A�*

	conv_loss�_�=O�j        )��P	�'>���A�*

	conv_losso��=X}@j        )��P	�W>���A�*

	conv_loss1�=���        )��P	��>���A�*

	conv_loss�-�=/�l        )��P	��>���A�*

	conv_lossbQ�=�g��        )��P	�>���A�*

	conv_loss�͝=�_�        )��P	�?���A�*

	conv_loss��=�J�y        )��P	�E?���A�*

	conv_loss��=F��r        )��P	�t?���A�*

	conv_loss
��=~�Q�        )��P	ݥ?���A�*

	conv_loss���=�s�        )��P	��?���A�*

	conv_loss:��=�'�`        )��P	�@���A�*

	conv_loss{x�= rG        )��P	.2@���A�*

	conv_loss�5�=�߄�        )��P	a@���A�*

	conv_loss���=3\;�        )��P	2�@���A�*

	conv_losse�=���        )��P	��@���A�*

	conv_lossF��=2�0H        )��P	��@���A�*

	conv_loss��=4�z�        )��P	o A���A�*

	conv_lossb��=���5        )��P	'SA���A�*

	conv_lossw�=ذ��        )��P	�A���A�*

	conv_loss�,�=&6�L        )��P	�A���A�*

	conv_loss2��=��        )��P	��A���A�*

	conv_lossq�=�L�        )��P	�B���A�*

	conv_lossJ{�=���        )��P	SPB���A�*

	conv_lossk�=4	\�        )��P	�B���A�*

	conv_loss���=���I        )��P	��B���A�*

	conv_loss[�=㕧�        )��P	��B���A�*

	conv_loss}��=�3�P        )��P	�C���A�*

	conv_loss�κ=�*j�        )��P	�EC���A�*

	conv_loss���=��j�        )��P	�zC���A�*

	conv_loss���=��=        )��P	n�C���A�*

	conv_loss��=��C        )��P	��C���A�*

	conv_loss�"�=�[x�        )��P	�	D���A�*

	conv_loss`��=a��o        )��P	9D���A�*

	conv_loss7�=CR�        )��P	(yD���A�*

	conv_loss>��=�W�W        )��P	Z�D���A�*

	conv_loss%�=��a        )��P	q�D���A�*

	conv_loss=��=�#XX        )��P	�E���A�*

	conv_loss���=2�        )��P	W;E���A�*

	conv_loss˖=?+�        )��P	�nE���A�*

	conv_loss��=[�J        )��P	{�E���A�*

	conv_loss�,�=��l�        )��P	h�E���A�*

	conv_lossI4�=��m�        )��P	0F���A�*

	conv_loss�f�=�U,N        )��P	NF���A�*

	conv_loss��=�^�        )��P	�F���A�*

	conv_loss�#�=���        )��P	u�F���A�*

	conv_loss���=���        )��P	�F���A�*

	conv_loss�=���<        )��P	�,G���A�*

	conv_loss$�=�@        )��P	�]G���A�*

	conv_lossN�=��W�        )��P	Q�G���A�*

	conv_lossN�=y�s�        )��P	��G���A�*

	conv_lossbz�=^�        )��P	��G���A�*

	conv_lossȯ�=O�A�        )��P	�'H���A�*

	conv_loss�!�=M�&s        )��P	�^H���A�*

	conv_loss]��=�n�O        )��P	�H���A�*

	conv_loss3�=L=-�        )��P	<�H���A�*

	conv_loss@��=��bw        )��P	��H���A�*

	conv_lossE��=��        )��P	:#I���A�*

	conv_loss���=���        )��P	�SI���A�*

	conv_loss���=��q        )��P	ȂI���A�*

	conv_loss!��=V���        )��P	��I���A�*

	conv_loss,Ҿ= U�        )��P	h�I���A�*

	conv_loss���=U�t�        )��P	/J���A�*

	conv_lossK��=��7E        )��P	�BJ���A�*

	conv_loss`3�=F�25        )��P	>tJ���A�*

	conv_loss =�=��6�        )��P	��J���A�*

	conv_loss�6�=e�J�        )��P	=�J���A�*

	conv_loss\v�=mc�/        )��P	IK���A�*

	conv_loss���=S���        )��P	�7K���A�*

	conv_loss��=@�        )��P	�iK���A�*

	conv_loss��|=զ��        )��P	��K���A�*

	conv_losshן=:�~�        )��P	f�K���A�*

	conv_loss!��=��G        )��P	_�K���A�*

	conv_loss�>�=�
U�        )��P	ҍM���A�*

	conv_lossz)�=;צm        )��P	�M���A�*

	conv_losso�=���S        )��P	�M���A�*

	conv_loss�=9���        )��P	B&N���A�*

	conv_losse��=��K�        )��P	�XN���A�*

	conv_loss���=T(��        )��P	��N���A�*

	conv_loss'|�='+,�        )��P	H�N���A�*

	conv_loss��=ծ��        )��P	P�N���A�*

	conv_loss���=�F�q        )��P	�-O���A�*

	conv_loss ��=Y���        )��P	�`O���A�*

	conv_loss7Ͼ=�xy�        )��P	��O���A�*

	conv_loss4'�=U��)        )��P	��O���A�*

	conv_loss��=~�d        )��P	� P���A�*

	conv_loss��=����        )��P	w3P���A�*

	conv_loss�آ=��        )��P	*dP���A�*

	conv_loss�`�=N���        )��P	�P���A�*

	conv_losst��=�S~}        )��P	��P���A�*

	conv_loss��=�~��        )��P	w�P���A�*

	conv_loss��=����        )��P	�&Q���A�*

	conv_loss���=]p��        )��P	�VQ���A�*

	conv_loss�=3N�@        )��P	��Q���A�*

	conv_lossG��=bG��        )��P	��Q���A�*

	conv_loss֑�=(<�k        )��P	M�Q���A�*

	conv_loss�:�==x�        )��P	�R���A�*

	conv_lossW��=��d        )��P	�GR���A�*

	conv_lossK�=��d(        )��P	�yR���A�*

	conv_loss�`�=� �        )��P	
�R���A�*

	conv_loss��=���        )��P	��R���A�*

	conv_lossD�=�F        )��P	qS���A�*

	conv_loss���=����        )��P	o8S���A�*

	conv_loss���=X�a�        )��P	�gS���A�*

	conv_lossN��=�J        )��P	?�S���A�*

	conv_loss���=|�Ĕ        )��P	3�S���A�*

	conv_loss���=��[�        )��P	��S���A�*

	conv_lossK9�=Zn��        )��P	X(T���A�*

	conv_loss_�=#U�N        )��P	YT���A�*

	conv_loss�T�=%��        )��P	.�T���A�*

	conv_loss��=_�        )��P	�T���A�*

	conv_loss���=D5��        )��P	��T���A�*

	conv_lossb��=kP��        )��P	Y#U���A�*

	conv_lossr	�=�.�        )��P	_U���A�*

	conv_lossՔ�=E�<>        )��P	��U���A�*

	conv_loss���=_,        )��P	7�U���A�*

	conv_loss���='x�6        )��P	��U���A�*

	conv_lossޗ=��\�        )��P	2V���A�*

	conv_loss��=51�Q        )��P	1dV���A�*

	conv_loss���=�Q�        )��P	G�V���A�*

	conv_lossi��=�"y/        )��P	��V���A�*

	conv_loss0�=�e+        )��P	u�V���A�*

	conv_loss�ˠ=��        )��P	3W���A�*

	conv_loss_�=jI��        )��P	&�W���A�*

	conv_lossm�=2g        )��P	-�W���A�*

	conv_lossjΣ=�;�        )��P	��W���A�*

	conv_loss�v�=�3        )��P	PX���A�*

	conv_loss��=t��x        )��P	/NX���A�*

	conv_loss~V�=�ɳ~        )��P	)�X���A�*

	conv_loss�P�=m�        )��P	8�X���A�*

	conv_loss���=�xM�        )��P	��X���A�*

	conv_loss���=x-��        )��P	�(Y���A�*

	conv_loss��=_O�        )��P	o[Y���A�*

	conv_lossO��=g��        )��P	��Y���A�*

	conv_loss>R�=��g        )��P	��Y���A�*

	conv_loss>u�=Z���        )��P	��Y���A�*

	conv_losss��=~[t�        )��P	�0Z���A�*

	conv_loss'>�=�Ú        )��P	�bZ���A�*

	conv_loss��=��V        )��P	�Z���A�*

	conv_lossgA�=�71�        )��P	��Z���A�*

	conv_loss�L�=�a�)        )��P	��Z���A�*

	conv_loss�{�=wM��        )��P	�,[���A�*

	conv_loss+&�=oq�        )��P	�_[���A�*

	conv_lossz��=�<n�        )��P	A�[���A�*

	conv_loss3=�=ѝ�        )��P	o�[���A�*

	conv_loss�F�=p�vY        )��P	��[���A�*

	conv_loss��=ҳ��        )��P	�*\���A�*

	conv_loss.	�=�M�        )��P	w\\���A�*

	conv_loss4F�=�컟        )��P	#�\���A�*

	conv_lossT�=��*�        )��P	��\���A�*

	conv_loss&�=V>        )��P	��\���A�*

	conv_lossR��=����        )��P	#]���A�*

	conv_loss�=�7P�        )��P	=U]���A�*

	conv_loss���=쟅�        )��P	8�]���A�*

	conv_lossY��=���W        )��P	��]���A�*

	conv_loss$��=[�M$        )��P	�]���A�*

	conv_loss�J�=�b�        )��P	�^���A�*

	conv_loss�{�=��        )��P	I^���A�*

	conv_loss��=�Q\`        )��P	�z^���A�*

	conv_lossF�=_m�        )��P	*�^���A�*

	conv_loss��=r�1-        )��P	��^���A�*

	conv_loss1)�=�z�b        )��P	�_���A�*

	conv_loss���=Ձ��        )��P	�E_���A�*

	conv_loss|��=0��        )��P	�w_���A�*

	conv_loss��=���        )��P	ǩ_���A�*

	conv_lossr:�=LMe�        )��P	��_���A�*

	conv_lossF��=K��<        )��P	f`���A�*

	conv_loss��=08�*        )��P	>=`���A�*

	conv_loss�^�=���        )��P	�p`���A�*

	conv_loss�>�=skW�        )��P	�`���A�*

	conv_loss��=#�q�        )��P	��`���A�*

	conv_loss��=�{��        )��P		a���A�*

	conv_loss�y�=ez;3        )��P	o<a���A�*

	conv_loss�Y�=D.S        )��P	)�a���A�*

	conv_lossXЗ=����        )��P	o�a���A�*

	conv_lossMp�=��|P        )��P	��a���A�*

	conv_loss%b�=�vM�        )��P	�/b���A�*

	conv_loss��~=�/H�        )��P	ib���A�*

	conv_loss�_�=��        )��P	a�b���A�*

	conv_lossT�z=��]        )��P	��b���A�*

	conv_loss�>�=Щ,"        )��P	�c���A�*

	conv_loss\o�=�� �        )��P	�7c���A�*

	conv_loss^q�=Yu��        )��P	�nc���A�*

	conv_loss{��=���        )��P	U�c���A�*

	conv_lossq'�=�R�        )��P	��c���A�*

	conv_loss�`�=���I        )��P	�#d���A�*

	conv_lossu�=ƃ<I        )��P	�ed���A�*

	conv_loss���=O(��        )��P	s�d���A�*

	conv_loss�,�=g�        )��P	��d���A�*

	conv_loss H�=#�ot        )��P	�d���A�*

	conv_lossL�=hP^        )��P	�+e���A�*

	conv_loss�Ȑ=[z        )��P	g]e���A�*

	conv_loss�K�=>G,        )��P	��e���A�*

	conv_loss�R�=�B�O        )��P	%�e���A�*

	conv_loss67�=�X�        )��P	��e���A�*

	conv_lossU��=�S��        )��P	�'f���A�*

	conv_lossLݲ="�        )��P	D]f���A�*

	conv_lossv��=c��        )��P	G�f���A�*

	conv_lossf��=孙�        )��P	��f���A�*

	conv_loss���=�|�)        )��P	� g���A�*

	conv_loss�$�=u���        )��P	�4g���A�*

	conv_loss���=0/t�        )��P	rgg���A�*

	conv_lossD�=��B         )��P	��g���A�*

	conv_loss��=cl�        )��P	��g���A�*

	conv_loss|�=0���        )��P	�g���A�*

	conv_lossbU�=�K1�        )��P	�0h���A�*

	conv_loss�T�=���        )��P	�bh���A�*

	conv_lossy��=8���        )��P	��h���A�*

	conv_loss�!�=��v        )��P	�h���A�*

	conv_loss���=?��        )��P	W�h���A�*

	conv_lossΗ�=�?v�        )��P	�/i���A�*

	conv_loss2�=ۥ�g        )��P	�bi���A�*

	conv_loss,��=c�5        )��P	�i���A�*

	conv_loss�{�=^��        )��P	��i���A�*

	conv_loss�_�=�V�$        )��P	��i���A�*

	conv_loss�8�=H�-        )��P	�/j���A�*

	conv_loss���=�7#�        )��P	T`j���A�*

	conv_loss���=/�0        )��P	w�j���A�*

	conv_loss[��=,�/�        )��P	��j���A�*

	conv_loss��=���_        )��P	 �j���A�*

	conv_loss���=����        )��P	�+k���A�*

	conv_loss�Ҭ=�:��        )��P	`^k���A�*

	conv_loss�D�=���y        )��P	�k���A�*

	conv_loss+y�=��+D        )��P	c�k���A�*

	conv_loss���=���        )��P	�l���A�*

	conv_loss@�=���        )��P	�8l���A�*

	conv_loss��=�s�        )��P	okl���A�*

	conv_loss�)�=�]�        )��P	؞l���A�*

	conv_lossN��=OI3�        )��P	��l���A�*

	conv_lossw��=���        )��P	�m���A�*

	conv_losss	�=jL�w        )��P	�8m���A�*

	conv_loss�=��6        )��P	:qm���A�*

	conv_lossŶ�=�aA\        )��P	�m���A�*

	conv_lossZ¼=�C�        )��P	P�m���A�*

	conv_loss�G�=^�N        )��P	�n���A�*

	conv_loss�j�=���        )��P	�Wn���A�*

	conv_loss�a�=���p        )��P	*�n���A�*

	conv_loss��=�=        )��P	"�n���A�*

	conv_loss�j�=f���        )��P	��n���A�*

	conv_loss�҇=@��<        )��P	Ho���A�*

	conv_loss�j�=� ��        )��P	�Po���A�*

	conv_lossx�=�m��        )��P	/�o���A�*

	conv_lossW��=y�GZ        )��P	�o���A�*

	conv_loss���=����        )��P	:�o���A�*

	conv_lossNu�=F��        )��P	�p���A�*

	conv_loss72�=�� �        )��P	;Jp���A�*

	conv_loss�"�=xC�:        )��P	�{p���A�*

	conv_lossa�=e[ټ        )��P	٭p���A�*

	conv_lossf�=fGt        )��P	��p���A�*

	conv_losst_�=��p3        )��P	�q���A�*

	conv_loss_@�=�V��        )��P	�Dq���A�*

	conv_loss4�=�Pl�        )��P	9xq���A�*

	conv_loss�Q�=�7�        )��P	�q���A�*

	conv_loss?��=�(��        )��P	{�q���A�*

	conv_lossÖ�=P�ʭ        )��P	r���A�*

	conv_loss��=y�        )��P	l>r���A�*

	conv_lossI�=�=�        )��P	�tr���A�*

	conv_loss;�=]$}        )��P	b�r���A�*

	conv_loss[��=�:]'        )��P	��r���A�*

	conv_loss�3�=�ڰA        )��P	'"s���A�*

	conv_loss�=�ӡ6        )��P	�Zs���A�*

	conv_loss�=�=�ؑY        )��P	
�s���A�*

	conv_lossw|�=�.n�        )��P	�s���A�*

	conv_loss���=�N�%        )��P	��s���A�*

	conv_lossu��=��~�        )��P	$t���A�*

	conv_loss�:k=	#J�        )��P	iTt���A�*

	conv_lossT��=�E�P        )��P	#�t���A�*

	conv_lossx4�=*�8�        )��P	¹t���A�*

	conv_lossp�=��b        )��P	?�t���A�*

	conv_loss��=F��r        )��P	B u���A�*

	conv_loss�Q�=���        )��P	�Qu���A�*

	conv_loss�=c�[�        )��P	��u���A�*

	conv_loss���=rdMv        )��P	v�u���A�*

	conv_lossէ�=�j�        )��P	DHw���A�*

	conv_loss���=q���        )��P	Vxw���A�*

	conv_losswc�=l�}        )��P	��w���A�*

	conv_loss)��=��!x        )��P	Q�w���A�*

	conv_lossǁ�=���        )��P	�x���A�*

	conv_loss�(�=�9�        )��P	;Dx���A�*

	conv_loss���=�Ƽ�        )��P	wx���A�*

	conv_loss�{�=tV        )��P	��x���A�*

	conv_loss��=�m�        )��P	k�x���A�*

	conv_loss8�=T�t        )��P	�y���A�*

	conv_loss'b�=�F�?        )��P	IFy���A�*

	conv_loss���=0-��        )��P	�uy���A�*

	conv_loss{��=�FX        )��P	O�y���A�*

	conv_loss̱�=Q�        )��P	��y���A�*

	conv_loss���=7        )��P	�
z���A�*

	conv_lossZw�=k���        )��P	Az���A�*

	conv_losslP�=��=9        )��P	�qz���A�*

	conv_loss��=�v^        )��P	/�z���A�*

	conv_loss\�=��01        )��P	E�z���A�*

	conv_loss���=W.~�        )��P	~{���A�*

	conv_loss�9�=g9{7        )��P	�V{���A�*

	conv_lossZ<�=�4�&        )��P	�{���A�*

	conv_loss�B�=�|        )��P	�{���A�*

	conv_loss��=��L�        )��P	��{���A�*

	conv_loss^�=IF�!        )��P	� |���A�*

	conv_loss>�=x��L        )��P	�P|���A�*

	conv_loss���=&mY        )��P	�|���A�*

	conv_lossצ�=����        )��P	o�|���A�*

	conv_loss� �=/3�        )��P	��|���A�*

	conv_loss6?�=��+        )��P	�}���A�*

	conv_loss���=�:*e        )��P	�K}���A�*

	conv_loss"x�=u`��        )��P	��}���A�*

	conv_lossD6�=�(��        )��P	Ҳ}���A�*

	conv_loss	\j=qY<        )��P	C�}���A�*

	conv_loss�Y�=�⃽        )��P	�~���A�*

	conv_loss��=0��        )��P	{Q~���A�*

	conv_loss"��=̙^        )��P	��~���A�*

	conv_loss\޶=��        )��P	��~���A�*

	conv_loss���=�K�5        )��P	T�~���A�*

	conv_loss�z=��.�        )��P	9���A�*

	conv_lossX��=ґ�        )��P	nA���A�*

	conv_lossW:�=��$        )��P	�u���A�*

	conv_loss���=_:�        )��P	t����A�*

	conv_lossh��=�i@h        )��P	g����A�*

	conv_loss*=��         )��P	{����A�*

	conv_loss[�=���        )��P	MK����A�*

	conv_lossSù=�y�        )��P	������A�*

	conv_loss]��=ӎ�        )��P	}�����A�*

	conv_loss���=����        )��P	�����A�*

	conv_loss��=?X��        )��P	�����A�*

	conv_loss;��=��|�        )��P	CY����A�*

	conv_loss�^�=�U֊        )��P	8�����A�*

	conv_loss2$�=8F�        )��P	N�����A�*

	conv_loss��=r�Z�        )��P	����A�*

	conv_loss��=2���        )��P	�&����A�*

	conv_loss=��=���        )��P	�a����A�*

	conv_loss9��=T�@�        )��P	ۙ����A�*

	conv_loss��=1��        )��P	Rɂ���A�*

	conv_lossq�=G��        )��P	,����A�*

	conv_lossv�=�� 4        )��P	13����A�*

	conv_loss	{�=1��@        )��P	cc����A�*

	conv_loss��=����        )��P	\�����A�*

	conv_loss���=HL�        )��P	�̓���A�*

	conv_loss�4�=�=J�        )��P	������A�*

	conv_loss鶞=��        )��P	�/����A�*

	conv_loss�c�=��'d        )��P	�a����A�*

	conv_loss��o=)�0�        )��P	�����A�*

	conv_lossBc�=<��        )��P	sʄ���A�*

	conv_loss2��=Z̍        )��P	>�����A�*

	conv_loss:&�=3�C�        )��P	�5����A�*

	conv_loss
+�=#�+        )��P	�f����A�*

	conv_loss(t�=v��        )��P	f�����A�*

	conv_lossy��=�̳�        )��P	�ǅ���A�*

	conv_loss�Y�=7���        )��P	�����A�*

	conv_lossO�=�<�	        )��P	|5����A�*

	conv_loss4�=��>�        )��P	g����A�*

	conv_losssܶ="ic        )��P	����A�*

	conv_loss��=�JC�        )��P	�̆���A�*

	conv_lossXB�=�(�B        )��P	|�����A�*

	conv_loss=��=��j�        )��P	-����A�*

	conv_loss�r�=.$�`        )��P	ud����A�*

	conv_loss�%|=��        )��P	������A�*

	conv_loss'�=5/��        )��P	ḋ���A�*

	conv_loss��=:v�u        )��P	L�����A�*

	conv_losss��=��        )��P	�-����A�*

	conv_lossΙ=iC�e        )��P	�]����A�*

	conv_loss��=ڊ�e        )��P	Ҏ����A�*

	conv_loss�_�=a�        )��P	Y�����A�*

	conv_loss��==�T�        )��P	�����A�*

	conv_lossC��=�t        )��P	7$����A�*

	conv_loss*�=�;O        )��P	=V����A�*

	conv_loss���=�}�        )��P	0�����A�*

	conv_loss�ʵ=�O�        )��P	�����A�*

	conv_loss	��=��k&        )��P	�����A�*

	conv_loss:eK=I%u�        )��P	d����A�*

	conv_loss��=�j�        )��P	'C����A�*

	conv_lossb+�=@c>3        )��P	Vq����A�*

	conv_loss��=~#I        )��P	������A�*

	conv_loss�3�=���        )��P	Dъ���A�*

	conv_lossMC�=��+        )��P	w9����A�*

	conv_lossV�=�k��        )��P	bx����A�*

	conv_loss�~�=oP�        )��P	������A�*

	conv_loss�`�=�^�M        )��P	֏���A�*

	conv_loss���=��        )��P	E����A�*

	conv_lossn�=��0�        )��P	A:����A�*

	conv_loss��=��i�        )��P	�j����A�*

	conv_loss- >�W�U        )��P	������A�*

	conv_loss/�=��ƺ        )��P	�ǐ���A�*

	conv_loss��=y3        )��P	�����A�*

	conv_loss;��=����        )��P	�6����A�*

	conv_loss@��=�(�J        )��P	�r����A�*

	conv_loss���=bd        )��P	������A�*

	conv_loss^=�=�k�P        )��P	�ؑ���A�*

	conv_loss˶�=�<        )��P	!����A�*

	conv_lossY�=,�-�        )��P	}B����A�*

	conv_loss�=P$�\        )��P	ur����A�*

	conv_loss3��=��l        )��P	ࡒ���A�*

	conv_lossg>�=���        )��P	�В���A�*

	conv_loss�@�=��9�        )��P	�����A�*

	conv_loss��=v �        )��P	B4����A�*

	conv_loss���=��C        )��P	�b����A�*

	conv_lossF�=�!n�        )��P	6�����A�*

	conv_losse�=�:.        )��P	%�����A�*

	conv_loss7��=���
        )��P	������A�*

	conv_loss�f�=�O�        )��P	�/����A�*

	conv_loss���=���        )��P	�`����A�*

	conv_loss��=U��        )��P	������A�*

	conv_loss��=�p�L        )��P	̿����A�*

	conv_loss.-�=��c�        )��P	�����A�*

	conv_lossn؆=��        )��P	A����A�*

	conv_lossX��=�e        )��P	�K����A�*

	conv_loss7�=U�Z�        )��P	�z����A�*

	conv_loss���=Y2��        )��P	^�����A�*

	conv_losse�=�w�8        )��P	�ؕ���A�*

	conv_lossֳ=j	��        )��P	����A�*

	conv_loss\��=ʇ�        )��P	6����A�*

	conv_loss�=�=
D��        )��P	Fd����A�*

	conv_loss�Ѽ=CJ�]        )��P	������A�*

	conv_lossIܲ=�n7        )��P	�Ė���A�*

	conv_loss���=t���        )��P	������A�*

	conv_lossZ:�=a���        )��P	D"����A�*

	conv_losss;�=��c        )��P	�T����A�*

	conv_loss޻�=9E�V        )��P	プ���A�*

	conv_loss��=p�`�        )��P	�����A�*

	conv_lossD�=mc�U        )��P	�����A�*

	conv_loss{��=*r4{        )��P	�����A�*

	conv_loss���=�Ŵ3        )��P	�@����A�*

	conv_loss:��=Hu        )��P	
o����A�*

	conv_loss���=�.5&        )��P	������A�*

	conv_loss���=\;1        )��P	�ј���A�*

	conv_loss�Ͱ=����        )��P	�����A�*

	conv_loss�=��p�        )��P	�B����A�*

	conv_loss���=��PT        )��P	�s����A�*

	conv_lossӄ=����        )��P	������A�*

	conv_lossH��=�S�        )��P	Gٙ���A�*

	conv_loss���={���        )��P	�
����A�*

	conv_loss��=��(g        )��P	5C����A�*

	conv_loss�r�=�c        )��P	�t����A�*

	conv_loss��=��w	        )��P	������A�*

	conv_lossܷ=Q�&�        )��P	<ޚ���A�*

	conv_lossmߐ=��R:        )��P	�����A�*

	conv_loss-�=]��6        )��P	A����A�*

	conv_loss�٘=X���        )��P	�x����A�*

	conv_loss���=��9        )��P	������A�*

	conv_loss3C�=�
`        )��P	������A�*

	conv_loss��v=Հ6�        )��P	�����A�*

	conv_lossb�=jpE        )��P	$P����A�*

	conv_loss�=��G�        )��P	������A�*

	conv_loss�6�=��        )��P	"�����A�*

	conv_loss�R�=wo��        )��P	�����A�*

	conv_loss)g�=��o�        )��P	d����A�*

	conv_loss޿�=�<�         )��P	�C����A�*

	conv_loss���=B�Aq        )��P	�t����A�*

	conv_lossV>�=��        )��P	٢����A�*

	conv_loss�q�=3G        )��P	7ѝ���A�*

	conv_loss��=����        )��P	����A�*

	conv_loss]+�=Q��T        )��P	.3����A�*

	conv_lossV�=K��3        )��P	cc����A�*

	conv_loss��=��        )��P	ڑ����A�*

	conv_loss�=4f�@        )��P	>�����A�*

	conv_lossϞ�=b��,        )��P	&����A�*

	conv_lossl�=;B        )��P	�����A�*

	conv_lossdg�=�4�Z        )��P	�M����A�*

	conv_loss�ˎ=�2        )��P	e�����A�*

	conv_loss>�=A��        )��P	������A�*

	conv_lossl�=!�\        )��P	�����A�*

	conv_lossG=��Ԍ        )��P	V����A�*

	conv_loss���=E�<�        )��P	@����A�*

	conv_loss��=`�:W        )��P	�w����A�*

	conv_loss�=��;        )��P	߰����A�*

	conv_loss�?�=�        )��P	J����A�*

	conv_loss��=�Ԥ�        )��P	����A�*

	conv_losst�=��        )��P	�I����A�*

	conv_losse�=�%�V        )��P	�}����A�*

	conv_loss���=[}X�        )��P	)�����A�*

	conv_loss=�=�ʛ        )��P	 ����A�*

	conv_loss��=Ci        )��P	����A�*

	conv_loss�h=����        )��P	�B����A�*

	conv_loss�޲=O�
        )��P	0����A�*

	conv_loss��=�%�        )��P	,�����A�*

	conv_loss���=�QS�        )��P	�I����A�*

	conv_loss앛=Ia8�        )��P	̓����A�*

	conv_loss{>�=���        )��P	E�����A�*

	conv_lossl�=�:��        )��P	������A�*

	conv_loss�֭=�3�        )��P	�����A�*

	conv_lossR��=�E�        )��P	]����A�*

	conv_loss�ݫ=z���        )��P	������A�*

	conv_loss���=<1bm        )��P	�ȥ���A�*

	conv_loss��=����        )��P	o�����A�*

	conv_lossύ�==`�        )��P	,,����A�*

	conv_losse˵=$�4        )��P	e����A�*

	conv_loss�4�=�P�        )��P	F�����A�*

	conv_loss]Ҿ=���        )��P	�ʦ���A�*

	conv_lossW�=*���        )��P	L�����A�*

	conv_lossj��=F2U        )��P	z*����A�*

	conv_lossƽ�=�Ș�        )��P	}f����A�*

	conv_loss%4�=���[        )��P	$�����A�*

	conv_loss�=z�cF        )��P	�ħ���A�*

	conv_lossq�=����        )��P	{�����A�*

	conv_loss��=�č�        )��P	�*����A�*

	conv_lossLW�=�`��        )��P	lg����A�*

	conv_loss�wi=��K        )��P	2�����A�*

	conv_loss�t�=���D        )��P	�Ѩ���A�*

	conv_loss�<�=�ߴ�        )��P	����A�*

	conv_loss[3�=�귢        )��P	C4����A�*

	conv_loss�l�=���        )��P	�b����A�*

	conv_loss*6�=ɭA�        )��P	3�����A�*

	conv_loss���=���        )��P	�©���A�*

	conv_loss���=���        )��P	[����A�*

	conv_loss��E=T��m        )��P	�$����A�*

	conv_lossvo�=��2�        )��P	LT����A�*

	conv_loss9�= �G�        )��P	������A�*

	conv_loss�չ==�:        )��P	������A�*

	conv_lossk	�=1��        )��P	�����A�*

	conv_loss2�=|�z        )��P	r����A�*

	conv_loss�Ʃ=A�>b        )��P	�H����A�*

	conv_loss�(�=C#��        )��P	w����A�*

	conv_loss�=�R�i        )��P	N�����A�*

	conv_lossFŴ=ߙ>�        )��P	�֫���A�*

	conv_loss���=&BDv        )��P	�����A�*

	conv_loss���=ۊ        )��P	�9����A�*

	conv_lossU�=��U        )��P	 i����A�*

	conv_lossm�=���        )��P	������A�*

	conv_loss�=���i        )��P	cȬ���A�*

	conv_lossSu�=s��]        )��P	������A�*

	conv_loss��c=��oY        )��P	�)����A�*

	conv_loss���=��C        )��P	�Z����A�*

	conv_loss}��=�-�{        )��P	*�����A�*

	conv_lossZ_�=�z˸        )��P	޺����A�*

	conv_loss��=�pyn        )��P	Z����A�*

	conv_lossð=���        )��P	7����A�*

	conv_loss�;�=�T��        )��P	ql����A�*

	conv_loss=�=T�h�        )��P	������A�*

	conv_loss�R�=�d�        )��P	�Ϯ���A�*

	conv_loss ��=���        )��P	������A�*

	conv_lossoȞ=k[        )��P	1����A�*

	conv_loss
�=�~z~        )��P	1_����A�*

	conv_loss�z�=�y�        )��P	떯���A�*

	conv_loss�ה=���        )��P	eү���A�*

	conv_loss1�=Db��        )��P	�����A�*

	conv_loss�H�=�!        )��P	:����A�*

	conv_loss3]�=�E�=        )��P	�h����A�*

	conv_loss�+W=ȕz�        )��P	������A�*

	conv_lossv�}=,��        )��P	�Ұ���A�*

	conv_loss}�=����        )��P	�����A�*

	conv_loss��~=b        )��P	�3����A�*

	conv_loss��}=�i�        )��P	�d����A�*

	conv_loss���=�D@�        )��P	ǟ����A�*

	conv_loss���=͞�        )��P	@б���A�*

	conv_loss �=�/۔        )��P	� ����A�*

	conv_loss�M�=b6T{        )��P	@1����A�*

	conv_loss^�=���        )��P	j����A�*

	conv_lossBѝ=��U&        )��P	������A�*

	conv_loss��=�hV�        )��P	�Ѳ���A�*

	conv_loss�={���        )��P	����A�*

	conv_loss|׮=���        )��P	�0����A�*

	conv_lossn��=LQWk        )��P	%a����A�*

	conv_lossKW�=&�        )��P	������A�*

	conv_lossXؗ=��        )��P	������A�*

	conv_lossm�=����        )��P	�����A�*

	conv_loss��=�q7�        )��P	� ����A�*

	conv_loss��=��#�        )��P	DP����A�*

	conv_loss�5�=�g9        )��P	�����A�*

	conv_loss>C�=�t�}        )��P	!�����A�*

	conv_loss���=6F�B        )��P	�ߴ���A�*

	conv_loss:|�=�+8         )��P	}����A�*

	conv_loss=$�=�N�n        )��P	�>����A�*

	conv_lossԫ�=V�։        )��P	^p����A�*

	conv_lossU�=Ez�        )��P	������A�*

	conv_loss��=#���        )��P	�е���A�*

	conv_loss�R�=!�!        )��P	% ����A�*

	conv_lossɱ=���        )��P	�/����A�*

	conv_lossڜ=\M        )��P	_a����A�*

	conv_loss�'�=��-|        )��P	������A�*

	conv_lossF�=��[�        )��P	������A�*

	conv_loss���=p�a        )��P	<����A�*

	conv_loss���=�W�        )��P	-"����A�*

	conv_loss"%�=r��        )��P	�P����A�*

	conv_lossT~�=���        )��P	�����A�*

	conv_lossg]=�I0h        )��P	U�����A�*

	conv_lossJ�=����        )��P	x����A�*

	conv_loss�X�=�5�        )��P	q����A�*

	conv_loss��Y=���        )��P	�O����A�*

	conv_loss��=Ų�        )��P	�~����A�*

	conv_loss:�=�dy�        )��P	G�����A�*

	conv_loss�}�=b�e        )��P	�����A�*

	conv_loss��c=<0��        )��P	�����A�*

	conv_loss�:�=��S�        )��P	N����A�*

	conv_loss���=43�        )��P	ڄ����A�*

	conv_loss$�=�6        )��P	������A�*

	conv_lossXK�=-�T        )��P	������A�*

	conv_loss���=Ό;
        )��P	�.����A�*

	conv_loss�=��p
        )��P	*^����A�*

	conv_loss`�=w�@        )��P	������A�*

	conv_loss�=8�i�        )��P	������A�*

	conv_loss7�=Vh�z        )��P	����A�*

	conv_loss�=�u�O        )��P	� ����A�*

	conv_lossҤ�=� ��        )��P	4P����A�*

	conv_loss�=���        )��P	�����A�*

	conv_lossb<�=�}�        )��P	����A�*

	conv_lossKg�=;�qV        )��P	�߻���A�*

	conv_losslV�=��r�        )��P	�����A�*

	conv_loss�ݤ=����        )��P	u=����A�*

	conv_loss�=c�        )��P	n����A�*

	conv_loss��=�5P�        )��P	������A�*

	conv_loss�/�=wq�x        )��P	�μ���A�*

	conv_lossœ�=��#        )��P	������A�*

	conv_lossF��=�ض�        )��P	�/����A�*

	conv_lossV*�=c�\        )��P	t^����A�*

	conv_loss�&�=d9J        )��P	������A�*

	conv_lossOw�=C�wj        )��P	"�����A�*

	conv_lossWz�=����        )��P	�����A�*

	conv_lossN��=-��i        )��P	�����A�*

	conv_loss��=�a        )��P	RN����A�*

	conv_lossZ4�=g���        )��P	�~����A�*

	conv_loss�)�=��u        )��P	������A�*

	conv_loss�J�='��        )��P	�����A�*

	conv_loss�s�=�?�c        )��P	�����A�*

	conv_loss*��=�UMB        )��P	d>����A�*

	conv_lossg"�=ޜ��        )��P	�u����A�*

	conv_lossX�}=�l�        )��P	X�����A�*

	conv_loss���=Z���        )��P	Hٿ���A�*

	conv_loss�z�=Өj�        )��P	�����A�*

	conv_loss=��=�A        )��P	�D����A�*

	conv_lossW��=.��        )��P	1u����A�*

	conv_lossÊ�=	�<�        )��P	������A�*

	conv_loss�h�=�$��        )��P	K�����A�*

	conv_loss�#�=r9K�        )��P	�����A�*

	conv_loss�լ=�&�        )��P	;����A�*

	conv_loss���=����        )��P	�u����A�*

	conv_loss���=��)        )��P	������A�*

	conv_loss�=�
/        )��P	M�����A�*

	conv_loss ��=�R        )��P	�-����A�*

	conv_loss���=��)�        )��P	Oa����A�*

	conv_loss�p�=���'        )��P	Z�����A�*

	conv_loss���=�C/�        )��P	:�����A�*

	conv_loss�`�=���        )��P	������A�*

	conv_loss��=�I��        )��P	�@����A�*

	conv_loss
�=�`�        )��P	�t����A�*

	conv_losseΑ=�%        )��P	ǧ����A�*

	conv_loss���=���g        )��P	W�����A�*

	conv_loss��t=L�        )��P	O����A�*

	conv_lossA�==��1        )��P	�X����A�*

	conv_loss���=�H\�        )��P	p�����A�*

	conv_lossG۷=��ѳ        )��P	������A�*

	conv_loss�
�=[�F4        )��P	^�����A�*

	conv_loss縯=���$        )��P	�*����A�*

	conv_loss,"�=^�'        )��P	�\����A�*

	conv_loss��=�M��        )��P	������A�*

	conv_lossI �= n��        )��P	Z�����A�*

	conv_loss���=�Y��        )��P	I�����A�*

	conv_loss�*�=@��        )��P	*����A�*

	conv_lossJW�=A��@        )��P	^_����A�*

	conv_loss�
�=�8l�        )��P	<�����A�*

	conv_loss�0�=2�(�        )��P	t�����A�*

	conv_lossr��=����        )��P	������A�*

	conv_loss,��=c֊*        )��P	'.����A�*

	conv_loss���=��        )��P	�a����A�*

	conv_loss�F�=��.�        )��P	������A�*

	conv_loss�Ѝ=J�V�        )��P	�����A�*

	conv_lossgJ�=�:�        )��P	M�����A�*

	conv_loss|��=y�        )��P	3����A�*

	conv_loss(�=�kV        )��P	�r����A�*

	conv_loss���=U��a        )��P	�����A�*

	conv_loss�1�=PT�        )��P	������A�*

	conv_loss)L�=J�Q�        )��P	:����A�*

	conv_lossb*�=�M�        )��P	�M����A�*

	conv_loss��=MQ�        )��P	����A�*

	conv_loss���=���        )��P	˹����A�*

	conv_loss�q=V��,        )��P	+�����A�*

	conv_loss#�=ާ&        )��P	{(����A�*

	conv_loss��=��m         )��P	�]����A�*

	conv_loss�O�=zS        )��P	#�����A�*

	conv_loss
y=��0z        )��P	������A�*

	conv_loss .�=����        )��P	r�����A�*

	conv_loss��=wf�        )��P	�.����A�*

	conv_loss슌=W�[�        )��P	)b����A�*

	conv_loss���=�*��        )��P	n�����A�*

	conv_losssҦ=�=ޠ        )��P	R�����A�*

	conv_loss��=�_        )��P	�����A�*

	conv_lossT�=�J�        )��P	������A�*

	conv_loss9Qt=2�y        )��P	������A�*

	conv_loss���=�Lf�        )��P	�����A�*

	conv_loss��=�OM�        )��P	%Y����A�*

	conv_loss9�=O�7        )��P	7�����A�*

	conv_loss�ܠ=,��z        )��P	[�����A�*

	conv_loss"K�=m��        )��P	P����A�*

	conv_loss.��=OUh         )��P	�?����A�*

	conv_loss�=;�        )��P	������A�*

	conv_loss���=r�P        )��P	p�����A�*

	conv_lossu˘=|���        )��P	h�����A�*

	conv_loss찇=%��a        )��P	o����A�*

	conv_loss�P�=~��        )��P	�R����A�*

	conv_loss�Ө=��J        )��P	E�����A�*

	conv_loss��x=�i$�        )��P	������A�*

	conv_loss���=�!~�        )��P	�&����A�*

	conv_loss�B�=@�R�        )��P	�[����A�*

	conv_loss��=��)        )��P	������A�*

	conv_loss#i�=$��$        )��P	2�����A�*

	conv_loss�w�=��R        )��P	�
����A�*

	conv_loss$B�=� �o        )��P	�=����A�*

	conv_lossvO�=���        )��P	w����A�*

	conv_loss r�=�(x�        )��P	�����A�*

	conv_loss�?�=��Q~        )��P	������A�*

	conv_loss��=�hv�        )��P	g����A�*

	conv_loss$��=����        )��P	+W����A�*

	conv_loss]��=�j#        )��P	������A�*

	conv_loss�=g<�;        )��P	n�����A�*

	conv_losse�=�KjL        )��P		�����A�*

	conv_loss&��=�\ذ        )��P	 -����A�*

	conv_loss�K�=�:        )��P	�`����A�*

	conv_loss[�=�s��        )��P	L�����A�*

	conv_loss !�=�a+�        )��P	������A�*

	conv_losss��=�m        )��P	�����A�*

	conv_loss^�=R���        )��P	
@����A�*

	conv_lossʌ=�@ �        )��P	]r����A�*

	conv_lossڬ�=`n�        )��P	Y�����A�*

	conv_lossY��=X�        )��P	�����A�*

	conv_loss
f�=K�`        )��P	�����A�*

	conv_loss��=T"-        )��P	@����A�*

	conv_lossd��=	�û        )��P	�z����A�*

	conv_loss���=����        )��P	������A�*

	conv_lossL�=���_        )��P	M�����A�*

	conv_loss�M�=,>        )��P	�#����A�*

	conv_loss���=C�"        )��P	�W����A�*

	conv_lossƐ�=��d�        )��P	������A�*

	conv_loss�m}=Pū�        )��P	������A�*

	conv_loss���=�_�         )��P	V����A�*

	conv_lossZd�=C�e        )��P	a<����A�*

	conv_loss���=<�        )��P	hr����A�*

	conv_loss;�=^l�Y        )��P	x�����A�*

	conv_loss��=��r�        )��P	�����A�*

	conv_loss�o�=�4��        )��P	�(����A�*

	conv_loss���=B�        )��P	%a����A�*

	conv_loss���=�[�        )��P	������A�*

	conv_loss?�=�b�r        )��P	������A�*

	conv_lossj��=]t        )��P	�����A�*

	conv_loss���=�'�t        )��P	�J����A�*

	conv_lossq�h=�N�        )��P	W�����A�*

	conv_loss��=����        )��P	A�����A�*

	conv_lossP@�=�f�        )��P	�����A�*

	conv_loss��=Cј"        )��P		7����A�*

	conv_loss
�=�lΔ        )��P	'j����A�*

	conv_lossJ�f=��~�        )��P	p�����A�*

	conv_loss��=��        )��P	f�����A�*

	conv_loss���=��
�        )��P	E����A�*

	conv_loss#|�=B<l�        )��P	kD����A�*

	conv_loss�9�=p]�[        )��P	�w����A�*

	conv_loss�y=ۼՒ        )��P	@�����A�*

	conv_losseV�=n� �        )��P	������A�*

	conv_loss���=���)        )��P	O����A�*

	conv_lossl^�=�$X        )��P	R����A�*

	conv_loss?>�=~9��        )��P	,�����A�*

	conv_loss(Y�=��ϻ        )��P	������A�*

	conv_loss�r�=���        )��P	7�����A�*

	conv_loss��=ǚ9~        )��P	N-����A�*

	conv_loss3�=ە��        )��P	�a����A�*

	conv_loss��=����        )��P	������A�*

	conv_loss6A�=��        )��P	������A�*

	conv_loss�d�=�)�        )��P	6�����A�*

	conv_loss(��=�O��        )��P	�,����A�*

	conv_loss�R�={9��        )��P	 e����A�*

	conv_loss���=nl        )��P	�����A�*

	conv_losswn�=���        )��P	������A�*

	conv_lossh+�=���        )��P	G����A�*

	conv_loss���=�g{        )��P	h4����A�*

	conv_losse�=L��U        )��P	�g����A�*

	conv_loss}��=8kf&        )��P	i�����A�*

	conv_lossF��=��        )��P	������A�*

	conv_loss❕=Ԋ��        )��P	�����A�*

	conv_loss��=T\1        )��P	<����A�*

	conv_loss�ʃ=�U�        )��P	�o����A�*

	conv_lossXy�=ab��        )��P	������A�*

	conv_lossD��=����        )��P	������A�*

	conv_lossKT�=�L8�        )��P	m����A�*

	conv_loss�z�=�[U        )��P	�?����A�*

	conv_lossud�=��>        )��P	�s����A�*

	conv_loss��=�I�        )��P	h�����A�*

	conv_loss筒=�Q	        )��P	������A�*

	conv_lossSGM=k�        )��P	y����A�*

	conv_loss�b\=�ν�        )��P	7_����A�*

	conv_loss#��=�u5"        )��P	 �����A�*

	conv_loss�6�={�}%        )��P	������A�*

	conv_loss���=H�K�        )��P	������A�*

	conv_loss���=�&        )��P	>1����A�*

	conv_lossW|�=�-�        )��P	�c����A�*

	conv_lossm��=��H�        )��P	������A�*

	conv_loss��=��H        )��P	�����A�*

	conv_lossGD�=/��        )��P	�����A�*

	conv_lossê�=���        )��P	^K����A�*

	conv_lossă�=�^��        )��P	�~����A�*

	conv_loss��=@�4        )��P	�����A�*

	conv_lossK��=�ڰ�        )��P	������A�*

	conv_loss�f�=4!z<        )��P	h+����A�*

	conv_loss�%�=~㷡        )��P	a����A�*

	conv_loss�,�=r�gB        )��P	�����A�*

	conv_loss��=L�a7        )��P	������A�*

	conv_loss��=���        )��P	"�����A�*

	conv_loss\x=����        )��P	9����A�*

	conv_loss>��=$�v�        )��P	Iq����A�*

	conv_loss�W�=ߔ�@        )��P	ʦ����A�*

	conv_loss~��=,��k        )��P	������A�*

	conv_loss/�=��D�        )��P	����A�*

	conv_lossT�=3l,        )��P	bG����A�*

	conv_loss���=nEW7        )��P	+{����A�*

	conv_loss�H�=�{o        )��P	9�����A�*

	conv_loss�ޮ=��,3        )��P	K�����A�*

	conv_loss�I�=â?�        )��P	����A�*

	conv_lossJ�=�#�E        )��P	?R����A�*

	conv_lossp�=�ޢ�        )��P	#�����A�*

	conv_lossx��=��˩        )��P	պ����A�*

	conv_loss�ב=��        )��P	������A�*

	conv_lossLe=o�*�        )��P	A&����A�*

	conv_loss*�x=�ͫ        )��P	(^����A�*

	conv_loss2��=��        )��P	������A�*

	conv_loss�i�=�B�l        )��P	�����A�*

	conv_lossJ��=���        )��P	:�����A�*

	conv_loss/��=K��        )��P	�(����A�*

	conv_loss�=Vv�h        )��P	A_����A�*

	conv_loss�q=�2ͣ        )��P	B�����A�*

	conv_lossؤ�=�tj�        )��P	�����A�*

	conv_loss��=ٺ=D        )��P	`�����A�*

	conv_loss�1�=.+�        )��P	�.����A�*

	conv_losso�=-,g�        )��P	Ra����A�*

	conv_loss�"�=�2Q{        )��P	'�����A�*

	conv_loss5-�=�m��        )��P	A�����A�*

	conv_loss�_�=GN;�        )��P	������A�*

	conv_losssG�=�G�W        )��P	�.����A�*

	conv_lossf��=ë�        )��P	�g����A�*

	conv_loss���=uj�        )��P	P�����A�*

	conv_loss%��=��        )��P	������A�*

	conv_lossQ�h=O0        )��P	5����A�*

	conv_loss �O=�jc        )��P	�H����A�*

	conv_losssf�=���        )��P	E����A�*

	conv_lossG�==�r
        )��P	1�����A�*

	conv_loss�ެ=o��        )��P	D�����A�*

	conv_loss$��=+��        )��P	
����A�*

	conv_lossfC�=P��        )��P	�P����A�*

	conv_lossB�|=3�	        )��P	%�����A�*

	conv_loss�E�=�O��        )��P	������A�*

	conv_loss(�=@4��        )��P	s�����A�*

	conv_loss�=0��        )��P	�&����A�*

	conv_loss�m�=_=,�        )��P	�d����A�*

	conv_loss7�=^&o�        )��P	9�����A�*

	conv_losse��=��eI        )��P	������A�*

	conv_loss���=����        )��P	�����A�*

	conv_lossc7�=x�j+        )��P	*8����A�*

	conv_loss~�=�	y�        )��P	�s����A�*

	conv_loss��=s���        )��P	x�����A�*

	conv_loss5o�=8�|        )��P	�����A�*

	conv_lossZ��=2�<�        )��P	\����A�*

	conv_lossO��=	
�        )��P	�I����A�*

	conv_loss���=��        )��P	������A�*

	conv_lossP�=��C�        )��P	�����A�*

	conv_lossa��=��        )��P	������A�*

	conv_loss��g=��wo        )��P	B'����A�*

	conv_loss�8�=��7;        )��P	Q]����A�*

	conv_loss���=b�        )��P	"�����A�*

	conv_loss��=�Ǚ        )��P	@�����A�*

	conv_loss
�{=��3A        )��P	W ����A�*

	conv_loss�o�=wlU�        )��P	@9����A�*

	conv_loss���=��        )��P	�m����A�*

	conv_lossQh�=��$v        )��P	������A�*

	conv_loss�;�=y��v        )��P	������A�*

	conv_loss�D�=G/!�        )��P	�����A�*

	conv_loss0�=��@        )��P	�?����A�*

	conv_loss�<�=iȪ        )��P	?s����A�*

	conv_loss��=Ȣ	�        )��P	������A�*

	conv_loss��=9��M        )��P	=�����A�*

	conv_loss�;�=؞L�        )��P	����A�*

	conv_loss���=�l�#        )��P	'B����A�*

	conv_lossX;�=�ON        )��P	xu����A�*

	conv_loss��=t�.        )��P	������A�*

	conv_loss�)�=oVB        )��P	B�����A�*

	conv_lossް�=zG�e        )��P	�����A�*

	conv_loss��a=ZZ�U        )��P	�F����A�*

	conv_loss�[�=�j�        )��P	x����A�*

	conv_loss�h�==�~        )��P	B�����A�*

	conv_loss0�=4 ��        )��P	�����A�*

	conv_loss��=�h�        )��P	N�����A�*

	conv_loss_�Q=Z��?        )��P	�|����A�*

	conv_loss�ƈ=`�z        )��P	������A�*

	conv_lossس=���         )��P	Q�����A�*

	conv_loss铷=�Sj        )��P	����A�*

	conv_lossK��=h��s        )��P	bB����A�*

	conv_loss,�x=ώ��        )��P	�q����A�*

	conv_loss���=�/`x        )��P	v�����A�*

	conv_loss3��=�Q�(        )��P	������A�*

	conv_loss��n=AT��        )��P	B  ���A�*

	conv_loss��v=S��        )��P	K  ���A�*

	conv_loss�c�=P׊�        )��P	�|  ���A�*

	conv_loss"��=� w�        )��P	߭  ���A�*

	conv_lossJ݆=
��        )��P	6�  ���A�*

	conv_loss��=>���        )��P	�  ���A�*

	conv_loss�5�=5i��        )��P	O ���A�*

	conv_loss7S�=VZ&)        )��P	�� ���A�*

	conv_loss���=�_�        )��P	�� ���A�*

	conv_lossHU�=U        )��P	X� ���A�*

	conv_loss�;j=�`�         )��P	R ���A�*

	conv_lossD�=j[�        )��P	�K ���A�*

	conv_loss��=U�r        )��P	} ���A�*

	conv_loss���=�_�        )��P	ڬ ���A�*

	conv_loss�ߥ=>�q�        )��P	�� ���A�*

	conv_loss�=uG@g        )��P	�	 ���A�*

	conv_lossߡ=�!��        )��P	�7 ���A�*

	conv_lossv��=иP�        )��P	�f ���A�*

	conv_loss?`�=�r        )��P	�� ���A�*

	conv_loss��=�Ig        )��P	$� ���A�*

	conv_loss��=u�b�        )��P	E ���A�*

	conv_loss�R�=�y��        )��P	�0 ���A�*

	conv_loss/Η=�y��        )��P	�` ���A�*

	conv_loss�W�=��b�        )��P	�� ���A�*

	conv_loss��=�pD�        )��P	�� ���A�*

	conv_loss�+�=Cdu0        )��P	- ���A�*

	conv_loss�{=W�X        )��P	f0 ���A�*

	conv_loss-G�=�nP        )��P	�h ���A�*

	conv_loss�T�==�"C        )��P	�� ���A�*

	conv_loss�т=�e        )��P	<� ���A�*

	conv_loss�W�=��        )��P	� ���A�*

	conv_loss���=�U��        )��P	�@ ���A�*

	conv_loss���=	��        )��P	Zq ���A�*

	conv_loss�)�= 4�p        )��P	� ���A�*

	conv_lossi��=f�        )��P	�� ���A�*

	conv_lossD�f=���        )��P	� ���A�*

	conv_loss���=  �9        )��P	�E ���A�*

	conv_loss���=Z!?        )��P	'v ���A�*

	conv_losssK�=����        )��P	� ���A�*

	conv_loss�=��>#        )��P	� ���A�*

	conv_loss5ҏ=5j#5        )��P	� ���A�*

	conv_loss��=,k�&        )��P	 K ���A�*

	conv_loss���=����        )��P	
� ���A�*

	conv_loss>�j=�S@�        )��P	�� ���A�*

	conv_loss.��=����        )��P	� ���A�*

	conv_loss��=��T�        )��P	a0	 ���A�*

	conv_lossf��=x�        )��P	Jj	 ���A�*

	conv_loss�2�=J^��        )��P	ޣ	 ���A�*

	conv_loss��=?x�        )��P	b�	 ���A�*

	conv_loss�_= ��J        )��P	U
 ���A�*

	conv_lossd��=�K�V        )��P	�X
 ���A�*

	conv_loss�(n=��;1        )��P	[�
 ���A�*

	conv_loss{�=fEH�        )��P	�
 ���A�*

	conv_loss�S�=6-2        )��P	P�
 ���A�*

	conv_loss9��=ƹ��        )��P	# ���A�*

	conv_loss�.�=��s        )��P	5` ���A�*

	conv_loss4��=�*t        )��P	� ���A�*

	conv_lossI�=���>        )��P	e� ���A�*

	conv_loss]�=�V        )��P	�� ���A�*

	conv_loss6�=����        )��P	B* ���A�*

	conv_lossk}�=���        )��P	 ^ ���A�*

	conv_loss���=�&�        )��P	=� ���A�*

	conv_loss���=̧ŭ        )��P	�� ���A�*

	conv_loss怬=��         )��P	A ���A�*

	conv_loss�ju=��x        )��P	�8 ���A�*

	conv_loss�L�=�rj        )��P	�m ���A�*

	conv_loss�w�=����        )��P	 � ���A�*

	conv_loss���=���        )��P	`� ���A�*

	conv_loss�Bp=���        )��P	� ���A�*

	conv_loss�e=Z�9        )��P	"0 ���A�*

	conv_lossSeo=k�        )��P	cc ���A�*

	conv_losscj�=�:=v        )��P	� ���A�*

	conv_lossr�?=A��        )��P	� ���A�*

	conv_loss�=R�Þ        )��P	�� ���A�*

	conv_loss�9�=�r�#        )��P	�, ���A�*

	conv_loss���=XR�a        )��P	}\ ���A�*

	conv_loss#��=�2H        )��P	�� ���A�*

	conv_loss��=�ĭ        )��P	w� ���A�*

	conv_loss4͌=z�d        )��P	�� ���A�*

	conv_loss�}�=�S`        )��P	A8 ���A�*

	conv_lossc��=�O�        )��P	Em ���A�*

	conv_loss�6�=dkl�        )��P	� ���A�*

	conv_loss�Vl=�?RQ        )��P	~� ���A�*

	conv_loss�y�=�-�        )��P	� ���A�*

	conv_loss��=��b�        )��P	�S ���A�*

	conv_lossh��=�8}        )��P	�� ���A�*

	conv_loss�x�=N��        )��P	� ���A�*

	conv_loss��=�x��        )��P	n ���A� *

	conv_loss7)�=z�|        )��P	�F ���A� *

	conv_loss��=��#n        )��P	;� ���A� *

	conv_loss���=���        )��P	R� ���A� *

	conv_loss�*�=�]\        )��P	L� ���A� *

	conv_lossiS�=Q�        )��P	�F ���A� *

	conv_loss*�=$"r@        )��P	�u ���A� *

	conv_loss��}=�8�v        )��P	�� ���A� *

	conv_lossa<�=95a�        )��P	,� ���A� *

	conv_loss%�=�=bi        )��P	( ���A� *

	conv_lossW��=K���        )��P	)P ���A� *

	conv_loss�q~=n�M        )��P	� ���A� *

	conv_lossﶵ=��        )��P	q� ���A� *

	conv_loss8�=+W�u        )��P	%� ���A� *

	conv_lossI|�=��        )��P	1' ���A� *

	conv_lossձr=����        )��P	�] ���A� *

	conv_loss^%�=��m        )��P	@� ���A� *

	conv_loss��=�*e        )��P	�� ���A� *

	conv_loss��=��&|        )��P	�� ���A� *

	conv_loss��Z=��*�        )��P	�" ���A� *

	conv_loss"�l=鶭�        )��P	�U ���A� *

	conv_loss�Z�=�q�w        )��P	'� ���A� *

	conv_loss�K�=��v        )��P	� ���A� *

	conv_loss�\�=�<0        )��P	�� ���A� *

	conv_loss��=�\U�        )��P	� ���A� *

	conv_loss6p�=���V        )��P	�A ���A� *

	conv_lossZ��=�k(7        )��P	ѥ ���A� *

	conv_loss"m=�f        )��P	� ���A� *

	conv_loss�r=?�{h        )��P	� ���A� *

	conv_loss�n�=���k        )��P	�3 ���A� *

	conv_loss{W�=m<�P        )��P	:j ���A� *

	conv_loss�=ITy�        )��P	j� ���A� *

	conv_losse�=��uf        )��P	:� ���A� *

	conv_loss��=g�t�        )��P	�� ���A� *

	conv_lossO��=��i        )��P	) ���A� *

	conv_lossts�=[��        )��P	f[ ���A� *

	conv_losss{=ˮ��        )��P	Ԋ ���A� *

	conv_loss�dZ=��M�        )��P	� ���A� *

	conv_lossۊ�=.�        )��P	�� ���A� *

	conv_loss���=D��        )��P	�% ���A� *

	conv_loss"ty=<        )��P	�V ���A� *

	conv_loss.�=Dt��        )��P	+� ���A� *

	conv_lossa�=>��        )��P	߻ ���A� *

	conv_loss�¥=�d$        )��P	�� ���A� *

	conv_loss�.�=�s��        )��P	* ���A� *

	conv_loss��K=�Gw        )��P	Hg ���A� *

	conv_lossU%�=��        )��P	� ���A� *

	conv_loss�q}=u��C        )��P	�� ���A� *

	conv_loss��=ڈ�^        )��P	�� ���A� *

	conv_loss:
t=��t�        )��P	�& ���A� *

	conv_lossJ��=����        )��P	-V ���A� *

	conv_loss�Ϝ=���-        )��P	� ���A� *

	conv_losshi�=J���        )��P	l� ���A� *

	conv_loss,Y�=oM�        )��P	 � ���A� *

	conv_loss'��=$�[�        )��P	! ���A� *

	conv_loss�߶=�b�V        )��P	` ���A� *

	conv_losshŒ=����        )��P	z� ���A� *

	conv_loss�^= L�t        )��P	� ���A� *

	conv_loss�K�=�2��        )��P	U ���A� *

	conv_loss��=@i�        )��P	�> ���A� *

	conv_lossȍ=W��        )��P	,~ ���A� *

	conv_lossC�l=����        )��P	E� ���A� *

	conv_loss���=��l�        )��P	�� ���A� *

	conv_lossϚo==��A        )��P	> ���A� *

	conv_lossXs�=|f1�        )��P	�P ���A� *

	conv_loss�u�=�t        )��P	1� ���A� *

	conv_loss���=b�+.        )��P	X� ���A� *

	conv_loss��=�]�        )��P	�� ���A� *

	conv_loss뢄=Q^�        )��P	E1  ���A� *

	conv_lossl�=U��        )��P	Y`  ���A� *

	conv_loss)��=j�5        )��P	�  ���A� *

	conv_loss2$�=����        )��P	��  ���A� *

	conv_loss9�=���        )��P	 ! ���A� *

	conv_lossG��=��        )��P	�1! ���A� *

	conv_loss���=�E��        )��P	za! ���A� *

	conv_loss���=�|w_        )��P	F�! ���A� *

	conv_loss$<�=���        )��P	��! ���A� *

	conv_loss��=/=Qz        )��P	��! ���A� *

	conv_loss�v�=�6��        )��P	�2" ���A� *

	conv_loss�ۣ=n��        )��P	�a" ���A� *

	conv_loss4jw=�X�        )��P	s�" ���A� *

	conv_loss�w=ښ�        )��P	7�" ���A� *

	conv_lossު=��o�        )��P	��" ���A� *

	conv_loss:��=W�Y        )��P	�,# ���A� *

	conv_lossMd�=d�l�        )��P	\i# ���A� *

	conv_loss<�=��        )��P	�# ���A� *

	conv_loss\Ԗ=��*�        )��P	��# ���A� *

	conv_lossDµ=����        )��P	�# ���A� *

	conv_loss���=[|q        )��P	�)$ ���A� *

	conv_lossUD�=�J�        )��P	%l$ ���A� *

	conv_loss�҉=���        )��P	w�$ ���A� *

	conv_loss�x=��c        )��P	L�$ ���A� *

	conv_loss�ɳ=:9�#        )��P	�$ ���A� *

	conv_loss���=8�J�        )��P	H+% ���A� *

	conv_loss��=Pt��        )��P	e% ���A� *

	conv_lossJ�=H��        )��P	8�% ���A� *

	conv_loss���=���?        )��P	K�% ���A� *

	conv_loss�4�=���d        )��P	��% ���A� *

	conv_loss.d�=d4�Y        )��P	$& ���A� *

	conv_loss�{�=e�y�        )��P	�R& ���A� *

	conv_lossP�d=�
��        )��P	܁& ���A� *

	conv_lossߦ�=��_r        )��P	�& ���A� *

	conv_loss���=m�e        )��P	��& ���A� *

	conv_loss�$�=���        )��P		' ���A� *

	conv_loss��=���Q        )��P	N@' ���A� *

	conv_loss~�=��c�        )��P	��( ���A� *

	conv_loss&ʐ=��J�        )��P	<) ���A� *

	conv_loss�(1=�M�        )��P	�:) ���A� *

	conv_loss���=�m�        )��P	<i) ���A� *

	conv_lossq�c=� ��        )��P	&�) ���A� *

	conv_loss�ѝ=�)N�        )��P	��) ���A� *

	conv_loss6�v=.�
�        )��P	{�) ���A� *

	conv_loss́�=�!ae        )��P	@-* ���A� *

	conv_lossJ��=_$�        )��P	�\* ���A� *

	conv_loss̽=u�֧        )��P	ښ* ���A� *

	conv_lossF�=k��        )��P	F�* ���A� *

	conv_lossm�=~{��        )��P	��* ���A� *

	conv_loss<�=��m        )��P	�*+ ���A� *

	conv_loss˪�=�R�         )��P	�Y+ ���A� *

	conv_loss���=
|	        )��P	��+ ���A� *

	conv_lossK��=Ő+�        )��P	��+ ���A� *

	conv_lossV��=mboN        )��P	� , ���A� *

	conv_lossn�l=@2�        )��P	_2, ���A� *

	conv_loss=.�=�
�        )��P	8d, ���A� *

	conv_loss���={��        )��P	̑, ���A� *

	conv_loss�ϫ=��z        )��P	)�, ���A� *

	conv_loss�؉=Ghq'        )��P	��, ���A� *

	conv_loss{ڴ=�z�        )��P	�- ���A� *

	conv_loss���=Z��        )��P	N- ���A�!*

	conv_lossԩ�=b�c        )��P	�z- ���A�!*

	conv_lossFq=�r>?        )��P	"�- ���A�!*

	conv_loss�M�=�WA        )��P	��- ���A�!*

	conv_loss~�M=���.        )��P	z. ���A�!*

	conv_loss��=3O        )��P	6>. ���A�!*

	conv_loss�t=7%��        )��P	'q. ���A�!*

	conv_loss��=���.        )��P	��. ���A�!*

	conv_loss��j=H4pb        )��P	p�. ���A�!*

	conv_loss�h�=.[�        )��P	��. ���A�!*

	conv_loss�~=�X^�        )��P	b-/ ���A�!*

	conv_lossӋ�=D�KQ        )��P	]/ ���A�!*

	conv_loss���=c��s        )��P	�/ ���A�!*

	conv_loss���=�#        )��P	��/ ���A�!*

	conv_loss��=5��        )��P	��/ ���A�!*

	conv_lossB��=�;�        )��P	� 0 ���A�!*

	conv_loss�=w��n        )��P	�P0 ���A�!*

	conv_loss^˕=NC�        )��P	��0 ���A�!*

	conv_lossl�=T��<        )��P	��0 ���A�!*

	conv_loss��=�:Q        )��P	��0 ���A�!*

	conv_loss���=�I��        )��P	�1 ���A�!*

	conv_loss���=��iK        )��P	�?1 ���A�!*

	conv_loss��==?��        )��P	�q1 ���A�!*

	conv_loss�Ї=�_�$        )��P	R�1 ���A�!*

	conv_loss�I�=X��        )��P	��1 ���A�!*

	conv_loss�ʫ=h�2�        )��P	�2 ���A�!*

	conv_loss|��=���Z        )��P	�12 ���A�!*

	conv_lossfm=��u        )��P	at2 ���A�!*

	conv_lossD�=���        )��P	1�2 ���A�!*

	conv_loss���=xܺ        )��P	��2 ���A�!*

	conv_loss�i�=�	9        )��P	3 ���A�!*

	conv_loss�C�=�,        )��P	;<3 ���A�!*

	conv_loss���=���        )��P	r3 ���A�!*

	conv_lossh��=mN��        )��P	w�3 ���A�!*

	conv_loss��=�,�        )��P	��3 ���A�!*

	conv_loss�1�=�$L        )��P	4 ���A�!*

	conv_loss���=*�TH        )��P	pL4 ���A�!*

	conv_loss�=�S�        )��P	�{4 ���A�!*

	conv_loss/6�=Q�(�        )��P	0�4 ���A�!*

	conv_loss��=�IZ�        )��P	��4 ���A�!*

	conv_loss��=���*        )��P	�5 ���A�!*

	conv_loss��}=Zmn�        )��P	�C5 ���A�!*

	conv_loss�Ր=�5}�        )��P	�s5 ���A�!*

	conv_loss��=sj2�        )��P	�5 ���A�!*

	conv_loss�k�=�z�        )��P	h�5 ���A�!*

	conv_loss+�=�Z޿        )��P	�6 ���A�!*

	conv_loss
ԃ=ik��        )��P	?6 ���A�!*

	conv_loss�9�=LNv�        )��P	zq6 ���A�!*

	conv_loss'��=�ėj        )��P	آ6 ���A�!*

	conv_loss�˒=�촷        )��P	F�6 ���A�!*

	conv_loss ��=e��B        )��P	�7 ���A�!*

	conv_loss���=�s�        )��P	�97 ���A�!*

	conv_loss'��=K�x�        )��P	j7 ���A�!*

	conv_loss�=#��        )��P	Ӛ7 ���A�!*

	conv_lossH��=m*�        )��P	��7 ���A�!*

	conv_loss}��=I�P�        )��P	�7 ���A�!*

	conv_loss�=$;��        )��P	".8 ���A�!*

	conv_loss�@�=�z�        )��P	�b8 ���A�!*

	conv_loss�c�=e-4        )��P	��8 ���A�!*

	conv_loss�C�=ٷA�        )��P	��8 ���A�!*

	conv_lossj��=���r        )��P	�9 ���A�!*

	conv_loss���=���        )��P	R29 ���A�!*

	conv_loss�9�=_,        )��P	qa9 ���A�!*

	conv_loss���=�\^]        )��P	�9 ���A�!*

	conv_loss'�|=�W�d        )��P	l�9 ���A�!*

	conv_loss�t�=�|R?        )��P	G�9 ���A�!*

	conv_loss3/p=���u        )��P	M: ���A�!*

	conv_lossq�=Z`��        )��P	CR: ���A�!*

	conv_loss�(�=�F:H        )��P	�: ���A�!*

	conv_loss�!�=�-�        )��P	7�: ���A�!*

	conv_losslz�=�ݟ|        )��P	��: ���A�!*

	conv_loss.~�=��\�        )��P	�; ���A�!*

	conv_loss��=�V�        )��P	�@; ���A�!*

	conv_loss
z=
�D^        )��P	�p; ���A�!*

	conv_loss&3�=�w        )��P	��; ���A�!*

	conv_loss���=�/~        )��P	��; ���A�!*

	conv_loss�d�=9��        )��P	�< ���A�!*

	conv_loss*'w=j��        )��P	�O< ���A�!*

	conv_loss�o�=`�e�        )��P	��< ���A�!*

	conv_lossx�R=�u��        )��P	'�< ���A�!*

	conv_lossr�='\        )��P	��< ���A�!*

	conv_lossW_�=��GX        )��P	g= ���A�!*

	conv_loss�}�=5�Q�        )��P	�N= ���A�!*

	conv_loss�`�=�@��        )��P	��= ���A�!*

	conv_loss�=�ŖL        )��P	��= ���A�!*

	conv_loss���=%���        )��P	��= ���A�!*

	conv_loss�p�=HGc~        )��P	�E> ���A�!*

	conv_loss���=z�_        )��P	>z> ���A�!*

	conv_loss}��=c`y�        )��P	��> ���A�!*

	conv_loss@��=*��a        )��P	�> ���A�!*

	conv_loss�n�=���        )��P	�? ���A�!*

	conv_loss;��=���h        )��P	�N? ���A�!*

	conv_lossX8�=��^        )��P	2�? ���A�!*

	conv_loss��=���        )��P	��? ���A�!*

	conv_loss�qK=��        )��P	5�? ���A�!*

	conv_loss�^�=ojP        )��P	z@ ���A�!*

	conv_loss7ܛ=q;�        )��P	�N@ ���A�!*

	conv_loss�0m=&<�?        )��P	��@ ���A�!*

	conv_loss�ć=q�W        )��P	��@ ���A�!*

	conv_loss8�=Ҋo2        )��P	C�@ ���A�!*

	conv_loss��=֋ۅ        )��P	@A ���A�!*

	conv_loss�r_=#J��        )��P	PIA ���A�!*

	conv_loss}"�=��~�        )��P	hyA ���A�!*

	conv_lossn,�=���r        )��P	>�A ���A�!*

	conv_loss�6q=�OFA        )��P	(�A ���A�!*

	conv_loss�Sx=���        )��P	�B ���A�!*

	conv_loss�{�=̹�A        )��P	�EB ���A�!*

	conv_loss��=`?�        )��P	qxB ���A�!*

	conv_loss 4�=9}�g        )��P	��B ���A�!*

	conv_loss[Q=x`*b        )��P	��B ���A�!*

	conv_loss﷐=R<x        )��P	C ���A�!*

	conv_loss�F�=�'�        )��P	oJC ���A�!*

	conv_loss�j�=pZ        )��P	]zC ���A�!*

	conv_loss�E�=�	��        )��P	��C ���A�!*

	conv_lossZD�=^��        )��P	*�C ���A�!*

	conv_loss�i=yW
:        )��P	Q
D ���A�!*

	conv_loss&`�=��g        )��P	-GD ���A�!*

	conv_lossM�n=̛o        )��P	{D ���A�!*

	conv_loss6M�==��        )��P	�D ���A�!*

	conv_loss ��=Ɗ<�        )��P	t�D ���A�!*

	conv_loss��=��j�        )��P	E ���A�!*

	conv_lossAj�=*�4�        )��P	SVE ���A�!*

	conv_loss��L=	*I�        )��P	D�E ���A�!*

	conv_loss֡�=���
        )��P	c�E ���A�!*

	conv_loss#�=���)        )��P	��E ���A�!*

	conv_loss��=�Y�        )��P	�F ���A�!*

	conv_loss�5�=���        )��P	�XF ���A�!*

	conv_loss��i=�$�2        )��P	�F ���A�!*

	conv_loss��=���        )��P	��F ���A�"*

	conv_loss�@=�?��        )��P	T G ���A�"*

	conv_loss���=��T�        )��P	=5G ���A�"*

	conv_loss磠=���'        )��P	hG ���A�"*

	conv_loss�5�=@�l�        )��P	��G ���A�"*

	conv_loss��=�P�        )��P	<�G ���A�"*

	conv_lossob=/9U�        )��P	�H ���A�"*

	conv_loss�(Y=���c        )��P	?CH ���A�"*

	conv_loss5K�=4�h{        )��P	,{H ���A�"*

	conv_loss�b=� �        )��P	M�H ���A�"*

	conv_lossS�=屬�        )��P	.�H ���A�"*

	conv_loss��=yk�J        )��P	�I ���A�"*

	conv_lossCڟ="�y�        )��P	�CI ���A�"*

	conv_loss+(p=E­�        )��P	�{I ���A�"*

	conv_loss��=�D��        )��P	ˮI ���A�"*

	conv_loss���=�{�e        )��P	%�I ���A�"*

	conv_loss��l=^J8        )��P	�J ���A�"*

	conv_loss/�=��ry        )��P	;KJ ���A�"*

	conv_loss�s=�4        )��P	��J ���A�"*

	conv_loss�z�=ES�&        )��P	�J ���A�"*

	conv_loss�k=K>I,        )��P	m�J ���A�"*

	conv_lossu�N=���v        )��P	�"K ���A�"*

	conv_lossof�=rq��        )��P	CZK ���A�"*

	conv_loss�u�=� n        )��P	�K ���A�"*

	conv_loss��w=�g��        )��P	��K ���A�"*

	conv_loss�\�=��57        )��P	m�K ���A�"*

	conv_loss/O=�$f@        )��P	�.L ���A�"*

	conv_loss%e�=�;�1        )��P	w`L ���A�"*

	conv_lossOy=�4-�        )��P	��L ���A�"*

	conv_loss�%�=��Nv        )��P	��L ���A�"*

	conv_loss��o=�P�         )��P	qM ���A�"*

	conv_loss�P=x�:�        )��P	�GM ���A�"*

	conv_lossP�=���c        )��P	 xM ���A�"*

	conv_loss���=QҖ�        )��P	��M ���A�"*

	conv_lossN�=syW'        )��P	|�M ���A�"*

	conv_lossܳ�=U�        )��P	[N ���A�"*

	conv_loss
��=��+        )��P	�<N ���A�"*

	conv_loss"Pq=D��1        )��P	inN ���A�"*

	conv_loss��p=?$�8        )��P	�N ���A�"*

	conv_loss�n=��ъ        )��P	��N ���A�"*

	conv_lossd�j=�\�        )��P	�O ���A�"*

	conv_loss��=��        )��P	�7O ���A�"*

	conv_loss`��=�ܨ*        )��P	lhO ���A�"*

	conv_loss3k�=F�O�        )��P	)�O ���A�"*

	conv_loss�q=O/*�        )��P	��O ���A�"*

	conv_loss��=$�1�        )��P	U�O ���A�"*

	conv_loss,��=u��        )��P	�.P ���A�"*

	conv_lossB�=!X�        )��P	_P ���A�"*

	conv_lossaߏ=�ì�        )��P	a�P ���A�"*

	conv_loss��w=u�o        )��P	�&R ���A�"*

	conv_loss���=u�        )��P	qXR ���A�"*

	conv_loss�|r=���        )��P	��R ���A�"*

	conv_losss��=y��+        )��P	j�R ���A�"*

	conv_loss��=+�Y        )��P	��R ���A�"*

	conv_loss�F�=Wv	        )��P	x.S ���A�"*

	conv_lossU��='ۛ�        )��P	%`S ���A�"*

	conv_loss���=�`�9        )��P	�S ���A�"*

	conv_loss�z�=�8�        )��P	��S ���A�"*

	conv_loss��]=A�*        )��P	�T ���A�"*

	conv_loss>�=X8�v        )��P	�7T ���A�"*

	conv_lossi��=�|�        )��P	-qT ���A�"*

	conv_loss�Ԇ=k<��        )��P	��T ���A�"*

	conv_loss��=��        )��P	I�T ���A�"*

	conv_loss�I=#�y        )��P	LU ���A�"*

	conv_loss�Ȍ=�CU        )��P	)?U ���A�"*

	conv_loss��j=G �        )��P	spU ���A�"*

	conv_loss��=&��R        )��P	ɟU ���A�"*

	conv_loss�K�=?�	        )��P	4�U ���A�"*

	conv_lossqvC=��        )��P	�V ���A�"*

	conv_loss&Q�=6���        )��P	V6V ���A�"*

	conv_loss��[=�4�        )��P	�fV ���A�"*

	conv_lossSg�=ݥ�>        )��P	|�V ���A�"*

	conv_loss��=o�         )��P	��V ���A�"*

	conv_lossL�> J��        )��P	}�V ���A�"*

	conv_loss��=�$q�        )��P	*W ���A�"*

	conv_loss/6�=ˊ�        )��P	�ZW ���A�"*

	conv_loss܈�=X���        )��P	�W ���A�"*

	conv_lossY0T=E��        )��P	o�W ���A�"*

	conv_loss8�=��cv        )��P	��W ���A�"*

	conv_loss�B�=�d$X        )��P	qX ���A�"*

	conv_loss߬�=?�'�        )��P	OX ���A�"*

	conv_loss���=�\��        )��P	�X ���A�"*

	conv_loss6��=�Qɍ        )��P	��X ���A�"*

	conv_lossT��=�s�\        )��P	�X ���A�"*

	conv_lossK+\=`#��        )��P	8Y ���A�"*

	conv_lossS#�=�	        )��P	�MY ���A�"*

	conv_loss7��=�~E        )��P	k}Y ���A�"*

	conv_losslǣ=� '        )��P	,�Y ���A�"*

	conv_loss��=�F        )��P	j�Y ���A�"*

	conv_loss��=�z��        )��P	vZ ���A�"*

	conv_loss�_�=�U��        )��P	�FZ ���A�"*

	conv_loss�m=��        )��P	{Z ���A�"*

	conv_loss5��=����        )��P	��Z ���A�"*

	conv_loss4ˈ=��v�        )��P	��Z ���A�"*

	conv_loss��=��        )��P	�[ ���A�"*

	conv_loss'�=N��r        )��P	�>[ ���A�"*

	conv_loss�Q=W�¤        )��P	�p[ ���A�"*

	conv_loss�p=�c"�        )��P	��[ ���A�"*

	conv_loss���=�8�        )��P	��[ ���A�"*

	conv_loss��=�݃        )��P	a\ ���A�"*

	conv_loss=^�=�qWo        )��P	�M\ ���A�"*

	conv_lossCv=���        )��P	7~\ ���A�"*

	conv_loss���=}��        )��P	ͱ\ ���A�"*

	conv_loss�.�=��        )��P	_�\ ���A�"*

	conv_loss��o=$s��        )��P	�] ���A�"*

	conv_loss���=��	�        )��P	�O] ���A�"*

	conv_loss���=�\38        )��P	b�] ���A�"*

	conv_loss ��=���L        )��P	'�] ���A�"*

	conv_loss��=G�Z�        )��P	P�] ���A�"*

	conv_loss>I�=(���        )��P	`&^ ���A�"*

	conv_loss?|u=~�&        )��P	�d^ ���A�"*

	conv_loss�)�=�-�?        )��P	��^ ���A�"*

	conv_loss49�=���z        )��P	M�^ ���A�"*

	conv_loss��=~�5        )��P	v_ ���A�"*

	conv_loss0%�=@��c        )��P	]?_ ���A�"*

	conv_loss��=���I        )��P	Vx_ ���A�"*

	conv_loss�`�=����        )��P	M�_ ���A�"*

	conv_loss���=�4�        )��P	G�_ ���A�"*

	conv_loss�(�=�[��        )��P	` ���A�"*

	conv_lossW<�=k��        )��P	C>` ���A�"*

	conv_lossR �=��        )��P	�p` ���A�"*

	conv_lossgRI=��כ        )��P	�` ���A�"*

	conv_loss��V=�z3�        )��P	_�` ���A�"*

	conv_loss��&= �        )��P	�a ���A�"*

	conv_lossIZ_=��G        )��P	~Oa ���A�"*

	conv_loss�
�=�Ae8        )��P	�a ���A�"*

	conv_loss�j�=���        )��P	��a ���A�"*

	conv_lossf��=:M�X        )��P	�a ���A�"*

	conv_loss�|{=C�        )��P	;b ���A�#*

	conv_loss)4z=����        )��P	�Kb ���A�#*

	conv_loss�E�='A�Z        )��P	+}b ���A�#*

	conv_loss���=�I�        )��P	\�b ���A�#*

	conv_lossы=�\y        )��P	�b ���A�#*

	conv_loss�آ=���o        )��P	�c ���A�#*

	conv_loss �=F��        )��P	�Fc ���A�#*

	conv_loss�!�=�]        )��P	�yc ���A�#*

	conv_loss��=0Ӽ�        )��P	��c ���A�#*

	conv_loss��=Z>��        )��P	��c ���A�#*

	conv_loss���=��i        )��P	�d ���A�#*

	conv_lossi�=��j4        )��P	>d ���A�#*

	conv_loss}Fb=���s        )��P	�od ���A�#*

	conv_loss�eu=ۅ|�        )��P	��d ���A�#*

	conv_losss?�=���        )��P	��d ���A�#*

	conv_loss�GT=��*�        )��P		e ���A�#*

	conv_lossw��=�jXT        )��P	a;e ���A�#*

	conv_loss�	G=�F��        )��P	�me ���A�#*

	conv_lossdW�=<Y�t        )��P	��e ���A�#*

	conv_losszt�=�2]�        )��P	��e ���A�#*

	conv_loss�{�=~�x        )��P	Έj ���A�#*

	conv_loss�v=��        )��P	��j ���A�#*

	conv_loss���=C�I        )��P	�k ���A�#*

	conv_loss;�=���        )��P	rGk ���A�#*

	conv_loss�Jr=�Gw&        )��P	�{k ���A�#*

	conv_lossԷK=���        )��P	Z�k ���A�#*

	conv_loss��=V�3        )��P	��k ���A�#*

	conv_loss�~�=,�B�        )��P	�l ���A�#*

	conv_loss���=���E        )��P	�Bl ���A�#*

	conv_loss���=#�0"        )��P	�tl ���A�#*

	conv_loss�%�=TB�        )��P	�l ���A�#*

	conv_loss�=�o��        )��P	��l ���A�#*

	conv_loss�r=���        )��P	�m ���A�#*

	conv_lossfe�=�(S        )��P	Bm ���A�#*

	conv_loss�s~=A�        )��P	�zm ���A�#*

	conv_lossMf�=�U>         )��P	�m ���A�#*

	conv_loss�F�=|���        )��P	z�m ���A�#*

	conv_loss��=�hZ        )��P	n ���A�#*

	conv_loss��d=\Gͣ        )��P	&Un ���A�#*

	conv_loss�:Z=`�;        )��P	Ån ���A�#*

	conv_loss� �=�i        )��P	��n ���A�#*

	conv_loss�O=�N�        )��P	S�n ���A�#*

	conv_loss���=RlJ�        )��P	o ���A�#*

	conv_loss��{=��W�        )��P	$To ���A�#*

	conv_loss**�=�m�        )��P	X�o ���A�#*

	conv_loss"�=|�`�        )��P	N�o ���A�#*

	conv_loss�ύ=���        )��P	�o ���A�#*

	conv_lossƝ�=m���        )��P	rp ���A�#*

	conv_loss��Z=���H        )��P	R?p ���A�#*

	conv_loss�W�=7��        )��P	�mp ���A�#*

	conv_lossp�|=;�wY        )��P	K�p ���A�#*

	conv_loss�{�=U��j        )��P	v�p ���A�#*

	conv_loss�F�=����        )��P	��p ���A�#*

	conv_loss�~=��E)        )��P	n+q ���A�#*

	conv_loss�Y=��/        )��P	�iq ���A�#*

	conv_loss༕={}1;        )��P	N�q ���A�#*

	conv_loss=S�=7�        )��P	��q ���A�#*

	conv_loss�`�=��GT        )��P	�q ���A�#*

	conv_loss* Y=�x        )��P	A*r ���A�#*

	conv_loss[�\=����        )��P	�Xr ���A�#*

	conv_loss��r=D��        )��P	A�r ���A�#*

	conv_loss�c�=�0�1        )��P	��r ���A�#*

	conv_loss@�=7���        )��P	��r ���A�#*

	conv_lossl�=m8F4        )��P	Cs ���A�#*

	conv_loss2�T=�u��        )��P	�Hs ���A�#*

	conv_lossb�=$��        )��P	`ws ���A�#*

	conv_loss��=1��<        )��P	��s ���A�#*

	conv_lossI�=}��`        )��P	c�s ���A�#*

	conv_loss��=x-��        )��P	�t ���A�#*

	conv_loss.h=�E�H        )��P	76t ���A�#*

	conv_loss�q7=h��+        )��P	�et ���A�#*

	conv_loss�r�=/��        )��P	/�t ���A�#*

	conv_lossa�P=�9F        )��P	�t ���A�#*

	conv_loss��\=3��%        )��P	qu ���A�#*

	conv_loss���=�p        )��P	�Du ���A�#*

	conv_lossr�X==5G=        )��P	u�u ���A�#*

	conv_loss�7=(��}        )��P	��u ���A�#*

	conv_loss��[=�e��        )��P	]�u ���A�#*

	conv_loss��=3@<�        )��P	`v ���A�#*

	conv_loss́�=5��#        )��P	�Nv ���A�#*

	conv_loss���=�3T#        )��P	v�v ���A�#*

	conv_loss��=�},�        )��P	��v ���A�#*

	conv_loss더=E�s
        )��P	��v ���A�#*

	conv_lossd��=\�/        )��P	�$w ���A�#*

	conv_lossrE�=���        )��P	HWw ���A�#*

	conv_loss��=��>�        )��P	˘w ���A�#*

	conv_loss���=IQtE        )��P	]�w ���A�#*

	conv_loss�\�=.ݾ        )��P	v�w ���A�#*

	conv_loss��P=X�ҕ        )��P	}-x ���A�#*

	conv_lossұ�=x�~        )��P	�bx ���A�#*

	conv_loss�=X�+�        )��P	�x ���A�#*

	conv_loss܁�=���|        )��P	��x ���A�#*

	conv_loss+~p=ǀ~	        )��P	7y ���A�#*

	conv_lossm�=����        )��P	�9y ���A�#*

	conv_lossZ=�=�9�        )��P	�jy ���A�#*

	conv_lossi��=o�_u        )��P	��y ���A�#*

	conv_lossV�l=�߳�        )��P	;�y ���A�#*

	conv_lossW�=#��[        )��P	v
z ���A�#*

	conv_loss�yz=�G�]        )��P	y<z ���A�#*

	conv_loss��:=��,�        )��P	�nz ���A�#*

	conv_loss[LQ=~��        )��P	P�z ���A�#*

	conv_loss	��=�o��        )��P	%�z ���A�#*

	conv_lossّz=����        )��P	�{ ���A�#*

	conv_loss-�-=�)�]        )��P	_9{ ���A�#*

	conv_lossskP=p<        )��P	j{ ���A�#*

	conv_lossMi�=��?        )��P	�{ ���A�#*

	conv_lossM׏=n%��        )��P	��{ ���A�#*

	conv_loss�Jb=���        )��P	�| ���A�#*

	conv_loss>W�=e�S        )��P	*4| ���A�#*

	conv_loss��q=jc��        )��P	�e| ���A�#*

	conv_loss�O=̧�H        )��P	��| ���A�#*

	conv_loss���=�H��        )��P	�| ���A�#*

	conv_loss���=W��        )��P	�| ���A�#*

	conv_loss��=L�        )��P	g)} ���A�#*

	conv_lossms�=2~)�        )��P	\} ���A�#*

	conv_loss��t=~6�        )��P	o�} ���A�#*

	conv_loss�B�=��        )��P	��} ���A�#*

	conv_loss� �=F��#        )��P	I�} ���A�#*

	conv_lossB��=��        )��P	�%~ ���A�#*

	conv_loss��=:3        )��P	W~ ���A�#*

	conv_loss`n�=���g        )��P	K�~ ���A�#*

	conv_loss=B�=��:�        )��P	�<� ���A�#*

	conv_lossW�=?Z�/        )��P	�p� ���A�#*

	conv_loss'�z=S�        )��P	à� ���A�#*

	conv_loss�B�=�15        )��P	�Ҁ ���A�#*

	conv_loss�O{=!���        )��P		� ���A�#*

	conv_lossp��=0LN        )��P	�E� ���A�#*

	conv_lossq��=LMB        )��P	�x� ���A�#*

	conv_loss[��=A��b        )��P	б� ���A�$*

	conv_loss~�=`���        )��P	5� ���A�$*

	conv_loss��=A#�        )��P	O� ���A�$*

	conv_loss�Ƌ=�r�7        )��P	H� ���A�$*

	conv_lossQE�=��&        )��P	$z� ���A�$*

	conv_loss��=P"�l        )��P	C�� ���A�$*

	conv_lossE�=� S�        )��P	u݂ ���A�$*

	conv_loss��=���=        )��P	�� ���A�$*

	conv_lossد�=p���        )��P	�>� ���A�$*

	conv_loss/=S=H���        )��P	�m� ���A�$*

	conv_loss%:P=����        )��P	��� ���A�$*

	conv_lossV�=Cf�        )��P	�΃ ���A�$*

	conv_loss5��=�&�        )��P	� � ���A�$*

	conv_loss"��=u��        )��P	�6� ���A�$*

	conv_lossfj}=��S�        )��P	6j� ���A�$*

	conv_loss���=���        )��P	��� ���A�$*

	conv_loss~�v=���        )��P	Oʄ ���A�$*

	conv_loss��y=V�-�        )��P	Q� ���A�$*

	conv_loss�H�=��R�        )��P	�;� ���A�$*

	conv_loss�ւ=�F�        )��P	,s� ���A�$*

	conv_loss�m�=2��        )��P	ᤅ ���A�$*

	conv_loss�U3=���        )��P	JՅ ���A�$*

	conv_loss'�@=n�!�        )��P	�� ���A�$*

	conv_loss�U�=�G~�        )��P	d6� ���A�$*

	conv_loss�x=�'�D        )��P	�m� ���A�$*

	conv_loss]�=��I�        )��P	h�� ���A�$*

	conv_loss��=n>W�        )��P	�ن ���A�$*

	conv_loss��=��ʅ        )��P	� ���A�$*

	conv_lossT�=Ľ�U        )��P	QE� ���A�$*

	conv_loss1ۆ=W3��        )��P	�|� ���A�$*

	conv_loss&�h=�f��        )��P	̭� ���A�$*

	conv_lossE[=7�OG        )��P	�܇ ���A�$*

	conv_lossײ=�uG        )��P	�
� ���A�$*

	conv_lossYQ=g�~        )��P	�9� ���A�$*

	conv_loss��L=��e�        )��P	�r� ���A�$*

	conv_loss�+�=���;        )��P	��� ���A�$*

	conv_loss�K�=����        )��P	�҈ ���A�$*

	conv_lossX×=����        )��P	t� ���A�$*

	conv_lossm=����        )��P	�8� ���A�$*

	conv_loss�c={���        )��P	�~� ���A�$*

	conv_loss�y�=|d��        )��P	֯� ���A�$*

	conv_loss=��=�]Τ        )��P	�� ���A�$*

	conv_loss���=F/�         )��P	�� ���A�$*

	conv_loss*N�=g<�        )��P	�S� ���A�$*

	conv_loss`��=����        )��P	8�� ���A�$*

	conv_losso�9=w�j        )��P	��� ���A�$*

	conv_lossA��=��|�        )��P	� ���A�$*

	conv_loss�-r=���        )��P	�� ���A�$*

	conv_loss�ߏ=�@�        )��P	SR� ���A�$*

	conv_loss.ao=WA�        )��P	N�� ���A�$*

	conv_loss��N=��9        )��P	Ƌ ���A�$*

	conv_loss8��=K���        )��P	:� ���A�$*

	conv_loss��Z=���.        )��P	G4� ���A�$*

	conv_loss���==���        )��P	�c� ���A�$*

	conv_loss�W=9M2r        )��P	x�� ���A�$*

	conv_loss��=)2�        )��P	�̌ ���A�$*

	conv_lossf�=̌        )��P	y�� ���A�$*

	conv_losss�Q=�/Ї        )��P	�/� ���A�$*

	conv_losspnf=+4�        )��P	�a� ���A�$*

	conv_loss�%~=2yٺ        )��P	��� ���A�$*

	conv_loss�!n=�y�        )��P	�ɍ ���A�$*

	conv_loss�>�+��        )��P	��� ���A�$*

	conv_loss��=�#a%        )��P	/� ���A�$*

	conv_loss]Y=b�m        )��P	�b� ���A�$*

	conv_lossj�=���5        )��P	ٞ� ���A�$*

	conv_loss�Y_=M�C�        )��P	O֎ ���A�$*

	conv_loss��=`X�        )��P	\	� ���A�$*

	conv_loss[S�=�'�        )��P	�8� ���A�$*

	conv_loss�=�zd        )��P	Ah� ���A�$*

	conv_loss2L�=�-�l        )��P	� ���A�$*

	conv_loss\�=���        )��P	vЏ ���A�$*

	conv_lossǛ�=2�9�        )��P	 � ���A�$*

	conv_loss�IT=UQKx        )��P	�0� ���A�$*

	conv_loss�H�=��W�        )��P	�`� ���A�$*

	conv_lossz�=�$,        )��P	9�� ���A�$*

	conv_loss�[=�?m�        )��P	�͐ ���A�$*

	conv_loss�<�=�=�,        )��P	��� ���A�$*

	conv_loss���=��h        )��P	�/� ���A�$*

	conv_lossL�=��X�        )��P	�a� ���A�$*

	conv_loss{�E=;�E�        )��P	��� ���A�$*

	conv_loss���=m��        )��P	Wґ ���A�$*

	conv_loss��=��&l        )��P	�� ���A�$*

	conv_loss��B=9#�F        )��P	?6� ���A�$*

	conv_loss1*�=dDPw        )��P	�h� ���A�$*

	conv_loss�Ȇ=p��         )��P	a�� ���A�$*

	conv_lossx¥=��b        )��P	ג ���A�$*

	conv_loss!��=��H�        )��P	a� ���A�$*

	conv_loss��=����        )��P	�<� ���A�$*

	conv_loss���=��        )��P	�r� ���A�$*

	conv_lossch=WU��        )��P	z�� ���A�$*

	conv_lossȻ�=�B�S        )��P	ߓ ���A�$*

	conv_loss�~�=`�;        )��P	,� ���A�$*

	conv_loss�{y=Pſ        )��P	A� ���A�$*

	conv_losse��=���        )��P	ׂ� ���A�$*

	conv_loss� �=�z�        )��P	\�� ���A�$*

	conv_loss��[=��6�        )��P		� ���A�$*

	conv_loss��R=b�3        )��P	�� ���A�$*

	conv_loss�.Z=?Xo"        )��P	�P� ���A�$*

	conv_loss�/�=[2i        )��P	��� ���A�$*

	conv_loss�{�=�]�        )��P	��� ���A�$*

	conv_loss�h}=�f�)        )��P	i�� ���A�$*

	conv_lossdЅ=3�F        )��P	�6� ���A�$*

	conv_lossC�q=�>        )��P	#i� ���A�$*

	conv_loss��n=��F}        )��P	{�� ���A�$*

	conv_loss���=ݖ �        )��P	$Ֆ ���A�$*

	conv_loss
T'=��<        )��P	� ���A�$*

	conv_loss�]w=�^d        )��P	�6� ���A�$*

	conv_lossܷl=��pd        )��P	3g� ���A�$*

	conv_loss#\v=�.1�        )��P	}�� ���A�$*

	conv_loss`��=��        )��P	�ɗ ���A�$*

	conv_loss��=-Q��        )��P	��� ���A�$*

	conv_loss�
�=��&�        )��P	�,� ���A�$*

	conv_loss�l�=��ސ        )��P	`c� ���A�$*

	conv_lossM͒=|a|2        )��P	��� ���A�$*

	conv_losslx�=�!�R        )��P	.Θ ���A�$*

	conv_loss��y=&���        )��P	��� ���A�$*

	conv_loss��~=ͱHX        )��P	w2� ���A�$*

	conv_lossZO�=��-T        )��P	2c� ���A�$*

	conv_loss�˄=z�;�        )��P	Ҕ� ���A�$*

	conv_loss��F=d�A        )��P	uǙ ���A�$*

	conv_loss؈U=����        )��P	b�� ���A�$*

	conv_loss�Re=����        )��P	.)� ���A�$*

	conv_loss�>n=�ger        )��P	�X� ���A�$*

	conv_loss��=�\�b        )��P	�� ���A�$*

	conv_loss/�p=jB�,        )��P	��� ���A�$*

	conv_loss5�y=wS�S        )��P	A� ���A�$*

	conv_loss��=KW��        )��P	� ���A�$*

	conv_loss�W�=���        )��P	-L� ���A�$*

	conv_loss��=B�        )��P	�|� ���A�%*

	conv_loss�u=�2�Z        )��P	�� ���A�%*

	conv_loss�~3=6���        )��P	+� ���A�%*

	conv_loss���=C�r�        )��P	�� ���A�%*

	conv_lossC�f=����        )��P	�H� ���A�%*

	conv_loss�q6=��k        )��P	�{� ���A�%*

	conv_loss>~=�7�_        )��P	6�� ���A�%*

	conv_loss"=^=�\G        )��P	�� ���A�%*

	conv_loss}�=�"`�        )��P	1� ���A�%*

	conv_loss��=�HI�        )��P	�Q� ���A�%*

	conv_lossc�V=Z_�         )��P	��� ���A�%*

	conv_loss��=&�:t        )��P	g�� ���A�%*

	conv_loss2��=�x�        )��P	�� ���A�%*

	conv_losspk�=\:h�        )��P	� ���A�%*

	conv_loss�ͫ=����        )��P	9F� ���A�%*

	conv_loss�6�=z���        )��P	3�� ���A�%*

	conv_lossj=��m�        )��P	J�� ���A�%*

	conv_loss;?E=�f&�        )��P	�� ���A�%*

	conv_lossH~='�W�        )��P	�� ���A�%*

	conv_lossre=���        )��P	�L� ���A�%*

	conv_loss0�X=1�        )��P	P~� ���A�%*

	conv_loss���= �L�        )��P	s�� ���A�%*

	conv_lossC�=< Tu        )��P	�� ���A�%*

	conv_loss��=Ik�        )��P	�-� ���A�%*

	conv_lossʲ=T�;�        )��P	�f� ���A�%*

	conv_loss=8�=� �        )��P	��� ���A�%*

	conv_loss�ԛ=+l�;        )��P	ʠ ���A�%*

	conv_lossF��=�6�%        )��P	}�� ���A�%*

	conv_lossL<�=���        )��P	A3� ���A�%*

	conv_loss\�|=Vk�        )��P	�c� ���A�%*

	conv_loss4�=�u@        )��P	͓� ���A�%*

	conv_lossǘb=bX�P        )��P	�ġ ���A�%*

	conv_losss�=�̙;        )��P	��� ���A�%*

	conv_lossF?�=�"e�        )��P	�'� ���A�%*

	conv_loss�g�=��p�        )��P	�Y� ���A�%*

	conv_loss@=�=!vH�        )��P	�� ���A�%*

	conv_loss=8p=�\��        )��P	ʢ ���A�%*

	conv_loss�0c=�z        )��P	0�� ���A�%*

	conv_loss#��=��#T        )��P	�/� ���A�%*

	conv_loss�u=s��        )��P	�_� ���A�%*

	conv_loss�M=�4        )��P	C�� ���A�%*

	conv_loss;}x=��        )��P	ʿ� ���A�%*

	conv_lossfu�=\��,        )��P	7� ���A�%*

	conv_loss�Y�=�W�O        )��P	(� ���A�%*

	conv_loss�)�=�@B�        )��P	BN� ���A�%*

	conv_lossK�=6���        )��P	=~� ���A�%*

	conv_lossX��=}�.�        )��P	V�� ���A�%*

	conv_loss�=ɂ��        )��P	�� ���A�%*

	conv_lossiD�=��@�        )��P	� ���A�%*

	conv_lossQ��=uPi�        )��P	�U� ���A�%*

	conv_loss&=x=��        )��P	-�� ���A�%*

	conv_loss/�z=���        )��P	#Х ���A�%*

	conv_loss/H�=\n�4        )��P	!� ���A�%*

	conv_loss�
�= ª�        )��P	�0� ���A�%*

	conv_loss":\=�p�d        )��P	�a� ���A�%*

	conv_loss�;�=b�S�        )��P	w�� ���A�%*

	conv_loss��=Cuz(        )��P	U˦ ���A�%*

	conv_loss��Z=;K_'        )��P	��� ���A�%*

	conv_loss�YV=c�g�        )��P	�,� ���A�%*

	conv_loss�=D��        )��P	�\� ���A�%*

	conv_loss�=�)0A        )��P	㒧 ���A�%*

	conv_lossI|�=uWQ~        )��P	�Ƨ ���A�%*

	conv_lossЗ7=o{�        )��P	z�� ���A�%*

	conv_loss���=}�         )��P	&+� ���A�%*

	conv_loss�!�=e��        )��P	�Z� ���A�%*

	conv_loss`S�=��g        )��P	�� ���A�%*

	conv_lossט=X-�k        )��P	�� ���A�%*

	conv_loss�1�=��M        )��P	~L� ���A�%*

	conv_loss�ɋ=�Y�        )��P	H}� ���A�%*

	conv_loss�dx=T~^        )��P	�� ���A�%*

	conv_lossE��=���        )��P	�� ���A�%*

	conv_loss�=̕��        )��P	a� ���A�%*

	conv_loss`�a=u�k        )��P	O� ���A�%*

	conv_loss��=ܧ��        )��P	s�� ���A�%*

	conv_lossBǆ=	���        )��P	ī ���A�%*

	conv_loss��r=\�        )��P	�� ���A�%*

	conv_losse��=o��        )��P	�'� ���A�%*

	conv_loss�j�=!p0I        )��P	QX� ���A�%*

	conv_loss��=�|�        )��P	Ջ� ���A�%*

	conv_lossѪi=ژ1I        )��P	�̬ ���A�%*

	conv_loss@,X=��.X        )��P	/� ���A�%*

	conv_loss,�N=1�hJ        )��P	�1� ���A�%*

	conv_loss�cw= ȶ        )��P	Ub� ���A�%*

	conv_loss�|=�C�        )��P	�� ���A�%*

	conv_loss��=, ��        )��P	�ĭ ���A�%*

	conv_lossn�= �Vu        )��P	��� ���A�%*

	conv_lossqAv=bOȎ        )��P	�%� ���A�%*

	conv_loss���=8JЧ        )��P	9U� ���A�%*

	conv_loss6\�=xv�        )��P	��� ���A�%*

	conv_lossM��=�K�`        )��P	r�� ���A�%*

	conv_losss�=.��j        )��P	�� ���A�%*

	conv_loss�^�=�        )��P	T"� ���A�%*

	conv_loss�Y�=ڒц        )��P	aV� ���A�%*

	conv_loss:r�=�
�        )��P	��� ���A�%*

	conv_lossnޘ=���        )��P	�� ���A�%*

	conv_loss���=���f        )��P	`� ���A�%*

	conv_loss"N�=��N        )��P	�� ���A�%*

	conv_loss[m+="��        )��P	�I� ���A�%*

	conv_loss�R�=�V�        )��P	{� ���A�%*

	conv_lossG/=��f<        )��P	��� ���A�%*

	conv_loss��=x��        )��P	�� ���A�%*

	conv_loss׃`=��D        )��P	� ���A�%*

	conv_loss�@�=�f��        )��P	J� ���A�%*

	conv_lossF�5=�L�/        )��P	X{� ���A�%*

	conv_loss�)]=q8 �        )��P	 �� ���A�%*

	conv_loss��x=���A        )��P	�۱ ���A�%*

	conv_loss�Et=b���        )��P	n� ���A�%*

	conv_loss�s�=��S        )��P	�=� ���A�%*

	conv_lossD5�=�M�{        )��P	�o� ���A�%*

	conv_lossh@F=���
        )��P	�� ���A�%*

	conv_loss��e=��        )��P	�Ӳ ���A�%*

	conv_loss�=O$F}        )��P	�� ���A�%*

	conv_loss�As=G>ȟ        )��P	D� ���A�%*

	conv_loss�=�� �        )��P	�x� ���A�%*

	conv_lossZƕ=�%D�        )��P	a�� ���A�%*

	conv_loss�C�=*D�        )��P	m� ���A�%*

	conv_loss��_=�u��        )��P	�L� ���A�%*

	conv_loss�_=^��        )��P	҄� ���A�%*

	conv_loss5Å=?=�        )��P	��� ���A�%*

	conv_lossA �=3���        )��P	`�� ���A�%*

	conv_lossbޖ=0S�        )��P	�.� ���A�%*

	conv_loss=Mb=н        )��P	�_� ���A�%*

	conv_loss�=�=E�4�        )��P	W�� ���A�%*

	conv_loss��p=�J��        )��P	�ҵ ���A�%*

	conv_loss�>�=����        )��P	p� ���A�%*

	conv_loss�rg=���t        )��P	�I� ���A�%*

	conv_lossI=��l        )��P	��� ���A�%*

	conv_lossO��=Ǥ�        )��P	�� ���A�%*

	conv_loss��V=���        )��P	��� ���A�&*

	conv_loss�:�=�=�/        )��P	;3� ���A�&*

	conv_lossK �=��d�        )��P	\c� ���A�&*

	conv_lossq#�=�.�l        )��P	ϑ� ���A�&*

	conv_lossaB=�[#        )��P	��� ���A�&*

	conv_loss~�`=B&?        )��P	]�� ���A�&*

	conv_loss詖=
���        )��P	#;� ���A�&*

	conv_loss�_=���        )��P	�h� ���A�&*

	conv_lossN��=E9        )��P	c�� ���A�&*

	conv_loss�z�=��r        )��P	�Ƹ ���A�&*

	conv_lossJ��=�]l        )��P	��� ���A�&*

	conv_loss*��=�b�        )��P	)� ���A�&*

	conv_loss�=Xt��        )��P	�a� ���A�&*

	conv_loss��~=�ʽ�        )��P	�� ���A�&*

	conv_lossI�r=�X/=        )��P	g͹ ���A�&*

	conv_loss��~=�0�        )��P	�� ���A�&*

	conv_loss=�n=�]B�        )��P	-� ���A�&*

	conv_loss�=I=$޵        )��P	vq� ���A�&*

	conv_losssu=����        )��P	]�� ���A�&*

	conv_loss� z=ͅ��        )��P	H� ���A�&*

	conv_lossO3�=��1*        )��P	�� ���A�&*

	conv_loss�7�=�[ݚ        )��P	}N� ���A�&*

	conv_loss5n�=�oT        )��P	9�� ���A�&*

	conv_loss��=�B�        )��P	��� ���A�&*

	conv_loss�Q�=f���        )��P	D�� ���A�&*

	conv_loss�=%o��        )��P	&/� ���A�&*

	conv_loss̘�=��s�        )��P	�^� ���A�&*

	conv_loss{=�"�x        )��P	��� ���A�&*

	conv_loss
1|=߃5�        )��P	�ȼ ���A�&*

	conv_loss��R=��h        )��P	-�� ���A�&*

	conv_loss:�=r}��        )��P	�&� ���A�&*

	conv_loss0�s=Fk�        )��P	�X� ���A�&*

	conv_lossߣ]=�ƞ�        )��P	�� ���A�&*

	conv_lossx�=�/��        )��P	}�� ���A�&*

	conv_loss<��=y�{w        )��P	�� ���A�&*

	conv_lossDY=�T2�        )��P	U� ���A�&*

	conv_loss(ܜ=��6        )��P	�O� ���A�&*

	conv_loss�R=1}�        )��P	��� ���A�&*

	conv_loss��=$�:�        )��P		ɾ ���A�&*

	conv_lossm=��ɂ        )��P	��� ���A�&*

	conv_lossF�o=Nُ        )��P	~/� ���A�&*

	conv_losstL==�
�}        )��P	�h� ���A�&*

	conv_loss��{=w!�        )��P	ɟ� ���A�&*

	conv_loss:z�=s�Lz        )��P	!޿ ���A�&*

	conv_loss���=w��        )��P	b� ���A�&*

	conv_loss�<�=�        )��P		S� ���A�&*

	conv_lossD0_=�L        )��P	2�� ���A�&*

	conv_loss�p�='j�        )��P	-�� ���A�&*

	conv_loss�5�=H���        )��P	f� ���A�&*

	conv_lossM�s=�]�        )��P	r5� ���A�&*

	conv_loss�4�=�7�
        )��P	%h� ���A�&*

	conv_loss��K=V2�        )��P	��� ���A�&*

	conv_losseQ�=8        )��P	T�� ���A�&*

	conv_loss���=��        )��P	�� ���A�&*

	conv_loss�.h=��2        )��P	:� ���A�&*

	conv_loss�6w=���        )��P	k� ���A�&*

	conv_loss�/�=�?,h        )��P	��� ���A�&*

	conv_loss��=�o        )��P	��� ���A�&*

	conv_loss���=�KV�        )��P	�� ���A�&*

	conv_lossA�=X��        )��P	I� ���A�&*

	conv_lossTsl=�m��        )��P	��� ���A�&*

	conv_loss�r=�4�B        )��P	��� ���A�&*

	conv_loss��=��K        )��P	��� ���A�&*

	conv_loss`b=���        )��P	V� ���A�&*

	conv_losswz=�p        )��P	�O� ���A�&*

	conv_loss�9�=.�j        )��P	��� ���A�&*

	conv_lossJ�w=s��        )��P	o�� ���A�&*

	conv_loss��=�!�        )��P	<�� ���A�&*

	conv_loss
o=2m�        )��P	�"� ���A�&*

	conv_loss�Z�=��9        )��P	.]� ���A�&*

	conv_lossR�=D\��        )��P	ˏ� ���A�&*

	conv_loss�ō=X�mJ        )��P	&�� ���A�&*

	conv_losshj�=��H        )��P	�� ���A�&*

	conv_loss���=��N        )��P	�8� ���A�&*

	conv_loss=Ȕ=���        )��P	=j� ���A�&*

	conv_loss�Ҋ=��h        )��P	��� ���A�&*

	conv_loss�P�=y��<        )��P	W�� ���A�&*

	conv_lossM;�=D��=        )��P	�� ���A�&*

	conv_loss~�j=�#��        )��P	,>� ���A�&*

	conv_loss)�R=�;��        )��P	xq� ���A�&*

	conv_loss٩~=�E�        )��P	˱� ���A�&*

	conv_lossV��=]&�+        )��P	$�� ���A�&*

	conv_loss���=I C        )��P	�� ���A�&*

	conv_loss�k�=��8        )��P	{K� ���A�&*

	conv_lossÿZ=W6P�        )��P	/�� ���A�&*

	conv_loss���=f��E        )��P	:�� ���A�&*

	conv_loss�H=��lD        )��P	�� ���A�&*

	conv_loss+L=p�Ou        )��P	�0� ���A�&*

	conv_loss�f=Al��        )��P	J`� ���A�&*

	conv_loss��Z=t���        )��P	$�� ���A�&*

	conv_lossl�G=        )��P	��� ���A�&*

	conv_loss�(�=m'�        )��P	a�� ���A�&*

	conv_loss�^�=�l�        )��P	p"� ���A�&*

	conv_lossҘ�=��Ɣ        )��P	!Q� ���A�&*

	conv_loss��o=._uw        )��P	Â� ���A�&*

	conv_loss$O=�Ov        )��P	Ĺ� ���A�&*

	conv_loss}T�= �p4        )��P	��� ���A�&*

	conv_loss�=,�        )��P	�7� ���A�&*

	conv_loss�O&=�(5�        )��P	$l� ���A�&*

	conv_loss��Z=/���        )��P	ٟ� ���A�&*

	conv_loss��B=���        )��P	�� ���A�&*

	conv_loss���=1�s�        )��P	��� ���A�&*

	conv_loss��l=���e        )��P	g0� ���A�&*

	conv_loss`V�=K�I"        )��P	1f� ���A�&*

	conv_loss���=���        )��P	w�� ���A�&*

	conv_losst�P=�Z'.        )��P	)�� ���A�&*

	conv_loss��=�]��        )��P	��� ���A�&*

	conv_loss=�=J��        )��P	�#� ���A�&*

	conv_loss�d�=Pu*        )��P	Gn� ���A�&*

	conv_loss�O=�I^        )��P	 �� ���A�&*

	conv_loss1%y=�x�$        )��P	��� ���A�&*

	conv_loss��=m��        )��P	�� ���A�&*

	conv_lossт=.��P        )��P	J� ���A�&*

	conv_loss^�=���        )��P	�|� ���A�&*

	conv_loss�*�=`�l~        )��P	��� ���A�&*

	conv_lossK�=&�U        )��P	H�� ���A�&*

	conv_loss��D=pS�2        )��P	(
� ���A�&*

	conv_loss�v=���        )��P	�;� ���A�&*

	conv_loss��=I3�        )��P	�j� ���A�&*

	conv_loss7�t=�f�@        )��P	0�� ���A�&*

	conv_loss��W=eG        )��P	��� ���A�&*

	conv_loss��|=��        )��P	��� ���A�&*

	conv_loss�R�=as�o        )��P	60� ���A�&*

	conv_loss�Ӈ=�LiI        )��P	�j� ���A�&*

	conv_loss)S:=ȅA�        )��P	ѝ� ���A�&*

	conv_loss~ R="�m        )��P	��� ���A�&*

	conv_lossf�=���        )��P	|�� ���A�&*

	conv_lossRذ=��        )��P	�-� ���A�&*

	conv_lossb_=L?��        )��P	 h� ���A�'*

	conv_loss2do=� 9        )��P	�� ���A�'*

	conv_loss�)N=����        )��P	��� ���A�'*

	conv_loss��e=���        )��P	��� ���A�'*

	conv_loss,�\=�i�        )��P	7)� ���A�'*

	conv_loss`�=�G        )��P	�c� ���A�'*

	conv_loss*�Y=��k�        )��P	Ӓ� ���A�'*

	conv_loss��=���        )��P	��� ���A�'*

	conv_loss��=3K�:        )��P	� ���A�'*

	conv_loss�χ=c�        )��P	~�� ���A�'*

	conv_lossK�b=.�        )��P	��� ���A�'*

	conv_loss7�c=*���        )��P	v� ���A�'*

	conv_lossZ�=C
7�        )��P	M>� ���A�'*

	conv_loss֎E=�6��        )��P	�o� ���A�'*

	conv_lossZ��=Ē&        )��P	�� ���A�'*

	conv_lossK͉=� �        )��P	��� ���A�'*

	conv_loss�˱=�1�        )��P	*�� ���A�'*

	conv_lossX/�=�'        )��P	F3� ���A�'*

	conv_loss�Ű=m��Q        )��P	Sp� ���A�'*

	conv_loss�w�=&>�E        )��P	��� ���A�'*

	conv_loss�V=s~�        )��P	|�� ���A�'*

	conv_loss牵=hU�!        )��P	>� ���A�'*

	conv_loss�=}=5        )��P	5B� ���A�'*

	conv_loss�D=��U�        )��P	�p� ���A�'*

	conv_losss2=ɀ��        )��P	F�� ���A�'*

	conv_loss˦�=(+�o        )��P	�� ���A�'*

	conv_loss�j=����        )��P	�� ���A�'*

	conv_loss��f=RT}�        )��P	�:� ���A�'*

	conv_loss���=:'Q�        )��P	�k� ���A�'*

	conv_loss�wP=T��        )��P	�� ���A�'*

	conv_loss!֋=�~��        )��P	��� ���A�'*

	conv_lossW``=�K��        )��P	� ���A�'*

	conv_loss��_=jA�        )��P	.2� ���A�'*

	conv_loss��=����        )��P	kj� ���A�'*

	conv_loss)�l=-��:        )��P	^�� ���A�'*

	conv_loss�/m=�r�         )��P	��� ���A�'*

	conv_loss���=���9        )��P	[� ���A�'*

	conv_loss:�=A�{�        )��P		@� ���A�'*

	conv_loss���=lZdv        )��P	Qr� ���A�'*

	conv_loss{�=�mk�        )��P	ߢ� ���A�'*

	conv_loss��A=���        )��P	��� ���A�'*

	conv_loss��=na         )��P	O� ���A�'*

	conv_loss�=�)%�        )��P	�H� ���A�'*

	conv_loss>ʔ=�P�        )��P	�� ���A�'*

	conv_loss��,=�#�(        )��P	w�� ���A�'*

	conv_loss��q=n��        )��P	��� ���A�'*

	conv_loss�]�=yg��        )��P	"� ���A�'*

	conv_loss��Q=2��        )��P	uA� ���A�'*

	conv_loss�g�=�l'        )��P	0q� ���A�'*

	conv_loss�xX=��K�        )��P	��� ���A�'*

	conv_loss��=��v        )��P	�� ���A�'*

	conv_loss	͵={!        )��P	�� ���A�'*

	conv_lossU�u=�3�T        )��P	u3� ���A�'*

	conv_loss^3==e�        )��P	pe� ���A�'*

	conv_loss�g0=*��        )��P	�� ���A�'*

	conv_lossO?=6g�        )��P	�� ���A�'*

	conv_loss'ڂ=�c��        )��P	P�� ���A�'*

	conv_loss�W�=���I        )��P	�(� ���A�'*

	conv_loss�g�=���h        )��P	�X� ���A�'*

	conv_lossm�=2D��        )��P	��� ���A�'*

	conv_loss�X�=L���        )��P	f�� ���A�'*

	conv_loss��}=���        )��P	��� ���A�'*

	conv_loss"�b=�;0�        )��P	�.� ���A�'*

	conv_loss�=�p��        )��P	�`� ���A�'*

	conv_loss���="cm        )��P	l�� ���A�'*

	conv_lossѡ{= Aُ        )��P	��� ���A�'*

	conv_loss�lE=��        )��P	p�� ���A�'*

	conv_loss�]=����        )��P	$� ���A�'*

	conv_loss��=۴&8        )��P	*b� ���A�'*

	conv_loss��~=��+        )��P	�� ���A�'*

	conv_lossb �=�v�        )��P	��� ���A�'*

	conv_loss�і=);q[        )��P	Q�� ���A�'*

	conv_loss��=��O�        )��P	�,� ���A�'*

	conv_loss��=xj�        )��P	_� ���A�'*

	conv_lossx��=����        )��P	�� ���A�'*

	conv_loss�`�=�"�A        )��P	�� ���A�'*

	conv_loss�>�=a��        )��P	��� ���A�'*

	conv_loss Z=�\��        )��P	�)� ���A�'*

	conv_loss� h=4m�^        )��P	�W� ���A�'*

	conv_loss�)g=�8�'        )��P	*�� ���A�'*

	conv_loss5PY=��V�        )��P	��� ���A�'*

	conv_loss��j="���        )��P	O�� ���A�'*

	conv_lossmԉ=-֦        )��P	�2� ���A�'*

	conv_lossei�=_ڼ        )��P	�b� ���A�'*

	conv_loss	&g=!��x        )��P	��� ���A�'*

	conv_lossݧ�=�ϻ        )��P	��� ���A�'*

	conv_lossC�=W�޵        )��P	��� ���A�'*

	conv_loss��=˰�a        )��P	�!� ���A�'*

	conv_lossj�W=��q        )��P	�P� ���A�'*

	conv_loss_Vi=�.8        )��P	�� ���A�'*

	conv_loss�v�=v+Yq        )��P	ݭ� ���A�'*

	conv_lossO9=��"o        )��P	��� ���A�'*

	conv_lossy0�= �$�        )��P	�� ���A�'*

	conv_loss�!=S�'        )��P	�A� ���A�'*

	conv_lossc��=��M        )��P	�|� ���A�'*

	conv_loss�2�=��.�        )��P	�� ���A�'*

	conv_loss:��=�"        )��P	0�� ���A�'*

	conv_loss�̔=<e�        )��P	�� ���A�'*

	conv_loss��=�B�K        )��P	0H� ���A�'*

	conv_loss⇸=�i��        )��P	Kw� ���A�'*

	conv_loss���=�^�        )��P	��� ���A�'*

	conv_loss�qY=U�        )��P	��� ���A�'*

	conv_loss	΀=G���        )��P	i� ���A�'*

	conv_loss�x�=�p��        )��P	S;� ���A�'*

	conv_loss��=�e        )��P	^q� ���A�'*

	conv_loss�c�=3�U�        )��P	��� ���A�'*

	conv_loss4�I=�f8&        )��P	��� ���A�'*

	conv_loss���=���        )��P	�� ���A�'*

	conv_loss���=�:        )��P	'<� ���A�'*

	conv_loss�t=��T�        )��P	$}� ���A�'*

	conv_losso՟=��        )��P	p�� ���A�'*

	conv_loss�S4=[�t�        )��P	Z�� ���A�'*

	conv_loss\ٍ=�ث�        )��P	�� ���A�'*

	conv_lossBG�=�q�
        )��P	|A� ���A�'*

	conv_loss�	z=���n        )��P	H}� ���A�'*

	conv_lossZ�=JF�
        )��P	�� ���A�'*

	conv_loss� �=z��P        )��P	��� ���A�'*

	conv_loss?��=xi�+        )��P	#	� ���A�'*

	conv_loss�Sk=�1��        )��P	�?� ���A�'*

	conv_loss�p�=)kw�        )��P	��� ���A�'*

	conv_lossr��=�AU�        )��P	�� ���A�'*

	conv_loss�Ό=ci)m        )��P	�� ���A�'*

	conv_loss=>=VJ�        )��P	�� ���A�'*

	conv_lossʑ=kuB�        )��P	fM� ���A�'*

	conv_losspȀ=�DJ�        )��P	ֈ� ���A�'*

	conv_loss���=j��        )��P	'�� ���A�'*

	conv_loss�n�=]L�g        )��P	y�� ���A�'*

	conv_loss�J=U���        )��P	� ���A�'*

	conv_loss��=WH��        )��P	�H� ���A�(*

	conv_losshC�=�p        )��P	�z� ���A�(*

	conv_loss��=��Ǐ        )��P	�� ���A�(*

	conv_loss`#u=���D        )��P	��� ���A�(*

	conv_lossV�-=��T�        )��P	r� ���A�(*

	conv_lossF��=�ނ�        )��P	�O� ���A�(*

	conv_lossG
�=0F��        )��P	y�� ���A�(*

	conv_loss���=���r        )��P	��� ���A�(*

	conv_loss-�X=�8�        )��P	��� ���A�(*

	conv_loss�e�="��        )��P	6$� ���A�(*

	conv_loss� <=��y�        )��P	V� ���A�(*

	conv_loss ��=�n��        )��P	��� ���A�(*

	conv_lossD�9=�T        )��P	��� ���A�(*

	conv_loss�=PO~        )��P	��� ���A�(*

	conv_lossn�q=Z^8O        )��P	�� ���A�(*

	conv_loss�W~=�,�(        )��P	EL� ���A�(*

	conv_lossޓ=��#�        )��P	`�� ���A�(*

	conv_loss(�=B��Q        )��P	��� ���A�(*

	conv_lossm�=��        )��P	��� ���A�(*

	conv_loss7�=-|��        )��P	�"� ���A�(*

	conv_loss&ӏ=�I�@        )��P	�U� ���A�(*

	conv_loss�B}=�P        )��P	Ƈ� ���A�(*

	conv_loss�\=��д        )��P	��� ���A�(*

	conv_loss(s~==���        )��P	��� ���A�(*

	conv_loss���=�iټ        )��P	�� ���A�(*

	conv_loss�Q�=�k:�        )��P	eE� ���A�(*

	conv_loss{��=`�й        )��P	�x� ���A�(*

	conv_loss\I�=����        )��P	��� ���A�(*

	conv_loss��L=����        )��P	�� ���A�(*

	conv_loss[�i=�B�X        )��P	�� ���A�(*

	conv_loss�Ճ=x�Ok        )��P	J� ���A�(*

	conv_loss�=�{�%        )��P	��� ���A�(*

	conv_lossj*u=�_�        )��P	��� ���A�(*

	conv_loss�֚=�ح         )��P	�� ���A�(*

	conv_loss�F�=��        )��P	s7� ���A�(*

	conv_loss�5
=5���        )��P	xh� ���A�(*

	conv_loss@A=1+��        )��P	K�� ���A�(*

	conv_lossN�G=ތ-         )��P	�� ���A�(*

	conv_loss�L=N��G        )��P	-� ���A�(*

	conv_loss�DM=�LI�        )��P	j:� ���A�(*

	conv_loss�j~=J�!        )��P	�q� ���A�(*

	conv_loss�&�=K��        )��P	��� ���A�(*

	conv_loss���=�|��        )��P	��� ���A�(*

	conv_lossk\=��͙        )��P	�� ���A�(*

	conv_loss�Z=��.        )��P	�I� ���A�(*

	conv_losss�f=K�M        )��P	y� ���A�(*

	conv_loss�`=L�X        )��P	}�� ���A�(*

	conv_loss�.=b��        )��P	��� ���A�(*

	conv_loss/�e=���        )��P	� ���A�(*

	conv_loss��=�n�        )��P	�B� ���A�(*

	conv_losse��= B0H        )��P	ev� ���A�(*

	conv_loss��:=�O:        )��P	�� ���A�(*

	conv_loss���=�8�        )��P	��� ���A�(*

	conv_loss\hE=h��        )��P	w� ���A�(*

	conv_loss�;J=S���        )��P	�H� ���A�(*

	conv_loss�q�=��        )��P	!y� ���A�(*

	conv_lossG�d=�r        )��P	�� ���A�(*

	conv_loss�q=��         )��P	�� ���A�(*

	conv_lossp��=��O        )��P	8� ���A�(*

	conv_loss,�=����        )��P	j5� ���A�(*

	conv_loss
_l=8i�        )��P	�d� ���A�(*

	conv_losst��=�3Ɍ        )��P	>�� ���A�(*

	conv_loss���=�-�        )��P	��� ���A�(*

	conv_loss���=��oE        )��P	��� ���A�(*

	conv_lossTy�=�ڐq        )��P	�'� ���A�(*

	conv_loss�bu=#�~        )��P	V� ���A�(*

	conv_loss>f=����        )��P	��� ���A�(*

	conv_loss�>r=湳�        )��P	x�� ���A�(*

	conv_loss�y�=�)�        )��P	�� ���A�(*

	conv_lossyM�=�c��        )��P	�� ���A�(*

	conv_loss)�7=����        )��P	D@� ���A�(*

	conv_loss=F�=�*T        )��P	"o� ���A�(*

	conv_loss�'a=�ߎ        )��P	L�� ���A�(*

	conv_loss _�=��k        )��P	��� ���A�(*

	conv_loss�F�=<H�7        )��P	��� ���A�(*

	conv_lossy��=���y        )��P	w.� ���A�(*

	conv_loss�f�=6�x>        )��P	^� ���A�(*

	conv_loss��h=�-��        )��P	]�� ���A�(*

	conv_loss�f=Ƕ��        )��P	{�� ���A�(*

	conv_loss���=�|�        )��P	��� ���A�(*

	conv_loss���=p���        )��P	k! !���A�(*

	conv_loss5'#=©�|        )��P	*�!���A�(*

	conv_loss�,Y=2�        )��P	R�!���A�(*

	conv_lossL��=�]{        )��P	R!���A�(*

	conv_loss^1�=�0��        )��P	>P!���A�(*

	conv_lossm�= !�        )��P	��!���A�(*

	conv_loss�=�=���        )��P	��!���A�(*

	conv_loss�3�={���        )��P	e�!���A�(*

	conv_lossa�=�Ex�        )��P	]!���A�(*

	conv_lossry=Ҟ        )��P	�[!���A�(*

	conv_loss��y=MR��        )��P	�!���A�(*

	conv_loss�=�AS�        )��P	�!���A�(*

	conv_lossar�=V��w        )��P	��!���A�(*

	conv_loss׭R=��        )��P	�&!���A�(*

	conv_loss��@=~���        )��P	Mf!���A�(*

	conv_loss1�:=RX��        )��P	�!���A�(*

	conv_loss뷸=k�        )��P	��!���A�(*

	conv_loss /�=��a�        )��P	�!!���A�(*

	conv_loss#;=f¾        )��P	�a!���A�(*

	conv_loss�-]=���        )��P	,�!���A�(*

	conv_loss(��=z7T!        )��P	��!���A�(*

	conv_lossjg=X��S        )��P	.�!���A�(*

	conv_lossj4q=C��,        )��P	X+!���A�(*

	conv_loss�`�=�KP        )��P	�]!���A�(*

	conv_loss r=� O        )��P	��!���A�(*

	conv_loss�o=ѵ�T        )��P	>�!���A�(*

	conv_loss8�Y=�2s        )��P	��!���A�(*

	conv_loss���=몄        )��P	�&!���A�(*

	conv_lossc4==R��        )��P	�Y!���A�(*

	conv_loss�I=��se        )��P	��!���A�(*

	conv_loss��z=�zMd        )��P	��!���A�(*

	conv_loss�+=c%gR        )��P	z!���A�(*

	conv_lossg\=W��        )��P	�9!���A�(*

	conv_loss��z=����        )��P	�l!���A�(*

	conv_loss���=Ǔ��        )��P	(�!���A�(*

	conv_lossV��=��U�        )��P	��!���A�(*

	conv_loss=͑=��        )��P	i	!���A�(*

	conv_loss�^�=���        )��P	�7	!���A�(*

	conv_loss�El=׮~�        )��P	�h	!���A�(*

	conv_loss�ؙ=4�^�        )��P	�	!���A�(*

	conv_loss��=8%##        )��P	R�	!���A�(*

	conv_lossh�R=���        )��P	j 
!���A�(*

	conv_loss��F=�8�h        )��P	;3
!���A�(*

	conv_loss(Iq=��        )��P	vf
!���A�(*

	conv_loss�V}=ZYw        )��P	ԗ
!���A�(*

	conv_loss~J�=�T�        )��P	�
!���A�(*

	conv_loss�lU=E�&G        )��P	��
!���A�(*

	conv_loss���=wu��        )��P	�,!���A�(*

	conv_lossT+\=O��s        )��P	~^!���A�)*

	conv_losso?K=U̟        )��P	ȏ!���A�)*

	conv_loss"d�=�0 6        )��P	��!���A�)*

	conv_losseM�=ф��        )��P	�!���A�)*

	conv_lossNЩ=;�n        )��P	�9!���A�)*

	conv_loss;L=�hq�        )��P	�k!���A�)*

	conv_lossN@Z=�Լ/        )��P	(�!���A�)*

	conv_loss�̌=�^��        )��P	��!���A�)*

	conv_lossQ�X=�`w�        )��P	!���A�)*

	conv_loss�?m=��[�        )��P	�8!���A�)*

	conv_loss�b�=�S�        )��P	Wk!���A�)*

	conv_lossX��=s���        )��P	h�!���A�)*

	conv_loss�)y=���        )��P	��!���A�)*

	conv_loss���=��9r        )��P	�!���A�)*

	conv_loss�'�=�        )��P	�F!���A�)*

	conv_lossXؓ=�A��        )��P	,x!���A�)*

	conv_lossg|=���
        )��P	�!���A�)*

	conv_loss�d=]*YM        )��P	��!���A�)*

	conv_loss	�=�v�I        )��P	I!���A�)*

	conv_loss��k=F��        )��P	�N!���A�)*

	conv_lossZUO=~���        )��P	�!���A�)*

	conv_lossȡ�=@<��        )��P	��!���A�)*

	conv_loss:(�=�5�f        )��P	u�!���A�)*

	conv_loss�Q�=^00?        )��P	$!���A�)*

	conv_lossg}V=s���        )��P	�V!���A�)*

	conv_loss�|a=�R�        )��P	��!���A�)*

	conv_loss=v�=�C�        )��P	�!���A�)*

	conv_loss6Y=��F        )��P	��!���A�)*

	conv_lossg�)=�?��        )��P	r#!���A�)*

	conv_loss�)�=ѡ��        )��P	ZV!���A�)*

	conv_loss��e=mq��        )��P	�!���A�)*

	conv_lossJBn=��        )��P	��!���A�)*

	conv_loss�32=\+f        )��P	��!���A�)*

	conv_loss��F=��{        )��P	�!���A�)*

	conv_loss��u=�TF"        )��P	�P!���A�)*

	conv_loss%^f=t%q        )��P	d�!���A�)*

	conv_losscG�=;|��        )��P	t�!���A�)*

	conv_loss~�V= �        )��P	��!���A�)*

	conv_loss�ߤ=e;e        )��P	6!���A�)*

	conv_loss��=,���        )��P	K!���A�)*

	conv_loss۳x=�        )��P	�|!���A�)*

	conv_lossx8�=Y��        )��P	��!���A�)*

	conv_loss^�=|�"        )��P	��!���A�)*

	conv_loss�q=?+މ        )��P	�!���A�)*

	conv_loss=�        )��P	�C!���A�)*

	conv_lossK=r�+@        )��P	4u!���A�)*

	conv_loss�[�=Y�=k        )��P	�!���A�)*

	conv_lossG�c=�'        )��P	B�!���A�)*

	conv_loss��S=η�n        )��P	�!���A�)*

	conv_loss�=����        )��P	;@!���A�)*

	conv_lossw�[=�I47        )��P	�r!���A�)*

	conv_loss�.=�/��        )��P	��!���A�)*

	conv_loss�Kg=�<U�        )��P	��!���A�)*

	conv_lossj�=��Il        )��P	D!���A�)*

	conv_loss�j�=샟9        )��P	oL!���A�)*

	conv_loss_2i=��v�        )��P	��!���A�)*

	conv_lossg�H=���N        )��P	�!���A�)*

	conv_loss�>�=�G��        )��P	��!���A�)*

	conv_loss��/=�W        )��P	�!���A�)*

	conv_loss��@=o{��        )��P	�N!���A�)*

	conv_loss̝s=�e��        )��P	�!���A�)*

	conv_lossT�=fK��        )��P	�!���A�)*

	conv_lossW?n=&��        )��P	F�!���A�)*

	conv_loss#�A=];��        )��P	O)!���A�)*

	conv_lossfP�=\�@        )��P	�[!���A�)*

	conv_lossXK�=8ѐ�        )��P	M�!���A�)*

	conv_loss��I=oX��        )��P	
�!���A�)*

	conv_loss"�D=s�&        )��P	��!���A�)*

	conv_loss�d=���=        )��P	 1!���A�)*

	conv_loss���=��^�        )��P	jg!���A�)*

	conv_lossS��=?��        )��P	�!���A�)*

	conv_loss�x�=�� _        )��P	��!���A�)*

	conv_loss�QH=^[G        )��P	��!���A�)*

	conv_loss!%B=��        )��P	�-!���A�)*

	conv_loss�N�=o�E*        )��P	a!���A�)*

	conv_lossS�m=���        )��P	=�!���A�)*

	conv_lossK��=ş}M        )��P	(�!���A�)*

	conv_loss�Y�=ZX֝        )��P	��!���A�)*

	conv_loss1?�=H�*        )��P	�#!���A�)*

	conv_loss͖�=���m        )��P	�T!���A�)*

	conv_loss0�=���        )��P	4�!���A�)*

	conv_loss
ӏ=3^}        )��P	#�!���A�)*

	conv_loss)߈=�і        )��P	��!���A�)*

	conv_losssQ�=��        )��P	u!���A�)*

	conv_loss][=iM�L        )��P	XK!���A�)*

	conv_loss���=d�        )��P	�}!���A�)*

	conv_loss/�>=6+r�        )��P	�!���A�)*

	conv_lossSw=���        )��P	��!���A�)*

	conv_loss�6C=�Ŀ�        )��P	�!���A�)*

	conv_loss쒕=����        )��P	�C!���A�)*

	conv_loss0��=k�y        )��P	�s!���A�)*

	conv_loss��=���        )��P	Q�!���A�)*

	conv_loss��=�u�|        )��P	�!���A�)*

	conv_loss���=��wD        )��P	�!���A�)*

	conv_loss�kf=j ��        )��P	XJ!���A�)*

	conv_loss��<=P���        )��P	2�!���A�)*

	conv_loss�"�=�wF        )��P	C�!���A�)*

	conv_lossr�=%Z��        )��P	��!���A�)*

	conv_loss�f=/�G�        )��P	�,!���A�)*

	conv_loss�Ծ=�oj        )��P	|^!���A�)*

	conv_lossl�=�y�l        )��P	��!���A�)*

	conv_loss�Ś=���%        )��P	{�!���A�)*

	conv_loss��=��L        )��P	_�!���A�)*

	conv_loss�=���        )��P	�6 !���A�)*

	conv_loss��U=��\b        )��P	?i !���A�)*

	conv_loss�v=����        )��P	� !���A�)*

	conv_loss�(]=��        )��P	�� !���A�)*

	conv_loss��^=.t��        )��P	�!!���A�)*

	conv_losstۅ=�ql        )��P	&?!!���A�)*

	conv_loss�Xh=ک        )��P	hp!!���A�)*

	conv_lossM��=��q        )��P	��!!���A�)*

	conv_lossX�=�mY        )��P	q�!!���A�)*

	conv_loss6�q=s7�        )��P	�"!���A�)*

	conv_loss)s=���C        )��P	�J"!���A�)*

	conv_losslJ�=���#        )��P	r}"!���A�)*

	conv_loss:ދ=a�c	        )��P	��"!���A�)*

	conv_loss��=iv=        )��P	Y�"!���A�)*

	conv_lossm4�=q�g        )��P	^"#!���A�)*

	conv_loss�qG=�d�        )��P	x^#!���A�)*

	conv_loss=r�=_?��        )��P	Ӕ#!���A�)*

	conv_lossp�w=���        )��P	��#!���A�)*

	conv_loss㪭=��H        )��P	��#!���A�)*

	conv_loss��h=��.?        )��P	)$!���A�)*

	conv_loss�=N��        )��P	c$!���A�)*

	conv_loss{K�=��>}        )��P	q�$!���A�)*

	conv_loss�Y==CbG        )��P	��$!���A�)*

	conv_loss��n=�A�        )��P	G�$!���A�)*

	conv_loss��P=^�        )��P	�9%!���A�)*

	conv_loss��t=���.        )��P	Hl%!���A�**

	conv_lossc=k]�        )��P	i�%!���A�**

	conv_loss� k=x�="        )��P	b�%!���A�**

	conv_loss�e�=�H�        )��P	&!���A�**

	conv_loss���=#��Y        )��P	�?&!���A�**

	conv_lossuXZ=���        )��P	�t&!���A�**

	conv_loss4�O=88��        )��P	P�&!���A�**

	conv_loss6h=�ƌ        )��P	��&!���A�**

	conv_loss��U=2�        )��P	*
'!���A�**

	conv_loss��g=)�m�        )��P	�H'!���A�**

	conv_lossii^=���P        )��P	�z'!���A�**

	conv_lossu�=%�u�        )��P	��'!���A�**

	conv_loss-�3=c��\        )��P	��'!���A�**

	conv_loss��#=�N>�        )��P	(!���A�**

	conv_loss��Y=ڧ@<        )��P	�I(!���A�**

	conv_loss>��=F_��        )��P	��(!���A�**

	conv_loss�W^=g�B        )��P	Ϲ(!���A�**

	conv_loss�f�=��        )��P	��(!���A�**

	conv_loss	��=TY�        )��P	�)!���A�**

	conv_losso�x=���        )��P	�Q)!���A�**

	conv_loss��
>��m�        )��P	̏)!���A�**

	conv_lossp��=6�        )��P	��)!���A�**

	conv_loss��l=�= �        )��P	!�)!���A�**

	conv_lossUG?=k��!        )��P	'*!���A�**

	conv_lossj��=�	+        )��P	�[*!���A�**

	conv_loss��k=<r��        )��P	�,!���A�**

	conv_loss��M=���        )��P	�9,!���A�**

	conv_loss|�=8�|        )��P	�k,!���A�**

	conv_loss�Î=�t!        )��P	�,!���A�**

	conv_loss[s:=���        )��P	�,!���A�**

	conv_loss^Ɲ=���        )��P	�-!���A�**

	conv_lossLUX=�p��        )��P	WE-!���A�**

	conv_loss~�8=��"        )��P	5|-!���A�**

	conv_loss3Ij=����        )��P	K�-!���A�**

	conv_loss@�=�o�)        )��P	A�-!���A�**

	conv_loss"�=V���        )��P	�.!���A�**

	conv_lossK\B=u�4�        )��P	wM.!���A�**

	conv_lossU��=�J_�        )��P	c~.!���A�**

	conv_loss9י=u>�         )��P	+�.!���A�**

	conv_loss�T=J/��        )��P	n�.!���A�**

	conv_loss�i�=��r3        )��P	)/!���A�**

	conv_loss�P�=���        )��P	�G/!���A�**

	conv_loss���=3%��        )��P	iy/!���A�**

	conv_lossS�E=e�/�        )��P	�/!���A�**

	conv_loss-�G=a�T        )��P	\�/!���A�**

	conv_lossN\�=�QrK        )��P	�0!���A�**

	conv_lossi�k=J���        )��P	�O0!���A�**

	conv_loss�L�=]F�0        )��P	҃0!���A�**

	conv_loss��=�P��        )��P	��0!���A�**

	conv_loss�	v=���w        )��P	!�0!���A�**

	conv_loss�p=��O        )��P	�1!���A�**

	conv_loss"�s=�N�0        )��P	;I1!���A�**

	conv_loss|f�=�u�        )��P	�z1!���A�**

	conv_lossؤ�=he��        )��P	v�1!���A�**

	conv_losss�L=��*        )��P	��1!���A�**

	conv_lossX=�Z�        )��P	�2!���A�**

	conv_loss-]H={�/        )��P	�R2!���A�**

	conv_loss��=��so        )��P	��2!���A�**

	conv_loss��=�T1        )��P	�2!���A�**

	conv_lossO=���Q        )��P	*�2!���A�**

	conv_loss3�<=�^G�        )��P	-3!���A�**

	conv_loss�q=,�b�        )��P	�K3!���A�**

	conv_loss��z=˘#o        )��P	�}3!���A�**

	conv_loss\��=�Y�        )��P	x�3!���A�**

	conv_loss8p=�.��        )��P	W�3!���A�**

	conv_loss�T�=$O��        )��P	4!���A�**

	conv_losso�c=�gy        )��P	SJ4!���A�**

	conv_lossf�Y=�7O�        )��P	|4!���A�**

	conv_lossq�W=V�z�        )��P	��4!���A�**

	conv_loss`֦=Ip�        )��P	t�4!���A�**

	conv_loss�?t=��d4        )��P	q5!���A�**

	conv_loss���=�A�J        )��P	1B5!���A�**

	conv_lossW�=���        )��P	�s5!���A�**

	conv_loss�9=Ҧ=�        )��P	Z�5!���A�**

	conv_loss�|�=�]�K        )��P	��5!���A�**

	conv_lossh�=F��        )��P	�6!���A�**

	conv_loss;�E=	���        )��P	�O6!���A�**

	conv_loss�C�=��N�        )��P	Ł6!���A�**

	conv_losso`^=�z%�        )��P	��6!���A�**

	conv_loss��= �@        )��P	��6!���A�**

	conv_loss�u\='wy        )��P	�7!���A�**

	conv_loss��=I��Z        )��P	ve7!���A�**

	conv_loss���=��E�        )��P	E�7!���A�**

	conv_loss�7�=f�        )��P	\�7!���A�**

	conv_loss(�y=\�`(        )��P	�8!���A�**

	conv_loss�S=\!;�        )��P	tS8!���A�**

	conv_lossW�o=��.>        )��P	*�8!���A�**

	conv_loss1�=��v        )��P	��8!���A�**

	conv_loss�JL=�u�?        )��P	s�8!���A�**

	conv_loss���=9���        )��P	B9!���A�**

	conv_losstuK=�͍!        )��P	�M9!���A�**

	conv_lossʃ=�dͷ        )��P	H�9!���A�**

	conv_loss_�{=�0��        )��P	��9!���A�**

	conv_lossL T=�b�        )��P	��9!���A�**

	conv_loss��[=��$�        )��P	�":!���A�**

	conv_loss��=.h"G        )��P	�Z:!���A�**

	conv_loss_)�=\�	P        )��P	��:!���A�**

	conv_loss��=���        )��P	տ:!���A�**

	conv_loss��f=����        )��P	u�:!���A�**

	conv_loss��>=s���        )��P	�#;!���A�**

	conv_loss^U=��        )��P	T;!���A�**

	conv_loss
�==k�(�        )��P	��;!���A�**

	conv_loss���=@f �        )��P	,�;!���A�**

	conv_loss5�x=�s"�        )��P	��;!���A�**

	conv_loss�r_=�|        )��P	
<!���A�**

	conv_loss��>=�>��        )��P	R<!���A�**

	conv_lossb�=(��~        )��P	�<!���A�**

	conv_loss|�:=���        )��P	^�<!���A�**

	conv_lossj�G=�+�)        )��P	��<!���A�**

	conv_loss�{=���        )��P	�=!���A�**

	conv_loss�݃=���Y        )��P	�K=!���A�**

	conv_lossʡs=�4t�        )��P	��=!���A�**

	conv_loss�=���        )��P	n�=!���A�**

	conv_loss4Ј=ڡ*�        )��P	��=!���A�**

	conv_lossP�P=��]h        )��P	>!���A�**

	conv_loss�UG=8�؇        )��P	@I>!���A�**

	conv_loss׋=����        )��P	5{>!���A�**

	conv_loss�^=��(        )��P	d�>!���A�**

	conv_loss��e=q�p        )��P	^�>!���A�**

	conv_lossSr=�_t�        )��P	�?!���A�**

	conv_loss=�_=��t�        )��P	L?!���A�**

	conv_loss� =���        )��P	�?!���A�**

	conv_loss�˚=�}�        )��P	�?!���A�**

	conv_loss�I\=@�E        )��P	Y�?!���A�**

	conv_loss��s=����        )��P	��D!���A�**

	conv_loss��=�b��        )��P	O�D!���A�**

	conv_loss"ϥ=?d�        )��P	fE!���A�**

	conv_loss9D=�xя        )��P	�IE!���A�**

	conv_loss0l=��y        )��P	�xE!���A�+*

	conv_lossD�=�E�        )��P	��E!���A�+*

	conv_loss��l=	9�=        )��P	e�E!���A�+*

	conv_loss�3y=p���        )��P	F!���A�+*

	conv_loss{da=��        )��P	4@F!���A�+*

	conv_lossN*�=�Y;�        )��P	�sF!���A�+*

	conv_loss��Z=b0�W        )��P	0�F!���A�+*

	conv_loss"�$=?M        )��P	M�F!���A�+*

	conv_loss?4X=6�}�        )��P	�G!���A�+*

	conv_loss��V=y��i        )��P	NG!���A�+*

	conv_loss��=�ڇ�        )��P	g}G!���A�+*

	conv_loss�zR=S�S        )��P	�G!���A�+*

	conv_loss�&�=�|        )��P	��G!���A�+*

	conv_loss��=�d��        )��P	� H!���A�+*

	conv_loss�=��	        )��P	�PH!���A�+*

	conv_lossF�@=��^�        )��P	��H!���A�+*

	conv_loss^=�u�        )��P	�H!���A�+*

	conv_loss�>=aI�        )��P	)�H!���A�+*

	conv_loss���=�}2        )��P	�!I!���A�+*

	conv_loss�:==���        )��P	VI!���A�+*

	conv_loss|�=7�/+        )��P	a�I!���A�+*

	conv_loss�B�=��k        )��P	(�I!���A�+*

	conv_loss@aa=��O,        )��P	k�I!���A�+*

	conv_lossy�w=�!�f        )��P	!J!���A�+*

	conv_loss��r=����        )��P	�PJ!���A�+*

	conv_loss'��=�Pen        )��P	:J!���A�+*

	conv_loss�
�=wJ�        )��P	��J!���A�+*

	conv_loss�PS=PO��        )��P	S�J!���A�+*

	conv_loss<d�=i�}M        )��P	� K!���A�+*

	conv_loss5S�=�g�        )��P	QPK!���A�+*

	conv_loss2zm=��V        )��P	��K!���A�+*

	conv_lossf�v=��        )��P	κK!���A�+*

	conv_loss��=�j0        )��P	��K!���A�+*

	conv_lossBsi=7сA        )��P	�'L!���A�+*

	conv_loss���=�Y2J        )��P	�VL!���A�+*

	conv_loss�[=L���        )��P	Y�L!���A�+*

	conv_lossV��=���7        )��P	�L!���A�+*

	conv_loss�1W=7[�        )��P	��L!���A�+*

	conv_loss"�=d�r        )��P	�'M!���A�+*

	conv_loss�Of=��`        )��P		WM!���A�+*

	conv_lossB!e=Z���        )��P	��M!���A�+*

	conv_loss�Đ=���        )��P	��M!���A�+*

	conv_lossL`g=�4        )��P	��M!���A�+*

	conv_loss��=��F|        )��P	D+N!���A�+*

	conv_lossѬ=�׊�        )��P	{aN!���A�+*

	conv_lossKo�=-�Ӿ        )��P	��N!���A�+*

	conv_lossb �=q��        )��P	�N!���A�+*

	conv_loss+�L=��h        )��P	�O!���A�+*

	conv_loss�SB=�M��        )��P	]1O!���A�+*

	conv_lossn��=����        )��P	�_O!���A�+*

	conv_loss�CE=Q��        )��P	��O!���A�+*

	conv_loss��h=I�Y�        )��P	%�O!���A�+*

	conv_loss=N�=��<�        )��P	�P!���A�+*

	conv_loss�[w=\X�        )��P	3P!���A�+*

	conv_losssg�=�n�         )��P	�cP!���A�+*

	conv_loss|�=�G�        )��P	3�P!���A�+*

	conv_lossYi�=l�3         )��P	K�P!���A�+*

	conv_loss�H=2xI�        )��P	�Q!���A�+*

	conv_loss��L=mҨ_        )��P	GQ!���A�+*

	conv_loss�S=ޯ�        )��P	�tQ!���A�+*

	conv_loss�M=��L        )��P	`�Q!���A�+*

	conv_loss|2=���'        )��P	��Q!���A�+*

	conv_loss��6=�/�6        )��P	R!���A�+*

	conv_loss�'�=����        )��P	ADR!���A�+*

	conv_loss��V=�T
9        )��P	 rR!���A�+*

	conv_loss��s=�mH        )��P	P�R!���A�+*

	conv_lossw�=�k՗        )��P	�R!���A�+*

	conv_loss��Y=@v�        )��P	
S!���A�+*

	conv_loss1�=��H        )��P	lBS!���A�+*

	conv_loss�Ke=����        )��P	�vS!���A�+*

	conv_loss�L=B(��        )��P	m�S!���A�+*

	conv_loss�{Q=P�j�        )��P	��S!���A�+*

	conv_loss�F3=k�m/        )��P	�T!���A�+*

	conv_loss�<=+kp        )��P	�HT!���A�+*

	conv_loss�^=�?�        )��P	�yT!���A�+*

	conv_lossYVt=.f��        )��P	��T!���A�+*

	conv_loss�P=e,�        )��P	�T!���A�+*

	conv_loss5�=�1��        )��P	~U!���A�+*

	conv_loss���=#��        )��P	Y?U!���A�+*

	conv_lossCM=��=        )��P	�pU!���A�+*

	conv_lossM=��}p        )��P	�U!���A�+*

	conv_loss��K=��g        )��P	��U!���A�+*

	conv_loss�6=��VW        )��P	�V!���A�+*

	conv_loss+ȉ=9v        )��P	ZBV!���A�+*

	conv_loss��'=)��        )��P	�qV!���A�+*

	conv_loss�֡=��tT        )��P	ΡV!���A�+*

	conv_loss� �=%�E<        )��P	8�V!���A�+*

	conv_loss��=3w��        )��P	yW!���A�+*

	conv_loss"*=Ǳu�        )��P	|5W!���A�+*

	conv_loss0[a=�ڭ/        )��P	BdW!���A�+*

	conv_loss`Oo=);�%        )��P	ÒW!���A�+*

	conv_loss�B=°��        )��P	�W!���A�+*

	conv_lossZoT=8Co�        )��P	��W!���A�+*

	conv_lossM�=���\        )��P	�X!���A�+*

	conv_loss��M=�=�        )��P	�OX!���A�+*

	conv_loss�*�= ��#        )��P	 �X!���A�+*

	conv_loss�=�'Zc        )��P	]�X!���A�+*

	conv_loss'x�=�]��        )��P	JYZ!���A�+*

	conv_lossVU=	�po        )��P	��Z!���A�+*

	conv_loss�@r=��'        )��P	׼Z!���A�+*

	conv_loss?�k=LJ'5        )��P	��Z!���A�+*

	conv_loss�E�=ÒS�        )��P	([!���A�+*

	conv_lossK�=����        )��P	b[!���A�+*

	conv_loss=)�=��i        )��P	��[!���A�+*

	conv_loss�]z=,�
�        )��P	��[!���A�+*

	conv_loss6*b=n�        )��P	��[!���A�+*

	conv_loss��I=m��y        )��P	`8\!���A�+*

	conv_losskD�=G��         )��P	�j\!���A�+*

	conv_loss`׌=(��J        )��P	�\!���A�+*

	conv_lossRhJ=����        )��P	��\!���A�+*

	conv_loss0{=7�X�        )��P	��\!���A�+*

	conv_loss��;=�gk�        )��P	�.]!���A�+*

	conv_loss_J�=�31         )��P	ed]!���A�+*

	conv_loss�X�=�r_        )��P	��]!���A�+*

	conv_loss�t}=�E�        )��P	��]!���A�+*

	conv_loss��{=�La�        )��P	��]!���A�+*

	conv_loss�ߚ=/�\-        )��P	�,^!���A�+*

	conv_loss�	�=����        )��P	m_^!���A�+*

	conv_loss+��=y\�        )��P	8�^!���A�+*

	conv_lossv�g=����        )��P	��^!���A�+*

	conv_loss�"O=$�Y        )��P	v�^!���A�+*

	conv_loss�h=�^%        )��P	�&_!���A�+*

	conv_loss+AY=�F�        )��P	/W_!���A�+*

	conv_loss��Q=�隲        )��P	O�_!���A�+*

	conv_loss���=UUN�        )��P	��_!���A�+*

	conv_loss�==�[�        )��P	M�_!���A�+*

	conv_loss�&~=Џ��        )��P	`!���A�+*

	conv_loss̽h={v�_        )��P	�G`!���A�+*

	conv_loss�o/=k0Y        )��P	/�`!���A�,*

	conv_loss=
�=n1�=        )��P	@�`!���A�,*

	conv_lossjcb=���        )��P	�`!���A�,*

	conv_loss`�=+��        )��P	ea!���A�,*

	conv_loss��=L�a�        )��P	<Ia!���A�,*

	conv_lossWW�=�މ�        )��P	��a!���A�,*

	conv_lossW݉=��%�        )��P	:�a!���A�,*

	conv_loss�@H=C�|�        )��P	��a!���A�,*

	conv_loss,�=�G�        )��P	�b!���A�,*

	conv_loss�M�=d�Yt        )��P	
Sb!���A�,*

	conv_lossM��=O+�        )��P	/�b!���A�,*

	conv_losse�9=���7        )��P	��b!���A�,*

	conv_loss�Rt=>#VD        )��P	��b!���A�,*

	conv_loss���=�c'        )��P	Kc!���A�,*

	conv_loss�t�=���c        )��P	�Oc!���A�,*

	conv_loss�?�=�n37        )��P	Rc!���A�,*

	conv_loss�o=���@        )��P	�c!���A�,*

	conv_loss�a=L/�7        )��P	��c!���A�,*

	conv_losskJX=�!�        )��P	�d!���A�,*

	conv_lossphz=��q�        )��P	QYd!���A�,*

	conv_loss��8=�oy        )��P	��d!���A�,*

	conv_loss@�=7��        )��P	1�d!���A�,*

	conv_lossHE�=UP��        )��P	��d!���A�,*

	conv_lossѺT=�E&s        )��P	0e!���A�,*

	conv_loss���=�.��        )��P	Bce!���A�,*

	conv_loss/�}=�	�t        )��P	�e!���A�,*

	conv_loss�a?=�N        )��P	}�e!���A�,*

	conv_loss�z=�dǅ        )��P	v f!���A�,*

	conv_loss?h=$�<        )��P	�/f!���A�,*

	conv_loss��=	�I�        )��P	df!���A�,*

	conv_loss�=F=����        )��P	��f!���A�,*

	conv_loss��=�.��        )��P	�f!���A�,*

	conv_loss+1S=b(�        )��P	��f!���A�,*

	conv_loss�Ð=�X�]        )��P	j,g!���A�,*

	conv_loss��}=d0�E        )��P	7cg!���A�,*

	conv_lossu�o=W�E        )��P	�g!���A�,*

	conv_loss���=� I�        )��P	�g!���A�,*

	conv_lossDLU=�GTN        )��P	;�g!���A�,*

	conv_loss`�=�d        )��P	o-h!���A�,*

	conv_loss.A�=6\`*        )��P	�]h!���A�,*

	conv_lossHM�=�<.        )��P	��h!���A�,*

	conv_loss�s=�!g        )��P	��h!���A�,*

	conv_lossј�={%�6        )��P	}�h!���A�,*

	conv_loss�>�=��}�        )��P	�#i!���A�,*

	conv_loss��`=}�G        )��P	�Qi!���A�,*

	conv_loss�)=�"�        )��P	j�i!���A�,*

	conv_lossr�=T@��        )��P	B�i!���A�,*

	conv_loss�Sq=S~(        )��P	^�i!���A�,*

	conv_loss�G�=���        )��P	{j!���A�,*

	conv_loss|�=�![U        )��P	bj!���A�,*

	conv_loss��;=�չ        )��P	��j!���A�,*

	conv_loss��h=ZX]�        )��P	E�j!���A�,*

	conv_loss�~V=f�Q�        )��P	�k!���A�,*

	conv_lossWd=��        )��P	4k!���A�,*

	conv_loss��p=%oL�        )��P	fk!���A�,*

	conv_loss��=xb��        )��P	q�k!���A�,*

	conv_loss�/'=O<�'        )��P	��k!���A�,*

	conv_loss��/=�2        )��P	S�k!���A�,*

	conv_loss©'=�v        )��P	�(l!���A�,*

	conv_loss���=�#�1        )��P	_Wl!���A�,*

	conv_loss֘{=%i�'        )��P	�l!���A�,*

	conv_loss�wp=Ə�        )��P	��l!���A�,*

	conv_lossFWf=����        )��P	s�l!���A�,*

	conv_lossd�=�iW�        )��P	H%m!���A�,*

	conv_loss�=���        )��P	�Vm!���A�,*

	conv_loss�V�= ^        )��P	e�m!���A�,*

	conv_loss��=#�        )��P	Ҹm!���A�,*

	conv_loss1v=��K        )��P	V�m!���A�,*

	conv_lossd��=����        )��P	p!n!���A�,*

	conv_lossdΆ=�{�*        )��P	in!���A�,*

	conv_loss�Qa=��         )��P	b�n!���A�,*

	conv_losss�u=��%J        )��P	��n!���A�,*

	conv_lossɆ�=�B        )��P	��n!���A�,*

	conv_lossP�=�cV�        )��P	1=o!���A�,*

	conv_loss���=}��        )��P	�oo!���A�,*

	conv_loss�j�=���{        )��P	�o!���A�,*

	conv_lossq?=�?P        )��P	��o!���A�,*

	conv_loss�\_=
��        )��P	�p!���A�,*

	conv_loss���=�g�q        )��P	�>p!���A�,*

	conv_loss])�=g��        )��P	ftp!���A�,*

	conv_loss��_=���        )��P	'�p!���A�,*

	conv_loss��=Q�*f        )��P	L�p!���A�,*

	conv_loss��=�Dc�        )��P	-q!���A�,*

	conv_loss�
�=���p        )��P	ZDq!���A�,*

	conv_loss
=���        )��P	�wq!���A�,*

	conv_loss�4�==C��        )��P	�q!���A�,*

	conv_lossT��=>���        )��P	��q!���A�,*

	conv_loss�>v=ʍ��        )��P	�+r!���A�,*

	conv_loss��=O�JW        )��P	\r!���A�,*

	conv_lossG��=۽��        )��P	��r!���A�,*

	conv_loss�iD=¸��        )��P	K�r!���A�,*

	conv_loss�c�=�h�        )��P	W�r!���A�,*

	conv_loss�Az='�Z�        )��P	�s!���A�,*

	conv_loss�d=W9N2        )��P	Zs!���A�,*

	conv_loss�o*=��%�        )��P	��s!���A�,*

	conv_loss=�H=��~�        )��P	��s!���A�,*

	conv_loss�ʥ=���        )��P	��s!���A�,*

	conv_loss��^=�r�^        )��P	Qt!���A�,*

	conv_loss$=w�=        )��P	�Tt!���A�,*

	conv_loss�S=(�_�        )��P	�t!���A�,*

	conv_loss%%V=���h        )��P	0�t!���A�,*

	conv_loss {�=|�[�        )��P	k�t!���A�,*

	conv_lossf�B=�l3#        )��P	�u!���A�,*

	conv_loss�G�=�~�*        )��P	EXu!���A�,*

	conv_loss���=����        )��P	��u!���A�,*

	conv_loss<��=�[=�        )��P	�u!���A�,*

	conv_loss�L=k�        )��P	��u!���A�,*

	conv_losst�D=i��        )��P	�v!���A�,*

	conv_loss��|=,ԡ�        )��P	�Uv!���A�,*

	conv_loss�/=�мE        )��P	یv!���A�,*

	conv_loss�`=�#S�        )��P	*�v!���A�,*

	conv_lossa�<=M��A        )��P	��v!���A�,*

	conv_loss�ac=�m��        )��P	�)w!���A�,*

	conv_loss�=#�k        )��P	�gw!���A�,*

	conv_loss��<=b���        )��P	_�w!���A�,*

	conv_loss��;=�0e        )��P	t�w!���A�,*

	conv_loss��f=�R%�        )��P	��w!���A�,*

	conv_lossWjh=D��!        )��P	�+x!���A�,*

	conv_loss��O=���R        )��P	jx!���A�,*

	conv_loss�I=��7        )��P	7�x!���A�,*

	conv_loss��r=�'C5        )��P	,�x!���A�,*

	conv_loss�
N=��^�        )��P	�y!���A�,*

	conv_lossP\b=(87        )��P	�Ky!���A�,*

	conv_loss��=���        )��P	N�y!���A�,*

	conv_loss�ǎ=y�        )��P	�y!���A�,*

	conv_loss�,�=�o�        )��P	.�y!���A�,*

	conv_loss�k7=#�M�        )��P	�z!���A�,*

	conv_loss��g=;�C�        )��P	�Kz!���A�,*

	conv_loss�l=EQ�Q        )��P	 �z!���A�-*

	conv_loss�ı=��{�        )��P	�z!���A�-*

	conv_loss�f�=ײ        )��P	��z!���A�-*

	conv_loss�vz=����        )��P	d{!���A�-*

	conv_loss��f=����        )��P	�E{!���A�-*

	conv_loss�=}PX�        )��P	x{!���A�-*

	conv_lossrj=8�G�        )��P	í{!���A�-*

	conv_loss@|$=_        )��P	�{!���A�-*

	conv_loss���=;N��        )��P	s|!���A�-*

	conv_loss�z�=
z0        )��P	�G|!���A�-*

	conv_loss%�B=���        )��P	�w|!���A�-*

	conv_loss/L=j0S        )��P	��|!���A�-*

	conv_loss��D=��5�        )��P	�|!���A�-*

	conv_loss�c=��        )��P	�
}!���A�-*

	conv_loss�28=��6�        )��P	y:}!���A�-*

	conv_lossV�=\�2u        )��P	�l}!���A�-*

	conv_loss�F�=t�O        )��P	��}!���A�-*

	conv_loss�c�=t�ò        )��P	:�}!���A�-*

	conv_loss��=���        )��P	�~!���A�-*

	conv_loss\s�=߄��        )��P	3~!���A�-*

	conv_lossIG=��        )��P	�a~!���A�-*

	conv_loss>�6=��:        )��P	��~!���A�-*

	conv_loss�9�=�zmf        )��P	�~!���A�-*

	conv_lossY߈=��q9        )��P	s�~!���A�-*

	conv_loss]�w='�~�        )��P	�&!���A�-*

	conv_loss�w,=��P�        )��P	�W!���A�-*

	conv_lossN_=���        )��P	=�!���A�-*

	conv_lossr��=Ƥ�k        )��P	c�!���A�-*

	conv_lossW�o=� ��        )��P	o�!���A�-*

	conv_loss`��=1�3C        )��P	!�!���A�-*

	conv_loss@��=�/l        )��P	S�!���A�-*

	conv_loss�'P=��3�        )��P	x��!���A�-*

	conv_loss�f=Za�u        )��P	���!���A�-*

	conv_loss�/o=n��        )��P	��!���A�-*

	conv_loss��c=OZl�        )��P	��!���A�-*

	conv_loss�&�=I�q�        )��P	UG�!���A�-*

	conv_loss��=Wy�        )��P	�w�!���A�-*

	conv_lossy�\=V`�d        )��P	��!���A�-*

	conv_loss��7=�?а        )��P	��!���A�-*

	conv_loss Rp=9qg        )��P	��!���A�-*

	conv_lossT�=x         )��P	[F�!���A�-*

	conv_loss���=;�i        )��P	*؃!���A�-*

	conv_loss/7u=OUuK        )��P	f�!���A�-*

	conv_lossQr�=�D�W        )��P	d8�!���A�-*

	conv_loss��K=��C�        )��P	 h�!���A�-*

	conv_loss��=k�x        )��P	���!���A�-*

	conv_loss�2=�        )��P	.Մ!���A�-*

	conv_loss4d=�%t        )��P	��!���A�-*

	conv_loss�=�U        )��P	�3�!���A�-*

	conv_lossއ=��u�        )��P	d�!���A�-*

	conv_loss#K�=�~�        )��P	���!���A�-*

	conv_loss��4=h���        )��P	�څ!���A�-*

	conv_loss�H=�,I�        )��P	s�!���A�-*

	conv_loss
��=^u9�        )��P		L�!���A�-*

	conv_loss��=�6�2        )��P	}�!���A�-*

	conv_lossk�=g���        )��P	���!���A�-*

	conv_lossJȎ=�pS\        )��P	��!���A�-*

	conv_loss�M�=���        )��P	n�!���A�-*

	conv_loss���=���,        )��P	�I�!���A�-*

	conv_lossô,=��3        )��P	�x�!���A�-*

	conv_losswI�=�S�f        )��P	n��!���A�-*

	conv_loss׬�=6�h        )��P	��!���A�-*

	conv_loss�:A=
y        )��P	�!���A�-*

	conv_lossO�=i���        )��P	�R�!���A�-*

	conv_loss*�#=|:��        )��P	D��!���A�-*

	conv_loss3s!=K�
        )��P	��!���A�-*

	conv_loss�6�=n��        )��P	�!���A�-*

	conv_loss�L�=4�T�        )��P	�%�!���A�-*

	conv_loss�kW=��
        )��P	U�!���A�-*

	conv_loss�l�=.��        )��P	���!���A�-*

	conv_lossN25=���        )��P	%��!���A�-*

	conv_loss��O=�g�         )��P	��!���A�-*

	conv_loss�<=L1�	        )��P	��!���A�-*

	conv_loss�u�=���        )��P	�H�!���A�-*

	conv_loss�ԍ=w�a�        )��P	�x�!���A�-*

	conv_loss��A=Eu�y        )��P	_��!���A�-*

	conv_loss��=k��        )��P	�؊!���A�-*

	conv_loss�E�=�;I         )��P	
�!���A�-*

	conv_loss:Ŏ=*���        )��P	�<�!���A�-*

	conv_loss(�l=d	��        )��P	�l�!���A�-*

	conv_loss=|XS�        )��P	 ��!���A�-*

	conv_loss�K|=KAli        )��P	�͋!���A�-*

	conv_lossR�p=���8        )��P	5 �!���A�-*

	conv_lossq[=�0�        )��P	�0�!���A�-*

	conv_lossR��<�oD;        )��P	a�!���A�-*

	conv_lossЉS=&)��        )��P	���!���A�-*

	conv_loss��i=�]i        )��P	Ō!���A�-*

	conv_loss|{=����        )��P	���!���A�-*

	conv_loss��==CG��        )��P	-�!���A�-*

	conv_loss�W`==�;�        )��P	^�!���A�-*

	conv_lossֲZ=��M        )��P	���!���A�-*

	conv_loss�ۅ=:�V�        )��P	hԍ!���A�-*

	conv_lossQ�=I��f        )��P	��!���A�-*

	conv_lossE�=1q�        )��P	C�!���A�-*

	conv_loss'l=��^�        )��P	Lu�!���A�-*

	conv_loss\��=R�Q�        )��P	r��!���A�-*

	conv_lossdK=���        )��P	�؎!���A�-*

	conv_lossh��=�ٱ�        )��P	��!���A�-*

	conv_lossE��=�n�        )��P	DE�!���A�-*

	conv_loss���=�B�        )��P	�z�!���A�-*

	conv_loss�k=�y��        )��P	���!���A�-*

	conv_loss�Hf=�{c        )��P	��!���A�-*

	conv_loss2�g=@�oN        )��P	�(�!���A�-*

	conv_loss�gU=;&a        )��P	j\�!���A�-*

	conv_loss�q�= W�        )��P	���!���A�-*

	conv_lossޞ={�A,        )��P	�!���A�-*

	conv_loss�1Z=����        )��P	��!���A�-*

	conv_loss:��=_��        )��P	�3�!���A�-*

	conv_loss u�=�k        )��P	�c�!���A�-*

	conv_loss��+=���         )��P	I��!���A�-*

	conv_lossy4T=����        )��P	�ő!���A�-*

	conv_loss\��=�J�        )��P	���!���A�-*

	conv_loss�ۑ=�'�        )��P	e#�!���A�-*

	conv_loss,Ԡ=M��?        )��P	$Z�!���A�-*

	conv_lossJ�z=�2��        )��P	W��!���A�-*

	conv_loss�Z�=��!�        )��P	zĒ!���A�-*

	conv_loss�U�=zlw        )��P	��!���A�-*

	conv_loss��Z=�"        )��P	1&�!���A�-*

	conv_loss�>/=����        )��P	UV�!���A�-*

	conv_loss/��=��Z        )��P	��!���A�-*

	conv_loss0�=삘�        )��P	���!���A�-*

	conv_loss��A=����        )��P	-�!���A�-*

	conv_loss�K�=�N        )��P	��!���A�-*

	conv_lossF8K=眏�        )��P	�J�!���A�-*

	conv_loss"�=�K�X        )��P	my�!���A�-*

	conv_loss#.=x#��        )��P	��!���A�-*

	conv_lossw�=�\�        )��P	�ڔ!���A�-*

	conv_lossI�`=8�W�        )��P	�	�!���A�-*

	conv_lossߴd=c�W^        )��P	t8�!���A�.*

	conv_loss/s=R�R        )��P	[h�!���A�.*

	conv_loss�/�=qǶ:        )��P	���!���A�.*

	conv_loss�T�=]���        )��P	�ƕ!���A�.*

	conv_loss�bK=��v        )��P	��!���A�.*

	conv_loss=Č=]h+         )��P	h)�!���A�.*

	conv_loss۞a=	P�        )��P	JY�!���A�.*

	conv_lossr.=?~b�        )��P	�!���A�.*

	conv_loss^R8=��*        )��P	���!���A�.*

	conv_lossf�m=u���        )��P	��!���A�.*

	conv_lossJ�;=al        )��P	��!���A�.*

	conv_loss6r=���        )��P	�Q�!���A�.*

	conv_loss�w=�kS        )��P	㊗!���A�.*

	conv_lossݘ4=��>(        )��P	�Η!���A�.*

	conv_loss�L=J�Κ        )��P	��!���A�.*

	conv_loss�`W=D���        )��P	EA�!���A�.*

	conv_loss��]=+�M        )��P	�w�!���A�.*

	conv_loss��v=�O�n        )��P	���!���A�.*

	conv_lossO�D=UA�        )��P	ݘ!���A�.*

	conv_loss�j�=.�B�        )��P	��!���A�.*

	conv_loss��~=sH�        )��P	kH�!���A�.*

	conv_loss6\n=?F|�        )��P	�z�!���A�.*

	conv_loss]�"=6        )��P	ܪ�!���A�.*

	conv_loss?N�=2��Y        )��P	C�!���A�.*

	conv_loss���=��
{        )��P	��!���A�.*

	conv_lossИ	=���        )��P	�U�!���A�.*

	conv_loss��=:h��        )��P	1��!���A�.*

	conv_lossx=���        )��P	J��!���A�.*

	conv_loss���=C��        )��P	|�!���A�.*

	conv_loss���=$��        )��P	�!�!���A�.*

	conv_loss]z�=�ۻS        )��P	�Y�!���A�.*

	conv_loss1�s=7�u        )��P	��!���A�.*

	conv_lossnTx=Q�        )��P	�ś!���A�.*

	conv_lossO�=�"�        )��P	� �!���A�.*

	conv_loss�\p=���F        )��P	�>�!���A�.*

	conv_lossJ�=IZ�R        )��P	4~�!���A�.*

	conv_loss�\�=�JD<        )��P	���!���A�.*

	conv_lossJ�=��1        )��P	?�!���A�.*

	conv_loss�`=@qQF        )��P	��!���A�.*

	conv_loss���=rv�        )��P	�N�!���A�.*

	conv_loss��B=Ǹ�        )��P	ო!���A�.*

	conv_loss)�H=	Z        )��P	ϵ�!���A�.*

	conv_loss�sV=�vw�        )��P	��!���A�.*

	conv_losss��=#C-        )��P	��!���A�.*

	conv_loss���=T6�K        )��P	#\�!���A�.*

	conv_loss�w�=#�`!        )��P	n��!���A�.*

	conv_loss�V=J��        )��P	�̞!���A�.*

	conv_loss��c=�ĥ        )��P	.�!���A�.*

	conv_lossS�`=�߈�        )��P	�:�!���A�.*

	conv_lossp�=Aw�        )��P	���!���A�.*

	conv_loss�*I=׃^�        )��P	�̟!���A�.*

	conv_loss�x=��I        )��P	,��!���A�.*

	conv_loss�P|=\�W        )��P	J,�!���A�.*

	conv_loss��w=����        )��P	�Z�!���A�.*

	conv_lossO��=V��        )��P	 ��!���A�.*

	conv_lossm�'=��        )��P	� !���A�.*

	conv_loss�_u="��        )��P	+��!���A�.*

	conv_loss5ox=m�j�        )��P	%�!���A�.*

	conv_loss�vf=o�yp        )��P	V�!���A�.*

	conv_loss���=}x|U        )��P	L��!���A�.*

	conv_loss��,=E@b,        )��P	�͡!���A�.*

	conv_loss�D�=�y��        )��P	% �!���A�.*

	conv_loss�[k=�"�7        )��P	1�!���A�.*

	conv_loss�H=j{&�        )��P	�r�!���A�.*

	conv_loss�b=82         )��P	���!���A�.*

	conv_loss��Z=r��=        )��P	`ۢ!���A�.*

	conv_loss��e=��-/        )��P	a�!���A�.*

	conv_loss��=;k�@        )��P	z:�!���A�.*

	conv_losse�u=�c�b        )��P	�i�!���A�.*

	conv_loss�{|=[|¿        )��P	D��!���A�.*

	conv_lossցi=O�0�        )��P	2У!���A�.*

	conv_loss��T='��C        )��P	��!���A�.*

	conv_loss�7q=#�B        )��P	,C�!���A�.*

	conv_loss��`=9@        )��P	�v�!���A�.*

	conv_lossWS/=���        )��P	���!���A�.*

	conv_losswzo=g�<        )��P	C�!���A�.*

	conv_loss��6=Iz M        )��P	��!���A�.*

	conv_loss/�$=}��        )��P	�E�!���A�.*

	conv_loss/�=E�R�        )��P	Gs�!���A�.*

	conv_loss7�N="�C        )��P	9��!���A�.*

	conv_loss9|_=3��?        )��P	٥!���A�.*

	conv_lossQn=S�B�        )��P	Q�!���A�.*

	conv_loss#ڃ=aCg        )��P	?�!���A�.*

	conv_lossw�U=��!        )��P	Um�!���A�.*

	conv_loss�oY=�=�        )��P	���!���A�.*

	conv_loss?6�="}|�        )��P	��!���A�.*

	conv_loss�BB=�w��        )��P	w�!���A�.*

	conv_lossE0=/'I        )��P	~F�!���A�.*

	conv_lossr`=x���        )��P	kw�!���A�.*

	conv_loss���=����        )��P	å�!���A�.*

	conv_loss�~=h]n        )��P	]ԧ!���A�.*

	conv_loss�k6=�-d        )��P	��!���A�.*

	conv_loss�Cl=�:)M        )��P	�8�!���A�.*

	conv_loss�y*=;mdT        )��P	j�!���A�.*

	conv_loss^J�=`ҏt        )��P	���!���A�.*

	conv_loss�V=�_��        )��P	/ɨ!���A�.*

	conv_lossF�=Eڄ        )��P	z��!���A�.*

	conv_lossI3d=tM:&        )��P	�)�!���A�.*

	conv_loss�W=?f�        )��P	�X�!���A�.*

	conv_loss�x�=i�J        )��P	���!���A�.*

	conv_loss��U=����        )��P	A��!���A�.*

	conv_loss 6�=�{��        )��P	?�!���A�.*

	conv_loss�A�=grD        )��P	��!���A�.*

	conv_loss�DW=�C'        )��P	�A�!���A�.*

	conv_loss<�b=8֎�        )��P	Qs�!���A�.*

	conv_lossMe=q׸�        )��P	=��!���A�.*

	conv_loss�M=�@�        )��P	_Ӫ!���A�.*

	conv_loss��(=3ǩ�        )��P	��!���A�.*

	conv_lossT�m=�J        )��P	�1�!���A�.*

	conv_loss@�=�n˺        )��P	�`�!���A�.*

	conv_loss�L=�W�:        )��P	F��!���A�.*

	conv_loss�X=����        )��P	���!���A�.*

	conv_loss=�=���J