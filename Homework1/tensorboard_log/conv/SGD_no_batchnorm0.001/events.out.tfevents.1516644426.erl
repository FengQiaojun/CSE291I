       �K"	  ����Abrain.Event:2�V����      D(�	�	����A"��
~
PlaceholderPlaceholder*/
_output_shapes
:���������*$
shape:���������*
dtype0
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

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 *
dtype0*&
_output_shapes
:
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
conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
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
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
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
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*
T0*
strides
*
data_formatNHWC*
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
.conv2d_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *��:>
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
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_3/kernel
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
conv2d_4/kernel/AssignAssignconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel
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
dtype0*
_output_shapes
:*
valueB"      
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
dense/kernel/readIdentitydense/kernel*
_output_shapes
:	�
d*
T0*
_class
loc:@dense/kernel
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
-dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *�'o�*
dtype0
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

seed *
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes

:d

�
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
: *

Tidx0*
	keep_dims( *
T0
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
:*

Tidx0*
	keep_dims( *
T0
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
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
out_type0*
_output_shapes
:*
T0
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
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
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
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
_output_shapes
:
*
T0*
data_formatNHWC
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
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_7_grad/ReluGraddense/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������
*
transpose_a( 
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
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
strides
*
data_formatNHWC*
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_2/Conv2D_grad/tuple/group_deps*E
_class;
97loc:@gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
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
5gradients/conv2d/Conv2D_grad/tuple/control_dependencyIdentity0gradients/conv2d/Conv2D_grad/Conv2DBackpropInput.^gradients/conv2d/Conv2D_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*C
_class9
75loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropInput
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
 *o�:*
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
: "��G�      ?�K	������AJ��
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
.conv2d/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:* 
_class
loc:@conv2d/kernel*%
valueB"            *
dtype0
�
,conv2d/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d/kernel*
valueB
 *����
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
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel
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
.conv2d_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *��:>*
dtype0
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
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel
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
.conv2d_3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
valueB
 *�>*
dtype0
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
.conv2d_4/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel*
valueB
 *HY�*
dtype0
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
VariableV2*"
_class
loc:@conv2d_4/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
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
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_5/kernel
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
0conv2d_6/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_6/kernel*%
valueB"            
�
.conv2d_6/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_6/kernel*
valueB
 *���
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
_class
loc:@dense/kernel*
valueB"\  d   *
dtype0*
_output_shapes
:
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
_output_shapes
:	�
d*
T0*
_class
loc:@dense/kernel
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
-dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *�'o�*
dtype0
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
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
T0
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
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:
*
T0*
_class
loc:@dense_1/bias
�
dense_2/MatMulMatMulRelu_7dense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������
*
transpose_a( 
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
ConstConst*
_output_shapes
:*
valueB"       *
dtype0
`
MeanMeanlogistic_lossConst*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
X
conv_loss/tagsConst*
dtype0*
_output_shapes
: *
valueB B	conv_loss
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
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*'
_output_shapes
:���������
*

Tmultiples0
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
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
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
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
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
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������
*
T0
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
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
7gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������
*
T0
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
N* 
_output_shapes
::*
T0*
out_type0
�
2gradients/conv2d_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_6/Conv2D_grad/ShapeNconv2d_5/kernel/readgradients/Relu_5_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
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
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/Relu_2_grad/ReluGrad*
T0*
strides
*
data_formatNHWC*
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
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/Relu_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
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
5gradients/conv2d/Conv2D_grad/tuple/control_dependencyIdentity0gradients/conv2d/Conv2D_grad/Conv2DBackpropInput.^gradients/conv2d/Conv2D_grad/tuple/group_deps*C
_class9
75loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
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
 *o�:*
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
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0��h�       `/�#	������A*

	conv_loss�0?J�       QKD	�]����A*

	conv_loss��0?!)%       QKD	>�����A*

	conv_loss+c0?R�S       QKD	������A*

	conv_lossId0?me[o       QKD	������A*

	conv_lossI�0?�s�F       QKD	*3����A*

	conv_loss��0?��\       QKD	uf����A*

	conv_loss=T0?wX&�       QKD	H�����A*

	conv_loss�s0?�!�       QKD	O�����A*

	conv_loss�_0?gtv�       QKD	�����A	*

	conv_lossv0?�vW       QKD	�A����A
*

	conv_loss�S0?-O�o       QKD	 |����A*

	conv_lossJH0?�rص       QKD	������A*

	conv_loss�K0?h��u       QKD	~�����A*

	conv_loss�00?���2       QKD	�+����A*

	conv_lossu0?�q��       QKD	`^����A*

	conv_loss�\0?�#V       QKD	%�����A*

	conv_lossO]0?H�@       QKD	v�����A*

	conv_lossBU0?���       QKD	O�����A*

	conv_loss�J0?�۹       QKD	%����A*

	conv_loss�S0?����       QKD	lW����A*

	conv_loss�0?���       QKD	V�����A*

	conv_loss�N0?�Q�       QKD	s�����A*

	conv_loss�J0?%�v       QKD	,�����A*

	conv_loss40?�/86       QKD	&����A*

	conv_loss�E0?�K-       QKD	�W����A*

	conv_loss|>0?�k�       QKD	�����A*

	conv_loss�0?b�`J       QKD	�����A*

	conv_loss�S0?@�:       QKD	=�����A*

	conv_loss20?��X�       QKD	�#����A*

	conv_lossR0?A4�       QKD	V����A*

	conv_losss0?�t�T       QKD	�����A*

	conv_lossv30?_��       QKD	:�����A *

	conv_loss�'0?�       QKD	M�����A!*

	conv_loss60?Z�       QKD	\+����A"*

	conv_loss�0?�0a�       QKD	�_����A#*

	conv_loss�30?�i�       QKD	�����A$*

	conv_loss�D0?�~       QKD	M�����A%*

	conv_loss�)0?�3       QKD	U�����A&*

	conv_loss�0?��       QKD	d+����A'*

	conv_lossa+0?�}��       QKD	c����A(*

	conv_loss{-0?�b�O       QKD	c�����A)*

	conv_lossY"0?0��       QKD	������A**

	conv_loss�0?�CL�       QKD	������A+*

	conv_lossF0?���       QKD	00����A,*

	conv_loss��/?��a       QKD	j����A-*

	conv_loss,@0?���i       QKD	������A.*

	conv_lossl0?�,       QKD	������A/*

	conv_loss�0?K��N       QKD	Q����A0*

	conv_loss��/?�S��       QKD	�?����A1*

	conv_loss��/?�ʝ       QKD	�r����A2*

	conv_lossX�/?	�u       QKD	������A3*

	conv_loss��/?h�        QKD	������A4*

	conv_loss 0?��J       QKD	
����A5*

	conv_loss>0?R,0�       QKD	YM����A6*

	conv_loss6
0?�2F�       QKD	�~����A7*

	conv_loss0?N?��       QKD	������A8*

	conv_loss�0?GY       QKD	�����A9*

	conv_loss��/?���       QKD	J����A:*

	conv_loss�/?�%�f       QKD	?����A;*

	conv_loss�0?s���       QKD	������A<*

	conv_loss�0?߱@�       QKD	y�����A=*

	conv_loss��/?��5�       QKD	������A>*

	conv_lossJ0?˙�       QKD	�����A?*

	conv_loss%�/?���       QKD	�P����A@*

	conv_loss�0?�I�       QKD	�����AA*

	conv_loss�0?�mC       QKD	������AB*

	conv_loss��/?[�W�       QKD	������AC*

	conv_loss��/?�i>       QKD	����AD*

	conv_loss��/?5��S       QKD	uH����AE*

	conv_loss��/?j;�y       QKD	�x����AF*

	conv_loss�/?�6��       QKD	�����AG*

	conv_loss&�/?�=?       QKD	=�����AH*

	conv_lossи/?��       QKD	�����AI*

	conv_loss��/?4i�       QKD	?D����AJ*

	conv_lossq�/?�b$       QKD	�t����AK*

	conv_loss��/?��~       QKD	ؤ����AL*

	conv_loss��/?��       QKD	������AM*

	conv_loss�/?\#�l       QKD	5	����AN*

	conv_loss�/?h��       QKD	@:����AO*

	conv_loss��/?𲷜       QKD	�h����AP*

	conv_loss��/?+�2�       QKD	M�����AQ*

	conv_losst�/?���       QKD	b�����AR*

	conv_loss?�/?�dA       QKD	������AS*

	conv_loss!�/?\|�       QKD	�*����AT*

	conv_loss�/?ra��       QKD	�Z����AU*

	conv_lossƑ/?1]`�       QKD	������AV*

	conv_loss��/?��eg       QKD	�����AW*

	conv_loss^�/?�%c�       QKD	������AX*

	conv_loss:�/?)ҕ�       QKD	r����AY*

	conv_loss(�/?�QK�       QKD	�E����AZ*

	conv_loss�/?#��'       QKD	hv����A[*

	conv_lossX�/?�y       QKD	f�����A\*

	conv_loss9�/?!���       QKD	������A]*

	conv_loss͗/?�,�       QKD	
����A^*

	conv_loss��/?R]       QKD	U:����A_*

	conv_lossQ�/?U�^]       QKD	k����A`*

	conv_loss�/?�["�       QKD	F�����Aa*

	conv_loss��/?\u��       QKD	������Ab*

	conv_loss-r/?���#       QKD	� ����Ac*

	conv_lossԄ/?��H�       QKD	2����Ad*

	conv_loss��/?Ysi       QKD	mu����Ae*

	conv_loss��/?JJ�;       QKD	Ȧ����Af*

	conv_lossm�/?���       QKD	z�����Ag*

	conv_loss��/?�9!       QKD	6����Ah*

	conv_lossȍ/?��B8       QKD	�:����Ai*

	conv_loss�/?Z���       QKD	�i����Aj*

	conv_loss-�/?�Z��       QKD	������Ak*

	conv_lossF�/?�]�       QKD	b�����Al*

	conv_lossk�/?+�[�       QKD	�����Am*

	conv_loss��/?�k*       QKD	�9����An*

	conv_lossvu/?4�g       QKD	Ki����Ao*

	conv_loss�/?$�#�       QKD	������Ap*

	conv_lossՅ/?��0       QKD	�����Aq*

	conv_loss7�/?j���       QKD	�����Ar*

	conv_loss>�/?iv�S       QKD	NB����As*

	conv_loss�y/?�lp�       QKD	�q����At*

	conv_lossd/?��AT       QKD	�����Au*

	conv_loss��/?�b��       QKD	=�����Av*

	conv_loss��/?���       QKD	�����Aw*

	conv_loss|�/?/3��       QKD	2����Ax*

	conv_loss�Y/?=��       QKD	�`����Ay*

	conv_losssM/?l%�       QKD	������Az*

	conv_loss�i/?�56       QKD	������A{*

	conv_loss�l/?r��       QKD	������A|*

	conv_lossT/?_
�~       QKD	x"����A}*

	conv_loss�/?|35       QKD	�U����A~*

	conv_loss,]/?F(`2       QKD	H�����A*

	conv_losslD/?�5��        )��P	U�����A�*

	conv_loss�G/?W��Q        )��P	������A�*

	conv_loss<�/?ݿ��        )��P	�����A�*

	conv_loss]/?�?f�        )��P	�H����A�*

	conv_loss�G/?,L�|        )��P	�x����A�*

	conv_loss�S/?4��F        )��P	Y�����A�*

	conv_loss�./?�tn        )��P	������A�*

	conv_loss)c/?���        )��P	(����A�*

	conv_loss�0/?fN        )��P	y>����A�*

	conv_lossQ/?p�W        )��P	'o����A�*

	conv_lossP/?,��        )��P	������A�*

	conv_loss�_/?��{�        )��P	������A�*

	conv_loss0/?�b�        )��P		����A�*

	conv_loss�P/?�G        )��P	�4����A�*

	conv_lossu$/?bSLT        )��P	Tg����A�*

	conv_lossC,/?-��p        )��P	A�����A�*

	conv_loss�J/?����        )��P	������A�*

	conv_lossEE/?�G;�        )��P	�����A�*

	conv_loss42/?ؒ$|        )��P	s5����A�*

	conv_loss�?/?�y�*        )��P	[h����A�*

	conv_loss�?/?�L�        )��P	֛����A�*

	conv_losso3/?
�hj        )��P	������A�*

	conv_loss"/?-�D        )��P	W�����A�*

	conv_loss}'/?�@,�        )��P	�D����A�*

	conv_loss*R/?~Mh        )��P	gy����A�*

	conv_loss�//?�
        )��P	�����A�*

	conv_loss�&/?E�        )��P	������A�*

	conv_loss	�.?y/��        )��P	Q����A�*

	conv_loss�T/?�8?�        )��P	�E����A�*

	conv_loss/?V$��        )��P	*y����A�*

	conv_loss�!/?��b        )��P	�����A�*

	conv_lossN/?M[&�        )��P	C�����A�*

	conv_loss��.?�?3$        )��P	0)����A�*

	conv_loss'./?"�h        )��P	c[����A�*

	conv_loss:/?;�x$        )��P	�����A�*

	conv_loss�-/?I�        )��P	y�����A�*

	conv_loss��.?�uӺ        )��P	�����A�*

	conv_loss"�.?Ns��        )��P	�:����A�*

	conv_loss.�.?^6"�        )��P	�n����A�*

	conv_loss�!/?�%:�        )��P	������A�*

	conv_lossaD/?&ճ�        )��P	������A�*

	conv_loss��.?����        )��P	�����A�*

	conv_lossG/?R��        )��P	?����A�*

	conv_loss$�.?�
�        )��P	�q����A�*

	conv_loss�+/?h�'j        )��P	.�����A�*

	conv_lossS�.?MH4        )��P	������A�*

	conv_loss
/?�H\�        )��P	F����A�*

	conv_loss�/?�	�        )��P	�A����A�*

	conv_loss�)/?;�Q�        )��P	�u����A�*

	conv_loss0�.?ND�        )��P	������A�*

	conv_lossM�.?b�6�        )��P	(�����A�*

	conv_loss��.?���        )��P	�����A�*

	conv_loss\�.?��!�        )��P	mF����A�*

	conv_loss!�.?���H        )��P	~y����A�*

	conv_loss��.?>aX        )��P	������A�*

	conv_loss�.?���        )��P	������A�*

	conv_loss��.?�.�        )��P	m����A�*

	conv_loss��.?��
i        )��P	�K����A�*

	conv_loss��.?n���        )��P	B�����A�*

	conv_loss=�.?9�$        )��P	}�����A�*

	conv_lossh�.?7UD�        )��P	������A�*

	conv_loss��.?����        )��P	����A�*

	conv_loss2�.?���)        )��P	�P����A�*

	conv_loss��.?K2�=        )��P	�����A�*

	conv_lossr�.?o��        )��P	-�����A�*

	conv_loss��.?���}        )��P	������A�*

	conv_loss��.?��+�        )��P		����A�*

	conv_loss��.?2wګ        )��P	�Q����A�*

	conv_loss�.?��4�        )��P	p�����A�*

	conv_loss"�.?����        )��P	�����A�*

	conv_lossn�.?����        )��P	�����A�*

	conv_loss\�.?�
��        )��P	�����A�*

	conv_loss �.?�@�"        )��P	�N����A�*

	conv_loss�.?��t�        )��P	 �����A�*

	conv_loss��.?h�,        )��P	�����A�*

	conv_loss��.?�*�        )��P	�K����A�*

	conv_loss��.?Q6a        )��P	5�����A�*

	conv_loss��.?Qj        )��P	t�����A�*

	conv_loss-�.?%J        )��P	������A�*

	conv_loss�.?�,��        )��P	'����A�*

	conv_loss/�.?Rk��        )��P	uc����A�*

	conv_loss	�.?Q��        )��P		�����A�*

	conv_loss[�.? �/        )��P	������A�*

	conv_loss��.?G��        )��P	L�����A�*

	conv_lossۃ.?���s        )��P	*����A�*

	conv_loss��.?L9B[        )��P	�_����A�*

	conv_loss!�.?밵+        )��P	H�����A�*

	conv_lossЫ.?W���        )��P	������A�*

	conv_loss�.?�<        )��P	������A�*

	conv_loss�.?jNb�        )��P	M*����A�*

	conv_loss׳.?[u��        )��P	e����A�*

	conv_loss�m.?�n?        )��P	К����A�*

	conv_lossT�.?U/         )��P	9�����A�*

	conv_loss��.?�s]m        )��P	�����A�*

	conv_loss�{.?L��        )��P	$7����A�*

	conv_loss��.??B��        )��P	i����A�*

	conv_lossƐ.?��f_        )��P	������A�*

	conv_loss�.?�`$�        )��P	������A�*

	conv_losss�.?�U��        )��P	H����A�*

	conv_loss�{.?e��s        )��P	2����A�*

	conv_loss@O.?�kz(        )��P	�b����A�*

	conv_loss^~.?�D�        )��P	ѓ����A�*

	conv_loss�`.?���A        )��P	t�����A�*

	conv_loss�{.?^        )��P	H�����A�*

	conv_loss5�.?y�=�        )��P	1(����A�*

	conv_lossz.?��σ        )��P	d[����A�*

	conv_lossoE.?T'.        )��P	������A�*

	conv_loss(|.?j���        )��P	�����A�*

	conv_loss[n.?��7n        )��P	p�����A�*

	conv_lossGH.?�K�        )��P	�!����A�*

	conv_lossp^.?�nP
        )��P	�T����A�*

	conv_loss�S.?��"        )��P	������A�*

	conv_loss�c.?6�%        )��P	������A�*

	conv_loss�j.?7l�        )��P	{�����A�*

	conv_loss�.?߱o        )��P	�����A�*

	conv_loss;.?�\/        )��P	\I����A�*

	conv_lossJ\.?����        )��P	�{����A�*

	conv_loss�.?cg5�        )��P	�����A�*

	conv_loss�k.?N�        )��P	������A�*

	conv_lossAK.?�w�        )��P	D����A�*

	conv_loss�0.?E���        )��P	�A����A�*

	conv_loss�6.?�xZ�        )��P	Yu����A�*

	conv_loss�S.?M���        )��P	�����A�*

	conv_loss�L.?�4�        )��P	������A�*

	conv_lossq.?z8�O        )��P	 ����A�*

	conv_loss &.?O��        )��P	�R����A�*

	conv_losseQ.?:���        )��P	n�����A�*

	conv_loss#.?l��y        )��P	������A�*

	conv_loss�L.?�i4�        )��P	������A�*

	conv_loss
.?��Y�        )��P	y/����A�*

	conv_loss�C.?�Rz�        )��P	�k����A�*

	conv_loss�J.?f�!        )��P	������A�*

	conv_loss�0.?2�Ј        )��P	������A�*

	conv_loss�.?/�f%        )��P	�����A�*

	conv_loss�4.?��t�        )��P	�8����A�*

	conv_lossH-.?qx��        )��P	�k����A�*

	conv_loss�$.?
iF�        )��P	������A�*

	conv_loss��-?$�X        )��P	������A�*

	conv_loss".?�=í        )��P	�����A�*

	conv_loss�4.?;��        )��P	b>����A�*

	conv_loss��-?7_        )��P	gw����A�*

	conv_loss�#.?��':        )��P	N�����A�*

	conv_lossl.?�d��        )��P	������A�*

	conv_lossD�-?{�m        )��P	����A�*

	conv_loss�,.?t���        )��P	�I����A�*

	conv_loss� .?5�nj        )��P	|����A�*

	conv_loss�.?Z2�        )��P	������A�*

	conv_lossX!.?���        )��P	D�����A�*

	conv_lossI.?����        )��P	h����A�*

	conv_loss�.?���        )��P	�@����A�*

	conv_lossM.?���        )��P	 r����A�*

	conv_lossh�-?�1X'        )��P	�����A�*

	conv_loss��-?dt/G        )��P	�����A�*

	conv_lossS�-?���        )��P	����A�*

	conv_loss��-?e�E        )��P	o;����A�*

	conv_loss`�-?ǘR�        )��P	�p����A�*

	conv_loss��-?Mё�        )��P	J�����A�*

	conv_loss��-?�ݼ�        )��P	J�����A�*

	conv_loss��-?�0��        )��P	|����A�*

	conv_loss��-?wH��        )��P	r6����A�*

	conv_loss��-?1���        )��P	�m����A�*

	conv_loss��-?J��        )��P	?�����A�*

	conv_lossh�-?��b�        )��P	�����A�*

	conv_losse.?5zw�        )��P	a����A�*

	conv_loss��-?X��5        )��P	L:����A�*

	conv_loss��-?Ru�        )��P	�s����A�*

	conv_loss�-?��D        )��P	p�����A�*

	conv_loss��-?_��        )��P	������A�*

	conv_loss�-?L�j�        )��P	�����A�*

	conv_lossa�-?�Rݏ        )��P	�J����A�*

	conv_loss�-?W4        )��P	�y����A�*

	conv_loss�x-?�>��        )��P	,�����A�*

	conv_loss�-?su{�        )��P	������A�*

	conv_lossi�-?��d        )��P	�"����A�*

	conv_loss*�-?����        )��P	�V����A�*

	conv_loss�-?U���        )��P	3�����A�*

	conv_loss�-?�Ӹ        )��P	������A�*

	conv_loss�-?��Š        )��P	h�����A�*

	conv_lossb�-?��)        )��P	=. ���A�*

	conv_loss�-?��G        )��P	j` ���A�*

	conv_lossn�-?Dl�        )��P	�� ���A�*

	conv_loss .?���        )��P	� ���A�*

	conv_loss/�-?��_        )��P	���A�*

	conv_loss�-?��Ξ        )��P	�4���A�*

	conv_loss��-?ͭ        )��P	�e���A�*

	conv_loss9�-??�#�        )��P	4����A�*

	conv_loss��-?*��        )��P	x����A�*

	conv_loss��-?bӶy        )��P	>���A�*

	conv_loss|�-?&e��        )��P	u5���A�*

	conv_loss�t-?��        )��P	ml���A�*

	conv_loss �-?���*        )��P	٢���A�*

	conv_loss �-?a�+_        )��P	�����A�*

	conv_loss:�-?P��J        )��P	[���A�*

	conv_loss9�-?�p        )��P	f;���A�*

	conv_lossΔ-?��ɰ        )��P	)m���A�*

	conv_losse�-?��Y�        )��P	����A�*

	conv_loss�w-?+"��        )��P	����A�*

	conv_lossG-?��        )��P	�����A�*

	conv_lossw-?���        )��P	
1���A�*

	conv_loss|�-?ܚ	�        )��P	�d���A�*

	conv_lossd\-?�u��        )��P	����A�*

	conv_lossX�-?�ۑ-        )��P	�����A�*

	conv_lossJe-?~��s        )��P	�����A�*

	conv_loss5Y-?q���        )��P	-0���A�*

	conv_lossgA-? `W�        )��P	�a���A�*

	conv_loss@R-?��        )��P	d����A�*

	conv_lossA:-?�ä�        )��P	Y����A�*

	conv_loss�5-?��H�        )��P	�����A�*

	conv_loss��-?;~�;        )��P	Y*���A�*

	conv_loss>N-?�!-        )��P	\���A�*

	conv_lossE-?g�	�        )��P	�����A�*

	conv_loss�r-?�û�        )��P	4����A�*

	conv_loss�7-?���[        )��P	�����A�*

	conv_loss�-?��        )��P	�"���A�*

	conv_lossJ/-?��#        )��P	T���A�*

	conv_loss�7-?��g�        )��P	�����A�*

	conv_loss�m-?��        )��P	�����A�*

	conv_loss6-?;�|�        )��P	�����A�*

	conv_loss26-?��E�        )��P	����A�*

	conv_loss?6-?�$�        )��P	fR���A�*

	conv_lossb5-?��J=        )��P	�����A�*

	conv_loss�1-?|�        )��P	P����A�*

	conv_loss}0-?���f        )��P	F����A�*

	conv_loss7-?xh        )��P	2-	���A�*

	conv_loss�-?/myf        )��P	b`	���A�*

	conv_loss��,?�        )��P	��	���A�*

	conv_loss*-?/���        )��P	�	���A�*

	conv_loss�1-?���        )��P	j�	���A�*

	conv_loss�=-?�z��        )��P	�6
���A�*

	conv_loss�"-?yBĆ        )��P	�g
���A�*

	conv_loss�-?`�;�        )��P	Z�
���A�*

	conv_loss'�,? �_�        )��P	�
���A�*

	conv_loss4-?�J�H        )��P	�
���A�*

	conv_lossr
-?��'        )��P	�J���A�*

	conv_loss�-?��x        )��P	H~���A�*

	conv_loss��,?��'�        )��P	F����A�*

	conv_loss��,?9V@        )��P	Y����A�*

	conv_lossX-?B�e        )��P	����A�*

	conv_lossw-?�w�        )��P	�D���A�*

	conv_loss��,?TZQ�        )��P	�y���A�*

	conv_lossA#-?���        )��P	����A�*

	conv_loss�-?ñ*        )��P	����A�*

	conv_loss��,?O�Ҝ        )��P	d���A�*

	conv_loss�-?O���        )��P	J���A�*

	conv_loss�,?n&��        )��P	?}���A�*

	conv_loss�,?��        )��P	�����A�*

	conv_loss]�,?^��        )��P	����A�*

	conv_loss��,?�nN@        )��P	@���A�*

	conv_loss��,?�մ        )��P	�G���A�*

	conv_loss_�,?�78�        )��P	_|���A�*

	conv_loss��,?Ee        )��P	Э���A�*

	conv_loss��,?WJU�        )��P	%����A�*

	conv_loss�u,?J�z}        )��P	Z���A�*

	conv_loss�,?}I�`        )��P		B���A�*

	conv_loss�,?6.�        )��P	�s���A�*

	conv_loss��,?����        )��P	|����A�*

	conv_lossx�,?�2�        )��P	=����A�*

	conv_losso�,?��W        )��P	A
���A�*

	conv_loss��,?��        )��P	�;���A�*

	conv_loss��,?�ɧ*        )��P	jm���A�*

	conv_loss��,?C�	*        )��P	����A�*

	conv_loss��,?����        )��P	2����A�*

	conv_lossk�,?
J        )��P	n���A�*

	conv_lossp�,?g�B	        )��P	�9���A�*

	conv_lossc�,?X�Nn        )��P	�l���A�*

	conv_lossJ|,?�O۶        )��P	�����A�*

	conv_losss�,?`��'        )��P	2����A�*

	conv_loss#�,?��        )��P	����A�*

	conv_loss�,?�,?        )��P	�5���A�*

	conv_loss�,?�s�.        )��P	�g���A�*

	conv_loss��,?(�gj        )��P	�����A�*

	conv_loss�,?y��        )��P	�����A�*

	conv_loss��,?oM�@        )��P	 ���A�*

	conv_lossR�,?�Q�        )��P	����A�*

	conv_loss�E,?��u�        )��P	u����A�*

	conv_lossN�,?W�5~        )��P		����A�*

	conv_loss��,?3m�        )��P	�)���A�*

	conv_loss|,?�"�i        )��P	�\���A�*

	conv_losss�,?��Y        )��P	����A�*

	conv_lossVp,?f�Ξ        )��P	1����A�*

	conv_loss��,?��k�        )��P	)����A�*

	conv_loss�g,?�۠�        )��P	0���A�*

	conv_loss��,?y'�        )��P	{e���A�*

	conv_loss�Y,?A� t        )��P	p����A�*

	conv_losstf,?�Z��        )��P	����A�*

	conv_lossH�,?���        )��P	����A�*

	conv_losseA,?r��        )��P	D���A�*

	conv_loss�,?��V<        )��P	s���A�*

	conv_lossZU,?�O��        )��P	ۡ���A�*

	conv_loss=h,?u�X        )��P	����A�*

	conv_loss ^,?��2        )��P	� ���A�*

	conv_loss�,?G��        )��P	}0���A�*

	conv_loss�%,?>��8        )��P	�`���A�*

	conv_loss�S,?�z��        )��P	����A�*

	conv_lossC_,?��l�        )��P	�����A�*

	conv_loss\,?�9�(        )��P	�����A�*

	conv_loss�<,?�Z�        )��P	�*���A�*

	conv_loss�|,?3� �        )��P	|[���A�*

	conv_lossk�+?�        )��P	ߊ���A�*

	conv_loss�`,?�H'        )��P	4����A�*

	conv_loss�K,?�6�R        )��P	7����A�*

	conv_losso?,?��        )��P	����A�*

	conv_loss�0,?���        )��P	PO���A�*

	conv_loss�4,?V��        )��P	�����A�*

	conv_loss�a,?炨�        )��P	����A�*

	conv_loss��+?��bA        )��P	�����A�*

	conv_loss��+?AYT�        )��P	9���A�*

	conv_loss��+?;��        )��P	�F���A�*

	conv_loss|�+?���O        )��P	�u���A�*

	conv_loss�B,?4#��        )��P	ݦ���A�*

	conv_loss�+?����        )��P	�����A�*

	conv_lossC,?[���        )��P	����A�*

	conv_loss1�+?ArB�        )��P	7���A�*

	conv_lossw,?4%'�        )��P	Cg���A�*

	conv_loss*,?(;.�        )��P	ؕ���A�*

	conv_loss��+?�0�        )��P	�����A�*

	conv_lossS9,?��h        )��P	����A�*

	conv_loss%�+?�u��        )��P	�-���A�*

	conv_lossX�+?*�7s        )��P	�a���A�*

	conv_loss�,?�u�        )��P	Y����A�*

	conv_loss��+?��Z        )��P	F����A�*

	conv_loss�,?֣��        )��P	����A�*

	conv_loss�,?�a*        )��P	�/���A�*

	conv_loss��+?9�7�        )��P	�x���A�*

	conv_lossY�+?Fh�h        )��P	����A�*

	conv_loss��+?��V|        )��P	K����A�*

	conv_loss��+?�BY        )��P	����A�*

	conv_loss�+?%d�         )��P	;M���A�*

	conv_loss��+?
c��        )��P	����A�*

	conv_loss��+?)�        )��P	پ���A�*

	conv_lossD�+?��Q        )��P	�����A�*

	conv_loss=�+?��        )��P	�, ���A�*

	conv_loss�~+?��)K        )��P	 a ���A�*

	conv_loss��+?4�        )��P	f� ���A�*

	conv_loss�+?7��        )��P	�� ���A�*

	conv_loss��+?}���        )��P	�!���A�*

	conv_loss��+?�y�4        )��P	HQ!���A�*

	conv_loss�a+?s[��        )��P	�!���A�*

	conv_loss��+?;\Dt        )��P	��!���A�*

	conv_loss�{+?�h�        )��P	J�!���A�*

	conv_loss��+?�.&�        )��P	8$"���A�*

	conv_loss��+?"OE�        )��P	�Y"���A�*

	conv_loss�+?�s��        )��P	+�"���A�*

	conv_loss��+?AG��        )��P	��"���A�*

	conv_loss��+?ELZx        )��P	��"���A�*

	conv_lossXZ+?��%        )��P	�:#���A�*

	conv_losswu+?@d�_        )��P	�t#���A�*

	conv_loss��+?o,,�        )��P	�#���A�*

	conv_loss�H+?׮�7        )��P	��#���A�*

	conv_loss�+?*�o        )��P	�!$���A�*

	conv_loss	W+?�׌[        )��P	|W$���A�*

	conv_loss	^+?�K�        )��P	ފ$���A�*

	conv_loss�+?\LC�        )��P	u�$���A�*

	conv_loss�l+?�~`        )��P	u�$���A�*

	conv_lossD�+?]�]        )��P	�+%���A�*

	conv_loss+G+?��A(        )��P	g`%���A�*

	conv_lossdI+?L��z        )��P	�%���A�*

	conv_lossЎ+?f�~�        )��P	*�%���A�*

	conv_loss�A+?���        )��P	 &���A�*

	conv_loss�\+?"8G�        )��P	}4&���A�*

	conv_loss
}+?:У�        )��P	i&���A�*

	conv_loss��+?���!        )��P	��&���A�*

	conv_loss=+?P<D�        )��P	��&���A�*

	conv_loss
A+?�S�        )��P	'���A�*

	conv_lossh#+?���i        )��P	�8'���A�*

	conv_loss�2+?�-��        )��P	�n'���A�*

	conv_loss,+?0�m        )��P	��'���A�*

	conv_loss5a+?l�ܲ        )��P	��'���A�*

	conv_loss�Q+?�6�        )��P	 (���A�*

	conv_lossw+?Pms�        )��P	TE(���A�*

	conv_loss��*?�:�        )��P	Gy(���A�*

	conv_lossN +?��;z        )��P	��(���A�*

	conv_loss~+?�	��        )��P	=�-���A�*

	conv_loss=+?ρE�        )��P	A�-���A�*

	conv_loss 1+?,�        )��P	�$.���A�*

	conv_loss�!+?5x        )��P	=T.���A�*

	conv_loss��*?��v�        )��P	��.���A�*

	conv_losstR+?���z        )��P	�.���A�*

	conv_loss�*?<~        )��P	��.���A�*

	conv_loss��*?�Ƌ        )��P	�!/���A�*

	conv_loss�*?U[�        )��P	"V/���A�*

	conv_loss��*?�a��        )��P	r�/���A�*

	conv_lossR+?Ca�"        )��P	b�/���A�*

	conv_losst�*?��E        )��P	��/���A�*

	conv_loss�*?�5�r        )��P	�80���A�*

	conv_loss<�*?ݝw�        )��P	�g0���A�*

	conv_loss��*?��L        )��P	b�0���A�*

	conv_loss��*?��\        )��P	{�0���A�*

	conv_loss^+?�|t�        )��P	��0���A�*

	conv_loss��*?�p�        )��P	�(1���A�*

	conv_lossX�*?�(��        )��P	�V1���A�*

	conv_loss3�*?��5        )��P	�1���A�*

	conv_losss�*?���        )��P	��1���A�*

	conv_loss��*?8/
K        )��P	��1���A�*

	conv_loss��*?�3$#        )��P	�2���A�*

	conv_loss!�*?�(V�        )��P	�N2���A�*

	conv_loss~�*?8���        )��P	�2���A�*

	conv_loss[�*?���.        )��P	��2���A�*

	conv_lossӶ*?�(�        )��P	l�2���A�*

	conv_loss/�*?H�         )��P	�3���A�*

	conv_loss�*?����        )��P	0@3���A�*

	conv_lossag*?b*�        )��P	�n3���A�*

	conv_lossi�*?�U�        )��P	L�3���A�*

	conv_lossW*?��j�        )��P	��3���A�*

	conv_loss��*?LA�0        )��P	��3���A�*

	conv_loss"2*?�$�        )��P	�*4���A�*

	conv_lossʃ*?<�        )��P	�]4���A�*

	conv_lossC�*?}i�        )��P	׋4���A�*

	conv_loss�A*?x9�        )��P	��4���A�*

	conv_loss9�*?��)        )��P	�4���A�*

	conv_loss܊*?t�0�        )��P	�5���A�*

	conv_loss*?{C��        )��P	�H5���A�*

	conv_loss�v*?T���        )��P	�x5���A�*

	conv_lossyf*?��l        )��P	��5���A�*

	conv_loss�W*?gn"+        )��P	��5���A�*

	conv_loss�6*?T7�        )��P	�6���A�*

	conv_lossIE*?���        )��P	�=6���A�*

	conv_loss�m*?9��        )��P	�o6���A�*

	conv_loss3d*?3@�        )��P	�6���A�*

	conv_loss�*?@h�        )��P	��6���A�*

	conv_lossRU*?!���        )��P	��6���A�*

	conv_loss�m*?��6R        )��P	t-7���A�*

	conv_loss�(*?KL�%        )��P	�\7���A�*

	conv_lossW*?<�+y        )��P	-�7���A�*

	conv_loss1*?Tw��        )��P	;�7���A�*

	conv_lossS*?���4        )��P	R�7���A�*

	conv_lossE*?C�W7        )��P	n.8���A�*

	conv_loss��*?���-        )��P	mc8���A�*

	conv_lossK*?�T:D        )��P	w�8���A�*

	conv_lossrY*?%� �        )��P	V�8���A�*

	conv_loss *?!�z�        )��P	p�8���A�*

	conv_loss�*?���        )��P	L"9���A�*

	conv_lossD*?�V\        )��P	�_9���A�*

	conv_loss2d*?��2#        )��P	'�9���A�*

	conv_loss*?@r,�        )��P	�9���A�*

	conv_loss�	*?�gy�        )��P	��9���A�*

	conv_loss4�)?J�NX        )��P	+:���A�*

	conv_loss��)?B�Z        )��P	`k:���A�*

	conv_loss��)?r�Ԟ        )��P	�:���A�*

	conv_loss��)?��o'        )��P	��:���A�*

	conv_loss��)?ze�        )��P	�:���A�*

	conv_loss�c)?�
�*        )��P	I+;���A�*

	conv_lossn�)?_h�        )��P	�Z;���A�*

	conv_loss�*?��Cy        )��P	I�;���A�*

	conv_lossX�)?Hئ        )��P	<�;���A�*

	conv_lossr�)?;�3        )��P	��;���A�*

	conv_loss��)?_ՄD        )��P	�<���A�*

	conv_loss��)?J"YE        )��P	�G<���A�*

	conv_lossZ�)?B�Y        )��P	Aw<���A�*

	conv_loss��)?��c�        )��P	�<���A�*

	conv_loss��)?K���        )��P	��<���A�*

	conv_loss��)?� ��        )��P	�
=���A�*

	conv_loss�L)?1;C�        )��P	�7=���A�*

	conv_loss�i)?a]"        )��P	h=���A�*

	conv_loss&�)?͊�F        )��P	��=���A�*

	conv_loss��)?ߏ�        )��P	��=���A�*

	conv_loss�)?2���        )��P	'�=���A�*

	conv_loss��)?�p��        )��P	}*>���A�*

	conv_loss�)?�T�:        )��P	�Y>���A�*

	conv_loss��)?�?d�        )��P	_�>���A�*

	conv_loss�)?G��        )��P	��>���A�*

	conv_lossAQ)?я�I        )��P	b�>���A�*

	conv_loss�)?�w��        )��P	�?���A�*

	conv_loss��)?I��        )��P	_I?���A�*

	conv_loss�)?���m        )��P	�x?���A�*

	conv_loss
w)?ɵ��        )��P	W�?���A�*

	conv_loss:�)?XO�k        )��P	��?���A�*

	conv_lossB)?��:        )��P	@���A�*

	conv_loss�)?�� �        )��P	�;@���A�*

	conv_loss��(?��5-        )��P	fj@���A�*

	conv_lossOu)?E{�        )��P	�@���A�*

	conv_loss�0)?d��u        )��P	#�@���A�*

	conv_loss�P)?��        )��P	Q�@���A�*

	conv_loss)?��+        )��P	<�B���A�*

	conv_lossFx)?����        )��P	��B���A�*

	conv_loss�7)?0��t        )��P	 �B���A�*

	conv_loss[#)?~��        )��P	�%C���A�*

	conv_loss��(?Oz        )��P	VC���A�*

	conv_loss�,)?Mwh�        )��P	��C���A�*

	conv_loss�5)?�z��        )��P	��C���A�*

	conv_loss��(?�ɮ        )��P	K�C���A�*

	conv_loss�)?��        )��P	�3D���A�*

	conv_loss��(?]���        )��P	uhD���A�*

	conv_lossT?)?��        )��P	ҧD���A�*

	conv_losssz(?��        )��P	��D���A�*

	conv_loss�(?S`�$        )��P	4E���A�*

	conv_loss��(?]�g�        )��P	3;E���A�*

	conv_loss��(? 䪡        )��P	SkE���A�*

	conv_loss,4)?���        )��P	ڛE���A�*

	conv_lossA�(?(�P�        )��P	v�E���A�*

	conv_loss6�(?x�E�        )��P	4�E���A�*

	conv_loss��(?�eX�        )��P	w-F���A�*

	conv_loss��(?rT}        )��P	c^F���A�*

	conv_loss��(?	^Y4        )��P	*�F���A�*

	conv_loss��(?� �        )��P	�F���A�*

	conv_loss�C(?�8        )��P	 �F���A�*

	conv_loss��(?V붽        )��P	�)G���A�*

	conv_lossH�(?����        )��P	�YG���A�*

	conv_loss3�(?mÎP        )��P	��G���A�*

	conv_loss9�(?�Y�        )��P	K�G���A�*

	conv_lossi(?{ʴ        )��P	��G���A�*

	conv_lossx(?��v,        )��P	�H���A�*

	conv_loss�}(?%$��        )��P	RJH���A�*

	conv_lossu"(?���        )��P	?|H���A�*

	conv_lossie(?�7�X        )��P	5�H���A�*

	conv_loss�(?A���        )��P	x�H���A�*

	conv_loss5u(?��O�        )��P	cI���A�*

	conv_lossݳ(?O��        )��P	�@I���A�*

	conv_loss�o(?��_        )��P	qqI���A�*

	conv_loss��(?x�        )��P	ΟI���A�*

	conv_loss](?��_�        )��P	X�I���A�*

	conv_loss^(?+.4v        )��P	�J���A�*

	conv_lossct(?��W�        )��P	�5J���A�*

	conv_loss(c(?h��        )��P	TeJ���A�*

	conv_loss7!(?֖7t        )��P	%�J���A�*

	conv_loss	(?�-�5        )��P	S�J���A�*

	conv_loss+u(?�+�7        )��P	"�J���A�*

	conv_loss�(?0B�        )��P	I&K���A�*

	conv_loss�~(?��.�        )��P		WK���A�*

	conv_loss�8(?��G        )��P	�K���A�*

	conv_loss��'?��9�        )��P	Z�K���A�*

	conv_lossK!(?f�r*        )��P	��K���A�*

	conv_loss$(?1��$        )��P	�"L���A�*

	conv_loss�'?=��        )��P	cfL���A�*

	conv_loss��'?�w�o        )��P	O�L���A�*

	conv_loss�K(?� ��        )��P	Q�L���A�*

	conv_lossW�'?'��<        )��P	��L���A�*

	conv_loss�(?�̯x        )��P	 +M���A�*

	conv_lossd�'?���W        )��P	W[M���A�*

	conv_lossX�'?�`L�        )��P	�M���A�*

	conv_loss��'?U�߂        )��P	�M���A�*

	conv_loss+�'?j �        )��P	G�M���A�*

	conv_lossl�'?�x�        )��P	,N���A�*

	conv_loss��'?��Xg        )��P	V^N���A�*

	conv_loss�'?ˋ��        )��P	��N���A�*

	conv_loss��'?�~��        )��P	��N���A�*

	conv_loss�-(?�_�        )��P	��N���A�*

	conv_loss�|'?d���        )��P	G%O���A�*

	conv_loss��'?� ;        )��P	kWO���A�*

	conv_loss4'?�y�        )��P	��O���A�*

	conv_loss9e'?�v�        )��P	��O���A�*

	conv_loss��'?��c        )��P	l�O���A�*

	conv_lossۧ'?υ��        )��P	�P���A�*

	conv_loss��'?�0W�        )��P	�FP���A�*

	conv_loss�'?e��        )��P	�yP���A�*

	conv_loss�j'?�~]        )��P	İP���A�*

	conv_loss�G'?����        )��P	��P���A�*

	conv_lossR�&?4Υ        )��P	Q���A�*

	conv_loss�O'?��[�        )��P	�HQ���A�*

	conv_loss�E'?�VD        )��P	rxQ���A�*

	conv_loss�8'?�"k�        )��P	��Q���A�*

	conv_loss�o'?�l
]        )��P	��Q���A�*

	conv_lossyn'?�ox        )��P	R���A�*

	conv_loss�V'?���        )��P	�5R���A�*

	conv_loss��&?Ac�        )��P	:hR���A�*

	conv_lossf+'?U��        )��P	�R���A�*

	conv_loss�8'?,�~        )��P	��R���A�*

	conv_loss�$'?�8        )��P	��R���A�*

	conv_loss�'?�c�_        )��P	�&S���A�*

	conv_lossg�&?j�        )��P	[YS���A�*

	conv_loss�'?0Tv        )��P	}�S���A�*

	conv_lossP)'?[j�        )��P	o�S���A�*

	conv_loss,'?ߕ�        )��P	��S���A�*

	conv_loss��&?�z]�        )��P	T���A�*

	conv_lossn�&?��/�        )��P	KT���A�*

	conv_lossH�&?��J�        )��P	L}T���A�*

	conv_loss�&?톔�        )��P	��T���A�*

	conv_loss�/'?"P��        )��P	'�T���A�*

	conv_loss��&?�`        )��P	�U���A�*

	conv_loss��&?"1��        )��P	�@U���A�*

	conv_losss&?'C�f        )��P	�oU���A�*

	conv_loss`V&?Ӆ�,        )��P	��U���A�*

	conv_loss�&?�HC�        )��P	��U���A�*

	conv_lossX�&?�KO        )��P	 V���A�*

	conv_loss��&?��        )��P	�FV���A�*

	conv_lossVV&?�R�        )��P	�|V���A�*

	conv_loss�&?��J�        )��P	 �V���A�*

	conv_loss;�%?�|]t        )��P	��V���A�*

	conv_lossǪ&?�Gq        )��P	�)W���A�*

	conv_loss�Y&?�i�        )��P	Y\W���A�*

	conv_lossX�&?����        )��P	P�W���A�*

	conv_lossč&?XMM        )��P	��W���A�*

	conv_loss(&?��+�        )��P	X�W���A�*

	conv_loss�i&?D?l�        )��P	�BX���A�*

	conv_loss=5&?�c�J        )��P	^~X���A�*

	conv_lossk5&?(25        )��P	_�X���A�*

	conv_loss�&?����        )��P	��X���A�*

	conv_loss:�%?x�        )��P	�Y���A�*

	conv_loss=�%?#�G�        )��P	5KY���A�*

	conv_loss�&?�!��        )��P	R�Y���A�*

	conv_loss�&?�b��        )��P	��Y���A�*

	conv_loss�t%?~<��        )��P	��Y���A�*

	conv_lossU�%?GK#�        )��P	�Z���A�*

	conv_loss�%?�x�~        )��P	�NZ���A�*

	conv_loss�%?�G�        )��P	�Z���A�*

	conv_lossi1&?�LS        )��P	�Z���A�*

	conv_loss��%?3�        )��P	g�Z���A�*

	conv_loss�%?��t�        )��P	�[���A�*

	conv_loss�%?A�`        )��P	O[���A�*

	conv_loss�j%?ck:        )��P	ـ[���A�*

	conv_loss��%?n�k�        )��P	T�[���A�*

	conv_lossx�%?�O        )��P	�[���A�*

	conv_loss�&%?�x0�        )��P	X\���A�*

	conv_loss"%?��        )��P	HN\���A�*

	conv_loss|%?���        )��P	V�\���A�*

	conv_loss�Z%?;��2        )��P	�\���A�*

	conv_loss�;%?U��P        )��P	��\���A�*

	conv_lossqa%?�߷�        )��P	�]���A�*

	conv_loss֧%?�æ�        )��P	-O]���A�*

	conv_lossS�%?��m        )��P	�]���A�*

	conv_loss%?�&z+        )��P	3�]���A�*

	conv_loss�D%?L�`X        )��P	��]���A�*

	conv_loss��$?�        )��P	�^���A�*

	conv_loss�$?p��        )��P	�N^���A�*

	conv_loss�S%?#W�        )��P	9�^���A�*

	conv_lossπ%?�|��        )��P	ʵ^���A�*

	conv_loss�P%?�7I�        )��P	��^���A�*

	conv_lossG�$?�Y�`        )��P	D_���A�*

	conv_loss��$?���        )��P	�R_���A�*

	conv_loss��$?�(~        )��P	�_���A�*

	conv_loss��$?����        )��P	3�_���A�*

	conv_loss/|%?a���        )��P	��_���A�*

	conv_loss��$?�g(�        )��P	7"`���A�*

	conv_loss�%?�Qd�        )��P	k`���A�*

	conv_lossV9$?(�?P        )��P	�`���A�*

	conv_loss��$?FH�H        )��P	��`���A�*

	conv_loss�^$?����        )��P	�a���A�*

	conv_loss�%?.�{'        )��P	�>a���A�*

	conv_loss�s$?{lM�        )��P	>�a���A�*

	conv_loss�#?�^��        )��P	$�a���A�*

	conv_loss�n$?�*-        )��P	G�a���A�*

	conv_loss�$?��3�        )��P	�(b���A�*

	conv_loss[�#?�r�        )��P	 db���A�*

	conv_lossJI$? �p        )��P	2�b���A�*

	conv_loss�%?.�=�        )��P	��b���A�*

	conv_lossO�#?$�#        )��P	c���A�*

	conv_loss��#?�;R�        )��P	b8c���A�*

	conv_loss�S$?�!��        )��P	�mc���A�*

	conv_loss4J$?S� 6        )��P	��c���A�*

	conv_loss�$?��D	        )��P	��c���A�*

	conv_lossp
$?�k\�        )��P	�d���A�*

	conv_lossˡ$?�䊲        )��P	�9d���A�*

	conv_loss�W#?�ڋ�        )��P	md���A�*

	conv_loss��#?��7        )��P	|�d���A�*

	conv_lossN5$?`T׍        )��P	��d���A�*

	conv_loss�r#?ӥ�'        )��P	�e���A�*

	conv_loss/$?���#        )��P	HMe���A�*

	conv_loss�$?��'        )��P	ԃe���A�*

	conv_loss�I$?���        )��P	��e���A�*

	conv_loss�[$?���        )��P	��e���A�*

	conv_loss'0#?��        )��P	Tf���A�*

	conv_loss%%#?���        )��P	�Of���A�*

	conv_loss��#?���~        )��P	Ѓf���A�*

	conv_loss0�#?�Y        )��P	$�f���A�*

	conv_lossx�"?��7�        )��P	D�f���A�*

	conv_loss�l#?��kG        )��P	�g���A�*

	conv_loss�]#?U�@�        )��P	�Qg���A�*

	conv_loss�L#?����        )��P	D�g���A�*

	conv_loss)$?v�#=        )��P	κg���A�*

	conv_loss
#?E��        )��P	��g���A�*

	conv_loss�$#?rf��        )��P	�h���A�*

	conv_loss�O#?�B�A        )��P	$Qh���A�*

	conv_loss�#?	�q�        )��P	��h���A�*

	conv_lossB#?�+�u        )��P	�h���A�*

	conv_loss�"?�q��        )��P	��h���A�*

	conv_lossa
#?�{�=        )��P	�i���A�*

	conv_loss��"?��}�        )��P	�Oi���A�*

	conv_loss(�"?��+        )��P	��i���A�*

	conv_loss��"?Cd��        )��P	=�i���A�*

	conv_loss��"?�?,^        )��P	��i���A�*

	conv_lossb�"?35H        )��P	^!j���A�*

	conv_lossW�"?�e�        )��P	�Uj���A�*

	conv_loss�"?�'�        )��P	�j���A�*

	conv_loss�-"?eRRu        )��P	vl���A�*

	conv_lossv�"?��        )��P	�Ll���A�*

	conv_loss�p"?��O        )��P	�l���A�*

	conv_loss�|"?�'r        )��P	U�l���A�*

	conv_lossO%"?J�^�        )��P	r�l���A�*

	conv_lossQF"?p�\�        )��P	vm���A�*

	conv_losso�!?X���        )��P	�Bm���A�*

	conv_lossM�!?w��        )��P	xsm���A�*

	conv_loss�9"?��v�        )��P	\�m���A�*

	conv_losske"?����        )��P	��m���A�*

	conv_loss{�!?MtK,        )��P	,n���A�*

	conv_loss2�!?���        )��P	�:n���A�*

	conv_loss�!??PU7        )��P	|kn���A�*

	conv_loss*"?�
X        )��P	ԛn���A�*

	conv_loss�g"?[<`        )��P	p�n���A�*

	conv_loss�!?���        )��P	�o���A�*

	conv_loss��!?mv        )��P	,Ko���A�*

	conv_loss�>!?�h_        )��P	J}o���A�*

	conv_loss�!?��L�        )��P	İo���A�*

	conv_loss�!?`��J        )��P	��o���A�*

	conv_loss��!?R���        )��P	p���A�*

	conv_loss�T!?5��Q        )��P	KAp���A�*

	conv_loss�!?3�7        )��P	op���A�*

	conv_loss, "?\�*        )��P	��p���A�*

	conv_loss�t ?��l�        )��P	��p���A�*

	conv_loss?+!?W�kF        )��P	�q���A�*

	conv_loss�l!?�pD        )��P	W?q���A�*

	conv_loss�!?'���        )��P	
qq���A�*

	conv_losso!?])�        )��P	y�q���A�*

	conv_loss�� ?J��        )��P	��q���A�*

	conv_loss�!!?��>�        )��P	�r���A�*

	conv_loss%� ?q�f        )��P	.6r���A�*

	conv_lossJJ ?^��p        )��P	Sdr���A�*

	conv_lossp� ?��q0        )��P	�r���A�*

	conv_loss�� ?�R�        )��P	�r���A�*

	conv_loss�e ?�Bl�        )��P	��r���A�*

	conv_loss<| ?����        )��P	F"s���A�*

	conv_loss�� ?����        )��P	xQs���A�*

	conv_loss�(!?����        )��P	��s���A�*

	conv_loss�� ?�tB�        )��P	Ͳs���A�*

	conv_loss"D ?����        )��P	+�s���A�*

	conv_loss��?~�&�        )��P	�t���A�*

	conv_loss�� ?����        )��P	�>t���A�*

	conv_loss��?k4I        )��P	�lt���A�*

	conv_loss. ?ְ��        )��P	��t���A�*

	conv_lossl ?�f�        )��P	��t���A�*

	conv_loss82 ?8�O         )��P	�u���A�*

	conv_loss=�?)TK�        )��P	�?u���A�*

	conv_loss��?���8        )��P	ou���A�*

	conv_loss2�?
 �        )��P	V�u���A�*

	conv_loss�$ ?����        )��P	��u���A�*

	conv_loss��?����        )��P	�v���A�*

	conv_lossO;?�>��        )��P	�:v���A�*

	conv_loss��?'���        )��P	qjv���A�*

	conv_lossbP?|dN	        )��P	�v���A�*

	conv_loss�;?CM        )��P	��v���A�*

	conv_loss�"?��=�        )��P	�w���A�*

	conv_loss�G?���        )��P	�Bw���A�*

	conv_loss??���        )��P	�ww���A�*

	conv_loss��?���        )��P	��w���A�*

	conv_loss�?Z�s�        )��P	��w���A�*

	conv_loss��?%�{        )��P	�x���A�*

	conv_lossV?��ZM        )��P	�Ax���A�*

	conv_loss�?F�3        )��P	Gox���A�*

	conv_loss4�?	��        )��P	��x���A�*

	conv_loss�?)H��        )��P	��x���A�*

	conv_lossf?���        )��P	�	y���A�*

	conv_loss�5?`-�        )��P	<y���A�*

	conv_loss�?Ow`�        )��P	Wjy���A�*

	conv_loss��?�/��        )��P	Ęy���A�*

	conv_lossr�?f�9A        )��P	��y���A�*

	conv_lossF�?��;�        )��P	��y���A�*

	conv_loss��?��9        )��P	�%z���A�*

	conv_loss	A?���        )��P	�Vz���A�*

	conv_loss��?�y��        )��P	��z���A�*

	conv_loss��?����        )��P	��z���A�*

	conv_lossq�?���        )��P	��z���A�*

	conv_loss�a?�_[h        )��P	2{���A�*

	conv_loss��?sc|        )��P	�F{���A�*

	conv_loss��?Bw?        )��P	�v{���A�*

	conv_lossN�?���\        )��P	8�{���A�*

	conv_loss��?:�We        )��P	�{���A�*

	conv_loss� ?�        )��P	�|���A�*

	conv_loss�?��        )��P	�7|���A�*

	conv_loss�?�B�|        )��P	�i|���A�*

	conv_loss�Y?�^��        )��P	5�|���A�*

	conv_loss��?8�
        )��P	D�|���A�*

	conv_loss�?Q>�C        )��P	��|���A�*

	conv_loss��?fY�        )��P	*}���A�*

	conv_loss��?a"��        )��P	.\}���A�*

	conv_loss�#?ݮ��        )��P	}���A�*

	conv_loss�s?q�i        )��P	�}���A�*

	conv_loss�?Z��j        )��P	��}���A�*

	conv_lossPP?��=        )��P	�~���A�*

	conv_loss��?;g��        )��P	!N~���A�*

	conv_loss��?��F        )��P	�}~���A�*

	conv_lossc�?J��        )��P	w�~���A�*

	conv_loss�/?��W�        )��P	(�~���A�*

	conv_losst�?
0�U        )��P	]���A�*

	conv_loss��?����        )��P	�A���A�*

	conv_loss3�?D$b�        )��P	����A�*

	conv_loss9?�P�        )��P	g����A�*

	conv_loss��?�� C        )��P	c����A�*

	conv_loss��?�[H�        )��P	'����A�*

	conv_lossf?9 ��        )��P	�S����A�*

	conv_loss��?�B�        )��P	邀���A�*

	conv_loss�,?�ۛ�        )��P	������A�*

	conv_loss��?#8�1        )��P	�����A�*

	conv_loss�x?1 z�        )��P	����A�*

	conv_lossk�? R��        )��P	V����A�*

	conv_loss̋?LԊ        )��P	G�����A�*

	conv_lossv)?�,��        )��P	������A�*

	conv_loss�(?�y!        )��P	����A�*

	conv_loss�?gH�F        )��P	�����A�*

	conv_lossR9?Q<�        )��P	+N����A�*

	conv_loss��?��k        )��P	v~����A�*

	conv_loss��?Hǝ�        )��P	/�����A�*

	conv_lossD�?�¸        )��P	m����A�*

	conv_loss~�?��w|        )��P	�����A�*

	conv_loss�s?��K�        )��P	�M����A�*

	conv_losse|?���        )��P	����A�*

	conv_lossOU?�_E�        )��P	������A�*

	conv_loss��?�W�$        )��P	w����A�*

	conv_loss��?3H�        )��P	�����A�*

	conv_loss��?�Ӧ�        )��P	�B����A�*

	conv_loss��?�2        )��P	�q����A�*

	conv_loss?�/�        )��P	�����A�*

	conv_loss��?��T        )��P	�ф���A�*

	conv_loss�?�z�/        )��P	� ����A�*

	conv_lossNv?�Kr�        )��P	�1����A�*

	conv_loss�0?���        )��P	�d����A�*

	conv_loss�l?�X        )��P	 �����A�*

	conv_loss?ౡ�        )��P	�Ʌ���A�*

	conv_loss��?�k        )��P	B�����A�*

	conv_loss�?s�1~        )��P	_+����A�*

	conv_loss�n?��
p        )��P	yZ����A�*

	conv_loss�e?�;0�        )��P	�����A�*

	conv_loss�?O5�g        )��P	n�����A�*

	conv_loss՟?�8'        )��P	�����A�*

	conv_lossF�?A]�L        )��P	�����A�*

	conv_loss+?c���        )��P	M����A�*

	conv_loss�Q?S�$        )��P	!}����A�*

	conv_loss L?��        )��P	~�����A�*

	conv_lossV?��/        )��P	�����A�*

	conv_loss�-?���s        )��P	�����A�*

	conv_loss�F?Yș        )��P	B����A�*

	conv_loss��?��9�        )��P	�r����A�*

	conv_loss�?�;^s        )��P	������A�*

	conv_loss��?V"W�        )��P	�Ո���A�*

	conv_lossJ?�"L        )��P	�����A�*

	conv_loss��?�>V�        )��P	G����A�*

	conv_loss��?��4/        )��P		w����A�*

	conv_loss��?KO        )��P	Ħ����A�*

	conv_loss�<?%lm        )��P	�׉���A�*

	conv_loss/u?����        )��P	�����A�*

	conv_loss��?��΃        )��P	�A����A�*

	conv_loss�8?�ޗ        )��P	�w����A�*

	conv_loss}?����        )��P	᱊���A�*

	conv_loss1�?oƚ        )��P	b����A�*

	conv_lossa/?�G�        )��P	� ����A�*

	conv_loss��?�Vܦ        )��P	>Q����A�*

	conv_loss�.?�"G        )��P	u�����A�*

	conv_loss�A?&.��        )��P	̹����A�*

	conv_loss�?V��        )��P	�����A�*

	conv_loss�?:b�        )��P	�����A�*

	conv_loss��?��_�        )��P	�K����A�*

	conv_loss�?۠H        )��P	�}����A�*

	conv_loss�??@i�        )��P	������A�*

	conv_loss��?����        )��P	�����A�*

	conv_loss�H?4�"        )��P	(����A�*

	conv_lossܮ?iͮg        )��P	�S����A�*

	conv_loss�8?pM��        )��P	������A�*

	conv_lossp?�"(�        )��P	������A�*

	conv_loss�
?�rsN        )��P	�����A�*

	conv_lossj?��O        )��P	M����A�*

	conv_loss% ?99�        )��P	lI����A�*

	conv_loss��?b/��        )��P	�{����A�*

	conv_lossL?��pq        )��P	H�����A�*

	conv_loss7�?d�V        )��P	������A�*

	conv_lossS4?"~��        )��P	�����A�*

	conv_lossHT?Dh�        )��P	,A����A�*

	conv_loss�?�&-V        )��P	Vu����A�*

	conv_loss ?o��&        )��P	C�����A�*

	conv_loss�?
{        )��P	�Տ���A�*

	conv_loss[�?EfG        )��P	����A�*

	conv_lossj?�ہa        )��P	�7����A�*

	conv_loss�&?�M�        )��P	�g����A�*

	conv_loss'm?u��"        )��P	������A�*

	conv_loss~�?A�        )��P	4ʐ���A�*

	conv_lossO^?�ȉg        )��P	������A�*

	conv_loss�?\5-        )��P	Q-����A�*

	conv_loss��?S ��        )��P	:]����A�*

	conv_lossZ|?�Y�        )��P	������A�*

	conv_loss�P?"� �        )��P	����A�*

	conv_lossR�?}c�        )��P	
�����A�*

	conv_loss��?���        )��P	
&����A�*

	conv_lossl�?�}        )��P	�W����A�*

	conv_loss]i?PI%�        )��P	y�����A�*

	conv_losseu?'        )��P	7�����A�*

	conv_loss͘?F��        )��P	�g����A�*

	conv_loss�v?��E        )��P	F�����A�*

	conv_loss�?/��<        )��P	�)����A�*

	conv_loss�?��R        )��P	3]����A�*

	conv_lossVj?�┌        )��P	�����A�*

	conv_loss��?,FS        )��P	������A�*

	conv_loss�?�]�        )��P	m����A�*

	conv_lossX4?��Z        )��P	"����A�*

	conv_loss��?����        )��P	�Q����A�*

	conv_lossl�?�=�        )��P	7����A�*

	conv_loss�?t�1        )��P	㮚���A�*

	conv_loss��?�(۞        )��P	Z����A�*

	conv_loss=?`R�j        )��P	t����A�*

	conv_loss(I?2%7Y        )��P	?G����A�*

	conv_lossK�?���A        )��P	�v����A�*

	conv_loss0-?<ѱ2        )��P	�����A�*

	conv_loss��?0BP        )��P	�؛���A�*

	conv_loss��?Zj��        )��P	�����A�*

	conv_loss��?�VW#        )��P	�D����A�*

	conv_loss �?�ٰ�        )��P	<v����A�*

	conv_loss҈
?u���        )��P	p�����A�*

	conv_losshn?�K��        )��P	�؜���A�*

	conv_lossX�?e\'        )��P	e	����A�*

	conv_loss��
?)*VI        )��P	�7����A�*

	conv_lossD?��H        )��P	�g����A�*

	conv_loss�
?�        )��P	������A�*

	conv_loss��?�U�}        )��P	mŝ���A�*

	conv_loss�
?Wlb|        )��P	�����A�*

	conv_loss�?��l        )��P	"����A�*

	conv_lossI?�<�        )��P	{P����A�*

	conv_loss�
?�KJ�        )��P		�����A�*

	conv_loss��
?���M        )��P	������A�*

	conv_loss�I?��        )��P	�����A�*

	conv_lossr
?�n�e        )��P	�����A�*

	conv_lossAH?�1��        )��P	xD����A�*

	conv_loss3�?ӷ�        )��P	s����A�*

	conv_lossB�?��        )��P	栟���A�*

	conv_loss/�?:�        )��P	(П���A�*

	conv_lossu�?�5?\        )��P	 ����A�*

	conv_loss��?�P��        )��P	{-����A�*

	conv_loss�1
?}R�        )��P	Q]����A�*

	conv_loss$q?p_&f        )��P	�����A�*

	conv_loss��?�{e        )��P	l�����A�*

	conv_lossR?
Uk�        )��P	a����A�*

	conv_lossWi?|��-        )��P	&����A�*

	conv_loss��?�t;�        )��P	DI����A�*

	conv_loss=X?�^7�        )��P	�x����A�*

	conv_loss�*?���F        )��P	٪����A�*

	conv_loss��?��$        )��P	Pݡ���A�*

	conv_loss��?��o�        )��P	�����A�*

	conv_loss~l?
��        )��P	�<����A�*

	conv_loss��?A��        )��P	�����A�*

	conv_loss�?�xܖ        )��P	�����A�*

	conv_loss��?�U�        )��P	�����A�*

	conv_losss�?�� �        )��P	�����A�*

	conv_loss9�?!��(        )��P	VH����A�*

	conv_lossY\?ȟ	�        )��P	X����A�*

	conv_loss��?���        )��P	ҷ����A�*

	conv_loss�?�չW        )��P	U����A�*

	conv_loss��?�0k3        )��P	�'����A�*

	conv_loss(??��`�        )��P	�^����A�*

	conv_loss3�?�t`        )��P	]�����A�*

	conv_loss��?�7�t        )��P	I¤���A�*

	conv_loss��?U.��        )��P	R����A�*

	conv_lossE;?3bU�        )��P	�$����A�*

	conv_loss��?��ځ        )��P	�`����A�*

	conv_loss8,?*�Q[        )��P	_�����A�*

	conv_loss��?��0        )��P	Z�����A�*

	conv_loss�U?:�        )��P	u����A�*

	conv_loss�0?"�7        )��P	S+����A�*

	conv_lossȭ ?���        )��P	�`����A�*

	conv_loss�6 ?�\�|        )��P	嘦���A�*

	conv_loss� ?$���        )��P	�ɦ���A�*

	conv_loss_�?m�qA        )��P	=�����A�*

	conv_lossNP�>P��R        )��P	�-����A�*

	conv_loss �>�Op        )��P	�]����A�*

	conv_loss���>�#9        )��P	������A�*

	conv_loss� ?;��        )��P	�����A�*

	conv_loss���>	c��        )��P	�����A�*

	conv_lossGT�>�ihF        )��P	I ����A�*

	conv_lossZ'?n{�        )��P	gR����A�*

	conv_loss%@�>i�        )��P	�����A�*

	conv_loss@ ?7ø        )��P	������A�*

	conv_lossH�>�.Ψ        )��P	����A�*

	conv_loss��>���2        )��P	l����A�*

	conv_lossp��>�,�A        )��P	K����A�*

	conv_lossS�>��;        )��P	�~����A�*

	conv_loss���>b\��        )��P	������A�*

	conv_loss.��>��V�        )��P	+����A�*

	conv_loss_��>�O>        )��P	:����A�*

	conv_loss���>�Ìz        )��P	C����A�*

	conv_loss2h�>Jo��        )��P	�s����A�*

	conv_loss���>˫��        )��P	������A�*

	conv_loss��> l��        )��P	�֪���A�*

	conv_loss):�>^�4        )��P	_����A�*

	conv_loss���>ɟ�)        )��P	�6����A�*

	conv_loss��>[���        )��P	mh����A�*

	conv_loss��>Z�j�        )��P	������A�*

	conv_lossɚ�>K�=V        )��P	t˫���A�*

	conv_loss��>�%�        )��P	&����A�*

	conv_loss99�>�~u�        )��P	�2����A�*

	conv_loss���>>(��        )��P	z����A�*

	conv_loss�b�>_�T�        )��P	䫬���A�*

	conv_lossQ;�>�zo}        )��P		߬���A�*

	conv_loss}V�>��xK        )��P	�����A�*

	conv_lossE��>��<%        )��P	�E����A�*

	conv_lossc��>���        )��P	�w����A�*

	conv_lossŰ�>�b�o        )��P	S�����A�*

	conv_loss��>Փ<�        )��P	N����A�*

	conv_loss���>�A�        )��P	f&����A�*

	conv_loss�k�>K�        )��P	Y����A�*

	conv_loss�u�>v�        )��P	������A�*

	conv_loss�T�>�o�        )��P	C®���A�*

	conv_lossIf�>�y�{        )��P	�����A�*

	conv_loss�A�>ޙ�/        )��P	�'����A�*

	conv_lossjc�>G�-        )��P	�Y����A�*

	conv_lossiY�>�^��        )��P	L�����A�*

	conv_loss��>��8        )��P	P�����A�*

	conv_loss���>aiyL        )��P	�����A�*

	conv_lossIl�>4Uy        )��P	@"����A�*

	conv_loss �>0=        )��P	�Y����A�*

	conv_lossc�>��n        )��P	������A�*

	conv_loss���>O'6�        )��P	Y�����A�*

	conv_loss�m�>uN�        )��P	q�����A�*

	conv_loss���>0�q�        )��P	D*����A�*

	conv_lossr��>bb��        )��P	�\����A�*

	conv_loss�_�>N].        )��P	������A�*

	conv_loss���>��e        )��P	������A�*

	conv_lossn6�>f܆        )��P	������A�*

	conv_lossD��>06�B        )��P	y%����A�*

	conv_loss���>�Yn        )��P	�V����A�*

	conv_loss=��>!�\�        )��P	׉����A�*

	conv_loss�L�>ʸ�        )��P	������A�*

	conv_loss��>���N        )��P	V�����A�*

	conv_lossZ5�>��e        )��P	P����A�*

	conv_lossH��>�8Q�        )��P	dP����A�*

	conv_loss���>'�;        )��P	������A�*

	conv_loss�R�>�3�n        )��P	ڴ����A�*

	conv_loss�@�>�̲        )��P	R����A�*

	conv_loss���>A��$        )��P	�����A�*

	conv_loss��>\?�        )��P	�J����A�*

	conv_lossS��>2W�        )��P	�}����A�*

	conv_lossJ9�>��I�        )��P	������A�*

	conv_loss}�>;]�f        )��P	�����A�*

	conv_losst��>{�u�        )��P	I����A�*

	conv_loss}��>+*ף        )��P	nF����A�*

	conv_loss���>��@�        )��P	�w����A�*

	conv_loss÷�>2E�        )��P	������A�*

	conv_lossq��>Lf�i        )��P	�ܵ���A�*

	conv_loss{�>�}�M        )��P	)����A�*

	conv_loss���>�e7        )��P	�?����A�*

	conv_loss^��>%���        )��P	������A�*

	conv_lossFb�>���        )��P	C�����A�	*

	conv_lossT"�>�s��        )��P	�����A�	*

	conv_loss���>�!Am        )��P	�����A�	*

	conv_loss�<�>��F        )��P	S����A�	*

	conv_losse��>��j        )��P	�����A�	*

	conv_loss��>ȑ��        )��P	ɸ����A�	*

	conv_loss6��>7�n        )��P	n����A�	*

	conv_loss��>�N
g        )��P	;����A�	*

	conv_loss\��>mb&        )��P	Ar����A�	*

	conv_loss�O�>�ppL        )��P	O�����A�	*

	conv_loss�p�>Ǘ6        )��P	�����A�	*

	conv_loss���>R�_>        )��P	����A�	*

	conv_loss���>Yس        )��P	D����A�	*

	conv_lossNM�>|r3        )��P	�u����A�	*

	conv_lossj��>{Ƅ�        )��P	f�����A�	*

	conv_loss���>��~�        )��P	ڹ���A�	*

	conv_lossk��>����        )��P	����A�	*

	conv_loss���>���        )��P	F����A�	*

	conv_loss���>�i�        )��P	�����A�	*

	conv_loss5�>��w�        )��P	4ź���A�	*

	conv_loss�`�>֮`�        )��P	�����A�	*

	conv_loss��>�#A2        )��P	�-����A�	*

	conv_loss6T�>��        )��P	xa����A�	*

	conv_lossX��>?��        )��P	%�����A�	*

	conv_loss���>枰!        )��P	aǻ���A�	*

	conv_loss�1�>�d�o        )��P	^�����A�	*

	conv_loss��>����        )��P	6)����A�	*

	conv_loss2��>lt�n        )��P	�Z����A�	*

	conv_loss9�>�Q�        )��P	G�����A�	*

	conv_loss�>&�]x        )��P	OƼ���A�	*

	conv_loss���>��2[        )��P	�����A�	*

	conv_loss��>\e��        )��P	.>����A�	*

	conv_loss�!�>]��        )��P	Kq����A�	*

	conv_lossG��>c�ȶ        )��P	������A�	*

	conv_lossT,�>�Kp        )��P	�׽���A�	*

	conv_loss���>8\        )��P	�	����A�	*

	conv_loss6��>�D&�        )��P	h=����A�	*

	conv_loss�p�>^hF        )��P	en����A�	*

	conv_lossV�>���@        )��P	������A�	*

	conv_loss��>h |2        )��P	Ѿ���A�	*

	conv_loss���>��!�        )��P	�����A�	*

	conv_loss��>�i�        )��P	�4����A�	*

	conv_lossY�>��)�        )��P	j����A�	*

	conv_loss?V�>@�`        )��P	9�����A�	*

	conv_loss�4�>����        )��P	̿���A�	*

	conv_loss9��>BL        )��P	�����A�	*

	conv_lossK��>}2_        )��P	�,����A�	*

	conv_loss\[�>��        )��P	�\����A�	*

	conv_lossÚ�>ae*'        )��P	�����A�	*

	conv_loss��>����        )��P	2#����A�	*

	conv_loss���>��ɹ        )��P	#V����A�	*

	conv_lossݱ�>�h        )��P	�����A�	*

	conv_loss�b�>4���        )��P	�����A�	*

	conv_loss|��>NRi#        )��P	�����A�	*

	conv_loss�r�>=9&�        )��P	�*����A�	*

	conv_loss��>�&�        )��P	C\����A�	*

	conv_loss粼>L��        )��P	������A�	*

	conv_loss(�>@D�        )��P	W�����A�	*

	conv_lossV$�>c�i        )��P	K�����A�	*

	conv_loss���>u�        )��P	69����A�	*

	conv_lossp��>Ҵr	        )��P	 s����A�	*

	conv_loss�)�>�Q�        )��P	}�����A�	*

	conv_loss���>���        )��P	�����A�	*

	conv_loss�4�>����        )��P	�����A�	*

	conv_loss�z�>i�ś        )��P	C?����A�	*

	conv_lossJa�>8�ia        )��P	�q����A�	*

	conv_loss��>~;X�        )��P	m�����A�	*

	conv_lossq��>���        )��P	N�����A�	*

	conv_loss�&�>�ԩA        )��P	����A�	*

	conv_loss��>�U&�        )��P	%4����A�	*

	conv_loss���>��#        )��P	f����A�	*

	conv_lossײ�>�)~        )��P	������A�	*

	conv_loss���>B�M�        )��P	h�����A�	*

	conv_loss���>�lN�        )��P	������A�	*

	conv_loss�ػ>�V�p        )��P	H+����A�	*

	conv_loss���>;o��        )��P	]����A�	*

	conv_loss�Ӽ>�{        )��P	<�����A�	*

	conv_loss�:�>�h:�        )��P	������A�	*

	conv_loss/P�>~���        )��P	������A�	*

	conv_lossx^�>y��d        )��P	!����A�	*

	conv_loss�(�>�ʃ        )��P	�Q����A�	*

	conv_loss��>���        )��P	 �����A�	*

	conv_loss���>�D�        )��P	������A�	*

	conv_loss��>Z��        )��P	������A�	*

	conv_loss�%�>��9        )��P	�����A�	*

	conv_lossP.�>%�~�        )��P	�F����A�	*

	conv_loss���>^~7G        )��P	�v����A�	*

	conv_loss\*�>�h�        )��P	������A�	*

	conv_lossN�>5�h        )��P	������A�	*

	conv_lossR<�>�a��        )��P	b����A�	*

	conv_loss=��>���a        )��P	�9����A�	*

	conv_loss���>?�-        )��P	�j����A�	*

	conv_loss��>�u��        )��P	������A�	*

	conv_loss��>��1Y        )��P	"�����A�	*

	conv_lossW�>:��@        )��P	I�����A�	*

	conv_loss{!�>���F        )��P	�/����A�	*

	conv_loss7�>g��0        )��P	vc����A�	*

	conv_loss�>���        )��P	�����A�	*

	conv_loss?��>(G��        )��P	}�����A�	*

	conv_loss� �>�q        )��P	�����A�	*

	conv_lossI�>�ϴ<        )��P	�8����A�	*

	conv_loss�t�>U��        )��P	�i����A�	*

	conv_loss�=�>���5        )��P	������A�	*

	conv_lossZ�>ޓ�|        )��P	������A�	*

	conv_lossJ�>�RD�        )��P	����A�	*

	conv_lossUƷ>����        )��P	�3����A�	*

	conv_lossd�>mi@8        )��P	�m����A�	*

	conv_loss.4�>�J�!        )��P	p�����A�	*

	conv_loss\9�>���        )��P	;�����A�	*

	conv_loss���>`Xg�        )��P	�����A�	*

	conv_loss3�>�WG        )��P	
=����A�	*

	conv_loss��>$�        )��P	˂����A�	*

	conv_loss_ߵ>Ͻ�        )��P	������A�	*

	conv_loss9q�>&C�        )��P	f�����A�	*

	conv_loss}˷>@��        )��P	�%����A�	*

	conv_loss���>�q�        )��P	�V����A�	*

	conv_lossh�>�3{�        )��P	������A�	*

	conv_loss��>�        )��P	������A�	*

	conv_loss�m�>'L�]        )��P	������A�	*

	conv_loss�߶>Ϫ��        )��P	����A�	*

	conv_loss?��>bJ        )��P	�M����A�	*

	conv_loss�&�>�G�V        )��P	�����A�	*

	conv_lossï�>�wQ        )��P	������A�	*

	conv_lossCԸ>dU        )��P	9�����A�	*

	conv_loss��>���&        )��P	a����A�	*

	conv_lossX�>�h�P        )��P	�A����A�	*

	conv_loss{´>D
`�        )��P	:v����A�	*

	conv_loss���>�8zo        )��P	������A�	*

	conv_loss"¶>NI        )��P	������A�
*

	conv_loss�n�>���5        )��P	����A�
*

	conv_lossp��>�q�[        )��P	�;����A�
*

	conv_loss��>�Α        )��P	�m����A�
*

	conv_loss�Ҷ>Z��        )��P	A�����A�
*

	conv_losss;�>��M        )��P	������A�
*

	conv_loss|+�>���        )��P	�����A�
*

	conv_loss��>��\        )��P	�6����A�
*

	conv_lossj2�>��q        )��P	Qf����A�
*

	conv_loss�ж>��$        )��P	:�����A�
*

	conv_lossç�>�0)[        )��P	������A�
*

	conv_lossuQ�>�!ў        )��P	\�����A�
*

	conv_loss��>��P�        )��P	�*����A�
*

	conv_lossUƳ>��F        )��P	;^����A�
*

	conv_loss�(�>"+�        )��P	������A�
*

	conv_loss���>n��        )��P	ž����A�
*

	conv_lossQ�>�@l�        )��P	������A�
*

	conv_loss���>Tp9        )��P	� ����A�
*

	conv_loss�-�>L-E        )��P	@R����A�
*

	conv_loss� �>K6+0        )��P	ل����A�
*

	conv_loss�q�>��e�        )��P	�����A�
*

	conv_loss�W�> �N        )��P	�����A�
*

	conv_loss���>���         )��P	�,����A�
*

	conv_loss�>0��        )��P	+]����A�
*

	conv_loss��>����        )��P	������A�
*

	conv_losst��>���i        )��P	������A�
*

	conv_loss{��>&�ɽ        )��P	�����A�
*

	conv_loss�U�>��i        )��P	�-����A�
*

	conv_loss�X�>�?Q�        )��P	�`����A�
*

	conv_loss]�>��kS        )��P	������A�
*

	conv_loss�X�>^���        )��P	L�����A�
*

	conv_loss̥�>�[        )��P	�����A�
*

	conv_lossG��>��D        )��P	�6����A�
*

	conv_lossN�>��*        )��P	,k����A�
*

	conv_loss��>��k�        )��P	h�����A�
*

	conv_loss쬳>�~�        )��P	������A�
*

	conv_loss�q�>�Xx        )��P	�����A�
*

	conv_lossf2�>"C�M        )��P	�D����A�
*

	conv_loss�)�>���`        )��P	�v����A�
*

	conv_loss���>ћ|�        )��P	�����A�
*

	conv_loss�w�>rs|        )��P	������A�
*

	conv_loss�y�>2���        )��P	�����A�
*

	conv_loss���>���        )��P	A����A�
*

	conv_loss�ǯ>��3�        )��P	�r����A�
*

	conv_lossX�>q�6�        )��P	9�����A�
*

	conv_lossd}�>����        )��P	������A�
*

	conv_loss�(�>p���        )��P	I����A�
*

	conv_loss�*�>���        )��P	�<����A�
*

	conv_loss�߱>�Y��        )��P	�m����A�
*

	conv_lossY��>�ʍ�        )��P	������A�
*

	conv_loss�>\޻        )��P	������A�
*

	conv_loss�p�>��        )��P	�����A�
*

	conv_lossk��>n��        )��P	�5����A�
*

	conv_lossDư>�{��        )��P	�g����A�
*

	conv_lossJ��>(d
}        )��P	�����A�
*

	conv_lossiޱ>��^�        )��P	������A�
*

	conv_lossL��>,\&        )��P	�����A�
*

	conv_loss��>��g4        )��P	 0����A�
*

	conv_lossr�>!Q��        )��P	a����A�
*

	conv_loss|�>��@�        )��P	*�����A�
*

	conv_loss1�>�w�F        )��P	������A�
*

	conv_loss�Ʊ>+N�;        )��P	������A�
*

	conv_lossD�>�O�        )��P	�)����A�
*

	conv_lossc{�>��,�        )��P	�Z����A�
*

	conv_loss.�>���        )��P	A�����A�
*

	conv_loss)��>�H��        )��P	������A�
*

	conv_loss:Ӵ>-���        )��P	������A�
*

	conv_lossz�><��        )��P	w$����A�
*

	conv_lossl4�>tjG�        )��P	=X����A�
*

	conv_loss���>��x�        )��P	������A�
*

	conv_lossQ��>��G�        )��P	?�����A�
*

	conv_loss�K�>:��        )��P	�����A�
*

	conv_loss���><�>�        )��P	�5����A�
*

	conv_loss�h�>s��        )��P	gg����A�
*

	conv_lossv��>lr|        )��P	͜����A�
*

	conv_loss�6�>���        )��P	�����A�
*

	conv_lossҮ>���        )��P	�����A�
*

	conv_loss5ڬ>�6�        )��P	�6����A�
*

	conv_loss�>N���        )��P	=g����A�
*

	conv_loss�a�>-�O        )��P	�����A�
*

	conv_loss�>��/�        )��P	������A�
*

	conv_loss"��>mR�
        )��P	�����A�
*

	conv_loss4�>ے��        )��P	m9����A�
*

	conv_loss�Y�>���3        )��P	�j����A�
*

	conv_loss�j�>�j׮        )��P	������A�
*

	conv_loss��>�_        )��P	l�����A�
*

	conv_lossᛲ>�'        )��P	�����A�
*

	conv_losse��>���e        )��P	[G����A�
*

	conv_loss'E�>�==        )��P	�z����A�
*

	conv_losss��>�R�        )��P	D�����A�
*

	conv_loss���>f�q�        )��P	������A�
*

	conv_loss�Q�>�K�j        )��P	�����A�
*

	conv_loss��>}���        )��P	O����A�
*

	conv_loss��>F��        )��P	�����A�
*

	conv_loss#�>C��        )��P	0�����A�
*

	conv_lossd��>V6d�        )��P	�����A�
*

	conv_loss���>+z�=        )��P	�%����A�
*

	conv_loss]�>�5��        )��P	�`����A�
*

	conv_loss��>:��        )��P	;�����A�
*

	conv_loss��>C�+�        )��P	E�����A�
*

	conv_loss��>��ԯ        )��P	������A�
*

	conv_loss*��>Ɛ�        )��P	,����A�
*

	conv_loss&��>���>        )��P	�_����A�
*

	conv_loss���>W�ԋ        )��P	@�����A�
*

	conv_loss���>�y��        )��P	�����A�
*

	conv_loss΀�>��K        )��P	������A�
*

	conv_lossx��>�2B�        )��P	�-����A�
*

	conv_loss���>p�/�        )��P	N`����A�
*

	conv_loss��>Qrvo        )��P	d�����A�
*

	conv_loss�5�>�.        )��P	[�����A�
*

	conv_lossp&�>�;m�        )��P	J�����A�
*

	conv_loss�s�>4��        )��P	z)����A�
*

	conv_loss�l�>�9        )��P	�[����A�
*

	conv_loss�Ӯ>2��        )��P	b�����A�
*

	conv_lossv-�>n_�K        )��P	������A�
*

	conv_loss�Ѱ>w�D        )��P	d�����A�
*

	conv_loss��>�]0�        )��P	�&����A�
*

	conv_loss(�>��o        )��P	�Y����A�
*

	conv_lossW��>�O��        )��P	������A�
*

	conv_lossX:�>�/m        )��P	������A�
*

	conv_loss�J�>v@�        )��P	�����A�
*

	conv_loss�ѯ>z<��        )��P	N�����A�
*

	conv_lossc�>\��Z        )��P	9�����A�
*

	conv_loss=�>�*n�        )��P	(�����A�
*

	conv_loss��>Qݢ        )��P	�$����A�
*

	conv_loss�p�>2~�        )��P	}X����A�
*

	conv_lossÅ�>Xᬓ        )��P	\�����A�
*

	conv_loss�M�>s#�3        )��P	������A�
*

	conv_loss�-�>�j�_        )��P	�����A�*

	conv_loss8U�>;�;�        )��P	�Q����A�*

	conv_loss�4�>ʴ�        )��P	������A�*

	conv_lossC��>MyE�        )��P	������A�*

	conv_loss�۱>pN�(        )��P	������A�*

	conv_lossY��>���D        )��P	�����A�*

	conv_loss�A�>�w        )��P	X����A�*

	conv_loss�q�>3�u�        )��P	�����A�*

	conv_loss�E�>�Bf        )��P	������A�*

	conv_loss���>��fi        )��P	B�����A�*

	conv_loss��>)E�        )��P	2&����A�*

	conv_loss�ˮ>�� }        )��P	a����A�*

	conv_loss�ٱ>iM�x        )��P	������A�*

	conv_loss=��>�S�        )��P	������A�*

	conv_loss��>����        )��P	� ����A�*

	conv_loss�q�>��Q�        )��P	2����A�*

	conv_loss��>R6�        )��P	�e����A�*

	conv_lossݒ�>�g��        )��P	0�����A�*

	conv_loss,c�>&	h�        )��P	������A�*

	conv_loss0V�>�9��        )��P	������A�*

	conv_lossa��>�qSS        )��P	�.����A�*

	conv_loss�s�>4�%        )��P	�`����A�*

	conv_loss�3�>�o$        )��P	7�����A�*

	conv_loss�ծ>ن��        )��P	������A�*

	conv_lossY+�>��        )��P	<�����A�*

	conv_loss���>�ޒQ        )��P	�*����A�*

	conv_lossy��>ٰ�        )��P	R\����A�*

	conv_lossI�>i�a�        )��P	 �����A�*

	conv_loss ��>�ܱ        )��P	������A�*

	conv_loss���>�ޜk        )��P	������A�*

	conv_loss:�>�yqo        )��P	�*����A�*

	conv_loss �>}h�\        )��P	�]����A�*

	conv_loss��>~ ʑ        )��P	?�����A�*

	conv_loss��><w�        )��P	�����A�*

	conv_loss+а>.;��        )��P	������A�*

	conv_loss��>Ao]        )��P	�(����A�*

	conv_lossiw�>e��        )��P	�Y����A�*

	conv_loss.ܬ>x�Ƌ        )��P	������A�*

	conv_lossC�>4�r        )��P	a�����A�*

	conv_lossuF�>6���        )��P	W�����A�*

	conv_loss��>.�Wa        )��P	 "����A�*

	conv_lossxF�>�Y�        )��P	�S����A�*

	conv_loss��>�ڨ        )��P	������A�*

	conv_loss^I�>>�y�        )��P	�����A�*

	conv_lossX^�>� �        )��P	�����A�*

	conv_loss��>4��M        )��P	�0����A�*

	conv_loss]�>��"�        )��P	�f����A�*

	conv_loss,�>�-�        )��P	%�����A�*

	conv_lossM��>5���        )��P	������A�*

	conv_loss{T�>܆�K        )��P	�����A�*

	conv_loss#�>6��        )��P	G����A�*

	conv_loss9�>_	�        )��P	������A�*

	conv_loss�5�>�VdI        )��P	������A�*

	conv_loss���>�c��        )��P	������A�*

	conv_loss���>����        )��P	x����A�*

	conv_loss$�>5�        )��P	/T����A�*

	conv_loss�Y�>;�~        )��P	������A�*

	conv_loss��>���T        )��P	1�����A�*

	conv_loss�A�>Gl��        )��P	������A�*

	conv_loss��>r�9�        )��P	.����A�*

	conv_lossET�>I��?        )��P	f����A�*

	conv_lossށ�>}��{        )��P	������A�*

	conv_lossɲ�>=^�        )��P	������A�*

	conv_loss�ޯ>� �        )��P	�����A�*

	conv_loss���>���        )��P	�7����A�*

	conv_loss_r�>�ʕ        )��P	�k����A�*

	conv_loss�٬>��w	        )��P	"�����A�*

	conv_lossxg�>��c@        )��P	�����A�*

	conv_lossȪ�>9l�}        )��P	� ����A�*

	conv_losseŮ>8v��        )��P	F3����A�*

	conv_loss�ڬ>X�        )��P	]f����A�*

	conv_losso�>���;        )��P	`�����A�*

	conv_lossQ�>��M        )��P	#�����A�*

	conv_loss���>q+�P        )��P	������A�*

	conv_loss��>�5�|        )��P	1����A�*

	conv_loss0Z�>Lݞ        )��P	_b����A�*

	conv_loss��>�j�i        )��P	������A�*

	conv_lossDխ>��M�        )��P	 �����A�*

	conv_lossxl�>���        )��P	P�����A�*

	conv_loss���>�k)�        )��P	�/����A�*

	conv_lossW��>1[G        )��P	�a����A�*

	conv_loss���>	0�        )��P	'�����A�*

	conv_loss~�>n�J�        )��P	A�����A�*

	conv_lossO��>�F        )��P	������A�*

	conv_loss�ĭ>�^��        )��P	�*����A�*

	conv_loss�&�>jA8        )��P	�[����A�*

	conv_loss���>K�j        )��P	�����A�*

	conv_loss�@�>�E�Q        )��P	M�����A�*

	conv_loss�/�>�{��        )��P	!�����A�*

	conv_loss�"�>\'�        )��P	�(����A�*

	conv_loss�b�>f��        )��P	�Y����A�*

	conv_loss6�>���        )��P	[�����A�*

	conv_lossJʫ>�Kl        )��P	�B���A�*

	conv_loss�֬>1?*�        )��P	����A�*

	conv_loss��>c�|�        )��P	M����A�*

	conv_loss�խ>�[c        )��P	3����A�*

	conv_loss�߫>���        )��P	����A�*

	conv_lossge�>�@d�        )��P	�J���A�*

	conv_loss�+�>`�v�        )��P	z���A�*

	conv_lossD�>`]}        )��P	֨���A�*

	conv_loss\��>'3��        )��P	!����A�*

	conv_loss#�>t���        )��P	� ���A�*

	conv_loss��>�0��        )��P	kW���A�*

	conv_loss� �>�WL�        )��P	����A�*

	conv_lossmǮ>fT�E        )��P	�����A�*

	conv_loss5�>���        )��P	,����A�*

	conv_loss���>��N
        )��P	�"���A�*

	conv_lossz"�>^�ӳ        )��P	�P���A�*

	conv_loss�H�>H�c        )��P	�~���A�*

	conv_loss�{�>�0F�        )��P	7����A�*

	conv_loss�>b-        )��P	����A�*

	conv_loss���>�2�(        )��P	����A�*

	conv_loss֖�>�4�?        )��P	;���A�*

	conv_losss�>�� ;        )��P	�s���A�*

	conv_loss��>��5        )��P	����A�*

	conv_loss*C�>����        )��P	�����A�*

	conv_loss7�>�'1P        )��P	~	���A�*

	conv_loss쟮>�⽟        )��P	�6	���A�*

	conv_loss6F�>}%w�        )��P	�f	���A�*

	conv_loss]��>�C�        )��P	 �	���A�*

	conv_loss;��>���        )��P	Q�	���A�*

	conv_loss5;�>0`        )��P	)�	���A�*

	conv_loss;�>|L�        )��P	.
���A�*

	conv_loss��>�ߨK        )��P	�]
���A�*

	conv_losss�>b�!�        )��P	R�
���A�*

	conv_loss�m�>�4�        )��P	��
���A�*

	conv_loss_�>:|�8        )��P	��
���A�*

	conv_loss y�>c�fa        )��P	;0���A�*

	conv_lossB�>[k��        )��P	�^���A�*

	conv_loss�P�>��L@        )��P	�����A�*

	conv_loss	[�>�'x        )��P	����A�*

	conv_lossb �>:��7        )��P	1����A�*

	conv_loss���>��        )��P	����A�*

	conv_loss�]�>��U�        )��P	�L���A�*

	conv_loss���>���f        )��P	6z���A�*

	conv_loss��>K���        )��P	0����A�*

	conv_loss#y�>BR�        )��P	�����A�*

	conv_loss�f�>y��        )��P	�	���A�*

	conv_loss��>�ò        )��P	�;���A�*

	conv_lossp��>BH�        )��P	el���A�*

	conv_lossv��>���        )��P	�����A�*

	conv_loss+��>�x�        )��P	�����A�*

	conv_loss%�>j���        )��P	-����A�*

	conv_loss8��>�-E-        )��P	�9���A�*

	conv_loss���>���        )��P	i���A�*

	conv_loss.��>���8        )��P	����A�*

	conv_lossש�>{N�         )��P	j����A�*

	conv_loss�`�>�,Q        )��P	�����A�*

	conv_loss�۬>a��E        )��P	k-���A�*

	conv_lossQ�><O        )��P	}e���A�*

	conv_lossR&�>T�B        )��P	ە���A�*

	conv_loss���>�95        )��P	c����A�*

	conv_lossr��>#��m        )��P	���A�*

	conv_loss��>H�z        )��P	tL���A�*

	conv_loss�T�>PQ�        )��P	�|���A�*

	conv_loss=5�>�|UK        )��P	�����A�*

	conv_loss��>c��]        )��P	�����A�*

	conv_lossGV�>�ށ�        )��P	4���A�*

	conv_loss�n�><{�z        )��P	�5���A�*

	conv_loss�̮>�e��        )��P	�c���A�*

	conv_loss�m�>���        )��P	�����A�*

	conv_loss1I�>�	��        )��P	�����A�*

	conv_loss,��>�h�4        )��P	T����A�*

	conv_lossq�>U�I3        )��P	*"���A�*

	conv_loss���>�b۬        )��P	$P���A�*

	conv_loss�ʩ>�q�I        )��P	C����A�*

	conv_loss}W�>�Ҥ\        )��P	�����A�*

	conv_loss"�>q�        )��P	�����A�*

	conv_loss��>�Vw        )��P	� ���A�*

	conv_loss
լ>�t˦        )��P	Q���A�*

	conv_loss"�>G��|        )��P	�����A�*

	conv_loss�!�>VI�        )��P	}����A�*

	conv_lossu��>��4�        )��P	�����A�*

	conv_loss�}�>nN#        )��P	����A�*

	conv_loss�ح>P0        )��P	}?���A�*

	conv_loss�Y�>�Q�        )��P	�q���A�*

	conv_loss��>5X7        )��P	����A�*

	conv_loss��>tO��        )��P	W����A�*

	conv_losstΫ>G"	,        )��P	� ���A�*

	conv_loss���>E�mc        )��P	C1���A�*

	conv_lossD۫> �        )��P	c`���A�*

	conv_lossqT�>-��        )��P	ܐ���A�*

	conv_losscD�>\�mg        )��P	\����A�*

	conv_loss���>k���        )��P	F����A�*

	conv_loss��>��F�        )��P	����A�*

	conv_loss{Ȫ>�{Ѐ        )��P	�K���A�*

	conv_loss��>.~S        )��P	\{���A�*

	conv_loss�;�>�Y�`        )��P	m����A�*

	conv_loss�ĭ>��        )��P	g����A�*

	conv_loss��>VR�,        )��P	�	���A�*

	conv_lossI�>o�2�        )��P	�9���A�*

	conv_loss�=�>f�|m        )��P	Ll���A�*

	conv_loss5b�>���&        )��P	I����A�*

	conv_losse=�>r�]        )��P	.���A�*

	conv_loss�ի>��K        )��P		`���A�*

	conv_loss�ޫ>�ØH        )��P	�����A�*

	conv_lossҥ�>��G        )��P	�����A�*

	conv_loss:��>�Ά4        )��P	)����A�*

	conv_loss��>e�f        )��P	�1���A�*

	conv_lossU��>˅Q�        )��P	Ie���A�*

	conv_loss�l�>[}}        )��P	S����A�*

	conv_loss���>����        )��P	�����A�*

	conv_losss�>\�O�        )��P	�	���A�*

	conv_loss�G�>%.{        )��P	.9���A�*

	conv_loss���>@/�        )��P	�h���A�*

	conv_loss~��>�[��        )��P	�����A�*

	conv_lossP��>��t�        )��P	�����A�*

	conv_loss�d�>7�N�        )��P	q���A�*

	conv_loss��>P,Ԡ        )��P	#0���A�*

	conv_loss�8�>ڏ�v        )��P	a���A�*

	conv_loss���>�n)        )��P	����A�*

	conv_loss���> ��g        )��P	P����A�*

	conv_loss;��>_g��        )��P	����A�*

	conv_loss
�>%��        )��P	�5���A�*

	conv_lossX%�>L�rq        )��P	e���A�*

	conv_loss��>r{z        )��P	V����A�*

	conv_lossw��>�h|        )��P	�����A�*

	conv_loss�׬>QDՒ        )��P	�����A�*

	conv_loss���>�A�        )��P	T%���A�*

	conv_loss�1�>��e�        )��P	6W���A�*

	conv_lossp2�>���T        )��P	�����A�*

	conv_loss�U�>�u8        )��P	����A�*

	conv_loss\�>���        )��P	�����A�*

	conv_lossFU�>�z�        )��P	����A�*

	conv_lossu�>*��        )��P	�I���A�*

	conv_lossQ�>��y        )��P	�y���A�*

	conv_loss��>Y�gc        )��P	����A�*

	conv_loss�q�>Dġ        )��P	A����A�*

	conv_loss]��>,_�        )��P	� ���A�*

	conv_lossދ�>S��6        )��P	B= ���A�*

	conv_lossW;�>���8        )��P	pp ���A�*

	conv_loss��>.֮�        )��P	� ���A�*

	conv_loss%O�>�Ы�        )��P	�� ���A�*

	conv_loss.:�>�Q        )��P	)!���A�*

	conv_loss���>C�!        )��P	:2!���A�*

	conv_loss��>��%�        )��P	=`!���A�*

	conv_loss��>��)        )��P	��!���A�*

	conv_loss��>����        )��P	m�!���A�*

	conv_lossua�>QC        )��P	�!���A�*

	conv_loss��>/�        )��P	�'"���A�*

	conv_loss�>���	        )��P	�W"���A�*

	conv_loss@!�>D��        )��P	W�"���A�*

	conv_loss�Ȭ>~AgP        )��P	߶"���A�*

	conv_lossVЩ>�!�        )��P	m�"���A�*

	conv_lossz��>Z�        )��P	�+#���A�*

	conv_lossћ�>�{�        )��P	(\#���A�*

	conv_loss/l�>�	�c        )��P	�#���A�*

	conv_lossȺ�>��X        )��P	@�#���A�*

	conv_loss�ǫ>�_C�        )��P	$���A�*

	conv_lossXH�>���        )��P	�<$���A�*

	conv_lossJ�>?�X        )��P	-r$���A�*

	conv_lossNh�>���        )��P	[�$���A�*

	conv_loss0��>y���        )��P	��$���A�*

	conv_loss�y�>�T��        )��P	�%���A�*

	conv_lossf}�>��0        )��P	�=%���A�*

	conv_loss��>�^��        )��P	�m%���A�*

	conv_loss�U�>�8*�        )��P	��%���A�*

	conv_loss���>�+�        )��P	l�%���A�*

	conv_loss?��>fk�        )��P	�&���A�*

	conv_loss���>�oM_        )��P	RB&���A�*

	conv_lossa��>��sE        )��P	�u&���A�*

	conv_loss���>*�܀        )��P	��&���A�*

	conv_loss���>4�#�        )��P	��&���A�*

	conv_loss���>
!�        )��P	�'���A�*

	conv_loss��>�	        )��P	�I'���A�*

	conv_lossٻ�>E,�        )��P	y{'���A�*

	conv_loss�.�>�ۧ�        )��P	]�'���A�*

	conv_loss��>FoO        )��P	��'���A�*

	conv_loss-��>v`�        )��P	(���A�*

	conv_lossN�>��*�        )��P	�B(���A�*

	conv_loss���>���        )��P	?u(���A�*

	conv_lossF��>��]�        )��P	L�(���A�*

	conv_loss=B�>��b        )��P	Y�(���A�*

	conv_losslo�>�U        )��P	�)���A�*

	conv_loss���>��2R        )��P	�3)���A�*

	conv_loss&J�>�V�{        )��P	�b)���A�*

	conv_loss�|�>��}        )��P	o�)���A�*

	conv_lossa�>�L	        )��P	��)���A�*

	conv_loss�Ы>��`/        )��P	��)���A�*

	conv_lossj��>�b��        )��P	�*���A�*

	conv_loss��>c�S�        )��P	#N*���A�*

	conv_loss��>6���        )��P	�}*���A�*

	conv_loss�B�>~4        )��P	��*���A�*

	conv_loss��>�"        )��P	�*���A�*

	conv_lossp�>p"�        )��P	�+���A�*

	conv_loss�)�>p��        )��P	�=+���A�*

	conv_loss�ר>�8�X        )��P	kl+���A�*

	conv_loss��>��        )��P	��+���A�*

	conv_loss�H�>�p6�        )��P	��+���A�*

	conv_loss1��>��Y        )��P	��+���A�*

	conv_lossx�>�V^�        )��P	V&,���A�*

	conv_loss_��>a��[        )��P	>W,���A�*

	conv_lossSF�>�s}�        )��P	��,���A�*

	conv_lossҦ�>�~�        )��P	�,���A�*

	conv_loss���>��        )��P	\�,���A�*

	conv_lossh��>���        )��P	�(-���A�*

	conv_loss��>�(        )��P	�Y-���A�*

	conv_loss?ѧ> TXC        )��P	e�-���A�*

	conv_loss�>Ѱ��        )��P	e�-���A�*

	conv_lossկ�>�/        )��P	=�-���A�*

	conv_loss"��>V�r�        )��P	y,.���A�*

	conv_loss�>ďX�        )��P	6j.���A�*

	conv_loss���>���        )��P	��.���A�*

	conv_loss�x�>�0�        )��P	6�.���A�*

	conv_loss�X�> �b�        )��P	� /���A�*

	conv_losss��>���P        )��P	7/���A�*

	conv_losso��>�3�@        )��P	
h/���A�*

	conv_loss��>��Z        )��P	Õ/���A�*

	conv_loss�:�>/��        )��P	l�/���A�*

	conv_lossf��>4]�[        )��P	 �/���A�*

	conv_lossO��>�>�        )��P	!0���A�*

	conv_lossu۫>��        )��P	�R0���A�*

	conv_loss�ݩ>���        )��P	��0���A�*

	conv_loss���>��f        )��P	s�0���A�*

	conv_loss.R�>�t4        )��P	I�0���A�*

	conv_loss�%�>>n{        )��P	�+1���A�*

	conv_loss��>�I��        )��P	�Z1���A�*

	conv_loss-h�>R���        )��P	�1���A�*

	conv_loss�)�>.s�E        )��P	��1���A�*

	conv_loss1�>�zm        )��P	��1���A�*

	conv_loss@ �>o�        )��P	n2���A�*

	conv_loss4��>�]
6        )��P	�E2���A�*

	conv_lossVw�>|���        )��P	+u2���A�*

	conv_loss㉪>�K�        )��P	��2���A�*

	conv_lossu�>�տ        )��P	s�2���A�*

	conv_loss�Z�>�ի�        )��P	��2���A�*

	conv_loss�ݪ>`{�t        )��P	/-3���A�*

	conv_loss&��>�;�        )��P	�\3���A�*

	conv_loss&p�>�Y�,        )��P	؋3���A�*

	conv_loss�><F�        )��P	�3���A�*

	conv_loss�b�>[Zr�        )��P	��3���A�*

	conv_loss��>q~�        )��P	�4���A�*

	conv_loss"E�>���        )��P	+F4���A�*

	conv_loss�e�>���        )��P	�u4���A�*

	conv_loss�f�>P��        )��P	��4���A�*

	conv_lossf��>�΃        )��P	��4���A�*

	conv_loss@��>��+O        )��P	��4���A�*

	conv_loss ˭>��ޭ        )��P	�-5���A�*

	conv_loss��>ƒ%E        )��P	�\5���A�*

	conv_loss�%�>椈�        )��P	�5���A�*

	conv_loss���>���        )��P	��5���A�*

	conv_loss��>�ੲ        )��P	��5���A�*

	conv_losscz�>�l_-        )��P	�6���A�*

	conv_loss]Y�>�<7�        )��P	�Y6���A�*

	conv_lossF��>�7]        )��P	ň6���A�*

	conv_loss���>|� �        )��P	��6���A�*

	conv_loss+��>[Te        )��P	��6���A�*

	conv_loss���>$��8        )��P	 7���A�*

	conv_loss��>���D        )��P	�Y7���A�*

	conv_lossQ�>^��        )��P	1�7���A�*

	conv_loss�i�>���        )��P	�7���A�*

	conv_loss��>�=s        )��P	~�7���A�*

	conv_losse'�>�X%�        )��P	�-8���A�*

	conv_loss���>�zq        )��P	p8���A�*

	conv_lossB�>j~        )��P	��8���A�*

	conv_lossM�>`l�<        )��P	]�8���A�*

	conv_lossI��>�n�        )��P	�9���A�*

	conv_loss(Ĭ><�G	        )��P	B9���A�*

	conv_loss��>T��M        )��P	�q9���A�*

	conv_loss䢨>t�U        )��P	��9���A�*

	conv_loss�u�>�넆        )��P	M�9���A�*

	conv_loss\�>�~�        )��P	e:���A�*

	conv_loss�"�>
7H        )��P	�4:���A�*

	conv_loss[��>wc!�        )��P	Kd:���A�*

	conv_loss��>
"H        )��P	U�:���A�*

	conv_loss���>�S        )��P	a�:���A�*

	conv_loss�ک>Ch��        )��P	��:���A�*

	conv_lossf6�>��=*        )��P	'$;���A�*

	conv_loss��>#��        )��P	�R;���A�*

	conv_loss�w�>I��        )��P	܂;���A�*

	conv_loss��>J��        )��P	�;���A�*

	conv_loss ح>���        )��P	!�;���A�*

	conv_lossޫ>��0�        )��P	�<���A�*

	conv_loss!ک>����        )��P	�=<���A�*

	conv_loss�ި>?g�e        )��P	p<���A�*

	conv_loss뮫>��D        )��P	*�<���A�*

	conv_loss�*�>���        )��P	=�<���A�*

	conv_loss?è>hF�        )��P	�=���A�*

	conv_loss��>���        )��P	�3=���A�*

	conv_loss�/�>i��b        )��P	�c=���A�*

	conv_lossl�>��י        )��P	ߔ=���A�*

	conv_lossr��>�z��        )��P	X�=���A�*

	conv_lossy��>	}�        )��P	��=���A�*

	conv_loss���>ʆ        )��P	�'>���A�*

	conv_loss�L�>�i��        )��P	cY>���A�*

	conv_loss�?�>U��        )��P	q�>���A�*

	conv_loss�"�>�T��        )��P	i�>���A�*

	conv_loss!��><̷        )��P	��>���A�*

	conv_loss%��>���        )��P	�?���A�*

	conv_loss�y�>j�^        )��P	�M?���A�*

	conv_loss1�>�"�        )��P	�~?���A�*

	conv_loss��>LQ��        )��P	A�?���A�*

	conv_loss"2�>~e+�        )��P	��?���A�*

	conv_loss�ͪ>��        )��P	�qA���A�*

	conv_lossq�>!���        )��P	��A���A�*

	conv_lossm�>����        )��P	v�A���A�*

	conv_loss�l�>N#4�        )��P	�	B���A�*

	conv_loss�o�>���        )��P	�<B���A�*

	conv_loss�l�>���$        )��P	�qB���A�*

	conv_lossD �>����        )��P	��B���A�*

	conv_loss��>�'�        )��P	��B���A�*

	conv_loss�"�>(���        )��P	EC���A�*

	conv_loss�J�>w]]`        )��P	�MC���A�*

	conv_loss��>l�!�        )��P	�}C���A�*

	conv_loss��>$�D�        )��P	̶C���A�*

	conv_loss��>���        )��P	��C���A�*

	conv_lossf~�>.�9        )��P	�D���A�*

	conv_loss̫>8��E        )��P	IHD���A�*

	conv_lossC�>���        )��P	xD���A�*

	conv_loss	��>`z�\        )��P	m�D���A�*

	conv_lossM��>�v        )��P	[�D���A�*

	conv_loss��>�Fgw        )��P	IE���A�*

	conv_loss<�>B�        )��P	`GE���A�*

	conv_loss)��>��R�        )��P	�vE���A�*

	conv_lossVq�>H4F�        )��P	��E���A�*

	conv_losstc�>����        )��P	��E���A�*

	conv_lossȪ>k5��        )��P	�F���A�*

	conv_loss���>m�J        )��P	p6F���A�*

	conv_lossr�>mkj�        )��P	AgF���A�*

	conv_loss+q�>Z-�-        )��P	�F���A�*

	conv_loss3%�>�qw        )��P	�F���A�*

	conv_lossѫ>��c        )��P	T�F���A�*

	conv_loss���>�h��        )��P	;$G���A�*

	conv_loss��>
��u        )��P	�TG���A�*

	conv_loss�M�>C-��        )��P	��G���A�*

	conv_loss�]�>G��        )��P	��G���A�*

	conv_loss��>�d        )��P	9�G���A�*

	conv_lossD`�>�x2        )��P	H���A�*

	conv_loss*4�>�.fr        )��P	2HH���A�*

	conv_losse��>m�1�        )��P	wH���A�*

	conv_loss���>��~Q        )��P	@�H���A�*

	conv_loss@��>��Uz        )��P	��H���A�*

	conv_loss�Ī>BӍv        )��P	�I���A�*

	conv_loss���>cxs        )��P	6I���A�*

	conv_lossQ�>u�$�        )��P	�dI���A�*

	conv_loss���>��6�        )��P	ݕI���A�*

	conv_loss���>{��p        )��P	��I���A�*

	conv_loss�W�>�z\�        )��P	�I���A�*

	conv_loss� �>c;�#        )��P	g(J���A�*

	conv_loss�Ы>K��z        )��P	XJ���A�*

	conv_lossL��>˫�        )��P	��J���A�*

	conv_loss��>; 2        )��P	��J���A�*

	conv_loss]��>B��R        )��P	w�J���A�*

	conv_loss'��>�J9�        )��P	�$K���A�*

	conv_lossuH�>j��        )��P	�UK���A�*

	conv_loss��>�v[�        )��P	�K���A�*

	conv_loss�,�>��ؾ        )��P	ŻK���A�*

	conv_loss4�>� �        )��P	��K���A�*

	conv_loss_F�>�P        )��P	z3L���A�*

	conv_loss=�>@w݇        )��P	*iL���A�*

	conv_loss��>>��+        )��P	��L���A�*

	conv_lossh�>x��        )��P	��L���A�*

	conv_loss���>���k        )��P	�
M���A�*

	conv_loss�̩>��a9        )��P	|;M���A�*

	conv_loss(�>��z�        )��P	`jM���A�*

	conv_loss�Ƭ>-���        )��P	��M���A�*

	conv_loss��>�3        )��P	�M���A�*

	conv_lossu©>g�3�        )��P	�	N���A�*

	conv_loss�ͨ>KLJK        )��P	�;N���A�*

	conv_loss�e�>G�        )��P	�jN���A�*

	conv_loss�!�>��M�        )��P	z�N���A�*

	conv_loss䒧>@SD        )��P	��N���A�*

	conv_loss�t�>���        )��P	d�N���A�*

	conv_loss�X�>{��        )��P	e5O���A�*

	conv_loss�ɨ>%ؙA        )��P	ygO���A�*

	conv_loss���>ʘ��        )��P	��O���A�*

	conv_loss�©>l��7        )��P	n�O���A�*

	conv_loss���>�=��        )��P	W�O���A�*

	conv_loss#��>`�         )��P	�%P���A�*

	conv_loss�K�>��RY        )��P	VP���A�*

	conv_loss���>��Fw        )��P	�P���A�*

	conv_loss�I�>/�V        )��P	e�P���A�*

	conv_loss�˫>�H�        )��P	E�P���A�*

	conv_loss�R�>��Ap        )��P	7Q���A�*

	conv_lossOD�>���        )��P	�DQ���A�*

	conv_lossԨ>�2I         )��P	�tQ���A�*

	conv_loss���>�߸U        )��P	��Q���A�*

	conv_loss���>:��9        )��P	��Q���A�*

	conv_loss,�> ڊt        )��P	LR���A�*

	conv_lossȤ�>("�        )��P	9R���A�*

	conv_loss���>�%��        )��P	�iR���A�*

	conv_loss�	�>>{S�        )��P	3�R���A�*

	conv_loss)K�>g�wd        )��P	�R���A�*

	conv_loss�é>u��        )��P	�R���A�*

	conv_loss[~�>��n�        )��P	d-S���A�*

	conv_loss���>2�vd        )��P	._S���A�*

	conv_loss��>�=�        )��P	��S���A�*

	conv_loss�N�>��O        )��P	��S���A�*

	conv_loss��>���(        )��P	��S���A�*

	conv_lossp�>�4%        )��P	2$T���A�*

	conv_lossȩ>eC�        )��P	LTT���A�*

	conv_loss�`�>�A�        )��P	4�T���A�*

	conv_loss���>�x@_        )��P	i�T���A�*

	conv_loss#J�>5���        )��P	�T���A�*

	conv_loss�>WN��        )��P	>,U���A�*

	conv_loss=�>�c        )��P	V\U���A�*

	conv_loss��>vF;�        )��P	�U���A�*

	conv_lossө>O%        )��P	��U���A�*

	conv_lossfJ�>o�        )��P	d�U���A�*

	conv_loss�C�>����        )��P	�8V���A�*

	conv_loss`a�>�3�        )��P	�jV���A�*

	conv_loss�e�>A3WH        )��P	��V���A�*

	conv_loss��>e�i        )��P	��V���A�*

	conv_loss���>��=�        )��P	�W���A�*

	conv_loss �>�C        )��P	OW���A�*

	conv_lossr��>�`�        )��P	aW���A�*

	conv_loss���>�|��        )��P	*�W���A�*

	conv_loss���>0�G�        )��P	�W���A�*

	conv_loss��>�,1�        )��P	�X���A�*

	conv_loss�>!9;        )��P	!@X���A�*

	conv_loss��>��ex        )��P	hpX���A�*

	conv_loss[��>�f5        )��P	�X���A�*

	conv_loss�7�>��t�        )��P	��X���A�*

	conv_lossX�>��~        )��P	�Y���A�*

	conv_lossYo�>8l�        )��P	�9Y���A�*

	conv_loss�h�>K�CA        )��P	�nY���A�*

	conv_loss((�>@~n�        )��P	��Y���A�*

	conv_loss�2�>��tu        )��P	�Y���A�*

	conv_lossX�>t0p�        )��P	�Z���A�*

	conv_loss�ϩ>��        )��P	�3Z���A�*

	conv_loss`��>��U        )��P	SdZ���A�*

	conv_loss��>�U8        )��P	r�Z���A�*

	conv_loss.�>x�:?        )��P	��Z���A�*

	conv_lossY �>�ՙ        )��P	�Z���A�*

	conv_loss�~�>�:[�        )��P	p)[���A�*

	conv_lossd9�>GOp        )��P	EY[���A�*

	conv_loss���>I�h        )��P	��[���A�*

	conv_loss>��>�qw�        )��P	�[���A�*

	conv_lossga�>3�k�        )��P	F�[���A�*

	conv_lossid�>UJ�        )��P	\���A�*

	conv_loss�/�>ط�*        )��P	�M\���A�*

	conv_loss��>	4��        )��P	�~\���A�*

	conv_lossm�>y)��        )��P	e�\���A�*

	conv_loss^�>A&l        )��P	l�\���A�*

	conv_loss���>�A        )��P	]���A�*

	conv_loss�ҩ>lTJ        )��P	 >]���A�*

	conv_loss,Ƨ>���        )��P	zo]���A�*

	conv_loss|��>���D        )��P	X�]���A�*

	conv_loss;{�>���        )��P	��]���A�*

	conv_lossG��>4{�e        )��P	�^���A�*

	conv_lossXݩ>"��        )��P	�3^���A�*

	conv_loss=P�>D7�        )��P	6c^���A�*

	conv_lossR��>+M��        )��P	p�^���A�*

	conv_loss¶�>�3�        )��P	�^���A�*

	conv_loss�A�>v�e+        )��P	�_���A�*

	conv_loss��>C�:�        )��P	8_���A�*

	conv_loss�ݩ>A)�        )��P	rg_���A�*

	conv_loss ��>��n�        )��P	�_���A�*

	conv_loss�M�>�Ha^        )��P	N�_���A�*

	conv_loss��>��u�        )��P	� `���A�*

	conv_lossT�>*�H�        )��P	1`���A�*

	conv_loss:&�>h/z        )��P	�``���A�*

	conv_loss�@�>ܫ�        )��P	�`���A�*

	conv_lossun�>*߸�        )��P	��`���A�*

	conv_loss��>��l�        )��P	�a���A�*

	conv_loss٫>7e��        )��P	PHa���A�*

	conv_lossm�>H�P        )��P	�ya���A�*

	conv_loss�]�>�]�        )��P	��a���A�*

	conv_loss��>�r��        )��P	�a���A�*

	conv_lossLԩ>}^
G        )��P	*b���A�*

	conv_loss�a�>��V6        )��P	�Fb���A�*

	conv_loss�m�>Ζ�\        )��P	<xb���A�*

	conv_loss���>�=�P        )��P	��b���A�*

	conv_loss�X�>-��T        )��P	��b���A�*

	conv_loss�6�>Ow��        )��P	Sc���A�*

	conv_loss脨>��qI        )��P	HHc���A�*

	conv_losso��>?�k�        )��P	�c���A�*

	conv_lossn �>�N�        )��P	��c���A�*

	conv_lossu�>���s        )��P	��c���A�*

	conv_loss���>�Wl        )��P	�d���A�*

	conv_loss棦>;���        )��P	�Nd���A�*

	conv_loss���>`��;        )��P	M�d���A�*

	conv_loss13�>BxD�        )��P	��d���A�*

	conv_loss��>���h        )��P	w�d���A�*

	conv_loss=p�>6�"s        )��P	e���A�*

	conv_loss���>N��2        )��P	�?e���A�*

	conv_loss��>�ԅ�        )��P	3qe���A�*

	conv_lossf��>��l4        )��P	��e���A�*

	conv_loss]y�>e�1�        )��P	0�e���A�*

	conv_lossDª>��ц        )��P	�f���A�*

	conv_loss�P�>t3�-        )��P	�2f���A�*

	conv_lossnH�>{"!        )��P	�df���A�*

	conv_lossS�>�emg        )��P	��f���A�*

	conv_lossޟ�>_}�        )��P	��f���A�*

	conv_lossz.�>Ľǫ        )��P	<�f���A�*

	conv_lossA�>;���        )��P	�,g���A�*

	conv_loss@�>.=��        )��P	�^g���A�*

	conv_loss&q�>Ӯ        )��P	-�g���A�*

	conv_loss��>E���        )��P	�g���A�*

	conv_lossǫ>V5��        )��P	-�g���A�*

	conv_loss�8�>��        )��P	'"h���A�*

	conv_loss�Q�>�Y        )��P	�Rh���A�*

	conv_lossg�>�ډa        )��P	g�l���A�*

	conv_loss�>�x        )��P	�dn���A�*

	conv_lossi@�>�Ȕ        )��P	Εn���A�*

	conv_loss�Χ>Վ�I        )��P	��n���A�*

	conv_loss5K�>￤W        )��P	7�n���A�*

	conv_loss���>.
�        )��P	�(o���A�*

	conv_lossFũ>r�J�        )��P	Xo���A�*

	conv_loss�E�>��T        )��P	��o���A�*

	conv_loss���>��!�        )��P	ֽo���A�*

	conv_loss�d�>�v�        )��P	p���A�*

	conv_loss.m�>��        )��P	�>p���A�*

	conv_loss/l�>�jf�        )��P	imp���A�*

	conv_loss��>u��H        )��P	.�p���A�*

	conv_lossM�>5��        )��P	��p���A�*

	conv_loss��>�/h�        )��P	Xq���A�*

	conv_loss���>t��6        )��P	4q���A�*

	conv_loss�
�>�U�        )��P	�vq���A�*

	conv_loss�ʪ>��        )��P	��q���A�*

	conv_lossO��>�6]�        )��P	#�q���A�*

	conv_loss�x�>�;�I        )��P	kr���A�*

	conv_loss��>�6�        )��P	�Cr���A�*

	conv_lossZ�>��Q�        )��P	�wr���A�*

	conv_loss��>�b6C        )��P	��r���A�*

	conv_loss���>�K�        )��P	��r���A�*

	conv_lossw�>f�@l        )��P	j	s���A�*

	conv_loss��>e��        )��P		:s���A�*

	conv_loss�Y�>����        )��P	�js���A�*

	conv_lossЩ>��6d        )��P	Ϝs���A�*

	conv_loss�x�>��         )��P	J�s���A�*

	conv_loss7
�>�        )��P	D�s���A�*

	conv_loss�b�>DV�:        )��P	Q.t���A�*

	conv_lossC��>�]�!        )��P	9`t���A�*

	conv_loss, �>ʖ�X        )��P	̐t���A�*

	conv_loss}��>�+�        )��P	@�t���A�*

	conv_loss��>Ӗ��        )��P	��t���A�*

	conv_loss���>7�>�        )��P	� u���A�*

	conv_lossq�>Oá�        )��P	QPu���A�*

	conv_lossz�>�"��        )��P	Zu���A�*

	conv_loss���>��        )��P	L�u���A�*

	conv_loss#��>��R        )��P	��u���A�*

	conv_loss���>��0        )��P	�v���A�*

	conv_lossI��>��-        )��P	�<v���A�*

	conv_loss���>��        )��P	�lv���A�*

	conv_lossQ�>O�ʝ        )��P	�v���A�*

	conv_losso��>zM�        )��P	@�v���A�*

	conv_loss�˩>E��        )��P	uw���A�*

	conv_loss�4�>ڪ�v        )��P	�1w���A�*

	conv_loss�t�>�^I        )��P	�aw���A�*

	conv_loss�t�>UY�        )��P	�w���A�*

	conv_loss9,�>�
oM        )��P	H�w���A�*

	conv_loss�-�>9"�        )��P	��w���A�*

	conv_lossܐ�>����        )��P	u.x���A�*

	conv_loss��>�^�        )��P	�^x���A�*

	conv_lossR�>wŰ        )��P	V�x���A�*

	conv_lossQ�>��G        )��P	O�x���A�*

	conv_loss8�>ۘ        )��P	<�x���A�*

	conv_loss�o�>�;<        )��P	�,y���A�*

	conv_loss��>m[�        )��P	�]y���A�*

	conv_loss�ث>y�        )��P	��y���A�*

	conv_loss��>�8�        )��P	��y���A�*

	conv_loss�ީ>��Q�        )��P	�z���A�*

	conv_loss ֧>�        )��P	&>z���A�*

	conv_losse��>��'5        )��P	poz���A�*

	conv_lossU�>sz�        )��P	�z���A�*

	conv_loss��>���a        )��P	��z���A�*

	conv_lossA��>5C��        )��P	�{���A�*

	conv_lossC`�>;B        )��P	:{���A�*

	conv_loss���>*M��        )��P	$k{���A�*

	conv_loss���>��        )��P	��{���A�*

	conv_loss�E�>|jĮ        )��P	�{���A�*

	conv_loss죦>B*Ѵ        )��P	h�{���A�*

	conv_loss7J�>�2�        )��P	�+|���A�*

	conv_loss�^�>�=�        )��P	ga|���A�*

	conv_loss�B�>�ق0        )��P	�|���A�*

	conv_loss���>Y�]�        )��P	��|���A�*

	conv_loss�s�>�3��        )��P	}���A�*

	conv_loss&1�>0([        )��P	�4}���A�*

	conv_loss��>���        )��P	�k}���A�*

	conv_loss�>��G        )��P	2�}���A�*

	conv_lossXܨ>�
�        )��P	F�}���A�*

	conv_loss%ب>��z        )��P	��}���A�*

	conv_lossNR�>�1�        )��P	s1~���A�*

	conv_loss�}�>�y��        )��P	(a~���A�*

	conv_loss�F�>a�Qw        )��P	�~���A�*

	conv_lossd$�>q�)R        )��P	^�~���A�*

	conv_loss��>��˥        )��P	�~���A�*

	conv_loss��>�8�        )��P	K2���A�*

	conv_lossר>5�W�        )��P	�b���A�*

	conv_lossGϧ>�R        )��P	Г���A�*

	conv_lossͨ>'��        )��P	�����A�*

	conv_loss���>
�b        )��P	|����A�*

	conv_loss���>O�'�        )��P	s&����A�*

	conv_loss���>Y?ә        )��P	W����A�*

	conv_lossFܦ>��W�        )��P	z�����A�*

	conv_lossb�>S�         )��P	6�����A�*

	conv_loss�}�>!��V        )��P	�����A�*

	conv_lossWw�>T8R+        )��P	�����A�*

	conv_loss���>�:        )��P	�K����A�*

	conv_loss�W�>AY�        )��P	_{����A�*

	conv_loss���>�l        )��P	������A�*

	conv_loss/\�>5�=        )��P	�ہ���A�*

	conv_loss���>��oG        )��P	L����A�*

	conv_loss�h�>�D��        )��P	�N����A�*

	conv_loss/�>��y        )��P	7�����A�*

	conv_loss:X�>'Sh         )��P	#�����A�*

	conv_loss�J�>�i�        )��P	!����A�*

	conv_lossb�>�Gp        )��P	c����A�*

	conv_loss:�>�	#�        )��P	�O����A�*

	conv_loss�ǧ>|�8        )��P	������A�*

	conv_loss���>��X�        )��P	�����A�*

	conv_loss���>��l        )��P	� ����A�*

	conv_lossp@�>�kU%        )��P	7����A�*

	conv_loss�A�>y�:�        )��P	�{����A�*

	conv_lossЧ>:�        )��P	������A�*

	conv_loss�Ĩ>��        )��P	?�����A�*

	conv_lossٛ�>��@        )��P	�����A�*

	conv_loss��>}1�        )��P	�G����A�*

	conv_loss���>Pk�1        )��P	x����A�*

	conv_loss6��>%Ҡ%        )��P	ϭ����A�*

	conv_loss뱦> � �        )��P	�ޅ���A�*

	conv_loss�>:���        )��P	�����A�*

	conv_loss2��>�V        )��P	�D����A�*

	conv_loss���>I�_2        )��P	y}����A�*

	conv_lossޅ�>�t��        )��P	������A�*

	conv_loss���>x�0        )��P	�����A�*

	conv_loss���>��]�        )��P	#����A�*

	conv_loss�0�>;j/        )��P	�Q����A�*

	conv_loss���>��\<        )��P	ꅇ���A�*

	conv_loss¨>�Ͼ�        )��P	�����A�*

	conv_loss^,�>��        )��P	�����A�*

	conv_loss���>v�d�        )��P	1����A�*

	conv_lossk\�>����        )��P	(M����A�*

	conv_lossI�>��8        )��P	�~����A�*

	conv_loss��>!�`        )��P	ޯ����A�*

	conv_loss��>&�,�        )��P	�����A�*

	conv_loss"Ҩ>�yp        )��P	O����A�*

	conv_lossp*�>d��        )��P	NG����A�*

	conv_lossM`�>C�F�        )��P	�w����A�*

	conv_lossJ��>l��U        )��P	������A�*

	conv_loss8D�>�g        )��P	ۉ���A�*

	conv_lossoG�>xә
        )��P	����A�*

	conv_loss�o�>�L         )��P	�@����A�*

	conv_loss�ۨ>i<        )��P	q����A�*

	conv_loss)B�>���        )��P	�����A�*

	conv_loss�4�>¼�!        )��P	�Ԋ���A�*

	conv_loss�@�>�/N        )��P	3����A�*

	conv_loss|e�>҅F�        )��P	89����A�*

	conv_loss˖�>���d        )��P	�l����A�*

	conv_loss�
�>a��        )��P	����A�*

	conv_loss/]�>��R        )��P	|΋���A�*

	conv_lossN��>!z�        )��P	������A�*

	conv_loss���>P��        )��P	�G����A�*

	conv_loss�ܦ>8M�+        )��P	t~����A�*

	conv_loss���>y���        )��P	Z�����A�*

	conv_loss�9�>��_4        )��P	����A�*

	conv_lossDz�>�2��        )��P	|)����A�*

	conv_loss|M�>���        )��P	�a����A�*

	conv_lossBȧ>y���        )��P	k�����A�*

	conv_lossҹ�>�A�        )��P	W����A�*

	conv_loss׼�>�u        )��P	�-����A�*

	conv_loss��>�I�        )��P	�c����A�*

	conv_lossC�>��I�        )��P	@�����A�*

	conv_loss���>4��        )��P	�Վ���A�*

	conv_loss���>�)n:        )��P	�����A�*

	conv_lossNا>�Şh        )��P	_;����A�*

	conv_lossq<�>�d�        )��P	�n����A�*

	conv_loss�z�>�/�h        )��P	������A�*

	conv_loss㎧>fC�P        )��P	w֏���A�*

	conv_loss2��>����        )��P	�����A�*

	conv_loss��>�HI        )��P	�D����A�*

	conv_loss�Ħ>X>�        )��P	`�����A�*

	conv_loss���>R��        )��P	M�����A�*

	conv_loss���>֯�v        )��P	B����A�*

	conv_lossN �>���g        )��P	�5����A�*

	conv_losse��>�&��        )��P	�j����A�*

	conv_loss��>��+m        )��P	������A�*

	conv_loss옧>��2        )��P	ӑ���A�*

	conv_loss�L�>�pM/        )��P	�����A�*

	conv_loss�7�>��'        )��P	�:����A�*

	conv_loss��>�Wh        )��P	&t����A�*

	conv_loss=n�>�*��        )��P	J�����A�*

	conv_loss�J�>��]�        )��P	�����A�*

	conv_loss�>��        )��P	N ����A�*

	conv_loss�=�>ŻL        )��P	pb����A�*

	conv_loss�Z�>N[�        )��P	9�����A�*

	conv_loss�A�>vF@        )��P	�֓���A�*

	conv_loss��>,kC�        )��P	�����A�*

	conv_lossK1�>�+Ę        )��P	�<����A�*

	conv_lossBy�>qyH        )��P	�m����A�*

	conv_lossF��>%L$_        )��P	������A�*

	conv_loss�Ө>�F�        )��P	|ϔ���A�*

	conv_lossjͥ>7���        )��P	c����A�*

	conv_loss�V�>MI�        )��P	_2����A�*

	conv_lossXV�>�cN        )��P	Mc����A�*

	conv_loss���>�T�        )��P	������A�*

	conv_loss��>�q\        )��P	�˕���A�*

	conv_loss_��>ɔi        )��P	�����A�*

	conv_loss,6�>&���        )��P	�4����A�*

	conv_loss���>q�        )��P	�c����A�*

	conv_loss�e�>	�xi        )��P	.�����A�*

	conv_lossV�>�\�        )��P	KĖ���A�*

	conv_lossW�>L��        )��P	ӄ����A�*

	conv_losso�>�a�c        )��P	A�����A�*

	conv_loss"&�>^��        )��P	�����A�*

	conv_loss��>I�i�        )��P	�)����A�*

	conv_loss��>��        )��P	8X����A�*

	conv_loss�*�>�g�         )��P	�����A�*

	conv_loss|b�>e2�v        )��P	������A�*

	conv_loss^�>�,k        )��P	L�����A�*

	conv_lossU��>e���        )��P	�0����A�*

	conv_loss��>�&J�        )��P	w`����A�*

	conv_lossE��>����        )��P	������A�*

	conv_lossd�>~5hs        )��P	����A�*

	conv_loss虦>�;        )��P	�#����A�*

	conv_loss)��>�kF        )��P	TU����A�*

	conv_loss���>�w         )��P	������A�*

	conv_loss�I�>ʥAN        )��P	����A�*

	conv_lossCW�>	�E�        )��P	������A�*

	conv_loss���>�<�        )��P	�1����A�*

	conv_loss��>+�g        )��P	Ae����A�*

	conv_lossJϥ>��g        )��P	%�����A�*

	conv_loss٤>_�'x        )��P	|̜���A�*

	conv_loss�s�>3)%�        )��P	>����A�*

	conv_lossA�>_c        )��P	vL����A�*

	conv_lossYJ�>콆        )��P	엝���A�*

	conv_lossT�>�ݓ        )��P	�ʝ���A�*

	conv_losspƥ>��?{        )��P	-����A�*

	conv_loss �>����        )��P	�9����A�*

	conv_loss���>f~�        )��P	<k����A�*

	conv_loss9�>���        )��P	Q�����A�*

	conv_loss��>Ǭ�D        )��P	�מ���A�*

	conv_loss7��>j��p        )��P	?����A�*

	conv_loss~��>����        )��P	e?����A�*

	conv_lossqѥ>���        )��P	�p����A�*

	conv_loss$�>%��        )��P	������A�*

	conv_loss�ۦ>�#F6        )��P	O����A�*

	conv_lossV��>_6�        )��P	U"����A�*

	conv_lossG�>1�r        )��P	�_����A�*

	conv_loss�:�>_-��        )��P	I�����A�*

	conv_loss���>ß��        )��P	Aɠ���A�*

	conv_loss$E�>� �o        )��P	������A�*

	conv_loss�=�>���        )��P	y(����A�*

	conv_loss��>q@�G        )��P	�W����A�*

	conv_loss˦�>x���        )��P	ꇡ���A�*

	conv_loss:z�>&��(        )��P	������A�*

	conv_lossk��>E�\        )��P	�����A�*

	conv_loss�٧>oN;B        )��P	*����A�*

	conv_loss>]�>�f�5        )��P	%T����A�*

	conv_loss�M�>Me<:        )��P	������A�*

	conv_loss�+�>��rH        )��P	齢���A�*

	conv_loss���>͊��        )��P	h����A�*

	conv_loss=Q�>����        )��P	4����A�*

	conv_loss�r�>(�m        )��P	�c����A�*

	conv_loss�0�>D�J        )��P	������A�*

	conv_loss�!�>��!�        )��P	Kģ���A�*

	conv_loss��>�&w�        )��P	������A�*

	conv_loss{��>i��        )��P	-����A�*

	conv_loss���>&,�        )��P	�]����A�*

	conv_loss�֥>�i^(        )��P	.�����A�*

	conv_lossg�>||^        )��P	W�����A�*

	conv_loss�@�>c��        )��P	������A�*

	conv_loss��>�&^[        )��P	X,����A�*

	conv_loss�|�>Q�e�        )��P	\����A�*

	conv_loss��>����        )��P	ۍ����A�*

	conv_losso}�>���        )��P	������A�*

	conv_losskz�>�Rd�        )��P	������A�*

	conv_loss-;�>�d$        )��P	r����A�*

	conv_loss�ͧ>��c�        )��P	�M����A�*

	conv_loss�ȥ>�g��        )��P	�~����A�*

	conv_loss�^�>��x�        )��P	Ů����A�*

	conv_loss��>�j�<        )��P	�ܦ���A�*

	conv_loss��>}��q        )��P	�����A�*

	conv_loss�#�>6)�'        )��P	�;����A�*

	conv_loss3��>����        )��P	Sl����A�*

	conv_loss11�> �Zz        )��P	������A�*

	conv_loss��>bZ         )��P	+̧���A�*

	conv_lossڭ�>�A�        )��P		�����A�*

	conv_loss���>���        )��P	=+����A�*

	conv_loss!�>���        )��P	^����A�*

	conv_lossJ��>|g��        )��P	쌨���A�*

	conv_loss<�>����        )��P	{�����A�*

	conv_loss�>�[�Y        )��P	i�����A�*

	conv_loss~��>�.�        )��P	����A�*

	conv_loss��>�3>        )��P	[J����A�*

	conv_loss�M�>��h        )��P	�{����A�*

	conv_loss�e�>���{        )��P	�����A�*

	conv_lossN��>��{q        )��P	����A�*

	conv_loss���>�1Z�        )��P	�����A�*

	conv_loss�ܧ>����        )��P	�K����A�*

	conv_loss�=�>�N��        )��P	�|����A�*

	conv_loss9��>W�        )��P	�����A�*

	conv_lossv��>ۢ-j        )��P	qڪ���A�*

	conv_loss'@�>wk?b        )��P	p����A�*

	conv_loss��>OA}        )��P	�?����A�*

	conv_lossͧ>��         )��P	�n����A�*

	conv_loss���>�p�        )��P	颫���A�*

	conv_lossy��>��*        )��P	ҫ���A�*

	conv_loss�=�>jx        )��P	����A�*

	conv_loss{Ө>r��        )��P	�3����A�*

	conv_loss�Ц>1�bf        )��P	�i����A�*

	conv_lossoŦ>OlN�        )��P	�����A�*

	conv_loss��>ǀ��        )��P	9ܬ���A�*

	conv_loss~��>���)        )��P	\����A�*

	conv_loss�N�>*���        )��P	,<����A�*

	conv_loss��>�g�s        )��P	@s����A�*

	conv_loss��>E[�?        )��P	������A�*

	conv_loss��>֭�         )��P	�ӭ���A�*

	conv_loss|�>#���        )��P	�����A�*

	conv_loss��>�.�i        )��P	:B����A�*

	conv_loss���>p���        )��P	�y����A�*

	conv_loss�u�>�L��        )��P	������A�*

	conv_loss��>��*�        )��P	�����A�*

	conv_loss�Q�>�Is        )��P	�����A�*

	conv_lossx�>-o�        )��P	^X����A�*

	conv_loss��>'�8�        )��P	戯���A�*

	conv_lossF�>�&        )��P	������A�*

	conv_lossB�>��        )��P	�����A�*

	conv_lossi�>V�=        )��P	W����A�*

	conv_lossWr�>I�oj        )��P	NG����A�*

	conv_loss:`�>�r�3        )��P	�w����A�*

	conv_lossI�>��V        )��P	������A�*

	conv_loss�v�>��j        )��P	`ְ���A�*

	conv_loss� �>����        )��P	Q����A�*

	conv_loss�>����        )��P	�2����A�*

	conv_loss~}�>����        )��P	Vb����A�*

	conv_loss��>#��#        )��P	9�����A�*

	conv_loss^I�>�~��        )��P	ű���A�*

	conv_loss.;�>��b        )��P	������A�*

	conv_loss-'�>Ð��        )��P	$����A�*

	conv_loss�0�>8��j        )��P	qS����A�*

	conv_loss�%�>a[        )��P	����A�*

	conv_loss���>n4q[        )��P	˱����A�*

	conv_loss�}�>XP        )��P	1����A�*

	conv_lossK�>����        )��P	�����A�*

	conv_loss���>SHNn        )��P	�@����A�*

	conv_lossބ�>�Sq�        )��P	�n����A�*

	conv_loss<�>v�U�        )��P	A�����A�*

	conv_lossR��>���        )��P	�̳���A�*

	conv_loss�Ȧ>�%.        )��P	������A�*

	conv_loss�T�>a��        )��P	�/����A�*

	conv_lossz��>�Iq        )��P	�_����A�*

	conv_lossnO�>[ a        )��P	������A�*

	conv_loss�ߤ>2�        )��P	a�����A�*

	conv_lossCצ>��        )��P	�����A�*

	conv_loss�d�>~�W�        )��P	^����A�*

	conv_loss�L�>Q�.        )��P	�K����A�*

	conv_loss�k�>�x�        )��P	q|����A�*

	conv_loss��>��H        )��P	������A�*

	conv_loss�g�>��@�        )��P	=����A�*

	conv_loss��>���        )��P	�����A�*

	conv_lossɶ�>0�J�        )��P	�H����A�*

	conv_lossj��>�'��        )��P	������A�*

	conv_losse�>tH�        )��P	�϶���A�*

	conv_loss*�>�;f�        )��P	�����A�*

	conv_loss��>!��p        )��P	@3����A�*

	conv_loss���>0��        )��P	�p����A�*

	conv_loss��>�e0        )��P	s�����A�*

	conv_loss/;�>���        )��P	MϷ���A�*

	conv_loss�#�>g�9        )��P	������A�*

	conv_loss�T�>q��;        )��P	�1����A�*

	conv_loss�N�>?�        )��P	Rq����A�*

	conv_loss�ͧ>ƫ��        )��P	@�����A�*

	conv_loss�	�>gu��        )��P	�ظ���A�*

	conv_loss�V�>Ev/�        )��P	�����A�*

	conv_loss�|�>��+�        )��P	�B����A�*

	conv_loss�-�>!7!g        )��P	�~����A�*

	conv_losskN�>=Z �        )��P	�����A�*

	conv_loss���>81�        )��P	y����A�*

	conv_loss�U�>7p�!        )��P	
����A�*

	conv_loss�g�>N�a?        )��P	fM����A�*

	conv_loss�ħ>U���        )��P	X}����A�*

	conv_loss��>�#Q        )��P	3�����A�*

	conv_loss�l�>k��W        )��P	wۺ���A�*

	conv_lossY�>�\C        )��P	]����A�*

	conv_loss��>��Fn        )��P	�;����A�*

	conv_loss��>z�D�        )��P	�p����A�*

	conv_loss���>LM�        )��P	������A�*

	conv_loss���>Gn$�        )��P	�ٻ���A�*

	conv_lossJШ>�ee�        )��P	.����A�*

	conv_lossk\�>=��        )��P	m>����A�*

	conv_lossa�>9d��        )��P	;p����A�*

	conv_loss@�>��\�        )��P	_�����A�*

	conv_loss��>���        )��P	�Լ���A�*

	conv_loss�1�>¬HS        )��P	@����A�*

	conv_loss+/�>�x-j        )��P	�6����A�*

	conv_loss{�>�_
        )��P	8g����A�*

	conv_loss��>���N        )��P	>�����A�*

	conv_losss�>�f0%        )��P	Qɽ���A�*

	conv_loss��>f���        )��P	;�����A�*

	conv_loss��>f���        )��P	�*����A�*

	conv_loss��>e��        )��P	[����A�*

	conv_loss��>g��T        )��P	������A�*

	conv_loss��>�|        )��P	�����A�*

	conv_lossC�>gB%        )��P	�����A�*

	conv_lossK��>Nb@Z        )��P	7"����A�*

	conv_lossr٨>�k�/        )��P	�R����A�*

	conv_loss�>�>��V        )��P	������A�*

	conv_loss?a�>�r�e        )��P	޳����A�*

	conv_loss?�>4>�        )��P	
����A�*

	conv_loss�Ĩ>����        )��P	s����A�*

	conv_loss�$�>q=Cv        )��P	�G����A�*

	conv_loss�>�Ss#        )��P	������A�*

	conv_loss�~�>4��I        )��P	�����A�*

	conv_lossjK�>��S�        )��P	�@����A�*

	conv_lossPק>�'��        )��P	;w����A�*

	conv_loss@)�>R/F        )��P	�����A�*

	conv_loss�F�>��S        )��P	-�����A�*

	conv_loss�"�>����        )��P	p����A�*

	conv_lossaG�>�8�        )��P	�I����A�*

	conv_lossѠ�>�C�        )��P	�~����A�*

	conv_loss���>լ1^        )��P	β����A�*

	conv_loss�ؤ>�1{�        )��P	<�����A�*

	conv_lossm�>T�ӂ        )��P	{����A�*

	conv_loss�]�>,�<        )��P	�I����A�*

	conv_loss��>�zW�        )��P	vz����A�*

	conv_loss� �>Cz�        )��P	k�����A�*

	conv_loss>R�>�*�        )��P	������A�*

	conv_loss|��>�$�        )��P	�����A�*

	conv_loss)ߦ>�8��        )��P	�4����A�*

	conv_loss�A�>��V        )��P	*d����A�*

	conv_loss�2�>TbY        )��P	̙����A�*

	conv_loss̭�>�`��        )��P	�����A�*

	conv_lossއ�>�؛a        )��P	T�����A�*

	conv_loss0}�>�R��        )��P	8,����A�*

	conv_loss%�>���        )��P	TZ����A�*

	conv_lossl�>��?        )��P	������A�*

	conv_loss��>��        )��P	������A�*

	conv_loss ��>��-        )��P	������A�*

	conv_loss�٤>b���        )��P	�����A�*

	conv_lossا>Z���        )��P	�K����A�*

	conv_loss߬�>ywŚ        )��P	$z����A�*

	conv_loss�j�>�x7b        )��P	������A�*

	conv_lossb[�>���        )��P	(�����A�*

	conv_lossg3�>�;�H        )��P	�����A�*

	conv_loss�m�>��2�        )��P	�1����A�*

	conv_loss�§>.Y\�        )��P	[^����A�*

	conv_loss�]�>>\t+        )��P	�����A�*

	conv_loss�X�>�b�        )��P	f�����A�*

	conv_lossm��>x\G        )��P	y�����A�*

	conv_loss�֨>��oI        )��P	����A�*

	conv_loss졨>Ug_8        )��P	�H����A�*

	conv_loss��>��x�        )��P	w����A�*

	conv_lossda�>o��        )��P	������A�*

	conv_loss���>�	u        )��P	������A�*

	conv_loss��>��xG        )��P	�����A�*

	conv_loss�m�>�'B�        )��P	�@����A�*

	conv_lossK��>T�o�        )��P	�p����A�*

	conv_lossƤ>��        )��P	������A�*

	conv_loss:��>磵�        )��P	������A�*

	conv_loss�3�>���        )��P	�	����A�*

	conv_loss=t�>�3�        )��P	�7����A�*

	conv_loss�>�S�4        )��P	hw����A�*

	conv_loss���>��;�        )��P	�����A�*

	conv_loss濦>��Pt        )��P	,�����A�*

	conv_loss]��>�(��        )��P	�����A�*

	conv_loss�w�>ѧ��        )��P	�7����A�*

	conv_loss�æ>.�s        )��P	@g����A�*

	conv_loss:�>%��4        )��P	@�����A�*

	conv_loss���>�}��        )��P	#�����A�*

	conv_loss�r�>E�i        )��P	�����A�*

	conv_loss^��>���        )��P	�A����A�*

	conv_lossc��>	b        )��P	�x����A�*

	conv_loss��>�!ݷ        )��P	������A�*

	conv_loss��>̋֏        )��P	q�����A�*

	conv_loss�F�> l)�        )��P	�����A�*

	conv_loss*��>�!�q        )��P	RB����A�*

	conv_loss�Χ>��$        )��P	&r����A�*

	conv_lossH�>�p��        )��P	������A�*

	conv_loss�>hw$        )��P	������A�*

	conv_loss���>���        )��P	?����A�*

	conv_loss�
�>2��W        )��P	f3����A�*

	conv_losshǦ>�ext        )��P	�b����A�*

	conv_loss�a�>1̯        )��P	`�����A�*

	conv_loss�B�>��        )��P	������A�*

	conv_loss��>��"        )��P	�����A�*

	conv_loss�^�>�s        )��P	�/����A�*

	conv_loss�]�>Q�8[        )��P	_����A�*

	conv_lossf�>Ey�a        )��P	`�����A�*

	conv_loss:O�>seE        )��P	.�����A�*

	conv_loss�(�>��o        )��P	������A�*

	conv_loss�m�>��i�        )��P	=#����A�*

	conv_loss�ˤ>b�2u        )��P	+S����A�*

	conv_loss�ؤ>y;A        )��P	������A�*

	conv_loss�>�R�        )��P	������A�*

	conv_loss�|�>��x        )��P	������A�*

	conv_lossH�>_4ҭ        )��P	)����A�*

	conv_lossD�>#ܾ        )��P	�B����A�*

	conv_lossl�>��ʲ        )��P	�p����A�*

	conv_loss���>Z�`e        )��P	������A�*

	conv_loss~��>��V�        )��P	�����A�*

	conv_loss��>�ͼN        )��P	~�����A�*

	conv_loss�ާ>�kk        )��P	�-����A�*

	conv_loss���>��Y�        )��P	i_����A�*

	conv_loss���>�my�        )��P	������A�*

	conv_loss=��>^��(        )��P	�����A�*

	conv_loss��>*��        )��P	������A�*

	conv_loss�p�>8*Y#        )��P	�����A�*

	conv_loss�ѥ>�`o�        )��P	�J����A�*

	conv_loss�C�>@���        )��P	�{����A�*

	conv_lossp�>{�#^        )��P	u�����A�*

	conv_loss���>x�        )��P	p;����A�*

	conv_loss���>4��        )��P	�}����A�*

	conv_lossߧ>t"�        )��P	�����A�*

	conv_lossL��>k75e        )��P	�����A�*

	conv_lossŖ�>e��p        )��P	�����A�*

	conv_lossT�>���        )��P	�D����A�*

	conv_loss쾥>��[�        )��P	|����A�*

	conv_loss8��>�[b?        )��P	Y�����A�*

	conv_loss`��>%�1:        )��P	������A�*

	conv_loss=Y�>�y$        )��P	>����A�*

	conv_lossE��>�u�z        )��P	UL����A�*

	conv_loss��>;��        )��P	G�����A�*

	conv_lossh�>5��        )��P	F�����A�*

	conv_lossX5�>��V�        )��P	�����A�*

	conv_lossMv�>�ݰ        )��P	�����A�*

	conv_loss@��>�T        )��P	6L����A�*

	conv_loss�-�>����        )��P	M�����A�*

	conv_loss춤>�	,H        )��P	۴����A�*

	conv_lossB�>ݷ        )��P	Z�����A�*

	conv_lossR�>Q�!-        )��P	�����A�*

	conv_loss���>��h�        )��P	dB����A�*

	conv_loss�ڥ>o��        )��P	�p����A�*

	conv_loss��>1�        )��P	k�����A�*

	conv_lossڏ�>1�!�        )��P	 �����A�*

	conv_loss2w�>y���        )��P	����A�*

	conv_lossG¤>��        )��P	�2����A�*

	conv_loss���>��        )��P	�a����A�*

	conv_loss�?�>c,(        )��P	�����A�*

	conv_loss\�>O#Z3        )��P	1�����A�*

	conv_loss6s�>�3�        )��P	������A�*

	conv_lossj}�>_        )��P	%$����A�*

	conv_lossp��>Ji��        )��P	hS����A�*

	conv_loss��>mN��        )��P	}�����A�*

	conv_loss��>-�>O        )��P	Z�����A�*

	conv_lossL\�>[u(        )��P	z�����A�*

	conv_loss.��>w�        )��P	�����A�*

	conv_loss���>o���        )��P	SG����A�*

	conv_loss䬥>���        )��P	?v����A�*

	conv_lossT��>XnW        )��P	�����A�*

	conv_lossj��>_�A        )��P	�����A�*

	conv_loss��>~d�        )��P	m����A�*

	conv_loss�9�>g��        )��P	^4����A�*

	conv_lossI9�>v�,        )��P	�c����A�*

	conv_loss���>mVy:        )��P	�����A�*

	conv_loss膦>7��1        )��P	�����A�*

	conv_loss���>�|�        )��P	Q�����A�*

	conv_lossۤ>�x	�        )��P	�&����A�*

	conv_loss�P�>;�2        )��P	�W����A�*

	conv_lossň�>C(        )��P	V�����A�*

	conv_loss���>�x�t        )��P	ȵ����A�*

	conv_loss��>�|�        )��P	������A�*

	conv_lossi��>~x=        )��P	Z'����A�*

	conv_loss�ң>O��        )��P	-W����A�*

	conv_loss���>�b�Z        )��P	Ո����A�*

	conv_lossF�>�fE        )��P	i�����A�*

	conv_loss���>��S�        )��P	Y�����A�*

	conv_loss��>6f�        )��P	$����A�*

	conv_lossC�>7<        )��P	kS����A�*

	conv_lossʎ�>��J~        )��P	�����A�*

	conv_loss�Z�>���        )��P	�����A�*

	conv_loss_�>��#        )��P	������A�*

	conv_losss��>� a�        )��P	R*����A�*

	conv_loss�ɤ>N[.�        )��P	�]����A�*

	conv_loss��>�� �        )��P	f�����A�*

	conv_lossé�>�'�/        )��P	1�����A�*

	conv_loss���>`:5�        )��P	������A�*

	conv_loss�ǥ>���s        )��P	O$����A�*

	conv_loss�T�>��G�        )��P	�[����A�*

	conv_loss��>{�q�        )��P	Ŏ����A�*

	conv_loss��>J��        )��P	ܾ����A�*

	conv_lossZ��>u�I8        )��P	9�����A�*

	conv_loss�.�>����        )��P	� ����A�*

	conv_loss�0�>�        )��P	R����A�*

	conv_loss)��>sZa�        )��P	�����A�*

	conv_loss���>b�	        )��P	������A�*

	conv_loss��>��]        )��P	@�����A�*

	conv_lossh��>�nq        )��P	@����A�*

	conv_loss`��>��d�        )��P	�C����A�*

	conv_loss��>����        )��P	�r����A�*

	conv_lossa��>-߹�        )��P	W�����A�*

	conv_loss�>����        )��P	V�����A�*

	conv_loss���>���B        )��P	�����A�*

	conv_loss�̤>4���        )��P	%3����A�*

	conv_loss�v�>h��        )��P	|a����A�*

	conv_loss��>K��        )��P	�����A�*

	conv_loss���>��g�        )��P	�����A�*

	conv_loss�B�>��|-        )��P	������A�*

	conv_loss���>P��        )��P	O����A�*

	conv_loss�0�>�@Y        )��P	�O����A�*

	conv_lossu�>��        )��P	�����A�*

	conv_loss:R�>j�        )��P	s�����A�*

	conv_loss�e�>��AN        )��P	]�����A�*

	conv_loss&�>~ul�        )��P	;����A�*

	conv_lossw�>T[��        )��P	PE����A�*

	conv_loss℧>�=AG        )��P	�x����A�*

	conv_loss���>�? r        )��P	C�����A�*

	conv_loss�0�>��mw        )��P	.�����A�*

	conv_loss�Q�>��y        )��P		����A�*

	conv_loss/��><�x        )��P	�:����A�*

	conv_loss&/�>V�	        )��P	)j����A�*

	conv_lossj��>��        )��P	.�����A�*

	conv_lossԬ�>�ޭS        )��P	�,����A�*

	conv_loss'ߥ>�`:�        )��P	]����A�*

	conv_loss�>���        )��P	+�����A�*

	conv_loss"l�>�5z$        )��P	������A�*

	conv_loss�X�>qU{        )��P	������A�*

	conv_loss^d�>�2Y        )��P	�%����A�*

	conv_lossQ�>���Q        )��P	*V����A�*

	conv_loss]ע>��D�        )��P	������A�*

	conv_loss��>^��D        )��P	������A�*

	conv_lossDE�>�D�o        )��P	{�����A�*

	conv_loss0�>�$6<        )��P	l/����A�*

	conv_loss?˥>�+q        )��P	Yg����A�*

	conv_lossT.�>R�        )��P	������A�*

	conv_loss-�>P�N�        )��P	�����A�*

	conv_loss��>�u�        )��P	u����A�*

	conv_lossJB�>s]�        )��P	�5����A�*

	conv_lossd��>|d��        )��P	�i����A�*

	conv_loss�B�>7A�        )��P		�����A�*

	conv_loss� �>���        )��P	������A�*

	conv_loss�f�>�$�_        )��P	G����A�*

	conv_loss�Ӧ>RZW        )��P	�;����A�*

	conv_loss0��>�M8        )��P	�m����A�*

	conv_lossq�>����        )��P	�����A�*

	conv_loss��>�L"        )��P	�����A�*

	conv_lossc	�>�xy        )��P	����A�*

	conv_loss�K�>��        )��P	�>����A�*

	conv_loss�>$�q        )��P	r����A�*

	conv_loss���>·        )��P	������A�*

	conv_loss�C�>�$�o        )��P	,�����A�*

	conv_loss{��>�|�X        )��P	m����A�*

	conv_loss���>��9{        )��P	w3����A�*

	conv_lossZ[�>��<        )��P	�c����A�*

	conv_loss��>��0�        )��P	������A�*

	conv_loss�P�>�M�T        )��P	������A�*

	conv_lossj¦>���        )��P	������A�*

	conv_loss�9�>��'A        )��P	 )����A�*

	conv_loss�¥>�6�        )��P	vZ����A�*

	conv_lossD�>���        )��P	�����A�*

	conv_loss�̧>/�!�        )��P	������A�*

	conv_loss#��>B��        )��P	W�����A�*

	conv_loss��>j#q�        )��P	�����A�*

	conv_loss4Ҥ>o�        )��P	uK����A�*

	conv_loss�>n�E        )��P	�z����A�*

	conv_lossgͥ>� ��        )��P	������A�*

	conv_loss�D�>i��h        )��P	r�����A�*

	conv_lossll�>MkF�        )��P	�����A�*

	conv_loss#�>�d�        )��P	B;����A�*

	conv_loss���>��l�        )��P	_n����A�*

	conv_loss@g�>��c�        )��P	������A�*

	conv_loss���>��H�        )��P	L�����A�*

	conv_lossA�>��+|        )��P	�����A�*

	conv_loss��>�d�e        )��P	YE����A�*

	conv_losss?�>�Uv�        )��P	�u����A�*

	conv_loss���>ާ        )��P	�����A�*

	conv_lossA0�>�Al        )��P	8�����A�*

	conv_lossI��>�ծ        )��P	m����A�*

	conv_loss�B�>5[AW        )��P	k<����A�*

	conv_lossK�>��        )��P	s����A�*

	conv_lossy&�>>Hq�        )��P	�����A�*

	conv_loss-I�>���        )��P	������A�*

	conv_losst0�>S�e        )��P	�����A�*

	conv_losssF�>�U�        )��P	�4����A�*

	conv_loss��>ԥ��        )��P	�k����A�*

	conv_loss��>�-        )��P	բ����A�*

	conv_loss��>ߧI�        )��P	������A�*

	conv_lossz��>|6�B        )��P	�	����A�*

	conv_loss��>wA;�        )��P	9����A�*

	conv_loss��>�Et�        )��P	j����A�*

	conv_loss< �>�`d        )��P	$�����A�*

	conv_loss�q�>�Gn�        )��P	������A�*

	conv_loss/*�>|(ҟ        )��P	X�����A�*

	conv_loss�>=w�P        )��P	
+����A�*

	conv_loss���>C1��        )��P	l\����A�*

	conv_loss;Ť>r>��        )��P	k�����A�*

	conv_loss���>$ۀ        )��P	�����A�*

	conv_loss�ŧ>*�#�        )��P	`�����A�*

	conv_loss>B�?        )��P	�����A�*

	conv_loss���>?.�O        )��P	UL����A�*

	conv_loss���>����        )��P	@|����A�*

	conv_lossA�>ܒV{        )��P	�����A�*

	conv_loss���>���        )��P	C�����A�*

	conv_lossߧ>���W        )��P	�����A�*

	conv_lossɹ�>ݮd�        )��P	dA����A�*

	conv_loss��>��        )��P	�s����A�*

	conv_loss^��>��:�        )��P	d�����A�*

	conv_loss��>�>��        )��P	h�����A�*

	conv_lossBl�>�^i        )��P	�
����A�*

	conv_loss���><���        )��P	R;����A�*

	conv_lossդ>v�        )��P	{k����A�*

	conv_loss&W�>O��b        )��P	�����A�*

	conv_loss��>t	ܧ        )��P	������A�*

	conv_loss���>���V        )��P	_�����A�*

	conv_lossbǤ>���        )��P	�+ ���A�*

	conv_lossyڥ>�)��        )��P	[\ ���A�*

	conv_loss�ǧ>�Zl        )��P	c� ���A�*

	conv_lossb�>��        )��P	1� ���A�*

	conv_loss�F�>	��        )��P	�� ���A�*

	conv_loss;��>��k        )��P	����A�*

	conv_lossց�>��j�        )��P	�N���A�*

	conv_lossGG�>jg��        )��P	����A�*

	conv_loss���>�и/        )��P	�����A�*

	conv_lossT�>��5�        )��P	����A�*

	conv_lossFզ>��N_        )��P	����A�*

	conv_loss�x�>����        )��P	.R���A�*

	conv_loss���>"�5�        )��P	�����A�*

	conv_loss���>���N        )��P	\����A�*

	conv_loss��>R�x�        )��P	~����A�*

	conv_loss�ޥ>��        )��P	���A�*

	conv_loss�M�>R�u        )��P	�G���A�*

	conv_loss�ѥ>�;�        )��P	[z���A�*

	conv_loss�o�>�(a�        )��P	����A�*

	conv_loss$��> �mW        )��P	S����A�*

	conv_loss�a�>HW�        )��P	���A�*

	conv_loss���>�q��        )��P	<E���A�*

	conv_lossL��>�P��        )��P	�x���A�*

	conv_loss��>�)��        )��P	����A�*

	conv_lossS2�>:�        )��P	�����A�*

	conv_lossX��>_�Z        )��P	b���A�*

	conv_loss<r�>+���        )��P	FD���A�*

	conv_lossҦ>�|��        )��P	�s���A�*

	conv_loss�}�>�~��        )��P	�����A�*

	conv_loss�k�>;MP        )��P	�����A�*

	conv_losskť>O��        )��P	����A�*

	conv_loss��>��w        )��P	�6���A�*

	conv_loss?�>� ��        )��P	Ci���A�*

	conv_lossm��>Y��:        )��P	̛���A�*

	conv_loss��><�_�        )��P	�����A�*

	conv_loss���>E�߆        )��P	�����A�*

	conv_loss�@�>��W        )��P	C*���A�*

	conv_loss�|�>@ԙ�        )��P	[���A�*

	conv_losse��>Sn�        )��P	�����A�*

	conv_loss�E�>��        )��P	����A�*

	conv_loss$��>��9�        )��P	�����A�*

	conv_lossɪ�>)Z+�        )��P	�!���A�*

	conv_lossL�>1�,        )��P	�R���A�*

	conv_loss�y�>>bI�        )��P	M����A�*

	conv_loss��>�>@�        )��P	�����A�*

	conv_loss��>~�M�        )��P	!����A�*

	conv_loss;u�>S�v}        )��P	�	���A�*

	conv_loss��>>��        )��P	CF	���A�*

	conv_lossJl�>��G        )��P	{w	���A�*

	conv_loss�|�>^C��        )��P	X�	���A�*

	conv_loss�˩>�w��        )��P	n�	���A�*

	conv_loss�s�>D�]        )��P	�
���A�*

	conv_loss㧨>�Dg>        )��P	|:
���A�*

	conv_loss�>cΝ�        )��P	�|
���A�*

	conv_loss��>��        )��P	�
���A�*

	conv_loss��>��_        )��P	H�
���A�*

	conv_loss���>$��V        )��P	����A�*

	conv_loss�ӣ>_�.        )��P	�B���A�*

	conv_loss���>�]        )��P	x����A�*

	conv_loss���>y�B�        )��P	�����A�*

	conv_loss��>��ޔ        )��P	�����A�*

	conv_loss3X�>�Zrt        )��P	-"���A�*

	conv_loss�`�><r�M        )��P	xW���A�*

	conv_loss_��>��k�        )��P	y����A�*

	conv_lossX¤>�G �        )��P	*����A�*

	conv_loss٧>��'        )��P	�����A�*

	conv_loss쀤>����        )��P	�.���A�*

	conv_lossY��>c+�        )��P	�h���A�*

	conv_loss��>��0{        )��P	ʜ���A�*

	conv_loss��>o�֨        )��P	�����A�*

	conv_loss�5�>�);}        )��P	����A�*

	conv_loss&��>*!        )��P	�>���A�*

	conv_loss(�>8�c�        )��P	�r���A�*

	conv_loss���>q%_        )��P	�����A�*

	conv_loss��>N4|        )��P	�����A�*

	conv_loss��>�7        )��P	���A�*

	conv_lossJ��>��h        )��P	�L���A�*

	conv_lossݣ>�,��        )��P	����A�*

	conv_loss�6�>����        )��P	$����A�*

	conv_loss×�>��0        )��P	����A�*

	conv_lossS��>w�ݡ        )��P	����A�*

	conv_loss�C�>E�]�        )��P	dL���A�*

	conv_loss�>���        )��P	�~���A�*

	conv_loss)��>蚧�        )��P	�����A�*

	conv_loss6إ>v�<E        )��P	e����A�*

	conv_loss3�>�~e2        )��P	����A�*

	conv_lossp�>\ e        )��P	L���A�*

	conv_lossD��>,r�^        )��P	/����A�*

	conv_loss͆�>��i        )��P	l����A�*

	conv_loss'��>�}        )��P	�����A�*

	conv_loss��>1�К        )��P	N���A�*

	conv_lossB�>��|k        )��P	�Q���A�*

	conv_lossٵ�>'�z&        )��P	r����A�*

	conv_loss��>���;        )��P	)����A�*

	conv_losseL�>�\        )��P	c����A�*

	conv_lossL��>5.�_        )��P	Q���A�*

	conv_loss���>��G�        )��P	�Q���A�*

	conv_loss�z�>�gb�        )��P	����A�*

	conv_loss��>g�J�        )��P	����A�*

	conv_loss���>1��@        )��P	�����A�*

	conv_loss8��>,\V        )��P	�#���A�*

	conv_loss"J�>���&        )��P	rW���A�*

	conv_lossc�>�-�        )��P	�����A�*

	conv_loss�w�>F;�        )��P	�����A�*

	conv_loss�k�>w�B        )��P	����A�*

	conv_losstʤ>�(�Z        )��P	�%���A�*

	conv_loss���>}�ݘ        )��P	sY���A�*

	conv_loss��> ��R        )��P	�����A�*

	conv_lossc¦>)��        )��P	�!���A�*

	conv_lossˁ�>?W�        )��P	UT���A�*

	conv_loss�ͤ>���        )��P	)����A�*

	conv_loss��>L���        )��P	�����A�*

	conv_loss�J�>D�K�        )��P	�����A�*

	conv_lossr}�>�        )��P	N#���A�*

	conv_loss�(�>��        )��P	3]���A�*

	conv_loss��>
u:�        )��P	p����A�*

	conv_loss^:�>���        )��P	����A�*

	conv_loss΄�>��%        )��P	����A�*

	conv_loss�4�>O4��        )��P	#;���A�*

	conv_loss�S�>ϰ+\        )��P	�t���A�*

	conv_loss�>
O/V        )��P	3����A�*

	conv_loss�ĥ>���        )��P	����A�*

	conv_losso��>���        )��P	����A�*

	conv_loss3h�>�J5        )��P	�@���A�*

	conv_loss�[�>d"о        )��P	�v���A�*

	conv_loss[��>�G��        )��P	K����A�*

	conv_loss`q�>��K�        )��P	�����A�*

	conv_loss��>��n�        )��P	a���A�*

	conv_loss�E�>T���        )��P	 P���A�*

	conv_loss�n�>mm�L        )��P	 ����A�*

	conv_loss�ݤ>z���        )��P	�����A�*

	conv_loss+ɤ>��;        )��P	�����A�*

	conv_loss�>-ϣZ        )��P	J"���A�*

	conv_loss[�>ڽ�L        )��P	TV���A�*

	conv_loss���>Y�o�        )��P	�����A�*

	conv_loss�
�>E��        )��P	����A�*

	conv_loss@��>�u+�        )��P	����A�*

	conv_loss��>�9�U        )��P	g"���A�*

	conv_loss=��>c�        )��P	�U���A�*

	conv_loss��>E�L�        )��P	d����A�*

	conv_loss���>YF        )��P	,����A�*

	conv_loss셥>�V(�        )��P	�����A�*

	conv_loss1}�>_'        )��P		#���A�*

	conv_loss=�>P [        )��P	�U���A�*

	conv_lossC$�>��[        )��P	�����A�*

	conv_loss�_�>�	        )��P	����A�*

	conv_lossң>�׊�        )��P	{����A�*

	conv_loss���>2��        )��P	O*���A�*

	conv_loss��>�j�        )��P	;c���A�*

	conv_loss�>ا�A        )��P	�����A�*

	conv_lossTG�>�        )��P	�����A�*

	conv_loss��>��D�        )��P	Z ���A�*

	conv_lossZ��>	.        )��P	�8 ���A�*

	conv_loss?�>Rڱ        )��P	�n ���A�*

	conv_loss Φ>^Q�        )��P	�� ���A�*

	conv_loss��>O�ز        )��P	p� ���A�*

	conv_loss��>R���        )��P	�
!���A�*

	conv_loss�S�>IǮ�        )��P	�;!���A�*

	conv_loss��>&���        )��P	`�!���A�*

	conv_loss��>UD'y        )��P	̷!���A�*

	conv_loss�֡>Yn�        )��P	��!���A�*

	conv_loss ��>��7        )��P	 "���A�*

	conv_lossC��>{y�        )��P	qU"���A�*

	conv_loss���>3��{        )��P	��"���A�*

	conv_loss�ϧ>��u        )��P	�"���A�*

	conv_loss�g�>�"�        )��P	�#���A�*

	conv_loss��>�y�&        )��P	:P#���A�*

	conv_lossh��>%�[�        )��P	�#���A�*

	conv_lossl��>��aF        )��P	��#���A�*

	conv_lossm�>�6.�        )��P	��#���A�*

	conv_loss}��>���        )��P	�&$���A�*

	conv_loss�y�>Z� �        )��P	�[$���A�*

	conv_loss.��>�ƭ�        )��P	v�$���A�*

	conv_loss��>�tR        )��P	�$���A�*

	conv_loss�v�>�iS�        )��P	��$���A�*

	conv_loss5��>ﻖ�        )��P	�&%���A�*

	conv_loss�9�>NN,�        )��P	'_%���A�*

	conv_loss���>��ʪ        )��P	Η%���A�*

	conv_loss3�>kX�U        )��P	!�%���A�*

	conv_lossg�>�F��        )��P	��%���A�*

	conv_lossN}�>����        )��P	3&���A�*

	conv_loss���>���        )��P	dg&���A�*

	conv_loss~E�>���        )��P	=�&���A�*

	conv_loss��>���        )��P	}�&���A�*

	conv_loss��>4Ћ        )��P	H'���A�*

	conv_loss�;�>f.��        )��P	�6'���A�*

	conv_lossH�>�	        )��P	pi'���A�*

	conv_lossu��>Z�3�        )��P	��'���A�*

	conv_lossX
�>���v        )��P	O�'���A�*

	conv_loss=$�>a:^        )��P	L(���A�*

	conv_loss?�>Dt�        )��P	�5(���A�*

	conv_loss��>�
�        )��P	k(���A�*

	conv_losss΢>,��\        )��P	J�(���A�*

	conv_lossgG�>��
�        )��P	��(���A�*

	conv_loss}[�>�1        )��P	h)���A�*

	conv_loss෥>��        )��P	�9)���A�*

	conv_loss�~�>�B6W        )��P	ql)���A�*

	conv_loss�8�>���
        )��P	�)���A�*

	conv_loss㕥>�+��        )��P	V�)���A�*

	conv_loss��>�/        )��P	�*���A�*

	conv_loss�s�>h�~        )��P	�7*���A�*

	conv_lossbP�>��        )��P	 j*���A�*

	conv_loss:.�>��F�        )��P	��*���A�*

	conv_loss~��>�v1�        )��P	��*���A�*

	conv_loss춤>K�2        )��P	Y+���A�*

	conv_loss���>����        )��P	p<+���A�*

	conv_loss�[�>3�{�        )��P	�o+���A�*

	conv_loss��>�kJ        )��P	"�+���A�*

	conv_loss ��>:�X        )��P	��+���A�*

	conv_loss�M�>�'�        )��P	�,���A�*

	conv_loss��>�1(        )��P	\Q,���A�*

	conv_loss�w�>��]�        )��P	'�,���A�*

	conv_lossw�>���L        )��P	F�,���A�*

	conv_lossż�>ʑ<�        )��P	C-���A�*

	conv_losse��>�G��        )��P	}<-���A�*

	conv_loss݈�>R��=        )��P	�r-���A�*

	conv_lossg��>+5
	        )��P	P�-���A�*

	conv_lossnT�>�c��        )��P	�-���A�*

	conv_loss�7�>���        )��P	�.���A�*

	conv_loss��>Sz&�        )��P	AD.���A�*

	conv_loss��>y��"        )��P	Kv.���A�*

	conv_loss��>��u�        )��P	�.���A�*

	conv_loss=��>}f�,        )��P	I�.���A�*

	conv_loss��>�ۯM        )��P	�/���A�*

	conv_loss$̤>�k�x        )��P	fJ/���A�*

	conv_loss�[�>oxA�        )��P	��/���A�*

	conv_loss��>��b        )��P	��/���A�*

	conv_lossd-�>�%�        )��P	��/���A�*

	conv_loss���>�h^�        )��P	D0���A�*

	conv_lossd��>)b�        )��P	:N0���A�*

	conv_lossc@�>�J2�        )��P	�0���A�*

	conv_lossBc�>()�        )��P	5�0���A�*

	conv_loss�%�>'�-        )��P	 �0���A�*

	conv_loss���>Ű�K        )��P	;1���A�*

	conv_loss��>���        )��P	RM1���A�*

	conv_lossf$�>C�A�        )��P	�1���A�*

	conv_loss;�>QC�&        )��P	��1���A�*

	conv_lossY��>�� 2        )��P	��1���A�*

	conv_loss�c�>�Rq        )��P	�2���A�*

	conv_loss�ڣ>ψ!�        )��P	�O2���A�*

	conv_loss�t�>��A        )��P	��2���A�*

	conv_loss�>K�ջ        )��P	��2���A�*

	conv_loss���>#���        )��P	�2���A�*

	conv_loss]�>�+��        )��P	�!3���A�*

	conv_loss"��>��K        )��P	�T3���A�*

	conv_loss���>|��        )��P	��3���A�*

	conv_loss��>��7�        )��P	|�3���A�*

	conv_loss�>�>+��e        )��P	��3���A�*

	conv_loss�ʤ>+�        )��P	!4���A�*

	conv_loss���>oi�        )��P	�T4���A�*

	conv_lossG�>Q���        )��P	B�4���A�*

	conv_loss��>�&4�        )��P	H�4���A�*

	conv_loss�>�>0|[k        )��P	+�4���A�*

	conv_loss���>���        )��P	!5���A�*

	conv_loss��>#�jg        )��P	~R5���A�*

	conv_loss���>t��S        )��P	.�5���A�*

	conv_losse�>����        )��P	��5���A�*

	conv_loss2Z�>B��0        )��P	��5���A�*

	conv_loss�$�>�'�        )��P	�46���A�*

	conv_loss���>5�L�        )��P	�f6���A�*

	conv_lossA��>x��        )��P	��6���A�*

	conv_lossl[�>�4�        )��P	n�6���A�*

	conv_lossʫ�>�x�         )��P	:7���A�*

	conv_lossqG�>isC�        )��P	PB7���A�*

	conv_lossx2�>��2	        )��P	�u7���A�*

	conv_loss=��>�N�        )��P	F�7���A�*

	conv_loss^|�>�̙�        )��P	j�7���A�*

	conv_losshN�>IP8�        )��P	n8���A�*

	conv_loss^�>�:�        )��P	BV8���A�*

	conv_loss{k�>L^f        )��P	��8���A�*

	conv_loss�գ>o��        )��P	Ժ8���A�*

	conv_loss��>=z        )��P	��8���A�*

	conv_loss���>_k|        )��P	j9���A�*

	conv_loss� �>c�ϵ        )��P	�U9���A�*

	conv_loss���>/�A        )��P	�9���A�*

	conv_losslޤ>.h�h        )��P	��9���A�*

	conv_loss>ߣ>�B�i        )��P	~�9���A�*

	conv_loss�e�>�m��        )��P	B/:���A�*

	conv_loss��>���        )��P	xb:���A�*

	conv_loss� �>��1        )��P	j�:���A�*

	conv_loss-]�>J{        )��P	��:���A�*

	conv_lossgw�>�	�6        )��P	��:���A�*

	conv_loss̷�>�q�        )��P	2;���A�*

	conv_lossYc�>�N�J        )��P	f;���A�*

	conv_losski�>2gF        )��P	��;���A�*

	conv_lossRT�>���V        )��P	O�;���A�*

	conv_loss�N�>�9<(        )��P	<���A�*

	conv_loss""�>����        )��P	66<���A�*

	conv_loss�A�>6S�m        )��P	*k<���A�*

	conv_lossΟ>���        )��P	T�<���A�*

	conv_loss|]�>�=�b        )��P	��<���A�*

	conv_lossFס>J�F+        )��P	�=���A�*

	conv_lossuܢ><�|n        )��P	V9=���A�*

	conv_lossԥ>��M        )��P	�l=���A�*

	conv_lossa�>��Q        )��P	�=���A�*

	conv_loss,��>���~        )��P	��=���A�*

	conv_loss���>���        )��P	�>���A�*

	conv_loss��>���        )��P	�9>���A�*

	conv_loss�J�>.0�e        )��P	�l>���A�*

	conv_loss�'�>�}�        )��P	��>���A�*

	conv_loss"K�>W
��        )��P	D�>���A�*

	conv_lossa��>�j��        )��P	8?���A�*

	conv_lossP��>=�A\        )��P	y<?���A�*

	conv_loss{R�>Ǩ:5        )��P	4p?���A�*

	conv_loss���>b�L�        )��P	/�?���A�*

	conv_loss6N�>Q�nq        )��P	��?���A�*

	conv_loss��>��        )��P	@���A�*

	conv_loss}�>dA        )��P	Y�D���A�*

	conv_loss�$�>���q        )��P	vF���A�*

	conv_loss�2�>�%�        )��P	˴F���A�*

	conv_loss齢>����        )��P	R�F���A�*

	conv_loss=��>N��        )��P	�G���A�*

	conv_losseY�>-��        )��P	cIG���A�*

	conv_loss���>��S        )��P	yG���A�*

	conv_loss�u�>V-��        )��P	u�G���A�*

	conv_lossǑ�>	��)        )��P	�G���A�*

	conv_loss<�>"=1�        )��P	DH���A�*

	conv_lossO2�><�        )��P	[CH���A�*

	conv_lossGk�>r�s        )��P	|H���A�*

	conv_loss��>�c~�        )��P	��H���A�*

	conv_loss"��>VBz$        )��P	}�H���A�*

	conv_loss���><��T        )��P	wI���A�*

	conv_loss�@�>�}��        )��P	�NI���A�*

	conv_loss���>�z|        )��P	�~I���A�*

	conv_loss4��>Nn9        )��P	��I���A�*

	conv_lossf��>��7�        )��P	Y�I���A�*

	conv_loss���>$&c2        )��P	rJ���A�*

	conv_lossȱ�>J5�n        )��P	Y8J���A�*

	conv_loss��>�E'	        )��P	gJ���A�*

	conv_loss�@�>�.L�        )��P	Y�J���A�*

	conv_loss�à>��:w        )��P	�J���A�*

	conv_loss���>G��        )��P	��J���A�*

	conv_loss��>3���        )��P	�'K���A�*

	conv_loss휥>��6z        )��P	�YK���A�*

	conv_lossED�>?ϡJ        )��P	�K���A�*

	conv_loss�O�>Ⳁ�        )��P	#�K���A�*

	conv_losszN�>����        )��P	��K���A�*

	conv_loss�$�>��        )��P	�L���A�*

	conv_loss?آ>�v        )��P	`KL���A�*

	conv_lossн�>�        )��P	{zL���A�*

	conv_loss�1�>����        )��P	�L���A�*

	conv_loss�?�>����        )��P	��L���A�*

	conv_loss���>Mhz�        )��P	fM���A�*

	conv_loss�R�>���d        )��P	:5M���A�*

	conv_loss/գ>d:��        )��P	�dM���A�*

	conv_lossgc�>+&%�        )��P		�M���A�*

	conv_loss<C�>���o        )��P	m�M���A�*

	conv_loss�M�>DPG        )��P	��M���A�*

	conv_loss_L�>�k��        )��P	#N���A�*

	conv_loss$L�>�(        )��P	~QN���A�*

	conv_loss�w�>�+�        )��P	��N���A�*

	conv_lossJ��>�dK        )��P	��N���A�*

	conv_loss�m�>� {        )��P	C�N���A�*

	conv_lossw��>(8�O        )��P	�O���A�*

	conv_loss�ף>mS�        )��P	�AO���A�*

	conv_loss[��>$|O        )��P	�oO���A�*

	conv_lossxӠ>O��        )��P	��O���A�*

	conv_loss�o�>q���        )��P	Z�O���A�*

	conv_loss�X�>R�        )��P	P���A�*

	conv_loss⑤>/ޡ         )��P	�AP���A�*

	conv_loss���>�j�Z        )��P	�pP���A�*

	conv_loss[��>~��        )��P	T�P���A�*

	conv_lossŬ�>��        )��P	x�P���A�*

	conv_lossmɤ>�2GH        )��P	VQ���A�*

	conv_loss���>���        )��P	�>Q���A�*

	conv_lossQʠ>�W�        )��P	1nQ���A�*

	conv_loss�>�L3|        )��P	ҝQ���A�*

	conv_loss��>"b?�        )��P	��Q���A�*

	conv_loss�~�>�\��        )��P	��Q���A�*

	conv_loss�{�>~B߆        )��P	�.R���A�*

	conv_loss7�>�J�        )��P	�]R���A�*

	conv_lossٗ�>"!�        )��P	q�R���A�*

	conv_losseף>=�1�        )��P	<�R���A�*

	conv_lossH�>-��        )��P	x	S���A�*

	conv_loss-&�>�5�        )��P	�9S���A�*

	conv_lossB��>�Ij�        )��P	�kS���A�*

	conv_loss�$�>!�v        )��P	��S���A�*

	conv_loss�§>�R7�        )��P	C�S���A�*

	conv_loss�p�>�qn        )��P	X�S���A�*

	conv_loss�>v���        )��P	�*T���A�*

	conv_loss�w�>PUKo        )��P	UYT���A�*

	conv_loss��>��KX        )��P	��T���A�*

	conv_lossṃ>"c�        )��P	U�T���A�*

	conv_loss�'�>YQ        )��P	c�T���A�*

	conv_loss���>b�/�        )��P	}U���A�*

	conv_loss�.�>ڀt"        )��P	�GU���A�*

	conv_loss/b�>���        )��P	{xU���A�*

	conv_lossi�>.3        )��P	��U���A�*

	conv_loss�ף>���=        )��P	��U���A�*

	conv_loss*�>	��        )��P	/V���A�*

	conv_loss��>�.�        )��P	�3V���A�*

	conv_loss�Q�>
��*        )��P	cV���A�*

	conv_lossǡ�>1��        )��P	ђV���A�*

	conv_loss�Ѧ>���        )��P	��V���A�*

	conv_loss�D�>�	�W        )��P	��V���A�*

	conv_loss�=�>4�v�        )��P	�#W���A�*

	conv_loss��>�ұ0        )��P	�TW���A�*

	conv_loss�٢>��%Y        )��P	C�W���A�*

	conv_loss3��>�TX        )��P	ŸW���A�*

	conv_loss ��>ݝ�:        )��P	��W���A�*

	conv_loss���>�"d        )��P	�X���A�*

	conv_loss/�>䷵�        )��P	�EX���A�*

	conv_loss=£>#^_�        )��P	yX���A�*

	conv_loss��>��H�        )��P	�X���A�*

	conv_loss�$�>�ǻS        )��P	!�X���A�*

	conv_loss��>�ɪ/        )��P	�Y���A�*

	conv_loss���>o�L        )��P	�:Y���A�*

	conv_loss�e�>W�l        )��P	�iY���A�*

	conv_loss�7�>���        )��P	*�Y���A�*

	conv_loss���>p_��        )��P	��Y���A�*

	conv_lossア>�?�        )��P	�Z���A�*

	conv_loss�!�>6��        )��P	�EZ���A�*

	conv_loss��>ۋ)p        )��P	�tZ���A�*

	conv_loss*"�>J��U        )��P	��Z���A�*

	conv_losst5�>��k         )��P	��Z���A�*

	conv_lossA(�>��
        )��P	�	[���A�*

	conv_lossW�>���        )��P	F[���A�*

	conv_loss�Х>Y���        )��P	S~[���A�*

	conv_loss���>� ;�        )��P	޳[���A�*

	conv_lossG�>o��        )��P	?�[���A�*

	conv_loss��>g��~        )��P	�\���A�*

	conv_loss�>Lʞ�        )��P	�V\���A�*

	conv_lossL?�>'��        )��P	��\���A�*

	conv_loss1��>�=�        )��P	p�\���A�*

	conv_lossȭ�>s��#        )��P	��\���A�*

	conv_loss!��>¹v�        )��P	R-]���A�*

	conv_loss�ƣ>ծ�)        )��P	�`]���A�*

	conv_loss�A�>��        )��P	M�]���A�*

	conv_loss�/�>�33(        )��P	��]���A�*

	conv_loss'U�>l�`6        )��P	D�]���A�*

	conv_loss�T�>��        )��P	�*^���A�*

	conv_loss��>.�9f        )��P	�\^���A�*

	conv_loss^��>�Uq        )��P	�^���A�*

	conv_loss��>Hv]        )��P	�^���A�*

	conv_loss��>
)�        )��P	@�^���A�*

	conv_lossj��>�}�K        )��P	b&_���A�*

	conv_loss��>�X�8        )��P	AX_���A�*

	conv_loss��>���        )��P	ʋ_���A�*

	conv_loss/�>���}        )��P	��_���A�*

	conv_lossߌ�>�&�        )��P	/�_���A�*

	conv_loss,�>d^D�        )��P	�#`���A�*

	conv_lossZa�>��!�        )��P	FT`���A�*

	conv_loss�>f�^        )��P	��`���A�*

	conv_loss��>ej��        )��P	ʸ`���A�*

	conv_loss�A�>�¤�        )��P	��`���A�*

	conv_loss�#�>����        )��P	 a���A�*

	conv_loss��>�>�z        )��P	�Ra���A�*

	conv_loss<�>͇$        )��P	"�a���A�*

	conv_loss'>�>}=$R        )��P	4�a���A�*

	conv_loss��>H�ڱ        )��P	�a���A�*

	conv_loss׏�>�S��        )��P	b���A�*

	conv_loss!�>�a�        )��P	�Kb���A�*

	conv_loss�>Ə�r        )��P	e~b���A�*

	conv_lossaE�>�'c        )��P	ׯb���A�*

	conv_loss��>�yo�        )��P	c�b���A�*

	conv_loss{��>��ͥ        )��P	4c���A�*

	conv_loss���>��t�        )��P	{Fc���A�*

	conv_loss���>>I��        )��P	;wc���A�*

	conv_loss��>�\�        )��P	�c���A�*

	conv_lossu�>:m�        )��P	y�c���A�*

	conv_loss���>MM��        )��P	� d���A�*

	conv_loss�n�>i�-�        )��P	�Ud���A�*

	conv_loss�|�>�m��        )��P	ŉd���A�*

	conv_loss/��>�X!=        )��P	ڻd���A�*

	conv_loss���>���        )��P	��d���A�*

	conv_loss��><:3�        )��P	( e���A�*

	conv_lossy��>�}-v        )��P	�Ue���A�*

	conv_loss�:�>�x}        )��P	�e���A�*

	conv_lossz@�>	\L0        )��P	c�e���A�*

	conv_loss�>��V�        )��P	��e���A�*

	conv_loss�ء>��[        )��P	.-f���A�*

	conv_loss=�>��d        )��P	p_f���A�*

	conv_loss�'�>��6        )��P	9�f���A�*

	conv_lossS>�>ʷ��        )��P	��f���A�*

	conv_lossq��>`�S)        )��P	�g���A�*

	conv_loss��>�V��        )��P	E7g���A�*

	conv_loss7١>�d	.        )��P	�hg���A�*

	conv_loss"��>��6         )��P	�g���A�*

	conv_loss�£>͟�&        )��P	��g���A�*

	conv_loss-T�>�i6        )��P	�g���A�*

	conv_lossEã>�nMD        )��P	�2h���A�*

	conv_loss�>Tr��        )��P	�dh���A�*

	conv_loss�%�>�[l�        )��P	(�h���A�*

	conv_loss���>$�Cy        )��P	�h���A�*

	conv_lossޢ�>S���        )��P	��h���A�*

	conv_lossǢ>�s�        )��P	�+i���A�*

	conv_lossu��>�Q38        )��P	Q^i���A�*

	conv_lossH��>`:�        )��P	��i���A�*

	conv_lossN�>�Ҏ        )��P	I�i���A�*

	conv_lossk��>o�9:        )��P	��i���A�*

	conv_lossH��>ݚ6        )��P	n*j���A�*

	conv_loss�Т>;d��        )��P	X]j���A�*

	conv_loss��>���        )��P	Q�j���A�*

	conv_loss=�>BAZ�        )��P	��j���A�*

	conv_loss���>O�-8        )��P	#�j���A�*

	conv_loss_��>����        )��P	�&k���A�*

	conv_loss蟣>��!        )��P	�Yk���A�*

	conv_loss�̡>$         )��P	y�k���A�*

	conv_lossm��>����        )��P	+�k���A�*

	conv_loss�ʟ>\aţ        )��P	��k���A�*

	conv_lossFn�>gO��        )��P	G&l���A�*

	conv_lossH��>=��        )��P	![l���A�*

	conv_lossݡ�>u���        )��P	�l���A�*

	conv_loss�Ý>���        )��P	��l���A�*

	conv_loss���>��        )��P	R�l���A�*

	conv_loss�=�>�!j�        )��P	y)m���A�*

	conv_lossܤ>�O��        )��P	�[m���A�*

	conv_loss���>d��U        )��P	y�m���A�*

	conv_loss�>P        )��P	R-o���A�*

	conv_loss���>����        )��P	�eo���A�*

	conv_lossqS�>�8        )��P	y�o���A�*

	conv_lossk�>U+�@        )��P	�o���A�*

	conv_lossͣ�>����        )��P	Fp���A�*

	conv_loss�>0��5        )��P	�9p���A�*

	conv_loss�%�>�O/�        )��P	nip���A�*

	conv_loss*Ƥ>~�        )��P	&�p���A�*

	conv_loss�F�>-;t�        )��P	2�p���A�*

	conv_loss�	�>�u�        )��P	�q���A�*

	conv_loss��>�s�A        )��P	�Aq���A�*

	conv_loss��>�ʁG        )��P	�rq���A�*

	conv_lossr�>����        )��P	q�q���A�*

	conv_lossN �>�ѯS        )��P	4�q���A�*

	conv_loss���>'!��        )��P	Hr���A�*

	conv_loss9��>�W8"        )��P	o?r���A�*

	conv_loss�O�>O
p;        )��P	Enr���A�*

	conv_loss��>�5�        )��P	��r���A�*

	conv_lossm)�>��n        )��P	��r���A�*

	conv_losss�>��\�        )��P	��r���A�*

	conv_lossmš>��        )��P	b,s���A�*

	conv_lossX�>=�/�        )��P	/cs���A�*

	conv_loss���>���        )��P	��s���A�*

	conv_loss���>��*�        )��P	��s���A�*

	conv_lossH�>/8        )��P	@�s���A�*

	conv_loss��>4`Ƽ        )��P	Q.t���A�*

	conv_loss���>���        )��P	L^t���A�*

	conv_loss�>�        )��P	A�t���A�*

	conv_loss* �>
��8        )��P	��t���A�*

	conv_loss���>"c��        )��P	A�t���A�*

	conv_loss�Ɵ>#.��        )��P	u���A�*

	conv_loss%��>�D�        )��P	�Pu���A�*

	conv_loss|Ӥ>�6��        )��P	�u���A�*

	conv_loss	��>��        )��P	|�u���A�*

	conv_loss�L�>��h        )��P	z�u���A�*

	conv_loss/d�><$��        )��P	�v���A�*

	conv_loss���>e\�        )��P	Dv���A�*

	conv_loss#�>��<        )��P	�sv���A�*

	conv_loss�X�>=d��        )��P	�v���A�*

	conv_loss��>ج
        )��P	��v���A�*

	conv_loss�ա>��@        )��P	�w���A�*

	conv_losss�>�h��        )��P	�2w���A�*

	conv_losse/�>x$�&        )��P	�gw���A�*

	conv_loss`��>���        )��P	ۙw���A�*

	conv_loss'��>D�l        )��P	��w���A�*

	conv_loss�#�>,B�z        )��P	-�w���A�*

	conv_loss�ơ>��=�        )��P	�-x���A�*

	conv_loss/W�>�\�G        )��P	2\x���A�*

	conv_lossȥ>2Wg�        )��P	y�x���A�*

	conv_loss�$�>ۮm�        )��P	��x���A�*

	conv_lossc�>�ix�        )��P	D�x���A�*

	conv_lossX�>�V         )��P	�/y���A�*

	conv_loss���>)��        )��P	�ay���A�*

	conv_loss
�>�o�        )��P	A�y���A�*

	conv_lossh(�>�_        )��P	�y���A�*

	conv_loss���>'3�        )��P	U�y���A�*

	conv_loss��>`\�x        )��P	o'z���A�*

	conv_loss�n�>��Rj        )��P	afz���A�*

	conv_loss���>�\�        )��P	�z���A�*

	conv_lossf��>�c�w        )��P	=�z���A�*

	conv_loss��>�}2Y        )��P	{���A�*

	conv_loss���>2o6�        )��P	E:{���A�*

	conv_lossù�>��.�        )��P	s{���A�*

	conv_lossN�>7<m         )��P	��{���A�*

	conv_loss��>Q[��        )��P	8�{���A�*

	conv_loss0�>�r�
        )��P	�|���A�*

	conv_loss��>����        )��P	�4|���A�*

	conv_lossت�>AӞm        )��P	�o|���A�*

	conv_lossT;�>��Z�        )��P	�|���A�*

	conv_losss3�>��I        )��P	}�|���A�*

	conv_losswE�>�чb        )��P	��|���A�*

	conv_lossٗ�>�]�S        )��P	}/}���A�*

	conv_losspע>pd         )��P	�d}���A�*

	conv_loss=��>I2̅        )��P	��}���A�*

	conv_loss�g�>K�{�        )��P	o�}���A�*

	conv_lossu��>2�{�        )��P	~���A�*

	conv_loss�`�>��r        )��P	 @~���A�*

	conv_lossXڥ>��si        )��P	�m~���A�*

	conv_loss��>�,a�        )��P	��~���A�*

	conv_loss"�>Y�N        )��P	*�~���A�*

	conv_loss�L�>'�X        )��P	����A�*

	conv_loss�s�>�<�	        )��P	�5���A�*

	conv_loss�ܣ>����        )��P	�d���A�*

	conv_lossLǣ>��Wu        )��P	�����A�*

	conv_loss�Q�>��i        )��P	�����A�*

	conv_lossI4�>���        )��P	�����A�*

	conv_loss�ՠ>�6�Q        )��P		����A�*

	conv_loss!��>��z        )��P	VK����A�*

	conv_loss���>���        )��P	^z����A�*

	conv_loss�o�>����        )��P	B�����A�*

	conv_lossn9�>���C        )��P	i����A�*

	conv_lossBП>Z�3        )��P	(����A�*

	conv_lossP�>��d        )��P	�E����A�*

	conv_loss9�>���@        )��P	�u����A�*

	conv_lossj(�>���        )��P	������A�*

	conv_loss�-�>��        )��P	7ԁ���A�*

	conv_losss��>V$�]        )��P	�����A�*

	conv_loss/��>�H	        )��P	06����A�*

	conv_loss���>��)�        )��P	�f����A�*

	conv_loss�z�>����        )��P	������A�*

	conv_loss��>.�^C        )��P		܂���A�*

	conv_loss/С>��        )��P	�����A�*

	conv_loss���>\���        )��P	 ?����A�*

	conv_lossJ^�>�t*I        )��P	�p����A�*

	conv_lossj��>�e�        )��P	z�����A�*

	conv_lossD��>ц�B        )��P	
����A�*

	conv_loss
��>ַ^�        )��P	�����A�*

	conv_loss��>�!�N        )��P	F����A�*

	conv_loss�¢>�g        )��P	�����A�*

	conv_loss��>-��        )��P	������A�*

	conv_lossU!�>q3�|        )��P	z����A�*

	conv_loss��>��->        )��P	+����A�*

	conv_loss��>�I~        )��P	da����A�*

	conv_loss:b�>;��        )��P	������A�*

	conv_loss�v�>l�9        )��P	i˅���A�*

	conv_loss�s�>J�        )��P	����A�*

	conv_loss/�>�4��        )��P	75����A�*

	conv_loss��>�t�        )��P	Df����A�*

	conv_loss�>��        )��P	X�����A�*

	conv_loss�d�>���        )��P	.ˆ���A�*

	conv_loss�q�>�y�        )��P	�����A�*

	conv_loss��>e�]        )��P	�3����A�*

	conv_loss��>���        )��P	�f����A�*

	conv_loss�̡>�[L�        )��P	i�����A�*

	conv_loss��>�Q��        )��P	�ڇ���A�*

	conv_lossϡ>��(�        )��P	&����A�*

	conv_loss/Z�>��i        )��P	@����A�*

	conv_loss>z|�/        )��P	Ks����A�*

	conv_lossg��>	1�v        )��P	������A�*

	conv_losshڡ>F�fC        )��P	$و���A�*

	conv_loss�V�>�]�I        )��P	����A�*

	conv_lossKמ>�T&        )��P	"B����A�*

	conv_loss�1�>����        )��P	qt����A�*

	conv_loss��>��,        )��P	������A�*

	conv_loss&�>J�n        )��P	A։���A�*

	conv_loss�Ģ>�i        )��P	�����A�*

	conv_loss�"�>���        )��P	G7����A�*

	conv_lossޟ>���        )��P	h����A�*

	conv_loss袡>�S��        )��P	\�����A�*

	conv_loss�7�>mr�        )��P	Ǌ���A�*

	conv_loss��>`wY        )��P	������A�*

	conv_loss��>�q�a        )��P	�)����A�*

	conv_lossM6�>�J~�        )��P	"\����A�*

	conv_loss胠>�5&        )��P	Y�����A�*

	conv_loss�h�>��"F        )��P	㽋���A�*

	conv_loss���>�[         )��P	n����A�*

	conv_lossQ�>�q�        )��P	�%����A�*

	conv_loss�ڢ>G�        )��P	�W����A�*

	conv_lossNȠ>x�AQ        )��P	Q�����A�*

	conv_loss,]�>j
�        )��P	������A�*

	conv_lossͶ�>�0�        )��P	������A�*

	conv_loss�^�>>��        )��P	�5����A�*

	conv_loss"%�>t��        )��P	�h����A�*

	conv_loss4�>���        )��P	�����A�*

	conv_loss��>ns8        )��P	�ˍ���A�*

	conv_loss<Ġ>����        )��P	������A�*

	conv_loss�˟>����        )��P	�8����A�*

	conv_loss/ՠ>sKf        )��P	h����A�*

	conv_loss��>>�9T        )��P	�����A�*

	conv_loss�&�>�+Ê        )��P	*֎���A�*

	conv_loss٩�>���y        )��P	�����A�*

	conv_loss���>�؛V        )��P	�L����A�*

	conv_loss5	�>�V�        )��P	�����A�*

	conv_loss��>���        )��P	������A�*

	conv_lossm��>�Ǝ�        )��P	 ����A�*

	conv_lossɬ�>]�>        )��P	����A�*

	conv_loss/r�>n��        )��P	mI����A�*

	conv_loss׀�>�fYP        )��P	=|����A�*

	conv_loss�H�>�Is        )��P	ۭ����A�*

	conv_loss���>bE        )��P	ߐ���A�*

	conv_loss�t�>�MS        )��P	?����A�*

	conv_loss��>xuڑ        )��P	�A����A�*

	conv_loss�y�>u=�.        )��P	�q����A�*

	conv_lossk�>����        )��P	������A�*

	conv_loss(?�>Av��        )��P	�ϑ���A�*

	conv_loss���>�@S�        )��P	U����A�*

	conv_loss_��>����        )��P	w2����A�*

	conv_losscQ�>ʫ�        )��P	zb����A�*

	conv_lossۏ�>� �S        )��P	������A�*

	conv_lossUԦ>���I        )��P	rʒ���A�*

	conv_lossK�>1�:        )��P	]�����A�*

	conv_loss��>���&        )��P	|,����A�*

	conv_lossQǞ>�x�        )��P	id����A�*

	conv_loss�^�>(�+�        )��P	������A�*

	conv_lossjޡ>JVh_        )��P	�Ɠ���A�*

	conv_loss���>����        )��P	������A�*

	conv_loss��>��q        )��P	�.����A�*

	conv_loss�r�>� R�        )��P	�l����A�*

	conv_lossgk�>����        )��P	%�����A�*

	conv_loss�S�>\G�,        )��P	IҔ���A�*

	conv_lossx&�>F\�K        )��P	&����A�*

	conv_lossz/�>��        )��P	3����A�*

	conv_loss[�>HD��        )��P	Ud����A�*

	conv_loss�g�>+�\�        )��P	?�����A�*

	conv_loss���>�}kc        )��P	ە���A�*

	conv_loss��>\N�        )��P	�����A�*

	conv_lossU�>w���        )��P	6B����A�*

	conv_loss���>dj0        )��P	|x����A�*

	conv_loss�}�>'�S        )��P	������A�*

	conv_loss�ߠ>���x        )��P	hۖ���A�*

	conv_loss��>Ba�        )��P	Vn����A�*

	conv_loss#�>D        )��P	�����A�*

	conv_loss`ˠ>wT        )��P	�И���A�*

	conv_loss4��>��        )��P	�����A�*

	conv_loss�>�`�v        )��P	.8����A�*

	conv_loss�!�>P�&�        )��P	5i����A�*

	conv_loss�;�>:'|        )��P	������A�*

	conv_lossX��>��        )��P	�ә���A�*

	conv_loss���>XԺ�        )��P	�
����A�*

	conv_loss�@�>7;�        )��P	�>����A�*

	conv_loss�Ȟ>�{W�        )��P	�n����A�*

	conv_loss���>�5��        )��P	׫����A�*

	conv_loss(7�>�ΐ8        )��P	5ݚ���A�*

	conv_loss+�>��|J        )��P	n����A�*

	conv_loss��>r��        )��P	A����A�*

	conv_loss��>7Y<�        )��P	�q����A�*

	conv_loss9�>,���        )��P	8�����A�*

	conv_loss�_�>Yj�        )��P	#����A�*

	conv_loss�ȡ>g�c�        )��P	V$����A�*

	conv_lossǜ>��I�        )��P	eZ����A�*

	conv_loss+�>'��        )��P	}�����A�*

	conv_loss��>�B^�        )��P	�̜���A�*

	conv_lossy�>%Ĕ�        )��P	F�����A�*

	conv_lossk�>J>��        )��P	�/����A�*

	conv_loss�Y�>û!:        )��P	�_����A�*

	conv_loss�i�>e5�k        )��P	V�����A�*

	conv_loss;l�> ��(        )��P	5ŝ���A�*

	conv_loss烈>����        )��P	|�����A�*

	conv_loss���>�F�        )��P	W(����A�*

	conv_loss� �>�|��        )��P	sY����A�*

	conv_loss ܟ>Y��        )��P	�����A�*

	conv_loss�y�>�w�x        )��P	�Ş���A�*

	conv_loss'&�>���        )��P	������A�*

	conv_lossv��>\�gl        )��P	�)����A�*

	conv_loss�s�>��f�        )��P	Z����A�*

	conv_loss�_�>N�ֱ        )��P	0�����A�*

	conv_lossGG�>�G~        )��P	�ğ���A�*

	conv_loss1 �>]�ޡ        )��P	������A�*

	conv_loss���>pƖ        )��P	�(����A�*

	conv_loss�g�>a��        )��P	�\����A�*

	conv_lossGl�>��        )��P	������A�*

	conv_loss[��>�S        )��P	ɼ����A�*

	conv_losss��>��X        )��P	S�����A�*

	conv_loss�0�>�Q�        )��P	j����A�*

	conv_loss��>�[�        )��P	QO����A�*

	conv_loss�%�>E'��        )��P	؀����A�*

	conv_lossq&�>�\�        )��P	������A�*

	conv_loss�9�>br*�        )��P	@����A�*

	conv_loss���>H�ĵ        )��P	�����A�*

	conv_losst�>���        )��P	<B����A�*

	conv_loss�>�b�M        )��P	Ⅲ���A�*

	conv_loss��>�4H5        )��P	׵����A�*

	conv_loss�A�>��b(        )��P	������A�*

	conv_loss�4�>p��c        )��P	� ����A�*

	conv_lossL��>�t�        )��P	sR����A�*

	conv_lossu��>�X�        )��P	-�����A�*

	conv_loss���>58�8        )��P	k�����A�*

	conv_lossQ�>��7H        )��P	~����A�*

	conv_lossѝ>��S�        )��P	�&����A�*

	conv_loss?Z�>C�*        )��P	�[����A�*

	conv_lossV�>F#��        )��P	������A�*

	conv_losse~�>��s�        )��P	������A�*

	conv_loss���>$���        )��P	������A�*

	conv_lossu��>*��F        )��P	o*����A�*

	conv_loss���>���        )��P	�Z����A�*

	conv_loss��>09*�        )��P	����A�*

	conv_lossjd�>��K        )��P	4٥���A�*

	conv_loss=�>*�b        )��P	6����A�*

	conv_loss��>�5��        )��P	,B����A�*

	conv_lossŨ�>y��        )��P	�q����A�*

	conv_loss<�>��nb        )��P	ģ����A�*

	conv_loss�%�>��
        )��P	����A�*

	conv_loss֠�>�p�{        )��P	����A�*

	conv_loss�%�>F8        )��P	�L����A�*

	conv_losspk�>��L�        )��P	[}����A�*

	conv_loss�%�>��}�        )��P	ʮ����A�*

	conv_losse�>z�E�        )��P	����A�*

	conv_loss��>���        )��P	&����A�*

	conv_loss��>l~=        )��P	�B����A�*

	conv_loss\�>:��g        )��P	;s����A�*

	conv_lossyR�>/9(        )��P	������A�*

	conv_lossf�>K���        )��P	Ԩ���A�*

	conv_loss'$�>l���        )��P	}����A�*

	conv_loss4�>�s��        )��P	�;����A�*

	conv_loss�֟>��!�        )��P	�k����A�*

	conv_losse��>l�        )��P	������A�*

	conv_loss@_�>ƐA=        )��P	�Ω���A�*

	conv_lossѼ�>� �[        )��P	U�����A�*

	conv_lossI��>�g�        )��P	�/����A�*

	conv_loss#��>��        )��P	�d����A�*

	conv_loss�מ>=A+q        )��P	x�����A�*

	conv_lossJD�>�EM4        )��P	�Ȫ���A�*

	conv_lossș�>D�.j        )��P	������A�*

	conv_loss	�>(jQ        )��P	|)����A�*

	conv_loss!��>��A�        )��P	�Y����A�*

	conv_lossh!�>���z        )��P	Ȍ����A�*

	conv_loss��>t��        )��P	sī���A�*

	conv_loss[�>��a        )��P	������A�*

	conv_loss}��>hG�$        )��P	�4����A�*

	conv_loss���>��?        )��P	h۰���A�*

	conv_loss���>&�S%        )��P	����A�*

	conv_loss"�>��kp        )��P	�N����A�*

	conv_lossU��>L��        )��P	�����A�*

	conv_loss���>��?�        )��P	A�����A�*

	conv_loss=��>݅��        )��P	�����A�*

	conv_loss�Ϡ>��        )��P	����A�*

	conv_loss��> �BY        )��P	cZ����A�*

	conv_loss��>C��Z        )��P	������A�*

	conv_loss��>�u��        )��P	�Ȳ���A�*

	conv_loss�&�>��_Y        )��P	w�����A�*

	conv_loss���>�["<        )��P	�-����A�*

	conv_loss�>�>aB�|        )��P	e����A�*

	conv_loss~N�>�J�        )��P	������A�*

	conv_loss+��>_�H�        )��P	�ó���A�*

	conv_loss)�>��        )��P	�����A�*

	conv_loss\�>�+        )��P	 $����A�*

	conv_lossl��>խ��        )��P	�V����A�*

	conv_loss�̞>J��        )��P	І����A�*

	conv_losso��>JYx�        )��P	̶����A�*

	conv_loss[/�>���;        )��P	����A�*

	conv_loss��>���T        )��P	%����A�*

	conv_loss�q�>�;�        )��P	�W����A�*

	conv_loss��>o�&�        )��P	v�����A�*

	conv_loss��>�|�        )��P	������A�*

	conv_loss兤>C��        )��P	#����A�*

	conv_loss��>I-��        )��P	�����A�*

	conv_loss؟>����        )��P	�H����A�*

	conv_loss/�>=        )��P	�w����A�*

	conv_loss�¥>�q��        )��P	R�����A�*

	conv_lossj��>�+L        )��P	cض���A�*

	conv_loss�>؎�R        )��P	����A�*

	conv_loss@Ԡ>x��        )��P	�5����A�*

	conv_loss*%�>"&~	        )��P	af����A�*

	conv_loss���>6D��        )��P	������A�*

	conv_loss7�>��jb        )��P	�ŷ���A�*

	conv_loss}�>��        )��P	�����A�*

	conv_loss�c�>�o��        )��P	�%����A�*

	conv_loss(��>i`]        )��P	�U����A�*

	conv_loss!0�>eM;        )��P	�����A�*

	conv_loss�i�>4b�        )��P	/�����A�*

	conv_loss3\�>�,�        )��P	�����A�*

	conv_loss��>ٓ�        )��P	6F����A�*

	conv_lossm��>��2V        )��P	r~����A�*

	conv_lossY�>I�'        )��P	#�����A�*

	conv_loss�ՠ>�/0        )��P	����A�*

	conv_loss�>� Av        )��P	�&����A�*

	conv_lossXK�>���        )��P	e[����A�*

	conv_lossl��>O?�        )��P	L�����A�*

	conv_loss��>�8A<        )��P	 ̺���A�*

	conv_loss��>��        )��P	������A�*

	conv_loss���>�G�_        )��P	�N����A�*

	conv_loss�)�>���~        )��P	������A�*

	conv_lossaL�>%=��        )��P	������A�*

	conv_lossH�>Q�7�        )��P	K����A�*

	conv_loss�6�>-��        )��P	� ����A�*

	conv_loss"�>&��N        )��P	�U����A�*

	conv_lossOk�>�ր8        )��P	.�����A�*

	conv_lossK��>g2��        )��P	Aμ���A�*

	conv_loss>�>G�ˑ        )��P	����A�*

	conv_lossX�>��G�        )��P	�:����A�*

	conv_loss�u�>#�A        )��P	�i����A�*

	conv_lossx*�>�}�|        )��P	R�����A�*

	conv_loss���>=g+        )��P	I׽���A�*

	conv_lossC�>]x �        )��P	!����A�*

	conv_loss���>�=��        )��P	W>����A�*

	conv_loss��>�̕�        )��P	m����A�*

	conv_losss2�>Z�n�        )��P	������A�*

	conv_loss��>vy��        )��P	yپ���A�*

	conv_lossMЛ>G���        )��P	0����A�*

	conv_lossM6�>N��0        )��P	�O����A�*

	conv_loss-٢>�"��        )��P	w�����A�*

	conv_loss�u�>&J�.        )��P	������A�*

	conv_loss��>��        )��P	�����A�*

	conv_loss�̞>��O        )��P	�(����A�*

	conv_lossf�>��ޗ        )��P	gZ����A�*

	conv_loss!��>�GX�        )��P	������A�*

	conv_loss���>~k .        )��P	D�����A�*

	conv_loss�:�>�w�        )��P	4�����A�*

	conv_lossk�>��        )��P	k/����A�*

	conv_loss9۠>m��         )��P	�_����A�*

	conv_loss⥠>�G�h        )��P	՛����A�*

	conv_loss�A�>���        )��P	������A�*

	conv_loss<�>U[��        )��P	�����A�*

	conv_loss��>s6��        )��P	�M����A�*

	conv_losssݟ>�J�        )��P	�����A�*

	conv_lossY��>��o�        )��P	������A�*

	conv_lossZy�>f���        )��P	v�����A�*

	conv_lossT�>�o�        )��P	�����A�*

	conv_loss&@�>����        )��P	Z����A�*

	conv_loss4}�>Z���        )��P	������A�*

	conv_loss/��>�oJN        )��P	�����A�*

	conv_loss��>D��        )��P	L�����A�*

	conv_loss�4�>)Pg        )��P	P6����A�*

	conv_loss6�>��        )��P	k����A�*

	conv_loss��>�t&�        )��P	�����A�*

	conv_loss�բ>��2        )��P	������A�*

	conv_loss�>|G��        )��P	�����A�*

	conv_lossf�>q��        )��P	�E����A�*

	conv_loss��>ͱ�        )��P	ev����A�*

	conv_loss�Ƞ>���M        )��P	T�����A�*

	conv_lossm��>&F��        )��P	$B����A�*

	conv_loss ١>puM�        )��P	�t����A�*

	conv_loss89�>0�        )��P	������A�*

	conv_loss�>�kI�        )��P	n�����A�*

	conv_lossq��>�.�Q        )��P	�����A�*

	conv_lossl��>>J�        )��P	�J����A�*

	conv_lossG�>�7�        )��P	W�����A�*

	conv_loss�՝>&���        )��P	Q�����A�*

	conv_loss��>뗋�        )��P	1�����A�*

	conv_loss�o�>)oWK        )��P	�1����A�*

	conv_losspP�>X�5        )��P	*k����A�*

	conv_loss(�>���        )��P	������A�*

	conv_loss��>$:i        )��P	�����A�*

	conv_loss]7�>x|"        )��P	�����A�*

	conv_loss��>�&        )��P	�L����A�*

	conv_lossߞ>�Bϥ        )��P	������A�*

	conv_loss˴�>s+��        )��P	������A�*

	conv_lossew�>�pF%        )��P	������A�*

	conv_lossa��>f�%        )��P	�-����A�*

	conv_lossPS�>��        )��P	�a����A�*

	conv_lossNx�>�y8�        )��P	������A�*

	conv_lossH�>ph�        )��P	M�����A�*

	conv_loss-��>AXٵ        )��P	����A�*

	conv_loss�0�>�+        )��P	�7����A�*

	conv_loss�G�>��g0        )��P	:l����A�*

	conv_loss��>���~        )��P	!�����A�*

	conv_loss`j�>�[�        )��P	������A�*

	conv_lossT�>��(�        )��P	������A�*

	conv_loss	۞>`���        )��P	�;����A�*

	conv_loss-�>9�I        )��P	`m����A�*

	conv_loss�>z-6        )��P	������A�*

	conv_loss�n�>d��i        )��P	������A�*

	conv_loss���>��        )��P	:�����A�*

	conv_loss�ğ>JH��        )��P	_=����A�*

	conv_lossZ`�>RPS+        )��P	�n����A�*

	conv_loss�ɝ>��6f        )��P	6�����A�*

	conv_lossDѝ>in        )��P	�����A�*

	conv_loss-��>��a�        )��P	1����A�*

	conv_lossX̝>��        )��P	�H����A�*

	conv_loss寞>$F7        )��P	����A�*

	conv_loss4�>τ��        )��P	'�����A�*

	conv_lossJ֟>[�T�        )��P	�����A�*

	conv_lossá�>�B�        )��P	I����A�*

	conv_loss2�>A@�        )��P	�L����A�*

	conv_lossP�>���        )��P	������A�*

	conv_lossRi�>�ֽ        )��P	�����A�*

	conv_loss"��>g        )��P	������A�*

	conv_loss�a�>5�v7        )��P	�!����A�*

	conv_loss�3�>�+�        )��P	�U����A�*

	conv_loss0��>���F        )��P	������A�*

	conv_loss\)�>�˦r        )��P	�����A�*

	conv_loss'��>+�C�        )��P	����A�*

	conv_loss�Ԣ>�k��        )��P	8����A�*

	conv_loss�y�>�D�        )��P	 s����A�*

	conv_loss{�>���        )��P	�����A�*

	conv_loss�	�>��;        )��P	=�����A�*

	conv_loss��>n�<[        )��P	�����A�*

	conv_loss:�>O�Q�        )��P	�9����A�*

	conv_loss�I�>n�b�        )��P	�{����A�*

	conv_loss�<�>W@Q        )��P	������A�*

	conv_loss���>��s        )��P	������A�*

	conv_loss���>��t        )��P	�����A�*

	conv_loss��>�"=        )��P	�N����A�*

	conv_loss���>�ݑM        )��P	�����A�*

	conv_loss��>if        )��P	/�����A�*

	conv_loss=��>�z�        )��P	�����A�*

	conv_lossn��>�~��        )��P	����A�*

	conv_loss��>�֘        )��P	fL����A�*

	conv_loss>Ş>�Q        )��P	������A�*

	conv_loss4�>Fqh        )��P	������A�*

	conv_loss��>�tȓ        )��P	c�����A�*

	conv_lossD��>�l�        )��P	%6����A�*

	conv_loss�'�>K�	        )��P	zi����A�*

	conv_loss1y�>�B,        )��P	������A�*

	conv_loss2��><��        )��P	�����A�*

	conv_loss]��>Qd�        )��P	[����A�*

	conv_loss���>��.        )��P	�3����A�*

	conv_loss�'�>�D�        )��P	�c����A�*

	conv_loss��>U,D�        )��P	������A�*

	conv_loss��>���	        )��P	�����A�*

	conv_loss�Z�>��օ        )��P	������A�*

	conv_lossy��>װb!        )��P	�.����A�*

	conv_loss���>�5�        )��P	�d����A�*

	conv_loss�;�>���        )��P	�����A�*

	conv_loss�>T�VW        )��P	������A�*

	conv_lossD#�>�EK        )��P	����A�*

	conv_loss[�>,�պ        )��P	�:����A�*

	conv_lossg{�>;�8,        )��P	~����A�*

	conv_loss��>Ǥ��        )��P	�����A�*

	conv_loss��>��m�        )��P	������A�*

	conv_loss�*�> ��        )��P	�����A�*

	conv_loss�;�>r��>        )��P	D����A�*

	conv_loss"��>68m        )��P	�u����A�*

	conv_loss�>r��        )��P	F�����A�*

	conv_loss/��>�+�e        )��P	�����A�*

	conv_loss��>A;�H        )��P	(����A�*

	conv_loss���>��[*        )��P	hT����A�*

	conv_loss���>M���        )��P	������A�*

	conv_loss��>2�O2        )��P	������A�*

	conv_loss���>\��        )��P	S�����A�*

	conv_lossz�>P�        )��P	
4����A�*

	conv_loss�g�>���        )��P	<d����A�*

	conv_loss��>���        )��P	4�����A�*

	conv_loss���>&B��        )��P	������A�*

	conv_loss��>���        )��P	.�����A�*

	conv_loss���>�Bb�        )��P	K.����A�*

	conv_lossW�>���>        )��P	�l����A�*

	conv_lossg�>ux�k        )��P	������A�*

	conv_lossc��>[Y?        )��P	������A�*

	conv_loss���>u��        )��P	u����A�*

	conv_loss
�>�{        )��P	�I����A�*

	conv_loss�؛>�?��        )��P	W|����A�*

	conv_lossGZ�>�m�        )��P	7�����A�*

	conv_loss'�>)<�f        )��P	F�����A�*

	conv_loss�֟>3���        )��P	 ����A�*

	conv_lossʺ�>���        )��P	�D����A�*

	conv_lossQ��>5��        )��P	�s����A�*

	conv_loss2`�>��/9        )��P	������A�*

	conv_lossl�>ZG�i        )��P	������A�*

	conv_loss�b�>�\|�        )��P	5����A�*

	conv_loss��>ګ%        )��P	D����A�*

	conv_loss�Y�>�M{�        )��P	lv����A�*

	conv_loss3�>r�P        )��P	�����A�*

	conv_loss��>�b�        )��P	�����A�*

	conv_loss/Ɵ>GI	+        )��P	�����A�*

	conv_loss���>6�)0        )��P	M9����A�*

	conv_loss��>t�;"        )��P	#i����A�*

	conv_lossc��>���1        )��P	�����A�*

	conv_loss�k�>>�        )��P	�����A�*

	conv_loss���>=���        )��P	G����A�*

	conv_lossXZ�>p�<&        )��P	�8����A�*

	conv_loss��>�$d        )��P	w����A�*

	conv_lossRP�>�ГG        )��P	Ŵ����A�*

	conv_loss|ݞ>�,!n        )��P	������A�*

	conv_loss�E�>�5G        )��P	�����A�*

	conv_loss�c�>�W�        )��P	=Q����A�*

	conv_lossd@�>��o�        )��P	c�����A�*

	conv_loss��>}��J        )��P	-�����A�*

	conv_loss�>��V        )��P	L�����A�*

	conv_losshҠ>4`-V        )��P	!����A�*

	conv_loss�$�>����        )��P	�T����A�*

	conv_loss���>c�{�        )��P	b�����A�*

	conv_loss��>!��        )��P	�����A�*

	conv_loss�>U_NT        )��P	������A�*

	conv_loss=d�>󈿹        )��P	k%����A�*

	conv_loss��>d�Gc        )��P	�X����A�*

	conv_loss�ך>�O[�        )��P	'�����A�*

	conv_lossѢ�>,@�        )��P	������A�*

	conv_loss�s�>(�a        )��P	)�����A�*

	conv_loss&,�>�2�R        )��P	�����A�*

	conv_losszn�>��r�        )��P	cX����A�*

	conv_loss��>        )��P	Ɇ����A�*

	conv_loss���>N{f        )��P	u�����A�*

	conv_lossb�>���T        )��P	������A�*

	conv_loss3Y�>,z��        )��P	�����A�*

	conv_loss�Н>q�        )��P	�L����A�*

	conv_loss&��>�&�        )��P	������A�*

	conv_loss��>Q}5i        )��P	=�����A�*

	conv_lossi�>��        )��P	������A�*

	conv_lossLϝ>M7��        )��P	[2����A�*

	conv_lossWA�>�;�        )��P	c����A�*

	conv_lossU�>�D�        )��P	j�����A�*

	conv_loss6�>�ߜ0        )��P	������A�*

	conv_loss(�>�H�F        )��P	� ����A�*

	conv_loss�8�>;�        )��P	0/����A�*

	conv_lossvW�>��=V        )��P	�^����A�*

	conv_loss��>����        )��P	ʌ����A�*

	conv_loss�j�>b���        )��P	i�����A�*

	conv_lossù�>�*r        )��P	�����A�*

	conv_loss?ț>]f�(        )��P	)0����A�*

	conv_loss�՚>���        )��P	�b����A�*

	conv_lossρ�>�u��        )��P	ܓ����A�*

	conv_loss��>/y��        )��P	������A�*

	conv_lossMS�>�˿        )��P	l�����A�*

	conv_loss�Q�>�IL�        )��P	�3����A�*

	conv_loss�қ>0��        )��P	�d����A�*

	conv_loss��>�T�;        )��P	�����A�*

	conv_loss1Ϟ>���        )��P	&�����A�*

	conv_loss��>L�5h        )��P	k�����A�*

	conv_losszƟ>2ݱD        )��P	�2����A�*

	conv_lossɠ>6M�L        )��P	�d����A�*

	conv_loss�Y�>�ƿ�        )��P	������A�*

	conv_loss���>t���        )��P	b�����A�*

	conv_lossW͚>n���        )��P	������A�*

	conv_loss@9�>(�        )��P	�3����A�*

	conv_lossB3�>5��9        )��P	�g����A�*

	conv_loss�d�>6�K        )��P	������A�*

	conv_lossht�>7i��        )��P	{�����A�*

	conv_loss�ʞ>��        )��P	]�����A�*

	conv_lossn��>b��{        )��P	�=����A�*

	conv_lossmn�>ٮ��        )��P	�m����A�*

	conv_loss��>9y��        )��P	������A�*

	conv_loss���>��        )��P	�����A�*

	conv_loss�z�>+f�;        )��P	-�����A�*

	conv_lossÿ�>�<��        )��P	u,����A�*

	conv_loss7��>L�R        )��P	z^����A�*

	conv_loss@-�>�wn8        )��P	ϐ����A�*

	conv_losss��>$$�7        )��P	j�����A�*

	conv_loss��>�-Y1        )��P	b�����A�*

	conv_lossd�>:�A        )��P	"����A�*

	conv_loss[s�>\~�        )��P	 �����A�*

	conv_loss�>m�c"        )��P	X�����A�*

	conv_loss��>��        )��P	�����A�*

	conv_loss�(�>7�a        )��P	J^����A�*

	conv_lossj��>�wg        )��P	������A�*

	conv_loss��>?H�S        )��P	������A�*

	conv_loss���>��Ϩ        )��P	������A�*

	conv_loss�(�>)8|        )��P	� ����A�*

	conv_loss�?�>�Z�        )��P	tX����A�*

	conv_loss*��>���	        )��P	l�����A�*

	conv_lossꦟ>��c        )��P	�����A�*

	conv_loss��>��q�        )��P	������A�*

	conv_loss��>���        )��P	--����A�*

	conv_lossp?�>���        )��P	�c����A�*

	conv_loss>�k��        )��P	ܘ����A�*

	conv_losss�>bjxk        )��P	������A�*

	conv_loss�b�>�G�        )��P	v�����A�*

	conv_losseǞ>��X<        )��P	�'����A�*

	conv_loss��>�        )��P	�V����A�*

	conv_loss���>����        )��P	x�����A�*

	conv_lossV��>�a��        )��P	������A�*

	conv_loss�ߞ>��u        )��P	������A�*

	conv_loss쌝>�ʭc        )��P	G����A�*

	conv_loss�m�>���'        )��P	XG����A�*

	conv_loss���>�#        )��P	�v����A�*

	conv_loss
�>�d(        )��P	�����A�*

	conv_loss��>��{        )��P	4�����A�*

	conv_lossC�>����        )��P	K
����A�*

	conv_loss�w�>���C        )��P	+<����A�*

	conv_loss���>+r�c        )��P	m����A�*

	conv_loss���>/d��        )��P	<�����A�*

	conv_loss(V�>�ڤN        )��P	D�����A�*

	conv_loss�O�>(c�        )��P	�����A�*

	conv_loss)��>�4�b        )��P	m;����A�*

	conv_loss�(�>3��Q        )��P	�k����A�*

	conv_loss�H�> h#        )��P	ך����A�*

	conv_loss4��>���        )��P	������A�*

	conv_lossʩ�>f���        )��P	s�����A�*

	conv_loss7!�>�{�        )��P	�,����A�*

	conv_loss`�>d�'        )��P	#^����A�*

	conv_loss�>�ž�        )��P	������A�*

	conv_lossU�>v��        )��P	+�����A�*

	conv_lossA�>��        )��P	,�����A�*

	conv_loss�>8-�        )��P	9#����A�*

	conv_loss��>�_`d        )��P	�U����A�*

	conv_loss�@�>%�g�        )��P	"�����A�*

	conv_lossw0�>N��a        )��P	}�����A�*

	conv_loss�h�>8۳�        )��P	������A�*

	conv_loss�ܝ>.#�        )��P	Q&����A�*

	conv_loss�5�>A͸�        )��P	�X����A�*

	conv_lossǠ>��nb        )��P	ܜ����A�*

	conv_loss��>�3��        )��P	������A�*

	conv_lossU�>E�        )��P	�����A�*

	conv_loss��>X��        )��P	�:����A�*

	conv_lossiٙ>�/D�        )��P	k����A�*

	conv_loss�H�>ݭ�        )��P	H�����A�*

	conv_lossH�>;�        )��P	������A�*

	conv_loss�O�>Rx-
        )��P	w�����A�*

	conv_lossCՙ>�uZ@        )��P	�6����A�*

	conv_lossƜ>v�I        )��P	�k����A�*

	conv_loss?�>i��        )��P	������A�*

	conv_lossÞ�>���        )��P	������A�*

	conv_loss.��>B���        )��P	� ����A�*

	conv_loss�K�>+o -        )��P	;8����A�*

	conv_loss�۠>�8N        )��P	s}����A�*

	conv_loss���>
��r        )��P	������A�*

	conv_loss[2�>���        )��P	k�����A�*

	conv_loss��>0���        )��P	s����A�*

	conv_loss�^�>&�jO        )��P	�E����A�*

	conv_loss*О>�R��        )��P	�u����A�*

	conv_loss���>�~�H        )��P	������A�*

	conv_loss�k�>��O�        )��P	V�����A�*

	conv_loss9��>�+�        )��P	� ���A�*

	conv_loss~1�>R�p         )��P	�9 ���A�*

	conv_loss��>��ށ        )��P	�h ���A�*

	conv_loss���>y�fE        )��P	$� ���A�*

	conv_lossB��>P>b        )��P	�� ���A�*

	conv_lossyɢ>��G        )��P	C���A�*

	conv_lossf�>�n        )��P	�B���A�*

	conv_loss��>��        )��P	\q���A�*

	conv_lossT��>���R        )��P	֣���A�*

	conv_lossQ]�>߉�        )��P	}����A�*

	conv_lossT>�>�ڙp        )��P	�
���A�*

	conv_loss���>z?�        )��P	�;���A�*

	conv_lossW��>5v        )��P	yo���A�*

	conv_loss�(�><�}T        )��P	H����A�*

	conv_lossXћ>« j        )��P	�����A�*

	conv_loss���>kE6        )��P	����A�*

	conv_loss�<�>��^�        )��P	�D���A�*

	conv_loss��>�@U�        )��P	�}���A�*

	conv_loss�3�>�Y�        )��P	"����A�*

	conv_loss�_�>:�2        )��P	�����A�*

	conv_loss���>2	>�        )��P	���A�*

	conv_loss%��>y�        )��P	GO���A�*

	conv_loss���>[��        )��P	�|���A�*

	conv_loss���>g%�R        )��P	����A�*

	conv_loss+B�>Qyg�        )��P	�����A�*

	conv_loss�R�>3DQ        )��P	����A�*

	conv_loss��>�Dl�        )��P	�E���A�*

	conv_lossz��>��4        )��P	�x���A�*

	conv_loss(�>n�        )��P	r����A�*

	conv_lossr�>���y        )��P	����A�*

	conv_loss?�>��        )��P	�:���A�*

	conv_loss��>�5�V        )��P	3����A�*

	conv_lossf�>~74        )��P	C:���A�*

	conv_loss�Ğ>U�7�        )��P	�����A�*

	conv_lossͤ�>��g        )��P	5���A�*

	conv_loss�%�>/j$�        )��P	����A�*

	conv_loss���>�7        )��P	 [	���A�*

	conv_loss`��>>Q��        )��P	J�	���A�*

	conv_loss�C�>�b:�        )��P	1
���A�*

	conv_lossX��>�UUJ        )��P	�
���A�*

	conv_loss���>���        )��P	� ���A�*

	conv_lossm�>���        )��P	Ho���A�*

	conv_lossѷ�>|���        )��P	�����A�*

	conv_loss���>�)�        )��P	�)���A�*

	conv_lossB�>xէ�        )��P	����A�*

	conv_loss{c�>�V        )��P	�����A�*

	conv_loss���>}�R        )��P	e/���A�*

	conv_loss�>�Sa        )��P	b����A�*

	conv_loss���>@�L6        )��P	'����A�*

	conv_loss�w�>���        )��P	�N���A�*

	conv_loss^{�>7�;l        )��P	�����A�*

	conv_loss�ҙ>���+        )��P	����A�*

	conv_loss�v�>g~U�        )��P	�U���A�*

	conv_loss�>���C        )��P	ҭ���A�*

	conv_lossn��>�;�s        )��P	����A�*

	conv_loss��>��        )��P	�Y���A�*

	conv_losstA�>&0        )��P	u����A�*

	conv_loss���>�K�'        )��P	Q���A�*

	conv_lossgʛ>>粤        )��P	�V���A�*

	conv_lossΘ>���=        )��P	����A�*

	conv_loss��>�L0        )��P	�����A�*

	conv_lossu�>��d�        )��P	�U���A�*

	conv_loss�+�>��        )��P	ګ���A�*

	conv_loss�y�>r��        )��P	v���A�*

	conv_lossX�>a��        )��P	�W���A�*

	conv_loss�_�>j��        )��P	����A�*

	conv_loss[
�>���w        )��P	����A�*

	conv_loss���>M:d�        )��P	�X���A�*

	conv_loss��>��OR        )��P	�����A�*

	conv_loss<��>���        )��P	)���A�*

	conv_loss�c�>���        )��P	�_���A�*

	conv_loss��>}x"�        )��P	v����A�*

	conv_loss�>�*t        )��P	u���A�*

	conv_loss$�>C�]o        )��P	�i���A�*

	conv_lossˊ�>��M        )��P	����A�*

	conv_loss�!�>�N�t        )��P	+���A�*

	conv_lossY�>����        )��P	Nl���A�*

	conv_loss7e�>�H;        )��P	.����A�*

	conv_loss\�>]Б�        )��P	*P���A�*

	conv_lossS��>)��        )��P	D����A�*

	conv_loss�ʛ>��+H        )��P	�����A�*

	conv_loss��>
���        )��P	� ���A�*

	conv_loss�ț>�k�         )��P	�e���A�*

	conv_loss�f�>�}�        )��P	p����A�*

	conv_loss;��>a        )��P	@���A�*

	conv_loss9X�>8	��        )��P	{e���A�*

	conv_lossQ�>���        )��P	�����A�*

	conv_lossu��>>��        )��P	�����A�*

	conv_loss��>�	n        )��P	-2���A�*

	conv_loss�o�>�N��        )��P	Yt���A�*

	conv_loss~ϝ>$fb�        )��P	R����A�*

	conv_loss�G�>��H�        )��P	k����A�*

	conv_losss��>�@�        )��P	�=���A�*

	conv_loss�]�>ɕ��        )��P	�����A�*

	conv_lossr˞> �        )��P	����A�*

	conv_loss)��>o"^        )��P	����A�*

	conv_lossy�>���        )��P	xb���A�*

	conv_lossͽ�>��0t        )��P	�����A�*

	conv_lossh��>bu��        )��P	����A�*

	conv_loss	j�>K�7         )��P	�,���A�*

	conv_loss"&�>S6��        )��P	�p���A�*

	conv_loss	�>�:�        )��P	�����A�*

	conv_loss��>����        )��P	�����A�*

	conv_loss8��>��|�        )��P	#?���A�*

	conv_loss.7�>_��"        )��P	����A�*

	conv_loss�|�>�|P�        )��P	9����A�*

	conv_lossA�>���        )��P	� ���A�*

	conv_lossv�>C>�        )��P	4] ���A�*

	conv_loss
��>걦        )��P	�� ���A�*

	conv_lossL^�>Oh��        )��P	�� ���A�*

	conv_loss�q�>�!ɗ        )��P	r)!���A�*

	conv_loss��>�?9W        )��P	�o!���A�*

	conv_losss�>n�%        )��P	Թ!���A�*

	conv_loss'�>�        )��P	 "���A�*

	conv_lossA�>�Ck        )��P	�`"���A�*

	conv_loss��>��        )��P	�"���A�*

	conv_loss1۞>��4        )��P	��"���A�*

	conv_lossg6�>����        )��P	�1#���A�*

	conv_loss>=�>H�N-        )��P	�x#���A�*

	conv_loss'�>�X        )��P	�#���A�*

	conv_loss ��>�s�>        )��P	�$���A�*

	conv_loss�M�>�,֐        )��P	�`$���A�*

	conv_loss`s�>�        )��P	ͪ$���A�*

	conv_loss!ԡ>*��T        )��P	��$���A�*

	conv_lossɝ>"D�]        )��P	53%���A�*

	conv_loss���>/�L�        )��P	�t%���A�*

	conv_loss@�>�0�        )��P	E�%���A�*

	conv_loss���>��-        )��P	Zj+���A�*

	conv_loss�	�>v���        )��P	u-���A�*

	conv_lossyx�>%̏�        )��P	�B-���A�*

	conv_loss�^�>�Lw	        )��P	Sp-���A�*

	conv_loss���>"��        )��P	�-���A�*

	conv_lossPU�>�7��        )��P	*�-���A�*

	conv_loss���>K�F�        )��P	�.���A�*

	conv_loss��>az`�        )��P	�@.���A�*

	conv_loss�2�>#�K        )��P	�~.���A�*

	conv_loss�1�>��        )��P	��.���A�*

	conv_loss�[�>��n        )��P	�.���A�*

	conv_loss=��>Q�jf        )��P	k/���A�*

	conv_lossHB�>����        )��P	�V/���A�*

	conv_loss�ޝ>��#u        )��P	V�/���A�*

	conv_loss��>�P��        )��P	��/���A�*

	conv_loss�#�>PG1�        )��P	4�/���A�*

	conv_loss�^�>�8�        )��P	H*0���A�*

	conv_lossh��>�L�c        )��P	vZ0���A�*

	conv_loss 7�>/�)�        )��P	H�0���A�*

	conv_lossf�>"��        )��P	��0���A�*

	conv_loss�}�>?��        )��P	�1���A�*

	conv_loss��>��u�        )��P	j61���A�*

	conv_loss5�>/W��        )��P	�g1���A�*

	conv_loss�J�>����        )��P	��1���A�*

	conv_loss�m�>`�<g        )��P	��1���A�*

	conv_loss�Ü>\�/�        )��P	��1���A�*

	conv_loss�S�>��8        )��P	�%2���A�*

	conv_loss���>�9��        )��P	�T2���A�*

	conv_loss ��>�s�h        )��P	>�2���A�*

	conv_loss^a�>IFN�        )��P	��2���A�*

	conv_loss�~�>oYo�        )��P	}�2���A�*

	conv_lossU��>N��        )��P	�3���A�*

	conv_loss���>�H+�        )��P	D3���A�*

	conv_loss�c�>� h�        )��P	�t3���A�*

	conv_lossѐ�>�0O�        )��P	ƣ3���A�*

	conv_loss��>Yy��        )��P	$�3���A�*

	conv_loss�>[�)t        )��P	4���A�*

	conv_loss#�>a�V        )��P	014���A�*

	conv_loss���>Ɗ�        )��P	�`4���A�*

	conv_lossg
�>�:��        )��P	��4���A�*

	conv_loss��>�ǘ�        )��P	̾4���A�*

	conv_loss�>��%v        )��P	~�4���A�*

	conv_lossm�>��QI        )��P	�5���A�*

	conv_loss��>��t�        )��P	�L5���A�*

	conv_loss#��>��l�        )��P	�~5���A�*

	conv_loss�F�>���        )��P	J�5���A�*

	conv_loss�G�>���         )��P	��5���A�*

	conv_loss|=�> P\        )��P	b6���A�*

	conv_loss���>��2O        )��P	�<6���A�*

	conv_lossbF�>q�g�        )��P	o6���A�*

	conv_loss���>��        )��P		�6���A�*

	conv_loss6c�>�<ky        )��P	U�6���A�*

	conv_lossk��>��*�        )��P	�7���A�*

	conv_loss.��>�!M�        )��P	q=7���A�*

	conv_lossSO�>6#��        )��P	~l7���A�*

	conv_loss|��>*o	�        )��P	w�7���A�*

	conv_loss0�>b�Vb        )��P	[�7���A�*

	conv_loss�?�>OP        )��P	�	8���A�*

	conv_loss/6�>n-+�        )��P	�>8���A�*

	conv_lossJ�>�d�        )��P	){8���A�*

	conv_loss��>fo        )��P	��8���A�*

	conv_losso��>���        )��P	c�8���A�*

	conv_loss�t�>,��"        )��P	�!9���A�*

	conv_loss�>�>���v        )��P	�S9���A�*

	conv_lossY�>o�:�        )��P	~�9���A�*

	conv_loss���>{s        )��P	j�9���A�*

	conv_loss�>��,        )��P	��9���A�*

	conv_loss�ә>W��%        )��P	�:���A�*

	conv_loss���> ]E        )��P	�?:���A�*

	conv_lossX��>���v        )��P	p:���A�*

	conv_loss"�>�T�4        )��P	��:���A�*

	conv_loss�>�O9�        )��P	��:���A�*

	conv_lossY��>t�]�        )��P	X$;���A�*

	conv_lossk��>ikm3        )��P	$V;���A�*

	conv_loss픜>-�u@        )��P	ą;���A�*

	conv_loss4J�>��p�        )��P	a�;���A�*

	conv_loss���>r\��        )��P	��;���A�*

	conv_loss��>����        )��P	�<���A�*

	conv_loss�e�>e�\�        )��P	J<���A�*

	conv_loss�ћ>뱢;        )��P	y<���A�*

	conv_lossd4�>�;��        )��P	!�<���A�*

	conv_loss+�>���        )��P	��<���A�*

	conv_loss*ߝ>-"4'        )��P	�	=���A�*

	conv_loss�v�>�(�z        )��P	09=���A�*

	conv_lossmG�>�ؔ        )��P	i=���A�*

	conv_loss���>����        )��P	��=���A�*

	conv_loss���>n��        )��P	��=���A�*

	conv_loss�-�>�j�        )��P	s�=���A�*

	conv_loss���>P��        )��P	�.>���A�*

	conv_loss{A�>	��e        )��P	]>���A�*

	conv_loss)[�>�[W.        )��P	��>���A�*

	conv_losso��>�h��        )��P	W�>���A�*

	conv_loss	c�>^A�        )��P	@�>���A�*

	conv_loss�1�>l]        )��P	� ?���A�*

	conv_loss'�>�\��        )��P	�Q?���A�*

	conv_loss�E�>���        )��P	&�?���A�*

	conv_loss�4�>��        )��P	��?���A� *

	conv_loss���>����        )��P	��?���A� *

	conv_loss�p�>�V        )��P	y@���A� *

	conv_lossW@�>$��K        )��P	�@@���A� *

	conv_loss]v�>�Ź        )��P	 o@���A� *

	conv_loss��>�sS�        )��P	C�@���A� *

	conv_loss���>��`]        )��P	{�@���A� *

	conv_loss`�>�J��        )��P	�A���A� *

	conv_loss�Z�>;�        )��P	�JA���A� *

	conv_loss��>�vx�        )��P	�A���A� *

	conv_loss"�>6TFy        )��P	��A���A� *

	conv_lossj��>��(L        )��P	��A���A� *

	conv_loss/��>���        )��P	sB���A� *

	conv_loss�Ϝ>��k%        )��P	�NB���A� *

	conv_loss���>Խ.        )��P	͊B���A� *

	conv_loss'��>�|>        )��P	j�B���A� *

	conv_loss�Ŝ>ߴf�        )��P	��B���A� *

	conv_loss�q�>Z�        )��P	C���A� *

	conv_loss|Ú>h-D        )��P	�TC���A� *

	conv_lossI��>�}�        )��P	ʄC���A� *

	conv_loss�w�>s�t�        )��P	Q�C���A� *

	conv_loss:v�>��Ќ        )��P	��C���A� *

	conv_loss;��>|�\2        )��P	$D���A� *

	conv_loss�ї>I��        )��P	�HD���A� *

	conv_loss̖>�}1        )��P	�{D���A� *

	conv_lossu�>��rR        )��P	L�D���A� *

	conv_loss]�>z$�        )��P	�D���A� *

	conv_loss���>%��A        )��P	pE���A� *

	conv_loss֝>^�        )��P	QE���A� *

	conv_loss2�>Ȉ�=        )��P	-�E���A� *

	conv_loss�k�>�$`"        )��P	d�E���A� *

	conv_loss�F�>���        )��P	v�E���A� *

	conv_loss&w�>.�kO        )��P	�F���A� *

	conv_loss��>e��K        )��P	�>F���A� *

	conv_loss��>��.�        )��P	�nF���A� *

	conv_lossЗ�>��H        )��P	D�F���A� *

	conv_loss�ę>Z�        )��P	/�F���A� *

	conv_loss�ܛ>ëye        )��P	��F���A� *

	conv_loss%��>����        )��P	�/G���A� *

	conv_losst��>�(         )��P	8_G���A� *

	conv_loss�H�>6d        )��P	?�G���A� *

	conv_loss-��>�qs        )��P	��G���A� *

	conv_loss*��>���u        )��P	��G���A� *

	conv_loss(��>�xGy        )��P	�'H���A� *

	conv_lossה�>|y�B        )��P	f_H���A� *

	conv_loss�Й>U�d        )��P	��H���A� *

	conv_loss*g�>�`        )��P	x�H���A� *

	conv_loss队>s�}        )��P	��H���A� *

	conv_loss� �>r���        )��P	�#I���A� *

	conv_lossq��><���        )��P	�RI���A� *

	conv_loss�>��E        )��P	ԂI���A� *

	conv_loss�9�>�k�        )��P	��I���A� *

	conv_loss6i�>cv        )��P	2�I���A� *

	conv_lossP�>��T�        )��P	�J���A� *

	conv_loss:�>��$A        )��P	�OJ���A� *

	conv_loss���>�f�        )��P	N�J���A� *

	conv_loss�G�>���        )��P	!�J���A� *

	conv_loss���>*���        )��P	 �J���A� *

	conv_lossM��>i8?�        )��P	6 K���A� *

	conv_losss6�>���H        )��P	%TK���A� *

	conv_loss�+�>��M        )��P	$�K���A� *

	conv_loss�ޜ>��i�        )��P	>�K���A� *

	conv_loss��>��q        )��P	��K���A� *

	conv_loss��>�SQ        )��P	�'L���A� *

	conv_loss�З>u��D        )��P		_L���A� *

	conv_loss���>��Qs        )��P	�L���A� *

	conv_loss���>eI��        )��P	��L���A� *

	conv_lossʪ�>����        )��P	M�L���A� *

	conv_loss���>�db*        )��P	�1M���A� *

	conv_loss���>��˶        )��P	raM���A� *

	conv_loss��>���A        )��P	�M���A� *

	conv_loss��>��~�        )��P	��M���A� *

	conv_loss�(�><�
�        )��P	��M���A� *

	conv_lossJݖ>/��        )��P	�1N���A� *

	conv_loss�ߝ>�1        )��P	FeN���A� *

	conv_loss ��>=s        )��P	Z�N���A� *

	conv_lossة�>`
/�        )��P	��N���A� *

	conv_lossG��>*R��        )��P	O���A� *

	conv_loss@a�>��        )��P	4:O���A� *

	conv_losso�>�܏�        )��P	 mO���A� *

	conv_loss4ę>���        )��P	<�O���A� *

	conv_loss^*�>�W�         )��P	�O���A� *

	conv_loss���>��Y	        )��P	j�O���A� *

	conv_loss���>���0        )��P	.P���A� *

	conv_loss���>���        )��P	^P���A� *

	conv_loss�/�>�64        )��P	��P���A� *

	conv_loss�O�>����        )��P	�P���A� *

	conv_lossi3�>����        )��P	[�P���A� *

	conv_lossc��> љ        )��P	�&Q���A� *

	conv_loss� �>>��        )��P	�UQ���A� *

	conv_loss�-�>Jc�n        )��P	"�Q���A� *

	conv_loss�E�>J���        )��P	��Q���A� *

	conv_lossl>��c�        )��P	u�Q���A� *

	conv_lossV\�>{R�6        )��P	�'R���A� *

	conv_loss�ؘ>n�V        )��P	j\R���A� *

	conv_loss�[�>��<        )��P	��R���A� *

	conv_loss��>����        )��P	��R���A� *

	conv_loss��>�?û        )��P	4�R���A� *

	conv_lossP�>�|�f        )��P	�'S���A� *

	conv_lossh>�>�|�        )��P	xXS���A� *

	conv_loss�7�>�@�        )��P	��S���A� *

	conv_loss���>��^�        )��P	��S���A� *

	conv_loss|��>�V
        )��P	�S���A� *

	conv_loss{�>��/        )��P	�T���A� *

	conv_lossǚ�>��Y�        )��P	~RT���A� *

	conv_loss"ޘ>GűU        )��P	g�U���A� *

	conv_loss	��>��3�        )��P	QV���A� *

	conv_loss�;�>�i�+        )��P	;GV���A� *

	conv_loss���>VA��        )��P	$yV���A� *

	conv_loss]��>�o�W        )��P	1�V���A� *

	conv_loss�>[OAX        )��P	��V���A� *

	conv_loss)��>w���        )��P	W���A� *

	conv_loss�*�>�J�7        )��P	AW���A� *

	conv_loss2Ν>1�        )��P	P{W���A� *

	conv_loss���>�CSv        )��P	��W���A� *

	conv_loss��>��!�        )��P	9�W���A� *

	conv_loss�J�>i��        )��P	�X���A� *

	conv_lossO{�>���        )��P	�FX���A� *

	conv_loss��>2�        )��P	�vX���A� *

	conv_loss��>�|�j        )��P	˥X���A� *

	conv_loss�)�>v��2        )��P	)�X���A� *

	conv_loss�Ŗ>��=        )��P	�Y���A� *

	conv_loss��>�F        )��P	�PY���A� *

	conv_loss2`�>��~        )��P	�Y���A� *

	conv_loss^U�>¿�T        )��P	��Y���A� *

	conv_lossٖ>44_�        )��P	d�Y���A� *

	conv_loss���>Q.�A        )��P	#Z���A� *

	conv_loss�̚>ī��        )��P	�?Z���A� *

	conv_loss=��>�Q�a        )��P	�mZ���A�!*

	conv_loss#U�>(b��        )��P	�Z���A�!*

	conv_loss٨�>S��        )��P	��Z���A�!*

	conv_loss�͝>�W�        )��P	��Z���A�!*

	conv_loss�?�>a��l        )��P	Y+[���A�!*

	conv_loss�0�>���        )��P	�X[���A�!*

	conv_loss���>a�        )��P	��[���A�!*

	conv_loss�h�>��$        )��P	˷[���A�!*

	conv_loss2�>.�Z�        )��P	 �[���A�!*

	conv_loss���>��)        )��P	�\���A�!*

	conv_loss��>�X�        )��P	�F\���A�!*

	conv_loss��>o�Q        )��P	yu\���A�!*

	conv_loss���>5�p�        )��P	2�\���A�!*

	conv_loss"=�>���        )��P	��\���A�!*

	conv_loss�ؚ>�38        )��P	 ]���A�!*

	conv_loss���>�͉-        )��P	O3]���A�!*

	conv_loss�s�>����        )��P	7e]���A�!*

	conv_lossԌ�>�{i        )��P	ɔ]���A�!*

	conv_loss'��>��H�        )��P	��]���A�!*

	conv_loss�e�>Bky\        )��P	�
^���A�!*

	conv_loss�t�>:�        )��P	�N^���A�!*

	conv_loss@ٙ>�O��        )��P	H�^���A�!*

	conv_lossa�>Δ�R        )��P	=�^���A�!*

	conv_loss�[�>�%�T        )��P	��^���A�!*

	conv_lossq �>��-        )��P	�_���A�!*

	conv_lossD;�>?��3        )��P	�R_���A�!*

	conv_lossB#�>��        )��P	�_���A�!*

	conv_loss�љ>I
��        )��P	��_���A�!*

	conv_loss�)�>�x%        )��P	g�_���A�!*

	conv_loss��>U�8�        )��P	� `���A�!*

	conv_loss�U�>��        )��P	�O`���A�!*

	conv_loss1�>h�@        )��P	�`���A�!*

	conv_loss;Ԙ>��        )��P	6�`���A�!*

	conv_loss���>��F�        )��P	)�`���A�!*

	conv_loss�z�>{��        )��P	D!a���A�!*

	conv_loss�>���
        )��P	�]a���A�!*

	conv_loss�>�I        )��P	�a���A�!*

	conv_loss���>�O��        )��P	\�a���A�!*

	conv_lossJʙ>�RZ        )��P	<�a���A�!*

	conv_loss�$�>]v��        )��P	f b���A�!*

	conv_loss&ۘ>m�>        )��P	�Sb���A�!*

	conv_lossX��>���        )��P	؂b���A�!*

	conv_loss���>*�t�        )��P	T�b���A�!*

	conv_losswk�>5�_�        )��P	/�b���A�!*

	conv_loss���>:�        )��P	Mc���A�!*

	conv_loss�ɗ>W�Pj        )��P	CBc���A�!*

	conv_loss��>� �        )��P	Iwc���A�!*

	conv_loss筜>��        )��P	�c���A�!*

	conv_loss`��>�>�]        )��P	��c���A�!*

	conv_loss~ܚ>	�>        )��P	�	d���A�!*

	conv_loss�>�cش        )��P	�8d���A�!*

	conv_loss�"�>o��        )��P	�fd���A�!*

	conv_loss� �>kN�9        )��P	�d���A�!*

	conv_loss�^�>,�Hz        )��P	��d���A�!*

	conv_loss��>�_        )��P	��d���A�!*

	conv_loss�>(&��        )��P	�$e���A�!*

	conv_loss�c�>ȗN@        )��P	�Se���A�!*

	conv_loss��>�vS�        )��P	Æe���A�!*

	conv_loss��>%�?�        )��P	��e���A�!*

	conv_loss���>���        )��P	��e���A�!*

	conv_loss���>����        )��P	vf���A�!*

	conv_loss���>mt�K        )��P	�Df���A�!*

	conv_loss�+�>�        )��P		vf���A�!*

	conv_loss�؛>
fA         )��P	�f���A�!*

	conv_loss ��>˙^        )��P	y�f���A�!*

	conv_loss�p�>�k�5        )��P	�g���A�!*

	conv_losseۘ>_ͫ        )��P	M0g���A�!*

	conv_loss�e�>eV��        )��P	bg���A�!*

	conv_loss��>v��4        )��P	.�g���A�!*

	conv_lossڛ>��+X        )��P	��g���A�!*

	conv_loss
�>�j        )��P	Z�g���A�!*

	conv_lossf�>�g�	        )��P	Th���A�!*

	conv_loss7#�>�ǲ�        )��P	.Nh���A�!*

	conv_loss{a�>���        )��P	�}h���A�!*

	conv_lossZ�>�4��        )��P	?�h���A�!*

	conv_lossN��>߼�U        )��P	x�h���A�!*

	conv_lossK;�>g��        )��P	�i���A�!*

	conv_lossk�>]�p5        )��P	jVi���A�!*

	conv_loss���>���        )��P	5�i���A�!*

	conv_loss>�>��Z        )��P	��i���A�!*

	conv_loss6>ѠV�        )��P	W�i���A�!*

	conv_loss��>��Kq        )��P	�j���A�!*

	conv_loss�{�>j/�        )��P	$Gj���A�!*

	conv_loss�Ԙ>u�        )��P	Uvj���A�!*

	conv_loss �>�$/]        )��P	u�j���A�!*

	conv_loss�F�>Os{C        )��P	��j���A�!*

	conv_lossȏ�><�n�        )��P	w k���A�!*

	conv_lossD^�>��l        )��P	~Rk���A�!*

	conv_lossL"�>�9ʭ        )��P	��k���A�!*

	conv_loss��>PZ��        )��P	f�k���A�!*

	conv_lossJ��>��u        )��P	��k���A�!*

	conv_loss0ڕ>,l3�        )��P	�"l���A�!*

	conv_loss?�>�        )��P	aRl���A�!*

	conv_loss���>#���        )��P	��l���A�!*

	conv_loss�i�>Ð        )��P	�l���A�!*

	conv_loss���>���        )��P	}�l���A�!*

	conv_loss��>E�O        )��P	wm���A�!*

	conv_loss��>z��        )��P	Fm���A�!*

	conv_loss /�>w��G        )��P	�zm���A�!*

	conv_lossL��>�e�        )��P	c�m���A�!*

	conv_loss���>� �g        )��P	��m���A�!*

	conv_loss���>\dx�        )��P	+n���A�!*

	conv_loss˛>�N"         )��P	�Cn���A�!*

	conv_lossKV�>_�f�        )��P	�sn���A�!*

	conv_loss&ݗ>pr0I        )��P	9�n���A�!*

	conv_loss3r�>��        )��P	��n���A�!*

	conv_loss�3�>�D �        )��P	}o���A�!*

	conv_losseV�>�9��        )��P	"8o���A�!*

	conv_loss��>�V�        )��P	$ho���A�!*

	conv_lossg��>x�Gq        )��P	��o���A�!*

	conv_loss��>��H        )��P	�o���A�!*

	conv_loss�2�>�\'        )��P	�p���A�!*

	conv_lossE�>|%�        )��P	�;p���A�!*

	conv_loss���>���>        )��P	�sp���A�!*

	conv_loss/��>�=b�        )��P	Y�p���A�!*

	conv_loss?.�>1}        )��P	5�p���A�!*

	conv_losso>�>n��        )��P	 q���A�!*

	conv_loss_f�>���        )��P	�9q���A�!*

	conv_loss_�>h�
�        )��P	Ghq���A�!*

	conv_loss
��>E3        )��P	�q���A�!*

	conv_loss8�>@��L        )��P	��q���A�!*

	conv_loss���>K�        )��P	��q���A�!*

	conv_loss�>Q��        )��P	z*r���A�!*

	conv_loss]��>��17        )��P	�]r���A�!*

	conv_lossJߙ>���        )��P	��r���A�!*

	conv_loss�h�>���W        )��P	?�r���A�!*

	conv_loss9�>����        )��P	G�r���A�!*

	conv_loss�
�>�l�        )��P	�?s���A�!*

	conv_loss/Ŕ>U�+�        )��P	�ns���A�"*

	conv_lossNu�>�>t�        )��P	��s���A�"*

	conv_loss�ڗ>��g�        )��P	��s���A�"*

	conv_loss��>2t��        )��P	rt���A�"*

	conv_lossd�>�G�        )��P	G3t���A�"*

	conv_loss���>)j��        )��P	yct���A�"*

	conv_loss,f�>��'        )��P	D�t���A�"*

	conv_loss�J�>�Q��        )��P	�t���A�"*

	conv_loss�G�>��Y        )��P	�u���A�"*

	conv_loss���>J #k        )��P	�Iu���A�"*

	conv_lossə�>�ߩ�        )��P	"|u���A�"*

	conv_loss�;�>ۂ=G        )��P	T�u���A�"*

	conv_loss��><l        )��P	��u���A�"*

	conv_loss¾�>y�        )��P	cv���A�"*

	conv_loss8��>-g        )��P	JFv���A�"*

	conv_loss<�>z^��        )��P	 wv���A�"*

	conv_loss&��>�+�        )��P	p�v���A�"*

	conv_loss�F�>�c        )��P	G�v���A�"*

	conv_loss��>n(z        )��P	�w���A�"*

	conv_losss��>9�!�        )��P	/<w���A�"*

	conv_losswؙ>��-        )��P	�nw���A�"*

	conv_loss��>=ѽ�        )��P	��w���A�"*

	conv_loss�`�>�Bo�        )��P	��w���A�"*

	conv_loss_��>y��[        )��P	�x���A�"*

	conv_loss�֚>37�        )��P	7x���A�"*

	conv_loss_F�>�j��        )��P	Sfx���A�"*

	conv_loss��>�q��        )��P		�x���A�"*

	conv_loss�3�>fl&�        )��P	=�x���A�"*

	conv_loss���>��&�        )��P	��x���A�"*

	conv_loss��>��O        )��P	�(y���A�"*

	conv_loss���>���        )��P	1Yy���A�"*

	conv_loss ��>�i@S        )��P	H�y���A�"*

	conv_loss�?�>�ʃ�        )��P	��y���A�"*

	conv_loss�ʙ>�*��        )��P	�y���A�"*

	conv_loss�)�>��<�        )��P	�z���A�"*

	conv_loss��>ר3        )��P	�Lz���A�"*

	conv_loss_?�>�(�        )��P	I}z���A�"*

	conv_lossu��>6�        )��P	4�z���A�"*

	conv_loss��>@z�        )��P	��z���A�"*

	conv_loss]��>��        )��P	]{���A�"*

	conv_loss��>H��|        )��P	iD{���A�"*

	conv_lossuo�>���        )��P	�s{���A�"*

	conv_loss���>��)�        )��P	Y�{���A�"*

	conv_loss��>51��        )��P	�{���A�"*

	conv_loss&B�>��4V        )��P	N|���A�"*

	conv_loss�t�>��W        )��P	�3|���A�"*

	conv_lossș>��m�        )��P	�f|���A�"*

	conv_loss���>��h        )��P	u�|���A�"*

	conv_lossJ`�>��O        )��P	[�|���A�"*

	conv_losso`�>8�s        )��P	_\~���A�"*

	conv_loss
=�>��!D        )��P	�~���A�"*

	conv_loss)Й>���o        )��P	��~���A�"*

	conv_lossNS�>�/<�        )��P	)�~���A�"*

	conv_loss�2�>*&�        )��P	�)���A�"*

	conv_loss�]�>8��        )��P	�[���A�"*

	conv_lossgٖ>OяE        )��P	
����A�"*

	conv_loss��>9��        )��P	<����A�"*

	conv_loss6��>���        )��P	�����A�"*

	conv_lossYp�>Pؤ        )��P	�4����A�"*

	conv_loss�U�>��:        )��P	?h����A�"*

	conv_loss�>�>L��        )��P	������A�"*

	conv_losse]�>�gD        )��P	�ɀ���A�"*

	conv_loss@��>8�z        )��P	������A�"*

	conv_loss�ɘ>!�T�        )��P	22����A�"*

	conv_loss�g�>iu�        )��P	pb����A�"*

	conv_loss�:�>y� �        )��P	�����A�"*

	conv_lossH�>�f��        )��P	�΁���A�"*

	conv_loss���>J��&        )��P	*����A�"*

	conv_loss�ĕ>��׀        )��P	VF����A�"*

	conv_loss�(�>�8��        )��P	\y����A�"*

	conv_loss z�>�f        )��P	a�����A�"*

	conv_loss@֛>|�:8        )��P	�܂���A�"*

	conv_loss�S�>���Q        )��P	�����A�"*

	conv_loss�ӗ>|��        )��P	�C����A�"*

	conv_lossRH�>�ܱ        )��P	cw����A�"*

	conv_loss9X�>�ݳ�        )��P	騃���A�"*

	conv_loss�̘>#�9�        )��P	:ك���A�"*

	conv_loss"͕>!�@%        )��P	�����A�"*

	conv_loss�I�>tC�        )��P	.P����A�"*

	conv_loss�ݖ>�ěK        )��P	e�����A�"*

	conv_loss��>�7�        )��P	�����A�"*

	conv_loss�{�>���n        )��P	�����A�"*

	conv_loss��>�#��        )��P	����A�"*

	conv_loss��>��        )��P	AK����A�"*

	conv_loss�>�p��        )��P	�}����A�"*

	conv_lossj��>�
�P        )��P	c�����A�"*

	conv_loss`�>I�        )��P	������A�"*

	conv_loss�)�>�e         )��P	�����A�"*

	conv_losslS�>^U        )��P	�?����A�"*

	conv_loss r�>�O        )��P	�p����A�"*

	conv_lossXϗ>��        )��P	֠����A�"*

	conv_loss�ȓ>�;7\        )��P	lӆ���A�"*

	conv_losspY�>ur��        )��P	�����A�"*

	conv_loss Ӓ>if��        )��P	�4����A�"*

	conv_loss|��>��        )��P	�q����A�"*

	conv_lossH�>H��        )��P	@�����A�"*

	conv_lossE&�>M @'        )��P	8އ���A�"*

	conv_lossޘ>)��        )��P	`����A�"*

	conv_lossm�>6�,�        )��P	pC����A�"*

	conv_loss�d�>�?�/        )��P	/�����A�"*

	conv_lossճ�>v>@C        )��P	[�����A�"*

	conv_lossI7�>���        )��P	������A�"*

	conv_loss��>j�pL        )��P	�+����A�"*

	conv_loss2��>���        )��P	q����A�"*

	conv_losst��>C�:[        )��P	H�����A�"*

	conv_loss"��>1���        )��P	�։���A�"*

	conv_loss��>�E��        )��P	����A�"*

	conv_loss}і>��:        )��P	�9����A�"*

	conv_loss,c�>��l`        )��P	dx����A�"*

	conv_loss�f�>�$        )��P	u�����A�"*

	conv_losss �>��:        )��P	U؊���A�"*

	conv_loss���>>��        )��P	�����A�"*

	conv_loss1)�>R���        )��P	1:����A�"*

	conv_loss�z�>e�>        )��P	i����A�"*

	conv_loss��>!��p        )��P	������A�"*

	conv_loss�T�>49,�        )��P	M؋���A�"*

	conv_loss*��>�v�        )��P	�����A�"*

	conv_loss��>)F         )��P	kB����A�"*

	conv_loss;e�>��}G        )��P	tu����A�"*

	conv_loss�V�>~e�        )��P	I�����A�"*

	conv_loss�M�>B^U�        )��P	�֌���A�"*

	conv_loss��>��[�        )��P	�����A�"*

	conv_loss t�>�ۭ        )��P	�6����A�"*

	conv_loss���> ���        )��P	j����A�"*

	conv_loss)<�>��A        )��P	������A�"*

	conv_loss�ɕ><\C�        )��P	�ʍ���A�"*

	conv_loss���>'��        )��P	t�����A�"*

	conv_loss���>cE@        )��P	�)����A�"*

	conv_loss��>3�uJ        )��P	;[����A�#*

	conv_loss���>��r�        )��P	������A�#*

	conv_loss��>/r�)        )��P	������A�#*

	conv_loss��>���        )��P	R�����A�#*

	conv_loss�j�>vM��        )��P	"����A�#*

	conv_loss3И>P�        )��P	�R����A�#*

	conv_loss!��>���        )��P	4�����A�#*

	conv_loss0�>�}p        )��P	㳏���A�#*

	conv_losse�>j�c�        )��P	x����A�#*

	conv_loss-�>�\M�        )��P	�����A�#*

	conv_loss��>ђk�        )��P	bH����A�#*

	conv_loss��>b�        )��P	�y����A�#*

	conv_lossL՗>k#�        )��P	�����A�#*

	conv_lossI�>5���        )��P	�ې���A�#*

	conv_loss�U�>@�e        )��P	����A�#*

	conv_loss+��>���        )��P	<����A�#*

	conv_lossw�>5�7        )��P	�m����A�#*

	conv_loss��>�n�        )��P	������A�#*

	conv_loss��>�<�        )��P	�Б���A�#*

	conv_loss'ސ>��l�        )��P	� ����A�#*

	conv_loss���>	�        )��P	�����A�#*

	conv_loss�]�>�c��        )��P	.ܖ���A�#*

	conv_loss�k�>)��        )��P	�����A�#*

	conv_lossY�>� �=        )��P	�:����A�#*

	conv_loss�#�>T���        )��P	�l����A�#*

	conv_loss��>��>        )��P	Y�����A�#*

	conv_loss��>�8:�        )��P	�ٗ���A�#*

	conv_loss��>��        )��P		����A�#*

	conv_loss�X�>R^��        )��P	EB����A�#*

	conv_loss���>����        )��P	�|����A�#*

	conv_loss#d�>��U        )��P	������A�#*

	conv_loss���>���        )��P	�����A�#*

	conv_loss�>��h�        )��P	!����A�#*

	conv_loss}�>]4�        )��P	�O����A�#*

	conv_loss�Ē>)�Z        )��P	�~����A�#*

	conv_loss',�>�G��        )��P	������A�#*

	conv_loss��>�G1	        )��P	�ݙ���A�#*

	conv_loss��>��!,        )��P	�����A�#*

	conv_loss�ϕ>T���        )��P	�<����A�#*

	conv_lossw^�>9"        )��P	{m����A�#*

	conv_loss��>��~        )��P	������A�#*

	conv_lossn>�}�~        )��P	�ؚ���A�#*

	conv_loss	�>mFL        )��P	}����A�#*

	conv_loss]�>��5|        )��P	�A����A�#*

	conv_loss9�>X�F�        )��P	p����A�#*

	conv_lossC�>&N�        )��P	N�����A�#*

	conv_loss7�>�� ~        )��P	�Л���A�#*

	conv_lossȇ>!�"�        )��P	(
����A�#*

	conv_lossq�>��        )��P	�:����A�#*

	conv_loss�H�>�ݗ�        )��P	�j����A�#*

	conv_loss���>e$��        )��P	������A�#*

	conv_loss(��>�ޯ�        )��P	�М���A�#*

	conv_loss���>�ZQ�        )��P	� ����A�#*

	conv_loss�>��}�        )��P	�2����A�#*

	conv_lossaڎ>+"-/        )��P	yi����A�#*

	conv_loss�C�>��0�        )��P	�����A�#*

	conv_loss�>�i        )��P	Hԝ���A�#*

	conv_lossfN�>��a        )��P	�����A�#*

	conv_loss�ܖ>��;�        )��P	�1����A�#*

	conv_loss"_�>�%�        )��P	(`����A�#*

	conv_lossRؔ>33T#        )��P	������A�#*

	conv_lossFE�>���        )��P	x�����A�#*

	conv_loss*S�>�@H        )��P	�����A�#*

	conv_loss)��>!!��        )��P	K����A�#*

	conv_lossXۖ>�±�        )��P	�I����A�#*

	conv_loss^R�>Hv��        )��P	k|����A�#*

	conv_loss��>��b        )��P	]�����A�#*

	conv_loss&�>���\        )��P	Pߟ���A�#*

	conv_loss�G�>bVPd        )��P	w����A�#*

	conv_loss�8�>-�@4        )��P	@B����A�#*

	conv_loss[��>���        )��P	�t����A�#*

	conv_loss��>��.�        )��P	p�����A�#*

	conv_loss^�>
���        )��P	\����A�#*

	conv_loss�O�>8b�        )��P	����A�#*

	conv_loss�Y�>���        )��P	�J����A�#*

	conv_loss�*�>)AE        )��P	�{����A�#*

	conv_loss���>ą�        )��P	v�����A�#*

	conv_loss���>�h��        )��P	Iݡ���A�#*

	conv_loss7�>'��]        )��P	�����A�#*

	conv_loss���>�s��        )��P	�V����A�#*

	conv_loss|�>|H�D        )��P	R�����A�#*

	conv_loss���>��#        )��P	dâ���A�#*

	conv_loss���>8�չ        )��P	�����A�#*

	conv_loss"#�>c&��        )��P	41����A�#*

	conv_loss��>��_        )��P	�c����A�#*

	conv_loss(��>�w�6        )��P	U�����A�#*

	conv_loss�1�>J:sM        )��P	�ţ���A�#*

	conv_loss��>�)��        )��P	������A�#*

	conv_loss_�>�"�        )��P	�/����A�#*

	conv_loss�>���        )��P	T_����A�#*

	conv_losse:�>�DA        )��P	/�����A�#*

	conv_lossD?�>3�]        )��P	Ĥ���A�#*

	conv_lossJ,�>�=Am        )��P	0�����A�#*

	conv_loss�֔>\[OO        )��P	�<����A�#*

	conv_loss?�> �P=        )��P	q����A�#*

	conv_loss���>���T        )��P	������A�#*

	conv_lossRy�>��~t        )��P	IХ���A�#*

	conv_lossMi�>��        )��P	� ����A�#*

	conv_loss��>w���        )��P	�;����A�#*

	conv_lossݠ�>����        )��P	j����A�#*

	conv_loss��>�ok'        )��P	E�����A�#*

	conv_loss�ה>�/R�        )��P	/̦���A�#*

	conv_loss�q�>��{        )��P	[�����A�#*

	conv_loss���>���        )��P	-����A�#*

	conv_loss���>�6��        )��P	wd����A�#*

	conv_loss�>=9�        )��P	ß����A�#*

	conv_loss�> �Uf        )��P	4ק���A�#*

	conv_loss6j�>ɥ�        )��P	n	����A�#*

	conv_loss�^�>�pp        )��P	dD����A�#*

	conv_loss	J�>%N�4        )��P	�u����A�#*

	conv_loss{�>��;'        )��P	������A�#*

	conv_loss�W�>k��9        )��P	�ը���A�#*

	conv_loss7]�>i�.�        )��P	u����A�#*

	conv_lossO@�>'!�Z        )��P	�6����A�#*

	conv_loss��>��w        )��P	-i����A�#*

	conv_loss�9�>����        )��P	������A�#*

	conv_loss�ȗ>Nq�P        )��P	�ɩ���A�#*

	conv_losse��>�^�7        )��P	������A�#*

	conv_loss$��>w�:%        )��P	Q?����A�#*

	conv_lossI��>����        )��P	7o����A�#*

	conv_loss��>����        )��P	Π����A�#*

	conv_loss�t�>��/�        )��P	�N����A�#*

	conv_loss1X�>��z�        )��P	Ƅ����A�#*

	conv_loss��>d!!        )��P	=�����A�#*

	conv_loss*�>��
        )��P	�����A�#*

	conv_loss�I�>gs�C        )��P	�#����A�#*

	conv_loss��>��*�        )��P	F^����A�#*

	conv_lossu>�>3�#        )��P	����A�#*

	conv_loss��>�S�>        )��P	������A�$*

	conv_loss�a�>g���        )��P	p����A�$*

	conv_lossk-�>�>A�        )��P	�"����A�$*

	conv_loss��>��x        )��P	�`����A�$*

	conv_loss�{�>u)\        )��P	{�����A�$*

	conv_lossfI�>�^u�        )��P	�����A�$*

	conv_loss# �>|s�        )��P	�����A�$*

	conv_lossR>�>��s        )��P	�����A�$*

	conv_loss3�>�5(        )��P	vX����A�$*

	conv_loss@q�>u�|�        )��P	҉����A�$*

	conv_lossc�>�֡�        )��P	z�����A�$*

	conv_loss���>
!'�        )��P	�����A�$*

	conv_loss���>�EH�        )��P	�����A�$*

	conv_loss�~�>��45        )��P	�H����A�$*

	conv_loss�]�>"�E        )��P	bv����A�$*

	conv_loss轖>�5        )��P	l�����A�$*

	conv_lossm��>6D�        )��P	�ְ���A�$*

	conv_loss��>�]Б        )��P	�����A�$*

	conv_lossc��>[��        )��P	6����A�$*

	conv_loss%�>�OM        )��P	@d����A�$*

	conv_loss�@�>Y�Ki        )��P	{�����A�$*

	conv_loss-�>�4�        )��P	h�����A�$*

	conv_loss���>$�1        )��P	�����A�$*

	conv_loss/h�>�]�        )��P	�"����A�$*

	conv_loss���>ow��        )��P	�Q����A�$*

	conv_lossJ�>�A,        )��P	������A�$*

	conv_loss�S�>�6        )��P	5�����A�$*

	conv_loss��>d�>N        )��P	����A�$*

	conv_loss�l�>5;�        )��P	�����A�$*

	conv_lossFm�>�~8�        )��P	D����A�$*

	conv_lossx|�>�O�        )��P	Ar����A�$*

	conv_loss�\�>��$�        )��P	������A�$*

	conv_loss��>͓��        )��P	^ϳ���A�$*

	conv_loss-��>I��        )��P	����A�$*

	conv_loss���>!��X        )��P	�8����A�$*

	conv_loss��>�3�        )��P	�r����A�$*

	conv_loss���>����        )��P	c�����A�$*

	conv_lossŐ>���        )��P	�մ���A�$*

	conv_losssJ�>�1        )��P	����A�$*

	conv_loss��>�W         )��P	�7����A�$*

	conv_loss�ϕ>/�.        )��P	Nf����A�$*

	conv_lossו�>SO��        )��P	T�����A�$*

	conv_lossP\�>�t��        )��P	bõ���A�$*

	conv_lossuV�>�]�S        )��P	^����A�$*

	conv_loss��>��        )��P	�8����A�$*

	conv_loss��>fi+�        )��P	ih����A�$*

	conv_loss/V�>:�iQ        )��P	������A�$*

	conv_loss7�>��^        )��P	�̶���A�$*

	conv_loss��>� �        )��P	{�����A�$*

	conv_loss���>��^        )��P	�/����A�$*

	conv_lossk�>�|��        )��P	G_����A�$*

	conv_lossf�>~�yO        )��P	������A�$*

	conv_losssѓ>��X%        )��P	�Է���A�$*

	conv_loss��>��v�        )��P	h����A�$*

	conv_lossv��>�Ų�        )��P	�4����A�$*

	conv_lossƃ�>M���        )��P	h����A�$*

	conv_lossPN�>�d�        )��P	*�����A�$*

	conv_loss�A�>�	\�        )��P	�Ѹ���A�$*

	conv_loss���>\'q        )��P	z�����A�$*

	conv_lossU�>�B'        )��P	3����A�$*

	conv_lossVL�>s�*�        )��P	Wf����A�$*

	conv_lossW��>��i�        )��P	������A�$*

	conv_loss�>�FB        )��P	�ɹ���A�$*

	conv_loss�7�>�JD�        )��P	������A�$*

	conv_loss Е> C-        )��P	�*����A�$*

	conv_loss�>�a�E        )��P	�\����A�$*

	conv_lossИ�>���W        )��P	0�����A�$*

	conv_losscx�>J@��        )��P	F�����A�$*

	conv_loss}<�>�#�O        )��P	�����A�$*

	conv_losst��>B#�        )��P	 ����A�$*

	conv_loss%�>���        )��P	6P����A�$*

	conv_loss�ʒ>nF,        )��P	�����A�$*

	conv_loss��>P|,        )��P	װ����A�$*

	conv_loss�D�>��`        )��P	����A�$*

	conv_loss�M�>j��        )��P	i����A�$*

	conv_lossӢ�>�\`�        )��P	YD����A�$*

	conv_loss�s�>cg�7        )��P	�t����A�$*

	conv_loss�)�>��%        )��P	r�����A�$*

	conv_lossmN�>���        )��P	�Ҽ���A�$*

	conv_lossQ�>��g~        )��P	����A�$*

	conv_lossG#�>��oA        )��P	�3����A�$*

	conv_loss�H�>���        )��P	*b����A�$*

	conv_loss!��>yCf�        )��P	������A�$*

	conv_loss΢�>��)        )��P	}�����A�$*

	conv_lossM�>�|�d        )��P	�����A�$*

	conv_loss�a�>�H�        )��P	�����A�$*

	conv_lossB�>�iZI        )��P	�I����A�$*

	conv_loss�T�>-�d�        )��P	�w����A�$*

	conv_lossFL�>�.�        )��P	������A�$*

	conv_lossog�>��U�        )��P	�ؾ���A�$*

	conv_lossi�>�9�        )��P	8	����A�$*

	conv_loss�g�>��        )��P	�7����A�$*

	conv_lossG�>��:        )��P	Xf����A�$*

	conv_loss�C�>�}s�        )��P	{�����A�$*

	conv_lossH��>��>        )��P	�Կ���A�$*

	conv_lossK�>����        )��P	r����A�$*

	conv_loss���>Q~�        )��P	�0����A�$*

	conv_loss�ݏ>���        )��P	*d����A�$*

	conv_loss��>�9e�        )��P	������A�$*

	conv_loss���>"vM        )��P	������A�$*

	conv_losszԔ>A�QN        )��P	�����A�$*

	conv_loss^��>5�U        )��P	C����A�$*

	conv_loss�|�>B �[        )��P	ox����A�$*

	conv_lossI�>1��P        )��P	,�����A�$*

	conv_loss�#�>u�@J        )��P	������A�$*

	conv_loss�̐>�0��        )��P	�����A�$*

	conv_loss��>�ɿ8        )��P	\<����A�$*

	conv_lossّ�>0x�        )��P	[o����A�$*

	conv_loss"l�>�B�_        )��P	j�����A�$*

	conv_lossXX�>}�:0        )��P	������A�$*

	conv_loss�>�s)m        )��P	q�����A�$*

	conv_lossː>!MU        )��P	1����A�$*

	conv_lossц�>>_F8        )��P	�g����A�$*

	conv_loss��>+%�_        )��P	!�����A�$*

	conv_losssb�>�bW        )��P	������A�$*

	conv_loss�V�>9�h        )��P	H�����A�$*

	conv_losse�>cp%�        )��P	�:����A�$*

	conv_loss��>7        )��P	 m����A�$*

	conv_loss|��>�$        )��P	������A�$*

	conv_loss�!�>��#        )��P	J�����A�$*

	conv_lossO�>���T        )��P	�
����A�$*

	conv_loss���>3y)T        )��P	F;����A�$*

	conv_loss�d�>��-�        )��P	{i����A�$*

	conv_lossϭ�>Y�        )��P	�����A�$*

	conv_lossӨ�>T(=        )��P	�����A�$*

	conv_loss�p�>n;�L        )��P	������A�$*

	conv_lossw��>�*F        )��P	60����A�$*

	conv_losso��>��|        )��P	:c����A�$*

	conv_lossh&�>��PN        )��P	`�����A�%*

	conv_loss0͒>9%[        )��P	7�����A�%*

	conv_loss�7�>R��        )��P	g�����A�%*

	conv_loss� �>Tʡ�        )��P	^$����A�%*

	conv_loss1��>C�i        )��P	�R����A�%*

	conv_lossuݓ>]�@        )��P	+�����A�%*

	conv_loss|�>����        )��P	������A�%*

	conv_loss�>�>9��        )��P	������A�%*

	conv_lossX[�>_{�        )��P	�����A�%*

	conv_loss���>�.        )��P	C?����A�%*

	conv_loss���>�4��        )��P	n����A�%*

	conv_lossv�>F�
        )��P	t�����A�%*

	conv_loss��>u�)        )��P	������A�%*

	conv_loss���>#�7/        )��P	������A�%*

	conv_loss�Y�>t��        )��P	v*����A�%*

	conv_lossΔ>�|4]        )��P	�j����A�%*

	conv_loss���>�1�!        )��P	N�����A�%*

	conv_lossc�>���N        )��P	������A�%*

	conv_loss�>���        )��P	������A�%*

	conv_lossz1�>>�        )��P	},����A�%*

	conv_loss�>E��C        )��P	�[����A�%*

	conv_lossoi�>׃lR        )��P	v�����A�%*

	conv_loss݋>�wQ�        )��P	�����A�%*

	conv_lossWݖ>%���        )��P		�����A�%*

	conv_loss].�>� }�        )��P	�,����A�%*

	conv_loss�Z�>�[��        )��P	b_����A�%*

	conv_loss8��>@Wh�        )��P	�����A�%*

	conv_losspG�>��؆        )��P	�����A�%*

	conv_loss���> �_�        )��P	������A�%*

	conv_loss	�>֬Y        )��P	�����A�%*

	conv_loss-̓>�&��        )��P	*M����A�%*

	conv_loss���>���        )��P	�{����A�%*

	conv_loss[�> ��c        )��P	l�����A�%*

	conv_loss�ܒ>��        )��P	q�����A�%*

	conv_loss�E�>�M�        )��P	l����A�%*

	conv_loss���>Õ�        )��P	�:����A�%*

	conv_loss��>�H6�        )��P	�m����A�%*

	conv_loss
�>~";        )��P	?�����A�%*

	conv_loss���>$        )��P	G�����A�%*

	conv_loss��>J�fI        )��P	X����A�%*

	conv_loss�~�>S�W�        )��P	.4����A�%*

	conv_loss��>fW�&        )��P	Mb����A�%*

	conv_loss���>;9�T        )��P	�����A�%*

	conv_loss��>�{S�        )��P	������A�%*

	conv_loss��>x0 �        )��P	������A�%*

	conv_loss�͖>j&�        )��P	 ����A�%*

	conv_loss� �>%�	�        )��P	lR����A�%*

	conv_lossr��>(�,�        )��P	�����A�%*

	conv_loss��>�7$�        )��P	L�����A�%*

	conv_loss��>�|ו        )��P	p�����A�%*

	conv_loss ��>��z�        )��P	�����A�%*

	conv_loss7��>~KS        )��P	P����A�%*

	conv_loss�&�>^�        )��P	G�����A�%*

	conv_loss�>���o        )��P	������A�%*

	conv_loss�>����        )��P	������A�%*

	conv_loss��>��Dk        )��P	����A�%*

	conv_loss��>�D�y        )��P	qS����A�%*

	conv_loss4!�>=�Q�        )��P	Ӂ����A�%*

	conv_loss�m�>�k޷        )��P	������A�%*

	conv_lossn �>WX�        )��P	i�����A�%*

	conv_losslV�>E��s        )��P	\����A�%*

	conv_loss=��>|�        )��P	�?����A�%*

	conv_loss֔>�'^L        )��P	�u����A�%*

	conv_loss�>ǧ�        )��P	6�����A�%*

	conv_loss|V�>�݃$        )��P	e�����A�%*

	conv_loss���>��(y        )��P	,r����A�%*

	conv_loss�>��:o        )��P	������A�%*

	conv_loss���>_A        )��P	�����A�%*

	conv_loss.�>��        )��P	
����A�%*

	conv_loss�,�>�Sp�        )��P	\:����A�%*

	conv_loss�t�>h�N        )��P	Rk����A�%*

	conv_loss���>ވ�        )��P	ș����A�%*

	conv_loss�A�>(>?Z        )��P	G�����A�%*

	conv_loss*b�>�k�z        )��P	z�����A�%*

	conv_losse�>R�(�        )��P	V-����A�%*

	conv_loss��>�j�&        )��P	�i����A�%*

	conv_loss�M�>��9�        )��P	�����A�%*

	conv_loss��>���        )��P	B�����A�%*

	conv_loss�z�>4���        )��P	~�����A�%*

	conv_loss�%�>� ��        )��P	#2����A�%*

	conv_loss�`�>�Hΐ        )��P	lb����A�%*

	conv_loss�Ē>J��        )��P	�����A�%*

	conv_loss6>��e        )��P	2�����A�%*

	conv_losskʒ>U�0w        )��P	�����A�%*

	conv_loss��>;n�        )��P	�.����A�%*

	conv_loss�^�>L��        )��P	�]����A�%*

	conv_loss��>$RF        )��P	������A�%*

	conv_loss|5�>,U�        )��P	������A�%*

	conv_lossd�>�@��        )��P	�����A�%*

	conv_loss�p�>�b�        )��P	G����A�%*

	conv_loss�d�>jGZ�        )��P	�L����A�%*

	conv_loss��>a�l        )��P	H}����A�%*

	conv_loss�c�>v�]�        )��P	R�����A�%*

	conv_loss�L�>�9 �        )��P	{�����A�%*

	conv_loss��>\�l<        )��P	����A�%*

	conv_lossV��>��B�        )��P	�>����A�%*

	conv_loss�ד>��[        )��P	�o����A�%*

	conv_loss�ܒ>]a�        )��P	G�����A�%*

	conv_lossNl�>��xp        )��P	������A�%*

	conv_loss��>����        )��P	� ����A�%*

	conv_loss�׎>|?N*        )��P	�/����A�%*

	conv_lossHŌ>�l<        )��P	�^����A�%*

	conv_loss���>���"        )��P	ɏ����A�%*

	conv_loss��>b�>        )��P	$�����A�%*

	conv_loss՗�>�S��        )��P	y�����A�%*

	conv_lossH�>O�7�        )��P	�����A�%*

	conv_losse��>�.�        )��P	�N����A�%*

	conv_loss!�>[�e        )��P	�����A�%*

	conv_loss)�>pV�`        )��P	S�����A�%*

	conv_loss��>���        )��P	������A�%*

	conv_lossW��>����        )��P	�&����A�%*

	conv_lossj�>�D�        )��P	�W����A�%*

	conv_loss�>h��        )��P	������A�%*

	conv_losso��>sO�'        )��P	������A�%*

	conv_loss�>�	�        )��P	������A�%*

	conv_loss�č>��U.        )��P	05����A�%*

	conv_lossa�>z[b        )��P	�f����A�%*

	conv_lossۗ�>G��         )��P	#�����A�%*

	conv_loss[�>�Օ�        )��P	A�����A�%*

	conv_loss���>�(�D        )��P	0�����A�%*

	conv_loss�C�>8��W        )��P	y6����A�%*

	conv_loss8ǎ>�	6�        )��P	�l����A�%*

	conv_loss�͒>�Q�        )��P	������A�%*

	conv_lossפ�>Q���        )��P	F�����A�%*

	conv_loss3��>�'�A        )��P	�����A�%*

	conv_loss��>��        )��P	�>����A�%*

	conv_loss�t�>��}        )��P	�q����A�%*

	conv_lossĬ�>��        )��P	Ԯ����A�%*

	conv_loss�>-�d]        )��P	1�����A�&*

	conv_lossKy�>ovV<        )��P	D����A�&*

	conv_loss���>�hN@        )��P	�>����A�&*

	conv_loss�>�>KN        )��P	n����A�&*

	conv_losswx�>�j�        )��P	)�����A�&*

	conv_loss��>S�c        )��P	�����A�&*

	conv_loss�o�>Nnm        )��P	�����A�&*

	conv_loss)�>��k        )��P	�9����A�&*

	conv_loss�q�>��/�        )��P	�h����A�&*

	conv_loss��>r�O        )��P	җ����A�&*

	conv_lossͨ�>>�$�        )��P	������A�&*

	conv_loss��>3�0        )��P		�����A�&*

	conv_loss���>Z/�        )��P	o'����A�&*

	conv_loss%˒>�}�a        )��P	�Y����A�&*

	conv_lossXn�>m5`�        )��P	P�����A�&*

	conv_loss��>��\�        )��P	y�����A�&*

	conv_loss#/�>N;��        )��P	������A�&*

	conv_lossǜ�> �b�        )��P	�����A�&*

	conv_loss��>��wj        )��P	�H����A�&*

	conv_lossm��>3���        )��P	w����A�&*

	conv_loss{�>(�        )��P	<�����A�&*

	conv_loss�c�>�r�        )��P	������A�&*

	conv_loss�v�>�vJ�        )��P	�����A�&*

	conv_loss�j�>oȏ{        )��P	+4����A�&*

	conv_loss��>ʏ@,        )��P	�b����A�&*

	conv_lossΏ>`��        )��P	������A�&*

	conv_loss�M�>By�V        )��P	�����A�&*

	conv_lossy��> ��W        )��P	������A�&*

	conv_lossX3�>� 6V        )��P	m ����A�&*

	conv_loss��>�i��        )��P	�S����A�&*

	conv_loss�a�>���        )��P	������A�&*

	conv_loss�o�>�f�        )��P	f�����A�&*

	conv_loss�)�>׸2�        )��P	"�����A�&*

	conv_loss�g�>h�td        )��P	�1����A�&*

	conv_loss�	�>C���        )��P	se����A�&*

	conv_loss�'�>���<        )��P	������A�&*

	conv_loss�4�>Iǔ        )��P	������A�&*

	conv_loss��>��i�        )��P	T����A�&*

	conv_lossX��>�S߀        )��P	?����A�&*

	conv_loss��>"L�        )��P	�o����A�&*

	conv_loss�ؑ>�WJ        )��P	������A�&*

	conv_loss3��>2Q��        )��P	������A�&*

	conv_lossh��>�Q�c        )��P	�����A�&*

	conv_lossy��>�`�8        )��P	H����A�&*

	conv_loss2C�>�N�        )��P	|����A�&*

	conv_loss	�>���        )��P	�����A�&*

	conv_loss��>��t%        )��P	E�����A�&*

	conv_loss� �>��        )��P	����A�&*

	conv_loss�͊>�Ӯ        )��P	$F����A�&*

	conv_loss^��>��_        )��P	�v����A�&*

	conv_loss�Џ>����        )��P	7�����A�&*

	conv_loss��>���        )��P	�����A�&*

	conv_loss�h�>L}%        )��P	�����A�&*

	conv_loss�O�>�2�        )��P	(4����A�&*

	conv_loss��>�u�m        )��P	�d����A�&*

	conv_loss���>�kD        )��P	W�����A�&*

	conv_losse&�>���        )��P	������A�&*

	conv_loss`��>���        )��P	�����A�&*

	conv_loss�V�>�d��        )��P	E5����A�&*

	conv_loss���>J)�        )��P	e����A�&*

	conv_lossP͍>�t�        )��P	�����A�&*

	conv_lossu��>�>8�        )��P	������A�&*

	conv_loss,��>���        )��P	������A�&*

	conv_loss��>RN��        )��P	�'����A�&*

	conv_loss�+�>7:�        )��P	V����A�&*

	conv_loss�u�>�23H        )��P	[�����A�&*

	conv_lossj>�>*��9        )��P	�����A�&*

	conv_loss��>��"        )��P	������A�&*

	conv_lossp��>q�0�        )��P	�����A�&*

	conv_loss�
�>	�        )��P	�B����A�&*

	conv_loss	��>�V��        )��P	�r����A�&*

	conv_loss��>��        )��P	������A�&*

	conv_lossچ�>���        )��P	T�����A�&*

	conv_loss�:�>�=��        )��P	F ����A�&*

	conv_loss��>�ю)        )��P	�/����A�&*

	conv_loss�?�>�S\=        )��P	�_����A�&*

	conv_loss���>%;��        )��P	������A�&*

	conv_loss��>yo        )��P	b�����A�&*

	conv_loss��>Y<q�        )��P	^�����A�&*

	conv_loss�;�>��!�        )��P	�����A�&*

	conv_loss�>�=        )��P	;S����A�&*

	conv_loss�(�>z�V�        )��P	q�����A�&*

	conv_loss���>�$�        )��P	b�����A�&*

	conv_loss�r�>�U�5        )��P	I�����A�&*

	conv_loss�o�>r�jf        )��P	�����A�&*

	conv_loss���>���        )��P	"I����A�&*

	conv_loss��>��4        )��P	.{����A�&*

	conv_lossR��>2�%�        )��P	\�����A�&*

	conv_loss@��>�-�p        )��P	�����A�&*

	conv_loss$V�>�"��        )��P	�����A�&*

	conv_loss:@�>rU��        )��P	PV����A�&*

	conv_loss�ڏ>�*�d        )��P	V�����A�&*

	conv_loss���>F3G@        )��P	!�����A�&*

	conv_loss��>Y7�        )��P	n�����A�&*

	conv_loss���>
��        )��P	J����A�&*

	conv_lossJg�>C�dF        )��P	^����A�&*

	conv_loss4��>���}        )��P	�����A�&*

	conv_lossi��>k[Z        )��P	������A�&*

	conv_loss
��>�ݏL        )��P	�����A�&*

	conv_loss�z�>���        )��P	ZH����A�&*

	conv_loss�s�>i��R        )��P	�����A�&*

	conv_loss��>%I��        )��P	A�����A�&*

	conv_loss�ב>8�G�        )��P	������A�&*

	conv_loss��>v�Y        )��P	����A�&*

	conv_loss�1�>Y�}^        )��P	EF����A�&*

	conv_loss[k�>nd�        )��P	������A�&*

	conv_loss�َ>���D        )��P	�����A�&*

	conv_loss^=�>"І�        )��P	;�����A�&*

	conv_loss�"�>B���        )��P	�*����A�&*

	conv_loss�}�>���        )��P	�_����A�&*

	conv_loss�>��\        )��P	������A�&*

	conv_loss�F�>:W�        )��P	t�����A�&*

	conv_loss�Y�>�?�[        )��P	������A�&*

	conv_loss��><$	t        )��P	�!����A�&*

	conv_loss��>��$�        )��P	FS����A�&*

	conv_loss���>���        )��P	˅����A�&*

	conv_loss&)�>�*ɷ        )��P	¶����A�&*

	conv_loss���>x+�2        )��P	(�����A�&*

	conv_loss6��>�ɛ        )��P	�����A�&*

	conv_loss�B�>>5��        )��P	KJ����A�&*

	conv_loss|K�>�yS�        )��P	~����A�&*

	conv_loss�D�>ۍkA        )��P	������A�&*

	conv_loss%�>s(5        )��P	������A�&*

	conv_loss^-�>`x�        )��P	����A�&*

	conv_loss�>�$�        )��P	�A����A�&*

	conv_loss���>.�Li        )��P	?q����A�&*

	conv_lossl��>UI�        )��P	5�����A�&*

	conv_loss���>�|��        )��P	y�����A�&*

	conv_loss�;�>��        )��P	�����A�'*

	conv_loss0��>ʿ�        )��P	�4����A�'*

	conv_loss�Ǒ>C�N        )��P	3e����A�'*

	conv_lossʂ�>����        )��P	&�����A�'*

	conv_loss��>���g        )��P	������A�'*

	conv_loss�>�+ؑ        )��P	�����A�'*

	conv_loss�>�
O~        )��P	�*����A�'*

	conv_loss�Ɛ>�5p        )��P	^����A�'*

	conv_lossP4�>�һ        )��P	������A�'*

	conv_loss
��>��J�        )��P	Cx���A�'*

	conv_loss��>��?v        )��P	�����A�'*

	conv_loss:{�>�L-        )��P	�����A�'*

	conv_loss=��>~�ö        )��P	����A�'*

	conv_lossV�>�{�        )��P	�8���A�'*

	conv_loss�F�>�]�        )��P	q���A�'*

	conv_loss��>��2        )��P	.����A�'*

	conv_loss�T�>��:v        )��P	^����A�'*

	conv_loss�-�>��s	        )��P	���A�'*

	conv_loss	{�>L�?        )��P	I���A�'*

	conv_loss�N�>��9
        )��P	�w���A�'*

	conv_loss�͎>����        )��P	�����A�'*

	conv_loss�X�>��        )��P	2����A�'*

	conv_losso�>K�_        )��P	����A�'*

	conv_loss�>-)��        )��P	�C���A�'*

	conv_lossSݎ>�p         )��P	�s���A�'*

	conv_loss)z�>L��        )��P	W����A�'*

	conv_loss���>�̰        )��P	_����A�'*

	conv_loss���>���[        )��P	����A�'*

	conv_loss^*�>����        )��P	N<���A�'*

	conv_lossō>��I�        )��P	Ro���A�'*

	conv_loss_s�>��1K        )��P	ڝ���A�'*

	conv_losss(�>���{        )��P	�����A�'*

	conv_loss��>��t�        )��P	�����A�'*

	conv_loss��>�<b        )��P	�-���A�'*

	conv_loss` �>�AJ        )��P	I^���A�'*

	conv_loss"�>��2        )��P	�����A�'*

	conv_loss>�>��v        )��P	����A�'*

	conv_loss{��>�{        )��P	m����A�'*

	conv_loss���>�o�        )��P	����A�'*

	conv_loss={�>��p        )��P	�M���A�'*

	conv_loss��>��=�        )��P	k����A�'*

	conv_loss"7�>k��m        )��P	v����A�'*

	conv_loss�#�>���?        )��P	�����A�'*

	conv_loss���>-�X        )��P	"&���A�'*

	conv_loss���>�q�        )��P	�U���A�'*

	conv_loss�C�>��5        )��P	�����A�'*

	conv_loss �>E�w�        )��P	̳���A�'*

	conv_loss�͍>���O        )��P	�����A�'*

	conv_lossQ�>�V        )��P	�'	���A�'*

	conv_loss�ݔ>E���        )��P	�X	���A�'*

	conv_losse6�>Luh�        )��P	��	���A�'*

	conv_lossX��>O�P�        )��P	��	���A�'*

	conv_loss�`�>���}        )��P	r�	���A�'*

	conv_loss/V�>e(��        )��P	�
���A�'*

	conv_loss��>,;X@        )��P	�E
���A�'*

	conv_loss�)�>�qx�        )��P	�v
���A�'*

	conv_loss@]�>�.v�        )��P	�
���A�'*

	conv_lossT{�>��}�        )��P	U�
���A�'*

	conv_loss���>�رN        )��P	H���A�'*

	conv_loss��>m{y�        )��P	�F���A�'*

	conv_loss)>�.�u        )��P	Mu���A�'*

	conv_loss��>b��        )��P	�����A�'*

	conv_loss��>�i        )��P	$����A�'*

	conv_lossی>�=ʂ        )��P	���A�'*

	conv_lossm�>@6�C        )��P	UA���A�'*

	conv_loss�Ɍ>��If        )��P	!t���A�'*

	conv_lossl�>���2        )��P	�����A�'*

	conv_loss�>�S��        )��P	�����A�'*

	conv_loss���>�Ȳ        )��P	O"���A�'*

	conv_loss�\�>}�S?        )��P	D^���A�'*

	conv_loss<ȍ>����        )��P	W����A�'*

	conv_loss�i�>E�        )��P	�����A�'*

	conv_loss�̌>׿        )��P	y����A�'*

	conv_loss4
�>�J        )��P	�&���A�'*

	conv_loss(�>
I7        )��P	`���A�'*

	conv_lossv�>���        )��P	Z����A�'*

	conv_loss:�>iAF�        )��P	�����A�'*

	conv_loss뽋>X�!�        )��P	A����A�'*

	conv_loss�#�>�'I�        )��P	�'���A�'*

	conv_loss2�>H|.�        )��P	<^���A�'*

	conv_loss�̌>�bؿ        )��P	;����A�'*

	conv_lossTV�>�M        )��P	����A�'*

	conv_loss���>0�        )��P	�����A�'*

	conv_loss���>hur,        )��P	�'���A�'*

	conv_loss��>��        )��P	�W���A�'*

	conv_loss���>�&c�        )��P	]����A�'*

	conv_lossV��>a8��        )��P	�����A�'*

	conv_loss�>� h�        )��P	F����A�'*

	conv_loss���>�SIH        )��P	Y���A�'*

	conv_lossTC�>]h        )��P	;I���A�'*

	conv_loss�8�>}�1L        )��P	�x���A�'*

	conv_loss�
�>�'�!        )��P	�����A�'*

	conv_loss-d�>\�f�        )��P	�����A�'*

	conv_loss,��>���M        )��P	D���A�'*

	conv_lossE�>�        )��P	�;���A�'*

	conv_loss�>q~��        )��P	�s���A�'*

	conv_loss0��>����        )��P	�����A�'*

	conv_lossTЉ>l,#e        )��P	b����A�'*

	conv_loss�>���        )��P	���A�'*

	conv_loss��>��        )��P	�2���A�'*

	conv_loss���>o��        )��P	�b���A�'*

	conv_loss�ċ>���        )��P	�����A�'*

	conv_loss��>�N�u        )��P	����A�'*

	conv_lossЍ>)a/        )��P	�����A�'*

	conv_lossP)�>�	�        )��P	e9���A�'*

	conv_loss@��>���k        )��P	mr���A�'*

	conv_loss"��>2��        )��P	[����A�'*

	conv_loss綊>���        )��P	�����A�'*

	conv_loss�Y�>>�        )��P	Y���A�'*

	conv_lossfm�>>��        )��P	]R���A�'*

	conv_loss���>�g%^        )��P	�����A�'*

	conv_loss,I�>��V        )��P	[����A�'*

	conv_loss{Ύ>L��-        )��P	�����A�'*

	conv_lossl>\U�u        )��P	�*���A�'*

	conv_loss��>���        )��P	�X���A�'*

	conv_loss/�>�k�        )��P	�����A�'*

	conv_loss��>���        )��P	4����A�'*

	conv_lossV��>x��>        )��P	&���A�'*

	conv_loss�P�>A_�P        )��P	�5���A�'*

	conv_loss�>���0        )��P	�f���A�'*

	conv_lossգ�>g��        )��P	i����A�'*

	conv_loss�[�>��G        )��P	�����A�'*

	conv_loss�%�>��Cw        )��P	U����A�'*

	conv_loss�l�>ρ>�        )��P	�+���A�'*

	conv_loss̍>Q8�        )��P	�Z���A�'*

	conv_loss���>N ��        )��P	����A�'*

	conv_loss+w�>��        )��P	n����A�'*

	conv_lossn�>.�KS        )��P	h����A�'*

	conv_loss��>8qZ�        )��P	_���A�(*

	conv_lossQ��>Q��d        )��P	R���A�(*

	conv_loss��>�"DT        )��P	i����A�(*

	conv_loss/K�>S�        )��P	����A�(*

	conv_loss���>t7Y        )��P	q����A�(*

	conv_loss)J�>���        )��P	R���A�(*

	conv_loss��>*��/        )��P	�H���A�(*

	conv_loss��>8�B�        )��P	�v���A�(*

	conv_lossQj�>��C�        )��P	�����A�(*

	conv_loss�>�V�        )��P	����A�(*

	conv_loss v�>ɓ        )��P	����A�(*

	conv_loss�7�>>�}�        )��P	�6���A�(*

	conv_loss�|�>7�z�        )��P	?h���A�(*

	conv_lossZ�>�հ        )��P	]����A�(*

	conv_loss(�>���z        )��P	�����A�(*

	conv_loss���>���        )��P	G����A�(*

	conv_loss{͊>[�3�        )��P	v'���A�(*

	conv_loss���>���        )��P	�W���A�(*

	conv_loss�`�>��O�        )��P	:����A�(*

	conv_loss���>�1<�        )��P	����A�(*

	conv_loss���>X�da        )��P	����A�(*

	conv_loss��>kx+$        )��P	j���A�(*

	conv_loss�%�>��f        )��P	�D���A�(*

	conv_loss���>lڠ�        )��P	�w���A�(*

	conv_loss���>��s        )��P	�����A�(*

	conv_loss�z�>vc�        )��P	�����A�(*

	conv_loss�}�>W�L�        )��P	�	���A�(*

	conv_loss	�>����        )��P	�9���A�(*

	conv_loss$ߋ>�`i�        )��P	�i���A�(*

	conv_loss��>��G&        )��P	 ����A�(*

	conv_loss*,�>=ߑ        )��P	)����A�(*

	conv_lossjg�>�A        )��P	����A�(*

	conv_lossD��>e#~+        )��P	p=���A�(*

	conv_loss��>��:        )��P	Ql���A�(*

	conv_loss��>R�        )��P	�����A�(*

	conv_loss���>�5�S        )��P	b����A�(*

	conv_loss��>�^�d        )��P	�  ���A�(*

	conv_loss��>g��        )��P	5/ ���A�(*

	conv_loss�ǉ>�ۜQ        )��P	�] ���A�(*

	conv_lossC�>qJ�8        )��P	_� ���A�(*

	conv_loss	��>�.�X        )��P	l� ���A�(*

	conv_loss��>�S��        )��P	5!���A�(*

	conv_lossV�>�/        )��P	6!���A�(*

	conv_loss�/�>��KG        )��P	�e!���A�(*

	conv_lossB��>�<p�        )��P	��!���A�(*

	conv_loss�.�>���        )��P	��!���A�(*

	conv_lossЌ>�n�H        )��P	"���A�(*

	conv_loss�\�>gpA�        )��P	d/"���A�(*

	conv_loss���>��        )��P	2_"���A�(*

	conv_losswG�>�gg        )��P	c�"���A�(*

	conv_lossv8�>��TC        )��P	�"���A�(*

	conv_loss�=�>�Y�        )��P	j�"���A�(*

	conv_loss���>?ˬz        )��P	|%#���A�(*

	conv_losspH�>�Lk�        )��P	�^#���A�(*

	conv_lossHn�>��Ax        )��P	1�#���A�(*

	conv_loss���>���        )��P	��#���A�(*

	conv_lossW�>��M        )��P	��#���A�(*

	conv_lossZ?�>��"e        )��P	�)$���A�(*

	conv_loss
��>ȨN�        )��P	�Y$���A�(*

	conv_loss#Ŋ>�o��        )��P	D�$���A�(*

	conv_loss��>�]�7        )��P	/�$���A�(*

	conv_loss���>��        )��P	��$���A�(*

	conv_lossYՍ>i��        )��P	%���A�(*

	conv_lossO/�>�	�4        )��P	#M%���A�(*

	conv_loss�s�>�Z?�        )��P	}%���A�(*

	conv_loss���>�9*        )��P	w�%���A�(*

	conv_loss�&�>��        )��P	m�%���A�(*

	conv_lossG�>5�s        )��P	�&���A�(*

	conv_loss�/�>�'S        )��P	�>&���A�(*

	conv_loss��>�,�;        )��P	�n&���A�(*

	conv_loss[ċ>zN}u        )��P	̞&���A�(*

	conv_loss�g�>��        )��P	w�&���A�(*

	conv_loss�{�>ٷ�        )��P	� '���A�(*

	conv_losscb�>��.        )��P	�0'���A�(*

	conv_lossއ�>��_�        )��P	a'���A�(*

	conv_loss���>}���        )��P	�'���A�(*

	conv_loss"�>�~�I        )��P	��'���A�(*

	conv_loss婄>��i8        )��P	��'���A�(*

	conv_loss�/�>
��        )��P	�%(���A�(*

	conv_lossWz�>:k[        )��P	~X(���A�(*

	conv_lossoZ�>\���        )��P	_�(���A�(*

	conv_lossy�>��'R        )��P	�,*���A�(*

	conv_loss3�>�>�        )��P	�\*���A�(*

	conv_loss0֊>�pU        )��P	!�*���A�(*

	conv_lossvk�>S�љ        )��P	��*���A�(*

	conv_loss�]�>��ދ        )��P	��*���A�(*

	conv_loss/�>��1"        )��P	�1+���A�(*

	conv_loss:}�>��m        )��P	�e+���A�(*

	conv_lossX�>���        )��P	ߖ+���A�(*

	conv_loss=2�>�g�        )��P	��+���A�(*

	conv_lossz��>�l�        )��P	[
,���A�(*

	conv_loss�R�>�#�        )��P	a:,���A�(*

	conv_loss�s�>���        )��P	cj,���A�(*

	conv_loss�O�>��        )��P	ۚ,���A�(*

	conv_loss�-�>.8Y�        )��P	I�,���A�(*

	conv_loss�!�>����        )��P	�-���A�(*

	conv_loss[��>��d        )��P	�2-���A�(*

	conv_lossW�>�z�u        )��P	�j-���A�(*

	conv_lossɋ>��Py        )��P	��-���A�(*

	conv_lossՌ�>�~v        )��P	��-���A�(*

	conv_lossF�>-�D6        )��P	�.���A�(*

	conv_lossҕ�>��8i        )��P	B.���A�(*

	conv_loss�C�>��        )��P	s.���A�(*

	conv_loss�Њ>��          )��P	U�.���A�(*

	conv_loss >���N        )��P	��.���A�(*

	conv_loss2�>���        )��P	7/���A�(*

	conv_lossǋ>��        )��P	�6/���A�(*

	conv_loss���>OJ7`        )��P	�h/���A�(*

	conv_loss`.�>'�	!        )��P	��/���A�(*

	conv_loss�.�>�3B        )��P	��/���A�(*

	conv_lossK�>��L�        )��P	0���A�(*

	conv_loss��>R���        )��P	�C0���A�(*

	conv_loss��>~        )��P	�v0���A�(*

	conv_loss�+�>k��        )��P	*�0���A�(*

	conv_lossډ>ˡ?        )��P	��0���A�(*

	conv_lossa/�>\wS�        )��P	+1���A�(*

	conv_lossy@�>���        )��P	81���A�(*

	conv_loss�>a}Y        )��P	Ih1���A�(*

	conv_loss�ͅ>r�/        )��P	�1���A�(*

	conv_loss
|�>�}�        )��P	��1���A�(*

	conv_lossrz�>>���        )��P	�1���A�(*

	conv_loss�e�>^�        )��P	�+2���A�(*

	conv_losscn�>ǃ�F        )��P	�_2���A�(*

	conv_loss�+�>��1�        )��P	��2���A�(*

	conv_loss��>�1
        )��P	��2���A�(*

	conv_loss�?�>`�V        )��P	��2���A�(*

	conv_lossy@�>�?�        )��P	a!3���A�(*

	conv_loss���><��=        )��P	[R3���A�(*

	conv_lossS�>��j        )��P	)�3���A�)*

	conv_loss��>�#U�        )��P	��3���A�)*

	conv_loss��>�        )��P	��3���A�)*

	conv_loss��>,��`        )��P	�.4���A�)*

	conv_lossX��>�G��        )��P	�f4���A�)*

	conv_loss�Њ>lBS�        )��P	g�4���A�)*

	conv_loss2��>���        )��P	^�4���A�)*

	conv_lossU��>���e        )��P	?5���A�)*

	conv_loss���>�,�j        )��P	�>5���A�)*

	conv_loss��>��Km        )��P	<w5���A�)*

	conv_loss�>�F�#        )��P	_�5���A�)*

	conv_lossم>o9        )��P	��5���A�)*

	conv_loss~ �>MF�        )��P	F6���A�)*

	conv_loss.r�>��Ց        )��P	>6���A�)*

	conv_loss r�>�m2        )��P	�s6���A�)*

	conv_loss��>���)        )��P	�6���A�)*

	conv_loss�\�>ݛ�        )��P	��6���A�)*

	conv_loss�M�>���\        )��P	�7���A�)*

	conv_loss؍>=cŜ        )��P	p57���A�)*

	conv_lossXۄ>�iu        )��P	�f7���A�)*

	conv_lossz0�>9E��        )��P	��7���A�)*

	conv_loss�> ���        )��P	��7���A�)*

	conv_loss�>��        )��P	&8���A�)*

	conv_loss���>o�i        )��P	�=8���A�)*

	conv_loss��>s�)        )��P	�x8���A�)*

	conv_loss;<�>���u        )��P	�8���A�)*

	conv_losse�>����        )��P	��8���A�)*

	conv_lossC��>,H/�        )��P	>9���A�)*

	conv_loss듆>�f��        )��P	E<9���A�)*

	conv_loss?p�>M9�n        )��P	0l9���A�)*

	conv_loss#��>���        )��P	ǝ9���A�)*

	conv_loss�S�>;��        )��P	��9���A�)*

	conv_loss�g�>S��A        )��P	�:���A�)*

	conv_loss��>�oDJ        )��P	B:���A�)*

	conv_losss�>%�s�        )��P	��:���A�)*

	conv_loss.އ>L l        )��P	̲:���A�)*

	conv_loss��>	Sj        )��P	+�:���A�)*

	conv_lossj��><:\6        )��P	�;���A�)*

	conv_loss2o�>k�&(        )��P	�D;���A�)*

	conv_loss��>)=eN        )��P	lu;���A�)*

	conv_loss�D�>Z�JY        )��P	��;���A�)*

	conv_lossF��>=�7�        )��P	!�;���A�)*

	conv_loss�>�>�        )��P	B<���A�)*

	conv_lossu��>nMP        )��P	5><���A�)*

	conv_loss��>%���        )��P	�o<���A�)*

	conv_loss~��>��>        )��P	ʳ<���A�)*

	conv_loss�G�>�(��        )��P	��<���A�)*

	conv_loss,_�>/	`�        )��P	*=���A�)*

	conv_loss��>�P}X        )��P	�I=���A�)*

	conv_lossa(�><��        )��P		�=���A�)*

	conv_loss>�|<        )��P	@�=���A�)*

	conv_loss�X�>G%��        )��P	��=���A�)*

	conv_lossj�>l�        )��P	�>���A�)*

	conv_loss�+�>q�h        )��P	4Y>���A�)*

	conv_loss���>g�>�        )��P	d�>���A�)*

	conv_loss�S�>?Q�(        )��P	��>���A�)*

	conv_loss�*�>@2�        )��P	:�>���A�)*

	conv_loss���>�b        )��P	�3?���A�)*

	conv_lossJ��>�#�        )��P	�s?���A�)*

	conv_loss.ȅ>��@        )��P	�?���A�)*

	conv_loss� �>����        )��P	�?���A�)*

	conv_loss�B�>4���        )��P	_@���A�)*

	conv_loss=Ҍ>�:�        )��P	9@���A�)*

	conv_loss��>��;        )��P	%k@���A�)*

	conv_loss�U�>�_�K        )��P	�@���A�)*

	conv_lossM��>�i}�        )��P	�@���A�)*

	conv_lossU�>�=�+        )��P	�A���A�)*

	conv_loss���>+��J        )��P	�CA���A�)*

	conv_loss\e�>��Z        )��P	sA���A�)*

	conv_loss���>'��{        )��P	��A���A�)*

	conv_loss�>N`�        )��P	��A���A�)*

	conv_lossLb�>��Q        )��P	(B���A�)*

	conv_loss���>>��        )��P	^AB���A�)*

	conv_lossg�>^��        )��P	�qB���A�)*

	conv_loss\q�>sjE�        )��P	=�B���A�)*

	conv_loss`��>��@�        )��P	�B���A�)*

	conv_lossd�>|��        )��P	|C���A�)*

	conv_lossb3�>h�*e        )��P	�5C���A�)*

	conv_loss}�>���        )��P	�fC���A�)*

	conv_lossK�>��{        )��P	��C���A�)*

	conv_loss�>��z'        )��P	`�C���A�)*

	conv_loss�>��7        )��P	��C���A�)*

	conv_loss8�z>ha�        )��P	�-D���A�)*

	conv_loss���>��4        )��P	T^D���A�)*

	conv_loss�o�>�o        )��P	V�D���A�)*

	conv_lossmw�>���	        )��P	��D���A�)*

	conv_loss.��>��Wm        )��P	��D���A�)*

	conv_loss-)�>�Q�        )��P	F)E���A�)*

	conv_lossf�>�0�@        )��P	�YE���A�)*

	conv_loss#~�>N�        )��P	z�E���A�)*

	conv_loss�4�>P,B�        )��P	��E���A�)*

	conv_lossއ>e��S        )��P	��E���A�)*

	conv_loss�>��        )��P	�F���A�)*

	conv_loss��>�TF�        )��P	2KF���A�)*

	conv_loss���>����        )��P	+|F���A�)*

	conv_lossʠ�>����        )��P	ʫF���A�)*

	conv_loss5˃>cx�|        )��P	�F���A�)*

	conv_loss�>;H]P        )��P	�G���A�)*

	conv_loss!,�>�'O7        )��P	"CG���A�)*

	conv_loss�>?MY�        )��P	wG���A�)*

	conv_losse׆>���        )��P	ީG���A�)*

	conv_lossX؇>��Nd        )��P	�G���A�)*

	conv_lossi��>∤
        )��P	�H���A�)*

	conv_lossZ��>d�g        )��P	�OH���A�)*

	conv_lossB@�>^�&        )��P	&�H���A�)*

	conv_loss>��>���        )��P	6�H���A�)*

	conv_loss�h�>�r��        )��P	W�H���A�)*

	conv_lossT�>��U�        )��P	�I���A�)*

	conv_loss��>�TuY        )��P	TI���A�)*

	conv_loss��>��g        )��P	b�I���A�)*

	conv_loss@��>�*        )��P	W�I���A�)*

	conv_loss�>�۾p        )��P	��I���A�)*

	conv_loss��>�9�t        )��P	�-J���A�)*

	conv_loss�L�>�X!�        )��P	�^J���A�)*

	conv_loss���>1�        )��P	L�J���A�)*

	conv_loss��>���        )��P	<�J���A�)*

	conv_loss�Æ>�JF�        )��P	�J���A�)*

	conv_loss�>���        )��P	�.K���A�)*

	conv_lossBh�>���W        )��P	�aK���A�)*

	conv_loss'!�>f�e        )��P	�K���A�)*

	conv_loss��>���        )��P	�K���A�)*

	conv_loss�<�>�C�        )��P	rL���A�)*

	conv_losso�>j���        )��P	�EL���A�)*

	conv_loss�>5��'        )��P	zuL���A�)*

	conv_loss��z>N���        )��P	�L���A�)*

	conv_lossׅ>��        )��P	��L���A�)*

	conv_loss���>jri        )��P	�M���A�)*

	conv_loss��>�� �        )��P	u3M���A�)*

	conv_loss��>���s        )��P	cM���A�**

	conv_loss/�>���d        )��P	Q�M���A�**

	conv_loss��>ȃ<�        )��P	�M���A�**

	conv_loss�܆>��݅        )��P	��M���A�**

	conv_lossDʂ>W��        )��P	)&N���A�**

	conv_loss��>烞�        )��P	ZN���A�**

	conv_loss���>p�z�        )��P	I�N���A�**

	conv_loss���>�4        )��P	o�N���A�**

	conv_loss�r�>�R�g        )��P	��N���A�**

	conv_loss��>t0        )��P	J O���A�**

	conv_lossws�>\Hf        )��P	�PO���A�**

	conv_loss�'�>�8�        )��P	��O���A�**

	conv_loss�S�>c�
{        )��P	F�O���A�**

	conv_loss}�>��T        )��P	��O���A�**

	conv_loss��>���4        )��P	IP���A�**

	conv_loss��}>@�&.        )��P	�EP���A�**

	conv_loss��~>�C�        )��P	�wP���A�**

	conv_loss���>��K/        )��P	��P���A�**

	conv_loss��>���        )��P	��P���A�**

	conv_loss�ȃ>^W�        )��P	�Q���A�**

	conv_loss_�>���        )��P	?>Q���A�**

	conv_loss*�>��V        )��P	{pQ���A�**

	conv_lossT�>#��        )��P	�Q���A�**

	conv_lossZy�>��        )��P	��Q���A�**

	conv_loss��>0�G<        )��P	�R���A�**

	conv_lossы�>�1�j        )��P	ΜS���A�**

	conv_lossh��>h�"        )��P	K�S���A�**

	conv_loss�Հ>��0�        )��P	��S���A�**

	conv_loss���>�*�        )��P	�2T���A�**

	conv_lossQy�> JK        )��P	mcT���A�**

	conv_lossH�>L[�w        )��P	�T���A�**

	conv_loss.�>o�        )��P	��T���A�**

	conv_loss͋>IQ�        )��P	�T���A�**

	conv_loss�{>�{a        )��P	/U���A�**

	conv_loss�>{�޵        )��P	U\U���A�**

	conv_loss�+�>��C        )��P	U�U���A�**

	conv_loss���>����        )��P	9�U���A�**

	conv_lossƆ>�?�=        )��P	�U���A�**

	conv_loss4V�>�k�-        )��P	�'V���A�**

	conv_losscJ�>*:��        )��P	&_V���A�**

	conv_loss�}>��Q�        )��P	��V���A�**

	conv_loss Pt>z�7        )��P	M�V���A�**

	conv_losso��>�2ʫ        )��P	��V���A�**

	conv_lossA�>���        )��P	HW���A�**

	conv_lossj8�>��&�        )��P	zIW���A�**

	conv_loss�U�>�%��        )��P	�{W���A�**

	conv_loss��z>�t��        )��P	ڬW���A�**

	conv_loss�E�>[7�7        )��P	8�W���A�**

	conv_losss��>� u�        )��P	X���A�**

	conv_loss�c�>�5e�        )��P	�:X���A�**

	conv_loss�Ն>D�Y        )��P	�iX���A�**

	conv_loss,�>{��9        )��P	�X���A�**

	conv_loss�F�>��        )��P	��X���A�**

	conv_loss��>ǟ"        )��P	J�X���A�**

	conv_lossx��>��b        )��P	)Y���A�**

	conv_lossg�>���        )��P	RWY���A�**

	conv_lossB:�>�]�        )��P	ֈY���A�**

	conv_lossu��>�$�         )��P	̷Y���A�**

	conv_lossi3�>5g/�        )��P	G�Y���A�**

	conv_loss�h�>8m�s        )��P	�Z���A�**

	conv_loss�K~>р�
        )��P	�GZ���A�**

	conv_loss?��>d9_m        )��P	�xZ���A�**

	conv_loss���>3F2        )��P	:�Z���A�**

	conv_loss���>P��        )��P	{�Z���A�**

	conv_loss\ف>UW��        )��P	
[���A�**

	conv_lossV3�>k��        )��P	�9[���A�**

	conv_lossŀ>�֝�        )��P	�k[���A�**

	conv_loss.�> �
A        )��P	�[���A�**

	conv_loss�/�>n�        )��P	j�[���A�**

	conv_loss�$�>�(        )��P	�[���A�**

	conv_loss�0�>Xn�	        )��P	�1\���A�**

	conv_loss{Ā>�2        )��P	qc\���A�**

	conv_loss�T�>�	�>        )��P	Ě\���A�**

	conv_loss�w�>�)�5        )��P	��\���A�**

	conv_loss��>���        )��P	]���A�**

	conv_lossJ}�>O�%�        )��P	�P]���A�**

	conv_loss��>Q�d        )��P	��]���A�**

	conv_lossA��>��V        )��P	��]���A�**

	conv_loss[��>#{~        )��P	��]���A�**

	conv_loss[j{>��]        )��P	a*^���A�**

	conv_loss��}>\���        )��P	c`^���A�**

	conv_loss8�>:,�(        )��P	Z�^���A�**

	conv_loss?�>�C�        )��P	��^���A�**

	conv_lossB��>_��        )��P	��^���A�**

	conv_loss}��>5��        )��P	�._���A�**

	conv_loss�k}>�$jI        )��P	�d_���A�**

	conv_loss�y�>�z�,        )��P	��_���A�**

	conv_loss�K}>::��        )��P		�_���A�**

	conv_loss΃>S�YS        )��P	�`���A�**

	conv_loss�р>��        )��P	�7`���A�**

	conv_loss�?|>�3IJ        )��P	�g`���A�**

	conv_loss`6�>�ĵx        )��P	��`���A�**

	conv_lossU~>:�E        )��P	��`���A�**

	conv_loss2�>����        )��P	�a���A�**

	conv_loss�A~>=2��        )��P	s8a���A�**

	conv_lossM:�>�؟        )��P	@ha���A�**

	conv_lossDg�>HI^q        )��P	j�a���A�**

	conv_loss��{>�7        )��P	��a���A�**

	conv_losse��>PI�        )��P	��a���A�**

	conv_loss�_�>���        )��P	7,b���A�**

	conv_loss[�>l�\�        )��P	.^b���A�**

	conv_loss��>i��+        )��P	0�b���A�**

	conv_lossޏ�>R��K        )��P	��b���A�**

	conv_loss<�><)=        )��P	+�b���A�**

	conv_loss��>���Z        )��P	p$c���A�**

	conv_loss��>?2        )��P	�Vc���A�**

	conv_loss�A�>v�<        )��P	��c���A�**

	conv_lossڠ�>�-�        )��P	��c���A�**

	conv_loss��y>�`S�        )��P	l�c���A�**

	conv_loss��>�T@�        )��P	�d���A�**

	conv_loss��>j*��        )��P	xGd���A�**

	conv_loss+��>kW�        )��P	�ud���A�**

	conv_loss�b�>K��        )��P	��d���A�**

	conv_loss��~>&>DD        )��P	3�d���A�**

	conv_loss�a�>�Z)�        )��P	�e���A�**

	conv_loss!y>U&_         )��P	:e���A�**

	conv_lossSk�>�[��        )��P	�le���A�**

	conv_loss�|�>�Ր�        )��P	��e���A�**

	conv_loss��>�'��        )��P	=�e���A�**

	conv_loss���>�|��        )��P	zf���A�**

	conv_lossYl�>�        )��P	�4f���A�**

	conv_lossT}>ؗ��        )��P	�gf���A�**

	conv_lossr�t>:�m        )��P	��f���A�**

	conv_loss�x>���        )��P	��f���A�**

	conv_lossBx>���        )��P	��k���A�**

	conv_loss��>�ar        )��P	y�k���A�**

	conv_loss� �>���        )��P	7�k���A�**

	conv_loss��>B��n        )��P	�'l���A�**

	conv_loss"ŀ>iϘm        )��P	�Ul���A�+*

	conv_loss��>�F7        )��P	��l���A�+*

	conv_loss|�~>��bM        )��P	{�l���A�+*

	conv_loss��>�        )��P	�l���A�+*

	conv_loss��x>�M�5        )��P	�)m���A�+*

	conv_loss��|>� y�        )��P	�`m���A�+*

	conv_loss�r~>��+        )��P	S�m���A�+*

	conv_loss�>�ud4        )��P	7�m���A�+*

	conv_loss��>3��        )��P	��m���A�+*

	conv_lossF�>H�-        )��P	�+n���A�+*

	conv_loss�K�>R��        )��P	cYn���A�+*

	conv_loss�x|>'E��        )��P	&�n���A�+*

	conv_loss�Ȅ>±]�        )��P	F�n���A�+*

	conv_lossʉ�>��v        )��P	A�n���A�+*

	conv_loss���>�(~�        )��P	g&o���A�+*

	conv_loss�>��/w        )��P	�`o���A�+*

	conv_losso��>@b��        )��P	�o���A�+*

	conv_loss��}>r�p�        )��P	�o���A�+*

	conv_loss,�l>_���        )��P	��o���A�+*

	conv_loss}u>�tZ        )��P	9p���A�+*

	conv_loss�_�>�Ǵ�        )��P	~hp���A�+*

	conv_loss��>7��y        )��P	��p���A�+*

	conv_loss�#�>3_�        )��P	��p���A�+*

	conv_loss+[�>�t��        )��P	��p���A�+*

	conv_loss��v>���        )��P	�%q���A�+*

	conv_lossHM�>UAX        )��P	RTq���A�+*

	conv_lossNF�>�P�        )��P	f�q���A�+*

	conv_loss�L~>��        )��P	E�q���A�+*

	conv_loss��v>�T*        )��P	d�q���A�+*

	conv_loss�.�>���E        )��P	C)r���A�+*

	conv_loss�{>.G2�        )��P	�Yr���A�+*

	conv_loss~�w>Y
?h        )��P	t�r���A�+*

	conv_lossf�>%�ߜ        )��P	��r���A�+*

	conv_loss��r>:a�@        )��P	��r���A�+*

	conv_loss-rt>'�K�        )��P	�s���A�+*

	conv_loss�ρ>.�*�        )��P	�Es���A�+*

	conv_loss��>@-?        )��P	�ts���A�+*

	conv_lossb�|>��߹        )��P	̦s���A�+*

	conv_lossD{>��ɝ        )��P	��s���A�+*

	conv_loss�9�>�Dp�        )��P	�t���A�+*

	conv_lossC�~>U9�        )��P	�6t���A�+*

	conv_lossU��>w���        )��P	�ft���A�+*

	conv_losswFx>�}��        )��P	ʕt���A�+*

	conv_loss9�s>��c�        )��P	^�t���A�+*

	conv_lossS}>AR�j        )��P	��t���A�+*

	conv_lossÖ>0J        )��P	�"u���A�+*

	conv_loss��}>�|��        )��P	�Qu���A�+*

	conv_loss�<�>X��        )��P	8�u���A�+*

	conv_lossV��>���        )��P	��u���A�+*

	conv_loss��|>�$�        )��P	��u���A�+*

	conv_lossO��>$X6�        )��P	�'v���A�+*

	conv_lossa&�>�y/        )��P	�Yv���A�+*

	conv_loss�k�>�}�        )��P	؏v���A�+*

	conv_lossj�>@WD        )��P	�v���A�+*

	conv_loss��r>+��        )��P	i�v���A�+*

	conv_loss߂>���)        )��P	C5w���A�+*

	conv_loss��>"���        )��P	�jw���A�+*

	conv_loss�[}>�܍        )��P	��w���A�+*

	conv_loss�3>0��        )��P	��w���A�+*

	conv_loss�t�>2"g�        )��P	��w���A�+*

	conv_lossX:�>w0�R        )��P	Y/x���A�+*

	conv_loss��>��d�        )��P	�hx���A�+*

	conv_loss�\�>����        )��P	v�x���A�+*

	conv_loss(��>�        )��P	�x���A�+*

	conv_loss��>�[3�        )��P	��x���A�+*

	conv_loss�P�>?�@        )��P	�1y���A�+*

	conv_loss�@t>ZG�6        )��P	 jy���A�+*

	conv_loss(u>��Q�        )��P	��y���A�+*

	conv_lossH�q>^�W�        )��P	��y���A�+*

	conv_loss�>����        )��P	6�y���A�+*

	conv_loss�y>���        )��P	1z���A�+*

	conv_lossLN|>����        )��P	�`z���A�+*

	conv_loss�z�>w�t�        )��P	�z���A�+*

	conv_loss���>���"        )��P	z�z���A�+*

	conv_lossz>�d�l        )��P	 �z���A�+*

	conv_loss��>�n�        )��P	.${���A�+*

	conv_loss��x>���         )��P	�U{���A�+*

	conv_loss�q�>`�r�        )��P	_�{���A�+*

	conv_loss�>;J�?        )��P	e�{���A�+*

	conv_loss�>&T4�        )��P	�{���A�+*

	conv_loss0T�>Y��        )��P	�|���A�+*

	conv_loss$\�>u�u        )��P	L|���A�+*

	conv_lossӅo>�        )��P	L||���A�+*

	conv_loss)z>L��        )��P	��|���A�+*

	conv_loss>n]ُ        )��P	�|���A�+*

	conv_loss�Cy>�!U        )��P	�}���A�+*

	conv_loss;3�>��:        )��P	3A}���A�+*

	conv_lossXOu>����        )��P	�q}���A�+*

	conv_loss��{>����        )��P	��}���A�+*

	conv_loss��>v+��        )��P	;�}���A�+*

	conv_losso�>�d��        )��P	~~���A�+*

	conv_loss��~>O���        )��P	�3~���A�+*

	conv_loss��~>����        )��P	If~���A�+*

	conv_loss&�y>�a�X        )��P	Ӕ~���A�+*

	conv_losss>�nx        )��P	��~���A�+*

	conv_loss���>�`��        )��P	��~���A�+*

	conv_loss��p>|=�Q        )��P	�)���A�+*

	conv_loss]ǀ>3#y�        )��P	������A�+*

	conv_loss��x>5Xs        )��P	�����A�+*

	conv_loss��{>��rU        )��P	�����A�+*

	conv_lossrq>�v�X        )��P	ZV����A�+*

	conv_lossX{>��        )��P	�����A�+*

	conv_loss+	�>t��        )��P	�ā���A�+*

	conv_loss��z>��&�        )��P	������A�+*

	conv_lossJv>���8        )��P	�(����A�+*

	conv_loss��{>����        )��P	�_����A�+*

	conv_loss�r>-y��        )��P	�����A�+*

	conv_loss0�w>y4�|        )��P	�̂���A�+*

	conv_losse�>`$P�        )��P	������A�+*

	conv_loss��>�e@y        )��P	5����A�+*

	conv_loss�|>��e        )��P	Sk����A�+*

	conv_loss��}>
oڧ        )��P	������A�+*

	conv_lossM�z>�a�        )��P	mփ���A�+*

	conv_loss��k>�'BJ        )��P	
����A�+*

	conv_lossT�r>��D*        )��P	�;����A�+*

	conv_loss�~>aw        )��P	Wl����A�+*

	conv_loss�x>�%�F        )��P	4�����A�+*

	conv_loss�D>���        )��P	�̄���A�+*

	conv_loss>�>�U�`        )��P	$�����A�+*

	conv_lossت~>ǵ�=        )��P	:/����A�+*

	conv_loss��}>�
��        )��P	�`����A�+*

	conv_loss`�n>�}K�        )��P	t�����A�+*

	conv_lossa߃>��3        )��P	I���A�+*

	conv_lossI�t>��        )��P	L����A�+*

	conv_lossÎ~>��%        )��P	�"����A�+*

	conv_loss�w>;��*        )��P	�S����A�+*

	conv_loss�t>�hI        )��P	������A�+*

	conv_lossூ>�]��        )��P	I�����A�+*

	conv_loss��{>�9��        )��P	:����A�,*

	conv_loss���>��e�        )��P	;����A�,*

	conv_loss��}>����        )��P	�G����A�,*

	conv_losss>1��         )��P	v����A�,*

	conv_loss��{>��<z        )��P	R�����A�,*

	conv_loss�9}>O�M        )��P	�ه���A�,*

	conv_loss`aw>��        )��P	�����A�,*

	conv_loss�k>�9��        )��P	�=����A�,*

	conv_loss	ҁ>@:�        )��P	jp����A�,*

	conv_lossldv>dH��        )��P	"�����A�,*

	conv_loss&�r>�m�        )��P	ш���A�,*

	conv_loss�!v>"}�        )��P	�����A�,*

	conv_loss��p>���	        )��P	5����A�,*

	conv_loss��z>��,        )��P	ff����A�,*

	conv_lossC�v>����        )��P	2�����A�,*

	conv_loss6�{>P�R        )��P	�ʉ���A�,*

	conv_loss��y>M�        )��P	� ����A�,*

	conv_loss�Ԁ><��        )��P	=2����A�,*

	conv_loss��x>:t�+        )��P	�b����A�,*

	conv_loss�wu>���8        )��P	������A�,*

	conv_lossd1�>�A        )��P	܊���A�,*

	conv_lossu:t>���        )��P	�����A�,*

	conv_loss�2o>��y=        )��P	+I����A�,*

	conv_loss|>�v�        )��P	n{����A�,*

	conv_loss�w>��g�        )��P	#�����A�,*

	conv_loss��l>���R        )��P	����A�,*

	conv_loss�Lk>���        )��P	����A�,*

	conv_loss�S�>kTGQ        )��P	UN����A�,*

	conv_loss��p>��t3        )��P	������A�,*

	conv_losss�u>xx{h        )��P	������A�,*

	conv_losslt>էK�        )��P	^����A�,*

	conv_loss}>��O        )��P	Y(����A�,*

	conv_lossTjz>!
~�        )��P	�a����A�,*

	conv_loss��v>���        )��P	ʚ����A�,*

	conv_loss�D}>}8X        )��P	�΍���A�,*

	conv_lossv>�F�R        )��P	+����A�,*

	conv_loss[�>���        )��P	A4����A�,*

	conv_lossv��>�>�k        )��P	.f����A�,*

	conv_loss�9}>-ҹ        )��P	#�����A�,*

	conv_loss=h>��B        )��P	Ɏ���A�,*

	conv_loss'a�>��        )��P	�����A�,*

	conv_loss�#�>�W�        )��P	,����A�,*

	conv_lossc�x>�3��        )��P	b����A�,*

	conv_loss:
r>�{3        )��P	2�����A�,*

	conv_loss�Fy>d�0        )��P	ŏ���A�,*

	conv_lossK#k>�R�        )��P	�����A�,*

	conv_loss�y>w���        )��P	^B����A�,*

	conv_lossjrg>��1        )��P	ts����A�,*

	conv_lossy�>T/�        )��P	+�����A�,*

	conv_lossq�|>�#b        )��P	�֐���A�,*

	conv_loss�>�7fq        )��P	.����A�,*

	conv_lossI"s>E�L�        )��P	e6����A�,*

	conv_loss"�w>�e�~        )��P	yi����A�,*

	conv_loss��>�z)�        )��P	������A�,*

	conv_lossa�f>�[�a        )��P	�ˑ���A�,*

	conv_loss1�w>��Bt        )��P	.�����A�,*

	conv_loss<w>1%�        )��P	v2����A�,*

	conv_loss�nq>Ǡv=        )��P	�l����A�,*

	conv_loss�r>Aף�        )��P	U�����A�,*

	conv_loss��y>���[        )��P	�Ғ���A�,*

	conv_lossǲk>��W�        )��P	N����A�,*

	conv_loss�:r>�)��        )��P	24����A�,*

	conv_loss/�>�o��        )��P	gd����A�,*

	conv_loss�bf>�mvr        )��P	�����A�,*

	conv_lossXkq>�qp        )��P	ȓ���A�,*

	conv_loss�v>���        )��P	������A�,*

	conv_loss��u>A��        )��P	?4����A�,*

	conv_loss�'t>!��        )��P	�d����A�,*

	conv_loss�r> ��}        )��P	֔����A�,*

	conv_loss��m>lGaL        )��P	�ٔ���A�,*

	conv_loss��>�K�        )��P	�����A�,*

	conv_loss��k>dR�u        )��P	�G����A�,*

	conv_loss��t>�і        )��P	�~����A�,*

	conv_loss��n>쓶�        )��P	������A�,*

	conv_loss�Zp>�ב�        )��P	�����A�,*

	conv_loss9o>��j        )��P	�����A�,*

	conv_loss��x>�7�        )��P	FR����A�,*

	conv_loss��s>{Z�,        )��P	V�����A�,*

	conv_lossN�x>��cF        )��P	������A�,*

	conv_loss�\m>�>)\        )��P	�����A�,*

	conv_loss�m>{4r        )��P	\����A�,*

	conv_loss�w>�h�        )��P	iS����A�,*

	conv_loss�wx>`���        )��P	�����A�,*

	conv_loss�/v>�E��        )��P	&�����A�,*

	conv_loss�~s>Մ��        )��P	������A�,*

	conv_loss�~y>�R~        )��P	{'����A�,*

	conv_loss�jq>}�L�        )��P	9]����A�,*

	conv_lossa�|>���        )��P	/�����A�,*

	conv_loss=t>��.        )��P	Ⱦ����A�,*

	conv_loss�@n>�/~P        )��P	�����A�,*

	conv_lossϔk>_�2�        )��P	� ����A�,*

	conv_loss��i>���        )��P	�P����A�,*

	conv_loss�c|>_�p{        )��P	t�����A�,*

	conv_loss��n>����        )��P	d�����A�,*

	conv_losse�u>(S        )��P	�����A�,*

	conv_loss5�u>�g�Y        )��P	�����A�,*

	conv_lossSKx>�3B        )��P	DU����A�,*

	conv_loss(+w>���^        )��P	c�����A�,*

	conv_lossچx>L�'        )��P	������A�,*

	conv_lossX�v>��jH        )��P	�����A�,*

	conv_lossr>*g*=        )��P	}#����A�,*

	conv_loss�r>2�(�        )��P	�T����A�,*

	conv_loss-|o>5���        )��P	������A�,*

	conv_lossXHt>��%2        )��P	�����A�,*

	conv_lossKTm>����        )��P	�����A�,*

	conv_loss�p>��	        )��P	0����A�,*

	conv_loss��w>m]�        )��P	�N����A�,*

	conv_lossp&r>����        )��P	#~����A�,*

	conv_loss�m>P*ʈ        )��P	R�����A�,*

	conv_losss>�uC        )��P	fߜ���A�,*

	conv_loss߭m><�3        )��P	�����A�,*

	conv_losss�h>
�(        )��P	JF����A�,*

	conv_loss�|o>{��3        )��P	_x����A�,*

	conv_lossd~m>��        )��P	������A�,*

	conv_loss�s>��S/        )��P	�؝���A�,*

	conv_loss(t>R)B        )��P	
����A�,*

	conv_loss�Om>eU�g        )��P	;����A�,*

	conv_loss��h>��2        )��P	�j����A�,*

	conv_loss_�o>�?�a        )��P	˛����A�,*

	conv_loss�d>�{�        )��P	t����A�,*

	conv_lossxwp>��?        )��P	����A�,*

	conv_lossđf>��        )��P	�B����A�,*

	conv_loss���>�K
        )��P	;x����A�,*

	conv_loss�0q>��.D        )��P	������A�,*

	conv_loss$�~>K+��        )��P	mڟ���A�,*

	conv_loss��t>%��        )��P	1����A�,*

	conv_loss(9s>��}        )��P	?;����A�,*

	conv_loss�h>���        )��P	z����A�,*

	conv_lossZMq>iV�        )��P	(�����A�-*

	conv_loss�{n>�Y26        )��P	�����A�-*

	conv_loss��e>%̜�        )��P	�����A�-*

	conv_loss"�t>����        )��P	�H����A�-*

	conv_lossO�p>)��y        )��P	$z����A�-*

	conv_loss!Op>pv��        )��P	������A�-*

	conv_loss�s> sr�        )��P	�����A�-*

	conv_loss
u>�a��        )��P	)����A�-*

	conv_loss1%l>!�*_        )��P	\h����A�-*

	conv_loss}��><$��        )��P	,�����A�-*

	conv_loss�z>��P        )��P	kҢ���A�-*

	conv_loss�Bt>Q��        )��P	�����A�-*

	conv_loss�zd>�H@�        )��P	#8����A�-*

	conv_loss�.m>�1��        )��P	�h����A�-*

	conv_loss�.z>dt�        )��P	������A�-*

	conv_loss,�t>�0(        )��P	%ӣ���A�-*

	conv_lossJ�z>v��        )��P	x����A�-*

	conv_loss��l>OEI�        )��P	�<����A�-*

	conv_lossx+o>S��        )��P	'w����A�-*

	conv_lossUp>o        )��P	I�����A�-*

	conv_loss�}>%9�r        )��P	�ݤ���A�-*

	conv_loss�q>���e        )��P	�����A�-*

	conv_loss�ge>��.�        )��P	>����A�-*

	conv_lossWl>8��        )��P	|m����A�-*

	conv_lossL(r>��Z�        )��P	䠥���A�-*

	conv_loss��t>;��~        )��P	�ѥ���A�-*

	conv_loss��{>�6��        )��P	�����A�-*

	conv_loss8�m>�2��        )��P	&2����A�-*

	conv_loss�Cs>��h�        )��P	0c����A�-*

	conv_loss�l>�GJ)        )��P	������A�-*

	conv_loss�n>6�8�        )��P	8Ħ���A�-*

	conv_loss��k>:�        )��P	G�����A�-*

	conv_loss�z>�2�        )��P	e$����A�-*

	conv_loss�xj>	���        )��P	�R����A�-*

	conv_loss�6q>�=        )��P	/�����A�-*

	conv_loss�In>���_        )��P	������A�-*

	conv_loss�L�>�?�v        )��P	����A�-*

	conv_loss�]j>i�<        )��P	b����A�-*

	conv_loss7%`>�k05        )��P	nE����A�-*

	conv_loss�(g>9�|        )��P	�t����A�-*

	conv_loss��d>�=�        )��P	 �����A�-*

	conv_loss�Lf>�+�o        )��P	�<����A�-*

	conv_loss�|m>J�RQ        )��P	dm����A�-*

	conv_loss`p>79Ae        )��P	U�����A�-*

	conv_loss_Rz>��1        )��P	�ʪ���A�-*

	conv_loss�k>�8=        )��P	�����A�-*

	conv_loss�zn>��y        )��P	6-����A�-*

	conv_loss�\t>=��O        )��P	�[����A�-*

	conv_loss�yl>X6F
        )��P	������A�-*

	conv_loss)v>1�{        )��P	������A�-*

	conv_loss:ds>=e{        )��P	l����A�-*

	conv_lossa4c>���        )��P	�6����A�-*

	conv_lossZx>X�]?        )��P	�e����A�-*

	conv_loss�@m>l;Tm        )��P	͔����A�-*

	conv_loss=g>�Q��        )��P	�Ĭ���A�-*

	conv_loss��h>�m��        )��P		����A�-*

	conv_loss�h>)L        )��P	*����A�-*

	conv_lossa�s>��Y         )��P	�X����A�-*

	conv_loss�Zd>a�"        )��P	�����A�-*

	conv_loss��n>��a%        )��P	8�����A�-*

	conv_losssGb>@K��        )��P	�����A�-*

	conv_loss4�i>Fq�$        )��P	|����A�-*

	conv_lossps>˯W�        )��P	�L����A�-*

	conv_loss�}i>P��        )��P	f�����A�-*

	conv_lossB�i>;�5�        )��P	�����A�-*

	conv_loss�Hh>��/<        )��P	9����A�-*

	conv_loss�pg>��M        )��P	j����A�-*

	conv_loss�r>�4��        )��P	F����A�-*

	conv_loss�o>�[��        )��P	@u����A�-*

	conv_loss�g>@�        )��P	դ����A�-*

	conv_lossT�c>N!��        )��P	�ӯ���A�-*

	conv_loss�\o>x�n�        )��P	����A�-*

	conv_loss��p>8��6        )��P	?6����A�-*

	conv_loss�Fi> �        )��P	d����A�-*

	conv_lossm�n>T�M�        )��P	�����A�-*

	conv_loss�Dh>P��c        )��P	9°���A�-*

	conv_loss��r>c��D        )��P	d����A�-*

	conv_loss�v>��E        )��P	�����A�-*

	conv_loss�wl>΂;�        )��P	>N����A�-*

	conv_loss�)q>´�        )��P	`|����A�-*

	conv_loss&�n>���        )��P	�����A�-*

	conv_lossu�r>,��        )��P	�ڱ���A�-*

	conv_loss��b>��#�        )��P	_
����A�-*

	conv_loss�o>�@"�        )��P	D;����A�-*

	conv_loss�_>���        )��P	�h����A�-*

	conv_loss�-d>��h        )��P	~�����A�-*

	conv_loss>d>H 1'        )��P	Ʋ���A�-*

	conv_loss�p>hܮ        )��P	������A�-*

	conv_lossshw>!�E�        )��P	<+����A�-*

	conv_loss�}p> �         )��P	�Y����A�-*

	conv_loss��d>b*��        )��P	ꌳ���A�-*

	conv_loss%r>�0wq        )��P	`г���A�-*

	conv_loss	�s>��Ӽ        )��P	�����A�-*

	conv_loss�.c>�3�        )��P	m3����A�-*

	conv_loss��m>���        )��P	�f����A�-*

	conv_lossҿl>"%^        )��P	������A�-*

	conv_loss?�j>��a�        )��P	tǴ���A�-*

	conv_loss��p>�        )��P	������A�-*

	conv_loss�m>��O6        )��P	7����A�-*

	conv_lossЯs>��u        )��P	�m����A�-*

	conv_loss�o>�?         )��P	
�����A�-*

	conv_loss�;`>�=        )��P	�����A�-*

	conv_loss��m>��-�        )��P	�����A�-*

	conv_loss�b>y�%R        )��P	_����A�-*

	conv_loss�j>?@��        )��P	�����A�-*

	conv_loss�ck>d��x        )��P	1ж���A�-*

	conv_lossuj>��1�        )��P	G����A�-*

	conv_loss�Lm>�U        )��P	5����A�-*

	conv_loss��e>D;�        )��P	�l����A�-*

	conv_lossN�g> �V        )��P	R�����A�-*

	conv_loss�e>��        )��P	̷���A�-*

	conv_loss� h>T�;        )��P	������A�-*

	conv_loss=�o>�\�        )��P	�)����A�-*

	conv_lossRkh>ӋY�        )��P	�_����A�-*

	conv_lossF�b>�Rx        )��P	ו����A�-*

	conv_losssn>�C��        )��P	Tȸ���A�-*

	conv_lossdUd>Iq}�        )��P	�����A�-*

	conv_lossT!_>�&�        )��P	s)����A�-*

	conv_loss?�e>���        )��P	�X����A�-*

	conv_loss_�e>*�        )��P	���A�-*

	conv_loss\�f>��s        )��P	O�����A�-*

	conv_lossG�n>`O�        )��P	�����A�-*

	conv_loss�7m>�2        )��P	`����A�-*

	conv_loss��m>�p        )��P	 K����A�-*

	conv_lossrfr>P��        )��P	z����A�-*

	conv_loss��e>=���        )��P	ϫ����A�-*

	conv_loss5�i>�%#�        )��P	(ܺ���A�-*

	conv_loss��r>}��        )��P	�
����A�-*

	conv_losso�f>t|�        )��P	U:����A�.*

	conv_loss,[j>T�:�        )��P	�j����A�.*

	conv_loss�YZ>bNm1        )��P	ę����A�.*

	conv_loss�d>a��        )��P	�Ȼ���A�.*

	conv_loss�m>�i��        )��P	`�����A�.*

	conv_loss�^e>�� f        )��P	�&����A�.*

	conv_lossV�h>����        )��P	�U����A�.*

	conv_loss��l>ł`�        )��P	�����A�.*

	conv_lossI�h>ܠm�        )��P	W�����A�.*

	conv_loss��i>�.        )��P	:����A�.*

	conv_loss�l>_;7        )��P	&����A�.*

	conv_loss�%o>��8�        )��P	�M����A�.*

	conv_loss��p>恲n        )��P	~����A�.*

	conv_lossrR>L'��        )��P	������A�.*

	conv_lossf�d>^��8        )��P	q����A�.*

	conv_loss��j>��:�        )��P	�#����A�.*

	conv_lossnLe>��L        )��P	�X����A�.*

	conv_loss�`>3�,        )��P	������A�.*

	conv_lossb�e>���t        )��P	�ž���A�.*

	conv_loss8�l>Jbf�        )��P	������A�.*

	conv_lossp�o>E��        )��P	2����A�.*

	conv_loss�b>A���        )��P	j����A�.*

	conv_lossz�p>�F        )��P	�����A�.*

	conv_loss�s>	[M�        )��P	�ܿ���A�.*

	conv_lossN}g>5���        )��P	N����A�.*

	conv_loss�d>�)��        )��P	�J����A�.*

	conv_loss"�m>��n�        )��P	�|����A�.*

	conv_loss��m>ZL�        )��P	������A�.*

	conv_lossfa>�a�        )��P	c�����A�.*

	conv_loss��^>[�        )��P	�����A�.*

	conv_lossNa>!���        )��P	tO����A�.*

	conv_loss�*c>I���        )��P	������A�.*

	conv_loss�ic>	Z�        )��P	+�����A�.*

	conv_loss�g>�P�        )��P	
�����A�.*

	conv_loss�p>_:        )��P	����A�.*

	conv_lossc�q>L�[�        )��P	G����A�.*

	conv_loss��e>��O        )��P	�v����A�.*

	conv_lossc�i>gb�        )��P	<�����A�.*

	conv_loss8�n>C���        )��P	������A�.*

	conv_loss_(e>�[C�        )��P	V����A�.*

	conv_loss��[>���w        )��P	�6����A�.*

	conv_loss�c>�3�        )��P	g����A�.*

	conv_losso�f> &x�        )��P	ۗ����A�.*

	conv_lossڊf>K�W        )��P	������A�.*

	conv_lossXWb>;�%�        )��P	_�����A�.*

	conv_loss�~c>��o        )��P	,����A�.*

	conv_lossQ#o>� �        )��P	�\����A�.*

	conv_loss��b>�j+        )��P	!�����A�.*

	conv_lossi�a>O���        )��P	S�����A�.*

	conv_lossAvd>��C�        )��P	������A�.*

	conv_loss�B]>Qw�        )��P	�����A�.*

	conv_loss	'i>�2L        )��P	+Q����A�.*

	conv_loss
bm>��Ժ        )��P	������A�.*

	conv_loss�^q>�52        )��P	ж����A�.*

	conv_loss'H[>BU        )��P	������A�.*

	conv_loss��j>��t        )��P	�����A�.*

	conv_loss!Ea>1W�        )��P	�K����A�.*

	conv_loss]Z>��A�        )��P	4�����A�.*

	conv_loss@�d>	�t�        )��P	������A�.*

	conv_lossݕb>@"O�        )��P	U�����A�.*

	conv_loss�pn>���X        )��P	E����A�.*

	conv_loss��_>����        )��P	wC����A�.*

	conv_loss2�p>�;&        )��P	�u����A�.*

	conv_loss/@U>=��        )��P	=�����A�.*

	conv_loss �g>}�        )��P	�����A�.*

	conv_loss A[>�!\�        )��P	U"����A�.*

	conv_loss�wX>�19�        )��P	7m����A�.*

	conv_loss�#]>����        )��P	e�����A�.*

	conv_loss�xe>�U�+        )��P	������A�.*

	conv_lossB�e>-8@�        )��P	d����A�.*

	conv_loss�w>�Ig�        )��P	(B����A�.*

	conv_loss�A\>��        )��P	Tu����A�.*

	conv_lossh0c>���j        )��P	������A�.*

	conv_lossǗg>��D        )��P	n�����A�.*

	conv_lossh�^>K��        )��P	 ����A�.*

	conv_loss��`>��4.        )��P	PR����A�.*

	conv_loss�V>�DO�        )��P	v�����A�.*

	conv_loss�Mh>��r�        )��P	Q�����A�.*

	conv_lossLpp>���        )��P	�����A�.*

	conv_loss��i>
�"        )��P	�����A�.*

	conv_loss�<f>���        )��P	(L����A�.*

	conv_loss�
^>�s        )��P	
}����A�.*

	conv_lossZ�m>���        )��P	P�����A�.*

	conv_loss^e>�K4        )��P	+�����A�.*

	conv_loss�i>��.        )��P	E����A�.*

	conv_loss"�]>:Fw�        )��P	�B����A�.*

	conv_loss"?W>��i        )��P	=t����A�.*

	conv_loss��W>�nU�        )��P	R�����A�.*

	conv_loss��[>�~�        )��P	�����A�.*

	conv_loss`�b>U���        )��P	a
����A�.*

	conv_lossdPa>�r        )��P	0:����A�.*

	conv_lossA*_>Ԥ��        )��P	�j����A�.*

	conv_loss4b>\���        )��P	������A�.*

	conv_loss�~Q>��        )��P	������A�.*

	conv_loss)�d>��=P        )��P	t ����A�.*

	conv_loss�u{>�l�        )��P	y0����A�.*

	conv_loss?vd>|x��        )��P	hc����A�.*

	conv_loss�t_>����        )��P	ϕ����A�.*

	conv_lossS�c>nAi*        )��P	y�����A�.*

	conv_lossuIc>}��N        )��P	������A�.*

	conv_lossg`>����        )��P	f*����A�.*

	conv_loss�@W>`6*�        )��P	&[����A�.*

	conv_lossk[>N��}        )��P	�����A�.*

	conv_loss��g>{V3�        )��P	�����A�.*

	conv_loss�`>@<6        )��P	������A�.*

	conv_loss^k>G�	        )��P	�!����A�.*

	conv_lossb�[>�H�        )��P	�V����A�.*

	conv_loss?�S>D��        )��P	(�����A�.*

	conv_loss��a>�bbd        )��P	m�����A�.*

	conv_loss�Y>;ơ        )��P	+�����A�.*

	conv_loss�c]>��\�        )��P	�����A�.*

	conv_lossu�l>�.�        )��P	�O����A�.*

	conv_lossI�Y>/�@