       �K"	  �����Abrain.Event:2"c7?�      q� �	�韉��A"��
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
_class
loc:@conv2d/kernel*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
:*

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 *
dtype0
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
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
T0
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
.conv2d_2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *��:>*
dtype0*
_output_shapes
: 
�
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0
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
conv2d_4/Conv2DConv2DRelu_2conv2d_3/kernel/read*
strides
*
data_formatNHWC*
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
dtype0*
_output_shapes
:*
valueB"      
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
^
Reshape/shapeConst*
valueB"����P  *
dtype0*
_output_shapes
:
j
ReshapeReshapeRelu_4Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"P  d   *
dtype0*
_output_shapes
:
�
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *>�*
dtype0*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *>=*
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
:	�d
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
:	�d*
T0*
_class
loc:@dense/kernel
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
:	�d*
T0
�
dense/kernel
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	�d
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d
�
dense/MatMulMatMulReshapedense/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
N
Relu_5Reludense/MatMul*
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
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

�
dense_2/MatMulMatMulRelu_5dense_1/kernel/read*
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
gradients/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
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
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
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
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
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
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select*'
_output_shapes
:���������
*
T0
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
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_57gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
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
gradients/Relu_5_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_5*
T0*'
_output_shapes
:���������d
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_5_grad/ReluGraddense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/Relu_5_grad/ReluGrad*
T0*
_output_shapes
:	�d*
transpose_a(*
transpose_b( 
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�d
b
gradients/Reshape_grad/ShapeShapeRelu_4*
out_type0*
_output_shapes
:*
T0
�
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
gradients/Relu_4_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_4*
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
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�d*
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
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( 
�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_3/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_4/kernel/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
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
: "ʊ|+�      ���	GMꟉ��AJ��
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
Ttype*1.4.12v1.4.0-19-ga52c8d9��
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
,conv2d/kernel/Initializer/random_uniform/minConst*
_output_shapes
: * 
_class
loc:@conv2d/kernel*
valueB
 *����*
dtype0
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
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
conv2d_2/kernel
VariableV2*"
_class
loc:@conv2d_2/kernel*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
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
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
T0
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
Relu_4Reluconv2d_5/Conv2D*/
_output_shapes
:���������*
T0
^
Reshape/shapeConst*
valueB"����P  *
dtype0*
_output_shapes
:
j
ReshapeReshapeRelu_4Reshape/shape*
Tshape0*(
_output_shapes
:����������*
T0
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"P  d   *
dtype0*
_output_shapes
:
�
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *>�*
dtype0*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *>=*
dtype0*
_output_shapes
: 
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	�d*

seed *
T0*
_class
loc:@dense/kernel
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
:	�d
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d
�
dense/kernel
VariableV2*
_class
loc:@dense/kernel*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d*
shared_name 
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*
_class
loc:@dense/kernel
v
dense/kernel/readIdentitydense/kernel*
_output_shapes
:	�d*
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
Relu_5Reludense/MatMul*
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
dense_2/MatMulMatMulRelu_5dense_1/kernel/read*'
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
dtype0*
_output_shapes
:*
valueB"       
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
: *
	keep_dims( *

Tidx0*
T0
�
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0
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
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
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
:*
	keep_dims( *

Tidx0
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
$gradients/logistic_loss/mul_grad/mulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Placeholder_1*
T0*'
_output_shapes
:���������

�
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
Tshape0*'
_output_shapes
:���������
*
T0
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
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_57gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
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
gradients/Relu_5_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_5*'
_output_shapes
:���������d*
T0
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_5_grad/ReluGraddense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/Relu_5_grad/ReluGrad*
transpose_b( *
T0*
_output_shapes
:	�d*
transpose_a(
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�d
b
gradients/Reshape_grad/ShapeShapeRelu_4*
T0*
out_type0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:���������*
T0
�
gradients/Relu_4_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_4*
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
7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_5/Conv2D_grad/tuple/group_deps*E
_class;
97loc:@gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������*
T0
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
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read* 
_output_shapes
::*
T0*
out_type0*
N
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
gradients/Relu_grad/ReluGradReluGrad7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyRelu*/
_output_shapes
:���������*
T0
�
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read*
out_type0*
N* 
_output_shapes
::*
T0
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
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
use_locking( *
T0
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d
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
�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_3/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_4/kernel/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
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
	summaries

conv_loss:0"�
trainable_variables��
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
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"
train_op

GradientDescent"�
	variables��
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
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0�3Q�       `/�#	@�����A*

	conv_loss+�1?G�3M       QKD	������A*

	conv_loss�v1?��A       QKD	n�����A*

	conv_loss��1?Ne#        QKD	������A*

	conv_loss)�1?M�r�       QKD	~����A*

	conv_loss�n1?��|�       QKD	�-����A*

	conv_loss${1?�bj       QKD	\V����A*

	conv_lossT1?L���       QKD	�~����A*

	conv_loss@b1?��P       QKD	������A*

	conv_loss�r1?"��       QKD	�����A	*

	conv_lossa{1?����       QKD	�����A
*

	conv_loss��1?��ŧ       QKD	
3����A*

	conv_loss/C1?�� �       QKD	q����A*

	conv_loss�[1?ӥ@�       QKD	B�����A*

	conv_lossLz1?%�
n       QKD	L�����A*

	conv_loss�G1?�^�       QKD	������A*

	conv_loss�P1?+�S0       QKD	�����A*

	conv_lossbi1?� �       QKD	�I����A*

	conv_lossc1?��z       QKD	�~����A*

	conv_lossB41?��       QKD	W�����A*

	conv_loss
a1?msv       QKD	U�����A*

	conv_loss�-1?��$�       QKD	k�����A*

	conv_loss�C1?�ۮ�       QKD	X$����A*

	conv_loss3-1?�<��       QKD	oK����A*

	conv_lossK1?�K�&       QKD	�s����A*

	conv_loss�1?>�       QKD	|�����A*

	conv_lossK1?�#�       QKD	������A*

	conv_loss�1?8E�       QKD	
�����A*

	conv_loss�1?�:�L       QKD	"$����A*

	conv_loss1?��AM       QKD	�N����A*

	conv_loss�1?�{        QKD	�x����A*

	conv_loss�I1? ]�       QKD	������A*

	conv_loss.�0?���       QKD	t�����A *

	conv_loss 1?�a       QKD	������A!*

	conv_loss�41?m|�       QKD	w����A"*

	conv_loss�%1?w�&       QKD	�E����A#*

	conv_loss1?��|w       QKD	�m����A$*

	conv_loss�	1?\�
	       QKD	������A%*

	conv_loss��0?$���       QKD	������A&*

	conv_lossA1??���       QKD	A�����A'*

	conv_loss��0?�ȇ-       QKD	�����A(*

	conv_lossv�0?{$�       QKD	!=����A)*

	conv_lossp�0?ȳfw       QKD	1f����A**

	conv_loss��0?X~��       QKD	������A+*

	conv_loss��0?�1\�       QKD	Ƿ����A,*

	conv_loss��0?|��,       QKD	>�����A-*

	conv_lossu�0?/	�_       QKD	�	 ����A.*

	conv_loss?�0?����       QKD	\1 ����A/*

	conv_loss��0?��YR       QKD	�\ ����A0*

	conv_loss�0?k��0       QKD	(� ����A1*

	conv_loss��0?�ʄ�       QKD	�� ����A2*

	conv_lossG�0?���       QKD	�� ����A3*

	conv_loss��0?6��Q       QKD	U!����A4*

	conv_loss��0?�|�?       QKD	�;!����A5*

	conv_loss}�0?�-H-       QKD	�b!����A6*

	conv_loss�0?o���       QKD	��!����A7*

	conv_lossE�0?4,�       QKD	n�!����A8*

	conv_lossJ�0?̤�U       QKD	��!����A9*

	conv_loss=�0?^yr�       QKD	;"����A:*

	conv_loss��0?����       QKD	H"����A;*

	conv_loss�0?����       QKD	�p"����A<*

	conv_loss>�0?����       QKD	=�"����A=*

	conv_loss/�0?q�Ds       QKD	��"����A>*

	conv_loss��0?k�y�       QKD	F�"����A?*

	conv_loss��0?��+       QKD	&&#����A@*

	conv_loss��0?d�)]       QKD	�O#����AA*

	conv_loss[�0?l�	       QKD	py#����AB*

	conv_loss�0?[5�+       QKD	��#����AC*

	conv_loss��0?�m�       QKD	Q�#����AD*

	conv_lossҖ0?���       QKD	T$����AE*

	conv_loss�0?�,�       QKD	�,$����AF*

	conv_loss/�0?� �       QKD	T$����AG*

	conv_loss��0?fI��       QKD	s~$����AH*

	conv_lossc�0?��ѣ       QKD	6�$����AI*

	conv_lossji0?Ԥ��       QKD	p�$����AJ*

	conv_loss�~0?�}?       QKD	��$����AK*

	conv_losso{0?���n       QKD	e)%����AL*

	conv_lossl�0?��Km       QKD	T%����AM*

	conv_loss�0?��@Z       QKD	��%����AN*

	conv_lossY]0?^�	       QKD	W�%����AO*

	conv_loss�s0?U�w       QKD	�%����AP*

	conv_loss*`0?J=�T       QKD	3�%����AQ*

	conv_lossxe0?Z��9       QKD	a$&����AR*

	conv_loss��0?<���       QKD	�M&����AS*

	conv_loss�H0?��U       QKD	w&����AT*

	conv_loss�W0?����       QKD	��&����AU*

	conv_lossB0?��^�       QKD	��&����AV*

	conv_loss@Q0?��;Z       QKD	��&����AW*

	conv_loss�c0?~��       QKD	�'����AX*

	conv_lossjN0?:9       QKD	�D'����AY*

	conv_loss#E0?.Y9�       QKD	�m'����AZ*

	conv_loss�;0?f_}�       QKD	��'����A[*

	conv_loss�I0?�B       QKD	��'����A\*

	conv_loss�K0?�I1�       QKD	��'����A]*

	conv_loss�]0?���       QKD	&(����A^*

	conv_lossH40?���e       QKD	�<(����A_*

	conv_loss�Q0?����       QKD	�e(����A`*

	conv_lossfO0?�zf       QKD	��(����Aa*

	conv_loss\?0?�h       QKD	۶(����Ab*

	conv_loss>#0?�S@	       QKD	��(����Ac*

	conv_loss
0?��α       QKD	�)����Ad*

	conv_loss�E0?�%<�       QKD	�C)����Ae*

	conv_lossf?0?w�D       QKD	 l)����Af*

	conv_lossb!0?���       QKD	w�)����Ag*

	conv_loss�0?���       QKD	�)����Ah*

	conv_loss�0?��K�       QKD	��)����Ai*

	conv_loss�0?�|=       QKD	�*����Aj*

	conv_loss$0?�V�0       QKD	�;*����Ak*

	conv_loss8�/?�1�1       QKD	�d*����Al*

	conv_loss�/?�=��       QKD	�*����Am*

	conv_lossP�/?��       QKD	S�*����An*

	conv_losse�/?��T�       QKD	�*����Ao*

	conv_loss�0?���8       QKD	R+����Ap*

	conv_loss`�/?8�8�       QKD	�F+����Aq*

	conv_loss/0?�ؕ.       QKD	�q+����Ar*

	conv_loss��/?	�       QKD	K�+����As*

	conv_loss��/?㯷G       QKD	��+����At*

	conv_loss�/?�!U�       QKD	��+����Au*

	conv_loss��/?�>�       QKD	\,,����Av*

	conv_loss(�/?z��9       QKD	7],����Aw*

	conv_loss4�/?}!S       QKD	ɋ,����Ax*

	conv_loss� 0?ؽC       QKD	��,����Ay*

	conv_lossb�/?v��4       QKD	`�,����Az*

	conv_lossV 0?i
7�       QKD	�-����A{*

	conv_loss��/?T�0       QKD	:-����A|*

	conv_lossm�/?ÿSD       QKD	�g-����A}*

	conv_loss��/?,��       QKD	j�-����A~*

	conv_loss��/?+QC�       QKD	s�-����A*

	conv_loss��/?Q�D        )��P	u�-����A�*

	conv_loss��/?p!<        )��P	R.����A�*

	conv_loss��/?��hx        )��P	�=.����A�*

	conv_loss�/?�V΅        )��P	�k.����A�*

	conv_loss��/?��ë        )��P	G�.����A�*

	conv_lossʰ/?��
�        )��P	��.����A�*

	conv_loss��/?c��v        )��P	�.����A�*

	conv_lossw�/?{q�e        )��P	\/����A�*

	conv_loss'�/?��H        )��P	�C/����A�*

	conv_loss�/?��        )��P	fo/����A�*

	conv_lossɵ/?z(W�        )��P	ޘ/����A�*

	conv_lossщ/?"I#�        )��P	!�/����A�*

	conv_loss٥/?D�Z|        )��P	1�/����A�*

	conv_lossj/?Jڈ        )��P	10����A�*

	conv_loss�/?yTY�        )��P	�J0����A�*

	conv_loss��/?�.Y�        )��P	"v0����A�*

	conv_loss9�/?��3        )��P	Р0����A�*

	conv_loss��/?:	�Q        )��P	W�0����A�*

	conv_losst/?Z���        )��P	&�0����A�*

	conv_lossҐ/?7(�        )��P	� 1����A�*

	conv_loss~�/?��        )��P	�K1����A�*

	conv_loss��/?5�v        )��P	ou1����A�*

	conv_lossLz/?�!�        )��P	�1����A�*

	conv_loss`/?�]��        )��P	��1����A�*

	conv_loss�m/?8�vV        )��P	B2����A�*

	conv_lossjn/?iC�        )��P	�42����A�*

	conv_loss�n/?���        )��P	c2����A�*

	conv_loss1n/?�3�        )��P	��2����A�*

	conv_loss f/?�>8�        )��P	:�2����A�*

	conv_loss�Y/?���        )��P	�2����A�*

	conv_lossu/?�E�P        )��P	�3����A�*

	conv_loss�4/?���#        )��P	gB3����A�*

	conv_loss�\/?:�f        )��P	-x3����A�*

	conv_loss�I/?ޱ��        )��P	}�3����A�*

	conv_lossT:/?��!V        )��P	�3����A�*

	conv_loss�V/?�{��        )��P	34����A�*

	conv_loss�0/?o�-r        )��P	�B4����A�*

	conv_loss8#/?���        )��P	Im4����A�*

	conv_lossP/?��S�        )��P	w�4����A�*

	conv_loss�B/?s��        )��P	��4����A�*

	conv_lossYF/?�J�        )��P	�4����A�*

	conv_lossC/?Ta P        )��P	|5����A�*

	conv_loss�?/?�q
�        )��P	�I5����A�*

	conv_loss1/?S$�        )��P	�t5����A�*

	conv_loss�7/?z	]�        )��P	:�5����A�*

	conv_loss�/?8N�m        )��P	`�5����A�*

	conv_loss�/?�o�        )��P	-�5����A�*

	conv_loss/:/?����        )��P	�(6����A�*

	conv_lossB/? ��        )��P	�Y6����A�*

	conv_loss�%/?H?�        )��P	�6����A�*

	conv_loss\!/?�Ȭ�        )��P	�6����A�*

	conv_lossr)/?���        )��P	��6����A�*

	conv_loss�/?�=Y        )��P	a7����A�*

	conv_loss"	/?XՌ�        )��P	XF7����A�*

	conv_loss��.?�        )��P	/p7����A�*

	conv_lossY/?<���        )��P	��7����A�*

	conv_loss"/?\i�_        )��P	~�7����A�*

	conv_loss��.?�M:�        )��P	7�7����A�*

	conv_loss��.?�Z�j        )��P	�8����A�*

	conv_loss��.?�%k        )��P	K8����A�*

	conv_loss|
/?�4�        )��P	wu8����A�*

	conv_loss��.?��`�        )��P	Π8����A�*

	conv_loss%�.?��'�        )��P	O�8����A�*

	conv_loss��.?/�'        )��P	��8����A�*

	conv_loss��.?(�A        )��P	#9����A�*

	conv_loss�.?����        )��P	�L9����A�*

	conv_loss8�.?��f        )��P	#w9����A�*

	conv_loss��.?����        )��P	��9����A�*

	conv_lossc�.?��g        )��P	��9����A�*

	conv_loss�.?ǃ��        )��P	��9����A�*

	conv_lossq�.?��e�        )��P	w&:����A�*

	conv_loss�.?v��         )��P	_P:����A�*

	conv_lossӾ.?�|        )��P	�|:����A�*

	conv_loss
�.?3)EW        )��P	C�;����A�*

	conv_loss�.?�.G        )��P	�<����A�*

	conv_loss��.?l:�        )��P	V?<����A�*

	conv_loss��.?�P�n        )��P	ui<����A�*

	conv_lossئ.?�܌g        )��P	��<����A�*

	conv_lossP�.?�U�        )��P	W�<����A�*

	conv_loss��.?J        )��P	l�<����A�*

	conv_losst�.?G<w&        )��P	�$=����A�*

	conv_loss �.?;��v        )��P	�P=����A�*

	conv_loss�.?��        )��P	ފ=����A�*

	conv_lossj�.?�G��        )��P	¾=����A�*

	conv_lossJ�.?8��        )��P	�=����A�*

	conv_loss�}.?�?�        )��P	+1>����A�*

	conv_loss��.?p#�        )��P	a>����A�*

	conv_loss��.?��6        )��P	��>����A�*

	conv_lossޓ.?*Gˑ        )��P	�>����A�*

	conv_loss��.?���        )��P	[�>����A�*

	conv_loss��.?]~:Z        )��P	�?����A�*

	conv_lossZu.?Ԥ�        )��P	sD?����A�*

	conv_loss�{.?Ks��        )��P	�n?����A�*

	conv_loss�t.?5^ �        )��P	�?����A�*

	conv_lossn�.?d�9�        )��P	��?����A�*

	conv_loss�s.?s�,+        )��P	��?����A�*

	conv_lossh�.?��&�        )��P	4@����A�*

	conv_loss&].?���{        )��P	�P@����A�*

	conv_loss�Q.?���        )��P	�@����A�*

	conv_lossXs.?�"�p        )��P	,�@����A�*

	conv_loss��.?���        )��P	@�@����A�*

	conv_loss�y.?S�Y�        )��P	�A����A�*

	conv_loss$m.?�S�        )��P	:<A����A�*

	conv_loss�R.?m�-�        )��P	�iA����A�*

	conv_loss�N.??��D        )��P	>�A����A�*

	conv_loss�s.?��N        )��P	�A����A�*

	conv_loss�Q.?��(N        )��P	�A����A�*

	conv_loss�L.?�Q'        )��P	�B����A�*

	conv_lossx.?�Ħi        )��P	�FB����A�*

	conv_loss�Z.?�]�        )��P	�qB����A�*

	conv_loss�.?�g�s        )��P	��B����A�*

	conv_loss�;.?eUlB        )��P	t�B����A�*

	conv_lossi.?@}�c        )��P	��B����A�*

	conv_loss� .?LD�        )��P	"C����A�*

	conv_loss�;.?%���        )��P	�MC����A�*

	conv_loss$.?WF��        )��P	�{C����A�*

	conv_lossb*.?I^�1        )��P	��C����A�*

	conv_loss�.?�6        )��P	��C����A�*

	conv_loss� .?R�F�        )��P	��C����A�*

	conv_loss".?��u�        )��P	�'D����A�*

	conv_loss�.?-�[�        )��P	KSD����A�*

	conv_loss�&.?�B��        )��P	WD����A�*

	conv_loss�.?6M$�        )��P	��D����A�*

	conv_loss��-?�c�B        )��P	��D����A�*

	conv_lossd�-?�1$        )��P	TE����A�*

	conv_lossR�-?��8        )��P	�<E����A�*

	conv_lossL�-?� �        )��P	�gE����A�*

	conv_loss��-?|R�        )��P	��E����A�*

	conv_loss.?!��        )��P	��E����A�*

	conv_loss�.?o�^�        )��P	�E����A�*

	conv_losss�-?�֭�        )��P	�$F����A�*

	conv_loss��-?���        )��P	aQF����A�*

	conv_loss.�-?s&J�        )��P	��F����A�*

	conv_loss��-?g �        )��P	d�F����A�*

	conv_lossu�-?%"8%        )��P	��F����A�*

	conv_loss[�-?Alg|        )��P	�G����A�*

	conv_loss0�-?����        )��P	�:G����A�*

	conv_loss��-?-[.$        )��P	�gG����A�*

	conv_lossW�-?�?��        )��P	�G����A�*

	conv_loss��-?I��T        )��P	��G����A�*

	conv_loss�-?���(        )��P	��G����A�*

	conv_lossQz-?�J        )��P	�.H����A�*

	conv_loss��-?Q���        )��P	XZH����A�*

	conv_loss��-?ź�        )��P	/�H����A�*

	conv_lossi�-?�c��        )��P	%�H����A�*

	conv_loss��-?&�M�        )��P	��H����A�*

	conv_lossy�-?+1        )��P	@I����A�*

	conv_lossf�-?o�Y�        )��P	�;I����A�*

	conv_lossS�-?��        )��P	�dI����A�*

	conv_lossv�-?�v�        )��P	�I����A�*

	conv_loss�-?�+�        )��P	g�I����A�*

	conv_loss"�-?a�+a        )��P	,�I����A�*

	conv_lossԢ-?9�i        )��P	ZJ����A�*

	conv_loss��-?�MS        )��P	�?J����A�*

	conv_loss��-?.�5�        )��P	�iJ����A�*

	conv_loss9�-?ݝ
        )��P	��J����A�*

	conv_lossT�-?�/`R        )��P	+�J����A�*

	conv_loss|-?����        )��P	;�J����A�*

	conv_lossT�-?>~00        )��P	�K����A�*

	conv_loss��-?��V        )��P	fGK����A�*

	conv_lossS�-?��        )��P	�tK����A�*

	conv_loss��-?���        )��P	B�K����A�*

	conv_loss?y-?"��O        )��P	��K����A�*

	conv_loss�d-?غ�!        )��P	��K����A�*

	conv_loss_t-?�}        )��P	)!L����A�*

	conv_loss:j-?�Ӎ+        )��P	KL����A�*

	conv_loss��-?�2�K        )��P	-xL����A�*

	conv_loss�V-?��        )��P	4�L����A�*

	conv_loss�r-?jS��        )��P	��L����A�*

	conv_loss�b-?���        )��P	��L����A�*

	conv_loss�l-?,+ײ        )��P	�"M����A�*

	conv_lossHF-?�U�        )��P	�MM����A�*

	conv_loss�<-?�_W�        )��P	�|M����A�*

	conv_loss\`-?k~�l        )��P	�M����A�*

	conv_lossXj-?��^        )��P	j�M����A�*

	conv_lossn@-?�#�]        )��P	hN����A�*

	conv_loss�K-?J��        )��P	A:N����A�*

	conv_lossTG-?��b�        )��P	hN����A�*

	conv_lossP^-? ~�s        )��P	ޒN����A�*

	conv_loss)B-?���        )��P	��N����A�*

	conv_loss�0-??�F}        )��P	��N����A�*

	conv_loss�-?����        )��P	.O����A�*

	conv_loss�:-?<� G        )��P	�GO����A�*

	conv_loss:-?��        )��P	�}O����A�*

	conv_loss�-?���^        )��P	U�O����A�*

	conv_loss4-?>�p&        )��P	��O����A�*

	conv_loss-?Q�E        )��P	yP����A�*

	conv_losst -?ɱ�3        )��P	�@P����A�*

	conv_loss�-?���q        )��P	�kP����A�*

	conv_loss{-?.ͪV        )��P	~�P����A�*

	conv_loss�-?��        )��P	�P����A�*

	conv_lossB-?�8�        )��P	^�P����A�*

	conv_loss�
-? ,�        )��P	Q����A�*

	conv_loss��,?�="Q        )��P	EFQ����A�*

	conv_loss -?�n0�        )��P	�pQ����A�*

	conv_lossE�,?�Q��        )��P	��Q����A�*

	conv_loss��,?v���        )��P	�Q����A�*

	conv_lossJ-?�n��        )��P	W�Q����A�*

	conv_loss!�,?\�+        )��P	�'R����A�*

	conv_loss��,?P7�s        )��P	]VR����A�*

	conv_loss	�,?E��        )��P	��R����A�*

	conv_loss��,?���8        )��P	��R����A�*

	conv_lossD�,?����        )��P	��R����A�*

	conv_loss��,?�g�        )��P	�S����A�*

	conv_loss:�,?|�O1        )��P	3S����A�*

	conv_loss��,?���        )��P	�]S����A�*

	conv_loss��,?����        )��P	��S����A�*

	conv_lossb�,?t���        )��P	/�S����A�*

	conv_loss��,?���        )��P	��S����A�*

	conv_loss��,?�^�        )��P	T����A�*

	conv_loss��,?yr�        )��P	�6T����A�*

	conv_loss@�,?dB!�        )��P	�bT����A�*

	conv_loss��,?g~�!        )��P	�T����A�*

	conv_loss��,?��t        )��P	ӺT����A�*

	conv_lossw�,?�� �        )��P	��T����A�*

	conv_loss��,?t�/�        )��P	�U����A�*

	conv_loss��,?�7$�        )��P	e?U����A�*

	conv_loss>�,?PL�        )��P	nkU����A�*

	conv_loss�,?Id�}        )��P	��U����A�*

	conv_lossό,?�h��        )��P	8�U����A�*

	conv_loss&�,?�9y        )��P	��U����A�*

	conv_loss,�,?�V��        )��P	JV����A�*

	conv_loss֑,?��ba        )��P	dGV����A�*

	conv_lossXm,?�g�        )��P	�V����A�*

	conv_loss��,?���/        )��P	/�V����A�*

	conv_lossX,?]�        )��P	z�V����A�*

	conv_loss�x,?���j        )��P	W����A�*

	conv_lossma,?�:��        )��P	X2W����A�*

	conv_lossa,?��*�        )��P	�^W����A�*

	conv_loss>M,?A�6�        )��P	�W����A�*

	conv_loss6N,?6^*�        )��P	�W����A�*

	conv_losslY,?�bυ        )��P	`�W����A�*

	conv_loss�K,?��o        )��P	�X����A�*

	conv_loss�V,?~uj?        )��P	�HX����A�*

	conv_loss�>,?��Ò        )��P	�tX����A�*

	conv_lossy<,?.�݂        )��P	�X����A�*

	conv_lossDS,?y�l        )��P	��X����A�*

	conv_loss(6,?"�i�        )��P	A�X����A�*

	conv_loss�A,?�~��        )��P	
'Y����A�*

	conv_loss{@,?Fa�        )��P	eXY����A�*

	conv_loss�H,?m=        )��P	�Y����A�*

	conv_loss#,?y���        )��P	�Y����A�*

	conv_loss�5,?m��        )��P	z�Y����A�*

	conv_loss�,?x�M        )��P	�Z����A�*

	conv_lossP9,?ob��        )��P	�FZ����A�*

	conv_loss��+?��7b        )��P	�rZ����A�*

	conv_loss�,?Y\w�        )��P	[�Z����A�*

	conv_loss �+?qj�        )��P	��Z����A�*

	conv_loss�,?��        )��P	��Z����A�*

	conv_loss�,?�w`�        )��P	�[����A�*

	conv_lossL�+?�O0        )��P	XX[����A�*

	conv_loss=,?��D�        )��P	j�[����A�*

	conv_loss�,?�+�        )��P	G�[����A�*

	conv_loss��+?+�~h        )��P	��[����A�*

	conv_loss��+?�T'�        )��P	W\����A�*

	conv_loss�,?s�ģ        )��P	�;\����A�*

	conv_loss(�+?�g��        )��P	ni\����A�*

	conv_loss\�+?D���        )��P	e�\����A�*

	conv_loss�+?YOB        )��P	��\����A�*

	conv_losss�+?1�$�        )��P	I�\����A�*

	conv_loss�,?�e�:        )��P	�]����A�*

	conv_loss1�+?��^@        )��P	�F]����A�*

	conv_lossQ�+?Tr}�        )��P	�r]����A�*

	conv_loss�+?<��        )��P	B�]����A�*

	conv_lossn�+?Q��        )��P	��]����A�*

	conv_loss�+?Y���        )��P	��]����A�*

	conv_loss�,?t�S        )��P	n#^����A�*

	conv_loss�+?.�        )��P	�O^����A�*

	conv_loss�+?q=&        )��P	7|^����A�*

	conv_lossx�+?±��        )��P	
�^����A�*

	conv_loss@�+?�iK:        )��P	�^����A�*

	conv_losst�+?����        )��P	6_����A�*

	conv_loss��+?�B*7        )��P	�-_����A�*

	conv_loss&�+?��%'        )��P	�`����A�*

	conv_lossr�+?�Ck�        )��P	f�`����A�*

	conv_loss�^+?$�2        )��P	j�`����A�*

	conv_loss��+?$���        )��P	5a����A�*

	conv_loss��+?�v��        )��P	�\a����A�*

	conv_loss�z+?�$��        )��P	��a����A�*

	conv_lossل+?z^�0        )��P	�a����A�*

	conv_loss�z+?�(�        )��P	�a����A�*

	conv_lossg+?7O�        )��P	Ob����A�*

	conv_loss6x+?���        )��P	K@b����A�*

	conv_lossLl+?�j�        )��P	�ob����A�*

	conv_loss[+?<�VF        )��P	��b����A�*

	conv_loss�|+?�Kwy        )��P	)�b����A�*

	conv_lossRO+?=
        )��P	��b����A�*

	conv_loss�8+?>{�        )��P	�!c����A�*

	conv_loss�[+?��*        )��P	�Pc����A�*

	conv_loss�f+?2�G        )��P	�|c����A�*

	conv_loss�w+?��~-        )��P	s�c����A�*

	conv_loss�=+?����        )��P	��c����A�*

	conv_loss�<+?�gpV        )��P	6d����A�*

	conv_loss�S+?_��        )��P	J9d����A�*

	conv_lossbe+?���        )��P	ocd����A�*

	conv_lossfi+?I]�@        )��P	��d����A�*

	conv_lossW+?d�J3        )��P	t�d����A�*

	conv_loss�4+?��xr        )��P	��d����A�*

	conv_loss�+?�ά�        )��P	�e����A�*

	conv_loss�T+?k:        )��P	9e����A�*

	conv_loss�#+?4AR        )��P	�de����A�*

	conv_loss�+?����        )��P	Аe����A�*

	conv_loss4+?���        )��P	\�e����A�*

	conv_loss�+?m�w        )��P	��e����A�*

	conv_loss+?)��        )��P	f����A�*

	conv_loss�'+?XU��        )��P	hBf����A�*

	conv_loss+?u�@        )��P	tnf����A�*

	conv_lossg+?LX�_        )��P	��f����A�*

	conv_loss7	+?.��        )��P	j�f����A�*

	conv_loss��*?�zǗ        )��P	��f����A�*

	conv_loss��*?� )c        )��P	�g����A�*

	conv_loss@+?�c�        )��P	zGg����A�*

	conv_loss�	+?�u�        )��P	%sg����A�*

	conv_loss��*?��xZ        )��P	�g����A�*

	conv_loss �*?-�<�        )��P	A�g����A�*

	conv_loss��*?���        )��P	w�g����A�*

	conv_loss��*?�{�Y        )��P	�"h����A�*

	conv_loss��*?�� g        )��P	UPh����A�*

	conv_loss@�*?�۲b        )��P	�{h����A�*

	conv_lossݲ*?�_1        )��P	��h����A�*

	conv_lossW�*?V�\*        )��P	��h����A�*

	conv_lossa�*?����        )��P	��h����A�*

	conv_loss�*?��f�        )��P	e'i����A�*

	conv_lossc�*?ZW��        )��P	�fi����A�*

	conv_lossF�*?�P�        )��P	8�i����A�*

	conv_loss�*?�^��        )��P	Ѿi����A�*

	conv_loss¯*?�#h�        )��P	��i����A�*

	conv_lossg�*?@��        )��P	>j����A�*

	conv_loss��*?S}i        )��P	Hj����A�*

	conv_loss��*?�+lp        )��P	�rj����A�*

	conv_lossj�*?T��|        )��P	��j����A�*

	conv_loss7{*?9(&        )��P	\�j����A�*

	conv_lossٛ*?���        )��P	�k����A�*

	conv_loss�*?Lw��        )��P	�5k����A�*

	conv_loss�m*?��O�        )��P	�fk����A�*

	conv_loss=�*?-�#�        )��P	1�k����A�*

	conv_loss��*?g�R        )��P	��k����A�*

	conv_loss҉*?icG        )��P	�k����A�*

	conv_loss��*?�{�9        )��P	)'l����A�*

	conv_loss�N*?�d|        )��P	Rl����A�*

	conv_loss�@*?"Qى        )��P	�}l����A�*

	conv_loss�j*?PqY�        )��P	�l����A�*

	conv_lossO**??k�        )��P	6�l����A�*

	conv_loss�C*?g�%�        )��P	~m����A�*

	conv_loss_>*?FpK�        )��P	�-m����A�*

	conv_loss�b*?El�B        )��P	�Ym����A�*

	conv_lossiB*?�ơ	        )��P	e�m����A�*

	conv_loss�	*?��O�        )��P	�m����A�*

	conv_loss�B*?�        )��P	��m����A�*

	conv_loss�*?�}s        )��P	�n����A�*

	conv_lossS*?v.�T        )��P	Cn����A�*

	conv_loss�$*?��II        )��P	�pn����A�*

	conv_loss�*?dS�        )��P	�n����A�*

	conv_loss�?*?�}�        )��P	f�n����A�*

	conv_lossJ�)?Zk��        )��P	��n����A�*

	conv_loss-*?)��h        )��P	{o����A�*

	conv_loss�*?��Q�        )��P	�Ko����A�*

	conv_loss!4*?z��        )��P	�wo����A�*

	conv_loss��)?T�        )��P	L�o����A�*

	conv_loss~�)?���        )��P	��o����A�*

	conv_loss&*?W��        )��P	m�o����A�*

	conv_lossR�)?B)�        )��P	o!p����A�*

	conv_lossR�)? 0��        )��P	.Mp����A�*

	conv_loss3�)?�|9        )��P	
yp����A�*

	conv_loss��)?څ�
        )��P	��p����A�*

	conv_loss�*?�bt        )��P	�p����A�*

	conv_loss��)?��"        )��P	Q�p����A�*

	conv_lossO�)?¹��        )��P	�)q����A�*

	conv_loss1�)?�t�        )��P	mSq����A�*

	conv_lossʙ)?����        )��P	~q����A�*

	conv_loss�)?���B        )��P	J�q����A�*

	conv_loss>�)?��        )��P	��q����A�*

	conv_loss&p)?��`�        )��P	U�v����A�*

	conv_loss��)?��h�        )��P	��v����A�*

	conv_losswk)?W�x�        )��P	e�v����A�*

	conv_loss�t)?�N&        )��P	�w����A�*

	conv_lossc�)?�\�h        )��P	�7w����A�*

	conv_loss��)?�ۥ�        )��P	[dw����A�*

	conv_lossV�)?���        )��P	ލw����A�*

	conv_loss�y)?�y�        )��P		�w����A�*

	conv_lossŰ)?��P�        )��P	��w����A�*

	conv_loss�)?HS�        )��P	�x����A�*

	conv_loss�C)?1T8�        )��P	�Kx����A�*

	conv_loss=�)?�۰-        )��P	�vx����A�*

	conv_loss�l)?	�8�        )��P	��x����A�*

	conv_loss��)?Jn�3        )��P	��x����A�*

	conv_loss�T)?� .�        )��P	��x����A�*

	conv_lossW�)?P�O        )��P	�y����A�*

	conv_loss�e)?��j        )��P	ISy����A�*

	conv_loss�X)?x���        )��P	xy����A�*

	conv_loss�j)?߮"        )��P	�y����A�*

	conv_loss�-)?���        )��P	��y����A�*

	conv_lossY)?�$        )��P	� z����A�*

	conv_loss�V)?q��        )��P	j(z����A�*

	conv_loss�")?��(        )��P	 Uz����A�*

	conv_loss�)?�d�        )��P	y�z����A�*

	conv_lossBQ)?�dX        )��P	��z����A�*

	conv_loss�J)?�b��        )��P	O�z����A�*

	conv_lossU,)?�.X        )��P	){����A�*

	conv_loss!)?j���        )��P	�*{����A�*

	conv_loss�(?��        )��P	�T{����A�*

	conv_loss�=)?����        )��P	�|{����A�*

	conv_loss��(?�o��        )��P	��{����A�*

	conv_lossn�(?�r s        )��P	��{����A�*

	conv_losst�(?"N�        )��P	<�{����A�*

	conv_lossS�(?��}        )��P	�%|����A�*

	conv_lossh)?3"�*        )��P	�N|����A�*

	conv_loss��(?�d�        )��P	�u|����A�*

	conv_loss��(?�v�        )��P	��|����A�*

	conv_loss:�(?it۴        )��P	�|����A�*

	conv_loss��(?@'�        )��P	1�|����A�*

	conv_lossL�(?X!�
        )��P	�}����A�*

	conv_loss��(?;�#        )��P	zG}����A�*

	conv_loss��(?>	B7        )��P	Jp}����A�*

	conv_loss��(?�3?5        )��P	X�}����A�*

	conv_loss��(?>Ȋ        )��P	p�}����A�*

	conv_loss��(?@���        )��P	l�}����A�*

	conv_loss̇(?�vה        )��P	�~����A�*

	conv_lossa�(?�-        )��P	�<~����A�*

	conv_lossQ�(?�3nM        )��P	�d~����A�*

	conv_lossc(?���        )��P	�~����A�*

	conv_loss��(?y<        )��P	��~����A�*

	conv_loss}�(?�X:K        )��P	[�~����A�*

	conv_loss̚(? �0�        )��P	�����A�*

	conv_loss5I(?r�        )��P	�@����A�*

	conv_losss(?��=�        )��P	�l����A�*

	conv_loss�)(?q��<        )��P	������A�*

	conv_lossEj(?�E��        )��P	������A�*

	conv_loss�1(?����        )��P	������A�*

	conv_loss�i(?@|��        )��P	�(�����A�*

	conv_loss�1(?����        )��P	Q�����A�*

	conv_lossJ(?��        )��P	C������A�*

	conv_lossJ4(?�w��        )��P	9������A�*

	conv_loss�l(?�|)        )��P	D뀠���A�*

	conv_lossD�'?�<�        )��P	������A�*

	conv_loss�(?|��        )��P	@�����A�*

	conv_loss�(?;��E        )��P	~j�����A�*

	conv_lossO�'?f��^        )��P	ʙ�����A�*

	conv_loss�#(?y֙u        )��P	3ā����A�*

	conv_loss� (?����        )��P	r󁠉��A�*

	conv_loss�&(?lC�Z        )��P	�"�����A�*

	conv_loss.(?�Q��        )��P	�O�����A�*

	conv_loss�'?��1�        )��P	�z�����A�*

	conv_loss�(?��        )��P	~������A�*

	conv_loss��'?r��D        )��P	�΂����A�*

	conv_loss_(?���        )��P	q������A�*

	conv_loss�'?��G        )��P	?!�����A�*

	conv_lossB�'?���s        )��P	�I�����A�*

	conv_lossC�'?H�JI        )��P	Gr�����A�*

	conv_loss(?<�!`        )��P	<������A�*

	conv_lossސ'?�mA        )��P	�Ń����A�*

	conv_lossP�'?��Z        )��P	-���A�*

	conv_lossh�'?��I        )��P	V�����A�*

	conv_loss��'?���        )��P	VA�����A�*

	conv_lossЛ'?V�        )��P	�i�����A�*

	conv_loss�n'?ʬ�        )��P	L������A�*

	conv_loss�k'?�3�        )��P	彄����A�*

	conv_loss$�'?	�I�        )��P	H焠���A�*

	conv_loss v'?,�        )��P	v�����A�*

	conv_lossZ'?���3        )��P	�:�����A�*

	conv_loss��'?6��.        )��P	�f�����A�*

	conv_loss0�'?�L�        )��P	�������A�*

	conv_loss�l'?�c        )��P	����A�*

	conv_loss@}'?m��q        )��P	%煠���A�*

	conv_lossVJ'?j���        )��P	������A�*

	conv_lossT7'?��j        )��P	�:�����A�*

	conv_loss�"'?Pm�        )��P	�d�����A�*

	conv_lossZ'?�ar�        )��P	e������A�*

	conv_loss=J'?,���        )��P	񸆠���A�*

	conv_loss['?����        )��P	�ᆠ���A�*

	conv_lossTM'?E���        )��P	�����A�*

	conv_loss8'?����        )��P	l4�����A�*

	conv_loss�<'?�ζ�        )��P	�]�����A�*

	conv_loss=�&?i�Y        )��P	Tǈ����A�*

	conv_loss��&?�]A        )��P	�񈠉��A�*

	conv_loss"'?rG�        )��P	������A�*

	conv_loss�'?	W�        )��P	�F�����A�*

	conv_loss�&? m�        )��P	&s�����A�*

	conv_loss��&?d̎�        )��P	`������A�*

	conv_loss$
'?�,��        )��P	�ԉ����A�*

	conv_losse�&?rwȅ        )��P	������A�*

	conv_loss�)'?��L�        )��P	�+�����A�*

	conv_lossc$'?c���        )��P	�^�����A�*

	conv_loss¿&?̌�<        )��P	�������A�*

	conv_loss7�&?�bzZ        )��P	�������A�*

	conv_loss:�&?���$        )��P	�㊠���A�*

	conv_loss��&?\��        )��P	������A�*

	conv_loss��&?V��        )��P	�5�����A�*

	conv_loss�&?kK�        )��P	�]�����A�*

	conv_loss�<&?�ܨ�        )��P	�������A�*

	conv_lossap&?�L`�        )��P	�������A�*

	conv_loss�&?QL        )��P	"݋����A�*

	conv_lossBs&?�G�        )��P	������A�*

	conv_loss�&?��Ѫ        )��P	�?�����A�*

	conv_losssa&?��j        )��P	�l�����A�*

	conv_loss�x&?q�(        )��P	ۗ�����A�*

	conv_loss��&?�,d\        )��P	\������A�*

	conv_lossOG&?���M        )��P	쌠���A�*

	conv_lossq�&?*��        )��P	������A�*

	conv_lossr&?�2�O        )��P	�B�����A�*

	conv_loss�+&?�ˡ;        )��P	�n�����A�*

	conv_loss��%?,��        )��P	������A�*

	conv_loss�&?f$0*        )��P	Jƍ����A�*

	conv_lossUT&?��        )��P	�򍠉��A�*

	conv_loss�&?��ʽ        )��P	������A�*

	conv_loss�(&?m�        )��P	�J�����A�*

	conv_loss��%?�+!�        )��P	�u�����A�*

	conv_loss�&?�w'�        )��P	ʡ�����A�*

	conv_loss9&?�&i�        )��P	�̎����A�*

	conv_losss�%?s�         )��P	R������A�*

	conv_loss|�%?"Em�        )��P	##�����A�*

	conv_lossA�%?�ݾ        )��P	�M�����A�*

	conv_loss9�%?oe9        )��P	y�����A�*

	conv_loss��%?���)        )��P	A������A�*

	conv_loss#�%?��6        )��P	�̏����A�*

	conv_lossK�%?!�=�        )��P	�������A�*

	conv_loss1w%?N��        )��P	� �����A�*

	conv_loss0o%?����        )��P	)L�����A�*

	conv_loss�|%?����        )��P	�y�����A�*

	conv_loss�%?��#u        )��P	£�����A�*

	conv_lossW%?���U        )��P	eΐ����A�*

	conv_loss��%?��1�        )��P	������A�*

	conv_loss{t%?� f        )��P	�"�����A�*

	conv_loss�%?�OL�        )��P	4`�����A�*

	conv_loss*U%?�_3{        )��P	C������A�*

	conv_loss��$?��.        )��P	3������A�*

	conv_lossFo%?�:˷        )��P	mᑠ���A�*

	conv_lossU#%?J���        )��P	!�����A�*

	conv_loss�%?�Pɹ        )��P	�9�����A�*

	conv_loss��%?)��        )��P	hg�����A�*

	conv_loss�r%?�V�        )��P	�������A�*

	conv_loss��$?6ߡ        )��P	�������A�*

	conv_loss��$?�?�B        )��P	=꒠���A�*

	conv_loss�%?g�
�        )��P	�%�����A�*

	conv_loss��$?fc�}        )��P	NR�����A�*

	conv_lossX0%?��`�        )��P	�������A�*

	conv_losse"%?Jp�        )��P	�������A�*

	conv_loss�2%?�<A?        )��P	v㓠���A�*

	conv_lossu�$?�q��        )��P	V�����A�*

	conv_loss(�$?��        )��P	�F�����A�*

	conv_loss�$?7Jj�        )��P	Mr�����A�*

	conv_loss]$?��        )��P	����A�*

	conv_loss��$?�]��        )��P	�Ɣ����A�*

	conv_loss��$?����        )��P	P񔠉��A�*

	conv_loss��$?&L�        )��P	=�����A�*

	conv_loss=Z$?r��        )��P	�D�����A�*

	conv_loss��$?Fg��        )��P	
q�����A�*

	conv_loss}$?���Z        )��P	ʛ�����A�*

	conv_loss[ $?0m�(        )��P	�ŕ����A�*

	conv_loss��$?4�b�        )��P	�󕠉��A�*

	conv_lossW$?-�ew        )��P	������A�*

	conv_loss�B$?P�5�        )��P	�J�����A�*

	conv_loss�R$?�=�)        )��P	0u�����A�*

	conv_loss�Z$?уF=        )��P	 ������A�*

	conv_loss��#?�'c        )��P	�̖����A�*

	conv_loss�L$?���        )��P	;������A�*

	conv_lossw6$?��v�        )��P	x!�����A�*

	conv_loss�$?��mr        )��P	VM�����A�*

	conv_loss~�#?�;�        )��P	cy�����A�*

	conv_loss�c#?�0�        )��P	Ϣ�����A�*

	conv_loss�#?,:y        )��P	q͗����A�*

	conv_loss��#?���        )��P	�������A�*

	conv_loss!�#?���        )��P	�$�����A�*

	conv_lossя#?8���        )��P	3N�����A�*

	conv_loss��#?�%��        )��P	�{�����A�*

	conv_loss��#?@��u        )��P	ר�����A�*

	conv_loss7;#?T�Q�        )��P	�Ә����A�*

	conv_lossa�#?_��        )��P	�������A�*

	conv_loss�|#?�V�y        )��P	?)�����A�*

	conv_loss�2#?{�d�        )��P	U�����A�*

	conv_lossJS#?��+        )��P	�������A�*

	conv_lossc#?�b��        )��P	諙����A�*

	conv_loss�#?I�%        )��P	�י����A�*

	conv_lossT#?=��        )��P	������A�*

	conv_loss#??.L         )��P	3?�����A�*

	conv_loss�#? 7�        )��P	�j�����A�*

	conv_lossSO#?��t        )��P	ڕ�����A�*

	conv_loss�S#?�57        )��P	�������A�*

	conv_loss�#?<��u        )��P	�욠���A�*

	conv_loss|�"?H�D�        )��P	�!�����A�*

	conv_loss�O"?�� p        )��P	�P�����A�*

	conv_loss�"?N�	O        )��P	倛����A�*

	conv_loss�"?�Ue�        )��P	n������A�*

	conv_loss�9#?��        )��P	�㛠���A�*

	conv_loss��"?Ĳ��        )��P	������A�*

	conv_loss��"?|�A        )��P	�<�����A�*

	conv_loss\�"?��7D        )��P	`i�����A�*

	conv_loss'#?Г�l        )��P	"������A�*

	conv_loss�;"?&�        )��P	Ü����A�*

	conv_lossS"?$        )��P	2𜠉��A�*

	conv_lossZ�"?3 q�        )��P	�����A�*

	conv_loss�`"?ݡ        )��P	mF�����A�*

	conv_lossM."?v�        )��P	Cq�����A�*

	conv_loss"%"?$�,�        )��P	�������A�*

	conv_losswC!?����        )��P	�ѝ����A�*

	conv_lossÛ!?9(	F        )��P	������A�*

	conv_lossx�"?)��        )��P	h)�����A�*

	conv_loss��!?�;�        )��P	�W�����A�*

	conv_loss�!?Y64        )��P	�������A�*

	conv_loss�"?��=�        )��P	�������A�*

	conv_lossZ�!?!���        )��P	�؞����A�*

	conv_lossF�!?��        )��P	������A�*

	conv_loss�!?f�r        )��P	?.�����A�*

	conv_loss�!?JL�        )��P	]X�����A�*

	conv_loss4�!?.��        )��P	C������A�*

	conv_loss��!?�ҷ�        )��P	ꭟ����A�*

	conv_loss�� ?�d�]        )��P	�ן����A�*

	conv_loss�!?6q        )��P	������A�*

	conv_loss!!?���w        )��P	�0�����A�*

	conv_lossB?!?��a*        )��P	^Z�����A�*

	conv_loss$<!?���0        )��P	v������A�*

	conv_loss�� ?�8A�        )��P	󮠠���A�*

	conv_lossC!?y�        )��P	�٠����A�*

	conv_loss0� ?e�J        )��P	������A�*

	conv_loss�� ?���        )��P	g-�����A�*

	conv_loss� ?���        )��P	X�����A�*

	conv_loss�� ?���        )��P	�������A�*

	conv_loss�� ?��	-        )��P	�������A�*

	conv_loss � ?h�i#        )��P	�֡����A�*

	conv_loss@, ?Zr�        )��P	� �����A�*

	conv_loss�` ?"�q        )��P	V,�����A�*

	conv_lossss ?#(4N        )��P	\V�����A�*

	conv_loss_� ?`�$        )��P	������A�*

	conv_loss��?��0        )��P	q������A�*

	conv_loss��?�\	�        )��P	Bꢠ���A�*

	conv_loss�� ?=ܹI        )��P	z�����A�*

	conv_loss�_ ?Ŷ �        )��P	SA�����A�*

	conv_loss.�?�1        )��P	�q�����A�*

	conv_loss� ?iT�        )��P	򤣠���A�*

	conv_loss��?DHVj        )��P	�ϣ����A�*

	conv_loss|�?��k        )��P	!������A�*

	conv_loss�?�=��        )��P	'�����A�*

	conv_loss�q?��P�        )��P	�W�����A�*

	conv_lossl]?\.        )��P	t������A�*

	conv_lossա?��S        )��P	~������A�*

	conv_loss=�?�Ɓ        )��P	�ᤠ���A�*

	conv_loss
�?���        )��P	)�����A�*

	conv_loss?��        )��P	�;�����A�*

	conv_loss�"?v�r�        )��P	�g�����A�*

	conv_loss��?� w        )��P		������A�*

	conv_loss(�?��p�        )��P	@������A�*

	conv_loss�	?ł�n        )��P	l률���A�*

	conv_loss+�?խ        )��P	������A�*

	conv_lossq�?��Q�        )��P	CD�����A�*

	conv_loss�?+0        )��P	�s�����A�*

	conv_loss��?2M�u        )��P	�������A�*

	conv_loss�?��I        )��P	�ئ����A�*

	conv_losswC?
蘖        )��P	4�����A�*

	conv_loss(a?Ӻʠ        )��P	.,�����A�*

	conv_lossS?�D�Q        )��P	iW�����A�*

	conv_lossE�?%s�        )��P	d������A�*

	conv_loss�?��W�        )��P	A������A�*

	conv_lossn�?b��        )��P	�������A�*

	conv_loss�?���        )��P	�����A�*

	conv_loss�(?�e�        )��P	�D�����A�*

	conv_loss/?�g��        )��P	�p�����A�*

	conv_loss�?�s�]        )��P	�������A�*

	conv_loss��?1���        )��P	�ƨ����A�*

	conv_loss��?��Q        )��P	x򨠉��A�*

	conv_loss2�?�)        )��P	E�����A�*

	conv_loss��?���        )��P	�K�����A�*

	conv_lossjn?؝�        )��P	Ax�����A�*

	conv_lossč?���        )��P	G������A�*

	conv_loss��?�f        )��P	8ө����A�*

	conv_loss`?�'��        )��P	O������A�*

	conv_loss"�?YJX        )��P	*�����A�*

	conv_loss&�?˰�Y        )��P	�S�����A�*

	conv_loss�?�V�Z        )��P	�����A�*

	conv_lossJ�?�U��        )��P	*������A�*

	conv_losso�?�V�G        )��P	�Ӫ����A�*

	conv_loss?X?���        )��P	^������A�*

	conv_loss1?�/��        )��P	$(�����A�*

	conv_loss�?]N�        )��P	iT�����A�*

	conv_loss��?a&��        )��P	�������A�*

	conv_loss#�?��Z+        )��P	�笠���A�*

	conv_losss�?`"�T        )��P	<�����A�*

	conv_loss��?v!�        )��P	�<�����A�*

	conv_loss&�?Q�:        )��P	i�����A�*

	conv_loss"?ƙ��        )��P	T������A�*

	conv_loss�=?�\zR        )��P	�í����A�*

	conv_loss��?�'��        )��P	%������A�*

	conv_loss$c?7#6�        )��P	#�����A�*

	conv_loss��?����        )��P	�V�����A�*

	conv_loss5�?�9        )��P	y������A�*

	conv_lossʬ?��         )��P	a������A�*

	conv_loss3?�0�        )��P	⮠���A�*

	conv_losst?�9��        )��P	������A�*

	conv_loss�?�        )��P	�C�����A�*

	conv_loss�!?��L        )��P	u�����A�*

	conv_loss#�?5Y�        )��P	)������A�*

	conv_loss;D?�+=        )��P	5̯����A�*

	conv_lossy1?��ϡ        )��P	a������A�*

	conv_loss�u?��l�        )��P	y$�����A�*

	conv_loss�o?#c�        )��P	6N�����A�*

	conv_loss6�?\l��        )��P	{�����A�*

	conv_losss.?.��        )��P	$������A�*

	conv_loss�K?=#�q        )��P	yӰ����A�*

	conv_loss��?�U^�        )��P	�������A�*

	conv_loss�E?�մ�        )��P		,�����A�*

	conv_loss�D?=K!�        )��P	�T�����A�*

	conv_loss�W? i�        )��P	~�����A�*

	conv_loss^�?➋        )��P	������A�*

	conv_loss��?x�W        )��P	�ױ����A�*

	conv_loss8&?��]�        )��P	������A�*

	conv_losss?Ed-�        )��P	1�����A�*

	conv_loss�?��D>        )��P	�[�����A�*

	conv_loss��?�T��        )��P	G������A�*

	conv_loss��?ؐ7T        )��P	5������A�*

	conv_lossWy?1�3k        )��P	޲����A�*

	conv_loss|?����        )��P	������A�*

	conv_lossf�?b{        )��P	g1�����A�*

	conv_lossX�?|�ӡ        )��P	\�����A�*

	conv_lossH�?e �        )��P	솳����A�*

	conv_loss��?-H��        )��P	�������A�*

	conv_loss�?1��        )��P	�ݳ����A�*

	conv_loss��?�t+�        )��P	�	�����A�*

	conv_loss��?�@��        )��P	�2�����A�*

	conv_loss�{?s�ov        )��P	"]�����A�*

	conv_loss^�?�`        )��P	܆�����A�*

	conv_loss^�?Q��        )��P	������A�*

	conv_lossJ^?\I�F        )��P	�ܴ����A�*

	conv_loss(b?�?        )��P	�����A�*

	conv_loss�?��T�        )��P	�3�����A�*

	conv_loss�P?p�V        )��P	�p�����A�*

	conv_loss2?^��-        )��P	'������A�*

	conv_losse{?�j۰        )��P	�ȵ����A�*

	conv_loss$�?��        )��P	������A�*

	conv_loss�2?_UѮ        )��P	�%�����A�*

	conv_loss� ?�s6o        )��P	�O�����A�*

	conv_loss=6?�Rp�        )��P	�z�����A�*

	conv_loss��?�oV�        )��P	.������A�*

	conv_lossu?mVx        )��P	Y׶����A�*

	conv_lossJW?Dy�        )��P	�����A�*

	conv_loss;�?���        )��P	D�����A�*

	conv_lossE�?��T=        )��P	�r�����A�*

	conv_loss��?����        )��P	e������A�*

	conv_lossr?���        )��P	^ŷ����A�*

	conv_loss�	?���        )��P	 񷠉��A�*

	conv_loss��?k        )��P	q�����A�*

	conv_loss?��-        )��P	�K�����A�*

	conv_loss�@?��+�        )��P	y�����A�*

	conv_loss�>?{��3        )��P	#������A�*

	conv_loss�?�[S        )��P	�͸����A�*

	conv_loss?F��/        )��P	�������A�*

	conv_loss~6?L��        )��P	�$�����A�*

	conv_lossa]?#U_~        )��P	�Y�����A�*

	conv_loss�@?��{        )��P	o������A�*

	conv_loss��?6�v2        )��P	H������A�*

	conv_lossn�?2�uA        )��P	;⹠���A�*

	conv_lossSX?�at        )��P	������A�*

	conv_lossQ�?}���        )��P	9�����A�*

	conv_loss%�?�(��        )��P	�c�����A�*

	conv_lossi?ʕ        )��P	�������A�*

	conv_loss"�?�[�        )��P	1������A�*

	conv_loss��?��cL        )��P	h溠���A�*

	conv_loss�?ڡp        )��P	������A�*

	conv_loss�?��        )��P	L;�����A�*

	conv_lossب?�nc�        )��P	�d�����A�*

	conv_loss/�?�i        )��P	�������A�*

	conv_lossX�?���        )��P	|������A�*

	conv_losskC?$�8        )��P	廠���A�*

	conv_loss�?�oN�        )��P	������A�*

	conv_loss|?��#        )��P	�8�����A�*

	conv_loss��?Ф]u        )��P	 d�����A�*

	conv_loss0�?0�        )��P	������A�*

	conv_loss�?	0
         )��P	_������A�*

	conv_losse;?�x�}        )��P	*张���A�*

	conv_loss�?��U        )��P	0�����A�*

	conv_loss��?Ŗ�        )��P	{=�����A�*

	conv_loss�f?�m�9        )��P	g�����A�*

	conv_loss3�?��        )��P	������A�*

	conv_loss��?(���        )��P	�������A�*

	conv_loss�{?�SH�        )��P	�㽠���A�*

	conv_loss:w?`4x�        )��P	������A�*

	conv_loss�?��|v        )��P	I�����A�*

	conv_loss�
?�|#G        )��P	es�����A�*

	conv_lossx�
?�/u        )��P	ߝ�����A�*

	conv_loss�
?�T}�        )��P	;����A�*

	conv_lossHY
?r}�        )��P	�������A�*

	conv_loss:�	?�)        )��P	j%�����A�*

	conv_loss��	?��H        )��P	<^�����A�*

	conv_lossVs
?D��Y        )��P	ډ�����A�*

	conv_lossQ	?�0W�        )��P	m������A�*

	conv_loss�*
?��
y        )��P	�뿠���A�*

	conv_loss��
?5P%?        )��P	������A�*

	conv_loss��?!s�        )��P	�C�����A�*

	conv_loss?�	?���        )��P	�s�����A�*

	conv_lossu	?�u�        )��P	������A�*

	conv_loss��?��        )��P	�������A�*

	conv_loss�h?Sj�J        )��P	�������A�*

	conv_loss�m? (�        )��P	�&�����A�*

	conv_loss�?NI�        )��P	T�����A�*

	conv_lossk�?(pM        )��P	)�����A�*

	conv_loss�D?��~         )��P	˩�����A�*

	conv_loss��?�i�W        )��P	�������A�*

	conv_lossݳ?���        )��P	�������A�*

	conv_loss8�?��Be        )��P	�- ���A�*

	conv_loss�u?�{         )��P	fZ ���A�*

	conv_loss_?C�g7        )��P	�� ���A�*

	conv_loss��?ɪ��        )��P		� ���A�*

	conv_loss��?R���        )��P	�� ���A�*

	conv_loss��?^��        )��P	uà���A�*

	conv_loss��?�X<,        )��P	&6à���A�*

	conv_loss��?�%�?        )��P	�bà���A�*

	conv_loss�-?���        )��P	��à���A�*

	conv_lossL9?U��        )��P	f�à���A�*

	conv_loss��?^y�H        )��P	��à���A�*

	conv_lossU}?��f{        )��P	� Ġ���A�*

	conv_loss��?:�(        )��P	�KĠ���A�*

	conv_loss=e?����        )��P	 xĠ���A�*

	conv_loss]?���        )��P	�Ġ���A�*

	conv_loss3}�>�^�        )��P	��Ġ���A�*

	conv_lossU� ?o�4        )��P	%�Ġ���A�*

	conv_loss���>�@�        )��P	�"Š���A�*

	conv_loss��>�.��        )��P	_MŠ���A�*

	conv_lossq��>����        )��P	%xŠ���A�*

	conv_loss���>���        )��P	A�Š���A�*

	conv_loss"w�>HKH        )��P	M�Š���A�*

	conv_loss��>��$|        )��P	��Š���A�*

	conv_loss�$�>#w�}        )��P	�%Ơ���A�*

	conv_loss�T�>��SF        )��P	�PƠ���A�*

	conv_loss���>]�`        )��P	�zƠ���A�*

	conv_lossg_�>��        )��P	4�Ơ���A�*

	conv_loss���>�'�x        )��P	��Ơ���A�*

	conv_lossr��>�ڬ        )��P	�Ǡ���A�*

	conv_lossi�>m�r        )��P	>Ǡ���A�*

	conv_loss��>�j�
        )��P	yiǠ���A�*

	conv_loss��>���|        )��P	�Ǡ���A�*

	conv_loss8`�>���#        )��P	P�Ǡ���A�*

	conv_loss3u�>"�&�        )��P	o�Ǡ���A�*

	conv_lossJp�>
���        )��P	kȠ���A�*

	conv_loss�%�>�V޲        )��P	�BȠ���A�*

	conv_lossKe�>2���        )��P	�oȠ���A�*

	conv_lossM��>�Y�        )��P	��Ƞ���A�*

	conv_loss=y�>8�j        )��P	��Ƞ���A�*

	conv_loss�%�>�,        )��P	�ɠ���A�*

	conv_loss�(�>3���        )��P	<6ɠ���A�*

	conv_lossr@�>㻪?        )��P	'`ɠ���A�*

	conv_loss���>�C�        )��P	��ɠ���A�*

	conv_loss�5�>��D"        )��P	^�ɠ���A�*

	conv_loss40�>�D�        )��P	��ɠ���A�*

	conv_loss-��>� �@        )��P	Jʠ���A�*

	conv_loss�L�>�ih�        )��P	�Eʠ���A�*

	conv_lossAt�>WF��        )��P	Vpʠ���A�*

	conv_loss]��>�Z�        )��P	�ʠ���A�*

	conv_loss��>���        )��P	N�ʠ���A�*

	conv_loss���>֑�        )��P	��ʠ���A�*

	conv_lossɢ�>t�d.        )��P	%ˠ���A�*

	conv_loss���>�sq�        )��P	Qˠ���A�*

	conv_loss���>�        )��P	wˠ���A�*

	conv_loss~��>�нp        )��P	ܪˠ���A�*

	conv_losstf�>+z�        )��P	L�ˠ���A�*

	conv_loss���>���        )��P	�̠���A�*

	conv_loss��>
�Ǳ        )��P	l.̠���A�*

	conv_loss�"�>���[        )��P	�X̠���A�*

	conv_loss��>�g�	        )��P	�̠���A�*

	conv_loss���>���        )��P	��̠���A�*

	conv_loss���>��_u        )��P	"�̠���A�*

	conv_loss��>��        )��P	`͠���A�*

	conv_lossε�>�j�'        )��P	�/͠���A�*

	conv_loss#��>����        )��P	�Z͠���A�*

	conv_loss��>G��        )��P	}�͠���A�*

	conv_loss���>=H2�        )��P	��͠���A�*

	conv_loss�H�>�*.        )��P	��͠���A�*

	conv_loss;z�>(�~h        )��P	�Π���A�*

	conv_lossA��>&>        )��P	�?Π���A�*

	conv_loss���>��I        )��P	jΠ���A�*

	conv_loss��>�΅�        )��P	8�Π���A�*

	conv_lossN]�>�?        )��P	��Π���A�*

	conv_loss!�>?sH�        )��P	��Π���A�*

	conv_loss���>�'o        )��P	�Ϡ���A�*

	conv_loss�~�>�_�        )��P	}@Ϡ���A�*

	conv_lossX]�>X��        )��P	J�Ӡ���A�*

	conv_loss�D�> &�w        )��P	�Cՠ���A�*

	conv_lossw4�>�.d        )��P	9nՠ���A�*

	conv_loss ��>���5        )��P	��ՠ���A�*

	conv_loss�E�>�\��        )��P	Z�ՠ���A�*

	conv_loss���>���|        )��P	G�ՠ���A�*

	conv_lossoZ�>���|        )��P	�֠���A�*

	conv_loss1�>�&�S        )��P	n;֠���A�*

	conv_loss��>�te        )��P	Pf֠���A�*

	conv_lossd�>�}j        )��P	r�֠���A�*

	conv_lossp�>�]��        )��P	�֠���A�*

	conv_loss�W�>��p�        )��P	��֠���A�*

	conv_lossoW�>ӱ        )��P	�2נ���A�*

	conv_lossY��>T��        )��P	r[נ���A�*

	conv_lossnb�>���?        )��P	��נ���A�*

	conv_loss�{�>�pL�        )��P	��נ���A�*

	conv_loss�A�>L���        )��P	��נ���A�*

	conv_lossR#�>6�        )��P	�ؠ���A�*

	conv_loss_��>4�        )��P	>7ؠ���A�*

	conv_loss8��>�ý\        )��P	|aؠ���A�*

	conv_loss|7�>�.��        )��P	�ؠ���A�*

	conv_loss�L�>���Y        )��P	ʵؠ���A�*

	conv_loss�g�>:,f�        )��P	=�ؠ���A�*

	conv_lossf �>((�        )��P	X٠���A�*

	conv_lossc��>�P)[        )��P	�4٠���A�*

	conv_loss9��>p���        )��P	�_٠���A�*

	conv_loss�>X[g        )��P	Ή٠���A�*

	conv_loss>W�>C���        )��P	��٠���A�*

	conv_loss1w�>1��4        )��P	��٠���A�*

	conv_loss�t�>���        )��P	�ڠ���A�*

	conv_loss)��>8&s        )��P	E1ڠ���A�*

	conv_loss��>R��&        )��P	Yڠ���A�*

	conv_loss���>�21�        )��P	��ڠ���A�*

	conv_loss�-�>i���        )��P	خڠ���A�*

	conv_loss�E�>f��        )��P	h�ڠ���A�*

	conv_lossa�>7S��        )��P	�۠���A�*

	conv_loss��>g�.        )��P	H/۠���A�*

	conv_loss���>k�xV        )��P	�X۠���A�*

	conv_loss3K�>3]U�        )��P	J�۠���A�*

	conv_lossX�>x�g        )��P	��۠���A�*

	conv_loss��>f�^�        )��P	��۠���A�*

	conv_loss��>;O|�        )��P	w�۠���A�*

	conv_lossԌ�>3f��        )��P	�$ܠ���A�*

	conv_loss���>��>e        )��P	�Lܠ���A�*

	conv_loss4�>��h_        )��P	�tܠ���A�*

	conv_lossLr�>s���        )��P	��ܠ���A�*

	conv_loss��>��        )��P	��ܠ���A�*

	conv_loss�\�>����        )��P	��ܠ���A�*

	conv_loss���>��/R        )��P	eݠ���A�*

	conv_loss���>        )��P	ODݠ���A�*

	conv_lossG��>�ǺC        )��P	�mݠ���A�*

	conv_loss���>��{�        )��P	�ݠ���A�*

	conv_loss�/�>H��        )��P	��ݠ���A�*

	conv_loss�H�>ڐ��        )��P	0�ݠ���A�*

	conv_loss Դ>I�e        )��P	�"ޠ���A�*

	conv_loss#��>��X/        )��P	�Nޠ���A�*

	conv_loss���>�g�        )��P	�xޠ���A�*

	conv_loss�۴>�#�        )��P	�ޠ���A�*

	conv_loss�x�>��҂        )��P	J�ޠ���A�*

	conv_loss���>�l�A        )��P	�ޠ���A�*

	conv_loss��>� G�        )��P	u&ߠ���A�*

	conv_lossr�>���8        )��P	dߠ���A�*

	conv_lossе>*ɝ        )��P	��ߠ���A�*

	conv_loss\�>'s��        )��P	@�ߠ���A�*

	conv_loss.��>}76        )��P	o�ߠ���A�*

	conv_loss�Ѷ>���0        )��P	�ࠉ��A�*

	conv_loss�
�>/�J        )��P	�Jࠉ��A�*

	conv_loss�ƶ>��wv        )��P	�uࠉ��A�*

	conv_losss"�>-_N        )��P	��ࠉ��A�*

	conv_losso��>f��        )��P	^�ࠉ��A�*

	conv_loss0n�>�0�c        )��P	��ࠉ��A�*

	conv_loss���>-�D5        )��P	� ᠉��A�*

	conv_loss��>M�d�        )��P	�J᠉��A�*

	conv_loss%ŵ>�]��        )��P	w᠉��A�*

	conv_loss�>�[��        )��P	��᠉��A�*

	conv_loss�Q�>��        )��P	��᠉��A�*

	conv_loss�}�>��        )��P	�⠉��A�*

	conv_loss���>'���        )��P	W-⠉��A�*

	conv_loss���>QǵJ        )��P	�V⠉��A�*

	conv_loss6��>�'��        )��P	�⠉��A�*

	conv_losss��>���        )��P	~�⠉��A�*

	conv_loss���>�T�        )��P	��⠉��A�*

	conv_loss���>0�ʾ        )��P	$㠉��A�*

	conv_loss�Ϭ>�>V�        )��P	�0㠉��A�*

	conv_loss���>"�K        )��P	^㠉��A�*

	conv_loss���>�دS        )��P	/�㠉��A�*

	conv_loss�l�>j	t        )��P	c�㠉��A�*

	conv_lossM�>�<�        )��P	�㠉��A�*

	conv_loss(��>�n��        )��P	�	䠉��A�*

	conv_loss^��>�N|_        )��P	"5䠉��A�*

	conv_loss�x�>Ĝ�$        )��P		^䠉��A�*

	conv_loss��>�/�        )��P	Q�䠉��A�*

	conv_loss�Ю>�|Ne        )��P	��䠉��A�*

	conv_loss�U�>:x�x        )��P	u�䠉��A�*

	conv_lossگ>�T        )��P	{堉��A�*

	conv_loss��>NLa        )��P	N0堉��A�*

	conv_loss�3�>���        )��P	�Z堉��A�*

	conv_loss'h�>j4�:        )��P	3�堉��A�*

	conv_loss6��>m�}        )��P	�堉��A�*

	conv_losskd�>w�\�        )��P	��堉��A�*

	conv_loss<�>�kI        )��P	e栉��A�*

	conv_loss��>��^        )��P	�C栉��A�*

	conv_lossl�>uzZ?        )��P	�n栉��A�*

	conv_loss��>�q.�        )��P	��栉��A�*

	conv_lossl �>��        )��P	�栉��A�*

	conv_loss��>T���        )��P	��栉��A�*

	conv_lossGޫ>��        )��P	="砉��A�*

	conv_lossC��>O��        )��P	�O砉��A�*

	conv_loss�>�>�D        )��P	@y砉��A�*

	conv_loss�Ȭ>vl��        )��P	#�砉��A�*

	conv_loss.�>��9�        )��P	��砉��A�*

	conv_losspƮ>Tn�        )��P	�蠉��A�*

	conv_loss0	�>rn�        )��P	4蠉��A�*

	conv_lossS��>۠-        )��P	^`蠉��A�*

	conv_loss2(�>��J        )��P	��蠉��A�*

	conv_loss�Y�>����        )��P	��蠉��A�*

	conv_loss�'�>�w��        )��P	�蠉��A�*

	conv_loss��>�#�Y        )��P	�頉��A�*

	conv_loss���>?�)        )��P	(R頉��A�*

	conv_loss�0�>��8�        )��P	�頉��A�*

	conv_lossq��>�U�        )��P	E�頉��A�*

	conv_loss�ө>c{Y]        )��P	��頉��A�*

	conv_loss��>���v        )��P	V ꠉ��A�*

	conv_lossp$�>��        )��P	B*ꠉ��A�*

	conv_loss�e�>?�/�        )��P	mSꠉ��A�*

	conv_loss���>����        )��P	�ꠉ��A�*

	conv_lossI�>��	�        )��P	ʬꠉ��A�*

	conv_lossb�>~>�        )��P	��ꠉ��A�*

	conv_loss�<�>wA|        )��P	Q렉��A�*

	conv_loss��> ��6        )��P	}-렉��A�*

	conv_loss\%�>�}        )��P	�U렉��A�*

	conv_loss�#�>��bs        )��P	�렉��A�*

	conv_loss!>�>��e        )��P	֩렉��A�*

	conv_loss���>х?�        )��P	��렉��A�*

	conv_lossS�>�\MV        )��P	p�렉��A�*

	conv_lossv��>Z�q�        )��P	�(젉��A�*

	conv_loss�V�>��,        )��P	4Q젉��A�*

	conv_loss3��>u�RQ        )��P	�젉��A�*

	conv_loss���>�Qɚ        )��P	��젉��A�*

	conv_loss䘨>����        )��P	��젉��A�*

	conv_loss�q�>5�}        )��P		�젉��A�*

	conv_loss�c�>�Q�W        )��P	)�����A�*

	conv_loss��>���9        )��P	�S�����A�*

	conv_lossXm�>���[        )��P	�~�����A�*

	conv_loss 5�>�lq        )��P	,������A�*

	conv_loss���>!i�        )��P	r������A�*

	conv_loss�C�>�Y&�        )��P	Z��A�*

	conv_loss y�>d��k        )��P	�+��A�*

	conv_loss�a�>k��        )��P	DW��A�*

	conv_loss�c�>щ�>        )��P	%���A�*

	conv_loss�^�>1�        )��P	���A�*

	conv_loss0�>4R
        )��P	����A�*

	conv_loss� �>!*��        )��P	���A�	*

	conv_loss�W�>�<M        )��P	?��A�	*

	conv_loss���>�@}        )��P	xi��A�	*

	conv_loss���>��l        )��P	Ɨ��A�	*

	conv_lossY*�>���        )��P	����A�	*

	conv_loss�?�>����        )��P	5���A�	*

	conv_loss��>Y�e        )��P	1"𠉙�A�	*

	conv_loss��>yZ        )��P	�L𠉙�A�	*

	conv_loss>�>���        )��P	�}𠉙�A�	*

	conv_loss��>�q]        )��P	;�𠉙�A�	*

	conv_loss8ڬ>���g        )��P	��𠉙�A�	*

	conv_loss��>n�[�        )��P	.񠉙�A�	*

	conv_lossu��>�ۡ        )��P	�@񠉙�A�	*

	conv_loss�)�>����        )��P	�m񠉙�A�	*

	conv_loss7y�>�G�t        )��P	+�񠉙�A�	*

	conv_loss��>���        )��P	��񠉙�A�	*

	conv_loss�ۦ>2�,p        )��P	��񠉙�A�	*

	conv_loss=Ѧ>�#a        )��P	�!򠉙�A�	*

	conv_loss�s�>�R�        )��P	]M򠉙�A�	*

	conv_loss#d�>��;        )��P	x򠉙�A�	*

	conv_loss�h�>4H{�        )��P	��򠉙�A�	*

	conv_loss�Ԩ>�_�        )��P	#�򠉙�A�	*

	conv_loss�٩>���e        )��P	��򠉙�A�	*

	conv_loss�N�>߭/        )��P	�#󠉙�A�	*

	conv_loss�+�>E���        )��P	�O󠉙�A�	*

	conv_loss/i�>�P        )��P	��󠉙�A�	*

	conv_loss�:�>ʊ0�        )��P	m�󠉙�A�	*

	conv_loss��>x��@        )��P	������A�	*

	conv_loss��>4��        )��P	}0�����A�	*

	conv_lossʤ�>�;7        )��P	�[�����A�	*

	conv_loss��>x��        )��P	:������A�	*

	conv_lossE�>9
m�        )��P	Я�����A�	*

	conv_loss9R�>�8mB        )��P	�������A�	*

	conv_loss�>RW��        )��P	������A�	*

	conv_loss	��>�b��        )��P	/�����A�	*

	conv_loss��>J���        )��P	�[�����A�	*

	conv_loss	ҩ>�<�o        )��P	l������A�	*

	conv_lossy�>ly        )��P	9������A�	*

	conv_loss��>�R��        )��P	a������A�	*

	conv_loss�E�>#�y�        )��P	������A�	*

	conv_loss�b�>@Tʰ        )��P	�B�����A�	*

	conv_loss��>��4        )��P	�m�����A�	*

	conv_loss�P�>�:��        )��P	�������A�	*

	conv_loss,�>�j�        )��P	2������A�	*

	conv_loss[��>�_        )��P	'������A�	*

	conv_lossP��>��H�        )��P	������A�	*

	conv_lossg��>P�\�        )��P	�H�����A�	*

	conv_lossee�>Q�?        )��P	jr�����A�	*

	conv_lossC��>K�=�        )��P	՜�����A�	*

	conv_loss��>�CA�        )��P	v�����A�	*

	conv_lossh�>4�(M        )��P	_2�����A�	*

	conv_loss���>�c8        )��P	�]�����A�	*

	conv_loss�/�>ç��        )��P	�������A�	*

	conv_loss�r�>��\w        )��P	�������A�	*

	conv_loss{g�>C���        )��P	�������A�	*

	conv_lossC!�>��H\        )��P	������A�	*

	conv_loss@��>$�A        )��P	�G�����A�	*

	conv_losss٩>�i�C        )��P	r�����A�	*

	conv_loss/R�>���         )��P	Ǣ�����A�	*

	conv_lossD��>��t�        )��P	�������A�	*

	conv_loss|�>�]�        )��P	/�����A�	*

	conv_loss���>K�O�        )��P	B�����A�	*

	conv_loss'��>,��        )��P	�n�����A�	*

	conv_loss3C�>�<��        )��P	�������A�	*

	conv_loss.��>�=�        )��P	�������A�	*

	conv_loss*a�>��z        )��P	/������A�	*

	conv_loss���>�\s]        )��P	������A�	*

	conv_loss;�>�A{        )��P	�C�����A�	*

	conv_loss���>@ ��        )��P	�o�����A�	*

	conv_lossd�>wo��        )��P	V������A�	*

	conv_loss�ʦ>�F        )��P	�������A�	*

	conv_lossv�>N�;L        )��P	R������A�	*

	conv_losse�>��        )��P	A�����A�	*

	conv_loss���>���e        )��P	�H�����A�	*

	conv_lossT�>��P        )��P	~�����A�	*

	conv_lossQj�>_X$�        )��P	�������A�	*

	conv_loss_�>�m�         )��P	l������A�	*

	conv_loss�^�>�'N        )��P	*�����A�	*

	conv_loss�G�>"i{D        )��P	�8�����A�	*

	conv_lossd�>��g        )��P	le�����A�	*

	conv_loss{n�>w`%n        )��P	�������A�	*

	conv_loss4�>^CP:        )��P	������A�	*

	conv_loss�&�>3KF�        )��P	Z������A�	*

	conv_loss֦>�F9        )��P	������A�	*

	conv_lossNo�>�5k�        )��P	P;�����A�	*

	conv_loss8h�>ya%S        )��P	�d�����A�	*

	conv_loss��>��!�        )��P	�������A�	*

	conv_loss�G�>с        )��P	+������A�	*

	conv_loss� �>1sy�        )��P	�������A�	*

	conv_loss���>���        )��P	� ����A�	*

	conv_loss�0�>�ѧ        )��P	r? ����A�	*

	conv_loss4��>f5��        )��P	Tk ����A�	*

	conv_loss&ҡ>3�u6        )��P	l� ����A�	*

	conv_lossށ�>q\�]        )��P	Q� ����A�	*

	conv_lossFͥ>�Ю�        )��P	:� ����A�	*

	conv_loss˥>m��        )��P	?����A�	*

	conv_loss(�>��         )��P	XG����A�	*

	conv_loss�ס>��R        )��P		r����A�	*

	conv_lossu��>�N��        )��P	!�����A�	*

	conv_lossA@�>|hd        )��P	O�����A�	*

	conv_loss9%�>�UcE        )��P	����A�	*

	conv_loss�Z�>[Q~        )��P	�+����A�	*

	conv_losse��>��p:        )��P	2X����A�	*

	conv_loss�˦>N���        )��P	d�����A�	*

	conv_loss� �>:/�)        )��P	:�����A�	*

	conv_lossϙ�>��        )��P		�����A�	*

	conv_loss��>XaK=        )��P	�����A�	*

	conv_loss3��>f��C        )��P	g0����A�	*

	conv_loss�ӣ>�n�        )��P	�`����A�	*

	conv_lossp�>�=��        )��P	 �����A�	*

	conv_loss��>X�s        )��P	{�����A�	*

	conv_loss�t�>���$        )��P	������A�	*

	conv_lossՂ�>���h        )��P	�����A�	*

	conv_loss01�>�%�~        )��P	a:����A�	*

	conv_loss
=�>g��        )��P	m����A�	*

	conv_loss)z�>B��O        )��P	ɖ����A�	*

	conv_loss.m�>�+"x        )��P	E�����A�	*

	conv_losstڥ>��z�        )��P	������A�	*

	conv_loss���>��s        )��P	c'����A�	*

	conv_loss��>M�"�        )��P	�V����A�	*

	conv_loss�,�>~-�        )��P	X�����A�	*

	conv_loss(e�>����        )��P	h�����A�	*

	conv_loss)R�>��!        )��P	]�����A�	*

	conv_lossJ�>�x�        )��P	� ����A�	*

	conv_loss�ͤ>#;�        )��P	W+����A�	*

	conv_lossXΥ>�a�S        )��P	�U����A�	*

	conv_loss�f�>�Z��        )��P	!�����A�	*

	conv_loss�_�>�G�        )��P	V�����A�	*

	conv_loss
�>��Ф        )��P	������A�
*

	conv_loss���>i�Qm        )��P	����A�
*

	conv_lossb��>"ۉ        )��P	�6����A�
*

	conv_loss���>�'<;        )��P	�b����A�
*

	conv_loss܈�>jlzi        )��P	������A�
*

	conv_lossF3�>�P�0        )��P	������A�
*

	conv_loss�٣>��i�        )��P	������A�
*

	conv_loss���>R��        )��P	9����A�
*

	conv_loss��>�n-        )��P	�8����A�
*

	conv_lossO�>d�]{        )��P	%b����A�
*

	conv_loss�|�>b�        )��P	Ќ����A�
*

	conv_lossWq�>:�"�        )��P	�����A�
*

	conv_loss!��>��[�        )��P	Z�����A�
*

	conv_loss�Ц>�{*        )��P	{	����A�
*

	conv_loss}��> F]        )��P	�:	����A�
*

	conv_loss-��>�܏        )��P	e	����A�
*

	conv_loss�1�>E��T        )��P	�	����A�
*

	conv_loss~ܣ>b�bR        )��P	E�	����A�
*

	conv_loss�A�>���        )��P	��	����A�
*

	conv_loss3��>*���        )��P	�
����A�
*

	conv_loss��>��õ        )��P	�:
����A�
*

	conv_loss9W�>υ�`        )��P	�u
����A�
*

	conv_loss�Y�>�B�        )��P	s�
����A�
*

	conv_lossa0�>���        )��P	^�
����A�
*

	conv_loss�6�>�`t@        )��P	��
����A�
*

	conv_loss�¤>���%        )��P	z$����A�
*

	conv_loss�!�>.N�        )��P	�M����A�
*

	conv_loss!�>0�y�        )��P	hw����A�
*

	conv_loss�Z�>�C �        )��P	Ǫ����A�
*

	conv_losslѥ>�o        )��P	������A�
*

	conv_loss	�>yD�        )��P	� ����A�
*

	conv_loss�1�>�SF        )��P	�,����A�
*

	conv_losswm�>3o��        )��P	�W����A�
*

	conv_lossH�>��s	        )��P	R�����A�
*

	conv_loss�ަ>"`�        )��P	#�����A�
*

	conv_loss���>

I        )��P	������A�
*

	conv_loss懣>Y�D        )��P	�����A�
*

	conv_loss�ߤ>/hbL        )��P	�?����A�
*

	conv_loss$:�>*N�        )��P	Ml����A�
*

	conv_lossI��>dՙ�        )��P	������A�
*

	conv_loss�^�>���?        )��P	1�����A�
*

	conv_lossJ��>�:�        )��P	������A�
*

	conv_loss��>��!        )��P	����A�
*

	conv_lossz�>�)|        )��P	hH����A�
*

	conv_loss���>|u�        )��P	�s����A�
*

	conv_loss�k�>AC=	        )��P	2�����A�
*

	conv_loss�U�>N        )��P	������A�
*

	conv_loss鋣>2*�        )��P	������A�
*

	conv_loss|�>CP�        )��P	^ ����A�
*

	conv_loss�*�>�/�Z        )��P	bM����A�
*

	conv_lossJ�>�%        )��P	9w����A�
*

	conv_loss�Ţ>�~J�        )��P	ӡ����A�
*

	conv_lossˣ>+SWH        )��P	,�����A�
*

	conv_lossL4�>>R��        )��P	I�����A�
*

	conv_loss��>�'�a        )��P	S"����A�
*

	conv_loss���>��        )��P	�J����A�
*

	conv_loss�G�>�        )��P	|u����A�
*

	conv_loss��>�,6}        )��P	1�����A�
*

	conv_lossi��>s�;�        )��P	������A�
*

	conv_loss�>� �        )��P	�����A�
*

	conv_loss	��>�"m�        )��P	�����A�
*

	conv_loss�ܥ>;�q        )��P	gI����A�
*

	conv_loss7�>��6        )��P	�t����A�
*

	conv_loss��>H��        )��P	G�����A�
*

	conv_lossy�>�eM        )��P	������A�
*

	conv_loss���>fOR        )��P	�����A�
*

	conv_loss���>�@�         )��P	7#����A�
*

	conv_lossX�>�\B�        )��P	�O����A�
*

	conv_loss�ۡ>��mo        )��P	�{����A�
*

	conv_loss��>rЯ        )��P	i�����A�
*

	conv_loss ��>�A:�        )��P	������A�
*

	conv_loss�ڟ>�_\        )��P	+����A�
*

	conv_loss�D�>Q���        )��P	�=����A�
*

	conv_loss��>̘8        )��P	Cj����A�
*

	conv_loss+�>
91G        )��P	:�����A�
*

	conv_loss�>f
I\        )��P	!�����A�
*

	conv_loss�h�>V�x�        )��P	c�����A�
*

	conv_loss��>w�0        )��P	�"����A�
*

	conv_lossCG�>��	&        )��P	�Y����A�
*

	conv_loss��>1K\�        )��P	d�����A�
*

	conv_loss,��>�̛         )��P	�����A�
*

	conv_loss@ԡ>4��        )��P	������A�
*

	conv_loss���>䵀�        )��P	0����A�
*

	conv_loss@��>���        )��P	�C����A�
*

	conv_lossݠ>-�e        )��P	�m����A�
*

	conv_loss�à>'ߩ�        )��P	�����A�
*

	conv_loss	��>TA�        )��P	������A�
*

	conv_loss�.�>!��        )��P	C����A�
*

	conv_loss���>��x        )��P	W0����A�
*

	conv_loss���>N�Ͽ        )��P	�[����A�
*

	conv_loss|�>���        )��P	������A�
*

	conv_loss٧>P�        )��P	�����A�
*

	conv_loss�ڠ> \�[        )��P	������A�
*

	conv_loss��>4kL        )��P	�����A�
*

	conv_lossPÞ>y�̇        )��P	�=����A�
*

	conv_loss�[�>��P        )��P	/i����A�
*

	conv_loss���>�ш�        )��P	�����A�
*

	conv_lossih�>`,�J        )��P	M�����A�
*

	conv_loss F�> �t�        )��P	;�����A�
*

	conv_loss�$�>!w޷        )��P	����A�
*

	conv_lossa�>��1�        )��P	-?����A�
*

	conv_loss���>�	��        )��P	�i����A�
*

	conv_loss�T�>rIf8        )��P	j�����A�
*

	conv_loss��>�3�        )��P	a�����A�
*

	conv_lossj"�>�@�*        )��P	������A�
*

	conv_loss"��>3��N        )��P	�����A�
*

	conv_loss̠>���+        )��P	(>����A�
*

	conv_loss��>�`9        )��P	�h����A�
*

	conv_loss�e�>�s��        )��P	j�����A�
*

	conv_loss���>���        )��P	������A�
*

	conv_losss��>�$Ss        )��P	z�����A�
*

	conv_lossH��>�r��        )��P	r'����A�
*

	conv_loss�;�>a�        )��P	�Q����A�
*

	conv_lossqC�>�͐�        )��P	S|����A�
*

	conv_loss��>�R7�        )��P	�����A�
*

	conv_loss�$�>��3D        )��P	������A�
*

	conv_loss��>�]�+        )��P	�����A�
*

	conv_lossl�>��8        )��P	�)����A�
*

	conv_loss���>��C�        )��P	V����A�
*

	conv_lossͮ�>R��        )��P	�����A�
*

	conv_loss��>Ǉ��        )��P	-�����A�
*

	conv_loss�
�>*݁�        )��P	�*����A�
*

	conv_lossPl�>Ӑ�        )��P	�U����A�
*

	conv_loss�R�>�N'�        )��P	'����A�
*

	conv_loss���>h=�{        )��P	������A�
*

	conv_loss�ݞ>C�k�        )��P	������A�
*

	conv_loss�t�>�50        )��P	������A�
*

	conv_lossJ�>��        )��P	�;����A�
*

	conv_loss�p�>U]�        )��P	<s����A�*

	conv_loss��>�r��        )��P	�����A�*

	conv_loss��>!~o        )��P	i�����A�*

	conv_loss��>A�3�        )��P		�����A�*

	conv_losskң>�1�        )��P	1����A�*

	conv_loss��>��K        )��P	�^����A�*

	conv_lossB�>μ
        )��P		�����A�*

	conv_lossU��>;2`a        )��P	r�����A�*

	conv_loss��>��        )��P	�����A�*

	conv_loss�-�>�i
&        )��P	
 ����A�*

	conv_loss:��>g("@        )��P	94 ����A�*

	conv_lossB �>�TP        )��P	P` ����A�*

	conv_loss���>��R        )��P	*� ����A�*

	conv_loss���>����        )��P	�� ����A�*

	conv_loss��>;ǡ        )��P	�� ����A�*

	conv_loss���>�i        )��P		!����A�*

	conv_loss�V�>�J��        )��P	�K!����A�*

	conv_loss���>|�F        )��P	�u!����A�*

	conv_loss���>,'��        )��P	��!����A�*

	conv_loss�>�H��        )��P	��!����A�*

	conv_loss֡>��        )��P	��!����A�*

	conv_loss���>s0��        )��P	""����A�*

	conv_loss�t�>��p        )��P	M"����A�*

	conv_loss���>����        )��P	�w"����A�*

	conv_loss�s�>lh�        )��P	�"����A�*

	conv_lossI�> jG        )��P	��"����A�*

	conv_loss�W�>J-In        )��P	]�"����A�*

	conv_loss�ߠ>�7|*        )��P	2 #����A�*

	conv_lossK=�>��ap        )��P	N#����A�*

	conv_loss��>5u         )��P	�y#����A�*

	conv_lossp��>�`B        )��P	ף#����A�*

	conv_loss��>c"�         )��P	��#����A�*

	conv_loss4f�>%��n        )��P	��#����A�*

	conv_loss��>Rg^        )��P	�$$����A�*

	conv_lossRi�>�ƣ�        )��P	Q$����A�*

	conv_lossŜ>�5{|        )��P	7|$����A�*

	conv_lossˠ�>�Zt>        )��P	t�$����A�*

	conv_loss���>�l��        )��P	6�$����A�*

	conv_loss�S�>�_�        )��P	
�$����A�*

	conv_lossY��>F�R�        )��P	V'%����A�*

	conv_loss�Ƞ>	.�        )��P	VS%����A�*

	conv_loss�}�>pȒ%        )��P	�{%����A�*

	conv_loss��>SF�        )��P	W�%����A�*

	conv_loss��>�yc�        )��P	��%����A�*

	conv_loss�E�>�k�        )��P	F&����A�*

	conv_loss���>qu̵        )��P	s8&����A�*

	conv_lossr�>�BnV        )��P	d&����A�*

	conv_loss���>��p        )��P	(�&����A�*

	conv_loss�џ>�l�_        )��P	6�&����A�*

	conv_loss�;�>���        )��P	��&����A�*

	conv_lossj �>��y�        )��P	"'����A�*

	conv_loss��>�C��        )��P	 X'����A�*

	conv_loss)ĝ>f�R        )��P	ֈ'����A�*

	conv_lossH�>��        )��P	C�'����A�*

	conv_loss�
�>V3��        )��P	-�'����A�*

	conv_loss���>ɛ�        )��P	Y(����A�*

	conv_loss��>U��B        )��P	_(����A�*

	conv_loss�Т>y ��        )��P	��(����A�*

	conv_loss��>��^�        )��P	ú(����A�*

	conv_loss�B�>_�v        )��P	��(����A�*

	conv_loss-�>J%�        )��P	�)����A�*

	conv_lossh��>y�V        )��P	�?)����A�*

	conv_loss.:�>*���        )��P	�k)����A�*

	conv_loss��>�<        )��P	L�)����A�*

	conv_loss�m�>���        )��P	d�)����A�*

	conv_loss&��>b��I        )��P	��)����A�*

	conv_loss���>\�        )��P	B*����A�*

	conv_loss�6�>s��K        )��P	-G*����A�*

	conv_loss���>ߟ�$        )��P	�r*����A�*

	conv_loss��>g/        )��P	��*����A�*

	conv_loss�h�>ވ�        )��P	�*����A�*

	conv_loss���>�M�b        )��P	�*����A�*

	conv_lossm�>���<        )��P	`"+����A�*

	conv_loss&ϝ>?�.        )��P	/M+����A�*

	conv_lossFH�>x�	        )��P	�|+����A�*

	conv_loss�>�Y�E        )��P	G�+����A�*

	conv_lossVb�>��!H        )��P	0�+����A�*

	conv_loss�7�><K��        )��P	|�+����A�*

	conv_loss��>���        )��P	(,����A�*

	conv_loss���>�2x[        )��P	�R,����A�*

	conv_loss��>�'m�        )��P	M~,����A�*

	conv_lossժ�>ļ,�        )��P	ק,����A�*

	conv_loss-�>&��        )��P	��,����A�*

	conv_losst]�>Ҝk        )��P	~�,����A�*

	conv_loss]j�>�m�E        )��P	�&-����A�*

	conv_lossŧ�>].�H        )��P	R-����A�*

	conv_loss
�>��t        )��P	�|-����A�*

	conv_losss�>�@dF        )��P	��-����A�*

	conv_lossd��>�+�        )��P	�-����A�*

	conv_loss�t�>��        )��P	`.����A�*

	conv_lossמ>���        )��P	o+.����A�*

	conv_loss/b�>�F�        )��P	hW.����A�*

	conv_loss��>�s�        )��P	��2����A�*

	conv_loss`~�>���        )��P	+-3����A�*

	conv_loss�D�>��&        )��P	�W3����A�*

	conv_lossh�>;��        )��P	%�3����A�*

	conv_loss�"�>RHjR        )��P	�3����A�*

	conv_lossU*�>�%ܫ        )��P	��3����A�*

	conv_loss��>�tzF        )��P	�4����A�*

	conv_lossej�>%U��        )��P	�+4����A�*

	conv_loss�_�>zr;�        )��P	2V4����A�*

	conv_losse��>��o        )��P	5�4����A�*

	conv_losss��>	�Ϭ        )��P	ƨ4����A�*

	conv_loss�ʞ>�b7�        )��P	�4����A�*

	conv_loss5�>�*�l        )��P	�5����A�*

	conv_loss(��>\j��        )��P	�?5����A�*

	conv_loss�M�>�&��        )��P	zi5����A�*

	conv_loss�>W���        )��P	t�5����A�*

	conv_loss��>4��m        )��P	�5����A�*

	conv_loss���>��PO        )��P	k�5����A�*

	conv_loss��>�&�        )��P	e6����A�*

	conv_loss�s�>�
Ko        )��P	�A6����A�*

	conv_loss�{�>
�        )��P	�j6����A�*

	conv_lossY,�>8`[�        )��P	��6����A�*

	conv_lossϚ�>�X}        )��P	��6����A�*

	conv_loss0K�>h�!        )��P	��6����A�*

	conv_loss��>�Q��        )��P	�7����A�*

	conv_loss���>5�@        )��P	77����A�*

	conv_losscj�>�v?�        )��P	�b7����A�*

	conv_lossQ�>0j �        )��P	�7����A�*

	conv_loss���>�?j        )��P	��7����A�*

	conv_loss�>��?�        )��P	��7����A�*

	conv_loss�+�>X��        )��P	�8����A�*

	conv_loss�p�>Ɨp        )��P	\@8����A�*

	conv_loss!Ϛ>��        )��P	#i8����A�*

	conv_loss�D�>����        )��P	��8����A�*

	conv_loss�ȟ>���j        )��P	��8����A�*

	conv_loss}8�>��z        )��P	��8����A�*

	conv_loss�O�>8�>�        )��P	�9����A�*

	conv_loss�Z�>zH?�        )��P	C99����A�*

	conv_loss׭�>��}        )��P	�a9����A�*

	conv_loss��>��r<        )��P	<�9����A�*

	conv_loss��>/�	        )��P	�9����A�*

	conv_loss^�>#z$g        )��P	
�9����A�*

	conv_loss��>&�[�        )��P	G	:����A�*

	conv_loss8�>~��*        )��P	�2:����A�*

	conv_loss��>�2�        )��P	�_:����A�*

	conv_lossa�>��U        )��P	��:����A�*

	conv_lossO��>�"^�        )��P	˱:����A�*

	conv_loss�m�>,}�        )��P	�:����A�*

	conv_loss�u�>�M        )��P	G;����A�*

	conv_loss�9�>3��Y        )��P	X.;����A�*

	conv_loss���>DO�        )��P	�Y;����A�*

	conv_lossv��>N�        )��P	��;����A�*

	conv_lossh�>0V�d        )��P	��;����A�*

	conv_lossPv�>2��        )��P	O�;����A�*

	conv_loss�>9.ӯ        )��P	�<����A�*

	conv_lossߡ�>���        )��P	�6<����A�*

	conv_loss�5�>�QA�        )��P	�`<����A�*

	conv_loss#��>�7��        )��P	Ð<����A�*

	conv_loss_Ú>6�W        )��P	��<����A�*

	conv_loss|�>ȱ�        )��P	��<����A�*

	conv_lossؒ�>�6k8        )��P	�+=����A�*

	conv_loss^��>�g��        )��P	BV=����A�*

	conv_loss�a�>W�K1        )��P	=�=����A�*

	conv_loss	��>�A�A        )��P	��=����A�*

	conv_loss��>D�b�        )��P	3�=����A�*

	conv_loss���>���        )��P	,>����A�*

	conv_loss ��>)�P�        )��P	�+>����A�*

	conv_loss0�>�ՙ-        )��P	sS>����A�*

	conv_loss˔�>W�h        )��P	�z>����A�*

	conv_loss��>ƻB\        )��P	Ѣ>����A�*

	conv_loss�V�>��        )��P	`�>����A�*

	conv_lossF)�>-r��        )��P	��>����A�*

	conv_loss��>��?�        )��P	7%?����A�*

	conv_loss���>YyS        )��P	�Q?����A�*

	conv_loss���>�|r�        )��P	�~?����A�*

	conv_loss���>Fr��        )��P	��?����A�*

	conv_loss��>��-        )��P	��?����A�*

	conv_loss�P�>e���        )��P	��?����A�*

	conv_lossL�>_�K�        )��P	�#@����A�*

	conv_loss1W�>�[G�        )��P	�L@����A�*

	conv_lossj`�>=��        )��P	&w@����A�*

	conv_lossڪ�>�7j�        )��P	ǟ@����A�*

	conv_lossmL�>~���        )��P	��@����A�*

	conv_lossFܛ>b�݉        )��P	��@����A�*

	conv_lossʪ�>E��        )��P	�'A����A�*

	conv_loss��>�B(        )��P	�RA����A�*

	conv_loss�ٛ>��ͱ        )��P	~�A����A�*

	conv_lossŗ�>��+        )��P	��A����A�*

	conv_loss�P�>}��G        )��P	c�A����A�*

	conv_lossT��>��K�        )��P	wB����A�*

	conv_loss� �>=���        )��P	�0B����A�*

	conv_loss�1�>���T        )��P	QZB����A�*

	conv_loss3:�>`�Y        )��P	�B����A�*

	conv_loss�h�>p���        )��P	M�B����A�*

	conv_loss�E�>,(��        )��P	H�B����A�*

	conv_loss���>�G�Z        )��P	t�B����A�*

	conv_loss���>*a]i        )��P	�(C����A�*

	conv_loss��>��v�        )��P	�QC����A�*

	conv_loss(��>9�?�        )��P	qzC����A�*

	conv_lossx��>B1<        )��P	��C����A�*

	conv_loss �>Ǽw9        )��P	X�C����A�*

	conv_loss�F�>��        )��P	�7E����A�*

	conv_lossg�>�#��        )��P	~^E����A�*

	conv_loss�;�>qK�        )��P	��E����A�*

	conv_loss%C�>~�7        )��P	T�E����A�*

	conv_loss��>4(�        )��P	��E����A�*

	conv_lossLB�>���        )��P	ZF����A�*

	conv_loss뚗>���        )��P	$/F����A�*

	conv_loss�$�>���y        )��P	�`F����A�*

	conv_loss�?�>�L�        )��P	$�F����A�*

	conv_loss�n�>YG        )��P	_�F����A�*

	conv_loss$h�>�uG5        )��P	G����A�*

	conv_lossJ��>��        )��P	�.G����A�*

	conv_loss���>��<�        )��P	�^G����A�*

	conv_loss���>N���        )��P	�G����A�*

	conv_loss���>���(        )��P	��G����A�*

	conv_loss8�>=�H�        )��P	��G����A�*

	conv_loss.��>�m6�        )��P	H����A�*

	conv_loss�x�>!:��        )��P	*0H����A�*

	conv_loss���>�Á        )��P	�ZH����A�*

	conv_loss^Ӝ>���?        )��P	܃H����A�*

	conv_loss$�>S��        )��P	F�H����A�*

	conv_loss5�>��        )��P	��H����A�*

	conv_loss�٘>{��        )��P	|�H����A�*

	conv_loss���>��"�        )��P	-I����A�*

	conv_loss,�>¾�u        )��P	�_I����A�*

	conv_loss˱�>GPJ        )��P	w�I����A�*

	conv_loss��>���        )��P	�I����A�*

	conv_loss R�>��)f        )��P	"�I����A�*

	conv_loss^j�>�#��        )��P	L	J����A�*

	conv_loss��>��	�        )��P	%2J����A�*

	conv_loss5��>��C        )��P	'lJ����A�*

	conv_loss��>��         )��P	��J����A�*

	conv_lossC8�>udv�        )��P	��J����A�*

	conv_lossߕ�>�k��        )��P	$�J����A�*

	conv_loss��>�G�        )��P	�K����A�*

	conv_losss��>��8        )��P	o6K����A�*

	conv_loss��>9>�        )��P	P^K����A�*

	conv_loss�k�>L!        )��P	�K����A�*

	conv_loss=R�>Rp��        )��P	��K����A�*

	conv_loss���>��        )��P	��K����A�*

	conv_loss�X�>���`        )��P	��K����A�*

	conv_loss�*�>�<�        )��P	�'L����A�*

	conv_loss}�>ᶾ�        )��P	FPL����A�*

	conv_lossrs�>s7J        )��P	yL����A�*

	conv_loss�>.��        )��P	M�L����A�*

	conv_loss���>lB4        )��P	O�L����A�*

	conv_lossgi�>�Y<�        )��P	��L����A�*

	conv_loss��>�zv        )��P	9M����A�*

	conv_loss���>%���        )��P	�HM����A�*

	conv_loss��>�W$�        )��P	,sM����A�*

	conv_loss�ט>HZ�        )��P	6�M����A�*

	conv_lossl�>���        )��P	�M����A�*

	conv_lossƍ�>ʝ�        )��P	��M����A�*

	conv_loss���>d�sb        )��P	�'N����A�*

	conv_losst��>m<�        )��P	�SN����A�*

	conv_lossF�>u-�        )��P	m�N����A�*

	conv_loss=�>��H        )��P	L�N����A�*

	conv_loss<F�>%�:�        )��P	��N����A�*

	conv_loss�Ҙ>V�#        )��P	O����A�*

	conv_lossJS�>�|5?        )��P	-O����A�*

	conv_loss&��>�F�        )��P	*WO����A�*

	conv_lossy�>����        )��P	x�O����A�*

	conv_loss��>����        )��P	r�O����A�*

	conv_loss銕>4�jf        )��P	��O����A�*

	conv_loss��>=%�k        )��P	�P����A�*

	conv_loss�v�>���        )��P	�<P����A�*

	conv_loss "�>�(7�        )��P	�eP����A�*

	conv_lossa �>��ޟ        )��P	�P����A�*

	conv_loss�>���h        )��P	�P����A�*

	conv_loss��>6�        )��P	p�P����A�*

	conv_loss�?�>��LU        )��P	�Q����A�*

	conv_loss�/�>ix(        )��P	EGQ����A�*

	conv_lossoW�>��%        )��P	-pQ����A�*

	conv_loss�Җ>��V�        )��P	,�Q����A�*

	conv_lossĥ�>���k        )��P	�Q����A�*

	conv_loss7@�>m��        )��P	��Q����A�*

	conv_loss�|�>>��        )��P	*R����A�*

	conv_loss�;�>���        )��P	;R����A�*

	conv_loss���>O,�        )��P	�aR����A�*

	conv_lossk��>�8�        )��P	��R����A�*

	conv_loss*�>kk�        )��P	��R����A�*

	conv_loss�g�>�#�        )��P	��R����A�*

	conv_loss�c�>��	        )��P	9S����A�*

	conv_loss�ɒ>d���        )��P	1-S����A�*

	conv_loss]�>�ܸ        )��P	�WS����A�*

	conv_loss���>5��        )��P	��S����A�*

	conv_lossǶ�>��N�        )��P	��S����A�*

	conv_loss�!�><C�        )��P	&�S����A�*

	conv_loss��>���        )��P	��S����A�*

	conv_loss���>�/}�        )��P	&&T����A�*

	conv_loss�?�>M�[�        )��P	5OT����A�*

	conv_loss�`�>��J�        )��P	�xT����A�*

	conv_lossN(�>!hG�        )��P	:�T����A�*

	conv_loss"N�>�k        )��P	>�T����A�*

	conv_loss���>T��        )��P	l�T����A�*

	conv_loss��>�a]�        )��P	QU����A�*

	conv_loss��>���        )��P	-IU����A�*

	conv_loss�٘>>�R�        )��P	�sU����A�*

	conv_loss]ʘ>09*2        )��P	7�U����A�*

	conv_loss��>j�>        )��P	4�U����A�*

	conv_loss.��>�pl^        )��P	j�U����A�*

	conv_losss��>���        )��P	m,V����A�*

	conv_loss���>.�G        )��P	�UV����A�*

	conv_losse)�>�[��        )��P	�V����A�*

	conv_loss%Ԗ>�,�        )��P	�V����A�*

	conv_lossw;�>A��.        )��P	4�V����A�*

	conv_loss�V�>�2Z�        )��P	~W����A�*

	conv_lossoŖ>f�O�        )��P	50W����A�*

	conv_loss�?�>�6E        )��P	�YW����A�*

	conv_losswN�>�c<�        )��P	��W����A�*

	conv_losswW�>C�-�        )��P	�W����A�*

	conv_lossG��> ~�N        )��P	��W����A�*

	conv_loss�ܚ>�=�3        )��P	�X����A�*

	conv_loss�>T�-        )��P	5X����A�*

	conv_lossq��>���        )��P	�bX����A�*

	conv_lossr��>�O��        )��P	M�X����A�*

	conv_lossӒ>�y<        )��P	̺X����A�*

	conv_loss��>��<        )��P	�X����A�*

	conv_loss���>����        )��P	�Y����A�*

	conv_loss�Ә>`�>        )��P	�AY����A�*

	conv_loss�>RXo�        )��P	�lY����A�*

	conv_loss��>���        )��P	6�Y����A�*

	conv_loss��>;��        )��P	��Y����A�*

	conv_loss���>h8-�        )��P	�Y����A�*

	conv_lossH	�>�4\�        )��P	�Z����A�*

	conv_loss���>�ZE        )��P	�AZ����A�*

	conv_lossv.�>�i�        )��P	QjZ����A�*

	conv_loss�4�>/�V�        )��P	ÓZ����A�*

	conv_loss��>Þ�        )��P	�Z����A�*

	conv_loss��>�{��        )��P	�Z����A�*

	conv_losslh�>(�:.        )��P	w[����A�*

	conv_loss�1�>��        )��P	 7[����A�*

	conv_lossѺ�>�][�        )��P	}b[����A�*

	conv_lossv�>�2z�        )��P	��[����A�*

	conv_loss�c�>?6        )��P	S�[����A�*

	conv_loss�>8��        )��P	��[����A�*

	conv_loss��>�O�"        )��P	�\����A�*

	conv_loss#�>�v��        )��P	@>\����A�*

	conv_loss�@�>�	(K        )��P	�i\����A�*

	conv_loss|�>�,��        )��P	��\����A�*

	conv_lossu)�>�__        )��P	��\����A�*

	conv_loss� �>��#�        )��P	��\����A�*

	conv_loss@��>=yB        )��P	�]����A�*

	conv_loss k�>*�P�        )��P	DG]����A�*

	conv_loss�S�>���        )��P	)r]����A�*

	conv_loss���>b��=        )��P	ڜ]����A�*

	conv_loss%�>yȟ@        )��P	%�]����A�*

	conv_loss�r�>VX/�        )��P	C�]����A�*

	conv_loss3��>�^��        )��P	F!^����A�*

	conv_loss،�>	�        )��P	�O^����A�*

	conv_loss���>*kr        )��P	.�^����A�*

	conv_lossi�>u�/%        )��P	|�^����A�*

	conv_loss�v�>���U        )��P	��^����A�*

	conv_loss��>f�w        )��P	X_����A�*

	conv_loss�H�>l�        )��P	�D_����A�*

	conv_loss��>䪘n        )��P	Aq_����A�*

	conv_loss�͕>35,\        )��P	N�_����A�*

	conv_loss}ߗ>L��        )��P	^�_����A�*

	conv_lossh��>��w�        )��P	�
`����A�*

	conv_lossx�>m-}�        )��P	5=`����A�*

	conv_lossZ(�>�CY        )��P	l`����A�*

	conv_lossñ�>Eh�        )��P	��`����A�*

	conv_loss��>��        )��P	��`����A�*

	conv_loss���>^$�        )��P	��`����A�*

	conv_loss���>6aT#        )��P	�'a����A�*

	conv_lossQ��>����        )��P	Sa����A�*

	conv_loss��>ѱ��        )��P	2�a����A�*

	conv_loss��>�         )��P	��a����A�*

	conv_loss��>��2        )��P	��a����A�*

	conv_loss$6�>���        )��P	�b����A�*

	conv_lossȩ�>��#        )��P	�-b����A�*

	conv_lossg�>�h        )��P	�]b����A�*

	conv_loss��>"m�        )��P	��b����A�*

	conv_lossɆ�>t�        )��P	��b����A�*

	conv_lossG�>�+@c        )��P	��b����A�*

	conv_loss�m�>��`�        )��P	�c����A�*

	conv_loss=��>O&        )��P	�Ac����A�*

	conv_loss�ٕ>b��I        )��P	-mc����A�*

	conv_loss� �>�N�K        )��P	A�c����A�*

	conv_loss+�>��        )��P	��c����A�*

	conv_lossC�>4�O�        )��P	=�c����A�*

	conv_loss=ŗ>���#        )��P	�d����A�*

	conv_loss���>-�        )��P	�Gd����A�*

	conv_lossV��>@�/         )��P	Hqd����A�*

	conv_loss�f�>=��9        )��P	��d����A�*

	conv_lossM�>�ɳu        )��P	j�d����A�*

	conv_lossY��>Bv)        )��P	��d����A�*

	conv_loss=.�>�t�        )��P	�e����A�*

	conv_loss���>�d�        )��P	�He����A�*

	conv_loss|�>�q�        )��P	+se����A�*

	conv_loss���>�Z��        )��P	I�e����A�*

	conv_loss��>�CD        )��P	f�e����A�*

	conv_loss]j�>Ge        )��P	q�e����A�*

	conv_losse�>
�c        )��P	�#f����A�*

	conv_loss��>D��6        )��P	UOf����A�*

	conv_loss!1�>#�        )��P	�zf����A�*

	conv_loss%,�>�@��        )��P	@�f����A�*

	conv_lossOw�>���e        )��P	�f����A�*

	conv_loss�f�>��yu        )��P	�g����A�*

	conv_lossc��>+ÐS        )��P	�0g����A�*

	conv_loss�ǖ>Q�39        )��P	��h����A�*

	conv_lossIg�>�L�q        )��P	D�h����A�*

	conv_lossL4�>=���        )��P	^i����A�*

	conv_loss���>t��B        )��P	�?i����A�*

	conv_loss��>���        )��P	�ni����A�*

	conv_lossW�>�]N�        )��P	��i����A�*

	conv_loss��>/���        )��P	��i����A�*

	conv_lossTB�>��]        )��P	u�i����A�*

	conv_losss��>1�٤        )��P	�)j����A�*

	conv_loss�}�>�j�
        )��P	�[j����A�*

	conv_lossiA�>H�	�        )��P	Ǎj����A�*

	conv_loss�5�> �2�        )��P	�j����A�*

	conv_loss/��>?аS        )��P	��j����A�*

	conv_lossW��>�I�        )��P	�
k����A�*

	conv_loss_f�>��'o        )��P	�5k����A�*

	conv_loss�ؐ>P���        )��P	�_k����A�*

	conv_loss�e�>10��        )��P	�k����A�*

	conv_losseΔ>���        )��P	��k����A�*

	conv_lossD��>CRh�        )��P	��k����A�*

	conv_loss�Y�>L ��        )��P	�l����A�*

	conv_loss��>�YF�        )��P	�<l����A�*

	conv_lossM�>��-�        )��P	�ql����A�*

	conv_loss!<�>2G��        )��P	 l����A�*

	conv_lossݯ�>턨%        )��P	G�l����A�*

	conv_lossJ�>���Z        )��P	��l����A�*

	conv_loss��>{��[        )��P	�$m����A�*

	conv_lossg��><`�	        )��P	�Qm����A�*

	conv_lossE�>���        )��P	{m����A�*

	conv_loss�|�>�r	        )��P	ʥm����A�*

	conv_loss�(�>i�G*        )��P	��m����A�*

	conv_lossl�>0��6        )��P	�m����A�*

	conv_loss���>���?        )��P	B(n����A�*

	conv_loss/�>/
%	        )��P	�Rn����A�*

	conv_loss��>��,9        )��P	p~n����A�*

	conv_loss���>�Rר        )��P	ܩn����A�*

	conv_loss
�>M�N%        )��P	��n����A�*

	conv_lossT�>�        )��P	��n����A�*

	conv_loss>�>#�!�        )��P	�)o����A�*

	conv_loss�Y�>֘ٱ        )��P	kSo����A�*

	conv_loss��>���        )��P	H}o����A�*

	conv_loss��>3`2V        )��P		�o����A�*

	conv_loss��>�2�s        )��P	�o����A�*

	conv_loss_��>ԡ�        )��P	:�o����A�*

	conv_lossJ��>�N�[        )��P	�&p����A�*

	conv_loss�ɒ>�T�%        )��P	:Qp����A�*

	conv_loss���>ʏ^        )��P	�}p����A�*

	conv_loss3�>\[�^        )��P	h�p����A�*

	conv_loss3Ɏ>�_?        )��P	5�p����A�*

	conv_loss��>��yU        )��P	��p����A�*

	conv_lossݯ�>����        )��P	�*q����A�*

	conv_loss��>_�        )��P	0hq����A�*

	conv_lossTő>d��r        )��P	�q����A�*

	conv_loss��>Iі+        )��P	j�q����A�*

	conv_loss���>�bˎ        )��P	��q����A�*

	conv_lossI7�>)f;U        )��P	�r����A�*

	conv_loss��>{UVW        )��P	h?r����A�*

	conv_loss�c�>*1ԯ        )��P	%ir����A�*

	conv_loss��>_�AU        )��P	%�r����A�*

	conv_loss�E�>V-J�        )��P	�r����A�*

	conv_loss��>��=�        )��P	D�r����A�*

	conv_loss���>f���        )��P	Cs����A�*

	conv_loss_��>Po�3        )��P	Ds����A�*

	conv_loss��>��`        )��P	 ns����A�*

	conv_lossw��>Iȡ�        )��P	��s����A�*

	conv_loss��>� ��        )��P	��s����A�*

	conv_loss}Y�>�]��        )��P	�s����A�*

	conv_loss��>��,�        )��P	)t����A�*

	conv_loss���>�d"        )��P	 Tt����A�*

	conv_loss(_�>�w(^        )��P	!�t����A�*

	conv_loss��>�0        )��P	W�t����A�*

	conv_loss��>�PŞ        )��P	��t����A�*

	conv_lossÖ�>�й�        )��P	��t����A�*

	conv_lossr
�>l,�        )��P	�)u����A�*

	conv_loss���>��V�        )��P	�Su����A�*

	conv_loss���>��gC        )��P	^u����A�*

	conv_lossRS�>Ro�t        )��P	s�u����A�*

	conv_loss/��>���        )��P	V�u����A�*

	conv_loss8�>Fnj�        )��P	�v����A�*

	conv_loss}R�>�P        )��P	N.v����A�*

	conv_lossя�>>�        )��P	�Xv����A�*

	conv_losswH�>?�G
        )��P	ӂv����A�*

	conv_loss)>�_        )��P	��v����A�*

	conv_loss�x�>���u        )��P	��v����A�*

	conv_lossK��>��w�        )��P	�w����A�*

	conv_loss�`�>��T        )��P	�,w����A�*

	conv_loss��>h�OM        )��P	Xw����A�*

	conv_lossWԒ>�(u�        )��P	�w����A�*

	conv_loss)�>��lb        )��P	I�w����A�*

	conv_loss$�>�B�        )��P	z�w����A�*

	conv_loss�W�>q�c�        )��P	sx����A�*

	conv_loss>A�>?P�:        )��P	�-x����A�*

	conv_lossEZ�>6G��        )��P	?Xx����A�*

	conv_loss��>���        )��P	��x����A�*

	conv_lossI�>�^�        )��P	W�x����A�*

	conv_loss4��>�        )��P	�x����A�*

	conv_lossD��>�J`�        )��P	y����A�*

	conv_lossj��>+��        )��P	�,y����A�*

	conv_loss�̏>��!�        )��P	�Wy����A�*

	conv_lossX~�>��He        )��P	i�y����A�*

	conv_loss��>1O\        )��P	�y����A�*

	conv_lossq�>]Bf�        )��P	��y����A�*

	conv_loss��>˙i�        )��P	�z����A�*

	conv_lossa�>�mwR        )��P	l=z����A�*

	conv_loss�Ő>��>�        )��P	�hz����A�*

	conv_lossJ�>�q1�        )��P		�z����A�*

	conv_lossY^�>�&-=        )��P	 �z����A�*

	conv_loss�v�>��*        )��P	x�z����A�*

	conv_loss�Q�>hD        )��P	�!{����A�*

	conv_loss;��>O�uw        )��P	�L{����A�*

	conv_loss���>�6�i        )��P	�y{����A�*

	conv_loss�%�>JED�        )��P	6�{����A�*

	conv_loss�G�>��
        )��P	��{����A�*

	conv_lossZŎ>�Ɏ�        )��P	�|����A�*

	conv_loss�׉>�D,!        )��P	7|����A�*

	conv_loss��>ǻ��        )��P	�`|����A�*

	conv_loss9؈>��_        )��P	؊|����A�*

	conv_loss��>        )��P	i�|����A�*

	conv_loss�E�>W��u        )��P	��|����A�*

	conv_loss�ߌ>��r�        )��P	}����A�*

	conv_loss�ۊ>�+XH        )��P	c9}����A�*

	conv_loss<[�>���d        )��P	c}����A�*

	conv_loss�>|�a�        )��P	�}����A�*

	conv_loss��>^��        )��P	޺}����A�*

	conv_loss���>�ѻ�        )��P	��}����A�*

	conv_lossؐ>��        )��P	s~����A�*

	conv_loss�J�>^ܾ        )��P	�@~����A�*

	conv_loss�ǋ>�8.�        )��P	�n~����A�*

	conv_loss���>Q[�        )��P	��~����A�*

	conv_loss�Z�>]Î        )��P	p�~����A�*

	conv_loss���>oQ^        )��P	��~����A�*

	conv_loss耎>|2��        )��P	�����A�*

	conv_loss܋>ꥲ�        )��P	zE����A�*

	conv_loss���>7k�s        )��P	p����A�*

	conv_loss%�>��9�        )��P	������A�*

	conv_loss�~�>�4��        )��P	������A�*

	conv_loss�É>��6        )��P	�����A�*

	conv_loss���>/[��        )��P	������A�*

	conv_loss�(�>���A        )��P	G�����A�*

	conv_lossT�>�&8        )��P	�q�����A�*

	conv_loss���>RHL        )��P	~������A�*

	conv_lossKp�>�r<        )��P	sǀ����A�*

	conv_loss _�>6�N        )��P	�񀡉��A�*

	conv_loss�R�>|���        )��P	j�����A�*

	conv_losscD�>4c        )��P	�I�����A�*

	conv_loss?܍>ᅇB        )��P	�s�����A�*

	conv_loss흐>86�6        )��P	}������A�*

	conv_loss�F�>����        )��P	́����A�*

	conv_lossZÑ>e�        )��P	X������A�*

	conv_loss��>~��k        )��P	�#�����A�*

	conv_loss�ǌ>�ʋ0        )��P	�P�����A�*

	conv_loss'�>��?        )��P	������A�*

	conv_loss�c�>��m         )��P	b������A�*

	conv_loss���>�֢        )��P	�������A�*

	conv_loss}@�>Dib<        )��P	�
�����A�*

	conv_loss�׋>�Y�
        )��P	9�����A�*

	conv_loss<��>��B        )��P	je�����A�*

	conv_losstC�>��8�        )��P	�������A�*

	conv_loss�Ȉ>� ��        )��P	�������A�*

	conv_loss3��>Z�8v        )��P	F⃡���A�*

	conv_loss���>��'@        )��P	������A�*

	conv_loss0L�>J͸        )��P	�>�����A�*

	conv_loss�U�>��σ        )��P	�k�����A�*

	conv_loss�݈> �}g        )��P	떄����A�*

	conv_loss�y�>��0�        )��P	-����A�*

	conv_loss�X�>��        )��P	=섡���A�*

	conv_loss�w�>�P�        )��P	G�����A�*

	conv_loss[^�>�\7�        )��P	$G�����A�*

	conv_loss�z�>Iv        )��P	�p�����A�*

	conv_loss^�>�,�Z        )��P	����A�*

	conv_loss`��>P�^6        )��P	�؅����A�*

	conv_loss�@�>�F�        )��P	������A�*

	conv_loss犉>��4�        )��P	n/�����A�*

	conv_lossۉ>��r�        )��P	NZ�����A�*

	conv_loss�2�> ��        )��P	%������A�*

	conv_lossm�>���        )��P	L������A�*

	conv_loss^��>LB�        )��P	�؆����A�*

	conv_loss��>��
        )��P	������A�*

	conv_loss$��>�l7        )��P	�0�����A�*

	conv_loss/�>��}�        )��P	_\�����A�*

	conv_loss�W�>GAC�        )��P	�������A�*

	conv_lossI�>ɵϴ        )��P	v������A�*

	conv_loss��>3�|�        )��P	�އ����A�*

	conv_loss�8�>���        )��P	������A�*

	conv_loss���>�F9A        )��P	8�����A�*

	conv_loss7�>����        )��P	�c�����A�*

	conv_loss���>4M�        )��P	�������A�*

	conv_loss �>9�Ri        )��P	�������A�*

	conv_losspP�>s��        )��P	 刡���A�*

	conv_lossÚ�>��3        )��P	������A�*

	conv_loss�)�>��+�        )��P	S<�����A�*

	conv_lossv�>/�D�        )��P	�g�����A�*

	conv_loss���>@\�        )��P	�������A�*

	conv_loss��>�ԇ�        )��P	&������A�*

	conv_loss|~�>2��        )��P	�퉡���A�*

	conv_loss0�>��m        )��P	������A�*

	conv_loss���>�lt        )��P	�H�����A�*

	conv_loss&�>_%\        )��P	�u�����A�*

	conv_loss��>�Z>        )��P	�������A�*

	conv_lossD=�>d]��        )��P	̊����A�*

	conv_lossB�>�V��        )��P	Չ�����A�*

	conv_loss僇>�(�s        )��P	�������A�*

	conv_lossF��>O۶<        )��P	� �����A�*

	conv_loss�X�>q���        )��P	�J�����A�*

	conv_loss]i�>��
        )��P	�u�����A�*

	conv_loss�]�>{���        )��P	A������A�*

	conv_lossA��>�4`�        )��P	ˑ����A�*

	conv_loss�w�>��J        )��P	�������A�*

	conv_loss� �>,��n        )��P	.�����A�*

	conv_lossb��>��.�        )��P	xL�����A�*

	conv_loss��>9HDj        )��P	iy�����A�*

	conv_loss��>.p=i        )��P	,������A�*

	conv_loss\܃>Sjo        )��P	Nܒ����A�*

	conv_loss��>��<        )��P	������A�*

	conv_loss�>�t        )��P	d,�����A�*

	conv_loss$��>� �@        )��P	�V�����A�*

	conv_lossԉ>?���        )��P	�����A�*

	conv_loss��>]�d�        )��P	Ҷ�����A�*

	conv_loss�7�>�$��        )��P	%ߓ����A�*

	conv_loss9y�>A�        )��P	�	�����A�*

	conv_loss���>��q�        )��P	.2�����A�*

	conv_lossƪ�>n+��        )��P	_�����A�*

	conv_loss�S�>j{aG        )��P	i������A�*

	conv_loss|Å>
�&i        )��P	0������A�*

	conv_loss�x�>��`�        )��P	�۔����A�*

	conv_loss�ш>r�ڎ        )��P	:�����A�*

	conv_loss$�>�۪        )��P	�:�����A�*

	conv_lossQĆ>⊿�        )��P	6e�����A�*

	conv_loss;��>�%        )��P	�������A�*

	conv_loss�܅><�n        )��P	�������A�*

	conv_loss���>~�~        )��P	�������A�*

	conv_lossuH�>#95Q        )��P	������A�*

	conv_losss��>����        )��P	w6�����A�*

	conv_loss�Յ>5Q1F        )��P	Fc�����A�*

	conv_loss٧�>'��        )��P	{������A�*

	conv_loss}J�>��        )��P	�������A�*

	conv_loss��>�Shz        )��P	�㖡���A�*

	conv_loss,�>{��        )��P	L�����A�*

	conv_lossn�>үg        )��P	4�����A�*

	conv_loss�'�>�C�        )��P	N`�����A�*

	conv_loss�0�>S��        )��P	������A�*

	conv_lossz��>�k��        )��P	�������A�*

	conv_loss��>@O        )��P	�ߗ����A�*

	conv_loss�ׅ>�I�        )��P	0�����A�*

	conv_loss:.�>s�         )��P	�4�����A�*

	conv_loss!/�>	v4        )��P	p\�����A�*

	conv_lossc~�>]�D�        )��P	�������A�*

	conv_losss�>��C        )��P	Ŭ�����A�*

	conv_loss���>�=�        )��P	�Ԙ����A�*

	conv_loss>��Z�        )��P	������A�*

	conv_loss��>���        )��P	%�����A�*

	conv_loss�΃>�m        )��P	�_�����A�*

	conv_loss�̌>L5�;        )��P	"������A�*

	conv_loss��>��/�        )��P	#������A�*

	conv_loss=#�>F�jb        )��P	ᙡ���A�*

	conv_loss��>��^�        )��P	������A�*

	conv_lossg}�>��l        )��P	:�����A�*

	conv_loss���>��        )��P	�d�����A�*

	conv_loss�|�>bt��        )��P	0������A�*

	conv_lossU��>�`�        )��P	e������A�*

	conv_losso��>�t��        )��P	�⚡���A�*

	conv_lossrb�>߳ߨ        )��P	F�����A�*

	conv_loss�/�>����        )��P	#E�����A�*

	conv_loss�a�>Psv�        )��P	/o�����A�*

	conv_lossu�>␞�        )��P	�������A�*

	conv_loss��>�F��        )��P	ě����A�*

	conv_loss!�>�E
        )��P	_𛡉��A�*

	conv_loss6H�>-%&y        )��P	W%�����A�*

	conv_loss�Z�>=�2<        )��P	VX�����A�*

	conv_lossJ$�>�\�"        )��P	������A�*

	conv_loss�x�>���S        )��P	�������A�*

	conv_loss�=�>�,��        )��P	朡���A�*

	conv_lossң�>���(        )��P	������A�*

	conv_loss=�>q�>        )��P	@�����A�*

	conv_loss�^�>}�`7        )��P	m�����A�*

	conv_loss.c�>���        )��P	�������A�*

	conv_loss�`�>�J��        )��P	�ȝ����A�*

	conv_loss_h}>d�Z        )��P	�������A�*

	conv_lossxm�>�>�        )��P	�!�����A�*

	conv_lossݜ�>��=        )��P	L�����A�*

	conv_loss�p�>�Q}o        )��P	Hx�����A�*

	conv_lossɅ>���        )��P	v������A�*

	conv_lossX��>�\dE        )��P	�͞����A�*

	conv_lossh�>�?��        )��P	�������A�*

	conv_loss>��>�8B>        )��P	�$�����A�*

	conv_loss���>���        )��P	�O�����A�*

	conv_loss�m�>V�~        )��P	�{�����A�*

	conv_losscO�>��*r        )��P	ŧ�����A�*

	conv_lossEڅ>8?�        )��P	gџ����A�*

	conv_loss�?}>�|�^        )��P	�������A�*

	conv_lossw��>}4>�        )��P	S&�����A�*

	conv_loss�׈>P>        )��P	�P�����A�*

	conv_loss�p�>x�
.        )��P	S|�����A�*

	conv_lossSc�>g)j]        )��P	�������A�*

	conv_lossQ|�>K�t        )��P	�Ԡ����A�*

	conv_losss�>Mrm�        )��P	�����A�*

	conv_loss�>����        )��P	�.�����A�*

	conv_loss�>��w�        )��P	Z�����A�*

	conv_lossȂ>�
�        )��P	H������A�*

	conv_lossߪ�>��+        )��P	v������A�*

	conv_loss@�>戳        )��P	Eݡ����A�*

	conv_loss�!�>�  ]        )��P	������A�*

	conv_loss�m�>�auo        )��P	^E�����A�*

	conv_lossx�>�{	        )��P	�r�����A�*

	conv_loss�J�>)�        )��P	����A�*

	conv_lossf��>L��        )��P	�΢����A�*

	conv_loss6~>9��        )��P	�������A�*

	conv_loss3I�>�&��        )��P	�)�����A�*

	conv_loss��>2�I        )��P	'`�����A�*

	conv_loss�e�>
�=�        )��P	������A�*

	conv_loss�>P�\+        )��P	������A�*

	conv_loss�c�>�*�R        )��P	�������A�*

	conv_lossv|>���        )��P	D&�����A�*

	conv_loss�Ё>��*�        )��P	^�����A�*

	conv_lossl*�>��\        )��P	q������A�*

	conv_loss,�>iݑ        )��P	������A�*

	conv_loss�τ>lP�        )��P	�ᤡ���A�*

	conv_loss��>�U�        )��P	������A�*

	conv_lossI��>�VBR        )��P	0<�����A�*

	conv_loss틃>���        )��P	7g�����A�*

	conv_loss�W�>�gp�        )��P	ݒ�����A�*

	conv_loss�E�>���P        )��P	(������A�*

	conv_loss:#�>�pk        )��P	祡���A�*

	conv_lossm�>���        )��P	"�����A�*

	conv_loss�'�>�`d        )��P	�?�����A�*

	conv_loss~t�>��N�        )��P	�o�����A�*

	conv_lossI�>I`.        )��P	o������A�*

	conv_loss�>{��        )��P	pҦ����A�*

	conv_lossw�>����        )��P	� �����A�*

	conv_loss^��>X{�N        )��P	�-�����A�*

	conv_loss>\o�        )��P	l[�����A�*

	conv_loss�>�/�        )��P	�������A�*

	conv_loss钀>�\Qf        )��P	�������A�*

	conv_lossr��>4k�y        )��P	�⧡���A�*

	conv_loss��>�8I        )��P	�����A�*

	conv_loss4�y>V�p=        )��P	A7�����A�*

	conv_loss�'�>5qD9        )��P	�b�����A�*

	conv_lossY}>���        )��P	4������A�*

	conv_loss�l�>�R�P        )��P	�������A�*

	conv_loss�~>��c        )��P	K稡���A�*

	conv_loss�p>��8�        )��P	������A�*

	conv_loss�a}>��ػ        )��P	�B�����A�*

	conv_loss��|>� Ǉ        )��P	�n�����A�*

	conv_loss�/�>�'�        )��P	�������A�*

	conv_lossA��>�(�        )��P	(Ʃ����A�*

	conv_loss}�>�t�t        )��P	�񩡉��A�*

	conv_loss�.�>��9        )��P	�����A�*

	conv_loss˄>+��"        )��P	{H�����A�*

	conv_loss�"�>���.        )��P	s�����A�*

	conv_loss	S�>�S[g        )��P	�������A�*

	conv_loss�!~>#�9�        )��P	�ʪ����A�*

	conv_lossV��>�f�        )��P	������A�*

	conv_loss��x>����        )��P	�6�����A�*

	conv_loss�E�>� ؈        )��P	8d�����A�*

	conv_loss*��>�-�'        )��P	Y������A�*

	conv_loss5ʂ>�I�r        )��P	񾫡���A�*

	conv_lossQ$�>k��        )��P	�뫡���A�*

	conv_loss|>|�]        )��P	#�����A�*

	conv_lossEށ>�eO'        )��P	�C�����A�*

	conv_loss�ۄ>����        )��P	�o�����A�*

	conv_loss�c�>1���        )��P	@������A�*

	conv_loss�z>
*�A        )��P	ᬡ���A�*

	conv_loss�m�>�ϩ        )��P	������A�*

	conv_loss���>�}�c        )��P	�:�����A�*

	conv_loss��>ҳL2        )��P	�f�����A�*

	conv_loss�E�>���$        )��P	������A�*

	conv_loss�Zz>��2        )��P	�έ����A�*

	conv_loss��>�LC�        )��P	O������A�*

	conv_loss[6�>Q*ɦ        )��P	�/�����A�*

	conv_loss�q�>��i�        )��P	�]�����A�*

	conv_lossNp{>��{�        )��P	u������A�*

	conv_loss�ƀ>���        )��P	ٵ�����A�*

	conv_loss���>��y�        )��P	`஡���A�*

	conv_loss�+�>�:�        )��P	�����A�*

	conv_lossiEu>�$��        )��P	69�����A�*

	conv_loss^�>�D�        )��P	e�����A�*

	conv_loss5��>�+i�        )��P	�������A�*

	conv_lossE�x>t�.r        )��P	B������A�*

	conv_loss���>L�$G        )��P	-篡���A�*

	conv_loss<W�>�_�        )��P	������A�*

	conv_lossfn~>m�?        )��P	?�����A�*

	conv_loss>� &<        )��P	!i�����A�*

	conv_loss�r~>�<        )��P	ݖ�����A�*

	conv_loss�>�r��        )��P	�°����A�*

	conv_loss�S�>l�)�        )��P	������A�*

	conv_loss(�>�)$�        )��P	(�����A�*

	conv_loss홁>��ʾ        )��P	�C�����A�*

	conv_loss�g�> �S        )��P	�o�����A�*

	conv_loss��t>�p�        )��P	囱����A�*

	conv_loss�6�>%���        )��P	�Ʊ����A�*

	conv_loss��>"��        )��P	:򱡉��A�*

	conv_lossxz>�"�        )��P	������A�*

	conv_loss"}>��\�        )��P	K�����A�*

	conv_loss1y�>����        )��P	�u�����A�*

	conv_loss��{>}�]�        )��P	
������A�*

	conv_loss\w�>0G��        )��P	�ɲ����A�*

	conv_loss�>a$-�        )��P	�������A�*

	conv_loss�ۀ>��u        )��P	�"�����A�*

	conv_loss?Ӏ>0"��        )��P	XN�����A�*

	conv_loss�s>[KI        )��P	�~�����A�*

	conv_loss2��>6v$0        )��P	3������A�*

	conv_lossك>��        )��P	������A�*

	conv_loss[v>mzv�        )��P	:G�����A�*

	conv_lossщu>�we�        )��P	ps�����A�*

	conv_loss�v>K�x�        )��P	ާ�����A�*

	conv_lossYw>/	�m        )��P	�嵡���A�*

	conv_loss	xq>y.�W        )��P	������A�*

	conv_losss�>l�6g        )��P	�D�����A�*

	conv_loss�>c�hy        )��P	�t�����A�*

	conv_lossǂ>TFFu        )��P	������A�*

	conv_lossj�x>��o#        )��P	�ɶ����A�*

	conv_lossTր>�A��        )��P	K�����A�*

	conv_loss��p>�)_        )��P	v0�����A�*

	conv_loss��~>mR�         )��P	R[�����A�*

	conv_loss�mn>/\z�        )��P	䅷����A�*

	conv_loss~>j��c        )��P	�������A�*

	conv_loss?
s>z�.        )��P	�۷����A�*

	conv_loss>�s>xO�K        )��P			�����A�*

	conv_loss��u>CL܊        )��P	�9�����A�*

	conv_loss�q�>j$p�        )��P	9s�����A�*

	conv_loss{r>A���        )��P	�������A�*

	conv_lossֵ�>�D�        )��P	�θ����A�*

	conv_loss(q>0�=        )��P	������A�*

	conv_loss��}>G�|^        )��P	4%�����A�*

	conv_loss�\�>�l'V        )��P	rQ�����A�*

	conv_loss}u>.!�        )��P	?|�����A�*

	conv_loss�(z>���B        )��P	�������A�*

	conv_lossV�p>�	{�        )��P	�ѹ����A�*

	conv_loss��k>�^H}        )��P	F������A�*

	conv_loss�v>y7�e        )��P	�(�����A�*

	conv_loss%�j>%��        )��P	U�����A�*

	conv_loss�>x6�        )��P	m�����A�*

	conv_loss��y>׿��        )��P	)������A�*

	conv_loss��>���Y        )��P	D׺����A�*

	conv_loss��h>vMV�        )��P	U�����A�*

	conv_lossz�>~��        )��P	0�����A�*

	conv_lossq�x>��J        )��P	\�����A�*

	conv_losss�>(cEo        )��P		������A�*

	conv_loss��w>%��E        )��P	�������A�*

	conv_loss�Kr>u�v        )��P	h仡���A�*

	conv_loss�e|>�m��        )��P	3�����A�*

	conv_loss�z>|ת�        )��P	N:�����A�*

	conv_lossRl><;a�        )��P	�h�����A�*

	conv_loss�m>f'        )��P	+������A�*

	conv_loss��x>�O�        )��P	u������A�*

	conv_loss�l}>��N        )��P	�������A�*

	conv_loss;�|>c��        )��P	������A�*

	conv_loss�s>0��;        )��P	rC�����A�*

	conv_loss�|>�g�i        )��P	pq�����A�*

	conv_losss�x>m� A        )��P	�������A�*

	conv_loss|?}>-T{        )��P	Iƽ����A�*

	conv_loss�r{>�p�        )��P	������A�*

	conv_loss�w>��#        )��P	"0�����A�*

	conv_loss�;r>�,MB        )��P	�Z�����A�*

	conv_loss�&�>2�        )��P	������A�*

	conv_loss��d>(�$        )��P	�ž����A�*

	conv_lossL6w>N�f�        )��P		������A�*

	conv_loss�u>D��        )��P	f"�����A�*

	conv_lossBax>~O(�        )��P	{N�����A�*

	conv_lossFM{>_/�        )��P	}������A�*

	conv_loss{�t>��6        )��P	�������A�*

	conv_loss#�p>ܖ��        )��P	+󿡉��A�*

	conv_loss�z>MqΡ        )��P	$!�����A�*

	conv_lossB�>oTA�        )��P	AM�����A�*

	conv_loss/�m>���        )��P	9~�����A�*

	conv_loss(�z>ɭB        )��P	x������A�*

	conv_lossE:>	�C        )��P	�������A�*

	conv_loss�(r>̵R�        )��P	������A�*

	conv_loss�n>m        )��P	�2�����A�*

	conv_loss�Kq>p�c�        )��P	�_�����A�*

	conv_loss�u|>�$��        )��P	�������A�*

	conv_loss��>KRk�        )��P	�������A�*

	conv_loss��>�r+        )��P	$������A�*

	conv_loss��w>��A	        )��P	v¡���A�*

	conv_loss~�{>���$        )��P	�E¡���A�*

	conv_loss�u>g��t        )��P	�{¡���A�*

	conv_lossuVn>�$        )��P	��¡���A�*

	conv_losssyk>�%�        )��P	%�¡���A�*

	conv_lossAw>��4        )��P	oá���A�*

	conv_losse�u>����        )��P	{,á���A�*

	conv_lossAnr>�
3m        )��P	Zá���A�*

	conv_lossK�{>�@��        )��P	b�á���A�*

	conv_loss~Ui>!�6�        )��P	��á���A�*

	conv_loss�x>� ��        )��P	i�á���A�*

	conv_loss�{>�C�        )��P	�ġ���A�*

	conv_lossly>�-i        )��P	@:ġ���A�*

	conv_loss��o>��QW        )��P	+fġ���A�*

	conv_lossM/w>��tO        )��P	��ġ���A�*

	conv_lossz�k>p4�<        )��P	��ġ���A�*

	conv_loss��j>,�S�        )��P	��ġ���A�*

	conv_loss�m>��        )��P	�š���A�*

	conv_loss�Ks>��G        )��P	�@š���A�*

	conv_loss��d>#0&        )��P	]lš���A�*

	conv_lossW�]>�p>        )��P	.�š���A�*

	conv_loss�
s>N��        )��P	��š���A�*

	conv_lossskp>��        )��P	��š���A�*

	conv_loss�n>�$        )��P	ơ���A�*

	conv_loss�#w>～T        )��P	�Fơ���A�*

	conv_loss��g>�|%�        )��P	�rơ���A�*

	conv_lossL\p>�D��        )��P	�ơ���A�*

	conv_loss�%f>p	�`        )��P	��ơ���A�*

	conv_loss��p>�r�        )��P	zǡ���A�*

	conv_loss��t>[S��        )��P	3-ǡ���A�*

	conv_loss�k>���c        )��P	&Wǡ���A�*

	conv_lossҢy>]�t�        )��P	,�ǡ���A�*

	conv_loss��>z^u^        )��P	�ǡ���A�*

	conv_loss�@o>��        )��P	�ǡ���A�*

	conv_loss6xr>L�f�        )��P	.ȡ���A�*

	conv_loss��p>����        )��P	9ȡ���A�*

	conv_loss��r>���        )��P	�dȡ���A�*

	conv_loss�c>d�7-        )��P	��ȡ���A�*

	conv_lossar>��'�        )��P	��ȡ���A�*

	conv_loss��n>�o�,        )��P	��ȡ���A�*

	conv_loss�Dr>7�s        )��P	=)ɡ���A�*

	conv_loss=�m>@|��        )��P	�Sɡ���A�*

	conv_loss�.s>�[5�        )��P	�~ɡ���A�*

	conv_loss��d>E�n6        )��P	�ɡ���A�*

	conv_loss�fd>��פ        )��P	��ɡ���A�*

	conv_losss}x>(rm        )��P	rʡ���A�*

	conv_loss��t>���        )��P	bFʡ���A�*

	conv_loss�k>�l�&        )��P	rʡ���A�*

	conv_loss �n>k��        )��P	ޝʡ���A�*

	conv_loss8�f>�
��        )��P	��ʡ���A�*

	conv_lossj�i>F&{        )��P	��ʡ���A�*

	conv_loss��b>=��        )��P	�$ˡ���A�*

	conv_lossj�f>d�        )��P	=Pˡ���A�*

	conv_loss�Km>��]        )��P	�{ˡ���A�*

	conv_loss.�t>��E        )��P	l�ˡ���A�*

	conv_lossrHg>�        )��P	��ˡ���A�*

	conv_loss
ms>q���        )��P	�̡���A�*

	conv_lossޞh>����        )��P	[.̡���A�*

	conv_loss��[>�Je�        )��P	�X̡���A�*

	conv_loss�0a>����        )��P	q�̡���A�*

	conv_lossf�n>L��        )��P	"�̡���A�*

	conv_loss;m>�܃�        )��P	%�̡���A�*

	conv_loss�s>�}(�        )��P	͡���A�*

	conv_loss�^p>�X        )��P	p2͡���A�*

	conv_loss�Gl>:KR        )��P	�\͡���A�*

	conv_loss��u>B�        )��P	��͡���A�*

	conv_loss�j>Y�t+        )��P	��͡���A�*

	conv_loss��_>�h��        )��P	��͡���A�*

	conv_loss�Dr>i�#�        )��P	/Ρ���A�*

	conv_lossGpt>D�V�        )��P	�9Ρ���A�*

	conv_loss�5a>"�0        )��P	SeΡ���A�*

	conv_loss��l>/%��        )��P	ԐΡ���A�*

	conv_loss�<f>��        )��P	��Ρ���A�*

	conv_loss"�e>=g��        )��P	\�Ρ���A�*

	conv_loss�=f>"�6�        )��P	�ϡ���A�*

	conv_lossW&n> �W�        )��P	�?ϡ���A�*

	conv_loss�`>����        )��P	�iϡ���A�*

	conv_loss��e>���4        )��P	��ϡ���A�*

	conv_losstCa>��Y"        )��P	b�ϡ���A�*

	conv_loss7k>�/�        )��P	��ϡ���A�*

	conv_loss|�[>`��        )��P	�(С���A�*

	conv_loss�Qd>�C�f        )��P	�TС���A�*

	conv_loss�pl>	_��        )��P	�С���A�*

	conv_loss��k>6gR        )��P	��С���A�*

	conv_loss �[>�"�        )��P	��С���A�*

	conv_loss��h>����        )��P	�ѡ���A�*

	conv_loss~�]>c�K�        )��P	�Hѡ���A�*

	conv_loss�pk>O��L        )��P	-ѡ���A�*

	conv_loss]2i>��y�        )��P	�ѡ���A�*

	conv_loss"�_>��(�        )��P	`�ѡ���A�*

	conv_lossp�f>�v��        )��P	�ҡ���A�*

	conv_loss�\>���H        )��P	Kҡ���A�*

	conv_loss��g>Kg�8        )��P	vҡ���A�*

	conv_loss�8j>���q        )��P	��ҡ���A�*

	conv_loss!/}>���p        )��P	��ҡ���A�*

	conv_loss��`>Aes�        )��P	y�ҡ���A�*

	conv_loss"�\>�I��        )��P	1'ӡ���A�*

	conv_loss0�k>\ҝ~        )��P	GRӡ���A�*

	conv_lossb�f>\1S�        )��P	~ӡ���A�*

	conv_loss
�h>��C        )��P	��ӡ���A�*

	conv_losss�k>�	�        )��P	��ӡ���A�*

	conv_loss/9g>�7�        )��P	
ԡ���A�*

	conv_loss�se>�Q�        )��P	�9ԡ���A�*

	conv_lossu�d>�$��        )��P	�fԡ���A�*

	conv_loss�xa>��i        )��P	B�ԡ���A�*

	conv_loss]�j>�ꗤ        )��P	1�ԡ���A�*

	conv_loss�za>��(�        )��P	J�ԡ���A�*

	conv_lossF�Q>`��        )��P	�ա���A�*

	conv_lossFcm>)��        )��P	nHա���A�*

	conv_lossXNU>H3��        )��P	6tա���A�*

	conv_loss�r>΄a6        )��P	.�ա���A�*

	conv_loss�fp>�]�~        )��P	+�ա���A�*

	conv_loss�mh>��!�        )��P	,�ա���A�*

	conv_loss��_>(���        )��P	d#֡���A�*

	conv_loss�+c>�}B        )��P	=P֡���A�*

	conv_loss�_k>�!�&        )��P	4֡���A�*

	conv_loss�+i>���        )��P	|�֡���A�*

	conv_lossA�l>�)�        )��P	��֡���A�*

	conv_losss�_>���[        )��P	��֡���A�*

	conv_loss�,p>ڽޥ        )��P	?)ס���A�*

	conv_lossf�_>9�tN        )��P	jSס���A�*

	conv_loss��g>�L��        )��P	p}ס���A�*

	conv_lossͭ[>i�ƍ        )��P	<�ס���A�*

	conv_loss�$m>
��        )��P	��ס���A�*

	conv_loss�#d>�9u�        )��P	��ס���A�*

	conv_losst�c>@�`�        )��P	/ء���A�*

	conv_loss��a>$��        )��P	�Yء���A�*

	conv_lossvW>β�"        )��P	̓ء���A�*

	conv_loss+�_>16��        )��P	��١���A�*

	conv_loss�ge>S��K        )��P	G'ڡ���A�*

	conv_loss0�c>Z�<        )��P	9Vڡ���A�*

	conv_lossD#V>���v        )��P	?�ڡ���A�*

	conv_loss�e>�N3        )��P	��ڡ���A�*

	conv_loss�d>_�v        )��P	�ڡ���A�*

	conv_loss&-\>��        )��P	|ۡ���A�*

	conv_loss�a>(�p�        )��P	?ۡ���A�*

	conv_loss��c>i�        )��P	}tۡ���A�*

	conv_loss}�^>��C        )��P	o�ۡ���A�*

	conv_lossZ>|�NR        )��P		�ۡ���A�*

	conv_loss9_>,��n        )��P	��ۡ���A�*

	conv_loss@i>���        )��P	4;ܡ���A�*

	conv_loss��R>��ם        )��P	-jܡ���A�*

	conv_loss�d>ܓ>        )��P	��ܡ���A�*

	conv_loss� M>�oީ        )��P	(�ܡ���A�*

	conv_loss��d>��i�        )��P	��ܡ���A�*

	conv_loss�]>ݼ>�        )��P	Mݡ���A�*

	conv_loss޴d>z�	�        )��P	Dݡ���A�*

	conv_loss4_>H�>�        )��P	eqݡ���A�*

	conv_loss_2k>��        )��P	ٛݡ���A�*

	conv_loss<�`>��y�        )��P	�ݡ���A�*

	conv_loss(�f>�T�R        )��P	��ݡ���A�*

	conv_loss�^>=�^,        )��P	0ޡ���A�*

	conv_loss:>^>�9��        )��P	�\ޡ���A�*

	conv_loss�&U>v�~�        )��P	I�ޡ���A�*

	conv_loss/Vb>e̖        )��P	��ޡ���A�*

	conv_lossڨ\>�L        )��P	��ޡ���A�*

	conv_loss8�a>H��\        )��P	7ߡ���A�*

	conv_loss\�O>�\Z        )��P	#8ߡ���A�*

	conv_loss�p\>B\�        )��P	2dߡ���A�*

	conv_lossDf>.�        )��P	F�ߡ���A�*

	conv_loss�X>�U�        )��P	��ߡ���A�*

	conv_loss��X>	rh�        )��P	�ߡ���A�*

	conv_lossTQ>=7�        )��P	*ࡉ��A�*

	conv_loss�W>J=�        )��P	�>ࡉ��A�*

	conv_lossr4k>��³        )��P	�jࡉ��A�*

	conv_lossÙf>V��z        )��P	,�ࡉ��A�*

	conv_lossÂ_>d���        )��P	��ࡉ��A�*

	conv_loss��T>c�{        )��P	>�ࡉ��A�*

	conv_loss��W>`��^        )��P	�ᡉ��A�*

	conv_loss?n>C�G        )��P	�Gᡉ��A�*

	conv_lossd�e>�[[�        )��P	�rᡉ��A�*

	conv_loss�]>���(        )��P	��ᡉ��A�*

	conv_loss�t>�T�        )��P	��ᡉ��A�*

	conv_loss.?b>��+�        )��P	��ᡉ��A�*

	conv_loss�R>���        )��P	⡉��A�*

	conv_lossh�G>��#�        )��P	VH⡉��A�*

	conv_loss�pX>�v�        )��P	u⡉��A�*

	conv_loss#�T>��J�        )��P	�⡉��A�*

	conv_loss�Ce>�w��        )��P	w�⡉��A�*

	conv_loss��O>y]a<        )��P	�	㡉��A�*

	conv_loss�{W>��1l        )��P	i8㡉��A�*

	conv_loss 4X>��s        )��P	�c㡉��A�*

	conv_loss��a>���        )��P	o�㡉��A�*

	conv_lossv�O>�!-�        )��P	y�㡉��A�*

	conv_loss^>�+}�        )��P	W�㡉��A�*

	conv_lossx�Z>Iæ        )��P	�䡉��A�*

	conv_loss(�_>�i=�        )��P	C䡉��A�*

	conv_loss{.S>��{        )��P	�}䡉��A�*

	conv_loss�J>I�"�        )��P	w�䡉��A�*

	conv_loss)�Z>�wJ        )��P	��䡉��A�*

	conv_loss�-V>;�ԅ        )��P	V塉��A�*

	conv_loss3�P>��         )��P	�4塉��A�*

	conv_loss%cW>�JH        )��P	9e塉��A�*

	conv_loss�tD>�t        )��P	��塉��A�*

	conv_loss��V>:��@        )��P	!�塉��A�*

	conv_loss�IS>
9�        )��P	�塉��A�*

	conv_loss"$\>���        )��P	�$桉��A�*

	conv_loss�]>F�        )��P	�P桉��A�*

	conv_loss5�]>]��g        )��P	�}桉��A�*

	conv_lossem>h�3W        )��P	
�桉��A�*

	conv_loss�oJ>���S        )��P	��桉��A�*

	conv_lossp�\>��P�        )��P	� 硉��A�*

	conv_loss$j>��2        )��P	y,硉��A�*

	conv_loss�c]>�j�        )��P	LW硉��A�*

	conv_loss. ]>y�6@        )��P	��硉��A�*

	conv_loss�U>=�ټ        )��P	!�硉��A�*

	conv_loss�\>֏        )��P	�硉��A�*

	conv_loss!7[>|�
f        )��P	衉��A�*

	conv_loss8t^>@C�$        )��P	�4衉��A�*

	conv_loss��Z>�	�        )��P	m`衉��A�*

	conv_loss�7R>	��i        )��P	|�衉��A�*

	conv_loss�\>*��        )��P	�衉��A�*

	conv_loss��U>Տ
H        )��P	��衉��A�*

	conv_loss�,Q>BY�e        )��P	�顉��A�*

	conv_loss��T>�9��        )��P	:<顉��A�*

	conv_loss�O>�)7        )��P	Uk顉��A�*

	conv_loss�FK>u��S        )��P	Ж顉��A�*

	conv_losse�M>u��        )��P	&�顉��A�*

	conv_loss��V>�n$w        )��P	-�顉��A�*

	conv_loss��[>�</        )��P	iꡉ��A�*

	conv_lossS�H>�<��        )��P	<Kꡉ��A�*

	conv_lossķT>����        )��P	�vꡉ��A�*

	conv_loss�Z>����        )��P	*�ꡉ��A�*

	conv_loss�6Z>J�>        )��P	��ꡉ��A�*

	conv_loss�F>v;�        )��P	=�ꡉ��A�*

	conv_loss�$W>d���        )��P	x&롉��A�*

	conv_losscAW>�v��        )��P	fS롉��A�*

	conv_loss��X>���        )��P	����A�*

	conv_loss�BV>C�7        )��P	�-𡉙�A�*

	conv_loss��Y>�ψ        )��P	�V𡉙�A�*

	conv_loss�P>�p�0        )��P	V�𡉙�A�*

	conv_lossQ>�9        )��P	��𡉙�A�*

	conv_loss��T>�D�        )��P	��𡉙�A�*

	conv_loss�S>�bv        )��P	��𡉙�A�*

	conv_loss�NN>���        )��P	n%񡉙�A�*

	conv_losscT>��1p        )��P	�N񡉙�A�*

	conv_loss!]S>я@        )��P	Uv񡉙�A�*

	conv_loss~ZS>HbQo        )��P	͞񡉙�A�*

	conv_lossO�X>�<�        )��P	�񡉙�A�*

	conv_loss�V>�        )��P	�򡉙�A�*

	conv_lossI>�E�        )��P	�A򡉙�A�*

	conv_loss�'T>,��        )��P	[k򡉙�A�*

	conv_lossP�H>�W�        )��P	w�򡉙�A�*

	conv_loss�n=>:��        )��P	�򡉙�A�*

	conv_loss�P>X��f        )��P	��򡉙�A�*

	conv_loss�U>�˹        )��P	Q󡉙�A�*

	conv_lossUgR>�qO        )��P	�;󡉙�A�*

	conv_loss@ME>c�G        )��P	�g󡉙�A�*

	conv_loss3,^>"�        )��P	��󡉙�A�*

	conv_loss�R>qժ�        )��P	v�󡉙�A�*

	conv_lossuS>A��!        )��P	��󡉙�A�*

	conv_loss�I>�c�        )��P	[�����A�*

	conv_lossw�R>oT�        )��P	u0�����A�*

	conv_loss��V>�w.4        )��P	�Z�����A�*

	conv_loss�X>38��        )��P	�������A�*

	conv_loss�?F>�&ݥ        )��P	a������A�*

	conv_lossJ)M>K��        )��P	#������A�*

	conv_loss�-M>G�R        )��P	b�����A�*

	conv_loss`�Y>+�        )��P	w:�����A�*

	conv_loss��B>����        )��P	�e�����A�*

	conv_losso�D>\i��        )��P	�������A�*

	conv_lossR�J>�?�        )��P	͸�����A�*

	conv_loss�m^>��z�        )��P	�������A�*

	conv_loss0�I>̐j�        )��P	h�����A�*

	conv_lossһ?>۰[        )��P	�7�����A�*

	conv_loss@`>=��        )��P	�a�����A�*

	conv_lossR�K>&)�        )��P	������A�*

	conv_lossQUO>֗n�        )��P	ܶ�����A�*

	conv_losskN>�l�        )��P	������A�*

	conv_loss�gN>sP��        )��P	�	�����A�*

	conv_lossyH>J'�?        )��P	�1�����A�*

	conv_loss��E>����        )��P	Y�����A�*

	conv_loss�a>���        )��P	Z������A�*

	conv_lossL&V>Oh;�        )��P	ڭ�����A�*

	conv_loss�D>w���        )��P	�������A�*

	conv_loss2�R>V�'        )��P	�������A�*

	conv_loss5>9K�        )��P	�(�����A�*

	conv_lossJ�F>Ñ/�        )��P	�U�����A�*

	conv_loss��R>�1�=        )��P	-������A�*

	conv_loss�kC>���S        )��P	u������A�*

	conv_lossgqC>�V�        )��P	�������A�*

	conv_loss�R>H��        )��P	������A�*

	conv_loss�oK>cP�Q        )��P	�6�����A�*

	conv_loss�G>�l        )��P	}a�����A�*

	conv_loss�@>��Y�        )��P	^������A�*

	conv_loss�fG>Dxn�        )��P	�������A�*

	conv_lossI�C>�}qz        )��P	f������A�*

	conv_loss�YI>w�7�        )��P	%�����A�*

	conv_loss?�R>��2!        )��P	�J�����A�*

	conv_loss �E>����        )��P	vt�����A�*

	conv_lossa�H>ٛ��        )��P	H������A�*

	conv_lossn�<>���        )��P	�������A�*

	conv_loss�O>>>��        )��P	l������A�*

	conv_lossyO]>�KX�        )��P	�$�����A�*

	conv_loss�T>���        )��P	�N�����A�*

	conv_loss"]:>�D        )��P	rz�����A�*

	conv_loss+L>�C        )��P	P������A�*

	conv_lossW�Q>I��        )��P	������A�*

	conv_loss�I>��ɉ        )��P	�������A�*

	conv_lossHBN>�X�J        )��P		'�����A�*

	conv_loss׃T>h�)        )��P	[S�����A�*

	conv_loss`_O>վ�        )��P	
}�����A�*

	conv_lossJ>,��        )��P	�������A�*

	conv_lossj�C>���        )��P	�������A�*

	conv_loss�QL>���d        )��P	{������A�*

	conv_loss�AF>;��l        )��P	�*�����A�*

	conv_loss5�E>��U�        )��P	1T�����A�*

	conv_loss�G>��`<        )��P	~�����A�*

	conv_loss��G>��[        )��P	i������A�*

	conv_loss]GU>��        )��P	?������A�*

	conv_lossՕA>�x        )��P	�����A�*

	conv_loss�N@>�/Wo        )��P	,�����A�*

	conv_loss-Q>�Id        )��P	�X�����A�*

	conv_loss��H>���7        )��P	ق�����A�*

	conv_loss��Q>$"        )��P	������A�*

	conv_losse�:>�H-?        )��P	A������A�*

	conv_loss��A>�        )��P	)�����A�*

	conv_loss��@>J���        )��P	�,�����A�*

	conv_loss��C>�~.]        )��P	!U�����A�*

	conv_loss��B>���8        )��P	�����A�*

	conv_loss�U>�+q�        )��P	������A�*

	conv_loss��:>j�bM        )��P	L������A�*

	conv_loss��1>�t�g        )��P	?������A�*

	conv_lossf�W>�b��        )��P	�$ ����A�*

	conv_lossϞ;>q|�        )��P	�N ����A�*

	conv_lossJ�E>�cǼ        )��P	O{ ����A�*

	conv_loss��9>Lu`        )��P	f� ����A�*

	conv_lossL�R>�J%c        )��P	v� ����A�*

	conv_loss!>N>kϑ        )��P	�K����A�*

	conv_loss�G?>
&P        )��P	�w����A�*

	conv_loss�dE>x��        )��P	�����A�*

	conv_loss��?>�D��        )��P	������A�*

	conv_loss�Q>����        )��P	������A�*

	conv_loss�m9>?y��        )��P	�#����A�*

	conv_loss!�V>q�q�        )��P	�S����A�*

	conv_loss��D>Q��        )��P	�}����A�*

	conv_loss��F>]�i(        )��P	P�����A�*

	conv_lossPG>�1        )��P	������A�*

	conv_loss��5>R;�5        )��P	N����A�*

	conv_loss�E>���        )��P	DF����A�*

	conv_loss�mE>�,�
        )��P	�o����A�*

	conv_loss�{D>�D�        )��P	g�����A�*

	conv_loss�[>>G�        )��P	c�����A�*

	conv_loss�g2>iR�`        )��P	������A�*

	conv_loss٭C>��8l        )��P	a����A�*

	conv_loss@�M>T
��        )��P	C:����A�*

	conv_lossnX<>-�0        )��P	d����A�*

	conv_loss�GG>˗�;        )��P	�����A�*

	conv_losst�>>u@�J        )��P	������A�*

	conv_loss�F>��
4        )��P	�����A�*

	conv_loss0�F>�h>�        )��P	�����A�*

	conv_loss�sK>)E_        )��P	f>����A�*

	conv_loss��H>��ȣ        )��P	�m����A�*

	conv_lossZ�=>H-)+        )��P	������A�*

	conv_loss��9>YL�A        )��P	������A�*

	conv_lossh�M>���        )��P	�����A�*

	conv_loss�/>-_FH        )��P	�����A�*

	conv_loss-=>3��        )��P	�E����A�*

	conv_lossk�L><E��        )��P	�o����A�*

	conv_loss��N>��(         )��P	������A�*

	conv_loss�z@>�OY        )��P	������A�*

	conv_loss��7>�O¬        )��P	������A�*

	conv_loss�(F>��u�        )��P	�����A�*

	conv_lossW0>k���        )��P	�?����A�*

	conv_loss��Q>��}        )��P	�h����A�*

	conv_lossr�=>���        )��P	k�����A�*

	conv_loss=.>%���        )��P	G�����A�*

	conv_loss}Q>�$�H        )��P	_�����A�*

	conv_loss��@>r��]        )��P	�	����A�*

	conv_loss{�8>����        )��P	v:	����A�*

	conv_loss��B>9�{!        )��P	�b	����A�*

	conv_loss2<>�	�        )��P	�	����A�*

	conv_loss��;>���!        )��P	F�	����A�*

	conv_lossɰ>>/
j        )��P	G�	����A�*

	conv_lossqA>>�^��        )��P	�
����A�*

	conv_loss�k@>��        )��P	�7
����A�*

	conv_loss��5>4b        )��P	�a
����A�*

	conv_loss�2>�Ib�        )��P	�
����A�*

	conv_loss2P>>g���        )��P	V�
����A�*

	conv_lossvnF>��~        )��P	�
����A�*

	conv_lossIT>7�B�        )��P	�����A�*

	conv_loss�:>�8�        )��P	�E����A�*

	conv_loss�Z9>�˝        )��P	yt����A�*

	conv_loss7�A>�7#�        )��P	������A�*

	conv_loss/?>��J        )��P	������A�*

	conv_loss2�;>zL}�        )��P	'�����A�*

	conv_losss=>��F        )��P	�����A�*

	conv_loss'6M>Y��u        )��P	�D����A�*

	conv_loss�{9>p        )��P	�y����A�*

	conv_loss��<>A~~        )��P	}�����A�*

	conv_loss��D>j2uR        )��P	������A�*

	conv_loss�G>��C        )��P	I�����A�*

	conv_lossVb<>��        )��P	�!����A�*

	conv_lossg�C>�ݸ�        )��P	�J����A�*

	conv_loss�Y/>ݫʹ        )��P	ʄ����A�*

	conv_loss$<>\B*        )��P	������A�*

	conv_loss�D>w+��        )��P	�����A�*

	conv_loss�\I>r'@�        )��P	�����A�*

	conv_lossn�?>�[�(        )��P	�4����A�*

	conv_loss� @>�o�C        )��P	�`����A�*

	conv_loss��B>�=~        )��P	7�����A�*

	conv_loss�4>q��        )��P	������A�*

	conv_lossPp5>����        )��P	������A�*

	conv_loss� 6>��hM        )��P	�����A�*

	conv_loss��;>�a��        )��P	Q2����A�*

	conv_loss��9>�P~�        )��P	^\����A�*

	conv_loss'5?>�"        )��P	8�����A�*

	conv_loss@�'>�o�        )��P	������A�*

	conv_loss�0>ϋA�        )��P	������A�*

	conv_loss��C>���.        )��P	�����A�*

	conv_loss,2>n���        )��P	�*����A�*

	conv_loss}�9>u�m�        )��P	�R����A�*

	conv_loss6�E>Y|[        )��P	�}����A�*

	conv_loss��=>�*ߚ        )��P	������A�*

	conv_loss��G>��/        )��P	?�����A�*

	conv_lossdL>HC�        )��P	������A�*

	conv_loss�B>���C        )��P	�#����A�*

	conv_loss"n;>���$        )��P	�L����A�*

	conv_loss�hC>R��        )��P	/v����A�*

	conv_losse4>��15        )��P	b�����A�*

	conv_loss�8<>'���        )��P	C�����A�*

	conv_loss11.>��,�        )��P	������A�*

	conv_loss��->��         )��P	����A�*

	conv_loss�o7>�(�2        )��P	�C����A�*

	conv_loss�r->�0�        )��P	qo����A�*

	conv_loss/�7>�珏        )��P	�����A�*

	conv_lossFS*>s�<R        )��P	������A�*

	conv_lossq�B>>�
�        )��P	������A�*

	conv_lossNE:>����        )��P	M#����A�*

	conv_lossc�>l���        )��P	8L����A�*

	conv_loss Z7>�e��        )��P	�u����A�*

	conv_loss->��        )��P	������A�*

	conv_lossތ1>�D        )��P	������A�*

	conv_loss� %>��        )��P	������A�*

	conv_loss�I7>��        )��P	!����A�*

	conv_lossb7>e�V        )��P	�K����A�*

	conv_loss2�+>��r        )��P	�v����A�*

	conv_loss�x@>����        )��P	v�����A�*

	conv_loss�0?>r`q        )��P	������A�*

	conv_loss�,>����        )��P	����A�*

	conv_loss}3>�ǌ�        )��P	*2����A�*

	conv_loss73>�?�K        )��P	]`����A�*

	conv_loss8�?>6A��        )��P	�����A�*

	conv_loss��%>�_(�        )��P	m�����A�*

	conv_loss�|5>�!�        )��P	l�����A�*

	conv_loss��->�̾        )��P	����A�*

	conv_lossWv:>��L        )��P	�F����A�*

	conv_loss��*>^��        )��P	�r����A�*

	conv_loss
$>`�E�        )��P	~�����A�*

	conv_loss��4>W���        )��P	������A�*

	conv_loss��?>��        )��P	������A�*

	conv_lossGc&>�WLM        )��P	: ����A�*

	conv_loss�A>N��/        )��P	
K����A�*

	conv_lossfX9>�cAn        )��P	�x����A�*

	conv_loss�+>�O4�        )��P	ե����A�*

	conv_loss�9>>]<�        )��P	�����A�*

	conv_loss�7>���;        )��P	�����A�*

	conv_loss	V*>��        )��P	q+����A�*

	conv_loss�u3>�*d        )��P	�U����A�*

	conv_loss��+>���        )��P	������A�*

	conv_lossA"F>�j}2        )��P	_�����A�*

	conv_loss_�->q�ָ        )��P	������A�*

	conv_lossa�5>�0��        )��P	� ����A�*

	conv_loss��(>"�R         )��P	�+����A�*

	conv_loss��/>�N%�        )��P	vT����A�*

	conv_loss�l/>OB�        )��P	Z�����A�*

	conv_loss��1>'��        )��P	`�����A�*

	conv_loss�=.>�n�        )��P	H�����A�*

	conv_loss�>>�V��        )��P	����A�*

	conv_loss�E>v�r�        )��P	�,����A�*

	conv_loss��*>���        )��P	�W����A�*

	conv_loss��(>���        )��P	&�����A�*

	conv_loss�7>+�<�        )��P	������A�*

	conv_lossm�;>�5��        )��P	������A�*

	conv_lossJ�>>Q���        )��P	� ����A�*

	conv_lossMF6>�dc        )��P	�,����A�*

	conv_loss�m5>�t��        )��P	�[����A�*

	conv_loss+>/'m        )��P	E�����A�*

	conv_loss�]9>瑻        )��P	R�����A�*

	conv_lossk�=>�i�f        )��P	H�����A�*

	conv_loss�=H>��C        )��P	x����A�*

	conv_loss<</>As�S        )��P	�H����A�*

	conv_lossŤ5>�/        )��P	{����A�*

	conv_lossc&*>�r        )��P	������A�*

	conv_loss��4>ɐ�/        )��P	M�����A�*

	conv_loss{�!>'<\v        )��P	�����A�*

	conv_lossy�1>&        )��P	�D����A�*

	conv_loss��7>UT��        )��P	lw����A�*

	conv_loss�Z.>69�%        )��P	T�����A�*

	conv_losseg6>�V�	        )��P	�����A�*

	conv_lossvd*>up�        )��P	[����A�*

	conv_lossޑ > �         )��P	03����A�*

	conv_loss��7>l�
<        )��P	�^����A�*

	conv_loss��9>�>��        )��P	y�����A�*

	conv_loss�)>����        )��P	�����A�*

	conv_loss��1>=nQo        )��P	������A�*

	conv_loss�A2>���U        )��P	����A�*

	conv_loss��0>.���        )��P	oA����A�*

	conv_loss�?>��@        )��P	Dq����A�*

	conv_loss�k&><��        )��P	c�����A�*

	conv_loss#� >9��e        )��P	D�����A�*

	conv_loss�O>Q|/        )��P	� ����A�*

	conv_loss�(>m�9�        )��P	�/ ����A�*

	conv_loss�V6>\�i�        )��P	Z\ ����A�*

	conv_lossFF>W�^@        )��P	,� ����A�*

	conv_lossț+>o�n        )��P	� ����A�*

	conv_losst{6>����        )��P	� ����A�*

	conv_loss��(>�        )��P	+!����A�*

	conv_loss��;>A�M�        )��P	TB!����A�*

	conv_loss7�1>�y��        )��P	o!����A�*

	conv_loss )>�J�        )��P	k�!����A�*

	conv_loss63>��        )��P	�!����A�*

	conv_lossm&1>Ϡ�>        )��P	�!����A�*

	conv_loss$F=>=
��        )��P	u""����A�*

	conv_loss�4#>�LM�        )��P	BO"����A�*

	conv_loss��2>���/        )��P	�~"����A�*

	conv_lossXX;>���        )��P	��"����A�*

	conv_loss��+>�s��        )��P	
�"����A�*

	conv_loss��+>�f�        )��P	l#����A�*

	conv_lossD'>�@�S        )��P	]3#����A�*

	conv_loss�4>p�9        )��P	_#����A�*

	conv_loss��)>��Ж        )��P	�#����A�*

	conv_lossB-+>�ܚ�        )��P	�#����A�*

	conv_loss]1>Bo��        )��P	��#����A�*

	conv_lossE0>7�+�        )��P	5$����A�*

	conv_lossa&>��n�        )��P	"A$����A�*

	conv_loss��7>���        )��P	�k$����A�*

	conv_loss�'>)t"        )��P	B�$����A�*

	conv_loss�� >ķ�        )��P	f&����A�*

	conv_loss��>�n��        )��P	~3&����A�*

	conv_loss�`&>��        )��P	�^&����A�*

	conv_loss��#>,�        )��P	��&����A�*

	conv_loss��>��O        )��P	��&����A�*

	conv_loss��>>��,S        )��P	U�&����A�*

	conv_loss�+>���        )��P	D'����A�*

	conv_loss�h6>pr-�        )��P	�N'����A�*

	conv_lossQe5>x�nW        )��P	'����A�*

	conv_loss�C1>�Q�        )��P	�'����A�*

	conv_loss��#>*�G�        )��P	��'����A�*

	conv_loss˦6>}�4        )��P	2(����A�*

	conv_loss�s>>��Q        )��P	p:(����A�*

	conv_lossbS#>dh�        )��P	�p(����A�*

	conv_lossd['>,"��        )��P	�(����A�*

	conv_loss�H*>�ͯ        )��P	j�(����A�*

	conv_loss�y4>��        )��P	��(����A�*

	conv_lossb�$>T �P        )��P	�#)����A�*

	conv_loss��.>y�h�        )��P	�Q)����A�*

	conv_loss��">�-�        )��P	~�)����A�*

	conv_loss�*><�        )��P	��)����A�*

	conv_lossq�.>#ƙ�        )��P	��)����A�*

	conv_loss	�#> z��        )��P	�*����A�*

	conv_loss#�A>n<��        )��P	uH*����A�*

	conv_loss�(>��/$        )��P	�u*����A�*

	conv_loss�+>��C        )��P	Р*����A�*

	conv_loss�<7>���        )��P	��*����A�*

	conv_loss�p9>6F�        )��P	��*����A�*

	conv_lossZ�->��        )��P	'+����A�*

	conv_loss�->y�D        )��P	\Q+����A�*

	conv_loss�:(>��u5        )��P	�~+����A�*

	conv_loss8x,>`G�        )��P	 �+����A�*

	conv_loss6*$>�~�u        )��P	��+����A�*

	conv_loss�2>��m�        )��P	Y,����A�*

	conv_loss (>��*�        )��P	�4,����A�*

	conv_lossa.>+�!        )��P	Mb,����A�*

	conv_loss]0.>/ij        )��P	��,����A�*

	conv_lossnF->�,�        )��P	0�,����A�*

	conv_lossA�.>�mI        )��P	G�,����A�*

	conv_lossp�$>���        )��P	-����A�*

	conv_lossE%>��(6        )��P	gB-����A�*

	conv_loss�z0>|�d�        )��P	o-����A�*

	conv_loss{'>]^Rt        )��P	��-����A�*

	conv_lossՄ#>\n9        )��P	@�-����A�*

	conv_loss��">pı�        )��P	�-����A�*

	conv_loss&>$��9        )��P	e".����A�*

	conv_lossc8>���4        )��P	:N.����A�*

	conv_loss'�>� je        )��P	Uz.����A�*

	conv_lossL">���        )��P	K�.����A�*

	conv_loss�s>�:�        )��P	��.����A�*

	conv_lossT�=>u(2        )��P	L/����A�*

	conv_lossB0>�;�P        )��P	 @/����A�*

	conv_loss�>�(�        )��P	kk/����A�*

	conv_loss 31>�/��        )��P	|�/����A�*

	conv_loss�f*>�P        )��P	��/����A�*

	conv_loss�>�3t        )��P		�/����A�*

	conv_loss��1>tcbj        )��P	�"0����A�*

	conv_loss��&>Spc        )��P	�O0����A�*

	conv_lossMZ%>�zF�        )��P	f�0����A�*

	conv_loss)�>��        )��P	5�0����A�*

	conv_loss��'>���A        )��P	5�0����A�*

	conv_loss6�%>�z��        )��P	�1����A�*

	conv_loss�)>�b��        )��P	�M1����A�*

	conv_loss�/>-i�W        )��P	|1����A�*

	conv_lossd0>���        )��P	��1����A�*

	conv_loss!J>M�        )��P	�1����A�*

	conv_lossl�!>ͱj        )��P	�2����A�*

	conv_loss��.>>�        )��P	k52����A�*

	conv_loss��,>zyi�        )��P	c2����A�*

	conv_loss�&>5ل        )��P	��2����A�*

	conv_loss%z>�@ݽ        )��P	u�2����A�*

	conv_loss�{,>yͼ�        )��P	^�2����A�*

	conv_loss��0>���h        )��P	C3����A�*

	conv_loss�">�"�        )��P	`J3����A�*

	conv_lossR!>� ӑ        )��P	�u3����A�*

	conv_lossnJ>��1        )��P	��3����A�*

	conv_loss܊>�O�        )��P	&�3����A�*

	conv_lossC#>,�N        )��P	��3����A�*

	conv_loss�h:>�>?        )��P	|+4����A�*

	conv_loss� (>A�ƽ        )��P	LY4����A�*

	conv_loss1�!>���        )��P	̈́4����A�*

	conv_loss4�$>� �        )��P	�4����A�*

	conv_loss�`>�Ɨ�        )��P	��4����A�*

	conv_lossU�>�¡        )��P	z5����A�*

	conv_lossb >;�        )��P	:5����A�*

	conv_loss��,>5}.        )��P	>f5����A�*

	conv_loss��>9�
        )��P	�5����A�*

	conv_loss��>�g�y        )��P	�5����A�*

	conv_lossG&)>u��c        )��P	d�5����A�*

	conv_lossa8/>�o~J        )��P	6����A�*

	conv_lossJ�>���G        )��P	H6����A�*

	conv_loss�&>Q�G�        )��P	't6����A�*

	conv_loss�>+�u        )��P	�6����A�*

	conv_loss��>GAW�        )��P	�6����A�*

	conv_loss`=->��Kp        )��P	=�6����A�*

	conv_loss,0>'��        )��P	(7����A�*

	conv_lossi�->U�,        )��P	�T7����A�*

	conv_lossP�:>i�_        )��P	k�7����A�*

	conv_losss�>�N�        )��P	��7����A�*

	conv_loss�,>=Q=        )��P	��7����A�*

	conv_loss=� >��        )��P	V8����A�*

	conv_loss~�>��'�        )��P	�G8����A�*

	conv_loss�o>�k�        )��P	Wt8����A�*

	conv_lossf�&>���1        )��P	{�8����A�*

	conv_loss�>I�o�        )��P	��8����A�*

	conv_loss��>] N+        )��P	�9����A�*

	conv_lossk�(>�:�v        )��P	L.9����A�*

	conv_loss:C>\�        )��P	jZ9����A�*

	conv_loss�J1>Yy�        )��P	Ċ9����A�*

	conv_lossO3> �cq        )��P	��9����A�*

	conv_loss�25>D�_        )��P	��9����A�*

	conv_loss��>��        )��P	�%:����A�*

	conv_loss�W	>��6        )��P	{R:����A�*

	conv_loss5,#>���        )��P	!�:����A�*

	conv_loss�#>���        )��P	.�:����A�*

	conv_loss�59>,]h�        )��P	��:����A�*

	conv_loss{Z$>.��        )��P	X;����A�*

	conv_lossHg>��Ԧ        )��P	A;����A�*

	conv_lossW�#>[�Ǩ        )��P	.r;����A�*

	conv_loss�">����        )��P	"�;����A�*

	conv_loss�!>�w�E        )��P	��;����A�*

	conv_losse\#>@W�5        )��P	��;����A�*

	conv_lossK>&cF        )��P	)<����A�*

	conv_lossQ�)>�N�$        )��P	�W<����A�*

	conv_loss��>Ao�U        )��P	�<����A�*

	conv_loss��>o��>        )��P	z�<����A�*

	conv_loss�E>2c��        )��P	 �<����A�*

	conv_loss��&>n���        )��P	=����A�*

	conv_loss(>z5��        )��P	�;=����A�*

	conv_loss>��^e        )��P	h=����A�*

	conv_loss)>�0�        )��P	�=����A�*

	conv_losshY>���        )��P	�=����A�*

	conv_lossJ�>��9        )��P	��=����A�*

	conv_loss� >��fX        )��P	�>����A�*

	conv_lossQ�>��k        )��P	mF>����A�*

	conv_lossR (>K6eL        )��P	�s>����A�*

	conv_loss[>���*        )��P	��>����A�*

	conv_loss=W>g9^        )��P	��>����A�*

	conv_loss��>��        )��P	��>����A�*

	conv_loss��>�J&        )��P	)?����A�*

	conv_loss��>��U�        )��P	�T?����A�*

	conv_lossA�>~�        )��P	��?����A�*

	conv_loss�">/�kH        )��P	�?����A�*

	conv_losse0>�^�        )��P	��?����A�*

	conv_loss=�>B\�C        )��P	�@����A�*

	conv_loss0�>Ba�        )��P	�;@����A�*

	conv_losss� >�>O        )��P	i@����A�*

	conv_loss��!>kP�        )��P	��@����A�*

	conv_loss{�*>d��t        )��P	��@����A�*

	conv_lossWY>�        )��P	d�@����A�*

	conv_loss��>>�        )��P	�.A����A�*

	conv_loss��>.g�        )��P	-]A����A�*

	conv_loss?�&>N��        )��P	��A����A�*

	conv_loss�I>���!        )��P	�A����A�*

	conv_loss0>��        )��P	�A����A�*

	conv_loss�M&>����        )��P	!B����A�*

	conv_loss�">	Vj�        )��P	�LB����A�*

	conv_loss>!>V�v        )��P	�yB����A�*

	conv_loss�� >"[Bk        )��P	�B����A�*

	conv_loss�>p��3        )��P	��B����A�*

	conv_lossw.>���        )��P	"#C����A�*

	conv_lossk�>�s&�        )��P	QC����A�*

	conv_loss��#>�AV        )��P	W~C����A�*

	conv_losseH)>�7�0        )��P	�C����A�*

	conv_lossR�>u��        )��P	s�C����A�*

	conv_loss��>Ʃ@T        )��P	LD����A�*

	conv_loss,�>ћU)        )��P	�1D����A�*

	conv_loss�>�1c�        )��P	!aD����A�*

	conv_loss3�%>1��        )��P	f�D����A�*

	conv_loss�>IoD�        )��P	��D����A�*

	conv_loss�_$>�h        )��P	�D����A�*

	conv_loss�^ >a��<        )��P	ME����A�*

	conv_lossſ>��WQ        )��P	�KE����A�*

	conv_loss�p>Ll�        )��P	�~E����A�*

	conv_loss�>�Ȭ        )��P	W�E����A�*

	conv_lossB- >��s        )��P	��E����A�*

	conv_lossp�>�_,        )��P	4
F����A�*

	conv_loss@�+>���        )��P	.9F����A�*

	conv_loss��>}b�        )��P	�fF����A�*

	conv_losseb>1�.�        )��P	w�F����A�*

	conv_loss	�">����        )��P	��F����A�*

	conv_loss�y+>�*�+        )��P	�F����A�*

	conv_loss޷%>����        )��P	�G����A�*

	conv_loss�$>��        )��P	MIG����A�*

	conv_lossI>�"6        )��P	�uG����A�*

	conv_lossF>���.        )��P	��G����A�*

	conv_lossy�>��R\        )��P	��G����A�*

	conv_loss�>�
��        )��P	��G����A�*

	conv_lossJ>27��        )��P	*'H����A�*

	conv_loss��>��*        )��P	>TH����A�*

	conv_lossF�
>����        )��P	#�H����A�*

	conv_lossȐ/>�ʇ`        )��P	ԮH����A�*

	conv_losse�>��.q        )��P	��H����A�*

	conv_lossI�'>l�        )��P	�I����A�*

	conv_loss��
>Osl�        )��P	�3I����A�*

	conv_lossS&>�Bt�        )��P	�_I����A�*

	conv_loss�� >��        )��P	o�I����A�*

	conv_loss8�>�-��        )��P	ŶI����A�*

	conv_lossh5">�	�C        )��P	��I����A�*

	conv_loss�M>|�K