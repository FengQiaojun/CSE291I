       �K"	   ����Abrain.Event:2�1x!ޕ      ��7�	������A"ѫ
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
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*
data_formatNHWC*
strides
*
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
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
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
conv2d_3/Conv2DConv2DRelu_1conv2d_2/kernel/read*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Y
Relu_2Reluconv2d_3/Conv2D*/
_output_shapes
:���������*
T0
^
Reshape/shapeConst*
_output_shapes
:*
valueB"����0	  *
dtype0
j
ReshapeReshapeRelu_2Reshape/shape*(
_output_shapes
:����������*
T0*
Tshape0
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"0	  d   *
dtype0*
_output_shapes
:
�
+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *�J�
�
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *�J=*
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
:	�d
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	�d*
T0*
_class
loc:@dense/kernel
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d
�
dense/kernel
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_output_shapes
:	�d*
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
:	�d
�
dense/MatMulMatMulReshapedense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
N
Relu_3Reludense/MatMul*
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

seed *
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes

:d

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
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:d

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
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

�
dense_2/MatMulMatMulRelu_3dense_1/kernel/read*'
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
conv_loss/tagsConst*
_output_shapes
: *
valueB B	conv_loss*
dtype0
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
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
$gradients/logistic_loss/mul_grad/mulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Placeholder_1*'
_output_shapes
:���������
*
T0
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
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
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
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_37gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:d
*
transpose_a(*
transpose_b( *
T0
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
gradients/Relu_3_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_3*
T0*'
_output_shapes
:���������d
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_3_grad/ReluGraddense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/Relu_3_grad/ReluGrad*
T0*
_output_shapes
:	�d*
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
:����������
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�d
b
gradients/Reshape_grad/ShapeShapeRelu_2*
_output_shapes
:*
T0*
out_type0
�
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
gradients/Relu_2_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_2*
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
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/Relu_1_grad/ReluGrad*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides

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
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/Relu_grad/ReluGrad*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d
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

�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
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
Mean_1MeanCastConst_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: "��}Q�      ֵU�	PB����AJ��
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
Ttype*1.4.12v1.4.0-19-ga52c8d9ѫ
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
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
T0
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
Relu_1Reluconv2d_2/Conv2D*/
_output_shapes
:���������*
T0
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
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
conv2d_2/kernel/readIdentityconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:*
T0
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Y
Relu_2Reluconv2d_3/Conv2D*/
_output_shapes
:���������*
T0
^
Reshape/shapeConst*
valueB"����0	  *
dtype0*
_output_shapes
:
j
ReshapeReshapeRelu_2Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
-dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"0	  d   *
dtype0
�
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *�J�*
dtype0*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *�J=*
dtype0*
_output_shapes
: 
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�d*

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
:	�d
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d
�
dense/kernel
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*
_class
loc:@dense/kernel
v
dense/kernel/readIdentitydense/kernel*
_output_shapes
:	�d*
T0*
_class
loc:@dense/kernel
�
dense/MatMulMatMulReshapedense/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
N
Relu_3Reludense/MatMul*
T0*'
_output_shapes
:���������d
�
/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*!
_class
loc:@dense_1/kernel*
valueB"d   
   *
dtype0
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
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:d
*
T0*!
_class
loc:@dense_1/kernel
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
dense_2/MatMulMatMulRelu_3dense_1/kernel/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
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
conv_loss/tagsConst*
_output_shapes
: *
valueB B	conv_loss*
dtype0
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
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
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
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
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
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������
*
T0
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
:*

Tidx0*
	keep_dims( *
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
:*

Tidx0*
	keep_dims( 
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
9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_2/BiasAdd_grad/BiasAddGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

�
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������d*
transpose_a( 
�
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_37gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
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
8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:d
*
T0
�
gradients/Relu_3_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_3*
T0*'
_output_shapes
:���������d
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/Relu_3_grad/ReluGraddense/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/Relu_3_grad/ReluGrad*
transpose_b( *
T0*
_output_shapes
:	�d*
transpose_a(
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:����������
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�d
b
gradients/Reshape_grad/ShapeShapeRelu_2*
_output_shapes
:*
T0*
out_type0
�
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
gradients/Relu_2_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_2*
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
GradientDescent/learning_rateConst*
_output_shapes
: *
valueB
 *o;*
dtype0
�
9GradientDescent/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelGradientDescent/learning_rate7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
use_locking( *
T0
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:*
use_locking( 
�
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:*
use_locking( *
T0
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d
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
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( *
T0
�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
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
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_1MeanCastConst_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: ""�
	variables��
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
q
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"�
trainable_variables��
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
q
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:0
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
GradientDescent�.�]       `/�#	C
G����A*

	conv_loss�2?�L2�       QKD	�J����A*

	conv_lossy�1?c�0       QKD	�J����A*

	conv_loss"$2?�EZ�       QKD	+�J����A*

	conv_loss�1?a[��       QKD	��J����A*

	conv_loss|�1?E�:a       QKD	
K����A*

	conv_loss�
2?@       QKD	�>K����A*

	conv_lossG�1?�A       QKD		`K����A*

	conv_loss\�1?i�!       QKD	�K����A*

	conv_loss��1?���       QKD	��K����A	*

	conv_loss��1?��J�       QKD	G�K����A
*

	conv_lossQ�1?�!b�       QKD	O�K����A*

	conv_loss��1?����       QKD	�L����A*

	conv_loss�1?��̙       QKD	B+L����A*

	conv_loss��1?8�(�       QKD	�ML����A*

	conv_loss�1?5��a       QKD	qL����A*

	conv_loss��1?k���       QKD	АL����A*

	conv_loss޴1?Ț�       QKD	��L����A*

	conv_lossA�1?sZd       QKD	/�L����A*

	conv_loss�1?Ӄ        QKD	
M����A*

	conv_loss��1?2��       QKD	-M����A*

	conv_loss��1?4I�U       QKD	�MM����A*

	conv_loss��1?ĺX�       QKD	LlM����A*

	conv_loss�~1?�D��       QKD	��M����A*

	conv_loss��1?�,�       QKD	�M����A*

	conv_loss��1?i�v_       QKD	V�M����A*

	conv_loss�1?hJ�Z       QKD	��M����A*

	conv_lossï1?`#B�       QKD	�N����A*

	conv_lossy�1?[���       QKD	�/N����A*

	conv_lossN�1?��       QKD	ON����A*

	conv_loss�1?uo"�       QKD	�qN����A*

	conv_loss#�1?0
$|       QKD	(�N����A*

	conv_loss��1?���n       QKD	��N����A *

	conv_lossde1?F�\D       QKD	W�N����A!*

	conv_loss�{1?���y       QKD	I�N����A"*

	conv_lossc�1?��#�       QKD	dO����A#*

	conv_lossB�1? a)�       QKD	
+O����A$*

	conv_loss+�1?u��w       QKD	�JO����A%*

	conv_loss3�1?.       QKD	�iO����A&*

	conv_loss��1?Ԟ�
       QKD	o�O����A'*

	conv_loss�s1?�v       QKD	��O����A(*

	conv_lossq�1?Ѧ�       QKD	I�O����A)*

	conv_loss�51?>&9       QKD	��O����A**

	conv_loss�X1?$��       QKD	�P����A+*

	conv_loss�`1?�笅       QKD	
-P����A,*

	conv_loss�41?�\��       QKD	�LP����A-*

	conv_lossa�1? �<E       QKD	ZkP����A.*

	conv_loss�>1?P<       QKD	M�P����A/*

	conv_loss�,1?���O       QKD	@�P����A0*

	conv_lossA-1?��\�       QKD	;�P����A1*

	conv_loss�K1? 5�       QKD	{�P����A2*

	conv_losskM1?��q�       QKD	�Q����A3*

	conv_lossG1?�I
�       QKD	4Q����A4*

	conv_lossGa1?���       QKD	TQ����A5*

	conv_lossAL1?�!o       QKD	�vQ����A6*

	conv_lossb<1?��2i       QKD	�Q����A7*

	conv_loss>1?pAO       QKD	ŷQ����A8*

	conv_loss�@1?r��       QKD	J�Q����A9*

	conv_lossd>1?��w       QKD	�R����A:*

	conv_loss}1?���]       QKD	�!R����A;*

	conv_loss�31?�`��       QKD		@R����A<*

	conv_loss�G1?�f�       QKD	�`R����A=*

	conv_loss�%1?F^�5       QKD	T�R����A>*

	conv_loss-1?+;�       QKD	��R����A?*

	conv_loss�1?��       QKD	�R����A@*

	conv_loss841?`�#�       QKD	��R����AA*

	conv_lossX?1?�/Q�       QKD	]S����AB*

	conv_lossS=1?�=c�       QKD	�=S����AC*

	conv_lossW1?¨       QKD	�^S����AD*

	conv_loss&1?�	W�       QKD	�S����AE*

	conv_loss�1?@�       QKD	g�S����AF*

	conv_loss�1?.�\�       QKD	a�S����AG*

	conv_loss�1?�!�       QKD	 �S����AH*

	conv_loss��0?�ys       QKD	��S����AI*

	conv_lossh1?C�       QKD	�!T����AJ*

	conv_loss
1?��       QKD	LBT����AK*

	conv_loss�1?$p,�       QKD	�hT����AL*

	conv_lossq1?���B       QKD	��T����AM*

	conv_loss1?��D       QKD	�T����AN*

	conv_loss� 1?B�5�       QKD	��T����AO*

	conv_loss��0?���       QKD	m�T����AP*

	conv_loss�0?q�nT       QKD	�U����AQ*

	conv_loss��0?�֎,       QKD	�8U����AR*

	conv_loss�1?��F       QKD	�YU����AS*

	conv_loss��0?�D`       QKD	8zU����AT*

	conv_loss��0?>m�S       QKD	�U����AU*

	conv_loss��0?@B�y       QKD	��U����AV*

	conv_loss)1?E���       QKD	��U����AW*

	conv_loss1?8r*�       QKD	u�U����AX*

	conv_loss?�0?�.y       QKD	�V����AY*

	conv_loss��0?����       QKD	�=V����AZ*

	conv_lossm�0?-K@<       QKD	�]V����A[*

	conv_lossS�0?932       QKD	�~V����A\*

	conv_loss�1?����       QKD	\�V����A]*

	conv_loss��0?b�q       QKD	��V����A^*

	conv_loss�0?��       QKD	��V����A_*

	conv_lossϴ0?       QKD	�W����A`*

	conv_loss��0?��       QKD	�%W����Aa*

	conv_loss�0?���%       QKD	jEW����Ab*

	conv_loss��0?��)       QKD	eW����Ac*

	conv_loss�0?��b       QKD	q�W����Ad*

	conv_loss��0?�       QKD	�W����Ae*

	conv_loss+�0?�!
       QKD	h�W����Af*

	conv_loss��0?��P       QKD	��W����Ag*

	conv_loss��0?����       QKD	�X����Ah*

	conv_loss��0?#�o�       QKD	�5X����Ai*

	conv_lossn�0?����       QKD	�XX����Aj*

	conv_loss��0?0��       QKD	�wX����Ak*

	conv_loss=�0?�E       QKD	��X����Al*

	conv_loss3�0?z.Wm       QKD	O�X����Am*

	conv_loss�0?�f�       QKD	��X����An*

	conv_loss�0?Z�C       QKD	;�X����Ao*

	conv_lossݜ0?�2�6       QKD	�Y����Ap*

	conv_loss`0?52�f       QKD	�CY����Aq*

	conv_loss�0?@��{       QKD	�gY����Ar*

	conv_loss5p0?�Ko�       QKD	��Y����As*

	conv_loss�h0?a�jm       QKD	=�Y����At*

	conv_lossKp0?�!.       QKD	��Y����Au*

	conv_loss�f0?e䈄       QKD	��Y����Av*

	conv_loss��0?ђ�       QKD	nZ����Aw*

	conv_loss0y0?��}       QKD	�KZ����Ax*

	conv_loss"j0?��/I       QKD	_mZ����Ay*

	conv_lossPi0?�{B�       QKD	��Z����Az*

	conv_lossѕ0?|��A       QKD	��Z����A{*

	conv_loss?�0?V͢       QKD	��Z����A|*

	conv_lossC]0?E��       QKD	�Z����A}*

	conv_loss�_0?�/F       QKD	o[����A~*

	conv_loss?~0?�q+       QKD	�4[����A*

	conv_loss9�0?��        )��P	CV[����A�*

	conv_loss�q0?:�'        )��P	�v[����A�*

	conv_loss�Z0?%!E}        )��P	�[����A�*

	conv_lossRY0?g0�?        )��P	j�[����A�*

	conv_loss�M0?��G        )��P	&�[����A�*

	conv_loss�C0?AH�        )��P	C\����A�*

	conv_loss�H0?�`         )��P	�&\����A�*

	conv_loss�Z0?�r�'        )��P	HJ\����A�*

	conv_loss�d0?T��        )��P	�j\����A�*

	conv_loss�p0?6h`t        )��P	�\����A�*

	conv_loss�O0?J
�        )��P	��\����A�*

	conv_loss�@0?��::        )��P	��\����A�*

	conv_loss�T0?�	�        )��P	�\����A�*

	conv_loss�50?y�V        )��P	�]����A�*

	conv_loss�&0?�_        )��P	14]����A�*

	conv_loss#�/?�	�w        )��P	/T]����A�*

	conv_loss(90?"q��        )��P	Qt]����A�*

	conv_loss90?3��        )��P	6�]����A�*

	conv_loss�90?�wM        )��P	�]����A�*

	conv_loss�#0?0*N        )��P	��]����A�*

	conv_loss�W0?|4:        )��P	c�]����A�*

	conv_loss�N0?x��        )��P	�^����A�*

	conv_loss��/?�k        )��P	�@^����A�*

	conv_loss�0?_���        )��P	�q^����A�*

	conv_loss�-0?���        )��P	��^����A�*

	conv_loss��/?1�f�        )��P	��^����A�*

	conv_lossKA0?���        )��P	��^����A�*

	conv_lossv0?�>H        )��P	2�^����A�*

	conv_loss�I0?�"        )��P	�_����A�*

	conv_loss0?4 ^�        )��P	�:_����A�*

	conv_loss7�/?I|��        )��P	\[_����A�*

	conv_lossN0?go\�        )��P	e�_����A�*

	conv_loss��/?��z        )��P	��_����A�*

	conv_lossL�/?�)        )��P	��_����A�*

	conv_loss,�/?��V        )��P	��_����A�*

	conv_loss�/?wm�L        )��P	�`����A�*

	conv_loss��/?�-�T        )��P	�1`����A�*

	conv_loss��/?n�4        )��P	�U`����A�*

	conv_loss90?���        )��P	F~`����A�*

	conv_loss�0?/k�        )��P	l�`����A�*

	conv_lossG0?tU��        )��P	��`����A�*

	conv_loss#�/?�Zr)        )��P	L�`����A�*

	conv_loss��/?��        )��P	�a����A�*

	conv_loss��/?ʂX�        )��P	�.a����A�*

	conv_lossK 0?��'        )��P	�Ra����A�*

	conv_loss<�/?�Dl�        )��P	Dva����A�*

	conv_loss�0?���        )��P	�a����A�*

	conv_lossh�/?��ʞ        )��P	N�a����A�*

	conv_loss��/?�WXl        )��P	d�a����A�*

	conv_loss��/?��]        )��P	��a����A�*

	conv_loss<�/?��x        )��P	�b����A�*

	conv_loss��/?G���        )��P	�8b����A�*

	conv_losss�/?���        )��P	�Xb����A�*

	conv_loss}�/?�ۀ        )��P	�yb����A�*

	conv_loss=�/?�N�w        )��P	��b����A�*

	conv_lossz�/?v&�Z        )��P	Ⱦb����A�*

	conv_loss��/?}�        )��P	��b����A�*

	conv_loss��/?-h_�        )��P	� c����A�*

	conv_loss��/?�/��        )��P	.#c����A�*

	conv_loss��/?ԣ        )��P	�Bc����A�*

	conv_loss��/?�ج�        )��P	6bc����A�*

	conv_losse�/?�Ny        )��P	��c����A�*

	conv_loss��/?D��        )��P	��c����A�*

	conv_loss;}/?m�n        )��P	��c����A�*

	conv_loss��/?���        )��P	�c����A�*

	conv_loss�/?����        )��P	�d����A�*

	conv_lossa�/?KĒ        )��P	�&d����A�*

	conv_loss��/?��<�        )��P	�Hd����A�*

	conv_loss8�/?6���        )��P	�kd����A�*

	conv_lossݜ/?���        )��P	��d����A�*

	conv_lossں/??���        )��P	f�d����A�*

	conv_loss#�/?��1u        )��P	~�d����A�*

	conv_lossѓ/?��V        )��P	��d����A�*

	conv_loss>e/?�I�        )��P	#f����A�*

	conv_loss�i/?�U�        )��P	j)f����A�*

	conv_loss�1/?��j�        )��P	Jf����A�*

	conv_loss4�/?߅�R        )��P	Jlf����A�*

	conv_loss�/?-��<        )��P	B�f����A�*

	conv_loss�P/?9�5        )��P	�f����A�*

	conv_loss��/?���c        )��P	$�f����A�*

	conv_lossKr/?h�3V        )��P	��f����A�*

	conv_loss�/?�j�;        )��P	�g����A�*

	conv_loss�w/?���        )��P	�7g����A�*

	conv_loss$e/?q�y2        )��P	�Wg����A�*

	conv_lossH/?Q^(v        )��P	Sxg����A�*

	conv_losss�/?����        )��P	��g����A�*

	conv_loss�^/?�j��        )��P	�g����A�*

	conv_loss�`/?��>        )��P	��g����A�*

	conv_loss�O/?x�'        )��P	�h����A�*

	conv_loss06/?��f�        )��P	�.h����A�*

	conv_losseG/?�k�~        )��P	�Qh����A�*

	conv_lossW^/?e�3        )��P	Oxh����A�*

	conv_loss��/?�%�j        )��P	��h����A�*

	conv_losspH/?+�{6        )��P	;�h����A�*

	conv_lossg./?Űۮ        )��P	F�h����A�*

	conv_loss�T/?[I:x        )��P	i����A�*

	conv_loss)$/?��        )��P	2i����A�*

	conv_loss�,/?��        )��P	hSi����A�*

	conv_lossE/?�:X�        )��P	tui����A�*

	conv_lossr;/?���        )��P		�i����A�*

	conv_loss�3/?�.�        )��P	z�i����A�*

	conv_loss{?/?9>1�        )��P	~�i����A�*

	conv_loss6P/?�g��        )��P	�i����A�*

	conv_loss�2/?R�        )��P	�j����A�*

	conv_loss�K/?��f        )��P	u?j����A�*

	conv_loss�/?!!�        )��P	�aj����A�*

	conv_loss�"/?��'        )��P	�j����A�*

	conv_loss�/?�F�        )��P	ܣj����A�*

	conv_loss�//?'���        )��P	��j����A�*

	conv_loss�%/?��	        )��P	�j����A�*

	conv_loss�/?���        )��P	�k����A�*

	conv_loss��.?E�%�        )��P	�;k����A�*

	conv_lossm�.?���*        )��P	.]k����A�*

	conv_loss2$/?I8I�        )��P	�}k����A�*

	conv_loss�/?3k��        )��P	O�k����A�*

	conv_loss!�.?���        )��P	_�k����A�*

	conv_loss�/?@27�        )��P	��k����A�*

	conv_loss{/?���        )��P	 l����A�*

	conv_loss�</?��;        )��P	]"l����A�*

	conv_loss8/?1�x�        )��P	rEl����A�*

	conv_loss�/?���5        )��P	lgl����A�*

	conv_loss�.?��)w        )��P	�l����A�*

	conv_loss��.?�D�Y        )��P	a�l����A�*

	conv_loss�.?�cm�        )��P	&�l����A�*

	conv_loss
�.?[ng        )��P	]�l����A�*

	conv_loss��.?�lu2        )��P	�m����A�*

	conv_loss\�.?�ߨ�        )��P	�<m����A�*

	conv_loss��.?f�'        )��P	<`m����A�*

	conv_loss��.?�.        )��P	W�m����A�*

	conv_loss�/?XRV�        )��P	B�m����A�*

	conv_lossJ�.?;�C�        )��P	n�m����A�*

	conv_loss�.?Õ�        )��P	��m����A�*

	conv_loss//?=�I        )��P	
n����A�*

	conv_loss�.?e��        )��P	�-n����A�*

	conv_lossI�.?]{3        )��P	�Pn����A�*

	conv_loss�.?G\         )��P	�vn����A�*

	conv_loss��.?J�nr        )��P	^�n����A�*

	conv_loss��.?�[P�        )��P	��n����A�*

	conv_loss��.?��*�        )��P	��n����A�*

	conv_loss��.?1�GD        )��P	�
o����A�*

	conv_lossֵ.?UE�        )��P	�.o����A�*

	conv_loss��.?��K�        )��P	APo����A�*

	conv_loss�.?�g<�        )��P	�no����A�*

	conv_loss��.?~)d�        )��P	X�o����A�*

	conv_loss��.?���        )��P	Ѳo����A�*

	conv_loss��.?;��S        )��P	��o����A�*

	conv_loss��.?`�-�        )��P	��o����A�*

	conv_lossϴ.?&B`,        )��P	5p����A�*

	conv_lossС.?G��        )��P	�Ep����A�*

	conv_loss˿.?!s�*        )��P	�fp����A�*

	conv_loss�e.?�z�        )��P	��p����A�*

	conv_loss��.?�N        )��P	��p����A�*

	conv_loss��.?	6�/        )��P	�p����A�*

	conv_loss��.?��|        )��P	��p����A�*

	conv_lossd�.?���i        )��P	Gq����A�*

	conv_lossr.?&�V        )��P	�5q����A�*

	conv_loss7�.?�%�        )��P	�Wq����A�*

	conv_loss��.?�̕l        )��P	�zq����A�*

	conv_losskO.?��i        )��P	�q����A�*

	conv_loss�n.?W�        )��P	;�q����A�*

	conv_loss�v.?'�z�        )��P	��q����A�*

	conv_loss �.?(��        )��P	�r����A�*

	conv_losssk.?���\        )��P	�!r����A�*

	conv_loss�t.?��K�        )��P	Br����A�*

	conv_loss[`.?k
        )��P	�cr����A�*

	conv_lossl_.?sf�        )��P	:�r����A�*

	conv_lossT�.?.��        )��P	��r����A�*

	conv_loss�<.?��	�        )��P	,�r����A�*

	conv_loss_.?�y��        )��P	��r����A�*

	conv_loss[.?����        )��P	D	s����A�*

	conv_lossA.?e�c�        )��P	�)s����A�*

	conv_loss9.?S[G�        )��P	Ks����A�*

	conv_lossvL.?����        )��P	�ns����A�*

	conv_lossS=.?:���        )��P	��s����A�*

	conv_loss�c.?��z�        )��P	��s����A�*

	conv_loss�G.?�N�        )��P	��s����A�*

	conv_loss�C.?nؚ�        )��P	jt����A�*

	conv_loss�O.?����        )��P	't����A�*

	conv_loss�O.?����        )��P	�It����A�*

	conv_loss.?��i�        )��P	Ait����A�*

	conv_loss�O.?�G��        )��P	��t����A�*

	conv_losst.?.f	�        )��P	�t����A�*

	conv_lossZ[.?��W�        )��P	Q�t����A�*

	conv_loss�=.?���H        )��P	�t����A�*

	conv_lossq9.?>
k
        )��P	6u����A�*

	conv_lossAf.?���f        )��P	#Iu����A�*

	conv_loss�T.?�&�M        )��P	�mu����A�*

	conv_loss�;.?���        )��P	��u����A�*

	conv_loss\.?pR[>        )��P	�u����A�*

	conv_loss� .?��        )��P	#�u����A�*

	conv_loss�%.?w�&        )��P	��u����A�*

	conv_loss�.?st�a        )��P	v����A�*

	conv_lossE.?�n�<        )��P	�<v����A�*

	conv_loss�.?'<�        )��P	u^v����A�*

	conv_lossJ8.?M��        )��P	�v����A�*

	conv_loss$�-?�R%�        )��P	��v����A�*

	conv_loss�.?+���        )��P	#�v����A�*

	conv_loss�-?��        )��P	c�v����A�*

	conv_loss?9.?# 9+        )��P	,w����A�*

	conv_loss��-?����        )��P	�+w����A�*

	conv_loss^�-?y��        )��P	�Mw����A�*

	conv_loss�-?h��.        )��P	�qw����A�*

	conv_lossL.?��e�        )��P	.�w����A�*

	conv_loss��-?؄O        )��P	�w����A�*

	conv_loss��-?�}��        )��P	/�w����A�*

	conv_loss��-?%v2        )��P	�x����A�*

	conv_loss��-?&�P�        )��P	'&x����A�*

	conv_loss�
.?�3=%        )��P	IKx����A�*

	conv_loss5�-?&��l        )��P	�mx����A�*

	conv_lossf�-?��,Q        )��P	��x����A�*

	conv_loss��-?yρ        )��P	M�x����A�*

	conv_loss|�-?'���        )��P	E�x����A�*

	conv_loss��-?�5��        )��P	h�x����A�*

	conv_lossT�-?��#s        )��P	
y����A�*

	conv_lossh�-?�j�        )��P	]6y����A�*

	conv_loss��-?�<�        )��P	xWy����A�*

	conv_lossc�-?a�r�        )��P	1wy����A�*

	conv_losss�-?V��a        )��P	��y����A�*

	conv_lossZ�-?FY��        )��P	�y����A�*

	conv_loss�-?C���        )��P	�y����A�*

	conv_lossU�-?�L�E        )��P	v�y����A�*

	conv_loss�-?LUJ        )��P	 z����A�*

	conv_loss�-?��T        )��P	 =z����A�*

	conv_loss|�-?K`�H        )��P	nz����A�*

	conv_lossi�-?i���        )��P	��z����A�*

	conv_loss)�-?=���        )��P	h�z����A�*

	conv_lossڮ-?,��        )��P	��z����A�*

	conv_loss��-?�*�        )��P	3�z����A�*

	conv_lossF�-?D^        )��P	�&{����A�*

	conv_loss��-?l��        )��P	fT{����A�*

	conv_loss�-?[,q�        )��P	Cu{����A�*

	conv_loss��-?E¼        )��P	��{����A�*

	conv_loss��-?Ȇ��        )��P	V�{����A�*

	conv_loss�-?��        )��P	��{����A�*

	conv_loss�-?���B        )��P	@�{����A�*

	conv_loss�y-??�6]        )��P	�|����A�*

	conv_lossBc-?��)�        )��P	�B|����A�*

	conv_loss�s-?�6j�        )��P	;i|����A�*

	conv_lossۂ-?���"        )��P	΍|����A�*

	conv_loss��-?绍�        )��P	��|����A�*

	conv_loss�-?9��V        )��P	b�|����A�*

	conv_loss"�-?M��p        )��P	X�|����A�*

	conv_loss�-?���h        )��P	�}����A�*

	conv_loss�m-?�z        )��P	�;}����A�*

	conv_loss��-?�
x�        )��P	\}����A�*

	conv_lossk�-?�F�        )��P	�~}����A�*

	conv_losshi-?��        )��P	a�}����A�*

	conv_loss�c-?�N�        )��P	�}����A�*

	conv_loss�[-?l5�        )��P	�}����A�*

	conv_loss�r-?�u>�        )��P	��}����A�*

	conv_loss�z-?�Q��        )��P	�"~����A�*

	conv_loss̚-?��L        )��P	�D~����A�*

	conv_loss�t-?n�o        )��P	zh~����A�*

	conv_lossQW-?y��D        )��P	X�~����A�*

	conv_lossdR-?#���        )��P	�~����A�*

	conv_loss�G-?j��        )��P	@�~����A�*

	conv_loss�*-?0�        )��P	P�~����A�*

	conv_loss�x-?�5��        )��P	�����A�*

	conv_loss@g-?jy�        )��P	�7����A�*

	conv_loss0G-?n"�        )��P	w[����A�*

	conv_loss�q-?m"Ћ        )��P	'|����A�*

	conv_loss"/-?N�Fo        )��P	������A�*

	conv_loss�!-?j���        )��P	g�����A�*

	conv_lossDO-?����        )��P	������A�*

	conv_losscE-?���        )��P	������A�*

	conv_loss�_-?�<��        )��P	������A�*

	conv_lossm1-?�qD        )��P	&A�����A�*

	conv_loss'-?rV�_        )��P	�b�����A�*

	conv_loss�)-?{F�@        )��P	܃�����A�*

	conv_loss4:-? [�u        )��P	7������A�*

	conv_loss� -?�hW�        )��P	ŀ����A�*

	conv_loss�-?E�g=        )��P	������A�*

	conv_loss�#-?��f        )��P	������A�*

	conv_loss2-?�f�        )��P	D�����A�*

	conv_loss�H-?���        )��P	�=�����A�*

	conv_lossX-?*���        )��P	?^�����A�*

	conv_loss��,?c�        )��P	������A�*

	conv_loss?6-?��\        )��P	�������A�*

	conv_lossf$-?1lʏ        )��P	������A�*

	conv_loss�.-?���        )��P	������A�*

	conv_loss$�,?��        )��P	������A�*

	conv_lossU�,?���        )��P	�%�����A�*

	conv_loss��,?����        )��P	�G�����A�*

	conv_loss�-?�B�y        )��P	h�����A�*

	conv_lossC(-?�b_R        )��P	�������A�*

	conv_loss9�,?6�{        )��P	������A�*

	conv_loss�,?�O��        )��P	M�����A�*

	conv_loss>!-?�#�        )��P	������A�*

	conv_loss[�,?JWؿ        )��P	�0�����A�*

	conv_lossy�,?e��        )��P	gW�����A�*

	conv_loss��,?F�~\        )��P	ۄ�����A�*

	conv_loss�,?���<        )��P	�������A�*

	conv_lossg�,?IĠ        )��P	�̄����A�*

	conv_loss
�,?�`L�        )��P	S�����A�*

	conv_loss��,?��a        )��P	[�����A�*

	conv_loss[-?���        )��P	�7�����A�*

	conv_loss8�,?Z�[        )��P	Y�����A�*

	conv_losse�,?�f�f        )��P	z�����A�*

	conv_lossK	-?8n�}        )��P	�������A�*

	conv_loss�,?��E        )��P	p������A�*

	conv_loss��,?ҝ��        )��P	h݅����A�*

	conv_loss��,?7D-        )��P	�������A�*

	conv_loss��,?*;j	        )��P	�$�����A�*

	conv_loss��,?�c�#        )��P	�G�����A�*

	conv_lossX�,?� �        )��P	�j�����A�*

	conv_lossS�,?ן�e        )��P	o������A�*

	conv_loss��,?�o��        )��P	e������A�*

	conv_lossU�,?�Kq�        )��P	~ֆ����A�*

	conv_loss�,?Oʸh        )��P	�������A�*

	conv_lossA�,?�O��        )��P	�����A�*

	conv_loss�,?c�jR        )��P	A�����A�*

	conv_lossa�,?6��        )��P	(d�����A�*

	conv_loss�~,?Z�JP        )��P	������A�*

	conv_lossU�,?�
�        )��P	ɪ�����A�*

	conv_lossRw,?>��        )��P	͇����A�*

	conv_lossʳ,?t�l        )��P	�������A�*

	conv_loss\�,?��P        )��P	�����A�*

	conv_loss�,?�3S�        )��P	�5�����A�*

	conv_loss�h,?�)c        )��P	>X�����A�*

	conv_loss�f,?�u6        )��P	z�����A�*

	conv_loss�q,?�H�        )��P	�������A�*

	conv_loss�,?�
IE        )��P	�������A�*

	conv_loss j,?(բ�        )��P	������A�*

	conv_loss�,?�	�s        )��P	������A�*

	conv_loss�c,?Xٺ        )��P	4�����A�*

	conv_loss�{,?Ʋ�        )��P	�X�����A�*

	conv_loss�O,?e!�        )��P	)|�����A�*

	conv_loss�`,?J^-�        )��P	~������A�*

	conv_loss�v,?@.6l        )��P	!ĉ����A�*

	conv_loss�g,?���Q        )��P	c�����A�*

	conv_loss{w,?�L�M        )��P	������A�*

	conv_loss�Q,?H�8        )��P	�)�����A�*

	conv_loss"`,?�)�        )��P	�L�����A�*

	conv_loss\},?U���        )��P	�p�����A�*

	conv_loss�K,?]��9        )��P	0������A�*

	conv_loss%�,?Ez3        )��P	�Ŋ����A�*

	conv_loss�B,?���        )��P	.�����A�*

	conv_loss�,?[�E        )��P	?�����A�*

	conv_loss�^,?S�~        )��P	M3�����A�*

	conv_lossaR,?�4p�        )��P	�U�����A�*

	conv_loss%W,?g%5        )��P	$y�����A�*

	conv_lossi+,?H�}�        )��P	�������A�*

	conv_loss4I,?���        )��P	�Ӌ����A�*

	conv_loss�k,?��w�        )��P	�������A�*

	conv_lossQ],?A��        )��P	7�����A�*

	conv_lossC',?Ƭ�_        )��P	>�����A�*

	conv_loss�*,?V�Ji        )��P	s`�����A�*

	conv_lossS/,?�:�        )��P	������A�*

	conv_loss�,?�0�        )��P	諌����A�*

	conv_loss��+?�m�         )��P	�Ό����A�*

	conv_lossA,?o�I^        )��P	 �����A�*

	conv_loss�&,?�B�        )��P	�����A�*

	conv_loss!),?��        )��P	�5�����A�*

	conv_lossl,?��        )��P	�X�����A�*

	conv_loss��+?Κ��        )��P	�~�����A�*

	conv_losseA,?�䇥        )��P	S������A�*

	conv_loss��+?>*�8        )��P	�������A�*

	conv_lossw/,?�:�J        )��P	������A�*

	conv_loss*,?�89�        )��P	������A�*

	conv_loss;
,?F�j        )��P	�$�����A�*

	conv_loss�,?[T�
        )��P	CI�����A�*

	conv_loss�",?\�        )��P	�m�����A�*

	conv_loss�,?W���        )��P	�������A�*

	conv_loss��+?�k�        )��P	������A�*

	conv_loss	,?;���        )��P	�ӎ����A�*

	conv_lossy,?���        )��P	�������A�*

	conv_loss��+?S��        )��P	������A�*

	conv_loss&�+?����        )��P	�8�����A�*

	conv_loss��+?v�#        )��P	�[�����A�*

	conv_loss��+?7��D        )��P	Y~�����A�*

	conv_loss!�+?R"sq        )��P	�������A�*

	conv_lossg�+?���        )��P	Hŏ����A�*

	conv_loss�+?��O        )��P	�t�����A�*

	conv_lossJ�+?J�Q        )��P	?������A�*

	conv_loss��+?��        )��P	�Ɣ����A�*

	conv_loss��+?�K�y        )��P	������A�*

	conv_loss�+?��04        )��P	2�����A�*

	conv_lossu�+?j��        )��P	�)�����A�*

	conv_loss|�+?[o�u        )��P	YK�����A�*

	conv_loss��+?���\        )��P	Fl�����A�*

	conv_loss��+?N6��        )��P	�������A�*

	conv_loss�v+?AR6�        )��P	������A�*

	conv_lossm�+?����        )��P	p˕����A�*

	conv_loss��+?�/        )��P	 �����A�*

	conv_lossj�+?���        )��P	������A�*

	conv_lossj�+?k	        )��P	(G�����A�*

	conv_loss�+?3�g        )��P	Xi�����A�*

	conv_loss��+?1��        )��P	.������A�*

	conv_loss�+?�.%&        )��P	�������A�*

	conv_loss��+?[��"        )��P	#̖����A�*

	conv_lossW+?(X�8        )��P	H�����A�*

	conv_lossܬ+?��3        )��P	=�����A�*

	conv_loss��+?��{        )��P	R4�����A�*

	conv_loss�z+?���        )��P	�T�����A�*

	conv_loss�p+?E1�g        )��P	 t�����A�*

	conv_loss5�+?~�        )��P	�������A�*

	conv_loss�+?��8�        )��P	�������A�*

	conv_loss��+?<F        )��P	�֗����A�*

	conv_loss�+?���>        )��P	�������A�*

	conv_lossa�+?���P        )��P	������A�*

	conv_loss)�+?8]�        )��P	Q=�����A�*

	conv_loss��+?��ֲ        )��P	�b�����A�*

	conv_lossH{+?7=n�        )��P	 ������A�*

	conv_loss��+?Kp&/        )��P	'������A�*

	conv_loss�r+?�^��        )��P	�ɘ����A�*

	conv_loss^c+?1��	        )��P	������A�*

	conv_loss�x+?yN�l        )��P	������A�*

	conv_lossY�+?�2��        )��P	)%�����A�*

	conv_loss�b+?j�4        )��P	�D�����A�*

	conv_loss�O+?zQ�        )��P	 d�����A�*

	conv_loss�+?        )��P	������A�*

	conv_loss�e+?���        )��P	�������A�*

	conv_lossS~+?�'�y        )��P	�ř����A�*

	conv_loss�c+?�4)        )��P	������A�*

	conv_loss�O+?|.��        )��P	�����A�*

	conv_loss[+?g:U        )��P	,'�����A�*

	conv_loss�O+?pY�        )��P	FG�����A�*

	conv_loss�+?9��        )��P	re�����A�*

	conv_lossMO+?~���        )��P	�������A�*

	conv_loss�$+?���V        )��P	l������A�*

	conv_loss�{+?��Y�        )��P	Ś����A�*

	conv_lossh8+?���        )��P	u�����A�*

	conv_losss7+?]���        )��P	������A�*

	conv_lossf+?+X_        )��P	�1�����A�*

	conv_lossVA+?��Ա        )��P	5R�����A�*

	conv_loss�)+?͜�        )��P	,q�����A�*

	conv_loss�L+?���a        )��P	쏛����A�*

	conv_loss�@+?'a
�        )��P	ڲ�����A�*

	conv_loss�++?�T�        )��P	4ԛ����A�*

	conv_loss��*?H�s        )��P	�������A�*

	conv_loss�+?�~C        )��P	������A�*

	conv_loss�+?�wF�        )��P	�8�����A�*

	conv_lossN�*?��\�        )��P	q\�����A�*

	conv_loss8�*?�vß        )��P	E������A�*

	conv_loss;+?�ǑR        )��P	�������A�*

	conv_lossI�*?�Uڰ        )��P	�М����A�*

	conv_loss��*?����        )��P	�����A�*

	conv_lossɚ*?8?�        )��P	������A�*

	conv_loss+?G��        )��P	�4�����A�*

	conv_losse+?�s�        )��P	!U�����A�*

	conv_loss+?_ۛ�        )��P	{������A�*

	conv_loss��*?O��        )��P	�������A�*

	conv_loss�+?�]H�        )��P	tȝ����A�*

	conv_loss�*?����        )��P	�����A�*

	conv_loss��*?f�_D        )��P	v�����A�*

	conv_lossy�*?�/�        )��P	T-�����A�*

	conv_lossV�*?);        )��P	�N�����A�*

	conv_lossq�*?�$|        )��P	zn�����A�*

	conv_loss��*?��\�        )��P	X������A�*

	conv_loss��*?j���        )��P	�������A�*

	conv_loss�*?��        )��P	�Ϟ����A�*

	conv_loss˓*?�hW�        )��P	�����A�*

	conv_loss�*??��        )��P	d�����A�*

	conv_loss��*?<,	�        )��P	�-�����A�*

	conv_loss�*?H���        )��P	�M�����A�*

	conv_loss/�*?[�U2        )��P	jl�����A�*

	conv_loss]�*?���6        )��P	�����A�*

	conv_loss�*?˲P?        )��P	�������A�*

	conv_loss1�*?�=        )��P	Dџ����A�*

	conv_loss(�*?԰�        )��P	������A�*

	conv_loss>�*?�\�"        )��P	������A�*

	conv_loss�}*?�j�        )��P	U2�����A�*

	conv_loss@r*?����        )��P	vR�����A�*

	conv_loss��*?s�T        )��P	r�����A�*

	conv_lossm�*?�
�        )��P	����A�*

	conv_loss?�*?�R��        )��P	������A�*

	conv_loss�*?����        )��P	�Ӡ����A�*

	conv_loss�~*?w-n        )��P	�������A�*

	conv_losse�*?��4.        )��P	u�����A�*

	conv_loss)�*?Ru��        )��P	h7�����A�*

	conv_loss9�*?�        )��P	*X�����A�*

	conv_loss�?*?P�C�        )��P	�w�����A�*

	conv_loss�s*?zs�G        )��P	l������A�*

	conv_lossې*?���        )��P	䧢����A�*

	conv_lossl*?��A�        )��P	Ȣ����A�*

	conv_loss�u*?%nx�        )��P	������A�*

	conv_lossbt*?]^h        )��P	�
�����A�*

	conv_lossK*?3�-V        )��P	�*�����A�*

	conv_loss�*?�m�        )��P	8K�����A�*

	conv_lossXc*?x]�[        )��P	�k�����A�*

	conv_loss��*?y���        )��P	ɋ�����A�*

	conv_lossBb*?5���        )��P	�������A�*

	conv_loss�o*?��;�        )��P	o٣����A�*

	conv_lossM*?���        )��P	�������A�*

	conv_lossd*?7^��        )��P	�����A�*

	conv_lossc6*?x�`�        )��P	�B�����A�*

	conv_loss�S*?��^        )��P	�e�����A�*

	conv_loss�H*?����        )��P	������A�*

	conv_lossV(*?~R��        )��P	}������A�*

	conv_loss�.*?c�-        )��P	�ͤ����A�*

	conv_loss�*? ��`        )��P	������A�*

	conv_loss�9*?{ ��        )��P	�����A�*

	conv_loss�6*?��ɷ        )��P	�9�����A�*

	conv_loss<*?L(��        )��P	_�����A�*

	conv_loss�Y*?�p{        )��P	�������A�*

	conv_loss�*?k�        )��P	������A�*

	conv_loss��)?�_�        )��P	�ɥ����A�*

	conv_loss�*?AF��        )��P	\�����A�*

	conv_loss,*?��w�        )��P	������A�*

	conv_loss�&*?w�)x        )��P	�,�����A�*

	conv_loss*?�g�        )��P	-N�����A�*

	conv_loss_'*?5�N�        )��P	�o�����A�*

	conv_loss�*?���        )��P	j������A�*

	conv_loss�*?�!&�        )��P	ɱ�����A�*

	conv_loss�*?R�a�        )��P	>Ѧ����A�*

	conv_lossE�)?>q�G        )��P	�������A�*

	conv_loss�;*?]-�        )��P	������A�*

	conv_loss�*?6lE        )��P	B7�����A�*

	conv_loss4*?"9��        )��P	}X�����A�*

	conv_loss��)?�E�h        )��P	x�����A�*

	conv_loss�)?/c�"        )��P	ʘ�����A�*

	conv_loss��)?�֫�        )��P	{������A�*

	conv_loss��)?B���        )��P	ڧ����A�*

	conv_loss��)?~z        )��P	������A�*

	conv_lossT�)?�*�        )��P	������A�*

	conv_lossS�)?RT�        )��P	�:�����A�*

	conv_loss��)?�52        )��P	�Z�����A�*

	conv_loss�)?YG��        )��P	jz�����A�*

	conv_lossx�)?� ��        )��P	қ�����A�*

	conv_loss�)?}]��        )��P	a������A�*

	conv_loss��)?ޑS�        )��P	_ݨ����A�*

	conv_lossR�)?��d$        )��P	W������A�*

	conv_losso�)?eP�        )��P	J�����A�*

	conv_loss|�)?�i8�        )��P	cK�����A�*

	conv_loss)?�[�        )��P	�i�����A�*

	conv_lossl�)?�[        )��P	�������A�*

	conv_loss˗)?�@        )��P	�������A�*

	conv_lossئ)?o�}        )��P	�ʩ����A�*

	conv_lossӝ)?$��=        )��P	������A�*

	conv_loss�)?��C�        )��P	������A�*

	conv_lossK�)?Hr�9        )��P	O0�����A�*

	conv_loss�)?��o�        )��P	�Q�����A�*

	conv_lossB�)?^�!�        )��P	�p�����A�*

	conv_loss{�)?�D�?        )��P	������A�*

	conv_loss�)?�h        )��P	�������A�*

	conv_loss׫)?����        )��P	�֪����A�*

	conv_lossć)?�/��        )��P	�������A�*

	conv_lossΎ)?v�w�        )��P	�"�����A�*

	conv_loss j)?�\{        )��P	D�����A�*

	conv_loss��)?��]        )��P	�e�����A�*

	conv_loss�t)?�nW        )��P	�������A�*

	conv_loss�)?����        )��P	7������A�*

	conv_loss#`)?iT8        )��P	ǫ����A�*

	conv_loss�l)?��`4        )��P	������A�*

	conv_loss:s)?\M҈        )��P	:�����A�*

	conv_lossM)?G�,E        )��P	�4�����A�*

	conv_loss�n)?EZN        )��P	�U�����A�*

	conv_lossr�)?ԫte        )��P	Ry�����A�*

	conv_lossBh)?q�W        )��P	[������A�*

	conv_lossb)?�Nx�        )��P	 Ŭ����A�*

	conv_lossL)?_�]        )��P	������A�*

	conv_loss�E)?��7        )��P	�����A�*

	conv_loss�@)?����        )��P	�-�����A�*

	conv_loss�/)?�S<        )��P	�N�����A�*

	conv_loss�<)?탌!        )��P	Io�����A�*

	conv_loss-W)?�%        )��P	�������A�*

	conv_loss|2)?U�
        )��P	<������A�*

	conv_loss�c)?���4        )��P	gԭ����A�*

	conv_loss�])?��Ե        )��P	�������A�*

	conv_loss�7)?��t�        )��P	������A�*

	conv_loss#1)?*`N^        )��P	,7�����A�*

	conv_loss�[)?����        )��P	�V�����A�*

	conv_lossG�(?7��X        )��P	y�����A�*

	conv_loss� )?b��3        )��P	������A�*

	conv_loss�)?
ۂ        )��P	&������A�*

	conv_loss��(?�יP        )��P	8ݮ����A�*

	conv_loss�))?�F��        )��P	������A�*

	conv_loss�)?��T        )��P	�-�����A�*

	conv_loss)?���c        )��P	�M�����A�*

	conv_lossu)?��VZ        )��P	�n�����A�*

	conv_lossC)?���        )��P	������A�*

	conv_loss��(?�*�g        )��P	ֲ�����A�*

	conv_loss��(?vN�        )��P	�ӯ����A�*

	conv_lossb)?�xRu        )��P	"�����A�*

	conv_loss��(?���        )��P	Q$�����A�*

	conv_loss�(?B�ɿ        )��P	�D�����A�*

	conv_loss��(?�l�g        )��P	�e�����A�*

	conv_loss��(?)�
N        )��P	3������A�*

	conv_loss̺(?�M�        )��P	٫�����A�*

	conv_lossM�(?x3�        )��P	�̰����A�*

	conv_lossj�(?%µ(        )��P	������A�*

	conv_loss�(?�Gu�        )��P	������A�*

	conv_lossF�(?�h�        )��P	�5�����A�*

	conv_loss��(?�8�        )��P	�U�����A�*

	conv_loss��(?e�s        )��P	�u�����A�*

	conv_lossF�(?�kH�        )��P	Ѥ�����A�*

	conv_loss��(?�rm�        )��P	Aб����A�*

	conv_loss�(?����        )��P	4�����A�*

	conv_lossD�(?���<        )��P	������A�*

	conv_lossG�(?��l�        )��P	�>�����A�*

	conv_loss\�(?���        )��P	"a�����A�*

	conv_loss�o(?=�        )��P	�������A�*

	conv_loss�(?~���        )��P	�������A�*

	conv_loss�(?��#&        )��P	�²����A�*

	conv_loss8�(?��̬        )��P	�����A�*

	conv_loss��(?��4�        )��P	������A�*

	conv_lossΕ(?�;O        )��P	$�����A�*

	conv_loss�^(?���        )��P	}E�����A�*

	conv_loss�a(?��+        )��P	�g�����A�*

	conv_loss�m(?t��        )��P	↳����A�*

	conv_loss+�(?{��        )��P	�������A�*

	conv_loss��(?翍�        )��P	oǳ����A�*

	conv_loss��(? b�        )��P	������A�*

	conv_lossW�(?��d        )��P	������A�*

	conv_lossh�(?��.8        )��P	o0�����A�*

	conv_lossd(?p�        )��P	�S�����A�*

	conv_losse^(?����        )��P	�y�����A�*

	conv_lossm~(?�3�U        )��P	�������A�*

	conv_lossj(?ż�        )��P	$������A�*

	conv_loss�_(?�c�)        )��P	������A�*

	conv_loss9(?��"        )��P	\�����A�*

	conv_loss�`(?B"        )��P	r$�����A�*

	conv_lossJ(?y��        )��P	�J�����A�*

	conv_lossH(?�e6�        )��P	�l�����A�*

	conv_lossB(?���        )��P	�������A�*

	conv_loss�](?�)         )��P	^������A�*

	conv_loss_,(?�c˼        )��P	^ϵ����A�*

	conv_lossSi(?=��{        )��P	������A�*

	conv_loss�Y(?��r�        )��P	o�����A�*

	conv_loss�N(?*l�        )��P	w?�����A�*

	conv_losse>(?���R        )��P	�`�����A�*

	conv_loss(?5��        )��P	�������A�*

	conv_loss��'?3.�        )��P	w������A�*

	conv_lossk(?M�D        )��P	�۶����A�*

	conv_loss�'(?��y        )��P	�������A�*

	conv_loss�7(?��4N        )��P	������A�*

	conv_lossd(?s2        )��P	�?�����A�*

	conv_loss�(?���x        )��P	�`�����A�*

	conv_loss�(?�ֱn        )��P	�������A�*

	conv_lossj�'?�0_        )��P	c������A�*

	conv_lossD�'?��y        )��P	�ŷ����A�*

	conv_loss�@(?ns-�        )��P	k�����A�*

	conv_loss��'?�D�=        )��P	������A�*

	conv_loss'2(?���"        )��P	�,�����A�*

	conv_lossD�'?AyE        )��P	U�����A�*

	conv_loss(?�t��        )��P	|�����A�*

	conv_loss��'?�9��        )��P	K������A�*

	conv_lossD(?���c        )��P	�¸����A�*

	conv_lossq	(?܋��        )��P	~�����A�*

	conv_losst�'?`A @        )��P	�����A�*

	conv_loss��'?��ލ        )��P	�)�����A�*

	conv_loss��'?��V        )��P	YY�����A�*

	conv_loss�'?uҺH        )��P	t{�����A�*

	conv_loss+�'?v2�        )��P	ל�����A�*

	conv_loss��'?b�)�        )��P	�������A�*

	conv_loss%�'?�`        )��P	$�����A�*

	conv_lossb�'?	��-        )��P	������A�*

	conv_lossq�'?t�kb        )��P	}$�����A�*

	conv_loss��'?[sp        )��P	;F�����A�*

	conv_loss�'?��        )��P	.f�����A�*

	conv_lossd�'?9���        )��P	�������A�*

	conv_loss%u'?w^��        )��P	������A�*

	conv_loss�'?P���        )��P	Ǻ����A�*

	conv_lossF�'?�qv�        )��P	������A�*

	conv_lossi�'?�A        )��P	9�����A�*

	conv_lossWt'?���        )��P	�-�����A�*

	conv_loss�9'?��U        )��P	�N�����A�*

	conv_loss��'?T��        )��P	�p�����A�*

	conv_lossd'?�)�        )��P	X������A�*

	conv_loss��'?��X        )��P	'������A�*

	conv_loss"I'?�^�,        )��P	�Ի����A�*

	conv_lossa�'?�/        )��P	e������A�*

	conv_lossLS'?����        )��P	������A�*

	conv_loss�z'?��*        )��P	�=�����A�*

	conv_lossͧ'?tL�        )��P	|b�����A�*

	conv_loss��'?��u�        )��P	������A�*

	conv_loss�/'?���,        )��P	�������A�*

	conv_lossxW'?���        )��P	o˼����A�*

	conv_lossg'?��Fw        )��P	������A�*

	conv_lossZI'?n��x        )��P	l�����A�*

	conv_loss�'?���        )��P	�3�����A�*

	conv_loss�'?3*<        )��P	!Y�����A�*

	conv_loss�N'?*D,�        )��P	|�����A�*

	conv_loss�'?֟&�        )��P	Ȍ�����A�*

	conv_lossf]'?��˂        )��P	s������A�*

	conv_lossg{'?��L5        )��P	aӾ����A�*

	conv_loss�U'?�x�,        )��P	>�����A�*

	conv_loss��&?�h�        )��P	\�����A�*

	conv_lossT'?�]1        )��P	6�����A�*

	conv_loss|#'?�Z�        )��P	@X�����A�*

	conv_loss��&?���Q        )��P	�����A�*

	conv_loss�'?p��        )��P	ڣ�����A�*

	conv_loss��&?� ��        )��P	8ÿ����A�*

	conv_loss�'?9��l        )��P	U�����A�*

	conv_loss�&?�7k�        )��P	<�����A�*

	conv_lossG�&?���        )��P	�.�����A�*

	conv_losse�&?�         )��P	GN�����A�*

	conv_loss��&?�4�        )��P	Yn�����A�*

	conv_lossB�&?F>�q        )��P	k������A�*

	conv_loss=�&?F�z        )��P	�������A�*

	conv_loss�&?c��        )��P	�������A�*

	conv_lossA�&?�1�        )��P	�����A�*

	conv_loss?�&?v�G        )��P	�(�����A�*

	conv_loss��&?��ҵ        )��P	J�����A�*

	conv_loss��&?�
         )��P	�i�����A�*

	conv_lossi�&?�v�        )��P	������A�*

	conv_lossl�&?�;p        )��P	�������A�*

	conv_lossֺ&?��        )��P	�������A�*

	conv_lossK�&?��p        )��P	�������A�*

	conv_loss��&?�t��        )��P	������A�*

	conv_loss��&?�8�        )��P	�2�����A�*

	conv_loss�'?F��J        )��P	cU�����A�*

	conv_loss��&?`J�        )��P	u�����A�*

	conv_lossʒ&?���        )��P	|������A�*

	conv_loss�&?��C        )��P	и�����A�*

	conv_loss�&?�[�i        )��P	@������A�*

	conv_loss:b&?<tT        )��P	�������A�*

	conv_loss]&?���        )��P	�����A�*

	conv_loss�o&?�4�|        )��P	o?�����A�*

	conv_loss��&?�V�        )��P	c�����A�*

	conv_loss;�&?��{        )��P	�������A�*

	conv_loss��&?�Q;'        )��P	h������A�*

	conv_loss+q&?u�>        )��P	�������A�*

	conv_loss�H&?���E        )��P	�������A�*

	conv_loss�&?���        )��P	������A�*

	conv_losssp&?�XI        )��P	�'�����A�*

	conv_loss�&?0���        )��P	.K�����A�*

	conv_loss%n&?�@f�        )��P	>o�����A�*

	conv_loss�&&?}�C+        )��P	�������A�*

	conv_loss�H&?H+h        )��P	>������A�*

	conv_loss��&?j\�l        )��P	�������A�*

	conv_loss'�%?�˂�        )��P	�������A�*

	conv_loss�l&?`ݩ/        )��P	������A�*

	conv_loss�&?�+ �        )��P	mC�����A�*

	conv_loss&?��A�        )��P	f�����A�*

	conv_lossq&?���t        )��P	t������A�*

	conv_loss^W&?��C�        )��P	������A�*

	conv_loss�e&?��'        )��P	�������A�*

	conv_loss&;&?;E        )��P	e������A�*

	conv_loss�&?��,k        )��P	 �����A�*

	conv_loss�M&?���        )��P	,�����A�*

	conv_loss�$&?��]        )��P	�N�����A�*

	conv_loss<&? �        )��P	bp�����A�*

	conv_lossn�%?_O��        )��P	�������A�*

	conv_loss�L&?u�AU        )��P	E������A�*

	conv_loss&?�ϵ�        )��P	�������A�*

	conv_loss:&?���        )��P	�����A�*

	conv_loss��%?K��        )��P	d4�����A�*

	conv_loss;w%?μN        )��P	bU�����A�*

	conv_loss��%?p]w�        )��P	�w�����A�*

	conv_lossX,&?Gi9.        )��P	h������A�*

	conv_loss��%?�A[        )��P	�������A�*

	conv_loss�%?p���        )��P	Y������A�*

	conv_loss�0&?&��        )��P	0�����A�*

	conv_lossSt%?#Ƥ�        )��P	�4�����A�*

	conv_loss�%?��'        )��P	][�����A�*

	conv_loss��%?�R �        )��P	�����A�*

	conv_loss��%?[@�        )��P	{������A�*

	conv_loss�%?2��        )��P	������A�*

	conv_loss��%?-�j4        )��P	_������A�*

	conv_lossy�%?        )��P	u�����A�*

	conv_loss�%?O9_z        )��P	x&�����A�*

	conv_lossdX%?��#'        )��P	�G�����A�*

	conv_lossfe%?��"�        )��P	i�����A�*

	conv_loss�i%?��v        )��P	Ί�����A�*

	conv_lossX�%?��Y�        )��P	V������A�*

	conv_loss�F%?(mF        )��P	=������A�*

	conv_lossۧ%?��^        )��P	$������A�*

	conv_lossM�%?Mo��        )��P	������A�*

	conv_loss|C%?�l��        )��P	N8�����A�*

	conv_loss�x%?X���        )��P	R]�����A�*

	conv_loss7H%?"�        )��P	1�����A�*

	conv_loss^;%?��        )��P	ǡ�����A�*

	conv_loss`0%?���        )��P	�������A�*

	conv_loss�/%?��J�        )��P	�������A�*

	conv_loss%?��=+        )��P	������A�*

	conv_loss�%?�(\�        )��P	.�����A�*

	conv_loss�0%?����        )��P	�M�����A�*

	conv_loss�1%?.�Y        )��P	�p�����A�*

	conv_loss�)%?d`��        )��P	4������A�*

	conv_loss%?
[A�        )��P	`������A�*

	conv_loss%%?�xB        )��P	C������A�*

	conv_lossL0%?5ى�        )��P	�������A�*

	conv_loss�$?��
        )��P	�-�����A�*

	conv_loss�C%?�gȮ        )��P	-O�����A�*

	conv_lossV�$?���        )��P	o�����A�*

	conv_loss �$?�=W�        )��P	�������A�*

	conv_loss*�$?6���        )��P	�������A�*

	conv_loss��$?��H�        )��P	@������A�*

	conv_lossE�$?n        )��P	�������A�*

	conv_loss)�$?��9j        )��P	W�����A�*

	conv_loss�%?/`i        )��P	t>�����A�*

	conv_loss�$?�s�        )��P	�d�����A�*

	conv_lossV�$?>mF        )��P	#������A�*

	conv_loss��$?�UCW        )��P	P������A�*

	conv_loss��$?�B��        )��P	������A�*

	conv_loss��$?u2y]        )��P	= �����A�*

	conv_loss�$?�;        )��P	)"�����A�*

	conv_loss�$?-xZZ        )��P	�D�����A�*

	conv_loss��$?���        )��P	e�����A�*

	conv_loss�$?*���        )��P	�������A�*

	conv_loss�$?��p        )��P	
������A�*

	conv_loss�$?�6\        )��P	�������A�*

	conv_loss-�$?tW�#        )��P	������A�*

	conv_loss��$?U��        )��P	������A�*

	conv_loss�K$?�	��        )��P	i7�����A�*

	conv_lossי$?�3�        )��P	�Y�����A�*

	conv_loss�5$?�R��        )��P	�y�����A�*

	conv_loss��$?�A+        )��P	3������A�*

	conv_loss�$?P��        )��P	a������A�*

	conv_loss��$?�3�        )��P	�������A�*

	conv_lossd�$?�0�X        )��P	������A�*

	conv_loss�9$?��        )��P	�)�����A�*

	conv_loss��#?Z�9A        )��P	JL�����A�*

	conv_loss�B$??��(        )��P	l�����A�*

	conv_loss*&$?HS��        )��P		������A�*

	conv_loss�O$?6�T        )��P	�������A�*

	conv_loss=S$?k�        )��P	)������A�*

	conv_lossFF$?��O        )��P	�������A�*

	conv_loss�=$?��7�        )��P	������A�*

	conv_loss?�#?����        )��P	�=�����A�*

	conv_loss"$$?.���        )��P	�`�����A�*

	conv_loss��#?�p��        )��P	�������A�*

	conv_loss��#?��zj        )��P	������A�*

	conv_loss2�#?ŧ�        )��P	�������A�*

	conv_loss��#?};��        )��P	�������A�*

	conv_loss� $?N�V�        )��P	�
�����A�*

	conv_loss�/$?9��        )��P	�-�����A�*

	conv_loss(�#?�B�        )��P	*O�����A�*

	conv_lossE�#?D��.        )��P	�q�����A�*

	conv_loss!�#?��z7        )��P	y������A�*

	conv_losst�#?��vd        )��P	�������A�*

	conv_loss�$?�        )��P	�������A�*

	conv_lossѡ#?�ku�        )��P	������A�*

	conv_loss�#?��Gt        )��P	k%�����A�*

	conv_loss�$?��d�        )��P	HF�����A�*

	conv_loss\k#?���        )��P	�h�����A�*

	conv_loss�#?S.�a        )��P	�������A�*

	conv_loss�E#?�F/        )��P	������A�*

	conv_loss#?�4�        )��P	�������A�*

	conv_loss=N#?�?        )��P	�������A�*

	conv_loss�#?���e        )��P	8!�����A�*

	conv_loss��#?ga,        )��P	G�����A�*

	conv_loss�#?�x��        )��P	\i�����A�*

	conv_loss�s#?��]1        )��P	������A�*

	conv_loss�#?�*R�        )��P	T������A�*

	conv_lossX#?�{-�        )��P	�������A�*

	conv_loss�e#?�
�!        )��P	�������A�*

	conv_loss`'#?~ z]        )��P	a,�����A�*

	conv_loss=�"?��        )��P	>N�����A�*

	conv_loss_o#? �F�        )��P	^o�����A�*

	conv_lossQ7#?���w        )��P	�������A�*

	conv_loss�#?�X        )��P	Y������A�*

	conv_loss̽"?i
K�        )��P	������A�*

	conv_loss��#?���        )��P	@������A�*

	conv_lossEu#?��r        )��P	E�����A�*

	conv_loss
g"?Y�        )��P	�<�����A�*

	conv_loss!#?��w        )��P	�\�����A�*

	conv_loss��"?�&�        )��P	�|�����A�*

	conv_loss3�"?R��T        )��P	X������A�*

	conv_loss��"?켩�        )��P	�������A�*

	conv_loss�"?&��        )��P	������A�*

	conv_loss��"?��F'        )��P	�����A�*

	conv_loss{�"?��t�        )��P	�$�����A�*

	conv_lossO�"?��	�        )��P	9F�����A�*

	conv_loss]�"?zX�        )��P	hl�����A�*

	conv_lossF�"?��H�        )��P	�������A�*

	conv_loss+R"?iS        )��P	ش�����A�*

	conv_loss��"?)��&        )��P	������A�*

	conv_loss/}"?vL�"        )��P	������A�*

	conv_lossM"?�|��        )��P	r�����A�*

	conv_loss��"?n@�n        )��P	�:�����A�*

	conv_loss�1"?��        )��P	�]�����A�*

	conv_loss#�"?��=        )��P	�~�����A�*

	conv_lossW"?�I�e        )��P	4������A�*

	conv_loss�["?wT;Y        )��P	\������A�*

	conv_loss"?�W�        )��P	�������A�*

	conv_loss{J"?#�?U        )��P	������A�*

	conv_lossW�"??�
�        )��P	�%�����A�*

	conv_lossp�!?��        )��P	�F�����A�*

	conv_lossHx"?tذY        )��P	h�����A�*

	conv_lossb"?���        )��P	������A�*

	conv_loss�-"?��        )��P	3�����A�*

	conv_loss��!?�        )��P	�0�����A�*

	conv_loss�W"?*E��        )��P	cT�����A�*

	conv_losssV"?ΐ,        )��P	as�����A�*

	conv_loss*&"?�
        )��P	q������A�*

	conv_loss��!?�_        )��P	}������A�*

	conv_loss�!?�º�        )��P	�������A�*

	conv_loss�1!?�;Mh        )��P	�������A�*

	conv_loss�B"?���        )��P	������A�*

	conv_loss�!?��        )��P	"2�����A�*

	conv_loss�"?�\4]        )��P	�P�����A�*

	conv_lossd~!?�@w        )��P	|������A�*

	conv_loss��!?/�X        )��P	r������A�*

	conv_lossg�!?W �        )��P	N������A�*

	conv_loss�T"?���!        )��P	8������A�*

	conv_loss�F!?�a        )��P	������A�*

	conv_loss�k!?�!��        )��P	�/�����A�*

	conv_lossN�!? eX        )��P	�N�����A�*

	conv_loss��!?j�ف        )��P	o�����A�*

	conv_loss�_!?2�X�        )��P	͝�����A�*

	conv_loss��!?Ԃ�        )��P	�������A�*

	conv_loss�+!?Ĥ�(        )��P	�������A�*

	conv_lossn8!?�V��        )��P	������A�*

	conv_loss�5!?��#�        )��P	�&�����A�*

	conv_loss�>!?+' �        )��P	sF�����A�*

	conv_lossc?!?l��]        )��P	;e�����A�*

	conv_loss]B!?B�        )��P	I������A�*

	conv_lossSc!?�D70        )��P	������A�*

	conv_loss^!?�~Lv        )��P	!������A�*

	conv_loss�� ?��aB        )��P	|������A�*

	conv_lossm!?mc�        )��P	������A�*

	conv_loss�Y!?a]        )��P	�!�����A�*

	conv_lossG<!?ڔz�        )��P	�A�����A�*

	conv_loss�� ?^��        )��P	�a�����A�*

	conv_losszC!?���\        )��P	�������A�*

	conv_loss� ?XMF�        )��P	a������A�*

	conv_loss2� ?2-��        )��P	�������A�*

	conv_loss�s ?�K3        )��P	E������A�*

	conv_loss�z ?�<�         )��P	$�����A�*

	conv_loss�!?u��i        )��P	�:�����A�*

	conv_loss�� ?Z�*v        )��P	\Z�����A�*

	conv_lossX^ ?���        )��P	�|�����A�*

	conv_loss�� ?�vJx        )��P	[������A�*

	conv_loss̭ ?Z��        )��P	[������A�*

	conv_loss�� ?ϿQD        )��P	������A�*

	conv_loss� ?��        )��P	C�����A�*

	conv_losse� ?�(��        )��P	(�����A�*

	conv_loss�k ?�n�        )��P	KH�����A�*

	conv_loss]{ ?�6        )��P	�k�����A�*

	conv_loss�' ?lm�        )��P	������A�*

	conv_loss"�?m�L�        )��P	�������A�*

	conv_loss6 ?�	:�        )��P	�������A�*

	conv_loss��?ħ8�        )��P	�������A�*

	conv_lossӀ?���N        )��P	F�����A�*

	conv_loss�< ?0x��        )��P	=�����A�*

	conv_loss�' ?Nc��        )��P	�]�����A�*

	conv_losse� ?za��        )��P	U������A�*

	conv_loss�?q��r        )��P	������A�*

	conv_loss9P ?x?^�        )��P	�������A�*

	conv_loss��?Q��W        )��P	_������A�*

	conv_lossh ?�2^�        )��P	������A�*

	conv_lossa�?I��        )��P	�6�����A�*

	conv_loss� ?�%r        )��P	:Z�����A�*

	conv_lossHK?��0?        )��P	Q������A�*

	conv_loss�?�*^        )��P	ĩ�����A�*

	conv_loss�%?iꦗ        )��P	�������A�*

	conv_loss�  ?�	�        )��P	�������A�*

	conv_loss�?�ݻ        )��P	�����A�*

	conv_lossY�?��#}        )��P	�9�����A�*

	conv_loss�}?���        )��P	�[�����A�*

	conv_loss��?Ə@�        )��P	�~�����A�*

	conv_loss��?���        )��P	������A�*

	conv_loss�?>t�        )��P	�������A�*

	conv_lossϳ?u��        )��P	������A�*

	conv_lossJ�?�y�        )��P	������A�*

	conv_loss��?���        )��P	%9�����A�*

	conv_lossTB?
�K5        )��P	�\�����A�*

	conv_loss��?�3{        )��P	�~�����A�*

	conv_loss&6?J�/        )��P	������A�*

	conv_loss%�?��ҭ        )��P	�������A�*

	conv_loss/�?�B��        )��P	"������A�*

	conv_loss��?���-        )��P	�	�����A�*

	conv_lossk?����        )��P	),�����A�*

	conv_lossH?�"#�        )��P	�O�����A�*

	conv_loss@�?v�+        )��P	Wr�����A�*

	conv_loss�J?��,�        )��P	_������A�*

	conv_loss۠?3e�`        )��P	ȼ�����A�*

	conv_loss""?[g��        )��P	�������A�*

	conv_lossE�?����        )��P	c �����A�*

	conv_loss�?��V�        )��P	�#�����A�*

	conv_loss��?��&        )��P	�E�����A�*

	conv_loss�?}�Tt        )��P	�i�����A�*

	conv_loss�?�1��        )��P	B������A�*

	conv_loss@�?GQ�V        )��P	_������A�*

	conv_loss�?��o        )��P	�������A�*

	conv_loss��?�?�        )��P	�������A�*

	conv_loss��?5��2        )��P	������A�*

	conv_loss�-?؇�7        )��P	>�����A�*

	conv_loss�?-Z~b        )��P	`�����A�*

	conv_lossL3?�+        )��P	W������A�*

	conv_loss&�?צ`        )��P	������A�*

	conv_loss)?x�kT        )��P	�������A�*

	conv_loss-P?���        )��P	�������A�*

	conv_loss�K?΍ϫ        )��P	N�����A�*

	conv_loss��?='��        )��P	>�����A�*

	conv_loss�?-�        )��P	�a�����A�*

	conv_loss��?1H�        )��P	P������A�*

	conv_loss;?	�d�        )��P	q������A�*

	conv_loss�9?����        )��P	V������A�*

	conv_lossO.?$ }        )��P	�������A�*

	conv_lossE�?���;        )��P	������A�*

	conv_lossAz?B%,]        )��P	�H�����A�*

	conv_loss6Y?���        )��P	-m�����A�*

	conv_loss�k?�aJ�        )��P	j������A�*

	conv_lossf�?h�,        )��P	m������A�*

	conv_loss�?��+        )��P	C������A�*

	conv_loss&�?�        )��P	 �����A�*

	conv_loss��?=��        )��P	�"�����A�*

	conv_loss��?�r��        )��P	hE�����A�*

	conv_loss7�?�&�        )��P	�g�����A�*

	conv_loss:�?���K        )��P	,������A�*

	conv_loss�M?�'�        )��P	�������A�*

	conv_losse?+��e        )��P	H������A�*

	conv_loss7�?*Q)        )��P	�������A�*

	conv_lossv?62_F        )��P	R�����A�*

	conv_loss�?��N        )��P	�:�����A�*

	conv_loss�V?��#�        )��P	6^�����A�*

	conv_loss�<?�	@h        )��P	_������A�*

	conv_loss'?HR�Z        )��P	������A�*

	conv_loss+?�V        )��P	�������A�*

	conv_loss��?���        )��P	�����A�*

	conv_losszg?ޑ�        )��P	�%�����A�*

	conv_loss�?Ji&�        )��P	_I�����A�*

	conv_loss��?�x�o        )��P	�l�����A�*

	conv_lossi?|8]�        )��P	|������A�*

	conv_loss̰?�9�G        )��P	�������A�*

	conv_lossQ�?J\�        )��P	�������A�*

	conv_loss?A?���        )��P	�������A�*

	conv_loss7Q?�Fo�        )��P	������A�*

	conv_loss�{?mLd/        )��P	�A�����A�*

	conv_loss�f?��        )��P	c�����A�*

	conv_losss�?Ȑ'�        )��P	�������A�*

	conv_loss�6?O�
)        )��P	p������A�*

	conv_lossw�?���        )��P	������A�*

	conv_loss?��?
        )��P	H������A�*

	conv_loss+�?�N��        )��P	4�����A�*

	conv_loss�?��Q        )��P	T4�����A�*

	conv_loss�"?����        )��P	�W�����A�*

	conv_loss�?1io�        )��P	�{�����A�*

	conv_loss��?ְ&	        )��P	�������A�*

	conv_loss�R?�D��        )��P	տ�����A�*

	conv_loss)�?c���        )��P	�������A�*

	conv_lossL�?���        )��P	������A�	*

	conv_lossqp?�{o        )��P	N?�����A�	*

	conv_loss��?��s$        )��P	9`�����A�	*

	conv_loss"�?�Kd�        )��P	�������A�	*

	conv_loss��?�*?a        )��P	 ������A�	*

	conv_loss1�?uj'�        )��P	]������A�	*

	conv_lossi�?ɂ�        )��P	������A�	*

	conv_loss�@??4��        )��P	�!�����A�	*

	conv_lossE?�LO�        )��P	5D�����A�	*

	conv_loss-|?NP(�        )��P	�h�����A�	*

	conv_lossv?2��        )��P	`������A�	*

	conv_loss�?�Y�C        )��P	������A�	*

	conv_loss��?D���        )��P	�������A�	*

	conv_loss�>?3��        )��P	'	�����A�	*

	conv_loss�F?�8ۓ        )��P	Z0�����A�	*

	conv_loss??V�T        )��P	+R�����A�	*

	conv_loss'�?���        )��P	4s�����A�	*

	conv_loss�?�ھ�        )��P	{������A�	*

	conv_loss�?#N        )��P	�������A�	*

	conv_loss��?Y��        )��P	�������A�	*

	conv_loss��?�X        )��P	�������A�	*

	conv_lossrs?e�h{        )��P	,�����A�	*

	conv_lossMy?r;��        )��P	@O�����A�	*

	conv_loss��?{Gb�        )��P	 r�����A�	*

	conv_loss��?Z'�1        )��P	�������A�	*

	conv_loss�?\n��        )��P	Q������A�	*

	conv_loss#^?�ӛ        )��P	�������A�	*

	conv_loss��?��`�        )��P	Y �����A�	*

	conv_loss\�?<�$        )��P	G4�����A�	*

	conv_loss��?��-�        )��P	�]�����A�	*

	conv_loss�S?��Ǽ        )��P	o������A�	*

	conv_lossb�?�T1        )��P	u������A�	*

	conv_loss��?Rr��        )��P	�������A�	*

	conv_lossG?2��I        )��P	Y������A�	*

	conv_loss�#?�ף�        )��P	������A�	*

	conv_loss�?��        )��P	;�����A�	*

	conv_lossM.?���        )��P	_�����A�	*

	conv_lossc�?��~        )��P	������A�	*

	conv_losscW?\�c        )��P	]������A�	*

	conv_loss��?��Z|        )��P	�������A�	*

	conv_loss�q?�p�,        )��P	i������A�	*

	conv_lossv?��c�        )��P	|�����A�	*

	conv_loss�?��	        )��P	�1�����A�	*

	conv_loss(?���         )��P	eY�����A�	*

	conv_loss�U?ƀ��        )��P	,{�����A�	*

	conv_loss�,?��\        )��P	P������A�	*

	conv_loss�?sO\�        )��P	�������A�	*

	conv_loss�*?(_��        )��P	�������A�	*

	conv_loss�%?USG        )��P	-�����A�	*

	conv_loss%�?��        )��P	�,�����A�	*

	conv_loss~Y?o9��        )��P	Z�����A�	*

	conv_loss�?�d{        )��P	�}�����A�	*

	conv_loss��?����        )��P	1������A�	*

	conv_loss1?�.        )��P	�������A�	*

	conv_loss��?o>�        )��P	a������A�	*

	conv_loss��?I�Mv        )��P	������A�	*

	conv_loss�??�yd        )��P	�3�����A�	*

	conv_loss??���o        )��P	Yf�����A�	*

	conv_loss�0?����        )��P	�������A�	*

	conv_loss�?h�%�        )��P	@������A�	*

	conv_lossO?f�ޚ        )��P	E������A�	*

	conv_loss�*?K�o�        )��P	� �����A�	*

	conv_loss�:?c�3�        )��P	�#�����A�	*

	conv_lossi�?�(xX        )��P	eE�����A�	*

	conv_loss��?+N;�        )��P	�l�����A�	*

	conv_loss�R?��x6        )��P	�������A�	*

	conv_lossRb?@
S�        )��P	P������A�	*

	conv_loss��?_�y        )��P	?������A�	*

	conv_lossF?���s        )��P	�������A�	*

	conv_loss�?�z��        )��P	Y�����A�	*

	conv_loss��?ǽ8I        )��P	=�����A�	*

	conv_loss��?=��P        )��P	�l�����A�	*

	conv_lossI�?��        )��P	�������A�	*

	conv_loss�?�+ul        )��P	������A�	*

	conv_loss�Y?�0�        )��P	�������A�	*

	conv_loss��?�n�        )��P	_������A�	*

	conv_loss�=?��_�        )��P	� ����A�	*

	conv_loss�?�`l�        )��P	jI ����A�	*

	conv_lossB�?��M        )��P	$m ����A�	*

	conv_loss=�?i���        )��P	�� ����A�	*

	conv_loss�:?���L        )��P	F� ����A�	*

	conv_loss�?�S)F        )��P	�� ����A�	*

	conv_lossF;?�8��        )��P	�� ����A�	*

	conv_lossb�?D�Z�        )��P	�����A�	*

	conv_loss6�?��5�        )��P	�=����A�	*

	conv_loss��?%���        )��P	�a����A�	*

	conv_loss��?���        )��P	T�����A�	*

	conv_loss �?��W�        )��P	������A�	*

	conv_loss�(?�c�        )��P	������A�	*

	conv_lossm:?��        )��P	������A�	*

	conv_lossV�?�>l        )��P	����A�	*

	conv_loss�n?��        )��P	�>����A�	*

	conv_loss7?�=��        )��P	�c����A�	*

	conv_lossъ?�ة-        )��P	.�����A�	*

	conv_lossJ*?���        )��P	<�����A�	*

	conv_loss�_?�i0X        )��P	+�����A�	*

	conv_lossS�?�tw�        )��P	�����A�	*

	conv_lossz�?�?�{        )��P	:1����A�	*

	conv_loss_�?�.        )��P	�S����A�	*

	conv_loss�O?���        )��P	>�����A�	*

	conv_loss�W?���        )��P	ϫ����A�	*

	conv_loss9d?.٢        )��P	������A�	*

	conv_loss�^?|Z�        )��P	������A�	*

	conv_loss�?E�r?        )��P	�����A�	*

	conv_loss5�?AtE        )��P	^8����A�	*

	conv_loss	M?V��V        )��P	GZ����A�	*

	conv_loss�?4�+        )��P	}����A�	*

	conv_lossO�?#%L        )��P	������A�	*

	conv_lossB�?��@�        )��P	������A�	*

	conv_loss�?�Ε        )��P	`�����A�	*

	conv_lossV+?�f��        )��P	�����A�	*

	conv_losss�?bjU0        )��P	�8����A�	*

	conv_lossK?x<Ν        )��P	ra����A�	*

	conv_loss�?��1        )��P	�����A�	*

	conv_loss�q?pYxW        )��P	������A�	*

	conv_loss�?�l��        )��P	2�����A�	*

	conv_lossc�?�|��        )��P	K�����A�	*

	conv_loss��?{	��        )��P	����A�	*

	conv_lossHZ?I���        )��P	�4����A�	*

	conv_lossZ�?��./        )��P	�V����A�	*

	conv_loss�^?�� $        )��P	�x����A�	*

	conv_loss�1?.��A        )��P	�����A�	*

	conv_loss�(?Q�        )��P	������A�	*

	conv_loss�?M�Zs        )��P	������A�	*

	conv_loss*�?�bV�        )��P	�����A�	*

	conv_loss�_?�q�        )��P	�'����A�	*

	conv_loss��?[���        )��P	�I����A�	*

	conv_loss��?�a�F        )��P	�l����A�	*

	conv_loss�?���N        )��P	̞����A�
*

	conv_loss��?�`�C        )��P	�����A�
*

	conv_loss �?~��        )��P	������A�
*

	conv_loss0&?�z�        )��P	3
����A�
*

	conv_loss��?��        )��P	�+����A�
*

	conv_loss��?��C        )��P	�P����A�
*

	conv_lossY�?��
        )��P	u����A�
*

	conv_loss�?�5�v        )��P	W�����A�
*

	conv_loss�^?<���        )��P	������A�
*

	conv_loss��?&�I�        )��P	������A�
*

	conv_loss�o?����        )��P	S	����A�
*

	conv_loss��?���        )��P	�)	����A�
*

	conv_loss��?|iN        )��P	�N	����A�
*

	conv_lossM7?rY~	        )��P	+q	����A�
*

	conv_loss�2?'�@�        )��P	@�	����A�
*

	conv_loss��?uA=�        )��P	[�	����A�
*

	conv_loss��?�ǯ        )��P	}�	����A�
*

	conv_lossuz
?ω        )��P	^
����A�
*

	conv_loss�?���        )��P	a#
����A�
*

	conv_loss��?����        )��P	F
����A�
*

	conv_loss��?�4��        )��P	�i
����A�
*

	conv_loss��
?���        )��P	Q�
����A�
*

	conv_loss�2
?���        )��P	R�
����A�
*

	conv_loss|�?�n        )��P	��
����A�
*

	conv_loss`b?��S        )��P	����A�
*

	conv_loss�z
?�d�        )��P	J4����A�
*

	conv_loss�3	?�=F        )��P	�l����A�
*

	conv_loss�	?���        )��P	ď����A�
*

	conv_loss)	?��?�        )��P	�����A�
*

	conv_loss�H	?¢�G        )��P	������A�
*

	conv_loss2�	?�x#        )��P	�����A�
*

	conv_loss�	?���        )��P	�%����A�
*

	conv_loss�R	?�
��        )��P	�H����A�
*

	conv_loss~�	?�T�x        )��P	�n����A�
*

	conv_loss�?H�{�        )��P	.�����A�
*

	conv_loss�?���        )��P	������A�
*

	conv_loss��?Ia��        )��P	�����A�
*

	conv_loss??V[?�        )��P	K����A�
*

	conv_loss��?�d�        )��P	�>����A�
*

	conv_loss*x	?��b        )��P	�a����A�
*

	conv_loss�>?�j�        )��P	Y�����A�
*

	conv_loss��?/}V        )��P	<�����A�
*

	conv_loss'�	?c��        )��P	y�����A�
*

	conv_loss�0?��1b        )��P	g����A�
*

	conv_lossc+?�)R�        )��P	�(����A�
*

	conv_loss[�?��Jl        )��P	�K����A�
*

	conv_loss��?Bҧ�        )��P	�n����A�
*

	conv_lossTY?�|�        )��P	������A�
*

	conv_loss�C?#�Ǒ        )��P	Z�����A�
*

	conv_loss1�?R4��        )��P	������A�
*

	conv_lossQ�?�2>�        )��P	P	����A�
*

	conv_lossA�?<�8�        )��P	X2����A�
*

	conv_loss��?h�\B        )��P	�V����A�
*

	conv_loss#F?*��        )��P	wx����A�
*

	conv_loss��?��7_        )��P	ߙ����A�
*

	conv_lossf)?.�8�        )��P	������A�
*

	conv_loss�$?vG�        )��P	~�����A�
*

	conv_loss�W?	+��        )��P	�����A�
*

	conv_loss��?]7        )��P	�$����A�
*

	conv_loss"?���        )��P	�F����A�
*

	conv_loss��?���        )��P	li����A�
*

	conv_loss��?3���        )��P	�����A�
*

	conv_loss��?B��        )��P	�����A�
*

	conv_loss��?eNB*        )��P	������A�
*

	conv_loss?�|R�        )��P	�����A�
*

	conv_loss�?(O�        )��P	�#����A�
*

	conv_loss!?A.g        )��P	�F����A�
*

	conv_loss/?m	�        )��P	h����A�
*

	conv_loss8�?��F        )��P	�����A�
*

	conv_loss�%?
v5g        )��P	�����A�
*

	conv_lossl�?S�        )��P	������A�
*

	conv_loss��?^�ϵ        )��P	�����A�
*

	conv_loss���>;�\        )��P	�)����A�
*

	conv_loss?K�}        )��P	fL����A�
*

	conv_loss??E��A        )��P	�n����A�
*

	conv_loss��?3�p�        )��P	X�����A�
*

	conv_loss3� ?wy�        )��P	������A�
*

	conv_loss�� ?�0�        )��P	������A�
*

	conv_loss�&�>���A        )��P	�����A�
*

	conv_loss���>����        )��P	0����A�
*

	conv_loss\q?�Zyj        )��P	�R����A�
*

	conv_loss��>�y83        )��P	�t����A�
*

	conv_loss���>*�8i        )��P	������A�
*

	conv_losst��>L �h        )��P	ѽ����A�
*

	conv_lossa��>%�{�        )��P	������A�
*

	conv_loss���>e��	        )��P	�����A�
*

	conv_loss���>���        )��P	!?����A�
*

	conv_loss�m ?�F��        )��P	�d����A�
*

	conv_loss���>���        )��P	M�����A�
*

	conv_loss	p�>��t�        )��P	~�����A�
*

	conv_losso;�>��_x        )��P	������A�
*

	conv_loss!��>D{��        )��P	������A�
*

	conv_loss��>����        )��P	�����A�
*

	conv_loss�s�>@"]�        )��P	9����A�
*

	conv_loss]B�>W���        )��P	7]����A�
*

	conv_loss���>�*��        )��P	������A�
*

	conv_loss" �>�YE        )��P	�����A�
*

	conv_loss� ?K��'        )��P	�����A�
*

	conv_loss���>`)k        )��P	������A�
*

	conv_loss<��>>h�        )��P	�����A�
*

	conv_loss��>Ҍ��        )��P	%5����A�
*

	conv_loss�N�>g�`J        )��P	�[����A�
*

	conv_loss��>[0�        )��P	b}����A�
*

	conv_loss��>��        )��P	�����A�
*

	conv_loss�G�>�@�^        )��P	������A�
*

	conv_lossi��>z��5        )��P	������A�
*

	conv_lossP�>NKI�        )��P	�	����A�
*

	conv_loss��>b���        )��P	�+����A�
*

	conv_lossu��>셋y        )��P	N����A�
*

	conv_loss�:�>de>6        )��P	�p����A�
*

	conv_loss�o�>@�%        )��P	������A�
*

	conv_lossP�>���P        )��P	˵����A�
*

	conv_loss���>�T]        )��P	 �����A�
*

	conv_loss���>���        )��P	������A�
*

	conv_loss���>w��        )��P	 ����A�
*

	conv_loss3��>���        )��P	:C����A�
*

	conv_loss���>?V�        )��P	�i����A�
*

	conv_loss2��>fq"D        )��P	������A�
*

	conv_losse��>���]        )��P	�����A�
*

	conv_loss���>�z        )��P	������A�
*

	conv_loss��>>s)        )��P	������A�
*

	conv_loss�l�>��{        )��P	� ����A�
*

	conv_loss�@�>�J
?        )��P	%J����A�
*

	conv_lossg��>L���        )��P	�p����A�
*

	conv_loss���>'m��        )��P	Z�����A�
*

	conv_lossގ�>'A��        )��P	�����A�
*

	conv_loss�
�>|qN.        )��P	T�����A�
*

	conv_lossX��>�z!        )��P	�����A�
*

	conv_losse��>��3�        )��P	�6����A�*

	conv_loss/�>o���        )��P	�[����A�*

	conv_loss�3�>y2T�        )��P	�����A�*

	conv_loss$�>͹,�        )��P	Ԩ����A�*

	conv_loss��>7��R        )��P	������A�*

	conv_loss�=�>hlu1        )��P	������A�*

	conv_loss�G�>�*�        )��P	����A�*

	conv_loss$��>$��4        )��P	�K����A�*

	conv_loss���>�J�        )��P	lo����A�*

	conv_loss!��>GE�        )��P	đ����A�*

	conv_loss���>�E/<        )��P	������A�*

	conv_lossw?�>�(��        )��P	������A�*

	conv_loss���>`�t�        )��P	i�����A�*

	conv_loss�:�>�Xu<        )��P	� ����A�*

	conv_loss��>C.<�        )��P	�K����A�*

	conv_loss��>�9Y        )��P	pn����A�*

	conv_loss:_�>L�7�        )��P	y�����A�*

	conv_lossE8�>�W{7        )��P	������A�*

	conv_loss9c�>�uoP        )��P	������A�*

	conv_loss6�>?��        )��P	������A�*

	conv_loss���>�J>�        )��P	W����A�*

	conv_loss3�>9�B�        )��P	�G����A�*

	conv_loss���>k16�        )��P	Hn����A�*

	conv_lossm��>O� K        )��P	������A�*

	conv_loss���>�U�Z        )��P	�����A�*

	conv_loss���>�'�        )��P	"�����A�*

	conv_lossӖ�>��`�        )��P	�����A�*

	conv_loss\.�>\�?I        )��P	�/����A�*

	conv_lossm��>H��q        )��P	GU����A�*

	conv_lossx$�>_�        )��P	Cy����A�*

	conv_loss���>A�k�        )��P	������A�*

	conv_loss���>�|�}        )��P	������A�*

	conv_lossom�>GN��        )��P	�����A�*

	conv_lossB��>�Q'�        )��P	� ����A�*

	conv_loss� �>�IS        )��P	�& ����A�*

	conv_loss���>�j��        )��P	RH ����A�*

	conv_loss��>K.�T        )��P	1l ����A�*

	conv_lossx�>�        )��P	� ����A�*

	conv_lossَ�>\��2        )��P	q� ����A�*

	conv_loss/��>'&e$        )��P	�� ����A�*

	conv_loss�?�>���        )��P	�� ����A�*

	conv_loss�;�>A���        )��P	1!����A�*

	conv_loss?�>�C        )��P	�@!����A�*

	conv_loss�(�>aj�        )��P	�q!����A�*

	conv_loss��>���        )��P	ǔ!����A�*

	conv_loss���>!�Z�        )��P	5�!����A�*

	conv_loss���>|���        )��P	�!����A�*

	conv_lossu7�>����        )��P	D�!����A�*

	conv_loss���>Z�
0        )��P	�""����A�*

	conv_loss$�>��7�        )��P	_G"����A�*

	conv_loss}Z�>��2�        )��P	)p"����A�*

	conv_loss�e�>�n{.        )��P	��"����A�*

	conv_lossr��>���        )��P	��"����A�*

	conv_loss�(�>P�i�        )��P	�"����A�*

	conv_losssq�>\3�        )��P	��"����A�*

	conv_lossT�>����        )��P	M #����A�*

	conv_loss���>��M�        )��P	:G#����A�*

	conv_lossv�>e�X        )��P	s#����A�*

	conv_loss��>U���        )��P	C�#����A�*

	conv_loss�0�>UU��        )��P	��#����A�*

	conv_loss���>���        )��P	��#����A�*

	conv_loss�l�>|^��        )��P	�$����A�*

	conv_loss�>$�[        )��P	|($����A�*

	conv_loss5�>��~�        )��P	�J$����A�*

	conv_lossg~�>ݙ?0        )��P	8z$����A�*

	conv_loss5��>����        )��P	��$����A�*

	conv_loss�^�>o���        )��P	��$����A�*

	conv_lossږ�>Oiܧ        )��P	��$����A�*

	conv_loss!H�>P��        )��P	"%����A�*

	conv_loss�W�>���        )��P	�'%����A�*

	conv_loss�N�>����        )��P	�J%����A�*

	conv_loss��>zX�        )��P	 n%����A�*

	conv_loss���>}D�        )��P	^�%����A�*

	conv_lossw��>�d�        )��P	��%����A�*

	conv_lossd�>��o�        )��P	a�%����A�*

	conv_loss���>���        )��P	�&����A�*

	conv_loss��>>q�4        )��P	�2&����A�*

	conv_lossM��>lp|o        )��P	aU&����A�*

	conv_lossVZ�>Ugz        )��P	�{&����A�*

	conv_loss��>JL�        )��P	�&����A�*

	conv_lossG6�>/{�        )��P	I�&����A�*

	conv_loss�n�>��4�        )��P	A�&����A�*

	conv_loss=�>a�$�        )��P	�'����A�*

	conv_loss�.�>G��q        )��P	S('����A�*

	conv_loss(�>�4��        )��P	�K'����A�*

	conv_lossx��>	��        )��P	�m'����A�*

	conv_loss��>�9��        )��P	v�'����A�*

	conv_loss��>�7�        )��P	��'����A�*

	conv_lossrZ�>G�,        )��P	��'����A�*

	conv_loss�2�>�N�        )��P	0�'����A�*

	conv_loss�E�>;��        )��P	(����A�*

	conv_loss9�>)02        )��P	@(����A�*

	conv_loss���>�E�Z        )��P	��,����A�*

	conv_loss�$�>����        )��P	�
-����A�*

	conv_lossx��>��        )��P	o+-����A�*

	conv_loss[(�>.��T        )��P	�K-����A�*

	conv_lossA|�>d�y�        )��P	�m-����A�*

	conv_loss<�>#f\%        )��P	ɍ-����A�*

	conv_lossh�>�X�        )��P	ŭ-����A�*

	conv_loss���>�z��        )��P		�-����A�*

	conv_loss�
�>D\�3        )��P	��-����A�*

	conv_loss�T�>�        )��P	�.����A�*

	conv_loss���>Ahw        )��P	�/.����A�*

	conv_loss2�>{�`�        )��P	N.����A�*

	conv_loss�r�>ӄ�z        )��P	�o.����A�*

	conv_loss���>���        )��P	��.����A�*

	conv_loss���>x`        )��P	ع.����A�*

	conv_loss�,�>=���        )��P	s�.����A�*

	conv_loss��>����        )��P	��.����A�*

	conv_lossa��>_�9d        )��P	�/����A�*

	conv_loss@��>�U�2        )��P	�>/����A�*

	conv_lossVL�>5�        )��P	�]/����A�*

	conv_loss���>�F��        )��P	�}/����A�*

	conv_loss���>Ĩ��        )��P	̜/����A�*

	conv_loss_��>F%"�        )��P	��/����A�*

	conv_loss�	�>����        )��P	z�/����A�*

	conv_loss4��>#�
V        )��P	�0����A�*

	conv_loss= �>��E        )��P	�20����A�*

	conv_losss��>����        )��P	�U0����A�*

	conv_lossǠ�>ב��        )��P	%y0����A�*

	conv_loss�r�>��`�        )��P		�0����A�*

	conv_loss@��>x��6        )��P	q�0����A�*

	conv_loss��>kL�$        )��P	��0����A�*

	conv_loss;��>n��        )��P	��0����A�*

	conv_loss�8�>T���        )��P	�1����A�*

	conv_loss��>@�I�        )��P	o81����A�*

	conv_lossӷ�>}%�        )��P	�X1����A�*

	conv_loss���>�+$X        )��P	"x1����A�*

	conv_loss~7�>�1�        )��P	!�1����A�*

	conv_loss���>���        )��P	[�1����A�*

	conv_loss*��>�y        )��P	��1����A�*

	conv_loss�^�>%L+Y        )��P	��1����A�*

	conv_lossK�>p��        )��P	�2����A�*

	conv_loss?<�>��M�        )��P	�82����A�*

	conv_lossF��>��f        )��P	aV2����A�*

	conv_lossH��>��O        )��P	�t2����A�*

	conv_loss'g�>�O        )��P	`�2����A�*

	conv_loss��>�x�^        )��P	m�2����A�*

	conv_loss�>5h        )��P	�2����A�*

	conv_loss���>�p�h        )��P	��2����A�*

	conv_losso�>��{�        )��P	�3����A�*

	conv_loss�ž>�
        )��P	�33����A�*

	conv_lossޘ�>e���        )��P	�S3����A�*

	conv_lossx��>�V��        )��P	d�3����A�*

	conv_loss��>	E|        )��P	�3����A�*

	conv_losse$�>5���        )��P	��3����A�*

	conv_loss4,�>�W�C        )��P	��3����A�*

	conv_lossjV�>'��        )��P	�4����A�*

	conv_loss]��>�q        )��P	�'4����A�*

	conv_lossb�>��Z        )��P	AI4����A�*

	conv_loss�z�>3hƫ        )��P	
j4����A�*

	conv_loss���>�1��        )��P	8�4����A�*

	conv_loss���>�h�        )��P	��4����A�*

	conv_lossh��>�8��        )��P	��4����A�*

	conv_loss�ͺ>* %        )��P	��4����A�*

	conv_loss��>�ו        )��P	5(5����A�*

	conv_lossN�>���        )��P	�J5����A�*

	conv_loss���>Œ��        )��P	�k5����A�*

	conv_loss���>||~0        )��P	;�5����A�*

	conv_loss\��>�:�        )��P	��5����A�*

	conv_lossP��>2{b        )��P	��5����A�*

	conv_loss6ݼ>�҅�        )��P	(�5����A�*

	conv_lossӶ> 
��        )��P	46����A�*

	conv_loss�l�>�fc�        )��P	�86����A�*

	conv_lossi��>��b        )��P	EY6����A�*

	conv_lossr�>�Ò        )��P	�{6����A�*

	conv_loss��>�\5        )��P	=�6����A�*

	conv_loss�F�>h8�        )��P	�6����A�*

	conv_loss
#�>|T        )��P	��6����A�*

	conv_loss�ۺ>V���        )��P	f7����A�*

	conv_loss�m�>"�        )��P	�$7����A�*

	conv_loss�ݶ>/c�        )��P	_G7����A�*

	conv_lossV��>�        )��P	�i7����A�*

	conv_loss�޴>���        )��P	|�7����A�*

	conv_loss��>���        )��P	��7����A�*

	conv_losss��>F���        )��P	��7����A�*

	conv_loss\�>�F"�        )��P	U�7����A�*

	conv_loss^�>�
J        )��P	�8����A�*

	conv_lossݥ�>���        )��P	28����A�*

	conv_lossG��>��f�        )��P	P8����A�*

	conv_loss"��>ݰw        )��P	8q8����A�*

	conv_loss�N�>���O        )��P	Ő8����A�*

	conv_lossz��>�he        )��P	��8����A�*

	conv_losseR�>>V�        )��P	]�8����A�*

	conv_lossy��>�%y�        )��P	��8����A�*

	conv_loss���>'��        )��P	c9����A�*

	conv_losss��>^�fY        )��P	�59����A�*

	conv_loss^ݷ>�Z�~        )��P	�X9����A�*

	conv_loss�x�>#��        )��P	�w9����A�*

	conv_loss�h�>&��        )��P	З9����A�*

	conv_lossX��>p�13        )��P	��9����A�*

	conv_loss܆�>_X`        )��P	��9����A�*

	conv_loss�a�>B��;        )��P	��9����A�*

	conv_loss?�>S�[        )��P	�	;����A�*

	conv_lossm2�>z/��        )��P	�(;����A�*

	conv_loss�6�>�Wu�        )��P	�L;����A�*

	conv_loss��>8�1        )��P	�n;����A�*

	conv_loss~}�>��7        )��P	��;����A�*

	conv_lossܥ�>'_*        )��P	�;����A�*

	conv_loss�e�> O��        )��P	{�;����A�*

	conv_loss�)�>��.        )��P	��;����A�*

	conv_lossmO�>}e�        )��P	�<����A�*

	conv_loss�>��J        )��P	O?<����A�*

	conv_losss߮>2��<        )��P	~_<����A�*

	conv_loss�>����        )��P	��<����A�*

	conv_lossiӯ>�|vw        )��P	 �<����A�*

	conv_lossvg�>��U(        )��P	��<����A�*

	conv_loss�w�>'�`�        )��P	
=����A�*

	conv_loss��>3�G�        )��P	 *=����A�*

	conv_loss�w�>*�`        )��P	�I=����A�*

	conv_lossҲ>�V��        )��P	<j=����A�*

	conv_loss� �>x�=�        )��P	?�=����A�*

	conv_loss�@�>�Lɾ        )��P	��=����A�*

	conv_loss�V�>$�w        )��P	��=����A�*

	conv_loss�>�C�;        )��P	��=����A�*

	conv_loss �>�)&�        )��P		>����A�*

	conv_loss��>%��G        )��P	()>����A�*

	conv_lossD�>㔤z        )��P	�H>����A�*

	conv_loss=�>�ܟ�        )��P	�i>����A�*

	conv_loss���>����        )��P	q�>����A�*

	conv_lossf5�>�#b�        )��P	X�>����A�*

	conv_loss���>���        )��P	W�>����A�*

	conv_loss���>�J��        )��P	�>����A�*

	conv_lossk��>���K        )��P	z?����A�*

	conv_loss�L�>㞚�        )��P	�@?����A�*

	conv_lossd��>�F�r        )��P	zd?����A�*

	conv_lossX�>;�I�        )��P	��?����A�*

	conv_lossRZ�>X�        )��P	��?����A�*

	conv_lossd�>m�`$        )��P	B�?����A�*

	conv_loss�6�>�N�V        )��P	c�?����A�*

	conv_loss�i�>�s�&        )��P	 @����A�*

	conv_loss{{�>D��        )��P	�1@����A�*

	conv_loss���>k���        )��P	dV@����A�*

	conv_loss�/�>�8�        )��P	y@����A�*

	conv_loss���>HZ՘        )��P	֘@����A�*

	conv_loss��>>��>        )��P	H�@����A�*

	conv_loss�¯>8��D        )��P	+�@����A�*

	conv_loss���>L�:        )��P	�@����A�*

	conv_loss� �>`��        )��P	OA����A�*

	conv_loss3ɳ>e
�g        )��P	]<A����A�*

	conv_loss�>���        )��P	�[A����A�*

	conv_lossu/�>< u�        )��P	,|A����A�*

	conv_loss�ö>����        )��P	y�A����A�*

	conv_loss(�>�;gh        )��P	��A����A�*

	conv_loss�E�>z�D�        )��P	v�A����A�*

	conv_losso��>y�,�        )��P	�B����A�*

	conv_loss_ǯ>�/�1        )��P	�7B����A�*

	conv_loss�k�>����        )��P	[B����A�*

	conv_loss��>{.��        )��P	�zB����A�*

	conv_loss��>B�        )��P	�B����A�*

	conv_loss_O�>wm�        )��P	��B����A�*

	conv_loss@+�>cn�&        )��P	��B����A�*

	conv_lossG}�>��        )��P	��B����A�*

	conv_lossz��>-��        )��P	rC����A�*

	conv_loss��>
m        )��P	\EC����A�*

	conv_loss�P�>�6��        )��P	�eC����A�*

	conv_loss�߯>�gA�        )��P	�C����A�*

	conv_lossޮ>{ C�        )��P	��C����A�*

	conv_loss'�>p��        )��P	��C����A�*

	conv_lossS�>m g        )��P	��C����A�*

	conv_loss�q�>��}�        )��P	�D����A�*

	conv_lossoU�>��c�        )��P	U@D����A�*

	conv_loss㲱>�$)�        )��P	XgD����A�*

	conv_loss���>�/h�        )��P	+�D����A�*

	conv_loss��>�d;?        )��P	�D����A�*

	conv_loss���>O�Y        )��P	p�D����A�*

	conv_loss�s�>t��]        )��P	L�D����A�*

	conv_lossK��>'�        )��P	�E����A�*

	conv_loss���>X2�        )��P	�2E����A�*

	conv_loss's�>��QI        )��P	�QE����A�*

	conv_loss�L�>�ה�        )��P	lqE����A�*

	conv_loss35�>��l�        )��P	��E����A�*

	conv_loss� �>�N        )��P	#�E����A�*

	conv_loss[��>kX        )��P	��E����A�*

	conv_loss���>��s        )��P	��E����A�*

	conv_lossHW�>3�`~        )��P	F����A�*

	conv_loss�5�>O��        )��P	5F����A�*

	conv_loss���>�nI        )��P	�TF����A�*

	conv_loss\I�>w=	�        )��P	�tF����A�*

	conv_lossn[�>E�f        )��P	�F����A�*

	conv_lossS�>�#q�        )��P	ȴF����A�*

	conv_loss�c�>��N(        )��P	��F����A�*

	conv_lossIc�>���        )��P	��F����A�*

	conv_loss�i�>��        )��P	�G����A�*

	conv_loss��>��+?        )��P	�4G����A�*

	conv_loss�m�>���        )��P	UG����A�*

	conv_loss2�>��        )��P	�tG����A�*

	conv_lossϐ�>JJ��        )��P	��G����A�*

	conv_loss8��>M���        )��P	��G����A�*

	conv_loss<�>n�ru        )��P	U�G����A�*

	conv_loss�C�>����        )��P	��G����A�*

	conv_loss�ٮ>-]p        )��P	�H����A�*

	conv_loss�>\��R        )��P	~6H����A�*

	conv_loss�,�>:?"c        )��P	�dH����A�*

	conv_loss�̩>ȿ�(        )��P	q�H����A�*

	conv_loss�>�ce        )��P	��H����A�*

	conv_loss>%�>�Y�)        )��P	��H����A�*

	conv_lossd��>i	�        )��P	�H����A�*

	conv_losst�>���        )��P	�	I����A�*

	conv_lossX�>r�@D        )��P	-I����A�*

	conv_loss�>���        )��P	�LI����A�*

	conv_loss׍�>��'        )��P	�xI����A�*

	conv_lossά>��        )��P	��I����A�*

	conv_loss��>tg|         )��P	j�I����A�*

	conv_loss>۫>��Tu        )��P	�I����A�*

	conv_loss(%�>��qu        )��P	)�I����A�*

	conv_lossQd�>.\�x        )��P	�!J����A�*

	conv_loss�~�>lp9�        )��P	�AJ����A�*

	conv_loss���>�Ki        )��P	bJ����A�*

	conv_loss�;�>NvP        )��P	��J����A�*

	conv_loss�?�>���\        )��P	R�J����A�*

	conv_loss%��>��R        )��P	��J����A�*

	conv_loss�-�>�){        )��P	��J����A�*

	conv_loss��>!�Q�        )��P	TK����A�*

	conv_loss��>AI��        )��P	�'K����A�*

	conv_loss��>0��        )��P	]HK����A�*

	conv_loss]�>X��        )��P	jK����A�*

	conv_loss&��>�P/�        )��P	ȊK����A�*

	conv_lossPf�>���        )��P	��K����A�*

	conv_lossWa�>N�4�        )��P	�K����A�*

	conv_loss�l�> x�        )��P	��K����A�*

	conv_loss�*�>��        )��P	�L����A�*

	conv_loss��>�O��        )��P	8L����A�*

	conv_loss	�>ަ�O        )��P	XL����A�*

	conv_loss9C�>=��        )��P	lwL����A�*

	conv_lossub�>=��        )��P	B�L����A�*

	conv_loss!q�>�aL        )��P	*�L����A�*

	conv_loss�>��sj        )��P	��L����A�*

	conv_lossw3�>���        )��P	M����A�*

	conv_loss�?�>����        )��P	�.M����A�*

	conv_loss�$�>�^ډ        )��P	TPM����A�*

	conv_loss�¬>�Q��        )��P	roM����A�*

	conv_lossNg�>l�Q        )��P	ԑM����A�*

	conv_loss��>��<6        )��P	n�M����A�*

	conv_loss�?�>�s�        )��P	��M����A�*

	conv_lossiG�>�`��        )��P	~�M����A�*

	conv_lossY��>W���        )��P	BN����A�*

	conv_loss6|�>�a/�        )��P	>1N����A�*

	conv_lossd�>�U�"        )��P	"PN����A�*

	conv_lossC�>�x*j        )��P	sN����A�*

	conv_loss�U�>�9�        )��P	z�N����A�*

	conv_loss�7�>�2V        )��P	дN����A�*

	conv_loss�Y�>Ğ��        )��P	�N����A�*

	conv_loss��>��r        )��P	�O����A�*

	conv_loss���>={X        )��P	�#O����A�*

	conv_lossf��>$_N        )��P	�DO����A�*

	conv_loss��>3.��        )��P	�eO����A�*

	conv_loss�s�>JL��        )��P	Y�O����A�*

	conv_loss��>��        )��P	�O����A�*

	conv_loss���>Y�l�        )��P	��O����A�*

	conv_loss+��>��X�        )��P	_�O����A�*

	conv_lossaΨ>exq3        )��P	�P����A�*

	conv_loss��>�1�        )��P	�/P����A�*

	conv_lossZ��>�-��        )��P	�OP����A�*

	conv_loss�d�>RQ��        )��P	�rP����A�*

	conv_loss��>ϒ�        )��P	��P����A�*

	conv_loss���>_5|�        )��P	��P����A�*

	conv_loss8��>Trv        )��P	�P����A�*

	conv_loss7�>�`g        )��P	�Q����A�*

	conv_lossfO�>Ί-�        )��P	�>Q����A�*

	conv_loss�ǩ>�a        )��P	M^Q����A�*

	conv_loss���>w�        )��P	lQ����A�*

	conv_lossH\�>sbSq        )��P	M�Q����A�*

	conv_loss�Z�>I4�0        )��P	t�Q����A�*

	conv_lossᄭ>����        )��P	>�Q����A�*

	conv_loss:�>���y        )��P	R����A�*

	conv_loss<��>{��V        )��P	u1R����A�*

	conv_loss�J�>�3��        )��P	}PR����A�*

	conv_lossg�>��G        )��P	�qR����A�*

	conv_loss���>_�K        )��P	B�R����A�*

	conv_loss���>\��)        )��P	t�R����A�*

	conv_losso�>n�9        )��P	��R����A�*

	conv_lossU2�>1��4        )��P	�R����A�*

	conv_loss���>Ai˗        )��P	�S����A�*

	conv_loss�~�>2+sx        )��P	6S����A�*

	conv_lossJ7�>��D        )��P	�\S����A�*

	conv_loss���> 
        )��P	.�S����A�*

	conv_loss���>�T,        )��P	l�S����A�*

	conv_loss��>�P2�        )��P	�S����A�*

	conv_loss��>�ܟ�        )��P	��S����A�*

	conv_loss��>�sЁ        )��P	\	T����A�*

	conv_lossl�>N@.        )��P	�)T����A�*

	conv_lossp:�>R�n�        )��P	LT����A�*

	conv_lossx�>Q4�^        )��P	kT����A�*

	conv_loss�Z�>s�_        )��P	��T����A�*

	conv_loss풦>]{cJ        )��P	1�T����A�*

	conv_loss �>,��/        )��P	��T����A�*

	conv_loss:�>��sp        )��P	��T����A�*

	conv_loss	��>�s��        )��P	�U����A�*

	conv_loss�#�>�@X_        )��P	�6U����A�*

	conv_lossJ�>8�B�        )��P	v[U����A�*

	conv_losst�>���        )��P	q{U����A�*

	conv_loss�d�>D�FZ        )��P	`�U����A�*

	conv_loss'��>h#G�        )��P	5�V����A�*

	conv_loss{�>/���        )��P	��V����A�*

	conv_loss�L�>Oe�        )��P	��V����A�*

	conv_losse�>��ͪ        )��P	�W����A�*

	conv_loss6�>c���        )��P	f6W����A�*

	conv_loss�>�.�        )��P	�[W����A�*

	conv_loss���>s铚        )��P	�}W����A�*

	conv_loss���>	s<        )��P	w�W����A�*

	conv_loss�1�>���g        )��P	0�W����A�*

	conv_lossAw�>K�%�        )��P	��W����A�*

	conv_loss�}�>į�        )��P	>X����A�*

	conv_loss��>���        )��P	%'X����A�*

	conv_loss�̨>@�d        )��P	�MX����A�*

	conv_loss�ĩ>w��+        )��P	�sX����A�*

	conv_lossé>�V!V        )��P	}�X����A�*

	conv_loss���>.|�f        )��P	[�X����A�*

	conv_loss�%�>$f+X        )��P	?�X����A�*

	conv_lossRh�>kڙ�        )��P	OY����A�*

	conv_loss���>�Cґ        )��P	1Y����A�*

	conv_loss�:�>絘c        )��P	�QY����A�*

	conv_loss�x�>;�        )��P	htY����A�*

	conv_losss?�>���T        )��P	 �Y����A�*

	conv_loss��>wep        )��P	��Y����A�*

	conv_loss�M�>Xw�+        )��P	��Y����A�*

	conv_loss6*�>��C         )��P	Q�Y����A�*

	conv_loss�D�>���        )��P	6Z����A�*

	conv_loss;Ȧ>,9?�        )��P	<Z����A�*

	conv_loss�c�>,�Wc        )��P	X]Z����A�*

	conv_loss�z�>����        )��P	S}Z����A�*

	conv_loss��>�36�        )��P	ߝZ����A�*

	conv_loss��>1%L        )��P	��Z����A�*

	conv_loss��>��6�        )��P	�Z����A�*

	conv_loss�=�>�fI�        )��P	6[����A�*

	conv_lossO�>s��        )��P	�*[����A�*

	conv_lossӅ�>R�`9        )��P	�N[����A�*

	conv_loss���>�7w        )��P	q[����A�*

	conv_loss���>N�        )��P	P�[����A�*

	conv_loss%�>�N��        )��P	��[����A�*

	conv_loss�>�>h��        )��P	��[����A�*

	conv_loss8	�>�q`Y        )��P	2�[����A�*

	conv_loss�ަ>{m��        )��P	7 \����A�*

	conv_loss H�>�ufg        )��P	�E\����A�*

	conv_lossڪ�>�D�        )��P	�f\����A�*

	conv_loss3�>��W�        )��P	(�\����A�*

	conv_loss�q�>�_9�        )��P	�\����A�*

	conv_loss�$�>�� �        )��P	��\����A�*

	conv_loss�Ш>��\        )��P	��\����A�*

	conv_loss���>�;>        )��P	�]����A�*

	conv_loss�=�>��        )��P	�5]����A�*

	conv_lossͧ>�0@        )��P	uU]����A�*

	conv_loss2�>6�`�        )��P	F�]����A�*

	conv_loss�ɦ>9        )��P	@�]����A�*

	conv_loss�ä>�8t-        )��P	��]����A�*

	conv_loss�5�>Om�        )��P	��]����A�*

	conv_loss�y�>>�        )��P	^����A�*

	conv_loss-��>^��        )��P	l:^����A�*

	conv_loss��>�� �        )��P	�Y^����A�*

	conv_loss��>�ps        )��P	�x^����A�*

	conv_lossIҦ>n��E        )��P	a�^����A�*

	conv_loss)0�>^�#        )��P	9�^����A�*

	conv_lossho�>��#        )��P	F�^����A�*

	conv_losseX�>ш        )��P	��^����A�*

	conv_loss�Ũ>
�^6        )��P	b'_����A�*

	conv_lossp?�>���        )��P	�F_����A�*

	conv_lossEh�>#j%�        )��P	�i_����A�*

	conv_loss��>���<        )��P	��_����A�*

	conv_loss���>Hp�        )��P	��_����A�*

	conv_loss�Q�>��t        )��P	x�_����A�*

	conv_loss�)�>���        )��P	��_����A�*

	conv_loss�
�>�o{T        )��P	B`����A�*

	conv_loss���>��g�        )��P	�;`����A�*

	conv_lossX��>�r�        )��P	�``����A�*

	conv_loss�F�>�2?        )��P	�`����A�*

	conv_loss�֭>�C>L        )��P	��`����A�*

	conv_losso�>z�؁        )��P	��`����A�*

	conv_lossU2�>eI��        )��P	��`����A�*

	conv_loss!��>{:��        )��P	�a����A�*

	conv_lossFn�>���j        )��P	�'a����A�*

	conv_lossT�>@FH        )��P	]Ia����A�*

	conv_lossN�>��"        )��P	�ha����A�*

	conv_loss�(�>��^        )��P	4�a����A�*

	conv_losseO�>RY��        )��P	r�a����A�*

	conv_loss�ĩ>�~D�        )��P	��a����A�*

	conv_loss,!�>w�K        )��P	��a����A�*

	conv_loss֍�>O_�        )��P	�
b����A�*

	conv_loss���>s�c        )��P	Y-b����A�*

	conv_loss�F�>��d�        )��P	�Lb����A�*

	conv_loss�V�>:m        )��P	$lb����A�*

	conv_loss��>��׳        )��P	ȍb����A�*

	conv_loss��>�a��        )��P	��b����A�*

	conv_loss,ި>=l�        )��P	�b����A�*

	conv_loss���>^�e�        )��P	��b����A�*

	conv_loss�ݥ>"��        )��P	�c����A�*

	conv_loss�[�>��l        )��P	62c����A�*

	conv_loss�L�>#�        )��P	�Qc����A�*

	conv_losse��>�e�        )��P	&rc����A�*

	conv_lossƂ�>���        )��P	#�c����A�*

	conv_loss�Ң>2Y]z        )��P	߱c����A�*

	conv_loss�x�>��        )��P	_�c����A�*

	conv_lossR*�>��'        )��P	��c����A�*

	conv_loss�o�>�7        )��P	�d����A�*

	conv_lossq/�>r��        )��P	_Ad����A�*

	conv_lossD5�>����        )��P	�ad����A�*

	conv_loss]J�> '        )��P	˂d����A�*

	conv_loss��>	�s�        )��P	�d����A�*

	conv_loss�Φ>9V�        )��P	��d����A�*

	conv_loss�>�Y        )��P	��d����A�*

	conv_loss�u�>��bN        )��P	be����A�*

	conv_lossW��>q�=�        )��P	j5e����A�*

	conv_loss"ƨ>ǃ��        )��P	4^e����A�*

	conv_loss�0�>C��        )��P	�e����A�*

	conv_loss{�>�6�t        )��P	x�e����A�*

	conv_loss4��>CI{f        )��P	`�e����A�*

	conv_losso¥>�i:>        )��P	��e����A�*

	conv_loss�Z�>�&h�        )��P	�f����A�*

	conv_loss���>�	�        )��P	�/f����A�*

	conv_loss/c�>�{�        )��P	�Yf����A�*

	conv_loss�^�>7��        )��P	�xf����A�*

	conv_loss h�> ��s        )��P	'�f����A�*

	conv_loss���>���        )��P	̹f����A�*

	conv_loss_�>Kh��        )��P	��f����A�*

	conv_lossû�>}Ƃn        )��P	��f����A�*

	conv_loss��>�Z�+        )��P	�g����A�*

	conv_lossr�>:��H        )��P	�<g����A�*

	conv_loss�_�>��1�        )��P	o[g����A�*

	conv_loss�z�>���I        )��P	#g����A�*

	conv_loss�ڦ>[�"�        )��P	��g����A�*

	conv_lossV�>�*�K        )��P	$�g����A�*

	conv_loss��>���K        )��P	o�g����A�*

	conv_loss�>��#�        )��P	�h����A�*

	conv_loss�0�>o0        )��P	�,h����A�*

	conv_losscm�>!�        )��P	$Mh����A�*

	conv_loss^x�>��]        )��P	�kh����A�*

	conv_loss�B�>��E        )��P	j�h����A�*

	conv_loss.r�>\&I'        )��P	>�h����A�*

	conv_lossޭ�>?��&        )��P	��h����A�*

	conv_lossA��>��dU        )��P	��h����A�*

	conv_loss��>���        )��P	�i����A�*

	conv_loss&�>�*        )��P	:3i����A�*

	conv_loss���>g�+        )��P	�Si����A�*

	conv_loss�C�>��R�        )��P	|si����A�*

	conv_loss��>Nǃ        )��P	�i����A�*

	conv_lossH��>2F~�        )��P	a�i����A�*

	conv_loss��>-�'�        )��P	��i����A�*

	conv_loss��>X��        )��P	��i����A�*

	conv_loss�>����        )��P	�j����A�*

	conv_loss=&�>��O        )��P	�6j����A�*

	conv_loss��>�>̬        )��P	<Zj����A�*

	conv_lossʤ>���        )��P	�yj����A�*

	conv_loss�W�>�/b�        )��P	O�j����A�*

	conv_loss�[�>���        )��P	��j����A�*

	conv_lossxB�>or�        )��P	0�j����A�*

	conv_lossx�>�Py        )��P	wk����A�*

	conv_loss� �>�J        )��P	!'k����A�*

	conv_loss�D�>�7�        )��P	�Ek����A�*

	conv_lossY�>�d�k        )��P	�jk����A�*

	conv_loss�̣>��P�        )��P	�k����A�*

	conv_loss�Ť>��c        )��P	�k����A�*

	conv_loss���>T�T�        )��P	
�k����A�*

	conv_loss�6�>V?t�        )��P	��k����A�*

	conv_loss�ä>�,#Z        )��P	�l����A�*

	conv_loss���>��T        )��P	3l����A�*

	conv_loss��>�JT�        )��P	�Ul����A�*

	conv_lossPУ>8"�(        )��P	 �l����A�*

	conv_lossޅ�>;_�        )��P	�l����A�*

	conv_loss��>�7x        )��P	[�l����A�*

	conv_loss��>}G�        )��P	��l����A�*

	conv_loss�J�>�uo        )��P	`m����A�*

	conv_lossFϤ>�o��        )��P	�9m����A�*

	conv_loss�Ǡ>H
A�        )��P	�Ym����A�*

	conv_loss�̣>�P��        )��P	(�m����A�*

	conv_loss���>1��        )��P	��m����A�*

	conv_loss���> a>�        )��P	�m����A�*

	conv_loss&أ>�|��        )��P	��m����A�*

	conv_lossE��>�S�        )��P	�n����A�*

	conv_loss^�>����        )��P	 #n����A�*

	conv_lossk(�>10�2        )��P	�Bn����A�*

	conv_loss��>� �        )��P	�bn����A�*

	conv_loss���>��        )��P	0�n����A�*

	conv_loss���>���        )��P	v�n����A�*

	conv_loss_2�>�t4�        )��P	'�n����A�*

	conv_loss��>�8bu        )��P	=�n����A�*

	conv_lossJ�>�#��        )��P	�o����A�*

	conv_loss�ߢ>��"#        )��P	1o����A�*

	conv_loss�ݡ>ac        )��P	�To����A�*

	conv_loss�_�>�fG�        )��P	�yo����A�*

	conv_lossä�>����        )��P	�o����A�*

	conv_lossJ��>��        )��P	��o����A�*

	conv_loss�Ф>���=        )��P	�o����A�*

	conv_loss��>�41$        )��P	;�o����A�*

	conv_loss4�>�,�;        )��P	p����A�*

	conv_loss��>th`�        )��P	�>p����A�*

	conv_loss�Y�>7���        )��P	7^p����A�*

	conv_lossWP�>��m�        )��P	 p����A�*

	conv_loss�J�>��`�        )��P	_�p����A�*

	conv_loss��>�;��        )��P	��p����A�*

	conv_loss걡>u7��        )��P	��p����A�*

	conv_lossef�>o��        )��P	Yq����A�*

	conv_lossa��>��2�        )��P	$q����A�*

	conv_loss�v�> (y�        )��P	Gru����A�*

	conv_loss�˥>�I$        )��P	�v����A�*

	conv_loss��>c�i        )��P	:�v����A�*

	conv_lossº�>��n;        )��P	��v����A�*

	conv_lossx�>���        )��P	��v����A�*

	conv_lossƧ�>E���        )��P	
w����A�*

	conv_loss:֤>L��:        )��P	�+w����A�*

	conv_loss�V�>R���        )��P	�Lw����A�*

	conv_loss��>�K��        )��P	mw����A�*

	conv_lossU��>�z��        )��P	"�w����A�*

	conv_loss�>�>��        )��P	�w����A�*

	conv_loss!1�>�h�        )��P	�w����A�*

	conv_losssΧ>-�        )��P	��w����A�*

	conv_loss��>����        )��P	�x����A�*

	conv_loss�
�><�5r        )��P	-2x����A�*

	conv_loss�+�>�c��        )��P	�Wx����A�*

	conv_loss}v�>d�U3        )��P	�{x����A�*

	conv_loss夢>�m{        )��P	ٚx����A�*

	conv_loss#��>����        )��P	Y�x����A�*

	conv_loss#:�>�P��        )��P	�x����A�*

	conv_losse�>9]�        )��P	ly����A�*

	conv_loss�O�>��4�        )��P	�,y����A�*

	conv_lossbգ>�6�        )��P	�Ry����A�*

	conv_loss�V�>M�        )��P	�wy����A�*

	conv_loss+�>Ţƻ        )��P	+�y����A�*

	conv_loss���>�0�        )��P	c�y����A�*

	conv_loss��>Ѫqf        )��P	��y����A�*

	conv_loss��>]凱        )��P	��y����A�*

	conv_loss��>t&5�        )��P	�z����A�*

	conv_losse(�>�F��        )��P	�8z����A�*

	conv_lossa��>��        )��P	|Yz����A�*

	conv_lossr>�>�v{        )��P	�xz����A�*

	conv_lossn��>R�3`        )��P	Ԙz����A�*

	conv_lossu�>q�        )��P	ʹz����A�*

	conv_lossJ�>ϔ{a        )��P	�z����A�*

	conv_loss�o�>��        )��P	�z����A�*

	conv_loss���>��        )��P	�{����A�*

	conv_loss�ء>4�z        )��P	�B{����A�*

	conv_loss��>����        )��P	b{����A�*

	conv_lossuv�>��ҧ        )��P	p�{����A�*

	conv_loss�;�>zU��        )��P	��{����A�*

	conv_loss���>ے�        )��P	��{����A�*

	conv_losse��>*�}�        )��P	��{����A�*

	conv_loss��>�r        )��P	$|����A�*

	conv_loss���>��k�        )��P	6|����A�*

	conv_loss��>ĩ��        )��P	@Y|����A�*

	conv_loss��>c;؜        )��P	�x|����A�*

	conv_loss6�>(}�        )��P	Й|����A�*

	conv_lossz@�>C�ǥ        )��P	~�|����A�*

	conv_loss¢>a�Z        )��P	_�|����A�*

	conv_lossW)�>"�L        )��P	]}����A�*

	conv_lossF�>K�        )��P	c0}����A�*

	conv_loss�;�>K�        )��P	�Q}����A�*

	conv_lossJ�>�M_        )��P	sp}����A�*

	conv_loss�&�>ϯ�        )��P	�}����A�*

	conv_loss�ˡ>��        )��P	6�}����A�*

	conv_loss4$�>�۪        )��P	��}����A�*

	conv_loss2��>�{�        )��P	/�}����A�*

	conv_lossvo�>X���        )��P	�~����A�*

	conv_loss��>,?ԟ        )��P	!5~����A�*

	conv_loss�B�>�-�        )��P	�c~����A�*

	conv_loss�4�>(��        )��P	��~����A�*

	conv_loss��>� ��        )��P	��~����A�*

	conv_loss��>�F��        )��P	9�~����A�*

	conv_lossC̠>KA�         )��P	G�~����A�*

	conv_lossoJ�>��I        )��P	[����A�*

	conv_lossD�>�>        )��P	�5����A�*

	conv_lossew�>��HI        )��P	�V����A�*

	conv_lossu��>�l        )��P	�u����A�*

	conv_lossī�>��$�        )��P	Q�����A�*

	conv_lossD!�>��.        )��P	������A�*

	conv_loss�\�>bt�        )��P	"�����A�*

	conv_loss�O�>�^        )��P		�����A�*

	conv_loss��>�I        )��P	������A�*

	conv_lossH!�>�+w�        )��P	(4�����A�*

	conv_lossW�>:��        )��P	�S�����A�*

	conv_lossA��>��5�        )��P	ou�����A�*

	conv_lossW�>]�        )��P	�������A�*

	conv_loss(�>�5+�        )��P	]������A�*

	conv_loss/�>���]        )��P	}ڀ����A�*

	conv_lossN&�>l�,�        )��P	������A�*

	conv_loss݅�>t��        )��P	�&�����A�*

	conv_loss�؞>�ݥ�        )��P	�H�����A�*

	conv_lossM2�>�R��        )��P	�i�����A�*

	conv_loss<��>�㤤        )��P	������A�*

	conv_loss���>�u�        )��P	⪁����A�*

	conv_loss�ף>��        )��P	Hˁ����A�*

	conv_lossƺ�>���        )��P	�����A�*

	conv_lossZe�>85M        )��P	?�����A�*

	conv_losss��>��
`        )��P	b+�����A�*

	conv_lossŎ�>�n!        )��P	�J�����A�*

	conv_loss� �>��W        )��P	�j�����A�*

	conv_lossׇ�>��j        )��P	ً�����A�*

	conv_loss�7�>��        )��P	�������A�*

	conv_lossɤ>�*Y/        )��P	4̂����A�*

	conv_lossރ�>���        )��P	������A�*

	conv_loss~*�>����        )��P	"�����A�*

	conv_lossƒ�>Mد(        )��P	�*�����A�*

	conv_loss���>�1L        )��P	�L�����A�*

	conv_loss吣>�<�        )��P	Co�����A�*

	conv_loss�Ӣ>�INL        )��P	+������A�*

	conv_loss��>���        )��P	Ž�����A�*

	conv_loss���>�?�        )��P	+݃����A�*

	conv_loss���>��u        )��P	�������A�*

	conv_loss��>�l5�        )��P	4�����A�*

	conv_lossN�>i��        )��P	�=�����A�*

	conv_loss�>��`P        )��P	W_�����A�*

	conv_loss^�>):��        )��P	1������A�*

	conv_loss��>6�s�        )��P	ᡄ����A�*

	conv_loss���>Hy�        )��P	�������A�*

	conv_loss]��>�[}�        )��P	�߄����A�*

	conv_loss"��>@��        )��P	�������A�*

	conv_loss"ס>���2        )��P	�*�����A�*

	conv_loss't�>V���        )��P	�K�����A�*

	conv_loss���>
h�z        )��P	�k�����A�*

	conv_loss�l�>Aqj        )��P	�������A�*

	conv_loss":�>2I��        )��P	{������A�*

	conv_loss4�>����        )��P	�م����A�*

	conv_loss���>vB�c        )��P	�������A�*

	conv_loss�	�>?TUX        )��P	**�����A�*

	conv_loss�Z�>L-�A        )��P	EL�����A�*

	conv_loss�>"c�        )��P	m�����A�*

	conv_loss��>��g        )��P	鍆����A�*

	conv_lossm��>':>�        )��P	�������A�*

	conv_loss���>��        )��P	Cˆ����A�*

	conv_loss�+�>.J~�        )��P	w�����A�*

	conv_loss��>��/$        )��P	������A�*

	conv_loss��>�`Om        )��P	q.�����A�*

	conv_lossp�>x%�O        )��P	�M�����A�*

	conv_loss�-�>�3        )��P	(n�����A�*

	conv_loss���>��        )��P	�������A�*

	conv_lossF�>�%#�        )��P	�������A�*

	conv_loss�1�>ĩX        )��P	�ч����A�*

	conv_loss���>b��<        )��P	y�����A�*

	conv_loss��>'�α        )��P	%�����A�*

	conv_loss��>�\H        )��P	(1�����A�*

	conv_loss_ �>) C�        )��P	�R�����A�*

	conv_loss0:�>�Oh        )��P	�s�����A�*

	conv_lossel�>p��        )��P	5������A�*

	conv_lossN��>���B        )��P	������A�*

	conv_loss%�>�x�^        )��P	�ӈ����A�*

	conv_loss���>n6H�        )��P	k������A�*

	conv_loss���>��5�        )��P	h�����A�*

	conv_lossǤ�>ʫ�`        )��P	)7�����A�*

	conv_loss���>�o_�        )��P	V�����A�*

	conv_loss���>];��        )��P	!v�����A�*

	conv_lossX�>�
u�        )��P	o������A�*

	conv_loss���>/�D&        )��P	y������A�*

	conv_loss�K�>P��        )��P	zډ����A�*

	conv_loss�ɟ>�d�        )��P	�������A�*

	conv_loss�>f���        )��P	?�����A�*

	conv_loss���>uՃ?        )��P	I�����A�*

	conv_loss�s�>EzK�        )��P	�g�����A�*

	conv_loss��>��
        )��P	������A�*

	conv_lossÞ>�e�        )��P	������A�*

	conv_lossCu�>��g�        )��P	�Ŋ����A�*

	conv_loss ��>V�L}        )��P	$�����A�*

	conv_loss깡>6{u&        )��P	������A�*

	conv_loss��>x=�:        )��P	S)�����A�*

	conv_loss41�>n��U        )��P	AQ�����A�*

	conv_loss���>����        )��P	Er�����A�*

	conv_loss�;�>�w�        )��P	�������A�*

	conv_lossK5�>���        )��P	ϲ�����A�*

	conv_lossZ��>Q	��        )��P	%ы����A�*

	conv_loss�՞>�Piz        )��P	�����A�*

	conv_loss�!�>�S�        )��P	������A�*

	conv_loss�П>e�{N        )��P	C�����A�*

	conv_loss���>�x��        )��P	>f�����A�*

	conv_lossr`�>8-�Q        )��P	?������A�*

	conv_lossW�>�!        )��P	�������A�*

	conv_lossu٠>�r        )��P	�������A�*

	conv_lossx��>3�em        )��P	������A�*

	conv_loss.ѝ>��K�        )��P	-������A�*

	conv_loss��>yx�        )��P	������A�*

	conv_loss�)�>�F&        )��P	/;�����A�*

	conv_loss+.�>�7V        )��P	R\�����A�*

	conv_loss�"�>��        )��P	�������A�*

	conv_loss��>L��        )��P	c������A�*

	conv_loss�>�,B        )��P	:͍����A�*

	conv_losse��>�^a        )��P	�����A�*

	conv_loss�8�>�zi        )��P	{�����A�*

	conv_loss�x�>_~\o        )��P	1�����A�*

	conv_loss��>��        )��P	�R�����A�*

	conv_loss��>���        )��P	s�����A�*

	conv_lossշ�>;���        )��P	 ������A�*

	conv_lossQn�>�w%�        )��P	�������A�*

	conv_loss�2�>ā�z        )��P	�ӎ����A�*

	conv_loss�أ>4"�        )��P	�������A�*

	conv_loss�ŝ>A�]�        )��P	�����A�*

	conv_loss�ğ>�{^        )��P	�8�����A�*

	conv_loss ��>@�<�        )��P	Y�����A�*

	conv_lossT��>�a�        )��P	�y�����A�*

	conv_loss��>�B�        )��P	�������A�*

	conv_loss'��>*^-�        )��P	������A�*

	conv_loss^
�>�	^        )��P	�؏����A�*

	conv_lossT=�>��Z�        )��P	�������A�*

	conv_loss
G�>;�7�        )��P	������A�*

	conv_loss+��>�        )��P	�8�����A�*

	conv_loss�k�>㩋e        )��P	B\�����A�*

	conv_loss|��>�x�P        )��P	E������A�*

	conv_lossr�>�?�i        )��P	-������A�*

	conv_loss���>p�#�        )��P	?������A�*

	conv_loss��>8$F        )��P	�͑����A�*

	conv_loss~q�>ã��        )��P	a�����A�*

	conv_loss!��><P|t        )��P	������A�*

	conv_loss��>G��        )��P	�.�����A�*

	conv_loss&��>�A��        )��P	�R�����A�*

	conv_lossg�>~�X=        )��P	.u�����A�*

	conv_loss��>��#        )��P	�������A�*

	conv_loss���>��E\        )��P	�������A�*

	conv_lossC�>w|؇        )��P	+Ғ����A�*

	conv_loss��>��r�        )��P	������A�*

	conv_lossu
�>���        )��P	������A�*

	conv_loss���>XR��        )��P	�2�����A�*

	conv_lossi�>l3Li        )��P	EW�����A�*

	conv_loss�2�>�o�`        )��P	4������A�*

	conv_lossi�>�b�k        )��P	������A�*

	conv_loss]��>?"�s        )��P	�������A�*

	conv_loss�8�>"�oh        )��P	H�����A�*

	conv_loss@��>
�c        )��P	������A�*

	conv_loss�ܛ>��T        )��P	�"�����A�*

	conv_loss��>;��9        )��P	eB�����A�*

	conv_lossɬ�>�O,�        )��P	Gn�����A�*

	conv_loss# �>Q�1        )��P	������A�*

	conv_loss���>���        )��P	_������A�*

	conv_loss�Ƞ>�;�n        )��P	�͔����A�*

	conv_loss��>^-        )��P	������A�*

	conv_loss���>	���        )��P	������A�*

	conv_loss�e�>��-]        )��P	pD�����A�*

	conv_loss[��>���        )��P	�e�����A�*

	conv_loss��>�Y'�        )��P	̅�����A�*

	conv_loss��>$	#        )��P	Ӧ�����A�*

	conv_loss ~�>��5�        )��P	Kŕ����A�*

	conv_loss/��>PNe4        )��P	�����A�*

	conv_loss"��>n��}        )��P	�����A�*

	conv_lossC��>#�"	        )��P	s'�����A�*

	conv_lossJ��>��]1        )��P	G�����A�*

	conv_loss���>��%�        )��P	�h�����A�*

	conv_loss��>_-��        )��P	�������A�*

	conv_loss�R�>��        )��P	�������A�*

	conv_loss?��> {�/        )��P	Xɖ����A�*

	conv_loss�&�>i�Yo        )��P	������A�*

	conv_loss�u�>ȎR#        )��P	�	�����A�*

	conv_loss�6�>[��        )��P	S)�����A�*

	conv_lossE`�>�l��        )��P	sI�����A�*

	conv_loss�>�\;�        )��P	�h�����A�*

	conv_lossQ�>χ�        )��P	b������A�*

	conv_loss���>�|        )��P	N������A�*

	conv_loss���>X��        )��P	�ŗ����A�*

	conv_losszo�>��.9        )��P	������A�*

	conv_lossGk�>�        )��P	j�����A�*

	conv_lossⳚ>�ca        )��P	�3�����A�*

	conv_loss"��>�?{        )��P	S�����A�*

	conv_loss#��>Y���        )��P	�q�����A�*

	conv_losss�>�a��        )��P	������A�*

	conv_loss俘>���^        )��P	ʳ�����A�*

	conv_loss��>4�L�        )��P	Q֘����A�*

	conv_loss J�>Y�~�        )��P	X������A�*

	conv_loss�>z�b        )��P	������A�*

	conv_loss6O�>۬�:        )��P	�8�����A�*

	conv_loss��>����        )��P	�Y�����A�*

	conv_loss�S�>fuN�        )��P	Kx�����A�*

	conv_loss��>3��P        )��P	�������A�*

	conv_loss*s�>&�]�        )��P	q������A�*

	conv_lossq]�>Q
�\        )��P	
ݙ����A�*

	conv_loss>.�>E���        )��P	P�����A�*

	conv_loss���>�G��        )��P	c,�����A�*

	conv_loss���>j��8        )��P	+Q�����A�*

	conv_loss�W�>����        )��P	�p�����A�*

	conv_loss)�>��0        )��P	�������A�*

	conv_loss���>f��S        )��P	ռ�����A�*

	conv_loss��>z��7        )��P	Iܚ����A�*

	conv_loss9��>�>         )��P	�������A�*

	conv_lossJҝ>���        )��P	������A�*

	conv_lossK��>�z        )��P	�<�����A�*

	conv_loss0˚> ��        )��P	![�����A�*

	conv_loss���>àX�        )��P	}z�����A�*

	conv_loss'��>�Т�        )��P	x������A�*

	conv_loss�N�>�0[�        )��P	�������A�*

	conv_loss^}�>_��        )��P	�ڛ����A�*

	conv_loss���>7�x        )��P	�������A�*

	conv_loss;�>�	��        )��P	f�����A�*

	conv_loss_Қ>%���        )��P	X:�����A�*

	conv_loss��>Z�%�        )��P	>Z�����A�*

	conv_loss���>�q�
        )��P	������A�*

	conv_lossb
�>\�A�        )��P	7������A�*

	conv_loss�a�>��        )��P	�Ԝ����A�*

	conv_lossѯ�>�"<Z        )��P	�������A�*

	conv_loss���>�ɢP        )��P	�����A�*

	conv_lossI|�>͋mg        )��P	�5�����A�*

	conv_loss���>7?f�        )��P	�U�����A�*

	conv_loss_��>Q-�        )��P	6v�����A�*

	conv_loss̝>�Թj        )��P	ɕ�����A�*

	conv_loss:n�>�ʳa        )��P	5������A�*

	conv_loss%��>�.�I        )��P	;ם����A�*

	conv_loss��>�ս8        )��P	�������A�*

	conv_loss��>��`�        )��P	v�����A�*

	conv_losswڛ>o�*        )��P	s4�����A�*

	conv_loss�ٜ>'�b        )��P	tT�����A�*

	conv_loss�N�>a"�        )��P	cu�����A�*

	conv_loss��>2p_�        )��P	P������A�*

	conv_loss�x�>��|        )��P	/Ğ����A�*

	conv_loss�>|(Y        )��P	������A�*

	conv_losskv�>Zl��        )��P	������A�*

	conv_loss|}�>|��N        )��P	�%�����A�*

	conv_loss�.�>�lY        )��P	�E�����A�*

	conv_lossy��>���J        )��P	qg�����A�*

	conv_lossC[�>�C7        )��P	w������A�*

	conv_loss#`�>2{��        )��P	7������A�*

	conv_lossb�>��'�        )��P	:՟����A�*

	conv_loss=�>���        )��P	
������A�*

	conv_loss���>+��;        )��P	r�����A�*

	conv_loss�T�>���        )��P	]5�����A�*

	conv_loss/y�>&o        )��P	V�����A�*

	conv_lossO��>��Y        )��P	�u�����A�*

	conv_loss�>Q~��        )��P	�������A�*

	conv_loss�t�>#Ϣ[        )��P	�Ġ����A�*

	conv_loss䤟>J
��        )��P	������A�*

	conv_loss���>F�Q        )��P	������A�*

	conv_loss/��>���        )��P	�%�����A�*

	conv_losslĠ>��        )��P	�D�����A�*

	conv_lossP��>��du        )��P	3f�����A�*

	conv_loss|o�>f4'E        )��P	�������A�*

	conv_loss�s�>�_�u        )��P	t������A�*

	conv_lossh,�>'��        )��P	ѡ����A�*

	conv_loss��>��        )��P	/������A�*

	conv_lossY��>�2!        )��P	������A�*

	conv_loss�{�>�)&        )��P	=4�����A�*

	conv_loss,K�>%i�        )��P	�R�����A�*

	conv_lossCL�>��3�        )��P	Vv�����A�*

	conv_lossb�>[S�U        )��P	~������A�*

	conv_lossȥ�>S��g        )��P	7������A�*

	conv_loss�>�]�J        )��P	�Ѣ����A�*

	conv_loss/�>C�*�        )��P	(�����A�*

	conv_loss���>?#��        )��P	<�����A�*

	conv_lossRQ�>�"o        )��P	�-�����A�*

	conv_loss��>v'         )��P	]L�����A�*

	conv_loss�Й>�}H        )��P	�m�����A�*

	conv_loss;=�>t~1        )��P	�������A�*

	conv_loss�n�>;�w�        )��P	ޯ�����A�*

	conv_loss���>uq�        )��P	Nϣ����A�*

	conv_lossOؚ>m]O�        )��P	;�����A�*

	conv_loss�Z�>��#=        )��P	������A�*

	conv_lossv�>*��8        )��P	�,�����A�*

	conv_loss3�>��v�        )��P	ZN�����A�*

	conv_loss"�><�        )��P	�m�����A�*

	conv_loss֓�>�5�         )��P	猤����A�*

	conv_loss`�>�[��        )��P	�������A�*

	conv_loss��>Γ�        )��P	Τ����A�*

	conv_loss���>ؾ}U        )��P	I������A�*

	conv_losseɚ>i˪L        )��P	�
�����A�*

	conv_loss�a�>L
<�        )��P	�<�����A�*

	conv_lossM��>��        )��P	PZ�����A�*

	conv_lossKu�>�bCT        )��P	�y�����A�*

	conv_loss���>_���        )��P	Ϙ�����A�*

	conv_loss��>I!J,        )��P	�������A�*

	conv_loss�\�>���n        )��P	gإ����A�*

	conv_loss߹�>x�#�        )��P	�������A�*

	conv_loss��>��K         )��P	������A�*

	conv_lossx��>��M        )��P	F;�����A�*

	conv_lossI�>��N�        )��P	VZ�����A�*

	conv_loss��>�K        )��P	�z�����A�*

	conv_loss.c�>�W��        )��P	5������A�*

	conv_loss�'�>��&        )��P	AǦ����A�*

	conv_lossw`�>��        )��P	T�����A�*

	conv_lossD�>[�9�        )��P	L�����A�*

	conv_loss��>���        )��P	2�����A�*

	conv_loss-/�>b:�        )��P	uR�����A�*

	conv_loss���>>)��        )��P	�q�����A�*

	conv_lossR��>�p        )��P	ڔ�����A�*

	conv_loss�6�>ݻD�        )��P	1������A�*

	conv_loss���>��{�        )��P	�ԧ����A�*

	conv_loss�R�>��"�        )��P	������A�*

	conv_loss8�>�*~�        )��P	������A�*

	conv_loss�K�>��        )��P	�8�����A�*

	conv_loss6��>v;�"        )��P	�X�����A�*

	conv_lossؙ>�]pC        )��P	�x�����A�*

	conv_lossSę>�|�9        )��P	�������A�*

	conv_lossIP�>�IX        )��P	�������A�*

	conv_loss���>�_a"        )��P	ڨ����A�*

	conv_losse �>��p�        )��P	�������A�*

	conv_loss�4�>ӅvT        )��P	�#�����A�*

	conv_lossL2�>�<iL        )��P	JL�����A�*

	conv_loss'��>f�s�        )��P	�p�����A�*

	conv_loss6�>�L��        )��P	c������A�*

	conv_loss��>��4        )��P	ѱ�����A�*

	conv_loss�`�>}%n        )��P	0ѩ����A�*

	conv_loss�˚>�&�        )��P	������A�*

	conv_lossA�>�}�        )��P	������A�*

	conv_lossDԖ>{�A        )��P	�3�����A�*

	conv_loss�^�>�=��        )��P	S�����A�*

	conv_loss���>�il�        )��P	�q�����A�*

	conv_loss�b�>[Gt�        )��P	ϑ�����A�*

	conv_lossu/�>E�        )��P	K������A�*

	conv_loss��>�rU!        )��P	�Ҫ����A�*

	conv_lossW�>�.�        )��P	������A�*

	conv_loss릙>l.�        )��P	~�����A�*

	conv_loss��>N���        )��P	�5�����A�*

	conv_loss���>�OA�        )��P	�U�����A�*

	conv_lossV�>W��5        )��P	`u�����A�*

	conv_loss�>���c        )��P	Օ�����A�*

	conv_lossSa�>"4¢        )��P	t������A�*

	conv_lossA$�>�ky�        )��P	�Ȭ����A�*

	conv_lossn6�>��V�        )��P	1�����A�*

	conv_loss�ڗ>��#|        )��P	�
�����A�*

	conv_loss��>��~        )��P	�*�����A�*

	conv_loss�E�>���t        )��P	�K�����A�*

	conv_lossH$�>Hf�H        )��P	 k�����A�*

	conv_lossX˘>|�r�        )��P	ŋ�����A�*

	conv_loss0+�>�*�        )��P	�������A�*

	conv_loss�M�>J�        )��P	I̭����A�*

	conv_lossZg�>��	�        )��P	������A�*

	conv_loss��>V�o        )��P	�
�����A�*

	conv_loss	��>2��3        )��P	�.�����A�*

	conv_lossL��>�N`�        )��P	b�����A�*

	conv_loss��>�)��        )��P	>������A�*

	conv_loss�̙>��a�        )��P	�������A�*

	conv_loss�2�>� �        )��P	�Ʈ����A�*

	conv_loss�З>���        )��P	������A�*

	conv_loss��>��,a        )��P	�����A�*

	conv_loss3�>�d�A        )��P	�)�����A�*

	conv_loss���>:!�.        )��P	�K�����A�*

	conv_lossz/�>� ��        )��P	+s�����A�*

	conv_loss�>8�`        )��P	������A�*

	conv_lossg#�>\Q�        )��P	Ǳ�����A�*

	conv_loss5��>��9        )��P	�Я����A�*

	conv_lossfϙ>���S        )��P	v������A�*

	conv_losszs�>�_x        )��P	������A�*

	conv_loss��>8E        )��P	/7�����A�*

	conv_lossn?�>5�0�        )��P	dV�����A�*

	conv_lossɕ>Q���        )��P	�w�����A�*

	conv_lossJ��>�u��        )��P	�������A�*

	conv_loss:��>.=��        )��P	�������A�*

	conv_loss�+�>���        )��P	�߰����A�*

	conv_loss䘘>2I�        )��P	� �����A�*

	conv_loss���>I_��        )��P	n"�����A�*

	conv_loss��>*l_�        )��P	*C�����A�*

	conv_loss���>X1i�        )��P	|d�����A�*

	conv_loss�>�;�H        )��P	2������A�*

	conv_loss;ʗ>��F        )��P	�������A�*

	conv_loss��>Э$        )��P	�ȱ����A�*

	conv_loss�@�>}�        )��P	�����A�*

	conv_lossQ�>�e
b        )��P	������A�*

	conv_loss$$�>Qa,Z        )��P	.�����A�*

	conv_loss��>*� w        )��P	�N�����A�*

	conv_lossֺ�>��        )��P	�m�����A�*

	conv_loss��>�n��        )��P	f������A�*

	conv_lossU�>9��        )��P	������A�*

	conv_loss��>p0��        )��P	�Ѳ����A�*

	conv_loss9Q�>p�J        )��P	�����A�*

	conv_lossh�>7���        )��P	f�����A�*

	conv_loss�L�>[�#        )��P	?�����A�*

	conv_lossX2�>c�Y        )��P	�^�����A�*

	conv_loss��>_��G        )��P	�������A�*

	conv_loss*7�>K�A�        )��P	7������A�*

	conv_loss�Y�>�;#         )��P	澳����A�*

	conv_loss���>g4�6        )��P	������A�*

	conv_loss[��>�1h        )��P	�����A�*

	conv_loss�%�>���9        )��P	3(�����A�*

	conv_lossч�>1�R�        )��P	�H�����A�*

	conv_lossJ�>�p`�        )��P	Gh�����A�*

	conv_loss���>���        )��P	�������A�*

	conv_lossc*�>=�/C        )��P	�������A�*

	conv_loss�G�>�}        )��P	6ڴ����A�*

	conv_loss}��>��hs        )��P	�������A�*

	conv_loss�ٓ>(�         )��P	������A�*

	conv_loss'U�>�ս�        )��P	q<�����A�*

	conv_loss]��>��f�        )��P	�[�����A�*

	conv_lossͦ�>cŇ        )��P	�������A�*

	conv_loss��>��6@        )��P	^������A�*

	conv_loss/N�>���        )��P	�ɵ����A�*

	conv_loss���>J'��        )��P	P������A�*

	conv_lossԽ�>�k��        )��P	Z�����A�*

	conv_lossWz�>gDd�        )��P	/;�����A�*

	conv_loss��>,��W        )��P	\�����A�*

	conv_loss�b�>6��        )��P	�{�����A�*

	conv_loss	��>�T
q        )��P	^������A�*

	conv_loss*�>>v��        )��P	�������A�*

	conv_loss""�>��H        )��P	�����A�*

	conv_loss��>J�,�        )��P	������A�*

	conv_loss(�>F        )��P	�6�����A�*

	conv_loss��>�75        )��P	�V�����A�*

	conv_loss��>_(�        )��P	)������A�*

	conv_loss)v�>zT�`        )��P	d������A�*

	conv_loss0Θ>i�(        )��P	h������A�*

	conv_loss*�>עNa        )��P	�ܷ����A�*

	conv_loss&X�>�h��        )��P	�������A�*

	conv_loss�Z�>�Ev        )��P	�,�����A�*

	conv_loss�D�>Fp~f        )��P	8P�����A�*

	conv_loss4�>�2AH        )��P	lt�����A�*

	conv_lossVn�>�w�
        )��P	�������A�*

	conv_lossqQ�>��ʚ        )��P	�������A�*

	conv_loss���>�YX+        )��P	�߸����A�*

	conv_loss���> c7        )��P	T������A�*

	conv_loss4�>�ĳ�        )��P	������A�*

	conv_loss���>fz�/        )��P	�@�����A�*

	conv_loss�_�>~�"e        )��P	O_�����A�*

	conv_lossUx�>Z��        )��P	 ������A�*

	conv_loss﷓>n�#�        )��P	U������A�*

	conv_loss�ؔ>�Z	        )��P	�Ĺ����A�*

	conv_loss+}�>����        )��P	b �����A�*

	conv_loss�w�>9v��        )��P	�-�����A�*

	conv_lossS��>���        )��P	�M�����A�*

	conv_loss�>��        )��P	[n�����A�*

	conv_loss��>Ç�        )��P	ߒ�����A�*

	conv_loss7�>�T�r        )��P	ѱ�����A�*

	conv_loss�(�>I�i        )��P	�Ծ����A�*

	conv_loss��>M}�u        )��P	�������A�*

	conv_loss���>���        )��P	�����A�*

	conv_loss���>�{�        )��P	4�����A�*

	conv_lossW��>�C~i        )��P	�V�����A�*

	conv_lossM�>Qҿ�        )��P	hw�����A�*

	conv_loss��>�~-|        )��P	������A�*

	conv_loss?��>;�3�        )��P	�������A�*

	conv_loss���>�!��        )��P	������A�*

	conv_loss]�>�a��        )��P	������A�*

	conv_loss^z�>��=        )��P	�4�����A�*

	conv_lossW�>���c        )��P	~U�����A�*

	conv_loss# �>n�        )��P	�t�����A�*

	conv_loss��>���Q        )��P	�������A�*

	conv_loss��>{�.�        )��P	������A�*

	conv_loss�D�>��1�        )��P	�������A�*

	conv_loss���>�.�        )��P	a������A�*

	conv_loss@��>e�Rh        )��P	������A�*

	conv_loss�I�>5�        )��P	�?�����A�*

	conv_lossד�>�6I#        )��P	+`�����A�*

	conv_loss��>\
O�        )��P	������A�*

	conv_loss���>�+wZ        )��P	o������A�*

	conv_loss��>l�lx        )��P	L������A�*

	conv_loss���>uT��        )��P	H������A�*

	conv_loss�j�>�t�        )��P	�������A�*

	conv_loss�W�>&�!�        )��P	������A�*

	conv_loss���>K��        )��P	�A�����A�*

	conv_loss��>U�^        )��P	�f�����A�*

	conv_lossU�>���        )��P	9������A�*

	conv_loss���>=h�=        )��P	"������A�*

	conv_loss{Õ>�sIB        )��P	������A�*

	conv_lossq�>sΈ�        )��P	�������A�*

	conv_loss�&�>�Xf        )��P	f�����A�*

	conv_loss�6�>��        )��P	Z3�����A�*

	conv_loss��>98�]        )��P	�T�����A�*

	conv_loss�t�>���        )��P	Zv�����A�*

	conv_lossg��>٨�        )��P	>������A�*

	conv_loss���>7O	s        )��P	-������A�*

	conv_loss�>3�        )��P	�������A�*

	conv_loss��>���        )��P	������A�*

	conv_loss�I�>4���        )��P	h�����A�*

	conv_lossOt�>� �        )��P	8�����A�*

	conv_loss�y�>V��f        )��P	W�����A�*

	conv_lossC��>�8.�        )��P	Pw�����A�*

	conv_loss�E�>�8Ă        )��P	�������A�*

	conv_lossTh�>2YpB        )��P	������A�*

	conv_loss&U�>7��        )��P	������A�*

	conv_loss>��>��)�        )��P	������A�*

	conv_loss���>�u<;        )��P	#�����A�*

	conv_loss�B�>5_$�        )��P	+B�����A�*

	conv_loss2ߔ>����        )��P	�b�����A�*

	conv_loss|o�>ͯ�        )��P	�������A�*

	conv_loss��>iX�E        )��P	�������A�*

	conv_loss�t�>�E}
        )��P	n������A�*

	conv_loss(ȗ>K�'x        )��P	�������A�*

	conv_loss���>I�
        )��P	s�����A�*

	conv_loss��>P�B        )��P	M4�����A�*

	conv_loss��>܀�D        )��P	xT�����A�*

	conv_loss#��>�ܞ�        )��P	�t�����A�*

	conv_lossSm�>Ѹ3�        )��P	�������A�*

	conv_loss�N�>-��        )��P	������A�*

	conv_loss�_�>��M        )��P	)������A�*

	conv_loss���>X���        )��P	z������A�*

	conv_loss7�>�+        )��P	������A�*

	conv_loss=K�>��t(        )��P	9I�����A�*

	conv_loss�C�>n��4        )��P	5o�����A�*

	conv_lossc�>\I�        )��P	�������A�*

	conv_loss�2�>���|        )��P	�������A�*

	conv_loss�	�> �L�        )��P	�������A�*

	conv_loss ��>?x        )��P	v������A�*

	conv_loss3q�>m�^>        )��P	������A�*

	conv_loss̍>�g�        )��P	L8�����A�*

	conv_loss��>}�[f        )��P	WX�����A�*

	conv_loss^ �>�^��        )��P	�x�����A�*

	conv_loss���>q��*        )��P	W������A�*

	conv_loss�ˎ>��L�        )��P	�������A�*

	conv_loss�>�p!        )��P	�������A�*

	conv_loss줓>��-        )��P	[������A�*

	conv_lossĔ�>4��b        )��P	������A�*

	conv_loss䵑>S)�        )��P	Y?�����A�*

	conv_loss���>e��f        )��P	�_�����A�*

	conv_loss�А>@� v        )��P	P�����A�*

	conv_loss���>啙        )��P	l������A�*

	conv_losss�>,OL        )��P	�������A�*

	conv_loss���>�a+        )��P	D������A�*

	conv_lossKY�>����        )��P	| �����A�*

	conv_lossԏ>S)��        )��P	M!�����A�*

	conv_lossG&�>��m        )��P	 A�����A�*

	conv_loss�S�>�fZ�        )��P	�c�����A�*

	conv_loss�f�>�u�        )��P	�������A�*

	conv_loss���>�>�}        )��P	9������A�*

	conv_loss߿�>���        )��P	�������A�*

	conv_loss#��>����        )��P	������A�*

	conv_loss���>gDE\        )��P	�����A�*

	conv_loss��>;�\        )��P	�#�����A�*

	conv_loss���>:\(�        )��P	b2�����A�*

	conv_loss��>���        )��P	�R�����A�*

	conv_loss�5�>vgJ�        )��P	vs�����A�*

	conv_loss���>+��        )��P	/������A�*

	conv_loss�
�>й۱        )��P	|������A�*

	conv_loss���>�kDa        )��P	\������A�*

	conv_losse#�>p|�?        )��P	O�����A�*

	conv_lossP��>cj�        )��P	�#�����A�*

	conv_loss�c�>�Xy9        )��P	�F�����A�*

	conv_loss��> )I        )��P	�k�����A�*

	conv_loss���>��H        )��P	I������A�*

	conv_loss�m�>jP��        )��P	7������A�*

	conv_loss��>Q�3d        )��P	y������A�*

	conv_lossi��>���        )��P	�������A�*

	conv_lossv��>5�t�        )��P	�����A�*

	conv_lossWJ�>����        )��P	L6�����A�*

	conv_lossé�>:��8        )��P	'a�����A�*

	conv_loss�-�>��s�        )��P	&������A�*

	conv_loss4ܒ>�         )��P	+������A�*

	conv_loss�d�>�ϩ�        )��P	)������A�*

	conv_loss�u�>W�O        )��P	x������A�*

	conv_loss/8�>���        )��P	)�����A�*

	conv_lossႏ>���        )��P	�5�����A�*

	conv_losswI�>y�        )��P	�V�����A�*

	conv_lossL��>�}��        )��P	[w�����A�*

	conv_lossJ%�>4{g�        )��P	�������A�*

	conv_loss���>�J�h        )��P	ٶ�����A�*

	conv_loss"�>�1J[        )��P	�������A�*

	conv_loss��>co�h        )��P	�������A�*

	conv_lossZ��>A���        )��P	p�����A�*

	conv_lossΒ>��M\        )��P	�3�����A�*

	conv_lossWD�>��        )��P	 T�����A�*

	conv_loss5��>��^         )��P	ks�����A�*

	conv_lossĊ�>��@�        )��P	.������A�*

	conv_loss��>�BI        )��P	�������A�*

	conv_lossn�>n
�P        )��P	������A�*

	conv_loss���>�%	        )��P	�������A�*

	conv_loss���>�*S        )��P	������A�*

	conv_loss�Γ>���p        )��P	�6�����A�*

	conv_lossu��>��        )��P	�Z�����A�*

	conv_loss��>7��        )��P	�}�����A�*

	conv_loss�z�>O���        )��P	������A�*

	conv_loss(�>���        )��P	�������A�*

	conv_loss�@�>}�u�        )��P	�������A�*

	conv_loss"��>u�2        )��P	O�����A�*

	conv_loss]��>x��        )��P	 #�����A�*

	conv_loss:�>}�bX        )��P	�C�����A�*

	conv_loss�ڎ>j��        )��P	�b�����A�*

	conv_loss�5�>��        )��P	�������A�*

	conv_losse�>���        )��P	K������A�*

	conv_lossfj�>/��        )��P	I������A�*

	conv_loss�L�>�^��        )��P	�������A�*

	conv_loss勐>q�A+        )��P	p�����A�*

	conv_loss���>���        )��P	�6�����A�*

	conv_lossO�>m�U4        )��P	]V�����A�*

	conv_loss���>*�?        )��P	(w�����A�*

	conv_loss�f�>�;�        )��P	+������A�*

	conv_loss빐>��        )��P	g������A�*

	conv_loss��>���        )��P	�������A�*

	conv_lossq��>ܻ*        )��P	B������A�*

	conv_loss�[�>�>��        )��P	�"�����A�*

	conv_loss���>�Z|        )��P	eD�����A�*

	conv_loss�ʒ>0e��        )��P	`e�����A�*

	conv_lossN�>&��        )��P	�������A�*

	conv_loss��>����        )��P	a������A�*

	conv_loss���>W��        )��P	?������A�*

	conv_lossp3�>����        )��P	�������A�*

	conv_loss�>�u�{        )��P	�����A�*

	conv_loss.1�>f�[�        )��P	.=�����A�*

	conv_loss��>�8%�        )��P	r\�����A�*

	conv_loss*3�>� O        )��P	�������A�*

	conv_lossum�>�*        )��P	������A�*

	conv_loss:a�>���        )��P	\������A�*

	conv_loss���>��#        )��P	�������A�*

	conv_loss	�>�J�        )��P	������A�*

	conv_loss�؎>��"F        )��P	�&�����A�*

	conv_loss���>��+�        )��P	�F�����A�*

	conv_loss��>)�_P        )��P	�g�����A�*

	conv_loss;3�>�	��        )��P	t������A�*

	conv_loss�$�>S���        )��P	ֲ�����A�*

	conv_loss�u�>��fq        )��P	l������A�*

	conv_loss�S�>R]j        )��P	�������A�*

	conv_losss��>��^>        )��P	G�����A�*

	conv_loss6�>�"��        )��P	�<�����A�*

	conv_loss�#�>���G        )��P	\�����A�*

	conv_loss�A�>>�!�        )��P	`|�����A�*

	conv_loss��>ޞ��        )��P	�������A�*

	conv_lossa��>�v~        )��P	�������A�*

	conv_loss�5�>�:�.        )��P	?������A�*

	conv_loss�#�>I���        )��P	�������A�*

	conv_loss9ӎ>��        )��P	q�����A�*

	conv_loss꾎>��V        )��P	V;�����A�*

	conv_loss���>|��        )��P	�Z�����A�*

	conv_loss�4�>+H�        )��P	z�����A�*

	conv_loss3ȏ>Q="        )��P	�������A�*

	conv_loss�ҏ>�� �        )��P	�������A�*

	conv_lossw�>H�:�        )��P	/������A�*

	conv_loss���>]�        )��P	�������A�*

	conv_loss���>i%        )��P	������A�*

	conv_loss�d�>�s3�        )��P	�:�����A�*

	conv_loss{:�>ӜJP        )��P	�g�����A�*

	conv_loss7��>LV��        )��P	������A�*

	conv_lossᵏ>&��        )��P	�������A�*

	conv_loss6p�>:�J*        )��P	�������A�*

	conv_lossr�>9M��        )��P	\������A�*

	conv_loss���>��b_        )��P	�
�����A�*

	conv_loss��>(��        )��P	,�����A�*

	conv_losszI�>�8C�        )��P	�J�����A�*

	conv_lossi{�>��         )��P	�j�����A�*

	conv_loss	��>���         )��P	w������A�*

	conv_loss�Ҏ>��e<        )��P	�������A�*

	conv_loss؀�>a���        )��P	�������A�*

	conv_loss���>����        )��P	]������A�*

	conv_loss��>w�n        )��P	������A�*

	conv_loss�+�>z� K        )��P	@:�����A�*

	conv_loss�U�>5c�J        )��P	�[�����A�*

	conv_loss�N�>F,zD        )��P	$������A�*

	conv_loss�>S��        )��P	�������A�*

	conv_loss��>�s�        )��P	)������A�*

	conv_loss��>�Pn|        )��P	g������A�*

	conv_loss)4�>��H3        )��P	������A�*

	conv_loss@��>�O        )��P	B9�����A�*

	conv_loss��>�̶        )��P	Y�����A�*

	conv_loss���>�o�        )��P	�x�����A�*

	conv_lossG`�>��^d        )��P	Θ�����A�*

	conv_loss���>v��8        )��P	������A�*

	conv_loss3��>\�cG        )��P	O������A�*

	conv_loss��>5�w,        )��P	q������A�*

	conv_loss��>�m�&        )��P	������A�*

	conv_loss�>��)�        )��P	)6�����A�*

	conv_lossD��>��Բ        )��P	TU�����A�*

	conv_loss*V�>�{�        )��P	fu�����A�*

	conv_loss�ʍ>#��        )��P	ߕ�����A�*

	conv_loss>�>�fr�        )��P	(������A�*

	conv_lossGh�>�X,�        )��P	w������A�*

	conv_loss�`�>_���        )��P	"������A�*

	conv_lossӎ>(��M        )��P	������A�*

	conv_loss?Ê>(S�J        )��P	�8�����A�*

	conv_loss�m�>��Yr        )��P	Y�����A�*

	conv_loss��>�c�        )��P	z�����A�*

	conv_loss1͏>���
        )��P	�������A�*

	conv_loss���>H��H        )��P	������A�*

	conv_loss�>)١�        )��P	k������A�*

	conv_loss��>Sv0        )��P	�������A�*

	conv_loss�>&B�        )��P	I�����A�*

	conv_loss�3�>ĜJ�        )��P	b>�����A�*

	conv_loss׋>a��i        )��P	#b�����A�*

	conv_lossq{�>���        )��P	�������A�*

	conv_loss��>cd�L        )��P	�������A�*

	conv_loss�o�>��E        )��P	G������A�*

	conv_loss�>�M�        )��P	�������A�*

	conv_loss6��>�g�        )��P	������A�*

	conv_lossxِ>�)m[        )��P	�/�����A�*

	conv_lossBf�>!�n        )��P	4R�����A�*

	conv_lossz��>��)�        )��P	�t�����A�*

	conv_loss�:�>�(�        )��P	�������A�*

	conv_loss폍>��\R        )��P	<������A�*

	conv_loss��>F�M�        )��P	G������A�*

	conv_loss�[�>e��        )��P	������A�*

	conv_loss�ΐ>{gK        )��P	�$�����A�*

	conv_loss���>N�`�        )��P	�D�����A�*

	conv_loss�C�>5q1<        )��P	�e�����A�*

	conv_lossd�>�֞�        )��P	\������A�*

	conv_loss���>'c�        )��P	�������A�*

	conv_loss�D�>-$��        )��P	������A�*

	conv_loss�~�>pl��        )��P	�������A�*

	conv_loss4��>���        )��P	�����A�*

	conv_loss�y�>�=        )��P	c1�����A�*

	conv_lossQ��>�_K^        )��P	Q�����A�*

	conv_loss���>`�$�        )��P	bq�����A�*

	conv_loss5�>N�o�        )��P	�������A�*

	conv_loss���>i7        )��P	�������A�*

	conv_loss�Ŏ>�P�        )��P	�������A�*

	conv_lossN�>�=��        )��P	 ������A�*

	conv_loss)Ό>�0��        )��P	������A�*

	conv_loss�>���        )��P	�8�����A�*

	conv_lossL�>3��         )��P		[�����A�*

	conv_lossF�>�ji�        )��P	*{�����A�*

	conv_lossoa�>���=        )��P	D������A�*

	conv_losso��>���O        )��P	M������A�*

	conv_loss��>�,0Y        )��P	�������A�*

	conv_loss$�>�,�        )��P	} �����A�*

	conv_lossz�>a�        )��P		�����A�*

	conv_loss�A�>�(Z�        )��P	�O�����A�*

	conv_loss�J�>~b�        )��P	�o�����A�*

	conv_loss*;�>6�qY        )��P	�������A�*

	conv_lossQ �>��4E        )��P	�������A�*

	conv_loss^��>q`=        )��P	>������A�*

	conv_lossU��>�jlZ        )��P	�������A�*

	conv_lossY�>���>        )��P	x�����A�*

	conv_loss�J�>�z�P        )��P	k/�����A�*

	conv_lossb��>	�
�        )��P	dR�����A�*

	conv_lossr�><̲        )��P	�p�����A�*

	conv_lossp�>���V        )��P	!������A�*

	conv_loss��>w,�        )��P	j������A�*

	conv_lossȮ�>v�7        )��P	�������A�*

	conv_loss�^�>���        )��P	�������A�*

	conv_loss+�>�RW?        )��P	������A�*

	conv_lossp5�>�Wfd        )��P	�/�����A�*

	conv_lossbW�>C�(�        )��P	^N�����A�*

	conv_lossvۉ>��W        )��P	�_�����A�*

	conv_loss��>�A��        )��P	�������A�*

	conv_loss��>[        )��P	k������A�*

	conv_loss�W�>��4�        )��P	r������A�*

	conv_loss^7�>B��        )��P	������A�*

	conv_loss��>�õ\        )��P	}������A�*

	conv_loss�;�>RG}�        )��P	a!�����A�*

	conv_loss�[�>?O�        )��P	R@�����A�*

	conv_loss�c�>�MF        )��P	�_�����A�*

	conv_loss���>��9        )��P	�������A�*

	conv_loss���>,��        )��P	������A�*

	conv_loss8�>[0        )��P	�������A�*

	conv_lossE\�>|:��        )��P	�������A�*

	conv_loss��>�R�        )��P	�����A�*

	conv_loss���>���        )��P	�0�����A�*

	conv_lossbv�>KkAM        )��P	�P�����A�*

	conv_lossvD�>��`�        )��P	�o�����A�*

	conv_loss�]�>�&�        )��P	9������A�*

	conv_lossn�>1Q�        )��P	�������A�*

	conv_loss�@�>�%��        )��P	�������A�*

	conv_lossO+�>i�*        )��P	g������A�*

	conv_loss�F�>M�        )��P	������A�*

	conv_loss=V�>��\�        )��P	U<�����A�*

	conv_loss�P�>��^�        )��P	�]�����A�*

	conv_loss���>3^4=        )��P	�|�����A�*

	conv_loss�+�>_#�        )��P	Т�����A�*

	conv_loss�_�>�UT�        )��P	�������A�*

	conv_loss��>=m��        )��P	������A�*

	conv_loss�>n��        )��P	������A�*

	conv_loss���>����        )��P	 /�����A�*

	conv_lossń�>j�        )��P	�Q�����A�*

	conv_loss�C�>�#Ύ        )��P	�r�����A�*

	conv_loss�.�>����        )��P	T������A�*

	conv_loss��>t\�        )��P	�������A�*

	conv_loss��>s�`        )��P	!������A�*

	conv_loss��>��ީ        )��P	�������A�*

	conv_lossXT�>Q,�}        )��P	G�����A�*

	conv_loss�>����        )��P	[9�����A�*

	conv_loss��>��2�        )��P	�]�����A�*

	conv_lossP��>`��o        )��P	������A�*

	conv_loss�	�>P&!        )��P	�������A�*

	conv_loss��>��        )��P	I������A�*

	conv_loss��>Q��<        )��P	�������A�*

	conv_loss)�>ch�i        )��P	������A�*

	conv_loss�s�>�s^I        )��P	�'�����A�*

	conv_loss-��>A��        )��P	�H�����A�*

	conv_loss���>���        )��P	0k�����A�*

	conv_loss��>�i�M        )��P	�������A�*

	conv_lossŜ�>�/B�        )��P	������A�*

	conv_loss���>Ƀ        )��P	������A�*

	conv_loss�Љ>�C,        )��P	�������A�*

	conv_loss!*�>	��        )��P	u�����A�*

	conv_loss��>B{�        )��P	�>�����A�*

	conv_lossgÃ>T�28        )��P	�_�����A�*

	conv_losss��>�+U�        )��P	������A�*

	conv_lossW�>�p5$        )��P	������A�*

	conv_loss��>��n�        )��P	Z������A�*

	conv_loss��>���O        )��P	M������A�*

	conv_lossL
�>s�#        )��P	������A�*

	conv_loss��>'禹        )��P	&�����A�*

	conv_lossM��>j�D        )��P	%O�����A�*

	conv_loss�y�>a�D�        )��P	�o�����A�*

	conv_loss썈>@���        )��P	֎�����A�*

	conv_loss��>PvQ�        )��P	ٱ�����A�*

	conv_loss�B�>YУ        )��P	������A�*

	conv_loss�E�>20L        )��P	 �����A�*

	conv_loss,�>�P��        )��P	0#�����A�*

	conv_loss��>k4M�        )��P	]J�����A�*

	conv_loss���>ةb        )��P	�n�����A�*

	conv_lossw��>	��        )��P	~������A�*

	conv_loss��>Mn=P        )��P	B������A�*

	conv_loss98�>�)4}        )��P	Q������A�*

	conv_loss�m�>
z�j        )��P	v������A�*

	conv_loss�U�>0        )��P	^�����A�*

	conv_loss[=�>?8�        )��P	�2�����A�*

	conv_lossvo�>>�        )��P	S�����A�*

	conv_lossF�>�=�        )��P	�t�����A�*

	conv_lossbو>�ͥ)        )��P	֕�����A�*

	conv_loss���>EV�        )��P	�������A�*

	conv_lossW΄>Z��        )��P	S������A�*

	conv_loss��>��`�        )��P	�������A�*

	conv_loss�,�>�d�+        )��P	������A�*

	conv_lossoX�>�)��        )��P	nC�����A�*

	conv_lossA��>����        )��P	Dh�����A�*

	conv_loss+-�>d�s�        )��P	�������A�*

	conv_loss~��>�(�        )��P	������A�*

	conv_loss�v�>̙��        )��P	x������A�*

	conv_loss�u�>rċ�        )��P	�������A�*

	conv_loss^Ӆ>��&o        )��P	R�����A�*

	conv_loss���>J        )��P	�7�����A�*

	conv_loss ܇>g�G        )��P	�_�����A�*

	conv_loss�ى>��2~        )��P	ă�����A�*

	conv_loss��>nYT�        )��P	ѥ�����A�*

	conv_loss�r�>ud�        )��P	�������A�*

	conv_lossЮ�> y�%        )��P	T������A�*

	conv_loss�Ì>�[�5        )��P	�
�����A�*

	conv_loss~:�>�?��        )��P	J,�����A�*

	conv_loss�Ɔ><��        )��P	�L�����A�*

	conv_lossx�>��        )��P	cn�����A�*

	conv_loss�K�>���        )��P	������A�*

	conv_loss٪�>�qr�        )��P	�������A�*

	conv_lossW�>�JR�        )��P	�������A�*

	conv_loss���>qq�        )��P	,	�����A�*

	conv_lossM�>�͍}        )��P	�)�����A�*

	conv_loss��>�t��        )��P	nJ�����A�*

	conv_lossjv�>2��4        )��P	\n�����A�*

	conv_loss��>�RK�        )��P	������A�*

	conv_loss�Z�>�u6M        )��P	v������A�*

	conv_lossק�>�3Sd        )��P	n������A�*

	conv_loss��>c�O�        )��P	q������A�*

	conv_losssx~>9e6�        )��P	 �����A�*

	conv_loss���>i         )��P	_:�����A�*

	conv_loss���>~�=\        )��P	k�����A�*

	conv_loss�Ԉ>�/}�        )��P	������A�*

	conv_loss���>z��        )��P	�������A�*

	conv_lossyՃ>��Z        )��P	q������A�*

	conv_lossms�>n���        )��P	�������A�*

	conv_loss#��><��a        )��P	2�����A�*

	conv_loss�/�>��V        )��P	p3�����A�*

	conv_loss=`�>re�        )��P	�g�����A�*

	conv_loss�B{>il�h        )��P	X������A�*

	conv_loss�x�>>��        )��P	������A�*

	conv_loss��>��        )��P	�������A�*

	conv_loss���>�g�z        )��P	�������A�*

	conv_loss��>� �        )��P	������A�*

	conv_losso��>�Ү�        )��P	z7�����A�*

	conv_loss�h�>�L,�        )��P	jY�����A�*

	conv_loss�/�>r���        )��P	3{�����A�*

	conv_loss/��>x��        )��P	�������A�*

	conv_loss�O�>����        )��P	������A�*

	conv_loss�7�>���        )��P	�������A�*

	conv_loss���>�        )��P	�������A�*

	conv_loss|�>9��        )��P	� �����A�*

	conv_loss]Z�>�gӚ        )��P	zA�����A�*

	conv_loss�{�>"�@E        )��P	Fa�����A�*

	conv_lossc)�>�!�=        )��P	Ȅ�����A�*

	conv_loss�I�>����        )��P	�������A�*

	conv_loss�φ>*n�k        )��P	h������A�*

	conv_loss���>@yQn        )��P	�������A�*

	conv_loss��>{�W        )��P	@
�����A�*

	conv_loss)�>��[o        )��P	�*�����A�*

	conv_lossނ>oH�        )��P	&K�����A�*

	conv_loss�Y�>,u        )��P	bm�����A�*

	conv_loss8d�>��),        )��P	�������A�*

	conv_loss�8�>�҅�        )��P	������A�*

	conv_loss�c�>�v{m        )��P	�������A�*

	conv_loss➈>����        )��P	%������A�*

	conv_lossi�>S�o        )��P	k�����A�*

	conv_lossOG�>ߘ��        )��P	�3�����A�*

	conv_loss�Ȅ>��)�        )��P	�T�����A�*

	conv_loss0�> $�        )��P	#������A�*

	conv_loss/��>�3i        )��P	$������A�*

	conv_loss8��>���        )��P	'������A�*

	conv_loss��y>��\        )��P	������A�*

	conv_loss�~>DncM        )��P	J�����A�*

	conv_lossɈ>[�ۃ        )��P	�.�����A�*

	conv_loss��>���~        )��P	WS�����A�*

	conv_loss庀>�l˾        )��P	������A�*

	conv_lossxd�>ˠ��        )��P	}������A�*

	conv_loss؁>�[o        )��P	D������A�*

	conv_loss�>Dk�        )��P	�������A�*

	conv_loss=��>G�O        )��P	o�����A�*

	conv_loss#�>Y���        )��P	x.�����A�*

	conv_loss=�>P�Q        )��P	T�����A�*

	conv_lossO�>m�        )��P	�u�����A�*

	conv_loss��>	A�M        )��P	�������A�*

	conv_loss�F�>�A*�        )��P	������A�*

	conv_loss���>�=F        )��P	������A�*

	conv_loss��>���        )��P	�������A�*

	conv_loss���>e͹a        )��P	#�����A�*

	conv_lossju�>��%        )��P	�F�����A�*

	conv_loss�n�>q�p        )��P	�h�����A�*

	conv_lossKĀ>�{N+        )��P	������A�*

	conv_loss�i�>,oO�        )��P	)������A�*

	conv_loss��>��G�        )��P	�������A�*

	conv_lossYɂ>�$�q        )��P	�������A�*

	conv_loss��>#���        )��P	9�����A�*

	conv_loss�T>���        )��P	�B�����A�*

	conv_loss 1�>��ʰ        )��P	�g�����A�*

	conv_loss�a�>��Yx        )��P	�������A�*

	conv_loss�f�>��        )��P	O������A�*

	conv_lossK_w>]��        )��P	�������A�*

	conv_lossq��>��f�        )��P	�������A�*

	conv_loss��>:q��        )��P	k ����A�*

	conv_loss;��>���
        )��P	�3 ����A�*

	conv_loss�5�>R�        )��P	�S ����A�*

	conv_loss��>�G��        )��P	�s ����A�*

	conv_loss�H�>���V        )��P	�� ����A�*

	conv_loss�i�>69l        )��P	�� ����A�*

	conv_loss/@�>?4�        )��P	1� ����A�*

	conv_loss��>t�m�        )��P	�� ����A�*

	conv_losso�>�<�        )��P	�����A�*

	conv_loss���>d��        )��P	�:����A�*

	conv_loss�v�>��        )��P	�[����A�*

	conv_lossY��>`=N�        )��P	�}����A�*

	conv_lossDm�>�\�3        )��P	�����A�*

	conv_lossٟ�>��$        )��P	�����A�*

	conv_loss���>l��g        )��P	�����A�*

	conv_lossAY�>�ni        )��P	�����A�*

	conv_loss	ʅ>f6�[