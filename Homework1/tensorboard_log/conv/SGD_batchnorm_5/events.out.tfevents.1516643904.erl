       �K"	   ����Abrain.Event:2���i��     �g��	�7����A"��
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
dtype0*
_output_shapes
: * 
_class
loc:@conv2d/kernel*
valueB
 *���>
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
conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*&
_output_shapes
:*
T0
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
�
*batch_normalization/gamma/Initializer/onesConst*
_output_shapes
:*,
_class"
 loc:@batch_normalization/gamma*
valueB*  �?*
dtype0
�
batch_normalization/gamma
VariableV2*,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
 batch_normalization/gamma/AssignAssignbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:
�
batch_normalization/gamma/readIdentitybatch_normalization/gamma*
_output_shapes
:*
T0*,
_class"
 loc:@batch_normalization/gamma
�
*batch_normalization/beta/Initializer/zerosConst*+
_class!
loc:@batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization/beta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@batch_normalization/beta
�
batch_normalization/beta/AssignAssignbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:*
use_locking(
�
batch_normalization/beta/readIdentitybatch_normalization/beta*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:
�
1batch_normalization/moving_mean/Initializer/zerosConst*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container 
�
&batch_normalization/moving_mean/AssignAssignbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:
�
$batch_normalization/moving_mean/readIdentitybatch_normalization/moving_mean*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
4batch_normalization/moving_variance/Initializer/onesConst*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
#batch_normalization/moving_variance
VariableV2*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:
�
*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
_output_shapes
:*
use_locking(*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
validate_shape(
�
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
n
batch_normalization/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes

::
s
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
T0
*
_output_shapes
:
q
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
T0
*
_output_shapes
:
^
 batch_normalization/cond/pred_idIdentityPlaceholder_2*
_output_shapes
:*
T0

�
batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
�
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchconv2d/Conv2D batch_normalization/cond/pred_id*
T0* 
_class
loc:@conv2d/Conv2D*J
_output_shapes8
6:���������:���������
�
0batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
::
�
0batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id* 
_output_shapes
::*
T0*+
_class!
loc:@batch_normalization/beta
�
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:12batch_normalization/cond/FusedBatchNorm/Switch_2:1batch_normalization/cond/Const batch_normalization/cond/Const_1*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:*
T0
�
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchconv2d/Conv2D batch_normalization/cond/pred_id*
T0* 
_class
loc:@conv2d/Conv2D*J
_output_shapes8
6:���������:���������
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
::
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id* 
_output_shapes
::*
T0*+
_class!
loc:@batch_normalization/beta
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read batch_normalization/cond/pred_id*
T0*2
_class(
&$loc:@batch_normalization/moving_mean* 
_output_shapes
::
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read batch_normalization/cond/pred_id*
T0*6
_class,
*(loc:@batch_normalization/moving_variance* 
_output_shapes
::
�
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_22batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0
�
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
 batch_normalization/cond/Merge_1Merge+batch_normalization/cond/FusedBatchNorm_1:1)batch_normalization/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
_output_shapes

:: *
T0*
N
i
$batch_normalization/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
d
"batch_normalization/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
batch_normalization/ExpandDims
ExpandDims$batch_normalization/ExpandDims/input"batch_normalization/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
k
&batch_normalization/ExpandDims_1/inputConst*
_output_shapes
: *
valueB
 *    *
dtype0
f
$batch_normalization/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization/ExpandDims_1
ExpandDims&batch_normalization/ExpandDims_1/input$batch_normalization/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
k
!batch_normalization/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization/ReshapeReshapePlaceholder_2!batch_normalization/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization/SelectSelectbatch_normalization/Reshapebatch_normalization/ExpandDims batch_normalization/ExpandDims_1*
T0*
_output_shapes
:
z
batch_normalization/SqueezeSqueezebatch_normalization/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
(batch_normalization/AssignMovingAvg/readIdentitybatch_normalization/moving_mean*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
'batch_normalization/AssignMovingAvg/SubSub(batch_normalization/AssignMovingAvg/read batch_normalization/cond/Merge_1*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
'batch_normalization/AssignMovingAvg/MulMul'batch_normalization/AssignMovingAvg/Subbatch_normalization/Squeeze*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
#batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/Mul*
_output_shapes
:*
use_locking( *
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
*batch_normalization/AssignMovingAvg_1/readIdentity#batch_normalization/moving_variance*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
�
)batch_normalization/AssignMovingAvg_1/SubSub*batch_normalization/AssignMovingAvg_1/read batch_normalization/cond/Merge_2*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
)batch_normalization/AssignMovingAvg_1/MulMul)batch_normalization/AssignMovingAvg_1/Subbatch_normalization/Squeeze*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
%batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/Mul*
use_locking( *
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
f
ReluRelubatch_normalization/cond/Merge*
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
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_1/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
�
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_1/gamma
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container 
�
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
:
�
 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
�
,batch_normalization_1/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_1/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:
�
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
�
3batch_normalization_1/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_1/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_1/moving_mean
�
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:
�
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
6batch_normalization_1/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB*  �?
�
%batch_normalization_1/moving_variance
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container 
�
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
p
!batch_normalization_2/cond/SwitchSwitchPlaceholder_2Placeholder_2*
_output_shapes

::*
T0

w
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
_output_shapes
:*
T0

`
"batch_normalization_2/cond/pred_idIdentityPlaceholder_2*
T0
*
_output_shapes
:
�
 batch_normalization_2/cond/ConstConst$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
�
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*"
_class
loc:@conv2d_2/Conv2D
�
2batch_normalization_2/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::
�
2batch_normalization_2/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::
�
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:14batch_normalization_2/cond/FusedBatchNorm/Switch_2:1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(
�
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id*"
_class
loc:@conv2d_2/Conv2D*J
_output_shapes8
6:���������:���������*
T0
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::*
T0
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::*
T0
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_1/moving_mean/read"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
::
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_1/moving_variance/read"batch_normalization_2/cond/pred_id* 
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_24batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0
�
 batch_normalization_2/cond/MergeMerge+batch_normalization_2/cond/FusedBatchNorm_1)batch_normalization_2/cond/FusedBatchNorm*
N*1
_output_shapes
:���������: *
T0
�
"batch_normalization_2/cond/Merge_1Merge-batch_normalization_2/cond/FusedBatchNorm_1:1+batch_normalization_2/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_2/cond/Merge_2Merge-batch_normalization_2/cond/FusedBatchNorm_1:2+batch_normalization_2/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
k
&batch_normalization_2/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
f
$batch_normalization_2/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization_2/ExpandDims
ExpandDims&batch_normalization_2/ExpandDims/input$batch_normalization_2/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
m
(batch_normalization_2/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
&batch_normalization_2/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_2/ExpandDims_1
ExpandDims(batch_normalization_2/ExpandDims_1/input&batch_normalization_2/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
m
#batch_normalization_2/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
batch_normalization_2/ReshapeReshapePlaceholder_2#batch_normalization_2/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_2/SelectSelectbatch_normalization_2/Reshape batch_normalization_2/ExpandDims"batch_normalization_2/ExpandDims_1*
_output_shapes
:*
T0
~
batch_normalization_2/SqueezeSqueezebatch_normalization_2/Select*
T0*
_output_shapes
: *
squeeze_dims
 
�
*batch_normalization_2/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
)batch_normalization_2/AssignMovingAvg/SubSub*batch_normalization_2/AssignMovingAvg/read"batch_normalization_2/cond/Merge_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
)batch_normalization_2/AssignMovingAvg/MulMul)batch_normalization_2/AssignMovingAvg/Subbatch_normalization_2/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_2/AssignMovingAvg/Mul*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
,batch_normalization_2/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
�
+batch_normalization_2/AssignMovingAvg_1/SubSub,batch_normalization_2/AssignMovingAvg_1/read"batch_normalization_2/cond/Merge_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
+batch_normalization_2/AssignMovingAvg_1/MulMul+batch_normalization_2/AssignMovingAvg_1/Subbatch_normalization_2/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
�
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_2/AssignMovingAvg_1/Mul*
_output_shapes
:*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
j
Relu_1Relu batch_normalization_2/cond/Merge*
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
conv2d_3/Conv2DConv2DRelu_1conv2d_2/kernel/read*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
,batch_normalization_2/gamma/Initializer/onesConst*
_output_shapes
:*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  �?*
dtype0
�
batch_normalization_2/gamma
VariableV2*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:
�
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:
�
 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
,batch_normalization_2/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_2/beta
VariableV2*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_2/beta
�
3batch_normalization_2/moving_mean/Initializer/zerosConst*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    *
dtype0
�
!batch_normalization_2/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_2/moving_mean
�
(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
:
�
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
6batch_normalization_2/moving_variance/Initializer/onesConst*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB*  �?*
dtype0
�
%batch_normalization_2/moving_variance
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container 
�
,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes
:
�
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
p
!batch_normalization_3/cond/SwitchSwitchPlaceholder_2Placeholder_2*
_output_shapes

::*
T0

w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
:
`
"batch_normalization_3/cond/pred_idIdentityPlaceholder_2*
T0
*
_output_shapes
:
�
 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
�
0batch_normalization_3/cond/FusedBatchNorm/SwitchSwitchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id*
T0*"
_class
loc:@conv2d_3/Conv2D*J
_output_shapes8
6:���������:���������
�
2batch_normalization_3/cond/FusedBatchNorm/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::
�
2batch_normalization_3/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::
�
)batch_normalization_3/cond/FusedBatchNormFusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:14batch_normalization_3/cond/FusedBatchNorm/Switch_2:1 batch_normalization_3/cond/Const"batch_normalization_3/cond/Const_1*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:
�
2batch_normalization_3/cond/FusedBatchNorm_1/SwitchSwitchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id*
T0*"
_class
loc:@conv2d_3/Conv2D*J
_output_shapes8
6:���������:���������
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read"batch_normalization_3/cond/pred_id*4
_class*
(&loc:@batch_normalization_2/moving_mean* 
_output_shapes
::*
T0
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
::
�
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_24batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
_output_shapes

:: *
T0*
N
�
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
k
&batch_normalization_3/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
f
$batch_normalization_3/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
 batch_normalization_3/ExpandDims
ExpandDims&batch_normalization_3/ExpandDims/input$batch_normalization_3/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
m
(batch_normalization_3/ExpandDims_1/inputConst*
_output_shapes
: *
valueB
 *    *
dtype0
h
&batch_normalization_3/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_3/ExpandDims_1
ExpandDims(batch_normalization_3/ExpandDims_1/input&batch_normalization_3/ExpandDims_1/dim*
T0*
_output_shapes
:*

Tdim0
m
#batch_normalization_3/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_3/ReshapeReshapePlaceholder_2#batch_normalization_3/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_3/SelectSelectbatch_normalization_3/Reshape batch_normalization_3/ExpandDims"batch_normalization_3/ExpandDims_1*
T0*
_output_shapes
:
~
batch_normalization_3/SqueezeSqueezebatch_normalization_3/Select*
_output_shapes
: *
squeeze_dims
 *
T0
�
*batch_normalization_3/AssignMovingAvg/readIdentity!batch_normalization_2/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
)batch_normalization_3/AssignMovingAvg/SubSub*batch_normalization_3/AssignMovingAvg/read"batch_normalization_3/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
)batch_normalization_3/AssignMovingAvg/MulMul)batch_normalization_3/AssignMovingAvg/Subbatch_normalization_3/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
%batch_normalization_3/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean)batch_normalization_3/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
,batch_normalization_3/AssignMovingAvg_1/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
+batch_normalization_3/AssignMovingAvg_1/SubSub,batch_normalization_3/AssignMovingAvg_1/read"batch_normalization_3/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
+batch_normalization_3/AssignMovingAvg_1/MulMul+batch_normalization_3/AssignMovingAvg_1/Subbatch_normalization_3/Squeeze*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0
�
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_3/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
j
Relu_2Relu batch_normalization_3/cond/Merge*
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
conv2d_4/Conv2DConv2DRelu_2conv2d_3/kernel/read*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
,batch_normalization_3/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@batch_normalization_3/gamma*
valueB*  �?
�
batch_normalization_3/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes
:
�
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:
�
,batch_normalization_3/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_3/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_3/beta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_3/beta
�
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_3/beta
�
3batch_normalization_3/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_3/moving_mean
VariableV2*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:
�
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
6batch_normalization_3/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB*  �?
�
%batch_normalization_3/moving_variance
VariableV2*8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:*
T0
p
!batch_normalization_4/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes

::
w
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
_output_shapes
:*
T0

`
"batch_normalization_4/cond/pred_idIdentityPlaceholder_2*
_output_shapes
:*
T0

�
 batch_normalization_4/cond/ConstConst$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/cond/Const_1Const$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
0batch_normalization_4/cond/FusedBatchNorm/SwitchSwitchconv2d_4/Conv2D"batch_normalization_4/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*"
_class
loc:@conv2d_4/Conv2D
�
2batch_normalization_4/cond/FusedBatchNorm/Switch_1Switch batch_normalization_3/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma* 
_output_shapes
::
�
2batch_normalization_4/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_3/beta/read"batch_normalization_4/cond/pred_id*-
_class#
!loc:@batch_normalization_3/beta* 
_output_shapes
::*
T0
�
)batch_normalization_4/cond/FusedBatchNormFusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm/Switch:14batch_normalization_4/cond/FusedBatchNorm/Switch_1:14batch_normalization_4/cond/FusedBatchNorm/Switch_2:1 batch_normalization_4/cond/Const"batch_normalization_4/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(
�
2batch_normalization_4/cond/FusedBatchNorm_1/SwitchSwitchconv2d_4/Conv2D"batch_normalization_4/cond/pred_id*
T0*"
_class
loc:@conv2d_4/Conv2D*J
_output_shapes8
6:���������:���������
�
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_3/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma* 
_output_shapes
::
�
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_3/beta/read"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta* 
_output_shapes
::
�
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_3/moving_mean/read"batch_normalization_4/cond/pred_id*4
_class*
(&loc:@batch_normalization_3/moving_mean* 
_output_shapes
::*
T0
�
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_3/moving_variance/read"batch_normalization_4/cond/pred_id* 
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
+batch_normalization_4/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm_1/Switch4batch_normalization_4/cond/FusedBatchNorm_1/Switch_14batch_normalization_4/cond/FusedBatchNorm_1/Switch_24batch_normalization_4/cond/FusedBatchNorm_1/Switch_34batch_normalization_4/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0
�
 batch_normalization_4/cond/MergeMerge+batch_normalization_4/cond/FusedBatchNorm_1)batch_normalization_4/cond/FusedBatchNorm*1
_output_shapes
:���������: *
T0*
N
�
"batch_normalization_4/cond/Merge_1Merge-batch_normalization_4/cond/FusedBatchNorm_1:1+batch_normalization_4/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_4/cond/Merge_2Merge-batch_normalization_4/cond/FusedBatchNorm_1:2+batch_normalization_4/cond/FusedBatchNorm:2*
_output_shapes

:: *
T0*
N
k
&batch_normalization_4/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
f
$batch_normalization_4/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization_4/ExpandDims
ExpandDims&batch_normalization_4/ExpandDims/input$batch_normalization_4/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
m
(batch_normalization_4/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
&batch_normalization_4/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_1
ExpandDims(batch_normalization_4/ExpandDims_1/input&batch_normalization_4/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
m
#batch_normalization_4/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_4/ReshapeReshapePlaceholder_2#batch_normalization_4/Reshape/shape*
_output_shapes
:*
T0
*
Tshape0
�
batch_normalization_4/SelectSelectbatch_normalization_4/Reshape batch_normalization_4/ExpandDims"batch_normalization_4/ExpandDims_1*
_output_shapes
:*
T0
~
batch_normalization_4/SqueezeSqueezebatch_normalization_4/Select*
T0*
_output_shapes
: *
squeeze_dims
 
�
*batch_normalization_4/AssignMovingAvg/readIdentity!batch_normalization_3/moving_mean*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
)batch_normalization_4/AssignMovingAvg/SubSub*batch_normalization_4/AssignMovingAvg/read"batch_normalization_4/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
)batch_normalization_4/AssignMovingAvg/MulMul)batch_normalization_4/AssignMovingAvg/Subbatch_normalization_4/Squeeze*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_4/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
,batch_normalization_4/AssignMovingAvg_1/readIdentity%batch_normalization_3/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
�
+batch_normalization_4/AssignMovingAvg_1/SubSub,batch_normalization_4/AssignMovingAvg_1/read"batch_normalization_4/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
�
+batch_normalization_4/AssignMovingAvg_1/MulMul+batch_normalization_4/AssignMovingAvg_1/Subbatch_normalization_4/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
�
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_4/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
j
Relu_3Relu batch_normalization_4/cond/Merge*
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
�
,batch_normalization_4/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@batch_normalization_4/gamma*
valueB*  �?
�
batch_normalization_4/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
"batch_normalization_4/gamma/AssignAssignbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
�
 batch_normalization_4/gamma/readIdentitybatch_normalization_4/gamma*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:
�
,batch_normalization_4/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_4/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_4/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:
�
!batch_normalization_4/beta/AssignAssignbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization_4/beta/readIdentitybatch_normalization_4/beta*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:
�
3batch_normalization_4/moving_mean/Initializer/zerosConst*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean*
valueB*    *
dtype0
�
!batch_normalization_4/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container 
�
(batch_normalization_4/moving_mean/AssignAssign!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_4/moving_mean*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
&batch_normalization_4/moving_mean/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
6batch_normalization_4/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB*  �?
�
%batch_normalization_4/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:
�
,batch_normalization_4/moving_variance/AssignAssign%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
*batch_normalization_4/moving_variance/readIdentity%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:*
T0
p
!batch_normalization_5/cond/SwitchSwitchPlaceholder_2Placeholder_2*
_output_shapes

::*
T0

w
#batch_normalization_5/cond/switch_tIdentity#batch_normalization_5/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_5/cond/switch_fIdentity!batch_normalization_5/cond/Switch*
_output_shapes
:*
T0

`
"batch_normalization_5/cond/pred_idIdentityPlaceholder_2*
T0
*
_output_shapes
:
�
 batch_normalization_5/cond/ConstConst$^batch_normalization_5/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
�
"batch_normalization_5/cond/Const_1Const$^batch_normalization_5/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
0batch_normalization_5/cond/FusedBatchNorm/SwitchSwitchconv2d_5/Conv2D"batch_normalization_5/cond/pred_id*
T0*"
_class
loc:@conv2d_5/Conv2D*J
_output_shapes8
6:���������:���������
�
2batch_normalization_5/cond/FusedBatchNorm/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma* 
_output_shapes
::
�
2batch_normalization_5/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_5/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
::
�
)batch_normalization_5/cond/FusedBatchNormFusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm/Switch:14batch_normalization_5/cond/FusedBatchNorm/Switch_1:14batch_normalization_5/cond/FusedBatchNorm/Switch_2:1 batch_normalization_5/cond/Const"batch_normalization_5/cond/Const_1*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:
�
2batch_normalization_5/cond/FusedBatchNorm_1/SwitchSwitchconv2d_5/Conv2D"batch_normalization_5/cond/pred_id*"
_class
loc:@conv2d_5/Conv2D*J
_output_shapes8
6:���������:���������*
T0
�
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_5/cond/pred_id* 
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_4/gamma
�
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_5/cond/pred_id* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_4/beta
�
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_4/moving_mean/read"batch_normalization_5/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean* 
_output_shapes
::
�
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_4/moving_variance/read"batch_normalization_5/cond/pred_id* 
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
+batch_normalization_5/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm_1/Switch4batch_normalization_5/cond/FusedBatchNorm_1/Switch_14batch_normalization_5/cond/FusedBatchNorm_1/Switch_24batch_normalization_5/cond/FusedBatchNorm_1/Switch_34batch_normalization_5/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0
�
 batch_normalization_5/cond/MergeMerge+batch_normalization_5/cond/FusedBatchNorm_1)batch_normalization_5/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_5/cond/Merge_1Merge-batch_normalization_5/cond/FusedBatchNorm_1:1+batch_normalization_5/cond/FusedBatchNorm:1*
N*
_output_shapes

:: *
T0
�
"batch_normalization_5/cond/Merge_2Merge-batch_normalization_5/cond/FusedBatchNorm_1:2+batch_normalization_5/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
k
&batch_normalization_5/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
f
$batch_normalization_5/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization_5/ExpandDims
ExpandDims&batch_normalization_5/ExpandDims/input$batch_normalization_5/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
m
(batch_normalization_5/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
&batch_normalization_5/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_5/ExpandDims_1
ExpandDims(batch_normalization_5/ExpandDims_1/input&batch_normalization_5/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
m
#batch_normalization_5/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_5/ReshapeReshapePlaceholder_2#batch_normalization_5/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_5/SelectSelectbatch_normalization_5/Reshape batch_normalization_5/ExpandDims"batch_normalization_5/ExpandDims_1*
_output_shapes
:*
T0
~
batch_normalization_5/SqueezeSqueezebatch_normalization_5/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
*batch_normalization_5/AssignMovingAvg/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
)batch_normalization_5/AssignMovingAvg/SubSub*batch_normalization_5/AssignMovingAvg/read"batch_normalization_5/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
)batch_normalization_5/AssignMovingAvg/MulMul)batch_normalization_5/AssignMovingAvg/Subbatch_normalization_5/Squeeze*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
%batch_normalization_5/AssignMovingAvg	AssignSub!batch_normalization_4/moving_mean)batch_normalization_5/AssignMovingAvg/Mul*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
,batch_normalization_5/AssignMovingAvg_1/readIdentity%batch_normalization_4/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:
�
+batch_normalization_5/AssignMovingAvg_1/SubSub,batch_normalization_5/AssignMovingAvg_1/read"batch_normalization_5/cond/Merge_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
+batch_normalization_5/AssignMovingAvg_1/MulMul+batch_normalization_5/AssignMovingAvg_1/Subbatch_normalization_5/Squeeze*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
'batch_normalization_5/AssignMovingAvg_1	AssignSub%batch_normalization_4/moving_variance+batch_normalization_5/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:
j
Relu_4Relu batch_normalization_5/cond/Merge*/
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
ReshapeReshapeRelu_4Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
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
dtype0*
_output_shapes
:	�d*

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
dtype0*
_output_shapes
:	�d*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	�d
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
�
,batch_normalization_5/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_5/gamma*
valueBd*  �?*
dtype0*
_output_shapes
:d
�
batch_normalization_5/gamma
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@batch_normalization_5/gamma*
	container 
�
"batch_normalization_5/gamma/AssignAssignbatch_normalization_5/gamma,batch_normalization_5/gamma/Initializer/ones*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes
:d*
use_locking(
�
 batch_normalization_5/gamma/readIdentitybatch_normalization_5/gamma*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:d
�
,batch_normalization_5/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_5/beta*
valueBd*    *
dtype0*
_output_shapes
:d
�
batch_normalization_5/beta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_5/beta*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
!batch_normalization_5/beta/AssignAssignbatch_normalization_5/beta,batch_normalization_5/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes
:d
�
batch_normalization_5/beta/readIdentitybatch_normalization_5/beta*
_output_shapes
:d*
T0*-
_class#
!loc:@batch_normalization_5/beta
�
3batch_normalization_5/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_5/moving_mean*
valueBd*    *
dtype0*
_output_shapes
:d
�
!batch_normalization_5/moving_mean
VariableV2*
shared_name *4
_class*
(&loc:@batch_normalization_5/moving_mean*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
(batch_normalization_5/moving_mean/AssignAssign!batch_normalization_5/moving_mean3batch_normalization_5/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes
:d
�
&batch_normalization_5/moving_mean/readIdentity!batch_normalization_5/moving_mean*
_output_shapes
:d*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
�
6batch_normalization_5/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_5/moving_variance*
valueBd*  �?*
dtype0*
_output_shapes
:d
�
%batch_normalization_5/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_5/moving_variance*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
,batch_normalization_5/moving_variance/AssignAssign%batch_normalization_5/moving_variance6batch_normalization_5/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
validate_shape(*
_output_shapes
:d
�
*batch_normalization_5/moving_variance/readIdentity%batch_normalization_5/moving_variance*
_output_shapes
:d*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
~
4batch_normalization_6/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_6/moments/meanMeandense/MatMul4batch_normalization_6/moments/mean/reduction_indices*
_output_shapes

:d*
	keep_dims(*

Tidx0*
T0
�
*batch_normalization_6/moments/StopGradientStopGradient"batch_normalization_6/moments/mean*
T0*
_output_shapes

:d
�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense/MatMul*batch_normalization_6/moments/StopGradient*
T0*'
_output_shapes
:���������d
�
8batch_normalization_6/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_6/moments/varianceMean/batch_normalization_6/moments/SquaredDifference8batch_normalization_6/moments/variance/reduction_indices*
T0*
_output_shapes

:d*
	keep_dims(*

Tidx0
�
%batch_normalization_6/moments/SqueezeSqueeze"batch_normalization_6/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:d
�
'batch_normalization_6/moments/Squeeze_1Squeeze&batch_normalization_6/moments/variance*
_output_shapes
:d*
squeeze_dims
 *
T0
f
$batch_normalization_6/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization_6/ExpandDims
ExpandDims%batch_normalization_6/moments/Squeeze$batch_normalization_6/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:d
h
&batch_normalization_6/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_6/ExpandDims_1
ExpandDims&batch_normalization_5/moving_mean/read&batch_normalization_6/ExpandDims_1/dim*
_output_shapes

:d*

Tdim0*
T0
m
#batch_normalization_6/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_6/ReshapeReshapePlaceholder_2#batch_normalization_6/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

�
batch_normalization_6/SelectSelectbatch_normalization_6/Reshape batch_normalization_6/ExpandDims"batch_normalization_6/ExpandDims_1*
T0*
_output_shapes

:d
�
batch_normalization_6/SqueezeSqueezebatch_normalization_6/Select*
squeeze_dims
 *
T0*
_output_shapes
:d
h
&batch_normalization_6/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_6/ExpandDims_2
ExpandDims'batch_normalization_6/moments/Squeeze_1&batch_normalization_6/ExpandDims_2/dim*
_output_shapes

:d*

Tdim0*
T0
h
&batch_normalization_6/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_6/ExpandDims_3
ExpandDims*batch_normalization_5/moving_variance/read&batch_normalization_6/ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes

:d
o
%batch_normalization_6/Reshape_1/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_6/Reshape_1ReshapePlaceholder_2%batch_normalization_6/Reshape_1/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_6/Select_1Selectbatch_normalization_6/Reshape_1"batch_normalization_6/ExpandDims_2"batch_normalization_6/ExpandDims_3*
_output_shapes

:d*
T0
�
batch_normalization_6/Squeeze_1Squeezebatch_normalization_6/Select_1*
T0*
_output_shapes
:d*
squeeze_dims
 
m
(batch_normalization_6/ExpandDims_4/inputConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
h
&batch_normalization_6/ExpandDims_4/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_6/ExpandDims_4
ExpandDims(batch_normalization_6/ExpandDims_4/input&batch_normalization_6/ExpandDims_4/dim*
_output_shapes
:*

Tdim0*
T0
m
(batch_normalization_6/ExpandDims_5/inputConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
h
&batch_normalization_6/ExpandDims_5/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_6/ExpandDims_5
ExpandDims(batch_normalization_6/ExpandDims_5/input&batch_normalization_6/ExpandDims_5/dim*
_output_shapes
:*

Tdim0*
T0
o
%batch_normalization_6/Reshape_2/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_6/Reshape_2ReshapePlaceholder_2%batch_normalization_6/Reshape_2/shape*
_output_shapes
:*
T0
*
Tshape0
�
batch_normalization_6/Select_2Selectbatch_normalization_6/Reshape_2"batch_normalization_6/ExpandDims_4"batch_normalization_6/ExpandDims_5*
_output_shapes
:*
T0
�
batch_normalization_6/Squeeze_2Squeezebatch_normalization_6/Select_2*
squeeze_dims
 *
T0*
_output_shapes
: 
�
+batch_normalization_6/AssignMovingAvg/sub/xConst*
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0*
_output_shapes
: 
�
)batch_normalization_6/AssignMovingAvg/subSub+batch_normalization_6/AssignMovingAvg/sub/xbatch_normalization_6/Squeeze_2*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
: 
�
+batch_normalization_6/AssignMovingAvg/sub_1Sub&batch_normalization_5/moving_mean/readbatch_normalization_6/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:d
�
)batch_normalization_6/AssignMovingAvg/mulMul+batch_normalization_6/AssignMovingAvg/sub_1)batch_normalization_6/AssignMovingAvg/sub*
_output_shapes
:d*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
�
%batch_normalization_6/AssignMovingAvg	AssignSub!batch_normalization_5/moving_mean)batch_normalization_6/AssignMovingAvg/mul*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:d*
use_locking( 
�
-batch_normalization_6/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0*
_output_shapes
: 
�
+batch_normalization_6/AssignMovingAvg_1/subSub-batch_normalization_6/AssignMovingAvg_1/sub/xbatch_normalization_6/Squeeze_2*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
�
-batch_normalization_6/AssignMovingAvg_1/sub_1Sub*batch_normalization_5/moving_variance/readbatch_normalization_6/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:d
�
+batch_normalization_6/AssignMovingAvg_1/mulMul-batch_normalization_6/AssignMovingAvg_1/sub_1+batch_normalization_6/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:d
�
'batch_normalization_6/AssignMovingAvg_1	AssignSub%batch_normalization_5/moving_variance+batch_normalization_6/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:d
j
%batch_normalization_6/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_6/batchnorm/addAddbatch_normalization_6/Squeeze_1%batch_normalization_6/batchnorm/add/y*
T0*
_output_shapes
:d
x
%batch_normalization_6/batchnorm/RsqrtRsqrt#batch_normalization_6/batchnorm/add*
T0*
_output_shapes
:d
�
#batch_normalization_6/batchnorm/mulMul%batch_normalization_6/batchnorm/Rsqrt batch_normalization_5/gamma/read*
T0*
_output_shapes
:d
�
%batch_normalization_6/batchnorm/mul_1Muldense/MatMul#batch_normalization_6/batchnorm/mul*'
_output_shapes
:���������d*
T0
�
%batch_normalization_6/batchnorm/mul_2Mulbatch_normalization_6/Squeeze#batch_normalization_6/batchnorm/mul*
_output_shapes
:d*
T0
�
#batch_normalization_6/batchnorm/subSubbatch_normalization_5/beta/read%batch_normalization_6/batchnorm/mul_2*
_output_shapes
:d*
T0
�
%batch_normalization_6/batchnorm/add_1Add%batch_normalization_6/batchnorm/mul_1#batch_normalization_6/batchnorm/sub*
T0*'
_output_shapes
:���������d
g
Relu_5Relu%batch_normalization_6/batchnorm/add_1*'
_output_shapes
:���������d*
T0
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
VariableV2*
shape
:d
*
dtype0*
_output_shapes

:d
*
shared_name *!
_class
loc:@dense_1/kernel*
	container 
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
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
:
*
T0
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
	conv_lossScalarSummaryconv_loss/tagsMean*
_output_shapes
: *
T0
�
gradients/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
gradients/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
!gradients/Mean_grad/Reshape/shapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB"      
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
�
gradients/Mean_grad/ShapeShapelogistic_loss$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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

�
gradients/Mean_grad/Shape_1Shapelogistic_loss$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/Shape_2Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
gradients/Mean_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
gradients/Mean_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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

�
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0*
out_type0
�
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
�
-gradients/logistic_loss_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
�
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:���������
*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape
�
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:���������

�
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
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
�
(gradients/logistic_loss/Log1p_grad/add/xConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_18^gradients/logistic_loss_grad/tuple/control_dependency_1*
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
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:���������
*
T0
�
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikedense_2/BiasAdd$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*'
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
�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
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
�
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
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

�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
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
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
_output_shapes
:
*
T0
�
/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1^gradients/AddN+^gradients/dense_2/BiasAdd_grad/BiasAddGrad
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
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_57gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:d
*
transpose_a(*
transpose_b( 
�
.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1
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
gradients/Relu_5_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_5*'
_output_shapes
:���������d*
T0
�
:gradients/batch_normalization_6/batchnorm/add_1_grad/ShapeShape%batch_normalization_6/batchnorm/mul_1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/add_1_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB:d
�
Jgradients/batch_normalization_6/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_6/batchnorm/add_1_grad/Shape<gradients/batch_normalization_6/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/batch_normalization_6/batchnorm/add_1_grad/SumSumgradients/Relu_5_grad/ReluGradJgradients/batch_normalization_6/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<gradients/batch_normalization_6/batchnorm/add_1_grad/ReshapeReshape8gradients/batch_normalization_6/batchnorm/add_1_grad/Sum:gradients/batch_normalization_6/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_6/batchnorm/add_1_grad/Sum_1Sumgradients/Relu_5_grad/ReluGradLgradients/batch_normalization_6/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape_1Reshape:gradients/batch_normalization_6/batchnorm/add_1_grad/Sum_1<gradients/batch_normalization_6/batchnorm/add_1_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
�
Egradients/batch_normalization_6/batchnorm/add_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1=^gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape?^gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape_1
�
Mgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_6/batchnorm/add_1_grad/ReshapeF^gradients/batch_normalization_6/batchnorm/add_1_grad/tuple/group_deps*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������d*
T0
�
Ogradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape_1F^gradients/batch_normalization_6/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes
:d*
T0*Q
_classG
ECloc:@gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape_1
�
:gradients/batch_normalization_6/batchnorm/mul_1_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_6/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape<gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients/batch_normalization_6/batchnorm/mul_1_grad/mulMulMgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency#batch_normalization_6/batchnorm/mul*
T0*'
_output_shapes
:���������d
�
8gradients/batch_normalization_6/batchnorm/mul_1_grad/SumSum8gradients/batch_normalization_6/batchnorm/mul_1_grad/mulJgradients/batch_normalization_6/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<gradients/batch_normalization_6/batchnorm/mul_1_grad/ReshapeReshape8gradients/batch_normalization_6/batchnorm/mul_1_grad/Sum:gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape*'
_output_shapes
:���������d*
T0*
Tshape0
�
:gradients/batch_normalization_6/batchnorm/mul_1_grad/mul_1Muldense/MatMulMgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_6/batchnorm/mul_1_grad/Sum_1Sum:gradients/batch_normalization_6/batchnorm/mul_1_grad/mul_1Lgradients/batch_normalization_6/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1Reshape:gradients/batch_normalization_6/batchnorm/mul_1_grad/Sum_1<gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Egradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1=^gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape?^gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1
�
Mgradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_6/batchnorm/mul_1_grad/ReshapeF^gradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������d
�
Ogradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1F^gradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/sub_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_6/batchnorm/sub_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Hgradients/batch_normalization_6/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_6/batchnorm/sub_grad/Shape:gradients/batch_normalization_6/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
6gradients/batch_normalization_6/batchnorm/sub_grad/SumSumOgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency_1Hgradients/batch_normalization_6/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:gradients/batch_normalization_6/batchnorm/sub_grad/ReshapeReshape6gradients/batch_normalization_6/batchnorm/sub_grad/Sum8gradients/batch_normalization_6/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/sub_grad/Sum_1SumOgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency_1Jgradients/batch_normalization_6/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6gradients/batch_normalization_6/batchnorm/sub_grad/NegNeg8gradients/batch_normalization_6/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/sub_grad/Reshape_1Reshape6gradients/batch_normalization_6/batchnorm/sub_grad/Neg:gradients/batch_normalization_6/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Cgradients/batch_normalization_6/batchnorm/sub_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1;^gradients/batch_normalization_6/batchnorm/sub_grad/Reshape=^gradients/batch_normalization_6/batchnorm/sub_grad/Reshape_1
�
Kgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_6/batchnorm/sub_grad/ReshapeD^gradients/batch_normalization_6/batchnorm/sub_grad/tuple/group_deps*M
_classC
A?loc:@gradients/batch_normalization_6/batchnorm/sub_grad/Reshape*
_output_shapes
:d*
T0
�
Mgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_6/batchnorm/sub_grad/Reshape_1D^gradients/batch_normalization_6/batchnorm/sub_grad/tuple/group_deps*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/sub_grad/Reshape_1*
_output_shapes
:d*
T0
�
:gradients/batch_normalization_6/batchnorm/mul_2_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB:d
�
<gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
Jgradients/batch_normalization_6/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape<gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/batch_normalization_6/batchnorm/mul_2_grad/mulMulMgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependency_1#batch_normalization_6/batchnorm/mul*
_output_shapes
:d*
T0
�
8gradients/batch_normalization_6/batchnorm/mul_2_grad/SumSum8gradients/batch_normalization_6/batchnorm/mul_2_grad/mulJgradients/batch_normalization_6/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<gradients/batch_normalization_6/batchnorm/mul_2_grad/ReshapeReshape8gradients/batch_normalization_6/batchnorm/mul_2_grad/Sum:gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
:gradients/batch_normalization_6/batchnorm/mul_2_grad/mul_1Mulbatch_normalization_6/SqueezeMgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependency_1*
T0*
_output_shapes
:d
�
:gradients/batch_normalization_6/batchnorm/mul_2_grad/Sum_1Sum:gradients/batch_normalization_6/batchnorm/mul_2_grad/mul_1Lgradients/batch_normalization_6/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape_1Reshape:gradients/batch_normalization_6/batchnorm/mul_2_grad/Sum_1<gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
�
Egradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1=^gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape?^gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape_1
�
Mgradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_6/batchnorm/mul_2_grad/ReshapeF^gradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/group_deps*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape*
_output_shapes
:d*
T0
�
Ogradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape_1F^gradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape_1*
_output_shapes
:d
�
2gradients/batch_normalization_6/Squeeze_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
4gradients/batch_normalization_6/Squeeze_grad/ReshapeReshapeMgradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/control_dependency2gradients/batch_normalization_6/Squeeze_grad/Shape*
_output_shapes

:d*
T0*
Tshape0
�
gradients/AddN_1AddNOgradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/control_dependency_1Ogradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/mul_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_6/batchnorm/mul_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Hgradients/batch_normalization_6/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_6/batchnorm/mul_grad/Shape:gradients/batch_normalization_6/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
6gradients/batch_normalization_6/batchnorm/mul_grad/mulMulgradients/AddN_1 batch_normalization_5/gamma/read*
T0*
_output_shapes
:d
�
6gradients/batch_normalization_6/batchnorm/mul_grad/SumSum6gradients/batch_normalization_6/batchnorm/mul_grad/mulHgradients/batch_normalization_6/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:gradients/batch_normalization_6/batchnorm/mul_grad/ReshapeReshape6gradients/batch_normalization_6/batchnorm/mul_grad/Sum8gradients/batch_normalization_6/batchnorm/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/mul_grad/mul_1Mul%batch_normalization_6/batchnorm/Rsqrtgradients/AddN_1*
T0*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/mul_grad/Sum_1Sum8gradients/batch_normalization_6/batchnorm/mul_grad/mul_1Jgradients/batch_normalization_6/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/mul_grad/Reshape_1Reshape8gradients/batch_normalization_6/batchnorm/mul_grad/Sum_1:gradients/batch_normalization_6/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Cgradients/batch_normalization_6/batchnorm/mul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1;^gradients/batch_normalization_6/batchnorm/mul_grad/Reshape=^gradients/batch_normalization_6/batchnorm/mul_grad/Reshape_1
�
Kgradients/batch_normalization_6/batchnorm/mul_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_6/batchnorm/mul_grad/ReshapeD^gradients/batch_normalization_6/batchnorm/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/batch_normalization_6/batchnorm/mul_grad/Reshape*
_output_shapes
:d
�
Mgradients/batch_normalization_6/batchnorm/mul_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_6/batchnorm/mul_grad/Reshape_1D^gradients/batch_normalization_6/batchnorm/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/mul_grad/Reshape_1*
_output_shapes
:d
�
6gradients/batch_normalization_6/Select_grad/zeros_likeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueBd*    *
dtype0*
_output_shapes

:d
�
2gradients/batch_normalization_6/Select_grad/SelectSelectbatch_normalization_6/Reshape4gradients/batch_normalization_6/Squeeze_grad/Reshape6gradients/batch_normalization_6/Select_grad/zeros_like*
T0*
_output_shapes

:d
�
4gradients/batch_normalization_6/Select_grad/Select_1Selectbatch_normalization_6/Reshape6gradients/batch_normalization_6/Select_grad/zeros_like4gradients/batch_normalization_6/Squeeze_grad/Reshape*
T0*
_output_shapes

:d
�
<gradients/batch_normalization_6/Select_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/batch_normalization_6/Select_grad/Select5^gradients/batch_normalization_6/Select_grad/Select_1
�
Dgradients/batch_normalization_6/Select_grad/tuple/control_dependencyIdentity2gradients/batch_normalization_6/Select_grad/Select=^gradients/batch_normalization_6/Select_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/batch_normalization_6/Select_grad/Select*
_output_shapes

:d
�
Fgradients/batch_normalization_6/Select_grad/tuple/control_dependency_1Identity4gradients/batch_normalization_6/Select_grad/Select_1=^gradients/batch_normalization_6/Select_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/batch_normalization_6/Select_grad/Select_1*
_output_shapes

:d
�
>gradients/batch_normalization_6/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_6/batchnorm/RsqrtKgradients/batch_normalization_6/batchnorm/mul_grad/tuple/control_dependency*
T0*
_output_shapes
:d
�
5gradients/batch_normalization_6/ExpandDims_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
7gradients/batch_normalization_6/ExpandDims_grad/ReshapeReshapeDgradients/batch_normalization_6/Select_grad/tuple/control_dependency5gradients/batch_normalization_6/ExpandDims_grad/Shape*
Tshape0*
_output_shapes
:d*
T0
�
8gradients/batch_normalization_6/batchnorm/add_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_6/batchnorm/add_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
Hgradients/batch_normalization_6/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_6/batchnorm/add_grad/Shape:gradients/batch_normalization_6/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients/batch_normalization_6/batchnorm/add_grad/SumSum>gradients/batch_normalization_6/batchnorm/Rsqrt_grad/RsqrtGradHgradients/batch_normalization_6/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:gradients/batch_normalization_6/batchnorm/add_grad/ReshapeReshape6gradients/batch_normalization_6/batchnorm/add_grad/Sum8gradients/batch_normalization_6/batchnorm/add_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
8gradients/batch_normalization_6/batchnorm/add_grad/Sum_1Sum>gradients/batch_normalization_6/batchnorm/Rsqrt_grad/RsqrtGradJgradients/batch_normalization_6/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<gradients/batch_normalization_6/batchnorm/add_grad/Reshape_1Reshape8gradients/batch_normalization_6/batchnorm/add_grad/Sum_1:gradients/batch_normalization_6/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Cgradients/batch_normalization_6/batchnorm/add_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1;^gradients/batch_normalization_6/batchnorm/add_grad/Reshape=^gradients/batch_normalization_6/batchnorm/add_grad/Reshape_1
�
Kgradients/batch_normalization_6/batchnorm/add_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_6/batchnorm/add_grad/ReshapeD^gradients/batch_normalization_6/batchnorm/add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/batch_normalization_6/batchnorm/add_grad/Reshape*
_output_shapes
:d
�
Mgradients/batch_normalization_6/batchnorm/add_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_6/batchnorm/add_grad/Reshape_1D^gradients/batch_normalization_6/batchnorm/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
:gradients/batch_normalization_6/moments/Squeeze_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
<gradients/batch_normalization_6/moments/Squeeze_grad/ReshapeReshape7gradients/batch_normalization_6/ExpandDims_grad/Reshape:gradients/batch_normalization_6/moments/Squeeze_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
4gradients/batch_normalization_6/Squeeze_1_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
6gradients/batch_normalization_6/Squeeze_1_grad/ReshapeReshapeKgradients/batch_normalization_6/batchnorm/add_grad/tuple/control_dependency4gradients/batch_normalization_6/Squeeze_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
8gradients/batch_normalization_6/Select_1_grad/zeros_likeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes

:d*
valueBd*    
�
4gradients/batch_normalization_6/Select_1_grad/SelectSelectbatch_normalization_6/Reshape_16gradients/batch_normalization_6/Squeeze_1_grad/Reshape8gradients/batch_normalization_6/Select_1_grad/zeros_like*
T0*
_output_shapes

:d
�
6gradients/batch_normalization_6/Select_1_grad/Select_1Selectbatch_normalization_6/Reshape_18gradients/batch_normalization_6/Select_1_grad/zeros_like6gradients/batch_normalization_6/Squeeze_1_grad/Reshape*
T0*
_output_shapes

:d
�
>gradients/batch_normalization_6/Select_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_15^gradients/batch_normalization_6/Select_1_grad/Select7^gradients/batch_normalization_6/Select_1_grad/Select_1
�
Fgradients/batch_normalization_6/Select_1_grad/tuple/control_dependencyIdentity4gradients/batch_normalization_6/Select_1_grad/Select?^gradients/batch_normalization_6/Select_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/batch_normalization_6/Select_1_grad/Select*
_output_shapes

:d
�
Hgradients/batch_normalization_6/Select_1_grad/tuple/control_dependency_1Identity6gradients/batch_normalization_6/Select_1_grad/Select_1?^gradients/batch_normalization_6/Select_1_grad/tuple/group_deps*
_output_shapes

:d*
T0*I
_class?
=;loc:@gradients/batch_normalization_6/Select_1_grad/Select_1
�
7gradients/batch_normalization_6/ExpandDims_2_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
9gradients/batch_normalization_6/ExpandDims_2_grad/ReshapeReshapeFgradients/batch_normalization_6/Select_1_grad/tuple/control_dependency7gradients/batch_normalization_6/ExpandDims_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
<gradients/batch_normalization_6/moments/Squeeze_1_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
valueB"   d   *
dtype0
�
>gradients/batch_normalization_6/moments/Squeeze_1_grad/ReshapeReshape9gradients/batch_normalization_6/ExpandDims_2_grad/Reshape<gradients/batch_normalization_6/moments/Squeeze_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
;gradients/batch_normalization_6/moments/variance_grad/ShapeShape/batch_normalization_6/moments/SquaredDifference$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
:gradients/batch_normalization_6/moments/variance_grad/SizeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
9gradients/batch_normalization_6/moments/variance_grad/addAdd8batch_normalization_6/moments/variance/reduction_indices:gradients/batch_normalization_6/moments/variance_grad/Size*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_6/moments/variance_grad/modFloorMod9gradients/batch_normalization_6/moments/variance_grad/add:gradients/batch_normalization_6/moments/variance_grad/Size*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
_output_shapes
:
�
=gradients/batch_normalization_6/moments/variance_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
:
�
Agradients/batch_normalization_6/moments/variance_grad/range/startConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B : *N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Agradients/batch_normalization_6/moments/variance_grad/range/deltaConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
;gradients/batch_normalization_6/moments/variance_grad/rangeRangeAgradients/batch_normalization_6/moments/variance_grad/range/start:gradients/batch_normalization_6/moments/variance_grad/SizeAgradients/batch_normalization_6/moments/variance_grad/range/delta*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
_output_shapes
:*

Tidx0
�
@gradients/batch_normalization_6/moments/variance_grad/Fill/valueConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
:gradients/batch_normalization_6/moments/variance_grad/FillFill=gradients/batch_normalization_6/moments/variance_grad/Shape_1@gradients/batch_normalization_6/moments/variance_grad/Fill/value*
_output_shapes
:*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape
�
Cgradients/batch_normalization_6/moments/variance_grad/DynamicStitchDynamicStitch;gradients/batch_normalization_6/moments/variance_grad/range9gradients/batch_normalization_6/moments/variance_grad/mod;gradients/batch_normalization_6/moments/variance_grad/Shape:gradients/batch_normalization_6/moments/variance_grad/Fill*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
N*#
_output_shapes
:���������*
T0
�
?gradients/batch_normalization_6/moments/variance_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
=gradients/batch_normalization_6/moments/variance_grad/MaximumMaximumCgradients/batch_normalization_6/moments/variance_grad/DynamicStitch?gradients/batch_normalization_6/moments/variance_grad/Maximum/y*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*#
_output_shapes
:���������
�
>gradients/batch_normalization_6/moments/variance_grad/floordivFloorDiv;gradients/batch_normalization_6/moments/variance_grad/Shape=gradients/batch_normalization_6/moments/variance_grad/Maximum*
_output_shapes
:*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape
�
=gradients/batch_normalization_6/moments/variance_grad/ReshapeReshape>gradients/batch_normalization_6/moments/Squeeze_1_grad/ReshapeCgradients/batch_normalization_6/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
:gradients/batch_normalization_6/moments/variance_grad/TileTile=gradients/batch_normalization_6/moments/variance_grad/Reshape>gradients/batch_normalization_6/moments/variance_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
=gradients/batch_normalization_6/moments/variance_grad/Shape_2Shape/batch_normalization_6/moments/SquaredDifference$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
=gradients/batch_normalization_6/moments/variance_grad/Shape_3Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB"   d   
�
;gradients/batch_normalization_6/moments/variance_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB: *P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_6/moments/variance_grad/ProdProd=gradients/batch_normalization_6/moments/variance_grad/Shape_2;gradients/batch_normalization_6/moments/variance_grad/Const*
	keep_dims( *

Tidx0*
T0*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
_output_shapes
: 
�
=gradients/batch_normalization_6/moments/variance_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB: *P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
<gradients/batch_normalization_6/moments/variance_grad/Prod_1Prod=gradients/batch_normalization_6/moments/variance_grad/Shape_3=gradients/batch_normalization_6/moments/variance_grad/Const_1*
	keep_dims( *

Tidx0*
T0*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Agradients/batch_normalization_6/moments/variance_grad/Maximum_1/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
�
?gradients/batch_normalization_6/moments/variance_grad/Maximum_1Maximum<gradients/batch_normalization_6/moments/variance_grad/Prod_1Agradients/batch_normalization_6/moments/variance_grad/Maximum_1/y*
T0*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
_output_shapes
: 
�
@gradients/batch_normalization_6/moments/variance_grad/floordiv_1FloorDiv:gradients/batch_normalization_6/moments/variance_grad/Prod?gradients/batch_normalization_6/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2
�
:gradients/batch_normalization_6/moments/variance_grad/CastCast@gradients/batch_normalization_6/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
=gradients/batch_normalization_6/moments/variance_grad/truedivRealDiv:gradients/batch_normalization_6/moments/variance_grad/Tile:gradients/batch_normalization_6/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_6/moments/SquaredDifference_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
Fgradients/batch_normalization_6/moments/SquaredDifference_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
Tgradients/batch_normalization_6/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/batch_normalization_6/moments/SquaredDifference_grad/ShapeFgradients/batch_normalization_6/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Egradients/batch_normalization_6/moments/SquaredDifference_grad/scalarConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1>^gradients/batch_normalization_6/moments/variance_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
�
Bgradients/batch_normalization_6/moments/SquaredDifference_grad/mulMulEgradients/batch_normalization_6/moments/SquaredDifference_grad/scalar=gradients/batch_normalization_6/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������d
�
Bgradients/batch_normalization_6/moments/SquaredDifference_grad/subSubdense/MatMul*batch_normalization_6/moments/StopGradient$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1>^gradients/batch_normalization_6/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_6/moments/SquaredDifference_grad/mul_1MulBgradients/batch_normalization_6/moments/SquaredDifference_grad/mulBgradients/batch_normalization_6/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������d
�
Bgradients/batch_normalization_6/moments/SquaredDifference_grad/SumSumDgradients/batch_normalization_6/moments/SquaredDifference_grad/mul_1Tgradients/batch_normalization_6/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Fgradients/batch_normalization_6/moments/SquaredDifference_grad/ReshapeReshapeBgradients/batch_normalization_6/moments/SquaredDifference_grad/SumDgradients/batch_normalization_6/moments/SquaredDifference_grad/Shape*'
_output_shapes
:���������d*
T0*
Tshape0
�
Dgradients/batch_normalization_6/moments/SquaredDifference_grad/Sum_1SumDgradients/batch_normalization_6/moments/SquaredDifference_grad/mul_1Vgradients/batch_normalization_6/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Hgradients/batch_normalization_6/moments/SquaredDifference_grad/Reshape_1ReshapeDgradients/batch_normalization_6/moments/SquaredDifference_grad/Sum_1Fgradients/batch_normalization_6/moments/SquaredDifference_grad/Shape_1*
_output_shapes

:d*
T0*
Tshape0
�
Bgradients/batch_normalization_6/moments/SquaredDifference_grad/NegNegHgradients/batch_normalization_6/moments/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes

:d
�
Ogradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1G^gradients/batch_normalization_6/moments/SquaredDifference_grad/ReshapeC^gradients/batch_normalization_6/moments/SquaredDifference_grad/Neg
�
Wgradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/control_dependencyIdentityFgradients/batch_normalization_6/moments/SquaredDifference_grad/ReshapeP^gradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/batch_normalization_6/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������d
�
Ygradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityBgradients/batch_normalization_6/moments/SquaredDifference_grad/NegP^gradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/batch_normalization_6/moments/SquaredDifference_grad/Neg*
_output_shapes

:d
�
7gradients/batch_normalization_6/moments/mean_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
6gradients/batch_normalization_6/moments/mean_grad/SizeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
value	B :*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0
�
5gradients/batch_normalization_6/moments/mean_grad/addAdd4batch_normalization_6/moments/mean/reduction_indices6gradients/batch_normalization_6/moments/mean_grad/Size*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
_output_shapes
:
�
5gradients/batch_normalization_6/moments/mean_grad/modFloorMod5gradients/batch_normalization_6/moments/mean_grad/add6gradients/batch_normalization_6/moments/mean_grad/Size*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_6/moments/mean_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
�
=gradients/batch_normalization_6/moments/mean_grad/range/startConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
value	B : *J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0
�
=gradients/batch_normalization_6/moments/mean_grad/range/deltaConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
value	B :*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape
�
7gradients/batch_normalization_6/moments/mean_grad/rangeRange=gradients/batch_normalization_6/moments/mean_grad/range/start6gradients/batch_normalization_6/moments/mean_grad/Size=gradients/batch_normalization_6/moments/mean_grad/range/delta*
_output_shapes
:*

Tidx0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape
�
<gradients/batch_normalization_6/moments/mean_grad/Fill/valueConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
value	B :*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape
�
6gradients/batch_normalization_6/moments/mean_grad/FillFill9gradients/batch_normalization_6/moments/mean_grad/Shape_1<gradients/batch_normalization_6/moments/mean_grad/Fill/value*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
_output_shapes
:
�
?gradients/batch_normalization_6/moments/mean_grad/DynamicStitchDynamicStitch7gradients/batch_normalization_6/moments/mean_grad/range5gradients/batch_normalization_6/moments/mean_grad/mod7gradients/batch_normalization_6/moments/mean_grad/Shape6gradients/batch_normalization_6/moments/mean_grad/Fill*#
_output_shapes
:���������*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
N
�
;gradients/batch_normalization_6/moments/mean_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
9gradients/batch_normalization_6/moments/mean_grad/MaximumMaximum?gradients/batch_normalization_6/moments/mean_grad/DynamicStitch;gradients/batch_normalization_6/moments/mean_grad/Maximum/y*#
_output_shapes
:���������*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape
�
:gradients/batch_normalization_6/moments/mean_grad/floordivFloorDiv7gradients/batch_normalization_6/moments/mean_grad/Shape9gradients/batch_normalization_6/moments/mean_grad/Maximum*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape
�
9gradients/batch_normalization_6/moments/mean_grad/ReshapeReshape<gradients/batch_normalization_6/moments/Squeeze_grad/Reshape?gradients/batch_normalization_6/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
6gradients/batch_normalization_6/moments/mean_grad/TileTile9gradients/batch_normalization_6/moments/mean_grad/Reshape:gradients/batch_normalization_6/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
9gradients/batch_normalization_6/moments/mean_grad/Shape_2Shapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
9gradients/batch_normalization_6/moments/mean_grad/Shape_3Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
7gradients/batch_normalization_6/moments/mean_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB: *L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
6gradients/batch_normalization_6/moments/mean_grad/ProdProd9gradients/batch_normalization_6/moments/mean_grad/Shape_27gradients/batch_normalization_6/moments/mean_grad/Const*
T0*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
�
9gradients/batch_normalization_6/moments/mean_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB: *L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
8gradients/batch_normalization_6/moments/mean_grad/Prod_1Prod9gradients/batch_normalization_6/moments/mean_grad/Shape_39gradients/batch_normalization_6/moments/mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
_output_shapes
: 
�
=gradients/batch_normalization_6/moments/mean_grad/Maximum_1/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
�
;gradients/batch_normalization_6/moments/mean_grad/Maximum_1Maximum8gradients/batch_normalization_6/moments/mean_grad/Prod_1=gradients/batch_normalization_6/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2
�
<gradients/batch_normalization_6/moments/mean_grad/floordiv_1FloorDiv6gradients/batch_normalization_6/moments/mean_grad/Prod;gradients/batch_normalization_6/moments/mean_grad/Maximum_1*
T0*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
_output_shapes
: 
�
6gradients/batch_normalization_6/moments/mean_grad/CastCast<gradients/batch_normalization_6/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
9gradients/batch_normalization_6/moments/mean_grad/truedivRealDiv6gradients/batch_normalization_6/moments/mean_grad/Tile6gradients/batch_normalization_6/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������d
�
gradients/AddN_2AddNMgradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/control_dependencyWgradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/control_dependency9gradients/batch_normalization_6/moments/mean_grad/truediv*
N*'
_output_shapes
:���������d*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/AddN_2dense/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/AddN_2*
_output_shapes
:	�d*
transpose_a(*
transpose_b( *
T0
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
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
:	�d*
T0
�
gradients/Reshape_grad/ShapeShapeRelu_4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*/
_output_shapes
:���������*
T0*
Tshape0
�
gradients/Relu_4_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_4*/
_output_shapes
:���������*
T0
�
9gradients/batch_normalization_5/cond/Merge_grad/cond_gradSwitchgradients/Relu_4_grad/ReluGrad"batch_normalization_5/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_5/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^gradients/batch_normalization_5/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_5/cond/Merge_grad/cond_gradA^gradients/batch_normalization_5/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad
�
Jgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_5/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_5/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad
�
gradients/zeros_like	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_1	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_2	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_3	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency2batch_normalization_5/cond/FusedBatchNorm_1/Switch4batch_normalization_5/cond/FusedBatchNorm_1/Switch_14batch_normalization_5/cond/FusedBatchNorm_1/Switch_34batch_normalization_5/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1N^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������*
T0
�
Ugradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Ugradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
gradients/zeros_like_4	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_5	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_6	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_7	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency_12batch_normalization_5/cond/FusedBatchNorm/Switch:14batch_normalization_5/cond/FusedBatchNorm/Switch_1:1+batch_normalization_5/cond/FusedBatchNorm:3+batch_normalization_5/cond/FusedBatchNorm:4*
epsilon%o�:*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(
�
Igradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
gradients/SwitchSwitchconv2d_5/Conv2D"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
{
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*/
_output_shapes
:���������
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_2Shapegradients/Switch_1:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_1/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
j
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_2/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
_output_shapes

:: *
T0*
N
�
gradients/Switch_3Switchconv2d_5/Conv2D"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
c
gradients/Shape_4Shapegradients/Switch_3*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_3/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*/
_output_shapes
:���������*
T0
�
Igradients/batch_normalization_5/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_4Switch batch_normalization_4/gamma/read"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
c
gradients/Shape_5Shapegradients/Switch_4*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_4/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
j
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_5Switchbatch_normalization_4/beta/read"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
c
gradients/Shape_6Shapegradients/Switch_5*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_5/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
N*
_output_shapes

:: *
T0
�
gradients/AddN_3AddNKgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������
�
%gradients/conv2d_5/Conv2D_grad/ShapeNShapeNRelu_3conv2d_4/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
out_type0*
N* 
_output_shapes
::*
T0
�
2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_4/kernel/readgradients/AddN_3*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/AddN_3*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
/gradients/conv2d_5/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_4AddNMgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_1_grad/cond_grad*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:*
T0
�
gradients/AddN_5AddNMgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
gradients/Relu_3_grad/ReluGradReluGrad7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyRelu_3*
T0*/
_output_shapes
:���������
�
9gradients/batch_normalization_4/cond/Merge_grad/cond_gradSwitchgradients/Relu_3_grad/ReluGrad"batch_normalization_4/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_4/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^gradients/batch_normalization_4/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_4/cond/Merge_grad/cond_gradA^gradients/batch_normalization_4/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
�
Jgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_4/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad*/
_output_shapes
:���������
�
gradients/zeros_like_8	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_9	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_10	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_11	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency2batch_normalization_4/cond/FusedBatchNorm_1/Switch4batch_normalization_4/cond/FusedBatchNorm_1/Switch_14batch_normalization_4/cond/FusedBatchNorm_1/Switch_34batch_normalization_4/cond/FusedBatchNorm_1/Switch_4*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0*
data_formatNHWC
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1N^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Ugradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Ugradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
gradients/zeros_like_12	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_13	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_14	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_15	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency_12batch_normalization_4/cond/FusedBatchNorm/Switch:14batch_normalization_4/cond/FusedBatchNorm/Switch_1:1+batch_normalization_4/cond/FusedBatchNorm:3+batch_normalization_4/cond/FusedBatchNorm:4*
epsilon%o�:*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(
�
Igradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
gradients/Switch_6Switchconv2d_4/Conv2D"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
e
gradients/Shape_7Shapegradients/Switch_6:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_6/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*/
_output_shapes
:���������
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_7Switch batch_normalization_3/gamma/read"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_8Shapegradients/Switch_7:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_7/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_8Switchbatch_normalization_3/beta/read"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_9Shapegradients/Switch_8:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_8/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
N*
_output_shapes

:: *
T0
�
gradients/Switch_9Switchconv2d_4/Conv2D"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
d
gradients/Shape_10Shapegradients/Switch_9*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_9/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
�
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization_4/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_10Switch batch_normalization_3/gamma/read"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_11Shapegradients/Switch_10*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_10/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_11Switchbatch_normalization_3/beta/read"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_12Shapegradients/Switch_11*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_11/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_6AddNKgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_grad/cond_grad*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N
�
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNRelu_2conv2d_3/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
N* 
_output_shapes
::*
T0*
out_type0
�
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/AddN_6*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2'gradients/conv2d_4/Conv2D_grad/ShapeN:1gradients/AddN_6*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
/gradients/conv2d_4/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_7AddNMgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_8AddNMgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
gradients/Relu_2_grad/ReluGradReluGrad7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyRelu_2*/
_output_shapes
:���������*
T0
�
9gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitchgradients/Relu_2_grad/ReluGrad"batch_normalization_3/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_3/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^gradients/batch_normalization_3/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_3/cond/Merge_grad/cond_gradA^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:���������*
T0
�
Jgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:���������*
T0
�
gradients/zeros_like_16	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_17	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_18	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_19	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1N^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
gradients/zeros_like_20	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_21	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_22	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_23	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_12batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:1+batch_normalization_3/cond/FusedBatchNorm:3+batch_normalization_3/cond/FusedBatchNorm:4*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:*
T0*
data_formatNHWC
�
Igradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
gradients/Switch_12Switchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
g
gradients/Shape_13Shapegradients/Switch_12:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_12/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*/
_output_shapes
:���������
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_13Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
g
gradients/Shape_14Shapegradients/Switch_13:1*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_13/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_14Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_15Shapegradients/Switch_14:1*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_14/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_15Switchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
e
gradients/Shape_16Shapegradients/Switch_15*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_15/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_16Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_17Shapegradients/Switch_16*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_16/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
m
gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
N*
_output_shapes

:: *
T0
�
gradients/Switch_17Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_18Shapegradients/Switch_17*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_17/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
m
gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_9AddNKgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
out_type0*
N* 
_output_shapes
::*
T0
�
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/AddN_9*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/AddN_9*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
/gradients/conv2d_3/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_10AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_11AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
gradients/Relu_1_grad/ReluGradReluGrad7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyRelu_1*
T0*/
_output_shapes
:���������
�
9gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchgradients/Relu_1_grad/ReluGrad"batch_normalization_2/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_2/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^gradients/batch_normalization_2/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_2/cond/Merge_grad/cond_gradA^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
Jgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*/
_output_shapes
:���������
�
gradients/zeros_like_24	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_25	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_26	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_27	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0*
data_formatNHWC
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1N^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
gradients/zeros_like_28	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_29	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_30	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_31	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_12batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:1+batch_normalization_2/cond/FusedBatchNorm:3+batch_normalization_2/cond/FusedBatchNorm:4*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:*
T0
�
Igradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
gradients/Switch_18Switchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
g
gradients/Shape_19Shapegradients/Switch_18:1*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_18/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_18Fillgradients/Shape_19gradients/zeros_18/Const*/
_output_shapes
:���������*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_18*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_19Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_20Shapegradients/Switch_19:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_19/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_19Fillgradients/Shape_20gradients/zeros_19/Const*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_19*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_20Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_21Shapegradients/Switch_20:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_20/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
m
gradients/zeros_20Fillgradients/Shape_21gradients/zeros_20/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_20*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_21Switchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
e
gradients/Shape_22Shapegradients/Switch_21*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_21/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
�
gradients/zeros_21Fillgradients/Shape_22gradients/zeros_21/Const*/
_output_shapes
:���������*
T0
�
Igradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_21*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_22Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_23Shapegradients/Switch_22*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_22/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_22Fillgradients/Shape_23gradients/zeros_22/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_22*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_23Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_24Shapegradients/Switch_23*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_23/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
m
gradients/zeros_23Fillgradients/Shape_24gradients/zeros_23/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_23*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_12AddNKgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������
�
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0*
out_type0*
N
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/AddN_12*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/AddN_12*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
/gradients/conv2d_2/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_13AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_14AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
gradients/Relu_grad/ReluGradReluGrad7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:���������
�
7gradients/batch_normalization/cond/Merge_grad/cond_gradSwitchgradients/Relu_grad/ReluGrad batch_normalization/cond/pred_id*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
>gradients/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_18^gradients/batch_normalization/cond/Merge_grad/cond_grad
�
Fgradients/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentity7gradients/batch_normalization/cond/Merge_grad/cond_grad?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:���������
�
Hgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_1Identity9gradients/batch_normalization/cond/Merge_grad/cond_grad:1?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
�
gradients/zeros_like_32	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_33	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_34	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_35	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradFgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
Igradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityKgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradJ^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1J^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2J^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
gradients/zeros_like_36	ZerosLike)batch_normalization/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_37	ZerosLike)batch_normalization/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_38	ZerosLike)batch_normalization/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_39	ZerosLike)batch_normalization/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Igradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_10batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:1)batch_normalization/cond/FusedBatchNorm:3)batch_normalization/cond/FusedBatchNorm:4*
epsilon%o�:*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(
�
Ggradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1J^gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Ogradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityIgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradH^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
gradients/Switch_24Switchconv2d/Conv2D batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
g
gradients/Shape_25Shapegradients/Switch_24:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_24/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
�
gradients/zeros_24Fillgradients/Shape_25gradients/zeros_24/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_24*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_25Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_26Shapegradients/Switch_25:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_25/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_25Fillgradients/Shape_26gradients/zeros_25/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_25*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_26Switchbatch_normalization/beta/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
g
gradients/Shape_27Shapegradients/Switch_26:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_26/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_26Fillgradients/Shape_27gradients/zeros_26/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_26*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_27Switchconv2d/Conv2D batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
e
gradients/Shape_28Shapegradients/Switch_27*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_27/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_27Fillgradients/Shape_28gradients/zeros_27/Const*/
_output_shapes
:���������*
T0
�
Ggradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeOgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_27*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_28Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_29Shapegradients/Switch_28*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_28/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
m
gradients/zeros_28Fillgradients/Shape_29gradients/zeros_28/Const*
T0*
_output_shapes
:
�
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_28*
N*
_output_shapes

:: *
T0
�
gradients/Switch_29Switchbatch_normalization/beta/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_30Shapegradients/Switch_29*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_29/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_29Fillgradients/Shape_30gradients/zeros_29/Const*
_output_shapes
:*
T0
�
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_29*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_15AddNIgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradGgradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������*
T0
�
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
N* 
_output_shapes
::
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/AddN_15*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/AddN_15*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_11^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_16AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_17AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*
_output_shapes
:*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad
�
GradientDescent/learning_rateConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
EGradientDescent/update_batch_normalization/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization/gammaGradientDescent/learning_rategradients/AddN_16*
use_locking( *
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:
�
DGradientDescent/update_batch_normalization/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization/betaGradientDescent/learning_rategradients/AddN_17*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:*
use_locking( *
T0
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
GGradientDescent/update_batch_normalization_1/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_1/gammaGradientDescent/learning_rategradients/AddN_13*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
�
FGradientDescent/update_batch_normalization_1/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_1/betaGradientDescent/learning_rategradients/AddN_14*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:*
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
GGradientDescent/update_batch_normalization_2/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_2/gammaGradientDescent/learning_rategradients/AddN_10*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
FGradientDescent/update_batch_normalization_2/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_2/betaGradientDescent/learning_rategradients/AddN_11*
_output_shapes
:*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_2/beta
�
;GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentApplyGradientDescentconv2d_3/kernelGradientDescent/learning_rate9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
GGradientDescent/update_batch_normalization_3/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_3/gammaGradientDescent/learning_rategradients/AddN_7*
_output_shapes
:*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
FGradientDescent/update_batch_normalization_3/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_3/betaGradientDescent/learning_rategradients/AddN_8*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:
�
;GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentApplyGradientDescentconv2d_4/kernelGradientDescent/learning_rate9gradients/conv2d_5/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:*
use_locking( *
T0
�
GGradientDescent/update_batch_normalization_4/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_4/gammaGradientDescent/learning_rategradients/AddN_4*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:*
use_locking( 
�
FGradientDescent/update_batch_normalization_4/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_4/betaGradientDescent/learning_rategradients/AddN_5*
_output_shapes
:*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_4/beta
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d
�
GGradientDescent/update_batch_normalization_5/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_5/gammaGradientDescent/learning_rateMgradients/batch_normalization_6/batchnorm/mul_grad/tuple/control_dependency_1*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:d*
use_locking( *
T0
�
FGradientDescent/update_batch_normalization_5/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_5/betaGradientDescent/learning_rateKgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes
:d
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
�
GradientDescentNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^GradientDescent/update_conv2d/kernel/ApplyGradientDescentF^GradientDescent/update_batch_normalization/gamma/ApplyGradientDescentE^GradientDescent/update_batch_normalization/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_1/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_1/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_2/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_2/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_3/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_3/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_4/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_4/beta/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_5/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_5/beta/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
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
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_1MeanCastConst_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: " �xq:     �S�t	f<:����AJ��
�+�+
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
p
	AssignSub
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
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
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
�
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%��8"
data_formatstringNHWC"
is_trainingbool(
�
FusedBatchNormGrad

y_backprop"T
x"T

scale"T
reserve_space_1"T
reserve_space_2"T

x_backprop"T
scale_backprop"T
offset_backprop"T
reserve_space_3"T
reserve_space_4"T"
Ttype:
2"
epsilonfloat%��8"
data_formatstringNHWC"
is_trainingbool(
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
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
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
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
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
-
Rsqrt
x"T
y"T"
Ttype:	
2
:
	RsqrtGrad
y"T
dy"T
z"T"
Ttype:	
2
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
F
SquaredDifference
x"T
y"T
z"T"
Ttype:
	2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
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
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
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
Ttype*1.4.12v1.4.0-19-ga52c8d9��
~
PlaceholderPlaceholder*$
shape:���������*
dtype0*/
_output_shapes
:���������
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
conv2d/Conv2DConv2DPlaceholderconv2d/kernel/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������
�
*batch_normalization/gamma/Initializer/onesConst*,
_class"
 loc:@batch_normalization/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization/gamma
VariableV2*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
 batch_normalization/gamma/AssignAssignbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
batch_normalization/gamma/readIdentitybatch_normalization/gamma*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:
�
*batch_normalization/beta/Initializer/zerosConst*+
_class!
loc:@batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container *
shape:
�
batch_normalization/beta/AssignAssignbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization/beta/readIdentitybatch_normalization/beta*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:
�
1batch_normalization/moving_mean/Initializer/zerosConst*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container 
�
&batch_normalization/moving_mean/AssignAssignbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:
�
$batch_normalization/moving_mean/readIdentitybatch_normalization/moving_mean*
_output_shapes
:*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
4batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes
:*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB*  �?*
dtype0
�
#batch_normalization/moving_variance
VariableV2*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:
�
*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*6
_class,
*(loc:@batch_normalization/moving_variance*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
n
batch_normalization/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes

::
s
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
T0
*
_output_shapes
:
q
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
_output_shapes
:*
T0

^
 batch_normalization/cond/pred_idIdentityPlaceholder_2*
_output_shapes
:*
T0

�
batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
�
 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchconv2d/Conv2D batch_normalization/cond/pred_id* 
_class
loc:@conv2d/Conv2D*J
_output_shapes8
6:���������:���������*
T0
�
0batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
::
�
0batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
::
�
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:12batch_normalization/cond/FusedBatchNorm/Switch_2:1batch_normalization/cond/Const batch_normalization/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(
�
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchconv2d/Conv2D batch_normalization/cond/pred_id*
T0* 
_class
loc:@conv2d/Conv2D*J
_output_shapes8
6:���������:���������
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
::
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
::
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read batch_normalization/cond/pred_id*2
_class(
&$loc:@batch_normalization/moving_mean* 
_output_shapes
::*
T0
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read batch_normalization/cond/pred_id*
T0*6
_class,
*(loc:@batch_normalization/moving_variance* 
_output_shapes
::
�
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_22batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
�
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*1
_output_shapes
:���������: *
T0*
N
�
 batch_normalization/cond/Merge_1Merge+batch_normalization/cond/FusedBatchNorm_1:1)batch_normalization/cond/FusedBatchNorm:1*
N*
_output_shapes

:: *
T0
�
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
i
$batch_normalization/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
d
"batch_normalization/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
batch_normalization/ExpandDims
ExpandDims$batch_normalization/ExpandDims/input"batch_normalization/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
k
&batch_normalization/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
$batch_normalization/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization/ExpandDims_1
ExpandDims&batch_normalization/ExpandDims_1/input$batch_normalization/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
k
!batch_normalization/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
batch_normalization/ReshapeReshapePlaceholder_2!batch_normalization/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization/SelectSelectbatch_normalization/Reshapebatch_normalization/ExpandDims batch_normalization/ExpandDims_1*
_output_shapes
:*
T0
z
batch_normalization/SqueezeSqueezebatch_normalization/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
(batch_normalization/AssignMovingAvg/readIdentitybatch_normalization/moving_mean*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
'batch_normalization/AssignMovingAvg/SubSub(batch_normalization/AssignMovingAvg/read batch_normalization/cond/Merge_1*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
'batch_normalization/AssignMovingAvg/MulMul'batch_normalization/AssignMovingAvg/Subbatch_normalization/Squeeze*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
#batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/Mul*
use_locking( *
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
*batch_normalization/AssignMovingAvg_1/readIdentity#batch_normalization/moving_variance*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
�
)batch_normalization/AssignMovingAvg_1/SubSub*batch_normalization/AssignMovingAvg_1/read batch_normalization/cond/Merge_2*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
�
)batch_normalization/AssignMovingAvg_1/MulMul)batch_normalization/AssignMovingAvg_1/Subbatch_normalization/Squeeze*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
�
%batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/Mul*
use_locking( *
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
f
ReluRelubatch_normalization/cond/Merge*
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
conv2d_2/Conv2DConv2DReluconv2d_1/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0
�
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_1/gamma
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container 
�
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
�
,batch_normalization_1/beta/Initializer/zerosConst*
_output_shapes
:*-
_class#
!loc:@batch_normalization_1/beta*
valueB*    *
dtype0
�
batch_normalization_1/beta
VariableV2*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:*
dtype0
�
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:
�
3batch_normalization_1/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB*    
�
!batch_normalization_1/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container 
�
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
6batch_normalization_1/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
%batch_normalization_1/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:
�
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
:
�
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
p
!batch_normalization_2/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes

::
w
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
:
`
"batch_normalization_2/cond/pred_idIdentityPlaceholder_2*
_output_shapes
:*
T0

�
 batch_normalization_2/cond/ConstConst$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id*
T0*"
_class
loc:@conv2d_2/Conv2D*J
_output_shapes8
6:���������:���������
�
2batch_normalization_2/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::
�
2batch_normalization_2/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::
�
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:14batch_normalization_2/cond/FusedBatchNorm/Switch_2:1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(
�
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id*
T0*"
_class
loc:@conv2d_2/Conv2D*J
_output_shapes8
6:���������:���������
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::*
T0
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::*
T0
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_1/moving_mean/read"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
::
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_1/moving_variance/read"batch_normalization_2/cond/pred_id*8
_class.
,*loc:@batch_normalization_1/moving_variance* 
_output_shapes
::*
T0
�
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_24batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0
�
 batch_normalization_2/cond/MergeMerge+batch_normalization_2/cond/FusedBatchNorm_1)batch_normalization_2/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_2/cond/Merge_1Merge-batch_normalization_2/cond/FusedBatchNorm_1:1+batch_normalization_2/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_2/cond/Merge_2Merge-batch_normalization_2/cond/FusedBatchNorm_1:2+batch_normalization_2/cond/FusedBatchNorm:2*
N*
_output_shapes

:: *
T0
k
&batch_normalization_2/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
f
$batch_normalization_2/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0
�
 batch_normalization_2/ExpandDims
ExpandDims&batch_normalization_2/ExpandDims/input$batch_normalization_2/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
m
(batch_normalization_2/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
&batch_normalization_2/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_2/ExpandDims_1
ExpandDims(batch_normalization_2/ExpandDims_1/input&batch_normalization_2/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
m
#batch_normalization_2/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
batch_normalization_2/ReshapeReshapePlaceholder_2#batch_normalization_2/Reshape/shape*
Tshape0*
_output_shapes
:*
T0

�
batch_normalization_2/SelectSelectbatch_normalization_2/Reshape batch_normalization_2/ExpandDims"batch_normalization_2/ExpandDims_1*
T0*
_output_shapes
:
~
batch_normalization_2/SqueezeSqueezebatch_normalization_2/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
*batch_normalization_2/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
)batch_normalization_2/AssignMovingAvg/SubSub*batch_normalization_2/AssignMovingAvg/read"batch_normalization_2/cond/Merge_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
)batch_normalization_2/AssignMovingAvg/MulMul)batch_normalization_2/AssignMovingAvg/Subbatch_normalization_2/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_2/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
,batch_normalization_2/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
+batch_normalization_2/AssignMovingAvg_1/SubSub,batch_normalization_2/AssignMovingAvg_1/read"batch_normalization_2/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
�
+batch_normalization_2/AssignMovingAvg_1/MulMul+batch_normalization_2/AssignMovingAvg_1/Subbatch_normalization_2/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
�
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_2/AssignMovingAvg_1/Mul*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:*
use_locking( 
j
Relu_1Relu batch_normalization_2/cond/Merge*/
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
�
,batch_normalization_2/gamma/Initializer/onesConst*
_output_shapes
:*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  �?*
dtype0
�
batch_normalization_2/gamma
VariableV2*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:
�
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:
�
 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
,batch_normalization_2/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_2/beta
VariableV2*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:
�
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_2/beta
�
3batch_normalization_2/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    
�
!batch_normalization_2/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_2/moving_mean
�
(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
:*
use_locking(
�
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0
�
6batch_normalization_2/moving_variance/Initializer/onesConst*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB*  �?*
dtype0
�
%batch_normalization_2/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:
�
,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
p
!batch_normalization_3/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes

::
w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
_output_shapes
:*
T0

`
"batch_normalization_3/cond/pred_idIdentityPlaceholder_2*
_output_shapes
:*
T0

�
 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
�
0batch_normalization_3/cond/FusedBatchNorm/SwitchSwitchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id*
T0*"
_class
loc:@conv2d_3/Conv2D*J
_output_shapes8
6:���������:���������
�
2batch_normalization_3/cond/FusedBatchNorm/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::
�
2batch_normalization_3/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::
�
)batch_normalization_3/cond/FusedBatchNormFusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:14batch_normalization_3/cond/FusedBatchNorm/Switch_2:1 batch_normalization_3/cond/Const"batch_normalization_3/cond/Const_1*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:*
T0*
data_formatNHWC
�
2batch_normalization_3/cond/FusedBatchNorm_1/SwitchSwitchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id*
T0*"
_class
loc:@conv2d_3/Conv2D*J
_output_shapes8
6:���������:���������
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::*
T0
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read"batch_normalization_3/cond/pred_id* 
_output_shapes
::*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read"batch_normalization_3/cond/pred_id* 
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_24batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
_output_shapes

:: *
T0*
N
k
&batch_normalization_3/ExpandDims/inputConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
f
$batch_normalization_3/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization_3/ExpandDims
ExpandDims&batch_normalization_3/ExpandDims/input$batch_normalization_3/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
m
(batch_normalization_3/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
&batch_normalization_3/ExpandDims_1/dimConst*
_output_shapes
: *
value	B : *
dtype0
�
"batch_normalization_3/ExpandDims_1
ExpandDims(batch_normalization_3/ExpandDims_1/input&batch_normalization_3/ExpandDims_1/dim*
T0*
_output_shapes
:*

Tdim0
m
#batch_normalization_3/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_3/ReshapeReshapePlaceholder_2#batch_normalization_3/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_3/SelectSelectbatch_normalization_3/Reshape batch_normalization_3/ExpandDims"batch_normalization_3/ExpandDims_1*
T0*
_output_shapes
:
~
batch_normalization_3/SqueezeSqueezebatch_normalization_3/Select*
_output_shapes
: *
squeeze_dims
 *
T0
�
*batch_normalization_3/AssignMovingAvg/readIdentity!batch_normalization_2/moving_mean*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
)batch_normalization_3/AssignMovingAvg/SubSub*batch_normalization_3/AssignMovingAvg/read"batch_normalization_3/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
)batch_normalization_3/AssignMovingAvg/MulMul)batch_normalization_3/AssignMovingAvg/Subbatch_normalization_3/Squeeze*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
%batch_normalization_3/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean)batch_normalization_3/AssignMovingAvg/Mul*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
,batch_normalization_3/AssignMovingAvg_1/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
+batch_normalization_3/AssignMovingAvg_1/SubSub,batch_normalization_3/AssignMovingAvg_1/read"batch_normalization_3/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
+batch_normalization_3/AssignMovingAvg_1/MulMul+batch_normalization_3/AssignMovingAvg_1/Subbatch_normalization_3/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_3/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
j
Relu_2Relu batch_normalization_3/cond/Merge*
T0*/
_output_shapes
:���������
�
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_3/kernel*%
valueB"            
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
�
,batch_normalization_3/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_3/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_3/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma
�
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes
:
�
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
_output_shapes
:*
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
,batch_normalization_3/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_3/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_3/beta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_3/beta
�
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:
�
3batch_normalization_3/moving_mean/Initializer/zerosConst*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB*    *
dtype0
�
!batch_normalization_3/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container 
�
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(
�
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
6batch_normalization_3/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB*  �?
�
%batch_normalization_3/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:
�
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes
:
�
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
p
!batch_normalization_4/cond/SwitchSwitchPlaceholder_2Placeholder_2*
_output_shapes

::*
T0

w
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
:
`
"batch_normalization_4/cond/pred_idIdentityPlaceholder_2*
T0
*
_output_shapes
:
�
 batch_normalization_4/cond/ConstConst$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/cond/Const_1Const$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
0batch_normalization_4/cond/FusedBatchNorm/SwitchSwitchconv2d_4/Conv2D"batch_normalization_4/cond/pred_id*"
_class
loc:@conv2d_4/Conv2D*J
_output_shapes8
6:���������:���������*
T0
�
2batch_normalization_4/cond/FusedBatchNorm/Switch_1Switch batch_normalization_3/gamma/read"batch_normalization_4/cond/pred_id* 
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
2batch_normalization_4/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_3/beta/read"batch_normalization_4/cond/pred_id* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_3/beta
�
)batch_normalization_4/cond/FusedBatchNormFusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm/Switch:14batch_normalization_4/cond/FusedBatchNorm/Switch_1:14batch_normalization_4/cond/FusedBatchNorm/Switch_2:1 batch_normalization_4/cond/Const"batch_normalization_4/cond/Const_1*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:
�
2batch_normalization_4/cond/FusedBatchNorm_1/SwitchSwitchconv2d_4/Conv2D"batch_normalization_4/cond/pred_id*
T0*"
_class
loc:@conv2d_4/Conv2D*J
_output_shapes8
6:���������:���������
�
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_3/gamma/read"batch_normalization_4/cond/pred_id* 
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_3/beta/read"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta* 
_output_shapes
::
�
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_3/moving_mean/read"batch_normalization_4/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean* 
_output_shapes
::
�
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_3/moving_variance/read"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance* 
_output_shapes
::
�
+batch_normalization_4/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm_1/Switch4batch_normalization_4/cond/FusedBatchNorm_1/Switch_14batch_normalization_4/cond/FusedBatchNorm_1/Switch_24batch_normalization_4/cond/FusedBatchNorm_1/Switch_34batch_normalization_4/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
�
 batch_normalization_4/cond/MergeMerge+batch_normalization_4/cond/FusedBatchNorm_1)batch_normalization_4/cond/FusedBatchNorm*
N*1
_output_shapes
:���������: *
T0
�
"batch_normalization_4/cond/Merge_1Merge-batch_normalization_4/cond/FusedBatchNorm_1:1+batch_normalization_4/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_4/cond/Merge_2Merge-batch_normalization_4/cond/FusedBatchNorm_1:2+batch_normalization_4/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
k
&batch_normalization_4/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
f
$batch_normalization_4/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization_4/ExpandDims
ExpandDims&batch_normalization_4/ExpandDims/input$batch_normalization_4/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
m
(batch_normalization_4/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
&batch_normalization_4/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_1
ExpandDims(batch_normalization_4/ExpandDims_1/input&batch_normalization_4/ExpandDims_1/dim*
T0*
_output_shapes
:*

Tdim0
m
#batch_normalization_4/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_4/ReshapeReshapePlaceholder_2#batch_normalization_4/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_4/SelectSelectbatch_normalization_4/Reshape batch_normalization_4/ExpandDims"batch_normalization_4/ExpandDims_1*
T0*
_output_shapes
:
~
batch_normalization_4/SqueezeSqueezebatch_normalization_4/Select*
_output_shapes
: *
squeeze_dims
 *
T0
�
*batch_normalization_4/AssignMovingAvg/readIdentity!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:*
T0
�
)batch_normalization_4/AssignMovingAvg/SubSub*batch_normalization_4/AssignMovingAvg/read"batch_normalization_4/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
)batch_normalization_4/AssignMovingAvg/MulMul)batch_normalization_4/AssignMovingAvg/Subbatch_normalization_4/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_4/AssignMovingAvg/Mul*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
,batch_normalization_4/AssignMovingAvg_1/readIdentity%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:*
T0
�
+batch_normalization_4/AssignMovingAvg_1/SubSub,batch_normalization_4/AssignMovingAvg_1/read"batch_normalization_4/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
�
+batch_normalization_4/AssignMovingAvg_1/MulMul+batch_normalization_4/AssignMovingAvg_1/Subbatch_normalization_4/Squeeze*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_4/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
j
Relu_3Relu batch_normalization_4/cond/Merge*
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
conv2d_5/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
conv2d_5/Conv2DConv2DRelu_3conv2d_4/kernel/read*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������*
T0*
data_formatNHWC*
strides

�
,batch_normalization_4/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_4/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_4/gamma
VariableV2*.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
"batch_normalization_4/gamma/AssignAssignbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes
:
�
 batch_normalization_4/gamma/readIdentitybatch_normalization_4/gamma*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:
�
,batch_normalization_4/beta/Initializer/zerosConst*
_output_shapes
:*-
_class#
!loc:@batch_normalization_4/beta*
valueB*    *
dtype0
�
batch_normalization_4/beta
VariableV2*-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
!batch_normalization_4/beta/AssignAssignbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes
:*
use_locking(
�
batch_normalization_4/beta/readIdentitybatch_normalization_4/beta*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_4/beta
�
3batch_normalization_4/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_4/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_4/moving_mean
VariableV2*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container *
shape:
�
(batch_normalization_4/moving_mean/AssignAssign!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
validate_shape(
�
&batch_normalization_4/moving_mean/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
6batch_normalization_4/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
%batch_normalization_4/moving_variance
VariableV2*
dtype0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container *
shape:
�
,batch_normalization_4/moving_variance/AssignAssign%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
*batch_normalization_4/moving_variance/readIdentity%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:*
T0
p
!batch_normalization_5/cond/SwitchSwitchPlaceholder_2Placeholder_2*
_output_shapes

::*
T0

w
#batch_normalization_5/cond/switch_tIdentity#batch_normalization_5/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_5/cond/switch_fIdentity!batch_normalization_5/cond/Switch*
T0
*
_output_shapes
:
`
"batch_normalization_5/cond/pred_idIdentityPlaceholder_2*
T0
*
_output_shapes
:
�
 batch_normalization_5/cond/ConstConst$^batch_normalization_5/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_5/cond/Const_1Const$^batch_normalization_5/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
0batch_normalization_5/cond/FusedBatchNorm/SwitchSwitchconv2d_5/Conv2D"batch_normalization_5/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*"
_class
loc:@conv2d_5/Conv2D
�
2batch_normalization_5/cond/FusedBatchNorm/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_5/cond/pred_id*.
_class$
" loc:@batch_normalization_4/gamma* 
_output_shapes
::*
T0
�
2batch_normalization_5/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_5/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
::
�
)batch_normalization_5/cond/FusedBatchNormFusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm/Switch:14batch_normalization_5/cond/FusedBatchNorm/Switch_1:14batch_normalization_5/cond/FusedBatchNorm/Switch_2:1 batch_normalization_5/cond/Const"batch_normalization_5/cond/Const_1*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:
�
2batch_normalization_5/cond/FusedBatchNorm_1/SwitchSwitchconv2d_5/Conv2D"batch_normalization_5/cond/pred_id*
T0*"
_class
loc:@conv2d_5/Conv2D*J
_output_shapes8
6:���������:���������
�
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma* 
_output_shapes
::
�
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_5/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta* 
_output_shapes
::
�
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_4/moving_mean/read"batch_normalization_5/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean* 
_output_shapes
::
�
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_4/moving_variance/read"batch_normalization_5/cond/pred_id* 
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
+batch_normalization_5/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm_1/Switch4batch_normalization_5/cond/FusedBatchNorm_1/Switch_14batch_normalization_5/cond/FusedBatchNorm_1/Switch_24batch_normalization_5/cond/FusedBatchNorm_1/Switch_34batch_normalization_5/cond/FusedBatchNorm_1/Switch_4*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0*
data_formatNHWC
�
 batch_normalization_5/cond/MergeMerge+batch_normalization_5/cond/FusedBatchNorm_1)batch_normalization_5/cond/FusedBatchNorm*
N*1
_output_shapes
:���������: *
T0
�
"batch_normalization_5/cond/Merge_1Merge-batch_normalization_5/cond/FusedBatchNorm_1:1+batch_normalization_5/cond/FusedBatchNorm:1*
N*
_output_shapes

:: *
T0
�
"batch_normalization_5/cond/Merge_2Merge-batch_normalization_5/cond/FusedBatchNorm_1:2+batch_normalization_5/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
k
&batch_normalization_5/ExpandDims/inputConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
f
$batch_normalization_5/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization_5/ExpandDims
ExpandDims&batch_normalization_5/ExpandDims/input$batch_normalization_5/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
m
(batch_normalization_5/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
&batch_normalization_5/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
"batch_normalization_5/ExpandDims_1
ExpandDims(batch_normalization_5/ExpandDims_1/input&batch_normalization_5/ExpandDims_1/dim*
_output_shapes
:*

Tdim0*
T0
m
#batch_normalization_5/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_5/ReshapeReshapePlaceholder_2#batch_normalization_5/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_5/SelectSelectbatch_normalization_5/Reshape batch_normalization_5/ExpandDims"batch_normalization_5/ExpandDims_1*
T0*
_output_shapes
:
~
batch_normalization_5/SqueezeSqueezebatch_normalization_5/Select*
_output_shapes
: *
squeeze_dims
 *
T0
�
*batch_normalization_5/AssignMovingAvg/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
)batch_normalization_5/AssignMovingAvg/SubSub*batch_normalization_5/AssignMovingAvg/read"batch_normalization_5/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
)batch_normalization_5/AssignMovingAvg/MulMul)batch_normalization_5/AssignMovingAvg/Subbatch_normalization_5/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
%batch_normalization_5/AssignMovingAvg	AssignSub!batch_normalization_4/moving_mean)batch_normalization_5/AssignMovingAvg/Mul*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:*
use_locking( 
�
,batch_normalization_5/AssignMovingAvg_1/readIdentity%batch_normalization_4/moving_variance*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
+batch_normalization_5/AssignMovingAvg_1/SubSub,batch_normalization_5/AssignMovingAvg_1/read"batch_normalization_5/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:
�
+batch_normalization_5/AssignMovingAvg_1/MulMul+batch_normalization_5/AssignMovingAvg_1/Subbatch_normalization_5/Squeeze*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:
�
'batch_normalization_5/AssignMovingAvg_1	AssignSub%batch_normalization_4/moving_variance+batch_normalization_5/AssignMovingAvg_1/Mul*
_output_shapes
:*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
j
Relu_4Relu batch_normalization_5/cond/Merge*
T0*/
_output_shapes
:���������
^
Reshape/shapeConst*
_output_shapes
:*
valueB"����P  *
dtype0
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
+dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *>=*
dtype0
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�d*

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
dtype0*
_output_shapes
:	�d*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	�d
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
dense/MatMulMatMulReshapedense/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
�
,batch_normalization_5/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_5/gamma*
valueBd*  �?*
dtype0*
_output_shapes
:d
�
batch_normalization_5/gamma
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@batch_normalization_5/gamma*
	container 
�
"batch_normalization_5/gamma/AssignAssignbatch_normalization_5/gamma,batch_normalization_5/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes
:d
�
 batch_normalization_5/gamma/readIdentitybatch_normalization_5/gamma*
_output_shapes
:d*
T0*.
_class$
" loc:@batch_normalization_5/gamma
�
,batch_normalization_5/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:d*-
_class#
!loc:@batch_normalization_5/beta*
valueBd*    
�
batch_normalization_5/beta
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *-
_class#
!loc:@batch_normalization_5/beta*
	container 
�
!batch_normalization_5/beta/AssignAssignbatch_normalization_5/beta,batch_normalization_5/beta/Initializer/zeros*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes
:d*
use_locking(
�
batch_normalization_5/beta/readIdentitybatch_normalization_5/beta*
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes
:d
�
3batch_normalization_5/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_5/moving_mean*
valueBd*    *
dtype0*
_output_shapes
:d
�
!batch_normalization_5/moving_mean
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *4
_class*
(&loc:@batch_normalization_5/moving_mean*
	container *
shape:d
�
(batch_normalization_5/moving_mean/AssignAssign!batch_normalization_5/moving_mean3batch_normalization_5/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes
:d
�
&batch_normalization_5/moving_mean/readIdentity!batch_normalization_5/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:d
�
6batch_normalization_5/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_5/moving_variance*
valueBd*  �?*
dtype0*
_output_shapes
:d
�
%batch_normalization_5/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_5/moving_variance*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
,batch_normalization_5/moving_variance/AssignAssign%batch_normalization_5/moving_variance6batch_normalization_5/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
�
*batch_normalization_5/moving_variance/readIdentity%batch_normalization_5/moving_variance*
_output_shapes
:d*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
~
4batch_normalization_6/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB: 
�
"batch_normalization_6/moments/meanMeandense/MatMul4batch_normalization_6/moments/mean/reduction_indices*
_output_shapes

:d*
	keep_dims(*

Tidx0*
T0
�
*batch_normalization_6/moments/StopGradientStopGradient"batch_normalization_6/moments/mean*
T0*
_output_shapes

:d
�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense/MatMul*batch_normalization_6/moments/StopGradient*
T0*'
_output_shapes
:���������d
�
8batch_normalization_6/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_6/moments/varianceMean/batch_normalization_6/moments/SquaredDifference8batch_normalization_6/moments/variance/reduction_indices*
T0*
_output_shapes

:d*
	keep_dims(*

Tidx0
�
%batch_normalization_6/moments/SqueezeSqueeze"batch_normalization_6/moments/mean*
squeeze_dims
 *
T0*
_output_shapes
:d
�
'batch_normalization_6/moments/Squeeze_1Squeeze&batch_normalization_6/moments/variance*
_output_shapes
:d*
squeeze_dims
 *
T0
f
$batch_normalization_6/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0
�
 batch_normalization_6/ExpandDims
ExpandDims%batch_normalization_6/moments/Squeeze$batch_normalization_6/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:d
h
&batch_normalization_6/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
"batch_normalization_6/ExpandDims_1
ExpandDims&batch_normalization_5/moving_mean/read&batch_normalization_6/ExpandDims_1/dim*
_output_shapes

:d*

Tdim0*
T0
m
#batch_normalization_6/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_6/ReshapeReshapePlaceholder_2#batch_normalization_6/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_6/SelectSelectbatch_normalization_6/Reshape batch_normalization_6/ExpandDims"batch_normalization_6/ExpandDims_1*
_output_shapes

:d*
T0
�
batch_normalization_6/SqueezeSqueezebatch_normalization_6/Select*
T0*
_output_shapes
:d*
squeeze_dims
 
h
&batch_normalization_6/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_6/ExpandDims_2
ExpandDims'batch_normalization_6/moments/Squeeze_1&batch_normalization_6/ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes

:d
h
&batch_normalization_6/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_6/ExpandDims_3
ExpandDims*batch_normalization_5/moving_variance/read&batch_normalization_6/ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes

:d
o
%batch_normalization_6/Reshape_1/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_6/Reshape_1ReshapePlaceholder_2%batch_normalization_6/Reshape_1/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_6/Select_1Selectbatch_normalization_6/Reshape_1"batch_normalization_6/ExpandDims_2"batch_normalization_6/ExpandDims_3*
_output_shapes

:d*
T0
�
batch_normalization_6/Squeeze_1Squeezebatch_normalization_6/Select_1*
T0*
_output_shapes
:d*
squeeze_dims
 
m
(batch_normalization_6/ExpandDims_4/inputConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
h
&batch_normalization_6/ExpandDims_4/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_6/ExpandDims_4
ExpandDims(batch_normalization_6/ExpandDims_4/input&batch_normalization_6/ExpandDims_4/dim*
T0*
_output_shapes
:*

Tdim0
m
(batch_normalization_6/ExpandDims_5/inputConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
h
&batch_normalization_6/ExpandDims_5/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
"batch_normalization_6/ExpandDims_5
ExpandDims(batch_normalization_6/ExpandDims_5/input&batch_normalization_6/ExpandDims_5/dim*
_output_shapes
:*

Tdim0*
T0
o
%batch_normalization_6/Reshape_2/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
batch_normalization_6/Reshape_2ReshapePlaceholder_2%batch_normalization_6/Reshape_2/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_6/Select_2Selectbatch_normalization_6/Reshape_2"batch_normalization_6/ExpandDims_4"batch_normalization_6/ExpandDims_5*
_output_shapes
:*
T0
�
batch_normalization_6/Squeeze_2Squeezebatch_normalization_6/Select_2*
_output_shapes
: *
squeeze_dims
 *
T0
�
+batch_normalization_6/AssignMovingAvg/sub/xConst*
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0*
_output_shapes
: 
�
)batch_normalization_6/AssignMovingAvg/subSub+batch_normalization_6/AssignMovingAvg/sub/xbatch_normalization_6/Squeeze_2*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
�
+batch_normalization_6/AssignMovingAvg/sub_1Sub&batch_normalization_5/moving_mean/readbatch_normalization_6/Squeeze*
_output_shapes
:d*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
�
)batch_normalization_6/AssignMovingAvg/mulMul+batch_normalization_6/AssignMovingAvg/sub_1)batch_normalization_6/AssignMovingAvg/sub*
_output_shapes
:d*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
�
%batch_normalization_6/AssignMovingAvg	AssignSub!batch_normalization_5/moving_mean)batch_normalization_6/AssignMovingAvg/mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:d
�
-batch_normalization_6/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0*
_output_shapes
: 
�
+batch_normalization_6/AssignMovingAvg_1/subSub-batch_normalization_6/AssignMovingAvg_1/sub/xbatch_normalization_6/Squeeze_2*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
: *
T0
�
-batch_normalization_6/AssignMovingAvg_1/sub_1Sub*batch_normalization_5/moving_variance/readbatch_normalization_6/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:d
�
+batch_normalization_6/AssignMovingAvg_1/mulMul-batch_normalization_6/AssignMovingAvg_1/sub_1+batch_normalization_6/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:d
�
'batch_normalization_6/AssignMovingAvg_1	AssignSub%batch_normalization_5/moving_variance+batch_normalization_6/AssignMovingAvg_1/mul*
_output_shapes
:d*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
j
%batch_normalization_6/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_6/batchnorm/addAddbatch_normalization_6/Squeeze_1%batch_normalization_6/batchnorm/add/y*
_output_shapes
:d*
T0
x
%batch_normalization_6/batchnorm/RsqrtRsqrt#batch_normalization_6/batchnorm/add*
T0*
_output_shapes
:d
�
#batch_normalization_6/batchnorm/mulMul%batch_normalization_6/batchnorm/Rsqrt batch_normalization_5/gamma/read*
T0*
_output_shapes
:d
�
%batch_normalization_6/batchnorm/mul_1Muldense/MatMul#batch_normalization_6/batchnorm/mul*
T0*'
_output_shapes
:���������d
�
%batch_normalization_6/batchnorm/mul_2Mulbatch_normalization_6/Squeeze#batch_normalization_6/batchnorm/mul*
T0*
_output_shapes
:d
�
#batch_normalization_6/batchnorm/subSubbatch_normalization_5/beta/read%batch_normalization_6/batchnorm/mul_2*
T0*
_output_shapes
:d
�
%batch_normalization_6/batchnorm/add_1Add%batch_normalization_6/batchnorm/mul_1#batch_normalization_6/batchnorm/sub*'
_output_shapes
:���������d*
T0
g
Relu_5Relu%batch_normalization_6/batchnorm/add_1*
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
�
gradients/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
gradients/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
!gradients/Mean_grad/Reshape/shapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
gradients/Mean_grad/ShapeShapelogistic_loss$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
gradients/Mean_grad/Shape_1Shapelogistic_loss$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0*
out_type0
�
gradients/Mean_grad/Shape_2Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB 
�
gradients/Mean_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
gradients/Mean_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1*
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
�
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

�
-gradients/logistic_loss_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
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

�
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
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

�
(gradients/logistic_loss/Log1p_grad/add/xConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_18^gradients/logistic_loss_grad/tuple/control_dependency_1*
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

�
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikedense_2/BiasAdd$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*'
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
�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
�
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������

�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������
*
T0
�
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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

�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
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
�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*'
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

�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
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
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
_output_shapes
:
*
T0
�
/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1^gradients/AddN+^gradients/dense_2/BiasAdd_grad/BiasAddGrad
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
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_57gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:d
*
transpose_a(
�
.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1
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
�
:gradients/batch_normalization_6/batchnorm/add_1_grad/ShapeShape%batch_normalization_6/batchnorm/mul_1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/add_1_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB:d
�
Jgradients/batch_normalization_6/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_6/batchnorm/add_1_grad/Shape<gradients/batch_normalization_6/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients/batch_normalization_6/batchnorm/add_1_grad/SumSumgradients/Relu_5_grad/ReluGradJgradients/batch_normalization_6/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<gradients/batch_normalization_6/batchnorm/add_1_grad/ReshapeReshape8gradients/batch_normalization_6/batchnorm/add_1_grad/Sum:gradients/batch_normalization_6/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_6/batchnorm/add_1_grad/Sum_1Sumgradients/Relu_5_grad/ReluGradLgradients/batch_normalization_6/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape_1Reshape:gradients/batch_normalization_6/batchnorm/add_1_grad/Sum_1<gradients/batch_normalization_6/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Egradients/batch_normalization_6/batchnorm/add_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1=^gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape?^gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape_1
�
Mgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_6/batchnorm/add_1_grad/ReshapeF^gradients/batch_normalization_6/batchnorm/add_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������d
�
Ogradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape_1F^gradients/batch_normalization_6/batchnorm/add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_6/batchnorm/add_1_grad/Reshape_1*
_output_shapes
:d
�
:gradients/batch_normalization_6/batchnorm/mul_1_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_6/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape<gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/batch_normalization_6/batchnorm/mul_1_grad/mulMulMgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency#batch_normalization_6/batchnorm/mul*
T0*'
_output_shapes
:���������d
�
8gradients/batch_normalization_6/batchnorm/mul_1_grad/SumSum8gradients/batch_normalization_6/batchnorm/mul_1_grad/mulJgradients/batch_normalization_6/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<gradients/batch_normalization_6/batchnorm/mul_1_grad/ReshapeReshape8gradients/batch_normalization_6/batchnorm/mul_1_grad/Sum:gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_6/batchnorm/mul_1_grad/mul_1Muldense/MatMulMgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency*'
_output_shapes
:���������d*
T0
�
:gradients/batch_normalization_6/batchnorm/mul_1_grad/Sum_1Sum:gradients/batch_normalization_6/batchnorm/mul_1_grad/mul_1Lgradients/batch_normalization_6/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1Reshape:gradients/batch_normalization_6/batchnorm/mul_1_grad/Sum_1<gradients/batch_normalization_6/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Egradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1=^gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape?^gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1
�
Mgradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_6/batchnorm/mul_1_grad/ReshapeF^gradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape*'
_output_shapes
:���������d
�
Ogradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1F^gradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/sub_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_6/batchnorm/sub_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Hgradients/batch_normalization_6/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_6/batchnorm/sub_grad/Shape:gradients/batch_normalization_6/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
6gradients/batch_normalization_6/batchnorm/sub_grad/SumSumOgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency_1Hgradients/batch_normalization_6/batchnorm/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
:gradients/batch_normalization_6/batchnorm/sub_grad/ReshapeReshape6gradients/batch_normalization_6/batchnorm/sub_grad/Sum8gradients/batch_normalization_6/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/sub_grad/Sum_1SumOgradients/batch_normalization_6/batchnorm/add_1_grad/tuple/control_dependency_1Jgradients/batch_normalization_6/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6gradients/batch_normalization_6/batchnorm/sub_grad/NegNeg8gradients/batch_normalization_6/batchnorm/sub_grad/Sum_1*
T0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/sub_grad/Reshape_1Reshape6gradients/batch_normalization_6/batchnorm/sub_grad/Neg:gradients/batch_normalization_6/batchnorm/sub_grad/Shape_1*
Tshape0*
_output_shapes
:d*
T0
�
Cgradients/batch_normalization_6/batchnorm/sub_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1;^gradients/batch_normalization_6/batchnorm/sub_grad/Reshape=^gradients/batch_normalization_6/batchnorm/sub_grad/Reshape_1
�
Kgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_6/batchnorm/sub_grad/ReshapeD^gradients/batch_normalization_6/batchnorm/sub_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/batch_normalization_6/batchnorm/sub_grad/Reshape*
_output_shapes
:d
�
Mgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_6/batchnorm/sub_grad/Reshape_1D^gradients/batch_normalization_6/batchnorm/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/sub_grad/Reshape_1*
_output_shapes
:d
�
:gradients/batch_normalization_6/batchnorm/mul_2_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
<gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_6/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape<gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/batch_normalization_6/batchnorm/mul_2_grad/mulMulMgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependency_1#batch_normalization_6/batchnorm/mul*
T0*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/mul_2_grad/SumSum8gradients/batch_normalization_6/batchnorm/mul_2_grad/mulJgradients/batch_normalization_6/batchnorm/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/mul_2_grad/ReshapeReshape8gradients/batch_normalization_6/batchnorm/mul_2_grad/Sum:gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
:gradients/batch_normalization_6/batchnorm/mul_2_grad/mul_1Mulbatch_normalization_6/SqueezeMgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0
�
:gradients/batch_normalization_6/batchnorm/mul_2_grad/Sum_1Sum:gradients/batch_normalization_6/batchnorm/mul_2_grad/mul_1Lgradients/batch_normalization_6/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape_1Reshape:gradients/batch_normalization_6/batchnorm/mul_2_grad/Sum_1<gradients/batch_normalization_6/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Egradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1=^gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape?^gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape_1
�
Mgradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_6/batchnorm/mul_2_grad/ReshapeF^gradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes
:d*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape
�
Ogradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape_1F^gradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_6/batchnorm/mul_2_grad/Reshape_1*
_output_shapes
:d
�
2gradients/batch_normalization_6/Squeeze_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
4gradients/batch_normalization_6/Squeeze_grad/ReshapeReshapeMgradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/control_dependency2gradients/batch_normalization_6/Squeeze_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
gradients/AddN_1AddNOgradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/control_dependency_1Ogradients/batch_normalization_6/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/mul_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_6/batchnorm/mul_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Hgradients/batch_normalization_6/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_6/batchnorm/mul_grad/Shape:gradients/batch_normalization_6/batchnorm/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
6gradients/batch_normalization_6/batchnorm/mul_grad/mulMulgradients/AddN_1 batch_normalization_5/gamma/read*
_output_shapes
:d*
T0
�
6gradients/batch_normalization_6/batchnorm/mul_grad/SumSum6gradients/batch_normalization_6/batchnorm/mul_grad/mulHgradients/batch_normalization_6/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:gradients/batch_normalization_6/batchnorm/mul_grad/ReshapeReshape6gradients/batch_normalization_6/batchnorm/mul_grad/Sum8gradients/batch_normalization_6/batchnorm/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/mul_grad/mul_1Mul%batch_normalization_6/batchnorm/Rsqrtgradients/AddN_1*
_output_shapes
:d*
T0
�
8gradients/batch_normalization_6/batchnorm/mul_grad/Sum_1Sum8gradients/batch_normalization_6/batchnorm/mul_grad/mul_1Jgradients/batch_normalization_6/batchnorm/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<gradients/batch_normalization_6/batchnorm/mul_grad/Reshape_1Reshape8gradients/batch_normalization_6/batchnorm/mul_grad/Sum_1:gradients/batch_normalization_6/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Cgradients/batch_normalization_6/batchnorm/mul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1;^gradients/batch_normalization_6/batchnorm/mul_grad/Reshape=^gradients/batch_normalization_6/batchnorm/mul_grad/Reshape_1
�
Kgradients/batch_normalization_6/batchnorm/mul_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_6/batchnorm/mul_grad/ReshapeD^gradients/batch_normalization_6/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:d*
T0*M
_classC
A?loc:@gradients/batch_normalization_6/batchnorm/mul_grad/Reshape
�
Mgradients/batch_normalization_6/batchnorm/mul_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_6/batchnorm/mul_grad/Reshape_1D^gradients/batch_normalization_6/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:d*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/mul_grad/Reshape_1
�
6gradients/batch_normalization_6/Select_grad/zeros_likeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueBd*    *
dtype0*
_output_shapes

:d
�
2gradients/batch_normalization_6/Select_grad/SelectSelectbatch_normalization_6/Reshape4gradients/batch_normalization_6/Squeeze_grad/Reshape6gradients/batch_normalization_6/Select_grad/zeros_like*
T0*
_output_shapes

:d
�
4gradients/batch_normalization_6/Select_grad/Select_1Selectbatch_normalization_6/Reshape6gradients/batch_normalization_6/Select_grad/zeros_like4gradients/batch_normalization_6/Squeeze_grad/Reshape*
T0*
_output_shapes

:d
�
<gradients/batch_normalization_6/Select_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/batch_normalization_6/Select_grad/Select5^gradients/batch_normalization_6/Select_grad/Select_1
�
Dgradients/batch_normalization_6/Select_grad/tuple/control_dependencyIdentity2gradients/batch_normalization_6/Select_grad/Select=^gradients/batch_normalization_6/Select_grad/tuple/group_deps*E
_class;
97loc:@gradients/batch_normalization_6/Select_grad/Select*
_output_shapes

:d*
T0
�
Fgradients/batch_normalization_6/Select_grad/tuple/control_dependency_1Identity4gradients/batch_normalization_6/Select_grad/Select_1=^gradients/batch_normalization_6/Select_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/batch_normalization_6/Select_grad/Select_1*
_output_shapes

:d
�
>gradients/batch_normalization_6/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_6/batchnorm/RsqrtKgradients/batch_normalization_6/batchnorm/mul_grad/tuple/control_dependency*
_output_shapes
:d*
T0
�
5gradients/batch_normalization_6/ExpandDims_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
7gradients/batch_normalization_6/ExpandDims_grad/ReshapeReshapeDgradients/batch_normalization_6/Select_grad/tuple/control_dependency5gradients/batch_normalization_6/ExpandDims_grad/Shape*
Tshape0*
_output_shapes
:d*
T0
�
8gradients/batch_normalization_6/batchnorm/add_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_6/batchnorm/add_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
Hgradients/batch_normalization_6/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_6/batchnorm/add_grad/Shape:gradients/batch_normalization_6/batchnorm/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
6gradients/batch_normalization_6/batchnorm/add_grad/SumSum>gradients/batch_normalization_6/batchnorm/Rsqrt_grad/RsqrtGradHgradients/batch_normalization_6/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:gradients/batch_normalization_6/batchnorm/add_grad/ReshapeReshape6gradients/batch_normalization_6/batchnorm/add_grad/Sum8gradients/batch_normalization_6/batchnorm/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
8gradients/batch_normalization_6/batchnorm/add_grad/Sum_1Sum>gradients/batch_normalization_6/batchnorm/Rsqrt_grad/RsqrtGradJgradients/batch_normalization_6/batchnorm/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<gradients/batch_normalization_6/batchnorm/add_grad/Reshape_1Reshape8gradients/batch_normalization_6/batchnorm/add_grad/Sum_1:gradients/batch_normalization_6/batchnorm/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Cgradients/batch_normalization_6/batchnorm/add_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1;^gradients/batch_normalization_6/batchnorm/add_grad/Reshape=^gradients/batch_normalization_6/batchnorm/add_grad/Reshape_1
�
Kgradients/batch_normalization_6/batchnorm/add_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_6/batchnorm/add_grad/ReshapeD^gradients/batch_normalization_6/batchnorm/add_grad/tuple/group_deps*
_output_shapes
:d*
T0*M
_classC
A?loc:@gradients/batch_normalization_6/batchnorm/add_grad/Reshape
�
Mgradients/batch_normalization_6/batchnorm/add_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_6/batchnorm/add_grad/Reshape_1D^gradients/batch_normalization_6/batchnorm/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
:gradients/batch_normalization_6/moments/Squeeze_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
<gradients/batch_normalization_6/moments/Squeeze_grad/ReshapeReshape7gradients/batch_normalization_6/ExpandDims_grad/Reshape:gradients/batch_normalization_6/moments/Squeeze_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
4gradients/batch_normalization_6/Squeeze_1_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
6gradients/batch_normalization_6/Squeeze_1_grad/ReshapeReshapeKgradients/batch_normalization_6/batchnorm/add_grad/tuple/control_dependency4gradients/batch_normalization_6/Squeeze_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
8gradients/batch_normalization_6/Select_1_grad/zeros_likeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueBd*    *
dtype0*
_output_shapes

:d
�
4gradients/batch_normalization_6/Select_1_grad/SelectSelectbatch_normalization_6/Reshape_16gradients/batch_normalization_6/Squeeze_1_grad/Reshape8gradients/batch_normalization_6/Select_1_grad/zeros_like*
T0*
_output_shapes

:d
�
6gradients/batch_normalization_6/Select_1_grad/Select_1Selectbatch_normalization_6/Reshape_18gradients/batch_normalization_6/Select_1_grad/zeros_like6gradients/batch_normalization_6/Squeeze_1_grad/Reshape*
T0*
_output_shapes

:d
�
>gradients/batch_normalization_6/Select_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_15^gradients/batch_normalization_6/Select_1_grad/Select7^gradients/batch_normalization_6/Select_1_grad/Select_1
�
Fgradients/batch_normalization_6/Select_1_grad/tuple/control_dependencyIdentity4gradients/batch_normalization_6/Select_1_grad/Select?^gradients/batch_normalization_6/Select_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/batch_normalization_6/Select_1_grad/Select*
_output_shapes

:d
�
Hgradients/batch_normalization_6/Select_1_grad/tuple/control_dependency_1Identity6gradients/batch_normalization_6/Select_1_grad/Select_1?^gradients/batch_normalization_6/Select_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/batch_normalization_6/Select_1_grad/Select_1*
_output_shapes

:d
�
7gradients/batch_normalization_6/ExpandDims_2_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB:d
�
9gradients/batch_normalization_6/ExpandDims_2_grad/ReshapeReshapeFgradients/batch_normalization_6/Select_1_grad/tuple/control_dependency7gradients/batch_normalization_6/ExpandDims_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
<gradients/batch_normalization_6/moments/Squeeze_1_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
valueB"   d   *
dtype0
�
>gradients/batch_normalization_6/moments/Squeeze_1_grad/ReshapeReshape9gradients/batch_normalization_6/ExpandDims_2_grad/Reshape<gradients/batch_normalization_6/moments/Squeeze_1_grad/Shape*
_output_shapes

:d*
T0*
Tshape0
�
;gradients/batch_normalization_6/moments/variance_grad/ShapeShape/batch_normalization_6/moments/SquaredDifference$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0*
out_type0
�
:gradients/batch_normalization_6/moments/variance_grad/SizeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
9gradients/batch_normalization_6/moments/variance_grad/addAdd8batch_normalization_6/moments/variance/reduction_indices:gradients/batch_normalization_6/moments/variance_grad/Size*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_6/moments/variance_grad/modFloorMod9gradients/batch_normalization_6/moments/variance_grad/add:gradients/batch_normalization_6/moments/variance_grad/Size*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
_output_shapes
:*
T0
�
=gradients/batch_normalization_6/moments/variance_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB:*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape
�
Agradients/batch_normalization_6/moments/variance_grad/range/startConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B : *N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Agradients/batch_normalization_6/moments/variance_grad/range/deltaConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
;gradients/batch_normalization_6/moments/variance_grad/rangeRangeAgradients/batch_normalization_6/moments/variance_grad/range/start:gradients/batch_normalization_6/moments/variance_grad/SizeAgradients/batch_normalization_6/moments/variance_grad/range/delta*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
_output_shapes
:*

Tidx0
�
@gradients/batch_normalization_6/moments/variance_grad/Fill/valueConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
:gradients/batch_normalization_6/moments/variance_grad/FillFill=gradients/batch_normalization_6/moments/variance_grad/Shape_1@gradients/batch_normalization_6/moments/variance_grad/Fill/value*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
_output_shapes
:
�
Cgradients/batch_normalization_6/moments/variance_grad/DynamicStitchDynamicStitch;gradients/batch_normalization_6/moments/variance_grad/range9gradients/batch_normalization_6/moments/variance_grad/mod;gradients/batch_normalization_6/moments/variance_grad/Shape:gradients/batch_normalization_6/moments/variance_grad/Fill*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
N*#
_output_shapes
:���������
�
?gradients/batch_normalization_6/moments/variance_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
=gradients/batch_normalization_6/moments/variance_grad/MaximumMaximumCgradients/batch_normalization_6/moments/variance_grad/DynamicStitch?gradients/batch_normalization_6/moments/variance_grad/Maximum/y*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*#
_output_shapes
:���������
�
>gradients/batch_normalization_6/moments/variance_grad/floordivFloorDiv;gradients/batch_normalization_6/moments/variance_grad/Shape=gradients/batch_normalization_6/moments/variance_grad/Maximum*
T0*N
_classD
B@loc:@gradients/batch_normalization_6/moments/variance_grad/Shape*
_output_shapes
:
�
=gradients/batch_normalization_6/moments/variance_grad/ReshapeReshape>gradients/batch_normalization_6/moments/Squeeze_1_grad/ReshapeCgradients/batch_normalization_6/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
:gradients/batch_normalization_6/moments/variance_grad/TileTile=gradients/batch_normalization_6/moments/variance_grad/Reshape>gradients/batch_normalization_6/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
=gradients/batch_normalization_6/moments/variance_grad/Shape_2Shape/batch_normalization_6/moments/SquaredDifference$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
=gradients/batch_normalization_6/moments/variance_grad/Shape_3Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
valueB"   d   *
dtype0
�
;gradients/batch_normalization_6/moments/variance_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB: *P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_6/moments/variance_grad/ProdProd=gradients/batch_normalization_6/moments/variance_grad/Shape_2;gradients/batch_normalization_6/moments/variance_grad/Const*
T0*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
�
=gradients/batch_normalization_6/moments/variance_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB: *P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
<gradients/batch_normalization_6/moments/variance_grad/Prod_1Prod=gradients/batch_normalization_6/moments/variance_grad/Shape_3=gradients/batch_normalization_6/moments/variance_grad/Const_1*
T0*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Agradients/batch_normalization_6/moments/variance_grad/Maximum_1/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
�
?gradients/batch_normalization_6/moments/variance_grad/Maximum_1Maximum<gradients/batch_normalization_6/moments/variance_grad/Prod_1Agradients/batch_normalization_6/moments/variance_grad/Maximum_1/y*
T0*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
_output_shapes
: 
�
@gradients/batch_normalization_6/moments/variance_grad/floordiv_1FloorDiv:gradients/batch_normalization_6/moments/variance_grad/Prod?gradients/batch_normalization_6/moments/variance_grad/Maximum_1*
T0*P
_classF
DBloc:@gradients/batch_normalization_6/moments/variance_grad/Shape_2*
_output_shapes
: 
�
:gradients/batch_normalization_6/moments/variance_grad/CastCast@gradients/batch_normalization_6/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
=gradients/batch_normalization_6/moments/variance_grad/truedivRealDiv:gradients/batch_normalization_6/moments/variance_grad/Tile:gradients/batch_normalization_6/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_6/moments/SquaredDifference_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
Fgradients/batch_normalization_6/moments/SquaredDifference_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
valueB"   d   *
dtype0
�
Tgradients/batch_normalization_6/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/batch_normalization_6/moments/SquaredDifference_grad/ShapeFgradients/batch_normalization_6/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Egradients/batch_normalization_6/moments/SquaredDifference_grad/scalarConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1>^gradients/batch_normalization_6/moments/variance_grad/truediv*
_output_shapes
: *
valueB
 *   @*
dtype0
�
Bgradients/batch_normalization_6/moments/SquaredDifference_grad/mulMulEgradients/batch_normalization_6/moments/SquaredDifference_grad/scalar=gradients/batch_normalization_6/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������d
�
Bgradients/batch_normalization_6/moments/SquaredDifference_grad/subSubdense/MatMul*batch_normalization_6/moments/StopGradient$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1>^gradients/batch_normalization_6/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_6/moments/SquaredDifference_grad/mul_1MulBgradients/batch_normalization_6/moments/SquaredDifference_grad/mulBgradients/batch_normalization_6/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������d
�
Bgradients/batch_normalization_6/moments/SquaredDifference_grad/SumSumDgradients/batch_normalization_6/moments/SquaredDifference_grad/mul_1Tgradients/batch_normalization_6/moments/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Fgradients/batch_normalization_6/moments/SquaredDifference_grad/ReshapeReshapeBgradients/batch_normalization_6/moments/SquaredDifference_grad/SumDgradients/batch_normalization_6/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_6/moments/SquaredDifference_grad/Sum_1SumDgradients/batch_normalization_6/moments/SquaredDifference_grad/mul_1Vgradients/batch_normalization_6/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Hgradients/batch_normalization_6/moments/SquaredDifference_grad/Reshape_1ReshapeDgradients/batch_normalization_6/moments/SquaredDifference_grad/Sum_1Fgradients/batch_normalization_6/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:d
�
Bgradients/batch_normalization_6/moments/SquaredDifference_grad/NegNegHgradients/batch_normalization_6/moments/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes

:d
�
Ogradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1G^gradients/batch_normalization_6/moments/SquaredDifference_grad/ReshapeC^gradients/batch_normalization_6/moments/SquaredDifference_grad/Neg
�
Wgradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/control_dependencyIdentityFgradients/batch_normalization_6/moments/SquaredDifference_grad/ReshapeP^gradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/batch_normalization_6/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������d
�
Ygradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityBgradients/batch_normalization_6/moments/SquaredDifference_grad/NegP^gradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/group_deps*
_output_shapes

:d*
T0*U
_classK
IGloc:@gradients/batch_normalization_6/moments/SquaredDifference_grad/Neg
�
7gradients/batch_normalization_6/moments/mean_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
6gradients/batch_normalization_6/moments/mean_grad/SizeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
5gradients/batch_normalization_6/moments/mean_grad/addAdd4batch_normalization_6/moments/mean/reduction_indices6gradients/batch_normalization_6/moments/mean_grad/Size*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
_output_shapes
:*
T0
�
5gradients/batch_normalization_6/moments/mean_grad/modFloorMod5gradients/batch_normalization_6/moments/mean_grad/add6gradients/batch_normalization_6/moments/mean_grad/Size*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_6/moments/mean_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB:*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
�
=gradients/batch_normalization_6/moments/mean_grad/range/startConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B : *J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
=gradients/batch_normalization_6/moments/mean_grad/range/deltaConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
7gradients/batch_normalization_6/moments/mean_grad/rangeRange=gradients/batch_normalization_6/moments/mean_grad/range/start6gradients/batch_normalization_6/moments/mean_grad/Size=gradients/batch_normalization_6/moments/mean_grad/range/delta*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
�
<gradients/batch_normalization_6/moments/mean_grad/Fill/valueConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
6gradients/batch_normalization_6/moments/mean_grad/FillFill9gradients/batch_normalization_6/moments/mean_grad/Shape_1<gradients/batch_normalization_6/moments/mean_grad/Fill/value*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
_output_shapes
:
�
?gradients/batch_normalization_6/moments/mean_grad/DynamicStitchDynamicStitch7gradients/batch_normalization_6/moments/mean_grad/range5gradients/batch_normalization_6/moments/mean_grad/mod7gradients/batch_normalization_6/moments/mean_grad/Shape6gradients/batch_normalization_6/moments/mean_grad/Fill*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
N*#
_output_shapes
:���������
�
;gradients/batch_normalization_6/moments/mean_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
9gradients/batch_normalization_6/moments/mean_grad/MaximumMaximum?gradients/batch_normalization_6/moments/mean_grad/DynamicStitch;gradients/batch_normalization_6/moments/mean_grad/Maximum/y*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*#
_output_shapes
:���������
�
:gradients/batch_normalization_6/moments/mean_grad/floordivFloorDiv7gradients/batch_normalization_6/moments/mean_grad/Shape9gradients/batch_normalization_6/moments/mean_grad/Maximum*
T0*J
_class@
><loc:@gradients/batch_normalization_6/moments/mean_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_6/moments/mean_grad/ReshapeReshape<gradients/batch_normalization_6/moments/Squeeze_grad/Reshape?gradients/batch_normalization_6/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
6gradients/batch_normalization_6/moments/mean_grad/TileTile9gradients/batch_normalization_6/moments/mean_grad/Reshape:gradients/batch_normalization_6/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
9gradients/batch_normalization_6/moments/mean_grad/Shape_2Shapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
9gradients/batch_normalization_6/moments/mean_grad/Shape_3Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB"   d   
�
7gradients/batch_normalization_6/moments/mean_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB: *L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
6gradients/batch_normalization_6/moments/mean_grad/ProdProd9gradients/batch_normalization_6/moments/mean_grad/Shape_27gradients/batch_normalization_6/moments/mean_grad/Const*
T0*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
�
9gradients/batch_normalization_6/moments/mean_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB: *L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
8gradients/batch_normalization_6/moments/mean_grad/Prod_1Prod9gradients/batch_normalization_6/moments/mean_grad/Shape_39gradients/batch_normalization_6/moments/mean_grad/Const_1*
T0*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
_output_shapes
: *
	keep_dims( *

Tidx0
�
=gradients/batch_normalization_6/moments/mean_grad/Maximum_1/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
value	B :*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
�
;gradients/batch_normalization_6/moments/mean_grad/Maximum_1Maximum8gradients/batch_normalization_6/moments/mean_grad/Prod_1=gradients/batch_normalization_6/moments/mean_grad/Maximum_1/y*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2*
_output_shapes
: *
T0
�
<gradients/batch_normalization_6/moments/mean_grad/floordiv_1FloorDiv6gradients/batch_normalization_6/moments/mean_grad/Prod;gradients/batch_normalization_6/moments/mean_grad/Maximum_1*
_output_shapes
: *
T0*L
_classB
@>loc:@gradients/batch_normalization_6/moments/mean_grad/Shape_2
�
6gradients/batch_normalization_6/moments/mean_grad/CastCast<gradients/batch_normalization_6/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
9gradients/batch_normalization_6/moments/mean_grad/truedivRealDiv6gradients/batch_normalization_6/moments/mean_grad/Tile6gradients/batch_normalization_6/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������d
�
gradients/AddN_2AddNMgradients/batch_normalization_6/batchnorm/mul_1_grad/tuple/control_dependencyWgradients/batch_normalization_6/moments/SquaredDifference_grad/tuple/control_dependency9gradients/batch_normalization_6/moments/mean_grad/truediv*
N*'
_output_shapes
:���������d*
T0*O
_classE
CAloc:@gradients/batch_normalization_6/batchnorm/mul_1_grad/Reshape
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/AddN_2dense/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/AddN_2*
T0*
_output_shapes
:	�d*
transpose_a(*
transpose_b( 
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�d
�
gradients/Reshape_grad/ShapeShapeRelu_4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
gradients/Relu_4_grad/ReluGradReluGradgradients/Reshape_grad/ReshapeRelu_4*/
_output_shapes
:���������*
T0
�
9gradients/batch_normalization_5/cond/Merge_grad/cond_gradSwitchgradients/Relu_4_grad/ReluGrad"batch_normalization_5/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_5/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^gradients/batch_normalization_5/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_5/cond/Merge_grad/cond_gradA^gradients/batch_normalization_5/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*/
_output_shapes
:���������
�
Jgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_5/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_5/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*/
_output_shapes
:���������
�
gradients/zeros_like	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_1	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_2	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_3	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency2batch_normalization_5/cond/FusedBatchNorm_1/Switch4batch_normalization_5/cond/FusedBatchNorm_1/Switch_14batch_normalization_5/cond/FusedBatchNorm_1/Switch_34batch_normalization_5/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1N^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Ugradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Ugradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
gradients/zeros_like_4	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_5	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_6	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_7	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency_12batch_normalization_5/cond/FusedBatchNorm/Switch:14batch_normalization_5/cond/FusedBatchNorm/Switch_1:1+batch_normalization_5/cond/FusedBatchNorm:3+batch_normalization_5/cond/FusedBatchNorm:4*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:*
T0
�
Igradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
gradients/SwitchSwitchconv2d_5/Conv2D"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
{
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*/
_output_shapes
:���������
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_2Shapegradients/Switch_1:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_1/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
j
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_2/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_3Switchconv2d_5/Conv2D"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
c
gradients/Shape_4Shapegradients/Switch_3*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_3/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*/
_output_shapes
:���������*
T0
�
Igradients/batch_normalization_5/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_4Switch batch_normalization_4/gamma/read"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
c
gradients/Shape_5Shapegradients/Switch_4*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_4/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_5Switchbatch_normalization_4/beta/read"batch_normalization_5/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
c
gradients/Shape_6Shapegradients/Switch_5*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_5/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
j
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_3AddNKgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������
�
%gradients/conv2d_5/Conv2D_grad/ShapeNShapeNRelu_3conv2d_4/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_5/Conv2D_grad/ShapeNconv2d_4/kernel/readgradients/AddN_3*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_3'gradients/conv2d_5/Conv2D_grad/ShapeN:1gradients/AddN_3*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
/gradients/conv2d_5/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/conv2d_5/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_5/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_4AddNMgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_5AddNMgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
gradients/Relu_3_grad/ReluGradReluGrad7gradients/conv2d_5/Conv2D_grad/tuple/control_dependencyRelu_3*
T0*/
_output_shapes
:���������
�
9gradients/batch_normalization_4/cond/Merge_grad/cond_gradSwitchgradients/Relu_3_grad/ReluGrad"batch_normalization_4/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_4/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^gradients/batch_normalization_4/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_4/cond/Merge_grad/cond_gradA^gradients/batch_normalization_4/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
�
Jgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_4/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
�
gradients/zeros_like_8	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_9	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_10	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_11	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency2batch_normalization_4/cond/FusedBatchNorm_1/Switch4batch_normalization_4/cond/FusedBatchNorm_1/Switch_14batch_normalization_4/cond/FusedBatchNorm_1/Switch_34batch_normalization_4/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1N^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������*
T0
�
Ugradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Ugradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
gradients/zeros_like_12	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_13	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_14	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_15	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency_12batch_normalization_4/cond/FusedBatchNorm/Switch:14batch_normalization_4/cond/FusedBatchNorm/Switch_1:1+batch_normalization_4/cond/FusedBatchNorm:3+batch_normalization_4/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:
�
Igradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
gradients/Switch_6Switchconv2d_4/Conv2D"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
e
gradients/Shape_7Shapegradients/Switch_6:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_6/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*/
_output_shapes
:���������
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_7Switch batch_normalization_3/gamma/read"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_8Shapegradients/Switch_7:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_7/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_8Switchbatch_normalization_3/beta/read"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_9Shapegradients/Switch_8:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_8/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
_output_shapes

:: *
T0*
N
�
gradients/Switch_9Switchconv2d_4/Conv2D"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
d
gradients/Shape_10Shapegradients/Switch_9*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_9/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization_4/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_10Switch batch_normalization_3/gamma/read"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_11Shapegradients/Switch_10*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_10/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_11Switchbatch_normalization_3/beta/read"batch_normalization_4/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_12Shapegradients/Switch_11*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_11/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
N*
_output_shapes

:: *
T0
�
gradients/AddN_6AddNKgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_grad
�
%gradients/conv2d_4/Conv2D_grad/ShapeNShapeNRelu_2conv2d_3/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
out_type0*
N* 
_output_shapes
::*
T0
�
2gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_4/Conv2D_grad/ShapeNconv2d_3/kernel/readgradients/AddN_6*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides

�
3gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_2'gradients/conv2d_4/Conv2D_grad/ShapeN:1gradients/AddN_6*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
/gradients/conv2d_4/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_4/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_7AddNMgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_8AddNMgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
gradients/Relu_2_grad/ReluGradReluGrad7gradients/conv2d_4/Conv2D_grad/tuple/control_dependencyRelu_2*/
_output_shapes
:���������*
T0
�
9gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitchgradients/Relu_2_grad/ReluGrad"batch_normalization_3/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
@gradients/batch_normalization_3/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^gradients/batch_normalization_3/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_3/cond/Merge_grad/cond_gradA^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:���������
�
Jgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:���������
�
gradients/zeros_like_16	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_17	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_18	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_19	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1N^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
gradients/zeros_like_20	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_21	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_22	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_23	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_12batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:1+batch_normalization_3/cond/FusedBatchNorm:3+batch_normalization_3/cond/FusedBatchNorm:4*
epsilon%o�:*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(
�
Igradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
gradients/Switch_12Switchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
g
gradients/Shape_13Shapegradients/Switch_12:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_12/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*/
_output_shapes
:���������*
T0
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_13Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_14Shapegradients/Switch_13:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_13/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
N*
_output_shapes

:: *
T0
�
gradients/Switch_14Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_15Shapegradients/Switch_14:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_14/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
N*
_output_shapes

:: *
T0
�
gradients/Switch_15Switchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
e
gradients/Shape_16Shapegradients/Switch_15*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_15/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*/
_output_shapes
:���������*
T0
�
Igradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_16Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_17Shapegradients/Switch_16*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_16/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
m
gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_17Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_18Shapegradients/Switch_17*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_17/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
m
gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_9AddNKgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_grad*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
N* 
_output_shapes
::*
T0*
out_type0
�
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/AddN_9*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides

�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/AddN_9*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
/gradients/conv2d_3/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_10AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_11AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
gradients/Relu_1_grad/ReluGradReluGrad7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyRelu_1*
T0*/
_output_shapes
:���������
�
9gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchgradients/Relu_1_grad/ReluGrad"batch_normalization_2/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_2/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^gradients/batch_normalization_2/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_2/cond/Merge_grad/cond_gradA^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
Jgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*/
_output_shapes
:���������*
T0
�
gradients/zeros_like_24	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_25	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_26	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_27	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1N^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
gradients/zeros_like_28	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_29	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_30	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_31	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_12batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:1+batch_normalization_2/cond/FusedBatchNorm:3+batch_normalization_2/cond/FusedBatchNorm:4*
epsilon%o�:*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(
�
Igradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
gradients/Switch_18Switchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
g
gradients/Shape_19Shapegradients/Switch_18:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_18/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_18Fillgradients/Shape_19gradients/zeros_18/Const*/
_output_shapes
:���������*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_18*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_19Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_20Shapegradients/Switch_19:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_19/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_19Fillgradients/Shape_20gradients/zeros_19/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_19*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_20Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_21Shapegradients/Switch_20:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_20/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_20Fillgradients/Shape_21gradients/zeros_20/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_20*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_21Switchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
e
gradients/Shape_22Shapegradients/Switch_21*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_21/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_21Fillgradients/Shape_22gradients/zeros_21/Const*/
_output_shapes
:���������*
T0
�
Igradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_21*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_22Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_23Shapegradients/Switch_22*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_22/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_22Fillgradients/Shape_23gradients/zeros_22/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_22*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_23Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_24Shapegradients/Switch_23*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_23/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
m
gradients/zeros_23Fillgradients/Shape_24gradients/zeros_23/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_23*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_12AddNKgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_grad*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N
�
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/AddN_12*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides

�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/AddN_12*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
/gradients/conv2d_2/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_13^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_13AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_14AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_grad*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:*
T0
�
gradients/Relu_grad/ReluGradReluGrad7gradients/conv2d_2/Conv2D_grad/tuple/control_dependencyRelu*
T0*/
_output_shapes
:���������
�
7gradients/batch_normalization/cond/Merge_grad/cond_gradSwitchgradients/Relu_grad/ReluGrad batch_normalization/cond/pred_id*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
>gradients/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_18^gradients/batch_normalization/cond/Merge_grad/cond_grad
�
Fgradients/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentity7gradients/batch_normalization/cond/Merge_grad/cond_grad?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:���������*
T0
�
Hgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_1Identity9gradients/batch_normalization/cond/Merge_grad/cond_grad:1?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:���������
�
gradients/zeros_like_32	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_33	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_34	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_35	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradFgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0
�
Igradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1L^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityKgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradJ^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1J^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2J^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
gradients/zeros_like_36	ZerosLike)batch_normalization/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_37	ZerosLike)batch_normalization/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_38	ZerosLike)batch_normalization/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_39	ZerosLike)batch_normalization/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Igradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_10batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:1)batch_normalization/cond/FusedBatchNorm:3)batch_normalization/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:
�
Ggradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1J^gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Ogradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityIgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradH^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
gradients/Switch_24Switchconv2d/Conv2D batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
g
gradients/Shape_25Shapegradients/Switch_24:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_24/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
�
gradients/zeros_24Fillgradients/Shape_25gradients/zeros_24/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_24*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_25Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
g
gradients/Shape_26Shapegradients/Switch_25:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_25/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_25Fillgradients/Shape_26gradients/zeros_25/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_25*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_26Switchbatch_normalization/beta/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_27Shapegradients/Switch_26:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_26/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_26Fillgradients/Shape_27gradients/zeros_26/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_26*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_27Switchconv2d/Conv2D batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
e
gradients/Shape_28Shapegradients/Switch_27*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_27/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
�
gradients/zeros_27Fillgradients/Shape_28gradients/zeros_27/Const*/
_output_shapes
:���������*
T0
�
Ggradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeOgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_27*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_28Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_29Shapegradients/Switch_28*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_28/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_28Fillgradients/Shape_29gradients/zeros_28/Const*
T0*
_output_shapes
:
�
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_28*
_output_shapes

:: *
T0*
N
�
gradients/Switch_29Switchbatch_normalization/beta/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_30Shapegradients/Switch_29*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_29/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_29Fillgradients/Shape_30gradients/zeros_29/Const*
T0*
_output_shapes
:
�
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_29*
_output_shapes

:: *
T0*
N
�
gradients/AddN_15AddNIgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradGgradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������
�
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
T0*
out_type0*
N* 
_output_shapes
::
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/AddN_15*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/AddN_15*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_11^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_16AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_17AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
GradientDescent/learning_rateConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1*
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
EGradientDescent/update_batch_normalization/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization/gammaGradientDescent/learning_rategradients/AddN_16*
use_locking( *
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:
�
DGradientDescent/update_batch_normalization/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization/betaGradientDescent/learning_rategradients/AddN_17*
use_locking( *
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:*
use_locking( 
�
GGradientDescent/update_batch_normalization_1/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_1/gammaGradientDescent/learning_rategradients/AddN_13*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
�
FGradientDescent/update_batch_normalization_1/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_1/betaGradientDescent/learning_rategradients/AddN_14*
_output_shapes
:*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
GGradientDescent/update_batch_normalization_2/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_2/gammaGradientDescent/learning_rategradients/AddN_10*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:*
use_locking( *
T0
�
FGradientDescent/update_batch_normalization_2/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_2/betaGradientDescent/learning_rategradients/AddN_11*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
�
;GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentApplyGradientDescentconv2d_3/kernelGradientDescent/learning_rate9gradients/conv2d_4/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
GGradientDescent/update_batch_normalization_3/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_3/gammaGradientDescent/learning_rategradients/AddN_7*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:
�
FGradientDescent/update_batch_normalization_3/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_3/betaGradientDescent/learning_rategradients/AddN_8*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:*
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
GGradientDescent/update_batch_normalization_4/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_4/gammaGradientDescent/learning_rategradients/AddN_4*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:*
use_locking( *
T0
�
FGradientDescent/update_batch_normalization_4/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_4/betaGradientDescent/learning_rategradients/AddN_5*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:*
use_locking( *
T0
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	�d*
use_locking( 
�
GGradientDescent/update_batch_normalization_5/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_5/gammaGradientDescent/learning_rateMgradients/batch_normalization_6/batchnorm/mul_grad/tuple/control_dependency_1*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
:d*
use_locking( *
T0
�
FGradientDescent/update_batch_normalization_5/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_5/betaGradientDescent/learning_rateKgradients/batch_normalization_6/batchnorm/sub_grad/tuple/control_dependency*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes
:d*
use_locking( *
T0
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
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
_output_shapes
:
*
use_locking( *
T0
�
GradientDescentNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1&^batch_normalization_5/AssignMovingAvg(^batch_normalization_5/AssignMovingAvg_1&^batch_normalization_6/AssignMovingAvg(^batch_normalization_6/AssignMovingAvg_1:^GradientDescent/update_conv2d/kernel/ApplyGradientDescentF^GradientDescent/update_batch_normalization/gamma/ApplyGradientDescentE^GradientDescent/update_batch_normalization/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_1/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_1/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_2/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_2/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_3/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_3/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_3/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_4/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_4/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_4/beta/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_5/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_5/beta/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
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
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
: ""�
trainable_variables��
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
�
batch_normalization/gamma:0 batch_normalization/gamma/Assign batch_normalization/gamma/read:02,batch_normalization/gamma/Initializer/ones:0
�
batch_normalization/beta:0batch_normalization/beta/Assignbatch_normalization/beta/read:02,batch_normalization/beta/Initializer/zeros:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
�
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:02.batch_normalization_1/gamma/Initializer/ones:0
�
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:02.batch_normalization_1/beta/Initializer/zeros:0
q
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:0
�
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:02.batch_normalization_2/gamma/Initializer/ones:0
�
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:02.batch_normalization_2/beta/Initializer/zeros:0
q
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02,conv2d_3/kernel/Initializer/random_uniform:0
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02.batch_normalization_3/gamma/Initializer/ones:0
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02.batch_normalization_3/beta/Initializer/zeros:0
q
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02,conv2d_4/kernel/Initializer/random_uniform:0
�
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign"batch_normalization_4/gamma/read:02.batch_normalization_4/gamma/Initializer/ones:0
�
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign!batch_normalization_4/beta/read:02.batch_normalization_4/beta/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
�
batch_normalization_5/gamma:0"batch_normalization_5/gamma/Assign"batch_normalization_5/gamma/read:02.batch_normalization_5/gamma/Initializer/ones:0
�
batch_normalization_5/beta:0!batch_normalization_5/beta/Assign!batch_normalization_5/beta/read:02.batch_normalization_5/beta/Initializer/zeros:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"
	summaries

conv_loss:0"�&
	variables�&�&
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
�
batch_normalization/gamma:0 batch_normalization/gamma/Assign batch_normalization/gamma/read:02,batch_normalization/gamma/Initializer/ones:0
�
batch_normalization/beta:0batch_normalization/beta/Assignbatch_normalization/beta/read:02,batch_normalization/beta/Initializer/zeros:0
�
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign&batch_normalization/moving_mean/read:023batch_normalization/moving_mean/Initializer/zeros:0
�
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign*batch_normalization/moving_variance/read:026batch_normalization/moving_variance/Initializer/ones:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
�
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:02.batch_normalization_1/gamma/Initializer/ones:0
�
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:02.batch_normalization_1/beta/Initializer/zeros:0
�
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign(batch_normalization_1/moving_mean/read:025batch_normalization_1/moving_mean/Initializer/zeros:0
�
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign,batch_normalization_1/moving_variance/read:028batch_normalization_1/moving_variance/Initializer/ones:0
q
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:0
�
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:02.batch_normalization_2/gamma/Initializer/ones:0
�
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:02.batch_normalization_2/beta/Initializer/zeros:0
�
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign(batch_normalization_2/moving_mean/read:025batch_normalization_2/moving_mean/Initializer/zeros:0
�
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign,batch_normalization_2/moving_variance/read:028batch_normalization_2/moving_variance/Initializer/ones:0
q
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02,conv2d_3/kernel/Initializer/random_uniform:0
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02.batch_normalization_3/gamma/Initializer/ones:0
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02.batch_normalization_3/beta/Initializer/zeros:0
�
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign(batch_normalization_3/moving_mean/read:025batch_normalization_3/moving_mean/Initializer/zeros:0
�
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign,batch_normalization_3/moving_variance/read:028batch_normalization_3/moving_variance/Initializer/ones:0
q
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02,conv2d_4/kernel/Initializer/random_uniform:0
�
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign"batch_normalization_4/gamma/read:02.batch_normalization_4/gamma/Initializer/ones:0
�
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign!batch_normalization_4/beta/read:02.batch_normalization_4/beta/Initializer/zeros:0
�
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign(batch_normalization_4/moving_mean/read:025batch_normalization_4/moving_mean/Initializer/zeros:0
�
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign,batch_normalization_4/moving_variance/read:028batch_normalization_4/moving_variance/Initializer/ones:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
�
batch_normalization_5/gamma:0"batch_normalization_5/gamma/Assign"batch_normalization_5/gamma/read:02.batch_normalization_5/gamma/Initializer/ones:0
�
batch_normalization_5/beta:0!batch_normalization_5/beta/Assign!batch_normalization_5/beta/read:02.batch_normalization_5/beta/Initializer/zeros:0
�
#batch_normalization_5/moving_mean:0(batch_normalization_5/moving_mean/Assign(batch_normalization_5/moving_mean/read:025batch_normalization_5/moving_mean/Initializer/zeros:0
�
'batch_normalization_5/moving_variance:0,batch_normalization_5/moving_variance/Assign,batch_normalization_5/moving_variance/read:028batch_normalization_5/moving_variance/Initializer/ones:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"
train_op

GradientDescent"�[
cond_context�[�[
�
"batch_normalization/cond/cond_text"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_t:0 *�
batch_normalization/beta/read:0
 batch_normalization/cond/Const:0
"batch_normalization/cond/Const_1:0
0batch_normalization/cond/FusedBatchNorm/Switch:1
2batch_normalization/cond/FusedBatchNorm/Switch_1:1
2batch_normalization/cond/FusedBatchNorm/Switch_2:1
)batch_normalization/cond/FusedBatchNorm:0
)batch_normalization/cond/FusedBatchNorm:1
)batch_normalization/cond/FusedBatchNorm:2
)batch_normalization/cond/FusedBatchNorm:3
)batch_normalization/cond/FusedBatchNorm:4
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_t:0
 batch_normalization/gamma/read:0
conv2d/Conv2D:0V
 batch_normalization/gamma/read:02batch_normalization/cond/FusedBatchNorm/Switch_1:1U
batch_normalization/beta/read:02batch_normalization/cond/FusedBatchNorm/Switch_2:1C
conv2d/Conv2D:00batch_normalization/cond/FusedBatchNorm/Switch:1
�

$batch_normalization/cond/cond_text_1"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_f:0*�	
batch_normalization/beta/read:0
2batch_normalization/cond/FusedBatchNorm_1/Switch:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_1:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_2:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_3:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_4:0
+batch_normalization/cond/FusedBatchNorm_1:0
+batch_normalization/cond/FusedBatchNorm_1:1
+batch_normalization/cond/FusedBatchNorm_1:2
+batch_normalization/cond/FusedBatchNorm_1:3
+batch_normalization/cond/FusedBatchNorm_1:4
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_f:0
 batch_normalization/gamma/read:0
&batch_normalization/moving_mean/read:0
*batch_normalization/moving_variance/read:0
conv2d/Conv2D:0X
 batch_normalization/gamma/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_1:0W
batch_normalization/beta/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_2:0E
conv2d/Conv2D:02batch_normalization/cond/FusedBatchNorm_1/Switch:0^
&batch_normalization/moving_mean/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_3:0b
*batch_normalization/moving_variance/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_4:0
�
$batch_normalization_2/cond/cond_text$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_t:0 *�
!batch_normalization_1/beta/read:0
"batch_normalization_1/gamma/read:0
"batch_normalization_2/cond/Const:0
$batch_normalization_2/cond/Const_1:0
2batch_normalization_2/cond/FusedBatchNorm/Switch:1
4batch_normalization_2/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_2/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_2/cond/FusedBatchNorm:0
+batch_normalization_2/cond/FusedBatchNorm:1
+batch_normalization_2/cond/FusedBatchNorm:2
+batch_normalization_2/cond/FusedBatchNorm:3
+batch_normalization_2/cond/FusedBatchNorm:4
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_t:0
conv2d_2/Conv2D:0G
conv2d_2/Conv2D:02batch_normalization_2/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_1/gamma/read:04batch_normalization_2/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_1/beta/read:04batch_normalization_2/cond/FusedBatchNorm/Switch_2:1
�

&batch_normalization_2/cond/cond_text_1$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_f:0*�	
!batch_normalization_1/beta/read:0
"batch_normalization_1/gamma/read:0
(batch_normalization_1/moving_mean/read:0
,batch_normalization_1/moving_variance/read:0
4batch_normalization_2/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_2/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_2/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_2/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_2/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_2/cond/FusedBatchNorm_1:0
-batch_normalization_2/cond/FusedBatchNorm_1:1
-batch_normalization_2/cond/FusedBatchNorm_1:2
-batch_normalization_2/cond/FusedBatchNorm_1:3
-batch_normalization_2/cond/FusedBatchNorm_1:4
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_f:0
conv2d_2/Conv2D:0b
(batch_normalization_1/moving_mean/read:06batch_normalization_2/cond/FusedBatchNorm_1/Switch_3:0f
,batch_normalization_1/moving_variance/read:06batch_normalization_2/cond/FusedBatchNorm_1/Switch_4:0I
conv2d_2/Conv2D:04batch_normalization_2/cond/FusedBatchNorm_1/Switch:0\
"batch_normalization_1/gamma/read:06batch_normalization_2/cond/FusedBatchNorm_1/Switch_1:0[
!batch_normalization_1/beta/read:06batch_normalization_2/cond/FusedBatchNorm_1/Switch_2:0
�
$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *�
!batch_normalization_2/beta/read:0
"batch_normalization_2/gamma/read:0
"batch_normalization_3/cond/Const:0
$batch_normalization_3/cond/Const_1:0
2batch_normalization_3/cond/FusedBatchNorm/Switch:1
4batch_normalization_3/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_3/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_3/cond/FusedBatchNorm:0
+batch_normalization_3/cond/FusedBatchNorm:1
+batch_normalization_3/cond/FusedBatchNorm:2
+batch_normalization_3/cond/FusedBatchNorm:3
+batch_normalization_3/cond/FusedBatchNorm:4
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0
conv2d_3/Conv2D:0Y
!batch_normalization_2/beta/read:04batch_normalization_3/cond/FusedBatchNorm/Switch_2:1G
conv2d_3/Conv2D:02batch_normalization_3/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_2/gamma/read:04batch_normalization_3/cond/FusedBatchNorm/Switch_1:1
�

&batch_normalization_3/cond/cond_text_1$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_f:0*�	
!batch_normalization_2/beta/read:0
"batch_normalization_2/gamma/read:0
(batch_normalization_2/moving_mean/read:0
,batch_normalization_2/moving_variance/read:0
4batch_normalization_3/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_3/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_3/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_3/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_3/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_3/cond/FusedBatchNorm_1:0
-batch_normalization_3/cond/FusedBatchNorm_1:1
-batch_normalization_3/cond/FusedBatchNorm_1:2
-batch_normalization_3/cond/FusedBatchNorm_1:3
-batch_normalization_3/cond/FusedBatchNorm_1:4
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_f:0
conv2d_3/Conv2D:0b
(batch_normalization_2/moving_mean/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_3:0f
,batch_normalization_2/moving_variance/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_4:0[
!batch_normalization_2/beta/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_2:0I
conv2d_3/Conv2D:04batch_normalization_3/cond/FusedBatchNorm_1/Switch:0\
"batch_normalization_2/gamma/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_1:0
�
$batch_normalization_4/cond/cond_text$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_t:0 *�
!batch_normalization_3/beta/read:0
"batch_normalization_3/gamma/read:0
"batch_normalization_4/cond/Const:0
$batch_normalization_4/cond/Const_1:0
2batch_normalization_4/cond/FusedBatchNorm/Switch:1
4batch_normalization_4/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_4/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_4/cond/FusedBatchNorm:0
+batch_normalization_4/cond/FusedBatchNorm:1
+batch_normalization_4/cond/FusedBatchNorm:2
+batch_normalization_4/cond/FusedBatchNorm:3
+batch_normalization_4/cond/FusedBatchNorm:4
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_t:0
conv2d_4/Conv2D:0G
conv2d_4/Conv2D:02batch_normalization_4/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_3/gamma/read:04batch_normalization_4/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_3/beta/read:04batch_normalization_4/cond/FusedBatchNorm/Switch_2:1
�

&batch_normalization_4/cond/cond_text_1$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_f:0*�	
!batch_normalization_3/beta/read:0
"batch_normalization_3/gamma/read:0
(batch_normalization_3/moving_mean/read:0
,batch_normalization_3/moving_variance/read:0
4batch_normalization_4/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_4/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_4/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_4/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_4/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_4/cond/FusedBatchNorm_1:0
-batch_normalization_4/cond/FusedBatchNorm_1:1
-batch_normalization_4/cond/FusedBatchNorm_1:2
-batch_normalization_4/cond/FusedBatchNorm_1:3
-batch_normalization_4/cond/FusedBatchNorm_1:4
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_f:0
conv2d_4/Conv2D:0\
"batch_normalization_3/gamma/read:06batch_normalization_4/cond/FusedBatchNorm_1/Switch_1:0[
!batch_normalization_3/beta/read:06batch_normalization_4/cond/FusedBatchNorm_1/Switch_2:0b
(batch_normalization_3/moving_mean/read:06batch_normalization_4/cond/FusedBatchNorm_1/Switch_3:0f
,batch_normalization_3/moving_variance/read:06batch_normalization_4/cond/FusedBatchNorm_1/Switch_4:0I
conv2d_4/Conv2D:04batch_normalization_4/cond/FusedBatchNorm_1/Switch:0
�
$batch_normalization_5/cond/cond_text$batch_normalization_5/cond/pred_id:0%batch_normalization_5/cond/switch_t:0 *�
!batch_normalization_4/beta/read:0
"batch_normalization_4/gamma/read:0
"batch_normalization_5/cond/Const:0
$batch_normalization_5/cond/Const_1:0
2batch_normalization_5/cond/FusedBatchNorm/Switch:1
4batch_normalization_5/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_5/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_5/cond/FusedBatchNorm:0
+batch_normalization_5/cond/FusedBatchNorm:1
+batch_normalization_5/cond/FusedBatchNorm:2
+batch_normalization_5/cond/FusedBatchNorm:3
+batch_normalization_5/cond/FusedBatchNorm:4
$batch_normalization_5/cond/pred_id:0
%batch_normalization_5/cond/switch_t:0
conv2d_5/Conv2D:0G
conv2d_5/Conv2D:02batch_normalization_5/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_4/gamma/read:04batch_normalization_5/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_4/beta/read:04batch_normalization_5/cond/FusedBatchNorm/Switch_2:1
�

&batch_normalization_5/cond/cond_text_1$batch_normalization_5/cond/pred_id:0%batch_normalization_5/cond/switch_f:0*�	
!batch_normalization_4/beta/read:0
"batch_normalization_4/gamma/read:0
(batch_normalization_4/moving_mean/read:0
,batch_normalization_4/moving_variance/read:0
4batch_normalization_5/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_5/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_5/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_5/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_5/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_5/cond/FusedBatchNorm_1:0
-batch_normalization_5/cond/FusedBatchNorm_1:1
-batch_normalization_5/cond/FusedBatchNorm_1:2
-batch_normalization_5/cond/FusedBatchNorm_1:3
-batch_normalization_5/cond/FusedBatchNorm_1:4
$batch_normalization_5/cond/pred_id:0
%batch_normalization_5/cond/switch_f:0
conv2d_5/Conv2D:0f
,batch_normalization_4/moving_variance/read:06batch_normalization_5/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_4/moving_mean/read:06batch_normalization_5/cond/FusedBatchNorm_1/Switch_3:0[
!batch_normalization_4/beta/read:06batch_normalization_5/cond/FusedBatchNorm_1/Switch_2:0I
conv2d_5/Conv2D:04batch_normalization_5/cond/FusedBatchNorm_1/Switch:0\
"batch_normalization_4/gamma/read:06batch_normalization_5/cond/FusedBatchNorm_1/Switch_1:0"�

update_ops�
�
%batch_normalization/AssignMovingAvg:0
'batch_normalization/AssignMovingAvg_1:0
'batch_normalization_2/AssignMovingAvg:0
)batch_normalization_2/AssignMovingAvg_1:0
'batch_normalization_3/AssignMovingAvg:0
)batch_normalization_3/AssignMovingAvg_1:0
'batch_normalization_4/AssignMovingAvg:0
)batch_normalization_4/AssignMovingAvg_1:0
'batch_normalization_5/AssignMovingAvg:0
)batch_normalization_5/AssignMovingAvg_1:0
'batch_normalization_6/AssignMovingAvg:0
)batch_normalization_6/AssignMovingAvg_1:0��+$       `/�#	�g����A*

	conv_loss#�;?���       QKD	y<o����A*

	conv_loss:�:?�ey        QKD	�o����A*

	conv_loss.v>?�N�M       QKD	-�o����A*

	conv_lossB?�\x       QKD	ap����A*

	conv_loss�D?�}��       QKD	~Hp����A*

	conv_lossw�<?۬K�       QKD	ҙp����A*

	conv_lossŴ8?���       QKD	��p����A*

	conv_loss��;?Z�@~       QKD	r)q����A*

	conv_lossA�;?�vێ       QKD	�jq����A	*

	conv_loss��<? ˗�       QKD	��q����A
*

	conv_loss�^<?��&�       QKD	�q����A*

	conv_losswu<?�zfC       QKD	[:r����A*

	conv_loss�[=?�6�       QKD		�r����A*

	conv_loss�j:?j�       QKD	��r����A*

	conv_loss��;?c��,       QKD	s����A*

	conv_loss��8?-       QKD	�Ks����A*

	conv_lossB94?����       QKD	��s����A*

	conv_loss�:?�y2       QKD	�s����A*

	conv_loss-�6?��k       QKD	�t����A*

	conv_loss�#<?d_C-       QKD	St����A*

	conv_loss�6<?�i'a       QKD	ϔt����A*

	conv_loss��:?���I       QKD	L�t����A*

	conv_loss�
<? s       QKD	�"u����A*

	conv_loss� 5?��Լ       QKD	�eu����A*

	conv_loss�S<?��>       QKD	��u����A*

	conv_loss&\7?&��       QKD	�u����A*

	conv_loss>d:?�t       QKD	�+v����A*

	conv_lossrp9?K�	       QKD	Rlv����A*

	conv_lossU67?Z��|       QKD	��v����A*

	conv_lossA57?�w       QKD	"�v����A*

	conv_loss+g8?��H       QKD	�0w����A*

	conv_lossE�3?�We       QKD	\sw����A *

	conv_loss8?�T�6       QKD	J�w����A!*

	conv_loss��8?]��^       QKD	u�w����A"*

	conv_lossI�5?%�)       QKD	Z9x����A#*

	conv_loss�45?��	       QKD	�zx����A$*

	conv_loss�]/?����       QKD	ܼx����A%*

	conv_lossg6?��e`       QKD	��x����A&*

	conv_loss/�4?c"��       QKD	�?y����A'*

	conv_loss�P3?�r       QKD	ށy����A(*

	conv_loss"�6?���       QKD	.�y����A)*

	conv_loss�5?c'�       QKD	Gz����A**

	conv_loss��/?�       QKD	nEz����A+*

	conv_loss5�6?@�_       QKD	��z����A,*

	conv_loss#�2?��|       QKD	p�z����A-*

	conv_lossu�5?NOX�       QKD	�z����A.*

	conv_loss��4?���       QKD	�;{����A/*

	conv_loss(w2?"9��       QKD	�y{����A0*

	conv_lossѠ0?�E�*       QKD	H�{����A1*

	conv_loss��4?�X2       QKD	U�{����A2*

	conv_loss%o7?ҧ��       QKD	�J|����A3*

	conv_loss΋2?�p�       QKD	��|����A4*

	conv_loss��6?֓�\       QKD	��|����A5*

	conv_loss��0?˦��       QKD	�}����A6*

	conv_lossE6?T�n�       QKD	�R}����A7*

	conv_loss�4?�[�,       QKD	��}����A8*

	conv_loss�)0?
��       QKD	E�}����A9*

	conv_loss�/?�tV       QKD	�&~����A:*

	conv_lossڗ1?�)q       QKD	h~����A;*

	conv_loss0.?T ��       QKD	��~����A<*

	conv_lossT�.?�|W       QKD	��~����A=*

	conv_lossy�/?inG       QKD	�-����A>*

	conv_loss/?����       QKD	�n����A?*

	conv_loss��2?}x��       QKD	������A@*

	conv_loss�b1?rI�       QKD	:�����AA*

	conv_loss�)/?W���       QKD	d,�����AB*

	conv_loss�).?�YIj       QKD	Tk�����AC*

	conv_loss��1?䒽�       QKD	������AD*

	conv_loss��)?���p       QKD	逐���AE*

	conv_loss��2?����       QKD	�&�����AF*

	conv_lossB-,?(Y!+       QKD	�c�����AG*

	conv_loss�4?ߔ��       QKD	m������AH*

	conv_lossm�2?��v       QKD	�������AI*

	conv_lossr�-?�J�m       QKD	������AJ*

	conv_loss�)?�8�}       QKD	`�����AK*

	conv_loss��+?�s       QKD	枂����AL*

	conv_lossT�.?�b~�       QKD	@߂����AM*

	conv_lossOo5?��1|       QKD	�����AN*

	conv_loss�C.?u��j       QKD	�Z�����AO*

	conv_lossey+?Tp%�       QKD	������AP*

	conv_losss -?�n+�       QKD	R؃����AQ*

	conv_lossu*?��;{       QKD	K�����AR*

	conv_lossI*?V9i       QKD	�R�����AS*

	conv_losso�-?fN��       QKD	�������AT*

	conv_loss�,?��       QKD	΄����AU*

	conv_loss�i)?�	��       QKD	�����AV*

	conv_loss��*?s�1       QKD	�J�����AW*

	conv_lossp�2?�<�       QKD	9������AX*

	conv_lossЛ-?lz��       QKD	�ƅ����AY*

	conv_loss* +?�@��       QKD	"�����AZ*

	conv_loss%�*?�s�       QKD	�>�����A[*

	conv_loss�s)?k��       QKD	]������A\*

	conv_loss;�)?i�i-       QKD	y̆����A]*

	conv_loss)?=O^B       QKD	�	�����A^*

	conv_loss��)?��       QKD	�F�����A_*

	conv_losse�(?�p�]       QKD	*������A`*

	conv_loss9K)?���       QKD	�����Aa*

	conv_loss=.?�C       QKD	a �����Ab*

	conv_loss�&?s[�       QKD	�=�����Ac*

	conv_loss�H)?�HO       QKD	1{�����Ad*

	conv_loss�O)?�(�       QKD	@̈����Ae*

	conv_loss�6%?*Ұ�       QKD	������Af*

	conv_loss�_+?��[       QKD	�_�����Ag*

	conv_loss�}(?f�       QKD	�������Ah*

	conv_loss�(?SY��       QKD	Vމ����Ai*

	conv_loss��&?)C�a       QKD	������Aj*

	conv_loss�T$?_,ݷ       QKD	�f�����Ak*

	conv_loss�+?F�       QKD	�������Al*

	conv_loss?y'?>���       QKD	�犐���Am*

	conv_loss�~)?� <�       QKD	B)�����An*

	conv_loss��)?���[       QKD	 r�����Ao*

	conv_loss��%?�:�U       QKD	F������Ap*

	conv_loss�%?�մ       QKD	�������Aq*

	conv_lossԽ$?o}J�       QKD	Y4�����Ar*

	conv_loss!?�?;�       QKD	hq�����As*

	conv_loss�%?vd��       QKD	A������At*

	conv_loss� %?�^i�       QKD	�쌐���Au*

	conv_loss��$?�dUM       QKD	)�����Av*

	conv_loss�#?���       QKD	Pf�����Aw*

	conv_loss�%(?��PY       QKD	㤍����Ax*

	conv_loss��"?�x	�       QKD	�㍐���Ay*

	conv_loss��%?ύ��       QKD	r�����Az*

	conv_loss3?�#��       QKD	$\�����A{*

	conv_loss�$?�F��       QKD	������A|*

	conv_loss"�!?	 [�       QKD	�ڎ����A}*

	conv_loss�#?X�x:       QKD	I�����A~*

	conv_loss��$?���t       QKD	�X�����A*

	conv_loss[�#?@�"�        )��P	�������A�*

	conv_loss� ?g`�        )��P	�Տ����A�*

	conv_loss�5 ?���]        )��P	������A�*

	conv_loss�+$?��p        )��P	8R�����A�*

	conv_loss�"?A]Қ        )��P	菐����A�*

	conv_loss�^"?�	�.        )��P	eΐ����A�*

	conv_loss/s#?6K\        )��P	������A�*

	conv_lossO�(?Fc��        )��P	�I�����A�*

	conv_loss�x!?��:�        )��P	�������A�*

	conv_loss�<"?��z�        )��P	đ����A�*

	conv_loss� ?'_�        )��P	�����A�*

	conv_lossF6$?4�5        )��P	�=�����A�*

	conv_lossG+%?�}��        )��P	c}�����A�*

	conv_lossp�"?$/2*        )��P	[������A�*

	conv_lossƻ&?��	Y        )��P	������A�*

	conv_loss/ ?MBJ        )��P	7�����A�*

	conv_loss�$?��h        )��P	Ct�����A�*

	conv_loss��!?���h        )��P	w������A�*

	conv_lossɃ?���        )��P	����A�*

	conv_loss�$?��K�        )��P	�+�����A�*

	conv_loss5#?�_�(        )��P	�k�����A�*

	conv_loss�?�;T�        )��P	�������A�*

	conv_lossUY ?wݢ/        )��P	�攐���A�*

	conv_loss�!? }j�        )��P	9�����A�*

	conv_loss��?]K�H        )��P	�w�����A�*

	conv_loss��?ق��        )��P	�������A�*

	conv_loss��?���        )��P	�������A�*

	conv_loss��?�_�        )��P	I8�����A�*

	conv_lossEj ?6/�        )��P	�v�����A�*

	conv_lossߪ?9�k        )��P	�������A�*

	conv_loss��!?�^�        )��P	�����A�*

	conv_loss��?��"        )��P	?B�����A�*

	conv_loss3!?����        )��P	,�����A�*

	conv_loss�/?M?ze        )��P	������A�*

	conv_lossC�!?��        )��P	������A�*

	conv_loss��?�(�        )��P	L�����A�*

	conv_loss��?�C�\        )��P	�������A�*

	conv_loss�_?z�        )��P		ɘ����A�*

	conv_loss��?
��R        )��P	������A�*

	conv_loss�?����        )��P	�D�����A�*

	conv_loss3I"?�B�:        )��P	�������A�*

	conv_loss&?,�        )��P		����A�*

	conv_loss=�?�r�        )��P	�������A�*

	conv_lossɝ?��9�        )��P	�>�����A�*

	conv_loss?-(�R        )��P	|�����A�*

	conv_lossz
?9��        )��P	;������A�*

	conv_loss�s?�R;|        )��P	������A�*

	conv_loss(A?u/*        )��P	�5�����A�*

	conv_loss��?a��[        )��P	�t�����A�*

	conv_loss:1?���        )��P	F������A�*

	conv_lossI�?���        )��P	�񛐉��A�*

	conv_loss�|?�?4�        )��P	Y1�����A�*

	conv_loss�?�t
�        )��P	3n�����A�*

	conv_loss4
?iO�        )��P	j������A�*

	conv_loss�?K�        )��P	|윐���A�*

	conv_loss��?�N        )��P	*�����A�*

	conv_loss
%?D�>�        )��P	�h�����A�*

	conv_loss<�?��Q�        )��P	�������A�*

	conv_loss]l?��!�        )��P	R䝐���A�*

	conv_loss�?�O��        )��P	L"�����A�*

	conv_loss�?�#��        )��P	T^�����A�*

	conv_lossں?�^        )��P	������A�*

	conv_loss��?�@`        )��P	�ٞ����A�*

	conv_lossB�?���        )��P	������A�*

	conv_lossv?7�[        )��P	�T�����A�*

	conv_lossw?��!        )��P	Z������A�*

	conv_loss�x?�1��        )��P	�ҟ����A�*

	conv_loss��?�"vC        )��P	������A�*

	conv_lossp�?���        )��P	4M�����A�*

	conv_loss�w?KЄ        )��P	�������A�*

	conv_loss\y?!��}        )��P	3ɠ����A�*

	conv_loss
�?��W�        )��P	������A�*

	conv_loss�?Xl�        )��P	oG�����A�*

	conv_loss�?�K!�        )��P	�Y�����A�*

	conv_loss�c?U�ۜ        )��P	v������A�*

	conv_loss��?K��8        )��P	�ң����A�*

	conv_lossC?gw!�        )��P	������A�*

	conv_loss��?�k        )��P	,s�����A�*

	conv_lossm�?�&A�        )��P	¤����A�*

	conv_lossg]?�=��        )��P	������A�*

	conv_lossѲ?+W�J        )��P	�N�����A�*

	conv_lossCT?�Lۙ        )��P	�������A�*

	conv_losse!?�P85        )��P	�˥����A�*

	conv_loss^?4���        )��P	�����A�*

	conv_losskC?vN�F        )��P	�D�����A�*

	conv_loss̙?���        )��P	ԁ�����A�*

	conv_loss�*?Ě�        )��P	Ծ�����A�*

	conv_loss��?9�1�        )��P	�����A�*

	conv_loss�G?b�        )��P	I�����A�*

	conv_loss��?>�û        )��P	3������A�*

	conv_loss�8?}�g        )��P	�ʧ����A�*

	conv_lossR\?ã        )��P	t�����A�*

	conv_lossew?��-1        )��P	�C�����A�*

	conv_lossB�?۹a9        )��P	
������A�*

	conv_loss��?�<7        )��P	������A�*

	conv_loss�?n���        )��P	a������A�*

	conv_loss@?�T�        )��P	
5�����A�*

	conv_losss�?\�x        )��P	1s�����A�*

	conv_lossE_?���        )��P	L������A�*

	conv_loss��?��҈        )��P	����A�*

	conv_lossO?��        )��P	�,�����A�*

	conv_loss�?w�>�        )��P	�i�����A�*

	conv_loss*�?'���        )��P	ɦ�����A�*

	conv_loss��?#p�X        )��P	_㪐���A�*

	conv_loss�Q?�O��        )��P	�"�����A�*

	conv_loss�L?�V�        )��P	w`�����A�*

	conv_loss��?la�c        )��P	�������A�*

	conv_losswG?�!&h        )��P	�٫����A�*

	conv_loss6?�{�p        )��P	������A�*

	conv_loss��?r;��        )��P	�V�����A�*

	conv_loss��?%O�        )��P	=������A�*

	conv_loss8�?�t~        )��P	�ά����A�*

	conv_loss1?����        )��P	W�����A�*

	conv_loss�?�!�        )��P	�K�����A�*

	conv_loss��?`ݯ�        )��P	�������A�*

	conv_losssU?)��y        )��P	�ŭ����A�*

	conv_loss-f?�Y~        )��P	Y�����A�*

	conv_lossn�?\e�Q        )��P	/@�����A�*

	conv_loss�q?����        )��P	�}�����A�*

	conv_loss,?�s�+        )��P	?������A�*

	conv_loss�?Q�\?        )��P	�������A�*

	conv_loss�P?;���        )��P	D7�����A�*

	conv_loss&?h�T�        )��P	�t�����A�*

	conv_loss��?�	�k        )��P	YƯ����A�*

	conv_loss��?�}�?        )��P	������A�*

	conv_loss�?C<�        )��P	�?�����A�*

	conv_loss��?��:�        )��P	^������A�*

	conv_loss:,?C��        )��P	�°����A�*

	conv_loss�?���        )��P	c�����A�*

	conv_loss�N?H���        )��P	�J�����A�*

	conv_lossY7?wx�        )��P	7������A�*

	conv_lossy�?��c4        )��P	ر����A�*

	conv_loss	�?�q��        )��P	�����A�*

	conv_loss�?���        )��P	�S�����A�*

	conv_loss�?*��!        )��P	㛲����A�*

	conv_loss�>?����        )��P	ٲ����A�*

	conv_loss��?�}h�        )��P	�����A�*

	conv_loss��?���        )��P	U�����A�*

	conv_loss.A?\i��        )��P	嘳����A�*

	conv_lossH?���        )��P	�Ⳑ���A�*

	conv_loss��?���        )��P	�#�����A�*

	conv_loss�M?�}        )��P	kb�����A�*

	conv_loss�?eK        )��P	u������A�*

	conv_losst?��*        )��P	J޴����A�*

	conv_loss�=?~^         )��P	������A�*

	conv_loss�c?��Wp        )��P	�W�����A�*

	conv_loss�?���
        )��P	.������A�*

	conv_lossg�	?� +        )��P	�ҵ����A�*

	conv_loss��
?y�.        )��P	������A�*

	conv_loss�?��Q        )��P	CM�����A�*

	conv_loss��?S_�        )��P	j������A�*

	conv_loss��?R�J�        )��P	�ɶ����A�*

	conv_loss��?���        )��P	������A�*

	conv_loss�k?�]�        )��P	�F�����A�*

	conv_loss�~?0A&�        )��P	�������A�*

	conv_loss�"?6�7�        )��P	�������A�*

	conv_lossN�?Q.�"        )��P	�������A�*

	conv_loss��?�j�?        )��P	68�����A�*

	conv_loss�?��P        )��P	u�����A�*

	conv_loss�??t�        )��P	x������A�*

	conv_loss]j?K�}�        )��P	����A�*

	conv_loss(g?��=        )��P	'-�����A�*

	conv_loss�	?s;��        )��P	�k�����A�*

	conv_loss�i?Z���        )��P	�������A�*

	conv_loss<?��:        )��P	n湐���A�*

	conv_lossî
?�J�        )��P	R#�����A�*

	conv_lossj"?_�        )��P	ta�����A�*

	conv_loss)�?5���        )��P	������A�*

	conv_loss�}
?>�aA        )��P	�ۺ����A�*

	conv_loss��
?�"�q        )��P	u�����A�*

	conv_loss�5?:⫝        )��P	�U�����A�*

	conv_lossd?�!.o        )��P	�������A�*

	conv_loss�b?�'��        )��P	*һ����A�*

	conv_loss��?\��k        )��P	+%�����A�*

	conv_loss�?K�I        )��P	�b�����A�*

	conv_loss�?� }�        )��P	�������A�*

	conv_loss��	?|��        )��P	C޼����A�*

	conv_lossbU	?⧨�        )��P	������A�*

	conv_losscI?{�}        )��P	�c�����A�*

	conv_loss��	?�Z'x        )��P	�������A�*

	conv_lossÃ
?�G6        )��P	�ｐ���A�*

	conv_loss�?mmf�        )��P	�,�����A�*

	conv_loss!P?��p�        )��P	�i�����A�*

	conv_loss;?���Q        )��P	�������A�*

	conv_loss�%?�l�s        )��P	�徐���A�*

	conv_loss�R?���        )��P	\#�����A�*

	conv_loss()?F��        )��P	�`�����A�*

	conv_loss��?{�>        )��P	?������A�*

	conv_loss��?�P�        )��P	�쿐���A�*

	conv_lossJ�
?WC��        )��P	d0�����A�*

	conv_lossĊ?�G�        )��P	r�����A�*

	conv_loss�_?"�y�        )��P	�������A�*

	conv_loss��? ��        )��P	u������A�*

	conv_lossİ	?�� R        )��P	g,�����A�*

	conv_lossZ�?ͦN+        )��P	Fj�����A�*

	conv_loss�?4v)�        )��P	������A�*

	conv_lossAN?(�H        )��P	�������A�*

	conv_loss�r?��N        )��P	S#���A�*

	conv_loss�, ?���        )��P	\a���A�*

	conv_loss(?+��_        )��P	�����A�*

	conv_lossF�?p�        )��P	�����A�*

	conv_lossl?v�w�        )��P	�Ð���A�*

	conv_loss.�?c~us        )��P	�ZÐ���A�*

	conv_loss�O?����        )��P	t�Ð���A�*

	conv_lossˡ?H�O�        )��P	��Ð���A�*

	conv_loss̳?���R        )��P	aĐ���A�*

	conv_lossS�?Ӏ�        )��P	PĐ���A�*

	conv_loss3-?zh�        )��P	�Đ���A�*

	conv_loss��?�$��        )��P	�Đ���A�*

	conv_loss�??^q�p        )��P	u
Ő���A�*

	conv_loss�M?$��        )��P	�GŐ���A�*

	conv_loss�?���        )��P	~�Ő���A�*

	conv_lossoH?9�x�        )��P	��Ő���A�*

	conv_loss":?��        )��P	} Ɛ���A�*

	conv_loss�?�o2        )��P	a>Ɛ���A�*

	conv_loss9E?��I        )��P	�{Ɛ���A�*

	conv_loss�
?Mæ�        )��P	��Ɛ���A�*

	conv_losss&?J8�6        )��P	��Ɛ���A�*

	conv_loss� ?J�        )��P	4ǐ���A�*

	conv_loss޵?%^7        )��P	�qǐ���A�*

	conv_lossO�?9(�_        )��P	@�ǐ���A�*

	conv_loss�i?�1cj        )��P	��ǐ���A�*

	conv_loss��?(�        )��P	P-Ȑ���A�*

	conv_loss��?���0        )��P	]Ȑ���A�*

	conv_loss�?o�,�        )��P	~�Ȑ���A�*

	conv_losso� ?�(        )��P	3�Ȑ���A�*

	conv_loss��?��        )��P	�Aɐ���A�*

	conv_loss�V?��^�        )��P	��ɐ���A�*

	conv_lossd�?�Ff        )��P	X�ɐ���A�*

	conv_loss?�R�l        )��P	w	ʐ���A�*

	conv_loss/�?��        )��P	�^ʐ���A�*

	conv_loss��?�}��        )��P	'�ʐ���A�*

	conv_loss�?z�        )��P	��ʐ���A�*

	conv_lossm�?��        )��P	� ː���A�*

	conv_lossԔ?��ݳ        )��P	�^ː���A�*

	conv_loss �?�p�!        )��P	��ː���A�*

	conv_loss��??N�|        )��P	��ː���A�*

	conv_loss?b?�w�        )��P	6̐���A�*

	conv_lossx�??ӑ�        )��P	GU̐���A�*

	conv_lossJU?��        )��P	��̐���A�*

	conv_loss� ?a�W�        )��P	��̐���A�*

	conv_lossg?B�C        )��P	v͐���A�*

	conv_loss� �>)ae        )��P	^͐���A�*

	conv_loss�I?sM��        )��P	7�͐���A�*

	conv_loss�H�>��-        )��P	��͐���A�*

	conv_lossmA?�*c        )��P	!ΐ���A�*

	conv_loss���>Z���        )��P	�Wΐ���A�*

	conv_loss��>���        )��P	�ΐ���A�*

	conv_loss�{?��I�        )��P	3�ΐ���A�*

	conv_loss�'?`���        )��P	Wϐ���A�*

	conv_loss� ?�T�        )��P	�Qϐ���A�*

	conv_loss!��>q�        )��P	 �ϐ���A�*

	conv_loss) ?��8k        )��P	��ϐ���A�*

	conv_lossL�?�Wwh        )��P	�А���A�*

	conv_loss���>w?�b        )��P	+IА���A�*

	conv_loss3O?ɠ        )��P	�А���A�*

	conv_loss�6?<V�l        )��P	��А���A�*

	conv_loss�
 ?�[��        )��P	z ѐ���A�*

	conv_loss׊ ?D�        )��P	�?ѐ���A�*

	conv_loss�
�>J��        )��P	{}ѐ���A�*

	conv_lossA ?�8l        )��P	�ѐ���A�*

	conv_loss���>Neӡ        )��P	\�ѐ���A�*

	conv_loss���>;��        )��P	:Ґ���A�*

	conv_loss�k�>[:X�        )��P	�xҐ���A�*

	conv_loss-?�yw        )��P	�Ґ���A�*

	conv_losst��>a�7p        )��P	��Ґ���A�*

	conv_loss`��>+�C�        )��P	$2Ӑ���A�*

	conv_loss�r�>�R�E        )��P	�oӐ���A�*

	conv_loss�v�>
}ܑ        )��P	�Ӑ���A�*

	conv_loss���>�k,        )��P	��Ӑ���A�*

	conv_loss{n?�c�        )��P	�(Ԑ���A�*

	conv_lossW%?����        )��P	5gԐ���A�*

	conv_loss��>�4/�        )��P	��Ԑ���A�*

	conv_loss�[�>��'�        )��P	h�֐���A�*

	conv_loss_��>��1c        )��P	�א���A�*

	conv_loss1k�>�0=        )��P	Iא���A�*

	conv_loss�I�>�`P�        )��P	��א���A�*

	conv_loss�f�>�,�I        )��P	�א���A�*

	conv_loss���>�]�b        )��P	Oؐ���A�*

	conv_loss�� ?nz6        )��P	�Rؐ���A�*

	conv_losst-�>�Ñ�        )��P	��ؐ���A�*

	conv_loss��?kgb        )��P	
�ؐ���A�*

	conv_lossO_ ?����        )��P	�ِ���A�*

	conv_loss*��>*H�        )��P	�Zِ���A�*

	conv_lossqd ?�;�u        )��P	��ِ���A�*

	conv_loss�_�>�`��        )��P	��ِ���A�*

	conv_loss�)�>�ud5        )��P	�ڐ���A�*

	conv_loss�{�>��        )��P	�]ڐ���A�*

	conv_loss�d�>���        )��P	�ڐ���A�*

	conv_lossh�>�I�        )��P	�ڐ���A�*

	conv_loss��>�cJ�        )��P	ې���A�*

	conv_loss���>�Z'        )��P	*Sې���A�*

	conv_lossQ��>��+�        )��P	�ې���A�*

	conv_loss�{�>9J��        )��P	��ې���A�*

	conv_lossLT�>�v        )��P	�ܐ���A�*

	conv_loss}D�>���        )��P	,Lܐ���A�*

	conv_loss(�>ʼ�8        )��P	W�ܐ���A�*

	conv_loss��>��"C        )��P	W�ܐ���A�*

	conv_lossXF�>��;]        )��P	�ݐ���A�*

	conv_loss�]�>�**4        )��P	'?ݐ���A�*

	conv_loss� �><qc�        )��P	�{ݐ���A�*

	conv_loss2�>չI        )��P	Ʒݐ���A�*

	conv_lossp��>��,�        )��P	j�ݐ���A�*

	conv_loss��>����        )��P	H0ސ���A�*

	conv_loss���>'��        )��P	�mސ���A�*

	conv_loss��>EU)V        )��P	5�ސ���A�*

	conv_loss��>�oݥ        )��P	��ސ���A�*

	conv_loss�1�>�I��        )��P	�&ߐ���A�*

	conv_lossw�>�x )        )��P	�dߐ���A�*

	conv_loss��>���        )��P	�ߐ���A�*

	conv_lossV�>�i��        )��P	�ߐ���A�*

	conv_losse�>Eo��        )��P	4�����A�*

	conv_loss�"�>�;.�        )��P	�\�����A�*

	conv_loss��>7�x        )��P	@������A�*

	conv_loss���>x��        )��P	,������A�*

	conv_loss���>����        )��P	2ᐉ��A�*

	conv_lossҚ�>r��        )��P	�Sᐉ��A�*

	conv_loss���>�|�`        )��P	M�ᐉ��A�*

	conv_loss���>7X�        )��P	��ᐉ��A�*

	conv_lossX��>� �        )��P	t␉��A�*

	conv_loss���>���>        )��P	�H␉��A�*

	conv_loss7��>��t]        )��P	�␉��A�*

	conv_lossĻ�>���        )��P	��␉��A�*

	conv_loss=��>�[��        )��P	�㐉��A�*

	conv_loss��>�t�1        )��P	q㐉��A�*

	conv_loss��>��@}        )��P	l�㐉��A�*

	conv_loss=��>�Z,�        )��P	��㐉��A�*

	conv_lossK�>R�k�        )��P	(2䐉��A�*

	conv_loss���>�T�4        )��P	�}䐉��A�*

	conv_loss�m�>���        )��P	��䐉��A�*

	conv_lossHA�>Y�#        )��P	=吉��A�*

	conv_loss��>
���        )��P	�C吉��A�*

	conv_loss�6�>%�hf        )��P	�吉��A�*

	conv_loss't�>�/q        )��P	�吉��A�*

	conv_loss���>&���        )��P	�搉��A�*

	conv_loss��>����        )��P	�P搉��A�*

	conv_lossuB�>[�        )��P	��搉��A�*

	conv_loss���>-�;$        )��P	��搉��A�*

	conv_loss��>�bP        )��P	�琉��A�*

	conv_lossv��>��H        )��P	�M琉��A�*

	conv_lossa��>�S�W        )��P	R�琉��A�*

	conv_loss�4�>�.p�        )��P		�琉��A�*

	conv_losscJ�>���        )��P	�萉��A�*

	conv_loss'��>95�        )��P	QE萉��A�*

	conv_lossS��>\�        )��P	��萉��A�*

	conv_loss�;�>�'�8        )��P	��萉��A�*

	conv_loss���>6�        )��P	��萉��A�*

	conv_lossd�>�}N        )��P	�:鐉��A�*

	conv_loss���>���        )��P	�x鐉��A�*

	conv_loss���>ZR'        )��P	�鐉��A�*

	conv_loss��>[\	2        )��P	V�鐉��A�*

	conv_loss�w�>88        )��P	�3ꐉ��A�*

	conv_loss���>q�6        )��P	�qꐉ��A�*

	conv_loss=A�>w�ύ        )��P	9�ꐉ��A�*

	conv_lossj��>r%        )��P	p�ꐉ��A�*

	conv_loss���>��        )��P	�)됉��A�*

	conv_lossN��>�2        )��P	%g됉��A�*

	conv_loss�f�>r&�e        )��P	�됉��A�*

	conv_loss��>�*�        )��P	��됉��A�*

	conv_lossH�>F�ʌ        )��P	"쐉��A�*

	conv_lossŐ�>�R�        )��P	`_쐉��A�*

	conv_loss&I�>5���        )��P	��쐉��A�*

	conv_lossTt�>�pX         )��P	��쐉��A�*

	conv_lossYv�>	�9�        )��P	<*퐉��A�*

	conv_loss���>Z�n        )��P	g퐉��A�*

	conv_loss<$�>�%�|        )��P	��퐉��A�*

	conv_loss���>���g        )��P	��퐉��A�*

	conv_loss)�>��21        )��P	o ��A�*

	conv_loss�b�>����        )��P	�`��A�*

	conv_loss�a�>[f��        )��P	����A�*

	conv_loss#��>UR��        )��P	����A�*

	conv_lossM1�>����        )��P	X��A�*

	conv_loss�	�>E�        )��P	w�󐉙�A�*

	conv_loss-O�>�j�        )��P	\�󐉙�A�*

	conv_loss��>� �        )��P	/�����A�*

	conv_loss�*�>�:5        )��P	+k�����A�*

	conv_loss>T�>t��        )��P	޶�����A�*

	conv_loss���>"Ȃ&        )��P	�������A�*

	conv_loss���>�d��        )��P	C�����A�*

	conv_loss�d�>E��        )��P	A������A�*

	conv_loss��>�	��        )��P	�������A�*

	conv_loss֣�>#Z�        )��P	j�����A�*

	conv_loss��>m��        )��P	�Z�����A�*

	conv_loss���>���m        )��P	Z������A�*

	conv_lossE��>�B�        )��P	�������A�*

	conv_lossp��>�Pfm        )��P	������A�*

	conv_loss�p�>G�        )��P	�M�����A�*

	conv_lossj�>�M        )��P	�������A�*

	conv_loss.��>����        )��P	�������A�*

	conv_loss�Z�>r���        )��P	�����A�*

	conv_lossϭ�>��G0        )��P	�P�����A�*

	conv_loss���>��Y@        )��P	A������A�*

	conv_loss���>�s��        )��P	������A�*

	conv_loss���>4~�        )��P	6�����A�*

	conv_loss���>�e��        )��P	ZI�����A�*

	conv_loss f�>�}%        )��P	]������A�*

	conv_loss.��>��4�        )��P	n������A�*

	conv_losse��>L�;=        )��P	 �����A�*

	conv_loss*�>��]        )��P	�@�����A�*

	conv_lossP��>̻��        )��P	������A�*

	conv_lossz4�>�2
�        )��P	�������A�*

	conv_lossJU�>a�p        )��P	������A�*

	conv_loss��>�,{        )��P	�7�����A�*

	conv_loss'��>��9#        )��P	�u�����A�*

	conv_loss!��>Ox"        )��P	Ӳ�����A�*

	conv_loss���>f`�        )��P	�������A�*

	conv_loss���>�        )��P	l,�����A�*

	conv_loss���>���        )��P	 k�����A�*

	conv_lossC��>u+}        )��P	c������A�*

	conv_loss���>Q�
�        )��P	#������A�*

	conv_loss��>��6�        )��P	�%�����A�*

	conv_loss���>�N%        )��P	Nd�����A�*

	conv_loss�*�>�J�,        )��P	�������A�*

	conv_loss�Y�>�G�        )��P	�������A�*

	conv_lossLF�>��I�        )��P	3�����A�*

	conv_loss���>���q        )��P	�X�����A�*

	conv_loss:*�>�q�        )��P	R������A�*

	conv_loss$C�>sN$�        )��P	h������A�*

	conv_loss��>Չ[�        )��P	c�����A�*

	conv_loss��>��m        )��P	N�����A�*

	conv_loss��>��r        )��P	n������A�*

	conv_lossR8�>�9��        )��P	�������A�*

	conv_loss���>���a        )��P	 ����A�*

	conv_loss��>!�8        )��P	�j ����A�*

	conv_loss��>l瑭        )��P	� ����A�*

	conv_loss���>�t#�        )��P	�� ����A�*

	conv_loss�)�>2�        )��P	V(����A�*

	conv_loss¡�>BIi�        )��P	ag����A�*

	conv_loss���>u��;        )��P	������A�*

	conv_lossH��>d4U�        )��P	�����A�*

	conv_loss�{�>_���        )��P	a����A�*

	conv_lossl|�>�n�N        )��P	������A�*

	conv_lossI��>}��        )��P	������A�*

	conv_loss���>���,        )��P	����A�*

	conv_loss�|�>n>��        )��P	�b����A�*

	conv_loss�8�>,`�@        )��P	w�����A�*

	conv_loss���>����        )��P	������A�*

	conv_losslV�>ș�&        )��P	�"����A�*

	conv_lossVB�>f��P        )��P	?c����A�*

	conv_loss���>�\r�        )��P	������A�*

	conv_lossD��>����        )��P	������A�*

	conv_lossk]�>VՇ=        )��P	�����A�*

	conv_loss��>ɟl�        )��P	�V����A�*

	conv_lossT��>���        )��P	������A�*

	conv_loss���>0�<�        )��P	������A�*

	conv_loss#h�>�>A�        )��P	�����A�*

	conv_lossפ�>?PY        )��P	�M����A�*

	conv_lossiA�>�#�        )��P	`�����A�*

	conv_lossя�>�`J        )��P		�����A�*

	conv_loss=��>�ݽ        )��P	Z	����A�*

	conv_loss��>�gQ�        )��P	�F����A�*

	conv_loss�~�>���        )��P	�����A�*

	conv_lossX)�>0,�        )��P	(�����A�*

	conv_loss{��>�t|E        )��P	������A�*

	conv_loss޵�>�߶        )��P	>����A�*

	conv_loss�Y�>xp�        )��P	�z����A�*

	conv_loss���>/*��        )��P	޸����A�*

	conv_losst��>� �        )��P	������A�*

	conv_loss�c�>I��X        )��P	�=	����A�*

	conv_loss���>8ڶd        )��P	D~	����A�*

	conv_loss�v�>I�8        )��P	��	����A�*

	conv_lossga�>�t5        )��P	��	����A�*

	conv_loss��>4C$O        )��P	9
����A�*

	conv_loss���>a��        )��P	�
����A�*

	conv_loss8q�>d�%F        )��P	 �
����A�*

	conv_lossv3�>Zzw�        )��P	��
����A�*

	conv_loss���>��B�        )��P	�6����A�*

	conv_loss�,�>�}"        )��P	�s����A�*

	conv_loss���>Yo�        )��P	�����A�*

	conv_loss��>��D        )��P	,�����A�*

	conv_loss���>��b4        )��P	�=����A�*

	conv_loss>��>]�^�        )��P	�|����A�*

	conv_loss?�>^�d=        )��P	ݹ����A�*

	conv_lossJn�><���        )��P	 �����A�*

	conv_loss���>ʗ=Z        )��P	�����A�*

	conv_loss[ �>��j�        )��P	�M����A�*

	conv_loss"��>j��        )��P	�����A�*

	conv_loss'��>x9�        )��P	>�����A�*

	conv_loss��>_AV(        )��P	�����A�*

	conv_loss?8�>2`��        )��P	�R����A�*

	conv_loss'�>4���        )��P	U�����A�*

	conv_lossw��>w���        )��P	������A�*

	conv_loss���>1��3        )��P	& ����A�*

	conv_loss�\�>�ϟm        )��P	z^����A�*

	conv_lossQ��>�ʹI        )��P	������A�*

	conv_loss���>Ir�        )��P	"�����A�*

	conv_loss��>(�:�        )��P	�����A�*

	conv_loss���>W�        )��P	�\����A�*

	conv_lossف�>��B        )��P	�����A�*

	conv_loss_x�>\�"�        )��P	�����A�*

	conv_loss-"�>oݠ�        )��P	�����A�*

	conv_loss�[�>$��(        )��P	?[����A�*

	conv_lossr�>�        )��P	;�����A�*

	conv_loss@�>5�!Q        )��P	�����A�*

	conv_loss��>m���        )��P	����A�*

	conv_loss�<�>酲�        )��P	�Y����A�*

	conv_loss���>��\Y        )��P	������A�*

	conv_lossk��>͢�        )��P	�����A�*

	conv_loss�]�>m��        )��P	����A�*

	conv_loss��>l&        )��P	�O����A�*

	conv_loss���>oԗ�        )��P	a�����A�*

	conv_losse��>^w"        )��P	N�����A�*

	conv_loss���>4J�(        )��P	�����A�*

	conv_loss��>[�+�        )��P	LE����A�*

	conv_loss��>,2��        )��P	�����A�*

	conv_lossV&�>a���        )��P	������A�*

	conv_lossz�>�xW        )��P	L ����A�*

	conv_lossy2�>+Iy        )��P	S=����A�*

	conv_loss��>�"�        )��P	cz����A�*

	conv_loss�r�>��D�        )��P	������A�*

	conv_loss���>F�F�        )��P	������A�*

	conv_loss�D�>"���        )��P	�2����A�*

	conv_loss|��>��        )��P	r����A�*

	conv_loss���>e?|�        )��P	;�����A�*

	conv_loss-�>ڎN        )��P	%�����A�*

	conv_loss��>�F�4        )��P	�+����A�*

	conv_loss�
�>m��         )��P	/i����A�*

	conv_loss�^�>ȩ��        )��P	�����A�*

	conv_loss$��>��D�        )��P	������A�*

	conv_lossӥ�><�3�        )��P	6 ����A�*

	conv_loss/��>�7"        )��P	N_����A�*

	conv_loss�^�>$��        )��P	1�����A�*

	conv_lossi�>�)        )��P	�����A�*

	conv_loss[�>?�P�        )��P	>+����A�*

	conv_loss,��>OW^A        )��P	�g����A�*

	conv_loss�<�>^?��        )��P	K�����A�*

	conv_loss���>�p+�        )��P	������A�*

	conv_loss���>�#��        )��P	�/����A�*

	conv_loss\��>��H        )��P	0n����A�*

	conv_loss��>d}��        )��P	z�����A�*

	conv_loss z�>���T        )��P	������A�*

	conv_lossn]�>�h��        )��P	�?����A�*

	conv_loss� �>+�2�        )��P	}�����A�*

	conv_loss�q�>�~�        )��P	������A�*

	conv_loss��>4Ok        )��P	����A�*

	conv_loss���>�
�r        )��P	�?����A�*

	conv_loss��>�hg�        )��P	�~����A�*

	conv_loss�i�>�_e        )��P	�����A�*

	conv_loss��>��[U        )��P	8�����A�*

	conv_loss���>�?        )��P	�4����A�*

	conv_lossE��>5$<�        )��P	>q����A�*

	conv_loss�u�>�~Fy        )��P	������A�*

	conv_loss��>M���        )��P	������A�*

	conv_loss���>�~�        )��P	�' ����A�*

	conv_loss�2�>��T        )��P	od ����A�*

	conv_loss�v�>�!B        )��P	գ ����A�*

	conv_loss���>��1        )��P	K� ����A�*

	conv_loss{A�>x�        )��P	2 !����A�*

	conv_loss,O�>9�x�        )��P	^!����A�*

	conv_lossb!�>�'J�        )��P	��!����A�*

	conv_lossy[�>�Z�        )��P	��!����A�*

	conv_lossj��>����        )��P	�"����A�*

	conv_lossш�>�$�        )��P	�P"����A�*

	conv_loss�v�>��+f        )��P	)�"����A�*

	conv_loss��>wN��        )��P	t�"����A�*

	conv_lossc9�>'M�f        )��P	#����A�*

	conv_loss&��>c"�\        )��P	�G#����A�*

	conv_loss���>�Y^        )��P	��#����A�*

	conv_loss"��>#���        )��P	��#����A�*

	conv_loss%��>O �F        )��P	��#����A�*

	conv_lossX�>M�cB        )��P	�:$����A�*

	conv_loss�y�>x�zG        )��P	�y$����A�*

	conv_lossg�>1���        )��P	��$����A�*

	conv_lossi��>_<�l        )��P	N�$����A�*

	conv_lossB��>���        )��P	d2%����A�*

	conv_lossz��>M$r        )��P	q%����A�*

	conv_loss0m�>�5�S        )��P	��%����A�*

	conv_loss���>��#�        )��P	1�%����A�*

	conv_loss�c�>�C�        )��P	�%&����A�*

	conv_loss��>��&T        )��P	c&����A�*

	conv_lossZ5�>d9��        )��P	�&����A�*

	conv_loss%��>�jN        )��P	��&����A�*

	conv_lossN5�>�{1�        )��P	�'����A�*

	conv_loss�o�>���;        )��P	x|'����A�*

	conv_loss���>Rl�        )��P	�'����A�*

	conv_loss�f�>�ŋ�        )��P	��'����A�*

	conv_lossp]�>j�J�        )��P	�;(����A�*

	conv_loss[X�>	��v        )��P	��(����A�*

	conv_loss2z�>��aJ        )��P	��(����A�*

	conv_loss7F�>?57d        )��P	\)����A�*

	conv_loss���>�%�#        )��P	lO)����A�*

	conv_loss�%�>�o�5        )��P	��)����A�*

	conv_loss� �>���/        )��P	��)����A�*

	conv_loss���>��%        )��P	|*����A�*

	conv_lossDf�>����        )��P	B[*����A�*

	conv_loss�Z�>��җ        )��P	a�*����A�*

	conv_loss��>c��        )��P	Y�*����A�*

	conv_loss�w�>� &        )��P	�+����A�*

	conv_loss�M�>M�̞        )��P	�U+����A�*

	conv_loss���>\f:        )��P	̑+����A�*

	conv_loss���>���        )��P	�+����A�*

	conv_loss/��>�_�        )��P	�,����A�*

	conv_loss���>)��        )��P	�K,����A�*

	conv_loss�}�>�Z�        )��P	��,����A�*

	conv_loss���>�k"        )��P	W�,����A�*

	conv_loss�o�>Ǩ��        )��P	
-����A�*

	conv_loss�v�>T[n<        )��P	?A-����A�*

	conv_lossS��>�W1        )��P	�-����A�*

	conv_lossm�>���        )��P	j�-����A�*

	conv_loss��>��D�        )��P	Z�-����A�*

	conv_lossp��>]0*�        )��P	�9.����A�*

	conv_loss���>n��        )��P	�w.����A�*

	conv_loss�^�>B\>G        )��P	��.����A�*

	conv_losst��>�#V        )��P	q�.����A�*

	conv_lossVR�>w�R�        )��P	�-/����A�*

	conv_lossag�>�٬�        )��P	�m/����A�*

	conv_loss ��>)W�        )��P	�/����A�*

	conv_loss#o�>rm��        )��P	��/����A�*

	conv_lossG�>MP��        )��P	�00����A�*

	conv_loss���>���j        )��P	�o0����A�*

	conv_loss�V�>���        )��P	%�0����A�*

	conv_loss�M�>L)        )��P	E�0����A�*

	conv_loss���>��        )��P	�/1����A�*

	conv_loss���>ޅ=�        )��P	�u1����A�*

	conv_loss���>�U��        )��P	��1����A�*

	conv_lossS��>�G        )��P	��1����A�*

	conv_loss�w�>G�        )��P	/52����A�*

	conv_loss�>;A�        )��P	s2����A�*

	conv_lossBB�>j���        )��P	ɳ2����A�*

	conv_lossCX�>��        )��P	��2����A�*

	conv_loss��>A��        )��P	�/3����A�*

	conv_lossU�>��0        )��P	l3����A�*

	conv_loss<w�>S�C�        )��P	D�3����A�*

	conv_lossr�>ܖ�o        )��P	��3����A�*

	conv_lossD�>���        )��P	94����A�*

	conv_lossI��>@sN        )��P	�w4����A�*

	conv_loss���>�iE'        )��P	�4����A�*

	conv_loss�B�>�;?�        )��P	�5����A�*

	conv_lossc��>Y��B        )��P	BC5����A�*

	conv_loss̢�> �n        )��P	*�5����A�*

	conv_loss���>��        )��P	`�5����A�*

	conv_lossw��>�j�Y        )��P	
6����A�*

	conv_loss���>/�M        )��P	0H6����A�*

	conv_lossF@�>iӺ�        )��P	j�6����A�*

	conv_loss0W�>P��        )��P	��6����A�*

	conv_loss.��>�)k�        )��P	�7����A�*

	conv_losso��>D��        )��P	FW7����A�*

	conv_loss3��>3�E        )��P	O�7����A�*

	conv_loss���>���@        )��P	��7����A�*

	conv_loss���>UE�        )��P	�8����A�*

	conv_loss�E�>v��        )��P	kO8����A�*

	conv_loss�T�>��Y        )��P	�8����A�*

	conv_loss=f�>%��        )��P	F�8����A�*

	conv_loss�$�>�L��        )��P	<9����A�*

	conv_losssy�>~r�        )��P	�E9����A�*

	conv_loss�P�>����        )��P	�9����A�*

	conv_loss��>�p��        )��P	��9����A�*

	conv_lossSM�>�<�"        )��P	M�9����A�*

	conv_lossO��>��v        )��P	�<:����A�*

	conv_lossF%�>�%5�        )��P	ly:����A�*

	conv_lossG_�>T�Z        )��P	�:����A�*

	conv_loss촽>��Г        )��P	p�:����A�*

	conv_loss6�>4�N$        )��P	91;����A�*

	conv_lossE��>0���        )��P	rm;����A�*

	conv_lossB��>�OB�        )��P	�;����A�*

	conv_loss>��>%��Y        )��P	��;����A�*

	conv_loss�>�?�        )��P	y%<����A�*

	conv_lossL�>As�        )��P	�b<����A�*

	conv_loss95�>0��        )��P	��<����A�*

	conv_lossc�>�T        )��P	��<����A�*

	conv_lossˬ�>GV�        )��P	�=����A�*

	conv_loss���>�hi        )��P	NW=����A�*

	conv_loss��>җt�        )��P	��=����A�*

	conv_loss���>�E�        )��P	V�=����A�*

	conv_loss��>���z        )��P	Q>����A�*

	conv_loss��><�6        )��P	�J>����A�*

	conv_loss�>�>r��7        )��P	��>����A�*

	conv_loss[�>g���        )��P	��>����A�*

	conv_loss`��>�/s�        )��P	x?����A�*

	conv_lossں>r�g�        )��P	�C?����A�*

	conv_loss*��>�[��        )��P	��?����A�*

	conv_loss(}�>\=        )��P	>�?����A�*

	conv_loss#�>b��^        )��P	��?����A�*

	conv_loss���>�>hF        )��P	�B����A�*

	conv_loss��>�j�        )��P	tKB����A�*

	conv_lossʠ�>jO�Y        )��P	�B����A�*

	conv_loss��>���        )��P	��B����A�*

	conv_loss�T�>���<        )��P	�C����A�*

	conv_loss^�>�q        )��P	|YC����A�*

	conv_loss�ɻ>��:        )��P	̟C����A�*

	conv_loss��>�kB�        )��P	��C����A�*

	conv_lossJ��>�l�        )��P	�"D����A�*

	conv_loss],�>�⛜        )��P	DhD����A�*

	conv_loss���>4�X        )��P	�D����A�*

	conv_lossֽ>����        )��P	��D����A�*

	conv_loss�>����        )��P	� E����A�*

	conv_loss�:�>�b{        )��P	y^E����A�*

	conv_loss��>�Yq�        )��P	��E����A�*

	conv_lossD�>"�"�        )��P	��E����A�*

	conv_loss�L�>Z8�        )��P	�'F����A�*

	conv_lossT�>�K�        )��P	BfF����A�*

	conv_lossÔ�>C�É        )��P	��F����A�*

	conv_loss���>�y
�        )��P	�F����A�*

	conv_loss��>t#i�        )��P	`!G����A�*

	conv_loss&��>����        )��P	�]G����A�*

	conv_lossM��>֧�	        )��P	^�G����A�*

	conv_loss4��>b��R        )��P	E�G����A�*

	conv_loss�-�>}�CL        )��P	�H����A�*

	conv_loss���>�ۦ        )��P	�TH����A�*

	conv_lossӵ�>L�]        )��P	�H����A�*

	conv_loss���>��1[        )��P	Z�H����A�*

	conv_lossӝ�>k�oj        )��P	I����A�*

	conv_loss���>a�t        )��P	�HI����A�*

	conv_loss�_�>�H�        )��P	��I����A�*

	conv_loss�^�>+��m        )��P	�I����A�*

	conv_lossj\�>\p>        )��P	`J����A�*

	conv_loss|Ƿ>�Դ#        )��P	�BJ����A�*

	conv_lossC�>�jF        )��P	��J����A�*

	conv_lossۣ�>+ݨ�        )��P	
�J����A�*

	conv_lossIվ>�}}�        )��P	K�J����A�*

	conv_loss�}�>�se        )��P	�9K����A�*

	conv_loss�#�>XO�        )��P	wK����A�*

	conv_loss���>��W        )��P	X�K����A�*

	conv_loss}v�>=l%        )��P	�K����A�*

	conv_loss"9�>���        )��P	V.L����A�*

	conv_loss[�>OU�z        )��P	�lL����A�*

	conv_loss�+�>@�D�        )��P	��L����A�*

	conv_loss��>��q        )��P	��L����A�*

	conv_loss5��>ِ��        )��P	F$M����A�*

	conv_loss��>�4�        )��P	!aM����A�*

	conv_lossָ>Ft7�        )��P	��M����A�*

	conv_loss�G�>x]�Z        )��P	��M����A�*

	conv_lossT$�>�(�V        )��P	�N����A�*

	conv_loss"ܷ>�0        )��P	�gN����A�*

	conv_loss��>�b�b        )��P	t�N����A�*

	conv_loss��>3�=        )��P	;�N����A�*

	conv_lossC��>_�`        )��P	�#O����A�*

	conv_loss�C�>��        )��P	�aO����A�*

	conv_loss�>u�8<        )��P	A�O����A�*

	conv_loss|W�>(�z        )��P	��O����A�*

	conv_loss�A�>em�        )��P	�@P����A�*

	conv_loss�>�S        )��P	~P����A�*

	conv_loss1˼>N�-!        )��P	��P����A�*

	conv_loss7��>I�2        )��P	�Q����A�*

	conv_loss���>��b�        )��P	WQ����A�*

	conv_lossŻ>�eݦ        )��P	��Q����A�*

	conv_loss���>�JBH        )��P	�Q����A�*

	conv_loss���>���i        )��P	�R����A�*

	conv_loss���>6��B        )��P	�XR����A�*

	conv_loss��>�U3        )��P	 �R����A�*

	conv_loss��>��f]        )��P	F�R����A�*

	conv_loss���>��        )��P	�S����A�*

	conv_loss���>۬�]        )��P	�TS����A�*

	conv_losso�>7'�        )��P	��S����A�*

	conv_loss�i�>�R�B        )��P	&�S����A�*

	conv_loss��>v�        )��P	DT����A�*

	conv_loss���>jC�        )��P	.MT����A�*

	conv_loss��>7��9        )��P		�T����A�*

	conv_loss�;�>�t�[        )��P	��T����A�*

	conv_loss\׸>�BE�        )��P	�U����A�*

	conv_loss8o�>��@        )��P	]CU����A�*

	conv_loss5.�>�-        )��P	|U����A�*

	conv_lossy�>�h�%        )��P	&�U����A�*

	conv_loss�/�>�_-	        )��P	�V����A�*

	conv_loss��>�e�        )��P	�MV����A�*

	conv_loss��>���        )��P	+�V����A�*

	conv_loss�0�>�!�        )��P	.�V����A�*

	conv_lossIȪ>owTu        )��P	�W����A�*

	conv_lossK��>k��        )��P	�TW����A�*

	conv_lossѱ>;�        )��P	�W����A�*

	conv_loss%�>�r�        )��P	�W����A�*

	conv_loss��>ԼHE        )��P	�X����A�*

	conv_loss���>';v        )��P	*OX����A�*

	conv_loss��>=�m�        )��P	U�X����A�*

	conv_loss�9�>�w;        )��P	{�X����A�*

	conv_loss���>��(        )��P	�Y����A�*

	conv_lossWs�>3�>M        )��P	�DY����A�*

	conv_loss÷>`��        )��P	s�Y����A�*

	conv_loss�,�>�eS}        )��P	�Y����A�*

	conv_loss�Z�>�@�        )��P	�Y����A�*

	conv_lossz�>��ͺ        )��P	<;Z����A�*

	conv_lossTG�>�=8        )��P	+wZ����A�*

	conv_loss�7�>$�*�        )��P	�Z����A�*

	conv_loss��>n��        )��P	�[����A�*

	conv_loss���>��j�        )��P	�@[����A�*

	conv_loss�%�>lo�        )��P	�}[����A�*

	conv_loss�c�>$�:        )��P	��[����A�*

	conv_loss�	�>iȢ�        )��P	@\����A�*

	conv_lossN*�>��I}        )��P	�O\����A�*

	conv_lossf��>�XM�        )��P	q�\����A�*

	conv_loss쳭>p֏�        )��P	��\����A�*

	conv_loss���>1J}�        )��P	r!]����A�*

	conv_lossST�>�0��        )��P	K`]����A�*

	conv_loss��>���        )��P	��]����A�*

	conv_loss�x�>s6�        )��P	"�]����A�*

	conv_loss��>6�w�        )��P	�^����A�*

	conv_loss���>�K��        )��P	�V^����A�*

	conv_lossv]�>� 	�        )��P	�^����A�*

	conv_loss/��>1!_        )��P	��^����A�*

	conv_loss���>�v�        )��P	�_����A�*

	conv_loss��>�	t        )��P	�[_����A�*

	conv_loss�ٵ>J���        )��P	��_����A�*

	conv_loss�:�>�'        )��P	n�_����A�*

	conv_loss쬮>��Q        )��P	5`����A�*

	conv_loss���>�]        )��P	�R`����A�*

	conv_lossNٳ>�V��        )��P	��`����A�*

	conv_loss"̲>�T�        )��P	�`����A�*

	conv_loss
"�>�|D        )��P	a����A�*

	conv_loss�!�>��o%        )��P	�Ja����A�*

	conv_lossQi�>���H        )��P	<�a����A�*

	conv_lossF\�>�j9        )��P	��a����A�*

	conv_loss�>���g        )��P	�b����A�*

	conv_loss�J�>W\         )��P	Ab����A�*

	conv_loss��>-踴        )��P	
b����A�*

	conv_loss$��>��        )��P	u�b����A�*

	conv_loss=�> Գ�        )��P	(�b����A�*

	conv_loss8,�>�3M        )��P	�6c����A�*

	conv_loss�Y�>l� 1        )��P	Ctc����A�*

	conv_lossi	�>N�ֽ        )��P	��c����A�*

	conv_losss�>���        )��P	�c����A�*

	conv_lossI��>CVL        )��P	M1d����A�*

	conv_lossJ��>��d        )��P	god����A�*

	conv_loss���>��)S        )��P	��d����A�*

	conv_loss���> :         )��P	q�d����A�*

	conv_lossr��>��ű        )��P	�'e����A�*

	conv_lossޱ>���        )��P	�ee����A�*

	conv_loss�ʬ>��F�        )��P	ؤe����A�*

	conv_loss%Ѱ>�ʓ        )��P	b�e����A�*

	conv_lossu��>���        )��P	�f����A�*

	conv_loss���>��B        )��P	s[f����A�*

	conv_lossQ0�> ��        )��P	u�f����A�*

	conv_losss��>g��        )��P	��f����A�*

	conv_losso��>&1Oo        )��P	�g����A�*

	conv_loss��>M�UN        )��P	1hg����A�*

	conv_loss���>M�/Q        )��P	�g����A�*

	conv_loss�4�>��        )��P	�g����A�*

	conv_loss�>@Ր        )��P	$h����A�*

	conv_loss�ɭ>���\        )��P	oh����A�*

	conv_loss�-�>�ʪ?        )��P	��h����A�*

	conv_lossHg�>4�        )��P	b�h����A�*

	conv_loss���> ��o        )��P	I@i����A�*

	conv_loss&߭>�G��        )��P	��i����A�*

	conv_loss)��>�Y�h        )��P	��i����A�*

	conv_loss}o�>BvG�        )��P	��i����A�*

	conv_lossu��>��CJ        )��P	m<j����A�*

	conv_losst��>�V        )��P	{j����A�*

	conv_loss(7�>�)        )��P	��j����A�*

	conv_loss(v�>�C	        )��P	h�j����A�*

	conv_lossS�>(���        )��P	�6k����A�*

	conv_loss�-�>+U�        )��P	�|k����A�*

	conv_loss��>���        )��P	��k����A�*

	conv_loss��>�;�        )��P	C�k����A�*

	conv_loss
F�>�6"        )��P	�8l����A�*

	conv_lossv�>�� '        )��P	Xxl����A�*

	conv_loss�>,)K�        )��P	W�l����A�*

	conv_loss駯>>YT�        )��P	S�l����A�*

	conv_loss�ب>�oM        )��P	l.m����A�*

	conv_loss?�>���V        )��P	�lm����A�*

	conv_loss��>����        )��P	��m����A�*

	conv_loss���>���y        )��P	o�m����A�*

	conv_loss��>��L        )��P	R'n����A�*

	conv_lossm�>&)��        )��P	Gfn����A�*

	conv_lossv��>qx�        )��P	̣n����A�*

	conv_lossC��>��o�        )��P	��n����A�*

	conv_loss���>���        )��P	�o����A�*

	conv_lossW��>�ڒL        )��P	XXo����A�*

	conv_loss�	�>�W.L        )��P	͗o����A�*

	conv_loss8ȩ>Q�Oo        )��P	[�o����A�*

	conv_loss��>��7        )��P	�p����A�*

	conv_lossҶ�>���        )��P	�Mp����A�*

	conv_loss�v�>��.        )��P	M�p����A�*

	conv_loss)�>`�m�        )��P	��p����A�*

	conv_loss)��>���        )��P	�q����A�*

	conv_loss	Ȧ>�B`        )��P	�Dq����A�*

	conv_loss���>�7
�        )��P	��q����A�*

	conv_loss��>�[Qo        )��P	3�q����A�*

	conv_loss���>�'E�        )��P	�q����A�*

	conv_loss=u�>�y��        )��P	9;r����A�*

	conv_loss�)�>�L         )��P	�zr����A�*

	conv_loss�h�> ��r        )��P	�r����A�*

	conv_loss���>�K�t        )��P	L�r����A�*

	conv_loss�t�>A�N2        )��P	2s����A�*

	conv_losse��>���        )��P	��w����A�*

	conv_lossY��>!c��        )��P	�y����A�*

	conv_loss�r�>26Ă        )��P	��y����A�*

	conv_loss�ʭ>���'        )��P	�-z����A�*

	conv_lossA�>��        )��P	�nz����A�*

	conv_loss|ʯ>�}	G        )��P	�z����A�*

	conv_lossuͫ>��Q�        )��P	A�z����A�*

	conv_loss�e�>���        )��P	�8{����A�*

	conv_loss�Ψ>c���        )��P	v|{����A�*

	conv_loss�̥>�)t�        )��P	)�{����A�*

	conv_loss?�>%�        )��P	��{����A�*

	conv_loss�>�Q0+        )��P	Y<|����A�*

	conv_loss�2�>=&_E        )��P	�y|����A�*

	conv_lossXn�>ai6        )��P	6�|����A�*

	conv_loss�U�>2ݖu        )��P	��|����A�*

	conv_lossb��>!$J        )��P	�M}����A�*

	conv_loss��>�
        )��P	��}����A�*

	conv_loss�\�>��ͼ        )��P	��}����A�*

	conv_loss��>5�        )��P	�~����A�*

	conv_lossR��>�=�z        )��P	UM~����A�*

	conv_loss,�>�H_K        )��P	v�~����A�*

	conv_loss*�>_��        )��P	��~����A�*

	conv_loss'��>����        )��P	I����A�*

	conv_loss=�>�܍F        )��P	hC����A�*

	conv_loss��>f�W,        )��P	�����A�*

	conv_loss%��>��        )��P	������A�*

	conv_loss�ܣ>�H��        )��P	7�����A�*

	conv_loss/Z�>��	�        )��P	�J�����A�*

	conv_loss��>:�K        )��P	1������A�*

	conv_lossd�>�)�        )��P	QȀ����A�*

	conv_loss߇�>}s��        )��P	^�����A�*

	conv_loss�A�>(!        )��P	eS�����A�*

	conv_lossD�>�P,        )��P	T������A�*

	conv_loss�ڬ>ʇ�        )��P	�ρ����A�*

	conv_lossY֠>�P        )��P	������A�*

	conv_loss{��>e�s        )��P	[J�����A�*

	conv_lossUj�>}-�        )��P	(������A�*

	conv_lossc�>�)�        )��P	�ł����A�*

	conv_loss�ϣ>�V��        )��P	S�����A�*

	conv_loss���>��A�        )��P	�?�����A�*

	conv_lossy�>K�W�        )��P	�{�����A�*

	conv_loss�j�>`m��        )��P	�������A�*

	conv_loss �> (+        )��P	E������A�*

	conv_loss�[�>��;�        )��P	~3�����A�*

	conv_loss��>��=        )��P	�q�����A�*

	conv_loss���>�q�        )��P	B������A�*

	conv_loss�ã>��N�        )��P	�넑���A�*

	conv_loss�'�>i�f        )��P	�(�����A�*

	conv_loss61�>ͤ'�        )��P	�e�����A�*

	conv_lossд�>�        )��P	7������A�*

	conv_loss,m�>�y/        )��P	Vᅑ���A�*

	conv_loss:r�>�g�4        )��P	^2�����A�*

	conv_lossO_�>�u=�        )��P	�p�����A�*

	conv_loss)��>�z�         )��P	�������A�*

	conv_loss��>����        )��P	����A�*

	conv_loss:�>�K8        )��P	�9�����A�*

	conv_loss���>�ށ�        )��P	 w�����A�*

	conv_lossǨ>Q͗�        )��P	������A�*

	conv_loss/��>�4�        )��P	�������A�*

	conv_loss��>a�        )��P	�?�����A�*

	conv_loss��>�z6�        )��P	�~�����A�*

	conv_lossYV�>[���        )��P	�ψ����A�*

	conv_lossFC�>��f        )��P	������A�*

	conv_lossv�>��        )��P	uH�����A�*

	conv_loss�e�>��m        )��P	�������A�*

	conv_loss��>���*        )��P	vՉ����A�*

	conv_loss+u�>U�pQ        )��P	������A�*

	conv_loss�?�>v�9�        )��P	 Q�����A�*

	conv_lossuB�>�Jr        )��P	�������A�*

	conv_loss�G�>��        )��P	�ˊ����A�*

	conv_loss���>����        )��P	3�����A�*

	conv_loss�9�>�Pc�        )��P	�D�����A�*

	conv_loss�>�a�t        )��P	Z������A�*

	conv_loss��>���2        )��P	i������A�*

	conv_loss�V�>�-�        )��P	<������A�*

	conv_lossg��>P���        )��P	z9�����A�*

	conv_loss��>5�^        )��P	Dw�����A�*

	conv_loss�>|0)�        )��P	�������A�*

	conv_lossoZ�>���l        )��P	�񌑉��A�*

	conv_loss-��>yΗ�        )��P	V/�����A�*

	conv_lossW�>245a        )��P	Gm�����A�*

	conv_loss���>T�L�        )��P	J������A�*

	conv_loss�W�>�i_�        )��P	(ꍑ���A�*

	conv_loss{�>�x6        )��P	�'�����A�*

	conv_loss���>���d        )��P	vd�����A�*

	conv_lossۈ�>=R%        )��P	!������A�*

	conv_loss�Ο>8{�        )��P	fߎ����A�*

	conv_loss��>j＿        )��P	������A�*

	conv_lossڻ�>Y��        )��P	�Y�����A�*

	conv_loss���>)�        )��P	�����A�*

	conv_loss�Y�>׼��        )��P	ҏ����A�*

	conv_loss�u�>/~]        )��P	�����A�*

	conv_loss�9�>�I��        )��P	�M�����A�*

	conv_loss>���        )��P	̋�����A�*

	conv_loss6�>j �        )��P	7ː����A�*

	conv_loss]a�>��G        )��P	Z	�����A�*

	conv_loss3m�>Ӻ�+        )��P	yF�����A�*

	conv_lossK3�>��$        )��P	>������A�*

	conv_losscr�>�m��        )��P	z������A�*

	conv_lossl�>�y�s        )��P	�������A�*

	conv_loss��>/�"�        )��P	[:�����A�*

	conv_lossQ�>�U)        )��P	������A�*

	conv_loss�h�> �        )��P	Lƒ����A�*

	conv_lossD�>S���        )��P	_�����A�*

	conv_loss��>�=y�        )��P	�G�����A�*

	conv_lossF�>����        )��P	כ�����A�*

	conv_loss�Ѡ>ȥχ        )��P	�ߓ����A�*

	conv_loss�z�>���        )��P	�%�����A�*

	conv_loss�>�l�"        )��P	*f�����A�*

	conv_loss���>Z�5<        )��P	g������A�*

	conv_loss���>�j�Z        )��P	����A�*

	conv_lossC8�>k�|�        )��P	++�����A�*

	conv_loss��>	<        )��P	�i�����A�*

	conv_losswϟ>��c^        )��P	�������A�*

	conv_loss	�>d�s�        )��P	�ꕑ���A�*

	conv_loss�F�>C�|�        )��P	z0�����A�*

	conv_loss>�^�        )��P	�q�����A�*

	conv_loss �><uG        )��P	5������A�*

	conv_loss��>K�M�        )��P	����A�*

	conv_lossǣ>�Ⱦ�        )��P	�,�����A�*

	conv_loss���>BP׵        )��P	j�����A�*

	conv_loss�I�>1��"        )��P	������A�*

	conv_loss�9�>�_��        )��P	�嗑���A�*

	conv_loss�>�ن        )��P	6$�����A�*

	conv_loss�0�>A�        )��P	�b�����A�*

	conv_lossJ?�>f�2�        )��P	�������A�*

	conv_loss�ۢ>O�	        )��P	ߘ����A�*

	conv_loss(��>�A��        )��P	������A�*

	conv_loss��>]eP�        )��P	a[�����A�*

	conv_losspW�>%���        )��P	�����A�*

	conv_losse��>��¬        )��P	�ՙ����A�*

	conv_loss���>�E�1        )��P	������A�*

	conv_lossNۣ>���_        )��P	�O�����A�*

	conv_lossu͠>`63)        )��P	������A�*

	conv_lossv�>[4�        )��P	gʚ����A�*

	conv_loss���>����        )��P	Q	�����A�*

	conv_loss��>�L�        )��P	H�����A�*

	conv_loss�a�>1��        )��P	g������A�*

	conv_lossb��>�%ç        )��P	�����A�*

	conv_loss�>�|�        )��P	� �����A�*

	conv_loss,�>U�        )��P	v=�����A�*

	conv_lossQ�>�O�        )��P	�z�����A�*

	conv_lossӖ�>���        )��P	ܷ�����A�*

	conv_loss��>�Q�-        )��P	]������A�*

	conv_loss ܛ>�]��        )��P	M1�����A�*

	conv_loss8t�>��        )��P	�n�����A�*

	conv_loss���>���?        )��P	E������A�*

	conv_lossf��>�bܶ        )��P	�靑���A�*

	conv_losse��>�J�        )��P	((�����A�*

	conv_loss*ԙ>H���        )��P	Qe�����A�*

	conv_lossh��>��i        )��P	�������A�*

	conv_lossL�>a�]�        )��P	#󞑉��A�*

	conv_lossݪ�>V~_V        )��P	</�����A�	*

	conv_loss��>���1        )��P	)k�����A�	*

	conv_lossY��>��q        )��P	�������A�	*

	conv_loss�$�>��        )��P	�����A�	*

	conv_loss�˝>_|ԓ        )��P	�O�����A�	*

	conv_lossX*�>}i}@        )��P	M������A�	*

	conv_loss~ �>�
(<        )��P	pՠ����A�	*

	conv_loss��>)v�!        )��P	������A�	*

	conv_loss��>�[cj        )��P	�W�����A�	*

	conv_lossS�>A'�~        )��P	!������A�	*

	conv_loss0�>�\��        )��P	�ա����A�	*

	conv_lossV�>�j��        )��P	������A�	*

	conv_loss�F�>Pϙ        )��P	Q�����A�	*

	conv_losse\�>#���        )��P	ꔢ����A�	*

	conv_loss�]�>�y�        )��P	�֢����A�	*

	conv_loss��>^��e        )��P	p�����A�	*

	conv_loss��>���E        )��P	Y�����A�	*

	conv_loss��>���        )��P	"������A�	*

	conv_lossˠ>��        )��P	�ԣ����A�	*

	conv_loss\�>��a�        )��P	�!�����A�	*

	conv_loss�#�>���        )��P	�^�����A�	*

	conv_loss�ٟ>@��<        )��P	8������A�	*

	conv_lossC��>�LA�        )��P	�פ����A�	*

	conv_loss�ݜ>x��        )��P	o�����A�	*

	conv_lossJ��>����        )��P	�[�����A�	*

	conv_loss��>}AY�        )��P	&������A�	*

	conv_loss5�>�o�        )��P	�ۥ����A�	*

	conv_lossyY�>(f3X        )��P	�����A�	*

	conv_loss�7�>        )��P	�X�����A�	*

	conv_loss܃�>�7�W        )��P	�������A�	*

	conv_loss���>��        )��P	֦����A�	*

	conv_loss���>n1�^        )��P	������A�	*

	conv_loss��>ﴌ�        )��P	�R�����A�	*

	conv_loss+��>t��        )��P	a������A�	*

	conv_lossޕ>3��        )��P	̧����A�	*

	conv_lossL��>�\�        )��P	������A�	*

	conv_loss�Κ>���2        )��P	�F�����A�	*

	conv_lossĄ�>�Z�        )��P	΃�����A�	*

	conv_lossՅ�>��b        )��P	�������A�	*

	conv_loss��><>7        )��P	�������A�	*

	conv_loss�W�>�Qv�        )��P	<�����A�	*

	conv_loss��>�38v        )��P	Xy�����A�	*

	conv_loss4�>J�g        )��P	p������A�	*

	conv_lossNǛ>4s+        )��P	�򩑉��A�	*

	conv_loss��>�tS�        )��P	0�����A�	*

	conv_loss��><A�        )��P	wn�����A�	*

	conv_lossG�>���        )��P	w������A�	*

	conv_loss5k�>���-        )��P	�ꪑ���A�	*

	conv_loss��>k�	�        )��P	�(�����A�	*

	conv_loss� �>h!A        )��P	�;�����A�	*

	conv_loss꜕>46F�        )��P	�x�����A�	*

	conv_loss��>R�        )��P	h������A�	*

	conv_lossӃ�>�X��        )��P	b������A�	*

	conv_loss�&�>����        )��P	�5�����A�	*

	conv_loss�>�>�ͣ�        )��P	o~�����A�	*

	conv_loss���>r���        )��P	1������A�	*

	conv_lossS'�>�?D�        )��P	������A�	*

	conv_losso��>���        )��P	�K�����A�	*

	conv_lossc^�>���        )��P	A������A�	*

	conv_lossۇ�>�        )��P	�ϯ����A�	*

	conv_loss��>\�dM        )��P	������A�	*

	conv_lossR��>^U=$        )��P	K�����A�	*

	conv_loss�>Pu �        )��P	x������A�	*

	conv_lossH�>
]��        )��P	 ǰ����A�	*

	conv_loss[��>k#v        )��P	������A�	*

	conv_lossvG�>紙�        )��P	aA�����A�	*

	conv_loss��>^�x        )��P	 ������A�	*

	conv_loss�n�>�_�        )��P	F±����A�	*

	conv_loss=b�>�        )��P	������A�	*

	conv_loss��>�!��        )��P	�B�����A�	*

	conv_loss��>)        )��P	ɀ�����A�	*

	conv_loss��>D@��        )��P	̽�����A�	*

	conv_lossֺ�>��$�        )��P	������A�	*

	conv_loss(T�>��}�        )��P	�;�����A�	*

	conv_loss��>)�        )��P	�z�����A�	*

	conv_loss �>�8��        )��P	�������A�	*

	conv_lossP��>~���        )��P	.������A�	*

	conv_lossr��>'�DY        )��P	�6�����A�	*

	conv_loss�&�>t��        )��P	�s�����A�	*

	conv_loss�>�>wBQ
        )��P	;������A�	*

	conv_loss�n�>GW�
        )��P	�������A�	*

	conv_loss?��>/�/,        )��P	E>�����A�	*

	conv_lossI3�>4T        )��P	�|�����A�	*

	conv_lossů�>c��        )��P	l������A�	*

	conv_loss*ڔ>iBC        )��P	�������A�	*

	conv_loss�<�>/�uV        )��P	�5�����A�	*

	conv_loss�T�>5!��        )��P	*r�����A�	*

	conv_loss�"�>��z�        )��P	Я�����A�	*

	conv_lossF'�>��.[        )��P	춑���A�	*

	conv_loss�|�>�d�        )��P	�(�����A�	*

	conv_lossmΑ>>I^�        )��P	3g�����A�	*

	conv_lossO�>�\ �        )��P	�������A�	*

	conv_loss���>N��'        )��P	�߷����A�	*

	conv_loss1�>�.g\        )��P	]�����A�	*

	conv_loss�ۖ>K���        )��P	�X�����A�	*

	conv_loss�"�>���        )��P	\������A�	*

	conv_loss6ؑ>�-�w        )��P	�ո����A�	*

	conv_loss�ޕ>��/�        )��P	������A�	*

	conv_loss���>>YY        )��P	@T�����A�	*

	conv_loss:��>&�        )��P	�������A�	*

	conv_loss=t�>�AX        )��P	�乑���A�	*

	conv_loss:S�>����        )��P	b"�����A�	*

	conv_loss"��>
��/        )��P	�g�����A�	*

	conv_loss4��>YBj^        )��P	2������A�	*

	conv_loss�ݛ>%��        )��P	L𺑉��A�	*

	conv_loss�5�>mX4        )��P	r4�����A�	*

	conv_loss��>�l?        )��P	�w�����A�	*

	conv_loss�-�>�G��        )��P	�Ȼ����A�	*

	conv_loss`Q�>�J��        )��P	T�����A�	*

	conv_lossc�>�0�_        )��P	8N�����A�	*

	conv_loss�8�>b��        )��P	�������A�	*

	conv_loss���>�s;        )��P	�ȼ����A�	*

	conv_loss��>N�        )��P	������A�	*

	conv_loss6�>BM��        )��P	�B�����A�	*

	conv_loss�A�>�L4        )��P	р�����A�	*

	conv_lossә>�)e�        )��P	�������A�	*

	conv_loss��>�e        )��P	������A�	*

	conv_loss0��>��        )��P	E9�����A�	*

	conv_loss���>x        )��P	�������A�	*

	conv_lossa�>�0�        )��P	�þ����A�	*

	conv_loss\��>I��        )��P	� �����A�	*

	conv_loss$��>ږ3�        )��P	3>�����A�	*

	conv_loss�{�>�Q�        )��P	�}�����A�	*

	conv_loss��>0e)        )��P	������A�	*

	conv_loss�ғ>�`��        )��P	6������A�	*

	conv_loss�ď>'��c        )��P	�4�����A�	*

	conv_loss�8�>48rk        )��P	�q�����A�	*

	conv_loss� �>l���        )��P	������A�	*

	conv_loss�>p�[�        )��P	�������A�
*

	conv_loss��>�d�         )��P	�'�����A�
*

	conv_lossm\�>w��M        )��P	ce�����A�
*

	conv_loss���>�x�f        )��P	������A�
*

	conv_lossVȒ>2CL�        )��P	�������A�
*

	conv_loss!�>��\        )��P	R���A�
*

	conv_loss���>��Y        )��P	LY���A�
*

	conv_lossd��>�9�        )��P	����A�
*

	conv_loss*�>|�h
        )��P	����A�
*

	conv_loss.�>��e&        )��P	�Ñ���A�
*

	conv_loss���>�֝g        )��P	�LÑ���A�
*

	conv_lossu�>�F��        )��P	Y�Ñ���A�
*

	conv_loss�œ>^t��        )��P	��Ñ���A�
*

	conv_loss=ʗ>+>&6        )��P	uđ���A�
*

	conv_loss�ە>R?�7        )��P	�Uđ���A�
*

	conv_loss,f�>����        )��P	�đ���A�
*

	conv_losseh�>�NVW        )��P	��đ���A�
*

	conv_lossq�>A�        )��P	7ő���A�
*

	conv_loss�Z�>�V�        )��P	�^ő���A�
*

	conv_loss=��>m	Q�        )��P	�ő���A�
*

	conv_lossBW�>�<��        )��P	-�ő���A�
*

	conv_losse��>��T        )��P	0)Ƒ���A�
*

	conv_loss�P�>���        )��P	AfƑ���A�
*

	conv_loss�!�>��2�        )��P	(�Ƒ���A�
*

	conv_loss�ڔ>;�6        )��P	��Ƒ���A�
*

	conv_losse�>�z�        )��P	�2Ǒ���A�
*

	conv_lossa�>
�}�        )��P	'rǑ���A�
*

	conv_loss�	�>�٩�        )��P	8�Ǒ���A�
*

	conv_loss�W�>8��        )��P	/�Ǒ���A�
*

	conv_loss谓>��0        )��P	.Aȑ���A�
*

	conv_loss5��>��9        )��P	��ȑ���A�
*

	conv_lossݥ�>(}>8        )��P	<�ȑ���A�
*

	conv_loss���>��b        )��P	�	ɑ���A�
*

	conv_loss�Ƌ>I��C        )��P	�Fɑ���A�
*

	conv_loss)͐>L@��        )��P	
�ɑ���A�
*

	conv_loss.�>�`��        )��P	�ɑ���A�
*

	conv_lossꦎ>/	s8        )��P	% ʑ���A�
*

	conv_loss��>�`�%        )��P	#Eʑ���A�
*

	conv_loss��>+�r�        )��P	]�ʑ���A�
*

	conv_loss��>�ض�        )��P	��ʑ���A�
*

	conv_loss?,�>Q�/�        )��P	8ˑ���A�
*

	conv_loss���> a        )��P	zXˑ���A�
*

	conv_loss��>#L�7        )��P	��ˑ���A�
*

	conv_lossVx�>8{�        )��P	��ˑ���A�
*

	conv_loss�1�>��]�        )��P	�̑���A�
*

	conv_loss�!�>�6�        )��P	gS̑���A�
*

	conv_loss/��>�0,        )��P	]�̑���A�
*

	conv_loss�ޔ>���        )��P	c�̑���A�
*

	conv_loss��>��D;        )��P	�͑���A�
*

	conv_lossc�>x�        )��P	&O͑���A�
*

	conv_loss`�>e�Ƴ        )��P	��͑���A�
*

	conv_lossс�>݂<�        )��P	��͑���A�
*

	conv_loss�ڍ>��        )��P	)Α���A�
*

	conv_loss���>����        )��P	FΑ���A�
*

	conv_loss�`�>r��b        )��P	C�Α���A�
*

	conv_loss}��>>�~        )��P	��Α���A�
*

	conv_loss?ō>q��        )��P	l�Α���A�
*

	conv_loss`M�>`�g�        )��P	@>ϑ���A�
*

	conv_loss4�>s�4u        )��P	{ϑ���A�
*

	conv_loss��>o�>�        )��P	��ϑ���A�
*

	conv_loss��>7{kk        )��P	��ϑ���A�
*

	conv_loss���>��.        )��P	�3Б���A�
*

	conv_loss�e�>!��        )��P	sБ���A�
*

	conv_loss�G�>���b        )��P	ïБ���A�
*

	conv_lossČ>�:�P        )��P	�Б���A�
*

	conv_lossc>�>��L�        )��P	�*ё���A�
*

	conv_loss�h�>����        )��P	�hё���A�
*

	conv_loss��>�Lσ        )��P	��ё���A�
*

	conv_loss��>�)�        )��P	��ё���A�
*

	conv_loss��>�|��        )��P	T#ґ���A�
*

	conv_loss=�>���Y        )��P	�_ґ���A�
*

	conv_loss�!�>:��        )��P	Ӱґ���A�
*

	conv_loss�/�>�2�        )��P	��ґ���A�
*

	conv_loss?ݍ>��9        )��P	�,ӑ���A�
*

	conv_lossն�>�~��        )��P	.nӑ���A�
*

	conv_loss���>���        )��P	W�ӑ���A�
*

	conv_lossmj�>ݩB�        )��P	+�ӑ���A�
*

	conv_loss���>(��G        )��P	+&ԑ���A�
*

	conv_loss���>	��        )��P	:qԑ���A�
*

	conv_loss���>j(K        )��P	��ԑ���A�
*

	conv_loss�z�>v        )��P	��ԑ���A�
*

	conv_lossu�>�h��        )��P	~5Ց���A�
*

	conv_loss���> ���        )��P	z�Ց���A�
*

	conv_loss�>���        )��P	��Ց���A�
*

	conv_loss���>��sP        )��P	��Ց���A�
*

	conv_loss�]�>�P        )��P	;֑���A�
*

	conv_loss!�>ԅu        )��P	�y֑���A�
*

	conv_lossb��>����        )��P	%�֑���A�
*

	conv_loss�ٔ> Ϙ�        )��P	�֑���A�
*

	conv_loss���>&T�        )��P	u3ב���A�
*

	conv_lossf�>pd*�        )��P	7qב���A�
*

	conv_loss�:�>�sM�        )��P	�ב���A�
*

	conv_loss2�>�{�        )��P	�ב���A�
*

	conv_lossa�>w `T        )��P	+ؑ���A�
*

	conv_loss���>��        )��P	�iؑ���A�
*

	conv_loss9-�>^Y�        )��P	��ؑ���A�
*

	conv_loss�ڐ>��6p        )��P	��ؑ���A�
*

	conv_loss)0�>��e        )��P	"ّ���A�
*

	conv_loss7.�>Ng��        )��P	�_ّ���A�
*

	conv_losst.�>ݰ        )��P	�ّ���A�
*

	conv_loss�`�>:`�        )��P	��ّ���A�
*

	conv_loss	�>C�A        )��P	Cڑ���A�
*

	conv_loss��>n�3�        )��P	uVڑ���A�
*

	conv_lossHy�>� ��        )��P	��ڑ���A�
*

	conv_lossO5�> '�        )��P	��ڑ���A�
*

	conv_loss��>�}w        )��P	Dۑ���A�
*

	conv_loss�l�>�I�        )��P	JJۑ���A�
*

	conv_lossjA�>��Uk        )��P	:�ۑ���A�
*

	conv_loss���>�>�        )��P	�ۑ���A�
*

	conv_loss�\�>卵n        )��P	ܑ���A�
*

	conv_lossY�>�nd        )��P	Rܑ���A�
*

	conv_loss�މ>��A�        )��P	)�ܑ���A�
*

	conv_lossA��>�Y��        )��P	��ܑ���A�
*

	conv_loss��>�@y*        )��P	 ݑ���A�
*

	conv_loss c�>�N�        )��P	�^ݑ���A�
*

	conv_loss�ۓ>�7        )��P		�ݑ���A�
*

	conv_loss�{�>���        )��P	�ݑ���A�
*

	conv_lossCr�>a�M        )��P	ޑ���A�
*

	conv_loss},�>1�        )��P	�Tޑ���A�
*

	conv_loss���> �`�        )��P	j�ޑ���A�
*

	conv_lossmI�>��8e        )��P	��ޑ���A�
*

	conv_lossK�>ã��        )��P	�������A�
*

	conv_lossOǈ>��        )��P	$ᑉ��A�
*

	conv_loss���>>�|(        )��P	z`ᑉ��A�
*

	conv_lossU�>�6�        )��P	��ᑉ��A�
*

	conv_loss�6�>!�f�        )��P	�ᑉ��A�
*

	conv_loss��>Щ@u        )��P	�#⑉��A�
*

	conv_loss��>��        )��P	�`⑉��A�
*

	conv_loss�	�>���        )��P	�⑉��A�*

	conv_loss�(�>	?�        )��P	�⑉��A�*

	conv_loss �>M��        )��P	�'㑉��A�*

	conv_lossI�>]��        )��P	Ge㑉��A�*

	conv_lossD�>4�w        )��P	��㑉��A�*

	conv_loss���>���w        )��P	S�㑉��A�*

	conv_loss���>���H        )��P	i(䑉��A�*

	conv_lossy(�>�+        )��P	�k䑉��A�*

	conv_loss��>urd        )��P	�䑉��A�*

	conv_loss���>~�H�        )��P	�䑉��A�*

	conv_loss���>N��        )��P	'&呉��A�*

	conv_loss�<�>����        )��P	(d呉��A�*

	conv_loss�n�>[1L�        )��P	��呉��A�*

	conv_loss��>�͝B        )��P	��呉��A�*

	conv_loss��>^�6�        )��P	[摉��A�*

	conv_loss�|�>����        )��P	[摉��A�*

	conv_loss���>ߗ�        )��P	��摉��A�*

	conv_loss��>�kd�        )��P	Q�摉��A�*

	conv_loss���> /XA        )��P	�瑉��A�*

	conv_lossY��>�3**        )��P	�T瑉��A�*

	conv_loss�م>��66        )��P	��瑉��A�*

	conv_loss���>k|u        )��P	��瑉��A�*

	conv_loss�I�>�0        )��P	�葉��A�*

	conv_loss�؉>���v        )��P	�H葉��A�*

	conv_loss���>���
        )��P	�葉��A�*

	conv_loss���>4%p        )��P	��葉��A�*

	conv_loss��>�XC�        )��P	�鑉��A�*

	conv_loss�)�>2H��        )��P	_A鑉��A�*

	conv_loss�q�>��        )��P	�~鑉��A�*

	conv_loss��>���g        )��P	��鑉��A�*

	conv_lossՆ>����        )��P	d�鑉��A�*

	conv_loss��>���~        )��P	�7ꑉ��A�*

	conv_loss/s�>����        )��P	�tꑉ��A�*

	conv_loss�o�>H0�        )��P	��ꑉ��A�*

	conv_loss��>�v-        )��P	��ꑉ��A�*

	conv_lossS�>��!        )��P	�)둉��A�*

	conv_lossZ2�>��        )��P	�e둉��A�*

	conv_lossW�>�0        )��P	�둉��A�*

	conv_loss�ʆ>�CW�        )��P	��둉��A�*

	conv_loss��>��v        )��P	�쑉��A�*

	conv_loss<4�>����        )��P	�\쑉��A�*

	conv_loss�·>1� �        )��P	3�쑉��A�*

	conv_loss쳊>���        )��P	��쑉��A�*

	conv_lossw��>�-	        )��P	�$푉��A�*

	conv_loss� �>�qN�        )��P	tc푉��A�*

	conv_loss�g�><�.Z        )��P	�푉��A�*

	conv_lossQC�>+bXH        )��P	��푉��A�*

	conv_loss���>s�A�        )��P	�;��A�*

	conv_loss�-�> [�        )��P	�|��A�*

	conv_lossό�>s���        )��P	����A�*

	conv_lossU�>��        )��P	7��A�*

	conv_loss�r�>���P        )��P	�H��A�*

	conv_loss�،>���        )��P	����A�*

	conv_loss,�>1��X        )��P	'���A�*

	conv_lossS�>O#        )��P	�𑉙�A�*

	conv_loss���>�]��        )��P	�>𑉙�A�*

	conv_loss� �>���        )��P	_�𑉙�A�*

	conv_lossU�>�r�        )��P	S�𑉙�A�*

	conv_loss���>�        )��P	C񑉙�A�*

	conv_loss^Չ>�]v        )��P	�\񑉙�A�*

	conv_loss�B�>vud�        )��P	��񑉙�A�*

	conv_loss�!�>�qQa        )��P	2�񑉙�A�*

	conv_loss��>�ҡ�        )��P	�򑉙�A�*

	conv_loss�0�>��<        )��P	�S򑉙�A�*

	conv_loss���>�V        )��P	ב򑉙�A�*

	conv_loss9��>B��        )��P	��򑉙�A�*

	conv_loss(�>��3�        )��P	�󑉙�A�*

	conv_loss���>���}        )��P	jW󑉙�A�*

	conv_loss��>���        )��P	P�󑉙�A�*

	conv_loss�E�>���Q        )��P	��󑉙�A�*

	conv_losso��>�`        )��P	������A�*

	conv_loss��>�f�        )��P	3X�����A�*

	conv_loss��>�-e        )��P	Ԕ�����A�*

	conv_lossŪ�>�t$�        )��P	>������A�*

	conv_loss�#�>��o        )��P	*�����A�*

	conv_loss�ދ>�55        )��P	"M�����A�*

	conv_loss�ރ>�F��        )��P	J������A�*

	conv_lossn�>%?�        )��P	������A�*

	conv_loss��>l&<�        )��P	?�����A�*

	conv_loss�Á>|�A        )��P	�D�����A�*

	conv_lossl(�>�~v#        )��P	Յ�����A�*

	conv_loss��>D��d        )��P	Y������A�*

	conv_loss��>�        )��P	� �����A�*

	conv_losss(�>icq�        )��P	 ?�����A�*

	conv_loss�-�>��        )��P	}�����A�*

	conv_loss��>��_        )��P		������A�*

	conv_loss��>���;        )��P	�������A�*

	conv_lossb�>
�q        )��P	�4�����A�*

	conv_loss=h�>���        )��P	�r�����A�*

	conv_loss#L}>��m�        )��P	������A�*

	conv_loss\+�>��9:        )��P	~������A�*

	conv_loss���>��        )��P	�*�����A�*

	conv_loss�ˉ>�:9        )��P	�������A�*

	conv_loss�{>
jRD        )��P	q������A�*

	conv_loss��>��&        )��P	>.�����A�*

	conv_loss�݈>(         )��P	m�����A�*

	conv_loss�ۆ>��g        )��P	^������A�*

	conv_loss��>�e        )��P	X������A�*

	conv_loss�և>Mca�        )��P	�+�����A�*

	conv_loss�H�>���        )��P	�h�����A�*

	conv_loss_=�>Y��        )��P	A������A�*

	conv_loss��>6y'        )��P	�������A�*

	conv_loss�ă>klx(        )��P	e> ����A�*

	conv_lossF҇>��6�        )��P	�{ ����A�*

	conv_lossU��>I��P        )��P	+� ����A�*

	conv_lossog�>%��n        )��P	N� ����A�*

	conv_loss��>����        )��P	K;����A�*

	conv_loss�f�>R��p        )��P	�x����A�*

	conv_loss��>y=k:        )��P	l�����A�*

	conv_loss�$�>�_        )��P	������A�*

	conv_loss��>���$        )��P	J0����A�*

	conv_loss��>����        )��P	fs����A�*

	conv_loss�\�>�D��        )��P	������A�*

	conv_loss���>
�I�        )��P	������A�*

	conv_loss�R�>FD�        )��P	�3����A�*

	conv_loss���>�;��        )��P	�p����A�*

	conv_loss'��>��dv        )��P	�����A�*

	conv_loss�څ>�0v`        )��P	�����A�*

	conv_loss�2�>�+�        )��P	�'����A�*

	conv_loss���>��        )��P	�e����A�*

	conv_loss�q�>X"l        )��P	ţ����A�*

	conv_loss�ۄ>���n        )��P	������A�*

	conv_loss��>�|        )��P	V����A�*

	conv_lossȉ>��d�        )��P	 \����A�*

	conv_lossN�>:Hg�        )��P	H�����A�*

	conv_loss��>���n        )��P	x�����A�*

	conv_lossZor>� �        )��P	m����A�*

	conv_loss�>\��P        )��P	�T����A�*

	conv_loss��>��        )��P	ϑ����A�*

	conv_lossq�>��K�        )��P	<�����A�*

	conv_lossu��>�xC        )��P	�����A�*

	conv_loss��>dw��        )��P	�K����A�*

	conv_loss?:�>�د9        )��P	^�����A�*

	conv_loss]U�>~�+�        )��P	������A�*

	conv_loss+[�>�G�$        )��P	�����A�*

	conv_loss�>�=��        )��P	�B����A�*

	conv_lossh5�>�La�        )��P	�����A�*

	conv_loss���>]�i        )��P	 �����A�*

	conv_lossd�{>�2�m        )��P	C�����A�*

	conv_loss�U�>���        )��P	�8	����A�*

	conv_loss1��>RI4        )��P	�w	����A�*

	conv_lossD�>���[        )��P	��	����A�*

	conv_loss�/�>�w7U        )��P	_�	����A�*

	conv_loss�#�>��)*        )��P	F
����A�*

	conv_loss붃>�Z        )��P	��
����A�*

	conv_loss� �>�q�        )��P	��
����A�*

	conv_loss�?~>Y'|�        )��P	�����A�*

	conv_loss�"�>s��        )��P	�?����A�*

	conv_loss�O�>���$        )��P	�}����A�*

	conv_loss�ԁ>G�69        )��P	ƺ����A�*

	conv_loss\V�>�z�        )��P	�����A�*

	conv_loss�c�>i[0-        )��P	F����A�*

	conv_loss�R�>"
�        )��P	X�����A�*

	conv_loss6~�>k~��        )��P	������A�*

	conv_loss"�>��        )��P	�����A�*

	conv_loss�8�>Tg�        )��P	�Z����A�*

	conv_lossJ�|>��3>        )��P	M�����A�*

	conv_loss�}>�>�        )��P	������A�*

	conv_loss�A�>>��'        )��P	-����A�*

	conv_loss���>?̜�        )��P	�Q����A�*

	conv_loss�"�>{'L�        )��P	������A�*

	conv_loss�]�>�Lf�        )��P	������A�*

	conv_loss�у>4<��        )��P		����A�*

	conv_loss��{>��h        )��P	H����A�*

	conv_loss��>}y��        )��P	o�����A�*

	conv_lossw8>.}�        )��P	n�����A�*

	conv_loss�U�>	�[�        )��P	�����A�*

	conv_loss� �>��.�        )��P	�A����A�*

	conv_loss1�>k#�/        )��P	�����A�*

	conv_loss��>%A��        )��P	������A�*

	conv_losss�|>��K        )��P	������A�*

	conv_loss�E{>�'�        )��P	�9����A�*

	conv_loss;�|>��X�        )��P	�y����A�*

	conv_lossUÁ>T�'�        )��P	A�����A�*

	conv_loss��>�)��        )��P	�����A�*

	conv_loss�j~>V�<"        )��P	�3����A�*

	conv_lossd�z>�i�        )��P	q����A�*

	conv_loss��>���        )��P	�����A�*

	conv_lossET�>���        )��P	L�����A�*

	conv_loss��><�o�        )��P	z)����A�*

	conv_lossI*�>N��        )��P	 f����A�*

	conv_lossI��>��3        )��P	�����A�*

	conv_lossZ�>�Je        )��P	������A�*

	conv_loss�Q�>.Y�c        )��P	�����A�*

	conv_loss�(�>�@Z        )��P	�Y����A�*

	conv_loss,I�>��5        )��P	������A�*

	conv_losss�>2=�        )��P	������A�*

	conv_lossF;~>~"�Y        )��P	5����A�*

	conv_loss,�>'�3        )��P	,N����A�*

	conv_lossF�>�!        )��P	������A�*

	conv_loss��w>)��        )��P	�����A�*

	conv_lossM�|>0        )��P	�����A�*

	conv_loss���>����        )��P	�F����A�*

	conv_loss���>l��        )��P	S`����A�*

	conv_loss�|>�ͤ        )��P	w�����A�*

	conv_loss}>�n        )��P	 �����A�*

	conv_lossCl�>Ur1�        )��P	R)����A�*

	conv_loss�U�>si_�        )��P	Mr����A�*

	conv_loss|u>J`|�        )��P	�����A�*

	conv_lossK9>��5D        )��P	������A�*

	conv_loss݁>��        )��P	$>����A�*

	conv_loss��>^��1        )��P	B�����A�*

	conv_lossɑ�>��%�        )��P	������A�*

	conv_loss�K�>iR�        )��P	������A�*

	conv_lossy�u>4�D�        )��P	�:����A�*

	conv_lossC{>�H�R        )��P	�����A�*

	conv_lossQ��>c
!�        )��P	������A�*

	conv_loss��>3�P�        )��P	q����A�*

	conv_loss��>��
        )��P	>����A�*

	conv_loss�>p��        )��P	�z����A�*

	conv_loss��>��Yk        )��P	*�����A�*

	conv_lossJ�>uNS�        )��P	�����A�*

	conv_loss���>���        )��P	�1����A�*

	conv_loss�Ȅ>�-�l        )��P	;o����A�*

	conv_loss��{>{pW        )��P	U�����A�*

	conv_loss/�~>�a�        )��P	������A�*

	conv_loss@Gz>X8��        )��P	w(����A�*

	conv_loss��>�6�        )��P	f����A�*

	conv_lossv��>3[��        )��P	������A�*

	conv_loss��}>0H�        )��P	������A�*

	conv_loss��v>:�y        )��P	�����A�*

	conv_lossp}>��Pd        )��P	�Z����A�*

	conv_loss���>�}T        )��P	3�����A�*

	conv_loss��|> ��_        )��P	������A�*

	conv_loss8�>_	�1        )��P	 ����A�*

	conv_loss�ۀ>Q�.        )��P	rN ����A�*

	conv_lossd��>!.�        )��P	p� ����A�*

	conv_loss��>���u        )��P	�� ����A�*

	conv_loss�:�>�n�        )��P	�!����A�*

	conv_loss��>5��        )��P	G!����A�*

	conv_loss���>�c�        )��P	�!����A�*

	conv_loss�}>���        )��P	)�!����A�*

	conv_loss�y>-`�        )��P	��!����A�*

	conv_loss��>�]��        )��P	�;"����A�*

	conv_lossݿ�>�d�        )��P	�y"����A�*

	conv_lossaX~>1o!�        )��P	�"����A�*

	conv_loss��{>fe��        )��P	��"����A�*

	conv_loss�F�> ]=7        )��P	�1#����A�*

	conv_loss��z>cch�        )��P	�m#����A�*

	conv_lossh0�>PZPP        )��P	@�#����A�*

	conv_loss�6q>L�̠        )��P	��#����A�*

	conv_loss���>W�        )��P	K%$����A�*

	conv_lossɾ|>�b{        )��P	�b$����A�*

	conv_lossl}�>���        )��P	8�$����A�*

	conv_lossH�{>.U�        )��P	?�$����A�*

	conv_lossG�u>��        )��P	�+%����A�*

	conv_loss
e~>]��        )��P	�o%����A�*

	conv_loss8�y>8h��        )��P	��%����A�*

	conv_loss�|>���N        )��P	�&����A�*

	conv_loss�*q>����        )��P	N&����A�*

	conv_lossJ�z>%��        )��P	�&����A�*

	conv_loss��v>"�        )��P	��&����A�*

	conv_loss�y>$�
        )��P	�'����A�*

	conv_loss$|>�ge        )��P	�M'����A�*

	conv_loss�%t>is�o        )��P	k�'����A�*

	conv_loss�Uz>Q?�        )��P	
�'����A�*

	conv_loss�?�>t��;        )��P	-(����A�*

	conv_loss�܂>��в        )��P	5P(����A�*

	conv_loss�ǀ>�]        )��P	��(����A�*

	conv_loss{�~>�7��        )��P	S�(����A�*

	conv_loss�؂>86        )��P	�)����A�*

	conv_loss}>��R�        )��P	aJ)����A�*

	conv_lossZ�x>�7�        )��P	ʆ)����A�*

	conv_loss�Z�>P5�        )��P	��)����A�*

	conv_loss�m�>7�Ʈ        )��P	7*����A�*

	conv_loss�>7j�         )��P	zA*����A�*

	conv_loss�d�>�C�q        )��P	�~*����A�*

	conv_loss� �>�ҟ        )��P	Ҽ*����A�*

	conv_lossvx>��D        )��P	q�*����A�*

	conv_loss�ф>w�3        )��P	�9+����A�*

	conv_loss2X~>�gj        )��P	 v+����A�*

	conv_loss�s>�,�        )��P	m�+����A�*

	conv_loss#(}>/�.        )��P	w�+����A�*

	conv_loss��~>1a(        )��P	L.,����A�*

	conv_loss> �>�Y��        )��P	�k,����A�*

	conv_loss}&{>��Ȗ        )��P	�,����A�*

	conv_loss�d�>��        )��P	1�,����A�*

	conv_lossǀ~>��C%        )��P	Z"-����A�*

	conv_loss�>��[        )��P	�`-����A�*

	conv_loss�q{>���        )��P	=�-����A�*

	conv_loss,�s>3�?�        )��P	=�-����A�*

	conv_loss�>8�1        )��P	d.����A�*

	conv_loss�ix>\�        )��P	�U.����A�*

	conv_loss
>�K�!        )��P	:�.����A�*

	conv_loss�Â>��        )��P	��.����A�*

	conv_loss��>����        )��P	h/����A�*

	conv_loss0�>�D�q        )��P	O/����A�*

	conv_loss.Cu>-��z        )��P	��/����A�*

	conv_loss9!{>P[�        )��P	G�/����A�*

	conv_loss�3w>���        )��P	�0����A�*

	conv_loss���>��Y�        )��P	�@0����A�*

	conv_loss��p>��<        )��P	R}0����A�*

	conv_loss�Xz>{�,        )��P	��0����A�*

	conv_loss_N|>3ᚱ        )��P	
1����A�*

	conv_loss6}>�ݭ        )��P	+G1����A�*

	conv_loss�v>a��        )��P	��1����A�*

	conv_loss_M�>�D�        )��P	��1����A�*

	conv_loss��s>�%�o        )��P	2����A�*

	conv_lossT:y>f�/        )��P	�e2����A�*

	conv_loss<n>	V�        )��P	έ2����A�*

	conv_loss��>V��        )��P	�2����A�*

	conv_loss���>��m        )��P	�43����A�*

	conv_loss�me>���        )��P	�r3����A�*

	conv_loss�v>4s�        )��P	��3����A�*

	conv_loss��r>����        )��P	��3����A�*

	conv_loss��w>Lv�        )��P	�(4����A�*

	conv_loss��p>N��N        )��P	�g4����A�*

	conv_loss3qy>4j)        )��P	*�4����A�*

	conv_loss�-y>��        )��P	6�4����A�*

	conv_lossT �>M�\e        )��P	o25����A�*

	conv_loss4�s>�&�        )��P	Uq5����A�*

	conv_lossm�~>$��        )��P	��5����A�*

	conv_loss��v>�\v0        )��P	�5����A�*

	conv_loss(�v>]�        )��P	�)6����A�*

	conv_loss-�r>�!��        )��P	�x6����A�*

	conv_loss��s>S��        )��P	��6����A�*

	conv_loss�Pq>�z��        )��P	��6����A�*

	conv_loss�ց>��ǖ        )��P	�17����A�*

	conv_lossT�i>�nR�        )��P	�o7����A�*

	conv_lossC�v>(�        )��P	��7����A�*

	conv_loss$#{>\WFM        )��P	�7����A�*

	conv_loss#>}>�'׊        )��P	n&8����A�*

	conv_loss�v>O!�        )��P	�b8����A�*

	conv_loss�zo>"3�        )��P	̟8����A�*

	conv_loss���>�x]        )��P	��8����A�*

	conv_loss��w>��        )��P	*9����A�*

	conv_lossI�x>ػ�        )��P	Ri9����A�*

	conv_lossk�o>��^        )��P	�9����A�*

	conv_loss�{>�aK        )��P	�9����A�*

	conv_loss���>�        )��P	S :����A�*

	conv_loss���>6�        )��P	�\:����A�*

	conv_loss�vt>�פ�        )��P	m�:����A�*

	conv_lossΙ}>�Ym        )��P	��:����A�*

	conv_loss9Uo>�xө        )��P	E;����A�*

	conv_loss���>����        )��P	MR;����A�*

	conv_loss z>�]_        )��P	M�;����A�*

	conv_loss[�t>Q��K        )��P	��;����A�*

	conv_lossc}>��@�        )��P	�<����A�*

	conv_loss��l>h-p�        )��P	�I<����A�*

	conv_loss��t>�T��        )��P	��<����A�*

	conv_loss��r>��J;        )��P	/�<����A�*

	conv_lossWcy>U�`        )��P	�=����A�*

	conv_loss��z>?�`�        )��P	MS=����A�*

	conv_loss0�t>��        )��P	x�=����A�*

	conv_loss�|>��        )��P	��=����A�*

	conv_loss�q>Y��        )��P	�>����A�*

	conv_loss�=u>H��        )��P	d_>����A�*

	conv_lossVRi>5h#�        )��P	��>����A�*

	conv_loss�kz>����        )��P	/�>����A�*

	conv_loss]4z>m�D�        )��P	$1?����A�*

	conv_lossi�>b��$        )��P	9?����A�*

	conv_loss��x>���        )��P	��?����A�*

	conv_loss�s>���        )��P	r�?����A�*

	conv_loss�p> ��        )��P	�9@����A�*

	conv_loss��n>bn�`        )��P	r�@����A�*

	conv_lossX$�>s_��        )��P	��@����A�*

	conv_loss��u>����        )��P	e�@����A�*

	conv_loss��y>T�y        )��P	gAA����A�*

	conv_lossj��>e�t�        )��P	�A����A�*

	conv_loss�jo>6�|        )��P	�A����A�*

	conv_loss5�>����        )��P	�B����A�*

	conv_loss��x>�fv�        )��P	�AB����A�*

	conv_lossKUt>{�a        )��P	�~B����A�*

	conv_lossÖ�>�<��        )��P	x�B����A�*

	conv_lossZw>��        )��P	j�B����A�*

	conv_loss@uu>�M �        )��P	G6C����A�*

	conv_loss m>шW�        )��P	�rC����A�*

	conv_loss�Q{>���        )��P	U�C����A�*

	conv_loss�(|>����        )��P	��C����A�*

	conv_loss~s>���[        )��P	|+D����A�*

	conv_loss�
d>N���        )��P	�iD����A�*

	conv_loss�3t>����        )��P	֥D����A�*

	conv_loss�x>K�t        )��P	�D����A�*

	conv_lossakj>ٓ        )��P	/!E����A�*

	conv_loss� i>�3�        )��P	']E����A�*

	conv_lossc@i>��        )��P	]�E����A�*

	conv_loss��u>&���        )��P	k�E����A�*

	conv_lossXw>�         )��P	WF����A�*

	conv_loss=�p>�ﮑ        )��P	JQF����A�*

	conv_loss��u>_�L        )��P	��F����A�*

	conv_loss��f>o���        )��P	��F����A�*

	conv_loss��r>���        )��P	�	G����A�*

	conv_loss��p>m�A�        )��P	HG����A�*

	conv_loss�x>��O�        )��P	�G����A�*

	conv_losseUm>�eg        )��P	��G����A�*

	conv_loss:Hf>G |�        )��P	�H����A�*

	conv_lossxuo>gs��        )��P	�?H����A�*

	conv_loss~�s>�s        )��P	�|H����A�*

	conv_lossR�w>�o�        )��P	��H����A�*

	conv_lossE�r>��%        )��P	�H����A�*

	conv_loss)�|>v]�1        )��P	o4I����A�*

	conv_lossu2q>sj�O        )��P	TsI����A�*

	conv_loss~�|>N��        )��P	"�I����A�*

	conv_losssLn>���j        )��P	}�K����A�*

	conv_loss�n>�h        )��P	XL����A�*

	conv_loss���>�'w�        )��P	�UL����A�*

	conv_loss��p>��ns        )��P	�L����A�*

	conv_loss��~>���        )��P	S�L����A�*

	conv_lossY�o>��s        )��P	�M����A�*

	conv_loss��q>���N        )��P	^ZM����A�*

	conv_lossUw>�L[�        )��P	W�M����A�*

	conv_loss �w>y�c        )��P	��M����A�*

	conv_lossj�j>�0(�        )��P	22N����A�*

	conv_loss�l>��q�        )��P	eqN����A�*

	conv_loss��u>�R�        )��P	�N����A�*

	conv_loss��d>�"{        )��P	k�N����A�*

	conv_lossȽy>Q�~        )��P	�(O����A�*

	conv_loss&�p>ߋx        )��P	JfO����A�*

	conv_loss/�k>(3        )��P	�O����A�*

	conv_lossGk>R�        )��P	��O����A�*

	conv_loss,�n>#+Lj        )��P	�P����A�*

	conv_lossC�p>ʼ        )��P	�\P����A�*

	conv_loss�=f>�2�        )��P	�P����A�*

	conv_loss,&d>����        )��P	x�P����A�*

	conv_loss%Hx> ɘB        )��P	�Q����A�*

	conv_loss�\|>k)        )��P	{QQ����A�*

	conv_loss�g>w�<�        )��P	8�Q����A�*

	conv_loss��]>Bt��        )��P	3�Q����A�*

	conv_loss�_u>K���        )��P	hR����A�*

	conv_losssBw>~j�        )��P	fER����A�*

	conv_lossz}l>�4        )��P	~�R����A�*

	conv_loss�Sh>U�O        )��P	��R����A�*

	conv_loss*�m>i �        )��P	5�R����A�*

	conv_loss��y>}�        )��P	w;S����A�*

	conv_loss�n>�[�        )��P	 yS����A�*

	conv_loss��u>�;        )��P	;�S����A�*

	conv_loss�8l>r=�        )��P	��S����A�*

	conv_lossl@k>��{A        )��P	d2T����A�*

	conv_lossONm>Fk.�        )��P	1nT����A�*

	conv_losssl>���        )��P	u�T����A�*

	conv_loss
b>��'�        )��P	��T����A�*

	conv_lossXZs>����        )��P	)&U����A�*

	conv_lossi>c�ل        )��P	�cU����A�*

	conv_lossM�c>��j        )��P	��U����A�*

	conv_lossJ�~>Z*�        )��P	��U����A�*

	conv_loss%�w>t��        )��P	�V����A�*

	conv_loss�p>���        )��P	�UV����A�*

	conv_loss��q>[j�        )��P	ْV����A�*

	conv_loss��d>�245        )��P	{�V����A�*

	conv_lossV�j>F�[        )��P	\W����A�*

	conv_loss��m>ᜉ�        )��P	�IW����A�*

	conv_loss��k>�r�y        )��P	��W����A�*

	conv_loss��g>�+N        )��P	s�W����A�*

	conv_lossn*h>�8�        )��P	eX����A�*

	conv_loss�,~>�5L        )��P	�OX����A�*

	conv_loss��n>��(�        )��P	_�X����A�*

	conv_loss�+n>�R        )��P	��X����A�*

	conv_loss��t>8��\        )��P	9Y����A�*

	conv_loss�m>���w        )��P	�SY����A�*

	conv_loss��i>Ы��        )��P	��Y����A�*

	conv_loss*�z>���        )��P	��Y����A�*

	conv_lossD�m>o}��        )��P	6Z����A�*

	conv_loss<�u>Y���        )��P	�bZ����A�*

	conv_loss�i>v�W�        )��P	��Z����A�*

	conv_loss\>�K�        )��P	��Z����A�*

	conv_loss�p>���        )��P	�#[����A�*

	conv_loss.-i>%{4        )��P	�a[����A�*

	conv_loss��k>��q        )��P	��[����A�*

	conv_loss�Do>S[�!        )��P	�[����A�*

	conv_lossSus>q;��        )��P	2\����A�*

	conv_loss�y>�        )��P	<R\����A�*

	conv_lossm�y>���]        )��P	(�\����A�*

	conv_loss�,j>M��        )��P	u�\����A�*

	conv_loss�`}>%Nq�        )��P	�]����A�*

	conv_lossM3k>�:U�        )��P	�H]����A�*

	conv_loss[p>'��X        )��P	�]����A�*

	conv_loss��j>A��	        )��P	��]����A�*

	conv_loss�d>	�        )��P	0^����A�*

	conv_loss�e>�~�        )��P	b>^����A�*

	conv_loss!Pl>��J�        )��P	nz^����A�*

	conv_loss�)m>䫉�        )��P	=�^����A�*

	conv_loss�*a>
��        )��P	+�^����A�*

	conv_loss��l>��N�        )��P	+2_����A�*

	conv_loss�w>'��/        )��P	\o_����A�*

	conv_loss"f>>�#�        )��P	?�_����A�*

	conv_loss	g>�/H�        )��P	��_����A�*

	conv_loss3p>�y�v        )��P	@'`����A�*

	conv_lossky>��q        )��P	�c`����A�*

	conv_loss6�h>Y �        )��P	M�`����A�*

	conv_loss)Be>���        )��P	�`����A�*

	conv_lossTBs>+���        )��P	�a����A�*

	conv_loss��h>辒�        )��P	�Xa����A�*

	conv_loss�(g>�,�I        )��P	��a����A�*

	conv_lossv�y>��1�        )��P	��a����A�*

	conv_loss8>r>��k�        )��P	�b����A�*

	conv_loss��d>V��        )��P	UPb����A�*

	conv_loss�j>?c�        )��P	��b����A�*

	conv_loss�Ov>۾g�        )��P	z�b����A�*

	conv_loss]ob>�'��        )��P	a	c����A�*

	conv_loss�<�>�]j�        )��P	�Ic����A�*

	conv_loss>�n>��_j        )��P	�c����A�*

	conv_loss_|m>��>�        )��P	��c����A�*

	conv_loss�gu>=(�]        )��P	d����A�*

	conv_loss�xy>�y��        )��P	iad����A�*

	conv_loss+k>*���        )��P	y�d����A�*

	conv_loss��p>�΁        )��P	��d����A�*

	conv_loss�Xm>e��        )��P	b(e����A�*

	conv_loss��v>�N�        )��P	�ge����A�*

	conv_loss%LW>��	O        )��P	�e����A�*

	conv_loss"u>�%W�        )��P	w�e����A�*

	conv_lossb+n>�O9p        )��P	�8f����A�*

	conv_loss��m>��         )��P	�wf����A�*

	conv_loss�Ns>�۲        )��P	p�f����A�*

	conv_lossL�g>��/w        )��P	�g����A�*

	conv_lossg3_>ڔb        )��P		Bg����A�*

	conv_loss=k>al$�        )��P	.�g����A�*

	conv_loss��a>l���        )��P	�g����A�*

	conv_losss�i>qh        )��P	q�g����A�*

	conv_loss�oq>��e�        )��P	d8h����A�*

	conv_loss�f>4КR        )��P	$xh����A�*

	conv_loss�^>wL�        )��P	��h����A�*

	conv_loss��d>�Â        )��P	��h����A�*

	conv_loss�g>+K_3        )��P	�1i����A�*

	conv_loss��g>�3��        )��P	}pi����A�*

	conv_loss�p}>��o        )��P	1�i����A�*

	conv_loss�Ie>�LE        )��P	m�i����A�*

	conv_loss�i>���        )��P	j(j����A�*

	conv_lossgf>�et�        )��P	�gj����A�*

	conv_loss�n>Ѣo/        )��P	�j����A�*

	conv_lossSo>�0        )��P	��j����A�*

	conv_loss�Ev>�Ր        )��P	|k����A�*

	conv_loss�r>bq�        )��P	�Zk����A�*

	conv_loss��h>��t�        )��P	��k����A�*

	conv_loss�e>�,q�        )��P	��k����A�*

	conv_loss��a>�W�_        )��P	�l����A�*

	conv_loss�&h>��|A        )��P	�Ol����A�*

	conv_loss��c>V��        )��P	0�l����A�*

	conv_loss.]>�ɿ&        )��P	\�l����A�*

	conv_loss6�x>%��        )��P	�	m����A�*

	conv_losss{v>�K9        )��P	:Gm����A�*

	conv_loss�h>�>�;        )��P	
�m����A�*

	conv_loss��d>_���        )��P	?�m����A�*

	conv_loss��b>Ť        )��P	�m����A�*

	conv_loss,�g>c�Q�        )��P	<n����A�*

	conv_lossdk>6��        )��P	�zn����A�*

	conv_lossz�s>���[        )��P	u�n����A�*

	conv_loss�rb>O2
�        )��P	,�n����A�*

	conv_loss�u^>��        )��P	�1o����A�*

	conv_loss�/h>&�        )��P	Kno����A�*

	conv_loss�ld>�9v�        )��P		�o����A�*

	conv_loss�4u>�p��        )��P	��o����A�*

	conv_lossޤ`>�՞�        )��P	�'p����A�*

	conv_loss��m>r        )��P	�ep����A�*

	conv_lossJs>3�        )��P	��p����A�*

	conv_loss�T>�oZ-        )��P	W�p����A�*

	conv_lossJ4j>-v 6        )��P	K/q����A�*

	conv_lossVFl>���        )��P	+rq����A�*

	conv_loss��[>	Җ        )��P	t�q����A�*

	conv_loss9v>��4�        )��P	��q����A�*

	conv_loss�f>G��        )��P	�=r����A�*

	conv_lossG�o>�L^        )��P	=�r����A�*

	conv_loss�	[>
�<        )��P	Z�r����A�*

	conv_loss/e>]�        )��P	%s����A�*

	conv_loss6z^>� [        )��P	gKs����A�*

	conv_loss-)c>͵�V        )��P	.�s����A�*

	conv_loss��n>��w        )��P	��s����A�*

	conv_loss�p>1<u]        )��P	�t����A�*

	conv_lossfli>ւ:�        )��P	 Vt����A�*

	conv_lossyk_>˳��        )��P	G�t����A�*

	conv_loss��n>0H��        )��P	��t����A�*

	conv_loss�Og>\p�        )��P	�u����A�*

	conv_loss��e>�
[y        )��P	�Ku����A�*

	conv_loss��_>+;��        )��P	�u����A�*

	conv_loss�/e>]�5M        )��P	�u����A�*

	conv_loss�l>��X        )��P	Fv����A�*

	conv_lossia>Hj�        )��P	\Bv����A�*

	conv_loss,RZ>�c�        )��P	��v����A�*

	conv_loss�&f>*�L        )��P	m�v����A�*

	conv_loss��X>����        )��P	�v����A�*

	conv_loss %\>�CH        )��P	�8w����A�*

	conv_loss�b>���        )��P	�uw����A�*

	conv_loss{d>w9�x        )��P	��w����A�*

	conv_loss�_>O`��        )��P	h�w����A�*

	conv_loss9�o>�	p	        )��P	N.x����A�*

	conv_lossjg>/�z        )��P	dlx����A�*

	conv_loss��r>�-c        )��P	��x����A�*

	conv_lossYa>H�A        )��P	(�x����A�*

	conv_lossF�n>�J|        )��P	^$y����A�*

	conv_loss��c>����        )��P	�ay����A�*

	conv_loss�yb>���        )��P	�y����A�*

	conv_lossm�T>3�u        )��P	'�y����A�*

	conv_loss��l>�c~�        )��P	z����A�*

	conv_loss�[d>�f�	        )��P	�Tz����A�*

	conv_loss��`>ݶ~�        )��P	��z����A�*

	conv_losspj>E|�M        )��P	[�z����A�*

	conv_loss��[>��M        )��P	`{����A�*

	conv_losswI^>ߨ=        )��P	AK{����A�*

	conv_lossM�e>��        )��P	��{����A�*

	conv_lossf�i>!�<�        )��P	��{����A�*

	conv_loss��g>@��        )��P	�|����A�*

	conv_loss��k>k���        )��P	5?|����A�*

	conv_loss�>l>dV��        )��P	�{|����A�*

	conv_lossEb>��˱        )��P	�〒���A�*

	conv_loss�Z>��        )��P	3������A�*

	conv_loss��]>����        )��P	�6�����A�*

	conv_lossa�h>-k�        )��P	�s�����A�*

	conv_lossg`>*��        )��P	�������A�*

	conv_lossEV>p���        )��P	�󃒉��A�*

	conv_lossI"V>�d#        )��P	1�����A�*

	conv_loss�c>����        )��P	pv�����A�*

	conv_loss�eX>���<        )��P	�������A�*

	conv_loss��a>��gk        )��P	� �����A�*

	conv_lossE�Z>��h        )��P	�F�����A�*

	conv_loss��a>y�        )��P	l������A�*

	conv_loss�`>�5�v        )��P	؅����A�*

	conv_loss��d>grus        )��P	������A�*

	conv_loss�l>��j        )��P	iR�����A�*

	conv_loss˴o>�g�        )��P	������A�*

	conv_loss/f]>�E2�        )��P	�ˆ����A�*

	conv_loss��p>�6��        )��P	g�����A�*

	conv_loss֮a>�T��        )��P	 C�����A�*

	conv_loss�xV>-2�x        )��P	w�����A�*

	conv_loss=�b>o�7        )��P	�������A�*

	conv_loss4+\>�ǖ        )��P	;������A�*

	conv_lossp0_>/�        )��P	h5�����A�*

	conv_loss�HZ>���#        )��P	�q�����A�*

	conv_loss�>a>`�&�        )��P	'������A�*

	conv_lossc2b>h�U,        )��P	#툒���A�*

	conv_loss:�_>aXQ        )��P	�(�����A�*

	conv_lossq�`>R��        )��P	�e�����A�*

	conv_loss��c>`�{        )��P	@������A�*

	conv_lossTf^>o        )��P	�艒���A�*

	conv_loss�[>���d        )��P	:/�����A�*

	conv_lossi�U>rm        )��P	�o�����A�*

	conv_lossv	j>Ѝ�b        )��P	4������A�*

	conv_loss��i>h�q�        )��P	W������A�*

	conv_loss��U>"���        )��P	�9�����A�*

	conv_loss�n^>{�0_        )��P	+w�����A�*

	conv_loss�9f>H��        )��P	}������A�*

	conv_loss��n>:��-        )��P	n������A�*

	conv_lossQb>n*�        )��P	�-�����A�*

	conv_loss�ke>���?        )��P	Bk�����A�*

	conv_loss\e]>�        )��P	�������A�*

	conv_loss'�j>�`R&        )��P	�挒���A�*

	conv_lossE]>ع         )��P	$�����A�*

	conv_loss�a>�4��        )��P	�b�����A�*

	conv_loss��b>P;?k        )��P	������A�*

	conv_loss�/a>���        )��P	�ۍ����A�*

	conv_loss�=]>�UC�        )��P	������A�*

	conv_lossލY>1K�        )��P	�U�����A�*

	conv_loss�d>��՗        )��P	C������A�*

	conv_losse3c>۠Ĝ        )��P	Ҏ����A�*

	conv_loss/�^>`�K�        )��P	������A�*

	conv_loss�E]>c���        )��P	�a�����A�*

	conv_lossb�_>I�s        )��P	������A�*

	conv_loss�\>E��        )��P	�ޏ����A�*

	conv_loss4f>�V!k        )��P	!�����A�*

	conv_lossj�`>��        )��P	P_�����A�*

	conv_lossN�b>Ì�        )��P	D������A�*

	conv_loss0na>���"        )��P	�ꐒ���A�*

	conv_loss_�b>xj��        )��P	�-�����A�*

	conv_lossD2]>l{C�        )��P	�l�����A�*

	conv_loss�za>�<�        )��P	u������A�*

	conv_loss��]>]���        )��P	]������A�*

	conv_loss�kh>�F��        )��P	�7�����A�*

	conv_loss��b>�qt        )��P	�u�����A�*

	conv_loss2Vb>J��        )��P	�������A�*

	conv_loss�ch>\z�        )��P	s���A�*

	conv_lossK�`>N���        )��P	�,�����A�*

	conv_loss'[>+��0        )��P	Ej�����A�*

	conv_lossS�c>�M�        )��P	Q������A�*

	conv_loss�T`>�m�g        )��P	s哒���A�*

	conv_loss��h>��        )��P	K$�����A�*

	conv_loss�]W>��        )��P	�b�����A�*

	conv_loss��S>�\�        )��P	����A�*

	conv_loss
�a>��        )��P	ߔ����A�*

	conv_loss��Y>#���        )��P	������A�*

	conv_lossh>!��        )��P	%Z�����A�*

	conv_losshK[>35��        )��P	z������A�*

	conv_loss��h>�R��        )��P	�ؕ����A�*

	conv_lossgH>C��        )��P	\�����A�*

	conv_loss?i>��+'        )��P	�U�����A�*

	conv_loss�^Y>�t4�        )��P	M������A�*

	conv_lossoCT>�ރ�        )��P	%і����A�*

	conv_loss��\>?J�v        )��P	x�����A�*

	conv_lossh�Q>�$}.        )��P	�L�����A�*

	conv_lossdQY>�	Ѓ        )��P	⊗����A�*

	conv_loss�x[>h΃�        )��P	 ȗ����A�*

	conv_loss��U>u�b        )��P	}�����A�*

	conv_lossJ_>Z�Ϸ        )��P	C�����A�*

	conv_loss��V>�撡        )��P	�������A�*

	conv_loss��[>�<�        )��P	������A�*

	conv_lossp�_>����        )��P	�������A�*

	conv_loss��`>�=P/        )��P	�8�����A�*

	conv_loss�Jc>�w�_        )��P	�u�����A�*

	conv_loss�`>�]1�        )��P	W������A�*

	conv_loss�V>QP
�        )��P	j𙒉��A�*

	conv_lossP�Y>7=�        )��P	H-�����A�*

	conv_loss�V>7��-        )��P	Qj�����A�*

	conv_lossk�W>���        )��P	R������A�*

	conv_lossZ>+�[G        )��P	�䚒���A�*

	conv_loss1aZ>(h�4        )��P	"�����A�*

	conv_lossr�a>
]e�        )��P	6_�����A�*

	conv_loss|*Z>$b1�        )��P	f������A�*

	conv_loss�>[>�t�        )��P	���A�*

	conv_lossX�N>�ޚ�        )��P	s+�����A�*

	conv_lossIr>ҩ��        )��P	�o�����A�*

	conv_loss��^>�4-        )��P	�������A�*

	conv_loss;a>��>        )��P	3霒���A�*

	conv_loss֗P>�z��        )��P	_0�����A�*

	conv_lossǀ`>�V��        )��P	Gs�����A�*

	conv_lossA`>���%        )��P	������A�*

	conv_loss�Q>Z=po        )��P	�흒���A�*

	conv_lossBdY>�,6        )��P	�8�����A�*

	conv_loss6Rc>�=VP        )��P	}�����A�*

	conv_loss%TW>,D�        )��P	�������A�*

	conv_lossq�Z>�c'        )��P	�������A�*

	conv_loss��Y>��+        )��P	J=�����A�*

	conv_losslpY>��A        )��P	�z�����A�*

	conv_loss) T>���        )��P	������A�*

	conv_loss�U>OX�        )��P	 ������A�*

	conv_loss�V>u��%        )��P	`5�����A�*

	conv_lossF�]>�yV        )��P	�q�����A�*

	conv_loss��Y>���        )��P	Q������A�*

	conv_lossv�e>KZ��        )��P	����A�*

	conv_loss��\>�cv�        )��P	�-�����A�*

	conv_loss/&[>Ы��        )��P	�k�����A�*

	conv_losse�X>����        )��P	}������A�*

	conv_loss�FP>���3        )��P	�衒���A�*

	conv_loss)rV>Q���        )��P	>$�����A�*

	conv_lossU>��C        )��P	=a�����A�*

	conv_lossgX>�K�        )��P	������A�*

	conv_loss��Y>AԿC        )��P	�ڢ����A�*

	conv_loss��X>%�d        )��P	������A�*

	conv_loss��]>���        )��P	�V�����A�*

	conv_loss��V>j�|t        )��P	`������A�*

	conv_loss�Q`>�q(C        )��P	Mѣ����A�*

	conv_loss�\>�`��        )��P	������A�*

	conv_loss$	Q>���        )��P	0L�����A�*

	conv_loss�6\>�(�g        )��P	������A�*

	conv_lossU]d>k��{        )��P	Ǥ����A�*

	conv_loss��_>��t        )��P	?�����A�*

	conv_loss��b>�x        )��P	KC�����A�*

	conv_lossÓ`>���        )��P	р�����A�*

	conv_loss�H`>�!�[        )��P	������A�*

	conv_lossw�\>�{Nj        )��P	�������A�*

	conv_lossmsZ>���'        )��P	88�����A�*

	conv_lossοY>`��        )��P	�u�����A�*

	conv_loss��_>�* �        )��P	�������A�*

	conv_loss��_>��}�        )��P	�𦒉��A�*

	conv_lossLX>��o�        )��P	
-�����A�*

	conv_loss,CU>�YA�        )��P	�j�����A�*

	conv_loss�.g>أI�        )��P	੧����A�*

	conv_loss/�^>;�v�        )��P	a������A�*

	conv_lossp�`>��)        )��P	@:�����A�*

	conv_lossG�e>�=T�        )��P	z�����A�*

	conv_loss�oT>F��;        )��P	u������A�*

	conv_loss�	Z>ZŴ        )��P	V������A�*

	conv_loss/�U>��"        )��P	<<�����A�*

	conv_loss��O>hƪ�        )��P	J������A�*

	conv_loss��S>��y�        )��P	(ͩ����A�*

	conv_loss�3[><\H/        )��P	-�����A�*

	conv_loss9�W>h�F        )��P	I�����A�*

	conv_loss�^>�"�        )��P	�������A�*

	conv_lossC$h>7�        )��P	�ժ����A�*

	conv_loss(�W>c/M~        )��P	������A�*

	conv_lossq�W><L��        )��P	4\�����A�*

	conv_loss]�W>
���        )��P	�������A�*

	conv_loss�7\>��*�        )��P	~ګ����A�*

	conv_lossL�M>_+        )��P	C�����A�*

	conv_loss�J>���n        )��P	�W�����A�*

	conv_loss��_>k_�u        )��P	�������A�*

	conv_loss�N>?��        )��P	�Ԭ����A�*

	conv_loss�&S>|đa        )��P	�����A�*

	conv_loss�&W>���c        )��P	�R�����A�*

	conv_loss%�\>���U        )��P	ʗ�����A�*

	conv_loss��H>P�-        )��P	�ح����A�*

	conv_loss"@[>���        )��P	������A�*

	conv_loss��^>T��l        )��P	MY�����A�*

	conv_loss=b[>���        )��P	�������A�*

	conv_loss��S>���3        )��P	�ծ����A�*

	conv_lossp\>P�u        )��P	8�����A�*

	conv_lossjQ>y�b        )��P	�T�����A�*

	conv_loss��d>���        )��P	�������A�*

	conv_loss��V>S��l        )��P	lկ����A�*

	conv_loss�db>�G�        )��P	������A�*

	conv_loss��^>i��        )��P	�Q�����A�*

	conv_loss�
[>ӽq�        )��P	�������A�*

	conv_loss+'Y>��4d        )��P	%ϰ����A�*

	conv_loss�M>�`�        )��P	�����A�*

	conv_lossC(T>7.�        )��P	�M�����A�*

	conv_loss��`>���        )��P	�������A�*

	conv_lossi^>/L��        )��P	�ʱ����A�*

	conv_lossJ�`>���T        )��P	P�����A�*

	conv_loss�W>����        )��P	G�����A�*

	conv_lossXU>b͊        )��P	&������A�*

	conv_loss�da>���L        )��P	OĲ����A�*

	conv_loss`�T>֕^        )��P	������A�*

	conv_loss^>�C�K        )��P	�A�����A�*

	conv_loss��X>���Y        )��P	W������A�*

	conv_loss�`>v��        )��P	{������A�*

	conv_lossrJ>��O�        )��P	�������A�*

	conv_loss�JT>�#�        )��P	>�����A�*

	conv_loss~�M>�i�8        )��P	O�����A�*

	conv_loss�
Z>զv&        )��P	�������A�*

	conv_loss(�V>a��        )��P	ɶ����A�*

	conv_loss�;S>�!�        )��P	������A�*

	conv_losss�W>4�7v        )��P	zA�����A�*

	conv_loss^�Q>���        )��P	������A�*

	conv_loss�Z>�lw�        )��P	#ķ����A�*

	conv_loss�tN>��        )��P	N�����A�*

	conv_loss�fU>��1        )��P	 J�����A�*

	conv_loss�I>��Ӈ        )��P	������A�*

	conv_loss&�e>��Z�        )��P	Bո����A�*

	conv_lossEKW>(s��        )��P	������A�*

	conv_loss��M>8��        )��P	O�����A�*

	conv_loss �R>��        )��P	�������A�*

	conv_losss�V>�Z[        )��P	�͹����A�*

	conv_loss�TX>R�ms        )��P	?�����A�*

	conv_loss�O>b�_        )��P	�X�����A�*

	conv_loss�,X>M�.�        )��P	������A�*

	conv_lossh�O>�b�        )��P	�ֺ����A�*

	conv_loss�VM>�        )��P	������A�*

	conv_loss"�L>�Hh        )��P	�P�����A�*

	conv_loss@dN>TUo�        )��P	Ԍ�����A�*

	conv_loss��U>�Hy        )��P	2ɻ����A�*

	conv_lossZEX>�)a        )��P	B�����A�*

	conv_loss	�[>���$        )��P	�D�����A�*

	conv_loss�Q>~�M�        )��P	p������A�*

	conv_lossoU>�Τ        )��P	󽼒���A�*

	conv_loss��W>�x�,        )��P	P������A�*

	conv_lossȜR>K�_�        )��P	U7�����A�*

	conv_loss,X[>��M        )��P	<t�����A�*

	conv_loss��Z>�g�        )��P	B������A�*

	conv_loss�hS>��t�        )��P	N���A�*

	conv_loss��O>RT~9        )��P	6,�����A�*

	conv_loss%�_>1�IH        )��P	[i�����A�*

	conv_lossT�Z>u���        )��P	������A�*

	conv_loss��H>:x�/        )��P	�������A�*

	conv_loss��_>�>�        )��P	�6�����A�*

	conv_loss9W[>QG��        )��P	Hs�����A�*

	conv_loss��Z>�uN        )��P	�������A�*

	conv_loss��U>���        )��P	�������A�*

	conv_loss�M>0�v        )��P	Y+�����A�*

	conv_loss�ZW>�w"�        )��P	2h�����A�*

	conv_loss�VO>x��        )��P	Υ�����A�*

	conv_loss�iT>�x�>        )��P	\������A�*

	conv_loss��T>��b;        )��P	A�����A�*

	conv_loss'�V>[�@6        )��P	�[�����A�*

	conv_loss�W>�f��        )��P	������A�*

	conv_lossT�U>w]\	        )��P	z������A�*

	conv_loss�pT>FΫj        )��P	����A�*

	conv_loss�J>ȋ'�        )��P	�R���A�*

	conv_loss�-V>�0_O        )��P	����A�*

	conv_lossI�W>��H>        )��P	�����A�*

	conv_lossޞZ>�-\�        )��P	�Ò���A�*

	conv_loss��Z>-��        )��P	1bÒ���A�*

	conv_lossA[>�b�9        )��P	�Ò���A�*

	conv_loss��O>��U2        )��P	��Ò���A�*

	conv_lossѦ[>�9�        )��P	"(Ē���A�*

	conv_loss��U>�8�        )��P	�wĒ���A�*

	conv_loss�'N>\0�        )��P	W�Ē���A�*

	conv_lossJS>j`�D        )��P	��Ē���A�*

	conv_loss+T>�8�        )��P	�6Œ���A�*

	conv_losss�O>��y�        )��P	�zŒ���A�*

	conv_loss@�K>#�Mx        )��P	W�Œ���A�*

	conv_losspFV>|o        )��P	��Œ���A�*

	conv_loss�U>y��k        )��P		5ƒ���A�*

	conv_loss�SJ>�V9        )��P	�yƒ���A�*

	conv_loss�mJ>]�՚        )��P	)�ƒ���A�*

	conv_lossj�S>��A        )��P	�ǒ���A�*

	conv_loss�P>�m�        )��P	�>ǒ���A�*

	conv_loss^qD>��        )��P	�zǒ���A�*

	conv_loss�N>˄��        )��P	ַǒ���A�*

	conv_loss7�T>9-�<        )��P	��ǒ���A�*

	conv_loss=�Z>U���        )��P	�2Ȓ���A�*

	conv_lossGHQ>T�+�        )��P	�oȒ���A�*

	conv_loss%Y>�        )��P	}�Ȓ���A�*

	conv_losse�Z>u���        )��P	�Ȓ���A�*

	conv_loss�eR>f�Dp        )��P	F'ɒ���A�*

	conv_lossV�Q>��̺        )��P	�eɒ���A�*

	conv_lossR>c�         )��P	,�ɒ���A�*

	conv_loss�+Z>~3�W        )��P	��ɒ���A�*

	conv_losso�K>���        )��P	�ʒ���A�*

	conv_loss.hE>�#��        )��P	�[ʒ���A�*

	conv_loss2AQ>.q��        )��P	2�ʒ���A�*

	conv_loss��c>��nW        )��P	F�ʒ���A�*

	conv_lossN>��`�        )��P	:˒���A�*

	conv_loss�\S>���        )��P	S˒���A�*

	conv_loss��H>��s        )��P	��˒���A�*

	conv_loss��N>�0��        )��P	��˒���A�*

	conv_loss�&U>��r�        )��P	�	̒���A�*

	conv_loss��]>ʜ��        )��P	F̒���A�*

	conv_lossچS>R�6�        )��P	�̒���A�*

	conv_loss�Q>Bm�        )��P	R�̒���A�*

	conv_lossYX>w�        )��P	��̒���A�*

	conv_lossD|C>��]�        )��P	�:͒���A�*

	conv_loss��X>��        )��P	�w͒���A�*

	conv_loss�7K>�R&        )��P	�͒���A�*

	conv_loss�I>����        )��P	z�͒���A�*

	conv_loss^Q>1�VR        )��P	�0Β���A�*

	conv_loss��L>���]        )��P	�mΒ���A�*

	conv_loss�l]>����        )��P	�Β���A�*

	conv_loss˼]>(O�f        )��P	��Β���A�*

	conv_lossD&S>tK=        )��P	9ϒ���A�*

	conv_loss$iO>W�`a        )��P	Hvϒ���A�*

	conv_loss�
L>
{oH        )��P	�ϒ���A�*

	conv_loss��W>T�e        )��P	��ϒ���A�*

	conv_loss8�Z>N��        )��P	l4В���A�*

	conv_lossY�M>����        )��P	wВ���A�*

	conv_loss�~J>T+�        )��P	b�В���A�*

	conv_loss��?>TX%�        )��P	Rђ���A�*

	conv_loss;�O>N�S        )��P	�Kђ���A�*

	conv_lossFqO>,��         )��P	��ђ���A�*

	conv_losscX>7�        )��P	��ђ���A�*

	conv_loss3�L>�k�        )��P	TҒ���A�*

	conv_loss<A>"�        )��P	�MҒ���A�*

	conv_lossO�T>���        )��P	�Ғ���A�*

	conv_loss�1O>b;��        )��P	��Ғ���A�*

	conv_loss-�O>���        )��P	\
Ӓ���A�*

	conv_loss��P>��o�        )��P	iKӒ���A�*

	conv_loss\�Q>����        )��P	ŇӒ���A�*

	conv_lossi	I>S$�%        )��P	��Ӓ���A�*

	conv_lossZ�J>N)�Z        )��P	�Ԓ���A�*

	conv_loss)I>\|�l        )��P	W@Ԓ���A�*

	conv_lossS>&_�j        )��P	�}Ԓ���A�*

	conv_loss�1H>��''        )��P	�Ԓ���A�*

	conv_loss�`M>�Q        )��P	C�Ԓ���A�*

	conv_loss�<O>LM�        )��P	�=Ւ���A�*

	conv_loss��R>^`��        )��P	8zՒ���A�*

	conv_loss�O>�ek-        )��P	��Ւ���A�*

	conv_loss�L>�$        )��P	��Ւ���A�*

	conv_loss	,T>��P        )��P	�3֒���A�*

	conv_loss��L>E�Lo        )��P	,q֒���A�*

	conv_loss/�O>� �        )��P	`�֒���A�*

	conv_loss=�P>����        )��P	��֒���A�*

	conv_lossvW>ϭRY        )��P	+ג���A�*

	conv_lossO�M>����        )��P	�iג���A�*

	conv_loss,�B>^        )��P	>�ג���A�*

	conv_loss�G>y�O.        )��P	f�ג���A�*

	conv_loss�S>I̍Y        )��P	�!ؒ���A�*

	conv_loss�W>�'�        )��P	g^ؒ���A�*

	conv_loss�Z>�V+        )��P	ڛؒ���A�*

	conv_loss��I>
��        )��P	�ؒ���A�*

	conv_lossUxN>J��        )��P	�ْ���A�*

	conv_loss�.[>')�        )��P	sSْ���A�*

	conv_loss�>L>���        )��P	q�ْ���A�*

	conv_lossQ�N>��A        )��P	��ْ���A�*

	conv_lossx,B>����        )��P	h
ڒ���A�*

	conv_loss:�J>�ۧ
        )��P	�Gڒ���A�*

	conv_losss�U>a��&        )��P	��ڒ���A�*

	conv_loss�GW>w%�n        )��P	�ڒ���A�*

	conv_lossgaM>׃�H        )��P	3ے���A�*

	conv_loss�0O>��7�        )��P	�Uے���A�*

	conv_loss.tb>�        )��P	ړے���A�*

	conv_loss,1N>F�WO        )��P	-�ے���A�*

	conv_loss�I>g[�        )��P	Hܒ���A�*

	conv_lossgM>j{�        )��P	�iܒ���A�*

	conv_loss��M>�o;%        )��P	��ܒ���A�*

	conv_loss�KL>� H        )��P	]�ܒ���A�*

	conv_loss�)O> G�        )��P	�Mݒ���A�*

	conv_lossQJ>	n�        )��P	�ݒ���A�*

	conv_loss��@>#�8s        )��P	2�ݒ���A�*

	conv_lossZ^T>��^�        )��P	dޒ���A�*

	conv_losso�C>g,p(        )��P	xRޒ���A�*

	conv_loss�PC>=mi�        )��P	V�ޒ���A�*

	conv_losssC>e�Lh        )��P	�ޒ���A�*

	conv_lossXXV>���4        )��P	�
ߒ���A�*

	conv_loss��K>} �+        )��P	iGߒ���A�*

	conv_loss�aH>ȓ�        )��P	�ߒ���A�*

	conv_loss�9M>Bm��        )��P	`�ߒ���A�*

	conv_loss��T>ɥR�        )��P	�#�����A�*

	conv_lossXR>+�R        )��P	�`�����A�*

	conv_loss�H>��        )��P	ʝ�����A�*

	conv_loss��M>�ܺ        )��P	�������A�*

	conv_lossp�U>=��        )��P	�ᒉ��A�*

	conv_loss�~D>@L')        )��P	SUᒉ��A�*

	conv_loss�/I>�kz        )��P	I�ᒉ��A�*

	conv_loss��M>��ׄ        )��P	%�ᒉ��A�*

	conv_loss:]G>�Th�        )��P	⒉��A�*

	conv_loss>�M>ܧ��        )��P	nM⒉��A�*

	conv_loss��U>Ґ�        )��P	ۋ⒉��A�*

	conv_loss �M>K��6        )��P	��⒉��A�*

	conv_loss88C>~X	-        )��P	O
㒉��A�*

	conv_loss.>Z>4ojZ        )��P	'G㒉��A�*

	conv_losss�K>����        )��P	�㒉��A�*

	conv_lossjK>ww3#        )��P	��㒉��A�*

	conv_loss�R>��Ӝ        )��P	�䒉��A�*

	conv_loss��H>���        )��P	�B䒉��A�*

	conv_loss�D>EtN�        )��P	ŀ䒉��A�*

	conv_loss�>B>L�=�        )��P	��䒉��A�*

	conv_loss�AI>�zG�        )��P	
咉��A�*

	conv_losseL>��Sw        )��P	�K咉��A�*

	conv_loss�L>��        )��P	��咉��A�*

	conv_lossgoK>I�r        )��P	��咉��A�*

	conv_loss�IC>�6,v        )��P	�撉��A�*

	conv_loss�RF>͙֭        )��P	PQ撉��A�*

	conv_losss]>�v��        )��P	*�撉��A�*

	conv_lossl�;>��w�        )��P	*�撉��A�*

	conv_loss-�F>|3�        )��P	璉��A�*

	conv_loss�J>	�F�        )��P	CK璉��A�*

	conv_loss�S>H��        )��P	s�璉��A�*

	conv_lossj[L>P]$�        )��P	n�璉��A�*

	conv_loss��U>�c��        )��P	��钉��A�*

	conv_lossu�H>��_        )��P	�ꒉ��A�*

	conv_loss�ID>�ڍ�        )��P	?Rꒉ��A�*

	conv_loss�AF>�7�        )��P	�ꒉ��A�*

	conv_lossGzK>��MW        )��P	��ꒉ��A�*

	conv_loss�?K>g�OL        )��P	�뒉��A�*

	conv_loss�AJ>[��]        )��P	�L뒉��A�*

	conv_loss�uR>fhl�        )��P	��뒉��A�*

	conv_loss�H>�Uκ        )��P	��뒉��A�*

	conv_loss��?>ά(        )��P	�쒉��A�*

	conv_lossO�@>5���        )��P	�Z쒉��A�*

	conv_lossH>��uE        )��P	M�쒉��A�*

	conv_loss��F>�]�^        )��P	��쒉��A�*

	conv_lossJFO>"9�        )��P	q$풉��A�*

	conv_loss	nA>r��E        )��P	Bb풉��A�*

	conv_loss[�K>����        )��P	r�풉��A�*

	conv_lossfO@>��0.        )��P	-�풉��A�*

	conv_loss�C> L�k        )��P	B��A�*

	conv_lossU�I>�)��        )��P	�[��A�*

	conv_lossR7P>�54        )��P	"���A�*

	conv_loss�pE>��J        )��P	����A�*

	conv_loss�;J>�|�d        )��P	a��A�*

	conv_loss� S>5���        )��P	)U��A�*

	conv_loss��A>�9�'        )��P	���A�*

	conv_loss�I>�Rb�        )��P	����A�*

	conv_loss�vE>3b��        )��P	K𒉙�A�*

	conv_loss��N>���        )��P	M𒉙�A�*

	conv_loss�KX>[�$3        )��P	{�𒉙�A�*

	conv_lossK�O>�F�        )��P	��𒉙�A�*

	conv_loss��;>��Y        )��P	�񒉙�A�*

	conv_loss�F>z��S        )��P	�C񒉙�A�*

	conv_lossM�I>����        )��P	��񒉙�A�*

	conv_loss&'I>oa�         )��P	 �񒉙�A�*

	conv_loss*�W>}"Y�        )��P	2�񒉙�A�*

	conv_loss0�H>��$�        )��P	9򒉙�A�*

	conv_loss��Q>NF        )��P	�v򒉙�A�*

	conv_loss#I>Aط        )��P	�򒉙�A�*

	conv_loss�G>����        )��P	t�򒉙�A�*

	conv_loss Y>��        )��P	�-󒉙�A�*

	conv_loss�MV>Jվ%        )��P	nk󒉙�A�*

	conv_losscM>���m        )��P	��󒉙�A�*

	conv_loss�JH>K��        )��P	��󒉙�A�*

	conv_loss�P>���x        )��P	7%�����A�*

	conv_loss�T>oeG        )��P	�a�����A�*

	conv_loss�1D>����        )��P	i������A�*

	conv_loss�H=>4l�        )��P	������A�*

	conv_loss��E>N2��        )��P	������A�*

	conv_loss��G>{�X!        )��P	�Y�����A�*

	conv_loss�F>r��        )��P	
������A�*

	conv_loss�H>R�        )��P	������A�*

	conv_lossB>֧�        )��P	�#�����A�*

	conv_loss&CD>�e*�        )��P	�a�����A�*

	conv_loss�CN>͑a        )��P	������A�*

	conv_loss��B>~�,V        )��P	2������A�*

	conv_loss�U>V���        )��P	� �����A�*

	conv_lossV�K>c�L        )��P	h\�����A�*

	conv_loss�`>>P>�W        )��P	�������A�*

	conv_loss(aA>����        )��P	������A�*

	conv_loss3�G>��f        )��P	� �����A�*

	conv_lossoO>?�i�        )��P	�^�����A�*

	conv_lossէ>>�V�o        )��P	�������A�*

	conv_loss2�S>��.�        )��P	�������A�*

	conv_loss�B>sT         )��P	�,�����A�*

	conv_loss�9>Y��        )��P	Xk�����A�*

	conv_lossJC>Z&�        )��P	������A�*

	conv_loss��G>�        )��P	�������A�*

	conv_loss65B>�@^k        )��P	#�����A�*

	conv_loss%9>+�YI        )��P	�`�����A�*

	conv_loss�=H>'v5�        )��P	o������A�*

	conv_loss��N>sqB        )��P	�������A�*

	conv_loss L>F8        )��P	������A�*

	conv_loss��?>}���        )��P	�X�����A�*

	conv_lossW�D>jhgb        )��P	�������A�*

	conv_losseH>o@�1        )��P	�������A�*

	conv_losswwA>�g��        )��P	������A�*

	conv_loss�D>-��/        )��P	�K�����A�*

	conv_loss:NC>̟8�        )��P	������A�*

	conv_lossj+G>�        )��P	�������A�*

	conv_loss �M>��Ɠ        )��P	�����A�*

	conv_loss��I>��W�        )��P	�@�����A�*

	conv_loss�8>$gb�        )��P	�}�����A�*

	conv_loss�m@>�ù�        )��P	�������A�*

	conv_loss5R>v��        )��P	�������A�*

	conv_lossMgB>�-m        )��P	�8�����A�*

	conv_lossQ>��        )��P	�t�����A�*

	conv_loss6TP>	AK        )��P	������A�*

	conv_lossbtD>��W)        )��P	�������A�*

	conv_loss�O>ØH        )��P	�-�����A�*

	conv_loss�K>�dk�        )��P	j�����A�*

	conv_loss�K>�Z9        )��P	J������A�*

	conv_lossm�M>`�        )��P	�������A�*

	conv_loss:dK>��        )��P	'% ����A�*

	conv_lossY5E>;1��        )��P	�c ����A�*

	conv_lossDG>�V��        )��P	�� ����A�*

	conv_loss��;>o�>�        )��P	�� ����A�*

	conv_loss'�?>#;[u        )��P	N����A�*

	conv_lossa�A>0�^        )��P	HY����A�*

	conv_loss��:>���        )��P	������A�*

	conv_loss2!F>���        )��P	P�����A�*

	conv_loss�O>��bK        )��P	d����A�*

	conv_loss��I>�d�X        )��P	Ͷ����A�*

	conv_loss��Q>l�lP        )��P	�����A�*

	conv_loss �?>�Ya�        )��P	R4����A�*

	conv_loss�B>R��        )��P	�u����A�*

	conv_loss(�F>�x        )��P	7�����A�*

	conv_loss��A>=��        )��P	������A�*

	conv_losspL>����        )��P	F����A�*

	conv_loss�>>����        )��P	������A�*

	conv_loss��F>@#��        )��P	������A�*

	conv_loss�>>�r�'        )��P		����A�*

	conv_lossen=>E�i        )��P	_a	����A�*

	conv_loss�eP>�K��        )��P	]�	����A�*

	conv_loss I>&>�+        )��P	��	����A�*

	conv_loss?G>0�[        )��P	D
����A�*

	conv_loss|KP>���        )��P	�X
����A�*

	conv_loss��Q>��J�        )��P	X�
����A�*

	conv_lossf&G>���        )��P	��
����A�*

	conv_loss��@>o�T        )��P	g ����A�*

	conv_loss�F>L�=S        )��P	_����A�*

	conv_loss�SD>h-�        )��P	������A�*

	conv_loss��H>p��        )��P		�����A�*

	conv_loss�?>ߘT%        )��P	D����A�*

	conv_loss�	G>���(        )��P	>V����A�*

	conv_loss�=A>w(OR        )��P	4�����A�*

	conv_loss�P>����        )��P	 �����A�*

	conv_loss�F>D���        )��P	�����A�*

	conv_lossPE>"X         )��P	>L����A�*

	conv_losswrD>�^{f        )��P	�����A�*

	conv_loss��F>��        )��P	E�����A�*

	conv_loss7A>;�{        )��P	�����A�*

	conv_loss��C>��y�        )��P	�@����A�*

	conv_loss�FS>I�Bk        )��P	[}����A�*

	conv_loss��H>p8        )��P	I�����A�*

	conv_loss�>>@"��        )��P	������A�*

	conv_lossU�/>�W�        )��P	�5����A�*

	conv_loss��5>��C�        )��P	�t����A�*

	conv_loss�zM>�        )��P	?�����A�*

	conv_loss=1B>���        )��P	|�����A�*

	conv_loss��J>�\\)        )��P	�-����A�*

	conv_loss�ZA>E�]b        )��P	Qj����A�*

	conv_lossC�9>�,��        )��P	�����A�*

	conv_lossP�C>����        )��P	�����A�*

	conv_lossu<>�A��        )��P	O"����A�*

	conv_loss��:>���*        )��P	*_����A�*

	conv_lossy�3>;�9�        )��P	Ț����A�*

	conv_lossg�9>>�^        )��P	������A�*

	conv_loss��C>J��2        )��P	W����A�*

	conv_loss��C>�+t        )��P	BR����A�*

	conv_loss;�K>��$�        )��P	������A�*

	conv_loss�G>��        )��P	������A�*

	conv_loss��?>��Z�        )��P	�����A�*

	conv_loss��A>H[�        )��P	[����A�*

	conv_loss��>>��Z3        )��P	�����A�*

	conv_loss:O9>mH�        )��P	R�����A�*

	conv_lossW�?>T�;        )��P	?����A�*

	conv_loss �?>��"n        )��P	�V����A�*

	conv_loss��:>�G�~        )��P	������A�*

	conv_lossL�1>��W        )��P	D�����A�*

	conv_loss�YB>�Hs        )��P	"+����A�*

	conv_loss��=>��        )��P	 g����A�*

	conv_loss"�I>(AA�        )��P	ţ����A�*

	conv_loss�+G>x{�        )��P	������A�*

	conv_lossA�?>���>        )��P	�)����A�*

	conv_lossҤD>���        )��P	�f����A�*

	conv_loss��<>Nx�        )��P	֤����A�*

	conv_loss��G>k�d        )��P	�����A�*

	conv_loss�t:>���z        )��P	�.����A�*

	conv_loss�
O>��6�        )��P	�p����A�*

	conv_loss��:>/*�        )��P	I�����A�*

	conv_loss=	?>MS��        )��P	2�����A�*

	conv_loss�@>�F�        )��P	N+����A�*

	conv_lossn�[>3 �        )��P	�j����A�*

	conv_loss8�E>*GT        )��P	I�����A�*

	conv_lossPcF>��q=        )��P	�����A�*

	conv_loss�%D>2{�O        )��P	%����A�*

	conv_loss]�<>��V        )��P	�d����A�*

	conv_loss0�D>�_��        )��P	������A�*

	conv_loss�G>߷��        )��P	������A�*

	conv_loss}2>�e��        )��P	8 ����A�*

	conv_lossКE>ՙ7        )��P	�\����A�*

	conv_loss��;>Q{�        )��P	2�����A�*

	conv_loss�9=>�<
;        )��P	#�����A�*

	conv_loss��I>h�j�        )��P	�����A�*

	conv_lossI�B>���o        )��P	dS����A�*

	conv_loss�tN>`�`        )��P	Q�����A�*

	conv_loss�{D>�e�        )��P	������A�*

	conv_loss��?>�n��        )��P	�����A�*

	conv_lossl7>�`�U        )��P	gI����A�*

	conv_lossN�?>��18        )��P	L�����A�*

	conv_loss8�B>���        )��P	b�����A�*

	conv_loss��?>Ơ?        )��P	a����A�*

	conv_loss}�9>�LS        )��P	�>����A�*

	conv_loss�;G>=�$        )��P	I|����A�*

	conv_loss�<>�J��        )��P	Ը����A�*

	conv_loss�L>�	3        )��P	������A�*

	conv_loss�@7>�xx.        )��P	*4����A�*

	conv_loss^�J>��>�        )��P	�q����A�*

	conv_loss%:>^�T�        )��P	i�����A�*

	conv_loss��P>A���        )��P	������A�*

	conv_loss�&H>�8�;        )��P	�*����A�*

	conv_loss[�N>�{H�        )��P	�;!����A�*

	conv_loss=qC>��1�        )��P	x!����A�*

	conv_loss�E>�b        )��P	��!����A�*

	conv_loss&�4>�NV        )��P	k�!����A�*

	conv_loss��3>�۰�        )��P	gA"����A�*

	conv_loss�7B>P`/]        )��P	�~"����A�*

	conv_loss �7>��k�        )��P	��"����A�*

	conv_lossu8;>��/        )��P	 #����A�*

	conv_loss;>�>        )��P	�K#����A�*

	conv_loss�*B>�V��        )��P	ϒ#����A�*

	conv_loss�(C>O��\        )��P	^�#����A�*

	conv_loss�C>���        )��P	$����A�*

	conv_loss�W:>�x�2        )��P	1R$����A�*

	conv_loss�A>"��x        )��P	��$����A�*

	conv_lossF�>>�n e        )��P	��$����A�*

	conv_lossV�B>?�        )��P	�
%����A�*

	conv_loss��;>/Y7�        )��P	�G%����A�*

	conv_loss\�D>g��q        )��P	��%����A�*

	conv_loss �B>�ޕ        )��P	R�%����A�*

	conv_loss��@>��s        )��P	 &����A�*

	conv_loss��:>D8        )��P	c=&����A�*

	conv_losse8>ފ�G        )��P	]{&����A�*

	conv_losst=>u��        )��P	7�&����A�*

	conv_lossm3>&�ۙ        )��P	��&����A�*

	conv_loss�g7>����        )��P	�5'����A�*

	conv_loss�$@>��$        )��P	pt'����A�*

	conv_lossD�C>14�        )��P	?�'����A�*

	conv_lossh�6>6�m�        )��P	��'����A�*

	conv_loss�?>�}�0        )��P	q>(����A�*

	conv_loss�G>���        )��P	\{(����A�*

	conv_loss'�<>�Ӏ{        )��P	��(����A�*

	conv_loss�_9>	��        )��P	�(����A�*

	conv_lossi(F>�tB        )��P	W5)����A�*

	conv_loss�R>���+        )��P	�r)����A�*

	conv_lossyZ<>W1�u        )��P	�)����A�*

	conv_loss0C>���        )��P	8�)����A�*

	conv_loss{I>����        )��P	d/*����A�*

	conv_loss	{2>����        )��P	u*����A�*

	conv_loss�V:>��TB        )��P	�*����A�*

	conv_lossе@>+�o�        )��P	��*����A�*

	conv_loss�wG>�bز        )��P	2+����A�*

	conv_lossRL>���        )��P	�s+����A�*

	conv_loss9�>>�b'        )��P	��+����A�*

	conv_lossv9>���        )��P	��+����A�*

	conv_loss�:D>�;        )��P	�.,����A�*

	conv_lossQ�8>��}        )��P	�y,����A�*

	conv_loss�<>���        )��P	�,����A�*

	conv_lossI>�M        )��P	��,����A�*

	conv_loss�aA>�=�4        )��P	o2-����A�*

	conv_loss]G>E&�        )��P	�o-����A�*

	conv_loss�bG>���        )��P	��-����A�*

	conv_lossNB>�箛        )��P	$�-����A�*

	conv_loss��>>�Z="        )��P	�_.����A�*

	conv_loss�Q4>�lya        )��P	Q�.����A�*

	conv_loss��=>��2        )��P	��.����A�*

	conv_loss�{;>�U�o        )��P	Z/����A�*

	conv_loss��9>)N��        )��P	�q/����A�*

	conv_lossa$=>"�xn        )��P	��/����A�*

	conv_loss��6>��ז        )��P	5�/����A�*

	conv_loss>�=>�q        )��P	w20����A�*

	conv_loss�9A>���        )��P	]~0����A�*

	conv_loss�9>�W��        )��P	��0����A�*

	conv_loss��=>��        )��P	��0����A�*

	conv_loss� G>�(        )��P	�:1����A�*

	conv_loss�"9>*F�        )��P	�x1����A�*

	conv_lossK�5>��m�        )��P	�1����A�*

	conv_loss�&8>��        )��P	��1����A�*

	conv_loss��;>$e        )��P	�12����A�*

	conv_lossph;>n1L�        )��P	�n2����A�*

	conv_loss �@><���        )��P	<�2����A�*

	conv_loss��3>|�        )��P	��2����A�*

	conv_lossJ>I�        )��P	�&3����A�*

	conv_loss�2:>��7~        )��P	Ye3����A�*

	conv_lossgG5>0g7K        )��P	��3����A�*

	conv_loss�A>�>        )��P	��3����A�*

	conv_loss�}:>�՘        )��P	i34����A�*

	conv_losss<>�[;�        )��P	�p4����A�*

	conv_lossBF>�e�l        )��P	��4����A�*

	conv_lossC�C>���o        )��P	��4����A�*

	conv_loss�`B>\�        )��P	�%5����A�*

	conv_lossPDC>���        )��P	nc5����A�*

	conv_loss��9>څ�        )��P	��5����A�*

	conv_loss�.7>%d�        )��P	��5����A�*

	conv_loss"�8>���        )��P	^6����A�*

	conv_lossD7N>����        )��P	�Z6����A�*

	conv_lossNW=>��ed        )��P	7�6����A�*

	conv_loss��=>���        )��P	�6����A�*

	conv_loss��@>Uj�F        )��P	�7����A�*

	conv_lossV2>��X�        )��P	�R7����A�*

	conv_losssA>�%��        )��P	ȑ7����A�*

	conv_loss�|:>�#�        )��P	m�7����A�*

	conv_losst�/>�G�        )��P	�
8����A�*

	conv_loss�B>��U^        )��P	H8����A�*

	conv_losst�4>��'        )��P	[�8����A�*

	conv_loss>�9>�,e        )��P	��8����A�*

	conv_lossKD>C{�"        )��P	� 9����A�*

	conv_lossN]I>+�US        )��P	0=9����A�*

	conv_loss�);>e�YU        )��P	�z9����A�*

	conv_loss/�7>T���        )��P	�9����A�*

	conv_loss��;>����        )��P	�9����A�*

	conv_loss�j;>�Y,
        )��P	�F:����A�*

	conv_loss�R>>�+�t        )��P	+�:����A�*

	conv_loss��A>g-��        )��P	:�:����A�*

	conv_loss��;>S�(        )��P	�;����A�*

	conv_loss�{<>��2�        )��P	C;����A�*

	conv_loss�@>f��        )��P	�;����A�*

	conv_lossX�H>k)�        )��P	�;����A�*

	conv_loss�7>t���        )��P	9	<����A�*

	conv_loss�H>����        )��P	�H<����A�*

	conv_loss��F>`��        )��P	B�<����A�*

	conv_loss��1>ES�         )��P	��<����A�*

	conv_loss��;>�6��        )��P	 ,=����A�*

	conv_loss7�/>Aa�        )��P	�k=����A�*

	conv_loss[�F>�(�,        )��P	]�=����A�*

	conv_loss��@>0�P        )��P	a�=����A�*

	conv_lossv:>;A1        )��P	�!>����A�*

	conv_loss9@>dc��        )��P	_>����A�*

	conv_loss�@>�`׌        )��P	��>����A�*

	conv_loss�->>�.        )��P	��>����A�*

	conv_loss�H8>w⭊        )��P	(?����A�*

	conv_loss/QC>J`��        )��P	f?����A�*

	conv_loss�N>(�        )��P	��?����A�*

	conv_loss0�8>���#        )��P	n�?����A�*

	conv_losso�<>ri        )��P	� @����A�*

	conv_loss�s=>����        )��P	�]@����A�*

	conv_loss�m@>��x        )��P	&�@����A�*

	conv_loss�:>���        )��P	��@����A�*

	conv_loss6>>��&w        )��P	�A����A�*

	conv_loss��8>�c1        )��P	�UA����A�*

	conv_loss��7>�B�        )��P	ŔA����A�*

	conv_loss�J?>��vw        )��P	��A����A�*

	conv_lossSi:>���        )��P	�B����A�*

	conv_lossԑ,>r�ّ        )��P	�MB����A�*

	conv_loss6�8>��b        )��P	D�B����A�*

	conv_loss��,>Xil+        )��P	��B����A�*

	conv_loss�cB>r]��        )��P	C����A�*

	conv_lossі>>T��        )��P	�AC����A�*

	conv_loss��9>�CA        )��P	�C����A�*

	conv_loss;�8>�Xa        )��P	�C����A�*

	conv_lossz8>�n"        )��P	��C����A�*

	conv_lossY�E>W1Z        )��P	 <D����A�*

	conv_lossJdB>D �        )��P	ZxD����A�*

	conv_loss�<>���        )��P	�D����A�*

	conv_loss	�*>��        )��P	J�D����A�*

	conv_loss �B>�F�        )��P	�.E����A�*

	conv_loss��9>��K�        )��P	|kE����A�*

	conv_loss�8>S`��        )��P	ĩE����A�*

	conv_loss�9>���O        )��P	��E����A�*

	conv_lossH�8>�W�>        )��P	�&F����A�*

	conv_loss�A.>�	$        )��P	�cF����A�*

	conv_loss{*1>-�6j        )��P	޵F����A�*

	conv_loss�0?>u/�V        )��P	W�F����A�*

	conv_loss��4>�N�        )��P	z0G����A�*

	conv_loss��0>DP֙        )��P	�rG����A�*

	conv_losse7>t�l�        )��P	V�G����A�*

	conv_loss#=C>��ܴ        )��P	�G����A�*

	conv_loss�@>�w\�        )��P	p@H����A�*

	conv_loss/0>�G}9        )��P	��H����A�*

	conv_loss�9>�~ޮ        )��P	��H����A�*

	conv_loss��?>��v        )��P	a I����A�*

	conv_loss�I8>�h�*        )��P	 MI����A�*

	conv_loss�>>V9Xt        )��P	��I����A�*

	conv_loss�7>qݾ�        )��P	9�I����A�*

	conv_lossi�5>*�j        )��P	�J����A�*

	conv_loss�]>>#6�5        )��P	JJ����A�*

	conv_loss�N7>��E        )��P	ۇJ����A�*

	conv_loss�J>ƺD�        )��P	��J����A�*

	conv_loss��/>oe�        )��P	�K����A�*

	conv_loss�04>���        )��P	�>K����A�*

	conv_loss�w->!��        )��P	o|K����A�*

	conv_lossS�9>J�ܬ        )��P	1�K����A�*

	conv_lossMS1>���        )��P	!�K����A�*

	conv_losss;3>E=p(        )��P	�7L����A�*

	conv_loss�r;><��^        )��P	�vL����A�*

	conv_loss$@>N���        )��P	�L����A�*

	conv_loss��5>��}        )��P	B�L����A�*

	conv_loss�5=>�Rd�        )��P	�0M����A�*

	conv_lossB>�jH�        )��P	�mM����A�*

	conv_lossu�0>.OM        )��P	��M����A�*

	conv_loss�W=>�Wp        )��P	��M����A�*

	conv_lossѬ3>���d        )��P	�&N����A�*

	conv_lossZ\<>�a	        )��P	~cN����A�*

	conv_loss��=>`iA;        )��P	��N����A�*

	conv_loss��?>� �O        )��P	[�N����A�*

	conv_loss��P>��C�        )��P	1O����A�*

	conv_loss�.6>D��:        )��P	�UO����A�*

	conv_loss�=.>��/j        )��P	��O����A�*

	conv_loss1�;>N6 �        )��P	��O����A�*

	conv_lossC�?>ڙ3<        )��P	�P����A�*

	conv_lossU�3>���i        )��P	�XP����A�*

	conv_loss�<>ȁ?m        )��P	ƖP����A�*

	conv_loss�,5>���        )��P	��P����A�*

	conv_lossi�.>��Ǵ        )��P	kQ����A�*

	conv_loss#�6>����        )��P	�\Q����A�*

	conv_lossJ�5>HӅ\        )��P	I�Q����A�*

	conv_loss��8>����        )��P	��Q����A�*

	conv_loss�3>�6��        )��P	:R����A�*

	conv_loss��8>h�2|        )��P	�PR����A�*

	conv_loss�(6>}�        )��P	��R����A�*

	conv_lossR->�f1        )��P	��R����A�*

	conv_lossu�9>�F �        )��P	��T����A�*

	conv_loss�<>xh#�        )��P	VU����A�*

	conv_loss�A>�G�	        )��P	E\U����A�*

	conv_lossE^D>>o��        )��P	�U����A�*

	conv_loss��@>/T}�        )��P	c�U����A�*

	conv_losscD9>�4�        )��P	�)V����A�*

	conv_loss� @>7T�        )��P	�~V����A�*

	conv_loss�<>�г�        )��P	5�V����A�*

	conv_lossLr?>�)z        )��P	��V����A�*

	conv_lossJ�@>��%�        )��P	�9W����A�*

	conv_loss��5>�	�        )��P	��W����A�*

	conv_loss6�@>t        )��P	v�W����A�*

	conv_loss��=>�\��        )��P	�X����A�*

	conv_loss� 3>Y�[
        )��P	�?X����A�*

	conv_loss/[<>)�        )��P	Y�X����A�*

	conv_losse 7>�e�	        )��P	W�X����A�*

	conv_loss��->��5�        )��P	�Y����A�*

	conv_loss�A9>:��#        )��P	LY����A�*

	conv_loss��7>���Z        )��P	��Y����A�*

	conv_loss��8>��7        )��P	��Y����A�*

	conv_loss�0;>�p�        )��P	Z����A�*

	conv_lossyo?>SG�x        )��P	�DZ����A�*

	conv_loss�fD>=ܧ`        )��P	�Z����A�*

	conv_losso6D>�R��        )��P	��Z����A�*

	conv_loss A>�r^        )��P	��Z����A�*

	conv_loss1�A>� t        )��P	�;[����A�*

	conv_lossN.0>��jZ        )��P	"x[����A�*

	conv_loss�!3>��        )��P	��[����A�*

	conv_losskC>����        )��P	��[����A�*

	conv_lossȼ3>wJ*�        )��P	�2\����A�*

	conv_loss�!=>2�B�        )��P	�o\����A�*

	conv_loss˰+>�2AQ        )��P	��\����A�*

	conv_loss�)2>�N�        )��P	n�\����A�*

	conv_loss��/>�v��        )��P	H)]����A�*

	conv_lossq�3>V��        )��P	�h]����A�*

	conv_lossl6>v
�        )��P	��]����A�*

	conv_loss0>�6�        )��P	��]����A�*

	conv_loss�31>�֜i        )��P	t^����A�*

	conv_loss$3>�i�         )��P	 \^����A�*

	conv_loss��(>Vv6�        )��P	И^����A�*

	conv_loss��7>[�(�        )��P	��^����A�*

	conv_lossK�8>׵�        )��P	�_����A�*

	conv_losso91>�=g�        )��P	bP_����A�*

	conv_loss��8>i�O�        )��P	�_����A�*

	conv_loss2p,>����        )��P	��_����A�*

	conv_lossz�5>}��0        )��P	K
`����A�*

	conv_loss�L>Hh�^        )��P	�F`����A�*

	conv_loss�Z*>�2h        )��P	΃`����A�*

	conv_loss�5>*m-�        )��P	�`����A�*

	conv_loss8>�WJ�        )��P	�a����A�*

	conv_loss?8>��t�        )��P	�]a����A�*

	conv_loss�,8>�        )��P	ۙa����A�*

	conv_lossj�+>6N�U        )��P	��a����A�*

	conv_lossj/>ْT        )��P	�b����A�*

	conv_loss�6>�h�        )��P	�Yb����A�*

	conv_loss�Y6>X2�        )��P	^�b����A�*

	conv_loss��6>��x�        )��P	�b����A�*

	conv_loss��*>r��        )��P	_7c����A�*

	conv_loss��6>        )��P	-wc����A�*

	conv_loss��9>�:�        )��P	ٴc����A�*

	conv_loss>�3>eIj        )��P	��c����A�*

	conv_loss(�8>/���        )��P	+=d����A�*

	conv_loss�36>���        )��P	={d����A�*

	conv_loss��:>̚y�        )��P	M�d����A�*

	conv_loss��8>�u4
        )��P	�e����A�*

	conv_loss�1F>,T��        )��P	)We����A�*

	conv_loss�*;>��        )��P	��e����A�*

	conv_loss�c/>	���        )��P	��e����A�*

	conv_lossʙ/>���t        )��P	�f����A�*

	conv_loss$g.>Ö%�        )��P	�Pf����A�*

	conv_lossj;>�\�        )��P	�f����A�*

	conv_loss�m->���_        )��P	��f����A�*

	conv_lossemI>�d�        )��P	g����A�*

	conv_loss��6>�6�        )��P	�Dg����A�*

	conv_loss68>���U        )��P	"�g����A�*

	conv_lossY	C>S��0        )��P	��g����A�*

	conv_loss��7>D&�b        )��P	��g����A�*

	conv_loss&�8>����        )��P	�9h����A�*

	conv_loss9�>>N�t�        )��P	"wh����A�*

	conv_loss}@>���<        )��P	��h����A�*

	conv_lossK</>zڣ        )��P	v�h����A�*

	conv_loss��0>R!_�        )��P	-i����A�*

	conv_loss�3>	�        )��P	[ii����A�*

	conv_loss��7>�l�Z        )��P	c�i����A�*

	conv_loss��5>����        )��P	�i����A�*

	conv_loss��3>Y}��        )��P	�j����A�*

	conv_loss�`7>��f        )��P	�`j����A�*

	conv_losseb5>llE�        )��P	��j����A�*

	conv_loss�
6>ʱ��        )��P	��j����A�*

	conv_lossj&3>���        )��P	(k����A�*

	conv_loss�<;>��v�        )��P	�Sk����A�*

	conv_loss�H4>�6�2        )��P	�k����A�*

	conv_loss��9><���        )��P	e�k����A�*

	conv_losssl3>�*        )��P	�l����A�*

	conv_loss1=:>� �        )��P	�Jl����A�*

	conv_losskf?>[;�%        )��P	��l����A�*

	conv_loss~{5>��J+        )��P	��l����A�*

	conv_loss]�0>9�W�        )��P	�l����A�*

	conv_loss��6>u95$        )��P	�<m����A�*

	conv_loss�O5> �o�        )��P	�{m����A�*

	conv_loss
E(>=`ӯ        )��P	o�m����A�*

	conv_loss��2>�,4	        )��P	�n����A�*

	conv_loss�p2>���        )��P	�En����A�*

	conv_loss08>�;��        )��P	��n����A�*

	conv_loss�p3>#�C�        )��P	k�n����A�*

	conv_loss�0>s�p�        )��P	o����A�*

	conv_lossMW>>պ��        )��P	�\o����A�*

	conv_loss_�D>tg %        )��P	P�o����A�*

	conv_loss��5>+i�        )��P	��o����A�*

	conv_loss�D/>�d]�        )��P	ip����A�*

	conv_loss�"*>쇯�        )��P	9dp����A�*

	conv_loss<=>�`�?        )��P	'�p����A�*

	conv_lossn�*>���S        )��P	��p����A�*

	conv_loss��8>,=��        )��P	(q����A�*

	conv_lossm'>�Y��        )��P	:]q����A�*

	conv_lossʩ6>3�Q        )��P	a�q����A�*

	conv_loss&�2>�b��        )��P	
�q����A�*

	conv_loss�9>���        )��P	d,r����A�*

	conv_loss�P3>B�k        )��P	lr����A�*

	conv_loss��?>��
�        )��P	a�r����A�*

	conv_lossd.>Y+��        )��P	(�r����A�*

	conv_loss;�2>xU#w        )��P	�%s����A�*

	conv_loss��'>��7�        )��P	�cs����A�*

	conv_loss"^->��R        )��P	v�s����A�*

	conv_lossw�D>yK        )��P	e�s����A�*

	conv_losspu.>m�
h        )��P	�t����A�*

	conv_loss�a1>$�        )��P	(Zt����A�*

	conv_loss;�,>��p        )��P	�t����A�*

	conv_loss��->�3        )��P	��t����A�*

	conv_loss�TA>�xv�        )��P	�u����A�*

	conv_loss�4>�2M        )��P	Nu����A�*

	conv_loss��3>h��        )��P	��u����A�*

	conv_lossqA:>f#        )��P	��u����A�*

	conv_loss5>%���        )��P	 v����A�*

	conv_loss`�2>�f�        )��P	�Dv����A�*

	conv_loss��2>!,�        )��P	.�v����A�*

	conv_lossh�1><��r        )��P	Q�v����A�*

	conv_loss�e1>�j��        )��P	�w����A�*

	conv_lossM�3>G'3        )��P	�Cw����A�*

	conv_loss��=>6�6�        )��P	��w����A�*

	conv_loss��0>�
�[        )��P	r�w����A�*

	conv_loss
�,>��W        )��P	rx����A�*

	conv_loss��2>�K&1        )��P	�Ax����A�*

	conv_loss��*>��x!        )��P	�x����A�*

	conv_loss�~1>�֭        )��P	��x����A�*

	conv_loss�8*>)�Z        )��P	f�x����A�*

	conv_loss� 8>V8        )��P	>y����A�*

	conv_loss�1>��(        )��P	-y����A�*

	conv_loss+�1>��w        )��P	мy����A�*

	conv_lossy9>Y��i        )��P	B�y����A�*

	conv_lossb�2>�|�        )��P	mJz����A�*

	conv_loss�g6>����        )��P	ӈz����A�*

	conv_loss�*>���r        )��P	��z����A�*

	conv_loss�@>�Z�        )��P	�{����A�*

	conv_lossL�>>�m�        )��P	cH{����A�*

	conv_loss}->L��        )��P	r�{����A�*

	conv_loss��9>���        )��P	��{����A�*

	conv_loss�6>6�        )��P	�&|����A�*

	conv_loss��1>�>��        )��P	�f|����A�*

	conv_loss��.>ZMG        )��P	g�|����A�*

	conv_loss�4>N�@        )��P	��|����A�*

	conv_loss\H3>�a�        )��P	F*}����A�*

	conv_loss#�6>>���        )��P	�g}����A�*

	conv_loss�0>M��        )��P	ʥ}����A�*

	conv_loss��2>-�Q�        )��P	��}����A�*

	conv_loss��5>�oZ        )��P	�"~����A�*

	conv_loss��&>���        )��P	m~����A�*

	conv_loss��3>����        )��P	Q�~����A�*

	conv_loss��&>{TL�        )��P	��~����A�*

	conv_lossn�6>t�1        )��P	l+����A�*

	conv_loss��*>��3        )��P	jj����A�*

	conv_lossu73>���        )��P	�����A�*

	conv_lossKE>x_Z�        )��P	������A�*

	conv_loss6~->��\,        )��P	�"�����A�*

	conv_lossec->N)r        )��P	&^�����A�*

	conv_loss9m,>0��s        )��P	U������A�*

	conv_loss�C>J؋        )��P	�ۀ����A�*

	conv_lossW�->�B        )��P	������A�*

	conv_loss�%>��gC        )��P	^U�����A�*

	conv_lossX->���#        )��P	�������A�*

	conv_lossܚ1>#��Q        )��P	ҁ����A�*

	conv_loss��=>��u�        )��P	������A�*

	conv_loss9-4>"�s[        )��P	L�����A�*

	conv_loss��.>�ϰo        )��P	�������A�*

	conv_lossG*2>�J�        )��P	�Ƃ����A�*

	conv_loss�t9>{�0_        )��P	A�����A�*

	conv_loss��6>A �f        )��P	�>�����A�*

	conv_lossW�9>�1v�        )��P	{�����A�*

	conv_loss�->��ù        )��P	�������A�*

	conv_loss�;8>7        )��P	������A�*

	conv_loss��<>��T�        )��P	�5�����A�*

	conv_lossڶ0>���        )��P	\s�����A�*

	conv_lossUh8>n:�$        )��P	������A�*

	conv_loss-,>&v7�        )��P	�������A�*

	conv_loss�~B>1'�         )��P	.�����A�*

	conv_loss�;>%_kU        )��P	<j�����A�*

	conv_loss��0>�/h        )��P	�������A�*

	conv_lossqh9>o�}        )��P	�煓���A�*

	conv_lossU->s��        )��P	�%�����A�*

	conv_lossFC8>=��