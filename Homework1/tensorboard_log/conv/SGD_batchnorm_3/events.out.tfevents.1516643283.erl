       �K"	  ���Abrain.Event:2��l�9      ܽ�	˪���A"��
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
,conv2d/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: * 
_class
loc:@conv2d/kernel*
valueB
 *���>*
dtype0
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
conv2d/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:
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
VariableV2*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
:
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:
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
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
n
batch_normalization/cond/SwitchSwitchPlaceholder_2Placeholder_2*
_output_shapes

::*
T0

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
0batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id* 
_output_shapes
::*
T0*,
_class"
 loc:@batch_normalization/gamma
�
0batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
::
�
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:12batch_normalization/cond/FusedBatchNorm/Switch_2:1batch_normalization/cond/Const batch_normalization/cond/Const_1*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:
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
2batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
::*
T0
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
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
N*
_output_shapes

:: *
T0
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
'batch_normalization/AssignMovingAvg/SubSub(batch_normalization/AssignMovingAvg/read batch_normalization/cond/Merge_1*
_output_shapes
:*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
'batch_normalization/AssignMovingAvg/MulMul'batch_normalization/AssignMovingAvg/Subbatch_normalization/Squeeze*
_output_shapes
:*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
#batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/Mul*
_output_shapes
:*
use_locking( *
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
*batch_normalization/AssignMovingAvg_1/readIdentity#batch_normalization/moving_variance*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
)batch_normalization/AssignMovingAvg_1/SubSub*batch_normalization/AssignMovingAvg_1/read batch_normalization/cond/Merge_2*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:*
T0
�
)batch_normalization/AssignMovingAvg_1/MulMul)batch_normalization/AssignMovingAvg_1/Subbatch_normalization/Squeeze*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:*
T0
�
%batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/Mul*
use_locking( *
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
f
ReluRelubatch_normalization/cond/Merge*/
_output_shapes
:���������*
T0
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel*%
valueB"            *
dtype0
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
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:*
T0
g
conv2d_2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
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
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
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
,batch_normalization_1/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*-
_class#
!loc:@batch_normalization_1/beta*
valueB*    
�
batch_normalization_1/beta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:*
dtype0*
_output_shapes
:
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
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:
�
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
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
VariableV2*
_output_shapes
:*
shared_name *8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:*
dtype0
�
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
:*
use_locking(
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
"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id*"
_class
loc:@conv2d_2/Conv2D*J
_output_shapes8
6:���������:���������*
T0
�
2batch_normalization_2/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::*
T0
�
2batch_normalization_2/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::*
T0
�
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:14batch_normalization_2/cond/FusedBatchNorm/Switch_2:1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:*
T0*
data_formatNHWC
�
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id*
T0*"
_class
loc:@conv2d_2/Conv2D*J
_output_shapes8
6:���������:���������
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
::
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
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_24batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
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
ExpandDims&batch_normalization_2/ExpandDims/input$batch_normalization_2/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
m
(batch_normalization_2/ExpandDims_1/inputConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
#batch_normalization_2/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
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
batch_normalization_2/SqueezeSqueezebatch_normalization_2/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
*batch_normalization_2/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
)batch_normalization_2/AssignMovingAvg/SubSub*batch_normalization_2/AssignMovingAvg/read"batch_normalization_2/cond/Merge_1*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:*
T0
�
)batch_normalization_2/AssignMovingAvg/MulMul)batch_normalization_2/AssignMovingAvg/Subbatch_normalization_2/Squeeze*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_2/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
,batch_normalization_2/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
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
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_2/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(
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
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(
�
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
�
3batch_normalization_2/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_2/moving_mean
VariableV2*
dtype0*
_output_shapes
:*
shared_name *4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:
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
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
6batch_normalization_2/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB*  �?
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
,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes
:*
use_locking(
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
"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
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
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_24batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
�
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
N*
_output_shapes

:: *
T0
�
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
N*
_output_shapes

:: *
T0
k
&batch_normalization_3/ExpandDims/inputConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0
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
(batch_normalization_3/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
h
&batch_normalization_3/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_3/ExpandDims_1
ExpandDims(batch_normalization_3/ExpandDims_1/input&batch_normalization_3/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
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
batch_normalization_3/SelectSelectbatch_normalization_3/Reshape batch_normalization_3/ExpandDims"batch_normalization_3/ExpandDims_1*
_output_shapes
:*
T0
~
batch_normalization_3/SqueezeSqueezebatch_normalization_3/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
�
*batch_normalization_3/AssignMovingAvg/readIdentity!batch_normalization_2/moving_mean*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
)batch_normalization_3/AssignMovingAvg/SubSub*batch_normalization_3/AssignMovingAvg/read"batch_normalization_3/cond/Merge_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
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
,batch_normalization_3/AssignMovingAvg_1/readIdentity%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0
�
+batch_normalization_3/AssignMovingAvg_1/SubSub,batch_normalization_3/AssignMovingAvg_1/read"batch_normalization_3/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
+batch_normalization_3/AssignMovingAvg_1/MulMul+batch_normalization_3/AssignMovingAvg_1/Subbatch_normalization_3/Squeeze*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
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
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"0	  d   *
dtype0*
_output_shapes
:
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
:	�d
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	�d*
T0*
_class
loc:@dense/kernel
�
dense/kernel
VariableV2*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d*
shared_name *
_class
loc:@dense/kernel
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
dense/MatMulMatMulReshapedense/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
�
,batch_normalization_3/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_3/gamma*
valueBd*  �?*
dtype0*
_output_shapes
:d
�
batch_normalization_3/gamma
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:d
�
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes
:d
�
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
_output_shapes
:d*
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
,batch_normalization_3/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_3/beta*
valueBd*    *
dtype0*
_output_shapes
:d
�
batch_normalization_3/beta
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:d
�
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
_output_shapes
:d*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(
�
batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:d
�
3batch_normalization_3/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueBd*    *
dtype0*
_output_shapes
:d
�
!batch_normalization_3/moving_mean
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:d
�
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
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
:d
�
6batch_normalization_3/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueBd*  �?*
dtype0*
_output_shapes
:d
�
%batch_normalization_3/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
�
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:d
~
4batch_normalization_4/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_4/moments/meanMeandense/MatMul4batch_normalization_4/moments/mean/reduction_indices*
T0*
_output_shapes

:d*

Tidx0*
	keep_dims(
�
*batch_normalization_4/moments/StopGradientStopGradient"batch_normalization_4/moments/mean*
T0*
_output_shapes

:d
�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense/MatMul*batch_normalization_4/moments/StopGradient*
T0*'
_output_shapes
:���������d
�
8batch_normalization_4/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_4/moments/varianceMean/batch_normalization_4/moments/SquaredDifference8batch_normalization_4/moments/variance/reduction_indices*
T0*
_output_shapes

:d*

Tidx0*
	keep_dims(
�
%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
_output_shapes
:d*
squeeze_dims
 *
T0
�
'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:d
f
$batch_normalization_4/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
 batch_normalization_4/ExpandDims
ExpandDims%batch_normalization_4/moments/Squeeze$batch_normalization_4/ExpandDims/dim*
T0*
_output_shapes

:d*

Tdim0
h
&batch_normalization_4/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_1
ExpandDims&batch_normalization_3/moving_mean/read&batch_normalization_4/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:d
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
T0*
_output_shapes

:d
�
batch_normalization_4/SqueezeSqueezebatch_normalization_4/Select*
_output_shapes
:d*
squeeze_dims
 *
T0
h
&batch_normalization_4/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
"batch_normalization_4/ExpandDims_2
ExpandDims'batch_normalization_4/moments/Squeeze_1&batch_normalization_4/ExpandDims_2/dim*
T0*
_output_shapes

:d*

Tdim0
h
&batch_normalization_4/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_3
ExpandDims*batch_normalization_3/moving_variance/read&batch_normalization_4/ExpandDims_3/dim*
T0*
_output_shapes

:d*

Tdim0
o
%batch_normalization_4/Reshape_1/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_4/Reshape_1ReshapePlaceholder_2%batch_normalization_4/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
�
batch_normalization_4/Select_1Selectbatch_normalization_4/Reshape_1"batch_normalization_4/ExpandDims_2"batch_normalization_4/ExpandDims_3*
T0*
_output_shapes

:d
�
batch_normalization_4/Squeeze_1Squeezebatch_normalization_4/Select_1*
_output_shapes
:d*
squeeze_dims
 *
T0
m
(batch_normalization_4/ExpandDims_4/inputConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
h
&batch_normalization_4/ExpandDims_4/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_4
ExpandDims(batch_normalization_4/ExpandDims_4/input&batch_normalization_4/ExpandDims_4/dim*
_output_shapes
:*

Tdim0*
T0
m
(batch_normalization_4/ExpandDims_5/inputConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
h
&batch_normalization_4/ExpandDims_5/dimConst*
_output_shapes
: *
value	B : *
dtype0
�
"batch_normalization_4/ExpandDims_5
ExpandDims(batch_normalization_4/ExpandDims_5/input&batch_normalization_4/ExpandDims_5/dim*
_output_shapes
:*

Tdim0*
T0
o
%batch_normalization_4/Reshape_2/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
batch_normalization_4/Reshape_2ReshapePlaceholder_2%batch_normalization_4/Reshape_2/shape*
_output_shapes
:*
T0
*
Tshape0
�
batch_normalization_4/Select_2Selectbatch_normalization_4/Reshape_2"batch_normalization_4/ExpandDims_4"batch_normalization_4/ExpandDims_5*
T0*
_output_shapes
:
�
batch_normalization_4/Squeeze_2Squeezebatch_normalization_4/Select_2*
T0*
_output_shapes
: *
squeeze_dims
 
�
+batch_normalization_4/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/xbatch_normalization_4/Squeeze_2*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
+batch_normalization_4/AssignMovingAvg/sub_1Sub&batch_normalization_3/moving_mean/readbatch_normalization_4/Squeeze*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:d
�
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:d*
T0
�
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_4/AssignMovingAvg/mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:d
�
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
�
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/xbatch_normalization_4/Squeeze_2*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: *
T0
�
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub*batch_normalization_3/moving_variance/readbatch_normalization_4/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:d
�
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:d
�
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:d
j
%batch_normalization_4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_4/batchnorm/addAddbatch_normalization_4/Squeeze_1%batch_normalization_4/batchnorm/add/y*
_output_shapes
:d*
T0
x
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
T0*
_output_shapes
:d
�
#batch_normalization_4/batchnorm/mulMul%batch_normalization_4/batchnorm/Rsqrt batch_normalization_3/gamma/read*
T0*
_output_shapes
:d
�
%batch_normalization_4/batchnorm/mul_1Muldense/MatMul#batch_normalization_4/batchnorm/mul*
T0*'
_output_shapes
:���������d
�
%batch_normalization_4/batchnorm/mul_2Mulbatch_normalization_4/Squeeze#batch_normalization_4/batchnorm/mul*
_output_shapes
:d*
T0
�
#batch_normalization_4/batchnorm/subSubbatch_normalization_3/beta/read%batch_normalization_4/batchnorm/mul_2*
T0*
_output_shapes
:d
�
%batch_normalization_4/batchnorm/add_1Add%batch_normalization_4/batchnorm/mul_1#batch_normalization_4/batchnorm/sub*'
_output_shapes
:���������d*
T0
g
Relu_3Relu%batch_normalization_4/batchnorm/add_1*
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
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:
*
T0*
_class
loc:@dense_1/bias
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
�
gradients/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
gradients/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
!gradients/Mean_grad/Reshape/shapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
gradients/Mean_grad/ShapeShapelogistic_loss$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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

�
gradients/Mean_grad/Shape_1Shapelogistic_loss$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/Shape_2Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
valueB *
dtype0
�
gradients/Mean_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@gradients/Mean_grad/Shape_1
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1*
_output_shapes
: 
�
gradients/Mean_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
gradients/Mean_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
�
-gradients/logistic_loss_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
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

�
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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
:*

Tidx0*
	keep_dims( 
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

�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
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

�
(gradients/logistic_loss/Log1p_grad/add/xConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_18^gradients/logistic_loss_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *  �?*
dtype0
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

�
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikedense_2/BiasAdd$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
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

�
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:���������
*
T0
�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
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

�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*'
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

�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
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
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
_output_shapes
:
*
T0
�
/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1^gradients/AddN+^gradients/dense_2/BiasAdd_grad/BiasAddGrad
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
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_37gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:d
*
transpose_a(*
transpose_b( 
�
.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1
�
6gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_2/MatMul_grad/MatMul/^gradients/dense_2/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_2/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:d
*
T0
�
gradients/Relu_3_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_3*'
_output_shapes
:���������d*
T0
�
:gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeShape%batch_normalization_4/batchnorm/mul_1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0*
out_type0
�
<gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_4/batchnorm/add_1_grad/Shape<gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/batch_normalization_4/batchnorm/add_1_grad/SumSumgradients/Relu_3_grad/ReluGradJgradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
<gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeReshape8gradients/batch_normalization_4/batchnorm/add_1_grad/Sum:gradients/batch_normalization_4/batchnorm/add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1Sumgradients/Relu_3_grad/ReluGradLgradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
>gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1Reshape:gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1<gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Egradients/batch_normalization_4/batchnorm/add_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1=^gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape?^gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1
�
Mgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeF^gradients/batch_normalization_4/batchnorm/add_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������d
�
Ogradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1F^gradients/batch_normalization_4/batchnorm/add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
_output_shapes
:d
�
:gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
<gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape<gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/batch_normalization_4/batchnorm/mul_1_grad/mulMulMgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency#batch_normalization_4/batchnorm/mul*
T0*'
_output_shapes
:���������d
�
8gradients/batch_normalization_4/batchnorm/mul_1_grad/SumSum8gradients/batch_normalization_4/batchnorm/mul_1_grad/mulJgradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
<gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeReshape8gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum:gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_4/batchnorm/mul_1_grad/mul_1Muldense/MatMulMgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1Sum:gradients/batch_normalization_4/batchnorm/mul_1_grad/mul_1Lgradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
>gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1Reshape:gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1<gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Egradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1=^gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape?^gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1
�
Mgradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeF^gradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape
�
Ogradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1F^gradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes
:d*
T0*Q
_classG
ECloc:@gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1
�
8gradients/batch_normalization_4/batchnorm/sub_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_4/batchnorm/sub_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Hgradients/batch_normalization_4/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_4/batchnorm/sub_grad/Shape:gradients/batch_normalization_4/batchnorm/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
6gradients/batch_normalization_4/batchnorm/sub_grad/SumSumOgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency_1Hgradients/batch_normalization_4/batchnorm/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
:gradients/batch_normalization_4/batchnorm/sub_grad/ReshapeReshape6gradients/batch_normalization_4/batchnorm/sub_grad/Sum8gradients/batch_normalization_4/batchnorm/sub_grad/Shape*
Tshape0*
_output_shapes
:d*
T0
�
8gradients/batch_normalization_4/batchnorm/sub_grad/Sum_1SumOgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency_1Jgradients/batch_normalization_4/batchnorm/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
6gradients/batch_normalization_4/batchnorm/sub_grad/NegNeg8gradients/batch_normalization_4/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
�
<gradients/batch_normalization_4/batchnorm/sub_grad/Reshape_1Reshape6gradients/batch_normalization_4/batchnorm/sub_grad/Neg:gradients/batch_normalization_4/batchnorm/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Cgradients/batch_normalization_4/batchnorm/sub_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1;^gradients/batch_normalization_4/batchnorm/sub_grad/Reshape=^gradients/batch_normalization_4/batchnorm/sub_grad/Reshape_1
�
Kgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_4/batchnorm/sub_grad/ReshapeD^gradients/batch_normalization_4/batchnorm/sub_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/batch_normalization_4/batchnorm/sub_grad/Reshape*
_output_shapes
:d
�
Mgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_4/batchnorm/sub_grad/Reshape_1D^gradients/batch_normalization_4/batchnorm/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/sub_grad/Reshape_1*
_output_shapes
:d
�
:gradients/batch_normalization_4/batchnorm/mul_2_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
<gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_4/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape<gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/batch_normalization_4/batchnorm/mul_2_grad/mulMulMgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependency_1#batch_normalization_4/batchnorm/mul*
T0*
_output_shapes
:d
�
8gradients/batch_normalization_4/batchnorm/mul_2_grad/SumSum8gradients/batch_normalization_4/batchnorm/mul_2_grad/mulJgradients/batch_normalization_4/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
<gradients/batch_normalization_4/batchnorm/mul_2_grad/ReshapeReshape8gradients/batch_normalization_4/batchnorm/mul_2_grad/Sum:gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
:gradients/batch_normalization_4/batchnorm/mul_2_grad/mul_1Mulbatch_normalization_4/SqueezeMgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0
�
:gradients/batch_normalization_4/batchnorm/mul_2_grad/Sum_1Sum:gradients/batch_normalization_4/batchnorm/mul_2_grad/mul_1Lgradients/batch_normalization_4/batchnorm/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
>gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape_1Reshape:gradients/batch_normalization_4/batchnorm/mul_2_grad/Sum_1<gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
�
Egradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1=^gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape?^gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape_1
�
Mgradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_4/batchnorm/mul_2_grad/ReshapeF^gradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes
:d*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape
�
Ogradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape_1F^gradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape_1*
_output_shapes
:d
�
2gradients/batch_normalization_4/Squeeze_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB"   d   
�
4gradients/batch_normalization_4/Squeeze_grad/ReshapeReshapeMgradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/control_dependency2gradients/batch_normalization_4/Squeeze_grad/Shape*
_output_shapes

:d*
T0*
Tshape0
�
gradients/AddN_1AddNOgradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/control_dependency_1Ogradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes
:d*
T0*Q
_classG
ECloc:@gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1
�
8gradients/batch_normalization_4/batchnorm/mul_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_4/batchnorm/mul_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB:d
�
Hgradients/batch_normalization_4/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_4/batchnorm/mul_grad/Shape:gradients/batch_normalization_4/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients/batch_normalization_4/batchnorm/mul_grad/mulMulgradients/AddN_1 batch_normalization_3/gamma/read*
_output_shapes
:d*
T0
�
6gradients/batch_normalization_4/batchnorm/mul_grad/SumSum6gradients/batch_normalization_4/batchnorm/mul_grad/mulHgradients/batch_normalization_4/batchnorm/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
:gradients/batch_normalization_4/batchnorm/mul_grad/ReshapeReshape6gradients/batch_normalization_4/batchnorm/mul_grad/Sum8gradients/batch_normalization_4/batchnorm/mul_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
8gradients/batch_normalization_4/batchnorm/mul_grad/mul_1Mul%batch_normalization_4/batchnorm/Rsqrtgradients/AddN_1*
_output_shapes
:d*
T0
�
8gradients/batch_normalization_4/batchnorm/mul_grad/Sum_1Sum8gradients/batch_normalization_4/batchnorm/mul_grad/mul_1Jgradients/batch_normalization_4/batchnorm/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
<gradients/batch_normalization_4/batchnorm/mul_grad/Reshape_1Reshape8gradients/batch_normalization_4/batchnorm/mul_grad/Sum_1:gradients/batch_normalization_4/batchnorm/mul_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
�
Cgradients/batch_normalization_4/batchnorm/mul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1;^gradients/batch_normalization_4/batchnorm/mul_grad/Reshape=^gradients/batch_normalization_4/batchnorm/mul_grad/Reshape_1
�
Kgradients/batch_normalization_4/batchnorm/mul_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_4/batchnorm/mul_grad/ReshapeD^gradients/batch_normalization_4/batchnorm/mul_grad/tuple/group_deps*
_output_shapes
:d*
T0*M
_classC
A?loc:@gradients/batch_normalization_4/batchnorm/mul_grad/Reshape
�
Mgradients/batch_normalization_4/batchnorm/mul_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_4/batchnorm/mul_grad/Reshape_1D^gradients/batch_normalization_4/batchnorm/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/mul_grad/Reshape_1*
_output_shapes
:d
�
6gradients/batch_normalization_4/Select_grad/zeros_likeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes

:d*
valueBd*    
�
2gradients/batch_normalization_4/Select_grad/SelectSelectbatch_normalization_4/Reshape4gradients/batch_normalization_4/Squeeze_grad/Reshape6gradients/batch_normalization_4/Select_grad/zeros_like*
T0*
_output_shapes

:d
�
4gradients/batch_normalization_4/Select_grad/Select_1Selectbatch_normalization_4/Reshape6gradients/batch_normalization_4/Select_grad/zeros_like4gradients/batch_normalization_4/Squeeze_grad/Reshape*
_output_shapes

:d*
T0
�
<gradients/batch_normalization_4/Select_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_13^gradients/batch_normalization_4/Select_grad/Select5^gradients/batch_normalization_4/Select_grad/Select_1
�
Dgradients/batch_normalization_4/Select_grad/tuple/control_dependencyIdentity2gradients/batch_normalization_4/Select_grad/Select=^gradients/batch_normalization_4/Select_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/batch_normalization_4/Select_grad/Select*
_output_shapes

:d
�
Fgradients/batch_normalization_4/Select_grad/tuple/control_dependency_1Identity4gradients/batch_normalization_4/Select_grad/Select_1=^gradients/batch_normalization_4/Select_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/batch_normalization_4/Select_grad/Select_1*
_output_shapes

:d
�
>gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_4/batchnorm/RsqrtKgradients/batch_normalization_4/batchnorm/mul_grad/tuple/control_dependency*
T0*
_output_shapes
:d
�
5gradients/batch_normalization_4/ExpandDims_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
7gradients/batch_normalization_4/ExpandDims_grad/ReshapeReshapeDgradients/batch_normalization_4/Select_grad/tuple/control_dependency5gradients/batch_normalization_4/ExpandDims_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
8gradients/batch_normalization_4/batchnorm/add_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
:gradients/batch_normalization_4/batchnorm/add_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
Hgradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_4/batchnorm/add_grad/Shape:gradients/batch_normalization_4/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients/batch_normalization_4/batchnorm/add_grad/SumSum>gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradHgradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
:gradients/batch_normalization_4/batchnorm/add_grad/ReshapeReshape6gradients/batch_normalization_4/batchnorm/add_grad/Sum8gradients/batch_normalization_4/batchnorm/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
8gradients/batch_normalization_4/batchnorm/add_grad/Sum_1Sum>gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradJgradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
<gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1Reshape8gradients/batch_normalization_4/batchnorm/add_grad/Sum_1:gradients/batch_normalization_4/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Cgradients/batch_normalization_4/batchnorm/add_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1;^gradients/batch_normalization_4/batchnorm/add_grad/Reshape=^gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1
�
Kgradients/batch_normalization_4/batchnorm/add_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_4/batchnorm/add_grad/ReshapeD^gradients/batch_normalization_4/batchnorm/add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/batch_normalization_4/batchnorm/add_grad/Reshape*
_output_shapes
:d
�
Mgradients/batch_normalization_4/batchnorm/add_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1D^gradients/batch_normalization_4/batchnorm/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
:gradients/batch_normalization_4/moments/Squeeze_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
<gradients/batch_normalization_4/moments/Squeeze_grad/ReshapeReshape7gradients/batch_normalization_4/ExpandDims_grad/Reshape:gradients/batch_normalization_4/moments/Squeeze_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
4gradients/batch_normalization_4/Squeeze_1_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB"   d   
�
6gradients/batch_normalization_4/Squeeze_1_grad/ReshapeReshapeKgradients/batch_normalization_4/batchnorm/add_grad/tuple/control_dependency4gradients/batch_normalization_4/Squeeze_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
8gradients/batch_normalization_4/Select_1_grad/zeros_likeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes

:d*
valueBd*    *
dtype0
�
4gradients/batch_normalization_4/Select_1_grad/SelectSelectbatch_normalization_4/Reshape_16gradients/batch_normalization_4/Squeeze_1_grad/Reshape8gradients/batch_normalization_4/Select_1_grad/zeros_like*
_output_shapes

:d*
T0
�
6gradients/batch_normalization_4/Select_1_grad/Select_1Selectbatch_normalization_4/Reshape_18gradients/batch_normalization_4/Select_1_grad/zeros_like6gradients/batch_normalization_4/Squeeze_1_grad/Reshape*
_output_shapes

:d*
T0
�
>gradients/batch_normalization_4/Select_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_15^gradients/batch_normalization_4/Select_1_grad/Select7^gradients/batch_normalization_4/Select_1_grad/Select_1
�
Fgradients/batch_normalization_4/Select_1_grad/tuple/control_dependencyIdentity4gradients/batch_normalization_4/Select_1_grad/Select?^gradients/batch_normalization_4/Select_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/batch_normalization_4/Select_1_grad/Select*
_output_shapes

:d
�
Hgradients/batch_normalization_4/Select_1_grad/tuple/control_dependency_1Identity6gradients/batch_normalization_4/Select_1_grad/Select_1?^gradients/batch_normalization_4/Select_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/batch_normalization_4/Select_1_grad/Select_1*
_output_shapes

:d
�
7gradients/batch_normalization_4/ExpandDims_2_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
9gradients/batch_normalization_4/ExpandDims_2_grad/ReshapeReshapeFgradients/batch_normalization_4/Select_1_grad/tuple/control_dependency7gradients/batch_normalization_4/ExpandDims_2_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
<gradients/batch_normalization_4/moments/Squeeze_1_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB"   d   
�
>gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeReshape9gradients/batch_normalization_4/ExpandDims_2_grad/Reshape<gradients/batch_normalization_4/moments/Squeeze_1_grad/Shape*
_output_shapes

:d*
T0*
Tshape0
�
;gradients/batch_normalization_4/moments/variance_grad/ShapeShape/batch_normalization_4/moments/SquaredDifference$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0*
out_type0
�
:gradients/batch_normalization_4/moments/variance_grad/SizeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
9gradients/batch_normalization_4/moments/variance_grad/addAdd8batch_normalization_4/moments/variance/reduction_indices:gradients/batch_normalization_4/moments/variance_grad/Size*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_4/moments/variance_grad/modFloorMod9gradients/batch_normalization_4/moments/variance_grad/add:gradients/batch_normalization_4/moments/variance_grad/Size*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:*
T0
�
=gradients/batch_normalization_4/moments/variance_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB:*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0
�
Agradients/batch_normalization_4/moments/variance_grad/range/startConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B : *N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Agradients/batch_normalization_4/moments/variance_grad/range/deltaConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
;gradients/batch_normalization_4/moments/variance_grad/rangeRangeAgradients/batch_normalization_4/moments/variance_grad/range/start:gradients/batch_normalization_4/moments/variance_grad/SizeAgradients/batch_normalization_4/moments/variance_grad/range/delta*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:*

Tidx0
�
@gradients/batch_normalization_4/moments/variance_grad/Fill/valueConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
:gradients/batch_normalization_4/moments/variance_grad/FillFill=gradients/batch_normalization_4/moments/variance_grad/Shape_1@gradients/batch_normalization_4/moments/variance_grad/Fill/value*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:
�
Cgradients/batch_normalization_4/moments/variance_grad/DynamicStitchDynamicStitch;gradients/batch_normalization_4/moments/variance_grad/range9gradients/batch_normalization_4/moments/variance_grad/mod;gradients/batch_normalization_4/moments/variance_grad/Shape:gradients/batch_normalization_4/moments/variance_grad/Fill*#
_output_shapes
:���������*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
N
�
?gradients/batch_normalization_4/moments/variance_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
=gradients/batch_normalization_4/moments/variance_grad/MaximumMaximumCgradients/batch_normalization_4/moments/variance_grad/DynamicStitch?gradients/batch_normalization_4/moments/variance_grad/Maximum/y*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*#
_output_shapes
:���������*
T0
�
>gradients/batch_normalization_4/moments/variance_grad/floordivFloorDiv;gradients/batch_normalization_4/moments/variance_grad/Shape=gradients/batch_normalization_4/moments/variance_grad/Maximum*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:
�
=gradients/batch_normalization_4/moments/variance_grad/ReshapeReshape>gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeCgradients/batch_normalization_4/moments/variance_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
:gradients/batch_normalization_4/moments/variance_grad/TileTile=gradients/batch_normalization_4/moments/variance_grad/Reshape>gradients/batch_normalization_4/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
=gradients/batch_normalization_4/moments/variance_grad/Shape_2Shape/batch_normalization_4/moments/SquaredDifference$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
=gradients/batch_normalization_4/moments/variance_grad/Shape_3Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
;gradients/batch_normalization_4/moments/variance_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB: *P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_4/moments/variance_grad/ProdProd=gradients/batch_normalization_4/moments/variance_grad/Shape_2;gradients/batch_normalization_4/moments/variance_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2
�
=gradients/batch_normalization_4/moments/variance_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB: *P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
dtype0
�
<gradients/batch_normalization_4/moments/variance_grad/Prod_1Prod=gradients/batch_normalization_4/moments/variance_grad/Shape_3=gradients/batch_normalization_4/moments/variance_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2
�
Agradients/batch_normalization_4/moments/variance_grad/Maximum_1/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
value	B :*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
dtype0
�
?gradients/batch_normalization_4/moments/variance_grad/Maximum_1Maximum<gradients/batch_normalization_4/moments/variance_grad/Prod_1Agradients/batch_normalization_4/moments/variance_grad/Maximum_1/y*
T0*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
_output_shapes
: 
�
@gradients/batch_normalization_4/moments/variance_grad/floordiv_1FloorDiv:gradients/batch_normalization_4/moments/variance_grad/Prod?gradients/batch_normalization_4/moments/variance_grad/Maximum_1*
T0*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
_output_shapes
: 
�
:gradients/batch_normalization_4/moments/variance_grad/CastCast@gradients/batch_normalization_4/moments/variance_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
=gradients/batch_normalization_4/moments/variance_grad/truedivRealDiv:gradients/batch_normalization_4/moments/variance_grad/Tile:gradients/batch_normalization_4/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
Fgradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB"   d   *
dtype0
�
Tgradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeFgradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Egradients/batch_normalization_4/moments/SquaredDifference_grad/scalarConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1>^gradients/batch_normalization_4/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
Bgradients/batch_normalization_4/moments/SquaredDifference_grad/mulMulEgradients/batch_normalization_4/moments/SquaredDifference_grad/scalar=gradients/batch_normalization_4/moments/variance_grad/truediv*'
_output_shapes
:���������d*
T0
�
Bgradients/batch_normalization_4/moments/SquaredDifference_grad/subSubdense/MatMul*batch_normalization_4/moments/StopGradient$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1>^gradients/batch_normalization_4/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1MulBgradients/batch_normalization_4/moments/SquaredDifference_grad/mulBgradients/batch_normalization_4/moments/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������d
�
Bgradients/batch_normalization_4/moments/SquaredDifference_grad/SumSumDgradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1Tgradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Fgradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeReshapeBgradients/batch_normalization_4/moments/SquaredDifference_grad/SumDgradients/batch_normalization_4/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1SumDgradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1Vgradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Hgradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1ReshapeDgradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1Fgradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:d
�
Bgradients/batch_normalization_4/moments/SquaredDifference_grad/NegNegHgradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes

:d
�
Ogradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1G^gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeC^gradients/batch_normalization_4/moments/SquaredDifference_grad/Neg
�
Wgradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/control_dependencyIdentityFgradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeP^gradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������d
�
Ygradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityBgradients/batch_normalization_4/moments/SquaredDifference_grad/NegP^gradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/batch_normalization_4/moments/SquaredDifference_grad/Neg*
_output_shapes

:d
�
7gradients/batch_normalization_4/moments/mean_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
6gradients/batch_normalization_4/moments/mean_grad/SizeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
5gradients/batch_normalization_4/moments/mean_grad/addAdd4batch_normalization_4/moments/mean/reduction_indices6gradients/batch_normalization_4/moments/mean_grad/Size*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
_output_shapes
:
�
5gradients/batch_normalization_4/moments/mean_grad/modFloorMod5gradients/batch_normalization_4/moments/mean_grad/add6gradients/batch_normalization_4/moments/mean_grad/Size*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
_output_shapes
:*
T0
�
9gradients/batch_normalization_4/moments/mean_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB:*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape
�
=gradients/batch_normalization_4/moments/mean_grad/range/startConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B : *J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
=gradients/batch_normalization_4/moments/mean_grad/range/deltaConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
7gradients/batch_normalization_4/moments/mean_grad/rangeRange=gradients/batch_normalization_4/moments/mean_grad/range/start6gradients/batch_normalization_4/moments/mean_grad/Size=gradients/batch_normalization_4/moments/mean_grad/range/delta*

Tidx0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
_output_shapes
:
�
<gradients/batch_normalization_4/moments/mean_grad/Fill/valueConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
6gradients/batch_normalization_4/moments/mean_grad/FillFill9gradients/batch_normalization_4/moments/mean_grad/Shape_1<gradients/batch_normalization_4/moments/mean_grad/Fill/value*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
_output_shapes
:
�
?gradients/batch_normalization_4/moments/mean_grad/DynamicStitchDynamicStitch7gradients/batch_normalization_4/moments/mean_grad/range5gradients/batch_normalization_4/moments/mean_grad/mod7gradients/batch_normalization_4/moments/mean_grad/Shape6gradients/batch_normalization_4/moments/mean_grad/Fill*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
N*#
_output_shapes
:���������
�
;gradients/batch_normalization_4/moments/mean_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
9gradients/batch_normalization_4/moments/mean_grad/MaximumMaximum?gradients/batch_normalization_4/moments/mean_grad/DynamicStitch;gradients/batch_normalization_4/moments/mean_grad/Maximum/y*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*#
_output_shapes
:���������
�
:gradients/batch_normalization_4/moments/mean_grad/floordivFloorDiv7gradients/batch_normalization_4/moments/mean_grad/Shape9gradients/batch_normalization_4/moments/mean_grad/Maximum*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_4/moments/mean_grad/ReshapeReshape<gradients/batch_normalization_4/moments/Squeeze_grad/Reshape?gradients/batch_normalization_4/moments/mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
6gradients/batch_normalization_4/moments/mean_grad/TileTile9gradients/batch_normalization_4/moments/mean_grad/Reshape:gradients/batch_normalization_4/moments/mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
9gradients/batch_normalization_4/moments/mean_grad/Shape_2Shapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
9gradients/batch_normalization_4/moments/mean_grad/Shape_3Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
7gradients/batch_normalization_4/moments/mean_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB: *L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
6gradients/batch_normalization_4/moments/mean_grad/ProdProd9gradients/batch_normalization_4/moments/mean_grad/Shape_27gradients/batch_normalization_4/moments/mean_grad/Const*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
9gradients/batch_normalization_4/moments/mean_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
:*
valueB: *L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2
�
8gradients/batch_normalization_4/moments/mean_grad/Prod_1Prod9gradients/batch_normalization_4/moments/mean_grad/Shape_39gradients/batch_normalization_4/moments/mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2
�
=gradients/batch_normalization_4/moments/mean_grad/Maximum_1/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
�
;gradients/batch_normalization_4/moments/mean_grad/Maximum_1Maximum8gradients/batch_normalization_4/moments/mean_grad/Prod_1=gradients/batch_normalization_4/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2
�
<gradients/batch_normalization_4/moments/mean_grad/floordiv_1FloorDiv6gradients/batch_normalization_4/moments/mean_grad/Prod;gradients/batch_normalization_4/moments/mean_grad/Maximum_1*
T0*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
_output_shapes
: 
�
6gradients/batch_normalization_4/moments/mean_grad/CastCast<gradients/batch_normalization_4/moments/mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
9gradients/batch_normalization_4/moments/mean_grad/truedivRealDiv6gradients/batch_normalization_4/moments/mean_grad/Tile6gradients/batch_normalization_4/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������d
�
gradients/AddN_2AddNMgradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/control_dependencyWgradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/control_dependency9gradients/batch_normalization_4/moments/mean_grad/truediv*'
_output_shapes
:���������d*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape*
N
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/AddN_2dense/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/AddN_2*
T0*
_output_shapes
:	�d*
transpose_a(*
transpose_b( 
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
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
�
gradients/Reshape_grad/ShapeShapeRelu_2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
out_type0*
_output_shapes
:*
T0
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
�
9gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitchgradients/Relu_2_grad/ReluGrad"batch_normalization_3/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_3/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1:^gradients/batch_normalization_3/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_3/cond/Merge_grad/cond_gradA^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
Jgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:���������
�
gradients/zeros_like	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_1	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_2	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_3	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1N^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
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
�
gradients/zeros_like_4	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_5	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_6	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_7	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
Igradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1L^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
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
�
gradients/SwitchSwitchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
c
gradients/Shape_1Shapegradients/Switch:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
{
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*/
_output_shapes
:���������*
T0
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_2Shapegradients/Switch_1:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_1/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
N*
_output_shapes

:: *
T0
�
gradients/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_2/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
N*
_output_shapes

:: *
T0
�
gradients/Switch_3Switchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
c
gradients/Shape_4Shapegradients/Switch_3*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_3/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*/
_output_shapes
:���������*
T0
�
Igradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_4Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
c
gradients/Shape_5Shapegradients/Switch_4*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_4/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
j
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_5Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
c
gradients/Shape_6Shapegradients/Switch_5*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_5/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
j
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_3AddNKgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/AddN_3*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/AddN_3*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
/gradients/conv2d_3/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_13^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_4AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N
�
gradients/AddN_5AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
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
�
@gradients/batch_normalization_2/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1:^gradients/batch_normalization_2/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_2/cond/Merge_grad/cond_gradA^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*/
_output_shapes
:���������
�
Jgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*/
_output_shapes
:���������*
T0
�
gradients/zeros_like_8	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_9	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_10	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_11	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1N^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
gradients/zeros_like_12	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_13	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_14	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_15	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_12batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:1+batch_normalization_2/cond/FusedBatchNorm:3+batch_normalization_2/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:
�
Igradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1L^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
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
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
gradients/Switch_6Switchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
e
gradients/Shape_7Shapegradients/Switch_6:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_6/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*/
_output_shapes
:���������*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_7Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_8Shapegradients/Switch_7:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_7/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_8Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_9Shapegradients/Switch_8:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_8/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
j
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_9Switchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
d
gradients/Shape_10Shapegradients/Switch_9*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_9/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_10Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_11Shapegradients/Switch_10*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_10/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_11Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_12Shapegradients/Switch_11*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_11/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
N*
_output_shapes

:: *
T0
�
gradients/AddN_6AddNKgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_grad*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N
�
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0*
out_type0*
N
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/AddN_6*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/AddN_6*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
/gradients/conv2d_2/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_13^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter
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
gradients/AddN_7AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_8AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
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
7gradients/batch_normalization/cond/Merge_grad/cond_gradSwitchgradients/Relu_grad/ReluGrad batch_normalization/cond/pred_id*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*J
_output_shapes8
6:���������:���������*
T0
�
>gradients/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_18^gradients/batch_normalization/cond/Merge_grad/cond_grad
�
Fgradients/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentity7gradients/batch_normalization/cond/Merge_grad/cond_grad?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:���������*
T0
�
Hgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_1Identity9gradients/batch_normalization/cond/Merge_grad/cond_grad:1?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad
�
gradients/zeros_like_16	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_17	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_18	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_19	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradFgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
Igradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1L^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityKgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradJ^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
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
�
gradients/zeros_like_20	ZerosLike)batch_normalization/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_21	ZerosLike)batch_normalization/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_22	ZerosLike)batch_normalization/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_23	ZerosLike)batch_normalization/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Igradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_10batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:1)batch_normalization/cond/FusedBatchNorm:3)batch_normalization/cond/FusedBatchNorm:4*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:*
T0
�
Ggradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1J^gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Ogradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityIgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradH^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
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
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
gradients/Switch_12Switchconv2d/Conv2D batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
g
gradients/Shape_13Shapegradients/Switch_12:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_12/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
�
gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*/
_output_shapes
:���������*
T0
�
Igradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_13Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
g
gradients/Shape_14Shapegradients/Switch_13:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_13/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
T0*
_output_shapes
:
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_14Switchbatch_normalization/beta/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_15Shapegradients/Switch_14:1*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_14/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
N*
_output_shapes

:: *
T0
�
gradients/Switch_15Switchconv2d/Conv2D batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
e
gradients/Shape_16Shapegradients/Switch_15*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_15/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*/
_output_shapes
:���������
�
Ggradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeOgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_16Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_17Shapegradients/Switch_16*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_16/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
T0*
_output_shapes
:
�
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_17Switchbatch_normalization/beta/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_18Shapegradients/Switch_17*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_17/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
T0*
_output_shapes
:
�
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_9AddNIgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradGgradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*/
_output_shapes
:���������*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad
�
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
out_type0*
N* 
_output_shapes
::*
T0
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/AddN_9*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/AddN_9*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_11^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
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
gradients/AddN_10AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_11AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
GradientDescent/learning_rateConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
EGradientDescent/update_batch_normalization/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization/gammaGradientDescent/learning_rategradients/AddN_10*
use_locking( *
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:
�
DGradientDescent/update_batch_normalization/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization/betaGradientDescent/learning_rategradients/AddN_11*
use_locking( *
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
GGradientDescent/update_batch_normalization_1/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_1/gammaGradientDescent/learning_rategradients/AddN_7*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
�
FGradientDescent/update_batch_normalization_1/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_1/betaGradientDescent/learning_rategradients/AddN_8*
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
GGradientDescent/update_batch_normalization_2/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_2/gammaGradientDescent/learning_rategradients/AddN_4*
_output_shapes
:*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_2/gamma
�
FGradientDescent/update_batch_normalization_2/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_2/betaGradientDescent/learning_rategradients/AddN_5*
_output_shapes
:*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_2/beta
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�d*
use_locking( *
T0*
_class
loc:@dense/kernel
�
GGradientDescent/update_batch_normalization_3/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_3/gammaGradientDescent/learning_rateMgradients/batch_normalization_4/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:d
�
FGradientDescent/update_batch_normalization_3/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_3/betaGradientDescent/learning_rateKgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes
:d*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_3/beta
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
�

GradientDescentNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1:^GradientDescent/update_conv2d/kernel/ApplyGradientDescentF^GradientDescent/update_batch_normalization/gamma/ApplyGradientDescentE^GradientDescent/update_batch_normalization/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_1/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_1/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_2/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_2/beta/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_3/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_3/beta/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
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
: "�O�߀     ����	�b���AJҁ
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
Ttype*1.4.12v1.4.0-19-ga52c8d9��
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
 batch_normalization/gamma/AssignAssignbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:*
use_locking(
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
batch_normalization/beta/AssignAssignbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta
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
VariableV2*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
:
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
VariableV2*
dtype0*
_output_shapes
:*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:
�
*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
validate_shape(*
_output_shapes
:
�
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:*
T0
n
batch_normalization/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes

::
s
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
_output_shapes
:*
T0

q
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
T0
*
_output_shapes
:
^
 batch_normalization/cond/pred_idIdentityPlaceholder_2*
T0
*
_output_shapes
:
�
batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchconv2d/Conv2D batch_normalization/cond/pred_id*
T0* 
_class
loc:@conv2d/Conv2D*J
_output_shapes8
6:���������:���������
�
0batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
::*
T0
�
0batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
::
�
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:12batch_normalization/cond/FusedBatchNorm/Switch_2:1batch_normalization/cond/Const batch_normalization/cond/Const_1*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:*
T0*
data_formatNHWC
�
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchconv2d/Conv2D batch_normalization/cond/pred_id*
T0* 
_class
loc:@conv2d/Conv2D*J
_output_shapes8
6:���������:���������
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
::*
T0
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id* 
_output_shapes
::*
T0*+
_class!
loc:@batch_normalization/beta
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read batch_normalization/cond/pred_id* 
_output_shapes
::*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
2batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read batch_normalization/cond/pred_id* 
_output_shapes
::*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_22batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
�
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
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
"batch_normalization/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
�
batch_normalization/ExpandDims
ExpandDims$batch_normalization/ExpandDims/input"batch_normalization/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
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
batch_normalization/SqueezeSqueezebatch_normalization/Select*
_output_shapes
: *
squeeze_dims
 *
T0
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
#batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/Mul*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:*
use_locking( *
T0
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
)batch_normalization/AssignMovingAvg_1/MulMul)batch_normalization/AssignMovingAvg_1/Subbatch_normalization/Squeeze*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
�
%batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/Mul*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:*
use_locking( *
T0
f
ReluRelubatch_normalization/cond/Merge*/
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
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma
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
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(
�
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_1/beta
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
VariableV2*4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(
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
2batch_normalization_2/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::*
T0
�
2batch_normalization_2/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_1/beta
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
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
::
�
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_1/beta
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
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_24batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
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
$batch_normalization_2/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0
�
 batch_normalization_2/ExpandDims
ExpandDims&batch_normalization_2/ExpandDims/input$batch_normalization_2/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
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
#batch_normalization_2/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_2/ReshapeReshapePlaceholder_2#batch_normalization_2/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_2/SelectSelectbatch_normalization_2/Reshape batch_normalization_2/ExpandDims"batch_normalization_2/ExpandDims_1*
T0*
_output_shapes
:
~
batch_normalization_2/SqueezeSqueezebatch_normalization_2/Select*
_output_shapes
: *
squeeze_dims
 *
T0
�
*batch_normalization_2/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
)batch_normalization_2/AssignMovingAvg/SubSub*batch_normalization_2/AssignMovingAvg/read"batch_normalization_2/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
)batch_normalization_2/AssignMovingAvg/MulMul)batch_normalization_2/AssignMovingAvg/Subbatch_normalization_2/Squeeze*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_2/AssignMovingAvg/Mul*
_output_shapes
:*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
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
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_2/AssignMovingAvg_1/Mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:*
use_locking( *
T0
j
Relu_1Relu batch_normalization_2/cond/Merge*/
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
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
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
,batch_normalization_2/beta/Initializer/zerosConst*
_output_shapes
:*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0
�
batch_normalization_2/beta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta
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
VariableV2*8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
!batch_normalization_3/cond/SwitchSwitchPlaceholder_2Placeholder_2*
T0
*
_output_shapes

::
w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
:
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
 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
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
2batch_normalization_3/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_2/beta
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
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_2/beta
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read"batch_normalization_3/cond/pred_id* 
_output_shapes
::*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
::
�
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_24batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
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
ExpandDims&batch_normalization_3/ExpandDims/input$batch_normalization_3/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
m
(batch_normalization_3/ExpandDims_1/inputConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
batch_normalization_3/SqueezeSqueezebatch_normalization_3/Select*
squeeze_dims
 *
T0*
_output_shapes
: 
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
)batch_normalization_3/AssignMovingAvg/MulMul)batch_normalization_3/AssignMovingAvg/Subbatch_normalization_3/Squeeze*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
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
+batch_normalization_3/AssignMovingAvg_1/MulMul+batch_normalization_3/AssignMovingAvg_1/Subbatch_normalization_3/Squeeze*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_3/AssignMovingAvg_1/Mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
use_locking( *
T0
j
Relu_2Relu batch_normalization_3/cond/Merge*/
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
ReshapeReshapeRelu_2Reshape/shape*
Tshape0*(
_output_shapes
:����������*
T0
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
_class
loc:@dense/kernel*
valueB
 *�J�*
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
 *�J=
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_output_shapes
:	�d*

seed *
T0*
_class
loc:@dense/kernel*
seed2 *
dtype0
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
VariableV2*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d*
shared_name *
_class
loc:@dense/kernel
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	�d
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
�
,batch_normalization_3/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_3/gamma*
valueBd*  �?*
dtype0*
_output_shapes
:d
�
batch_normalization_3/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes
:d
�
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:d
�
,batch_normalization_3/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_3/beta*
valueBd*    *
dtype0*
_output_shapes
:d
�
batch_normalization_3/beta
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container 
�
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes
:d
�
batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:d
�
3batch_normalization_3/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueBd*    *
dtype0*
_output_shapes
:d
�
!batch_normalization_3/moving_mean
VariableV2*
shared_name *4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes
:d
�
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*
_output_shapes
:d*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
6batch_normalization_3/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueBd*  �?*
dtype0*
_output_shapes
:d
�
%batch_normalization_3/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
_output_shapes
:d*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
validate_shape(
�
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:d
~
4batch_normalization_4/moments/mean/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
"batch_normalization_4/moments/meanMeandense/MatMul4batch_normalization_4/moments/mean/reduction_indices*
T0*
_output_shapes

:d*

Tidx0*
	keep_dims(
�
*batch_normalization_4/moments/StopGradientStopGradient"batch_normalization_4/moments/mean*
_output_shapes

:d*
T0
�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense/MatMul*batch_normalization_4/moments/StopGradient*
T0*'
_output_shapes
:���������d
�
8batch_normalization_4/moments/variance/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
�
&batch_normalization_4/moments/varianceMean/batch_normalization_4/moments/SquaredDifference8batch_normalization_4/moments/variance/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:d
�
%batch_normalization_4/moments/SqueezeSqueeze"batch_normalization_4/moments/mean*
_output_shapes
:d*
squeeze_dims
 *
T0
�
'batch_normalization_4/moments/Squeeze_1Squeeze&batch_normalization_4/moments/variance*
T0*
_output_shapes
:d*
squeeze_dims
 
f
$batch_normalization_4/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 batch_normalization_4/ExpandDims
ExpandDims%batch_normalization_4/moments/Squeeze$batch_normalization_4/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:d
h
&batch_normalization_4/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_1
ExpandDims&batch_normalization_3/moving_mean/read&batch_normalization_4/ExpandDims_1/dim*
T0*
_output_shapes

:d*

Tdim0
m
#batch_normalization_4/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
batch_normalization_4/ReshapeReshapePlaceholder_2#batch_normalization_4/Reshape/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_4/SelectSelectbatch_normalization_4/Reshape batch_normalization_4/ExpandDims"batch_normalization_4/ExpandDims_1*
_output_shapes

:d*
T0
�
batch_normalization_4/SqueezeSqueezebatch_normalization_4/Select*
T0*
_output_shapes
:d*
squeeze_dims
 
h
&batch_normalization_4/ExpandDims_2/dimConst*
_output_shapes
: *
value	B : *
dtype0
�
"batch_normalization_4/ExpandDims_2
ExpandDims'batch_normalization_4/moments/Squeeze_1&batch_normalization_4/ExpandDims_2/dim*
T0*
_output_shapes

:d*

Tdim0
h
&batch_normalization_4/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_3
ExpandDims*batch_normalization_3/moving_variance/read&batch_normalization_4/ExpandDims_3/dim*
T0*
_output_shapes

:d*

Tdim0
o
%batch_normalization_4/Reshape_1/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_4/Reshape_1ReshapePlaceholder_2%batch_normalization_4/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
�
batch_normalization_4/Select_1Selectbatch_normalization_4/Reshape_1"batch_normalization_4/ExpandDims_2"batch_normalization_4/ExpandDims_3*
_output_shapes

:d*
T0
�
batch_normalization_4/Squeeze_1Squeezebatch_normalization_4/Select_1*
_output_shapes
:d*
squeeze_dims
 *
T0
m
(batch_normalization_4/ExpandDims_4/inputConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
h
&batch_normalization_4/ExpandDims_4/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_4
ExpandDims(batch_normalization_4/ExpandDims_4/input&batch_normalization_4/ExpandDims_4/dim*
T0*
_output_shapes
:*

Tdim0
m
(batch_normalization_4/ExpandDims_5/inputConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
h
&batch_normalization_4/ExpandDims_5/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/ExpandDims_5
ExpandDims(batch_normalization_4/ExpandDims_5/input&batch_normalization_4/ExpandDims_5/dim*
_output_shapes
:*

Tdim0*
T0
o
%batch_normalization_4/Reshape_2/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
batch_normalization_4/Reshape_2ReshapePlaceholder_2%batch_normalization_4/Reshape_2/shape*
T0
*
Tshape0*
_output_shapes
:
�
batch_normalization_4/Select_2Selectbatch_normalization_4/Reshape_2"batch_normalization_4/ExpandDims_4"batch_normalization_4/ExpandDims_5*
_output_shapes
:*
T0
�
batch_normalization_4/Squeeze_2Squeezebatch_normalization_4/Select_2*
squeeze_dims
 *
T0*
_output_shapes
: 
�
+batch_normalization_4/AssignMovingAvg/sub/xConst*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
�
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/xbatch_normalization_4/Squeeze_2*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: 
�
+batch_normalization_4/AssignMovingAvg/sub_1Sub&batch_normalization_3/moving_mean/readbatch_normalization_4/Squeeze*
_output_shapes
:d*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*
_output_shapes
:d*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_4/AssignMovingAvg/mul*
_output_shapes
:d*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
�
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/xbatch_normalization_4/Squeeze_2*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub*batch_normalization_3/moving_variance/readbatch_normalization_4/Squeeze_1*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:d
�
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:d
�
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:d*
use_locking( *
T0
j
%batch_normalization_4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
#batch_normalization_4/batchnorm/addAddbatch_normalization_4/Squeeze_1%batch_normalization_4/batchnorm/add/y*
T0*
_output_shapes
:d
x
%batch_normalization_4/batchnorm/RsqrtRsqrt#batch_normalization_4/batchnorm/add*
_output_shapes
:d*
T0
�
#batch_normalization_4/batchnorm/mulMul%batch_normalization_4/batchnorm/Rsqrt batch_normalization_3/gamma/read*
T0*
_output_shapes
:d
�
%batch_normalization_4/batchnorm/mul_1Muldense/MatMul#batch_normalization_4/batchnorm/mul*
T0*'
_output_shapes
:���������d
�
%batch_normalization_4/batchnorm/mul_2Mulbatch_normalization_4/Squeeze#batch_normalization_4/batchnorm/mul*
T0*
_output_shapes
:d
�
#batch_normalization_4/batchnorm/subSubbatch_normalization_3/beta/read%batch_normalization_4/batchnorm/mul_2*
_output_shapes
:d*
T0
�
%batch_normalization_4/batchnorm/add_1Add%batch_normalization_4/batchnorm/mul_1#batch_normalization_4/batchnorm/sub*
T0*'
_output_shapes
:���������d
g
Relu_3Relu%batch_normalization_4/batchnorm/add_1*'
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
�
gradients/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB 
�
gradients/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
!gradients/Mean_grad/Reshape/shapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
gradients/Mean_grad/ShapeShapelogistic_loss$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
gradients/Mean_grad/Shape_1Shapelogistic_loss$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/Shape_2Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
gradients/Mean_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
gradients/Mean_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

�
-gradients/logistic_loss_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
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

�
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
out_type0*
_output_shapes
:*
T0
�
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
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

�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
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

�
(gradients/logistic_loss/Log1p_grad/add/xConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_18^gradients/logistic_loss_grad/tuple/control_dependency_1*
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

�
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikedense_2/BiasAdd$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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

�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
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
�
&gradients/logistic_loss/mul_grad/ShapeShapedense_2/BiasAdd$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/logistic_loss/mul_grad/Shape_1ShapePlaceholder_1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
&gradients/logistic_loss/mul_grad/mul_1Muldense_2/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������
*
T0
�
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
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

�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
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
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:

�
/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1^gradients/AddN+^gradients/dense_2/BiasAdd_grad/BiasAddGrad
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
&gradients/dense_2/MatMul_grad/MatMul_1MatMulRelu_37gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:d
*
transpose_a(*
transpose_b( *
T0
�
.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1
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
gradients/Relu_3_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyRelu_3*
T0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_4/batchnorm/add_1_grad/ShapeShape%batch_normalization_4/batchnorm/mul_1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
<gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_4/batchnorm/add_1_grad/Shape<gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients/batch_normalization_4/batchnorm/add_1_grad/SumSumgradients/Relu_3_grad/ReluGradJgradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
<gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeReshape8gradients/batch_normalization_4/batchnorm/add_1_grad/Sum:gradients/batch_normalization_4/batchnorm/add_1_grad/Shape*
Tshape0*'
_output_shapes
:���������d*
T0
�
:gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1Sumgradients/Relu_3_grad/ReluGradLgradients/batch_normalization_4/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
>gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1Reshape:gradients/batch_normalization_4/batchnorm/add_1_grad/Sum_1<gradients/batch_normalization_4/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Egradients/batch_normalization_4/batchnorm/add_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1=^gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape?^gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1
�
Mgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_4/batchnorm/add_1_grad/ReshapeF^gradients/batch_normalization_4/batchnorm/add_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape*'
_output_shapes
:���������d
�
Ogradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1F^gradients/batch_normalization_4/batchnorm/add_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_4/batchnorm/add_1_grad/Reshape_1*
_output_shapes
:d
�
:gradients/batch_normalization_4/batchnorm/mul_1_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0*
out_type0
�
<gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape<gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients/batch_normalization_4/batchnorm/mul_1_grad/mulMulMgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency#batch_normalization_4/batchnorm/mul*'
_output_shapes
:���������d*
T0
�
8gradients/batch_normalization_4/batchnorm/mul_1_grad/SumSum8gradients/batch_normalization_4/batchnorm/mul_1_grad/mulJgradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
<gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeReshape8gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum:gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:���������d*
T0
�
:gradients/batch_normalization_4/batchnorm/mul_1_grad/mul_1Muldense/MatMulMgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������d
�
:gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1Sum:gradients/batch_normalization_4/batchnorm/mul_1_grad/mul_1Lgradients/batch_normalization_4/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
>gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1Reshape:gradients/batch_normalization_4/batchnorm/mul_1_grad/Sum_1<gradients/batch_normalization_4/batchnorm/mul_1_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
�
Egradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1=^gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape?^gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1
�
Mgradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_4/batchnorm/mul_1_grad/ReshapeF^gradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape
�
Ogradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1F^gradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1*
_output_shapes
:d
�
8gradients/batch_normalization_4/batchnorm/sub_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
:gradients/batch_normalization_4/batchnorm/sub_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
Hgradients/batch_normalization_4/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_4/batchnorm/sub_grad/Shape:gradients/batch_normalization_4/batchnorm/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients/batch_normalization_4/batchnorm/sub_grad/SumSumOgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency_1Hgradients/batch_normalization_4/batchnorm/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:gradients/batch_normalization_4/batchnorm/sub_grad/ReshapeReshape6gradients/batch_normalization_4/batchnorm/sub_grad/Sum8gradients/batch_normalization_4/batchnorm/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
8gradients/batch_normalization_4/batchnorm/sub_grad/Sum_1SumOgradients/batch_normalization_4/batchnorm/add_1_grad/tuple/control_dependency_1Jgradients/batch_normalization_4/batchnorm/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
6gradients/batch_normalization_4/batchnorm/sub_grad/NegNeg8gradients/batch_normalization_4/batchnorm/sub_grad/Sum_1*
_output_shapes
:*
T0
�
<gradients/batch_normalization_4/batchnorm/sub_grad/Reshape_1Reshape6gradients/batch_normalization_4/batchnorm/sub_grad/Neg:gradients/batch_normalization_4/batchnorm/sub_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
�
Cgradients/batch_normalization_4/batchnorm/sub_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1;^gradients/batch_normalization_4/batchnorm/sub_grad/Reshape=^gradients/batch_normalization_4/batchnorm/sub_grad/Reshape_1
�
Kgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_4/batchnorm/sub_grad/ReshapeD^gradients/batch_normalization_4/batchnorm/sub_grad/tuple/group_deps*M
_classC
A?loc:@gradients/batch_normalization_4/batchnorm/sub_grad/Reshape*
_output_shapes
:d*
T0
�
Mgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_4/batchnorm/sub_grad/Reshape_1D^gradients/batch_normalization_4/batchnorm/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/sub_grad/Reshape_1*
_output_shapes
:d
�
:gradients/batch_normalization_4/batchnorm/mul_2_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
<gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Jgradients/batch_normalization_4/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape<gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients/batch_normalization_4/batchnorm/mul_2_grad/mulMulMgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependency_1#batch_normalization_4/batchnorm/mul*
_output_shapes
:d*
T0
�
8gradients/batch_normalization_4/batchnorm/mul_2_grad/SumSum8gradients/batch_normalization_4/batchnorm/mul_2_grad/mulJgradients/batch_normalization_4/batchnorm/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
<gradients/batch_normalization_4/batchnorm/mul_2_grad/ReshapeReshape8gradients/batch_normalization_4/batchnorm/mul_2_grad/Sum:gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
:gradients/batch_normalization_4/batchnorm/mul_2_grad/mul_1Mulbatch_normalization_4/SqueezeMgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0
�
:gradients/batch_normalization_4/batchnorm/mul_2_grad/Sum_1Sum:gradients/batch_normalization_4/batchnorm/mul_2_grad/mul_1Lgradients/batch_normalization_4/batchnorm/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
>gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape_1Reshape:gradients/batch_normalization_4/batchnorm/mul_2_grad/Sum_1<gradients/batch_normalization_4/batchnorm/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Egradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1=^gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape?^gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape_1
�
Mgradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/control_dependencyIdentity<gradients/batch_normalization_4/batchnorm/mul_2_grad/ReshapeF^gradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape*
_output_shapes
:d
�
Ogradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/control_dependency_1Identity>gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape_1F^gradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/batch_normalization_4/batchnorm/mul_2_grad/Reshape_1*
_output_shapes
:d
�
2gradients/batch_normalization_4/Squeeze_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
4gradients/batch_normalization_4/Squeeze_grad/ReshapeReshapeMgradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/control_dependency2gradients/batch_normalization_4/Squeeze_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
gradients/AddN_1AddNOgradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/control_dependency_1Ogradients/batch_normalization_4/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes
:d
�
8gradients/batch_normalization_4/batchnorm/mul_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
:gradients/batch_normalization_4/batchnorm/mul_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
Hgradients/batch_normalization_4/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_4/batchnorm/mul_grad/Shape:gradients/batch_normalization_4/batchnorm/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients/batch_normalization_4/batchnorm/mul_grad/mulMulgradients/AddN_1 batch_normalization_3/gamma/read*
_output_shapes
:d*
T0
�
6gradients/batch_normalization_4/batchnorm/mul_grad/SumSum6gradients/batch_normalization_4/batchnorm/mul_grad/mulHgradients/batch_normalization_4/batchnorm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:gradients/batch_normalization_4/batchnorm/mul_grad/ReshapeReshape6gradients/batch_normalization_4/batchnorm/mul_grad/Sum8gradients/batch_normalization_4/batchnorm/mul_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
8gradients/batch_normalization_4/batchnorm/mul_grad/mul_1Mul%batch_normalization_4/batchnorm/Rsqrtgradients/AddN_1*
T0*
_output_shapes
:d
�
8gradients/batch_normalization_4/batchnorm/mul_grad/Sum_1Sum8gradients/batch_normalization_4/batchnorm/mul_grad/mul_1Jgradients/batch_normalization_4/batchnorm/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
<gradients/batch_normalization_4/batchnorm/mul_grad/Reshape_1Reshape8gradients/batch_normalization_4/batchnorm/mul_grad/Sum_1:gradients/batch_normalization_4/batchnorm/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
�
Cgradients/batch_normalization_4/batchnorm/mul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1;^gradients/batch_normalization_4/batchnorm/mul_grad/Reshape=^gradients/batch_normalization_4/batchnorm/mul_grad/Reshape_1
�
Kgradients/batch_normalization_4/batchnorm/mul_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_4/batchnorm/mul_grad/ReshapeD^gradients/batch_normalization_4/batchnorm/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/batch_normalization_4/batchnorm/mul_grad/Reshape*
_output_shapes
:d
�
Mgradients/batch_normalization_4/batchnorm/mul_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_4/batchnorm/mul_grad/Reshape_1D^gradients/batch_normalization_4/batchnorm/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/mul_grad/Reshape_1*
_output_shapes
:d
�
6gradients/batch_normalization_4/Select_grad/zeros_likeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueBd*    *
dtype0*
_output_shapes

:d
�
2gradients/batch_normalization_4/Select_grad/SelectSelectbatch_normalization_4/Reshape4gradients/batch_normalization_4/Squeeze_grad/Reshape6gradients/batch_normalization_4/Select_grad/zeros_like*
T0*
_output_shapes

:d
�
4gradients/batch_normalization_4/Select_grad/Select_1Selectbatch_normalization_4/Reshape6gradients/batch_normalization_4/Select_grad/zeros_like4gradients/batch_normalization_4/Squeeze_grad/Reshape*
T0*
_output_shapes

:d
�
<gradients/batch_normalization_4/Select_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_13^gradients/batch_normalization_4/Select_grad/Select5^gradients/batch_normalization_4/Select_grad/Select_1
�
Dgradients/batch_normalization_4/Select_grad/tuple/control_dependencyIdentity2gradients/batch_normalization_4/Select_grad/Select=^gradients/batch_normalization_4/Select_grad/tuple/group_deps*
_output_shapes

:d*
T0*E
_class;
97loc:@gradients/batch_normalization_4/Select_grad/Select
�
Fgradients/batch_normalization_4/Select_grad/tuple/control_dependency_1Identity4gradients/batch_normalization_4/Select_grad/Select_1=^gradients/batch_normalization_4/Select_grad/tuple/group_deps*
_output_shapes

:d*
T0*G
_class=
;9loc:@gradients/batch_normalization_4/Select_grad/Select_1
�
>gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad%batch_normalization_4/batchnorm/RsqrtKgradients/batch_normalization_4/batchnorm/mul_grad/tuple/control_dependency*
T0*
_output_shapes
:d
�
5gradients/batch_normalization_4/ExpandDims_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
7gradients/batch_normalization_4/ExpandDims_grad/ReshapeReshapeDgradients/batch_normalization_4/Select_grad/tuple/control_dependency5gradients/batch_normalization_4/ExpandDims_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
8gradients/batch_normalization_4/batchnorm/add_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB:d*
dtype0
�
:gradients/batch_normalization_4/batchnorm/add_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB *
dtype0*
_output_shapes
: 
�
Hgradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/batch_normalization_4/batchnorm/add_grad/Shape:gradients/batch_normalization_4/batchnorm/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients/batch_normalization_4/batchnorm/add_grad/SumSum>gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradHgradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:gradients/batch_normalization_4/batchnorm/add_grad/ReshapeReshape6gradients/batch_normalization_4/batchnorm/add_grad/Sum8gradients/batch_normalization_4/batchnorm/add_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
8gradients/batch_normalization_4/batchnorm/add_grad/Sum_1Sum>gradients/batch_normalization_4/batchnorm/Rsqrt_grad/RsqrtGradJgradients/batch_normalization_4/batchnorm/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
<gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1Reshape8gradients/batch_normalization_4/batchnorm/add_grad/Sum_1:gradients/batch_normalization_4/batchnorm/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Cgradients/batch_normalization_4/batchnorm/add_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1;^gradients/batch_normalization_4/batchnorm/add_grad/Reshape=^gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1
�
Kgradients/batch_normalization_4/batchnorm/add_grad/tuple/control_dependencyIdentity:gradients/batch_normalization_4/batchnorm/add_grad/ReshapeD^gradients/batch_normalization_4/batchnorm/add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/batch_normalization_4/batchnorm/add_grad/Reshape*
_output_shapes
:d
�
Mgradients/batch_normalization_4/batchnorm/add_grad/tuple/control_dependency_1Identity<gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1D^gradients/batch_normalization_4/batchnorm/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
:gradients/batch_normalization_4/moments/Squeeze_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
<gradients/batch_normalization_4/moments/Squeeze_grad/ReshapeReshape7gradients/batch_normalization_4/ExpandDims_grad/Reshape:gradients/batch_normalization_4/moments/Squeeze_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
4gradients/batch_normalization_4/Squeeze_1_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
6gradients/batch_normalization_4/Squeeze_1_grad/ReshapeReshapeKgradients/batch_normalization_4/batchnorm/add_grad/tuple/control_dependency4gradients/batch_normalization_4/Squeeze_1_grad/Shape*
T0*
Tshape0*
_output_shapes

:d
�
8gradients/batch_normalization_4/Select_1_grad/zeros_likeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueBd*    *
dtype0*
_output_shapes

:d
�
4gradients/batch_normalization_4/Select_1_grad/SelectSelectbatch_normalization_4/Reshape_16gradients/batch_normalization_4/Squeeze_1_grad/Reshape8gradients/batch_normalization_4/Select_1_grad/zeros_like*
T0*
_output_shapes

:d
�
6gradients/batch_normalization_4/Select_1_grad/Select_1Selectbatch_normalization_4/Reshape_18gradients/batch_normalization_4/Select_1_grad/zeros_like6gradients/batch_normalization_4/Squeeze_1_grad/Reshape*
_output_shapes

:d*
T0
�
>gradients/batch_normalization_4/Select_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_15^gradients/batch_normalization_4/Select_1_grad/Select7^gradients/batch_normalization_4/Select_1_grad/Select_1
�
Fgradients/batch_normalization_4/Select_1_grad/tuple/control_dependencyIdentity4gradients/batch_normalization_4/Select_1_grad/Select?^gradients/batch_normalization_4/Select_1_grad/tuple/group_deps*
_output_shapes

:d*
T0*G
_class=
;9loc:@gradients/batch_normalization_4/Select_1_grad/Select
�
Hgradients/batch_normalization_4/Select_1_grad/tuple/control_dependency_1Identity6gradients/batch_normalization_4/Select_1_grad/Select_1?^gradients/batch_normalization_4/Select_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/batch_normalization_4/Select_1_grad/Select_1*
_output_shapes

:d
�
7gradients/batch_normalization_4/ExpandDims_2_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:d*
dtype0*
_output_shapes
:
�
9gradients/batch_normalization_4/ExpandDims_2_grad/ReshapeReshapeFgradients/batch_normalization_4/Select_1_grad/tuple/control_dependency7gradients/batch_normalization_4/ExpandDims_2_grad/Shape*
_output_shapes
:d*
T0*
Tshape0
�
<gradients/batch_normalization_4/moments/Squeeze_1_grad/ShapeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
>gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeReshape9gradients/batch_normalization_4/ExpandDims_2_grad/Reshape<gradients/batch_normalization_4/moments/Squeeze_1_grad/Shape*
Tshape0*
_output_shapes

:d*
T0
�
;gradients/batch_normalization_4/moments/variance_grad/ShapeShape/batch_normalization_4/moments/SquaredDifference$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
:gradients/batch_normalization_4/moments/variance_grad/SizeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
9gradients/batch_normalization_4/moments/variance_grad/addAdd8batch_normalization_4/moments/variance/reduction_indices:gradients/batch_normalization_4/moments/variance_grad/Size*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_4/moments/variance_grad/modFloorMod9gradients/batch_normalization_4/moments/variance_grad/add:gradients/batch_normalization_4/moments/variance_grad/Size*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:*
T0
�
=gradients/batch_normalization_4/moments/variance_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
:
�
Agradients/batch_normalization_4/moments/variance_grad/range/startConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B : *N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Agradients/batch_normalization_4/moments/variance_grad/range/deltaConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
;gradients/batch_normalization_4/moments/variance_grad/rangeRangeAgradients/batch_normalization_4/moments/variance_grad/range/start:gradients/batch_normalization_4/moments/variance_grad/SizeAgradients/batch_normalization_4/moments/variance_grad/range/delta*
_output_shapes
:*

Tidx0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape
�
@gradients/batch_normalization_4/moments/variance_grad/Fill/valueConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
:gradients/batch_normalization_4/moments/variance_grad/FillFill=gradients/batch_normalization_4/moments/variance_grad/Shape_1@gradients/batch_normalization_4/moments/variance_grad/Fill/value*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:
�
Cgradients/batch_normalization_4/moments/variance_grad/DynamicStitchDynamicStitch;gradients/batch_normalization_4/moments/variance_grad/range9gradients/batch_normalization_4/moments/variance_grad/mod;gradients/batch_normalization_4/moments/variance_grad/Shape:gradients/batch_normalization_4/moments/variance_grad/Fill*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
N*#
_output_shapes
:���������
�
?gradients/batch_normalization_4/moments/variance_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
=gradients/batch_normalization_4/moments/variance_grad/MaximumMaximumCgradients/batch_normalization_4/moments/variance_grad/DynamicStitch?gradients/batch_normalization_4/moments/variance_grad/Maximum/y*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*#
_output_shapes
:���������
�
>gradients/batch_normalization_4/moments/variance_grad/floordivFloorDiv;gradients/batch_normalization_4/moments/variance_grad/Shape=gradients/batch_normalization_4/moments/variance_grad/Maximum*
T0*N
_classD
B@loc:@gradients/batch_normalization_4/moments/variance_grad/Shape*
_output_shapes
:
�
=gradients/batch_normalization_4/moments/variance_grad/ReshapeReshape>gradients/batch_normalization_4/moments/Squeeze_1_grad/ReshapeCgradients/batch_normalization_4/moments/variance_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
:gradients/batch_normalization_4/moments/variance_grad/TileTile=gradients/batch_normalization_4/moments/variance_grad/Reshape>gradients/batch_normalization_4/moments/variance_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
=gradients/batch_normalization_4/moments/variance_grad/Shape_2Shape/batch_normalization_4/moments/SquaredDifference$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0*
out_type0
�
=gradients/batch_normalization_4/moments/variance_grad/Shape_3Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
;gradients/batch_normalization_4/moments/variance_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
valueB: *P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
dtype0
�
:gradients/batch_normalization_4/moments/variance_grad/ProdProd=gradients/batch_normalization_4/moments/variance_grad/Shape_2;gradients/batch_normalization_4/moments/variance_grad/Const*

Tidx0*
	keep_dims( *
T0*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
_output_shapes
: 
�
=gradients/batch_normalization_4/moments/variance_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB: *P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
<gradients/batch_normalization_4/moments/variance_grad/Prod_1Prod=gradients/batch_normalization_4/moments/variance_grad/Shape_3=gradients/batch_normalization_4/moments/variance_grad/Const_1*
T0*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Agradients/batch_normalization_4/moments/variance_grad/Maximum_1/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
value	B :*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2
�
?gradients/batch_normalization_4/moments/variance_grad/Maximum_1Maximum<gradients/batch_normalization_4/moments/variance_grad/Prod_1Agradients/batch_normalization_4/moments/variance_grad/Maximum_1/y*
T0*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2*
_output_shapes
: 
�
@gradients/batch_normalization_4/moments/variance_grad/floordiv_1FloorDiv:gradients/batch_normalization_4/moments/variance_grad/Prod?gradients/batch_normalization_4/moments/variance_grad/Maximum_1*
_output_shapes
: *
T0*P
_classF
DBloc:@gradients/batch_normalization_4/moments/variance_grad/Shape_2
�
:gradients/batch_normalization_4/moments/variance_grad/CastCast@gradients/batch_normalization_4/moments/variance_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
=gradients/batch_normalization_4/moments/variance_grad/truedivRealDiv:gradients/batch_normalization_4/moments/variance_grad/Tile:gradients/batch_normalization_4/moments/variance_grad/Cast*
T0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
Fgradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
Tgradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients/batch_normalization_4/moments/SquaredDifference_grad/ShapeFgradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Egradients/batch_normalization_4/moments/SquaredDifference_grad/scalarConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1>^gradients/batch_normalization_4/moments/variance_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
Bgradients/batch_normalization_4/moments/SquaredDifference_grad/mulMulEgradients/batch_normalization_4/moments/SquaredDifference_grad/scalar=gradients/batch_normalization_4/moments/variance_grad/truediv*'
_output_shapes
:���������d*
T0
�
Bgradients/batch_normalization_4/moments/SquaredDifference_grad/subSubdense/MatMul*batch_normalization_4/moments/StopGradient$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1>^gradients/batch_normalization_4/moments/variance_grad/truediv*
T0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1MulBgradients/batch_normalization_4/moments/SquaredDifference_grad/mulBgradients/batch_normalization_4/moments/SquaredDifference_grad/sub*'
_output_shapes
:���������d*
T0
�
Bgradients/batch_normalization_4/moments/SquaredDifference_grad/SumSumDgradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1Tgradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Fgradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeReshapeBgradients/batch_normalization_4/moments/SquaredDifference_grad/SumDgradients/batch_normalization_4/moments/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
�
Dgradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1SumDgradients/batch_normalization_4/moments/SquaredDifference_grad/mul_1Vgradients/batch_normalization_4/moments/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Hgradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1ReshapeDgradients/batch_normalization_4/moments/SquaredDifference_grad/Sum_1Fgradients/batch_normalization_4/moments/SquaredDifference_grad/Shape_1*
_output_shapes

:d*
T0*
Tshape0
�
Bgradients/batch_normalization_4/moments/SquaredDifference_grad/NegNegHgradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes

:d
�
Ogradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1G^gradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeC^gradients/batch_normalization_4/moments/SquaredDifference_grad/Neg
�
Wgradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/control_dependencyIdentityFgradients/batch_normalization_4/moments/SquaredDifference_grad/ReshapeP^gradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/group_deps*Y
_classO
MKloc:@gradients/batch_normalization_4/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:���������d*
T0
�
Ygradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityBgradients/batch_normalization_4/moments/SquaredDifference_grad/NegP^gradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/group_deps*U
_classK
IGloc:@gradients/batch_normalization_4/moments/SquaredDifference_grad/Neg*
_output_shapes

:d*
T0
�
7gradients/batch_normalization_4/moments/mean_grad/ShapeShapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
6gradients/batch_normalization_4/moments/mean_grad/SizeConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
value	B :*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0
�
5gradients/batch_normalization_4/moments/mean_grad/addAdd4batch_normalization_4/moments/mean/reduction_indices6gradients/batch_normalization_4/moments/mean_grad/Size*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape
�
5gradients/batch_normalization_4/moments/mean_grad/modFloorMod5gradients/batch_normalization_4/moments/mean_grad/add6gradients/batch_normalization_4/moments/mean_grad/Size*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
_output_shapes
:
�
9gradients/batch_normalization_4/moments/mean_grad/Shape_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB:*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
�
=gradients/batch_normalization_4/moments/mean_grad/range/startConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B : *J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
=gradients/batch_normalization_4/moments/mean_grad/range/deltaConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
7gradients/batch_normalization_4/moments/mean_grad/rangeRange=gradients/batch_normalization_4/moments/mean_grad/range/start6gradients/batch_normalization_4/moments/mean_grad/Size=gradients/batch_normalization_4/moments/mean_grad/range/delta*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
�
<gradients/batch_normalization_4/moments/mean_grad/Fill/valueConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
value	B :*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0
�
6gradients/batch_normalization_4/moments/mean_grad/FillFill9gradients/batch_normalization_4/moments/mean_grad/Shape_1<gradients/batch_normalization_4/moments/mean_grad/Fill/value*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
_output_shapes
:
�
?gradients/batch_normalization_4/moments/mean_grad/DynamicStitchDynamicStitch7gradients/batch_normalization_4/moments/mean_grad/range5gradients/batch_normalization_4/moments/mean_grad/mod7gradients/batch_normalization_4/moments/mean_grad/Shape6gradients/batch_normalization_4/moments/mean_grad/Fill*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
N*#
_output_shapes
:���������
�
;gradients/batch_normalization_4/moments/mean_grad/Maximum/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
value	B :*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
9gradients/batch_normalization_4/moments/mean_grad/MaximumMaximum?gradients/batch_normalization_4/moments/mean_grad/DynamicStitch;gradients/batch_normalization_4/moments/mean_grad/Maximum/y*#
_output_shapes
:���������*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape
�
:gradients/batch_normalization_4/moments/mean_grad/floordivFloorDiv7gradients/batch_normalization_4/moments/mean_grad/Shape9gradients/batch_normalization_4/moments/mean_grad/Maximum*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/batch_normalization_4/moments/mean_grad/Shape
�
9gradients/batch_normalization_4/moments/mean_grad/ReshapeReshape<gradients/batch_normalization_4/moments/Squeeze_grad/Reshape?gradients/batch_normalization_4/moments/mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
6gradients/batch_normalization_4/moments/mean_grad/TileTile9gradients/batch_normalization_4/moments/mean_grad/Reshape:gradients/batch_normalization_4/moments/mean_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
9gradients/batch_normalization_4/moments/mean_grad/Shape_2Shapedense/MatMul$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
�
9gradients/batch_normalization_4/moments/mean_grad/Shape_3Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB"   d   *
dtype0*
_output_shapes
:
�
7gradients/batch_normalization_4/moments/mean_grad/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB: *L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
6gradients/batch_normalization_4/moments/mean_grad/ProdProd9gradients/batch_normalization_4/moments/mean_grad/Shape_27gradients/batch_normalization_4/moments/mean_grad/Const*
T0*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
9gradients/batch_normalization_4/moments/mean_grad/Const_1Const$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB: *L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
8gradients/batch_normalization_4/moments/mean_grad/Prod_1Prod9gradients/batch_normalization_4/moments/mean_grad/Shape_39gradients/batch_normalization_4/moments/mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2
�
=gradients/batch_normalization_4/moments/mean_grad/Maximum_1/yConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
value	B :*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
dtype0
�
;gradients/batch_normalization_4/moments/mean_grad/Maximum_1Maximum8gradients/batch_normalization_4/moments/mean_grad/Prod_1=gradients/batch_normalization_4/moments/mean_grad/Maximum_1/y*
_output_shapes
: *
T0*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2
�
<gradients/batch_normalization_4/moments/mean_grad/floordiv_1FloorDiv6gradients/batch_normalization_4/moments/mean_grad/Prod;gradients/batch_normalization_4/moments/mean_grad/Maximum_1*
T0*L
_classB
@>loc:@gradients/batch_normalization_4/moments/mean_grad/Shape_2*
_output_shapes
: 
�
6gradients/batch_normalization_4/moments/mean_grad/CastCast<gradients/batch_normalization_4/moments/mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
9gradients/batch_normalization_4/moments/mean_grad/truedivRealDiv6gradients/batch_normalization_4/moments/mean_grad/Tile6gradients/batch_normalization_4/moments/mean_grad/Cast*
T0*'
_output_shapes
:���������d
�
gradients/AddN_2AddNMgradients/batch_normalization_4/batchnorm/mul_1_grad/tuple/control_dependencyWgradients/batch_normalization_4/moments/SquaredDifference_grad/tuple/control_dependency9gradients/batch_normalization_4/moments/mean_grad/truediv*
T0*O
_classE
CAloc:@gradients/batch_normalization_4/batchnorm/mul_1_grad/Reshape*
N*'
_output_shapes
:���������d
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/AddN_2dense/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
$gradients/dense/MatMul_grad/MatMul_1MatMulReshapegradients/AddN_2*
_output_shapes
:	�d*
transpose_a(*
transpose_b( *
T0
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
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
�
gradients/Reshape_grad/ShapeShapeRelu_2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
_output_shapes
:
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
�
9gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitchgradients/Relu_2_grad/ReluGrad"batch_normalization_3/cond/pred_id*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*J
_output_shapes8
6:���������:���������*
T0
�
@gradients/batch_normalization_3/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1:^gradients/batch_normalization_3/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_3/cond/Merge_grad/cond_gradA^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
Jgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad
�
gradients/zeros_like	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_1	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_2	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_3	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1N^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
gradients/zeros_like_4	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_5	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_6	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_7	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_12batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:1+batch_normalization_3/cond/FusedBatchNorm:3+batch_normalization_3/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:
�
Igradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1L^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������*
T0
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
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
gradients/SwitchSwitchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    
{
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*/
_output_shapes
:���������
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_2Shapegradients/Switch_1:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_1/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
_output_shapes
:*
T0
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_2/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_3Switchconv2d_3/Conv2D"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
c
gradients/Shape_4Shapegradients/Switch_3*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_3/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_3*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_4Switch batch_normalization_2/gamma/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
c
gradients/Shape_5Shapegradients/Switch_4*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_4/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_4*
_output_shapes

:: *
T0*
N
�
gradients/Switch_5Switchbatch_normalization_2/beta/read"batch_normalization_3/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
c
gradients/Shape_6Shapegradients/Switch_5*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_5/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_5*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_3AddNKgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_grad*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������*
T0
�
%gradients/conv2d_3/Conv2D_grad/ShapeNShapeNRelu_1conv2d_2/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_3/Conv2D_grad/ShapeNconv2d_2/kernel/readgradients/AddN_3*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_1'gradients/conv2d_3/Conv2D_grad/ShapeN:1gradients/AddN_3*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
/gradients/conv2d_3/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_13^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_3/Conv2D_grad/Conv2DBackpropFilter
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
gradients/AddN_4AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N
�
gradients/AddN_5AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
gradients/Relu_1_grad/ReluGradReluGrad7gradients/conv2d_3/Conv2D_grad/tuple/control_dependencyRelu_1*/
_output_shapes
:���������*
T0
�
9gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitchgradients/Relu_1_grad/ReluGrad"batch_normalization_2/cond/pred_id*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*J
_output_shapes8
6:���������:���������
�
@gradients/batch_normalization_2/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1:^gradients/batch_normalization_2/cond/Merge_grad/cond_grad
�
Hgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_2/cond/Merge_grad/cond_gradA^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad*/
_output_shapes
:���������
�
Jgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
�
gradients/zeros_like_8	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_9	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_10	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_11	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1N^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
gradients/zeros_like_12	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_13	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_14	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_15	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_12batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:1+batch_normalization_2/cond/FusedBatchNorm:3+batch_normalization_2/cond/FusedBatchNorm:4*
epsilon%o�:*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(
�
Igradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1L^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������
�
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
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
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
gradients/Switch_6Switchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
e
gradients/Shape_7Shapegradients/Switch_6:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_6/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*/
_output_shapes
:���������*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*1
_output_shapes
:���������: *
T0*
N
�
gradients/Switch_7Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_8Shapegradients/Switch_7:1*
out_type0*
_output_shapes
:*
T0
�
gradients/zeros_7/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_8Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_9Shapegradients/Switch_8:1*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_8/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
j
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*
_output_shapes
:
�
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_9Switchconv2d_2/Conv2D"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*J
_output_shapes8
6:���������:���������*
T0
d
gradients/Shape_10Shapegradients/Switch_9*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_9/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_gradMergeQgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_9*
T0*
N*1
_output_shapes
:���������: 
�
gradients/Switch_10Switch batch_normalization_1/gamma/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_11Shapegradients/Switch_10*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_10/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_10*
N*
_output_shapes

:: *
T0
�
gradients/Switch_11Switchbatch_normalization_1/beta/read"batch_normalization_2/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_12Shapegradients/Switch_11*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_11/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_11*
T0*
N*
_output_shapes

:: 
�
gradients/AddN_6AddNKgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*/
_output_shapes
:���������*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_grad
�
%gradients/conv2d_2/Conv2D_grad/ShapeNShapeNReluconv2d_1/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
N* 
_output_shapes
::
�
2gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_2/Conv2D_grad/ShapeNconv2d_1/kernel/readgradients/AddN_6*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
3gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu'gradients/conv2d_2/Conv2D_grad/ShapeN:1gradients/AddN_6*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
/gradients/conv2d_2/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_13^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_2/Conv2D_grad/Conv2DBackpropFilter
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
gradients/AddN_7AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
gradients/AddN_8AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
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
�
>gradients/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_18^gradients/batch_normalization/cond/Merge_grad/cond_grad
�
Fgradients/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentity7gradients/batch_normalization/cond/Merge_grad/cond_grad?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:���������
�
Hgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_1Identity9gradients/batch_normalization/cond/Merge_grad/cond_grad:1?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:���������
�
gradients/zeros_like_16	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_17	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_18	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_19	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradFgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:*
T0*
data_formatNHWC
�
Igradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1L^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
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
�
gradients/zeros_like_20	ZerosLike)batch_normalization/cond/FusedBatchNorm:1$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
gradients/zeros_like_21	ZerosLike)batch_normalization/cond/FusedBatchNorm:2$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_22	ZerosLike)batch_normalization/cond/FusedBatchNorm:3$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
_output_shapes
:
�
gradients/zeros_like_23	ZerosLike)batch_normalization/cond/FusedBatchNorm:4$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
:*
T0
�
Igradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_10batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:1)batch_normalization/cond/FusedBatchNorm:3)batch_normalization/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*C
_output_shapes1
/:���������::: : *
is_training(*
epsilon%o�:
�
Ggradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1J^gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Ogradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityIgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradH^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
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
�
gradients/Switch_12Switchconv2d/Conv2D batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
g
gradients/Shape_13Shapegradients/Switch_12:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_12/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*/
_output_shapes
:���������
�
Igradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*1
_output_shapes
:���������: *
T0*
N
�
gradients/Switch_13Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
g
gradients/Shape_14Shapegradients/Switch_13:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_13/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
N*
_output_shapes

:: *
T0
�
gradients/Switch_14Switchbatch_normalization/beta/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
g
gradients/Shape_15Shapegradients/Switch_14:1*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_14/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
_output_shapes
:*
T0
�
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
T0*
N*
_output_shapes

:: 
�
gradients/Switch_15Switchconv2d/Conv2D batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*J
_output_shapes8
6:���������:���������
e
gradients/Shape_16Shapegradients/Switch_15*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_15/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*
T0*/
_output_shapes
:���������
�
Ggradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeOgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencygradients/zeros_15*
N*1
_output_shapes
:���������: *
T0
�
gradients/Switch_16Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1* 
_output_shapes
::*
T0
e
gradients/Shape_17Shapegradients/Switch_16*
_output_shapes
:*
T0*
out_type0
�
gradients/zeros_16/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
valueB
 *    *
dtype0*
_output_shapes
: 
m
gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
_output_shapes
:*
T0
�
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1gradients/zeros_16*
N*
_output_shapes

:: *
T0
�
gradients/Switch_17Switchbatch_normalization/beta/read batch_normalization/cond/pred_id$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0* 
_output_shapes
::
e
gradients/Shape_18Shapegradients/Switch_17*
T0*
out_type0*
_output_shapes
:
�
gradients/zeros_17/ConstConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
_output_shapes
: *
valueB
 *    *
dtype0
m
gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
_output_shapes
:*
T0
�
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2gradients/zeros_17*
N*
_output_shapes

:: *
T0
�
gradients/AddN_9AddNIgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradGgradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������
�
#gradients/conv2d/Conv2D_grad/ShapeNShapeNPlaceholderconv2d/kernel/read$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
T0*
out_type0*
N* 
_output_shapes
::
�
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/readgradients/AddN_9*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder%gradients/conv2d/Conv2D_grad/ShapeN:1gradients/AddN_9*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_11^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
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
�
gradients/AddN_10AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:*
T0
�
gradients/AddN_11AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
GradientDescent/learning_rateConst$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1*
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
EGradientDescent/update_batch_normalization/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization/gammaGradientDescent/learning_rategradients/AddN_10*
use_locking( *
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:
�
DGradientDescent/update_batch_normalization/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization/betaGradientDescent/learning_rategradients/AddN_11*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:*
use_locking( 
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate9gradients/conv2d_2/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel
�
GGradientDescent/update_batch_normalization_1/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_1/gammaGradientDescent/learning_rategradients/AddN_7*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
:
�
FGradientDescent/update_batch_normalization_1/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_1/betaGradientDescent/learning_rategradients/AddN_8*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:*
use_locking( *
T0
�
;GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentApplyGradientDescentconv2d_2/kernelGradientDescent/learning_rate9gradients/conv2d_3/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
GGradientDescent/update_batch_normalization_2/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_2/gammaGradientDescent/learning_rategradients/AddN_4*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:*
use_locking( *
T0
�
FGradientDescent/update_batch_normalization_2/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_2/betaGradientDescent/learning_rategradients/AddN_5*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:*
use_locking( *
T0
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel*
_output_shapes
:	�d*
use_locking( *
T0
�
GGradientDescent/update_batch_normalization_3/gamma/ApplyGradientDescentApplyGradientDescentbatch_normalization_3/gammaGradientDescent/learning_rateMgradients/batch_normalization_4/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:d
�
FGradientDescent/update_batch_normalization_3/beta/ApplyGradientDescentApplyGradientDescentbatch_normalization_3/betaGradientDescent/learning_rateKgradients/batch_normalization_4/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_3/beta*
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
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

�

GradientDescentNoOp$^batch_normalization/AssignMovingAvg&^batch_normalization/AssignMovingAvg_1&^batch_normalization_2/AssignMovingAvg(^batch_normalization_2/AssignMovingAvg_1&^batch_normalization_3/AssignMovingAvg(^batch_normalization_3/AssignMovingAvg_1&^batch_normalization_4/AssignMovingAvg(^batch_normalization_4/AssignMovingAvg_1:^GradientDescent/update_conv2d/kernel/ApplyGradientDescentF^GradientDescent/update_batch_normalization/gamma/ApplyGradientDescentE^GradientDescent/update_batch_normalization/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_1/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_1/beta/ApplyGradientDescent<^GradientDescent/update_conv2d_2/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_2/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_2/beta/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescentH^GradientDescent/update_batch_normalization_3/gamma/ApplyGradientDescentG^GradientDescent/update_batch_normalization_3/beta/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent
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
: *

Tidx0*
	keep_dims( 
N
Merge/MergeSummaryMergeSummary	conv_loss*
N*
_output_shapes
: ""
	summaries

conv_loss:0"�
trainable_variables��
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
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02.batch_normalization_3/gamma/Initializer/ones:0
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02.batch_normalization_3/beta/Initializer/zeros:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"�
	variables��
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
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02.batch_normalization_3/gamma/Initializer/ones:0
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02.batch_normalization_3/beta/Initializer/zeros:0
�
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign(batch_normalization_3/moving_mean/read:025batch_normalization_3/moving_mean/Initializer/zeros:0
�
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign,batch_normalization_3/moving_variance/read:028batch_normalization_3/moving_variance/Initializer/ones:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0"�6
cond_context�6�6
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
conv2d/Conv2D:0b
*batch_normalization/moving_variance/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_4:0^
&batch_normalization/moving_mean/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_3:0X
 batch_normalization/gamma/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_1:0W
batch_normalization/beta/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_2:0E
conv2d/Conv2D:02batch_normalization/cond/FusedBatchNorm_1/Switch:0
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
conv2d_3/Conv2D:0I
conv2d_3/Conv2D:04batch_normalization_3/cond/FusedBatchNorm_1/Switch:0\
"batch_normalization_2/gamma/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_1:0b
(batch_normalization_2/moving_mean/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_3:0f
,batch_normalization_2/moving_variance/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_4:0[
!batch_normalization_2/beta/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_2:0"
train_op

GradientDescent"�

update_ops�
�
%batch_normalization/AssignMovingAvg:0
'batch_normalization/AssignMovingAvg_1:0
'batch_normalization_2/AssignMovingAvg:0
)batch_normalization_2/AssignMovingAvg_1:0
'batch_normalization_3/AssignMovingAvg:0
)batch_normalization_3/AssignMovingAvg_1:0
'batch_normalization_4/AssignMovingAvg:0
)batch_normalization_4/AssignMovingAvg_1:0�nK       `/�#	P���A*

	conv_lossdOP?�"�       QKD	�"���A*

	conv_loss�.J?}�z       QKD	mX���A*

	conv_lossc)N?��`�       QKD	����A*

	conv_loss�kH?��V       QKD	s����A*

	conv_loss�O?#�z       QKD	e���A*

	conv_loss�`L?��-       QKD	�7���A*

	conv_loss�gM?;1.k       QKD	�t���A*

	conv_loss�K?�`F�       QKD	����A*

	conv_loss��M?Y �p       QKD	!����A	*

	conv_loss�-K?]dص       QKD	  ����A
*

	conv_lossc�K?V6�E       QKD	H^ ����A*

	conv_loss#�I?:\�3       QKD	� ����A*

	conv_lossg�G?drn.       QKD	L� ����A*

	conv_lossxFJ?�r�n       QKD	�����A*

	conv_loss�K?!XŃ       QKD	I����A*

	conv_loss��K?�yO�       QKD	�����A*

	conv_loss��H?�LU       QKD	7�����A*

	conv_lossn�H?���       QKD	n�����A*

	conv_loss#�C?�;��       QKD	b!����A*

	conv_loss�7J?�1�       QKD	�V����A*

	conv_loss�OJ?۳��       QKD	ˋ����A*

	conv_lossI?��(       QKD	 �����A*

	conv_lossB�J?�O8       QKD	������A*

	conv_lossJ?f��       QKD	,����A*

	conv_loss~�J?�-'       QKD	�b����A*

	conv_loss��I?��       QKD	-�����A*

	conv_loss��G?�|�:       QKD	������A*

	conv_loss"B?5�_       QKD	�����A*

	conv_loss�J?6+       QKD	
<����A*

	conv_losszB?e��       QKD	�p����A*

	conv_loss�F?q8&       QKD	������A*

	conv_lossdF?G�X�       QKD	\�����A *

	conv_loss��I?���       QKD	�����A!*

	conv_lossxE?��Đ       QKD	�[����A"*

	conv_loss~�F?0n.9       QKD	A�����A#*

	conv_loss�5K?[�u�       QKD	������A$*

	conv_lossH?�E�       QKD	�����A%*

	conv_lossoD?���       QKD	�3����A&*

	conv_loss�1A?�s       QKD	j����A'*

	conv_loss<,G?̞��       QKD	�����A(*

	conv_loss[F?����       QKD	 �����A)*

	conv_lossK�H?dAGn       QKD	����A**

	conv_lossLD?θ��       QKD	�9����A+*

	conv_loss��E?�C�       QKD	9l����A,*

	conv_loss,wC?#�.       QKD	]�����A-*

	conv_lossKD?AC��       QKD	������A.*

	conv_lossL�F?����       QKD	K����A/*

	conv_loss#6C?����       QKD	_5����A0*

	conv_loss�u>?�U�&       QKD	Ag����A1*

	conv_lossg�B?PzC�       QKD	{�����A2*

	conv_loss�B?5�7�       QKD	������A3*

	conv_loss�,C?�t�<       QKD	T	����A4*

	conv_lossB�D?�܃h       QKD	�@	����A5*

	conv_lossL�??ʴ��       QKD	��	����A6*

	conv_loss�[??��y       QKD	��	����A7*

	conv_loss�>?��'1       QKD	��	����A8*

	conv_loss.�A?d>3�       QKD	&
����A9*

	conv_loss��>?��V       QKD	/Z
����A:*

	conv_lossZ@?}-g       QKD	��
����A;*

	conv_loss�@?dI��       QKD	 �
����A<*

	conv_loss,�A?1�[       QKD	����A=*

	conv_lossÇ@?�l�       QKD	G����A>*

	conv_loss�??��1�       QKD	�z����A?*

	conv_loss��A?w��o       QKD	I�����A@*

	conv_loss�7>?� �       QKD	_�����AA*

	conv_loss! D?�Iw�       QKD	�����AB*

	conv_lossޡ=?3�0�       QKD	I����AC*

	conv_loss�M9?�"�        QKD	<|����AD*

	conv_lossŶ<?y3�~       QKD	������AE*

	conv_loss~�;?�i�       QKD	������AF*

	conv_loss,�>?F�W�       QKD	����AG*

	conv_lossF�??C�>�       QKD	VJ����AH*

	conv_loss6�<?�/       QKD	~����AI*

	conv_loss	>A?�.�o       QKD	������AJ*

	conv_lossO<?��B�       QKD	������AK*

	conv_loss<?jқH       QKD	�����AL*

	conv_loss�>?'BJ�       QKD	�J����AM*

	conv_lossϚ<??b��       QKD	�}����AN*

	conv_lossف;?{[7       QKD	f�����AO*

	conv_loss�;?Ig]       QKD	�����AP*

	conv_loss��=?HN�       QKD	'����AQ*

	conv_loss�:?�F�       QKD	G����AR*

	conv_loss�=?�v�:       QKD	�x����AS*

	conv_loss��:?�r       QKD	-�����AT*

	conv_lossRh;?�I�4       QKD	������AU*

	conv_loss4�<?�!V       QKD	�����AV*

	conv_loss�/7?�ea�       QKD	�F����AW*

	conv_loss��9?ƾ\o       QKD	�x����AX*

	conv_lossj<?3R��       QKD	ƫ����AY*

	conv_loss/E<?s���       QKD	K�����AZ*

	conv_loss3I9?V�HC       QKD	C����A[*

	conv_loss Z:?h�=4       QKD	DG����A\*

	conv_lossu�9?��<+       QKD	�y����A]*

	conv_lossi�;?��2�       QKD	4�����A^*

	conv_loss�19?A�S       QKD	������A_*

	conv_loss��8?�=�       QKD	�
����A`*

	conv_lossݕ8?�Ę6       QKD	:����Aa*

	conv_loss�:?6���       QKD	�j����Ab*

	conv_loss��9?P�a       QKD	Κ����Ac*

	conv_loss�9?�U\�       QKD	������Ad*

	conv_lossm�9?�I�       QKD	�����Ae*

	conv_loss)�;?ob�       QKD	�@����Af*

	conv_loss��7?c#�       QKD	?p����Ag*

	conv_loss�?5?7)�:       QKD	6�����Ah*

	conv_loss4;?�y       QKD	������Ai*

	conv_loss�*8?`�       QKD	L����Aj*

	conv_losst(7?����       QKD	�C����Ak*

	conv_loss�'9?
D�       QKD	�t����Al*

	conv_loss�E7?�+�       QKD	C�����Am*

	conv_lossݦ2?��       QKD	z�����An*

	conv_loss)�4?���
       QKD	�%����Ao*

	conv_loss�H7?u�O�       QKD	[����Ap*

	conv_loss��9?mpp�       QKD	�����Aq*

	conv_loss�5?�%U�       QKD	*�����Ar*

	conv_lossyE9?����       QKD	=�����As*

	conv_loss9�5?� =�       QKD	�)����At*

	conv_loss�V5?�M:$       QKD	�j����Au*

	conv_loss]�4?M@��       QKD	{�����Av*

	conv_loss1�4? ��       QKD	B�����Aw*

	conv_loss9f6?w
�U       QKD	������Ax*

	conv_loss��8?��       QKD	{.����Ay*

	conv_lossj�6?�O��       QKD	`����Az*

	conv_loss,�8?ǅ0(       QKD	]�����A{*

	conv_loss��5?Ү�       QKD	T�����A|*

	conv_loss�8?5�\=       QKD	������A}*

	conv_loss~k7?kM/)       QKD	e"����A~*

	conv_lossY�/?��~       QKD	�T����A*

	conv_loss>�5?�f`        )��P	�����A�*

	conv_loss�?1?�p�        )��P	������A�*

	conv_loss��6?���        )��P	������A�*

	conv_loss=�1?��-        )��P	S����A�*

	conv_lossAF4?��V*        )��P	2H����A�*

	conv_loss�O3?(�d        )��P	Kx����A�*

	conv_loss�0?�|-z        )��P	������A�*

	conv_loss3?�҉�        )��P	������A�*

	conv_lossc�2?{'K        )��P	`����A�*

	conv_loss�1?��r        )��P	C<����A�*

	conv_loss��0?d�=E        )��P	l����A�*

	conv_loss��0?��M        )��P	������A�*

	conv_lossgJ3?<�Ό        )��P	������A�*

	conv_losscm4?kB�e        )��P	������A�*

	conv_loss��3?<�%        )��P	/����A�*

	conv_loss8"3?��3?        )��P	F_����A�*

	conv_loss{3?64/M        )��P	������A�*

	conv_lossz_.?R� �        )��P	������A�*

	conv_loss�	4?y�l�        )��P	������A�*

	conv_loss��3?�t�        )��P	�$����A�*

	conv_loss�#/?s��a        )��P	�U����A�*

	conv_loss�r.?����        )��P	d�����A�*

	conv_loss��/?s~��        )��P	I�����A�*

	conv_loss��2?���        )��P	������A�*

	conv_loss�r.? ��9        )��P	<%����A�*

	conv_loss��/?W�
�        )��P	FT����A�*

	conv_loss��+?�h��        )��P	1�����A�*

	conv_loss�0?�_Z        )��P	T�����A�*

	conv_loss*�+?O��T        )��P	_�����A�*

	conv_loss��-?�PU        )��P	�����A�*

	conv_loss��.?�@5        )��P	XP����A�*

	conv_loss��0?+L�|        )��P	U�����A�*

	conv_loss��0?P�I        )��P	�����A�*

	conv_lossɏ,?۪L�        )��P	�����A�*

	conv_loss<�.?ս=V        )��P	�C����A�*

	conv_loss:,?��ɬ        )��P	Cv����A�*

	conv_loss��-?"��        )��P	�����A�*

	conv_loss�b*?n!{        )��P	�����A�*

	conv_loss��-?[���        )��P	S ����A�*

	conv_loss��.?��q        )��P	m5 ����A�*

	conv_loss��)?d�@�        )��P	He ����A�*

	conv_loss�,?֮5w        )��P	.� ����A�*

	conv_loss�+?I��s        )��P	:� ����A�*

	conv_loss��,?z6��        )��P	�� ����A�*

	conv_loss�0?��U        )��P	1$!����A�*

	conv_loss��,?^��        )��P	�S!����A�*

	conv_loss��*?�ú        )��P	�!����A�*

	conv_loss�.?�&��        )��P	�!����A�*

	conv_loss��'?;���        )��P	s�!����A�*

	conv_loss��*?�B��        )��P	f"����A�*

	conv_losse�,?�lQ        )��P	5D"����A�*

	conv_lossd�0?�4�        )��P	t"����A�*

	conv_loss��+?�O        )��P	��"����A�*

	conv_loss�t-?��        )��P	��"����A�*

	conv_loss��(?�V�        )��P	l#����A�*

	conv_loss��*?#)6�        )��P	=1#����A�*

	conv_loss34*?)~��        )��P	fa#����A�*

	conv_loss�],?�%��        )��P	��#����A�*

	conv_loss�t*?~�wH        )��P	��#����A�*

	conv_loss�z*?�h�        )��P	�#����A�*

	conv_loss�/)?1�3�        )��P	�$����A�*

	conv_lossp+?t��        )��P	�L$����A�*

	conv_lossjb'?��P4        )��P	:|$����A�*

	conv_lossx�'?��N�        )��P	ƫ$����A�*

	conv_loss�p&?�2�8        )��P	��$����A�*

	conv_loss�3&?Qo5        )��P	�%����A�*

	conv_loss�*?��8�        )��P	�<%����A�*

	conv_loss�7,?X���        )��P	Il%����A�*

	conv_lossX%?��s�        )��P	��%����A�*

	conv_loss�U,?��c�        )��P	7�%����A�*

	conv_lossl�%?3)�        )��P	�%����A�*

	conv_loss�$?���d        )��P	�*&����A�*

	conv_loss�S$?�~        )��P	�[&����A�*

	conv_loss��&?3�Y        )��P	��'����A�*

	conv_loss�7%?��o�        )��P	�(����A�*

	conv_loss��"?3Z��        )��P	�H(����A�*

	conv_lossv'?ʘK)        )��P	.{(����A�*

	conv_loss�z%?괒~        )��P	E�(����A�*

	conv_lossd&'?��"h        )��P	��(����A�*

	conv_loss�'?m�3�        )��P	�)����A�*

	conv_loss�p(?mX��        )��P	�T)����A�*

	conv_loss��%?$yh�        )��P	`�)����A�*

	conv_lossD�&?�|Ux        )��P	.�)����A�*

	conv_loss�#?�?$�        )��P	��)����A�*

	conv_lossm$?�M        )��P	@,*����A�*

	conv_loss.�%?�H��        )��P	L^*����A�*

	conv_lossަ&?)�        )��P	1�*����A�*

	conv_loss�!?(K�6        )��P	�*����A�*

	conv_loss�(?�ل�        )��P	��*����A�*

	conv_loss($?���        )��P	�'+����A�*

	conv_loss6�"?&�-�        )��P	eX+����A�*

	conv_loss��&?Ү��        )��P	G�+����A�*

	conv_lossp|'?=�V*        )��P	P�+����A�*

	conv_loss�q$?���U        )��P	��+����A�*

	conv_loss��?	���        )��P	�+,����A�*

	conv_loss��!?����        )��P	s\,����A�*

	conv_loss,&?B0��        )��P	y�,����A�*

	conv_loss��$?����        )��P	��,����A�*

	conv_loss	D$?����        )��P	7�,����A�*

	conv_loss >(?���d        )��P	�-����A�*

	conv_loss9�?��Q        )��P	�M-����A�*

	conv_loss�' ?��r;        )��P	�}-����A�*

	conv_loss4Q"?vC[        )��P	ͭ-����A�*

	conv_loss��"?�e9�        )��P	��-����A�*

	conv_loss/�#?N=c        )��P	�.����A�*

	conv_loss�N ?^���        )��P	�>.����A�*

	conv_loss�t"?U]        )��P	Jl.����A�*

	conv_loss܂!?����        )��P	ۜ.����A�*

	conv_loss��&?�j�        )��P	T�.����A�*

	conv_loss�2#?<�~�        )��P	��.����A�*

	conv_lossJ>"?�M<�        )��P	�,/����A�*

	conv_loss�?�Q��        )��P	\/����A�*

	conv_loss��"?Y���        )��P	�/����A�*

	conv_loss��!?+��        )��P	r�/����A�*

	conv_loss��#?��*        )��P	Q�/����A�*

	conv_loss��!?)�37        )��P	�0����A�*

	conv_loss^�"?^av�        )��P	�M0����A�*

	conv_loss��"?x�n8        )��P	��0����A�*

	conv_lossƣ"?��*        )��P	��0����A�*

	conv_loss��!?=E�        )��P	'1����A�*

	conv_losspp"?��ö        )��P	^21����A�*

	conv_loss%�!?�        )��P	�c1����A�*

	conv_loss�4!?^LV        )��P	��1����A�*

	conv_loss��?x���        )��P	��1����A�*

	conv_loss��?U{]�        )��P	�2����A�*

	conv_lossW�?�KC        )��P	�52����A�*

	conv_loss�?��NW        )��P	Nd2����A�*

	conv_loss��?��y�        )��P	��2����A�*

	conv_loss3�?���v        )��P	��2����A�*

	conv_lossY
 ?��_E        )��P	�
3����A�*

	conv_loss.?E���        )��P	�B3����A�*

	conv_loss~[?�Y��        )��P	�{3����A�*

	conv_lossPh?��yp        )��P	�3����A�*

	conv_loss�d? E��        )��P	8�3����A�*

	conv_loss�("?��}        )��P	X4����A�*

	conv_lossC?���        )��P	�E4����A�*

	conv_lossԷ?LӒ�        )��P	�u4����A�*

	conv_loss�?����        )��P	��4����A�*

	conv_loss�?u��        )��P	��4����A�*

	conv_losse�?�Y�[        )��P	 5����A�*

	conv_loss�]?�a�;        )��P	85����A�*

	conv_losss�?�em+        )��P	h5����A�*

	conv_lossB3?��{L        )��P	1�5����A�*

	conv_lossmp?b�|        )��P	��5����A�*

	conv_lossg?0?�        )��P	�6����A�*

	conv_loss:3?�K��        )��P	�86����A�*

	conv_loss�#?�N�        )��P	�g6����A�*

	conv_lossF^?�،t        )��P	�6����A�*

	conv_lossk)?���(        )��P	v�6����A�*

	conv_lossa?����        )��P	��6����A�*

	conv_loss�o? k        )��P	n)7����A�*

	conv_loss��?i��p        )��P	�Y7����A�*

	conv_lossD?	�v�        )��P	�7����A�*

	conv_lossT??_g��        )��P	�7����A�*

	conv_lossH?{ZP        )��P	��7����A�*

	conv_loss��?�e��        )��P	�8����A�*

	conv_loss;?�{�
        )��P	�G8����A�*

	conv_loss?Q�}�        )��P	�v8����A�*

	conv_lossy�?I�	�        )��P	�8����A�*

	conv_loss��?X���        )��P	��8����A�*

	conv_loss��?�!�        )��P	@9����A�*

	conv_loss]�?��+        )��P	69����A�*

	conv_loss�i??4O�        )��P	�d9����A�*

	conv_losss?�|        )��P	�9����A�*

	conv_loss��?��n        )��P	�9����A�*

	conv_loss�G?�@?�        )��P	��9����A�*

	conv_loss#?@�h        )��P	�#:����A�*

	conv_loss�?H_Y�        )��P	1S:����A�*

	conv_lossX�?��'�        )��P	�:����A�*

	conv_loss��?R�H�        )��P	��:����A�*

	conv_loss4�?�NQ�        )��P	�:����A�*

	conv_loss�c?��        )��P	�;����A�*

	conv_loss�x?���o        )��P	�D;����A�*

	conv_loss
/?��W        )��P	��;����A�*

	conv_loss��?4���        )��P	Z�;����A�*

	conv_loss$?�PT�        )��P	=�;����A�*

	conv_loss�o?8?'�        )��P	�<����A�*

	conv_loss�??�hH        )��P	�H<����A�*

	conv_loss3
?R=�9        )��P	�z<����A�*

	conv_loss%?+�V�        )��P	��<����A�*

	conv_loss�:?c��        )��P	��<����A�*

	conv_loss{A?����        )��P	�=����A�*

	conv_lossK?�{��        )��P	|W=����A�*

	conv_loss��?�&�        )��P	�=����A�*

	conv_lossL?���b        )��P	!�=����A�*

	conv_loss��?㤢�        )��P	�=����A�*

	conv_lossi�?��>�        )��P	 #>����A�*

	conv_lossSG?E4�        )��P	�R>����A�*

	conv_losss�?��        )��P	:�>����A�*

	conv_loss%D?��P+        )��P	׳>����A�*

	conv_loss6�?�X5        )��P	��>����A�*

	conv_lossA�?���        )��P	#?����A�*

	conv_loss��?���:        )��P	E?����A�*

	conv_loss�x?���        )��P	�s?����A�*

	conv_loss�V?��CV        )��P	-�?����A�*

	conv_loss1%?��>        )��P	�?����A�*

	conv_loss��?G~q�        )��P	M@����A�*

	conv_losszI?l!�        )��P	O?@����A�*

	conv_loss~?�?V�        )��P	@o@����A�*

	conv_lossI�?��D        )��P	?�@����A�*

	conv_lossa?���?        )��P	��@����A�*

	conv_loss�?s���        )��P	��@����A�*

	conv_lossJ"?M���        )��P	�0A����A�*

	conv_lossX ?���V        )��P	�`A����A�*

	conv_loss_?��;        )��P	��A����A�*

	conv_lossڰ?h�^        )��P	�A����A�*

	conv_loss{?;��        )��P	��A����A�*

	conv_loss<<?@Q�        )��P	<B����A�*

	conv_loss~s?�( T        )��P	OB����A�*

	conv_loss��?
>.�        )��P	�B����A�*

	conv_loss.�?�ϸ%        )��P	߯B����A�*

	conv_loss<j?�,w        )��P	��B����A�*

	conv_loss��?­�;        )��P	KC����A�*

	conv_loss�?rŝ=        )��P	\BC����A�*

	conv_lossC�?V���        )��P	�qC����A�*

	conv_loss�?���1        )��P	9�C����A�*

	conv_loss�P?����        )��P	��C����A�*

	conv_losss�?�?.�        )��P	�D����A�*

	conv_loss�P?1�`�        )��P	=4D����A�*

	conv_loss��?��z�        )��P	
dD����A�*

	conv_loss?bN��        )��P	ܒD����A�*

	conv_lossB�?�tE        )��P	�D����A�*

	conv_lossX�?��        )��P	��D����A�*

	conv_loss��?��        )��P	F3E����A�*

	conv_loss��?�@        )��P	�bE����A�*

	conv_lossk
?�"0�        )��P	\�E����A�*

	conv_loss��?���        )��P	.�E����A�*

	conv_lossO�?�h�(        )��P	z�E����A�*

	conv_loss�z?�,��        )��P	~$F����A�*

	conv_loss��?}ϼ        )��P	?^F����A�*

	conv_loss��?|�F!        )��P	=�F����A�*

	conv_loss~|?m�x        )��P	�F����A�*

	conv_loss�?+φ
        )��P	-�F����A�*

	conv_loss[?WD��        )��P	w*G����A�*

	conv_loss�3?��ku        )��P	�oG����A�*

	conv_loss��?�        )��P	͠G����A�*

	conv_lossҪ?.�ٙ        )��P	:�G����A�*

	conv_lossÌ?@&�w        )��P	VH����A�*

	conv_loss�?o^�        )��P	�0H����A�*

	conv_loss3�?�x�        )��P	�aH����A�*

	conv_loss�?�]�        )��P	c�H����A�*

	conv_loss��?�g        )��P	-�H����A�*

	conv_loss~g?�!9�        )��P	��H����A�*

	conv_loss}�?�i�        )��P	� I����A�*

	conv_loss/6?	�qB        )��P	�PI����A�*

	conv_loss4�?���        )��P	��I����A�*

	conv_loss�F?T��Z        )��P	N�I����A�*

	conv_loss��? ��)        )��P	��I����A�*

	conv_lossq�?��Jh        )��P	�J����A�*

	conv_loss�C?]�t�        )��P	AJ����A�*

	conv_loss��?I�        )��P	�rJ����A�*

	conv_loss�;?�$(        )��P	�J����A�*

	conv_lossfS?�<�\        )��P	��J����A�*

	conv_loss��?S���        )��P	K����A�*

	conv_lossg	?�D        )��P	F3K����A�*

	conv_lossw?z.\        )��P	�aK����A�*

	conv_loss�?d�w        )��P		�K����A�*

	conv_lossx�?���        )��P	��K����A�*

	conv_loss<?�pf�        )��P	u�K����A�*

	conv_lossX;?���        )��P	�L����A�*

	conv_loss
??���        )��P	xOL����A�*

	conv_loss�S?�ü        )��P	�~L����A�*

	conv_loss�?�eI        )��P	1�L����A�*

	conv_lossNW?Ф3*        )��P	��L����A�*

	conv_loss��?�l-�        )��P	�M����A�*

	conv_lossȽ
?2�L        )��P	�?M����A�*

	conv_lossЬ?�G        )��P	$oM����A�*

	conv_loss<�?pŬ        )��P	��M����A�*

	conv_loss��?���        )��P	>�M����A�*

	conv_loss/�?<���        )��P	��M����A�*

	conv_loss-�?%���        )��P	1N����A�*

	conv_loss�?L���        )��P	aN����A�*

	conv_lossQ?@ö        )��P	�N����A�*

	conv_loss[?"1~#        )��P		.P����A�*

	conv_loss�
?N^��        )��P	?_P����A�*

	conv_loss�f?s��        )��P	]�P����A�*

	conv_losst�
?|4M�        )��P	��P����A�*

	conv_loss�X
?���u        )��P	"�P����A�*

	conv_loss<�?M��        )��P	#Q����A�*

	conv_loss�X?�� �        )��P	GVQ����A�*

	conv_loss�
?�@B�        )��P	a�Q����A�*

	conv_loss��?��        )��P	��Q����A�*

	conv_loss�~?͕��        )��P	AR����A�*

	conv_loss$�	?3��        )��P	�2R����A�*

	conv_loss��?t.�V        )��P	oaR����A�*

	conv_loss��
?���        )��P	��R����A�*

	conv_loss��	??���        )��P	��R����A�*

	conv_loss�e?�	V�        )��P	��R����A�*

	conv_loss3�?Ry�        )��P	�/S����A�*

	conv_loss�?�g�        )��P	-`S����A�*

	conv_loss��
?�dV�        )��P	��S����A�*

	conv_loss�	?�C~�        )��P	�S����A�*

	conv_loss��	?���        )��P	��S����A�*

	conv_loss��?�d��        )��P	7+T����A�*

	conv_loss��	?���V        )��P	\T����A�*

	conv_loss�C?�u�
        )��P	Z�T����A�*

	conv_loss��?F�=        )��P	��T����A�*

	conv_loss=�	?��/A        )��P	7U����A�*

	conv_loss��	?E�<�        )��P	!5U����A�*

	conv_loss�?��I        )��P	�gU����A�*

	conv_lossmb?:��        )��P	J�U����A�*

	conv_loss_U?�{�=        )��P	9�U����A�*

	conv_loss!�?44Y"        )��P	A�U����A�*

	conv_loss�
?>g��        )��P	�'V����A�*

	conv_loss!�?��        )��P	�XV����A�*

	conv_loss��?QY�\        )��P	��V����A�*

	conv_loss�<?�%Ť        )��P	��V����A�*

	conv_lossĔ?"�]�        )��P	(W����A�*

	conv_loss�?͇�A        )��P	S4W����A�*

	conv_loss�>?�QJ        )��P	�eW����A�*

	conv_loss-�?���R        )��P	4�W����A�*

	conv_loss��	?r}��        )��P	�W����A�*

	conv_loss�Y?��        )��P	A�W����A�*

	conv_loss��?����        )��P	M$X����A�*

	conv_lossݩ
?�~��        )��P	_TX����A�*

	conv_loss��?F9u;        )��P	o�X����A�*

	conv_lossJ�?~��        )��P	j�X����A�*

	conv_lossAD?T ��        )��P	#�X����A�*

	conv_loss�	?��^�        )��P	Y����A�*

	conv_lossC�?@io�        )��P	L?Y����A�*

	conv_loss��?_��X        )��P	JoY����A�*

	conv_loss�]?"t{�        )��P	ɞY����A�*

	conv_loss��?$��        )��P	S�Y����A�*

	conv_loss�
?b���        )��P	�Z����A�*

	conv_loss�=?�$Ɯ        )��P	>Z����A�*

	conv_loss!v?범�        )��P	XnZ����A�*

	conv_lossq�?�r�x        )��P	��Z����A�*

	conv_loss�u?�}'"        )��P	z�Z����A�*

	conv_loss��?*�F�        )��P	L[����A�*

	conv_loss�?[���        )��P	:@[����A�*

	conv_loss�-?�*�        )��P	Fp[����A�*

	conv_losss�?);��        )��P	p�[����A�*

	conv_loss��?K��        )��P	_\����A�*

	conv_loss�?1�        )��P	=\����A�*

	conv_loss]Z?���        )��P	Ln\����A�*

	conv_loss�r?Ɉ�        )��P	�\����A�*

	conv_loss��?�[s<        )��P	�\����A�*

	conv_loss��?^�[�        )��P	4�\����A�*

	conv_loss�J?�vB5        )��P	�,]����A�*

	conv_loss�?���4        )��P	�[]����A�*

	conv_loss��?x$r        )��P	T�]����A�*

	conv_loss*�?�Rm�        )��P	��]����A�*

	conv_lossD?x�X        )��P	]�]����A�*

	conv_lossEz?\R*P        )��P	2$^����A�*

	conv_loss��?9ָ        )��P	L\^����A�*

	conv_losse2?;rM�        )��P	�^����A�*

	conv_loss�?8h�        )��P	2�^����A�*

	conv_loss["?v;�        )��P	 �^����A�*

	conv_loss^)�>(�>K        )��P	�_����A�*

	conv_lossѕ?�K�|        )��P	.N_����A�*

	conv_lossX�?�	�        )��P	~_����A�*

	conv_loss�3?&A��        )��P	�_����A�*

	conv_loss�B?�i��        )��P	��_����A�*

	conv_loss	=?�-�p        )��P	�`����A�*

	conv_lossvL?dwW        )��P	/@`����A�*

	conv_lossK?i�I�        )��P	p`����A�*

	conv_loss��?d��        )��P	K�`����A�*

	conv_loss��>��b�        )��P	��`����A�*

	conv_lossX?��y        )��P	 a����A�*

	conv_loss�?W �        )��P	�0a����A�*

	conv_loss�� ?�-e�        )��P	i_a����A�*

	conv_loss�?�v��        )��P	ُa����A�*

	conv_loss>� ?��k�        )��P	m�a����A�*

	conv_lossud?�Oh7        )��P	��a����A�*

	conv_loss���>�]��        )��P	�b����A�*

	conv_loss[�?�i&�        )��P	�Nb����A�*

	conv_loss=a?W_W0        )��P	�}b����A�*

	conv_loss�o?� f        )��P		�b����A�*

	conv_loss:�?�^_y        )��P	��b����A�*

	conv_lossE?�d        )��P	1c����A�*

	conv_loss??C�Ih        )��P	B=c����A�*

	conv_loss��?��<        )��P	�mc����A�*

	conv_loss�(�>n��        )��P	��g����A�*

	conv_loss�j?�%=�        )��P	�h����A�*

	conv_loss �?�n�        )��P		Kh����A�*

	conv_loss��>	�A:        )��P	�yh����A�*

	conv_loss},�>��8L        )��P	�h����A�*

	conv_loss��?�W}�        )��P	|�h����A�*

	conv_loss��>1��        )��P	%i����A�*

	conv_loss�	�>�U        )��P	�?i����A�*

	conv_loss���>�s5        )��P	ni����A�*

	conv_loss�v ?�$�        )��P	¤i����A�*

	conv_loss�o?�r��        )��P	��i����A�*

	conv_loss��?p��;        )��P	Tj����A�*

	conv_loss�m?>�        )��P	a>j����A�*

	conv_losso��>v೔        )��P	nj����A�*

	conv_loss<?��        )��P	Ծj����A�*

	conv_lossN^�>���C        )��P	k����A�*

	conv_loss��?U�ׅ        )��P	K4k����A�*

	conv_loss�?��ؒ        )��P	�ck����A�*

	conv_loss�3 ?(ps�        )��P	j�k����A�*

	conv_lossm��>��2�        )��P	��k����A�*

	conv_losss1�>�-_N        )��P	��k����A�*

	conv_loss�;�>'�q�        )��P	�#l����A�*

	conv_loss-��>����        )��P	|Rl����A�*

	conv_loss(}�>�F3"        )��P	ˁl����A�*

	conv_loss
C�>�%        )��P	�l����A�*

	conv_lossU�>2��        )��P	;�l����A�*

	conv_loss�e�>�ϙ�        )��P	�m����A�*

	conv_loss���>)�        )��P	�@m����A�*

	conv_loss�?�En        )��P	�om����A�*

	conv_loss�A�>�A        )��P	��m����A�*

	conv_loss���>�ʿ�        )��P	K�m����A�*

	conv_loss/h ?�j�3        )��P	��m����A�*

	conv_loss'�>Az)%        )��P	�.n����A�*

	conv_loss�?��        )��P	B^n����A�*

	conv_loss�t�>F	�G        )��P	��n����A�*

	conv_lossG��>o��        )��P	�n����A�*

	conv_lossd��>��]�        )��P	f�n����A�*

	conv_loss��>�g�        )��P	�o����A�*

	conv_lossn��>�ck        )��P	�Ko����A�*

	conv_lossy��>F��W        )��P	�zo����A�*

	conv_loss(]�>�қb        )��P	��o����A�*

	conv_lossG0�>|�]�        )��P	G�o����A�*

	conv_lossj��>�o�E        )��P	Sp����A�*

	conv_loss�#�>h�;�        )��P	7p����A�*

	conv_loss�_�>Q)@�        )��P	ep����A�*

	conv_lossk��>Se<        )��P	b�p����A�*

	conv_loss~/�>��/�        )��P	T�p����A�*

	conv_loss��>)���        )��P	��p����A�*

	conv_lossņ�>�'��        )��P	� q����A�*

	conv_loss�C�>�p��        )��P	�Oq����A�*

	conv_loss���>�#6X        )��P	q����A�*

	conv_loss�x�>X��        )��P	��q����A�*

	conv_loss���>X�ɔ        )��P	��q����A�*

	conv_loss��>�,�        )��P	8r����A�*

	conv_loss5�>�
��        )��P	�Lr����A�*

	conv_loss��>��H        )��P	�r����A�*

	conv_loss�>z���        )��P	t�r����A�*

	conv_loss���>(�_         )��P	�r����A�*

	conv_loss(i�>In�_        )��P	�s����A�*

	conv_loss���>W(        )��P	3Hs����A�*

	conv_lossUc�>+���        )��P	�{s����A�*

	conv_lossͲ�>���        )��P	�s����A�*

	conv_loss���>�ˊ�        )��P	�s����A�*

	conv_loss���>[�        )��P	.t����A�*

	conv_loss���>k&=�        )��P	�@t����A�*

	conv_lossz!�>�ٟ1        )��P	"pt����A�*

	conv_loss/��>Gj�        )��P	��t����A�*

	conv_lossi��>�0�        )��P	0�t����A�*

	conv_loss��>�'�        )��P	�	u����A�*

	conv_loss�N�>@\Y�        )��P	=u����A�*

	conv_loss5��>�@�        )��P	Omu����A�*

	conv_loss4�>+�5        )��P	�u����A�*

	conv_loss%��>�?��        )��P	M�u����A�*

	conv_loss���>����        )��P	7v����A�*

	conv_lossp��>B%��        )��P	�=v����A�*

	conv_loss��>w'        )��P	]mv����A�*

	conv_loss١�>�b a        )��P	ɜv����A�*

	conv_loss�c�>�R#�        )��P	2�v����A�*

	conv_loss�,�>����        )��P	��v����A�*

	conv_loss���>b�n�        )��P	�*w����A�*

	conv_loss���>��b�        )��P	�[w����A�*

	conv_loss
�>�.^�        )��P	��w����A�*

	conv_lossw�>�rf�        )��P	s�w����A�*

	conv_loss<��>��(        )��P	M�w����A�*

	conv_loss�W�>�Aa%        )��P	�x����A�*

	conv_loss���>1I�        )��P	�Hx����A�*

	conv_loss���>�;Bc        )��P	�wx����A�*

	conv_loss"�>�f�        )��P	q�x����A�*

	conv_loss_h�>"(�        )��P	��x����A�*

	conv_loss�D�>���        )��P	�y����A�*

	conv_loss���>�sߦ        )��P	O4y����A�*

	conv_loss���>q���        )��P	�by����A�*

	conv_loss���>~vs        )��P	]�y����A�*

	conv_lossy��>,�k        )��P	��y����A�*

	conv_loss"��>�=        )��P	��y����A�*

	conv_loss���>.�1        )��P	�z����A�*

	conv_loss���>����        )��P	^Nz����A�*

	conv_loss�
�>!	��        )��P	z����A�*

	conv_lossZ��>Ca>M        )��P	��z����A�*

	conv_lossA��>,=�e        )��P	�z����A�*

	conv_loss� �>ú>�        )��P	{����A�*

	conv_loss�:�>�bPs        )��P	Q�|����A�*

	conv_loss϶�>�{        )��P	'�|����A�*

	conv_loss���>�\�+        )��P	��|����A�*

	conv_lossy!�>=��h        )��P	�6}����A�*

	conv_loss!!�>N�d        )��P	�d}����A�*

	conv_loss���>A�~        )��P	$�}����A�*

	conv_loss�Z�>+E��        )��P	��}����A�*

	conv_loss�[�>�/^�        )��P	��}����A�*

	conv_loss�H�>��݆        )��P	�8~����A�*

	conv_loss���>"��|        )��P	xg~����A�*

	conv_loss�	�>��)        )��P	;�~����A�*

	conv_loss���>8        )��P	��~����A�*

	conv_loss?��>�.�        )��P	�����A�*

	conv_loss��>F;��        )��P	E����A�*

	conv_loss�1�>��E         )��P	�t����A�*

	conv_loss�D�>b�M�        )��P	�����A�*

	conv_loss�x�>sw=>        )��P	������A�*

	conv_loss��>�F�n        )��P	������A�*

	conv_loss���>#H�        )��P	�3�����A�*

	conv_lossr�>c-S        )��P	Hc�����A�*

	conv_loss��>�o�j        )��P	k������A�*

	conv_loss�n�>�e�7        )��P	�������A�*

	conv_loss`�>y>g�        )��P	o������A�*

	conv_loss���>���        )��P	b�����A�*

	conv_loss,-�>ɜ�        )��P	�M�����A�*

	conv_lossؓ�>����        )��P	�|�����A�*

	conv_loss4��>��^�        )��P	7������A�*

	conv_loss@I�>��M�        )��P	�ہ����A�*

	conv_lossv#�>�=(�        )��P	 �����A�*

	conv_loss��>��|'        )��P	M;�����A�*

	conv_loss���>�ch�        )��P	i�����A�*

	conv_loss���>YP��        )��P	Q������A�*

	conv_loss$�>��?f        )��P	Ƃ����A�*

	conv_loss��><
��        )��P	5������A�*

	conv_loss%|�>R�hR        )��P	�$�����A�*

	conv_lossZ��>�M�        )��P	eS�����A�*

	conv_loss�>X;�c        )��P	[������A�*

	conv_loss��>�'o�        )��P	�������A�*

	conv_lossR��>�]��        )��P	!�����A�*

	conv_lossX��>�|#        )��P	������A�*

	conv_loss���>O.�n        )��P	XA�����A�*

	conv_lossY��>�[�n        )��P	�o�����A�*

	conv_loss;��>��$        )��P	"������A�*

	conv_loss)B�>]Ux�        )��P	�τ����A�*

	conv_loss���>x��m        )��P	x������A�*

	conv_loss3�>iyc4        )��P	�-�����A�*

	conv_loss���>k3�        )��P	�]�����A�*

	conv_loss��>��Nd        )��P	Y������A�*

	conv_loss��>B�=        )��P	������A�*

	conv_loss1=�>n�        )��P	������A�*

	conv_loss,	�>��:W        )��P	9.�����A�*

	conv_loss3�>H�W        )��P	k]�����A�*

	conv_loss��>c��\        )��P	�������A�*

	conv_loss�D�>C��}        )��P	������A�*

	conv_loss��>�~�F        )��P	������A�*

	conv_loss�U�>/TSx        )��P	h�����A�*

	conv_loss؛�>9��        )��P	NZ�����A�*

	conv_loss��>��EE        )��P	0������A�*

	conv_loss|��>��(�        )��P	�������A�*

	conv_loss)��>���"        )��P	F������A�*

	conv_lossy��>e�[        )��P	("�����A�*

	conv_loss���>�r;�        )��P	�P�����A�*

	conv_lossqp�>`�:        )��P	�����A�*

	conv_loss���>Qc(�        )��P	U������A�*

	conv_lossgc�>7���        )��P	t�����A�*

	conv_loss���>�+�e        )��P	A%�����A�*

	conv_loss<Y�>.�c�        )��P	�W�����A�*

	conv_loss/�>����        )��P	�������A�*

	conv_loss�g�>�m�]        )��P	�������A�*

	conv_loss�U�>9��        )��P	������A�*

	conv_lossO��>�T��        )��P	(�����A�*

	conv_loss��>�!�=        )��P	�D�����A�*

	conv_loss8D�>C�k        )��P	't�����A�*

	conv_loss�K�>�V��        )��P	�������A�*

	conv_loss@�>	tq�        )��P	 Ҋ����A�*

	conv_loss���>i�
H        )��P	������A�*

	conv_lossC��>X]�`        )��P	52�����A�*

	conv_loss?��>�F��        )��P	a�����A�*

	conv_loss���>pu        )��P	f������A�*

	conv_loss�Z�>���        )��P	ǿ�����A�*

	conv_lossW)�>7u�        )��P	������A�*

	conv_loss�r�>�"��        )��P	������A�*

	conv_loss���>*�$&        )��P	�P�����A�*

	conv_loss��>�k��        )��P	;�����A�*

	conv_loss���>Hmכ        )��P	?������A�*

	conv_loss�,�>Y�VY        )��P	�݌����A�*

	conv_lossӖ�>ޚ��        )��P	�����A�*

	conv_loss�<�>3���        )��P	�:�����A�*

	conv_loss��>�䡼        )��P	�i�����A�*

	conv_lossl?�>E�.        )��P	-������A�*

	conv_loss��>�-K        )��P	'ȍ����A�*

	conv_loss`s�>�        )��P	�������A�*

	conv_loss���>�f        )��P	&�����A�*

	conv_loss@%�>>O��        )��P	&T�����A�*

	conv_loss;�>�ʧ�        )��P	������A�*

	conv_loss���>�&h'        )��P	w������A�*

	conv_loss�m�>f¨        )��P	������A�*

	conv_loss�N�>q3��        )��P	������A�*

	conv_loss�Z�>ZB        )��P	(A�����A�*

	conv_loss���>�˗�        )��P	q�����A�*

	conv_loss_��>�u�        )��P	�������A�*

	conv_loss"
�>n�        )��P	Oݏ����A�*

	conv_lossv(�>t]�k        )��P	������A�*

	conv_loss\;�>y��        )��P	<�����A�*

	conv_loss�U�>��        )��P	fq�����A�*

	conv_lossp�>p�Y�        )��P	�������A�*

	conv_lossI��>�w+�        )��P	t������A�*

	conv_loss��>��\�        )��P	�%�����A�*

	conv_loss��>ic	        )��P	�Y�����A�*

	conv_loss��>��N        )��P	ĉ�����A�*

	conv_lossc
�>B�`h        )��P	�������A�*

	conv_loss5�>�3        )��P	i�����A�*

	conv_losshw�>굛�        )��P	� �����A�*

	conv_loss"��>=ˎ@        )��P	iO�����A�*

	conv_lossZ��>@r�*        )��P	������A�*

	conv_loss8��>I
o�        )��P	������A�*

	conv_loss�@�>U_��        )��P	k�����A�*

	conv_loss�A�>�ꟈ        )��P	������A�*

	conv_lossw�>��ܥ        )��P	nG�����A�*

	conv_loss���>wF�        )��P	�x�����A�*

	conv_loss3��>*�qP        )��P	�������A�*

	conv_loss���>H[e�        )��P	�ד����A�*

	conv_lossR1�>�CP�        )��P	������A�*

	conv_loss_V�>��ο        )��P	O6�����A�*

	conv_loss�
�>���        )��P	�d�����A�*

	conv_loss+�>��5        )��P	�������A�*

	conv_loss�G�>�0�        )��P	�Ô����A�*

	conv_loss'4�>�u��        )��P	������A�*

	conv_loss���>Jֻ        )��P	q!�����A�*

	conv_loss�/�>�{�!        )��P	�Q�����A�*

	conv_loss�X�>ӿ/�        )��P	�������A�*

	conv_loss)�>V�v�        )��P	w������A�*

	conv_loss�&�> ��        )��P	�ޕ����A�*

	conv_loss���>�U        )��P	-�����A�*

	conv_lossV�>M9�;        )��P	<�����A�*

	conv_loss��>uG�        )��P	�j�����A�*

	conv_loss�>z�?r        )��P	�����A�*

	conv_lossI��>s3"        )��P	�ɖ����A�*

	conv_lossOu�>��2<        )��P	6������A�*

	conv_lossv��>��        )��P	�(�����A�*

	conv_lossg'�>����        )��P	!Y�����A�*

	conv_loss��> �z        )��P	�������A�*

	conv_loss�N�>cAEM        )��P	9������A�*

	conv_loss���>x��        )��P	&�����A�*

	conv_loss#�>�)�        )��P	�����A�*

	conv_loss���>[J\|        )��P	�C�����A�*

	conv_loss���>��߀        )��P	�s�����A�*

	conv_loss��>X�>�        )��P	ҡ�����A�*

	conv_lossGP�>�s:        )��P	�И����A�*

	conv_loss�l�>�D_        )��P	 �����A�*

	conv_lossG�>�稒        )��P	>�����A�*

	conv_loss �>�{'        )��P	Rm�����A�*

	conv_loss���>�5        )��P	������A�*

	conv_lossw.�>/A�        )��P	�˙����A�*

	conv_loss�w�>��\7        )��P	� �����A�*

	conv_loss��>UϏ�        )��P	�0�����A�*

	conv_loss�>V'�        )��P	�c�����A�*

	conv_lossOx�>x��        )��P	s������A�*

	conv_loss�b�>o��"        )��P	�ۚ����A�*

	conv_loss���>���        )��P	������A�*

	conv_lossc��>�͗        )��P	HD�����A�*

	conv_loss���>��Y�        )��P	�r�����A�*

	conv_loss���>�S�        )��P	 ������A�*

	conv_loss%��>�c        )��P	�Л����A�*

	conv_loss�f�>?�i�        )��P	������A�*

	conv_loss�E�>��|�        )��P	W7�����A�*

	conv_loss�G�>��n�        )��P	kg�����A�*

	conv_losse~�>l��F        )��P	������A�*

	conv_lossWw�>!`^C        )��P	�Ɯ����A�*

	conv_loss��>U	g�        )��P	!������A�*

	conv_loss�n�>�%{Y        )��P	�4�����A�*

	conv_loss���>��4#        )��P	Ih�����A�*

	conv_loss��>�X�        )��P	�������A�*

	conv_loss���>!���        )��P	�ŝ����A�*

	conv_loss��>ݟ�o        )��P	7������A�*

	conv_loss���>���        )��P	�#�����A�*

	conv_lossB��>/?�:        )��P	(S�����A�*

	conv_lossk�>��P*        )��P	恞����A�*

	conv_loss�r�>�-1�        )��P	�������A�*

	conv_lossh��>y���        )��P	<�����A�*

	conv_lossՋ�>�T�        )��P	|�����A�*

	conv_loss=��>L?        )��P	q?�����A�*

	conv_lossTk�>R�D�        )��P	n�����A�*

	conv_loss��>�$�        )��P	㝟����A�*

	conv_loss���>m�%�        )��P	�͟����A�*

	conv_loss��>�eI�        )��P	������A�*

	conv_loss���>r%�        )��P	�,�����A�*

	conv_loss���>��`b        )��P	�[�����A�*

	conv_loss*�>`�6�        )��P	Љ�����A�*

	conv_lossef�>y?R�        )��P	x������A�*

	conv_loss��>�e�        )��P	������A�*

	conv_lossCl�>.҅X        )��P	������A�*

	conv_loss���>ஙQ        )��P	�C�����A�*

	conv_loss��>����        )��P	�r�����A�*

	conv_loss&.�><���        )��P	G������A�*

	conv_loss�S�>��ga        )��P	pϡ����A�*

	conv_lossˆ�>��7�        )��P	F������A�*

	conv_loss��>o�4�        )��P	�-�����A�*

	conv_lossV��>R��h        )��P	�\�����A�*

	conv_loss���>��x        )��P	1������A�*

	conv_losst��>����        )��P	�,�����A�*

	conv_lossǝ�>o`4        )��P	�\�����A�*

	conv_lossSS�>���>        )��P	/������A�*

	conv_loss�h�>���|        )��P	&ޤ����A�*

	conv_loss�b�>a��        )��P	O�����A�*

	conv_lossֿ�>�9�        )��P	�H�����A�*

	conv_loss�t�>7�H        )��P	zx�����A�*

	conv_loss�1�>2F�        )��P	�������A�*

	conv_loss��>�q�        )��P	�إ����A�*

	conv_loss���>7�oh        )��P	>�����A�*

	conv_lossw��>���(        )��P	�D�����A�*

	conv_loss���>�o�        )��P	�t�����A�*

	conv_loss���>,��        )��P	ܤ�����A�*

	conv_loss1��><؈�        )��P	�Ӧ����A�*

	conv_losss��>8ꋄ        )��P	������A�*

	conv_lossG*�>K�i=        )��P	�@�����A�*

	conv_loss6C�>v�        )��P	�q�����A�*

	conv_loss���>>�        )��P	ţ�����A�*

	conv_lossa��>��8�        )��P	a�����A�*

	conv_loss�!�>@f�c        )��P	������A�*

	conv_loss�{�>�ʀ        )��P	GB�����A�*

	conv_loss�k�>W}��        )��P	�p�����A�*

	conv_loss���>����        )��P	Ȟ�����A�*

	conv_losst��>_D+s        )��P	�Ψ����A�*

	conv_loss=��>���@        )��P	�������A�*

	conv_loss���>�-�<        )��P	�-�����A�*

	conv_loss���>��%�        )��P	�]�����A�*

	conv_loss09�>?>��        )��P	k������A�*

	conv_lossNE�>U�%�        )��P	������A�*

	conv_loss�#�>L���        )��P	|�����A�*

	conv_lossY��>�&�O        )��P	������A�*

	conv_loss#��>�\R8        )��P	l_�����A�*

	conv_loss`��>��TS        )��P	J������A�*

	conv_loss���>����        )��P	Q������A�*

	conv_loss;S�>7|D�        )��P	6�����A�*

	conv_loss���>�1Z        )��P	������A�*

	conv_loss6��>��        )��P	�I�����A�*

	conv_loss��> r��        )��P	�w�����A�*

	conv_loss���>��_�        )��P	e������A�*

	conv_lossW�>u�ul        )��P	�ի����A�*

	conv_lossX��>�utH        )��P	������A�*

	conv_lossV��>��s        )��P	�3�����A�*

	conv_loss��>�|A        )��P	]b�����A�*

	conv_lossHQ�>&�˩        )��P	ѐ�����A�*

	conv_loss���>��[        )��P	x������A�*

	conv_loss�"�>�R        )��P	������A�*

	conv_lossH��>=t�3        )��P	������A�*

	conv_lossX�>!,��        )��P	�L�����A�*

	conv_losshZ�>P�~:        )��P	�|�����A�*

	conv_loss��>�V��        )��P	�������A�*

	conv_lossi�>��        )��P	������A�*

	conv_loss�`�>Qǒp        )��P	V�����A�*

	conv_lossz!�>�\s        )��P	�G�����A�*

	conv_losse��>�]        )��P	�w�����A�*

	conv_loss5�>���        )��P	B������A�*

	conv_loss��>yz�)        )��P	�ٮ����A�*

	conv_loss79�>2�        )��P		�����A�*

	conv_loss3	�>|Z�        )��P	�9�����A�*

	conv_loss���>'p�b        )��P	1u�����A�*

	conv_loss���>�}t(        )��P	�������A�*

	conv_loss�>���O        )��P	�ۯ����A�*

	conv_loss�.�>��%�        )��P	�
�����A�*

	conv_loss��>:$>�        )��P	:�����A�*

	conv_loss/��>�?
        )��P	�s�����A�*

	conv_loss�	�>w�~        )��P	t������A�*

	conv_losso��>jC~        )��P	�Ѱ����A�*

	conv_loss`��>�G^7        )��P	������A�*

	conv_losss��>[B�h        )��P	B8�����A�*

	conv_loss4��>�Ly�        )��P	�z�����A�*

	conv_loss���>�:l        )��P	�������A�*

	conv_loss%��>�Q�M        )��P	�ڱ����A�*

	conv_loss}m�>.�3I        )��P	�	�����A�*

	conv_loss@�>$�        )��P	t9�����A�*

	conv_loss� �>
x��        )��P	�h�����A�*

	conv_loss�|�>W�-        )��P	1������A�*

	conv_loss� �>��b�        )��P	lǲ����A�*

	conv_loss[�>Q��        )��P	������A�*

	conv_loss��>���)        )��P	s%�����A�*

	conv_loss���>JΊ        )��P	�U�����A�*

	conv_lossF/�>7�&�        )��P	�����A�*

	conv_loss=��>��g-        )��P	�������A�*

	conv_loss���>@�T�        )��P	������A�*

	conv_lossIf�>�f�        )��P	�����A�*

	conv_loss^�>a%Hr        )��P	�@�����A�*

	conv_lossX��>����        )��P	�q�����A�*

	conv_lossޘ�>�m��        )��P	�������A�*

	conv_lossf1�>bۻ8        )��P	{д����A�*

	conv_lossʔ�>�7�6        )��P	�������A�*

	conv_lossJ�>
@�        )��P	�.�����A�*

	conv_loss��>�'��        )��P	w]�����A�*

	conv_loss���>�kٙ        )��P	�������A�*

	conv_lossÞ�>:l|�        )��P	H������A�*

	conv_loss���>(��i        )��P	0�����A�*

	conv_losse�>�f\�        )��P	������A�*

	conv_lossO)�>�2�        )��P	M�����A�*

	conv_loss���>��{�        )��P	�{�����A�*

	conv_loss��>f�L�        )��P	A������A�*

	conv_lossE6�>%���        )��P	�ڶ����A�*

	conv_loss���><���        )��P	O	�����A�*

	conv_lossW-�>r
X1        )��P	�8�����A�*

	conv_loss9��>��|        )��P	�v�����A�*

	conv_loss���>xA��        )��P	�������A�*

	conv_loss;ο>���R        )��P	gշ����A�*

	conv_lossX�>`��j        )��P	�����A�*

	conv_loss�>�>=�7�        )��P	m6�����A�*

	conv_loss��>�@�        )��P	re�����A�*

	conv_lossL�>�@�        )��P	M������A�*

	conv_lossR��>�d�L        )��P	(ݸ����A�*

	conv_lossJ>�>�ס�        )��P	:�����A�*

	conv_loss�6�>�;6        )��P	�K�����A�*

	conv_loss��>A�        )��P	[}�����A�*

	conv_losscR�>&��	        )��P	-������A�*

	conv_loss�O�>�*��        )��P	������A�*

	conv_loss���>�ٺ�        )��P	i�����A�*

	conv_loss��>��%        )��P	@H�����A�*

	conv_lossY��>�')        )��P	�w�����A�*

	conv_lossJ��>��,�        )��P	P������A�*

	conv_loss���>e�nW        )��P	uӺ����A�*

	conv_lossVt�>J�V        )��P	������A�*

	conv_lossG�>�3        )��P	lP�����A�*

	conv_lossƯ�>���[        )��P		������A�*

	conv_loss�v�>"Ԧ'        )��P	�������A�*

	conv_loss{��>o*'�        )��P	�����A�*

	conv_lossP�>��        )��P	S�����A�*

	conv_lossB��>��3        )��P	UA�����A�*

	conv_loss�x�>���        )��P	^p�����A�*

	conv_lossl��>W        )��P	n������A�*

	conv_lossC	�>&�L        )��P	�ϼ����A�*

	conv_loss���>ԙ+�        )��P	r������A�*

	conv_loss�{�>�uYV        )��P	*-�����A�*

	conv_loss�+�>?'��        )��P	�[�����A�*

	conv_loss��>^sz^        )��P	Z������A�*

	conv_loss�V�>�4 �        )��P	�������A�*

	conv_loss��>$�	D        )��P	������A�*

	conv_loss^��>��M�        )��P	������A�*

	conv_lossu��>��4�        )��P	lJ�����A�*

	conv_loss >�>�Ad        )��P	�y�����A�*

	conv_loss5Ҿ>�-�M        )��P	����A�*

	conv_lossWY�>�i)        )��P	"پ����A�*

	conv_loss��>��#�        )��P	������A�*

	conv_loss*^�>�1DE        )��P	46�����A�*

	conv_lossT�>��        )��P	Re�����A�*

	conv_lossK$�>�R�I        )��P	�������A�*

	conv_loss�k�>Hk@B        )��P	>¿����A�*

	conv_loss��>�ӿ        )��P	������A�*

	conv_lossO��>2DL�        )��P	�!�����A�*

	conv_loss���>���2        )��P	�Q�����A�*

	conv_lossb$�>Yx�        )��P	[������A�*

	conv_loss\��>�H�        )��P	Ӯ�����A�*

	conv_loss	F�>��        )��P	�������A�*

	conv_loss���>c[�!        )��P	������A�*

	conv_loss���>�v�C        )��P	�J�����A�*

	conv_loss獾>�k�        )��P	�{�����A�*

	conv_loss�F�>��        )��P	 ������A�*

	conv_loss�C�>�^��        )��P	�������A�*

	conv_loss�R�>D�@�        )��P	������A�*

	conv_loss|F�>v��!        )��P	e?�����A�*

	conv_loss=$�>6�        )��P	�n�����A�*

	conv_loss��>����        )��P	r������A�*

	conv_loss���>��Vf        )��P	�������A�*

	conv_loss9��>�n�w        )��P	Y�����A�*

	conv_loss���>�0s        )��P	�I�����A�*

	conv_loss���>)�(�        )��P	yx�����A�*

	conv_loss��>�U�5        )��P	�������A�*

	conv_lossj��>�e�        )��P	�������A�*

	conv_loss���>���        )��P	�"�����A�*

	conv_loss춽>f�ǹ        )��P	PQ�����A�*

	conv_loss���>�G�        )��P	k������A�*

	conv_losss0�>r��        )��P	�������A�*

	conv_loss���>�Js        )��P	H������A�*

	conv_loss#��>s�"�        )��P	�����A�*

	conv_loss`��>����        )��P	�E�����A�*

	conv_loss��>I�        )��P	^}�����A�*

	conv_loss&�>)��C        )��P	������A�*

	conv_lossuP�>�E�O        )��P	q������A�*

	conv_loss�ӽ>�i�        )��P	������A�*

	conv_lossa(�>��        )��P	�?�����A�*

	conv_loss�,�>*�N�        )��P	in�����A�*

	conv_loss���>qB�/        )��P	D������A�*

	conv_loss���>��Q        )��P	�������A�*

	conv_loss���>����        )��P	�������A�*

	conv_lossj�>�&�        )��P	�/�����A�*

	conv_lossj`�>�?]         )��P	�`�����A�*

	conv_lossmֽ>�(V_        )��P	�������A�*

	conv_lossD��>7�s�        )��P	�������A�*

	conv_loss�׽>�_��        )��P	������A�*

	conv_lossGh�>�j��        )��P	Q �����A�*

	conv_loss ݼ>;M        )��P	�O�����A�*

	conv_loss��>F��        )��P	 ������A�*

	conv_loss�w�>��        )��P	r������A�*

	conv_loss�ξ>H)I        )��P	{������A�*

	conv_loss�ظ>N�b�        )��P	������A�*

	conv_loss�>�G�        )��P	A=�����A�*

	conv_loss�c�>���        )��P	:s�����A�*

	conv_loss`߼>�s$x        )��P	�������A�*

	conv_loss捾>�Ej�        )��P	������A�*

	conv_loss��>�tJ        )��P	������A�*

	conv_loss=�>����        )��P	dF�����A�*

	conv_loss~<�>���U        )��P	E}�����A�*

	conv_loss���>���        )��P	�������A�*

	conv_loss0?�>,vi�        )��P	�^�����A�*

	conv_loss�μ>����        )��P	ӏ�����A�*

	conv_loss!e�>��0M        )��P	9������A�*

	conv_loss�/�>�2��        )��P	x������A�*

	conv_loss��>�m��        )��P	1"�����A�*

	conv_loss"�>��J        )��P	�Q�����A�*

	conv_loss���>z���        )��P	E������A�*

	conv_loss�>�acm        )��P	\������A�*

	conv_lossf�>*���        )��P	�������A�*

	conv_loss¼�>f�a�        )��P	Q0�����A�*

	conv_loss��>�WE�        )��P	�n�����A�*

	conv_loss&�>�T�        )��P	n������A�*

	conv_lossQϲ>9A�y        )��P	b������A�*

	conv_loss��>tɉ4        )��P	�������A�*

	conv_loss�N�>U��        )��P	�,�����A�*

	conv_loss��>A�'        )��P	�\�����A�*

	conv_lossvV�>���        )��P	L������A�*

	conv_lossx�>��#�        )��P	һ�����A�*

	conv_loss=ջ>껒�        )��P	������A�*

	conv_loss���>xޟ�        )��P	������A�*

	conv_loss���>"x��        )��P	,O�����A�*

	conv_lossp��>y3��        )��P	F������A�*

	conv_lossTе>6��R        )��P	�������A�*

	conv_loss���>E��        )��P	,������A�*

	conv_loss���>�t��        )��P	%�����A�*

	conv_lossO:�>�.��        )��P	�U�����A�*

	conv_loss���>Ra�Q        )��P	E������A�*

	conv_loss6̹>�`C0        )��P	�������A�*

	conv_loss7�>լ�        )��P	7������A�*

	conv_loss��>�g         )��P	�����A�*

	conv_loss2�>�s+�        )��P	PC�����A�*

	conv_lossU�>�A�        )��P	�q�����A�*

	conv_loss¾�>�cQC        )��P	/������A�*

	conv_loss�F�>�g�        )��P	�������A�*

	conv_loss�Ͻ>��{Y        )��P	f �����A�*

	conv_loss��>Lv�        )��P	10�����A�*

	conv_loss�l�>�#kP        )��P	�^�����A�*

	conv_loss[�>�Qr�        )��P	b������A�*

	conv_lossPz�>s��x        )��P	¾�����A�*

	conv_loss�ɸ>1R\        )��P	c������A�*

	conv_loss<T�>W�2�        )��P	b�����A�*

	conv_loss�6�>cz%        )��P	>K�����A�*

	conv_lossP��>ۘ�        )��P	�y�����A�*

	conv_loss�:�>��vg        )��P	I������A�*

	conv_losst�>����        )��P	v������A�*

	conv_loss��>�)�        )��P	������A�*

	conv_loss9�>ۻI        )��P	6�����A�*

	conv_loss�ʳ>s��        )��P	�d�����A�*

	conv_loss9��>(�7�        )��P	������A�*

	conv_lossW	�>�p         )��P	�������A�*

	conv_loss�j�>W���        )��P	j�����A�*

	conv_loss��>l���        )��P	�1�����A�*

	conv_lossT5�>�`�I        )��P	a�����A�*

	conv_loss� �>��e�        )��P	�������A�*

	conv_loss���>�l        )��P	6������A�*

	conv_loss��>g��        )��P	������A�*

	conv_loss!ΰ>g���        )��P	P4�����A�*

	conv_loss4��>�ϳy        )��P	bc�����A�*

	conv_loss1q�>F�B�        )��P	������A�*

	conv_lossr��>4��h        )��P	9������A�*

	conv_loss�յ>��|        )��P	'�����A�*

	conv_loss3<�>��qa        )��P	�Y�����A�*

	conv_loss�4�>�u        )��P	�������A�*

	conv_loss���>�k�        )��P	�������A�*

	conv_loss -�>��/�        )��P	=������A�*

	conv_loss��>��        )��P	(!�����A�*

	conv_loss�C�>j�}        )��P	�P�����A�*

	conv_loss��>jY�         )��P	������A�*

	conv_loss���>J`�        )��P	į�����A�*

	conv_loss�ȳ>�,�        )��P	�������A�*

	conv_lossRb�>��*        )��P	*�����A�*

	conv_loss��>�y�[        )��P	\>�����A�*

	conv_loss{��>j���        )��P	�r�����A�*

	conv_loss�;�>80{~        )��P	A������A�*

	conv_loss�h�>&��w        )��P	!������A�*

	conv_loss�P�>�Y#        )��P	�����A�*

	conv_lossQ��>xC��        )��P	_B�����A�*

	conv_loss�ʸ>7P"�        )��P	\s�����A�*

	conv_lossBT�>�+��        )��P	�������A�*

	conv_lossK�>t1�        )��P	@������A�*

	conv_loss�B�>��^        )��P	j �����A�*

	conv_loss&�>�T��        )��P	�/�����A�*

	conv_loss�>z8S�        )��P	�^�����A�*

	conv_loss�Ƿ>�p�        )��P	������A�*

	conv_losshb�>�}D�        )��P	�������A�*

	conv_lossU��>�\3�        )��P	H������A�*

	conv_loss.�>˾�        )��P	������A�*

	conv_loss�ѯ>��D        )��P	�M�����A�*

	conv_loss�E�>7�?t        )��P	V|�����A�*

	conv_lossE%�>7�#        )��P	�������A�*

	conv_lossk��>���6        )��P	�������A�*

	conv_loss�A�>!9F        )��P	,
�����A�*

	conv_loss@{�>�(z4        )��P	J9�����A�*

	conv_loss0o�>�S�        )��P	Ch�����A�*

	conv_loss��>��        )��P	B������A�*

	conv_loss�Z�>�g�        )��P	-������A�*

	conv_loss�>�        )��P	�������A�*

	conv_lossX��>��        )��P	=#�����A�*

	conv_lossd��>ԭB�        )��P	.S�����A�*

	conv_loss�֭>S�L�        )��P	w������A�*

	conv_loss�F�>�_��        )��P	�������A�*

	conv_loss���>���D        )��P	�������A�*

	conv_loss�M�>EF�        )��P	������A�*

	conv_loss���>T�!        )��P	�K�����A�*

	conv_loss���>���1        )��P	U~�����A�*

	conv_loss\�>Tc��        )��P	ۭ�����A�*

	conv_loss?�>Z�{�        )��P	������A�*

	conv_lossV�>��z        )��P	Y*�����A�*

	conv_lossk�>�k}Y        )��P	Rc�����A�*

	conv_lossJ�>�z
�        )��P	\������A�*

	conv_loss�c�>�R�        )��P	r������A�*

	conv_loss���>���        )��P	������A�*

	conv_loss?�>�<��        )��P	�9�����A�*

	conv_loss菸>�'�        )��P	m�����A�*

	conv_lossS>�>#���        )��P	e������A�*

	conv_loss���>Y�q        )��P	�������A�*

	conv_lossZe�>>>�!        )��P	k������A�*

	conv_lossXd�>��ڹ        )��P	�)�����A�*

	conv_loss���>�3|�        )��P	cY�����A�*

	conv_loss/��>w!�        )��P	������A�*

	conv_loss���>�y--        )��P	ظ�����A�*

	conv_loss�S�>5��        )��P	�������A�*

	conv_loss�C�>���F        )��P	x�����A�*

	conv_loss	~�>��Γ        )��P	�J�����A�*

	conv_loss?��>�
��        )��P	Qy�����A�*

	conv_lossŹ�>16��        )��P	֨�����A�*

	conv_loss�Ͱ>�L        )��P	������A�*

	conv_loss��>pS�c        )��P	������A�*

	conv_loss��>�*��        )��P	B7�����A�*

	conv_loss��>Դ�m        )��P	�e�����A�*

	conv_lossk-�>3&��        )��P	������A�*

	conv_losss��>"m;        )��P	�������A�*

	conv_loss���>�܉�        )��P	�������A�*

	conv_losso��>�8��        )��P	�%�����A�*

	conv_loss7y�>Y�b�        )��P	&T�����A�*

	conv_loss�>��5        )��P	݂�����A�*

	conv_loss���>����        )��P	=������A�*

	conv_loss�>����        )��P	#������A�*

	conv_loss�>p2�        )��P	!�����A�*

	conv_loss^��>L�=        )��P	�P�����A�*

	conv_loss�/�>��?        )��P	�~�����A�*

	conv_loss)ڬ>L��        )��P	������A�*

	conv_lossE9�>���        )��P	�������A�*

	conv_loss���> X{�        )��P	������A�*

	conv_losstǰ>�IA        )��P	�=�����A�*

	conv_loss��>{�$�        )��P	Rl�����A�*

	conv_loss�q�>ӛ        )��P	�������A�*

	conv_loss>�>����        )��P	9������A�*

	conv_loss�m�>�=X        )��P	/������A�*

	conv_loss��> �=�        )��P	`&�����A�*

	conv_loss�>�P {        )��P	�f�����A�*

	conv_lossI&�>7�;#        )��P	#������A�	*

	conv_loss��>�Ƌ        )��P	u������A�	*

	conv_loss�b�>�,NM        )��P	2������A�	*

	conv_loss�ì>.~Q        )��P	1(�����A�	*

	conv_loss�.�>0�I�        )��P	x`�����A�	*

	conv_losss��>n-�        )��P	�������A�	*

	conv_loss��>wJ&k        )��P	�������A�	*

	conv_losshK�>vdx        )��P	/������A�	*

	conv_loss�o�>�[V        )��P	�-�����A�	*

	conv_lossΡ�>w�@        )��P	�d�����A�	*

	conv_lossG�>ʒ��        )��P	������A�	*

	conv_lossz�>�{�        )��P	�������A�	*

	conv_loss���>EMs         )��P	 ������A�	*

	conv_lossL�>�[�        )��P	b-�����A�	*

	conv_loss=��>�(�        )��P	�`�����A�	*

	conv_loss��>����        )��P	ː�����A�	*

	conv_lossͧ�>.��Q        )��P	\������A�	*

	conv_loss�Ů>H��|        )��P	�������A�	*

	conv_loss�<�>��        )��P	������A�	*

	conv_loss�>�[�        )��P	�O�����A�	*

	conv_loss�>7�)i        )��P	$������A�	*

	conv_loss#��>f�qy        )��P	 ������A�	*

	conv_lossF��>�D        )��P	�������A�	*

	conv_loss�z�>��:�        )��P	?�����A�	*

	conv_loss���>��        )��P	TC�����A�	*

	conv_loss�6�>���        )��P	�|�����A�	*

	conv_loss˻�>e���        )��P	������A�	*

	conv_loss�y�>���Q        )��P	?������A�	*

	conv_loss�<�>���        )��P	^�����A�	*

	conv_loss]�>.wY+        )��P	BG�����A�	*

	conv_loss��>�;#�        )��P	(v�����A�	*

	conv_loss}�>R4�        )��P	ަ�����A�	*

	conv_lossY̭>�fN        )��P	D������A�	*

	conv_loss��>���.        )��P	)�����A�	*

	conv_loss�s�>'��        )��P	�5�����A�	*

	conv_loss�X�>���        )��P	ue�����A�	*

	conv_loss@�>�C        )��P	������A�	*

	conv_lossjV�>�"x        )��P	C������A�	*

	conv_lossDͪ>|�XY        )��P	B������A�	*

	conv_loss�ʩ>v^S        )��P	a#�����A�	*

	conv_loss7��>�        )��P	�R�����A�	*

	conv_lossb�>1Ѣ.        )��P	s������A�	*

	conv_lossWi�>�H,�        )��P	������A�	*

	conv_lossOm�>�,6        )��P	q������A�	*

	conv_loss%צ>D��        )��P	������A�	*

	conv_loss���>,�2        )��P	MA�����A�	*

	conv_lossӮ�>hՐu        )��P	qp�����A�	*

	conv_lossZx�>�.�t        )��P	n������A�	*

	conv_loss�Ƨ>e|2�        )��P	d������A�	*

	conv_lossK��>�͜2        )��P	v]�����A�	*

	conv_loss)�>�%
        )��P	�������A�	*

	conv_loss�s�>����        )��P	������A�	*

	conv_loss�ۧ>�Vz        )��P	S������A�	*

	conv_lossw}�>��z�        )��P	 �����A�	*

	conv_loss`c�>~>��        )��P	!O�����A�	*

	conv_lossä>=��        )��P	~�����A�	*

	conv_loss�Ȫ>�MS�        )��P	7������A�	*

	conv_loss�w�> �Bm        )��P	�������A�	*

	conv_loss��>��_�        )��P	� �����A�	*

	conv_lossl��>͡&�        )��P	)^�����A�	*

	conv_lossٺ�>:y��        )��P	u������A�	*

	conv_loss�˯>�[�        )��P	}������A�	*

	conv_lossyǪ>�pl	        )��P	������A�	*

	conv_loss���>��        )��P	�)�����A�	*

	conv_lossw��>�Tgc        )��P	+X�����A�	*

	conv_lossa�>�f*)        )��P	ˈ�����A�	*

	conv_lossވ�>��6
        )��P	�������A�	*

	conv_loss�ڨ>���        )��P	�������A�	*

	conv_loss�֤>M}��        )��P	������A�	*

	conv_lossJ��>����        )��P	�B�����A�	*

	conv_loss�2�> �
        )��P	sq�����A�	*

	conv_lossz�>2���        )��P	~������A�	*

	conv_loss�>3�p>        )��P	Q������A�	*

	conv_losse�>����        )��P	������A�	*

	conv_loss-�>4��        )��P	D�����A�	*

	conv_lossNo�>�o��        )��P	�r�����A�	*

	conv_loss�D�>� �        )��P	�������A�	*

	conv_loss�C�>7��N        )��P	�������A�	*

	conv_lossHA�>����        )��P	�������A�	*

	conv_loss���>F�Z�        )��P	t=�����A�	*

	conv_loss���>��B�        )��P	8n�����A�	*

	conv_loss~�>�2{        )��P	"������A�	*

	conv_lossL�>����        )��P	������A�	*

	conv_loss��>A���        )��P	 �����A�	*

	conv_lossd"�>K�72        )��P	�.�����A�	*

	conv_loss�)�>3m\�        )��P	h_�����A�	*

	conv_lossP"�>`�7V        )��P	�������A�	*

	conv_loss[8�>G�ޏ        )��P	t������A�	*

	conv_loss:w�>"��(        )��P	2������A�	*

	conv_lossm��>#f,�        )��P	) ����A�	*

	conv_loss�ۣ>->        )��P	uJ ����A�	*

	conv_lossl-�>����        )��P	;x ����A�	*

	conv_lossXt�>S�]�        )��P	J� ����A�	*

	conv_loss���>����        )��P	V� ����A�	*

	conv_loss:�>���%        )��P	k����A�	*

	conv_loss�U�>��        )��P	d4����A�	*

	conv_lossZ��>@���        )��P	&d����A�	*

	conv_loss.��>�ۛ        )��P	������A�	*

	conv_loss��>�B�S        )��P	p�����A�	*

	conv_loss|,�>( i        )��P	�����A�	*

	conv_loss�-�>@W�        )��P	�0����A�	*

	conv_loss%9�>K�o�        )��P	�`����A�	*

	conv_loss���>�]�        )��P	�����A�	*

	conv_lossd�>	1N        )��P	������A�	*

	conv_loss%y�>v@�Z        )��P	�����A�	*

	conv_loss��>[�<�        )��P	�0����A�	*

	conv_loss'��>F`R0        )��P	k`����A�	*

	conv_loss�+�>9�        )��P	������A�	*

	conv_lossϝ�>	�B�        )��P	������A�	*

	conv_lossv��> >3�        )��P	j	����A�	*

	conv_loss[�>�Ҵ�        )��P	�>����A�	*

	conv_lossK��>�Z��        )��P	�r����A�	*

	conv_loss�>\W        )��P	/�����A�	*

	conv_loss���>�8%j        )��P	U�����A�	*

	conv_loss���>����        )��P	8	����A�	*

	conv_loss�9�>
��        )��P	l8����A�	*

	conv_loss ��>~7{�        )��P	 z����A�	*

	conv_loss�r�>���        )��P	p�����A�	*

	conv_loss<S�>q�,        )��P	`�����A�	*

	conv_loss,̧>>��        )��P	T����A�	*

	conv_loss
�>�x��        )��P	BU����A�	*

	conv_loss���>��=&        )��P	������A�	*

	conv_loss�>�>��+�        )��P	$�����A�	*

	conv_loss  �>����        )��P	%�����A�	*

	conv_loss?ˠ>��        )��P	�����A�	*

	conv_lossm�>i)        )��P	-E����A�	*

	conv_loss=��>S^F        )��P	�s����A�	*

	conv_loss�4�>��a�        )��P	4�����A�	*

	conv_loss�أ>^���        )��P	������A�
*

	conv_loss+�>PN�        )��P	=����A�
*

	conv_loss�O�>�B~�        )��P	r0����A�
*

	conv_lossX��>����        )��P	�^����A�
*

	conv_loss���>�J�        )��P	������A�
*

	conv_lossUd�>���6        )��P	z�����A�
*

	conv_loss�X�>�-�'        )��P	�����A�
*

	conv_loss���>�J?�        )��P	�	����A�
*

	conv_lossS�>���'        )��P	K	����A�
*

	conv_loss-*�>0�Q        )��P	5z	����A�
*

	conv_loss:-�>Cfvn        )��P	�	����A�
*

	conv_loss��>'�P�        )��P	G�	����A�
*

	conv_loss�j�>���c        )��P	�
����A�
*

	conv_loss<�>e��d        )��P	�5
����A�
*

	conv_loss�D�>Q��        )��P	Jf
����A�
*

	conv_loss�[�>�\�        )��P	��
����A�
*

	conv_lossQ@�>���z        )��P	��
����A�
*

	conv_loss���>ss]        )��P	.�
����A�
*

	conv_lossz��>�X        )��P	l#����A�
*

	conv_loss�ˤ>��H        )��P	IS����A�
*

	conv_loss0̣>��M        )��P	؁����A�
*

	conv_loss⺢>[ϭ�        )��P	������A�
*

	conv_loss%'�>��e7        )��P	V�����A�
*

	conv_loss�e�>ֿP        )��P	S����A�
*

	conv_lossf��>�
�*        )��P	�K����A�
*

	conv_loss�-�>f�F        )��P	@~����A�
*

	conv_lossp�>f�v�        )��P	N�����A�
*

	conv_loss�`�>�G        )��P	$�����A�
*

	conv_loss���>;��"        )��P	����A�
*

	conv_loss^��>Fg��        )��P	~^����A�
*

	conv_loss=�>���        )��P	В����A�
*

	conv_loss
|�>�l�"        )��P	������A�
*

	conv_loss��>����        )��P	������A�
*

	conv_loss�ġ>ӊc�        )��P	%����A�
*

	conv_loss�Z�>1�F2        )��P	�e����A�
*

	conv_loss���>����        )��P	�����A�
*

	conv_loss���>VZ�        )��P	<�����A�
*

	conv_loss<��>\nx�        )��P	������A�
*

	conv_loss3�>o}��        )��P	�+����A�
*

	conv_lossO��>	4�        )��P	NZ����A�
*

	conv_loss���>|\<�        )��P	9�����A�
*

	conv_loss�]�>с-h        )��P	�����A�
*

	conv_loss�Ȟ> ��        )��P	������A�
*

	conv_loss�ݣ>Pb��        )��P	�(����A�
*

	conv_loss�ã>սH�        )��P	zX����A�
*

	conv_lossO��>j���        )��P	/�����A�
*

	conv_lossab�>��l        )��P	������A�
*

	conv_loss�Ӣ>�A�        )��P	������A�
*

	conv_lossl��>���E        )��P	7����A�
*

	conv_loss1��>����        )��P	�G����A�
*

	conv_lossbp�>>�|        )��P	�w����A�
*

	conv_lossّ�>�w��        )��P	�����A�
*

	conv_lossr��>����        )��P	������A�
*

	conv_loss��>RT��        )��P	$����A�
*

	conv_losss2�>F���        )��P	�7����A�
*

	conv_loss���>��7�        )��P	�f����A�
*

	conv_lossj��>d���        )��P	r�����A�
*

	conv_loss稗>|�U�        )��P	������A�
*

	conv_loss ��>�F�        )��P	>�����A�
*

	conv_loss��>����        )��P	�&����A�
*

	conv_loss{��>�_        )��P	�U����A�
*

	conv_loss�;�>��3        )��P	R�����A�
*

	conv_loss�0�>X�O�        )��P	d�����A�
*

	conv_loss�!�>2I�        )��P	�����A�
*

	conv_losso�>D$�        )��P	
����A�
*

	conv_loss���>�fC2        )��P	6@����A�
*

	conv_lossߊ�>a        )��P	o����A�
*

	conv_loss���>�ҚH        )��P	}�����A�
*

	conv_lossp��>"�o�        )��P	}�����A�
*

	conv_loss׌�>k�        )��P	b�����A�
*

	conv_loss쮟>�5�r        )��P	,����A�
*

	conv_lossO�>��A        )��P	.m����A�
*

	conv_loss�>���        )��P	������A�
*

	conv_loss��>�1�        )��P	������A�
*

	conv_loss��>�߭/        )��P	������A�
*

	conv_loss�h�>��q        )��P	�-����A�
*

	conv_lossD��>d�l�        )��P	^]����A�
*

	conv_loss�3�>�%�        )��P	ό����A�
*

	conv_loss���>0)�        )��P	������A�
*

	conv_loss^[�>q-x        )��P	������A�
*

	conv_loss���>Db�        )��P	�$����A�
*

	conv_loss�*�>��	q        )��P	�T����A�
*

	conv_loss�P�>�:]        )��P	,�����A�
*

	conv_loss�y�>k���        )��P	������A�
*

	conv_lossV�>l"�        )��P	������A�
*

	conv_lossS�>�Hr        )��P	����A�
*

	conv_loss<�>h` �        )��P	�N����A�
*

	conv_loss1�>2�ox        )��P	������A�
*

	conv_loss`7�>{=�        )��P	������A�
*

	conv_loss��>�'�        )��P	g�����A�
*

	conv_lossH�>�3�        )��P	o����A�
*

	conv_loss�p�>>5�        )��P	oI����A�
*

	conv_lossy�>��Y        )��P	y����A�
*

	conv_loss�~�>��*7        )��P	ݧ����A�
*

	conv_lossoĞ>��c        )��P	.�����A�
*

	conv_lossp�>"y1        )��P	����A�
*

	conv_loss�ś>���e        )��P	�6����A�
*

	conv_loss���>�R�t        )��P	&d����A�
*

	conv_loss�s�>����        )��P	�����A�
*

	conv_losse��>���        )��P	\�����A�
*

	conv_loss��>��}4        )��P	\�����A�
*

	conv_loss�@�>Uu=N        )��P	�$����A�
*

	conv_loss)>��0o        )��P	�S����A�
*

	conv_loss-q�>���        )��P	������A�
*

	conv_loss�>Đ�4        )��P	D�����A�
*

	conv_loss��>=�'W        )��P	~�����A�
*

	conv_loss$��>G;hW        )��P	�����A�
*

	conv_lossMq�>aЩ        )��P	"C����A�
*

	conv_lossP��>a�g        )��P	�q����A�
*

	conv_loss��> ���        )��P	x�����A�
*

	conv_loss�Ӝ>�U,}        )��P	M�����A�
*

	conv_loss���>�z�        )��P	������A�
*

	conv_loss�3�>T��x        )��P	�.����A�
*

	conv_loss�Փ>�|        )��P	�`����A�
*

	conv_losst��>�tf�        )��P	������A�
*

	conv_loss�q�>D@�        )��P	������A�
*

	conv_loss�@�>ܐ�        )��P	������A�
*

	conv_loss��>�l	        )��P	�����A�
*

	conv_loss�~�>�xu/        )��P	gJ����A�
*

	conv_loss���>��d        )��P	.y����A�
*

	conv_loss�Ė>�} U        )��P	�����A�
*

	conv_lossT��>�ƓN        )��P	n3 ����A�
*

	conv_loss[i�>� q        )��P	bb ����A�
*

	conv_losso�>���r        )��P	ϐ ����A�
*

	conv_loss͑>�i/�        )��P	�� ����A�
*

	conv_loss7v�>�O��        )��P	�� ����A�
*

	conv_lossr�>[п        )��P	$"!����A�
*

	conv_loss���>�a7        )��P	�Q!����A�
*

	conv_loss��>5'        )��P	j�!����A�*

	conv_lossߞ�>���        )��P	α!����A�*

	conv_loss�S�>����        )��P	@�!����A�*

	conv_lossٗ>Y���        )��P	�"����A�*

	conv_losse��>����        )��P	�L"����A�*

	conv_loss���>G���        )��P	f�"����A�*

	conv_loss-�>�' �        )��P	��"����A�*

	conv_loss�~�>E��        )��P	��"����A�*

	conv_lossT�>~V�        )��P	� #����A�*

	conv_loss�d�>=Lzm        )��P	Q#����A�*

	conv_loss�ޙ>D@E�        )��P	�#����A�*

	conv_loss���>���        )��P	@�#����A�*

	conv_loss"�>U���        )��P	��#����A�*

	conv_loss�<�>v���        )��P	�$����A�*

	conv_lossZ-�>"qu        )��P	L;$����A�*

	conv_loss�T�>t49        )��P	�i$����A�*

	conv_lossuo�>����        )��P	F�$����A�*

	conv_lossw[�>n�B        )��P	��$����A�*

	conv_lossl��>��        )��P	��$����A�*

	conv_loss"��>���        )��P	?9%����A�*

	conv_loss�Z�>��        )��P	$i%����A�*

	conv_loss���>V�p        )��P	��%����A�*

	conv_loss��>��?        )��P	��%����A�*

	conv_loss�>�R�        )��P	��%����A�*

	conv_lossXN�>�;t        )��P	&&����A�*

	conv_loss�>�<�        )��P	�T&����A�*

	conv_lossI�>CQ        )��P	˃&����A�*

	conv_loss���>�$�        )��P	V�&����A�*

	conv_lossZ+�>���        )��P	�&����A�*

	conv_loss`�>�9��        )��P	N'����A�*

	conv_lossk5�>�\�j        )��P	.@'����A�*

	conv_lossE�>�5�        )��P	p'����A�*

	conv_losssW�>�#��        )��P	N�'����A�*

	conv_loss�Y�>���G        )��P	��'����A�*

	conv_loss�͗>s �        )��P	��'����A�*

	conv_lossX��>��/�        )��P	q+(����A�*

	conv_lossQ=�>g�m&        )��P	IZ(����A�*

	conv_loss���>3�}�        )��P	�(����A�*

	conv_loss��>��Cd        )��P	ɷ(����A�*

	conv_loss2�>ꇞH        )��P	F�(����A�*

	conv_lossm��>�툏        )��P	�)����A�*

	conv_loss��>)��$        )��P	wE)����A�*

	conv_lossߛ>�K�*        )��P	(t)����A�*

	conv_loss+̒>T��        )��P	��)����A�*

	conv_loss�z�>�=-        )��P	�)����A�*

	conv_loss���>O���        )��P	�*����A�*

	conv_loss�%�>�Ii        )��P	w?*����A�*

	conv_loss)F�>�!<�        )��P	/q*����A�*

	conv_loss>7�>þ�?        )��P	�*����A�*

	conv_loss�6�>�$פ        )��P	#�*����A�*

	conv_loss�>��-        )��P	8+����A�*

	conv_loss3��>S�
        )��P	�8+����A�*

	conv_loss8B�>��v�        )��P	�l+����A�*

	conv_loss�y�>���        )��P	ݜ+����A�*

	conv_loss�	�>َ��        )��P	'�+����A�*

	conv_loss{��>���o        )��P	��+����A�*

	conv_loss�>0 �        )��P	5,����A�*

	conv_lossJ �>��        )��P	�j,����A�*

	conv_lossOӑ>�oP�        )��P	��,����A�*

	conv_losss��>��R        )��P	�,����A�*

	conv_loss�>�ĳe        )��P	�-����A�*

	conv_loss
�>&�        )��P	�7-����A�*

	conv_loss'u�>��6�        )��P	�g-����A�*

	conv_loss�H�>��        )��P	"�-����A�*

	conv_lossA �>Y^��        )��P	F�-����A�*

	conv_losss�>��xC        )��P	��-����A�*

	conv_loss�p�>��        )��P	�$.����A�*

	conv_loss{�>8G	8        )��P	�S.����A�*

	conv_loss��>��E�        )��P	��.����A�*

	conv_loss4��>xpX�        )��P	��.����A�*

	conv_loss]��>���        )��P	��.����A�*

	conv_loss�J�>X`        )��P	H/����A�*

	conv_loss�M�>Φ�:        )��P	�B/����A�*

	conv_loss{��>ӦTR        )��P	�p/����A�*

	conv_loss0�>�k��        )��P	��/����A�*

	conv_loss��>���[        )��P	��/����A�*

	conv_lossD�>�O��        )��P	! 0����A�*

	conv_loss1ْ>�hpj        )��P	/0����A�*

	conv_lossA!�>˾O        )��P	�]0����A�*

	conv_loss�~�>s��        )��P	�0����A�*

	conv_loss���>�qY        )��P	˼0����A�*

	conv_lossYߓ>�	��        )��P	��0����A�*

	conv_loss��>1-��        )��P	�1����A�*

	conv_loss�g�>'�;        )��P	�K1����A�*

	conv_loss��>_�        )��P	4{1����A�*

	conv_lossI��>���m        )��P	��1����A�*

	conv_loss�S�>*��        )��P	p�1����A�*

	conv_lossAN�>~���        )��P	�2����A�*

	conv_lossb�>Ǩ�Z        )��P	�:2����A�*

	conv_loss�1�>��e        )��P	k2����A�*

	conv_loss��>ܴ��        )��P	?�2����A�*

	conv_loss�[�>���        )��P	��2����A�*

	conv_lossb.�>G��w        )��P	�7����A�*

	conv_loss�Í>�{)�        )��P	W7����A�*

	conv_loss��>��S        )��P	�7����A�*

	conv_loss���>q+T        )��P	µ7����A�*

	conv_lossn'�>W<I-        )��P	��7����A�*

	conv_loss�>��Ì        )��P	�8����A�*

	conv_loss69�>�mGJ        )��P	)I8����A�*

	conv_loss/��>�fܖ        )��P	�w8����A�*

	conv_lossN�>+�R        )��P	J�8����A�*

	conv_loss��>�p7        )��P	
�8����A�*

	conv_loss�і>��X�        )��P	�39����A�*

	conv_loss�כ>!@`g        )��P	�e9����A�*

	conv_loss�R�>�u        )��P	|�9����A�*

	conv_lossex�>δ2        )��P	*�9����A�*

	conv_loss���>M�7�        )��P	5�9����A�*

	conv_loss�'�>l *O        )��P	[.:����A�*

	conv_losss�>�[        )��P	�\:����A�*

	conv_loss�f�>�na�        )��P	n�:����A�*

	conv_loss���>.ǜ        )��P	��:����A�*

	conv_lossd�>�W2�        )��P	��:����A�*

	conv_lossU�>��        )��P	);����A�*

	conv_losss&�>�3To        )��P	�J;����A�*

	conv_lossY��>EX��        )��P	�;����A�*

	conv_loss���>S#��        )��P	��;����A�*

	conv_loss�{�>h��        )��P	5�;����A�*

	conv_loss��>0��S        )��P	�<����A�*

	conv_lossAɒ>+��~        )��P	H<����A�*

	conv_losse��>�6�A        )��P	�v<����A�*

	conv_loss�)�>��(        )��P	3�<����A�*

	conv_lossN
�>�%        )��P	z�<����A�*

	conv_loss��>R["�        )��P		=����A�*

	conv_loss��>Q�N�        )��P	�8=����A�*

	conv_loss�Љ>;�~�        )��P	�i=����A�*

	conv_loss]�>�Ρ�        )��P	�=����A�*

	conv_loss��>�G�        )��P	�=����A�*

	conv_loss(%�>�)J]        )��P	�>����A�*

	conv_loss��>*��e        )��P	�=>����A�*

	conv_loss�*�>�E�        )��P	�p>����A�*

	conv_loss�?�>8BF        )��P	E�>����A�*

	conv_lossG�>- ~�        )��P	��>����A�*

	conv_loss�׏>iN�k        )��P	j�>����A�*

	conv_loss'��>G�K�        )��P	�-?����A�*

	conv_loss:�>�̴j        )��P	:\?����A�*

	conv_loss���>���/        )��P	0�?����A�*

	conv_loss�G�>�:r�        )��P	��?����A�*

	conv_lossÛ�>�Ԧ        )��P	��?����A�*

	conv_loss��>e,�*        )��P	o@����A�*

	conv_lossd��>sUZc        )��P	JJ@����A�*

	conv_loss��>Z���        )��P	�z@����A�*

	conv_loss6A�>����        )��P	Ъ@����A�*

	conv_loss�ܓ>�+l        )��P	��@����A�*

	conv_loss���>���#        )��P	�A����A�*

	conv_loss��>����        )��P	�FA����A�*

	conv_loss�i�>���U        )��P	xvA����A�*

	conv_loss���>��gz        )��P	q�A����A�*

	conv_lossՑ>�m�]        )��P	��A����A�*

	conv_loss숊>��^        )��P	B����A�*

	conv_lossֵ�>�U         )��P	�9B����A�*

	conv_loss�>L��        )��P	FiB����A�*

	conv_loss	U�>"c&        )��P	Y�B����A�*

	conv_loss��>���u        )��P	!�B����A�*

	conv_loss�:�>tF~�        )��P	y*C����A�*

	conv_loss�B�>6��        )��P	�cC����A�*

	conv_loss��>9���        )��P	��C����A�*

	conv_loss���>��Q        )��P	]�C����A�*

	conv_loss���>_��3        )��P	�C����A�*

	conv_loss��>�I�        )��P	S)D����A�*

	conv_loss���>����        )��P	�VD����A�*

	conv_lossʡ�>`&�        )��P	!�D����A�*

	conv_loss'(�>s        )��P	$�D����A�*

	conv_lossꨔ>�q�p        )��P	>�D����A�*

	conv_loss�.�>�w}�        )��P	�E����A�*

	conv_loss�Œ>G��        )��P	�@E����A�*

	conv_lossV`�>��#/        )��P	=oE����A�*

	conv_loss���>/@��        )��P	��E����A�*

	conv_loss#U�>�u׍        )��P	W�E����A�*

	conv_loss-w�>��        )��P	�E����A�*

	conv_loss��>��;Y        )��P	T-F����A�*

	conv_loss�M�>{i�        )��P	�\F����A�*

	conv_loss��>�P�        )��P	ތF����A�*

	conv_loss�c�>ǔ�H        )��P	κF����A�*

	conv_loss�8�>���        )��P	��F����A�*

	conv_loss��>i*%�        )��P	kG����A�*

	conv_lossr�>ydQ�        )��P	oJG����A�*

	conv_loss��>B���        )��P	VyG����A�*

	conv_losst�>        )��P	T�G����A�*

	conv_loss���>^���        )��P	_�G����A�*

	conv_loss�V�>^�        )��P	�H����A�*

	conv_lossx��>�揟        )��P	(7H����A�*

	conv_loss�t�>�̢        )��P	�gH����A�*

	conv_lossu�>{d��        )��P	��H����A�*

	conv_lossan�>�:�b        )��P	z�H����A�*

	conv_loss���>V���        )��P	��H����A�*

	conv_loss&�>��\�        )��P	y"I����A�*

	conv_loss,0�>e�j        )��P	QI����A�*

	conv_lossF"�>𸐈        )��P	i�I����A�*

	conv_loss��>����        )��P	۱I����A�*

	conv_loss}�>\�X�        )��P	<�I����A�*

	conv_loss]F�>�`        )��P	�J����A�*

	conv_losswk�>��nK        )��P		BJ����A�*

	conv_lossg�>h��         )��P	QqJ����A�*

	conv_loss�ό>��=�        )��P	��K����A�*

	conv_lossw��>���        )��P	�/L����A�*

	conv_loss�ӌ>�c4�        )��P	__L����A�*

	conv_loss���>���         )��P	_�L����A�*

	conv_lossǺ�>VM�        )��P	W�L����A�*

	conv_loss�z�>���        )��P	��L����A�*

	conv_lossc�>b��        )��P	�7M����A�*

	conv_loss�!�>Br[        )��P	�kM����A�*

	conv_loss��>p���        )��P	��M����A�*

	conv_loss�}�>��        )��P	~�M����A�*

	conv_loss���>�F        )��P	�	N����A�*

	conv_loss�Y�>� ��        )��P	BN����A�*

	conv_loss���>g��        )��P	�qN����A�*

	conv_loss..�>gZ�D        )��P	|�N����A�*

	conv_lossd��>�ۥ        )��P	��N����A�*

	conv_loss�ē>����        )��P	��N����A�*

	conv_loss��>� yV        )��P	�,O����A�*

	conv_loss�ŋ>~>$�        )��P	�ZO����A�*

	conv_loss��>��c�        )��P	K�O����A�*

	conv_losspZ�>^,4        )��P	��O����A�*

	conv_loss�%�>��/        )��P	�P����A�*

	conv_loss�0�>@�c        )��P	EBP����A�*

	conv_loss�v�>��K        )��P	nrP����A�*

	conv_lossip�>�Kg�        )��P	ѡP����A�*

	conv_loss�U�>�69        )��P	�P����A�*

	conv_loss�>
]��        )��P	( Q����A�*

	conv_loss*S�>����        )��P	�.Q����A�*

	conv_loss��>�މ        )��P	}\Q����A�*

	conv_loss$Џ>?��E        )��P	�Q����A�*

	conv_loss���>�qߘ        )��P	n�Q����A�*

	conv_loss���>����        )��P	}�Q����A�*

	conv_loss~�>;�Cx        )��P	@R����A�*

	conv_loss��>K��        )��P	iJR����A�*

	conv_loss�Z�>�3�        )��P	~yR����A�*

	conv_lossc�>���        )��P	�R����A�*

	conv_loss�ۉ>с�x        )��P	5�R����A�*

	conv_loss���>Ȕ%N        )��P	iS����A�*

	conv_loss6�>���y        )��P	6S����A�*

	conv_lossCv�>a�Ra        )��P	�fS����A�*

	conv_loss�A�>>e �        )��P	��S����A�*

	conv_loss�%�>��/�        )��P	R�S����A�*

	conv_loss��>v,[�        )��P	��S����A�*

	conv_loss:��>e�,�        )��P	!T����A�*

	conv_loss���>���#        )��P	UOT����A�*

	conv_loss���>��n        )��P	d~T����A�*

	conv_loss�
�>7~�W        )��P	��T����A�*

	conv_loss�#�>jݑ�        )��P	��T����A�*

	conv_lossp��>v��)        )��P	�U����A�*

	conv_loss_�>��2�        )��P	�:U����A�*

	conv_loss�w�>�
        )��P	�jU����A�*

	conv_loss��>����        )��P	C�U����A�*

	conv_lossLk�>Y��        )��P	6�U����A�*

	conv_loss=��>W��#        )��P	�V����A�*

	conv_loss`,�>�zEd        )��P	�5V����A�*

	conv_lossΈ>p,Է        )��P	WiV����A�*

	conv_loss�o�><l�        )��P	�V����A�*

	conv_loss�>	�)        )��P	j�V����A�*

	conv_loss T�>���        )��P	B�V����A�*

	conv_lossw�>��q�        )��P	Y5W����A�*

	conv_lossDي>Y�zn        )��P	�oW����A�*

	conv_losse��>6Y�        )��P	,�W����A�*

	conv_loss�(�>�CA        )��P	r�W����A�*

	conv_loss"s�>	��q        )��P	�X����A�*

	conv_lossKG�>`��{        )��P	�@X����A�*

	conv_loss��>"���        )��P	pX����A�*

	conv_loss�҈>�5(        )��P	��X����A�*

	conv_loss{n�>*�=�        )��P	M�X����A�*

	conv_lossu�>S_"f        )��P	�Y����A�*

	conv_loss���>�j�	        )��P	]4Y����A�*

	conv_loss�j�>����        )��P	�cY����A�*

	conv_loss��>e���        )��P	��Y����A�*

	conv_loss��>�i�        )��P	5�Y����A�*

	conv_loss�ƈ>:mnw        )��P	b�Y����A�*

	conv_loss���>�T�        )��P	0Z����A�*

	conv_lossx��>h�@S        )��P	_Z����A�*

	conv_lossw̒>�z�        )��P	��Z����A�*

	conv_loss8��>��        )��P	E�Z����A�*

	conv_lossB�>���        )��P	��Z����A�*

	conv_loss�b�>���        )��P	[����A�*

	conv_lossX;�>U �         )��P	IL[����A�*

	conv_lossA��>���Z        )��P	I{[����A�*

	conv_loss���>��X        )��P	��[����A�*

	conv_loss�8�> m%;        )��P	��[����A�*

	conv_loss֥�>����        )��P	�\����A�*

	conv_loss�e�>z��        )��P	�6\����A�*

	conv_loss1��>r�`H        )��P	^f\����A�*

	conv_loss^�>�	�&        )��P	P�\����A�*

	conv_loss�o�>"��        )��P	��\����A�*

	conv_loss�w�>�
�        )��P	��\����A�*

	conv_loss��>�ƁQ        )��P	]$]����A�*

	conv_loss_e�>C4V        )��P	�R]����A�*

	conv_loss!'�>��        )��P	j�]����A�*

	conv_lossf8�>�of        )��P	F�]����A�*

	conv_loss&��>�E?S        )��P	P�]����A�*

	conv_loss-�>n��x        )��P	9^����A�*

	conv_lossÚ�>WЕJ        )��P	w;^����A�*

	conv_loss���>�>�        )��P	�i^����A�*

	conv_lossg�>�3�2        )��P	�^����A�*

	conv_loss�f�>:��b        )��P	(�^����A�*

	conv_loss�׊>K-��        )��P	2�^����A�*

	conv_loss3��>���        )��P	�8_����A�*

	conv_loss1Ռ>>�_        )��P	�h_����A�*

	conv_loss�h�>�e��        )��P	��_����A�*

	conv_loss��>?�        )��P	��_����A�*

	conv_loss暆>HgH�        )��P	L�_����A�*

	conv_lossJ�>X�e        )��P	�,`����A�*

	conv_loss�d�>KI$        )��P	-\`����A�*

	conv_loss[U�>!��        )��P	�`����A�*

	conv_losst��>s�Y�        )��P	��`����A�*

	conv_lossb.�>cg]        )��P	Ta����A�*

	conv_lossê�>���w        )��P	q6a����A�*

	conv_lossnK�>SO��        )��P	�la����A�*

	conv_loss}~�>��c�        )��P	W�a����A�*

	conv_loss���>�p        )��P	p�a����A�*

	conv_lossJ��>,á|        )��P	
b����A�*

	conv_loss���>�nnC        )��P	�9b����A�*

	conv_lossD��>(;�        )��P	bhb����A�*

	conv_lossȫ�>TX(        )��P	��b����A�*

	conv_loss�&�>�<��        )��P	�b����A�*

	conv_loss��>z'x        )��P	s�b����A�*

	conv_loss��>z�\�        )��P	�%c����A�*

	conv_loss��>:!Sj        )��P	�Uc����A�*

	conv_loss:T�>�̝.        )��P	+�c����A�*

	conv_loss�ن>���t        )��P	I�c����A�*

	conv_loss,΋>�s�        )��P	��c����A�*

	conv_loss?��>-��         )��P	4d����A�*

	conv_loss�z�>P�?�        )��P	�cd����A�*

	conv_loss^��>��aU        )��P	��d����A�*

	conv_lossxJ�>-���        )��P	��d����A�*

	conv_loss:�>@ʲ%        )��P	��d����A�*

	conv_loss���>�Î�        )��P	�e����A�*

	conv_lossࠆ>�Q�        )��P	�Ne����A�*

	conv_lossRb�>�PJ        )��P	�}e����A�*

	conv_lossw�>\�]�        )��P	�e����A�*

	conv_loss�I�>$���        )��P	�e����A�*

	conv_loss��>�1 %        )��P	�f����A�*

	conv_lossT-�>����        )��P	9<f����A�*

	conv_loss�>�<�        )��P	�lf����A�*

	conv_loss���>���        )��P	)�f����A�*

	conv_loss��>+Χ*        )��P	��f����A�*

	conv_loss���>p�2Y        )��P	\�f����A�*

	conv_loss΂>�6�        )��P	�+g����A�*

	conv_lossp��>��(        )��P	�Yg����A�*

	conv_loss���>�ǻE        )��P	��g����A�*

	conv_loss�)�>�A:�        )��P	Թg����A�*

	conv_loss�I�>�!        )��P	�g����A�*

	conv_loss�n�>͟�A        )��P	�h����A�*

	conv_loss	��>`̳�        )��P	�Kh����A�*

	conv_loss@�>V��        )��P	�{h����A�*

	conv_loss�>N)y�        )��P	p�h����A�*

	conv_loss�V�>`��        )��P	��h����A�*

	conv_lossX,}>	T��        )��P	�i����A�*

	conv_lossNL�>�9)�        )��P	,Hi����A�*

	conv_loss�a�>��        )��P	{vi����A�*

	conv_lossf�>�v�        )��P	��i����A�*

	conv_loss4�>NB        )��P	��i����A�*

	conv_loss��>̻��        )��P	�
j����A�*

	conv_loss�C�>��n�        )��P	gCj����A�*

	conv_loss���>�!        )��P	�rj����A�*

	conv_loss���>ɫ�        )��P	��j����A�*

	conv_loss�#�>ڐ�        )��P	��j����A�*

	conv_loss��>|���        )��P	
k����A�*

	conv_lossl��>xf�        )��P	�Ak����A�*

	conv_loss��>�`$        )��P	xk����A�*

	conv_loss��>"Q-/        )��P	��k����A�*

	conv_loss�A�>]C��        )��P	]�k����A�*

	conv_loss�N�>z�        )��P	�l����A�*

	conv_loss�Ӂ>捯        )��P	MEl����A�*

	conv_loss��>��<�        )��P	vul����A�*

	conv_loss���>;���        )��P	�l����A�*

	conv_loss1��>��T        )��P	�l����A�*

	conv_loss��>7=        )��P	�m����A�*

	conv_lossp�>t�;        )��P	�4m����A�*

	conv_lossS�>_�        )��P	�dm����A�*

	conv_loss-��>؟��        )��P	-�m����A�*

	conv_loss=�>�ީ�        )��P	7�m����A�*

	conv_lossvQ�>��a        )��P	��m����A�*

	conv_loss*y�>���l        )��P	5"n����A�*

	conv_loss�چ>,|0�        )��P	NQn����A�*

	conv_loss5̊>��4        )��P	|�n����A�*

	conv_lossB�>� *f        )��P	`�n����A�*

	conv_loss5g�>N�        )��P	7�n����A�*

	conv_lossJ��>�2J        )��P	?o����A�*

	conv_loss+�>d�D        )��P	�<o����A�*

	conv_lossr�>�6        )��P	Pko����A�*

	conv_loss!E�>8�m�        )��P	��o����A�*

	conv_loss	n�>K�$r        )��P	��o����A�*

	conv_lossX�>��k        )��P	��o����A�*

	conv_loss�H�>��        )��P	�&p����A�*

	conv_loss��>����        )��P	OVp����A�*

	conv_loss��>a��        )��P	[�p����A�*

	conv_loss�ԋ>��f�        )��P	j�p����A�*

	conv_loss�3�>�)��        )��P	f�p����A�*

	conv_lossh��>Q���        )��P	�q����A�*

	conv_loss��>���        )��P	�Cq����A�*

	conv_loss��>Z�	�        )��P	mrq����A�*

	conv_lossV˅>xQ�a        )��P	 �q����A�*

	conv_loss�g�>���        )��P	��q����A�*

	conv_loss��>�b��        )��P	!�q����A�*

	conv_loss���>�Zi�        )��P	�-r����A�*

	conv_loss�f�>>��        )��P	/�s����A�*

	conv_lossaB�>���        )��P	.�s����A�*

	conv_loss��>�/        )��P	S-t����A�*

	conv_loss���>k�c        )��P	ct����A�*

	conv_loss1߅>���L        )��P	B�t����A�*

	conv_lossjs�>���        )��P	^�t����A�*

	conv_loss��>�b=�        )��P	��t����A�*

	conv_lossD��>j�        )��P	�u����A�*

	conv_loss��>[��R        )��P	�Uu����A�*

	conv_lossf~>4�D�        )��P	p�u����A�*

	conv_loss��>� ��        )��P	j�u����A�*

	conv_loss�>í@`        )��P	�v����A�*

	conv_lossd�~>(��g        )��P	'7v����A�*

	conv_lossSr�>5��        )��P	fv����A�*

	conv_lossAh�>V�(        )��P	w�v����A�*

	conv_loss�τ>j��6        )��P	��v����A�*

	conv_loss���>�۰        )��P	` w����A�*

	conv_loss*O�>�	��        )��P	�/w����A�*

	conv_loss?�>+�<�        )��P	k_w����A�*

	conv_loss5�>��{�        )��P	M�w����A�*

	conv_loss@l�>�|{�        )��P	�w����A�*

	conv_loss�ъ>�5��        )��P	A�w����A�*

	conv_loss| �>0ǜ        )��P	%x����A�*

	conv_loss��>����        )��P	AZx����A�*

	conv_loss��>uq�        )��P	��x����A�*

	conv_loss�H�>�/-�        )��P	C�x����A�*

	conv_loss�y�>0j�8        )��P	C�x����A�*

	conv_loss���>D��0        )��P	�y����A�*

	conv_loss��>ٍK�        )��P	 Jy����A�*

	conv_loss��>�g�@        )��P	'zy����A�*

	conv_loss��>]P�P        )��P	��y����A�*

	conv_lossu��>E}         )��P	�y����A�*

	conv_loss�|>����        )��P	q	z����A�*

	conv_loss�By>:��o        )��P	�7z����A�*

	conv_loss��>�@�6        )��P	Zfz����A�*

	conv_loss�5�>)�G        )��P	;�z����A�*

	conv_loss!�>��        )��P	��z����A�*

	conv_loss,�>��1}        )��P	��z����A�*

	conv_loss⫄>�[?        )��P	#{����A�*

	conv_loss?Z>5��        )��P	TR{����A�*

	conv_loss��>�aq        )��P	Á{����A�*

	conv_loss\�|>�ûr        )��P	q�{����A�*

	conv_loss>ط g        )��P	��{����A�*

	conv_loss���>�r��        )��P	�|����A�*

	conv_loss���> p�$        )��P	�>|����A�*

	conv_lossE˂>�QN/        )��P	�m|����A�*

	conv_loss�ċ>�*
j        )��P	Z�|����A�*

	conv_loss��~>gn�
        )��P	e�|����A�*

	conv_loss��>��JP        )��P	X�|����A�*

	conv_loss 4�>�d�[        )��P	�,}����A�*

	conv_lossW/>��M        )��P	l}����A�*

	conv_loss�L>{et        )��P	�}����A�*

	conv_loss���>��        )��P	��}����A�*

	conv_loss�σ>ǂ�<        )��P	B�}����A�*

	conv_loss��>�]<(        )��P	�)~����A�*

	conv_loss�L|>h�-]        )��P	�X~����A�*

	conv_lossR��>���        )��P	��~����A�*

	conv_loss��>e�XV        )��P	��~����A�*

	conv_lossKkv>tx��        )��P	7�~����A�*

	conv_loss�~�>f�S�        )��P	�����A�*

	conv_loss^s}>m��P        )��P	;Q����A�*

	conv_loss�}>V��        )��P	�����A�*

	conv_loss*N�>�+Q�        )��P	K�����A�*

	conv_loss.��>��C        )��P	�����A�*

	conv_lossm��>���B        )��P	�.�����A�*

	conv_loss+U�>��!        )��P	$^�����A�*

	conv_loss�0�> �$M        )��P	�������A�*

	conv_loss�&�>�-�B        )��P	�������A�*

	conv_lossz~�><I��        )��P	�����A�*

	conv_loss�>�ٮ�        )��P	#�����A�*

	conv_loss�ǂ>���        )��P	oI�����A�*

	conv_loss��>��P�        )��P	�x�����A�*

	conv_lossq|>�k�^        )��P	�������A�*

	conv_loss~y~><��        )��P	2ׁ����A�*

	conv_lossuL�>���z        )��P	������A�*

	conv_loss��>ώ��        )��P	�6�����A�*

	conv_loss��|>�6�        )��P	�e�����A�*

	conv_loss1Y�>6�B        )��P	�������A�*

	conv_lossqi�>�h��        )��P	�����A�*

	conv_loss�K�>��Ȣ        )��P	N�����A�*

	conv_loss��~>2bv        )��P	c"�����A�*

	conv_lossޫ|>���        )��P	�Q�����A�*

	conv_loss8Y�>���        )��P	:������A�*

	conv_loss�>Oi�        )��P	�������A�*

	conv_loss/@>L�
y        )��P	�������A�*

	conv_loss1�u>�'C�        )��P	������A�*

	conv_loss�"�>liw        )��P	V=�����A�*

	conv_loss��~>9�*        )��P	�k�����A�*

	conv_loss���>�l�'        )��P	Λ�����A�*

	conv_loss�_�>�݉�        )��P	Ā����A�*

	conv_lossX{>���q        )��P	�������A�*

	conv_lossp>��4        )��P	�)�����A�*

	conv_lossZ�{>��        )��P	X�����A�*

	conv_lossVZ�>�.�        )��P	�������A�*

	conv_loss�zw>2��z        )��P	9������A�*

	conv_loss<�}>���7        )��P	b�����A�*

	conv_lossc��>�X��        )��P	�����A�*

	conv_lossWz>p�=K        )��P	�A�����A�*

	conv_lossS�}>i�:        )��P	\q�����A�*

	conv_loss�R�>�`�W        )��P	順����A�*

	conv_loss�8�>0�״        )��P	������A�*

	conv_loss��>|��)        )��P	������A�*

	conv_loss�t>~�z        )��P	�@�����A�*

	conv_loss(��>w�0        )��P	_p�����A�*

	conv_loss1x>�դ�        )��P	�������A�*

	conv_loss��>ç�G        )��P	nև����A�*

	conv_loss���>X���        )��P	K�����A�*

	conv_lossˀp>����        )��P	"=�����A�*

	conv_loss}�>P�M        )��P	�j�����A�*

	conv_lossI�z>��        )��P	G������A�*

	conv_lossWg|>5�52        )��P	�͈����A�*

	conv_lossj}>��f<        )��P	������A�*

	conv_lossT�>�#ef        )��P	/F�����A�*

	conv_loss�Ow>���        )��P	Gv�����A�*

	conv_loss��}>�yi�        )��P	ϭ�����A�*

	conv_loss
K~>�"]        )��P	������A�*

	conv_loss�v�>� n�        )��P	�����A�*

	conv_lossLr>�P��        )��P	�D�����A�*

	conv_loss�L}>�p�        )��P	Mt�����A�*

	conv_loss��>��r�        )��P	ߦ�����A�*

	conv_loss�Iz>ﶪ	        )��P	\׊����A�*

	conv_lossÞy>�jTr        )��P	?�����A�*

	conv_loss�?|>A�:        )��P	5A�����A�*

	conv_loss�ׄ>
��	        )��P	p�����A�*

	conv_loss��|>�9��        )��P	ݟ�����A�*

	conv_lossx�v>�t        )��P	΋����A�*

	conv_lossk/�>���a        )��P	]�����A�*

	conv_lossw\z>�{�#        )��P	c8�����A�*

	conv_loss�\w>���C        )��P	�m�����A�*

	conv_loss�z>v&:�        )��P	N������A�*

	conv_loss+��>�q�        )��P	sΌ����A�*

	conv_loss���>�#T        )��P	�������A�*

	conv_loss=r>~?s        )��P	,,�����A�*

	conv_loss�kz>�a,        )��P	I[�����A�*

	conv_loss��w>|�̭        )��P	Ë�����A�*

	conv_loss��>��]        )��P	������A�*

	conv_loss��>��Y        )��P	������A�*

	conv_loss>/�        )��P	.�����A�*

	conv_loss�{>�H=1        )��P	�K�����A�*

	conv_loss0�~>�W�        )��P	�{�����A�*

	conv_loss��{>5چ#        )��P	ª�����A�*

	conv_loss�G�>��$        )��P	�ڎ����A�*

	conv_loss��>�5wk        )��P	�	�����A�*

	conv_loss���>5X5�        )��P	�8�����A�*

	conv_loss�J}>��        )��P	�g�����A�*

	conv_loss�ۀ>����        )��P	�������A�*

	conv_loss�w>�fo-        )��P	�Ə����A�*

	conv_loss��|>�]N�        )��P	�������A�*

	conv_loss\wu>���        )��P	%�����A�*

	conv_lossR�w>H[J�        )��P	�R�����A�*

	conv_loss"ov>WA�G        )��P	ǎ�����A�*

	conv_loss�x�>�ܼC        )��P	2������A�*

	conv_lossb�|>��w        )��P	������A�*

	conv_loss"́>����        )��P	@�����A�*

	conv_loss��v>�7�        )��P	P�����A�*

	conv_loss��>���<        )��P	̇�����A�*

	conv_lossj�i>���        )��P	����A�*

	conv_lossJ}>�D�o        )��P	�����A�*

	conv_loss `{>a�D        )��P	������A�*

	conv_lossN\o>��        )��P	mP�����A�*

	conv_loss�H�>#��        )��P	߈�����A�*

	conv_loss�Bn>_{��        )��P	������A�*

	conv_lossc�>V�,B        )��P	0�����A�*

	conv_loss-�{>S=3�        )��P	������A�*

	conv_loss"�s>��-�        )��P	.E�����A�*

	conv_loss�4y>'5I        )��P	t�����A�*

	conv_loss7�~>�ѡ        )��P	ب�����A�*

	conv_loss?,>Y#ی        )��P	�ݓ����A�*

	conv_loss��|>�9�        )��P	o�����A�*

	conv_loss��>���%        )��P	B?�����A�*

	conv_loss��v>]�ޠ        )��P	�n�����A�*

	conv_loss�s>Z���        )��P	+������A�*

	conv_loss��}>V��E        )��P	�̔����A�*

	conv_lossw�m>��}        )��P	������A�*

	conv_loss��y>b7��        )��P	1+�����A�*

	conv_loss� y>�ji        )��P	
[�����A�*

	conv_loss�Mm>{�n        )��P	�������A�*

	conv_loss�{>�J        )��P	������A�*

	conv_loss�{>�[�T        )��P	E�����A�*

	conv_loss'�u>h��H        )��P	������A�*

	conv_loss��z>�=P�        )��P	�F�����A�*

	conv_loss7�u>t5<X        )��P	%u�����A�*

	conv_lossJ��>�n�	        )��P	�������A�*

	conv_loss�5t>]��        )��P	�Ӗ����A�*

	conv_loss5t>9�c;        )��P	 �����A�*

	conv_loss	�x>Ҿ        )��P	�2�����A�*

	conv_loss�`�>��*r        )��P	�a�����A�*

	conv_loss<�v>���1        )��P	>������A�*

	conv_loss�i�>M�P        )��P	¿�����A�*

	conv_loss�?n>G���        )��P	������A�*

	conv_loss�!�>N,8�        )��P	|�����A�*

	conv_loss_�}>&�Hx        )��P	fK�����A�*

	conv_loss@�{>�9�        )��P	�y�����A�*

	conv_loss�v>�Y��        )��P	稘����A�*

	conv_loss��z>�/_�        )��P	�֘����A�*

	conv_loss��|>����        )��P	y�����A�*

	conv_loss�\{>&K��        )��P	�3�����A�*

	conv_loss��>�ʆ        )��P	c�����A�*

	conv_loss��r>�z�        )��P	(������A�*

	conv_losse�r>]�M%        )��P	w�����A�*

	conv_loss��x>{k]2        )��P	�o�����A�*

	conv_loss��t>��U        )��P	�������A�*

	conv_loss�~{>�$A        )��P	]Ο����A�*

	conv_loss�{>���        )��P	�������A�*

	conv_loss%S�>꧉        )��P	�0�����A�*

	conv_loss)�~>����        )��P	Bb�����A�*

	conv_loss��~>^ ��        )��P	Ԓ�����A�*

	conv_losskup>�^�        )��P	+ ����A�*

	conv_loss��z>�AM�        )��P	v �����A�*

	conv_loss\�t>��]j        )��P	(4�����A�*

	conv_losswR�>�r,        )��P	/f�����A�*

	conv_loss�Lr>��S        )��P	S������A�*

	conv_lossC,w>'g6�        )��P	[š����A�*

	conv_loss�v>د��        )��P	�������A�*

	conv_loss�:}>���        )��P	�+�����A�*

	conv_loss��k>?�ُ        )��P	�\�����A�*

	conv_loss!�s>��7        )��P	�������A�*

	conv_loss��~>j�5        )��P	#������A�*

	conv_loss�r>��A�        )��P	�������A�*

	conv_loss�Ey>2Rv        )��P	�(�����A�*

	conv_loss�{>��        )��P	�X�����A�*

	conv_loss�'v>x4F        )��P	�������A�*

	conv_loss��>�D�&        )��P	깣����A�*

	conv_loss]|{>�#�        )��P	������A�*

	conv_loss��o>�Oz        )��P	T�����A�*

	conv_loss]�u>�{�:        )��P	tH�����A�*

	conv_loss�ey>^�V�        )��P	vv�����A�*

	conv_lossXEz>�v�        )��P	ե�����A�*

	conv_loss��{>���        )��P	�Ԥ����A�*

	conv_loss5]p>ќ��        )��P	6�����A�*

	conv_loss=�r>�t~A        )��P	�3�����A�*

	conv_loss�w>T�         )��P	;c�����A�*

	conv_lossėx>h��        )��P	�������A�*

	conv_lossXj>O��3        )��P	�������A�*

	conv_loss��x>kǎ�        )��P	������A�*

	conv_loss�u>{�"        )��P	u�����A�*

	conv_loss��h>���        )��P	oN�����A�*

	conv_loss��v>T�ł        )��P	�������A�*

	conv_loss=�w>���        )��P	�������A�*

	conv_loss��{>F1�        )��P	@�����A�*

	conv_loss��w>:_�        )��P	������A�*

	conv_lossq�g>DQ�M        )��P	bI�����A�*

	conv_loss�St>�ed�        )��P	y�����A�*

	conv_loss��p>��6        )��P	�������A�*

	conv_loss�h>�q�v        )��P	mا����A�*

	conv_loss s>�㗈        )��P	}�����A�*

	conv_loss̓m>v��        )��P	H6�����A�*

	conv_loss��w>+���        )��P	<e�����A�*

	conv_loss��p>ӎ'
        )��P	������A�*

	conv_loss�k>�bϐ        )��P	NŨ����A�*

	conv_loss#2t>�ל        )��P	������A�*

	conv_lossQ�y>R'RW        )��P	�2�����A�*

	conv_lossk�p>*�pw        )��P	�b�����A�*

	conv_loss_Xp>Zj��        )��P	�������A�*

	conv_lossq��>gy�C        )��P	xƩ����A�*

	conv_lossSt>��.        )��P	N������A�*

	conv_loss͖u>=�=W        )��P	$-�����A�*

	conv_loss_nr>�b�s        )��P	�t�����A�*

	conv_loss��v>g�3�        )��P	N������A�*

	conv_lossۮi>t��        )��P	�ߪ����A�*

	conv_loss��g>m�.,        )��P	t�����A�*

	conv_loss%&v>]���        )��P	�L�����A�*

	conv_loss
n>��|�        )��P	�}�����A�*

	conv_lossJUf>J&8�        )��P	1������A�*

	conv_lossB!q>�$z        )��P	`۫����A�*

	conv_losseXt>�b�        )��P	;�����A�*

	conv_loss�ks>�X��        )��P	�G�����A�*

	conv_loss`n>tMj        )��P	kw�����A�*

	conv_loss�3z>��Ҵ        )��P	�������A�*

	conv_loss�W{>��^�        )��P	�ܬ����A�*

	conv_loss��u>�K}�        )��P	f�����A�*

	conv_loss�>u>�Xo        )��P	jE�����A�*

	conv_lossu�s>kA��        )��P	�v�����A�*

	conv_loss1�x>�΂        )��P	�������A�*

	conv_loss�?q>�Sk�        )��P	խ����A�*

	conv_loss!�p>ه�	        )��P	6�����A�*

	conv_loss�}c>�p         )��P	�1�����A�*

	conv_loss=�n>/c|�        )��P	@`�����A�*

	conv_loss��q>@��x        )��P	ݏ�����A�*

	conv_lossJiv>�.T        )��P	˿�����A�*

	conv_loss��x>6~�        )��P	������A�*

	conv_loss�?l>s��        )��P	�����A�*

	conv_loss�q>Nt��        )��P	�N�����A�*

	conv_loss�r>��W        )��P	K~�����A�*

	conv_lossk>��        )��P	֬�����A�*

	conv_loss�Yf>[hs        )��P	�ۯ����A�*

	conv_loss�(w>��8        )��P	������A�*

	conv_loss�vq>���        )��P	�J�����A�*

	conv_loss��s>�+�Q        )��P	�y�����A�*

	conv_loss�x>o�l        )��P	�������A�*

	conv_loss9w>0�J        )��P	�װ����A�*

	conv_loss��m>]�55        )��P	������A�*

	conv_lossR�s>�AR�        )��P	�6�����A�*

	conv_loss�x>��Z�        )��P	�x�����A�*

	conv_loss�2q>���         )��P	������A�*

	conv_losso�m>��        )��P	iݱ����A�*

	conv_loss�F~>c/��        )��P	������A�*

	conv_loss�tq>I$�X        )��P	�H�����A�*

	conv_loss�>�>��'�        )��P	 x�����A�*

	conv_lossC�>|Ƒ        )��P	�������A�*

	conv_lossX0�>a�j�        )��P	������A�*

	conv_loss��o>�E��        )��P	H�����A�*

	conv_lossX�n>��{K        )��P	C�����A�*

	conv_loss�au>{}�        )��P	r�����A�*

	conv_loss��|>�x�        )��P	ĥ�����A�*

	conv_lossP�w>Rp��        )��P	�Գ����A�*

	conv_loss�Qt>
BZ        )��P	������A�*

	conv_loss�n>�+�Y        )��P	�1�����A�*

	conv_loss��x>�;��        )��P	gn�����A�*

	conv_loss6�t>���@        )��P	�������A�*

	conv_loss.mi>,]�        )��P	�ٴ����A�*

	conv_loss��p>QB��        )��P	�	�����A�*

	conv_lossn�r>�iOD        )��P	�9�����A�*

	conv_loss�u>���        )��P	eh�����A�*

	conv_lossX^p>D���        )��P	؟�����A�*

	conv_loss��c>�c�Y        )��P	�ϵ����A�*

	conv_loss�r>���        )��P	�������A�*

	conv_loss�dw>2��        )��P	[.�����A�*

	conv_loss�l>��X�        )��P	�\�����A�*

	conv_loss^Vk>�NC        )��P	�������A�*

	conv_loss��t>�C��        )��P	!������A�*

	conv_loss�s>	��        )��P	2�����A�*

	conv_loss��w>���D        )��P	h7�����A�*

	conv_loss��i>��        )��P	uh�����A�*

	conv_loss��b>i�        )��P	�����A�*

	conv_loss��l>8p�Q        )��P	CƷ����A�*

	conv_loss��o>Kyx�        )��P	<������A�*

	conv_loss�i>E�I        )��P	&�����A�*

	conv_lossbtl>U@d�        )��P	�U�����A�*

	conv_loss�l>IGF        )��P	������A�*

	conv_loss�z>|���        )��P	������A�*

	conv_loss�Vt>4���        )��P	Q�����A�*

	conv_loss2h>)Ƭ        )��P	�����A�*

	conv_loss_�x>HR�        )��P	@�����A�*

	conv_loss$�m>.�Z        )��P	&p�����A�*

	conv_loss�ch>	�uC        )��P	�������A�*

	conv_lossk>�D�        )��P	lι����A�*

	conv_lossP�t>k���        )��P	�������A�*

	conv_loss)�~>G@#�        )��P	%.�����A�*

	conv_loss��m>=� �        )��P	:]�����A�*

	conv_loss�Je>r�J        )��P	�������A�*

	conv_losskn>h�^�        )��P	Ž�����A�*

	conv_loss	pc>[Ϻ^        )��P	r�����A�*

	conv_loss�t>�O�        )��P	������A�*

	conv_loss� |>N}��        )��P	�I�����A�*

	conv_loss^�v>�H        )��P	y�����A�*

	conv_loss�r>�U��        )��P	)������A�*

	conv_loss�hm>��{j        )��P	<ػ����A�*

	conv_loss�cl>�82{        )��P	?�����A�*

	conv_lossg?t>1=EE        )��P	�6�����A�*

	conv_loss]/j>-q�        )��P	Fv�����A�*

	conv_loss"�a>��V�        )��P	�������A�*

	conv_loss��i>(�        )��P	�ּ����A�*

	conv_loss��`>�QQz        )��P	������A�*

	conv_loss��a>`R��        )��P	@�����A�*

	conv_loss��b>_f(�        )��P	vq�����A�*

	conv_loss�d>�De�        )��P	������A�*

	conv_loss�Xw>y�G�        )��P	�ڽ����A�*

	conv_lossp>��2        )��P	������A�*

	conv_lossCl>��        )��P	I�����A�*

	conv_lossm�f>��@�        )��P	˂�����A�*

	conv_loss]�o>�K	�        )��P	������A�*

	conv_lossBn>V�e;        )��P	T�����A�*

	conv_loss�el>�,*{        )��P	_�����A�*

	conv_lossѹd>-7!O        )��P	iN�����A�*

	conv_loss��o>����        )��P	}�����A�*

	conv_loss�m>�4�        )��P	�������A�*

	conv_lossjl>q�K�        )��P	6ۿ����A�*

	conv_loss�q>�Q�s        )��P	������A�*

	conv_lossqAq>�T"        )��P	p:�����A�*

	conv_lossB�r>���        )��P	)i�����A�*

	conv_lossís>jj��        )��P	"������A�*

	conv_loss�Ms>�x�        )��P	A������A�*

	conv_loss�i>���6        )��P	�������A�*

	conv_lossF�k>�xa�        )��P	�&�����A�*

	conv_loss��j>�f�        )��P	�U�����A�*

	conv_loss�gq>����        )��P	�������A�*

	conv_loss�|o>��B�        )��P	R������A�*

	conv_loss�t>gT#�        )��P	\������A�*

	conv_loss�8b>.�        )��P	������A�*

	conv_loss��j>��j)        )��P	�C�����A�*

	conv_loss4�j>ǽ�h        )��P	r�����A�*

	conv_loss5%m>���        )��P	������A�*

	conv_loss6]>e�/        )��P	�������A�*

	conv_loss�Mf>m�"-        )��P	q�����A�*

	conv_lossV�f>B�&�        )��P	�1�����A�*

	conv_loss/m~>���        )��P	z`�����A�*

	conv_lossW+v>"dC:        )��P	F������A�*

	conv_loss��g>�:-z        )��P	������A�*

	conv_loss�2o>�!�        )��P	|������A�*

	conv_lossszj>@ܟ�        )��P	2�����A�*

	conv_loss�yd>��@        )��P	�K�����A�*

	conv_loss��j>�[�4        )��P	�z�����A�*

	conv_lossfbf>�.7]        )��P	�������A�*

	conv_loss�e>���        )��P	�������A�*

	conv_loss�p>�!        )��P	������A�*

	conv_loss�^>`g��        )��P	�7�����A�*

	conv_loss��h>���        )��P	`f�����A�*

	conv_loss��f>1��^        )��P	B������A�*

	conv_loss�m>���        )��P	�������A�*

	conv_loss��e>u�9        )��P	�V�����A�*

	conv_lossl>q=X�        )��P	<������A�*

	conv_loss1�h>}���        )��P	N������A�*

	conv_losst�f>b5p�        )��P	�������A�*

	conv_lossL�k>4�.        )��P	�����A�*

	conv_lossB`j>	�r�        )��P	�G�����A�*

	conv_loss�\i>.�\�        )��P	�}�����A�*

	conv_loss<�b>ZEБ        )��P	u������A�*

	conv_loss$�m>X/�y        )��P	�������A�*

	conv_loss�Zl>��RD        )��P	�&�����A�*

	conv_loss�#k>=��        )��P	TV�����A�*

	conv_lossVbn>��#�        )��P	߄�����A�*

	conv_lossqkh>ƶV        )��P	�������A�*

	conv_loss`>.���        )��P	�������A�*

	conv_loss��a>�{��        )��P	P�����A�*

	conv_lossRf>M �        )��P	FH�����A�*

	conv_loss��c>���        )��P	�v�����A�*

	conv_loss�k>��j        )��P	P������A�*

	conv_lossH�i>�
�        )��P	�������A�*

	conv_loss��k>I,��        )��P	v�����A�*

	conv_lossn�g>_�V        )��P	�=�����A�*

	conv_lossĬg>�#        )��P	Ny�����A�*

	conv_loss<�n>���l        )��P	�������A�*

	conv_loss�xi>6I6        )��P	������A�*

	conv_lossZ>&�S�        )��P	������A�*

	conv_loss��g>r�k7        )��P	�7�����A�*

	conv_loss�<o>�4�l        )��P	^e�����A�*

	conv_loss��Y>�m�+        )��P	H������A�*

	conv_loss�yu>�?d�        )��P	�������A�*

	conv_loss�\a>��/�        )��P	������A�*

	conv_loss�Ba>U���        )��P	�!�����A�*

	conv_loss�Ck>
"�        )��P	&P�����A�*

	conv_loss�bc> �c        )��P	f������A�*

	conv_lossj�}>�슓        )��P	ӯ�����A�*

	conv_loss��t>�E��        )��P	J������A�*

	conv_loss��f>�bO<        )��P	*�����A�*

	conv_lossݵq>t���        )��P	f<�����A�*

	conv_loss�dq>S��        )��P	�k�����A�*

	conv_loss��n>��s�        )��P	3������A�*

	conv_loss��d>��u;        )��P	�������A�*

	conv_lossye>PxCt        )��P	&������A�*

	conv_loss��_>�Mi_        )��P	�'�����A�*

	conv_loss�=h>G�F�        )��P	�V�����A�*

	conv_loss��f>&�U        )��P	V������A�*

	conv_lossx�d>��        )��P	������A�*

	conv_lossX!t>��,�        )��P	Q������A�*

	conv_loss�"i><Q        )��P	������A�*

	conv_lossR'm>O�	�        )��P	zB�����A�*

	conv_losspi>V��        )��P	qq�����A�*

	conv_lossB1g>��        )��P	<������A�*

	conv_loss]Kk>68�        )��P	�������A�*

	conv_loss'�d>��ѱ        )��P	h�����A�*

	conv_loss^�b>p��d        )��P	�=�����A�*

	conv_loss�o>�xrJ        )��P	�l�����A�*

	conv_loss��d>:bZ�        )��P	�������A�*

	conv_lossڶu>�BM        )��P	l������A�*

	conv_loss�=j>K��        )��P	<�����A�*

	conv_loss�f>1K�        )��P	�0�����A�*

	conv_loss��a>z���        )��P	0k�����A�*

	conv_loss�h>��        )��P	9������A�*

	conv_lossv/[>���        )��P	�������A�*

	conv_lossGuZ>Y^�        )��P	������A�*

	conv_loss�f>��6�        )��P	DB�����A�*

	conv_loss�Zl>��&�        )��P	�~�����A�*

	conv_loss�U>(t��        )��P	L������A�*

	conv_loss7�d>;ǁ        )��P	������A�*

	conv_loss-d>^:�        )��P	������A�*

	conv_lossSaj>P4S�        )��P	)<�����A�*

	conv_loss��l>�S�"        )��P	�i�����A�*

	conv_loss��f>6��s        )��P	F������A�*

	conv_loss�ql>ʖ�        )��P	�������A�*

	conv_loss�ah>mK�        )��P	�������A�*

	conv_loss��g>>V��        )��P	u+�����A�*

	conv_lossop>����        )��P	5`�����A�*

	conv_lossM�_>ݾ�*        )��P	������A�*

	conv_loss�\>?�        )��P	N������A�*

	conv_loss�5d>�@�        )��P	������A�*

	conv_loss��i>᤬i        )��P	�"�����A�*

	conv_lossr�f>�`'        )��P	@S�����A�*

	conv_lossUj>�%Q        )��P	k������A�*

	conv_loss{li>�~�
        )��P	\������A�*

	conv_loss�.i>`�ph        )��P	E������A�*

	conv_loss*�c>�,�        )��P	5�����A�*

	conv_lossS�l>@��|        )��P	�E�����A�*

	conv_lossG^f>M<�        )��P	�������A�*

	conv_loss�h>����        )��P	�������A�*

	conv_loss�e>��        )��P	�����A�*

	conv_losss(e>�z�2        )��P	�2�����A�*

	conv_loss��]>%�F        )��P	ta�����A�*

	conv_loss8T]>Ub�r        )��P	͏�����A�*

	conv_loss>�a>^��        )��P	�������A�*

	conv_loss��j>ܗt        )��P	�������A�*

	conv_lossD�`>'F�        )��P	������A�*

	conv_loss��^>f�S�        )��P	�H�����A�*

	conv_loss�e>����        )��P	ux�����A�*

	conv_loss��Y>1��-        )��P	�������A�*

	conv_loss�e>�<�        )��P	�������A�*

	conv_loss]>��B        )��P	������A�*

	conv_loss�i>�/��        )��P	�6�����A�*

	conv_loss�3o>6��        )��P	�d�����A�*

	conv_loss�e>�!V        )��P	�������A�*

	conv_loss'lk>���        )��P	�������A�*

	conv_lossF[>�|2�        )��P	������A�*

	conv_loss<6j>��C        )��P	�3�����A�*

	conv_loss^c>A3��        )��P	�e�����A�*

	conv_loss� c>q��4        )��P	ϕ�����A�*

	conv_loss��q>hR        )��P	������A�*

	conv_loss�lb>!�'�        )��P	�������A�*

	conv_loss�9c>�)        )��P	G,�����A�*

	conv_loss5\>�Q��        )��P	�`�����A�*

	conv_losseW^>s_        )��P	������A�*

	conv_loss}~p>��0�        )��P	}������A�*

	conv_loss��q>8UHE        )��P	�����A�*

	conv_loss�I[>"�Fs        )��P	�7�����A�*

	conv_lossޕ_>-�Ȕ        )��P	,g�����A�*

	conv_loss�"]>�d(M        )��P	�������A�*

	conv_loss�`>o��        )��P	o������A�*

	conv_loss��d>�"3�        )��P	������A�*

	conv_loss7�Z>T�_�        )��P	�!�����A�*

	conv_loss�wf>V��        )��P	Q�����A�*

	conv_loss�+\>�K��        )��P	������A�*

	conv_loss=]]>�R�5        )��P	߮�����A�*

	conv_loss5�g>�0�        )��P	�������A�*

	conv_loss�d>�͡        )��P	������A�*

	conv_lossL�g>\#��        )��P	�=�����A�*

	conv_lossסp>�s        )��P	^m�����A�*

	conv_loss&Sf>�U��        )��P	������A�*

	conv_loss@G[>�ͲI        )��P	�������A�*

	conv_loss�\p>�!�        )��P	�������A�*

	conv_loss��i>d��        )��P	7*�����A�*

	conv_loss�>d>.*8�        )��P	AY�����A�*

	conv_lossSH[>؏��        )��P	������A�*

	conv_lossc`>u�D        )��P	�������A�*

	conv_loss�2\>�        )��P	a������A�*

	conv_loss��[>o�h        )��P	�����A�*

	conv_lossP�\>����        )��P	D�����A�*

	conv_loss�h>�de�        )��P	r�����A�*

	conv_loss1�`>U��^        )��P	"������A�*

	conv_lossYZ>[}        )��P	������A�*

	conv_loss��q>mԪ�        )��P	v �����A�*

	conv_lossDf>)��        )��P	n0�����A�*

	conv_loss]�X>��5�        )��P	�^�����A�*

	conv_loss�\\>��sE        )��P	@������A�*

	conv_loss�,c>��F        )��P	S������A�*

	conv_loss�Wc>�,�b        )��P	�������A�*

	conv_loss �b>��g        )��P	������A�*

	conv_lossX�j>U�	�        )��P	�H�����A�*

	conv_lossl�f>�"��        )��P	Fw�����A�*

	conv_loss+�Z>Y�z        )��P	�������A�*

	conv_loss=�b>R۟�        )��P	�������A�*

	conv_loss[Y^>늻�        )��P	M�����A�*

	conv_loss|�e>�u�        )��P	vD�����A�*

	conv_loss�Ri>JND        )��P	=r�����A�*

	conv_lossx�c>��        )��P	������A�*

	conv_loss+Gm>>R)f        )��P	�������A�*

	conv_loss|`>���        )��P	������A�*

	conv_loss?�\>�۾�        )��P	�<�����A�*

	conv_loss��_>T���        )��P	hk�����A�*

	conv_losss%^>����        )��P	������A�*

	conv_lossv�f>�nk        )��P	�������A�*

	conv_loss�X>��m        )��P	p������A�*

	conv_loss��\>��̉        )��P	�4�����A�*

	conv_loss�JY>TN%`        )��P		d�����A�*

	conv_loss˪b>�:�        )��P	n������A�*

	conv_lossU)^>cu�<        )��P	�������A�*

	conv_loss�Ed>p��E        )��P	�������A�*

	conv_loss��T>Xф:        )��P	�1�����A�*

	conv_loss)�b>�NN�        )��P	[d�����A�*

	conv_loss�X>O���        )��P	�������A�*

	conv_lossGZ>���        )��P	�������A�*

	conv_lossk�^>D�        )��P	,������A�*

	conv_loss�<T>;)b�        )��P	�"�����A�*

	conv_loss��\>��!I        )��P	hQ�����A�*

	conv_loss��_>R;�        )��P	9������A�*

	conv_lossy�`>�b��        )��P	�������A�*

	conv_loss�=U>���        )��P	������A�*

	conv_loss�&j>�R_�        )��P	%�����A�*

	conv_loss�|g>�G��        )��P	�=�����A�*

	conv_loss�([>*5}�        )��P	:n�����A�*

	conv_loss�g`>���        )��P	ޞ�����A�*

	conv_loss_�^>9��1        )��P	�������A�*

	conv_loss^�c>pT��        )��P	�������A�*

	conv_loss�-]>��Y        )��P	3.�����A�*

	conv_lossˮo>W�5        )��P	h]�����A�*

	conv_lossbi>5���        )��P	�������A�*

	conv_loss�Oh>�K�@        )��P	}������A�*

	conv_lossaqP> �         )��P	�������A�*

	conv_loss��o>����        )��P	~�����A�*

	conv_loss�@c>�X        )��P	I�����A�*

	conv_loss`�]>�]�        )��P	kx�����A�*

	conv_lossbs\>��!�        )��P	�������A�*

	conv_loss��b>��        )��P	�������A�*

	conv_lossb_l>Az�{        )��P	r�����A�*

	conv_loss�{b>���g        )��P	l4�����A�*

	conv_loss�V>��        )��P	cd�����A�*

	conv_loss#X>�C�$        )��P	4������A�*

	conv_loss�9`>���        )��P	Կ�����A�*

	conv_loss/>V>S�u        )��P	P������A�*

	conv_loss�-]>�)4�        )��P	>�����A�*

	conv_loss6jd>L���        )��P	dL�����A�*

	conv_lossaZc>F�I        )��P	H������A�*

	conv_loss��^>Rs�(        )��P	R�����A�*

	conv_lossSPd>�bw�        )��P	w7�����A�*

	conv_loss3�_>�G�        )��P	�k�����A�*

	conv_lossHa>��zZ        )��P	K������A�*

	conv_loss�4Y>�H        )��P	�������A�*

	conv_loss��e>��8        )��P	:������A�*

	conv_loss�^>_ϸ�        )��P	y,�����A�*

	conv_lossc�O>�±        )��P	#Z�����A�*

	conv_loss��h>��ȸ        )��P	�������A�*

	conv_loss��^>3��        )��P	�������A�*

	conv_loss�\>{��        )��P	 �����A�*

	conv_lossuY>��Go        )��P	�7�����A�*

	conv_loss��c>�c��        )��P	�h�����A�*

	conv_lossN]>`f        )��P	u������A�*

	conv_loss�b`>�.�        )��P	�������A�*

	conv_loss{
W>�3f        )��P	�������A�*

	conv_loss�\>Q��:        )��P	�,�����A�*

	conv_lossNr_>��͖        )��P	�[�����A�*

	conv_lossc_>��M        )��P	�������A�*

	conv_loss�<U>�=�        )��P	������A�*

	conv_lossp�`>���        )��P	�������A�*

	conv_loss�h_>��L        )��P	������A�*

	conv_loss�Z>�}N        )��P	I�����A�*

	conv_loss��_>7�	�        )��P	�x�����A�*

	conv_lossO[>�^��        )��P	�������A�*

	conv_lossSX>H�a        )��P	�������A�*

	conv_loss��b>H��\        )��P	M�����A�*

	conv_loss� b>m���        )��P	�:�����A�*

	conv_loss�-P>J�        )��P	�i�����A�*

	conv_loss��V>��        )��P	>������A�*

	conv_loss�z`>E� �        )��P	�������A�*

	conv_loss7�U>BZj�        )��P	=������A�*

	conv_loss%kd>*N��        )��P	�%�����A�*

	conv_lossD�T>�*^O        )��P	,U�����A�*

	conv_loss�+g>�.Z�        )��P	X������A�*

	conv_lossӨe>.Fa$        )��P	T������A�*

	conv_loss��d>Ry6        )��P	�������A�*

	conv_loss��X>s̤        )��P	������A�*

	conv_loss��]>��	+        )��P	�>�����A�*

	conv_loss��W>:h�        )��P	�o�����A�*

	conv_loss�k>�vLw        )��P	f������A�*

	conv_lossl�^>?'C        )��P	������A�*

	conv_loss"�_>*!_        )��P	�������A�*

	conv_loss�{W>�ݺ�        )��P	�.�����A�*

	conv_loss��W>���        )��P	�\�����A�*

	conv_lossb|_>Q��        )��P	ۊ�����A�*

	conv_lossp�f>٨�        )��P	������A�*

	conv_loss��`>?V�        )��P	�������A�*

	conv_lossc!U>N�        )��P	������A�*

	conv_lossQ�Z>�g�        )��P	�W�����A�*

	conv_loss�Z>�y:�        )��P	'������A�*

	conv_loss��W>e�1        )��P	�������A�*

	conv_loss��S>��N        )��P	B������A�*

	conv_loss�D[>��U        )��P	 �����A�*

	conv_losso�S>`X��        )��P	I�����A�*

	conv_loss��W>[%J�        )��P	�y�����A�*

	conv_lossކe>�+        )��P	�������A�*

	conv_loss��a>�Sws        )��P	�������A�*

	conv_loss8�S>���        )��P	!�����A�*

	conv_loss;dW>H ��        )��P	�L�����A�*

	conv_loss� [>ë�        )��P	|�����A�*

	conv_loss��Y>'qI        )��P	������A�*

	conv_loss��`>?�C        )��P	�������A�*

	conv_lossz�[>n�Ѣ        )��P	;:�����A�*

	conv_loss2�W>!K,        )��P	.l�����A�*

	conv_lossіV>r�7O        )��P	������A�*

	conv_loss�{L>$1�        )��P	H������A�*

	conv_loss�f>��-        )��P	q������A�*

	conv_loss
\>�a��        )��P	w)�����A�*

	conv_lossґV>��M        )��P	�X�����A�*

	conv_loss'�Q>[��        )��P	S������A�*

	conv_loss�Y>zqm6        )��P	������A�*

	conv_loss�X>�C�5        )��P	Z������A�*

	conv_lossR~b>O@�T        )��P	A�����A�*

	conv_loss��]>���I        )��P	�M�����A�*

	conv_loss��S>.�,c        )��P	�������A�*

	conv_loss{O^>�v�        )��P	�������A�*

	conv_loss�(^>鋓�        )��P	�������A�*

	conv_loss�:N>_S        )��P	_+�����A�*

	conv_loss��]>+,[�        )��P	8Y�����A�*

	conv_loss�:]>��<�        )��P	������A�*

	conv_loss�V>|2Y�        )��P	�������A�*

	conv_loss�	[>\g�        )��P	x������A�*

	conv_loss��Q>�S�R        )��P	������A�*

	conv_loss~[>p��        )��P	F�����A�*

	conv_lossCL>:n%        )��P	�s�����A�*

	conv_loss"�X>H�{�        )��P	c������A�*

	conv_loss��S>����        )��P	�������A�*

	conv_loss�Q[>u���        )��P	, ����A�*

	conv_loss�k>�hԌ        )��P	^/ ����A�*

	conv_lossHV>�� �        )��P	�^ ����A�*

	conv_loss4Q>%��        )��P	� ����A�*

	conv_loss�\>�6�        )��P	�� ����A�*

	conv_loss��I>�V��        )��P	(� ����A�*

	conv_loss��X>֜�#        )��P	v����A�*

	conv_loss1xI>�'@        )��P	I����A�*

	conv_loss:f>U�<9        )��P	�y����A�*

	conv_loss��Y>��F�        )��P	������A�*

	conv_losso_>[I��        )��P	#�����A�*

	conv_loss�U>P���        )��P	�7����A�*

	conv_lossYnZ>|`��        )��P	8f����A�*

	conv_lossݚZ>"̕        )��P	������A�*

	conv_lossYV>v�O        )��P	������A�*

	conv_loss�PM>�W.        )��P	'�����A�*

	conv_lossC�W>'<$�        )��P	�1����A�*

	conv_loss��J>�9
�        )��P	�`����A�*

	conv_loss��Z>��n�        )��P	������A�*

	conv_loss��R>���U        )��P	K�����A�*

	conv_loss�]^>�d$8        )��P	�	����A�*

	conv_loss��U>��        )��P	bA����A�*

	conv_loss�,b>���        )��P	+r����A�*

	conv_loss)dV>b�Ň        )��P	�����A�*

	conv_loss��e>�&�        )��P	������A�*

	conv_lossfrd>i��6        )��P	 	����A�*

	conv_loss}>Y>�i��        )��P	�.	����A�*

	conv_loss>9R>��V        )��P	&]	����A�*

	conv_loss��]>4)�        )��P	�	����A�*

	conv_loss=�W>߻h�        )��P	5�	����A�*

	conv_loss��X>�u        )��P	��	����A�*

	conv_loss�dK>�R's        )��P	�
����A�*

	conv_loss��O>x�%        )��P	�T
����A�*

	conv_lossZ_>n���        )��P	�
����A�*

	conv_losspX>��Ӫ        )��P	��
����A�*

	conv_loss6Q>�#��        )��P	a�
����A�*

	conv_loss&�Y>�;#        )��P	 "����A�*

	conv_loss�U>�x(>        )��P	�P����A�*

	conv_loss��O>Z�,�        )��P	�����A�*

	conv_lossfL>�>�)        )��P	R�����A�*

	conv_lossI�I>T��        )��P	������A�*

	conv_loss5�X>��;�        )��P	8����A�*

	conv_loss"dU>�~t�        )��P	�>����A�*

	conv_lossAT>�5��        )��P	<n����A�*

	conv_loss�W>�͞�        )��P	[�����A�*

	conv_lossCLW>�Ӻ�        )��P	�����A�*

	conv_loss�_Y>�WL�        )��P	������A�*

	conv_lossd�Q>���$        )��P	-����A�*

	conv_lossU>���(        )��P	�\����A�*

	conv_loss�U>3�6�        )��P	�����A�*

	conv_loss�P>wɭ�        )��P	9�����A�*

	conv_loss��]>�&N        )��P	X�����A�*

	conv_lossN�c>�8�        )��P	j����A�*

	conv_lossrbh>#P�        )��P	�J����A�*

	conv_loss2P>I �t        )��P	�y����A�*

	conv_loss�7V>lz�7        )��P	I�����A�*

	conv_lossJV>
���        )��P	@�����A�*

	conv_lossu�^>0�f|        )��P	O����A�*

	conv_loss��M>�T-        )��P	�4����A�*

	conv_loss�(L>X?�$        )��P	;e����A�*

	conv_loss��S>Y��        )��P	�����A�*

	conv_loss�vQ>����        )��P	������A�*

	conv_lossQ�P> :��        )��P	� ����A�*

	conv_loss]P>Zw�8        )��P	q/����A�*

	conv_loss�R> ��$        )��P	�]����A�*

	conv_loss��i>F�6n        )��P	�����A�*

	conv_loss!S_>x�,z        )��P	/�����A�*

	conv_loss]�T>(��        )��P	������A�*

	conv_loss+OY>]ޣ�        )��P	����A�*

	conv_loss�Z[>��ɖ        )��P	N����A�*

	conv_loss�8Y>GŖ        )��P	������A�*

	conv_loss�L>����        )��P	.�����A�*

	conv_losss�Q>�!�        )��P	�����A�*

	conv_lossQMY>K	�}        )��P	?E����A�*

	conv_lossRdW>b���        )��P	������A�*

	conv_loss��P>��        )��P	������A�*

	conv_loss��J>����        )��P	Y�����A�*

	conv_lossw�\>��*�        )��P	9����A�*

	conv_loss��M>��Vp        )��P	EB����A�*

	conv_loss��b>��y�        )��P	�q����A�*

	conv_loss'LQ>�?��        )��P	4�����A�*

	conv_losse�\>/VK        )��P	A�����A�*

	conv_loss�mS>���3        )��P	n����A�*

	conv_loss��[>��F        )��P	�7����A�*

	conv_loss.QR>@�H        )��P		n����A�*

	conv_loss��M>0�)        )��P	Ƞ����A�*

	conv_loss��Z>hY^        )��P	������A�*

	conv_lossI	V>!zJ        )��P	� ����A�*

	conv_loss^h>�G        )��P	�/����A�*

	conv_loss�P>�G[        )��P	�_����A�*

	conv_lossSZH>���D        )��P	������A�*

	conv_losseHY>10M,        )��P	u�����A�*

	conv_loss��R>D:��        )��P	�����A�*

	conv_loss�CL>h�0        )��P	�����A�*

	conv_loss��R>6��O        )��P	�J����A�*

	conv_lossXe>e)        )��P	y����A�*

	conv_loss��R>M��        )��P	�����A�*

	conv_loss�QW>����        )��P	������A�*

	conv_loss�QN>�}��        )��P	����A�*

	conv_loss��U>;��        )��P	�6����A�*

	conv_loss��V>L�c        )��P	�f����A�*

	conv_loss��]>���        )��P	r�����A�*

	conv_loss �T>�v8O        )��P	������A�*

	conv_loss�T>�-�        )��P	������A�*

	conv_loss�3W>��6,        )��P	�$����A�*

	conv_loss?N>3�        )��P	�T����A�*

	conv_loss;R>#��N        )��P	�����A�*

	conv_loss��U>
ml        )��P	�����A�*

	conv_loss�K>m��<        )��P	������A�*

	conv_loss�c>�q/W        )��P	�����A�*

	conv_loss�K>�z#�        )��P	<?����A�*

	conv_loss�M>F��        )��P	s�����A�*

	conv_loss=�Y>��U%        )��P	������A�*

	conv_loss��W>Х
        )��P	�,����A�*

	conv_loss�'R>�_P        )��P	�_����A�*

	conv_lossWQ>)@�?        )��P	b�����A�*

	conv_lossDPR>�|��        )��P	K�����A�*

	conv_lossϝK>��d        )��P	�
����A�*

	conv_loss�R>v�c        )��P	�:����A�*

	conv_loss�vR>���        )��P	!r����A�*

	conv_loss��O>�        )��P	������A�*

	conv_lossU>*F�{        )��P	������A�*

	conv_lossUS>� �        )��P	�����A�*

	conv_loss��L>*�Az        )��P	!B����A�*

	conv_lossW�G>�)sp        )��P	q����A�*

	conv_lossmN>�?�(        )��P	o�����A�*

	conv_loss)�N>�q=�        )��P	������A�*

	conv_losssN>����        )��P	������A�*

	conv_lossEl_>� ��        )��P	.����A�*

	conv_loss�cQ>��{        )��P	�e����A�*

	conv_lossT%T>L��        )��P	������A�*

	conv_lossB�T>l81�        )��P	������A�*

	conv_loss��L>����        )��P	J����A�*

	conv_lossr�W>2h+t        )��P	>3����A�*

	conv_loss0ST>� ��        )��P	d����A�*

	conv_loss.�S>�G|>        )��P	�����A�*

	conv_loss�yG>��1�        )��P	������A�*

	conv_lossP�K>��!        )��P	D�����A�*

	conv_loss�J>C��        )��P	" ����A�*

	conv_lossqX>�U��        )��P	�Q ����A�*

	conv_loss�wY>hT�^        )��P	�� ����A�*

	conv_lossN�N>�}f        )��P	ܳ ����A�*

	conv_lossïQ>K�Ҏ        )��P	�� ����A�*

	conv_loss�cR>xW�        )��P	�!����A�*

	conv_loss��U>ͥK        )��P	�A!����A�*

	conv_loss&T>c�        )��P	�q!����A�*

	conv_loss�`S>�,��        )��P	J�!����A�*

	conv_loss�6H>����        )��P	?�!����A�*

	conv_lossjL>�~	�        )��P	M�!����A�*

	conv_loss?[W>��"        )��P	�,"����A�*

	conv_loss�M>-�AA        )��P	x["����A�*

	conv_loss�N>���        )��P	{�"����A�*

	conv_loss'W>���~        )��P	.�"����A�*

	conv_loss��U>�ҿ        )��P	m�"����A�*

	conv_loss�P>8�        )��P	�#����A�*

	conv_loss��T>��f        )��P	 K#����A�*

	conv_loss8�\>���        )��P	�z#����A�*

	conv_loss�O\>�ب        )��P	X�#����A�*

	conv_loss�W>��        )��P	��#����A�*

	conv_lossj�P>�L��        )��P	�$����A�*

	conv_loss�HQ>�#��        )��P	�?$����A�*

	conv_lossޯQ>�($�        )��P	@~$����A�*

	conv_loss�EX>�r�        )��P	I�$����A�*

	conv_loss��H>#� @        )��P	��$����A�*

	conv_loss�(S>5��        )��P	n%����A�*

	conv_loss�;S>���        )��P	�M%����A�*

	conv_lossU�]>��L\        )��P	�|%����A�*

	conv_lossK�Q>�N$�        )��P	��%����A�*

	conv_lossxF>�W�J        )��P	��%����A�*

	conv_loss::D>t��         )��P	�'&����A�*

	conv_loss@�N>(3y�        )��P	�Y&����A�*

	conv_lossZJ>	l        )��P	!�&����A�*

	conv_loss��]>b*J        )��P	��&����A�*

	conv_loss�Q>x�~        )��P	(�&����A�*

	conv_lossh�L>�!�        )��P	V!'����A�*

	conv_lossÅL>$oAL        )��P	BP'����A�*

	conv_loss)&I>��        )��P	~'����A�*

	conv_loss6�V>!u.3        )��P	�'����A�*

	conv_loss�@[>+*U        )��P	b�'����A�*

	conv_loss^�S>uZ�        )��P	�
(����A�*

	conv_loss��T>٭�t        )��P	�:(����A�*

	conv_loss=OY>(o:�        )��P	 p(����A�*

	conv_loss nZ>k�X�        )��P	�(����A�*

	conv_lossebV>�F�&        )��P	�(����A�*

	conv_loss��]>oh�        )��P	�)����A�*

	conv_lossrV>�\��        )��P	�6)����A�*

	conv_losst#W>��P        )��P	�e)����A�*

	conv_loss��G>�駬        )��P	Օ)����A�*

	conv_loss�BU>��+�        )��P	[�)����A�*

	conv_loss	zM><Z�        )��P	��)����A�*

	conv_loss�GN>���)        )��P	+#*����A�*

	conv_loss�gF>\{��        )��P	�R*����A�*

	conv_lossw�P>nO�        )��P	=�*����A�*

	conv_loss��P>ϐ�m        )��P	)�*����A�*

	conv_loss�;T>jxA�        )��P	V�*����A�*

	conv_loss�.X>��        )��P	{+����A�*

	conv_loss��^>iF�y        )��P	Q@+����A�*

	conv_loss�?D>�bH        )��P	cn+����A�*

	conv_loss%�F>pB        )��P	�+����A�*

	conv_loss~�L>.�2�        )��P	{�+����A�*

	conv_loss��T>��d        )��P	��+����A�*

	conv_loss�2H>Cv-�        )��P	�*,����A�*

	conv_loss��T>ʰR        )��P	�Y,����A�*

	conv_loss"*Y>��        )��P	�,����A�*

	conv_loss�J> ��        )��P	ȶ,����A�*

	conv_loss�,V>���        )��P	.�,����A�*

	conv_loss|K>��/        )��P	Y-����A�*

	conv_loss�Q>�5x        )��P	]E-����A�*

	conv_lossiSS>.        )��P	w-����A�*

	conv_loss�|U>cC        )��P	r�-����A�*

	conv_loss�F>et<,        )��P	t�-����A�*

	conv_loss�m_>���        )��P	(.����A�*

	conv_loss��R>��f        )��P	>F.����A�*

	conv_lossDO>���        )��P	ut.����A�*

	conv_lossJ=W>��        )��P	��.����A�*

	conv_loss�4Q>W��C        )��P	 �.����A�*

	conv_loss��J>,J�F        )��P	/����A�*

	conv_loss�^W>�r:        )��P	/L/����A�*

	conv_lossDQ>�hQ        )��P	�z/����A�*

	conv_loss��J>�r-        )��P	y�/����A�*

	conv_lossC�D>-��O        )��P	��/����A�*

	conv_lossD?>�0c        )��P	U0����A�*

	conv_loss�I>���1        )��P	3]0����A�*

	conv_losskHX>�Bu        )��P	��0����A�*

	conv_loss��S>=�S{        )��P	��0����A�*

	conv_lossJ�Y>9��        )��P	��0����A�*

	conv_loss�^G>{�7I        )��P	H1����A�*

	conv_loss�S>0{B        )��P	�L1����A�*

	conv_loss*�A>rR�<        )��P	iz1����A�*

	conv_lossmM>+9(�        )��P	k�1����A�*

	conv_loss��Q>Q|o�        )��P	e�1����A�*

	conv_lossjR>)f��        )��P	V2����A�*

	conv_loss]R>�(4�        )��P	$62����A�*

	conv_loss��N>څn        )��P	ef2����A�*

	conv_loss�U>37Ի        )��P	��2����A�*

	conv_lossSS>�2�        )��P	w�2����A�*

	conv_loss��Q>��        )��P	�3����A�*

	conv_loss��N>���        )��P	�83����A�*

	conv_loss�BN>�o7:        )��P	!g3����A�*

	conv_loss��T>���        )��P	��3����A�*

	conv_loss��V>�»        )��P	��3����A�*

	conv_lossk�J>\n�_        )��P	��3����A�*

	conv_loss<O>���        )��P	�#4����A�*

	conv_lossF�N>� L�        )��P	nS4����A�*

	conv_loss0�O>a��        )��P	��4����A�*

	conv_loss]hX>Ӿk�        )��P	��4����A�*

	conv_loss^oO>m�)        )��P	��4����A�*

	conv_loss��H>�k�        )��P	5����A�*

	conv_loss��J>R��        )��P	A5����A�*

	conv_loss�(H>��p�        )��P	�o5����A�*

	conv_loss��Q>��8        )��P	��5����A�*

	conv_loss�E>oi��        )��P	��5����A�*

	conv_loss�>F>k��5        )��P	K�5����A�*

	conv_loss�1E>o^8        )��P	�.6����A�*

	conv_lossϙF>���        )��P	}]6����A�*

	conv_lossÓT>���9        )��P	w�6����A�*

	conv_loss�K>�z:[        )��P	��6����A�*

	conv_loss!�F>�M        )��P	(�6����A�*

	conv_loss N>� VE        )��P	S7����A�*

	conv_lossGF> ��}        )��P	>H7����A�*

	conv_loss��N>��2�        )��P	tw7����A�*

	conv_lossTU>+�`        )��P	6�7����A�*

	conv_loss]mJ>��x        )��P	��7����A�*

	conv_loss��E>]`K�        )��P	�8����A�*

	conv_loss;M>�z��        )��P	�B8����A�*

	conv_loss�/?>wo        )��P	?w8����A�*

	conv_loss�~A>pQ�9        )��P	H�8����A�*

	conv_loss��M>��<�        )��P	A�8����A�*

	conv_lossY�H>�w�        )��P	�9����A�*

	conv_loss
�J>�f��        )��P	�=9����A�*

	conv_loss�O>��ڈ        )��P	r9����A�*

	conv_loss8R>����        )��P	��9����A�*

	conv_loss?HD>�|��        )��P	o�9����A�*

	conv_loss
O>Fo�        )��P	�:����A�*

	conv_loss�hP>X��h        )��P	�C:����A�*

	conv_loss�.L>�]�        )��P	Jv:����A�*

	conv_lossl8K>=�>�        )��P	k�:����A�*

	conv_loss��G>]��Z        )��P	)�:����A�*

	conv_loss�N>���w        )��P	Q;����A�*

	conv_loss4\S>$��        )��P	�3;����A�*

	conv_loss�iF>(\�        )��P	�b;����A�*

	conv_lossf�J>I�2J        )��P	�;����A�*

	conv_loss��E>]�        )��P	ȿ;����A�*

	conv_losssL>r��Q        )��P	x�;����A�*

	conv_lossiLG>��Y        )��P	<����A�*

	conv_loss�vJ>�R        )��P	N<����A�*

	conv_loss�~@>4�*        )��P	�|<����A�*

	conv_loss"�S>͉�4        )��P	׭<����A�*

	conv_lossW�H>���        )��P	��<����A�*

	conv_loss�G>�k"7        )��P	�=����A�*

	conv_loss��?>����        )��P	E<=����A�*

	conv_loss"U>��{        )��P	�l=����A�*

	conv_loss��F>&|O)        )��P	��=����A�*

	conv_losssT>��u        )��P	��=����A�*

	conv_loss=S>Z��A        )��P	F�=����A�*

	conv_loss�}N>G�8        )��P	)>����A�*

	conv_loss��D>��t~        )��P	�W>����A�*

	conv_lossA�N>�,        )��P	x�>����A�*

	conv_loss�D>�:�%        )��P	��>����A�*

	conv_loss{S>��"        )��P	��>����A�*

	conv_loss0�M>e�Թ        )��P	�?����A�*

	conv_lossR�E>Nt        )��P	!C?����A�*

	conv_lossH�M>�t?�        )��P	Js?����A�*

	conv_loss9:?>Kx��        )��P	��?����A�*

	conv_loss|T>?��        )��P	��?����A�*

	conv_loss��?>�P�t        )��P	� @����A�*

	conv_lossH�V>�TD
        )��P	&.@����A�*

	conv_loss"�R>���        )��P	�\@����A�*

	conv_loss��V>��w4        )��P	;�@����A�*

	conv_loss��G>n��        )��P	{�@����A�*

	conv_loss��O>0��5        )��P	��@����A�*

	conv_loss�L`>q���        )��P	�vB����A�*

	conv_loss��>>�%%�        )��P	��B����A�*

	conv_lossʒR>s��[        )��P	�B����A�*

	conv_loss9�H>y���        )��P	C����A�*

	conv_loss�G>{E        )��P	�8C����A�*

	conv_lossεI>����        )��P	�hC����A�*

	conv_loss��^>э�        )��P	d�C����A�*

	conv_loss��R>�U�        )��P	��C����A�*

	conv_loss�*Q>m��        )��P	R�C����A�*

	conv_loss3:L>��bv        )��P	#:D����A�*

	conv_lossW�6>O���        )��P	�rD����A�*

	conv_loss�I>~��        )��P	��D����A�*

	conv_loss�ES>`J�        )��P	��D����A�*

	conv_loss�H>v���        )��P	FE����A�*

	conv_lossP B>��>�        )��P	0?E����A�*

	conv_loss#H>b.`�        )��P	EmE����A�*

	conv_loss�~E>�
0i        )��P	P�E����A�*

	conv_loss�U>�g�_        )��P	��E����A�*

	conv_loss�W>���*        )��P	W�E����A�*

	conv_loss��I>n��$        )��P	&F����A�*

	conv_loss�M>pF/�        )��P	1UF����A�*

	conv_loss �?>��        )��P	[�F����A�*

	conv_loss( H>��\�        )��P	�F����A�*

	conv_loss�K>��e        )��P	��F����A�*

	conv_loss��H>)xB,        )��P	3.G����A�*

	conv_lossN�M>���        )��P	�^G����A�*

	conv_loss/L>,��        )��P	��G����A�*

	conv_loss�5C>:N��        )��P	��G����A�*

	conv_loss��J>s��        )��P	!�G����A�*

	conv_loss'�K>~�-E        )��P	tH����A�*

	conv_loss�^G>w��        )��P	>JH����A�*

	conv_lossmyH>l��!        )��P	�xH����A�*

	conv_loss��F>|���        )��P	اH����A�*

	conv_lossg�G>��):        )��P	�H����A�*

	conv_lossU$C>���        )��P	SI����A�*

	conv_lossgU>I���        )��P	D6I����A�*

	conv_loss�C>&W�        )��P	�eI����A�*

	conv_loss�S>)/��        )��P	�I����A�*

	conv_loss��D>��`�        )��P	��I����A�*

	conv_loss��I>�m<        )��P	��I����A�*

	conv_loss�S?>.5B#        )��P	�$J����A�*

	conv_loss%�G>��B�        )��P	UJ����A�*

	conv_loss4�N>�>�m        )��P	��J����A�*

	conv_lossA9?>��!)        )��P	!�J����A�*

	conv_loss�{G>��s�        )��P	n�J����A�*

	conv_loss��J>�O^        )��P	5K����A�*

	conv_loss��I>m��        )��P	�@K����A�*

	conv_loss�D6>9qg�        )��P	qK����A�*

	conv_loss��5>��y�        )��P	��K����A�*

	conv_lossYP@>���        )��P	;�K����A�*

	conv_loss͸L>/�        )��P	Q"L����A�*

	conv_loss�SE>�=ZM        )��P	�PL����A�*

	conv_lossǴH>E�-�        )��P	�L����A�*

	conv_loss*"I>����        )��P	�L����A�*

	conv_loss�	T>��y        )��P	8�L����A�*

	conv_loss&3W>�M�q        )��P	�M����A�*

	conv_loss�K>���        )��P	�KM����A�*

	conv_loss�F>�8��        )��P	LzM����A�*

	conv_lossfVI>���"        )��P	(�M����A�*

	conv_loss2�D>��j�        )��P	/�M����A�*

	conv_loss�;>�#aR        )��P	?N����A�*

	conv_loss*
K>`��        )��P	]YN����A�*

	conv_loss/ D>x��        )��P	��N����A�*

	conv_lossX�E>�vx        )��P	u�N����A�*

	conv_losscI>���n        )��P	��N����A�*

	conv_loss�A>�;�        )��P	� O����A�*

	conv_losss�@>OS�        )��P	zPO����A�*

	conv_lossE H>���        )��P	�O����A�*

	conv_loss�8>�L��        )��P	u�O����A�*

	conv_loss��L>���>        )��P	D�O����A�*

	conv_lossc�Q>�qp�        )��P	�P����A�*

	conv_loss&>>u�X.        )��P	.<P����A�*

	conv_lossxbF>7�\�        )��P	�jP����A�*

	conv_loss�D><b��        )��P	�P����A�*

	conv_loss��N>�=&�        )��P	��P����A�*

	conv_loss�gC>ۓ*m        )��P	F�P����A�*

	conv_lossX�@>���        )��P	Q&Q����A�*

	conv_lossw�E>�$ǯ        )��P	�XQ����A�*

	conv_lossN�R>F�        )��P	E�Q����A�*

	conv_lossWXG>�8�        )��P	V�Q����A�*

	conv_loss�BD>ߕ�        )��P	��Q����A�*

	conv_loss�D>l�J        )��P	vR����A�*

	conv_loss��J>����        )��P	GR����A�*

	conv_loss�B>��v        )��P	�uR����A�*

	conv_loss�	=>����        )��P	ɤR����A�*

	conv_loss��F>F�|�        )��P	��R����A�*

	conv_loss�@>$rk        )��P	iS����A�*

	conv_loss�>>�]�m        )��P	U0S����A�*

	conv_loss��?>K)k~        )��P	�_S����A�*

	conv_loss'�O>�>�K        )��P	_�S����A�*

	conv_lossl�M>��        )��P	��S����A�*

	conv_loss�xF>,�        )��P	X�S����A�*

	conv_loss�D>ʨ�        )��P	1T����A�*

	conv_loss/{G>FK�        )��P	�KT����A�*

	conv_loss�B>n/PQ        )��P	�zT����A�*

	conv_loss�JA>�v#�        )��P	�T����A�*

	conv_lossW=B>o�        )��P	��T����A�*

	conv_loss� L>�A�E        )��P	�U����A�*

	conv_loss`H>��&        )��P	�<U����A�*

	conv_loss�}K>AT?.        )��P	]jU����A�*

	conv_loss�J>1��        )��P	1�U����A�*

	conv_loss��G>��?X        )��P	��U����A�*

	conv_lossD�>>}S1}        )��P	`
V����A�*

	conv_loss��Z>���        )��P	�9V����A�*

	conv_loss)C>;�+�        )��P	�pV����A�*

	conv_loss}�G>l�z        )��P	1�V����A�*

	conv_loss{�@>��ټ        )��P	��V����A�*

	conv_loss� D>M�Sg        )��P	��V����A�*

	conv_loss$A9>�5�        )��P	�-W����A�*

	conv_loss��J> <�        )��P	�`W����A�*

	conv_loss�K@>k��        )��P	��W����A�*

	conv_lossE�<>?Qct        )��P	�W����A�*

	conv_loss�3>>���\        )��P	e�W����A�*

	conv_loss�)O>���        )��P	�(X����A�*

	conv_loss�OF>:XE!        )��P	M]X����A�*

	conv_loss]e=>�ִV        )��P	�X����A�*

	conv_loss��J>��=2        )��P	��X����A�*

	conv_loss��B>���        )��P	
�X����A�*

	conv_loss�[C>�W�        )��P	$!Y����A�*

	conv_loss9>}z\�        )��P	~QY����A�*

	conv_loss�i4>&�GU        )��P	=�Y����A�*

	conv_loss��G>�p�        )��P	a�Y����A�*

	conv_lossJH=>a�P�        )��P	��Y����A�*

	conv_loss��H>�y�        )��P	=Z����A�*

	conv_loss��A>��Y        )��P	q;Z����A�*

	conv_loss��U>�n��        )��P	�iZ����A�*

	conv_loss0FQ>��"D        )��P	�Z����A�*

	conv_loss��J>���K        )��P	p�Z����A�*

	conv_lossM�F>��        )��P	j�Z����A�*

	conv_loss�0>>�� �        )��P	�*[����A�*

	conv_lossEF>�B��        )��P	Z[����A�*

	conv_loss(�S>�\�        )��P	�[����A�*

	conv_lossW=I>�	��        )��P	��[����A�*

	conv_loss~�K>3M�        )��P	S�[����A�*

	conv_loss^8>5��        )��P	L\����A�*

	conv_loss[!C>��]        )��P	xG\����A�*

	conv_loss�A6>rdE        )��P	��\����A�*

	conv_loss��K>��f�        )��P	#�\����A�*

	conv_lossg/:>���        )��P	��\����A�*

	conv_lossiOG>�        )��P	�]����A�*

	conv_loss�C>e�S�        )��P	HD]����A�*

	conv_loss��L>��        )��P	8s]����A�*

	conv_loss $A>Л6�        )��P	N�]����A�*

	conv_loss%Z@>p�Ew        )��P	��]����A�*

	conv_loss��J>yA��        )��P	  ^����A�*

	conv_loss6�L>���        )��P	.^����A�*

	conv_loss��B>��P9        )��P	G]^����A�*

	conv_loss�wA>�D�Y        )��P	��^����A�*

	conv_loss�V<>����        )��P	;�^����A�*

	conv_lossP^F>���        )��P	��^����A�*

	conv_loss�|D>Qe�l        )��P	B(_����A�*

	conv_loss�E>�6�>        )��P	�W_����A�*

	conv_loss��L>.ˁ�        )��P	E�_����A�*

	conv_loss6�@>�0@x        )��P	1�_����A�*

	conv_loss<�F>+s��        )��P	��_����A�*

	conv_loss�K>�U�c        )��P	M`����A�*

	conv_lossb1E>Eo2        )��P	gJ`����A�*

	conv_lossI�>>|��d        )��P	Uy`����A�*

	conv_loss��K>qk        )��P	��`����A�*

	conv_loss��B>�]
        )��P	�`����A�*

	conv_lossSB>�J��        )��P	|a����A�*

	conv_loss=�;>�>�        )��P	6>a����A�*

	conv_loss }I>yG��        )��P	1la����A�*

	conv_loss ,?>o�~}        )��P	��a����A�*

	conv_lossg�E>��=N        )��P	�a����A�*

	conv_loss�oN>Lq��        )��P	�b����A�*

	conv_loss H>�©        )��P	�4b����A�*

	conv_loss�=I>�<�        )��P	�jb����A�*

	conv_loss+�D>v��W        )��P	��b����A�*

	conv_loss�I>��Op        )��P	/�b����A�*

	conv_loss5M>/t�        )��P	�c����A�*

	conv_loss�q?>v���        )��P	�4c����A�*

	conv_loss��>>��V�        )��P	cc����A�*

	conv_loss�0E>���/        )��P	��c����A�*

	conv_loss|B>��Q�        )��P	�c����A�*

	conv_loss�CC>�	]�        )��P	��c����A�*

	conv_loss�F>��        )��P	0d����A�*

	conv_loss!uF>�-��        )��P	�Md����A�*

	conv_loss�*D>A� �        )��P	}d����A�*

	conv_loss��K>�֣W        )��P	�d����A�*

	conv_lossQ{E>�pjh        )��P	��d����A�*

	conv_lossoZP>g ��        )��P	�e����A�*

	conv_lossi�I>��3�        )��P	�:e����A�*

	conv_loss��B>��        )��P	je����A�*

	conv_lossDF>>^u1�        )��P	��e����A�*

	conv_loss�wE>�3�        )��P	��e����A�*

	conv_loss�C>^�J        )��P	��e����A�*

	conv_loss�e9>��|>        )��P	�%f����A�*

	conv_loss?5>e�gT        )��P	�Uf����A�*

	conv_loss�tG>a�U        )��P	�f����A�*

	conv_loss�O>���K        )��P	�f����A�*

	conv_loss��4>��        )��P	1�f����A�*

	conv_loss�D>Ӓ��        )��P	6g����A�*

	conv_loss��>>K�        )��P	"Bg����A�*

	conv_loss��G>Rq�        )��P	�qg����A�*

	conv_losslB>_M        )��P	ơg����A�*

	conv_loss��0>�4        )��P	��g����A�*

	conv_lossHnI>�-�        )��P	N�g����A�*

	conv_loss9^D>дq        )��P	�.h����A�*

	conv_loss�G>�-!