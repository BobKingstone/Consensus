нь
ЪЃ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.12unknown8Ј▀
њ
deep_q_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,X*,
shared_namedeep_q_network/dense/kernel
І
/deep_q_network/dense/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network/dense/kernel*
_output_shapes

:,X*
dtype0
і
deep_q_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:X**
shared_namedeep_q_network/dense/bias
Ѓ
-deep_q_network/dense/bias/Read/ReadVariableOpReadVariableOpdeep_q_network/dense/bias*
_output_shapes
:X*
dtype0
Ќ
deep_q_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	X░*.
shared_namedeep_q_network/dense_1/kernel
љ
1deep_q_network/dense_1/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_1/kernel*
_output_shapes
:	X░*
dtype0
Ј
deep_q_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:░*,
shared_namedeep_q_network/dense_1/bias
ѕ
/deep_q_network/dense_1/bias/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_1/bias*
_output_shapes	
:░*
dtype0
Ќ
deep_q_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	░X*.
shared_namedeep_q_network/dense_2/kernel
љ
1deep_q_network/dense_2/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_2/kernel*
_output_shapes
:	░X*
dtype0
ј
deep_q_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*,
shared_namedeep_q_network/dense_2/bias
Є
/deep_q_network/dense_2/bias/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_2/bias*
_output_shapes
:X*
dtype0
ќ
deep_q_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:X*.
shared_namedeep_q_network/dense_3/kernel
Ј
1deep_q_network/dense_3/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_3/kernel*
_output_shapes

:X*
dtype0
ј
deep_q_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namedeep_q_network/dense_3/bias
Є
/deep_q_network/dense_3/bias/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
а
"Adam/deep_q_network/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,X*3
shared_name$"Adam/deep_q_network/dense/kernel/m
Ў
6Adam/deep_q_network/dense/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/deep_q_network/dense/kernel/m*
_output_shapes

:,X*
dtype0
ў
 Adam/deep_q_network/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*1
shared_name" Adam/deep_q_network/dense/bias/m
Љ
4Adam/deep_q_network/dense/bias/m/Read/ReadVariableOpReadVariableOp Adam/deep_q_network/dense/bias/m*
_output_shapes
:X*
dtype0
Ц
$Adam/deep_q_network/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	X░*5
shared_name&$Adam/deep_q_network/dense_1/kernel/m
ъ
8Adam/deep_q_network/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/deep_q_network/dense_1/kernel/m*
_output_shapes
:	X░*
dtype0
Ю
"Adam/deep_q_network/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:░*3
shared_name$"Adam/deep_q_network/dense_1/bias/m
ќ
6Adam/deep_q_network/dense_1/bias/m/Read/ReadVariableOpReadVariableOp"Adam/deep_q_network/dense_1/bias/m*
_output_shapes	
:░*
dtype0
Ц
$Adam/deep_q_network/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	░X*5
shared_name&$Adam/deep_q_network/dense_2/kernel/m
ъ
8Adam/deep_q_network/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/deep_q_network/dense_2/kernel/m*
_output_shapes
:	░X*
dtype0
ю
"Adam/deep_q_network/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*3
shared_name$"Adam/deep_q_network/dense_2/bias/m
Ћ
6Adam/deep_q_network/dense_2/bias/m/Read/ReadVariableOpReadVariableOp"Adam/deep_q_network/dense_2/bias/m*
_output_shapes
:X*
dtype0
ц
$Adam/deep_q_network/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:X*5
shared_name&$Adam/deep_q_network/dense_3/kernel/m
Ю
8Adam/deep_q_network/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/deep_q_network/dense_3/kernel/m*
_output_shapes

:X*
dtype0
ю
"Adam/deep_q_network/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/deep_q_network/dense_3/bias/m
Ћ
6Adam/deep_q_network/dense_3/bias/m/Read/ReadVariableOpReadVariableOp"Adam/deep_q_network/dense_3/bias/m*
_output_shapes
:*
dtype0
а
"Adam/deep_q_network/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:,X*3
shared_name$"Adam/deep_q_network/dense/kernel/v
Ў
6Adam/deep_q_network/dense/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/deep_q_network/dense/kernel/v*
_output_shapes

:,X*
dtype0
ў
 Adam/deep_q_network/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*1
shared_name" Adam/deep_q_network/dense/bias/v
Љ
4Adam/deep_q_network/dense/bias/v/Read/ReadVariableOpReadVariableOp Adam/deep_q_network/dense/bias/v*
_output_shapes
:X*
dtype0
Ц
$Adam/deep_q_network/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	X░*5
shared_name&$Adam/deep_q_network/dense_1/kernel/v
ъ
8Adam/deep_q_network/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/deep_q_network/dense_1/kernel/v*
_output_shapes
:	X░*
dtype0
Ю
"Adam/deep_q_network/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:░*3
shared_name$"Adam/deep_q_network/dense_1/bias/v
ќ
6Adam/deep_q_network/dense_1/bias/v/Read/ReadVariableOpReadVariableOp"Adam/deep_q_network/dense_1/bias/v*
_output_shapes	
:░*
dtype0
Ц
$Adam/deep_q_network/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	░X*5
shared_name&$Adam/deep_q_network/dense_2/kernel/v
ъ
8Adam/deep_q_network/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/deep_q_network/dense_2/kernel/v*
_output_shapes
:	░X*
dtype0
ю
"Adam/deep_q_network/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*3
shared_name$"Adam/deep_q_network/dense_2/bias/v
Ћ
6Adam/deep_q_network/dense_2/bias/v/Read/ReadVariableOpReadVariableOp"Adam/deep_q_network/dense_2/bias/v*
_output_shapes
:X*
dtype0
ц
$Adam/deep_q_network/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:X*5
shared_name&$Adam/deep_q_network/dense_3/kernel/v
Ю
8Adam/deep_q_network/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/deep_q_network/dense_3/kernel/v*
_output_shapes

:X*
dtype0
ю
"Adam/deep_q_network/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/deep_q_network/dense_3/bias/v
Ћ
6Adam/deep_q_network/dense_3/bias/v/Read/ReadVariableOpReadVariableOp"Adam/deep_q_network/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
­*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ф*
valueА*Bъ* BЌ*
А

dense1

dense2

dense3

dense4
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
л
#iter

$beta_1

%beta_2
	&decay
'learning_ratemFmGmHmImJmKmLmMvNvOvPvQvRvSvTvU
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
Г

(layers
)metrics
trainable_variables
	variables
*layer_metrics
+non_trainable_variables
,layer_regularization_losses
regularization_losses
 
YW
VARIABLE_VALUEdeep_q_network/dense/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdeep_q_network/dense/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Г

-layers
.metrics
trainable_variables
	variables
/layer_metrics
0non_trainable_variables
1layer_regularization_losses
regularization_losses
[Y
VARIABLE_VALUEdeep_q_network/dense_1/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdeep_q_network/dense_1/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Г

2layers
3metrics
trainable_variables
	variables
4layer_metrics
5non_trainable_variables
6layer_regularization_losses
regularization_losses
[Y
VARIABLE_VALUEdeep_q_network/dense_2/kernel(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdeep_q_network/dense_2/bias&dense3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Г

7layers
8metrics
trainable_variables
	variables
9layer_metrics
:non_trainable_variables
;layer_regularization_losses
regularization_losses
[Y
VARIABLE_VALUEdeep_q_network/dense_3/kernel(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdeep_q_network/dense_3/bias&dense4/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Г

<layers
=metrics
trainable_variables
 	variables
>layer_metrics
?non_trainable_variables
@layer_regularization_losses
!regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

A0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Btotal
	Ccount
D	variables
E	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

D	variables
|z
VARIABLE_VALUE"Adam/deep_q_network/dense/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/deep_q_network/dense/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/deep_q_network/dense_1/kernel/mDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/deep_q_network/dense_1/bias/mBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/deep_q_network/dense_2/kernel/mDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/deep_q_network/dense_2/bias/mBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/deep_q_network/dense_3/kernel/mDdense4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/deep_q_network/dense_3/bias/mBdense4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE"Adam/deep_q_network/dense/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/deep_q_network/dense/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/deep_q_network/dense_1/kernel/vDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/deep_q_network/dense_1/bias/vBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/deep_q_network/dense_2/kernel/vDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/deep_q_network/dense_2/bias/vBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/deep_q_network/dense_3/kernel/vDdense4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/deep_q_network/dense_3/bias/vBdense4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:         ,*
dtype0*
shape:         ,
»
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1deep_q_network/dense/kerneldeep_q_network/dense/biasdeep_q_network/dense_1/kerneldeep_q_network/dense_1/biasdeep_q_network/dense_2/kerneldeep_q_network/dense_2/biasdeep_q_network/dense_3/kerneldeep_q_network/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *.
f)R'
%__inference_signature_wrapper_8749560
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
«
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/deep_q_network/dense/kernel/Read/ReadVariableOp-deep_q_network/dense/bias/Read/ReadVariableOp1deep_q_network/dense_1/kernel/Read/ReadVariableOp/deep_q_network/dense_1/bias/Read/ReadVariableOp1deep_q_network/dense_2/kernel/Read/ReadVariableOp/deep_q_network/dense_2/bias/Read/ReadVariableOp1deep_q_network/dense_3/kernel/Read/ReadVariableOp/deep_q_network/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp6Adam/deep_q_network/dense/kernel/m/Read/ReadVariableOp4Adam/deep_q_network/dense/bias/m/Read/ReadVariableOp8Adam/deep_q_network/dense_1/kernel/m/Read/ReadVariableOp6Adam/deep_q_network/dense_1/bias/m/Read/ReadVariableOp8Adam/deep_q_network/dense_2/kernel/m/Read/ReadVariableOp6Adam/deep_q_network/dense_2/bias/m/Read/ReadVariableOp8Adam/deep_q_network/dense_3/kernel/m/Read/ReadVariableOp6Adam/deep_q_network/dense_3/bias/m/Read/ReadVariableOp6Adam/deep_q_network/dense/kernel/v/Read/ReadVariableOp4Adam/deep_q_network/dense/bias/v/Read/ReadVariableOp8Adam/deep_q_network/dense_1/kernel/v/Read/ReadVariableOp6Adam/deep_q_network/dense_1/bias/v/Read/ReadVariableOp8Adam/deep_q_network/dense_2/kernel/v/Read/ReadVariableOp6Adam/deep_q_network/dense_2/bias/v/Read/ReadVariableOp8Adam/deep_q_network/dense_3/kernel/v/Read/ReadVariableOp6Adam/deep_q_network/dense_3/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__traced_save_8749963
й	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedeep_q_network/dense/kerneldeep_q_network/dense/biasdeep_q_network/dense_1/kerneldeep_q_network/dense_1/biasdeep_q_network/dense_2/kerneldeep_q_network/dense_2/biasdeep_q_network/dense_3/kerneldeep_q_network/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount"Adam/deep_q_network/dense/kernel/m Adam/deep_q_network/dense/bias/m$Adam/deep_q_network/dense_1/kernel/m"Adam/deep_q_network/dense_1/bias/m$Adam/deep_q_network/dense_2/kernel/m"Adam/deep_q_network/dense_2/bias/m$Adam/deep_q_network/dense_3/kernel/m"Adam/deep_q_network/dense_3/bias/m"Adam/deep_q_network/dense/kernel/v Adam/deep_q_network/dense/bias/v$Adam/deep_q_network/dense_1/kernel/v"Adam/deep_q_network/dense_1/bias/v$Adam/deep_q_network/dense_2/kernel/v"Adam/deep_q_network/dense_2/bias/v$Adam/deep_q_network/dense_3/kernel/v"Adam/deep_q_network/dense_3/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference__traced_restore_8750066њЛ
њ	
П
D__inference_dense_3_layer_call_and_return_conditional_losses_8749397

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         X::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         X
 
_user_specified_nameinputs
╩є
╦
#__inference__traced_restore_8750066
file_prefix0
,assignvariableop_deep_q_network_dense_kernel0
,assignvariableop_1_deep_q_network_dense_bias4
0assignvariableop_2_deep_q_network_dense_1_kernel2
.assignvariableop_3_deep_q_network_dense_1_bias4
0assignvariableop_4_deep_q_network_dense_2_kernel2
.assignvariableop_5_deep_q_network_dense_2_bias4
0assignvariableop_6_deep_q_network_dense_3_kernel2
.assignvariableop_7_deep_q_network_dense_3_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count:
6assignvariableop_15_adam_deep_q_network_dense_kernel_m8
4assignvariableop_16_adam_deep_q_network_dense_bias_m<
8assignvariableop_17_adam_deep_q_network_dense_1_kernel_m:
6assignvariableop_18_adam_deep_q_network_dense_1_bias_m<
8assignvariableop_19_adam_deep_q_network_dense_2_kernel_m:
6assignvariableop_20_adam_deep_q_network_dense_2_bias_m<
8assignvariableop_21_adam_deep_q_network_dense_3_kernel_m:
6assignvariableop_22_adam_deep_q_network_dense_3_bias_m:
6assignvariableop_23_adam_deep_q_network_dense_kernel_v8
4assignvariableop_24_adam_deep_q_network_dense_bias_v<
8assignvariableop_25_adam_deep_q_network_dense_1_kernel_v:
6assignvariableop_26_adam_deep_q_network_dense_1_bias_v<
8assignvariableop_27_adam_deep_q_network_dense_2_kernel_v:
6assignvariableop_28_adam_deep_q_network_dense_2_bias_v<
8assignvariableop_29_adam_deep_q_network_dense_3_kernel_v:
6assignvariableop_30_adam_deep_q_network_dense_3_bias_v
identity_32ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9љ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*ю
valueњBЈ B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense3/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ќ
_output_shapesЃ
ђ::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityФ
AssignVariableOpAssignVariableOp,assignvariableop_deep_q_network_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1▒
AssignVariableOp_1AssignVariableOp,assignvariableop_1_deep_q_network_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2х
AssignVariableOp_2AssignVariableOp0assignvariableop_2_deep_q_network_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3│
AssignVariableOp_3AssignVariableOp.assignvariableop_3_deep_q_network_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4х
AssignVariableOp_4AssignVariableOp0assignvariableop_4_deep_q_network_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5│
AssignVariableOp_5AssignVariableOp.assignvariableop_5_deep_q_network_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6х
AssignVariableOp_6AssignVariableOp0assignvariableop_6_deep_q_network_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7│
AssignVariableOp_7AssignVariableOp.assignvariableop_7_deep_q_network_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8А
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Б
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Д
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11д
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13А
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14А
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Й
AssignVariableOp_15AssignVariableOp6assignvariableop_15_adam_deep_q_network_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╝
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_deep_q_network_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17└
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_deep_q_network_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Й
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_deep_q_network_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19└
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_deep_q_network_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Й
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_deep_q_network_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21└
AssignVariableOp_21AssignVariableOp8assignvariableop_21_adam_deep_q_network_dense_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Й
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_deep_q_network_dense_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Й
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_deep_q_network_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╝
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_deep_q_network_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25└
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_deep_q_network_dense_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Й
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_deep_q_network_dense_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27└
AssignVariableOp_27AssignVariableOp8assignvariableop_27_adam_deep_q_network_dense_2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Й
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_deep_q_network_dense_2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29└
AssignVariableOp_29AssignVariableOp8assignvariableop_29_adam_deep_q_network_dense_3_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Й
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_deep_q_network_dense_3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpѕ
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31ч
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*њ
_input_shapesђ
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
В	
█
B__inference_dense_layer_call_and_return_conditional_losses_8749779

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         X2
ReluЌ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ,::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs
Ь$
╩
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749726

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбdense_3/BiasAdd/ReadVariableOpбdense_3/MatMul/ReadVariableOpЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:,X*
dtype02
dense/MatMul/ReadVariableOpЁ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         X2

dense/Reluд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	X░*
dtype02
dense_1/MatMul/ReadVariableOpъ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
dense_1/MatMulЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:░*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ░2
dense_1/Reluд
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	░X*
dtype02
dense_2/MatMul/ReadVariableOpЪ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_2/MatMulц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02 
dense_2/BiasAdd/ReadVariableOpА
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
dense_2/ReluЦ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:X*
dtype02
dense_3/MatMul/ReadVariableOpЪ
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/MatMulц
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpА
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/BiasAddВ
IdentityIdentitydense_3/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs
ы	
П
D__inference_dense_2_layer_call_and_return_conditional_losses_8749371

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	░X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         X2
ReluЌ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ░::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ░
 
_user_specified_nameinputs
я
~
)__inference_dense_2_layer_call_fn_8749828

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_87493712
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ░::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ░
 
_user_specified_nameinputs
«
Я
0__inference_deep_q_network_layer_call_fn_8749643
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_deep_q_network_layer_call_and_return_conditional_losses_87494652
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         ,
!
_user_specified_name	input_1
Щ
Н
%__inference_signature_wrapper_8749560
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__wrapped_model_87493022
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         ,
!
_user_specified_name	input_1
З	
П
D__inference_dense_1_layer_call_and_return_conditional_losses_8749799

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	X░*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:░*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ░2
Reluў
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ░2

Identity"
identityIdentity:output:0*.
_input_shapes
:         X::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         X
 
_user_specified_nameinputs
Ф
▀
0__inference_deep_q_network_layer_call_fn_8749768

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_deep_q_network_layer_call_and_return_conditional_losses_87495102
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs
ы$
╦
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749622
input_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбdense_3/BiasAdd/ReadVariableOpбdense_3/MatMul/ReadVariableOpЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:,X*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         X2

dense/Reluд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	X░*
dtype02
dense_1/MatMul/ReadVariableOpъ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
dense_1/MatMulЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:░*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ░2
dense_1/Reluд
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	░X*
dtype02
dense_2/MatMul/ReadVariableOpЪ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_2/MatMulц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02 
dense_2/BiasAdd/ReadVariableOpА
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
dense_2/ReluЦ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:X*
dtype02
dense_3/MatMul/ReadVariableOpЪ
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/MatMulц
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpА
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/BiasAddВ
IdentityIdentitydense_3/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:         ,
!
_user_specified_name	input_1
Ф
▀
0__inference_deep_q_network_layer_call_fn_8749747

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_deep_q_network_layer_call_and_return_conditional_losses_87494652
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs
«
Я
0__inference_deep_q_network_layer_call_fn_8749664
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_deep_q_network_layer_call_and_return_conditional_losses_87495102
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         ,
!
_user_specified_name	input_1
ы$
╦
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749591
input_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбdense_3/BiasAdd/ReadVariableOpбdense_3/MatMul/ReadVariableOpЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:,X*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         X2

dense/Reluд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	X░*
dtype02
dense_1/MatMul/ReadVariableOpъ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
dense_1/MatMulЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:░*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ░2
dense_1/Reluд
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	░X*
dtype02
dense_2/MatMul/ReadVariableOpЪ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_2/MatMulц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02 
dense_2/BiasAdd/ReadVariableOpА
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
dense_2/ReluЦ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:X*
dtype02
dense_3/MatMul/ReadVariableOpЪ
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/MatMulц
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpА
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/BiasAddВ
IdentityIdentitydense_3/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:         ,
!
_user_specified_name	input_1
Ы0
њ
"__inference__wrapped_model_8749302
input_17
3deep_q_network_dense_matmul_readvariableop_resource8
4deep_q_network_dense_biasadd_readvariableop_resource9
5deep_q_network_dense_1_matmul_readvariableop_resource:
6deep_q_network_dense_1_biasadd_readvariableop_resource9
5deep_q_network_dense_2_matmul_readvariableop_resource:
6deep_q_network_dense_2_biasadd_readvariableop_resource9
5deep_q_network_dense_3_matmul_readvariableop_resource:
6deep_q_network_dense_3_biasadd_readvariableop_resource
identityѕб+deep_q_network/dense/BiasAdd/ReadVariableOpб*deep_q_network/dense/MatMul/ReadVariableOpб-deep_q_network/dense_1/BiasAdd/ReadVariableOpб,deep_q_network/dense_1/MatMul/ReadVariableOpб-deep_q_network/dense_2/BiasAdd/ReadVariableOpб,deep_q_network/dense_2/MatMul/ReadVariableOpб-deep_q_network/dense_3/BiasAdd/ReadVariableOpб,deep_q_network/dense_3/MatMul/ReadVariableOp╠
*deep_q_network/dense/MatMul/ReadVariableOpReadVariableOp3deep_q_network_dense_matmul_readvariableop_resource*
_output_shapes

:,X*
dtype02,
*deep_q_network/dense/MatMul/ReadVariableOp│
deep_q_network/dense/MatMulMatMulinput_12deep_q_network/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
deep_q_network/dense/MatMul╦
+deep_q_network/dense/BiasAdd/ReadVariableOpReadVariableOp4deep_q_network_dense_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02-
+deep_q_network/dense/BiasAdd/ReadVariableOpН
deep_q_network/dense/BiasAddBiasAdd%deep_q_network/dense/MatMul:product:03deep_q_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
deep_q_network/dense/BiasAddЌ
deep_q_network/dense/ReluRelu%deep_q_network/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
deep_q_network/dense/ReluМ
,deep_q_network/dense_1/MatMul/ReadVariableOpReadVariableOp5deep_q_network_dense_1_matmul_readvariableop_resource*
_output_shapes
:	X░*
dtype02.
,deep_q_network/dense_1/MatMul/ReadVariableOp┌
deep_q_network/dense_1/MatMulMatMul'deep_q_network/dense/Relu:activations:04deep_q_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
deep_q_network/dense_1/MatMulм
-deep_q_network/dense_1/BiasAdd/ReadVariableOpReadVariableOp6deep_q_network_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:░*
dtype02/
-deep_q_network/dense_1/BiasAdd/ReadVariableOpя
deep_q_network/dense_1/BiasAddBiasAdd'deep_q_network/dense_1/MatMul:product:05deep_q_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2 
deep_q_network/dense_1/BiasAddъ
deep_q_network/dense_1/ReluRelu'deep_q_network/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ░2
deep_q_network/dense_1/ReluМ
,deep_q_network/dense_2/MatMul/ReadVariableOpReadVariableOp5deep_q_network_dense_2_matmul_readvariableop_resource*
_output_shapes
:	░X*
dtype02.
,deep_q_network/dense_2/MatMul/ReadVariableOp█
deep_q_network/dense_2/MatMulMatMul)deep_q_network/dense_1/Relu:activations:04deep_q_network/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
deep_q_network/dense_2/MatMulЛ
-deep_q_network/dense_2/BiasAdd/ReadVariableOpReadVariableOp6deep_q_network_dense_2_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02/
-deep_q_network/dense_2/BiasAdd/ReadVariableOpП
deep_q_network/dense_2/BiasAddBiasAdd'deep_q_network/dense_2/MatMul:product:05deep_q_network/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2 
deep_q_network/dense_2/BiasAddЮ
deep_q_network/dense_2/ReluRelu'deep_q_network/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
deep_q_network/dense_2/Reluм
,deep_q_network/dense_3/MatMul/ReadVariableOpReadVariableOp5deep_q_network_dense_3_matmul_readvariableop_resource*
_output_shapes

:X*
dtype02.
,deep_q_network/dense_3/MatMul/ReadVariableOp█
deep_q_network/dense_3/MatMulMatMul)deep_q_network/dense_2/Relu:activations:04deep_q_network/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
deep_q_network/dense_3/MatMulЛ
-deep_q_network/dense_3/BiasAdd/ReadVariableOpReadVariableOp6deep_q_network_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-deep_q_network/dense_3/BiasAdd/ReadVariableOpП
deep_q_network/dense_3/BiasAddBiasAdd'deep_q_network/dense_3/MatMul:product:05deep_q_network/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
deep_q_network/dense_3/BiasAddз
IdentityIdentity'deep_q_network/dense_3/BiasAdd:output:0,^deep_q_network/dense/BiasAdd/ReadVariableOp+^deep_q_network/dense/MatMul/ReadVariableOp.^deep_q_network/dense_1/BiasAdd/ReadVariableOp-^deep_q_network/dense_1/MatMul/ReadVariableOp.^deep_q_network/dense_2/BiasAdd/ReadVariableOp-^deep_q_network/dense_2/MatMul/ReadVariableOp.^deep_q_network/dense_3/BiasAdd/ReadVariableOp-^deep_q_network/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::2Z
+deep_q_network/dense/BiasAdd/ReadVariableOp+deep_q_network/dense/BiasAdd/ReadVariableOp2X
*deep_q_network/dense/MatMul/ReadVariableOp*deep_q_network/dense/MatMul/ReadVariableOp2^
-deep_q_network/dense_1/BiasAdd/ReadVariableOp-deep_q_network/dense_1/BiasAdd/ReadVariableOp2\
,deep_q_network/dense_1/MatMul/ReadVariableOp,deep_q_network/dense_1/MatMul/ReadVariableOp2^
-deep_q_network/dense_2/BiasAdd/ReadVariableOp-deep_q_network/dense_2/BiasAdd/ReadVariableOp2\
,deep_q_network/dense_2/MatMul/ReadVariableOp,deep_q_network/dense_2/MatMul/ReadVariableOp2^
-deep_q_network/dense_3/BiasAdd/ReadVariableOp-deep_q_network/dense_3/BiasAdd/ReadVariableOp2\
,deep_q_network/dense_3/MatMul/ReadVariableOp,deep_q_network/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:         ,
!
_user_specified_name	input_1
я
~
)__inference_dense_1_layer_call_fn_8749808

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ░*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_87493442
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ░2

Identity"
identityIdentity:output:0*.
_input_shapes
:         X::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         X
 
_user_specified_nameinputs
Ь$
╩
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749695

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpбdense_3/BiasAdd/ReadVariableOpбdense_3/MatMul/ReadVariableOpЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:,X*
dtype02
dense/MatMul/ReadVariableOpЁ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         X2

dense/Reluд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	X░*
dtype02
dense_1/MatMul/ReadVariableOpъ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
dense_1/MatMulЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:░*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ░2
dense_1/Reluд
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	░X*
dtype02
dense_2/MatMul/ReadVariableOpЪ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_2/MatMulц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02 
dense_2/BiasAdd/ReadVariableOpА
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         X2
dense_2/ReluЦ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:X*
dtype02
dense_3/MatMul/ReadVariableOpЪ
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/MatMulц
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpА
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/BiasAddВ
IdentityIdentitydense_3/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs
▄
~
)__inference_dense_3_layer_call_fn_8749847

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_87493972
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         X::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         X
 
_user_specified_nameinputs
п
|
'__inference_dense_layer_call_fn_8749788

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_87493172
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ,::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs
З	
П
D__inference_dense_1_layer_call_and_return_conditional_losses_8749344

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	X░*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:░*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ░2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ░2
Reluў
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ░2

Identity"
identityIdentity:output:0*.
_input_shapes
:         X::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         X
 
_user_specified_nameinputs
В	
█
B__inference_dense_layer_call_and_return_conditional_losses_8749317

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:,X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         X2
ReluЌ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ,::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs
ы	
П
D__inference_dense_2_layer_call_and_return_conditional_losses_8749819

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	░X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         X2
ReluЌ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         X2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ░::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ░
 
_user_specified_nameinputs
Ј
ћ
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749465

inputs
dense_8749444
dense_8749446
dense_1_8749449
dense_1_8749451
dense_2_8749454
dense_2_8749456
dense_3_8749459
dense_3_8749461
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallѕ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8749444dense_8749446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_87493172
dense/StatefulPartitionedCall│
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_8749449dense_1_8749451*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ░*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_87493442!
dense_1/StatefulPartitionedCall┤
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_8749454dense_2_8749456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_87493712!
dense_2/StatefulPartitionedCall┤
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_8749459dense_3_8749461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_87493972!
dense_3/StatefulPartitionedCallѓ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs
њ	
П
D__inference_dense_3_layer_call_and_return_conditional_losses_8749838

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         X::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         X
 
_user_specified_nameinputs
╔G
х
 __inference__traced_save_8749963
file_prefix:
6savev2_deep_q_network_dense_kernel_read_readvariableop8
4savev2_deep_q_network_dense_bias_read_readvariableop<
8savev2_deep_q_network_dense_1_kernel_read_readvariableop:
6savev2_deep_q_network_dense_1_bias_read_readvariableop<
8savev2_deep_q_network_dense_2_kernel_read_readvariableop:
6savev2_deep_q_network_dense_2_bias_read_readvariableop<
8savev2_deep_q_network_dense_3_kernel_read_readvariableop:
6savev2_deep_q_network_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopA
=savev2_adam_deep_q_network_dense_kernel_m_read_readvariableop?
;savev2_adam_deep_q_network_dense_bias_m_read_readvariableopC
?savev2_adam_deep_q_network_dense_1_kernel_m_read_readvariableopA
=savev2_adam_deep_q_network_dense_1_bias_m_read_readvariableopC
?savev2_adam_deep_q_network_dense_2_kernel_m_read_readvariableopA
=savev2_adam_deep_q_network_dense_2_bias_m_read_readvariableopC
?savev2_adam_deep_q_network_dense_3_kernel_m_read_readvariableopA
=savev2_adam_deep_q_network_dense_3_bias_m_read_readvariableopA
=savev2_adam_deep_q_network_dense_kernel_v_read_readvariableop?
;savev2_adam_deep_q_network_dense_bias_v_read_readvariableopC
?savev2_adam_deep_q_network_dense_1_kernel_v_read_readvariableopA
=savev2_adam_deep_q_network_dense_1_bias_v_read_readvariableopC
?savev2_adam_deep_q_network_dense_2_kernel_v_read_readvariableopA
=savev2_adam_deep_q_network_dense_2_bias_v_read_readvariableopC
?savev2_adam_deep_q_network_dense_3_kernel_v_read_readvariableopA
=savev2_adam_deep_q_network_dense_3_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameі
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*ю
valueњBЈ B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense3/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╚
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЦ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_deep_q_network_dense_kernel_read_readvariableop4savev2_deep_q_network_dense_bias_read_readvariableop8savev2_deep_q_network_dense_1_kernel_read_readvariableop6savev2_deep_q_network_dense_1_bias_read_readvariableop8savev2_deep_q_network_dense_2_kernel_read_readvariableop6savev2_deep_q_network_dense_2_bias_read_readvariableop8savev2_deep_q_network_dense_3_kernel_read_readvariableop6savev2_deep_q_network_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop=savev2_adam_deep_q_network_dense_kernel_m_read_readvariableop;savev2_adam_deep_q_network_dense_bias_m_read_readvariableop?savev2_adam_deep_q_network_dense_1_kernel_m_read_readvariableop=savev2_adam_deep_q_network_dense_1_bias_m_read_readvariableop?savev2_adam_deep_q_network_dense_2_kernel_m_read_readvariableop=savev2_adam_deep_q_network_dense_2_bias_m_read_readvariableop?savev2_adam_deep_q_network_dense_3_kernel_m_read_readvariableop=savev2_adam_deep_q_network_dense_3_bias_m_read_readvariableop=savev2_adam_deep_q_network_dense_kernel_v_read_readvariableop;savev2_adam_deep_q_network_dense_bias_v_read_readvariableop?savev2_adam_deep_q_network_dense_1_kernel_v_read_readvariableop=savev2_adam_deep_q_network_dense_1_bias_v_read_readvariableop?savev2_adam_deep_q_network_dense_2_kernel_v_read_readvariableop=savev2_adam_deep_q_network_dense_2_bias_v_read_readvariableop?savev2_adam_deep_q_network_dense_3_kernel_v_read_readvariableop=savev2_adam_deep_q_network_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*­
_input_shapesя
█: :,X:X:	X░:░:	░X:X:X:: : : : : : : :,X:X:	X░:░:	░X:X:X::,X:X:	X░:░:	░X:X:X:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:,X: 

_output_shapes
:X:%!

_output_shapes
:	X░:!

_output_shapes	
:░:%!

_output_shapes
:	░X: 

_output_shapes
:X:$ 

_output_shapes

:X: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:,X: 

_output_shapes
:X:%!

_output_shapes
:	X░:!

_output_shapes	
:░:%!

_output_shapes
:	░X: 

_output_shapes
:X:$ 

_output_shapes

:X: 

_output_shapes
::$ 

_output_shapes

:,X: 

_output_shapes
:X:%!

_output_shapes
:	X░:!

_output_shapes	
:░:%!

_output_shapes
:	░X: 

_output_shapes
:X:$ 

_output_shapes

:X: 

_output_shapes
:: 

_output_shapes
: 
Ј
ћ
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749510

inputs
dense_8749489
dense_8749491
dense_1_8749494
dense_1_8749496
dense_2_8749499
dense_2_8749501
dense_3_8749504
dense_3_8749506
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallѕ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8749489dense_8749491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_87493172
dense/StatefulPartitionedCall│
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_8749494dense_1_8749496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ░*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_87493442!
dense_1/StatefulPartitionedCall┤
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_8749499dense_2_8749501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         X*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_87493712!
dense_2/StatefulPartitionedCall┤
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_8749504dense_3_8749506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_87493972!
dense_3/StatefulPartitionedCallѓ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         ,::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:         ,
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ф
serving_defaultЌ
;
input_10
serving_default_input_1:0         ,<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:┐~
─

dense1

dense2

dense3

dense4
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
*V&call_and_return_all_conditional_losses
W__call__
X_default_save_signature"╔
_tf_keras_model»{"class_name": "DeepQNetwork", "name": "deep_q_network", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DeepQNetwork"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.004999999888241291, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
р

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"╝
_tf_keras_layerб{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, [44]]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, [44]]}, "dtype": "float32", "units": 88, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 44}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 44]}}
№

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*[&call_and_return_all_conditional_losses
\__call__"╩
_tf_keras_layer░{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 176, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 88}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 88]}}
­

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 88, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 176}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 176]}}
№

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
*_&call_and_return_all_conditional_losses
`__call__"╩
_tf_keras_layer░{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 88}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 88]}}
с
#iter

$beta_1

%beta_2
	&decay
'learning_ratemFmGmHmImJmKmLmMvNvOvPvQvRvSvTvU"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
╩

(layers
)metrics
trainable_variables
	variables
*layer_metrics
+non_trainable_variables
,layer_regularization_losses
regularization_losses
W__call__
X_default_save_signature
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
-:+,X2deep_q_network/dense/kernel
':%X2deep_q_network/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г

-layers
.metrics
trainable_variables
	variables
/layer_metrics
0non_trainable_variables
1layer_regularization_losses
regularization_losses
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
0:.	X░2deep_q_network/dense_1/kernel
*:(░2deep_q_network/dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г

2layers
3metrics
trainable_variables
	variables
4layer_metrics
5non_trainable_variables
6layer_regularization_losses
regularization_losses
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
0:.	░X2deep_q_network/dense_2/kernel
):'X2deep_q_network/dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г

7layers
8metrics
trainable_variables
	variables
9layer_metrics
:non_trainable_variables
;layer_regularization_losses
regularization_losses
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
/:-X2deep_q_network/dense_3/kernel
):'2deep_q_network/dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г

<layers
=metrics
trainable_variables
 	variables
>layer_metrics
?non_trainable_variables
@layer_regularization_losses
!regularization_losses
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
0
1
2
3"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╗
	Btotal
	Ccount
D	variables
E	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
B0
C1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
2:0,X2"Adam/deep_q_network/dense/kernel/m
,:*X2 Adam/deep_q_network/dense/bias/m
5:3	X░2$Adam/deep_q_network/dense_1/kernel/m
/:-░2"Adam/deep_q_network/dense_1/bias/m
5:3	░X2$Adam/deep_q_network/dense_2/kernel/m
.:,X2"Adam/deep_q_network/dense_2/bias/m
4:2X2$Adam/deep_q_network/dense_3/kernel/m
.:,2"Adam/deep_q_network/dense_3/bias/m
2:0,X2"Adam/deep_q_network/dense/kernel/v
,:*X2 Adam/deep_q_network/dense/bias/v
5:3	X░2$Adam/deep_q_network/dense_1/kernel/v
/:-░2"Adam/deep_q_network/dense_1/bias/v
5:3	░X2$Adam/deep_q_network/dense_2/kernel/v
.:,X2"Adam/deep_q_network/dense_2/bias/v
4:2X2$Adam/deep_q_network/dense_3/kernel/v
.:,2"Adam/deep_q_network/dense_3/bias/v
Щ2э
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749695
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749622
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749726
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749591└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ј2І
0__inference_deep_q_network_layer_call_fn_8749643
0__inference_deep_q_network_layer_call_fn_8749664
0__inference_deep_q_network_layer_call_fn_8749768
0__inference_deep_q_network_layer_call_fn_8749747└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Я2П
"__inference__wrapped_model_8749302Х
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *&б#
!і
input_1         ,
В2ж
B__inference_dense_layer_call_and_return_conditional_losses_8749779б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_layer_call_fn_8749788б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_1_layer_call_and_return_conditional_losses_8749799б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_1_layer_call_fn_8749808б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_2_layer_call_and_return_conditional_losses_8749819б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_2_layer_call_fn_8749828б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_3_layer_call_and_return_conditional_losses_8749838б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_3_layer_call_fn_8749847б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠B╔
%__inference_signature_wrapper_8749560input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Ќ
"__inference__wrapped_model_8749302q0б-
&б#
!і
input_1         ,
ф "3ф0
.
output_1"і
output_1         ║
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749591k8б5
.б+
!і
input_1         ,
p

 
ф "%б"
і
0         
џ ║
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749622k8б5
.б+
!і
input_1         ,
p 

 
ф "%б"
і
0         
џ ╣
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749695j7б4
-б*
 і
inputs         ,
p

 
ф "%б"
і
0         
џ ╣
K__inference_deep_q_network_layer_call_and_return_conditional_losses_8749726j7б4
-б*
 і
inputs         ,
p 

 
ф "%б"
і
0         
џ њ
0__inference_deep_q_network_layer_call_fn_8749643^8б5
.б+
!і
input_1         ,
p

 
ф "і         њ
0__inference_deep_q_network_layer_call_fn_8749664^8б5
.б+
!і
input_1         ,
p 

 
ф "і         Љ
0__inference_deep_q_network_layer_call_fn_8749747]7б4
-б*
 і
inputs         ,
p

 
ф "і         Љ
0__inference_deep_q_network_layer_call_fn_8749768]7б4
-б*
 і
inputs         ,
p 

 
ф "і         Ц
D__inference_dense_1_layer_call_and_return_conditional_losses_8749799]/б,
%б"
 і
inputs         X
ф "&б#
і
0         ░
џ }
)__inference_dense_1_layer_call_fn_8749808P/б,
%б"
 і
inputs         X
ф "і         ░Ц
D__inference_dense_2_layer_call_and_return_conditional_losses_8749819]0б-
&б#
!і
inputs         ░
ф "%б"
і
0         X
џ }
)__inference_dense_2_layer_call_fn_8749828P0б-
&б#
!і
inputs         ░
ф "і         Xц
D__inference_dense_3_layer_call_and_return_conditional_losses_8749838\/б,
%б"
 і
inputs         X
ф "%б"
і
0         
џ |
)__inference_dense_3_layer_call_fn_8749847O/б,
%б"
 і
inputs         X
ф "і         б
B__inference_dense_layer_call_and_return_conditional_losses_8749779\/б,
%б"
 і
inputs         ,
ф "%б"
і
0         X
џ z
'__inference_dense_layer_call_fn_8749788O/б,
%б"
 і
inputs         ,
ф "і         XЦ
%__inference_signature_wrapper_8749560|;б8
б 
1ф.
,
input_1!і
input_1         ,"3ф0
.
output_1"і
output_1         