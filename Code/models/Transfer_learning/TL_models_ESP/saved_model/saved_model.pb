??
?$?$
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
P

ComplexAbs
x"T	
y"Tout"
Ttype0:
2"
Touttype0:
2
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
J
FFT
input"Tcomplex
output"Tcomplex"
Tcomplextype0:
2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 ?
:
Less
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??
?
Adam/classification_head/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/classification_head/bias/v
?
3Adam/classification_head/bias/v/Read/ReadVariableOpReadVariableOpAdam/classification_head/bias/v*
_output_shapes
:*
dtype0
?
!Adam/classification_head/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!Adam/classification_head/kernel/v
?
5Adam/classification_head/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/classification_head/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/classification_head/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/classification_head/bias/m
?
3Adam/classification_head/bias/m/Read/ReadVariableOpReadVariableOpAdam/classification_head/bias/m*
_output_shapes
:*
dtype0
?
!Adam/classification_head/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!Adam/classification_head/kernel/m
?
5Adam/classification_head/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/classification_head/kernel/m*
_output_shapes
:	?*
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
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
?
classification_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameclassification_head/bias
?
,classification_head/bias/Read/ReadVariableOpReadVariableOpclassification_head/bias*
_output_shapes
:*
dtype0
?
classification_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_nameclassification_head/kernel
?
.classification_head/kernel/Read/ReadVariableOpReadVariableOpclassification_head/kernel*
_output_shapes
:	?*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
?
 audio_preprocessing_layer/windowVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" audio_preprocessing_layer/window
?
4audio_preprocessing_layer/window/Read/ReadVariableOpReadVariableOp audio_preprocessing_layer/window*
_output_shapes	
:?*
dtype0
I
ConstConst*
_output_shapes
: *
dtype0*
valueB	 :??
J
Const_1Const*
_output_shapes
: *
dtype0*
value
B :?
J
Const_2Const*
_output_shapes
: *
dtype0*
value
B :?

NoOpNoOp
?m
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*?l
value?lB?l B?l
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures*
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

signatures
#_self_saveable_object_factories*
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
 layer_with_weights-5
 layer-12
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'	optimizer*
b
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412*

30
41*
* 
?
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
:trace_0
;trace_1
<trace_2
=trace_3* 
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
* 

Bserving_default* 
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

(window
(_window
#I_self_saveable_object_factories*

(0*
* 
* 
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
6
Strace_0
Ttrace_1
Utrace_2
Vtrace_3* 

Wserving_default* 
* 
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

)kernel
*bias
 ^_jit_compiled_convolution_op*
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
?
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

+kernel
,bias
 k_jit_compiled_convolution_op*
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
?
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

-kernel
.bias
 x_jit_compiled_convolution_op*
?
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses* 
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

/kernel
0bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

1kernel
2bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

3kernel
4bias*
Z
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411*

30
41*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
V
?trace_0
?trace_1
?trace_2
?trace_3
?trace_4
?trace_5* 
V
?trace_0
?trace_1
?trace_2
?trace_3
?trace_4
?trace_5* 
q
	?iter
?beta_1
?beta_2

?decay
?learning_rate3m?4m?3v?4v?*
`Z
VARIABLE_VALUE audio_preprocessing_layer/window&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_2/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_3/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_3/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_4/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_4/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEclassification_head/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEclassification_head/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
R
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

(0*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

(0*

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

)0
*1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

+0
,1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

-0
.1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

/0
01*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

10
21*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

30
41*

30
41*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
J
)0
*1
+2
,3
-4
.5
/6
07
18
29*
b
0
1
2
3
4
5
6
7
8
9
10
11
 12*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
a[
VARIABLE_VALUE	Adam/iter>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/beta_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/beta_2@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE
Adam/decay?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/learning_rateGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*

(0*
* 
* 
* 
* 
* 
* 

)0
*1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

-0
.1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

/0
01*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

10
21*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
jd
VARIABLE_VALUEtotal_1Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcount_1Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
hb
VARIABLE_VALUEtotalIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEcountIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
??
VARIABLE_VALUE!Adam/classification_head/kernel/mXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/classification_head/bias/mXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/classification_head/kernel/vXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/classification_head/bias/vXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
#serving_default_audio_preproc_inputPlaceholder*)
_output_shapes
:???????????*
dtype0*
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCall#serving_default_audio_preproc_inputConst audio_preprocessing_layer/windowConst_1Const_2conv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasdense_1/kerneldense_1/biasclassification_head/kernelclassification_head/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_41840
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4audio_preprocessing_layer/window/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp.classification_head/kernel/Read/ReadVariableOp,classification_head/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp5Adam/classification_head/kernel/m/Read/ReadVariableOp3Adam/classification_head/bias/m/Read/ReadVariableOp5Adam/classification_head/kernel/v/Read/ReadVariableOp3Adam/classification_head/bias/v/Read/ReadVariableOpConst_3*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_42801
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename audio_preprocessing_layer/windowconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasdense_1/kerneldense_1/biasclassification_head/kernelclassification_head/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount!Adam/classification_head/kernel/mAdam/classification_head/bias/m!Adam/classification_head/kernel/vAdam/classification_head/bias/v*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_42889??
?
?
*__inference_Assert_AssertGuard_true_95_116%
!assert_assertguard_identity_equal
"
assert_assertguard_placeholder!
assert_assertguard_identity_1
P
Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2
Assert/AssertGuard/NoOp?
Assert/AssertGuard/IdentityIdentity!assert_assertguard_identity_equal^Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: :???????????: 

_output_shapes
: :/+
)
_output_shapes
:???????????
?
?
+__inference_Assert_AssertGuard_false_96_180#
assert_assertguard_assert_equal
&
"assert_assertguard_assert_waveform!
assert_assertguard_identity_1
??Assert/AssertGuard/Assert?
Assert/AssertGuard/AssertAssertassert_assertguard_assert_equal"assert_assertguard_assert_waveform*

T
2*
_output_shapes
 2
Assert/AssertGuard/Assert?
Assert/AssertGuard/IdentityIdentityassert_assertguard_assert_equal^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: :???????????26
Assert/AssertGuard/AssertAssert/AssertGuard/Assert: 

_output_shapes
: :/+
)
_output_shapes
:???????????
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_40939

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0a
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?H
ReluReluBiasAdd:output:0*
T0*
_output_shapes
:	?Y
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes
:	?w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_42199

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_41388f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:#???????????????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
9
_output_shapes'
%:#???????????????????
 
_user_specified_nameinputs
?4
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_41278
conv2d_1_input(
conv2d_1_41240:
conv2d_1_41242:(
conv2d_2_41246: 
conv2d_2_41248: (
conv2d_3_41252:  
conv2d_3_41254: (
conv2d_4_41258:  
conv2d_4_41260: !
dense_1_41266:
??
dense_1_41268:	?,
classification_head_41272:	?'
classification_head_41274:
identity??+classification_head/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_41240conv2d_1_41242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40852?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:p* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40795?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_41246conv2d_2_41248*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:m *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40870?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:
6 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_40807?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_41252conv2d_3_41254*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:	3 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40888?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_40819?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_41258conv2d_4_41260*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40906?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_40831?
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_40919?
dropout_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_40926?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_41266dense_1_41268*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_40939?
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_40950?
+classification_head/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0classification_head_41272classification_head_41274*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_classification_head_layer_call_and_return_conditional_losses_40963z
IdentityIdentity4classification_head/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp,^classification_head/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2Z
+classification_head/StatefulPartitionedCall+classification_head/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
'
_output_shapes
:+?
(
_user_specified_nameconv2d_1_input
?9
?

__inference__traced_save_42801
file_prefix?
;savev2_audio_preprocessing_layer_window_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop9
5savev2_classification_head_kernel_read_readvariableop7
3savev2_classification_head_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop@
<savev2_adam_classification_head_kernel_m_read_readvariableop>
:savev2_adam_classification_head_bias_m_read_readvariableop@
<savev2_adam_classification_head_kernel_v_read_readvariableop>
:savev2_adam_classification_head_bias_v_read_readvariableop
savev2_const_3

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_audio_preprocessing_layer_window_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop5savev2_classification_head_kernel_read_readvariableop3savev2_classification_head_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop<savev2_adam_classification_head_kernel_m_read_readvariableop:savev2_adam_classification_head_bias_m_read_readvariableop<savev2_adam_classification_head_kernel_v_read_readvariableop:savev2_adam_classification_head_bias_v_read_readvariableopsavev2_const_3"/device:CPU:0*
_output_shapes
 *)
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?::: : :  : :  : :
??:?:	?:: : : : : : : : : :	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:?:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 	

_output_shapes
: :&
"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?	
?
-__inference_Assert_1_AssertGuard_false_115_86'
#assert_1_assertguard_assert_equal_1
7
3assert_1_assertguard_assert_readvariableop_resource#
assert_1_assertguard_identity_1
??Assert_1/AssertGuard/Assert?
*Assert_1/AssertGuard/Assert/ReadVariableOpReadVariableOp3assert_1_assertguard_assert_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*Assert_1/AssertGuard/Assert/ReadVariableOp?
Assert_1/AssertGuard/AssertAssert#assert_1_assertguard_assert_equal_12Assert_1/AssertGuard/Assert/ReadVariableOp:value:0*

T
2*
_output_shapes
 2
Assert_1/AssertGuard/Assert?
Assert_1/AssertGuard/IdentityIdentity#assert_1_assertguard_assert_equal_1^Assert_1/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert_1/AssertGuard/Identity?
Assert_1/AssertGuard/Identity_1Identity&Assert_1/AssertGuard/Identity:output:0^Assert_1/AssertGuard/Assert*
T0
*
_output_shapes
: 2!
Assert_1/AssertGuard/Identity_1"K
assert_1_assertguard_identity_1(Assert_1/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :2:
Assert_1/AssertGuard/AssertAssert_1/AssertGuard/Assert: 

_output_shapes
: 
?
K
/__inference_max_pooling2d_4_layer_call_fn_42587

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_40831?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40697

inputs#
audio_preprocessing_layer_40687.
audio_preprocessing_layer_40689:	?#
audio_preprocessing_layer_40691#
audio_preprocessing_layer_40693
identity??1audio_preprocessing_layer/StatefulPartitionedCall?
1audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallinputsaudio_preprocessing_layer_40687audio_preprocessing_layer_40689audio_preprocessing_layer_40691audio_preprocessing_layer_40693*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
IdentityIdentity:audio_preprocessing_layer/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????z
NoOpNoOp2^audio_preprocessing_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 2f
1audio_preprocessing_layer/StatefulPartitionedCall1audio_preprocessing_layer/StatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40736

inputs#
audio_preprocessing_layer_40726.
audio_preprocessing_layer_40728:	?#
audio_preprocessing_layer_40730#
audio_preprocessing_layer_40732
identity??1audio_preprocessing_layer/StatefulPartitionedCall?
1audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallinputsaudio_preprocessing_layer_40726audio_preprocessing_layer_40728audio_preprocessing_layer_40730audio_preprocessing_layer_40732*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
IdentityIdentity:audio_preprocessing_layer/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????z
NoOpNoOp2^audio_preprocessing_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 2f
1audio_preprocessing_layer/StatefulPartitionedCall1audio_preprocessing_layer/StatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?7
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_41181

inputs(
conv2d_1_41143:
conv2d_1_41145:(
conv2d_2_41149: 
conv2d_2_41151: (
conv2d_3_41155:  
conv2d_3_41157: (
conv2d_4_41161:  
conv2d_4_41163: !
dense_1_41169:
??
dense_1_41171:	?,
classification_head_41175:	?'
classification_head_41177:
identity??+classification_head/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_41143conv2d_1_41145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40852?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:p* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40795?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_41149conv2d_2_41151*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:m *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40870?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:
6 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_40807?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_41155conv2d_3_41157*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:	3 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40888?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_40819?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_41161conv2d_4_41163*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40906?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_40831?
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_40919?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_41060?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_41169dense_1_41171*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_40939?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_41027?
+classification_head/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0classification_head_41175classification_head_41177*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_classification_head_layer_call_and_return_conditional_losses_40963z
IdentityIdentity4classification_head/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp,^classification_head/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2Z
+classification_head/StatefulPartitionedCall+classification_head/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?
?
,__inference_sequential_3_layer_call_fn_41450
audio_preproc_input
unknown
	unknown_0:	?
	unknown_1
	unknown_2#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallaudio_preproc_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_41415f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
)
_output_shapes
:???????????
-
_user_specified_nameaudio_preproc_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_42562

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_call_364

inputs
	equal_1_x
unknown
	unknown_0
	unknown_1
identity??Assert/AssertGuard?Assert_1/Assert?StatefulPartitionedCall?StatefulPartitionedCall_1N
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankT
Equal/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Equal/yy
EqualEqualRank:output:0Equal/y:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal?
Assert/AssertGuardIf	Equal:z:0	Equal:z:0inputs*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *=
else_branch.R,
*__inference_Assert_AssertGuard_false_65_24*
output_shapes
: *=
then_branch.R,
*__inference_Assert_AssertGuard_true_64_1682
Assert/AssertGuard?
Assert/AssertGuard/IdentityIdentityAssert/AssertGuard:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/IdentityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
Equal_1Equal	equal_1_xstrided_slice:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_1H
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1?
Assert_1/AssertAssertEqual_1:z:0Shape_1:output:0^Assert/AssertGuard*

T
2*
_output_shapes
 2
Assert_1/Assert?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_tf_webaudio_spectrogram_3162
StatefulPartitionedCall?
StatefulPartitionedCall_1StatefulPartitionedCall StatefulPartitionedCall:output:0^Assert_1/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_z_normalize_spectrogram_672
StatefulPartitionedCall_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDims"StatefulPartitionedCall_1:output:0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
IdentityIdentityExpandDims:output:0^Assert/AssertGuard^Assert_1/Assert^StatefulPartitionedCall^StatefulPartitionedCall_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: :: : 2(
Assert/AssertGuardAssert/AssertGuard2"
Assert_1/AssertAssert_1/Assert22
StatefulPartitionedCallStatefulPartitionedCall26
StatefulPartitionedCall_1StatefulPartitionedCall_1:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_conv2d_3_layer_call_fn_42541

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:	3 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40888n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:	3 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
6 : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
6 
 
_user_specified_nameinputs
?j
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_42060

inputs1
-audio_preproc_audio_preprocessing_layer_41986<
-audio_preproc_audio_preprocessing_layer_41988:	?1
-audio_preproc_audio_preprocessing_layer_419901
-audio_preproc_audio_preprocessing_layer_41992N
4sequential_1_conv2d_1_conv2d_readvariableop_resource:C
5sequential_1_conv2d_1_biasadd_readvariableop_resource:N
4sequential_1_conv2d_2_conv2d_readvariableop_resource: C
5sequential_1_conv2d_2_biasadd_readvariableop_resource: N
4sequential_1_conv2d_3_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_3_biasadd_readvariableop_resource: N
4sequential_1_conv2d_4_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_4_biasadd_readvariableop_resource: G
3sequential_1_dense_1_matmul_readvariableop_resource:
??C
4sequential_1_dense_1_biasadd_readvariableop_resource:	?R
?sequential_1_classification_head_matmul_readvariableop_resource:	?N
@sequential_1_classification_head_biasadd_readvariableop_resource:
identity???audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall?7sequential_1/classification_head/BiasAdd/ReadVariableOp?6sequential_1/classification_head/MatMul/ReadVariableOp?,sequential_1/conv2d_1/BiasAdd/ReadVariableOp?+sequential_1/conv2d_1/Conv2D/ReadVariableOp?,sequential_1/conv2d_2/BiasAdd/ReadVariableOp?+sequential_1/conv2d_2/Conv2D/ReadVariableOp?,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?+sequential_1/conv2d_3/Conv2D/ReadVariableOp?,sequential_1/conv2d_4/BiasAdd/ReadVariableOp?+sequential_1/conv2d_4/Conv2D/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?
?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallinputs-audio_preproc_audio_preprocessing_layer_41986-audio_preproc_audio_preprocessing_layer_41988-audio_preproc_audio_preprocessing_layer_41990-audio_preproc_audio_preprocessing_layer_41992*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:+?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_1/conv2d_1/Conv2DConv2DHaudio_preproc/audio_preprocessing_layer/StatefulPartitionedCall:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?|
sequential_1/conv2d_1/ReluRelu&sequential_1/conv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
$sequential_1/max_pooling2d_1/MaxPoolMaxPool(sequential_1/conv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
+sequential_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_1/conv2d_2/Conv2DConv2D-sequential_1/max_pooling2d_1/MaxPool:output:03sequential_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
,sequential_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_1/conv2d_2/BiasAddBiasAdd%sequential_1/conv2d_2/Conv2D:output:04sequential_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m {
sequential_1/conv2d_2/ReluRelu&sequential_1/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
$sequential_1/max_pooling2d_2/MaxPoolMaxPool(sequential_1/conv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_1/conv2d_3/Conv2DConv2D-sequential_1/max_pooling2d_2/MaxPool:output:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 {
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
$sequential_1/max_pooling2d_3/MaxPoolMaxPool(sequential_1/conv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_1/conv2d_4/Conv2DConv2D-sequential_1/max_pooling2d_3/MaxPool:output:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
$sequential_1/max_pooling2d_4/MaxPoolMaxPool(sequential_1/conv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
sequential_1/flatten_1/ReshapeReshape-sequential_1/max_pooling2d_4/MaxPool:output:0%sequential_1/flatten_1/Const:output:0*
T0*
_output_shapes
:	?i
$sequential_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
"sequential_1/dropout_1/dropout/MulMul'sequential_1/flatten_1/Reshape:output:0-sequential_1/dropout_1/dropout/Const:output:0*
T0*
_output_shapes
:	?u
$sequential_1/dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
;sequential_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_1/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0r
-sequential_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
+sequential_1/dropout_1/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_1/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	??
#sequential_1/dropout_1/dropout/CastCast/sequential_1/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	??
$sequential_1/dropout_1/dropout/Mul_1Mul&sequential_1/dropout_1/dropout/Mul:z:0'sequential_1/dropout_1/dropout/Cast:y:0*
T0*
_output_shapes
:	??
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_1/dropout/Mul_1:z:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?i
$sequential_1/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
"sequential_1/dropout_2/dropout/MulMul'sequential_1/dense_1/Relu:activations:0-sequential_1/dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	?u
$sequential_1/dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
;sequential_1/dropout_2/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0r
-sequential_1/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
+sequential_1/dropout_2/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_2/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	??
#sequential_1/dropout_2/dropout/CastCast/sequential_1/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	??
$sequential_1/dropout_2/dropout/Mul_1Mul&sequential_1/dropout_2/dropout/Mul:z:0'sequential_1/dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	??
6sequential_1/classification_head/MatMul/ReadVariableOpReadVariableOp?sequential_1_classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
'sequential_1/classification_head/MatMulMatMul(sequential_1/dropout_2/dropout/Mul_1:z:0>sequential_1/classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
7sequential_1/classification_head/BiasAdd/ReadVariableOpReadVariableOp@sequential_1_classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_1/classification_head/BiasAddBiasAdd1sequential_1/classification_head/MatMul:product:0?sequential_1/classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:?
(sequential_1/classification_head/SoftmaxSoftmax1sequential_1/classification_head/BiasAdd:output:0*
T0*
_output_shapes

:x
IdentityIdentity2sequential_1/classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp@^audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall8^sequential_1/classification_head/BiasAdd/ReadVariableOp7^sequential_1/classification_head/MatMul/ReadVariableOp-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp-^sequential_1/conv2d_2/BiasAdd/ReadVariableOp,^sequential_1/conv2d_2/Conv2D/ReadVariableOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 2?
?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall2r
7sequential_1/classification_head/BiasAdd/ReadVariableOp7sequential_1/classification_head/BiasAdd/ReadVariableOp2p
6sequential_1/classification_head/MatMul/ReadVariableOp6sequential_1/classification_head/MatMul/ReadVariableOp2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_2/BiasAdd/ReadVariableOp,sequential_1/conv2d_2/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_2/Conv2D/ReadVariableOp+sequential_1/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_3_layer_call_fn_41920

inputs
unknown
	unknown_0:	?
	unknown_1
	unknown_2#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_41653f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_conv2d_2_layer_call_fn_42511

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:m *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40870n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:m `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:p: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:p
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41763
audio_preproc_input
audio_preproc_41728"
audio_preproc_41730:	?
audio_preproc_41732
audio_preproc_41734,
sequential_1_41737: 
sequential_1_41739:,
sequential_1_41741:  
sequential_1_41743: ,
sequential_1_41745:   
sequential_1_41747: ,
sequential_1_41749:   
sequential_1_41751: &
sequential_1_41753:
??!
sequential_1_41755:	?%
sequential_1_41757:	? 
sequential_1_41759:
identity??%audio_preproc/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?
%audio_preproc/StatefulPartitionedCallStatefulPartitionedCallaudio_preproc_inputaudio_preproc_41728audio_preproc_41730audio_preproc_41732audio_preproc_41734*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40697?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall.audio_preproc/StatefulPartitionedCall:output:0sequential_1_41737sequential_1_41739sequential_1_41741sequential_1_41743sequential_1_41745sequential_1_41747sequential_1_41749sequential_1_41751sequential_1_41753sequential_1_41755sequential_1_41757sequential_1_41759*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_41388s
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp&^audio_preproc/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 2N
%audio_preproc/StatefulPartitionedCall%audio_preproc/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:^ Z
)
_output_shapes
:???????????
-
_user_specified_nameaudio_preproc_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?X
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41983

inputs1
-audio_preproc_audio_preprocessing_layer_41923<
-audio_preproc_audio_preprocessing_layer_41925:	?1
-audio_preproc_audio_preprocessing_layer_419271
-audio_preproc_audio_preprocessing_layer_41929N
4sequential_1_conv2d_1_conv2d_readvariableop_resource:C
5sequential_1_conv2d_1_biasadd_readvariableop_resource:N
4sequential_1_conv2d_2_conv2d_readvariableop_resource: C
5sequential_1_conv2d_2_biasadd_readvariableop_resource: N
4sequential_1_conv2d_3_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_3_biasadd_readvariableop_resource: N
4sequential_1_conv2d_4_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_4_biasadd_readvariableop_resource: G
3sequential_1_dense_1_matmul_readvariableop_resource:
??C
4sequential_1_dense_1_biasadd_readvariableop_resource:	?R
?sequential_1_classification_head_matmul_readvariableop_resource:	?N
@sequential_1_classification_head_biasadd_readvariableop_resource:
identity???audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall?7sequential_1/classification_head/BiasAdd/ReadVariableOp?6sequential_1/classification_head/MatMul/ReadVariableOp?,sequential_1/conv2d_1/BiasAdd/ReadVariableOp?+sequential_1/conv2d_1/Conv2D/ReadVariableOp?,sequential_1/conv2d_2/BiasAdd/ReadVariableOp?+sequential_1/conv2d_2/Conv2D/ReadVariableOp?,sequential_1/conv2d_3/BiasAdd/ReadVariableOp?+sequential_1/conv2d_3/Conv2D/ReadVariableOp?,sequential_1/conv2d_4/BiasAdd/ReadVariableOp?+sequential_1/conv2d_4/Conv2D/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?
?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallinputs-audio_preproc_audio_preprocessing_layer_41923-audio_preproc_audio_preprocessing_layer_41925-audio_preproc_audio_preprocessing_layer_41927-audio_preproc_audio_preprocessing_layer_41929*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:+?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_1/conv2d_1/Conv2DConv2DHaudio_preproc/audio_preprocessing_layer/StatefulPartitionedCall:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?|
sequential_1/conv2d_1/ReluRelu&sequential_1/conv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
$sequential_1/max_pooling2d_1/MaxPoolMaxPool(sequential_1/conv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
+sequential_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
sequential_1/conv2d_2/Conv2DConv2D-sequential_1/max_pooling2d_1/MaxPool:output:03sequential_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
,sequential_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_1/conv2d_2/BiasAddBiasAdd%sequential_1/conv2d_2/Conv2D:output:04sequential_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m {
sequential_1/conv2d_2/ReluRelu&sequential_1/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
$sequential_1/max_pooling2d_2/MaxPoolMaxPool(sequential_1/conv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_1/conv2d_3/Conv2DConv2D-sequential_1/max_pooling2d_2/MaxPool:output:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 {
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
$sequential_1/max_pooling2d_3/MaxPoolMaxPool(sequential_1/conv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
sequential_1/conv2d_4/Conv2DConv2D-sequential_1/max_pooling2d_3/MaxPool:output:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
$sequential_1/max_pooling2d_4/MaxPoolMaxPool(sequential_1/conv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
sequential_1/flatten_1/ReshapeReshape-sequential_1/max_pooling2d_4/MaxPool:output:0%sequential_1/flatten_1/Const:output:0*
T0*
_output_shapes
:	?~
sequential_1/dropout_1/IdentityIdentity'sequential_1/flatten_1/Reshape:output:0*
T0*
_output_shapes
:	??
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?r
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?~
sequential_1/dropout_2/IdentityIdentity'sequential_1/dense_1/Relu:activations:0*
T0*
_output_shapes
:	??
6sequential_1/classification_head/MatMul/ReadVariableOpReadVariableOp?sequential_1_classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
'sequential_1/classification_head/MatMulMatMul(sequential_1/dropout_2/Identity:output:0>sequential_1/classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
7sequential_1/classification_head/BiasAdd/ReadVariableOpReadVariableOp@sequential_1_classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
(sequential_1/classification_head/BiasAddBiasAdd1sequential_1/classification_head/MatMul:product:0?sequential_1/classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:?
(sequential_1/classification_head/SoftmaxSoftmax1sequential_1/classification_head/BiasAdd:output:0*
T0*
_output_shapes

:x
IdentityIdentity2sequential_1/classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp@^audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall8^sequential_1/classification_head/BiasAdd/ReadVariableOp7^sequential_1/classification_head/MatMul/ReadVariableOp-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp-^sequential_1/conv2d_2/BiasAdd/ReadVariableOp,^sequential_1/conv2d_2/Conv2D/ReadVariableOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 2?
?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall2r
7sequential_1/classification_head/BiasAdd/ReadVariableOp7sequential_1/classification_head/BiasAdd/ReadVariableOp2p
6sequential_1/classification_head/MatMul/ReadVariableOp6sequential_1/classification_head/MatMul/ReadVariableOp2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_2/BiasAdd/ReadVariableOp,sequential_1/conv2d_2/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_2/Conv2D/ReadVariableOp+sequential_1/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
E
)__inference_dropout_2_layer_call_fn_42655

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_40950X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?	
?
-__inference_audio_preproc_layer_call_fn_40708#
audio_preprocessing_layer_input
unknown
	unknown_0:	?
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallaudio_preprocessing_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40697?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
)
_output_shapes
:???????????
9
_user_specified_name!audio_preprocessing_layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_Assert_AssertGuard_true_197_186%
!assert_assertguard_identity_equal
"
assert_assertguard_placeholder!
assert_assertguard_identity_1
P
Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2
Assert/AssertGuard/NoOp?
Assert/AssertGuard/IdentityIdentity!assert_assertguard_identity_equal^Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :: 

_output_shapes
: 
?
?
-__inference_audio_preproc_layer_call_fn_42073

inputs
unknown
	unknown_0:	?
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40697?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40773#
audio_preprocessing_layer_input#
audio_preprocessing_layer_40763.
audio_preprocessing_layer_40765:	?#
audio_preprocessing_layer_40767#
audio_preprocessing_layer_40769
identity??1audio_preprocessing_layer/StatefulPartitionedCall?
1audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallaudio_preprocessing_layer_inputaudio_preprocessing_layer_40763audio_preprocessing_layer_40765audio_preprocessing_layer_40767audio_preprocessing_layer_40769*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
IdentityIdentity:audio_preprocessing_layer/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????z
NoOpNoOp2^audio_preprocessing_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 2f
1audio_preprocessing_layer/StatefulPartitionedCall1audio_preprocessing_layer/StatefulPartitionedCall:j f
)
_output_shapes
:???????????
9
_user_specified_name!audio_preprocessing_layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_signature_wrapper_41840
audio_preproc_input
unknown
	unknown_0:	?
	unknown_1
	unknown_2#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallaudio_preproc_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_40680f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
)
_output_shapes
:???????????
-
_user_specified_nameaudio_preproc_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_40819

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_42481

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:*?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:+?: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?
?
R__inference_audio_preprocessing_layer_layer_call_and_return_conditional_losses_424

inputs
	equal_1_x
unknown
	unknown_0
	unknown_1
identity??Assert/AssertGuard?Assert_1/Assert?StatefulPartitionedCall?StatefulPartitionedCall_1N
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankT
Equal/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Equal/yy
EqualEqualRank:output:0Equal/y:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal?
Assert/AssertGuardIf	Equal:z:0	Equal:z:0inputs*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *>
else_branch/R-
+__inference_Assert_AssertGuard_false_415_92*
output_shapes
: *=
then_branch.R,
*__inference_Assert_AssertGuard_true_414_792
Assert/AssertGuard?
Assert/AssertGuard/IdentityIdentityAssert/AssertGuard:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/IdentityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
Equal_1Equal	equal_1_xstrided_slice:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_1H
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1?
Assert_1/AssertAssertEqual_1:z:0Shape_1:output:0^Assert/AssertGuard*

T
2*
_output_shapes
 2
Assert_1/Assert?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_tf_webaudio_spectrogram_3162
StatefulPartitionedCall?
StatefulPartitionedCall_1StatefulPartitionedCall StatefulPartitionedCall:output:0^Assert_1/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_z_normalize_spectrogram_672
StatefulPartitionedCall_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDims"StatefulPartitionedCall_1:output:0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
IdentityIdentityExpandDims:output:0^Assert/AssertGuard^Assert_1/Assert^StatefulPartitionedCall^StatefulPartitionedCall_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: :: : 2(
Assert/AssertGuardAssert/AssertGuard2"
Assert_1/AssertAssert_1/Assert22
StatefulPartitionedCallStatefulPartitionedCall26
StatefulPartitionedCall_1StatefulPartitionedCall_1:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_cond_true_217_208
cond_size_waveform_frame
cond_equal_size
cond_identity??cond/Assert/AssertGuardY
	cond/SizeSizecond_size_waveform_frame*
T0*
_output_shapes
: 2
	cond/Size?

cond/EqualEqualcond/Size:output:0cond_equal_size*
T0*
_output_shapes
: *
incompatible_shape_error( 2

cond/Equal?
cond/Assert/AssertGuardIfcond/Equal:z:0cond/Equal:z:0cond_size_waveform_frame*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *C
else_branch4R2
0__inference_cond_Assert_AssertGuard_false_225_73*
output_shapes
: *C
then_branch4R2
0__inference_cond_Assert_AssertGuard_true_224_1982
cond/Assert/AssertGuard?
 cond/Assert/AssertGuard/IdentityIdentity cond/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2"
 cond/Assert/AssertGuard/IdentityZ

cond/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2

cond/Constz
cond/IdentityIdentitycond/Const:output:0^cond/Assert/AssertGuard*
T0*
_output_shapes
: 2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:'???????????????????????????: 22
cond/Assert/AssertGuardcond/Assert/AssertGuard:C ?
=
_output_shapes+
):'???????????????????????????:

_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_42228

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_41549f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:#???????????????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
9
_output_shapes'
%:#???????????????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_42650

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0a
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0n
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?H
ReluReluBiasAdd:output:0*
T0*
_output_shapes
:	?Y
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes
:	?w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_42665

inputs

identity_1F
IdentityIdentityinputs*
T0*
_output_shapes
:	?S

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_42677

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @\
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	?^
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?g
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?a
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	?Q
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40786#
audio_preprocessing_layer_input#
audio_preprocessing_layer_40776.
audio_preprocessing_layer_40778:	?#
audio_preprocessing_layer_40780#
audio_preprocessing_layer_40782
identity??1audio_preprocessing_layer/StatefulPartitionedCall?
1audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallaudio_preprocessing_layer_inputaudio_preprocessing_layer_40776audio_preprocessing_layer_40778audio_preprocessing_layer_40780audio_preprocessing_layer_40782*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
IdentityIdentity:audio_preprocessing_layer/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????z
NoOpNoOp2^audio_preprocessing_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 2f
1audio_preprocessing_layer/StatefulPartitionedCall1audio_preprocessing_layer/StatefulPartitionedCall:j f
)
_output_shapes
:???????????
9
_user_specified_name!audio_preprocessing_layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?@
?	
G__inference_sequential_1_layer_call_and_return_conditional_losses_42282

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:  6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:  6
(conv2d_4_biasadd_readvariableop_resource: :
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?E
2classification_head_matmul_readvariableop_resource:	?A
3classification_head_biasadd_readvariableop_resource:
identity??*classification_head/BiasAdd/ReadVariableOp?)classification_head/MatMul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?b
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m a
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 a
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: a
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_1/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_1/Const:output:0*
T0*
_output_shapes
:	?d
dropout_1/IdentityIdentityflatten_1/Reshape:output:0*
T0*
_output_shapes
:	??
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?X
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?d
dropout_2/IdentityIdentitydense_1/Relu:activations:0*
T0*
_output_shapes
:	??
)classification_head/MatMul/ReadVariableOpReadVariableOp2classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
classification_head/MatMulMatMuldropout_2/Identity:output:01classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
*classification_head/BiasAdd/ReadVariableOpReadVariableOp3classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
classification_head/BiasAddBiasAdd$classification_head/MatMul:product:02classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
classification_head/SoftmaxSoftmax$classification_head/BiasAdd:output:0*
T0*
_output_shapes

:k
IdentityIdentity%classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp+^classification_head/BiasAdd/ReadVariableOp*^classification_head/MatMul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2X
*classification_head/BiasAdd/ReadVariableOp*classification_head/BiasAdd/ReadVariableOp2V
)classification_head/MatMul/ReadVariableOp)classification_head/MatMul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_41027

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @\
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	?^
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?g
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?a
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	?Q
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
__inference_cond_false_218_162
cond_shape_waveform_frame
cond_equal_size
cond_identity??cond/Assert/AssertGuarda

cond/ShapeShapecond_shape_waveform_frame*
T0*
_output_shapes
:2

cond/Shape?
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
cond/strided_slice/stack?
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice/stack_1?
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice?

cond/EqualEqualcond/strided_slice:output:0cond_equal_size*
T0*
_output_shapes
: *
incompatible_shape_error( 2

cond/Equal?
cond/Assert/AssertGuardIfcond/Equal:z:0cond/Equal:z:0cond_shape_waveform_frame*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *D
else_branch5R3
1__inference_cond_Assert_AssertGuard_false_250_146*
output_shapes
: *C
then_branch4R2
0__inference_cond_Assert_AssertGuard_true_249_1102
cond/Assert/AssertGuard?
 cond/Assert/AssertGuard/IdentityIdentity cond/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2"
 cond/Assert/AssertGuard/IdentityZ

cond/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2

cond/Const^
cond/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 2
cond/Const_1^
cond/Const_2Const*
_output_shapes
: *
dtype0*
value	B : 2
cond/Const_2|
cond/IdentityIdentitycond/Const_2:output:0^cond/Assert/AssertGuard*
T0*
_output_shapes
: 2
cond/Identity"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:'???????????????????????????: 22
cond/Assert/AssertGuardcond/Assert/AssertGuard:C ?
=
_output_shapes+
):'???????????????????????????:

_output_shapes
: 
?h
?
!__inference__traced_restore_42889
file_prefix@
1assignvariableop_audio_preprocessing_layer_window:	?<
"assignvariableop_1_conv2d_1_kernel:.
 assignvariableop_2_conv2d_1_bias:<
"assignvariableop_3_conv2d_2_kernel: .
 assignvariableop_4_conv2d_2_bias: <
"assignvariableop_5_conv2d_3_kernel:  .
 assignvariableop_6_conv2d_3_bias: <
"assignvariableop_7_conv2d_4_kernel:  .
 assignvariableop_8_conv2d_4_bias: 5
!assignvariableop_9_dense_1_kernel:
??/
 assignvariableop_10_dense_1_bias:	?A
.assignvariableop_11_classification_head_kernel:	?:
,assignvariableop_12_classification_head_bias:'
assignvariableop_13_adam_iter:	 )
assignvariableop_14_adam_beta_1: )
assignvariableop_15_adam_beta_2: (
assignvariableop_16_adam_decay: 0
&assignvariableop_17_adam_learning_rate: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: #
assignvariableop_20_total: #
assignvariableop_21_count: H
5assignvariableop_22_adam_classification_head_kernel_m:	?A
3assignvariableop_23_adam_classification_head_bias_m:H
5assignvariableop_24_adam_classification_head_kernel_v:	?A
3assignvariableop_25_adam_classification_head_bias_v:
identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/11/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/12/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp1assignvariableop_audio_preprocessing_layer_windowIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_2_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_3_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_3_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_4_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_4_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp.assignvariableop_11_classification_head_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp,assignvariableop_12_classification_head_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_iterIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp&assignvariableop_17_adam_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_classification_head_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_classification_head_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_classification_head_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_classification_head_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
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
?4
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_40970

inputs(
conv2d_1_40853:
conv2d_1_40855:(
conv2d_2_40871: 
conv2d_2_40873: (
conv2d_3_40889:  
conv2d_3_40891: (
conv2d_4_40907:  
conv2d_4_40909: !
dense_1_40940:
??
dense_1_40942:	?,
classification_head_40964:	?'
classification_head_40966:
identity??+classification_head/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_40853conv2d_1_40855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40852?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:p* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40795?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_40871conv2d_2_40873*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:m *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40870?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:
6 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_40807?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_40889conv2d_3_40891*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:	3 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40888?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_40819?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_40907conv2d_4_40909*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40906?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_40831?
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_40919?
dropout_1/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_40926?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_40940dense_1_40942*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_40939?
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_40950?
+classification_head/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0classification_head_40964classification_head_40966*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_classification_head_layer_call_and_return_conditional_losses_40963z
IdentityIdentity4classification_head/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp,^classification_head/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2Z
+classification_head/StatefulPartitionedCall+classification_head/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_42603

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  T
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	?P
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?
?
(__inference_restored_function_body_37644

inputs
unknown
	unknown_0:	?
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_audio_preprocessing_layer_layer_call_and_return_conditional_losses_529?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_42522

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0t
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m O
ReluReluBiasAdd:output:0*
T0*&
_output_shapes
:m `
IdentityIdentityRelu:activations:0^NoOp*
T0*&
_output_shapes
:m w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:p
 
_user_specified_nameinputs
?i
?
 __inference__wrapped_model_40680
audio_preproc_input>
:sequential_3_audio_preproc_audio_preprocessing_layer_40620I
:sequential_3_audio_preproc_audio_preprocessing_layer_40622:	?>
:sequential_3_audio_preproc_audio_preprocessing_layer_40624>
:sequential_3_audio_preproc_audio_preprocessing_layer_40626[
Asequential_3_sequential_1_conv2d_1_conv2d_readvariableop_resource:P
Bsequential_3_sequential_1_conv2d_1_biasadd_readvariableop_resource:[
Asequential_3_sequential_1_conv2d_2_conv2d_readvariableop_resource: P
Bsequential_3_sequential_1_conv2d_2_biasadd_readvariableop_resource: [
Asequential_3_sequential_1_conv2d_3_conv2d_readvariableop_resource:  P
Bsequential_3_sequential_1_conv2d_3_biasadd_readvariableop_resource: [
Asequential_3_sequential_1_conv2d_4_conv2d_readvariableop_resource:  P
Bsequential_3_sequential_1_conv2d_4_biasadd_readvariableop_resource: T
@sequential_3_sequential_1_dense_1_matmul_readvariableop_resource:
??P
Asequential_3_sequential_1_dense_1_biasadd_readvariableop_resource:	?_
Lsequential_3_sequential_1_classification_head_matmul_readvariableop_resource:	?[
Msequential_3_sequential_1_classification_head_biasadd_readvariableop_resource:
identity??Lsequential_3/audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall?Dsequential_3/sequential_1/classification_head/BiasAdd/ReadVariableOp?Csequential_3/sequential_1/classification_head/MatMul/ReadVariableOp?9sequential_3/sequential_1/conv2d_1/BiasAdd/ReadVariableOp?8sequential_3/sequential_1/conv2d_1/Conv2D/ReadVariableOp?9sequential_3/sequential_1/conv2d_2/BiasAdd/ReadVariableOp?8sequential_3/sequential_1/conv2d_2/Conv2D/ReadVariableOp?9sequential_3/sequential_1/conv2d_3/BiasAdd/ReadVariableOp?8sequential_3/sequential_1/conv2d_3/Conv2D/ReadVariableOp?9sequential_3/sequential_1/conv2d_4/BiasAdd/ReadVariableOp?8sequential_3/sequential_1/conv2d_4/Conv2D/ReadVariableOp?8sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOp?7sequential_3/sequential_1/dense_1/MatMul/ReadVariableOp?
Lsequential_3/audio_preproc/audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallaudio_preproc_input:sequential_3_audio_preproc_audio_preprocessing_layer_40620:sequential_3_audio_preproc_audio_preprocessing_layer_40622:sequential_3_audio_preproc_audio_preprocessing_layer_40624:sequential_3_audio_preproc_audio_preprocessing_layer_40626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:+?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
8sequential_3/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpAsequential_3_sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
)sequential_3/sequential_1/conv2d_1/Conv2DConv2DUsequential_3/audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall:output:0@sequential_3/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
9sequential_3/sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpBsequential_3_sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*sequential_3/sequential_1/conv2d_1/BiasAddBiasAdd2sequential_3/sequential_1/conv2d_1/Conv2D:output:0Asequential_3/sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*??
'sequential_3/sequential_1/conv2d_1/ReluRelu3sequential_3/sequential_1/conv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
1sequential_3/sequential_1/max_pooling2d_1/MaxPoolMaxPool5sequential_3/sequential_1/conv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
8sequential_3/sequential_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOpAsequential_3_sequential_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
)sequential_3/sequential_1/conv2d_2/Conv2DConv2D:sequential_3/sequential_1/max_pooling2d_1/MaxPool:output:0@sequential_3/sequential_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
9sequential_3/sequential_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpBsequential_3_sequential_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
*sequential_3/sequential_1/conv2d_2/BiasAddBiasAdd2sequential_3/sequential_1/conv2d_2/Conv2D:output:0Asequential_3/sequential_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m ?
'sequential_3/sequential_1/conv2d_2/ReluRelu3sequential_3/sequential_1/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
1sequential_3/sequential_1/max_pooling2d_2/MaxPoolMaxPool5sequential_3/sequential_1/conv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
8sequential_3/sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOpAsequential_3_sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
)sequential_3/sequential_1/conv2d_3/Conv2DConv2D:sequential_3/sequential_1/max_pooling2d_2/MaxPool:output:0@sequential_3/sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
9sequential_3/sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpBsequential_3_sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
*sequential_3/sequential_1/conv2d_3/BiasAddBiasAdd2sequential_3/sequential_1/conv2d_3/Conv2D:output:0Asequential_3/sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 ?
'sequential_3/sequential_1/conv2d_3/ReluRelu3sequential_3/sequential_1/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
1sequential_3/sequential_1/max_pooling2d_3/MaxPoolMaxPool5sequential_3/sequential_1/conv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
8sequential_3/sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOpAsequential_3_sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
)sequential_3/sequential_1/conv2d_4/Conv2DConv2D:sequential_3/sequential_1/max_pooling2d_3/MaxPool:output:0@sequential_3/sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
9sequential_3/sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpBsequential_3_sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
*sequential_3/sequential_1/conv2d_4/BiasAddBiasAdd2sequential_3/sequential_1/conv2d_4/Conv2D:output:0Asequential_3/sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
'sequential_3/sequential_1/conv2d_4/ReluRelu3sequential_3/sequential_1/conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
1sequential_3/sequential_1/max_pooling2d_4/MaxPoolMaxPool5sequential_3/sequential_1/conv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
z
)sequential_3/sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
+sequential_3/sequential_1/flatten_1/ReshapeReshape:sequential_3/sequential_1/max_pooling2d_4/MaxPool:output:02sequential_3/sequential_1/flatten_1/Const:output:0*
T0*
_output_shapes
:	??
,sequential_3/sequential_1/dropout_1/IdentityIdentity4sequential_3/sequential_1/flatten_1/Reshape:output:0*
T0*
_output_shapes
:	??
7sequential_3/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp@sequential_3_sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
(sequential_3/sequential_1/dense_1/MatMulMatMul5sequential_3/sequential_1/dropout_1/Identity:output:0?sequential_3/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
8sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpAsequential_3_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)sequential_3/sequential_1/dense_1/BiasAddBiasAdd2sequential_3/sequential_1/dense_1/MatMul:product:0@sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
&sequential_3/sequential_1/dense_1/ReluRelu2sequential_3/sequential_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	??
,sequential_3/sequential_1/dropout_2/IdentityIdentity4sequential_3/sequential_1/dense_1/Relu:activations:0*
T0*
_output_shapes
:	??
Csequential_3/sequential_1/classification_head/MatMul/ReadVariableOpReadVariableOpLsequential_3_sequential_1_classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
4sequential_3/sequential_1/classification_head/MatMulMatMul5sequential_3/sequential_1/dropout_2/Identity:output:0Ksequential_3/sequential_1/classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
Dsequential_3/sequential_1/classification_head/BiasAdd/ReadVariableOpReadVariableOpMsequential_3_sequential_1_classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_3/sequential_1/classification_head/BiasAddBiasAdd>sequential_3/sequential_1/classification_head/MatMul:product:0Lsequential_3/sequential_1/classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:?
5sequential_3/sequential_1/classification_head/SoftmaxSoftmax>sequential_3/sequential_1/classification_head/BiasAdd:output:0*
T0*
_output_shapes

:?
IdentityIdentity?sequential_3/sequential_1/classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOpM^sequential_3/audio_preproc/audio_preprocessing_layer/StatefulPartitionedCallE^sequential_3/sequential_1/classification_head/BiasAdd/ReadVariableOpD^sequential_3/sequential_1/classification_head/MatMul/ReadVariableOp:^sequential_3/sequential_1/conv2d_1/BiasAdd/ReadVariableOp9^sequential_3/sequential_1/conv2d_1/Conv2D/ReadVariableOp:^sequential_3/sequential_1/conv2d_2/BiasAdd/ReadVariableOp9^sequential_3/sequential_1/conv2d_2/Conv2D/ReadVariableOp:^sequential_3/sequential_1/conv2d_3/BiasAdd/ReadVariableOp9^sequential_3/sequential_1/conv2d_3/Conv2D/ReadVariableOp:^sequential_3/sequential_1/conv2d_4/BiasAdd/ReadVariableOp9^sequential_3/sequential_1/conv2d_4/Conv2D/ReadVariableOp9^sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOp8^sequential_3/sequential_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 2?
Lsequential_3/audio_preproc/audio_preprocessing_layer/StatefulPartitionedCallLsequential_3/audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall2?
Dsequential_3/sequential_1/classification_head/BiasAdd/ReadVariableOpDsequential_3/sequential_1/classification_head/BiasAdd/ReadVariableOp2?
Csequential_3/sequential_1/classification_head/MatMul/ReadVariableOpCsequential_3/sequential_1/classification_head/MatMul/ReadVariableOp2v
9sequential_3/sequential_1/conv2d_1/BiasAdd/ReadVariableOp9sequential_3/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2t
8sequential_3/sequential_1/conv2d_1/Conv2D/ReadVariableOp8sequential_3/sequential_1/conv2d_1/Conv2D/ReadVariableOp2v
9sequential_3/sequential_1/conv2d_2/BiasAdd/ReadVariableOp9sequential_3/sequential_1/conv2d_2/BiasAdd/ReadVariableOp2t
8sequential_3/sequential_1/conv2d_2/Conv2D/ReadVariableOp8sequential_3/sequential_1/conv2d_2/Conv2D/ReadVariableOp2v
9sequential_3/sequential_1/conv2d_3/BiasAdd/ReadVariableOp9sequential_3/sequential_1/conv2d_3/BiasAdd/ReadVariableOp2t
8sequential_3/sequential_1/conv2d_3/Conv2D/ReadVariableOp8sequential_3/sequential_1/conv2d_3/Conv2D/ReadVariableOp2v
9sequential_3/sequential_1/conv2d_4/BiasAdd/ReadVariableOp9sequential_3/sequential_1/conv2d_4/BiasAdd/ReadVariableOp2t
8sequential_3/sequential_1/conv2d_4/Conv2D/ReadVariableOp8sequential_3/sequential_1/conv2d_4/Conv2D/ReadVariableOp2t
8sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOp8sequential_3/sequential_1/dense_1/BiasAdd/ReadVariableOp2r
7sequential_3/sequential_1/dense_1/MatMul/ReadVariableOp7sequential_3/sequential_1/dense_1/MatMul/ReadVariableOp:^ Z
)
_output_shapes
:???????????
-
_user_specified_nameaudio_preproc_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40870

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0t
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m O
ReluReluBiasAdd:output:0*
T0*&
_output_shapes
:m `
IdentityIdentityRelu:activations:0^NoOp*
T0*&
_output_shapes
:m w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:p
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_42170

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_41181f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?
?
7__inference_audio_preprocessing_layer_layer_call_fn_496

inputs
unknown
	unknown_0:	?
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_audio_preprocessing_layer_layer_call_and_return_conditional_losses_4242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: :: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_Assert_AssertGuard_true_298_41,
(assert_assertguard_identity_greaterequal
"
assert_assertguard_placeholder!
assert_assertguard_identity_1
P
Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2
Assert/AssertGuard/NoOp?
Assert/AssertGuard/IdentityIdentity(assert_assertguard_identity_greaterequal^Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+: :'???????????????????????????: 

_output_shapes
: :C?
=
_output_shapes+
):'???????????????????????????
?
?
,__inference_Assert_AssertGuard_false_664_174#
assert_assertguard_assert_equal
$
 assert_assertguard_assert_inputs!
assert_assertguard_identity_1
??Assert/AssertGuard/Assert?
Assert/AssertGuard/AssertAssertassert_assertguard_assert_equal assert_assertguard_assert_inputs*

T
2*
_output_shapes
 2
Assert/AssertGuard/Assert?
Assert/AssertGuard/IdentityIdentityassert_assertguard_assert_equal^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: :???????????26
Assert/AssertGuard/AssertAssert/AssertGuard/Assert: 

_output_shapes
: :/+
)
_output_shapes
:???????????
?
?
-__inference_audio_preproc_layer_call_fn_42086

inputs
unknown
	unknown_0:	?
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40736?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?@
?	
G__inference_sequential_1_layer_call_and_return_conditional_losses_42404

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:  6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:  6
(conv2d_4_biasadd_readvariableop_resource: :
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?E
2classification_head_matmul_readvariableop_resource:	?A
3classification_head_biasadd_readvariableop_resource:
identity??*classification_head/BiasAdd/ReadVariableOp?)classification_head/MatMul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?b
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m a
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 a
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: a
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_1/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_1/Const:output:0*
T0*
_output_shapes
:	?d
dropout_1/IdentityIdentityflatten_1/Reshape:output:0*
T0*
_output_shapes
:	??
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?X
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?d
dropout_2/IdentityIdentitydense_1/Relu:activations:0*
T0*
_output_shapes
:	??
)classification_head/MatMul/ReadVariableOpReadVariableOp2classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
classification_head/MatMulMatMuldropout_2/Identity:output:01classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
*classification_head/BiasAdd/ReadVariableOpReadVariableOp3classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
classification_head/BiasAddBiasAdd$classification_head/MatMul:product:02classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
classification_head/SoftmaxSoftmax$classification_head/BiasAdd:output:0*
T0*
_output_shapes

:k
IdentityIdentity%classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp+^classification_head/BiasAdd/ReadVariableOp*^classification_head/MatMul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2X
*classification_head/BiasAdd/ReadVariableOp*classification_head/BiasAdd/ReadVariableOp2V
)classification_head/MatMul/ReadVariableOp)classification_head/MatMul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:a ]
9
_output_shapes'
%:#???????????????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_42582

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0t
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: O
ReluReluBiasAdd:output:0*
T0*&
_output_shapes
: `
IdentityIdentityRelu:activations:0^NoOp*
T0*&
_output_shapes
: w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_40919

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  T
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	?P
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?
?
!__inference_signature_wrapper_391#
audio_preprocessing_layer_input
unknown
	unknown_0:	?
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallaudio_preprocessing_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_3822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: :: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
)
_output_shapes
:???????????
9
_user_specified_name!audio_preprocessing_layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40888

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0t
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 O
ReluReluBiasAdd:output:0*
T0*&
_output_shapes
:	3 `
IdentityIdentityRelu:activations:0^NoOp*
T0*&
_output_shapes
:	3 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
6 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:
6 
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_42592

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
y
*__inference_tf_webaudio_power_spectrum_233
waveform_frame

window
identity??Assert/AssertGuard?condn
Rank/ReadVariableOpReadVariableOpwindow*
_output_shapes	
:?*
dtype02
Rank/ReadVariableOpN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankT
Equal/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Equal/yy
EqualEqualRank:output:0Equal/y:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equalr
Assert/ReadVariableOpReadVariableOpwindow*
_output_shapes	
:?*
dtype02
Assert/ReadVariableOp?
Assert/AssertGuardIf	Equal:z:0	Equal:z:0window*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*>
else_branch/R-
+__inference_Assert_AssertGuard_false_198_18*
output_shapes
: *>
then_branch/R-
+__inference_Assert_AssertGuard_true_197_1862
Assert/AssertGuard?
Assert/AssertGuard/IdentityIdentityAssert/AssertGuard:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identityn
Size/ReadVariableOpReadVariableOpwindow*
_output_shapes	
:?*
dtype02
Size/ReadVariableOpO
SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
SizeR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
	Equal_1/y?
Equal_1EqualRank_1:output:0Equal_1/y:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_1?
condIfEqual_1:z:0waveform_frameSize:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *1
else_branch"R 
__inference_cond_false_218_162*
output_shapes
: *0
then_branch!R
__inference_cond_true_217_2082
condZ
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes
: 2
cond/Identityl
mul/ReadVariableOpReadVariableOpwindow*
_output_shapes	
:?*
dtype02
mul/ReadVariableOp}
mulMulwaveform_framemul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2
mull
CastCastmul:z:0*

DstT0*

SrcT0*5
_output_shapes#
!:???????????????????2
CastR
FFTFFTCast:y:0*5
_output_shapes#
!:???????????????????2
FFT]
Abs
ComplexAbsFFT:output:0*5
_output_shapes#
!:???????????????????2
AbsZ
LogLogAbs:y:0*
T0*5
_output_shapes#
!:???????????????????2
Log?
IdentityIdentityLog:y:0^Assert/AssertGuard^cond*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:'???????????????????????????:2(
Assert/AssertGuardAssert/AssertGuard2
condcond:m i
=
_output_shapes+
):'???????????????????????????
(
_user_specified_namewaveform_frame:&"
 
_user_specified_namewindow
?
?
+__inference_Assert_AssertGuard_false_415_92#
assert_assertguard_assert_equal
$
 assert_assertguard_assert_inputs!
assert_assertguard_identity_1
??Assert/AssertGuard/Assert?
Assert/AssertGuard/AssertAssertassert_assertguard_assert_equal assert_assertguard_assert_inputs*

T
2*
_output_shapes
 2
Assert/AssertGuard/Assert?
Assert/AssertGuard/IdentityIdentityassert_assertguard_assert_equal^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: :???????????26
Assert/AssertGuard/AssertAssert/AssertGuard/Assert: 

_output_shapes
: :/+
)
_output_shapes
:???????????
?
?
__inference__wrapped_model_382#
audio_preprocessing_layer_input/
+audio_preproc_audio_preprocessing_layer_355:
+audio_preproc_audio_preprocessing_layer_357:	?/
+audio_preproc_audio_preprocessing_layer_359/
+audio_preproc_audio_preprocessing_layer_361
identity???audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall?
?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallaudio_preprocessing_layer_input+audio_preproc_audio_preprocessing_layer_355+audio_preproc_audio_preprocessing_layer_357+audio_preproc_audio_preprocessing_layer_359+audio_preproc_audio_preprocessing_layer_361*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_3642A
?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall?
IdentityIdentityHaudio_preproc/audio_preprocessing_layer/StatefulPartitionedCall:output:0@^audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: :: : 2?
?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall?audio_preproc/audio_preprocessing_layer/StatefulPartitionedCall:j f
)
_output_shapes
:???????????
9
_user_specified_name!audio_preprocessing_layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?O
?	
G__inference_sequential_1_layer_call_and_return_conditional_losses_41549

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:  6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:  6
(conv2d_4_biasadd_readvariableop_resource: :
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?E
2classification_head_matmul_readvariableop_resource:	?A
3classification_head_biasadd_readvariableop_resource:
identity??*classification_head/BiasAdd/ReadVariableOp?)classification_head/MatMul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?b
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m a
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 a
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: a
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_1/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_1/Const:output:0*
T0*
_output_shapes
:	?\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_1/dropout/MulMulflatten_1/Reshape:output:0 dropout_1/dropout/Const:output:0*
T0*
_output_shapes
:	?h
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?{
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*
_output_shapes
:	??
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?X
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_2/dropout/MulMuldense_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	?h
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?{
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	??
)classification_head/MatMul/ReadVariableOpReadVariableOp2classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
classification_head/MatMulMatMuldropout_2/dropout/Mul_1:z:01classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
*classification_head/BiasAdd/ReadVariableOpReadVariableOp3classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
classification_head/BiasAddBiasAdd$classification_head/MatMul:product:02classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
classification_head/SoftmaxSoftmax$classification_head/BiasAdd:output:0*
T0*
_output_shapes

:k
IdentityIdentity%classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp+^classification_head/BiasAdd/ReadVariableOp*^classification_head/MatMul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2X
*classification_head/BiasAdd/ReadVariableOp*classification_head/BiasAdd/ReadVariableOp2V
)classification_head/MatMul/ReadVariableOp)classification_head/MatMul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:a ]
9
_output_shapes'
%:#???????????????????
 
_user_specified_nameinputs
?	
?
N__inference_classification_head_layer_call_and_return_conditional_losses_40963

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:M
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:W
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
_
&__inference_z_normalize_spectrogram_67
spectrogram
identity??Assert/AssertGuardN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
Rankb
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B :2
GreaterEqual/yu
GreaterEqualGreaterEqualRank:output:0GreaterEqual/y:output:0*
T0*
_output_shapes
: 2
GreaterEqual?
Assert/AssertGuardIfGreaterEqual:z:0GreaterEqual:z:0spectrogram*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *>
else_branch/R-
+__inference_Assert_AssertGuard_false_299_47*
output_shapes
: *=
then_branch.R,
*__inference_Assert_AssertGuard_true_298_412
Assert/AssertGuard?
Assert/AssertGuard/IdentityIdentityAssert/AssertGuard:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????????2 
moments/mean/reduction_indices?
moments/meanMeanspectrogram'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:?????????2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferencespectrogrammoments/StopGradient:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????????2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
moments/varianceS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??82
Const}
subSubspectrogrammoments/mean:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
sube
SqrtSqrtmoments/variance:output:0*
T0*+
_output_shapes
:?????????2
Sqrtc
addAddV2Sqrt:y:0Const:output:0*
T0*+
_output_shapes
:?????????2
addw
truedivRealDivsub:z:0add:z:0*
T0*=
_output_shapes+
):'???????????????????????????2	
truediv?
IdentityIdentitytruediv:z:0^Assert/AssertGuard*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????2(
Assert/AssertGuardAssert/AssertGuard:j f
=
_output_shapes+
):'???????????????????????????
%
_user_specified_namespectrogram
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_40807

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_42502

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_Assert_AssertGuard_false_65_24#
assert_assertguard_assert_equal
$
 assert_assertguard_assert_inputs!
assert_assertguard_identity_1
??Assert/AssertGuard/Assert?
Assert/AssertGuard/AssertAssertassert_assertguard_assert_equal assert_assertguard_assert_inputs*

T
2*
_output_shapes
 2
Assert/AssertGuard/Assert?
Assert/AssertGuard/IdentityIdentityassert_assertguard_assert_equal^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: :???????????26
Assert/AssertGuard/AssertAssert/AssertGuard/Assert: 

_output_shapes
: :/+
)
_output_shapes
:???????????
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40906

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0t
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: O
ReluReluBiasAdd:output:0*
T0*&
_output_shapes
: `
IdentityIdentityRelu:activations:0^NoOp*
T0*&
_output_shapes
: w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_42608

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_40926X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
(__inference_conv2d_4_layer_call_fn_42571

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40906n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_41237
conv2d_1_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_41181f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:+?
(
_user_specified_nameconv2d_1_input
?
?
+__inference_Assert_AssertGuard_true_663_104%
!assert_assertguard_identity_equal
"
assert_assertguard_placeholder!
assert_assertguard_identity_1
P
Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2
Assert/AssertGuard/NoOp?
Assert/AssertGuard/IdentityIdentity!assert_assertguard_identity_equal^Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: :???????????: 

_output_shapes
: :/+
)
_output_shapes
:???????????
?
K
/__inference_max_pooling2d_3_layer_call_fn_42557

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_40819?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_40831

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_41060

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????\
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	?^
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?g
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?a
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	?Q
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_40997
conv2d_1_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_40970f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:+?
(
_user_specified_nameconv2d_1_input
?
b
)__inference_dropout_2_layer_call_fn_42660

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_41027g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_42618

inputs

identity_1F
IdentityIdentityinputs*
T0*
_output_shapes
:	?S

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_42497

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40795?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_40926

inputs

identity_1F
IdentityIdentityinputs*
T0*
_output_shapes
:	?S

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_42532

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40852

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0u
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:*?a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:*?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:+?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?	
?
+__inference_Assert_AssertGuard_false_198_18#
assert_assertguard_assert_equal
5
1assert_assertguard_assert_readvariableop_resource!
assert_assertguard_identity_1
??Assert/AssertGuard/Assert?
(Assert/AssertGuard/Assert/ReadVariableOpReadVariableOp1assert_assertguard_assert_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(Assert/AssertGuard/Assert/ReadVariableOp?
Assert/AssertGuard/AssertAssertassert_assertguard_assert_equal0Assert/AssertGuard/Assert/ReadVariableOp:value:0*

T
2*
_output_shapes
 2
Assert/AssertGuard/Assert?
Assert/AssertGuard/IdentityIdentityassert_assertguard_assert_equal^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :26
Assert/AssertGuard/AssertAssert/AssertGuard/Assert: 

_output_shapes
: 
?@
?	
G__inference_sequential_1_layer_call_and_return_conditional_losses_41388

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:  6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:  6
(conv2d_4_biasadd_readvariableop_resource: :
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?E
2classification_head_matmul_readvariableop_resource:	?A
3classification_head_biasadd_readvariableop_resource:
identity??*classification_head/BiasAdd/ReadVariableOp?)classification_head/MatMul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?b
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m a
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 a
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: a
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_1/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_1/Const:output:0*
T0*
_output_shapes
:	?d
dropout_1/IdentityIdentityflatten_1/Reshape:output:0*
T0*
_output_shapes
:	??
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?X
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?d
dropout_2/IdentityIdentitydense_1/Relu:activations:0*
T0*
_output_shapes
:	??
)classification_head/MatMul/ReadVariableOpReadVariableOp2classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
classification_head/MatMulMatMuldropout_2/Identity:output:01classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
*classification_head/BiasAdd/ReadVariableOpReadVariableOp3classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
classification_head/BiasAddBiasAdd$classification_head/MatMul:product:02classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
classification_head/SoftmaxSoftmax$classification_head/BiasAdd:output:0*
T0*
_output_shapes

:k
IdentityIdentity%classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp+^classification_head/BiasAdd/ReadVariableOp*^classification_head/MatMul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2X
*classification_head/BiasAdd/ReadVariableOp*classification_head/BiasAdd/ReadVariableOp2V
)classification_head/MatMul/ReadVariableOp)classification_head/MatMul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:a ]
9
_output_shapes'
%:#???????????????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_42597

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_40919X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?
?
,__inference_sequential_3_layer_call_fn_41725
audio_preproc_input
unknown
	unknown_0:	?
	unknown_1
	unknown_2#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallaudio_preproc_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_41653f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
)
_output_shapes
:???????????
-
_user_specified_nameaudio_preproc_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?7
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_41319
conv2d_1_input(
conv2d_1_41281:
conv2d_1_41283:(
conv2d_2_41287: 
conv2d_2_41289: (
conv2d_3_41293:  
conv2d_3_41295: (
conv2d_4_41299:  
conv2d_4_41301: !
dense_1_41307:
??
dense_1_41309:	?,
classification_head_41313:	?'
classification_head_41315:
identity??+classification_head/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_41281conv2d_1_41283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40852?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:p* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40795?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_41287conv2d_2_41289*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:m *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_40870?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:
6 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_40807?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_41293conv2d_3_41295*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:	3 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_40888?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_40819?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_41299conv2d_4_41301*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_40906?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_40831?
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_40919?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_41060?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_41307dense_1_41309*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_40939?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_41027?
+classification_head/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0classification_head_41313classification_head_41315*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_classification_head_layer_call_and_return_conditional_losses_40963z
IdentityIdentity4classification_head/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp,^classification_head/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2Z
+classification_head/StatefulPartitionedCall+classification_head/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:W S
'
_output_shapes
:+?
(
_user_specified_nameconv2d_1_input
?	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_42630

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????\
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	?^
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?g
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?a
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	?Q
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_2_layer_call_fn_42527

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_40807?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_42112

inputs#
audio_preprocessing_layer_42102.
audio_preprocessing_layer_42104:	?#
audio_preprocessing_layer_42106#
audio_preprocessing_layer_42108
identity??1audio_preprocessing_layer/StatefulPartitionedCall?
1audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallinputsaudio_preprocessing_layer_42102audio_preprocessing_layer_42104audio_preprocessing_layer_42106audio_preprocessing_layer_42108*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
IdentityIdentity:audio_preprocessing_layer/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????z
NoOpNoOp2^audio_preprocessing_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 2f
1audio_preprocessing_layer/StatefulPartitionedCall1audio_preprocessing_layer/StatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_1_layer_call_fn_42639

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_40939g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
R__inference_audio_preprocessing_layer_layer_call_and_return_conditional_losses_529

inputs
	equal_1_x
unknown:	?
	unknown_0
	unknown_1
identity??Assert/AssertGuard?Assert_1/Assert?StatefulPartitionedCall?StatefulPartitionedCall_1N
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankT
Equal/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Equal/yy
EqualEqualRank:output:0Equal/y:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal?
Assert/AssertGuardIf	Equal:z:0	Equal:z:0inputs*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *?
else_branch0R.
,__inference_Assert_AssertGuard_false_664_174*
output_shapes
: *>
then_branch/R-
+__inference_Assert_AssertGuard_true_663_1042
Assert/AssertGuard?
Assert/AssertGuard/IdentityIdentityAssert/AssertGuard:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/IdentityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
Equal_1Equal	equal_1_xstrided_slice:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_1H
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1?
Assert_1/AssertAssertEqual_1:z:0Shape_1:output:0^Assert/AssertGuard*

T
2*
_output_shapes
 2
Assert_1/Assert?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_tf_webaudio_spectrogram_3162
StatefulPartitionedCall?
StatefulPartitionedCall_1StatefulPartitionedCall StatefulPartitionedCall:output:0^Assert_1/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_z_normalize_spectrogram_672
StatefulPartitionedCall_1k
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDims"StatefulPartitionedCall_1:output:0ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
IdentityIdentityExpandDims:output:0^Assert/AssertGuard^Assert_1/Assert^StatefulPartitionedCall^StatefulPartitionedCall_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: :: : 2(
Assert/AssertGuardAssert/AssertGuard2"
Assert_1/AssertAssert_1/Assert22
StatefulPartitionedCallStatefulPartitionedCall26
StatefulPartitionedCall_1StatefulPartitionedCall_1:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?O
?	
G__inference_sequential_1_layer_call_and_return_conditional_losses_42472

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:  6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:  6
(conv2d_4_biasadd_readvariableop_resource: :
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?E
2classification_head_matmul_readvariableop_resource:	?A
3classification_head_biasadd_readvariableop_resource:
identity??*classification_head/BiasAdd/ReadVariableOp?)classification_head/MatMul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?b
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m a
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 a
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: a
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_1/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_1/Const:output:0*
T0*
_output_shapes
:	?\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_1/dropout/MulMulflatten_1/Reshape:output:0 dropout_1/dropout/Const:output:0*
T0*
_output_shapes
:	?h
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?{
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*
_output_shapes
:	??
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?X
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_2/dropout/MulMuldense_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	?h
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?{
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	??
)classification_head/MatMul/ReadVariableOpReadVariableOp2classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
classification_head/MatMulMatMuldropout_2/dropout/Mul_1:z:01classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
*classification_head/BiasAdd/ReadVariableOpReadVariableOp3classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
classification_head/BiasAddBiasAdd$classification_head/MatMul:product:02classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
classification_head/SoftmaxSoftmax$classification_head/BiasAdd:output:0*
T0*
_output_shapes

:k
IdentityIdentity%classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp+^classification_head/BiasAdd/ReadVariableOp*^classification_head/MatMul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2X
*classification_head/BiasAdd/ReadVariableOp*classification_head/BiasAdd/ReadVariableOp2V
)classification_head/MatMul/ReadVariableOp)classification_head/MatMul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:a ]
9
_output_shapes'
%:#???????????????????
 
_user_specified_nameinputs
?
?
*__inference_Assert_AssertGuard_true_64_168%
!assert_assertguard_identity_equal
"
assert_assertguard_placeholder!
assert_assertguard_identity_1
P
Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2
Assert/AssertGuard/NoOp?
Assert/AssertGuard/IdentityIdentity!assert_assertguard_identity_equal^Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: :???????????: 

_output_shapes
: :/+
)
_output_shapes
:???????????
?	
?
-__inference_audio_preproc_layer_call_fn_40760#
audio_preprocessing_layer_input
unknown
	unknown_0:	?
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallaudio_preprocessing_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40736?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
)
_output_shapes
:???????????
9
_user_specified_name!audio_preprocessing_layer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
0__inference_cond_Assert_AssertGuard_false_225_73-
)cond_assert_assertguard_assert_cond_equal
1
-cond_assert_assertguard_assert_waveform_frame&
"cond_assert_assertguard_identity_1
??cond/Assert/AssertGuard/Assert?
cond/Assert/AssertGuard/AssertAssert)cond_assert_assertguard_assert_cond_equal-cond_assert_assertguard_assert_waveform_frame*

T
2*
_output_shapes
 2 
cond/Assert/AssertGuard/Assert?
 cond/Assert/AssertGuard/IdentityIdentity)cond_assert_assertguard_assert_cond_equal^cond/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2"
 cond/Assert/AssertGuard/Identity?
"cond/Assert/AssertGuard/Identity_1Identity)cond/Assert/AssertGuard/Identity:output:0^cond/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2$
"cond/Assert/AssertGuard/Identity_1"Q
"cond_assert_assertguard_identity_1+cond/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+: :'???????????????????????????2@
cond/Assert/AssertGuard/Assertcond/Assert/AssertGuard/Assert: 

_output_shapes
: :C?
=
_output_shapes+
):'???????????????????????????
?
?
0__inference_cond_Assert_AssertGuard_true_224_198/
+cond_assert_assertguard_identity_cond_equal
'
#cond_assert_assertguard_placeholder&
"cond_assert_assertguard_identity_1
Z
cond/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2
cond/Assert/AssertGuard/NoOp?
 cond/Assert/AssertGuard/IdentityIdentity+cond_assert_assertguard_identity_cond_equal^cond/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2"
 cond/Assert/AssertGuard/Identity?
"cond/Assert/AssertGuard/Identity_1Identity)cond/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2$
"cond/Assert/AssertGuard/Identity_1"Q
"cond_assert_assertguard_identity_1+cond/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+: :'???????????????????????????: 

_output_shapes
: :C?
=
_output_shapes+
):'???????????????????????????
?
?
,__inference_sequential_3_layer_call_fn_41883

inputs
unknown
	unknown_0:	?
	unknown_1
	unknown_2#
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_41415f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_classification_head_layer_call_fn_42686

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_classification_head_layer_call_and_return_conditional_losses_40963f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?	
?
N__inference_classification_head_layer_call_and_return_conditional_losses_42697

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:M
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:W
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41415

inputs
audio_preproc_41326"
audio_preproc_41328:	?
audio_preproc_41330
audio_preproc_41332,
sequential_1_41389: 
sequential_1_41391:,
sequential_1_41393:  
sequential_1_41395: ,
sequential_1_41397:   
sequential_1_41399: ,
sequential_1_41401:   
sequential_1_41403: &
sequential_1_41405:
??!
sequential_1_41407:	?%
sequential_1_41409:	? 
sequential_1_41411:
identity??%audio_preproc/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?
%audio_preproc/StatefulPartitionedCallStatefulPartitionedCallinputsaudio_preproc_41326audio_preproc_41328audio_preproc_41330audio_preproc_41332*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40697?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall.audio_preproc/StatefulPartitionedCall:output:0sequential_1_41389sequential_1_41391sequential_1_41393sequential_1_41395sequential_1_41397sequential_1_41399sequential_1_41401sequential_1_41403sequential_1_41405sequential_1_41407sequential_1_41409sequential_1_41411*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_41388s
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp&^audio_preproc/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 2N
%audio_preproc/StatefulPartitionedCall%audio_preproc/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?\
?
'__inference_tf_webaudio_spectrogram_316
waveform

window
frame_shift
frequency_truncation
identity??Assert/AssertGuard?Assert_1/AssertGuard?Assert_2/Assert?Assert_3/Assert?Assert_4/Assert?Assert_5/Assert?StatefulPartitionedCallN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankT
Equal/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Equal/yy
EqualEqualRank:output:0Equal/y:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal?
Assert/AssertGuardIf	Equal:z:0	Equal:z:0waveform*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *>
else_branch/R-
+__inference_Assert_AssertGuard_false_96_180*
output_shapes
: *=
then_branch.R,
*__inference_Assert_AssertGuard_true_95_1162
Assert/AssertGuard?
Assert/AssertGuard/IdentityIdentityAssert/AssertGuard:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identityr
Rank_1/ReadVariableOpReadVariableOpwindow*
_output_shapes	
:?*
dtype02
Rank_1/ReadVariableOpR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
	Equal_1/y?
Equal_1EqualRank_1:output:0Equal_1/y:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_1v
Assert_1/ReadVariableOpReadVariableOpwindow*
_output_shapes	
:?*
dtype02
Assert_1/ReadVariableOp?
Assert_1/AssertGuardIfEqual_1:z:0Equal_1:z:0window*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *#
_read_only_resource_inputs
*@
else_branch1R/
-__inference_Assert_1_AssertGuard_false_115_86*
output_shapes
: *@
then_branch1R/
-__inference_Assert_1_AssertGuard_true_114_1922
Assert_1/AssertGuard?
Assert_1/AssertGuard/IdentityIdentityAssert_1/AssertGuard:output:0*
T0
*
_output_shapes
: 2
Assert_1/AssertGuard/IdentityF
ShapeShapewaveform*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicen
Size/ReadVariableOpReadVariableOpwindow*
_output_shapes	
:?*
dtype02
Size/ReadVariableOpO
SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
SizeZ

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :2

floordiv/ye
floordivFloorDivSize:output:0floordiv/y:output:0*
T0*
_output_shapes
: 2

floordivw
Equal_2Equalframe_shiftfloordiv:z:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_2?
Assert_2/AssertAssertEqual_2:z:0frame_shiftSize:output:0^Assert/AssertGuard*
T
2*
_output_shapes
 2
Assert_2/AssertC
Size_1Sizewaveform*
T0*
_output_shapes
: 2
Size_1n

floordiv_1FloorDivSize_1:output:0strided_slice:output:0*
T0*
_output_shapes
: 2

floordiv_1b

floordiv_2FloorDivfloordiv_1:z:0frame_shift*
T0*
_output_shapes
: 2

floordiv_2O
mulMulframe_shiftfloordiv_2:z:0*
T0*
_output_shapes
: 2
mulk
Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
Slice/beging
Slice/size/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Slice/size/0n

Slice/sizePackSlice/size/0:output:0mul:z:0*
N*
T0*
_output_shapes
:2

Slice/size?
SliceSlicewaveformSlice/begin:output:0Slice/size:output:0*
Index0*
T0*0
_output_shapes
:??????????????????2
Slicec
	zeros/mulMulstrided_slice:output:0frame_shift*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessw
zeros/packedPackstrided_slice:output:0frame_shift*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const~
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*0
_output_shapes
:??????????????????2
zeros\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2zeros:output:0Slice:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????????????2
concat?
Reshape/shapePackstrided_slice:output:0floordiv_2:z:0frame_shift*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeSlice:output:0Reshape/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
ReshapeP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yT
addAddV2floordiv_2:z:0add/y:output:0*
T0*
_output_shapes
: 2
add?
Reshape_1/shapePackstrided_slice:output:0add:z:0frame_shift*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapeconcat:output:0Reshape_1/shape:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Reshape_1s
Slice_1/beginConst*
_output_shapes
:*
dtype0*!
valueB"            2
Slice_1/begink
Slice_1/size/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Slice_1/size/0?
Slice_1/sizePackSlice_1/size/0:output:0floordiv_2:z:0frame_shift*
N*
T0*
_output_shapes
:2
Slice_1/size?
Slice_1SliceReshape_1:output:0Slice_1/begin:output:0Slice_1/size:output:0*
Index0*
T0*=
_output_shapes+
):'???????????????????????????2	
Slice_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2Slice_1:output:0Reshape:output:0concat_1/axis:output:0*
N*
T0*=
_output_shapes+
):'???????????????????????????2

concat_1S
Shape_1Shapeconcat_1:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
Equal_3Equalstrided_slice_1:output:0strided_slice:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_3S
Shape_2Shapeconcat_1:output:0*
T0*
_output_shapes
:2	
Shape_2?
Assert_3/AssertAssertEqual_3:z:0Shape_2:output:0strided_slice:output:0^Assert_2/Assert*
T
2*
_output_shapes
 2
Assert_3/AssertS
Shape_3Shapeconcat_1:output:0*
T0*
_output_shapes
:2	
Shape_3x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape_3:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2?
Equal_4Equalstrided_slice_2:output:0floordiv_2:z:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_4S
Shape_4Shapeconcat_1:output:0*
T0*
_output_shapes
:2	
Shape_4?
Assert_4/AssertAssertEqual_4:z:0Shape_4:output:0floordiv_2:z:0^Assert_3/Assert*
T
2*
_output_shapes
 2
Assert_4/AssertS
Shape_5Shapeconcat_1:output:0*
T0*
_output_shapes
:2	
Shape_5x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape_5:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
Equal_5Equalstrided_slice_3:output:0Size:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2	
Equal_5S
Shape_6Shapeconcat_1:output:0*
T0*
_output_shapes
:2	
Shape_6?
Assert_5/AssertAssertEqual_5:z:0Shape_6:output:0Size:output:0^Assert_4/Assert*
T
2*
_output_shapes
 2
Assert_5/Assert?
StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0window*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *3
f.R,
*__inference_tf_webaudio_power_spectrum_2332
StatefulPartitionedCalls
Slice_2/beginConst*
_output_shapes
:*
dtype0*!
valueB"            2
Slice_2/begink
Slice_2/size/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Slice_2/size/0?
Slice_2/sizePackSlice_2/size/0:output:0floordiv_2:z:0frequency_truncation*
N*
T0*
_output_shapes
:2
Slice_2/size?
Slice_2Slice StatefulPartitionedCall:output:0Slice_2/begin:output:0Slice_2/size:output:0*
Index0*
T0*=
_output_shapes+
):'???????????????????????????2	
Slice_2?
IdentityIdentitySlice_2:output:0^Assert/AssertGuard^Assert_1/AssertGuard^Assert_2/Assert^Assert_3/Assert^Assert_4/Assert^Assert_5/Assert^StatefulPartitionedCall*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:: : 2(
Assert/AssertGuardAssert/AssertGuard2,
Assert_1/AssertGuardAssert_1/AssertGuard2"
Assert_2/AssertAssert_2/Assert2"
Assert_3/AssertAssert_3/Assert2"
Assert_4/AssertAssert_4/Assert2"
Assert_5/AssertAssert_5/Assert22
StatefulPartitionedCallStatefulPartitionedCall:S O
)
_output_shapes
:???????????
"
_user_specified_name
waveform:&"
 
_user_specified_namewindow:C?

_output_shapes
: 
%
_user_specified_nameframe_shift:LH

_output_shapes
: 
.
_user_specified_namefrequency_truncation
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_42492

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0u
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:*?a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:*?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:+?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41653

inputs
audio_preproc_41618"
audio_preproc_41620:	?
audio_preproc_41622
audio_preproc_41624,
sequential_1_41627: 
sequential_1_41629:,
sequential_1_41631:  
sequential_1_41633: ,
sequential_1_41635:   
sequential_1_41637: ,
sequential_1_41639:   
sequential_1_41641: &
sequential_1_41643:
??!
sequential_1_41645:	?%
sequential_1_41647:	? 
sequential_1_41649:
identity??%audio_preproc/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?
%audio_preproc/StatefulPartitionedCallStatefulPartitionedCallinputsaudio_preproc_41618audio_preproc_41620audio_preproc_41622audio_preproc_41624*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40736?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall.audio_preproc/StatefulPartitionedCall:output:0sequential_1_41627sequential_1_41629sequential_1_41631sequential_1_41633sequential_1_41635sequential_1_41637sequential_1_41639sequential_1_41641sequential_1_41643sequential_1_41645sequential_1_41647sequential_1_41649*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_41549s
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp&^audio_preproc/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 2N
%audio_preproc/StatefulPartitionedCall%audio_preproc/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
1__inference_cond_Assert_AssertGuard_false_250_146-
)cond_assert_assertguard_assert_cond_equal
1
-cond_assert_assertguard_assert_waveform_frame&
"cond_assert_assertguard_identity_1
??cond/Assert/AssertGuard/Assert?
cond/Assert/AssertGuard/AssertAssert)cond_assert_assertguard_assert_cond_equal-cond_assert_assertguard_assert_waveform_frame*

T
2*
_output_shapes
 2 
cond/Assert/AssertGuard/Assert?
 cond/Assert/AssertGuard/IdentityIdentity)cond_assert_assertguard_assert_cond_equal^cond/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2"
 cond/Assert/AssertGuard/Identity?
"cond/Assert/AssertGuard/Identity_1Identity)cond/Assert/AssertGuard/Identity:output:0^cond/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2$
"cond/Assert/AssertGuard/Identity_1"Q
"cond_assert_assertguard_identity_1+cond/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+: :'???????????????????????????2@
cond/Assert/AssertGuard/Assertcond/Assert/AssertGuard/Assert: 

_output_shapes
: :C?
=
_output_shapes+
):'???????????????????????????
?
?
*__inference_Assert_AssertGuard_true_414_79%
!assert_assertguard_identity_equal
"
assert_assertguard_placeholder!
assert_assertguard_identity_1
P
Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2
Assert/AssertGuard/NoOp?
Assert/AssertGuard/IdentityIdentity!assert_assertguard_identity_equal^Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: :???????????: 

_output_shapes
: :/+
)
_output_shapes
:???????????
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_42552

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0t
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 O
ReluReluBiasAdd:output:0*
T0*&
_output_shapes
:	3 `
IdentityIdentityRelu:activations:0^NoOp*
T0*&
_output_shapes
:	3 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:
6 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:N J
&
_output_shapes
:
6 
 
_user_specified_nameinputs
?
?
-__inference_Assert_1_AssertGuard_true_114_192)
%assert_1_assertguard_identity_equal_1
$
 assert_1_assertguard_placeholder#
assert_1_assertguard_identity_1
T
Assert_1/AssertGuard/NoOpNoOp*
_output_shapes
 2
Assert_1/AssertGuard/NoOp?
Assert_1/AssertGuard/IdentityIdentity%assert_1_assertguard_identity_equal_1^Assert_1/AssertGuard/NoOp*
T0
*
_output_shapes
: 2
Assert_1/AssertGuard/Identity?
Assert_1/AssertGuard/Identity_1Identity&Assert_1/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2!
Assert_1/AssertGuard/Identity_1"K
assert_1_assertguard_identity_1(Assert_1/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :: 

_output_shapes
: 
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40795

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_42613

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_41060g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_42099

inputs#
audio_preprocessing_layer_42089.
audio_preprocessing_layer_42091:	?#
audio_preprocessing_layer_42093#
audio_preprocessing_layer_42095
identity??1audio_preprocessing_layer/StatefulPartitionedCall?
1audio_preprocessing_layer/StatefulPartitionedCallStatefulPartitionedCallinputsaudio_preprocessing_layer_42089audio_preprocessing_layer_42091audio_preprocessing_layer_42093audio_preprocessing_layer_42095*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_restored_function_body_37644?
IdentityIdentity:audio_preprocessing_layer/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#???????????????????z
NoOpNoOp2^audio_preprocessing_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????: : : : 2f
1audio_preprocessing_layer/StatefulPartitionedCall1audio_preprocessing_layer/StatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_40950

inputs

identity_1F
IdentityIdentityinputs*
T0*
_output_shapes
:	?S

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	?:G C

_output_shapes
:	?
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_42141

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_40970f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?O
?	
G__inference_sequential_1_layer_call_and_return_conditional_losses_42350

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource: A
'conv2d_3_conv2d_readvariableop_resource:  6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:  6
(conv2d_4_biasadd_readvariableop_resource: :
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?E
2classification_head_matmul_readvariableop_resource:	?A
3classification_head_biasadd_readvariableop_resource:
identity??*classification_head/BiasAdd/ReadVariableOp?)classification_head/MatMul/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:*?b
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*'
_output_shapes
:*??
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*&
_output_shapes
:p*
ksize
*
paddingVALID*
strides
?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:m *
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:m a
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:m ?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*&
_output_shapes
:
6 *
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 *
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:	3 a
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:	3 ?
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: a
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
: ?
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu:activations:0*&
_output_shapes
: *
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_1/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_1/Const:output:0*
T0*
_output_shapes
:	?\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_1/dropout/MulMulflatten_1/Reshape:output:0 dropout_1/dropout/Const:output:0*
T0*
_output_shapes
:	?h
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?{
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*
_output_shapes
:	??
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?X
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	?\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_2/dropout/MulMuldense_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*
_output_shapes
:	?h
dropout_2/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   ?  ?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*
_output_shapes
:	?*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	?{
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*
_output_shapes
:	??
)classification_head/MatMul/ReadVariableOpReadVariableOp2classification_head_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
classification_head/MatMulMatMuldropout_2/dropout/Mul_1:z:01classification_head/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
*classification_head/BiasAdd/ReadVariableOpReadVariableOp3classification_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
classification_head/BiasAddBiasAdd$classification_head/MatMul:product:02classification_head/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:u
classification_head/SoftmaxSoftmax$classification_head/BiasAdd:output:0*
T0*
_output_shapes

:k
IdentityIdentity%classification_head/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp+^classification_head/BiasAdd/ReadVariableOp*^classification_head/MatMul/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:+?: : : : : : : : : : : : 2X
*classification_head/BiasAdd/ReadVariableOp*classification_head/BiasAdd/ReadVariableOp2V
)classification_head/MatMul/ReadVariableOp)classification_head/MatMul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:+?
 
_user_specified_nameinputs
?
?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41801
audio_preproc_input
audio_preproc_41766"
audio_preproc_41768:	?
audio_preproc_41770
audio_preproc_41772,
sequential_1_41775: 
sequential_1_41777:,
sequential_1_41779:  
sequential_1_41781: ,
sequential_1_41783:   
sequential_1_41785: ,
sequential_1_41787:   
sequential_1_41789: &
sequential_1_41791:
??!
sequential_1_41793:	?%
sequential_1_41795:	? 
sequential_1_41797:
identity??%audio_preproc/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?
%audio_preproc/StatefulPartitionedCallStatefulPartitionedCallaudio_preproc_inputaudio_preproc_41766audio_preproc_41768audio_preproc_41770audio_preproc_41772*
Tin	
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40736?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall.audio_preproc/StatefulPartitionedCall:output:0sequential_1_41775sequential_1_41777sequential_1_41779sequential_1_41781sequential_1_41783sequential_1_41785sequential_1_41787sequential_1_41789sequential_1_41791sequential_1_41793sequential_1_41795sequential_1_41797*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_41549s
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp&^audio_preproc/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:???????????: : : : : : : : : : : : : : : : 2N
%audio_preproc/StatefulPartitionedCall%audio_preproc/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:^ Z
)
_output_shapes
:???????????
-
_user_specified_nameaudio_preproc_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_Assert_AssertGuard_false_299_47*
&assert_assertguard_assert_greaterequal
)
%assert_assertguard_assert_spectrogram!
assert_assertguard_identity_1
??Assert/AssertGuard/Assert?
Assert/AssertGuard/AssertAssert&assert_assertguard_assert_greaterequal%assert_assertguard_assert_spectrogram*

T
2*
_output_shapes
 2
Assert/AssertGuard/Assert?
Assert/AssertGuard/IdentityIdentity&assert_assertguard_assert_greaterequal^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity?
Assert/AssertGuard/Identity_1Identity$Assert/AssertGuard/Identity:output:0^Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2
Assert/AssertGuard/Identity_1"G
assert_assertguard_identity_1&Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+: :'???????????????????????????26
Assert/AssertGuard/AssertAssert/AssertGuard/Assert: 

_output_shapes
: :C?
=
_output_shapes+
):'???????????????????????????
?
?
0__inference_cond_Assert_AssertGuard_true_249_110/
+cond_assert_assertguard_identity_cond_equal
'
#cond_assert_assertguard_placeholder&
"cond_assert_assertguard_identity_1
Z
cond/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2
cond/Assert/AssertGuard/NoOp?
 cond/Assert/AssertGuard/IdentityIdentity+cond_assert_assertguard_identity_cond_equal^cond/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2"
 cond/Assert/AssertGuard/Identity?
"cond/Assert/AssertGuard/Identity_1Identity)cond/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2$
"cond/Assert/AssertGuard/Identity_1"Q
"cond_assert_assertguard_identity_1+cond/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+: :'???????????????????????????: 

_output_shapes
: :C?
=
_output_shapes+
):'???????????????????????????"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
audio_preproc_input>
%serving_default_audio_preproc_input:0???????????7
sequential_1'
StatefulPartitionedCall:0tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

signatures
#_self_saveable_object_factories"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
 layer_with_weights-5
 layer-12
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'	optimizer"
_tf_keras_sequential
~
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
:trace_0
;trace_1
<trace_2
=trace_32?
,__inference_sequential_3_layer_call_fn_41450
,__inference_sequential_3_layer_call_fn_41883
,__inference_sequential_3_layer_call_fn_41920
,__inference_sequential_3_layer_call_fn_41725?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z:trace_0z;trace_1z<trace_2z=trace_3
?
>trace_0
?trace_1
@trace_2
Atrace_32?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41983
G__inference_sequential_3_layer_call_and_return_conditional_losses_42060
G__inference_sequential_3_layer_call_and_return_conditional_losses_41763
G__inference_sequential_3_layer_call_and_return_conditional_losses_41801?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z>trace_0z?trace_1z@trace_2zAtrace_3
?B?
 __inference__wrapped_model_40680audio_preproc_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Bserving_default"
signature_map
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

(window
(_window
#I_self_saveable_object_factories"
_tf_keras_layer
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32?
-__inference_audio_preproc_layer_call_fn_40708
-__inference_audio_preproc_layer_call_fn_42073
-__inference_audio_preproc_layer_call_fn_42086
-__inference_audio_preproc_layer_call_fn_40760?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
?
Strace_0
Ttrace_1
Utrace_2
Vtrace_32?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_42099
H__inference_audio_preproc_layer_call_and_return_conditional_losses_42112
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40773
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40786?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zStrace_0zTtrace_1zUtrace_2zVtrace_3
,
Wserving_default"
signature_map
 "
trackable_dict_wrapper
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

)kernel
*bias
 ^_jit_compiled_convolution_op"
_tf_keras_layer
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
?
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

+kernel
,bias
 k_jit_compiled_convolution_op"
_tf_keras_layer
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
?
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

-kernel
.bias
 x_jit_compiled_convolution_op"
_tf_keras_layer
?
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

/kernel
0bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

1kernel
2bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
v
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_3
?trace_4
?trace_52?
,__inference_sequential_1_layer_call_fn_40997
,__inference_sequential_1_layer_call_fn_42141
,__inference_sequential_1_layer_call_fn_42170
,__inference_sequential_1_layer_call_fn_41237
,__inference_sequential_1_layer_call_fn_42199
,__inference_sequential_1_layer_call_fn_42228?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3z?trace_4z?trace_5
?
?trace_0
?trace_1
?trace_2
?trace_3
?trace_4
?trace_52?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42282
G__inference_sequential_1_layer_call_and_return_conditional_losses_42350
G__inference_sequential_1_layer_call_and_return_conditional_losses_41278
G__inference_sequential_1_layer_call_and_return_conditional_losses_41319
G__inference_sequential_1_layer_call_and_return_conditional_losses_42404
G__inference_sequential_1_layer_call_and_return_conditional_losses_42472?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3z?trace_4z?trace_5
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate3m?4m?3v?4v?"
	optimizer
-:+?2 audio_preprocessing_layer/window
):'2conv2d_1/kernel
:2conv2d_1/bias
):' 2conv2d_2/kernel
: 2conv2d_2/bias
):'  2conv2d_3/kernel
: 2conv2d_3/bias
):'  2conv2d_4/kernel
: 2conv2d_4/bias
": 
??2dense_1/kernel
:?2dense_1/bias
-:+	?2classification_head/kernel
&:$2classification_head/bias
n
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_sequential_3_layer_call_fn_41450audio_preproc_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_3_layer_call_fn_41883inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_3_layer_call_fn_41920inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_3_layer_call_fn_41725audio_preproc_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41983inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_3_layer_call_and_return_conditional_losses_42060inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41763audio_preproc_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41801audio_preproc_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_41840audio_preproc_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
7__inference_audio_preprocessing_layer_layer_call_fn_496?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
R__inference_audio_preprocessing_layer_layer_call_and_return_conditional_losses_529?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
'
(0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
-__inference_audio_preproc_layer_call_fn_40708audio_preprocessing_layer_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_audio_preproc_layer_call_fn_42073inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_audio_preproc_layer_call_fn_42086inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_audio_preproc_layer_call_fn_40760audio_preprocessing_layer_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_42099inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_42112inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40773audio_preprocessing_layer_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40786audio_preprocessing_layer_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
HBF
!__inference_signature_wrapper_391audio_preprocessing_layer_input
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_1_layer_call_fn_42481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_42492?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_1_layer_call_fn_42497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_42502?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_2_layer_call_fn_42511?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_42522?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_2_layer_call_fn_42527?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_42532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_3_layer_call_fn_42541?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_42552?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_3_layer_call_fn_42557?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_42562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_conv2d_4_layer_call_fn_42571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_42582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_4_layer_call_fn_42587?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_42592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_flatten_1_layer_call_fn_42597?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_flatten_1_layer_call_and_return_conditional_losses_42603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_1_layer_call_fn_42608
)__inference_dropout_1_layer_call_fn_42613?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_1_layer_call_and_return_conditional_losses_42618
D__inference_dropout_1_layer_call_and_return_conditional_losses_42630?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_1_layer_call_fn_42639?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_1_layer_call_and_return_conditional_losses_42650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_2_layer_call_fn_42655
)__inference_dropout_2_layer_call_fn_42660?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_2_layer_call_and_return_conditional_losses_42665
D__inference_dropout_2_layer_call_and_return_conditional_losses_42677?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_classification_head_layer_call_fn_42686?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_classification_head_layer_call_and_return_conditional_losses_42697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
f
)0
*1
+2
,3
-4
.5
/6
07
18
29"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
8
9
10
11
 12"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_sequential_1_layer_call_fn_40997conv2d_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_1_layer_call_fn_42141inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_1_layer_call_fn_42170inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_1_layer_call_fn_41237conv2d_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_1_layer_call_fn_42199inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_sequential_1_layer_call_fn_42228inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42282inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42350inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_41278conv2d_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_41319conv2d_1_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42404inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42472inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
7__inference_audio_preprocessing_layer_layer_call_fn_496"?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
R__inference_audio_preprocessing_layer_layer_call_and_return_conditional_losses_529"?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_1_layer_call_fn_42481inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_42492inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
/__inference_max_pooling2d_1_layer_call_fn_42497inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_42502inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_2_layer_call_fn_42511inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_42522inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
/__inference_max_pooling2d_2_layer_call_fn_42527inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_42532inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_3_layer_call_fn_42541inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_42552inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
/__inference_max_pooling2d_3_layer_call_fn_42557inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_42562inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_conv2d_4_layer_call_fn_42571inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_42582inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
/__inference_max_pooling2d_4_layer_call_fn_42587inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_42592inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_flatten_1_layer_call_fn_42597inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_flatten_1_layer_call_and_return_conditional_losses_42603inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dropout_1_layer_call_fn_42608inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_1_layer_call_fn_42613inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_42618inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_42630inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_1_layer_call_fn_42639inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_1_layer_call_and_return_conditional_losses_42650inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dropout_2_layer_call_fn_42655inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_2_layer_call_fn_42660inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_2_layer_call_and_return_conditional_losses_42665inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_2_layer_call_and_return_conditional_losses_42677inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
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
?B?
3__inference_classification_head_layer_call_fn_42686inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_classification_head_layer_call_and_return_conditional_losses_42697inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
2:0	?2!Adam/classification_head/kernel/m
+:)2Adam/classification_head/bias/m
2:0	?2!Adam/classification_head/kernel/v
+:)2Adam/classification_head/bias/v
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant?
 __inference__wrapped_model_40680??(??)*+,-./01234>?;
4?1
/?,
audio_preproc_input???????????
? "2?/
-
sequential_1?
sequential_1?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40773??(??R?O
H?E
;?8
audio_preprocessing_layer_input???????????
p 

 
? "7?4
-?*
0#???????????????????
? ?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_40786??(??R?O
H?E
;?8
audio_preprocessing_layer_input???????????
p

 
? "7?4
-?*
0#???????????????????
? ?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_42099}?(??9?6
/?,
"?
inputs???????????
p 

 
? "7?4
-?*
0#???????????????????
? ?
H__inference_audio_preproc_layer_call_and_return_conditional_losses_42112}?(??9?6
/?,
"?
inputs???????????
p

 
? "7?4
-?*
0#???????????????????
? ?
-__inference_audio_preproc_layer_call_fn_40708??(??R?O
H?E
;?8
audio_preprocessing_layer_input???????????
p 

 
? "*?'#????????????????????
-__inference_audio_preproc_layer_call_fn_40760??(??R?O
H?E
;?8
audio_preprocessing_layer_input???????????
p

 
? "*?'#????????????????????
-__inference_audio_preproc_layer_call_fn_42073p?(??9?6
/?,
"?
inputs???????????
p 

 
? "*?'#????????????????????
-__inference_audio_preproc_layer_call_fn_42086p?(??9?6
/?,
"?
inputs???????????
p

 
? "*?'#????????????????????
R__inference_audio_preprocessing_layer_layer_call_and_return_conditional_losses_529}?(??1?.
'?$
"?
inputs???????????
? "??<
5?2
0+???????????????????????????
? ?
7__inference_audio_preprocessing_layer_layer_call_fn_496p?(??1?.
'?$
"?
inputs???????????
? "2?/+????????????????????????????
N__inference_classification_head_layer_call_and_return_conditional_losses_42697K34'?$
?
?
inputs	?
? "?
?
0
? u
3__inference_classification_head_layer_call_fn_42686>34'?$
?
?
inputs	?
? "??
C__inference_conv2d_1_layer_call_and_return_conditional_losses_42492\)*/?,
%?"
 ?
inputs+?
? "%?"
?
0*?
? {
(__inference_conv2d_1_layer_call_fn_42481O)*/?,
%?"
 ?
inputs+?
? "?*??
C__inference_conv2d_2_layer_call_and_return_conditional_losses_42522Z+,.?+
$?!
?
inputsp
? "$?!
?
0m 
? y
(__inference_conv2d_2_layer_call_fn_42511M+,.?+
$?!
?
inputsp
? "?m ?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_42552Z-..?+
$?!
?
inputs
6 
? "$?!
?
0	3 
? y
(__inference_conv2d_3_layer_call_fn_42541M-..?+
$?!
?
inputs
6 
? "?	3 ?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_42582Z/0.?+
$?!
?
inputs 
? "$?!
?
0 
? y
(__inference_conv2d_4_layer_call_fn_42571M/0.?+
$?!
?
inputs 
? "? ?
B__inference_dense_1_layer_call_and_return_conditional_losses_42650L12'?$
?
?
inputs	?
? "?
?
0	?
? j
'__inference_dense_1_layer_call_fn_42639?12'?$
?
?
inputs	?
? "?	??
D__inference_dropout_1_layer_call_and_return_conditional_losses_42618L+?(
!?
?
inputs	?
p 
? "?
?
0	?
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_42630L+?(
!?
?
inputs	?
p
? "?
?
0	?
? l
)__inference_dropout_1_layer_call_fn_42608?+?(
!?
?
inputs	?
p 
? "?	?l
)__inference_dropout_1_layer_call_fn_42613?+?(
!?
?
inputs	?
p
? "?	??
D__inference_dropout_2_layer_call_and_return_conditional_losses_42665L+?(
!?
?
inputs	?
p 
? "?
?
0	?
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_42677L+?(
!?
?
inputs	?
p
? "?
?
0	?
? l
)__inference_dropout_2_layer_call_fn_42655?+?(
!?
?
inputs	?
p 
? "?	?l
)__inference_dropout_2_layer_call_fn_42660?+?(
!?
?
inputs	?
p
? "?	??
D__inference_flatten_1_layer_call_and_return_conditional_losses_42603O.?+
$?!
?
inputs 
? "?
?
0	?
? o
)__inference_flatten_1_layer_call_fn_42597B.?+
$?!
?
inputs 
? "?	??
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_42502?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_42497?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_42532?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_2_layer_call_fn_42527?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_42562?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_3_layer_call_fn_42557?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_42592?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_4_layer_call_fn_42587?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_41278m)*+,-./01234??<
5?2
(?%
conv2d_1_input+?
p 

 
? "?
?
0
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_41319m)*+,-./01234??<
5?2
(?%
conv2d_1_input+?
p

 
? "?
?
0
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42282e)*+,-./012347?4
-?*
 ?
inputs+?
p 

 
? "?
?
0
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42350e)*+,-./012347?4
-?*
 ?
inputs+?
p

 
? "?
?
0
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42404w)*+,-./01234I?F
??<
2?/
inputs#???????????????????
p 

 
? "?
?
0
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_42472w)*+,-./01234I?F
??<
2?/
inputs#???????????????????
p

 
? "?
?
0
? ?
,__inference_sequential_1_layer_call_fn_40997`)*+,-./01234??<
5?2
(?%
conv2d_1_input+?
p 

 
? "??
,__inference_sequential_1_layer_call_fn_41237`)*+,-./01234??<
5?2
(?%
conv2d_1_input+?
p

 
? "??
,__inference_sequential_1_layer_call_fn_42141X)*+,-./012347?4
-?*
 ?
inputs+?
p 

 
? "??
,__inference_sequential_1_layer_call_fn_42170X)*+,-./012347?4
-?*
 ?
inputs+?
p

 
? "??
,__inference_sequential_1_layer_call_fn_42199j)*+,-./01234I?F
??<
2?/
inputs#???????????????????
p 

 
? "??
,__inference_sequential_1_layer_call_fn_42228j)*+,-./01234I?F
??<
2?/
inputs#???????????????????
p

 
? "??
G__inference_sequential_3_layer_call_and_return_conditional_losses_41763{?(??)*+,-./01234F?C
<?9
/?,
audio_preproc_input???????????
p 

 
? "?
?
0
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41801{?(??)*+,-./01234F?C
<?9
/?,
audio_preproc_input???????????
p

 
? "?
?
0
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_41983n?(??)*+,-./012349?6
/?,
"?
inputs???????????
p 

 
? "?
?
0
? ?
G__inference_sequential_3_layer_call_and_return_conditional_losses_42060n?(??)*+,-./012349?6
/?,
"?
inputs???????????
p

 
? "?
?
0
? ?
,__inference_sequential_3_layer_call_fn_41450n?(??)*+,-./01234F?C
<?9
/?,
audio_preproc_input???????????
p 

 
? "??
,__inference_sequential_3_layer_call_fn_41725n?(??)*+,-./01234F?C
<?9
/?,
audio_preproc_input???????????
p

 
? "??
,__inference_sequential_3_layer_call_fn_41883a?(??)*+,-./012349?6
/?,
"?
inputs???????????
p 

 
? "??
,__inference_sequential_3_layer_call_fn_41920a?(??)*+,-./012349?6
/?,
"?
inputs???????????
p

 
? "??
!__inference_signature_wrapper_391??(??m?j
? 
c?`
^
audio_preprocessing_layer_input;?8
audio_preprocessing_layer_input???????????"o?l
j
audio_preprocessing_layerM?J
audio_preprocessing_layer+????????????????????????????
#__inference_signature_wrapper_41840??(??)*+,-./01234U?R
? 
K?H
F
audio_preproc_input/?,
audio_preproc_input???????????"2?/
-
sequential_1?
sequential_1