# Simple Torch Tensor Notes

## Torch Settings

### torch.get_default_dtype() -> torch.dtype
获取创建张量时的默认数据类型。

### torch.set_default_dtype(d) -> None
设置创建浮点张量时的默认数据类型。
#### Args
- d: torch.dtype – the floating point dtype to make the default. Either torch.float32 or torch.float64.

### torch.set_default_device(device)
设置创建浮点张量时的默认设备。
默认为cpu。
#### Args
- device (device or string) – the device to set as default

### torch.set_printoptions(precision, threshold, edgeitems, linewidth, profile, sci_mode)
设置张量被打印时的显示方式
#### Args
- precision – 浮点数小数点后保留位数 (default = 4).
- threshold – 数组最大元素数阈值，超过后张量将会进行缩略，不会被全部打印 (default = 1000).
- edgeitems – 缩略情况下每个维度最开始打印元素数 (default = 3).
- linewidth – 每行显示字符数 (default = 80). Thresholded matrices will ignore this parameter.
- profile – Sane defaults for pretty printing. Can override with any of the above options. (any one of default, short, full)
- sci_mode – 是否使用科学记数法。若为None则由torch._tensor_str._Formatter决定。This value is automatically chosen by the framework.(default = None)

## Tensor Creation Operations

### torch.tensor(data, dtype=None, device=None, requires_grad=False) -> Tensor
创建张量。  
创建方式为拷贝，底层数据不共用。
> When working with tensors prefer using torch.Tensor.clone(), torch.Tensor.detach(), and torch.Tensor.requires_grad_() for readability. 
> 
> Letting t be a tensor, torch.tensor(t) is equivalent to t.clone().detach(), 
> and torch.tensor(t, requires_grad=True) is equivalent to t.clone().detach().requires_grad_(True).
#### Args
- data (array_like) – Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
#### Examples
```python
>>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor([[ 0.1000,  1.2000],
        [ 2.2000,  3.1000],
        [ 4.9000,  5.2000]])

>>> torch.tensor([0, 1])  # Type inference on data
tensor([ 0,  1])

>>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
...              dtype=torch.float64,
...              device=torch.device('cuda:0'))  # creates a double tensor on a CUDA device
tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

>>> torch.tensor(3.14159)  # Create a zero-dimensional (scalar) tensor
tensor(3.1416)

>>> torch.tensor([])  # Create an empty tensor (of size (0,))
tensor([])
```

### torch.from_numpy(ndarray) -> Tensor
从numpy数组创建张量。  
底层数据共用，修改numpy数组会同时改变张量。
#### Args
- ndarray: numpy.ndarray
#### Examples
```python
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([-1,  2,  3])
```

### torch.as_tensor(data, dtype=None, device=None) -> Tensor
从data创建张量。 
原data为tensor的情形，若目标dtype和device与原dtype和device均一致，则直接返回原张量；若目标dtype与原dtype不一致，或device不一致，则拷贝数据创建新张量，底层数据不共享。此时可视为使用data.to(dtype=dtype, device=device)  
原data为ndarray的情形，创建时会调用torch.from_numpy(ndarray)。底层数据共用，修改data会同时改变张量。
#### Args
- data (array_like) – Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
#### Examples
```python
>>> a = numpy.array([1, 2, 3])
>>> t = torch.as_tensor(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([-1,  2,  3])

>>> a = numpy.array([1, 2, 3])
>>> t = torch.as_tensor(a, device=torch.device('cuda'))
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([1,  2,  3])
```

## Tensor Generation Operations

### torch.zeros(*size, dtype=None, device=None, requires_grad=False) -> Tensor
生成指定形状的全0张量。  
`size`即可传入若干整数，也可传入列表之类的元素为整数序列
#### Args
- size: Ints or List[Int] – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
#### Examples
```python
>>> torch.zeros(2, 3)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])

>>> torch.zeros([2, 3])
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
```

### torch.ones(*size, dtype=None, device=None, requires_grad=False) -> Tensor
生成指定形状的全1张量。  
`size`即可传入若干整数，也可传入列表之类的元素为整数序列
#### Args
- size: Ints or List[Int] – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
#### Examples
```python
>>> torch.ones(2, 3)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])

>>> torch.ones([2, 3])
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
```

### torch.zeros_like(input, dtype=None, device=None, requires_grad=False) -> Tensor
生成与传入张量相同形状的全0张量。
#### Args
- input: Tensor – the size of input will determine size of the output tensor.
#### Examples
```python
>>> input = torch.empty(2, 3)
>>> torch.zeros_like(input)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
```

### torch.ones_like(input, dtype=None, device=None, requires_grad=False) -> Tensor
生成与传入张量相同形状的全1张量。
#### Args
- input: Tensor – the size of input will determine size of the output tensor.
#### Examples
```python
>>> input = torch.empty(2, 3)
>>> torch.ones_like(input)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
```

### torch.full(size, fill_value, *, dtype=None, device=None, requires_grad=False) -> Tensor
生成指定形状的张量，元素的值均为`fill_value`。
#### Args
- size: list, tuple, or torch.Size of integers - 定义生成张量的形状
- fill_value: Scalar – 用于填充的张量
#### Example
```python
>>> torch.full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]])
```

### torch.full_like(input, fill_value, *, dtype=None, device=None, requires_grad=False) -> Tensor
生成与传入张量相同形状的张量，元素的值均为`fill_value`。
#### Args
- input: Tensor – 传入张量，其形状将决定生成张量的形状
- fill_value: Scalar – 用于填充的张量
#### Example
```python
>>> a = torch.randn(2, 3)
>>> torch.full_like(a, 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]])
```

### torch.arange(start=0, end, step=1, dtype=None, device=None, requires_grad=False) -> Tensor
生成从`step`到`end`的以`step`为步长的1维张量，不包括`end`。  
其中共有(`end` - `start`) \ `step`个元素。
#### Args
dtype (torch.dtype, optional) – 如果`step`，`end`或`step`参数中有任何一个为浮点数，则生成张量dtype为默认浮点数类型；否则默认为int64

> the desired data type of returned tensor. 
> Default: if None, uses a global default (see torch.set_default_dtype()).
> If dtype is not given, infer the data type from the other input arguments. 
> If any of start, end, or step are floating-point, the dtype is inferred to be the default dtype, see get_default_dtype(). 
> Otherwise, the dtype is inferred to be torch.int64.

#### Examples
```python
>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])
>>> torch.arange(1, 4)
tensor([ 1,  2,  3])
>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```

### torch.linspace(start, end, steps, dtype=None, device=None, requires_grad=False) -> Tensor
生成从`step`到`end`的1维张量，包括`end`。  
其中共有`step`个元素。
#### Example
```python
>>> torch.linspace(3, 10, steps=5)
tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
>>> torch.linspace(-10, 10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
>>> torch.linspace(start=-10, end=10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
>>> torch.linspace(start=-10, end=10, steps=1)
tensor([-10.])
```

### torch.logspace(start, end, steps, base=10.0, *, dtype=None, device=None, requires_grad=False) -> Tensor
生成从`bace ^ step`到`bace ^ end`的1维张量，包括`end`。  
其中共有`step`个元素。
相当于将linspace中每个元素都作为base的指数进行运算。
#### Example
```python
>>> torch.logspace(start=-10, end=10, steps=5)
tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
>>> torch.logspace(start=0.1, end=1.0, steps=5)
tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
>>> torch.logspace(start=0.1, end=1.0, steps=1)
tensor([1.2589])
>>> torch.logspace(start=2, end=2, steps=1, base=2)
tensor([4.0])
```

### torch.eye(n, m=None, *, dtype=None, device=None, requires_grad=False) -> Tensor
生成主对角线全为1，其他全为0的2维张量。  
若指定m与n不相等，则多出的列均补0，缺少的列均截断。  
#### Args
- n: int – 生成张量的行数
- m: int (optional) – 生成张量的列数，默认与n一致
#### Example
```python
>>> torch.eye(3)
tensor([[ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]])
>>> torch.eye(3, 4)
tensor([[ 1.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  1.,  0.]])
>>> torch.eye(3, 2)
tensor([[ 1.,  0.],
        [ 0.,  1.],
        [ 0.,  0.]])
```

### torch.heaviside(input, values, *, out=None) -> Tensor
通过计算`input`中每个元素的 Heaviside 阶跃函数，生成相应张量。  
生成张量与`input`形状相同。
Heaviside 阶跃函数定义如下：

        heaviside(input, value) = 0, if input < 0    
                                  value, if input == 0  
                                  1, if input > 1 
  
#### Args
- input (Tensor) – the input tensor.
- values (Tensor) – The values to use where input is zero.
#### Example
```python
>>> input = torch.tensor([-1.5, 0, 2.0])
>>> values = torch.tensor([0.5])
>>> torch.heaviside(input, values)
tensor([0.0000, 0.5000, 1.0000])
>>> values = torch.tensor([1.2, -2.0, 3.5])
>>> torch.heaviside(input, values)
tensor([0., -2., 1.])
```

## Torch RNG

### torch.Generator(device='cpu')
创建并返回一个生成器对象，该对象管理生成伪随机数的算法的状态。  
在许多就地随机采样函数中用作关键字参数。  
事实上，在不指定generator的情况下，默认使用cpu上的全局随机数生成器`torch.default_generator`。以下的api即是直接对全局RNG进行操作。
#### Attributes
- device - 所在设备
- seed() - 从`std::random_device`或当前时间获取非确定性随机数，并使用它来为生成器提供种子。
- manual_seed(seed) - 设置用于生成随机数的种子。返回一个 torch.Generator 对象。任何32位整数都是有效的种子。
- initial_seed() - 返回用于生成随机数的初始种子。
- get_state() - 获取生成器状态。
- set_state(new_state) - 设置生成器状态。

### torch.seed() -> int
设置一个随机的用于生成随机数的种子。
返回其设置种子的python int整数。
从`std::random_device`或当前时间获取非确定性随机数，并使用它来为默认全局生成器提供种子。
#### Args
- seed: int – The desired seed. 
> Value must be within the inclusive range [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]. Otherwise, a RuntimeError is raised.   
> Negative inputs are remapped to positive values with the formula 0xffff_ffff_ffff_ffff + seed.

### torch.manual_seed(seed) -> torch.Generator
手动指定一个用于生成随机数的种子。  
注意，每次进行含有随机生成的操作时，若想指定该seed，需要重新显式设置。  
当使用cuda时，还需考虑`torch.cuda.manual_seed(random_seed)`和`torch.backends.cudnn.deterministic = True`
#### Args
- seed: int – The desired seed. 
> Value must be within the inclusive range [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]. Otherwise, a RuntimeError is raised.   
> Negative inputs are remapped to positive values with the formula 0xffff_ffff_ffff_ffff + seed.

### torch.initial_seed() -> int
获取默认全局随机数生成器的初始种子。

### torch.get_rng_state() -> torch.ByteTensor
获取torch的随机数生成器(RNG)状态。

### torch.set_rng_state(new_state) -> None
设置torch的随机数生成器(RNG)状态。
#### Args
- new_state: torch.ByteTensor – The desired state

## Tensor Random Sampling Generation

### torch.bernoulli(input, *, generator=None) -> Tensor
以`input`为“值为1”的概率进行伯努利分布的采样。  
out = Bernoulli(p=input)  
生成张量的形状与`input`相同。
#### Args
- input: Tensor – the input tensor of probability values for the Bernoulli distribution
- generator: torch.Generator (optional) – a pseudorandom number generator for sampling
#### Example
```python
>>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
>>> a
tensor([[ 0.1737,  0.0950,  0.3609],
        [ 0.7148,  0.0289,  0.2676],
        [ 0.9456,  0.8937,  0.7202]])
>>> torch.bernoulli(a)
tensor([[ 1.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.]])
>>> a = torch.ones(3, 3) # probability of drawing "1" is 1
>>> torch.bernoulli(a)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
>>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
>>> torch.bernoulli(a)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
```
### torch.rand(*size, generator=None, dtype=None, device=None, requires_grad=False) -> Tensor
生成一个张量，该张量填充随机数服从区间[0,1)上均匀分布。  
张量的形状由参数`size`定义。
#### Args
- size: Sequence of Int or Turple like or torch.Size – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
- generator: torch.Generator (optional) – a pseudorandom number generator for sampling
#### Example
```python
>>> torch.rand(4)
tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
>>> torch.rand(2, 3)
tensor([[ 0.8237,  0.5781,  0.6879],
        [ 0.3816,  0.7249,  0.0998]])
```

### torch.rand_like(input, dtype=None, device=None, requires_grad=False) -> Tensor
生成一个张量，该张量填充随机数服从区间[0,1)上均匀分布。  
张量的形状由与`input`传入的张量一致。  
`torch.rand_like(input)`与`torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`等价。  
但注意无法指定`generator`。
#### Args
- input: Tensor – the input tensor of probability values for the Bernoulli distribution

### torch.randint(low=0, high, size, \*, generator=None, dtype=None, device=None, requires_grad=False) -> Tensor
生成一个张量，该张量填充有在区间[`low`, `high`)均匀生成的随机整数。  
张量的形状由参数`size`定义。
#### Args
- low: int (optional) – Lowest integer to be drawn from the distribution. Default: 0.
- high: int – One above the highest integer to be drawn from the distribution.
- size: tuple – a tuple defining the shape of the output tensor.
#### Example
```python
>>> torch.randint(3, 5, (3,))
tensor([4, 3, 4])
>>> torch.randint(10, (2, 2))
tensor([[0, 2],
        [5, 5]])
>>> torch.randint(3, 10, (2, 2))
tensor([[4, 5],
        [6, 7]])
```

### torch.randint_like(input, low=0, high, dtype=None, device=None, requires_grad=False) -> Tensor
生成一个张量，该张量填充有在区间[`low`, `high`)均匀生成的随机整数。    
张量的形状由与`input`传入的张量一致。    
#### Args
- input: Tensor – the size of input will determine size of the output tensor.  
- low: int (optional) – Lowest integer to be drawn from the distribution. Default: 0.  
- high: int – One above the highest integer to be drawn from the distribution.  

### torch.randn(*size, generator=None, dtype=None, device=None, requires_grad=False) -> Tensor
生成一个张量，该张量由均值为0、方差为1的正态分布中的随机数填充。  
张量的形状由参数`size`定义。  
#### Args
- size (int...) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.  
- generator (torch.Generator, optional) – a pseudorandom number generator for sampling  
#### Examples
```python
>>> torch.randn(4)
tensor([-2.1436,  0.9966,  2.3426, -0.6366])
>>> torch.randn(2, 3)
tensor([[ 1.5954,  2.8929, -1.0923],
        [ 1.1719, -0.4709, -0.1996]])
```

### torch.randn_like(input, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor
生成一个张量，该张量由均值为0、方差为1的正态分布中的随机数填充。  
张量的形状由与`input`传入的张量一致。  
#### Args
- input: Tensor – the size of input will determine size of the output tensor.

### torch.randperm(n,  generator=None, out=None, dtype=torch.int64, device=None, requires_grad=False) -> Tensor
生成一个张量，该张量的元素为从0到n-1的所有整数的随机排列。  
#### Args
- n: int – the upper bound (exclusive)  
- generator: torch.Generator (optional) – a pseudorandom number generator for sampling  
#### Examples
```python
>>> torch.randperm(4)
tensor([2, 1, 0, 3])
```

## Tensor Type Operations

|          Data Type           |             dtype             | CPU tensor |        GPU tensor        |
|:----------------------------:|:-----------------------------:| :----: |:------------------------:|
|    32-bit floating point     | torch.float32 or torch.float  | torch.FloatTensor |  torch.cuda.FloatTensor  |
|    64-bit floating point     | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor  | 
| 16-bit floating point (FP16) | torch.float16 or torch.half | torch.HalfTensor | torch.cuda.HalfTensor |
| 16-bit floating point (BF16) | torch.bfloat16 | torch.BFloat16Tensor | torch.cuda.BFloat16Tensor |
| 8-bit integer (unsigned) | torch.uint8 | torch.ByteTensor | torch.cuda.ByteTensor |
| 8-bit integer (signed) | torch.int8 | torch.CharTensor | torch.cuda.CharTensor |
| 16-bit integer (signed) | torch.int16 or torch.short | torch.ShortTensor | torch.cuda.ShortTensor |
| 32-bit integer (signed) | torch.int32 or torch.int | torch.IntTensor | torch.cuda.IntTensor |
| 64-bit integer (signed) | torch.int64 or torch.long | torch.LongTensor | torch.cuda.LongTensor |
| Boolean | torch.bool | torch.BoolTensor | torch.cuda.BoolTensor |


>FP16（半精度）：在 FP16 中，浮点数用 16 位表示。它由 1 个符号位、5 位指数和 10 位分数（尾数）组成。这种有限的精度允许表示各种数字，但对于非常小或非常大的值，它会牺牲精度。
>
>BF16 （BFloat16）：BF16 也使用 16 位，但发行版不同。它有 1 个符号位，指数有 8 位，尾数有 7 位。此格式旨在为小值保留更高的精度，同时仍能容纳各种数字。


### tensor.type() -> str
获取张量的数值类型。  
返回的是字符串。
#### Args
- None
#### Examples
```python
>>> a
tensor([[1., 2., 3.],
        [1., 1., 1.]])
>>> a.type()
'torch.FloatTensor'
```

### tensor.dtype
获取张量的数值精度。  
返回的是torch数值类型。
#### Args
- None
#### Examples
```python
>>> a
tensor([[1., 2., 3.],
        [1., 1., 1.]])
>>> a.dtype
torch.float32
```

### tensor.type(dtype) -> torch.Tensor
进行张量数值类型的转换。  
一般返回一个新的张量，底层数据不共享；当目标类型与原类型已经一致时直接返回原张量。
#### Args
- dtype (dtype or string): The desired type
- non_blocking (bool): If ``True``, and the source is in pinned memory
        and destination is on the GPU or vice versa, the copy is performed
        asynchronously with respect to the host. Otherwise, the argument
        has no effect.
#### Examples
```python
>>> a
tensor([[1., 2., 3.],
        [1., 1., 1.]])
>>> a.dtype
torch.float32
>>> a.type(torch.int32).dtype
torch.int32
```

### tensor.to(dtype) -> torch.Tensor
进行张量数值类型的转换。  
一般返回一个新的张量，底层数据不共享；当目标类型与原类型已经一致时直接返回原张量。
#### Args
- dtype (dtype or string): The desired type
- non_blocking (bool): If ``True``, and the source is in pinned memory
        and destination is on the GPU or vice versa, the copy is performed
        asynchronously with respect to the host. Otherwise, the argument
        has no effect.
- memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.
#### Examples
```python
>>> a
tensor([[1., 2., 3.],
        [1., 1., 1.]])
>>> a.dtype
torch.float32
>>> a.to(torch.int32).dtype
torch.int32
```

### tensor.view(dtype) -> Tensor
改变数据类型解释。  
对于原先的底层二进制数采用不同dtype进行解释。  
原dtype与目标dtype所需元素不一致时(如int32只需要1个数，而虚数cfloat需要两个)，对tensor的最后一个维度进行缩放。  
返回一个新的张量。
#### Args
- dtype: torch.dtype
#### Examples
```python
>>> x = torch.randn(4, 4)
>>> x
tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
        [-0.1520,  0.7472,  0.5617, -0.8649],
        [-2.4724, -0.0334, -0.2976, -0.8499],
        [-0.2109,  1.9913, -0.9607, -0.6123]])
>>> x.dtype
torch.float32
>>> y = x.view(torch.int32)
>>> y
tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
        [-1105482831,  1061112040,  1057999968, -1084397505],
        [-1071760287, -1123489973, -1097310419, -1084649136],
        [-1101533110,  1073668768, -1082790149, -1088634448]],
    dtype=torch.int32)
>>> y[0, 0] = 1000000000
>>> x
tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
        [-0.1520,  0.7472,  0.5617, -0.8649],
        [-2.4724, -0.0334, -0.2976, -0.8499],
        [-0.2109,  1.9913, -0.9607, -0.6123]])
>>> x.view(torch.cfloat)
tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
        [-0.1520+0.7472j,  0.5617-0.8649j],
        [-2.4724-0.0334j, -0.2976-0.8499j],
        [-0.2109+1.9913j, -0.9607-0.6123j]])
>>> x.view(torch.cfloat).size()
torch.Size([4, 2])
>>> x.view(torch.uint8)
tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
           8, 191],
        [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
          93, 191],
        [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
          89, 191],
        [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
          28, 191]], dtype=torch.uint8)
>>> x.view(torch.uint8).size()
torch.Size([4, 16])
```

## Tensor Dimension View Operations

### Tensor.shape -> torch.Size
获取张量的形状

### Tensor.size -> torch.Size
获取张量的形状

### torch.reshape(input, shape) -> Tensor
改变视图。  
在不改变张量中数据顺序的情况下调整张量的形状。dim和shape均可变，但shape各元素乘积不变。  
可以指定某一维度长度stride为-1以自适应，且只允许存在一个这样的inferred dimension。  
适用于在内存上连续存储和非连续存储的张量。  
在可能的情况下，返回的张量与原张量使用相同的底层存储(仅是视图)；否则会返回一份拷贝。
#### Args
- input: torch.Tensor
- shape: Tuple[Int] or List[Int] or torch.Tensor[Int] or torch.Size[Int]
#### Examples
```python
a = torch.tensor([[1,2,3],[1,2,3]])
a.shape #torch.Size([2,3])
a_r = a.reshape((6)) #-> torch.tensor([1,2,3,1,2,3])
a_r.shape #torch.Size([6])
```

### tensor.view(*shape) -> Tensor
改变视图。  
在不改变张量中数据顺序的情况下调整张量的形状。dim和shape均可变，但shape各元素乘积不变。  
可以指定某一维度长度stride为-1以自适应，且只允许存在一个这样的inferred dimension。
仅适用于在内存上连续存储的张量。  
在可能的情况下，返回的张量与原张量使用相同的底层存储(仅是视图)；否则会返回一份拷贝。
#### Args
- shape: Ints or torch.Size
#### Examples
```python
a = torch.tensor([[1,2,3],[1,2,3]])
a.shape #torch.Size([2,3])
a_v1 = a.view(6) #-> torch.tensor([1,2,3,1,2,3])
a_v1.shape #torch.Size([6])
a_v2 = a.view(3,-1) #-> torch.tensor([[1,2],[3,1],[2,3]])
a_v2.shape #torch.Size([3,2])
```

### torch.squeeze(input, dim=None) -> Tensor
取消长度为1的维度。   
不指定dim时取消所有stride=1的维度；指定时若指定对应维度符合stride=1，取消之，否则不改变。  
返回的张量与原张量使用相同的底层存储。
#### Args
- input: torch.Tensor
- dim: Int or Tuple[Ints] or List[Ints] or None (if given, the input will be squeezed
only in the specified dimensions.)
#### Examples
```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()
torch.Size([2, 2, 1, 2])
>>> y = torch.squeeze(x, (1, 2, 3))
>>> y.size()
torch.Size([2, 2, 2])
```

### torch.unsqueeze(input, dim) -> Tensor
在指定维度上插入(创建)新的维度。   
dim可以如python索引特性一般使用负数index，但使用此方法时一次只能指定一个维度。  
返回的张量与原张量使用相同的底层存储。
#### Args
- input: torch.Tensor
- dim: Int
#### Examples
```python
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```

### torch.transpose(input, dim0, dim1) -> Tensor
交换两个指定维度的次序。  
返回的张量与原张量使用相同的底层存储。如果是稀疏张量，则生成的Tensor不共享底层存储。
#### Args
- input: torch.Tensor
- dim0: Int
- dim1: Int
#### Examples
```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 1.0028, -0.9893,  0.5809],
        [-0.1669,  0.7299,  0.4942]])
>>> torch.transpose(x, 0, 1)
tensor([[ 1.0028, -0.1669],
        [-0.9893,  0.7299],
        [ 0.5809,  0.4942]])
```

### torch.permute(input, dims) -> Tensor
按传入的`dims`参数重排维度顺序  
返回的张量与原张量使用相同的底层存储。
#### Args
- input: torch.Tensor
- dims: Turple or Int
#### Examples
```python
>>> x = torch.randn(2, 3, 5)
>>> x.size()
torch.Size([2, 3, 5])
>>> torch.permute(x, (2, 0, 1)).size()
torch.Size([5, 2, 3])
```

## Tensor Stack and Split Operations

### torch.cat(tensors, dim=0) -> Tensor
在指定的已有维度上将两个张量拼接。  
拼接顺序按照tensors序列的前后顺序。  
除了指定的`dim`维度的stride，其他维度上各待拼接张量之形状必须一致。
#### Args
- tensors: sequence of Tensors
- dim: Int
#### Examples
```python
>>> x = torch.tensor([[ 0.6580, -1.0969, -0.4614],
                      [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```

### torch.stack(tensors, dim=0) -> Tensor
在指定的新维度上将两个张量堆叠。  
堆叠顺序按照tensors序列的前后顺序。  
各待拼接张量之形状必须一致。
#### Args
- tensors: sequence of Tensors
- dim: Int
#### Examples
```python
>>> a = torch.tensor([[1,2,3],[1,1,1]])
>>> b = torch.tensor([[0,0,0],[0,0,0]])
>>> torch.stack([a,b],0)
tensor([[[1, 2, 3],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0]]])
>>> torch.stack([a,b],1)
tensor([[[1, 2, 3],
         [0, 0, 0]],
        [[1, 1, 1],
         [0, 0, 0]]])
>>> torch.stack([a,b],2)
tensor([[[1, 0],
         [2, 0],
         [3, 0]],
        [[1, 0],
         [1, 0],
         [1, 0]]])
```

### torch.split(tensor, split_size_or_sections, dim=0) -> seq
在指定的现有维度上将张量按指定各份长度进行分割。  
如果`split_size_or_sections`输入的是整数，则以此指定每个分割在该维度上的长度。最后一份分割不够的话会短一些。  
如果`split_size_or_sections`输入的是列表，则以此列表各元素指定每一段分割之长度。总分割份数等于列表之长度(元素个数),列表元素之和应等于张量在该维度的长度。
>If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible). Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.
>
>If split_size_or_sections is a list, then tensor will be split into len(split_size_or_sections) chunks with sizes in dim according to split_size_or_sections.
#### Args
- tensor: torch.Tensor
- split_size_or_sections: Int or List[Int] (size of a single chunk or list of sizes for each chunk)
- dim: Int
#### Examples
```python
>>> a = torch.arange(10).reshape(5, 2)
>>> a
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
>>> torch.split(a, 2)
(tensor([[0, 1],
         [2, 3]]),
 tensor([[4, 5],
         [6, 7]]),
 tensor([[8, 9]]))
>>> torch.split(a, [1, 4])
(tensor([[0, 1]]),
 tensor([[2, 3],
         [4, 5],
         [6, 7],
         [8, 9]]))
```

### torch.unbind(input, dim=0) -> seq
在指定的现有维度上将张量以1为步长进行分割。
#### Args
- tensor: torch.Tensor
- dim: Int
#### Examples
```python
>>> torch.unbind(torch.tensor([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]]))
(tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
```


## Tensor Filling Operations

### torch.tile(input, dims) → Tensor
将张量在各其维度上复制指定倍数。  
返回的张量与原张量使用相同的底层存储。如果是稀疏张量，则生成的Tensor不共享底层存储。  
dims为每个维度上张量被复制的倍数。长度与原张量的dim数一致。  
dims长度小于张量dim数的情况下，dims将从第一位开始补1 (prepend)  
张量dim数小于dims长度的情况下，张量维度也将从第一位开始补1(prepend)
>If dims specifies fewer dimensions than input has, then ones are prepended to dims until all dimensions are specified. For example, if input has shape (8, 6, 4, 2) and dims is (2, 2), then dims is treated as (1, 1, 2, 2).
> 
>Analogously, if input has fewer dimensions than dims specifies, then input is treated as if it were unsqueezed at dimension zero until it has as many dimensions as dims specifies. For example, if input has shape (4, 2) and dims is (3, 3, 2, 2), then input is treated as if it had the shape (1, 1, 4, 2).
#### Args
- input: torch.Tensor
- dims: Tuple or List 
#### Examples
```python
>>> x = torch.tensor([1, 2, 3])
>>> x.tile((2,))
tensor([1, 2, 3, 1, 2, 3])
>>> y = torch.tensor([[1, 2], [3, 4]])
>>> torch.tile(y, (2, 2))
tensor([[1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4]])
```

### torch.nn.functional.pad(input, pad, mode='constant', value=None) -> Tensor
将张量在指定维度上前后填充指定个数的指定值。  
填充从最后一维开始,`pad`中每两个元素指定该一维度前后填充的个数。
>For example, to pad only the last dimension of the input tensor, then pad has the form (padding_left,padding_right);   
>to pad the last 2 dimensions of the input tensor, then use (padding_left,padding_right, padding_top , padding_bottom);   
>to pad the last 3 dimensions, use (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back).
> 
#### Args
- input: torch.Tensor
- pad: Tuple
- mode: Str  – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
- value: Float – fill value for 'constant' padding. Default: 0
#### Examples
```python
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p1d = (1, 1) # pad last dim by 1 on each side
>>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
>>> print(out.size())
torch.Size([3, 3, 4, 4])
>>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
>>> out = F.pad(t4d, p2d, "constant", 0)
>>> print(out.size())
torch.Size([3, 3, 8, 4])
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
>>> out = F.pad(t4d, p3d, "constant", 0)
>>> print(out.size())
torch.Size([3, 9, 7, 3])
```

## Torch Gradient Computation Mod 梯度追踪的开关

### torch.set_grad_enabled
设置梯度计算打开或关闭的上下文管理器。也可以用作函数，但不可用作装饰器。   
set_grad_enabled 将根据其参数`mod`启用或禁用梯度。  
这个上下文管理器是线程本地的；它不会影响其他线程中的计算。  
#### Args
- mode: bool – 是否开启梯度跟踪  
#### Examples
```python
>>> x = torch.tensor([1.], requires_grad=True)
>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...     y = x * 2
>>> y.requires_grad
False
>>> _ = torch.set_grad_enabled(True)
>>> y = x * 2
>>> y.requires_grad
True
>>> _ = torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False
```

### torch.is_grad_enabled()
获取当前是否开启梯度跟踪

### torch.no_grad(orig_func=None)
禁用梯度计算的上下文管理器。也可用作装饰器。  
当确定不会调用 Tensor.backward() 时，禁用梯度计算对于推理很有用。它将减少计算的内存消耗，否则需要 require_grad=True。在此模式下，每次计算的结果都将具有requires_grad=False，即使输入具有requires_grad=True。  
例外：所有工厂函数或创建新张量并传入`requires_grad`参数的函数都不会受到此模式的影响。  
这个上下文管理器是线程本地的；它不会影响其他线程中的计算。  

#### Examples
```python
>>> x = torch.tensor([1.], requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False
>>> @torch.no_grad()
... def doubler(x):
...     return x * 2
>>> z = doubler(x)
>>> z.requires_grad
False
>>> @torch.no_grad
... def tripler(x):
...     return x * 3
>>> z = tripler(x)
>>> z.requires_grad
False
>>> # factory function exception
>>> with torch.no_grad():
...     a = torch.nn.Parameter(torch.rand(10))
>>> a.requires_grad
True
```

### torch.enable_grad(orig_func=None)
启用梯度计算的上下文管理器。也可用作装饰器。  
如果已通过 no_grad 或 set_grad_enabled 禁用梯度计算，则启用梯度计算。  
这个上下文管理器是线程本地的；它不会影响其他线程中的计算。  

#### Examples
```python
>>> x = torch.tensor([1.], requires_grad=True)
>>> with torch.no_grad():
...     with torch.enable_grad():
...         y = x * 2
>>> y.requires_grad
True
>>> y.backward()
>>> x.grad
tensor([2.])
>>> @torch.enable_grad()
... def doubler(x):
...     return x * 2
>>> with torch.no_grad():
...     z = doubler(x)
>>> z.requires_grad
True
>>> @torch.enable_grad
... def tripler(x):
...     return x * 3
>>> with torch.no_grad():
...     z = tripler(x)
>>> z.requires_grad
True
```

## Mathematical Pointwise Ops 数学逐点操作

|     torch API      |     Tensor API      |        Alias        |                                                   Description                                                   |
|:------------------:|:-------------------:|:-------------------:|:---------------------------------------------------------------------------------------------------------------:|
|    torch.add()     |    Tensor.add()     |          /          |                                                        加                                                        |
|    torch.sub()     |    Tensor.sub()     |      subtract       |                                                        减                                                        |
|    torch.mul()     |    Tensor.mul()     |      multiply       |                                                        乘                                                        |
|    torch.div()     |    Tensor.div()     |       divide        |                                                        除                                                        |
|   torch.round()    |   Tensor.round()    |          /          |                                                      四舍五入                                                       |
|    torch.ceil()    |    Tensor.ceil()    |          /          |                                                 大于或等于每个元素的最大整数                                                  |
|   torch.floor()    |   Tensor.floor()    |          /          |                                                 小于或等于每个元素的最大整数                                                  |
| torch.reciprocal() | Tensor.reciprocal() |          /          |                                                       取倒数                                                       |
|  torch.positive()  |  Tensor.positive()  |          /          |                                                       取正                                                        |
|    torch.neg()     |    Tensor.neg()     |      negative       |                                                       取负                                                        |   
|    torch.abs()     |    Tensor.abs()     |      absolute       |                                                       绝对值                                                       |
|    torch.sign()    |    Tensor.sign()    |         sgn         |                                                取符号，正为1，负为-1，零为0                                                 |
|   torch.trunc()    |   Tensor.trunc()    |         fix         |                                                      取截断整数                                                      |
|    torch.frac()    |    Tensor.frac()    |          /          |                                                      取截断小数                                                      |
|    torch.pow()     |    Tensor.pow()     |          /          |                                                       乘方                                                        |
|   torch.square()   |   Tensor.square()   |          /          |                                                     平方 x^2                                                      |
|    torch.sqrt()    |    Tensor.sqrt()    |          /          |                                                   开方 x^(1/2)                                                    |
|   torch.rsqrt()    |   Tensor.rsqrt()    |          /          |                                                 开方并取倒数 x^(-1/2)                                                 |
|    torch.exp()     |    Tensor.exp()     |          /          |                                                       e^x                                                       |
|    torch.exp2()    |    Tensor.exp2()    |          /          |                                                       2^x                                                       |
|   torch.expm1()    |   Tensor.expm1()    |          /          |                                                      e^x-1                                                      |
|    torch.log()     |    Tensor.log()     |          /          |                                                       lnx                                                       |
|    torch.log2()    |    Tensor.log2()    |          /          |                                                     log2 x                                                      |
|   torch.log10()    |   Tensor.log10()    |          /          |                                                     log10 x                                                     |
|   torch.log1p()    |   Tensor.log1p()    |          /          |                                                     ln(1+x)                                                     |
|   torch.hypot()    |   Tensor.hypot()    |          /          |                                                 (x^2 + y^2)^0.5                                                 |
| torch.logaddexp()  | Tensor.logaddexp()  |          /          |                                                  ln(e^x + e^y)                                                  |
| torch.logaddexp2() | Tensor.logaddexp2() |          /          |                                                 log2(e^x + e^y)                                                 |
|    torch.sin()     |    Tensor.sin()     |          /          |                                                      sinx                                                       |
|    torch.cos()     |    Tensor.cos()     |          /          |                                                      cosx                                                       |
|    torch.tan()     |    Tensor.tan()     |          /          |                                                      tanx                                                       |
|    torch.asin()    |    Tensor.asin()    |       arcsin        |                                                     arcsinx                                                     |
|    torch.acos()    |    Tensor.acos()    |       arccos        |                                                     arccosx                                                     |
|    torch.atan()    |    Tensor.atan()    |       arctan        |                                                     arctanx                                                     |
|   torch.asinh()    |   Tensor.asinh()    |       arcsosh       |                                                双曲正弦 (e^x-e^x)/2                                                 |
|   torch.acosh()    |   Tensor.acosh()    |       arccosh       |                                                双曲余弦 (e^x+e^x)/2                                                 |
|   torch.atanh()    |   Tensor.atanh()    |       arctanh       |                                            双曲正切 (e^x-e^x)/(e^x+e^x)                                             |
|  torch.deg2rad()   |  Tensor.deg2rad()   |          /          |                                                    角度制转换为弧度制                                                    |
|  torch.rad2deg()   |  Tensor.rad2deg()   |          /          |                                                    弧度制转换为角度制                                                    |
|   torch.lgamma()   |   Tensor.lgamma()   |          /          |                                                  ln(gamma(x))                                                   |
|   torch.clamp()    |   Tensor.clamp()    |        clip         |                         将输入中的所有元素限制在 [ min, max ] 范围内。y=min(max(x,min_value),max_value)                         
|    torch.lerp()    |    Tensor.lerp()    |                     |                                         插值 out=start+weight×(end−start)                                         |
|  torch.sigmoid()   |  Tensor.sigmoid()   | torch.special.expit |                                                1 / (1 + e ^ -x)                                                 |
| torch.nan_to_num() | Tensor.nan_to_num() |          /          | 将输入中的 NaN、正无穷大和负无穷大值分别替换为 nan、posinf 和 neginf 指定的值。默认情况下，NaN 替换为零，正无穷大替换为输入数据类型可表示的最大有限值，负无穷大替换为输入数据类型可表示的最小有限值 |


## Reduction Operation 约简操作
|       torch API       |       Tensor API       | dim&keepdim Args |     Description      |
|:---------------------:|:----------------------:|:----------------:|:--------------------:|
|    torch.argmax()     |    Tensor.argmax()     |        √         |        最大值的索引        |
|    torch.argmin()     |    Tensor.argmin()     |        √         |        最小值的索引        |
|     torch.amax()      |     Tensor.amax()      |        √         |        最大值的数值        |
|     torch.amin()      |     Tensor.amin()      |        √         |        最小值的数值        |
|    torch.aminmax()    |    Tensor.aminmax()    |        √         |      最小值和最大值的数值      |
|      torch.max()      |      Tensor.max()      |        √         |      最大值的数值和索引       |
|      torch.min()      |      Tensor.min()      |        √         |      最小值的数值和索引       |
|      torch.all()      |      Tensor.all()      |        √         |    元素是否全为True(非0)    |
|      torch.any()      |      Tensor.any()      |        √         |    元素是否存在True(非0)    |
| torch.count_nonzero() | Tensor.count_nonzero() |        √         |        非0元素个数        |
|     torch.mean()      |     Tensor.mean()      |        √         |      计算所有元素的均值       |
|    torch.nanmean()    |    Tensor.nanmean()    |        √         |    计算所有非NaN元素的均值     |
|      torch.sum()      |      Tensor.sum()      |        √         |      计算所有元素的均值       |
|    torch.nansum()     |    Tensor.nansum()     |        √         |    计算所有非NaN元素的总和     |
|      torch.var()      |      Tensor.var()      |        √         |      计算所有元素的总和       |
|   torch.var_mean()    |           /            |        √         |   计算所有非NaN元素的方差和均值   |
|      torch.std()      |      Tensor.std()      |        √         |      计算所有元素的标准差      |
|   torch.std_mean()    |           /            |        √         |  计算所有非NaN元素的标准差和均值   |
|    torch.median()     |    Tensor.median()     |        √         |      计算所有元素的中位数      |
|   torch.nanmedian()   |   Tensor.nanmedian()   |        √         |    计算所有非NaN元素的中位数    |
|   torch.quantile()    |   Tensor.quantile()    |        √         |     计算所有元素的n分位数      |
|  torch.nanquantile()  |  Tensor.nanquantile()  |        √         |   计算所有非NaN元素的n分位数    |
|     torch.prod()      |     Tensor.prod()      |        √         |       计算所有元素相乘       |
|     torch.norm()      |     Tensor.norm()      |        √         |      计算张量的n阶范数       |
|     torch.dist()      |     Tensor.dist()      |        ×         | 计算两个张量的距离（张量之差的n阶范数） |


> ### 关于dim与keepdim  
> 当`dim`未指定时，即对张量全局进行操作（计算）。`dim`参数指定张量在指定维度上进行计算，可以指定1个或多个维度。  
> 例如，对一个形状为[2,3,4]的张量，指定`dim=1`，返回张量的形状为`[2,4]`，每次操作即对一个形状为[3]的张量切片进行计算。指定`dim=(0,1)`，返回张量的形状为`[4]`，每次操作即对一个形状为`[2,3]`的张量切片进行计算。  
> `keepdim=True`时则会保持输出张量与原输入张量的维度数一致，上例返回的形状则会变为`[2,1,4]`和`[1,1,4]`，相当于在原本因操作而失去的维度上进行`unsqueeze`。



### torch.argmax(input, dim=None, keepdim=False) -> LongTensor
获取`input`张量上最大值的索引。  
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。  
#### Args
- input: Tensor – the input tensor.  
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作  
- keepdim: bool – 输出张量是否保持与输入相同的维度  
#### Examples
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
        [-0.7401, -0.8805, -0.3402, -1.1936],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])
>>> torch.argmax(a)
tensor(0)
>>> torch.argmax(a, dim=1)
tensor([ 0,  2,  0,  1])
```

### torch.amax(input, dim=None, keepdim=False) -> LongTensor
获取`input`张量上最大值的数值。  
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。
#### Args
- input: Tensor – the input tensor.  
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局最大值)。指定dim时即返回沿dim维的各张量切片上最大值的索引。   
- keepdim: bool – 输出张量是否保持与输入相同的维度  
> `amax()` 在相等值之间均匀分布梯度，而 `max(dim)`仅将梯度传播到源张量中的单个索引。
#### Examples
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
        [-0.7158,  1.1775,  2.0992,  0.4817],
        [-0.0053,  0.0164, -1.3738, -0.0507],
        [ 1.9700,  1.1106, -1.0318, -1.0816]])
>>> torch.amax(a)
tensor(2.0992)
>>> torch.amax(a, 1)
tensor([1.4878, 2.0992, 0.0164, 1.9700])
```

### torch.aminmax(input, *, dim=None, keepdim=False) -> namedtuple[values:., indices:.]
获取`input`张量上最大值和最小值的数值。  
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。
#### Args
- input: Tensor – the input tensor.  
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局最大值)。指定dim时即返回沿dim维的各张量切片上最大值的数值。   
- keepdim: bool – 输出张量是否保持与输入相同的维度  
#### Examples
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
        [-0.7158,  1.1775,  2.0992,  0.4817],
        [-0.0053,  0.0164, -1.3738, -0.0507],
        [ 1.9700,  1.1106, -1.0318, -1.0816]])
>>> torch.aminmax(a)
torch.return_types.aminmax(
        min=tensor(-1.3738),
        max=tensor(2.0992)
        )
>>> torch.aminmax(a, dim=1)
torch.return_types.max(
        min=tensor([-0.2491, -0.7158, -1.3738, -1.0816]),
        max=tensor([1.4878, 2.0992, 0.0164, 1.9700])
        )
```

### torch.max(input, dim=None, keepdim=False) -> namedtuple[values:., indices:.]
获取`input`张量上最大值的数值和索引。  
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。  
#### Args
- input: Tensor – the input tensor.  
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局最大值)，指定dim时即返回沿dim维的各张量切片上最大值的数值和索引。    
- keepdim: bool – 输出张量是否保持与输入相同的维度（相当于额外unsqueeze操作，维度相同但形状不相同）  
> `amax()` 在相等值之间均匀分布梯度，而 `max(dim)`仅将梯度传播到源张量中的单个索引。
#### Examples
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
        [-0.7158,  1.1775,  2.0992,  0.4817],
        [-0.0053,  0.0164, -1.3738, -0.0507],
        [ 1.9700,  1.1106, -1.0318, -1.0816]])
>>> torch.max(a)
    
>>> torch.max(a, 1)
torch.return_types.max(
        values=tensor([1.4878, 2.0992, 0.0164, 1.9700]),
        indices=tensor([ 0,  2,  0,  1])
        )
```

> ###  类似地：  
> `torch.argmin`获取张量最小值索引   
> `torch.amin`获取张量最小值数值  
> `torch.min`获取张量最小值数值和索引  

### torch.all(input, dim=None, keepdim=False) -> Tensor
判断张量各是否全为True。  
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。  
#### Args
- input: Tensor – the input tensor.  
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局是否均为True)；指定dim时即返回沿dim维的各张量切片是否均为True。  
- keepdim: bool – 输出张量是否保持与输入相同的维度（相当于额外unsqueeze操作，维度相同但形状不相同）  
#### Examples
```python
>>> a = torch.rand(4, 2).bool()
>>> a
tensor([[True, True],
        [True, False],
        [True, True],
        [True, True]], dtype=torch.bool)
>>> torch.all(a, dim=1)
tensor([ True, False,  True,  True], dtype=torch.bool)
>>> torch.all(a, dim=0)
tensor([ True, False], dtype=torch.bool
```

### torch.any(input, dim=None, keepdim=False) -> Tensor
判断张量各是否有元素为True。  
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。  
#### Args
- input: Tensor – the input tensor.  
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局是否均为True)；指定dim时即返回沿dim维的各张量切片是否有元素为True。  
- keepdim: bool – 输出张量是否保持与输入相同的维度（相当于额外unsqueeze操作，维度相同但形状不相同）  
#### Examples
```python
>>> a = torch.randn(4, 2) < 0
>>> a
tensor([[ True,  True],
        [False,  True],
        [ True,  True],
        [False, False]])
>>> torch.any(a, 1)
tensor([ True,  True,  True, False])
>>> torch.any(a, 0)
tensor([True, True])
```



### torch.mean(input, dim=None, keepdim=False) -> Tensor
获取张量各元素的均值。  
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。  
#### Args
- input: Tensor  
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局均值)；指定dim时即返回沿dim维的各张量切片的均值。    
- keepdim: bool – 输出张量是否保持与输入相同的维度（相当于额外unsqueeze操作，维度相同但形状不相同）
#### Examples
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
        [-0.9644,  1.0131, -0.6549, -1.4279],
        [-0.2951, -1.3350, -0.7694,  0.5600],
        [ 1.0842, -0.9580,  0.3623,  0.2343]])
>>> torch.mean(a)
tensor(-0.2010)
>>> torch.mean(a, 1)
tensor([-0.0163, -0.5085, -0.4599,  0.1807])
>>> torch.mean(a, 1, True)
tensor([[-0.0163],
        [-0.5085],
        [-0.4599],
        [ 0.1807]])
```

### torch.sum(input, dim=None, keepdim=False) -> Tensor
获取张量各元素的总和。    
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。    
#### Args
- input: Tensor    
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局均值)；指定dim时即返回沿dim维的各张量切片的均值。      
- keepdim: bool – 输出张量是否保持与输入相同的维度（相当于额外unsqueeze操作，维度相同但形状不相同）  
#### Examples
```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
        [-0.2993,  0.9138,  0.9337, -1.6864],
        [ 0.1132,  0.7892, -0.1003,  0.5688],
        [ 0.3637, -0.9906, -0.4752, -1.5197]])
>>> torch.sum(a, 1)
tensor([-0.4598, -0.1381,  1.3708, -2.6217])
>>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
>>> torch.sum(b, (2, 1))
tensor([  435.,  1335.,  2235.,  3135.])
```

### torch.var(input, dim=None, *, correction=1, keepdim=False, out=None) -> Tensor
获取张量各元素的方差。  
Var(X) = Σ(x- E(X))^2 / (N-correction)，其中N为元素个数，correction为修正项。  
#### Args
- - input: Tensor       
- - dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局是否均为True)；指定dim时即返回沿dim维的各张量切片的方差。  
- - correction: int - 修正项，默认使用Bessel修正，为1。  
- - keepdim: bool – 输出张量是否保持与输入相同的维度（相当于额外unsqueeze操作，维度相同但形状不相同）      
#### Examples
```python
>>> a = torch.tensor(
...     [[ 0.2035,  1.2959,  1.8101, -0.4644],
...      [ 1.5027, -0.3270,  0.5905,  0.6538],
...      [-1.5745,  1.3330, -0.5596, -0.6548],
...      [ 0.1264, -0.5080,  1.6420,  0.1992]])
>>> torch.var(a, dim=1, keepdim=True)
tensor([[1.0631],
        [0.5590],
        [1.4893],
        [0.8258]])
```



### torch.norm(input, p='fro', dim=None, keepdim=False)->Tensor
返回张量的p阶范数。  
||X||p = (Σ|x|^p)^(1/p)    
特别地，当p=0时，||X||p为X中非0元素的个数；当p=float("inf")时，||X||p=max(|x|)     
若`input`张量的形状为[d_1,d_2,d_3,...,d_n], 指定`dim`维长度为d_dim，返回张量形状为[d_1, d_2, ..., d_{dim-1}, d_{dim+1}, ..., d_n]。    
#### Args
- input: Tensor     
- p: int or str - 范数的阶数。在所有情况下，Frobenius 范数都会产生与 p=2 相同的结果，除非 dim 是三个或更多 dim 的列表，在这种情况下，Frobenius 范数会引发错误。Nuclear范数只能在精确的两个维度上计算。  
- dim: Int or List – 进行索引的维度。为None时则将张量展平（Flatten）后再进行操作(即返回张量全局是否均为True)；指定dim时即返回沿dim维的各张量切片的范数。    
- keepdim: bool – 输出张量是否保持与输入相同的维度（相当于额外unsqueeze操作，维度相同但形状不相同）    
#### Examples
```python
>>> import torch
>>> a = torch.arange(9, dtype= torch.float) - 4
>>> b = a.reshape((3, 3))
>>> torch.norm(a)
tensor(7.7460)
>>> torch.norm(b)
tensor(7.7460)
>>> torch.norm(a, float('inf'))
tensor(4.)
>>> torch.norm(b, float('inf'))
tensor(4.)
>>> c = torch.tensor([[ 1, 2, 3], [-1, 1, 4]] , dtype=torch.float)
>>> torch.norm(c, dim=0)
tensor([1.4142, 2.2361, 5.0000])
>>> torch.norm(c, dim=1)
tensor([3.7417, 4.2426])
>>> torch.norm(c, p=1, dim=1)
tensor([6., 6.])
>>> d = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
>>> torch.norm(d, dim=(1, 2))
tensor([ 3.7417, 11.2250])
>>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
(tensor(3.7417), tensor(11.2250))
```

### torch.dist(input, other, p=2) -> Tensor
返回两个张量间的距离。即张量`input - output`的p阶范数。     
||X||p = (Σ|x|^p)^(1/p)  
特别地，当p=0时，||X||p为X中非0元素的个数；当p=float("inf")时，||X||p=max(|x|)   
#### Args
- input: Tensor   
- other: Tensor  
- p: int - 范数的阶数  
#### Examples
```python
>>> x = torch.randn(4)
>>> x
tensor([-1.5393, -0.8675,  0.5916,  1.6321])
>>> y = torch.randn(4)
>>> y
tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
>>> torch.dist(x, y, 3.5)
tensor(1.6727)
>>> torch.dist(x, y, 3)
tensor(1.6973)
>>> torch.dist(x, y, 0)
tensor(4.)
>>> torch.dist(x, y, 1)
tensor(2.6537)
```

## Distribution

### torch.distributions.distribution.Distribution
概率分布的抽象基类。

#### property
- mean 分布均值
- mode 分布众数
- event_shape 不考虑batcch，每次采样返回的形状
- batch_shape 批处理形状
- stddev 分布标准差
- variance 分布方差
- support 返回表示此分布support的 Constraint 对象。

#### cdf(value)
累计分布函数

#### icdf(value)
逆累积分布函数

#### log_pdf(value)
概率密度函数

#### entropy()
返回分布熵，通过`batch_shape`进行批处理。  
H(p) = -Σp(x)log[p(x)]

#### perplexity()
返回分布困惑度，通过`batch_shape`进行批处理。  
PP(p) = e^H(p) = Π p(x)^[-p(x)]


#### sample(sample_shape=torch.Size([]))
生成`sample_shape`形状的样本。  
如果分布参数是批处理的，则生成`sample_shape`形状的样本批次。

#### sample_n(n)
生成 n 个样本。  
如果分布参数是批处理的，则生成n批样本。

#### rsample(sample_shape=torch.Size([]))
重置参数采样。

#### enumerate_support(expand=True)
返回包含离散分布支持的所有值的张量。结果将在维度 0 上进行枚举，因此结果的形状将为 `(cardinality,) + batch_shape + event_shape` （对于单变量分布, `event_shape = ()` ）。  
注意，这枚举了锁步 `[[0, 0], [1, 1], ...]` 中的所有批处理张量。当 `expand=False` 时，枚举沿着 dim 0 进行，但剩余的批次维度是单例维度`[[0], [1], ...` 。   
要迭代完整的笛卡尔积，请使用 `itertools.product(m.enumerate_support()`。

#### expand(batch_shape, _instance=None)
批处理扩展。  
当Distribution本身`batch_size`不为`[]`时，注意传入`batch_shape`的末尾几个维度形状要与原形状相同。例如原本`batch_size`为`[4,3]`，扩展形状应为`[..., 4, 3]`。

> 推荐看这篇博客 https://bochang.me/blog/posts/pytorch-distributions/

### Distributions 
>施工中

|  Distribution   | Description |                  Parameters                  |
|:---------------:|:-----------:|:--------------------------------------------:|
|Bernoulli| 伯努利分布（二项分布） |                 probs:[0,1]                  |
|Beta|    贝塔分布     | concentration1:(0,+∞), concentration0:(0,+∞) |
|Binomial|    二项分布     |      total_count:int[0,+∞), probs:[0,1]      |
|Categorical|
|Cauchy|
|Chi2|
|ContinuousBernoulli|
|Dirichlet|
|Exponential|
|FisherSnedecor|
|Gamma|
|Geometric|
|Gumbel|
|HalfCauchy|
|HalfNormal|
|Independent|
|InverseGamma|
|Kumaraswamy|
|LKJCholesky|
|Laplace|
||  LogNormal  |
|NegativeBinomial|    负二项分布    |       total_count:[0,+∞), probs:[0,1)        |
|Normal| 正态分布 |             loc:R, scale:(0,+∞)              |
|OneHotCategorical|
|Pareto|
|Poisson| 泊松分布 |                 rate:[0,+∞)                  |
|StudentT| T分布 |                  df:(0,+∞)                   |
|Uniform| 随机分布 |                 low:R high:R                 |
|VonMises|
|Weibull|
|Wishart|

> logits - the log-odds of sampling 赔率的对数。  
> logits与probs的转换公式为：logits = ln[probs/(1-probs)], probs = 1/(1+e^-logits)




## Tensorboard

### tensorboard --logdir
指定日志文件夹，打开tensorboard

### torch.utils.tensorboard.writer.SummaryWriter
日志写入类。






