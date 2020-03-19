Transformer

在Seq2Seq结构中

RNN 处理序列信息， 必须要看到前边的输入，才可以决定下一时刻的输出， 无法并行计算。

堆叠多层的CNN, 可以进行并行计算。高层的CNN filter可以捕捉到sequence的信息，但是需要很多层CNN。

Self-Attention layer 被提出取代RNN

PAPER(Attention is all you need)

<img src="C:\Users\hanshangzhuang\Desktop\11.jpg" alt="11" style="zoom:120%;" />

Example:

Input(x1, x2, x3), Output(b1, b2, b3)
$$
a^i = Wx^i
$$
从a会分出三个vector，q, k, v， q: to match others, k: to be matched, v: information to be extracted
$$
q^i = W^qa^i;~~~k^i=W^ka^i;~~~v^i=W^va^i
$$
用q 去注意每一个k, 生成一个值（计算方法多样），这里举例计算b1。
$$
Scaled~~Dot-Product Attention:~~\alpha_1,_i = q^1\cdot k^i/\sqrt{d}
$$
d is the dim of q and k.

下一步会接一个Soft-max 层。
$$
\hat \alpha _1,_i = exp(\alpha_1,_i)/\sum exp(\alpha_1,_j)
$$
得到alpha hat之后，就可以计算输出b1。
$$
b^1 = \sum_i \hat\alpha_1,_i v^i
$$
可以看到b1考虑到了整个序列的信息，实际上，通过控制hat alpha的值可以控制输出b1考虑局部信息 or 全局信息 or 指定某个位置的信息。exp: 如果让x3输入的节点产生的hat alpha为0， 则可以控制b1只考虑x1, x2的信息。

并行计算：
$$
q^1 q^2 q^3 = W^q a^1 a^2 a^3 a^4;
$$
k, v 同理。K = k1, k2, k3;   V = v1, v2, v3
$$
matrix~\alpha_i,_j = K^T~ * Q
$$

$$
matrix~ \hat\alpha _i,_j = softmax(\alpha_i,_j)
$$

用hat alpha 矩阵乘 V, 就可以得到输出矩阵O。



self-Attention layer 对input的顺序不敏感，远处的信息和近处的信息都是同样被attention。所以在生成ai的时候要另外加入位置信息 ei (not learned from data)。

[linked video](https://www.youtube.com/watch?v=ugWDIIOHtPA)