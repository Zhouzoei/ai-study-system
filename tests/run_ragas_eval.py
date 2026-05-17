import os
import sys
import json
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.pipeline import EnhancedRAGPipeline
from engines.evaluator import EvalSample


EVAL_DOC = """
# 深度学习基础

## 1. 神经网络概述

神经网络是一种模拟生物神经系统的计算模型，由大量的人工神经元相互连接而成。每个神经元接收输入信号，通过激活函数处理后产生输出。神经网络的核心思想是通过学习数据中的模式来完成任务，而非依赖人工编写的规则。

神经网络的基本结构包括输入层、隐藏层和输出层。输入层负责接收原始数据，隐藏层负责特征提取和变换，输出层负责产生最终的预测结果。深度神经网络是指具有多个隐藏层的神经网络，"深度"即指层数之多。

### 1.1 前向传播

前向传播是神经网络的基本计算过程。数据从输入层开始，逐层经过线性变换和非线性激活，最终到达输出层。在每一层中，神经元将输入与权重相乘并加上偏置，然后通过激活函数产生输出。数学表达式为：z = Wx + b，a = f(z)，其中W是权重矩阵，x是输入向量，b是偏置向量，f是激活函数。

常用的激活函数包括ReLU、Sigmoid和Tanh。ReLU函数定义为f(x) = max(0, x)，它解决了Sigmoid函数在深层网络中的梯度消失问题，是目前最常用的激活函数。Sigmoid函数将输出压缩到(0,1)区间，常用于二分类的输出层。Tanh函数将输出压缩到(-1,1)区间，在某些场景下优于Sigmoid。

### 1.2 损失函数

损失函数用于衡量模型预测值与真实值之间的差距。回归任务常用均方误差损失（MSE），分类任务常用交叉熵损失。均方误差损失计算预测值与真实值之差的平方的均值。交叉熵损失衡量预测概率分布与真实分布之间的差异，在分类任务中表现更好。

对于多分类问题，通常使用Softmax函数将输出转换为概率分布，然后计算交叉熵损失。Softmax函数定义为：softmax(z_i) = exp(z_i) / Σexp(z_j)，它确保所有输出概率之和为1。

### 1.3 激活函数详解

激活函数是神经网络中引入非线性的关键组件。没有激活函数，多层神经网络只能表示线性变换，无法学习复杂的非线性模式。ReLU函数的导数为0或1，计算简单且不会造成梯度消失，但存在"神经元死亡"问题——当输入为负时梯度为0，神经元可能永远无法更新。Leaky ReLU通过给负区间一个小的斜率来解决这个问题，通常取alpha=0.01。ELU函数在负区间使用指数函数，输出均值接近零，加速收敛。

Sigmoid函数的输出范围在0到1之间，适合表示概率。但其导数最大值为0.25，深层网络中梯度连乘后快速衰减，导致梯度消失。Tanh函数输出范围为-1到1，均值为零，比Sigmoid更优，但同样有梯度饱和问题。在实际应用中，隐藏层首选ReLU及其变体，输出层根据任务选择：二分类用Sigmoid，多分类用Softmax，回归用线性激活。

### 1.4 权重初始化

权重初始化对深度神经网络的训练至关重要。不好的初始化会导致梯度消失或梯度爆炸。Xavier初始化（也被称为Glorot初始化）适用于Sigmoid和Tanh激活函数，权重从均匀分布或正态分布中采样，方差为1/n_in或2/(n_in+n_out)。He初始化专为ReLU设计，方差为2/n_in。

零初始化是最直观但最不可取的方法——所有神经元输出相同，梯度相同，无法打破对称性。过大初始化会导致梯度爆炸，过小会导致梯度消失。Batch Normalization可以减轻对初始化的依赖，通过规范化每层的输入分布，使训练更稳定。

## 2. 反向传播算法

反向传播算法是训练神经网络的核心算法，由Rumelhart等人在1986年提出。它通过链式法则高效地计算损失函数对每个参数的梯度，从而实现参数的优化更新。

### 2.1 链式法则

链式法则是微积分中求复合函数导数的基本法则。在神经网络中，由于计算过程是多个函数的复合，因此需要使用链式法则逐层计算梯度。对于复合函数y = f(g(x))，其导数为dy/dx = dy/dg * dg/dx。在多层网络中，梯度从输出层向输入层逐层传递，这就是"反向传播"名称的由来。

### 2.2 梯度计算

在反向传播中，首先计算损失函数对输出层参数的梯度，然后逐层向输入层方向计算。每一层的梯度计算依赖于上一层的梯度结果。具体来说，对于第l层，其权重梯度为∂L/∂W_l = ∂L/∂z_l * a_{l-1}^T，其中a_{l-1}是上一层的激活值。偏置梯度为∂L/∂b_l = ∂L/∂z_l。

梯度计算的效率是反向传播的关键优势。通过复用中间计算结果，反向传播的计算复杂度与前向传播相当，这比数值梯度方法快了数个数量级。数值梯度每次只扰动一个参数，计算一次前向传播，对于有百万参数的网络来说完全不可行。

### 2.3 梯度消失与梯度爆炸

梯度消失是指在深层网络中，梯度在反向传播过程中不断缩小，导致靠近输入层的参数几乎无法更新。这主要是因为Sigmoid和Tanh激活函数的导数最大值分别为0.25和1，多层相乘后梯度呈指数级衰减。梯度消失导致深层网络无法有效学习，模型性能随着层数增加反而下降。

梯度爆炸则相反，梯度在传播过程中不断增大，导致参数更新幅度过大，模型无法收敛。梯度爆炸通常可以通过梯度裁剪来缓解，即当梯度范数超过阈值时，将其缩放到阈值范围内。梯度裁剪的常用阈值在1到10之间。

解决方案包括：使用ReLU激活函数替代Sigmoid，采用残差连接（ResNet）使梯度可以直接传播，使用批归一化（Batch Normalization）稳定每层的输入分布，以及使用合适的权重初始化方法如Xavier初始化或He初始化。

### 2.4 残差网络

残差网络（ResNet）是解决深层网络梯度消失的里程碑式工作。残差网络的核心思想是引入跳跃连接（skip connection），允许梯度直接通过恒等映射传播到前面的层。在传统网络中，每一层学习的是从输入到输出的直接映射H(x)。在残差网络中，每一层学习的是残差函数F(x) = H(x) - x，实际输出为F(x) + x。

跳跃连接不会增加额外的参数和计算量，但极大地改善了梯度流动。ResNet使得训练上百层的深度网络成为可能。在ImageNet比赛中，ResNet-152的深度是VGG-19的8倍，但参数量更少，性能更好。残差连接也被广泛应用于Transformer架构中，是现代深度学习的基础设计模式之一。

### 2.5 批归一化

批归一化（Batch Normalization）是另一种解决训练不稳定问题的技术。它在每层的激活函数之前，对每个mini-batch的数据进行规范化，使其均值为0、方差为1。然后通过可学习的缩放参数γ和平移参数β恢复表达能力。批归一化的完整计算为：y = γ * (x - μ)/√(σ²+ε) + β。

批归一化的优点包括：加速训练收敛、允许使用更大的学习率、减轻对初始化的依赖、具有一定的正则化效果。在训练时，μ和σ基于当前batch计算；在推理时，使用训练过程中累计的全局均值和方差。需要注意的是，批归一化对batch size敏感，当batch size很小时效果不佳。Layer Normalization是批归一化的替代方案，不依赖batch维度，在Transformer中广泛使用。

## 3. 优化算法

优化算法决定了模型如何利用梯度信息来更新参数，直接影响训练速度和最终性能。

### 3.1 随机梯度下降

随机梯度下降（SGD）是最基本的优化算法，每次使用一个或一小批样本来更新参数。参数更新公式为：θ = θ - η * ∇L(θ)，其中η是学习率，∇L(θ)是损失函数的梯度。SGD的优点是计算简单、内存开销小，缺点是收敛速度慢且容易陷入局部最优。

学习率是SGD最重要的超参数。学习率过大会导致训练不稳定甚至发散，学习率过小则收敛缓慢。实践中常使用学习率衰减策略，在训练初期使用较大的学习率快速收敛，后期逐渐减小学习率精细调整。常见的学习率衰减方式包括阶梯衰减、指数衰减和余弦退火。

### 3.2 动量法

动量法在SGD的基础上引入了动量项，模拟物理中的惯性效应。更新公式为：v = βv + ∇L(θ)，θ = θ - ηv，其中β是动量系数（通常为0.9）。动量法能够加速在梯度方向一致的维度上的更新，抑制在梯度方向震荡的维度上的更新，从而加快收敛。当梯度方向变化剧烈时，动量起到平滑作用，减少震荡。

Nesterov动量是动量法的改进版本，它先按照当前动量方向前进一步，然后在新位置计算梯度。这种"前瞻"策略使得优化更加准确，在凸优化问题中有更好的理论保证。Nesterov动量的更新公式为：v = βv + ∇L(θ-βv)，θ = θ - ηv。实际应用中，Nesterov动量通常比标准动量收敛更快、更稳定。

### 3.3 自适应学习率算法

自适应学习率算法为每个参数维护独立的学习率，根据梯度的历史信息自动调整。AdaGrad累积梯度平方和，对频繁更新的参数使用较小的学习率，但存在学习率单调递减的问题。RMSProp通过指数移动平均替代累积，解决了AdaGrad学习率递减的问题，衰减率ρ通常取0.9或0.99。

Adam算法结合了动量法和RMSProp的优点，同时维护梯度的一阶矩估计和二阶矩估计。Adam的更新公式为：m = β₁m + (1-β₁)g，v = β₂v + (1-β₂)g²，然后进行偏差校正：m_hat = m/(1-β₁^t)，v_hat = v/(1-β₂^t)，最终更新为θ = θ - η * m_hat/(√v_hat+ε)。Adam是目前最常用的优化算法，在大多数任务上都有良好的表现，默认超参数（β₁=0.9, β₂=0.999, ε=1e-8）通常不需要调整。

### 3.4 学习率调度

学习率调度是优化过程中重要的策略。余弦退火调度按照余弦函数周期性地调整学习率，从初始学习率逐渐下降到最小值。热重启随机梯度下降（SGDR）在余弦退火的基础上引入周期性重启，每次重启后学习率重置为初始值，帮助模型跳出局部最优。One Cycle策略在一个周期内先升温再降温，在保持高学习率的同时保证训练稳定性。

学习率预热在训练初期使用较小的学习率，逐步增加到目标学习率，避免模型参数在初始阶段剧烈变化。预热通常持续几个epoch，然后切换到主调度策略。学习率调度的选择对模型最终性能有显著影响，通常需要根据任务和数据集进行调整。

## 4. 正则化技术

正则化技术用于防止模型过拟合，提高泛化能力。过拟合是指模型在训练数据上表现很好，但在未见数据上表现较差的现象。

### 4.1 L1和L2正则化

L1正则化在损失函数中加入参数绝对值之和的惩罚项，倾向于产生稀疏的权重矩阵，可以用于特征选择。L2正则化（也称为权重衰减）加入参数平方和的惩罚项，倾向于使权重值较小但非零，防止某些权重过大。数学上，L1正则化后的损失为L_reg = L + λΣ|w_i|，L2正则化后的损失为L_reg = L + λΣw_i²，其中λ是正则化强度。

L1和L2正则化的本质区别在于权重更新的方式。L1正则化每次减去一个常数，权重的绝对值逐渐缩小到零；L2正则化每次减去一个与权重成比例的量，权重趋近于零但不会完全为零。在实际应用中，L2正则化更常用，因为它的解更稳定。Elastic Net结合了L1和L2，同时具有稀疏性和稳定性。

### 4.2 Dropout

Dropout是深度学习中最常用的正则化技术之一。在训练过程中，Dropout以概率p随机将某些神经元的输出置为零，使得模型不能过度依赖任何一个神经元。这相当于训练了多个子网络的集成，在测试时使用所有神经元但将输出乘以(1-p)来保持期望值不变。

Dropout的概率p通常设置为0.5，但在不同层可以使用不同的丢弃率。对于输入层，通常使用较小的丢弃率（如0.2），因为输入信息的丢失代价更高。对于全连接层，使用较大的丢弃率（如0.5）。Dropout在卷积层中较少使用，因为卷积层的参数共享本身就具有正则化效果。Spatial Dropout是专门为卷积层设计的变体，它丢弃整个特征图而不是单个神经元。

### 4.3 数据增强

数据增强通过对训练数据施加随机变换来扩充数据集，是防止过拟合的有效方法。在图像领域，常用的增强方法包括随机裁剪、水平翻转、颜色抖动和旋转等。在自然语言处理领域，常用的方法包括同义词替换、随机删除和回译等。在语音领域，常用的方法包括速度调整、音量变化和背景噪声叠加等。

数据增强的核心思想是让模型看到更多样化的输入，从而学习到更加鲁棒的特征。好的数据增强策略应该在不改变数据语义的前提下增加数据的多样性。Mixup是另一种数据增强方法，它通过线性插值两个训练样本及其标签来创建新的训练样本。CutMix在Mixup的基础上引入了区域丢弃，随机裁剪一个图像的区域粘贴到另一个图像上。

### 4.4 Early Stopping

Early Stopping是最简单有效的正则化方法之一。在训练过程中监控验证集的性能，当验证集性能在连续多个epoch内不再提升时，提前终止训练。Early Stopping可以防止模型在训练集上过度优化，同时节省计算资源。

Early Stopping的主要参数是patience，即允许验证集性能不提升的连续epoch数。patience太大会导致过拟合，太小会导致欠拟合。实践中通常设置patience为10到50。Early Stopping通常与其他正则化方法结合使用，如L2正则化和Dropout。Model Checkpoint与Early Stopping配合使用，在验证集性能最好时保存模型权重。

## 5. 模型评估与选择

### 5.1 交叉验证

交叉验证是评估模型泛化性能的标准方法。K折交叉验证将数据集分为K份，每次使用K-1份训练、1份验证，重复K次后取平均值。K通常取5或10。留一法交叉验证（LOOCV）是K折交叉验证的特例，K等于样本数，适用于小样本场景。

交叉验证的优点是评估结果稳定可靠，缺点是在大规模数据集上计算开销大。在实践中，如果数据量充足，简单验证集划分（训练集-验证集-测试集）通常就足够了。分层交叉验证在分类任务中保持每折的类别分布与原始数据一致，避免因类别分布不均导致的评估偏差。

### 5.2 性能指标

分类任务的常用指标包括准确率、精确率、召回率、F1分数和AUC。准确率是预测正确的样本比例，但在类别不平衡时不可靠。精确率是预测为正类中实际为正类的比例，召回率是实际为正类中被预测为正类的比例，F1分数是精确率和召回率的调和平均。

回归任务的常用指标包括均方误差（MSE）、平均绝对误差（MAE）和R²分数。MSE对异常值敏感，MAE更稳健，R²分数衡量模型对数据方差的解释程度。混淆矩阵是分类任务的基础分析工具，展示了各类别下的预测分布情况。ROC曲线和PR曲线用于评估模型的排序性能。
"""


EVAL_SAMPLES = [
    EvalSample(
        question="什么是神经网络？",
        ground_truth="神经网络是一种模拟生物神经系统的计算模型，由大量的人工神经元相互连接而成。每个神经元接收输入信号，通过激活函数处理后产生输出。其核心思想是通过学习数据中的模式来完成任务，而非依赖人工编写的规则。基本结构包括输入层、隐藏层和输出层。",
    ),
    EvalSample(
        question="前向传播的计算过程是什么？",
        ground_truth="前向传播是神经网络的基本计算过程。数据从输入层开始，逐层经过线性变换和非线性激活，最终到达输出层。在每一层中，神经元将输入与权重相乘并加上偏置，然后通过激活函数产生输出。数学表达式为z=Wx+b，a=f(z)，其中W是权重矩阵，x是输入向量，b是偏置向量，f是激活函数。",
    ),
    EvalSample(
        question="ReLU激活函数有什么优势？",
        ground_truth="ReLU函数定义为f(x)=max(0,x)，它解决了Sigmoid函数在深层网络中的梯度消失问题，是目前最常用的激活函数。",
    ),
    EvalSample(
        question="交叉熵损失适用于什么任务？",
        ground_truth="交叉熵损失适用于分类任务，它衡量预测概率分布与真实分布之间的差异。对于多分类问题，通常使用Softmax函数将输出转换为概率分布，然后计算交叉熵损失。",
    ),
    EvalSample(
        question="为什么反向传播比数值梯度方法快？",
        ground_truth="反向传播通过复用中间计算结果，计算复杂度与前向传播相当，比数值梯度方法快了数个数量级。它利用链式法则逐层计算梯度，梯度从输出层向输入层逐层传递。",
    ),
    EvalSample(
        question="梯度消失的原因是什么？",
        ground_truth="梯度消失主要是因为Sigmoid和Tanh激活函数的导数最大值分别为0.25和1，多层相乘后梯度呈指数级衰减。在深层网络中，梯度在反向传播过程中不断缩小，导致靠近输入层的参数几乎无法更新。",
    ),
    EvalSample(
        question="梯度消失和梯度爆炸有什么区别？",
        ground_truth="梯度消失是梯度在反向传播中不断缩小，导致靠近输入层的参数几乎无法更新；梯度爆炸则是梯度不断增大，导致参数更新幅度过大，模型无法收敛。消失主要由Sigmoid/Tanh导数小于1导致，爆炸则可能由权重初始化不当引起。",
    ),
    EvalSample(
        question="如何解决梯度消失问题？",
        ground_truth="解决方案包括：使用ReLU激活函数替代Sigmoid，采用残差连接（ResNet）使梯度可以直接传播，使用批归一化（Batch Normalization）稳定每层的输入分布，以及使用合适的权重初始化方法如Xavier初始化或He初始化。",
    ),
    EvalSample(
        question="SGD的学习率如何影响训练？",
        ground_truth="学习率过大会导致训练不稳定甚至发散，学习率过小则收敛缓慢。实践中常使用学习率衰减策略，在训练初期使用较大的学习率快速收敛，后期逐渐减小学习率精细调整。",
    ),
    EvalSample(
        question="动量法相比SGD有什么改进？",
        ground_truth="动量法在SGD基础上引入了动量项，模拟物理中的惯性效应。它能够加速在梯度方向一致的维度上的更新，抑制在梯度方向震荡的维度上的更新，从而加快收敛。",
    ),
    EvalSample(
        question="Adam优化算法有什么特点？",
        ground_truth="Adam算法结合了动量法和RMSProp的优点，同时维护梯度的一阶矩估计和二阶矩估计。更新公式中包含偏置校正，使得训练初期也能保持稳定。Adam是目前最常用的优化算法，默认超参数通常不需要调整。",
    ),
    EvalSample(
        question="Adam和RMSProp有什么区别？",
        ground_truth="RMSProp通过指数移动平均替代AdaGrad的累积，解决了学习率递减问题，但只有二阶矩估计。Adam同时维护梯度的一阶矩估计（类似动量）和二阶矩估计（类似RMSProp），并包含偏置校正，在训练初期更稳定。",
    ),
    EvalSample(
        question="L1和L2正则化有什么区别？",
        ground_truth="L1正则化加入参数绝对值之和的惩罚项，倾向于产生稀疏的权重矩阵，可用于特征选择。L2正则化加入参数平方和的惩罚项，倾向于使权重值较小但非零，防止某些权重过大。",
    ),
    EvalSample(
        question="Dropout是如何防止过拟合的？",
        ground_truth="Dropout在训练过程中以概率p随机将某些神经元的输出置为零，使模型不能过度依赖任何一个神经元。这相当于训练了多个子网络的集成。测试时使用所有神经元但将输出乘以(1-p)保持期望值不变。",
    ),
    EvalSample(
        question="数据增强的核心思想是什么？",
        ground_truth="数据增强的核心思想是让模型看到更多样化的输入，从而学习到更加鲁棒的特征。好的数据增强策略应该在不改变数据语义的前提下增加数据的多样性。",
    ),
    EvalSample(
        question="如何训练一个深度神经网络？",
        ground_truth="训练深度神经网络的步骤包括：1)前向传播计算预测值；2)通过损失函数计算预测与真实的差距；3)反向传播计算梯度；4)使用优化算法更新参数；5)重复以上步骤直到收敛。同时需要使用正则化技术防止过拟合。",
    ),
    EvalSample(
        question="残差网络是如何解决梯度消失问题的？",
        ground_truth="残差网络通过引入跳跃连接（skip connection）来解决梯度消失问题。跳跃连接允许梯度直接通过恒等映射传播到前面的层，每一层学习的是残差函数F(x)=H(x)-x而不是直接映射H(x)。残差连接不会增加额外参数和计算量，使得训练上百层的深度网络成为可能。",
    ),
    EvalSample(
        question="批归一化的作用是什么？",
        ground_truth="批归一化在每层激活函数之前对mini-batch数据进行规范化，使其均值为0、方差为1，然后通过可学习的缩放参数和平移参数恢复表达能力。其优点包括：加速训练收敛、允许使用更大的学习率、减轻对初始化的依赖、具有一定的正则化效果。",
    ),
    EvalSample(
        question="Xavier初始化和He初始化有什么区别？",
        ground_truth="Xavier初始化适用于Sigmoid和Tanh激活函数，权重方差为1/n_in或2/(n_in+n_out)。He初始化专为ReLU设计，方差为2/n_in。零初始化会导致所有神经元输出相同，无法打破对称性。过大会导致梯度爆炸，过小会导致梯度消失。",
    ),
    EvalSample(
        question="Early Stopping是如何防止过拟合的？",
        ground_truth="Early Stopping在训练过程中监控验证集的性能，当验证集性能在连续多个epoch内不再提升时提前终止训练。主要参数是patience（允许不提升的连续epoch数），通常设置为10到50。它是最简单有效的正则化方法之一。",
    ),
]


def clean_db_files():
    for f in ["tree_store.db", "conversation_memory.db", "learning_planner.db",
              "progress_tracker.db", "knowledge_graph.db", "document_manager.db",
              "learning_reminder.db", "learning_analytics.db"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


def clean_qdrant_collection():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        url = os.getenv("QDRANT_URL")
        key = os.getenv("QDRANT_API_KEY")
        if url and key:
            client = QdrantClient(url=url, api_key=key, timeout=10)
            existing = [c.name for c in client.get_collections().collections]
            for cname in ["hierarchical_chunks", "qa_system_vectors"]:
                if cname in existing:
                    client.delete_collection(cname)
                    print(f"  Cleared Qdrant collection: {cname}")
    except Exception:
        pass


def run_evaluation():
    print("=" * 70)
    print("RAGAS Evaluation - Baseline")
    print("=" * 70)

    clean_db_files()
    clean_qdrant_collection()

    from utils.llm_service import LLMService
    from utils.embedding_service import EmbeddingService

    llm_service = LLMService()
    embed_service = EmbeddingService()

    llm_func = llm_service.invoke
    embed_func = embed_service.embed

    pipeline = EnhancedRAGPipeline(embed_func=embed_func, llm_func=llm_func)

    print("\n[1/4] Ingesting document...")
    ingest_result = pipeline.ingest(EVAL_DOC, doc_id="dl_basics", title="深度学习基础")
    print(f"  Nodes: {ingest_result['total_nodes']}, "
          f"L1={ingest_result['level_counts'].get(1, 0)}, "
          f"L2={ingest_result['level_counts'].get(2, 0)}, "
          f"L3={ingest_result['level_counts'].get(3, 0)}")

    print("\n[2/4] Preparing eval samples...")
    print(f"  Total samples: {len(EVAL_SAMPLES)}")

    strategies = [
        {"use_hybrid": False, "use_reranker": False, "use_rewriting": False, "name": "vector_only"},
        {"use_hybrid": True, "use_reranker": False, "use_rewriting": False, "name": "hybrid"},
        {"use_hybrid": True, "use_reranker": True, "use_rewriting": False, "name": "hybrid_reranker"},
        {"use_hybrid": True, "use_reranker": False, "use_rewriting": True, "name": "hybrid_rewrite"},
        {"use_hybrid": True, "use_reranker": True, "use_rewriting": True, "name": "hybrid_rewrite_reranker"},
    ]

    all_results = {}

    for idx, strat in enumerate(strategies):
        print(f"\n[3/4] Evaluating strategy {idx+1}/{len(strategies)}: {strat['name']}...")
        samples_copy = []
        for s in EVAL_SAMPLES:
            samples_copy.append(EvalSample(
                question=s.question,
                ground_truth=s.ground_truth,
            ))

        result = pipeline.evaluate(
            samples_copy,
            use_hybrid=strat["use_hybrid"],
            use_reranker=strat["use_reranker"],
            use_rewriting=strat.get("use_rewriting", False),
        )

        all_results[strat["name"]] = result
        summary = result.get("summary", {})
        print(f"  Faithfulness:     {summary.get('avg_faithfulness', 'N/A')}")
        print(f"  Answer Relevancy: {summary.get('avg_answer_relevancy', 'N/A')}")
        print(f"  Context Precision:{summary.get('avg_context_precision', 'N/A')}")
        print(f"  Context Recall:   {summary.get('avg_context_recall', 'N/A')}")
        print(f"  Avg Time:         {summary.get('avg_total_time_ms', 'N/A')}ms")

    print("\n[4/4] Generating comparison report...")
    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)
    print(f"{'Metric':<22} {'vector_only':>14} {'hybrid':>14} {'hybrid+rank':>14} {'+rewrite':>14} {'+rew+rank':>14}")
    print("-" * 70)

    metrics = [
        ("Faithfulness", "avg_faithfulness"),
        ("Answer Relevancy", "avg_answer_relevancy"),
        ("Context Precision", "avg_context_precision"),
        ("Context Recall", "avg_context_recall"),
        ("Avg Time (ms)", "avg_total_time_ms"),
    ]

    for label, key in metrics:
        row = []
        for strat in strategies:
            val = all_results[strat["name"]].get("summary", {}).get(key, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        print(f"{label:<22} {row[0]:>14} {row[1]:>14} {row[2]:>14} {row[3]:>14} {row[4]:>14}")

    report = {}
    for strat in strategies:
        name = strat["name"]
        report[name] = all_results[name].get("summary", {})

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_baseline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {report_path}")

    pipeline.close()
    clean_db_files()

    return report


if __name__ == "__main__":
    run_evaluation()