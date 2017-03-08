# -*- coding:utf-8 -*-
""""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
# 给每一个char一个编码
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for.也就是从序列中截取样本的长度
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  input是转换成integer的文档(根据每个char的编码)
  
  input targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  # 
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
	# 这里的求微分过程分为对权重矩阵求微分和对隐藏层求微分。前者只需要前一层的值和
	# 后一层的误差，后者需要前一层权重矩阵的转置和后一层矩阵的误差
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
	# dhnext是一个辅助性的变量，用来记录当前层对下一层的影响。
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    # so brutal force!
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  # 返回经过这一个样本sequence训练之后的loss,更新项，和隐藏层
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  h是前一个input训练所得的hidden_layer，seed_ix是当前的input的第一个char,n是希望生成
  新char的个数
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    # 每次循环都更新了h.
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    # 根据计算出来的概率分布p，采样得到最优的下一个字符ix
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    # 以ix和当前的h为输入，在下一次循环中计算新的概率p
    x[ix] = 1
    ixes.append(ix)
  return ixes


# p是当前读取文件的cursor.n是已经完成的迭代次数
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
# 对loss做一个初始化。当然完全可以把初始值定为0，但这里是判断为对当前inputs
# 序列要预测的每一个值(总共有seq_length个)，都以1.0/vocab_size的概率进行估计
# 然后累加起来。alternatively,如果初始化为0，对各个参数的更新是没有影响的，
# 但是总的loss值会在最初的阶段有比较大的波动。所以这里对smoothloss进行非0初始化
# 的意义是让loss更平滑，也就是smooth_loss的意思。无非就是画出来更好看啦

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # 每次迭代，取长度为seq_length的序列作为input,向后滑动一位作为output
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
	# 利用当前input的第一个字符，生成接下来的200个字符
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # 计算这个sample input的损失函数的权值更新
  # 将上一个训练样本t=T的h_prev作为这个样本t=1的h_prev,因为样本在时间上是
  # 连续且不重复的，所以这样做没有问题。整个过程就是把全序列划分成很多小
  # sample,相继投入模型进行训练。
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  # 更新用于显示的smooth_loss
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    # 对五个参数分别更新,注意这里是本地更新,param实际上是打包了五个变量
    # 在每一轮迭代中mem都会加上d值的平方，这就导致param的改变随着过程的进行
    # 越来越小。
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
