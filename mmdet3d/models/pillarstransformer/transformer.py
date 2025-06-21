# -*- codeing = utf-8 -*-
# @Time : 2023/6/13 19:46
# @Author : 张庭恺
# @File : transformer.py
# @Software : PyCharm
import copy
import math
from mmcv.utils import Registry



import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# 输入 格式（B , N_points , 3）
class SelfAttention_xyz(nn.Module):
	def __init__(self, input_dim):
		super(SelfAttention_xyz, self).__init__()
		# 查询矩阵
		self.query = nn.Linear(input_dim, input_dim)
		# 键矩阵
		self.key = nn.Linear(input_dim, input_dim)
		# 值矩阵
		self.value = nn.Linear(input_dim, input_dim)
		self.sofa_max = nn.Softmax(dim=-1)

	def forward(self, x):
		query = self.query(x)
		key = self.key(x)
		value = self.value(x)

		# x : (b,n,features) 2*4096*4
		# QK相乘得到相似度得分
		# score : shape(b,4096,4096)
		score = (torch.bmm(query, torch.transpose(key, 1, 2))) / torch.sqrt(torch.tensor(x.size(-1))).float()

		attention_weights = self.sofa_max(score)
		weighted_valure = torch.bmm(attention_weights, value)
		out_put = weighted_valure

		return out_put


class Lyer_norm(nn.Module):
	def __init__(self, input_dim, dropout=0.1):
		super(Lyer_norm, self).__init__()

		# 归一化
		self.layer_norm = nn.LayerNorm(input_dim)
		# MultiHead_SelfAttention输出形状为(B , 512,256)
		# self.sub_layer = MultiHead_SelfAttention(input_dim,head_num)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()

	def forward(self, x, sub_layer):
		# 这是用激活函数的
		# 残差边
		output = self.dropout(self.relu(self.layer_norm(sub_layer(x) + x)))
		# 这是不用激活函数的
		# output = self.dropout(self.layer_norm(sub_layer(x) + x))

		# 输出tensor 形状 (B , 512 , 256)
		return output


# (B, 256 ,512)
# 这是单头注意力
class SelfAttention_feature(nn.Module):
	def __init__(self, input_dim):
		super(SelfAttention_feature, self).__init__()
		# 查询矩阵
		self.query = nn.Linear(input_dim, input_dim)
		# 键矩阵
		self.key = nn.Linear(input_dim, input_dim)
		# 值矩阵
		self.value = nn.Linear(input_dim, input_dim)
		self.sofa_max = nn.Softmax(dim=-1)

	def forward(self, x):
		'''
		输入的feature：B *256 *512
		需要先进行transpose
		'''  # x : (B, 256 ,512) --> transpose --> (B , 512 , 256)
		x = torch.transpose(x, 1, 2)
		query = self.query(x)
		key = self.key(x)
		value = self.value(x)

		# Q,K,V 都是 ( B , 512 ,256 )
		# QK相乘得到相似度得分
		# score : shape(b,512,512)
		# score = (torch.bmm(query, torch.transpose(key, 1, 2))) / torch.sqrt(torch.tensor(x.size(-1))).float()
		score = (torch.bmm(query, torch.transpose(key, 1, 2))) / torch.sqrt(torch.tensor(x.size(-1))).float()

		attention_weights = self.sofa_max(score)
		# weighted_valure: (B , 512 ,256)
		weighted_valure = torch.bmm(attention_weights, value)
		# 最后再进行一个转置，将特征维度放到第二维
		# out_put: (B, 256, 512)
		out_put = weighted_valure.transpose(1, 2)

		return out_put


# 就是两个线性变化
class FeedForward(nn.Module):
	'''
	w2(relu((w1(layer_norm(x))+b1)))+b2
	'''

	# 这个2048是原始论文里面定义的我们尝试一下是否对我们模型的性能有提升
	def __init__(self, input_dim, ff_dim=2048, dropout=0.1):
		super(FeedForward, self).__init__()
		self.w1 = nn.Linear(in_features=input_dim, out_features=ff_dim)
		self.w2 = nn.Linear(in_features=ff_dim, out_features=input_dim)
		self.lay_norm = nn.LayerNorm(input_dim,eps = 1e-6)
		self.relu = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, x):
		inter = self.dropout1(self.relu(self.w1((self.lay_norm(x)))))
		outer = self.dropout2(self.w2(inter))
		return outer


# 我们只使用encoder不需要Generator
class Generator(nn.Module):
	# input ：256
	# 这一层不需要 激活
	# 也不需要标准化
	def __init__(self, input_dim):
		super(Generator, self).__init__()
		self.linear = nn.Linear(input_dim, input_dim)

		# self.norm = nn.LayerNorm(input_dim)

	def forward(self, x):
		return F.log_softmax(self.linear(x), dim=-1)


# ( B , Features , N_P) --> ( 2 , 256 , 512 )
class MultiHead_SelfAttention(nn.Module):
	def __init__(self, input_dim, head_num):
		super(MultiHead_SelfAttention, self).__init__()
		self.head_num = head_num
		self.input_dim = input_dim
		assert input_dim % head_num == 0
		self.sub_dim = input_dim // head_num
		# 先是来自同源的 QKV 然后再做自注意力
		self.query = nn.Linear(self.input_dim, self.input_dim)
		self.key = nn.Linear(self.input_dim, self.input_dim)
		self.value = nn.Linear(self.input_dim, self.input_dim)

		self.linear_out = nn.Linear(self.input_dim, self.input_dim)
		self.dropout = nn.Dropout(0.1)
		self.softmax = nn.Softmax(-1)

		self.attn = None

		self._reset_parameters()

	def _reset_parameters(self):
		nn.init.xavier_uniform_(self.query.weight)
		nn.init.xavier_uniform_(self.key.weight)
		nn.init.xavier_uniform_(self.value.weight)
	# ( batch_size , Features , num_points) --> ( 2 , 256 , 512 ) --> (2,512,256)
	# 第二次修改后 我们直接在输入的时候就把特征维度放到最后面
	def forward(self, x):  # x: (2,256,512) --> transpose       最后需要把头数放到序列大小前面
		# Q = self.query(torch.transpose(x, 1, 2)).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1,2)
		# K = self.key(torch.transpose(x, 1, 2)).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1,2)
		# V = self.value(torch.transpose(x, 1, 2)).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1,2)

		# 形状为 （B , 512,256） --> (b , 512, 8 ,32)  -->transpose ( B , head_num , 512 , 32)
		Q = self.query(x).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1, 2)
		K = self.key(x).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1, 2)
		V = self.value(x).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1, 2)
		# head_num = 8
		# 输入形状为（B , head_num , 512 , 32 ）
		output,self.attn = self.multihead_self_att(Q, K, V,self.dropout)
		attention = self.linear_out(output)
		# 进行转置于原始形状对应
		'''
		这里我们不再在transformer里面转置，而是到最后再进行转置
		# （B, 256, 512）
		# output = attention.transpose(1,2)
		'''

		return attention

	def multihead_self_att(self, Q, K, V,dropout = None):
		# 输入形状为（B , head_num , 512 , 32 ）
		batch_size, head_num, point_num, sub_features = Q.size()
		# weighted_values = []
		# (B ,head_num , 512 , 512 )
		# score = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(Q.size(-1)))
		score = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(sub_features))

		# softmax
		atten_weights = dropout(F.softmax(score, -1))

		# 注意力得分矩阵与V相乘
		# （B, head_num, 512, 32 ）
		weighted_atten = torch.matmul(atten_weights, V)
		# （B, head_num, 512, 32 ） transpose --> （B, 512, head_num, 32 ） --> （B, 512, 256 ）
		output = weighted_atten.transpose(1, 2).contiguous().view(batch_size, -1, head_num * sub_features).contiguous()

		'''
		这里面是没有用matmul的结果
		for i in range(head_num):
			# 任何一个sub_Q,K,V的形状都是B,512,32
			sub_Q = Q[:, [i], ...].squeeze(1)       #第一次错误是因为我的squeeze的参数设置为0 这样一来的话，当我们的batchsiez为1的时候也会被压缩
			sub_K = K[:, [i], ...].squeeze(1)
			sub_V = V[:, [i], ...].squeeze(1)

			# 每个头的自注意力查询 然后进行缩放
			# (B,512,32) mm (B , 32 , 512)
			score = torch.bmm(sub_Q, torch.transpose(sub_K, 1, 2)) / torch.sqrt(torch.tensor(sub_Q.size(-1)))
			# (B , 512 , 512)
			attention_weights = self.softmax(score)

			# (B , 512 , 512) mm (B , 512 , 32)
			weighted_value = torch.bmm(attention_weights, sub_V)
			weighted_values.append(weighted_value)
		'''

		# torch.tensor(weighted_values)
		# # 堆叠到一起 的形状 (B , 512, 256)
		# atten_value = torch.stack(weighted_values, dim=2).view(batch_size, point_num, -1)
		# 这里还需要把最后得到的注意力特征转置 ， 于原始的特征在第二维对应上

		return output , weighted_atten


# 输入是一个批次的点云
class Transformer_Encoder(nn.Module):
	def __init__(self, input_dim, head_num,ff_dim=2048):
		super(Transformer_Encoder, self).__init__()
		# 1.自注意力
		self.multi_head = MultiHead_SelfAttention(input_dim=input_dim, head_num=head_num)
		# 2.残差边和归一化
		self.layer_norm1 = Lyer_norm(input_dim=input_dim)
		# 3.FFN（前馈传播）
		self.ffn = FeedForward(input_dim=input_dim,ff_dim=ff_dim)
		# 4.残差边和归一化
		self.layer_norm2 = Lyer_norm(input_dim=input_dim)

	def forward(self, x):
		atten = self.layer_norm1(x, self.multi_head)

		output = self.layer_norm2(atten, self.ffn)

		return output


def clone(model, n):
	return nn.ModuleList([copy.deepcopy(model) for _ in range(n)])




# 位置编码
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=0.1)

		# 计算位置编码矩阵
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)        #偶数维度的编码
		pe[:, 1::2] = torch.cos(position * div_term)        #奇数维度的编码
		pe = pe.unsqueeze(0).transpose(0, 1)

		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


class My_Transformer_features(nn.Module):
	'''
	搭出我们自己的积木
	我定义的有4个encoder层
	'''

	def __init__(self, input_dim, head_num,len, n_encoder=6):
		super(My_Transformer_features, self).__init__()
		# 自注意力
		self.input_dim = input_dim
		self.head_num = head_num
		self.n_decoder = n_encoder
		# self.encoder_layers = nn.ModuleList([Transformer_Encoder(input_dim,head_num) for i in range(n_encoder)])

		self.PE = PositionalEncoding(d_model = input_dim,max_len=len)
		self.encoder_layer = Transformer_Encoder(self.input_dim, self.head_num)
		# self.encoder_layers = nn.ModuleList([self.encoder_layer for _ in range(n_encoder)])
		self.encoder_layers = clone(self.encoder_layer,self.n_decoder)
		self.generator = Generator(input_dim)


	# ModelList需要运用for循环
	def forward(self, x):
		x = self.PE(x)
		inter = x
		for i in range(self.n_decoder):
			inter = self.encoder_layers[i](inter)

		output = self.generator(inter)
		# 输出tensor形状(B,512,256)
		return output


class My_Transformer_xyz(nn.Module):
	'''
	搭出我们自己的积木
	我定义的有4个encoder层
	'''

	def __init__(self, input_dim, head_num, ff_dim,n_encoder=6 ):
		super(My_Transformer_xyz, self).__init__()
		# 自注意力
		self.input_dim = input_dim
		self.head_num = head_num
		self.n_decoder = n_encoder
		self.ff_dim = ff_dim
		# self.encoder_layers = nn.ModuleList([Transformer_Encoder(input_dim,head_num) for i in range(n_encoder)])
		self.encoder_layer = Transformer_Encoder(self.input_dim, self.head_num,ff_dim=self.ff_dim)
		self.encoder_layers = nn.ModuleList([self.encoder_layer for _ in range(n_encoder)])
		self.generator = Generator(input_dim)


	# ModelList需要运用for循环
	def forward(self, x):
		inter = x
		for i in range(self.n_decoder):
			inter = self.encoder_layers[i](inter)

		output = self.generator(inter)
		# 输出tensor形状(B,512,256)
		return output

# if __name__ == '__main__':
# 	# start = time.process_time()
# 	# device = torch.device('cuda')
# 	# transformer = My_Transformer_features(64, 8, 2).to(device)
# 	#
# 	# x = torch.rand((1, 5000, 64)).to(device)
# 	# current_memory_allocated = torch.cuda.memory_allocated()
# 	# current_memory_cached = torch.cuda.memory_reserved()
# 	#
# 	# y = transformer(x)
# 	# max_memory_allocated = torch.cuda.max_memory_allocated()
# 	# max_memory_cached = torch.cuda.memory_reserved()
# 	# end = time.process_time()
# 	# cost = end - start
# 	# print(y.size())
# 	# print(cost)
# 	# print("Current memory allocated:", current_memory_allocated)
# 	# print("Current memory cached:", current_memory_cached)
# 	# print("Max memory allocated:", max_memory_allocated)
# 	# print("Max memory cached:", max_memory_cached)
# 	x = torch.rand(5000,2,8)
# 	encoding = PositionalEncoding(8)
# 	pos_x = encoding(x)
# 	print(x.shape)
