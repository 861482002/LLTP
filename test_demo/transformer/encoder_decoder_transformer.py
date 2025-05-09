# -*- codeing = utf-8 -*-
# @Time : 2023/6/12 14:10
# @Author : 张庭恺
# @File : encoder_decoder_transformer.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
	def __init__(self, input_size, head_count):
		super(SelfAttention, self).__init__()
		self.input_size = input_size
		self.head_count = head_count

		self.query_projection = nn.Linear(input_size, input_size)
		self.key_projection = nn.Linear(input_size, input_size)
		self.value_projection = nn.Linear(input_size, input_size)
		self.output_projection = nn.Linear(input_size, input_size)

	def forward(self, x):
		batch_size, seq_len, _ = x.size()

		queries = self.query_projection(x)
		keys = self.key_projection(x)
		values = self.value_projection(x)

		queries = queries.view(batch_size * self.head_count, seq_len, -1)
		keys = keys.view(batch_size * self.head_count, seq_len, -1)
		values = values.view(batch_size * self.head_count, seq_len, -1)

		attention_weights = torch.bmm(queries, keys.transpose(1, 2))
		attention_weights = F.softmax(attention_weights, dim=2)

		attention_output = torch.bmm(attention_weights, values)
		attention_output = attention_output.view(batch_size, seq_len, -1)
		attention_output = self.output_projection(attention_output)

		return attention_output

# 就是一个非线性激活
class FeedForward(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(FeedForward, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size

		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, input_size)

	def forward(self, x):
		hidden = F.relu(self.fc1(x))
		output = self.fc2(hidden)
		return output


class TransformerEncoderLayer(nn.Module):
	def __init__(self, input_size, head_count, hidden_size):
		super(TransformerEncoderLayer, self).__init__()
		self.input_size = input_size
		self.head_count = head_count
		self.hidden_size = hidden_size

		self.self_attention = SelfAttention(input_size, head_count)
		self.feed_forward = FeedForward(input_size, hidden_size)

	def forward(self, x):
		attention_output = self.self_attention(x)
		residual1 = attention_output + x

		feed_forward_output = self.feed_forward(residual1)
		output = feed_forward_output + residual1

		return output


class TransformerEncoder(nn.Module):
	def __init__(self, input_size, head_count, hidden_size, num_layers):
		super(TransformerEncoder, self).__init__()
		self.input_size = input_size
		self.head_count = head_count
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.layers = nn.ModuleList([
			TransformerEncoderLayer(input_size, head_count, hidden_size)
			for _ in range(num_layers)
		])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x


class TransformerDecoderLayer(nn.Module):
	def __init__(self, input_size, head_count, hidden_size):
		super(TransformerDecoderLayer, self).__init__()
		self.input_size = input_size
		self.head_count = head_count
		self.hidden_size = hidden_size

		self.self_attention = SelfAttention(input_size, head_count)
		self.encoder_attention = SelfAttention(input_size, head_count)
		self.feed_forward = FeedForward(input_size, hidden_size)

	def forward(self, x, encoder_output):
		self_attention_output = self.self_attention(x)
		residual1 = self_attention_output + x

		encoder_attention_output = self.encoder_attention(encoder_output)
		residual2 = encoder_attention_output + residual1

		feed_forward_output = self.feed_forward(residual2)
		output = feed_forward_output + residual2

		return output


class TransformerDecoder(nn.Module):
	def __init__(self, input_size, head_count, hidden_size, num_layers):
		super(TransformerDecoder, self).__init__()
		self.input_size = input_size
		self.head_count = head_count
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.layers = nn.ModuleList([
			TransformerDecoderLayer(input_size, head_count, hidden_size)
			for _ in range(num_layers)
		])

	def forward(self, x, encoder_output):
		for layer in self.layers:
			x = layer(x, encoder_output)
		return x


class Transformer(nn.Module):
	def __init__(self, input_size, output_size, head_count, hidden_size, enc_layers, dec_layers):
		super(Transformer, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.head_count = head_count
		self.hidden_size = hidden_size
		self.enc_layers = enc_layers
		self.dec_layers = dec_layers

		self.encoder = TransformerEncoder(input_size, head_count, hidden_size, enc_layers)
		self.decoder = TransformerDecoder(input_size, head_count, hidden_size, dec_layers)
		self.output_projection = nn.Linear(input_size, output_size)

	def forward(self, input_seq, target_seq):
		encoder_output = self.encoder(input_seq)
		decoder_output = self.decoder(target_seq, encoder_output)
		output = self.output_projection(decoder_output)
		return output


# 测试Transformer编码器-解码器模型
input_size = 256
output_size = 128
head_count = 8
hidden_size = 512
enc_layers = 6
dec_layers = 6

seq_len = 10
batch_size = 32

input_tensor = torch.randn(batch_size, seq_len, input_size)
target_tensor = torch.randn(batch_size, seq_len, output_size)

transformer = Transformer(input_size, output_size, head_count, hidden_size, enc_layers, dec_layers)
output_tensor = transformer(input_tensor, target_tensor)

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)

