import torch.nn as nn
import torch

class ClassicRNN(nn.Module):
  """
  Classical RNN
  """
  def __init__(self, device, input_size, hidden_size, num_layers, dropout_ratio=0.0):
    super(ClassicRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_ratio)
    self.W = nn.Parameter(torch.randn([hidden_size, input_size]).type(torch.float32))
    self.b = nn.Parameter(torch.randn([input_size]).type(torch.float32))
    self.Softmax = nn.Softmax(dim=1)

  def forward(self, x):
    # hidden state tensor should be the shape (num_layer, batch_size, hidden_size)
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
    # nn.RNN input should be the shape (sequence_size, batch_size, input_size)
    # Because those datasets are only have 1 input feature, just unsqueeze the input vector
    x = x.unsqueeze(-1).transpose(0, 1)
    outputs, _ = self.rnn(x, h0)
    outputs = outputs[-1]  # 최종 예측 Hidden Layer
    y = torch.mm(outputs, self.W) + self.b
    return y
