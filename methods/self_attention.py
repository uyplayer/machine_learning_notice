# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2019/11/20 下午4:22
# Tool ：PyCharm

'''
https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/79739287
https://msd.misuland.com/pd/13340603045208861
'''


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size  # 另一种语言的词汇量
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):  # forward的参数是decoder的输入
        # decoder的input是另一种语言的词汇,要么是target,要么是上一个单元返回的output中概率最大的一个
        # 初始的hidden用的是encoder的最后一个hidden输出
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # 将embedded的256词向量和hidden的256词向量合在一起,变成512维向量
        # 再用线性全连接变成10维(最长句子词汇数),在算softmax,看
        attn_weight = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        # torch.cat用于粘贴,dim=1指dim1方向粘贴
        # torch.bmm是批矩阵乘操作,attention里将encoder的输出和attention权值相乘
        # bmm: (1,1,10)*(1,10,256),权重*向量,得到attention向量
        # unsqueeze用于插入一个维度(修改维度)
        attn_applied = torch.bmm(attn_weight.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weight

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)