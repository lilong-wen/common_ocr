import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class rnn_decoder(nn.Module):

    def __init__(self, hidden_size,  output_size, dropout_p = 0.1):


        super(rnn_decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_p)

        self.embedding = nn.Embedding(self.output_size, 256)
        self.gru1 = nn.GRUCell(256, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 256)

        self.gru2 = nn.GRUCell(1024, self.hidden_size)
        self.out = nn.Linear(128, self.output_size)
        self.emb1 = nn.Linear(256, 128)
        self.emb2 = nn.Linear(256, 128)

        self.conv1 = nn.Conv2d(1, 1, kernel_size = 3, stride = 1, padding = 1)
        self.conv_et = nn.Conv2d(1, 1, kernel_size = 3, stride = 1, padding = 1)
        self.conv_tan = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)

        self.fc2 = nn.Linear(self.hidden_size, 128)

        self.ua = nn.Linear(1024, 256)
        self.uf = nn.Linear(1, 256)
        self.v = nn.Linear(256, 1)
        self.wc = nn.Linear(1024, 128)

        self.bn = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, input_y, input_hidden, img_feature, \
                feature_w, attention_sum, alpha_t, \
                feature_h, batch_size, h_mask, w_mask, gpu):

        '''
        input_y:  s'_t = GRU_1(y_t-1, s_t-1) : {y_t-1}
        input_hidden: s_t = GRU_2(s'_t, c_t) : {s_t}
        img_feature: image featere : A
        feature_w: feature w
        attention_sum: c_t
        alpha_t: c_t before MLP
        feature_h: feature h
        batch_size: batch size
        h_mask:
        w_mask:
        gpu:
        '''

        '''
        s'_t = GRU1(y_t-1, s_t)
        ---------------------------
        F = Q * sum(k_l) {t-1, ... L}
        e_ti = V_att * tanh(U_s * s'_t + U_a * a_i + U_f * f_i)
        a_ti = exp(e_ti) / sum(e_ti)  {1,...L}
        c_t = sum(k_ti * a_ti)
        ---------------------------
        s_t = GRU2(c_t, s'_t)
        '''

        # (b, h, w)
        et_mask = torch.zeros(batch_size,feature_h, feature_w).to(device)

        # set padding area to zero and non-padding area to 1
        for i in range(batch_size):
            et_mask[i][: h_mask[i], : w_mask[i]] = 1

        # (b, 1, h, w)
        et_mask_4 = et_mask.unsqueeze(1)
        ## _______________________________________________________________
        ## st_1 = GRU(y_t-1, st)

        # input_y: (b)   input_y_embedded: (b, 256)
        input_y_embedded = self.embedding(input_y)
        input_y_embedded = self.dropout(input_y_embedded)
        # input_hidden: (b, 1, hidden_size)
        input_hidden = input_hidden.squeeze(1)

        '''
        GRU  input: x_t , h_t-1
             output: h_t
        rt ​= σ(W_r * x_t​ + b_r​ + W_r​ * h_t−1 ​+ b_r​)
        zt​ = σ(W_z * x_t ​+ b_z​ + W_z​ * h_t−1​ + b_z​)
        nt​ = tanh(W_n​ * x_t​ + b_n​ + r_t ​∗ (W_n * ​h_t−1 ​+ b_n​)
        ht​ = (1−zt​) ∗ nt​ + zt ​∗ h_t−1​​
        '''

        # input_y_embedded:(b, 256)  input_hidden(b, hidden_size)
        st_1 = self.gru1(input_y_embedded, input_hidden)
        st_1 = self.fc1(st_1)
        st_1 = st_1.unsqueeze(1).unsqueeze(2)

        ## _____________________________________________________________
        ## F = Q * sum(k_0)
        #img_feature_trans = img_feature.permute(0, 3, 1, 2)

        # the img_feature originally (b,c,h,w), changing to (b, h, w, c)
        # channel last
        img_feature_trans = img_feature.transpose(1, 2).transpose(2, 3)

        # size(b,1,y_len,c)
        alpha_t = self.conv1(alpha_t)
        attention_sum += alpha_t

        ## _____________________________________________________________
        ## e_t = V_att * tanh(U_s * s'_t + U_a * a_i + U_f * f_i)
        #attention_sum_trans = attention_sum.permute(0, 3, 1, 2)
        # (6,1,6,15) --> (6, 6, 15, 1)
        # channel last
        attention_sum_trans = attention_sum.transpose(1, 2).transpose(2,3)

        img_feature1 = self.ua(img_feature_trans)
        attention_sum1 = self.uf(attention_sum_trans)

        print(st_1.size())
        print(img_feature1.size())
        print(attention_sum1.size())
        et = st_1 + img_feature1 + attention_sum1
        print(et.size())
        exit()
        #et_trans = et.permute(0, 3, 1, 2)
        et_trans = et.transpose(2, 3).transpose(1, 2)

        et_trans = self.conv_tan(et_trans)
        # concentrate on non-padding area
        et_trans = et_trans * et_mask_4

        et_trans = self.bn1(et_trans)
        et_trans = self.tanh(et_trans)
        #print(et_trans.size())
        #et_trans = et_trans.permute(0, 3, 1, 2)
        et_trans = et_trans.transpose(1,2).transpose(2,3)

        et = self.v(et_trans)
        et = et.squeeze(3)
        ## _____________________________________________________________
        ## a_ti = exp(e_ti) / sum(e_ti)
        et_div_all = torch.zeros(batch_size, 1, feature_h, feature_w).to(device)

        et_exp = torch.exp(et) * et_mask
        et_sum = torch.sum(et_exp, dim = 1)
        et_sum = torch.sum(et_sum, dim = 1)

        for i in range(batch_size):
            et_div = et_exp[i]/(et_sum[i]+1e-8)
            et_div = et_div.unsqueeze(0)
            et_div_all[i] = et_div

        ## _____________________________________________________________
        ## c_t = sum(k_t * a_t)
        ct = et_div_all * img_feature
        ct = ct.sum(dim = 2)
        ct = ct.sum(dim = 2)

        ## _____________________________________________________________
        ## s_t = GRU(c_t, s'_t)
        st = self.gru2(ct, st_1)
        hidden_next = st.view(batch_size, 1, self.hidden_size)

        # compute the output (batch,128)
        hidden2 = self.fc2(st)
        embedded2 = self.emb2(input_y_embedded)
        ct2 = self.wc(ct)

        ## _____________________________________________________________
        ## y_t = sum(all_previous input and output)
        et_next =  hidden2 + embedded2 + ct2
        et_next_drop = self.dropout(et_next)
        output = F.log_softmax(self.out(et_next_drop), dim = 1)

        output = output.unsqueeze(1)

        return output, hidden_next, et_div_all, attention_sum


    def initHidden(self, batch_size):

        result = Variable(torch.randn(batch_size, 1, self.hidden_size)).to(device)

        return result
