from compute_loss import my_train
import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(encoder, attn_decoder, train_loader, \
          batch_size, hidden_size, \
          gpu, len_train, epoch,\
          criterion, decoder_input_init, decoder_hidden_init):

    teacher_forcing_ratio = 1
    lr_rate = 0.0001
    flag = 0
    exprate = 0

    encoder_optimizer = torch.optim.SGD(encoder.parameters(),
                                         lr=lr_rate,
                                         momentum=0.9)
    decoder_optimizer = torch.optim.SGD(attn_decoder.parameters(),
                                         lr=lr_rate,
                                         momentum=0.9)
    running_loss=0
    whole_loss = 0

    #encoder.train(mode=True)
    encoder.train(mode=False)
    attn_decoder.train(mode=True)

    for step,(x_batch,y_batch) in enumerate(train_loader):

        h_mask = []
        w_mask = []
        for x_img in x_batch:
            # img: x_img[0]  real pixels pad with zero
            # mask: x_img[1] all one pad with zero
            size_mask = x_img[1].size()
            first_row_sequence = str(x_img[1][0])
            first_column_sequence = str(x_img[1][:,0])
            real_img_w = first_row_sequence.count('1')
            real_img_h = first_column_sequence.count('1')
            w_mask.append(int(real_img_w / 16) + 1) # encoder decrease input size to 1/16
            h_mask.append(int(real_img_h / 16) + 1)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # size b 1024 h/16 w/16
        output_feature = encoder(x_batch)
        x_mean = []

        for feature_item in output_feature:
            x_mean.append(float(torch.mean(feature_item)))


        for i in range(batch_size):

            decoder_hidden_init[i] = decoder_hidden_init[i]*x_mean[i]
            decoder_hidden_init[i] = torch.tanh(decoder_hidden_init[i])

        #print(output_feature.size()) # 6 1024 6 15
        output_feature_size = output_feature.size()

        out_feature_h = output_feature_size[2]
        out_feature_w = output_feature_size[3]

        # current batch label length
        target_length = y_batch.size()[1]

        attention_sum_init = torch.zeros(batch_size,
                                         1,
                                         out_feature_h,
                                         out_feature_w).to(device)
        decoder_attention_init = torch.zeros(batch_size,
                                             1,
                                             out_feature_h,
                                             out_feature_w).to(device)

        running_loss += my_train(target_length,
                                 attn_decoder,
                                 output_feature,
                                 out_feature_w,
                                 y_batch,
                                 criterion,
                                 encoder_optimizer,
                                 decoder_optimizer,
                                 out_feature_h,
                                 h_mask,
                                 w_mask,
                                 gpu,
                                 decoder_input_init,
                                 decoder_hidden_init,
                                 attention_sum_init,
                                 decoder_attention_init,
                                 teacher_forcing_ratio,
                                 batch_size)


        if step % 20 == 19:
            pre = ((step+1)/len_train)*100*batch_size
            whole_loss += running_loss
            running_loss = running_loss/(batch_size*20)
            print(f'epoch is {epoch}, lr rate is {lr_rate:.5f}, \
            te is {teacher_forcing_ratio:.3f}, \
            batch_size is {batch_size}, \
            loading for {pre:.3f}%%, \
            running_loss is {running_loss:.5f}')

            running_loss = 0

    loss_avarage = whole_loss / len_train
    print(f"epoch is {epoch}, the whole loss is {loss_avarage}")
