import torch
import random


def my_train(target_length,
             attn_decoder1,
             output_highfeature,
             output_area,
             y,
             criterion,
             encoder_optimizer1,
             decoder_optimizer1,
             dense_input,
             h_mask,
             w_mask,
             gpu,
             decoder_input,
             decoder_hidden,
             attention_sum,
             decoder_attention,
             teacher_forcing_ratio,
             batch_size):

    loss = 0
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        False

    flag_z = [0]*batch_size

    if use_teacher_forcing:
        encoder_optimizer1.zero_grad()
        decoder_optimizer1.zero_grad()
        my_num = 0

        for di in range(target_length):
            decoder_output, \
                decoder_hidden, \
                decoder_attention, \
                attention_sum = attn_decoder1(decoder_input,
                                              decoder_hidden,
                                              output_highfeature,
                                              output_area,
                                              attention_sum,
                                              decoder_attention,
                                              dense_input,
                                              batch_size,
                                              h_mask,
                                              w_mask,
                                              gpu)

            y = y.unsqueeze(0)
            for i in range(batch_size):

                if int(y[0][i][di]) == 0:
                    flag_z[i] = flag_z[i]+1
                    if flag_z[i] > 1: # scape padding value
                        continue
                    else:
                        loss += criterion(decoder_output[i], y[:,i,di])
                else:
                    loss += criterion(decoder_output[i], y[:,i,di])

            # print(decoder_output[i])
            # print(8*"*")
            # print(y[:,i,di])
            if int(y[0][0][di]) == 0:
                break
            decoder_input = y[:,:,di]
            decoder_input = decoder_input.squeeze(0)
            y = y.squeeze(0)

        #print(loss)
        loss.backward()

        encoder_optimizer1.step()
        decoder_optimizer1.step()
        return loss.item()

    else:
        encoder_optimizer1.zero_grad()
        decoder_optimizer1.zero_grad()
        my_num = 0
        for di in range(target_length):
            decoder_output, \
                decoder_hidden, \
                decoder_attention,\
                attention_sum= attn_decoder1(decoder_input,
                                             decoder_hidden,
                                             output_highfeature,
                                             output_area,
                                             attention_sum,
                                             decoder_attention,
                                             dense_input,
                                             batch_size,
                                             h_mask,
                                             w_mask,
                                             gpu)
            #print(decoder_output.size()) 1*10*112
            #print(y.size())  1*37
            #topi (b,1)
            topv,topi = torch.max(decoder_output,2)
            decoder_input = topi
            decoder_input = decoder_input.view(batch_size)

            y = y.unsqueeze(0)
            #print(y_t)

            # 1*bs*17
            for k in range(batch_size):
                if int(y[0][k][di]) == 0:
                    flag_z[k] = flag_z[k]+1
                    if flag_z[k] > 1:
                        continue
                    else:
                        loss += criterion(decoder_output[k], y[:,k,di])
                else:
                    loss += criterion(decoder_output[k], y[:,k,di])

            y = y.squeeze(0)
            # if int(topi[0]) == 0:
            #     break
        loss.backward()
        encoder_optimizer1.step()
        decoder_optimizer1.step()
        return loss.item()
