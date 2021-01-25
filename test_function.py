import torch
import torch.nn as nn
from utils import cmp_result

def test(encoder, attn_decoder1, test_loader, opt, device):
    # this is the prediction and compute wer loss
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    whole_loss_t = 0

    encoder.eval()
    attn_decoder1.eval()
    print('begin testing')

    for step_t, (x_t, y_t) in enumerate(test_loader):
        x_real_high = x_t.size()[2]
        x_real_width = x_t.size()[3]
        if x_t.size()[0]<opt.batch_size:
            break
        print('testing for %.3f%%'%(step_t*100*opt.batch_size/opt.len_test_data),end='\r')
        h_mask_t = []
        w_mask_t = []
        for i in x_t:
            #h*w
            size_mask_t = i[1].size()
            s_w_t = str(i[1][0])
            s_h_t = str(i[1][:,1])
            w_t = s_w_t.count('1')
            h_t = s_h_t.count('1')
            h_comp_t = int(h_t/16)+1
            w_comp_t = int(w_t/16)+1
            h_mask_t.append(h_comp_t)
            w_mask_t.append(w_comp_t)

        x_t = x_t.to(device)
        y_t = y_t.to(device)
        output_highfeature_t = encoder(x_t)

        x_mean_t = torch.mean(output_highfeature_t)
        x_mean_t = float(x_mean_t)
        output_area_t1 = output_highfeature_t.size()
        output_area_t = output_area_t1[3]
        dense_input = output_area_t1[2]

        decoder_input_t = torch.LongTensor([220]*opt.batch_size)
        decoder_input_t = decoder_input_t.cuda()
        decoder_hidden_t = torch.randn(opt.batch_size, 1, opt.hidden_size).cuda()
        nn.init.xavier_uniform_(decoder_hidden_t)

        x_mean_t=[]
        for i in output_highfeature_t:
            x_mean_t.append(float(torch.mean(i)))
        # x_mean = torch.mean(output_highfeature)
        # x_mean = float(x_mean)
        for i in range(opt.batch_size):
            decoder_hidden_t[i] = decoder_hidden_t[i]*x_mean_t[i]
            decoder_hidden_t[i] = torch.tanh(decoder_hidden_t[i])

        prediction = torch.zeros(opt.batch_size,opt.maxlen)
        #label = torch.zeros(batch_size,opt.maxlen)
        prediction_sub = []
        label_sub = []
        decoder_attention_t = torch.zeros(opt.batch_size,1,dense_input,output_area_t).cuda()
        attention_sum_t = torch.zeros(opt.batch_size,1,dense_input,output_area_t).cuda()
        flag_z_t = [0]*opt.batch_size
        loss_t = 0
        m = torch.nn.ZeroPad2d((0,opt.maxlen-y_t.size()[1],0,0))
        y_t = m(y_t)
        for i in range(opt.maxlen):
            decoder_output, \
                decoder_hidden_t, \
                decoder_attention_t, \
                attention_sum_t = attn_decoder1(decoder_input_t,
                                                decoder_hidden_t,
                                                output_highfeature_t,
                                                output_area_t,
                                                attention_sum_t,
                                                decoder_attention_t,
                                                dense_input,
                                                opt.batch_size,
                                                h_mask_t,
                                                w_mask_t,
                                                opt.gpu)

            ### you can see the attention when testing

            # print('this is',i)
            # for i in range(batch_size):
            #     x_real = numpy.array(x_t[i][0].data.cpu())

            #     show = numpy.array(decoder_attention_t[i][0].data.cpu())
            #     show = imresize(show,(x_real_width,x_real_high))
            #     k_max = show.max()
            #     show = show/k_max

            #     show_x = x_real+show
            #     plt.imshow(show_x, interpolation='nearest', cmap='gray_r')
            #     plt.show()

            topv,topi = torch.max(decoder_output,2)
            # if torch.sum(y_t[0,:,i])==0:
            #     y_t = y_t.squeeze(0)
            #     break
            if torch.sum(topi)==0:
                break
            decoder_input_t = topi
            decoder_input_t = decoder_input_t.view(opt.batch_size)

            # prediction
            prediction[:,i] = decoder_input_t

        for i in range(opt.batch_size):
            for j in range(opt.maxlen):
                if int(prediction[i][j]) ==0:
                    break
                else:
                    prediction_sub.append(int(prediction[i][j]))
            if len(prediction_sub)<opt.maxlen:
                prediction_sub.append(0)

            for k in range(y_t.size()[1]):
                if int(y_t[i][k]) ==0:
                    break
                else:
                    label_sub.append(int(y_t[i][k]))
            label_sub.append(0)

            dist, llen = cmp_result(label_sub, prediction_sub)
            total_dist += dist
            total_label += llen
            total_line += 1
            if dist == 0:
                total_line_rec = total_line_rec+ 1

            label_sub = []
            prediction_sub = []

    print('total_line_rec is',total_line_rec)
    wer = float(total_dist) / total_label
    sacc = float(total_line_rec) / total_line
    print('wer is %.5f' % (wer))
    print('sacc is %.5f ' % (sacc))
    # print('whole loss is %.5f'%(whole_loss_t/925))
    # with open("training_data/wer_%.5f_pre_GN_te05_d02_all.txt" % (opt.lr_rate), "a") as f:
    #     f.write("%s\n" % (str(wer)))

    if (sacc > opt.exprate):
        opt.exprate = sacc
        print(opt.exprate)
        print("saving the model....")
        print('encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl' %(opt.lr_rate))
        torch.save(encoder.state_dict(), 'model/encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'%(opt.lr_rate))
        torch.save(attn_decoder1.state_dict(), 'model/attn_decoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'%(opt.lr_rate))
        print("done")
        op.flag = 0
    else:
        opt.flag = opt.flag+1
        print('the best is %f' % (opt.exprate))
        print('the loss is bigger than before,so do not save the model')

    if opt.flag == 10:
        opt.lr_rate = opt.lr_rate*0.1
        opt.flag = 0
