datasets=['./train.pkl','./train_caption.txt']
valid_datasets=['./test.pkl', './test_caption.txt']
dictionaries=['./dictionary.txt']
batch_Imagesize=500000
valid_batch_Imagesize=500000
# batch_size for training and testing
batch_size=6
batch_size_t=6
# the max (label length/Image size) in training and testing
# you can change 'maxlen','maxImagesize' by the size of your GPU
maxlen=48
maxImagesize= 100000
# hidden_size in RNN
hidden_size = 256
# teacher_forcing_ratio
teacher_forcing_ratio = 1
# change the gpu id
gpu = [0,1]
# learning rate
lr_rate = 0.0001
# flag to remember when to change the learning rate
flag = 0
# exprate
exprate = 0
