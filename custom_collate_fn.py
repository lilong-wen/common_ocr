import torch

def collate_fn(batch):

    batch.sort(key = lambda x: len(x[1]), reverse = True)

    img_batch, label_batch = zip(*batch)

    max_width = 0
    max_height = 0
    max_label_len = 0

    for image_item in img_batch:
        size = image_item.size()

        if size[1] > max_height:
            max_height = size[1]
        if size[2] > max_width:
            max_width = size[2]

    for label_item in label_batch:

        if len(label_item) + 1 > max_label_len:
            max_label_len = len(label_item) + 1

    IS_FIRST_IMG = True
    for img_item in img_batch:

        img_item = img_item.float()
        image_size_h, image_size_w = img_item.size()[1:]

        image_mask = torch.ones(1,
                                image_size_h,
                                image_size_w).type(torch.FloatTensor)
        image_mask = image_mask * 255.0
        image_with_mask = torch.cat((img_item, image_mask), dim=0)
        padding_h = max_height - image_size_h
        padding_w = max_width - image_size_w

        #left,right,top,bottom
        padding_obj = torch.nn.ZeroPad2d((0, padding_w, 0, padding_h))
        padded_image_with_mask = padding_obj(image_with_mask)
        padded_image_with_mask = padded_image_with_mask.unsqueeze(0)

        if IS_FIRST_IMG:
            padded_batch = padded_image_with_mask
            IS_FIRST_IMG = False
        else:
            padded_batch = torch.cat((padded_batch, padded_image_with_mask),
                                     dim = 0)
    padded_batch = padded_batch / 255.0

    IS_FIRST_LAB = True
    for lab_item in label_batch:
        lab_item = lab_item.long()
        lab_item = lab_item.unsqueeze(0)
        lab_item_len = lab_item.size()[1]

        padding_obj = torch.nn.ZeroPad2d((0,
                                          max_label_len - lab_item_len,
                                          0,
                                          0))
        padded_label = padding_obj(lab_item)
        if IS_FIRST_LAB:
            padded_lab_batch = padded_label
            IS_FIRST_LAB = False
        else:
            padded_lab_batch = torch.cat((padded_lab_batch, padded_label),
                                         dim = 0)

    return padded_batch, padded_lab_batch
