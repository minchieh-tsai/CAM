
from sample2 import initialize_model
import torch
from PIL import Image
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

# # example
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)

# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']



# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 1

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

## load trained model
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
checkpoint = torch.load('./model_v2/epoch_14.pth')
model_ft.load_state_dict(checkpoint['state_dict'])

model_ft.eval()

## load images

img_path = "../data/hymenoptera_data/test_LGC"
# img = Image.open(img_path)

transform =transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
predict = datasets.ImageFolder(root=img_path ,transform=transform)
predict_img  = torch.utils.data.DataLoader(predict, 
                                           batch_size=batch_size, 
                                           num_workers=0)

output_feature = []
def hook(module, inputdata, output):
    # print(output.data)
    output_feature.append(output.data)
    # plt.figure()
    # plt.imshow(output.data[0,0,:,:])

handle = model_ft._modules['features'].register_forward_hook(hook)
weights = (model_ft._modules['classifier'].weight).data

img_list =[]
predict_list =[]
for inputs, labels in predict_img:
    img_list.append(inputs)
    # inputs = inputs.to('cuda')
    output = model_ft(inputs)
    _, preds = torch.max(output, 1)
    predict_list.append(preds)
    
# 用完hook后删除
handle.remove()

mean_cam_class2_list=[]
for i, pre in enumerate(predict_list):

    if pre == 1:
        # 將weight 和 feature相乘
        # n = 0
        cam = []
        cam_class2_list = []

        
        
        weights_np = weights.numpy()
        feature = np.reshape(output_feature[i].numpy(),(1024,7,7))
        # cam_class1  = ((weights_np[0,:].reshape(1024,1,1)) * feature ).sum(0)
        cam_class2  = ((weights_np[1,:].reshape(1024,1,1)) * feature ).sum(0)
    
            
        # # 加速8倍的CAM(後來沒用)
        # var = (output_feature[0].numpy() * weights_np[1].reshape(1, 1024, 1, 1))
        # cam_class2 = var.sum(1)
        
    
        # # A2C類
        # plt.figure()
        # plt.imshow(cam_class1)
        
        # A4C類
        # plt.figure()
        # plt.imshow(cam_class2)
        
        # 找出7*7的平均值，並且將每個7*7圖片都放入cam_class2
    
        cam_class2_list.append(cam_class2)
        mean_cam_class2 = np.mean(cam_class2) #一張CAM平均
        mean_cam_class2_list.append(mean_cam_class2)
        
        # 利用平均將高於平均的留下，低於平均的刪除
        total_mean = np.mean(mean_cam_class2_list)
        total_std = np.std(mean_cam_class2_list)
        # 正常4.9123397,std=2.3089697
        
        threshold = 4.9123397 - 2.3089697
        
        up_mean = np.zeros((7,7))
        up_mean[cam_class2 >= threshold] =255
        up_mean[cam_class2 < threshold] = 0
        
        # 畫出保留的部分
        # fig,axes=plt.subplots(2,2)
        # plt.figure()
        # plt.imshow(up_mean)
        
        
        # # 把7*7的圖改成224*224
        ic = cv2.resize(cam_class2, (224, 224), interpolation=cv2.INTER_CUBIC)
        # cam.append(ic)
        
        # 因為input放入transform轉換，所以再轉回來
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255])
        
        inv_tensor = inv_normalize(img_list[i].squeeze(0))
        
        pred_img = (inv_tensor.numpy()[0,:,:])*255
        
        # 利用平均將高於平均的留下，低於平均的刪除(經過內插的)
        intor_mean = np.zeros((224,224))
        intor_mean[ic >= threshold] =255
        intor_mean[ic < threshold] =0
        
        # 畫出彩色重疊的圖片
        # plt.figure()
        # plt.imshow(ic)
        # plt.imshow(pred_img, cmap='gray')
        # h = plt.imshow(ic,cmap='jet',alpha=0.4)
        # plt.clim(vmin=-25,vmax=25)
        # plt.colorbar(h)
        
    
        # 畫出四格圖片
        plt.figure()
    
        ax1 = plt.subplot(221)
        ax1.imshow(pred_img,cmap='gray')
        plt.title(f"input_{i}")
        
        # ax2 = plt.subplot(222)
        # ax2.imshow(cam_class2)
        # plt.title("CAM")
        
        ax3 = plt.subplot(223)
        ax3.imshow(up_mean,cmap='gray')
        plt.title("Threshold")
        
        ax4 = plt.subplot(224)
        ax4.imshow(intor_mean,cmap='gray')
        plt.title("Threshold_resize")
        
        #調整間距大小 
        plt.subplots_adjust(left=0.125,
                            bottom=0.1, 
                            right=0.75, 
                            top=0.95, 
                            wspace=0.05, 
                            hspace=0.4)
    
    
        # plt.show()
    else:
        print(f"{i}")
# # 看目前上下左右的差距
# plt.subplot_tool()
# plt.show()












