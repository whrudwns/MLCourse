#data already exists
#incomplete  to be modified

'''
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

trans = transforms.Compose([
    transforms.Resize((32,64))
])

train_data = torchvision.datasets.ImageFolder(root="/home/j/MLCourse/final/chest_xray/train",transform = trans)


for num, value in enumerate(train_data):
    data, label = value
    #print(num, data, label)

    if(label == 0):
        data.save('/home/j/MLCourse/final/normal/%d_%d.jpeg'%(num, label)) #replace your directory address
    else:
        data.save("/home/j/MLCourse/final/pneumonia/%d_%d.jpeg"%(num, label)) #replace your directory address
'''


# Collect files into one directory after code execution
