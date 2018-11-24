import torch
import torchvision
import sys
import os
from transform import transform_training, transform_testing, transform_training_dogs,transform_testing_dogs
import config as cf
from sklearn.datasets import load_files 

DIR = "data\\dog"
train = "train"
test = "test"
def dataset(dataset_name):

    if (dataset_name == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
    
    elif (dataset_name == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 100
        inputs = 3
    
    elif (dataset_name == 'mnist'):
        print("| Preparing MNIST dataset...")
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    
    elif (dataset_name == 'fashionmnist'):
        print("| Preparing FASHIONMNIST dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    elif (dataset_name == 'stl10'):
        print("| Preparing STL10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_training())
        testset = torchvision.datasets.STL10(root='./data',  split='test', download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
        
    elif (dataset_name == 'dog'):
        print("| Preparing Dog Breed dataset...")
        sys.stdout.write("| ")
        
        #root=os.path.join(DIR, dataset_name)
#        trainset=torchvision.datasets.ImageFolder(root=os.path.join(DIR, 'train'), transform=transform_training_dogs())
#        testset=torchvision.datasets.ImageFolder(root=os.path.join(DIR, 'test'), transform=transform_testing_dogs())
        
        root = os.path.join(DIR, train)
        trainset = torchvision.datasets.ImageFolder(root, transform=transform_training_dogs())
        
        root = os.path.join(DIR, test)
        testset = torchvision.datasets.ImageFolder(root, transform=transform_testing_dogs())
        outputs=120
        inputs=3
             
            
            
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader, outputs, inputs

