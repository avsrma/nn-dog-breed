#!/usr/bin/env python
# coding: utf-8

# In[83]:


from __future__ import print_function


# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[85]:


import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt


# In[86]:


use_cuda = torch.cuda.is_available()

if use_cuda:
    print("CUDA!")


# In[87]:


from dataset import dataset
#from AlexNet import AlexNet
import torchvision.models as models
from train_test import start_train_test


# In[88]:


trainloader, testloader, outputs, inputs = dataset('cifar10')
print ('Output classes: {}\nInput channels: {}'.format(outputs, inputs))


# trainloader, testloader, outputs, inputs = dataset('dog-breed')
# print ('Output classes: {}\nInput channels: {}'.format(outputs, inputs))

# In[89]:


trainloader, testloader, outputs, inputs = dataset('dog')
print ('Output classes: {}\nInput channels: {}'.format(outputs, inputs))


# In[90]:


net = AlexNet(num_classes = outputs, inputs=inputs)
file_name = 'alexnet-'


# In[91]:


net = models.resnet18(pretrained = False, num_classes = outputs)
file_name = 'resnet-'


# In[92]:


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# In[93]:


criterion = nn.CrossEntropyLoss()


# In[94]:


train_loss, test_loss = start_train_test(net, trainloader, testloader, criterion)


# In[ ]:


plt.plot(train_loss)
plt.ylabel('Train Loss')
plt.show()


# In[ ]:


plt.plot(test_loss)
plt.ylabel('Test Loss')
plt.show()


# In[ ]:




