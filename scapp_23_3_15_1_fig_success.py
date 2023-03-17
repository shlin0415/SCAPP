#!/usr/bin/env python
# coding: utf-8

# In[1]:


with open('SCAPP_0_Copy1.py','r') as f:
    exec(f.read())


# In[2]:


filepaths = ['/public3/labmember/zhengdh/data/rice/PRJNA767589/SRR16127058/counts.tsv.gz',
 '/public3/labmember/zhengdh/data/rice/PRJNA767589/SRR16127059/counts.tsv.gz',
 '/public3/labmember/zhengdh/data/rice/PRJNA767589/SRR16127057/counts.tsv.gz',
 '/public3/labmember/zhengdh/data/rice/PRJNA767589/SRR16127055/counts.tsv.gz',
 '/public3/labmember/zhengdh/data/rice/PRJNA767589/SRR16127056/counts.tsv.gz']


# In[3]:


scapp = SCAPP_0()
scapp.to_annotation_0(filepaths = filepaths)


# In[4]:


del scapp
print("Memory collecting...")
gc.collect()
print("Memory information: ")
info = psutil.virtual_memory()
print("Used: ")
print(psutil.Process(os.getpid()).memory_info().rss)
print("Total: ")
print(info.total)
print("Used (%): ")
print(info.percent)


# In[5]:


with open('SCAPP_0_Copy1.py','r') as f:
    exec(f.read())


# In[6]:


filepath =  '/public3/labmember/zhengdh/data/rice/PRJNA767589/SRR16127056/counts.tsv.gz'
scapp = SCAPP_0()
scapp.to_annotation_0(filepath = filepath)


# In[ ]:





# In[7]:


del scapp
print("Memory collecting...")
gc.collect()
print("Memory information: ")
info = psutil.virtual_memory()
print("Used: ")
print(psutil.Process(os.getpid()).memory_info().rss)
print("Total: ")
print(info.total)
print("Used (%): ")
print(info.percent)


# In[8]:


with open('SCAPP_0_Copy1.py','r') as f:
    exec(f.read())


# In[10]:


filepath =  '/public3/labmember/zhengdh/data/rice/PRJNA767589/SRR16127056/counts.tsv.gz'
scapp = SCAPP_0()
scapp.to_annotation_0(dpi=600,filepath = filepath)


# In[ ]:





# In[ ]:





# In[ ]:




