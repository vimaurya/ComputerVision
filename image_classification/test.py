import torch
from helper_functions import predict_weather
from helper_functions import weatherClassificationModelv0 as wc0
from helper_functions import weatherClassificationModelv1 as wc1
from helper_functions import weatherClassificationModelv2 as wc2


modelv0 = wc0()
modelv0.load_state_dict(torch.load('../models/weather_classification_modelv0.pth'))

modelv1 = wc1()
modelv1.load_state_dict(torch.load('../models/weather_classification_modelv1.pth'))

modelv2 = wc2()
modelv2.load_state_dict(torch.load('../models/weather_classification_modelv2.pth'))


img_class = 'shine'

img_num = 66

print(f"modelv0 : {predict_weather(modelv0, img_class, img_num)}")
print(f"modelv1 : {predict_weather(modelv1, img_class, img_num)}")
print(f"modelv2 : {predict_weather(modelv2, img_class, img_num)}")


#%%
