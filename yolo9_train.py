import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
import torch
torch.cuda.empty_cache()

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.cuda.empty_cache()
from ultralytics import YOLO
import configparser

# Установка детерминированного поведения CUDA
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train(config):
	model = YOLO(config["name_model"]) 

	results = model.train(data=config["data_config"], 
						  pretrained=config.getboolean("pretrained"),
						  cos_lr=config.getboolean("cos_lr"),
						  project=config["project"],
						  epochs=config.getint("epochs"), 
						  batch=config.getint("batch"), 
						  imgsz=config.getint("imgsz"), 
						  device=list(config["device"].split(",")),
						  verbose=config.getboolean("verbose"), 
						  name=config["name"], 
						  optimizer=config["optimizer"],
						  close_mosaic=config.getint("close_mosaic"), 
						  single_cls=config.getboolean("single_cls"),
						  amp=config.getboolean("amp"), 
						  dropout=config.getfloat("dropout"),
						  lr0=config.getfloat("lr0"),
						  lrf=config.getfloat("lrf"),
						  warmup_epochs=config.getint("warmup_epochs"), 
						  augment=config.getboolean("augment"),
						  plots=config.getboolean("plots"),
						  save_period=config.getint("save_period"), 
						  workers=config.getint("workers"))
		                  
	path = model.export(format="onnx") 

if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read("Config/Config.ini")
    train(conf["Train"])
