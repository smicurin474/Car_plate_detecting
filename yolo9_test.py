from ultralytics import YOLO
import configparser
from pathlib import Path

def test(config):
	model = YOLO(config["model_path"]) 
	model_path = Path(config["model_path"])
	metrics = model.val(data=config["data_config"],
						project=str(model_path.parent / ("val_" + model_path.stem)),
						split=config["split"],
						imgsz=config.getint("imgsz"),
						batch=config.getint("batch"),
						conf=config.getfloat("conf"),
						iou=config.getfloat("iou"),
						half=config.getboolean("half"),
						device=list(config["device"].split(",")),
						plots=config.getboolean("plots"))

if __name__ == '__main__':
    conf = configparser.ConfigParser()
    conf.read("Config/Config.ini")
    test(conf["Test"])
