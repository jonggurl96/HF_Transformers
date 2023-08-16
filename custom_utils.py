import gc
import torch
from pynvml import *


def clear_gpu():
	nvmlInit()
	device_count = nvmlDeviceGetCount()
	print_string_list = []
	
	for i in range(device_count):
		handle = nvmlDeviceGetHandleByIndex(i)
		info = nvmlDeviceGetMemoryInfo(handle)
		
		print_string_list.append({
			"gpu": f"GPU {torch.cuda.get_device_name(i)}",
			"memory_occupied": f"memory occupied: {info.used / (1024**3):.2f}GB"
		})

	gc.collect()
	torch.cuda.empty_cache()
	
	for i in range(device_count):
		print(f"\n{print_string_list[i]['gpu']}")
		print(print_string_list[i]["memory_occupied"])
		print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(i) / (1024**3):.2f}GB")
		print(f"Reserved GPU Memory: {torch.cuda.memory_reserved(i) / (1024**3):.2f}GB")


class CustomObject:
	
	def __init__(self):
		pass
	
	def init_wrapper(self):
		self.__dict__.clear()
		clear_gpu()
	
	def __del__(self):
		clear_gpu()
		
