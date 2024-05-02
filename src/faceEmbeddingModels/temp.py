

import pickle
from keras.models import load_model

# Load the saved model
model = load_model('my_model.h5')

# Load the training history
with open('my_model.h5_history.pkl', 'rb') as f:
    history = pickle.load(f)

print(f"model :{model}")


# import matplotlib.pyplot as plt
#
# # Plot training and validation accuracy
# plt.plot(history['acc'], label='Training Accuracy')
# plt.plot(history['val_acc'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# import tensorflow
# print("TENSORFLOW")
# print(tensorflow.__version__)
#
# import keras
# print("KERAS")
# print(keras.__version__)



# # # # Plot training and validation loss
# plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#
# plt.plot(history['lr'], label='Learning Rate')
# plt.title('Learning Rate Schedule')
# plt.xlabel('Epochs')
# plt.ylabel('Learning Rate')
# plt.legend()
# plt.show()


import psutil
import platform

def get_system_resources():
    # Get CPU information
    cpu_info = f"CPU: {platform.processor()}"
    cpu_cores = f"Cores: {psutil.cpu_count(logical=True)}"
    cpu_usage = f"CPU Usage: {psutil.cpu_percent()}%"

    # Get memory (RAM) information
    memory = psutil.virtual_memory()
    total_memory = f"Total Memory: {round(memory.total / (1024 ** 3), 2)} GB"
    available_memory = f"Available Memory: {round(memory.available / (1024 ** 3), 2)} GB"
    memory_usage = f"Memory Usage: {memory.percent}%"

    # Get disk usage
    disk_usage = psutil.disk_usage('/')
    total_disk_space = f"Total Disk Space: {round(disk_usage.total / (1024 ** 3), 2)} GB"
    used_disk_space = f"Used Disk Space: {round(disk_usage.used / (1024 ** 3), 2)} GB"
    disk_usage_percent = f"Disk Usage: {disk_usage.percent}%"

    # Get operating system information
    os_info = f"OS: {platform.system()} {platform.release()}"

    # Combine all information
    system_info = "\n".join([cpu_info, cpu_cores, cpu_usage, total_memory, available_memory, memory_usage,
                             total_disk_space, used_disk_space, disk_usage_percent, os_info])
    return system_info

# Usage
print(get_system_resources())






