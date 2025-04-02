import os
import resource
import psutil
import time


def monitor_system():
    while True:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_usage}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        print(f"Memory Usage: {memory.percent}%")
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        print(f"Disk Usage: {disk_usage.percent}%")
        
        # Network usage
        network = psutil.net_io_counters()
        print(f"Bytes Sent: {network.bytes_sent}, Bytes Received: {network.bytes_recv}")
        
        # Adding a separator for readability
        print("-" * 50)
        
        # Wait for a few seconds before the next update
        time.sleep(5)
if __name__ == '__main__':
    monitor_system()