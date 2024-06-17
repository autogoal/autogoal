import multiprocessing
import psutil
import pynvml
import time
import json

def dummy_child_process():
    """Dummy child process that runs for 10 seconds."""
    print("Child process started")
    time.sleep(10)
    print("Child process ended")

def monitor_resources(process, interval=0.5):
    """Monitor CPU, RAM, and GPU usage at specified intervals."""
    pynvml.nvmlInit()
    stats = []
    
    cpu_percent_initialized = False
    
    while process.is_alive():
        start_time = time.time()
        
        # Initialize psutil.cpu_percent to start measuring
        if not cpu_percent_initialized:
            psutil.cpu_percent()
            cpu_percent_initialized = True

        # CPU and RAM usage (non-blocking)
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB

        # GPU usage
        gpu_stats = []
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)

            gpu_stats.append({
                "device_id": i,
                "gpu_utilization": util_rate.gpu,
                "memory_utilization": util_rate.memory,
                "memory_used": mem_info.used / (1024 ** 3),  # GB
            })

        # Aggregate data
        stats.append({
            "timestamp": time.time(),
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "gpu_stats": gpu_stats,
        })
        
        # Calculate how much time the monitoring took
        end_time = time.time()
        monitoring_duration = end_time - start_time
        
        # Sleep for the remainder of the interval, if applicable
        if monitoring_duration < interval:
            time.sleep(interval - monitoring_duration)

    pynvml.nvmlShutdown()
    return stats

def main():
    # Start child process
    process = multiprocessing.Process(target=dummy_child_process)
    process.start()

    # Monitor resources
    stats = monitor_resources(process)

    # Save stats to JSON
    with open("resource_usage.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("Monitoring finished and data has been saved to resource_usage.json")

if __name__ == "__main__":
    main()
