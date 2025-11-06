import psutil
import time
import GPUtil  # pip install gputil


def log_system_stats(epoch):
    """Ghi lại sử dụng tài nguyên hệ thống: CPU, RAM."""
    # Lấy thông tin sử dụng CPU (%)
    cpu_percent = psutil.cpu_percent(interval=1)

    # Lấy thông tin sử dụng RAM
    ram_info = psutil.virtual_memory()
    ram_total = ram_info.total / (1024 ** 3)  # Chuyển đổi sang GB
    ram_used = ram_info.used / (1024 ** 3)  # Chuyển đổi sang GB
    ram_percent = ram_info.percent

    # Ghi thông tin hệ thống
    log_message = (f"Epoch {epoch}: CPU usage: {cpu_percent}%, "
                   f"RAM used: {ram_used:.2f}GB/{ram_total:.2f}GB ({ram_percent}% used)")

    # In ra console hoặc ghi vào log file
    print(log_message)


def log_gpu_temperature(epoch):
    """Ghi lại nhiệt độ của các GPU đang hoạt động."""
    # Lấy danh sách các GPU đang hoạt động
    gpus = GPUtil.getGPUs()

    # Lưu nhiệt độ của từng GPU
    gpu_temperatures = []

    for gpu in gpus:
        gpu_id = gpu.id
        temperature = gpu.temperature
        gpu_temperatures.append((gpu_id, temperature))

        # Ghi thông tin nhiệt độ
        log_message = f"Epoch {epoch}: GPU {gpu_id} temperature: {temperature}°C"
        print(log_message)

    # Trả về danh sách nhiệt độ để sử dụng tiếp trong chương trình
    return gpu_temperatures


def check_gpu_status(epoch):
    log_system_stats(epoch)
    temperatures = log_gpu_temperature(epoch)
    # If GPU temperature over 85°C, pause training
    overheat = False
    for gpu_id, temperature in temperatures:
        if temperature >= 85:
            print(f"GPU {gpu_id} temperature too high ({temperature}°C), pause training to protect hardware.")
            overheat = True
    if overheat:
        # Pause training until temperature decrease
        while True:
            print("Waiting for GPU temperature below 75°C...")
            time.sleep(60)  # wait 60 seconds before check again
            temperatures = log_gpu_temperature(epoch)
            all_cool = True
            for gpu_id, temperature in temperatures:
                if temperature >= 75:
                    all_cool = False
                    break
            if all_cool:
                print("GPU temperature is safe, continue training.")
                break
