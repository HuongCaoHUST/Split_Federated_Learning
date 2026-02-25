import os
import json
import time
import csv
import subprocess
from datetime import datetime

# Ghi lại thời điểm bắt đầu
START_TIME = time.time()

def get_next_log_filename(base_name, ext="csv"):
    i = 1
    while os.path.exists(f"{base_name}_{i}.{ext}"):
        i += 1
    return f"{base_name}_{i}.{ext}"

DOCKER_LOG_FILE = get_next_log_filename("docker_stats")
GPU_PROC_LOG_FILE = get_next_log_filename("gpu_stats")

# --- PHẦN TÍCH HỢP: ÁNH XẠ PID ---

def get_container_pid_map():
    """Lấy danh sách ánh xạ PID -> Container ID"""
    pid_to_cid = {}
    try:
        # Lấy tất cả ID container đang chạy
        cids_res = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True, check=True)
        cids = [cid for cid in cids_res.stdout.strip().split('\n') if cid]
        
        for cid in cids:
            # Inspect từng container để lấy PID gốc trên Host
            inspect_res = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True, check=True)
            data = json.loads(inspect_res.stdout)
            pid = data[0]['State']['Pid']
            if pid and pid != 0:
                pid_to_cid[str(pid)] = cid # Lưu dưới dạng string để dễ so sánh
    except Exception as e:
        print(f"Lỗi khi lấy PID mapping: {e}")
    return pid_to_cid

# Khởi tạo bảng ánh xạ một lần duy nhất khi chạy script
PID_MAP = get_container_pid_map()

# --- CÁC HÀM LOG DỮ LIỆU ---

def log_docker_stats():
    uptime = round(time.time() - START_TIME, 2)
    output = os.popen('docker stats --no-stream --format "{{json .}}"').read().strip().splitlines()
    if not output: return

    stats_list = []
    for line in output:
        try:
            data = json.loads(line)
            row = {"uptime_seconds": uptime, "timestamp": datetime.now().isoformat()}
            row.update(data)
            stats_list.append(row)
        except: continue

    if stats_list:
        file_exists = os.path.isfile(DOCKER_LOG_FILE)
        with open(DOCKER_LOG_FILE, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=stats_list[0].keys())
            if not file_exists: writer.writeheader()
            writer.writerows(stats_list)
    print(f"[{uptime}s] Logged Docker stats")

def log_gpu_processes():
    uptime = round(time.time() - START_TIME, 2)
    gpu_output = os.popen(
        'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits'
    ).read().strip().splitlines()

    proc_stats = []
    # Thêm cột container_id vào header
    headers = ["uptime_seconds", "timestamp", "gpu_uuid", "pid", "container_id", "process_name", "used_memory_mib"]
    
    for line in gpu_output:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            pid = parts[1]
            # TÌM KIẾM: PID này có thuộc container nào không?
            cid = PID_MAP.get(pid, "HOST / UNKNOWN") 
            
            proc_stats.append({
                "uptime_seconds": uptime,
                "timestamp": datetime.now().isoformat(),
                "gpu_uuid": parts[0],
                "pid": pid,
                "container_id": cid, # Cột mới tích hợp
                "process_name": parts[2],
                "used_memory_mib": parts[3]
            })

    if proc_stats:
        file_exists = os.path.isfile(GPU_PROC_LOG_FILE)
        with open(GPU_PROC_LOG_FILE, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists: writer.writeheader()
            writer.writerows(proc_stats)
    print(f"[{uptime}s] Logged GPU processes (Mapped to Containers)")

if __name__ == "__main__":
    print("--- MONITORING START ---")
    print(f"Bảng ánh xạ đã sẵn sàng: {len(PID_MAP)} containers tìm thấy.")
    print(f"Log files: \n1. {DOCKER_LOG_FILE}\n2. {GPU_PROC_LOG_FILE}")
    
    try:
        while True:
            log_docker_stats()
            # log_gpu_processes()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped.")