import os
import json
import time
import subprocess
import threading
import re
from datetime import datetime
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class DeviceMonitor:
    def __init__(self, run_id, gateway_url='localhost:9091', interval=5):
        self.run_id = run_id
        self.gateway_url = gateway_url.replace("http://", "")
        self.interval = interval
        self.registry = CollectorRegistry()
        self._setup_metrics()

    def _setup_metrics(self):
        self.docker_cpu = Gauge('docker_cpu_usage_percent', 'CPU usage %', 
                                ['run_id', 'container_name'], registry=self.registry)
        self.docker_mem = Gauge('docker_mem_usage_bytes', 'Memory usage in GB', 
                                ['run_id', 'container_name'], registry=self.registry)
        self.gpu_mem = Gauge('gpu_process_mem_mib', 'GPU memory usage in GB', 
                             ['run_id', 'gpu_uuid', 'container_name', 'pid'], registry=self.registry)

    def _parse_value(self, value_str):
        try:
            num = re.findall(r"[-+]?\d*\.\d+|\d+", value_str)
            return float(num[0]) if num else 0.0
        except:
            return 0.0
        
    def _parse_to_gb(self, value_str):
        try:
            raw_val = value_str.split('/')[0].strip().upper()
            match = re.search(r"([0-9.]+)\s*([A-Z]*)", raw_val)
            if not match:
                return 0.0
            
            number = float(match.group(1))
            unit = match.group(2)

            if 'G' in unit:
                return number
            elif 'M' in unit:
                return number / 1024
            elif 'K' in unit:
                return number / (1024**2)
            elif 'B' in unit:
                return number / (1024**3)
            
            return number
        except:
            return 0.0

    def _get_pid_to_container_map(self):
        pid_map = {}
        try:
            res = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True)
            cids = res.stdout.strip().split('\n')
            for cid in cids:
                if not cid: continue
                inspect = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True)
                data = json.loads(inspect.stdout)
                pid = str(data[0]['State']['Pid'])
                name = data[0]['Name'].lstrip('/')
                pid_map[pid] = name
        except Exception as e:
            print(f"Error mapping PID: {e}")
        return pid_map

    def _docker_loop(self):
        while True:
            try:
                output = os.popen('docker stats --no-stream --format "{{json .}}"').read().strip().splitlines()
                for line in output:
                    data = json.loads(line)
                    name = data['Name']
                    
                    self.docker_cpu.labels(run_id=self.run_id, container_name=name).set(self._parse_value(data['CPUPerc']))
                    self.docker_mem.labels(run_id=self.run_id, container_name=name).set(self._parse_to_gb(data['MemUsage']))
                
                push_to_gateway(self.gateway_url, job=f'docker_{self.run_id}', registry=self.registry)
            except Exception as e:
                print(f"Docker Thread Error: {e}")
            time.sleep(self.interval)

    def _gpu_loop(self):
        while True:
            try:
                pid_map = self._get_pid_to_container_map()
                cmd = 'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits'
                gpu_output = os.popen(cmd).read().strip().splitlines()

                for line in gpu_output:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        uuid, pid, mem = parts[0], parts[1], parts[2]
                        c_name = pid_map.get(pid, "HOST/UNKNOWN")
                        
                        self.gpu_mem.labels(
                            run_id=self.run_id, 
                            gpu_uuid=uuid, 
                            container_name=c_name, 
                            pid=pid
                        ).set(float(mem))
                
                push_to_gateway(self.gateway_url, job=f'gpu_{self.run_id}', registry=self.registry)
            except Exception as e:
                print(f"GPU Thread Error: {e}")
            time.sleep(self.interval)

    def start(self):
        t1 = threading.Thread(target=self._docker_loop, daemon=True)
        t2 = threading.Thread(target=self._gpu_loop, daemon=True)
        
        t1.start()
        # t2.start()
        print(f"Monitoring started for {self.run_id}...")
        # try:
        #     while True:
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     print("Stopping...")

if __name__ == "__main__":
    RUN_ID = "experiment_v1_alpha" 
    monitor = DeviceMonitor(run_id=RUN_ID, gateway_url='localhost:9091')
    monitor.start()