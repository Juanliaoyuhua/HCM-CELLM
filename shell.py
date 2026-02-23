import os
import sys
import logging
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(script_name, description):
    logger.info(f"开始运行{description}...")
    try:
        result = subprocess.run([sys.executable, script_name],
                                capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info(f"{description}运行成功")
            logger.info(f"输出: {result.stdout[-500:]}")
        else:
            logger.error(f"{description}运行失败，返回码: {result.returncode}")
            logger.error(f"错误输出: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error(f"{description}运行超时")
    except Exception as e:
        logger.error(f"运行{description}时发生异常: {e}")

def main():
    logger.info("开始依次运行三个实验程序...")

    os.makedirs("Result_data/cloud-edge", exist_ok=True)
    os.makedirs("Result_data/ppo", exist_ok=True)

    run_experiment("main_edge_only.py", "全边缘推理基线实验")
    time.sleep(5)

    run_experiment("main_cloud_edge.py", "云边协同推理基线实验")
    time.sleep(5)

    run_experiment("PPO.py", "PPO-SLMCP算法实验")
    logger.info("所有实验程序已运行完成！")

if __name__ == "__main__":
    main()