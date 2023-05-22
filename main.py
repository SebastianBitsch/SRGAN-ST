import os
from config import Config
from train import train

def get_jobindex(fallback:int = 0) -> int:
    """Get the job-index set in bash. This is mostly for array jobs where multiple models are trained in parallel"""
    num = os.getenv('jobindex')
    return num if num else fallback


def benchmark_experiment(config: Config) -> Config:
    config.EXP.NAME = "plain-benchmark-run"
    return config


if __name__ == "__main__":

    # Get job-index from bash, if the job is not an array it will be zero
    job_index = get_jobindex()

    # Edit config based on 
    config = Config()

    config.EXP.NAME = "plain-benchmark-run-no-labelsmoothing"

    print(f"Running job: {job_index}")
    train(config = config)
    print(f"Findighed job: {job_index}")
