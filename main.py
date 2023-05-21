import os
from config import Config
from train import train

if __name__ == "__main__":
    # Get the job-index set in bash. This is mostly for array jobs where multiple models are trained in parallel
    job_index = os.getenv('jobindex')

    print(f"Running job: {job_index}")
    
    # Edit config based on 
    config = Config()


    train(config = config)

    print(f"Findighed job: {job_index}")
