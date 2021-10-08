from pathlib import Path

from config.config import Config
from pipeline import OrchidPipeline

config = Config.load_config(Path("config/config.json"))

def main():
    pipeline = OrchidPipeline.from_config(config)
    pipeline_output = pipeline()

if __name__=="__main__":
	main()
