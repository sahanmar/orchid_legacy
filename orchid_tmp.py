from config import Config
from pipeline import OrchidPipeline

if __name__ == '__main__':
    config = Config.from_path()
    pipeline = OrchidPipeline(config=config)
    pipeline()
