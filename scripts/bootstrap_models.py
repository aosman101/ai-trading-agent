from app.config import get_settings
from app.training.retrainer import ModelTrainer
from app.utils.logging import configure_logging

if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level)
    trainer = ModelTrainer()
    trainer.bootstrap_all()
    print("Bootstrap complete")
