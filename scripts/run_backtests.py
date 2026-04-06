from pathlib import Path

from app.backtesting.engine import WalkForwardBacktester
from app.config import get_settings
from app.utils.logging import configure_logging

if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level)
    backtester = WalkForwardBacktester()
    report = backtester.run_all()
    output_path = Path(settings.data_dir) / "backtest_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Backtest report saved to {output_path}")
    print(report)
