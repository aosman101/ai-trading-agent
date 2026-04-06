from __future__ import annotations

import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler

from app.config import get_settings
from app.orchestrator import TradingOrchestrator
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    orchestrator = TradingOrchestrator()

    scheduler = BlockingScheduler(timezone=settings.timezone)
    scheduler.add_job(
        orchestrator.run_cycle,
        trigger="interval",
        minutes=settings.worker_poll_minutes,
        id="trade-cycle",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.add_job(
        orchestrator.retrain,
        trigger="cron",
        hour=1,
        minute=15,
        id="nightly-retrain",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    def _shutdown(signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully", sig_name)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("Starting worker. First cycle will run immediately.")
    orchestrator.run_cycle()
    scheduler.start()


if __name__ == "__main__":
    main()
