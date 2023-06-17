from multiprocessing.managers import SyncManager
from typing import Optional, Tuple
import queue

address = "127.0.0.1"
port = 50000
password = "password"


class QueueManager(SyncManager):
    required_engine_config: Optional[str] = None

    def requires_engine_config(self) -> bool:
        return self.required_engine_config is not None

    @property
    def engine_config(self) -> Optional[str]:
        return QueueManager.required_engine_config

    @staticmethod
    def set_engine_config(engine_config: str) -> None:
        QueueManager.required_engine_config = engine_config


def connect_to_manager() -> Tuple[queue.Queue, queue.Queue, Optional[str]]:
    QueueManager.register("input_queue")
    QueueManager.register("output_queue")
    manager = QueueManager(address=(address, port), authkey=password.encode("utf-8"))
    manager.connect()
    return manager.input_queue(), manager.output_queue(), manager.engine_config
