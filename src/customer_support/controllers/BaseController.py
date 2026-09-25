import random
import string
from pathlib import Path

from customer_support.helpers.config import Settings, get_settings


class BaseController:
    def __init__(self, settings: Settings | None = None):
        self.app_settings = settings or get_settings()

        self.base_dir = Path(self.app_settings.ASSETS_DIR)
        self.files_dir = self.base_dir / "files"
        self.data_dir = self.base_dir / "database"

    def generate_random_string(self, length: int = 12) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    def get_database_path(self, db_name: str) -> str:
        database_path = self.data_dir / db_name
        database_path.mkdir(parents=True, exist_ok=True)
        return str(database_path)
