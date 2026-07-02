from helpers.config import get_settings, Settings

class BaseController:
    def __init__(self, config: Settings = get_settings()):
        self.app_settings = config