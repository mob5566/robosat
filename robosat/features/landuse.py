from .base_handler import BaseHandler

class LanduseHandler(BaseHandler):
    def __init__(self):
        super().__init__(5, 5, 0.005, 15)
