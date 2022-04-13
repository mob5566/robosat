from .base_handler import BaseHandler

class BridgeHandler(BaseHandler):
    def __init__(self):
        super().__init__(5, 5, 0.01, 15)
