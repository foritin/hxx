class BacktestReportGenerator:

    def __init__(self, logger=None):
        self.logger = logger

    def log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
