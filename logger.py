import logging
import sys

class MultiLogger:
    def __init__(self, log_func, log_file_path=None):
        self.log_func = log_func
        self._in_write = False  # re-entrancy guard, see write()
        self.logger = logging.getLogger("multi_logger")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # If a valid log_file_path is provided, add a FileHandler
        if log_file_path:
            try:
                self.file_handler = logging.FileHandler(log_file_path)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                self.file_handler.setFormatter(formatter)
                self.logger.addHandler(self.file_handler)
            except Exception as e:
                self.log_func(f"Error setting up file logging: {e}")

    def write(self, message):
        if message.strip():
            if self._in_write:
                # log_func itself called print() (every CLI script's log callback is a plain
                # `print(msg)` wrapper -- run_lutzia_pipeline.py, run_flare_pipeline.py,
                # buzzsuite_cli.py). Since sys.stdout is still THIS MultiLogger while log_func
                # runs, that print() would call write() again -> log_func -> print() -> write()
                # ... forever, hitting Python's recursion limit (confirmed live: any CLI-driven
                # call to extract_average_background's stdout redirect raised
                # "RecursionError: maximum recursion depth exceeded" on its very first print(),
                # aborting background-image computation for every remaining segment with no
                # visible cause -- see DEVLOG). Break the cycle by writing straight to the real
                # console instead of back through self.log_func.
                try:
                    sys.__stdout__.write(message if message.endswith('\n') else message + '\n')
                except Exception:
                    pass
                return
            self._in_write = True
            try:
                # Log to the provided log function
                self.log_func(message)
            finally:
                self._in_write = False
            # Log to file only if a file handler was successfully added
            if self.logger.hasHandlers():
                self.logger.debug(message)

    def flush(self):
        if self.logger.hasHandlers():
            for handler in self.logger.handlers:
                handler.flush()

    def progress(self, current, total, bar_length=20):
        fraction = current / total
        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '
        ending = '\n' if current == total else '\r'
        message = f'Progress: [{arrow}{padding}] {int(fraction*100)}% {ending}'
        # log_func is a 1-arg callable everywhere it's actually passed in (VideoAnalyzerApp.log,
        # every CLI script's print wrapper, etc.) -- end='' would raise TypeError the moment this
        # method is ever called. Currently unused (grep confirms only a commented-out caller in
        # misc_functions.py) but fixed here so it's not a landmine if it's picked up later.
        self.log_func(message.strip())
        print(message, end='')  # Print to stdout to also handle in console

class LogFunctionStream:
    def __init__(self, log_func):
        self.log_func = log_func

    def write(self, message):
        if message.strip():
            self.log_func(message)  # Write to the UI logging function

    def flush(self):
        pass  # No need to flush the UI log function
