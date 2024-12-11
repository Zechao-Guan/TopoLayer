

from __future__ import print_function


HIDE_CURSOR = '\x1b[?25l'
SHOW_CURSOR = '\x1b[?25h'


class WriteMixin(object):
    hide_cursor = False

    def __init__(self, message=None, **kwargs):
        super(WriteMixin, self).__init__(**kwargs)
        self._width = 0
        if message:
            self.message = message

        if self.file.isatty():
            if self.hide_cursor:
                print(HIDE_CURSOR, end='', file=self.file)
            print(self.message, end='', file=self.file)
            self.file.flush()

    def write(self, s):
        if self.file.isatty():
            b = '\b' * self._width
            c = s.ljust(self._width)
            print(b + c, end='', file=self.file)
            self._width = max(self._width, len(s))
            self.file.flush()

    def finish(self):
        if self.file.isatty() and self.hide_cursor:
            print(SHOW_CURSOR, end='', file=self.file)


class WritelnMixin(object):
    hide_cursor = False

    def __init__(self, message=None, **kwargs):
        super(WritelnMixin, self).__init__(**kwargs)
        if message:
            self.message = message

        if self.file.isatty() and self.hide_cursor:
            print(HIDE_CURSOR, end='', file=self.file)

    def clearln(self):
        if self.file.isatty():
            print('\r\x1b[K', end='', file=self.file)

    def writeln(self, line):
        if self.file.isatty():
            self.clearln()
            print(line, end='', file=self.file)
            self.file.flush()

    def finish(self):
        if self.file.isatty():
            print(file=self.file)
            if self.hide_cursor:
                print(SHOW_CURSOR, end='', file=self.file)


from signal import signal, SIGINT
from sys import exit


class SigIntMixin(object):
    """Registers a signal handler that calls finish on SIGINT"""

    def __init__(self, *args, **kwargs):
        super(SigIntMixin, self).__init__(*args, **kwargs)
        signal(SIGINT, self._sigint_handler)

    def _sigint_handler(self, signum, frame):
        self.finish()
        exit(0)
