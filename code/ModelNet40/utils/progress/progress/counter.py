# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from . import Infinite, Progress
from .helpers import WriteMixin


class Counter(WriteMixin, Infinite):
    message = ''
    hide_cursor = True

    def update(self):
        self.write(str(self.index))


class Countdown(WriteMixin, Progress):
    hide_cursor = True

    def update(self):
        self.write(str(self.remaining))


class Stack(WriteMixin, Progress):
    phases = (' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█')
    hide_cursor = True

    def update(self):
        nphases = len(self.phases)
        i = min(nphases - 1, int(self.progress * nphases))
        self.write(self.phases[i])


class Pie(Stack):
    phases = ('○', '◔', '◑', '◕', '●')
