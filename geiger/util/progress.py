import sys
from time import time


class Progress():
    """
    A nice progress bar which provides time estimates.
    """
    green = '\033[92m'
    clear = '\033[0m'

    def __init__(self, tag=''):
        self.tag = tag

    def print_progress(self, percent):
        """
        Show a progress bar.
        """
        if not hasattr(self, 'start_time'):
            self.start_time = time()
            elapsed_time = 0
        else:
            elapsed_time = time() - self.start_time
        if percent == 0:
            estimated = 0
        else:
            estimated = elapsed_time/percent
        remaining = estimated - elapsed_time
        percent *= 100

        if remaining > 3600:
            countdown = '{:8.2f}hrs'.format(remaining/3600)
        elif remaining > 60:
            countdown = '{:8.2f}min'.format(remaining/60)
        else:
            countdown = '{:8.2f}sec'.format(remaining)

        p = int(percent)
        width = 100 - p
        info = '{0:8.3f}% {1} {2}'.format(percent, countdown, self.tag)
        sys.stdout.write('\r[{3}{0}{4}{1}] {2}'.format('|' * p, ' ' * width, info, self.green, self.clear))

        if percent == 100:
            print('\n')
