import sys

def progress_bar(current, total, bar_length=50, text="Progress"):
    anitext = ['\\', '|', '/', '-']
    percent = float(current) / total
    abs = f"{{{current} / {total}}}"
    arrow = '|' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    text = '[' + anitext[(current % 4)] + '] ' + text
    sys.stdout.write("\r{0}: [{1}] {2}% {3}".format(text, arrow + spaces, int(round(percent * 100)), abs))
    sys.stdout.flush()
