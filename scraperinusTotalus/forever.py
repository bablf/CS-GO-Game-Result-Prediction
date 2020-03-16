#!/usr/bin/python3

# https://www.alexkras.com/how-to-restart-python-script-after-exception-and-run-it-forever/

from subprocess import Popen
import sys

filename = "scraperinusTotalus.py" #sys.argv[1]
while True:
    print("\nStarting " + filename)
    p = Popen("python " + filename, shell=True)
    p.wait()