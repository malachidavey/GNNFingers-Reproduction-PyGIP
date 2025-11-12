#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pygip.attacks.graphmi_attack import main as graphmi_main

if __name__ == '__main__':
    graphmi_main()
