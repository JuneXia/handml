# -*- coding: UTF-8 -*-
import os
import sys
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)

import socket
import getpass

home_path = os.environ['HOME']
user_name = getpass.getuser()
host_name = socket.gethostname()

