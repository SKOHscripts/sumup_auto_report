#!/bin/bash
set -e

cd /home/skoh/SynologyDrive/Documents/Scripts/sumup
/usr/bin/git pull

/usr/bin/python3 /home/skoh/SynologyDrive/Documents/Scripts/sumup/sumup_stocks.py \
  >> /home/skoh/SynologyDrive/Documents/Scripts/sumup/sumup_stocks.log 2>&1
