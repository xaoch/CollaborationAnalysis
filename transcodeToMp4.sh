#!/bin/bash

ffmpeg -y -i $1 "${1/%avi/mp4}"