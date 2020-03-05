#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker run -v $DIR:/home/akhil/work -it --rm -p 8888:8888 tf:v1
