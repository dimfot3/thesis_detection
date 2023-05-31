#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YGyyKCt1gSrtv6ZFUQ1_-gusMBG_rnNC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YGyyKCt1gSrtv6ZFUQ1_-gusMBG_rnNC" -O Pointnetweights.zip && rm -rf /tmp/cookies.txt
unzip Pointnetweights.zip
rm Pointnetweights.zip

