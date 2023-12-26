#!/bin/bash
echo "4"
source ~/software/anaconda3/etc/profile.d/conda.sh
conda activate zmk1
sleep 20
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s1.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s1.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s2.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s2.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s3.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s3.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s4.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s4.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s5.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s5.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s6.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s6.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s7.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s7.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s8.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s8.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s9.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s9.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s10.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s10.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s11.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s11.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s12.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s12.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s13.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s13.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s14.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s14.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s15.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s15.log 2>&1 &
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/s16.py > /public/home/ssct005t/project/zmk/device/16/test/log4/s16.log 2>&1 &
sleep 10
nohup python3 /public/home/ssct005t/project/zmk/device/16/test/sub_cloud.py > /public/home/ssct005t/project/zmk/device/16/test/log4/sub_cloud.log 2>&1 &
python3 /public/home/ssct005t/project/zmk/device/16/test/c4.py > /public/home/ssct005t/project/zmk/device/16/test/log4/c4.log 2>&1
