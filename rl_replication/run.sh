nohup python s1.py > log/s1.log 2>&1 &
nohup python s2.py > log/s2.log 2>&1 &
nohup python s3.py > log/s3.log 2>&1 &
nohup python s4.py > log/s4.log 2>&1 &
nohup python sub_cloud.py > log/sub.log 2>&1 &
sleep 5
nohup python c5.py > log/ga.log 2>&1 &