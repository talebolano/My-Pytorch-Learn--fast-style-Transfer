# My Pytorch Learn--fast style Transfer

a simple sample for fast style Transfer

## how to test

python genrate.py --image test.jpg --out_name style.jpg
## how to train 
mkdir data && cd data && mkdir image

move your train image in data/image/

move your style image in data

python train.py

## result

![原始图 -w500](image/amber.jpg){:height="50%" width="50%"} ![效果图 -w500](image/pre3.jpg){:height="50%" width="50%"}