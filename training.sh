# python finetune.py --name normal --random-seed 42 --i 0
# python finetune.py --name normal --random-seed 34 --i 1
# python finetune.py --name normal --random-seed 48 --i 2
# python finetune.py --name normal --random-seed 23 --i 3

python finetune.py --name old_data4 --random-seed 42
python finetune.py --name old_data5 --random-seed 34
python finetune.py --name old_data6 --random-seed 48
python finetune.py --name old_data7 --random-seed 23

# bash training.sh