type=$1

for ratio in '0.00' '0.01' '0.05' '0.1' '0.15' '0.2'
do
for num in {1..5}
do
python -u UDAGCN_our_ot_combine.py --source acm --target citation  --seed $num --aug_ratio1 $ratio --ptype $type --sample OT-random
python -u UDAGCN_our_ot_combine.py --source acm --target dblp --seed $num --aug_ratio1 $ratio --ptype $type --sample OT-random
python -u UDAGCN_our_ot_combine.py --source citation --target acm --seed $num --aug_ratio1 $ratio --ptype $type --sample OT-random
python -u UDAGCN_our_ot_combine.py --source citation --target dblp  --seed $num --aug_ratio1 $ratio --ptype $type --sample OT-random
python -u UDAGCN_our_ot_combine.py --source dblp --target acm  --seed $num --aug_ratio1 $ratio --ptype $type --sample OT-random
python -u UDAGCN_our_ot_combine.py --source dblp --target citation  --seed $num --aug_ratio1 $ratio --ptype $type --sample OT-random
done
done