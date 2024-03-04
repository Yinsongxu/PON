MODEL=resnet18
LOSS=PoissonFocal
nvidia-smi
hostname

for seed in {2020..2024}
do
for i in {0..4}  
do  
python main.py \
    --model $MODEL \
    --loss $LOSS \
    --split $i\
    --seed $seed\
    --snapshot_path dump_ornl/$seed/$MODEL/$LOSS/split_$i/ \
    -bs 16 \
    --lr 1e-4 \
    --max_epoch 60 \
    --weight-decay 1e-5
done
done

for seed in {2020..2024}
do
python test.py \
    --model $MODEL \
    --loss $LOSS \
    --ckpt_path dump_ornl/$seed/$MODEL/$LOSS \
    --snapshot_path dump/test
done

