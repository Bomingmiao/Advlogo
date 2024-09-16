
python train_fgsm.py -np \
-cfg=diff_pgd/ssd.yaml \
-s=./results/dog/test \
-n=test \
--seed=33 \
--prompt="a dog, 8k"
