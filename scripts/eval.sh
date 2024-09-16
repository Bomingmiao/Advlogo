

python evaluate.py \
-p ./results/dog/v5_latent/v5-dog_epoch_35.png \
-cfg ./configs/eval/coco80.yaml \
-lp ./data/INRIAPerson/Test/labels \
-dr ./data/INRIAPerson/Test/pos \
-s ./data/test \
-e 0

python evaluate.py \
-p ./results/dog/v5_latent/v5-dog_epoch_35.png \
-cfg ./configs/eval/coco91.yaml \
-lp ./data/INRIAPerson/Test/labels \
-dr ./data/INRIAPerson/Test/pos \
-s ./data/test \
-e 0

python evaluate.py \
-p ./results/dog/v5_latent/v5-dog_epoch_35.png \
-cfg ./configs/eval/coco80.yaml \
-lp ./data/INRIAPerson/Test/labels \
-dr ./data/INRIAPerson/Test/pos \
-s ./data/test \
-ud \
-e 0

python evaluate.py \
-p ./results/dog/v5_latent/v5-dog_epoch_35.png \
-cfg ./configs/eval/coco91.yaml \
-lp ./data/INRIAPerson/Test/labels \
-dr ./data/INRIAPerson/Test/pos \
-s ./data/test \
-ud \
-e 0