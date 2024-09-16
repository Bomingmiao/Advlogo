models=(v3 v3tiny v4 v4tiny v5 ssd faster_rcnn)
configs=(coco80_diff coco91_diff)

for MODEL in "${models[@]}"
do
    
    python train_fgsm.py -np \
    -cfg="diff_pgd/$MODEL.yaml" \
    -s="./results/dog/{$MODEL}_latent2" \
    -n="$MODEL-dog" \
    --seed=33 \
    --prompt="a dog, 8k"

    for CONFIG in "${configs[@]}"
    do
        python evaluate.py \
        -p "./results/dog/{$MODEL}_latent2/$MODEL-dog_epoch_10.png" \
        -cfg "./configs/eval/$CONFIG.yaml" \
        -lp "./data/INRIAPerson/Test/labels" \
        -dr "./data/INRIAPerson/Test/pos" \
        -s "./data/test" \
        -e 0
    done
done

