prompts=("a house, 8k")
seeds=(33)
names=(house)
configs=(coco80_diff coco91_diff)

for i in "${!prompts[@]}"; do
    PROMPT=${prompts[$i]}
    SEED=${seeds[$i]}
    NAME=${names[$i]}
    
    python train_fgsm.py -np \
    -cfg="diff_pgd/v3.yaml" \
    -s="./results/${NAME}/v3" \
    -n="v3-$NAME" \
    --seed=$SEED \
    --prompt="$PROMPT"
'''
    for CONFIG in "${configs[@]}"; do
        python evaluate.py \
        -p "./results/${NAME}/faster_rcnn/faster_rcnn-${NAME}_epoch_50.png" \
        -cfg "./configs/eval/$CONFIG.yaml" \
        -lp "./data/INRIAPerson/Test/labels" \
        -dr "./data/INRIAPerson/Test/pos" \
        -s "./data/test" \
        -e 0
    done
'''
done

