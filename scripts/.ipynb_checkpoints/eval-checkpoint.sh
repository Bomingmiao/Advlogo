'''
cuda=$1


PROJECT_DIR=$(pwd)
patch_path=$PROJECT_DIR/results/v5-demo.png
save_dir=$PROJECT_DIR/eval/inria/demo
targets=(eval/coco80 eval/coco91) # Run twice to test both yolo-models(coco80) and torch-models(coco91).

for config in ${targets[@]}
do
  echo "Config file: $config"
  echo "Evaluating the adversarial patch: $patch_path"

  cmd="CUDA_VISIBLE_DEVICES=${device} python evaluate.py \
  -p $patch_path \
  -cfg ./configs/$config.yaml \
  -lp $PROJECT_DIR/data/INRIAPerson/Test/labels \
  -dr $PROJECT_DIR/data/INRIAPerson/Test/pos \
  -s $save_dir \
  -e 0 &"

  echo $cmd
  eval $cmd
  sleep 2
done
# Use an absolute path for -lp(label path), -dr(preprocesser directory) and -s(save path).
# For detailed supports of the arguments:
#python evaluate.py -h
'''
python evaluate.py \
-p ./results/demo/v5-combine-demo.png \
-cfg ./configs/eval/coco80.yaml \
-lp ./data/INRIAPerson/Test/labels \
-dr ./data/INRIAPerson/Test/pos \
-s ./data/test \
-e 0