# AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models


[**Paper**](https://arxiv.org/abs/2409.07002)
**Abstract:** With the rapid development of deep learning, object detectors have demonstrated impressive performance; however, vulnerabilities still exist in certain scenarios. Current research exploring the vulnerabilities using adversarial patches often struggles to balance the trade-off between attack effectiveness and visual quality. To address this problem, we propose a novel framework of patch attack from semantic perspective, which we refer to as AdvLogo. Based on the hypothesis that every semantic space contains an adversarial subspace where images can cause detectors to fail in recognizing objects, we leverage the semantic understanding of the diffusion denoising process and drive the process to adversarial subareas by perturbing the latent and unconditional embeddings at the last timestep. To mitigate the distribution shift that exposes a negative impact on image quality, we apply perturbation to the latent in frequency domain with the Fourier Transform. Experimental results demonstrate that AdvLogo achieves strong attack performance while maintaining high visual quality.

## Framework Overview
We provide a main pipeline to craft a universal adversarial patch to achieve cross-model & cross-instance attack on detectors, 
and support evaluations on given data & models.

![](readme/framework.png)

## Install
### Environment

```bash
conda create -n advlogo python=3.8
conda activate advlogo
pip install -r requirements.txt
```

 **Data**

| Data        |                                             Generated Labels                                             |                                              Source                                              |                                            
|-------------|:--------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| INRIAPerson |  [GoogleDrive](https://drive.google.com/drive/folders/1zKO6yXllhReiDS04WKkb6JIkxvAW2s_9?usp=share_link)  |               [Paper](https://hal.inria.fr/docs/00/54/85/12/PDF/hog_cvpr2005.pdf)                |

See more details in [Docs](./readme/data.md).


### Run

#### Evaluation

The evaluation metrics of the **Mean Average Precision([mAP](https://github.com/Cartucho/mAP))** is provided.

```bash
# You can run the demo script directly:
bash ./scripts/eval.sh 0 # gpu id
```

```bash
# To run the full command in the root proj dir:
python evaluate.py \
-p ./results/v5-demo.png \
-cfg ./configs/eval/coco80.yaml \
-lp ./data/INRIAPerson/Test/labels \
-dr ./data/INRIAPerson/Test/pos \
-s ./data/test \
-e 0 # attack class id

# for torch-models(coco91): replace -cfg with ./configs/eval/coco91.yaml

# For detailed supports of the arguments:
python evaluate.py -h
```

#### Training
```bash
# You can run the demo script directly:
bash ./scripts/train.sh 0 -np
# args: 0 gpu-id, -np new tensorboard process
```

```bash
# Or run the full command:
python train_optim.py -np \
-cfg=demo.yaml \
-s=./results/demo \
-n=v5-combine-demo # patch name & tensorboard name

# For detailed supports of the arguments:
python train_optim.py -h
```
The default save path of tensorboard logs is **runs/**.

Modify the config .yaml files for custom settings, see details in [**README**](https://github.com/VDIGPKU/T-SEA/blob/main/configs/README.yaml).



## Acknowledgements

* AdvPatch - [**Paper**](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.pdf) 
| [Source Code](https://gitlab.com/EAVISE/adversarial-yolo)

## Citation
```
@article{miao2024advlogo,
  title={AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models},
  author={Miao, Boming and Li, Chunxiao and Zhu, Yao and Sun, Weixiang and Wang, Zizhe and Wang, Xiaoyi and Xie, Chuanlong},
  journal={arXiv preprint arXiv:2409.07002},
  year={2024}
}
```


## License

The project is only free for academic research purposes, but needs authorization forcommerce. For commerce permission, please contact bomingmiao@gmail.com.
