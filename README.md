# AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models
##  [**Paper**](https://arxiv.org/abs/2409.07002)
**Abstract:** With the rapid development of deep learning, object detectors have demonstrated impressive performance; however, vulnerabilities still exist in certain scenarios. Current research exploring the vulnerabilities using adversarial patches often struggles to balance the trade-off between attack effectiveness and visual quality. To address this problem, we propose a novel framework of patch attack from semantic perspective, which we refer to as AdvLogo. Based on the hypothesis that every semantic space contains an adversarial subspace where images can cause detectors to fail in recognizing objects, we leverage the semantic understanding of the diffusion denoising process and drive the process to adversarial subareas by perturbing the latent and unconditional embeddings at the last timestep. To mitigate the distribution shift that exposes a negative impact on image quality, we apply perturbation to the latent in frequency domain with the Fourier Transform. Experimental results demonstrate that AdvLogo achieves strong attack performance while maintaining high visual quality.

## Framework Overview
![](readme/framework.png)

## Install
### Environment

```bash
conda create -n advlogo python=3.8
conda activate advlogo
pip install -r requirements.txt
```
**Diffusion Models**:
The Stable Diffusion 2.1 can be accessed from [here](https://huggingface.co/stabilityai/stable-diffusion-2-1).
You can download the model and place it in the directiory
- AdvLogo/
  - /stable-diffusion-2-1
      - /feature_extractor
      - /scheduler
      - /text_encoder
      - /tokenizer
      - ...
  - /configs

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
bash ./scripts/eval.sh 
```

#### Training
To train the AdvLogo, you can run the following command:
```bash
bash ./scripts/train_advlogo.sh
# args: 0 gpu-id, -np new tensorboard process
```
or you can run the full command:
```bash
python train_fgsm.py -np \
-cfg=advlogo/v3.yaml \
-s=./results/advlogo/v3 \
-n=v3-dog \
--seed=33 \
--prompt="a dog, 8k"
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
