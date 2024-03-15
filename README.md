<div align="center">
  <img src="./.asset/favicon.png" width="30%">
</div>

# SVD Xtend

**Stable Video Diffusion Training Code and Extensions 🚀**

## :bulb: Highlight

- **Finetuning SVD.** See [Part 1](#part-1-training).
- **Tracklet-Conditioned Video Generation.** Building upon SVD, you can control the movement of objects using tracklets(bbox). See [Part 2](#part-2-tracklet2video).

## Part 1: Training

### Comparison
```python
size=(512, 320), motion_bucket_id=127, fps=7, noise_aug_strength=0.00
generator=torch.manual_seed(111)
```
| Init Image        | Before Fine-tuning |After Fine-tuning |
|---------------|-----------------------------|-----------------------------|
| ![demo](https://github.com/pixeli99/SVD_Xtend/assets/46072190/1587c4b5-c104-4d22-8d56-c86e8c716b06)    | ![ori](https://github.com/pixeli99/SVD_Xtend/assets/46072190/18b5af34-d38f-4d19-8856-77895466d152)   | ![ft](https://github.com/pixeli99/SVD_Xtend/assets/46072190/c464397e-aa05-4d8e-9563-3cc78ad04cb3)|
| ![demo](https://github.com/pixeli99/SVD_Xtend/assets/46072190/af3bd957-5b8e-4c21-8791-c9a295761973)    | ![ori](https://github.com/pixeli99/SVD_Xtend/assets/46072190/26d38418-b6fa-40a5-afa6-b278d088638f)   | ![ft](https://github.com/pixeli99/SVD_Xtend/assets/46072190/a49264da-6ccf-48d7-914f-8b0fff9bc99e)|
| ![demo](https://github.com/pixeli99/SVD_Xtend/assets/46072190/2a761c41-d6b2-48b8-a63c-505780369484)    | ![ori](https://github.com/pixeli99/SVD_Xtend/assets/46072190/579bed68-2b31-45d5-8cf2-a4e768fec126)   | ![ft](https://github.com/pixeli99/SVD_Xtend/assets/46072190/eaffe1d5-999b-4d27-8d77-d8e8fd1cd380)|
| ![demo](https://github.com/pixeli99/SVD_Xtend/assets/46072190/09619a6e-50a2-4aec-afb7-d34c071da425)    | ![ori](https://github.com/pixeli99/SVD_Xtend/assets/46072190/2e525ede-474e-499a-9bc5-8f60700ca3fb)   | ![ft](https://github.com/pixeli99/SVD_Xtend/assets/46072190/ec77f39f-653a-4fa7-8ac0-68f8512f9ddb)|

### Video Data Processing
Note that BDD100K is a driving video/image dataset, but this is not a necessity for training. Any video can be used to initiate your training. Please refer to the `DummyDataset` data reading logic. In short, you only need to modify `self.base_folder`. Then arrange your videos in the following file structure:
```bash
self.base_folder
    ├── video_name1
    │   ├── video_frame1
    │   ├── video_frame2
    │   ...
    ├── video_name2
    │   ├── video_frame1
        ├── ...
```
### Training Configuration(on the BDD100K dataset)
This training configuration is for reference only, I set all parameters of unet to be trainable during the training and adopted a learning rate of 1e-5.
```bash
accelerate launch train_svd.py \
    --pretrained_model_name_or_path=/path/to/weight \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=1000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200
```


## Part 2: Tracklet2Video

### Tracklet2Video
We have attempted to incorporate layout control on top of img2video, which makes the motion of objects more controllable, similar to what is demonstrated in the image below. The code and weights will be updated soon.
It should be noted that we use a resolution of `512*320` for SVD to generate videos, so the quality of the generated videos appears to be poor (which is somewhat unfair to SVD), but our intention is to demonstrate the effectiveness of tracklet control, and we will resolve the issue with video quality as soon as possible.
| Init Image        | Gen Video by SVD | Gen Video by Ours |
|---------------|-----------------------------|-----------------------------|
| ![demo1](https://github.com/pixeli99/SVD_Xtend/assets/46072190/e705b4bc-143d-4b56-ac52-df7a728e1731)    | ![svd1](https://github.com/pixeli99/SVD_Xtend/assets/46072190/6d6a44ef-3587-43d9-a078-1f8f4d293097)   |  ![gen1](https://github.com/pixeli99/SVD_Xtend/assets/46072190/35207fb6-343f-44aa-bef0-58d0fc7bd2c1)   |
| ![demo2](https://github.com/pixeli99/SVD_Xtend/assets/46072190/7fe80f97-8a51-457a-b4d8-e20d14f9669e) | ![svd2](https://github.com/pixeli99/SVD_Xtend/assets/46072190/3d87df43-afc8-4917-aaa7-2c432d2cc6f6)   |  ![gen2](https://github.com/pixeli99/SVD_Xtend/assets/46072190/91a16c1d-02c9-4379-8d4a-8fd58f9f0913)   |

### Methods

We have utilized the `Self-Tracking` training from [Boximator](https://arxiv.org/abs/2402.01566) and the `Instance-Enhancer` from [TrackDiffusion](https://arxiv.org/abs/2312.00651).
For more details, please refer to the paper.

## :label: TODO List

- [ ] Support text2video (WIP)
- [x] Support more conditional inputs, such as layout

## :hearts: Acknowledgement

Our model is related to [Diffusers](https://github.com/huggingface/diffusers) and [Stability AI](https://github.com/Stability-AI/generative-models). Thanks for their great work!

Thanks [Boximator](https://boximator.github.io/) and [GLIGEN](https://github.com/gligen/GLIGEN) for their awesome models.

## :black_nib: Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{li2023trackdiffusion,
  title={Trackdiffusion: Multi-object tracking data generation via diffusion models},
  author={Li, Pengxiang and Liu, Zhili and Chen, Kai and Hong, Lanqing and Zhuge, Yunzhi and Yeung, Dit-Yan and Lu, Huchuan and Jia, Xu},
  journal={arXiv preprint arXiv:2312.00651},
  year={2023}
}
```