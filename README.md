# SVD_Xtend

**Stable Video Diffusion Training Code 🚀**

## Comparison
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

## Video Data Processing
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
## Training Configuration(on the BDD100K dataset)
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

## Disclaimer

While the codebase is functional and provides an enhancement in video generation(maybe? 🤷), it's important to note that there are still some uncertainties regarding the finer details of its implementation.

## TODO List

- [ ] Support text2video (WIP)
- [ ] Support more conditional inputs, such as layout

## Contribution

Feel free to fork this repository, submit pull requests, or open issues to discuss potential changes or report bugs. With your valuable input, we can continuously improve SVD_Xtend for the community.
