The original code of intrinsic dimension code came from: https://github.com/jgamper/intrinsic-dimensionality
Please cite: 
@misc{jgamper2020intrinsic,
  title   = "Intrinsic-dimensionality Pytorch",
  author  = "Gamper, Jevgenij",
  year    = "2020",
  url     = "https://github.com/jgamper/intrinsic-dimensionality"
}

@article{li2018measuring,
  title={Measuring the intrinsic dimension of objective landscapes},
  author={Li, Chunyuan and Farkhoor, Heerad and Liu, Rosanne and Yosinski, Jason},
  journal={arXiv preprint arXiv:1804.08838},
  year={2018}
}

Note: I made small changes to init and forward function to accomodate multiple inputs and ensure that 
responsible tensors are on the same device. 
