# RI-Mamba: Rotation-Invariant Mamba for Robust Text-to-Shape Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-2602.11673-b31b1b.svg)](https://arxiv.org/abs/2602.11673)
[![CVPR 2026](https://img.shields.io/badge/Venue-CVPR%202026-blue.svg)](https://cvpr.thecvf.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official implementation of the paper: **"RI-Mamba: Rotation-Invariant Mamba for Robust Text-to-Shape Retrieval"** (CVPR 2026).

---

## 📢 News
- **[Feb 2026]** Our paper is available on [arXiv](https://arxiv.org/abs/2602.11673).
- **[Feb 2026]** RI-Mamba has been accepted to **CVPR 2026**! 🎉
- **[Apr 2026]** Initial repository released. Code is coming soon!

## 📝 Abstract
3D assets have rapidly expanded in quantity and diversity due to the growing popularity of virtual reality and gaming. As a result, text-to-shape retrieval has become essential in facilitating intuitive search within large repositories. However, existing methods require canonical poses and support few object categories, limiting their real-world applicability where objects can belong to diverse classes and appear in random orientations. To address this challenge, we propose RI-Mamba, the first rotation-invariant state-space model for point clouds. RI-Mamba defines global and local reference frames to disentangle pose from geometry and uses Hilbert sorting to construct token sequences with meaningful geometric structure while maintaining rotation invariance. We further introduce a novel strategy to compute orientational embeddings and reintegrate them via feature-wise linear modulation, effectively recovering spatial context and enhancing model expressiveness. Our strategy is inherently compatible with state-space models and operates in linear time. To scale up retrieval, we adopt cross-modal contrastive learning with automated triplet generation, allowing training on diverse datasets without manual annotation. Extensive experiments demonstrate RI-Mamba's superior representational capacity and robustness, achieving state-of-the-art performance on the OmniObject3D benchmark across more than 200 object categories under arbitrary orientations.

## 🛠️ Code and Models
The code and pretrained models will be made available soon. Stay tuned!

## 📌 Citation
If you find our work useful, please consider citing:

```bibtex
@inproceedings{nguyen2026rimamba,
  title={RI-Mamba: Rotation-Invariant Mamba for Robust Text-to-Shape Retrieval},
  author={Nguyen, Khanh and Edirimuni, Dasith de Silva and Hassan, Ghulam Mubashar and Mian, Ajmal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 📧 Contact
For any questions, please contact Khanh Nguyen at [duykhanh.nguyen@research.uwa.edu.au](mailto:duykhanh.nguyen@research.uwa.edu.au).
