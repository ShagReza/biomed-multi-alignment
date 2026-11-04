# Installation

MAMMAL needs its own environment: it uses newer fuse-med-ml, torch and transformers than MMELON, and installing it
into the MMELON environment would break MMELON.

Tested on Google Cloud Vertex AI Workbench (Debian 12, Python 3.10, micromamba). Any conda-style tool works the same.

1. Create the environment and install the repository with its examples' extras:
   ```bash
   micromamba create -y -p ~/envs/biomed-mammal python=3.10
   micromamba activate ~/envs/biomed-mammal
   pip install -e ~/repos/biomed-multi-alignment[examples]
   ```
2. Install the extra packages the AIRCHECK notebooks use:
   ```bash
   pip install -r ~/repos/biomed-multi-alignment/aircheck/requirements-aircheck.txt
   ```
3. Register it as a Jupyter kernel, then pick **biomed-mammal** as the kernel for the notebooks:
   ```bash
   python -m ipykernel install --user --name biomed-mammal --display-name "biomed-mammal"
   ```

Check:
```bash
python -c "from mammal.model import Mammal; import torch; print('OK', torch.__version__, torch.cuda.is_available())"
```

The pretrained model (`ibm/biomed.omics.bl.sm.ma-ted-458m`, about 1.8 GB) is downloaded from Hugging Face the first
time a notebook loads it.
