# Setup (m1/m2 macs)

General directions: [https://developer.apple.com/metal/tensorflow-plugin/](https://developer.apple.com/metal/tensorflow-plugin/).

1. Install Conda

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/etc/fish/conf.d/conda.fish
# source source ~/miniconda/bin/activate # if using zsh
```

2. Install TF and related deps in a Conda env

```shell
conda create --name tf-m2 python
conda activate tf-m2
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos==2.9.2
python -m pip install tensorflow-metal==0.5.1
```

3. Verify using this script

```python
import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)
```
