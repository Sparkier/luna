# Luna

Inspired by [Lucid](https://github.com/tensorflow/lucid), **Luna** is a Feature Visualization package for Tensorflow.
While Lucid does not support Tensorflow 2, **Luna** was built with Tensorflow 2 at its core.

**Luna is under active development. It is research code and not production-ready.**

## Usage

You can use this package directly with your code.
If you place this package next to your python file, you can use it like this in your file:

```
from luna.pretrained_models import models
from luna.featurevis import featurevis, images, image_reader


model = models.model_inceptionv3()
image = images.initialize_image(224, 224)

iterations = 10
learning_rate = 200

blur = False
scale = False
pad_crop = True
flip = False
rotation = True
noise = False
color_aug = True



opt_param = featurevis.OptimizationParameters(iterations, learning_rate)
aug_param = featurevis.AugmentationParameters(blur, scale, pad_crop, flip, rotation, noise, color_aug)

loss, image = featurevis.visualize_filter(image, model, "mixed6", 10, opt_param, aug_param)
print(loss)
images.save_image(image, name="test")
image_reader.save_npy_as_png("test.npy", ".")
```

## Contributing

We greatly appreciate any effort to improve **Luna**.
If you want to contribute to its development, see our [contribution guidelines](./CONTRIBUTING.md).

## Recomended Reading

- [Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)
- [Using Artiﬁcial Intelligence to Augment Human Intelligence](https://distill.pub/2017/aia/)
- [Visualizing Representations: Deep Learning and Human Beings](http://colah.github.io/posts/2015-01-Visualizing-Representations/)
- [Differentiable Image Parameterizations](https://distill.pub/2018/differentiable-parameterizations/)
- [Activation Atlas](https://distill.pub/2019/activation-atlas/)

## Related Talks

- [Lessons from a year of Distill ML Research](https://www.youtube.com/watch?v=jlZsgUZaIyY) (Shan Carter, OpenVisConf)
- [Machine Learning for Visualization](https://www.youtube.com/watch?v=6n-kCYn0zxU) (Ian Johnson, OpenVisConf)

## Community

While we admire their work, we have no affiliation with the Lucid authors or project. Nonetheless, if you are interested in research like this, the Distill slack ([join link](http://slack.distill.pub)) might be a good place for you to get to know other people in this area.

On the awesome Distill slack, Lucid has its own `#proj-lucid` channel, where general questions about the technology are discussed.
