# Reflection: CNN Generalization, CIFAR-10, and Scaling to ImageNet

These labs showed me that image-classification performance depends on more than model size. The strongest results come from matching the model’s inductive bias to the data, controlling overfitting, and making optimization and data processing efficient enough to support the scale of the problem.

## Why CNNs generalize better than MLPs on images

CNNs generalize better on images mainly because their architecture reflects the structure of visual data. An MLP flattens an image into a vector, so a 32×32 RGB image becomes 3,072 inputs and the original spatial relationships are no longer explicit. Every input unit is connected to every hidden unit, which introduces many parameters without providing a preference for nearby pixels or repeated visual patterns.

Convolutional layers instead use local connections and shared weights. A small filter can detect the same edge, texture, or shape wherever it appears in an image. This weight sharing gives CNNs translation equivariance and greatly improves parameter efficiency. Stacking convolutional layers also creates a hierarchy: early layers represent simple patterns, while later layers combine them into more complex objects. Pooling adds some tolerance to small shifts and reduces computation. These properties form a useful inductive bias, allowing CNNs to learn useful features from fewer examples and usually overfit less than a comparably sized MLP. This advantage is not automatic; it still depends on suitable training data, regularization, and optimization.

## What limits performance on CIFAR-10

CIFAR-10 performance is limited by both the dataset and the training process. Its images are only 32×32 pixels, so fine details can be difficult to recover. Similar categories create additional ambiguity: my strongest confusions were cat versus dog and automobile versus truck. Dog, cat, and deer also had the lowest per-class validation accuracy. The high-confidence error gallery showed that visually ambiguous images can still produce nearly 100% confident incorrect predictions, so accuracy alone does not describe model reliability.

My experiments also identified overfitting as the main limitation. The deeper CNN reached 88.89% training accuracy but only 72.78% final validation accuracy, a 16.11-point gap. Validation loss began rising after epoch 5, while the best validation accuracy of 74.10% occurred at epoch 7. Thus, performance was not primarily limited by an inability to fit the training set. The model was learning training-specific patterns that did not transfer consistently.

The Lab 2 comparisons reinforced this point. Increasing depth improved best validation accuracy, but batch normalization produced only a small gain and dropout at 0.5 did not beat the baseline. Simply adding capacity or applying a regularization technique without tuning it is therefore not enough. Limited training data, lack of augmentation, hyperparameters, and the learning-rate schedule all constrain the final result.

## How I would scale to ImageNet

I would not scale by merely adding more convolutional layers. ImageNet has roughly 1.2 million training images, 1,000 classes, and much larger inputs, commonly 224×224 pixels. I would use a residual architecture such as ResNet, or another established convolutional or vision-transformer backbone, because residual connections and normalization make deeper networks easier to optimize. Stronger augmentation, weight decay, label smoothing, learning-rate warmup, and a decay schedule would help control generalization and training stability.

The systems side would be equally important. Larger images and models require substantially more memory and computation, so I would use GPU training, mixed precision, efficient data loading, checkpointing, and distributed training when necessary. A pretrained backbone would be preferable when permitted because it can reuse features learned from large-scale data. Finally, I would monitor top-1 and top-5 validation accuracy, per-class behavior, and confidence errors rather than selecting a model from training loss alone.

Overall, the main lesson is that better performance comes from combining an appropriate architecture with regularization, optimization, and careful error analysis. My next controlled experiment would be to add random cropping and horizontal flipping to the deeper CIFAR-10 CNN, then repeat the same validation and confusion-matrix analysis.
