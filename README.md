# DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation

## Introduction

Representing 3D shapes is an open problem in 3D computer vision and computer graphics. A good 3D shape representation should be suitable for rendering, 3D reconstruction, object recognition, and many others. It should be memory efficient and computationally efficient while allowing us to get quantitative results. In this post, I discuss DeepSDF, a method for learning a 3D shape representation method (signed distance functions) using a neural network. 

### What is a Signed Distance Function (SDF)?

Let's start with what a signed distance function (SDF) is. It is a function that takes a point (a 3D point in our case) and gives its signed distance to the closest surface. The sign of this distance is `+` if the point is inside the surface, `-` otherwise. Note that in shape representation, we assume the shape is a single closed surface.

Signed-distance functions can represent surfaces smoothly. If we consider two points, one with `+` and one with `-` signed distance, we can locate a point by their weighted average. Sampling many points near the surface, we can obtain a set of very precise locations for surface points. Rendering can be done with well-known algorithms, such as ray-casting and marching cubes. Last, but not least, some open-source software is available to reconstruct SDFs from depth images. 

### DeepSDF Motivations
SDFs are awesome! However, to be used in practical applications, they need to be stored as voxels (3D images). Voxels are very inefficient in terms of disk space, memory use and computation. Therefore, we have to discretize too much for efficiency, at the expense of losing accuracy.

DeepSDF people instead suggest learning the SDFs with a neural network. An SDF looks a lot like a nonlinear binary classifier, and neural networks are very good nonlinear classifiers!


### DeepSDF Contribution
At a high level,  authors first define an embedding lookup, where each shape in a shape category has an embedding latent vector. They then train a single deep fully-connected neural network per shape category. The network takes the embedding vector and a 3D point as the input, and gives signed distance value as the output. New embedding vectors can be obtained for unknown shapes via interpolation or by optimizing for a new embedding vector.

DeepSDF allows training a single neural network per an entire database of shapes, in our case a database of shapes in a single category. Embedding vectors can be pretty small, like 128 or 256 dimensional. Event if shapes were represented as very small voxel grids of size 32x32x32, a single voxel grid would take up the space of 128 embeddings!

The authors address three problems using their models. First, they try to compress a known category of shapes, measuring the accuracy of reconstruction using the shared neural network and small embeddings. Then, they address the same problem with unknown shapes to assess the learning capability of their model. Third, they attack the problem of shape completion, which is, as we will see, a very natural extension of DeepSDF framework.

If you don't understand some details, don't worry, I will discuss further details on how they formulate the problem and train their models exactly. Before that, let's take a small break from DeepSDF and review shortly some related works that will serve our understanding.


## Related Works
### Shape Representation

Previously, we discussed there are many alternatives for representing 3D shapes. Some of these representations are obviously related to our approach, while others serve as a baseline for comparison.

Authors think three lines of work in shape representation is relevant. Namely point-based, mesh-based, and voxel-based ones.

#### Point-Based
Point Cloud is one of the most popular representations used in 3D. They define shapes as an unordered set of 3D points on the surface. In modern depth sensors (LiDARs, Kinect, etc.), point clouds are raw format obtained from sensors.

Recently, there is a hype around using them with neural networks, especially in 3D recognition. In this line of works, people can reach their goals by using a sparse set of point clouds (usually 1k to 4k points). At this common scale, point clouds are computationally efficient and compact.

The downside of point clouds is that their lack of topology information. We basically have a bunch of points without any information on how these points are related to others. For rendering, we may require a lot of points (100k or so per shape in my own experience) for this reason. Moreover, the authors claim point clouds are not suitable for generating closed (watertight) surfaces.

#### Mesh-based
Polygonal Meshes represent shapes as groups of points that form polygons on the surface. A triangular mesh is the default data format used for rendering in modern graphics hardware, and much more efficient than point clouds in this use case. Synthetic 3D models created by artists are often in mesh format by default.

In the world of meshes, one related work very similar to DeepSDF approach is AltasNet. This work also uses embeddings and shared neural networks per category. They parameterize squares in 2D and then deform them into 3D rectangles, obtaining a quad mesh.

The aforementioned square-based AtlasNet approach cannot create closed surfaces, so they also define a version that deforms a parametric sphere. This version drops some fine details but can represent closed surfaces.

#### Voxel-based
Voxels are basically 3D images. Instead of having pixels in each point of a 2D grid, we have a voxel in each point of a 3D grid. Voxels can be used to represent different data formats (including SDFs) as a function of discretized 3D grid coordinates.

Voxels are super-easy to work with using neural networks! Just replace all `Conv2D` and `BatchNorm2D` layers with `Conv3D` and `BatchNorm3D` layers (and all others end with 2D with 3D) and you have the architecture! The problem is their inefficiency. In 3D shapes, surfaces are often quadratic while volumes are often cubic (compare volume vs. area formula of cube and sphere). As we increase the voxel resolution, memory use increases faster than the number of voxels on the surface. Since surfaces are what we are most interested in, voxels are wasteful.

One important related work in voxels is OGN, the Octree-generating network. OGN uses a hierarchical data representation for voxels called Octrees to generate shapes. OGN predicts both the structure and content of an octree with a convolutional neural network to represent shapes.

### Generative Models
When it comes to popularity, generative models GANs and VAEs are very big. Here, the authors point out their architecture is a special kind of generative model, called Auto-Decoder.

An Auto-Decoder is an auto-encoder without an encoder. It learns latent codes and a decoder for representing the data. Authors claim this architecture is easier to train compared to GANs and VAEs. They also ask why the encoder part is necessary if we just need the decoder for the task in hand.

### Shape Completion
For shape completion, the authors compare their results with an architecture called 3D-EPN. This architecture predicts a coarse, low-resolution voxel grid from an incomplete one. Then, they increase this resolution by retrieving and using reference shapes. Another significance of this approach is that it uses SDFs in the voxel form.


## Methodology
In this section, I discuss the authors' mathematical formulation of SDFs as neural networks. As the authors did in the paper, I first start with the naive idea of training a neural network per SDF, and then study the re-formulation as a generative model. I also explain the data generation process, model and training details they used.

### SDFs as Neural Networks
We want a neural network model (or simply a parametric function approximator) that can produce signed distance values for a given point. Here, let's forget about shape databases and consider we have a single shape `X`. Our dataset is sampled from this shape, as ground-truth `<point, sdf(point)>` pairs. We can now basically overfit a neural network to this shape. The resulting neural network becomes our DeepSDF representation.

Another important detail to note is DeepSDF technically learns a Truncated Signed Distance Function (TSDF). TSDF allows marking regions too far away from the surface as empty, making the SDF concentrate around the points in the surface, which are the ones that are important for rendering.

Clearly, this approach is not feasible. Yes, using voxels were inefficient, but so does having a several-million parameter neural network trained for each shape! This brings us to our auto-decoder based formulation.

### SDFs as Auto-Decoders
As noted before, DeepSDF is actually an auto-decoder. To be more specific, DeepSDF can be seen as a conditional auto-decoder. In this formulation, embedding vectors are latent codes, 3D points are conditions, and the neural network is the decoder. While a similar formulation is possible with other generative models, like GANs, authors chose to use an Auto-Decoder.  

Here, we rather have an array of `X`s, while every `X` still is a shape with `(point, sdf(point))` pairs. We have a single neural network and a set of latent vector `z`s, one vector per shape. Using this training data, we optimize both latent vectors and neural network parameters using backpropagation and gradient descent.

The tricky situation here is to use this model at inference. If you have previous experience in embeddings used in natural language processing (NLP), you would note that the situation here is pretty different. In NLP, each embedding vector represents a discrete element (often a character or a word). Both training and test data in a particular language is a sequence of these discrete elements, and a finite vocabulary can be assumed. In the DeepSDF case, infinite possible instances of a category, we can have infinitely many different objects. We, therefore, cannot assume there is a finite vocabulary of possible objects.

What instead can be done is to optimize a new latent vector of an upcoming shape. We freeze the network parameters, and simply using the fact that neural networks are differentiable with respect to their inputs, we optimize latent vectors using gradient-descent.

There are two very important points to make about the inference formulation. First, ground-truth point-sdf pairs are available in inference time as well! However, these ground-truth points are available only for a sample of points, and we want to obtain the continuous function for the whole shape. This detail is especially important for shape-completion, as we will see a little bit later. Second, inference requires an entire training process with multiple backward passes of the network. This makes the model very slow at inference, probably the biggest downside of DeepSDF approach.

### Dataset Preparation
Of course, having a model is not enough to do deep learning, we also need data! Similar to most other works in 3D deep learning, they use a Computer-Aided Design (CAD) model dataset to obtain clean 3D shapes. In particular, they use the ShapeNet dataset, one of the most popular CAD model datasets (together with ModelNet). They normalize shapes into the unit sphere and ensure all shapes have a canonical pose (there are no upside-down chairs!).

SDFs can be obtained from meshes, and as a synthetic dataset, ShapeNet consists of meshes. However, instead of using meshes, they sampled a lot of points from different viewpoints and sampled SDFs around them! It is ironic that they technically used a point-based representation for data preparation, but we should note that the task is sampling data, not rendering. Since authors want watertight (closed) surfaces being produced, they also employed a heuristic to eliminate non-watertight shapes. They do so by counting triangles whose different sides are observed from opposite views and eliminate the shapes where this count is large. This elimination is the major advantage of using a multi-view approach.


### Model and Training Details
In most of their experiments, authors used an 8-layer Fully Connected Network with a residual connection from input to layer 4. They use standard ReLU activations, but not-so-standard weight normalization instead of more popular batch normalization. The advantage of weight normalization over batch normalization is that it doesn't correlate with different shapes with each other. Adam optimizer with standard parameters is used. The truncation parameter is set to 0.1, and latent vectors are initialized from a normal distribution with small variance (reportedly 0.001). Tanh activation is used to ensure outputs are normalized. Training took 8 hours on 8 GPUs.

