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

I will discuss further details on how they formulate the problem and train their models exactly. Before that, let's take a small break from DeepSDF and review shortly some related works that will serve our understanding.


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
