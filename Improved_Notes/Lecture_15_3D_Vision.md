# CS231n (Spring 2025) — Lecture 15: 3D Vision

> **Instructor:** Jiajun Wu
> **Date:** May 22, 2025

---

## 1. Why 3D Matters

2D images are projections of a 3D world. Understanding 3D structure is essential for:
- Robotics (grasping, navigation)
- Augmented/Virtual Reality
- Autonomous driving
- Content creation (games, movies)

The core challenge: how to **represent**, **reconstruct**, and **reason about** 3D geometry computationally.

---

## 2. 3D Shape Representations

### The Two Families

| | Explicit | Implicit |
|------|----------|----------|
| **Definition** | All points are given directly | Points satisfy a specified relationship (e.g., $f(x, y, z) = 0$) |
| **Sampling** | Easy — just plug in parameter values | Can be hard — need to find where $f = 0$ |
| **Inside/Outside Test** | Hard — need to compute winding numbers or ray casts | Easy — just evaluate $f(x, y, z)$ and check the sign |
| **Example** | Mesh, point cloud, parametric surface | Signed distance function, algebraic surface |

### Non-Parametric Representations

**Point Clouds:**
- Simplest representation: an unordered set of $(x, y, z)$ coordinates (possibly with normals)
- Often results from 3D scanners, depth cameras, or LiDAR
- **Advantages:** Simple, can represent any geometry
- **Limitations:** No connectivity information (hard to render smoothly), no topology, difficult to edit or simplify

**Polygon Meshes:**
- Boundary representation of objects: vertices + faces (typically triangles)
- **Advantages:** Standard in graphics (GPUs are optimized for triangles), supports subdivision/simplification/regularization, explicit topology
- **Limitations:** Fixed topology (changing genus requires remeshing), can be irregular
- Example: Digital Michelangelo Project — 28 million vertices, 56 million triangles
- Example: Google Earth — trillions of triangles from satellite photography

**Mesh Operations:**
| Operation | Purpose |
|-----------|---------|
| **Subdivision** | Increase resolution via interpolation |
| **Simplification** | Decrease resolution while preserving appearance |
| **Regularization** | Improve mesh quality (more uniform sampling) |

### Parametric Representations

**Parametric Curves:** Define a curve as the range of a function $f: \mathbb{R} \rightarrow \mathbb{R}^3$. Example — a circle in 2D: $f(t) = (r\cos t, r\sin t)$.

**Parametric Surfaces:** $f: \mathbb{R}^2 \rightarrow \mathbb{R}^3$. Example — a sphere: $f(u, v) = (r\cos u \sin v, r\sin u \sin v, r\cos v)$.

**Bezier Curves & Surfaces:** Piecewise polynomial curves controlled by "control points" — the foundation of modern computer graphics and CAD.

### Implicit Representations

**Algebraic Surfaces:** The zero set of a polynomial in $x, y, z$. Simple shapes (spheres, cylinders) have exact algebraic forms, but complex shapes require high-degree polynomials.

**Constructive Solid Geometry (CSG):** Combine simple implicit primitives via Boolean operations (union $\cup$, intersection $\cap$, difference $\setminus$). Produces complex geometry from simple building blocks.

**Signed Distance Functions (SDF):** For any point in space, $f(x, y, z)$ gives the signed distance to the nearest surface (negative = inside, positive = outside, zero = on surface). Enables smooth blending between shapes.

---

## 3. Neural Implicit Representations

Instead of hand-crafting $f(x, y, z)$, use a **neural network** to represent the implicit function:

$$f_\theta: (x, y, z) \rightarrow \text{[occupancy, SDF value, density, ...]}$$

**Why neural?** A neural network is a universal function approximator — it can represent arbitrarily complex geometry with a fixed architecture.

### Neural Radiance Fields (NeRF)

Represent a 3D scene as a continuous volumetric field:

$$F_\theta: (x, y, z, \theta, \phi) \rightarrow (R, G, B, \sigma)$$

- **Input:** 3D position $(x, y, z)$ plus viewing direction $(\theta, \phi)$
- **Output:** Emitted color $(R, G, B)$ and volume density $\sigma$ at that point

**Training:** Given a set of 2D images from known camera poses, render each pixel by tracing a ray through the volume and compositing colors, then compare against the ground-truth pixel color.

**Rendering Equation:** For a ray $r(t) = o + td$, the expected color is:

$$C(r) = \int_{t_n}^{t_f} T(t) \cdot \sigma(r(t)) \cdot c(r(t), d) \, dt$$

where $T(t) = \exp(-\int_{t_n}^t \sigma(r(s)) \, ds)$ is the accumulated transmittance.

**Key properties:**
- Produces photo-realistic novel views
- Compact: one MLP represents an entire scene
- Continuous: can render at any resolution

### Beyond NeRF

- **Instant NGP:** Hash-grid encoding → orders of magnitude faster training
- **3D Gaussian Splatting:** Explicit 3D Gaussians as primitives → real-time rendering with NeRF-quality results
- **Mesh extraction:** Marching Cubes on the learned implicit field → convert to standard mesh format

---

## Core Takeaways

1. 3D representations fall into **explicit** (points, meshes, parametric surfaces) and **implicit** (SDF, algebraic surfaces, neural fields) — each with complementary strengths.
2. **Explicit = easy sampling, hard queries.** **Implicit = easy queries, hard sampling.**
3. **Meshes** are the workhorse of computer graphics; **point clouds** are the raw output of most 3D sensors.
4. **Neural implicit representations** (NeRF, etc.) use neural networks to represent continuous 3D geometry — producing photo-realistic novel views.
5. The field is rapidly evolving: from slow MLP-based rendering (NeRF) to real-time explicit representations (3D Gaussian Splatting).

---

*Notes compiled from CS231n Spring 2025 Lecture 15 slides and course materials.*
