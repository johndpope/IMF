# implicit-motion-function

see current training here 
https://wandb.ai/snoozie/IMF?nw=nwusersnoozie


The frame_skip in VideoDataset - is only at 1. 


![Image](ok.png)





**Intuition:**
The Implicit Motion Function (IMF) is like a sophisticated video compression and reconstruction system that works by understanding the essence of motion between video frames, rather than just tracking pixel movements.

Analogy: The Art Gallery Curator

Imagine you're a curator at an art gallery that specializes in "living paintings" - artworks that change over time. Your job is to efficiently store these paintings and recreate them on demand. Here's how the IMF works, explained through this analogy:

1. Dense Feature Encoder (The Detailed Sketcher):
   This is like an artist who quickly sketches the key details of each painting. Instead of copying every brush stroke, they capture the essence - the shapes, colors, and textures that make the painting unique.

2. Latent Token Encoder (The Essence Capturer):
   This is like a perfumer who can capture the "essence" of a scent in a tiny vial. For each painting, they distill its core characteristics into a small, concentrated form. This essence represents not just how the painting looks, but how it tends to change over time.

3. Latent Token Decoder (The Essence Interpreter):
   This is like a magician who can take the vial of essence and use it to conjure up a ghostly image of the original painting. It's not perfect, but it captures the key features and potential changes.

4. Implicit Motion Alignment (The Change Predictor):
   This is like a psychic who can look at two versions of a painting and understand exactly how one transformed into the other. They don't just see the surface changes, but understand the underlying patterns and rules of how the painting evolves.

5. Frame Decoder (The Reconstruction Artist):
   This is the artist who takes all the information from the others - the detailed sketch, the essence, and the understanding of changes - and recreates the full painting in its new state.

Now, imagine you're storing and recreating a living painting:

1. You start with two versions of the painting - the "reference" (how it looked before) and the "current" (how it looks now).

2. The Detailed Sketcher quickly captures the key features of the reference painting.

3. The Essence Capturer distills both the reference and current paintings into their essences.

4. The Essence Interpreter takes these essences and creates ghostly versions of both paintings.

5. The Change Predictor compares these ghostly versions and figures out exactly how the painting changed.

6. Finally, the Reconstruction Artist uses the detailed sketch of the reference, combined with the understanding of changes, to recreate the current painting.

The magic of this system is that you don't need to store every detail of how the painting changes. Instead, you store the essence and learn the rules of how it evolves. This allows for efficient storage (compression) and accurate recreation (reconstruction) of the living paintings.

In the context of video:
- The "living paintings" are video frames.
- The "essence" is the compact latent representation.
- The "changes" are the motions and transformations between frames.

By understanding the essence of each frame and the patterns of how frames change, the IMF can efficiently compress video information and accurately reconstruct frames, even handling complex motions and new content that traditional methods might struggle with.
