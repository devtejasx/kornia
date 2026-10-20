`unsharp_mask` and `UnsharpMask` no longer raise inside an autocast region for a float32 image; the blur is cast back to the image dtype before it is combined with the image.
