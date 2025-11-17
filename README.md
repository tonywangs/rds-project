# rds-project

Random dot stereogram generator that creates red/cyan anaglyph images. Two implementations are included: one that starts with identical dot layers and only modifies the red layer inside the shape, and another that uses independent dot assignment outside the shape with stereo correspondence enforced inside. Both generate square ring shapes and output separate left/right eye images plus merged versions. The shape functions are straightforward to swap out if you want to use different shapes. Requires numpy and pillow.
