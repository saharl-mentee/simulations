SetFactory("OpenCASCADE");
Merge "fin_asm2.STEP";

//+
Coherence;

//+
Physical Volume("fin", 742) = {3, 2};
//+
Physical Volume("base", 743) = {1};
//+
Physical Surface("convection", 744) = {87, 111, 136, 82, 138, 137, 134, 192, 143, 105, 80, 88, 110, 248, 145, 189, 142, 141, 81, 250, 225, 135, 144, 133, 252, 227, 146, 140, 254, 229, 256, 231, 139, 258, 233, 187, 186, 173, 188, 190, 191, 260, 171, 235, 195, 193, 174, 194, 226, 247, 262, 170, 237, 198, 196, 175, 197, 228, 249, 264, 169, 239, 201, 199, 176, 200, 230, 251, 266, 168, 241, 204, 202, 177, 203, 232, 253, 268, 167, 243, 207, 205, 178, 206, 234, 255, 184, 166, 245, 210, 208, 179, 209, 236, 257, 165, 213, 211, 180, 212, 238, 259, 164, 216, 214, 181, 215, 240, 261, 163, 219, 217, 182, 218, 242, 263, 162, 222, 220, 183, 221, 149, 244, 265, 161, 224, 155, 223, 269, 246, 267, 160, 150, 148, 147, 127, 109, 112, 85, 126, 93, 125, 159, 154, 130, 99, 128, 157, 96, 84, 100, 95, 79, 92, 129, 98, 124, 116, 151, 120, 106, 119, 86, 97, 114, 123, 121, 101, 104, 89, 90, 103, 122, 102, 107, 132, 94, 91, 113, 117, 115, 118, 108};
//+
Physical Surface("insulation", 745) = {83, 131, 152, 156, 153, 158};
//+
Physical Surface("heat_source", 746) = {172, 185};

// 2. Clear Global Mesh Defaults
// These tell Gmsh: "Don't decide sizes for me."
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
// Mesh.MeshSizeFromVolume = 1;



//+
Field[1] = Constant;
//+
Field[1].VIn = 1.5;
//+
Field[1].VOut = 100;
//+
Field[1].VolumesList = {1};

//+
Field[2] = Constant;
//+
Field[2].VIn = 0.7;
//+
Field[2].VOut = 100;
//+
Field[2].VolumesList = {2};

//+
Field[3] = Constant;
//+
Field[3].VIn = 0.5;
//+
Field[3].VOut = 100;
//+
Field[3].VolumesList = {3};

//+
Field[4] = Min;
//+
Field[4].FieldsList = {1, 2, 3};
//+
Background Field = 4;

//+
Mesh.MeshSizeFromPoints = 0;
//+
Mesh.MeshSizeFromCurvature = 0;

