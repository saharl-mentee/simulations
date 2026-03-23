SetFactory("OpenCASCADE");
Merge "fin_asm.STEP";

//+
Coherence;

//+
Physical Volume("fin", 536) = {3, 2};
//+
Physical Volume("base", 537) = {1};
//+
Physical Surface("convection", 538) = {32, 8, 56, 3, 58, 57, 26, 63, 54, 1, 31, 113, 9, 62, 65, 61, 2, 110, 169, 64, 55, 53, 146, 171, 60, 66, 148, 173, 150, 175, 59, 152, 108, 177, 107, 109, 112, 94, 111, 92, 116, 154, 114, 179, 115, 168, 95, 147, 91, 119, 156, 117, 181, 118, 170, 96, 149, 90, 122, 158, 120, 183, 121, 172, 97, 151, 89, 125, 160, 123, 185, 124, 174, 98, 153, 88, 128, 162, 126, 187, 127, 176, 99, 155, 87, 131, 105, 164, 129, 189, 130, 178, 100, 157, 86, 134, 166, 132, 133, 180, 101, 159, 85, 137, 135, 136, 182, 102, 161, 84, 140, 138, 139, 184, 103, 163, 83, 143, 141, 142, 186, 104, 165, 82, 145, 144, 75, 188, 167, 69, 190, 81, 70, 67, 68, 33, 48, 76, 30, 73, 78, 6, 47, 51, 72, 80, 74, 46, 4, 77, 14, 49, 5, 16, 20, 50, 79, 17, 21, 45, 71, 19, 13, 7, 37, 18, 41, 27, 40, 44, 22, 35, 11, 10, 24, 42, 25, 23, 43, 52, 15, 28, 12, 34, 38, 36, 39, 29};

Physical Surface("fin_heat_source", 539) = {93, 106};

// 2. Clear Global Mesh Defaults
// These tell Gmsh: "Don't decide sizes for me."
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
// Mesh.MeshSizeFromVolume = 1;



//+
Field[1] = Constant;
//+
Field[1].VIn = 3;
//+
Field[1].VOut = 100;
//+
Field[1].VolumesList = {1};

//+
Field[2] = Constant;
//+
Field[2].VIn = 1;
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


