SetFactory("OpenCASCADE");
Merge "fin_asm.STEP";

// to solve boundary mesh issues caused by the STEP
BooleanFragments{ Volume{1,2,3}; }{ }

// v() = Volume{:}; // Get all volumes
// BooleanFragments{ Volume{v()}; }{ } // Ensures shared interfaces
// Recursive Color Red{ Volume{1}; }   // Debug: Check if Vol 1 is what you think
// Recursive Color Green{ Volume{2}; }
// Recursive Color Blue{ Volume{3}; }

//+
Coherence;

//+
// Physical Surface("new_convection", 604) = {113, 169, 110, 171, 32, 146, 173, 148, 175, 56, 150, 58, 3, 177, 63, 8, 26, 1, 152, 179, 62, 54, 57, 154, 181, 108, 31, 109, 65, 107, 94, 156, 112, 116, 64, 183, 114, 115, 111, 92, 95, 158, 168, 119, 61, 185, 117, 118, 147, 2, 91, 55, 96, 160, 170, 122, 9, 187, 120, 121, 149, 90, 53, 97, 162, 172, 125, 66, 60, 189, 123, 124, 151, 89, 98, 164, 174, 128, 126, 127, 153, 88, 99, 166, 176, 131, 129, 130, 155, 87, 100, 178, 134, 105, 132, 133, 157, 86, 101, 180, 137, 59, 135, 136, 159, 85, 102, 182, 140, 138, 139, 161, 84, 103, 184, 143, 141, 142, 163, 83, 104, 186, 145, 144, 202, 207, 165, 82, 188, 167, 81, 196, 199, 208, 69, 33, 14, 205, 70, 80, 206, 193, 49, 74, 20, 77, 48, 67, 68, 5, 203, 30, 17, 6, 16, 204, 21, 47, 45, 19, 46, 71, 18, 13, 37, 41, 22, 11, 7, 24, 10, 27, 40, 23, 44, 35, 52, 25, 42, 15, 43, 12, 28, 38, 36, 34, 39, 29};
//+
// Physical Surface("conduction", 603) = {209, 210};
//+
// Physical Volume("base", 601) = {1};
//+
// Physical Volume("fin", 602) = {2, 3};

// 2. Clear Global Mesh Defaults
// These tell Gmsh: "Don't decide sizes for me."
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
// Mesh.MeshSizeFromVolume = 1;

//+
Field[1] = Constant;
//+
Field[1].VIn = 10;
//+
Field[1].VOut = 20;
//+
Field[1].VolumesList = {1};

//+
Field[2] = Constant;
//+
Field[2].VIn = 2;
//+
Field[2].VOut = 3;
//+
Field[2].VolumesList = {2};

//+
Field[3] = Constant;
//+
Field[3].VIn = 0.5;
//+
Field[3].VOut = 1;
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

//+
Physical Surface("new_new_convection", 605) = {32, 8, 56, 3, 58, 57, 26, 63, 54, 1, 113, 31, 9, 62, 65, 61, 2, 110, 169, 64, 55, 53, 146, 171, 60, 66, 148, 173, 150, 175, 59, 152, 108, 177, 107, 109, 112, 94, 111, 92, 116, 154, 114, 179, 115, 168, 95, 147, 91, 119, 156, 117, 181, 118, 170, 96, 149, 90, 122, 158, 120, 183, 121, 172, 97, 151, 89, 125, 160, 123, 185, 124, 174, 98, 153, 88, 128, 162, 126, 187, 127, 176, 99, 155, 87, 131, 105, 164, 129, 189, 130, 178, 100, 157, 86, 134, 166, 132, 210, 133, 180, 101, 159, 85, 137, 135, 136, 182, 102, 161, 84, 140, 138, 139, 184, 103, 163, 83, 209, 143, 141, 142, 186, 104, 165, 82, 145, 144, 207, 202, 188, 167, 69, 81, 196, 199, 70, 67, 68, 33, 48, 208, 30, 6, 47, 205, 206, 80, 74, 46, 203, 193, 77, 14, 49, 5, 16, 20, 204, 17, 21, 45, 19, 71, 13, 7, 37, 18, 41, 27, 40, 44, 22, 35, 11, 10, 24, 42, 25, 23, 43, 52, 15, 28, 12, 34, 38, 36, 39, 29};
