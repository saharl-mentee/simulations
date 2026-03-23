SetFactory("OpenCASCADE");
Merge "standard_fin.STEP";
//+
Physical Surface("base", 13) = {1};
//+
Physical Surface("conv", 14) = {5, 4, 2, 3, 6};
//+
Physical Volume("cond", 15) = {1};
//+
Transfinite Curve {4, 2, 9, 6} = 10 Using Progression 1;
//+
Transfinite Curve {1, 3, 8, 10} = 5 Using Progression 1;
//+
Transfinite Curve {11, 12, 7, 5} = 20 Using Progression 1;
//+
Transfinite Surface {1, 2, 3, 4, 5, 6};
//+
Transfinite Surface {1};
//+
Recombine Surface {1, 2, 3, 4, 5, 6};
//+
Transfinite Volume{1};
