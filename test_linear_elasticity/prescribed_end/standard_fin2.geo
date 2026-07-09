SetFactory("OpenCASCADE");
Merge "standard_fin.STEP";
//+
Physical Surface("base", 13) = {1};
//+
Physical Surface("conv", 14) = {5, 2, 4, 6};
//+
Physical Surface("Temp", 16) = {3};
//+
Physical Volume("cond", 15) = {1};
//+
Point(9) = {180, 0, 25, 1.0};
//+
Physical Point("deflection", 18) = {9};
//+
Point{1} In Volume{the_volume_id};
//+
Physical Curve("edge", 19) = {9};
//+
Transfinite Curve {4, 2, 9, 6} = 31 Using Progression 1;
//+
Transfinite Curve {1, 3, 8, 10} = 15 Using Progression 1;
//+
Transfinite Curve {11, 12, 7, 5} = 60 Using Progression 1;
//+
Transfinite Surface {1, 2, 3, 4, 5, 6};
//+
Transfinite Surface {1};
//+
Recombine Surface {1, 2, 3, 4, 5, 6};
//+
Transfinite Volume{1};
//+
Show "*";


