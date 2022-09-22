
            elem_size = DefineNumber[ 0.0333333, Name "Parameters/elem_size" ];
            L         = DefineNumber[ 1, Name "Parameters/L"];
            H         = DefineNumber[ 1, Name "Parameters/H"];
            Point(1) = {0, 0, 0, elem_size};
            Point(2) = {L, 0, 0, elem_size};
            Point(3) = {L, H, 0, elem_size};
            Point(4) = {0, H, 0, elem_size};
            Line(1) = {1, 2}; 
            Line(2) = {2, 3}; 
            Line(3) = {3, 4}; 
            Line(4) = {4, 1};
            Line Loop(1) = {1, 2, 3, 4};
            Plane Surface(1) = {1};
            Transfinite Surface{1} AlternateRight;
            Mesh.MshFileVersion = 2.0;
            