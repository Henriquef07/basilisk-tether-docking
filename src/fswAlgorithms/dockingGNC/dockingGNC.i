%module dockingGNC
%{
   #include "dockingGNC.h"
%}

%include "swig_conly_data.i"
%include "std_vector.i"

%template(DoubleVector) std::vector<double>;

%include "dockingGNC.h"