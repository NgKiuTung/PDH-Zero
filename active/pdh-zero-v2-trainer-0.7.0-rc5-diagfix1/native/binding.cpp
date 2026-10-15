// Copyright 2026 PDH-Zero contributors. Project-local research use.
// NOTE: A single call computes a batch. Buffer owners stay alive across GIL release.
#include "features.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
namespace py=pybind11;
using Integers=py::array_t<std::int32_t, py::array::c_style>;

py::tuple Encode(const Integers& states,const Integers& actions,const Integers& offsets) {
  if(states.ndim()!=2 || states.shape(1)!=pdh::kPackedState || states.shape(0)<1
     || states.shape(0)>4096 || actions.ndim()!=2 || actions.shape(1)!=pdh::kPackedAction
     || actions.shape(0)<1 || actions.shape(0)>262144 || offsets.ndim()!=1
     || offsets.shape(0)!=states.shape(0)+1)
    throw std::invalid_argument("native.input.shape");
  auto b=states.shape(0); auto a=actions.shape(0);
  const auto* cuts=offsets.data();
  if(cuts[0]!=0 || cuts[b]!=a) throw std::invalid_argument("native.input.offset");
  for(py::ssize_t i=0;i<b;++i)
    if(cuts[i]<0 || cuts[i+1]<=cuts[i] || cuts[i+1]>a || cuts[i+1]-cuts[i]>64)
      throw std::invalid_argument("native.input.offset");
  py::array_t<double> obs({b,static_cast<py::ssize_t>(pdh::kObservation)});
  py::array_t<double> features({a,static_cast<py::ssize_t>(pdh::kActionFeatures)});
  py::array_t<double> scores(a);
  const auto* s=states.data(); const auto* acts=actions.data();
  double* o=obs.mutable_data(); double* f=features.mutable_data(); double* h=scores.mutable_data();
  {
    py::gil_scoped_release release;
    for(py::ssize_t i=0;i<b;++i) {
      const auto* row=s+i*pdh::kPackedState;
      auto ctx=pdh::Parse(row);
      pdh::Observation(row,ctx,o+i*pdh::kObservation);
      for(int j=cuts[i];j<cuts[i+1];++j)
        h[j]=pdh::ActionFeatures(ctx,row,acts+j*pdh::kPackedAction,f+j*pdh::kActionFeatures);
    }
  }
  return py::make_tuple(obs,features,scores);
}
PYBIND11_MODULE(_pdh_native,m) {
  m.attr("SCHEMA_VERSION")=1;
  m.def("encode_batch",&Encode,py::arg("states").noconvert(),
        py::arg("actions").noconvert(),py::arg("offsets").noconvert());
}
