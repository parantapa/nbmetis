// METIS bindings for Python using Nanobind

#include <metis.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;
using namespace nb::literals;

using IdxA = nb::ndarray<idx_t, nb::shape<-1>, nb::device::cpu>;
using RealA = nb::ndarray<real_t, nb::shape<-1>, nb::device::cpu>;

// (Return value, edgecut)
using PartGraphRet = std::pair<int, idx_t>;

PartGraphRet PartGraph(idx_t nvtxs, idx_t ncon, idx_t nparts, IdxA xadj,
                       IdxA adjncy, IdxA vwgt, IdxA vsize, IdxA adjwgt,
                       RealA tpwgts, RealA ubvec, IdxA options, IdxA part,
                       int part_kway) {
  idx_t edgecut = 0;

  idx_t *nvtxs_p = &nvtxs;
  idx_t *ncon_p = &ncon;

  idx_t *xadj_p = xadj.data();
  idx_t *adjncy_p = adjncy.data();

  idx_t *vwgt_p = vwgt.size() == 0 ? nullptr : vwgt.data();
  idx_t *vsize_p = vsize.size() == 0 ? nullptr : vsize.data();
  idx_t *adjwgt_p = adjwgt.size() == 0 ? nullptr : adjwgt.data();

  idx_t *nparts_p = &nparts;

  real_t *tpwgts_p = tpwgts.size() == 0 ? nullptr : tpwgts.data();
  real_t *ubvec_p = ubvec.size() == 0 ? nullptr : ubvec.data();
  idx_t *options_p = options.size() == 0 ? nullptr : options.data();

  idx_t *edgecut_p = &edgecut;
  idx_t *part_p = part.data();

  int ret;
  if (part_kway) {
    ret = METIS_PartGraphKway(nvtxs_p, ncon_p, xadj_p, adjncy_p, vwgt_p,
                              vsize_p, adjwgt_p, nparts_p, tpwgts_p, ubvec_p,
                              options_p, edgecut_p, part_p);
  } else {
    ret = METIS_PartGraphRecursive(nvtxs_p, ncon_p, xadj_p, adjncy_p, vwgt_p,
                                   vsize_p, adjwgt_p, nparts_p, tpwgts_p,
                                   ubvec_p, options_p, edgecut_p, part_p);
  }

  return PartGraphRet(ret, edgecut);
}

int NodeND(idx_t nvtxs, IdxA xadj, IdxA adjncy, IdxA vwgt, IdxA options,
           IdxA perm, IdxA iperm) {

  // std::cout << "nvtxs = " << nvtxs << "\n";
  // std::cout << "xadj.size() = " << xadj.size() << "\n";
  // std::cout << "adjncy.size() = " << adjncy.size() << "\n";
  // std::cout << "vwgt.size() = " << vwgt.size() << "\n";
  // std::cout << "options.size() = " << options.size() << "\n";
  // std::cout << "perm.size() = " << perm.size() << "\n";
  // std::cout << "iperm.size() = " << iperm.size() << "\n";

  idx_t *nvtxs_p = &nvtxs;
  idx_t *xadj_p = xadj.data();
  idx_t *adjncy_p = adjncy.data();
  idx_t *vwgt_p = vwgt.size() == 0 ? nullptr : vwgt.data();
  idx_t *options_p = options.size() == 0 ? nullptr : options.data();

  idx_t *perm_p = perm.data();
  idx_t *iperm_p = iperm.data();

  // for (int i = 0; i < nvtxs; i++) {
  //     int start = xadj_p[i];
  //     int end = xadj_p[i+1];
  //     for (int j = start; j < end; j++) {
  //         std::cout << "edge" << i << " " << adjncy_p[j] << "\n";
  //     }
  // }

  return METIS_NodeND(nvtxs_p, xadj_p, adjncy_p, vwgt_p, options_p, perm_p,
                      iperm_p);
}

int SetDefaultOptions(IdxA options) {
  return METIS_SetDefaultOptions(options.data());
}

#define EXPORT_CONSTANT(m, a) m.attr(#a) = int(METIS_##a)

NB_MODULE(_nbmetis, m) {
  m.doc() = "Yet anothor Python binding for METIS";

  m.attr("SIZEOF_IDX_T") = sizeof(idx_t);
  m.attr("SIZEOF_REAL_T") = sizeof(real_t);

  EXPORT_CONSTANT(m, VER_MAJOR);
  EXPORT_CONSTANT(m, VER_MINOR);
  EXPORT_CONSTANT(m, VER_SUBMINOR);

  EXPORT_CONSTANT(m, NOPTIONS);

  EXPORT_CONSTANT(m, OK);
  EXPORT_CONSTANT(m, ERROR_INPUT);
  EXPORT_CONSTANT(m, ERROR_MEMORY);
  EXPORT_CONSTANT(m, ERROR);

  EXPORT_CONSTANT(m, OP_PMETIS);
  EXPORT_CONSTANT(m, OP_KMETIS);
  EXPORT_CONSTANT(m, OP_OMETIS);

  EXPORT_CONSTANT(m, OPTION_PTYPE);
  EXPORT_CONSTANT(m, OPTION_OBJTYPE);
  EXPORT_CONSTANT(m, OPTION_CTYPE);
  EXPORT_CONSTANT(m, OPTION_IPTYPE);
  EXPORT_CONSTANT(m, OPTION_RTYPE);
  EXPORT_CONSTANT(m, OPTION_DBGLVL);
  EXPORT_CONSTANT(m, OPTION_NITER);
  EXPORT_CONSTANT(m, OPTION_NCUTS);
  EXPORT_CONSTANT(m, OPTION_SEED);
  EXPORT_CONSTANT(m, OPTION_NO2HOP);
  EXPORT_CONSTANT(m, OPTION_MINCONN);
  EXPORT_CONSTANT(m, OPTION_CONTIG);
  EXPORT_CONSTANT(m, OPTION_COMPRESS);
  EXPORT_CONSTANT(m, OPTION_CCORDER);
  EXPORT_CONSTANT(m, OPTION_PFACTOR);
  EXPORT_CONSTANT(m, OPTION_NSEPS);
  EXPORT_CONSTANT(m, OPTION_UFACTOR);
  EXPORT_CONSTANT(m, OPTION_NUMBERING);

  EXPORT_CONSTANT(m, PTYPE_RB);
  EXPORT_CONSTANT(m, PTYPE_KWAY);

  EXPORT_CONSTANT(m, GTYPE_DUAL);
  EXPORT_CONSTANT(m, GTYPE_NODAL);

  EXPORT_CONSTANT(m, CTYPE_RM);
  EXPORT_CONSTANT(m, CTYPE_SHEM);

  EXPORT_CONSTANT(m, IPTYPE_GROW);
  EXPORT_CONSTANT(m, IPTYPE_RANDOM);
  EXPORT_CONSTANT(m, IPTYPE_EDGE);
  EXPORT_CONSTANT(m, IPTYPE_NODE);
  EXPORT_CONSTANT(m, IPTYPE_METISRB);

  EXPORT_CONSTANT(m, RTYPE_FM);
  EXPORT_CONSTANT(m, RTYPE_GREEDY);
  EXPORT_CONSTANT(m, RTYPE_SEP2SIDED);
  EXPORT_CONSTANT(m, RTYPE_SEP1SIDED);

  EXPORT_CONSTANT(m, DBG_INFO);
  EXPORT_CONSTANT(m, DBG_TIME);
  EXPORT_CONSTANT(m, DBG_COARSEN);
  EXPORT_CONSTANT(m, DBG_REFINE);
  EXPORT_CONSTANT(m, DBG_IPART);
  EXPORT_CONSTANT(m, DBG_MOVEINFO);
  EXPORT_CONSTANT(m, DBG_SEPINFO);
  EXPORT_CONSTANT(m, DBG_CONNINFO);
  EXPORT_CONSTANT(m, DBG_CONTIGINFO);
  EXPORT_CONSTANT(m, DBG_MEMORY);

  EXPORT_CONSTANT(m, OBJTYPE_CUT);
  EXPORT_CONSTANT(m, OBJTYPE_VOL);
  EXPORT_CONSTANT(m, OBJTYPE_NODE);

  m.def("PartGraph", &PartGraph, "nvtxs"_a, "ncon"_a, "nparts"_a, "xadj"_a,
        "adjncy"_a, "vwgt"_a, "vsize"_a, "adjwgt"_a, "tpwgts"_a, "ubvec"_a,
        "options"_a, "part"_a, "part_kway"_a,
        "Is used to partition a graph into k parts using either multilevel "
        "recursive bisection or multilevel k-way partitioning.");

  m.def("NodeND", &NodeND, "nvtxs"_a, "xadj"_a, "adjncy"_a, "vwgt"_a,
        "options"_a, "perm"_a, "iperm"_a,
        "This function computes fill reducing orderings of sparse matrices "
        "using the multilevel nested dissection algorithm.");

  m.def("SetDefaultOptions", &SetDefaultOptions, "options"_a,
        "Initializes the options array into its default values.");
}
