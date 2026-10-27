#pragma once

#include <cstdint>

namespace strata::kernels {

/// FlashHead (tools/flashhead.py): the MTP draft head as retrieval over the whole vocabulary.  For each of n_rows
/// rows, ids[row * m] (m = n_static + probes * b) gets the n_static static token ids, then the b token ids of each of
/// the `probes` clusters with the highest scores[row * n_clusters + c] (c2t: n_clusters * b ids, cluster c's at
/// c * b).  The picked clusters keep their index order (those above the cut first, then the tied ones), so the list
/// is the same on every run.  One block a row; probes <= 1024, probes <= n_clusters.  Graph-capturable.
void flashhead_select(const float* scores, int n_rows, int n_clusters, int probes, const int32_t* c2t, int b,
                      const int32_t* stat, int n_static, int32_t* ids, void* stream);

}  // namespace strata::kernels
