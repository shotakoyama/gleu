from .aggreg import make_dhrn_accum, make_hdrn_accum
from .count import make_dx_xlen


def load_corpora(args):
    s_dat = load_text(args.source_path)
    rs_dat = load_paired_texts(args.ref_path_list)
    hs_dat = load_paired_texts(args.hyp_path_list)
    assert len(s_dat) == len(rs_dat) == len(hs_dat)
    dr_rlen = make_dx_xlen(rs_dat)
    dh_hlen = make_dx_xlen(hs_dat)
    return s_dat, rs_dat, hs_dat, dr_rlen, dh_hlen


def load_text(path):
    with open(path) as f:
        data = [x.rstrip('\n') for x in f]
    return data


def load_paired_texts(path_list):
    data = [load_text(path) for path in path_list]
    return list(zip(*data))


def load_hdrn_data(args):
    s_dat, rs_dat, hs_dat, dr_rlen, dh_hlen = load_corpora(args)
    h_dats = list(zip(*hs_dat))
    hrnd_sacc = make_hdrn_accum(args.n, s_dat, rs_dat, h_dats)
    return s_dat, rs_dat, hs_dat, dr_rlen, dh_hlen, hrnd_sacc


def load_dhrn_data(args):
    s_dat, rs_dat, hs_dat, dr_rlen, dh_hlen = load_corpora(args)
    dhrn_sacc = make_dhrn_accum(args.n, s_dat, rs_dat, hs_dat)
    return s_dat, rs_dat, hs_dat, dr_rlen, dh_hlen, dhrn_sacc

