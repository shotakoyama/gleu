import numpy as np
from .load import load_dhrn_data, load_hdrn_data
from .score import rn_accum_to_gleu
from .util import argmax
from .verbose import NVerbose
from .result import simple_result, table_result


def sent_main(args):
    if args.verbose:
        sent_verbose(args)
    else:
        sent_simple(args)


def sent_simple(args):
    _, _, _, dr_rlen, dh_hlen, dhrn_accum = load_dhrn_data(args)

    for d, hrn_accum in enumerate(dhrn_accum):
        h_score = [
            rn_accum_to_gleu(
                rn_accum,
                dr_rlen[d],
                dh_hlen[d, h],
                args.max)
            for h, rn_accum
            in enumerate(hrn_accum)]
        results = [
            simple_result(score, args.digit)
            for score
            in h_score]
        line = '\t'.join(results)
        print(line)


def sent_verbose(args):
    s_dat, rs_dat, hs_dat, dr_rlen, dh_hlen, dhrn_accum = load_dhrn_data(args)

    for d, hrn_accum in enumerate(dhrn_accum):
        for h, rn_accum in enumerate(hrn_accum):
            r_nvb = [
                NVerbose(
                    n_accum,
                    dr_rlen[d, r],
                    dh_hlen[d, h])
                for r, n_accum
                in enumerate(rn_accum)]
            rmax = argmax(r_nvb)
            for r, nvb in enumerate(r_nvb):
                table = table_result(nvb, args.digit)
                chosen = '*' if r == rmax else ' '
                print(f'S-{d+1}   \t{s_dat[d]}')
                print(f'H-{d+1}-{h+1} \t{hs_dat[d][h]}')
                print(f'R-{d+1}-{r+1}{chosen}\t{rs_dat[d][r]}')
                print(table)


def mean_main(args):
    _, _, _, dr_rlen, dh_hlen, hdrn_accum = load_hdrn_data(args)

    for h, drn_accum in enumerate(hdrn_accum):
        d_score = [
            rn_accum_to_gleu(
                rn_accum,
                dr_rlen[d],
                dh_hlen[d, h],
                args.max)
            for d, rn_accum
            in enumerate(drn_accum)]
        line = simple_result(np.mean(d_score), args.digit)
        print(args.hyp_path_list[h] + '\t' + line)

