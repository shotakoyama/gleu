import numpy as np
from .load import load_dhrn_data, load_hdrn_data
from .score import rn_sent_accum_to_gleu
from .util import argmax
from .verbose import SentNVerbose
from .result import simple_result, table_result


def sent_main(args):
    if args.verbose:
        sent_verbose(args)
    else:
        sent_simple(args)


def sent_simple(args):
    _, _, _, dr_rlen, dh_hlen, dhrn_sacc = load_dhrn_data(args)

    for d, hrn_sacc in enumerate(dhrn_sacc):
        h_score = [
            rn_sent_accum_to_gleu(
                rn_sacc,
                dr_rlen[d],
                dh_hlen[d, h],
                args.max)
            for h, rn_sacc
            in enumerate(hrn_sacc)]
        results = [
            simple_result(score, args.digit)
            for score
            in h_score]
        line = '\t'.join(results)
        print(line)


def sent_verbose(args):
    s_dat, rs_dat, hs_dat, dr_rlen, dh_hlen, dhrn_sacc = load_dhrn_data(args)

    for d, hrn_sacc in enumerate(dhrn_sacc):
        for h, rn_sacc in enumerate(hrn_sacc):
            r_snvb = [
                SentNVerbose(
                    n_sacc,
                    dr_rlen[d, r],
                    dh_hlen[d, h])
                for r, n_sacc
                in enumerate(rn_sacc)]
            rmax = argmax(r_snvb)
            for r, snvb in enumerate(r_snvb):
                table = table_result(snvb, args.digit)
                chosen = '*' if r == rmax else ' '
                print(f'S-{d+1}   \t{s_dat[d]}')
                print(f'H-{d+1}-{h+1} \t{hs_dat[d][h]}')
                print(f'R-{d+1}-{r+1}{chosen}\t{rs_dat[d][r]}')
                print(table)


def mean_main(args):
    _, _, _, dr_rlen, dh_hlen, hdrn_sacc = load_hdrn_data(args)

    for h, drn_sacc in enumerate(hdrn_sacc):
        d_score = [
            rn_sent_accum_to_gleu(
                rn_sacc,
                dr_rlen[d],
                dh_hlen[d, h],
                args.max)
            for d, rn_sacc
            in enumerate(drn_sacc)]
        line = simple_result(np.mean(d_score), args.digit)
        print(args.hyp_path_list[h] + '\t' + line)

