import numpy as np
import random
from multiprocessing import Pool
from .load import load_hdrn_data
from .score import (
        drn_sent_accum_to_d_rmax,
        drn_sent_accum_to_n_data_accum,
        rindex_to_rhlen,
        n_accum_to_gleu)
from .util import make_id_rindex
from .verbose import CorpusNVerbose
from .result import simple_result, table_result


def corpus_main(args):
    if args.max:
        corpus_max(args)
    else:
        corpus_sampling(args)


def corpus_max(args):
    _, _, _, dr_rlen, dh_hlen, hdrn_sacc = load_hdrn_data(args)

    for h, drn_sacc in enumerate(hdrn_sacc):
        d_hlen = dh_hlen[:, h]
        d_rmax = drn_sent_accum_to_d_rmax(drn_sacc, dr_rlen, d_hlen)
        n_dacc = drn_sent_accum_to_n_data_accum(drn_sacc, d_rmax)
        rlen, hlen = rindex_to_rhlen(dr_rlen, d_hlen, d_rmax)

        if args.verbose:
            cnvb = CorpusNVerbose(n_dacc, rlen, hlen)
            table = table_result(cnvb, args.digit)
            print(args.hyp_path_list[h])
            print(table)
        else:
            gleu = n_accum_to_gleu(n_dacc, rlen, hlen)
            line = simple_result(gleu, args.digit)
            print(args.hyp_path_list[h] + '\t' + line)


def corpus_sampling(args):
    _, _, _, dr_rlen, dh_hlen, hdrn_sacc = load_hdrn_data(args)
    id_rindex = make_id_rindex(args.iter, len(dr_rlen), len(args.ref_path_list), args.fix_seed)

    for h, drn_sacc in enumerate(hdrn_sacc):
        d_hlen = dh_hlen[:, h]

        if args.proc > 1:
            gleu_args = [
                (drn_sacc, d_rindex, dr_rlen, d_hlen)
                for d_rindex
                in id_rindex]
            with Pool(args.proc) as pool:
                gleus = pool.starmap(drn_sacc_to_gleu, gleu_args)
        else:
            gleus = [
                drn_sacc_to_gleu(drn_sacc, d_rindex, dr_rlen, d_hlen)
                for d_rindex
                in id_rindex]

        gleu = np.mean(gleus)
        line = simple_result(gleu, args.digit)
        print(args.hyp_path_list[h] + '\t' + line)


def drn_sacc_to_gleu(drn_sacc, d_rindex, dr_rlen, d_hlen):
    n_dacc = drn_sent_accum_to_n_data_accum(drn_sacc, d_rindex)
    rlen, hlen = rindex_to_rhlen(dr_rlen, d_hlen, d_rindex)
    gleu = n_accum_to_gleu(n_dacc, rlen, hlen)
    return gleu

