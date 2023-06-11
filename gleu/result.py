import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from prettytable import PrettyTable


def round_half_up(x, digit):
    if x == np.inf or x == -np.inf:
        return str(x)

    digit = Decimal('0.' + '0' * digit)
    x = Decimal(str(x)).quantize(digit, rounding = ROUND_HALF_UP)
    return str(x)


def simple_result(f, digit):
    f = 100 * f
    f = round_half_up(f, digit)
    return f


def table_result(nvb, digit):
    table = PrettyTable(nvb.header)
    for lst in nvb.iter_row(digit):
        table.add_row(lst)
    table.add_row(nvb.total_row(digit))
    return table

